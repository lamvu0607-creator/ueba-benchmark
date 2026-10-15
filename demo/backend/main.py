"""Run locally with: uvicorn demo.backend.main:app"""

from __future__ import annotations

import logging
import asyncio
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header

from demo.backend.service import DemoInputError, DemoService

logger = logging.getLogger(__name__)
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
MAX_UPLOAD = 512 * 1024 * 1024


def error_response(error: str, message: str, code: int) -> JSONResponse:
    return JSONResponse({"status": "error", "error": error, "message": message}, status_code=code)


async def receive_file(request: Request, path: Path) -> str:
    """Stream multipart directly to disk so the 512 MB limit applies while receiving."""
    media_type, options = parse_options_header(request.headers.get("content-type", "").encode())
    boundary = options.get(b"boundary")
    if media_type != b"multipart/form-data" or not boundary:
        raise DemoInputError("invalid_file", "Cần multipart/form-data với field file.")
    filename, field, header_name, header_value = "", "", b"", b""
    seen_file = False
    completed = False
    count = 0
    with path.open("wb") as output:
        def begin():
            nonlocal field, header_name, header_value
            field, header_name, header_value = "", b"", b""

        def header_field(data, start, end):
            nonlocal header_name
            header_name += data[start:end]

        def header_value_part(data, start, end):
            nonlocal header_value
            header_value += data[start:end]

        def header_end():
            nonlocal field, filename, seen_file, header_name, header_value
            if header_name.lower() == b"content-disposition":
                _, params = parse_options_header(header_value)
                field = params.get(b"name", b"").decode("utf-8", "replace")
                if field == "file":
                    if seen_file:
                        raise DemoInputError("invalid_file", "Chỉ gửi một file mỗi request.")
                    seen_file = True
                    filename = params.get(b"filename", b"").decode("utf-8", "replace")
            header_name, header_value = b"", b""

        def part_data(data, start, end):
            nonlocal count
            if field == "file":
                count += end - start
                if count > MAX_UPLOAD:
                    raise DemoInputError("file_too_large", "File vượt giới hạn 512 MB.", 413)
                output.write(data[start:end])

        def done():
            nonlocal completed
            completed = True

        parser = MultipartParser(boundary, callbacks={
            "on_part_begin": begin, "on_header_field": header_field,
            "on_header_value": header_value_part, "on_header_end": header_end,
            "on_part_data": part_data, "on_end": done,
        })
        try:
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_UPLOAD + 65536:
                    raise DemoInputError("file_too_large", "Request vượt giới hạn 512 MB.", 413)
                parser.write(chunk)
            parser.finalize()
        except DemoInputError:
            raise
        except Exception as exc:
            raise DemoInputError("invalid_file", "Multipart không hợp lệ.") from exc
    if not completed or not filename:
        raise DemoInputError("invalid_file", "Thiếu multipart field file.")
    return filename


def create_app(artifact_dir: Path | None = None, cleaned_dir: Path | None = None) -> FastAPI:
    service = DemoService(**({"artifact_dir": artifact_dir} if artifact_dir else {}),
                          **({"cleaned_dir": cleaned_dir} if cleaned_dir else {}))
    busy = threading.Lock()
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ueba-demo")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            service.load()
        except Exception:
            logger.exception("Không thể nạp artifact demo")
        try:
            yield
        finally:
            worker.shutdown(wait=True)

    app = FastAPI(title="UEBA Log Analyzer", lifespan=lifespan)
    app.state.demo_service = service
    page = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assets = {name: (FRONTEND / name).read_bytes() for name in ("style.css", "app.js")}

    @app.get("/")
    async def index():
        return HTMLResponse(page)

    @app.get("/static/{asset}")
    async def static(asset: str):
        if asset not in assets:
            raise HTTPException(status_code=404)
        return Response(assets[asset], media_type="text/css" if asset.endswith(".css") else "text/javascript")

    @app.get("/api/health")
    async def health():
        state = service.health()
        return JSONResponse(state, status_code=200 if state["status"] == "ready" else 503)

    @app.post("/api/analyze")
    async def analyze(request: Request):
        if not busy.acquire(blocking=False):
            return error_response("server_busy", "Server đang xử lý một batch khác.", 409)
        try:
            if service.health()["status"] != "ready":
                return error_response("artifact_unavailable", "Artifact demo chưa sẵn sàng.", 503)
            with tempfile.TemporaryDirectory(prefix="ueba-upload-", dir=service.cleaned_dir.parent) as folder:
                path = Path(folder) / "upload.parquet"
                filename = await receive_file(request, path)
                future = worker.submit(service.analyze, path, filename)
                # Polling keeps health/409 responsive; this runtime cannot wake asyncio from Polars workers.
                try:
                    while not future.done():
                        await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    while not future.done():
                        await asyncio.sleep(0.05)
                    raise
                return future.result()
        except DemoInputError as exc:
            return error_response(exc.error, exc.message, exc.status_code)
        except Exception:
            logger.exception("Phân tích batch thất bại")
            return error_response("internal_error", "Không thể phân tích batch này.", 500)
        finally:
            busy.release()

    return app


app = create_app()
