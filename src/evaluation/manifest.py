"""
Module Manifest - Ghi ``run_manifest.json`` cho mỗi lần chạy benchmark (truy vết & tái lập).

Manifest trả lời đúng 4 câu hỏi khi có người hỏi "con số này ở đâu ra":
  1. Chạy trên **dữ liệu nào**? -> đường dẫn + số dòng + ``sha256`` của parquet.
  2. Chạy **mã nào**? -> commit hash + trạng thái dirty của working tree.
  3. Chạy với **cấu hình nào**? -> seed, split, K, ngân sách, danh sách đặc trưng, tham số mô hình.
  4. Chạy bằng **thư viện nào**? -> phiên bản Python/sklearn/scipy/polars/numpy/pandas/joblib.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

logger = logging.getLogger("ueba_benchmark.evaluation.manifest")

__all__ = [
    "build_manifest",
    "file_sha256",
    "git_state",
    "library_versions",
    "write_manifest",
]

MANIFEST_FORMAT = "ueba-run-manifest/1"
_TRACKED_LIBRARIES = ("scikit-learn", "scipy", "polars", "numpy", "pandas", "joblib", "pyarrow")


def file_sha256(path: Path | str, chunk_size: int = 1 << 20) -> Optional[str]:
    """``sha256`` của file (None nếu file không tồn tại) — dùng để chứng minh dữ liệu không đổi."""
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with open(source, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Commit hash + trạng thái dirty của repo (None nếu không phải repo git/không có git)."""
    root = str(repo_root or Path(__file__).resolve().parents[2])

    def _run(args: Sequence[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = _run(["rev-parse", "HEAD"])
    status = _run(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": None if status is None else bool(status),
    }


def library_versions() -> Dict[str, str]:
    """Phiên bản Python và các thư viện then chốt (thiếu gói -> ghi 'not-installed')."""
    versions = {"python": platform.python_version()}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_manifest(
    *,
    data_path: Path | str,
    params_path: Optional[Path | str] = None,
    config_path: Optional[Path | str] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    split: Optional[Mapping[str, Any]] = None,
    models: Optional[Sequence[Mapping[str, Any]]] = None,
    seed: int = 42,
    k: int = 20,
    budget_ratio: float = 0.05,
    feature_names: Optional[Sequence[str]] = None,
    feature_set: str = "core",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Dựng manifest đầy đủ cho một lần chạy benchmark."""
    data_file = Path(data_path)
    manifest: Dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "libraries": library_versions(),
        "platform": platform.platform(),
        "data": {
            "path": str(data_file),
            "sha256": file_sha256(data_file),
            "bytes": data_file.stat().st_size if data_file.is_file() else None,
        },
        "config": {
            "params_path": str(params_path) if params_path else None,
            "system_config_path": str(config_path) if config_path else None,
            "seed": int(seed),
            "k": int(k),
            "budget_ratio": float(budget_ratio),
            "feature_set": feature_set,
            "feature_names": list(feature_names or []),
        },
        "split": dict(split or {}),
        "models": [dict(model) for model in (models or [])],
        "artifacts": {key: (str(value) if value is not None else None) for key, value in (artifacts or {}).items()},
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def write_manifest(manifest: Mapping[str, Any], path: Path | str) -> Path:
    """Ghi manifest ra JSON (UTF-8, giữ tiếng Việt không escape)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Đã ghi manifest: '%s'.", target)
    return target
