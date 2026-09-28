"""Module chuyển đổi dữ liệu thô bz2 sang Apache Parquet.

Đặc điểm kỹ thuật:
- Lọc EventID 4624 và 4625 từ file nén wls_day-XX.bz2 (JSON Lines).
- Phân tách thành 2 thư mục riêng biệt: interim/event_4624 và interim/event_4625.
- Áp đặt Schema chuẩn 21 trường theo data_dictionary.md.
- Tối ưu Streaming & Batch ParquetWriter: peak RAM < 500MB, chống tràn bộ nhớ.
- Lọc chuỗi thô ở cấp độ C trước khi parse JSON để đạt tốc độ xử lý tối đa.
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Set

# Đảm bảo UTF-8 an toàn trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


import pyarrow as pa
import pyarrow.parquet as pq

# ==============================================================================
# 1. SCHEMA 21 TRƯỜNG CHUẨN THEO DATA DICTIONARY
# ==============================================================================
INTERIM_SCHEMA = pa.schema(
    [
        # Temporal
        ("Time", pa.int64()),
        # Event
        ("EventID", pa.int32()),
        # Host / Network
        ("LogHost", pa.string()),
        # Authentication
        ("LogonType", pa.int32()),
        ("LogonTypeDescription", pa.string()),
        # Identity
        ("UserName", pa.string()),
        ("DomainName", pa.string()),
        ("LogonID", pa.string()),
        ("SubjectUserName", pa.string()),
        ("SubjectDomainName", pa.string()),
        ("SubjectLogonID", pa.string()),
        # Authentication (tiếp)
        ("Status", pa.string()),
        # Host / Network (tiếp)
        ("Source", pa.string()),
        # Authentication (tiếp)
        ("ServiceName", pa.string()),
        # Host / Network (tiếp)
        ("Destination", pa.string()),
        # Authentication (tiếp)
        ("AuthenticationPackage", pa.string()),
        ("FailureReason", pa.string()),
        # Process
        ("ProcessName", pa.string()),
        ("ProcessID", pa.string()),
        ("ParentProcessName", pa.string()),
        ("ParentProcessID", pa.string()),
    ]
)

COLUMN_NAMES = [field.name for field in INTERIM_SCHEMA]


def extract_day_tag(filepath: Path) -> str:
    """Trích xuất mã ngày chuẩn hóa từ tên file (ví dụ: wls_day-01.bz2 -> day-01)."""
    match = re.search(r"day[-_]?(\d+)", filepath.name, re.IGNORECASE)
    if match:
        day_num = int(match.group(1))
        return f"day-{day_num:02d}"
    # Fallback nếu tên không theo định dạng chuẩn
    base = filepath.name
    for ext in [".bz2", ".json", ".csv", ".parquet"]:
        base = base.replace(ext, "")
    return base


def init_buffer() -> Dict[str, List]:
    """Khởi tạo buffer rỗng chứa các cột theo schema."""
    return {col: [] for col in COLUMN_NAMES}


def append_to_buffer(buffer: Dict[str, List], row_dict: Dict):
    """Bổ sung một record vào buffer theo đúng 21 trường và ép kiểu an toàn."""
    for col in COLUMN_NAMES:
        val = row_dict.get(col)
        if val is None or val == "":
            buffer[col].append(None)
        elif col == "Time" or col == "EventID":
            try:
                buffer[col].append(int(val))
            except (ValueError, TypeError):
                buffer[col].append(None)
        elif col == "LogonType":
            try:
                buffer[col].append(int(val))
            except (ValueError, TypeError):
                buffer[col].append(None)
        else:
            buffer[col].append(str(val))


def flush_buffer(
    buffer: Dict[str, List],
    writer: Optional[pq.ParquetWriter],
    output_path: Path,
) -> pq.ParquetWriter:
    """Ghi buffer hiện tại xuống file Parquet và reset buffer."""
    if not buffer["Time"]:
        return writer

    table = pa.Table.from_pydict(buffer, schema=INTERIM_SCHEMA)
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(
            str(output_path),
            schema=INTERIM_SCHEMA,
            compression="snappy",
            use_dictionary=True,
        )
    writer.write_table(table)
    for col in COLUMN_NAMES:
        buffer[col].clear()
    return writer


def convert_single_bz2(
    bz2_path: Path,
    interim_dir: Path,
    target_event_ids: Set[int] = {4624, 4625},
    batch_size: int = 50000,
    overwrite: bool = False,
    progress_step_pct: int = 10,
) -> Dict[int, int]:
    """Chuyển đổi một file bz2 thô sang Parquet cho các EventID chỉ định.

    Chỉ in log theo định dạng yêu cầu: Ngày và Phần trăm hoàn thành.
    """
    import bz2

    day_tag = extract_day_tag(bz2_path)
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Đường dẫn xuất chính thức và đường dẫn tạm thời (Atomic Write chống file hỏng khi dừng đột ngột)
    output_paths = {
        event_id: interim_dir
        / f"event_{event_id}"
        / f"event_{event_id}_{day_tag}.parquet"
        for event_id in target_event_ids
    }
    tmp_paths = {
        event_id: interim_dir
        / f"event_{event_id}"
        / f"event_{event_id}_{day_tag}.parquet.tmp"
        for event_id in target_event_ids
    }

    # Kiểm tra khả phục (Skip if already completed)
    if not overwrite and all(
        p.exists() and p.stat().st_size > 0 for p in output_paths.values()
    ):
        print(f"[{current_date}] [{day_tag}] 100% (Đã tồn tại - Bỏ qua)", flush=True)
        return {eid: 0 for eid in target_event_ids}

    # Đảm bảo thư mục đích tồn tại và dọn dẹp file .tmp cũ nếu có
    for p in output_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    for tmp_p in tmp_paths.values():
        if tmp_p.exists():
            try:
                tmp_p.unlink()
            except Exception:
                pass

    # Chuẩn bị writers và buffers
    buffers = {eid: init_buffer() for eid in target_event_ids}
    writers: Dict[int, Optional[pq.ParquetWriter]] = {
        eid: None for eid in target_event_ids
    }
    counts = {eid: 0 for eid in target_event_ids}

    total_compressed_bytes = os.path.getsize(bz2_path)
    last_reported_pct = -1

    # Khởi đầu ngày
    print(f"[{current_date}] [{day_tag}] 0% Bắt đầu xử lý", flush=True)

    line_count = 0
    try:
        with open(bz2_path, "rb") as raw_f:
            with bz2.open(raw_f, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_count += 1

                    # Cập nhật tiến độ theo chu kỳ số dòng (không bị phụ thuộc vào phân bố event)
                    if line_count % 25000 == 0 and total_compressed_bytes > 0:
                        pos = raw_f.tell()
                        pct = int((pos / total_compressed_bytes) * 100)
                        if pct >= last_reported_pct + progress_step_pct and pct < 100:
                            last_reported_pct = (pct // progress_step_pct) * progress_step_pct
                            print(f"[{current_date}] [{day_tag}] {last_reported_pct}%", flush=True)

                    # Tối ưu: Lọc chuỗi thô C-level trước khi gọi json.loads
                    if '"EventID": 4624' in line:
                        target_id = 4624
                    elif '"EventID": 4625' in line:
                        target_id = 4625
                    else:
                        continue

                    if target_id not in target_event_ids:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    actual_eid = record.get("EventID")
                    if actual_eid != target_id:
                        continue

                    append_to_buffer(buffers[target_id], record)
                    counts[target_id] += 1

                    # Xả đệm khi đạt ngưỡng batch để giải phóng RAM (ghi vào file .tmp)
                    if len(buffers[target_id]["Time"]) >= batch_size:
                        writers[target_id] = flush_buffer(
                            buffers[target_id],
                            writers[target_id],
                            tmp_paths[target_id],
                        )

        # Xả lượng buffer còn lại
        for eid in target_event_ids:
            writers[eid] = flush_buffer(
                buffers[eid], writers[eid], tmp_paths[eid]
            )

        # Đảm bảo file parquet được đóng và nguyên tử hóa (Atomic rename)
        for eid in target_event_ids:
            if writers[eid] is None:
                empty_table = pa.Table.from_pydict(
                    init_buffer(), schema=INTERIM_SCHEMA
                )
                pq.write_table(empty_table, str(tmp_paths[eid]))
            else:
                writers[eid].close()

            # Đổi tên .tmp -> .parquet chính thức sau khi ghi hoàn tất 100%
            if tmp_paths[eid].exists():
                if output_paths[eid].exists():
                    output_paths[eid].unlink()
                tmp_paths[eid].replace(output_paths[eid])

        print(
            f"[{current_date}] [{day_tag}] 100% Hoàn thành "
            f"(4624: {counts[4624]:,} records, 4625: {counts[4625]:,} records)",
            flush=True,
        )
        return counts

    except (Exception, KeyboardInterrupt) as e:
        # Dọn dẹp writers và xóa file tạm .tmp để không để lại dữ liệu rác
        for w in writers.values():
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
        for tmp_p in tmp_paths.values():
            if tmp_p.exists():
                try:
                    tmp_p.unlink()
                except Exception:
                    pass
        raise e


def convert_all_raw_to_interim(
    raw_dir: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    target_event_ids: Set[int] = {4624, 4625},
    batch_size: int = 50000,
    overwrite: bool = False,
    progress_step_pct: int = 10,
    max_files: Optional[int] = None,
    workers: int = 4,
) -> Dict[str, Dict[int, int]]:
    """Duyệt toàn bộ các file .bz2 trong raw_dir và chuyển đổi sang Parquet theo chuẩn.

    Hỗ trợ xử lý đa tiến trình (multiprocessing) qua tham số workers.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    raw_path = Path(raw_dir)
    interim_path = Path(interim_dir)

    raw_files = sorted(list(raw_path.glob("*.bz2")))
    if not raw_files:
        print(f"Cảnh báo: Không tìm thấy file .bz2 nào trong '{raw_path}'.", flush=True)
        return {}

    if max_files is not None:
        raw_files = raw_files[:max_files]

    total_files = len(raw_files)
    summary = {}

    current_date = datetime.now().strftime("%Y-%m-%d")
    print(
        f"[{current_date}] Bắt đầu xử lý {total_files} file log từ '{raw_path}' -> '{interim_path}' (Số worker: {workers})",
        flush=True,
    )

    if workers <= 1:
        # Xử lý tuần tự đơn tiến trình
        for idx, bz2_file in enumerate(raw_files, 1):
            day_tag = extract_day_tag(bz2_file)
            counts = convert_single_bz2(
                bz2_path=bz2_file,
                interim_dir=interim_path,
                target_event_ids=target_event_ids,
                batch_size=batch_size,
                overwrite=overwrite,
                progress_step_pct=progress_step_pct,
            )
            summary[day_tag] = counts
    else:
        # Xử lý song song đa tiến trình
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    convert_single_bz2,
                    bz2_path=bz2_file,
                    interim_dir=interim_path,
                    target_event_ids=target_event_ids,
                    batch_size=batch_size,
                    overwrite=overwrite,
                    progress_step_pct=progress_step_pct,
                ): bz2_file
                for bz2_file in raw_files
            }
            for future in as_completed(futures):
                bz2_file = futures[future]
                day_tag = extract_day_tag(bz2_file)
                try:
                    counts = future.result()
                    summary[day_tag] = counts
                except Exception as exc:
                    print(f"[{current_date}] [{day_tag}] LỖI TIẾN TRÌNH: {exc}", flush=True)

    print(f"[{current_date}] Tất cả {total_files} file đã được xử lý xong.", flush=True)
    return summary
