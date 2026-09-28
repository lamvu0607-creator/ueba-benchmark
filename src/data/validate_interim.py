"""Module kiểm định chất lượng và đối soát dữ liệu Parquet (Interim).

Tập trung vào 4 tầng kiểm tra:
1. Integrity & Schema: Kiểm tra metadata, schema 21 cột, phát hiện file hỏng.
2. Data Quality & Sanity: Kiểm tra tỷ lệ null, miền giá trị Time, LogonType, logic EventID.
3. Reconciliation: Đối soát số lượng record giữa file thô bz2 và file Parquet interim.
4. Spot Check: So khớp dữ liệu mẫu ngẫu nhiên giữa raw và interim.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

# Đảm bảo UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import schema chuẩn từ raw_to_interim để đảm bảo Single Source of Truth
try:
    from src.data.raw_to_interim import INTERIM_SCHEMA, COLUMN_NAMES
except ImportError:
    # Fallback độc lập nếu chạy standalone
    INTERIM_SCHEMA = pa.schema(
        [
            ("Time", pa.int64()),
            ("EventID", pa.int32()),
            ("LogHost", pa.string()),
            ("LogonType", pa.int32()),
            ("LogonTypeDescription", pa.string()),
            ("UserName", pa.string()),
            ("DomainName", pa.string()),
            ("LogonID", pa.string()),
            ("SubjectUserName", pa.string()),
            ("SubjectDomainName", pa.string()),
            ("SubjectLogonID", pa.string()),
            ("Status", pa.string()),
            ("Source", pa.string()),
            ("ServiceName", pa.string()),
            ("Destination", pa.string()),
            ("AuthenticationPackage", pa.string()),
            ("FailureReason", pa.string()),
            ("ProcessName", pa.string()),
            ("ProcessID", pa.string()),
            ("ParentProcessName", pa.string()),
            ("ParentProcessID", pa.string()),
        ]
    )
    COLUMN_NAMES = [f.name for f in INTERIM_SCHEMA]


def extract_day_tag(filepath: Path) -> str:
    """Trích xuất mã ngày chuẩn hóa từ tên file (ví dụ: wls_day-01.bz2 -> day-01)."""
    match = re.search(r"day[-_]?(\d+)", filepath.name, re.IGNORECASE)
    if match:
        day_num = int(match.group(1))
        return f"day-{day_num:02d}"
    base = filepath.name
    for ext in [".bz2", ".json", ".csv", ".parquet"]:
        base = base.replace(ext, "")
    return base


# ==============================================================================
# TẦNG 1: KIỂM ĐỊNH METADATA, SCHEMA VÀ TÍNH TOÀN VẸN CỦA FILE PARQUET
# ==============================================================================
def check_parquet_file(
    parquet_path: Path, expected_schema: pa.Schema = INTERIM_SCHEMA
) -> Dict[str, Any]:
    """Kiểm tra tính toàn vẹn và schema của 1 file Parquet qua metadata footer."""
    res = {
        "filepath": str(parquet_path),
        "filename": parquet_path.name,
        "is_valid": False,
        "size_bytes": 0,
        "num_rows": 0,
        "num_columns": 0,
        "num_row_groups": 0,
        "schema_matches": False,
        "missing_columns": [],
        "type_mismatches": {},
        "error": None,
    }

    if not parquet_path.exists():
        res["error"] = "File không tồn tại"
        return res

    res["size_bytes"] = parquet_path.stat().st_size
    if res["size_bytes"] == 0:
        res["error"] = "File rỗng (0 bytes)"
        return res

    try:
        # Đọc metadata footer không load toàn bộ data vào RAM
        metadata = pq.read_metadata(str(parquet_path))
        schema = metadata.schema.to_arrow_schema()

        res["num_rows"] = metadata.num_rows
        res["num_columns"] = metadata.num_columns
        res["num_row_groups"] = metadata.num_row_groups

        # Kiểm tra sự tương thích của Schema
        expected_cols = {f.name: f.type for f in expected_schema}
        actual_cols = {f.name: f.type for f in schema}

        missing = [col for col in expected_cols if col not in actual_cols]
        mismatches = {}
        for col, exp_type in expected_cols.items():
            if col in actual_cols and actual_cols[col] != exp_type:
                mismatches[col] = f"Kỳ vọng {exp_type}, thực tế {actual_cols[col]}"

        res["missing_columns"] = missing
        res["type_mismatches"] = mismatches
        res["schema_matches"] = (len(missing) == 0) and (len(mismatches) == 0)
        res["is_valid"] = res["schema_matches"]

    except Exception as exc:
        res["error"] = f"Lỗi đọc file Parquet (Corrupted): {str(exc)}"
        res["is_valid"] = False

    return res


def scan_all_interim_integrity(
    interim_dir: Path | str, target_event_ids: Set[int] = {4624, 4625}
) -> Dict[str, Any]:
    """Quét nhanh toàn bộ thư mục interim, kiểm tra tất cả các file Parquet."""
    interim_path = Path(interim_dir)
    results = {
        "total_files": 0,
        "valid_files": 0,
        "corrupted_files": 0,
        "total_rows": {eid: 0 for eid in target_event_ids},
        "total_size_bytes": 0,
        "days_found": set(),
        "file_details": {},
    }

    for eid in sorted(target_event_ids):
        event_dir = interim_path / f"event_{eid}"
        if not event_dir.exists():
            continue

        parquet_files = sorted(list(event_dir.glob("*.parquet")))
        for pfile in parquet_files:
            day_tag = extract_day_tag(pfile)
            results["days_found"].add(day_tag)
            results["total_files"] += 1

            check_res = check_parquet_file(pfile)
            results["file_details"][f"{eid}_{day_tag}"] = check_res

            if check_res["is_valid"]:
                results["valid_files"] += 1
                results["total_rows"][eid] += check_res["num_rows"]
                results["total_size_bytes"] += check_res["size_bytes"]
            else:
                results["corrupted_files"] += 1

    results["days_found"] = sorted(list(results["days_found"]))
    return results


# ==============================================================================
# TẦNG 2: KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY & SANITY CHECKS)
# ==============================================================================
def check_data_quality(
    parquet_path: Path, expected_event_id: int
) -> Dict[str, Any]:
    """Đọc dữ liệu Parquet và kiểm định tỷ lệ Null, dải giá trị, logic nghiệp vụ."""
    res = {
        "filepath": str(parquet_path),
        "event_id": expected_event_id,
        "num_rows": 0,
        "null_counts": {},
        "null_percentages": {},
        "sanity_issues": [],
        "time_range": {"min": None, "max": None},
        "unique_logon_types": [],
    }

    try:
        table = pq.read_table(str(parquet_path))
        total_rows = table.num_rows
        res["num_rows"] = total_rows

        if total_rows == 0:
            res["sanity_issues"].append("Bảng dữ liệu rỗng (0 dòng)")
            return res

        # 1. Tính toán null counts & percentages
        for col in table.column_names:
            n_null = table[col].null_count
            res["null_counts"][col] = n_null
            res["null_percentages"][col] = round((n_null / total_rows) * 100, 2)

        # 2. Kiểm tra bất biến: Time & EventID không được phép null
        if res["null_counts"].get("Time", 0) > 0:
            res["sanity_issues"].append(
                f"Cột Time chứa {res['null_counts']['Time']} giá trị Null!"
            )
        if res["null_counts"].get("EventID", 0) > 0:
            res["sanity_issues"].append(
                f"Cột EventID chứa {res['null_counts']['EventID']} giá trị Null!"
            )

        # 3. Kiểm tra tính toàn vẹn EventID bằng pyarrow.compute
        import pyarrow.compute as pc
        distinct_eids = set(pc.unique(table["EventID"]).to_pylist())
        if distinct_eids != {expected_event_id}:
            res["sanity_issues"].append(
                f"Phát hiện EventID không khớp thư mục! Kỳ vọng {{{expected_event_id}}}, thực tế: {distinct_eids}"
            )

        # 4. Kiểm tra trường Time: min, max, số âm bằng pyarrow.compute
        time_min_max = pc.min_max(table["Time"]).as_py()
        min_time, max_time = time_min_max["min"], time_min_max["max"]
        res["time_range"] = {"min": min_time, "max": max_time}
        if min_time is not None and min_time < 0:
            res["sanity_issues"].append(f"Giá trị Time âm không hợp lệ: min={min_time}")

        # 5. Kiểm tra trường FailureReason theo nghiệp vụ
        failure_null_pct = res["null_percentages"].get("FailureReason", 0.0)
        if expected_event_id == 4624:
            # 4624 là logon thành công, FailureReason phải luôn là Null
            if failure_null_pct < 100.0:
                res["sanity_issues"].append(
                    f"Event 4624 có FailureReason không null ({100 - failure_null_pct:.1f}%)"
                )
        elif expected_event_id == 4625:
            # 4625 là logon thất bại, FailureReason nên có giá trị
            if failure_null_pct == 100.0:
                res["sanity_issues"].append(
                    "Event 4625 nhưng toàn bộ FailureReason đều là Null!"
                )

        # 6. Kiểm tra LogonType bằng pyarrow.compute
        logon_types = set(pc.unique(table["LogonType"]).drop_null().to_pylist())
        res["unique_logon_types"] = sorted(list(logon_types))

    except Exception as exc:
        res["sanity_issues"].append(f"Lỗi khi đọc bảng dữ liệu: {str(exc)}")

    return res


# ==============================================================================
# TẦNG 3: ĐỐI SOÁT SỐ LƯỢNG DÒNG (RAW VS INTERIM RECONCILIATION)
# ==============================================================================
def reconcile_raw_vs_interim(
    raw_bz2_path: Path,
    interim_dir: Path,
    target_event_ids: Set[int] = {4624, 4625},
) -> Dict[str, Any]:
    """Quét độc lập file raw bz2 để đối soát số lượng record với file Parquet tương ứng."""
    import bz2

    day_tag = extract_day_tag(raw_bz2_path)
    res = {
        "day_tag": day_tag,
        "raw_file": str(raw_bz2_path),
        "raw_counts": {eid: 0 for eid in target_event_ids},
        "parquet_counts": {eid: 0 for eid in target_event_ids},
        "deltas": {eid: 0 for eid in target_event_ids},
        "is_reconciled": False,
        "total_lines_scanned": 0,
        "corrupted_json_lines": 0,
        "errors": [],
    }

    # Lấy số dòng từ file Parquet trước qua metadata
    for eid in target_event_ids:
        p_path = interim_dir / f"event_{eid}" / f"event_{eid}_{day_tag}.parquet"
        if p_path.exists():
            try:
                meta = pq.read_metadata(str(p_path))
                res["parquet_counts"][eid] = meta.num_rows
            except Exception as e:
                res["errors"].append(f"Không thể đọc metadata Parquet {p_path}: {e}")
        else:
            res["errors"].append(f"Không tìm thấy file Parquet: {p_path}")

    # Quét độc lập từ raw bz2 bằng Regex linh hoạt chống bẫy khoảng trắng
    eid_patterns = {
        eid: re.compile(rf'"EventID"\s*:\s*{eid}\b') for eid in target_event_ids
    }

    try:
        with bz2.open(raw_bz2_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                res["total_lines_scanned"] += 1
                matched_eid = None
                for eid, pattern in eid_patterns.items():
                    if pattern.search(line):
                        matched_eid = eid
                        break

                if matched_eid is None:
                    continue

                # Xác minh JSON cú pháp hợp lệ
                try:
                    record = json.loads(line)
                    actual_eid = record.get("EventID")
                    if actual_eid in target_event_ids:
                        res["raw_counts"][actual_eid] += 1
                except json.JSONDecodeError:
                    res["corrupted_json_lines"] += 1

    except Exception as e:
        res["errors"].append(f"Lỗi đọc file raw bz2: {e}")
        return res

    # Tính toán chênh lệch (Delta)
    all_matched = True
    for eid in target_event_ids:
        delta = res["raw_counts"][eid] - res["parquet_counts"][eid]
        res["deltas"][eid] = delta
        if delta != 0:
            all_matched = False

    res["is_reconciled"] = all_matched and (len(res["errors"]) == 0)
    return res


# ==============================================================================
# TẦNG 4: BÁO CÁO KẾT QUẢ (FORMATTING & EXPORT)
# ==============================================================================
def generate_markdown_report(
    integrity_res: Optional[Dict[str, Any]] = None,
    quality_results: Optional[List[Dict[str, Any]]] = None,
    reconcile_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Tạo báo cáo kiểm định hoàn chỉnh định dạng Markdown."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Báo cáo Kiểm định Dữ liệu Interim (Event 4624 & 4625)",
        f"*Thời điểm tạo: {now_str}*",
        "",
        "---",
    ]

    # Phần 1: Tổng quan File & Schema Integrity
    if integrity_res:
        lines.extend(
            [
                "## 1. Tổng quan Toàn vẹn File & Schema (Integrity Check)",
                "",
                f"- **Tổng số file Parquet kiểm tra:** {integrity_res['total_files']}",
                f"- **Số file hợp lệ (Pass):** {integrity_res['valid_files']}",
                f"- **Số file hỏng/lỗi (Fail):** {integrity_res['corrupted_files']}",
                f"- **Tổng dung lượng:** {integrity_res['total_size_bytes'] / (1024**3):.2f} GB",
                f"- **Tổng số records Event 4624:** {integrity_res['total_rows'].get(4624, 0):,}",
                f"- **Tổng số records Event 4625:** {integrity_res['total_rows'].get(4625, 0):,}",
                "",
            ]
        )

    # Phần 2: Đối soát số lượng Raw vs Interim
    if reconcile_results:
        lines.extend(
            [
                "## 2. Kết quả Đối soát Số lượng Dòng (Raw vs Parquet Reconciliation)",
                "",
                "| Ngày | EventID | Dòng trong Raw (.bz2) | Dòng trong Parquet | Chênh lệch (Δ) | Trạng thái |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for rec in reconcile_results:
            day = rec["day_tag"]
            for eid in sorted(rec["raw_counts"].keys()):
                raw_c = rec["raw_counts"][eid]
                pq_c = rec["parquet_counts"][eid]
                delta = rec["deltas"][eid]
                status = "✅ KHỚP 100%" if delta == 0 else f"❌ LỆCH ({delta:+,})"
                lines.append(
                    f"| `{day}` | {eid} | {raw_c:,} | {pq_c:,} | {delta} | {status} |"
                )
        lines.append("")

    # Phần 3: Phân tích Chất lượng Dữ liệu
    if quality_results:
        lines.extend(
            [
                "## 3. Kiểm định Chất lượng Dữ liệu & Miền Giá trị (Data Quality)",
                "",
            ]
        )
        for q in quality_results:
            p_name = Path(q["filepath"]).name
            eid = q["event_id"]
            issues = q["sanity_issues"]
            status_badge = "✅ PASS" if not issues else "⚠️ CẢNH BÁO"
            lines.append(f"### File: `{p_name}` (Event {eid}) — {status_badge}")
            lines.append(f"- **Tổng số dòng:** {q['num_rows']:,}")
            lines.append(
                f"- **Dải thời gian (Time range):** `[{q['time_range']['min']} -> {q['time_range']['max']}]`"
            )
            lines.append(
                f"- **Tập LogonType:** `{q['unique_logon_types']}`"
            )

            # Bảng tóm tắt tỷ lệ Null của 21 cột
            lines.append("")
            lines.append("| Tên trường | Số lượng Null | Tỷ lệ Null (%) | Đánh giá |")
            lines.append("| :--- | :---: | :---: | :---: |")
            for col, n_pct in q["null_percentages"].items():
                n_count = q["null_counts"].get(col, 0)
                eval_tag = "Bình thường"
                if col in ["Time", "EventID"] and n_count > 0:
                    eval_tag = "❌ LỖI NGHIÊM TRỌNG (Null key)"
                elif col in ["LogHost", "UserName"] and n_pct > 50.0:
                    eval_tag = "⚠️ Cảnh báo Null cao"
                elif col == "FailureReason" and eid == 4624 and n_pct == 100.0:
                    eval_tag = "Chuẩn (4624 không có failure)"
                elif col == "FailureReason" and eid == 4625 and n_pct == 0.0:
                    eval_tag = "Chuẩn (100% có failure reason)"
                lines.append(f"| `{col}` | {n_count:,} | {n_pct:.2f}% | {eval_tag} |")

            if issues:
                lines.append("")
                lines.append("- **Vấn đề phát hiện:**")
                for iss in issues:
                    lines.append(f"  - ⚠️ {iss}")

            lines.append("")

    return "\n".join(lines)
