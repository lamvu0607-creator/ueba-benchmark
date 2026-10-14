"""
Feature Schema Contract Loader & Validator.
Đọc, kiểm tra và bảo đảm tính hợp lệ của hợp đồng đặc trưng (configs/feature_schema.yaml).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import polars as pl


class FeatureSchema:
    def __init__(self, schema_path: Optional[Path | str] = None):
        self.schema_path = Path(schema_path) if schema_path else self._default_schema_path()
        self.raw_schema = self._load()

    @staticmethod
    def _default_schema_path() -> Path:
        # Tự tìm configs/feature_schema.yaml từ root dự án
        curr = Path(__file__).resolve().parent
        for candidate in [
            curr.parent.parent / "configs" / "feature_schema.yaml",
            Path("configs/feature_schema.yaml"),
            Path("feature_schema.yaml"),
        ]:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError("Không tìm thấy file 'configs/feature_schema.yaml'.")

    def _load(self) -> Dict[str, Any]:
        with open(self.schema_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data

    @property
    def identity_keys(self) -> List[str]:
        return self.raw_schema.get("identity", {}).get("keys", ["DomainName", "UserName", "day"])

    @property
    def label_key(self) -> str:
        return self.raw_schema.get("identity", {}).get("label", "entity_type")

    @property
    def label_values(self) -> List[str]:
        return self.raw_schema.get("identity", {}).get("label_values", ["Machine", "User", "Service", "System", "Admin", "Other"])

    @property
    def core_features(self) -> List[str]:
        return [
            f["name"]
            for f in self.raw_schema.get("features", [])
            if f.get("core", False)
        ]

    @property
    def feature_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {f["name"]: f for f in self.raw_schema.get("features", [])}

    @property
    def feature_dtypes(self) -> Dict[str, str]:
        """Tên đặc trưng -> kiểu dữ liệu khai báo trong schema (float64 / int32 / uint32 / uint8)."""
        return {f["name"]: f["dtype"] for f in self.raw_schema.get("features", []) if "dtype" in f}

    def validate_features(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Kiểm tra ma trận đặc trưng đối chiếu với hợp đồng:
        - Có đủ khoá danh tính không?
        - Có đủ các đặc trưng cốt lõi (hoặc biến thể thô tương ứng) không?
        - Các tỷ lệ (ratios) có nằm trong khoảng [0, 1] không?
        - Cột có mặt có đúng họ kiểu khai báo (số thực / số nguyên) không?
        """
        cols = set(df.columns)
        missing_keys = [k for k in self.identity_keys if k not in cols]
        missing_label = [self.label_key] if self.label_key not in cols else []

        # Kiểm tra tỷ lệ nằm ngoài đoạn [0, 1]
        ratio_cols = [c for c in cols if c.endswith("_ratio") or c.endswith("_share")]
        invalid_ratios = {}
        for rc in ratio_cols:
            s = df[rc].drop_nulls()
            if s.len() > 0:
                min_v = float(s.min())
                max_v = float(s.max())
                if min_v < -1e-6 or max_v > 1.0 + 1e-6:
                    invalid_ratios[rc] = {"min": min_v, "max": max_v}

        # Kiểu dữ liệu: so theo họ (số thực / số nguyên), không so độ rộng bit
        dtype_mismatch = {}
        for name, want in self.feature_dtypes.items():
            if name not in cols:
                continue
            got = df.schema[name]
            ok = got.is_float() if want.startswith("float") else got.is_integer()
            if not ok:
                dtype_mismatch[name] = {"expected": want, "actual": str(got)}

        return {
            "is_valid": (len(missing_keys) == 0 and len(missing_label) == 0 and len(invalid_ratios) == 0
                         and len(dtype_mismatch) == 0),
            "missing_keys": missing_keys,
            "missing_label": missing_label,
            "invalid_ratios": invalid_ratios,
            "dtype_mismatch": dtype_mismatch,
            "row_count": df.height,
            "column_count": df.width,
        }
