"""
Cấu hình chung cho bộ test của dự án.

Fixture ``temp_artifact_dir`` dùng thay ``tmp_path`` của pytest: trên một số máy Windows,
thư mục ``%TEMP%\\pytest-of-<user>`` bị khoá ACL (WinError 5) làm pytest không tạo được
base temp dir, trong khi việc tạo một thư mục tạm mới vẫn hoàn toàn bình thường.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture()
def temp_artifact_dir() -> Iterator[Path]:
    """Thư mục tạm ghi được để lưu artifact của test (.joblib, .csv...), tự dọn sau khi chạy."""
    path = Path(tempfile.mkdtemp(prefix="ueba-test-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
