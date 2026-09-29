"""
Unit tests for data cleaner module.
"""

from pathlib import Path
import polars as pl
import pytest
from src.data.cleaner import LOGON_TYPE_MAP, ESSENTIAL_COLUMNS


def test_logon_type_map():
    assert LOGON_TYPE_MAP[2] == "Interactive"
    assert LOGON_TYPE_MAP[3] == "Network"
    assert LOGON_TYPE_MAP[5] == "Service"
    assert LOGON_TYPE_MAP[9] == "NewCredentials"
    assert LOGON_TYPE_MAP[10] == "RemoteInteractive"


def test_essential_columns_do_not_contain_subject_or_process():
    # Kiểm tra theo PDF: Subject* và Process* phải bị loại bỏ
    for col in ESSENTIAL_COLUMNS:
        assert not col.startswith("Subject"), f"Cột {col} không được có mặt theo quyết định PDF"
        assert not col.startswith("Process"), f"Cột {col} không được có mặt theo quyết định PDF"
        assert not col.startswith("ParentProcess"), f"Cột {col} không được có mặt theo quyết định PDF"


def test_cleaner_imputation_logic():
    # Tạo dữ liệu giả lập có null ở DomainName, LogonID, Source và LogonTypeDescription
    df = pl.DataFrame({
        "Time": [100, 200],
        "EventID": [4624, 4625],
        "LogHost": ["Comp1", "Comp2"],
        "LogonType": [2, 5],
        "LogonTypeDescription": [None, ""],
        "UserName": ["User1", "User2"],
        "DomainName": [None, ""],
        "LogonID": [None, "0x123"],
        "Source": ["", None],
        "AuthenticationPackage": ["Kerberos", "NTLM"],
        "FailureReason": [None, "account locked out"],
    })

    # Áp dụng logic làm sạch
    df_clean = df.with_columns([
        pl.when(pl.col("DomainName").is_null() | (pl.col("DomainName").str.strip_chars() == ""))
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("DomainName").str.strip_chars().str.to_lowercase())
        .alias("DomainName"),

        pl.when(pl.col("LogonID").is_null() | (pl.col("LogonID").str.strip_chars() == ""))
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("LogonID").str.strip_chars())
        .alias("LogonID"),

        pl.when(pl.col("Source").is_null() | (pl.col("Source").str.strip_chars() == ""))
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("Source").str.strip_chars())
        .alias("Source"),

        pl.when(pl.col("LogonTypeDescription").is_null() | (pl.col("LogonTypeDescription").str.strip_chars() == ""))
        .then(pl.col("LogonType").replace_strict(LOGON_TYPE_MAP, default=pl.lit("Unknown")))
        .otherwise(pl.col("LogonTypeDescription"))
        .alias("LogonTypeDescription"),
    ])

    # Kiểm tra kết quả
    assert df_clean["DomainName"].to_list() == ["Unknown", "Unknown"]
    assert df_clean["LogonID"].to_list() == ["Unknown", "0x123"]
    assert df_clean["Source"].to_list() == ["Unknown", "Unknown"]
    assert df_clean["LogonTypeDescription"].to_list() == ["Interactive", "Service"]
