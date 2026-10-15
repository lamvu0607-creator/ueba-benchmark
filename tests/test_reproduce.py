import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location("reproduce", Path(__file__).resolve().parents[1] / "scripts/reproduce.py")
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


def test_verify_interim_rejects_changed_input(tmp_path):
    interim = tmp_path / "interim"
    path = interim / "event_4624" / "event_4624_day-01.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    import hashlib
    manifest = tmp_path / "hashes.txt"
    manifest.write_text(f"{hashlib.sha256(b'original').hexdigest()}  event_4624/event_4624_day-01.parquet\n")

    reproduce.verify_interim(interim, manifest)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Hash interim không khớp"):
        reproduce.verify_interim(interim, manifest)


def test_compare_rejects_missing_metric(tmp_path):
    reference = tmp_path / "reference.csv"
    result = tmp_path / "result.csv"
    row = {"model": "isolation_forest", **{f"{m}_mean": 0.5 for m in reproduce.METRICS}}
    pd.DataFrame([row]).to_csv(reference, index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "pr_auc_mean"}]).to_csv(result, index=False)

    with pytest.raises(ValueError, match="thiếu model/metric"):
        reproduce.compare(result, reference, 0.01, 0.05)
