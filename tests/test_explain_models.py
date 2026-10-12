"""Kiểm tra phần tổng hợp của scripts/evaluation/explain_models.py (nhóm ablation, chuẩn hoá tỷ phần)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("explain_models", ROOT / "scripts/evaluation/explain_models.py")
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)


def test_ablation_groups_cover_core_features_once():
    origin = em.schema_origin(ROOT / "configs/feature_schema.yaml")
    flat = [f for fs in em.ABLATION_GROUPS.values() for f in fs]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(origin)


def test_to_long_shares_sum_to_one_per_row():
    names = list(em.FEATURE_GROUP)[:3]
    sel = pd.DataFrame({"model": ["m", "m"], "row": [0, 1], "kind": ["victim", "victim"], "scenario": ["s", "s"]})
    contrib = np.array([[1.0, -3.0, 0.0], [0.0, 0.0, 0.0]])
    long = em.to_long(sel, np.zeros_like(contrib), contrib, names, {n: "core_v3" for n in names})
    assert long.groupby("row")["share"].sum().tolist() == pytest.approx([1.0, 0.0])
    assert long.loc[long["row"] == 0, "share"].tolist() == pytest.approx([0.25, 0.75, 0.0])


def test_kernel_shap_recovers_linear_contributions():
    pytest.importorskip("shap")
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(20, 3))
    w = np.array([2.0, -1.0, 0.0])
    x = np.array([[1.0, 2.0, 3.0]])
    phi, base = em.run_shap(lambda Z: Z @ w, bg, x, nsamples=200)
    assert phi[0] == pytest.approx(w * (x[0] - bg.mean(axis=0)), abs=1e-6)
    assert base == pytest.approx(float(bg.mean(axis=0) @ w))
