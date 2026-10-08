"""
Unit tests cho src/baselines/ (random, z-score robust, luật ECDF, ngưỡng ngân sách) và phần đánh giá chung
(PR-AUC theo tập con với tập âm chung, recall theo chiến dịch, loại eval_exclude).

Nhóm test RÒ RỈ khẳng định: không baseline nào nhận nhãn, mọi tham số/ngưỡng chỉ phụ thuộc train.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from src.baselines import rule_baseline, rule_stats, thresholding, zscore_baseline
from src.baselines import random_baseline as random_module
from src.baselines.random_baseline import STREAM_EVAL, RandomBaseline, expected_random_pr_auc
from src.baselines.rule_baseline import RULE_COLUMNS, EcdfRuleBaseline, attach_rule_statistics
from src.baselines.rule_stats import EventHistory, day_event_stats, estimate_lockout_threshold, lockout_episodes
from src.baselines.runner import fit_and_score_segment
from src.baselines.thresholding import alerts_per_day, apply_threshold, fit_budget_threshold
from src.baselines.zscore_baseline import MAD_CONSISTENCY, AccountRobustZScore, GlobalRobustZScore
from src.evaluation.labeled_eval import align_eval_labels, evaluate_detector, load_eval_labels
from src.evaluation.metrics import campaign_recall, operating_point, pr_auc_by_subset

BASELINE_DIR = Path(__file__).resolve().parents[1] / "src" / "baselines"


# --------------------------------------------------------------------------- #
# Dữ liệu giả
# --------------------------------------------------------------------------- #
def _matrix(n_accounts: int = 6, n_days: int = 10, seed: int = 0, day_offset: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_accounts):
        for d in range(1, n_days + 1):
            total = int(rng.integers(1, 50))
            rows.append({
                "DomainName": "dom", "UserName": f"User{a}", "day": d + day_offset, "entity_type": "User",
                "total_logons": float(total),
                "failure_ratio": float(rng.integers(0, total + 1) / total) if a % 2 else 0.0,
                "off_hours_ratio": float(rng.random()),
                "days_since_last_activity": None if d == 1 else 1.0,
                "f_a": float(rng.normal(10 * a, 1 + a)),
                "f_b": float(rng.normal(0, 1)),
                "r2_spray_accounts": float(rng.integers(0, 3)),
                "r4_new_sources": float(rng.integers(0, 2)),
                "r6_new_logontype_events": 0.0,
            })
    df = pl.DataFrame(rows)
    return df.with_columns(
        (pl.col("total_logons") * pl.col("failure_ratio")).round(0).alias("r1_failure_count"),
        (pl.col("total_logons") * pl.col("off_hours_ratio")).round(0).alias("r3_off_hours_count"),
    )


CFG = {"zscore": {"mad_floor": {"quantile": 0.01}, "per_account": {"min_train_days": 5}}, "random": {"seeds": [3]}}


# --------------------------------------------------------------------------- #
# Ngưỡng theo ngân sách
# --------------------------------------------------------------------------- #
def test_budget_threshold_is_train_quantile_and_respects_budget():
    scores = np.arange(1000, dtype=float)
    thr = fit_budget_threshold(scores, 0.01)
    assert thr.threshold == float(np.quantile(scores, 0.99, method="higher"))
    assert thr.train_alert_rate <= 0.01
    assert apply_threshold(np.array([thr.threshold, thr.threshold + 1]), thr).tolist() == [False, True]


def test_budget_threshold_ties_never_exceed_budget():
    scores = np.r_[np.zeros(990), np.ones(10)]
    thr = fit_budget_threshold(scores, 0.05)  # phân vị 0.95 = 0 -> chỉ 10 dòng điểm 1 bị cờ (1% <= 5%)
    assert thr.train_alert_rate == pytest.approx(0.01)


def test_alerts_per_day_counts():
    out = alerts_per_day([1, 0, 1, 1, 0], [1, 1, 2, 2, 3])
    assert out["n_days"] == 3
    assert out["alerts_per_day_max"] == 2 and out["alerts_per_day_min"] == 0


# --------------------------------------------------------------------------- #
# Baseline 1 — ngẫu nhiên
# --------------------------------------------------------------------------- #
def test_random_pr_auc_matches_positive_rate_over_seeds():
    rng = np.random.default_rng(0)
    labels = (rng.random(20_000) < 0.03).astype(int)
    expected = expected_random_pr_auc(labels)
    assert expected == pytest.approx(labels.mean())
    aps = [average_precision_score(labels, RandomBaseline(s).score(labels.size)) for s in range(40)]
    assert np.mean(aps) == pytest.approx(expected, abs=0.003)


def test_random_scores_reproducible_and_streams_differ():
    a, b = RandomBaseline(7).score(100, STREAM_EVAL), RandomBaseline(7).score(100, STREAM_EVAL)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, RandomBaseline(7).score(100, 0))
    assert a.min() >= 0 and a.max() < 1


# --------------------------------------------------------------------------- #
# Baseline 2 — z-score robust
# --------------------------------------------------------------------------- #
def test_global_zscore_matches_hand_computation():
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    model = GlobalRobustZScore(["x"]).fit(train)
    assert model.location_[0] == 3.0
    assert model.scale_[0] == pytest.approx(MAD_CONSISTENCY * 1.0)  # |dev| = 2,1,0,1,97 -> median 1
    assert model.score(pl.DataFrame({"x": [3.0, 6.0]})).tolist() == pytest.approx([0.0, 3 / MAD_CONSISTENCY])


def test_global_zscore_mad_zero_uses_floor_and_constant_excluded():
    train = pl.DataFrame({"x": [0.0] * 8 + [1.0, 5.0], "c": [2.0] * 10})
    model = GlobalRobustZScore(["x", "c"], mad_floor_quantile=0.0).fit(train)
    assert model.mad_[0] == 0
    assert model.scale_[0] == pytest.approx(MAD_CONSISTENCY * 1.0)  # min độ lệch khác 0 = 1
    assert model.floored_features_ == ["x"] and model.constant_features_ == ["c"]
    assert model.score(pl.DataFrame({"x": [0.0], "c": [50.0]}))[0] == 0.0


def test_global_zscore_imputes_null_with_train_median():
    train = pl.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0]})
    model = GlobalRobustZScore(["x"]).fit(train)
    assert model.score(pl.DataFrame({"x": [None]}, schema={"x": pl.Float64}))[0] == 0.0


def test_account_zscore_own_stats_and_global_fallback():
    rows = [{"DomainName": "d", "UserName": "rich", "day": i, "x": float(100 + i % 3)} for i in range(10)]
    rows += [{"DomainName": "d", "UserName": "poor", "day": i, "x": 0.0} for i in range(2)]
    train = pl.DataFrame(rows)
    model = AccountRobustZScore(["x"], min_train_days=5).fit(train)
    assert model.n_accounts_own_stats_ == 1
    test = pl.DataFrame({"DomainName": ["d", "d", "d"], "UserName": ["rich", "poor", "new"], "day": [11] * 3,
                         "x": [101.0, 101.0, 101.0]})
    s = model.score(test)
    assert s[0] == 0.0  # đúng median của chính tài khoản
    glob = GlobalRobustZScore(["x"]).fit(train).score(test)
    assert s[1] == pytest.approx(glob[1]) and s[2] == pytest.approx(glob[2])


# --------------------------------------------------------------------------- #
# Baseline 3 — luật ECDF
# --------------------------------------------------------------------------- #
def _rule_frame(values: dict) -> pl.DataFrame:
    n = len(next(iter(values.values())))
    return pl.DataFrame({c: values.get(c, [0.0] * n) for c in RULE_COLUMNS})


def test_ecdf_left_zero_scores_zero_and_above_max_scores_one():
    train = _rule_frame({"r1_failure_count": [0.0] * 95 + [1, 2, 3, 4, 5]})
    model = EcdfRuleBaseline().fit(train)
    s = model.score(_rule_frame({"r1_failure_count": [0.0, 3.0, 99.0]}))
    assert s.tolist() == pytest.approx([0.0, 0.97, 1.0])


def test_external_variant_uses_indicator_for_r1_r2_only():
    train = _rule_frame({"r1_failure_count": list(range(100)), "r3_off_hours_count": list(range(100))})
    model = EcdfRuleBaseline({"r1_failure_count": 50, "r2_spray_accounts": 30}).fit(train)
    comp = model.components(_rule_frame({"r1_failure_count": [49.0, 50.0], "r3_off_hours_count": [10.0, 10.0]}))
    assert comp[:, 0].tolist() == [0.0, 1.0]
    assert comp[:, 2].tolist() == pytest.approx([0.10, 0.10])
    assert model.name == "rule_external"


def test_attach_rule_statistics_derives_r1_r3_and_joins_event_stats():
    m = pl.DataFrame({"DomainName": ["d"], "UserName": ["u"], "day": [3], "total_logons": [10.0], "failure_ratio": [0.3],
                      "off_hours_ratio": [0.55], "days_since_last_activity": [2.0]})
    ev = pl.DataFrame({"DomainName": ["d"], "UserName": ["u"], "day": [3], "r2_spray_accounts": [4.0],
                       "r4_new_sources": [1.0], "r6_new_logontype_events": [None]}, schema_overrides={"r6_new_logontype_events": pl.Float64})
    out = attach_rule_statistics(m, ev)
    assert out["r1_failure_count"][0] == 3.0 and out["r3_off_hours_count"][0] == 6.0
    assert out["r2_spray_accounts"][0] == 4.0 and out["r6_new_logontype_events"][0] is None


def _events(rows):
    return pl.DataFrame(rows, schema={"DomainName": pl.String, "UserName": pl.String, "EventID": pl.Int64,
                                      "Source": pl.String, "LogonType": pl.Int64, "Time": pl.Int64,
                                      "FailureReason": pl.String}, orient="row")


def test_day_event_stats_spray_new_source_new_logontype():
    h = EventHistory()
    d1 = _events([("d", "a", 4624, "S1", 3, 0, None), ("d", "b", 4624, "S1", 3, 0, None)])
    first = day_event_stats(d1, 1, h).sort("UserName")
    assert first["r4_new_sources"].null_count() == 2  # chưa có lịch sử -> NULL

    d2 = _events([
        ("d", "a", 4625, "X", 3, 0, "bad"), ("d", "b", 4625, "X", 3, 0, "bad"), ("d", "c", 4625, "X", 3, 0, "bad"),
        ("d", "a", 4624, "S2", 10, 1, None), ("d", "a", 4624, "S2", 10, 2, None), ("d", "b", 4624, "S1", 3, 1, None),
    ])
    out = {r["UserName"]: r for r in day_event_stats(d2, 2, h).iter_rows(named=True)}
    assert out["a"]["r2_spray_accounts"] == 3.0  # Source X gây thất bại cho 3 tài khoản
    assert out["a"]["r4_new_sources"] == 2.0  # X, S2 mới với a
    assert out["a"]["r6_new_logontype_events"] == 2.0  # 2 sự kiện type 10 chưa từng thấy
    assert out["b"]["r4_new_sources"] == 1.0 and out["b"]["r6_new_logontype_events"] == 0.0
    assert out["c"]["r4_new_sources"] is None  # c xuất hiện lần đầu


def test_lockout_threshold_from_failures_before_first_lock():
    ev = _events([
        ("d", "a", 4625, "X", 3, t, "Unknown user name or bad password.") for t in range(5)
    ] + [("d", "a", 4625, "X", 3, 10, "Account locked out."), ("d", "a", 4625, "X", 3, 11, "bad")]
      + [("d", "b", 4625, "X", 3, 0, "Account locked out.")])
    eps = lockout_episodes(ev)
    assert sorted(eps.tolist()) == [5.0]  # b: 0 thất bại trước khoá -> bỏ
    assert estimate_lockout_threshold([3, 4, 5, 0])["lockout_threshold"] == 4


# --------------------------------------------------------------------------- #
# Đánh giá chung
# --------------------------------------------------------------------------- #
def test_pr_auc_by_subset_shares_negatives():
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.35, 0.05])
    subsets = np.array([None, None, None, None, "A", "A", "B"], dtype=object)
    out = pr_auc_by_subset(labels, scores, subsets)
    assert {k: v["n_neg"] for k, v in out.items()} == {"__all__": 4, "A": 4, "B": 4}
    assert out["B"]["pr_auc"] == pytest.approx(average_precision_score([0, 0, 0, 0, 1], [0.1, 0.2, 0.3, 0.4, 0.05]))
    assert out["A"]["random_pr_auc"] == pytest.approx(2 / 6)


def test_operating_point_and_campaign_recall():
    labels = np.array([1, 1, 1, 0, 0])
    flags = np.array([1, 0, 0, 1, 0])
    op = operating_point(labels, flags)
    assert op["precision"] == 0.5 and op["recall"] == pytest.approx(1 / 3)
    camp = campaign_recall(labels, flags, np.array(["c1", "c1", "c2", None, None], dtype=object))
    assert camp == {"n_campaigns": 2.0, "n_campaigns_detected": 1.0, "campaign_recall": 0.5}


def test_evaluate_detector_drops_eval_exclude(temp_artifact_dir):
    labels = pl.DataFrame({"DomainName": ["d", "d"], "UserName": ["u1", "u2"], "day": [50, 50],
                           "label": [1, 1], "eval_exclude": [False, True], "campaign_id": ["c1", "c2"]})
    path = temp_artifact_dir / "labels.parquet"
    labels.write_parquet(path)
    keys = pl.DataFrame({"DomainName": ["d"] * 4, "UserName": ["u1", "u2", "u3", "u4"], "day": [50] * 4})
    aligned = align_eval_labels(keys, load_eval_labels(path))
    res = evaluate_detector([0.9, 0.95, 0.1, 0.2], 0.5, aligned, daily_budgets=[1])["summary"]
    assert res["n_excluded"] == 1 and res["n_eval"] == 3
    assert res["n_alerts"] == 1 and res["recall"] == 1.0 and res["campaign_recall"] == 1.0
    assert res["pr_auc"] == 1.0 and res["random_pr_auc"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# RÒ RỈ train/test
# --------------------------------------------------------------------------- #
def test_no_baseline_fit_accepts_labels():
    for cls in (RandomBaseline, GlobalRobustZScore, AccountRobustZScore, EcdfRuleBaseline):
        params = set(inspect.signature(cls.fit).parameters) - {"self"}
        assert not params & {"labels", "y", "is_anomaly", "eval", "test"}, cls


def test_baseline_modules_never_read_labels():
    """Phân tích tĩnh: các module baseline không import module nhãn và không nhắc tới cột nhãn."""
    banned_names = {"is_anomaly", "eval_exclude", "campaign_id", "load_eval_labels", "align_eval_labels", "load_labels"}
    for module in (random_module, zscore_baseline, rule_baseline, rule_stats, thresholding):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any("labeled_eval" in n or "benchmark" in n for n in names), module.__name__
            if isinstance(node, ast.Name):
                assert node.id not in banned_names, (module.__name__, node.id)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in banned_names, (module.__name__, node.value)


def test_fitted_parameters_and_thresholds_independent_of_eval_data():
    train = _matrix(seed=1)
    features = ["f_a", "f_b"]
    eval_a = _matrix(seed=2, day_offset=10)
    eval_b = eval_a.with_columns(pl.col("f_a") * 1000, pl.col("r1_failure_count") + 500)  # tập test cực đoan

    tr_a, ev_a, _ = fit_and_score_segment(train, eval_a, features, CFG, {"r1_failure_count": 5})
    tr_b, ev_b, _ = fit_and_score_segment(train, eval_b, features, CFG, {"r1_failure_count": 5})
    for method in tr_a:
        assert np.array_equal(tr_a[method], tr_b[method]), method
        assert fit_budget_threshold(tr_a[method], 0.05) == fit_budget_threshold(tr_b[method], 0.05)
    assert not np.array_equal(ev_a["zscore_global"], ev_b["zscore_global"])


def test_scores_are_row_wise_independent_of_other_eval_rows():
    train, evaluation = _matrix(seed=3), _matrix(seed=4, day_offset=10)
    for model in (GlobalRobustZScore(["f_a", "f_b"]), AccountRobustZScore(["f_a", "f_b"], min_train_days=5),
                  EcdfRuleBaseline()):
        model.fit(train)
        full = model.score(evaluation)
        assert np.allclose(full[:7], model.score(evaluation.head(7)))
        assert np.allclose(full[::-1], model.score(evaluation.reverse()))


def test_floating_point_noise_is_treated_as_zero_deviation():
    """Độ lệch 1e-16 (làm tròn dấu phẩy động) không được thành sàn MAD -> |z| không bùng nổ."""
    train = pl.DataFrame({"x": [0.0] * 6 + [1e-16, -1e-16, 2.0, 4.0]})
    model = GlobalRobustZScore(["x"], mad_floor_quantile=0.0).fit(train)
    assert model.scale_[0] == pytest.approx(MAD_CONSISTENCY * 2.0)
    acct = AccountRobustZScore(["x"], mad_floor_quantile=0.0, min_train_days=3).fit(
        pl.DataFrame({"DomainName": ["d"] * 10, "UserName": ["u"] * 5 + ["v"] * 5, "day": list(range(10)),
                      "x": [0.0, 1e-16, -1e-16, 1e-16, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]}))
    assert acct.floor_[0] == pytest.approx(MAD_CONSISTENCY * 1.0)  # chỉ MAD thật của v (=1×1.4826)


def test_baseline_runner_restricts_block_and_keeps_train_thresholds(temp_artifact_dir, monkeypatch):
    import json
    from types import SimpleNamespace
    import yaml
    import src.baselines.runner as runner
    import src.features.schema as schema

    matrix = _matrix()
    data_path = temp_artifact_dir / 'matrix.parquet'
    matrix.write_parquet(data_path)
    sys_path = temp_artifact_dir / 'system.yaml'
    sys_path.write_text(yaml.safe_dump({'evaluation': {'split_day': 7}}))
    cfg_path = temp_artifact_dir / 'baselines.yaml'
    cfg_path.write_text(yaml.safe_dump({**CFG, 'common': {'use_segments': False, 'budget_ratio': 0.05}}))
    monkeypatch.setattr(schema, 'FeatureSchema', lambda: SimpleNamespace(core_features=['f_a', 'f_b']))
    # Event statistics are already present in the fixture; isolate runner scope from log I/O.
    monkeypatch.setattr(runner, 'load_or_compute_event_stats', lambda *a: (pl.DataFrame(), {'lockout_threshold': 5}))
    monkeypatch.setattr(runner, 'attach_rule_statistics', lambda df, stats: df)
    labels_path = temp_artifact_dir / 'labels.parquet'
    pl.DataFrame({'DomainName': ['dom', 'dom'], 'UserName': ['User0', 'User1'], 'day': [8, 8],
                  'is_anomaly': [1, 1], 'eval_exclude': [False, True]}).write_parquet(labels_path)
    outputs = []
    for name, days in [('full', None), ('dev', [8])]:
        outputs.append(runner.run_baselines(system_config_path=sys_path, baselines_config_path=cfg_path,
                       data_path=data_path, labels_path=labels_path, output_dir=temp_artifact_dir / name,
                       eval_days=days))
    full, scoped = outputs
    scores = pl.read_parquet(scoped['scores_path'])
    assert scores.filter(pl.col('split') == 'eval')['day'].unique().to_list() == [8]
    assert scores.filter(pl.col('split') == 'train')['day'].max() == 7
    assert scoped['thresholds'] == full['thresholds']
    assert set(scoped['metrics'].filter(pl.col('method') != 'random_multi_seed_mean')['n_eval']) == {5.0}
    meta = json.loads((temp_artifact_dir / 'dev' / 'thresholds.json').read_text(encoding='utf-8'))
    assert meta['eval_days'] == [8] and meta['split']['n_eval'] == 6
