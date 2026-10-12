import numpy as np

from src.evaluation.scenario import daily_topk_mask, rank_percentile, scenario_detection_rows, top_fraction_mask


def test_daily_topk_mask_per_day():
    scores = np.array([5, 1, 3, 9, 2, 8])
    days = np.array([1, 1, 1, 2, 2, 2])
    assert daily_topk_mask(scores, days, 1).tolist() == [True, False, False, True, False, False]
    assert daily_topk_mask(scores, days, 2).sum() == 4


def test_top_fraction_and_rank():
    scores = np.arange(10, dtype=float)
    assert top_fraction_mask(scores, 0.2).tolist() == [False] * 8 + [True, True]
    pct = rank_percentile(scores)
    assert pct[-1] == 0.1 and pct[0] == 1.0


def test_scenario_detection_rows():
    y = np.array([1, 1, 0, 0, 1, 0])
    sc = np.array(["a", "b", None, None, "a", None], dtype=object)
    scores = np.array([9, 1, 2, 3, 8, 0], dtype=float)
    flags = scores >= 8
    days = np.array([1, 1, 1, 2, 2, 2])
    rows = {r["scenario"]: r for r in scenario_detection_rows(y, sc, scores, flags, days, daily_ks=(1,))}
    assert rows["all"]["n_positive"] == 3
    assert rows["a"]["recall_threshold"] == 1.0 and rows["b"]["recall_threshold"] == 0.0
    assert rows["a"]["recall_top1_day"] == 1.0 and rows["b"]["recall_top1_day"] == 0.0
