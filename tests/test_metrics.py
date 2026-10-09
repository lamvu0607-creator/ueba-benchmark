

def test_precision_at_k_per_day_averages_daily_topk():
    """Mục 5.4: P@k theo ngày = trung bình qua ngày của tỷ lệ đúng trong k cảnh báo của ngày đó."""
    import numpy as np

    from src.evaluation.metrics import labeled_metrics, precision_at_k_per_day

    days = np.array([1, 1, 1, 2, 2, 2])
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.5])
    labels = np.array([1, 0, 0, 0, 0, 1])
    assert precision_at_k_per_day(labels, scores, days, 2) == 0.25          # ngày 1: 1/2, ngày 2: 0/2
    assert precision_at_k_per_day(labels, scores, days, 10) == (1 / 3 + 1 / 3) / 2   # ngày ít hơn k dòng
    out = labeled_metrics(labels, scores, days, ks=[2], daily_budgets=[1])
    assert out["precision_at_2_per_day"] == 0.25 and out["precision_at_2"] == 0.5
