"""
Ba baseline so sánh với IForest / LOF / OCSVM (cấu hình: ``configs/baselines.yaml``).

  * ``random_baseline``  — điểm ngẫu nhiên đều, seed cố định; PR-AUC kỳ vọng = tỷ lệ dương tính,
  * ``zscore_baseline``  — max_j |robust z| trên các đặc trưng core, toàn cục và theo tài khoản,
  * ``rule_baseline``    — 6 luật R1–R6 quy về ECDF train, điểm = max (+ biến thể ngưỡng ngoài),
  * ``rule_stats``       — R2/R4/R6 tính từ log sự kiện (không có sẵn trong ma trận),
  * ``thresholding``     — ngưỡng theo ngân sách cảnh báo, dùng chung cho mọi phương pháp,
  * ``runner``           — chạy end-to-end (``python main.py --stage baselines``).

Bất biến: mọi tham số học được đều fit trên TRAIN không nhãn; không module nào ở đây đọc nhãn.
"""

from src.baselines.random_baseline import RandomBaseline, expected_random_pr_auc
from src.baselines.rule_baseline import RULES, EcdfRuleBaseline, attach_rule_statistics
from src.baselines.thresholding import BudgetThreshold, alerts_per_day, apply_threshold, fit_budget_threshold
from src.baselines.zscore_baseline import MAD_CONSISTENCY, AccountRobustZScore, GlobalRobustZScore

__all__ = [
    "RandomBaseline",
    "expected_random_pr_auc",
    "RULES",
    "EcdfRuleBaseline",
    "attach_rule_statistics",
    "BudgetThreshold",
    "alerts_per_day",
    "apply_threshold",
    "fit_budget_threshold",
    "MAD_CONSISTENCY",
    "AccountRobustZScore",
    "GlobalRobustZScore",
]
