# File kết quả, biểu đồ, markdown đã commit (12/10 → 13/10/2026)

Khoảng thời gian: từ 00:00 ngày 12/10 (giờ VN) đến sáng 13/10. Hôm nay (13/10) chưa có commit mới; commit cuối là 21:41 ngày 12/10.
Không tính file code (.py), test và config (.yaml).

Nhánh: tất cả nằm trên **main**, trừ mục 5 (ablation nhóm đặc trưng) chỉ có trên nhánh **feature/target-rate-1pct**, chưa merge vào main.

## 0. Đọc trước tiên

- [reports/week5/tom_tat_tuan5.md](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/reports/week5/tom_tat_tuan5.md): tóm tắt tuần 5, kết luận chính của từng việc và file cần đọc.
- [README.md](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/README.md): chỉ sửa định dạng tiêu đề (commit 21:41).

## 1. Cửa sổ lịch sử 7 / 14 / 21 ngày (main)

- [experiments/grid/window_ablation/README.md](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/grid/window_ablation/README.md): báo cáo thí nghiệm, kết luận giữ 7 ngày.
- [window_comparison.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/grid/window_ablation/window_comparison.csv): PR-AUC, P@k của từng mô hình theo 3 cửa sổ.
- [window_scenarios.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/grid/window_ablation/window_scenarios.csv): ROC/PR-AUC theo kịch bản cho từng cửa sổ.
- Thư mục [w7](https://github.com/lamvu0607-creator/ueba-benchmark/tree/main/experiments/grid/window_ablation/w7), [w14](https://github.com/lamvu0607-creator/ueba-benchmark/tree/main/experiments/grid/window_ablation/w14), [w21](https://github.com/lamvu0607-creator/ueba-benchmark/tree/main/experiments/grid/window_ablation/w21): mỗi thư mục có grid_runs/grid_summary/grid_scenarios(_summary)/contamination_*/pr_curves .csv, grid_manifest.json và 4 biểu đồ trong `figures/` (contamination_sensitivity, grid_pr_auc, pr_curves, scenario_roc_auc).

## 2. Phân tích theo kịch bản (main)

- [reports/week5/phan_tich_theo_kich_ban.md](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/reports/week5/phan_tich_theo_kich_ban.md): báo cáo mô hình nào bắt tốt kịch bản nào.
- [scenario_detection.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/scenario/scenario_detection.csv): recall trong 100 cảnh báo/ngày theo kịch bản và mô hình.
- [scenario_auc_runs.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/scenario/scenario_auc_runs.csv): ROC/PR-AUC theo kịch bản cho từng run.
- [complementarity.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/scenario/complementarity.csv): mức bổ trợ giữa các mô hình.
- [missed_victims.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/scenario/missed_victims.csv): các nạn nhân bị tất cả mô hình bỏ sót.
- [analysis_manifest.json](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/scenario/analysis_manifest.json): thông tin chạy.
- Biểu đồ: [scenario_recall_heatmap.png](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/scenario/figures/scenario_recall_heatmap.png), [victim_rank_boxplot.png](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/scenario/figures/victim_rank_boxplot.png).

## 3. Review top 50 cảnh báo (main)

- [reports/week5/review_top50.md](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/reports/week5/review_top50.md): kết quả lượt review của Claude.
- [docs/huong_dan_review_top50.md](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/docs/huong_dan_review_top50.md): hướng dẫn cách review.
- [review_sheet.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/review_top50/review_sheet.csv): phiếu review (ẩn nhãn) để điền.
- [claude_review.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/review_top50/claude_review.csv): nhãn Claude đã điền.
- [answer_key.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/review_top50/answer_key.csv): đáp án (chỉ mở sau khi review xong).
- [alert_cards.html](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/review_top50/alert_cards.html) / [alert_cards.json](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/review_top50/alert_cards.json): thẻ chi tiết từng cảnh báo.
- [review_manifest.json](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/review_top50/review_manifest.json): thông tin tạo bộ review.

## 4. Giải thích SHAP / LIME (main)

- [reports/week5/shap_lime.md](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/reports/week5/shap_lime.md): báo cáo giải thích mô hình, làm cơ sở cho ablation.
- [feature_importance.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/feature_importance.csv), [group_importance.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/group_importance.csv): độ quan trọng theo đặc trưng và theo nhóm.
- [shap_lime_agreement.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/shap_lime_agreement.csv): mức đồng thuận giữa SHAP và LIME.
- [origin_share.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/origin_share.csv): tỉ trọng đóng góp theo nguồn đặc trưng.
- [ablation_plan.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/ablation_plan.csv): kế hoạch ablation rút ra từ SHAP.
- Dữ liệu thô: [shap_values.parquet](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/shap_values.parquet), [lime_weights.parquet](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/lime_weights.parquet), [explain_manifest.json](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/explain_manifest.json).
- Biểu đồ: [top_features_shap_lime.png](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/figures/top_features_shap_lime.png), [shap_group_by_scenario.png](https://github.com/lamvu0607-creator/ueba-benchmark/blob/main/experiments/analysis/explain/figures/shap_group_by_scenario.png).

## 5. Ablation nhóm đặc trưng (CHỈ trên nhánh feature/target-rate-1pct, chưa merge)

- [reports/week5/ablation_nhom_dac_trung.md](https://github.com/lamvu0607-creator/ueba-benchmark/blob/feature/target-rate-1pct/reports/week5/ablation_nhom_dac_trung.md): báo cáo ablation trên dev, 5 run.
- [summary.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/feature/target-rate-1pct/experiments/ablation/feature_groups/summary.csv), [runs.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/feature/target-rate-1pct/experiments/ablation/feature_groups/runs.csv): kết quả tổng và từng run khi bỏ từng nhóm.
- [delta_overall.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/feature/target-rate-1pct/experiments/ablation/feature_groups/delta_overall.csv), [delta_scenario.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/feature/target-rate-1pct/experiments/ablation/feature_groups/delta_scenario.csv): mức thay đổi so với đủ đặc trưng, tổng và theo kịch bản.
- [scenario_summary.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/feature/target-rate-1pct/experiments/ablation/feature_groups/scenario_summary.csv), [scenario_runs.csv](https://github.com/lamvu0607-creator/ueba-benchmark/blob/feature/target-rate-1pct/experiments/ablation/feature_groups/scenario_runs.csv): kết quả theo kịch bản.
- [ablation_configs.json](https://github.com/lamvu0607-creator/ueba-benchmark/blob/feature/target-rate-1pct/experiments/ablation/feature_groups/ablation_configs.json): định nghĩa các nhóm bị bỏ.
- Biểu đồ: [delta_recall_by_scenario.png](https://github.com/lamvu0607-creator/ueba-benchmark/blob/feature/target-rate-1pct/experiments/ablation/feature_groups/figures/delta_recall_by_scenario.png).
