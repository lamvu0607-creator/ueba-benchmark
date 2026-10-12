"""
Đo đồng thuận của review top 50 và tổng hợp nhãn theo mô hình (tuần 5).

    python scripts/evaluation/review_agreement.py --sheet experiments/analysis/review_top50/review_sheet.csv

- Cần hai cột `nhan_nguoi_1`, `nhan_nguoi_2` (a / b / c): Cohen's κ, % đồng ý, ma trận bất đồng, danh sách bất đồng.
- Nếu có `nhan_thong_nhat` (hoặc thiếu thì dùng nhãn khi hai người đồng ý): tỷ lệ a/b/c trong top 50 của từng mô hình
  (ghép `answer_key.csv`), kèm phân theo nửa trên (hạng 1–25) / nửa dưới (26–50).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABELS = ["a", "b", "c"]
NAMES = {"a": "đáng ngờ", "b": "giải thích được", "c": "nhiễu"}


def clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str[:1].where(lambda x: x.isin(LABELS))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--sheet", default="experiments/analysis/review_top50/review_sheet.csv")
    p.add_argument("--key", default="experiments/analysis/review_top50/answer_key.csv")
    p.add_argument("--out", default="experiments/analysis/review_top50")
    args = p.parse_args()

    sheet = pd.read_csv(args.sheet, encoding="utf-8-sig", dtype=str).fillna("")
    r1, r2 = clean(sheet["nhan_nguoi_1"]), clean(sheet["nhan_nguoi_2"])
    both = r1.notna() & r2.notna()
    out, report = Path(args.out), {"n_alerts": len(sheet), "n_rated_by_both": int(both.sum())}
    if both.any():
        report.update(kappa=round(float(cohen_kappa_score(r1[both], r2[both], labels=LABELS)), 3),
                      agreement=round(float((r1[both] == r2[both]).mean()), 3))
        cm = pd.crosstab(r1[both].rename("nguoi_1"), r2[both].rename("nguoi_2")).reindex(
            index=LABELS, columns=LABELS, fill_value=0)
        cm.to_csv(out / "agreement_matrix.csv", encoding="utf-8-sig")
        sheet[both & (r1 != r2)].to_csv(out / "disagreements.csv", index=False, encoding="utf-8-sig")
        print(f"Cohen's kappa = {report['kappa']}, đồng ý {report['agreement']:.0%} trên {both.sum()} cảnh báo")
        print(cm.to_string())

    final = clean(sheet["nhan_thong_nhat"]) if "nhan_thong_nhat" in sheet else pd.Series(index=sheet.index)
    final = final.fillna(r1.where(r1 == r2))
    if final.notna().any() and Path(args.key).is_file():
        key = pd.read_csv(args.key, encoding="utf-8-sig")
        lab = key.merge(sheet[["alert_id"]].assign(label=final.to_numpy()), on="alert_id")
        lab["half"] = (lab["rank"] > 25).map({False: "hạng 1–25", True: "hạng 26–50"})
        tab = lab.pivot_table(index="model", columns="label", values="alert_id", aggfunc="count", fill_value=0)
        tab = tab.reindex(columns=LABELS, fill_value=0)
        tab["chưa có nhãn"] = lab.groupby("model")["label"].apply(lambda s: int(s.isna().sum()))
        tab.rename(columns=NAMES).to_csv(out / "labels_by_model.csv", encoding="utf-8-sig")
        lab.pivot_table(index=["model", "half"], columns="label", values="alert_id", aggfunc="count",
                        fill_value=0).to_csv(out / "labels_by_model_half.csv", encoding="utf-8-sig")
        report["labels_by_model"] = tab.rename(columns=NAMES).to_dict(orient="index")
        print(tab.rename(columns=NAMES).to_string())
    (out / "agreement_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=int),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
