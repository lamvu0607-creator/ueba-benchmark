"""Vẽ reports/week5/figures/window_comparison.png từ window_comparison.csv."""
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=pd.read_csv("experiments/grid/window_ablation/window_comparison.csv")
models=[("one_class_svm","OCSVM","#2a78d6"),("isolation_forest","IF","#eb6834"),("local_outlier_factor","LOF","#1baf7a")]
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,(m,title) in zip(axes,[("recall_at_100_per_day","Recall top-100/ngày (chỉ số chính)"),("pr_auc","PR-AUC")]):
    for key,lab,c in models:
        s=d[d.model==key].sort_values("window_days")
        ax.errorbar(s.window_days,s[m+"_mean"],yerr=s[m+"_std"],color=c,lw=2,marker="o",ms=8,capsize=4,label=lab)
        ax.annotate(lab,(21,s[m+"_mean"].iloc[-1]),xytext=(8,0),textcoords="offset points",va="center",color="#333",fontsize=10)
    ax.set_xticks([7,14,21]); ax.set_xticklabels(["7 ngày","14 ngày","21 ngày"]); ax.set_xlim(5,24.5)
    ax.set_title(title,fontsize=12); ax.set_ylim(bottom=0)
    ax.grid(axis="y",color="#ddd",lw=0.8); ax.set_axisbelow(True)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,loc="lower center",ncol=3,bbox_to_anchor=(0.5,0))
fig.suptitle("Độ dài cửa sổ lịch sử: TB ± std qua 5 model seed, run test_seed20261061",fontsize=12)
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig("reports/week5/figures/window_comparison.png",dpi=150)
