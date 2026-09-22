"""Member F, step 3: two-stage creator churn model.

Mirrors the demand-side design exactly:
    stage 1  churn        - does the creator upload anything at all in days 8-30?
    stage 2  "into the void" - keeps uploading, but receives no response
    model    HistGradientBoosting (as on the demand side), benchmarked against
             logistic regression and a prior-probability baseline
    metrics  ROC-AUC and PR-AUC, with PR-AUC read against the base rate
    split    fixed 70/15/15 hash on creatorId

    python -m src.member_f_supply_side.churn_model
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config

OUT = config.DERIVED_DIR / "member_f"
OUT.mkdir(parents=True, exist_ok=True)
R = {}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
ACC, ORA, GREY = "#0D6E6E", "#C77B30", "#8899A0"

F = pd.read_parquet(OUT / "creator_features.parquet")
seg = pd.read_parquet(OUT / "creator_segments.parquet")
F = F.merge(seg, on="creatorId", how="left")

# ------------------------------------------------------------------- Features
# Everything comes from the early window (days 1-7) or from attributes fixed
# before it. c_level is BANNED: it is the creator-side analogue of the demand
# side's `level` field and carries the same leakage risk (see src/config.py).
FEATS = [
    # production behaviour
    "e_n_pub", "e_pub_days",
    # audience response received
    "e_impr", "e_clicks", "e_likes", "e_comments", "e_shares",
    "e_follows", "e_viewcomments", "e_homepage",
    # rates
    "e_ctr", "e_like_rate", "e_engagement", "e_engagement_per_impr",
    # catalogue and pre-determined attributes
    "e_cards_live", "c_followers", "c_follows_out", "c_tenure", "c_type",
]
BANNED = ["c_level"]
R["features"] = FEATS
R["banned_features"] = BANNED

pop = F[F.e_pub_days > 0].copy()            # study population
R["studypop"] = int(len(pop))


def evaluate(df, target, name, key):
    d = df.dropna(subset=[target])
    tr, te = d[d.split == "train"], d[d.split == "test"]
    Xtr, ytr = tr[FEATS].values, tr[target].values.astype(int)
    Xte, yte = te[FEATS].values, te[target].values.astype(int)
    base = float(yte.mean())

    out = {"n_train": len(tr), "n_test": len(te), "base_rate": round(base, 4)}
    print(f"\n{'=' * 66}\n{name}\n{'=' * 66}")
    print(f"Training {len(tr):,} | test {len(te):,} | base rate {base:.3f}")

    models = {
        "Dummy (base rate)": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Gradient Boosted Trees": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42),
    }
    preds = {}
    print(f"\n  {'Model':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'Lift':>8}")
    print("  " + "-" * 52)
    for mname, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, p) if len(np.unique(p)) > 1 else 0.5
        pr = average_precision_score(yte, p)
        out[mname] = {"roc_auc": round(float(roc), 4), "pr_auc": round(float(pr), 4),
                      "pr_lift_vs_base": round(float(pr / base), 3)}
        preds[mname] = p
        print(f"  {mname:<26}{roc:>9.3f}{pr:>9.3f}{pr / base:>8.2f}x")

    # Permutation importance of the tree model
    best = models["Gradient Boosted Trees"]
    pi = permutation_importance(best, Xte, yte, n_repeats=5, random_state=0,
                                scoring="roc_auc")
    imp = sorted(zip(FEATS, pi.importances_mean, pi.importances_std),
                 key=lambda x: -x[1])
    out["importance"] = [{"feature": f, "mean": round(float(m), 5),
                          "std": round(float(s), 5)} for f, m, s in imp]
    print("\n  Most important features (drop in ROC-AUC when permuted):")
    for f, m, s in imp[:8]:
        print(f"    {f:<24}{m:+.4f}")

    R[key] = out
    return yte, preds["Gradient Boosted Trees"], base, out


# ============================================================== Stage 1: churn
y1, p1, b1, o1 = evaluate(pop, "churn",
                          "STAGE 1 - CREATOR CHURN (uploads nothing in days 8-30)",
                          "stage1_churn")

# =================================================== Stage 2: into the void
y2, p2, b2, o2 = evaluate(pop[pop.churn == 0], "void",
                          "STAGE 2 - KEEPS PUBLISHING, BUT GETS NO RESPONSE",
                          "stage2_void")

# --------------------------------------------- Funnel (mirrors the demand side)
n = len(pop)
ret = float((pop.churn == 0).mean())
act = float((pop[pop.churn == 0].void == 0).mean())
R["funnel"] = {"creators_active_week1": n, "still_publishing": round(ret, 4),
               "and_getting_traction": round(act, 4),
               "both": round(ret * act, 4)}
print(f"\n{'=' * 66}\nFUNNEL\n{'=' * 66}")
print("Of 100 creators active in week 1:")
print(f"  -> {ret * 100:.0f} still upload during days 8-30")
print(f"  -> of those, {act * 100:.0f}% also get a response")
print(f"  => {ret * act * 100:.0f} of 100 stay productive AND are seen")

# ------------------------------------------------- Heterogeneity by segment
te = pop[pop.split == "test"]
mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                     max_leaf_nodes=31, l2_regularization=1.0,
                                     random_state=42)
tr = pop[pop.split == "train"]
mdl.fit(tr[FEATS].values, tr.churn.values)
te = te.assign(pred=mdl.predict_proba(te[FEATS].values)[:, 1])
segres = []
for s, gdf in te.groupby("segment"):
    if gdf.churn.nunique() < 2:
        continue
    segres.append({"segment": int(s), "n_test": len(gdf),
                   "churn_rate": round(float(gdf.churn.mean()), 4),
                   "roc_auc": round(float(roc_auc_score(gdf.churn, gdf.pred)), 4)})
R["by_segment"] = segres
print(f"\n{'=' * 66}\nBY CREATOR SEGMENT (test partition)\n{'=' * 66}")
print(pd.DataFrame(segres).to_string(index=False))

# --------------------------------------------------------------------- Figures
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for (y, p, b, lab, c) in [(y1, p1, b1, "Stage 1 - churn", ACC),
                          (y2, p2, b2, "Stage 2 - no traction", ORA)]:
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, color=c, lw=1.7,
               label=f"{lab}  (AUC {roc_auc_score(y, p):.3f})")
    pre, rec, _ = precision_recall_curve(y, p)
    ax[1].plot(rec, pre, color=c, lw=1.7,
               label=f"{lab}  (AP {average_precision_score(y, p):.3f})")
    ax[1].axhline(b, color=c, ls=":", lw=1)
ax[0].plot([0, 1], [0, 1], color=GREY, ls="--", lw=.9, label="Random (0.500)")
ax[0].set_xlabel("False positive rate")
ax[0].set_ylabel("True positive rate")
ax[0].set_title("(a) ROC curves", loc="left", fontsize=9.5)
ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].set_title("(b) Precision-recall (dotted = base rate)", loc="left", fontsize=9.5)
ax[1].legend(frameon=False, fontsize=7.5, loc="lower left")
ax[1].set_ylim(0, 1.02)
plt.savefig(OUT / "fig3_churn_curves.png")
plt.close()

# Feature importance, stage 1
imp = pd.DataFrame(o1["importance"]).head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.barh(imp.feature, imp["mean"], xerr=imp["std"],
        color=[ACC if v > 0.001 else GREY for v in imp["mean"]],
        error_kw=dict(lw=.7, ecolor=GREY))
ax.set_xlabel("Decrease in test ROC-AUC when permuted")
ax.set_title("Feature importance - stage 1 (creator churn)", loc="left", fontsize=9.5)
plt.savefig(OUT / "fig4_importance.png")
plt.close()

json.dump(R, open(OUT / "03_churn_model.json", "w"), indent=2)
print("\n-> fig3_churn_curves.png, fig4_importance.png, 03_churn_model.json")
