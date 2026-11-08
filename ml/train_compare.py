"""Train an XGBoost fraud score and compare it with the rules engine.

    python ml/train_compare.py            # full run, writes ml/results/
    python ml/train_compare.py --quick    # small run (used by CI)

Protocol (designed so the model gets no unfair advantage):
    * Time-based split: fit = first 16 days, validation = next 4, test = last 8.
      The model never trains on transactions that happen after the ones it is
      scored on.
    * Rules engine: alert threshold tuned on fit+validation (best F1), so the
      baseline is the best version of the rules, not a strawman. The as-designed
      threshold (score >= 50) is reported too.
    * Model threshold: picked on *validation* to raise the same share of alerts
      as the tuned rules. Both operating points are then frozen and scored once
      on the untouched test period.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from features import MODEL_FEATURES, build_features
from simulate import SimConfig, simulate

RESULTS = Path(__file__).resolve().parent / "results"
MINUTES_PER_ALERT = 5  # assumption for analyst-workload framing; change to your team's figure


def split(df: pd.DataFrame, start, days: int):
    day = (df.ts - pd.Timestamp(start)).dt.days
    fit_end, val_end = round(days * 16 / 28), round(days * 20 / 28)
    return df[day < fit_end], df[(day >= fit_end) & (day < val_end)], df[day >= val_end]


def best_f1_threshold(y, score) -> float:
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f1[:-1])])


def threshold_for_alert_rate(score, rate: float) -> float:
    return float(np.quantile(score, 1 - rate))


def operating_point(y, score, thr, amount) -> dict:
    alert = score >= thr
    y = y.astype(bool)
    tp, fp = int((alert & y).sum()), int((alert & ~y).sum())
    n, pos = len(y), int(y.sum())
    return {
        "threshold": round(float(thr), 4),
        "alerts": int(alert.sum()),
        "alerts_per_10k_txns": round(alert.sum() / n * 1e4, 1),
        "fraud_caught": tp,
        "fraud_total": pos,
        "recall": round(tp / pos, 4),
        "precision": round(tp / max(alert.sum(), 1), 4),
        "false_positive_rate": round(fp / (n - pos), 5),
        "fraud_value_caught_pct": round(float(amount[alert & y].sum() / amount[y].sum()), 4),
        "analyst_hours": round(alert.sum() * MINUTES_PER_ALERT / 60, 1),
    }


def run(cfg: SimConfig) -> dict:
    df = build_features(simulate(cfg))
    fit, val, test = split(df, cfg.start, cfg.days)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=2, scale_pos_weight=10,
        tree_method="hist", enable_categorical=True, eval_metric="aucpr",
        early_stopping_rounds=40, random_state=cfg.seed, n_jobs=4,
    )
    model.fit(fit[MODEL_FEATURES], fit.is_fraud, eval_set=[(val[MODEL_FEATURES], val.is_fraud)], verbose=False)

    s_val = model.predict_proba(val[MODEL_FEATURES])[:, 1]
    s_test = model.predict_proba(test[MODEL_FEATURES])[:, 1]

    trainval = pd.concat([fit, val])
    rules_thr = best_f1_threshold(trainval.is_fraud, trainval.rules_score)
    rules_rate_val = float((val.rules_score >= rules_thr).mean())
    model_thr_matched = threshold_for_alert_rate(s_val, rules_rate_val)
    model_thr_f1 = best_f1_threshold(val.is_fraud, s_val)

    y, amt = test.is_fraud.to_numpy(), test.amount.to_numpy()
    res = {
        "data": {
            "transactions": len(df), "fraud_rate": round(float(df.is_fraud.mean()), 5),
            "test_transactions": len(test), "test_fraud": int(y.sum()),
            "split": "time-based: fit 16d / validation 4d / test 8d (scaled to config days)",
            "note": "Synthetic SG/MY portfolio from ml/simulate.py, not real bank data.",
        },
        "ranking": {
            "rules_pr_auc": round(average_precision_score(y, test.rules_score), 4),
            "model_pr_auc": round(average_precision_score(y, s_test), 4),
            "rules_roc_auc": round(roc_auc_score(y, test.rules_score), 4),
            "model_roc_auc": round(roc_auc_score(y, s_test), 4),
        },
        "operating_points": {
            "rules_as_designed_ge50": operating_point(y, test.rules_score.to_numpy(), 50, amt),
            "rules_tuned": operating_point(y, test.rules_score.to_numpy(), rules_thr, amt),
            "xgboost_same_alert_budget": operating_point(y, s_test, model_thr_matched, amt),
            "xgboost_best_f1": operating_point(y, s_test, model_thr_f1, amt),
        },
        "minutes_per_alert_assumption": MINUTES_PER_ALERT,
    }

    # Recall by fraud scenario at the two headline operating points
    by_scn = {}
    for scn, g in test[test.is_fraud == 1].groupby("scenario"):
        idx = g.index
        by_scn[scn] = {
            "n": len(g),
            "rules_tuned": round(float((test.loc[idx, "rules_score"] >= rules_thr).mean()), 3),
            "xgboost": round(float((s_test[test.index.get_indexer(idx)] >= model_thr_matched).mean()), 3),
        }
    res["recall_by_scenario"] = by_scn

    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    res["top_features_by_gain"] = [str(k) for k in imp.index[:8]]

    curves = {
        "rules": precision_recall_curve(y, test.rules_score),
        "xgboost": precision_recall_curve(y, s_test),
    }
    return res, curves


def plot(res: dict, curves: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    MODEL, RULES = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
    for ax in (a1, a2):
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    op = res["operating_points"]
    for key, col, name in (("xgboost", MODEL, "XGBoost"), ("rules", RULES, "Rules engine")):
        p, r, _ = curves[key]
        p, r = p[:-1], r[:-1]  # drop sklearn's artificial (recall=0, precision=1) endpoint
        a1.plot(r, p, color=col, linewidth=2, label=name, drawstyle="steps-post" if key == "rules" else None)
    pr, pm = op["rules_tuned"], op["xgboost_same_alert_budget"]
    a1.scatter([pr["recall"]], [pr["precision"]], s=70, color=RULES, edgecolor=SURF, linewidth=2, zorder=3)
    a1.scatter([pm["recall"]], [pm["precision"]], s=70, color=MODEL, edgecolor=SURF, linewidth=2, zorder=3)
    a1.annotate(f"Rules (tuned)\nrecall {pr['recall']:.0%}", (pr["recall"], pr["precision"]),
                xytext=(-12, 14), textcoords="offset points", ha="right", color=INK2, fontsize=9)
    a1.annotate(f"XGBoost, same alert volume\nrecall {pm['recall']:.0%}", (pm["recall"], pm["precision"]),
                xytext=(10, 6), textcoords="offset points", color=INK2, fontsize=9)
    a1.set_xlim(0, 1.02)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel("Recall (share of fraud caught)")
    a1.set_ylabel("Precision (share of alerts that are fraud)")
    a1.set_title("Precision vs recall on the test period", loc="left", fontsize=11, color=INK)
    a1.legend(frameon=False, loc="center left")

    scn = res["recall_by_scenario"]
    names = sorted(scn, key=lambda k: -scn[k]["xgboost"])
    x = np.arange(len(names))
    w = 0.38
    b1 = a2.bar(x - w / 2 - 0.01, [scn[n]["rules_tuned"] for n in names], w, color=RULES, label="Rules engine (tuned)")
    b2 = a2.bar(x + w / 2 + 0.01, [scn[n]["xgboost"] for n in names], w, color=MODEL, label="XGBoost, same alert volume")
    for bars in (b1, b2):
        for b in bars:
            a2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.0%}",
                    ha="center", fontsize=8.5, color=INK2)
    a2.set_xticks(x, [n.replace("_", " ").title() + f"\n(n={scn[n]['n']})" for n in names])
    a2.set_ylim(0, 1.22)
    a2.set_yticks(np.arange(0, 1.01, 0.2))
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_ylabel("Recall")
    a2.set_title("Which fraud each method catches", loc="left", fontsize=11, color=INK)
    a2.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)

    fig.text(0.01, 0.005, "Synthetic SG/MY card data (ml/simulate.py). Time-based split; thresholds fixed before scoring the test period.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small, fast run")
    args = ap.parse_args()
    cfg = SimConfig(n_customers=600, days=28, compromised_share=0.15) if args.quick else SimConfig()
    res, curves = run(cfg)
    RESULTS.mkdir(exist_ok=True)
    tag = "quick" if args.quick else "full"
    (RESULTS / f"metrics_{tag}.json").write_text(json.dumps(res, indent=2))
    if not args.quick:
        plot(res, curves, RESULTS / "model_vs_rules.png")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
