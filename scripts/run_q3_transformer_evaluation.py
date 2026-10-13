"""
PG27 NLP Coursework — Q3 Transformer Evaluation Runner
=========================================================
Author:   Anil Emanuel
Module:   COMM061 — Natural Language Processing
Purpose:  Drives src.evaluation.evaluate_predictions over the 13 transformer
          prediction CSVs shipped from otter24 (commit b9c488a). Produces
          per-checkpoint overall metrics, per-variety metrics, and a master
          model x test-variety macro-F1 heatmap for the Q3 report.

Run from repo root:
    python scripts/run_q3_transformer_evaluation.py

Outputs:
    results/q3_transformer_evaluation/                per-checkpoint outputs
    results/q3_transformer_evaluation/per_variety/    per-checkpoint x variety outputs
    results/q3_transformer_evaluation.json            bundled summary
    results/q3_transformer_evaluation/heatmap.png     model x variety macro-F1 grid
    results/q3_transformer_evaluation/heatmap.csv     same numbers as a CSV
"""

import json
import os
import sys
import glob
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# letting Python find src/evaluation.py when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_predictions


# folder the otter24 inference run dumped CSVs into — Phase J input
PRED_DIR = "results/predictions_otter24/"

# all outputs land under one new folder so Phase H LoRA results stay untouched
OUT_DIR = "results/q3_transformer_evaluation/"
PER_VARIETY_DIR = os.path.join(OUT_DIR, "per_variety/")
JSON_PATH = "results/q3_transformer_evaluation.json"

# 0 is non-sarcastic, 1 is sarcastic — same convention as Phase H
CLASS_NAMES = ["non-sarcastic", "sarcastic"]

# every BESSTIE variety code we expect in the test set — used to slice each CSV
VARIETIES = ["en-UK", "en-AU", "en-IN"]


def _np_default(obj):
    # json.dump cant handle numpy scalars natively — mirroring the pattern
    # from Phase H so the two JSON files stay consistent
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def model_label_from_csv(csv_path):
    # results/predictions_otter24/sarcasm_au_au+all_predictions.csv -> sarcasm_au_au+all
    base = os.path.basename(csv_path)
    return base.replace("_predictions.csv", "")


def evaluate_one_variety_slice(df, variety, model_label):
    # filter rows for this variety, write to a temp CSV, hand it to
    # evaluate_predictions which expects a path on disk — keeps us from
    # rewriting the metrics logic just to take an in-memory frame
    sub = df[df["variety"] == variety]
    if len(sub) == 0:
        return None

    task_name = f"{model_label} {variety}"
    # NamedTemporaryFile cleans itself up on close on POSIX, but on Windows
    # we need delete=False + manual unlink — using mkstemp for portability
    fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="_tmp_eval_", dir=OUT_DIR)
    os.close(fd)
    try:
        sub.to_csv(tmp_path, index=False)
        result = evaluate_predictions(
            csv_path=tmp_path,
            task_name=task_name,
            class_names=CLASS_NAMES,
            output_dir=PER_VARIETY_DIR,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PER_VARIETY_DIR, exist_ok=True)

    csvs = sorted(glob.glob(os.path.join(PRED_DIR, "*_predictions.csv")))
    if not csvs:
        print(f"[abort] no prediction CSVs found in {PRED_DIR}")
        return

    print(f"[setup] found {len(csvs)} prediction CSVs")

    # one bag for everything — written out as JSON at the end
    summary = {
        "task": "Q3 — Transformer cross-variety sarcasm evaluation on BESSTIE test set",
        "source_dir": PRED_DIR,
        "varieties": VARIETIES,
        "class_names": CLASS_NAMES,
        "models": {},
    }

    # one row per (model, variety) — pivoted into the heatmap at the end
    grid_rows = []

    for csv in csvs:
        label = model_label_from_csv(csv)
        print(f"\n[model] {label}")

        # overall metrics — full test set (UK + AU + IN combined)
        overall = evaluate_predictions(
            csv_path=csv,
            task_name=label,
            class_names=CLASS_NAMES,
            output_dir=OUT_DIR,
        )
        overall_macro_f1 = float(overall["macro_f1"])
        overall_baseline = overall["baseline"]["baseline_macro_f1"]

        # per-variety slices — filter each CSV by the variety column
        df = pd.read_csv(csv)
        per_variety = {}
        for v in VARIETIES:
            v_result = evaluate_one_variety_slice(df, v, label)
            if v_result is None:
                per_variety[v] = {"macro_f1": None, "support": 0}
                grid_rows.append({"model": label, "variety": v, "macro_f1": np.nan})
                continue
            v_macro_f1 = float(v_result["macro_f1"])
            v_support = int(v_result["metrics_df"]
                            .loc[v_result["metrics_df"]["class"] == "Macro Average", "support"]
                            .iloc[0])
            per_variety[v] = {
                "macro_f1": round(v_macro_f1, 4),
                "support": v_support,
                "baseline_macro_f1": round(v_result["baseline"]["baseline_macro_f1"], 4),
            }
            grid_rows.append({"model": label, "variety": v, "macro_f1": v_macro_f1})

        summary["models"][label] = {
            "csv_source": csv,
            "overall_macro_f1": round(overall_macro_f1, 4),
            "overall_baseline_macro_f1": round(overall_baseline, 4),
            "per_variety": per_variety,
        }

    # build the master heatmap — rows = model, columns = variety, values = macro-F1
    grid_df = pd.DataFrame(grid_rows)
    pivot = grid_df.pivot(index="model", columns="variety", values="macro_f1")
    # keep variety order canonical (UK, AU, IN) instead of alphabetical
    pivot = pivot[[v for v in VARIETIES if v in pivot.columns]]
    pivot.to_csv(os.path.join(OUT_DIR, "heatmap.csv"))

    fig, ax = plt.subplots(figsize=(8, 9))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=0.3, vmax=0.9, ax=ax, cbar_kws={"label": "macro-F1"})
    ax.set_title("Q3 — Transformer macro-F1 by model × test variety")
    ax.set_xlabel("Test variety")
    ax.set_ylabel("Model checkpoint")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "heatmap.png"), dpi=150)
    plt.close()

    # write JSON last so partial runs still produce per-checkpoint files
    with open(JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=_np_default)

    print()
    print(f"[done] bundled summary written to {JSON_PATH}")
    print(f"[done] per-checkpoint outputs in {OUT_DIR}")
    print(f"[done] per-variety outputs in {PER_VARIETY_DIR}")
    print(f"[done] heatmap saved to {OUT_DIR}heatmap.png")


if __name__ == "__main__":
    main()
