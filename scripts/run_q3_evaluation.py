"""
PG27 NLP Coursework — Q3 LoRA Evaluation Runner
=================================================
Author:   Anil Emanuel
Module:   COMM061 — Natural Language Processing
Purpose:  Drives src.evaluation.evaluate_predictions on the two LoRA prediction
          CSVs (UK + AU) shipped in Phase G, then bundles the headline numbers
          into a single JSON file for the Q3 report.

Run from repo root:
    python scripts/run_q3_evaluation.py

Outputs:
    results/q3_evaluation/             per-CSV tables, confusion CSVs, PNGs
    results/q3_lora_evaluation.json    bundled summary used by the report
"""

import json
import os
import sys
import numpy as np

# letting Python find src/evaluation.py when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_predictions


# pinning the two CSVs Phase G produced — typing them anywhere else would drift
PREDICTIONS = [
    {
        "csv": "results/predictions/lora_en_uk_sarcasm_predictions.csv",
        "task_name": "LoRA en-UK Sarcasm",
        "variety": "en-UK",
        "adapter": "en-UK_seed42",
    },
    {
        "csv": "results/predictions/lora_en_au_sarcasm_predictions.csv",
        "task_name": "LoRA en-AU Sarcasm",
        "variety": "en-AU",
        "adapter": "en-AU_seed42",
    },
]

# 0 is non-sarcastic, 1 is sarcastic — matches BESSTIE Sarcasm column convention
CLASS_NAMES = ["non-sarcastic", "sarcastic"]
OUTPUT_DIR = "results/q3_evaluation/"
JSON_PATH = "results/q3_lora_evaluation.json"




def _np_default(obj):
    # json.dump cant handle numpy scalars natively — pandas int64/float64
    # round-trip through this default and come out as plain Python types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # one bag for everything — written out as JSON at the end
    summary = {
        "task": "Q3 — LoRA cross-variety sarcasm evaluation on BESSTIE test set",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "class_names": CLASS_NAMES,
        "adapters_evaluated": [],
        "varieties": {},
    }

    for spec in PREDICTIONS:
        csv = spec["csv"]
        if not os.path.isfile(csv):
            # Phase G must have shipped the CSV — bail rather than guess
            print(f"[SKIP] {csv} not found — Phase G must run first")
            continue

        result = evaluate_predictions(
            csv_path=csv,
            task_name=spec["task_name"],
            class_names=CLASS_NAMES,
            output_dir=OUTPUT_DIR,
        )

        # turning the metrics DataFrame into a list of dicts — JSON-friendly
        metrics_records = result["metrics_df"].to_dict(orient="records")

        # confusion DataFrame keeps row labels as the index — orient='index'
        # gives a nested dict that round-trips cleanly through json.dumps
        confusion_records = result["confusion_df"].to_dict(orient="index")

        baseline = result["baseline"]
        macro_f1 = result["macro_f1"]
        diff = macro_f1 - baseline["baseline_macro_f1"]

        # same 0.05 cutoff logic src.evaluation uses — keeping verdicts consistent
        if diff > 0.05:
            verdict = f"PASS — beats majority baseline by {round(diff, 4)}"
        elif diff > 0:
            verdict = f"WEAK — only beats baseline by {round(diff, 4)}"
        else:
            verdict = f"FAIL — does not beat majority baseline (diff {round(diff, 4)})"

        summary["adapters_evaluated"].append(spec["adapter"])
        summary["varieties"][spec["variety"]] = {
            "csv_source": csv,
            "task_name": spec["task_name"],
            "adapter": spec["adapter"],
            "metrics": metrics_records,
            "confusion_matrix": confusion_records,
            "majority_baseline": baseline,
            "model_macro_f1": round(macro_f1, 4),
            "macro_f1_lift_over_baseline": round(diff, 4),
            "verdict": verdict,
        }

    # writing JSON last — partial runs still print to stdout via evaluate_predictions
    with open(JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=_np_default)

    print()
    print(f"[done] bundled summary written to {JSON_PATH}")
    print(f"[done] per-CSV outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
