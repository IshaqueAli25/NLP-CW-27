"""
PG27 NLP Coursework — Q3 Evaluation Utility
============================================
Author:   Anil Emanuel
Module:   COMM061 — Natural Language Processing
Purpose:  Reusable evaluation functions for Q3.
          Takes prediction CSVs from teammates (Reshabh / Nakul / Priyanka)
          and produces per-class metrics, confusion matrices, and a
          majority-class baseline check.

Expected CSV columns (minimum):
    y_true, y_pred

Optional columns (used if present, for filtering/grouping):
    text, variety, train_variety, test_variety, adapter, model

USAGE EXAMPLE
-------------
    from src.evaluation import evaluate_predictions

    results = evaluate_predictions(
        csv_path="results/predictions/reshabh_roberta_sarcasm.csv",
        task_name="RoBERTa Sarcasm",
        class_names=["non-sarcastic", "sarcastic"],
        output_dir="results/q3_evaluation/"
    )
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
    f1_score,
    accuracy_score,
)


def compute_per_class_metrics(y_true, y_pred, class_names=None):
    """
    Compute per-class precision, recall, F1, and support.
    Also returns macro-F1 (the primary coursework metric).
    """
    # pulling every label that shows up in either column — the model might
    # never predict a class but it still needs a row in the table
    labels = sorted(list(set(list(y_true)) | set(list(y_pred))))
    if class_names is None:
        class_names = [str(label) for label in labels]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    df = pd.DataFrame({
        "class": class_names,
        "precision": np.round(precision, 4),
        "recall": np.round(recall, 4),
        "f1": np.round(f1, 4),
        "support": support.astype(int),
    })

    # macro-F1 is the headline number for Q3, so adding it as its own row
    # keeps everything in one table and saves us recomputing later
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    macro_row = pd.DataFrame([{
        "class": "Macro Average",
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f1, 4),
        "support": int(sum(support)),
    }])

    return pd.concat([df, macro_row], ignore_index=True)


def majority_class_baseline(y_true):
    """
    Compute what F1 a model would get if it always predicted
    the majority class. Q3 spec wants this — proves a real model
    is actually learning, not just guessing the dominant label.
    """
    y_true = pd.Series(y_true)
    majority = y_true.value_counts().idxmax()
    proportion = (y_true == majority).mean()

    # building a fake prediction list — every entry the majority label —
    # so sklearn metrics get something to chew on
    y_pred_majority = [majority] * len(y_true)
    baseline_accuracy = accuracy_score(y_true, y_pred_majority)
    baseline_macro_f1 = f1_score(
        y_true, y_pred_majority, average="macro", zero_division=0
    )

    return {
        "majority_class": majority,
        "majority_proportion": round(proportion, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "baseline_macro_f1": round(baseline_macro_f1, 4),
    }


def confusion_matrix_df(y_true, y_pred, class_names=None):
    """
    Return confusion matrix as a labeled DataFrame
    (rows = actual, columns = predicted).
    """
    labels = sorted(list(set(list(y_true)) | set(list(y_pred))))
    if class_names is None:
        class_names = [str(label) for label in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # prefixing 'actual_' / 'pred_' on rows and columns so the saved CSV
    # is readable on its own without needing this code as context
    return pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )


def plot_confusion_matrix(
    y_true, y_pred, class_names=None, title="Confusion Matrix", save_path=None
):
    """
    Plot a heatmap of the confusion matrix.
    Saves to disk if save_path provided. Returns the figure.
    """
    labels = sorted(list(set(list(y_true)) | set(list(y_pred))))
    if class_names is None:
        class_names = [str(label) for label in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        # making sure the output folder exists first — savefig won't create it
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")

    return fig


def evaluate_predictions(
    csv_path,
    task_name,
    class_names=None,
    output_dir="results/q3_evaluation/",
    y_true_col="y_true",
    y_pred_col="y_pred",
):
    """
    Full evaluation pipeline for one prediction CSV.

    Produces:
      - per-class metrics table (printed + saved as CSV)
      - confusion matrix DataFrame (printed + saved as CSV)
      - confusion matrix heatmap (saved as PNG)
      - majority-class baseline comparison (printed + saved as TXT)
    """
    os.makedirs(output_dir, exist_ok=True)
    # turning the task name into a safe filename — no spaces, no arrows, no slashes
    safe_name = task_name.lower().replace(" ", "_").replace("->", "to").replace("/", "_")

    df = pd.read_csv(csv_path)
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise KeyError(
            f"CSV must contain columns '{y_true_col}' and '{y_pred_col}'. "
            f"Got columns: {df.columns.tolist()}"
        )

    y_true = df[y_true_col].values
    y_pred = df[y_pred_col].values

    print("=" * 70)
    print(f"EVALUATION REPORT — {task_name}")
    print(f"Source: {csv_path}")
    print(f"Sample size: {len(df)}")
    print("=" * 70)

    metrics_df = compute_per_class_metrics(y_true, y_pred, class_names)
    print("\n--- Per-Class Metrics ---")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(
        os.path.join(output_dir, f"metrics_{safe_name}.csv"), index=False
    )

    cm_df = confusion_matrix_df(y_true, y_pred, class_names)
    print("\n--- Confusion Matrix ---")
    print(cm_df.to_string())
    cm_df.to_csv(os.path.join(output_dir, f"confusion_{safe_name}.csv"))

    baseline = majority_class_baseline(y_true)
    # pulling macro-F1 back out of the table — easier than recomputing it here
    macro_f1 = float(metrics_df.loc[
        metrics_df["class"] == "Macro Average", "f1"
    ].iloc[0])
    print("\n--- Majority-Class Baseline Check ---")
    print(f"Majority class:        {baseline['majority_class']}")
    print(f"Majority proportion:   {baseline['majority_proportion']:.2%}")
    print(f"Baseline macro-F1:     {baseline['baseline_macro_f1']:.4f}")
    print(f"Model macro-F1:        {macro_f1:.4f}")

    # using a 0.05 cutoff so genuine learning is filtered out from noise —
    # anything tighter would flag healthy models as weak
    diff = macro_f1 - baseline["baseline_macro_f1"]
    if diff > 0.05:
        verdict = f"PASS — model beats majority baseline by {diff:.4f}"
    elif diff > 0:
        verdict = f"WEAK — model only beats baseline by {diff:.4f}"
    else:
        verdict = f"FAIL — model does NOT beat majority baseline (diff {diff:.4f})"
    print(f"Verdict:               {verdict}")

    with open(os.path.join(output_dir, f"baseline_{safe_name}.txt"), "w") as f:
        f.write(f"Task: {task_name}\n")
        for k, v in baseline.items():
            f.write(f"{k}: {v}\n")
        f.write(f"model_macro_f1: {macro_f1}\n")
        f.write(f"verdict: {verdict}\n")

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        title=f"Confusion Matrix — {task_name}",
        save_path=os.path.join(output_dir, f"confusion_{safe_name}.png"),
    )
    plt.close()

    print("\n[done] All outputs saved to:", output_dir)
    print("=" * 70 + "\n")

    return {
        "metrics_df": metrics_df,
        "confusion_df": cm_df,
        "baseline": baseline,
        "macro_f1": macro_f1,
    }


def evaluate_cross_variety(
    csv_path,
    task_name="Cross-Variety Sarcasm",
    class_names=None,
    output_dir="results/q3_evaluation/cross_variety/",
):
    """
    Evaluate Nakul's Q2.2 cross-variety matrix.
    Expects a CSV with columns: train_variety, test_variety, y_true, y_pred.
    Produces a summary DataFrame of macro-F1 for every train -> test pair.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    required = {"train_variety", "test_variety", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rows = []
    for (train_v, test_v), sub in df.groupby(["train_variety", "test_variety"]):
        sub_name = f"{task_name} {train_v}->{test_v}"
        # writing each train->test slice to a tiny temp CSV so we can reuse
        # evaluate_predictions without duplicating its logic here
        tmp_csv = os.path.join(output_dir, f"_tmp_{train_v}_to_{test_v}.csv")
        sub.to_csv(tmp_csv, index=False)
        result = evaluate_predictions(
            csv_path=tmp_csv,
            task_name=sub_name,
            class_names=class_names,
            output_dir=output_dir,
        )
        os.remove(tmp_csv)
        rows.append({
            "train_variety": train_v,
            "test_variety": test_v,
            "macro_f1": result["macro_f1"],
            "baseline_macro_f1": result["baseline"]["baseline_macro_f1"],
        })

    matrix = pd.DataFrame(rows).pivot(
        index="train_variety", columns="test_variety", values="macro_f1"
    )
    print("\n--- Cross-Variety Macro-F1 Matrix ---")
    print(matrix.round(4).to_string())
    matrix.to_csv(os.path.join(output_dir, "cross_variety_matrix.csv"))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Cross-Variety Macro-F1 (rows = train, cols = test)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_variety_matrix.png"), dpi=150)
    plt.close()
    print(f"[saved] {output_dir}cross_variety_matrix.png")

    return matrix


if __name__ == "__main__":
    # quick smoke test using synthetic data — never used for any reported
    # number in the report, only here to prove the pipeline runs end to end
    import tempfile

    print("Running smoke test with synthetic data...\n")
    rng = np.random.RandomState(42)
    n = 200
    y_true_synthetic = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    # making the synthetic predictions roughly 75% accurate so the smoke
    # test exercises both correct and wrong paths through the metrics code
    y_pred_synthetic = np.where(
        rng.random(n) < 0.75, y_true_synthetic, 1 - y_true_synthetic
    )

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "synthetic.csv")
        pd.DataFrame({
            "y_true": y_true_synthetic,
            "y_pred": y_pred_synthetic,
        }).to_csv(csv_path, index=False)

        evaluate_predictions(
            csv_path=csv_path,
            task_name="Smoke Test",
            class_names=["non-sarcastic", "sarcastic"],
            output_dir=os.path.join(tmp, "out"),
        )
    print("Smoke test complete.")
