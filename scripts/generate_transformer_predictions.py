"""
scripts/generate_transformer_predictions.py

Run every transformer checkpoint in results/checkpoints/ over the BESSTIE
test set and save row-level prediction CSVs for Q3 evaluation.

Output: results/predictions_otter24/<ckpt_name>_predictions.csv
        columns: text, variety, source, y_true, y_pred, prob_0, prob_1
"""

import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# task we're evaluating — sarcasm column in besstie_test.csv
TASK_LABEL_COL = "Sarcasm"

# fallback tokenizer per HF model_type — checkpoint configs don't carry the
# original _name_or_path, so map by architecture instead
TOKENIZER_FOR_TYPE = {
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "deberta": "microsoft/deberta-base",
    "deberta-v2": "microsoft/deberta-v3-base",
}


def find_inner_checkpoint(ckpt_root: Path) -> Path | None:
    # Trainer saves weights inside results/checkpoints/<name>/checkpoint-XXX/
    # return the highest-numbered checkpoint-* subfolder that has config.json
    candidates = sorted(ckpt_root.glob("checkpoint-*"))
    for c in reversed(candidates):
        if (c / "config.json").is_file() and (c / "model.safetensors").is_file():
            return c
    # some saves drop the inner level entirely — accept that too
    if (ckpt_root / "config.json").is_file() and (ckpt_root / "model.safetensors").is_file():
        return ckpt_root
    return None


def pick_tokenizer_name(config_path: Path) -> str | None:
    # _name_or_path first if present, otherwise fall back to model_type map
    cfg = json.loads(config_path.read_text())
    nop = cfg.get("_name_or_path") or ""
    if nop and not nop.startswith("/") and "/" in nop or nop in {"roberta-base", "bert-base-uncased", "distilbert-base-uncased"}:
        return nop
    mtype = cfg.get("model_type", "")
    return TOKENIZER_FOR_TYPE.get(mtype)


def predict_one_checkpoint(ckpt_dir: Path, df: pd.DataFrame, device: str, batch_size: int):
    # load model + tokenizer; tokenize in batches; argmax + softmax probs
    cfg_path = ckpt_dir / "config.json"
    tok_name = pick_tokenizer_name(cfg_path)
    if tok_name is None:
        raise RuntimeError(f"no tokenizer mapping for {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    model.eval()

    texts = df["text"].astype(str).tolist()
    n = len(texts)
    all_preds = []
    all_p0 = []
    all_p1 = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            logits = model(**enc).logits  # shape (B, num_labels)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_preds.extend(preds.tolist())
            # binary head expected; if num_labels != 2 just pad the second column with NaN
            if probs.shape[1] >= 2:
                all_p0.extend(probs[:, 0].tolist())
                all_p1.extend(probs[:, 1].tolist())
            else:
                all_p0.extend(probs[:, 0].tolist())
                all_p1.extend([float("nan")] * len(batch))

    # release GPU memory before next checkpoint loads
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return all_preds, all_p0, all_p1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="repo root (default: cwd)")
    ap.add_argument("--test-csv", default="data/raw/besstie_test.csv")
    ap.add_argument("--out-dir", default="results/predictions_otter24")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only", default="", help="comma-separated checkpoint names; empty = all")
    ap.add_argument("--skip-existing", action="store_true", help="skip if output CSV already there")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    os.chdir(repo)
    print(f"[setup] repo = {repo}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device = {device}")

    # load test set once — same file for every checkpoint
    df = pd.read_csv(args.test_csv)
    print(f"[setup] test rows = {len(df)} | columns = {list(df.columns)}")
    if TASK_LABEL_COL not in df.columns:
        raise RuntimeError(f"label column '{TASK_LABEL_COL}' missing from test CSV")
    # sarcasm column is float 0.0/1.0 — cast to int for clean y_true
    y_true = df[TASK_LABEL_COL].astype(float).astype(int).tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover checkpoints
    ckpt_root_dir = repo / "results" / "checkpoints"
    all_roots = sorted([p for p in ckpt_root_dir.iterdir() if p.is_dir()])
    if args.only:
        wanted = set(args.only.split(","))
        all_roots = [p for p in all_roots if p.name in wanted]

    print(f"[setup] candidate checkpoints = {len(all_roots)}")

    summary = []
    for root in all_roots:
        name = root.name
        out_csv = out_dir / f"{name}_predictions.csv"

        if args.skip_existing and out_csv.exists():
            print(f"[skip] {name} — output already exists")
            continue

        inner = find_inner_checkpoint(root)
        if inner is None:
            print(f"[skip] {name} — no usable checkpoint dir (missing config/safetensors)")
            summary.append((name, "SKIP", 0.0))
            continue

        print(f"[run]  {name} → {inner.name}")
        t0 = time.time()
        try:
            preds, p0, p1 = predict_one_checkpoint(inner, df, device, args.batch_size)
        except Exception as e:
            print(f"[fail] {name}: {type(e).__name__}: {e}")
            summary.append((name, f"FAIL:{type(e).__name__}", 0.0))
            continue

        out = df[["text", "variety", "source"]].copy()
        out["y_true"] = y_true
        out["y_pred"] = preds
        out["prob_0"] = p0
        out["prob_1"] = p1
        out.to_csv(out_csv, index=False)

        dt = time.time() - t0
        # rough accuracy for the log so we see something sane during the run
        correct = sum(1 for a, b in zip(y_true, preds) if a == b)
        acc = correct / len(y_true) if y_true else 0.0
        print(f"[done] {name} — wrote {out_csv.name} | {len(preds)} rows | acc={acc:.4f} | {dt:.1f}s")
        summary.append((name, "OK", dt))

    print("\n[summary]")
    for name, status, dt in summary:
        print(f"  {name:35s} {status:20s} {dt:.1f}s")
    print(f"[summary] total checkpoints processed: {len(summary)}")


if __name__ == "__main__":
    main()
