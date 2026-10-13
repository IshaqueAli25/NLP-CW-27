"""
PG27 NLP Coursework — Generate LoRA Prediction CSVs
====================================================
Author:   Anil Emanuel
Module:   COMM061 — Natural Language Processing
Purpose:  Generate the 3 row-level prediction CSVs needed for Q3 evaluation
          of the LoRA sarcasm experiments. Replaces items 7, 8, 9 of the
          team requirements list (Priyanka's CSVs) — running her trained
          adapters on the test set ourselves instead of waiting.

Output (after running):
    results/predictions/lora_uk_sarcasm_predictions.csv
    results/predictions/lora_au_sarcasm_predictions.csv
    results/predictions/lora_in_sarcasm_predictions.csv

Each CSV has columns:  text, variety, adapter, y_true, y_pred

These can then be fed straight into src/evaluation.evaluate_predictions().

USAGE
-----
    python scripts/generate_lora_predictions.py \
        --test-csv path/to/besstie_test.csv \
        --adapter-root path/to/lora_adapters \
        --output-dir results/predictions/

If the test CSV is not in `text, label, variety` shape, edit the
COLUMN_MAP constant below to match your file.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np


# Qwen2.5-1.5B-Instruct is what Priyanka's adapter_config.json points at —
# any other base would refuse to load these adapters
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# only using the seed42 adapters — seed123 is kept as a backup for variance
# discussion in the report, not the headline number
VARIETIES = {
    "en-UK": "en-UK_seed42",
    "en-AU": "en-AU_seed42",
    "en-IN": "en-IN_seed42",
}

# mapping the test CSV's column names to the roles the script needs.
# edit these if the test CSV uses different names (e.g. 'sarcasm_label' instead of 'label')
COLUMN_MAP = {
    "text": "text",
    "label": "Sarcasm",
    "variety": "variety",
}


def run_inference_for_variety(
    base_model, tokenizer, adapter_path, variety,
    test_df, device, batch_size=16, max_length=128,
):
    """
    Load adapter, run inference on rows where variety matches, return DataFrame
    with columns: text, variety, adapter, y_true, y_pred.
    """
    import torch
    from peft import PeftModel

    print(f"\n[{variety}] Loading adapter from {adapter_path}...")
    if isinstance(base_model, PeftModel):
        # already a PEFT model from the previous loop iteration —
        # adding this adapter as a new named one rather than rebuilding
        adapter_name = variety
        if adapter_name not in base_model.peft_config:
            base_model.load_adapter(adapter_path, adapter_name=adapter_name)
        base_model.set_adapter(adapter_name)
        model = base_model
    else:
        # first variety in the loop — wrapping the base in PeftModel for the first time
        model = PeftModel.from_pretrained(
            base_model, adapter_path, adapter_name=variety,
        )
    model = model.to(device).eval()

    # picking out only the rows where the variety column matches —
    # each adapter was trained for one variety so we evaluate it on its own slice
    sub = test_df[test_df[COLUMN_MAP["variety"]] == variety].copy()
    if len(sub) == 0:
        print(f"[{variety}] WARN: no rows in test CSV with variety='{variety}'. Skipping.")
        return pd.DataFrame(), model

    texts = sub[COLUMN_MAP["text"]].astype(str).tolist()
    y_true = sub[COLUMN_MAP["label"]].astype(int).tolist()

    preds = []
    print(f"[{variety}] Running inference on {len(texts)} examples (batch_size={batch_size})...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length,
        ).to(device)
        # no_grad turns off autograd — we don't need gradients for inference
        # and turning them off saves real time and memory
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        # printing every 10 batches so progress is visible without flooding the terminal
        if (i // batch_size) % 10 == 0:
            print(f"  [{variety}] {i+len(batch)}/{len(texts)} done")

    return pd.DataFrame({
        "text": texts,
        "variety": variety,
        "adapter": adapter_path,
        "y_true": y_true,
        "y_pred": preds,
    }), model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-csv", required=True,
        help="Path to BESSTIE test CSV with columns: text, label, variety.",
    )
    parser.add_argument(
        "--adapter-root", required=True,
        help="Folder containing en-UK_seed42/, en-AU_seed42/, en-IN_seed42/ subfolders.",
    )
    parser.add_argument(
        "--output-dir", default="results/predictions/",
        help="Where to save the 3 prediction CSVs.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    # importing torch / transformers / peft inside main so the script can be
    # inspected (--help) without these heavy libraries installed
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading test CSV: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    for col_role, col_name in COLUMN_MAP.items():
        if col_name not in test_df.columns:
            sys.exit(
                f"ERROR: test CSV missing column '{col_name}' (role={col_role}). "
                f"Found columns: {test_df.columns.tolist()}. "
                f"Edit COLUMN_MAP at top of this script if your CSV uses different names."
            )
    print(f"Test set size: {len(test_df)}")
    print(f"Variety counts:\n{test_df[COLUMN_MAP['variety']].value_counts()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading base model {BASE_MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        # Qwen ships without a pad token — eos works as a safe stand-in
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=2,
    )
    # tell the model which token id is padding (Qwen tokenizer adds pad token via eos)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    # adapters were trained against a resized embed table — match it before LoRA wraps
    base_model.resize_token_embeddings(len(tokenizer))

    # reusing one base model across all 3 adapters — Qwen-1.5B is ~3 GB
    # to load, so doing it once instead of three times saves real time
    model = base_model
    for variety, subfolder in VARIETIES.items():
        adapter_path = os.path.join(args.adapter_root, subfolder)
        if not os.path.exists(adapter_path):
            print(f"[{variety}] SKIP: {adapter_path} not found.")
            continue
        df, model = run_inference_for_variety(
            model, tokenizer, adapter_path, variety,
            test_df, device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        if df.empty:
            continue
        out_path = os.path.join(
            args.output_dir,
            f"lora_{variety.lower().replace('-', '_')}_sarcasm_predictions.csv",
        )
        df.to_csv(out_path, index=False)
        print(f"[{variety}] Saved {len(df)} predictions to {out_path}")

    print("\n[done] All available LoRA prediction CSVs generated.")
    print("Next step: feed each CSV into src/evaluation.evaluate_predictions().")


if __name__ == "__main__":
    main()
