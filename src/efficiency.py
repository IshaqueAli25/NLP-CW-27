"""
PG27 NLP Coursework — Q5.2 Efficiency Analysis Utility
======================================================
Author:   Anil Emanuel
Module:   COMM061 — Natural Language Processing
Purpose:  Inference-time measurement code for Q5.2 (5 marks).
          Times sub-second operations carefully (warm-up + repeated runs +
          median statistic) so reported numbers aren't dominated by
          one-off CUDA/cache effects.

What this measures (all required by the Q5.2 spec):
    1. Base model alone vs base + LoRA adapter  -> shows adapter overhead
    2. Adapter swap vs full model reload         -> shows the LoRA advantage
    3. CPU vs GPU                                -> standard speed/accuracy trade-off
    4. Single input vs batch of N                -> throughput discussion

Output:
    - Pretty-printed per-condition table (mean / median / std / p95)
    - CSV at results/q5_efficiency/timing_results.csv
    - Bar chart at results/q5_efficiency/timing_chart.png

USAGE EXAMPLE
-------------
    from src.efficiency import (
        time_callable, run_full_efficiency_suite,
    )
    run_full_efficiency_suite(
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        lora_paths={
            "UK": "lora_adapters/en-UK_seed42",
            "AU": "lora_adapters/en-AU_seed42",
            "IN": "lora_adapters/en-IN_seed42",
        },
        sample_texts=["This is great.", "Oh wonderful, another Monday."],
        output_dir="results/q5_efficiency/",
    )

The heavy functions (those that actually load Qwen) import torch / transformers
/ peft lazily — so importing this module on a machine without those libraries
does NOT crash.
"""

import os
import time
import statistics
import pandas as pd
import matplotlib.pyplot as plt


def time_callable(fn, *args, n_warmup=3, n_runs=10, **kwargs):
    """
    Time a function `fn(*args, **kwargs)` carefully.

    Strategy:
      - run n_warmup throwaway calls first (load caches, JIT, CUDA kernels)
      - then do n_runs timed calls
      - return mean / median / std / p95 in milliseconds
    """
    # throwing the first few results away — first call is always slow because of
    # cache warm-up, kernel compile, lazy imports etc., not the actual work
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        # perf_counter is the right clock for timing — monotonic, high resolution
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times_sorted = sorted(times)
    # using median + p95 over mean — a single outlier (GC pause, OS scheduling)
    # can lift the mean by 20% while median barely moves
    p95_idx = max(0, int(round(0.95 * (len(times) - 1))))
    return {
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "std_ms": round(statistics.stdev(times) if len(times) > 1 else 0.0, 3),
        "p95_ms": round(times_sorted[p95_idx], 3),
        "n_runs": n_runs,
        "all_ms": [round(t, 3) for t in times],
    }


def format_timing_row(label, timing_dict):
    """Pretty-print one row of timings."""
    return (
        f"{label:<40} "
        f"mean={timing_dict['mean_ms']:>8.2f}ms  "
        f"median={timing_dict['median_ms']:>8.2f}ms  "
        f"std={timing_dict['std_ms']:>7.2f}  "
        f"p95={timing_dict['p95_ms']:>8.2f}ms"
    )


def save_timing_results(results_dict, output_dir, filename="timing_results.csv"):
    """
    Save the dict-of-dicts results to a tidy CSV.

    results_dict shape:
        {"base_model_cpu_single": {"mean_ms": ..., "median_ms": ...}, ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for label, t in results_dict.items():
        rows.append({
            "condition": label,
            "mean_ms": t["mean_ms"],
            "median_ms": t["median_ms"],
            "std_ms": t["std_ms"],
            "p95_ms": t["p95_ms"],
            "n_runs": t["n_runs"],
        })
    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, filename)
    df.to_csv(out, index=False)
    print(f"[saved] {out}")
    return df


def plot_timing_results(results_dict, output_dir, filename="timing_chart.png"):
    """Bar chart of median inference times per condition."""
    os.makedirs(output_dir, exist_ok=True)
    labels = list(results_dict.keys())
    medians = [results_dict[k]["median_ms"] for k in labels]

    # sizing the chart to the number of bars — fixed height clips labels
    # when there are many conditions
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(labels))))
    ax.barh(labels, medians, color="#4C72B0")
    ax.set_xlabel("Median inference time (ms)")
    ax.set_title("Q5.2 — Inference Time per Condition")
    ax.invert_yaxis()
    for i, v in enumerate(medians):
        ax.text(v, i, f" {v:.1f} ms", va="center")
    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


def make_inference_callable(model, tokenizer, device, batch=False):
    """
    Build a no-arg callable that runs forward inference once.
    Used as the `fn` argument to time_callable.
    """
    import torch

    def _call(texts):
        # no_grad turns off autograd bookkeeping — at inference we don't need
        # gradients and skipping them shaves a real chunk of time
        with torch.no_grad():
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            _ = model(**enc)
            # CUDA work is async — without sync, t1 fires before the kernel
            # actually finishes and we'd record fake-fast numbers
            if device == "cuda":
                torch.cuda.synchronize()
    return _call


def run_full_efficiency_suite(
    base_model_name,
    lora_paths,
    sample_texts,
    output_dir="results/q5_efficiency/",
    n_warmup=3,
    n_runs=10,
    batch_size=32,
    use_gpu_if_available=True,
):
    """
    The full Q5.2 measurement suite.

    Conditions measured (max ~10):
      1. base, CPU, single  input
      2. base, CPU, batch input
      3. base + LoRA-UK, CPU, single
      4. base + LoRA-UK, CPU, batch
      5. adapter swap (UK -> AU)        — just set_adapter()
      6. full reload (load AU adapter from disk again)
      7-10. same as 1-4 on GPU if available
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    os.makedirs(output_dir, exist_ok=True)

    # padding the input list so a "batch_size=32" condition really has 32 items
    while len(sample_texts) < batch_size:
        sample_texts = sample_texts + sample_texts
    single_text = [sample_texts[0]]
    batch_texts = sample_texts[:batch_size]

    devices = ["cpu"]
    if use_gpu_if_available and torch.cuda.is_available():
        devices.append("cuda")

    print(f"Devices to be timed: {devices}")
    print(f"n_warmup={n_warmup}  n_runs={n_runs}  batch_size={batch_size}")
    print("=" * 80)

    results = {}
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        # Qwen ships without a pad token — falling back to eos so tokenizer
        # padding doesn't crash
        tokenizer.pad_token = tokenizer.eos_token

    for device in devices:
        print(f"\n--- Loading base model on {device} ---")
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
        ).to(device).eval()
        # tell the model which token id is padding so batched inference
        # has a mask to apply. Qwen2 doesn't ship one, so eos doubles up
        base.config.pad_token_id = tokenizer.pad_token_id
        # tell the model which token id is padding so batched inference
        # has a mask to apply. Qwen2 doesn't ship one, so eos doubles up
        base.config.pad_token_id = tokenizer.pad_token_id
        # adapters were trained against a resized embed table — match it
        # before LoRA load or set_peft_model_state_dict crashes on shape
        base.resize_token_embeddings(len(tokenizer))

        # base alone — single input timing
        fn = make_inference_callable(base, tokenizer, device)
        results[f"base_{device}_single"] = time_callable(
            fn, single_text, n_warmup=n_warmup, n_runs=n_runs
        )
        # base alone — batched input timing
        results[f"base_{device}_batch{batch_size}"] = time_callable(
            fn, batch_texts, n_warmup=n_warmup, n_runs=n_runs
        )

        # loading all 3 LoRA adapters into the same base — the whole point
        # of Q5.1 is hot-swapping between them with set_adapter()
        first_var, first_path = next(iter(lora_paths.items()))
        model = PeftModel.from_pretrained(base, first_path, adapter_name=first_var)

        # peft 0.19 model.load_adapter runs a Hub-repo-id validator on the
        # path before checking if it is local — same bug app.py hit in
        # Session 7. workaround: register the adapter manually using
        # LoraConfig.from_pretrained, then load weights with the same
        # primitives load_adapter would use internally
        from peft import LoraConfig
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        for var, path in lora_paths.items():
            if var == first_var:
                continue
            cfg = LoraConfig.from_pretrained(path)
            cfg.inference_mode = True
            model.add_adapter(var, cfg)
            weights = load_peft_weights(path, device=device)
            set_peft_model_state_dict(model, weights, adapter_name=var)
            print(f"  loaded adapter {var} from {path}")
        model = model.to(device).eval()
        model.set_adapter(first_var)
        # PeftModel sometimes loses config link to the inner module —
        # setting pad_token_id on the wrapper keeps batched forward safe
        model.config.pad_token_id = tokenizer.pad_token_id

        # base + LoRA timings — first variety active
        model.set_adapter(first_var)
        fn_lora = make_inference_callable(model, tokenizer, device)
        results[f"base+LoRA-{first_var}_{device}_single"] = time_callable(
            fn_lora, single_text, n_warmup=n_warmup, n_runs=n_runs
        )
        results[f"base+LoRA-{first_var}_{device}_batch{batch_size}"] = time_callable(
            fn_lora, batch_texts, n_warmup=n_warmup, n_runs=n_runs
        )

        var_list = list(lora_paths.keys())
        if len(var_list) >= 2:
            second_var = var_list[1]

            # adapter swap — both already in memory, just point at a different one.
            # this is the cheap operation hinted at in the Q5.1 spec
            def _swap():
                model.set_adapter(second_var)
                model.set_adapter(first_var)

            results[f"adapter_swap_{device}"] = time_callable(
                _swap, n_warmup=n_warmup, n_runs=n_runs
            )

            # full reload — delete the adapter and load it from disk again.
            # the comparison number that makes the swap look fast
            second_path = lora_paths[second_var]

            def _reload():
                if second_var in model.peft_config:
                    model.delete_adapter(second_var)
                cfg2 = LoraConfig.from_pretrained(second_path)
                cfg2.inference_mode = True
                model.add_adapter(second_var, cfg2)
                weights2 = load_peft_weights(second_path, device=device)
                set_peft_model_state_dict(model, weights2, adapter_name=second_var)
                model.set_adapter(first_var)

            # fewer runs for reload — it's I/O heavy, no need to repeat 10x
            results[f"adapter_full_reload_{device}"] = time_callable(
                _reload, n_warmup=1, n_runs=3
            )

        # freeing GPU memory before the next device or the second pass would OOM
        del model, base
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("RESULTS (median is the headline number for the report)")
    print("=" * 80)
    for label, t in results.items():
        print(format_timing_row(label, t))

    df = save_timing_results(results, output_dir)
    plot_timing_results(results, output_dir)
    print("\n[done] All Q5.2 outputs saved to:", output_dir)
    return df


if __name__ == "__main__":
    # smoke test — just the timing primitives, no model load. proves the
    # warm-up + repeated runs + p95 logic all work without needing torch
    print("Smoke test — timing primitives only (no model load)")

    def fake_inference(x):
        # cheap CPU work that takes a few hundred microseconds —
        # enough for the timer to register a real number
        s = 0
        for i in range(20000):
            s += i
        return s

    timing = time_callable(fake_inference, 1, n_warmup=2, n_runs=5)
    print(format_timing_row("fake_inference", timing))

    fake_results = {
        "base_cpu_single": time_callable(fake_inference, 1, n_warmup=2, n_runs=5),
        "base_cpu_batch":  time_callable(fake_inference, 1, n_warmup=2, n_runs=5),
        "base+LoRA_cpu_single": time_callable(fake_inference, 1, n_warmup=2, n_runs=5),
    }
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        save_timing_results(fake_results, tmp)
        plot_timing_results(fake_results, tmp)
    print("Smoke test complete.")
