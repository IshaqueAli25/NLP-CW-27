"""
PG27 NLP Coursework — Q5.1 Deployment App
==========================================
Author:   Anil Emanuel
Module:   COMM061 — Natural Language Processing
Purpose:  Gradio web app demonstrating the trained sarcasm + sentiment
          classifiers on the BESSTIE dataset across three English varieties
          (en-UK, en-AU, en-IN).

Architecture
------------
SARCASM side  -> Qwen2.5-1.5B-Instruct base  +  LoRA adapter swap
                 (one trained adapter per variety, hot-swapped via PEFT)
SENTIMENT side -> Reshabh's RoBERTa-style fine-tuned model (single model,
                  variety dropdown is informational here — model was trained
                  on all three).

The app launches with text input + task dropdown (sarcasm/sentiment) +
variety dropdown (UK/AU/IN). When the user changes variety on the
sarcasm side, model.set_adapter() swaps the LoRA weights — the cheap
operation hinted at in the Q5.1 spec.

If model files cannot be loaded (e.g. running this file standalone for the
report screenshots without the full Drive folder mounted), a clear placeholder
banner appears and the predict button returns a stub. So the file is always
runnable for grading.

USAGE
-----
    python app.py
    -> http://127.0.0.1:7860
"""

import os
import time
import warnings

# Qwen2.5-1.5B-Instruct is what Priyanka's adapter_config.json points at —
# any other base would refuse to load these adapters
# BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# matching what Priyanka did at training time — resize the embedding
# to the tokenizer's vocab size so the LoRA adapters line up.
# done inside load_models() right after the base model loads.

# local paths to the LoRA adapters from Priyanka's Drive folder.
# update these to match wherever the Drive folder lives on your machine

# bootstrap — adapters live on HF Hub, downloaded once to a local cache
# folder so the existing loader (which expects local folders) runs unchanged.
# first run pulls ~1.9 GB. after that the snapshot is cached and skipped.
from huggingface_hub import snapshot_download

_HF_ADAPTERS = {
    "en-UK": "pc01093/pg27-en-UK-lora",
    "en-AU": "pc01093/pg27-en-AU-lora",
}
_ADAPTERS_HF_DIR = "./adapters_hf"
os.makedirs(_ADAPTERS_HF_DIR, exist_ok=True)

for _variety, _repo in _HF_ADAPTERS.items():
    _target = os.path.join(_ADAPTERS_HF_DIR, _variety)
    _weights = os.path.join(_target, "adapter_model.safetensors")
    if not os.path.exists(_weights):
        print(f"[bootstrap] downloading {_repo} -> {_target}")
        snapshot_download(
            repo_id=_repo,
            local_dir=_target,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
        )
    else:
        print(f"[bootstrap] {_variety} cached at {_target} — skipping")

LORA_ADAPTER_PATHS = {
    "en-UK": "./adapters_hf/en-UK",
    "en-AU": "./adapters_hf/en-AU",
    "en-IN": "./adapters/en-IN_seed42",
}

# blocked on Reshabh — once he pushes the RoBERTa to HF Hub, set this
# env var (or hardcode the model id below) and the sentiment side wakes up
SENTIMENT_MODEL_HF = os.environ.get("SENTIMENT_MODEL_HF", "")

SARCASM_CLASS_NAMES = ["non-sarcastic", "sarcastic"]
SENTIMENT_CLASS_NAMES = ["negative", "positive"]

class ModelBundle:
    """Container for whichever models we managed to load."""
    def __init__(self):
        self.sarcasm_ok = False
        self.sentiment_ok = False
        self.sarcasm_model = None
        self.sarcasm_tokenizer = None
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.active_variety = None
        # keeping a load log so the UI can surface why a model failed to load
        # without me having to dig through the terminal
        self.load_log = []

    def log(self, msg):
        self.load_log.append(msg)
        print(msg)

def load_models():
    """Try to load both heads. Catch errors so the app still launches."""
    bundle = ModelBundle()

    # sarcasm side — Qwen base + 3 LoRA adapters via PEFT
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
        )
        from peft import PeftModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle.log(f"[sarcasm] Loading Qwen base on {device}...")

        bundle.sarcasm_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if bundle.sarcasm_tokenizer.pad_token is None:
            # Qwen has no pad token by default — falling back to eos avoids a
            # tokenizer crash when batching
            bundle.sarcasm_tokenizer.pad_token = bundle.sarcasm_tokenizer.eos_token

        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=len(SARCASM_CLASS_NAMES),
        )
        # matching what Priyanka did at training time — resize the embedding
        # to the tokenizer's vocab size so the adapter weights line up
        base.resize_token_embeddings(len(bundle.sarcasm_tokenizer))
        # loading the first adapter via PeftModel.from_pretrained, then any
        # extra adapters via load_adapter — that's the API for multi-adapter
        first_variety, first_path = next(iter(LORA_ADAPTER_PATHS.items()))
        if not os.path.exists(first_path):
            raise FileNotFoundError(
                f"LoRA adapter folder not found: {first_path}. "
                "Update LORA_ADAPTER_PATHS to match your Drive folder."
            )

        model = PeftModel.from_pretrained(
            base, first_path, adapter_name=first_variety,
        )
        bundle.log(f"[sarcasm] Loaded adapter '{first_variety}' from {first_path}")

        # peft 0.19 model.load_adapter runs a Hub-repo-id validator on the
        # path before checking if it is local — that breaks on every path
        # format we tried. workaround: register the adapter manually using
        # LoraConfig.from_pretrained (which reads local folders cleanly),
        # then load weights with the same primitives load_adapter uses.
        from peft import LoraConfig
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        for variety, path in LORA_ADAPTER_PATHS.items():
            if variety == first_variety:
                continue
            weights_file = os.path.join(path, "adapter_model.safetensors")
            if os.path.exists(weights_file):
                cfg = LoraConfig.from_pretrained(path)
                cfg.inference_mode = True
                model.add_adapter(variety, cfg)
                weights = load_peft_weights(path, device=device)
                set_peft_model_state_dict(model, weights, adapter_name=variety)
                bundle.log(f"[sarcasm] Loaded adapter '{variety}' from {path}")
            else:
                # not fatal — the app still works for the varieties we did
                # find, just warning so it's obvious in the load log
                bundle.log(f"[sarcasm] WARN: missing adapter weights for '{variety}' (no adapter_model.safetensors at {path})")

        model = model.to(device).eval()
        model.set_adapter(first_variety)

        bundle.sarcasm_model = model
        bundle.active_variety = first_variety
        bundle.sarcasm_ok = True
        bundle.log("[sarcasm] OK — adapter swap ready.")

    except Exception as e:
        # catching everything on purpose — if any part of the load stack breaks,
        # the UI should still come up with a clear stub message
        bundle.log(f"[sarcasm] FAILED: {type(e).__name__}: {e}")

    # sentiment side — single fine-tuned model from HF Hub (Reshabh's)
    if SENTIMENT_MODEL_HF:
        try:
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
            )
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            bundle.log(f"[sentiment] Loading {SENTIMENT_MODEL_HF} on {device}...")
            bundle.sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_HF)
            bundle.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                SENTIMENT_MODEL_HF
            ).to(device).eval()
            bundle.sentiment_ok = True
            bundle.log("[sentiment] OK")
        except Exception as e:
            bundle.log(f"[sentiment] FAILED: {type(e).__name__}: {e}")
    else:
        bundle.log("[sentiment] SKIPPED — SENTIMENT_MODEL_HF not set "
                   "(set this env var to Reshabh's HF identifier when ready).")

    return bundle

def _softmax_predict(model, tokenizer, text):
    """Run forward, return (predicted_class_idx, probability_dict)."""
    import torch
    # picking the device off the model directly so we don't have to thread
    # it through every call site
    device = next(model.parameters()).device
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0]
    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
    pred_idx = int(probs.argmax())
    return pred_idx, probs

def predict_sarcasm(text, variety, bundle):
    """Predict sarcasm using the LoRA adapter matching the chosen variety."""
    if not bundle.sarcasm_ok:
        # graceful stub — the UI still works even when nothing loaded
        return {
            "label": "[stub] sarcasm model not loaded",
            "confidence": 0.0,
            "elapsed_ms": 0.0,
            "active_adapter": "none",
            "log": " | ".join(bundle.load_log),
        }

    if variety not in LORA_ADAPTER_PATHS:
        return {"label": f"unknown variety: {variety}",
                "confidence": 0.0, "elapsed_ms": 0.0,
                "active_adapter": "none", "log": ""}
    # configured but weights missing — happens when an adapter folder
    # exists but adapter_model.safetensors does not (e.g. en-IN at the
    # moment because Priyanka has not pushed those weights yet)
    if variety not in bundle.sarcasm_model.peft_config:
        return {"label": f"[unavailable] adapter for {variety} not loaded",
                "confidence": 0.0, "elapsed_ms": 0.0,
                "active_adapter": bundle.active_variety or "none",
                "log": "weights file not found at startup"}

    t0 = time.perf_counter()

    # the Q5.1 key operation — only swap the adapter if the user picked a
    # different variety. set_adapter is cheap but not free, so skip it
    # when nothing changed
    if bundle.active_variety != variety:
        bundle.sarcasm_model.set_adapter(variety)
        bundle.active_variety = variety

    pred_idx, probs = _softmax_predict(
        bundle.sarcasm_model, bundle.sarcasm_tokenizer, text,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "label": SARCASM_CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {SARCASM_CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))},
        "elapsed_ms": round(elapsed_ms, 1),
        "active_adapter": variety,
    }

def predict_sentiment(text, variety, bundle):
    """Predict sentiment. Variety is informational — model was trained on all three."""
    if not bundle.sentiment_ok:
        return {
            "label": "[stub] sentiment model not loaded "
                     "(set SENTIMENT_MODEL_HF env var to Reshabh's HF model id)",
            "confidence": 0.0,
            "elapsed_ms": 0.0,
            "log": " | ".join(bundle.load_log),
        }
    t0 = time.perf_counter()
    pred_idx, probs = _softmax_predict(
        bundle.sentiment_model, bundle.sentiment_tokenizer, text,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "label": SENTIMENT_CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {SENTIMENT_CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))},
        "elapsed_ms": round(elapsed_ms, 1),
        "context_variety": variety,
    }

def build_interface(bundle):
    import gradio as gr

    def _predict(text, task, variety):
        if not text or not text.strip():
            return "Please enter some text.", "", ""

        if task == "Sarcasm":
            r = predict_sarcasm(text, variety, bundle)
        else:
            r = predict_sentiment(text, variety, bundle)

        prediction_md = (
            f"### Prediction: **{r['label']}**\n"
            f"Confidence: **{r.get('confidence', 0):.2%}**"
        )
        if "probs" in r:
            probs_md = "\n".join(
                [f"- {k}: {v:.4f}" for k, v in r["probs"].items()]
            )
        else:
            probs_md = "_no probability vector available_"
        meta_md = (
            f"- Task: `{task}`\n"
            f"- Variety: `{variety}`\n"
            f"- Active adapter: `{r.get('active_adapter', 'n/a')}`\n"
            f"- Inference latency: **{r.get('elapsed_ms', 0)} ms**"
        )
        return prediction_md, probs_md, meta_md

    # surfacing model status in the UI itself — the marker shouldn't have to
    # peek at the terminal to know whether the heavy weights actually loaded
    status_lines = []
    status_lines.append(f"- Sarcasm head: {'LOADED ✅' if bundle.sarcasm_ok else 'STUB ⚠️'}")
    status_lines.append(f"- Sentiment head: {'LOADED ✅' if bundle.sentiment_ok else 'STUB ⚠️'}")
    if bundle.sarcasm_ok and bundle.sarcasm_model is not None:
        loaded = list(bundle.sarcasm_model.peft_config.keys())
        status_lines.append(f"- Adapters loaded: {', '.join(loaded)}")
    else:
        status_lines.append("- Adapters loaded: none")
    status_md = "\n".join(status_lines)

    with gr.Blocks(title="PG27 — BESSTIE Variety Classifier") as demo:
        gr.Markdown(
            "# PG27 — BESSTIE Variety-Aware Classifier\n"
            "Detects sarcasm or sentiment on text written in three English varieties: "
            "British (UK), Australian (AU), Indian (IN).\n\n"
            "**Sarcasm:** Qwen2.5-1.5B base + LoRA adapter (one per variety, hot-swapped).  \n"
            "**Sentiment:** Single fine-tuned RoBERTa-style model.\n"
        )
        gr.Markdown(f"### Model status\n{status_md}")

        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(
                    label="Input text",
                    placeholder="e.g. Oh wonderful, another Monday meeting.",
                    lines=4,
                )
                task_in = gr.Radio(
                    choices=["Sarcasm", "Sentiment"],
                    value="Sarcasm",
                    label="Task",
                )
                variety_in = gr.Dropdown(
                    choices=list(LORA_ADAPTER_PATHS.keys()),
                    value="en-UK",
                    label="English variety (selects LoRA adapter for sarcasm)",
                )
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                pred_out = gr.Markdown(label="Prediction")
                probs_out = gr.Markdown(label="Class probabilities")
                meta_out = gr.Markdown(label="Diagnostics")

        # one example per variety — saves the marker typing and shows the
        # variety dropdown actually changes behaviour
        examples = [
            ["Oh wonderful, another Monday meeting.", "Sarcasm", "en-UK"],
            ["The film was an absolute beauty, mate.", "Sentiment", "en-AU"],
            ["The food was simply outstanding, very tasty only.", "Sentiment", "en-IN"],
            ["Yeah right, that's exactly what I needed today.", "Sarcasm", "en-IN"],
        ]
        gr.Examples(examples=examples, inputs=[text_in, task_in, variety_in])

        btn.click(_predict, [text_in, task_in, variety_in],
                  [pred_out, probs_out, meta_out])

    return demo

if __name__ == "__main__":
    # silencing the noisy hf_hub / transformers warnings — they clutter the
    # terminal and the marker only cares about the [sarcasm] / [sentiment] log lines
    warnings.filterwarnings("ignore")
    print("=" * 70)
    print("PG27 — Q5.1 Deployment App")
    print("=" * 70)
    bundle = load_models()
    print("=" * 70)
    print("Launching Gradio UI on http://127.0.0.1:7860 ...")
    print("=" * 70)
    try:
        demo = build_interface(bundle)
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except ImportError:
        print("\nERROR: gradio not installed. Run:  pip install gradio")
