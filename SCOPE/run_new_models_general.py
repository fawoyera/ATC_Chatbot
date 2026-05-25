#!/usr/bin/env python3
"""
run_new_models_general.py — Run SCOPE on Llama-3.1-8B and Qwen2.5-7B
Domain-agnostic: ATC (ICAO) and maritime (SMCP) via --domain.

Conditions per model:
  C2:  SFT baseline
  C3:  DPO
  C4:  GCD on SFT checkpoint
  C11: SCOPE-full (proposed)

Usage:
  python run_new_models_general.py \
    --models llama qwen --conditions C2 C3 C11 C4 \
    --domain smcp \
    --grammar G_SMCP.lark \
    --vocab_path vocab_SMCP.json \
    --phrase_path ngram_whitelist_SMCP.json \
    --data smcp_pairs.json --test_data smcp_test.json \
    --train_script scope_train_general.py \
    --gcd_script evaluate_gcd_general.py \
    --output_root results_maritime

  python run_new_models_general.py \
    --models llama qwen \
    --conditions C2 C3 C11 C4 \
    --domain atc \
    --output_root results_new_models/
"""

import os, sys, subprocess, time, json, argparse
from pathlib import Path

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "llama": {
        "model_id":  "meta-llama/Llama-3.1-8B-Instruct",
        "shortname": "Llama-3.1-8B",
        "lr":        "1e-5",
        "epochs":    "3",
        "batch":     "4",
        "grad_accum":"4",
        "M_samples": "2",
    },
    "qwen2": {
        "model_id":  "Qwen/Qwen2.5-7B-Instruct",
        "shortname": "Qwen2.5-7B",
        "lr":        "1e-5",
        "epochs":    "3",
        "batch":     "4",
        "grad_accum":"4",
        "M_samples": "2",
    },
    "qwen": {
    "model_id":  "Qwen/Qwen3-8B",
    "shortname": "Qwen3-8B",
    "lr":        "1e-5",
    "epochs":    "3",
    "batch":     "4",
    "grad_accum":"4",
    "M_samples": "2",
    },
}

# ── Condition definitions ─────────────────────────────────────────────────────
# All conditions use:
#   lr = 2.48e-05 (Optuna-tuned; dominant hyperparameter at 80% importance)
#   lambda values = Optuna-tuned for whichever losses are active
#   lambda_ce = 1.0025 for compliance conditions (C5–C11)
#   lambda_ce = 1.0 for baselines (C1–C3) — compliance losses are off
# checkpoint_metric, early_stop_patience, warmup_ratio are set in `common`
# and apply to all conditions uniformly.

CONDITIONS = {
    # ── Baselines ─────────────────────────────────────────────────────────
    "C1": {
        "label": "Vanilla",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06", "--lambda_ce", "1.0",
                  "--epochs", "0", "--no_ltok", "--no_lphr", "--no_lcfg"],
    },
    "C2": {
        "label": "Standard SFT",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06", "--lambda_ce", "1.0",
                  "--no_ltok", "--no_lphr", "--no_lcfg"],
    },
    "C3": {
        "label": "DPO",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06", "--lambda_ce", "1.0",
                  "--dpo", "--dpo_beta", "0.1",
                  "--no_ltok", "--no_lphr", "--no_lcfg"],
    },
    # ── Ablation: single-level ─────────────────────────────────────────────
    # Tuned lambda for each active loss; lambda_ce = tuned value (1.0025)
    "C5": {
        "label": "SCOPE-tok",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06",
                  "--lambda_ce", "1.0960", "--lambda_tok", "0.8210",
                  "--no_lphr", "--no_lcfg"],
    },
    "C6": {
        "label": "SCOPE-phr-REINFORCE",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06",
                  "--lambda_ce", "1.0960", "--lambda_phr", "0.5732",
                  "--no_ltok", "--no_lcfg", "--M_samples", "1"],
    },
    "C7": {
        "label": "SCOPE-phr-GRPO",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06",
                  "--lambda_ce", "1.0960", "--lambda_phr", "0.5732",
                  "--no_ltok", "--no_lcfg", "--M_samples", "4"],
    },
    "C8": {
        "label": "SCOPE-cfg",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06",
                  "--lambda_ce", "1.0960", "--lambda_cfg", "1.8532",
                  "--no_ltok", "--no_lphr", "--M_samples", "4"],
    },
    # ── Ablation: two-level ────────────────────────────────────────────────
    "C9": {
        "label": "SCOPE-2L",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06",
                  "--lambda_ce", "1.0960", "--lambda_tok", "0.8210",
                  "--lambda_phr", "0.5732", "--no_lcfg", "--M_samples", "4"],
    },
    "C10": {
        "label": "SCOPE-REINFORCE",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06",
                  "--lambda_ce", "1.0960", "--lambda_tok", "0.8210",
                  "--lambda_phr", "0.5732", "--lambda_cfg", "1.8532",
                  "--M_samples", "1"],
    },
    # ── Proposed method (single C11, tuned weights) ───────────────────────
    "C11": {
        "label": "SCOPE-full (tuned)",
        "kind":  "train",
        "extra": ["--lr", "6.35e-06",
                  "--lambda_ce", "1.0960", "--lambda_tok", "0.8210",
                  "--lambda_phr", "0.5732", "--lambda_cfg", "1.8532",
                  "--gradnorm", "--M_samples", "4"],
    },
    # ── GCD inference-time baselines ──────────────────────────────────────
    "C4": {
        "label": "GCD on SFT",
        "kind":  "gcd",
        "source": "C2",
    },
    "C4a": {
        "label": "GCD Vanilla",
        "kind":  "gcd",
        "source": "C1",
    },
    "C4b": {
        "label": "SCOPE+GCD",
        "kind":  "gcd",
        "source": "C11",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def hf_login():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN not set. Add it to Colab Secrets.")
        sys.exit(1)
    try:
        from huggingface_hub import login
        login(token=token)
        print("✓ HuggingFace authenticated")
    except Exception as e:
        print(f"WARNING: HF login issue: {e} — proceeding")

def run_subprocess(cmd, log_path, label):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        proc.wait()
    elapsed = (time.time() - t0) / 60
    ok = proc.returncode == 0
    print(f"\n  {'✓ Complete' if ok else '✗ FAILED'} ({elapsed:.1f} min)")
    return ok

def load_metrics(results_path):
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None

def print_metrics(cond_id, label, r):
    if not r:
        return
    print(f"    {cond_id} {label}:")
    print(f"      C_tok={r.get('C_tok',0):.4f}  C_phr={r.get('C_phr',0):.4f}  "
          f"C_cfg={r.get('C_cfg',0):.4f}  C_bar={r.get('C_bar',0):.4f}")
    if any(k in r for k in ('slot_f1','da_f1','halluc_pct','bertscore')):
        print(f"      SlotF1={r.get('slot_f1',0):.4f}  DA-F1={r.get('da_f1',0):.4f}  "
              f"Hall%={r.get('halluc_pct',0):.3f}  BERT={r.get('bertscore',0):.4f}")

# ── Run one condition for one model ──────────────────────────────────────────
def run_condition(cond_id, model_key, model_info, args, out_root):
    cond      = CONDITIONS[cond_id]
    model_dir = out_root / model_key
    cond_dir  = model_dir / cond_id
    done_flag = cond_dir / "DONE"
    label     = f"{model_info['shortname']} / {cond['label']}"
    py        = sys.executable

    if done_flag.exists():
        print(f"\n  ✓ {label} — already complete, skipping")
        return load_metrics(cond_dir / "test_results.json")

    if cond["kind"] == "gcd":
        # GCD: evaluate_gcd_general.py on the SFT checkpoint
        src_ckpt = model_dir / cond["source"] / "best"
        if not src_ckpt.exists():
            print(f"\n  ✗ {label} — source checkpoint {src_ckpt} not found")
            print(f"    Run {cond['source']} first.")
            return None
        cmd = [
            py, args.gcd_script,
            "--model",           str(src_ckpt),
            "--data",            args.test_data,
            "--grammar",         args.grammar,
            "--output",          str(cond_dir),
            "--vocab",           args.vocab_path,
            "--phrase",          args.phrase_path,
            "--domain",          args.domain,
            "--bertscore_model", args.bertscore_model,
        ]

    else:  # train condition
        # Determine model-specific flags
        is_instruct = model_key in ("llama", "qwen", "qwen2")
        model_flags = []
        if is_instruct:
            model_flags += ["--use_chat_template", "--gradient_checkpointing"]

        cond_extra = cond.get("extra", [])
        # Allow CLI overrides of model registry defaults
        eff_epochs     = args.epochs     if args.epochs     else model_info["epochs"]
        eff_batch      = args.batch_size if args.batch_size else model_info["batch"]
        eff_accum      = args.grad_accum if args.grad_accum else model_info["grad_accum"]
        eff_M          = args.M_samples  if args.M_samples  else model_info["M_samples"]
        eff_seed       = args.seed       if args.seed       else "42"
        # Base lr: CLI override > model registry; per-condition --lr in extra wins
        eff_lr         = args.lr         if args.lr         else model_info["lr"]
        has_lr_override = "--lr" in cond_extra

        common = [
            "--model",      model_info["model_id"],
            "--train_data", args.train_data if args.train_data else "",
            "--val_data",   args.val_data   if args.val_data   else "",
            "--data",       args.data,
            "--test_data",  args.test_data,
            "--output",     str(cond_dir),
            "--epochs",     eff_epochs,
            "--batch_size", eff_batch,
            "--grad_accum", eff_accum,
            "--M_samples",  eff_M,
            "--max_new_tok","64",
            "--seed",       eff_seed,
            "--vocab_path", args.vocab_path,
            "--phrase_path",args.phrase_path,
            "--grammar",    args.grammar,
            "--domain",     args.domain,
            "--checkpoint_metric",   "c_bar",
            "--early_stop_patience", "2",
            "--warmup_ratio",        "0.1",
            "--bertscore_model",     args.bertscore_model,
        ] + model_flags + (
            [] if has_lr_override else ["--lr", eff_lr]
        ) + (
            ["--finetune_bert"] if getattr(args, "finetune_bert", False) else []
        )
        cmd = [py, args.train_script] + common + cond_extra

    ok = run_subprocess(cmd, cond_dir / "run.log", label)
    if ok:
        done_flag.touch()   # mark complete so --resume skips this
    r  = load_metrics(cond_dir / "test_results.json") if ok else None
    print_metrics(cond_id, cond["label"], r)
    return r

# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(all_results, gpt2_results=None):
    W = 100
    print(f"\n\n{'='*W}")
    print(f"{'RESULTS — ATC':^{W}}")
    print(f"{'='*W}")

    # Header — two rows to keep line width manageable
    print(f"  {'Model':<14} {'Condition':<22} "
          f"{'C_tok':>6} {'C_phr':>6} {'C_cfg':>6} {'C_bar':>6} "
          f"{'SlotF1':>7} {'DA-F1':>7} {'Hall%':>6} {'BERT':>6}")
    print("-" * W)

    def _row(shortname, cid, label, r, marker=""):
        if r is None:
            return
        cbar = r.get('C_bar', (r.get('C_tok',0)+r.get('C_phr',0)+r.get('C_cfg',0))/3)
        has_sem = any(k in r for k in ('slot_f1','da_f1','halluc_pct','bertscore'))
        sem = (f"{r.get('slot_f1',0):>7.4f} {r.get('da_f1',0):>7.4f} "
               f"{r.get('halluc_pct',0):>6.3f} {r.get('bertscore',0):>6.4f}"
               if has_sem else
               f"{'—':>7} {'—':>7} {'—':>6} {'—':>6}")
        print(f"  {shortname:<14} {label:<22} "
              f"{r.get('C_tok',0):>6.4f} {r.get('C_phr',0):>6.4f} "
              f"{r.get('C_cfg',0):>6.4f} {cbar:>6.4f} "
              f"{sem}{marker}")

    # GPT-2 reference rows
    if gpt2_results:
        for cid, r in gpt2_results.items():
            label = CONDITIONS.get(cid, {}).get("label", cid)[:22]
            _row("GPT-2 Large", cid, label, r)
        print()

    for model_key, cond_results in all_results.items():
        shortname = MODEL_REGISTRY[model_key]["shortname"]
        for cid, r in cond_results.items():
            if r is None:
                continue
            label = CONDITIONS.get(cid, {}).get("label", cid)[:22]
            marker = " ◀" if cid == "C11" else ""
            _row(shortname, cid, label, r, marker)
        print()

    print("=" * W)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",       nargs="+", default=["llama", "qwen"],
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--conditions",   nargs="+", default=["C2","C3","C11","C4"])
    parser.add_argument("--output_root",  default="results_new_models")
    parser.add_argument("--data",         default="atc_pairs.json")
    parser.add_argument("--test_data",    default="atc_test.json")
    # [CHANGE: PRE-SPLIT] fixed split files for reproducible ablation
    parser.add_argument("--train_data",   default="",
                        help="Pre-split train file (use with --val_data). "
                             "Takes priority over --data when set.")
    parser.add_argument("--val_data",     default="",
                        help="Pre-split validation file.")
    parser.add_argument("--bertscore_model", default="bert-base-uncased",
                        help="BERTScore encoder — use domain fine-tuned path after first run")
    parser.add_argument("--finetune_bert", action="store_true",
                        help="Fine-tune BERT on domain corpus for BERTScore (cached after first run)")
    parser.add_argument("--train_script", default="scope_train_general.py")
    parser.add_argument("--gcd_script",   default="evaluate_gcd_general.py")
    parser.add_argument("--grammar",      default="G_ATC.lark")
    parser.add_argument("--vocab_path",   default="vocab_ATC.json")
    parser.add_argument("--phrase_path",  default="ngram_whitelist_ATC.json")
    parser.add_argument("--domain",       default="atc", choices=["atc", "smcp"])
    parser.add_argument("--gpt2_results", default="results2")
    # Training args — allow override from run_train_all.py
    # These are used to override MODEL_REGISTRY defaults when passed explicitly.
    # If not passed, model_info values from the registry are used.
    parser.add_argument("--epochs",      default=None,
                        help="Override epochs from model registry")
    parser.add_argument("--batch_size",  default=None,
                        help="Override batch size from model registry")
    parser.add_argument("--grad_accum",  default=None,
                        help="Override grad_accum from model registry")
    parser.add_argument("--lr",          default=None,
                        help="Override base lr (per-condition lr still takes priority)")
    parser.add_argument("--M_samples",   default=None,
                        help="Override M_samples from model registry")
    parser.add_argument("--seed",        default="42")
    args = parser.parse_args()

    hf_login()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load GPT-2 reference results if available
    gpt2_results = {}
    gpt2_dir = Path(args.gpt2_results)
    for cid in args.conditions:
        p = gpt2_dir / cid / "test_results.json"
        if p.exists():
            with open(p) as f:
                gpt2_results[cid] = json.load(f)

    all_results = {}
    for model_key in args.models:
        model_info = MODEL_REGISTRY[model_key]
        print(f"\n\n{'#'*72}")
        print(f"# MODEL: {model_info['shortname']} ({model_info['model_id']})")
        print(f"{'#'*72}")

        cond_results = {}
        # Ensure C2 runs before C4 (GCD depends on C2 checkpoint)
        ordered_conditions = sorted(
            args.conditions,
            key=lambda c: 99 if c == "C4" else 0
        )
        for cond_id in ordered_conditions:
            # GCD must come after C2 — reorder if needed
            if cond_id == "C4" and "C2" not in cond_results:
                print(f"  Skipping C4 (GCD) — C2 not yet complete for {model_key}")
                continue
            r = run_condition(cond_id, model_key, model_info, args, out_root)
            cond_results[cond_id] = r

        # Run C4 after C2 if C2 just completed
        if "C4" in args.conditions and "C4" not in cond_results:
            r = run_condition("C4", model_key, model_info, args, out_root)
            cond_results["C4"] = r

        all_results[model_key] = cond_results

    print_summary(all_results, gpt2_results if gpt2_results else None)

    # Save combined results JSON
    summary_path = out_root / "multi_model_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            k: {cid: r for cid, r in v.items() if r}
            for k, v in all_results.items()
        }, f, indent=2)
    print(f"\nSaved: {summary_path}")

if __name__ == "__main__":
    main()
