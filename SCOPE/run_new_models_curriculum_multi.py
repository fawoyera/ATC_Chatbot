#!/usr/bin/env python3
"""
run_new_models_curriculum.py
==============================
Llama-3.1-8B and Qwen3-8B runner for SCOPE conditions with CURRICULUM LEARNING.
Mirrors run_new_models_general.py exactly, but:
  1. Calls scope_train_curriculum.py instead of scope_train_general.py
  2. Appends --curriculum to every training condition
  3. Results saved under <output_root>/<model>/curriculum/ subdirectory

All model configs, condition flags, and GCD logic are inherited unchanged
from run_new_models_general.py.  The only addition is the curriculum schedule.

Usage
-----
# All models, key conditions:
python run_new_models_curriculum.py \\
    --models llama qwen \\
    --conditions C2 C3 C11 C4 \\
    --domain atc \\
    --output_root results_curriculum \\
    --curriculum_phase1 0.333 \\
    --curriculum_phase2 0.667 \\
    --curriculum_ramp_steps 0

# Single model, full ablation:
python run_new_models_curriculum.py \\
    --models llama --conditions C1 C2 C3 C5 C6 C7 C8 C9 C10 C11 C4 C4a C4b
"""
Multi-GPU note:
  If 2+ GPUs are visible, scope_train_curriculum.py automatically uses
  device_map="auto" to split model layers across all GPUs and switches
  to 8-bit Adam. No extra flags needed.
  Ensure CUDA_VISIBLE_DEVICES is not set to a single GPU, e.g.:
    export CUDA_VISIBLE_DEVICES=0,1
    python run_new_models_curriculum.py ...
"""


import os, sys, subprocess, time, json, argparse
from pathlib import Path

# ── Model registry (identical to run_new_models_general.py) ──────────────────

MODEL_REGISTRY = {
    "llama": {
        "model_id":  "meta-llama/Llama-3.1-8B-Instruct",
        "shortname": "Llama-3.1-8B",
        "lr":        "2.48e-5",
        "epochs":    "5",
        "batch":     "4",
        "grad_accum":"4",
        "M_samples": "4",
    },
    "qwen": {
        "model_id":  "Qwen/Qwen3-8B",
        "shortname": "Qwen3-8B",
        "lr":        "2.48e-5",
        "epochs":    "5",
        "batch":     "4",
        "grad_accum":"4",
        "M_samples": "4",
    },
}

# ── Condition definitions (identical to run_new_models_general.py) ────────────

CONDITIONS = {
    "C1":  {"label": "Vanilla",             "kind": "train",
            "extra": ["--lambda_ce","1.0","--epochs","0",
                      "--no_ltok","--no_lphr","--no_lcfg"]},
    "C2":  {"label": "Standard SFT",        "kind": "train",
            "extra": ["--lambda_ce","1.0","--no_ltok","--no_lphr","--no_lcfg"]},
    "C3":  {"label": "DPO",                 "kind": "train",
            "extra": ["--lambda_ce","1.0","--dpo","--dpo_beta","0.1",
                      "--no_ltok","--no_lphr","--no_lcfg"]},
    "C5":  {"label": "SCOPE-tok",           "kind": "train",
            "extra": ["--lambda_ce","0.5","--lambda_tok","1.0",
                      "--no_lphr","--no_lcfg"]},
    "C6":  {"label": "SCOPE-phr-REINFORCE", "kind": "train",
            "extra": ["--lambda_ce","0.5","--no_ltok","--lambda_phr","0.5",
                      "--no_lcfg","--M_samples","1"]},
    "C7":  {"label": "SCOPE-phr-GRPO",      "kind": "train",
            "extra": ["--lambda_ce","0.5","--no_ltok","--lambda_phr","0.5",
                      "--no_lcfg","--M_samples","4"]},
    "C8":  {"label": "SCOPE-cfg",           "kind": "train",
            "extra": ["--lambda_ce","0.5","--no_ltok","--no_lphr",
                      "--lambda_cfg","1.43","--M_samples","4"]},
    "C9":  {"label": "SCOPE-2L",            "kind": "train",
            "extra": ["--lambda_ce","0.5","--lambda_tok","0.97",
                      "--lambda_phr","0.90","--no_lcfg","--M_samples","4"]},
    "C10": {"label": "SCOPE-REINFORCE",     "kind": "train",
            "extra": ["--lambda_ce","0.5","--lambda_tok","0.97",
                      "--lambda_phr","0.90","--lambda_cfg","1.43",
                      "--M_samples","1"]},
    "C11": {"label": "SCOPE-full (proposed)","kind": "train",
            "extra": ["--lambda_ce","0.5","--lambda_tok","0.97",
                      "--lambda_phr","0.90","--lambda_cfg","1.43",
                      "--M_samples","4","--gradnorm","--gradnorm_alpha","0.12"]},
    "C4":  {"label": "GCD on SFT",          "kind": "gcd",  "ckpt_source": "C2"},
    "C4a": {"label": "GCD on Vanilla",      "kind": "gcd",  "ckpt_source": "C1"},
    "C4b": {"label": "SCOPE+GCD",           "kind": "gcd",  "ckpt_source": "C11"},
}


def _run(cmd, log_path, dry_run=False):
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    if dry_run:
        print("  [DRY RUN]"); return True
    t0 = time.time()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stderr=subprocess.STDOUT, text=True, stdout=lf)
    print(f"  {'✓' if proc.returncode==0 else '✗'} "
          f"{'Complete' if proc.returncode==0 else 'FAILED'} "
          f"in {(time.time()-t0)/60:.1f} min")
    return proc.returncode == 0


def run_model(model_key, args):
    reg        = MODEL_REGISTRY[model_key]
    model_id   = reg["model_id"]
    shortname  = reg["shortname"]
    output_dir = Path(args.output_root) / model_key

    # Curriculum flags injected into every training condition
    CL = [
        "--curriculum",
        "--curriculum_phase1",     str(args.curriculum_phase1),
        "--curriculum_phase2",     str(args.curriculum_phase2),
        "--curriculum_ramp_steps", str(args.curriculum_ramp_steps),
    ]

    BASE = [
        "--model",       model_id,
        "--data",        args.data,
        "--test_data",   args.test_data,
        "--grammar",     args.grammar,
        "--vocab_path",  args.vocab_path,
        "--phrase_path", args.phrase_path,
        "--domain",      args.domain,
        "--lr",          reg["lr"],
        "--epochs",      reg["epochs"],
        "--batch_size",  reg["batch"],
        "--grad_accum",  reg["grad_accum"],
        "--max_new_tok", str(args.max_new_tok),
        "--seed",        str(args.seed),
        "--use_chat_template",
        "--gradient_checkpointing",
        "--early_stop_patience", "2",
        "--warmup_ratio", "0.1",
        "--use_8bit_adam",              # required for A100 40GB: reduces Adam states ~4x
    ]

    requested = set(args.conditions)
    print(f"\n{'='*70}\n  MODEL: {shortname} | CURRICULUM LEARNING\n{'='*70}")

    for cond_id in args.conditions:
        if cond_id not in CONDITIONS:
            print(f"  ⚠  Unknown condition {cond_id} — skipping"); continue

        c        = CONDITIONS[cond_id]
        out_dir  = output_dir / cond_id
        done_f   = out_dir / "DONE"
        print(f"\n  {'─'*60}")
        print(f"  {cond_id}: {c['label']} [{shortname}] [CURRICULUM]")

        if done_f.exists() and not args.force:
            print(f"  ✓ Already done — skipping"); continue

        out_dir.mkdir(parents=True, exist_ok=True)

        if c["kind"] == "train":
            # Append curriculum flags — skip CL for C1 (0 epochs, no training)
            cl_flags = [] if cond_id == "C1" else CL
            cmd = ([sys.executable, args.script]
                   + BASE + c["extra"] + cl_flags
                   + ["--output", str(out_dir)])
            ok = _run(cmd, out_dir / "training.log", args.dry_run)

        elif c["kind"] == "gcd":
            ckpt_path = output_dir / c["ckpt_source"] / "best"
            if not ckpt_path.exists():
                print(f"  ⚠  Checkpoint {ckpt_path} not found — skipping"); continue
            cmd = [
                sys.executable, args.gcd_script,
                "--model",       str(ckpt_path),
                "--test_data",   args.test_data,
                "--grammar",     args.grammar,
                "--vocab_path",  args.vocab_path,
                "--phrase_path", args.phrase_path,
                "--output",      str(out_dir),
                "--domain",      args.domain,
                "--max_new_tok", str(args.max_new_tok),
            ]
            ok = _run(cmd, out_dir / "gcd.log", args.dry_run)

        if ok and not args.dry_run:
            done_f.touch()


def parse_args():
    p = argparse.ArgumentParser(description="SCOPE curriculum runner — Llama/Qwen")
    p.add_argument("--models",      nargs="+", default=["llama","qwen"],
                   choices=list(MODEL_REGISTRY))
    p.add_argument("--conditions",  nargs="+",
                   default=["C2","C3","C11","C4"])
    p.add_argument("--data",        required=True)
    p.add_argument("--test_data",   required=True)
    p.add_argument("--grammar",     required=True)
    p.add_argument("--vocab_path",  required=True)
    p.add_argument("--phrase_path", required=True)
    p.add_argument("--domain",      default="atc")
    p.add_argument("--output_root", default="results_llama_qwen_curriculum")
    p.add_argument("--script",      default="scope_train_curriculum.py")
    p.add_argument("--gcd_script",  default="evaluate_gcd_general.py")
    p.add_argument("--max_new_tok", type=int,   default=80)
    p.add_argument("--seed",        type=int,   default=42)
    # Curriculum
    p.add_argument("--curriculum_phase1",     type=float, default=1/3)
    p.add_argument("--curriculum_phase2",     type=float, default=2/3)
    p.add_argument("--curriculum_ramp_steps", type=int,   default=0)
    p.add_argument("--force",    action="store_true")
    p.add_argument("--dry_run",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for m in args.models:
        run_model(m, args)
