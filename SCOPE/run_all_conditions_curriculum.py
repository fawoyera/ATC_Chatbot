#!/usr/bin/env python3
"""
run_all_conditions_curriculum.py
=================================
GPT-2 Large runner for all 13 SCOPE conditions with CURRICULUM LEARNING.
Mirrors run_all_conditions_general.py exactly, but:
  1. Calls scope_train_curriculum.py instead of scope_train_general.py
  2. Appends --curriculum to every training condition
  3. Adds --curriculum_phase1, --curriculum_phase2, --curriculum_ramp_steps
  4. Saves results under <output_root>_curriculum/ to avoid overwriting
     non-curriculum runs

All conditions are identical to run_all_conditions_general.py so that
curriculum vs non-curriculum results are directly comparable.

Usage
-----
python run_all_conditions_curriculum.py \\
    --data        atc_pairs.json \\
    --test_data   atc_test.json \\
    --model       gpt2-large \\
    --domain      atc \\
    --grammar     G_ATC.lark \\
    --vocab_path  vocab_ATC.json \\
    --phrase_path ngram_whitelist_ATC.json \\
    --output_root results_gpt2_curriculum \\
    --epochs 5 --batch_size 16 --M_samples 4 \\
    --curriculum_phase1 0.333 --curriculum_phase2 0.667 \\
    --curriculum_ramp_steps 50

# Run only specific conditions:
    --conditions C2 C5 C9 C11

# Dry run to verify commands before launching:
    --dry_run
"""

import os, sys, json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Helpers (identical to run_all_conditions_general.py) ─────────────────────

def _print_header(cond_id, label, description):
    print(f"\n{'='*70}")
    print(f"  {cond_id}: {label}  [CURRICULUM]")
    print(f"  {description}")
    print(f"{'='*70}")


def _run_subprocess(cmd, log_path, dry_run=False):
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    if dry_run:
        print("  [DRY RUN — not executing]")
        return True
    t0 = time.time()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stderr=subprocess.STDOUT, text=True, stdout=lf)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  ✗ FAILED (exit {proc.returncode}) — see {log_path}")
        return False
    print(f"  ✓ Complete in {elapsed/60:.1f} min")
    return True


# ── Condition definitions ─────────────────────────────────────────────────────

def make_conditions(args):
    """
    Returns (BASE_args, conditions_list).
    Every training condition automatically receives --curriculum plus the
    phase and ramp flags.  GCD conditions are unchanged (inference-time only).
    """
    BASE = [
        "--data",       args.data,
        "--test_data",  args.test_data,
        "--epochs",     str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr",         str(args.lr),
        "--max_new_tok",str(args.max_new_tok),
        "--seed",       str(args.seed),
        "--vocab_path", args.vocab_path,
        "--phrase_path",args.phrase_path,
        "--grammar",    args.grammar,
        "--domain",     args.domain,
    ]
    GRPO = ["--M_samples", str(args.M_samples)]

    # Curriculum flags appended to every training condition
    CL = [
        "--curriculum",
        "--curriculum_phase1",    str(args.curriculum_phase1),
        "--curriculum_phase2",    str(args.curriculum_phase2),
        "--curriculum_ramp_steps", str(args.curriculum_ramp_steps),
    ]

    dpo_ref = getattr(args, "dpo_ref", "")

    conditions = [
        # ── Baselines ─────────────────────────────────────────────────────────
        ("C1", "Vanilla",
         "No fine-tuning — curriculum has no effect (0 epochs)",
         "train",
         ["--model", args.model, "--epochs", "0",
          "--no_ltok", "--no_lphr", "--no_lcfg"]),
          # C1 does not append CL — epochs=0 means no training either way

        ("C2", "SFT",
         "Standard SFT — CE only; curriculum adds no extra losses",
         "train",
         ["--model", args.model, "--lambda_ce", "1.0",
          "--no_ltok", "--no_lphr", "--no_lcfg"] + CL),

        ("C3", "DPO",
         "DPO — curriculum introduces DPO at Phase 2",
         "train",
         ["--model", args.model, "--lambda_ce", "1.0",
          "--no_ltok", "--no_lphr", "--no_lcfg",
          "--dpo", "--dpo_beta", str(getattr(args, "dpo_beta", 0.1))]
         + (["--dpo_ref", dpo_ref] if dpo_ref else []) + CL),

        # ── Ablation: single-level ─────────────────────────────────────────────
        ("C5", "SCOPE-tok",
         "L_tok only — Phase 2 activates L_tok; Phase 3 unchanged",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_tok", "1.0", "--no_lphr", "--no_lcfg"] + CL),

        ("C6", "SCOPE-phr-REINFORCE",
         "L_phr REINFORCE — curriculum: CE→CE+tok→CE+tok+phr",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_phr", "0.5", "--no_ltok", "--no_lcfg",
          "--M_samples", "1"] + CL),

        ("C7", "SCOPE-phr-GRPO",
         "L_phr GRPO — curriculum: CE→CE+tok→CE+tok+phr(GRPO)",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_phr", "0.5", "--no_ltok", "--no_lcfg"] + GRPO + CL),

        ("C8", "SCOPE-cfg",
         "L_cfg only — curriculum: CE→CE+tok→CE+tok+cfg",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_cfg", "0.3", "--no_ltok", "--no_lphr"] + GRPO + CL),

        # ── Ablation: two-level ───────────────────────────────────────────────
        ("C9", "SCOPE-2L",
         "L_tok+L_phr — curriculum: CE→CE+tok→CE+tok+phr (no cfg)",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_tok", "1.0", "--lambda_phr", "0.5",
          "--no_lcfg"] + GRPO + CL),

        ("C10", "SCOPE-REINFORCE",
         "Full SCOPE REINFORCE — curriculum: CE→CE+tok→full(M=1)",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_tok", "1.0", "--lambda_phr", "0.5", "--lambda_cfg", "0.3",
          "--M_samples", "1"] + CL),

        # ── Proposed method ───────────────────────────────────────────────────
        ("C11", "SCOPE-full",
         "Full SCOPE GRPO+GradNorm — curriculum: CE→CE+tok→full",
         "train",
         ["--model", args.model, "--lambda_ce", "0.5",
          "--lambda_tok", "1.0", "--lambda_phr", "0.5", "--lambda_cfg", "0.3",
          ] + GRPO + CL),

        # ── GCD inference-time baselines — unchanged from non-curriculum ──────
        ("C4",  "GCD-SFT",     "GCD on SFT checkpoint",       "gcd", {"ckpt_source": "C2"}),
        ("C4a", "GCD-Vanilla", "GCD on Vanilla checkpoint",   "gcd", {"ckpt_source": "C1"}),
        ("C4b", "SCOPE+GCD",   "GCD on SCOPE-full checkpoint","gcd", {"ckpt_source": "C11"}),
    ]

    return BASE, conditions


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(args):
    output_root  = Path(args.output_root)
    script_path  = Path(args.script)
    gcd_script   = Path(args.gcd_script)

    BASE, conditions = make_conditions(args)

    # Filter to requested conditions
    requested = set(args.conditions) if args.conditions else None
    summary   = []

    for entry in conditions:
        cond_id, label, desc, kind, extra = entry

        if requested and cond_id not in requested:
            continue

        out_dir   = output_root / cond_id
        done_flag = out_dir / "DONE"
        _print_header(cond_id, label, desc)

        if done_flag.exists() and not args.force:
            print(f"  ✓ Already complete — skipping (use --force to re-run)")
            summary.append((cond_id, label, "skipped"))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        if kind == "train":
            cmd = ([sys.executable, str(script_path)]
                   + BASE + ["--output", str(out_dir)] + extra)
            ok  = _run_subprocess(cmd, out_dir / "training.log", args.dry_run)
            if ok and not args.dry_run:
                done_flag.touch()
            summary.append((cond_id, label, "ok" if ok else "FAILED"))

        elif kind == "gcd":
            ckpt_src  = extra["ckpt_source"]
            ckpt_path = output_root / ckpt_src / "best"
            if not ckpt_path.exists():
                print(f"  ⚠  Source checkpoint {ckpt_path} not found — skipping")
                summary.append((cond_id, label, "skipped-no-ckpt"))
                continue
            cmd = [
                sys.executable, str(gcd_script),
                "--model",       str(ckpt_path),
                "--test_data",   args.test_data,
                "--grammar",     args.grammar,
                "--vocab_path",  args.vocab_path,
                "--phrase_path", args.phrase_path,
                "--output",      str(out_dir),
                "--domain",      args.domain,
                "--max_new_tok", str(args.max_new_tok),
            ]
            ok = _run_subprocess(cmd, out_dir / "gcd.log", args.dry_run)
            if ok and not args.dry_run:
                done_flag.touch()
            summary.append((cond_id, label, "ok" if ok else "FAILED"))

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CURRICULUM RUN COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Output: {output_root.resolve()}")
    print(f"{'='*70}")
    for cid, lbl, status in summary:
        icon = "✓" if status in ("ok", "skipped") else "✗"
        print(f"  {icon}  {cid:<6} {lbl:<28} {status}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SCOPE curriculum runner — GPT-2 all conditions")
    # Data
    p.add_argument("--data",         required=True)
    p.add_argument("--test_data",    required=True)
    p.add_argument("--grammar",      required=True)
    p.add_argument("--vocab_path",   required=True)
    p.add_argument("--phrase_path",  required=True)
    p.add_argument("--domain",       default="atc")
    # Model
    p.add_argument("--model",        default="gpt2-large")
    p.add_argument("--output_root",  default="results_gpt2_curriculum")
    p.add_argument("--script",       default="scope_train_curriculum.py")
    p.add_argument("--gcd_script",   default="evaluate_gcd_general.py")
    # Training
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--max_new_tok",  type=int,   default=80)
    p.add_argument("--M_samples",    type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--dpo_beta",     type=float, default=0.1)
    p.add_argument("--dpo_ref",      default="")
    # Curriculum
    p.add_argument("--curriculum_phase1",     type=float, default=1/3,
                   help="Fraction of epochs for Phase 1 (CE only). Default: 1/3")
    p.add_argument("--curriculum_phase2",     type=float, default=2/3,
                   help="Fraction of epochs up to end of Phase 2 (CE+tok). Default: 2/3")
    p.add_argument("--curriculum_ramp_steps", type=int,   default=0,
                   help="Steps to ramp new lambdas at transitions. Default: 0 (hard switch)")
    # Control
    p.add_argument("--conditions",   nargs="*",  default=None,
                   help="Conditions to run. Default: all 13.")
    p.add_argument("--force",        action="store_true",
                   help="Re-run conditions even if DONE flag exists")
    p.add_argument("--dry_run",      action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
