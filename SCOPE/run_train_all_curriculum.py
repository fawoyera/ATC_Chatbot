#!/usr/bin/env python3
"""
run_train_all_curriculum.py
============================
Top-level orchestrator for ALL curriculum learning experiments.
Mirrors run_train_all.py exactly, but dispatches to:
  - run_all_conditions_curriculum.py  (for gpt2)
  - run_new_models_curriculum.py      (for llama, qwen)

Results are saved to <output_root>_curriculum/ by default so that
curriculum and non-curriculum runs coexist without overwriting each other.

After all runs complete, produces a comparison table that shows
curriculum vs non-curriculum results side-by-side for the conditions
in --compare_conditions (default: C2 C11).

Usage
-----
# Full run — all models, all conditions, ATC domain:
python run_train_all_curriculum.py \\
    --domain atc \\
    --models gpt2 llama qwen \\
    --conditions C2 C3 C5 C9 C11 C4

# GPT-2 only, full ablation (the primary curriculum experiment):
python run_train_all_curriculum.py \\
    --domain atc --models gpt2 \\
    --conditions C1 C2 C3 C5 C6 C7 C8 C9 C10 C11 C4 C4a C4b \\
    --curriculum_ramp_steps 50

# Colab path:
python run_train_all_curriculum.py \\
    --scope_dir /content/drive/MyDrive/ATC_Chatbot/SCOPE \\
    --domain atc --models gpt2 --conditions C2 C11

# Dry run:
python run_train_all_curriculum.py --dry_run --models gpt2 --conditions C2 C11
"""

import argparse, subprocess, sys, json, os
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified SCOPE curriculum runner — all models/domains")

    p.add_argument("--scope_dir",   default="/scratch/gilbreth/oawoyera/scope",
                   help="Root directory with all SCOPE scripts and data files. "
                        "For Colab: /content/drive/MyDrive/ATC_Chatbot/SCOPE")
    p.add_argument("--domain",      default="atc", choices=["atc","smcp"])
    p.add_argument("--models",      nargs="+", default=["gpt2","llama","qwen"],
                   choices=["gpt2","llama","qwen"])
    p.add_argument("--conditions",  nargs="+",
                   default=["C2","C3","C11","C4"],
                   help="Conditions to run. Use 'all' for all 13.")
    p.add_argument("--output_root", default=None,
                   help="Override output root. Default: <scope_dir>/results_<domain>_curriculum")
    # Curriculum schedule
    p.add_argument("--curriculum_phase1",     type=float, default=1/3,
                   help="Fraction of epochs for Phase 1 (CE only). Default: 1/3")
    p.add_argument("--curriculum_phase2",     type=float, default=2/3,
                   help="Fraction of epochs through Phase 2 (CE+tok). Default: 2/3")
    p.add_argument("--curriculum_ramp_steps", type=int,   default=0,
                   help="Steps to ramp new lambdas at phase transitions. "
                        "0=hard switch (default). 50-100 recommended for GPT-2.")
    # Comparison
    p.add_argument("--compare_conditions",    nargs="+",  default=["C2","C11"],
                   help="Conditions to include in curriculum vs non-curriculum "
                        "comparison table. Default: C2 C11")
    p.add_argument("--non_curriculum_root",   default=None,
                   help="Path to non-curriculum results for comparison table. "
                        "Default: <scope_dir>/results_<domain>")
    # Control
    p.add_argument("--force",    action="store_true")
    p.add_argument("--dry_run",  action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Domain config (mirrors run_train_all.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_domain_config(args):
    S = Path(args.scope_dir)
    if args.domain == "atc":
        return {
            "data":        str(S / "atc_pairs.json"),
            "test_data":   str(S / "atc_test.json"),
            "grammar":     str(S / "G_ATC.lark"),
            "vocab_path":  str(S / "vocab_ATC.json"),
            "phrase_path": str(S / "ngram_whitelist_ATC.json"),
            "domain":      "atc",
        }
    else:
        return {
            "data":        str(S / "smcp_pairs.json"),
            "test_data":   str(S / "smcp_test.json"),
            "grammar":     str(S / "G_SMCP.lark"),
            "vocab_path":  str(S / "vocab_SMCP.json"),
            "phrase_path": str(S / "ngram_whitelist_SMCP.json"),
            "domain":      "smcp",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Sub-runner dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def _run(cmd, dry_run=False):
    print(f"\n  LAUNCHING: {' '.join(str(c) for c in cmd)}\n")
    if dry_run:
        print("  [DRY RUN — not executing]"); return True
    proc = subprocess.run(cmd, text=True)
    return proc.returncode == 0


def run_gpt2(args, dc, output_root):
    S   = Path(args.scope_dir)
    cmd = [
        sys.executable, str(S / "run_all_conditions_curriculum.py"),
        "--data",                   dc["data"],
        "--test_data",              dc["test_data"],
        "--grammar",                dc["grammar"],
        "--vocab_path",             dc["vocab_path"],
        "--phrase_path",            dc["phrase_path"],
        "--domain",                 dc["domain"],
        "--model",                  "gpt2-large",
        "--output_root",            str(output_root / "gpt"),
        "--script",                 str(S / "scope_train_curriculum.py"),
        "--gcd_script",             str(S / "evaluate_gcd_general.py"),
        "--epochs",                 "5",
        "--batch_size",             "16",
        "--lr",                     "2e-4",
        "--max_new_tok",            "80",
        "--M_samples",              "4",
        "--curriculum_phase1",      str(args.curriculum_phase1),
        "--curriculum_phase2",      str(args.curriculum_phase2),
        "--curriculum_ramp_steps",  str(args.curriculum_ramp_steps),
        "--conditions",             *args.conditions,
    ]
    if args.force:   cmd.append("--force")
    if args.dry_run: cmd.append("--dry_run")
    return _run(cmd, args.dry_run)


def run_llama_qwen(args, dc, output_root, models):
    S   = Path(args.scope_dir)
    cmd = [
        sys.executable, str(S / "run_new_models_curriculum.py"),
        "--models",                 *models,
        "--conditions",             *args.conditions,
        "--data",                   dc["data"],
        "--test_data",              dc["test_data"],
        "--grammar",                dc["grammar"],
        "--vocab_path",             dc["vocab_path"],
        "--phrase_path",            dc["phrase_path"],
        "--domain",                 dc["domain"],
        "--output_root",            str(output_root),
        "--script",                 str(S / "scope_train_curriculum.py"),
        "--gcd_script",             str(S / "evaluate_gcd_general.py"),
        "--max_new_tok",            "80",
        "--curriculum_phase1",      str(args.curriculum_phase1),
        "--curriculum_phase2",      str(args.curriculum_phase2),
        "--curriculum_ramp_steps",  str(args.curriculum_ramp_steps),
    ]
    if args.force:   cmd.append("--force")
    if args.dry_run: cmd.append("--dry_run")
    return _run(cmd, args.dry_run)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Comparison table
# ═══════════════════════════════════════════════════════════════════════════════

METRICS = ["C_tok", "C_phr", "C_cfg", "C_bar",
           "slot_f1", "da_f1", "halluc_pct", "bertscore"]

MODEL_DIRS = {"gpt2": "gpt", "llama": "llama", "qwen": "qwen"}

def _read_result(results_root, model_key, cond_id, metric):
    d = MODEL_DIRS.get(model_key, model_key)
    p = Path(results_root) / d / cond_id / "test_results.json"
    if not p.exists(): return None
    try:
        data = json.loads(p.read_text())
        return data.get(metric, data.get("metrics", {}).get(metric))
    except Exception:
        return None


def print_comparison_table(args, output_root, dc):
    non_cur_root = (Path(args.non_curriculum_root) if args.non_curriculum_root
                    else Path(args.scope_dir) / f"results_{args.domain}")
    cur_root = output_root

    print(f"\n{'='*90}")
    print(f"  CURRICULUM vs NON-CURRICULUM COMPARISON")
    print(f"  Domain: {args.domain.upper()} | Conditions: {args.compare_conditions}")
    print(f"{'='*90}")

    header = f"  {'Model':<8} {'Cond':<5} {'Setting':<14}"
    for m in METRICS:
        header += f" {m:>9}"
    print(header)
    print("  " + "─" * (8 + 5 + 14 + len(METRICS) * 10))

    for model_key in args.models:
        for cond_id in args.compare_conditions:
            for setting, root in [("non-curriculum", non_cur_root),
                                   ("curriculum",     cur_root)]:
                row = f"  {model_key:<8} {cond_id:<5} {setting:<14}"
                for m in METRICS:
                    val = _read_result(root, model_key, cond_id, m)
                    row += f" {val:>9.4f}" if val is not None else f" {'—':>9}"
                print(row)
            print()

    print(f"{'='*90}")
    print(f"  Curriculum: {cur_root}")
    print(f"  Non-curriculum: {non_cur_root}")
    print(f"{'='*90}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    S    = Path(args.scope_dir)
    dc   = build_domain_config(args)

    # Expand "all" shorthand
    if "all" in args.conditions:
        args.conditions = ["C1","C2","C3","C5","C6","C7","C8",
                           "C9","C10","C11","C4","C4a","C4b"]

    # Output root
    output_root = (Path(args.output_root) if args.output_root
                   else S / f"results_{args.domain}_curriculum")
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  SCOPE CURRICULUM EXPERIMENT")
    print(f"  Domain    : {args.domain.upper()}")
    print(f"  Models    : {', '.join(args.models)}")
    print(f"  Conditions: {', '.join(args.conditions)}")
    print(f"  Phase 1   : epochs [0 .. {args.curriculum_phase1:.0%}]  → CE only")
    print(f"  Phase 2   : epochs [{args.curriculum_phase1:.0%} .. {args.curriculum_phase2:.0%}] → CE + L_tok")
    print(f"  Phase 3   : epochs [{args.curriculum_phase2:.0%} .. 100%]  → CE + L_tok + L_phr [+ L_cfg]")
    print(f"  Ramp steps: {args.curriculum_ramp_steps}")
    print(f"  Output    : {output_root}")
    print(f"{'='*70}\n")

    gpt2_models   = [m for m in args.models if m == "gpt2"]
    large_models  = [m for m in args.models if m != "gpt2"]

    if gpt2_models:
        print("\n── GPT-2 Large ──────────────────────────────────────────────")
        run_gpt2(args, dc, output_root)

    if large_models:
        print(f"\n── {', '.join(large_models)} ────────────────────────────────")
        run_llama_qwen(args, dc, output_root, large_models)

    # Print comparison table after runs complete
    if not args.dry_run:
        print_comparison_table(args, output_root, dc)


if __name__ == "__main__":
    main()
