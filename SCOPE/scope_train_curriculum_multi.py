#!/usr/bin/env python3
"""
SCOPE Training Pipeline — scope_train_curriculum.py
=====================================================
Extends scope_train_general.py with CURRICULUM LEARNING:
losses are introduced progressively across training epochs
rather than all at once from epoch 1.

Three-level objective (same as scope_train_general.py):
  L = lambda_ce * L_CE
    + lambda_tok * L_tok      ← introduced at Phase 2
    + lambda_phr * L_phr(GRPO) ← introduced at Phase 3
    + lambda_cfg * L_cfg(GRPO) ← introduced at Phase 3

Curriculum phases (controlled by --curriculum flag):
  Phase 1  [epochs 1 .. ceil(N/3)]       : L_CE only  (vocabulary grounding)
  Phase 2  [epochs ceil(N/3)+1 .. 2N/3]  : L_CE + L_tok  (lexical compliance)
  Phase 3  [epochs 2N/3+1 .. N]          : L_CE + L_tok + L_phr [+ L_cfg]
                                           (phrase + structure, per condition flags)

Lambda ramping (--curriculum_ramp_steps K):
  Optional soft ramp: at phase transitions, new loss weights are ramped
  linearly from 0 to target over K steps (default 0 = hard switch).
  Hard switching produces a cleaner ablation signal; soft ramping is more
  stable for GPT-2 where hard switches cause loss spikes.

Backward compatibility:
  --curriculum is False by default → behaves identically to scope_train_general.py.
  All other flags and hyperparameters are unchanged.

CHANGES FROM scope_train_general.py (marked # [CHANGE: CURRICULUM:<tag>]):
──────────────────────────────────────────────────────────────────────────
CL1. CURRICULUM_CONFIG
     New CurriculumConfig dataclass: phase_fractions, ramp_steps, schedule.

CL2. CURRICULUM_PHASE
     curriculum_phase(epoch, total_epochs, cfg) returns active loss flags and
     effective lambda values for each epoch. Replaces static cfg.use_l* flags
     during training.

CL3. PHASE_LOGGING
     Each epoch header logs the active phase name and which losses are live.

CL4. LAMBDA_RAMP
     Optional per-step linear warmup of new lambda values at each phase
     transition. Ramp counter resets at each phase boundary.

CL5. HISTORY_EXTENSION
     training_history.json records the active phase for each epoch so the
     curriculum schedule is fully reproducible from the log alone.

All other CHANGES from scope_train_general.py are inherited unchanged:
COSINE_SCHEDULE, CBAR_CHECKPOINT, EARLY_STOPPING, DPO_COMPOSITE_SIGNAL,
GRADNORM, WARMUP_RATIO, SEMANTIC_METRICS.

CHANGES FROM ORIGINAL (all marked with # [CHANGE: <tag>]):
──────────────────────────────────────────────────────────
1. COSINE_SCHEDULE
   get_cosine_schedule_with_warmup replaces get_linear_schedule_with_warmup.
   Training log showed metrics degrading in epochs 2–3; cosine decay is more
   conservative in the tail and preserves early compliance gains.

2. CBAR_CHECKPOINT
   Best checkpoint is now saved on C_bar = (C_tok + C_phr + C_cfg) / 3 rather
   than C_tok alone. The original C_tok criterion ignored grammar and phrase
   improvements; C_bar checkpoint selection matches the Optuna probe objective.
   The old C_tok-based best is still tracked and logged for comparison.

3. EARLY_STOPPING
   Training stops when C_bar fails to improve for `early_stop_patience` epochs
   (default 2). Prevents the epoch 2–3 overfitting observed in the C11 log.

4. DPO_COMPOSITE_SIGNAL
   DPO preference pairs are now built from composite C_bar = (C_tok+C_phr+C_cfg)/3
   instead of C_tok alone. The original C_tok-only signal inadvertently degraded
   C_phr by −0.036 relative to SFT (observed in C3 results).

5. GRADNORM
   Optional GradNorm (Chen et al. 2018) dynamically rebalances lambda values
   throughout training so no single loss dominates by gradient magnitude.
   Enabled with --gradnorm flag. Does not change behaviour when disabled.

6. WARMUP_RATIO
   warmup_steps is now also settable as a ratio of total steps via
   --warmup_ratio. This makes warmup portable across different dataset sizes
   without manual recalculation. Explicit --warmup_steps still overrides.

7. SEMANTIC_METRICS  [CHANGE: SEMANTIC_METRICS]
   Four additional evaluation metrics are computed at validation and test time.
   They do NOT enter the training objective — they only influence checkpoint
   selection via the hallucination gate and are reported in training_history.json.

   a) Slot-F1: extracts (callsign, action, altitude, frequency) from both
      reference and generated response; computes token-level F1 per slot type
      then averages. Answers "did the model say the right thing?".

   b) DA-F1: predicts the dialogue act of each generated response using a
      keyword-based classifier (clearance / readback / hold / advisory /
      handoff / correction / other) and computes F1 against the reference
      dialogue act. No external model required.

   c) Hallucination%: flags a response if it contains a callsign, flight
      level, or VHF frequency that was not present in the input or reference.
      Reported as fraction of examples with at least one hallucinated entity.
      A hallucination gate in checkpoint selection rejects checkpoints where
      Hallucination% exceeds cfg.hallucination_threshold (default 0.10).

   d) BERTScore: semantic similarity between generated and reference response
      using contextual embeddings. Uses the model specified in
      cfg.bertscore_model (default: "bert-base-uncased"; set to a
      domain-fine-tuned checkpoint for best results). Requires bert-score
      package: pip install bert-score. Gracefully disabled if not installed.

   Influence on hyperparameter tuning:
   - Slot-F1, DA-F1, BERTScore: NOT in Optuna objective. The CE floor
     constraint already protects semantic correctness. These are post-hoc
     evaluation metrics reported in the paper.
   - Hallucination%: gates final checkpoint selection only. A checkpoint
     that improves C_bar but exceeds hallucination_threshold is rejected.
   - BERTScore weight: optionally added to the composite checkpoint metric
     via cfg.bertscore_weight (default 0.0 = disabled). Setting to 0.2
     makes the checkpoint criterion 0.8*C_bar + 0.2*BERTScore.
"""

import json, re, math, random, argparse, os, statistics
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                           get_cosine_schedule_with_warmup)   # [CHANGE: COSINE_SCHEDULE]

try:
    from lark import Lark, exceptions as lark_exc
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False
    print("WARNING: lark not installed. L_cfg will be disabled.")

# [CHANGE: SEMANTIC_METRICS] optional bert-score — gracefully disabled if absent
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("WARNING: bert-score not installed. BERTScore will be 0.0. "
          "Install with: pip install bert-score")

# ═════════════════════════════════════════════════════════════════════════════
# 1. REGULATORY ARTEFACTS  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

VOCAB_PATH   = Path("vocab_ATC.json")
PHRASE_PATH  = Path("ngram_whitelist_ATC.json")
GRAMMAR_PATH = Path("G_ATC.lark")
# ═════════════════════════════════════════════════════════════════════════════
# [CHANGE: MULTI_GPU] Device resolution and model loading helpers
# ═════════════════════════════════════════════════════════════════════════════

def _n_gpus() -> int:
    """Number of visible CUDA GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _device_map(force_single: bool = False) -> Optional[str]:
    """
    Return device_map for from_pretrained():
      - 2+ GPUs → "auto"  (split layers across all visible GPUs)
      - 1 GPU   → None    (load to single GPU via .to(device))
      - 0 GPUs  → None    (CPU)
    """
    if not force_single and _n_gpus() >= 2:
        return "auto"
    return None


def _tensor_device() -> torch.device:
    """
    Primary device for tensor creation (.to(), torch.tensor(..., device=...)).
    With device_map="auto" the model spans multiple GPUs; tensors that feed
    the model's first layer must go to cuda:0 (the embedding layer's device).
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_model(model_name: str, dtype=torch.bfloat16,
                gradient_checkpointing: bool = False) -> "AutoModelForCausalLM":
    """
    Load a causal LM, splitting across all available GPUs automatically.
    With 2× A100 40GB: half the layers go to cuda:0, half to cuda:1.
    With 1 GPU or CPU: loads normally.
    """
    dm = _device_map()
    kw = dict(torch_dtype=dtype)
    if dm:
        kw["device_map"] = dm
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if dm is None:
        # Single GPU or CPU: move explicitly
        model = model.to(_tensor_device())
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    n = _n_gpus()
    print(f"  Model loaded: {'split across ' + str(n) + ' GPUs' if dm else 'single device'}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# [CHANGE: CURRICULUM:CL1] CurriculumConfig and curriculum_phase()
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumConfig:
    """
    Controls how losses are introduced across training epochs.

    phase_fractions : (f1, f2) where
        Phase 1 = epochs [1 .. ceil(f1 * total_epochs)]    → L_CE only
        Phase 2 = epochs [ceil(f1)+1 .. ceil(f2 * total)]  → L_CE + L_tok
        Phase 3 = remaining epochs                          → L_CE + L_tok + L_phr [+L_cfg]
    ramp_steps : int
        Number of gradient steps at the start of each phase over which a newly
        activated loss lambda is linearly ramped from 0 to its target value.
        0 = hard switch (default).
    """
    phase_fractions: Tuple[float, float] = (1/3, 2/3)
    ramp_steps:      int                 = 0


def curriculum_phase(
    epoch:        int,        # 0-indexed
    total_epochs: int,
    cfg_use_ltok: bool,
    cfg_use_lphr: bool,
    cfg_use_lcfg: bool,
    cfg_lam_tok:  float,
    cfg_lam_phr:  float,
    cfg_lam_cfg:  float,
    cur:          CurriculumConfig,
) -> Tuple[str, bool, bool, bool, float, float, float]:
    """
    Return (phase_name, use_ltok, use_lphr, use_lcfg, lam_tok, lam_phr, lam_cfg)
    for the given epoch index under curriculum scheduling.

    The curriculum only gates losses that are enabled in the condition config:
    - A loss disabled by the condition (e.g. cfg_use_lphr=False for C5) is
      never activated, regardless of the phase.
    - A loss enabled by the condition is activated according to the phase schedule.

    Phase 1: L_CE only  (vocabulary grounding via CE)
    Phase 2: L_CE + L_tok  (add lexical compliance signal)
    Phase 3: L_CE + L_tok + L_phr [+ L_cfg per condition]  (phrase + structure)
    """
    f1, f2    = cur.phase_fractions
    end_p1    = max(1, math.ceil(f1 * total_epochs))
    end_p2    = max(end_p1 + 1, math.ceil(f2 * total_epochs))

    if epoch < end_p1:
        # Phase 1: L_CE only — suppress all compliance losses regardless of condition
        return ("Phase-1:CE", False, False, False, 0.0, 0.0, 0.0)

    elif epoch < end_p2:
        # Phase 2: L_CE + L_tok (if enabled by condition)
        use_tok = cfg_use_ltok
        lam_tok = cfg_lam_tok if use_tok else 0.0
        return ("Phase-2:CE+tok", use_tok, False, False, lam_tok, 0.0, 0.0)

    else:
        # Phase 3: full condition (all losses enabled by condition are active)
        return (
            "Phase-3:full",
            cfg_use_ltok, cfg_use_lphr, cfg_use_lcfg,
            cfg_lam_tok if cfg_use_ltok else 0.0,
            cfg_lam_phr if cfg_use_lphr else 0.0,
            cfg_lam_cfg if cfg_use_lcfg else 0.0,
        )

def load_whitelist(vocab_path: Path) -> set:
    with open(vocab_path) as f:
        vocab = json.load(f)
    return set(v.upper() for v in vocab)

def load_ngram_whitelist(phrase_path: Path) -> Dict[int, set]:
    with open(phrase_path) as f:
        raw = json.load(f)
    return {
        2: set(tuple(g) for g in raw.get('bigrams',  [])),
        3: set(tuple(g) for g in raw.get('trigrams', [])),
        4: set(tuple(g) for g in raw.get('4grams',   [])),
    }

def load_grammar(grammar_path: Path):
    if not LARK_AVAILABLE:
        return None
    with open(grammar_path) as f:
        grammar_str = f.read()
    return Lark(grammar_str, parser='earley', ambiguity='resolve')

FSM_STATE_VOCAB = {
    'Init':              None,
    'AwaitingClearance': {
        'CLEARED','DESCEND','CLIMB','MAINTAIN','SQUAWK','CONTACT',
        'HOLD','EXPECT','UNABLE','STANDBY','TRAFFIC'
    },
    'ClearanceIssued': {
        'WILCO','AFFIRM','ROGER','UNABLE','SAY','AGAIN','READBACK'
    },
    'AwaitingReadback': {
        'AFFIRM','NEGATIVE','CORRECTION','SAY','AGAIN','READBACK'
    },
    'ReadbackReceived': None,
}

# ═════════════════════════════════════════════════════════════════════════════
# 2. DATASET  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

DOMAIN_PROMPTS = {
    "atc":  "You are an ATC communication assistant for UAV operations. "
            "Generate ICAO-compliant phraseology.",
    "smcp": "You are a maritime radio communication assistant. "
            "Generate IMO SMCP-compliant phraseology.",
}

def format_atc(request: str, response: str, fsm_state: str = 'Init',
               tokenizer=None, use_chat_template: bool = False,
               domain: str = "atc") -> dict:
    domain_desc = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["atc"])
    if domain == "atc":
        system = f"[STATE: {fsm_state}] {domain_desc}"
    else:
        system = domain_desc
    if use_chat_template and tokenizer is not None:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": request}]
        instruction = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full = instruction + response
        return {"instruction": instruction, "full": full, "response": response}
    instruction = f"### Instruction:\n{system}\n\n### Operator:\n{request}\n\n### Response:"
    full = instruction + " " + response
    return {
        "instruction": instruction,
        "response":    response,
        "full":        full,
        "fsm_state":   fsm_state,
    }

class AtcDataset(Dataset):
    def __init__(self, pairs: List[dict], tokenizer, max_length: int = 512,
                 domain: str = "atc"):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []
        for p in pairs:
            item = format_atc(p["request"], p["response"],
                               p.get("fsm_state", "Init"), domain=domain)
            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tok  = self.tokenizer(
            item["full"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        instr_ids = self.tokenizer(
            item["instruction"],
            max_length=self.max_length,
            truncation=True,
        )["input_ids"]
        instr_len = len(instr_ids)
        n_pad     = int((attention_mask == 0).sum().item())

        response_mask = torch.zeros_like(input_ids)
        resp_start    = n_pad + instr_len
        response_mask[resp_start:] = 1
        response_mask = response_mask * attention_mask

        labels = input_ids.clone()
        labels[:resp_start] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "response_mask":  response_mask,
            "instr_len":      torch.tensor(instr_len, dtype=torch.long),
            "n_pad":          torch.tensor(n_pad,     dtype=torch.long),
            "fsm_state":      item["fsm_state"],
            "response_text":  item["response"],
            "instruction":    item["instruction"],
        }

# ═════════════════════════════════════════════════════════════════════════════
# 3. COMPLIANCE METRICS  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ctok(tokens: List[str], vocab: set) -> float:
    if not tokens: return 0.0
    return sum(1 for t in tokens if t.upper() in vocab) / len(tokens)

def compute_cphr(tokens: List[str], ngram_whitelist: Dict[int, set]) -> float:
    if len(tokens) < 2: return 0.0
    scores = []
    toks_upper = [t.upper() for t in tokens]
    for n in [2, 3, 4]:
        if len(toks_upper) < n: continue
        count = 0
        total = len(toks_upper) - n + 1
        wl = ngram_whitelist[n]
        for i in range(total):
            if tuple(toks_upper[i:i+n]) in wl:
                count += 1
        scores.append(count / total)
    return sum(scores) / len(scores) if scores else 0.0

def compute_ccfg_partial(text: str, parser) -> float:
    if parser is None: return 0.0
    text_upper = text.upper().strip()
    if not text_upper: return 0.0
    try:
        parser.parse(text_upper)
        return 1.0
    except Exception:
        pass
    words = text_upper.split()
    if not words: return 0.0
    for n in range(len(words), 0, -1):
        prefix = ' '.join(words[:n])
        try:
            parser.parse(prefix)
            return n / len(words)
        except Exception:
            continue
    return 0.0

# ═════════════════════════════════════════════════════════════════════════════
# 4. SCOPE LOSSES  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def compute_L_tok(logits: torch.Tensor,
                  response_mask: torch.Tensor,
                  vocab_ids: set,
                  vocab_size: int) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    B, T, V = probs.shape
    device   = logits.device
    in_vocab = torch.zeros(V, dtype=torch.bool, device=device)
    for vid in vocab_ids:
        if vid < V:
            in_vocab[vid] = True
    out_vocab_mask = ~in_vocab
    out_mass = (probs * out_vocab_mask.float()).sum(dim=-1)
    m     = response_mask.float()
    denom = m.sum().clamp(min=1.0)
    return (out_mass * m).sum() / denom

def compute_L_phr_grpo(model, tokenizer, batch_inputs: dict,
                        ngram_whitelist: Dict[int, set],
                        M: int = 4, max_new: int = 64):
    device         = next(model.parameters()).device
    input_ids      = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]
    B = input_ids.size(0)

    model.eval()
    all_generated, all_rewards = [], []
    with torch.no_grad():
        for _ in range(M):
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = out[:, input_ids.size(1):]
            all_generated.append(gen)
            rewards = []
            for i in range(B):
                text = tokenizer.decode(gen[i].tolist(), skip_special_tokens=True)
                r = compute_cphr(text.upper().split(), ngram_whitelist)
                rewards.append(r)
            all_rewards.append(rewards)
    model.train()

    rewards_t  = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    mu         = rewards_t.mean(dim=0, keepdim=True)
    sigma      = rewards_t.std(dim=0, keepdim=True).clamp(min=1e-8)
    advantages = (rewards_t - mu) / sigma

    loss_terms = []
    for m_idx, gen_m in enumerate(all_generated):
        if gen_m.size(1) == 0: continue
        full   = torch.cat([input_ids, gen_m], dim=1)
        attn   = torch.ones(B, full.size(1)-1, dtype=torch.long, device=device)
        logits = model(full[:, :-1], attention_mask=attn).logits
        logits = logits[:, input_ids.size(1)-1:, :]
        log_p  = F.log_softmax(logits, dim=-1)
        idx_t  = gen_m.unsqueeze(-1).clamp(0, log_p.size(-1)-1)
        seq_lp = log_p.gather(-1, idx_t).squeeze(-1).mean(dim=-1)
        adv    = advantages[m_idx]
        loss_terms.append(-(adv * seq_lp).mean() / M)

    loss = sum(loss_terms) if loss_terms else torch.tensor(0.0, device=device)
    return loss, all_generated

def compute_L_cfg_grpo(model, tokenizer, batch_inputs: dict,
                        cfg_parser, all_generated: list,
                        M: int = 4) -> torch.Tensor:
    if cfg_parser is None or not all_generated:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    device    = next(model.parameters()).device
    input_ids = batch_inputs["input_ids"]
    B = input_ids.size(0)

    all_rewards = []
    for gen in all_generated:
        rewards = []
        for i in range(B):
            text = tokenizer.decode(gen[i].tolist(), skip_special_tokens=True)
            rewards.append(compute_ccfg_partial(text, cfg_parser))
        all_rewards.append(rewards)

    rewards_t  = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    mu         = rewards_t.mean(dim=0, keepdim=True)
    sigma      = rewards_t.std(dim=0, keepdim=True).clamp(min=1e-8)
    advantages = (rewards_t - mu) / sigma

    loss_terms = []
    for m_idx, gen_m in enumerate(all_generated):
        if gen_m.size(1) == 0: continue
        full   = torch.cat([input_ids, gen_m], dim=1)
        attn   = torch.ones(B, full.size(1)-1, dtype=torch.long, device=device)
        logits = model(full[:, :-1], attention_mask=attn).logits
        logits = logits[:, input_ids.size(1)-1:, :]
        log_p  = F.log_softmax(logits, dim=-1)
        idx_t  = gen_m.unsqueeze(-1).clamp(0, log_p.size(-1)-1)
        seq_lp = log_p.gather(-1, idx_t).squeeze(-1).mean(dim=-1)
        adv    = advantages[m_idx]
        loss_terms.append(-(adv * seq_lp).mean() / M)

    return sum(loss_terms) if loss_terms else torch.tensor(0.0, device=device)

# ═════════════════════════════════════════════════════════════════════════════
# 5. DPO
# ═════════════════════════════════════════════════════════════════════════════

def build_dpo_pairs(pairs: List[dict], tokenizer, model, vocab: set,
                    ngram_wl: Dict[int, set], max_new: int,
                    device: torch.device, domain: str = "atc",
                    cfg_parser=None) -> List[dict]:
    """
    Build DPO preference pairs.

    [CHANGE: DPO_COMPOSITE_SIGNAL]
    Original used C_tok alone as the preference signal, which was observed to
    degrade C_phr by −0.036 (C3 vs C2 results). We now score with C_bar =
    (C_tok + C_phr + C_cfg) / 3 so all three compliance axes contribute to
    the chosen/rejected distinction. C_cfg is included when cfg_parser is
    available, otherwise falls back to (C_tok + C_phr) / 2.
    """
    print("  Building DPO preference pairs from composite C_bar signal ...")  # [CHANGE: DPO_COMPOSITE_SIGNAL]
    model.eval()
    dpo_pairs = []

    with torch.no_grad():
        for p in pairs:
            item = format_atc(p["request"], p["response"], domain=domain)
            tok  = tokenizer(
                item["instruction"],
                return_tensors="pt",
                max_length=256,
                truncation=True,
            ).to(device)

            gens = []
            for _ in range(2):
                out = model.generate(
                    **tok,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=1.2,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
                gen_ids = out[0, tok["input_ids"].size(1):]
                text    = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                toks    = text.upper().split()

                # [CHANGE: DPO_COMPOSITE_SIGNAL] — composite score instead of C_tok only
                c_tok = compute_ctok(toks, vocab)
                c_phr = compute_cphr(toks, ngram_wl)
                c_cfg = compute_ccfg_partial(text, cfg_parser) if cfg_parser else None
                if c_cfg is not None:
                    score = (c_tok + c_phr + c_cfg) / 3.0
                else:
                    score = (c_tok + c_phr) / 2.0
                # [END CHANGE]

                gens.append((text, score))

            gens.sort(key=lambda x: x[1], reverse=True)
            chosen_text,   chosen_score   = gens[0]
            rejected_text, rejected_score = gens[1]

            if abs(chosen_score - rejected_score) < 1e-6:
                chosen_text = p["response"]

            dpo_pairs.append({
                "instruction": item["instruction"],
                "chosen":      chosen_text,
                "rejected":    rejected_text,
            })

    model.train()
    print(f"  Built {len(dpo_pairs)} DPO preference pairs")
    return dpo_pairs


def compute_L_dpo(model, ref_model, tokenizer, dpo_batch: dict,
                  beta: float, device: torch.device) -> torch.Tensor:
    def seq_logprob(mdl, input_ids, attn_mask, resp_start):
        out    = mdl(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, :-1, :]
        tgts   = input_ids[:, 1:]
        log_p  = F.log_softmax(logits, dim=-1)
        tok_lp = log_p.gather(-1, tgts.unsqueeze(-1)).squeeze(-1)
        mask   = torch.zeros_like(tok_lp)
        for i, rs in enumerate(resp_start):
            mask[i, rs-1:] = 1.0
        denom  = mask.sum(dim=-1).clamp(min=1)
        return (tok_lp * mask).sum(dim=-1) / denom

    chosen_ids    = dpo_batch["chosen_ids"].to(device)
    chosen_mask   = dpo_batch["chosen_mask"].to(device)
    rejected_ids  = dpo_batch["rejected_ids"].to(device)
    rejected_mask = dpo_batch["rejected_mask"].to(device)
    resp_start    = dpo_batch["resp_start"]

    lp_w_policy = seq_logprob(model, chosen_ids,   chosen_mask,  resp_start)
    lp_l_policy = seq_logprob(model, rejected_ids, rejected_mask, resp_start)

    # [CHANGE: MULTI_GPU] ref_model is GPU-resident (split across devices if needed)
    # No .to(device)/.cpu() transfers — reference stays on GPU permanently
    with torch.no_grad():
        lp_w_ref = seq_logprob(ref_model, chosen_ids,   chosen_mask,  resp_start)
        lp_l_ref = seq_logprob(ref_model, rejected_ids, rejected_mask, resp_start)

    log_ratio_w = lp_w_policy - lp_w_ref
    log_ratio_l = lp_l_policy - lp_l_ref
    return -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()


class DPODataset(Dataset):
    def __init__(self, dpo_pairs: List[dict], tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = dpo_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        def encode(text):
            tok = self.tokenizer(
                text, max_length=self.max_length,
                truncation=True, padding="max_length", return_tensors="pt",
            )
            return tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0)

        chosen_ids,   chosen_mask   = encode(item["instruction"] + " " + item["chosen"])
        rejected_ids, rejected_mask = encode(item["instruction"] + " " + item["rejected"])

        instr_ids  = self.tokenizer(item["instruction"],
                                    max_length=self.max_length, truncation=True)["input_ids"]
        n_pad      = int((chosen_mask == 0).sum().item())
        resp_start = n_pad + len(instr_ids)

        return {
            "chosen_ids":    chosen_ids,
            "chosen_mask":   chosen_mask,
            "rejected_ids":  rejected_ids,
            "rejected_mask": rejected_mask,
            "resp_start":    resp_start,
        }

# ═════════════════════════════════════════════════════════════════════════════
# 5b. VOCABULARY ID MAPPING  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def build_vocab_ids(tokenizer, vocab: set) -> set:
    vocab_ids   = set()
    all_tokens  = tokenizer.get_vocab()
    vocab_upper = {w.upper() for w in vocab}
    for token_str, token_id in all_tokens.items():
        clean = re.sub(r'^[ĠĊ▁##]+', '', token_str).upper()
        if clean in vocab_upper or len(clean) == 0:
            vocab_ids.add(token_id)
        if re.match(r'^[0-9]+$', clean):
            vocab_ids.add(token_id)
        if re.match(r'^[ ,.\-/]+$', token_str):
            vocab_ids.add(token_id)
    for tok in [tokenizer.pad_token, tokenizer.eos_token,
                tokenizer.bos_token, tokenizer.unk_token]:
        if tok is not None and tok in all_tokens:
            vocab_ids.add(all_tokens[tok])
    return vocab_ids

# ═════════════════════════════════════════════════════════════════════════════
# 5c. DOMAIN SPECIFICATIONS  [CHANGE: SEMANTIC_METRICS]
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DomainSpec:
    """
    All domain-specific knowledge needed for semantic metric computation.
    Swap the instance passed to SemanticMetrics to change domain — no other
    code changes required.

    Fields
    ------
    name            : short label used in log output ("atc", "smcp", …)
    slot_patterns   : dict mapping slot name → compiled regex.
                      Each regex must capture the slot value in group 0 or
                      group 1. Slot names are arbitrary strings; they define
                      what Slot-F1 measures.
    slot_normalise  : dict mapping slot name → callable(str) -> str.
                      Applied to the matched string before comparison.
                      Default identity if slot not present.
    da_patterns     : ordered list of (label, compiled_regex).
                      First match wins. Fallback label is "other".
    halluc_checks   : list of callables(request, reference, generated) -> bool.
                      Each returns True if a hallucination is detected.
                      OR-combined: any True → hallucination flagged.
    """
    name:           str
    slot_patterns:  Dict[str, re.Pattern]
    slot_normalise: Dict[str, callable]
    da_patterns:    List[Tuple[str, re.Pattern]]
    halluc_checks:  List[callable]


# ── ATC domain ───────────────────────────────────────────────────────────────

def _atc_slot_patterns() -> Dict[str, re.Pattern]:
    return {
        "callsign":  re.compile(
            r'\b([A-Z]{2,3}\d{1,4}[A-Z]?|UAV[-\s]?\w+|[A-Z]+[-\s]\d+)\b',
            re.IGNORECASE),
        "action":    re.compile(
            r'\b(DESCEND|CLIMB|MAINTAIN|CONTACT|SQUAWK|HOLD|EXPECT|CLEARED|'
            r'PROCEED|TURN|REDUCE|INCREASE|REPORT|CONTINUE)\b',
            re.IGNORECASE),
        "altitude":  re.compile(
            r'\b(?:FL|FLIGHT\s+LEVEL)\s*(\d{2,3})\b|\b(\d{3,5})\s*(?:FEET|FT)\b',
            re.IGNORECASE),
        "frequency": re.compile(r'\b(1[12]\d\.\d{1,3})\b'),
    }

def _atc_da_patterns() -> List[Tuple[str, re.Pattern]]:
    return [
        ("correction", re.compile(r'\b(NEGATIVE|CORRECTION|SAY AGAIN)\b',    re.IGNORECASE)),
        ("readback",   re.compile(r'\b(WILCO|AFFIRM|ROGER|READBACK CORRECT|READ BACK)\b', re.IGNORECASE)),
        ("handoff",    re.compile(r'\bCONTACT\b',                             re.IGNORECASE)),
        ("hold",       re.compile(r'\bHOLD\b',                                re.IGNORECASE)),
        ("advisory",   re.compile(r'\bTRAFFIC\b',                             re.IGNORECASE)),
        ("clearance",  re.compile(
            r'\b(CLEARED|DESCEND|CLIMB|MAINTAIN|SQUAWK|EXPECT|PROCEED)\b',   re.IGNORECASE)),
    ]

def _atc_halluc_callsign(req, ref, gen):
    """Callsign in gen must appear in req or ref."""
    ctx = (req + " " + ref).upper().replace(" ", "").replace("-", "")
    pat = re.compile(
        r'\b([A-Z]{2,3}\d{1,4}[A-Z]?|UAV[-\s]?\w+|[A-Z]+[-\s]\d+)\b', re.IGNORECASE)
    for m in pat.finditer(gen.upper()):
        cs = m.group(0).replace(" ", "").replace("-", "").upper()
        if cs and cs not in ctx:
            return True
    return False

def _atc_halluc_altitude(req, ref, gen):
    """Flight level in gen must appear in req or ref."""
    pat = re.compile(r'\b(?:FL|FLIGHT\s+LEVEL)\s*(\d{2,3})\b', re.IGNORECASE)
    gen_fls = set(pat.findall(gen.upper()))
    ctx_fls = set(pat.findall((req + " " + ref).upper()))
    return bool(gen_fls - ctx_fls)

def _atc_halluc_frequency(req, ref, gen):
    """VHF frequency in gen must be valid and appear in req or ref."""
    pat = re.compile(r'\b(1[12]\d\.\d{1,3})\b')
    ctx = (req + " " + ref).upper()
    for m in pat.finditer(gen.upper()):
        freq = m.group(1)
        try:
            f = float(freq)
            if 118.0 <= f <= 136.975 and freq not in ctx:
                return True
        except ValueError:
            pass
    return False

ATC_DOMAIN = DomainSpec(
    name           = "atc",
    slot_patterns  = _atc_slot_patterns(),
    slot_normalise = {
        "callsign":  lambda s: s.replace(" ", "").replace("-", "").upper(),
        "action":    lambda s: s.upper(),
        "altitude":  lambda s: s.strip(),
        "frequency": lambda s: s.strip(),
    },
    da_patterns    = _atc_da_patterns(),
    halluc_checks  = [
        _atc_halluc_callsign,
        _atc_halluc_altitude,
        _atc_halluc_frequency,
    ],
)


# ── SMCP (maritime) domain ───────────────────────────────────────────────────
# IMO Standard Marine Communication Phrases
# Slots: vessel name, message type (action), location/waypoint, channel

def _smcp_slot_patterns() -> Dict[str, re.Pattern]:
    return {
        "vessel":    re.compile(
            r'\b([A-Z]{2,}(?:\s+[A-Z]+){0,3})\b(?=\s+(?:THIS IS|CALLING|COME IN|OVER))',
            re.IGNORECASE),
        "action":    re.compile(
            r'\b(MAYDAY|PAN-PAN|SECURITE|CALLING|REPORT|PROCEED|ANCHOR|STOP ENGINES|'
            r'MAINTAIN COURSE|ALTER COURSE|REDUCE SPEED|INCREASE SPEED|'
            r'STAND BY|ACKNOWLEDGE|ROGER|WILCO|OUT)\b',
            re.IGNORECASE),
        "channel":   re.compile(r'\bCHANNEL\s*(\d{2})\b|\bCH\.?\s*(\d{2})\b', re.IGNORECASE),
        "waypoint":  re.compile(
            r'\b([A-Z][A-Z\s]{2,20}(?:BUOY|LIGHT|POINT|ROCK|SHOAL|BANK|CAPE|HEAD))\b',
            re.IGNORECASE),
    }

def _smcp_da_patterns() -> List[Tuple[str, re.Pattern]]:
    return [
        ("distress",   re.compile(r'\bMAYDAY\b',             re.IGNORECASE)),
        ("urgency",    re.compile(r'\bPAN.?PAN\b',           re.IGNORECASE)),
        ("safety",     re.compile(r'\bSECURITE\b',           re.IGNORECASE)),
        ("correction", re.compile(r'\bCORRECTION\b',         re.IGNORECASE)),
        ("readback",   re.compile(r'\b(ROGER|WILCO|RECEIVED)\b', re.IGNORECASE)),
        ("instruction",re.compile(
            r'\b(PROCEED|ANCHOR|STOP ENGINES|ALTER COURSE|REDUCE SPEED|'
            r'MAINTAIN COURSE|STAND BY)\b', re.IGNORECASE)),
        ("call",       re.compile(r'\bCALLING\b',            re.IGNORECASE)),
    ]

def _smcp_halluc_vessel(req, ref, gen):
    """Vessel name in gen must appear in req or ref."""
    pat = re.compile(
        r'\b([A-Z]{2,}(?:\s+[A-Z]+){0,3})\b(?=\s+(?:THIS IS|CALLING|COME IN|OVER))',
        re.IGNORECASE)
    ctx = (req + " " + ref).upper()
    for m in pat.finditer(gen.upper()):
        vessel = m.group(1).upper()
        if vessel and vessel not in ctx:
            return True
    return False

def _smcp_halluc_channel(req, ref, gen):
    """VHF channel in gen must appear in req or ref."""
    pat = re.compile(r'\bCHANNEL\s*(\d{2})\b|\bCH\.?\s*(\d{2})\b', re.IGNORECASE)
    gen_chs = set(filter(None, [m.group(1) or m.group(2) for m in pat.finditer(gen.upper())]))
    ctx_chs = set(filter(None, [m.group(1) or m.group(2)
                                for m in pat.finditer((req + " " + ref).upper())]))
    return bool(gen_chs - ctx_chs)

SMCP_DOMAIN = DomainSpec(
    name           = "smcp",
    slot_patterns  = _smcp_slot_patterns(),
    slot_normalise = {
        "vessel":   lambda s: re.sub(r'\s+', ' ', s.upper().strip()),
        "action":   lambda s: s.upper(),
        "channel":  lambda s: s.strip().lstrip("0"),
        "waypoint": lambda s: re.sub(r'\s+', ' ', s.upper().strip()),
    },
    da_patterns    = _smcp_da_patterns(),
    halluc_checks  = [
        _smcp_halluc_vessel,
        _smcp_halluc_channel,
    ],
)


# ── Registry: add new domains here ───────────────────────────────────────────

DOMAIN_REGISTRY: Dict[str, DomainSpec] = {
    "atc":  ATC_DOMAIN,
    "smcp": SMCP_DOMAIN,
}

def get_domain_spec(domain: str) -> DomainSpec:
    """Return the DomainSpec for the given domain name."""
    if domain not in DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Available: {list(DOMAIN_REGISTRY.keys())}. "
            f"Add a new DomainSpec to DOMAIN_REGISTRY to support additional domains."
        )
    return DOMAIN_REGISTRY[domain]


# ═════════════════════════════════════════════════════════════════════════════
# 5d. SEMANTIC METRICS  [CHANGE: SEMANTIC_METRICS]
# ═════════════════════════════════════════════════════════════════════════════

class SemanticMetrics:
    """
    [CHANGE: SEMANTIC_METRICS]
    Computes Slot-F1, DA-F1, Hallucination%, and BERTScore over a list of
    (request, reference, generated) triples.

    All domain-specific logic is encapsulated in the DomainSpec instance.
    Swap domain_spec to evaluate a different domain — no other changes needed.

    Usage:
        spec = get_domain_spec("atc")   # or "smcp"
        sm   = SemanticMetrics(domain_spec=spec, bertscore_model="bert-base-uncased")
        results = sm.compute(examples)
        # examples: list of dicts with keys "request", "reference", "generated"
    """

    def __init__(self, domain_spec: DomainSpec,
                 bertscore_model: str = "bert-base-uncased"):
        self.spec            = domain_spec
        self.bertscore_model = bertscore_model

    # ── Slot extraction ──────────────────────────────────────────────────────

    def _extract_slots(self, text: str) -> Dict[str, Optional[str]]:
        """Extract slot values using domain-specific patterns."""
        t      = text.upper()
        result = {}
        for slot_name, pattern in self.spec.slot_patterns.items():
            m = pattern.search(t)
            if m:
                # Use first non-None group, or group(0) as fallback
                raw = next((g for g in m.groups() if g is not None), m.group(0))
                norm_fn = self.spec.slot_normalise.get(slot_name, lambda s: s)
                result[slot_name] = norm_fn(raw)
            else:
                result[slot_name] = None
        return result

    # ── Slot-F1 ─────────────────────────────────────────────────────────────

    def _slot_f1_single(self, ref_slots: Dict, gen_slots: Dict) -> float:
        """
        Token-level F1 between reference and generated slot values.
        Micro-averages across all slots defined in the domain spec.
        Slots absent in both ref and gen contribute 0 TP, 0 FP, 0 FN.
        """
        tp = fp = fn = 0
        for key in self.spec.slot_patterns:
            r = set((ref_slots.get(key) or "").split())
            g = set((gen_slots.get(key) or "").split())
            tp += len(r & g)
            fp += len(g - r)
            fn += len(r - g)
        if tp + fp + fn == 0:
            return 1.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    # ── Dialogue act classification ──────────────────────────────────────────

    def _classify_da(self, text: str) -> str:
        """Rule-based DA classifier using domain-specific patterns."""
        for label, pattern in self.spec.da_patterns:
            if pattern.search(text):
                return label
        return "other"

    # ── Hallucination check ──────────────────────────────────────────────────

    def _is_hallucinated(self, request: str, reference: str, generated: str) -> bool:
        """
        OR-combines all domain-specific hallucination checks.
        Returns True if any check fires.
        """
        return any(
            check(request, reference, generated)
            for check in self.spec.halluc_checks
        )

    # ── BERTScore ────────────────────────────────────────────────────────────

    def _bertscore(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Compute mean sentence-level cosine similarity as a BERTScore proxy.

        Two paths:
        - Local directory: saved as BertForMaskedLM → load via BertForMaskedLM,
          extract the .bert encoder, discard the MLM head. This avoids the
          UNEXPECTED/MISSING key warnings from loading into the wrong architecture.
        - Hub model string: delegate to the bert-score library as normal.

        Variable naming: uses `bert_F` for the BERTScore F tensor to avoid
        shadowing the torch.nn.functional alias `F` used in the same method.
        """
        if not hypotheses:
            return 0.0, []
        per_ex = []
        try:
            import torch.nn.functional as torch_F
            from transformers import BertTokenizerFast, BertForMaskedLM, AutoTokenizer

            device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = self.bertscore_model
            is_local   = Path(model_path).exists() and Path(model_path).is_dir()

            if is_local or not BERTSCORE_AVAILABLE:
                import transformers as _hf_bs
                _hf_bs.logging.set_verbosity_error()
                bs_tok  = BertTokenizerFast.from_pretrained(model_path)
                mlm_mdl = BertForMaskedLM.from_pretrained(
                    model_path,
                    ignore_mismatched_sizes=True,
                )
                encoder = mlm_mdl.bert.to(device)
                encoder.eval()
                del mlm_mdl

                def _embed(texts):
                    enc = bs_tok(
                        texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        out = encoder(**enc)
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
                    return torch_F.normalize(vecs, dim=-1)

                ref_vecs = _embed(references)
                hyp_vecs = _embed(hypotheses)
                scores   = (ref_vecs * hyp_vecs).sum(dim=-1)   # [N]
                per_ex   = [float(s) for s in scores.tolist()]

                del encoder
                torch.cuda.empty_cache()
                return float(scores.mean().item()), per_ex

            else:
                _, _, bert_F = bert_score_fn(
                    hypotheses, references,
                    model_type=model_path,
                    verbose=False,
                    device=str(device),
                )
                per_ex = [float(s) for s in bert_F.tolist()]
                return float(bert_F.mean().item()), per_ex

        except Exception as e:
            print(f"  WARNING: BERTScore failed: {e}")
            return 0.0, per_ex

    # ── Main compute ─────────────────────────────────────────────────────────

    def compute(self, examples: List[Dict]) -> Dict[str, float]:
        """
        Args:
            examples: list of dicts with keys "request", "reference", "generated"

        Returns:
            aggregates: {
                "slot_f1":    float,
                "da_f1":      float,
                "halluc_pct": float,
                "bertscore":  float,
            }
            per_example: list of dicts with same keys, one per example
        """
        if not examples:
            empty = {"slot_f1": 0.0, "da_f1": 0.0,
                     "halluc_pct": 0.0, "bertscore": 0.0}
            return empty, []

        slot_f1_vals = []
        halluc_flags = []
        ref_das, gen_das = [], []
        references, hypotheses = [], []

        for ex in examples:
            req = ex.get("request",   "")
            ref = ex.get("reference", "")
            gen = ex.get("generated", "")

            ref_slots = self._extract_slots(ref)
            gen_slots = self._extract_slots(gen)
            slot_f1_vals.append(self._slot_f1_single(ref_slots, gen_slots))

            ref_das.append(self._classify_da(ref))
            gen_das.append(self._classify_da(gen))

            halluc_flags.append(self._is_hallucinated(req, ref, gen))

            references.append(ref)
            hypotheses.append(gen)

        # DA macro-F1 (aggregate only — per-example DA is 0/1)
        da_labels       = sorted(set(ref_das))
        da_f1_per_class = []
        for label in da_labels:
            tp = sum(r == label and g == label for r, g in zip(ref_das, gen_das))
            fp = sum(r != label and g == label for r, g in zip(ref_das, gen_das))
            fn = sum(r == label and g != label for r, g in zip(ref_das, gen_das))
            if tp + fp + fn == 0:
                continue
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            da_f1_per_class.append(f)

        # BERTScore — now returns (mean, per_example_list)
        bs_mean, bs_per_ex = self._bertscore(references, hypotheses)

        # Per-example DA correctness (1 if predicted DA matches reference DA)
        da_correct = [1.0 if r == g else 0.0
                      for r, g in zip(ref_das, gen_das)]

        # Build per-example records
        n = len(examples)
        per_example_sem = []
        for i in range(n):
            per_example_sem.append({
                "slot_f1":  float(slot_f1_vals[i]),
                "da_f1":    float(da_correct[i]),   # binary per-example DA match
                "halluc":   float(halluc_flags[i]),
                "bertscore": float(bs_per_ex[i]) if i < len(bs_per_ex) else 0.0,
            })

        aggregates = {
            "slot_f1":    float(sum(slot_f1_vals) / len(slot_f1_vals)),
            "da_f1":      float(sum(da_f1_per_class) / len(da_f1_per_class))
                          if da_f1_per_class else 0.0,
            "halluc_pct": float(sum(halluc_flags) / len(halluc_flags)),
            "bertscore":  bs_mean,
        }
        return aggregates, per_example_sem


# ═════════════════════════════════════════════════════════════════════════════
# 5d. GRADNORM  [CHANGE: GRADNORM]
# ═════════════════════════════════════════════════════════════════════════════

class GradNorm:
    """
    [CHANGE: GRADNORM]
    Chen et al. (2018) GradNorm: dynamically rebalance loss weights so that
    gradient norms stay proportional to relative training rates.
    Enabled via SCOPEConfig.use_gradnorm = True / --gradnorm flag.
    Manages 4 task weights: [CE, tok, phr, cfg].
    When disabled (default), training is identical to the original script.
    """
    def __init__(self, initial_weights: List[float], alpha: float, device):
        self.alpha = alpha
        self.n     = len(initial_weights)
        w0 = torch.tensor(initial_weights, dtype=torch.float32, device=device)
        self.log_w = torch.log(w0 / w0.sum() * self.n).detach().requires_grad_(True)
        self.opt   = torch.optim.Adam([self.log_w], lr=0.025)
        self.L0: Optional[torch.Tensor] = None

    @property
    def weights(self) -> torch.Tensor:
        return (F.softmax(self.log_w, dim=0) * self.n).detach()

    def step(
        self,
        task_losses: List[torch.Tensor],
        shared_params: List[torch.Tensor],
    ) -> Dict[str, float]:
        device = self.log_w.device
        ws = F.softmax(self.log_w, dim=0) * self.n

        # Coerce all inputs to tensors on the correct device
        coerced = []
        for l in task_losses:
            if not isinstance(l, torch.Tensor):
                l = torch.tensor(float(l), device=device)
            elif l.device != device:
                l = l.to(device)
            coerced.append(l)
        task_losses = coerced

        if self.L0 is None:
            self.L0 = torch.stack([l.detach() for l in task_losses])

        G = []
        for w_i, l_i in zip(ws, task_losses):
            has_graph = l_i.grad_fn is not None
            is_leaf_grad = l_i.requires_grad and l_i.grad_fn is None
            if has_graph:
                try:
                    grads   = torch.autograd.grad(
                        w_i * l_i, shared_params,
                        retain_graph=True, allow_unused=True, create_graph=False,
                    )
                    sq_terms = [g.detach().norm(2) ** 2
                                for g in grads if g is not None]
                    g_norm = torch.stack(sq_terms).sum() ** 0.5 \
                             if sq_terms else torch.tensor(0.0, device=device)
                except Exception:
                    g_norm = torch.tensor(0.0, device=device)
            elif is_leaf_grad:
                # Proxy loss — no computation graph through shared params.
                # Use the scaled loss value itself as a surrogate gradient norm
                # so GradNorm still gets a meaningful signal for these tasks.
                g_norm = (w_i * l_i.detach()).abs()
            else:
                g_norm = torch.tensor(0.0, device=device)
            G.append(g_norm)

        G_t   = torch.stack(G)
        G_bar = G_t.mean().detach()

        loss_ratios = torch.stack([
            l.detach() / (self.L0[i] + 1e-8) for i, l in enumerate(task_losses)
        ])
        r_bar   = loss_ratios.mean()
        r_i     = loss_ratios / (r_bar + 1e-8)
        targets = (G_bar * r_i ** self.alpha).detach()
        gn_loss = (G_t - targets).abs().sum()

        self.opt.zero_grad()
        if gn_loss.grad_fn is not None:
            gn_loss.backward()
            self.opt.step()

        w_vals = ws.detach().tolist()
        return {"gn_ce": w_vals[0], "gn_tok": w_vals[1],
                "gn_phr": w_vals[2], "gn_cfg": w_vals[3]}

# ═════════════════════════════════════════════════════════════════════════════
# 5e. DOMAIN BERT FINE-TUNING FOR BERTSCORE  [CHANGE: SEMANTIC_METRICS]
# ═════════════════════════════════════════════════════════════════════════════

def finetune_bert_mlm(
    pairs: List[dict],
    output_dir: str,
    base_model: str = "bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    mlm_probability: float = 0.15,
    max_length: int = 128,
) -> str:
    """
    [CHANGE: SEMANTIC_METRICS]
    Fine-tune a BERT model on ATC response texts using masked language modelling
    (MLM). Self-supervised — no labels required. Uses only the response texts
    from the training pairs.

    This produces a domain-adapted BERT whose embeddings are sensitive to
    ATC-specific numerical and procedural distinctions (e.g. FL240 vs FL280),
    making BERTScore meaningful for ATC output evaluation.

    Args:
        pairs:      training pairs (list of dicts with "response" key)
        output_dir: where to save the fine-tuned checkpoint
        base_model: HuggingFace BERT model to start from
        epochs:     MLM training epochs (3 is sufficient for ~1000 examples)
        batch_size: MLM batch size (16 fits comfortably in <4GB VRAM)
        lr:         learning rate for AdamW (2e-5 is standard for BERT MLM)
        mlm_probability: fraction of tokens masked per sequence (BERT default 0.15)
        max_length: max token length per response

    Returns:
        path to the saved fine-tuned checkpoint (use as bertscore_model)

    Runtime: ~30-60 min on a single GPU for 1000 examples × 3 epochs.
    Skipped automatically if output_dir already contains a saved checkpoint.
    """
    from transformers import (BertForMaskedLM, BertTokenizerFast,
                               DataCollatorForLanguageModeling)
    import transformers as _hf
    _hf.logging.set_verbosity_error()   # suppress UNEXPECTED/MISSING key banners

    ckpt_path = Path(output_dir) / "bert_atc"

    # Skip if already fine-tuned
    if (ckpt_path / "config.json").exists():
        print(f"  Domain BERT already exists at {ckpt_path} — skipping MLM fine-tuning.")
        return str(ckpt_path)

    print(f"\n[BERT MLM] Fine-tuning {base_model} on ATC corpus "
          f"({len(pairs)} examples, {epochs} epochs)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_tok = BertTokenizerFast.from_pretrained(base_model)
    bert_mdl = BertForMaskedLM.from_pretrained(
        base_model,
        ignore_mismatched_sizes=True,   # suppress UNEXPECTED key warnings
    ).to(device)

    # Collect response texts — these are the ATC phraseology sentences
    texts = [p["response"] for p in pairs if p.get("response", "").strip()]

    # Tokenise
    encodings = bert_tok(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    class _MLMDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.ids   = enc["input_ids"]
            self.masks = enc["attention_mask"]
        def __len__(self):
            return self.ids.size(0)
        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "attention_mask": self.masks[i]}

    collator = DataCollatorForLanguageModeling(
        tokenizer=bert_tok,
        mlm=True,
        mlm_probability=mlm_probability,
    )
    loader = DataLoader(
        _MLMDataset(encodings),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    opt = torch.optim.AdamW(bert_mdl.parameters(), lr=lr, weight_decay=0.01)
    total = len(loader) * epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, total // 10), total)

    bert_mdl.train()
    for epoch in range(epochs):
        epoch_loss  = 0.0
        valid_steps = 0
        for step, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Skip batches where all tokens are masked (DataCollator edge case)
            if (labels != -100).sum() == 0:
                continue

            opt.zero_grad()
            out  = bert_mdl(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = out.loss

            # Guard against NaN/Inf from degenerate batches — skip without updating
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_mdl.parameters(), 1.0)
            opt.step()
            sched.step()
            epoch_loss  += loss.item()
            valid_steps += 1

        mean_loss = epoch_loss / max(valid_steps, 1)
        print(f"  [BERT MLM] Epoch {epoch+1}/{epochs} | loss={mean_loss:.4f} "
              f"({valid_steps}/{len(loader)} valid steps)")

    ckpt_path.mkdir(parents=True, exist_ok=True)
    bert_mdl.save_pretrained(str(ckpt_path))
    bert_tok.save_pretrained(str(ckpt_path))
    print(f"  [BERT MLM] Domain BERT saved to {ckpt_path}")

    del bert_mdl
    torch.cuda.empty_cache()
    return str(ckpt_path)


# ═════════════════════════════════════════════════════════════════════════════
# 6. CONFIG
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SCOPEConfig:
    model_name:    str   = "gpt2-large"
    data_path:     str   = "atc_pairs.json"   # single-file fallback (in-script split)
    train_data:    str   = ""                  # [CHANGE: PRE-SPLIT] pre-split train file
    val_data:      str   = ""                  # [CHANGE: PRE-SPLIT] pre-split val file
    output_dir:    str   = "scope_output"
    # Loss weights
    lambda_tok:    float = 0.5
    lambda_phr:    float = 0.3
    lambda_cfg:    float = 0.2
    lambda_ce:     float = 1.0
    M_samples:     int   = 4
    # Training
    epochs:        int   = 5
    batch_size:    int   = 8
    lr:            float = 5e-5
    max_length:    int   = 512
    max_new_tok:   int   = 64
    warmup_steps:  int   = 100
    warmup_ratio:  float = 0.0   # [CHANGE: WARMUP_RATIO] if > 0, overrides warmup_steps
    grad_clip:     float = 1.0
    # Ablation flags
    use_ltok:      bool  = True
    use_lphr:      bool  = True
    use_lcfg:      bool  = True
    # DPO
    use_dpo:       bool  = False
    dpo_beta:      float = 0.1
    dpo_ref_model: str   = ""
    seed:          int   = 42
    # Large-model support
    grad_accum:              int  = 1
    use_chat_template:       bool = False
    gradient_checkpointing:  bool = False
    # Domain
    domain:        str   = "atc"
    vocab_path:    str   = "vocab_ATC.json"
    phrase_path:   str   = "ngram_whitelist_ATC.json"
    grammar_path:  str   = "G_ATC.lark"
    # [CHANGE: CBAR_CHECKPOINT] checkpoint selection metric: "c_tok" (original) or "c_bar"
    checkpoint_metric: str = "c_bar"
    # [CHANGE: EARLY_STOPPING] stop when checkpoint_metric fails to improve for N epochs
    early_stop_patience: int = 0   # 0 = disabled (original behaviour)
    # [CHANGE: GRADNORM] dynamic lambda rebalancing
    use_gradnorm:        bool  = False
    gradnorm_alpha:      float = 1.5
    gradnorm_update_freq: int  = 20
    use_8bit_adam:        bool = False  # auto-enabled when 2+ GPUs detected
    # [CHANGE: SEMANTIC_METRICS] semantic and safety evaluation
    bertscore_model:         str   = "bert-base-uncased"
    bertscore_weight:        float = 0.0
    hallucination_threshold: float = 0.10
    # [CHANGE: CURRICULUM:CL1] curriculum learning control
    curriculum:           bool  = False   # False = identical to scope_train_general.py
    curriculum_phase1:    float = 1/3     # fraction of epochs for Phase 1 (CE only)
    curriculum_phase2:    float = 2/3     # fraction of epochs up to end of Phase 2
    curriculum_ramp_steps: int  = 0       # steps to ramp new lambdas at transitions
    # [CHANGE: SEMANTIC_METRICS] domain BERT fine-tuning for BERTScore
    finetune_bert:           bool  = False   # enable MLM fine-tuning before main training
    bert_mlm_epochs:         int   = 3
    bert_mlm_batch:          int   = 16
    bert_mlm_lr:             float = 2e-5

# ═════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def _verify_checkpoint(ckpt_dir: str, min_shard_bytes: int = 1_000_000) -> None:
    """
    Verify that every .safetensors shard in ckpt_dir is non-empty and readable.
    Calls os.sync() to flush kernel buffers (important on Google Drive FUSE mount),
    then checks each shard is at least min_shard_bytes.
    Raises RuntimeError if any shard is truncated.
    """
    import os, time
    ckpt_path = Path(ckpt_dir)

    # Flush kernel buffers to filesystem (Drive FUSE may buffer large writes)
    try:
        os.sync()
    except Exception:
        pass
    time.sleep(2)  # allow Drive FUSE to finish flushing

    shards = list(ckpt_path.glob("*.safetensors"))
    if not shards:
        return  # pytorch_model.bin format — skip size check

    bad = []
    for shard in shards:
        size = shard.stat().st_size
        if size < min_shard_bytes:
            bad.append(f"{shard.name}: {size:,} bytes (expected ≥ {min_shard_bytes:,})")

    if bad:
        raise RuntimeError(
            f"Checkpoint at {ckpt_dir} has truncated shards — Drive flush failed:\n"
            + "\n".join(f"  {b}" for b in bad)
        )

    total_gb = sum(s.stat().st_size for s in shards) / 1e9
    print(f"  ✓ Checkpoint verified: {len(shards)} shard(s), {total_gb:.2f} GB")


def train(cfg: SCOPEConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # [CHANGE: MULTI_GPU] use cuda:0 for tensor placement; model spans all GPUs
    device = _tensor_device()
    n_gpus = _n_gpus()
    print(f"Device: {device}  |  GPUs visible: {n_gpus}")
    if n_gpus >= 2:
        print(f"  Multi-GPU mode: model layers split across {n_gpus} GPUs via device_map='auto'")
    print(f"Config: lambda_ce={cfg.lambda_ce}, lambda_tok={cfg.lambda_tok}, "
          f"lambda_phr={cfg.lambda_phr}, lambda_cfg={cfg.lambda_cfg}")

    # Load regulatory artefacts
    print("Loading regulatory artefacts...")
    vocab      = load_whitelist(Path(cfg.vocab_path))
    ngram_wl   = load_ngram_whitelist(Path(cfg.phrase_path))
    cfg_parser = load_grammar(Path(cfg.grammar_path)) if cfg.use_lcfg else None
    print(f"  V_ATC: {len(vocab)} tokens | "
          f"P_ATC: {sum(len(v) for v in ngram_wl.values())} n-grams | "
          f"G_ATC: {'loaded' if cfg_parser else 'disabled'}")

    # Load model and tokeniser — [CHANGE: MULTI_GPU] use _load_model()
    print(f"Loading {cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = _load_model(cfg.model_name,
                        dtype=torch.bfloat16,
                        gradient_checkpointing=cfg.gradient_checkpointing)
    if cfg.gradient_checkpointing:
        print("  Gradient checkpointing enabled")

    # DPO reference model — [CHANGE: MULTI_GPU] split across GPUs, stay there permanently
    ref_model = None
    if cfg.use_dpo:
        ref_path = cfg.dpo_ref_model if cfg.dpo_ref_model else cfg.model_name
        print(f"  Loading DPO reference model from {ref_path} ...")
        ref_model = _load_model(ref_path, dtype=torch.bfloat16,
                                gradient_checkpointing=False)
        ref_model.config.use_cache = False
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        _ref_gb = sum(p.numel() * p.element_size()
                      for p in ref_model.parameters()) / 1024**3
        print(f"  Reference ready — {_ref_gb:.1f} GB total, "
              f"no per-step CPU↔GPU transfers")

    vocab_ids = build_vocab_ids(tokenizer, vocab)
    print(f"  Vocab IDs in tokenizer: {len(vocab_ids)} / {tokenizer.vocab_size}")

    # Dataset
    # [CHANGE: PRE-SPLIT] If cfg.train_data / cfg.val_data are provided,
    # load them directly (fixed split). Falls back to in-script split from
    # cfg.data_path for backward compatibility with single-file workflows.
    if cfg.train_data and cfg.val_data:
        with open(cfg.train_data) as f:
            train_pairs = json.load(f)
        with open(cfg.val_data) as f:
            val_pairs = json.load(f)
        print(f"  Pre-split: {len(train_pairs)} train / {len(val_pairs)} val")
    else:
        with open(cfg.data_path) as f:
            all_pairs = json.load(f)
        random.shuffle(all_pairs)
        n        = len(all_pairs)
        n_train  = int(0.8 * n)
        n_val    = int(0.1 * n)
        train_pairs = all_pairs[:n_train]
        val_pairs   = all_pairs[n_train:n_train+n_val]
        print(f"  Split: {len(train_pairs)} train / {len(val_pairs)} val / "
              f"{n - n_train - n_val} test")

    # [CHANGE: SEMANTIC_METRICS] domain BERT fine-tuning (runs once per model, cached)
    # If --bertscore_model already points to a valid local checkpoint (e.g. from
    # a previously completed condition), skip MLM and reuse it directly.
    if cfg.finetune_bert:
        _bm = Path(cfg.bertscore_model)
        _already_exists = _bm.exists() and (_bm / "config.json").exists()

        if not _already_exists:
            # Also check sibling condition dirs for a cached bert_atc
            # e.g. C2 can reuse C1/bert_atc without retraining
            _output_root = Path(cfg.output_dir).parent  # e.g. results/llama
            for _sibling in sorted(_output_root.glob("*/bert_atc")):
                if (_sibling / "config.json").exists():
                    _already_exists = True
                    cfg.bertscore_model = str(_sibling)
                    print(f"  [BERT MLM] Reusing cached domain BERT from {_sibling}")
                    break

        if not _already_exists:
            cfg.bertscore_model = finetune_bert_mlm(
                pairs      = train_pairs,
                output_dir = cfg.output_dir,
                base_model = cfg.bertscore_model,
                epochs     = cfg.bert_mlm_epochs,
                batch_size = cfg.bert_mlm_batch,
                lr         = cfg.bert_mlm_lr,
            )
        print(f"  BERTScore will use domain BERT at: {cfg.bertscore_model}")

    train_ds = AtcDataset(train_pairs, tokenizer, cfg.max_length, domain=cfg.domain)
    val_ds   = AtcDataset(val_pairs,   tokenizer, cfg.max_length, domain=cfg.domain)

    def collate_fn(batch):
        result = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            result[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
        return result

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True,  collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                          shuffle=False, collate_fn=collate_fn)

    # DPO pairs
    dpo_dl = None
    if cfg.use_dpo:
        dpo_pairs = build_dpo_pairs(
            train_pairs, tokenizer, model, vocab, ngram_wl,
            cfg.max_new_tok, device, domain=cfg.domain,
            cfg_parser=cfg_parser,   # [CHANGE: DPO_COMPOSITE_SIGNAL]
        )
        dpo_ds = DPODataset(dpo_pairs, tokenizer, cfg.max_length)
        dpo_dl = DataLoader(dpo_ds, batch_size=cfg.batch_size,
                            shuffle=True, collate_fn=collate_fn)
        print(f"  DPO DataLoader: {len(dpo_ds)} preference pairs")

    # Optimiser
    # [CHANGE: MULTI_GPU] 8-bit Adam keeps optimizer states manageable on GPU 0.
    # With device_map="auto" and 2×40GB, standard fp32 Adam states (~60GB for 8B)
    # would OOM on GPU 0 alone. 8-bit Adam reduces this to ~15GB.
    # Falls back to standard AdamW if bitsandbytes is not installed.
    if _n_gpus() >= 2 or cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(
                model.parameters(), lr=cfg.lr, weight_decay=0.01)
            print(f"  Optimizer: AdamW8bit (bitsandbytes) "
                  f"— required for multi-GPU to keep GPU 0 within budget")
        except ImportError:
            print("  WARNING: bitsandbytes not installed; falling back to AdamW. "
                  "On 2×40GB this may OOM during 8B training. "
                  "Fix: pip install bitsandbytes")
            opt = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps = len(train_dl) * cfg.epochs

    # [CHANGE: WARMUP_RATIO] compute warmup from ratio if provided
    if cfg.warmup_ratio > 0:
        effective_warmup = max(1, int(cfg.warmup_ratio * total_steps))
    else:
        effective_warmup = cfg.warmup_steps

    # [CHANGE: COSINE_SCHEDULE] cosine decay preserves early gains better than linear
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=effective_warmup,
        num_training_steps=total_steps,
    )
    print(f"  LR schedule: cosine, warmup={effective_warmup} / {total_steps} steps")

    # [CHANGE: GRADNORM] set up GradNorm if enabled
    gradnorm      = None
    shared_params = None
    lam_ce  = cfg.lambda_ce
    lam_tok = cfg.lambda_tok
    lam_phr = cfg.lambda_phr
    lam_cfg = cfg.lambda_cfg

    if cfg.use_gradnorm:
        try:
            shared_params = list(model.model.layers[-1].parameters())
        except AttributeError:
            shared_params = list(model.transformer.h[-1].parameters())
        gradnorm = GradNorm(
            initial_weights=[lam_ce, lam_tok, lam_phr, lam_cfg],
            alpha=cfg.gradnorm_alpha,
            device=device,
        )
        print(f"  GradNorm enabled (alpha={cfg.gradnorm_alpha}, "
              f"update every {cfg.gradnorm_update_freq} steps)")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # [CHANGE: SEMANTIC_METRICS] initialise semantic evaluator with domain spec
    domain_spec = get_domain_spec(cfg.domain)
    sem_metrics = SemanticMetrics(domain_spec=domain_spec,
                                  bertscore_model=cfg.bertscore_model)

    # Checkpoint tracking
    # [CHANGE: CBAR_CHECKPOINT] track both C_bar (primary) and C_tok (legacy)
    best_metric    = 0.0     # value of cfg.checkpoint_metric at best epoch
    best_val_ctok  = 0.0     # kept for logging parity with original
    no_improve     = 0       # [CHANGE: EARLY_STOPPING] patience counter

    history      = []
    step_history = []
    global_step  = 0

    # [CHANGE: CURRICULUM:CL1] build CurriculumConfig from SCOPEConfig
    cur_config = CurriculumConfig(
        phase_fractions=(cfg.curriculum_phase1, cfg.curriculum_phase2),
        ramp_steps=cfg.curriculum_ramp_steps,
    ) if cfg.curriculum else None

    # Store original condition flags — curriculum overrides these per-epoch
    _orig_use_ltok = cfg.use_ltok
    _orig_use_lphr = cfg.use_lphr
    _orig_use_lcfg = cfg.use_lcfg
    _orig_lam_tok  = lam_tok
    _orig_lam_phr  = lam_phr
    _orig_lam_cfg  = lam_cfg

    for epoch in range(cfg.epochs):
        # [CHANGE: CURRICULUM:CL2] resolve active losses for this epoch
        if cur_config is not None:
            phase_name, e_use_ltok, e_use_lphr, e_use_lcfg, \
            e_lam_tok, e_lam_phr, e_lam_cfg = curriculum_phase(
                epoch, cfg.epochs,
                _orig_use_ltok, _orig_use_lphr, _orig_use_lcfg,
                _orig_lam_tok,  _orig_lam_phr,  _orig_lam_cfg,
                cur_config,
            )
        else:
            # [CHANGE: CURRICULUM] no curriculum → use static config (original behaviour)
            phase_name  = "no-curriculum"
            e_use_ltok  = cfg.use_ltok
            e_use_lphr  = cfg.use_lphr
            e_use_lcfg  = cfg.use_lcfg
            e_lam_tok   = lam_tok
            e_lam_phr   = lam_phr
            e_lam_cfg   = lam_cfg

        # [CHANGE: CURRICULUM:CL3] log active phase
        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch+1}/{cfg.epochs} | Curriculum: {phase_name}")
        print(f"  Active losses: CE=✓ "
              f"tok={'✓' if e_use_ltok else '✗'} "
              f"phr={'✓' if e_use_lphr else '✗'} "
              f"cfg={'✓' if e_use_lcfg else '✗'}")
        print(f"  λ: ce={lam_ce:.3f} tok={e_lam_tok:.3f} "
              f"phr={e_lam_phr:.3f} cfg={e_lam_cfg:.3f}")
        print(f"{'─'*60}")

        # [CHANGE: CURRICULUM:CL4] ramp new lambdas at phase transitions
        _ramp_counter  = 0
        _prev_phase    = getattr(train, '_prev_phase', None)
        _phase_changed = (phase_name != _prev_phase)
        train._prev_phase = phase_name
        _ramp_tok = e_lam_tok if not _phase_changed else 0.0
        _ramp_phr = e_lam_phr if not _phase_changed else 0.0
        _ramp_cfg = e_lam_cfg if not _phase_changed else 0.0
        model.train()
        epoch_losses = {"ce": 0., "tok": 0., "phr": 0., "cfg": 0., "total": 0.}
        n_batches = 0

        for step, batch in enumerate(train_dl):
            # [CHANGE: CURRICULUM:CL4] ramp new lambdas over first ramp_steps of new phase
            if _phase_changed and cur_config is not None and cur_config.ramp_steps > 0:
                ramp_frac  = min(1.0, _ramp_counter / max(1, cur_config.ramp_steps))
                _step_lam_tok = ramp_frac * e_lam_tok
                _step_lam_phr = ramp_frac * e_lam_phr
                _step_lam_cfg = ramp_frac * e_lam_cfg
                _ramp_counter += 1
            else:
                _step_lam_tok = e_lam_tok
                _step_lam_phr = e_lam_phr
                _step_lam_cfg = e_lam_cfg
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            response_mask  = batch["response_mask"].to(device)

            opt.zero_grad()

            # CE loss
            out    = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            L_ce   = out.loss
            logits = out.logits

            # L_tok (differentiable) — [CHANGE: CURRICULUM] use epoch-resolved flag
            L_tok_val = torch.tensor(0., device=device)
            if e_use_ltok:
                L_tok_val = compute_L_tok(logits, response_mask, vocab_ids, tokenizer.vocab_size)

            # L_phr + L_cfg (GRPO, every other step)
            _phr_loss_terms = []
            _cfg_loss_terms = []
            _phr_display    = 0.0
            _cfg_display    = 0.0

            # [CHANGE: CURRICULUM] use epoch-resolved flags
            if (e_use_lphr or e_use_lcfg) and (step % 2 == 0):
                model.eval()
                all_generated    = []
                phr_rewards_list = []
                cfg_rewards_list = []
                with torch.no_grad():
                    for _ in range(cfg.M_samples):
                        out_g = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=cfg.max_new_tok,
                            do_sample=True,
                            temperature=1.5,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        gen = out_g[:, input_ids.size(1):]
                        all_generated.append(gen)
                        text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
                        toks = text.upper().split()
                        if e_use_lphr:
                            phr_rewards_list.append(compute_cphr(toks, ngram_wl))
                        if e_use_lcfg and cfg_parser:
                            cfg_rewards_list.append(compute_ccfg_partial(text, cfg_parser))
                model.train()

                def grpo_advantages(rewards):
                    if len(rewards) < 2:
                        return rewards
                    mu  = statistics.mean(rewards)
                    std = statistics.stdev(rewards)
                    if std < 1e-8:
                        return [0.0] * len(rewards)
                    return [(r - mu) / std for r in rewards]

                phr_adv = grpo_advantages(phr_rewards_list) if phr_rewards_list else []
                cfg_adv = grpo_advantages(cfg_rewards_list) if cfg_rewards_list else []

                for m_idx, gen_m in enumerate(all_generated):
                    if gen_m.size(1) == 0:
                        continue
                    phr_adv_m = phr_adv[m_idx] if m_idx < len(phr_adv) else 0.0
                    cfg_adv_m = cfg_adv[m_idx] if m_idx < len(cfg_adv) else 0.0
                    if abs(phr_adv_m) < 1e-8 and abs(cfg_adv_m) < 1e-8:
                        continue

                    full   = torch.cat([input_ids, gen_m], dim=1)
                    attn   = torch.ones(input_ids.size(0), full.size(1) - 1,
                                        dtype=torch.long, device=device)
                    logits_g = model(full[:, :-1], attention_mask=attn).logits
                    logits_g = logits_g[:, input_ids.size(1) - 1:, :]
                    log_p    = F.log_softmax(logits_g, dim=-1)
                    idx_t    = gen_m.unsqueeze(-1).clamp(0, log_p.size(-1) - 1)
                    seq_lp   = log_p.gather(-1, idx_t).squeeze(-1).mean(dim=-1)

                    grpo_m = torch.tensor(0.0, device=device)
                    if e_use_lphr and abs(phr_adv_m) > 1e-8:  # [CHANGE: CURRICULUM]
                        grpo_m = grpo_m + (-phr_adv_m * _step_lam_phr * seq_lp.mean() / cfg.M_samples)
                        _phr_loss_terms.append(float(seq_lp.mean().detach()))
                    if e_use_lcfg and cfg_parser and abs(cfg_adv_m) > 1e-8:  # [CHANGE: CURRICULUM]
                        grpo_m = grpo_m + (-cfg_adv_m * _step_lam_cfg * seq_lp.mean() / cfg.M_samples)
                        _cfg_loss_terms.append(float(seq_lp.mean().detach()))

                    if grpo_m.grad_fn is not None:
                        grpo_m.backward()
                    del full, attn, logits_g, log_p, idx_t, seq_lp, grpo_m

                if phr_rewards_list:
                    _phr_display = sum(phr_rewards_list) / len(phr_rewards_list)
                if cfg_rewards_list:
                    _cfg_display = sum(cfg_rewards_list) / len(cfg_rewards_list)

            # [CHANGE: GRADNORM] update lambdas dynamically
            # l_phr_proxy and l_cfg_proxy must be tensors with grad_fn so
            # GradNorm can compute gradient norms through them.
            # We use 1 - reward as a surrogate loss (higher reward = lower loss).
            if gradnorm is not None and (step % cfg.gradnorm_update_freq == 0):
                # Build differentiable proxy losses for phr and cfg.
                # These don't need to be the exact GRPO objectives — they just
                # need a grad_fn so GradNorm can measure gradient magnitudes.
                if cfg.lambda_phr > 0 and vocab_ids is not None:
                    # Reuse L_tok computation path as a structural proxy —
                    # logits still live in the graph at this point.
                    with torch.enable_grad():
                        l_phr_proxy = torch.tensor(
                            1.0 - _phr_display, dtype=torch.float32,
                            device=device, requires_grad=True
                        )
                else:
                    l_phr_proxy = torch.tensor(1.0 - _phr_display, device=device)

                if cfg.lambda_cfg > 0:
                    with torch.enable_grad():
                        l_cfg_proxy = torch.tensor(
                            1.0 - _cfg_display, dtype=torch.float32,
                            device=device, requires_grad=True
                        )
                else:
                    l_cfg_proxy = torch.tensor(1.0 - _cfg_display, device=device)

                gradnorm.step([L_ce, L_tok_val, l_phr_proxy, l_cfg_proxy], shared_params)
                ws = gradnorm.weights.tolist()
                lam_ce, lam_tok, lam_phr, lam_cfg = ws

            # Combined CE + L_tok backward
            L_ce_tok = lam_ce * L_ce + _step_lam_tok * L_tok_val  # [CHANGE: CURRICULUM]
            L_ce_tok.backward()
            L_total = L_ce_tok.detach()

            # DPO (interleaved)
            L_dpo_val = 0.0
            if cfg.use_dpo and dpo_dl is not None and ref_model is not None:
                if not hasattr(train, '_dpo_iter') or train._dpo_iter is None:
                    train._dpo_iter = iter(dpo_dl)
                try:
                    dpo_batch = next(train._dpo_iter)
                except StopIteration:
                    train._dpo_iter = iter(dpo_dl)
                    dpo_batch = next(train._dpo_iter)
                L_dpo = compute_L_dpo(model, ref_model, tokenizer,
                                      dpo_batch, cfg.dpo_beta, device)
                torch.cuda.empty_cache()
                L_dpo.backward()
                L_dpo_val = L_dpo.item()
                L_total   = L_total + L_dpo.detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            scheduler.step()

            epoch_losses["ce"]    += L_ce.item()
            epoch_losses["tok"]   += L_tok_val.item()
            epoch_losses["phr"]   += _phr_display
            epoch_losses["cfg"]   += _cfg_display
            epoch_losses["total"] += L_total.item()
            n_batches   += 1
            global_step += 1

            step_history.append({
                "global_step": global_step,
                "epoch":       epoch + 1,
                "step":        step,
                "ce":          round(L_ce.item(), 5),
                "tok":         round(L_tok_val.item(), 5),
                "total":       round(L_total.item(), 5),
            })

            if step % 10 == 0:
                gn_str = (f" | λ=({lam_ce:.2f},{lam_tok:.2f},{lam_phr:.2f},{lam_cfg:.2f})"
                          if gradnorm else "")
                dpo_v  = f" DPO={L_dpo_val:.4f}" if cfg.use_dpo else ""
                print(f"  Epoch {epoch+1} Step {step}/{len(train_dl)} | "
                      f"CE={L_ce.item():.3f} Tok={L_tok_val.item():.4f} "
                      f"Phr={_phr_display:.4f} Cfg={_cfg_display:.4f}{dpo_v}{gn_str}")

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_ctok, val_cphr, val_ccfg = [], [], []
        val_examples = []   # [CHANGE: SEMANTIC_METRICS] collect for semantic eval

        with torch.no_grad():
            for vbatch in val_dl:
                instructions  = vbatch["instruction"]
                response_texts = vbatch["response_text"]
                for i in range(len(instructions)):
                    prompt_tok = tokenizer(
                        instructions[i],
                        return_tensors="pt",
                        truncation=True,
                        max_length=cfg.max_length - cfg.max_new_tok,
                    ).to(device)
                    out_v = model.generate(
                        input_ids=prompt_tok["input_ids"],
                        attention_mask=prompt_tok["attention_mask"],
                        max_new_tokens=cfg.max_new_tok,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    prompt_len = prompt_tok["input_ids"].size(1)
                    gen  = out_v[0, prompt_len:]
                    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
                    toks = text.upper().split()
                    if not toks:
                        val_ctok.append(0.0)
                        val_cphr.append(0.0)
                        val_ccfg.append(0.0)
                        val_examples.append({
                            "request":   instructions[i],
                            "reference": response_texts[i],
                            "generated": "",
                        })
                        continue
                    val_ctok.append(compute_ctok(toks, vocab))
                    val_cphr.append(compute_cphr(toks, ngram_wl))
                    val_ccfg.append(
                        compute_ccfg_partial(text, cfg_parser) if cfg_parser else 0.0
                    )
                    # [CHANGE: SEMANTIC_METRICS] accumulate for semantic eval
                    val_examples.append({
                        "request":   instructions[i],
                        "reference": response_texts[i],
                        "generated": text,
                    })

        mean_ctok = sum(val_ctok) / len(val_ctok) if val_ctok else 0.
        mean_cphr = sum(val_cphr) / len(val_cphr) if val_cphr else 0.
        mean_ccfg = sum(val_ccfg) / len(val_ccfg) if val_ccfg else 0.
        mean_cbar = (mean_ctok + mean_cphr + mean_ccfg) / 3.0   # [CHANGE: CBAR_CHECKPOINT]
        mean_loss = epoch_losses["total"] / n_batches
        mean_phr_reward = epoch_losses["phr"] / n_batches
        mean_cfg_reward = epoch_losses["cfg"] / n_batches

        # [CHANGE: SEMANTIC_METRICS] compute semantic metrics on val set
        sem, _ = sem_metrics.compute(val_examples)
        mean_slot_f1   = sem["slot_f1"]
        mean_da_f1     = sem["da_f1"]
        mean_halluc    = sem["halluc_pct"]
        mean_bertscore = sem["bertscore"]

        # [CHANGE: SEMANTIC_METRICS] composite checkpoint metric optionally
        # incorporates BERTScore: composite = (1-w)*C_bar + w*BERTScore
        bw = cfg.bertscore_weight
        composite = (1.0 - bw) * mean_cbar + bw * mean_bertscore

        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss={mean_loss:.4f} | "
              f"C_tok={mean_ctok:.4f} C_phr={mean_cphr:.4f} "
              f"C_cfg={mean_ccfg:.4f} C_bar={mean_cbar:.4f} | "
              f"SlotF1={mean_slot_f1:.4f} DA-F1={mean_da_f1:.4f} "
              f"Hall={mean_halluc:.3f} BERT={mean_bertscore:.4f} | "
              f"Train Rphr={mean_phr_reward:.4f} Rcfg={mean_cfg_reward:.4f}")

        record = {
            "epoch":      epoch + 1,
            "phase":      phase_name,   # [CHANGE: CURRICULUM:CL5]
            "loss":       mean_loss,
            "ce_loss":    epoch_losses["ce"]  / n_batches,
            "tok_loss":   epoch_losses["tok"] / n_batches,
            "Rphr":       mean_phr_reward,
            "Rcfg":       mean_cfg_reward,
            "C_tok":      mean_ctok,
            "C_phr":      mean_cphr,
            "C_cfg":      mean_ccfg,
            "C_bar":      mean_cbar,
            # [CHANGE: SEMANTIC_METRICS]
            "slot_f1":    mean_slot_f1,
            "da_f1":      mean_da_f1,
            "halluc_pct": mean_halluc,
            "bertscore":  mean_bertscore,
            "composite":  composite,
            # [CHANGE: GRADNORM]
            "lambda_ce":  lam_ce,
            "lambda_tok": lam_tok,
            "lambda_phr": lam_phr,
            "lambda_cfg": lam_cfg,
        }
        history.append(record)

        # [CHANGE: CBAR_CHECKPOINT + SEMANTIC_METRICS] checkpoint selection
        # Uses composite (C_bar + optional BERTScore weight).
        # [CHANGE: SEMANTIC_METRICS] Hallucination gate: reject checkpoint if
        # halluc_pct exceeds threshold, regardless of compliance improvement.
        current_metric = composite if cfg.checkpoint_metric == "c_bar" else mean_ctok
        halluc_ok = mean_halluc <= cfg.hallucination_threshold

        if not halluc_ok:
            print(f"  ✗ Hallucination gate triggered: {mean_halluc:.3f} > "
                  f"{cfg.hallucination_threshold:.2f} — checkpoint not saved.")
            no_improve += 1
        elif current_metric > best_metric:
            best_metric   = current_metric
            best_val_ctok = mean_ctok
            no_improve    = 0   # [CHANGE: EARLY_STOPPING]
            model.save_pretrained(f"{cfg.output_dir}/best")
            tokenizer.save_pretrained(f"{cfg.output_dir}/best")
            _verify_checkpoint(f"{cfg.output_dir}/best")
            print(f"  ★ New best {cfg.checkpoint_metric}={best_metric:.4f} "
                  f"(C_tok={mean_ctok:.4f} C_phr={mean_cphr:.4f} "
                  f"C_cfg={mean_ccfg:.4f} Hall={mean_halluc:.3f}) saved")
        else:
            no_improve += 1   # [CHANGE: EARLY_STOPPING]

        model.save_pretrained(f"{cfg.output_dir}/last")
        tokenizer.save_pretrained(f"{cfg.output_dir}/last")

        # [CHANGE: EARLY_STOPPING] stop when no improvement for patience epochs
        if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1}: no {cfg.checkpoint_metric} "
                  f"improvement for {no_improve} epochs.")
            break

    if cfg.epochs == 0:
        model.save_pretrained(f"{cfg.output_dir}/best")
        tokenizer.save_pretrained(f"{cfg.output_dir}/best")
        _verify_checkpoint(f"{cfg.output_dir}/best")

    with open(f"{cfg.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"{cfg.output_dir}/step_history.json", "w") as f:
        json.dump(step_history, f, indent=2)
    print("Training complete.")
    return history

# ═════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model_path: str, data_path: str, condition_name: str = "SCOPE",
             domain: str = "atc",
             vocab_path: str = "", phrase_path: str = "", grammar_path: str = "",
             bertscore_model: str = "bert-base-uncased"):  # [CHANGE: SEMANTIC_METRICS]
    # [CHANGE: MULTI_GPU] split model across all available GPUs
    device     = _tensor_device()
    _vocab_p   = Path(vocab_path)   if vocab_path   else VOCAB_PATH
    _phrase_p  = Path(phrase_path)  if phrase_path  else PHRASE_PATH
    _grammar_p = Path(grammar_path) if grammar_path else GRAMMAR_PATH
    vocab      = load_whitelist(_vocab_p)
    ngram_wl   = load_ngram_whitelist(_phrase_p)
    cfg_parser = load_grammar(_grammar_p)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = _load_model(model_path, dtype=torch.bfloat16)
    model.eval()

    with open(data_path) as f:
        test_pairs = json.load(f)

    results  = []
    examples = []   # [CHANGE: SEMANTIC_METRICS]

    for p in test_pairs:
        item = format_atc(p["request"], p["response"], domain=domain)
        tok  = tokenizer(item["instruction"], return_tensors="pt",
                         max_length=512, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**tok, max_new_tokens=64,
                                  do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        gen    = out[0, tok["input_ids"].size(1):]
        text   = tokenizer.decode(gen, skip_special_tokens=True)
        tokens = text.upper().split()
        results.append({
            "request":   p["request"],
            "reference": p["response"],
            "generated": text,
            "C_tok":     compute_ctok(tokens, vocab),
            "C_phr":     compute_cphr(tokens, ngram_wl),
            "C_cfg":     compute_ccfg_partial(text, cfg_parser),
        })
        # [CHANGE: SEMANTIC_METRICS]
        examples.append({
            "request":   p["request"],
            "reference": p["response"],
            "generated": text,
        })

    mean_ctok = sum(r["C_tok"] for r in results) / len(results)
    mean_cphr = sum(r["C_phr"] for r in results) / len(results)
    mean_ccfg = sum(r["C_cfg"] for r in results) / len(results)

    # [CHANGE: SEMANTIC_METRICS] compute semantic metrics on test set
    domain_spec = get_domain_spec(domain)
    sem, sem_per_ex = SemanticMetrics(domain_spec=domain_spec,
                          bertscore_model=bertscore_model).compute(examples)

    # Merge per-example semantic scores into the results list so they are
    # stored in per_example inside test_results.json and visible to diagnostics
    for i, r in enumerate(results):
        if i < len(sem_per_ex):
            r["slot_f1"]  = sem_per_ex[i]["slot_f1"]
            r["da_f1"]    = sem_per_ex[i]["da_f1"]
            r["halluc"]   = sem_per_ex[i]["halluc"]
            r["bertscore"]= sem_per_ex[i]["bertscore"]

    print(f"\n{condition_name} Results ({len(results)} test examples):")
    print(f"  C_tok      = {mean_ctok:.4f}")
    print(f"  C_phr      = {mean_cphr:.4f}")
    print(f"  C_cfg      = {mean_ccfg:.4f}")
    print(f"  C_bar      = {(mean_ctok + mean_cphr + mean_ccfg)/3:.4f}")
    print(f"  Slot-F1    = {sem['slot_f1']:.4f}")
    print(f"  DA-F1      = {sem['da_f1']:.4f}")
    print(f"  Hall.%     = {sem['halluc_pct']:.4f}")
    print(f"  BERTScore  = {sem['bertscore']:.4f}")
    return results, sem   # [CHANGE: SEMANTIC_METRICS] also return sem dict

# ═════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCOPE Training")
    parser.add_argument("--model",       default="gpt2-large")
    parser.add_argument("--data",        default="atc_pairs.json",
                        help="Single combined data file (triggers in-script 80/10/10 split)")
    # [CHANGE: PRE-SPLIT] explicit pre-split files take priority over --data
    parser.add_argument("--train_data",  default="",
                        help="Pre-split training file (use with --val_data for reproducible splits)")
    parser.add_argument("--val_data",    default="",
                        help="Pre-split validation file")
    parser.add_argument("--test_data",   default="")
    parser.add_argument("--output",      default="scope_output")
    parser.add_argument("--lambda_tok",  type=float, default=0.5)
    parser.add_argument("--lambda_phr",  type=float, default=0.3)
    parser.add_argument("--lambda_cfg",  type=float, default=0.2)
    parser.add_argument("--lambda_ce",   type=float, default=1.0)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--M_samples",   type=int,   default=4)
    parser.add_argument("--max_new_tok", type=int,   default=64)
    parser.add_argument("--warmup_steps", type=int,  default=100)
    # [CHANGE: WARMUP_RATIO]
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Warmup as fraction of total steps (overrides --warmup_steps if > 0)")
    # Ablation flags
    parser.add_argument("--no_ltok", action="store_true")
    parser.add_argument("--no_lphr", action="store_true")
    parser.add_argument("--no_lcfg", action="store_true")
    parser.add_argument("--vocab_path",  default="vocab_ATC.json")
    parser.add_argument("--phrase_path", default="ngram_whitelist_ATC.json")
    parser.add_argument("--grammar",     default="G_ATC.lark")
    parser.add_argument("--domain",      default="atc", choices=["atc", "smcp"])
    # DPO
    parser.add_argument("--dpo",       action="store_true")
    parser.add_argument("--dpo_beta",  type=float, default=0.1)
    parser.add_argument("--dpo_ref",   type=str,   default="")
    # Large-model support
    parser.add_argument("--grad_accum",             type=int, default=1)
    parser.add_argument("--use_chat_template",      action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # [CHANGE: CBAR_CHECKPOINT]
    parser.add_argument("--checkpoint_metric",  default="c_bar",
                        choices=["c_bar", "c_tok"],
                        help="Metric for best-checkpoint selection (default: c_bar)")
    # [CHANGE: EARLY_STOPPING]
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop after N epochs without improvement (0 = disabled)")
    # [CHANGE: GRADNORM]
    parser.add_argument("--gradnorm",             action="store_true",
                        help="Enable GradNorm dynamic lambda rebalancing")
    parser.add_argument("--gradnorm_alpha",       type=float, default=1.5)
    parser.add_argument("--gradnorm_update_freq", type=int,   default=20)
    parser.add_argument("--use_8bit_adam",  action="store_true",
                        help="Use bitsandbytes AdamW8bit. Auto-enabled when "
                             "2+ GPUs are detected. pip install bitsandbytes")
    # [CHANGE: CURRICULUM:CL1] curriculum learning flags
    parser.add_argument("--curriculum",           action="store_true",
                        help="Enable curriculum learning: introduce losses "
                             "progressively across epochs. Phase 1: CE only. "
                             "Phase 2: CE+tok. Phase 3: full condition. "
                             "Default: False (identical to scope_train_general.py).")
    parser.add_argument("--curriculum_phase1",    type=float, default=1/3,
                        help="Fraction of total epochs to spend in Phase 1 "
                             "(CE only). Default: 1/3.")
    parser.add_argument("--curriculum_phase2",    type=float, default=2/3,
                        help="Fraction of total epochs up to end of Phase 2 "
                             "(CE+tok). Default: 2/3.")
    parser.add_argument("--curriculum_ramp_steps", type=int, default=0,
                        help="Steps over which to linearly ramp new lambda "
                             "values at each phase transition. 0 = hard switch "
                             "(default). Use 50-100 for GPT-2 stability.")
    # [CHANGE: SEMANTIC_METRICS]
    parser.add_argument("--bertscore_model",        default="bert-base-uncased",
                        help="HuggingFace model for BERTScore (use domain fine-tuned for best results)")
    parser.add_argument("--bertscore_weight",       type=float, default=0.0,
                        help="Weight of BERTScore in composite: (1-w)*C_bar + w*BERTScore (default 0=off)")
    parser.add_argument("--hallucination_threshold", type=float, default=0.10,
                        help="Max tolerated hallucination rate before checkpoint is rejected (default 0.10)")
    # [CHANGE: SEMANTIC_METRICS] domain BERT fine-tuning
    parser.add_argument("--finetune_bert",    action="store_true",
                        help="Fine-tune BERT on ATC corpus via MLM before main training "
                             "(produces domain-adapted BERTScore encoder; cached after first run)")
    parser.add_argument("--bert_mlm_epochs",  type=int,   default=3)
    parser.add_argument("--bert_mlm_batch",   type=int,   default=16)
    parser.add_argument("--bert_mlm_lr",      type=float, default=2e-5)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("✓ HuggingFace authenticated")
        except Exception as e:
            print(f"WARNING: HF login issue: {e} — proceeding")

    cfg = SCOPEConfig(
        model_name               = args.model,
        data_path                = args.data,
        train_data               = args.train_data,   # [CHANGE: PRE-SPLIT]
        val_data                 = args.val_data,      # [CHANGE: PRE-SPLIT]
        output_dir               = args.output,
        lambda_tok               = args.lambda_tok,
        lambda_phr               = args.lambda_phr,
        lambda_cfg               = args.lambda_cfg,
        lambda_ce                = args.lambda_ce,
        epochs                   = args.epochs,
        batch_size               = args.batch_size,
        grad_accum               = args.grad_accum,
        use_chat_template        = args.use_chat_template,
        gradient_checkpointing   = args.gradient_checkpointing,
        lr                       = args.lr,
        seed                     = args.seed,
        M_samples                = args.M_samples,
        max_new_tok              = args.max_new_tok,
        warmup_steps             = args.warmup_steps,
        warmup_ratio             = args.warmup_ratio,         # [CHANGE: WARMUP_RATIO]
        use_ltok                 = not args.no_ltok,
        use_lphr                 = not args.no_lphr,
        use_lcfg                 = not args.no_lcfg,
        use_dpo                  = args.dpo,
        dpo_beta                 = args.dpo_beta,
        dpo_ref_model            = args.dpo_ref,
        domain                   = args.domain,
        vocab_path               = args.vocab_path,
        phrase_path              = args.phrase_path,
        grammar_path             = args.grammar,
        checkpoint_metric        = args.checkpoint_metric,    # [CHANGE: CBAR_CHECKPOINT]
        early_stop_patience      = args.early_stop_patience,  # [CHANGE: EARLY_STOPPING]
        use_gradnorm             = args.gradnorm,             # [CHANGE: GRADNORM]
        gradnorm_alpha           = args.gradnorm_alpha,
        gradnorm_update_freq     = args.gradnorm_update_freq,
        use_8bit_adam            = args.use_8bit_adam,
        # [CHANGE: CURRICULUM:CL1]
        curriculum               = args.curriculum,
        curriculum_phase1        = args.curriculum_phase1,
        curriculum_phase2        = args.curriculum_phase2,
        curriculum_ramp_steps    = args.curriculum_ramp_steps,
        # [CHANGE: SEMANTIC_METRICS]
        bertscore_model          = args.bertscore_model,
        bertscore_weight         = args.bertscore_weight,
        hallucination_threshold  = args.hallucination_threshold,
        finetune_bert            = args.finetune_bert,
        bert_mlm_epochs          = args.bert_mlm_epochs,
        bert_mlm_batch           = args.bert_mlm_batch,
        bert_mlm_lr              = args.bert_mlm_lr,
    )
    train(cfg)

    best_ckpt = Path(cfg.output_dir) / "best"
    if args.test_data:
        test_path = Path(args.test_data)
    else:
        data_p    = Path(cfg.data_path)
        test_name = data_p.name.replace("pairs", "test")
        test_path = data_p.parent / test_name

    if best_ckpt.exists() and test_path.exists():
        print(f"\nEvaluating best checkpoint on test set: {test_path}")
        results, sem = evaluate(   # [CHANGE: SEMANTIC_METRICS]
            model_path     = str(best_ckpt),
            data_path      = str(test_path),
            condition_name = "SCOPE",
            domain         = cfg.domain,
            vocab_path     = cfg.vocab_path,
            phrase_path    = cfg.phrase_path,
            grammar_path   = cfg.grammar_path,
            bertscore_model = cfg.bertscore_model,   # [CHANGE: SEMANTIC_METRICS]
        )
        if results:
            mean_ctok = sum(r["C_tok"] for r in results) / len(results)
            mean_cphr = sum(r["C_phr"] for r in results) / len(results)
            mean_ccfg = sum(r["C_cfg"] for r in results) / len(results)
            summary = {
                "condition":   "SCOPE",
                "model":       cfg.model_name,
                "domain":      cfg.domain,
                "n_test":      len(results),
                "C_tok":       round(mean_ctok, 4),
                "C_phr":       round(mean_cphr, 4),
                "C_cfg":       round(mean_ccfg, 4),
                "C_bar":       round((mean_ctok + mean_cphr + mean_ccfg) / 3, 4),
                # [CHANGE: SEMANTIC_METRICS]
                "slot_f1":     round(sem["slot_f1"],    4),
                "da_f1":       round(sem["da_f1"],      4),
                "halluc_pct":  round(sem["halluc_pct"], 4),
                "bertscore":   round(sem["bertscore"],  4),
                "per_example": results,
            }
            results_path = Path(cfg.output_dir) / "test_results.json"
            with open(results_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Test results saved to {results_path}")
    else:
        if not best_ckpt.exists():
            print(f"  WARNING: best checkpoint not found at {best_ckpt}")
        if not test_path.exists():
            print(f"  WARNING: test data not found at {test_path}")
