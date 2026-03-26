"""
ANIMA Phase 5 - Neural Anamnesis Training Loop
================================================

Wires the Neural Anamnesis grey matter service into the bridge training loop.
Trains a learned injection gate so retrieved memories actively improve LM loss.

Architecture flow per step:
    Context pass (no grad):
        tokenize(context) → harness.step() → field_state_after_context
        IF significance > write_threshold:
            write(field_state, significance) → Neural Anamnesis (Rust service)

    Query (no grad):
        query(field_state) → retrieved_memory  [field_dim]

    Target pass (with grad):
        tokenize(target) → base model logits (frozen)
        NeuralAnamnInjector(field_state, retrieved_memory) → blended_field
        Bridge3(blended_field, base_logits) → combined_logits
        Bridge1(hidden_states) → perturbation → field update
        CE loss on combined_logits

Trainable in Phase 5:
    NeuralAnamnInjector  — learned gate + scale for memory injection
    Bridge 3 output bridge — continues from Phase 3 checkpoint
    Bridge 1 input bridge  — continues from Phase 3 checkpoint

NOT trained in Phase 5:
    Neural Anamnesis Rust encoder — no backprop through HTTP.
    Encoder learns to route similar field states to similar sectors via
    offline distillation in Phase 6.

Author: Claude
Date: 2026-03-09
"""

import sys
import os
import time
import asyncio
import argparse
import json
import threading
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmf.core.field import LivingMemoryField
from lmf.configs.default import gpt_oss_20b_config
from lmf.bridges.harness import BridgeHarness
from lmf.bridges.anamnesis_client import NeuralAnamnClient
from lmf.training.training_data import TRAINING_PASSAGES_EXPANDED as TRAINING_PASSAGES


# ═══════════════════════════════════════════════════════════════════════
# Sync wrapper for the async Neural Anamnesis client
# ═══════════════════════════════════════════════════════════════════════

class SyncNeuralAnamnClient:
    def __init__(self, base_url: str = "http://localhost:6060"):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="neural-anamnesis-io"
        )
        self._thread.start()
        self._client: Optional[NeuralAnamnClient] = None
        self._available = False
        self._run(self._start(base_url))

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=10.0)

    async def _start(self, base_url: str):
        self._client = NeuralAnamnClient(base_url=base_url)
        await self._client.connect()
        for _ in range(50):
            if await self._client.health():
                self._available = True
                break
            await asyncio.sleep(0.1)

    def is_available(self) -> bool:
        return self._available and self._client is not None

    def write(self, pattern, significance, emotional_state=None) -> bool:
        if not self.is_available():
            return False
        return self._run(self._client.write(pattern, significance, emotional_state))

    def query(self, field_state, emotional_state=None, top_k=5):
        if not self.is_available():
            return None
        return self._run(self._client.query(field_state, emotional_state, top_k))

    def close(self):
        if self._client:
            self._run(self._client.close())
        self._loop.call_soon_threadsafe(self._loop.stop)


# ═══════════════════════════════════════════════════════════════════════
# Learned Injection Gate
# ═══════════════════════════════════════════════════════════════════════

class NeuralAnamnInjector(nn.Module):
    """
    Learned gate that blends retrieved Neural Anamnesis memory into field state.

    blended = field_state + gate * scale * retrieved_memory

    gate  = sigmoid(use_memory_bias) * sigmoid(gate_net([field || retrieved]))
    scale = exp(log_scale)

    Initialization (v2 — open enough for gradient flow):
        log_scale       = -1.5  →  scale ≈ 0.22  (was -3.0 → 0.05, too small)
        use_memory_bias =  0.0  →  gate  ≈ 0.50  (was -2.0 → 0.12, too closed)

    Why this matters:
        Gradient reaching injector params travels through:
            loss → bridge3 (gamma≈0.1) → blended_field → injection (scale × gate)
        With old init: effective gradient ≈ loss_grad × 0.1 × 0.05 × 0.12 ≈ 0.0006
        With new init: effective gradient ≈ loss_grad × 0.1 × 0.22 × 0.50 ≈ 0.011
        ~18x stronger signal — enough to actually learn.

    Additionally, an alignment auxiliary loss (computed in training_step) provides
    a direct gradient signal that bypasses the gamma attenuation entirely.
    """

    def __init__(self, field_dim: int):
        super().__init__()
        self.field_dim = field_dim

        self.gate_net = nn.Sequential(
            nn.Linear(field_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # v2: more open initialization so gradient actually flows
        self.log_scale = nn.Parameter(torch.tensor(-1.5))   # exp(-1.5) ≈ 0.22
        self.use_memory_bias = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.50

        self._inject_count = 0
        self._skip_count = 0

    @property
    def scale(self) -> float:
        return self.log_scale.exp().item()

    @property
    def use_memory_prob(self) -> float:
        return torch.sigmoid(self.use_memory_bias).item()

    def forward(
        self,
        field_state: torch.Tensor,
        retrieved: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if retrieved is None:
            self._skip_count += 1
            return field_state, torch.zeros(1, device=field_state.device)

        self._inject_count += 1

        fs  = field_state.detach()
        ret = retrieved.to(fs.device)

        combined = torch.cat([fs, ret], dim=-1)

        use_gate     = torch.sigmoid(self.use_memory_bias)
        content_gate = torch.sigmoid(self.gate_net(combined).squeeze(-1))
        gate  = use_gate * content_gate
        scale = self.log_scale.exp()

        blended = field_state + gate * scale * ret
        return blended, gate

    def stats(self) -> dict:
        total = self._inject_count + self._skip_count
        return {
            'inject_count':    self._inject_count,
            'skip_count':      self._skip_count,
            'inject_ratio':    self._inject_count / max(total, 1),
            'scale':           self.scale,
            'use_memory_prob': self.use_memory_prob,
        }


# ═══════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="ANIMA Phase 5: Neural Anamnesis Training")
    parser.add_argument("--steps",            type=int,   default=500)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--lr_injector",      type=float, default=1e-3,
                        help="Higher LR for injector — newly initialized, needs stronger signal")
    parser.add_argument("--kl_weight",        type=float, default=0.1)
    parser.add_argument("--b1_weight",        type=float, default=0.3)
    parser.add_argument("--benefit_weight",   type=float, default=0.5)
    parser.add_argument("--alignment_weight", type=float, default=0.3,
                        help="Weight for direct alignment auxiliary loss on injector")
    parser.add_argument("--grad_accum",       type=int,   default=4)
    parser.add_argument("--warmup",           type=int,   default=30)
    parser.add_argument("--log_every",        type=int,   default=5)
    parser.add_argument("--save_every",       type=int,   default=50)
    parser.add_argument("--write_threshold",  type=float, default=0.45)
    parser.add_argument("--top_k",            type=int,   default=5)
    parser.add_argument("--model_path",       type=str,   default=r"D:\gpt-oss-20b")
    parser.add_argument("--checkpoint",       type=str,   default=None)
    parser.add_argument("--checkpoint_dir",   type=str,   default=None)
    parser.add_argument("--anamnesis_url",    type=str,   default="http://localhost:6060")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# System loading
# ═══════════════════════════════════════════════════════════════════════

def load_system(model_path: str, checkpoint_path: Optional[str] = None):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    t0 = time.time()
    max_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            overhead_gb = 3.5 if total_gb < 20.0 else 3.0
            alloc_gb = max(1, int(total_gb - overhead_gb))
            max_mem[i] = f"{alloc_gb}GiB"
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({total_gb:.0f}GB) — allocating {alloc_gb}GiB")
    max_mem["cpu"] = "80GiB"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        max_memory=max_mem, offload_folder="offload_temp",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    config = gpt_oss_20b_config()
    config.device = "cpu"
    lmf = LivingMemoryField(config)
    harness = BridgeHarness(model=model, lmf=lmf, bridge_device="cpu")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading Phase 3 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        harness.input_bridge.load_state_dict(ckpt['input_bridge'])
        harness.output_bridge.load_state_dict(ckpt['output_bridge'])
        harness.memory_bridge.load_state_dict(ckpt['memory_bridge'])
        harness.lmf.load_state_dict(ckpt.get('lmf', harness.lmf.state_dict()))
        print(f"  Loaded from step {ckpt.get('step', '?')}, "
              f"running improvement: {ckpt.get('running_improvement', 0):+.4f}")
    else:
        print(f"  {'Checkpoint not found: ' + checkpoint_path if checkpoint_path else 'No checkpoint'} — starting fresh")

    return model, tokenizer, lmf, harness


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def tokenize_passage(tokenizer, model, text: str, max_length: int = 512):
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True, return_dict=True,
        max_length=max_length, truncation=True,
    )
    ids  = inputs["input_ids"].to(model.device)
    mask = inputs.get("attention_mask")
    if mask is not None:
        mask = mask.to(model.device)
    return ids, mask


def reset_field(lmf):
    lmf.field_state.zero_()
    for idx in range(lmf.working.max_patterns):
        if lmf.working.active_mask[idx]:
            lmf.working._clear_slot(idx)
    for idx in range(lmf.transient.max_patterns):
        if lmf.transient.active_mask[idx]:
            lmf.transient._clear_slot(idx)


def compute_lr_scale(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())


# ═══════════════════════════════════════════════════════════════════════
# Training step
# ═══════════════════════════════════════════════════════════════════════

def training_step(
    harness, model, tokenizer, lmf,
    injector: NeuralAnamnInjector,
    anamnesis: SyncNeuralAnamnClient,
    passage: dict,
    vocab_size: int,
    pad_token_id: int,
    write_threshold: float,
    top_k: int,
) -> dict:

    # ── Phase A: Context accumulation (no grad) ──────────────────────────
    reset_field(lmf)
    context_ids, context_mask = tokenize_passage(tokenizer, model, passage["context"])
    with torch.no_grad():
        ctx_result = harness.step(context_ids, context_mask)

    field_after_context = lmf.field_state.clone().cpu()

    # Sanitize — NaN/Inf corrupts JSON serialization to Rust service
    field_after_context = torch.nan_to_num(
        field_after_context, nan=0.0, posinf=1.0, neginf=-1.0
    )
    if torch.isnan(field_after_context).any():
        field_after_context = torch.zeros_like(field_after_context)

    field_norm = field_after_context.norm().item()
    significance = ctx_result['significance']
    sig_val = significance.item() if isinstance(significance, torch.Tensor) else significance

    # ── Phase B: Write to Neural Anamnesis ───────────────────────────────
    wrote_to_anamnesis = False
    if sig_val >= write_threshold and anamnesis.is_available():
        wrote_to_anamnesis = anamnesis.write(
            pattern=field_after_context,
            significance=min(sig_val, 1.0),
        )

    # ── Phase C: Query Neural Anamnesis ──────────────────────────────────
    retrieved_memory: Optional[torch.Tensor] = None
    if anamnesis.is_available():
        retrieved_memory = anamnesis.query(field_state=field_after_context, top_k=top_k)

    # ── Phase D: Target pass base (no grad) ──────────────────────────────
    target_ids, target_mask = tokenize_passage(tokenizer, model, passage["target"])
    shift_labels = target_ids[:, 1:].to("cpu").contiguous()

    with torch.no_grad():
        outputs = model(
            input_ids=target_ids, attention_mask=target_mask, output_hidden_states=True,
        )
        base_logits   = outputs.logits.to(device="cpu", dtype=torch.float32)
        hidden_states = outputs.hidden_states[-1].to(device="cpu", dtype=torch.float32)

    def ce_loss(logits):
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
        )

    with torch.no_grad():
        loss_base = ce_loss(base_logits).item()

    # ── Phase E: Injection + Bridge 3 (with grad) ────────────────────────
    blended_field, gate_value = injector(field_after_context, retrieved_memory)
    blended_on_bridge = blended_field.to(harness.bridge_device, dtype=torch.float32)

    bridge3_injected = harness.output_bridge(
        field_state=blended_on_bridge, model_logits=base_logits, return_components=True,
    )
    logits_injected = bridge3_injected['combined_logits']
    loss_injected   = ce_loss(logits_injected)

    # No-inject baseline (no grad — comparison only)
    with torch.no_grad():
        bridge3_no_inject = harness.output_bridge(
            field_state=field_after_context.to(harness.bridge_device, dtype=torch.float32),
            model_logits=base_logits,
        )
        loss_no_inject = ce_loss(bridge3_no_inject['combined_logits']).item()

    # Penalize injection degradation
    benefit_loss = F.relu(loss_injected - loss_no_inject)

    # ── Phase F: Injector alignment auxiliary loss ────────────────────────
    # The gradient path loss→bridge3(gamma≈0.1)→field→injector(scale×gate)
    # is heavily attenuated. This direct loss gives injector params a clean
    # gradient signal independent of that chain.
    #
    # Logic: if retrieved memory is aligned with the current field state
    # (high cosine similarity), it's relevant — the gate should open.
    # We encourage this by: alignment_loss = -cosine_sim.detach() * gate_value
    # So gate_value receives gradient ∝ how aligned retrieved is with field.
    alignment_loss = torch.tensor(0.0)
    if retrieved_memory is not None:
        cos_sim = F.cosine_similarity(
            retrieved_memory.unsqueeze(0),
            field_after_context.unsqueeze(0),
        )  # scalar ∈ (-1, 1)
        # Maximize gate when retrieved is positively aligned with field
        # gate_value retains grad through injector params
        alignment_loss = -(cos_sim.detach().clamp(min=0.0)) * gate_value

    # ── Phase G: Bridge 1 loss ───────────────────────────────────────────
    mask_cpu    = target_mask.to("cpu") if target_mask is not None else None
    bridge1_out = harness.input_bridge(hidden_states, mask_cpu, return_components=True)
    perturbation = bridge1_out['scaled_perturbation'][0]

    updated_field = blended_on_bridge.detach() + perturbation.to(harness.bridge_device)
    bridge3_b1    = harness.output_bridge(field_state=updated_field, model_logits=base_logits)
    loss_bridge1  = ce_loss(bridge3_b1['combined_logits'])

    # ── KL regularization ────────────────────────────────────────────────
    kl_loss = F.kl_div(
        F.log_softmax(logits_injected[:, -1, :], dim=-1),
        F.softmax(base_logits[:, -1, :].detach(), dim=-1),
        reduction='batchmean',
    )

    return {
        'loss_injected':     loss_injected,
        'loss_no_inject':    loss_no_inject,
        'loss_base':         loss_base,
        'loss_bridge1':      loss_bridge1,
        'benefit_loss':      benefit_loss,
        'alignment_loss':    alignment_loss,
        'kl_loss':           kl_loss,
        'field_norm':        field_norm,
        'significance':      sig_val,
        'gate_value':        gate_value.item() if isinstance(gate_value, torch.Tensor) else float(gate_value),
        'wrote_anamnesis':   wrote_to_anamnesis,
        'retrieved':         retrieved_memory is not None,
        'perturbation_norm': bridge1_out['scaled_perturbation'][0].norm().item(),
        'gamma':             bridge3_injected.get('effective_gamma', harness.output_bridge.effective_gamma),
        'working_memories':  lmf.get_status()['working_active'],
    }


# ═══════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════

def train(args):
    print("=" * 72)
    print("ANIMA Phase 5: Neural Anamnesis Training Loop")
    print("=" * 72)

    model, tokenizer, lmf, harness = load_system(args.model_path, args.checkpoint)
    vocab_size   = model.config.vocab_size
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    print(f"\nConnecting to Neural Anamnesis at {args.anamnesis_url}...")
    anamnesis = SyncNeuralAnamnClient(base_url=args.anamnesis_url)
    if anamnesis.is_available():
        print("  ✅ Neural Anamnesis service connected")
    else:
        print("  ⚠️  Neural Anamnesis unavailable — write/query will be skipped")

    injector = NeuralAnamnInjector(field_dim=lmf.field_dim)
    print(f"\nNeuralAnamnInjector v2: {sum(p.numel() for p in injector.parameters()):,} params")
    print(f"  Initial scale: {injector.scale:.3f}  use_memory_prob: {injector.use_memory_prob:.3f}")

    # Two param groups — injector gets 50x higher LR than bridges
    bridge_params = []
    for p in harness.input_bridge.parameters():
        if p.requires_grad: bridge_params.append(p)
    for name, p in harness.output_bridge.named_parameters():
        if p.requires_grad and 'lm_head' not in name: bridge_params.append(p)
    for p in harness.lmf.parameters():
        if p.requires_grad: bridge_params.append(p)
    injector_params = list(injector.parameters())

    optimizer = AdamW([
        {'params': bridge_params,   'lr': args.lr,          'initial_lr': args.lr},
        {'params': injector_params, 'lr': args.lr_injector, 'initial_lr': args.lr_injector},
    ], weight_decay=0.01)

    print(f"Bridge params:   {sum(p.numel() for p in bridge_params):,}  LR={args.lr}")
    print(f"Injector params: {sum(p.numel() for p in injector_params):,}  LR={args.lr_injector}")

    ckpt_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints'
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(ckpt_dir, "phase5_training_log.jsonl")
    log_file = open(log_path, "w")

    print(f"\nConfig:")
    print(f"  Steps: {args.steps}  |  LR bridges: {args.lr}  |  LR injector: {args.lr_injector}")
    print(f"  Write threshold: {args.write_threshold}  |  Benefit weight: {args.benefit_weight}")
    print(f"  Alignment weight: {args.alignment_weight}  |  KL weight: {args.kl_weight}")
    print(f"  Grad accum: {args.grad_accum}  |  Warmup: {args.warmup}")
    print()

    print(
        f"{'Step':>5} | {'L-inj':>7} | {'L-base':>7} | {'Impr':>7} | "
        f"{'Align':>6} | {'KL':>6} | {'Gate':>6} | {'Scale':>6} | "
        f"{'Wrote':>5} | {'Retr':>4}"
    )
    print("-" * 82)

    best_improvement   = -float("inf")
    running_improvement = 0.0
    total_writes       = 0
    total_retrievals   = 0

    for step in range(1, args.steps + 1):
        passage  = TRAINING_PASSAGES[(step - 1) % len(TRAINING_PASSAGES)]
        lr_scale = compute_lr_scale(step, args.warmup, args.steps)
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * lr_scale

        try:
            result = training_step(
                harness=harness, model=model, tokenizer=tokenizer, lmf=lmf,
                injector=injector, anamnesis=anamnesis, passage=passage,
                vocab_size=vocab_size, pad_token_id=pad_token_id,
                write_threshold=args.write_threshold, top_k=args.top_k,
            )
        except Exception as e:
            print(f"  Step {step} ERROR: {e}")
            import traceback; traceback.print_exc()
            # Flush CUDA cache on OOM errors to recover from fragmentation
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                reset_field(lmf)
                print(f"  ↳ CUDA cache cleared, continuing...")
            continue

        total_loss = (
            result['loss_injected']
            + args.b1_weight       * result['loss_bridge1']
            + args.kl_weight       * result['kl_loss']
            + args.benefit_weight  * result['benefit_loss']
            + args.alignment_weight * result['alignment_loss']
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  Step {step} WARNING: NaN/Inf loss — skipping backward")
            optimizer.zero_grad()
            reset_field(lmf)
            continue

        (total_loss / args.grad_accum).backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(bridge_params + injector_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            # Periodic CUDA cache flush to prevent fragmentation buildup
            if step % 50 == 0:
                torch.cuda.empty_cache()

        improvement = (result['loss_base'] -
                      result['loss_injected'].item())
        running_improvement = 0.9 * running_improvement + 0.1 * improvement
        if result['wrote_anamnesis']: total_writes      += 1
        if result['retrieved']:       total_retrievals  += 1
        if improvement > best_improvement: best_improvement = improvement

        if step % args.log_every == 0 or step == 1:
            print(
                f"{step:>5} | "
                f"{result['loss_injected'].item():>7.4f} | "
                f"{result['loss_base']:>7.4f} | "
                f"{improvement:>+7.4f} | "
                f"{result['alignment_loss'].item() if isinstance(result['alignment_loss'], torch.Tensor) else result['alignment_loss']:>6.4f} | "
                f"{result['kl_loss'].item():>6.4f} | "
                f"{result['gate_value']:>6.4f} | "
                f"{injector.scale:>6.4f} | "
                f"{'✅' if result['wrote_anamnesis'] else '  ':>5} | "
                f"{'✅' if result['retrieved'] else '  ':>4}"
            )

        log_entry = {
            'step':               step,
            'loss_injected':      result['loss_injected'].item(),
            'loss_no_inject':     result['loss_no_inject'],
            'loss_base':          result['loss_base'],
            'loss_bridge1':       result['loss_bridge1'].item(),
            'benefit_loss':       result['benefit_loss'].item(),
            'alignment_loss':     result['alignment_loss'].item() if isinstance(result['alignment_loss'], torch.Tensor) else result['alignment_loss'],
            'kl_loss':            result['kl_loss'].item(),
            'total_loss':         total_loss.item(),
            'improvement':        improvement,
            'running_improvement':running_improvement,
            'gate_value':         result['gate_value'],
            'injector_scale':     injector.scale,
            'use_memory_prob':    injector.use_memory_prob,
            'field_norm':         result['field_norm'],
            'significance':       result['significance'],
            'wrote_anamnesis':    result['wrote_anamnesis'],
            'retrieved':          result['retrieved'],
            'total_writes':       total_writes,
            'total_retrievals':   total_retrievals,
            'gamma':              result['gamma'],
            'perturbation_norm':  result['perturbation_norm'],
            'working_memories':   result['working_memories'],
            'lr_bridge':          optimizer.param_groups[0]['lr'],
            'lr_injector':        optimizer.param_groups[1]['lr'],
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        if step % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"phase5_step_{step}.pt")
            torch.save({
                'step': step, 'phase': 5,
                'input_bridge':        harness.input_bridge.state_dict(),
                'output_bridge':       harness.output_bridge.state_dict(),
                'memory_bridge':       harness.memory_bridge.state_dict(),
                'lmf':                 harness.lmf.state_dict(),
                'injector':            injector.state_dict(),
                'optimizer':           optimizer.state_dict(),
                'running_improvement': running_improvement,
                'best_improvement':    best_improvement,
                'injector_stats':      injector.stats(),
            }, ckpt_path)
            print(f"  ★ Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(ckpt_dir, "phase5_final.pt")
    torch.save({
        'step': step, 'phase': 5,
        'input_bridge':        harness.input_bridge.state_dict(),
        'output_bridge':       harness.output_bridge.state_dict(),
        'memory_bridge':       harness.memory_bridge.state_dict(),
        'lmf':                 harness.lmf.state_dict(),
        'injector':            injector.state_dict(),
        'optimizer':           optimizer.state_dict(),
        'running_improvement': running_improvement,
        'best_improvement':    best_improvement,
        'injector_stats':      injector.stats(),
    }, final_path)

    log_file.close()
    anamnesis.close()

    print()
    print("=" * 72)
    print("PHASE 5 TRAINING COMPLETE")
    print(f"  Best improvement:        {best_improvement:+.4f}")
    print(f"  Running improvement:     {running_improvement:+.4f}")
    print(f"  Total Anamnesis writes:  {total_writes}")
    print(f"  Total Anamnesis queries: {total_retrievals}")
    print(f"  Injector scale (final):  {injector.scale:.4f}")
    print(f"  Use-memory prob (final): {injector.use_memory_prob:.4f}")
    print(f"  Final checkpoint:        {final_path}")
    print(f"  Training log:            {log_path}")
    print("=" * 72)


if __name__ == "__main__":
    args = parse_args()
    train(args)
