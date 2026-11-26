"""
GenInherit Colab-Resilient Base Model Trainer (Pillar 2 Specification)
Implements fp16 mixed-precision, float32 loss stabilization, dynamic loss scaling,
cosine learning rate schedule, NaN detection guards, and automatic Drive checkpointing.
"""

import os
import sys
import time
import math
import yaml
import torch
import torch.nn as nn
from typing import Dict, Any

from model.model import GenInheritConfig, GenInheritBase
from train.losses import bio_composite_loss
from train.dataset import StreamingShardLoader


def get_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / (warmup_steps + 1)
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def save_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler, state: Dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "state": state
    }
    torch.save(payload, path)
    print(f"  💾 Saved checkpoint (Step {state['step']:,}, Tokens {state['tokens']:,}) -> {path}")


def load_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler) -> Dict[str, Any]:
    print(f"  🔄 Resuming training from checkpoint: {path}")
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    opt.load_state_dict(payload["opt"])
    if scaler and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    return payload["state"]


def train_loop(config_path: str, data_dir: str, out_dir: str, session_hours: float = 11.0):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  GENINHERIT-BASE TRAINING: {cfg['run']['name'].upper()}")
    print(f"  Device: {device} | Mixed Precision: fp16 (GradScaler) | Loss: float32-stable")
    print(f"  Data Directory: {data_dir}")
    print(f"  Output Checkpoint Directory: {out_dir}")
    print("=" * 80)

    # Initialize Model
    m_cfg = GenInheritConfig(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        head_dim=cfg["model"]["head_dim"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        rope_base=cfg["model"]["rope_base"],
        qk_norm=cfg["model"]["qk_norm"]
    )
    model = GenInheritBase(m_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model Initialized: {total_params / 1e6:.1f} Million Parameters")

    # Optimizer & Scaler
    peak_lr = float(cfg["optim"]["peak_lr"])
    min_lr = float(cfg["optim"].get("min_lr", peak_lr * 0.1))
    warmup_steps = int(cfg["optim"].get("warmup_steps", 500))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=float(cfg["optim"]["weight_decay"])
    )
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # State & Checkpoint
    state = {"step": 0, "tokens": 0, "epoch": 0}
    latest_path = os.path.join(out_dir, "latest.pt")
    if os.path.exists(latest_path):
        try:
            state = load_checkpoint(latest_path, model, opt, scaler)
            # Verify checkpoint has no NaNs
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                print("⚠️ Warning: Existing checkpoint contained NaNs! Removing corrupted file and starting cleanly from step 0.")
                os.remove(latest_path)
                model = GenInheritBase(m_cfg).to(device)
                opt = torch.optim.AdamW(
                    model.parameters(),
                    lr=peak_lr,
                    betas=tuple(cfg["optim"]["betas"]),
                    weight_decay=float(cfg["optim"]["weight_decay"])
                )
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                state = {"step": 0, "tokens": 0, "epoch": 0}
        except Exception as e:
            print(f"⚠️ Could not load checkpoint ({e}). Starting fresh from step 0.")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            state = {"step": 0, "tokens": 0, "epoch": 0}

    # Loader
    loader = StreamingShardLoader(data_dir, micro_batch=cfg["optim"]["micro_batch"])
    batch_gen = loader.stream_batches()

    deadline = time.time() + session_hours * 3600.0 - 600.0  # 10 min safety buffer
    accum_steps = cfg["optim"]["accum"]
    tokens_per_step = cfg["optim"]["micro_batch"] * accum_steps * m_cfg.max_seq_len
    total_steps = int(cfg["optim"]["total_tokens"] // tokens_per_step)

    print(f"Training Loop Running... (Target: {cfg['optim']['total_tokens']:,} tokens across {total_steps:,} steps)")
    t0 = time.time()

    while state["tokens"] < cfg["optim"]["total_tokens"]:
        # Update dynamic learning rate
        current_lr = get_lr(state["step"], warmup_steps, total_steps, peak_lr, min_lr)
        for param_group in opt.param_groups:
            param_group["lr"] = current_lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        nan_detected = False

        for micro_idx in range(accum_steps):
            b = next(batch_gen)
            # Send to device
            tok = b["tok"].long().to(device)
            pos = b["pos"].long().to(device)
            slot = b["slot"].long().to(device)
            gen = b["gen"].long().to(device)
            fld = b["fld"].long().to(device)
            num = b["num"].float().to(device)
            y = b["y"].long().to(device)
            w = b["w"].float().to(device)
            m = b["m"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(tok, pos, slots=slot, gens=gen, fields=fld, num=num)
                # Compute loss in float32 for absolute numerical stability
                loss = bio_composite_loss(logits.float(), y, w, m, lambda_z=float(cfg["loss"]["lambda_z"]))
                scaled_loss = loss / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                nan_detected = True
                break

            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            step_loss += loss.item() / accum_steps

        if nan_detected:
            print(f"⚠️ NaN detected at step {state['step']}! Skipping bad batch to protect model weights.")
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            state["tokens"] += tokens_per_step
            continue

        if use_amp:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        state["step"] += 1
        state["tokens"] += tokens_per_step

        if state["step"] % 10 == 0:
            elapsed = time.time() - t0
            t_speed = (10 * tokens_per_step) / max(0.1, elapsed)
            print(f"Step {state['step']:>6} | Loss: {step_loss:.4f} | LR: {current_lr:.2e} | Tokens: {state['tokens']:>11,} | Speed: {t_speed:>9,.0f} tok/s", flush=True)
            t0 = time.time()

        if state["step"] % cfg["eval"]["ckpt_every"] == 0 or time.time() > deadline:
            save_checkpoint(latest_path, model, opt, scaler, state)
            if time.time() > deadline:
                print("\n⚠️ Colab session time deadline reached. Saved latest state cleanly. Exiting.")
                return state

    # Save final milestone checkpoint
    end_path = os.path.join(out_dir, f"{cfg['run']['name']}_end.pt")
    save_checkpoint(end_path, model, opt, scaler, state)
    print("=" * 80)
    print(f"🎉 Training Milestone Completed! Final Checkpoint: {end_path}")
    print("=" * 80)
    return state
