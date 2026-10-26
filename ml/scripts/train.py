#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import random
import logging
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.model import sEMGModel
from src.data.dataset import sEMGDataset, semg_collate_fn
from config.training_config import CONFIG, get_optimizer_groups

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{ts}.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

def create_model(device: str = 'cpu') -> sEMGModel:
    model = sEMGModel(
        input_channels=CONFIG['n_channels'],
        hidden_dim=CONFIG['hidden_dim'],
        decoder_hidden=CONFIG['decoder_hidden'],
        num_gru_layers=CONFIG['num_gru_layers'],
        dropout=CONFIG['dropout'],
        lm_path=CONFIG.get('lm_path'),
        lm_alpha=CONFIG['lm_alpha'],
        lm_beta=CONFIG['lm_beta'],
        beam_width=CONFIG['beam_width'],
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model ready — total: {total:,}  trainable: {trainable:,}")
    return model

def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss, steps = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", unit="batch")
    for batch in pbar:
        semg = batch['semg'].to(device)
        phoneme_ids = batch['phoneme_ids'].to(device)
        phoneme_lengths = batch['phoneme_lengths'].to(device)
        input_lengths = batch['input_lengths'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            out = model(
                semg_data=semg,
                phoneme_targets=phoneme_ids,
                target_lengths=phoneme_lengths,
                input_lengths=input_lengths,
            )
            loss = out['total_loss']

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_norm'])
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        steps += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / max(steps, 1)


def validate(model, loader, device):
    model.eval()
    total_loss, steps = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            semg = batch['semg'].to(device)
            phoneme_ids = batch['phoneme_ids'].to(device)
            phoneme_lengths = batch['phoneme_lengths'].to(device)
            input_lengths = batch['input_lengths'].to(device)

            out = model(
                semg_data=semg,
                phoneme_targets=phoneme_ids,
                target_lengths=phoneme_lengths,
                input_lengths=input_lengths,
            )
            total_loss += out['total_loss'].item()
            steps += 1

    return total_loss / max(steps, 1)


def generate_samples(model, loader, device, n=3):
    model.eval()
    logger.info("\n─── Qualitative decode samples ───────────────────────────")
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= n:
                break
            semg = batch['semg'].to(device)
            texts = batch['texts']
            for i in range(len(texts)):
                if count >= n:
                    break
                word_strings = model.generate(semg[i:i+1], use_beam=True)
                pred = word_strings[0] if word_strings[0] else "<empty>"
                logger.info(f"  [{count+1}] target : {texts[i]}")
                logger.info(f"        pred   : {pred}")
                count += 1
    logger.info("──────────────────────────────────────────────────────────\n")

def main():
    seed = CONFIG.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    logger.info("Loading dataset …")
    full_ds = sEMGDataset(
        data_dir=CONFIG['data_dir'],
        signal_length=CONFIG['signal_length'],
        n_channels=CONFIG['n_channels'],
        sample_rate=CONFIG['sample_rate'],
        max_samples=CONFIG.get('max_samples'),
    )

    train_n = int(CONFIG['train_split'] * len(full_ds))
    val_n = len(full_ds) - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n])
    logger.info(f"Train: {train_n}  Val: {val_n}")

    nw = 0 if sys.platform == 'win32' else CONFIG.get('num_workers', 4)
    bs = CONFIG['batch_size']

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, collate_fn=semg_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, collate_fn=semg_collate_fn,
        pin_memory=True
    )

    model = create_model(device)
    epochs = CONFIG.get('epochs', 100)
    total_steps = len(train_loader) * epochs

    param_groups = get_optimizer_groups(model)
    optimizer = AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
    warmup_steps = int(CONFIG.get('warmup_ratio', 0.1) * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    amp_scaler = (
        torch.cuda.amp.GradScaler()
        if CONFIG.get('mixed_precision') and torch.cuda.is_available()
        else None
    )

    save_dir = os.path.join(CONFIG['save_dir'], f'run_{ts}')
    os.makedirs(save_dir, exist_ok=True)

    best_val, patience_ctr = float('inf'), 0
    patience = CONFIG.get('patience', 10)

    logger.info("Starting CTC phoneme training …")
    for epoch in range(1, epochs + 1):
        logger.info(f"=== Epoch {epoch}/{epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 amp_scaler, device, epoch)
        val_loss = validate(model, val_loader, device)

        logger.info(f"  train loss : {train_loss:.4f}")
        logger.info(f"  val   loss : {val_loss:.4f}")

        if epoch % 5 == 0:
            generate_samples(model, train_loader, device)

        torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pt'))

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            logger.info(f"  ✓ New best val loss {best_val:.4f} — model saved")
        else:
            patience_ctr += 1
            logger.info(f"  No improvement. Patience {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                logger.info("Early stopping.")
                break 

    logger.info("Training complete.")


if __name__ == "__main__":
    main()