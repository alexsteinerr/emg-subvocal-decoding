#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import random
import logging
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from transformers import AdamW
from tqdm import tqdm
import torch.nn as nn

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), '../../eeg/src/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../eeg/config/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from models.model import EEGModel
from src.ica import IntentConfidenceArbitration
from training_config import CONFIG
from data.dataset import EEGDataset
from data.utils import get_electrode_regions

file_handler = logging.FileHandler(f"training_ica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[stream_handler, file_handler]
)
logger = logging.getLogger(__name__)

def create_main_model(region_channel_counts, device, weights_path=None):
    model = EEGModel(
        region_channel_counts=region_channel_counts,
        hidden_dim=CONFIG['hidden_dim'],
        pretrained_model=CONFIG['pretrained_model'],
        decoder_prefix_len=CONFIG['decoder_prefix_len'],
        use_lora=CONFIG.get('use_lora', True),
        load_in_8bit=CONFIG.get('load_in_8bit', False),
    ).to(device)
    
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def create_ica_model(main_model, device):
    eeg_dim = main_model.hidden_dim
    llm_dim = main_model.decoder.model_dim
    
    ica_model = IntentConfidenceArbitration(
        eeg_dim=eeg_dim,
        llm_dim=llm_dim,
        hidden_dim=512
    ).to(device)
    
    return ica_model

def train_epoch(main_model, ica_model, dataloader, optimizer, device, epoch):
    ica_model.train()
    main_model.eval()
    
    total_loss = 0
    criterion = nn.BCELoss()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Train", unit="batch")
    
    for batch in pbar:
        eeg_data = [region.to(device) for region in batch['eeg']]
        text_ids = batch['decoder_input_ids'].to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            out_pos = main_model(
                eeg_data=eeg_data, 
                input_ids=text_ids, 
                output_hidden_states=True
            )
            eeg_feat_pos = out_pos['eeg_features'].detach()
            hidden_pos = out_pos['hidden_states'][-1].detach()

        score_pos = ica_model(eeg_feat_pos, hidden_pos)
        target_pos = torch.ones_like(score_pos)
        
        shuffled_ids = torch.roll(text_ids, shifts=1, dims=0)
        
        with torch.no_grad():
            out_neg = main_model(
                eeg_data=eeg_data, 
                input_ids=shuffled_ids, 
                output_hidden_states=True
            )
            hidden_neg = out_neg['hidden_states'][-1].detach()
        
        score_neg = ica_model(eeg_feat_pos, hidden_neg)
        target_neg = torch.zeros_like(score_neg)
        
        loss = criterion(score_pos, target_pos) + criterion(score_neg, target_neg)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'pos_conf': f"{score_pos.mean().item():.2f}",
            'neg_conf': f"{score_neg.mean().item():.2f}"
        })
            
    return total_loss / len(dataloader)

def validate(main_model, ica_model, dataloader, device):
    ica_model.eval()
    main_model.eval()
    total_loss = 0
    criterion = nn.BCELoss()
    correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Validating", unit="batch", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            eeg_data = [region.to(device) for region in batch['eeg']]
            text_ids = batch['decoder_input_ids'].to(device)
            
            out_pos = main_model(eeg_data=eeg_data, input_ids=text_ids, output_hidden_states=True)
            score_pos = ica_model(out_pos['eeg_features'], out_pos['hidden_states'][-1])
            
            shuffled_ids = torch.roll(text_ids, shifts=1, dims=0)
            out_neg = main_model(eeg_data=eeg_data, input_ids=shuffled_ids, output_hidden_states=True)
            score_neg = ica_model(out_pos['eeg_features'], out_neg['hidden_states'][-1])
            
            loss = criterion(score_pos, torch.ones_like(score_pos)) + \
                   criterion(score_neg, torch.zeros_like(score_neg))
            
            total_loss += loss.item()
            
            preds_pos = (score_pos > 0.5).float()
            preds_neg = (score_neg < 0.5).float()
            correct += ((preds_pos + preds_neg) == 2.0).sum().item()
            total_samples += len(text_ids)
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
            
    accuracy = correct / total_samples
    return total_loss / len(dataloader), accuracy

def main():
    seed = CONFIG.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    region_indices = get_electrode_regions()
    region_channel_counts = {region: len(indices) for region, indices in region_indices.items()}
    
    temp_model = create_main_model(region_channel_counts, "cpu") 
    tokenizer = temp_model.decoder.tokenizer
    del temp_model
    
    full_dataset = EEGDataset(
        data_dir=CONFIG['data_dir'],
        csv_path=CONFIG['montage_file'],
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        max_samples=CONFIG.get('max_samples', None)
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    num_workers = 0 if sys.platform == 'win32' else CONFIG.get('num_workers', 0)
    batch_size = CONFIG.get('batch_size', 8)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    MAIN_MODEL_PATH = "saved_models/best_model.pt" 
    
    main_model = create_main_model(region_channel_counts, device, weights_path=MAIN_MODEL_PATH)
    ica_model = create_ica_model(main_model, device)
    
    optimizer = AdamW(ica_model.parameters(), lr=1e-4)
    
    save_dir = os.path.join(CONFIG['save_dir'], 'ica_training')
    os.makedirs(save_dir, exist_ok=True)
    
    num_epochs = 10 
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch+1}/{num_epochs} ===")
        
        train_loss = train_epoch(main_model, ica_model, train_loader, optimizer, device, epoch)
        val_loss, val_acc = validate(main_model, ica_model, val_loader, device)
        
        logger.info(f"Results - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ica_model.state_dict(), os.path.join(save_dir, 'best_ica_model.pt'))
            
    logger.info("ICA Training Complete.")

if __name__ == "__main__":
    main()