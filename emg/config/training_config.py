# training_config.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CONFIG = {
    # ── data ────────────────────────────────────────────────────────────────
    'data_dir':       './data/',
    'save_dir':       './checkpoints/',
    'signal_length':  6000,
    'n_channels':     5,
    'sample_rate':    1000,
    'max_samples':    None,

    # ── model ───────────────────────────────────────────────────────────────
    'hidden_dim':       512,     # encoder output / seq feature dim
    'decoder_hidden':   256,     # phoneme decoder internal width
    'num_gru_layers':   2,
    'dropout':          0.2,

    # ── LM decoding (pyctcdecode + KenLM) ───────────────────────────────────
    # Set lm_path to a .arpa or .bin KenLM model file, or None for no LM.
    # Build one with: kenlm/build/bin/lmplz -o 4 < corpus.txt > lm.arpa
    'lm_path':          None,
    'lm_alpha':         0.5,      # LM weight in beam search
    'lm_beta':          1.0,      # word insertion bonus
    'beam_width':       50,       # beam search width at inference

    # ── training ────────────────────────────────────────────────────────────
    'epochs':           300,
    'batch_size':       8,
    'accumulation_steps': 4,
    'encoder_lr':       1e-3,
    'decoder_lr':       5e-4,
    'weight_decay':     0.01,
    'grad_clip_norm':   1.0,
    'warmup_ratio':     0.1,
    'patience':         10,

    # ── dataset splits ──────────────────────────────────────────────────────
    'train_split':  0.9,
    'val_split':    0.1,
    'seed':         42,

    # ── misc ────────────────────────────────────────────────────────────────
    'mixed_precision': True,
    'num_workers':     4,
}


def get_optimizer_groups(model, config=None):
    if config is None:
        config = CONFIG

    encoder_params = [p for n, p in model.named_parameters()
                      if 'encoder' in n and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters()
                      if 'decoder' in n and p.requires_grad]

    groups = []
    if encoder_params:
        groups.append({'params': encoder_params,
                       'lr': config['encoder_lr'],
                       'weight_decay': config['weight_decay']})
    if decoder_params:
        groups.append({'params': decoder_params,
                       'lr': config['decoder_lr'],
                       'weight_decay': config['weight_decay']})
    return groups