# model.py
import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Optional

from .encoder import Encoder
from .decoder import PhonemeDecoder, PhonemeToWordDecoder, ARPABET_PHONEMES

logger = logging.getLogger(__name__)


class sEMGModel(nn.Module):
    """
    sEMG → sequential encoder → CTC phoneme decoder → (optionally) LM beam search.

    Training  : CTC loss over phoneme targets
    Inference : greedy or beam-search decode → phoneme sequence → text
    """

    def __init__(
        self,
        input_channels:  int = 5,
        hidden_dim:      int = 512,      # encoder output dim (also seq dim)
        decoder_hidden:  int = 256,      # decoder internal width
        num_gru_layers:  int = 2,
        dropout:         float = 0.2,
        lm_path:         Optional[str] = None,   # KenLM .arpa/.bin path
        lm_alpha:        float = 0.5,
        lm_beta:         float = 1.0,
        beam_width:      int = 50,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim     = hidden_dim

        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
        )

        # The CTC decoder reads the 512-channel sequential encoder output
        self.decoder = PhonemeDecoder(
            encoder_dim=512,
            hidden_dim=decoder_hidden,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
            lm_path=lm_path,
            alpha=lm_alpha,
            beta=lm_beta,
            beam_width=beam_width,
        )

        # Phoneme → word segmentation using CMU Pronouncing Dictionary
        self.p2w = PhonemeToWordDecoder(beam_size=20)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"sEMGModel — total params: {total:,} | trainable: {trainable:,}")

    def _to_device_tensor(self, x):
        device = next(self.parameters()).device
        if isinstance(x, list):
            x = torch.stack(x) if isinstance(x[0], torch.Tensor) else \
                torch.tensor(np.array(x), dtype=torch.float32)
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)

    def forward(
        self,
        semg_data:       torch.Tensor,
        phoneme_targets: Optional[torch.Tensor] = None,   # [sum_target_lens]
        target_lengths:  Optional[torch.Tensor] = None,   # [B]
        input_lengths:   Optional[torch.Tensor] = None,   # [B] signal lengths
        **kwargs,
    ) -> dict:
        semg_data = self._to_device_tensor(semg_data)
        if semg_data.dim() == 2:
            semg_data = semg_data.unsqueeze(0)

        seq_features, pooled = self.encoder(semg_data)   # [B,512,T_out], [B,D]

        # guard NaN
        if torch.isnan(seq_features).any():
            seq_features = torch.nan_to_num(seq_features)

        # Compute actual frame lengths after encoder downsampling
        T_out = seq_features.size(2)
        if input_lengths is None:
            input_lengths = torch.full(
                (seq_features.size(0),), T_out,
                dtype=torch.long, device=seq_features.device
            )
        else:
            # Scale original sample lengths by encoder downsampling factor
            ds = getattr(self.encoder, 'downsample_factor', 64)
            input_lengths = (input_lengths.float() / ds).floor().long().clamp(min=1)

        out = self.decoder(
            seq_features=seq_features,
            input_lengths=input_lengths,
            phoneme_targets=phoneme_targets,
            target_lengths=target_lengths,
        )

        out['pooled_features'] = pooled
        out['total_loss']      = out['loss']
        return out

    @torch.no_grad()
    def generate(self, semg_data, use_beam: bool = True, **kwargs):
        semg_data = self._to_device_tensor(semg_data)
        if semg_data.dim() == 2:
            semg_data = semg_data.unsqueeze(0)

        seq_features, _ = self.encoder(semg_data)
        if torch.isnan(seq_features).any():
            seq_features = torch.zeros_like(seq_features)

        out       = self.decoder(seq_features)
        log_probs = out['log_probs']          # [B, T, N_PHONEMES]

        # CTC decode → phoneme sequences
        if use_beam:
            phone_seqs = self.decoder.decode_beam(log_probs)
        else:
            phone_seqs = self.decoder.decode_greedy(log_probs)

        # phoneme sequences → word strings via CMU dict Viterbi
        word_strings = self.p2w.decode_batch(phone_seqs)

        return word_strings   # List[str], one per batch item

    def get_model_info(self):
        return {
            'encoder':  {'input_channels': self.input_channels,
                         'seq_output_dim': 512},
            'decoder':  {'type': 'CTC-phoneme + LM',
                         'n_phonemes': len(ARPABET_PHONEMES),
                         'lm_path': self.decoder.lm_path},
        }