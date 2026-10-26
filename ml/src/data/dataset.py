import os
import pickle
import logging
import gc
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from functools import lru_cache
from typing import List, Optional
import re

try:
    from ..models.decoder import PHONEME2IDX, ARPABET_PHONEMES
except ImportError:
    PHONEME2IDX = {"<sil>": 0}
    ARPABET_PHONEMES = ["<sil>"]
    logger = logging.getLogger(__name__)
    logger.warning("Could not import decoder; using minimal phoneme set.")

logger = logging.getLogger(__name__)

try:
    from g2p_en import G2p
    _G2P = G2p()
    G2P_AVAILABLE = True
    logger.info("g2p_en loaded for phoneme conversion")
except ImportError:
    G2P_AVAILABLE = False
    _G2P = None
    logger.warning("g2p_en not installed. Run: pip install g2p_en")

SIG_CONF = {
    "sample_rate":       1000,
    "bandpass_low_hz":   0.5,
    "bandpass_high_hz":  8.0,
    "bandpass_order":    1,
    "mains_freq":        50.0,
    "notch_order":       3,
}

class SignalProcessor:
    def __init__(self, fs: float = 1000.0):
        self.fs = float(fs)
        self.sos_hp = signal.butter(1, 0.5, btype="highpass", fs=self.fs, output="sos")
        self.sos_bp = signal.butter(
            SIG_CONF["bandpass_order"],
            [SIG_CONF["bandpass_low_hz"], SIG_CONF["bandpass_high_hz"]],
            btype="bandpass", fs=self.fs, output="sos",
        )
        self.sos_notches = []
        nyq = self.fs / 2.0
        k = 1
        while True:
            fk = k * float(SIG_CONF["mains_freq"])
            if fk >= nyq: break
            self.sos_notches.append(
                signal.butter(SIG_CONF["notch_order"], [fk - 1, fk + 1],
                              btype='bandstop', fs=self.fs, output='sos')
            )
            k += 1

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < x.shape[1]:
            x = x.T
        y = x.astype(np.float64, copy=True)
        y -= y[0, :]
        y = signal.sosfiltfilt(self.sos_hp, y, axis=0)
        y -= np.mean(y, axis=0, keepdims=True)
        for sos in self.sos_notches:
            y = signal.sosfiltfilt(sos, y, axis=0)
        y = signal.sosfiltfilt(self.sos_bp, y, axis=0)
        return y.astype(np.float32)

def text_to_phonemes(text: str) -> List[str]:
    if not G2P_AVAILABLE or _G2P is None:
        raise RuntimeError("g2p_en is required. Install with: pip install g2p_en")
    raw = _G2P(text)
    phones = []
    for tok in raw:
        if tok == ' ' or not tok.isalpha() and not tok[:-1].isalpha():
            continue
        tok_clean = tok.rstrip('012')
        if tok_clean.upper() in PHONEME2IDX:
            phones.append(tok_clean.upper())
        else:
            logger.debug(f"Unknown phoneme: {tok!r} → skipping")
    return phones if phones else ["<sil>"]

def phonemes_to_ids(phones: List[str]) -> List[int]:
    return [PHONEME2IDX.get(p, PHONEME2IDX["<sil>"]) for p in phones]

def _expand_numbers(text: str) -> str:
    try:
        from num2words import num2words
    except ImportError:
        return text

    def _currency(m):
        symbol = m.group(1)
        amount = m.group(2)
        currency_map = {'$': 'dollar', '£': 'pound', '€': 'euro'}
        cur = currency_map.get(symbol, 'dollar')
        try:
            val = float(amount.replace(',', ''))
            major = int(val)
            minor = round((val - major) * 100)
            parts = [num2words(major) + f" {cur}{'s' if major != 1 else ''}"]
            if minor:
                parts.append(f"and {num2words(minor)} cent{'s' if minor != 1 else ''}")
            return ' '.join(parts)
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(r'([\$£€])([\d,]+(?:\.\d+)?)', _currency, text)

    def _ordinal(m):
        try:
            return num2words(int(m.group(1)), to='ordinal')
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', _ordinal, text)

    def _decimal(m):
        try:
            integer_part, decimal_part = m.group(0).split('.')
            words = num2words(int(integer_part))
            digit_words = ' '.join(num2words(int(d)) for d in decimal_part)
            return f"{words} point {digit_words}"
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(r'\b\d+\.\d+\b', _decimal, text)

    def _integer(m):
        try:
            n = int(m.group(0).replace(',', ''))
            return num2words(n)
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(r'\b[\d,]+\b', _integer, text)

    return text

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    text = _expand_numbers(text)
    text = re.sub(r"[^\w\s']", "", text)
    text = text.replace('-', ' ')
    text = " ".join(text.split())
    return text if text else "default text"

class sEMGDataset(Dataset):
    def __init__(
        self,
        data_dir:      str,
        signal_length: int = 6000,
        n_channels:    int = 5,
        sample_rate:   int = 1000,
        max_samples:   Optional[int] = None,
        use_cache:     bool = True,
        cache_size:    int = 128,
        eps:           float = 1e-6,
    ):
        self.signal_length = signal_length
        self.n_channels    = n_channels
        self.eps           = eps
        self.max_samples   = max_samples

        self.signal_processor = SignalProcessor(fs=sample_rate)
        self.data_key  = None
        self.text_key  = None

        self.data_files   = self._get_validated_data_files(data_dir)
        self.sample_index = self._build_sample_index()

        if len(self.sample_index) == 0:
            raise ValueError("No valid samples found in data directory")

        self._initialize_scaler()

        if use_cache:
            self._load_raw = lru_cache(maxsize=cache_size)(self._load_raw_uncached)
        else:
            self._load_raw = self._load_raw_uncached

        logger.info(f"sEMGDataset: {len(self)} samples | {n_channels} ch | g2p={'yes' if G2P_AVAILABLE else 'NO'}")

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        info   = self.sample_index[idx]
        sample = self._load_raw(info['file'], info['index'])

        if sample is None:
            return self._fallback(idx)

        raw_data = self._extract_data(sample)
        if raw_data is None:
            return self._fallback(idx)

        semg = self._process_signal(raw_data)
        if semg is None:
            return self._fallback(idx)

        text   = clean_text(self._extract_text(sample))
        phones = text_to_phonemes(text)
        phone_ids = phonemes_to_ids(phones)

        return {
            'semg':           torch.tensor(semg, dtype=torch.float32),
            'phoneme_ids':    torch.tensor(phone_ids, dtype=torch.long),
            'phoneme_length': torch.tensor(len(phone_ids), dtype=torch.long),
            'text':           text,
        }

    def _fallback(self, idx=0):
        phone_ids = [PHONEME2IDX["<sil>"]]
        return {
            'semg':           torch.zeros(self.n_channels, self.signal_length),
            'phoneme_ids':    torch.tensor(phone_ids, dtype=torch.long),
            'phoneme_length': torch.tensor(1, dtype=torch.long),
            'text':           "",
        }

    def _extract_data(self, sample):
        if self.data_key and self.data_key in sample:
            return sample[self.data_key]
        for k in ['raw_uV', 'proc_uV', 'input_features', 'emg', 'signal', 'data']:
            if k in sample:
                self.data_key = k
                return sample[k]
        logger.error(f"EMG data key not found. Keys: {list(sample.keys())}")
        return None

    def _extract_text(self, sample):
        if self.text_key and self.text_key in sample:
            return sample[self.text_key]
        for k in ['label', 'text', 'sentence', 'transcript', 'caption']:
            if k in sample:
                self.text_key = k
                v = sample[k]
                return " ".join(map(str, v)) if isinstance(v, list) else str(v)
        return ""

    def _process_signal(self, raw_data):
        try:
            x = np.array(raw_data, dtype=np.float32)
            if x.ndim == 1: x = x.reshape(1, -1)
            elif x.ndim > 2: x = x.squeeze()
            x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

            x = self.signal_processor.process(x)
            x = self._adjust_channels(x)

            if self.scaler is not None:
                x = self.scaler.transform(x.T).T
            else:
                mu  = np.mean(x, axis=1, keepdims=True)
                std = np.std(x,  axis=1, keepdims=True) + self.eps
                x   = (x - mu) / std

            x = np.clip(x, -5.0, 5.0)
            return self._pad_or_truncate(x)
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
            return None

    def _adjust_channels(self, x: np.ndarray) -> np.ndarray:
        C, T = x.shape
        if C == self.n_channels: return x
        if C  > self.n_channels: return x[:self.n_channels, :]
        return np.pad(x, ((0, self.n_channels - C), (0, 0)))

    def _pad_or_truncate(self, x: np.ndarray) -> np.ndarray:
        C, T = x.shape
        if T == self.signal_length: return x
        if T  > self.signal_length: return x[:, :self.signal_length]
        return np.pad(x, ((0, 0), (0, self.signal_length - T)))

    def _get_validated_data_files(self, data_dir):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data dir not found: {data_dir}")
        files = sorted([os.path.join(data_dir, f)
                        for f in os.listdir(data_dir) if f.endswith('.pkl')])
        if not files:
            raise ValueError(f"No .pkl files in {data_dir}")
        return files

    def _build_sample_index(self):
        idx, total = [], 0
        for fp in self.data_files:
            try:
                with open(fp, 'rb') as f:
                    loaded = pickle.load(f)
                n = len(loaded) if isinstance(loaded, list) else 1
                for i in range(n):
                    idx.append({'file': fp, 'index': i})
                    total += 1
                    if self.max_samples and total >= self.max_samples:
                        return idx
            except Exception as e:
                logger.warning(f"Failed to index {fp}: {e}")
        return idx

    def _initialize_scaler(self):
        n = min(200, len(self.sample_index))
        indices = np.random.choice(len(self.sample_index), n, replace=False)
        chunks  = []
        for i in indices:
            try:
                info = self.sample_index[i]
                s    = self._load_raw_uncached(info['file'], info['index'])
                raw  = self._extract_data(s)
                if raw is None: continue
                x = np.array(raw, dtype=np.float32)
                if x.ndim == 1: x = x.reshape(1, -1)
                x = self.signal_processor.process(x)
                x = self._adjust_channels(x)
                chunks.append(x)
            except Exception:
                pass
        if chunks:
            data = np.concatenate(chunks, axis=1).T
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(data)
            logger.info(f"Scaler fitted on {data.shape}")
        else:
            self.scaler = None
        gc.collect()

    def _load_raw_uncached(self, file_path: str, sample_idx: int):
        try:
            with open(file_path, 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, list):
                return loaded[sample_idx] if 0 <= sample_idx < len(loaded) else None
            return loaded if sample_idx == 0 else None
        except Exception as e:
            logger.error(f"Load error {file_path}[{sample_idx}]: {e}")
            return None

def semg_collate_fn(batch):
    semg_list       = [b['semg'] for b in batch]
    semg            = torch.stack(semg_list)
    phoneme_lengths = torch.stack([b['phoneme_length'] for b in batch])
    phoneme_ids     = torch.cat(  [b['phoneme_ids']    for b in batch])
    texts           = [b['text'] for b in batch]
    input_lengths   = torch.tensor([s.shape[-1] for s in semg_list], dtype=torch.long)

    return {
        'semg':            semg,
        'phoneme_ids':     phoneme_ids,
        'phoneme_lengths': phoneme_lengths,
        'input_lengths':   input_lengths,
        'texts':           texts,
    }