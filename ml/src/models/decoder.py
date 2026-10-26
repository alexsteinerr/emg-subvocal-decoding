# decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

ARPABET_PHONEMES = [
    # idx 0-39: real phones (passed to pyctcdecode)
    "<sil>",    # idx 0  – silence / padding
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B",  "CH", "D",  "DH", "EH", "ER",
    "EY", "F",  "G",  "HH", "IH", "IY",
    "JH", "K",  "L",  "M",  "N",  "NG",
    "OW", "OY", "P",  "R",  "S",  "SH",
    "T",  "TH", "UH", "UW", "V",  "W",
    "Y",  "Z",  "ZH",
    "<blank>",
]
PHONEME2IDX   = {p: i for i, p in enumerate(ARPABET_PHONEMES)}
N_PHONEMES    = len(ARPABET_PHONEMES)   # 41
CTC_BLANK_IDX = PHONEME2IDX["<blank>"] # 40

_DECODER_LABELS = ARPABET_PHONEMES[:-1]  # 40 entries

class TemporalCNNBlock(nn.Module):
    """Dilated causal-ish conv block for sequence modelling."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv  = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.norm  = nn.BatchNorm1d(channels)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(0.1)

    def forward(self, x):          # x: [B, C, T]
        return self.drop(self.act(self.norm(self.conv(x)))) + x


class PhonemeDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim:   int = 512,     # channels from the encoder's sequential output
        hidden_dim:    int = 256,
        num_gru_layers: int = 2,
        dropout:       float = 0.2,
        lm_path:       Optional[str] = None,   # path to KenLM .arpa or .bin
        alpha:         float = 0.5,            # LM weight in beam search
        beta:          float = 1.0,            # word insertion bonus
        beam_width:    int = 50,
    ):
        super().__init__()
        self.n_phonemes = N_PHONEMES
        self.lm_path    = lm_path
        self.alpha      = alpha
        self.beta       = beta
        self.beam_width = beam_width

        self.input_proj = nn.Sequential(
            nn.Conv1d(encoder_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.tcn = nn.Sequential(
            TemporalCNNBlock(hidden_dim, 3, 1),
            TemporalCNNBlock(hidden_dim, 3, 2),
            TemporalCNNBlock(hidden_dim, 3, 4),
            TemporalCNNBlock(hidden_dim, 3, 8),
        )

        self.bigru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )
        self.gru_norm = nn.LayerNorm(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, N_PHONEMES),
        )

        self._init_weights()
        self._decoder = None   # lazy-loaded pyctcdecode decoder

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        seq_features: torch.Tensor,          # [B, C_enc, T]
        input_lengths: Optional[torch.Tensor] = None,   # [B] original T values
        phoneme_targets: Optional[torch.Tensor] = None, # [sum_target_lens]
        target_lengths:  Optional[torch.Tensor] = None, # [B]
    ) -> dict:
        x = self.input_proj(seq_features)    # [B, hidden, T]
        x = self.tcn(x)                      # [B, hidden, T]
        x = x.transpose(1, 2)               # [B, T, hidden]

        if input_lengths is not None:
            # pack for variable-length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.bigru(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            x, _ = self.bigru(x)             # [B, T, hidden*2]

        x = self.gru_norm(x)
        logits    = self.classifier(x)       # [B, T, N_PHONEMES]
        log_probs = F.log_softmax(logits, dim=-1)

        out_lengths = input_lengths if input_lengths is not None \
                      else torch.full((seq_features.size(0),), seq_features.size(2),
                                      dtype=torch.long, device=seq_features.device)

        result = {'log_probs': log_probs, 'input_lengths': out_lengths}

        if phoneme_targets is not None and target_lengths is not None:
            # CTC expects [T, B, C]
            lp_ctc = log_probs.transpose(0, 1)
            loss = F.ctc_loss(
                lp_ctc, phoneme_targets,
                out_lengths, target_lengths,
                blank=CTC_BLANK_IDX,
                reduction='mean',
                zero_infinity=True,
            )
            result['loss'] = loss
        else:
            result['loss'] = torch.tensor(0.0, device=seq_features.device)

        return result

    def _get_ctc_decoder(self):
        """Lazy-load pyctcdecode beam-search decoder (with optional KenLM)."""
        if self._decoder is not None:
            return self._decoder
        try:
            from pyctcdecode import build_ctcdecoder
            if self.lm_path:
                import kenlm  # noqa: F401  – just a check
                self._decoder = build_ctcdecoder(
                    _DECODER_LABELS,          # 40 labels; pyctcdecode appends blank at idx 40
                    kenlm_model=self.lm_path,
                    alpha=self.alpha,
                    beta=self.beta,
                )
                logger.info(f"pyctcdecode decoder built with KenLM: {self.lm_path}")
            else:
                self._decoder = build_ctcdecoder(_DECODER_LABELS)
                logger.info("pyctcdecode decoder built without LM (greedy-ish beam)")
        except ImportError:
            logger.warning("pyctcdecode not installed – falling back to greedy decode")
            self._decoder = None
        return self._decoder

    @torch.no_grad()
    def decode_greedy(self, log_probs: torch.Tensor) -> List[List[str]]:
        """
        Simple greedy CTC collapse.
        log_probs: [B, T, N_PHONEMES]
        Returns list of phoneme sequences per batch item.
        """
        ids = log_probs.argmax(dim=-1).cpu().tolist()  # [B, T]
        results = []
        for seq in ids:
            phones, prev = [], -1
            for idx in seq:
                if idx != CTC_BLANK_IDX and idx != prev:
                    phones.append(ARPABET_PHONEMES[idx])
                prev = idx
            results.append(phones)
        return results

    @staticmethod
    def _retokenize_phonemes(concatenated: str) -> List[str]:
        """
        pyctcdecode concatenates label strings directly, so ARPAbet tokens
        like 'Y','AO','R' become 'YAOR'.  Recover the original tokens using
        greedy longest-match against the known ARPAbet inventory.

        e.g. 'YAORFOWN' → ['Y', 'AO', 'R', 'F', 'OW', 'N']
        """
        # Build lookup set once (sorted longest-first so we prefer 'SH' over 'S')
        vocab = sorted(
            [p for p in ARPABET_PHONEMES if p not in ('<blank>', '<sil>')],
            key=len, reverse=True
        )
        s, tokens = concatenated.upper(), []
        while s:
            matched = False
            for ph in vocab:
                if s.startswith(ph):
                    tokens.append(ph)
                    s = s[len(ph):]
                    matched = True
                    break
            if not matched:
                tokens.append(s[0])   # unknown char
                s = s[1:]
        return tokens

    @torch.no_grad()
    def decode_beam(self, log_probs: torch.Tensor) -> List[List[str]]:
        # If no KenLM is loaded, greedy is equally good and avoids the
        # pyctcdecode concatenation issue entirely.
        if self.lm_path is None:
            return self.decode_greedy(log_probs)

        decoder = self._get_ctc_decoder()
        lp_np   = log_probs.cpu().numpy()

        results = []
        if decoder is not None:
            for i in range(lp_np.shape[0]):
                raw = decoder.decode(lp_np[i], beam_width=self.beam_width)
                # re-tokenize the concatenated phoneme string back into tokens
                phones = self._retokenize_phonemes(raw) if raw else []
                results.append(phones)
        else:
            results = self.decode_greedy(log_probs)
        return results

    @torch.no_grad()
    def phones_to_text(self, phoneme_seqs: List[List[str]]) -> List[str]:
        """
        Convert a list of phoneme sequences to space-separated strings
        for downstream word-level language model rescoring or direct output.
        """
        return [" ".join(p) for p in phoneme_seqs]


class PhonemeToWordDecoder:
    """
    Converts a flat ARPAbet phoneme sequence into a word string.

    Algorithm
    ---------
    1. Build a reverse map:  (PH1, PH2, ...) → [word1, word2, ...]
       from the CMU Pronouncing Dictionary.
    2. Run Viterbi beam search over the phoneme sequence:
       at each position try all dictionary entries that match the next
       N phonemes; keep the top-K partial hypotheses scored by a
       simple unigram log-probability (word frequency from CMU counts).
    3. Return the best complete hypothesis.
    """

    def __init__(self, beam_size: int = 20, max_word_phones: int = 12):
        self.beam_size       = beam_size
        self.max_word_phones = max_word_phones
        self._cmu_reverse: Optional[dict] = None   # (phones,) → [word, ...]
        self._unigram:      Optional[dict] = None   # word → log_prob
        self._build_cmu_index()

    _STOPLIST = {
        # common CMU junk that shadows real words
        'fone', 'phon', 'trois', 'ertha', 'der', 'dhe', 'bey', 'gae',
        'tew', 'wun', 'iz', 'yer', 'tha', 'fer', 'ta', 'thru', 'thro',
        # single-letter CMU entries that collide with real phones
        'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    }

    def _build_cmu_index(self):
        """
        Build CMU reverse index with real English word frequencies.

        Frequency priority (highest wins):
          1. wordfreq package  (best, covers 1M+ word corpus)
          2. NLTK Brown corpus (good, ~1M word general corpus)
          3. Zipf-law heuristic on word length (fallback, no extra data)

        Any word in _STOPLIST or containing non-ASCII / digits is excluded.
        """
        import math

        try:
            import nltk
            try:
                from nltk.corpus import cmudict
                entries = cmudict.entries()
            except LookupError:
                nltk.download('cmudict', quiet=True)
                from nltk.corpus import cmudict
                entries = cmudict.entries()
        except ImportError:
            logger.warning("NLTK not installed. Run: pip install nltk")
            self._cmu_reverse = {}
            self._unigram     = {}
            return

        word_freq: dict = {}

        # Try wordfreq first (most accurate)
        try:
            from wordfreq import word_frequency
            _wordfreq_available = True
            logger.info("Using wordfreq for word probabilities")
        except ImportError:
            _wordfreq_available = False

        # Try NLTK Brown corpus next
        _brown_freq: dict = {}
        if not _wordfreq_available:
            try:
                try:
                    from nltk.corpus import brown
                    words = brown.words()
                except LookupError:
                    nltk.download('brown', quiet=True)
                    from nltk.corpus import brown
                    words = brown.words()
                for w in words:
                    w_lower = w.lower()
                    _brown_freq[w_lower] = _brown_freq.get(w_lower, 0) + 1
                _brown_total = sum(_brown_freq.values())
                logger.info(f"Using NLTK Brown corpus for word probabilities "
                            f"({len(_brown_freq):,} unique words)")
            except Exception:
                logger.info("Brown corpus unavailable — using Zipf heuristic")

        # build reverse phoneme index, skipping junk words #
        reverse: dict = {}
        all_words:  set = set()

        for word, phones in entries:
            # Filter: ASCII letters only, not in stoplist, not too short
            if (not word.isalpha() or
                    not word.isascii() or
                    word in self._STOPLIST or
                    len(word) < 2):
                continue

            key = tuple(p.rstrip('012') for p in phones)
            reverse.setdefault(key, []).append(word)
            all_words.add(word)

        self._cmu_reverse = reverse

        # assign log-probabilities #
        unigram: dict = {}
        for word in all_words:
            if _wordfreq_available:
                # wordfreq returns a true probability in [0,1]
                freq = word_frequency(word, 'en', minimum=1e-9)
                unigram[word] = math.log(freq)

            elif _brown_freq:
                # Brown corpus count → probability; floor at 0.1 counts
                count = _brown_freq.get(word, 0.1)
                unigram[word] = math.log(count / (_brown_total + 1))

            else:

                length_score = -len(word) + (2 if len(word) <= 6 else 0)
                unigram[word] = float(length_score)

        self._unigram = unigram
        logger.info(f"CMU index ready: {len(reverse):,} phone sequences, "
                    f"{len(all_words):,} words")

    def decode(self, phonemes: List[str]) -> str:
        """
        Segment a flat phoneme list into the most likely word sequence.
        """
        if not phonemes or not self._cmu_reverse:
            return " ".join(phonemes)   # fallback: return raw phones

        # Strip stress if present
        phones = [p.rstrip('012') for p in phonemes]
        N      = len(phones)

        import heapq

        # heap items: (neg_score, words_so_far, current_position)
        heap  = [(0.0, [], 0)]
        best  = {}   # position → (score, words)

        while heap:
            neg_score, words, pos = heapq.heappop(heap)
            score = -neg_score

            if pos == N:
                # Complete hypothesis
                return " ".join(words)

            # only keep beam_size hypotheses per position
            if pos in best and best[pos][0] >= score:
                continue
            best[pos] = (score, words)

            # Try all spans from pos of length 1 … max_word_phones
            max_span = min(self.max_word_phones, N - pos)
            for span in range(1, max_span + 1):
                key = tuple(phones[pos:pos + span])
                if key in self._cmu_reverse:
                    for word in self._cmu_reverse[key]:
                        word_score  = self._unigram.get(word, -10.0)
                        new_score   = score + word_score
                        new_words   = words + [word]
                        new_pos     = pos + span
                        heapq.heappush(heap, (-new_score, new_words, new_pos))

            # prevents the decoder from getting stuck on unknown phones.
            oov_score = score - 5.0   # penalise OOV heavily
            heapq.heappush(heap, (-oov_score, words + [phones[pos]], pos + 1))

        # If beam exhausted without reaching N (shouldn't happen due to OOV)
        if N in best:
            return " ".join(best[N][1])
        # Last resort
        return " ".join(phones)

    def decode_batch(self, phoneme_seqs: List[List[str]]) -> List[str]:
        """Decode a batch of phoneme sequences to word strings."""
        return [self.decode(phones) for phones in phoneme_seqs]