"""Educational byte-level BPE tokenizer with optional GPT-4-style regex pre-tokenization."""
from __future__ import annotations
from collections import Counter
import regex as re

# GPT-4 / cl100k_base pre-tokenization pattern (same segmentation expression).
GPT4_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]++|\s+(?!\S)|\s+"

class ByteBPE:
    def __init__(self, use_regex: bool = False):
        self.use_regex = use_regex
        self.pattern = re.compile(GPT4_PATTERN) if use_regex else None
        self.merges: list[tuple[bytes, bytes]] = []
        self.merge_counts: list[int] = []
        self.vocab: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def _pieces(self, text: str) -> list[bytes]:
        parts = self.pattern.findall(text) if self.pattern else [text]
        return [p.encode('utf-8') for p in parts if p]

    @staticmethod
    def _pair_counts(sequences: list[list[bytes]]) -> Counter:
        counts = Counter()
        for seq in sequences:
            counts.update(zip(seq, seq[1:]))
        return counts

    @staticmethod
    def _merge_pair(seq: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                out.append(seq[i] + seq[i + 1]); i += 2
            else:
                out.append(seq[i]); i += 1
        return out

    def train(self, text: str, target_vocab_size: int = 320, max_merges: int | None = None):
        sequences = [[bytes([b]) for b in piece] for piece in self._pieces(text)]
        initial = self._pair_counts(sequences)
        limit = max_merges if max_merges is not None else target_vocab_size - 256
        self.merges = []
        self.merge_counts = []
        for _ in range(max(0, min(limit, target_vocab_size - 256))):
            counts = self._pair_counts(sequences)
            if not counts: break
            # deterministic tie-break for reproducible runs
            pair, count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if count < 2: break
            token = pair[0] + pair[1]
            self.merges.append(pair)
            self.merge_counts.append(count)
            self.vocab[token] = len(self.vocab)
            self.id_to_bytes[self.vocab[token]] = token
            sequences = [self._merge_pair(seq, pair) for seq in sequences]
        self.initial_pair_counts = initial
        return self

    def _encode_piece(self, piece: bytes) -> list[int]:
        seq = [bytes([b]) for b in piece]
        for pair in self.merges:
            seq = self._merge_pair(seq, pair)
        return [self.vocab[token] for token in seq]

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in self._pieces(text): ids.extend(self._encode_piece(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b''.join(self.id_to_bytes[i] for i in ids)
        # Replacement character keeps malformed byte sequences visible without crashing.
        return raw.decode('utf-8', errors='replace')

    def token_bytes(self, idx: int) -> bytes:
        return self.id_to_bytes[idx]
