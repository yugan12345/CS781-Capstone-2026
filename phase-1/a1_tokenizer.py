"""A1 byte-level BPE — fast trainer + save/load."""
import heapq, json
from collections import Counter, defaultdict

SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]
UNK_TOKEN = "<unk>"

def _bytes_to_unicode():
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = keep[:]
    extra = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(keep, chars)}

BYTE2CH = _bytes_to_unicode()
CH2BYTE = {c: b for b, c in BYTE2CH.items()}

def word_to_symbols(word):
    return tuple(BYTE2CH[b] for b in (" " + word).encode("utf-8"))

class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = list(vocab)
        self.merges = dict(merges)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(self.vocab)
        self._cache = {}
        self._id2bytes = [
            b"" if t in SPECIALS else bytes(CH2BYTE[c] for c in t)
            for t in self.vocab
        ]

    def get_vocab_size(self): return self.vocab_size

    def _bpe(self, word):
        if word in self._cache: return self._cache[word]
        tokens = list(word_to_symbols(word))
        while len(tokens) > 1:
            best, best_rank = None, float("inf")
            for i in range(len(tokens) - 1):
                r = self.merges.get((tokens[i], tokens[i + 1]))
                if r is not None and r < best_rank:
                    best, best_rank = (tokens[i], tokens[i + 1]), r
            if best is None: break
            out, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    out.append(tokens[i] + tokens[i + 1]); i += 2
                else:
                    out.append(tokens[i]); i += 1
            tokens = out
        self._cache[word] = tokens
        return tokens

    def encode(self, text, add_special_tokens=False):
        unk = self.token_to_id[UNK_TOKEN]
        ids = []
        if add_special_tokens: ids.append(self.token_to_id["<s>"])
        for word in text.split():
            ids.extend(self.token_to_id.get(t, unk) for t in self._bpe(word))
        if add_special_tokens: ids.append(self.token_to_id["</s>"])
        return ids

    def decode(self, ids):
        table, n = self._id2bytes, len(self._id2bytes)
        raw = b"".join([table[i] for i in ids if 0 <= i < n])
        return raw.decode("utf-8", errors="replace").strip()

    def save(self, path):
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        data = {"specials": SPECIALS, "vocab": self.vocab,
                "merges": [[a, b] for (a, b), _ in ordered]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f: d = json.load(f)
        merges = {(a, b): i for i, (a, b) in enumerate(d["merges"])}
        return cls(d["vocab"], merges)

def train_tokenizer_from_counts(word_counts, vocab_size, log_every=2000):
    items = list(word_counts.items())
    words = [list(word_to_symbols(w)) for w, _ in items]
    freqs = [c for _, c in items]
    pair_counts = defaultdict(int)
    where = defaultdict(set)
    for i, syms in enumerate(words):
        for p in zip(syms, syms[1:]):
            pair_counts[p] += freqs[i]; where[p].add(i)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    vocab = SPECIALS + [BYTE2CH[b] for b in range(256)]
    seen, merges = set(vocab), {}
    while len(vocab) < vocab_size and heap:
        negc, best = heapq.heappop(heap)
        cur = pair_counts.get(best, 0)
        if cur <= 0: continue
        if -negc != cur:
            heapq.heappush(heap, (-cur, best)); continue
        a, b = best
        new_tok = a + b
        merges[best] = len(merges)
        if new_tok not in seen: vocab.append(new_tok); seen.add(new_tok)
        changed = set()
        for i in list(where[best]):
            syms, f = words[i], freqs[i]
            for p in zip(syms, syms[1:]):
                pair_counts[p] -= f; changed.add(p)
            out, j = [], 0
            while j < len(syms):
                if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                    out.append(new_tok); j += 2
                else: out.append(syms[j]); j += 1
            words[i] = out
            for p in zip(out, out[1:]):
                pair_counts[p] += f; where[p].add(i); changed.add(p)
        del where[best]
        for p in changed:
            if pair_counts[p] > 0: heapq.heappush(heap, (-pair_counts[p], p))
        if log_every and len(merges) % log_every == 0:
            print(f"  merges: {len(merges):,}  vocab: {len(vocab):,}", flush=True)
    return BPETokenizer(vocab, merges)
