import json
import re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=600, unk_token="<unk>", pad_token="<pad>", sos_token="<sos>", eos_token="<eos>"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        self.special_tokens = [pad_token, unk_token, sos_token, eos_token]
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3
        
        self.vocab = {}
        self.inv_vocab = {}
        self.merges = []

    def train(self, texts, max_merges=300):
        # Extract word frequencies
        word_counts = Counter()
        for t in texts:
            if not isinstance(t, str):
                continue
            for w in t.strip().split():
                # represent as tuple of chars with trailing boundary marker
                chars = tuple(list(w) + ['</w>'])
                word_counts[chars] += 1

        # Base alphabet
        vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        for word in word_counts:
            for ch in word:
                if ch not in vocab:
                    vocab[ch] = len(vocab)

        merges = []
        target_merges = min(max_merges, max(0, self.vocab_size - len(vocab)))

        for _ in range(target_merges):
            pair_counts = Counter()
            for word, freq in word_counts.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i+1])] += freq

            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break

            merged_token = "".join(best_pair)
            merges.append((best_pair, merged_token))

            # Fast tuple replacement
            p0, p1 = best_pair
            new_word_counts = {}
            for word, freq in word_counts.items():
                new_word = []
                i = 0
                L = len(word)
                while i < L:
                    if i < L - 1 and word[i] == p0 and word[i+1] == p1:
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_counts[tuple(new_word)] = freq
            word_counts = new_word_counts

            if merged_token not in vocab:
                vocab[merged_token] = len(vocab)

        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        return self

    def _tokenize_word(self, word):
        symbols = tuple(list(word) + ['</w>'])
        for (p0, p1), merged in self.merges:
            if p0 in symbols and p1 in symbols:
                new_syms = []
                i = 0
                L = len(symbols)
                while i < L:
                    if i < L - 1 and symbols[i] == p0 and symbols[i+1] == p1:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(symbols[i])
                        i += 1
                symbols = tuple(new_syms)
        return list(symbols)

    def tokenize(self, text):
        if not isinstance(text, str):
            return []
        tokens = []
        for w in text.strip().split():
            tokens.extend(self._tokenize_word(w))
        return tokens

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids):
        tokens = []
        for tid in token_ids:
            if tid in [self.pad_id, self.sos_id, self.eos_id]:
                continue
            tok = self.inv_vocab.get(tid, self.unk_token)
            tokens.append(tok)
            
        text = "".join(tokens).replace("</w>", " ")
        return re.sub(r'\s+', ' ', text).strip()

    def save(self, filepath):
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": [list(pair) + [rep] for pair, rep in self.merges]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.vocab = data["vocab"]
        tok.inv_vocab = {int(v): k for k, v in data["vocab"].items()}
        tok.merges = [((m[0], m[1]), m[2]) for m in data["merges"]]
        return tok
