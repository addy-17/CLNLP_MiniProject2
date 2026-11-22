import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.tokenizer import BPETokenizer
from src.data_loader import load_language_pair
from src.model import TransformerNMT
from src.metrics import evaluate_all_metrics

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_tok, tgt_tok, max_len=64):
        self.samples = []
        for src, tgt in pairs:
            src_ids = src_tok.encode(src, add_special_tokens=True)
            tgt_ids = tgt_tok.encode(tgt, add_special_tokens=True)
            if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
                self.samples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, pad_idx=0):
    src_list, tgt_list = zip(*batch)
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)

    padded_src = [s + [pad_idx] * (max_src - len(s)) for s in src_list]
    padded_tgt = [t + [pad_idx] * (max_tgt - len(t)) for t in tgt_list]

    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_tgt, dtype=torch.long)

def train_model(
    lang_key="karbi",
    max_samples=1000,
    vocab_size=600,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    d_model=128,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    output_dir=None
):
    print(f"=== Starting Training for {lang_key.upper()} (Low-Resource NMT) ===")
    pairs = load_language_pair(lang_key, max_samples=max_samples)
    print(f"Loaded {len(pairs)} pairs.")

    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.85)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Train samples: {len(train_pairs)}, Validation samples: {len(val_pairs)}")

    # 1. Feature Extraction & Tokenizer Training
    print("Training source & target BPE tokenizers...")
    src_tok = BPETokenizer(vocab_size=vocab_size)
    src_tok.train([p[0] for p in train_pairs], max_merges=300)

    tgt_tok = BPETokenizer(vocab_size=vocab_size)
    tgt_tok.train([p[1] for p in train_pairs], max_merges=300)

    print(f"Source Vocab Size: {len(src_tok.vocab)}, Target Vocab Size: {len(tgt_tok.vocab)}")

    train_ds = TranslationDataset(train_pairs, src_tok, tgt_tok)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 2. Model Initialization
    model = TransformerNMT(
        src_vocab_size=len(src_tok.vocab),
        tgt_vocab_size=len(tgt_tok.vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=d_model * 2,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 3. Model Training Loop
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for src_b, tgt_b in train_loader:
            src_b = src_b.to(device)
            tgt_b = tgt_b.to(device)

            tgt_input = tgt_b[:, :-1]
            tgt_expected = tgt_b[:, 1:]

            optimizer.zero_grad()
            logits = model(src_b, tgt_input)
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f}")
        history.append({"epoch": epoch, "loss": round(avg_loss, 4)})

    # 4. Comprehensive Evaluation (BLEU, chrF, TER)
    print("Evaluating comprehensive performance metrics on test/validation set...")
    model.eval()
    val_hyps = []
    val_refs = []
    with torch.no_grad():
        for src_text, ref_text in val_pairs[:120]:
            src_ids = torch.tensor([src_tok.encode(src_text)], device=device)
            memory, src_mask = model.encode(src_ids)
            cur_ids = [tgt_tok.sos_id]
            for _ in range(45):
                cur_tensor = torch.tensor([cur_ids], device=device)
                logits = model.decode_step(cur_tensor, memory, src_mask)
                next_id = logits[0, -1, :].argmax().item()
                if next_id == tgt_tok.eos_id:
                    break
                cur_ids.append(next_id)
            hyp_text = tgt_tok.decode(cur_ids)
            val_hyps.append(hyp_text)
            val_refs.append(ref_text)

    eval_scores = evaluate_all_metrics(val_hyps, val_refs)
    print(f"Evaluation Metrics: {eval_scores}")

    # 5. Model Checkpoint & Metadata Persistence
    if not output_dir:
        output_dir = os.path.join("models_saved", lang_key)
    os.makedirs(output_dir, exist_ok=True)

    src_tok.save(os.path.join(output_dir, "src_tok.json"))
    tgt_tok.save(os.path.join(output_dir, "tgt_tok.json"))
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    config = {
        "lang_key": lang_key,
        "src_vocab_size": len(src_tok.vocab),
        "tgt_vocab_size": len(tgt_tok.vocab),
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "metrics": eval_scores,
        "bleu": eval_scores["bleu"],
        "chrf": eval_scores["chrf"],
        "ter": eval_scores["ter"],
        "history": history,
        "sample_eval": [{"src": val_pairs[i][0], "ref": val_refs[i], "hyp": val_hyps[i]} for i in range(min(5, len(val_hyps)))]
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Model and tokenizers saved to {output_dir}")
    return config

if __name__ == "__main__":
    train_model(lang_key="karbi", max_samples=800, epochs=6)
