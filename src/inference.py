import os
import json
import torch
import torch.nn.functional as F
from src.tokenizer import BPETokenizer
from src.model import TransformerNMT

class TranslationEngine:
    def __init__(self, models_dir="models_saved"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def list_available_models(self):
        if not os.path.exists(self.models_dir):
            return []
        subdirs = [d for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))]
        available = []
        for d in subdirs:
            cfg_path = os.path.join(self.models_dir, d, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                available.append(cfg)
        return available

    def load_model(self, lang_key):
        if lang_key in self.loaded_models:
            return self.loaded_models[lang_key]

        model_path = os.path.join(self.models_dir, lang_key)
        cfg_path = os.path.join(model_path, "config.json")
        weights_path = os.path.join(model_path, "model.pt")
        src_tok_path = os.path.join(model_path, "src_tok.json")
        tgt_tok_path = os.path.join(model_path, "tgt_tok.json")

        if not all(os.path.exists(p) for p in [cfg_path, weights_path, src_tok_path, tgt_tok_path]):
            raise FileNotFoundError(f"Model files for '{lang_key}' are incomplete or missing.")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        src_tok = BPETokenizer.load(src_tok_path)
        tgt_tok = BPETokenizer.load(tgt_tok_path)

        model = TransformerNMT(
            src_vocab_size=cfg.get("src_vocab_size", len(src_tok.vocab)),
            tgt_vocab_size=cfg.get("tgt_vocab_size", len(tgt_tok.vocab)),
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_encoder_layers=cfg.get("num_encoder_layers", 2),
            num_decoder_layers=cfg.get("num_decoder_layers", 2),
            dim_feedforward=cfg.get("d_model", 128) * 2
        ).to(self.device)

        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()

        self.loaded_models[lang_key] = {
            "model": model,
            "src_tok": src_tok,
            "tgt_tok": tgt_tok,
            "config": cfg
        }
        return self.loaded_models[lang_key]

    def translate(self, text, lang_key="karbi", max_steps=50):
        m_bundle = self.load_model(lang_key)
        model = m_bundle["model"]
        src_tok = m_bundle["src_tok"]
        tgt_tok = m_bundle["tgt_tok"]

        src_tokens = src_tok.tokenize(text)
        src_ids = src_tok.encode(text, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            memory, src_mask = model.encode(src_tensor)
            cur_ids = [tgt_tok.sos_id]
            step_confidences = []

            for _ in range(max_steps):
                cur_tensor = torch.tensor([cur_ids], dtype=torch.long, device=self.device)
                logits = model.decode_step(cur_tensor, memory, src_mask)
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                best_prob, next_id_tensor = probs.max(dim=-1)
                next_id = next_id_tensor.item()

                if next_id == tgt_tok.eos_id:
                    break
                cur_ids.append(next_id)
                step_confidences.append(float(best_prob.item()))

        translated_text = tgt_tok.decode(cur_ids)
        target_tokens = [tgt_tok.inv_vocab.get(tid, "<unk>") for tid in cur_ids[1:]]

        # Calculate average confidence
        avg_confidence = round(sum(step_confidences) / max(len(step_confidences), 1) * 100, 1)

        return {
            "source_text": text,
            "translated_text": translated_text,
            "source_tokens": src_tokens,
            "target_tokens": target_tokens,
            "confidence": avg_confidence,
            "lang_key": lang_key
        }
