import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # src shape: (batch_size, src_len)
        # Returns True where padding exists: (batch_size, src_len)
        return (src == self.pad_idx)

    def make_tgt_mask(self, tgt):
        # tgt shape: (batch_size, tgt_len)
        tgt_len = tgt.shape[1]
        tgt_pad_mask = (tgt == self.pad_idx)
        # Causal triangular mask
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device)
        return tgt_pad_mask, tgt_causal_mask

    def forward(self, src, tgt):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)
        src_key_padding_mask = self.make_src_mask(src)
        tgt_key_padding_mask, tgt_mask = self.make_tgt_mask(tgt)

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        # logits shape: (batch_size, tgt_len, tgt_vocab_size)
        logits = self.fc_out(out)
        return logits

    def encode(self, src):
        src_key_padding_mask = self.make_src_mask(src)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode_step(self, tgt, memory, memory_key_padding_mask):
        tgt_key_padding_mask, tgt_mask = self.make_tgt_mask(tgt)
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        logits = self.fc_out(out)
        return logits
