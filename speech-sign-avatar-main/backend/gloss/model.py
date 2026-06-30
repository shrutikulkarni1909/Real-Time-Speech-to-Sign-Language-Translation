import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.encoder = nn.LSTM(
            embed_size, hidden_size, batch_first=True
        )

        self.decoder = nn.LSTM(
            embed_size, hidden_size, batch_first=True
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        """
        src: (batch_size, src_len)
        tgt: (batch_size, tgt_len)
        """

        # ---- Encoder ----
        src_emb = self.embedding(src)
        _, (hidden, cell) = self.encoder(src_emb)

        # ---- Decoder ----
        tgt_emb = self.embedding(tgt)
        decoder_out, _ = self.decoder(tgt_emb, (hidden, cell))

        # ---- Output vocab scores ----
        output = self.fc(decoder_out)
        return output
