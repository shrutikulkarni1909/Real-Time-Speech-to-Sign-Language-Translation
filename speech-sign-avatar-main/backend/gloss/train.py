import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import load_dataset_for_training, load_daily_pairs
from model import Seq2Seq
from tokenizer import SimpleTokenizer


def normalize_english(text):
    text = text.lower()
    remove = ["is", "am", "are", "the", "a", "an", "was", "were"]
    return " ".join([w for w in text.split() if w not in remove])


def pad_sequence(seq, max_len):
    return seq + [0] * (max_len - len(seq))


def train_model(csv_path, epochs=2, batch_size=8):

    # 1️⃣ Load datasets
    main_pairs = load_dataset_for_training(csv_path)
    daily_pairs = load_daily_pairs("backend/gloss/daily_pairs.txt")

    # Normalize
    pairs = []
    for e, g in daily_pairs + main_pairs:
        pairs.append((normalize_english(e), g))

    print("Training pairs:", len(pairs))

    english = [p[0] for p in pairs]
    gloss = [p[1] for p in pairs]

    # 2️⃣ Tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(english + gloss)

    MAX_LEN = 12
    encoded = []

    for e, g in pairs:
        src = tokenizer.encode(e)[:MAX_LEN]
        tgt = (
            [tokenizer.word2idx["<SOS>"]]
            + tokenizer.encode(g)[:MAX_LEN]
            + [tokenizer.word2idx["<EOS>"]]
        )
        encoded.append((src, tgt))

    max_src = max(len(s) for s, _ in encoded)
    max_tgt = max(len(t) for _, t in encoded)

    dataset = [
        (
            torch.tensor(pad_sequence(s, max_src)),
            torch.tensor(pad_sequence(t, max_tgt)),
        )
        for s, t in encoded
    ]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Seq2Seq(len(tokenizer.word2idx))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0

        for src, tgt in loader:
            optimizer.zero_grad()
            out = model(src, tgt[:, :-1])
            loss = criterion(
                out.reshape(-1, out.size(-1)),
                tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {loss_sum:.3f}")

    torch.save(model.state_dict(), "backend/gloss/gloss_model.pt")
    torch.save(tokenizer.word2idx, "backend/gloss/tokenizer_vocab.pt")

    print(" Training complete")


if __name__ == "__main__":
    train_model("backend/datasets/aslg_pc12/train.csv")
