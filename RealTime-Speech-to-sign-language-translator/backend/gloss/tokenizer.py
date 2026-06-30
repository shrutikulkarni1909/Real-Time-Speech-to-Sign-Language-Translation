#text to tokens

import re
from collections import Counter

class SimpleTokenizer:
  def __init__(self, min_freq=5):
        self.min_freq = min_freq

        # 🔹 Special tokens (FIX)
        self.word2idx = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }

        self.idx2word = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

  def clean_text(self, text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]" ,"", text)
    return text.strip()
  
  def build_vocab(self, sentences):
    counter = Counter()
    for sent in sentences:
      sent = self.clean_text(sent)
      counter.update(sent.split())

    for word, freq in counter.items():
      if freq>= self.min_freq and word not in self.word2idx:
        idx = len(self.word2idx)
        self.word2idx[word] = idx
        self.idx2word[idx] = word
  
  def encode(self,text):
    text = self.clean_text(text)
    tokens = text.split()
    return [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
  
  def decode(self, token_ids):
    words = []
    for tid in token_ids:
      if tid == self.word2idx["<EOS>"]:
        break
      if tid>3:
        words.append(self.idx2word.get(tid, ""))
    return " ".join(words)

# if __name__ == "__main__":
#     t = SimpleTokenizer()
#     t.build_vocab(["hello how are you", "HELLO YOU"])
#     print(t.encode("hello you"))
#     print(t.decode([4, 5]))
