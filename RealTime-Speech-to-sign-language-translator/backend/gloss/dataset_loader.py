import csv
import os
import re

def resolve_default_dataset_path():
    """Pick first existing ASLG train.csv from known project layouts."""
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "..", "datasets", "aslg_pc12", "train.csv"),
        os.path.join(base, "..", "datasets", "wlasl", "aslg_pc12", "train.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback to legacy/default location for clearer error messages downstream.
    return candidates[0]


DATASET_PATH = resolve_default_dataset_path()

# ---------------------------
# Normalization helpers
# ---------------------------

def normalize_english(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufeff", "")  # remove BOM
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_gloss(gloss: str) -> str:
    if not gloss:
        return ""
    gloss = gloss.replace("\ufeff", "")
    gloss = gloss.upper()
    gloss = gloss.replace("\n", " ")
    gloss = re.sub(r"\s+", " ", gloss).strip()
    return gloss


# ---------------------------
# Rule-based mapping loader
# ---------------------------

def load_english_to_gloss_mapping():
    """
    Used for rule-based inference.
    Returns:
        dict: { normalized_english_sentence : normalized_gloss }
    """
    mapping = {}

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "text" not in reader.fieldnames or "gloss" not in reader.fieldnames:
            raise ValueError("CSV must contain 'text' and 'gloss' columns")

        for row in reader:
            eng = normalize_english(row["text"])
            gloss = normalize_gloss(row["gloss"])

            if eng and gloss:
                mapping[eng] = gloss

    return mapping


# ---------------------------
# ML training loaders
# ---------------------------

def load_dataset_for_training(csv_path: str = DATASET_PATH):
    """
    Used for ML training (ASLG-PC12).
    Returns:
        list of (english_sentence, gloss_sentence)
    """
    pairs = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            eng = normalize_english(row["text"])
            gloss = normalize_gloss(row["gloss"])

            if eng and gloss:
                pairs.append((eng, gloss))

            if len(pairs) == 10000:
                break

    return pairs


def load_daily_pairs(txt_path: str):
    """
    Load daily-use English → Gloss pairs from txt file
    Format per line:
        english sentence ||| GLOSS SENTENCE
    """
    pairs = []

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Daily pairs file not found at {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            if "|||" not in line:
                continue

            eng, gloss = line.split("|||", 1)

            eng = normalize_english(eng.strip())
            gloss = normalize_gloss(gloss.strip())

            if eng and gloss:
                pairs.append((eng, gloss))

    return pairs


# ---------------------------
# Quick sanity tests
# ---------------------------

if __name__ == "__main__":
    print(" Mapping test:")
    mapping = load_english_to_gloss_mapping()
    for i, (e, g) in enumerate(mapping.items()):
        print(e, "->", g)
        if i == 4:
            break

    print("\n Training pairs test (ASLG):")
    pairs = load_dataset_for_training()
    for i, (e, g) in enumerate(pairs[:5]):
        print(e, "->", g)

    print("\n Daily pairs test:")
    daily = load_daily_pairs("backend/gloss/daily_pairs.txt")
    for i, (e, g) in enumerate(daily[:5]):
        print(e, "->", g)
