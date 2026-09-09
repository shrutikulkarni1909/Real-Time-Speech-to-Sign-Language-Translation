import re

def normalize_english(text: str) -> str:
    """
    Normalize English sentence before tokenization
    """
    if not text:
        return ""

    text = text.replace("\ufeff", "")
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_gloss(gloss: str) -> str:
    """
    Normalize gloss format
    """
    if not gloss:
        return ""

    gloss = gloss.replace("\ufeff", "")
    gloss = gloss.upper()
    gloss = gloss.replace("\n", " ")
    gloss = re.sub(r"\s+", " ", gloss).strip()
    return gloss

if __name__ == "__main__":
    s = "Hello!! I am Going to SCHOOL."
    print("Original:", s)
    print("Normalized:", normalize_english(s))
