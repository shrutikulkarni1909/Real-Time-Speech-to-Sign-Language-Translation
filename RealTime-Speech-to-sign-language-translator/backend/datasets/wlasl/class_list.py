import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE, "WLASL_v0.3.json")
OUT_PATH = os.path.join(BASE, "wlasl_class_list.txt")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# data is a list of dicts: each has "gloss"
glosses = sorted({item["gloss"] for item in data if "gloss" in item})

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for g in glosses:
        f.write(g + "\n")

print("✅ Saved:", OUT_PATH)
print("Total classes:", len(glosses))
