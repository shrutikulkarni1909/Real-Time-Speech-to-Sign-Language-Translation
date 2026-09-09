import json
import os
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))


def build_map(dataset_dir: str, out_path: str):
    json_path = os.path.join(dataset_dir, "WLASL_v0.3.json")
    videos_dir = os.path.join(dataset_dir, "videos")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing {json_path}")
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"Missing videos dir: {videos_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build video_id -> gloss mapping
    video_to_gloss = {}
    for item in data:
        gloss = item["gloss"]
        for inst in item["instances"]:
            vid = inst["video_id"]
            video_to_gloss[vid] = gloss

    result = {}
    for filename in os.listdir(videos_dir):
        if filename.endswith(".mp4"):
            vid = filename.replace(".mp4", "")
            if vid in video_to_gloss:
                result[filename] = video_to_gloss[vid]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Saved:", out_path)
    print("Total matched videos:", len(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build WLASL video->gloss map.")
    parser.add_argument(
        "--dataset-dir",
        default=BASE,
        help="Folder containing WLASL_v0.3.json and videos/",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(BASE, "auto_video_gloss_map.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()
    build_map(os.path.abspath(args.dataset_dir), os.path.abspath(args.out))
