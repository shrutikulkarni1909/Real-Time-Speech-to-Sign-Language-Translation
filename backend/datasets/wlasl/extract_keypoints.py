import cv2
import mediapipe as mp
import json
import os
import argparse
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "keypoints")

os.makedirs(OUT_DIR, exist_ok=True)

def resolve_holistic_module():
    """
    Resolve MediaPipe Holistic across different package layouts.
    Some builds expose it as `mediapipe.solutions.holistic`, while others
    require importing from `mediapipe.python.solutions`.
    """
    # Standard layout
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
        return mp.solutions.holistic

    # Fallback layout
    try:
        from mediapipe.python.solutions import holistic as holistic_mod  # type: ignore
        return holistic_mod
    except Exception as e:
        raise RuntimeError(
            "MediaPipe Holistic API is unavailable in this environment. "
            "Your mediapipe build exposes only `tasks` (no `solutions.holistic`). "
            f"Python version: {sys.version.split()[0]}. "
            "For this script, use Python 3.10/3.11 venv and reinstall requirements."
        ) from e


mp_holistic = resolve_holistic_module()

POSE_LM = 33
HAND_LM = 21

FRAME_STRIDE = 2
MAX_FRAMES = 120


def zeros(n):
    return [[0.0, 0.0, 0.0] for _ in range(n)]


def load_reverse_map(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        video_to_gloss = json.load(f)

    # reverse mapping
    gloss_to_video = {}
    for video, gloss in video_to_gloss.items():
        gloss = gloss.upper()
        if gloss not in gloss_to_video:
            gloss_to_video[gloss] = video

    return gloss_to_video


def extract(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frames = []
    idx = 0
    kept = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            idx += 1
            if idx % FRAME_STRIDE != 0:
                continue

            kept += 1
            if kept > MAX_FRAMES:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            pose = zeros(POSE_LM)
            lh = zeros(HAND_LM)
            rh = zeros(HAND_LM)

            if results.pose_landmarks:
                pose = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            if results.left_hand_landmarks:
                lh = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            if results.right_hand_landmarks:
                rh = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]

            frames.append({
                "pose": pose,
                "left_hand": lh,
                "right_hand": rh
            })

    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints from WLASL videos.")
    parser.add_argument(
        "--words",
        type=str,
        default="",
        help="Comma-separated gloss list, e.g. HELLO,THANK-YOU,BOOK",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract for all glosses found in auto_video_gloss_map.json",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=0,
        help="Optional cap for number of glosses to process (0 = no cap).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=BASE_DIR,
        help="WLASL folder containing videos/ and auto_video_gloss_map.json",
    )
    parser.add_argument(
        "--map-path",
        type=str,
        default="",
        help="Optional explicit path to auto_video_gloss_map.json",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    videos_dir = os.path.join(dataset_dir, "videos")
    map_path = os.path.abspath(args.map_path) if args.map_path else os.path.join(dataset_dir, "auto_video_gloss_map.json")

    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"Missing videos dir: {videos_dir}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Missing map JSON: {map_path}. Run build_demo_map first.")

    gloss_to_video = load_reverse_map(map_path)

    # Default demo set when no flags are passed.
    default_words = [
        "HELLO",
        "GOOD",
        "GOODBYE",
        "GO",
        "BOY",
        "GIRL",
        "BOOK",
        "BASKETBALL",
        "FLOWER",
        "FRIEND",
        "FUN",
        "AIRPLANE",
        "FLY",
        "BEAUTIFUL"
    ]

    if args.all:
        target_words = sorted(gloss_to_video.keys())
    elif args.words.strip():
        target_words = [w.strip().upper() for w in args.words.split(",") if w.strip()]
    else:
        target_words = default_words

    if args.max_words and args.max_words > 0:
        target_words = target_words[:args.max_words]

    print(f"Target glosses: {len(target_words)}")

    skipped = 0
    extracted = 0

    for gloss in target_words:
        if gloss not in gloss_to_video:
            print(f"No video found for {gloss}")
            continue

        video = gloss_to_video[gloss]
        video_path = os.path.join(videos_dir, video)

        if not os.path.exists(video_path):
            print(f"Missing file {video}")
            continue

        out_path = os.path.join(OUT_DIR, f"{gloss}.json")
        
        # ✅ IMPROVED: Check if already extracted
        if os.path.exists(out_path):
            print(f"✓ Already extracted {gloss}, skipping")
            skipped += 1
            continue

        print(f"Extracting {gloss} from {video} ...")
        frames = extract(video_path)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "gloss": gloss,
                "video": video,
                "frames": frames
            }, f)

        print(f"Saved {gloss}: {len(frames)} frames")
        extracted += 1

    print(f"\nKeypoint extraction complete. Extracted: {extracted}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
