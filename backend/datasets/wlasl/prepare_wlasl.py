import argparse
import os
import subprocess
import sys

import kagglehub


def run_module(module_name: str, args: list[str], python_cmd: list[str] | None = None) -> None:
    cmd = (python_cmd or [sys.executable]) + ["-m", module_name] + args
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def current_python_has_holistic() -> bool:
    try:
        import mediapipe as mp  # type: ignore
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WLASL from KaggleHub and prepare keypoints.")
    parser.add_argument("--all", action="store_true", help="Extract all glosses from map.")
    parser.add_argument("--words", default="", help="Comma-separated gloss list to extract.")
    parser.add_argument("--max-words", type=int, default=300, help="Cap extracted words (0 = no cap).")
    parser.add_argument(
        "--extract-python",
        default="",
        help="Optional python executable for keypoint extraction (recommended: Python 3.11 venv).",
    )
    args = parser.parse_args()

    dataset_path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
    dataset_path = os.path.abspath(dataset_path)
    print("Dataset downloaded/available at:", dataset_path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    map_out = os.path.join(repo_root, "backend", "datasets", "wlasl", "auto_video_gloss_map.json")

    run_module(
        "backend.datasets.wlasl.build_demo_map",
        ["--dataset-dir", dataset_path, "--out", map_out],
    )

    extract_args = [
        "--dataset-dir",
        dataset_path,
        "--map-path",
        map_out,
    ]
    if args.all:
        extract_args.append("--all")
    elif args.words.strip():
        extract_args.extend(["--words", args.words.strip()])

    if args.max_words > 0:
        extract_args.extend(["--max-words", str(args.max_words)])

    extract_python = [args.extract_python] if args.extract_python.strip() else [sys.executable]
    if not args.extract_python.strip() and not current_python_has_holistic():
        raise RuntimeError(
            "Current Python does not provide mediapipe.solutions.holistic.\n"
            "Install Python 3.11 and create a dedicated extractor venv, then run:\n"
            "  py -3.11 -m venv venv311\n"
            "  .\\venv311\\Scripts\\python -m pip install -r requirements.txt\n"
            "  .\\venv\\Scripts\\python -m backend.datasets.wlasl.prepare_wlasl --all --max-words 300 "
            "--extract-python .\\venv311\\Scripts\\python"
        )

    run_module("backend.datasets.wlasl.extract_keypoints", extract_args, python_cmd=extract_python)

    print("\nDone. Next, train gloss model:")
    print(f"{sys.executable} -m backend.gloss.train --epochs 5 --batch-size 16")


if __name__ == "__main__":
    main()

