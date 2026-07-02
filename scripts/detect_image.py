"""Run open-vocabulary detection on a single image with no ROS and no robot.

This is the fastest way to prove the VLM half of Lexicon works and to demo it in
an interview: give it a photo and a phrase, get back boxes drawn on the image.

Usage:
  pip install torch transformers pillow numpy
  python detect_image.py --image room.jpg --query "red chair" --query "backpack"
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Make the ROS2 package importable without a ROS install.
PKG = os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src", "lexicon")
sys.path.insert(0, os.path.abspath(PKG))

from lexicon.open_vocab_detector import OpenVocabDetector   # noqa: E402
from lexicon.command_parser import parse_instruction        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--query", action="append", default=[],
                    help="repeatable; or pass --instruction to parse one")
    ap.add_argument("--instruction", default="",
                    help="natural language, parsed to phrases (e.g. 'go to the red chair')")
    ap.add_argument("--model", default="google/owlvit-base-patch32")
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--out", default="detections.jpg")
    args = ap.parse_args()

    from PIL import Image, ImageDraw
    image = np.array(Image.open(args.image).convert("RGB"))

    queries = args.query
    if args.instruction:
        queries = parse_instruction(args.instruction)
        print(f"parsed instruction into queries: {queries}")
    if not queries:
        raise SystemExit("give at least one --query or an --instruction")

    det = OpenVocabDetector(model_name=args.model, score_thresh=args.thresh)
    print(f"loading {args.model} ...")
    det.load()
    dets, latency = det.detect(image, queries)
    print(f"inference: {latency:.0f} ms, {len(dets)} detection(s)")

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    for d in dets:
        x1, y1, x2, y2 = d.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 80), width=3)
        draw.text((x1 + 3, y1 + 3), f"{d.phrase} {d.score:.2f}", fill=(0, 255, 80))
        print(f"  {d.phrase:20s} {d.score:.3f}  box={[round(v) for v in d.box_xyxy]}")
    pil.save(args.out)
    print(f"annotated image -> {args.out}")


if __name__ == "__main__":
    main()
