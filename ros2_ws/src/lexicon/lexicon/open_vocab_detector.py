"""Open-vocabulary object detector: find arbitrary text queries in an image.

Default backbone is OWL-ViT (google/owlvit-base-patch32) through HuggingFace
transformers. It takes free-text queries like "red chair" and returns boxes,
so no retraining is needed to support a new object: you just ask for it. That
open vocabulary is the whole reason a VLM belongs in this pipeline instead of a
fixed 80-class detector.

Kept as a plain class with no ROS dependency so it can be:
  - loaded lazily by the lifecycle node on activation (the model is large;
    you do not want it resident until the node is actually in use),
  - driven directly by scripts/detect_image.py for a no-ROS demo.

Swapping the model: Grounding DINO and YOLO-World are stronger open-vocab
detectors with the same interface (text queries -> boxes). Implement the same
detect() signature and the rest of the stack is unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class OpenVocabDetection:
    phrase: str          # the query that matched
    score: float
    box_xyxy: List[float]  # pixel coords [x1, y1, x2, y2] in the input image

    @property
    def center(self):
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        """Box area in pixels. Useful for filtering tiny or noisy detections."""
        x1, y1, x2, y2 = self.box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def iou(self, other: "OpenVocabDetection") -> float:
        """Intersection-over-union with another detection, for NMS or dedup."""
        x1 = max(self.box_xyxy[0], other.box_xyxy[0])
        y1 = max(self.box_xyxy[1], other.box_xyxy[1])
        x2 = min(self.box_xyxy[2], other.box_xyxy[2])
        y2 = min(self.box_xyxy[3], other.box_xyxy[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - inter
        if union <= 0.0:
            return 0.0
        return inter / union


class OpenVocabDetector:
    def __init__(self, model_name: str = "google/owlvit-base-patch32",
                 device: str = "cpu", score_thresh: float = 0.1):
        self._model_name = model_name
        self._device = device
        self._thresh = score_thresh
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load weights. Called explicitly (for example from on_activate) so the
        cost is paid at a known time, not on the first frame."""
        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self._torch = torch
        self._processor = OwlViTProcessor.from_pretrained(self._model_name)
        self._model = OwlViTForObjectDetection.from_pretrained(self._model_name)
        self._model.to(self._device).eval()

    def unload(self) -> None:
        """Free the model (for example from on_deactivate) to release memory."""
        self._model = None
        self._processor = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def detect(self, image_rgb: np.ndarray, queries: List[str]
               ) -> tuple[List[OpenVocabDetection], float]:
        """Run detection for a list of text queries against one RGB image.

        Returns (detections, latency_ms). Detections come back sorted by score.
        """
        if not self.loaded:
            raise RuntimeError("detector not loaded; call load() first")
        torch = self._torch

        from PIL import Image
        pil = Image.fromarray(image_rgb)
        inputs = self._processor(text=[queries], images=pil, return_tensors="pt").to(self._device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self._model(**inputs)
        # target_sizes maps normalized boxes back to pixel coords (h, w).
        target_sizes = torch.tensor([pil.size[::-1]]).to(self._device)
        results = self._processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self._thresh)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0

        dets: List[OpenVocabDetection] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            dets.append(OpenVocabDetection(
                phrase=queries[int(label)],
                score=float(score),
                box_xyxy=[float(v) for v in box.tolist()],
            ))
        dets.sort(key=lambda d: -d.score)
        return dets, latency_ms
