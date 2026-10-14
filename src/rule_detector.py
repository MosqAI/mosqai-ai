"""Rules-based mosquito counter, used until a YOLO model is trained on mosquitoes.

It knows nothing about what a mosquito looks like, only the properties one has in
a trap photo: a small, elongated blob that stands out from the tube behind it.
Flies or debris with the same size and shape are counted too, so tune the limits
on real photos (see the __main__ block below).

With a `background` photo (the empty tube, sent by the device with each motion
event) only what is new in the frame is considered, which ignores the tube rim,
reflections and dust that are always there.
"""
import os
import sys

import cv2
import numpy as np
from dotenv import load_dotenv

RULES_VERSION = "rules-v2"
RING_WIDTH = 4                 # pixels around a blob checked for being plain surface
MAX_SIMILAR_RING_FRACTION = 0.25


class RuleBasedMosquitoDetector:
    def __init__(self, min_area=15, max_area=2500, min_elongation=1.5, min_contrast=40,
                 polarity="bright", background_size=41):
        if polarity not in ("bright", "dark"):
            raise ValueError(f"polarity must be 'bright' or 'dark', got {polarity!r}")
        self.min_area = min_area              # blob area in pixels of the 640x480 photo
        self.max_area = max_area
        self.min_elongation = min_elongation  # long side / short side of the blob
        self.min_contrast = min_contrast      # how much it differs (0-255) from what is behind it
        # bright: an LED-lit mosquito on a dark tube; dark: a mosquito on a light surface.
        self.polarity = polarity
        # Without a background photo, areas wider than this (shadows, the tube edge)
        # are treated as background.
        self.background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (background_size, background_size))
        self.join_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_WIDTH + 1, 2 * RING_WIDTH + 1))

    @classmethod
    def from_env(cls):
        return cls(
            min_area=float(os.getenv("RULES_MIN_AREA", "15")),
            max_area=float(os.getenv("RULES_MAX_AREA", "2500")),
            min_elongation=float(os.getenv("RULES_MIN_ELONGATION", "1.5")),
            min_contrast=int(os.getenv("RULES_MIN_CONTRAST", "40")),
            polarity=os.getenv("RULES_POLARITY", "bright").strip().lower(),
        )

    def detect(self, image_path, background_path=None):
        """Return the bounding boxes (x, y, w, h) of mosquito-like blobs."""
        gray = read_gray(image_path)

        if background_path:
            background = read_gray(background_path)
            if background.shape != gray.shape:
                background = cv2.resize(background, (gray.shape[1], gray.shape[0]))
            # Cancel a small auto-exposure shift between the two photos.
            shift = int(np.median(gray)) - int(np.median(background))
            background = np.clip(background.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            response = cv2.absdiff(gray, background)
        else:
            # Top-hat / black-hat keep only bright / dark details smaller than the
            # kernel, measured against their local surroundings.
            operation = cv2.MORPH_TOPHAT if self.polarity == "bright" else cv2.MORPH_BLACKHAT
            response = cv2.morphologyEx(gray, operation, self.background_kernel)

        mask = (response >= self.min_contrast).astype(np.uint8)
        # Strip 1-pixel lines (legs, antennae) so the shape test sees the body, then
        # reconnect body parts that thresholding split apart.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.join_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.join_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not self.min_area <= area <= self.max_area:
                continue
            (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
            if max(side_a, side_b) / max(min(side_a, side_b), 1.0) < self.min_elongation:
                continue
            # A background photo already removed large areas; without one, the
            # corners of a large area look like small blobs and must be ruled out.
            if not background_path and not self.is_isolated(gray, contour):
                continue
            boxes.append(cv2.boundingRect(contour))
        return boxes

    def count_mosquitoes(self, image_path, background_path=None):
        return len(self.detect(image_path, background_path))

    def is_isolated(self, gray, contour):
        """True if the blob sits on plain surface, not on the corner of a larger area like it."""
        x, y, w, h = cv2.boundingRect(contour)
        pad = RING_WIDTH + 1
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, gray.shape[1]), min(y + h + pad, gray.shape[0])

        blob = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.drawContours(blob, [contour], -1, 1, cv2.FILLED, offset=(-x0, -y0))
        ring = cv2.dilate(blob, self.ring_kernel) & (1 - blob)

        roi = gray[y0:y1, x0:x1]
        ring_pixels = roi[ring == 1].astype(np.int16)
        if ring_pixels.size == 0:
            return False
        blob_level = roi[blob == 1].mean()
        if self.polarity == "bright":
            similar = ring_pixels >= blob_level - self.min_contrast
        else:
            similar = ring_pixels <= blob_level + self.min_contrast
        # Legs are thin, so only a small part of a mosquito's surroundings looks like it.
        return similar.mean() <= MAX_SIMILAR_RING_FRACTION


def read_gray(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not decode image at location: {image_path}")
    return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)


if __name__ == "__main__":
    # Tuning aid: python -m src.rule_detector photo.jpg [empty_tube.jpg]
    # -> writes photo_detections.jpg with the counted blobs boxed in red.
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python -m src.rule_detector <photo.jpg> [background.jpg]")
    load_dotenv()  # same RULES_* settings as the server
    path = sys.argv[1]
    found = RuleBasedMosquitoDetector.from_env().detect(path, sys.argv[2] if len(sys.argv) == 3 else None)
    preview = cv2.imread(path)
    for x, y, w, h in found:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
    out_path = path.rsplit(".", 1)[0] + "_detections.jpg"
    cv2.imwrite(out_path, preview)
    print(f"{len(found)} mosquito-like blobs, preview written to {out_path}")
