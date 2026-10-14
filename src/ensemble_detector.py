"""YOLO + rules ensemble: a detection counts only when both agree.

The trained model knows what a mosquito looks like but can be fooled by LED glare
or a bright speck; the rules detector (with the empty-tube background) knows that
something new, mosquito-sized and mosquito-shaped is really there. Requiring both
cuts false positives from either one.
"""


class EnsembleDetector:
    def __init__(self, model, rules, margin=10):
        self.model = model
        self.rules = rules
        self.margin = margin  # pixels of slack: blur makes the two boxes differ a little

    def detect(self, image_path, background_path=None):
        candidates = self.rules.detect(image_path, background_path)
        if not candidates:
            return []
        return [box for box in self.model.detect(image_path) if self.confirmed(box, candidates)]

    def count_mosquitoes(self, image_path, background_path=None):
        return len(self.detect(image_path, background_path))

    def confirmed(self, box, candidates):
        x, y, w, h = box
        center_x, center_y = x + w / 2, y + h / 2
        return any(
            cx - self.margin <= center_x <= cx + cw + self.margin
            and cy - self.margin <= center_y <= cy + ch + self.margin
            for cx, cy, cw, ch in candidates
        )
