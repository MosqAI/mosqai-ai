from src.ensemble_detector import EnsembleDetector


class FixedDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def detect(self, image_path, background_path=None):
        self.calls.append((image_path, background_path))
        return self.boxes


def test_counts_only_model_detections_the_rules_confirm():
    model = FixedDetector([(100, 100, 20, 8), (400, 400, 20, 8), (50, 300, 20, 8)])
    rules = FixedDetector([(102, 98, 18, 10), (395, 405, 22, 6)])

    assert EnsembleDetector(model, rules).count_mosquitoes("frame.jpg", "empty.jpg") == 2
    # The rules see the background photo; the model looks at the frame alone.
    assert rules.calls == [("frame.jpg", "empty.jpg")]
    assert model.calls == [("frame.jpg", None)]


def test_margin_allows_for_blur():
    model = FixedDetector([(100, 100, 20, 8)])        # centre (110, 104)
    rules = FixedDetector([(115, 100, 20, 8)])        # starts 5 px right of the centre

    assert EnsembleDetector(model, rules, margin=10).count_mosquitoes("f.jpg") == 1
    assert EnsembleDetector(model, rules, margin=0).count_mosquitoes("f.jpg") == 0


def test_skips_the_model_when_the_rules_find_nothing():
    model = FixedDetector([(100, 100, 20, 8)])
    assert EnsembleDetector(model, FixedDetector([])).count_mosquitoes("f.jpg") == 0
    assert model.calls == []
