import numpy as np
import pytest

from src.inference_engine import MosqAInferenceEngine


def make_engine(class_ids=None, size=640):
    # Skip __init__ so no ONNX file is needed.
    engine = MosqAInferenceEngine.__new__(MosqAInferenceEngine)
    engine.confidence_threshold = 0.5
    engine.nms_threshold = 0.4
    engine.class_ids = None if class_ids is None else set(class_ids)
    engine.input_width = engine.input_height = size
    return engine


def yolo_output(detections, num_classes=2, num_boxes=20):
    """Build a (1, 4 + num_classes, num_boxes) tensor from (cx, cy, w, h, class_id, score) tuples."""
    out = np.zeros((1, 4 + num_classes, num_boxes), dtype=np.float32)
    for i, (cx, cy, w, h, class_id, score) in enumerate(detections):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + class_id, i] = score
    return out


def test_letterbox_keeps_aspect_ratio_and_pads():
    engine = make_engine()
    vga = np.full((480, 640, 3), 255, dtype=np.uint8)

    boxed = engine.letterbox(vga)

    assert boxed.shape == (640, 640, 3)
    assert (boxed[80:560] == 255).all()          # image centred, not stretched
    assert (boxed[:80] == 114).all() and (boxed[560:] == 114).all()


def test_preprocess_returns_nchw_float_tensor(tmp_path):
    import cv2
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), np.zeros((480, 640, 3), dtype=np.uint8))

    tensor, original = make_engine().preprocess(str(path))

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    assert original.shape == (480, 640, 3)


def test_preprocess_rejects_undecodable_file(tmp_path):
    path = tmp_path / "garbage.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        make_engine().preprocess(str(path))


def test_postprocess_counts_separate_boxes_and_merges_overlaps():
    output = yolo_output([
        (100, 100, 20, 20, 0, 0.9),
        (101, 101, 20, 20, 0, 0.8),   # overlaps the first -> suppressed
        (400, 400, 20, 20, 0, 0.7),
        (300, 300, 20, 20, 0, 0.3),   # below threshold
    ])
    assert make_engine().postprocess(output) == 2


def test_postprocess_filters_to_target_classes():
    output = yolo_output([(100, 100, 20, 20, 0, 0.9), (400, 400, 20, 20, 1, 0.9)])
    assert make_engine(class_ids=[1]).postprocess(output) == 1


def test_postprocess_empty_output_is_zero():
    assert make_engine().postprocess(yolo_output([])) == 0


def test_detect_maps_boxes_back_to_the_original_photo(tmp_path):
    import cv2
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), np.zeros((480, 640, 3), dtype=np.uint8))

    engine = make_engine()
    engine.input_name = "images"

    class FakeSession:
        def run(self, outputs, feeds):
            # 640x480 is letterboxed into 640x640 with 80 px of padding on top.
            return [yolo_output([(320, 80 + 240, 40, 20, 0, 0.9)])]

    engine.session = FakeSession()
    assert engine.detect(str(path)) == [(300, 230, 40, 20)]
    assert engine.count_mosquitoes(str(path), background_path="ignored.jpg") == 1
