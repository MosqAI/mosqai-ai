import cv2
import numpy as np
import pytest

from src.rule_detector import RuleBasedMosquitoDetector

DARK_TUBE = 25
LIT_MOSQUITO = (200, 200, 200)


def write(tmp_path, name, image):
    path = tmp_path / name
    cv2.imwrite(str(path), image)
    return str(path)


def frame(tmp_path, draw, level=DARK_TUBE, name="frame.jpg"):
    """Write a VGA photo of a plain surface at `level` plus whatever `draw` adds."""
    image = np.full((480, 640, 3), level, dtype=np.uint8)
    draw(image)
    return write(tmp_path, name, image)


def mosquito(image, center, color=LIT_MOSQUITO, leg_color=None, angle=0):
    cx, cy = center
    leg_color = leg_color or tuple(int(c * 0.6) for c in color)
    for dx in (-6, 0, 6):  # thin legs on both sides of the body
        cv2.line(image, (cx + dx, cy - 12), (cx + dx, cy + 12), leg_color, 1)
    cv2.ellipse(image, center, (12, 3), angle, 0, 360, color, -1)


def tube_fixtures(image):
    """Things always in view: the bright tube rim edge and a reflection speck."""
    cv2.rectangle(image, (0, 0), (639, 30), (180, 180, 180), -1)
    cv2.ellipse(image, (500, 100), (10, 3), 0, 0, 360, (210, 210, 210), -1)


# ── LED-lit mosquito on the dark tube (the trap's setup) ──

def test_counts_bright_mosquitoes_on_dark_tube(tmp_path):
    def draw(image):
        mosquito(image, (150, 150))
        mosquito(image, (450, 300), angle=60)

    assert RuleBasedMosquitoDetector().count_mosquitoes(frame(tmp_path, draw)) == 2


def test_ignores_bright_blobs_without_mosquito_properties(tmp_path):
    def draw(image):
        cv2.circle(image, (100, 100), 6, LIT_MOSQUITO, -1)                       # round
        cv2.circle(image, (300, 100), 1, LIT_MOSQUITO, -1)                       # speck
        cv2.rectangle(image, (350, 200), (600, 450), LIT_MOSQUITO, -1)           # glare patch
        cv2.ellipse(image, (150, 350), (12, 3), 0, 0, 360, (45, 45, 45), -1)     # too faint

    assert RuleBasedMosquitoDetector().count_mosquitoes(frame(tmp_path, draw)) == 0


def test_background_photo_ignores_what_is_always_in_view(tmp_path):
    empty = frame(tmp_path, tube_fixtures, name="empty.jpg")

    def draw(image):
        tube_fixtures(image)
        mosquito(image, (300, 300))

    with_mosquito = frame(tmp_path, draw)
    detector = RuleBasedMosquitoDetector()

    # The reflection speck is mosquito-shaped, so on its own it is miscounted...
    assert detector.count_mosquitoes(with_mosquito) == 2
    # ...but against the empty tube only the new mosquito is left.
    assert detector.count_mosquitoes(with_mosquito, empty) == 1
    assert detector.count_mosquitoes(empty, empty) == 0


def test_background_photo_tolerates_exposure_shift(tmp_path):
    empty = frame(tmp_path, tube_fixtures, name="empty.jpg")

    def draw(image):
        tube_fixtures(image)
        mosquito(image, (300, 300))
        image[:] = np.clip(image.astype(np.int16) + 15, 0, 255).astype(np.uint8)  # brighter exposure

    assert RuleBasedMosquitoDetector().count_mosquitoes(frame(tmp_path, draw), empty) == 1


def test_background_of_another_size_is_resized(tmp_path):
    small_empty = write(tmp_path, "empty.jpg", np.full((240, 320, 3), DARK_TUBE, dtype=np.uint8))

    def draw(image):
        mosquito(image, (300, 300))

    assert RuleBasedMosquitoDetector().count_mosquitoes(frame(tmp_path, draw), small_empty) == 1


# ── dark mosquito on a light surface ──

def test_dark_polarity_counts_dark_mosquitoes(tmp_path):
    def draw(image):
        mosquito(image, (150, 150), color=(40, 40, 40), leg_color=(90, 90, 90))
        mosquito(image, (450, 300), color=(40, 40, 40), leg_color=(90, 90, 90), angle=60)

    path = frame(tmp_path, draw, level=220)
    assert RuleBasedMosquitoDetector(polarity="dark").count_mosquitoes(path) == 2
    assert RuleBasedMosquitoDetector(polarity="bright").count_mosquitoes(path) == 0


def test_dark_polarity_ignores_shadow_corners(tmp_path):
    def draw(image):
        cv2.rectangle(image, (350, 200), (600, 450), (40, 40, 40), -1)

    assert RuleBasedMosquitoDetector(polarity="dark").count_mosquitoes(frame(tmp_path, draw, level=220)) == 0


def test_uneven_lighting_is_not_counted(tmp_path):
    def draw(image):
        image[:] = np.linspace(10, 90, 640, dtype=np.uint8)[None, :, None]
        mosquito(image, (200, 240))

    assert RuleBasedMosquitoDetector().count_mosquitoes(frame(tmp_path, draw)) == 1


def test_rejects_unknown_polarity():
    with pytest.raises(ValueError):
        RuleBasedMosquitoDetector(polarity="sideways")


def test_rejects_undecodable_file(tmp_path):
    path = tmp_path / "garbage.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        RuleBasedMosquitoDetector().count_mosquitoes(str(path))
