import os

import cv2
import numpy as np

TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp_frames"))
DARK_TUBE = 25


def tube_photo(mosquitoes=()):
    """A 640x480 view of the dark tube entrance, with LED-lit mosquitoes at the given centres."""
    photo = np.full((480, 640, 3), DARK_TUBE, dtype=np.uint8)
    cv2.rectangle(photo, (0, 0), (639, 30), (180, 180, 180), -1)              # tube rim
    cv2.ellipse(photo, (500, 100), (10, 3), 0, 0, 360, (210, 210, 210), -1)  # LED reflection
    for cx, cy in mosquitoes:
        for dx in (-6, 0, 6):
            cv2.line(photo, (cx + dx, cy - 12), (cx + dx, cy + 12), (120, 120, 120), 1)
        cv2.ellipse(photo, (cx, cy), (12, 3), 20, 0, 360, (200, 200, 200), -1)
    return photo


def write(name, photo):
    os.makedirs(TARGET_DIR, exist_ok=True)
    path = os.path.join(TARGET_DIR, name)
    cv2.imwrite(path, photo)
    return path


def create_mock_frame():
    """A tube photo with two mosquitoes in view."""
    output_path = write("mock_trap_capture.jpg", tube_photo([(200, 250), (420, 330)]))
    print(f"[TEST SETUP]: Mock frame created successfully at: {output_path}")
    return output_path


def create_mock_event():
    """(empty tube, burst photos) as the device sends them for one mosquito passing through."""
    empty = write("mock_empty_tube.jpg", tube_photo())
    burst = [
        write("mock_event_1.jpg", tube_photo([(200, 250)])),
        write("mock_event_2.jpg", tube_photo([(230, 210)])),  # moving towards the fan
        write("mock_event_3.jpg", tube_photo()),               # already pulled in
    ]
    return empty, burst


if __name__ == "__main__":
    create_mock_frame()
