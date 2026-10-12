import os
import numpy as np

def create_mock_frame():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp_frames"))
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, "mock_trap_capture.jpg")

    try:
        import cv2
        mock_frame = np.ones((640, 640, 3), dtype=np.uint8) * 120

        cv2.circle(mock_frame, (150, 200), 12, (20, 20, 20), -1)
        cv2.line(mock_frame, (150, 200), (130, 230), (0, 0, 0), 2)

        cv2.circle(mock_frame, (400, 450), 10, (15, 15, 15), -1)
        cv2.line(mock_frame, (400, 450), (420, 480), (0, 0, 0), 2)

        cv2.imwrite(output_path, mock_frame)
    except ImportError:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (640, 640), (120, 120, 120))
        draw = ImageDraw.Draw(img)
        draw.ellipse([150 - 12, 200 - 12, 150 + 12, 200 + 12], fill=(20, 20, 20))
        draw.line([(150, 200), (130, 230)], fill=(0, 0, 0), width=2)
        draw.ellipse([400 - 10, 450 - 10, 400 + 10, 450 + 10], fill=(15, 15, 15))
        draw.line([(400, 450), (420, 480)], fill=(0, 0, 0), width=2)
        img.save(output_path, "JPEG")

    print(f"[TEST SETUP]: Mock frame created successfully at: {output_path}")
    return output_path

if __name__ == "__main__":
    create_mock_frame()

