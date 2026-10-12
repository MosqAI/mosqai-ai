import numpy as np 
import cv2
import os 

def create_mock_frame():
    mock_frame = np.ones((640, 640, 3), dtype=np.uint8) * 120

    cv2.circle(mock_frame, (150, 200), 12, (20, 20, 20), -1)
    cv2.line(mock_frame, (150, 200), (130, 230), (0, 0, 0), 2)

    cv2.circle(mock_frame, (400, 450), 10, (15, 15, 15), -1)
    cv2.line(mock_frame, (400, 450), (420, 480), (0, 0, 0), 2)

    output_path = "tmp_frames/mock_trap_capture.jpg"
    os.makedirs("tmp_frames", exist_ok=True)
    
    cv2.imwrite(output_path, mock_frame)
    print(f"[TEST SETUP]: Mock frame created successfully at: {output_path}")

if __name__ == "__main__":
    create_mock_frame()
