import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# Model file path (must be in the same folder as eye.py)
MODEL_PATH = "face_landmarker.task"

# MediaPipe Face Landmarker landmark indices for EAR
# [outer_corner, top_1, top_2, inner_corner, bottom_2, bottom_1]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22      # EAR below this counts as eye closed
CONSEC_FRAMES = 2         # Consecutive frames to register a blink

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    coords = []
    for idx in eye_indices:
        pt = landmarks[idx]
        coords.append(np.array([pt.x * img_w, pt.y * img_h]))

    p1, p2, p3, p4, p5, p6 = coords

    dist_v1 = np.linalg.norm(p2 - p6)
    dist_v2 = np.linalg.norm(p3 - p5)
    dist_h  = np.linalg.norm(p1 - p4)

    if dist_h == 0:
        return 0.0

    return (dist_v1 + dist_v2) / (2.0 * dist_h)

def main():
    # Configure MediaPipe Tasks Vision FaceLandmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    closed_frames = 0
    blink_count = 0
    start_time = time.time()
    blinks_per_minute = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert OpenCV BGR image to MediaPipe Image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect landmarks
        detection_result = detector.detect(mp_image)

        ear = 0.0
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            ear = (right_ear + left_ear) / 2.0

            # Draw landmarks on screen
            for idx in RIGHT_EYE + LEFT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Blink detection state logic
            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                if closed_frames >= CONSEC_FRAMES:
                    blink_count += 1
                closed_frames = 0

        # Calculate BPM
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            blinks_per_minute = (blink_count / elapsed_time) * 60.0

        # Render stats overlay
        status_color = (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Rate: {blinks_per_minute:.1f} BPM", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)

        cv2.imshow("Real-Time EAR & Blink Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()