"""
Screen Shoulder-Surfing & Visual Privacy Monitor.

Protects a sensitive operator screen from unauthorized visual inspection by
estimating head/gaze orientation for every person the camera sees. Uses a
YOLO pose (keypoint) model instead of a custom-trained Roboflow model, since
COCO-pose keypoints already include nose/eyes/ears/shoulders - the same
keypoints the Roboflow spec calls for - so no extra labeling/training run is
needed to get a working demo.

Classes (assigned by heuristic, not by the model):
- Primary_Operator: the largest/most-centered detected person (assumed to be
  the person sitting in front of the screen).
- Bystander: every other detected person.

Gaze_Vector (head orientation) is approximated from the eye/nose keypoints:
a bystander is considered to be "looking at the screen" when their face is
oriented frontally toward the camera (the camera is assumed to be mounted at
the screen), sustained over several frames to avoid flicker.

Deployment output: blurs the display feed and shows a lock banner the moment
a bystander's gaze vector is judged to be aimed at the screen. Press 'q' to
quit.
"""
import time
import queue
import threading

import cv2
import numpy as np
import pyttsx3
from PIL import ImageGrab
from ultralytics import YOLO

cam = 0
PRIVACY_WINDOW = "PRIVACY_LOCK_OVERLAY"

# COCO pose keypoint indices used by Ultralytics pose models
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

KEYPOINT_CONF_THRESHOLD = 0.4     # ignore low-confidence/occluded keypoints
FRONTAL_GAZE_RATIO = 0.35         # how centered the nose must be between the eyes to count as "facing camera"
CONSECUTIVE_FRAMES_TO_LOCK = 5    # hysteresis: require a sustained gaze before locking
CONSECUTIVE_FRAMES_TO_UNLOCK = 8

# Set True to actually lock the Windows session (via ctypes) instead of just blurring the demo
# window. Left off by default so the live demo doesn't lock you out of your own machine.
ENABLE_REAL_SCREEN_LOCK = False


def lock_workstation():
    """Real deployment hook: locks the Windows session. Gated behind ENABLE_REAL_SCREEN_LOCK."""
    import ctypes
    ctypes.windll.user32.LockWorkStation()


def capture_blurred_screen():
    """Grab the actual desktop and return a heavily blurred BGR copy of it with a warning banner."""
    screenshot = np.array(ImageGrab.grab())
    frame = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(frame, (99, 99), 0)
    cv2.putText(blurred, "SCREEN LOCKED - SHOULDER SURFING DETECTED",
                (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
    return blurred


def show_privacy_overlay(blurred_screen):
    """Display the blurred screen fullscreen and on top of everything else, hiding the real desktop."""
    cv2.namedWindow(PRIVACY_WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(PRIVACY_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        cv2.setWindowProperty(PRIVACY_WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass  # older OpenCV builds may not expose this property
    cv2.imshow(PRIVACY_WINDOW, blurred_screen)


def hide_privacy_overlay():
    try:
        cv2.destroyWindow(PRIVACY_WINDOW)
    except cv2.error:
        pass


class Speaker:
    """Runs pyttsx3 on a background thread so speech never blocks the video loop."""

    def __init__(self, rate=175):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, args=(rate,), daemon=True)
        self._thread.start()

    def _worker(self, rate):
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        while True:
            text = self._queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()

    def speak(self, text):
        self._queue.put(text)

    def stop(self):
        self._queue.put(None)


def estimate_gaze(keypoints, confidences):
    """Return (is_facing_camera, gaze_origin, gaze_target) from eye/nose keypoints, or None."""
    nose_conf = confidences[NOSE]
    left_eye_conf, right_eye_conf = confidences[LEFT_EYE], confidences[RIGHT_EYE]

    if nose_conf < KEYPOINT_CONF_THRESHOLD or left_eye_conf < KEYPOINT_CONF_THRESHOLD or right_eye_conf < KEYPOINT_CONF_THRESHOLD:
        return None  # face too turned away / occluded to judge gaze reliably

    nose = keypoints[NOSE]
    left_eye, right_eye = keypoints[LEFT_EYE], keypoints[RIGHT_EYE]

    eye_center = (left_eye + right_eye) / 2
    eye_distance = np.linalg.norm(right_eye - left_eye)
    if eye_distance < 1e-3:
        return None

    # How far the nose sits off the eye midline, normalized by eye separation. A near-frontal face
    # keeps the nose close to centered; a profile/turned-away face pushes it far to one side.
    horizontal_offset = abs(nose[0] - eye_center[0]) / eye_distance
    is_facing_camera = horizontal_offset < FRONTAL_GAZE_RATIO

    return is_facing_camera, eye_center, nose


def classify_people(boxes_xywh):
    """Return the index of the Primary_Operator (largest box); everyone else is a Bystander."""
    areas = boxes_xywh[:, 2] * boxes_xywh[:, 3]
    return int(np.argmax(areas))


def run_shoulder_surfing_monitor():
    model = YOLO("yolo26n-pose.pt")

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    speaker = Speaker()
    lock_counter = 0
    unlock_counter = 0
    screen_locked = False
    blurred_screen = None
    prev_frame_time = 0

    print("Shoulder-surfing monitor running. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            results = model(frame, verbose=False)[0]
            display_frame = frame.copy()

            threat_detected = False
            bystander_gaze_count = 0
            primary_index = None

            if results.keypoints is not None and len(results.boxes) > 0:
                boxes_xywh = results.boxes.xywh.cpu().numpy()
                all_keypoints = results.keypoints.xy.cpu().numpy()
                all_confidences = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else None

                primary_index = classify_people(boxes_xywh)

                for i, (box, keypoints) in enumerate(zip(results.boxes.xyxy.cpu().numpy(), all_keypoints)):
                    confidences = all_confidences[i] if all_confidences is not None else np.ones(len(keypoints))
                    role = "Primary_Operator" if i == primary_index else "Bystander"
                    x1, y1, x2, y2 = box.astype(int)

                    gaze = estimate_gaze(keypoints, confidences)
                    label = role
                    box_color = (0, 255, 0)

                    if role == "Bystander" and gaze is not None:
                        is_facing_camera, eye_center, nose = gaze
                        if is_facing_camera:
                            label = "Bystander: GAZE ON SCREEN"
                            box_color = (0, 0, 255)
                            threat_detected = True
                            bystander_gaze_count += 1
                            cv2.arrowedLine(
                                display_frame,
                                tuple(eye_center.astype(int)),
                                tuple(nose.astype(int)),
                                (0, 0, 255),
                                2,
                                tipLength=0.4,
                            )
                        else:
                            label = "Bystander: looking away"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(display_frame, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

            # Hysteresis so the lock doesn't flicker on/off between single noisy frames.
            if threat_detected:
                lock_counter += 1
                unlock_counter = 0
            else:
                unlock_counter += 1
                lock_counter = 0

            if lock_counter >= CONSECUTIVE_FRAMES_TO_LOCK:
                if not screen_locked:
                    cause = (
                        f"{bystander_gaze_count} bystanders detected looking at the screen"
                        if bystander_gaze_count != 1 else "a bystander detected looking at the screen"
                    )
                    speaker.speak(f"Blurring screen. Cause: {cause}.")
                    blurred_screen = capture_blurred_screen()
                screen_locked = True
                if ENABLE_REAL_SCREEN_LOCK:
                    lock_workstation()
            if unlock_counter >= CONSECUTIVE_FRAMES_TO_UNLOCK:
                if screen_locked:
                    hide_privacy_overlay()
                screen_locked = False

            if screen_locked:
                show_privacy_overlay(blurred_screen)
                display_frame = cv2.GaussianBlur(display_frame, (55, 55), 0)
                cv2.putText(display_frame, "SCREEN LOCKED - SHOULDER SURFING DETECTED",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time) if prev_frame_time != 0 else 0
            prev_frame_time = current_time
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, display_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Shoulder-Surfing / Visual Privacy Monitor", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hide_privacy_overlay()
        cv2.destroyAllWindows()
        speaker.stop()


if __name__ == "__main__":
    run_shoulder_surfing_monitor()
