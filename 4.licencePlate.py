"""
License Plate & Vehicle Access Gate Pipeline.

Automated access control for a parking structure or secure facility gate.

Pipeline (multi-model workflow, mirroring the Roboflow Detection + Cropping +
OCR structure with locally available tools):
1. Detection  - YOLO (yolo26n.pt) detects vehicles (car/truck/bus/motorcycle).
2. Cropping   - classic OpenCV contour/edge analysis inside each vehicle box
   finds the high-aspect-ratio rectangular plate region and crops it.
3. OCR        - EasyOCR reads the alphanumeric string off the cropped plate.
4. Access     - the cleaned plate string is checked against an authorized
   allow-list to decide GRANTED / DENIED.

Deployment output: a dashboard combining the live camera feed, the cropped
plate snippet, the extracted text, and a GRANTED/DENIED banner.

Press 'a' to authorize the currently read plate (demo convenience).
Press 'q' to quit.

Camera source:
    On startup, a small dialog box asks for the camera source: leave it as
    "0" for the local webcam, or enter an IP camera URL, e.g.
    http://192.168.0.5:81/stream. Cancelling the dialog falls back to 0.
"""
import time
import tkinter as tk
from tkinter import simpledialog

import cv2
import numpy as np
import requests
import easyocr
from ultralytics import YOLO


def resolve_camera_source():
    """Ask for the camera source via a GUI dialog: a webcam index or an IP camera URL."""
    root = tk.Tk()
    root.withdraw()
    source = simpledialog.askstring(
        "Camera Source",
        "Enter webcam index (e.g. 0) or IP camera URL (e.g. http://192.168.0.8:81/stream):",
        initialvalue="0",
    )
    root.destroy()
    source = (source or "0").strip()
    return int(source) if source.isdigit() else source



cam = resolve_camera_source()


class MjpegStream:
    """Manual multipart MJPEG HTTP reader (e.g. ESP32-CAM/IP Webcam /stream or /video).

    OpenCV's FFMPEG backend frequently times out or desyncs ("overread", "Stream timeout
    triggered") on these continuous multipart streams, so this parses the JPEG frame
    boundaries directly instead of relying on cv2.VideoCapture for URL sources. It also
    auto-reconnects on transient read timeouts/drops, which are common over WiFi.
    """

    def __init__(self, url, connect_timeout=5, read_timeout=15, max_reconnect_attempts=5):
        self._url = url
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._max_reconnect_attempts = max_reconnect_attempts
        self._response = None
        self._iterator = None
        self._buffer = b""
        self._connect()

    def _connect(self):
        if self._response is not None:
            self._response.close()
        self._response = requests.get(
            self._url, stream=True, timeout=(self._connect_timeout, self._read_timeout))
        self._iterator = self._response.iter_content(chunk_size=1024)
        self._buffer = b""

    def isOpened(self):
        return self._response is not None and self._response.ok

    def read(self):
        for attempt in range(self._max_reconnect_attempts + 1):
            try:
                while True:
                    chunk = next(self._iterator, None)
                    if chunk is None:
                        break
                    self._buffer += chunk
                    start = self._buffer.find(b'\xff\xd8')  # JPEG start marker
                    end = self._buffer.find(b'\xff\xd9')    # JPEG end marker
                    if start != -1 and end != -1 and end > start:
                        jpg = self._buffer[start:end + 2]
                        self._buffer = self._buffer[end + 2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            return True, frame
            except requests.RequestException as e:
                print(f"MJPEG stream read error ({e}); reconnecting (attempt {attempt + 1}/{self._max_reconnect_attempts})...")

            if attempt < self._max_reconnect_attempts:
                try:
                    self._connect()
                except requests.RequestException as e:
                    print(f"MJPEG reconnect failed: {e}")

        return False, None

    def release(self):
        if self._response is not None:
            self._response.close()


def open_camera(source):
    """IP camera URLs use the manual MJPEG reader; a local webcam index uses cv2.VideoCapture."""
    if isinstance(source, str):
        return MjpegStream(source)
    return cv2.VideoCapture(source)

VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}
OCR_EVERY_N_FRAMES = 10          # EasyOCR is expensive, don't run it on every single frame
PLATE_MIN_ASPECT_RATIO = 2.0
PLATE_MAX_ASPECT_RATIO = 6.0
PLATE_MIN_WIDTH_RATIO = 0.15     # plate must be at least this fraction of the vehicle box width

# Simulated facility vehicle database. Press 'a' during the demo to add the
# currently read plate here at runtime.
AUTHORIZED_PLATES = {"ABC1234", "NDC2026"}


def find_plate_region(vehicle_roi):
    """Classic ANPR heuristic: look for a rectangular, high-aspect-ratio contour."""
    gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    min_width = vehicle_roi.shape[1] * PLATE_MIN_WIDTH_RATIO

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        aspect_ratio = w / float(h)
        if PLATE_MIN_ASPECT_RATIO <= aspect_ratio <= PLATE_MAX_ASPECT_RATIO and w > min_width:
            return x, y, w, h

    return None


def clean_plate_text(raw_text):
    return "".join(ch for ch in raw_text.upper() if ch.isalnum())


def read_plate_text(reader, plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""
    results = reader.readtext(plate_crop, detail=0)
    return clean_plate_text("".join(results))


def draw_dashboard(camera_feed, plate_crop, plate_text, access_status, pipeline_status):
    """Compose the camera feed + a sidebar with the cropped plate, text and access status."""
    feed_h, feed_w = camera_feed.shape[:2]
    sidebar_w = 320
    dashboard = np.zeros((feed_h, feed_w + sidebar_w, 3), dtype=np.uint8)
    dashboard[:, :feed_w] = camera_feed

    sidebar = dashboard[:, feed_w:]
    cv2.putText(sidebar, "VEHICLE ACCESS GATE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(sidebar, pipeline_status, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    if plate_crop is not None and plate_crop.size > 0:
        thumb = cv2.resize(plate_crop, (sidebar_w - 20, 90))
        sidebar[50:140, 10:sidebar_w - 10] = thumb
    else:
        cv2.putText(sidebar, "No plate detected", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.putText(sidebar, f"Plate: {plate_text or '---'}", (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    status_color = (0, 200, 0) if access_status == "GRANTED" else (0, 0, 255)
    cv2.rectangle(sidebar, (10, 200), (sidebar_w - 10, 260), status_color, -1)
    cv2.putText(sidebar, access_status, (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(sidebar, "Press 'a' to authorize this plate", (10, feed_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    return dashboard


def run_license_plate_gate():
    model = YOLO("yolo26n.pt")
    reader = easyocr.Reader(['en'])

    cap = open_camera(cam)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_count = 0
    plate_crop = None
    plate_text = ""
    access_status = "DENIED"
    prev_frame_time = 0

    print("License plate access gate running. Press 'a' to authorize the current plate, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        display_frame = frame.copy()
        results = model(frame, verbose=False)[0]

        vehicle_boxes = [
            box.xyxy[0].cpu().numpy().astype(int)
            for box in results.boxes
            if model.names[int(box.cls[0])] in VEHICLE_CLASS_NAMES
        ]

        pipeline_status = f"No vehicle detected ({len(results.boxes)} other objects seen)"

        for (x1, y1, x2, y2) in vehicle_boxes:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if vehicle_boxes:
            pipeline_status = f"{len(vehicle_boxes)} vehicle(s) detected - locating plate"

        if vehicle_boxes and frame_count % OCR_EVERY_N_FRAMES == 0:
            x1, y1, x2, y2 = vehicle_boxes[0]
            vehicle_roi = frame[y1:y2, x1:x2]
            plate_box = find_plate_region(vehicle_roi) if vehicle_roi.size else None

            if plate_box is not None:
                px, py, pw, ph = plate_box
                plate_crop = vehicle_roi[py:py + ph, px:px + pw]
                cv2.rectangle(display_frame, (x1 + px, y1 + py),
                              (x1 + px + pw, y1 + py + ph), (0, 0, 255), 2)
                plate_text = read_plate_text(reader, plate_crop) or plate_text
                access_status = "GRANTED" if plate_text in AUTHORIZED_PLATES else "DENIED"
                pipeline_status = f"Plate located - read '{plate_text or '?'}'"
                print(f"[frame {frame_count}] plate region found, OCR result: {plate_text!r}")
            else:
                pipeline_status = f"{len(vehicle_boxes)} vehicle(s) detected - no plate-shaped region found"
                print(f"[frame {frame_count}] {len(vehicle_boxes)} vehicle(s) but no plate-shaped contour found")

        frame_count += 1

        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = current_time
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, display_frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        dashboard = draw_dashboard(display_frame, plate_crop, plate_text, access_status, pipeline_status)
        cv2.imshow("License Plate & Vehicle Access Gate", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a") and plate_text:
            AUTHORIZED_PLATES.add(plate_text)
            access_status = "GRANTED"
            print(f"Authorized new plate: {plate_text}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_license_plate_gate()
