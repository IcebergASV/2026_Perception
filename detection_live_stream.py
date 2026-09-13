#!/usr/bin/env python3
"""
Live Buoy Detection - Jetson streaming to Ground Station
Run this on the Jetson. Open http://<JETSON_IP>:5000 on the ground station.

Requires control.py to be in the same directory.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time
from control import RealSenseCamera, quitThread, running


MODEL_PATH = "/home/icebergasv/iceberg-gpt.pt"
STREAM_PORT = 5000
JPEG_QUALITY = 80
CONF_THRESH = 0.4


class BuoyDetector:

    def __init__(self, model_path: str, conf_thresh: float):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        print("model loaded")

    def predict(self, frame: np.ndarray):
        """runs inference on a frame and returns the results."""
        return self.model(frame, conf=self.conf_thresh, verbose=False)

    def annotate(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of the frame."""
        annotated = frame.copy()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label  = f"{class_name} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw, y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, ts, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated


class FrameBuffer:
    """Thread-safe buffer to share the latest JPEG frame between threads."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def write(self, jpeg_bytes: bytes):
        with self._lock:
            self._frame = jpeg_bytes

    def read(self):
        with self._lock:
            return self._frame

    def is_ready(self) -> bool:
        with self._lock:
            return self._frame is not None


class CameraThread:
    """Runs the camera + inference loop in a background thread."""

    def __init__(self, buffer: FrameBuffer, jpeg_quality: int):
        self.camera = RealSenseCamera()
        self.detector = BuoyDetector(MODEL_PATH, CONF_THRESH)
        self.buffer = buffer
        self.jpeg_quality = jpeg_quality
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.camera.start()
        self._warmup()
        self._thread.start()
        print("detection loop started")

    def _warmup(self):
        print("warming up camera")
        for _ in range(5):
            self.camera.getColorFrame()
        print("done")

    def _loop(self):
        global running
        try:
            while running:
                color_frame, _ = self.camera.getColorFrame()

                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())

                results   = self.detector.predict(frame)
                annotated = self.detector.annotate(frame, results)

                _, jpeg = cv2.imencode(
                    '.jpg', annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                self.buffer.write(jpeg.tobytes())

        finally:
            self.camera.stop()


class StreamServer:
    """Serves the annotated frames as an MJPEG stream over HTTP."""

    def __init__(self, buffer: FrameBuffer, port: int):
        self.buffer = buffer
        self.port = port

    def start(self):
        buffer = self.buffer

        class Handler(BaseHTTPRequestHandler):

            def log_message(self, format, *args):
                pass  # suppress access log spam

            def do_GET(self):
                if self.path == "/":
                    self._serve_page()
                elif self.path == "/stream":
                    self._serve_stream()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_page(self):
                html = b"""
                <html>
                <head>
                    <title>Buoy Detection - Live Feed</title>
                    <style>
                        body { background: #111; display: flex; justify-content: center;
                               align-items: center; height: 100vh; margin: 0; flex-direction: column; }
                        img  { max-width: 100%; border: 2px solid #0f0; }
                        h2   { color: #0f0; font-family: monospace; text-align: center; }
                    </style>
                </head>
                <body>
                    <h2>&#128258; Live Buoy Detection</h2>
                    <img src="/stream" />
                </body>
                </html>
                """
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html)

            def _serve_stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = buffer.read()
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" +
                            frame +
                            b"\r\n"
                        )
                        self.wfile.flush()
                        time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Ground station disconnected cleanly

        server = HTTPServer(("0.0.0.0", self.port), Handler)
        print(f"[StreamServer] Live at http://<JETSON_IP>:{self.port}")
        print("[StreamServer] Open that URL in a browser on your ground station.")
        print("[StreamServer] Press Ctrl+C to stop.\n")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[StreamServer] Shutting down.")
            server.shutdown()


class App:
    """Top-level class that wires everything together."""

    def __init__(self):
        self.buffer = FrameBuffer()
        self.camera_thread = CameraThread(self.buffer, JPEG_QUALITY)
        self.stream_server = StreamServer(self.buffer, STREAM_PORT)

    def run(self):
        control_thread = threading.Thread(target=quitThread, daemon=True)
        control_thread.start()

        self.camera_thread.start()

        print("waiting for first frame")
        while not self.buffer.is_ready():
            time.sleep(0.1)

        self.stream_server.start()


if __name__ == "__main__":
    App().run()
