import cv2
import numpy as np
import requests
import argparse
import time
import threading
from collections import deque
from ultralytics import YOLO


DEFAULT_IP = "192.168.110.56"
STREAM_PORT = 81
STREAM_PATH = "/stream"

WINDOW_NAME = "ESP32-S3 YOLO HD Stream"

STREAM_CHUNK = 8192
MAX_STREAM_BUFFER = 2_000_000

DEFAULT_INFER_SIZE = 640
DEFAULT_SKIP_FRAMES = 1

DISPLAY_MARGIN = 0.96


class MJPEGReader:
    def __init__(self, url: str):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self._fps_times = deque(maxlen=60)

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._read_loop, daemon=True)
        thread.start()
        return self

    def _read_loop(self):
        headers = {
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        }

        while self.running:
            try:
                resp = requests.get(
                    self.url,
                    stream=True,
                    timeout=10,
                    headers=headers,
                )

                buf = b""

                for chunk in resp.iter_content(chunk_size=STREAM_CHUNK):
                    if not self.running:
                        break

                    if not chunk:
                        continue

                    buf += chunk

                    if len(buf) > MAX_STREAM_BUFFER:
                        start = buf.rfind(b"\xff\xd8")
                        if start != -1:
                            buf = buf[start:]
                        else:
                            buf = b""

                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2)

                    if start != -1 and end != -1:
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]

                        img = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )

                        if img is not None:
                            with self.lock:
                                self.frame = img
                            self._fps_times.append(time.monotonic())

            except Exception as e:
                print(f"[Stream] Lỗi kết nối, thử lại... ({e})")
                time.sleep(0.5)

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    @property
    def stream_fps(self):
        t = list(self._fps_times)
        if len(t) < 2:
            return 0.0
        return (len(t) - 1) / max(t[-1] - t[0], 1e-6)

    def stop(self):
        self.running = False


class YOLOWorker:
    def __init__(self, model_path, infer_size=640, skip_frames=1):
        self.model = YOLO(model_path)
        self.infer_size = int(infer_size)
        self.skip_frames = int(skip_frames)

        self.lock = threading.Lock()
        self.input_frame = None
        self.output_boxes = []
        self.running = False

        self.frame_id = 0
        self.infer_fps = 0.0
        self.last_infer_time = time.monotonic()

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()
        return self

    def submit(self, frame):
        with self.lock:
            self.input_frame = frame

    def get_boxes(self):
        with self.lock:
            return list(self.output_boxes), self.infer_fps

    def _resize_for_infer(self, frame):
        h, w = frame.shape[:2]

        if w >= h:
            new_w = self.infer_size
            new_h = int(h * self.infer_size / w)
        else:
            new_h = self.infer_size
            new_w = int(w * self.infer_size / h)

        new_w = max(32, (new_w // 32) * 32)
        new_h = max(32, (new_h // 32) * 32)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        sx = w / new_w
        sy = h / new_h

        return resized, sx, sy

    def _loop(self):
        while self.running:
            with self.lock:
                frame = None if self.input_frame is None else self.input_frame.copy()
                self.input_frame = None

            if frame is None:
                time.sleep(0.002)
                continue

            self.frame_id += 1

            if self.skip_frames > 0:
                if self.frame_id % (self.skip_frames + 1) != 0:
                    continue

            small, sx, sy = self._resize_for_infer(frame)

            result = self.model.predict(
                small,
                verbose=False,
                imgsz=self.infer_size,
            )[0]

            boxes = []

            if result.boxes is not None:
                names = result.names

                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = names.get(cls_id, str(cls_id))

                    x1, y1, x2, y2 = xyxy

                    x1 = int(x1 * sx)
                    y1 = int(y1 * sy)
                    x2 = int(x2 * sx)
                    y2 = int(y2 * sy)

                    boxes.append((x1, y1, x2, y2, conf, label))

            now = time.monotonic()
            self.infer_fps = 1.0 / max(now - self.last_infer_time, 1e-6)
            self.last_infer_time = now

            with self.lock:
                self.output_boxes = boxes

    def stop(self):
        self.running = False


def get_screen_size():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1280, 720


def resize_to_screen(frame, screen_w, screen_h):
    h, w = frame.shape[:2]

    max_w = int(screen_w * DISPLAY_MARGIN)
    max_h = int(screen_h * DISPLAY_MARGIN)

    scale = min(max_w / w, max_h / h)

    if scale <= 0:
        return frame

    new_w = int(w * scale)
    new_h = int(h * scale)

    if scale > 1.0:
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_AREA

    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def draw_boxes(frame, boxes):
    for x1, y1, x2, y2, conf, label in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)

        text = f"{label} {conf:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 180, 255),
            2,
        )


def draw_overlay(frame, stream_fps, infer_fps, display_fps, infer_size, skip_frames):
    h, w = frame.shape[:2]

    lines = [
        f"Source: {w}x{h}",
        f"Stream FPS: {stream_fps:.1f}",
        f"Infer FPS: {infer_fps:.1f}",
        f"Display FPS: {display_fps:.1f}",
        f"Infer size: {infer_size}",
        f"Skip frames: {skip_frames}",
        "Q: quit | F: fullscreen/window",
    ]

    x = 10
    y = 24

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            1,
        )
        y += 24


def run(ip, model_path, infer_size, skip_frames, fullscreen):
    url = f"http://{ip}:{STREAM_PORT}{STREAM_PATH}"

    print(f"[Stream] Connecting to {url}")
    print(f"[YOLO] Loading model: {model_path}")

    reader = MJPEGReader(url).start()
    worker = YOLOWorker(
        model_path=model_path,
        infer_size=infer_size,
        skip_frames=skip_frames,
    ).start()

    for _ in range(80):
        if reader.read() is not None:
            break
        time.sleep(0.1)
    else:
        print("Không nhận được frame. Kiểm tra IP/WiFi/ESP32-CAM.")
        reader.stop()
        worker.stop()
        return

    screen_w, screen_h = get_screen_size()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW_NAME, int(screen_w * DISPLAY_MARGIN), int(screen_h * DISPLAY_MARGIN))

    print("Stream OK. Nhấn Q để thoát, F để bật/tắt fullscreen.")

    t_last = time.monotonic()
    display_fps = 0.0
    is_fullscreen = fullscreen

    while True:
        frame = reader.read()

        if frame is None:
            continue

        worker.submit(frame)

        boxes, infer_fps = worker.get_boxes()

        view = frame.copy()
        draw_boxes(view, boxes)

        now = time.monotonic()
        display_fps = 1.0 / max(now - t_last, 1e-6)
        t_last = now

        draw_overlay(
            view,
            stream_fps=reader.stream_fps,
            infer_fps=infer_fps,
            display_fps=display_fps,
            infer_size=infer_size,
            skip_frames=skip_frames,
        )

        if not is_fullscreen:
            view = resize_to_screen(view, screen_w, screen_h)

        cv2.imshow(WINDOW_NAME, view)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("f"):
            is_fullscreen = not is_fullscreen

            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, int(screen_w * DISPLAY_MARGIN), int(screen_h * DISPLAY_MARGIN))

    reader.stop()
    worker.stop()
    cv2.destroyAllWindows()
    print("Đã dừng.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=DEFAULT_IP, help="IP của ESP32-S3")
    ap.add_argument("--model", default="yolov8n.pt", help="Model YOLO")
    ap.add_argument("--infer", type=int, default=DEFAULT_INFER_SIZE, help="Kích thước ảnh cho YOLO, ví dụ 640/800/960")
    ap.add_argument("--skip", type=int, default=DEFAULT_SKIP_FRAMES, help="Skip frame để giảm lag, ví dụ 0/1/2")
    ap.add_argument("--fullscreen", action="store_true", help="Hiển thị full màn hình")
    args = ap.parse_args()

    run(
        ip=args.ip,
        model_path=args.model,
        infer_size=args.infer,
        skip_frames=args.skip,
        fullscreen=args.fullscreen,
    )