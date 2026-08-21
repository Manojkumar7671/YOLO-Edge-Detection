"""
yolo_traffic.py
===============
YOLOv7 traffic vehicle detection module for the project
"Deep Learning for Edge Computing in Indian Smart Cities: Real-Time Analytics on
Low Power Devices".

This module wraps the trained YOLOv7 model and exposes a simple interface for
vehicle detection on images, video frames and webcam feeds.

Classes detected: car, threewheel, bus, truck, motorbike, van

Usage:
    detector = VehicleDetector(weights_path="models/best.pt")
    results = detector.detect_image("test.jpg")
    frame = detector.detect_frame(frame)          # OpenCV BGR frame
"""

import os
import time
import sys
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(PROJECT_DIR, "models", "best.pt")
DEFAULT_CLASSES = ["car", "threewheel", "bus", "truck", "motorbike", "van"]


class DetectionResult:
    """Container for one detection: class, confidence, bounding box."""

    def __init__(self, cls_id, cls_name, confidence, x1, y1, x2, y2):
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.confidence = float(confidence)
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def bbox(self):
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


class VehicleDetector:
    """YOLOv7 vehicle detector with non-maximum suppression."""

    def __init__(self, weights_path=None, img_size=320, conf_thresh=0.25,
                 iou_thresh=0.45):
        """Load the trained YOLOv7 model once (this is the slow part,
        amortised over all frames -> fast per-frame inference)."""
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        weights = weights_path or DEFAULT_WEIGHTS
        weights = os.path.abspath(weights)
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Weights file not found: {weights}\n"
                "Run training first (see README) or provide the path to "
                "runs/train/veh_cpu/weights/best.pt")

        # ---- build the YOLOv7 model and load fine-tuned weights ----
        try:
            import sys
            # Check for YOLOv7 repo in sibling or project root
            repo_candidates = [
                os.path.join(PROJECT_DIR, "yolov7"),
                os.path.join(PROJECT_DIR, "..", "yolov7")
            ]
            for p in repo_candidates:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
            from models.experimental import attempt_load  # YOLOv7 repo
        except ImportError:
            from models.experimental import attempt_load

        self.model = attempt_load(weights, map_location=self.device)
        self.model.to(self.device).eval()

        names = self.model.names
        if isinstance(names, dict):
            self.names = {int(k): str(v) for k, v in names.items()}
        else:  # list of class names
            self.names = {i: str(v) for i, v in enumerate(names)}
        self.stride = int(self.model.stride.max())

    # -----------------------------------------------------------------------
    # Core detection helpers
    # -----------------------------------------------------------------------
    def _preprocess(self, img):
        """Letterbox resize + normalisation + batch tensor."""
        import torch

        h0, w0 = img.shape[:2]
        r = min(self.img_size / h0, self.img_size / w0)
        h, w = int(h0 * r), int(w0 * r)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.img_size, self.img_size, 3), 114, np.uint8)
        top, left = (self.img_size - h) // 2, (self.img_size - w) // 2
        canvas[top:top + h, left:left + w] = img

        canvas = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
        canvas = np.ascontiguousarray(canvas)
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device), (w0, h0), r, (left, top)

    def _nms(self, boxes, scores):
        """Non-maximum suppression (cv2.dnn.NMSBoxes) -> keep best boxes."""
        boxes_xywh = np.array([
            [b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes])
        idxs = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores, self.conf_thresh, self.iou_thresh)
        return [int(i) for i in (idxs.flatten() if len(idxs) else [])]

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def detect_image(self, path, with_time=False):
        """Run detection on an image file. Returns (results, latency_ms?)."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.detect_frame(img, with_time=with_time)

    def detect_frame(self, frame, with_time=False):
        """Run detection on one OpenCV BGR frame.

        Returns:
            detections - list[DetectionResult]
            latency    - inference time in ms (only if with_time=True)
        """
        import torch

        t0 = time.perf_counter()
        tensor, orig_size, ratio, pad = self._preprocess(frame)

        with torch.no_grad():
            out, _ = self.model(tensor, augment=False)
        out = out[0].cpu().numpy()

        # decode predictions (XYWH -> XYXY) and scale back to original size
        w0, h0 = orig_size
        left, top = pad
        detections = []
        for row in out:
            cx, cy, bw, bh, conf = row[:5]
            if conf < self.conf_thresh:
                continue
            cls = int(row[5])
            x1 = ((cx - bw / 2) - left) / ratio
            y1 = ((cy - bh / 2) - top) / ratio
            x2 = ((cx + bw / 2) - left) / ratio
            y2 = ((cy + bh / 2) - top) / ratio
            detections.append(DetectionResult(
                cls, self.names.get(cls, str(cls)), conf,
                max(x1, 0), max(y1, 0), min(x2, w0), min(y2, h0)))

        latency = (time.perf_counter() - t0) * 1000.0  # ms

        if detections:
            keep = self._nms([d.bbox for d in detections],
                             [d.confidence for d in detections])
            detections = [detections[i] for i in keep]

        if with_time:
            return detections, latency
        return detections

    def detect_video(self, source, output=None, show=False,
                     on_frame=None):
        """Run detection on a video file / webcam feed (0, 1, ...).

        on_frame(frame, detections, latency_ms) is called per frame so the
        GUI can draw boxes itself. Returns when the video ends or 'q' is
        pressed.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        writer = None
        if output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                detections, latency = self.detect_frame(frame, with_time=True)
                if on_frame is not None:
                    frame = on_frame(frame, detections, latency)
                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow("Vehicle Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
