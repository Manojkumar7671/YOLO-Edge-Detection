"""
CloudServer.py
==============
Cloud analytics server for the project
"Deep Learning for Edge Computing in Indian Smart Cities: Real-Time Analytics
on Low Power Devices".

The cloud server receives *processed* detection reports forwarded by the
lightweight edge device (instead of raw camera images).  This simulates the
edge-cloud pipeline used in smart-city traffic analytics:

    [Traffic camera] --run YOLOv7 locally--> [edge device]
         --only JSON detection report (few KB)--> [CloudServer]

Because only the tiny JSON payload travels over the network, the edge-cloud
pipeline uses a fraction of the bandwidth that uploading raw images would
need, and the round-trip time per frame drops as well.

Run:
    pip install fastapi uvicorn
    python CloudServer.py          # starts on port 2222

Endpoints:
    POST /report      receive a detection report  {timestamp, frame_id,
                    latency_ms, count, detections[...], image_bytes?}
    GET  /report_log  list of all received reports (with arrival latency)
    GET  /stats       summary statistics + bandwidth comparison
    GET  /health      health check
"""

import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Smart City Cloud Analytics Server")

REPORTS: list = []                # received detection reports
_SERVER_DETECTOR = None       # lazily loaded server-side YOLOv7 model
RAW_SIZES: list = []              # simulated raw-image upload sizes (bytes)
START = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "cloud_log.json")


class Detection(BaseModel):
    cls_name: str
    confidence: float
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0


class Report(BaseModel):
    """Payload sent by the edge client after local inference."""
    frame_id: int
    timestamp: float
    edge_latency_ms: float = 0.0        # local inference time (ms)
    image_bytes: int = 0                # raw frame size (bytes)
    payload_bytes: int = 0              # JSON payload size (bytes)
    count: int = 0
    detections: list[Detection] = []
    mode: str = "edge"                  # "edge" | "cloud"


@app.on_event("shutdown")
def save_log():
    try:
        with open(LOG_FILE, "w") as fh:
            json.dump(REPORTS, fh, indent=1)
    except OSError:
        pass


@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - START, 1),
            "reports": len(REPORTS)}


@app.post("/report")
def receive_report(report: Report):
    """Receive one detection report from an edge device."""
    arrived = time.time()
    e2e_latency = (arrived - report.timestamp) * 1000.0  # ms

    entry = {
        "frame_id": report.frame_id,
        "mode": report.mode,
        "server_arrival_ms": round(e2e_latency, 3),
        "edge_latency_ms": round(report.edge_latency_ms, 3),
        "total_latency_ms": round(e2e_latency + report.edge_latency_ms, 3),
        "count": report.count,
        "image_bytes": report.image_bytes,
        "payload_bytes": report.payload_bytes,
        "timestamp": report.timestamp,
        "detections": [
            {"cls": d.cls_name, "conf": round(d.confidence, 3)}
            for d in report.detections],
    }
    REPORTS.append(entry)
    RAW_SIZES.append(report.image_bytes)
    print(f"[Cloud] frame {report.frame_id} ({report.mode}) "
          f"count={report.count} e2e={e2e_latency:.2f} ms")
    return {"accepted": True, "server_latency_ms": round(e2e_latency, 3)}


@app.get("/report_log")
def report_log(limit: int = 1000):
    return REPORTS[-limit:]


@app.post("/detect_raw")
async def detect_raw(image: UploadFile = File(...),
                     frame_id: int = Form(0)):
    """Naive cloud-only path: receive a RAW image, detect on the server.
    Used only for the edge-vs-cloud comparison (cloud_client.py / benchmark).
    In a real deployment the server would run the same YOLOv7 model."""
    global _SERVER_DETECTOR
    t0 = time.time()
    data = await image.read()
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "invalid image", "cloud_latency_ms": 0, "count": 0}

    if _SERVER_DETECTOR is None:
        try:
            from yolo_traffic import VehicleDetector
            weights = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", "best.pt")
            if os.path.isfile(weights):
                _SERVER_DETECTOR = VehicleDetector(weights_path=weights)
                print("[Cloud] server-side detector loaded")
            else:
                print("[Cloud] best.pt not found -> cloud detect disabled")
        except Exception as exc:
            print(f"[Cloud] server detector error: {exc}")

    count = 0
    if _SERVER_DETECTOR is not None:
        dets, _ = _SERVER_DETECTOR.detect_frame(frame)
        count = len(dets)
    cloud_ms = (time.time() - t0) * 1000.0
    server_ms = (time.time() - t0) * 1000.0
    print(f"[Cloud] raw frame {frame_id}: {len(data)/1024:.0f} KB, "
          f"detect {cloud_ms:.1f} ms, count={count}")
    return {"frame_id": frame_id,
            "server_latency_ms": round(server_ms, 3),
            "cloud_latency_ms": round(cloud_ms, 3),
            "raw_bytes": len(data),
            "count": count}


@app.get("/stats")
def stats():
    if not REPORTS:
        return {"message": "No reports received yet. Send POST /report first."}

    by_mode = defaultdict(list)
    for r in REPORTS:
        by_mode[r["mode"]].append(r)

    def summary(rows):
        if not rows:
            return None
        n = len(rows)
        total = [r["total_latency_ms"] for r in rows]
        return {
            "frames": n,
            "mean_total_latency_ms": round(sum(total) / n, 3),
            "min_total_latency_ms": round(min(total), 3),
            "max_total_latency_ms": round(max(total), 3),
            "mean_server_latency_ms": round(
                sum(r["server_arrival_ms"] for r in rows) / n, 3),
            "total_detections": sum(r["count"] for r in rows),
            "total_bytes_sent": sum(r["payload_bytes"] for r in rows),
            "raw_bytes_if_uploaded": sum(r["image_bytes"] for r in rows),
        }

    edge = summary(by_mode.get("edge", []))
    cloud = summary(by_mode.get("cloud", []))

    # bandwidth comparison: JSON report vs raw image upload
    report_bytes = sum(r["payload_bytes"] for r in REPORTS)
    raw_bytes = sum(r["image_bytes"] for r in REPORTS)
    reduction = (1 - report_bytes / raw_bytes) * 100 if raw_bytes else 0

    return {
        "total_frames": len(REPORTS),
        "uptime_s": round(time.time() - START, 1),
        "edge_mode": edge,
        "cloud_mode": cloud,
        "bandwidth": {
            "json_bytes_sent": report_bytes,
            "raw_image_bytes_if_uploaded": raw_bytes,
            "bandwidth_reduction_pct": round(reduction, 2),
        },
    }


if __name__ == "__main__":
    print("CloudServer starting on http://0.0.0.0:2222")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    uvicorn.run(app, host="0.0.0.0", port=2222, log_level="warning")
