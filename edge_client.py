"""
edge_client.py
==============
Edge client: runs YOLOv7 *locally* on the camera frame, then sends only a
tiny JSON detection report to the cloud server (CloudServer.py, port 2222).

Two modes:
    --video <path>   run on a video file
    --live           run on webcam (default camera 0)

Measures:
    * edge_latency_ms   : local inference time per frame (YOLOv7 on edge)
    * e2e latency       : network round-trip of the JSON report
    * bandwidth         : JSON bytes vs raw image bytes (bandwidth savings)

All timings are logged to console and written to edge_report.csv.

Requires: CloudServer running (`python CloudServer.py`) and the trained
weights at models/best.pt
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_traffic import VehicleDetector  # noqa: E402

SERVER = os.environ.get("CLOUD_SERVER", "http://localhost:2222")
BASE = os.path.dirname(os.path.abspath(__file__))


def frame_bytes(frame):
    """Bytes a raw frame would occupy if uploaded uncompressed (BGR)."""
    return int(frame.shape[0] * frame.shape[1] * 3)


def main():
    parser = argparse.ArgumentParser(description="Edge client")
    parser.add_argument("--video", default=None,
                        help="video file path (default: webcam 0)")
    parser.add_argument("--weights",
                        default=os.path.join(BASE, "models", "best.pt"),
                        help="path to trained YOLOv7 weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = all)")
    parser.add_argument("--out", default=os.path.join(BASE, "edge_report.csv"),
                        help="CSV log path")
    parser.add_argument("--limit", type=float, default=None,
                        help="simulated network latency limit in ms "
                             "(None = real network)")
    args = parser.parse_args()

    print(f"Loading model {args.weights} ...")
    det = VehicleDetector(weights_path=args.weights, conf_thresh=args.conf,
                          iou_thresh=args.iou)
    print("Model loaded. Server:", SERVER)

    source = args.video if args.video else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")

    colors = {
        "car": (0, 255, 0), "threewheel": (255, 165, 0), "bus": (255, 0, 255),
        "truck": (0, 128, 255), "motorbike": (255, 255, 0), "van": (128, 0, 255)}
    frame_id = 0
    rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        # ---- 1) local inference (the "edge" part) ----
        dets, edge_latency = det.detect_frame(frame, with_time=True)

        # ---- 2) send only the JSON report to cloud ----
        payload = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "edge_latency_ms": edge_latency,
            "image_bytes": frame_bytes(frame),
            "payload_bytes": 0,  # filled below
            "count": len(dets),
            "detections": [
                {
                    "cls_name": d.cls_name,
                    "confidence": float(d.confidence),
                    "x1": float(d.x1), "y1": float(d.y1),
                    "x2": float(d.x2), "y2": float(d.y2)
                }
                for d in dets],
            "mode": "edge",
        }
        payload["payload_bytes"] = len(
            __import__("json").dumps(payload).encode())
        try:
            t0 = time.perf_counter()
            resp = requests.post(f"{SERVER}/report", json=payload, timeout=5)
            net_ms = (time.perf_counter() - t0) * 1000.0
            ok_resp = resp.ok
            e2e = resp.json().get("server_latency_ms", net_ms)
        except requests.RequestException as exc:
            print(f"[edge] send failed frame {frame_id}: {exc}")
            e2e, ok_resp = 0.0, False

        # ---- 3) draw & log ----
        for d in dets:
            c = colors.get(d.cls_name, (0, 255, 255))
            cv2.rectangle(frame, d.bbox, c, 2)
            cv2.putText(frame, f"{d.cls_name} {d.confidence:.2f}",
                        (d.bbox[0], d.bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, c, 2)
        cv2.putText(frame, f"EDGE: {edge_latency:.1f} ms  | "
                           f"cloud {e2e:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Edge Pipeline (YOLOv7 local + JSON to cloud)", frame)

        rows.append([frame_id, edge_latency, e2e, len(dets),
                     payload["payload_bytes"], frame_bytes(frame), ok_resp])

        cv2.imshow_wait = 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if args.max_frames and frame_id >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "edge_latency_ms", "cloud_e2e_ms",
                    "detected_count", "payload_bytes", "raw_bytes",
                    "uploaded_ok"])
        w.writerows(rows)

    edge_ms = [r[1] for r in rows]
    e2e_ms = [r[2] for r in rows if r[2] > 0]
    print("\n===== EDGE REPORT =====")
    print(f"frames={len(rows)}")
    if edge_ms:
        print(f"edge inference mean={sum(edge_ms)/len(edge_ms):.2f} ms "
              f"min={min(edge_ms):.2f} max={max(edge_ms):.2f}")
    if e2e_ms:
        print(f"edge->cloud e2e mean={sum(e2e_ms)/len(e2e_ms):.2f} ms")
    print(f"Report saved to {args.out}")


if __name__ == "__main__":
    main()
