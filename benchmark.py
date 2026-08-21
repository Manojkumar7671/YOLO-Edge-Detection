"""
benchmark.py
============
Headless edge-vs-cloud benchmark for the project
"Deep Learning for Edge Computing in Indian Smart Cities: Real-Time Analytics
on Low Power Devices".

Measures (no GUI, no webcam):
    1. Edge pipeline:  local YOLOv7 inference latency + small JSON report
                       upload to CloudServer
    2. Cloud pipeline: raw image upload + server-side detection latency

Requires CloudServer running on port 2222:
    python CloudServer.py
    python benchmark.py --video sample.mp4 --frames 100

Outputs edge_report.csv / cloud_report.csv and prints a summary table.
"""

import argparse
import csv
import io
import json
import os
import sys
import time

import cv2
import numpy as np
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from yolo_traffic import VehicleDetector  # noqa: E402

SERVER = os.environ.get("CLOUD_SERVER", "http://localhost:2222")


def raw_jpeg_bytes(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(buf.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="video file")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--weights", default=os.path.join(BASE, "models",
                                                          "best.pt"))
    args = parser.parse_args()

    print("Loading model ...")
    det = VehicleDetector(weights_path=args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    frames = []
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"{len(frames)} frames sampled")

    # ---------------- EDGE pipeline ----------------
    edge_rows = []
    print("\n--- EDGE pipeline (local inference + JSON report) ---")
    for i, frame in enumerate(frames, 1):
        dets, lat = det.detect_frame(frame, with_time=True)
        payload = {
            "frame_id": i, "timestamp": time.time(),
            "edge_latency_ms": lat, "image_bytes": raw_jpeg_bytes(frame),
            "payload_bytes": 0, "count": len(dets),
            "detections": [
                {"cls_name": d.cls_name, "confidence": float(d.confidence),
                 "x1": float(d.x1), "y1": float(d.y1),
                 "x2": float(d.x2), "y2": float(d.y2)}
                for d in dets],
            "mode": "edge"}
        payload["payload_bytes"] = len(json.dumps(payload).encode())
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{SERVER}/report", json=payload, timeout=5)
            net_ms = (time.perf_counter() - t0) * 1000.0
            e2e = resp.json().get("server_latency_ms", net_ms) if resp.ok \
                else 0.0
        except requests.RequestException:
            net_ms = e2e = 0.0
        edge_rows.append([i, lat, net_ms, e2e, len(dets),
                          payload["payload_bytes"], raw_jpeg_bytes(frame)])
        if i % 20 == 0:
            print(f"  {i}/{len(frames)}")
    with open(os.path.join(BASE, "edge_report.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "edge_latency_ms", "network_ms",
                    "e2e_ms", "count", "payload_bytes", "raw_bytes"])
        w.writerows(edge_rows)

    # ---------------- CLOUD pipeline ----------------
    cloud_rows = []
    print("\n--- CLOUD pipeline (raw upload + server detect) ---")
    for i, frame in enumerate(frames, 1):
        _, buf = cv2.imencode(".jpg", frame,
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
        raw = buf.tobytes()
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{SERVER}/detect_raw",
                                 files={"image": (f"f{i}.jpg", raw,
                                                  "image/jpeg")},
                                 data={"frame_id": i}, timeout=30)
            if resp.ok:
                d = resp.json()
                cloud_ms = d["cloud_latency_ms"]
                server_ms = d.get("server_latency_ms", 0)
                count = d["count"]
            else:
                cloud_ms = server_ms = count = 0
        except requests.RequestException:
            cloud_ms = server_ms = count = 0
        total = (time.perf_counter() - t0) * 1000.0
        cloud_rows.append([i, len(raw), server_ms, cloud_ms, total, count])
        if i % 20 == 0:
            print(f"  {i}/{len(frames)}")
    with open(os.path.join(BASE, "cloud_report.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "raw_bytes", "server_ms", "cloud_ms",
                    "round_trip_ms", "count"])
        w.writerows(cloud_rows)

    # ---------------- summary ----------------
    n = len(edge_rows)
    edge_lat = [r[1] for r in edge_rows]
    edge_net = [r[2] for r in edge_rows if r[2] > 0]
    edge_e2e = [r[3] for r in edge_rows if r[3] > 0]
    cloud_rt = [r[4] for r in cloud_rows if r[4] > 0]
    edge_bytes = sum(r[5] for r in edge_rows)
    raw_bytes = sum(r[1] for r in cloud_rows)

    print("\n================ BENCHMARK SUMMARY ================")
    print(f"frames measured        : {n}")
    print(f"edge inference mean    : {sum(edge_lat)/n:.2f} ms "
          f"(min {min(edge_lat):.2f}, max {max(edge_lat):.2f})")
    print(f"edge report network RTT: {sum(edge_net)/len(edge_net):.2f} ms "
          f"(mean, {len(edge_net)} frames)")
    print(f"cloud round-trip mean  : {sum(cloud_rt)/len(cloud_rt):.2f} ms "
          f"(min {min(cloud_rt):.2f}, max {max(cloud_rt):.2f})")
    print(f"speedup (cloud/edge)   : "
          f"{(sum(cloud_rt)/len(cloud_rt))/(sum(edge_lat)/n):.2f}x")
    print(f"data edge sent         : {edge_bytes/1e6:.3f} MB (JSON)")
    print(f"data cloud sent        : {raw_bytes/1e6:.3f} MB (raw JPEG)")
    print(f"bandwidth reduction    : {(1-edge_bytes/raw_bytes)*100:.1f}%")
    print("===================================================")


if __name__ == "__main__":
    main()
