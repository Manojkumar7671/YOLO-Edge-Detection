"""
cloud_client.py
===============
Cloud-only client: uploads *raw* image frames to the cloud server over the
network and waits for a detection report back.  This simulates a naive
smart-city pipeline with no edge device (everything computed in the cloud).

    [Traffic camera] --upload raw image (MB)--> [CloudServer]
         --detect in cloud--> send JSON back

Used for comparison with edge_client.py:
    python cloud_client.py --video <path>   # uploads each frame

Add a POST /detect_raw endpoint to CloudServer before running.
"""

import argparse
import base64
import csv
import io
import os
import sys
import time

import cv2
import requests

SERVER = os.environ.get("CLOUD_SERVER", "http://localhost:2222")
BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Cloud-only client")
    parser.add_argument("--video", default=None,
                        help="video file path (default: webcam 0)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = all)")
    parser.add_argument("--out", default=os.path.join(BASE, "cloud_report.csv"),
                        help="CSV log path")
    args = parser.parse_args()

    source = args.video if args.video else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")

    frame_id = 0
    rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        # ---- upload the RAW frame (JPEG) to the cloud ----
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        raw_bytes = len(buf.tobytes())

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{SERVER}/detect_raw",
                files={"image": (f"frame{frame_id}.jpg", buf.tobytes(),
                                 "image/jpeg")},
                data={"frame_id": frame_id},
                timeout=30)
            if resp.ok:
                data = resp.json()
                cloud_ms = data["cloud_latency_ms"]
                count = data["count"]
                server_ms = data.get("server_latency_ms", 0)
            else:
                print(f"frame {frame_id} error {resp.status_code}")
                cloud_ms = server_ms = count = -1
        except requests.RequestException as exc:
            print(f"frame {frame_id} failed: {exc}")
            cloud_ms = server_ms = count = -1
        total_ms = (time.perf_counter() - t0) * 1000.0

        rows.append([frame_id, raw_bytes, server_ms, cloud_ms, total_ms, count])
        print(f"frame {frame_id}: upload={raw_bytes} bytes, "
              f"round-trip={total_ms:.1f} ms, count={count}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if args.max_frames and frame_id >= args.max_frames:
            break

    cap.release()

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "raw_bytes", "server_ms", "cloud_ms",
                    "round_trip_ms", "detected_count"])
        w.writerows(rows)

    rts = [r[4] for r in rows if r[4] > 0]
    if rts:
        print("\n===== CLOUD REPORT =====")
        print(f"frames={len(rows)} mean_round_trip={sum(rts)/len(rts):.2f} ms "
              f"min={min(rts):.2f} max={max(rts):.2f}")
        raw_total = sum(r[1] for r in rows)
        print(f"raw bytes uploaded={raw_total} bytes "
              f"({raw_total/1024/1024:.2f} MB)")
    print(f"Report saved to {args.out}")


if __name__ == "__main__":
    main()
