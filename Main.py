"""
Main.py
=======
GUI entry point for the project
"Deep Learning for Edge Computing in Indian Smart Cities: Real-Time Analytics
on Low Power Devices".

Five modules, one window:

    1. Load Model        - load the fine-tuned YOLOv7 weights (once)
    2. Run Detection     - webcam / video detection with the trained model
    3. Cloud Report      - fetch analytics from the cloud server (/stats)
    4. Edge Report       - run the edge pipeline and save edge_report.csv
    5. Compare (Graph)   - edge-vs-cloud latency + bandwidth graph
                           (matplotlib)

Requires:
    - CloudServer.py  running on port 2222  (modules 3, 5)
    - models/best.pt  trained weights       (modules 1, 2, 4, 5)

Run:
    python Main.py
"""

import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# make `from yolo_traffic import VehicleDetector` work next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                      # noqa: E402
import numpy as np              # noqa: E402

try:
    from yolo_traffic import VehicleDetector  # noqa: E402
    HAVE_DETECTOR = True
except Exception:
    HAVE_DETECTOR = False

try:
    import requests  # noqa: E402
    HAVE_HTTP = True
except Exception:
    HAVE_HTTP = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER = os.environ.get("CLOUD_SERVER", "http://localhost:2222")
WEIGHTS = os.path.join(BASE_DIR, "models", "best.pt")
EDGE_CSV = os.path.join(BASE_DIR, "edge_report.csv")
CLOUD_CSV = os.path.join(BASE_DIR, "cloud_report.csv")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("DL for Edge Computing - Indian Smart Cities "
                        "Traffic Analytics")
        self.root.geometry("760x460")
        self.root.configure(bg="#f5f7fa")

        self.detector = None
        self.cap = None
        self.running = False

        style = ttk.Style()
        style.configure("TButton", padding=6, font=("Helvetica", 11))
        style.configure("TLabel", background="#f5f7fa", font=("Helvetica", 11))

        header = ttk.Label(root,
                           text="YOLOv7 Vehicle Detection | Edge vs Cloud",
                           font=("Helvetica", 15, "bold"),
                           background="#f5f7fa")
        header.pack(pady=(14, 4))
        self.status = ttk.Label(root, text="Status: idle",
                                font=("Helvetica", 10), background="#f5f7fa")
        self.status.pack(pady=(0, 8))

        btns = tk.Frame(root, bg="#f5f7fa")
        btns.pack(pady=6)
        ttk.Button(btns, text="1. Load Model",
                   command=self.load_model).grid(row=0, column=0, padx=6,
                                                 pady=4)
        ttk.Button(btns, text="2. Run Detection (webcam)",
                   command=self.run_detection).grid(row=0, column=1, padx=6,
                                                    pady=4)
        ttk.Button(btns, text="3. Run on Video",
                   command=self.run_video).grid(row=0, column=2, padx=6,
                                                pady=4)
        ttk.Button(btns, text="4. Cloud Report",
                   command=self.cloud_report).grid(row=1, column=0, padx=6,
                                                   pady=4)
        ttk.Button(btns, text="5. Edge Report (CLI)",
                   command=self.edge_report).grid(row=1, column=1, padx=6,
                                                  pady=4)
        ttk.Button(btns, text="6. Compare Edge vs Cloud",
                   command=self.compare).grid(row=1, column=2, padx=6, pady=4)

        info = ttk.Label(root,
                         text="Start CloudServer.py (port 2222) before using "
                              "modules 4-6.\nWeights: models/best.pt",
                         font=("Helvetica", 9), background="#f5f7fa",
                         justify="left")
        info.pack(pady=(12, 4))

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------
    def set_status(self, msg):
        self.status.config(text="Status: " + msg)

    def draw(self, frame, dets, latency):
        colors = {"car": (0, 255, 0), "threewheel": (255, 165, 0),
                  "bus": (255, 0, 255), "truck": (0, 128, 255),
                  "motorbike": (255, 255, 0), "van": (128, 0, 255)}
        for d in dets:
            c = colors.get(d.cls_name, (0, 255, 255))
            cv2.rectangle(frame, d.bbox, c, 2)
            cv2.putText(frame, f"{d.cls_name} {d.confidence:.2f}",
                        (d.bbox[0], d.bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, c, 2)
        cv2.putText(frame, f"inference {latency:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame

    # -----------------------------------------------------------------------
    # modules
    # -----------------------------------------------------------------------
    def load_model(self):
        if not HAVE_DETECTOR:
            messagebox.showerror("Missing dependency",
                                 "yolo_traffic.py import failed. Install "
                                 "requirements (see README).")
            return
        path = filedialog.askopenfilename(
            title="Select best.pt (trained YOLOv7 weights)",
            initialdir=os.path.join(BASE_DIR, "models"),
            filetypes=[("Weights", "*.pt")])
        if not path:
            return
        try:
            self.set_status("loading model ...")
            self.detector = VehicleDetector(weights_path=path)
            self.set_status("model loaded OK")
            messagebox.showinfo("Model", "Model loaded successfully.\n"
                                         "You can now run detection.")
        except Exception as exc:
            self.set_status("model load failed")
            messagebox.showerror("Error", str(exc))

    def run_detection(self):
        if self.detector is None:
            messagebox.showinfo("Load model first",
                                "Click '1. Load Model' and select "
                                "models/best.pt first.")
            return
        if self.running:
            return
        self.running = True
        self.set_status("webcam detection running (press 'q' to stop)")

        def loop():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.set_status("webcam not available")
                return
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                dets, lat = self.detector.detect_frame(frame, with_time=True)
                frame = self.draw(frame, dets, lat)
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            self.set_status("idle")

        t = tk.Thread if hasattr(tk, "Thread") else None
        import threading
        threading.Thread(target=loop, daemon=True).start()

    def run_video(self):
        if self.detector is None:
            messagebox.showinfo("Load model first",
                                "Click '1. Load Model' and select "
                                "models/best.pt first.")
            return
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.set_status("video detection running (press 'q' to stop)")

        def loop():
            cap = cv2.VideoCapture(path)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                dets, lat = self.detector.detect_frame(frame, with_time=True)
                frame = self.draw(frame, dets, lat)
                cv2.imshow("Detection", frame)
                if cv2.waitKey(int(1000 / (cap.get(cv2.CAP_PROP_FPS) or 30)))\
                        & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
            self.set_status("idle")

        import threading
        threading.Thread(target=loop, daemon=True).start()

    def cloud_report(self):
        if not HAVE_HTTP:
            messagebox.showerror("Error", "Install 'requests'.")
            return
        try:
            r = requests.get(f"{SERVER}/stats", timeout=5)
            r.raise_for_status()
            s = r.json()
        except Exception as exc:
            messagebox.showerror("Cloud", f"Cannot reach {SERVER}: {exc}")
            return
        box = tk.Toplevel(self.root)
        box.title("Cloud Report")
        box.geometry("640x420")
        txt = tk.Text(box, wrap="word", font=("Courier", 10))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        import json
        txt.insert("1.0", json.dumps(s, indent=2))

    def edge_report(self):
        """Run the edge pipeline as a subprocess (same CLI you can call
        directly) and stream its console output into a window."""
        cmd = [sys.executable, os.path.join(BASE_DIR, "edge_client.py"),
               "--max-frames", "120"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    cwd=BASE_DIR)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return
        box = tk.Toplevel(self.root)
        box.title("Edge Pipeline Output (press 'q' to stop)")
        box.geometry("720x440")
        txt = tk.Text(box, wrap="word", font=("Courier", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)

        def read():
            for line in iter(proc.stdout.readline, ""):
                txt.insert("end", line)
                txt.see("end")
            proc.wait()
            txt.insert("end", "\n[done]\n")
        import threading
        threading.Thread(target=read, daemon=True).start()

    def compare(self):
        if not HAVE_PLT:
            messagebox.showerror("Error", "Install 'matplotlib'.")
            return
        if not os.path.isfile(EDGE_CSV) or not os.path.isfile(CLOUD_CSV):
            messagebox.showinfo(
                "Run pipelines first",
                "Run module 5 (Edge Report) and the cloud pipeline "
                "(cloud_client.py) first so the CSV logs exist.")
            return
        try:
            self.plot_compare()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def plot_compare(self):
        import csv as _csv
        edge_lat, e2e, edge_bytes = [], [], 0
        with open(EDGE_CSV) as fh:
            for row in _csv.DictReader(fh):
                edge_lat.append(float(row["edge_latency_ms"]))
                e2e.append(float(row["cloud_e2e_ms"]))
                edge_bytes += int(row["payload_bytes"])
        cloud_rt, cloud_bytes = [], 0
        with open(CLOUD_CSV) as fh:
            for row in _csv.DictReader(fh):
                cloud_rt.append(float(row["round_trip_ms"]))
                cloud_bytes += int(row["raw_bytes"])

        n = min(len(edge_lat), len(cloud_rt)) or 1
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        frames = list(range(1, n + 1))
        ax[0].plot(frames, edge_lat[:n], label="Edge inference (local)",
                   color="tab:green")
        ax[0].plot(frames, cloud_rt[:n], label="Cloud round-trip (raw upload)",
                   color="tab:red")
        ax[0].set_xlabel("frame"); ax[0].set_ylabel("latency (ms)")
        ax[0].set_title("Latency per frame"); ax[0].legend(); ax[0].grid()

        sizes = [edge_bytes / 1e6, cloud_bytes / 1e6]
        reduction = (1 - edge_bytes / cloud_bytes) * 100 if cloud_bytes else 0
        ax[1].bar(["Edge (JSON report)", "Cloud (raw images)"], sizes,
                  color=["tab:green", "tab:red"])
        ax[1].set_ylabel("total data sent (MB)")
        ax[1].set_title(f"Bandwidth - edge uses {reduction:.1f}% less")
        ax[1].grid(axis="y")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    App(tk.Tk()).root.mainloop()
