import cv2
import datetime
import os
import threading
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox, scrolledtext

# ====== Constants ======
MOTION_THRESHOLD = 15
MIN_CONTOUR_AREA = 500
MIN_ASPECT_RATIO = 0.1
OUTPUT_DIR = "motion_segments"
YOLO_PATH = r"C:\Users\Acer\OneDrive\Desktop\YOLO Model"

class MotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎥 Smart Gate Motion Detector")
        self.video_path = ""
        self.running = False

        self.net = cv2.dnn.readNet(f"{YOLO_PATH}\\yolov3.weights", f"{YOLO_PATH}\\yolov3.cfg")
        with open(f"{YOLO_PATH}\\coco.names", "r") as f:
            self.classes = f.read().splitlines()
        self.layer_names = self.net.getUnconnectedOutLayersNames()

        Label(root, text="Smart Gate Motion Detector", font=("Helvetica", 16, "bold")).pack(pady=10)
        Button(root, text="📁 Browse Video", command=self.browse_video, width=20).pack(pady=5)
        self.video_label = Label(root, text="No video selected", fg="gray")
        self.video_label.pack()
        Button(root, text="▶ Start Detection", command=self.start_detection, width=20, bg="green", fg="white").pack(pady=10)
        Button(root, text="📷 Live Camera Feed", command=self.start_live_camera, width=20, bg="blue", fg="white").pack(pady=5)

        Label(root, text="📋 Log Output:", anchor="w").pack(fill="x")
        self.log_output = scrolledtext.ScrolledText(root, height=10, state="disabled")
        self.log_output.pack(padx=10, pady=5, fill="both", expand=True)

        self.object_positions = {}

    def browse_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.video_label.config(text=f"Selected: {self.video_path}", fg="black")
            self.log(f"✅ Video selected: {self.video_path}")
        else:
            self.video_label.config(text="No video selected", fg="gray")

    def log(self, message):
        self.log_output.configure(state="normal")
        self.log_output.insert(END, f"{message}\n")
        self.log_output.see(END)
        self.log_output.configure(state="disabled")

    def start_detection(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return
        self.running = True
        threading.Thread(target=self.run_detection, daemon=True).start()

    def start_live_camera(self):
        self.video_path = 0
        self.running = True
        threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        frame_count = 0
        YOLO_EVERY_N_FRAMES = 5
        cached_labels = []

        self.log("🟢 Starting detection...")
        cap = cv2.VideoCapture(self.video_path)
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        if not ret:
            self.log("❌ Failed to read from source.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        frame_size = (int(cap.get(3)), int(cap.get(4)))

        self.log("🖼️ Select the gate area in the popup window.")
        gate_roi = cv2.selectROI("Select Gate", frame1, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Gate")
        gate_x, gate_y, gate_w, gate_h = gate_roi
        gate_rect = (gate_x, gate_y, gate_x + gate_w, gate_y + gate_h)
        gate_line_y = gate_y + gate_h // 2

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        motion_detected = False
        motion_frames = []
        segment_counter = 1
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None

        def track_direction(object_id, y1, y2):
            if object_id not in self.object_positions:
                self.object_positions[object_id] = []
            self.object_positions[object_id].append(y2)
            if len(self.object_positions[object_id]) > 2:
                self.object_positions[object_id].pop(0)
            if len(self.object_positions[object_id]) == 2:
                prev_y, curr_y = self.object_positions[object_id]
                if prev_y < gate_line_y <= curr_y:
                    self.log(f"🚶 Entry detected (ID: {object_id})")
                elif prev_y > gate_line_y >= curr_y:
                    self.log(f"🚶 Exit detected (ID: {object_id})")

        def is_inside_gate(x, y, w, h):
            motion_rect = (x, y, x + w, y + h)
            return not (motion_rect[2] < gate_rect[0] or motion_rect[0] > gate_rect[2] or motion_rect[3] < gate_rect[1] or motion_rect[1] > gate_rect[3])

        while ret and self.running:
            start_time = cv2.getTickCount()
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            _, thresh = cv2.threshold(blur, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            detected = False
            for contour in contours:
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w / float(h) < MIN_ASPECT_RATIO:
                    continue

                if is_inside_gate(x, y, w, h):
                   detected = True

                if frame_count % YOLO_EVERY_N_FRAMES == 0:
                    cached_labels = []  # Reset cache
                    roi = frame1[y:y + h, x:x + w]
                    blob = cv2.dnn.blobFromImage(roi, 1 / 255.0, (320, 320), swapRB=True, crop=False)
                    self.net.setInput(blob)
                    outputs = self.net.forward(self.layer_names)
            
                    for output in outputs:
                        for detection in output:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                label = self.classes[class_id]
                                cached_labels.append(label)

                for label in cached_labels:
                    cv2.putText(frame1, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    frame_count += 1



                    object_id = str(x + y)
                    track_direction(object_id, y, y + h)

                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame1, "Motion @Gate", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Gate visualization
            cv2.rectangle(frame1, (gate_x, gate_y), (gate_x + gate_w, gate_y + gate_h), (255, 0, 0), 2)
            cv2.line(frame1, (gate_x, gate_line_y), (gate_x + gate_w, gate_line_y), (255, 0, 0), 2)
            cv2.putText(frame1, "Gate Area", (gate_x, gate_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if detected:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame1, f"Motion Detected - {timestamp}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if not motion_detected:
                    motion_detected = True
                    out = cv2.VideoWriter(f"{OUTPUT_DIR}/segment_{segment_counter}.avi", fourcc, fps, frame_size)
                    self.log(f"🎥 Recording segment {segment_counter}...")
                    segment_counter += 1
                motion_frames.append(frame1.copy())
                out.write(frame1)
            else:
                if motion_detected and len(motion_frames) >= 1:
                    self.log(f"💾 Saved segment with {len(motion_frames)} frames.")
                motion_detected = False
                motion_frames.clear()
                if out:
                    out.release()
                    out = None

            cv2.imshow("Smart Camera Feed", frame1)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop early
                self.running = False
                break

            frame1 = frame2
            ret, frame2 = cap.read()

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        self.log("✅ Detection complete.")

# ====== Run App ======
if __name__ == "__main__":
    root = Tk()
    root.geometry("600x500")
    app = MotionDetectorApp(root)
    root.mainloop()
