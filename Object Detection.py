import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Prevent OpenMP conflicts

import cv2
import numpy as np
import threading
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

cv2.setNumThreads(0)  # Disable OpenMP conflicts in OpenCV


class VideoProcessor:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.model = YOLO("yolov8n.pt")  # YOLOv8 Nano for high performance
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()
        self.positions = []  # Store detected object positions for heatmap

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect_objects(self, frame):
        results = self.model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{self.model.names[cls]} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Store detected position (center of bounding box)
                self.positions.append(((x1 + x2) // 2, (y1 + y2) // 2))
        return frame

    def get_frame(self, detect_objects=False):
        if self.frame is not None:
            with self.lock:
                frame = self.frame.copy()
            if detect_objects:
                frame = self.detect_objects(frame)
            return frame
        return None

    def stop(self):
        self.running = False
        self.cap.release()


# ========== Streamlit UI ==========
st.title("Live Object Detection & Kalman Filter Tracking")

processor = VideoProcessor()
detect_objects = st.checkbox("Enable Object Detection")
frame_holder = st.empty()

# ========== Live Video Feed ==========
try:
    while processor.running:
        frame = processor.get_frame(detect_objects)
        if frame is not None:
            frame_holder.image(frame, channels="RGB")
except Exception as e:
    st.error(f"An error occurred: {e}")

processor.stop()


# ========== Kalman Filter for Simulated Laser Tracking ==========
st.title("Simulated Laser Tracking with Kalman Filter")

# Simulate moving object
np.random.seed(42)
time_steps = 100
true_distance = np.linspace(1, 10, time_steps)
noise = np.random.normal(0, 0.5, time_steps)
measured_distance = true_distance + noise

# Kalman Filter Setup
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[1.0], [0.0]])
kf.F = np.array([[1, 1], [0, 1]])
kf.H = np.array([[1, 0]])
kf.P *= 1000
kf.R = 0.5
kf.Q = np.array([[0.01, 0], [0, 0.01]])

# Apply Kalman Filter
filtered_distance = []
for z in measured_distance:
    kf.predict()
    kf.update(z)
    filtered_distance.append(kf.x[0, 0])

# Plot Kalman Filter Tracking
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(true_distance, label="True Distance", linestyle="dashed")
ax1.plot(measured_distance, label="Noisy Sensor Data", alpha=0.5)
ax1.plot(filtered_distance, label="Kalman Filter Output", linewidth=2)
ax1.set_xlabel("Time Steps")
ax1.set_ylabel("Distance (m)")
ax1.legend()
ax1.set_title("Simulated Laser Tracking with Kalman Filter")
st.pyplot(fig1)


# ========== Heatmap of Object Movement ==========
st.title("Heatmap of Object Movement")

if processor.positions:
    heatmap_size = (480, 640)  # Adjust based on video resolution
    heatmap_data = np.zeros(heatmap_size)

    for x, y in processor.positions:
        if 0 <= y < heatmap_size[0] and 0 <= x < heatmap_size[1]:
            heatmap_data[y, x] += 1  # Increment position count

    # Plot Heatmap
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, cmap="hot", cbar=True, xticklabels=False, yticklabels=False)
    ax2.set_title("Object Movement Heatmap")
    st.pyplot(fig2)
