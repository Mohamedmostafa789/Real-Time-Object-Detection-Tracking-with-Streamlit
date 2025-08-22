import os
import cv2
import numpy as np
import threading
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Correct spelling
from filterpy.kalman import KalmanFilter
import av

# Import necessary components from streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, get_twilio_ice_servers

# Set environment variable to prevent OpenMP conflicts with some libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Disable OpenMP conflicts in OpenCV
cv2.setNumThreads(0)

# Create the YOLO model outside of the class to load it only once
try:
    yolo_model = YOLO("yolov8n.pt")
    yolo_names = yolo_model.names
except Exception as e:
    st.error(f"Could not load YOLO model: {e}")
    yolo_model = None
    yolo_names = []

class SharedState:
    def __init__(self):
        self.detect_objects = True
        self.positions = []
        self.lock = threading.Lock()

shared_state = SharedState()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.positions = []
        self.detect_enabled = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        with shared_state.lock:
            self.detect_enabled = shared_state.detect_objects
        if self.detect_enabled and yolo_model:
            results = yolo_model(image, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = f"{yolo_names[cls]} ({conf:.2f})"
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self.positions.append(((x1 + x2) // 2, (y1 + y2) // 2))
        with shared_state.lock:
            shared_state.positions.extend(self.positions)
            self.positions = []
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# ========== Streamlit UI ==========
st.title("Live Object Detection & Kalman Filter Tracking")
st.markdown("---")

detect_objects = st.checkbox("Enable Object Detection", value=True)
with shared_state.lock:
    shared_state.detect_objects = detect_objects

# Get Twilio ICE servers securely from Streamlit secrets
try:
    twilio_ice_servers = get_twilio_ice_servers(
        st.secrets["twilio"]["account_sid"],
        st.secrets["twilio"]["auth_token"]
    )
except KeyError:
    st.warning("Twilio credentials not found. The app might not work on some networks. "
               "Add them to your Streamlit secrets for a more reliable connection.")
    twilio_ice_servers = []
except Exception as e:
    st.error(f"Error getting Twilio ICE servers: {e}")
    twilio_ice_servers = []

ctx = webrtc_streamer(
    key="live_detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": twilio_ice_servers}
)

# ========== Kalman Filter for Simulated Laser Tracking ==========
st.title("Simulated Laser Tracking with Kalman Filter")
np.random.seed(42)
time_steps = 100
true_distance = np.linspace(1, 10, time_steps)
noise = np.random.normal(0, 0.5, time_steps)
measured_distance = true_distance + noise
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[1.0], [0.0]])
kf.F = np.array([[1, 1], [0, 1]])
kf.H = np.array([[1, 0]])
kf.P *= 1000
kf.R = 0.5
kf.Q = np.array([[0.01, 0], [0, 0.01]])
filtered_distance = []
for z in measured_distance:
    kf.predict()
    kf.update(z)
    filtered_distance.append(kf.x[0, 0])
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
if ctx.state.playing:
    with shared_state.lock:
        positions = shared_state.positions.copy()
    if positions:
        heatmap_size = (480, 640)
        heatmap_data = np.zeros(heatmap_size)
        for x, y in positions:
            if 0 <= y < heatmap_size[0] and 0 <= x < heatmap_size[1]:
                heatmap_data[y, x] += 1
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_data, cmap="hot", cbar=True, xticklabels=False, yticklabels=False)
        ax2.set_title("Object Movement Heatmap")
        st.pyplot(fig2)
    else:
        st.info("Start the video stream to generate a heatmap.")
else:
    st.info("Start the video stream to generate a heatmap.")
