import os
import cv2
import numpy as np
import threading
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import av
import queue
import torch
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, get_twilio_ice_servers

# Set environment variable to prevent OpenMP conflicts with some libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Disable OpenMP conflicts in OpenCV
cv2.setNumThreads(0)

# --- Thread-safe Global State ---
class DetectionResults:
    def __init__(self):
        self.boxes = []
        self.lock = threading.Lock()

class DetectEnabledState:
    def __init__(self, value=True):
        self.value = value
        self.lock = threading.Lock()

# --- Model Loading (Cached) ---
@st.cache_resource
def load_yolo_model(model_name):
    """Loads a YOLO model and sets it to the GPU if available."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.success(f"YOLO model will use {device.upper()}.")
        yolo_model = YOLO(model_name)
        yolo_model.to(device)
        return yolo_model, yolo_model.names
    except Exception as e:
        st.error(f"Could not load YOLO model {model_name}: {e}")
        return None, []

# --- The Worker Thread ---
def detection_worker(detection_results, frame_queue, detect_enabled_state, yolo_model_ref):
    yolo_model, yolo_names = yolo_model_ref
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        with detect_enabled_state.lock:
            if not detect_enabled_state.value:
                continue

        if yolo_model:
            results = yolo_model(frame, verbose=False)
            
            new_boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = f"{yolo_names[cls]} ({conf:.2f})"
                    new_boxes.append({
                        'coords': (x1, y1, x2, y2),
                        'label': label,
                        'pos': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
            
            with detection_results.lock:
                detection_results.boxes = new_boxes

# --- The video processor class that runs in the main thread ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # A place to store all the positions for the heatmap, tied to the processor instance
        self.heatmap_positions = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        try:
            frame_queue.put_nowait(image.copy())
        except queue.Full:
            pass

        with detection_results.lock:
            latest_boxes = detection_results.boxes
            # Collect data for the heatmap from the current frame's detections
            for box in latest_boxes:
                self.heatmap_positions.append(box['pos'])

        # Draw the latest results from the worker thread
        with detection_results.lock:
            for box in latest_boxes:
                x1, y1, x2, y2 = box['coords']
                label = box['label']
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- Streamlit UI and Logic ---
st.title("Live Object Detection & Kalman Filter Tracking")
st.markdown("---")

detect_objects = st.checkbox("Enable Object Detection", value=True)
with detect_enabled_state.lock:
    detect_enabled_state.value = detect_objects

# Load the model only once
yolo_model, yolo_names = load_yolo_model("yolov8n.pt")
if not yolo_model:
    st.stop()

detect_enabled_state = DetectEnabledState()
detection_results = DetectionResults()
frame_queue = queue.Queue(maxsize=1)

detection_thread = threading.Thread(
    target=detection_worker, 
    args=(detection_results, frame_queue, detect_enabled_state, (yolo_model, yolo_names)), 
    daemon=True
)
if not detection_thread.is_alive():
    detection_thread.start()

# Get Twilio ICE servers from Streamlit secrets
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
    rtc_configuration={"iceServers": twilio_ice_servers},
    async_processing=True # Use async processing for better performance
)

# Store heatmap positions in session state for persistence
if "heatmap_positions" not in st.session_state:
    st.session_state.heatmap_positions = []

if ctx.state.playing and ctx.video_processor:
    # A cleaner way to collect data without a blocking loop
    # We now collect the data directly in the VideoProcessor.recv method
    st.info("The heatmap will be generated after the video stream is stopped.")
    # Append the positions from the video processor to the session state
    st.session_state.heatmap_positions.extend(ctx.video_processor.heatmap_positions)
    ctx.video_processor.heatmap_positions = [] # Clear the processor's list to avoid re-adding

# ========== Kalman Filter for Simulated Laser Tracking ==========
st.title("Simulated Laser Tracking with Kalman Filter")
# (The rest of the Kalman Filter code is the same)
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
if st.session_state.heatmap_positions:
    heatmap_size = (480, 640)
    heatmap_data = np.zeros(heatmap_size)
    for x, y in st.session_state.heatmap_positions:
        if 0 <= y < heatmap_size[0] and 0 <= x < heatmap_size[1]:
            heatmap_data[y, x] += 1
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, cmap="hot", cbar=True, xticklabels=False, yticklabels=False)
    ax2.set_title("Object Movement Heatmap")
    st.pyplot(fig2)
else:
    st.info("Start the video stream to generate a heatmap.")
