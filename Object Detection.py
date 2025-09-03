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
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, get_twilio_ice_servers
from scipy.optimize import linear_sum_assignment
import torch

# --- Environment Setup ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cv2.setNumThreads(0)

# --- Thread-safe Global State ---
class DetectionResults:
    def __init__(self):
        self.boxes = []
        self.latency = 0
        self.lock = threading.Lock()

class DetectEnabledState:
    def __init__(self, value=True):
        self.value = value
        self.lock = threading.Lock()

# --- Model Loading (Cached and Device Set) ---
@st.cache_resource
def load_yolo_model(model_name):
    """Loads a YOLO model and sets it to the GPU if available."""
    try:
        if torch.cuda.is_available():
            device = 'cuda'
            st.success("GPU is available and will be used for inference!")
        else:
            device = 'cpu'
            st.warning("No GPU detected. Falling back to CPU, which may be slow.")
        
        yolo_model = YOLO(model_name)
        yolo_model.to(device)
        
        # Ensure the model is ready by doing a dummy run
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        yolo_model(dummy_image, verbose=False)
        
        return yolo_model, yolo_model.names, device
    except Exception as e:
        st.error(f"Could not load YOLO model {model_name}: {e}")
        return None, [], 'cpu'

# --- Object Tracking with Kalman Filter ---
class ObjectTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = 0.5

    def create_kalman_filter(self, bbox_center):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([bbox_center[0], bbox_center[1], 0, 0])
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P = np.eye(4) * 1000
        kf.R = np.eye(2) * 1
        kf.Q = np.eye(4) * 0.1
        return kf

    def track_and_update(self, new_detections):
        # The rest of the tracking logic is unchanged
        if not new_detections and not self.tracks:
            return self.tracks

        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['kf'].predict()
            self.tracks[track_id]['bbox_pred'] = self._get_bbox_from_state(self.tracks[track_id]['kf'].x)
        
        if not new_detections:
            return self.tracks

        if not self.tracks:
            for det in new_detections:
                self.tracks[self.next_id] = {
                    'kf': self.create_kalman_filter(det['pos']),
                    'label': det['label'],
                    'bbox_true': det['coords'],
                    'pos': det['pos'],
                    'id': self.next_id
                }
                self.next_id += 1
            return self.tracks
        
        cost_matrix = np.full((len(self.tracks), len(new_detections)), 1.0)
        track_ids = list(self.tracks.keys())
        for i, track_id in enumerate(track_ids):
            for j, det in enumerate(new_detections):
                cost_matrix[i, j] = 1 - self._calculate_iou(self.tracks[track_id]['bbox_pred'], det['coords'])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for track_idx, det_idx in zip(row_ind, col_ind):
            if cost_matrix[track_idx, det_idx] < (1 - self.iou_threshold):
                track_id = track_ids[track_idx]
                det = new_detections[det_idx]
                self.tracks[track_id]['kf'].update(np.array(det['pos']))
                self.tracks[track_id]['bbox_true'] = det['coords']
                self.tracks[track_id]['label'] = det['label']

        unmatched_detections = set(range(len(new_detections))) - set(col_ind)
        for det_idx in unmatched_detections:
            det = new_detections[det_idx]
            self.tracks[self.next_id] = {
                'kf': self.create_kalman_filter(det['pos']),
                'label': det['label'],
                'bbox_true': det['coords'],
                'pos': det['pos'],
                'id': self.next_id
            }
            self.next_id += 1

        return self.tracks

    def _calculate_iou(self, box1, box2):
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2
        xi1 = max(x1_a, x1_b)
        yi1 = max(y1_a, y1_b)
        xi2 = min(x2_a, x2_b)
        yi2 = min(y2_a, y2_b)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_a - x1_a) * (y2_a - y1_a)
        box2_area = (x2_b - x1_b) * (y2_b - y1_b)
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou

    def _get_bbox_from_state(self, state):
        x, y, _, _ = state
        return [int(x - 25), int(y - 25), int(x + 25), int(y + 25)]

# --- The Worker Thread ---
def detection_worker(detection_results, frame_queue, detect_enabled_state, yolo_model_ref):
    yolo_model = yolo_model_ref[0]
    yolo_names = yolo_model_ref[1]
    
    while True:
        frame_data = frame_queue.get()
        if frame_data is None:
            break

        frame, start_time = frame_data
        
        with detect_enabled_state.lock:
            if not detect_enabled_state.value:
                detection_results.boxes = []
                continue

        # The model is now guaranteed to be on the correct device
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
                    'pos': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'conf': conf
                })
        
        latency = (time.time() - start_time) * 1000
        with detection_results.lock:
            detection_results.boxes = new_boxes
            detection_results.latency = latency

# --- Video Processor Class ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.object_tracker = ObjectTracker()
        self.heatmap_positions = []
        self.original_width = None
        self.original_height = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        
        if self.original_width is None:
            self.original_width = image.shape[1]
            self.original_height = image.shape[0]

        # Use a consistent and smaller input size for the model
        model_size = 640
        resized_image = cv2.resize(image, (model_size, model_size))

        try:
            frame_queue.put_nowait((resized_image.copy(), time.time()))
        except queue.Full:
            pass
        
        with detection_results.lock:
            latest_detections = detection_results.boxes
            latency = detection_results.latency

        tracked_objects = self.object_tracker.track_and_update(latest_detections)
        
        # Scale back the bounding box coordinates for drawing on the original frame
        x_scale = self.original_width / model_size
        y_scale = self.original_height / model_size

        # Draw latency information
        cv2.putText(image, f"Detection Latency: {latency:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        for track_id, track in tracked_objects.items():
            x1, y1, x2, y2 = track['bbox_true']
            label = track['label']
            
            # Scale coordinates back up for drawing
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)
            
            color = tuple(int(c) for c in (sns.color_palette("husl", 10)[track_id % 10] * 255))
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"ID {track['id']}: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Collect heatmap positions (scaled back to original dimensions)
            pos_x = (x1 + x2) // 2
            pos_y = (y1 + y2) // 2
            self.heatmap_positions.append((pos_x, pos_y))

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- Streamlit UI ---
st.title("Professional Object Detection & Tracking")
st.markdown("---")

model_selection = st.radio(
    "Select a YOLO model for detection:",
    ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'),
    help="Larger models are more accurate but run slower."
)

if "yolo_model" not in st.session_state or st.session_state.yolo_model_name != model_selection:
    st.session_state.yolo_model, st.session_state.yolo_names, st.session_state.device = load_yolo_model(model_selection)
    st.session_state.yolo_model_name = model_selection

if not st.session_state.yolo_model:
    st.stop()

detect_enabled_state = DetectEnabledState(True)
detection_results = DetectionResults()
frame_queue = queue.Queue(maxsize=1)

# Pass the model reference to the worker thread
detection_thread = threading.Thread(
    target=detection_worker,
    args=(detection_results, frame_queue, detect_enabled_state, 
          (st.session_state.yolo_model, st.session_state.yolo_names)),
    daemon=True
)
if not detection_thread.is_alive():
    detection_thread.start()

detect_objects = st.checkbox("Enable Object Detection", value=True)
with detect_enabled_state.lock:
    detect_enabled_state.value = detect_objects

try:
    twilio_ice_servers = get_twilio_ice_servers(st.secrets["twilio"]["account_sid"], st.secrets["twilio"]["auth_token"])
except KeyError:
    st.warning("Twilio credentials not found. The app might not work on some networks.")
    twilio_ice_servers = []
except Exception as e:
    st.error(f"Error getting Twilio ICE servers: {e}")
    twilio_ice_servers = []

ctx = webrtc_streamer(
    key="live_detection_pro_v3",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": twilio_ice_servers},
    async_processing=True
)

st.title("Heatmap of Object Movement")
st.markdown("This heatmap shows the accumulated path of tracked objects.")
if ctx.state.playing:
    st.info("The heatmap will be generated after the video stream is stopped.")
else:
    if "heatmap_positions" in st.session_state and st.session_state.heatmap_positions:
        # Get dimensions from a placeholder image, or the first frame's original size
        # A more robust way would be to pass dimensions from the processor
        try:
            heatmap_size = (480, 640)
            heatmap_data = np.zeros(heatmap_size)
            for x, y in st.session_state.heatmap_positions:
                if 0 <= y < heatmap_size[0] and 0 <= x < heatmap_size[1]:
                    heatmap_data[y, x] += 1
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(heatmap_data, cmap="hot", cbar=True, xticklabels=False, yticklabels=False)
            ax2.set_title("Object Movement Heatmap")
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Could not generate heatmap: {e}")
    else:
        st.info("Start the video stream to generate a heatmap.")

# Kalman Filter Simulation
st.title("Simulated Laser Tracking with Kalman Filter")
# ... (your original Kalman filter code remains here, unchanged)
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
