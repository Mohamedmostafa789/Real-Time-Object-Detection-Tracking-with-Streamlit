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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, get_twilio_ice_servers
from scipy.optimize import linear_sum_assignment

# Set environment variable to prevent OpenMP conflicts with some libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cv2.setNumThreads(0)

# --- Global State Management (Thread-safe) ---
class DetectionResults:
    def __init__(self):
        self.boxes = []
        self.lock = threading.Lock()

class DetectEnabledState:
    def __init__(self, value=True):
        self.value = value
        self.lock = threading.Lock()

# --- Model Loading (Once at start) ---
@st.cache_resource
def load_yolo_model(model_name):
    try:
        yolo_model = YOLO(model_name)
        return yolo_model, yolo_model.names
    except Exception as e:
        st.error(f"Could not load YOLO model {model_name}: {e}")
        return None, []

# --- Object Tracking with Kalman Filter ---
class ObjectTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = 0.5

    def create_kalman_filter(self, bbox_center):
        """Initializes a Kalman Filter for a new track."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([bbox_center[0], bbox_center[1], 0, 0]) # State: [x, y, vx, vy]
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]) # State Transition Matrix
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]]) # Measurement Function
        kf.P = np.eye(4) * 1000 # Covariance Matrix
        kf.R = np.eye(2) * 1 # Measurement Noise
        kf.Q = np.eye(4) * 0.1 # Process Noise
        return kf

    def track_and_update(self, new_detections):
        """Updates tracks with new detections using Hungarian algorithm (Munkres)."""
        # Predict all existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['kf'].predict()
            self.tracks[track_id]['bbox_pred'] = self._get_bbox_from_state(self.tracks[track_id]['kf'].x)

        if not new_detections:
            return self.tracks

        if not self.tracks:
            # First frame, create new tracks for all detections
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

        # Build cost matrix based on IOU between predicted tracks and new detections
        cost_matrix = np.full((len(self.tracks), len(new_detections)), 1.0)
        track_ids = list(self.tracks.keys())
        for i, track_id in enumerate(track_ids):
            for j, det in enumerate(new_detections):
                cost_matrix[i, j] = 1 - self._calculate_iou(self.tracks[track_id]['bbox_pred'], det['coords'])

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update matched tracks
        for track_idx, det_idx in zip(row_ind, col_ind):
            if cost_matrix[track_idx, det_idx] < (1 - self.iou_threshold):
                track_id = track_ids[track_idx]
                det = new_detections[det_idx]
                self.tracks[track_id]['kf'].update(np.array(det['pos']))
                self.tracks[track_id]['bbox_true'] = det['coords']
                self.tracks[track_id]['label'] = det['label'] # Update label in case of class change

        # Create new tracks for unmatched detections
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
        return [int(x - 25), int(y - 25), int(x + 25), int(y + 25)] # Placeholder for predicted box

# --- The Worker Thread ---
def detection_worker(detection_results, frame_queue, detect_enabled_state, yolo_model_ref):
    yolo_model = yolo_model_ref[0]
    yolo_names = yolo_model_ref[1]
    
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        with detect_enabled_state.lock:
            if not detect_enabled_state.value:
                detection_results.boxes = []
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
                        'pos': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'conf': conf
                    })
            with detection_results.lock:
                detection_results.boxes = new_boxes

# --- Video Processor Class ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.object_tracker = ObjectTracker()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        try:
            frame_queue.put_nowait(image.copy())
        except queue.Full:
            pass

        with detection_results.lock:
            latest_detections = detection_results.boxes

        # --- Professional Tracking step ---
        # The tracker now links detections across frames
        tracked_objects = self.object_tracker.track_and_update(latest_detections)

        # --- Drawing Phase ---
        for track_id, track in tracked_objects.items():
            # Use the tracked object's properties for drawing
            x1, y1, x2, y2 = track['bbox_true']
            label = track['label']
            
            # Use a unique color for each tracked ID
            color = tuple(int(c) for c in (sns.color_palette("husl", 10)[track_id % 10] * 255))
            
            # Draw the box and ID
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"ID {track['id']}: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Store the Kalman-filtered position for the heatmap
            heatmap_positions.append(track['pos'])

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- Streamlit UI ---
st.title("Professional Object Detection & Tracking")
st.markdown("---")

# User model selection
model_selection = st.radio(
    "Select a YOLO model for detection:",
    ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'),
    help="Larger models are more accurate but run slower."
)

if 'yolo_model' not in st.session_state or st.session_state.yolo_model_name != model_selection:
    st.session_state.yolo_model, st.session_state.yolo_names = load_yolo_model(model_selection)
    st.session_state.yolo_model_name = model_selection

if not st.session_state.yolo_model:
    st.stop()

# Thread-safe objects
detect_enabled_state = DetectEnabledState(True)
detection_results = DetectionResults()
frame_queue = queue.Queue(maxsize=1)

# Start the worker thread
detection_thread = threading.Thread(
    target=detection_worker,
    args=(detection_results, frame_queue, detect_enabled_state, (st.session_state.yolo_model, st.session_state.yolo_names)),
    daemon=True
)
if not detection_thread.is_alive():
    detection_thread.start()

detect_objects = st.checkbox("Enable Object Detection", value=True)
with detect_enabled_state.lock:
    detect_enabled_state.value = detect_objects

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
    key="live_detection_pro",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": twilio_ice_servers},
    async_processing=True # Use async processing for better performance
)

# A place to store all the positions for the heatmap
heatmap_positions = []

# Heatmap Section (Refined to only collect data when stream is active)
st.title("Heatmap of Object Movement")
st.markdown("This heatmap shows the accumulated path of tracked objects.")

if ctx.state.playing:
    # A cleaner way to collect data without a blocking loop
    st.info("The heatmap will be generated after the video stream is stopped.")
    # The data collection now happens directly in the VideoProcessor.recv method.
else:
    if heatmap_positions:
        # Generate the heatmap only when the stream stops and data has been collected
        heatmap_size = (480, 640)
        heatmap_data = np.zeros(heatmap_size)
        for x, y in heatmap_positions:
            if 0 <= y < heatmap_size[0] and 0 <= x < heatmap_size[1]:
                heatmap_data[y, x] += 1
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_data, cmap="hot", cbar=True, xticklabels=False, yticklabels=False)
        ax2.set_title("Object Movement Heatmap")
        st.pyplot(fig2)
    else:
        st.info("Start the video stream to generate a heatmap.")

# Existing Kalman Filter Simulation (kept for demonstration)
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
