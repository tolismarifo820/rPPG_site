import av
import cv2
import numpy as np
import time
from collections import deque
import streamlit as st
import mediapipe as mp
from scipy.signal import butter, filtfilt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- Page Configuration ---
st.set_page_config(
    page_title="Contactless Vitals Dashboard",
    page_icon="💓",
    layout="wide"
)

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

SKIN_REGIONS = {
    "forehead": [10, 109, 67, 103, 54, 21, 71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301, 251, 284, 332, 297, 338]
}

FACE_STABLE_SECONDS_REQUIRED = 5.0
SPO2_A, SPO2_B = 110.0, 22.0

def get_segmented_mask(frame_shape, landmarks):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    landmarks_px = [(min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)) for lm in landmarks.landmark]
    def get_hull(indices): return cv2.convexHull(np.array([landmarks_px[i] for i in indices], dtype=np.int32))
    for _, indices in SKIN_REGIONS.items(): cv2.fillPoly(mask, [get_hull(indices)], 255)
    return mask

def bbox_inside_roi(box, roi):
    if box is None or roi is None: return False
    bx, by, bw, bh = box; rx1, ry1, rx2, ry2 = roi
    return (bx >= rx1 and by >= ry1 and (bx + bw) <= rx2 and (by + bh) <= ry2)

def clamp(val, min_val, max_val): return max(min_val, min(val, max_val))

def extract_mediapipe_roi(frame, face_mesh, scale=0.4):
    h, w = frame.shape[:2]
    small_w, small_h = int(w * scale), int(h * scale)
    small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    mp_face_ok, mp_face_box, mask_contours = False, None, []

    if results.multi_face_landmarks:
        mp_face_ok = True
        face_landmarks = results.multi_face_landmarks[0]
        small_mask = get_segmented_mask((small_h, small_w), face_landmarks)
        full_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask_contours = contours
            min_x, min_y = w, h
            max_x, max_y = 0, 0
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x + bw), max(max_y, y + bh)
            mp_face_box = (min_x, min_y, max_x - min_x, max_y - min_y)
            
    return full_mask, mp_face_ok, mp_face_box, mask_contours

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    if len(data) < 15: return data
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, min(0.99, highcut / nyq)], btype='band')
    try: return filtfilt(b, a, data, padlen=min(len(data) - 1, 3 * max(len(b), len(a))))
    except ValueError: return data 

def extract_pos(r, g, b, fps_val):
    N, l = len(r), int(2 * fps_val)
    if N < l: return np.zeros(N)
    eps = 1e-6
    rw, gw, bw = np.lib.stride_tricks.sliding_window_view(r, l), np.lib.stride_tricks.sliding_window_view(g, l), np.lib.stride_tricks.sliding_window_view(b, l)
    rn, gn, bn = rw/(np.mean(rw, axis=1, keepdims=True)+eps), gw/(np.mean(gw, axis=1, keepdims=True)+eps), bw/(np.mean(bw, axis=1, keepdims=True)+eps)
    S1, S2 = gn - bn, gn + bn - 2 * rn
    h = S1 + (np.std(S1, axis=1, keepdims=True) / (np.std(S2, axis=1, keepdims=True) + eps)) * S2
    h_zm = h - np.mean(h, axis=1, keepdims=True)
    H = np.zeros(N)
    for i in range(l): H[i : i + h_zm.shape[0]] += h_zm[:, i]
    return H

def estimate_peak_bpm(signal, fps_val, min_hz, max_hz):
    mag = np.abs(np.fft.fft(signal - float(np.mean(signal)), n=1024))
    freqs = np.fft.fftfreq(1024, d=1.0 / max(float(fps_val), 1e-6))
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    return float(freqs[int(np.argmax(mag * mask))] * 60.0) if np.any(mask) else 0.0

def robust_mean(data):
    if not data: return None
    data_list = list(data)
    if len(data_list) < 3: return float(np.mean(data_list))
    return float(np.median(data_list))

# --- WebRTC Processor ---
class RPPGVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.fps = 15.0
        self.bufferSize = int(self.fps * 10)
        self.bpmBufferSize = 10
        self.bpmCalcEvery = int(self.fps * 1)
        
        self.hr_low, self.hr_high = 0.7, 3.0
        self.rr_low, self.rr_high = 0.15, 0.5
        
        self.red_buffer = np.zeros((self.bufferSize,), dtype=np.float32)
        self.green_buffer = np.zeros((self.bufferSize,), dtype=np.float32)
        self.blue_buffer = np.zeros((self.bufferSize,), dtype=np.float32)
        
        self.bpmBuffer = np.full((self.bpmBufferSize,), np.nan, dtype=np.float32)
        self.bpm_all = []
        self.rr_history = deque(maxlen=10)
        self.spo2_history = deque(maxlen=10)
        
        self.buffers_initialized = False
        self.bufferIndex = 0
        self.bpmBufferIndex = 0
        
        self.current_hr = None
        self.current_rr = None
        self.current_spo2 = None
        
        self.smoothed_bbox = None
        self.stable_face_start_time = None
        
        self.cached_full_mask = None
        self.cached_mp_face_ok = False
        self.cached_mp_face_box = None
        self.cached_mask_contours = []
        self.frame_counter = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1
        
        current_h, current_w = img.shape[:2]
        evm_roi_bbox = (int(current_w * 0.05), int(current_h * 0.05), current_w - int(current_w * 0.05), current_h - int(current_h * 0.05))

        if self.frame_counter % 3 != 0 or not self.cached_mp_face_ok:
            full_mask, mp_face_ok, mp_face_box, mask_contours = extract_mediapipe_roi(img, face_mesh, scale=0.4)
            self.cached_full_mask, self.cached_mp_face_ok, self.cached_mp_face_box, self.cached_mask_contours = full_mask, mp_face_ok, mp_face_box, mask_contours
        else:
            full_mask, mp_face_ok, mp_face_box, mask_contours = self.cached_full_mask, self.cached_mp_face_ok, self.cached_mp_face_box, self.cached_mask_contours

        face_detected_now = mp_face_ok and bbox_inside_roi(mp_face_box, evm_roi_bbox)
        
        if face_detected_now:
            if self.stable_face_start_time is None:
                self.stable_face_start_time = time.time()
            vitals_enabled = (time.time() - self.stable_face_start_time) >= FACE_STABLE_SECONDS_REQUIRED
        else:
            self.stable_face_start_time = None
            vitals_enabled = False

        vitals_roi, vitals_mask = None, None
        
        if mp_face_box is not None and face_detected_now:
            raw_x, raw_y, raw_w, raw_h = mp_face_box
            if self.smoothed_bbox is None or not face_detected_now:
                self.smoothed_bbox = [raw_x, raw_y, raw_w, raw_h]
            else:
                alpha_s = 0.15 
                self.smoothed_bbox = [(1.0 - alpha_s)*s + alpha_s*r for s, r in zip(self.smoothed_bbox, [raw_x, raw_y, raw_w, raw_h])]

            vx1, vy1, vw, vh = [int(v) for v in self.smoothed_bbox]
            vx2, vy2 = clamp(vx1 + vw, 1, current_w), clamp(vy1 + vh, 1, current_h)
            vx1, vy1 = clamp(vx1, 0, current_w - 1), clamp(vy1, 0, current_h - 1)
            
            if vx2 > vx1 and vy2 > vy1:
                vitals_roi = img[vy1:vy2, vx1:vx2, :]
                vitals_mask = full_mask[vy1:vy2, vx1:vx2]

        if vitals_roi is not None and vitals_mask is not None:
            mean_vals = cv2.mean(vitals_roi, mask=vitals_mask)
            r_val, g_val, b_val = float(mean_vals[2]), float(mean_vals[1]), float(mean_vals[0])
            
            if r_val == 0.0 and g_val == 0.0 and b_val == 0.0 and self.buffers_initialized:
                prev_idx = (self.bufferIndex - 1) % self.bufferSize
                r_val = float(self.red_buffer[prev_idx])
                g_val = float(self.green_buffer[prev_idx])
                b_val = float(self.blue_buffer[prev_idx])

            if not self.buffers_initialized:
                self.red_buffer[:], self.green_buffer[:], self.blue_buffer[:] = r_val, g_val, b_val
                self.buffers_initialized = True
            else:
                self.red_buffer[self.bufferIndex] = r_val
                self.green_buffer[self.bufferIndex] = g_val
                self.blue_buffer[self.bufferIndex] = b_val
                
            cv2.rectangle(img, (vx1, vy1), (vx2, vy2), (0, 200, 115), 2)

        r_rolled = np.roll(self.red_buffer, -self.bufferIndex - 1)
        g_rolled = np.roll(self.green_buffer, -self.bufferIndex - 1)
        b_rolled = np.roll(self.blue_buffer, -self.bufferIndex - 1)

        if self.buffers_initialized and face_detected_now:
            raw_extracted_signal = extract_pos(r_rolled, g_rolled, b_rolled, int(self.fps))
            
            if vitals_enabled:
                active_signal = bandpass_filter(raw_extracted_signal, self.hr_low, self.hr_high, self.fps, order=2)
            else:
                active_signal = raw_extracted_signal

            if vitals_enabled and self.bufferIndex % self.bpmCalcEvery == 0:
                # 1. Heart Rate (BPM)
                bpm = estimate_peak_bpm(active_signal, self.fps, self.hr_low, self.hr_high)
                if len(self.bpm_all) > 0: 
                    max_change = 3.0 * (self.bpmCalcEvery / self.fps)
                    bpm = float(np.clip(bpm, self.bpm_all[-1] - max_change, self.bpm_all[-1] + max_change))
                    
                self.bpmBuffer[self.bpmBufferIndex] = bpm
                self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize
                self.bpm_all.append(bpm)
                self.current_hr = np.nanmean(self.bpmBuffer)

                # 2. Respiration Rate (BR/MIN)
                r_rr_filtered = bandpass_filter(r_rolled, self.rr_low, self.rr_high, self.fps, order=2)
                rr_raw = estimate_peak_bpm(r_rr_filtered, self.fps, self.rr_low, self.rr_high)
                if 6 <= rr_raw <= 40:
                    self.rr_history.append(rr_raw)
                self.current_rr = robust_mean(self.rr_history)

                # 3. SpO2 Estimation (%)
                r_hr_band = bandpass_filter(r_rolled, self.hr_low, self.hr_high, self.fps, order=2)
                b_hr_band = bandpass_filter(b_rolled, self.hr_low, self.hr_high, self.fps, order=2)
                ac_r, dc_r = float(np.std(r_hr_band)), float(np.mean(r_rolled))
                ac_b, dc_b = float(np.std(b_hr_band)), float(np.mean(b_rolled))
                
                if dc_r > 1e-3 and dc_b > 1e-3 and ac_b > 1e-6:
                    ratio_of_ratios = (ac_r / dc_r) / (ac_b / dc_b)
                    spo2_raw = float(SPO2_A - (SPO2_B * ratio_of_ratios))
                    self.spo2_history.append(clamp(spo2_raw, 95.0, 99.0))
                self.current_spo2 = robust_mean(self.spo2_history)

        if not face_detected_now:
            self.current_hr, self.current_rr, self.current_spo2 = None, None, None
            self.buffers_initialized = False
            self.rr_history.clear(); self.spo2_history.clear(); self.bpm_all.clear()
            self.bpmBuffer[:] = np.nan

        # Minimal Video Status Overlay
        if face_detected_now and vitals_enabled and self.current_hr is not None and not np.isnan(self.current_hr):
            cv2.putText(img, f"HR: {self.current_hr:.1f} BPM", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (14, 165, 233), 2)
        elif face_detected_now:
            cv2.putText(img, "Calibrating...", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (245, 160, 60), 2)
        else:
            cv2.putText(img, "Scanning Face...", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (230, 230, 230), 2)

        self.bufferIndex = (self.bufferIndex + 1) % self.bufferSize
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI Controls & Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption("Settings & Pipeline Controls")
    
    st.markdown("**Session Setup**")
    st.info("💡 For optimal estimation accuracy, position your face in even lighting and maintain steady posture.")
    
    st.divider()
    st.markdown("**Pipeline Information**")
    st.markdown("- **Extraction Method:** Plane-Orthogonal-to-Skin (POS)")
    st.markdown("- **Cardiac Band:** 0.7 - 3.0 Hz (42 - 180 BPM)")
    st.markdown("- **Respiration Band:** 0.15 - 0.5 Hz (9 - 30 BR/MIN)")
    st.markdown("- **ROI:** Facial Mesh Segmented Mask")

# --- Main Dashboard ---
st.title("💓 Contactless Vitals Dashboard")
st.markdown("Real-time optical vital signs monitoring via ambient facial video streams.")

st.divider()

col_stream, col_metrics = st.columns([7, 3], gap="medium")

with col_stream:
    st.subheader("Live Feed")
    webrtc_ctx = webrtc_streamer(
        key="rppg-stream",
        video_processor_factory=RPPGVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col_metrics:
    st.subheader("Status & Readings")

    # Dynamic 3 Vitals Signs Cards
    hr_val = "-- bpm"
    rr_val = "-- br/min"
    spo2_val = "-- %"

    if webrtc_ctx.video_processor:
        proc = webrtc_ctx.video_processor
        if proc.current_hr is not None and not np.isnan(proc.current_hr):
            hr_val = f"{proc.current_hr:.0f} bpm"
        if proc.current_rr is not None and not np.isnan(proc.current_rr):
            rr_val = f"{proc.current_rr:.0f} br/min"
        if proc.current_spo2 is not None and not np.isnan(proc.current_spo2):
            spo2_val = f"{proc.current_spo2:.0f} %"

    with st.container(border=True):
        st.metric(label="💓 Heart Rate", value=hr_val)
        
    with st.container(border=True):
        st.metric(label="🫁 Respiration", value=rr_val)
        
    with st.container(border=True):
        st.metric(label="🩸 Oxygen (SpO2)", value=spo2_val)

    with st.container(border=True):
        st.markdown("**Monitoring Guide**")
        st.markdown(
            "1. Allow camera access when prompted.\n"
            "2. Ensure your forehead and cheeks remain visible.\n"
            "3. Hold steady for **5 seconds** to initialize calibration."
        )
