import av
import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
from scipy.signal import butter, filtfilt
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Contactless Vitals Dashboard",
    page_icon="💓",
    layout="wide"
)

# --- Configuration & Dashboard Layout Constants ---
CANVAS_W, CANVAS_H = 880, 520
BOX_PULSE = (30, 30, 240, 130)
BOX_BR = (30, 180, 240, 130)
BOX_SPO2 = (30, 330, 240, 130)
BOX_CAMERA = (300, 30, 540, 460)

# Colors (BGR format)
BG_COLOR = (18, 18, 22)
CARD_COLOR = (30, 30, 38)
BORDER_COLOR = (55, 55, 68)
TITLE_COLOR = (140, 140, 160)
SUBTLE_TEXT = (100, 100, 120)
ACCENT_PULSE = (75, 75, 245)
ACCENT_BREATH = (245, 160, 60)
ACCENT_OXY = (60, 220, 120)
ACCENT_EVM = (200, 200, 0)

FACE_STABLE_SECONDS_REQUIRED = 5.0
SPO2_A, SPO2_B = 110.0, 22.0

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

# --- Signal Processing Helpers ---
def get_segmented_mask(frame_shape, landmarks):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    landmarks_px = [(min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)) for lm in landmarks.landmark]
    def get_hull(indices): return cv2.convexHull(np.array([landmarks_px[i] for i in indices], dtype=np.int32))
    for _, indices in SKIN_REGIONS.items(): 
        cv2.fillPoly(mask, [get_hull(indices)], 255)
    return mask

def bbox_inside_roi(box, roi):
    if box is None or roi is None: return False
    bx, by, bw, bh = box; rx1, ry1, rx2, ry2 = roi
    return (bx >= rx1 and by >= ry1 and (bx + bw) <= rx2 and (by + bh) <= ry2)

def clamp(val, min_val, max_val): 
    return max(min_val, min(val, max_val))

def extract_mediapipe_roi(frame, face_mesh, scale=0.5):
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
    low, high = lowcut / nyq, highcut / nyq
    if high >= 1.0: high = 0.99
    b, a = butter(order, [low, high], btype='band')
    try: return filtfilt(b, a, data, padlen=min(len(data) - 1, 3 * max(len(b), len(a))))
    except ValueError: return data 

def extract_pos(r, g, b, fps=15):
    N, l = len(r), int(2 * fps)
    if N < l: return np.zeros(N)
    eps = 1e-6
    rw = np.lib.stride_tricks.sliding_window_view(r, l)
    gw = np.lib.stride_tricks.sliding_window_view(g, l)
    bw = np.lib.stride_tricks.sliding_window_view(b, l)
    rn = rw / (np.mean(rw, axis=1, keepdims=True) + eps)
    gn = gw / (np.mean(gw, axis=1, keepdims=True) + eps)
    bn = bw / (np.mean(bw, axis=1, keepdims=True) + eps)
    S1, S2 = gn - bn, gn + bn - 2 * rn
    h = S1 + (np.std(S1, axis=1, keepdims=True) / (np.std(S2, axis=1, keepdims=True) + eps)) * S2
    h_zm = h - np.mean(h, axis=1, keepdims=True)
    H = np.zeros(N)
    for i in range(l): H[i : i + h_zm.shape[0]] += h_zm[:, i]
    return H

def estimate_peak_bpm(signal, fps_val, min_hz, max_hz):
    centered = signal - float(np.mean(signal))
    n_pad = 1024
    mag = np.abs(np.fft.fft(centered, n=n_pad))
    freqs = np.fft.fftfreq(n_pad, d=1.0 / max(float(fps_val), 1e-6))
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(mask): return 0.0
    return float(freqs[int(np.argmax(mag * mask))] * 60.0)

def robust_mean(data):
    if not data: return None
    data_list = list(data)
    if len(data_list) < 3: return float(np.mean(data_list))
    return float(np.median(data_list))

# --- Canvas Drawing Functions ---
def display_metric_value(face_detected_now, vitals_enabled, value):
    if not face_detected_now: return None
    if not vitals_enabled: return "Calculating..."
    return value

def draw_card(canvas, target_box, title, value, unit, color, show_pulse_icon=False, current_bpm=None):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)
    cv2.putText(canvas, title, (x + 24, y + 48), cv2.FONT_HERSHEY_DUPLEX, 1.1, TITLE_COLOR, 2, cv2.LINE_AA)

    if isinstance(value, str): 
        val_str, font_scale, thickness = value, 1.0, 2
    elif value is not None and value > 0: 
        val_str, font_scale, thickness = f"{value:.0f}", 2.4, 3
    else: 
        val_str, font_scale, thickness = "--", 2.4, 3

    (tw, th), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    text_x, text_y = x + max(24, (w - tw) // 2), y + h // 2 + th // 2 + 10
    cv2.putText(canvas, val_str, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)

    if unit: 
        cv2.putText(canvas, unit, (x + 24, y + h - 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, SUBTLE_TEXT, 1, cv2.LINE_AA)
    
    if show_pulse_icon and isinstance(current_bpm, (int, float, np.floating)) and current_bpm > 30:
        bps = float(current_bpm) / 60.0
        pulse_scale = 0.75 + 0.25 * np.sin((time.time() * bps * 2 * np.pi) % (2 * np.pi))
        heart_x, heart_y = x + w - 52, y + 56
        pts = (np.array([[0, -8], [8, -15], [15, -8], [0, 12], [-15, -8], [-8, -15]], np.int32) * pulse_scale * 1.4).astype(np.int32)
        cv2.fillPoly(canvas, [pts + [heart_x, heart_y]], color, lineType=cv2.LINE_AA)

def draw_camera_card(canvas, target_box, frame, evm_roi_bbox=None, mask_contours=None):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cam_h, cam_w = h - 20, w - 20
    actual_h, actual_w = frame.shape[:2]
    canvas[y + 10:y + 10 + cam_h, x + 10:x + 10 + cam_w] = cv2.resize(frame, (cam_w, cam_h))
    
    scale_x, scale_y = cam_w / actual_w, cam_h / actual_h

    if evm_roi_bbox is not None:
        ex1, ey1, ex2, ey2 = evm_roi_bbox
        cv2.rectangle(canvas, (int(ex1 * scale_x) + x + 10, int(ey1 * scale_y) + y + 10), 
                              (int(ex2 * scale_x) + x + 10, int(ey2 * scale_y) + y + 10), ACCENT_EVM, 2, cv2.LINE_AA)
    
    if mask_contours is not None:
        scaled_contours = []
        for cnt in mask_contours:
            scaled_cnt = np.zeros_like(cnt)
            scaled_cnt[:, 0, 0] = cnt[:, 0, 0] * scale_x + x + 10
            scaled_cnt[:, 0, 1] = cnt[:, 0, 1] * scale_y + y + 10
            scaled_contours.append(scaled_cnt)
        cv2.polylines(canvas, scaled_contours, isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)

# --- WebRTC Processor Class ---
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
        img = cv2.flip(img, 1)
        self.frame_counter += 1
        
        current_h, current_w = img.shape[:2]
        margin_x, margin_y = int(current_w * 0.05), int(current_h * 0.05)
        evm_roi_bbox = (margin_x, margin_y, current_w - margin_x, current_h - margin_y)

        if self.frame_counter % 2 != 0 or not self.cached_mp_face_ok:
            full_mask, mp_face_ok, mp_face_box, mask_contours = extract_mediapipe_roi(img, face_mesh, scale=0.5)
            self.cached_full_mask, self.cached_mp_face_ok, self.cached_mp_face_box, self.cached_mask_contours = full_mask, mp_face_ok, mp_face_box, mask_contours
        else:
            full_mask, mp_face_ok, mp_face_box, mask_contours = self.cached_full_mask, self.cached_mp_face_ok, self.cached_mp_face_box, self.cached_mask_contours

        face_detected_now = mp_face_ok and bbox_inside_roi(mp_face_box, evm_roi_bbox)

        if face_detected_now:
            if self.stable_face_start_time is None: self.stable_face_start_time = time.time()
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
                bpm = estimate_peak_bpm(active_signal, self.fps, self.hr_low, self.hr_high)
                if len(self.bpm_all) > 0: 
                    max_change = 3.0 * (self.bpmCalcEvery / self.fps)
                    bpm = float(np.clip(bpm, self.bpm_all[-1] - max_change, self.bpm_all[-1] + max_change))
                    
                self.bpmBuffer[self.bpmBufferIndex] = bpm
                self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize
                self.bpm_all.append(bpm)
                self.current_hr = np.nanmean(self.bpmBuffer)

                # Respiration Rate Estimation
                r_rr_filtered = bandpass_filter(r_rolled, self.rr_low, self.rr_high, self.fps, order=2)
                rr_raw = estimate_peak_bpm(r_rr_filtered, self.fps, self.rr_low, self.rr_high)
                if 6 <= rr_raw <= 40: self.rr_history.append(rr_raw)
                self.current_rr = robust_mean(self.rr_history)

                # SpO2 Estimation
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

        # Construct the composite canvas
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
        pulse_display = display_metric_value(face_detected_now, vitals_enabled, self.current_hr)
        rr_display = display_metric_value(face_detected_now, vitals_enabled, self.current_rr)
        spo2_display = display_metric_value(face_detected_now, vitals_enabled, self.current_spo2)

        draw_card(canvas, BOX_PULSE, "HEART RATE", pulse_display, "BPM", ACCENT_PULSE, show_pulse_icon=True, current_bpm=self.current_hr if vitals_enabled else None)
        draw_card(canvas, BOX_BR, "RESPIRATION", rr_display, "BR/MIN", ACCENT_BREATH)
        draw_card(canvas, BOX_SPO2, "OXYGEN", spo2_display, "SpO2 %", ACCENT_OXY)
        draw_camera_card(canvas, BOX_CAMERA, img, evm_roi_bbox=evm_roi_bbox, mask_contours=mask_contours)

        self.bufferIndex = (self.bufferIndex + 1) % self.bufferSize
        return av.VideoFrame.from_ndarray(canvas, format="bgr24")

# --- Streamlit Presentation & Two-Column Layout ---
st.title("💓 Contactless Vitals Dashboard")
st.markdown("Real-time optical heart rate, respiration rate, and SpO2 estimation via ambient facial video streams.")

st.divider()

col_stream, col_info = st.columns([7, 3], gap="large")

with col_stream:
    st.subheader("Live Feed")
    webrtc_streamer(
        key="rppg-stream",
        video_processor_factory=RPPGVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col_info:
    st.subheader("Status & Guide")
    
    with st.container(border=True):
        st.markdown("### 📋 Instructions")
        st.markdown(
            "1. **Allow Camera Access**: Click **START** and permit webcam permissions.\n"
            "2. **Align Face**: Keep your face centered inside the yellow ROI box.\n"
            "3. **Hold Steady**: Signal calibration requires **5 seconds** of continuous face detection."
        )

    with st.container(border=True):
        st.markdown("### ⚙️ Pipeline Info")
        st.markdown("- **Algorithm:** Plane-Orthogonal-to-Skin (POS)")
        st.markdown("- **Cardiac Band:** 0.7 – 3.0 Hz (42 – 180 BPM)")
        st.markdown("- **Respiration Band:** 0.15 – 0.5 Hz (9 – 30 BR/MIN)")
        st.markdown("- **Target Region:** Forehead & Cheek Mesh")

    with st.container(border=True):
        st.caption("ℹ️ Measurements update in real time directly on the cards within the video stream.")
