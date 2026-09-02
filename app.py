import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt
import mediapipe as mp

# ==============================================================================
# MEDIAPIPE INITIALIZATION
# ==============================================================================
try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    try:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
    except ImportError:
        mp_face_mesh = mp.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Cheeks and forehead indices for stable arterial signal capture
SKIN_REGIONS = {
    "cheeks": [234, 93, 132, 58, 172, 136, 150, 454, 323, 361, 288, 397, 365, 379]
}

def get_segmented_mask(frame_shape, landmarks):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    landmarks_px = [(min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)) for lm in landmarks.landmark]
    def get_hull(indices): 
        return cv2.convexHull(np.array([landmarks_px[i] for i in indices], dtype=np.int32))
    for _, indices in SKIN_REGIONS.items(): 
        cv2.fillPoly(mask, [get_hull(indices)], 255)
    return mask

def clamp(val, min_val, max_val): 
    return max(min_val, min(val, max_val))

def bbox_inside_roi(box, roi):
    if box is None or roi is None: return False
    bx, by, bw, bh = box
    rx1, ry1, rx2, ry2 = roi
    return (bx >= rx1 and by >= ry1 and (bx + bw) <= rx2 and (by + bh) <= ry2)

def extract_mediapipe_roi(frame, mesh_obj, scale=0.4):
    h, w = frame.shape[:2]
    small_w, small_h = int(w * scale), int(h * scale)
    small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    results = mesh_obj.process(rgb_frame)
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
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x + bw), max(max_y, y + bh)
            mp_face_box = (min_x, min_y, max_x - min_x, max_y - min_y)
            
    return full_mask, mp_face_ok, mp_face_box, mask_contours

# ==============================================================================
# MATHEMATICAL SIGNAL PROCESSING & EVM
# ==============================================================================
def bgr_to_yiq(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(img)
    return 0.299*R + 0.587*G + 0.114*B, 0.596*R - 0.274*G - 0.322*B, 0.211*R - 0.523*G + 0.312*B

def yiq_to_bgr(Y, I, Q):
    img = cv2.merge([Y - 1.106*I + 1.703*Q, Y - 0.272*I - 0.647*Q, Y + 0.956*I + 0.621*Q])
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    if len(data) < 15: return data
    b, a = butter(order, [lowcut / (0.5 * fs), min(0.99, highcut / (0.5 * fs))], btype='band')
    try: 
        return filtfilt(b, a, data, padlen=min(len(data) - 1, 3 * max(len(b), len(a))))
    except ValueError: 
        return data

def extract_pos(r, g, b, fps_val):
    N, l = len(r), int(2 * fps_val)
    if N < l: return np.zeros(N, dtype=np.float32)
    eps = 1e-6
    rw = np.lib.stride_tricks.sliding_window_view(r, l)
    gw = np.lib.stride_tricks.sliding_window_view(g, l)
    bw = np.lib.stride_tricks.sliding_window_view(b, l)
    rn = rw / (np.mean(rw, axis=1, keepdims=True) + eps)
    gn = gw / (np.mean(gw, axis=1, keepdims=True) + eps)
    bn = bw / (np.mean(bw, axis=1, keepdims=True) + eps)
    S1 = gn - bn
    S2 = gn + bn - 2 * rn
    h = S1 + (np.std(S1, axis=1, keepdims=True) / (np.std(S2, axis=1, keepdims=True) + eps)) * S2
    h_zm = h - np.mean(h, axis=1, keepdims=True)
    H = np.zeros(N, dtype=np.float32)
    for i in range(l): 
        H[i : i + h_zm.shape[0]] += h_zm[:, i]
    return H

def estimate_peak_bpm(signal, fps_val, min_hz, max_hz):
    mag = np.abs(np.fft.fft(signal - float(np.mean(signal)), n=512))
    freqs = np.fft.fftfreq(512, d=1.0 / max(float(fps_val), 1e-6))
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    return float(freqs[int(np.argmax(mag * mask))] * 60.0) if np.any(mask) else 0.0

def robust_mean(data):
    if not data: return None
    return float(np.median(list(data))) if len(data) >= 3 else float(np.mean(list(data)))

# ==============================================================================
# SIGNAL QUALITY INDEX (SQI)
# ==============================================================================
def compute_sqi(motion_score, brightness_val, periodicity_score, face_detected, vitals_enabled):
    if not face_detected:
        return 0.0, "NO FACE"
    if not vitals_enabled:
        return 20.0, "LOCKING..."
    
    # Normalize motion (0 to 1, lower motion is better)
    m_factor = max(0.0, 1.0 - (motion_score / 15.0))
    # Brightness penalty if under 40 or over 220
    b_factor = 1.0 if (40.0 <= brightness_val <= 220.0) else 0.3
    # Periodicity factor (0 to 1)
    p_factor = min(1.0, max(0.0, periodicity_score))
    
    score = (0.35 * m_factor + 0.25 * b_factor + 0.40 * p_factor) * 100.0
    if score >= 75: label = "EXCELLENT"
    elif score >= 50: label = "GOOD"
    elif score >= 30: label = "ACCEPTABLE"
    else: label = "POOR SIGNAL"
    return score, label

# ==============================================================================
# BENTO GRID UI RENDERING
# ==============================================================================
BG_COLOR = (18, 18, 22)
CARD_COLOR = (30, 30, 38)
BORDER_COLOR = (55, 55, 68)
TITLE_COLOR = (140, 140, 160)
SUBTLE_TEXT = (100, 100, 120)
ACCENT_PULSE = (75, 75, 245)
ACCENT_BREATH = (245, 160, 60)
ACCENT_OXY = (60, 220, 120)
ACCENT_WAVE = (0, 220, 255)

def draw_card(canvas, box, title, value, unit, color, show_pulse=False, current_bpm=None):
    x, y, w, h = box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 1)
    cv2.putText(canvas, title, (x + 18, y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, TITLE_COLOR, 1, cv2.LINE_AA)

    if isinstance(value, str): 
        val_str, font_scale, thick = value, 0.75, 1
    elif value is not None and value > 0: 
        val_str, font_scale, thick = f"{value:.0f}", 1.9, 2
    else: 
        val_str, font_scale, thick = "--", 1.9, 2

    (tw, th), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thick)
    cv2.putText(canvas, val_str, (x + max(18, (w - tw) // 2), y + h // 2 + th // 2 + 5), 
                cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thick, cv2.LINE_AA)
    if unit: 
        cv2.putText(canvas, unit, (x + 18, y + h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.55, SUBTLE_TEXT, 1, cv2.LINE_AA)

    if show_pulse and isinstance(current_bpm, (int, float, np.floating)) and current_bpm > 30:
        pulse_scale = 0.75 + 0.25 * np.sin((time.time() * (float(current_bpm) / 60.0) * 2 * np.pi) % (2 * np.pi))
        pts = (np.array([[0, -8], [8, -15], [15, -8], [0, 12], [-15, -8], [-8, -15]], np.int32) * pulse_scale * 1.2).astype(np.int32)
        cv2.fillPoly(canvas, [pts + [x + w - 35, y + 35]], color, lineType=cv2.LINE_AA)

def draw_camera_card(canvas, box, frame, mask_contours=None, face_box=None):
    x, y, w, h = box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cam_h, cam_w = h - 16, w - 16
    actual_h, actual_w = frame.shape[:2]
    
    resized_cam = cv2.resize(frame, (cam_w, cam_h))
    canvas[y + 8:y + 8 + cam_h, x + 8:x + 8 + cam_w] = resized_cam
    scale_x, scale_y = cam_w / actual_w, cam_h / actual_h

    if face_box is not None:
        fx, fy, fw, fh = face_box
        cv2.rectangle(canvas, 
                      (int(fx * scale_x) + x + 8, int(fy * scale_y) + y + 8),
                      (int((fx + fw) * scale_x) + x + 8, int((fy + fh) * scale_y) + y + 8),
                      (0, 255, 200), 2, cv2.LINE_AA)

    if mask_contours:
        scaled_cnts = []
        for cnt in mask_contours:
            s_cnt = np.zeros_like(cnt)
            s_cnt[:, 0, 0] = cnt[:, 0, 0] * scale_x + x + 8
            s_cnt[:, 0, 1] = cnt[:, 0, 1] * scale_y + y + 8
            scaled_cnts.append(s_cnt)
        cv2.polylines(canvas, scaled_cnts, isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 1)

def draw_waveform_card(canvas, box, waveform, sqi_score, sqi_label):
    x, y, w, h = box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 1)
    
    cv2.putText(canvas, "PULSE WAVEFORM (POS)", (x + 18, y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, TITLE_COLOR, 1, cv2.LINE_AA)
    sqi_color = (0, 255, 120) if sqi_score >= 60 else (0, 180, 255) if sqi_score >= 35 else (0, 70, 240)
    cv2.putText(canvas, f"SQI: {sqi_score:.0f}% ({sqi_label})", (x + w - 240, y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, sqi_color, 1, cv2.LINE_AA)

    if len(waveform) > 1:
        plot_x = x + 18
        plot_y = y + 36
        plot_w = w - 36
        plot_h = h - 46
        
        arr = np.array(waveform, dtype=np.float32)
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        range_v = max(max_v - min_v, 1e-4)
        
        points = []
        for i, val in enumerate(arr):
            px = int(plot_x + (i / max(len(arr) - 1, 1)) * plot_w)
            py = int(plot_y + plot_h - ((val - min_v) / range_v) * plot_h)
            points.append((px, py))
            
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], isClosed=False, color=ACCENT_WAVE, thickness=2, lineType=cv2.LINE_AA)

# ==============================================================================
# WEBRTC STREAMING PROCESSOR
# ==============================================================================
class BentoRPPGProcessor(VideoProcessorBase):
    def __init__(self):
        self.fps = 15.0
        self.buffer_size = int(self.fps * 10)
        self.bpm_buffer_size = 10
        self.bpm_calc_every = int(self.fps * 1)
        self.face_stable_seconds = 4.0

        self.hr_low, self.hr_high = 0.7, 3.0
        self.rr_low, self.rr_high = 0.15, 0.5
        self.spo2_a, self.spo2_b = 100.0, 5.0

        # Buffers
        self.red_buffer = np.zeros((self.buffer_size,), dtype=np.float32)
        self.green_buffer = np.zeros((self.buffer_size,), dtype=np.float32)
        self.blue_buffer = np.zeros((self.buffer_size,), dtype=np.float32)
        self.waveform_buffer = deque([0.0] * self.buffer_size, maxlen=self.buffer_size)

        self.bpm_buffer = np.full((self.bpm_buffer_size,), np.nan, dtype=np.float32)
        self.bpm_history = []
        self.rr_history = deque(maxlen=10)
        self.spo2_history = deque(maxlen=10)

        # EVM State
        self.levels = 3
        self.alpha_evm = 40.0
        self.video_pyramid = None
        self.evm_initialized = False

        # Status & Quality
        self.buffer_index = 0
        self.bpm_index = 0
        self.frame_count = 0
        self.buffers_ready = False
        
        self.stable_face_time = None
        self.vitals_enabled = False
        self.smoothed_bbox = None

        self.current_hr = None
        self.current_rr = None
        self.current_spo2 = None
        self.current_sqi_score = 0.0
        self.current_sqi_label = "NO FACE"
        self.prev_gray_roi = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        h, w = img.shape[:2]

        evm_target_roi = (int(w * 0.05), int(h * 0.05), w - int(w * 0.05), h - int(h * 0.05))
        full_mask, face_ok, face_box, mask_contours = extract_mediapipe_roi(img, face_mesh, scale=0.4)
        face_detected = face_ok and bbox_inside_roi(face_box, evm_target_roi)

        # 1. Face Stabilization Gate
        if face_detected:
            if self.stable_face_time is None:
                self.stable_face_time = time.time()
            stable_dur = time.time() - self.stable_face_time
            self.vitals_enabled = (stable_dur >= self.face_stable_seconds)
        else:
            self.stable_face_time = None
            self.vitals_enabled = False
            self.current_hr, self.current_rr, self.current_spo2 = None, None, None

        vitals_roi, vitals_mask = None, None
        motion_score = 0.0
        brightness_val = 120.0

        if face_box is not None and face_detected:
            rx, ry, rw, rh = face_box
            if self.smoothed_bbox is None or not face_detected:
                self.smoothed_bbox = [rx, ry, rw, rh]
            else:
                a_sm = 0.15
                self.smoothed_bbox = [(1.0 - a_sm)*s + a_sm*r for s, r in zip(self.smoothed_bbox, [rx, ry, rw, rh])]

            vx1, vy1, vbw, vbh = [int(v) for v in self.smoothed_bbox]
            vx2, vy2 = clamp(vx1 + vbw, 1, w), clamp(vy1 + vbh, 1, h)
            vx1, vy1 = clamp(vx1, 0, w - 1), clamp(vy1, 0, h - 1)

            if vx2 > vx1 and vy2 > vy1:
                vitals_roi = img[vy1:vy2, vx1:vx2, :]
                vitals_mask = full_mask[vy1:vy2, vx1:vx2]
                
                gray_roi = cv2.cvtColor(vitals_roi, cv2.COLOR_BGR2GRAY)
                brightness_val = float(np.mean(gray_roi))
                if self.prev_gray_roi is not None and self.prev_gray_roi.shape == gray_roi.shape:
                    motion_score = float(np.mean(np.abs(gray_roi.astype(np.float32) - self.prev_gray_roi.astype(np.float32))))
                self.prev_gray_roi = gray_roi

        # 2. EVM Magnification Loop
        if vitals_roi is not None and vitals_mask is not None and self.vitals_enabled:
            Y, I, Q = bgr_to_yiq(vitals_roi)
            processed = cv2.merge([Y, I, Q])
            blurred = processed
            for _ in range(self.levels):
                blurred = cv2.pyrDown(blurred)

            if not self.evm_initialized or self.video_pyramid.shape[1:3] != blurred.shape[:2]:
                self.video_pyramid = np.zeros((self.buffer_size, blurred.shape[0], blurred.shape[1], 3), dtype=np.float32)
                self.evm_initialized = True

            self.video_pyramid[self.buffer_index] = blurred
            fft_pyr = np.fft.fft(np.roll(self.video_pyramid, -self.buffer_index - 1, axis=0), axis=0)
            
            freqs_evm = np.fft.fftfreq(self.buffer_size, d=1.0 / self.fps)
            mask_evm = (freqs_evm >= self.hr_low) & (freqs_evm <= self.hr_high)
            fft_pyr[~mask_evm] = 0
            
            filtered = np.real(np.fft.ifft(fft_pyr, axis=0))[-1] * self.alpha_evm
            filtered[:, :, 0] = 0.0  # Zero luminance channel to suppress lighting swings

            for _ in range(self.levels):
                filtered = cv2.pyrUp(filtered)
            filtered = cv2.resize(filtered, (vitals_roi.shape[1], vitals_roi.shape[0]))
            
            bin_mask = np.expand_dims((vitals_mask > 0).astype(np.float32), axis=-1)
            pulse_mod = np.clip(filtered * bin_mask, -15.0, 15.0)
            
            out_Y, out_I, out_Q = cv2.split(processed + pulse_mod)
            out_vitals = yiq_to_bgr(out_Y, out_I, out_Q)
            img[vy1:vy2, vx1:vx2, :] = out_vitals

        # 3. Extraction & Estimation
        if vitals_roi is not None and vitals_mask is not None:
            mean_c = cv2.mean(vitals_roi, mask=vitals_mask)
            r_val, g_val, b_val = float(mean_c[2]), float(mean_c[1]), float(mean_c[0])

            if not self.buffers_ready:
                self.red_buffer[:], self.green_buffer[:], self.blue_buffer[:] = r_val, g_val, b_val
                self.buffers_ready = True
            else:
                self.red_buffer[self.buffer_index] = r_val
                self.green_buffer[self.buffer_index] = g_val
                self.blue_buffer[self.buffer_index] = b_val

        r_rolled = np.roll(self.red_buffer, -self.buffer_index - 1)
        g_rolled = np.roll(self.green_buffer, -self.buffer_index - 1)
        b_rolled = np.roll(self.blue_buffer, -self.buffer_index - 1)

        periodicity = 0.0
        if self.buffers_ready and face_detected:
            pos_signal = extract_pos(r_rolled, g_rolled, b_rolled, self.fps)
            filtered_signal = bandpass_filter(pos_signal, self.hr_low, self.hr_high, self.fps, order=2)
            self.waveform_buffer.append(float(filtered_signal[-1]))

            # Periodicity calculation for SQI
            mag = np.abs(np.fft.fft(filtered_signal - np.mean(filtered_signal)))
            periodicity = float(np.max(mag) / (np.sum(mag) + 1e-6)) * 4.0

            if self.vitals_enabled and (self.buffer_index % self.bpm_calc_every == 0):
                # Heart Rate
                bpm = estimate_peak_bpm(filtered_signal, self.fps, self.hr_low, self.hr_high)
                if len(self.bpm_history) > 0:
                    max_d = 3.0 * (self.bpm_calc_every / self.fps)
                    bpm = float(np.clip(bpm, self.bpm_history[-1] - max_d, self.bpm_history[-1] + max_d))

                self.bpm_buffer[self.bpm_index] = bpm
                self.bpm_index = (self.bpm_index + 1) % self.bpm_buffer_size
                self.bpm_history.append(bpm)
                self.current_hr = np.nanmean(self.bpm_buffer)

                # Respiration Rate
                r_rr = bandpass_filter(r_rolled, self.rr_low, self.rr_high, self.fps, order=2)
                rr_val = estimate_peak_bpm(r_rr, self.fps, self.rr_low, self.rr_high)
                if 6 <= rr_val <= 40:
                    self.rr_history.append(rr_val)
                self.current_rr = robust_mean(self.rr_history)

                # SpO2
                ac_r = float(np.std(bandpass_filter(r_rolled, self.hr_low, self.hr_high, self.fps, order=2)))
                dc_r = float(np.mean(r_rolled))
                ac_b = float(np.std(bandpass_filter(b_rolled, self.hr_low, self.hr_high, self.fps, order=2)))
                dc_b = float(np.mean(b_rolled))

                if dc_r > 1e-3 and dc_b > 1e-3 and ac_b > 1e-6:
                    r_ratio = (ac_r / dc_r) / (ac_b / dc_b)
                    spo2_val = float(self.spo2_a - (self.spo2_b * r_ratio))
                    if 85.0 <= spo2_val <= 100.0:
                        self.spo2_history.append(min(spo2_val, 100.0))
                self.current_spo2 = robust_mean(self.spo2_history)
        else:
            self.waveform_buffer.append(0.0)

        self.current_sqi_score, self.current_sqi_label = compute_sqi(
            motion_score, brightness_val, periodicity, face_detected, self.vitals_enabled
        )
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size

        # 4. Synthesize 960x540 Bento Grid Canvas
        CANVAS_W, CANVAS_H = 960, 540
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

        # Left Column Cards
        draw_card(canvas, (20, 20, 240, 150), "HEART RATE", 
                  self.current_hr if self.vitals_enabled else ("LOCKING..." if face_detected else None), 
                  "BPM", ACCENT_PULSE, show_pulse=True, current_bpm=self.current_hr if self.vitals_enabled else None)
        
        draw_card(canvas, (20, 185, 240, 150), "RESPIRATION", 
                  self.current_rr if self.vitals_enabled else None, 
                  "BR/MIN", ACCENT_BREATH)
        
        draw_card(canvas, (20, 350, 240, 165), "OXYGEN", 
                  self.current_spo2 if self.vitals_enabled else None, 
                  "SpO2 %", ACCENT_OXY)

        # Right Column Cards
        draw_camera_card(canvas, (280, 20, 660, 350), img, mask_contours, self.smoothed_bbox)
        draw_waveform_card(canvas, (280, 385, 660, 130), self.waveform_buffer, self.current_sqi_score, self.current_sqi_label)

        return av.VideoFrame.from_ndarray(canvas, format="bgr24")

# ==============================================================================
# STREAMLIT ENTRYPOINT
# ==============================================================================
st.set_page_config(page_title="rPPG Vitals Monitor", layout="wide")
st.markdown("<h2 style='text-align: center; margin-bottom: 10px;'>Contactless Vitals Dashboard</h2>", unsafe_allow_html=True)

webrtc_streamer(
    key="rppg-bento-stream",
    video_processor_factory=BentoRPPGProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False}
)
