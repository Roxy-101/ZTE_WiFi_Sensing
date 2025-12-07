import time
import os
import numpy as np
import pandas as pd
from scipy import signal
from collections import deque
import tkinter as tk
from tkinter import font
import threading
from sklearn.decomposition import PCA
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# 1. 配置区域
# ==============================================================================
CSI_CSV_FILE = '/home/roxy-0x00/Desktop/escape_101/Demo/ZTE_WiFi_Sensing/data/20250922=155010_part0.csv'
BUFFER_SIZE = 256
CARRIER_FREQUENCY = 5.21e9
VALID_SUBCARRIER_RANGE = range(256, 512)
FALLBACK_SAMPLE_RATE = 500
# ==============================================================================

# --- 数据处理函数 (无需改动) ---
def parse_csi_string(csi_str):
    try:
        values = csi_str.strip().strip('()').split(',')
        return [int(v.strip()) for v in values if v.strip()]
    except: return []

def parse_csi_line(line, header):
    try:
        line_df = pd.read_csv(pd.io.common.StringIO(line), names=header, header=None)
        timestamp_us = int(line_df['timestamp'].iloc[0])
        csi_i = np.array(parse_csi_string(line_df['csi_i'].iloc[0]))
        csi_q = np.array(parse_csi_string(line_df['csi_q'].iloc[0]))
        min_len = min(len(csi_i), len(csi_q))
        if min_len > 0: return timestamp_us, csi_i[:min_len] + 1j * csi_q[:min_len]
        return None, None
    except Exception: return None, None

def filter_static_with_pca(csi_matrix):
    csi_phase = np.unwrap(np.angle(csi_matrix), axis=0)
    csi_phase_centered = csi_phase - np.mean(csi_phase, axis=0)
    pca = PCA(n_components=1)
    dynamic_phase = np.zeros_like(csi_phase_centered)
    for i in range(csi_phase_centered.shape[1]):
        data = csi_phase_centered[:, i].reshape(-1, 1)
        if data.shape[0] < 2: continue
        principal_component = pca.fit_transform(data)
        reconstructed_pc1 = pca.inverse_transform(principal_component)
        dynamic_phase[:, i] = (data - reconstructed_pc1).flatten()
    return dynamic_phase

def estimate_velocity_from_buffer(csi_buffer, carrier_freq_hz, app_ui):
    if len(csi_buffer) < BUFFER_SIZE: return None, None
    
    timestamps = [item[0] for item in csi_buffer]
    csi_matrix = np.array([item[1] for item in csi_buffer])
    
    time_span_s = (timestamps[-1] - timestamps[0]) / 1_000_000.0
    dynamic_sample_rate = (len(timestamps) - 1) / time_span_s if time_span_s > 0 else FALLBACK_SAMPLE_RATE

    valid_csi_matrix = csi_matrix[:, list(VALID_SUBCARRIER_RANGE)]
    
    # --- 【关键改动】将幅度矩阵传递给UI用于绘制热力图 ---
    csi_amplitudes = np.abs(valid_csi_matrix)
    app_ui.update_heatmap(csi_amplitudes)
    # ------------------------------------------------

    sanitized_phase_matrix = filter_static_with_pca(valid_csi_matrix)
    avg_sanitized_phase = np.mean(sanitized_phase_matrix, axis=1)
    
    fft_result = np.fft.fft(avg_sanitized_phase)
    fft_freqs = np.fft.fftfreq(BUFFER_SIZE, 1.0 / dynamic_sample_rate)
    
    positive_freq_mask = (fft_freqs > 1) 
    if not np.any(positive_freq_mask): return None, None

    fft_spectrum_abs = np.abs(fft_result)
    peak_index = np.argmax(fft_spectrum_abs[positive_freq_mask])
    doppler_shift_hz = fft_freqs[positive_freq_mask][peak_index]

    SPEED_OF_LIGHT = 3e8
    velocity_ms = (doppler_shift_hz * SPEED_OF_LIGHT) / (2 * carrier_freq_hz)
    
    app_ui.update_plot(fft_freqs[positive_freq_mask], fft_spectrum_abs[positive_freq_mask])
    
    return velocity_ms, dynamic_sample_rate

# --- 后台线程 (无需改动) ---
def file_monitoring_thread(app_ui):
    if not os.path.exists(CSI_CSV_FILE):
        app_ui.update_display("错误：文件不存在！", "")
        return

    csi_buffer = deque(maxlen=BUFFER_SIZE)
    try:
        with open(CSI_CSV_FILE, 'r') as f: header = f.readline().strip().split(',')
    except (IOError, StopIteration):
        app_ui.update_display("错误：文件无法读取或为空！", "")
        return

    app_ui.update_display("等待数据...", f"监控文件: {os.path.basename(CSI_CSV_FILE)}")

    with open(CSI_CSV_FILE, 'r') as file:
        file.seek(0, 2)
        while True:
            if not app_ui.is_running: break
            line = file.readline()
            if not line:
                time.sleep(0.01)
                continue
            
            timestamp, csi_vector = parse_csi_line(line, header)
            if csi_vector is not None and csi_vector.shape[0] >= max(VALID_SUBCARRIER_RANGE):
                csi_buffer.append((timestamp, csi_vector))
                if len(csi_buffer) == BUFFER_SIZE:
                    velocity, calculated_rate = estimate_velocity_from_buffer(
                        csi_buffer, CARRIER_FREQUENCY, app_ui
                    )
                    if velocity is not None:
                        rate_text = f"实时采样率: {calculated_rate:.2f} Hz"
                        velocity_text = f"{velocity:.2f} m/s"
                        app_ui.update_display(velocity_text, rate_text)
    print("后台监控线程已停止。")

# --- Tkinter UI 类 (布局调整，增加热力图) ---
class CsiUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("实时CSI感知：速度检测与信号热力图")
        self.geometry("800x800") # 增加窗口高度以容纳两个图
        
        # --- 创建字体 ---
        self.velocity_font = font.Font(family="Helvetica", size=48, weight="bold")
        self.info_font = font.Font(family="Helvetica", size=14)
        
        # --- 创建上半部分UI框架 ---
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.velocity_text = tk.StringVar()
        self.velocity_label = tk.Label(top_frame, textvariable=self.velocity_text, font=self.velocity_font, pady=10)
        self.velocity_label.pack()
        
        self.info_text = tk.StringVar()
        self.info_label = tk.Label(top_frame, textvariable=self.info_text, font=self.info_font, fg="gray")
        self.info_label.pack()
        
        # 多普勒频谱图
        self.fig_doppler = Figure(figsize=(8, 3), dpi=100)
        self.ax_doppler = self.fig_doppler.add_subplot(111)
        self.canvas_doppler = FigureCanvasTkAgg(self.fig_doppler, master=top_frame)
        self.canvas_doppler.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- 创建下半部分UI框架 (用于热力图) ---
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.fig_heatmap = Figure(figsize=(8, 3), dpi=100)
        self.ax_heatmap = self.fig_heatmap.add_subplot(111)
        self.canvas_heatmap = FigureCanvasTkAgg(self.fig_heatmap, master=bottom_frame)
        self.canvas_heatmap.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.is_running = True
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_display(self, velocity, info):
        self.velocity_text.set(velocity)
        self.info_text.set(info)

    def update_plot(self, freqs, spectrum_abs):
        self.ax_doppler.clear()
        spectrum_db = 20 * np.log10(spectrum_abs + 1e-9)
        self.ax_doppler.plot(freqs, spectrum_db)
        
        peak_idx = np.argmax(spectrum_db)
        peak_freq = freqs[peak_idx]
        peak_mag = spectrum_db[peak_idx]
        self.ax_doppler.plot(peak_freq, peak_mag, "x", color='red', markersize=10, label=f'Peak: {peak_freq:.2f} Hz')

        self.ax_doppler.set_title("Real-Time Doppler Spectrum")
        self.ax_doppler.set_xlabel("Dopperr Shift (Hz)")
        self.ax_doppler.set_ylabel("Magnitude (dB)")
        self.ax_doppler.grid(True)
        self.ax_doppler.legend()
        self.fig_doppler.tight_layout()
        self.canvas_doppler.draw()
    
    def update_heatmap(self, csi_amplitudes):
        """新增：更新CSI幅度热力图"""
        self.ax_heatmap.clear()
        
        # 转置数据，使子载波在Y轴，时间（数据包索引）在X轴
        heatmap_data = csi_amplitudes.T
        
        im = self.ax_heatmap.imshow(heatmap_data, 
                                    aspect='auto', 
                                    cmap='viridis', 
                                    origin='lower',
                                    extent=[0, heatmap_data.shape[1], min(VALID_SUBCARRIER_RANGE), max(VALID_SUBCARRIER_RANGE) + 1])
        
        self.ax_heatmap.set_title("Real-Time CSI Amplitude Heatmap")
        self.ax_heatmap.set_xlabel(f"Time (Last {heatmap_data.shape[1]} Packets)")
        self.ax_heatmap.set_ylabel("Subcarrier Index")
        
        # 如果colorbar不存在则创建，存在则更新
        if not hasattr(self, 'colorbar_heatmap'):
            self.colorbar_heatmap = self.fig_heatmap.colorbar(im, ax=self.ax_heatmap, label='CSI Amplitude')
        else:
            self.colorbar_heatmap.update_normal(im)

        self.fig_heatmap.tight_layout()
        self.canvas_heatmap.draw()

    def on_closing(self):
        self.is_running = False
        self.destroy()

# --- 主程序入口 (无需改动) ---
if __name__ == "__main__":
    try:
        import sklearn
    except ImportError:
        print("正在安装 scikit-learn 库...")
        os.system('pip install --user scikit-learn')

    app = CsiUI()
    monitor_thread = threading.Thread(target=file_monitoring_thread, args=(app,), daemon=True)
    monitor_thread.start()
    app.mainloop()