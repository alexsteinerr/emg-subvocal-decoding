import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import serial
import serial.tools.list_ports
import threading
import collections
import time
import pickle
import os
import numpy as np
import json
import random
from datetime import datetime
from scipy import signal

CONFIG_FILE = "config.json"
DEFAULT_CONF = {
    "baud_rate": 2000000, 
    "sample_rate": 1000,
    "visible_sec": 5,
    "channels": 5,
    "scale_factor": 0.02235,
    "root_dir": "Harvard_EMG_Dataset",
    "targets": {},
    "hp_cut_hz": 0.5,
    "hp_order": 1,
    "bandpass_low_hz": 20,
    "bandpass_high_hz": 150,
    "bandpass_order": 4,
    "mains_freq": 50.0,
    "notch_q": 30.0,
    "heartbeat_remove": True,
    "heartbeat_width_s": 0.12,
}

try:
    import harvard_data
    RAW_SENTENCES = harvard_data.get_harvard_sentences_with_ids()
    SENTENCE_TO_ID = {txt: idx for idx, txt in RAW_SENTENCES}
    DEFAULT_TARGETS = {txt: 15 for idx, txt in RAW_SENTENCES}
except ImportError:
    DEFAULT_TARGETS = {"Error: harvard_data.py missing": 10}
    SENTENCE_TO_ID = {}
    RAW_SENTENCES = []

DEFAULT_CONF["targets"] = DEFAULT_TARGETS
CONF = {}
WINDOW_SIZE = 0

class DataManager:
    @staticmethod
    def init_structure():
        if not os.path.exists(CONF["root_dir"]):
            os.makedirs(CONF["root_dir"])
        
        meta_path = os.path.join(CONF["root_dir"], "metadata.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump({
                    "dataset_name": "Harvard_EMG_OpenVocab",
                    "channels": CONF["channels"],
                    "sample_rate": CONF["sample_rate"],
                    "created_at": str(datetime.now()),
                    "processing": "standard_pipeline"
                }, f, indent=4)

        splits_dir = os.path.join(CONF["root_dir"], "splits")
        if not os.path.exists(splits_dir):
            os.makedirs(splits_dir)

    @staticmethod
    def get_session_dir(user_id):
        date_str = datetime.now().strftime("%Y%m%d")
        sess_name = f"Session_{date_str}"
        path = os.path.join(CONF["root_dir"], user_id, sess_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def update_splits(user_id, filename, sentence_text):
        if sentence_text not in SENTENCE_TO_ID:
            return
        s_id = SENTENCE_TO_ID[sentence_text]
        splits_dir = os.path.join(CONF["root_dir"], "splits")
        list_file = os.path.join(splits_dir, "train_list.txt") if s_id <= 650 else os.path.join(splits_dir, "test_list.txt")
        
        date_str = datetime.now().strftime("%Y%m%d")
        sess_name = f"Session_{date_str}"
        rel_path = os.path.join(user_id, sess_name, filename)

        with open(list_file, "a") as f:
            f.write(rel_path + "\n")

    @staticmethod
    def delete_entry(user_id, filename):
        """Deletes the .pkl file and removes it from manifest and splits."""
        sess_dir = DataManager.get_session_dir(user_id)
        file_path = os.path.join(sess_dir, filename)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting file: {e}")
        man_path = os.path.join(sess_dir, "manifest.json")
        if os.path.exists(man_path):
            try:
                with open(man_path, "r") as f:
                    data = json.load(f)
                
                new_data = [x for x in data if x.get("filename") != filename]
                
                with open(man_path, "w") as f:
                    json.dump(new_data, f, indent=2)
            except Exception as e:
                print(f"Error updating manifest: {e}")

        splits_dir = os.path.join(CONF["root_dir"], "splits")
        date_str = datetime.now().strftime("%Y%m%d")
        sess_name = f"Session_{date_str}"
        rel_path = os.path.join(user_id, sess_name, filename)

        for fname in ["train_list.txt", "test_list.txt"]:
            fpath = os.path.join(splits_dir, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r") as f:
                        lines = f.readlines()
                    
                    # Write back all lines except the one matching our file
                    with open(fpath, "w") as f:
                        for line in lines:
                            if rel_path not in line:
                                f.write(line)
                except Exception as e:
                    print(f"Error updating splits: {e}")

def load_config():
    global CONF, WINDOW_SIZE
    loaded = False
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                CONF = DEFAULT_CONF.copy()
                CONF.update({k: v for k, v in data.items() if k in CONF})
                loaded = True
        except:
            pass

    if not loaded:
        CONF = DEFAULT_CONF.copy()
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(CONF, f, indent=4)
        except:
            pass

    WINDOW_SIZE = int(CONF["sample_rate"] * CONF["visible_sec"])

load_config()
DataManager.init_structure()

class SignalProcessor:
    def __init__(self, fs: int):
        self.fs = fs
        self.sos_hp = signal.butter(CONF["hp_order"], CONF["hp_cut_hz"], btype="highpass", fs=self.fs, output="sos")
        self.sos_bp = signal.butter(CONF["bandpass_order"], [CONF["bandpass_low_hz"], CONF["bandpass_high_hz"]], btype="bandpass", fs=self.fs, output="sos")

        self.notches = []
        f0 = float(CONF["mains_freq"])
        nyq = self.fs / 2.0
        k = 1
        while (k * f0) < nyq:
            fk = k * f0
            b, a = signal.iirnotch(fk, CONF["notch_q"], fs=self.fs)
            self.notches.append((b, a, fk))
            k += 1

        self.hb_enabled = bool(CONF["heartbeat_remove"])
        width = max(8, int(CONF["heartbeat_width_s"] * self.fs))
        M = width * 6 + 1
        self.hb_wavelet = self.ricker_wavelet(M, float(width))
        self.hb_wavelet = self.hb_wavelet / (np.linalg.norm(self.hb_wavelet) + 1e-12)

    def ricker_wavelet(self, M: int, a: float) -> np.ndarray:
        t = np.arange(M, dtype=np.float64) - (M - 1) / 2.0
        A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
        wsq = a * a
        return A * (1.0 - (t * t) / wsq) * np.exp(-(t * t) / (2.0 * wsq))

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("process() expects shape (N, C)")
        N, C = x.shape
        if N < 32:
            return x.copy()

        y = x.astype(np.float64, copy=True)
        y -= y[0:1, :] 

        try:
            y = signal.sosfiltfilt(self.sos_hp, y, axis=0)
        except: pass

        y -= np.mean(y, axis=0, keepdims=True)

        for (b, a, fk) in self.notches:
            try:
                y = signal.filtfilt(b, a, y, axis=0)
            except: pass

        if self.hb_enabled:
            try:
                conv = np.zeros_like(y)
                for ch in range(C):
                    c = signal.fftconvolve(y[:, ch], self.hb_wavelet, mode="same")
                    conv[:, ch] = c
                denom = np.sum(conv * conv, axis=0) + 1e-12
                alpha = np.sum(y * conv, axis=0) / denom
                y = y - conv * alpha
            except: pass

        try:
            y = signal.sosfiltfilt(self.sos_bp, y, axis=0)
        except: pass

        y -= np.mean(y, axis=0, keepdims=True)
        return y.astype(np.float32)

class SerialWorker:
    def __init__(self):
        self.ser = None
        self.running = False
        self.lock = threading.Lock()
        self.plot_buffers = [collections.deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(CONF["channels"])]
        self.record_buffer = []
        self.is_recording = False
        self.start_time = 0.0

    def connect(self):
        ports = list(serial.tools.list_ports.comports())
        target = None
        priority = ["usbmodem", "usbserial", "COM", "ttyACM", "ttyUSB"]
        for p in ports:
            if any(x in p.device for x in priority):
                target = p.device
                break
        
        if target:
            try:
                self.ser = serial.Serial(target, CONF["baud_rate"], timeout=0.1)
                self.running = True
                threading.Thread(target=self.loop, daemon=True).start()
                return target
            except: pass
        return None

    def loop(self):
        try:
            self.ser.reset_input_buffer()
        except: return

        while self.running and self.ser and self.ser.is_open:
            lines_to_process = []
            try:
                while self.ser.in_waiting:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line and "Ch" not in line:
                        lines_to_process.append(line)
            except: pass

            if lines_to_process:
                new_data = []
                current_time = time.time()
                for line in lines_to_process:
                    try:
                        parts = line.split(",")
                        if len(parts) >= CONF["channels"]:
                            vals = [float(p) * CONF["scale_factor"] for p in parts[: CONF["channels"]]]
                            new_data.append(vals)
                    except: pass
                
                if new_data:
                    with self.lock:
                        for vals in new_data:
                            for i in range(CONF["channels"]):
                                self.plot_buffers[i].append(vals[i])
                            if self.is_recording:
                                self.record_buffer.append([current_time] + vals)
            else:
                time.sleep(0.001)

    def start_recording(self):
        with self.lock:
            self.record_buffer = []
            self.is_recording = True
            self.start_time = time.time()

    def stop_recording(self):
        with self.lock:
            self.is_recording = False
            return list(self.record_buffer)

    def get_stats(self):
        with self.lock:
            if self.is_recording:
                dur = time.time() - self.start_time
                count = len(self.record_buffer)
                hz = count / dur if dur > 0 else 0
                return count, dur, hz
            return 0, 0.0, 0.0

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ALSEE - Harvard Recorder")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")

        self.worker = SerialWorker()
        self.processor = SignalProcessor(CONF["sample_rate"])
        
        self.user_counts = {}
        self.batch_size = 10
        self.batch_list = []
        self.batch_ptr = 0
        self.autoscale_var = tk.BooleanVar(value=True)
        
        # State for Redo functionality
        self.last_saved_file = None
        self.last_saved_user = None

        self.setup_ui()
        port = self.worker.connect()
        self.status_lbl.config(text=f"CONNECTED: {port} @ {CONF['baud_rate']}" if port else "NO DEVICE", fg="#00FF00" if port else "#FF4444")

        self.x = np.linspace(0, CONF["visible_sec"], WINDOW_SIZE, endpoint=False)
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=40, blit=False)
        self.update_timer()

    def setup_ui(self):
        header = tk.Frame(self.root, bg="#2d2d2d", pady=15, padx=15)
        header.pack(fill=tk.X)

        tk.Label(header, text="USER:", bg="#2d2d2d", fg="#aaa", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.user_var = tk.StringVar(value="Subject_01")
        self.user_combo = ttk.Combobox(header, textvariable=self.user_var, values=["Subject_01", "Subject_02", "Subject_03", "Subject_04"], width=15)
        self.user_combo.pack(side=tk.LEFT, padx=10)
        self.user_combo.bind("<<ComboboxSelected>>", self.on_user_change)

        tk.Label(header, text="SENTENCE:", bg="#2d2d2d", fg="#aaa").pack(side=tk.LEFT)
        self.combo = ttk.Combobox(header, state="readonly", width=80)
        self.combo.pack(side=tk.LEFT, padx=5)
        self.combo.bind("<<ComboboxSelected>>", self.on_word_select)

        self.count_lbl = tk.Label(header, text="0/0", bg="#2d2d2d", fg="#00ffff", font=("Arial", 14, "bold"))
        self.count_lbl.pack(side=tk.LEFT, padx=15)
        self.status_lbl = tk.Label(header, text="Init...", bg="#2d2d2d", fg="#888")
        self.status_lbl.pack(side=tk.RIGHT)

        ctrl = tk.Frame(self.root, bg="#1e1e1e", pady=15)
        ctrl.pack(fill=tk.X)

        self.btn_start = tk.Button(ctrl, text="● REC", bg="#333", fg="white", font=("Arial", 12, "bold"), width=12, command=self.start_rec)
        self.btn_start.pack(side=tk.LEFT, padx=20)
        self.btn_stop = tk.Button(ctrl, text="■ STOP (Next)", bg="#333", fg="#aaa", font=("Arial", 12, "bold"), width=12, command=self.stop_rec, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        self.btn_redo = tk.Button(ctrl, text="⟲ REDO", bg="#333", fg="#FFD700", font=("Arial", 12, "bold"), width=12, command=self.redo_last, state=tk.DISABLED)
        self.btn_redo.pack(side=tk.LEFT, padx=10)

        self.chk_autoscale = tk.Checkbutton(ctrl, text="Auto Scale Axis", variable=self.autoscale_var, bg="#1e1e1e", fg="#00E5FF", selectcolor="#333", activebackground="#1e1e1e", activeforeground="#00E5FF", font=("Arial", 10, "bold"))
        self.chk_autoscale.pack(side=tk.LEFT, padx=20)

        self.stats_frame = tk.Frame(ctrl, bg="#1e1e1e")
        self.stats_frame.pack(side=tk.LEFT, padx=30)
        self.time_lbl = tk.Label(self.stats_frame, text="0.00 s", bg="#1e1e1e", fg="#555", font=("Consolas", 24))
        self.time_lbl.pack(side=tk.LEFT)
        self.hz_lbl = tk.Label(self.stats_frame, text="0 Hz", bg="#1e1e1e", fg="#333", font=("Consolas", 12))
        self.hz_lbl.pack(side=tk.LEFT, padx=10)

        plot_frame = tk.Frame(self.root, bg="black")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.fig = Figure(figsize=(10, 8), facecolor="#1e1e1e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#101010")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_color("#555")
        self.ax.spines["left"].set_color("#555")
        self.ax.tick_params(axis="x", colors="#888")
        self.ax.tick_params(axis="y", colors="#888")
        self.ax.grid(True, color="#222", linestyle=":", linewidth=0.5)
        self.ax.set_xlim(0, CONF["visible_sec"])

        self.lines = []
        colors = ["#00E5FF", "#FF4081", "#FFEA00", "#76FF03", "#AA00FF"]
        for i in range(CONF["channels"]):
            line, = self.ax.plot([], [], lw=1.2, color=colors[i % len(colors)], label=f"Ch {i+1}")
            self.lines.append(line)
        self.ax.legend(loc="upper right", facecolor="#1e1e1e", edgecolor="#333", labelcolor="#aaa")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.refresh_counts()
        self.init_batch()

    def init_batch(self):
        self.batch_list = []
        candidates = [k for k in CONF["targets"] if self.user_counts.get(k, 0) < CONF["targets"][k]]
        if candidates:
            self.batch_list = candidates
            random.shuffle(self.batch_list)
            self.batch_list = self.batch_list[: self.batch_size]
            self.batch_ptr = 0
            self.select_current_batch_item()

    def select_current_batch_item(self):
        if not self.batch_list: return
        key = self.batch_list[self.batch_ptr]
        vals = self.combo["values"]
        for idx, v in enumerate(vals):
            if key in v:
                self.combo.current(idx)
                self.on_word_select(None)
                break

    def advance_batch(self):
        if not self.batch_list: return
        curr_key = self.batch_list[self.batch_ptr]
        if self.user_counts.get(curr_key, 0) >= CONF["targets"][curr_key]:
            self.swap_completed_item(self.batch_ptr)
        else:
            self.batch_ptr = (self.batch_ptr + 1) % len(self.batch_list)
        self.select_current_batch_item()

    def swap_completed_item(self, index):
        current_batch_set = set(self.batch_list)
        all_candidates = [k for k in CONF["targets"] if k not in current_batch_set and self.user_counts.get(k, 0) < CONF["targets"][k]]
        if all_candidates:
            self.batch_list[index] = random.choice(all_candidates)
        else:
            self.batch_list.pop(index)
            if self.batch_ptr >= len(self.batch_list):
                self.batch_ptr = 0

    def on_user_change(self, event=None):
        self.refresh_counts()
        self.init_batch()
        self.btn_redo.config(state=tk.DISABLED, bg="#333") # Reset redo on user change

    def refresh_counts(self):
        user_path = os.path.join(CONF["root_dir"], self.user_var.get())
        self.user_counts = {k: 0 for k in CONF["targets"]}
        if os.path.exists(user_path):
            for root, dirs, files in os.walk(user_path):
                if "manifest.json" in files:
                    try:
                        with open(os.path.join(root, "manifest.json"), "r") as f:
                            for item in json.load(f):
                                lbl = item.get("label", "")
                                if lbl in self.user_counts: self.user_counts[lbl] += 1
                    except: pass
        
        items = []
        for k in CONF["targets"]:
            c = self.user_counts.get(k, 0)
            t = CONF["targets"][k]
            mark = "✔" if c >= t else ""
            items.append(f"{k[:90]}... {mark} [{c}/{t}]" if len(k) > 90 else f"{k} {mark} [{c}/{t}]")
        self.combo["values"] = items

    def on_word_select(self, event):
        idx = self.combo.current()
        if idx < 0: return
        val = self.combo.get()
        clean_text = val.rsplit(" [", 1)[0].rsplit(" ✔", 1)[0].strip()
        full_key = next((k for k in CONF["targets"] if k.startswith(clean_text)), clean_text)
        c = self.user_counts.get(full_key, 0)
        t = CONF["targets"].get(full_key, 15)
        self.count_lbl.config(text=f"{c} / {t}", fg="#00FF00" if c >= t else "#00FFFF")

    def start_rec(self):
        self.worker.start_recording()
        self.btn_start.config(state=tk.DISABLED, bg="#330000")
        self.btn_stop.config(state=tk.NORMAL, bg="#FF4444", fg="white")
        self.btn_redo.config(state=tk.DISABLED, bg="#333") # Disable redo during rec
        self.combo.config(state="disabled")

    def stop_rec(self):
        data = self.worker.stop_recording()
        if data:
            user = self.user_var.get()
            if self.batch_list and self.batch_list[self.batch_ptr] in CONF["targets"]:
                label_text = self.batch_list[self.batch_ptr]
            else:
                val = self.combo.get()
                clean_text = val.rsplit(" [", 1)[0].rsplit(" ✔", 1)[0].strip()
                label_text = next((k for k in CONF["targets"] if k.startswith(clean_text)), clean_text)

            sent_id = SENTENCE_TO_ID.get(label_text, 0)
            session_dir = DataManager.get_session_dir(user)
            fname = f"s{user[-2:] if len(user)>1 else '00'}_sent{sent_id:03d}_{datetime.now().strftime('%H%M%S')}.pkl"
            save_path = os.path.join(session_dir, fname)

            arr = np.array(data, dtype=np.float64)
            t = arr[:, 0]
            x_raw = arr[:, 1:1+CONF["channels"]].astype(np.float32)
            x_proc = self.processor.process(x_raw)

            payload = {
                "t": t, "raw_uV": x_raw, "proc_uV": x_proc,
                "fs": CONF["sample_rate"], "label": label_text,
                "sentence_id": sent_id, "timestamp": datetime.now().isoformat(),
                "processing_conf": CONF
            }
            with open(save_path, "wb") as f:
                pickle.dump(payload, f)

            man_path = os.path.join(session_dir, "manifest.json")
            manifest = []
            if os.path.exists(man_path):
                try:
                    with open(man_path, "r") as f: manifest = json.load(f)
                except: pass
            manifest.append({"filename": fname, "label": label_text, "id": sent_id})
            with open(man_path, "w") as f:
                json.dump(manifest, f, indent=2)

            DataManager.update_splits(user, fname, label_text)
            
            # --- Enable Redo ---
            self.last_saved_file = fname
            self.last_saved_user = user
            self.btn_redo.config(state=tk.NORMAL, bg="#444", fg="#FFD700")
            # -------------------

            self.refresh_counts()
            self.advance_batch()

        self.btn_start.config(state=tk.NORMAL, bg="#333")
        self.btn_stop.config(state=tk.DISABLED, bg="#333", fg="#aaa")
        self.combo.config(state="readonly")
        self.time_lbl.config(text="0.00 s", fg="#555")
        self.hz_lbl.config(text="0 Hz", fg="#333")

    def redo_last(self):
        """Deletes the last recording and moves the batch pointer back"""
        if self.last_saved_file and self.last_saved_user:
            DataManager.delete_entry(self.last_saved_user, self.last_saved_file)
            
            self.last_saved_file = None
            self.last_saved_user = None
            self.btn_redo.config(state=tk.DISABLED, bg="#333")
            
            self.refresh_counts()

            if self.batch_list:
                self.batch_ptr = (self.batch_ptr - 1) % len(self.batch_list)
                self.select_current_batch_item()

    def update_timer(self):
        if self.worker.is_recording:
            _, d, hz = self.worker.get_stats()
            self.time_lbl.config(text=f"{d:.2f} s", fg="#FF4444")
            color = "#00FF00" if hz > 900 else "#FF0000"
            self.hz_lbl.config(text=f"{int(hz)} Hz", fg=color)
        self.root.after(100, self.update_timer)

    def update_plot(self, frame):
        if not self.worker.running: return self.lines

        with self.worker.lock:
            snaps = [np.array(b, dtype=np.float32) for b in self.worker.plot_buffers]

        X = np.stack(snaps, axis=1)
        try: Y = self.processor.process(X)
        except: Y = X

        for ch, line in enumerate(self.lines):
            y = Y[:, ch]
            line.set_data(self.x[:len(y)], y)

        if self.autoscale_var.get() and (frame % 5 == 0):
            try:
                vals = Y.flatten()
                if vals.size > 0:
                    lo, hi = np.percentile(vals, [1, 99])
                    span = max(hi - lo, 10.0) 
                    mid = (hi + lo) / 2.0
                    cur_lo, cur_hi = self.ax.get_ylim()
                    tgt_lo, tgt_hi = mid - 0.6 * span, mid + 0.6 * span
                    new_lo = cur_lo * 0.8 + tgt_lo * 0.2
                    new_hi = cur_hi * 0.8 + tgt_hi * 0.2
                    self.ax.set_ylim(new_lo, new_hi)
            except: pass

        return self.lines

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()