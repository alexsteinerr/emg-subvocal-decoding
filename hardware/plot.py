import serial
import serial.tools.list_ports
import threading
import collections
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as signal

BAUD_RATE = 115200
SAMPLE_RATE = 250        # Hz
VISIBLE_SECONDS = 5      
NUM_CHANNELS = 5         
SCALE_FACTOR = 0.02235   # uV conversion

Y_LIMIT = 250.0           

# FILTER SETTINGS 
MAINS_FREQ = 50.0        
LOW_CUT = 0.5          
HIGH_CUT = 45.0          

WINDOW_SIZE = int(SAMPLE_RATE * VISIBLE_SECONDS)

data_buffers = [collections.deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(NUM_CHANNELS)]
is_running = True
lock = threading.Lock()

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "usbmodem" in p.device or "usbserial" in p.device or "COM" in p.device:
            return p.device
    return None

def read_serial_loop(port_name):
    global is_running
    print(f"Connecting to {port_name}...")
    
    try:
        with serial.Serial(port_name, BAUD_RATE, timeout=1) as ser:
            print("Connected! Streaming data...")
            ser.reset_input_buffer()
            
            while is_running:
                if ser.in_waiting:
                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if not line or "Ch" in line or "ADS" in line: continue
                        
                        parts = line.split(',')
                        if len(parts) >= NUM_CHANNELS:
                            with lock:
                                for i in range(NUM_CHANNELS):
                                    val = float(parts[i])
                                    uv_val = val * SCALE_FACTOR
                                    data_buffers[i].append(uv_val)
                    except ValueError:
                        pass
                else:
                    time.sleep(0.001)
    except Exception as e:
        print(f"Error: {e}")
        is_running = False

def apply_filter(data_array):
    # 1. Notch Filter
    b_notch, a_notch = signal.iirnotch(MAINS_FREQ, 30.0, SAMPLE_RATE)
    filtered = signal.filtfilt(b_notch, a_notch, data_array)
    
    # 2. Bandpass Filter (0.5 - 45 Hz)
    b_band, a_band = signal.butter(2, [LOW_CUT, HIGH_CUT], btype='bandpass', fs=SAMPLE_RATE)
    filtered = signal.filtfilt(b_band, a_band, filtered)
    
    return filtered

def start_plotting():
    plt.style.use('dark_background')
    fig, axes = plt.subplots(NUM_CHANNELS, 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title('ADS1299 EEG - Stable Axis')

    lines = []
    colors = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FFFFFF']
    x_data = np.linspace(0, VISIBLE_SECONDS, WINDOW_SIZE)

    for i, ax in enumerate(axes):
        line, = ax.plot(x_data, np.zeros(WINDOW_SIZE), lw=1, color=colors[i])
        lines.append(line)
        
        ax.set_ylabel(f'Ch {i+1} ($\mu$V)', fontsize=8)
        ax.set_ylim(-Y_LIMIT, Y_LIMIT)  
        
        ax.grid(True, color='#333333', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (Seconds)')

    def on_close(event):
        global is_running
        is_running = False

    fig.canvas.mpl_connect('close_event', on_close)

    def update(frame):
        if not is_running: return lines

        with lock:
            snapshots = [list(buf) for buf in data_buffers]

        for i, line in enumerate(lines):
            raw_data = np.array(snapshots[i])
            
            if len(raw_data) >= SAMPLE_RATE:
                try:
                    clean_data = apply_filter(raw_data)
                    line.set_ydata(clean_data)
                except Exception:
                    line.set_ydata(raw_data)
            else:
                line.set_ydata(raw_data)
                
        return lines

    ani = FuncAnimation(fig, update, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    target_port = find_arduino_port()
    if target_port:
        t = threading.Thread(target=read_serial_loop, args=(target_port,), daemon=True)
        t.start()
        start_plotting()
    else:
        print("No device found.")