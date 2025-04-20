import socket
import threading
import av
import numpy as np
import cv2
from flask import Flask, Response, request
import time
import torch
import os

# === YOLOv5 Model Setup ===
base = os.path.dirname(__file__)
repo = os.path.join(base, 'yolov5')
weights = os.path.join(base, 'yolov5s.pt')
if not os.path.isdir(repo) or not os.path.isfile(weights):
    raise FileNotFoundError("Missing yolov5/ or yolov5s.pt")
os.environ['YOLOv5_SKIP_UPDATE'] = '1'
model = torch.hub.load(repo, 'custom', path=weights, source='local')

TARGET_FPS = 30
interval = 1.0 / TARGET_FPS

# === SETUP ===
Tello_IP = '192.168.10.1'
COMMAND_PORT = 8889
VIDEO_PORT = 11111
LOCAL_COMMAND_PORT = 9000

# === UDP COMMAND SOCKET ===
command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
command_sock.bind(('', LOCAL_COMMAND_PORT))
command_sock.settimeout(1)

# === GLOBALS ===
frame_lock = threading.Lock()
latest_frame = None

processed_frame = None
processed_frame_lock = threading.Lock()

autonomous_tracking = False
autonomous_lock = threading.Lock()

command_state_lock = threading.Lock()
last_command_time = 0.0
current_delay = 0.0
CMD_DELAY = {
    'cw': 1.0,
    'ccw': 1.0,
    'forward': 2.0,
}

def send_command(cmd):
    with command_state_lock:
        print(f"[SEND] {cmd}")
        try:
            command_sock.sendto(cmd.encode('utf-8'), (Tello_IP, COMMAND_PORT))
            response, _ = command_sock.recvfrom(1024)
            print(f"[RESPONSE] {response.decode()}")
        except socket.timeout:
            print(f"[TIMEOUT] No response for: {cmd}")
        except Exception as e:
            print(f"[ERROR] Command '{cmd}' failed: {e}")

# === START VIDEO STREAM ===
try:
    send_command("command")
    send_command("streamon")
except Exception as e:
    print(f"[WARN] Drone not responding: {e}")

# === FLASK APP ===
app = Flask(__name__)
video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_socket.bind(('', VIDEO_PORT))

def receive_video():
    global latest_frame
    container = av.open(video_socket.makefile('rb'))
    for packet in container.demux():
        for frame in packet.decode():
            with frame_lock:
                latest_frame = frame.to_ndarray(format='bgr24')

def process_frames():
    global latest_frame, processed_frame, model, autonomous_tracking
    last_detect = 0.0
    FPS_DETECT = 5
    display_w = 480

    while True:
        current_time = time.time()
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()
        
        H, W = frame.shape[:2]
        small = cv2.resize(frame, (display_w, int(H * display_w / W)))
        results = model(small)
        dets = results.xyxy[0].cpu().numpy()

        weapons = []
        for x1, y1, x2, y2, conf, cls in dets:
            if int(cls) in [0, 1, 2] and conf > 0.5:
                fx, fy = W / display_w, H / small.shape[0]
                x1 = x1 * fx
                y1 = y1 * fy
                x2 = x2 * fx
                y2 = y2 * fy
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                weapons.append((x1, y1, x2, y2, conf, cls))

        with processed_frame_lock:
            processed_frame = frame.copy()

        with autonomous_lock:
            if autonomous_tracking and (current_time - last_detect >= 1.0 / FPS_DETECT):
                last_detect = current_time
                if weapons:
                    best = max(weapons, key=lambda x: x[4])
                    x1, y1, x2, y2, conf, cls = best
                    cx = (x1 + x2) / 2
                    bw = x2 - x1
                    off = (cx / W) - 0.5
                    center_tol = 0.10
                    size_thresh = 0.20

                    if abs(off) > center_tol:
                        cmd = 'cw 15' if off > 0 else 'ccw 15'
                    elif (bw / W) < size_thresh:
                        cmd = 'forward 30'
                    else:
                        cmd = None
                else:
                    cmd = 'cw 30'

                if cmd:
                    with command_state_lock:
                        if current_time - last_command_time >= current_delay:
                            key = cmd.split()[0]
                            send_command(cmd)
                            last_command_time = current_time
                            current_delay = CMD_DELAY.get(key, 1.0)
        time.sleep(0.01)

def generate_mjpeg():
    while True:
        start = time.time()
        with processed_frame_lock:
            p_frame = processed_frame.copy() if processed_frame is not None else None
        if p_frame is None:
            with frame_lock:
                p_frame = latest_frame.copy() if latest_frame is not None else None
        if p_frame is not None:
            frame = cv2.resize(p_frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
        elapsed = time.time() - start
        time.sleep(max(0, interval - elapsed))

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tello Drone UI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root {
                --accent: #4f46e5; --bg: #0f172a; --btn-bg: #1e293b;
                --btn-hover: #334155; --btn-border: #475569; --text: #f8fafc;
                --red: #ef4444; --blue: #3b82f6; --green: #10b981;
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            body {
                font-family: 'Inter', sans-serif;
                background: var(--bg);
                color: var(--text);
                height: 100vh;
                display: flex;
                flex-direction: column;
                padding: 10px;
                overflow: hidden;
            }
            h1 {
                text-align: center;
                margin: 5px 0;
                font-size: 1.8rem;
            }
            .main-grid {
                display: grid;
                grid-template-rows: 1fr auto;
                height: 100%;
                gap: 10px;
            }
            .video-container {
                background: black;
                border-radius: 8px;
                overflow: hidden;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .video-container img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            .control-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 8px;
            }
            .control-grid button {
                width: 100%;
                height: 100%;
                min-height: 60px;
                font-size: 1.2rem;
                padding: 10px;
            }
            .double-width {
                grid-column: span 2;
            }
            .takeoff {
                background: var(--green);
                border-color: var(--green);
            }
            .land {
                background: var(--red);
                border-color: var(--red);
            }
            .emergency {
                background: #b91c1c;
                border-color: #b91c1c;
                animation: pulse 1s infinite;
            }
            button {
                background: var(--btn-bg);
                color: var(--text);
                border: 2px solid var(--btn-border);
                border-radius: 8px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.2s;
            }
            button:hover {
                background: var(--btn-hover);
                transform: translateY(-2px);
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
        </style>
    </head>
    <body>
        <div class="main-grid">
            <div class="video-container">
                <img src="/video_feed">
            </div>
            
            <div class="control-grid">
                <!-- Movement Controls -->
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="forward 50">
                    <button>↑ Forward</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="left 50">
                    <button>← Left</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="right 50">
                    <button>→ Right</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="back 50">
                    <button>↓ Back</button>
                </form>
                
                <!-- Altitude Controls -->
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="up 50">
                    <button>🔼 Up</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="down 50">
                    <button>🔽 Down</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="ccw 45">
                    <button>⟲ Rotate Left</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="cw 45">
                    <button>⟳ Rotate Right</button>
                </form>
                
                <!-- Action Buttons -->
                <form action="/takeoff" method="post">
                    <button class="takeoff double-width">🚀 Takeoff</button>
                </form>
                <form action="/land" method="post">
                    <button class="land double-width">🛬 Land</button>
                </form>
                
                <!-- Special Functions -->
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="flip l">
                    <button>↩️ Flip Left</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="flip r">
                    <button>↪️ Flip Right</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="flip f">
                    <button>⤴️ Flip Forward</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="flip b">
                    <button>⤵️ Flip Back</button>
                </form>
                
                <!-- Utility Buttons -->
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="speed">
                    <button>🏎️ Speed</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="battery?">
                    <button>🔋 Battery</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="emergency">
                    <button class="emergency">🛑 Emergency Stop</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="command">
                    <button>Reconnect</button>
                </form>
                <form action="/toggle_tracking" method="post">
                    <button class="double-width" type="submit">Toggle Auto Tracking</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/takeoff', methods=['POST'])
def takeoff():
    send_command("takeoff")
    return '', 204

@app.route('/land', methods=['POST'])
def land():
    send_command("land")
    return '', 204

@app.route('/toggle_tracking', methods=['POST'])
def toggle_tracking():
    global autonomous_tracking
    with autonomous_lock:
        autonomous_tracking = not autonomous_tracking
    return '', 204

@app.route('/command', methods=['POST'])
def handle_command():
    cmd = request.form.get('cmd')
    if cmd:
        send_command(cmd)
    return '', 204

# === MAIN ===
if __name__ == '__main__':
    threading.Thread(target=receive_video, daemon=True).start()
    threading.Thread(target=process_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5005, debug=False)