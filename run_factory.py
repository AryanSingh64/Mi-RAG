import sys
import socket
import os
import subprocess
import time
import threading
from pathlib import Path

# Enable UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from server.main import app

def free_port(port=8000):
    """Automatically find and terminate any stale process blocking the port on Windows/Linux."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    pid = parts[-1]
                    if pid and pid != "0" and int(pid) != os.getpid():
                        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                        time.sleep(0.5)
        except Exception:
            pass
    else:
        try:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        except Exception:
            pass

def main():
    free_port(8000)
    print("==================================================")
    print(" [⚡] Starting Mi:RAG Desktop Application...")
    print(" [*] Local Server: http://127.0.0.1:8000")
    print("==================================================")

    # 1. Start FastAPI server in background thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning"),
        daemon=True
    )
    server_thread.start()

    # 2. Wait until server responds
    import urllib.request
    health_url = "http://127.0.0.1:8000/api/system/health"
    t0 = time.time()
    while time.time() - t0 < 15.0:
        try:
            with urllib.request.urlopen(health_url, timeout=0.8) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        time.sleep(0.15)

    # 3. Launch native standalone desktop window via pywebview (Edge WebView2)
    try:
        import webview
        window = webview.create_window(
            title="Mi:RAG Assistant",
            url="http://127.0.0.1:8000",
            width=1280,
            height=820,
            min_size=(960, 640),
            background_color="#090c15"
        )
        webview.start(private_mode=False)
    except Exception as e:
        print(f"[!] pywebview desktop window error: {e}. Falling back to browser...")
        import webbrowser
        webbrowser.open("http://127.0.0.1:8000")
        server_thread.join()

if __name__ == "__main__":
    main()
