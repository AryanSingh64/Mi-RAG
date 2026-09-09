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

def find_available_port(preferred=8000, max_tries=100):
    """Find preferred port if free, or dynamically bind to next available port without overriding."""
    for p in range(preferred, preferred + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('127.0.0.1', p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def main():
    port = find_available_port(8000)
    print("==================================================")
    print(f" [*] Local Server: http://127.0.0.1:{port}")
    print(f" [*] Opening browser tab to Training RAG Studio (http://127.0.0.1:{port})...")
    print("==================================================")

    # 1. Start FastAPI server in background thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning"),
        daemon=True
    )
    server_thread.start()

    # 2. Wait until server responds
    import urllib.request
    health_url = f"http://127.0.0.1:{port}/api/system/health"
    t0 = time.time()
    while time.time() - t0 < 15.0:
        try:
            with urllib.request.urlopen(health_url, timeout=0.8) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        time.sleep(0.15)

    # 3. Open browser tab for Training RAG Studio
    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{port}")

    # Keep server running until terminal is closed or Ctrl+C is pressed
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n [*] Stopping Local Server...")

if __name__ == "__main__":
    main()
