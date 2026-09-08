import sys
import socket
import os
import subprocess
import webbrowser
import time
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
                        print(f"[*] Port {port} occupied by PID {pid}. Freeing port...")
                        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                        time.sleep(0.5)
        except Exception:
            pass
    else:
        try:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        except Exception:
            pass

if __name__ == "__main__":
    free_port(8000)
    print("==================================================")
    print(" [>] Starting Mi:RAG Engine Studio...")
    print(" [*] Web Interface: http://localhost:8000")
    print("==================================================")
    
    def _open_when_ready():
        import urllib.request
        health_url = "http://127.0.0.1:8000/api/system/health"
        t0 = time.time()
        while time.time() - t0 < 20.0:
            try:
                with urllib.request.urlopen(health_url, timeout=0.8) as response:
                    if response.status == 200:
                        webbrowser.open("http://localhost:8000")
                        return
            except Exception:
                pass
            time.sleep(0.2)
        try:
            webbrowser.open("http://localhost:8000")
        except Exception:
            pass

    import threading
    ready_thread = threading.Thread(target=_open_when_ready, daemon=True)
    ready_thread.start()

    uvicorn.run(app, host="127.0.0.1", port=8000)
