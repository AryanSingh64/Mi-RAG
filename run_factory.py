import sys
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

if __name__ == "__main__":
    print("==================================================")
    print("🚀 Starting Autonomous RAG Factory Web Application...")
    print("🌐 Open in your browser: http://localhost:8000")
    print("==================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000)
