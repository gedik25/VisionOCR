"""
VisionOCR Web — Server Entry Point.
Runs Uvicorn server for the web interface.
"""

import sys
import os
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import socket

def find_free_port(start_port=8080):
    for port in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port

def main():
    default_port = find_free_port(8080)
    port = int(os.environ.get("PORT", default_port))
    host = os.environ.get("HOST", "127.0.0.1")

    print("\n" + "=" * 60)
    print("⚡ VisionOCR Web Stüdyosu Başlatılıyor...")
    print(f"🌐 Tarayıcınızda açın: http://{host}:{port}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "web.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
