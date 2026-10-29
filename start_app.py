"""
start_app.py
Turnkey launcher for PathoVision AI:
- Boots the FastAPI backend server (hosting APIs & interactive frontend)
- Automatically opens the default browser to http://127.0.0.1:8000
- Handles clean shutdown on Ctrl+C
"""

import sys
import os
import time
import socket
import threading
import webbrowser
from pathlib import Path

# Add project root to sys.path
# Safe stdout/stderr wrapper to prevent macOS Errno 5 (EIO) on detached tty / background execution
class _SafeStream:
    def __init__(self, target):
        self.target = target
    def write(self, s):
        try:
            return self.target.write(s)
        except OSError:
            pass
    def flush(self):
        try:
            return self.target.flush()
        except OSError:
            pass
    def __getattr__(self, name):
        return getattr(self.target, name)

if not isinstance(sys.stdout, _SafeStream):
    sys.stdout = _SafeStream(sys.stdout)
if not isinstance(sys.stderr, _SafeStream):
    sys.stderr = _SafeStream(sys.stderr)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def open_browser():
    """Polls until the server is listening, then launches the browser."""
    for _ in range(25):
        if is_port_in_use(PORT, HOST):
            time.sleep(0.6)
            print(f"\n[+] PathoVision AI Server active at {URL}")
            print(f"[+] Opening web browser to {URL}...")
            webbrowser.open(URL)
            return
        time.sleep(0.4)
    print(f"\n[!] Server taking longer than expected. Please open {URL} manually.")


def main():
    print("=" * 70)
    print("   PathoVision AI - Digital Pathology Cancer & Subtype Suite")
    print("   Liquid Crystal Contrast Dashboard & Automated Clinical Reporting")
    print("=" * 70)

    if is_port_in_use(PORT, HOST):
        print(f"\n[*] PathoVision server is already running on {URL}!")
        print(f"[*] Opening browser now...")
        webbrowser.open(URL)
        print("\nPress Enter or Ctrl+C to exit launcher.")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass
        return

    # Start browser opener in background daemon thread
    threading.Thread(target=open_browser, daemon=True).start()

    print(f"[*] Starting backend and web UI on {URL}...")
    print("[*] Press Ctrl+C in this console to stop the server.\n")

    try:
        import uvicorn
        uvicorn.run(
            "src.app.web_app:app",
            host=HOST,
            port=PORT,
            reload=False,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n[*] Shutting down PathoVision AI server. Done.")
    except Exception as e:
        print(f"\n[!] Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
