import time
import threading
import serial
import serial.tools.list_ports
import pyautogui
import sys

import cast
import fishing
import minigame

# ── Serial config ─────────────────────────────────────────────────────────────
BAUD_RATE   = 115200
SERIAL_PORT = None   # Set to e.g. "COM3", or leave None to auto-detect

# ── Global stop flag — set by serial thread, checked everywhere ───────────────
stop_flag = threading.Event()

def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if any(kw in (p.description or "").upper()
               for kw in ["CP210", "CH340", "USB SERIAL", "UART", "WCHUSB"]):
            return p.device
    if ports:
        return ports[0].device
    return None

def send_status(ser, msg: str):
    try:
        ser.write((msg + "\n").encode("utf-8"))
    except Exception:
        pass

# ── Background thread: watches serial for STOP at any time ───────────────────
def serial_listener(ser):
    while not stop_flag.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line == "STOP":
                print("\n[Serial] STOP received — stopping script.")
                stop_flag.set()
                return
        except Exception:
            pass

# ── Interruptible sleep — returns early if stop_flag is set ──────────────────
def sleep(seconds):
    stop_flag.wait(timeout=seconds)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    port = SERIAL_PORT or find_esp32_port()
    if not port:
        print("ERROR: No serial port found. Is the ESP32 plugged in?")
        sys.exit(1)

    print(f"Connecting to ESP32 on {port} at {BAUD_RATE} baud...")
    ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()
    print("Connected. Waiting for START signal from button...")
    send_status(ser, "WAITING")

    # ── Wait for START (blocking, no thread yet) ──────────────────────────────
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "START":
            print("START received — autofisher armed.")
            break

    # ── Start serial listener thread ──────────────────────────────────────────
    listener = threading.Thread(target=serial_listener, args=(ser,), daemon=True)
    listener.start()

    # ── Countdown ─────────────────────────────────────────────────────────────
    for i in range(5, 0, -1):
        if stop_flag.is_set():
            break
        send_status(ser, f"STARTING:{i}")
        print(f"  Starting in {i}...")
        sleep(1)

    if stop_flag.is_set():
        send_status(ser, "STOPPED")
        ser.close()
        return

    send_status(ser, "ARMED")
    print("Armed. Fishing loop starting.")

    # ── Main fishing loop ─────────────────────────────────────────────────────
    try:
        while not stop_flag.is_set():

            print("Casting...")
            send_status(ser, "CASTING")
            cast.cast()
            if stop_flag.is_set(): break
            sleep(0.15)

            print("Waiting for fish...")
            send_status(ser, "WAITING_FISH")
            fishing.fish()          # this blocks until exclamation point seen
            if stop_flag.is_set(): break

            send_status(ser, "FISH_DETECTED")
            print("Fish detected! Hooking...")
            pyautogui.mouseDown()
            sleep(0.5)
            pyautogui.mouseUp()
            if stop_flag.is_set(): break

            sleep(2)
            if stop_flag.is_set(): break

            send_status(ser, "MINIGAME")
            print("Minigame starting...")
            minigame.minigame()     # exits on its own when fish is caught
            if stop_flag.is_set(): break

            pyautogui.mouseDown()
            sleep(0.01)
            pyautogui.mouseUp()

            send_status(ser, "CAUGHT")
            print("Fish caught!")
            sleep(3)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")
    finally:
        stop_flag.set()
        send_status(ser, "STOPPED")
        pyautogui.mouseUp()   # safety — make sure mouse isn't stuck held
        ser.close()
        print("Stopped. Serial closed.")

if __name__ == "__main__":
    main()