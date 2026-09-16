"""Drive the Hot Aisle admin TUI (ssh admin.hotaisle.app) from a script.

The TUI needs a real terminal, so this opens a pseudo-terminal, connects,
waits <seconds> for the first screen, sends the keys in order (2.5 s apart,
WAIT<n> pauses n seconds), hangs up, and prints a rough text rendering of
the last screen. The raw byte stream goes to raw.bin next to this file for
a second look (see docs/gpu/01-bringup/06-agent-guide.md for the recipes).

    python3 env/hotaisle/tui.py 8                      # the team page
    python3 env/hotaisle/tui.py 8 n WAIT4              # the provisioning list
    python3 env/hotaisle/tui.py 8 n WAIT5 ENTER WAIT8 y WAIT25   # provision the selected VM

keys: literal strings, or the tokens ESC, ENTER, UP, DOWN, LEFT, RIGHT, TAB, CTRLC, WAIT<seconds>
"""
import os, pty, sys, time, select, re, signal, struct, fcntl, termios
KEYS = {"ESC": "\x1b", "ENTER": "\r", "UP": "\x1b[A", "DOWN": "\x1b[B", "LEFT": "\x1b[D", "RIGHT": "\x1b[C", "TAB": "\t", "CTRLC": "\x03"}
secs = float(sys.argv[1]); script = sys.argv[2:]
pid, fd = pty.fork()
if pid == 0:
    os.environ["TERM"] = "xterm-256color"
    os.execvp("ssh", ["ssh", "-tt", "-o", "BatchMode=yes", "admin.hotaisle.app"])
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 160, 0, 0))
out = b""
def drain(t):
    global out
    end = time.time() + t
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.2)
        if r:
            try: out += os.read(fd, 65536)
            except OSError: return
drain(secs)
for k in script:
    if k.startswith("WAIT"): drain(float(k[4:])); continue
    os.write(fd, KEYS.get(k, k).encode()); drain(2.5)
try: os.kill(pid, signal.SIGHUP)
except Exception: pass
# rough screen reconstruction: strip ANSI, keep last frame after final clear
txt = out.decode("utf-8", "replace")
frames = re.split(r"\x1b\[2J|\x1b\[H\x1b\[2J", txt)
last = frames[-1] if len(frames) > 1 else txt
clean = re.sub(r"\x1b\][^\x07]*\x07|\x1b\[[0-9;?]*[A-Za-z]|\x1b[=>]|\x1b\([B0]|\r", "", last)
lines = [l.rstrip() for l in clean.split("\n")]
print("\n".join(l for l in lines if l.strip()))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw.bin"), "wb").write(out)
