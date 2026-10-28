"""Windows keep-awake, battery check, and ntfy.sh notifications (the first two are no-ops on Linux).

Nothing in here may crash the run: every call swallows its own errors.
"""

from __future__ import annotations

import ctypes
import os
import secrets
import sys
import urllib.request
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def keep_awake() -> bool:
    """Ask Windows not to sleep while this process runs. Returns True on success."""
    if sys.platform != "win32":
        return False
    try:
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        return ctypes.windll.kernel32.SetThreadExecutionState(flags) != 0
    except Exception:
        return False


def allow_sleep() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            pass


class _PowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte),
        ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte),
        ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def on_battery() -> bool:
    """True only if Windows positively reports the AC line as offline."""
    if sys.platform != "win32":
        return False
    try:
        st = _PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            return False
        return st.ACLineStatus == 0
    except Exception:
        return False


def ntfy_topic(path: Path) -> str:
    """The run's ntfy topic: $PSIDUCK_NTFY_TOPIC, else the saved one, else a new random one."""
    try:
        env = os.environ.get("PSIDUCK_NTFY_TOPIC", "").strip()
        if env:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(env + "\n", encoding="utf-8")
            return env
        if path.exists():
            t = path.read_text(encoding="utf-8").strip()
            if t:
                return t
        t = "psiduck-" + secrets.token_hex(8)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(t + "\n", encoding="utf-8")
        return t
    except Exception:
        return "psiduck-" + secrets.token_hex(8)


def notify(topic: str, message: str, title: str = "Psiduck eval", priority: str = "default") -> bool:
    """POST to https://ntfy.sh/<topic>. Never raises."""
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title.encode("ascii", "replace").decode(), "Priority": priority},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return 200 <= r.status < 300
    except Exception:
        return False
