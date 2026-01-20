"""evdev-based controller passthrough."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import evdev
from evdev import UInput, ecodes


@dataclass
class ControllerState:
    roll: int = 0
    pitch: int = 0
    throttle: int = 0
    yaw: int = 0


# Axis mapping: evdev code -> state attribute
AXIS_MAP = {
    ecodes.ABS_X: "roll",
    ecodes.ABS_Y: "pitch",
    ecodes.ABS_Z: "throttle",
    ecodes.ABS_RX: "yaw",
}

# Path to cached controller capabilities
CAPS_FILE = Path(__file__).parent / "caps.json"


def load_caps() -> list[tuple[int, evdev.AbsInfo]] | None:
    """Load cached capabilities from file."""
    if not CAPS_FILE.exists():
        return None
    data = json.loads(CAPS_FILE.read_text())
    return [
        (int(code), evdev.AbsInfo(**info))
        for code, info in data.items()
    ]


def save_caps(caps: list[tuple[int, evdev.AbsInfo]]) -> None:
    """Save capabilities to cache file."""
    data = {
        str(code): {
            "value": info.value,
            "min": info.min,
            "max": info.max,
            "fuzz": info.fuzz,
            "flat": info.flat,
            "resolution": info.resolution,
        }
        for code, info in caps
        if code in AXIS_MAP
    }
    CAPS_FILE.write_text(json.dumps(data, indent=2))


def get_caps() -> list[tuple[int, evdev.AbsInfo]]:
    """Get capabilities from cache or raise if not cached."""
    caps = load_caps()
    if caps is None:
        raise RuntimeError(
            f"No cached capabilities. Run with physical controller first to generate {CAPS_FILE}"
        )
    return caps


def list_devices() -> list[tuple[str, str]]:
    """List available input devices."""
    devices = []
    for path in sorted(Path("/dev/input").glob("event*")):
        try:
            dev = evdev.InputDevice(str(path))
            devices.append((str(path), dev.name))
            dev.close()
        except (PermissionError, OSError):
            continue
    return devices


def find_device(pattern: str = "Radiomaster") -> str | None:
    """Find device matching pattern."""
    for path, name in list_devices():
        if pattern.lower() in name.lower():
            return path
    return None


class ControllerPassthrough:
    '''
    Grabs physical controller and relays its events to virtual controller after recording
    '''

    def __init__(
        self,
        device: str | None = None,
        pattern: str = "Joystick",
        grab: bool = True,
        button_callback: Callable[[int, int], None] | None = None,
    ):
        self._device_path = device
        self._pattern = pattern
        self._grab = grab
        self._physical: evdev.InputDevice | None = None
        self._virtual: UInput | None = None
        self._state = ControllerState()
        self._button_callback = button_callback

    def __enter__(self):
        # Find device if not specified
        if not self._device_path:
            self._device_path = find_device(self._pattern)
            if not self._device_path:
                raise RuntimeError(f"No device matching '{self._pattern}'. Available: {list_devices()}")

        self._physical = evdev.InputDevice(self._device_path)
        caps = self._physical.capabilities(absinfo=True)

        # Initialize state from current physical values
        for code, absinfo in caps.get(ecodes.EV_ABS, []):
            if code in AXIS_MAP:
                setattr(self._state, AXIS_MAP[code], absinfo.value)

        # Save capabilities for playback and create virtual device
        axis_caps = list(caps[ecodes.EV_ABS])
        save_caps(axis_caps)
        self._virtual = UInput({ecodes.EV_ABS: axis_caps}, name="DroneML-Controller")
        if self._grab:
            self._physical.grab()
        self._sync()
        return self

    def __exit__(self, *_):
        if self._physical:
            try:
                self._physical.ungrab()
            except OSError:
                pass
            self._physical.close()
        if self._virtual:
            self._virtual.close()

    def _sync(self):
        """Sync state to virtual device."""
        for code, attr in AXIS_MAP.items():
            self._virtual.write(ecodes.EV_ABS, code, getattr(self._state, attr))
        self._virtual.syn()

    def update(self) -> ControllerState:
        """Read pending events and return current state."""
        while event := self._physical.read_one():
            if event.type == ecodes.EV_ABS and event.code in AXIS_MAP:
                setattr(self._state, AXIS_MAP[event.code], event.value)
                self._sync()
            elif event.type == ecodes.EV_KEY and self._button_callback:
                self._button_callback(event.code, event.value)
        return self._state

    @property
    def device_name(self) -> str | None:
        return self._physical.name if self._physical else None


class VirtualController:
    '''
    Fake controller that does playback only from recordings.
    '''

    def __init__(self, name: str = "DroneML-Controller"):
        self._name = name
        self._virtual: UInput | None = None
        self._state = ControllerState()

    def __enter__(self):
        caps = get_caps()
        self._virtual = UInput({ecodes.EV_ABS: caps}, name=self._name)

        # Initialize all axes to center position
        for code, absinfo in caps:
            if code in AXIS_MAP:
                center = (absinfo.min + absinfo.max) // 2
                setattr(self._state, AXIS_MAP[code], center)
        self._sync()
        return self

    def __exit__(self, *_):
        if self._virtual:
            self._virtual.close()

    def _sync(self):
        """Sync state to virtual device."""
        for code, attr in AXIS_MAP.items():
            self._virtual.write(ecodes.EV_ABS, code, getattr(self._state, attr))
        self._virtual.syn()

    def write(self, roll: int, pitch: int, throttle: int, yaw: int):
        """Write values to virtual controller."""
        self._state = ControllerState(roll, pitch, throttle, yaw)
        self._sync()
