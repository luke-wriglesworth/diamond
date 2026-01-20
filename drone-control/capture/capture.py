"""Controller capture synced to OBS recording via WebSocket."""

import time
from dataclasses import dataclass
from pathlib import Path

import obsws_python as obs
import pandas as pd
from loguru import logger

from evdev import ecodes

from controller.linux import ControllerPassthrough, ControllerState


def parse_timecode(timecode: str) -> float:
    """Parse OBS timecode (HH:MM:SS.mmm) to seconds."""
    parts = timecode.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


@dataclass
class CaptureConfig:
    controller_hz: int = 1000  # Passthrough rate (match controller USB rate)
    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = "lsSf1gjOGdIlJ7bN"


class Capture:
    """Captures controller input synced to OBS video recording.

    - Controller passthrough runs at 200Hz (responsive flying)
    - Samples saved at ~60Hz (one per OBS timecode change, synced to video)
    - Output: CSV file alongside OBS video

    Usage:
        Capture().run()  # Ctrl+C to stop
    """

    def __init__(self, config: CaptureConfig | None = None):
        self.config = config or CaptureConfig()
        self._controller_interval = 1.0 / self.config.controller_hz

    def run(self) -> None:
        """Run capture until Ctrl+C."""
        cfg = self.config
        req = obs.ReqClient(host=cfg.obs_host, port=cfg.obs_port, password=cfg.obs_password)
        events = obs.EventClient(host=cfg.obs_host, port=cfg.obs_port, password=cfg.obs_password)

        def on_button(code: int, value: int):
            """Handle button to control recording (press=start, release=stop)."""
            if code == ecodes.BTN_SOUTH:
                if value == 1:  # Press
                    req.start_record()
                    logger.info("Recording started (button)")
                else:  # Release
                    req.stop_record()
                    logger.info("Recording stopped (button)")

        controller = ControllerPassthrough(button_callback=on_button)

        # State
        recording = False
        samples: list[dict] = []
        last_timecode: str | None = None
        video_path: str | None = None
        last_state = ControllerState()
        record_start_time: float = 0.0

        def on_record_state_changed(data):
            nonlocal recording, samples, last_timecode, video_path, record_start_time

            if data.output_state == "OBS_WEBSOCKET_OUTPUT_STARTED":
                record_start_time = time.perf_counter()
                video_path = getattr(req.get_record_status(), "output_path", None)
                samples = []
                last_timecode = None
                recording = True
                logger.info("Recording started")

            elif data.output_state == "OBS_WEBSOCKET_OUTPUT_STOPPED":
                recording = False
                final_path = getattr(data, "output_path", None) or video_path

                if final_path and samples:
                    csv_path = Path(final_path).with_suffix(".csv")
                    df = pd.DataFrame(samples)
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved {len(samples)} samples to {csv_path.name}")

                samples = []
                video_path = None

        events.callback.register(on_record_state_changed)

        logger.info(f"Connected to OBS at {cfg.obs_host}:{cfg.obs_port}")
        logger.info("Waiting for recording... Ctrl+C to stop")

        try:
            with controller:
                logger.info(f"Controller: {controller.device_name}")
                logger.info(f"Passthrough: {cfg.controller_hz}Hz, Samples: ~60Hz (OBS timecode)")

                # Check if already recording
                status = req.get_record_status()
                if status.output_active:
                    video_path = getattr(status, "output_path", None)
                    recording = True
                    logger.info("Already recording")

                next_update = time.perf_counter()

                while True:
                    now = time.perf_counter()

                    if now >= next_update:
                        # Update controller at 200Hz (passthrough for flying)
                        last_state = controller.update()

                        # Save sample at 200Hz (same rate as controller)
                        if recording:
                            samples.append({
                                "time": now - record_start_time,
                                "roll": last_state.roll,
                                "pitch": last_state.pitch,
                                "throttle": last_state.throttle,
                                "yaw": last_state.yaw,
                            })

                        next_update += self._controller_interval
                        if next_update < now:
                            next_update = now + self._controller_interval
                    else:
                        time.sleep(0.0001)

        except KeyboardInterrupt:
            logger.info("Stopped")
