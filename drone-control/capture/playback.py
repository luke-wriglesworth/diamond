"""Play back recorded controller inputs through virtual device."""

import argparse
import time
from pathlib import Path

import evdev
import pandas as pd
from evdev import ecodes
from loguru import logger

from controller.linux import VirtualController, find_device, list_devices


class ButtonMonitor:
    """Monitor BTN_EAST for toggle control."""

    def __init__(self, device_path: str):
        self.dev = evdev.InputDevice(device_path)
        self.pressed = False

    def wait_for_press(self) -> None:
        """Block until BTN_EAST is pressed."""
        logger.info(f"Press BTN_EAST on {self.dev.name} to start playback...")
        for event in self.dev.read_loop():
            if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_EAST and event.value == 1:
                self.pressed = True
                return

    def check_stopped(self) -> bool:
        """Non-blocking check if BTN_EAST was pressed again (toggle off)."""
        while event := self.dev.read_one():
            if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_EAST and event.value == 1:
                self.pressed = False
                return True
        return False

    def close(self) -> None:
        self.dev.close()


class Playback:
    """Plays back recorded controller inputs from CSV.

    Usage:
        Playback("recording.csv").run()  # Ctrl+C to stop
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def run(self) -> None:
        """Play back the recording, triggered by BTN_EAST press."""
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} samples from {self.csv_path.name}")

        # Find physical controller for button trigger
        device_path = find_device("Joystick")
        if not device_path:
            raise RuntimeError(f"No controller found. Available: {list_devices()}")

        with VirtualController() as controller:
            logger.info("Virtual controller created")
            button = ButtonMonitor(device_path)

            try:
                while True:
                    button.wait_for_press()
                    logger.info("Starting playback...")

                    first_row = df.iloc[0]
                    stopped = False


                    start_time = time.perf_counter()
                    first_sample_time = first_row["time"]

                    for _, row in df.iterrows():
                        target_time = start_time + (row["time"] - first_sample_time)
                        while time.perf_counter() < target_time:
                            pass

                        # Check for toggle off (after timing wait to avoid drift)
                        if button.check_stopped():
                            logger.info("Playback stopped (toggle)")
                            stopped = True
                            break

                        # Write to virtual controller
                        controller.write(
                            roll=int(row["roll"]),
                            pitch=int(row["pitch"]),
                            throttle=int(row["throttle"]),
                            yaw=int(row["yaw"]),
                        )

                    if not stopped:
                        logger.info("Playback complete")

            except KeyboardInterrupt:
                logger.info("Exiting")
            finally:
                button.close()


def main():
    parser = argparse.ArgumentParser(description="Play back recorded controller inputs")
    parser.add_argument("csv", type=Path, help="Path to CSV file with recorded inputs")
    args = parser.parse_args()

    Playback(args.csv).run()


if __name__ == "__main__":
    main()
