"""Visualize recorded drone input overlaid on video for alignment verification."""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger

def draw_rounded_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = -1,
    radius: int = 10,
) -> None:
    """Draw a rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if thickness == -1:
        # Filled rounded rectangle
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(frame, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(frame, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(frame, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(frame, (x2 - r, y2 - r), r, color, -1)
    else:
        # Border only
        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_stick(
    frame: np.ndarray,
    center: tuple[int, int],
    x_value: int,
    y_value: int,
    radius: int = 50,
    label: str = "",
    accent_color: tuple[int, int, int] = (0, 255, 200),
) -> None:
    """Draw a modern stick position indicator with glow effects."""
    # Normalize values from 0-2048 to -1 to 1
    x_norm = (x_value - 1024) / 1024
    y_norm = (y_value - 1024) / 1024

    # Draw outer ring with gradient effect (multiple circles for depth)
    cv2.circle(frame, center, radius + 2, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.circle(frame, center, radius, (70, 70, 70), 1, cv2.LINE_AA)

    # Draw subtle grid lines
    for i in [-0.5, 0, 0.5]:
        # Horizontal
        pt1 = (int(center[0] - radius * 0.8), int(center[1] + i * radius * 0.8))
        pt2 = (int(center[0] + radius * 0.8), int(center[1] + i * radius * 0.8))
        cv2.line(frame, pt1, pt2, (45, 45, 45), 1, cv2.LINE_AA)
        # Vertical
        pt1 = (int(center[0] + i * radius * 0.8), int(center[1] - radius * 0.8))
        pt2 = (int(center[0] + i * radius * 0.8), int(center[1] + radius * 0.8))
        cv2.line(frame, pt1, pt2, (45, 45, 45), 1, cv2.LINE_AA)

    # Calculate stick position
    stick_x = int(center[0] + x_norm * (radius - 8))
    stick_y = int(center[1] - y_norm * (radius - 8))

    # Draw movement trail (line from center to stick)
    cv2.line(frame, center, (stick_x, stick_y), (accent_color[0] // 3, accent_color[1] // 3, accent_color[2] // 3), 2, cv2.LINE_AA)

    # Draw glow effect for stick
    for glow_radius, alpha in [(14, 0.2), (11, 0.4), (8, 0.7)]:
        glow_color = (int(accent_color[0] * alpha), int(accent_color[1] * alpha), int(accent_color[2] * alpha))
        cv2.circle(frame, (stick_x, stick_y), glow_radius, glow_color, -1, cv2.LINE_AA)

    # Draw stick knob
    cv2.circle(frame, (stick_x, stick_y), 6, accent_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (stick_x, stick_y), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw label below
    if label:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = center[0] - text_size[0] // 2
        cv2.putText(
            frame,
            label,
            (text_x, center[1] + radius + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )


def draw_channel_bar(
    frame: np.ndarray,
    pos: tuple[int, int],
    value: int,
    label: str,
    bar_width: int = 80,
    bar_height: int = 8,
    accent_color: tuple[int, int, int] = (0, 255, 200),
) -> None:
    """Draw a single channel value with a visual bar indicator."""
    x, y = pos

    # Normalize value (0-2048 to 0-1)
    norm_val = value / 2048.0
    fill_width = int(bar_width * norm_val)

    # Draw label
    cv2.putText(
        frame,
        label,
        (x, y + bar_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )

    # Draw bar background
    bar_x = x + 18
    draw_rounded_rect(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), (35, 35, 35), -1, 3)

    # Draw filled portion with glow
    if fill_width > 2:
        glow_color = (accent_color[0] // 3, accent_color[1] // 3, accent_color[2] // 3)
        draw_rounded_rect(frame, (bar_x, y - 1), (bar_x + fill_width, y + bar_height + 1), glow_color, -1, 3)
        draw_rounded_rect(frame, (bar_x, y), (bar_x + fill_width, y + bar_height), accent_color, -1, 3)

    # Draw value text
    val_text = f"{value:4d}"
    cv2.putText(
        frame,
        val_text,
        (bar_x + bar_width + 6, y + bar_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    roll: int,
    pitch: int,
    throttle: int,
    yaw: int,
    time_sec: float,
    frame_num: int,
) -> np.ndarray:
    """Draw the complete input overlay on a frame - bottom-right HUD style."""
    h, w = frame.shape[:2]

    # Overlay dimensions and position (bottom-right corner, massive)
    overlay_w = 620
    overlay_h = 420
    margin = 30
    overlay_x = w - overlay_w - margin
    overlay_y = h - overlay_h - margin

    # Semi-transparent overlay background with rounded corners
    overlay = frame.copy()
    draw_rounded_rect(
        overlay,
        (overlay_x, overlay_y),
        (overlay_x + overlay_w, overlay_y + overlay_h),
        (20, 20, 20),
        -1,
        22,
    )
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # Draw border glow
    draw_rounded_rect(
        frame,
        (overlay_x, overlay_y),
        (overlay_x + overlay_w, overlay_y + overlay_h),
        (50, 50, 50),
        2,
        22,
    )

    # Accent colors (cyan/teal theme)
    accent_left = (200, 180, 0)    # Cyan-ish for left stick
    accent_right = (0, 255, 200)   # Teal for right stick

    # Stick positions - side by side in the panel (massive radius)
    stick_radius = 100
    left_center = (overlay_x + 150, overlay_y + 145)
    right_center = (overlay_x + overlay_w - 150, overlay_y + 145)

    # Draw sticks (Mode 2 layout)
    draw_stick(frame, left_center, yaw, throttle, radius=stick_radius, accent_color=accent_left)
    draw_stick(frame, right_center, roll, pitch, radius=stick_radius, accent_color=accent_right)

    # Channel bars below sticks (massive bars)
    bar_y_start = overlay_y + 290
    bar_spacing = 32
    bar_x = overlay_x + 25

    draw_channel_bar(frame, (bar_x, bar_y_start), throttle, "T", bar_width=175, bar_height=18, accent_color=accent_left)
    draw_channel_bar(frame, (bar_x, bar_y_start + bar_spacing), yaw, "Y", bar_width=175, bar_height=18, accent_color=accent_left)
    draw_channel_bar(frame, (bar_x + 295, bar_y_start), roll, "R", bar_width=175, bar_height=18, accent_color=accent_right)
    draw_channel_bar(frame, (bar_x + 295, bar_y_start + bar_spacing), pitch, "P", bar_width=175, bar_height=18, accent_color=accent_right)

    # Time display - larger in corner
    time_text = f"{time_sec:.2f}s"
    cv2.putText(
        frame,
        time_text,
        (overlay_x + overlay_w - 100, overlay_y + overlay_h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    return frame


def find_sample_for_time(df: pd.DataFrame, time_sec: float) -> pd.Series | None:
    """Find the input sample closest to the given video time."""
    if df.empty:
        return None

    # Find closest sample by time
    idx = (df["time"] - time_sec).abs().idxmin()
    return df.loc[idx]


def visualize(
    video_path: Path,
    csv_path: Path | None = None,
    output_path: Path | None = None,
    playback: bool = True,
    scale: float = 1.0,
    crf: int = 23,
) -> None:
    """Visualize input data overlaid on video.

    Args:
        video_path: Path to the video file (.mkv)
        csv_path: Path to CSV file (defaults to video_path with .csv extension)
        output_path: If provided, render to this file instead of playing back
        playback: If True and no output_path, play video with overlay in a window
        scale: Scale factor for display (e.g., 0.5 for half size)
        crf: H.264 quality (0-51, lower=better quality/larger file). 18=visually lossless, 23=default, 28=smaller
    """
    if csv_path is None:
        csv_path = video_path.with_suffix(".csv")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load input data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} input samples from {csv_path.name}")
    logger.info(f"Time range: {df['time'].min():.3f}s - {df['time'].max():.3f}s")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")

    # Setup output writer if rendering (use FFmpeg for proper H.264 compression)
    ffmpeg_proc = None
    if output_path:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        logger.info(f"Rendering to: {output_path} (CRF={crf})")

    frame_num = 0
    paused = False

    try:
        while True:
            if not paused or ffmpeg_proc:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate video time
                time_sec = frame_num / fps

                # Find corresponding input sample
                sample = find_sample_for_time(df, time_sec)

                if sample is not None:
                    frame = draw_overlay(
                        frame,
                        roll=int(sample["roll"]),
                        pitch=int(sample["pitch"]),
                        throttle=int(sample["throttle"]),
                        yaw=int(sample["yaw"]),
                        time_sec=time_sec,
                        frame_num=frame_num,
                    )

                # Scale frame for display
                if scale != 1.0 and not ffmpeg_proc:
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                if ffmpeg_proc:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                    if frame_num % 60 == 0:
                        progress = (frame_num / frame_count) * 100
                        logger.info(f"Progress: {progress:.1f}%")
                elif playback:
                    cv2.imshow("Drone Input Visualization", frame)

                frame_num += 1

            if playback and not ffmpeg_proc:
                # Handle keyboard input
                key = cv2.waitKey(1 if not paused else 0) & 0xFF

                if key == ord("q") or key == 27:  # q or ESC
                    break
                elif key == ord(" "):  # Space to pause/resume
                    paused = not paused
                elif key == ord("d") or key == 83:  # d or right arrow - forward 1 frame
                    paused = True
                    # Frame already advanced, just continue
                elif key == ord("a") or key == 81:  # a or left arrow - back 1 frame
                    paused = True
                    frame_num = max(0, frame_num - 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                elif key == ord("w") or key == 82:  # w or up - forward 1 second
                    frame_num = min(frame_count - 1, frame_num + int(fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                elif key == ord("s") or key == 84:  # s or down - back 1 second
                    frame_num = max(0, frame_num - int(fps) - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    finally:
        cap.release()
        if ffmpeg_proc:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            logger.info(f"Rendered {frame_num} frames to {output_path}")
        if playback and not ffmpeg_proc:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize drone input aligned with video recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (during playback):
  Space     Pause/Resume
  A / Left  Back 1 frame
  D / Right Forward 1 frame
  W / Up    Forward 1 second
  S / Down  Back 1 second
  Q / ESC   Quit

Examples:
  # Play back with overlay
  python -m capture.visualize captures/recording.mkv

  # Render to MP4 file
  python -m capture.visualize captures/recording.mkv -o output.mp4
""",
    )
    parser.add_argument("video", type=Path, help="Path to video file (.mkv)")
    parser.add_argument(
        "-c", "--csv", type=Path, help="Path to CSV file (default: same as video with .csv)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Render to output file instead of playback"
    )
    parser.add_argument(
        "-s", "--scale", type=float, default=1.0, help="Scale factor for display (default: 1.0)"
    )
    parser.add_argument(
        "--crf", type=int, default=23,
        help="H.264 quality for output (0-51, lower=better). 18=near-lossless, 23=default, 28=smaller file"
    )

    args = parser.parse_args()

    visualize(
        video_path=args.video,
        csv_path=args.csv,
        output_path=args.output,
        scale=args.scale,
        crf=args.crf,
    )


if __name__ == "__main__":
    main()
