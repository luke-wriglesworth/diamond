#!/usr/bin/env python3
"""
Convert FPV drone simulator recordings to Diamond training format.

Usage:
    python scripts/convert_fpv_data.py --input_dir /path/to/recordings --output_dir /path/to/dataset

Expected input structure:
    recordings/
    ├── flight_001.mkv
    ├── flight_001.csv
    ├── flight_002.mkv
    ├── flight_002.csv
    └── ...

CSV format (1000Hz RC inputs):
    time,roll,pitch,throttle,yaw
    0.0007844,1022,1026,648,1024
    ...

Output:
    dataset/
    ├── train/
    │   ├── 000/000/0.pt
    │   ├── 000/000/1.pt
    │   └── ...
    ├── test/
    │   └── ...
    └── info.pt
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import Dataset
from data.episode import Episode


# Configuration
TARGET_FPS = 15
LOW_RES_SIZE = (30, 56)  # H, W - matches CSGO
FULL_RES_SIZE = (150, 280)  # H, W - for optional upsampler

# Action discretization
NUM_BINS = 8  # Per channel: 8^4 = 4096 total actions
PWM_MIN = 1000
PWM_MAX = 2000


def discretize_pwm(value: float, min_pwm: float = PWM_MIN, max_pwm: float = PWM_MAX, num_bins: int = NUM_BINS) -> int:
    """Discretize a PWM value into a bin index."""
    normalized = (value - min_pwm) / (max_pwm - min_pwm)
    return int(np.clip(normalized * num_bins, 0, num_bins - 1))


def encode_action(roll: float, pitch: float, throttle: float, yaw: float, bins: int = NUM_BINS) -> int:
    """Encode 4 RC channels into a single discrete action index."""
    r = discretize_pwm(roll, num_bins=bins)
    p = discretize_pwm(pitch, num_bins=bins)
    t = discretize_pwm(throttle, num_bins=bins)
    y = discretize_pwm(yaw, num_bins=bins)
    return r * bins**3 + p * bins**2 + t * bins + y


def decode_action(action: int, bins: int = NUM_BINS) -> Tuple[int, int, int, int]:
    """Decode a discrete action index back to bin indices."""
    r = action // bins**3
    action = action % bins**3
    p = action // bins**2
    action = action % bins**2
    t = action // bins
    y = action % bins
    return r, p, t, y


def extract_frames(video_path: Path, output_dir: Path, fps: int = TARGET_FPS, size: Tuple[int, int] = LOW_RES_SIZE) -> int:
    """Extract frames from video at specified FPS and resolution using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = size
    output_pattern = str(output_dir / "frame_%05d.png")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={w}:{h}",
        "-q:v", "2",  # High quality
        output_pattern
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to extract frames from {video_path}")

    # Count extracted frames
    num_frames = len(list(output_dir.glob("frame_*.png")))
    return num_frames


def load_frames(frame_dir: Path, num_frames: int) -> torch.FloatTensor:
    """Load frames and convert to tensor with shape [T, C, H, W] in range [-1, 1]."""
    frames = []
    for i in range(1, num_frames + 1):
        frame_path = frame_dir / f"frame_{i:05d}.png"
        if not frame_path.exists():
            break

        img = Image.open(frame_path).convert("RGB")
        # Convert to tensor: [H, W, C] -> [C, H, W], normalize to [-1, 1]
        arr = np.array(img, dtype=np.float32)
        arr = arr / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        frames.append(tensor)

    return torch.stack(frames)


def sync_actions_to_frames(csv_path: Path, num_frames: int, fps: int = TARGET_FPS) -> torch.LongTensor:
    """
    Load RC inputs from CSV and sync to video frame timestamps.
    Returns discrete action indices for each frame.
    """
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ["time", "roll", "pitch", "throttle", "yaw"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    actions = []
    for frame_idx in range(num_frames):
        # Frame timestamp
        t = frame_idx / fps

        # Find nearest CSV row
        idx = (df["time"] - t).abs().idxmin()
        row = df.iloc[idx]

        # Discretize and encode
        action = encode_action(
            roll=row["roll"],
            pitch=row["pitch"],
            throttle=row["throttle"],
            yaw=row["yaw"]
        )
        actions.append(action)

    return torch.tensor(actions, dtype=torch.long)


def create_episode(obs: torch.FloatTensor, act: torch.LongTensor) -> Episode:
    """Create an Episode object from observations and actions."""
    T = obs.size(0)

    return Episode(
        obs=obs,
        act=act,
        rew=torch.zeros(T, dtype=torch.float32),
        end=torch.zeros(T, dtype=torch.uint8),
        trunc=torch.zeros(T, dtype=torch.uint8),
        info={}
    )


def find_recording_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching video and CSV file pairs."""
    pairs = []

    for video_path in sorted(input_dir.glob("*.mkv")):
        csv_path = video_path.with_suffix(".csv")
        if csv_path.exists():
            pairs.append((video_path, csv_path))
        else:
            print(f"Warning: No CSV found for {video_path.name}, skipping")

    # Also check for mp4 files
    for video_path in sorted(input_dir.glob("*.mp4")):
        csv_path = video_path.with_suffix(".csv")
        if csv_path.exists():
            pairs.append((video_path, csv_path))
        else:
            print(f"Warning: No CSV found for {video_path.name}, skipping")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Convert FPV recordings to Diamond format")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Directory containing .mkv/.mp4 and .csv file pairs")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for Diamond dataset")
    parser.add_argument("--fps", type=int, default=TARGET_FPS,
                        help=f"Target FPS for training (default: {TARGET_FPS})")
    parser.add_argument("--num_bins", type=int, default=NUM_BINS,
                        help=f"Number of bins per RC channel (default: {NUM_BINS}, total actions = bins^4)")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Fraction of episodes for test set (default: 0.1)")
    parser.add_argument("--low_res", type=str, default="30x56",
                        help="Low resolution for training as HxW (default: 30x56)")
    args = parser.parse_args()

    # Parse resolution
    h, w = map(int, args.low_res.split("x"))
    low_res_size = (h, w)

    # Update global config
    global NUM_BINS, TARGET_FPS, LOW_RES_SIZE
    NUM_BINS = args.num_bins
    TARGET_FPS = args.fps
    LOW_RES_SIZE = low_res_size

    print(f"Configuration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Target FPS: {args.fps}")
    print(f"  Resolution: {low_res_size[0]}x{low_res_size[1]}")
    print(f"  Action bins per channel: {args.num_bins}")
    print(f"  Total action space: {args.num_bins ** 4}")
    print()

    # Find recording pairs
    pairs = find_recording_pairs(args.input_dir)
    if not pairs:
        print("No valid recording pairs found!")
        return

    print(f"Found {len(pairs)} recording pairs")

    # Split into train/test
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    num_test = max(1, int(len(pairs) * args.test_split))
    test_indices = set(indices[:num_test])

    # Create datasets
    train_dir = args.output_dir / "train"
    test_dir = args.output_dir / "test"

    train_dataset = Dataset(train_dir, dataset_full_res=None)
    test_dataset = Dataset(test_dir, dataset_full_res=None)

    # Process each recording
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for i, (video_path, csv_path) in enumerate(tqdm(pairs, desc="Converting recordings")):
            try:
                # Extract frames
                frame_dir = tmp_path / f"frames_{i}"
                num_frames = extract_frames(video_path, frame_dir, fps=args.fps, size=low_res_size)

                if num_frames < 10:
                    print(f"Warning: {video_path.name} has only {num_frames} frames, skipping")
                    continue

                # Load frames
                obs = load_frames(frame_dir, num_frames)

                # Sync actions
                act = sync_actions_to_frames(csv_path, num_frames, fps=args.fps)

                # Create episode
                episode = create_episode(obs, act)

                # Add to appropriate dataset
                if i in test_indices:
                    test_dataset.add_episode(episode)
                else:
                    train_dataset.add_episode(episode)

                # Clean up frames
                for f in frame_dir.glob("*.png"):
                    f.unlink()
                frame_dir.rmdir()

            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
                continue

    # Save dataset metadata
    train_dataset.save_to_default_path()
    test_dataset.save_to_default_path()

    print()
    print(f"Conversion complete!")
    print(f"  Train: {train_dataset.num_episodes} episodes, {train_dataset.num_steps} frames")
    print(f"  Test: {test_dataset.num_episodes} episodes, {test_dataset.num_steps} frames")
    print()
    print(f"Update config/env/fpv.yaml with:")
    print(f"  path_data_low_res: {args.output_dir}")
    print(f"  num_actions: {args.num_bins ** 4}")


if __name__ == "__main__":
    main()
