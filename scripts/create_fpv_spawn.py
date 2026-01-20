#!/usr/bin/env python3
"""
Create spawn data for FPV world model inference.

The WorldModelEnv expects spawn directories containing:
    low_res.npy  - Initial observation sequence [T, C, H, W]
    act.npy      - Action sequence [T]
    next_act.npy - Future actions for replay mode [T]
    full_res.npy - Full resolution observations (optional)

This script extracts random starting points from FPV episodes.

Need to use sequence length 4 to match training data or it will crash.

Usage:
    python scripts/create_fpv_spawn.py --output_dir data/fpv/spawn --num_spawns 10 --seq_length 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_episode(episode_path: Path) -> dict:
    """Load episode from .pt file."""
    data = torch.load(episode_path, weights_only=False)
    return data


def create_spawn(
    episode: dict,
    start_idx: int,
    seq_length: int,
    future_length: int = 100
) -> dict:
    """Create spawn data from episode at given start index."""
    obs = episode["obs"]  # [T, C, H, W] uint8
    act = episode["act"]  # [T] int64

    end_idx = start_idx + seq_length
    future_end = min(end_idx + future_length, len(act))

    # Keep obs as uint8 [0-255] - WorldModelEnv converts to float [-1, 1]
    obs_seq = obs[start_idx:end_idx]
    act_seq = act[start_idx:end_idx]
    next_act = act[end_idx:future_end]

    return {
        "low_res": obs_seq.numpy(),  # [T, C, H, W] uint8 [0-255]
        "act": act_seq.numpy(),      # [T]
        "next_act": next_act.numpy(),  # [future_length]
    }


def main():
    parser = argparse.ArgumentParser(description="Create FPV spawn data")
    parser.add_argument("--data_dir", type=Path, default=Path("data/fpv/low_res"),
                        help="FPV dataset directory")
    parser.add_argument("--output_dir", type=Path, default=Path("data/fpv/spawn"),
                        help="Output directory for spawn data")
    parser.add_argument("--num_spawns", type=int, default=20,
                        help="Number of spawn points to create")
    parser.add_argument("--seq_length", type=int, default=16,
                        help="Number of conditioning frames")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use (train/test)")
    parser.add_argument("--full_res_dir", type=Path, default=None,
                        help="Full resolution data directory (optional)")
    args = parser.parse_args()

    split_dir = args.data_dir / args.split

    # Find all episode files
    episode_files = sorted(split_dir.glob("**/*.pt"))
    episode_files = [f for f in episode_files if f.name != "info.pt"]

    if not episode_files:
        print(f"No episodes found in {split_dir}")
        return

    print(f"Found {len(episode_files)} episodes in {split_dir}")

    # Load episodes
    episodes = []
    for ep_path in tqdm(episode_files, desc="Loading episodes"):
        try:
            ep = load_episode(ep_path)
            if len(ep["obs"]) >= args.seq_length + 50:  # Need enough frames
                episodes.append((ep_path, ep))
        except Exception as e:
            print(f"Error loading {ep_path}: {e}")

    if not episodes:
        print("No valid episodes found")
        return

    print(f"Loaded {len(episodes)} valid episodes")

    # Create spawn points
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    for i in tqdm(range(args.num_spawns), desc="Creating spawns"):
        # Pick random episode
        ep_path, episode = episodes[np.random.randint(len(episodes))]

        # Pick random start point
        max_start = len(episode["obs"]) - args.seq_length - 50
        start_idx = np.random.randint(0, max_start)

        # Create spawn data
        spawn = create_spawn(episode, start_idx, args.seq_length)

        # Save to directory
        spawn_dir = args.output_dir / f"{i:04d}"
        spawn_dir.mkdir(parents=True, exist_ok=True)

        np.save(spawn_dir / "low_res.npy", spawn["low_res"])
        np.save(spawn_dir / "act.npy", spawn["act"])
        np.save(spawn_dir / "next_act.npy", spawn["next_act"])

        # Create full_res from actual data or upscaled low_res
        if args.full_res_dir:
            # Use original_file_id to find the correct full_res episode
            original_id = episode["info"]["original_file_id"]
            # Build path using same hierarchy as FPVFullResDataset
            n = 3
            powers = np.arange(n)
            subfolders = np.floor((original_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
            subfolders = [int(x) for x in subfolders[::-1]]
            subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
            full_res_path = args.full_res_dir / subfolders / f"{original_id}.pt"
            if full_res_path.exists():
                full_res_ep = torch.load(full_res_path, weights_only=False)
                full_res = full_res_ep[start_idx:start_idx + args.seq_length]
                np.save(spawn_dir / "full_res.npy", full_res.numpy())  # uint8
            else:
                print(f"Warning: full_res not found at {full_res_path}, using upscaled low_res")
                import torch.nn.functional as F
                low_res = torch.from_numpy(spawn["low_res"]).float()
                full_res = F.interpolate(low_res, scale_factor=5, mode="bicubic")
                full_res = full_res.clamp(0, 255).byte()
                np.save(spawn_dir / "full_res.npy", full_res.numpy())
        else:
            # Create placeholder by upscaling low_res (keep as uint8)
            import torch.nn.functional as F
            low_res = torch.from_numpy(spawn["low_res"]).float()
            full_res = F.interpolate(low_res, scale_factor=5, mode="bicubic")
            full_res = full_res.clamp(0, 255).byte()
            np.save(spawn_dir / "full_res.npy", full_res.numpy())

    print(f"\nCreated {args.num_spawns} spawn points in {args.output_dir}")
    print(f"\nTo play the model:")
    print(f"  python src/play_fpv.py --checkpoint <path/to/checkpoint> --spawn-dir {args.output_dir}")


if __name__ == "__main__":
    main()
