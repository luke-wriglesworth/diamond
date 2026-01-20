#!/usr/bin/env python3
"""
Play the FPV drone world model using your physical controller.

Usage:
    python src/play_fpv.py --checkpoint outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/state.pt

Or for a specific agent checkpoint:
    python src/play_fpv.py --checkpoint outputs/.../checkpoints/agent_versions/agent_epoch_00050.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add drone-control to path
sys.path.insert(0, str(Path(__file__).parent.parent / "drone-control"))

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from agent import Agent
from envs import WorldModelEnv
from game.fpv_game import FPVGame
from game.fpv_play_env import FPVPlayEnv

OmegaConf.register_new_resolver("eval", eval)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play FPV world model with drone controller")
    parser.add_argument("-c", "--checkpoint", type=Path, required=True,
                        help="Path to checkpoint (state.pt or agent_epoch_XXXXX.pt)")
    parser.add_argument("-s", "--spawn-dir", type=Path, default=None,
                        help="Directory with spawn data (default: data/fpv/low_res/test)")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate")
    parser.add_argument("--size-multiplier", type=int, default=4,
                        help="Display size multiplier (default: 4)")
    parser.add_argument("--compile", action="store_true", help="Compile models for faster inference")
    parser.add_argument("--no-controller", action="store_true",
                        help="Use keyboard fallback instead of drone controller")
    parser.add_argument("--horizon", type=int, default=1000,
                        help="Max steps before auto-reset (default: 1000)")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """Load checkpoint, handling both state.pt and agent_epoch_XXXXX.pt formats."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # state.pt contains full trainer state with nested agent
    if "agent" in ckpt:
        return ckpt["agent"]
    # agent_epoch_XXXXX.pt is just the agent state dict
    return ckpt


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    # Load config
    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    # Override with FPV config
    cfg.env = OmegaConf.load(Path(__file__).parent.parent / "config/env/fpv.yaml")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Float32 matmul precision: {torch.get_float32_matmul_precision()}")

    # Load agent
    num_actions = cfg.env.num_actions
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()

    agent_state = load_checkpoint(args.checkpoint, device)
    agent.load_state_dict(agent_state, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Spawn directory for initial frames
    spawn_dir = args.spawn_dir
    if spawn_dir is None:
        spawn_dir = Path(cfg.env.path_data_low_res) / "test"

    if not spawn_dir.exists():
        print(f"Spawn directory not found: {spawn_dir}")
        print("You need test data to provide initial frames for the world model.")
        return

    # Create world model environment
    seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        seq_length = max(seq_length, cfg.agent.upsampler.inner_model.num_steps_conditioning)

    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env_cfg.horizon = args.horizon

    wm_env = WorldModelEnv(
        agent.denoiser,
        agent.upsampler,
        agent.rew_end_model,
        spawn_dir,
        num_envs=1,
        seq_length=seq_length,
        cfg=wm_env_cfg,
        return_denoising_trajectory=False
    )

    if device.type == "cuda" and args.compile:
        print("Compiling models...")
        wm_env.predict_next_obs = torch.compile(wm_env.predict_next_obs, mode="reduce-overhead")
        if wm_env.sampler_upsampling is not None:
            wm_env.upsample_next_obs = torch.compile(wm_env.upsample_next_obs, mode="reduce-overhead")

    # Create play environment
    play_env = FPVPlayEnv(agent, wm_env, use_controller=not args.no_controller)

    # Window size
    size = cfg.env.train.size
    if hasattr(size, '__iter__') and not isinstance(size, str):
        h, w = list(size)
    else:
        h, w = size, size
    size_h, size_w = h * args.size_multiplier, w * args.size_multiplier

    # Run game
    game = FPVGame(play_env, (size_h, size_w), fps=args.fps)
    game.run()


if __name__ == "__main__":
    main()
