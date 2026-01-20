"""FPV Play environment using drone controller."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from agent import Agent
from data import Dataset, Episode
from envs import WorldModelEnv
from fpv.action_processing import FPVAction, encode_fpv_action, decode_fpv_action, decode_to_bins


class FPVPlayEnv:
    """Play environment for FPV world model with drone controller support."""

    def __init__(
        self,
        agent: Agent,
        wm_env: WorldModelEnv,
        use_controller: bool = True,
        recording_mode: bool = False,
    ) -> None:
        self.agent = agent
        self.wm_env = wm_env
        self.use_controller = use_controller
        self.recording_mode = recording_mode
        self.is_human_player = True

        self.obs = None
        self.t = 0
        self.buffer = None
        self.rec_dataset = None

    def print_controls(self) -> None:
        print("\nFPV World Model Controls:")
        print("  Controller: Roll, Pitch, Throttle, Yaw")
        print("  Keyboard fallback: Arrow keys + W/S for throttle")

    def reset(self) -> Tuple[Tensor, None]:
        """Reset environment, loading initial frames from spawn data."""
        self.obs, _ = self.wm_env.reset()
        self.t = 0
        if self.recording_mode:
            self.reset_recording()
        return self.obs, None

    def reset_recording(self) -> None:
        self.buffer = defaultdict(list)
        self.buffer["info"] = defaultdict(list)
        dir = Path("dataset") / "rec_fpv"
        self.rec_dataset = Dataset(dir, None)
        self.rec_dataset.load_from_default_path()

    @torch.no_grad()
    def step(self, fpv_action: FPVAction) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """Step the world model with the given action."""
        action = encode_fpv_action(fpv_action, device=self.agent.device)

        next_obs, rew, end, trunc, env_info = self.wm_env.step(action)

        # Build header info for display
        r, p, t, y = decode_to_bins(action)
        header = [
            [
                f"FPV World Model",
                f"Timestep: {self.t + 1}",
                f"Horizon : {self.wm_env.horizon}",
                "",
                f"Roll    : {fpv_action.roll:4d} (bin {r})",
                f"Pitch   : {fpv_action.pitch:4d} (bin {p})",
                f"Throttle: {fpv_action.throttle:4d} (bin {t})",
                f"Yaw     : {fpv_action.yaw:4d} (bin {y})",
            ],
        ]

        info = {"header": header}
        if "obs_low_res" in env_info:
            info["obs_low_res"] = env_info["obs_low_res"]

        if self.recording_mode and self.buffer is not None:
            self.buffer["obs"].append(self.obs)
            self.buffer["act"].append(action)
            self.buffer["rew"].append(rew)
            self.buffer["end"].append(end)
            self.buffer["trunc"].append(trunc)

            if end or trunc:
                ep_dict = {k: torch.cat(v, dim=0) for k, v in self.buffer.items() if k != "info"}
                ep = Episode(**ep_dict, info={}).to("cpu")
                self.rec_dataset.add_episode(ep)
                self.rec_dataset.save_to_default_path()

        self.obs = next_obs
        self.t += 1

        return next_obs, rew, end, trunc, info

    def next_mode(self) -> bool:
        """Toggle control mode (no-op for FPV, always human)."""
        return False

    def next_axis_1(self) -> bool:
        return False

    def prev_axis_1(self) -> bool:
        return False

    def next_axis_2(self) -> bool:
        return False

    def prev_axis_2(self) -> bool:
        return False
