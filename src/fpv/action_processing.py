"""
FPV drone action encoding/decoding.

Converts between continuous PWM values (0-2048 from controller) and discrete action indices.
Uses the same encoding scheme as convert_fpv_data.py for consistency.
"""

from dataclasses import dataclass
from typing import Tuple

import torch


# Configuration - must match convert_fpv_data.py and fpv.yaml
NUM_BINS = 8  # Per channel: 8^4 = 4096 total actions
PWM_MIN = 0
PWM_MAX = 2048  # RadioMaster controller range


@dataclass
class FPVAction:
    """Raw FPV controller state."""
    roll: int  # 0-2048
    pitch: int  # 0-2048
    throttle: int  # 0-2048
    yaw: int  # 0-2048


def discretize_pwm(value: int, min_pwm: int = PWM_MIN, max_pwm: int = PWM_MAX, num_bins: int = NUM_BINS) -> int:
    """Discretize a PWM value into a bin index."""
    normalized = (value - min_pwm) / (max_pwm - min_pwm)
    return int(max(0, min(num_bins - 1, normalized * num_bins)))


def undiscretize_pwm(bin_idx: int, min_pwm: int = PWM_MIN, max_pwm: int = PWM_MAX, num_bins: int = NUM_BINS) -> int:
    """Convert bin index back to PWM value (center of bin)."""
    bin_width = (max_pwm - min_pwm) / num_bins
    return int(min_pwm + (bin_idx + 0.5) * bin_width)


def encode_fpv_action(action: FPVAction, device: torch.device = None) -> torch.LongTensor:
    """Encode FPV controller state to discrete action index."""
    r = discretize_pwm(action.roll)
    p = discretize_pwm(action.pitch)
    t = discretize_pwm(action.throttle)
    y = discretize_pwm(action.yaw)

    action_idx = r * NUM_BINS**3 + p * NUM_BINS**2 + t * NUM_BINS + y

    tensor = torch.tensor([action_idx], dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def decode_fpv_action(action: torch.LongTensor) -> FPVAction:
    """Decode discrete action index back to FPV controller state."""
    idx = action.item() if isinstance(action, torch.Tensor) else action

    r = idx // NUM_BINS**3
    idx = idx % NUM_BINS**3
    p = idx // NUM_BINS**2
    idx = idx % NUM_BINS**2
    t = idx // NUM_BINS
    y = idx % NUM_BINS

    return FPVAction(
        roll=undiscretize_pwm(r),
        pitch=undiscretize_pwm(p),
        throttle=undiscretize_pwm(t),
        yaw=undiscretize_pwm(y)
    )


def decode_to_bins(action: torch.LongTensor) -> Tuple[int, int, int, int]:
    """Decode action to bin indices (0-7 each)."""
    idx = action.item() if isinstance(action, torch.Tensor) else action

    r = idx // NUM_BINS**3
    idx = idx % NUM_BINS**3
    p = idx // NUM_BINS**2
    idx = idx % NUM_BINS**2
    t = idx // NUM_BINS
    y = idx % NUM_BINS

    return r, p, t, y
