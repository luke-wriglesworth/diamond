"""FPV Game loop with drone controller integration."""

from typing import Tuple

import numpy as np
import pygame
from PIL import Image

from fpv.action_processing import FPVAction
from .fpv_play_env import FPVPlayEnv

# Try to import controller, fall back gracefully
try:
    from controller.linux import ControllerPassthrough, ControllerState
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False
    ControllerPassthrough = None
    ControllerState = None


class FPVGame:
    """Game loop for FPV world model with drone controller support."""

    def __init__(
        self,
        play_env: FPVPlayEnv,
        size: Tuple[int, int],
        fps: int = 15,
    ) -> None:
        self.env = play_env
        self.height, self.width = size
        self.fps = fps
        self.use_controller = play_env.use_controller and CONTROLLER_AVAILABLE

        if self.use_controller:
            print("\nUsing physical drone controller")
            print("Controller will be grabbed - only this window receives input")
        else:
            if play_env.use_controller and not CONTROLLER_AVAILABLE:
                print("\nController module not available, using keyboard fallback")
            print("\nKeyboard controls:")
            print("  Arrow keys: Roll/Pitch")
            print("  W/S: Throttle up/down")
            print("  A/D: Yaw left/right")

        self.env.print_controls()
        print("\nGame controls:")
        print("  Enter: Reset")
        print("  Esc: Quit")
        print()
        input("Press enter to start")

    def run(self) -> None:
        pygame.init()

        header_height = 150
        header_width = 300
        font_size = 16
        screen = pygame.display.set_mode((self.width + 40, self.height + header_height + 60))
        pygame.display.set_caption("FPV World Model")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("mono", font_size)

        x_center = (self.width + 40) // 2
        y_header = 10
        header_rect = pygame.Rect(20, y_header, header_width, header_height)
        y_image = header_height + 30

        def clear_header():
            pygame.draw.rect(screen, pygame.Color("black"), header_rect)
            pygame.draw.rect(screen, pygame.Color("white"), header_rect, 1)

        def draw_text(text, idx_line, x_offset=5):
            y_pos = y_header + 5 + idx_line * font_size
            screen.blit(font.render(text, True, pygame.Color("white")), (25 + x_offset, y_pos))

        def draw_obs(obs):
            assert obs.ndim == 4 and obs.size(0) == 1
            img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            pygame_image = np.array(img.resize((self.width, self.height), resample=Image.BICUBIC)).transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(pygame_image)
            screen.blit(surface, (20, y_image))

        def reset():
            nonlocal obs, do_reset
            obs, _ = self.env.reset()
            pygame.event.clear()
            do_reset = False

        # Keyboard state for fallback controls
        keyboard_state = {
            "roll": 1024,
            "pitch": 1024,
            "throttle": 0,
            "yaw": 1024
        }
        key_step = 128  # How much each keypress changes the value

        def get_keyboard_action() -> FPVAction:
            """Get action from keyboard state."""
            keys = pygame.key.get_pressed()

            # Roll: left/right arrows
            if keys[pygame.K_LEFT]:
                keyboard_state["roll"] = max(0, keyboard_state["roll"] - key_step)
            elif keys[pygame.K_RIGHT]:
                keyboard_state["roll"] = min(2048, keyboard_state["roll"] + key_step)
            else:
                # Return to center
                keyboard_state["roll"] = 1024

            # Pitch: up/down arrows
            if keys[pygame.K_UP]:
                keyboard_state["pitch"] = max(0, keyboard_state["pitch"] - key_step)
            elif keys[pygame.K_DOWN]:
                keyboard_state["pitch"] = min(2048, keyboard_state["pitch"] + key_step)
            else:
                keyboard_state["pitch"] = 1024

            # Throttle: W/S (maintains value)
            if keys[pygame.K_w]:
                keyboard_state["throttle"] = min(2048, keyboard_state["throttle"] + key_step // 2)
            elif keys[pygame.K_s]:
                keyboard_state["throttle"] = max(0, keyboard_state["throttle"] - key_step // 2)

            # Yaw: A/D
            if keys[pygame.K_a]:
                keyboard_state["yaw"] = max(0, keyboard_state["yaw"] - key_step)
            elif keys[pygame.K_d]:
                keyboard_state["yaw"] = min(2048, keyboard_state["yaw"] + key_step)
            else:
                keyboard_state["yaw"] = 1024

            return FPVAction(
                roll=keyboard_state["roll"],
                pitch=keyboard_state["pitch"],
                throttle=keyboard_state["throttle"],
                yaw=keyboard_state["yaw"]
            )

        obs = None
        do_reset = True
        should_stop = False

        # Controller context manager (if using physical controller)
        controller_ctx = None
        controller = None

        try:
            if self.use_controller:
                controller_ctx = ControllerPassthrough(grab=True)
                controller = controller_ctx.__enter__()
                print(f"Connected to: {controller.device_name}")

            while not should_stop:
                if do_reset:
                    reset()

                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        should_stop = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            should_stop = True
                        elif event.key == pygame.K_RETURN:
                            do_reset = True

                if should_stop:
                    break

                # Get action from controller or keyboard
                if self.use_controller and controller is not None:
                    state = controller.update()
                    fpv_action = FPVAction(
                        roll=state.roll,
                        pitch=state.pitch,
                        throttle=state.throttle,
                        yaw=state.yaw
                    )
                else:
                    fpv_action = get_keyboard_action()

                # Step world model
                next_obs, rew, end, trunc, info = self.env.step(fpv_action)

                # Draw header
                screen.fill(pygame.Color("black"))
                clear_header()
                if info is not None and "header" in info:
                    for i, line in enumerate(info["header"][0]):
                        draw_text(line, i)

                # Draw observation
                draw_obs(obs)

                pygame.display.flip()
                clock.tick(self.fps)

                if end or trunc:
                    do_reset = True
                else:
                    obs = next_obs

        finally:
            if controller_ctx is not None:
                controller_ctx.__exit__(None, None, None)
            pygame.quit()
