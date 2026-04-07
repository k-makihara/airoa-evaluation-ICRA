#!/usr/bin/env python3
import argparse
import logging
import os
from typing import Any

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.rtc_wrapper import RTCPolicyWrapper
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from policy_client.base_policy import BasePolicy
from runtime_core.websocket_policy_server import WebsocketPolicyServer


LOGGER = logging.getLogger(__name__)

STATE_DIM = 8
ACTION_DIM = 11
ACTION_ORDER = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "gripper",
    "head_pan_joint",
    "head_tilt_joint",
    "base_x",
    "base_y",
    "base_t",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a GR00T HSR policy over websocket")
    parser.add_argument("--checkpoint-dir", required=True, help="Local checkpoint directory or Hugging Face model id")
    parser.add_argument("--host", default=os.environ.get("POLICY_SERVER_HOST", "0.0.0.0"), help="Bind host")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("POLICY_SERVER_PORT", "8000")),
        help="Bind port",
    )
    parser.add_argument(
        "--data-config",
        default=os.environ.get("GR00T_DATA_CONFIG", "hsr_v2"),
        help="GR00T data config key in DATA_CONFIG_MAP",
    )
    parser.add_argument(
        "--embodiment-tag",
        default=os.environ.get("GR00T_EMBODIMENT_TAG", EmbodimentTag.NEW_EMBODIMENT.value),
        help="Embodiment tag used when loading the policy",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("GR00T_DEVICE"),
        help='Torch device override. Defaults to "cuda" when available, otherwise "cpu".',
    )
    parser.add_argument(
        "--adopted-action-chunks",
        type=int,
        default=int(os.environ.get("GR00T_ADOPTED_ACTION_CHUNKS", "32")),
        help="Maximum number of predicted actions returned to the client",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=int(os.environ.get("GR00T_CONTROL_FREQ", "20")),
        help="Robot control frequency used by RTC wrapper",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=int(os.environ.get("GR00T_DENOISING_STEPS", "4")),
        help="Number of denoising steps for GR00T inference",
    )
    parser.add_argument(
        "--max-rtc-overlap-factor",
        type=float,
        default=float(os.environ.get("GR00T_MAX_RTC_OVERLAP_FACTOR", "0.75")),
        help="RTC overlap ratio",
    )
    return parser.parse_args()


class Gr00tWebPolicy(BasePolicy):
    def __init__(
        self,
        *,
        checkpoint_dir: str,
        data_config_name: str,
        embodiment_tag: str,
        adopted_action_chunks: int,
        control_freq: int,
        denoising_steps: int,
        max_rtc_overlap_factor: float,
        device: str | None = None,
    ) -> None:
        if data_config_name not in DATA_CONFIG_MAP:
            available = ", ".join(sorted(DATA_CONFIG_MAP))
            raise ValueError(f"Unknown GR00T data config: {data_config_name}. Available: {available}")

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        data_config = DATA_CONFIG_MAP[data_config_name]

        self._checkpoint_dir = checkpoint_dir
        self._data_config_name = data_config_name
        self._embodiment_tag = EmbodimentTag(embodiment_tag)
        self._adopted_action_chunks = max(1, int(adopted_action_chunks))
        self._device = resolved_device

        self._policy = Gr00tPolicy(
            model_path=checkpoint_dir,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=self._embodiment_tag,
            device=resolved_device,
        )
        self._rtc_policy = RTCPolicyWrapper(
            self._policy,
            control_freq=control_freq,
            denoising_steps=denoising_steps,
            max_rtc_overlap_factor=max_rtc_overlap_factor,
        )

        LOGGER.info(
            "Loaded GR00T checkpoint=%s data_config=%s embodiment_tag=%s device=%s",
            checkpoint_dir,
            data_config_name,
            self._embodiment_tag.value,
            resolved_device,
        )

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        validated = self._validate_observation(obs)
        policy_input = self._build_policy_input(validated)
        raw_action = self._rtc_policy.get_action(policy_input)
        actions = self._to_hsr_action_chunk(raw_action["action.relative"], validated["state"])
        return {"actions": actions}

    def reset(self) -> None:
        pass

    def _validate_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        missing = [key for key in ("head_rgb", "hand_rgb", "state", "prompt") if key not in obs]
        if missing:
            raise ValueError(f"Missing inference keys: {missing}")

        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state must have shape ({STATE_DIM},), got {state.shape}")

        head_rgb = np.asarray(obs["head_rgb"], dtype=np.uint8)
        hand_rgb = np.asarray(obs["hand_rgb"], dtype=np.uint8)
        if head_rgb.ndim != 3 or head_rgb.shape[-1] != 3:
            raise ValueError(f"head_rgb must have shape (H, W, 3), got {head_rgb.shape}")
        if hand_rgb.ndim != 3 or hand_rgb.shape[-1] != 3:
            raise ValueError(f"hand_rgb must have shape (H, W, 3), got {hand_rgb.shape}")

        prompt = str(obs["prompt"])
        return {
            "head_rgb": head_rgb,
            "hand_rgb": hand_rgb,
            "state": state,
            "prompt": prompt,
        }

    def _build_policy_input(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "video.head": np.expand_dims(obs["head_rgb"], axis=0),
            "video.hand": np.expand_dims(obs["hand_rgb"], axis=0),
            "state.arm": np.expand_dims(obs["state"][:5], axis=0),
            "state.gripper": np.expand_dims(np.expand_dims(obs["state"][5], axis=0), axis=0),
            "state.head": np.expand_dims(obs["state"][6:8], axis=0),
            "annotation.human.task_description": [obs["prompt"]],
        }

    def _to_hsr_action_chunk(self, action_relative: Any, state: np.ndarray) -> np.ndarray:
        chunk = np.asarray(action_relative, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] < ACTION_DIM:
            raise ValueError(f'action.relative must have shape (T, >= {ACTION_DIM}), got {chunk.shape}')

        chunk = chunk[: self._adopted_action_chunks]
        delta_offset = np.concatenate(
            [state[:5], np.array([0.0], dtype=np.float32), state[6:8], np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        )
        actions: list[np.ndarray] = []
        for row in chunk:
            # Keep the same HSR mapping as the reference yl script:
            # arm/gripper/base come from GR00T, while head stays at its current absolute position.
            translated = np.concatenate(
                [
                    row[0:5],
                    [row[5]],
                    [0.0, 0.0],
                    row[8:11],
                ]
            ).astype(np.float32, copy=False)
            actions.append(translated + delta_offset)

        stacked = np.stack(actions, axis=0)
        if not np.isfinite(stacked).all():
            raise ValueError("Predicted actions contain non-finite values")
        return stacked


def main() -> None:
    args = parse_args()
    policy = Gr00tWebPolicy(
        checkpoint_dir=args.checkpoint_dir,
        data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag,
        adopted_action_chunks=args.adopted_action_chunks,
        control_freq=args.control_freq,
        denoising_steps=args.denoising_steps,
        max_rtc_overlap_factor=args.max_rtc_overlap_factor,
        device=args.device,
    )

    metadata = {
        "checkpoint_dir": args.checkpoint_dir,
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "action_order": ACTION_ORDER,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "server_host": args.host,
        "server_port": args.port,
    }

    LOGGER.info("Serving GR00T websocket policy on %s:%s", args.host, args.port)
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
