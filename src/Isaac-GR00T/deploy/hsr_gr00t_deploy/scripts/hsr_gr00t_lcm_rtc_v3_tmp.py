#!/home/openpi/.venv/bin/python3
from collections import deque

#!/usr/bin/env python3
from typing import Any
import copy

import cv2
import numpy as np
import torch  # <-- 追加

# gr00t関連
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

from gr00t.model.policy import Gr00tPolicy

import lcm
from lcm_msgs import RequestMsg, ResponseMsg


def compressedimage_to_array_lcm(msg):
    # Convert signed bytes (-128~127) to unsigned (0~255)
    signed = np.array(msg.data[: msg.data_length], dtype=np.int8)
    unsigned = signed.astype(np.uint8)

    # TODO: 元コードにバグがあり，rgbに変換されずに使われていたのでそのままにしてある
    # 学習はバグありのままされていた？
    # そうならそのまま使って，そうでないなら以下の正しいコードを使う
    image = cv2.imdecode(unsigned, cv2.IMREAD_COLOR)[:, :, :]  # bgr -> rgb
    # image = cv2.imdecode(unsigned, cv2.IMREAD_COLOR)[:, :, ::-1]  # bgr -> rgb
    return image


class HSRLcmServer:
    GRIPPER_OPEN = 1
    GRIPPER_CLOSE = 0
    GRIPPER_CLOSE_THRESHOLD = 0.9  # グリッパーを閉じる閾値

    def __init__(self, policy, traj_hz=10.0):
        self.traj_hz = float(traj_hz)

        self._lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
        self._lc.subscribe("REQUEST_CHANNEL", self._handle_request)

        self.policy = policy

        self.joint_state_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]

    def handle(self):
        self._lc.handle()

    def _handle_request(self, channel, data):
        request = RequestMsg.decode(data)
        head_rgb = compressedimage_to_array_lcm(request.head_rgb)
        hand_rgb = compressedimage_to_array_lcm(request.hand_rgb)

        try:
            # self.joint_state_namesの順に並び替え
            joint_msg = request.joint_state
            name_list = joint_msg.name[: joint_msg.num_joints]
            position_list = joint_msg.position[: joint_msg.num_joints]
            joint_state = [position_list[name_list.index(name)] for name in self.joint_state_names]

            observation = {
                "head_rgb": head_rgb,
                "hand_rgb": hand_rgb,
                "joint_state": np.array(joint_state, dtype=np.float32),
                "instruction": request.instruction,
            }

            action = self.policy.act(observation)

            action_joint_names = self.joint_state_names + [
                "base_x",
                "base_y",
                "base_t",
            ]

            if action.shape[1] != len(action_joint_names):
                raise RuntimeError(f"Expected (T, {len(action_joint_names)}), got {action.shape}")

        except Exception as e:
            print(f"Error processing data: {e}")
            # 現在位置 + 台車速度なし（仮）で対応
            action = np.array([joint_state + [0.0, 0.0, 0.0]])

        rows, cols = action.shape
        response = ResponseMsg()
        response.hz = self.traj_hz
        response.num_joints = cols
        response.joint_names = action_joint_names
        response.rows = rows
        response.cols = cols
        response.result = action.tolist()

        self._lc.publish("ACTION_RESPONSE", response.encode())


class Gr00tHSRPolicy:
    """
    Gr00tPolicyをHSRロボットに適用するためのクラスです.
    HSREnvからセンサ情報を取得し, policyの計算後にアクションを環境に反映させます.
    """

    def __init__(
        self,
        model_path: str = "/home/veluga-g3/airoa/gr00t-microwave", 
        adopted_action_chunks: int = 15,
        num_traj: int = 1
    ):
        assert num_traj <= adopted_action_chunks, "num_traj must be <= adopted_action_chunks"
        self.dagtconfig = data_config = DATA_CONFIG_MAP["hsr_v2"]
        self.modality_config = data_config.modality_config()
        transforms = data_config.transform()
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=self.modality_config,
            modality_transform=transforms,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # HSRのembodiment tag
            device="cuda",
        )
        self.adopted_action_chunks = adopted_action_chunks
        self.action_queue = {
            "action.relative": deque(maxlen=self.adopted_action_chunks),  
        }
        self.num_traj = num_traj

        # デバッグ再生用（不要ならコメントアウトOK）
        #replay_episode = np.load("/home/veluga-g3/Downloads/episode_1511_action_relative.npz", allow_pickle=False)
        #self.actions_rel = replay_episode["actions"]
        self.frame_num = 0

        # 時間方向のEMA（任意）
        self.use_temp_ensem = False
        self.temporal_half_life = 8
        self._ema_action = None

        # ====== RTC 関連（デバッグ版に合わせた#実装） ======
        self.prev_chunk_world = None          # 直近のモデル出力（非正規化＝物理空間）
        self.rtc_d_steps = 0                  # d は基本0（freezeは悪化しやすい）
        self.rtc_s_steps = 4                  # 末尾の自由区間。小さめが安定（例: 4）
        self.rtc_lambda  = 0.0                # 中間の減衰。まずは 0
        self.rtc_beta    = 0.05               # ガイダンス強度（小さめから）
        self.rtc_guidance_clip = 0.5          # クリップ（0.25〜1.0で探索推奨）
        self.angle_indices = []               # 角度DoFがあれば指定（例: [10]）

    @staticmethod
    def _angle_wrap_inplace(actions: np.ndarray, indices: list[int] | None):
        if not indices:
            return
        for idx in indices:
            actions[..., idx] = np.arctan2(np.sin(actions[..., idx]), np.cos(actions[..., idx]))

    def reset_buffer(self):
        self.action_queue.clear()
        self.prev_chunk_world = None
        self._ema_action = None

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        obs: Dict[str, Any]
        return: np.ndarray : shape (T_out, 11)
        """
        # 既にキューに十分溜まっていれば、それを返す
        if len(self.action_queue["action.relative"]) >= self.num_traj:
            actions = []
            for _ in range(self.num_traj):
                action_relative = self.action_queue["action.relative"].popleft()
                action = np.concatenate(
                    [
                        action_relative[0:5],
                        [action_relative[5]],
                        action_relative[6:8],
                        action_relative[8:11],
                    ]
                )
                self.frame_num += 1

                # 差分 -> 絶対へ
                action = action + np.concatenate(
                    [obs["joint_state"][:5], np.array([0.0]), obs["joint_state"][6:8], np.array([0.0, 0.0, 0.0])]
                )

                if self.use_temp_ensem:
                    alpha = 1.0 - np.exp(-np.log(2) / max(1e-6, self.temporal_half_life))
                    if self._ema_action is None:
                        self._ema_action = action.astype(np.float64)
                    else:
                        self._ema_action = alpha * action + (1.0 - alpha) * self._ema_action
                    action = self._ema_action

                actions.append(action)
            return np.stack(actions)

        # === ここから新しいチャンク推論 ===
        video_head = np.expand_dims(obs["head_rgb"], axis=0)
        video_hand = np.expand_dims(obs["hand_rgb"], axis=0)
        state_arm  = np.expand_dims(obs["joint_state"][:5], axis=0)
        state_hand = np.expand_dims(np.expand_dims(obs["joint_state"][5], axis=0), axis=0)
        state_head = np.expand_dims(obs["joint_state"][6:8], axis=0)
        instruction = [obs["instruction"]]
        policy_input = {
            "video.head": video_head,
            "video.hand": video_hand,
            "state.arm" : state_arm,
            "state.gripper": state_hand,
            "state.head": state_head,
            "annotation.human.task_description": instruction,
        }

        H = self.policy.model.action_head.action_horizon
        D = self.policy.model.action_head.action_dim
        B = 1
        device = self.policy.model.device if hasattr(self.policy.model, "device") else "cuda"
        dtype  = (self.policy.model.action_head.dtype
                  if hasattr(self.policy.model.action_head, "dtype") else torch.float32)

        # --- W の作成（デバッグ版の形に揃える：先頭d=1.0, 中間=指数減衰(λ), 末尾s=0.0） ---
        d_steps = int(self.rtc_d_steps)
        # 実行で消費するぶんだけは必ず自由に（s >= num_traj）
        s_steps = max(int(self.rtc_s_steps), int(self.num_traj))
        W = self.policy.build_rtc_weight_mask(
            H, d_steps, s_steps, lam=self.rtc_lambda, B=B, D=D, device=device, dtype=dtype
        )

        # 11次元だけ誘導し、pad分は誘導しない（dim mask を合成）
        if D > 11:
            dim_mask = torch.cat(
                [torch.ones(11, device=device), torch.zeros(D - 11, device=device)]
            )[None, None, :].to(dtype)
            W = W * dim_mask  # W32 = W32 * dim_mask

        # --- 前チャンク（非正規化＝物理空間）を RTC に渡す ---
        # 実運用：直前の出力を左シフトして「消費済み」を抜いた形で保持（デバッグ版と同じ）
        rtc_prev = None if self.prev_chunk_world is None else self.prev_chunk_world

        rtc = {
            "rtc_prev_action": rtc_prev,  # policy 側で正規化・pad
            "rtc_weight_mask": W,
            "rtc_beta": float(self.rtc_beta),
            "rtc_guidance_clip": float(self.rtc_guidance_clip),
            # "rtc_angle_indices": self.angle_indices,  # 必要なら有効化
        }

        action_chunk = self.policy.get_action_rtc(policy_input, rtc)
        action_chunk_world = action_chunk["action.relative"]  # (H, 11) 非正規化＝物理空間

        # === 直近出力を次回ガイダンス用に「左シフト」して保存（消費分を抜く）===
        # デバッグ版と同じ: s = num_traj だけ消費、残りを上に詰めて末尾を0でパディング
        s = int(self.num_traj)
        A_prev = action_chunk_world[s:]
        A_prev = np.pad(A_prev, ((0, s), (0, 0)))  # 末尾パディング（物理空間でOK）
        # 必要なら角度DoF wrap
        self._angle_wrap_inplace(A_prev, self.angle_indices)
        self.prev_chunk_world = A_prev.copy()

        # === 実行分をキューに積む ===
        # 今回出したチャンクのうち、最初の num_traj は直ちに使うので、
        # 次ループ用に [num_traj : adopted_action_chunks) をキューに積む
        tail = action_chunk["action.relative"][self.num_traj:self.adopted_action_chunks]
        self.action_queue["action.relative"].extend(tail)

        # === 今ステップで返す分 ===
        actions = []
        for i in range(self.num_traj):
            action_relative = action_chunk["action.relative"][i]
            action = np.concatenate(
                [
                    action_relative[0:5],
                    [action_relative[5]],
                    action_relative[6:8],
                    action_relative[8:11],
                ]
            )
            self.frame_num += 1

            # 差分 -> 絶対へ
            action = action + np.concatenate(
                [obs["joint_state"][:5], np.array([0.0]), obs["joint_state"][6:8], np.array([0.0, 0.0, 0.0])]
            )

            if self.use_temp_ensem:
                alpha = 1.0 - np.exp(-np.log(2) / max(1e-6, self.temporal_half_life))
                if self._ema_action is None:
                    self._ema_action = action.astype(np.float64)
                else:
                    self._ema_action = alpha * action + (1.0 - alpha) * self._ema_action
                action = self._ema_action

            actions.append(action)

        return np.stack(actions)


def main():
    print("Start Issac-GR00T")

    # TODO: 引数でいい感じに処理するようにする
    checkpoint_dir = "/home/veluga-g3/airoa/ckpt/gr00t-microwave-chunk32_10hz"
    adopted_action_chunks = 31

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adopted_action_chunks: {adopted_action_chunks}")

    policy = Gr00tHSRPolicy(model_path=checkpoint_dir, adopted_action_chunks=adopted_action_chunks)

    # for i in range(5):
    #     rand_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    #     policy_input = {
    #         "head_rgb": rand_img,
    #         "hand_rgb": rand_img,
    #         "joint_state": np.array([0.0 for _ in range(8)]),
    #         "instruction": "Test prompt. Do not move.",
    #     }
    #     action = policy.act(policy_input)
    #     print(action)
    # import sys; sys.exit()

    lcm_hsr_server = HSRLcmServer(policy)

    print("start server...")
    try:
        while True:
            lcm_hsr_server.handle()
    except KeyboardInterrupt:
        pass

    return


if __name__ == "__main__":
    main()
