#!/home/openpi/.venv/bin/python3
from collections import deque
import threading, queue, time
#!/usr/bin/env python3
from typing import Any
import copy
import torch

import cv2
import numpy as np

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
            #print(observation)

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
        num_traj: int = 4
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
        #print(self.action_queue)
        self.num_traj = num_traj
        rand_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        policy_input = {
            "head_rgb": rand_img,
            "hand_rgb": rand_img,
            "joint_state": np.array([0.0 for _ in range(8)]),
            "instruction": "Test prompt. Do not move.",
        }
        #self.reset_buffer()

        replay_episode = np.load("/home/veluga-g3/Downloads/episode_1511_action_relative.npz", allow_pickle=False)
        self.actions_rel = replay_episode["actions"]
        self.frame_num = 0

        self.use_temp_ensem = True          # 無効にしたい時は False
        self.temporal_half_life = 8        # フレーム半減期(=約10ステップで重み半減)
        self._ema_action = None

        # ダブルバッファ & 進行状況
        self.active_chunk_world = None     # [H, 11] 非正規化
        self.next_chunk_world   = None     # [H, 11] 非正規化
        self.active_offset      = 0        # active 何ステップ消費したか
        self.stride             = num_traj # = 再計画ストライド
        self.exec_since_prev    = 0        # RTC 用 d（= 通常 stride）

        # 先読みワーカー
        self._prefetch_req = queue.Queue(maxsize=1)
        self._prefetch_res = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._worker.start()

        # RTCハイパラ
        self.s_min_ratio = 0.5
        self.mask_lambda = 0.3
        self.rtc_beta = 5.0
        self.rtc_guidance_clip = 1.0

    def _build_prev_shifted(self, P, stride, H):
        """P: [H,11] -> prev_shifted: [H,11], prev_mask: [H,11]"""
        prev_shifted = np.zeros_like(P)
        valid = max(0, H - stride)
        prev_shifted[:valid] = P[stride:]
        prev_mask = np.zeros_like(P, dtype=bool)
        prev_mask[:valid] = True
        return prev_shifted, prev_mask

    def _make_weight_mask(self, H, d, s, D32, device, dtype):
        # 既存のビルダを使用（[1,H,1] or [1,H,32] を返す想定）
        return self.policy.build_rtc_weight_mask(
            H=H, d=d, s=s, lam=self.mask_lambda,
            B=1, D=D32, device=device, dtype=dtype
        )

    def _prefetch_loop(self):
        # モデル呼び出しはこのスレッドに集約（競合回避）
        torch.set_grad_enabled(True)  # RTC(VJP)で勾配を使う
        while not self._stop.is_set():
            try:
                item = self._prefetch_req.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:  # poison pill
                break

            obs, P_world, stride, s_min_ratio = item
            try:
                # ---- 入力整形 ----
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

                H  = self.policy.model.action_head.action_horizon
                D32 = self.policy.model.action_head.action_dim
                device = self.policy.model.device if hasattr(self.policy.model, "device") else "cuda"
                dtype  = (self.policy.model.action_head.dtype
                          if hasattr(self.policy.model.action_head, "dtype") else torch.float32)

                # ---- RTC: d と s ----
                d_steps = stride
                s_min = max(1, int(H * s_min_ratio))
                s_steps = max(d_steps, s_min)

                # ---- 前チャンクのずらし（切替点に合わせる）----
                if P_world is not None:
                    prev_shifted, prev_mask = self._build_prev_shifted(P_world, stride, H)  # [H,11]
                else:
                    prev_shifted, prev_mask = None, None

                # ---- 重みマスク ----
                W = self._make_weight_mask(H, d_steps, s_steps, D32, device, dtype)

                rtc = {
                    "rtc_prev_action": prev_shifted,   # 11次元/非正規化（Policy側でTransform→32）
                    "rtc_prev_mask"  : prev_mask,      # 任意: 無効部を無視したいとき
                    "rtc_weight_mask": W,              # [1,H,32] or [1,H,1]
                    "rtc_beta": self.rtc_beta,
                    "rtc_guidance_clip": self.rtc_guidance_clip,
                }

                # ---- 推論（RTC つき）----
                # ※ get_action_rtc 内で Transform 正規化・32次元化・VJPガイダンス
                out = self.policy.get_action_rtc(policy_input, rtc)
                next_world = out["action.relative"]  # [H,11] 非正規化

                self._prefetch_res.put(next_world, timeout=0.01)
            except Exception as e:
                # 失敗したら None を入れてフォールバック
                self._prefetch_res.put(None)
            finally:
                self._prefetch_req.task_done()


    
    def reset_buffer(self):
        self.action_queue.clear()
        

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        H = self.policy.model.action_head.action_horizon

        # 1) 初回は同期ブートストラップ（遅延1回だけ許容）
        if self.active_chunk_world is None:
            # 同期で作る（d=0）
            self._prefetch_req.queue.clear()
            self._prefetch_res.queue.clear()
            self._prefetch_req.put((obs, None, 0, self.s_min_ratio))
            next_world = self._prefetch_res.get()
            if next_world is None:
                raise RuntimeError("RTC bootstrap failed")
            self.active_chunk_world = next_world
            self.active_offset = 0
            # 次をすぐ先読み（切替時用に d=stride で）
            self._prefetch_req.put((obs, self.active_chunk_world, self.stride, self.s_min_ratio))

        # 2) 現在のチャンクから stride 分だけ取り出し
        start = self.active_offset
        end   = min(H, start + self.stride)
        chunk_rel = self.active_chunk_world[start:end]  # [stride<=, 11]
        self.active_offset = end

        # 3) 出力の11次元→絶対値復元（あなたの元の処理）
        out_list = []
        for a_rel in chunk_rel:
            a_11 = np.concatenate([a_rel[0:5], [a_rel[5]], a_rel[6:8], a_rel[8:11]])
            a_abs = a_11 + np.concatenate([obs["joint_state"][:5],
                                           np.array([0]),
                                           obs["joint_state"][6:8],
                                           np.array([0,0,0])])
            out_list.append(a_abs)

        # 4) 切替境界に近づいたら、バックグラウンド結果を回収してスワップ
        #    境界: 残り < stride になったら切替
        remaining = H - self.active_offset
        if remaining < self.stride:
            try:
                # 用意できていれば使う／なければフォールバックで同期生成
                next_world = self._prefetch_res.get_nowait()
                if next_world is None:
                    raise queue.Empty
                self.active_chunk_world = next_world
                self.active_offset = 0
            except queue.Empty:
                # フォールバック：同期で作る（観測 obs、前チャンク active を prev として）
                self._prefetch_req.put((obs, self.active_chunk_world, self.stride, self.s_min_ratio))
                next_world = self._prefetch_res.get()
                if next_world is None:
                    # 最後まで耐える：古いチャンクの残りを使う（最悪ケース）
                    pass
                else:
                    self.active_chunk_world = next_world
                    self.active_offset = 0

            # 新しい active に切り替えたら、すぐ次を先読み要求
            self._prefetch_req.queue.clear()  # 直近の観測で上書き
            self._prefetch_req.put((obs, self.active_chunk_world, self.stride, self.s_min_ratio))

        return np.stack(out_list, axis=0)



def main():
    print("Start Issac-GR00T")

    # TODO: 引数でいい感じに処理するようにする
    checkpoint_dir = "/home/veluga-g3/airoa/gr00t-chunk16-5hz"
    adopted_action_chunks = 15

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adopted_action_chunks: {adopted_action_chunks}")

    policy = Gr00tHSRPolicy(model_path=checkpoint_dir,adopted_action_chunks=adopted_action_chunks)

    for i in range(100):
        rand_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        policy_input = {
            "head_rgb": rand_img,
            "hand_rgb": rand_img,
            "joint_state": np.array([0.0 for _ in range(8)]),
            "instruction": "Test prompt. Do not move.",
        }
        action = policy.act(policy_input)
        print(action)

    import sys; sys.exit()
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
