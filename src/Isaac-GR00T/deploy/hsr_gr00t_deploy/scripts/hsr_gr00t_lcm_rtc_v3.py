#!/home/openpi/.venv/bin/python3
from collections import deque
import threading, queue, time
#!/usr/bin/env python3
from typing import Any
import copy
import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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

@dataclass
class Episode:
    head_rgb: np.ndarray        # (T, H, W, 3) uint8
    hand_rgb: np.ndarray        # (T, H, W, 3) uint8
    joint_state: np.ndarray     # (T, 8) float32
    instruction: str            # 文字列
    action_relative: np.ndarray # (T, 11) float32 (GT)

def l2_per_step(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    pred, gt: [H, D]
    return:   [H] 各時刻の L2
    """
    return np.linalg.norm((pred - gt), axis=-1)

def _flatten_dict(d, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(d, dict):
        for k,v in d.items():
            _flatten_dict(v, f"{prefix}{k}.", out)
    else:
        out[prefix[:-1]] = d
    return out

def _find_key(flat, candidates):
    for c in candidates:
        if c in flat:
            return c
    return None

def _find_by_substring(flat, include_words, exclude_words=()):
    # 候補: キー名に含まれる語でスコア
    scored = []
    for k,v in flat.items():
        if not isinstance(v, np.ndarray):
            continue
        shp = getattr(v, "shape", None)
        if shp is None:
            continue
        # 画像らしい形
        if len(shp) == 3 and shp[-1] in (1,3):
            name = k.lower()
            if any(w in name for w in include_words) and not any(w in name for w in exclude_words):
                scored.append(k)
    # 単純に最初を返す
    return scored[0] if scored else None

def load_episode_lerobot(dataset_root: str,
                         split: str = "train",
                         episode_index: int = 0,
                         head_key_hint: str|None = None,
                         hand_key_hint: str|None = None,
                         state_key_hint: str|None = None,
                         action_key_hint: str|None = None,
                         instr_key_hint: str|None = None):
    """
    LeRobotSingleDataset を使って 1 エピソード分をまとめて取り出す。
    - あなたの訓練時と同じ data_config/transform 系はデバッガ側の policy 構築で使うので、
      ここでは「生データ」をそのまま集めるだけ（正規化・pad はしない）。
    """
    from gr00t.data.dataset import LeRobotSingleDataset
    

    data_config = DATA_CONFIG_MAP["hsr_v2"]
    embodiment_tag = EmbodimentTag("new_embodiment")
    # ここは“生”が欲しいので transform=None 推奨（正規化・pad は policy 側に任せる）
    ds = LeRobotSingleDataset(
        dataset_path=dataset_root,
        modality_configs=data_config.modality_config(),
        transforms=None,
        embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
        video_backend="torchvision_av",
        sample_every_n=3,

    )

    if hasattr(ds, "episode_ends"):
        ends = np.asarray(ds.episode_ends, dtype=np.int64)
        starts = np.concatenate(([0], ends[:-1] + 1))
        start = int(starts[episode_index])
        end   = int(ends[episode_index]) + 1
        idxs = range(start, end)
    elif hasattr(ds, "index_by_episode"):
        idxs = list(ds.index_by_episode[episode_index])
    else:
        # フォールバック: 全部を 1 エピソードとみなす
        idxs = range(len(ds))
    


    # ---- まず 1 サンプルでキーを決め打ち ----
    flat0 = _flatten_dict(ds[idxs.start if hasattr(idxs, "start") else idxs[0]])


    # ---- エピソード境界の取得（実装差異に広めに対応）----
    if hasattr(ds, "episode_ends"):
        # 典型: starts/ends から範囲を切り出す
        ends = np.asarray(ds.episode_ends, dtype=np.int64)
        starts = np.concatenate(([0], ends[:-1] + 1))
        start = int(starts[episode_index])
        end   = int(ends[episode_index]) + 1
        indices = range(start, end)
    elif hasattr(ds, "index_by_episode"):
        # もう一つのパターン: 辞書/リストでフレーム index を持っている
        indices = list(ds.index_by_episode[episode_index])
    else:
        # フォールバック: 全体を一つのエピソードとみなす
        indices = range(len(ds))

    # 画像キー候補（よくある名前）
    HEAD_CAND = [
        "head_rgb", "rgb_head", "video.head", "observation.head_rgb",
        "observation.rgb_head", "observation.video.head"
    ]
    HAND_CAND = [
        "hand_rgb", "rgb_gripper", "video.hand", "observation.hand_rgb",
        "observation.rgb_gripper", "observation.video.hand"
    ]
    # 状態/アクション/指示
    STATE_CAND = [
        "joint_state", "state", "qpos",
        "observation.joint_state", "observation.state"
    ]
    ACTION_CAND = [
        "action.relative", "action", "actions", "observation.action.relative"
    ]
    INSTR_CAND = [
        "instruction", "text", "language", "task",
        "annotation.human.task_description"
    ]

    head_key = head_key_hint or _find_key(flat0, HEAD_CAND) \
               or _find_by_substring(flat0, include_words=("head","front"))
    hand_key = hand_key_hint or _find_key(flat0, HAND_CAND) \
               or _find_by_substring(flat0, include_words=("hand","wrist","gripper"))
    state_key = state_key_hint or _find_key(flat0, STATE_CAND)
    action_key = action_key_hint or _find_key(flat0, ACTION_CAND)
    instr_key = instr_key_hint or _find_key(flat0, INSTR_CAND)

    # 1つも見つからない時はプリントして中断（まずキー名を確認する）
    if head_key is None or hand_key is None:
        print("[DEBUG] sample keys:")
        for k,v in flat0.items():
            shp = getattr(v,"shape",None)
            print(f" - {k:60s} shape={shp} dtype={getattr(v,'dtype',None)}")
        raise RuntimeError(
            f"Failed to find camera keys. "
            f"Detected head_key={head_key}, hand_key={hand_key}. "
            f"Please pass head_key_hint/hand_key_hint."
        )
    if action_key is None:
        raise RuntimeError("Failed to find action key; pass action_key_hint.")

    head_list, hand_list, q_list, act_list = [], [], [], []
    instruction = None

    for i in idxs:
        s = ds[i]
        flat = _flatten_dict(s)

        head = flat.get(head_key, None)[0]
        hand = flat.get(hand_key, None)[0]
        state_arm = flat.get('state.arm', None)
        state_gripper = flat.get('state.gripper', None)
        state_head = flat.get('state.head', None)
        state = np.concatenate([state_arm, state_gripper, state_head], axis=1)[0]
        action = flat.get(action_key, None)[0]
        instr = flat.get(instr_key, None)[0] if instr_key is not None else None

        if head is not None:
            h = np.asarray(head)
            if h.ndim == 3 and h.shape[-1] in (1,3):
                head_list.append(h.astype(np.uint8))
        if hand is not None:
            h = np.asarray(hand)
            if h.ndim == 3 and h.shape[-1] in (1,3):
                hand_list.append(h.astype(np.uint8))

        if state is not None:
            s = np.asarray(state).astype(np.float32)
            # (8,) を想定: arm5 + gripper1 + head2
            if s.ndim == 1 and s.shape[0] >= 8:
                q = np.concatenate([s[:5], s[5:6], s[6:8]], axis=0)
                q_list.append(q)
            elif s.ndim == 2 and s.shape[-1] >= 8:
                q = np.concatenate([s[:, :5], s[:, 5:6], s[:, 6:8]], axis=1)
                if q.shape[0] == 1:
                    q = q[0]
                q_list.append(q.astype(np.float32))

        if action is not None:
            a = np.asarray(action).astype(np.float32)
            # (11,) or (1,11) を想定
            if a.ndim == 2 and a.shape[0] == 1:
                a = a[0]
            if a.ndim == 1 and a.shape[0] >= 11:
                act_list.append(a[:11])

        if instruction is None:
            instruction = instr if isinstance(instr, str) else instr.decode("utf-8","ignore")

    # 最低限の整合
    T = min(len(head_list), len(hand_list), len(q_list), len(act_list))
    if T == 0:
        # まだダメなら、キー名の当たりが外れている
        print("[DEBUG] First-sample flattened keys again:")
        for k,v in flat0.items():
            shp = getattr(v,"shape",None)
            print(f" - {k:60s} shape={shp} dtype={getattr(v,'dtype',None)}")

    head = np.stack(head_list[:T], axis=0)  # (T,H,W,3)
    hand = np.stack(hand_list[:T], axis=0)
    qpos = np.stack(q_list[:T], axis=0).astype(np.float32)
    act  = np.stack(act_list[:T], axis=0).astype(np.float32)
    if instruction is None:
        instruction = "Follow the task."

    return Episode(
        head_rgb=head, hand_rgb=hand, joint_state=qpos,
        instruction=instruction, action_relative=act
    )
    

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
        adopted_action_chunks: int = 31,
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
        self.frame_num = 0

        self.use_temp_ensem = True          # 無効にしたい時は False
        self.temporal_half_life = 8        # フレーム半減期(=約10ステップで重み半減)
        self._ema_action = None

        # ダブルバッファ & 進行状況
        self.active_chunk_world = None     # [H, 11] 非正規化
        self.next_chunk_world   = None     # [H, 11] 非正規化
        # self.active_offset      = 0        # active 何ステップ消費したか
        self.stride             = num_traj # = 再計画ストライド
        # self.exec_since_prev    = 0        # RTC 用 d（= 通常 stride）

        # 先読みワーカー
        # self._prefetch_req = queue.Queue(maxsize=1)
        # self._prefetch_res = queue.Queue(maxsize=1)
        # self._stop = threading.Event()
        # self._worker = threading.Thread(target=self._prefetch_loop, daemon=True)
        # self._worker.start()

        # RTCハイパラ
        # self.s_min_ratio = 0.5
        # self.mask_lambda = 0.3
        

        self.H = self.policy.model.action_head.action_horizon
        self.D_model = self.policy.model.action_head.action_dim 
        self.prev_chunk_world: Optional[np.ndarray] = None
        d_steps = 0
        s_steps = 4
        lam = 0.0
        self.rtc_beta = 0.05
        self.rtc_guidance_clip = 0.5

        self.W = self.build_rtc_weight_mask(self.H, d_steps, s_steps, lam=lam,
                                          B=1, D=self.D_model,  # 内部で最終的に 32 にパディングされるが、policy側で合わせる実装にしていればOK
                                          device="cuda",
                                          dtype=self.policy.dtype if hasattr(self.policy, "dtype") else torch.float32)

        print(f"[info] horizon H={self.H}, action_dim (head)={self.D_model}")

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

    def build_rtc_weight_mask(self, H: int, d_steps: int, s_steps: int, lam: float,
                        B: int = 1, D: int = 11,
                        device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        W: [B, H, D]
        先頭 d: 1.0（ハード凍結）
        中間: λ 付き指数減衰
        末尾 s: 0.0（自由）
        """
        i = torch.arange(H, device=device, dtype=torch.float32)
        d = int(d_steps)
        s = int(s_steps)

        mid_len = H - d - s
        if mid_len <= 0:
            # 中間区間なし：先頭を1.0、末尾を0.0
            W_time = torch.where(i < d, torch.ones_like(i), torch.zeros_like(i))
        else:
            den = float(mid_len)  # c(d)=1, c(H-s)=0 になるよう正規化
            c = (H - s - i) / den
            c = torch.clamp(c, 0.0, 1.0)

            if lam == 0.0:
                # lim_{lam->0} (exp(lam c)-1)/(exp(lam)-1) = c
                w_mid = c
            else:
                lam_t = torch.tensor(lam, device=device, dtype=torch.float32)
                w_mid = torch.expm1(lam_t * c) / torch.expm1(lam_t)

            W_time = torch.where(i < d, torch.ones_like(i),
                    torch.where(i < H - s, w_mid, torch.zeros_like(i)))
        
        W_time[0] *= 0.7

        W = W_time.view(1, H, 1).expand(B, H, D).to(dtype)
        return W

    def shift_left(self, arr: np.ndarray, k: int = 1, fill: str = "zero"):
        out = np.empty_like(arr)
        out[:-k] = arr[k:]
        if fill == "repeat":
            out[-k:] = arr[-1:]
        else:  # "zero"
            out[-k:] = 0.0
        return out


    
    def reset_buffer(self):
        self.action_queue.clear()
        

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        

        # # 1) 初回は同期ブートストラップ（遅延1回だけ許容）
        # if self.active_chunk_world is None:
        #     # 同期で作る（d=0）
        #     self._prefetch_req.queue.clear()
        #     self._prefetch_res.queue.clear()
        #     self._prefetch_req.put((obs, None, 0, self.s_min_ratio))
        #     next_world = self._prefetch_res.get()
        #     if next_world is None:
        #         raise RuntimeError("RTC bootstrap failed")
        #     self.active_chunk_world = next_world
        #     self.active_offset = 0
        #     # 次をすぐ先読み（切替時用に d=stride で）
        #     self._prefetch_req.put((obs, self.active_chunk_world, self.stride, self.s_min_ratio))

        # 2) 現在のチャンクから stride 分だけ取り出し
        # start = self.active_offset
        # end   = min(H, start + self.stride)
        # chunk_rel = self.active_chunk_world[start:end]  # [stride<=, 11]
        # self.active_offset = end

        # # 3) 出力の11次元→絶対値復元（あなたの元の処理）
        # out_list = []
        # for a_rel in chunk_rel:
        #     a_11 = np.concatenate([a_rel[0:5], [a_rel[5]], a_rel[6:8], a_rel[8:11]])
        #     a_abs = a_11 + np.concatenate([obs["joint_state"][:5],
        #                                    np.array([0]),
        #                                    obs["joint_state"][6:8],
        #                                    np.array([0,0,0])])
        #     out_list.append(a_abs)

        # 4) 切替境界に近づいたら、バックグラウンド結果を回収してスワップ
        #    境界: 残り < stride になったら切替
        # remaining = H - self.active_offset
        # if remaining < self.stride:
        #     try:
        #         # 用意できていれば使う／なければフォールバックで同期生成
        #         next_world = self._prefetch_res.get_nowait()
        #         if next_world is None:
        #             raise queue.Empty
        #         self.active_chunk_world = next_world
        #         self.active_offset = 0
        #     except queue.Empty:
        #         # フォールバック：同期で作る（観測 obs、前チャンク active を prev として）
        #         self._prefetch_req.put((obs, self.active_chunk_world, self.stride, self.s_min_ratio))
        #         next_world = self._prefetch_res.get()
        #         if next_world is None:
        #             # 最後まで耐える：古いチャンクの残りを使う（最悪ケース）
        #             pass
        #         else:
        #             self.active_chunk_world = next_world
        #             self.active_offset = 0

        #     # 新しい active に切り替えたら、すぐ次を先読み要求
        #     self._prefetch_req.queue.clear()  # 直近の観測で上書き
        #     self._prefetch_req.put((obs, self.active_chunk_world, self.stride, self.s_min_ratio))

        # return np.stack(out_list, axis=0)
        if self.prev_chunk_world is None:
            rtc_prev = None
        else:
            rtc_prev = self.prev_chunk_world

        print("=== Gr00tHSRPolicy: Getting action from policy ===")
        # image shapeは(480, 640, 3) → (1, 480, 640, 3)
        video_head = np.expand_dims(obs["head_rgb"], axis=0)
        video_hand = np.expand_dims(obs["hand_rgb"], axis=0)
        state_arm = np.expand_dims(obs["joint_state"][:5], axis=0)  # armの状態
        state_hand = np.expand_dims(np.expand_dims(obs["joint_state"][5], axis=0), axis=0)  # handの状態
        state_head = np.expand_dims(obs["joint_state"][6:8], axis=0)  # headの状態
        instruction = [obs["instruction"]]  # タスクの説明
        policy_input = {
            "video.head": video_head,
            "video.hand": video_hand,
            "state.arm" : state_arm,  # armの状態
            "state.gripper": state_hand,  # handの状態
            "state.head": state_head,
            "annotation.human.task_description": instruction,  # タスクの説明
        }
        rtc = {
            "rtc_prev_action": rtc_prev,
            "rtc_weight_mask": self.W,             # [1,H,D] or [1,H,1] を policy 側で合わせる実装にしてあるはず
            "rtc_beta": float(self.rtc_beta),
            "rtc_guidance_clip": float(self.rtc_guidance_clip),
            # "rtc_angle_indices": angle_idx,  # 必要なら policy 側のwrapに渡す
            "rtc_dim_mask": torch.cat([torch.ones(11), torch.zeros(21)])[None,None,:], 
        }
        action_chunk = self.policy.get_action_rtc(policy_input, rtc)
        pred_chunk = action_chunk["action.relative"]

        A_prev = pred_chunk[self.stride:]                             # 左シフト
        A_prev = np.pad(A_prev, ((0, self.stride), (0, 0)))
        #prev_chunk_world = (gt_chunk.copy() if use_gt_prev else pred_chunk.copy())
        self.prev_chunk_world = A_prev.copy()

        actions = []

        self.action_queue["action.relative"].extend(pred_chunk[self.num_traj:self.adopted_action_chunks])

        for i in range(self.num_traj)
            action_relative = pred_chunk[i]
            action = np.concatenate(
                [
                    action_relative[0:5],
                    [action_relative[5]],
                    action_relative[6:8],
                    action_relative[8:11],
                ]
            )

            # 差分になっている行動を元に戻す
            action = action + np.concatenate(
                [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
            )
            actions.append(action)

        return actions

def angle_wrap_inplace(actions: np.ndarray, angle_indices: Optional[list[int]]):
    """
    必要なら角度成分を [-pi, pi] にwrap（オフライン評価の整合取り）
    actions: [*, D]
    """
    if not angle_indices:
        return
    for idx in angle_indices:
        actions[..., idx] = np.arctan2(np.sin(actions[..., idx]), np.cos(actions[..., idx]))



def main():
    print("Start Issac-GR00T")

    # TODO: 引数でいい感じに処理するようにする
    checkpoint_dir = "/home/group_25b505/group_6/workspace/user_00031_25b505/Isaac-GR00T/ckpt-microwave-open_the_door_chunk16_5hz_v2/checkpoint-10000"
    adopted_action_chunks = 15

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adopted_action_chunks: {adopted_action_chunks}")

    policy = Gr00tHSRPolicy(model_path=checkpoint_dir,adopted_action_chunks=adopted_action_chunks)

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
