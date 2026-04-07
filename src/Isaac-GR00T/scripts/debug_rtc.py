#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
debug_rtc_offline.py
1エピソードをリプレイしながら、RTC（ΠGDMガイダンス）をON/OFFで比較するオフライン・デバッガ。

前提:
- あなたのGR00Tコードベースに get_action() は既に動作
- get_action_rtc(observations, rtc_dict) も実装済み（前に話した正規化・Wマスク・ΠGDMなど）
- metadata.json に action.relative(11次元) の統計が入っている（既にOKとのこと）

エピソードの想定構造（必要に応じて下の `load_episode()` を編集）:
  npz ファイル例:
    - head_rgb: (T, H, W, 3)  uint8
    - hand_rgb: (T, H, W, 3)  uint8
    - joint_state: (T, 8)     float32  # arm5, gripper1, head2
    - instruction: () or (T,) str     # 単一文字列でOK
    - action_relative: (T, 11) float32  # 学習時の相対アクション（GT）

実行例:
    python debug_rtc_offline.py \
      --ckpt /path/to/checkpoint \
      --episode /path/to/episode_XXXX.npz \
      --num-steps 300 \
      --use-rtc 1 \
      --d-steps 0 \
      --s-ratio 0.5 \
      --beta 5.0 \
      --clip 1.0 \
      --lam 0.3
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import math

# === あなたの環境の import に合わせて ===
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.schema import EmbodimentTag


# --------------------------
# ローディング
# --------------------------
@dataclass
class Episode:
    head_rgb: np.ndarray        # (T, H, W, 3) uint8
    hand_rgb: np.ndarray        # (T, H, W, 3) uint8
    joint_state: np.ndarray     # (T, 8) float32
    instruction: str            # 文字列
    action_relative: np.ndarray # (T, 11) float32 (GT)

def load_episode(path: str) -> Episode:
    data = np.load(path, allow_pickle=True)
    # キー名は環境に合わせて調整してください
    head = data["head_rgb"]             # (T, H, W, 3)
    hand = data["hand_rgb"]             # (T, H, W, 3)
    joint = data["joint_state"]         # (T, 8)
    gt_act = data["action_relative"]    # (T, 11)
    # instruction がない場合は固定文字列でOK
    instr = data["instruction"].item() if "instruction" in data else "Follow the task."
    return Episode(head_rgb=head, hand_rgb=hand, joint_state=joint, instruction=instr, action_relative=gt_act)

def _pick(d, candidates):
    """辞書 d から候補キーのどれかを抜くユーティリティ（最初に見つかったものを返す）"""
    for k in candidates:
        if k in d:
            return d[k]
    return None

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
# --------------------------
# マスク（W）のユーティリティ
# --------------------------
def build_rtc_weight_mask(H: int, d_steps: int, s_steps: int, lam: float,
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


# --------------------------
# 評価（誤差等）
# --------------------------
def l2_per_step(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    pred, gt: [H, D]
    return:   [H] 各時刻の L2
    """
    return np.linalg.norm((pred - gt), axis=-1)


def angle_wrap_inplace(actions: np.ndarray, angle_indices: Optional[list[int]]):
    """
    必要なら角度成分を [-pi, pi] にwrap（オフライン評価の整合取り）
    actions: [*, D]
    """
    if not angle_indices:
        return
    for idx in angle_indices:
        actions[..., idx] = np.arctan2(np.sin(actions[..., idx]), np.cos(actions[..., idx]))

def shift_left(arr: np.ndarray, k: int = 1, fill: str = "zero"):
    out = np.empty_like(arr)
    out[:-k] = arr[k:]
    if fill == "repeat":
        out[-k:] = arr[-1:]
    else:  # "zero"
        out[-k:] = 0.0
    return out

# --------------------------
# メインのオフライン・リプレイ
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episode", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-steps", type=int, default=300)
    ap.add_argument("--use-rtc", type=int, default=1)      # 1: ΠGDM guidance 使用, 0: 通常 get_action
    ap.add_argument("--d-steps", type=int, default=0)
    ap.add_argument("--s-steps", type=int, default=16)  # s_steps = int(H * s_ratio)
    ap.add_argument("--beta", type=float, default=5.0)     # ΠGDM 係数
    ap.add_argument("--clip", type=float, default=1.0)     # guidance clip
    ap.add_argument("--lam", type=float, default=0.3)      # マスクの指数減衰係数
    ap.add_argument("--use-gt-prev", type=int, default=0)  # 1: 前チャンクにGTを使って誘導, 0: 直前予測を使う
    ap.add_argument("--angle-idx", type=str, default="")   # 角度次元のカンマ区切り例: "10" など
    args = ap.parse_args()

    angle_idx = [int(x) for x in args.angle_idx.split(",")] if args.angle_idx else []

    # 1) Episode 読み込み
    #ep = load_episode(args.episode)
    ep = load_episode_lerobot(
        args.episode, split="train", episode_index=0,
        head_key_hint="video.head",     # 例: 実キー名に合わせる
        hand_key_hint="video.hand",
        state_key_hint="state.arm",
        action_key_hint="action.relative",
        instr_key_hint="annotation.human.task_description"
    )

    T = ep.head_rgb.shape[0]

    # 2) Policy 準備（学習時の transform を使用）
    data_config = DATA_CONFIG_MAP["hsr_v2"]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.ckpt,
        modality_config=modality_config,
        modality_transform=transforms,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device=args.device,
    )
    model = policy.model
    H = model.action_head.action_horizon
    D_model = model.action_head.action_dim  # = 11 → 内部で pad して 32 へ（transform/ActionHead側）

    print(f"[info] horizon H={H}, action_dim (head)={D_model}, steps={args.num_steps}")

    # 3) ループ設定
    t = 0
    steps = min(args.num_steps, T - H - 1)
    use_rtc = bool(args.use_rtc)
    use_gt_prev = bool(args.use_gt_prev)

    # 前チャンク（“物理空間=非正規化”）の保存（RTCで使う）
    prev_chunk_world: Optional[np.ndarray] = None

    all_err_no_rtc = []
    all_err_rtc = []

    # 4) 先に「RTCなし」のベースラインも取りたい場合はここで切り替え可能
    #    ここでは1本のループで use_rtc=0/1 を切り替えながらも計測できるように2本走らせる

    def one_pass(use_rtc_flag: bool, beta=0.0, d_step=0, s_step=16, lam=0, stride=1) -> Tuple[np.ndarray, np.ndarray]:
        nonlocal t, prev_chunk_world

        t = 0
        prev_chunk_world = None
        per_step_errs = []    # [step] → 1 float (実際は今ステップで消費する d=0 の位置の誤差)
        mean_chunk_errs = []  # 各チャンクの L2 平均（H 区間）
        s = stride
        for step in range(0, steps, s):
            # 観測を1ステップ分作る（オンライン実装と同じ形）
            obs = {
                "head_rgb": ep.head_rgb[t],
                "hand_rgb": ep.hand_rgb[t],
                "joint_state": ep.joint_state[t],
                "instruction": ep.instruction,
            }

            # 目標（比較用）：次のチャンクのGT
            gt_chunk = ep.action_relative[t : t + H].copy()  # (H, 11)
            angle_wrap_inplace(gt_chunk, angle_idx)

            if not use_rtc_flag:
                # --- 普通の get_action で1チャンク出す ---
                pi_in = {
                    "video.head": np.expand_dims(obs["head_rgb"], axis=0),
                    "video.hand": np.expand_dims(obs["hand_rgb"], axis=0),
                    "state.arm":  np.expand_dims(obs["joint_state"][:5], axis=0),
                    "state.gripper": np.expand_dims(np.expand_dims(obs["joint_state"][5], axis=0), axis=0),
                    "state.head": np.expand_dims(obs["joint_state"][6:8], axis=0),
                    "annotation.human.task_description": [obs["instruction"]],
                }
                out = policy.get_action(pi_in)  # dict: {"action.relative": (H,11)}
                pred_chunk = out["action.relative"]
            else:
                # --- RTC（ΠGDM）有り ---
                d_steps = int(d_step)
                s_steps = int(s_step)
                W = build_rtc_weight_mask(H, d_steps, s_steps, lam=lam,
                                          B=1, D=D_model,  # 内部で最終的に 32 にパディングされるが、policy側で合わせる実装にしていればOK
                                          device=args.device,
                                          dtype=model.dtype if hasattr(model, "dtype") else torch.float32)

                # prev_chunk_world をどう作るか:
                #   use_gt_prev=1 → 直前の GT チャンクで誘導（理想的に合うはず）
                #   use_gt_prev=0 → 直前の予測を渡して、実運用に近い形で誘導
                # if prev_chunk_world is None:
                #     rtc_prev = None
                # else:
                #     rtc_prev = prev_chunk_world  # (H, 11) の numpy を想定（policy.get_action_rtc 内で正規化→pad）

                if prev_chunk_world is None:
                    if use_gt_prev:
                        rtc_prev = shift_left(gt_chunk, k=1, fill="zero")  # ★ 初回GTも左シフトして渡す
                    else:
                        rtc_prev = None
                else:
                    rtc_prev = prev_chunk_world

                # policy 入力
                pi_in = {
                    "video.head": np.expand_dims(obs["head_rgb"], axis=0),
                    "video.hand": np.expand_dims(obs["hand_rgb"], axis=0),
                    "state.arm":  np.expand_dims(obs["joint_state"][:5], axis=0),
                    "state.gripper": np.expand_dims(np.expand_dims(obs["joint_state"][5], axis=0), axis=0),
                    "state.head": np.expand_dims(obs["joint_state"][6:8], axis=0),
                    "annotation.human.task_description": [obs["instruction"]],
                }
                rtc = {
                    "rtc_prev_action": (gt_chunk if (use_gt_prev and prev_chunk_world is None) else rtc_prev),
                    "rtc_weight_mask": W,             # [1,H,D] or [1,H,1] を policy 側で合わせる実装にしてあるはず
                    "rtc_beta": float(beta),
                    "rtc_guidance_clip": float(args.clip),
                    # "rtc_angle_indices": angle_idx,  # 必要なら policy 側のwrapに渡す
                    "rtc_dim_mask": torch.cat([torch.ones(11), torch.zeros(21)])[None,None,:], 
                }
                
                out = policy.get_action_rtc(pi_in, rtc)
                pred_chunk = out["action.relative"]

            # 誤差
            # ここでは「チャンク全体」と「0番目(=今フレームで実際に使う箇所)」の2種類を記録
            angle_wrap_inplace(pred_chunk, angle_idx)
            l2_all = l2_per_step(pred_chunk, gt_chunk)  # [H]
            #per_step_errs.append(float(l2_all[0]))      # 今フレームで実際に使う最初の1ステップ
            per_step_errs.append(float(l2_all[:s].mean()))
            mean_chunk_errs.append(float(l2_all.mean()))

            # 次のループの prev を更新
            A_prev = pred_chunk[s:]                             # 左シフト
            A_prev = np.pad(A_prev, ((0, s), (0, 0)))
            #prev_chunk_world = (gt_chunk.copy() if use_gt_prev else pred_chunk.copy())
            prev_chunk_world = (gt_chunk.copy() if use_gt_prev else A_prev.copy())

            # 時刻を進める（ここでは stride=1。実機では num_traj 分だけポップするのが理想）
            t += s
            if t + H >= T:
                break

        return np.array(per_step_errs), np.array(mean_chunk_errs)

    # ベースライン（RTCなし）
    # strides = [1,2,4,8,16]
    # for stride in strides:
    #     print(f"stride={stride}")
    #     errs0, chunk_errs0 = one_pass(use_rtc_flag=False, stride=stride)
    #     print(f"[no-RTC] per-step L2: mean={errs0.mean():.4f}  median={np.median(errs0):.4f}  n={len(errs0)}")
    #     print(f"[no-RTC] per-chunk L2: mean={chunk_errs0.mean():.4f} n={len(chunk_errs0)}")

    # RTCあり
    betas = [0.05, 0.1, 0.2]
    #betas = [0.25]
    d_steps = [0, 1, 2, 4]
    s_steps = [4, 8, 16]
    lams = [0, 0.5, 1.0]
    strides = [1,2,4,8,16]
    for beta in betas:
        for d_step in d_steps:
            for s_step in s_steps:
                for lam in lams:
                    #for stride in strides:
                    for stride in range(1):
                        #print(f"Beta={beta}, d_step={d_step}, s_step={s_step}, lam={lam}, stride={stride}")
                        print(f"Beta={beta}, d_step={d_step}, s_step={s_step}, lam={lam}, stride={s_step}")
                        #errs1, chunk_errs1 = one_pass(use_rtc_flag=True, beta=beta, d_step=d_step, s_step=s_step, lam=lam, stride=stride)
                        errs1, chunk_errs1 = one_pass(use_rtc_flag=True, beta=beta, d_step=d_step, s_step=s_step, lam=lam, stride=s_step)
                        print(f"[RTC]    per-step L2: mean={errs1.mean():.4f}  median={np.median(errs1):.4f}  n={len(errs1)}")
                        print(f"[RTC]    per-chunk L2: mean={chunk_errs1.mean():.4f} n={len(chunk_errs1)}")

    # 簡易レポート
    # delta = errs0.mean() - errs1.mean()
    # print(f"[diff] per-step L2 improvement (no-RTC -> RTC): {delta:+.4f}")


if __name__ == "__main__":
    main()
