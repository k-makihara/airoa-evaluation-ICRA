#!/home/openpi/.venv/bin/python3
from collections import deque

#!/usr/bin/env python3
from typing import Any

import cv2
import numpy as np

# gr00t関連
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.eval.rtc_wrapper import RTCPolicyWrapper

from gr00t.model.policy import Gr00tPolicy
import torch
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
    GRIPPER_CLOSE_THRESHOLD = 0.1  # グリッパーを閉じる閾値

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
        num_traj: int = 1,
        use_temp_ensem: bool = False
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

        #replay_episode = np.load("/home/veluga-g3/Downloads/episode_1511_action_relative.npz", allow_pickle=False)
        #self.actions_rel = replay_episode["actions"]
        self.frame_num = 0

        self.use_temp_ensem = use_temp_ensem          # 無効にしたい時は False
        self.temporal_half_life = 8        # フレーム半減期(=約10ステップで重み半減)
        self._ema_action = None

    
        self.max_timesteps = 10000
        self.num_queries = adopted_action_chunks
        self.action_dim = 11
        if self.use_temp_ensem:
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.action_dim]).cuda()
        self.t = 0

        # for RTC configuration
        control_freq = 20
        denoising_steps = 4
        max_rtc_overlap_factor = 0.75
        self.rtc_policy = RTCPolicyWrapper(
            self.policy, control_freq, denoising_steps, max_rtc_overlap_factor
        )

    
    def reset_buffer(self):
        self.action_queue.clear()
        self.t = 0

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        obs: Dict[str, Any]
            センサ情報
            {
                "head_rgb": <np.ndarray shape (H, W, 3)>,
                "hand_rgb": <np.ndarray shape (H, W, 3)>,
                "joint_state": <np.ndarray shape (8,)>, # ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint"]
                "instruction": <str>,
            }
        return: np.ndarray : shape (11,)
            アクション
            [
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
        """
        if not self.use_temp_ensem:
            if len(self.action_queue["action.relative"]) >= self.num_traj:
                actions = []
                for _ in range(self.num_traj):
                    action_relative = self.action_queue["action.relative"].popleft()
                    #action_relative = self.actions_rel[self.frame_num]
                    action = np.concatenate(
                        [
                            action_relative[0:5],
                            [action_relative[5]],
                            #action_relative[7:9],
                            #action_relative[9:12],
                            #action_relative[6:8],
                            [0.0, 0.0],
                            action_relative[8:11],
                        ]
                    )
                    self.frame_num += 1

                    action = action + np.concatenate(
                        [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
                    )
                    actions.append(action)
                return np.stack(actions)


        print("=== Gr00tHSRPolicy: Getting action from policy ===")
        # image shapeは(480, 640, 3) → (1, 480, 640, 3)
        video_head = np.expand_dims(obs["head_rgb"], axis=0)
        video_hand = np.expand_dims(obs["hand_rgb"], axis=0)
        state_arm = np.expand_dims(obs["joint_state"][:5], axis=0)  # armの状態
        state_hand = np.expand_dims(np.expand_dims(obs["joint_state"][5], axis=0), axis=0)  # handの状態
        state_head = np.expand_dims(obs["joint_state"][6:8], axis=0)  # headの状態
        #print(state_hand)
        instruction = [obs["instruction"]]  # タスクの説明
        policy_input = {
            "video.head": video_head,
            "video.hand": video_hand,
            "state.arm" : state_arm,  # armの状態
            "state.gripper": state_hand,  # handの状態
            "state.head": state_head,
            "annotation.human.task_description": instruction,  # タスクの説明
        }
        action_chunk = self.rtc_policy.get_action(policy_input)
        #print(action_chunk["action.relative"])
        
        actions = []
        actions_notemp = []
        for i in range(len(action_chunk["action.relative"])):
            action_relative = action_chunk["action.relative"][i]
            action = np.concatenate(
                [
                    action_relative[0:5],
                    [action_relative[5]],
                    #[0.5],
                    #action_relative[6:8],
                    [0.0, 0.0],
                    action_relative[8:11],
                ]
            )

            # 差分になっている行動を元に戻す
            action = action + np.concatenate(
                [obs["joint_state"][:5], np.array([0.0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
            )

            actions.append(action)

        if self.use_temp_ensem:
            self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = torch.tensor(actions, dtype=torch.float32).cuda()
            actions_for_curr_step = self.all_time_actions[:, self.t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            #raw_action[5] = 1.2
            #print(raw_action)
            self.t += 1

            return raw_action
        else:
            self.action_queue["action.relative"].extend(action_chunk["action.relative"][self.num_traj:self.adopted_action_chunks])
            #action_relative = self.action_queue["action.relative"].popleft()
            action_relative = self.action_queue["action.relative"][0]
            #action_relative = self.actions_rel[self.frame_num]
            action_notemp = np.concatenate(
                [
                    action_relative[0:5],
                    [action_relative[5]],
                    #[1.2],
                    #action_relative[7:9],
                    #action_relative[9:12],
                    #action_relative[6:8],
                    [0.0, 0.0],
                    action_relative[8:11],
                ]
            )
            # 差分になっている行動を元に戻す
            action_notemp = action_notemp + np.concatenate(
                [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
            )
            #actions_notemp.append(action)
            actions_notemp.append(action_notemp)

            #return np.array(actions)
            return np.array(actions_notemp)

        


def main():
    print("Start Issac-GR00T")

    # TODO: 引数でいい感じに処理するようにする
    checkpoint_dir = "s3://airoa-fm-development-competition/group6/st2-mid-checkpoint"
    adopted_action_chunks = 32
    use_temp_ensem = False

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adopted_action_chunks: {adopted_action_chunks}")

    policy = Gr00tHSRPolicy(model_path=checkpoint_dir,adopted_action_chunks=adopted_action_chunks,use_temp_ensem=use_temp_ensem)

    # for i in range(20):
    #     rand_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    #     policy_input = {
    #         "head_rgb": rand_img,
    #         "hand_rgb": rand_img,
    #         "joint_state": np.array([0.0 for _ in range(8)]),
    #         "instruction": "Test prompt. Do not move.",
    #     }
    #     action = policy.act(policy_input)
    #     #print(action)

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
