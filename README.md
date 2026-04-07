# airoa-evaluation-ICRA

Participant evaluation runtime for ICRA 2026 VLA Workshop Competition.

## 1. Editable Scope

Main implementation targets:

- `server/`
- `src/`

## 2. Host Requirements

- Linux
- Docker Engine
- Docker Compose v2
- NVIDIA driver
- NVIDIA Container Toolkit

```bash
docker --version
docker compose version
nvidia-smi
```

Verified environment (2026-02-20):

- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GeForce RTX 5070 Ti
- NVIDIA driver: 580.126.09
- Docker: 29.0.1
- Docker Compose: v2.40.3

## 3. Implementation Workflow

1. Create a feature branch from the prepared base branch.
2. Add your model repository under `src/`.
3. Update `server/serve_hsr_policy_ws.py` and server-side dependencies for your model runtime.
4. Add an adapter in `src/` or `server/`.

Adapter definition:
An adapter is a thin conversion layer that maps between the HSR client contract and your model's native
input/output format. For required fields and shapes, just follow the `WebSocket I/O Contract` section below.

5. Run the test flow and confirm `Action executed.` appears in logs.
6. Submit your branch.

## 4. Required Environment Variables

```bash
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
```

## 5. Optional Policy-Specific Variables

Set policy-specific variables only when your server implementation requires them.

GR00T example:

```bash
export GR00T_DATA_CONFIG=hsr_v2
export GR00T_EMBODIMENT_TAG=new_embodiment
export GR00T_DEVICE=cuda
export GR00T_ADOPTED_ACTION_CHUNKS=32
export GR00T_DENOISING_STEPS=4
export GR00T_CONTROL_FREQ=20
export GR00T_MAX_RTC_OVERLAP_FACTOR=0.75
```

## 6. Test Flow

Start containers:

```bash
./RUN-DOCKER-CONTAINER.sh up
```

Enter client shell:

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

Run launch inside the container:

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

By default, `test_mode` is `true`. In this mode, the client uses synthetic random observations
(`head_rgb`, `hand_rgb`, and `state`) in an infinite loop and prints language/action logs.

## 7. Logs and Stop

```bash
./RUN-DOCKER-CONTAINER.sh logs policy_server
./RUN-DOCKER-CONTAINER.sh down
```

## 8. WebSocket I/O Contract

Inference request fields:

- `head_rgb`: image array `(H, W, 3)` (current HSR dataset profile: `(480, 640, 3)`)
- `hand_rgb`: image array `(H, W, 3)` (current HSR dataset profile: `(480, 640, 3)`)
- `state`: `(8,)`
- `prompt`: `str`

Inference response field:

- `actions` with shape `(T, 11)`, `T >= 1`

Action order:

- `[arm_lift_joint, arm_flex_joint, arm_roll_joint, wrist_flex_joint, wrist_roll_joint, gripper, head_pan_joint, head_tilt_joint, base_x, base_y, base_t]`

Value requirement:

- finite numeric values only
