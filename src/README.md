# src

Place your model implementation under this directory.

cd /path/to/airoa-evaluation-ICRA
export POLICY_CHECKPOINT_PATH=/path/to/ckpt
export GR00T_DATA_CONFIG=hsr_v2
export GR00T_EMBODIMENT_TAG=new_embodiment
export GR00T_DEVICE=cuda
export GR00T_ADOPTED_ACTION_CHUNKS=16
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh logs policy_server