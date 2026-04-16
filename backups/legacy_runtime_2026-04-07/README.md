Legacy runtime snapshot saved on 2026-04-07 before attempting a newer GPU-compatible server runtime.

Included files:
- `RUN-DOCKER-CONTAINER.sh`
- `docker-compose.yml`
- `server/Dockerfile`
- `server/entrypoint.sh`
- `server/serve_hsr_policy_ws.py`
- `src/Isaac-GR00T/gr00t/model/backbone/eagle_backbone.py`
- `src/Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/config.json`
- `src/Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/configuration_eagle2_5_vl.py`
- `src/Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py`
- `src/Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/radio_model.py`
- `src/Isaac-GR00T/gr00t/experiment/data_config.py`
- `src/Isaac-GR00T/gr00t/data/transform/video.py`
- `src/Isaac-GR00T/gr00t/utils/video.py`

This snapshot preserves the current fallback-based runtime behavior while a new GPU-targeted variant is developed.
