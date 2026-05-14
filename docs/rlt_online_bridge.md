# openpi-RLT ManiSkill Bridge

This bridge keeps the openpi-RLT pipeline intact and only adapts ManiSkill to its runtime protocol.

## Processes

Start the RLT feature server in the openpi-RLT environment:

```bash
cd third_party/openpi-RLT
uv run scripts/serve_rlt_policy.py \
  --config rlt_pi05_libero \
  --checkpoint checkpoints/rlt_pi05_libero/<exp>/<step> \
  --repo-id local/maniskill_myws_turn_globe_valve \
  --port 8000
```

Optionally start the online RL actor and replay services from the openpi-RLT runtime:

```bash
cd third_party/openpi-RLT/rlt_online_rl
python scripts/run_online_rl.py --config configs/tasks/agilex_ethernet/online_rl.yaml --system.role actor_service
python scripts/run_online_rl.py --config configs/tasks/agilex_ethernet/online_rl.yaml --system.role replay_manager
python scripts/run_online_rl.py --config configs/tasks/agilex_ethernet/online_rl.yaml --system.role learner_service
```

Then run ManiSkill:

```bash
python scripts/rlt/run_rlt_remote.py \
  --feature-server ws://127.0.0.1:8000 \
  --actor-url http://127.0.0.1:9101 \
  --replay-url http://127.0.0.1:9102 \
  --env-id TurnGlobeValve-v1 \
  --obs-mode rgb \
  --reward-mode dense
```

If `--actor-url` is omitted, the bridge executes `ref_chunk` directly. If the actor service has no snapshot yet, openpi-RLT falls back to `ref_chunk`; this is useful for warmup collection.

## Data Contract

The feature server response is consumed exactly as openpi-RLT expects:

- `z_rl`: compact RLToken feature.
- `proprio`: proprioceptive vector.
- `ref_chunk`: VLA reference action chunk.

The actor service request follows openpi-RLT `ActorRequest`:

- `z_rl`
- `proprio`
- `ref_chunk`
- `request_id`
- `episode_id`
- `step_id`
- `deterministic`

Replay transitions are sent to `/add` using the openpi-RLT `RLTTransition` field names:

- `z_rl`, `proprio`, `ref_chunk`
- `action_chunk`, `rewards`, `done`
- `next_z_rl`, `next_proprio`, `next_ref_chunk`
- `source`, `source_chunk`, `collection_phase`, `success`, `intervention_flag`, `episode_id`, `step_id`

The bridge does not import `openpi` or `rlt_online_rl` in the ManiSkill process, so it avoids package-name conflicts between official openpi and openpi-RLT.
