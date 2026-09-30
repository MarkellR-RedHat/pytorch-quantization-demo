# Raw measurements

Everything the dashboard shows is built from these files by `scripts/build_benchmark_file.py`.

- `2026-09-29-r1/`: the afternoon run. 5 requests per setup on one prompt through a port-forward from a laptop, plus the first three preset recordings under `captures/`. Kept as `history` in the benchmark file. The raw files don't record the prompt or the client.
- `2026-09-29-r2/`: the evening run, delivered by the laptop that drove the cluster. `sweeps/<SETUP>/` is `vllm bench serve` inside each pod (`c<N>.json` random prompts, `sharegpt-c<N>.json`, `single-t0.json` and `single-t0.7.json` one at a time); `evals/` is `lm_eval` output; `logs/` has each pod's startup log, versions, `nvidia-smi` samples at 64 in flight, and the spec decode counters read before and after each single-stream run; `captures/` are the preset recordings as delivered (`old-kv-prompt/` answers the earlier wording of the KV cache preset); `live/` is the Sep 30 check of the live path against the FP8 deployment; `scripts/` is the capture script as run.

The setup folders keep the names they were delivered with: `FP16` is the BF16 setup. `summary.json` was written on the laptop and its `results-2/` paths are that laptop's working directory. `notes.txt` was scrubbed of the cluster and node name and nothing else; its open line "GPTQ kernel TBD from startup logs" is answered by the INT4_RH startup log: `Using MacheteLinearKernel for GPTQMarlinLinearMethod`, the same kernel as the AWQ build. The manifests in `kubernetes/models/` match the args in the startup logs.
