# MetaDecision-31B-Rain

A self-hosted decision service for JevBench. It serves `POST /v1/systemone` for the unchanged
`typesafe` adapter of the official harness, and `GET /health`.

The model weights are exactly those of
[`MeTask-Rain/metask-jev-rain-31B-trained`](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e)
(the same files, pinned by SHA-256 in `MODEL_MANIFEST.json`): Google's official W4A16 checkpoint
`google/gemma-4-31B-it-qat-w4a16-ct` (unchanged) and our LoRA adapter, loaded together and never
merged. `MODEL_MANIFEST.json` names that repository's earlier revision `9392335f5a17395ecd70049f85b1909d53500144`, which
holds the same model files.

Data and attribution are in [DISCLOSURE.md](DISCLOSURE.md).

## Requirements

- One NVIDIA GPU used only by this service: compute capability 8.0 or newer, NVIDIA driver 580 or
  newer, 80 GB or more of GPU memory. Validated: steps 2-6 below, run with this package and the model files downloaded in step 2, completed the 231 public items on one NVIDIA A100 80GB (231/231, zero failed). The service starts vLLM with
  `gpu_memory_utilization=0.94` (it reserves 94% of the GPU memory), `max_model_len=6656`,
  `enable_lora` and `max_lora_rank=16`. The model files take 23.3 GB (base) plus 495 MB (adapter).
  The commands use the host's GPU 0 (`--gpus '"device=0"'`).
- Docker with the NVIDIA Container Toolkit.
- On the host: Git and Python with pip; step 2 installs the Hugging Face command line tool `hf`.
- Runtime image `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`
  (Python 3.12.3, vLLM 0.31.0, torch 2.13.0+cu130), the linux/amd64 image of vLLM release
  `v0.31.0`. Nothing is installed and the service has no network access.
- Model folder: the model files of https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e
  as regular files (step 2). The repository root holds the official base files; `adapter/` holds
  the adapter (`adapter_config.json`, `adapter_model.safetensors`). A Hugging Face cache
  `snapshots/<revision>` directory holds links and cannot be mounted on its own. At startup the
  service checks the SHA-256 of the small model files, the size of the weight file and the full
  SHA-256 of both adapter files (`MODEL_MANIFEST.json`), and refuses other weight, tokenizer,
  processor or template files in the folder; it never writes to the folder.
- Code: this repository (download command in step 1).
- Harness: https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc

## Run

Copy the commands as they are, in the same Bash terminal.

1. Obtain the code and harness, and create a fresh work directory. Run from a writable parent
   directory; the new directories must not already contain another run:

   ```bash
   git clone https://github.com/chenyuwang166-oss/MetaDecision-31B-Rain.git
   PKG="$(pwd)/MetaDecision-31B-Rain"
   git clone https://github.com/fstandhartinger/jevbench.git jevbench
   git -C jevbench checkout bb05a335bc809e61b20c0f745d25499a82b326fc
   JEVBENCH="$(pwd)/jevbench"
   MODEL="$(pwd)/model-31b"
   WORK="$(pwd)/work-31b-run1"
   mkdir "$MODEL" "$WORK"
   TASKS=/jevbench/datasets/public/easy.jsonl,/jevbench/datasets/public/original.jsonl,/jevbench/datasets/public/hard.jsonl
   ```

2. Install the download tool, fetch the pinned weights, and pull the pinned runtime image:

   ```bash
   python3 -m pip install huggingface_hub
   hf download MeTask-Rain/metask-jev-rain-31B-trained --revision 2feff076f07711fe2fefe7a6fdcedfe4132b773e --local-dir "$MODEL" --exclude 'service/*'
   docker pull vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
   ```

3. Start the service:

   ```bash
   docker run -d --name metadecision31b --init --gpus '"device=0"' --network none --read-only --shm-size 8g \
     -v "$PKG":/pkg:ro -v "$MODEL":/model:ro -v "$WORK":/work -v "$JEVBENCH":/jevbench:ro \
     --entrypoint bash vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b \
     /pkg/serve.sh --model /model --state-dir /work/state --run-id run1
   ```

4. Follow the log until the service is ready:

   ```bash
   docker logs -f metadecision31b
   ```

   When a line starting with `{"status": "ok"` appears, press Ctrl-C (this ends only the log
   display) and check health before step 5:

   ```bash
   docker exec metadecision31b python3 -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8000/health").read().decode())'
   ```

   If the log ends before that line, the container has exited: the reason is in
   `docker logs metadecision31b` and in `$WORK/state/run1-startup/attempt-*.log`; go to step 6.
   After correcting the cause, use a new `WORK` and restart from step 3.

5. Run the official harness in the same container, then summarize:

   ```bash
   docker exec -e PYTHONPATH=/jevbench metadecision31b python3 -B -m jevbench.cli run --adapter typesafe \
     --endpoint http://localhost:8000 --model MetaDecision-31B-Rain --key-env '' --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl --raw-dir /work/run1/raw \
     --manifest /work/run1/manifest.json --cap-usd 15 --price-in-per-m 0.09 --price-out-per-m 0.34
   docker exec -e PYTHONPATH=/jevbench metadecision31b python3 -B -m jevbench.cli summarize --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl
   ```

6. Stop and remove the container:

   ```bash
   docker stop -t 40 metadecision31b && docker rm metadecision31b
   ```

Each evaluation uses a new `WORK` and a new container. A run name that already exists below the
state directory is refused at startup ("use a new --state-dir"); with these commands that means a
new `WORK`.

## What to expect

- Health line: `"status": "ok"` and `"warnings": []`.
- Startup: `launch.py` verifies the model folder and the adapter, writes the run configuration
  to `$WORK/state/run1-startup/config.runtime.json` and starts vLLM. Each startup attempt waits
  at most 1800 s for readiness.
- Writes: the service writes files only below `$WORK/state` (plus shared memory in `/dev/shm`) and
  the harness only below `$WORK/run1`. The package and model mounts are read-only. Cache and
  configuration path variables (`HF_HOME`, `XDG_*`, `TRITON_*`, `TORCH_HOME` and similar) must not
  be passed to the container: a value outside `/work/state` is refused at startup. No script of
  this package deletes directories recursively.
- Responses: an item the service cannot process returns HTTP 422 without answer and without usage;
  after an engine failure, and from the third consecutive item failure on, the service answers
  every `POST /v1/systemone` with HTTP 503 and `/health` reports `"status": "error"`. No substitute
  answer is ever returned.

## Package checks (offline, no GPU or network)

```bash
cd "$PKG"
python3 -B pkg_check.py             # package integrity, configuration, adapter contract, serving policy
python3 -B selftest.py --offline    # synthetic software tests, including long-input compression
```

`MODEL_MANIFEST.json` pins the weight repository, revision and every model and adapter file;
`MANIFEST.sha256` binds all package files. `MODEL_CARD.md` describes the model.

Contact: chenyuwang166@gmail.com
