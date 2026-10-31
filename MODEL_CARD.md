---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-31B-it
pipeline_tag: text-generation
tags:
- gemma4
- compressed-tensors
- lora
- peft
- vllm
- jevbench
---

# MetaDecision-31B-Rain

Decision service for JevBench. Base model:
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it), used through Google's
official quantization-aware-training (QAT) releases of it. The model has two parts that are loaded
together and **not merged**; both are the files of
[`MeTask-Rain/metask-jev-rain-31B-trained`](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e),
unchanged and pinned by SHA-256 in `MODEL_MANIFEST.json` of the service code. This entry uses the same weights as that
entry and changes only the serving rules.

| Part | Location in that model repository | What it is |
|---|---|---|
| Base | repository root | The files of [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) at revision `52f3f65bc7a02d555763bc923bd1d9094898219d`, byte for byte: Google's official compressed-tensors W4A16 checkpoint (4-bit weights, group size 32, symmetric). |
| Adapter | `adapter/` | Our LoRA (PEFT format): rank 16, alpha 32, on the language model's q/k/v/o/gate/up/down projections, trained by us in BF16 on [`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) at revision `1e4d8beecacb8b7590c1d8bedd7335f687bf311f`. |

`adapter/adapter_model.safetensors` contains the trained checkpoint tensors unchanged plus, for the 10 full-attention layers (5, 11, …, 59) that compute values with the key projection, a v_proj LoRA that is a byte copy of the trained k_proj LoRA (20 tensors), so that vLLM applies the update to keys and values as in training.

## Serving

The service code (`service/` in the Hugging Face repository `MeTask-Rain/MetaDecision-31B-Rain`, mirrored at https://github.com/chenyuwang166-oss/MetaDecision-31B-Rain) starts vLLM 0.31.0 on a folder holding the files of that weight
repository, with `enable_lora`, `max_lora_rank=16`, and the `adapter/` folder as the single LoRA,
applied to every request. Each decision is read from the next-token log-probabilities of the answer
forms in one prompt pass; up to 49% of decisions then take a second, short drafting pass
(at most 192 tokens) under a 9 s deadline, a choice decision answered in one pass although its length allowed the drafting pass is read a second time with the options in reverse order when its first-pass margin is below 4, and the reported probabilities get a per-type post-calibration (DISCLOSURE.md).

Equivalent vLLM usage of the model:

```python
from vllm import LLM
from vllm.lora.request import LoRARequest
llm = LLM(model=MODEL_DIR, dtype="bfloat16", enable_lora=True, max_lora_rank=16)
adapter = LoRARequest("adapter", 1, MODEL_DIR + "/adapter")
# llm.generate(prompts, sampling_params, lora_request=adapter)
```

Facts on data and the serving rules are in `DISCLOSURE.md` of the service code (`service/DISCLOSURE.md` in the
Hugging Face repository).

Contact: chenyuwang166@gmail.com
