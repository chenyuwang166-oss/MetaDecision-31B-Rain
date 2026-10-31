# MetaDecision-31B-Rain: disclosure

## Model

- Base model: `google/gemma-4-31B-it`. The weights used are Google's official
  quantization-aware-training (QAT) releases of this model:
  - training base: `google/gemma-4-31B-it-qat-q4_0-unquantized` at revision
    `1e4d8beecacb8b7590c1d8bedd7335f687bf311f` (BF16);
  - served base: `google/gemma-4-31B-it-qat-w4a16-ct` at revision
    `52f3f65bc7a02d555763bc923bd1d9094898219d`, Google's compressed-tensors W4A16 checkpoint
    (4-bit weights, group size 32, symmetric; activations unquantized), used unchanged. We did not
    quantize anything.
- Own trained weights: the LoRA adapter in `adapter/` (PEFT format), rank 16, alpha 32, on the
  language model's q/k/v/o/gate/up/down projections, trained by us in BF16 on the training base.
- The adapter is not merged: vLLM 0.31.0 loads it with `enable_lora` and `max_lora_rank=16` and
  applies it to every request on top of the W4A16 base.
- In the 10 full-attention layers (5, 11, …, 59) Gemma 4 computes the attention values with the key
  projection (`attention_k_eq_v`); these layers have no v_proj, and vLLM serves them with separate
  key and value slots. `adapter/adapter_model.safetensors` therefore contains the trained checkpoint
  tensors unchanged plus, for these 10 layers, a v_proj LoRA that is a byte copy of the trained
  k_proj LoRA (20 tensors), so that the update applies to keys and values as in training.
- The base files and the adapter are exactly those of `MeTask-Rain/metask-jev-rain-31B-trained`
  (same SHA-256 for every file, `MODEL_MANIFEST.json`); no weight was trained or changed for this
  entry. It differs from that entry only in the serving rules below.

## Data

- The adapter is the one of `MeTask-Rain/metask-jev-rain-31B-trained`, trained on our own generated
  data as disclosed for that entry.
- Inherited from the earlier service of these weights, unchanged: the per-type temperatures of the
  first calibration step, fitted on the 231 public JevBench items, and the drafting prompt limit
  (840 tokens, below), chosen using the 231 public items together with our own development data.
  Fits and choices made on these public items are in-sample. The yes/no confidence constant 0.898
  (first-step answers carry 0.898 or 0.102) is the yes/no accuracy of our own held-out development
  data (1,419 yes/no items), rounded to 3 decimals; its value does not depend on any leaderboard
  result.
- The serving parameters new in this entry (drafting threshold and share, draft length, the re-read
  rule and the post-calibration below) were fitted on our own independently built development set.
  The 9 s deadline and the drafting chunk settings are fixed rules, not fitted.
- The 231 public items were also used for the validation run of the workflow in README.md; no
  parameter of this entry was chosen from that run.

## Serving

- Each decision is first read in one prompt pass: the answer distribution comes from the next-token
  log-probabilities of the answer forms. If the gap between the two highest answer scores of this
  pass is below a threshold (fitted so that 52.1% of the decisions of our development set
  that can take the drafting pass fall below it), the decision may take a second, drafting pass: the
  model writes at most 192 tokens of reasoning (the instruction asks for at most
  50 words), then the answer distribution is read again. If the draft ends without an
  answer line, an answer line is appended and the answer is read after the draft.
- Deadline: a decision that takes the drafting pass has a deadline of 9 s from the
  moment the service starts answering the request. Drafting runs in chunks sized so that the answer
  can still be read before the deadline (an engine call in progress is always completed, never
  abandoned); a draft that reaches its time budget is ended with an answer line, and a drafting pass
  that cannot be read in time returns the first-pass answer (with a `warning` note, also listed in
  `partial_errors`).
- Decisions whose drafting prompt (the first-pass prompt plus the drafting instruction) is
  longer than 840 tokens are answered in one pass. Inputs longer than 6,653 tokens never take the drafting
  pass and are answered from a context shortened to at most 6,653 tokens by the service's existing
  extraction rule; an input is not refused because of its length.
- Drafting share (slow-path quota 49%): a counter kept below the service's state
  directory admits a drafting pass only while drafting decisions stay
  at most 49% of all decisions so far.
- Re-read: a choice decision answered in one pass although its length allowed the drafting pass (for example because the drafting share or the cost fuse refused it), whose first-pass margin (the log-score difference between its two highest answers) is below 4, is read once more in one pass with the options in reverse order (each option keeps its text and label). The two answer distributions are averaged by label. The second read is included in the decision's token usage and in the cost fuse; it is not a drafting pass and does not count toward the drafting share.
- Cost fuse: from the 101st decision on, when the running engine cost per decision reaches 90% of
  USD 0.0646 per 1,000 decisions, drafting passes are limited to the share that brings the running
  cost back to 85% of that limit (none while first-pass decisions alone cost more than 85% of it),
  until the running cost falls to 85% of the limit. The fuse prices engine token counts at
  USD 0.0598 per million tokens, input and output alike (the GPU-time basis of the
  leaderboard's cost rule for this architecture). The fuse never changes the first pass.
- Answers after a drafting pass use the same first-step calibration parameters (per-type
  temperatures and yes/no confidence) as first-pass answers.
- Post-calibration: after the first-step calibration, the reported probabilities are transformed per type, without any model call or token (a temperature t replaces the probabilities p by p^(1/t), renormalised; parameters fitted on our own independently built development set; values in `config.json`, block `post_calibration`). yes/no: temperature 0.5 (the first-step 0.898/0.102 become 0.987/0.013); score: all probability mass is put on the most probable level. The answer label is recomputed from the transformed probabilities; types not listed are unchanged.

## Run conditions and usage accounting

- vLLM 0.31.0 with `VLLM_BATCH_INVARIANT=1` (also set by the launcher for serving) and
  `VLLM_USE_FLASHINFER_SAMPLER=0`, one request at a time; model loading and warm-up are excluded.
- Cost basis: measured engine token counts (input tokens minus cached prefix tokens; generated
  tokens).
