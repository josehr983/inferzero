# LFM2 MoE (`lfm2moe`): LFM2.5-24B-A2B

Support for Liquid AI's LFM2 / LFM2.5 mixture-of-experts models (GGUF architecture `lfm2moe`), written for
`LiquidAI/LFM2.5-24B-A2B-GGUF`, file `LFM2.5-24B-A2B-QAD-Q4_0.gguf` (13.6 GB). The whole model stays on the GPU:
no expert streaming, no expert file.

## Measured

Arc Pro B50 16 GB, Windows 11, driver 32.0.101.8805, 2026-11-23. llama.cpp is the official SYCL build b12646 with
the same file (`llama-bench -ngl 99`).

| `runz --bench-pp 4096 --n 128` | llama.cpp SYCL | two int8 passes (default) | one pass (`ARC_LFM_Q2=0`) |
|---|---|---|---|
| Prompt, 4,096 tokens | 608 tok/s | 1,634 tok/s | 2,290 - 2,350 tok/s |
| Generation, 128 tokens | 45.7 tok/s | 70.6 tok/s | 90.0 tok/s |
| KL against f32, prompt chunk (300 positions) | - | 0.0018 (top token 296 / 300) | 0.0167 (277 / 300) |
| KL against f32, `--kl-decode` | - | 0.0040 (295 / 300) | 0.0173 (276 / 300) |

**One pass is below the project's accuracy bar** (KL over 0.015, the top token right at ~92 % of the positions):
it is an opt-in for measurements, not a mode to answer with. With one pass, through the server: an 18,945-token prompt at
~1,330 tok/s, generation after it 65 tok/s; 8,192 tokens at ~1,900 tok/s.

- The f32 reference is a numpy script that follows llama.cpp's graph and gives its top tokens (`tools/ref_forward`
  does not know this architecture); `runz --kl-ref` read its log-probs of every position.
- GPU memory at `--ctx 81920`: 14.75 - 14.91 GiB with chunks of 2,048, 14.63 GiB with `--chunk 1024`.
- Perplexity (llama-perplexity's method, 510 scored tokens), one pass: 18.86 against the reference's 18.51; over
  12 chunks one pass 28.95, two passes 29.00.
- One pass: the first 12 greedy tokens equal llama.cpp's on two prompts (28 and 43 tokens), the first 5 on a
  third (6 tokens).
- `runz --snap-test`: device snapshot, host snapshot and saved conversation restored bit for bit.

Only this one card has run it so far. A Pro B60 test is planned in the next days; iGPUs, 8-lane GPUs and Linux are
untested.

## The model

40 blocks, `n_embd` 2048, 32 heads of 64. Each block: `x += op(rmsnorm(x))`, then `x += ffn(rmsnorm(x))`.

- `op`, 30 layers: a short convolution. `in_proj` (2048 -> 6144) gives `[b | c | x]`; a causal depthwise
  convolution of 3 taps over `b * x`, times `c`, then `out_proj`. Recurrent state: the last two rows of `b * x`.
- `op`, 10 layers (2, 6, ..., 38): attention, GQA with 8 KV heads, weighted RMS norm of q and k per head, NEOX
  RoPE, `1 / sqrt(64)`. `attention.head_count_kv` is per layer; 0 marks a convolution layer.
- `ffn`, layers 0 and 1: a dense SwiGLU of width 11,776.
- `ffn`, layers 2 - 39: 64 experts of width 1,536, 4 per token. Sigmoid router: the 4 largest of
  `prob + exp_probs_b`, weighted `prob / sum` (without the bias).
- Output: `token_embd_norm`, head tied to `token_embd`.

Reference: llama.cpp `src/models/lfm2.cpp`, `lfm2moe.cpp`, `lfm-base.cpp`.

## How it is built

A variant of the Qwen3 dense path (`hp.qwen3` with `hp.lfm`), `zero/engine_lfm.h`:

| Part | Code |
|---|---|
| Attention | `layer_qwen3_attn` as it is (float attention, f16 K / V) |
| Dense FFN of layers 0 - 1 | `layer_ffn` as it is |
| Prompt GEMM (Q4_0 decoded just in time, XMX) and generation GEMV | `matmul()` as it is |
| Snapshots, saved conversations, cancellation | the recurrent layers' `conv_state` / `S`, `recurrent(il)` = no KV heads |
| Short convolution | `layer_lfm_conv`, `lfm_conv` / `lfm_conv_state` |
| Experts | `layer_lfm_moe`, `lfm_moe_one`, `lfm_router` / `lfm_gather` / `lfm_scatter` / `lfm_add` / `lfm_combine4` |
| Chat | the GGUF template through minja, `src/chat_lfm.h` for the reply |

- The experts are resident weights (`hp.moe` stays false): each `ffn_*_exps` tensor is uploaded whole and an
  expert is a view of its rows with a `GpuWeight` of its own.
- Routing on the host: a kernel computes the 64 probabilities of each token, the host reads them and picks the
  experts. A prompt chunk groups its rows by expert: gather, quantize, `gate` and `up` (SwiGLU fused in the GEMM
  epilogue), `down`, weighted scatter.
- One token (`lfm_moe_one`): the quantized input `rmsnorm_quant` left is used as it is, each expert's `gate` and
  `up` run in one GEMV launch (`gemv_gu`), the four SwiGLUs are quantized as four rows of one launch and each `down`
  reads its row through views of the activation buffers. 33 -> 13 launches a layer, 59 -> 83 tok/s.
- Two int8 passes for the matmuls, as Qwen3 dense: the residual row within 0.01 - 0.7 % of the f32 reference
  after each layer of a 300-token prompt (one pass: 1 - 10 %). The one-token path above is the one-pass mode's
  (`ARC_LFM_Q2=0`); with two passes a token goes through the general path.

Outside `engine_lfm.h`:

- `kernels/ops_attn_g.cl`: heads of 64. `LOADF` loaded runs of 8 floats only, so with 4 a lane it loaded nothing
  and the attention's output was NaN (quantized to 0: the layer added nothing); `attn_fa_g` held 8 dims a lane.
- `engine_setup.h` `plan_weights`: the sum skips 3-D tensors, so the experts were not counted and the weights were
  planned resident; always just in time for this model.
- `src/tokenizer.h`: pre-tokenizer `lfm2` (llama.cpp's LLAMA3 rule: numbers in runs of up to 3 digits,
  `ignore_merges`, and `\p{L}+` where Qwen3.5's split has `[\p{L}\p{M}]+`: a combining mark ends a letter run).
  Same ids as `llama-tokenize` on a 16,547-token English file and on 712 tokens of Devanagari, Thai, Bengali, Tamil,
  pointed Hebrew and Arabic, jamo, emoji sequences and NFD accents (181 marks); with marks counted as letters 4
  places of that text differed.
- `runz.cpp`: BOS first for `lfm2` (the GGUF has no `add_bos_token`; llama.cpp adds it in code).

## Chat format

The reply is `<think>reasoning</think>answer`; the template always opens `<think>`. With thinking off
(`enable_thinking: false`, `reasoning_effort: none`) the server closes the block right after the prompt.

Tool calls are Python calls, `<|tool_call_start|>[name(arg='text', n=3, opts={"a": 1})]<|tool_call_end|>`:
`chat::parse_lfm_tool_calls` turns them into OpenAI `tool_calls`. No grammar constraint for them yet. A streamed
call is sent whole once its block closes. `tool_choice` "required" or a function starts the answer with
`<|tool_call_start|>[` (and the name); "none" never draws that token. A malformed call ends the parse with what was
read so far (`tests/test_chat_lfm.cpp`).

## Use

```
bin\runz.exe LFM2.5-24B-A2B-QAD-Q4_0.gguf --ctx 16384 --bench-pp 4096 --n 128
bin\serverz.exe -m LFM2.5-24B-A2B-QAD-Q4_0.gguf --ctx 81920
```

On a 16 GB card that also drives a monitor, `--ctx 81920 --chunk 1024` fits where chunks of 2,048 paged
(generation 80 -> 51 tok/s).

Environment: `ARC_LFM_Q2=0` one int8 pass (faster, below the accuracy bar: KL 0.017); with it, `ARC_LFM_FAST1=0` the general expert path
for single tokens and `ARC_LFM_GU=0` `gate` and `up` as two launches.

## Limits

- `expert_gating_func` 2 (sigmoid), heads of 64, 3 convolution taps; the one-token path combines exactly 4 experts.
  LFM2-8B-A1B has the same architecture and is untested.
- Text only: no image or audio input, no drafter. Dense `lfm2` and `lfm3` models are not covered.
- One conversation, one GPU, no drafter: `--slots N`, `--gpus N`, `--mtp`, `--dflash` and `--dspark` are refused at
  load.
- A prompt that changes before the last snapshot is read again from it (recurrent state), as with Qwen3.5.
- The prompt slows with depth: at 4K the attention is 18 % of the GPU time and the expert decode 16 %; the router
  kernel is a plain one (12 %).
- A full 80K context and long agent sessions have not been run.
- `llama-perplexity` (SYCL) gives 16.2 on the two chunks where the f32 reference gives 18.5 and this engine 18.9;
  why was not looked into.
