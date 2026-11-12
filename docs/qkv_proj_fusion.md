# `qkv_proj_fusion` — routed op contract v1

One call replaces the four steps a Qwen3.8-Flash-Next QSA decode/verify layer performs
before its sparse attention:

1. the `qkv_proj` GEMM (sigmoid gate + q + k + v),
2. the indexer `index_qk_proj` GEMM (index q + raw index k),
3. Gemma RMSNorm of q / k / index-q (the norm applies `1 + w`),
4. the partial NeoX mRoPE of q / k / index-q, plus the K/V, raw-index-K and rope-position
   cache stores.

Prefill keeps the framework's own path. RVQ / pool / selection, attention and the output
projection are outside this contract.

## AITER's side (this tree)

```
aiter/ops/qkv_proj_fusion.py      routed op (contract v1) + its native implementation
tests/plugin_contracts.json       the committed signature/fingerprint baseline
op_tests/test_qkv_proj_fusion_route_cpu.py   CPU tests (contract, routing, native numerics)
```

The op is routed through `aiter.plugin`: an enabled out-of-tree provider may serve a call;
a decline, no provider, nothing enabled or `AITER_PLUGINS=0` runs the native implementation.
Tuned implementations (e.g. a fused FlyDSL gfx1250 kernel) live in providers, not here.
`native_route(...)` names the native implementation of a call (`native.torch.v1` in v1).

## Signature

```python
qkv_proj_fusion(
    hidden_states, W_qkv, W_idx, q_norm_w, k_norm_w, iq_norm_w,
    cos_cache, sin_cache, positions, slot_mapping,
    k_cache, v_cache, raw_k_cache, pos_cache=None,
    rotary_dim=64, mrope_section_t=16, mrope_section_h=24, mrope_section_w=24,
    interleaved=True, eps=1e-06, mode="decode",
) -> tuple[Tensor, Tensor, Tensor]   # (gate, query, index_q)
```

| parameter | shape / dtype | meaning |
|---|---|---|
| `hidden_states` | `[M, 2560]` bf16, M ∈ [1, 1792] | packed layer input |
| `W_qkv` | `[13312, 2560]` bf16 | de-interleaved `gate | q | k | v` |
| `W_idx` | `[640, 2560]` bf16 | `index_q | index_k` |
| `q_norm_w`, `k_norm_w` | `[256]` bf16 | Gemma RMSNorm weights |
| `iq_norm_w` | `[128]` bf16 | Gemma RMSNorm weight |
| `cos_cache`, `sin_cache` | `[max_pos, rotary_dim // 2]` bf16 | cached mRoPE frequencies |
| `positions` | int64 `[M]`, `[1, M]` or `[3, M]` | rotation positions |
| `slot_mapping` | int64 `[M]` | flat cache slot (`page * block + row`); negative / out of range skips every plane |
| `k_cache`, `v_cache` | `[pages, block, 2, 256]` bf16, mutable | written in place |
| `raw_k_cache` | `[pages, block, 1, 128]` bf16, mutable | raw index K, written in place |
| `pos_cache` | `[pages, block, 1, 3]` int64, mutable, optional | canonical rope positions |
| `mode` | `"decode"` / `"verify"` | prefill must not call this op |

`block` ∈ {16, 32, 64}. Returns `gate [M, 6144] bf16`, `query [M, 24, 256] bf16`,
`index_q [M, 4, 128] bf16`.

The mRoPE is partial NeoX: `rotary_dim` of each head is rotated, `mrope_section_t|h|w` and
`interleaved` select the axis of each frequency, and every product is rounded to the
activation dtype as the framework's own fast path does.

## Contract rules

- The native implementation is **general**: any provider decline is safe, including CPU
  tensors and shapes outside the fused envelope. It is correctness-first, not fast.
- `ValueError` only for calls that violate the contract (prefill `mode`, wrong shapes or
  dtypes, block outside {16,32,64}, M outside [1,1792]).
- The routed signature is the contract: `tests/plugin_contracts.json` pins it and
  `tests/test_plugin_contracts.py` fails on any silent change. A new parameter or a changed
  default is a contract change and needs `version=2`.

## Providers

The out-of-tree gfx1250 provider (fused FlyDSL kernel) lives in the plugin repository; it
declares `op_versions = {"qkv_proj_fusion": {1}}` and serves only the calls its kernel was
qualified on, declining everything else to this native path.
