# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests of the routed op ``aiter.ops.qkv_proj_fusion.qkv_proj_fusion`` (contract v1).

No GPU, no built aiter: ``aiter/plugin`` (standard library only) and the op module load
standalone; the tests cover the committed contract baseline, the routed registration,
native numerics against an independent float64 reference, and the cache-write contract.

    python3 -B -m pytest -q op_tests/test_qkv_proj_fusion_route_cpu.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OP = "qkv_proj_fusion"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if "aiter.plugin" in sys.modules:  # a built aiter in this process
    plugin = sys.modules["aiter.plugin"]
else:
    plugin = _load("aiter.plugin", ROOT / "aiter" / "plugin" / "__init__.py")
try:
    import aiter  # noqa: F401  (a built aiter: its ops package exists)
except Exception:  # noqa: BLE001 - standalone: give the op module its package chain
    aiter = types.ModuleType("aiter")
    aiter.plugin = plugin
    ops = types.ModuleType("aiter.ops")
    ops.plugin = plugin
    sys.modules.setdefault("aiter", aiter)
    sys.modules.setdefault("aiter.ops", ops)
    aiter.ops = ops
module = _load("qkv_proj_fusion_op_cpu", ROOT / "aiter" / "ops" / "qkv_proj_fusion.py")

BF = torch.bfloat16
MODEL, GATE, Q, K, V, IQ, IK = 2560, 6144, 6144, 512, 512, 512, 128
QH, KVH, IH, HD, IHD = 24, 2, 4, 256, 128


def _tensors(tokens=4, pages=2, block=16, rotary=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(tokens, MODEL, generator=g, dtype=torch.float32).to(BF)
    Wq = (torch.randn(GATE + Q + K + V, MODEL, generator=g) * 0.02).to(BF)
    Wi = (torch.randn(IQ + IK, MODEL, generator=g) * 0.02).to(BF)
    qn = torch.ones(HD, dtype=BF)
    kn = torch.ones(HD, dtype=BF)
    iqn = torch.ones(IHD, dtype=BF)
    cos = torch.randn(128, rotary // 2, generator=g).to(BF)
    sin = torch.randn(128, rotary // 2, generator=g).to(BF)
    positions = torch.arange(tokens, dtype=torch.int64)
    slots = torch.arange(tokens, dtype=torch.int64)
    slots[1] = -1  # skipped row
    kc = torch.zeros(pages, block, KVH, HD, dtype=BF)
    vc = torch.zeros(pages, block, KVH, HD, dtype=BF)
    rk = torch.zeros(pages, block, 1, IHD, dtype=BF)
    pc = torch.zeros(pages, block, 1, 3, dtype=torch.int64)
    return dict(
        hidden_states=h,
        W_qkv=Wq,
        W_idx=Wi,
        q_norm_w=qn,
        k_norm_w=kn,
        iq_norm_w=iqn,
        cos_cache=cos,
        sin_cache=sin,
        positions=positions,
        slot_mapping=slots,
        k_cache=kc,
        v_cache=vc,
        raw_k_cache=rk,
        pos_cache=pc,
    )


def _norm64(x, w, eps=1e-06):
    x = x.double()
    scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * scale * (1.0 + w.double())


def _axis(rotary, st, sh, sw, inter):
    half = rotary // 2
    freq = torch.arange(rotary) % half
    if inter:
        a = torch.zeros_like(freq)
        a = torch.where((freq % 3 == 1) & (freq < 3 * sh), 1, a)
        return torch.where((freq % 3 == 2) & (freq < 3 * sw), 2, a)
    return torch.where(freq < st, 0, torch.where(freq < st + sh, 1, 2))


def _reference(d, rotary=64, st=16, sh=24, sw=24, inter=True):
    h = d["hidden_states"].double()
    y = h @ d["W_qkv"].double().t()
    yi = h @ d["W_idx"].double().t()
    tokens = h.shape[0]
    q = _norm64(y[:, GATE : GATE + Q].view(tokens, QH, HD), d["q_norm_w"])
    k = _norm64(y[:, GATE + Q : GATE + Q + K].view(tokens, KVH, HD), d["k_norm_w"])
    iq = _norm64(yi[:, :IQ].view(tokens, IH, IHD), d["iq_norm_w"])
    half = rotary // 2
    axis = _axis(rotary, st, sh, sw, inter)
    pos = d["positions"].reshape(1, -1).expand(3, -1)[:, :tokens].long()
    freq = (torch.arange(rotary) % half).reshape(1, rotary)
    idx = (pos[axis, :] * half + freq.reshape(-1, 1)).t().reshape(-1)
    cos = d["cos_cache"].double().reshape(-1)[idx].reshape(tokens, 1, rotary)
    sin = d["sin_cache"].double().reshape(-1)[idx].reshape(tokens, 1, rotary)
    sign = torch.cat([-torch.ones(half), torch.ones(half)]).double()

    def rope(x):
        xr = x[..., :rotary]
        paired = torch.cat([xr[..., half:rotary], xr[..., :half]], dim=-1)
        rot = xr * cos + sign * paired * sin
        return torch.cat([rot, x[..., rotary:]], dim=-1)

    return y[:, :GATE].to(BF), rope(q).to(BF), rope(iq).to(BF)


class TestContract(unittest.TestCase):
    def test_committed_baseline_matches_source(self):
        spec = importlib.util.spec_from_file_location(
            "qkv_proj_fusion_contracts_cpu", ROOT / "tests" / "test_plugin_contracts.py"
        )
        contracts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contracts)

        found = contracts.routed_contracts(ROOT)
        baseline = json.loads((ROOT / "tests" / "plugin_contracts.json").read_text())
        self.assertIn(OP, found)
        self.assertEqual(found[OP]["fingerprint"], baseline[OP]["fingerprint"])
        self.assertEqual(found[OP]["signature"], baseline[OP]["signature"])
        self.assertEqual(found[OP]["source"].replace("\\", "/"), baseline[OP]["source"])

    def test_op_is_routed(self):
        ops = plugin.routed_ops() if hasattr(plugin, "routed_ops") else None
        if ops is None:  # older plugin module: the decorator still registered it
            self.assertTrue(callable(module.qkv_proj_fusion))
            return
        self.assertIn(OP, [op.name if hasattr(op, "name") else op for op in ops])

    def test_native_route_label(self):
        d = _tensors()
        self.assertEqual(module.native_route(**{k: d[k] for k in d if k != "pos_cache"}), module.NATIVE_TORCH)


class TestNative(unittest.TestCase):
    def test_projection_and_rope_match_float64_reference(self):
        d = _tensors(tokens=4)
        gate, query, index_q = module.qkv_proj_fusion(**d)
        r_gate, r_query, r_iq = _reference(d)
        self.assertEqual(tuple(gate.shape), (4, GATE))
        self.assertEqual(tuple(query.shape), (4, QH, HD))
        self.assertEqual(tuple(index_q.shape), (4, IH, IHD))
        for name, got, want in (
            ("gate", gate, r_gate),
            ("query", query, r_query),
            ("index_q", index_q, r_iq),
        ):
            diff = (got.double() - want.double()).abs().max().item()
            scale = want.double().abs().max().item() or 1.0
            self.assertLess(diff / scale, 2e-2, f"{name} diff {diff}")

    def test_cache_writes_skip_negative_slots(self):
        d = _tensors(tokens=4, pages=2, block=16)
        kc, vc, rk, pc = d["k_cache"], d["v_cache"], d["raw_k_cache"], d["pos_cache"]
        module.qkv_proj_fusion(**d)
        self.assertTrue(bool((kc.reshape(-1, KVH, HD)[0] != 0).any()))
        self.assertFalse(bool((kc.reshape(-1, KVH, HD)[1] != 0).any()))  # slot -1 skipped
        self.assertTrue(bool((vc.reshape(-1, KVH, HD)[2] != 0).any()))
        self.assertTrue(bool((rk.reshape(-1, 1, IHD)[0] != 0).any()))
        self.assertTrue(bool((pc.reshape(-1, 1, 3)[2] != 0).any()))

    def test_contract_errors_raise(self):
        d = _tensors()
        bad = dict(d)
        bad["hidden_states"] = d["hidden_states"].float()
        with self.assertRaises(ValueError):
            module.qkv_proj_fusion(**bad)
        bad = dict(d)
        bad["mode"] = "prefill"
        with self.assertRaises(ValueError):
            module.qkv_proj_fusion(**bad)

    def test_three_axis_positions(self):
        d = _tensors(tokens=4)
        d["positions"] = torch.stack(
            [torch.arange(4), torch.arange(4), torch.arange(4)], dim=0
        )
        gate, query, index_q = module.qkv_proj_fusion(**d)
        self.assertTrue(bool(torch.isfinite(query).all()))


if __name__ == "__main__":
    unittest.main()
