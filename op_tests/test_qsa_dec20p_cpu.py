# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DEC20-PROMOTE (CPU only): knob-pinned decode-regime overlay rules / buckets.

* ``gen_profile.DEC20P_RULES`` / ``DEC20P_BUCKETS``: every overlay sits right before its base
  entry, has the base's relation / rows / table width / variant, pins exactly the table's
  knobs (+ the default-env ones), and the overlays of one base tile its requests x columns
  range with disjoint boxes;
* pending overlays never serve (the unchanged base ``pass`` entries do); once checked
  (``pass``) they precede their base and the launch plan carries the promoted knobs: fp8
  4-row decode group ``wc`` 32 -> builder slab 32 (router variant stays w64), fp8 draft
  ``wg_waves`` 4 over the wrapper's 'auto' 2, kernel-native verify3 buckets;
* knob validation (``wc`` fp8 4-row decode only; draft ``wg_waves`` 4 pin fp8 only) and the
  route guard (an env-applied fp8 draft wg_waves other than 'auto' still mismatches);
* ``gen_profile check`` staging: an overlay overlaps its base, never a sibling overlay.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_qsa_index_scorer_cpu as base

API, PR, K = base.API, base.PR, base.K
GEN = base._gen()
SFX = {4: "", 8: "_p8", 16: "_p16"}
_PROBE_Q = (1, 2, 8, 16, 17, 32, 33, 64, 65, 128, 256, 257, 448, 512)


def _committed():
    with open(base.PROFILE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _box_admits(e, q, c):
    return PR._range_ok(e["requests"], q) and PR.columns_admitted(e["columns"], c)


class OverlayTables(unittest.TestCase):
    def setUp(self):
        self.data = _committed()
        self.ids = GEN.dec20p_ids()

    def _check(self, key, table, extra_knobs):
        entries = self.data[key]
        order = [e["id"] for e in entries]
        by_id = {e["id"]: e for e in entries}
        n = 0
        for b0, items in table.items():
            for sfx in SFX.values():
                bid = b0 + sfx
                if bid not in by_id:
                    continue
                b = by_id[bid]
                ov = [by_id[f"{b0}_{t}{sfx}"] for t, *_ in items]
                # right before the base, in table order
                i = order.index(bid)
                self.assertEqual(order[i - len(ov) : i], [o["id"] for o in ov], bid)
                bvb = PR.split_knobs(b["vid"], b["params"])[0]
                for o, (_t, (qlo, qhi), cols, knobs) in zip(ov, items):
                    for k in ("cache_dtype", "rows", "table_width", "vid", "relation"):
                        self.assertEqual(o.get(k), b.get(k), (o["id"], k))
                    self.assertEqual(PR.page_of(o), PR.page_of(b))
                    self.assertEqual(o["requests"], {"min": qlo, "max": qhi}, o["id"])
                    self.assertEqual(o["columns"], cols, o["id"])
                    vb, kn = PR.split_knobs(o["vid"], o["params"])
                    if key == "rules":
                        # the base variant (router wave_cols kept even under wc)
                        self.assertEqual(vb, bvb, o["id"])
                    want = dict(extra_knobs(o), **knobs)
                    self.assertEqual(kn, want, o["id"])
                    self.assertIn(o["golden"]["status"], ("pending", "pass"), o["id"])
                    self.assertIn("DEC20-PROMOTE", o["notes"])
                # disjoint boxes tiling the base requests x columns range
                for q in _PROBE_Q:
                    if not PR._range_ok(b["requests"], q):
                        continue
                    for c in PR.column_values(b["columns"]):
                        hits = [o["id"] for o in ov if _box_admits(o, q, c)]
                        self.assertEqual(len(hits), 1, (bid, q, c, hits))
                n += 1
        return n

    def test_rules(self):
        n = self._check(
            "rules",
            GEN.DEC20P_RULES,
            lambda o: PR.default_knobs(o["cache_dtype"], o["regime"]),
        )
        self.assertEqual(n, 12)  # draft / decode x bf16 / fp8 x P 4/8/16
        self.assertEqual(
            sum(r["id"] in self.ids for r in self.data["rules"]),
            3 * sum(len(v) for v in GEN.DEC20P_RULES.values()),
        )

    def test_buckets(self):
        n = self._check("buckets", GEN.DEC20P_BUCKETS, lambda o: {"group_rows": 3})
        self.assertEqual(n, 6)

    def test_legacy_entries_unchanged(self):
        """The pre-promotion entries are all still there and keep their status."""
        legacy = GEN.strip_dec20p(self.data)
        self.assertEqual(len(legacy["rules"]), 39)
        self.assertEqual(len(legacy["buckets"]), 12)
        self.assertEqual(
            sum(r["golden"]["status"] == "pass" for r in legacy["rules"]), 39
        )
        self.assertEqual(
            sum(b["golden"]["status"] == "pass" for b in legacy["buckets"]), 6
        )

    def test_check_staging_overlap(self):
        by = {r["id"]: r for r in self.data["rules"]}
        for b0, items in GEN.DEC20P_RULES.items():
            for sfx in SFX.values():
                ov = [by[f"{b0}_{t}{sfx}"] for t, *_ in items]
                for o in ov:
                    self.assertTrue(GEN._rules_overlap(o, by[b0 + sfx]), o["id"])
                    for o2 in ov:
                        if o2 is not o:
                            self.assertFalse(
                                GEN._rules_overlap(o, o2), (o["id"], o2["id"])
                            )
        # another page / dtype never overlaps
        self.assertFalse(
            GEN._rules_overlap(by["decode_bf16_s16"], by["decode_bf16_p8"])
        )
        self.assertFalse(GEN._rules_overlap(by["decode_fp8_s"], by["decode_bf16"]))


class OverlayKnobValidation(unittest.TestCase):
    def setUp(self):
        self.data = _committed()

    def _rule(self, rid):
        return json.loads(
            json.dumps(next(r for r in self.data["rules"] if r["id"] == rid))
        )

    def _invalid(self, r):
        r["sig"] = PR.rule_signature(r)
        with self.assertRaises(PR.ProfileError) as cm:
            PR.validate_rule(r)
        return str(cm.exception)

    def test_wc_fp8_4row_only(self):
        r = self._rule("decode_bf16_s256")
        r["params"]["wc"] = 32
        self.assertIn("wc", self._invalid(r))
        b = json.loads(
            json.dumps(
                next(b for b in self.data["buckets"] if b["id"] == "verify3_g3_fp8_l")
            )
        )
        b["params"]["wc"] = 32
        b["sig"] = PR.bucket_signature(b)
        with self.assertRaises(PR.ProfileError):
            PR.validate_bucket(b)
        PR.validate_rule(self._rule("decode_fp8_s"))  # the committed pin is valid

    def test_draft_wg4_fp8_only(self):
        PR.validate_rule(self._rule("draft_fp8_s"))
        r = self._rule("draft_bf16_ll3")
        r["params"]["wg_waves"] = 4
        self.assertIn("wg_waves", self._invalid(r))


def _verified_overlays(data, name):
    d = json.loads(json.dumps(data))
    d["profile"] = name
    ids = GEN.dec20p_ids()
    for e in d["rules"] + d["buckets"]:
        if e["id"] in ids:
            e["golden"] = {
                "status": "pass",
                "kernel_sha256": d["kernel"]["sha256"],
                "samples_n": len(e["samples"]),
                "checked": [dict(s, **{"pass": True}) for s in e["samples"]],
            }
    return d


class OverlayDecisions(base.Env):
    def setUp(self):
        super().setUp()
        self._td = base._tmpdir()
        self.td = self._td.__enter__()
        name = "qsa_dec20p_v1"
        self.vpath = os.path.join(self.td, f"{name}.json")
        with open(self.vpath, "w") as fh:
            json.dump(_verified_overlays(_committed(), name), fh)

    def tearDown(self):
        self._td.__exit__(None, None, None)
        super().tearDown()

    def _plan(self, args, profile=None, **kw):
        d = self.decide(args, profile=profile or self.vpath, **kw)
        self.assertTrue(d.ok, d)
        return d, d._plan

    def test_pending_overlays_never_serve(self):
        for args, cell in (
            (
                base.call_args(rows=32, requests=8, width=256, columns=1024),
                "decode_bf16",
            ),
            (
                base.call_args(rows=32, requests=8, width=256, columns=1024, fmt="fp8"),
                "decode_fp8",
            ),
            (base.call_args(rows=8, requests=8, width=256, columns=1024), "draft_bf16"),
            (
                base.call_args(rows=1, requests=1, width=256, columns=1024, fmt="fp8"),
                "draft_fp8",
            ),
        ):
            d, p = self._plan(args, profile=PR.DEFAULT_PROFILE)
            committed = _committed()
            overlays_checked = all(
                e["golden"]["status"] == "pass"
                for e in committed["rules"]
                if e["id"].startswith(cell + "_")
            )
            if overlays_checked:
                # GPU-checked overlays (DEC20-PROMOTE step 6): an overlay serves, base does not
                self.assertTrue(d.cell.startswith(f"rule:{cell}_"), d)
                self.assertIn("ll", p.build_kw)
            else:
                # pending overlays are never selected: the unchanged base rule serves
                self.assertEqual(d.cell, f"rule:{cell}", d)
                self.assertNotIn("ll", p.build_kw)

    def test_decode_overlays(self):
        cases = (
            # (rows, requests, columns, width, dtype, page, rule, slab, knobs)
            (
                32,
                8,
                1024,
                256,
                "bf16",
                4,
                "decode_bf16_s16",
                32,
                {"ll": 3, "kd": 2, "wg_waves": 2},
            ),
            (
                128,
                32,
                1024,
                256,
                "bf16",
                4,
                "decode_bf16_s32",
                32,
                {"ll": 3, "sg": 2, "wg_waves": 2},
            ),
            (512, 128, 1024, 256, "bf16", 4, "decode_bf16_s256", 32, {"ll": 3}),
            (
                1792,
                448,
                1024,
                256,
                "bf16",
                4,
                "decode_bf16_s512",
                32,
                {"ll": 3, "slabs": 4, "sg": 2},
            ),
            (64, 16, 4096, 1024, "bf16", 4, "decode_bf16_m64", 32, {"ll": 3}),
            (
                1024,
                256,
                4096,
                1024,
                "bf16",
                4,
                "decode_bf16_m512",
                32,
                {"ll": 3, "slabs": 4, "sg": 2},
            ),
            (
                32,
                8,
                16384,
                4096,
                "bf16",
                4,
                "decode_bf16_l16",
                32,
                {"ll": 3, "wg_waves": 2},
            ),
            (
                256,
                64,
                16384,
                2048,
                "bf16",
                8,
                "decode_bf16_l512_p8",
                32,
                {"ll": 3, "slabs": 4, "sg": 2, "wg_waves": 2},
            ),
            (
                32,
                8,
                1024,
                64,
                "fp8",
                16,
                "decode_fp8_s_p16",
                32,
                {"ll": 3, "fk": 1, "kd": 1},
            ),
            (
                256,
                64,
                4096,
                1024,
                "fp8",
                4,
                "decode_fp8_s",
                32,
                {"ll": 3, "fk": 1, "kd": 1},
            ),
            (32, 8, 16384, 4096, "fp8", 4, "decode_fp8_l", 64, {"ll": 2, "fk": 1}),
        )
        for rows, q, cols, w, dt, page, rid, slab, knobs in cases:
            args = base.call_args(
                rows=rows, requests=q, width=w, columns=cols, fmt=dt, page=page
            )
            d, p = self._plan(args, buckets=False)
            self.assertEqual(d.cell, f"rule:{rid}", d)
            self.assertEqual(p.kind, "decg")
            self.assertEqual(p.build_args[2], slab, rid)
            want = dict({"group_rows": 4, "row_mode": "base", "wg_waves": 4}, **knobs)
            if page != 4:
                want["page_size"] = page
            self.assertEqual(p.build_kw, want, rid)
            self.assertNotIn("wc", p.build_kw)
            # the router variant is unchanged (fp8 4-row: w64); only the launch slab is pinned
            got = API.predict_route(K, dt, rows, q, cols, 0x1000, page)
            self.assertEqual(got[1]["wave_cols"], 64 if dt == "fp8" else 32)
            gx, _gy = p.grid(rows)
            self.assertGreater(gx, 0)

    def test_draft_overlays(self):
        cases = (
            (1, 1, 1024, 256, "bf16", "draft_bf16_q1", {"wg_waves": 2, "ll": 3}),
            (64, 64, 16384, 4096, "bf16", "draft_bf16_ll3", {"wg_waves": 4, "ll": 3}),
            (
                1,
                1,
                16384,
                4096,
                "fp8",
                "draft_fp8_q1",
                {"wg_waves": 4, "ll": 3, "fk": 1},
            ),
            (
                128,
                128,
                1024,
                256,
                "fp8",
                "draft_fp8_s",
                {"wg_waves": 4, "ll": 3, "fk": 1},
            ),
            (
                128,
                128,
                4096,
                1024,
                "fp8",
                "draft_fp8_l",
                {"wg_waves": 2, "ll": 3, "fk": 1},
            ),
        )
        for rows, q, cols, w, dt, rid, kw in cases:
            args = base.call_args(rows=rows, requests=q, width=w, columns=cols, fmt=dt)
            d, p = self._plan(args)
            self.assertEqual(d.cell, f"rule:{rid}", d)
            self.assertEqual((p.kind, p.build_kw), ("draft", kw), rid)

    def test_fp8_draft_env_knob_still_guarded(self):
        dr = base.call_args(rows=8, requests=8, width=256, columns=1024, fmt="fp8")
        self.assertEqual(self.decide(dr, profile=self.vpath).cell, "rule:draft_fp8_s")
        with mock.patch.object(K, "draft_wg_waves", lambda is_fp8: 1):
            self.assertReason(dr, "route_mismatch", profile=self.vpath)

    def test_verify3_buckets(self):
        cases = (
            (8, 1024, 256, "bf16", "verify3_g3_bf16_s32", {"ll": 3, "sg": 2}, 4),
            (64, 1024, 256, "bf16", "verify3_g3_bf16_s512", {"ll": 3}, 4),
            (8, 16384, 4096, "bf16", "verify3_g3_bf16_l", {"ll": 3}, 4),
            (
                8,
                1024,
                256,
                "fp8",
                "verify3_g3_fp8_s16",
                {"ll": 3, "fk": 1, "sg": 2, "sw": 1, "wg_waves": 2},
                4,
            ),
            (
                64,
                1024,
                256,
                "fp8",
                "verify3_g3_fp8_s512",
                {"ll": 3, "fk": 1, "wg_waves": 2},
                4,
            ),
            (8, 4096, 1024, "fp8", "verify3_g3_fp8_l", {"ll": 3, "fk": 1}, 4),
            (
                8,
                1024,
                64,
                "fp8",
                "verify3_g3_fp8_s16_p16",
                {"ll": 3, "fk": 1, "sg": 2, "sw": 1, "wg_waves": 2},
                16,
            ),
        )
        for C, cols, w, dt, bid, knobs, page in cases:
            args = base.call_args(
                rows=3 * C, requests=C, width=w, columns=cols, fmt=dt, page=page
            )
            d, p = self._plan(args)
            self.assertTrue(str(d.cell).startswith(f"bucket:{bid}>"), d)
            self.assertEqual(p.build_args[2], 32, bid)
            want = {"group_rows": 3, "row_mode": "dedup", "wg_waves": 4}
            want.update(knobs)
            if page != 4:
                want["page_size"] = page
            self.assertEqual(p.build_kw, want, bid)

    def test_overlay_samples_route(self):
        """Every overlay sample passes the route guard at its own shape (rules) / target
        shape (buckets), as ``decide`` and prewarm require."""
        data = _committed()
        ids = GEN.dec20p_ids()
        n = 0
        for e in data["rules"] + data["buckets"]:
            if e["id"] not in ids:
                continue
            page = PR.page_of(e)
            g3 = e.get("kind") == "group"
            for s in e["samples"][:8]:
                rows = 4 * s["requests"] if g3 else s["rows"]
                got = API.predict_route(
                    K, e["cache_dtype"], rows, s["requests"], s["columns"], 0x1000, page
                )
                if g3:
                    w3 = K.decode_group_route(
                        s["rows"], s["requests"], e["cache_dtype"] == "fp8", 0x1000, 3
                    )
                    got = (got[0], dict(got[1], wave_cols=int(w3)))
                self.assertTrue(
                    API._route_ok(*got, e["vid"], e["params"]), (e["id"], s)
                )
                n += 1
        self.assertGreater(n, 400)


if __name__ == "__main__":
    unittest.main()
