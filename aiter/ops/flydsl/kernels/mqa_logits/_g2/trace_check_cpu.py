# SPDX-License-Identifier: MIT
"""GPU-free **trace** check of the page-4 QSA kernel body (no flydsl, no GPU).

Why this exists
---------------
``compile_only.py`` needs the real ``flydsl`` and a compiler backend, and
``test_contract_cpu.py`` only covers the *host* wrapper plus a torch mirror of
the kernel's math.  Neither executes the kernel **body** (the traced Python
inside ``_build_kernel``), so a typo there is only found on the measurement node
-- which is expensive (one GPU-lock window, and a build failure in the middle of
a candidate evaluation).

This script installs a tiny *symbolic* ``flydsl`` mock: every ``fx`` value is a
``Val`` node that records the operation that produced it, every operator /
``.select`` / ``.maximumf`` / ``.bitcast`` / ``__getitem__`` builds a node, and
the buffer intrinsics count as loads/stores.  It then calls the traced kernel
body directly with symbolic arguments.  That exercises every Python line of the
body -- `range_constexpr` loop bounds, list shapes (a wrongly sized K/Q unpack
list raises ``IndexError`` on the first FMA), indexing, closure names, dtype
specialisation -- without a compiler.

It also prints the per-thread *op census* (loads / stores / mul / add / ...),
which is a cheap proxy for the instruction count the backend emits: on gfx1250
every ``buffer_load`` additionally costs a ``readfirstlane`` + ``cmpx`` + exec
narrowing sequence in the lowering (see ``records/g2/isa/``), so the load count
is the most useful single number to compare between kernel revisions:

    python trace_check_cpu.py                       # trace the working-tree kernel
    python trace_check_cpu.py --kernel other.py     # trace another revision
    python trace_check_cpu.py --json out.json

Exit code 0 iff both dtype specialisations traced without an exception.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "qsa_paged_mqa_logits_page4_gfx1250.py")
)

PAGE = 4
HEADS = 4
HDIM = 128
THREADS = 256
# Shape used for the trace only: any valid shape exercises the same code.
SHAPE = dict(rows=512, columns=65536, num_pages=738964, num_requests=2, width=16384)


# ---------------------------------------------------------------------------
# Symbolic mock of the flydsl surface this kernel uses
# ---------------------------------------------------------------------------
class _Count:
    """Op census: how many IR nodes of each kind the kernel body builds."""

    def __init__(self):
        self.n = {}

    def hit(self, kind, k=1):
        self.n[kind] = self.n.get(kind, 0) + k

    def as_dict(self):
        return dict(sorted(self.n.items(), key=lambda kv: -kv[1]))


COUNT = _Count()


class _Dtype:
    def __init__(self, name):
        self.name = name
        self.ir_type = f"!fly.{name}"

    def __call__(self, value=0):
        COUNT.hit("cast")
        return Val(f"cast<{self.name}>", self)

    def __repr__(self):
        return f"<{self.name}>"


class Val:
    """One traced value.  Every operator builds (and counts) a node."""

    __slots__ = ("op", "dtype", "items")

    def __init__(self, op, dtype=None, items=None):
        self.op = op
        self.dtype = dtype
        self.items = items

    # --- construction helpers -------------------------------------------
    def _bin(self, other, kind):
        COUNT.hit(kind)
        return Val(kind, self.dtype)

    def _cmp(self, other, kind):
        COUNT.hit(kind)
        return Val(kind, None)

    # --- arithmetic / logic ---------------------------------------------
    def __add__(self, o):        return self._bin(o, "add")
    def __radd__(self, o):       return self._bin(o, "add")
    def __sub__(self, o):        return self._bin(o, "sub")
    def __rsub__(self, o):       return self._bin(o, "sub")
    def __mul__(self, o):        return self._bin(o, "mul")
    def __rmul__(self, o):       return self._bin(o, "mul")
    def __floordiv__(self, o):   return self._bin(o, "div")
    def __mod__(self, o):        return self._bin(o, "mod")
    def __lshift__(self, o):     return self._bin(o, "shl")
    def __rshift__(self, o):     return self._bin(o, "shr")
    def __and__(self, o):        return self._bin(o, "and")
    def __or__(self, o):         return self._bin(o, "or")
    def __xor__(self, o):        return self._bin(o, "xor")
    def __neg__(self):           COUNT.hit("neg"); return Val("neg", self.dtype)
    def __invert__(self):        COUNT.hit("not"); return Val("not", self.dtype)
    def __gt__(self, o):         return self._cmp(o, "gt")
    def __ge__(self, o):         return self._cmp(o, "ge")
    def __lt__(self, o):         return self._cmp(o, "lt")
    def __le__(self, o):         return self._cmp(o, "le")
    def __eq__(self, o):         return self._cmp(o, "eq")
    def __ne__(self, o):         return self._cmp(o, "ne")
    def __hash__(self):          return id(self)

    # --- flydsl value methods -------------------------------------------
    def select(self, a, b):
        COUNT.hit("select")
        return Val("select", self.dtype)

    def maximumf(self, o):
        COUNT.hit("maximumf")
        return Val("maximumf", self.dtype)

    def bitcast(self, dtype):
        COUNT.hit("bitcast")
        return Val("bitcast", dtype)

    def ir_value(self):
        return self

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self.items is not None:
                return self.items[idx]
            COUNT.hit("extract")
            return Val("extract", self.dtype)
        COUNT.hit("dyn_extract")
        return Val("dyn_extract", self.dtype)

    def __repr__(self):
        return f"<{self.op}>"


class _Vector(Val):
    @staticmethod
    def from_elements(elems, dtype=None):
        COUNT.hit("vec_from_elements", len(elems))
        return _Vector("vector", dtype, list(elems))

    @staticmethod
    def filled(n, value, dtype=None):
        COUNT.hit("vec_filled", n)
        return _Vector("vector", dtype, [value] * n)

    def shuffle(self, other, idxs):
        COUNT.hit("shuffle")
        return _Vector("vector", self.dtype, [Val("shuffle")] * len(idxs))


class _TypeObj:
    """``fx.Int32`` / ``fx.Float32`` / ... : callable + ``.ir_type``."""

    def __init__(self, name):
        self.name = name
        self.ir_type = f"!fly.{name}"

    def __call__(self, value=0):
        return Val(f"cast<{self.name}>", self)

    def __repr__(self):
        return f"<{self.name}>"


class _Ctx:
    """``fx.thread_idx`` / ``fx.block_idx`` (``.x`` / ``.y`` / ``.z``)."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, axis):
        return Val(f"{self.name}.{axis}", None)


def _make_mock():
    expr = types.ModuleType("flydsl.expr")
    for name in ("Int8", "Int16", "Int32", "Int64", "Uint8", "Uint32",
                 "Float16", "Float32", "BFloat16", "Boolean"):
        setattr(expr, name, _TypeObj(name))
    expr.Vector = _Vector
    expr.range_constexpr = range
    expr.Tensor = Val
    expr.thread_idx = _Ctx("tid")
    expr.block_idx = _Ctx("bid")
    expr.Stream = lambda *a, **k: Val("stream")

    rocdl = types.ModuleType("flydsl.expr.rocdl")

    class RawPtrBufferLoadOp:                      # signature introspection
        def __init__(self, *args, aux=None, **kw):
            self.args = args

    def _load(result_type, ptr, off, aux0, aux=None):
        COUNT.hit("load")
        return Val("buffer_load", result_type)

    def _store(value, ptr, off, aux0, aux=None):
        COUNT.hit("store")
        return Val("buffer_store")

    rocdl.RawPtrBufferLoadOp = RawPtrBufferLoadOp
    rocdl.raw_ptr_buffer_load = _load
    rocdl.raw_ptr_buffer_store = _store

    def _bufptr(iterable, num_records_bytes=None, **kw):
        COUNT.hit("descriptor")
        return Val("bufptr")

    rocdl.make_buffer_ptr = _bufptr
    rocdl.get_buffer_rsrc = lambda ptr: (COUNT.hit("rsrc"), Val("rsrc"))[1]
    rocdl.BufferCopy = lambda *a, **k: Val("copy_atom")
    expr.rocdl = rocdl

    def _recast_iter(dtype, it):
        COUNT.hit("recast_iter")
        return Val("iter")

    def _get_iter(t):
        COUNT.hit("get_iter")
        return Val("iter")

    expr.recast_iter = _recast_iter
    expr.get_iter = _get_iter
    expr.make_view = lambda *a, **k: Val("view")
    expr.make_layout = lambda *a, **k: Val("layout")
    expr.make_rmem_tensor = lambda *a, **k: Val("rmem")
    expr.copy_atom_call = lambda *a, **k: Val("copy_call")
    expr.make_copy_atom = lambda *a, **k: Val("copy_atom")
    expr.memref_load_vec = lambda *a, **k: _Vector("vector", None, [Val("ld")])
    expr.memref_store_vec = lambda *a, **k: Val("st")

    compiler = types.ModuleType("flydsl.compiler")

    def _identity(*a, **k):
        # `@flyc.kernel(name=...)` / `@flyc.jit` (decorator form) and
        # `flyc.compile(exe, *args)` (direct-call form) both appear in the kernel.
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]

        def deco(fn):
            return fn
        return deco

    compiler.kernel = _identity
    compiler.jit = _identity
    compiler.compile = _identity
    compiler.protocol = types.ModuleType("flydsl.compiler.protocol")
    compiler.protocol.dsl_size_of = lambda *a, **k: 0

    ir = types.ModuleType("flydsl._mlir.ir")

    class _Attr:
        @staticmethod
        def get(ty, v):
            return Val("attr")

    class _IType:
        @staticmethod
        def get_signless(w):
            return Val(f"i{w}")

    class _VType:
        @staticmethod
        def get(shape, elem):
            return _Vector("vector", elem, [Val("elem")] * int(shape[0]))

    ir.IntegerAttr = _Attr
    ir.IntegerType = _IType
    ir.VectorType = _VType
    ir.Type = types.SimpleNamespace(parse=lambda s: Val("ty"))
    mlir = types.ModuleType("flydsl._mlir")
    mlir.ir = ir

    flydsl = types.ModuleType("flydsl")
    flydsl.expr = expr
    flydsl.compiler = compiler
    flydsl._mlir = mlir
    flydsl.__version__ = "trace-mock"
    return {
        "flydsl": flydsl,
        "flydsl.expr": expr,
        "flydsl.expr.rocdl": rocdl,
        "flydsl.compiler": compiler,
        "flydsl.compiler.protocol": compiler.protocol,
        "flydsl._mlir": mlir,
        "flydsl._mlir.ir": ir,
    }


# ---------------------------------------------------------------------------
# Trace one dtype specialisation of one kernel revision
# ---------------------------------------------------------------------------
def _kernel_body(launch):
    """The traced kernel body, pulled out of ``_build_kernel``'s closure."""
    for cell in launch.__closure__ or ():
        try:
            v = cell.cell_contents
        except ValueError:                       # empty cell
            continue
        if callable(v) and getattr(v, "__name__", "") == "kernel":
            return v
    raise RuntimeError("kernel body not found in _build_kernel's closure")


def trace_one(mod, is_fp8, threads=THREADS, divisor=128 ** 0.5, shape=None, rpt=1):
    shape = dict(SHAPE if shape is None else shape)
    COUNT.n.clear()
    launcher = mod._build_kernel(is_fp8, threads, float(divisor), rpt)
    body = _kernel_body(launcher)
    args = [Val("tensor") for _ in range(10)]
    args += [Val("kernarg") for _ in range(5)]
    body(*args)
    census = COUNT.as_dict()
    return {
        "dtype": "fp8" if is_fp8 else "bf16",
        "rpt": int(rpt),
        "kernel_name": launcher.kernel_name,
        "ops": census,
        "loads": census.get("load", 0),
        "stores": census.get("store", 0),
        "fma_like": census.get("mul", 0) + census.get("add", 0),
        "total_nodes": sum(census.values()),
        "shape": shape,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=KERNEL_PATH)
    ap.add_argument("--rpt", default="1,2,4",
                    help="comma-separated rows-per-thread values to trace")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rpts = [int(x) for x in str(args.rpt).split(",") if x.strip()]

    installed = _make_mock()
    sys.modules.update(installed)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("qsa_page4_trace", args.kernel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = {"kernel": args.kernel, "specializations": []}
        rc = 0
        for is_fp8 in (False, True):
            for rpt in rpts:
                try:
                    rec = trace_one(mod, is_fp8, rpt=rpt)
                    out["specializations"].append(rec)
                    print("[TRACED] %-4s rpt=%d %s"
                          % (rec["dtype"], rec["rpt"], rec["kernel_name"]))
                    print("         loads=%d stores=%d mul+add=%d nodes=%d"
                          % (rec["loads"], rec["stores"], rec["fma_like"],
                             rec["total_nodes"]))
                    print("         census: " + json.dumps(rec["ops"]))
                except Exception as exc:  # noqa: BLE001
                    rc = 1
                    import traceback
                    traceback.print_exc()
                    print("[FAILED] %s rpt=%d: %s"
                          % ("fp8" if is_fp8 else "bf16", rpt, exc))
                    out["specializations"].append(
                        {"dtype": "fp8" if is_fp8 else "bf16", "rpt": rpt,
                         "error": str(exc)})
        print("\ntrace check: %s" % ("PASS" if rc == 0 else "FAIL"))
    finally:
        for name in installed:
            sys.modules.pop(name, None)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
