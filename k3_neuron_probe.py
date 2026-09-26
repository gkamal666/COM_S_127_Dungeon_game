#!/usr/bin/env python3
"""Real Kimi K3 activation-conditioned neuron concentration probe.

Requires the public Colibri c/tools/k3_ref.py beside this file. The wrapper
replaces its local-shard reader with HTTP range reads, executes an exact
2-layer K3 prefix, and measures how many exact neuron triples are needed to
reproduce each native routed expert output.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import k3_ref

BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main"
GROUP = 32


def log(x: str) -> None:
    print(x, flush=True)


def get(url: str, span: tuple[int, int] | None = None, retries: int = 8) -> bytes:
    headers = {"User-Agent": "k3-cas-neuron-probe/1.0"}
    if span:
        headers["Range"] = f"bytes={span[0]}-{span[1]}"
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r:
                return r.read()
        except Exception as exc:
            last = exc
            log(f"retry {i+1}/{retries}: {exc!r}")
            time.sleep(2 * (i + 1))
    raise RuntimeError(last)


class HttpShards:
    """Drop-in replacement for k3_ref.Shards using byte-range reads."""
    downloaded = 0

    def __init__(self, directory: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        for name in ("model.safetensors.index.json", "config.json"):
            path = self.dir / name
            if not path.exists():
                raw = get(f"{BASE}/{name}")
                path.write_bytes(raw)
                HttpShards.downloaded += len(raw)
        self.wmap = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.hdrs: dict[str, tuple[dict[str, Any], int]] = {}

    def _hdr(self, fn: str):
        if fn in self.hdrs:
            return self.hdrs[fn]
        hp = self.dir / (fn.replace("/", "_") + ".header.json")
        if hp.exists():
            p = json.loads(hp.read_text())
            self.hdrs[fn] = p["header"], int(p["base"])
            return self.hdrs[fn]
        nraw = get(f"{BASE}/{fn}", (0, 7))
        n = struct.unpack("<Q", nraw)[0]
        raw = get(f"{BASE}/{fn}", (8, 8 + n - 1))
        h = json.loads(raw)
        h.pop("__metadata__", None)
        self.hdrs[fn] = h, 8 + n
        hp.write_text(json.dumps({"header": h, "base": 8 + n}))
        HttpShards.downloaded += len(nraw) + len(raw)
        return self.hdrs[fn]

    def meta(self, name: str):
        fn = self.wmap[name]
        h, base = self._hdr(fn)
        v = h[name]
        a, b = map(int, v["data_offsets"])
        return fn, v["dtype"], tuple(v["shape"]), base + a, base + b

    def get(self, name: str):
        cp = self.dir / (hashlib.sha256(name.encode()).hexdigest()[:20] + ".npy")
        if cp.exists():
            return np.load(cp, mmap_mode="r")
        fn, dtype, shape, a, b = self.meta(name)
        raw = get(f"{BASE}/{fn}", (a, b - 1))
        HttpShards.downloaded += len(raw)
        if dtype == "BF16":
            x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F32":
            x = np.frombuffer(raw, np.float32)
        elif dtype == "U8":
            x = np.frombuffer(raw, np.uint8)
        else:
            raise ValueError(dtype)
        np.save(cp, x.reshape(shape))
        return np.load(cp, mmap_mode="r")

    def rows(self, name: str, rows: list[int]):
        fn, dtype, shape, start, _ = self.meta(name)
        assert dtype == "BF16" and len(shape) == 2
        width = shape[1]
        out = np.empty((len(rows), width), np.float32)
        for i, row in enumerate(rows):
            a = start + row * width * 2
            raw = get(f"{BASE}/{fn}", (a, a + width * 2 - 1))
            HttpShards.downloaded += len(raw)
            out[i] = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        return out


def token_ids(model: k3_ref.K3Ref, count: int) -> list[int]:
    path = Path(model.S.dir) / "tiktoken.model"
    if not path.exists():
        raw = get(f"{BASE}/tiktoken.model")
        path.write_bytes(raw)
        HttpShards.downloaded += len(raw)
    vocab = {}
    for line in path.read_bytes().splitlines():
        if line.strip():
            t, rank = line.split()
            vocab[base64.b64decode(t)] = int(rank)
    pieces = [b"Hello", b",", b" world", b"!", b" Write", b" a", b" Python",
              b" function", b" that", b" solves", b" the", b" problem", b"."]
    ids = []
    for piece in pieces:
        ids.extend([vocab[piece]] if piece in vocab else [vocab[bytes([b])] for b in piece])
        if len(ids) >= count:
            break
    return (ids * math.ceil(count / len(ids)))[:count]


def layer1_input(m: k3_ref.K3Ref, x0: np.ndarray) -> np.ndarray:
    """Exact K3 execution through layer-1 attention, stopping before layer-1 MoE."""
    hidden = x0.copy()
    bres = []
    for li in (0, 1):
        lp = f"model.layers.{li}."
        in_ln = m.t(lp + "input_layernorm.weight")
        post_ln = m.t(lp + "post_attention_layernorm.weight")
        asw = m.t(lp + "self_attention_res_norm.weight") * m.t(lp + "self_attention_res_proj.weight").reshape(-1)
        msw = m.t(lp + "mlp_res_norm.weight") * m.t(lp + "mlp_res_proj.weight").reshape(-1)
        prefix = hidden.copy()
        have_prefix = True
        if bres:
            hidden = np.stack([k3_ref.res_mix(prefix[t], [b[t] for b in bres], asw, m.eps)
                               for t in range(len(prefix))])
        if li % m.res_bs == 0:
            bres.append(prefix.copy())
            have_prefix = False
        nrm = k3_ref.rmsnorm(hidden, in_ln, m.eps)
        att = m.kda(li, nrm) if li in m.kda_layers else m.mla(li, nrm)
        prefix = prefix + att if have_prefix else att
        mixed = np.stack([k3_ref.res_mix(prefix[t], [b[t] for b in bres], msw, m.eps)
                          for t in range(len(prefix))])
        mlp_input = k3_ref.rmsnorm(mixed, post_ln, m.eps)
        log(f"prefix layer {li} attention done")
        if li == 1:
            return mlp_input.astype(np.float32)
        hidden = (prefix + m.dense(li, mlp_input)).astype(np.float32)
        log("prefix layer 0 dense MLP done")
    raise AssertionError


def q(values):
    x = np.asarray(values, np.float64)
    return {"mean": float(x.mean()), "p10": float(np.quantile(x, .1)),
            "median": float(np.quantile(x, .5)), "p90": float(np.quantile(x, .9)),
            "max": float(x.max())}


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    k3_ref.Shards = HttpShards
    m = k3_ref.K3Ref(args.cache, 2)
    ids = token_ids(m, args.tokens)
    embed = next(k for k in m.S.wmap if k.endswith("model.embed_tokens.weight"))
    x0 = m.S.rows(embed, ids)
    log(f"token ids: {ids}")
    x = layer1_input(m, x0)
    p = "model.layers.1.block_sparse_moe."
    router = m.t(p + "gate.weight")
    bias = m.t(p + "gate.e_score_correction_bias")
    score = k3_ref.sigmoid(x @ router.T)
    K = int(m.tc["num_experts_per_token"])
    sel = np.argsort(-(score + bias[None]), axis=1, kind="stable")[:, :K]
    sw = np.take_along_axis(score, sel, axis=1); sw /= sw.sum(1, keepdims=True)
    z = x @ m.t(p + "routed_expert_down_proj.weight").T
    calls = defaultdict(list)
    for t in range(len(ids)):
        for e, w in zip(sel[t], sw[t]): calls[int(e)].append((t, float(w)))
    log(f"{len(ids)*K} routed calls; {len(calls)} unique experts")

    ns = sorted({min(int(v), int(m.tc["moe_intermediate_size"]))
                 for v in args.neurons.split(",") if int(v) > 0})
    rows = []
    t0 = time.time()
    for pos, (e, ecalls) in enumerate(sorted(calls.items()), 1):
        ep = f"{p}experts.{e}."
        log(f"expert {pos}/{len(calls)} id={e}, calls={len(ecalls)}")
        W1 = k3_ref.deq_mx4(m.t(ep + "w1.weight_packed"), m.t(ep + "w1.weight_scale"))
        W2 = k3_ref.deq_mx4(m.t(ep + "w2.weight_packed"), m.t(ep + "w2.weight_scale"))
        W3 = k3_ref.deq_mx4(m.t(ep + "w3.weight_packed"), m.t(ep + "w3.weight_scale"))
        cn2 = np.einsum("ij,ij->j", W2, W2, dtype=np.float64); cn = np.sqrt(cn2)
        for t, route_w in ecalls:
            a = k3_ref.situ(W1 @ z[t], W3 @ z[t], m.b1, m.b2).astype(np.float32)
            imp = np.abs(a.astype(np.float64)) * cn
            order = np.argsort(-imp, kind="stable")
            aa = a[order]; ww = W2[:, order]
            cum = np.cumsum(ww * aa[None], axis=1, dtype=np.float32)
            full = cum[:, -1]; yn = float(np.linalg.norm(full.astype(np.float64))) + 1e-30
            si = imp[order]; tri = np.cumsum(si[::-1])[::-1]
            a2 = np.cumsum((aa.astype(np.float64)**2)[::-1])[::-1]
            n2 = np.cumsum(cn2[order][::-1])[::-1]
            row = {"token": t, "expert": e, "route_weight": route_w,
                   "activation_pr": float(np.abs(a).sum()**2 / (np.sum(a.astype(np.float64)**2)+1e-30)),
                   "contribution_pr": float(imp.sum()**2 / (np.sum(imp**2)+1e-30)),
                   "partial": {}}
            for n in ns:
                part = cum[:, n-1]
                rel = float(np.linalg.norm(full.astype(np.float64)-part.astype(np.float64))/yn)
                cos = float(np.dot(full.astype(np.float64), part.astype(np.float64)) /
                            (yn*(np.linalg.norm(part.astype(np.float64))+1e-30)))
                row["partial"][str(n)] = {
                    "read_fraction": n/len(a), "actual_rel_l2": rel, "cosine": cos,
                    "triangle_bound_rel": float((tri[n] if n < len(a) else 0)/yn),
                    "frob_bound_rel": float((math.sqrt(a2[n]*n2[n]) if n < len(a) else 0)/yn),
                    "importance_mass_kept": float(si[:n].sum()/(si.sum()+1e-30))}
            rows.append(row)
            del a, imp, order, aa, ww, cum, full
        del W1, W2, W3
        log(f"  {time.time()-t0:.1f}s; range bytes {HttpShards.downloaded/1e9:.3f} GB")

    agg = {}
    for n in ns:
        k = str(n)
        actual = [r["partial"][k]["actual_rel_l2"] for r in rows]
        tri = [r["partial"][k]["triangle_bound_rel"] for r in rows]
        frob = [r["partial"][k]["frob_bound_rel"] for r in rows]
        mass = [r["partial"][k]["importance_mass_kept"] for r in rows]
        agg[k] = {"read_fraction": n/int(m.tc["moe_intermediate_size"]),
                  "actual_rel_l2": q(actual), "triangle_bound_rel": q(tri),
                  "frob_bound_rel": q(frob), "importance_mass_kept": q(mass),
                  "fraction_actual_le_1pct": float(np.mean(np.asarray(actual)<=.01)),
                  "fraction_triangle_bound_le_1pct": float(np.mean(np.asarray(tri)<=.01))}
    if "128" in agg and agg["128"]["actual_rel_l2"]["median"] <= .01:
        verdict = "GO_APPROX_SPARSE"
        interpretation = "Real routed K3 experts concentrate enough output in 128 neurons to justify a resident index and exact selected-neuron runtime; certification is the next gate."
    elif "256" in agg and agg["256"]["actual_rel_l2"]["median"] <= .03:
        verdict = "GO_CONCENTRATION_PARTIAL"
        interpretation = "Real routed K3 experts show useful activation-conditioned concentration, but a tighter index and tail certificate are required."
    else:
        verdict = "NO_STATIC_NEURON_SPARSE"
        interpretation = "Simple top-contribution neuron streaming is not concentrated enough on this real K3 prefix."
    result = {"schema":"k3-cas-neuron-v1", "token_ids":ids, "tokens":len(ids),
              "calls":len(rows), "unique_experts":len(calls), "bytes_downloaded":HttpShards.downloaded,
              "activation_pr":q([r["activation_pr"] for r in rows]),
              "contribution_pr":q([r["contribution_pr"] for r in rows]),
              "aggregate":agg, "verdict":verdict, "interpretation":interpretation,
              "rows":rows}
    (out/"result.json").write_text(json.dumps(result, indent=2))
    md=["# Kimi K3 Real Activation Neuron Probe","",f"**{verdict}**","",interpretation,"",
        f"Tokens: {len(ids)}; calls: {len(rows)}; unique experts: {len(calls)}; downloaded: {HttpShards.downloaded/1e9:.3f} GB","",
        "| neurons | bytes | median error | p90 error | median triangle bound | median frob bound | median mass |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for n in ns:
        a=agg[str(n)]; md.append(f"| {n} | {100*a['read_fraction']:.2f}% | {a['actual_rel_l2']['median']:.4g} | {a['actual_rel_l2']['p90']:.4g} | {a['triangle_bound_rel']['median']:.4g} | {a['frob_bound_rel']['median']:.4g} | {a['importance_mass_kept']['median']:.4f} |")
    md += ["",f"Median activation PR: {result['activation_pr']['median']:.1f}/3072",
           f"Median contribution PR: {result['contribution_pr']['median']:.1f}/3072"]
    (out/"REPORT.md").write_text("\n".join(md)+"\n")
    log(f"{verdict}: {interpretation}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tokens",type=int,default=4)
    ap.add_argument("--neurons",default="16,32,64,128,256,512,1024,1536,2048,3072")
    ap.add_argument("--cache",default="work/cache"); ap.add_argument("--out",default="work/result")
    run(ap.parse_args())


if __name__ == "__main__": main()
