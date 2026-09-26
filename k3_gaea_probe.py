#!/usr/bin/env python3
"""Kimi K3 gauge-aligned expert factor-stack probe.

This script tests the one exact representation branch that raw expert-basis
studies cannot test: align the 3072 hidden neurons of each SiTU-GLU expert
under the exact permutation symmetry

    W1 -> P W1,  W3 -> P W3,  W2 -> W2 P^T.

It range-reads a small set of native K3 MXFP4 experts, dequantizes them exactly,
solves a hidden-unit assignment against a reference expert, and reports:
  * raw vs aligned neuron-triple similarity;
  * aligned factor-stack spectrum (not product-space spectrum);
  * exact modal-value reuse after alignment;
  * the MXFP4 W2 scale-group fragmentation caused by the permutation;
  * calibrated synthetic positive and negative controls.

It does NOT claim a full runtime solution. It is a decisive measurement of an
unmeasured branch and is designed to fail loudly if the alignment machinery
cannot recover a known exact permutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main"
INDEX_NAME = "model.safetensors.index.json"
GROUP = 32
E2M1 = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def http(url: str, byte_range: tuple[int, int] | None = None, retries: int = 6) -> bytes:
    headers = {"User-Agent": "k3-gaea-probe/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as response:
                return response.read()
        except Exception as exc:
            last = exc
            wait = 2.0 * (attempt + 1)
            log(f"  retry {attempt + 1}/{retries} after {exc!r}; sleeping {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {retries} attempts: {last}")


@dataclass
class ShardHeader:
    header: dict[str, Any]
    data_start: int


class K3RangeReader:
    def __init__(self, cache: Path):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        index_path = cache / INDEX_NAME
        if not index_path.exists():
            log("Downloading K3 safetensors index (~60 MB) ...")
            index_path.write_bytes(http(f"{BASE}/{INDEX_NAME}"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self.headers: dict[str, ShardHeader] = {}
        self.bytes_downloaded = 0

    def load_header(self, shard: str) -> ShardHeader:
        if shard in self.headers:
            return self.headers[shard]
        hp = self.cache / f"{shard}.header.json"
        if hp.exists():
            payload = json.loads(hp.read_text(encoding="utf-8"))
            item = ShardHeader(payload["header"], int(payload["data_start"]))
            self.headers[shard] = item
            return item
        n = struct.unpack("<Q", http(f"{BASE}/{shard}", (0, 7)))[0]
        raw = http(f"{BASE}/{shard}", (8, 8 + n - 1))
        header = json.loads(raw)
        header.pop("__metadata__", None)
        item = ShardHeader(header, 8 + n)
        hp.write_text(
            json.dumps({"header": header, "data_start": item.data_start}),
            encoding="utf-8",
        )
        self.headers[shard] = item
        return item

    def tensor_u8(self, name: str) -> np.ndarray:
        out = self.cache / (name.replace(".", "_") + ".npy")
        if out.exists():
            return np.load(out, mmap_mode="r")
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"tensor not present in index: {name}")
        sh = self.load_header(shard)
        meta = sh.header[name]
        if meta["dtype"] != "U8":
            raise TypeError(f"expected U8 tensor for {name}, got {meta['dtype']}")
        a, b = map(int, meta["data_offsets"])
        raw = http(f"{BASE}/{shard}", (sh.data_start + a, sh.data_start + b - 1))
        expected = int(np.prod(meta["shape"], dtype=np.int64))
        if len(raw) != expected:
            raise IOError(f"short range read for {name}: {len(raw)} != {expected}")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(meta["shape"])
        np.save(out, arr)
        self.bytes_downloaded += len(raw)
        return np.load(out, mmap_mode="r")

    def matrix(self, stem: str) -> np.ndarray:
        packed = np.asarray(self.tensor_u8(f"{stem}.weight_packed"), dtype=np.uint8)
        scales = np.asarray(self.tensor_u8(f"{stem}.weight_scale"), dtype=np.uint8)
        return dequantize_mxfp4(packed, scales)


def dequantize_mxfp4(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("MXFP4 packed and scale tensors must both be matrices")
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    if scales.shape != (rows, cols // GROUP):
        raise ValueError(f"scale shape mismatch: packed={packed.shape}, scales={scales.shape}")
    nibbles = np.empty((rows, cols), dtype=np.uint8)
    nibbles[:, 0::2] = packed & 0x0F
    nibbles[:, 1::2] = packed >> 4
    values = E2M1[nibbles]
    factors = np.exp2(scales.astype(np.int16).astype(np.float32) - 127.0)
    return values * np.repeat(factors, GROUP, axis=1)


def norm_rows(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


@dataclass
class Projectors:
    r1: np.ndarray
    r3: np.ndarray
    r2: np.ndarray


def make_projectors(input_dim: int, output_dim: int, d: int, seed: int) -> Projectors:
    rng = np.random.default_rng(seed)
    def rademacher(rows: int) -> np.ndarray:
        signs = rng.integers(0, 2, size=(rows, d), dtype=np.int8)
        return ((signs * 2 - 1).astype(np.float32) / math.sqrt(d))
    return Projectors(rademacher(input_dim), rademacher(input_dim), rademacher(output_dim))


def descriptor(w1: np.ndarray, w2: np.ndarray, w3: np.ndarray, p: Projectors) -> np.ndarray:
    if w1.shape != w3.shape:
        raise ValueError("w1/w3 shape mismatch")
    inter, latent = w1.shape
    if w2.shape != (latent, inter):
        raise ValueError(f"w2 shape mismatch: {w2.shape} vs {(latent, inter)}")
    a = norm_rows(w1 @ p.r1)
    b = norm_rows(w3 @ p.r3)
    c = norm_rows(w2.T @ p.r2)
    mags = np.stack(
        [
            np.log1p(np.linalg.norm(w1, axis=1)),
            np.log1p(np.linalg.norm(w3, axis=1)),
            np.log1p(np.linalg.norm(w2, axis=0)),
        ],
        axis=1,
    ).astype(np.float32)
    mags = (mags - mags.mean(axis=0, keepdims=True)) / np.maximum(mags.std(axis=0, keepdims=True), 1e-6)
    mags *= 0.15
    return norm_rows(np.concatenate([a, b, c, mags], axis=1))


def solve_alignment(ref_desc: np.ndarray, target_desc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = ref_desc @ target_desc.T
    rows, cols = linear_sum_assignment(score, maximize=True)
    if not np.array_equal(rows, np.arange(ref_desc.shape[0])):
        raise RuntimeError("assignment rows are not canonical")
    return cols.astype(np.int32), score[rows, cols]


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.einsum("ij,ij->i", a, b, dtype=np.float64)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return (num / np.maximum(den, 1e-30)).astype(np.float64)


def summary_stats(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "mean": float(np.mean(x)), "min": float(q[0]), "p10": float(q[1]),
        "median": float(q[2]), "p90": float(q[3]), "max": float(q[4]),
    }


def rank_report_from_gram(g: np.ndarray) -> dict[str, Any]:
    g = (g + g.T) * 0.5
    ev = np.clip(np.linalg.eigvalsh(g)[::-1], 0.0, None)
    total = float(ev.sum())
    if total <= 0:
        return {"eigenvalues": ev.tolist(), "participation_ratio": 0.0, "ranks": {}}
    tail = np.clip(1.0 - np.cumsum(ev) / total, 0.0, None)
    ranks: dict[str, int] = {}
    for err in (0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        ok = np.nonzero(np.sqrt(tail) <= err)[0]
        ranks[f"{int(err * 100)}pct_error"] = int(ok[0] + 1 if len(ok) else len(ev))
    p = ev / total
    return {
        "eigenvalues": [float(v) for v in ev],
        "eigenvalue_fractions": [float(v) for v in p],
        "participation_ratio": float(1.0 / np.sum(p * p)),
        "ranks": ranks,
    }


def centered_gram(g: np.ndarray) -> np.ndarray:
    n = g.shape[0]
    h = np.eye(n, dtype=np.float64) - np.full((n, n), 1.0 / n, dtype=np.float64)
    return h @ g.astype(np.float64) @ h


def modal_reuse(m: np.memmap, sample_count: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    total = m.shape[1]
    k = min(sample_count, total)
    idx = np.sort(rng.choice(total, size=k, replace=False))
    values = np.asarray(m[:, idx], dtype=np.float32).view(np.uint32)
    values.sort(axis=0)
    n_exp = values.shape[0]
    current = np.ones(k, dtype=np.uint8)
    best = np.ones(k, dtype=np.uint8)
    entropy = np.zeros(k, dtype=np.float64)
    for i in range(1, n_exp):
        same = values[i] == values[i - 1]
        ended = ~same
        if np.any(ended):
            p = current[ended].astype(np.float64) / n_exp
            entropy[ended] -= p * np.log2(p)
        current = np.where(same, current + 1, 1).astype(np.uint8)
        best = np.maximum(best, current)
    p = current.astype(np.float64) / n_exp
    entropy -= p * np.log2(p)
    mode_fraction = best.astype(np.float64) / n_exp
    match = float(mode_fraction.mean())
    changed = 1.0 - match
    if changed <= 0.0 or changed >= 1.0:
        bitmap_entropy = 0.0
    else:
        bitmap_entropy = -changed * math.log2(changed) - (1.0 - changed) * math.log2(1.0 - changed)
    return {
        "sample_positions": int(k),
        "mean_modal_fraction": match,
        "mean_changed_fraction": changed,
        "mean_position_entropy_bits": float(entropy.mean()),
        "optimistic_match_bitmap_floor_bpw": float(bitmap_entropy),
        "mode_fraction_quantiles": {
            "p10": float(np.quantile(mode_fraction, 0.10)),
            "median": float(np.quantile(mode_fraction, 0.50)),
            "p90": float(np.quantile(mode_fraction, 0.90)),
            "max": float(mode_fraction.max()),
        },
        "fraction_positions_mode_ge_2": float(np.mean(best >= 2)),
        "fraction_positions_mode_ge_4": float(np.mean(best >= 4)),
        "fraction_positions_mode_ge_half": float(np.mean(best >= math.ceil(n_exp / 2))),
    }


def group_fragmentation(perm: np.ndarray) -> dict[str, float]:
    n = len(perm)
    if n % GROUP:
        raise ValueError("intermediate dimension must be divisible by MXFP4 group size")
    groups = perm.reshape(-1, GROUP) // GROUP
    unique = np.asarray([len(np.unique(row)) for row in groups], dtype=np.float64)
    ref_groups = np.arange(n, dtype=np.int32).reshape(-1, GROUP) // GROUP
    return {
        "mean_original_scale_groups_per_aligned_group": float(unique.mean()),
        "min_original_scale_groups_per_aligned_group": float(unique.min()),
        "max_original_scale_groups_per_aligned_group": float(unique.max()),
        "fraction_columns_remaining_in_same_group_id": float(np.mean(groups == ref_groups)),
        "fraction_columns_remaining_in_same_lane": float(np.mean((perm % GROUP) == (np.arange(n) % GROUP))),
    }


def situ_glu(g: np.ndarray, u: np.ndarray) -> np.ndarray:
    return (4.0 * np.tanh(g / 4.0) / (1.0 + np.exp(-g))) * (25.0 * np.tanh(u / 25.0))


def expert_forward(w1: np.ndarray, w2: np.ndarray, w3: np.ndarray, x: np.ndarray) -> np.ndarray:
    return w2 @ situ_glu(w1 @ x, w3 @ x)


def synthetic_controls(seed: int = 20260926) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    latent, inter, experts = 96, 64, 8
    p = make_projectors(latent, latent, 32, seed + 1)
    base = (
        rng.standard_normal((inter, latent), dtype=np.float32),
        rng.standard_normal((latent, inter), dtype=np.float32),
        rng.standard_normal((inter, latent), dtype=np.float32),
    )
    ref_desc = descriptor(*base, p)
    recovered = []
    max_output_diff = 0.0
    aligned_vectors = []
    raw_vectors = []
    x = rng.standard_normal(latent, dtype=np.float32)
    for _ in range(experts):
        perm = rng.permutation(inter)
        w1 = base[0][perm].copy()
        w2 = base[1][:, perm].copy()
        w3 = base[2][perm].copy()
        raw_vectors.append(np.concatenate([w1.ravel(), w2.ravel(), w3.ravel()]))
        got, _ = solve_alignment(ref_desc, descriptor(w1, w2, w3, p))
        aw1, aw2, aw3 = w1[got], w2[:, got], w3[got]
        aligned_vectors.append(np.concatenate([aw1.ravel(), aw2.ravel(), aw3.ravel()]))
        recovered.append(float(np.mean(got == np.argsort(perm))))
        max_output_diff = max(max_output_diff, float(np.max(np.abs(
            expert_forward(w1, w2, w3, x) - expert_forward(aw1, aw2, aw3, x)
        ))))
    raw = np.stack(raw_vectors).astype(np.float64)
    aligned = np.stack(aligned_vectors).astype(np.float64)
    raw_rank = rank_report_from_gram(centered_gram(raw @ raw.T))
    aligned_rank = rank_report_from_gram(centered_gram(aligned @ aligned.T))

    neg_vecs = []
    neg_ref = None
    for _ in range(experts):
        w1 = rng.standard_normal((inter, latent), dtype=np.float32)
        w2 = rng.standard_normal((latent, inter), dtype=np.float32)
        w3 = rng.standard_normal((inter, latent), dtype=np.float32)
        d = descriptor(w1, w2, w3, p)
        if neg_ref is None:
            neg_ref = d
            got = np.arange(inter)
        else:
            got, _ = solve_alignment(neg_ref, d)
        neg_vecs.append(np.concatenate([w1[got].ravel(), w2[:, got].ravel(), w3[got].ravel()]))
    neg = np.stack(neg_vecs).astype(np.float64)
    neg_rank = rank_report_from_gram(centered_gram(neg @ neg.T))

    result = {
        "exact_permutation_recovery_mean": float(np.mean(recovered)),
        "exact_permutation_recovery_min": float(np.min(recovered)),
        "max_forward_difference_after_exact_permutation": max_output_diff,
        "raw_permuted_centered_rank": raw_rank,
        "aligned_identical_centered_rank": aligned_rank,
        "aligned_random_negative_centered_rank": neg_rank,
    }
    if result["exact_permutation_recovery_min"] != 1.0:
        raise AssertionError(f"synthetic permutation recovery failed: {result}")
    if max_output_diff > 5e-4:
        raise AssertionError(f"permutation invariance control failed: {max_output_diff}")
    if neg_rank["participation_ratio"] < 4.0:
        raise AssertionError("negative-control experts collapsed unexpectedly")
    return result


def load_expert(reader: K3RangeReader, layer: int, expert: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"
    return reader.matrix(f"{stem}.w1"), reader.matrix(f"{stem}.w2"), reader.matrix(f"{stem}.w3")


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi K3 Gauge-Aligned Expert Probe", "",
        f"- Layer: `{result['layer']}`", f"- Experts sampled: `{result['experts']}`",
        f"- Bytes range-downloaded: `{result['bytes_downloaded']:,}`",
        f"- Factor-stack elements per expert: `{result['factor_elements_per_expert']:,}`", "",
        "## Calibration controls", "",
        f"- Exact permutation recovery: `{result['synthetic_controls']['exact_permutation_recovery_min']:.3f}` minimum",
        f"- Max forward difference after exact permutation: `{result['synthetic_controls']['max_forward_difference_after_exact_permutation']:.3e}`",
        "", "## Real K3 alignment", "",
    ]
    for item in result["per_expert_alignment"]:
        lines.append(
            f"- Expert {item['expert']}: descriptor cosine raw `{item['descriptor_raw']['mean']:.4f}` → "
            f"aligned `{item['descriptor_aligned']['mean']:.4f}`; exact triple cosine "
            f"w1 `{item['w1_aligned_cosine']['mean']:.4f}`, w3 `{item['w3_aligned_cosine']['mean']:.4f}`, "
            f"w2 `{item['w2_aligned_cosine']['mean']:.4f}`."
        )
    fs = result["factor_stack"]
    lines += [
        "", "## Aligned factor-stack spectrum", "",
        f"- Uncentred participation ratio: `{fs['uncentred']['participation_ratio']:.3f}`",
        f"- Centred participation ratio: `{fs['centred']['participation_ratio']:.3f}`",
        f"- Rank for 10% centred reconstruction error: `{fs['centred']['ranks']['10pct_error']}` / `{result['experts']}`",
        f"- Rank for 20% centred reconstruction error: `{fs['centred']['ranks']['20pct_error']}` / `{result['experts']}`",
        "", "## Exact modal-value reuse after alignment", "",
        f"- Mean modal fraction: `{result['modal_reuse']['mean_modal_fraction']:.4f}`",
        f"- Mean per-position entropy: `{result['modal_reuse']['mean_position_entropy_bits']:.4f}` bits/weight",
        f"- Optimistic match-bitmap floor: `{result['modal_reuse']['optimistic_match_bitmap_floor_bpw']:.4f}` bits/weight",
        "", "## W2 MXFP4 group fragmentation", "",
        f"- Mean original scale groups represented inside one aligned 32-column group: "
        f"`{result['w2_group_fragmentation_mean']['mean_original_scale_groups_per_aligned_group']:.2f}`",
        f"- Fraction staying in original group id: "
        f"`{result['w2_group_fragmentation_mean']['fraction_columns_remaining_in_same_group_id']:.4f}`",
        "", "## Interpretation", "", result["interpretation"], "",
        "This artifact is a measurement of the exact permutation-gauge branch. It is not, by itself, a complete runtime implementation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    controls = synthetic_controls(args.seed)
    log("Synthetic calibration controls: PASS")
    reader = K3RangeReader(Path(args.cache))
    log(f"Loading reference expert 0 from layer {args.layer} ...")
    ref_w1, ref_w2, ref_w3 = load_expert(reader, args.layer, 0)
    inter, latent = ref_w1.shape
    if ref_w3.shape != (inter, latent) or ref_w2.shape != (latent, inter):
        raise RuntimeError(f"unexpected K3 expert shapes: w1={ref_w1.shape}, w2={ref_w2.shape}, w3={ref_w3.shape}")
    projectors = make_projectors(latent, latent, args.sketch, args.seed)
    ref_desc = descriptor(ref_w1, ref_w2, ref_w3, projectors)
    total_elements = ref_w1.size + ref_w2.size + ref_w3.size
    factor_path = out_dir / "aligned_factor_stack.f32"
    factors = np.memmap(factor_path, mode="w+", dtype=np.float32, shape=(args.experts, total_elements))

    def write_factor(row: int, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> None:
        a = 0; b = w1.size
        factors[row, a:b] = w1.ravel()
        a = b; b = a + w3.size
        factors[row, a:b] = w3.ravel()
        factors[row, b:] = w2.ravel()

    write_factor(0, ref_w1, ref_w2, ref_w3)
    alignments: list[dict[str, Any]] = []
    frag_rows: list[dict[str, float]] = []
    t0 = time.time()
    for e in range(1, args.experts):
        log(f"Expert {e}/{args.experts - 1}: range-read, align, and write factor stack ...")
        w1, w2, w3 = load_expert(reader, args.layer, e)
        d = descriptor(w1, w2, w3, projectors)
        perm, matched_score = solve_alignment(ref_desc, d)
        raw_diag = np.einsum("ij,ij->i", ref_desc, d, dtype=np.float64)
        aw1, aw3, aw2 = w1[perm], w3[perm], w2[:, perm]
        frag = group_fragmentation(perm)
        frag_rows.append(frag)
        write_factor(e, aw1, aw2, aw3)
        alignments.append({
            "expert": e,
            "descriptor_raw": summary_stats(raw_diag),
            "descriptor_aligned": summary_stats(matched_score),
            "w1_aligned_cosine": summary_stats(row_cos(ref_w1, aw1)),
            "w3_aligned_cosine": summary_stats(row_cos(ref_w3, aw3)),
            "w2_aligned_cosine": summary_stats(row_cos(ref_w2.T, aw2.T)),
            "w2_group_fragmentation": frag,
            "permutation_sha256": hashlib.sha256(perm.tobytes()).hexdigest(),
        })
        del w1, w2, w3, aw1, aw2, aw3, d
        log(f"  done in {time.time() - t0:.1f}s cumulative; downloaded {reader.bytes_downloaded / 1e6:.1f} MB")

    factors.flush()
    log("Computing aligned factor-stack Gram matrix ...")
    gram = np.asarray(factors) @ np.asarray(factors).T
    fs_report = {
        "uncentred": rank_report_from_gram(gram.astype(np.float64)),
        "centred": rank_report_from_gram(centered_gram(gram)),
    }
    log("Sampling exact modal-value reuse across aligned experts ...")
    modal = modal_reuse(factors, args.mode_samples, args.seed + 77)
    del factors
    factor_path.unlink(missing_ok=True)

    frag_mean = {key: float(np.mean([row[key] for row in frag_rows])) for key in frag_rows[0]}
    desc_gain = float(np.mean([
        row["descriptor_aligned"]["mean"] - row["descriptor_raw"]["mean"] for row in alignments
    ]))
    centered_r10 = fs_report["centred"]["ranks"]["10pct_error"]
    optimistic_bpw = modal["mean_position_entropy_bits"]
    if desc_gain > 0.20 and centered_r10 <= max(2, args.experts // 3) and optimistic_bpw < 0.25:
        interpretation = (
            "GO for the next implementation stage: exact gauge alignment exposes a compact aligned factor stack and "
            "substantial exact value reuse. Build the layer-wide aligned prototype/basis container and a certified "
            "residual runtime; do not return to raw expert-space basis tests."
        )
    elif desc_gain > 0.10:
        interpretation = (
            "PARTIAL GO: the exact permutation gauge is real and materially improves correspondence, but this sample "
            "does not yet prove the compression ratio required by the laptop target. The next branch is activation-"
            "conditioned aligned charts with exact residual fallback, not another raw-weight shared basis."
        )
    else:
        interpretation = (
            "NO GO for simple permutation-aligned static factorization on this sample. The calibration passed, so the "
            "result is not the old raw-gauge false negative. The remaining positive branch is activation-conditioned "
            "certified charts (resident projected images plus exact fallback), which must be measured on routed K3 activation traces."
        )

    result = {
        "schema": "k3-gaea-probe-v1",
        "source_model": "moonshotai/Kimi-K3",
        "layer": args.layer,
        "experts": args.experts,
        "sketch_dimension_per_component": args.sketch,
        "seed": args.seed,
        "expert_shapes": {"w1": list(ref_w1.shape), "w2": list(ref_w2.shape), "w3": list(ref_w3.shape)},
        "factor_elements_per_expert": int(total_elements),
        "bytes_downloaded": int(reader.bytes_downloaded),
        "synthetic_controls": controls,
        "per_expert_alignment": alignments,
        "factor_stack": fs_report,
        "modal_reuse": modal,
        "w2_group_fragmentation_mean": frag_mean,
        "mean_descriptor_alignment_gain": desc_gain,
        "interpretation": interpretation,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(out_dir / "REPORT.md", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--sketch", type=int, default=64)
    ap.add_argument("--mode-samples", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--cache", default="work/cache")
    ap.add_argument("--out", default="work/result")
    args = ap.parse_args()
    if args.experts < 2:
        raise SystemExit("--experts must be at least 2")
    if args.synthetic_only:
        print(json.dumps(synthetic_controls(args.seed), indent=2))
        return 0
    result = run_real(args)
    log("\n" + result["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
