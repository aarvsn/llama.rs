#!/usr/bin/env python3
"""Generates reference values for llama.rs kernel tests.

Independently translates the ggml reference dequantization routines
(ggml-quants.c from ggml/ggml) into Python and emits random blocks plus
expected dequantized values and dot products as JSON. The Rust test suite
compares its kernels against these numbers, validating the block layouts
bit-for-bit without any C toolchain.

Usage: python3 scripts/generate_reference_values.py > tests/reference_values.json
"""

import json
import math
import struct
import sys

QK_K = 256
N_BLOCKS_PER_TYPE = 6


def lcg_bytes(seed: int, n: int) -> bytes:
    """Deterministic pseudo-random bytes (xorshift64* style)."""
    out = bytearray()
    state = seed & 0xFFFFFFFFFFFFFFFF
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append((state >> 56) & 0xFF)
    return bytes(out)


def lcg_floats(seed: int, n: int) -> list:
    out = []
    state = seed & 0xFFFFFFFFFFFFFFFF
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        v = ((state >> 40) & 0xFFFFFF) / float(1 << 24)  # [0, 1)
        out.append(v * 2.0 - 1.0)
    return out


def f16_to_f32(bits: int) -> float:
    """IEEE 754 binary16 -> float (exact)."""
    sign = -1.0 if bits & 0x8000 else 1.0
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x03FF
    if exp == 0:
        return sign * mant * 2.0**-24
    if exp == 0x1F:
        return sign * (math.inf if mant == 0 else math.nan)
    return sign * (1.0 + mant / 1024.0) * 2.0 ** (exp - 15)


def f32_to_f16(value: float) -> int:
    """Round-to-nearest-even float -> binary16 bits (via struct)."""
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0] if False else struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return 0x7C00


# ---------------------------------------------------------------------------
# Reference dequantizers, translated from ggml-quants.c
# ---------------------------------------------------------------------------

def dequant_q4_0(data: bytes) -> list:
    out = []
    for b in range(len(data) // 18):
        d = f16_to_f32(struct.unpack_from("<H", data, b * 18)[0])
        qs = data[b * 18 + 2 : b * 18 + 18]
        for j in range(16):
            out.append(((qs[j] & 0x0F) - 8) * d)
        for j in range(16):
            out.append(((qs[j] >> 4) - 8) * d)
    return out


def dequant_q4_1(data: bytes) -> list:
    out = []
    for b in range(len(data) // 20):
        d = f16_to_f32(struct.unpack_from("<H", data, b * 20)[0])
        m = f16_to_f32(struct.unpack_from("<H", data, b * 20 + 2)[0])
        qs = data[b * 20 + 4 : b * 20 + 20]
        for j in range(16):
            out.append((qs[j] & 0x0F) * d + m)
        for j in range(16):
            out.append((qs[j] >> 4) * d + m)
    return out


def dequant_q5_0(data: bytes) -> list:
    out = []
    for b in range(len(data) // 22):
        d = f16_to_f32(struct.unpack_from("<H", data, b * 22)[0])
        qh = struct.unpack_from("<I", data, b * 22 + 2)[0]
        qs = data[b * 22 + 6 : b * 22 + 22]
        first, second = [], []
        for j in range(16):
            xh0 = ((qh >> j) & 1) << 4
            xh1 = ((qh >> (j + 16)) & 1) << 4
            first.append((((qs[j] & 0x0F) | xh0) - 16) * d)
            second.append((((qs[j] >> 4) | xh1) - 16) * d)
        out += first
        out += second
    return out


def dequant_q5_1(data: bytes) -> list:
    out = []
    for b in range(len(data) // 24):
        d = f16_to_f32(struct.unpack_from("<H", data, b * 24)[0])
        m = f16_to_f32(struct.unpack_from("<H", data, b * 24 + 2)[0])
        qh = struct.unpack_from("<I", data, b * 24 + 4)[0]
        qs = data[b * 24 + 8 : b * 24 + 24]
        first, second = [], []
        for j in range(16):
            xh0 = ((qh >> j) & 1) << 4
            xh1 = ((qh >> (j + 16)) & 1) << 4
            first.append(((qs[j] & 0x0F) | xh0) * d + m)
            second.append(((qs[j] >> 4) | xh1) * d + m)
        out += first
        out += second
    return out


def dequant_q8_0(data: bytes) -> list:
    out = []
    for b in range(len(data) // 34):
        d = f16_to_f32(struct.unpack_from("<H", data, b * 34)[0])
        qs = data[b * 34 + 2 : b * 34 + 34]
        for j in range(32):
            q = qs[j] if qs[j] < 128 else qs[j] - 256
            out.append(q * d)
    return out


def get_scale_min_k4(j: int, q: bytes):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d, m


def dequant_q4_K(data: bytes) -> list:
    out = []
    for b in range(len(data) // 144):
        off = b * 144
        d = f16_to_f32(struct.unpack_from("<H", data, off)[0])
        dmin = f16_to_f32(struct.unpack_from("<H", data, off + 2)[0])
        scales = data[off + 4 : off + 16]
        q = data[off + 16 : off + 144]
        is_ = 0
        for j in range(0, QK_K, 64):
            sc, m = get_scale_min_k4(is_ + 0, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = get_scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            base = q[j // 2 : j // 2 + 32]
            for l in range(32):
                out.append(d1 * (base[l] & 0x0F) - m1)
            for l in range(32):
                out.append(d2 * (base[l] >> 4) - m2)
            is_ += 2
    return out


def dequant_q5_K(data: bytes) -> list:
    out = []
    for b in range(len(data) // 176):
        off = b * 176
        d = f16_to_f32(struct.unpack_from("<H", data, off)[0])
        dmin = f16_to_f32(struct.unpack_from("<H", data, off + 2)[0])
        scales = data[off + 4 : off + 16]
        qh = data[off + 16 : off + 48]
        ql = data[off + 48 : off + 176]
        is_ = 0
        u1, u2 = 1, 2
        for j in range(0, QK_K, 64):
            sc, m = get_scale_min_k4(is_ + 0, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = get_scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            base = ql[j // 2 : j // 2 + 32]
            for l in range(32):
                h = 16 if (qh[l] & u1) else 0
                out.append(d1 * ((base[l] & 0x0F) + h) - m1)
            for l in range(32):
                h = 16 if (qh[l] & u2) else 0
                out.append(d2 * ((base[l] >> 4) + h) - m2)
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return out


def dequant_q6_K(data: bytes) -> list:
    out = []
    for b in range(len(data) // 210):
        off = b * 210
        d = f16_to_f32(struct.unpack_from("<H", data, off)[0])
        ql = data[off + 2 : off + 130]
        qh = data[off + 130 : off + 194]
        sc = data[off + 194 : off + 210]
        for n in range(0, QK_K, 128):
            # The C reference advances ql += 64, qh += 32, sc += 8 per
            # 128-element half.
            qlo, qho, sco = n // 2, n // 4, n // 16
            g1, g2, g3, g4 = [], [], [], []
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[qlo + l + 0] & 0x0F) | (((qh[qho + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[qlo + l + 32] & 0x0F) | (((qh[qho + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[qlo + l + 0] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32
                s = lambda i: sc[sco + i] if sc[sco + i] < 128 else sc[sco + i] - 256
                g1.append(d * s(is_ + 0) * q1)
                g2.append(d * s(is_ + 2) * q2)
                g3.append(d * s(is_ + 4) * q3)
                g4.append(d * s(is_ + 6) * q4)
            out += g1 + g2 + g3 + g4
    return out


# ---------------------------------------------------------------------------
# Q8_0 activation quantizer (quantize_row_q8_0_ref)
# ---------------------------------------------------------------------------

def quantize_q8_0(x: list) -> bytes:
    out = bytearray()
    for b in range(len(x) // 32):
        chunk = x[b * 32 : b * 32 + 32]
        amax = max(abs(v) for v in chunk)
        d = amax / 127.0
        id_ = 1.0 / d if d != 0.0 else 0.0
        out += struct.pack("<H", f32_to_f16(d))
        for v in chunk:
            q = round_half_away(v * id_)
            q = max(-128, min(127, q))
            out += struct.pack("<b", q)
    return bytes(out)


def round_half_away(v: float) -> int:
    return int(math.floor(v + 0.5)) if v >= 0 else int(math.ceil(v - 0.5))


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

TYPES = {
    "q4_0": (18, 32, dequant_q4_0),
    "q4_1": (20, 32, dequant_q4_1),
    "q5_0": (22, 32, dequant_q5_0),
    "q5_1": (24, 32, dequant_q5_1),
    "q8_0": (34, 32, dequant_q8_0),
    "q4_k": (144, QK_K, dequant_q4_K),
    "q5_k": (176, QK_K, dequant_q5_K),
    "q6_k": (210, QK_K, dequant_q6_K),
}


def main():
    result = {}
    seed = 0x1234_ABCD
    for idx, (name, (block_bytes, block_elems, dequant)) in enumerate(TYPES.items()):
        n_blocks = N_BLOCKS_PER_TYPE
        n_elems = n_blocks * block_elems
        data = lcg_bytes(seed + idx * 7919, n_blocks * block_bytes)
        # Force benign scales so nothing is inf/nan.
        data = bytearray(data)
        for b in range(n_blocks):
            d_bits = f32_to_f16(0.05)
            data[b * block_bytes : b * block_bytes + 2] = struct.pack("<H", d_bits)
            if name in ("q4_1", "q5_1", "q4_k", "q5_k"):
                dmin_bits = f32_to_f16(0.01)
                data[b * block_bytes + 2 : b * block_bytes + 4] = struct.pack("<H", dmin_bits)
        data = bytes(data)

        expected_dequant = dequant(data)
        assert len(expected_dequant) == n_elems, (name, len(expected_dequant), n_elems)

        x = lcg_floats(seed + 991, n_elems)
        q8 = quantize_q8_0(x)
        q8_dequant = dequant_q8_0(q8)
        expected_dot = sum(a * b for a, b in zip(expected_dequant, q8_dequant))

        result[name] = {
            "blocks_hex": data.hex(),
            "n_blocks": n_blocks,
            "x": x,
            "q8_hex": q8.hex(),
            "expected_dequant": expected_dequant,
            "expected_dot": expected_dot,
        }

    json.dump(result, sys.stdout, separators=(",", ":"))
    print()


if __name__ == "__main__":
    main()
