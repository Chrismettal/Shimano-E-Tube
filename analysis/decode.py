#!/usr/bin/env python3
"""Shimano E-Tube BPSK command decoder - canonical pipeline.

Runs the full decode on a scope-capture directory and prints:
  1. capture geometry (segments, index range, points, fs)
  2. confidence-filtered command timeline (A/B by pi-immune differential head)
  3. A/B run structure per burst
  4. per-class majority-vote consensus (head + tail) and per-position agreement

Data model:
  - constant-envelope BPSK on I axis, carrier ~1.000 MHz, 16 symbols/frame.
  - per-frame abs bits are decoded vs the frame's strongest symbol (pi-ambiguous).
  - the DIFFERENTIAL head d[k]=ab[k]^ab[k-1] is invariant under a global phase
    flip, so it is the true class key (pi-immune).
  - global time order = SEGMENT INDEX (per-CSV timestamps are window-relative).

Usage:
  python decode.py <capture-dir>
"""
import sys
import numpy as np
from fsk import burst_segments
from final import lock, dh, ham, classify

BIM_MIN = 0.92        # bimodality quality gate
FC = 1.0e6            # nominal carrier
FC_TOL = 20e3         # carrier lock tolerance
N = 16                # symbols per frame


def decode_frames(d):
    """Return sorted [(seg_idx, cls, bim, fc, abs16)] for quality frames."""
    segs = burst_segments(d)
    out = []
    for idx, (_, t, v, fs) in enumerate(segs):
        if (t[-1] - t[0]) * 1e6 < 60:
            continue
        fc, vecs = lock(t, v, N)
        rk = int(np.argmax(np.abs(vecs)))
        ref = vecs[rk]
        bim = float(np.mean(np.abs(np.cos(2 * np.angle(vecs * np.conj(ref))))))
        if bim < BIM_MIN or abs(fc - FC) >= FC_TOL:
            continue
        ab = [1 if np.real(vecs[k] * np.conj(ref)) < 0 else 0 for k in range(N)]
        cls = classify(''.join(map(str, dh(ab)[:8])))
        if cls == '.':
            continue
        out.append((idx, cls, bim, fc, ab))
    out.sort()
    return out, segs


def bursts(frames, gap=3):
    """Group frames into bursts by segment-index proximity."""
    out = []
    for f in frames:
        if out and f[0] - out[-1][-1][0] <= gap:
            out[-1].append(f)
        else:
            out.append([f])
    return out


def consensus(frames, cls):
    """Group-aligned majority vote across all frames of a class.

    Frames are oriented to the strongest one via the differential head, then
    each symbol is majority-voted. Returns (abs_bits, diff_bits, agree_abs,
    agree_diff, segs).
    """
    grp = [f for f in frames if f[1] == cls]
    grp.sort(key=lambda g: -g[2])
    if not grp:
        return None
    ref = grp[0][4]
    aligned = []
    for _, _, _, _, ab in grp:
        a = ab[:]
        if ham(a, ref) > 8:
            a = [b ^ 1 for b in a]
        aligned.append(a)
    n = len(aligned)
    abs_c = [int(sum(col) > n / 2) for col in zip(*aligned)]
    diff_c = [0] + [abs_c[k] ^ abs_c[k - 1] for k in range(1, N)]
    agree_abs = [100 * sum(1 for a in aligned if a[k] == abs_c[k]) // n for k in range(N)]
    return abs_c, diff_c, agree_abs, [f[0] for f in grp]


def hex8(b):
    return format(int(''.join(map(str, b)), 2), '02x')


def main(d):
    frames, segs = decode_frames(d)
    pts = len(segs[0][1]) if segs else 0
    fs = segs[0][3] if segs else None
    fs_hz = fs if fs else 0
    print(f"capture       : {len(segs)} segments (index 1..{len(segs)}), "
          f"{pts} pts/seg, fs={fs_hz/1e6:.1f} MHz")
    print(f"quality frames: {len(frames)}  (bim>={BIM_MIN}, |fc-{FC/1e6}M|<{FC_TOL/1e3}kHz)\n")

    print("=== COMMAND TIMELINE (seg index = time order) ===")
    print("  seg   cls  dhead8   head   tail   bim     fc")
    for idx, cls, bim, fc, ab in frames:
        print(f"  {idx:3d}  {cls}  {''.join(map(str,dh(ab)[:8]))}  "
              f"{hex8(ab[:8])}  {hex8(ab[8:])}  {bim:.3f}  {fc/1e6:.4f}")

    print("\n=== A/B RUN STRUCTURE PER BURST ===")
    for bi, b in enumerate(bursts(frames)):
        seq = ''.join(f[1] for f in b)
        print(f"  burst {bi}: segs {b[0][0]}..{b[-1][0]}  n={len(b)}  run={seq}")

    print("\n=== PER-CLASS CONSENSUS ===")
    for cls in 'AB':
        r = consensus(frames, cls)
        if not r:
            print(f"  class {cls}: no frames")
            continue
        abs_c, diff_c, agree, seglist = r
        print(f"  class {cls} (N={len(seglist)} repeats, segs={seglist}):")
        print(f"    abs  consensus : {' '.join(format(int(''.join(map(str,abs_c[g:g+4])),2),'04b') for g in (0,4,8,12))}  hex={hex8(abs_c[:8])}{hex8(abs_c[8:])}")
        print(f"    abs  agree%    : {' '.join(map(str, agree))}")
        print(f"    diff consensus : {''.join(map(str, diff_c))}")

    print("\n=== TAIL ANALYSIS (per-frame, segment order) ===")
    for cls in 'AB':
        grp = [f for f in frames if f[1] == cls]
        print(f"  class {cls}:")
        for idx, _, _, _, ab in grp:
            print(f"    seg {idx:3d}  tail={hex8(ab[8:])}  ({int(''.join(map(str,ab[8:])),2):3d})")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/scopes/2024-01-01_ButtonDownManySegments')
