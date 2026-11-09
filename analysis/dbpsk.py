#!/usr/bin/env python3
"""Differential BPSK demod, refined: per-frame residual carrier search.

Idle frames (6 symbols) decode identically -> the differential method is
correct. Command frames (16 symbols) are corrupted by a per-capture
residual carrier offset that accumulates ~150 deg over the frame.
Fix: for each candidate fc, remove the best linear phase trend (the
residual offset) BEFORE taking phase steps, then score bimodality.

Idle frames must still come out identical afterwards (sanity check).
"""
import sys
import numpy as np
from fsk import burst_segments

def sym_corr(t, v, fs, fc, rate):
    T = 1/rate
    dur = t[-1]-t[0]
    n = int(np.floor(dur/T + 1e-9))
    phi = np.exp(-2j*np.pi*fc*(t - t[0]))
    c = np.cumsum(v*phi)
    edges = np.arange(n+1)*T
    idx = np.clip(np.searchsorted(t - t[0], edges), 0, len(v)-1)
    nz = idx[1:] > idx[:-1]
    return (c[idx[1:]] - c[idx[:-1]])[nz]

def demod_frame(t, v, fs, rate, fc0, band=80e3, nb=161):
    best = None
    for fc in fc0 + np.linspace(-band, band, nb):
        c = sym_corr(t, v, fs, fc, rate)
        if len(c) < 4: continue
        ph = np.unwrap(np.angle(c))
        # remove linear trend = residual carrier offset + initial phase
        x = np.arange(len(ph))
        coef, *_ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, ph, rcond=None)
        resid = ph - (coef[0]*x + coef[1])
        # rotate out the remaining constant phase (reference-free)
        m = np.mean(np.exp(1j*resid))
        if abs(m) < 0.05: continue
        rel = np.angle(resid*0 + (np.exp(1j*resid)/m))
        score = float(np.mean(np.abs(np.cos(2*rel))))
        if best is None or score > best[0]:
            best = (score, fc, rel, coef)
    if best is None:
        return None
    score, fc, rel, coef = best
    bit = (np.abs(rel) > np.pi/2).astype(int)
    return fc, bit, score, coef[0]*rate/ (2*np.pi)  # residual Hz

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]
    print(f"idle={len(idle)} cmd={len(cmd)}")
    rate = 125e3
    fc0 = 0.9999e6

    print("\n=== IDLE (first 8) - must be identical ===")
    for i in range(8):
        _, t, v, fs = idle[i*60]
        r = demod_frame(t, v, fs, rate, fc0)
        fc, bit, sc, hz = r
        print(f"  fc={fc/1e6:.5f} score={sc:.3f} res={hz:8.0f}Hz bits={''.join(map(str,bit))}")

    print("\n=== COMMAND ===")
    rows = []
    for i, (_, t, v, fs) in enumerate(cmd):
        fc, bit, sc, hz = demod_frame(t, v, fs, rate, fc0)
        rows.append(bit)
        print(f"[{i:2d}] fc={fc/1e6:.5f} score={sc:.3f} res={hz:8.0f}Hz bits={''.join(map(str,bit))}")

    L = max(len(r) for r in rows)
    R = np.zeros((len(rows), L), int)
    for i in range(len(rows)): R[i,:len(rows[i])] = rows[i]
    maj = (R.mean(axis=0) > 0.5).astype(int)
    print("\nconsensus :", ''.join(map(str,maj)))
    print("std       :", " ".join(f"{R[:,p].std():.2f}" for p in range(L)))
    for lab, arr in (("maj", maj), ("~maj", 1-maj)):
        hx = ' '.join(f'{int("".join(map(str,arr[p:p+8])),2):02x}' for p in range(0,L-L%8,8))
        print(f"hex({lab})   :", hx)

if __name__ == "__main__":
    main()
