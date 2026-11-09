#!/usr/bin/env python3
"""FINAL coherent BPSK demod with SCORE-DRIVEN carrier recovery.

The bimodality score mean|cos(2*phase_rel)| is 1.0 for a perfectly locked
carrier and collapses when fc is off (phase drifts across the frame). So we
use it as the lock metric: coarse FFT peak -> fine grid search around it
that maximizes the score. Every burst then demodulates cleanly.
"""
import sys
import numpy as np
from fsk import burst_segments

def lock_coarse(t, v, n=2**19):
    v = v - v.mean()
    dt = np.median(np.diff(t))
    X = np.fft.fft(v, n)
    f = np.fft.fftfreq(n, dt)
    band = (f>0.95e6)&(f<1.05e6)
    m = np.where(band, np.abs(X), 0)
    i = int(np.argmax(m))
    return f[i]

def score_of(vecs):
    if len(vecs) < 2: return 0.0
    rel = np.angle(vecs*np.conj(vecs[0]))
    return float(np.mean(np.abs(np.cos(2*rel))))

def demod(t, v, fs, n_sym):
    T = (t-t[0])
    v0 = v - v.mean()
    # stage 1: fine grid around coarse peak, maximize score
    fc0 = lock_coarse(t, v)
    best = None
    for fc in np.arange(fc0-6e3, fc0+6e3, 100.0):
        phi = np.exp(-2j*np.pi*fc*T)
        c = np.cumsum(v0*phi)
        edges = np.arange(n_sym+1)*(T[-1]/n_sym)
        idx = np.clip(np.searchsorted(T, edges), 0, len(v0)-1)
        vecs = (c[idx[1:]]-c[idx[:-1]])
        s = score_of(vecs)
        if best is None or s > best[0]: best = (s, fc)
    s, fc = best
    # stage 2: refine +/-500Hz at 50Hz
    for fc in np.arange(fc-5e2, fc+5e2, 50.0):
        phi = np.exp(-2j*np.pi*fc*T)
        c = np.cumsum(v0*phi)
        edges = np.arange(n_sym+1)*(T[-1]/n_sym)
        idx = np.clip(np.searchsorted(T, edges), 0, len(v0)-1)
        vecs = (c[idx[1:]]-c[idx[:-1]])
        s2 = score_of(vecs)
        if s2 > s: s, fc = s2, fc
    phi = np.exp(-2j*np.pi*fc*T)
    c = np.cumsum(v0*phi)
    edges = np.arange(n_sym+1)*(T[-1]/n_sym)
    idx = np.clip(np.searchsorted(T, edges), 0, len(v0)-1)
    vecs = (c[idx[1:]]-c[idx[:-1]])
    rel = np.angle(vecs*np.conj(vecs[0]))
    bit = (np.abs(rel) > np.pi/2).astype(int)
    return fc, bit, s

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]
    print(f"idle={len(idle)} cmd={len(cmd)}")

    print("\n=== IDLE (sample of 12) 6-symbol ===")
    for i in range(12):
        _, t, v, fs = idle[i*40]
        fc, bit, s = demod(t, v, fs, 6)
        print(f"  fc={fc/1e6:.5f} score={s:.3f} bits={''.join(map(str,bit))}")

    print("\n=== COMMAND 16-symbol (score-driven) ===")
    rows = []
    for i, (_, t, v, fs) in enumerate(cmd):
        fc, bit, s = demod(t, v, fs, 16)
        rows.append(bit)
        print(f"[{i:2d}] fc={fc/1e6:.5f} score={s:.3f} bits={''.join(map(str,bit))}")

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
