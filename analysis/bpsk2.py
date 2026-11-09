#!/usr/bin/env python3
"""Clean COHERENT BPSK demod.

Established facts:
  - Single carrier ~1.000 MHz (no second FSK tone).
  - ~16-symbol command frames, ~6-symbol idle frames.
  - Symbol period ~8.1 us (~123 kbaud).
  - Per-burst carrier fc must be locked (it drifts per capture); coherent
    per-symbol accumulation is robust to a small residual offset.

Method per burst:
  1. Zero-padded FFT -> fc (strongest bin in 0.95-1.05 MHz).
  2. Downconvert; accumulate per symbol window (coherent energy per symbol).
  3. Reference-free: diff each symbol's phase against the first symbol,
     so the arbitrary global phase cancels. bit = phase in (90,270) -> 1.
  4. Score = mean bimodality of the per-symbol coherent vector.
Then take majority-vote consensus across all command frames.
"""
import sys
import numpy as np
from fsk import burst_segments

def lock_fc(t, v, n=2**18):
    v = v - v.mean()
    dt = np.median(np.diff(t))
    X = np.fft.fft(v, n)
    f = np.fft.fftfreq(n, dt)
    band = (f>0.95e6)&(f<1.05e6)
    m = np.where(band, np.abs(X), 0)
    i = int(np.argmax(m))
    return f[i]

def coherent_symbols(t, v, fs, fc, n_sym, off=0.0):
    T = (t[-1]-t[0])/n_sym
    phi = np.exp(-2j*np.pi*fc*(t-t[0]))
    c = np.cumsum(v*phi)
    edges = off + np.arange(n_sym+1)*T
    idx = np.clip(np.searchsorted(t-t[0], edges), 0, len(v)-1)
    nz = idx[1:] > idx[:-1]
    vecs = (c[idx[1:]]-c[idx[:-1]])[nz]
    return vecs, T

def demod(t, v, fs, n_sym):
    fc = lock_fc(t, v)
    vecs, T = coherent_symbols(t, v, fs, fc, n_sym)
    ph = np.angle(vecs)
    # reference-free: phase relative to first symbol
    rel = np.angle(vecs*np.conj(vecs[0]))
    score = float(np.mean(np.abs(np.cos(2*rel))))
    bit = (np.abs(rel) > np.pi/2).astype(int)
    return fc, bit, score, rel, T

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]
    print(f"idle={len(idle)} cmd={len(cmd)}")

    print("\n=== IDLE (first 6) 6-symbol, ref-free ===")
    for i in range(6):
        _, t, v, fs = idle[i*60]
        fc, bit, sc, rel, T = demod(t, v, fs, 6)
        print(f"  fc={fc/1e6:.5f} score={sc:.3f} T={T*1e6:5.2f}us bits={''.join(map(str,bit))}")

    print("\n=== COMMAND 16-symbol, ref-free ===")
    rows = []
    for i, (_, t, v, fs) in enumerate(cmd):
        fc, bit, sc, rel, T = demod(t, v, fs, 16)
        rows.append(bit)
        print(f"[{i:2d}] fc={fc/1e6:.5f} score={sc:.3f} T={T*1e6:5.2f}us bits={''.join(map(str,bit))}")

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
