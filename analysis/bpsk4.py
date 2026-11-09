#!/usr/bin/env python3
"""BPSK demod FINAL: per-burst score-driven carrier lock.

Key insight: the carrier drifts by up to ~40 kHz between bursts. A wide
score-driven search (bimodality of per-symbol coherent vectors, referenced
to the highest-mag symbol) finds the true carrier for every burst. With
100 Hz refinement, residual drift < 0.5 deg/frame -> no trend removal
(removing it would distort the 0/180 staircase).
"""
import sys
import numpy as np
from fsk import burst_segments

def symvecs(t, v, fc, n_sym):
    T = t - t[0]; v0 = v - v.mean()
    c = np.cumsum(v0*np.exp(-2j*np.pi*fc*T))
    edges = np.arange(n_sym+1)*(T[-1]/n_sym)
    idx = np.clip(np.searchsorted(T, edges), 0, len(v0)-1)
    return (c[idx[1:]]-c[idx[:-1]])

def score_of(vecs):
    m = np.abs(vecs)
    if m.max()==0: return 0.0
    k = int(np.argmax(m))
    rel = np.angle(vecs*np.conj(vecs[k]))
    return float(np.mean(np.abs(np.cos(2*rel))))

def search(t, v, n_sym, lo, hi, step):
    best = (-1.0, None)
    for fc in np.arange(lo, hi, step):
        s = score_of(symvecs(t, v, fc, n_sym))
        if s > best[0]: best = (s, fc)
    return best

def demod(t, v, n_sym):
    s1, fc = search(t, v, n_sym, 0.955e6, 1.045e6, 500.0)
    s2, fc = search(t, v, n_sym, fc-1e3, fc+1e3, 50.0)
    vecs = symvecs(t, v, fc, n_sym)
    k = int(np.argmax(np.abs(vecs)))
    rel = np.angle(vecs*np.conj(vecs[k]))
    bit = (np.abs(rel) > np.pi/2).astype(int)
    # per-symbol magnitude (lead-in detection)
    mag = np.abs(vecs)
    return fc, bit, s2, k, mag

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]
    print(f"idle={len(idle)} cmd={len(cmd)}")

    print("\n=== IDLE (6-symbol, every 40th) ===")
    for i in range(0, len(idle), 40):
        _, t, v, fs = idle[i]
        fc, bit, s, k, mag = demod(t, v, 6)
        print(f"  fc={fc/1e6:.5f} score={s:.3f} ref={k} mag={np.round(mag/ (mag.max() or 1),2)} bits={''.join(map(str,bit))}")

    print("\n=== COMMAND (16-symbol) ===")
    rows = []; mags = []
    for i, (_, t, v, fs) in enumerate(cmd):
        fc, bit, s, k, mag = demod(t, v, 16)
        rows.append(bit); mags.append(mag)
        print(f"[{i:2d}] fc={fc/1e6:.5f} score={s:.3f} ref={k:2d} bits={''.join(map(str,bit))}")
        print(f"       mag={[''.join('#' if x>0.5*mag.max() else ('.' if x>0.15*mag.max() else ' ') for x in mag)]}")

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
