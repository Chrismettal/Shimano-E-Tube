#!/usr/bin/env python3
"""Diagnostic: FIXED global carrier, per-symbol structure dump.

Goal: see exactly which of the 16 symbols are constant across command
frames and which vary, and whether symbol 0 is a lead-in/preamble.
Uses one global fc estimated from the combined command signal.
"""
import sys
import numpy as np
from fsk import burst_segments

def global_fc(segs_cmd):
    # average the zero-padded FFT magnitude over all command bursts
    acc = np.zeros(2**19, float)
    cnt = 0
    for _, t, v, fs in segs_cmd:
        v = v - v.mean()
        acc += np.abs(np.fft.fft(v, 2**19))
        cnt += 1
    f = np.fft.fftfreq(2**19, np.median(np.diff(segs_cmd[0][1])))
    band = (f>0.95e6)&(f<1.05e6)
    m = np.where(band, acc, 0)
    return f[int(np.argmax(m))]

def sym_vectors(t, v, fs, fc, n_sym):
    T = t - t[0]
    v0 = v - v.mean()
    phi = np.exp(-2j*np.pi*fc*T)
    c = np.cumsum(v0*phi)
    edges = np.arange(n_sym+1)*(T[-1]/n_sym)
    idx = np.clip(np.searchsorted(T, edges), 0, len(v0)-1)
    vecs = (c[idx[1:]]-c[idx[:-1]])
    return vecs

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]
    fc = global_fc(cmd)
    print(f"global fc = {fc/1e6:.6f} MHz   (cmd={len(cmd)} idle={len(idle)})")

    # per-symbol, across all command frames: print phase (ref-free vs sym0) + mag
    print("\nsym  " + "  ".join(f"{i:2d}" for i in range(16)))
    allv = []
    for fi, (_, t, v, fs) in enumerate(cmd):
        vecs = sym_vectors(t, v, fs, fc, 16)
        rel = np.angle(vecs*np.conj(vecs[0]))
        mag = np.abs(vecs)
        allv.append((rel, mag))

    for row, (label, fn) in enumerate([("phase_ref0(deg)", lambda i: np.round(np.degrees(allv[i][0])),),
                                       ("mag", lambda i: allv[i][1])]):
        pass
    # phase table
    ph = "phs  " + "  ".join(f"{int(np.degrees(allv[i][0][k])):4d}" for i in range(len(cmd)) for k in [0])
    for k in range(16):
        line = f"s{k:2d} "
        for i in range(len(cmd)):
            deg = int(np.degrees(allv[i][0][k]))
            mark = '1' if abs(((deg+180)%360)-180)>90 else '0'  # ref-free bit
            line += f" {deg:4d}"
        print(line)
    # magnitude table (to spot lead-in/low-sym)
    print("\n  mag (per sym, per frame; '.'=low '*'=hi):")
    mags = np.array([allv[i][1] for i in range(len(cmd))])
    q = np.quantile(mags, 0.5)
    for k in range(16):
        line = f"s{k:2d} "
        for i in range(len(cmd)):
            line += ('*' if mags[i,k]>q else '.')
        print(line)
    # per-symbol majority bit + agreement
    bits = np.array([ (np.abs(np.degrees(allv[i][0]))>90).astype(int) for i in range(len(cmd)) ])
    maj = (bits.mean(axis=0)>0.5).astype(int)
    agr = bits.mean(axis=0)
    print("\nmaj  ", ''.join(map(str,maj)))
    print("agr  ", " ".join(f"{a:.2f}" for a in agr))

if __name__ == "__main__":
    main()
