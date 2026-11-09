#!/usr/bin/env python3
"""Absolute 2D constellation: I/Q of each symbol across all command frames.
Single global fc. Collects every symbol vector, de-rotates by the dominant
cluster angle so the main point sits at +I, prints a scatter + cluster count.
Reveals BPSK (2 pts) vs QPSK (4 pts) vs OQPSK (4 pts, offset phases).
Also skips the low-magnitude lead-in symbols.
"""
import sys
import numpy as np
from fsk import burst_segments

def global_fc(segs_cmd):
    acc = np.zeros(2**19, float)
    for _, t, v, fs in segs_cmd:
        v = v - v.mean()
        acc += np.abs(np.fft.fft(v, 2**19))
    f = np.fft.fftfreq(2**19, np.median(np.diff(segs_cmd[0][1])))
    band = (f>0.95e6)&(f<1.05e6)
    return f[int(np.argmax(np.where(band, acc, 0)))]

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    cmd = [s for s, du in zip(segs, durs) if du >= 60]
    fc = global_fc(cmd)
    print(f"global fc={fc/1e6:.6f}")

    N = 16
    I = np.zeros(len(cmd)); Q = np.zeros(len(cmd))
    for fi, (_, t, v, fs) in enumerate(cmd):
        T = t - t[0]; v0 = v - v.mean()
        c = np.cumsum(v0*np.exp(-2j*np.pi*fc*T))
        edges = np.arange(N+1)*(T[-1]/N)
        idx = np.clip(np.searchsorted(T, edges), 0, len(v0)-1)
        vecs = (c[idx[1:]]-c[idx[:-1]])
        # normalize each symbol by max mag in frame
        m = np.abs(vecs); m = m.max() if m.max()>0 else 1
        v = vecs/m
        I[fi] = v[2].real; Q[fi] = v[2].imag   # symbol index 2 (skip lead-in 0,1)

    ang = np.angle(I + 1j*Q)
    # dominant angle
    hist, _ = np.histogram(ang, bins=72, range=(-np.pi, np.pi))
    peak = _ = np.degrees(np.linspace(-np.pi, np.pi, 72, endpoint=False)[np.argmax(hist)]) + 2.5
    rot = np.exp(-1j*np.deg2rad(peak))
    print(f"de-rotated by {peak:.1f} deg (dominant cluster at sym2)")

    print("\nSymbol-2 I/Q across all frames (de-rotated), '.'=near +I:")
    for fi in range(len(cmd)):
        ii = I[fi]*np.cos(np.deg2rad(peak)) - Q[fi]*np.sin(np.deg2rad(peak))
        qq = I[fi]*np.sin(np.deg2rad(peak)) + Q[fi]*np.cos(np.deg2rad(peak))
        print(f"  [{fi:2d}] I={ii:+.3f} Q={qq:+.3f} ang={np.degrees(np.angle(ii+1j*qq)):6.1f}")

    # full scatter of ALL 16 symbols per frame (de-rotated by global angle of sym2 majority)
    # build cluster histogram over all symbol angles
    allang = []
    for fi, (_, t, v, fs) in enumerate(cmd):
        T = t - t[0]; v0 = v - v.mean()
        c = np.cumsum(v0*np.exp(-2j*np.pi*fc*T))
        edges = np.arange(N+1)*(T[-1]/N)
        idx = np.clip(np.searchsorted(T, edges), 0, len(v0)-1)
        vecs = (c[idx[1:]]-c[idx[:-1]])
        m = np.abs(vecs).max()
        vv = vecs/m
        allang.append(np.angle(vv) - np.deg2rad(peak))
    allang = np.concatenate(allang)
    hist, bins = np.histogram(allang, bins=36, range=(-np.pi, np.pi))
    print("\nangle histogram (deg, all symbols, de-rotated):")
    for i,h in enumerate(hist):
        c = '#'*int(h/ max(hist.max()/20,1))
        print(f"  {np.degrees(bins[i]):8.1f} to {np.degrees(bins[i+1]):8.1f}: {h:4d} {c}")

if __name__ == "__main__":
    main()
