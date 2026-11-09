#!/usr/bin/env python3
"""Inspect the true tone structure of a SINGLE command burst.

1. Measure tones with a fine-resolution IF histogram over the whole burst
   -> how many tones, where exactly?
2. Demodulate at the exact frame rate 6 sym / 49.8us = 120.5 kbit/s with
   the measured tones, and dump per-symbol (freq, energy1, energy2).
3. Same for one idle burst for comparison.
This tells us the ground truth before any consensus averaging.
"""
import sys
import numpy as np
from scipy.signal import hilbert, savgol_filter
from fsk import burst_segments

def if_hist(t, v, fs, nb=480, wlen=51, wdeg=2):
    z = hilbert(v)
    phi = np.unwrap(np.angle(z))
    instf = np.diff(phi)/(2*np.pi*np.diff(t))
    # smooth
    instf = savgol_filter(instf, wlen, wdeg)
    edges = np.linspace(0.9e6, 1.2e6, nb+1)
    hist, _ = np.histogram(instf, bins=edges)
    fc = (edges[:-1]+edges[1:])/2
    return fc, hist

def tone_peaks(fc, hist, thresh=0.05):
    h = hist/(hist.max()+1e-12)
    peaks = []
    for i in range(1, len(h)-1):
        if h[i] > h[i-1] and h[i] >= h[i+1] and h[i] > thresh:
            peaks.append((fc[i], h[i]))
    # merge close peaks
    out = []
    for f, a in peaks:
        if out and abs(f-out[-1][0]) < 25e3:
            if a > out[-1][1]: out[-1] = (f, a)
        else: out.append((f, a))
    return out

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]

    for tag, sel in (("IDLE", idle[100]), ("CMD", cmd[10])):
        _, t, v, fs = sel
        fc, hist = if_hist(t, v, fs)
        pk = tone_peaks(fc, hist)
        print(f"\n=== {tag}  dur={(t[-1]-t[0])*1e6:.2f}us ===")
        print("tone peaks:", "  ".join(f"{f/1e6:.5f}MHz(a={a:.2f})" for f, a in pk))

if __name__ == "__main__":
    main()
