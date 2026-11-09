#!/usr/bin/env python3
"""Definitive modulation test: high-res FFT power spectrum of idle vs command bursts.

- FSK (2 tone)   -> two well-separated peaks
- BPSK/PSK (1 tone) -> one main carrier lobe, possibly with phase-mod side lobes
We print the top spectral peaks with their separation and relative power.
"""
import sys
import numpy as np
from fsk import burst_segments

def top_peaks(P, f, n=8):
    P = P.copy()
    pk = []
    for _ in range(n):
        i = int(np.argmax(P))
        if P[i] <= 0: break
        pk.append((f[i], P[i]))
        # zero out a +-40kHz neighborhood to find the next peak
        P[np.abs(f-f[i])<40e3] = 0
    pk.sort(key=lambda x:-x[1])
    return pk

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]

    for tag, sel in (("IDLE", idle[100]), ("CMD", cmd[10])):
        _, t, v, fs = sel
        # window to reduce leakage
        w = np.hanning(len(v))
        V = np.fft.rfft((v-np.mean(v))*w)
        P = np.abs(V)**2
        f = np.fft.rfftfreq(len(v), d=1.0/fs)
        pk = top_peaks(P, f)
        tot = P.sum()
        print(f"\n=== {tag}  dur={(t[-1]-t[0])*1e6:.2f}us  fs={fs/1e6:.3f}MHz ===")
        main = pk[0][1]/tot
        for i,(ff,pp) in enumerate(pk[:6]):
            print(f"  #{i} {ff/1e6:.5f} MHz  rel={pp/tot*100:6.2f}%")
        if len(pk)>=2:
            print(f"  -> top2 separation: {abs(pk[0][0]-pk[1][0])/1e3:.1f} kHz")

if __name__ == "__main__":
    main()
