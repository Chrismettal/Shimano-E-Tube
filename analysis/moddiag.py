#!/usr/bin/env python3
"""Diagnose the modulation type (FSK vs BPSK) on command bursts.

For each 16-bit command burst we test two hypotheses and report the
per-symbol discriminability:

  FSK model:  symbol = whichever of f1/f2 carries more coherent energy
              contrast = |e1-e2|/(e1+e2)
  BPSK model: a single carrier fc; bit = sign of coherent correlation
              contrast = |Re(s)|/|s|  (how in-phase the symbol is)
              (a real BPSK symbol correlates to a real number, phase 0 or pi)

The model whose contrast stays high AND stable across all 16 symbols is
the real modulation.
"""
import sys
import numpy as np
from scipy import signal as sps
from fsk import burst_segments, measure_tones, optimize_tones, REPO

def sym_corr(t, v, fs, fc, rate, off):
    """Per-symbol complex correlation with a carrier fc. Returns complex array."""
    T = 1/rate
    dur = t[-1]-t[0]
    n = int(np.floor(dur/T + 1e-9))
    phi = np.exp(-2j*np.pi*fc*t)
    c = np.cumsum(v*phi)
    edges = t[0] + off + np.arange(n+1)*T
    idx = np.clip(np.searchsorted(t, edges), 0, len(v)-1)
    nz = idx[1:] > idx[:-1]
    return c[idx[1:]] - c[idx[:-1]]

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]
    print(f"idle={len(idle)} cmd={len(cmd)}")

    # pick a representative command burst (middle)
    _, t, v, fs = cmd[10]
    print(f"using cmd[10]  dur={(t[-1]-t[0])*1e6:.1f}us  fs={fs/1e6:.2f}MHz")
    li = int(np.argmax([len(s[2]) for s in segs]))
    _, tl, vl, fl = segs[li]
    f1, f2 = measure_tones(tl, vl, fl)
    f1, f2 = optimize_tones(idle[:10], f1, f2, 128e3, 5.859e-6, 0, 1)
    fc = 0.5*(f1+f2)
    print(f"f1={f1/1e6:.5f}  f2={f2/1e6:.5f}  fmid={fc/1e6:.5f}MHz")

    for rate in (120e3, 123.08e3, 125e3, 128e3):
        print(f"\n===== rate={rate/1e3:.2f} kbit/s =====")
        # FSK contrasts
        c1 = sym_corr(t, v, fs, f1, rate, 0.0)
        c2 = sym_corr(t, v, fs, f2, rate, 0.0)
        e1, e2 = np.abs(c1), np.abs(c2)
        fsk = np.abs(e1-e2)/(e1+e2)
        # BPSK at carrier candidates: f1, fmid, and f1 exactly
        print(" pos  e1      e2      FSKconf   BPSK@f1 |Re|/|s|  phase(deg)")
        for fc_try, label in [(f1,'f1'), (fc,'fmid')]:
            s = sym_corr(t, v, fs, fc_try, rate, 0.0)
            mag = np.abs(s); ph = np.angle(s)*180/np.pi
            bpsk = np.abs(np.real(s))/(mag+1e-12)
            line_b = " ".join(f"{bpsk[k]:6.2f}" for k in range(len(s)))
            line_ph = " ".join(f"{ph[k]:6.0f}" for k in range(len(s)))
            print(f"  {label}: BPSK|Re|/|s|: {line_b}")
            print(f"        phase(deg) : {line_ph}")
        line_f = " ".join(f"{fsk[k]:6.2f}" for k in range(len(s)))
        print(f"  FSK |e1-e2|/sum : {line_f}")
        # sign of e1>e2 as FSK bit; sign of Re(s) as BPSK bit
        fskbits = (e1>e2).astype(int)
        sb = sym_corr(t, v, fs, f1, rate, 0.0)
        bpskbits = (np.real(sb)>=0).astype(int)
        print(f"  FSK bits : {''.join(map(str,fskbits))}")
        print(f"  BPSK bits: {''.join(map(str,bpskbits))}")

if __name__ == "__main__":
    main()
