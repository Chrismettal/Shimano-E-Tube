#!/usr/bin/env python3
"""Dump the true time-domain structure of single bursts.

For one idle and one command burst:
  - envelope (amplitude) vs time
  - unwrapped phase vs time (raw, with residual carrier)
  - remove best-fit linear phase (carrier) -> residual phase trajectory
  - instantaneous frequency over time
Print a compact per-microsecond table so we can SEE the symbol structure.
"""
import sys
import numpy as np
from scipy.signal import hilbert, savgol_filter
from fsk import burst_segments

def analyze(tag, t, v, fs):
    v = v - v.mean()
    z = hilbert(v)
    env = np.abs(z)
    phase = np.angle(z)
    up = np.unwrap(phase)
    # best-fit linear carrier over the whole burst
    coef = np.polyfit(t, up, 1)
    resid = up - np.polyval(coef, t)
    ifo = np.diff(up)/(2*np.pi*np.diff(t))
    print(f"\n=== {tag}  dur={(t[-1]-t[0])*1e6:.2f}us  N={len(t)} ===")
    print(f"carrier fc = {coef[0]/(2*np.pi)/1e6:.6f} MHz")
    print(f"resid phase range = {np.degrees(resid.max()-resid.min()):7.1f} deg")
    # per-microsecond snapshot
    step = max(1, int(len(t)/( (t[-1]-t[0])*1e6 * 2 )))
    print("  us    env    resid(deg)   ifo(MHz)")
    for i in range(0, len(t), step):
        print(f"{(t[i]-t[0])*1e6:5.2f} {env[i]/env.max():5.2f} {np.degrees(resid[i]):9.1f} {ifo[i]/1e6:8.4f}")

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]
    _, it, iv, ifs = idle[100]
    _, ct, cv, cfs = cmd[10]
    analyze("IDLE ", it, iv, ifs)
    analyze("CMD  ", ct, cv, cfs)

if __name__ == "__main__":
    main()
