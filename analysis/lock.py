#!/usr/bin/env python3
"""Precise per-burst carrier lock + baseband phase inspection.

1. Zero-padded FFT of the raw burst; find the single strongest bin => fc.
2. Downconvert at exactly fc -> baseband complex signal.
3. If baseband phase is STABLE (two clusters at 0 and 180, low std) => BPSK, carrier locked.
   Print the per-symbol baseband phase. If it wanders continuously => carrier still off / it's FM.
"""
import sys
import numpy as np
from fsk import burst_segments

def lock_carrier(t, v, n=2**18):
    v = v - v.mean()
    dt = np.median(np.diff(t))
    X = np.fft.fft(v, n)
    f = np.fft.fftfreq(n, dt)
    m = np.abs(X)
    # only look in 0.9-1.1 MHz (positive freqs)
    band = (f>0.9e6)&(f<1.1e6)
    i = int(np.argmax(np.where(band, m, 0)))
    return f[i], m[i]/m.max()

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]

    for tag, sel in (("IDLE", idle[100]), ("CMD", cmd[10]), ("CMD2", cmd[5])):
        _, t, v, fs = sel
        v = v - v.mean()
        fc, rel = lock_carrier(t, v)
        x = v*np.exp(-2j*np.pi*fc*t)   # baseband
        ph = np.unwrap(np.angle(x))
        env = np.abs(x)
        print(f"\n=== {tag} fc={fc/1e6:.6f}MHz (rel {rel:.3f}) dur={(t[-1]-t[0])*1e6:.2f}us ===")
        print(f"  baseband: phase range={np.degrees(ph.max()-ph.min()):7.1f}  std={ph.std()*180/np.pi:5.1f}  env={env.mean():.3f}")
        # per ~6.25us symbol (16 sym/100us) dump
        T = (t[-1]-t[0])/16.0
        print("  sym   t_us   env   phase_deg  | marker")
        for k in range(16):
            i0 = int(k*T*fs); i1 = int((k+1)*T*fs)
            if i1>=len(t): i1=len(t)-1
            phk = np.angle(np.sum(x[i0:i1]))   # coherent-ish mean over window
            evk = env[i0:i1].mean()
            mark = '#' if evk>0.5 else '.'
            print(f"  {k:3d}  {t[i0]*1e6:7.2f}  {evk:.3f}   {np.degrees(phk):8.1f}   {mark}")

if __name__ == "__main__":
    main()
