#!/usr/bin/env python3
"""High-res zoom-FFT spectrum + per-burst coherent downconversion phase test.

Step 1: zoom-FFT (1.5M points) centered at the dominant tone -> resolve
        carrier + side lobes cleanly (the plain rfft was too coarse at 20kHz/bin).
Step 2: measure the precise carrier freq for this burst.
Step 3: downconvert at that precise carrier; if the phase is STABLE (bounded,
        little wander) -> BPSK phase keying. If phase wanders hugely -> the
        information is in the AMPLITUDE (ASK/OOK), not the phase.
"""
import sys
import numpy as np
from scipy.signal import hilbert
from fsk import burst_segments

def zoom_spec(t, v, fc, n=65536):
    v = v - v.mean()
    x = v*np.exp(-2j*np.pi*fc*t)
    X = np.fft.fft(x, n)
    dt = np.median(np.diff(t))
    f = np.fft.fftfreq(n, dt)
    P = np.abs(X)**2
    return f, P

def peak(P, f, lo, hi):
    m = (f>=lo)&(f<=hi)
    i = np.argmax(P[m])
    idx = np.where(m)[0][i]
    return f[idx], P[idx]

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]

    for tag, sel in (("IDLE", idle[100]), ("CMD", cmd[10])):
        _, t, v, fs = sel
        v = v - v.mean()
        f, P = zoom_spec(t, v, 1.0e6)   # baseband: carrier now at 0 Hz
        # find top 6 peaks in -150k..+150k around baseband
        Pw = P.copy()
        pk=[]
        for _ in range(6):
            ff, pp = peak(Pw, f, -150e3, 150e3)
            pk.append((ff, pp)); Pw[(f>ff-30e3)&(f<ff+30e3)]=0
        tot = P.sum()
        print(f"\n=== {tag} dur={(t[-1]-t[0])*1e6:.2f}us ===")
        for i,(ff,pp) in enumerate(sorted(pk,key=lambda x:-x[1])[:5]):
            print(f"  {ff/1e3:+9.2f} kHz(base)  {pp/tot*100:7.3f}%")
        # phase stability: the baseband signal x = v*exp(-j2pi*1e6*t) already exists
        x = v*np.exp(-2j*np.pi*1.0e6*t)
        ph = np.unwrap(np.angle(x))
        print(f"  baseband@1.000MHz : phase range={np.degrees(ph.max()-ph.min()):7.1f}deg  "
              f"std={ph.std()/np.pi*180:5.1f}deg  env_mean={np.abs(x).mean():.3f}")

if __name__ == "__main__":
    main()
