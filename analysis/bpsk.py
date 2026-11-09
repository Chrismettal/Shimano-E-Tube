#!/usr/bin/env python3
"""BPSK demodulator + spectrum check for the E-Tube command bursts.

Step 1: spectrum of one idle + one command burst -> is it one tone or two?
Step 2: BPSK demod with fine carrier search. For a candidate carrier fc
        and rate, compute per-symbol complex correlation c_k. The data are
        in arg(c_k) which should cluster near {0,pi} after removing a
        residual linear phase drift (carrier offset). Score each fc by how
        bimodal the phases are; the best fc gives the bits.
"""
import sys
import numpy as np
from scipy import signal as sps
from fsk import burst_segments, measure_tones, optimize_tones

def sym_corr(t, v, fs, fc, rate, off):
    T = 1/rate
    dur = t[-1]-t[0]
    n = int(np.floor(dur/T + 1e-9))
    phi = np.exp(-2j*np.pi*fc*(t - t[0]))
    c = np.cumsum(v*phi)
    edges = off + np.arange(n+1)*T
    idx = np.clip(np.searchsorted(t - t[0], edges), 0, len(v)-1)
    nz = idx[1:] > idx[:-1]
    sel = c[idx[1:]] - c[idx[:-1]]
    return sel[nz]

def spectrum_peak(t, v, fs):
    spec = np.abs(np.fft.rfft(v*np.hanning(len(v))))
    f = np.fft.rfftfreq(len(v), 1/fs)
    # ignore the very low freqs (DC/offset) and find top peak above 0.5MHz
    m = f > 0.5e6
    k = int(np.argmax(spec[m]))
    fi = np.where(m)[0]
    pk = fi[k]
    return f[pk], spec[pk]

def bpsk_demod(t, v, fs, rate, fc0, off=0.0, band=40e3, nb=61):
    """Search carrier around fc0; return (best_fc, bits, score, phases)."""
    fcs = fc0 + np.linspace(-band, band, nb)
    best = None
    for fc in fcs:
        c = sym_corr(t, v, fs, fc, rate, off)
        if len(c) < 8: continue
        ph = np.angle(c)
        # remove best linear trend (residual offset) then check bimodality
        x = np.arange(len(ph))
        # unwrap then linear fit
        pu = np.unwrap(ph)
        A = np.vstack([x, np.ones(len(x))]).T
        coef, *_ = np.linalg.lstsq(A, pu, rcond=None)
        resid = pu - (coef[0]*x + coef[1])
        # bimodality: distance to nearest of {0, pi}; small is good
        d0 = np.abs(resid); d1 = np.abs(resid-np.pi); d2 = np.abs(resid+np.pi)
        dmin = np.minimum(d0, np.minimum(d1, d2))
        score = float(np.mean(dmin))   # lower = more bimodal
        if best is None or score < best[0]:
            best = (score, fc, c, resid, coef)
    score, fc, c, resid, coef = best
    bits = (np.abs(np.sin(resid)) < np.abs(np.cos(resid)))  # near 0/pi -> 0 ; else 1
    # bit = 1 if closer to pi, 0 if closer to 0
    d0 = np.abs(resid); d1 = np.abs(resid-np.pi)
    bits = (d1 < d0).astype(int)
    slope_hz = coef[0]*rate  # residual carrier offset estimate (rad/sym -> Hz)
    return fc, bits, score, slope_hz

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]
    print(f"idle={len(idle)} cmd={len(cmd)}")

    # ---- spectrum of representative idle + command ----
    _, ti, vi, fi = idle[100]
    _, tc, vc, fc_ = cmd[10]
    pfi, a = spectrum_peak(ti, vi, fi)
    pfc, b = spectrum_peak(tc, vc, fc_)
    print(f"idle  spectrum peak: {pfi/1e6:.5f}MHz (mag={a:.0f})")
    print(f"cmd   spectrum peak: {pfc/1e6:.5f}MHz (mag={b:.0f})")
    # full top-3 peaks for the command
    spec = np.abs(np.fft.rfft(vc*np.hanning(len(vc))))
    fr = np.fft.rfftfreq(len(vc), 1/fc_)
    order = np.argsort(spec)[::-1][:5]
    print("cmd top-5 freqs:", " ".join(f"{fr[k]/1e6:.5f}MHz" for k in order))

    fc0 = pfc
    rate = 125e3
    print(f"\nBPSK demod @ rate={rate/1e3}kbit/s, fc0={fc0/1e6:.5f}MHz, band=+-40kHz")
    print(" idx  fc(MHz)  bits            bimodal_score  res_offset(Hz)")
    frames = []
    for i, (_, t, v, fs) in enumerate(cmd):
        fc, bits, score, slo = bpsk_demod(t, v, fs, rate, fc0)
        frames.append(bits)
        if i < 21:
            print(f"{i:3d}  {fc/1e6:.5f}  {''.join(map(str,bits))}  {score:.3f}         {slo:7.0f}")

    if frames:
        L = max(len(f) for f in frames)
        B = np.zeros((len(frames), L), int)
        for i in range(len(frames)): B[i,:len(frames[i])] = frames[i]
        maj = (B.mean(axis=0) > 0.5).astype(int)
        print("\nconsensus:", ''.join(map(str,maj)))
        print("variance :", " ".join(f"{B[:,p].std():.2f}" for p in range(L)))
        hx = ' '.join(f'{int("".join(map(str,maj[p:p+8])),2):02x}' for p in range(0,L-L%8,8))
        print("hex      :", hx)

if __name__ == "__main__":
    main()
