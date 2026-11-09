#!/usr/bin/env python3
"""Tone-independent calibration via idle-burst averaging + autocorrelation.

The idle bursts all carry the same repeating 000011 pattern. Averaging the
hundreds of them into a single high-SNR template lets us:
  1. measure the two FSK tones from a clean signal (IF histogram)
  2. find the symbol period from the template's autocorrelation / run-lengths
  3. establish the exact bit timing phase
This cross-checks the fsk.py calibration without assuming the tones up front.
"""
import sys, os
import numpy as np
from scipy import signal as sps
from fsk import burst_segments, measure_tones, REPO

def resample_to_grid(seg, fs):
    """Return the segment samples on a common grid starting at t0, step 1/fs."""
    _, t, v, _ = seg
    dur = t[-1] - t[0]
    n = int(np.floor(dur * fs))
    grid = np.arange(n) / fs
    return np.interp(grid, t, v)

def align_average(segs, fs, max_iter=3):
    """Cross-correlate each idle burst to a reference and average them aligned.
    Returns (template, lags)."""
    ref = resample_to_grid(segs[0], fs)
    ref = ref - ref.mean()
    center = len(ref)//2
    lags = np.zeros(len(segs), dtype=int)
    mean = None
    for _ in range(max_iter):
        acc = np.zeros(len(ref))
        for i, s in enumerate(segs):
            v = resample_to_grid(s, fs)
            v = v[:len(ref)] if len(v) >= len(ref) else np.pad(v, (0, len(ref)-len(v)))
            cc = np.correlate(ref, v - v.mean(), mode='full')
            p = int(np.argmax(cc))
            lag = p - center
            lag = int(np.clip(lag, -12, 12))
            lags[i] = lag
            acc += np.roll(v, lag)
        mean = acc / len(segs)
        ref = mean - mean.mean()
    return mean, lags

def run_lengths(bits):
    """Return (run lengths, bit values) of a binary sequence."""
    runs, vals = [], []
    cur = bits[0]; c = 1
    for b in bits[1:]:
        if b == cur: c += 1
        else:
            runs.append(c); vals.append(cur); cur = b; c = 1
    runs.append(c); vals.append(cur)
    return np.array(runs), np.array(vals)

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    print(f"segments: {len(segs)}  idle: {len(idle)}")
    fs = idle[0][3]
    print(f"fs={fs/1e6:.3f}MHz")

    # ---- average the idle bursts into a clean template ----
    tpl, lags = align_average(idle, fs)
    print(f"lag spread (samples): min={lags.min()} max={lags.max()}  "
          f"(jitter ~ +- {lags.max()/fs*1e9:.1f}ns)")
    t = np.arange(len(tpl))/fs

    # ---- tones from the clean template (high SNR) ----
    f1, f2 = measure_tones(t, tpl - tpl.mean(), fs)
    print(f"template tones: f1={f1/1e6:.6f}MHz  f2={f2/1e6:.6f}MHz  "
          f"diff={(f2-f1)/1e3:.2f}kHz  mid={(f1+f2)/2/1e6:.6f}MHz")

    # ---- symbol period from the template IF ----
    z = sps.hilbert(tpl - tpl.mean())
    dph = np.unwrap(np.angle(z))
    ifo = sps.savgol_filter(np.diff(dph)*fs/(2*np.pi), 5, 3)
    fmid = 0.5*(f1+f2)
    bits = (ifo > fmid).astype(int)  # 1 = f2, 0 = f1
    runs, vals = run_lengths(bits)
    print(f"IF bit runs (len): {runs[:24]}")
    print(f"IF bit vals: {vals[:24]}")
    # a run of n IF samples at fs => duration n/fs; that run = 1 symbol of one tone
    run_durs_us = runs*1e6/fs
    print(f"run durations (us): {[f'{x:.3f}' for x in run_durs_us[:24]]}")
    # the median single-symbol run = symbol period
    med = np.median(run_durs_us)
    print(f"median run = {med:.4f}us  => symbol rate ~ {1e6/med/1e3:.2f} kbit/s")
    # if pattern is 000011, the '1' runs are the shorter ones; estimate bit from the
    # smallest common run
    small = run_durs_us[run_durs_us < 2*med]
    if len(small):
        print(f"shortest-run cluster: median={np.median(small):.4f}us "
              f"=> {1e6/np.median(small)/1e3:.2f} kbit/s")

    # ---- autocorrelation of the template for the fundamental period ----
    x = tpl - tpl.mean()
    ac = np.correlate(x, x, mode='full')[len(x)-1:]
    ac = ac/ac[0]
    # find first strong local peak after the first dip
    from scipy.signal import find_peaks
    pk, _ = find_peaks(ac[10:], height=0.2)
    if len(pk):
        first = pk[0] + 10
        print(f"first autocorr peak at {first} samples = {first*1e6/fs:.4f}us")
        # the 000011 pattern repeats every 6 symbols; the peak could be 1 or 6 symbols
        for k in (1, 2, 3, 4, 6):
            print(f"  if first peak = {k} symbol(s): rate = {1e6/(first*1e6/fs)/1e3/k:.2f} kbit/s")

    # save the template for plotting / further use
    np.savez(os.path.join(os.path.dirname(__file__), 'idle_template.npz'),
             t=t, tpl=tpl, f1=f1, f2=f2)
    print("saved analysis/idle_template.npz")

if __name__ == "__main__":
    main()
