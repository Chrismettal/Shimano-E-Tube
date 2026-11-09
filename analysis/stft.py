#!/usr/bin/env python3
"""STFT spectrogram as ASCII + envelope cross-correlation of command bursts.

Spectrogram: downconvert near 1.0 MHz, 2us Hann window, 1us hop.
Print one row per hop: '*' marks where local power is in the top tiers.
Frequencies span 0.95-1.05 MHz in ~4kHz steps. If the signal is 2-GFSK we
will SEE the frequency jump between rows; if BPSK one steady row stays hot.
"""
import sys
import numpy as np
from fsk import burst_segments

def spectro(t, v, fs):
    v = v - v.mean()
    win_n = int(2e-6*fs)
    hop = int(1e-6*fs)
    w = np.hanning(win_n)
    f_lo, f_hi = 0.95e6, 1.05e6
    rows = []
    for s in range(0, len(v)-win_n, hop):
        seg = v[s:s+win_n]*w
        X = np.abs(np.fft.rfft(seg, 2048))
        fr = np.fft.rfftfreq(2048, 1.0/fs)
        m = (fr>=f_lo)&(fr<=f_hi)
        rows.append((X[m], fr[m]))
    Xs = np.vstack([r[0] for r in rows])
    fr = rows[0][1]
    thr = np.quantile(Xs, [0.50, 0.90, 0.99])
    chars = [' ', '.', ':', '*', '#']
    out = []
    for ri, (x, _) in enumerate(rows):
        line = []
        for b in range(len(fr)):
            vv = float(x[b])
            c = 0
            for i in range(len(thr)):
                if vv >= thr[i]: c = i+1
            line.append(chars[c])
        tm = (ri*hop+win_n/2)/fs*1e6
        out.append(f"{tm:6.1f} {''.join(line)}")
    # column header: frequency marks every 50kHz
    head = [" "*6]
    marks = {int((f-0.95e6)/4e3): f"{int(f/1e3)}k" for f in fr if abs(f%100e3)<5e3 or abs((f-1e6)%100e3)<5e3}
    hdr = [' ']*len(fr)
    for i, f in enumerate(fr):
        kk = int(round(f/50e3)*50e3)
        if abs(f-kk) < 10e3:
            for j, ch in enumerate(f"{kk/1e3:.0f}k"):
                if i+j < len(hdr): hdr[i+j]=ch
    return ''.join(hdr), out, fr

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    cmd = [s for s, du in zip(segs, durs) if du >= 60]
    idle = [s for s, du in zip(segs, durs) if du < 60]

    print("### COMMAND burst spectrogram (freq 0.95-1.05 MHz, 2us win, 1us hop)")
    hdr, rows, fr = spectro(cmd[10][1], cmd[10][2], cmd[10][3])
    print(hdr)
    for r in rows: print(r)

    print("\n### IDLE burst spectrogram")
    hdr, rows, fr = spectro(idle[100][1], idle[100][2], idle[100][3])
    print(hdr)
    for r in rows: print(r)

    # envelope cross-correlation of command bursts (phase-independent)
    print("\n### command envelope cross-correlation (vs cmd[10])")
    _, t0, v0, _ = cmd[10]
    ref = np.abs(np.hilbert(v0))
    ref = ref[ref>0]  # trim
    for i in (0, 5, 15, 20):
        _, t, v, fs = cmd[i]
        e = np.abs(np.hilbert(v))
        # coarse cross-correlate at 1us grid
        n = min(len(ref), len(e))
        r_ = ref[:n]; e_ = e[:n]
        cc = np.correlate(e_-e_.mean(), r_-r_.mean(), 'full')
        k = np.argmax(cc)
        lag = (k-(n-1))/fs*1e6
        print(f"  cmd[{i}]: peak lag={lag:6.2f}us  corr={cc[k]/(np.std(e_)*np.std(r_)*n):.3f}")

if __name__ == "__main__":
    main()
