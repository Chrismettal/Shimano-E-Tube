#!/usr/bin/env python3
"""Robust per-burst BPSK decoder: precise carrier via FFT, then eye-optimize symbol rate."""
import sys, os
import numpy as np
from scipy import signal as sps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_csv(path):
    t, v = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 2: continue
            a, b = line.split(",")
            t.append(float(a)); v.append(float(b))
    return np.array(t), np.array(v)

def find_bursts(t, v):
    z = sps.hilbert(v - np.mean(v))
    env = np.abs(z)
    thr = env.max()*0.15
    st = np.where(env > thr)[0]
    gr = np.split(st, np.where(np.diff(st) > 20)[0]+1)
    return [(g[0], g[-1]) for g in gr if len(g) > 10]

def decode_burst(tb, vb, fs):
    # precise carrier via FFT on windowed burst
    w = tb - tb[0]
    spec = np.abs(np.fft.rfft(vb*np.hanning(len(vb))))
    freqs = np.fft.rfftfreq(len(vb), 1/fs)
    m = (freqs > 0.8e6) & (freqs < 1.2e6)
    carr = freqs[m][np.argmax(spec[m])]
    iq = vb*np.exp(-1j*2*np.pi*carr*w)
    I = sps.filtfilt(*sps.butter(4, 250e3/(fs/2)), np.real(iq))
    Q = sps.filtfilt(*sps.butter(4, 250e3/(fs/2)), np.imag(iq))
    dur = tb[-1]-tb[0]
    best = None
    for sym in np.arange(50e3, 400e3, 2e3):
        n_sym = int(round(dur*sym))
        if n_sym < 3 or n_sym > 120: continue
        for off in np.linspace(0, 1/sym, 12, endpoint=False):
            ts = tb[0]+off+np.arange(n_sym)/sym
            if ts[0] < tb[0] or ts[-1] > tb[-1]: continue
            Is = np.interp(ts, tb, I)
            thr2 = (Is.max()+Is.min())/2
            hi = Is[Is>thr2]; lo = Is[Is<=thr2]
            if len(hi)<2 or len(lo)<2: continue
            sep = (hi.mean()-lo.mean())**2/(hi.std()+lo.std()+1e-9)
            if best is None or sep>best[0]:
                best=(sep,sym,off,Is,hi.mean(),lo.mean())
    sep,sym,off,Is,mhi,mlo = best
    bits = (Is > (mhi+mlo)/2).astype(int)
    return carr, sym, bits, I, Q

if __name__ == "__main__":
    path = sys.argv[1]
    t, v = load_csv(os.path.join(REPO, path))
    dt = np.median(np.diff(t)); fs = 1/dt
    v = v - np.mean(v)
    spans = find_bursts(t, v)
    print(f"file: {path}  fs={fs/1e6:.2f}MHz  bursts={len(spans)}")
    for bi,(a,b) in enumerate(spans[:int(sys.argv[2]) if len(sys.argv)>2 else 20]):
        carr,sym,bits,I,Q = decode_burst(t[a:b+1], v[a:b+1], fs)
        print(f"b{bi:3d} carr={carr/1e6:.4f}MHz sym={sym/1e3:.0f}k n={len(bits)} bits={''.join(map(str,bits))}")
