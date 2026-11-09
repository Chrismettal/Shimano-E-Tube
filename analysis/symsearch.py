#!/usr/bin/env python3
"""Symbol-rate search + BPSK decode for one burst.

Strategy:
  - demod to I/Q at carrier (1MHz)
  - for candidate symbol rates, sample I at best phase and look at level
    separation (eye quality): maximize (mean_hi - mean_lo)^2 / (var_hi+var_lo)
  - report best rate, then decode bits
"""
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

path = sys.argv[1]
t, v = load_csv(os.path.join(REPO, path))
dt = np.median(np.diff(t)); fs = 1/dt
v = v - np.mean(v)
z = sps.hilbert(v)

# burst spans
env = np.abs(z)
thr = env.max()*0.15
start = np.where(env > thr)[0]
groups = np.split(start, np.where(np.diff(start) > 20)[0]+1)
spans = [(g[0], g[-1]) for g in groups if len(g) > 10]

CARR = 1.0e6
results = []
for bi, (a, b) in enumerate(spans):
    tb, vb = t[a:b+1], v[a:b+1]
    iq = z[a:b+1]*np.exp(-1j*2*np.pi*CARR*(tb-tb[0]))
    I = sps.filtfilt(*sps.butter(4, 300e3/(fs/2)), np.real(iq))
    Q = sps.filtfilt(*sps.butter(4, 300e3/(fs/2)), np.imag(iq))
    dur = tb[-1]-tb[0]
    best = None
    for sym in np.arange(40e3, 600e3, 1e3):
        n_sym = int(round(dur*sym))
        if n_sym < 4 or n_sym > 200: continue
        # try all phase offsets, pick best level separation
        for off in np.linspace(0, 1/sym, 8, endpoint=False):
            ts = tb[0] + off + np.arange(n_sym)/(sym)
            if ts[0] < tb[0] or ts[-1] > tb[-1]: continue
            Is = np.interp(ts, tb, I)
            # two-level clustering
            thr2 = (Is.max()+Is.min())/2
            hi = Is[Is > thr2]; lo = Is[Is <= thr2]
            if len(hi) < 2 or len(lo) < 2: continue
            sep = (hi.mean()-lo.mean())**2 / (hi.std()+lo.std()+1e-9)
            if best is None or sep > best[0]:
                best = (sep, sym, off, Is, (hi.mean(), lo.mean()))
    sep, sym, off, Is, (mhi, mlo) = best
    bits = (Is > (mhi+mlo)/2).astype(int)
    results.append((bi, sym, bits))
    # byte framing both orders
    def to_bytes(bits, msb_first=True):
        out=[]
        for i in range(0, len(bits)-7, 8):
            ch = bits[i:i+8]
            out.append(int("".join(map(str, ch if msb_first else ch[::-1])),2))
        return out
    print(f"burst {bi}: best sym rate {sym/1e3:.0f} ksym/s  sep={sep:.2f}  levels lo={mlo:.3f} hi={mhi:.3f}  n_bits={len(bits)}")
    print(f"   bits: {''.join(map(str,bits))}")
    print(f"   bytes MSB: {' '.join(f'{x:02x}' for x in to_bytes(bits,True))}")
    print(f"   bytes LSB: {' '.join(f'{x:02x}' for x in to_bytes(bits,False))}")
