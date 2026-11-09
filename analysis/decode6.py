#!/usr/bin/env python3
"""Final: 6 bits/burst @ 120 kbit/s, per-burst carrier. Print symbol value sequence."""
import sys, os, glob, re
import numpy as np
from scipy import signal as sps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BR = 120e3
NB = 6

def load_csv(path):
    t, v = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 2: continue
            a, b = line.split(",")
            t.append(float(a)); v.append(float(b))
    return np.array(t), np.array(v)

def find_burst(t, v):
    z = sps.hilbert(v - np.mean(v))
    env = np.abs(z)
    thr = env.max()*0.15
    st = np.where(env > thr)[0]
    if len(st)==0: return None
    gr = np.split(st, np.where(np.diff(st) > 20)[0]+1)
    g = max((gg for gg in gr if len(gg)>10), key=len)
    return g[0], g[-1]

def decode6(t, v, fs):
    fb = find_burst(t, v)
    if fb is None: return None
    a, b = fb
    tb, vb = t[a:b+1], v[a:b+1]
    w = tb - tb[0]
    spec = np.abs(np.fft.rfft(vb*np.hanning(len(vb))))
    freqs = np.fft.rfftfreq(len(vb), 1/fs)
    m = (freqs>0.8e6)&(freqs<1.2e6)
    carr = freqs[m][np.argmax(spec[m])]
    iq = vb*np.exp(-1j*2*np.pi*carr*w)
    I = sps.filtfilt(*sps.butter(4, 250e3/(fs/2)), np.real(iq))
    # sample at 6 bit centers over the burst duration
    dur = tb[-1]-tb[0]
    ts = tb[0] + dur/2 - (NB-1)/(2*BR) + np.arange(NB)/BR
    # clamp to burst
    ts = np.clip(ts, tb[0], tb[-1])
    Is = np.interp(ts, tb, I)
    thr = (Is.max()+Is.min())/2
    bits = (Is>thr).astype(int)
    val = int(''.join(map(str,bits)),2)
    return bits, val, carr

def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(REPO,d,'*.csv')),
                   key=lambda p:int(re.search(r'segment-(\d+)_',p).group(1)))
    vals=[]; carrs=[]
    for p in files:
        t,v=load_csv(p); dt=np.median(np.diff(t)); fs=1/dt; v=v-np.mean(v)
        r=decode6(t,v,fs)
        if r is None: continue
        bits,val,carr=r
        vals.append(val); carrs.append(carr)
    vals=np.array(vals)
    print(f"bursts: {len(vals)}  carrier mean={np.mean(carrs)/1e6:.4f}MHz")
    print(f"distinct values: {sorted(set(vals.tolist()))}")
    # print sequence, compress runs of idle(3)
    from itertools import groupby
    print("sequence (idle=3 compressed as xN):")
    out=[]
    for k,g in groupby(vals):
        n=len(list(g))
        if k==3 and n>1: out.append(f"3x{n}")
        else: out.append(str(int(k)))
    print(" ".join(out))

if __name__=='__main__':
    main()
