#!/usr/bin/env python3
"""Measure bit rate via transition counting in I channel; build symbol histogram."""
import sys, os, glob, re
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

def find_burst(t, v):
    z = sps.hilbert(v - np.mean(v))
    env = np.abs(z)
    thr = env.max()*0.15
    st = np.where(env > thr)[0]
    if len(st)==0: return None
    gr = np.split(st, np.where(np.diff(st) > 20)[0]+1)
    g = max((gg for gg in gr if len(gg)>10), key=len)
    return g[0], g[-1]

def demod_I(tb, vb, fs):
    w = tb - tb[0]
    spec = np.abs(np.fft.rfft(vb*np.hanning(len(vb))))
    freqs = np.fft.rfftfreq(len(vb), 1/fs)
    m = (freqs>0.8e6)&(freqs<1.2e6)
    carr = freqs[m][np.argmax(spec[m])]
    iq = vb*np.exp(-1j*2*np.pi*carr*w)
    I = sps.filtfilt(*sps.butter(4, 250e3/(fs/2)), np.real(iq))
    return I, carr

def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(REPO,d,'*.csv')),
                   key=lambda p:int(re.search(r'segment-(\d+)_',p).group(1)))
    # For each burst: count I-level transitions, estimate bits
    trans=[]
    for p in files[:100]:
        t,v=load_csv(p); dt=np.median(np.diff(t)); fs=1/dt; v=v-np.mean(v)
        fb=find_burst(t,v)
        if fb is None: continue
        I,carr=demod_I(t[fb[0]:fb[1]+1],v[fb[0]:fb[1]+1],fs)
        thr=(I.max()+I.min())/2
        lvl=(I>thr).astype(int)
        # count transitions
        tr=np.sum(np.abs(np.diff(lvl))>0)
        trans.append(tr)
    trans=np.array(trans)
    print(f"I-level transitions/burst: median={np.median(trans):.0f} mean={trans.mean():.1f}")
    # For a 50us burst: if N transitions, bits ~ N+1
    # idle 000011 repeating: pattern period 6 bits, transitions: 0->0(0),0->0,0->1(1),1->1,1->0(1),0->0 => 2 transitions per 6 bits
    # so bits ~ 3 * transitions
    print(f"  -> implied bits/burst (3x trans): median={3*np.median(trans):.0f}")
    print(f"  -> implied bit rate: 3*{np.median(trans):.0f} bits / 50us = {3*np.median(trans)/50e-6/1e3:.0f} kbit/s")

if __name__=='__main__':
    main()
