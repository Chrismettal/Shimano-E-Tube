#!/usr/bin/env python3
"""Precise bit-rate measurement + per-burst 6-bit symbol extraction for a multi-burst capture."""
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

def bit_rate_from_I(I, fs):
    # autocorrelation of I to find bit period
    I = I - I.mean()
    ac = np.correlate(I, I, 'full')[len(I)-1:]
    ac = ac/ac[0]
    # find first strong peak after lag 0
    # search in a window corresponding to 50-200 us
    lo = int(50e-6*fs); hi=int(250e-6*fs)
    seg = ac[lo:hi]
    peak = lo + np.argmax(seg)
    return 1/(peak/fs)

def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(REPO,d,'*.csv')),
                   key=lambda p:int(re.search(r'segment-(\d+)_',p).group(1)))
    # measure bit rate on a few bursts
    brs=[]
    for p in files[:30]:
        t,v=load_csv(p); dt=np.median(np.diff(t)); fs=1/dt; v=v-np.mean(v)
        fb=find_burst(t,v)
        if fb is None: continue
        I,carr=demod_I(t[fb[0]:fb[1]+1],v[fb[0]:fb[1]+1],fs)
        br=bit_rate_from_I(I,fs)
        brs.append(br)
    brs=np.array(brs)
    print(f"bit rate (autocorr): median={np.median(brs)/1e3:.1f}k min={brs.min()/1e3:.1f}k max={brs.max()/1e3:.1f}k")

    # now extract 6-bit symbols at the median bit rate
    br=np.median(brs)
    syms=[]
    for p in files:
        t,v=load_csv(p); dt=np.median(np.diff(t)); fs=1/dt; v=v-np.mean(v)
        fb=find_burst(t,v)
        if fb is None: continue
        a,b=fb; I,carr=demod_I(t[a:b+1],v[a:b+1],fs)
        tb=t[a:b+1]
        dur=tb[-1]-tb[0]
        n=int(round(dur*br))
        ts=tb[0]+dur/2-(n-1)/(2*br)+np.arange(n)/br
        ts=ts[(ts>=tb[0])&(ts<=tb[-1])]
        Is=np.interp(ts,tb,I)
        thr=(Is.max()+Is.min())/2
        bits=(Is>thr).astype(int)
        syms.append(bits)
    # print the symbol sequence, marking non-idle
    from collections import Counter
    c=Counter(tuple(s) for s in syms)
    print("symbol histogram (top 15):")
    for s,cnt in c.most_common(15):
        print(f"   {''.join(map(str,s)):12s} {cnt:4d}  (val={int(''.join(map(str,s)),2):3d})")

if __name__=='__main__':
    main()
