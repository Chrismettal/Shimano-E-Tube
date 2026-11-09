#!/usr/bin/env python3
"""Instantaneous frequency analysis of a long burst + stacked eye of all long bursts."""
import sys, os, glob, re
import numpy as np
from scipy import signal as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(REPO,d,'*.csv')),
                   key=lambda p:int(re.search(r'segment-(\d+)_',p).group(1)))
    longbursts=[]
    for p in files:
        t,v=load_csv(p); dt=np.median(np.diff(t)); fs=1/dt; v=v-np.mean(v)
        fb=find_burst(t,v)
        if fb is None: continue
        a,b=fb
        if (t[b]-t[a])*1e6>100:
            longbursts.append((os.path.basename(p),t[a:b+1],v[a:b+1],fs))
    print(f"long bursts: {len(longbursts)}")
    name,tb,vb,fs=longbursts[0]
    # instantaneous frequency via hilbert phase derivative
    z=sps.hilbert(vb)
    phase=np.unwrap(np.angle(z))
    instf=np.diff(phase)/(2*np.pi)*fs
    fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
    ax[0].plot(tb*1e6,vb); ax[0].set_title(f"raw {name}"); ax[0].set_ylabel("V")
    ax[1].plot(tb[1:]*1e6,instf/1e6); ax[1].set_ylabel("inst freq (MHz)"); ax[1].grid(True)
    # envelope
    ax[2].plot(tb*1e6,np.abs(z)); ax[2].set_ylabel("env"); ax[2].set_xlabel("us"); ax[2].grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(REPO,'analysis','longburst.png'),dpi=110)
    print("saved analysis/longburst.png")
    # freq histogram
    h,edges=np.histogram(instf/1e6,bins=40,range=(0.5,1.5))
    print("inst freq peaks (MHz):", [f"{edges[i]:.3f}" for i in np.argsort(h)[-4:]])

if __name__=='__main__':
    main()
