#!/usr/bin/env python3
"""Per-burst duration stats; list longest bursts (likely non-idle command bursts)."""
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

def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(REPO,d,'*.csv')),
                   key=lambda p:int(re.search(r'segment-(\d+)_',p).group(1)))
    durs=[]
    for p in files:
        t,v=load_csv(p); dt=np.median(np.diff(t)); fs=1/dt; v=v-np.mean(v)
        fb=find_burst(t,v)
        if fb is None: continue
        a,b=fb
        durs.append((p,(t[b]-t[a])*1e6))
    durs=np.array([x[1] for x in durs])
    print(f"burst durations (us): min={durs.min():.1f} max={durs.max():.1f} median={np.median(durs):.1f}")
    print(f"  hist: <45:{np.sum(durs<45)} 45-50:{np.sum((durs>=45)&(durs<50))} 50-55:{np.sum((durs>=50)&(durs<55))} 55-60:{np.sum((durs>=55)&(durs<60))} >60:{np.sum(durs>=60)}")
    order=np.argsort(durs)[::-1]
    print("longest 15 bursts:")
    for o in order[:15]:
        print(f"   {os.path.basename(files[o]):24s} {durs[o]:.1f}us")

if __name__=='__main__':
    main()
