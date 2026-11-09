#!/usr/bin/env python3
"""Decode the long (command) bursts: 130us ~ 16 bits. Find exact bit count by eye."""
import sys, os, glob, re
import numpy as np
from scipy import signal as sps
from collections import Counter

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
    segs = sorted(glob.glob(os.path.join(REPO,'data/scopes/2024-01-01_ButtonDownManySegments/*.csv')))
    long = []
    for f in segs:
        t,v = load_csv(f)
        r = find_burst(t,v)
        if r:
            dur=(t[r[1]]-t[r[0]])*1e6
            if dur>100: long.append((f,dur))
    print(f"long bursts: {len(long)}")
    long.sort(key=lambda x:-x[1])
    f,dur = long[0]
    print(f"longest: {f.split('/')[-1]} dur={dur:.1f}us")
    t,v = load_csv(f)
    b0=np.argmax(v==1);b1=len(v)-1-np.argmax(v[::-1]==1)
    x=v[b0:b1]

    # search bit rate + bit count for clean eye
    best=None
    for br in np.arange(90e3,140e3,1e3):
        for n in range(12,21):
            step=n/x.size
            bits=''.join('1' if x[int(i*step)]>=0.5 else '0' for i in range(n))
            # eye separation: mean of bit-1 samples vs bit-0 samples
            s1=[x[int(i*step)] for i in range(n) if bits[i]=='1']
            s0=[x[int(i*step)] for i in range(n) if bits[i]=='0']
            if not s1 or not s0: continue
            sep=abs(np.mean(s1)-np.mean(s0))
            if best is None or sep>best[0]:
                best=(sep,br,n,bits)
    sep,br,n,bits=best
    print(f"best: br={br/1000:.1f}k nbits={n} sep={sep:.3f}")
    print(f"  bits: {bits}")
    for i in range(0,len(bits)-7,8):
        print(f"  byte {i//8}: {bits[i:i+8]} = 0x{int(bits[i:i+8],2):02x}")

    # now decode every long burst with these params and compare
    print("\nall long bursts:")
    pat=Counter()
    for f,dur in long:
        t,v = load_csv(f)
        b0=np.argmax(v==1);b1=len(v)-1-np.argmax(v[::-1]==1)
        x=v[b0:b1]
        n=int(round(dur*1e-6*br));step=n/x.size
        bits=''.join('1' if x[int(i*step)]>=0.5 else '0' for i in range(n))
        pat[bits]+=1
        print(f"  {f.split('/')[-1]}: {bits}")
    print("\nunique patterns:")
    for b,c in pat.most_common():
        print(f"  {c}x {b}")

if __name__=='__main__':
    main()
