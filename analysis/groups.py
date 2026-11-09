#!/usr/bin/env python3
"""Group command frames by differential head (first 6 bits) and report
within-group agreement. Confirms whether the 50/50 split is two genuine
frame classes. Prints per-group consensus + std.
"""
import sys
import numpy as np
from fsk import burst_segments

def symvecs(t, v, fc, n_sym):
    T = t - t[0]; v0 = v - v.mean()
    c = np.cumsum(v0*np.exp(-2j*np.pi*fc*T))
    edges = np.arange(n_sym+1)*(T[-1]/n_sym)
    idx = np.clip(np.searchsorted(T, edges), 0, len(v0)-1)
    return (c[idx[1:]]-c[idx[:-1]])

def score_of(vecs):
    m=np.abs(vecs)
    if m.max()==0: return 0.0
    k=int(np.argmax(m)); rel=np.angle(vecs*np.conj(vecs[k]))
    return float(np.mean(np.abs(np.cos(2*rel))))

def search(t,v,n_sym,lo,hi,step):
    best=(-1.0,None)
    for fc in np.arange(lo,hi,step):
        s=score_of(symvecs(t,v,fc,n_sym))
        if s>best[0]: best=(s,fc)
    return best

def lock(t,v,n_sym):
    s1,fc=search(t,v,n_sym,0.955e6,1.045e6,500.0)
    s2,fc=search(t,v,n_sym,fc-1e3,fc+1e3,50.0)
    return fc,s2,symvecs(t,v,fc,n_sym)

def main():
    d=sys.argv[1]
    segs=burst_segments(d)
    durs=[(t[-1]-t[0])*1e6 for _,t,v,fs in segs]
    cmd=[s for s,du in zip(segs,durs) if du>=60]
    recs=[]
    for i,(_,t,v,fs) in enumerate(cmd):
        fc,s,vecs=lock(t,v,16)
        n=len(vecs); dbit=np.zeros(n,int)
        for k in range(1,n):
            dbit[k]=1 if np.real(vecs[k]*np.conj(vecs[k-1]))<0 else 0
        recs.append((i,dbit,s,fc))
    # group by 6-bit head
    groups={}
    for i,dbit,s,fc in recs:
        key=''.join(map(str,dbit[:6]))
        groups.setdefault(key,[]).append((i,dbit))
    for key in sorted(groups, key=lambda k:-len(groups[k])):
        members=groups[key]
        print(f"\n=== head {key}  ({len(members)} frames: {[m[0] for m in members]})")
        G=np.array([m[1] for m in members])
        maj=(G.mean(0)>0.5).astype(int)
        agree=abs(G.mean(0)-0.5)*2
        print("  consensus :", ''.join(map(str,maj)))
        print("  agreement :", " ".join(f"{a:.2f}" for a in agree))
        print("  std       :", " ".join(f"{G[:,p].std():.2f}" for p in range(16)))
        hx=' '.join(f'{int("".join(map(str,maj[p:p+8])),2):02x}' for p in range(0,16,8))
        print("  hex       :", hx)
        for i,dbit in members:
            print(f"    [{i:2d}] {''.join(map(str,dbit))}")

if __name__=="__main__":
    main()
