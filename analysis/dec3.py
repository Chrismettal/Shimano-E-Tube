#!/usr/bin/env python3
"""Reliable BPSK command decode, grouped by differential class.
Per burst: coarse+fine fc (bimodality) then max-min refinement (flatten
weakest symbol). Hard-decode absolute bits vs the frame's strongest symbol
(up to a per-frame global complement). Group by the pi-immune differential
head; within a group align all frames to a group reference by matching the
strongest symbol, then per-group absolute consensus.
"""
import sys
import numpy as np
from fsk import burst_segments

def symvecs(t, v, fc, n_sym):
    T=t-t[0]; v0=v-v.mean()
    c=np.cumsum(v0*np.exp(-2j*np.pi*fc*T))
    edges=np.arange(n_sym+1)*(T[-1]/n_sym)
    idx=np.clip(np.searchsorted(T,edges),0,len(v0)-1)
    return c[idx[1:]]-c[idx[:-1]]

def metrics(vecs):
    m=np.abs(vecs)
    if m.max()==0: return 0.0,0.0
    k=int(np.argmax(m)); rel=np.angle(vecs*np.conj(vecs[k]))
    bim=float(np.mean(np.abs(np.cos(2*rel))))
    return bim, float(m.min()/(m.max()+1e-30))

def lock(t,v,n_sym):
    best=(-1.0,None)
    for fc in np.arange(0.955e6,1.045e6,500.0):
        b,_=metrics(symvecs(t,v,fc,n_sym))
        if b>best[0]: best=(b,fc)
    for fc in np.arange(best[1]-1e3,best[1]+1e3,50.0):
        b,_=metrics(symvecs(t,v,fc,n_sym))
        if b>best[0]: best=(b,fc)
    fc=best[1]
    best_mm=(-1.0,fc)
    for fc2 in np.arange(fc-500,fc+500,10.0):
        _,mm=metrics(symvecs(t,v,fc2,n_sym))
        if mm>best_mm[0]: best_mm=(mm,fc2)
    return best_mm[1],symvecs(t,v,fc,n_sym)

def dh(ab):
    d=np.zeros(len(ab),int)
    for k in range(1,len(ab)): d[k]=ab[k]^ab[k-1]
    return d

def main():
    d=sys.argv[1]
    segs=burst_segments(d)
    durs=[(t[-1]-t[0])*1e6 for _,t,v,fs in segs]
    cmd=[s for s,du in zip(segs,durs) if du>=60]
    recs=[]
    for i,(_,t,v,fs) in enumerate(cmd):
        fc,vecs=lock(t,v,16)
        n=len(vecs); m=np.abs(vecs); rk=int(np.argmax(m))
        ref=vecs[rk]
        ab=np.zeros(n,int)
        for k in range(n): ab[k]=1 if np.real(vecs[k]*np.conj(ref))<0 else 0
        b,_=metrics(vecs)
        recs.append(dict(i=i,fc=fc,bim=b,ab=ab,dk=rk,mag=m/(m.max()+1e-30)))
    # quality filter: bimodality>=0.92 and not a side-lobe mislock
    good=[r for r in recs if r["bim"]>=0.92 and abs(r["fc"]-1.0e6)<20e3]
    print(f"good frames: {len(good)}/{len(recs)} (bim>=0.92, |fc-1MHz|<20kHz)")
    # group by differential head (first 8 bits)
    groups={}
    for r in good:
        key=''.join(map(str,dh(r["ab"])[:8]))
        groups.setdefault(key,[]).append(r)
    for key in sorted(groups,key=lambda k:-len(groups[k])):
        g=groups[key]
        # group reference = first member
        refg=g[0]; rdk=refg["dk"]; refbit=refg["ab"][rdk]
        A=np.zeros((len(g),16),int)
        for j,r in enumerate(g):
            c=r["ab"]
            # align: flip so that r's strongest symbol matches refg's strongest bit
            if c[r["dk"]] != refbit: c=1-c
            A[j]=c
        maj=(A.mean(0)>0.5).astype(int)
        agree=abs(A.mean(0)-0.5)*2
        print(f"\n=== class head {key}  ({len(g)} frames {[r['i'] for r in g]})")
        print("  abs consensus (up to global complement):", ''.join(map(str,maj)))
        print("  agreement:", " ".join(f"{a:.2f}" for a in agree))
        hx=' '.join(f'{int("".join(map(str,maj[p:p+8])),2):02x}' for p in range(0,16,8))
        print("  hex       :", hx, "  (~", ' '.join(f'{int("".join(map(str,(1-maj)[p:(p+8)])),2):02x}' for p in range(0,16,8)), ")")
        print("  diff head :", ''.join(map(str,dh(maj))))
        for j,r in enumerate(g):
            c=r["ab"]
            if c[r["dk"]] != refbit: c=1-c
            print(f"    [{r['i']:2d}] bim={r['bim']:.3f} {''.join(map(str,c))}   dk={r['dk']}")

if __name__=="__main__":
    main()
