#!/usr/bin/env python3
"""Reliable BPSK command decode.
Per burst:
  1. coarse+fine fc search maximizing bimodality (2-level phases).
  2. FINE fc refinement maximizing MIN per-symbol coherent |v|  (flattens
     profile => phase constant in every bin, no straddled transitions).
  3. hard-decode each symbol: bit = sign(Re(v_k * conj(ref))) with ref =
     strongest symbol (absolute bits, up to a per-burst global complement).
  4. cross-burst alignment via differential head (pi-immune).
Reports per-symbol agreement across the (correct) frame classes.
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
    return bim, float(m.min()/ (m.max()+1e-30))

def lock(t,v,n_sym):
    best_bim=(-1.0,None)
    for fc in np.arange(0.955e6,1.045e6,500.0):
        b,_=metrics(symvecs(t,v,fc,n_sym))
        if b>best_bim[0]: best_bim=(b,fc)
    # fine by bim
    best=(-1.0,None)
    for fc in np.arange(best_bim[1]-1e3,best_bim[1]+1e3,50.0):
        b,_=metrics(symvecs(t,v,fc,n_sym))
        if b>best[0]: best=(b,fc)
    fc=best[1]
    # fine by max-min (flatten weakest symbol)
    best_mm=(-1.0,fc)
    for fc2 in np.arange(fc-500,fc+500,10.0):
        _,mm=metrics(symvecs(t,v,fc2,n_sym))
        if mm>best_mm[0]: best_mm=(mm,fc2)
    fc=best_mm[1]
    b,_=metrics(symvecs(t,v,fc,n_sym))
    return fc,b,symvecs(t,v,fc,n_sym)

def main():
    d=sys.argv[1]
    segs=burst_segments(d)
    durs=[(t[-1]-t[0])*1e6 for _,t,v,fs in segs]
    cmd=[s for s,du in zip(segs,durs) if du>=60]
    recs=[]
    for i,(_,t,v,fs) in enumerate(cmd):
        fc,b,vecs=lock(t,v,16)
        n=len(vecs); ref=vecs[int(np.argmax(np.abs(vecs)))]
        ab=np.zeros(n,int)   # absolute vs ref
        for k in range(n): ab[k]=1 if np.real(vecs[k]*np.conj(ref))<0 else 0
        db=np.zeros(n,int)   # differential
        for k in range(1,n): db[k]=1 if np.real(vecs[k]*np.conj(vecs[k-1]))<0 else 0
        recs.append((i,fc,b,ab,db))
    def dh(ab):
        d=np.zeros(len(ab),int)
        for k in range(1,len(ab)): d[k]=ab[k]^ab[k-1]
        return d
    ref=recs[0]
    refdh=dh(ref[3])
    R=np.zeros((len(recs),16),int)
    for i,fc,b,ab,db in recs:
        c=ab if np.array_equal(dh(ab)[:6],refdh[:6]) else (1-ab)
        R[i]=c
    print(f"{'frm':>3} {'fc':>9} {'bim':>5}  abs-bits (aligned)   diff-bits")
    for i,fc,b,ab,db in recs:
        print(f"{i:3d} {fc/1e6:9.5f} {b:5.3f}  {''.join(map(str,R[i]))}  {''.join(map(str,db))}")
    maj=(R.mean(0)>0.5).astype(int)
    agree=abs(R.mean(0)-0.5)*2
    print("\naligned consensus:", ''.join(map(str,maj)))
    print("agreement        :", " ".join(f"{a:.2f}" for a in agree))
    for lab,arr in (("maj",maj),("~maj",1-maj)):
        hx=' '.join(f'{int("".join(map(str,arr[p:p+8])),2):02x}' for p in range(0,16,8))
        print(f"hex({lab})         :", hx)

if __name__=="__main__":
    main()
