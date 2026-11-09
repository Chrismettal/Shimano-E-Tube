#!/usr/bin/env python3
"""Differential BPSK demod with per-burst score-locked carrier.
bit_i = 1 iff Re(v_i * conj(v_{i-1})) < 0  (phase flipped between symbols)
Reference-free, immune to carrier phase + pi-ambiguity.
Also prints coherent bits (vs max-mag ref) for comparison.
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
    m = np.abs(vecs)
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
    drows=[]
    print(f"{'frm':>3} {'fc':>9} {'score':>6}  diff-bits")
    for i,(_,t,v,fs) in enumerate(cmd):
        fc,s,vecs=lock(t,v,16)
        n=len(vecs)
        dbit=np.zeros(n,int)
        for k in range(1,n):
            dbit[k]=1 if np.real(vecs[k]*np.conj(vecs[k-1]))<0 else 0
        drows.append(dbit)
        print(f"{i:3d} {fc/1e6:9.5f} {s:6.3f}  {''.join(map(str,dbit))}")
    L=max(len(r) for r in drows)
    R=np.zeros((len(drows),L),int)
    for i in range(len(drows)): R[i,:len(drows[i])]=drows[i]
    maj=(R.mean(0)>0.5).astype(int)
    print("\ndiff consensus :", ''.join(map(str,maj)))
    print("std           :", " ".join(f"{R[:,p].std():.2f}" for p in range(L)))
    print("per-sym agree :", " ".join(f"{abs(R[:,p].mean()-0.5)*2:.2f}" for p in range(L)))
    for lab,arr in (("maj",maj),("~maj",1-maj)):
        hx=' '.join(f'{int("".join(map(str,arr[p:p+8])),2):02x}' for p in range(0,L-L%8,8))
        print(f"hex({lab})   :", hx)

if __name__=="__main__":
    main()
