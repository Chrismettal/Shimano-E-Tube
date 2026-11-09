#!/usr/bin/env python3
"""Find true symbol count by flattening per-symbol coherent magnitude.
For a candidate N, split burst into N equal bins, compute coherent |v| per
bin (with score-locked fc). The correct N makes the phase constant within
each bin -> |v|max/sqrt(raw) ~ flat (near 1 everywhere). Print a few N.
"""
import sys
import numpy as np
from fsk import burst_segments

def lock(t,v,n_sym):
    def s_of(fc):
        T=t-t[0]; v0=v-v.mean()
        c=np.cumsum(v0*np.exp(-2j*np.pi*fc*T))
        edges=np.arange(n_sym+1)*(T[-1]/n_sym)
        idx=np.clip(np.searchsorted(T,edges),0,len(v0)-1)
        vecs=c[idx[1:]]-c[idx[:-1]]; m=np.abs(vecs)
        if m.max()==0: return 0.0
        k=int(np.argmax(m)); rel=np.angle(vecs*np.conj(vecs[k]))
        return float(np.mean(np.abs(np.cos(2*rel))))
    best=(-1.0,None)
    for fc in np.arange(0.955e6,1.045e6,500.0):
        s=s_of(fc)
        if s>best[0]: best=(s,fc)
    for fc in np.arange(best[1]-1e3,best[1]+1e3,50.0):
        s=s_of(fc)
        if s>best[0]: best=(s,fc)
    return best[1],best[0]

def main():
    d=sys.argv[1]
    segs=burst_segments(d)
    durs=[(t[-1]-t[0])*1e6 for _,t,v,fs in segs]
    cmd=[s for s,du in zip(segs,durs) if du>=60]
    _,t,v,fs=cmd[7]
    fc,sc=lock(t,v,16)
    print(f"cmd[7] fc={fc/1e6:.5f} (lock score@16 = {sc:.3f})")
    T=t-t[0]; v0=v-v.mean()
    c16=np.cumsum(v0*np.exp(-2j*np.pi*fc*T))
    for N in (8,16,32,64):
        edges=np.arange(N+1)*(T[-1]/N)
        idx=np.clip(np.searchsorted(T,edges),0,len(v0)-1)
        vecs=c16[idx[1:]]-c16[idx[:-1]]
        m=np.abs(vecs); raw=np.array([(v0[idx[k]:idx[k+1]]**2).sum() for k in range(N)])
        # normalize coherent by sqrt(raw) -> 1 = phase constant, <1 = flip inside
        coh=m/np.sqrt(raw+1e-30)
        print(f"\nN={N:2d}  coh/sqrt(raw): min={coh.min():.2f} max={coh.max():.2f} "
              f"mean={coh.mean():.2f}  std={coh.std():.2f}  (flat~1 = correct)")
        print("   " + ' '.join(f"{x:4.2f}" for x in coh))

if __name__=="__main__":
    main()
