#!/usr/bin/env python3
"""Timeline of command frames classified by PI-IMMUNE differential head.
The absolute head (110100 vs 100001) is per-frame reference-dependent and
unreliable; the differential head d[k]=ab[k]^ab[k-1] is invariant under a
global phase flip, so it is the true class key. Print each command frame's
segment, its absolute bits, and its 8-bit diff head; then show the burst
structure (consecutive non-idle runs) and whether each burst is homogeneous.
"""
import sys
import numpy as np
from fsk import burst_segments

def lock(t,v,n_sym):
    def metrics(fc):
        T=t-t[0]; v0=v-v.mean()
        c=np.cumsum(v0*np.exp(-2j*np.pi*fc*T))
        edges=np.arange(n_sym+1)*(T[-1]/n_sym)
        idx=np.clip(np.searchsorted(T,edges),0,len(v0)-1)
        vecs=c[idx[1:]]-c[idx[:-1]]; m=np.abs(vecs)
        if m.max()==0: return 0.0,vecs
        k=int(np.argmax(m)); rel=np.angle(vecs*np.conj(vecs[k]))
        return float(np.mean(np.abs(np.cos(2*rel)))),vecs
    best=(-1.0,None)
    for fc in np.arange(0.955e6,1.045e6,500.0):
        b,_=metrics(fc)
        if b>best[0]: best=(b,fc)
    for fc in np.arange(best[1]-1e3,best[1]+1e3,50.0):
        b,_=metrics(fc)
        if b>best[0]: best=(b,fc)
    return best[1],metrics(best[1])[1]

def dh(ab):
    d=[0]+[ab[k]^ab[k-1] for k in range(1,len(ab))]
    return d

def main():
    d=sys.argv[1]
    segs=burst_segments(d)
    frames=[]
    for idx,(_,t,v,fs) in enumerate(segs):
        dur=(t[-1]-t[0])*1e6
        if dur<60: continue
        fc,vecs=lock(t,v,16)
        n=len(vecs); rk=int(np.argmax(np.abs(vecs))); ref=vecs[rk]
        ab=[1 if np.real(vecs[k]*np.conj(ref))<0 else 0 for k in range(n)]
        d8=''.join(map(str,dh(ab)[:8]))
        frames.append((idx,ab,d8,fc))
    print(f"command frames: {len(frames)}")
    print(f"{'seg':>4}  {'abs':16s}  diffhead8  fc")
    for idx,ab,d8,fc in frames:
        print(f"{idx:4d}  {''.join(map(str,ab))}  {d8}   {fc/1e6:.5f}M")
    # distinct diff heads
    from collections import Counter, defaultdict
    c=Counter(f[2] for f in frames)
    print("\ndiff-head classes (pi-immune):")
    for h,n in c.most_common():
        segs_h=[f[0] for f in frames if f[2]==h]
        print(f"  {h}  x{n}   segs={segs_h}")
    # burst structure: group consecutive command segments (gap<=2)
    print("\nbursts (consecutive command segments, gap<=2):")
    bursts=[]
    cur=[frames[0]]
    for f in frames[1:]:
        if f[0]-cur[-1][0] <= 2:
            cur.append(f)
        else:
            bursts.append(cur); cur=[f]
    bursts.append(cur)
    for b in bursts:
        heads=[f[2] for f in b]
        uniq=sorted(set(heads))
        homog = "HOMOGENEOUS" if len(uniq)==1 else "mixed"
        print(f"  segs {b[0][0]}..{b[-1][0]}  n={len(b)}  {homog}  heads={uniq}")

if __name__=="__main__":
    main()
