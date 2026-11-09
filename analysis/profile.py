#!/usr/bin/env python3
"""Per-symbol magnitude + energy profile for the two classes.
Stacks all class frames (differentially aligned), integrates at the locked fc,
splits into K symbol windows, and plots/prints mean |.| per symbol window.
A sharp energy drop at some K => true frame length K (tail is noise).
A flat strong profile through K=16 => 16-symbol frame; a drop at K=8 => 8-bit.
"""
import sys
import numpy as np
from fsk import burst_segments
from final import lock, dh, classify

def get(d, cls):
    segs=burst_segments(d)
    out=[]
    for idx,(_,t,v,fs) in enumerate(segs):
        if (t[-1]-t[0])*1e6<60: continue
        fc,vecs=lock(t,v,16)
        rk=int(np.argmax(np.abs(vecs))); ref=vecs[rk]
        bim=float(np.mean(np.abs(np.cos(2*np.angle(vecs*np.conj(ref))))))
        if bim<0.92 or abs(fc-1e6)>=20e3: continue
        ab=[1 if np.real(vecs[k]*np.conj(ref))<0 else 0 for k in range(16)]
        if classify(''.join(map(str,dh(ab)[:8])))!=cls: continue
        out.append((idx,t,v,fc,bim))
    return out

def profile(d, cls, K=16):
    grp=get(d,cls)
    if not grp: print(cls,'none'); return
    grp.sort(key=lambda g:-g[4])
    t0=grp[0]
    dur=(t0[1][-1]-t0[1][0])
    # align each frame to the same duration by re-locking its own burst
    means=np.zeros(K)
    n=0
    for idx,t,v,fc,bim in grp:
        T=t-t[0]
        if T[-1]<dur*0.9: continue  # mismatched burst length
        c=np.cumsum((v-v.mean())*np.exp(-2j*np.pi*fc*T))
        edges=np.arange(K+1)*(T[-1]/K)
        idxs=np.clip(np.searchsorted(T,edges),0,len(T)-1)
        vecs=c[idxs[1:]]-c[idxs[:-1]]
        means+=np.abs(vecs); n+=1
    means/=max(n,1)
    peak=means.max()
    print(f"\nCLASS {cls}  (n={n})  symbol magnitude /peak:")
    for k in range(K):
        bar='#'*int(20*means[k]/peak)
        print(f"  s{k:2d} {means[k]/peak:5.3f} {bar}")
    # threshold scan: energy ratio tail/head
    for Kt in (8,10,12,16):
        head=means[:Kt].mean(); tail=means[Kt:].mean() if Kt<K else 0
        print(f"  K={Kt}: head_avg={head/peak:.3f} tail_avg={tail/peak:.3f}")

if __name__=='__main__':
    d=sys.argv[1]
    for cls in 'AB':
        profile(d,cls)
