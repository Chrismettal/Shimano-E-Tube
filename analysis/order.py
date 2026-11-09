#!/usr/bin/env python3
"""Temporal ordering of the two command-frame classes.
Decode each command frame (locked fc, abs bits vs strongest symbol), assign
it to class A (head 110100) or B (head 100001) or ?, print the sequence in
time order with inter-arrival gaps. Also prints idle-frame positions for
context. Alternating A/B => pair protocol; runs => distinct commands.
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

def main():
    d=sys.argv[1]
    segs=burst_segments(d)  # already in segment (time) order
    events=[]  # (idx, type, label, ab)
    for idx,(_,t,v,fs) in enumerate(segs):
        dur=(t[-1]-t[0])*1e6
        if dur<60:
            events.append((idx,'I','',''))
        else:
            fc,vecs=lock(t,v,16)
            n=len(vecs); rk=int(np.argmax(np.abs(vecs))); ref=vecs[rk]
            ab=''.join('1' if np.real(vecs[k]*np.conj(ref))<0 else '0' for k in range(n))
            head=ab[:6]
            lab = 'A' if head=='110100' else ('B' if head=='100001' else '?')
            events.append((idx,lab,ab,fc))
    # segs is already index-sorted; keep that order
    print("time-order sequence (I=idle, A=head 110100, B=head 100001, ?=other):")
    prev=None
    for idx,lab,ab,fc in events:
        gap = '' if prev is None else f"(dseg={idx-prev:+d})"
        prev=idx
        extra = f"  {ab}" if lab!='I' else ''
        fcs = '' if lab=='I' else f"  fc={fc/1e6:.5f}M"
        print(f"  seg={idx:4d} {lab}{gap}{extra}{fcs}")
    seq=''.join(lab for _,lab,_,_ in events)
    print("\ncompact:", seq)
    ab_only=[lab for _,lab,_,_ in events if lab in 'AB']
    alt = sum(1 for i in range(1,len(ab_only)) if ab_only[i]!=ab_only[i-1])
    print(f"command frames: {len(ab_only)}  alternating-adjacencies: {alt}/{max(len(ab_only)-1,1)}")

if __name__=="__main__":
    main()
