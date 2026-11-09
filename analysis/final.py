#!/usr/bin/env python3
"""Final consolidation: confidence-filtered timeline of the two command classes.
Per frame: coarse+fine+maxmin fc lock, bimodality (SNR proxy), 16-bit diff head.
Classify into A (diff head ~00111011xxxx) or B (~01000110xxxx) allowing a 1-bit
trailing tolerance (SNR noise on the last head bit); flag low-confidence frames
(bim<0.92 or |fc-1MHz|>20kHz) as bad locks. Print burst-grouped timeline and the
A/B adjacency structure to test the 'paired transmission' hypothesis.
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
    if m.max()==0: return 0.0
    k=int(np.argmax(m)); rel=np.angle(vecs*np.conj(vecs[k]))
    return float(np.mean(np.abs(np.cos(2*rel))))

def lock(t,v,n_sym):
    best=(-1.0,None)
    for fc in np.arange(0.955e6,1.045e6,500.0):
        b=metrics(symvecs(t,v,fc,n_sym))
        if b>best[0]: best=(b,fc)
    for fc in np.arange(best[1]-1e3,best[1]+1e3,50.0):
        b=metrics(symvecs(t,v,fc,n_sym))
        if b>best[0]: best=(b,fc)
    fc=best[1]; best_mm=(-1.0,fc)
    for fc2 in np.arange(fc-500,fc+500,10.0):
        mm=np.abs(symvecs(t,v,fc2,n_sym)).min()
        if mm>best_mm[0]: best_mm=(mm,fc2)
    return best_mm[1],symvecs(t,v,best[1],n_sym)

def dh(ab):
    return [0]+[ab[k]^ab[k-1] for k in range(1,len(ab))]

A0='00111011'; B0='01000110'
def ham(x,y): return sum(a!=b for a,b in zip(x,y))

def classify(d8):
    if ham(d8,A0)<=1: return 'A'
    if ham(d8,B0)<=1: return 'B'
    return '.'

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
        d16=''.join(map(str,dh(ab)))
        bim=metrics(vecs)
        ok = bim>=0.92 and abs(fc-1.0e6)<20e3
        frames.append(dict(seg=idx,ab=ab,d16=d16,bim=bim,fc=fc,ok=ok,cls=classify(d16[:8])))
    print(f"{'seg':>4} {'cls':>3} {'conf':>4}  {'diff16':16s}  bim   fc")
    for f in frames:
        tag='ok ' if f['ok'] else 'BAD'
        print(f"{f['seg']:4d} {f['cls']:>3} {tag:>4}  {f['d16']:16s} {f['bim']:.3f} {f['fc']/1e6:.5f}M")
    # burst grouping on all command segs (gap<=2)
    bursts=[]; cur=[frames[0]]
    for f in frames[1:]:
        if f['seg']-cur[-1]['seg']<=2: cur.append(f)
        else: bursts.append(cur); cur=[f]
    bursts.append(cur)
    print("\nbursts (command activity runs):")
    for b in bursts:
        line=''.join(f['cls'] if f['ok'] else '?' for f in b)
        okline=''.join(f['cls'] for f in b if f['ok'])
        print(f"  segs {b[0]['seg']:3d}..{b[-1]['seg']:3d}  n={len(b)}  seq(with bad=?): {line}   clean: {okline}")
    # adjacency A<->B within bursts (clean frames only)
    pairs=0; tot=0; alt=0
    for b in bursts:
        cl=[f['cls'] for f in b if f['ok']]
        for i in range(1,len(cl)):
            tot+=1
            if cl[i]!=cl[i-1]: alt+=1
    print(f"\nclean adjacent transitions: {alt}/{tot} are A<->B (i.e. class changes)")

if __name__=="__main__":
    main()
