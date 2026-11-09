#!/usr/bin/env python3
"""Compare two command-frame classes with a FIXED reference (symbol 0).
Locks carrier per burst (score-driven), then prints each symbol's I/Q +
phase relative to symbol 0 (fixed, not max-mag). Reveals whether class A
and class B are the same frame rotated by a fixed angle (=> pi ambiguity
or ref rotation) or genuinely different sequences.
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
    k = int(np.argmax(m)); rel = np.angle(vecs*np.conj(vecs[k]))
    return float(np.mean(np.abs(np.cos(2*rel))))

def search(t, v, n_sym, lo, hi, step):
    best=(-1.0,None)
    for fc in np.arange(lo,hi,step):
        s=score_of(symvecs(t,v,fc,n_sym))
        if s>best[0]: best=(s,fc)
    return best

def show(tag, t, v, n_sym, ref=0):
    s1,fc=search(t,v,n_sym,0.955e6,1.045e6,500.0)
    s2,fc=search(t,v,n_sym,fc-1e3,fc+1e3,50.0)
    vecs=symvecs(t,v,fc,n_sym)
    m=np.abs(vecs); mm=m.max() or 1
    base=np.angle(vecs[ref])
    print(f"--- {tag}  fc={fc/1e6:.5f} score={s2:.3f} ref=sym{ref} (base phase={np.degrees(base):.1f})")
    print("  sym   I      Q     |m|/max   relphase   bit")
    for i in range(n_sym):
        ri = vecs[i]*np.exp(-1j*base)
        ang=np.degrees(np.angle(ri))
        bit = 1 if abs(ang)>90 else 0
        print(f"  s{i:2d} {ri.real:+.3f} {ri.imag:+.3f}  {m[i]/mm:5.2f}   {ang:+7.1f}    {bit}")
    bitstr=''.join(str(1 if abs(np.degrees(np.angle(vecs[i]*np.exp(-1j*base))))>90 else 0) for i in range(n_sym))
    print(f"  bits (ref sym{ref}): {bitstr}")

def main():
    d=sys.argv[1]
    segs=burst_segments(d)
    durs=[(t[-1]-t[0])*1e6 for _,t,v,fs in segs]
    cmd=[s for s,du in zip(segs,durs) if du>=60]
    # class A head 110100 -> frame 7 (s=0.998); class B head 100001 -> frame 8
    show("CLASS A (f7)", cmd[7][1], cmd[7][2], 16, ref=0)
    print()
    show("CLASS B (f8)", cmd[8][1], cmd[8][2], 16, ref=0)
    print()
    # also frame 7 with ref=2 (a high-mag symbol) to see the rotation
    show("CLASS A (f7) ref=2", cmd[7][1], cmd[7][2], 16, ref=2)

if __name__=="__main__":
    main()
