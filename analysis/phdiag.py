#!/usr/bin/env python3
"""Diagnostic: within-burst phase structure.
For one command burst, after score-locking fc, plot per-symbol UNWRAPPED
phase, then the phase MINUS a linear fit. If clean BPSK with a constant
freq offset, the residual should snap to a 2-level staircase (0/pi). If the
residual is a spiral/monotonic, the carrier drifts WITHIN the burst and a
single fc cannot demod it -> need per-symbol tracking.
Also print the per-symbol coherent magnitude.
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

def diag(tag, t, v, n_sym=16):
    s1,fc=search(t,v,n_sym,0.955e6,1.045e6,500.0)
    s2,fc=search(t,v,n_sym,fc-1e3,fc+1e3,50.0)
    vecs=symvecs(t,v,fc,n_sym)
    ph=np.unwrap(np.angle(vecs))
    m=np.abs(vecs)
    x=np.arange(n_sym)
    A=np.vstack([x,np.ones(n_sym)]).T
    (slope,icept),*_=np.linalg.lstsq(A,ph,rcond=None)
    resid=ph-(slope*x+icept)
    print(f"--- {tag}  fc={fc/1e6:.5f} score={s2:.3f}")
    print(f"  lin slope = {slope:8.3f} deg/sym  (=> {slope/(2*np.pi)* (t[-1]-t[0])*1e6/n_sym/1e6:+.4f} kHz drift)")
    print("  sym  phase(deg)  resid(deg)  |m|/max")
    mm=m.max() or 1
    for i in range(n_sym):
        print(f"  s{i:2d}  {ph[i]:+9.1f}  {resid[i]:+10.1f}   {m[i]/mm:5.2f}")
    # how 2-level is the residual?
    r2=resid-np.median(resid)
    bim=float(np.mean(np.abs(np.cos(r2))))
    print(f"  residual bimodality (2-levelness) = {bim:.3f}")

def main():
    d=sys.argv[1]
    segs=burst_segments(d)
    durs=[(t[-1]-t[0])*1e6 for _,t,v,fs in segs]
    cmd=[s for s,du in zip(segs,durs) if du>=60]
    for i in (7,8,14,15):
        diag(f"cmd[{i}]", cmd[i][1], cmd[i][2], 16)
        print()

if __name__=="__main__":
    main()
