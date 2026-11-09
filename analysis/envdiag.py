#!/usr/bin/env python3
"""Envelope+phase decomposition of a command burst.
Analytic signal (Hilbert) -> |a(t)| envelope and angle. Sampled at each of
the 16 symbol centers AND as a fine time trace. Determines whether the
carrier is gated (ASK/OOK) or constant-envelope BPSK with the magnitude
swings coming from a wrong frame division.
Also tries a few candidate symbol periods to see which equalizes |v_k|.
"""
import sys
import numpy as np
from scipy.signal import hilbert
from fsk import burst_segments

def search_fc(t, v, fc0, n_sym=16):
    # score-driven lock: coarse then fine near fc0
    def score(fc):
        T=t-t[0]; v0=v-v.mean()
        c=np.cumsum(v0*np.exp(-2j*np.pi*fc*T))
        edges=np.arange(n_sym+1)*(T[-1]/n_sym)
        idx=np.clip(np.searchsorted(T,edges),0,len(v0)-1)
        vecs=c[idx[1:]]-c[idx[:-1]]
        m=np.abs(vecs)
        if m.max()==0: return 0.0
        k=int(np.argmax(m)); rel=np.angle(vecs*np.conj(vecs[k]))
        return float(np.mean(np.abs(np.cos(2*rel))))
    lo,hi,step=fc0-1e4,fc0+1e4,250.0
    best=(-1,None)
    for fc in np.arange(lo,hi,step):
        s=score(fc)
        if s>best[0]: best=(s,fc)
    for fc in np.arange(best[1]-500,best[1]+500,10.0):
        s=score(fc)
        if s>best[0]: best=(s,fc)
    return best

def main():
    d=sys.argv[1]
    segs=burst_segments(d)
    durs=[(t[-1]-t[0])*1e6 for _,t,v,fs in segs]
    cmd=[s for s,du in zip(segs,durs) if du>=60]
    t,v,fs=cmd[7][1],cmd[7][2],cmd[7][3]
    dur=t[-1]-t[0]
    fc,_=search_fc(t,v,1.0e6)
    print(f"cmd[7] dur={dur*1e6:.2f}us fs={fs:.0f} fc={fc/1e6:.5f}")

    # analytic envelope
    v0=v-np.mean(v)
    a=hilbert(v0)
    env=np.abs(a); ang=np.angle(a)
    # downsample env trace to ~40 columns
    print("\nEnvelope time trace (|.|=low  * = hi):")
    N=60
    bins=np.array_split(env, N)
    row=''.join('#' if b.max()>0.5*env.max() else ('*' if b.max()>0.25*env.max() else '.') for b in bins)
    print("  "+row)
    print(f"  env max={env.max():.2f} min={env.min():.2f} mean={env.mean():.2f}")

    # per-symbol env at 16-way centers
    for nsym in (16,):
        centers=np.linspace(0,len(v0),nsym+1)
        print(f"\n16-way symbol env+phase:")
        print("  sym  env/ max   phase   I      Q")
        for k in range(nsym):
            c0=int(centers[k]); c1=int(centers[k+1])
            seg=a[c0:c1]
            m=env[c0:c1].mean()/env.max()
            # coherent phase
            T=t[c0:c1]-t[0]
            coh=np.sum(v0[c0:c1]*np.exp(-2j*np.pi*fc*T))
            print(f"  s{k:2d}   {m:5.2f}   {np.degrees(np.angle(coh)):7.1f}  {coh.real:+.2f} {coh.imag:+.2f}")

if __name__=="__main__":
    main()
