#!/usr/bin/env python3
"""Dump baseband I/Q trace of a command burst at fine resolution.
bb = v0 * exp(-j2pi fc t) with locked fc. For clean BPSK, Q~0 and I is a
+/- square wave; symbol flips are I sign changes. Prints a coarse time
trace of I (and Q) so we can SEE the symbols and count flips by eye.
"""
import sys
import numpy as np
from fsk import burst_segments

def lock(t,v,n_sym=16):
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
    for fi in (7,14):
        _,t,v,fs=cmd[fi]
        fc,_=lock(t,v)
        T=t-t[0]; v0=v-v.mean()
        bb=v0*np.exp(-2j*np.pi*fc*T)
        I=bb.real; Q=bb.imag
        print(f"\ncmd[{fi}] fc={fc/1e6:.5f}  dur={T[-1]*1e6:.1f}us  "
              f"I: min={I.min():.2f} max={I.max():.2f}  Q: min={Q.min():.2f} max={Q.max():.2f}")
        # trace: 80 columns, char by I sign+magnitude
        N=80
        print("  time-> (I) '+'=+I '-'=-I '.'=small ; row2=Q")
        rI=''; rQ=''
        for k in range(N):
            a=int(k*len(I)/N); b=int((k+1)*len(I)/N)
            ia=np.mean(I[a:b]); qa=np.mean(Q[a:b])
            rI += '+' if ia>0.15 else ('-' if ia<-0.15 else '.')
            rQ += '^' if qa>0.15 else ('v' if qa<-0.15 else '.')
        print("  I: "+rI)
        print("  Q: "+rQ)
        # count sign flips of I with min magnitude
        flips=0; last=None
        for val in I:
            s = 1 if val>0.15 else (-1 if val<-0.15 else 0)
            if s!=0 and last is not None and s!=last: flips+=1
            if s!=0: last=s
        print(f"  I sign flips (mag>0.15): {flips}")

if __name__=="__main__":
    main()
