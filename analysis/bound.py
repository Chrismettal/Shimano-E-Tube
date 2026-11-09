#!/usr/bin/env python3
"""Direct symbol-boundary detection.
Demodulate with locked fc -> baseband I/Q. Track phase; a BPSK symbol flip
is a ~180deg jump. Find jump times, measure inter-jump intervals, and the
gaps from burst edges. This reveals the true symbol period and how many
symbols are in the burst (and any lead-in).
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
    for fi in (7,14):
        _,t,v,fs=cmd[fi]
        fc,_=lock(t,v,16)
        T=t-t[0]; v0=v-v.mean()
        bb=v0*np.exp(-2j*np.pi*fc*T)   # baseband, phase jumps on symbol flips
        # low-pass to remove carrier noise? bb is already baseband complex.
        # instantaneous phase, but with noise -> smooth I,Q with a short mean first
        w=8
        I=np.convolve(bb.real,np.ones(w)/w,mode='same')
        Q=np.convolve(bb.imag,np.ones(w)/w,mode='same')
        ph=np.angle(I+1j*Q)
        # detect sign flips of the real part of baseband (BPSK): x=bb.real
        x=np.convolve(bb.real,np.ones(3)/3,mode='same')
        # find where x crosses zero with sufficient magnitude on both sides
        jumps=[]
        for k in range(1,len(x)):
            if x[k-1]*x[k]<0 and abs(x[k-1])>0.15 and abs(x[k])>0.15:
                jumps.append(T[k])
        # merge jumps closer than 2us
        mj=[]
        for j in jumps:
            if mj and j-mj[-1]<2e-6: mj[-1]=j
            else: mj.append(j)
        iv=np.diff(mj)*1e6
        print(f"\ncmd[{fi}] fc={fc/1e6:.5f}  n_jumps={len(mj)}  dur={T[-1]*1e6:.2f}us")
        print("  jump times us:", np.round(mj*1e6,2))
        print("  inter-jump us:", np.round(iv,2))
        print("  first jump @ %.2fus  last @ %.2fus" % (mj[0]*1e6,mj[-1]*1e6))
        print("  min/max inter-jump: %.2f / %.2f us" % (iv.min(),iv.max()))

if __name__=="__main__":
    main()
