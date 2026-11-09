#!/usr/bin/env python3
"""Decode all segments of a multi-burst capture in order and reconstruct the bit stream.

Each segment file = one ~50us burst with a re-centered (relative) time axis.
We demod each to BPSK at a fixed symbol rate and concatenate in segment order.
"""
import sys, os, glob, re
import numpy as np
from scipy import signal as sps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_csv(path):
    t, v = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 2: continue
            a, b = line.split(",")
            t.append(float(a)); v.append(float(b))
    return np.array(t), np.array(v)

def find_burst(t, v):
    z = sps.hilbert(v - np.mean(v))
    env = np.abs(z)
    thr = env.max()*0.15
    st = np.where(env > thr)[0]
    if len(st)==0: return None
    gr = np.split(st, np.where(np.diff(st) > 20)[0]+1)
    g = max((gg for gg in gr if len(gg)>10), key=len)
    return g[0], g[-1]

def decode_one(tb, vb, fs, sym):
    w = tb - tb[0]
    spec = np.abs(np.fft.rfft(vb*np.hanning(len(vb))))
    freqs = np.fft.rfftfreq(len(vb), 1/fs)
    m = (freqs>0.8e6)&(freqs<1.2e6)
    carr = freqs[m][np.argmax(spec[m])]
    iq = vb*np.exp(-1j*2*np.pi*carr*w)
    I = sps.filtfilt(*sps.butter(4, 250e3/(fs/2)), np.real(iq))
    dur = tb[-1]-tb[0]
    n_sym = int(round(dur*sym))
    # center-align: sample from middle
    ts = tb[0] + dur/2 - (n_sym-1)/(2*sym) + np.arange(n_sym)/sym
    ts = ts[(ts>=tb[0])&(ts<=tb[-1])]
    Is = np.interp(ts, tb, I)
    thr2 = (Is.max()+Is.min())/2
    hi=Is[Is>thr2]; lo=Is[Is<=thr2]
    if len(hi)<1 or len(lo)<1: return None, None
    bits=(Is>(hi.mean()+lo.mean())/2).astype(int)
    return bits, carr

def main():
    d = sys.argv[1]
    sym = float(sys.argv[2]) if len(sys.argv)>2 else 125e3
    files = sorted(glob.glob(os.path.join(REPO,d,'*.csv')),
                   key=lambda p:int(re.search(r'segment-(\d+)_',p).group(1)))
    allbits=[]; carrs=[]; lens=[]
    for p in files:
        t,v=load_csv(p); dt=np.median(np.diff(t)); fs=1/dt; v=v-np.mean(v)
        fb=find_burst(t,v)
        if fb is None: continue
        a,b=fb
        bits,carr=decode_one(t[a:b+1],v[a:b+1],fs,sym)
        if bits is None: continue
        allbits.append(bits); carrs.append(carr); lens.append(len(bits))
    carrs=np.array(carrs); lens=np.array(lens)
    print(f"segments decoded: {len(allbits)}")
    print(f"carrier: mean={carrs.mean()/1e6:.4f}MHz min={carrs.min()/1e6:.4f} max={carrs.max()/1e6:.4f}")
    print(f"bits/burst: min={lens.min()} max={lens.max()} mean={lens.mean():.2f}")
    from collections import Counter
    print("bit-length dist:",dict(Counter(lens)))
    # use the modal length; pad/truncate to it for clean framing
    mode = Counter(lens).most_common(1)[0][0]
    print(f"modal bits/burst = {mode}")
    stream=np.concatenate([ (b if len(b)==mode else (b[:mode] if len(b)>mode else np.pad(b,(0,mode-len(b)))) ) for b in allbits])
    print(f"total bits: {len(stream)}  -> {len(stream)/8:.1f} bytes")
    # print as bytes both orders
    def tobytes(bits, msb):
        out=[]
        for i in range(0,len(bits)-7,8):
            ch=bits[i:i+8]
            out.append(int(''.join(map(str,ch if msb else ch[::-1])),2))
        return out
    print("bytes MSB-first:", ' '.join(f'{x:02x}' for x in tobytes(stream,True)))
    print("bytes LSB-first:", ' '.join(f'{x:02x}' for x in tobytes(stream,False)))
    # also show first 200 raw bits
    print("raw bits[:200]:", ''.join(map(str,stream[:200])))

if __name__=='__main__':
    main()
