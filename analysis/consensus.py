#!/usr/bin/env python3
"""Group-aligned consensus decode of the two command classes.

Per class (A diff-head 00111011..., B diff-head 01000110...):
  - each frame's absolute 16-bit word is decoded vs its own strongest symbol
    (pi-ambiguous: could be flipped)
  - align every frame to the highest-bim reference: if hamming(ab, ref) > 8
    flip ab (uses the differential, which is pi-immune)
  - stack all aligned frames and majority-vote -> consensus absolute word
  - report per-position agreement for the 16-bit absolute AND differential
Squeezes SNR out of the repeats, especially the weak tail bits s6..s15.
"""
import sys
import numpy as np
from fsk import burst_segments
from final import lock, dh, ham, classify

def decode_frames(d):
    segs=burst_segments(d)
    frames=[]
    for idx,(_,t,v,fs) in enumerate(segs):
        dur=(t[-1]-t[0])*1e6
        if dur<60: continue
        fc,vecs=lock(t,v,16)
        rk=int(np.argmax(np.abs(vecs))); ref=vecs[rk]
        bim=float(np.mean(np.abs(np.cos(2*np.angle(vecs*np.conj(ref))))))
        if bim<0.92 or abs(fc-1.0e6)>=20e3: continue
        n=len(vecs)
        ab=[1 if np.real(vecs[k]*np.conj(ref))<0 else 0 for k in range(n)]
        cls=classify(''.join(map(str,dh(ab)[:8])))
        if cls=='.': continue
        frames.append((idx,ab,cls,bim))
    return frames

def consensus(frames, cls):
    grp=[f for f in frames if f[2]==cls]
    if not grp: return None
    grp=sorted(grp,key=lambda f:-f[3])
    ref=grp[0][1]
    aligned=[]
    for _,ab,_,_ in grp:
        a=ab[:]
        if ham(a,ref)>8: a=[b^1 for b in a]
        aligned.append(a)
    ab_con=[int(np.mean([row[k] for row in aligned])>0.5) for k in range(16)]
    dstack=[dh(r) for r in aligned]
    d_con=[int(np.mean([row[k] for row in dstack])>0.5) for k in range(16)]
    ab_agr=[float(np.mean([row[k] for row in aligned])==ab_con[k]) for k in range(16)]
    d_agr=[float(np.mean([row[k] for row in dstack])==d_con[k]) for k in range(16)]
    return dict(cls=cls,N=len(grp),ab_con=ab_con,ab_agr=ab_agr,d_con=d_con,d_agr=d_agr,
                segs=[f[0] for f in grp])

def fmt16(bits):
    return ' '.join(''.join(map(str,bits[i:i+4])) for i in range(0,16,4))

def bits_to_hex(bits):
    v=0
    for b in bits: v=(v<<1)|b
    return f"{v:0{len(bits)//4}x}"

def main(d):
    frames=decode_frames(d)
    for cls in 'AB':
        c=consensus(frames,cls)
        if not c:
            print(f"class {cls}: no frames"); continue
        print(f"\n=== CLASS {c['cls']}  (N={c['N']} repeats, segs={c['segs']}) ===")
        print("abs consensus :", fmt16(c['ab_con']), " hex=", bits_to_hex(c['ab_con']))
        print("abs agree%    :", ' '.join(f"{a*100:3.0f}" for a in c['ab_agr']))
        print("diff consensus:", fmt16(c['d_con']))
        print("diff agree%   :", ' '.join(f"{a*100:3.0f}" for a in c['d_agr']))

if __name__=="__main__":
    main(sys.argv[1])
