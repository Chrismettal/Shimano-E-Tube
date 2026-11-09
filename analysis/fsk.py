#!/usr/bin/env python3
"""Robust 2-tone FSK demodulator for the Shimano E-Tube wireless link.

Approach (phase-independent, matched-filter per symbol):
  - build analytic signal
  - for a candidate (bit_rate, timing_offset, tone->bit map), split the burst
    into symbol windows
  - per symbol compute complex correlation with each of the two tones,
    take |.| as energy, assign the bit to the dominant tone
  - confidence = |e1-e2|/(e1+e2)  (how decisive the symbol is)
  - a good rate/offset maximizes mean confidence AND, for idle segments,
    reproduces the repeating 000011 idle pattern

Calibration: measure the two tones precisely, then use idle segments to pin
down the exact symbol rate, the timing phase, and the tone->bit mapping.
Then decode the command (long) segments with the calibrated params.
"""
import sys, os, glob, re
import numpy as np
from scipy import signal as sps, optimize as spo

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F1_DEFAULT = 1.025311e6
F2_DEFAULT = 1.174834e6

def load_csv(path):
    t, v = [], []
    with open(path) as f:
        for line in f:
            try:
                a, b = line.split(",")
                t.append(float(a)); v.append(float(b))
            except ValueError:
                continue
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

def segnum(p):
    m = re.search(r'segment-(\d+)_', p)
    if m: return int(m.group(1))
    m = re.search(r'segment(\d+)', p)
    if m: return int(m.group(1))
    m = re.search(r'scope_(\d+)', p)
    if m: return int(m.group(1))
    return 0

def burst_segments(d):
    """Yield (basename, t, v, fs) for every CSV in a capture dir, in segment order."""
    files = sorted(glob.glob(os.path.join(REPO, d, "*.csv")), key=segnum)
    out = []
    for p in files:
        t, v = load_csv(p)
        dt = np.median(np.diff(t)); fs = 1/dt
        fb = find_burst(t, v - v.mean())
        if fb is None:
            continue
        a, b = fb
        out.append((os.path.basename(p), t[a:b+1], v[a:b+1] - v.mean(), fs))
    return out

def measure_tones(t, v, fs, f_lo=0.95e6, f_hi=1.30e6):
    """Return (f1, f2): the two FSK tones via instantaneous-frequency histogram.

    For FSK the instantaneous frequency toggles between the two mark/space
    frequencies. Histogram those values and take the two dominant clusters.
    """
    z = sps.hilbert(v)
    ph = np.angle(z)
    dph = np.unwrap(ph)
    # instantaneous angular frequency (rad/s), smoothed to reduce noise
    win = max(3, int(fs/2.0e6))  # ~0.5us
    ifo = sps.savgol_filter(np.diff(dph)*fs/(2*np.pi), win if win%2==1 else win+1, 3)
    m = (ifo > f_lo) & (ifo < f_hi)
    hist, bin_edges = np.histogram(ifo[m], bins=240, range=(f_lo, f_hi))
    centers = (bin_edges[:-1]+bin_edges[1:])/2
    # find two well-separated peaks
    order = np.argsort(hist)[::-1]
    peaks = []
    for i in order:
        if all(abs(centers[i]-p) > 80e3 for p in peaks):
            peaks.append(centers[i])
        if len(peaks) == 2:
            break
    if len(peaks) < 2:
        # fall back: strongest + next strongest bin centroid
        peaks = [centers[order[0]], centers[order[1]]]
    # weighted centroid around each peak (within +-60kHz)
    refined = []
    for pk in peaks:
        w = (np.abs(centers-pk) < 60e3)
        refined.append(np.sum(centers[w]*hist[w])/np.sum(hist[w]))
    refined = sorted(refined)
    return refined[0], refined[1]

def demod(t, v, fs, f1, f2, rate, off, map1, map2):
    """Demodulate burst to bits via per-symbol matched filter (vectorized).
    map1/map2 = bit value for f1/f2. Returns bits, conf."""
    T = 1.0/rate
    dur = t[-1]-t[0]
    n_sym = int(np.floor(dur/T + 1e-9))
    phi1 = np.exp(-2j*np.pi*f1*t)
    phi2 = np.exp(-2j*np.pi*f2*t)
    c1 = np.cumsum(v*phi1)
    c2 = np.cumsum(v*phi2)
    # window edges in sample index; clip so c[idx] stays in bounds
    edges = t[0] + off + np.arange(n_sym+1)*T
    idx = np.searchsorted(t, edges)
    idx = np.clip(idx, 0, len(v)-1)
    e1 = np.abs(c1[idx[1:]] - c1[idx[:-1]])
    e2 = np.abs(c2[idx[1:]] - c2[idx[:-1]])
    # drop any degenerate zero-width windows
    nz = (idx[1:] > idx[:-1])
    return np.where(e1[nz] > e2[nz], map1, map2), np.abs(e1[nz]-e2[nz])/(e1[nz]+e2[nz]+1e-12)

def idle_score(bits):
    """Fraction of adjacent-pair matches with the repeating 000011 pattern (any phase, both polarities)."""
    if len(bits) < 6: return 0.0, 0
    best = 0.0; best_rot = 0
    for pol in (0, 1):
        pat = np.array([0,0,0,1,1,0])
        if pol: pat = 1-pat
        for rot in range(6):
            p = np.roll(pat, rot)
            rep = np.tile(p, len(bits)//6+1)[:len(bits)]
            match = (bits == rep).mean()
            if match > best:
                best = match; best_rot = rot*6+pol
    return best, best_rot

def score_tones(segs, f1, f2, rate, off, map1, map2):
    """Combined score: pattern-match on idle + mean confidence (any seg). Higher=better."""
    tot_match=0.0; tot_conf=0.0; tot_n=0; n=0
    for _, t, v, fs in segs:
        bits, conf = demod(t, v, fs, f1, f2, rate, off, map1, map2)
        if len(bits) < 4 or len(conf)==0: continue
        if n < 3:  # first few are the idle ones in the list
            m, _ = idle_score(bits)
            tot_match += m*len(bits)
        tot_conf += conf.mean()*len(bits); tot_n += len(bits)
        n += 1
    if tot_n == 0: return -1.0
    return 0.5*(tot_conf/tot_n) + 0.5*(tot_match/tot_n if n>3 else tot_conf/tot_n)

def optimize_tones(idle_segs, f1_0, f2_0, rate, off, map1, map2):
    """Nelder-Mead refine (f1,f2) to maximize idle decision confidence + pattern match."""
    def neg(p):
        return -score_tones(idle_segs, p[0], p[1], rate, off, map1, map2)
    best = neg([f1_0, f2_0])
    # coarse grid refine in +-30kHz around each initial estimate (independent axes)
    for df1 in np.arange(-30e3, 30e3+1, 5e3):
        for df2 in np.arange(-30e3, 30e3+1, 5e3):
            c = neg([f1_0+df1, f2_0+df2])
            if c < best: best, p0 = c, (f1_0+df1, f2_0+df2)
    res = spo.minimize(neg, np.array(p0), method='Nelder-Mead',
                                options={'maxiter': 400, 'xatol': 200, 'fatol': 1e-5})
    if -res.fun > best:
        f1, f2 = float(res.x[0]), float(res.x[1])
    else:
        f1, f2 = p0
    return f1, f2

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    print(f"segments: {len(segs)}")
    # ---- 1) measure tones from the longest (cleanest) segment ----
    lens = [len(s[2]) for s in segs]
    li = int(np.argmax(lens))
    _, tl, vl, fl = segs[li]
    f1, f2 = measure_tones(tl, vl, fl)
    print(f"tones: f1={f1/1e6:.6f}MHz  f2={f2/1e6:.6f}MHz  (diff={(f2-f1)/1e3:.2f}kHz)")

    # ---- 2) classify idle vs command by duration ----
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    durs = np.array(durs)
    print(f"duration(us): min={durs.min():.1f} med={np.median(durs):.1f} max={durs.max():.1f}")
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]
    print(f"idle: {len(idle)}  command: {len(cmd)}")

    # ---- 3) calibrate rate/offset/mapping on idle segments (if any) ----
    if len(idle) < 3:
        print("WARNING: <3 idle segments; cannot calibrate. Using defaults 125kbit/s.")
        rate, off, map1, map2 = 125e3, 0.0, 0, 1
        print("CALIBRATED (default): rate=125.0kbit/s offset=0.000us f1->0 f2->1")
    else:
        best = None
        for rate in [x*1e3 for x in [100,110,120,125,128,130,140,150,160,180,200]]:
            for pol in (0,1):
                map1 = 0 if not pol else 1
                map2 = 1 if not pol else 0
                for off in np.arange(0, 1/rate, 1/(8*rate)):
                    tot_match=0; tot_conf=0; tot_n=0
                    for _, t, v, fs in idle[:30]:
                        bits, conf = demod(t, v, fs, f1, f2, rate, off, map1, map2)
                        if len(bits) < 4: continue
                        m, _ = idle_score(bits)
                        tot_match += m*len(bits); tot_conf += conf.mean()*len(bits); tot_n += len(bits)
                    if tot_n == 0: continue
                    score = (tot_match/tot_n)*0.5 + (tot_conf/tot_n)*0.5
                    if best is None or score > best[0]:
                        best = (score, rate, off, map1, map2, tot_match/tot_n, tot_conf/tot_n)
        score, rate, off, map1, map2, m, c = best
        print(f"\nCALIBRATED: rate={rate/1e3:.1f}kbit/s  offset={off*1e6:.3f}us  "
              f"f1->{map1} f2->{map2}  idle_pattern_match={m:.3f} mean_conf={c:.3f}")
        # ---- 3b) refine the two tones to maximize idle decision confidence ----
        f1, f2 = optimize_tones(idle[:10], f1, f2, rate, off, map1, map2)
        tm = tc = tn = 0
        for _, t, v, fs in idle[:30]:
            bits, conf = demod(t, v, fs, f1, f2, rate, off, map1, map2)
            if len(bits) < 4: continue
            mm, _ = idle_score(bits); tm += mm*len(bits); tc += conf.mean()*len(bits); tn += len(bits)
        print(f"TONES-REFINED: f1={f1/1e6:.6f}MHz  f2={f2/1e6:.6f}MHz  diff={(f2-f1)/1e3:.2f}kHz  "
              f"idle_match={tm/tn:.3f}  mean_conf={tc/tn:.3f}")

    # ---- 4) decode command segments with calibrated params ----
    print("\n=== COMMAND BURSTS ===")
    results = []
    for i, (_, t, v, fs) in enumerate(cmd):
        bits, conf = demod(t, v, fs, f1, f2, rate, off, map1, map2)
        dur = (t[-1]-t[0])*1e6
        results.append((i, os.path.basename(_), dur, bits, conf))
        print(f"[{i:2d}] {os.path.basename(_):24s} {dur:5.1f}us "
              f"n={len(bits)} conf={conf.mean():.2f} bits={''.join(map(str,bits))}")
    # byte view of each
    print("\n=== COMMAND BURSTS as hex (MSB-first) ===")
    for i, name, dur, bits, conf in results:
        if len(bits) % 8:
            hb = bits[:len(bits)-len(bits)%8]
        else:
            hb = bits
        hx = ' '.join(f'{int("".join(map(str,hb[j:j+8])),2):02x}' for j in range(0,len(hb),8))
        print(f"[{i:2d}] {name:24s} {hx}")

if __name__ == "__main__":
    main()
