#!/usr/bin/env python3
"""Envelope on/off pattern: the spectrogram showed the carrier is GATED
(on ~2us, off ~2us). Map the envelope to 0/1 at fine resolution and print
the binary pattern for idle and command bursts. Also try to find a stable
clock by autocorrelating the on/off pattern over all command bursts.
"""
import sys
import numpy as np
from scipy.signal import hilbert
from fsk import burst_segments

def env_pattern(t, v, fs, thr_frac=0.45):
    v = v - v.mean()
    env = np.abs(hilbert(v))
    # median of the "on" level for threshold
    on = env[env > env.mean()]
    thr = np.median(on)*thr_frac
    bits = (env > thr).astype(int)
    return env, thr, bits

def main():
    d = sys.argv[1]
    segs = burst_segments(d)
    durs = [(t[-1]-t[0])*1e6 for _, t, v, fs in segs]
    idle = [s for s, du in zip(segs, durs) if du < 60]
    cmd  = [s for s, du in zip(segs, durs) if du >= 60]

    for tag, sel in (("IDLE", idle[100]), ("CMD", cmd[10]), ("CMD2", cmd[5])):
        _, t, v, fs = sel
        env, thr, bits = env_pattern(t, v, fs)
        dur = (t[-1]-t[0])*1e6
        # print pattern at 1us resolution, each char = 0.5us (sampled)
        step = max(1, int(0.5e-6*fs))
        s = ''.join(str(bits[i]) for i in range(0, len(bits), step))
        print(f"\n=== {tag} dur={dur:.2f}us thr={thr:.3f} on={bits.mean():.2f} ===")
        print(s)
        # run-lengths
        rl = []
        cur = bits[0]; n = 0
        for b in bits:
            if b == cur: n += 1
            else:
                rl.append((cur, n/step*0.5)); cur = b; n = 1
        rl.append((cur, n/step*0.5))
        print("runs (bit,len_us):", " ".join(f"{'1' if b else '0'}{l:.1f}" for b,l in rl[:40]))

    # clock estimation: cross-correlate on/off bit patterns of all cmds (resampled to 1us)
    print("\n### on/off pattern agreement across command bursts (1us grid, trimmed)")
    N = 130
    grid = []
    for _, t, v, fs in cmd:
        env, thr, bits = env_pattern(t, v, fs)
        g = np.interp(np.linspace(0, N, 1300), np.linspace(0, (t[-1]-t[0])*1e6, len(bits)), bits)
        grid.append(g)
    G = np.vstack(grid)
    maj = (G.mean(axis=0) > 0.5).astype(int)
    s = ''.join(map(str, maj))
    print("majority on/off :", s)
    # mean agreement per position
    ag = G.mean(axis=0)
    print("agreement       :", "".join('.' if a<0.4 or a>0.6 else '?' for a in ag))

if __name__ == "__main__":
    main()
