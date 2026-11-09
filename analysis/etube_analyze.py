#!/usr/bin/env python3
"""Shimano E-Tube scope capture analysis.

Pipeline:
  1. Load scope CSV (time, voltage).
  2. Hilbert transform -> analytic signal.
  3. Envelope -> burst detection (packet framing).
  4. Per burst: estimate carrier freq, examine phase jumps (BPSK?) vs amplitude (ASK/OOK?).
  5. Bit extraction at symbol rate, byte framing, hex dump.
"""
import sys, os
import numpy as np
from scipy import signal as sps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_csv(path):
    t, v = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            a, b = line.split(",")
            t.append(float(a)); v.append(float(b))
    return np.array(t), np.array(v)

def analyze(path, burst_idx=0, sym_rate=None, verbose=True):
    t, v = load_csv(path)
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    if verbose:
        print(f"== {os.path.basename(path)}")
        print(f"   n={len(t)} dt={dt*1e9:.3f} ns fs={fs/1e6:.3f} MHz dur={(t[-1]-t[0])*1e6:.1f} us")
        print(f"   vmin={v.min():.4f} vmax={v.max():.4f}")

    # DC removal (AC coupled scope, but residual DC possible)
    v = v - np.mean(v)

    # Spectrum: find dominant carrier
    n = len(v)
    spec = np.abs(np.fft.rfft(v * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    top = spec.argsort()[-6:][::-1]
    if verbose:
        print("   top spectral peaks (Hz, mag):")
        for i in top:
            print(f"     {freqs[i]/1e6:.4f} MHz  {spec[i]:.3f}")
    carrier = freqs[top[0]]

    # Analytic signal
    z = sps.hilbert(v)
    env = np.abs(z)
    phase = np.angle(z)

    # Burst detection on envelope
    thr = env.max() * 0.15
    bursts = sps.find_peaks(env, height=thr, distance=int(20e-6 / dt))[0]
    # merge close peaks into burst spans
    spans = []
    if len(bursts):
        start = np.where(env > thr)[0]
        # group contiguous
        groups = np.split(start, np.where(np.diff(start) > 20)[0] + 1)
        for g in groups:
            if len(g) > 10:
                spans.append((g[0], g[-1]))
    if verbose:
        print(f"   envelope max={env.max():.4f} bursts={len(spans)}")
        for i, (a, b) in enumerate(spans[:12]):
            print(f"     burst {i}: t={t[a]*1e6:.1f}..{t[b]*1e6:.1f} us dur={(t[b]-t[a])*1e6:.1f} us")
    if not spans:
        return None
    a, b = spans[burst_idx] if burst_idx < len(spans) else spans[0]
    tb, vb = t[a:b+1], v[a:b+1]
    zb = z[a:b+1]
    envb = env[a:b+1]

    # Carrier estimate within burst
    n = len(vb)
    spec = np.abs(np.fft.rfft(vb * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    c = freqs[spec.argmax()]
    if verbose:
        print(f"   burst carrier est: {c/1e6:.4f} MHz, dur={(tb[-1]-tb[0])*1e6:.2f} us, n={n}")

    # Demod options:
    # A) ASK/OOK: envelope low-pass -> symbol levels
    # B) BPSK: phase (unwrap) -> jumps of pi
    # C) QPSK: 4 phase states
    # Mix: multiply by exp(-j2*pi*c*t), lowpass -> I/Q
    iq = zb * np.exp(-1j * 2*np.pi*c*(tb - tb[0]))
    I = np.real(iq); Q = np.imag(iq)
    # lowpass to symbol rate
    sym = sym_rate or (c * 0.5)  # guess: Nyquist of carrier -> 500k sym/s for 1MHz
    fc = sym
    b, a_ = sps.butter(4, fc / (fs/2), btype='low')
    Il = sps.filtfilt(b, a_, I)
    Ql = sps.filtfilt(b, a_, Q)
    if verbose:
        print(f"   I range {Il.min():.3f}..{Il.max():.3f}  Q range {Ql.min():.3f}..{Ql.max():.3f}")

    # sample at symbol rate
    n_sym = int((tb[-1]-tb[0]) * sym)
    tsym = np.linspace(tb[0], tb[-1], n_sym)
    Is = np.interp(tsym, tb, Il)
    Qs = np.interp(tsym, tb, Ql)
    envs = np.hypot(Is, Qs)
    if verbose:
        print(f"   symbol rate {sym/1e3:.0f} ks/s -> {n_sym} symbols")
        print(f"   symbol env: min={envs.min():.3f} max={envs.max():.3f} mean={envs.mean():.3f}")
        # histogram of env in 8 bins
        h, e = np.histogram(envs, bins=16)
        for hi, e0, e1 in zip(h, e[:-1], e[1:]):
            print(f"     [{e0:7.3f},{e1:7.3f}) {hi:6d} {'#'*int(hi/max(1,h.max())*40)}")

    # BPSK: sign of I after matched filter
    bits_bpsk = (Il > 0).astype(int)  # raw, per sample
    # symbol-level BPSK
    bits_sym = (Is > 0).astype(int)
    # ASK: threshold env at symbol level
    thr_env = (envs.max() + envs.min()) / 2
    bits_ask = (envs > thr_env).astype(int)

    def to_bytes(bits, lsbf=False):
        out = []
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i+j]
            out.append(byte)
        return out

    for name, bits in (("BPSK(sym)", bits_sym), ("ASK(sym)", bits_ask)):
        by = to_bytes(bits)
        if by:
            hx = " ".join(f"{x:02x}" for x in by[:48])
            if verbose:
                print(f"   {name} bytes ({len(by)}): {hx}")

    return dict(t=tb, v=vb, env=envs, Is=Is, Qs=Qs, tsym=tsym,
                bits_bpsk=bits_sym, bits_ask=bits_ask, carrier=c, sym=sym)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/scopes/2023-12-31_BurstZoomedIn/BurstZoomedIn.csv"
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    analyze(os.path.join(REPO, path), burst_idx=idx)
