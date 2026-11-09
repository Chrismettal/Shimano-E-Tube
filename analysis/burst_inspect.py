#!/usr/bin/env python3
import sys, os
import numpy as np
from scipy import signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_csv(path):
    t, v = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 2: continue
            a, b = line.split(",")
            t.append(float(a)); v.append(float(b))
    return np.array(t), np.array(v)

path = sys.argv[1]
t, v = load_csv(os.path.join(REPO, path))
dt = np.median(np.diff(t)); fs = 1/dt
v = v - np.mean(v)
z = sps.hilbert(v); env = np.abs(z); phase = np.unwrap(np.angle(z))

# find bursts
thr = env.max()*0.15
start = np.where(env > thr)[0]
groups = np.split(start, np.where(np.diff(start) > 20)[0]+1)
spans = [(g[0], g[-1]) for g in groups if len(g) > 10]
print("bursts:", [(round(t[a]*1e6,1), round(t[b]*1e6,1)) for a,b in spans])

a, b = spans[0]
tb, vb, eb, pb = t[a:b+1], v[a:b+1], env[a:b+1], phase[a:b+1]
print(f"burst dur {(tb[-1]-tb[0])*1e6:.2f} us, n={len(tb)}")

# carrier freq within burst
n=len(vb); spec=np.abs(np.fft.rfft(vb*np.hanning(n))); freqs=np.fft.rfftfreq(n,d=dt)
c=freqs[spec.argmax()]
print(f"carrier {c/1e6:.5f} MHz -> cycles in burst = {(tb[-1]-tb[0])*c:.1f}")

# demod at exact 1MHz
for carr in (1.0e6, c):
    iq = z[a:b+1]*np.exp(-1j*2*np.pi*carr*(tb-tb[0]))
    Il = sps.filtfilt(*sps.butter(4, 250e3/(fs/2)), np.real(iq))
    Ql = sps.filtfilt(*sps.butter(4, 250e3/(fs/2)), np.imag(iq))
    print(f"  carr={carr/1e6:.4f}MHz I[{Il.min():.3f},{Il.max():.3f}] Q[{Ql.min():.3f},{Ql.max():.3f}]")

# plot
fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
ax[0].plot(tb*1e6, vb, lw=0.5); ax[0].set_title(f"raw burst (carrier {c/1e6:.4f} MHz)")
ax[1].plot(tb*1e6, eb, lw=0.5); ax[1].set_title("envelope")
ax[2].plot(tb*1e6, pb, lw=0.5); ax[2].set_title("phase (unwrap, rad)")
ax[3].plot(tb*1e6, Il, lw=0.5, label="I"); ax[3].plot(tb*1e6, Ql, lw=0.5, label="Q"); ax[3].legend(); ax[3].set_title("demod I/Q @1MHz")
for a_ in ax: a_.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(REPO, "analysis", "burst0.png")
plt.savefig(out, dpi=110)
print("saved", out)
