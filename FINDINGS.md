# Shimano E-Tube BPSK Command Decode — Findings

Repo: `Shimano-E-Tube`
Data: `data/scopes/2024-01-01_ButtonDownManySegments`
Canonical decoder: `analysis/decode.py`
Python: `/mnt/Abominations/Managed/openclaw/workspace/.etube-venv/bin/python`

```
$ python analysis/decode.py data/scopes/2024-01-01_ButtonDownManySegments
```

## 1. Capture geometry (confirmed)

- **500 segments, index 1..500, no gaps** (verified by filename scan).
- Raw window: 3649 pts/segment, X-incr 6.85e-08 s (fs = 14.6 MHz), 250 us window.
  After burst-cropping (`fsk.find_burst`) the active burst is ~729 pts (~50 us).
- Instrument: DSOX1204G:CN61037459, 500 waveforms per file.
- **Key fact: each CSV's timestamps are window-relative** (every segment starts
  at ~0). Global time order is therefore the **segment index**, NOT per-file `t[0]`.
  Any cross-segment timeline must key on the index.
- This is a "button held down" capture: the remote retransmits a short command
  burst, then idles, then bursts again. 17 quality command frames across 5 bursts.

## 2. Modulation

- **Constant-envelope BPSK** on the I axis (Q ~= 0), carrier ~1.000 MHz,
  **16 symbols per frame** (~1 us/symbol, ~16 us frame).
- Carrier lock: coarse 500 Hz -> fine 50 Hz -> max-min (weakest-symbol) refine.
- Quality gate: bimodality >= 0.92 AND |fc - 1 MHz| < 20 kHz.

## 3. Two command classes (pi-immune differential head)

Per-frame absolute bits are decoded against the frame's strongest symbol, which
is **pi-ambiguous** (a 180 deg phase flip leaves the magnitude profile identical).
The **differential head** `d[k] = ab[k] ^ ab[k-1]` is invariant under a global
phase flip, so it is the true, pi-immune class key.

- **Class A** — diff head `00111011` (near-matches `00111010`, `00111001`) [9 frames]
- **Class B** — diff head `01000110` (near-match `01000111`)              [8 frames]

Within a class the differential head is 100% stable (first ~7 bits at 100%
agreement); the absolute head wanders by 1 bit because of the per-frame phase
orientation (see head-frequency table in section 5).

Per-symbol magnitude stays strong through **all 16 symbols** (no energy cliff),
so the frame is genuinely 16 symbols: an 8-bit header + an 8-bit payload.

## 4. The tail is a varying payload, NOT noise

The tail 8 bits differ frame to frame because each transmission carries different
data (a per-press counter / nonce / checksum), not because of SNR. Every quality
frame has **no low-decision-margin symbol** (all |Re/|.| > 0.5), so the tail
decodes cleanly per frame; the earlier "low-SNR Class B" concern was overstated.

- **Class A tail (true time order):** 69, 19, 14, 4, 1, 2, 0, 1, 3
  (segs 2,3,4,73,85,86,179,180,361). Values cluster low and increment locally
  (…0,1,2,3) — consistent with a per-transmission sequence counter.
- **Class B tail:** 68, 19, 15, 3, 15, 62, 15, 194 (segs 6,8,10,84,88,90,181,183).
  `0f` repeats 3x; values vary. `c2`/`3e` look like high-value or checksum bytes.

## 5. A/B run structure per burst

Button-hold = repeated bursts. Grouped by segment-index proximity (gap <= 3):

| burst | segs    | n | run    |
|-------|---------|---|--------|
| 0     | 2..10   | 6 | AAABBB |
| 1     | 73      | 1 | A      |
| 2     | 84..90  | 5 | BAABB  |
| 3     | 179..183| 4 | AABB   |
| 4     | 361     | 1 | A      |

The two classes are interleaved within a burst (not strictly alternating): a
button-hold emits a mixed A/B run, e.g. `AAABBB`, then `BAABB`, then `AABB`.
The A/B pairing is consistent across bursts (same two commands retransmitted).

### Head frequency within class (absolute head, pi-orientation dependent)

- Class A: `d2` x5, `d3` x3, `d1` x1  (diff heads `00111011`, `00111010`, `00111001`)
- Class B: `84` x4, `85` x3, `7b` x1  (diff heads `01000110` x5, `01000111` x3)

### Per-class consensus (group-aligned majority vote)

- **Class A:** abs consensus `d2 01` (hex), abs agreement 100% on the 6-bit
  header core, decaying through the payload. Diff consensus `0111 0110 0000 001`.
- **Class B:** abs consensus `7b f0` (hex), abs agreement 100% on the 7-bit
  header core. Diff consensus `1000 1100 0001 000`.

## 6. Scripts (analysis/)

- `decode.py`        **canonical pipeline**: geometry + timeline + bursts + consensus.
- `fsk.py`           load_csv, burst_segments (index-ordered), find_burst.
- `final.py`         carrier lock (coarse->fine->weakest), dh, ham, classify.
- `order2.py`        pi-immune differential-head timeline + burst structure.
- `consensus.py`     group-aligned majority-vote consensus per class.
- `profile.py`       per-symbol magnitude/energy profile (frame-length check).
- `bbtrace.py`       baseband I/Q trace (shows pure signal on I axis, Q ~= 0).

## 7. Open

- [ ] **Protocol semantics:** map the two class headers (A vs B) and the tail
      counter/checksum to the Shimano E-Tube (Di2) command set. The absolute
      header bytes are `d2` (A) and `7b` (B) in the consensus orientation; the
      true command ID needs the documented E-Tube air protocol (or a reference
      device) to confirm which bit order / phase is canonical.
- [ ] **Payload meaning:** the tail is a varying 8-bit field. Confirm whether it
      is a monotonically incrementing sequence number, a per-command parameter
      (e.g. assist level, target speed), or a checksum. Class A shows local
      increments (0->1->2->3); Class B is less clean and may be a different field.
- [ ] **Burst semantics:** understand why a button-hold emits a mixed A/B run
      (e.g. A = "press", B = "held/ack", or A/B = two different buttons).
