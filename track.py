#!/usr/bin/env python3
"""Acquire and track one satellite, then show its navigation data bits.

    python track.py data/gps4.bin --prn 7 --center 27000
"""
import argparse

import numpy as np

from gps.acquisition import acquire
from gps.io import read_hackrf
from gps.tracking import bit_sync, cn0_estimate, nav_bits, track

PREAMBLE = np.array([1, -1, -1, -1, 1, -1, 1, 1])  # 10001011, first 8 bits of every subframe

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def find_preambles(bits):
    """Positions where the preamble (or its inverse) appears. Costas loops
    have a 180-degree ambiguity, so the bits may be inverted overall.
    Real subframe starts repeat every 300 bits (6 s)."""
    c = np.correlate(bits.astype(int), PREAMBLE, mode="valid")
    return np.nonzero(np.abs(c) == 8)[0], c


def plot(res, cn0, bit_start, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), facecolor=SURFACE)
    for a in ax.flat:
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, linewidth=0.6)
        a.spines[["top", "right"]].set_visible(False)

    locked = ~res.fll_mode
    t0 = res.t[locked][0] + 1.0
    win = (res.t >= t0) & (res.t < t0 + 0.4)
    a = ax[0, 0]
    a.plot(res.t[win] - t0, res.I_P[win], color=BLUE, linewidth=1.2, label="I prompt")
    a.plot(res.t[win] - t0, res.Q_P[win], color=ORANGE, linewidth=1.2, label="Q prompt")
    a.set_title("Prompt correlator, 400 ms: data bits are the 20 ms steps in I", loc="left")
    a.set_xlabel("time (s)")
    a.legend(frameon=False, loc="upper right")

    a = ax[0, 1]
    sel = locked & (res.t > res.t[locked][0] + 1.0)
    a.scatter(res.I_P[sel], res.Q_P[sel], s=2, color=BLUE, alpha=0.3, linewidths=0)
    lim = np.percentile(np.abs(np.r_[res.I_P[sel], res.Q_P[sel]]), 99.5)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_title("I/Q after phase lock: two BPSK clusters on I", loc="left")
    a.set_xlabel("I")
    a.set_ylabel("Q")

    a = ax[1, 0]
    a.plot(res.t, res.carr_freq, color=BLUE, linewidth=1.2)
    a.axvspan(res.t[0], res.t[locked][0], color=GRID, alpha=0.6, linewidth=0)
    a.text(res.t[0], res.carr_freq.max(), " FLL pull-in", color=INK2, va="top")
    a.set_title("Carrier frequency (satellite Doppler + HackRF clock offset)", loc="left")
    a.set_xlabel("time (s)")
    a.set_ylabel("Hz")
    a.ticklabel_format(useOffset=False, axis="y")

    a = ax[1, 1]
    a.plot(np.arange(len(cn0)) + 0.5 + res.t[0], cn0, color=BLUE, linewidth=1.2, marker="o", markersize=4)
    a.set_title("Estimated C/N0 per second", loc="left")
    a.set_xlabel("time (s)")
    a.set_ylabel("dB-Hz")

    fig.suptitle(f"PRN {res.prn} tracking", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--prn", type=int, required=True)
    p.add_argument("--fs", type=float, default=4e6)
    p.add_argument("--center", type=float, default=0.0, help="Doppler search center (clock offset), Hz")
    p.add_argument("--doppler", type=float, default=8000)
    p.add_argument("--acq-ms", type=int, default=200)
    p.add_argument("--offset", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=None, help="how long to track (default: whole file)")
    p.add_argument("--plot", default=None, help="PNG path (default: data/track_prn<N>.png)")
    args = p.parse_args()

    iq = read_hackrf(args.file, args.fs, seconds=args.acq_ms * 1e-3 + 0.001, offset_seconds=args.offset)
    acq = acquire(iq, args.fs, prns=[args.prn], doppler_max=args.doppler, n_noncoh=args.acq_ms,
                  if_freq=args.center)[0]
    doppler = acq.doppler_hz + args.center
    print(f"Acquisition: PRN {acq.prn} metric {acq.metric:.2f} Doppler {doppler:.0f} Hz "
          f"code start sample {acq.code_phase_samples}")

    raw = np.memmap(args.file, dtype=np.int8, mode="r")
    head = raw[: int(args.fs) * 2].astype(np.float32)
    dc = head[0::2].mean() + 1j * head[1::2].mean()
    total_ms = int((len(raw) // 2 / args.fs - args.offset) * 1000) - 2
    n_ms = total_ms if args.seconds is None else int(args.seconds * 1000)

    start = int(args.offset * args.fs) + acq.code_phase_samples
    res = track(raw, args.fs, args.prn, start, doppler, n_ms, dc=dc)
    print(f"Tracked {len(res.t)} ms")

    locked = ~res.fll_mode
    cn0 = cn0_estimate(res.I_P[locked], res.Q_P[locked])
    print("C/N0 per second (dB-Hz):", " ".join(f"{v:.0f}" for v in cn0))
    pll_ratio = np.mean(np.abs(res.I_P[locked][1000:])) / np.mean(np.abs(res.Q_P[locked][1000:]))
    print(f"Mean |I|/|Q| after lock: {pll_ratio:.1f}  (phase-locked if well above 1)")

    off, hist = bit_sync(res.I_P)
    print(f"Bit sync: edges at ms {off} mod 20; transition histogram: {hist.tolist()}")
    bits, bit_start = nav_bits(res.I_P, off)
    print(f"{len(bits)} nav bits decoded")
    print("First 120 bits:", "".join("1" if b > 0 else "0" for b in bits[:120]))

    pre, _ = find_preambles(bits)
    print(f"Preamble matches at bit positions: {pre.tolist()}")
    diffs = np.diff(pre)
    if np.any(diffs % 300 == 0):
        print("Matches spaced by multiples of 300 bits: subframe sync looks real")

    out = args.plot or f"data/track_prn{args.prn}.png"
    plot(res, cn0, bit_start, out)
    np.savez(f"data/track_prn{args.prn}.npz", **{k: getattr(res, k) for k in res.__dataclass_fields__})
    print("Saved", out)


if __name__ == "__main__":
    main()
