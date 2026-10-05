#!/usr/bin/env python3
"""Run acquisition on a HackRF capture (or a synthetic file) and list satellites.

    python acquire.py data/gps.bin --fs 4e6
"""
import argparse

from gps.acquisition import acquire
from gps.io import read_hackrf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--fs", type=float, default=4e6, help="sample rate (Hz)")
    p.add_argument("--offset", type=float, default=0.5,
                   help="seconds to skip at the start (the HackRF AGC/PLL settles there)")
    p.add_argument("--ms", type=int, default=10, help="non-coherent integration (ms)")
    p.add_argument("--doppler", type=float, default=10e3,
                   help="+/- Doppler search range in Hz. The HackRF's stock crystal can be off by "
                        "several ppm (1 ppm = 1.575 kHz at L1). If nothing is found, try 30000.")
    p.add_argument("--center", type=float, default=0.0,
                   help="center of the Doppler search in Hz (e.g. the receiver clock offset once known)")
    p.add_argument("--threshold", type=float, default=2.5)
    args = p.parse_args()

    iq = read_hackrf(args.file, args.fs, seconds=args.ms * 1e-3 + 0.001, offset_seconds=args.offset)
    results = acquire(iq, args.fs, doppler_max=args.doppler, n_noncoh=args.ms, threshold=args.threshold,
                      if_freq=args.center)

    print(f"{'PRN':>4} {'metric':>7} {'Doppler(Hz)':>12} {'code phase(chips)':>18}")
    for r in sorted(results, key=lambda r: -r.metric):
        flag = " <-- detected" if r.detected else ""
        print(f"{r.prn:>4} {r.metric:>7.2f} {r.doppler_hz + args.center:>12.0f} {r.code_phase_chips:>18.1f}{flag}")
    print(f"\n{sum(r.detected for r in results)} satellites detected")


if __name__ == "__main__":
    main()
