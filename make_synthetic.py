#!/usr/bin/env python3
"""Write a synthetic GPS capture in HackRF format with known satellites.

    python make_synthetic.py data/synth.bin --seconds 2
"""
import argparse

from gps.io import write_hackrf
from gps.synth import SatSignal, generate

# Ground truth used for checking the receiver.
SATS = [
    SatSignal(prn=3, doppler_hz=1250.0, code_phase_chips=100.0, cn0_dbhz=47),
    SatSignal(prn=11, doppler_hz=-3420.0, code_phase_chips=512.3, cn0_dbhz=44),
    SatSignal(prn=17, doppler_hz=2875.0, code_phase_chips=800.7, cn0_dbhz=42),
    SatSignal(prn=24, doppler_hz=-650.0, code_phase_chips=10.0, cn0_dbhz=40),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--fs", type=float, default=4e6)
    p.add_argument("--seconds", type=float, default=2.0)
    args = p.parse_args()

    write_hackrf(args.file, generate(SATS, args.fs, args.seconds))
    print("wrote", args.file)
    for s in SATS:
        print(f"  PRN {s.prn:2d}  Doppler {s.doppler_hz:8.0f} Hz  code phase {s.code_phase_chips:7.1f} chips  "
              f"C/N0 {s.cn0_dbhz} dB-Hz")


if __name__ == "__main__":
    main()
