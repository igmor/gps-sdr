#!/usr/bin/env python3
"""Decode the navigation message from a tracking result and compute the
satellite's position and clock.

    python track.py data/gps4.bin --prn 7 --center 27000   # writes data/track_prn7.npz
    python nav.py data/track_prn7.npz
"""
import argparse
import datetime as dt

import numpy as np

from gps.navmsg import decode_ephemeris, find_subframes
from gps.orbit import C, ecef_to_lla, sat_clock_bias, sat_position, sat_velocity
from gps.tracking import bit_sync, nav_bits

GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
LEAP_SECONDS = 18  # GPS - UTC since 2017


def gps_to_utc(week, tow):
    return GPS_EPOCH + dt.timedelta(weeks=week, seconds=tow - LEAP_SECONDS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("track_file")
    args = p.parse_args()

    d = np.load(args.track_file)
    prn = int(d["prn"])
    off, _ = bit_sync(d["I_P"])
    bits, _ = nav_bits(d["I_P"], off)
    subframes = find_subframes(bits)

    print(f"PRN {prn}: {len(bits)} bits, {len(subframes)} subframes with valid TLM/HOW")
    for s in subframes:
        print(f"  bit {s.start_bit:5d}  subframe {s.id}  TOW(next) {s.tow * 6:6d} s  "
              f"parity {sum(s.parity_ok)}/10")

    eph = decode_ephemeris(subframes, prn)
    if eph is None:
        print("No complete, consistent set of subframes 1-3 yet (need ~18-30 s of clean data).")
        return

    print(f"\nEphemeris (IODE {eph.iode}, IODC {eph.iodc}, health {eph.health}, URA index {eph.ura})")
    print(f"  week {eph.week}   toe {eph.toe:.0f} s ({gps_to_utc(eph.week, eph.toe):%Y-%m-%d %H:%M} UTC)")
    print(f"  sqrt(A) {eph.sqrt_a:.6f} -> A {eph.sqrt_a ** 2 / 1e3:.3f} km")
    print(f"  e {eph.e:.8f}   i0 {np.degrees(eph.i0):.4f} deg   Omega0 {np.degrees(eph.omega0):.4f} deg")
    print(f"  omega {np.degrees(eph.omega):.4f} deg   M0 {np.degrees(eph.m0):.4f} deg")
    print(f"  delta_n {eph.delta_n:.4e} rad/s   Omega_dot {eph.omega_dot:.4e} rad/s   IDOT {eph.idot:.4e} rad/s")
    print(f"  Crs {eph.crs:.3f} m  Crc {eph.crc:.3f} m  Cus {eph.cus:.3e}  Cuc {eph.cuc:.3e}  "
          f"Cis {eph.cis:.3e}  Cic {eph.cic:.3e}")
    print(f"  clock: af0 {eph.af0:.6e} s  af1 {eph.af1:.4e}  af2 {eph.af2:.1e}  toc {eph.toc:.0f}  "
          f"TGD {eph.tgd:.3e} s")

    # Position at the start of the first decoded subframe (its GPS time is known exactly).
    s0 = subframes[0]
    t = s0.tow * 6 - 6
    pos, e_anom = sat_position(eph, t)
    vel = sat_velocity(eph, t)
    clk = sat_clock_bias(eph, t, e_anom)
    lat, lon, h = ecef_to_lla(pos)
    print(f"\nAt GPS TOW {t} s ({gps_to_utc(eph.week, t):%Y-%m-%d %H:%M:%S} UTC):")
    print(f"  ECEF   x {pos[0] / 1e3:12.3f} km   y {pos[1] / 1e3:12.3f} km   z {pos[2] / 1e3:12.3f} km")
    print(f"  |r| {np.linalg.norm(pos) / 1e3:.3f} km from Earth's center, speed {np.linalg.norm(vel):.1f} m/s (ECEF)")
    print(f"  sub-satellite point: lat {lat:.3f}  lon {lon:.3f}  altitude {h / 1e3:.1f} km")
    print(f"  satellite clock offset {clk * 1e6:.3f} us  (= {clk * C / 1e3:.2f} km of range)")


if __name__ == "__main__":
    main()
