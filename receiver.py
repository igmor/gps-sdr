#!/usr/bin/env python3
"""End-to-end GPS receiver: HackRF capture -> position.

    python receiver.py data/gps5.bin --center 27000          # standalone (cold) receiver
    python receiver.py data/gps5.bin --center 27000 --agps   # assisted: IGS orbits + rough position

With --agps the rough position comes from --lat/--lon, or else from the
last saved fix (data/fixes.json). The capture start time comes from --time
or the file's modification time minus its length.
"""
import argparse
import datetime as dt
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gps.acquisition import acquire, acquire_targeted
from gps.io import read_hackrf
from gps.navmsg import decode_ephemeris
from gps.orbit import ecef_to_lla, lla_to_ecef
from gps.pvt import TransmitClock, receive_time_estimate, solve
from gps.tracking import cn0_estimate, track


def _track_one(job):
    path, fs, offset, prn, code_phase_samples, carrier = job
    raw = np.memmap(path, dtype=np.int8, mode="r")
    head = raw[: int(fs) * 2].astype(np.float32)
    dc = head[0::2].mean() + 1j * head[1::2].mean()
    n_ms = int((len(raw) // 2 / fs - offset) * 1000) - 2
    res = track(raw, fs, prn, int(offset * fs) + code_phase_samples, carrier, n_ms, dc=dc)
    return {k: getattr(res, k) for k in res.__dataclass_fields__}


def cold_acquisition(args):
    iq = read_hackrf(args.file, args.fs, seconds=args.acq_ms * 1e-3 + 0.001, offset_seconds=args.offset)
    acqs = [a for a in acquire(iq, args.fs, doppler_max=args.doppler, n_noncoh=args.acq_ms,
                               if_freq=args.center) if a.detected]
    for a in acqs:
        a.doppler_hz += args.center
    return acqs, {}


def assisted_acquisition(args):
    from gps.agps import load_ephemerides, predict, utc_to_gps

    if args.time:
        t0 = dt.datetime.fromisoformat(args.time).replace(tzinfo=dt.timezone.utc)
    else:
        dur = os.path.getsize(args.file) / 2 / args.fs
        t0 = dt.datetime.fromtimestamp(os.path.getmtime(args.file), dt.timezone.utc) - dt.timedelta(seconds=dur)
    t0 += dt.timedelta(seconds=args.offset)
    week, tow = utc_to_gps(t0)
    prior = prior_position(args)
    print(f"A-GPS: capture time ~{t0:%Y-%m-%d %H:%M:%S} UTC, rough position "
          f"{ecef_to_lla(prior)[0]:.3f}, {ecef_to_lla(prior)[1]:.3f}")
    preds = predict(load_ephemerides(t0), prior, week, tow, min_el=args.min_el)
    print(f"  {len(preds)} satellites predicted above {args.min_el:.0f} deg: " +
          ", ".join(f"PRN {p['prn']} ({p['el']:.0f}°)" for p in preds))

    # 1) Receiver clock offset: wide search on the few highest satellites (strongest signals).
    iq = read_hackrf(args.file, args.fs, seconds=args.coh * args.noncoh * 1e-3 + 0.01,
                     offset_seconds=args.offset)
    top = preds[:4]
    wide = acquire(iq, args.fs, prns=[p["prn"] for p in top], doppler_max=args.doppler,
                   n_noncoh=args.acq_ms, if_freq=args.center)
    offs = [a.doppler_hz + args.center - p["doppler"] for a, p in zip(wide, top) if a.detected]
    if not offs:
        print("  clock calibration failed (no satellite found in the wide search); falling back to cold search")
        return cold_acquisition(args)
    clock_off = float(np.median(offs))
    print(f"  receiver clock offset from {len(offs)} satellite(s): {clock_off:+.0f} Hz "
          f"({-clock_off / 1575.42:+.2f} ppm)")

    # 2) Targeted coherent search of every predicted satellite.
    targets = [(p["prn"], p["doppler"] + clock_off) for p in preds]
    res = acquire_targeted(iq, args.fs, targets, half_window=args.window, n_coh=args.coh,
                           n_noncoh=args.noncoh, threshold=args.threshold)
    print("  targeted search (metric): " + ", ".join(
        f"{r.prn}:{r.metric:.1f}{'' if r.detected else '(x)'}" for r in res))
    return [r for r in res if r.detected], {p["prn"]: p["eph"] for p in preds}


def prior_position(args):
    if args.lat is not None and args.lon is not None:
        return lla_to_ecef(args.lat, args.lon, args.height)
    if os.path.exists("data/fixes.json"):
        m = json.load(open("data/fixes.json"))["mean"]
        return lla_to_ecef(m["lat"], m["lon"], m["h"])
    raise SystemExit("--agps needs a rough position: pass --lat/--lon (or have data/fixes.json from a previous fix)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--fs", type=float, default=4e6)
    p.add_argument("--center", type=float, default=0.0, help="Doppler search center = receiver clock offset (Hz)")
    p.add_argument("--doppler", type=float, default=8000)
    p.add_argument("--acq-ms", type=int, default=50)
    p.add_argument("--offset", type=float, default=0.5)
    p.add_argument("--every", type=float, default=1.0, help="seconds between fixes")
    p.add_argument("--json", default="data/fixes.json", help="where to save fixes for map.py")
    p.add_argument("--threshold", type=float, default=2.5)
    a = p.add_argument_group("assisted GPS")
    a.add_argument("--agps", action="store_true", help="use IGS broadcast orbits + rough position")
    a.add_argument("--lat", type=float)
    a.add_argument("--lon", type=float)
    a.add_argument("--height", type=float, default=0.0)
    a.add_argument("--time", help="capture start time, UTC ISO format (default: from file timestamp)")
    a.add_argument("--min-el", type=float, default=5.0, help="elevation mask (deg)")
    a.add_argument("--coh", type=int, default=10, help="coherent integration (ms)")
    a.add_argument("--noncoh", type=int, default=30, help="non-coherent blocks")
    a.add_argument("--window", type=float, default=300.0, help="+/- Hz around each predicted frequency")
    args = p.parse_args()

    acqs, igs_ephs = assisted_acquisition(args) if args.agps else cold_acquisition(args)
    print(f"Acquired {len(acqs)} satellites: " +
          ", ".join(f"PRN {a.prn} ({a.metric:.1f})" for a in sorted(acqs, key=lambda a: -a.metric)))

    jobs = [(args.file, args.fs, args.offset, a.prn, a.code_phase_samples, a.doppler_hz) for a in acqs]
    with ProcessPoolExecutor() as ex:
        tracks = dict(zip([a.prn for a in acqs], ex.map(_track_one, jobs)))

    # Per satellite: lock quality, ephemeris (own decode or IGS), transmit clock (HOW or predicted).
    clocks, ephs, cn0s, eph_src = {}, {}, {}, {}
    pending = []
    for prn, tr in sorted(tracks.items()):
        lk = ~tr["fll_mode"]
        cn0 = float(np.median(cn0_estimate(tr["I_P"][lk], tr["Q_P"][lk])))
        iq_ratio = np.mean(np.abs(tr["I_P"][lk][1000:])) / np.mean(np.abs(tr["Q_P"][lk][1000:]))
        if iq_ratio < 2.0:
            print(f"  PRN {prn:2d}: C/N0 {cn0:.0f} dB-Hz, no phase lock (|I|/|Q| {iq_ratio:.1f}), dropped")
            continue
        cn0s[prn] = cn0
        try:
            clk = TransmitClock.from_subframes(tr)
            own = decode_ephemeris(clk.subframes, prn)
        except ValueError:
            clk, own = None, None
        if own is not None and own.health == 0:
            ephs[prn], eph_src[prn] = own, "decoded"
        elif prn in igs_ephs:
            ephs[prn], eph_src[prn] = igs_ephs[prn], "IGS"
        else:
            print(f"  PRN {prn:2d}: C/N0 {cn0:.0f} dB-Hz, no usable ephemeris")
            continue
        if clk is not None:
            clocks[prn] = clk
        else:
            pending.append(prn)

    # Integer-millisecond resolution for satellites without a decoded HOW.
    if pending and clocks and args.agps:
        prior = prior_position(args)
        ref = max(clocks, key=lambda q: cn0s[q])
        n_ref = clocks[ref].cse[2000]
        t_rx = receive_time_estimate(clocks[ref], ephs[ref], prior, n_ref)
        for prn in pending:
            clocks[prn] = TransmitClock.from_prediction(tracks[prn], ephs[prn], prior, n_ref, t_rx, args.fs)
    # Self-check: predicting satellites that *do* have a HOW should reproduce it exactly.
    if args.agps and len([q for q in clocks if clocks[q].source == "HOW"]) >= 2:
        prior = prior_position(args)
        how = [q for q in clocks if clocks[q].source == "HOW"]
        ref = max(how, key=lambda q: cn0s[q])
        n_ref = clocks[ref].cse[2000]
        t_rx = receive_time_estimate(clocks[ref], ephs[ref], prior, n_ref)
        errs = [abs(TransmitClock.from_prediction(tracks[q], ephs[q], prior, n_ref, t_rx, args.fs)(n_ref)
                    - clocks[q](n_ref)) * 1e3 for q in how if q != ref]
        print(f"  ms-resolution self-check on {len(errs)} HOW satellites: max disagreement {max(errs):.6f} ms")

    for prn in sorted(clocks):
        print(f"  PRN {prn:2d}: C/N0 {cn0s[prn]:.0f} dB-Hz, ephemeris {eph_src[prn]:7s} (IODE {ephs[prn].iode:3d}), "
              f"time from {clocks[prn].source}")

    if len(clocks) < 4:
        print(f"Only {len(clocks)} usable satellites: need 4 for a fix.")
        return

    spans = [tracks[q]["code_start_exact"] for q in clocks]
    first = max(s[0] for s in spans) + 1.5 * args.fs
    last = min(s[-2] for s in spans)
    samples = np.arange(first, last, args.every * args.fs)
    fixes = [solve(n, clocks, ephs) for n in samples]

    f0 = fixes[0]
    print(f"\nSatellites at first fix (GPS TOW {f0.t_rx:.3f}):")
    print("   PRN   az(deg)  el(deg)  residual(m)")
    for prn in sorted(f0.azel):
        az, el = f0.azel[prn]
        print(f"   {prn:3d}   {az:7.1f}  {el:7.1f}  {f0.residuals[prn]:+9.2f}")
    print("  DOP: " + "  ".join(f"{k} {v:.2f}" for k, v in f0.dop.items()))

    lla = np.array([f.lla for f in fixes])
    enu_spread = _enu_spread(np.array([f.ecef for f in fixes]))
    print(f"\n{len(fixes)} fixes over {(last - first) / args.fs:.0f} s")
    print(f"  mean position: lat {lla[:, 0].mean():.6f}  lon {lla[:, 1].mean():.6f}  "
          f"height {lla[:, 2].mean():.1f} m (above WGS-84 ellipsoid)")
    print(f"  scatter (1-sigma): east {enu_spread[0]:.1f} m  north {enu_spread[1]:.1f} m  up {enu_spread[2]:.1f} m")
    # Receiver clock rate: fixes are spaced by a fixed number of *samples*; the solved
    # receive times say how much true GPS time actually elapsed between them.
    true_dt = (fixes[-1].t_rx - fixes[0].t_rx) / (len(fixes) - 1)
    ppm = (args.every - true_dt) / true_dt * 1e6
    print(f"  HackRF sample clock: {ppm:+.2f} ppm "
          f"(predicts a carrier offset of {-ppm * 1575.42:+.0f} Hz on top of satellite Doppler)")

    with open(args.json, "w") as fh:
        json.dump(fixes_json(args.file, next(iter(ephs.values())).week, fixes, lla, enu_spread, f0, cn0s, ppm),
                  fh, indent=1, default=float)
    print("Saved", args.json)


def fixes_json(source, week, fixes, lla, enu_spread, f_sky, cn0s, ppm):
    return {
        "source": source,
        "week": week,
        "fixes": [{"tow": f.t_rx, "lat": f.lla[0], "lon": f.lla[1], "h": f.lla[2]} for f in fixes],
        "mean": {"lat": lla[:, 0].mean(), "lon": lla[:, 1].mean(), "h": lla[:, 2].mean()},
        "sigma_enu": list(enu_spread),
        "dop": f_sky.dop,
        "sats": [{"prn": p, "az": f_sky.azel[p][0], "el": f_sky.azel[p][1], "cn0": cn0s[p]} for p in sorted(f_sky.azel)],
        "clock_ppm": ppm,
    }


def _enu_spread(ecef):
    lat, lon = np.radians(ecef_to_lla(ecef.mean(axis=0))[:2])
    R = np.array([[-np.sin(lon), np.cos(lon), 0],
                  [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                  [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])
    return (R @ (ecef - ecef.mean(axis=0)).T).std(axis=1)


if __name__ == "__main__":
    main()
