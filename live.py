#!/usr/bin/env python3
"""Live GPS receiver: stream from the HackRF, track satellites in parallel,
fix position every second, and show it on a self-updating map.

    python live.py --center 27000                     # live from the HackRF, assisted by default
    python live.py --file data/gps5.bin --center 27000   # replay a capture in real time (testing)
    python live.py --cold --center 27000              # no assistance: blind search of all 32 PRNs

The map is served at http://127.0.0.1:8765/live_map.html (local only).
Assistance needs a rough position (--lat/--lon, or the last saved fix in
data/fixes.json) and downloads the IGS broadcast orbit file for today.
"""
import argparse
import datetime as dt
import functools
import http.server
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time

import numpy as np

from gps.navmsg import decode_ephemeris
from gps.orbit import ecef_to_lla, lla_to_ecef
from gps.pvt import TransmitClock, receive_time_estimate, solve
from gps.stream import RECORD_FIELDS, FileSource, HackRFSource, RingBuffer, acq_worker, channel_worker
from gps.tracking import cn0_estimate

F = {name: i for i, name in enumerate(RECORD_FIELDS)}
KEEP_MS = 60_000          # tracking history kept per channel (multiple of 20 ms keeps bit alignment)
MAX_CHANNELS = 12


class Channel:
    def __init__(self, prn, proc):
        self.prn, self.proc = prn, proc
        self.rows = np.zeros((0, len(RECORD_FIELDS)))
        self.pending = []
        self.trimmed = 0          # rows dropped from the front so far
        self.anchor = None        # (absolute ms index, satellite time at that code period, source)
        self.eph, self.eph_src = None, None
        self.cn0, self.locked, self.unlocked_s = 0.0, False, 0
        self.pos = 0              # tracker's current sample (to measure lag)
        self.last_how_try = 0.0

    def add(self, recs, pos):
        self.pending.append(recs)   # cheap; concatenated once per second in flush()
        self.pos = pos

    def flush(self):
        if not self.pending:
            return
        self.rows = np.concatenate([self.rows] + self.pending)
        self.pending = []
        extra = len(self.rows) - KEEP_MS
        if extra > 0:
            extra -= extra % 20
            self.rows = self.rows[extra:]
            self.trimmed += extra

    def track_dict(self):
        r = self.rows
        return {"I_P": r[:, F["I_P"]], "Q_P": r[:, F["Q_P"]],
                "code_start_exact": r[:, F["code_start_exact"]], "fll_mode": r[:, F["fll_mode"]] > 0.5}

    def clock(self):
        if self.anchor is None:
            return None
        abs_ms, tow, src = self.anchor
        return TransmitClock(self.rows[:, F["code_start_exact"]], abs_ms - self.trimmed, tow, source=src)

    def update_quality(self):
        lk = self.rows[:, F["fll_mode"]] < 0.5
        I, Q = self.rows[lk, F["I_P"]][-1000:], self.rows[lk, F["Q_P"]][-1000:]
        if len(I) < 1000:
            self.locked = False
            return
        self.cn0 = float(cn0_estimate(I, Q)[0])
        self.locked = np.mean(np.abs(I)) / np.mean(np.abs(Q)) > 2.0 and self.cn0 > 28
        self.unlocked_s = 0 if self.locked else self.unlocked_s + 1


def serve_map(directory, port):
    handler = functools.partial(QuietHandler, directory=directory)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", help="replay a capture instead of streaming from the HackRF")
    p.add_argument("--fs", type=float, default=4e6)
    p.add_argument("--center", type=float, default=0.0, help="expected receiver clock offset (Hz); 27000 for this HackRF")
    p.add_argument("--doppler", type=float, default=8000, help="+/- Hz for the wide (calibration / cold) search")
    p.add_argument("--cold", action="store_true", help="no assistance")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--height", type=float, default=0.0)
    p.add_argument("--time", help="stream start time UTC (ISO); default now, or file timestamp for --file")
    p.add_argument("--min-el", type=float, default=5.0)
    p.add_argument("--buffer", type=float, default=30.0, help="ring buffer length (s)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-bias", action="store_true", help="do not power the antenna (passive antenna)")
    p.add_argument("--hackrf-transfer", default=os.path.expanduser("~/radioconda/bin/hackrf_transfer"))
    p.add_argument("--duration", type=float, default=0, help="stop after this many seconds (0 = until Ctrl-C)")
    args = p.parse_args()
    fs = args.fs

    # ---- assistance data
    prior = None
    if args.lat is not None and args.lon is not None:
        prior = lla_to_ecef(args.lat, args.lon, args.height)
    elif os.path.exists("data/fixes.json"):
        m = json.load(open("data/fixes.json"))["mean"]
        prior = lla_to_ecef(m["lat"], m["lon"], m["h"])
    if args.time:
        t_start = dt.datetime.fromisoformat(args.time).replace(tzinfo=dt.timezone.utc)
    elif args.file:
        dur = os.path.getsize(args.file) / 2 / fs
        t_start = dt.datetime.fromtimestamp(os.path.getmtime(args.file), dt.timezone.utc) - dt.timedelta(seconds=dur)
    else:
        t_start = dt.datetime.now(dt.timezone.utc)
    assisted = not args.cold and prior is not None
    igs = {}
    if assisted:
        from gps.agps import load_ephemerides, predict, utc_to_gps
        print("Loading IGS broadcast ephemerides...")
        igs = load_ephemerides(t_start)
        if not igs:
            print("  none available: running unassisted")
            assisted = False

    # ---- plumbing
    ring = RingBuffer(int(args.buffer * fs))
    ctx = mp.get_context("spawn")
    out_q, cmd_q, stop_ev = ctx.Queue(), ctx.Queue(), ctx.Event()
    acq_proc = ctx.Process(target=acq_worker, args=(ring.name, ring.cap, fs, cmd_q, out_q, stop_ev), daemon=True)
    acq_proc.start()
    if args.file:
        src = FileSource(ring, args.file, fs)
    else:
        src = HackRFSource(ring, args.hackrf_transfer, 1575.42e6, fs, bias=not args.no_bias)
    src.start()

    from map import write_live_page
    write_live_page("data/live_map.html")
    srv = serve_map(os.path.abspath("data"), args.port)
    map_url = f"http://127.0.0.1:{args.port}/live_map.html"

    channels = {}
    clock_off = None if assisted else args.center
    preds, pred_time = [], -1e9
    acq_busy, last_acq = False, -1e9
    dc = 0j
    fixes, fix_samples = [], []
    first_fix_s = None
    t0 = time.time()
    status = ""

    def stream_time():
        return t_start + dt.timedelta(seconds=ring.written() / fs)

    def spawn(prn, start, carrier):
        proc = ctx.Process(target=channel_worker,
                           args=(ring.name, ring.cap, fs, prn, int(start), carrier, dc, out_q, stop_ev), daemon=True)
        proc.start()
        channels[prn] = Channel(prn, proc)
        if assisted:
            for pr in preds:
                if pr["prn"] == prn:
                    channels[prn].eph, channels[prn].eph_src = pr["eph"], "IGS"

    def drop(prn, why):
        ch = channels.pop(prn)
        ch.proc.terminate()
        return f"PRN {prn} dropped: {why}"

    try:
        next_tick = time.time() + 1
        while True:
            # -- drain worker messages (time-boxed: channels catching up send ~40 batches/s each)
            drain_until = time.time() + 0.1
            while time.time() < drain_until:
                try:
                    msg = out_q.get(timeout=0.02)
                except queue.Empty:
                    break
                if msg[0] == "trk" and msg[1] in channels:
                    channels[msg[1]].add(msg[2], msg[3])
                elif msg[0] == "lost" and msg[1] in channels:
                    status = drop(msg[1], msg[2])
                elif msg[0] == "acq":
                    _, prn, start, carrier, metric, detected, kind = msg
                    if detected and prn not in channels and len(channels) < MAX_CHANNELS:
                        spawn(prn, start, carrier)
                        if assisted and clock_off is None:
                            pr = next(x for x in preds if x["prn"] == prn)
                            clock_off = carrier - pr["doppler"]
                            status = f"clock offset calibrated on PRN {prn}: {clock_off:+.0f} Hz"
                elif msg[0] == "acq_done":
                    acq_busy = False
            if time.time() < next_tick:
                continue
            next_tick += 1
            if src.error:
                status = src.error
            el_s = ring.written() / fs
            if args.duration and el_s > args.duration:
                print(f"\nduration reached ({el_s:.0f} s)")
                break
            if ring.written() < fs:
                continue
            if dc == 0j:
                head = ring.read_complex(ring.written() - int(fs // 2), ring.written())
                dc = head.mean() if head is not None else 0j

            # -- predictions and acquisition scheduling
            if assisted and el_s - pred_time > 30:
                week, tow = utc_to_gps(stream_time())
                preds = predict(igs, prior, week, tow, min_el=args.min_el)
                pred_time = el_s
            if not acq_busy and el_s - last_acq > 5:
                if assisted:
                    todo = [pr for pr in preds if pr["prn"] not in channels]
                    if clock_off is None:
                        job = ("wide", [(pr["prn"], args.center + pr["doppler"], args.doppler / 2)
                                        for pr in todo[:4]], 50)
                    else:
                        job = ("targeted", [(pr["prn"], pr["doppler"] + clock_off) for pr in todo], 300.0, 10, 30)
                else:
                    todo = [q for q in range(1, 33) if q not in channels]
                    job = ("wide", [(q, args.center, args.doppler) for q in todo], 50)
                if job[1]:
                    cmd_q.put(job)
                    acq_busy, last_acq = True, el_s

            # -- per-channel navigation
            for ch in channels.values():
                ch.flush()
            for prn, ch in list(channels.items()):
                if len(ch.rows) < 1500:
                    continue
                ch.update_quality()
                if ch.unlocked_s >= 5:
                    status = drop(prn, "lost lock")
                    continue
                if ch.locked and (ch.anchor is None or ch.anchor[2] != "HOW" or ch.eph_src != "decoded") \
                        and time.time() - ch.last_how_try > 2:
                    ch.last_how_try = time.time()
                    try:
                        c = TransmitClock.from_subframes(ch.track_dict())
                        ch.anchor = (c.ref_ms + ch.trimmed, c.ref_tow, "HOW")
                        e = decode_ephemeris(c.subframes, prn)
                        if e is not None and e.health == 0:
                            ch.eph, ch.eph_src = e, "decoded"
                    except ValueError:
                        pass

            # -- integer-ms resolution for channels without a HOW (needs a rough position)
            pos_est = fixes[-1].ecef if fixes else prior
            how = [c for c in channels.values() if c.locked and c.anchor and c.anchor[2] == "HOW" and c.eph]
            if pos_est is not None and how:
                ref = max(how, key=lambda c: c.cn0)
                for ch in channels.values():
                    if ch.locked and ch.eph and (ch.anchor is None or ch.anchor[2] == "predicted"):
                        rc = ref.clock()
                        n_ref = min(rc.cse[-2], ch.rows[-2, F["code_start_exact"]]) - 0.01 * fs
                        t_rx = receive_time_estimate(rc, ref.eph, pos_est, n_ref)
                        c = TransmitClock.from_prediction(ch.track_dict(), ch.eph, pos_est, n_ref, t_rx, fs)
                        ch.anchor = (c.ref_ms + ch.trimmed, c.ref_tow, "predicted")

            # -- position fix
            usable = {p: c for p, c in channels.items() if c.locked and c.anchor and c.eph}
            fix = None
            if len(usable) >= 4:
                clocks = {p: c.clock() for p, c in usable.items()}
                n = min(c.cse[-2] for c in clocks.values()) - 0.002 * fs
                try:
                    fix = solve(n, clocks, {p: c.eph for p, c in usable.items()})
                    if np.linalg.norm(fix.ecef) < 6.3e6 or np.linalg.norm(fix.ecef) > 6.5e6 or \
                            max(abs(r) for r in fix.residuals.values()) > 300:
                        status, fix = "fix rejected (implausible)", None
                except (ValueError, np.linalg.LinAlgError) as e:
                    status = f"fix failed: {e}"
            if fix is not None:
                fixes.append(fix)
                fix_samples.append(n)
                if first_fix_s is None:
                    first_fix_s = el_s
                fixes, fix_samples = fixes[-900:], fix_samples[-900:]
                write_json(fixes, fix_samples, usable, fs, args, preds)

            dashboard(args, el_s, ring, src, channels, preds, fix, first_fix_s, clock_off, assisted, status, map_url, fs)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("\nStopping...")
        stop_ev.set()
        src.stop()
        for ch in channels.values():
            ch.proc.join(timeout=1)
            if ch.proc.is_alive():
                ch.proc.terminate()
        acq_proc.join(timeout=1)
        if acq_proc.is_alive():
            acq_proc.terminate()
        srv.shutdown()
        ring.close(unlink=True)
        if fixes:
            m = np.array([f.lla for f in fixes[-60:]]).mean(axis=0)
            print(f"Last minute mean: {m[0]:.6f}, {m[1]:.6f}, {m[2]:.0f} m   ({len(fixes)} fixes, "
                  f"first fix after {first_fix_s:.0f} s)")


def write_json(fixes, samples, usable, fs, args, preds):
    recent = fixes[-60:]
    ecef = np.array([f.ecef for f in recent])
    lla = np.array([f.lla for f in recent])
    lat, lon = np.radians(lla[:, 0].mean()), np.radians(lla[:, 1].mean())
    R = np.array([[-np.sin(lon), np.cos(lon), 0],
                  [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                  [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])
    sig = (R @ (ecef - ecef.mean(axis=0)).T).std(axis=1) if len(recent) > 1 else np.zeros(3)
    ppm = 0.0
    if len(fixes) > 5:
        dt_true = fixes[-1].t_rx - fixes[-6].t_rx
        ppm = ((samples[-1] - samples[-6]) / fs - dt_true) / dt_true * 1e6
    last = fixes[-1]
    week = next(iter(usable.values())).eph.week
    data = {
        "live": True, "source": args.file or "HackRF One (live)", "week": week,
        "fixes": [{"tow": f.t_rx, "lat": f.lla[0], "lon": f.lla[1], "h": f.lla[2]} for f in fixes],
        "mean": {"lat": lla[:, 0].mean(), "lon": lla[:, 1].mean(), "h": lla[:, 2].mean()},
        "sigma_enu": sig.tolist(), "dop": last.dop, "clock_ppm": ppm,
        "sats": [{"prn": p, "az": last.azel[p][0], "el": last.azel[p][1], "cn0": usable[p].cn0,
                  "time": usable[p].anchor[2], "eph": usable[p].eph_src} for p in sorted(last.azel)],
    }
    tmp = "data/live_fixes.json.tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, default=float)
    os.replace(tmp, "data/live_fixes.json")


def dashboard(args, el_s, ring, src, channels, preds, fix, first_fix_s, clock_off, assisted, status, url, fs):
    pred_el = {p["prn"]: p["el"] for p in preds}
    lines = [f"GPS live receiver  |  {'replay ' + args.file if args.file else 'HackRF One'}  |  "
             f"{'assisted' if assisted else 'unassisted'}  |  stream {el_s:6.1f} s",
             f"clock offset: {'searching...' if clock_off is None else f'{clock_off:+.0f} Hz'}   "
             f"map: {url}", ""]
    lines.append(" PRN   elev   C/N0   lock   time src    ephemeris   behind")
    for prn in sorted(channels):
        ch = channels[prn]
        lag = (ring.written() - ch.pos) / fs
        el = fix.azel[prn][1] if fix is not None and prn in fix.azel else pred_el.get(prn, float("nan"))
        lines.append(f" {prn:3d}  {el:5.0f}°  {ch.cn0:5.0f}   {'yes' if ch.locked else ' - ':4s}  "
                     f"{(ch.anchor[2] if ch.anchor else '-'):10s}  {ch.eph_src or '-':10s}  {lag:5.1f} s")
    if assisted:
        missing = [p["prn"] for p in preds if p["prn"] not in channels]
        if missing:
            lines.append(f" predicted but not acquired yet: {missing}")
    lines.append("")
    if fix is not None:
        lat, lon, h = fix.lla
        lines.append(f" FIX  {lat:.6f}, {lon:.6f}   h {h:.0f} m   {len(fix.residuals)} sats   "
                     f"PDOP {fix.dop['PDOP']:.1f}   first fix after {first_fix_s:.0f} s")
    else:
        lines.append(" no fix yet (need 4 locked satellites with time and ephemeris)")
    if status:
        lines.append(f" {status}")
    sys.stdout.write("\x1b[H\x1b[2J" + "\n".join(lines) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
