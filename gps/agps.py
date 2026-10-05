"""Assisted GPS: use downloaded orbit data plus a rough position and time to
predict which satellites are up and where to find them.

A cold receiver must search 32 PRNs x +/-(5 kHz Doppler + clock error) x 1023
chips. With assistance (what phones get from the network) it knows:
  - which satellites are above the horizon (no wasted searches)
  - each satellite's Doppler to within a few Hz, so only the receiver's own
    clock error is unknown, and that is common to all satellites
  - the ephemeris, so it need not wait 18-30 s to decode it from the signal
A narrow frequency window makes long coherent integration affordable, which
is what lets assisted receivers pick up much weaker signals.
"""
import datetime as dt
import gzip
import os
import shutil
import time
import urllib.request

import numpy as np

from .ca_code import L1_FREQ
from .orbit import C, sat_position, sat_velocity
from .pvt import az_el, rotate_earth
from .rinex import best_ephemeris, read_nav

GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
LEAP_SECONDS = 18
BKG = "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{y}/{doy:03d}/BRDC00WRD_S_{y}{doy:03d}0000_01D_MN.rnx.gz"


def utc_to_gps(t_utc):
    """datetime (UTC) -> (GPS week, time of week in seconds)."""
    s = (t_utc - GPS_EPOCH).total_seconds() + LEAP_SECONDS
    week = int(s // 604800)
    return week, s - week * 604800


def fetch_brdc(day, cache_dir="data/brdc", max_age_s=3600):
    """Download (or reuse) the IGS merged broadcast ephemeris file for a UTC date.

    The current day's file is updated every 15 min, so a cached copy older
    than `max_age_s` is refreshed. Returns the path, or None if unavailable.
    """
    os.makedirs(cache_dir, exist_ok=True)
    doy = day.timetuple().tm_yday
    path = os.path.join(cache_dir, f"BRDC_{day.year}{doy:03d}.rnx")
    today = dt.datetime.now(dt.timezone.utc).date()
    fresh = os.path.exists(path) and (day < today or time.time() - os.path.getmtime(path) < max_age_s)
    if fresh:
        return path
    url = BKG.format(y=day.year, doy=doy)
    try:
        with urllib.request.urlopen(url, timeout=30) as r, open(path + ".gz", "wb") as fh:
            shutil.copyfileobj(r, fh)
        with gzip.open(path + ".gz") as g, open(path, "wb") as fh:
            shutil.copyfileobj(g, fh)
        os.remove(path + ".gz")
        return path
    except Exception as e:  # network down, file not published yet, ...
        print(f"  (could not download {url}: {e})")
        return path if os.path.exists(path) else None


def load_ephemerides(t_utc, cache_dir="data/brdc"):
    """All GPS ephemerides for the day of `t_utc` and the day before (covers just after midnight)."""
    ephs = {}
    for day in (t_utc.date() - dt.timedelta(days=1), t_utc.date()):
        p = fetch_brdc(day, cache_dir)
        if p:
            for prn, lst in read_nav(p).items():
                ephs.setdefault(prn, []).extend(lst)
    return ephs


def predict(ephs, rx_ecef, week, tow, min_el=5.0):
    """Visible satellites as seen from `rx_ecef` at GPS (week, tow).

    Returns a list of dicts sorted by elevation, with az/el and the
    satellite-induced Doppler (Hz). The receiver clock offset still has
    to be added before searching.
    """
    out = []
    rx = np.asarray(rx_ecef, dtype=float)
    for prn, lst in sorted(ephs.items()):
        e = best_ephemeris(lst, week, tow)
        if e is None:
            continue
        pos, _ = sat_position(e, tow - 0.075)
        pos = rotate_earth(pos, np.linalg.norm(pos - rx) / C)
        az, el = az_el(rx, pos)
        if el < min_el:
            continue
        los = (pos - rx) / np.linalg.norm(pos - rx)
        range_rate = sat_velocity(e, tow - 0.075) @ los   # receiver assumed static on the ground
        out.append({"prn": prn, "az": az, "el": el, "doppler": -range_rate * L1_FREQ / C, "eph": e})
    return sorted(out, key=lambda s: -s["el"])
