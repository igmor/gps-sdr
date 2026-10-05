"""Minimal RINEX 3 navigation file reader (GPS LNAV records only).

RINEX is the standard exchange format for GNSS data. IGS stations collect
the ephemerides every satellite broadcasts and publish them as daily
"BRDC" files. Each GPS record holds exactly the subframe 1-3 parameters
we decode ourselves in navmsg.py, already scaled to SI units.
"""
import datetime as dt
import re

from .navmsg import Ephemeris

GPS_EPOCH = dt.datetime(1980, 1, 6)
_NUM = re.compile(r"[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?")


def _nums(s):
    return [float(x.replace("D", "e").replace("d", "e")) for x in _NUM.findall(s)]


def read_nav(path):
    """Return {prn: [Ephemeris, ...]} for all GPS records in a RINEX 3 nav file."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = next(k for k, l in enumerate(lines) if "END OF HEADER" in l) + 1
    out = {}
    while i < len(lines):
        l = lines[i]
        if not l.startswith("G") or len(l) < 23:
            i += 1
            continue
        try:
            prn = int(l[1:3])
            y, mo, d, h, mi, s = (int(x) for x in l[4:23].split())
            v = _nums(l[23:]) + [x for ln in lines[i + 1:i + 8] for x in _nums(ln)]
        except ValueError:
            i += 1
            continue
        i += 8
        if len(v) < 28:
            continue
        toc_dt = dt.datetime(y, mo, d, h, mi, s)
        week = int(v[21])
        toc = (toc_dt - GPS_EPOCH).total_seconds() - week * 604800
        if toc < -302400:      # toc in the next week relative to the record's week number
            toc += 604800
        out.setdefault(prn, []).append(Ephemeris(
            prn=prn, af0=v[0], af1=v[1], af2=v[2], iode=int(v[3]), crs=v[4], delta_n=v[5], m0=v[6],
            cuc=v[7], e=v[8], cus=v[9], sqrt_a=v[10], toe=v[11], cic=v[12], omega0=v[13], cis=v[14],
            i0=v[15], crc=v[16], omega=v[17], omega_dot=v[18], idot=v[19], week=week,
            ura=0, health=int(v[24]), tgd=v[25], iodc=int(v[26]), toc=toc))
    return out


def best_ephemeris(ephs, week, tow, max_age=4 * 3600):
    """Healthy ephemeris with toe closest to (week, tow), or None if none is within `max_age`."""
    t = week * 604800 + tow
    best, best_dt = None, max_age
    for e in ephs:
        d = abs(e.week * 604800 + e.toe - t)
        if e.health == 0 and d <= best_dt:
            best, best_dt = e, d
    return best
