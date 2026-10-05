"""Pseudoranges and the position/time solution (PVT, minus the velocity for now).

Transmit time
    Each subframe's HOW gives the GPS time at which the subframe started
    (TOW*6 - 6). From that edge we count C/A code periods (exactly 1 ms each,
    in satellite time) and the fraction of the current period. That gives
    the time the satellite sent the signal we are receiving at any receiver
    sample N, to within nanoseconds.

Pseudorange
    rho_i = c * (t_rx - t_tx_i). We don't know t_rx (the receiver has no
    clock good enough), so we guess it and solve for the error as a 4th
    unknown. That's why 4 satellites are needed:

        rho_i + c*dt_sv_i - T_i = |r_sat_i - r| + b        (b = c * receiver clock error)

    dt_sv_i is the satellite clock correction from subframe 1, T_i the
    tropospheric delay. The ionosphere (another ~2-10 m) is ignored here.

Earth rotation (Sagnac)
    The satellite position is computed in ECEF at transmit time, but the
    Earth turns by OMEGA_E * tau (~70 ms) while the signal travels. We rotate
    the satellite position into the ECEF frame at receive time, otherwise
    the fix is off by ~30 m.

Least squares
    Linearize around the current guess (start at Earth's center) and iterate:
    H dx = residuals, where H rows are [-unit vector to satellite, 1].
    DOP (dilution of precision) = sqrt(diag((H^T H)^-1)): how geometry
    amplifies range errors into position errors.
"""
from dataclasses import dataclass

import numpy as np

from .navmsg import find_subframes
from .orbit import C, OMEGA_E, ecef_to_lla, sat_clock_bias, sat_position
from .tracking import bit_sync, nav_bits


class TransmitClock:
    """Maps a receiver sample index to the satellite's transmit time (GPS TOW,
    in satellite clock time).

    Two ways to anchor it:
      - from_subframes(): decode a subframe's HOW (needs ~6-8 s of clean bits)
      - from_prediction(): assisted. With a rough position and the receive
        time known from another satellite, predict the transmit time to well
        under 0.5 ms and round to the exact millisecond. C/A code periods start
        on exact GPS milliseconds, so that rounding fixes the time completely.
    """

    def __init__(self, cse, ref_ms, ref_tow, subframes=(), source="?"):
        self.cse = np.asarray(cse)
        self.ref_ms = ref_ms       # code period index ...
        self.ref_tow = ref_tow     # ... and the satellite time at which it started
        self.subframes = list(subframes)
        self.source = source

    @classmethod
    def from_subframes(cls, track):
        off, _ = bit_sync(track["I_P"])
        bits, bit0_ms = nav_bits(track["I_P"], off)
        subframes = [s for s in find_subframes(bits) if s.parity_ok[0] and s.parity_ok[1]]
        if not subframes:
            raise ValueError("no subframe found: cannot determine transmit time")
        s = subframes[0]
        return cls(track["code_start_exact"], bit0_ms + 20 * s.start_bit, s.tow * 6 - 6.0,
                   subframes, source="HOW")

    @classmethod
    def from_prediction(cls, track, eph, rx_ecef, n_ref, t_rx_ref, fs, skip_ms=600):
        """Integer-millisecond resolution.

        n_ref, t_rx_ref : a receiver sample index and its GPS receive time (from another satellite)
        rx_ecef         : rough receiver position (an error of tens of km is fine)
        """
        cse = np.asarray(track["code_start_exact"])
        k = max(skip_ms, int(np.searchsorted(cse, n_ref)) - 1)
        k = min(k, len(cse) - 2)
        t_rx = t_rx_ref + (cse[k] - n_ref) / fs        # receive time of that code period start
        t_tx = t_rx - 0.075
        for _ in range(3):                              # light-time iteration
            pos, e_anom = sat_position(eph, t_tx)
            rng = np.linalg.norm(rotate_earth(pos, 0.075) - rx_ecef)
            t_tx = t_rx - rng / C
        t_sv = t_tx + sat_clock_bias(eph, t_tx, e_anom)  # satellite clock reads GPS time + dt_sv
        return cls(cse, k, round(t_sv * 1000) / 1000, source="predicted")

    def __call__(self, n):
        k = np.searchsorted(self.cse, n) - 1
        if k < 0 or k + 1 >= len(self.cse):
            raise ValueError("sample outside tracked span")
        frac = (n - self.cse[k]) / (self.cse[k + 1] - self.cse[k])
        return self.ref_tow + (k - self.ref_ms + frac) * 1e-3


def receive_time_estimate(clock, eph, rx_ecef, n):
    """GPS receive time at sample n, from one satellite with known transmit time and a rough
    position: t_rx = t_tx + range/c - dt_sv. Good to ~(position error)/c."""
    t_sv = clock(n)
    pos, e_anom = sat_position(eph, t_sv)
    t_tx = t_sv - sat_clock_bias(eph, t_sv, e_anom)
    rng = np.linalg.norm(rotate_earth(pos, 0.075) - rx_ecef)
    return t_tx + rng / C


def rotate_earth(pos, tau):
    """Rotate an ECEF position by Earth's rotation over `tau` seconds."""
    th = OMEGA_E * tau
    c, s = np.cos(th), np.sin(th)
    return np.array([c * pos[0] + s * pos[1], -s * pos[0] + c * pos[1], pos[2]])


def az_el(rx, sat):
    """Azimuth and elevation (deg) of `sat` as seen from receiver `rx` (ECEF)."""
    lat, lon, _ = ecef_to_lla(rx)
    lat, lon = np.radians(lat), np.radians(lon)
    d = sat - rx
    e = -np.sin(lon) * d[0] + np.cos(lon) * d[1]
    n = -np.sin(lat) * np.cos(lon) * d[0] - np.sin(lat) * np.sin(lon) * d[1] + np.cos(lat) * d[2]
    u = np.cos(lat) * np.cos(lon) * d[0] + np.cos(lat) * np.sin(lon) * d[1] + np.sin(lat) * d[2]
    return np.degrees(np.arctan2(e, n)) % 360, np.degrees(np.arctan2(u, np.hypot(e, n)))


def tropo_delay(el_deg):
    """Simple tropospheric slant delay (m): ~2.4 m at zenith, growing toward the horizon."""
    return 2.47 / (np.sin(np.radians(el_deg)) + 0.0121)


@dataclass
class Fix:
    t_rx: float          # GPS TOW of the receive instant (s)
    ecef: np.ndarray
    lla: tuple
    clock_bias_m: float  # receiver clock error * c
    residuals: dict      # prn -> post-fit residual (m)
    dop: dict
    azel: dict           # prn -> (az, el)


def solve(sample, clocks, ephs, use_tropo=True, iters=10):
    """Position fix at receiver sample index `sample`.

    clocks : {prn: TransmitClock}, ephs : {prn: Ephemeris}
    """
    prns = sorted(clocks)
    t_tx = {p: clocks[p](sample) for p in prns}
    t_rx = max(t_tx.values()) + 0.075  # initial guess; the error is absorbed into b

    sats, pr = {}, {}
    for p in prns:
        _, e_anom = sat_position(ephs[p], t_tx[p])
        dt_sv = sat_clock_bias(ephs[p], t_tx[p], e_anom)
        t_corr = t_tx[p] - dt_sv
        sats[p], e_anom = sat_position(ephs[p], t_corr)
        dt_sv = sat_clock_bias(ephs[p], t_corr, e_anom)
        pr[p] = C * (t_rx - t_tx[p]) + C * dt_sv

    x = np.zeros(4)
    for _ in range(iters):
        H, dy, rot = [], [], {}
        for p in prns:
            tau = np.linalg.norm(sats[p] - x[:3]) / C
            rot[p] = rotate_earth(sats[p], tau)
            rng = np.linalg.norm(rot[p] - x[:3])
            trop = 0.0
            if use_tropo and np.linalg.norm(x[:3]) > 6.0e6:
                trop = tropo_delay(max(az_el(x[:3], rot[p])[1], 2.0))
            dy.append(pr[p] - trop - (rng + x[3]))
            H.append(np.r_[-(rot[p] - x[:3]) / rng, 1.0])
        H, dy = np.array(H), np.array(dy)
        dx = np.linalg.lstsq(H, dy, rcond=None)[0]
        x += dx
        if np.linalg.norm(dx[:3]) < 1e-4:
            break

    Q = np.linalg.inv(H.T @ H)
    lat, lon, _ = ecef_to_lla(x[:3])
    sl, cl, so, co = np.sin(np.radians(lat)), np.cos(np.radians(lat)), np.sin(np.radians(lon)), np.cos(np.radians(lon))
    R = np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])  # ECEF -> ENU
    Qenu = R @ Q[:3, :3] @ R.T
    dop = {"GDOP": np.sqrt(np.trace(Q)), "PDOP": np.sqrt(np.trace(Q[:3, :3])),
           "HDOP": np.sqrt(Qenu[0, 0] + Qenu[1, 1]), "VDOP": np.sqrt(Qenu[2, 2]), "TDOP": np.sqrt(Q[3, 3])}
    # final residuals are the dy of the last iteration minus the last correction
    res = dy - H @ dx
    return Fix(t_rx=t_rx - x[3] / C, ecef=x[:3].copy(), lla=ecef_to_lla(x[:3]), clock_bias_m=x[3],
               residuals=dict(zip(prns, res)), dop=dop, azel={p: az_el(x[:3], rot[p]) for p in prns})
