"""Satellite position and clock from broadcast ephemeris (IS-GPS-200 Table 20-IV),
plus coordinate helpers.

The orbit is a Kepler ellipse (semi-major axis A, eccentricity e, inclination
i0, right ascension Omega0, argument of perigee omega, mean anomaly M0 at
reference time toe), with linear rates (delta_n, omega_dot, idot) and six
harmonic correction terms (C_rs/C_rc for radius, C_us/C_uc for argument of
latitude, C_is/C_ic for inclination). These absorb Earth's oblateness, lunar
and solar pull, etc. over the roughly 4-hour fit interval.

Output is ECEF (Earth-centered, Earth-fixed) WGS-84 coordinates: the
frame rotates with the Earth, so the position comes out directly in
"ground" coordinates.
"""
import numpy as np

MU = 3.986005e14              # WGS-84 Earth gravitational parameter, m^3/s^2 (GPS value)
OMEGA_E = 7.2921151467e-5     # Earth rotation rate, rad/s
F_REL = -4.442807633e-10      # relativistic clock constant, s/sqrt(m)
C = 299792458.0               # speed of light, m/s
HALF_WEEK = 302400.0


def _wrap_week(dt):
    """Time differences across the week boundary (spec: keep within +/-302400 s)."""
    if dt > HALF_WEEK:
        return dt - 2 * HALF_WEEK
    if dt < -HALF_WEEK:
        return dt + 2 * HALF_WEEK
    return dt


def sat_position(eph, t):
    """ECEF position (m) of the satellite at GPS time-of-week `t` (s), plus the
    eccentric anomaly (needed for the relativistic clock term)."""
    a = eph.sqrt_a ** 2
    n = np.sqrt(MU / a ** 3) + eph.delta_n           # corrected mean motion
    tk = _wrap_week(t - eph.toe)
    m = eph.m0 + n * tk                              # mean anomaly

    e_anom = m                                       # Kepler's equation M = E - e sin E
    for _ in range(30):
        e_next = m + eph.e * np.sin(e_anom)
        if abs(e_next - e_anom) < 1e-13:
            break
        e_anom = e_next
    e_anom = e_next

    nu = np.arctan2(np.sqrt(1 - eph.e ** 2) * np.sin(e_anom), np.cos(e_anom) - eph.e)  # true anomaly
    phi = nu + eph.omega                             # argument of latitude
    s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
    u = phi + eph.cus * s2 + eph.cuc * c2
    r = a * (1 - eph.e * np.cos(e_anom)) + eph.crs * s2 + eph.crc * c2
    i = eph.i0 + eph.cis * s2 + eph.cic * c2 + eph.idot * tk

    xp, yp = r * np.cos(u), r * np.sin(u)            # position in the orbital plane
    om = eph.omega0 + (eph.omega_dot - OMEGA_E) * tk - OMEGA_E * eph.toe  # longitude of ascending node
    x = xp * np.cos(om) - yp * np.cos(i) * np.sin(om)
    y = xp * np.sin(om) + yp * np.cos(i) * np.cos(om)
    z = yp * np.sin(i)
    return np.array([x, y, z]), e_anom


def sat_clock_bias(eph, t, e_anom):
    """Satellite clock offset (s): polynomial + relativistic term - group delay (L1 C/A)."""
    dt = _wrap_week(t - eph.toc)
    rel = F_REL * eph.e * eph.sqrt_a * np.sin(e_anom)
    return eph.af0 + eph.af1 * dt + eph.af2 * dt ** 2 + rel - eph.tgd


def sat_velocity(eph, t, h=0.5):
    """ECEF velocity (m/s) by central difference."""
    return (sat_position(eph, t + h)[0] - sat_position(eph, t - h)[0]) / (2 * h)


def ecef_to_lla(xyz):
    """ECEF (m) -> geodetic latitude, longitude (deg), height (m) on WGS-84."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    x, y, z = xyz
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + h)))
    return np.degrees(lat), np.degrees(lon), h


def lla_to_ecef(lat_deg, lon_deg, h=0.0):
    """Geodetic latitude/longitude (deg) and height (m) -> ECEF (m), WGS-84."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - e2) + h) * np.sin(lat)])
