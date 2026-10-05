"""GPS L1 C/A navigation message (LNAV) decoding, per IS-GPS-200 section 20.3.

Structure (50 bit/s):
    frame    = 5 subframes, 1500 bits, 30 s
    subframe = 10 words, 300 bits, 6 s
    word     = 30 bits: 24 data bits + 6 parity bits

Every subframe starts with:
    word 1, TLM: preamble 10001011, then telemetry
    word 2, HOW (handover word): 17-bit TOW count (the GPS time, in units of
                 6 s, at which the *next* subframe starts) and a 3-bit subframe ID

Subframe 1 : week number, health, satellite clock correction (af0, af1, af2, toc, TGD)
Subframes 2-3 : ephemeris, the satellite's Keplerian orbit plus corrections
Subframes 4-5 : almanac, ionosphere and UTC parameters (paged over 25 frames)

Parity: an extended (32,26) Hamming code. Each word's parity also uses the
last two bits (D29*, D30*) of the previous word. If D30* = 1, the word's 24
data bits were transmitted inverted, and the decoder must flip them back.
"""
from dataclasses import dataclass

import numpy as np

PREAMBLE = np.array([1, 0, 0, 0, 1, 0, 1, 1], dtype=np.int8)
GPS_PI = 3.1415926535898  # the exact value the spec uses for semicircles -> radians

# Which data bits (1..24) feed each parity bit D25..D30.
_PARITY_TAPS = [
    (1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23),
    (2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24),
    (1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22),
    (2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23),
    (1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24),
    (3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24),
]
_PARITY_PREV = [0, 1, 0, 1, 1, 0]  # 0 -> XOR with D29*, 1 -> XOR with D30*


def check_word(word, d29s, d30s):
    """Check one 30-bit word. Returns (parity_ok, 24 corrected data bits)."""
    data = word[:24] ^ d30s
    prev = (d29s, d30s)
    for k, taps in enumerate(_PARITY_TAPS):
        p = prev[_PARITY_PREV[k]]
        for t in taps:
            p ^= data[t - 1]
        if p != word[24 + k]:
            return False, data
    return True, data


@dataclass
class Subframe:
    start_bit: int        # index in the bit stream where the preamble starts
    id: int               # 1..5
    tow: int              # HOW TOW count: GPS time of the next subframe start = tow * 6 s
    data: np.ndarray      # 300 bits, data bits corrected, parity bits left as received
    parity_ok: list       # per-word parity results


def find_subframes(bits_pm):
    """Find and parity-check all subframes in a +/-1 bit stream.

    Handles the Costas 180-degree ambiguity: the preamble may appear inverted,
    in which case the whole stream is flipped. Only subframes whose TLM and
    HOW words pass parity are returned.
    """
    out = []
    for polarity in (1, -1):
        logic = (bits_pm * polarity > 0).astype(np.int8)
        c = np.correlate(logic * 2 - 1, PREAMBLE * 2 - 1, mode="valid")
        for start in np.nonzero(c == 8)[0]:
            if start < 2 or start + 300 > len(logic):
                continue
            d29s, d30s = int(logic[start - 2]), int(logic[start - 1])
            data = np.empty(300, dtype=np.int8)
            ok = []
            for w in range(10):
                word = logic[start + 30 * w: start + 30 * w + 30]
                good, d = check_word(word, d29s, d30s)
                ok.append(good)
                data[30 * w: 30 * w + 24] = d
                data[30 * w + 24: 30 * w + 30] = word[24:]
                d29s, d30s = int(word[28]), int(word[29])
            if ok[0] and ok[1]:
                out.append(Subframe(start_bit=int(start), id=_u(data, 50, 3),
                                    tow=_u(data, 31, 17), data=data, parity_ok=ok))
    return sorted(out, key=lambda s: s.start_bit)


# --- field extraction (bit numbers are 1-indexed within the subframe, as in the spec) ---

def _u(data, first, n):
    """Unsigned field of `n` bits starting at subframe bit `first`."""
    v = 0
    for b in data[first - 1: first - 1 + n]:
        v = (v << 1) | int(b)
    return v


def _s(data, first, n):
    """Two's-complement signed field."""
    v = _u(data, first, n)
    return v - (1 << n) if v >> (n - 1) else v


def _u2(data, f1, n1, f2, n2):
    """Unsigned field split across two words (MSBs first)."""
    return (_u(data, f1, n1) << n2) | _u(data, f2, n2)


def _s2(data, f1, n1, f2, n2):
    v = _u2(data, f1, n1, f2, n2)
    n = n1 + n2
    return v - (1 << n) if v >> (n - 1) else v


@dataclass
class Ephemeris:
    prn: int
    week: int        # full GPS week (rollover resolved)
    health: int
    ura: int
    iodc: int
    iode: int
    tgd: float       # s
    toc: float       # s of week
    af0: float       # s
    af1: float       # s/s
    af2: float       # s/s^2
    crs: float       # m
    delta_n: float   # rad/s
    m0: float        # rad
    cuc: float       # rad
    e: float
    cus: float       # rad
    sqrt_a: float    # sqrt(m)
    toe: float       # s of week
    cic: float       # rad
    omega0: float    # rad
    cis: float       # rad
    i0: float        # rad
    crc: float       # m
    omega: float     # rad (argument of perigee)
    omega_dot: float  # rad/s
    idot: float      # rad/s


def decode_ephemeris(subframes, prn, week_era=2048):
    """Build an Ephemeris from the most recent subframes 1, 2, 3 with matching
    issue-of-data (IODC low byte == IODE in subframes 2 and 3), all words passing parity.

    The broadcast week number has only 10 bits (it rolls over every 1024 weeks),
    so `week_era` supplies the rollover: 2048 is valid from April 2019 to November 2038.
    """
    good = [s for s in subframes if all(s.parity_ok)]
    by_id = {}
    for s in good:
        by_id.setdefault(s.id, []).append(s)
    if not all(k in by_id for k in (1, 2, 3)):
        return None

    for s1 in reversed(by_id[1]):
        d1 = s1.data
        iodc = _u2(d1, 83, 2, 211, 8)
        for s2 in reversed(by_id[2]):
            for s3 in reversed(by_id[3]):
                d2, d3 = s2.data, s3.data
                iode2, iode3 = _u(d2, 61, 8), _u(d3, 271, 8)
                if iode2 == iode3 == (iodc & 0xFF):
                    return Ephemeris(
                        prn=prn,
                        week=week_era + _u(d1, 61, 10),
                        ura=_u(d1, 73, 4),
                        health=_u(d1, 77, 6),
                        iodc=iodc,
                        iode=iode2,
                        tgd=_s(d1, 197, 8) * 2.0 ** -31,
                        toc=_u(d1, 219, 16) * 2.0 ** 4,
                        af2=_s(d1, 241, 8) * 2.0 ** -55,
                        af1=_s(d1, 249, 16) * 2.0 ** -43,
                        af0=_s(d1, 271, 22) * 2.0 ** -31,
                        crs=_s(d2, 69, 16) * 2.0 ** -5,
                        delta_n=_s(d2, 91, 16) * 2.0 ** -43 * GPS_PI,
                        m0=_s2(d2, 107, 8, 121, 24) * 2.0 ** -31 * GPS_PI,
                        cuc=_s(d2, 151, 16) * 2.0 ** -29,
                        e=_u2(d2, 167, 8, 181, 24) * 2.0 ** -33,
                        cus=_s(d2, 211, 16) * 2.0 ** -29,
                        sqrt_a=_u2(d2, 227, 8, 241, 24) * 2.0 ** -19,
                        toe=_u(d2, 271, 16) * 2.0 ** 4,
                        cic=_s(d3, 61, 16) * 2.0 ** -29,
                        omega0=_s2(d3, 77, 8, 91, 24) * 2.0 ** -31 * GPS_PI,
                        cis=_s(d3, 121, 16) * 2.0 ** -29,
                        i0=_s2(d3, 137, 8, 151, 24) * 2.0 ** -31 * GPS_PI,
                        crc=_s(d3, 181, 16) * 2.0 ** -5,
                        omega=_s2(d3, 197, 8, 211, 24) * 2.0 ** -31 * GPS_PI,
                        omega_dot=_s(d3, 241, 24) * 2.0 ** -43 * GPS_PI,
                        idot=_s(d3, 279, 14) * 2.0 ** -43 * GPS_PI,
                    )
    return None
