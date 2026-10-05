import numpy as np

from gps.acquisition import acquire
from gps.ca_code import ca_code
from gps.synth import SatSignal, generate


def _first10_octal(prn):
    bits = (1 - ca_code(prn)[:10]) // 2  # back to logic 0/1
    return int("".join(map(str, bits)), 2)


def test_ca_first_chips_match_spec():
    # IS-GPS-200 Table 3-Ia, "First 10 chips C/A" (octal).
    expected = [0o1440, 0o1620, 0o1710, 0o1744, 0o1133, 0o1455, 0o1131, 0o1454,
                0o1626, 0o1504, 0o1642, 0o1750, 0o1764, 0o1772, 0o1775, 0o1776,
                0o1156, 0o1467, 0o1633, 0o1715, 0o1746, 0o1763, 0o1063, 0o1706,
                0o1743, 0o1761, 0o1770, 0o1774, 0o1127, 0o1453, 0o1625, 0o1712]
    for prn, octal in enumerate(expected, start=1):
        assert _first10_octal(prn) == octal, prn


def test_ca_autocorrelation_is_gold_like():
    c = ca_code(7).astype(np.int32)
    acf = np.array([np.dot(c, np.roll(c, k)) for k in range(1023)])
    assert acf[0] == 1023
    # Gold codes: off-peak values in {-65, -1, 63}
    assert set(acf[1:]) <= {-65, -1, 63}


def test_acquisition_recovers_synthetic_sats():
    fs = 4e6
    truth = [SatSignal(5, 2300.0, 333.0, 45), SatSignal(19, -1800.0, 900.5, 42)]
    iq = generate(truth, fs, 0.012)
    res = {r.prn: r for r in acquire(iq, fs, prns=[5, 19, 30], n_noncoh=10)}
    for s in truth:
        r = res[s.prn]
        assert r.detected
        assert abs(r.doppler_hz - s.doppler_hz) <= 100
        err = (r.code_phase_chips - s.code_phase_chips + 511.5) % 1023 - 511.5
        assert abs(err) < 0.5
    assert not res[30].detected


def _encode_word(data24, d29s, d30s):
    """Reference encoder: build the 30 transmitted bits for 24 data bits."""
    from gps.navmsg import _PARITY_PREV, _PARITY_TAPS
    prev = (d29s, d30s)
    parity = []
    for k, taps in enumerate(_PARITY_TAPS):
        p = prev[_PARITY_PREV[k]]
        for t in taps:
            p ^= int(data24[t - 1])
        parity.append(p)
    return np.concatenate([data24 ^ d30s, parity]).astype(np.int8)


def test_nav_parity_roundtrip_and_error_detection():
    from gps.navmsg import check_word
    rng = np.random.default_rng(1)
    for _ in range(200):
        data = rng.integers(0, 2, 24).astype(np.int8)
        d29s, d30s = rng.integers(0, 2, 2)
        word = _encode_word(data, int(d29s), int(d30s))
        ok, decoded = check_word(word, int(d29s), int(d30s))
        assert ok and np.array_equal(decoded, data)
        bad = word.copy()
        bad[rng.integers(0, 30)] ^= 1
        assert not check_word(bad, int(d29s), int(d30s))[0]  # any single-bit error is caught
