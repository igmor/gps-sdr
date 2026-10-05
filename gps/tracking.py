"""Tracking: lock onto one satellite and follow its code and carrier.

Acquisition gives a rough code phase (+/- half a sample) and Doppler
(+/- ~100 Hz). Tracking refines both continuously, one code period
(1 ms) at a time, with two feedback loops:

DLL (delay lock loop), for code timing
    Correlate with three copies of the code: Early (-0.5 chip), Prompt,
    Late (+0.5 chip). When aligned, |E| == |L|. The discriminator
    (|E|-|L|)/(|E|+|L|) says which way the code is slipping. The code rate
    is carrier-aided: scaled by (1 + f_carrier/L1), since the same
    Doppler/clock error that shifts the carrier also stretches the code.

Carrier loop, for frequency and phase
    1. FLL first (frequency lock loop): measures the phase rotation between
       consecutive prompt correlations, atan(cross/dot)/(2*pi*T). This pulls
       in large frequency errors.
    2. Then a Costas PLL (phase lock loop): atan(Q/I) is the phase error.
       The Costas form ignores 180-degree flips, so the 50 bit/s
       navigation data (BPSK) does not disturb the lock.

Once the PLL is locked, all the signal energy is in I (in-phase), and
the sign of I_prompt is the navigation data bit. It stays constant for
20 ms, then may flip.
"""
from dataclasses import dataclass

import numpy as np

from .ca_code import CHIP_RATE, CODE_LENGTH, L1_FREQ, ca_code


@dataclass
class TrackResult:
    prn: int
    t: np.ndarray            # time of each epoch (s from file start)
    code_start: np.ndarray   # absolute sample index of the first sample of each block
    code_start_exact: np.ndarray  # fractional sample index where code chip 0 begins (for pseudoranges)
    I_E: np.ndarray
    I_P: np.ndarray
    Q_P: np.ndarray
    I_L: np.ndarray
    carr_freq: np.ndarray    # Hz (includes receiver clock offset)
    code_freq: np.ndarray    # chips/s
    dll_err: np.ndarray
    pll_err: np.ndarray      # cycles
    fll_mode: np.ndarray     # True while the FLL is pulling in


def _loop_coeffs(noise_bw, zeta, gain):
    """Second-order loop filter (proportional + integral) coefficients."""
    wn = noise_bw * 8 * zeta / (4 * zeta ** 2 + 1)
    return gain / wn ** 2, 2 * zeta / wn  # tau1, tau2


RECORD_FIELDS = ("code_start", "code_start_exact", "t", "I_E", "I_P", "Q_P", "I_L",
                 "carr_freq", "code_freq", "dll_err", "pll_err", "fll_mode")


class Tracker:
    """Stateful tracking channel: feed it one code period of samples at a time.

    Used both for files (`track()` below) and for live streaming, where the
    samples come from a shared ring buffer.

        n = tr.next_block()           # how many samples the next code period needs
        rec = tr.update(x)            # x = complex samples [tr.pos, tr.pos + n)
    """

    def __init__(self, prn, fs, start_sample, carr_freq, dll_bw=2.0, pll_bw=15.0, fll_bw=10.0,
                 fll_ms=500, el_spacing=0.5):
        code = ca_code(prn)
        self.code_ext = np.concatenate([code[-1:], code, code[:2]])  # room for early/late wraparound
        self.prn, self.fs, self.el = prn, fs, el_spacing
        self.fll_bw, self.fll_ms = fll_bw, fll_ms
        self.tau1c, self.tau2c = _loop_coeffs(dll_bw, 0.7, 1.0)
        self.tau1p, self.tau2p = _loop_coeffs(pll_bw, 0.7, 0.25)
        self.pos = int(start_sample)
        self.rem_code = self.rem_carr = 0.0
        self.code_nco = self.old_code_err = 0.0
        self.carr_nco = self.old_carr_err = 0.0
        self.carr_freq = self.carr_base = carr_freq
        self.code_freq = CHIP_RATE * (1 + carr_freq / L1_FREQ)
        self.prev_p = None
        self.k = 0  # epochs processed

    def next_block(self):
        self.step = self.code_freq / self.fs
        return int(np.ceil((CODE_LENGTH - self.rem_code) / self.step))

    def update(self, x):
        """Process one code period. `x` must have exactly next_block() samples."""
        T, fs, step, blk = 1e-3, self.fs, self.step, len(x)

        # Carrier wipe-off
        n = np.arange(blk)
        phase = self.rem_carr + 2 * np.pi * self.carr_freq * n / fs
        bb = x * np.exp(-1j * phase)
        self.rem_carr = (phase[-1] + 2 * np.pi * self.carr_freq / fs) % (2 * np.pi)

        # Code wipe-off: early / prompt / late
        rem_code_at_start = self.rem_code
        chips = self.rem_code + step * n
        ce = self.code_ext
        e = bb @ ce[np.floor(chips - self.el).astype(np.int64) + 1]
        p = bb @ ce[np.floor(chips).astype(np.int64) + 1]       # +1 for the code_ext prefix
        l = bb @ ce[np.floor(chips + self.el).astype(np.int64) + 1]
        self.rem_code = chips[-1] + step - CODE_LENGTH

        # Carrier loop
        in_fll = self.k < self.fll_ms
        if in_fll:
            if self.prev_p is not None:
                pp = self.prev_p
                dot = pp.real * p.real + pp.imag * p.imag
                cross = pp.real * p.imag - pp.imag * p.real
                f_err = np.arctan(cross / dot) / (2 * np.pi * T) if dot != 0 else 0.0
                self.carr_freq += 4 * self.fll_bw * T * f_err  # first-order FLL
            self.carr_base = self.carr_freq
            carr_err = 0.0
        else:
            carr_err = np.arctan(p.imag / p.real) / (2 * np.pi) if p.real != 0 else 0.0
            self.carr_nco += self.tau2p / self.tau1p * (carr_err - self.old_carr_err) + carr_err * T / self.tau1p
            self.old_carr_err = carr_err
            self.carr_freq = self.carr_base + self.carr_nco
        self.prev_p = p

        # Code loop (carrier-aided)
        ae, al = abs(e), abs(l)
        code_err = (ae - al) / (ae + al)
        self.code_nco += self.tau2c / self.tau1c * (code_err - self.old_code_err) + code_err * T / self.tau1c
        self.old_code_err = code_err
        self.code_freq = CHIP_RATE * (1 + self.carr_freq / L1_FREQ) - self.code_nco

        rec = (self.pos, self.pos - rem_code_at_start / step, self.pos / fs, e.real, p.real, p.imag, l.real,
               self.carr_freq, self.code_freq, code_err, carr_err, in_fll)
        self.pos += blk
        self.k += 1
        return rec


def track(raw, fs, prn, start_sample, carr_freq, n_ms, dc=0j, **loop_args):
    """Track `prn` for `n_ms` code periods from a file.

    raw          : int8 array of interleaved I/Q (np.memmap is fine)
    start_sample : sample index where a C/A code period starts (from acquisition)
    carr_freq    : initial carrier frequency estimate (Hz)
    dc           : DC offset to subtract (the HackRF zero-IF spike)
    """
    tr = Tracker(prn, fs, start_sample, carr_freq, **loop_args)
    recs = []
    for _ in range(n_ms):
        blk = tr.next_block()
        if 2 * (tr.pos + blk) > len(raw):
            break
        seg = raw[2 * tr.pos: 2 * (tr.pos + blk)].astype(np.float32)
        recs.append(tr.update((seg[0::2] + 1j * seg[1::2]) - dc))
    cols = list(zip(*recs))
    out = {name: np.array(col) for name, col in zip(RECORD_FIELDS, cols)}
    out["code_start"] = out["code_start"].astype(np.int64)
    out["fll_mode"] = out["fll_mode"].astype(bool)
    return TrackResult(prn=prn, **out)


def cn0_estimate(I_P, Q_P, T=1e-3, window=1000):
    """C/N0 in dB-Hz over consecutive windows (needs PLL lock).

    With phase lock, I = A*d + noise and Q = noise, so
    SNR per epoch = mean(|I|)^2 / var(Q) = 2*C/N0*T, giving C/N0 = SNR / (2T).
    """
    n = len(I_P) // window
    out = np.empty(n)
    for i in range(n):
        sl = slice(i * window, (i + 1) * window)
        snr = np.mean(np.abs(I_P[sl])) ** 2 / np.var(Q_P[sl])
        out[i] = 10 * np.log10(max(snr, 1e-9) / (2 * T))
    return out


def bit_sync(I_P, skip_ms=1000):
    """Find where the 20 ms nav bits start, from where the sign of I_P flips.

    Returns (offset in [0, 20), histogram of transitions by ms mod 20).
    """
    s = np.sign(I_P[skip_ms:])
    flips = np.nonzero(s[1:] != s[:-1])[0] + 1 + skip_ms
    hist = np.bincount(flips % 20, minlength=20)
    return int(np.argmax(hist)), hist


def nav_bits(I_P, bit_offset, skip_ms=1000):
    """Sum I_P over each 20 ms bit and return the signs as +1/-1."""
    start = skip_ms + ((bit_offset - skip_ms) % 20)
    n = (len(I_P) - start) // 20
    sums = I_P[start: start + 20 * n].reshape(n, 20).sum(axis=1)
    return np.where(sums > 0, 1, -1).astype(np.int8), start
