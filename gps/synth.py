"""Synthetic GPS L1 C/A baseband signal generator (receive-side testing only).

Gives a known answer for every stage of the receiver: we choose each
satellite's Doppler, code phase, and C/N0, then check that acquisition
and tracking recover them. The output is written to a file and is never
transmitted.

Signal model for one satellite at complex baseband:

    s(t) = A * D(t) * C(t - tau) * exp(j*(2*pi*fd*t + phi))

    D   = navigation data bits, +/-1 at 50 bit/s (20 C/A periods per bit)
    C   = C/A code, chip rate 1.023 MHz * (1 + fd/L1)  (code Doppler)
    fd  = carrier Doppler (satellite motion, up to ~+/-5 kHz, plus receiver clock error)
    A   = amplitude set by C/N0 relative to the noise floor
"""
from dataclasses import dataclass, field

import numpy as np

from .ca_code import CHIP_RATE, CODE_LENGTH, L1_FREQ, ca_code


@dataclass
class SatSignal:
    prn: int
    doppler_hz: float
    code_phase_chips: float  # code phase at t=0, in chips [0, 1023)
    cn0_dbhz: float = 45.0   # typical open-sky values: 40-50 dB-Hz
    nav_bits: np.ndarray = field(default=None, repr=False)


def generate(sats, fs, seconds, noise_std=8.0, seed=0):
    """Return complex64 samples: the sum of `sats` plus complex Gaussian noise.

    `noise_std` is the per-component noise std in int8 units (roughly how a
    HackRF capture looks with sensible gains).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    sigma2 = 2 * noise_std ** 2  # total complex noise power
    out = noise_std * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    for s in sats:
        # C/N0 [Hz] = C / N0, and N0 = sigma2 / fs  =>  C = CN0 * sigma2 / fs
        amp = np.sqrt(10 ** (s.cn0_dbhz / 10) * sigma2 / fs)
        chip_rate = CHIP_RATE * (1 + s.doppler_hz / L1_FREQ)
        chips = s.code_phase_chips + t * chip_rate
        code = ca_code(s.prn)[np.floor(chips).astype(np.int64) % CODE_LENGTH]

        # One nav bit per 20 code periods (20460 chips), aligned to code epochs.
        bit_idx = np.floor(chips / (20 * CODE_LENGTH)).astype(np.int64)
        bits = s.nav_bits
        if bits is None:
            bits = rng.choice([-1, 1], size=bit_idx.max() + 1)
        data = bits[bit_idx % len(bits)]

        phase = 2 * np.pi * s.doppler_hz * t + rng.uniform(0, 2 * np.pi)
        out += amp * data * code * np.exp(1j * phase)

    return out.astype(np.complex64)
