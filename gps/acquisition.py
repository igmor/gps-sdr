"""Acquisition: find which satellites are visible and estimate their
Doppler and code phase.

GPS signals are about 20 dB *below* the thermal noise floor, so you
cannot see them in a spectrum plot. They only show up after correlating
with the right PRN code at the right code phase and the right Doppler.
Acquisition is a 2D search over:

    code phase : 1023 chips (= every sample offset within 1 ms)
    Doppler    : +/- a few kHz, in bins about 1/(2*T_coh) wide

Parallel code phase search (FFT-based): circular correlation over all
code phases at once is

    R(tau) = IFFT( FFT(x * e^{-j 2 pi fd t}) * conj(FFT(code)) )

so each Doppler bin costs one FFT/IFFT pair instead of 1023 dot products.

Coherent integration is 1 ms (one code period). To improve sensitivity we
sum |R|^2 over several consecutive ms (non-coherent integration), which
also avoids losing the signal when a nav data bit flips mid-block.
"""
from dataclasses import dataclass

import numpy as np

from .ca_code import CHIP_RATE, CODE_LENGTH, L1_FREQ, sampled_code


@dataclass
class AcqResult:
    prn: int
    detected: bool
    doppler_hz: float
    code_phase_samples: int
    code_phase_chips: float
    metric: float  # highest peak / second-highest peak (outside +/-1 chip)


def acquire(iq, fs, prns=range(1, 33), doppler_max=10e3, doppler_step=None,
            n_noncoh=10, threshold=2.5, if_freq=0.0):
    """Search for each PRN in `prns`. `iq` must hold at least `n_noncoh` ms.

    Doppler bins: with 1 ms coherent integration, a frequency error of
    500 Hz costs about 4 dB (sinc loss), so the default step is
    2/(3*T_coh) ~ 667 Hz, refined afterwards with a finer search.
    """
    spc = int(round(fs * 1e-3))  # samples per code period
    if len(iq) < spc * n_noncoh:
        raise ValueError("not enough samples for the requested integration")
    if doppler_step is None:
        doppler_step = 2 / 3 / 1e-3

    t = np.arange(spc) / fs
    blocks = iq[: spc * n_noncoh].reshape(n_noncoh, spc)
    dopplers = np.arange(-doppler_max, doppler_max + doppler_step / 2, doppler_step)

    results = []
    for prn in prns:
        code_fft_conj = np.conj(np.fft.fft(sampled_code(prn, fs, spc)))
        grid = _search(blocks, t, dopplers, code_fft_conj, if_freq)
        d_idx, c_idx = np.unravel_index(np.argmax(grid), grid.shape)
        metric = _peak_ratio(grid[d_idx], c_idx, fs)

        # Fine Doppler: re-search around the coarse peak in 50 Hz steps.
        fine = np.arange(dopplers[d_idx] - doppler_step, dopplers[d_idx] + doppler_step + 1, 50.0)
        fine_grid = _search(blocks, t, fine, code_fft_conj, if_freq, code_idx=c_idx)
        doppler = fine[np.argmax(fine_grid)]

        # Code phase convention: index where the code *starts* in the block.
        results.append(AcqResult(
            prn=prn,
            detected=metric > threshold,
            doppler_hz=float(doppler),
            code_phase_samples=int(c_idx),
            code_phase_chips=float((-c_idx * CHIP_RATE / fs) % CODE_LENGTH),
            metric=float(metric),
        ))
    return results


def _search(blocks, t, dopplers, code_fft_conj, if_freq, code_idx=None):
    """Non-coherent correlation power over Doppler x code phase.

    Code drift: a frequency offset (satellite Doppler, or the receiver's
    clock error, which drives both the LO and the ADC) also stretches the
    code by the same ratio fd/L1. With a HackRF that is ~17 ppm off, the
    code slides ~0.07 samples per ms, so after 100 ms the peak has moved
    by ~7 samples. To avoid smearing, each block's correlation is shifted
    back by the drift expected for the Doppler bin being tested.

    If `code_idx` is given, return only that code-phase column (1D).
    """
    spc = blocks.shape[1]
    out = np.zeros((len(dopplers), spc))
    for i, fd in enumerate(dopplers):
        wipe = np.exp(-2j * np.pi * (if_freq + fd) * t)
        drift = (if_freq + fd) / L1_FREQ * spc  # samples per code period
        for k, blk in enumerate(blocks):
            corr = np.fft.ifft(np.fft.fft(blk * wipe) * code_fft_conj)
            out[i] += np.roll(np.abs(corr) ** 2, int(round(k * drift)))
    return out if code_idx is None else out[:, code_idx]


def _peak_ratio(row, peak_idx, fs):
    """Highest peak / second-highest peak, ignoring +/-1 chip around the highest."""
    spc = len(row)
    excl = int(np.ceil(fs / CHIP_RATE))
    mask = np.ones(spc, dtype=bool)
    mask[(np.arange(peak_idx - excl, peak_idx + excl + 1)) % spc] = False
    return row[peak_idx] / row[mask].max()


def acquire_targeted(iq, fs, targets, half_window=500.0, n_coh=10, n_noncoh=30, threshold=2.5):
    """Assisted acquisition: search each PRN only near its predicted frequency,
    with `n_coh` ms of *coherent* integration.

    targets : list of (prn, predicted_carrier_hz), predicted Doppler plus receiver clock offset.

    Coherent integration over N ms improves SNR N times (10 dB for 10 ms),
    versus about sqrt(N) for non-coherent. The catches: Doppler bins must
    shrink to ~1/(2*N ms) (50 Hz for 10 ms), which is only affordable over
    a narrow window, and a nav bit flip inside a block cancels part of it
    (about 1 in 4 blocks of 10 ms; the non-coherent sum rides over that).
    """
    spc = int(round(fs * 1e-3))
    need = spc * n_coh * n_noncoh
    if len(iq) < need:
        raise ValueError("not enough samples for the requested integration")
    blocks = iq[:need].reshape(n_noncoh, n_coh, spc)
    t = np.arange(n_coh * spc).reshape(n_coh, spc) / fs   # phase continuous across each coherent block
    step = 1.0 / (2 * n_coh * 1e-3)
    offsets = np.arange(-half_window, half_window + step / 2, step)

    results = []
    for prn, f0 in targets:
        code_fft_conj = np.conj(np.fft.fft(sampled_code(prn, fs, spc)))
        grid = np.zeros((len(offsets), spc))
        for i, df in enumerate(offsets):
            f = f0 + df
            wipe = np.exp(-2j * np.pi * f * t)
            drift = f / L1_FREQ * spc
            corr = np.fft.ifft(np.fft.fft(blocks * wipe, axis=2) * code_fft_conj, axis=2)
            for b in range(n_noncoh):
                acc = np.zeros(spc, dtype=complex)
                for m in range(n_coh):
                    acc += np.roll(corr[b, m], int(round((b * n_coh + m) * drift)))
                grid[i] += np.abs(acc) ** 2
        d_idx, c_idx = np.unravel_index(np.argmax(grid), grid.shape)
        metric = _peak_ratio(grid[d_idx], c_idx, fs)
        results.append(AcqResult(
            prn=prn, detected=metric > threshold, doppler_hz=float(f0 + offsets[d_idx]),
            code_phase_samples=int(c_idx), code_phase_chips=float((-c_idx * CHIP_RATE / fs) % CODE_LENGTH),
            metric=float(metric)))
    return results
