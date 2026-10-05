"""Read/write HackRF sample files.

`hackrf_transfer -r` writes interleaved signed 8-bit I/Q: I0 Q0 I1 Q1 ...
"""
import numpy as np


def read_hackrf(path, fs, seconds=None, offset_seconds=0.0):
    """Load a HackRF capture as complex64, with the DC offset removed.

    The HackRF is a zero-IF receiver, so it has a DC spike at the center
    frequency. That is also the center of the GPS signal, so we subtract
    the mean.
    """
    start = int(offset_seconds * fs) * 2
    count = -1 if seconds is None else int(seconds * fs) * 2
    raw = np.fromfile(path, dtype=np.int8, count=count, offset=start)
    iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    return (iq - iq.mean()).astype(np.complex64)


def write_hackrf(path, iq):
    """Write complex samples in HackRF int8 I/Q format, clipping to [-127, 127]."""
    out = np.empty(2 * len(iq), dtype=np.int8)
    out[0::2] = np.clip(np.round(iq.real), -127, 127)
    out[1::2] = np.clip(np.round(iq.imag), -127, 127)
    out.tofile(path)
