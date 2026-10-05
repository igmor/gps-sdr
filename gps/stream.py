"""Live streaming plumbing: a shared-memory ring buffer of HackRF samples,
sample sources (HackRF or file replay), and the worker processes that
acquire and track from it.

    source thread --> RingBuffer (shared memory, ~30 s) --> acquisition worker (1 process)
                                                       \\-> tracking channel  (1 process per satellite)

Every sample has an absolute index (samples since the stream started), so
channels, acquisition and the position solver all agree on time. The
ring buffer holds enough history that a newly acquired satellite can start
tracking at the exact sample where acquisition found it, then catch up
(tracking runs ~4x faster than real time per channel).
"""
import queue
import subprocess
import threading
import time
from multiprocessing import shared_memory

import numpy as np

from .acquisition import acquire, acquire_targeted
from .tracking import RECORD_FIELDS, Tracker

HEADER = 64  # bytes reserved in front of the samples for the write counter


class Overrun(Exception):
    """The requested samples were already overwritten: the reader fell too far behind."""


class RingBuffer:
    """Single-writer, multi-reader ring of interleaved int8 I/Q in shared memory."""

    def __init__(self, capacity, name=None):
        self.cap = int(capacity)
        create = name is None
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=HEADER + 2 * self.cap)
        self.name = self.shm.name
        self.counter = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf, offset=0)
        self.data = np.ndarray((2 * self.cap,), dtype=np.int8, buffer=self.shm.buf, offset=HEADER)
        if create:
            self.counter[0] = 0

    def written(self):
        return int(self.counter[0])

    def write(self, chunk):
        """Append interleaved int8 I/Q (even length)."""
        n = len(chunk) // 2
        i = (self.written() % self.cap) * 2
        first = min(len(chunk), 2 * self.cap - i)
        self.data[i:i + first] = chunk[:first]
        if first < len(chunk):
            self.data[:len(chunk) - first] = chunk[first:]
        self.counter[0] += n  # publish only after the data is in place

    def read(self, a, b):
        """Samples [a, b) as interleaved int8, None if not written yet; raises Overrun if gone."""
        w = self.written()
        if b > w:
            return None
        if a < w - self.cap:
            raise Overrun
        i, j = (a % self.cap) * 2, (a % self.cap) * 2 + 2 * (b - a)
        out = self.data[i:j].copy() if j <= 2 * self.cap else \
            np.concatenate([self.data[i:], self.data[:j - 2 * self.cap]])
        if a < self.written() - self.cap:  # overwritten while we copied
            raise Overrun
        return out

    def read_complex(self, a, b, dc=0j):
        raw = self.read(a, b)
        if raw is None:
            return None
        f = raw.astype(np.float32)
        return (f[0::2] + 1j * f[1::2]) - dc

    def close(self, unlink=False):
        self.shm.close()
        if unlink:
            self.shm.unlink()


# ---------------------------------------------------------------- sources

class HackRFSource(threading.Thread):
    """Runs `hackrf_transfer -r -` and copies its stdout into the ring buffer."""

    def __init__(self, ring, hackrf_transfer, freq, fs, lna=32, vga=40, amp=True, bias=True, log="data/hackrf_live.log"):
        super().__init__(daemon=True)
        self.ring, self.stop_flag, self.error = ring, threading.Event(), None
        cmd = [hackrf_transfer, "-r", "-", "-f", str(int(freq)), "-s", str(int(fs)),
               "-l", str(lna), "-g", str(vga), "-a", "1" if amp else "0", "-p", "1" if bias else "0"]
        self.log = open(log, "w")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)

    def run(self):
        buf = bytearray(1 << 18)
        mv = memoryview(buf)
        carry = b""
        try:
            while not self.stop_flag.is_set():
                n = self.proc.stdout.readinto(buf)
                if not n:
                    self.error = f"hackrf_transfer exited (code {self.proc.poll()}), see {self.log.name}"
                    break
                data = carry + bytes(mv[:n]) if carry else mv[:n]
                even = len(data) - (len(data) % 2)
                self.ring.write(np.frombuffer(data[:even], dtype=np.int8))
                carry = bytes(data[even:])
        finally:
            self.stop()

    def stop(self):
        self.stop_flag.set()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class FileSource(threading.Thread):
    """Replays a HackRF capture into the ring buffer at real-time speed (for testing)."""

    def __init__(self, ring, path, fs, speed=1.0, chunk_s=0.02):
        super().__init__(daemon=True)
        self.ring, self.path, self.fs, self.speed = ring, path, fs, speed
        self.chunk = int(fs * chunk_s) * 2
        self.stop_flag, self.error = threading.Event(), None

    def run(self):
        t0 = time.time()
        sent = 0
        with open(self.path, "rb") as fh:
            while not self.stop_flag.is_set():
                b = fh.read(self.chunk)
                if len(b) < 2:
                    self.error = "end of file"
                    break
                self.ring.write(np.frombuffer(b[: len(b) - len(b) % 2], dtype=np.int8))
                sent += len(b) // 2
                ahead = sent / self.fs / self.speed - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)

    def stop(self):
        self.stop_flag.set()


# ---------------------------------------------------------------- worker processes

def channel_worker(ring_name, cap, fs, prn, start, carrier, dc, out_q, stop_ev, batch=100):
    """Track one satellite from the ring buffer, sending batches of records to `out_q`."""
    ring = RingBuffer(cap, name=ring_name)
    tr = Tracker(prn, fs, start, carrier)
    recs = []
    try:
        while not stop_ev.is_set():
            n = tr.next_block()
            try:
                x = ring.read_complex(tr.pos, tr.pos + n, dc)
            except Overrun:
                out_q.put(("lost", prn, "fell behind the ring buffer"))
                return
            if x is None:
                if recs:
                    out_q.put(("trk", prn, np.array(recs, dtype=np.float64), tr.pos))
                    recs = []
                time.sleep(0.005)
                continue
            recs.append(tr.update(x))
            if len(recs) >= batch:
                out_q.put(("trk", prn, np.array(recs, dtype=np.float64), tr.pos))
                recs = []
    finally:
        ring.close()


def acq_worker(ring_name, cap, fs, cmd_q, out_q, stop_ev):
    """Run acquisition jobs on the most recent samples.

    Jobs: ("wide", [(prn, center_hz, half_range_hz)], n_ms)
          ("targeted", [(prn, center_hz)], half_window, n_coh, n_noncoh)
    Each PRN's result is reported as soon as it is computed.
    """
    ring = RingBuffer(cap, name=ring_name)
    spc = int(round(fs * 1e-3))
    try:
        while not stop_ev.is_set():
            try:
                job = cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue
            kind = job[0]
            need = spc * (job[2] + 2 if kind == "wide" else job[3] * job[4] + 2)
            while ring.written() < need + int(fs) and not stop_ev.is_set():
                time.sleep(0.1)
            start = ring.written() - need
            head = ring.read_complex(max(0, start - int(fs // 4)), start)
            dc = head.mean() if head is not None and len(head) else 0j
            iq = ring.read_complex(start, start + need, dc)
            if iq is None:
                continue
            if kind == "wide":
                for prn, center, half in job[1]:
                    r = acquire(iq, fs, prns=[prn], doppler_max=half, n_noncoh=job[2], if_freq=center)[0]
                    out_q.put(("acq", prn, start + r.code_phase_samples, r.doppler_hz + center, r.metric,
                               r.detected, kind))
            else:
                for prn, center in job[1]:
                    r = acquire_targeted(iq, fs, [(prn, center)], half_window=job[2], n_coh=job[3],
                                         n_noncoh=job[4])[0]
                    out_q.put(("acq", prn, start + r.code_phase_samples, r.doppler_hz, r.metric,
                               r.detected, kind))
            out_q.put(("acq_done", kind))
    finally:
        ring.close()


__all__ = ["RingBuffer", "Overrun", "HackRFSource", "FileSource", "channel_worker", "acq_worker",
           "RECORD_FIELDS"]
