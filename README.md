# gps-sdr: a GPS L1 C/A receiver in Python, for learning

Receive-only. Uses a HackRF One plus an active GPS antenna. The reference spec is IS-GPS-200.

## Pipeline

| # | Stage | What you learn | Status |
|---|-------|----------------|--------|
| 1 | C/A code generation (`gps/ca_code.py`) | Gold codes, LFSRs, CDMA | done |
| 2 | Signal model and synthetic generator (`gps/synth.py`) | BPSK, Doppler, C/N0, signals below the noise floor | done |
| 3 | Acquisition (`gps/acquisition.py`) | FFT correlation, 2D Doppler x code-phase search | done |
| 4 | Tracking (`gps/tracking.py`, `track.py`) | DLL (code), Costas PLL/FLL (carrier), early/prompt/late correlators | done |
| 5 | Bit and frame sync | 20 ms bit edges, TLM preamble `10001011`, HOW, (32,26) Hamming parity | done (`gps/navmsg.py`) |
| 6 | Navigation message decoding (`gps/navmsg.py`, `nav.py`) | subframes 1-3: clock corrections and Keplerian ephemeris | done; matches IGS broadcast file exactly |
| 7 | Satellite position and clock (`gps/orbit.py`) | orbit propagation, relativistic correction, Earth rotation (Sagnac) | done; within ~3 m of IGS precise orbit |
| 8 | Pseudoranges (`gps/pvt.py`) | transmit time from TOW plus code phase; receive time | done |
| 9 | Position solution (`gps/pvt.py`, `receiver.py`) | iterative least squares for (x, y, z, clock bias), DOP, ECEF to lat/lon/alt | done (no velocity or ionosphere model yet) |

## Quick start

```bash
python make_synthetic.py data/synth.bin --seconds 1
python acquire.py data/synth.bin --offset 0.1

# real signal (antenna outdoors / at a window with clear sky)
scripts/capture.sh data/gps.bin 40
python acquire.py data/gps.bin --center 27000 --doppler 8000 --ms 200   # this HackRF's clock is ~17.6 ppm slow -> signals appear ~+27 kHz
python track.py data/gps.bin --prn 7 --center 27000
python nav.py data/track_prn7.npz

# everything end to end: capture -> position
python receiver.py data/gps.bin --center 27000
```

## Live, assisted (A-GPS), and the map

```bash
python live.py --center 27000                        # stream from the HackRF; map at http://127.0.0.1:8765/live_map.html
python live.py --file data/gps5.bin --center 27000   # replay a capture in real time (testing without the antenna)
python receiver.py data/gps5.bin --center 27000 --agps && python map.py
```

- `gps/stream.py`: shared-memory ring buffer, HackRF/file sources, acquisition and tracking worker processes.
- `gps/agps.py`, `gps/rinex.py`: download IGS broadcast orbits, predict visible satellites and their Doppler;
  targeted 10 ms coherent search (`acquire_targeted`); integer-ms transmit time (`TransmitClock.from_prediction`).
- `map.py`: fading trail (slider), mean and scatter of visible fixes, sky plot. Same page live or static.

## HackRF notes
- The stock crystal is +/-20 ppm, and 1 ppm is about 1.6 kHz at L1. If no satellites appear,
  widen the search with `--doppler 30000`. All satellites will share roughly the same offset,
  which is the receiver's clock error.
- `-p 1` powers the antenna's LNA through the coax. Use it only with an active antenna.
- Captures are int8 I/Q, 8 MB/s at 4 MS/s, so 40 s is about 320 MB.
