# AGENT.md: context for coding agents working on gps-sdr

## Purpose

A from-scratch GPS L1 C/A software receiver in Python, built to **learn** GPS algorithms and
protocols, not to compete with production receivers. Prefer clarity over speed:
- explain the algorithm in docstrings
- validate every stage against an independent truth (synthetic signals with known parameters,
  IGS broadcast/precise orbit files, the spec's tables)

Hardware: HackRF One (r10, firmware 2024.02.1) plus an active GPS patch antenna, powered through
the HackRF bias tee (`-p 1`). Receive only: never transmit on GNSS bands.

## Current state (end of phase 1)

All stages work end to end on real signals.

| Stage | Module | Validated against |
|---|---|---|
| C/A Gold codes | `gps/ca_code.py` | IS-GPS-200 first-10-chips table (all 32 PRNs); Gold autocorrelation values |
| Synthetic signal generator | `gps/synth.py` | used as ground truth by tests |
| Acquisition (FFT parallel code search, code-drift compensated) | `gps/acquisition.py` `acquire()` | synthetic; real sky |
| Assisted targeted acquisition (10 ms coherent) | `acquire_targeted()` | ~3x metric gain over 1 ms non-coherent on synthetic 30-36 dB-Hz signals |
| Tracking (DLL + FLL→Costas PLL, carrier-aided code) | `gps/tracking.py` `Tracker`, `track()` | synthetic C/N0 recovered within 1 dB; real 37-49 dB-Hz locks for 60 s |
| Bit/frame sync, parity, LNAV decode | `gps/navmsg.py` | every field matched the IGS BRDC file exactly (max relative difference 3e-13) |
| Satellite orbit and clock | `gps/orbit.py` | within 2.6-3.7 m of IGS ultra-rapid precise orbits |
| Pseudoranges, Sagnac, troposphere, least squares, DOP | `gps/pvt.py` | fixes agree with the user's true location |
| Integer-ms transmit time (A-GPS) | `TransmitClock.from_prediction()` | reproduced HOW-derived times exactly on 6 satellites |
| IGS orbit download, visibility and Doppler prediction | `gps/agps.py`, `gps/rinex.py` | predicted satellites matched the ones acquired |
| Live streaming (shared-memory ring buffer, worker processes) | `gps/stream.py`, `live.py` | replay and live HackRF; first fix in ~14-16 s |
| Map (Leaflet, fading trail, sky plot; static or live) | `map.py` | headless Chrome screenshots |

Best result so far: an outdoor 60 s capture with A-GPS gave **7 satellites, PDOP 3.2, 1σ scatter of
about 7-8 m horizontal and 14 m vertical**. Cold (unassisted) search found only 5 satellites, PDOP 8.

Entry points: `acquire.py`, `track.py`, `nav.py`, `receiver.py` (batch, `--agps`), `live.py`
(streaming, `--file` replays a capture), `map.py`, `make_synthetic.py`, `scripts/capture.sh`.

Tests: `tests/test_receiver.py` (C/A spec table, Gold autocorrelation, synthetic acquisition, nav parity).
pytest isn't installed on the dev Mac, so run them directly:
`python3 -c "import tests.test_receiver as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"`

## Environment gotchas (learned the hard way)

- **"No HackRF boards found" while macOS lists the device**: macOS USB accessory security. `ioreg -p IOUSB`
  shows devices as `!registered, !matched`. Fix: System Settings → Privacy & Security → "Allow
  accessories to connect", then replug. Reinstalling drivers or GNU Radio does not help.
- **Use the radioconda tools** (`~/radioconda/bin/hackrf_transfer`, version 2024.02.1, which matches the
  firmware). The old MacPorts 2021 copies come first on PATH. `scripts/capture.sh` and `live.py` default
  to radioconda.
- **This HackRF's crystal is about -17.6 ppm**, so every signal appears at about +27 kHz (it drifts
  roughly ±2 kHz with temperature). Use `--center 27000`. The receiver measures this itself
  (`HackRF sample clock` line in receiver.py; the clock offset line in live.py).
- **Code drift**: the same clock error stretches the C/A code by about 17.6 chips/s. Long non-coherent
  acquisition must shift each block's correlation (done in `_search` / `acquire_targeted`).
- **Map tiles from file://**: tile.openstreetmap.org returns HTTP 200 with an "Access blocked" image (it
  needs a Referer, and a localhost referer is also blocked). CARTO now needs an API key. Esri
  `server.arcgisonline.com` works. Blocked tiles still return 200, so always check a screenshot.
- **Live main loop**: tracking channels catching up produce ~40 batches/s each. Drain the queue with a
  time box and concatenate per tick, or the main loop starves (this happened once).
- **Privacy**: `data/` is git-ignored. Captures, fixes.json, live_fixes.json and map.html contain the
  user's location; never commit them or paste coordinates into docs.
- Python here is 3.9 (MacPorts), so avoid 3.10+ syntax.

## Architecture notes

- Sample indices are absolute (samples since the start of the file or stream). Every tracking record
  stores `code_start_exact` (fractional sample where chip 0 starts). Pseudoranges never need the
  nominal sample rate, so the HackRF clock error cancels.
- `TransmitClock` maps sample index → satellite transmit time, anchored either by a decoded HOW
  (`from_subframes`) or by A-GPS integer-ms resolution (`from_prediction`).
- `solve()` in `gps/pvt.py`: Earth-centre start, iterative least squares, Sagnac rotation, simple
  tropo model (2.47 m / (sin el + 0.0121)), **no ionosphere model yet**.
- `live.py` keeps 60 s of tracking history per channel (trimmed in 20 ms multiples to preserve bit
  alignment), stores HOW anchors as absolute ms indices, and writes `data/live_fixes.json` for the
  map page served on 127.0.0.1:8765.

## Possible improvements (rough priority)

1. **Ionosphere**: decode the Klobuchar alpha/beta parameters (subframe 4 page 18), or take them from the RINEX
   header (`GPSA`/`GPSB`), and apply them in `solve()`. Expect 2-10 m less bias.
2. **Carrier smoothing (Hatch filter)**: smooth code pseudoranges with accumulated carrier phase. The PLL
   already has the phase. Should cut fix scatter several times.
3. **Velocity**: solve for velocity and clock drift from the Doppler measurements (the "V" in PVT). Also
   gives a cleaner clock-ppm estimate.
4. **Weighted least squares and outlier rejection**: weight by elevation and C/N0, RAIM-style residual checks.
   Low satellites currently count as much as overhead ones.
5. **Exercise the A-GPS weak-satellite path on real data**: so far every satellite decoded its own HOW.
   Test by forcing `from_prediction` for non-reference channels, or with a weak capture.
6. **Faster first fix**: use `from_prediction` as soon as ONE HOW exists and a prior position is known (live
   already does this). Next, coarse-time navigation (no HOW at all, 5 satellites, solve for time too).
7. **Lower-elevation satellites**: longer coherent integration with data-bit wipe-off (the bits can come from
   a strong satellite or from the network). Also a better antenna position or ground plane.
8. **Performance**: tracking costs ~230 µs per epoch per channel in numpy. Vectorize, add numba, or batch
   several ms per call so 12+ channels run comfortably in real time.
9. **Kalman filter navigation** instead of independent per-epoch least squares (smoother track, handles
   short outages).
10. **More constellations and signals**: Galileo E1 (needs BOC(1,1) and 4 ms codes), GLONASS L1 (FDMA), or
    SBAS corrections. Sample rate and bandwidth limits apply.
11. **Hardware**: a TCXO for the HackRF (±0.5 ppm) would remove the ~27 kHz offset and its drift.
12. **Quality of life**: a pytest setup with real-capture fixtures (small clipped .bin files with the
    location stripped), CI, and a `--decimate`/lower sample-rate option.

## Working conventions

- Comments explain *why* and the physics, matching the existing docstring density.
- Check any change to acquisition, tracking or PVT against `data/gps5.bin` (outdoor, 60 s; not in git).
  Expected: 7 satellites with `--agps`, PDOP about 3.2, clock about -17.6 ppm.
- For anything that renders (maps, plots), take a headless Chrome screenshot and look at it.
