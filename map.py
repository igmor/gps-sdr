#!/usr/bin/env python3
"""Render receiver fixes as an interactive map (single HTML file, Leaflet + Esri tiles).

    python receiver.py data/gps5.bin --center 27000   # writes data/fixes.json
    python map.py                                     # writes data/map.html and opens it

live.py uses write_live_page(): the same page, but it re-fetches
live_fixes.json every 2 s from live.py's local web server.

Note: opening the page fetches map tiles from Esri's servers, so the tile
server sees which area you are viewing.
"""
import argparse
import json
import subprocess

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GPS fix</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {
    --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --grid: #e4e3df;
    --series-1: #2a78d6; --good: #0ca30c; --warning: #fab219;
  }
  @media (prefers-color-scheme: dark) {
    :root { --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --grid: #3a3a37; --series-1: #3987e5; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--surface); color: var(--ink); display: flex; height: 100vh; }
  #map { flex: 1; }
  aside { width: 350px; padding: 18px 20px; overflow-y: auto; border-left: 1px solid var(--grid); }
  h1 { font-size: 16px; margin: 0 0 2px; display: flex; align-items: center; gap: 8px; }
  h2 { font-size: 12px; font-weight: 600; color: var(--ink-2); text-transform: uppercase;
       letter-spacing: .04em; margin: 20px 0 8px; }
  .sub { color: var(--ink-2); margin: 0; }
  .hero { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; margin: 10px 0 0; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  td, th { padding: 3px 0; text-align: right; }
  td:first-child, th:first-child { text-align: left; color: var(--ink-2); font-weight: normal; }
  th { color: var(--ink-2); font-weight: normal; border-bottom: 1px solid var(--grid); }
  .legend { display: grid; grid-template-columns: 22px 1fr; gap: 6px 8px; align-items: center; }
  .note { color: var(--ink-2); font-size: 12px; }
  .trail { display: flex; align-items: center; gap: 10px; }
  .trail input { flex: 1; accent-color: var(--series-1); }
  .trail span { font-variant-numeric: tabular-nums; min-width: 48px; text-align: right; }
  .badge { font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 9px; border: 1px solid var(--grid);
           color: var(--ink-2); display: none; }
  .badge.on { display: inline-flex; align-items: center; gap: 5px; }
  .badge .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--good); }
  .badge.stale .dot { background: var(--warning); }
  svg text { fill: var(--ink-2); font-size: 10px; }
  svg .lbl { fill: var(--ink); font-size: 11px; font-weight: 600; }
  .tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
         border: 1px solid var(--grid); border-radius: 6px; padding: 6px 8px; font-size: 12px;
         box-shadow: 0 2px 8px rgba(0,0,0,.12); display: none; z-index: 1000; }
  .leaflet-tooltip { font: 12px/1.4 -apple-system, sans-serif; }
</style>
</head>
<body>
<div id="map" role="img" aria-label="Map of GPS position fixes"></div>
<aside>
  <h1>GPS position fix <span class="badge" id="live"><span class="dot"></span><span id="livetxt">LIVE</span></span></h1>
  <p class="sub" id="subtitle"></p>
  <p class="hero" id="latlon"></p>
  <p class="sub" id="height"></p>

  <h2>Trail</h2>
  <div class="trail">
    <input type="range" id="trail" min="10" max="300" step="10" aria-label="Trail length in seconds">
    <span id="trailval"></span>
  </div>
  <p class="note">Newer fixes are brighter; fixes older than the trail length disappear.
     Mean and scatter use only the visible fixes.</p>

  <h2>Legend</h2>
  <div class="legend">
    <svg width="22" height="14"><circle cx="11" cy="7" r="4" fill="var(--series-1)" stroke="var(--surface)" stroke-width="2"/></svg>
    <span id="lg-fixes">Fixes</span>
    <svg width="22" height="14"><circle cx="11" cy="7" r="6" fill="var(--ink)" stroke="var(--surface)" stroke-width="2"/></svg>
    <span id="lg-mean">Mean position</span>
    <svg width="22" height="14"><circle cx="11" cy="7" r="6" fill="none" stroke="var(--ink-2)" stroke-width="1.5" stroke-dasharray="3 2"/></svg>
    <span id="lg-r95">2DRMS</span>
  </div>

  <h2>Precision</h2>
  <table>
    <tr><td>Scatter east (1&sigma;)</td><td id="se"></td></tr>
    <tr><td>Scatter north (1&sigma;)</td><td id="sn"></td></tr>
    <tr><td>Scatter up (1&sigma;)</td><td id="su"></td></tr>
    <tr><td>HDOP / VDOP / PDOP</td><td id="dops"></td></tr>
    <tr><td>HackRF clock error</td><td id="ppm"></td></tr>
  </table>
  <p class="note">Scatter is the spread between fixes (precision), not the error from your true location (accuracy).</p>

  <h2 id="skytitle">Sky</h2>
  <svg id="sky" viewBox="0 0 300 300" width="300" height="300" role="img" aria-label="Sky plot of satellites"></svg>
  <table id="sattable"></table>
</aside>
<div class="tip" id="tip"></div>

<script>
const EMBEDDED = __DATA__;
const LIVE_URL = __LIVE_URL__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const $ = id => document.getElementById(id);
const GPS_EPOCH_MS = Date.UTC(1980, 0, 6), LEAP = 18;
const utc = (week, tow) => new Date(GPS_EPOCH_MS + (week * 604800 + tow - LEAP) * 1000);
const hms = d => d.toISOString().substr(11, 8);

// ---- map + basemaps (Esri: works from file:// without a key; OSM's servers block such pages)
const map = L.map('map');
const esri = (svc, label) => L.tileLayer(
  `https://server.arcgisonline.com/ArcGIS/rest/services/${svc}/MapServer/tile/{z}/{y}/{x}`,
  { maxZoom: 20, maxNativeZoom: 19, attribution: `Tiles &copy; Esri &mdash; ${label}` });
const street = esri('World_Street_Map', 'Esri, HERE, Garmin, USGS, OpenStreetMap contributors').addTo(map);
const imagery = esri('World_Imagery', 'Esri, Maxar, Earthstar Geographics');
L.control.layers({ 'Street map': street, 'Satellite imagery': imagery }, null, { collapsed: false }).addTo(map);
const layer = L.layerGroup().addTo(map);
let fitted = false;

let trailS = +(localStorage.getItem('gpsTrail') || 60), lastData = null;
$('trail').value = trailS;
$('trailval').textContent = trailS + ' s';
$('trail').addEventListener('input', e => {
  trailS = +e.target.value; localStorage.setItem('gpsTrail', trailS);
  $('trailval').textContent = trailS + ' s';
  if (lastData) render(lastData);
});

function render(D) {
  lastData = D;
  layer.clearLayers();
  const last = D.fixes[D.fixes.length - 1];
  // Fade by age relative to the newest fix; drop anything older than the trail length.
  const vis = D.fixes.map(f => ({ ...f, age: last.tow - f.tow })).filter(f => f.age <= trailS);
  const mLat = 111320, mLon = 111320 * Math.cos(last.lat * Math.PI / 180);
  const n = vis.length;
  const mean = { lat: vis.reduce((a, f) => a + f.lat, 0) / n, lon: vis.reduce((a, f) => a + f.lon, 0) / n,
                 h: vis.reduce((a, f) => a + f.h, 0) / n };
  const sd = g => Math.sqrt(vis.reduce((a, f) => a + g(f) ** 2, 0) / n);
  const se = sd(f => (f.lon - mean.lon) * mLon), sn = sd(f => (f.lat - mean.lat) * mLat), su = sd(f => f.h - mean.h);
  const r95 = Math.max(2 * Math.hypot(se, sn), 3);

  L.circle([mean.lat, mean.lon], { radius: r95, color: css('--ink-2'), weight: 1.5, dashArray: '4 3',
                                   fillOpacity: 0.04, interactive: false }).addTo(layer);
  vis.sort((a, b) => b.age - a.age);            // oldest first, so newer fixes draw on top
  vis.forEach(f => {
    const fresh = 1 - f.age / trailS;           // 1 = newest, 0 = at the threshold
    const isLast = f === vis[vis.length - 1];
    const rad = isLast ? 7 : 2.5 + 2.5 * fresh;
    const de = (f.lon - mean.lon) * mLon, dn = (f.lat - mean.lat) * mLat;
    const hit = L.circleMarker([f.lat, f.lon], { radius: 9, stroke: false, fillOpacity: 0 }).addTo(layer);
    const dot = L.circleMarker([f.lat, f.lon], {
      radius: rad, color: css('--surface'), weight: isLast ? 2 : 1, opacity: 0.25 + 0.75 * fresh,
      fillColor: css('--series-1'), fillOpacity: 0.08 + 0.92 * fresh, interactive: false }).addTo(layer);
    hit.bindTooltip(`<b>${isLast ? 'Latest fix' : f.age.toFixed(0) + ' s before latest'}</b> &middot; ` +
      `${hms(utc(D.week, f.tow))} UTC<br>${f.lat.toFixed(6)}, ${f.lon.toFixed(6)}<br>` +
      `${de >= 0 ? '+' : ''}${de.toFixed(1)} m E, ${dn >= 0 ? '+' : ''}${dn.toFixed(1)} m N of mean<br>` +
      `height ${f.h.toFixed(1)} m`, { direction: 'top', offset: [0, -6] });
    hit.on('mouseover', () => dot.setRadius(rad + 2)).on('mouseout', () => dot.setRadius(rad));
  });
  L.circleMarker([mean.lat, mean.lon], { radius: 7, color: css('--surface'), weight: 2, fillColor: css('--ink'), fillOpacity: 1 })
    .bindTooltip(`<b>Mean of ${n} visible fixes</b><br>${mean.lat.toFixed(6)}, ${mean.lon.toFixed(6)}`,
                 { direction: 'top', offset: [0, -8] })
    .addTo(layer);
  if (!fitted) { map.fitBounds(L.latLng([mean.lat, mean.lon]).toBounds(Math.max(r95 * 5, 60))); fitted = true; }

  // sidebar
  const t1 = utc(D.week, last.tow);
  $('subtitle').innerHTML = `${D.sats.length} satellites &middot; ${t1.toISOString().substr(0, 10)} ` +
    `${hms(t1)} UTC (latest) &middot; ${D.source}`;
  $('latlon').textContent = `${last.lat.toFixed(6)}, ${last.lon.toFixed(6)}`;
  $('height').textContent = `latest fix &middot; height ${last.h.toFixed(0)} m above WGS-84 ellipsoid`.replace('&middot;', '\u00b7');
  $('lg-fixes').textContent = `Fixes, newest brightest (${n} of ${D.fixes.length} within ${trailS} s)`;
  $('lg-mean').textContent = `Mean of visible fixes: ${mean.lat.toFixed(6)}, ${mean.lon.toFixed(6)}`;
  $('lg-r95').textContent = `2DRMS (~95% of horizontal fixes): ${r95.toFixed(0)} m`;
  $('se').textContent = se.toFixed(1) + ' m'; $('sn').textContent = sn.toFixed(1) + ' m'; $('su').textContent = su.toFixed(1) + ' m';
  $('dops').textContent = `${D.dop.HDOP.toFixed(1)} / ${D.dop.VDOP.toFixed(1)} / ${D.dop.PDOP.toFixed(1)}`;
  $('ppm').textContent = (D.clock_ppm >= 0 ? '+' : '') + D.clock_ppm.toFixed(2) + ' ppm';
  $('skytitle').textContent = D.live ? 'Sky now' : 'Sky at first fix';
  drawSky(D);
}

// ---- sky plot (center = zenith, edge = horizon, north up)
const NS = 'http://www.w3.org/2000/svg', cx = 150, cy = 150, R = 120, tip = $('tip');
const el2r = el => R * (90 - el) / 90;
function add(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  parent.appendChild(e); return e;
}
function drawSky(D) {
  const sky = $('sky'); sky.innerHTML = '';
  [0, 30, 60].forEach(el => {
    add('circle', { cx, cy, r: el2r(el), fill: 'none', stroke: css('--grid'), 'stroke-width': 1 }, sky);
    if (el > 0) add('text', { x: cx + 3, y: cy - el2r(el) - 3 }, sky).textContent = el + '°';
  });
  for (let a = 0; a < 360; a += 45) {
    const t = a * Math.PI / 180;
    add('line', { x1: cx, y1: cy, x2: cx + R * Math.sin(t), y2: cy - R * Math.cos(t), stroke: css('--grid'), 'stroke-width': 1 }, sky);
  }
  [['N', 0], ['E', 90], ['S', 180], ['W', 270]].forEach(([s, a]) => {
    const t = a * Math.PI / 180;
    add('text', { x: cx + (R + 14) * Math.sin(t), y: cy - (R + 14) * Math.cos(t) + 4, 'text-anchor': 'middle' }, sky).textContent = s;
  });
  const extra = D.sats.length && D.sats[0].time !== undefined;
  const table = $('sattable');
  table.innerHTML = '<tr><th>PRN</th><th>Az</th><th>El</th><th>C/N0</th>' +
                    (extra ? '<th>Time</th><th>Eph</th>' : '') + '</tr>';
  D.sats.forEach(s => {
    const t = s.az * Math.PI / 180, r = el2r(s.el);
    const x = cx + r * Math.sin(t), y = cy - r * Math.cos(t);
    const g = add('g', { tabindex: 0, 'aria-label': `PRN ${s.prn}, azimuth ${s.az.toFixed(0)}, elevation ${s.el.toFixed(0)}` }, sky);
    add('circle', { cx: x, cy: y, r: 14, fill: 'transparent' }, g);
    add('circle', { cx: x, cy: y, r: 6, fill: css('--series-1'), stroke: css('--surface'), 'stroke-width': 2 }, g);
    add('text', { x: x + 9, y: y + 4, class: 'lbl' }, g).textContent = s.prn;
    g.addEventListener('mousemove', ev => {
      tip.innerHTML = `<b>PRN ${s.prn}</b><br>azimuth ${s.az.toFixed(1)}&deg;<br>elevation ${s.el.toFixed(1)}&deg;<br>` +
                      `C/N0 ${s.cn0.toFixed(0)} dB-Hz` + (extra ? `<br>time: ${s.time}<br>ephemeris: ${s.eph}` : '');
      tip.style.display = 'block'; tip.style.left = (ev.clientX + 12) + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
    });
    g.addEventListener('mouseleave', () => tip.style.display = 'none');
    const row = table.insertRow();
    const cells = [s.prn, s.az.toFixed(0) + '°', s.el.toFixed(0) + '°', s.cn0.toFixed(0)];
    if (extra) cells.push(s.time, s.eph);
    cells.forEach(v => row.insertCell().textContent = v);
  });
}

// ---- live polling
if (LIVE_URL) {
  $('live').classList.add('on');
  let lastOk = 0;
  const poll = async () => {
    try {
      const r = await fetch(LIVE_URL + '?' + Date.now());
      if (r.ok) { render(await r.json()); lastOk = Date.now(); }
    } catch (e) {}
    const stale = Date.now() - lastOk > 6000;
    $('live').classList.toggle('stale', stale);
    $('livetxt').textContent = lastOk === 0 ? 'WAITING FOR FIX' : (stale ? 'NO UPDATE' : 'LIVE');
  };
  poll(); setInterval(poll, 2000);
  if (!EMBEDDED) map.setView([20, 0], 2);
} else {
  render(EMBEDDED);
}
</script>
</body>
</html>
"""


def render_page(data=None, live_url=None):
    return (TEMPLATE.replace("__DATA__", json.dumps(data) if data else "null")
                    .replace("__LIVE_URL__", json.dumps(live_url)))


def write_live_page(path, json_name="live_fixes.json"):
    with open(path, "w") as fh:
        fh.write(render_page(None, json_name))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json", nargs="?", default="data/fixes.json")
    p.add_argument("--out", default="data/map.html")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    with open(args.out, "w") as fh:
        fh.write(render_page(json.load(open(args.json))))
    print("Saved", args.out)
    if not args.no_open:
        subprocess.run(["open", args.out])


if __name__ == "__main__":
    main()
