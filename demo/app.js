/* Static demo: the model runs in web workers, the page only draws. */
(() => {
  "use strict";

  const M = RadModel;
  const $ = (sel) => document.querySelector(sel);

  const state = {
    manifest: null,
    probe: null, // {lat, lon, mu, logvar, label}
    readingUsv: 0.2,
    unit: "usv",
    mode: "baseline",
    grid: null,
    seq: 0,
    sites: [],
    samples: [],
    land: null,
  };

  const SEVERITY_LABEL = {
    NORMAL: "NORMAL",
    ELEVATED: "ELEVATED",
    ANOMALY_HIGH: "HIGH",
    ANOMALY_CRITICAL: "CRITICAL",
  };

  /* ---------- formatting ---------- */

  function fmt(v) {
    if (!isFinite(v)) return "n/a";
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a < 0.01) return v.toFixed(4);
    if (a < 1) return v.toFixed(3);
    if (a < 10) return v.toFixed(2);
    if (a < 100) return v.toFixed(1);
    return v.toFixed(0);
  }

  function fmtCoord(lat, lon) {
    return `${Math.abs(lat).toFixed(2)}\u00b0 ${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(2)}\u00b0 ${lon >= 0 ? "E" : "W"}`;
  }

  function fmtP(p) {
    if (p < 1e-4) return p.toExponential(1);
    return p.toPrecision(2);
  }

  /* ---------- worker pool ---------- */

  class Pool {
    constructor(size) {
      this.next = 1;
      this.pending = new Map();
      this.workers = [];
      for (let i = 0; i < size; i++) {
        const w = new Worker("worker.js");
        w.onmessage = (e) => {
          const p = this.pending.get(e.data.id);
          if (!p) return;
          this.pending.delete(e.data.id);
          if (e.data.ok) p.resolve(e.data);
          else p.reject(new Error(e.data.error));
        };
        w.onerror = (e) => {
          const msg = e.message || "worker failed";
          this.pending.forEach((p) => p.reject(new Error(msg)));
          this.pending.clear();
        };
        this.workers.push(w);
      }
    }

    call(worker, msg, transfer) {
      return new Promise((resolve, reject) => {
        const id = this.next++;
        this.pending.set(id, { resolve, reject });
        worker.postMessage({ id, ...msg }, transfer || []);
      });
    }

    init() {
      return Promise.all(this.workers.map((w) => this.call(w, { type: "init", base: "" })));
    }

    /* xyz: Float32Array of n*3. Splits across workers for large batches. */
    async predict(xyz) {
      const n = xyz.length / 3;
      const parts = n < 64 ? 1 : this.workers.length;
      const per = Math.ceil(n / parts);
      const jobs = [];
      for (let k = 0; k < parts; k++) {
        const lo = k * per;
        const hi = Math.min(n, lo + per);
        if (lo >= hi) break;
        const slice = xyz.slice(lo * 3, hi * 3);
        jobs.push(this.call(this.workers[k], { type: "predict", xyz: slice }, [slice.buffer]));
      }
      const res = await Promise.all(jobs);
      const mu = new Float32Array(n);
      const logvar = new Float32Array(n);
      let at = 0;
      for (const r of res) {
        mu.set(r.mu, at);
        logvar.set(r.logvar, at);
        at += r.mu.length;
      }
      return { mu, logvar };
    }
  }

  /* ---------- geometry helpers ---------- */

  const merc = (lat) => Math.log(Math.tan(Math.PI / 4 + (Math.max(-85, Math.min(85, lat)) * Math.PI) / 360));
  const invMerc = (y) => ((2 * Math.atan(Math.exp(y)) - Math.PI / 2) * 180) / Math.PI;

  function haversineKm(a, b, c, d) {
    const r = Math.PI / 180;
    const h =
      Math.sin(((c - a) * r) / 2) ** 2 + Math.cos(a * r) * Math.cos(c * r) * Math.sin(((d - b) * r) / 2) ** 2;
    return 12742 * Math.asin(Math.sqrt(h));
  }

  function decodeTopo(topo, name) {
    const [sx, sy] = topo.transform.scale;
    const [tx, ty] = topo.transform.translate;
    const arcs = topo.arcs.map((arc) => {
      let x = 0;
      let y = 0;
      return arc.map(([dx, dy]) => {
        x += dx;
        y += dy;
        return [x * sx + tx, y * sy + ty];
      });
    });
    const arcAt = (i) => (i >= 0 ? arcs[i] : arcs[~i].slice().reverse());
    const ring = (ids) => {
      const pts = [];
      ids.forEach((id, k) => {
        const a = arcAt(id);
        for (let j = k ? 1 : 0; j < a.length; j++) pts.push(a[j]);
      });
      // Unwrap longitudes so rings that cross the antimeridian stay continuous.
      let shift = 0;
      let prev = pts[0][0];
      const out = pts.map((p) => {
        if (p[0] - prev > 180) shift -= 360;
        else if (p[0] - prev < -180) shift += 360;
        prev = p[0];
        return [p[0] + shift, p[1]];
      });
      // A ring that encloses the pole (Antarctica) ends 360 degrees from where it started.
      if (Math.abs(out[out.length - 1][0] - out[0][0]) > 180) {
        out.push([out[out.length - 1][0], -90], [out[0][0], -90], [out[0][0], out[0][1]]);
      }
      const flat = new Float32Array(out.length * 2);
      out.forEach((p, i) => {
        flat[2 * i] = p[0];
        flat[2 * i + 1] = p[1];
      });
      return flat;
    };
    const polys = [];
    for (const g of topo.objects[name].geometries) {
      const list = g.type === "MultiPolygon" ? g.arcs : [g.arcs];
      for (const poly of list) {
        const rings = poly.map(ring);
        let w = 180, e = -180, s = 90, n = -90;
        for (let i = 0; i < rings[0].length; i += 2) {
          w = Math.min(w, rings[0][i]);
          e = Math.max(e, rings[0][i]);
          s = Math.min(s, rings[0][i + 1]);
          n = Math.max(n, rings[0][i + 1]);
        }
        polys.push({ rings, w, e, s, n });
      }
    }
    return polys;
  }

  /* ---------- colour ---------- */

  const RAMP = [
    [0, [34, 48, 59]],
    [0.5, [91, 122, 138]],
    [1, [217, 164, 65]],
  ];

  function ramp(t) {
    t = Math.min(1, Math.max(0, t));
    for (let i = 1; i < RAMP.length; i++) {
      if (t <= RAMP[i][0]) {
        const [t0, c0] = RAMP[i - 1];
        const [t1, c1] = RAMP[i];
        const f = (t - t0) / (t1 - t0);
        return [0, 1, 2].map((k) => Math.round(c0[k] + (c1[k] - c0[k]) * f));
      }
    }
    return RAMP[RAMP.length - 1][1];
  }

  const DOSE_LO = Math.log(0.03);
  const DOSE_HI = Math.log(0.45);
  const doseT = (v) => (Math.log(Math.max(v, 1e-6)) - DOSE_LO) / (DOSE_HI - DOSE_LO);
  const SIGMA_LO = 0.15;
  const SIGMA_HI = 1.5;

  const SURPRISE = {
    NORMAL: [46, 62, 75, 120],
    ELEVATED: [217, 164, 65, 160],
    ANOMALY_HIGH: [212, 121, 90, 200],
    ANOMALY_CRITICAL: [224, 85, 74, 230],
  };

  /* ---------- map ---------- */

  const map = L.map("map", {
    zoomControl: true,
    minZoom: 2,
    maxZoom: 9,
    maxBounds: [[-85, -180], [85, 180]],
    maxBoundsViscosity: 1,
    worldCopyJump: false,
  }).setView([28, 20], 2);

  map.createPane("base").style.zIndex = 200;
  map.createPane("raster").style.zIndex = 350;
  map.createPane("borders").style.zIndex = 400;
  map.createPane("samples").style.zIndex = 500;
  map.createPane("probe").style.zIndex = 600;
  map.attributionControl.setPrefix(false);
  map.attributionControl.addAttribution(
    'Land and borders <a href="https://www.naturalearthdata.com/">Natural Earth</a> via <a href="https://github.com/topojson/world-atlas">world-atlas</a> &middot; ' +
      'Data <a href="https://safecast.org/">Safecast</a> (CC0)'
  );

  const baseRenderer = L.canvas({ pane: "base", padding: 0.4 });
  const borderRenderer = L.canvas({ pane: "borders", padding: 0.4 });

  const shiftsOf = (poly) => {
    const out = [0];
    if (poly.e > 180) out.push(-360);
    if (poly.w < -180) out.push(360);
    return out;
  };

  function addBase(land, countries) {
    const latlngs = (rings, off) =>
      rings.map((r) => {
        const out = new Array(r.length / 2);
        for (let i = 0; i < out.length; i++) out[i] = [r[2 * i + 1], r[2 * i] + off];
        return out;
      });
    const landShapes = [];
    land.forEach((p) => shiftsOf(p).forEach((off) => landShapes.push(latlngs(p.rings, off))));
    L.polygon(landShapes, {
      pane: "base", renderer: baseRenderer, interactive: false,
      stroke: true, color: "#34414d", weight: 0.8, fillColor: "#171d23", fillOpacity: 1, smoothFactor: 0.6,
    }).addTo(map);
    const borders = [];
    countries.forEach((p) =>
      shiftsOf(p).forEach((off) => p.rings.forEach((r) => borders.push(latlngs([r], off)[0])))
    );
    L.polyline(borders, {
      pane: "borders", renderer: borderRenderer, interactive: false,
      color: "#e9e6df", opacity: 0.16, weight: 0.7, smoothFactor: 0.8,
    }).addTo(map);
  }

  const rasterImg = new Image();
  rasterImg.alt = "";
  const rasterLayer = L.imageOverlay(rasterImg, [[-85, -180], [85, 180]], { pane: "raster", interactive: false });
  const small = document.createElement("canvas");
  const big = document.createElement("canvas");

  const sampleRenderer = L.canvas({ pane: "samples" });
  const sampleLayer = L.layerGroup().addTo(map);
  const probeRing = L.circleMarker([0, 0], {
    pane: "probe",
    radius: 9,
    color: "#d9a441",
    weight: 2,
    fillOpacity: 0,
    interactive: false,
  });
  const probeDot = L.circleMarker([0, 0], {
    pane: "probe",
    radius: 2.5,
    color: "#d9a441",
    weight: 1,
    fillColor: "#d9a441",
    fillOpacity: 1,
    interactive: false,
  });

  /* ---------- raster ---------- */

  function paintSmall(g) {
    small.width = g.C;
    small.height = g.R;
    const ctx = small.getContext("2d");
    const img = ctx.createImageData(g.C, g.R);
    const d = img.data;
    const thr = state.manifest.thresholds;
    const obsLog = M.toLog(state.readingUsv);
    for (let i = 0; i < g.mu.length; i++) {
      let c;
      if (state.mode === "baseline") {
        c = [...ramp(doseT(M.fromLog(g.mu[i]))), 215];
      } else if (state.mode === "uncertainty") {
        const s = Math.exp(0.5 * g.logvar[i]);
        c = [...ramp((s - SIGMA_LO) / (SIGMA_HI - SIGMA_LO)), 215];
      } else {
        const z = (obsLog - g.mu[i]) / Math.max(Math.exp(0.5 * g.logvar[i]), 1e-4);
        const sev =
          z >= thr.anomaly_high_z_max
            ? "ANOMALY_CRITICAL"
            : z >= thr.elevated_z_max
              ? "ANOMALY_HIGH"
              : z >= thr.normal_z_max
                ? "ELEVATED"
                : "NORMAL";
        c = SURPRISE[sev];
      }
      d[4 * i] = c[0];
      d[4 * i + 1] = c[1];
      d[4 * i + 2] = c[2];
      d[4 * i + 3] = c[3];
    }
    ctx.putImageData(img, 0, 0);
  }

  function render() {
    const g = state.grid;
    if (!g) return;
    paintSmall(g);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const Wc = Math.max(2, Math.round(g.wpx * dpr));
    const Hc = Math.max(2, Math.round(g.hpx * dpr));
    big.width = Wc;
    big.height = Hc;
    const ctx = big.getContext("2d");
    ctx.imageSmoothingEnabled = state.mode !== "surprise";
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(small, 0, 0, Wc, Hc);

    if (state.land) {
      const yN = merc(g.north);
      const yS = merc(g.south);
      const kx = Wc / (g.east - g.west);
      const ky = Hc / (yN - yS);
      ctx.globalCompositeOperation = "destination-in";
      ctx.beginPath();
      for (const poly of state.land) {
        if (poly.n < g.south || poly.s > g.north) continue;
        for (const off of shiftsOf(poly)) {
          if (poly.e + off < g.west || poly.w + off > g.east) continue;
          for (const r of poly.rings) {
            for (let i = 0; i < r.length; i += 2) {
              const x = (r[i] + off - g.west) * kx;
              const y = (yN - merc(r[i + 1])) * ky;
              if (i === 0) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            }
            ctx.closePath();
          }
        }
      }
      ctx.fillStyle = "#000";
      ctx.fill("evenodd");
      ctx.globalCompositeOperation = "source-over";
    }

    rasterImg.src = big.toDataURL("image/png");
    rasterLayer.setBounds([[g.south, g.west], [g.north, g.east]]);
    if (!map.hasLayer(rasterLayer)) rasterLayer.addTo(map);
  }

  let renderQueued = false;
  function scheduleRender() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => {
      renderQueued = false;
      render();
    });
  }

  async function evalGrid(pool, west, east, north, south, C, R) {
    const yN = merc(north);
    const yS = merc(south);
    const xyz = new Float32Array(C * R * 3);
    let k = 0;
    for (let j = 0; j < R; j++) {
      const lat = invMerc(yN + ((j + 0.5) / R) * (yS - yN));
      for (let i = 0; i < C; i++) {
        const lon = west + ((i + 0.5) / C) * (east - west);
        const p = M.latLonToXyz(lat, lon);
        xyz[k++] = p[0];
        xyz[k++] = p[1];
        xyz[k++] = p[2];
      }
    }
    return pool.predict(xyz);
  }

  let pool = null;
  let refreshTimer = 0;

  async function refresh() {
    const seq = ++state.seq;
    const b = map.getBounds();
    const south = Math.max(-85, b.getSouth());
    const north = Math.min(85, b.getNorth());
    const west = Math.max(-180, b.getWest());
    const east = Math.min(180, b.getEast());
    const nw = map.latLngToContainerPoint([north, west]);
    const se = map.latLngToContainerPoint([south, east]);
    const wpx = se.x - nw.x;
    const hpx = se.y - nw.y;
    if (wpx < 4 || hpx < 4) return;

    let cell = 9;
    while ((wpx / cell) * (hpx / cell) > 7000) cell += 1;
    const C = Math.max(8, Math.round(wpx / cell));
    const R = Math.max(8, Math.round(hpx / cell));

    $("#busy").hidden = false;
    try {
      for (const div of [3, 1]) {
        const c = Math.max(8, Math.round(C / div));
        const r = Math.max(8, Math.round(R / div));
        const t0 = performance.now();
        const out = await evalGrid(pool, west, east, north, south, c, r);
        if (seq !== state.seq) return;
        state.grid = { C: c, R: r, west, east, north, south, wpx, hpx, mu: out.mu, logvar: out.logvar };
        render();
        $("#caption").textContent =
          `${c} x ${r} grid for this view, ${(c * r).toLocaleString("en-US")} forward passes in ` +
          `${Math.round(performance.now() - t0)} ms on ${pool.workers.length} worker threads.`;
      }
    } catch (err) {
      showFailure(err);
    } finally {
      if (seq === state.seq) $("#busy").hidden = true;
    }
  }

  function showFailure(err) {
    const el = $("#fail");
    el.hidden = false;
    el.textContent = `Model error: ${err.message}`;
  }

  /* ---------- probe and panel ---------- */

  function nearestSite(lat, lon) {
    let best = null;
    for (const s of state.sites) {
      const d = haversineKm(lat, lon, s.latitude, s.longitude);
      if (d < 60 && (!best || d < best.d)) best = { s, d };
    }
    return best;
  }

  async function setProbe(lat, lon, opts = {}) {
    lon = ((((lon + 180) % 360) + 360) % 360) - 180;
    lat = Math.max(-89.9, Math.min(89.9, lat));
    const p = M.latLonToXyz(lat, lon);
    try {
      const out = await pool.predict(new Float32Array(p));
      state.probe = { lat, lon, mu: out.mu[0], logvar: out.logvar[0], label: opts.label || "" };
    } catch (err) {
      showFailure(err);
      return;
    }
    probeRing.setLatLng([lat, lon]);
    probeDot.setLatLng([lat, lon]);
    if (!map.hasLayer(probeRing)) {
      probeRing.addTo(map);
      probeDot.addTo(map);
    }
    if (opts.reading != null) setReading(opts.reading, null);
    else renderPanel();
  }

  function renderPanel() {
    const pr = state.probe;
    if (!pr) return;
    const s = M.summarise(pr.mu, pr.logvar);
    const near = nearestSite(pr.lat, pr.lon);
    let place = fmtCoord(pr.lat, pr.lon);
    if (near) place += ` \u00b7 near ${near.s.name}`;
    if (pr.label) place += ` \u00b7 ${pr.label}`;
    $("#place").textContent = place;

    const wide = s.sigmaLog > 0.9;
    $("#expect").innerHTML =
      `The model expects <span class="n">${fmt(s.expected)} &micro;Sv/h</span> here. ` +
      `95% of readings should fall between <span class="n">${fmt(s.lower)}</span> and <span class="n">${fmt(s.upper)}</span>.` +
      (wide ? " That range is wide: the model is unsure about this area." : "");
    renderVerdict();
  }

  function describe(sc, s) {
    const fold = sc.fold >= 10 ? sc.fold.toFixed(0) : sc.fold.toFixed(1);
    switch (sc.severity) {
      case "NORMAL":
        return sc.fold >= 3 && s.sigmaLog > 0.9
          ? `${fold} times the expected value, yet it scores as normal because the model's range here is very wide.`
          : "Within the normal local range.";
      case "ELEVATED":
        return `Slightly elevated (${fold} times the expected value). Altitude, radon or ordinary local variation can explain this.`;
      case "ANOMALY_HIGH":
        return `Significant anomaly (${fold} times the expected value). A local source or contamination is plausible.`;
      default:
        return `Critical (${fold} times the expected value). Far outside anything the model considers natural for this place.`;
    }
  }

  function renderVerdict() {
    const pr = state.probe;
    if (!pr) return;
    const s = M.summarise(pr.mu, pr.logvar);
    const sc = M.score(state.readingUsv, pr.mu, s.sigmaLog, state.manifest.thresholds);
    $("#verdict").innerHTML =
      `<div class="line"><span class="sev sev-${sc.severity}">${SEVERITY_LABEL[sc.severity]}</span>` +
      `<span>Z = ${sc.z.toFixed(2)}</span><span>${sc.fold >= 10 ? sc.fold.toFixed(0) : sc.fold.toFixed(1)}x expected</span>` +
      `<span>p = ${fmtP(sc.p)}</span></div><p>${describe(sc, s)}</p>`;
    renderChart(pr, s);
    renderSites();
  }

  /* distribution chart: density of the dose rate on a log axis */
  function renderChart(pr, s) {
    const W = 340;
    const x0 = 18;
    const x1 = 322;
    const yBase = 118;
    const yTop = 22;
    const T0 = -2;
    const T1 = 2;
    const px = (t) => x0 + ((t - T0) / (T1 - T0)) * (x1 - x0);
    const N = 180;
    const dens = [];
    let peak = 0;
    for (let i = 0; i <= N; i++) {
      const t = T0 + ((T1 - T0) * i) / N;
      const y = Math.pow(10, t);
      const L = M.toLog(y);
      const zz = (L - pr.mu) / s.sigmaLog;
      const jac = (Math.LN10 * 10 * y) / (1 + 10 * y);
      const d = (Math.exp(-0.5 * zz * zz) / s.sigmaLog) * jac;
      dens.push([t, d]);
      peak = Math.max(peak, d);
    }
    const ypos = (d) => yBase - (d / peak) * (yBase - yTop);
    const tLo = Math.log10(Math.max(s.lower, 1e-4));
    const tHi = Math.log10(Math.max(s.upper, 1e-4));

    let area = `M${px(T0)},${yBase}`;
    let band = "";
    let bandStarted = false;
    dens.forEach(([t, d]) => {
      area += ` L${px(t).toFixed(1)},${ypos(d).toFixed(1)}`;
      if (t >= tLo && t <= tHi) {
        band += `${bandStarted ? " L" : `M${px(t).toFixed(1)},${yBase} L`}${px(t).toFixed(1)},${ypos(d).toFixed(1)}`;
        bandStarted = true;
      }
    });
    area += ` L${px(T1)},${yBase} Z`;
    if (bandStarted) {
      const last = dens.filter(([t]) => t >= tLo && t <= tHi).pop();
      band += ` L${px(last[0]).toFixed(1)},${yBase} Z`;
    }

    let ticks = "";
    for (let t = T0; t <= T1; t++) {
      const label = Math.pow(10, t);
      ticks +=
        `<line class="axis" x1="${px(t)}" x2="${px(t)}" y1="${yBase}" y2="${yBase + 4}"/>` +
        `<text x="${px(t)}" y="${yBase + 15}" text-anchor="middle">${label < 1 ? label : label.toFixed(0)}</text>`;
    }

    let zl = "";
    [2, 4, 8].forEach((z) => {
      const y = M.fromLog(pr.mu + z * s.sigmaLog);
      const t = Math.log10(Math.max(y, 1e-6));
      if (t > T0 && t < T1) {
        zl +=
          `<line class="zt" x1="${px(t)}" x2="${px(t)}" y1="${yTop - 6}" y2="${yBase}" stroke-dasharray="2 3" stroke-width="0.8"/>` +
          `<text x="${px(t)}" y="${yTop - 10}" text-anchor="middle">Z${z}</text>`;
      }
    });

    const tr = Math.min(T1, Math.max(T0, Math.log10(state.readingUsv)));
    const flip = px(tr) > W - 90;
    const readLabel = `${fmt(state.readingUsv)}`;
    const svg =
      `<path class="dens" d="${area}"/>` +
      (bandStarted ? `<path class="band" d="${band}"/>` : "") +
      `<line class="axis" x1="${x0}" x2="${x1}" y1="${yBase}" y2="${yBase}"/>` +
      ticks +
      zl +
      `<line class="read" x1="${px(tr)}" x2="${px(tr)}" y1="${yTop - 4}" y2="${yBase}"/>` +
      `<text class="read-label" x="${px(tr) + (flip ? -5 : 5)}" y="${yTop + 6}" text-anchor="${flip ? "end" : "start"}">${readLabel}</text>` +
      `<text x="${x1}" y="146" text-anchor="end">&micro;Sv/h, log scale</text>` +
      `<text x="${x0}" y="146">shaded: 95% range</text>`;
    $("#chart").innerHTML = svg;
  }

  /* ---------- reading controls ---------- */

  const READ_MIN = 0.01;
  const READ_MAX = 100;

  function usvToSlider(usv) {
    const t = (Math.log10(Math.min(READ_MAX, Math.max(READ_MIN, usv))) + 2) / 4;
    return Math.round(t * 1000);
  }

  function sliderToUsv(v) {
    return Math.pow(10, -2 + (4 * v) / 1000);
  }

  function toDisplay(usv) {
    return state.unit === "cpm" ? usv / state.manifest.cpm_to_usv : usv;
  }

  function setReading(usv, source) {
    usv = Math.min(1000, Math.max(0.001, usv));
    state.readingUsv = usv;
    if (source !== "range") $("#reading-range").value = usvToSlider(usv);
    if (source !== "num") $("#reading-num").value = Number(toDisplay(usv).toPrecision(3));
    renderPanel();
    if (state.mode === "surprise") scheduleRender();
  }

  function initControls() {
    const presets = [0.1, 0.5, 2, 5, 20];
    $("#chips").innerHTML = presets
      .map((v) => `<button type="button" data-v="${v}">${v} &micro;Sv/h</button>`)
      .join("");
    $("#chips").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b) setReading(parseFloat(b.dataset.v), null);
    });
    $("#reading-range").addEventListener("input", (e) => setReading(sliderToUsv(+e.target.value), "range"));
    $("#reading-num").addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      if (!isFinite(v) || v <= 0) return;
      setReading(state.unit === "cpm" ? v * state.manifest.cpm_to_usv : v, "num");
    });
    $("#reading-unit").addEventListener("change", (e) => {
      state.unit = e.target.value;
      $("#reading-num").value = Number(toDisplay(state.readingUsv).toPrecision(3));
    });
    document.querySelectorAll('input[name="mode"]').forEach((r) =>
      r.addEventListener("change", (e) => {
        state.mode = e.target.value;
        renderLegend();
        render();
      })
    );
    $("#samples-toggle").addEventListener("change", (e) => {
      if (e.target.checked) sampleLayer.addTo(map);
      else map.removeLayer(sampleLayer);
    });
  }

  function renderLegend() {
    const el = $("#legend");
    const grad = (labels) => {
      const stops = RAMP.map(([t, c]) => `rgb(${c.join(",")}) ${t * 100}%`).join(",");
      return (
        `<div><div class="ramp" style="background:linear-gradient(to right,${stops})"></div>` +
        `<div class="ramp-labels">${labels.map((l) => `<span>${l}</span>`).join("")}</div></div>`
      );
    };
    if (state.mode === "baseline") {
      el.innerHTML = grad(["0.03", "0.1", "0.45+"]) + "<span>expected dose rate, &micro;Sv/h</span>";
    } else if (state.mode === "uncertainty") {
      el.innerHTML = grad(["0.15", "0.8", "1.5"]) + "<span>model uncertainty (sigma in log space); lighter means less sure</span>";
    } else {
      const t = state.manifest.thresholds;
      const sw = (c, label) => `<span class="swatch"><i style="background:rgba(${c[0]},${c[1]},${c[2]},${c[3] / 255})"></i>${label}</span>`;
      el.innerHTML =
        sw(SURPRISE.NORMAL, `normal (Z &lt; ${t.normal_z_max})`) +
        sw(SURPRISE.ELEVATED, `elevated (Z ${t.normal_z_max}-${t.elevated_z_max})`) +
        sw(SURPRISE.ANOMALY_HIGH, `high (${t.elevated_z_max}-${t.anomaly_high_z_max})`) +
        sw(SURPRISE.ANOMALY_CRITICAL, `critical (&ge; ${t.anomaly_high_z_max})`) +
        "<span>for the reading set on the right, if it were measured in each place</span>";
    }
  }

  /* ---------- reference sites ---------- */

  function renderSites() {
    const thr = state.manifest.thresholds;
    const rows = state.sites
      .filter((s) => s.mu != null)
      .map((s) => {
        const sig = Math.exp(0.5 * s.logvar);
        const sc = M.score(state.readingUsv, s.mu, sig, thr);
        return (
          `<tr><td><button type="button" data-id="${s.id}">${s.name}</button></td>` +
          `<td class="v">${fmt(M.fromLog(s.mu))}</td>` +
          `<td class="s"><span class="sev sev-${sc.severity}">${SEVERITY_LABEL[sc.severity]}</span></td></tr>`
        );
      })
      .join("");
    $("#sites tbody").innerHTML = rows;
  }

  async function loadSites() {
    const sites = await fetch("data/landmarks.json").then((r) => r.json());
    const xyz = new Float32Array(sites.length * 3);
    sites.forEach((s, i) => xyz.set(M.latLonToXyz(s.latitude, s.longitude), 3 * i));
    const out = await pool.predict(xyz);
    sites.forEach((s, i) => {
      s.mu = out.mu[i];
      s.logvar = out.logvar[i];
    });
    state.sites = sites;
    const pick = (s) => {
      map.flyTo([s.latitude, s.longitude], 7, { duration: 1.2 });
      setProbe(s.latitude, s.longitude, { label: s.description });
    };
    $("#sites").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b) pick(state.sites.find((x) => x.id === b.dataset.id));
    });
    for (const s of sites) {
      L.circleMarker([s.latitude, s.longitude], {
        renderer: sampleRenderer,
        pane: "samples",
        radius: 5.5,
        weight: 1.3,
        color: "#d9a441",
        fillOpacity: 0,
        bubblingMouseEvents: false,
      })
        .bindTooltip(s.name, { direction: "top", offset: [0, -4] })
        .on("click", () => pick(s))
        .addTo(map);
    }
  }

  async function loadSamples() {
    const rows = await fetch("data/samples.json").then((r) => r.json());
    state.samples = rows;
    for (const [lat, lon, usv] of rows) {
      const [r, g, b] = ramp(doseT(usv));
      const m = L.circleMarker([lat, lon], {
        renderer: sampleRenderer,
        pane: "samples",
        radius: 3.2,
        weight: 0.8,
        color: "#e9e6df",
        opacity: 0.8,
        fillColor: `rgb(${r},${g},${b})`,
        fillOpacity: 0.95,
        bubblingMouseEvents: false,
      });
      m.bindTooltip(`Measured ${fmt(usv)} \u00b5Sv/h`, { direction: "top", offset: [0, -4] });
      m.on("click", () => setProbe(lat, lon, { reading: usv, label: "measured reading from the test split" }));
      m.addTo(sampleLayer);
    }
  }

  function renderLimits() {
    const m = state.manifest.metrics.neural;
    $("#limits").innerHTML =
      `This estimates the <b>typical</b> dose rate of a place, not a live measurement. On areas the network never saw in training it ` +
      `explains little of the variation (R&sup2; ${m.r2.toFixed(2)}), and its 95% range covers ${(m.picp_95 * 100).toFixed(1)}% of ` +
      `held-out readings only because the ranges are wide. Where data is thin, even a very high reading can score as normal: ` +
      `set 5 &micro;Sv/h and select Chernobyl. Dots on the map are real Safecast readings from the test split, which the model did not train on; ` +
      `select one to replay it. Away from surveyed areas the colours are extrapolation, which is why they turn streaky over Africa and Central Asia. ` +
      `Rings mark the reference sites.`;
  }

  /* ---------- start ---------- */

  async function start() {
    try {
      state.manifest = await fetch("model/model.json").then((r) => r.json());
      const size = Math.max(2, Math.min(6, (navigator.hardwareConcurrency || 4) - 1));
      pool = new Pool(size);
      initControls();
      renderLegend();
      renderLimits();
      await pool.init();

      Promise.all([
        fetch("data/land-50m.json").then((r) => r.json()),
        fetch("data/countries-50m.json").then((r) => r.json()),
      ])
        .then(([landTopo, countryTopo]) => {
          state.land = decodeTopo(landTopo, "land");
          addBase(state.land, decodeTopo(countryTopo, "countries"));
          render();
        })
        .catch(showFailure);

      await loadSites();
      loadSamples();
      setReading(state.readingUsv, null);
      map.on("click", (e) => setProbe(e.latlng.lat, e.latlng.lng));
      map.on("moveend", () => {
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(refresh, 150);
      });
      const fuku = state.sites.find((s) => s.id === "Fukushima_Daiichi");
      await setProbe(fuku.latitude, fuku.longitude, { label: fuku.description });
      refresh();
    } catch (err) {
      showFailure(err);
    }
  }

  start();
})();
