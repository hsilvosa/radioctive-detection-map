/* SpatialHarmonicNet forward pass and anomaly scoring in plain JavaScript.
 *
 * The weights come from scripts/export_demo.py. The page and the web workers
 * both load this file, so everything is attached to one global, RadModel.
 */
(function (root) {
  "use strict";

  const TARGET_SCALE = 10;

  function silu(x) {
    return x / (1 + Math.exp(-x));
  }

  /* Abramowitz-Stegun 7.1.26 is too coarse for tails; use a continued erfc. */
  function erfc(x) {
    const z = Math.abs(x);
    const t = 1 / (1 + 0.5 * z);
    const r =
      t *
      Math.exp(
        -z * z -
          1.26551223 +
          t *
            (1.00002368 +
              t *
                (0.37409196 +
                  t *
                    (0.09678418 +
                      t *
                        (-0.18628806 +
                          t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
      );
    return x >= 0 ? r : 2 - r;
  }

  function latLonToXyz(lat, lon) {
    const la = (lat * Math.PI) / 180;
    const lo = (lon * Math.PI) / 180;
    const c = Math.cos(la);
    return [c * Math.cos(lo), c * Math.sin(lo), Math.sin(la)];
  }

  class Net {
    constructor(manifest, weights) {
      this.m = manifest;
      this.w = weights;
      const F = manifest.num_frequencies;
      this.F = F;
      this.freq = new Float64Array(F);
      for (let f = 0; f < F; f++) {
        const e = F === 1 ? 0 : (manifest.max_frequency_log * f) / (F - 1);
        this.freq[f] = Math.pow(2, e) * Math.PI;
      }
      let widest = manifest.embed_dim;
      const scan = (layer) => {
        if (layer && layer.out) widest = Math.max(widest, layer.out);
      };
      manifest.ops.forEach((op) => {
        scan(op);
        if (op.op === "residual") {
          scan(op.fc1);
          scan(op.fc2);
        }
      });
      this.bufA = new Float32Array(widest);
      this.bufB = new Float32Array(widest);
      this.bufC = new Float32Array(widest);
    }

    linear(layer, src, dst) {
      const w = this.w;
      const nIn = layer.in;
      const nOut = layer.out;
      let row = layer.w;
      for (let j = 0; j < nOut; j++) {
        let s = w[layer.b + j];
        let i = 0;
        for (; i + 3 < nIn; i += 4) {
          s +=
            w[row + i] * src[i] +
            w[row + i + 1] * src[i + 1] +
            w[row + i + 2] * src[i + 2] +
            w[row + i + 3] * src[i + 3];
        }
        for (; i < nIn; i++) s += w[row + i] * src[i];
        dst[j] = s;
        row += nIn;
      }
    }

    layerNorm(layer, buf) {
      const n = layer.dim;
      const w = this.w;
      let mean = 0;
      for (let i = 0; i < n; i++) mean += buf[i];
      mean /= n;
      let v = 0;
      for (let i = 0; i < n; i++) {
        const d = buf[i] - mean;
        v += d * d;
      }
      const inv = 1 / Math.sqrt(v / n + layer.eps);
      for (let i = 0; i < n; i++) buf[i] = (buf[i] - mean) * inv * w[layer.g + i] + w[layer.b + i];
    }

    /* Forward pass for one unit-sphere point; returns [mu, logvar]. */
    forward(x, y, z) {
      const F = this.F;
      const a = this.bufA;
      const b = this.bufB;
      const c = this.bufC;
      a[0] = x;
      a[1] = y;
      a[2] = z;
      const xyz = [x, y, z];
      for (let ax = 0; ax < 3; ax++) {
        for (let f = 0; f < F; f++) {
          const s = xyz[ax] * this.freq[f];
          a[3 + ax * F + f] = Math.sin(s);
          a[3 + 3 * F + ax * F + f] = Math.cos(s);
        }
      }

      let cur = a;
      let other = b;
      let width = this.m.embed_dim;
      for (const op of this.m.ops) {
        if (op.op === "linear") {
          this.linear(op, cur, other);
          [cur, other] = [other, cur];
          width = op.out;
        } else if (op.op === "layernorm") {
          this.layerNorm(op, cur);
        } else if (op.op === "silu") {
          for (let i = 0; i < width; i++) cur[i] = silu(cur[i]);
        } else if (op.op === "residual") {
          // cur holds the block input; fc1/fc2 work in other and c.
          this.linear(op.fc1, cur, other);
          this.layerNorm(op.norm1, other);
          for (let i = 0; i < width; i++) other[i] = silu(other[i]);
          this.linear(op.fc2, other, c);
          this.layerNorm(op.norm2, c);
          for (let i = 0; i < width; i++) cur[i] = silu(c[i] + cur[i]);
        }
      }

      const head = (layers) => {
        this.linear(layers[0], cur, other);
        for (let i = 0; i < layers[0].out; i++) other[i] = silu(other[i]);
        this.linear(layers[1], other, c);
        return c[0];
      };
      const mu = head(this.m.heads.mu);
      let logvar = head(this.m.heads.logvar);
      logvar = Math.min(Math.max(logvar, this.m.logvar_clamp[0]), this.m.logvar_clamp[1]);
      return [mu, logvar];
    }

    /* xyz: Float32Array of n*3. Returns {mu, logvar} Float32Arrays. */
    predict(xyz) {
      const n = xyz.length / 3;
      const mu = new Float32Array(n);
      const logvar = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const r = this.forward(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
        mu[i] = r[0];
        logvar[i] = r[1];
      }
      return { mu, logvar };
    }
  }

  /* Target transform: y_log = log(1 + 10 y). */
  const toLog = (usv) => Math.log1p(TARGET_SCALE * usv);
  const fromLog = (yLog) => Math.max(Math.expm1(yLog) / TARGET_SCALE, 0);

  /* Same quantities as SpatialHarmonicNet.predict_radiation with one sample. */
  function summarise(mu, logvar) {
    const sigma = Math.exp(0.5 * logvar);
    return {
      muLog: mu,
      sigmaLog: sigma,
      expected: fromLog(mu),
      lower: fromLog(Math.max(mu - 1.96 * sigma, 0)),
      upper: fromLog(mu + 1.96 * sigma),
      sigmaPhysical: (sigma * Math.exp(mu)) / TARGET_SCALE,
    };
  }

  /* Same rules as RadiationAnomalyDetector.detect. */
  function score(observedUsv, muLog, sigmaLog, thresholds) {
    const z = (toLog(observedUsv) - muLog) / Math.max(sigmaLog, 1e-4);
    let severity = "NORMAL";
    if (z >= thresholds.anomaly_high_z_max) severity = "ANOMALY_CRITICAL";
    else if (z >= thresholds.elevated_z_max) severity = "ANOMALY_HIGH";
    else if (z >= thresholds.normal_z_max) severity = "ELEVATED";
    return {
      z,
      p: 0.5 * erfc(z / Math.SQRT2),
      fold: observedUsv / Math.max(fromLog(muLog), 1e-4),
      severity,
    };
  }

  root.RadModel = { Net, latLonToXyz, toLog, fromLog, summarise, score, erfc };
})(typeof self !== "undefined" ? self : this);
