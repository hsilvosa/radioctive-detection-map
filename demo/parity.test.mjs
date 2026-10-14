// Checks the JavaScript forward pass against PyTorch reference outputs.
// Run from the repository root: node demo/parity.test.mjs
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
globalThis.self = globalThis;
require("./model.js");
const { Net, latLonToXyz } = globalThis.RadModel;

const read = (p) => readFileSync(new URL(p, import.meta.url));
const manifest = JSON.parse(read("model/model.json"));
const raw = read("model/weights.bin");
const weights = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const parity = JSON.parse(read("data/parity.json"));

const net = new Net(manifest, weights);
let worstMu = 0;
let worstVar = 0;
const t0 = performance.now();
for (const p of parity) {
  const [x, y, z] = latLonToXyz(p.lat, p.lon);
  const [mu, logvar] = net.forward(x, y, z);
  worstMu = Math.max(worstMu, Math.abs(mu - p.mu));
  worstVar = Math.max(worstVar, Math.abs(logvar - p.logvar));
}
const ms = performance.now() - t0;

console.log(`points: ${parity.length}`);
console.log(`max |mu diff|: ${worstMu.toExponential(2)}`);
console.log(`max |logvar diff|: ${worstVar.toExponential(2)}`);
console.log(`time per point: ${(ms / parity.length).toFixed(3)} ms`);
if (worstMu > 2e-3 || worstVar > 2e-3) {
  console.error("parity check failed");
  process.exit(1);
}
console.log("parity ok");
