/* Web worker: loads the weights once and answers batched predictions. */
importScripts("model.js");

let net = null;

async function init(base) {
  const [manifest, buf] = await Promise.all([
    fetch(base + "model/model.json").then((r) => r.json()),
    fetch(base + "model/weights.bin").then((r) => r.arrayBuffer()),
  ]);
  net = new RadModel.Net(manifest, new Float32Array(buf));
}

self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    if (type === "init") {
      await init(e.data.base);
      self.postMessage({ id, ok: true });
    } else if (type === "predict") {
      const out = net.predict(e.data.xyz);
      self.postMessage({ id, ok: true, mu: out.mu, logvar: out.logvar }, [out.mu.buffer, out.logvar.buffer]);
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err) });
  }
};
