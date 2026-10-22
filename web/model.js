// In-browser inference for the exported self-pruning models.
// Works as a browser global (window.SparseNet) and as a Node module (used by the parity test).
(function (root) {
  "use strict";

  const MEAN = [0.4914, 0.4822, 0.4465];
  const STD = [0.2470, 0.2435, 0.2616];
  const CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];

  // Slice the typed arrays for each layer out of one model's binary buffer (no copies)
  function parseModel(meta, buffer) {
    const layers = meta.layers.map((l) => {
      const [outFeatures, inFeatures] = l.shape;
      return {
        outFeatures, inFeatures,
        index: new Int32Array(buffer, l.index_offset, l.nnz),
        value: new Float32Array(buffer, l.value_offset, l.nnz),
        bias: new Float32Array(buffer, l.bias_offset, outFeatures),
      };
    });
    return { meta, layers };
  }

  // Sparse matrix-vector product: only the surviving connections are visited,
  // so the cost is proportional to the pruned model, not the dense one.
  function sparseLinear(layer, x) {
    const out = Float32Array.from(layer.bias);
    const { index, value, inFeatures } = layer;
    for (let k = 0; k < index.length; k++) {
      const flat = index[k];
      const row = (flat / inFeatures) | 0;
      out[row] += value[k] * x[flat - row * inFeatures];
    }
    return out;
  }

  // input: Float32Array of length 3072 in CHW order, already normalized
  function forward(model, input) {
    let x = input;
    model.layers.forEach((layer, i) => {
      x = sparseLinear(layer, x);
      if (i < model.layers.length - 1) {
        for (let j = 0; j < x.length; j++) if (x[j] < 0) x[j] = 0; // ReLU (dropout is a no-op at inference)
      }
    });
    return x;
  }

  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = Array.from(logits, (v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // RGBA pixels of a 32x32 image (e.g. canvas ImageData.data) -> normalized CHW input
  function preprocess(rgba) {
    const x = new Float32Array(3 * 32 * 32);
    for (let p = 0; p < 32 * 32; p++) {
      for (let c = 0; c < 3; c++) {
        x[c * 1024 + p] = (rgba[p * 4 + c] / 255 - MEAN[c]) / STD[c];
      }
    }
    return x;
  }

  function predict(model, rgba) {
    const probs = softmax(forward(model, preprocess(rgba)));
    return CLASSES.map((label, i) => ({ label, p: probs[i] })).sort((a, b) => b.p - a.p);
  }

  const api = { CLASSES, parseModel, forward, preprocess, softmax, predict };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SparseNet = api;
})(this);
