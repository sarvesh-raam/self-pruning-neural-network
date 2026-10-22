(function () {
  "use strict";
  const EXAMPLES = ["0_airplane", "1_automobile", "2_bird", "3_cat", "4_deer", "5_dog", "6_frog", "7_horse", "8_ship", "9_truck"];
  const $ = (id) => document.getElementById(id);
  const preview = $("preview");
  const ctx = preview.getContext("2d", { willReadFrequently: true });
  let models = [];

  const fmtLambda = (l) => l.toExponential(0);
  const pct = (v, d = 1) => `${v.toFixed(d)}%`;
  const fmtBytes = (b) => (b >= 1e6 ? `${(b / 1e6).toFixed(2)} MB` : `${Math.round(b / 1e3)} KB`);
  const el = (tag, attrs = {}, text) => {
    const node = Object.assign(document.createElement(tag), attrs);
    if (text !== undefined) node.textContent = text;
    return node;
  };

  async function load() {
    const manifest = await (await fetch("models/manifest.json")).json();
    models = await Promise.all(manifest.models.map(async (meta) => {
      const buffer = await (await fetch(meta.file)).arrayBuffer();
      return SparseNet.parseModel(meta, buffer);
    }));
    renderStats();
    renderTable();
    renderModelCards();
    renderExamples();
    loadImage("img/examples/3_cat.png");
  }

  function renderStats() {
    const mid = models[Math.floor(models.length / 2)].meta;
    const stats = [
      [pct(mid.sparsity), `of weights removed (λ = ${fmtLambda(mid.lambda)})`],
      [pct(mid.accuracy), "test accuracy after pruning"],
      [`${Math.round(mid.compression)}×`, "fewer weights"],
      [fmtBytes(mid.bytes), "model download"],
    ];
    $("stats").replaceChildren(...stats.map(([value, label]) => {
      const s = el("div", { className: "stat" });
      s.append(el("b", {}, value), el("span", {}, label));
      return s;
    }));
  }

  function renderTable() {
    const head = el("tr");
    ["λ", "Test accuracy", "Sparsity", "Weights kept", "Download"].forEach((h) => head.append(el("th", { scope: "col" }, h)));
    const rows = models.map(({ meta }) => {
      const tr = el("tr");
      [fmtLambda(meta.lambda), pct(meta.accuracy, 2), pct(meta.sparsity, 2),
       `${meta.weights_remaining.toLocaleString()} / ${meta.weights_total.toLocaleString()}`, fmtBytes(meta.bytes)]
        .forEach((v) => tr.append(el("td", {}, v)));
      return tr;
    });
    $("table").replaceChildren(el("thead"), el("tbody"));
    $("table").tHead.append(head);
    $("table").tBodies[0].append(...rows);
  }

  function renderModelCards() {
    $("models").replaceChildren(...models.map(({ meta }, i) => {
      const card = el("div", { className: "panel model", id: `model-${i}` });
      const h = el("h3", {}, `λ = ${fmtLambda(meta.lambda)}`);
      h.append(el("small", { className: "time" }, ""));
      card.append(h, el("p", { className: "meta" },
        `${pct(meta.sparsity)} pruned · ${meta.weights_remaining.toLocaleString()} weights · ${pct(meta.accuracy)} test accuracy`));
      for (let r = 0; r < 3; r++) {
        const row = el("div", { className: "row" });
        const track = el("div", { className: "track" });
        track.append(el("div", { className: "fill" }));
        row.append(el("span", { className: "label" }, "–"), track, el("span", { className: "pct" }, ""));
        card.append(row);
      }
      return card;
    }));
  }

  function renderExamples() {
    $("examples").replaceChildren(...EXAMPLES.map((name) => {
      const label = name.split("_")[1];
      const b = el("button", { type: "button", title: label, ariaLabel: `Example: ${label}` });
      b.append(el("img", { src: `img/examples/${name}.png`, alt: label, loading: "lazy" }));
      b.addEventListener("click", () => loadImage(`img/examples/${name}.png`));
      return b;
    }));
  }

  // Center-crop to a square and downscale to the 32x32 input the network was trained on
  function draw(img) {
    const side = Math.min(img.naturalWidth, img.naturalHeight);
    const sx = (img.naturalWidth - side) / 2, sy = (img.naturalHeight - side) / 2;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, 32, 32);
    ctx.drawImage(img, sx, sy, side, side, 0, 0, 32, 32);
    classify(ctx.getImageData(0, 0, 32, 32).data);
  }

  function loadImage(src) {
    const img = new Image();
    img.onload = () => draw(img);
    img.src = src;
  }

  function classify(rgba) {
    models.forEach((model, i) => {
      const start = performance.now();
      const top = SparseNet.predict(model, rgba).slice(0, 3);
      const ms = performance.now() - start;
      const card = $(`model-${i}`);
      card.querySelector(".time").textContent = `${ms < 1 ? "<1" : ms.toFixed(1)} ms`;
      card.querySelectorAll(".row").forEach((row, r) => {
        row.querySelector(".label").textContent = top[r].label;
        row.querySelector(".fill").style.width = `${(top[r].p * 100).toFixed(1)}%`;
        row.querySelector(".pct").textContent = pct(top[r].p * 100, 0);
      });
    });
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { draw(img); URL.revokeObjectURL(url); };
    img.src = url;
  }

  const drop = $("drop"), input = $("file");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => handleFile(input.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); handleFile(e.dataTransfer.files[0]); });

  load().catch((err) => {
    $("models").replaceChildren(el("p", { className: "empty" }, `Could not load the models: ${err.message}`));
  });
})();
