const state = {
  files: { Observe: null, Reverse: null },
  sessionId: null,
  prediction: null,
  methods: [],
  side: null,
  heatmaps: {},  // "side|method" -> response of /api/explain
};

const $ = (sel, root = document) => root.querySelector(sel);
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function setStatus(el, text, { loading = false, error = false } = {}) {
  el.classList.toggle("error", error);
  el.innerHTML = (loading ? '<span class="spinner"></span>' : "") + esc(text);
}

async function api(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`);
  return body;
}

/* ---------------- 1. upload ---------------- */

function setFile(zone, file) {
  const side = zone.dataset.side;
  const preview = $(".dz-preview", zone);
  if (preview.src) URL.revokeObjectURL(preview.src);

  state.files[side] = file;
  zone.classList.toggle("has-image", !!file);
  $(".dz-empty", zone).hidden = !!file;
  $(".dz-clear", zone).hidden = !file;
  $(".dz-tag", zone).hidden = !file;
  preview.hidden = !file;
  if (file) preview.src = URL.createObjectURL(file);
  else preview.removeAttribute("src");

  $("#predict-btn").disabled = !state.files.Observe && !state.files.Reverse;
}

document.querySelectorAll(".dropzone").forEach((zone) => {
  const input = $("input", zone);
  input.addEventListener("change", () => input.files[0] && setFile(zone, input.files[0]));

  $(".dz-clear", zone).addEventListener("click", (e) => {
    e.preventDefault();
    input.value = "";
    setFile(zone, null);
  });

  ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, () => zone.classList.remove("dragover")));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
    if (file) setFile(zone, file);
  });
});

$("#predict-btn").addEventListener("click", async () => {
  const button = $("#predict-btn");
  const status = $("#predict-status");
  const form = new FormData();
  if (state.files.Observe) form.append("observe", state.files.Observe);
  if (state.files.Reverse) form.append("reverse", state.files.Reverse);

  button.disabled = true;
  setStatus(status, "Predicting…", { loading: true });
  try {
    const result = await api("/api/predict", { method: "POST", body: form });
    state.sessionId = result.session_id;
    state.prediction = result.prediction;
    state.heatmaps = {};
    setStatus(status, "");
    renderPrediction(result.prediction);
    setupXai(result.prediction);
  } catch (err) {
    setStatus(status, err.message, { error: true });
  } finally {
    button.disabled = false;
  }
});

/* ---------------- 2. prediction + catalogue ---------------- */

function renderPrediction(p) {
  const sides = [p.observe, p.reverse].filter(Boolean).map((s) => `
    <div class="side-box">
      <h4>${esc(s.side)} model <span class="muted small">(${esc(s.run)})</span></h4>
      ${s.top_k.map(([cls, prob]) => `
        <div class="topk-row">
          <span>Class ${cls}</span>
          <div class="topk-bar"><span style="width:${(prob * 100).toFixed(1)}%"></span></div>
          <span class="pct">${pct(prob)}</span>
        </div>`).join("")}
    </div>`).join("");

  $("#prediction").innerHTML = `
    <div class="headline">
      <span class="big">Class ${p.class_number} · RRC ${esc(p.catalogue.crawford_number)}</span>
      <span class="conf">confidence <strong>${pct(p.confidence)}</strong></span>
    </div>
    <div class="muted">${esc(p.catalogue.denomination)} · ${esc(p.catalogue.issuer)} · ${esc(p.catalogue.date_range)}</div>
    ${p.sides_agree === false ? '<div class="warning">Observe and Reverse models disagree — the averaged prediction is shown.</div>' : ""}
    <div class="sides">${sides}</div>`;
  $("#catalogue").innerHTML = renderCatalogue(p.catalogue);
  $("#result-section").hidden = false;
}

function fieldList(fields) {
  const entries = Object.entries(fields || {}).filter(([, v]) => v);
  if (!entries.length) return '<p class="muted small">No data.</p>';
  return `<dl class="fields">${entries.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
}

function renderCatalogue(c) {
  return fieldList({
    "Crawford number": c.crawford_number,
    Denomination: c.denomination,
    "Date range": c.date_range,
    Issuer: c.issuer,
    Observe: c.observe_description,
    Reverse: c.reverse_description,
  });
}

/* ---------------- 3. XAI ---------------- */

function setupXai(p) {
  const sides = [p.observe, p.reverse].filter(Boolean).map((s) => s.side);
  if (!sides.includes(state.side)) state.side = sides[0];

  $("#side-select").innerHTML = sides.map((s) =>
    `<button type="button" data-side="${s}" class="${s === state.side ? "active" : ""}">${s}</button>`).join("");
  $("#compare").hidden = true;
  $("#legend").hidden = true;
  setStatus($("#explain-status"), "");
  $("#xai-section").hidden = false;
}

$("#side-select").addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (!button) return;
  state.side = button.dataset.side;
  document.querySelectorAll("#side-select button").forEach((b) => b.classList.toggle("active", b === button));
  showCachedHeatmap();
});

$("#method-select").addEventListener("change", () => {
  updateMethodInfo();
  showCachedHeatmap();
});

function updateMethodInfo() {
  const method = state.methods.find((m) => m.id === $("#method-select").value);
  $("#method-info").textContent = method?.info || "";
}

function showCachedHeatmap() {
  const cached = state.heatmaps[`${state.side}|${$("#method-select").value}`];
  if (cached) showHeatmap(cached);
  else { $("#compare").hidden = true; $("#legend").hidden = true; }
}

$("#explain-btn").addEventListener("click", async () => {
  const method = $("#method-select").value;
  const side = state.side;
  const key = `${side}|${method}`;
  if (state.heatmaps[key]) return showHeatmap(state.heatmaps[key]);

  const button = $("#explain-btn");
  const status = $("#explain-status");
  button.disabled = true;
  setStatus(status, `Generating ${$("#method-select").selectedOptions[0].text} for ${side}…`, { loading: true });
  try {
    const result = await api("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, side, method }),
    });
    state.heatmaps[key] = result;
    setStatus(status, `Done in ${result.seconds}s`);
    if (state.side === side && $("#method-select").value === method) showHeatmap(result);
  } catch (err) {
    setStatus(status, err.message, { error: true });
  } finally {
    button.disabled = false;
  }
});

function showHeatmap(result) {
  $("#compare-raw").src = result.raw;
  $("#compare-overlay").src = result.overlay;
  $("#compare-method-label").textContent = `${result.method_name} · class ${result.class_number}`;
  $("#compare").hidden = false;

  $("#legend").innerHTML = result.signed
    ? '<span>against class</span><span class="ramp bwr"></span><span>supports class</span>'
    : '<span>low importance</span><span class="ramp jet"></span><span>high importance</span>';
  $("#legend").hidden = false;
}

$("#compare-range").addEventListener("input", (e) => {
  $("#compare").style.setProperty("--pos", `${e.target.value}%`);
});

/* ---------------- init ---------------- */

api("/api/methods").then(({ methods }) => {
  state.methods = methods;
  $("#method-select").innerHTML = methods.map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join("");
  updateMethodInfo();
}).catch((err) => setStatus($("#explain-status"), `Could not load methods: ${err.message}`, { error: true }));
