// Onion-FL Studio: a dependency-free single-page app over the Studio API.
// Every DOM node is built with h()/s(): data never goes through innerHTML.

const view = document.getElementById("view");
const COLORS = ["#7c3aed", "#2563eb", "#16a34a", "#dc2626", "#d97706", "#0891b2", "#db2777", "#4b5563"];
const LEVEL_COLORS = ["#7c3aed", "#2563eb", "#0891b2", "#16a34a", "#d97706"];
let timers = [];

// --- helpers ---------------------------------------------------------------------------

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (key === "style") node.setAttribute("style", value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function s(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

class ApiError extends Error {
  constructor(errors) {
    super(errors.join("\n"));
    this.errors = errors;
  }
}

async function api(path, options = {}) {
  const init = { ...options };
  if (options.json !== undefined) {
    init.method = options.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(options.json);
  }
  const response = await fetch(path, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(body.errors || [body.detail || `HTTP ${response.status}`]);
  return body;
}

const fmt = (x, digits = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : typeof x === "number" ? x.toFixed(digits).replace(/\.?0+$/, "") : String(x));
const short = (id) => (id ? `${id.slice(0, 12)}…` : "–");

function errorBox(error) {
  const lines = error instanceof ApiError ? error.errors : [String(error.message || error)];
  return h("div", { class: "errors" }, lines.map((line) => h("div", {}, line)));
}

function copyable(value) {
  return h("span", { class: "id" }, value, " ", h("button", { onclick: () => navigator.clipboard?.writeText(value), title: "Copiar" }, "⧉"));
}

function page(title, subtitle, ...content) {
  view.replaceChildren(...[h("h1", {}, title), subtitle ? h("p", { class: "sub" }, subtitle) : null, ...content].filter(Boolean));
}

function every(ms, fn) {
  const id = setInterval(fn, ms);
  timers.push(id);
}

// --- drawing ------------------------------------------------------------------------------

function drawTopology(graph, linkLabels = {}) {
  // Levels in rows, leaves spread evenly, each parent centred over its children.
  const children = {};
  for (const node of graph.nodes) (children[node.parent] ||= []).push(node);
  const root = graph.nodes.find((n) => n.parent === null);
  const leaves = [];
  (function walk(node) {
    const kids = children[node.id] || [];
    if (!kids.length) leaves.push(node.id);
    kids.forEach(walk);
  })(root);
  const width = Math.max(420, leaves.length * 130);
  const rowH = 96;
  const levels = graph.levels;
  const pos = {};
  leaves.forEach((id, i) => (pos[id] = { x: ((i + 0.5) * width) / leaves.length }));
  (function place(node) {
    const kids = children[node.id] || [];
    kids.forEach(place);
    if (kids.length) pos[node.id] = { x: kids.reduce((a, k) => a + pos[k.id].x, 0) / kids.length };
    pos[node.id].y = 36 + levels.indexOf(node.level) * rowH;
  })(root);
  const height = 36 + (levels.length - 1) * rowH + 70;
  const svg = s("svg", { class: "graph", viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img" });
  const links = graph.links.map((l) => ({ ...l, key: `${l.src}->${l.dst}` }));
  for (const link of links) {
    const a = pos[link.src], b = pos[link.dst];
    svg.append(s("line", { class: "link", x1: a.x, y1: a.y, x2: b.x, y2: b.y }));
    const label = linkLabels[link.src];
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    svg.append(s("text", { class: "link-label", x: mx + 4, y: my - 2 }, `${typeof link.transport === "string" ? link.transport : link.transport.name}/${link.codec}/${typeof link.profile === "string" ? link.profile : "custom"}`));
    if (label) svg.append(s("text", { class: "groups", x: mx + 4, y: my + 11 }, label));
  }
  const edgeY = 36 + (levels.length - 1) * rowH;
  for (const leaf of leaves) {
    const p = pos[leaf];
    svg.append(s("line", { class: "link", x1: p.x, y1: p.y, x2: p.x, y2: edgeY - 18 }));
    svg.append(s("rect", { class: "edgebox", x: p.x - 50, y: edgeY - 18, width: 100, height: 30, rx: 6 }));
    svg.append(s("text", { x: p.x, y: edgeY + 1, "text-anchor": "middle" }, `edges (${levels[levels.length - 1]})`));
    if (linkLabels["*"]) svg.append(s("text", { class: "groups", x: p.x, y: edgeY + 28, "text-anchor": "middle" }, linkLabels["*"]));
  }
  for (const node of graph.nodes) {
    const p = pos[node.id];
    const color = LEVEL_COLORS[levels.indexOf(node.level) % LEVEL_COLORS.length];
    svg.append(s("circle", { cx: p.x, cy: p.y, r: 15, fill: color, opacity: 0.9 }));
    svg.append(s("text", { x: p.x, y: p.y - 21, "text-anchor": "middle", "font-weight": 600 }, node.id));
    svg.append(s("text", { x: p.x, y: p.y + 4, "text-anchor": "middle", style: "fill:#fff", "font-size": 9 }, node.role === "coordinator" ? "root" : node.level));
  }
  return svg;
}

function lineChart(series, { yLabel = "", width = 640, height = 260 } = {}) {
  const points = series.flatMap((serie) => serie.points);
  if (!points.length) return h("p", { class: "muted" }, "Sin datos todavía.");
  const pad = { l: 46, r: 14, t: 12, b: 30 };
  const xs = points.map((p) => p.x);
  const ys = points.flatMap((p) => [p.y, p.lo, p.hi]).filter((v) => typeof v === "number" && !Number.isNaN(v));
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (x0 === x1) x1 = x0 + 1;
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const X = (x) => pad.l + ((x - x0) / (x1 - x0)) * (width - pad.l - pad.r);
  const Y = (y) => height - pad.b - ((y - y0) / (y1 - y0)) * (height - pad.t - pad.b);
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img" });
  for (let i = 0; i <= 4; i++) {
    const v = y0 + ((y1 - y0) * i) / 4;
    svg.append(s("line", { class: "grid", x1: pad.l, x2: width - pad.r, y1: Y(v), y2: Y(v) }));
    svg.append(s("text", { x: pad.l - 6, y: Y(v) + 4, "text-anchor": "end" }, fmt(v, 3)));
  }
  const ticks = [...new Set(xs)].sort((a, b) => a - b);
  const step = Math.ceil(ticks.length / 10);
  ticks.filter((_, i) => i % step === 0).forEach((t) => svg.append(s("text", { x: X(t), y: height - 10, "text-anchor": "middle" }, t)));
  svg.append(s("text", { x: 4, y: 10 }, yLabel));
  series.forEach((serie, i) => {
    const color = serie.color || COLORS[i % COLORS.length];
    const sorted = [...serie.points].sort((a, b) => a.x - b.x);
    const band = sorted.filter((p) => typeof p.lo === "number" && typeof p.hi === "number");
    if (band.length) {
      const upper = band.map((p) => `${X(p.x)},${Y(p.hi)}`);
      const lower = band.map((p) => `${X(p.x)},${Y(p.lo)}`).reverse();
      svg.append(s("polygon", { points: [...upper, ...lower].join(" "), fill: color, opacity: 0.15 }));
    }
    svg.append(s("polyline", { points: sorted.map((p) => `${X(p.x)},${Y(p.y)}`).join(" "), fill: "none", stroke: color, "stroke-width": 2 }));
    sorted.forEach((p) => svg.append(s("circle", { cx: X(p.x), cy: Y(p.y), r: 2.5, fill: color }, s("title", {}, `${serie.label}: ronda ${p.x} → ${fmt(p.y)}`))));
  });
  return h("div", {}, svg, h("div", { class: "legend" }, series.map((serie, i) => h("span", { style: `--c:${serie.color || COLORS[i % COLORS.length]}` }, serie.label))));
}

function stackedBars(composition) {
  const leaves = Object.keys(composition);
  const datasets = [...new Set(leaves.flatMap((leaf) => Object.keys(composition[leaf].datasets || {})))].sort();
  const max = Math.max(1, ...leaves.map((leaf) => Object.values(composition[leaf].datasets || {}).reduce((a, b) => a + b, 0)));
  const rowH = 26, width = 560, labelW = 110;
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${width} ${leaves.length * rowH + 8}`, width: "100%", role: "img" });
  leaves.forEach((leaf, i) => {
    let x = labelW;
    svg.append(s("text", { x: 0, y: i * rowH + 17 }, leaf));
    datasets.forEach((d, j) => {
      const value = (composition[leaf].datasets || {})[d] || 0;
      const w = (value / max) * (width - labelW - 60);
      if (value) svg.append(s("rect", { x, y: i * rowH + 4, width: w, height: 18, fill: COLORS[j % COLORS.length], rx: 3 }, s("title", {}, `${d}: ${value}`)));
      x += w;
    });
    const entropy = composition[leaf].entropy;
    svg.append(s("text", { x: x + 6, y: i * rowH + 17, class: "muted" }, entropy !== undefined ? `H=${fmt(entropy, 2)}` : ""));
  });
  return h("div", {}, svg, h("div", { class: "legend" }, datasets.map((d, j) => h("span", { style: `--c:${COLORS[j % COLORS.length]}` }, d))));
}

function table(columns, rows, onclick) {
  return h(
    "table",
    {},
    h("thead", {}, h("tr", {}, columns.map(([label]) => h("th", {}, label)))),
    h("tbody", {}, rows.map((row) => h("tr", { class: onclick ? "clickable" : "", onclick: onclick ? () => onclick(row) : null }, columns.map(([, render]) => h("td", {}, render(row)))))),
  );
}

function groupLabels(links) {
  // Short labels on the drawing; the full lists go in trafficTable.
  return Object.fromEntries(links.map((l) => [l.child, l.groups.length ? `${l.groups.length} grupo(s)` : "nada"]));
}

function trafficTable(links) {
  return table([["Enlace", (l) => (l.child === "*" ? `edges → ${l.parent_level}` : `${l.child} → ${l.parent}`)], ["Grupos que viajan", (l) => (l.groups.join(", ") || "nada")]], links);
}

// --- topologies -------------------------------------------------------------------------------

async function topologiesView(selected) {
  const list = await api("/api/topologies");
  const detail = h("div", {});
  const cards = list.map((t) =>
    h("button", { class: `card${t.name === selected ? " selected" : ""}`, onclick: () => (location.hash = `#/topologies/${t.name}`) },
      h("div", { class: "title" }, t.name),
      h("div", { class: "meta" }, `${t.levels.join(" → ")} · ${t.nodes} nodos · ${t.leaves} hojas`),
      h("div", { class: "meta mono" }, short(t.topology_id))));
  page("Topologías", "La biblioteca de topologies/. Cada una con su grafo, su topology_id y un editor que valida al momento.",
    h("div", { class: "split" },
      h("div", { class: "list" }, cards, h("button", { onclick: () => openEditor(detail, null) }, "+ Nueva topología")),
      detail));
  if (selected) await showTopology(detail, selected);
  else if (list.length) location.hash = `#/topologies/${list[0].name}`;
}

async function showTopology(container, name) {
  const body = await api(`/api/topologies/${encodeURIComponent(name)}`);
  const content = h("div", {});
  const tabs = { Grafo: () => graphPanel(body.graph), YAML: () => h("pre", {}, body.yaml), Editor: () => openEditor(content, body.graph, name) };
  const buttons = Object.keys(tabs).map((label) =>
    h("button", { class: "tab", onclick: (e) => { buttons.forEach((b) => b.classList.remove("active")); e.target.classList.add("active"); const out = tabs[label](); if (out) content.replaceChildren(out); } }, label));
  buttons[0].classList.add("active");
  content.replaceChildren(graphPanel(body.graph));
  container.replaceChildren(h("div", { class: "panel" }, h("h2", {}, name), h("div", { class: "tabs" }, buttons), content));
}

function graphPanel(graph) {
  return h("div", {}, drawTopology(graph), h("p", { class: "muted" }, "topology_id: ", copyable(graph.topology_id)));
}

function editorState(graph) {
  if (!graph) {
    return { name: "nueva", levels: ["global", "fog", "edge"], root: "cloud", nodes: { fog: [{ id: "fog_1", parent: "cloud", settings: {} }, { id: "fog_2", parent: "cloud", settings: {} }] }, links: { fog: { transport: "mqtt", codec: "json", profile: "wifi" } }, edge: { transport: "mqtt", codec: "json", profile: "4g" }, edgeSettings: {}, rootSettings: {} };
  }
  const state = { name: graph.name, levels: [...graph.levels], nodes: {}, links: {}, edge: { ...graph.edge.link_up }, edgeSettings: { ...graph.edge.settings } };
  for (const node of graph.nodes) {
    if (node.parent === null) { state.root = node.id; state.rootSettings = { ...node.settings }; continue; }
    (state.nodes[node.level] ||= []).push({ id: node.id, parent: node.parent, settings: { ...node.settings } });
    const link = graph.links.find((l) => l.src === node.id);
    if (link && !state.links[node.level]) state.links[node.level] = { transport: link.transport, codec: link.codec, profile: link.profile };
  }
  return state;
}

function compactFrom(state) {
  const out = { name: state.name, levels: state.levels, root: { id: state.root, ...state.rootSettings } };
  for (const level of state.levels.slice(1, -1)) {
    out[level] = {
      defaults: { link_up: state.links[level] || { transport: "mqtt", codec: "json", profile: "lan" } },
      nodes: (state.nodes[level] || []).map((n) => ({ id: n.id, parent: n.parent, ...n.settings })),
    };
  }
  out[state.levels[state.levels.length - 1]] = { link_up: state.edge, ...state.edgeSettings };
  return out;
}

function openEditor(container, graph, name) {
  const state = editorState(graph);
  const preview = h("div", {});
  const errors = h("div", { class: "errors" });
  const form = h("div", {});
  const profiles = ["lan", "wifi", "4g", "lora"];

  async function refresh() {
    errors.replaceChildren();
    try {
      const { graph: next } = await api("/api/topologies/validate", { json: compactFrom(state) });
      preview.replaceChildren(drawTopology(next), h("p", { class: "muted" }, "topology_id: ", copyable(next.topology_id)));
    } catch (error) {
      errors.replaceChildren(...errorBox(error).childNodes);
    }
  }

  function render() {
    const aggregationLevels = state.levels.slice(1, -1);
    const select = (value, options, onchange) => h("select", { onchange: (e) => { onchange(e.target.value); render(); refresh(); } }, options.map((o) => h("option", { value: o, selected: o === value }, o)));
    form.replaceChildren(
      h("div", { class: "row" },
        h("label", {}, "Nombre", h("input", { value: state.name, oninput: (e) => (state.name = e.target.value) })),
        h("label", {}, "Raíz", h("input", { value: state.root, onchange: (e) => { const old = state.root; state.root = e.target.value; (state.nodes[aggregationLevels[0]] || []).forEach((n) => n.parent === old && (n.parent = state.root)); render(); refresh(); } })),
        h("button", { onclick: () => { const level = `nivel_${state.levels.length - 1}`; state.levels.splice(state.levels.length - 1, 0, level); state.nodes[level] = []; render(); refresh(); } }, "+ Nivel de agregación")),
      ...aggregationLevels.map((level, index) => {
        const parents = index === 0 ? [state.root] : (state.nodes[aggregationLevels[index - 1]] || []).map((n) => n.id);
        const link = (state.links[level] ||= { transport: "mqtt", codec: "json", profile: "lan" });
        return h("div", { class: "editor-level" },
          h("div", { class: "row" },
            h("strong", {}, level),
            h("label", {}, "perfil", select(link.profile, profiles, (v) => (link.profile = v))),
            h("label", {}, "codec", select(link.codec, ["json", "npz"], (v) => (link.codec = v))),
            h("label", {}, "transporte", select(typeof link.transport === "string" ? link.transport : link.transport.name, ["mqtt", "memory"], (v) => (link.transport = v))),
            aggregationLevels.length > 1 ? h("button", { onclick: () => { state.levels = state.levels.filter((l) => l !== level); delete state.nodes[level]; render(); refresh(); } }, "Quitar nivel") : null),
          (state.nodes[level] || []).map((node, i) =>
            h("div", { class: "row" },
              h("input", { value: node.id, onchange: (e) => { node.id = e.target.value; render(); refresh(); } }),
              h("label", {}, "padre", select(node.parent, parents, (v) => (node.parent = v))),
              h("label", {}, "casa", h("input", { value: node.settings.home || "", placeholder: "dataset", onchange: (e) => { if (e.target.value) node.settings.home = e.target.value; else delete node.settings.home; refresh(); } })),
              h("button", { onclick: () => { state.nodes[level].splice(i, 1); render(); refresh(); } }, "×"))),
          h("button", { onclick: () => { state.nodes[level].push({ id: `${level}_${(state.nodes[level] || []).length + 1}`, parent: parents[0], settings: {} }); render(); refresh(); } }, "+ Nodo"));
      }),
      h("div", { class: "row" }, h("strong", {}, "edges"), h("label", {}, "perfil", select(state.edge.profile, profiles, (v) => (state.edge.profile = v)))),
      h("div", { class: "row" },
        h("button", { class: "primary", onclick: async () => {
          const target = prompt("Guardar como (topologies/<nombre>.yaml):", name || state.name);
          if (!target) return;
          try { await api(`/api/topologies/${encodeURIComponent(target)}`, { json: compactFrom(state), method: "PUT" }); location.hash = `#/topologies/${target}`; render(); }
          catch (error) { errors.replaceChildren(...errorBox(error).childNodes); }
        } }, "Guardar")));
  }

  render();
  refresh();
  // From a tab the editor sits inside the topology's panel; from "+ Nueva" it is its own panel.
  const wrapper = h("div", { class: name ? "" : "panel" }, h("h3", {}, name ? `Editar ${name}` : "Nueva topología"), form, errors, h("h3", {}, "Vista previa"), preview);
  container.replaceChildren(wrapper);
  return null;
}

// --- experiments -----------------------------------------------------------------------------

async function experimentsView(selected) {
  const list = await api("/api/experiments");
  const detail = h("div", {});
  page("Experimentos", "Los experimentos de experiments/: escenarios del barrido, plan en seco y lanzamiento.",
    h("div", { class: "split" },
      h("div", { class: "list" }, list.map((e) =>
        h("button", { class: `card${e.name === selected ? " selected" : ""}`, onclick: () => (location.hash = `#/experiments/${e.name}`) },
          h("div", { class: "title" }, e.name),
          h("div", { class: "meta" }, e.description || ""),
          h("div", { class: "meta" }, `${e.topology} · ${e.scenarios} escenario(s) × ${e.seeds} semilla(s)`)))),
      detail));
  if (selected) await showExperiment(detail, selected);
}

async function showExperiment(container, name) {
  const body = await api(`/api/experiments/${encodeURIComponent(name)}`);
  const output = h("div", {});
  const mode = h("select", {}, h("option", { value: "sim" }, "simulación"), h("option", { value: "real" }, "real (MQTT)"));
  const workers = h("input", { type: "number", min: 1, value: 1 });
  const scenarioSelect = h("select", {}, h("option", { value: "" }, "todos"), [...new Set(body.scenarios.map((x) => x.name))].map((n) => h("option", { value: n }, n)));
  container.replaceChildren(
    h("div", { class: "panel" },
      h("h2", {}, name),
      table([["Escenario", (r) => r.name], ["Semilla", (r) => r.seed], ["config_id", (r) => h("span", { class: "mono" }, short(r.config_id))]], body.scenarios),
      h("div", { class: "row" },
        h("button", { onclick: () => planExperiment(name, output) }, "Plan en seco"),
        h("label", {}, "modo", mode), h("label", {}, "trabajadores", workers), h("label", {}, "escenario", scenarioSelect),
        h("button", { class: "primary", onclick: async () => {
          try {
            const { pid } = await api(`/api/experiments/${encodeURIComponent(name)}/run`, { json: { mode: mode.value, workers: Number(workers.value), scenario: scenarioSelect.value || null } });
            output.replaceChildren(h("p", { class: "ok" }, `Lanzado (pid ${pid}). Sigue el progreso en `, h("a", { href: "#/runs" }, "Ejecuciones"), "."));
          } catch (error) { output.replaceChildren(errorBox(error)); }
        } }, "Ejecutar")),
      output),
    h("div", { class: "panel" }, h("h3", {}, "YAML"), h("pre", {}, body.yaml)));
}

async function planExperiment(name, output) {
  output.replaceChildren(h("p", { class: "muted" }, "Calculando el plan…"));
  try {
    const previews = await api(`/api/experiments/${encodeURIComponent(name)}/plan`, { json: {} });
    output.replaceChildren(...previews.map((p) =>
      h("div", { class: "panel" },
        h("h3", {}, `${p.scenario} · semilla ${p.seed}`),
        p.warnings?.length ? h("div", { class: "errors" }, p.warnings.map((w) => h("div", {}, "⚠ ", w))) : "",
        h("p", { class: "muted" }, "Composición por hoja (barras por dataset, H = entropía de la mezcla)"),
        stackedBars(p.composition),
        h("p", { class: "muted" }, "Grupos que viajan por cada enlace"),
        drawTopology(p.graph, groupLabels(p.traffic)),
        trafficTable(p.traffic),
        h("p", { class: "muted" }, "Roles: ", Object.entries(p.roles).map(([ds, r]) => `${ds}: ${r.test.length} test · ${r.val.length} val · ${r.train.length} train`).join(" | ")))));
  } catch (error) {
    output.replaceChildren(errorBox(error));
  }
}

// --- runs ---------------------------------------------------------------------------------------

async function runsView(selected) {
  if (selected) return runDetail(selected);
  const filter = h("select", { onchange: () => load() }, h("option", { value: "" }, "todos los estados"), ["running", "finished", "incomplete", "failed"].map((st) => h("option", { value: st }, st)));
  const holder = h("div", {});
  async function load() {
    const runs = await api(`/api/runs${filter.value ? `?status=${filter.value}` : ""}`);
    holder.replaceChildren(runs.length
      ? table([
          ["Ejecución", (r) => h("span", { class: "mono" }, r.run_id)],
          ["Experimento", (r) => r.experiment || "–"],
          ["Escenario", (r) => r.scenario || "–"],
          ["Semilla", (r) => r.seed],
          ["Estado", (r) => h("span", { class: `badge ${r.status}` }, r.status)],
          ["Rondas", (r) => r.rounds ?? "–"],
          ["Global (último)", (r) => { const g = r.final?.["global/*"]; return g ? fmt(g.accuracy ?? g.loss) : "–"; }],
        ], runs, (r) => (location.hash = `#/runs/${r.run_id}`))
      : h("p", { class: "muted" }, "No hay ejecuciones en runs/."));
    return runs;
  }
  page("Ejecuciones", "Lo ejecutado en runs/: estado, identidad y rendimiento por nivel. Las que siguen corriendo se actualizan solas.", h("div", { class: "row" }, filter), h("div", { class: "panel" }, holder));
  const runs = await load();
  if (runs.some((r) => r.status === "running")) every(3000, load);
}

async function runDetail(runId) {
  const detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
  const meta = detail.meta, summary = detail.summary || {};
  const charts = h("div", { class: "grid2" });
  const log = h("div", { class: "log" });
  let after = 0;

  async function seriesPanel(title, params, yLabel) {
    const query = new URLSearchParams(params).toString();
    const points = await api(`/api/runs/${encodeURIComponent(runId)}/series?${query}`);
    const series = points.length
      ? [{ label: `${title} (media)`, points: points.map((p) => ({ x: p.round, y: p.mean, lo: p.count > 1 ? p.min : undefined, hi: p.count > 1 ? p.max : undefined })) }]
      : [];
    return h("div", { class: "panel" }, h("h3", {}, title), lineChart(series, { yLabel }));
  }

  async function drawCharts() {
    const levels = detail.levels || ["global"];
    const panels = [await seriesPanel("Global · accuracy", { level: levels[0], name: "accuracy", model: "global", dataset: "*" }, "accuracy"),
      await seriesPanel("Global · pérdida de entrenamiento", { level: levels[0], name: "train_loss" }, "loss")];
    for (const level of levels.slice(1, -1)) {
      panels.push(await seriesPanel(`${level} · modelo de zona`, { level, name: "accuracy", model: "zone" }, "accuracy"));
      panels.push(await seriesPanel(`${level} · divergencia (coseno)`, { level, name: "divergence_cos" }, "coseno"));
    }
    panels.push(await seriesPanel("edge · modelo entrenado (local_val)", { level: levels[levels.length - 1], name: "accuracy", model: "local" }, "accuracy"));
    charts.replaceChildren(...panels);
  }

  async function poll() {
    const page_ = await api(`/api/runs/${encodeURIComponent(runId)}/events?after=${after}&limit=500`);
    after = page_.next;
    for (const event of page_.events.slice(-60)) {
      log.append(h("div", {}, `${fmt(event.t_virtual, 3)}s ${event.node} ${event.name} ${event.value ?? ""}`));
    }
    while (log.childNodes.length > 200) log.firstChild.remove();
    log.scrollTop = log.scrollHeight;
    return page_;
  }

  page(`Ejecución ${runId}`, meta.scenario ? `${meta.config?.name || ""} · ${meta.scenario} · semilla ${meta.seed}` : "",
    h("div", { class: "panel" },
      h("div", { class: "kpis" },
        [["Estado", h("span", { class: `badge ${meta.status}` }, meta.status)], ["Rondas", summary.rounds ?? "–"], ["Mensajes", summary.messages?.sent ?? "–"], ["Bytes", summary.messages?.bytes ?? "–"], ["Perdidos", summary.messages?.dropped ?? "–"], ["Quórum fallido", summary.quorum_failed ?? "–"]]
          .map(([label, value]) => h("div", { class: "kpi" }, h("div", { class: "value" }, value), h("div", { class: "label" }, label)))),
      h("h3", {}, "Identidad"),
      table([["", (r) => r[0]], ["", (r) => copyable(r[1] || "–")]], [["run_id", meta.run_id], ["topology_id", meta.topology_id], ["config_id", meta.config_id], ["data_id", meta.data_id], ["code", meta.code_version?.commit], ["run_hash", meta.run_hash]])),
    Object.keys(detail.composition || {}).length ? h("div", { class: "panel" }, h("h3", {}, "Composición por hoja"), stackedBars(detail.composition)) : null,
    charts,
    h("div", { class: "panel" }, h("h3", {}, "Eventos"), log));
  await drawCharts();
  await poll();
  if (meta.status === "running") {
    every(2000, async () => {
      const page_ = await poll();
      if (page_.events.length) await drawCharts();
    });
  }
}

// --- compare --------------------------------------------------------------------------------------

async function compareView() {
  const experiments = await api("/api/experiments");
  const experiment = h("select", {}, h("option", { value: "" }, "todas las ejecuciones"), experiments.map((e) => h("option", { value: e.name }, e.name)));
  const level = h("input", { value: "global", size: 10 });
  const metric = h("select", {}, ["accuracy", "loss", "macro_f1", "train_loss", "divergence_cos", "dataset_conflict", "participation"].map((m) => h("option", { value: m }, m)));
  const byChoices = ["topology_id", "scenario", "config_id", "dataset", "model"];
  const by = byChoices.map((key) => h("input", { type: "checkbox", value: key, checked: key === "topology_id" }));
  const output = h("div", {});
  async function run() {
    const keys = by.filter((b) => b.checked).map((b) => b.value).join(",");
    const query = new URLSearchParams({ level: level.value, metric: metric.value, by: keys });
    if (experiment.value) query.set("experiment", experiment.value);
    try {
      const rows = await api(`/api/compare?${query}`);
      if (!rows.length) return output.replaceChildren(h("p", { class: "muted" }, "Nada que comparar con esos filtros."));
      const keyCols = keys.split(",").filter(Boolean);
      const label = (r) => keyCols.map((k) => `${k}=${k.endsWith("_id") ? short(r[k]) : r[k] ?? "∅"}`).join(" · ");
      const groups = {};
      for (const row of rows) (groups[label(row)] ||= []).push(row);
      const series = Object.entries(groups).map(([name, items]) => ({ label: name, points: items.map((r) => ({ x: r.round, y: r.mean, lo: r.ci_low ?? undefined, hi: r.ci_high ?? undefined })) }));
      const last = Object.values(groups).map((items) => items.reduce((a, b) => (b.round > a.round ? b : a)));
      output.replaceChildren(
        h("div", { class: "panel" }, h("h3", {}, `${metric.value} en ${level.value}: media ± IC 95 % sobre las semillas`), lineChart(series, { yLabel: metric.value })),
        h("div", { class: "panel" }, h("h3", {}, "Última ronda"),
          table([["Grupo", label], ["Ronda", (r) => r.round], ["n", (r) => r.n], ["Media", (r) => fmt(r.mean)], ["IC 95 %", (r) => `${fmt(r.ci_low)} – ${fmt(r.ci_high)}`], ["Nodos min/max", (r) => `${fmt(r.node_min)} / ${fmt(r.node_max)}`], ["Dispersión", (r) => fmt(r.node_spread)]], last)));
    } catch (error) {
      output.replaceChildren(errorBox(error));
    }
  }
  page("Comparar", "Entre topologías, escenarios o datasets, en cualquier nivel: media ± IC sobre las semillas por ronda y la dispersión entre los nodos del nivel.",
    h("div", { class: "panel" },
      h("div", { class: "row" }, h("label", {}, "experimento", experiment), h("label", {}, "nivel", level), h("label", {}, "métrica", metric)),
      h("div", { class: "row" }, "agrupar por:", byChoices.map((key, i) => h("label", {}, by[i], key))),
      h("button", { class: "primary", onclick: run }, "Comparar")),
    output);
  run();
}

// --- tutorial --------------------------------------------------------------------------------------

async function tutorialView() {
  const [axes, topologies] = await Promise.all([api("/api/tutorial"), api("/api/topologies")]);
  page("Tutorial", "Qué decide cada eje del experimento, con una previsualización en seco que no entrena nada.",
    ...axes.map((axis) => h("div", { class: "panel" },
      h("h2", {}, axis.title, " ", h("span", { class: "muted mono" }, axis.kind)),
      h("p", {}, axis.explain),
      axis.preview ? previewWidget(axis, topologies) : null,
      h("details", {}, h("summary", {}, `${axis.plugins.length} opción(es)`),
        table([["Nombre", (p) => h("span", { class: "mono" }, p.name)], ["Qué es", (p) => h("div", {}, h("strong", {}, p.title), h("div", {}, p.description), p.explain ? h("div", { class: "muted" }, p.explain) : null)], ["Parámetros", (p) => paramsOf(p.params)]], axis.plugins)))));
}

function paramsOf(schema) {
  const props = schema?.properties || {};
  const names = Object.keys(props);
  if (!names.length) return h("span", { class: "muted" }, "ninguno");
  return h("div", {}, names.map((n) => h("div", { class: "mono" }, `${n}${props[n].default !== undefined ? ` = ${JSON.stringify(props[n].default)}` : ""}`, props[n].description ? h("span", { class: "muted" }, ` · ${props[n].description}`) : null)));
}

function previewWidget(axis, topologies) {
  const out = h("div", {});
  const topologySelect = h("select", {}, topologies.map((t) => h("option", { value: t.name }, t.name)));
  async function topologyBody() {
    const body = await api(`/api/topologies/${encodeURIComponent(topologySelect.value)}`);
    const graph = body.graph;
    const general = { name: graph.name, levels: graph.levels, nodes: graph.nodes.map((n) => ({ id: n.id, level: n.level, parent: n.parent, link_up: graph.links.find((l) => l.src === n.id) ? (({ transport, codec, profile }) => ({ transport, codec, profile }))(graph.links.find((l) => l.src === n.id)) : null, settings: n.settings })), edge: { link_up: graph.edge.link_up, settings: graph.edge.settings } };
    return general;
  }

  if (axis.preview === "sharing") {
    const preset = h("select", {}, axis.plugins.filter((p) => p.name !== "custom").map((p) => h("option", { value: p.name }, p.name)));
    const datasets = h("input", { value: "swell,sweet", size: 14 });
    const draw = async () => {
      try {
        const topology = await topologyBody();
        const preview = await api("/api/preview/sharing", { json: { topology, sharing: preset.value, datasets: datasets.value.split(",").map((d) => d.trim()).filter(Boolean) } });
        out.replaceChildren(
          drawTopology({ ...topology, topology_id: "", links: topology.nodes.filter((n) => n.link_up).map((n) => ({ src: n.id, dst: n.parent, ...n.link_up })), nodes: topology.nodes.map((n) => ({ ...n, role: n.parent === null ? "coordinator" : "aggregator" })), edge: topology.edge },
            groupLabels(preview.links)),
          trafficTable(preview.links),
          h("p", { class: "muted" }, "Lo que guarda cada nivel: ", Object.entries(preview.held).map(([level, groups]) => `${level}: ${groups.join(", ") || "nada"}`).join(" | ")),
          Object.keys(preview.model_requirements).length ? h("p", { class: "muted" }, "Requisitos del modelo: ", JSON.stringify(preview.model_requirements)) : "");
      } catch (error) { out.replaceChildren(errorBox(error)); }
    };
    [preset, datasets, topologySelect].forEach((el) => el.addEventListener("change", draw));
    draw();
    return h("div", {}, h("div", { class: "row" }, h("label", {}, "topología", topologySelect), h("label", {}, "compartición", preset), h("label", {}, "datasets", datasets)), out);
  }

  if (axis.preview === "placement") {
    const plugin = h("select", {}, ["mixing", "dirichlet", "label_skew", "pooled"].map((n) => h("option", { value: n }, n)));
    const value = h("input", { type: "range", min: 0, max: 1, step: 0.05, value: 0.5 });
    const valueLabel = h("span", { class: "mono" }, "0.5");
    const counts = h("input", { value: "swell=20,sweet=20", size: 18 });
    const draw = async () => {
      valueLabel.textContent = value.value;
      const name = plugin.value;
      const params = name === "mixing" ? { alpha: Number(value.value) } : name === "pooled" ? {} : { beta: Math.max(0.01, Number(value.value) * 5) };
      try {
        const datasets = Object.fromEntries(counts.value.split(",").map((pair) => pair.split("=")).filter((p) => p.length === 2).map(([k, v]) => [k.trim(), Number(v)]));
        const composition = await api("/api/preview/placement", { json: { topology: await topologyBody(), placement: { name, ...params }, datasets } });
        out.replaceChildren(stackedBars(composition), h("p", { class: "muted" }, "Casa de cada hoja: ", Object.entries(composition).map(([leaf, c]) => `${leaf}→${c.home}`).join(", ")));
      } catch (error) { out.replaceChildren(errorBox(error)); }
    };
    [plugin, value, counts, topologySelect].forEach((el) => el.addEventListener(el === value ? "input" : "change", draw));
    draw();
    return h("div", {}, h("div", { class: "row" }, h("label", {}, "topología", topologySelect), h("label", {}, "reparto", plugin), h("label", {}, "α / β", value, valueLabel), h("label", {}, "sujetos", counts)), out);
  }

  if (axis.preview === "link") {
    const size = h("input", { type: "number", value: 100000, min: 1 });
    const draw = async () => {
      try {
        const rows = await Promise.all(axis.plugins.map(async (p) => ({ name: p.name, ...(await api("/api/preview/link", { json: { profile: p.name, bytes: Number(size.value) } })) })));
        out.replaceChildren(table([["Perfil", (r) => r.name], ["Latencia p50", (r) => `${fmt(r.latency.p50 * 1000, 1)} ms`], ["p95", (r) => `${fmt(r.latency.p95 * 1000, 1)} ms`], ["Subida", (r) => `${fmt(r.transmission_up_s, 3)} s`], ["Bajada", (r) => `${fmt(r.transmission_down_s, 3)} s`], ["Pérdida", (r) => `${fmt(r.loss * 100, 2)} %`]], rows));
      } catch (error) { out.replaceChildren(errorBox(error)); }
    };
    size.addEventListener("change", draw);
    draw();
    return h("div", {}, h("div", { class: "row" }, h("label", {}, "tamaño del mensaje (bytes)", size)), out);
  }
  return null;
}

// --- router ----------------------------------------------------------------------------------------

const routes = { topologies: topologiesView, experiments: experimentsView, runs: runsView, compare: compareView, tutorial: tutorialView };

async function route() {
  timers.forEach(clearInterval);
  timers = [];
  const [name, ...rest] = location.hash.replace(/^#\/?/, "").split("/");
  const key = routes[name] ? name : "topologies";
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === key));
  try {
    await routes[key](rest.map(decodeURIComponent).join("/") || undefined);
  } catch (error) {
    page("Algo falló", "", errorBox(error));
  }
}

window.addEventListener("hashchange", route);
route();
