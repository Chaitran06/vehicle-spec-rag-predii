// F-150 Spec Finder — UI logic. No framework, no build step, so the same
// folder runs from `specrag serve` locally and as a static site on Vercel.

const API = (window.SPECRAG_API_BASE || "").replace(/\/$/, "");
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = { result: null, view: "sheet", meta: null };

// ---------------------------------------------------------------- helpers
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Wrap each occurrence of the given values in <mark>, matching whole numbers
// only, so "35" never lights up inside "135" or "3.5".
function markValues(text, values) {
  let html = esc(text);
  const uniq = [...new Set(values.filter(Boolean).map(String))].sort((a, b) => b.length - a.length);
  for (const v of uniq) {
    const pattern = esc(v).replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/-/g, "\\s?-\\s?");
    html = html.replace(new RegExp(`(?<![\\w.])(${pattern})(?![\\w]|\\.\\d)`, "g"), "<mark>$1</mark>");
  }
  return html;
}

async function api(path, body) {
  const res = await fetch(API + path, body ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  } : undefined);
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) {
    const detail = data && (typeof data.detail === "string" ? data.detail : data.detail?.[0]?.msg);
    throw new Error(detail || `The server answered ${res.status}.`);
  }
  return data;
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function slug(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 50) || "specs";
}

// The assignment's output shape, plus the fields that make a value checkable.
function assignmentJson(result) {
  return result.specs.map((s) => ({
    component: s.component,
    spec_type: s.spec_type,
    value: s.value,
    unit: s.unit,
    ...(s.alt_value ? { alt_value: s.alt_value, alt_unit: s.alt_unit } : {}),
    ...(s.condition ? { condition: s.condition } : {}),
    section: s.section,
    section_title: s.section_title,
    procedure: s.procedure,
    pages: s.pages,
    evidence: s.evidence,
    verified: s.verified,
  }));
}

function toCsv(result) {
  const cols = ["component", "spec_type", "value", "unit", "alt_value", "alt_unit", "condition",
    "section", "section_title", "procedure", "pages", "evidence", "verified"];
  const cell = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(","), ...result.specs.map((s) => cols.map((c) => cell(s[c])).join(","))].join("\n");
}

// ---------------------------------------------------------------- status
function setStatus(html, kind = "") {
  const el = $("#status");
  el.className = `status ${kind}`.trim();
  el.innerHTML = html;
}

// ---------------------------------------------------------------- render
function renderResult(result) {
  state.result = result;
  $("#results").hidden = false;

  const specs = result.specs;
  const sections = [...new Set(specs.map((s) => s.section))];
  const t = result.timings_ms || {};
  const engine = result.mode === "llm" ? esc(result.model) : "the rule-based parser";
  $("#summary").innerHTML = specs.length
    ? `<strong>${specs.length} ${specs.length === 1 ? "value" : "values"}</strong> from ${sections.length} ${sections.length === 1 ? "section" : "sections"} of the manual, extracted by ${engine} in ${((t.retrieve || 0) + (t.extract || 0)).toLocaleString()} ms`
    : `No values found by ${engine}`;

  $("#dl-json").disabled = $("#dl-csv").disabled = !specs.length;

  // Spec sheet, grouped by manual section.
  const sheet = $("#sheet");
  if (!specs.length) {
    sheet.innerHTML = `
      <div class="note">
        <p>${esc(result.note || "The retrieved sections do not state this specification.")}</p>
        <p>This manual covers suspension (204), driveline (205) and brakes (206) only. Check the excerpts below to see what was searched, or rephrase with the part name the manual uses.</p>
      </div>`;
  } else {
    const groups = new Map();
    for (const s of specs) {
      if (!groups.has(s.section)) groups.set(s.section, []);
      groups.get(s.section).push(s);
    }
    sheet.innerHTML = [...groups.entries()].map(([section, items]) => {
      const first = items[0];
      const procs = [...new Set(items.map((s) => s.procedure))].join(", ");
      const pages = [...new Set(items.map((s) => s.pages))].join(", ");
      return `
        <section class="group" aria-label="Section ${esc(section)}">
          <div class="group-head">
            <span class="group-id">${esc(section)}</span>
            <span class="group-title">${esc(first.section_title)}</span>
            <span class="group-where">${esc(procs)}, page ${esc(pages)}</span>
          </div>
          ${items.map(renderSpec).join("")}
        </section>`;
    }).join("") + (result.note ? `<div class="note"><p>${esc(result.note)}</p></div>` : "");
  }

  $("#json-view").textContent = JSON.stringify(assignmentJson(result), null, 2);
  setView(state.view);

  // Rejected — shown, not hidden: it is the evidence that the check works.
  const rej = result.rejected || [];
  $("#rejected").hidden = !rej.length;
  if (rej.length) {
    $("#rejected summary").innerHTML = `${rej.length} ${rej.length === 1 ? "answer was" : "answers were"} discarded <small>the value is not in the excerpts the model was given</small>`;
    $("#rejected-list").innerHTML = rej.map((r) => `
      <li>${esc(r.component || r.raw || "Unnamed")}: ${esc(r.value ?? "")} ${esc(r.unit ?? "")}
        <span class="why">${esc(r.reason || "")}</span></li>`).join("");
  }

  // Sources.
  const values = specs.map((s) => s.value);
  const src = result.sources || [];
  $("#sources summary").innerHTML = `Manual excerpts searched (${src.length}) <small>what the extractor read</small>`;
  $("#source-list").innerHTML = src.map((h) => {
    const ranks = h.dense_rank || h.bm25_rank
      ? `semantic #${h.dense_rank ?? "–"}, keyword #${h.bm25_rank ?? "–"}`
      : "added because a table row says “Refer to the procedure”";
    return `
      <li>
        <div class="source-head">
          <strong>${esc(h.section)} ${esc(h.section_title)}</strong>
          <span>${esc(h.procedure)}${h.heading && h.heading !== h.procedure ? `, ${esc(h.heading)}` : ""}</span>
          <span>page ${esc(h.pages)}</span>
          <span class="ranks">${ranks}</span>
        </div>
        <pre class="source-text">${markValues(h.text, values)}</pre>
      </li>`;
  }).join("");
}

function renderSpec(s) {
  const alt = s.alt_value ? `<span class="alt">${esc(s.alt_value)} ${esc(s.alt_unit || "")}</span>` : "";
  const values = [s.value, s.alt_value].filter(Boolean);
  return `
    <article class="spec">
      <div>
        <div class="spec-name">${esc(s.component)}</div>
        <div class="spec-meta">
          <span>${esc(s.spec_type)}</span>
          ${s.condition ? `<span class="condition">${esc(s.condition)}</span>` : ""}
        </div>
      </div>
      <div class="reading" aria-label="${esc(s.value)} ${esc(s.unit)}">
        <span class="value">${esc(s.value)}</span><span class="unit">${esc(s.unit)}</span>
        ${alt}
      </div>
      <p class="evidence"><span class="src">Found on page ${esc(s.pages)}</span>${markValues(s.evidence, values)}</p>
    </article>`;
}

function setView(view) {
  state.view = view;
  $$(".view-toggle button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
  $("#sheet").hidden = view !== "sheet";
  $("#json-view").hidden = view !== "json";
}

// ---------------------------------------------------------------- ask
async function ask(query) {
  const form = $("#ask-form");
  const body = {
    query,
    mode: form.mode.value,
    system: $("#system").value || null,
    top_k: Number($("#topk").value),
  };
  const btn = $("#submit");
  btn.disabled = true;
  setStatus(`<span class="working">Searching ${state.meta ? state.meta.pages.toLocaleString() + " pages" : "the manual"}</span>`);
  const url = new URL(location.href);
  url.searchParams.set("q", query);
  history.replaceState(null, "", url);
  try {
    const result = await api("/api/extract", body);
    setStatus("");
    renderResult(result);
  } catch (err) {
    setStatus(esc(err.message === "Failed to fetch"
      ? `Can't reach the API at ${API || location.origin}. If this is the Vercel site, check SPECRAG_API_BASE in config.js and that the API is running.`
      : err.message), "error");
  } finally {
    btn.disabled = false;
  }
}

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#q").value.trim();
  if (q.length >= 2) ask(q);
});
$$(".example").forEach((b) => b.addEventListener("click", () => {
  $("#q").value = b.textContent;
  ask(b.textContent);
}));
$$(".view-toggle button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
$("#dl-json").addEventListener("click", () => {
  const r = state.result;
  download(`${slug(r.query)}.json`, JSON.stringify(assignmentJson(r), null, 2), "application/json");
});
$("#dl-csv").addEventListener("click", () => {
  const r = state.result;
  download(`${slug(r.query)}.csv`, toCsv(r), "text/csv");
});

// ---------------------------------------------------------------- tabs
function selectTab(id) {
  $$(".tabs [role=tab]").forEach((t) => {
    const on = t.id === id;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $("#" + t.getAttribute("aria-controls")).hidden = !on;
  });
  if (id === "tab-browse" && !$("#b-rows").children.length) loadBrowse();
}
$$(".tabs [role=tab]").forEach((t) => {
  t.addEventListener("click", () => selectTab(t.id));
  t.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const tabs = $$(".tabs [role=tab]");
    const next = tabs[(tabs.indexOf(t) + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length];
    selectTab(next.id);
    next.focus();
  });
});

// ---------------------------------------------------------------- browse
let browseTimer;
async function loadBrowse() {
  const params = new URLSearchParams();
  if ($("#b-q").value.trim()) params.set("q", $("#b-q").value.trim());
  if ($("#b-system").value) params.set("system", $("#b-system").value);
  if ($("#b-type").value) params.set("spec_type", $("#b-type").value);
  try {
    const data = await api(`/api/specs?${params}`);
    $("#b-count").innerHTML = `<strong>${data.count}</strong> ${data.count === 1 ? "row" : "rows"}${data.count > data.rows.length ? `, showing the first ${data.rows.length}` : ""}`;
    $("#b-rows").innerHTML = data.rows.map((r) => `
      <tr>
        <td>${esc(r.component)}</td>
        <td class="muted">${esc(r.spec_type)}</td>
        <td class="num val">${esc(r.value)} ${esc(r.unit)}</td>
        <td class="muted alt">${esc((r.alt_values || []).join(", "))}</td>
        <td>${esc(r.section_id)} ${esc(r.section_title)}</td>
        <td class="muted">${esc(r.doc_title)}${r.source === "inline" ? " (step)" : ""}</td>
        <td class="num">${esc(r.page)}</td>
      </tr>`).join("");
  } catch (err) {
    $("#b-count").textContent = err.message;
  }
}
["#b-system", "#b-type"].forEach((s) => $(s).addEventListener("change", loadBrowse));
$("#b-q").addEventListener("input", () => {
  clearTimeout(browseTimer);
  browseTimer = setTimeout(loadBrowse, 200);
});

// ---------------------------------------------------------------- boot
(async function boot() {
  try {
    const meta = await api("/api/meta");
    state.meta = meta;
    $("#foot-meta").textContent =
      `${meta.pages.toLocaleString()} pages, ${meta.chunks.toLocaleString()} indexed excerpts and ` +
      `${meta.spec_rows.toLocaleString()} parsed spec rows. Retrieval: ${meta.embed_model} + BM25. ` +
      (meta.llm_available ? `Extraction: ${meta.llm_model}.` : "No LLM key on this server: rule-based extraction only.");
    if (!meta.llm_available) {
      const llm = $("input[name=mode][value=llm]");
      llm.disabled = true;
      $("input[name=mode][value=rules]").checked = true;
    }
    const types = [...new Set((await api("/api/specs?limit=2000")).rows.map((r) => r.spec_type))].sort();
    $("#b-type").insertAdjacentHTML("beforeend", types.map((t) => `<option>${esc(t)}</option>`).join(""));
  } catch (err) {
    setStatus(esc(`Can't reach the API at ${API || location.origin}: ${err.message}`), "error");
  }
  const q = new URLSearchParams(location.search).get("q");
  if (q) {
    $("#q").value = q;
    ask(q);
  }
})();
