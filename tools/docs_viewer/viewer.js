const state = {
  docs: null,
  packages: [],
  entities: [],
  packageById: new Map(),
  entityById: new Map(),
  searchRows: [],
};

const els = {
  error: document.querySelector("#error"),
  packageList: document.querySelector("#package-list"),
  searchInput: document.querySelector("#search-input"),
  searchResults: document.querySelector("#search-results"),
  view: document.querySelector("#view"),
};

init();

async function init() {
  try {
    const response = await fetch("docs.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`docs.json returned HTTP ${response.status}`);
    }

    const docs = await response.json();
    loadDocs(docs);
    renderPackageNav();
    renderRoute();
  } catch (error) {
    showError(`Could not load docs.json. ${error.message}`);
  }

  window.addEventListener("hashchange", renderRoute);
  els.searchInput.addEventListener("input", renderSearch);
}

function loadDocs(docs) {
  if (!docs || typeof docs !== "object") {
    throw new Error("Expected a JSON object.");
  }

  const packages = asArray(docs.packages);
  const entities = asArray(docs.entities);

  if (!packages.length && !entities.length) {
    throw new Error("Expected at least one package or entity.");
  }

  state.docs = docs;
  state.packages = packages.map((pkg, index) => normalizePackage(pkg, index));
  state.entities = entities.map((entity, index) => normalizeEntity(entity, index));
  state.packageById = new Map(state.packages.map((pkg) => [pkg.id, pkg]));
  state.entityById = new Map(state.entities.map((entity) => [entity.id, entity]));
  state.searchRows = buildSearchRows(docs.search);
}

function normalizePackage(pkg, index) {
  const id = stringValue(pkg.id || `package-${index}`);
  return {
    raw: pkg,
    id,
    name: stringValue(pkg.name || id),
    path: stringValue(pkg.path || ""),
    summary: firstSentence(pkg.docs),
    docs: stringValue(pkg.docs || ""),
    source: sourceInfo(pkg),
  };
}

function normalizeEntity(entity, index) {
  const id = stringValue(entity.id || `entity-${index}`);
  return {
    raw: entity,
    id,
    name: stringValue(entity.name || id),
    kind: stringValue(entity.kind || "entity"),
    packageId: stringValue(entity.packageId || ""),
    signature: stringValue(entity.signature || ""),
    summary: firstSentence(entity.docs),
    docs: stringValue(entity.docs || ""),
    source: sourceInfo(entity),
  };
}

function renderPackageNav() {
  clear(els.packageList);

  if (!state.packages.length) {
    appendText(els.packageList, "No packages");
    return;
  }

  for (const pkg of state.packages) {
    const link = document.createElement("a");
    link.href = packageHref(pkg.id);
    link.dataset.packageId = pkg.id;
    link.textContent = pkg.name;
    els.packageList.append(link);
  }
}

function renderRoute() {
  if (!state.docs) return;

  const route = parseHash();
  markActivePackage(route);
  renderSearch();
  clear(els.view);

  if (route.kind === "package") {
    renderPackage(route.id);
    return;
  }

  if (route.kind === "entity") {
    renderEntity(route.id);
    return;
  }

  renderHome();
}

function renderHome() {
  const firstPackage = state.packages[0];
  const schema = stringValue(state.docs.schemaVersion ?? "unknown");

  const header = pageHeader("Documentation", "Overview", `schemaVersion: ${schema}`);
  els.view.append(header);

  const counts = document.createElement("dl");
  counts.className = "meta-list section";
  addMeta(counts, "Packages", String(state.packages.length));
  addMeta(counts, "Entities", String(state.entities.length));
  els.view.append(counts);

  if (firstPackage) {
    const p = document.createElement("p");
    p.className = "section";
    const link = document.createElement("a");
    link.href = packageHref(firstPackage.id);
    link.textContent = `Open ${firstPackage.name}`;
    p.append(link);
    els.view.append(p);
  }
}

function renderPackage(id) {
  const pkg = state.packageById.get(id);
  if (!pkg) {
    renderMissing("Package", id);
    return;
  }

  els.view.append(pageHeader(pkg.name, "Package", pkg.summary || pkg.path));

  const meta = document.createElement("dl");
  meta.className = "meta-list section";
  addMeta(meta, "ID", pkg.id);
  if (pkg.path) addMeta(meta, "Path", pkg.path);
  addSourceMeta(meta, pkg.source);
  els.view.append(meta);

  appendDocs(pkg.docs);

  const entities = entitiesForPackage(pkg);
  appendEntityGroups(entities, "Entities");
}

function renderEntity(id) {
  const entity = state.entityById.get(id);
  if (!entity) {
    renderMissing("Entity", id);
    return;
  }

  els.view.append(pageHeader(entity.name, entity.kind, entity.summary));

  if (entity.signature) {
    const code = document.createElement("pre");
    code.className = "doc-block";
    code.textContent = entity.signature;
    els.view.append(section("Signature", code));
  }

  const meta = document.createElement("dl");
  meta.className = "meta-list section";
  addMeta(meta, "ID", entity.id);
  addMeta(meta, "Kind", entity.kind);
  if (entity.packageId) addLinkedPackageMeta(meta, entity.packageId);
  if (entity.group) addMeta(meta, "Group", entity.group);
  addSourceMeta(meta, entity.source);
  els.view.append(meta);

  appendDocs(entity.docs);
}

function appendEntityGroups(entities, title) {
  const wrapper = document.createElement("section");
  wrapper.className = "section";
  const h2 = document.createElement("h2");
  h2.textContent = title;
  wrapper.append(h2);

  if (!entities.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No entities.";
    wrapper.append(empty);
    els.view.append(wrapper);
    return;
  }

  const groups = groupEntities(entities);
  for (const [groupName, rows] of groups) {
    const group = document.createElement("div");
    group.className = "group";

    const h3 = document.createElement("h3");
    h3.textContent = groupName;
    group.append(h3);

    const list = document.createElement("div");
    list.className = "entity-list";
    for (const entity of rows) {
      list.append(entityCard(entity));
    }
    group.append(list);
    wrapper.append(group);
  }

  els.view.append(wrapper);
}

function renderSearch() {
  const query = els.searchInput.value.trim().toLowerCase();
  clear(els.searchResults);
  els.searchResults.hidden = !query;
  if (!query) return;

  const rows = state.searchRows
    .map((row) => ({ row, score: searchScore(row, query) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 25);

  const h2 = document.createElement("h2");
  h2.textContent = `Search results for "${els.searchInput.value.trim()}"`;
  els.searchResults.append(h2);

  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No matches.";
    els.searchResults.append(empty);
    return;
  }

  for (const { row } of rows) {
    const link = document.createElement("a");
    link.className = "result-card";
    link.href = row.href;

    const title = document.createElement("strong");
    title.textContent = row.title;
    link.append(title);

    const meta = document.createElement("span");
    meta.className = "kind";
    meta.textContent = row.type;
    link.append(meta);

    if (row.text) {
      const p = document.createElement("p");
      p.textContent = snippet(row.text, 160);
      link.append(p);
    }

    els.searchResults.append(link);
  }
}

function buildSearchRows(search) {
  const rows = [];

  for (const item of asArray(search)) {
    const type = stringValue(item.type || "result");
    const id = stringValue(item.id);
    const href = type === "package" ? packageHref(id) : entityHref(id);
    rows.push({
      type,
      title: stringValue(item.title || id),
      text: stringValue(item.text || ""),
      href,
    });
  }

  return rows;
}

function sourceInfo(item) {
  const position = item.position && typeof item.position === "object" ? item.position : {};
  const file = stringValue(position.fileId || item.path || "");
  const line = stringValue(position.line || "");
  const url = stringValue(item.sourceUrl || "");
  return { file, line, url };
}

function addSourceMeta(meta, source) {
  if (source.url) {
    const link = document.createElement("a");
    link.href = source.url;
    link.textContent = source.file ? sourceLabel(source) : source.url;
    addMetaNode(meta, "Source", link);
    return;
  }

  if (source.file) addMeta(meta, "Source", sourceLabel(source));
}

function addLinkedPackageMeta(meta, packageId) {
  const pkg = state.packageById.get(packageId);
  if (!pkg) {
    addMeta(meta, "Package", packageId);
    return;
  }

  const link = document.createElement("a");
  link.href = packageHref(pkg.id);
  link.textContent = pkg.name;
  addMetaNode(meta, "Package", link);
}

function entitiesForPackage(pkg) {
  return state.entities.filter((entity) => entity.packageId === pkg.id);
}

function groupEntities(entities) {
  const map = new Map();

  for (const entity of entities) {
    const key = entity.kind || "Entities";
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(entity);
  }

  return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

function entityCard(entity) {
  const link = document.createElement("a");
  link.className = "entity-card";
  link.href = entityHref(entity.id);

  const title = document.createElement("strong");
  title.textContent = entity.name;
  link.append(title);

  const kind = document.createElement("span");
  kind.className = "kind";
  kind.textContent = entity.kind;
  link.append(kind);

  const text = entity.summary || entity.signature;
  if (text) {
    const p = document.createElement("p");
    p.textContent = snippet(text, 180);
    link.append(p);
  }

  return link;
}

function appendDocs(text) {
  if (!text) return;
  const block = document.createElement("div");
  block.className = "doc-block";
  block.textContent = text;
  els.view.append(section("Docs", block));
}

function pageHeader(title, eyebrow, summary) {
  const header = document.createElement("header");

  const eye = document.createElement("div");
  eye.className = "eyebrow";
  eye.textContent = eyebrow;
  header.append(eye);

  const h1 = document.createElement("h1");
  h1.className = "title";
  h1.textContent = title;
  header.append(h1);

  if (summary) {
    const p = document.createElement("p");
    p.className = "summary";
    p.textContent = snippet(summary, 300);
    header.append(p);
  }

  return header;
}

function section(title, child) {
  const wrapper = document.createElement("section");
  wrapper.className = "section";
  const h2 = document.createElement("h2");
  h2.textContent = title;
  wrapper.append(h2, child);
  return wrapper;
}

function renderMissing(kind, id) {
  els.view.append(pageHeader(`${kind} not found`, "Missing", id));
}

function parseHash() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "package" && parts[1]) return { kind: "package", id: decode(parts[1]) };
  if (parts[0] === "entity" && parts[1]) return { kind: "entity", id: decode(parts[1]) };
  return { kind: "home", id: "" };
}

function markActivePackage(route) {
  for (const link of els.packageList.querySelectorAll("a")) {
    link.classList.toggle("active", route.kind === "package" && link.dataset.packageId === route.id);
  }
}

function addMeta(list, key, value) {
  const node = document.createTextNode(value);
  addMetaNode(list, key, node);
}

function addMetaNode(list, key, valueNode) {
  const dt = document.createElement("dt");
  dt.textContent = key;
  const dd = document.createElement("dd");
  dd.append(valueNode);
  list.append(dt, dd);
}

function showError(message) {
  els.error.hidden = false;
  els.error.textContent = message;
  clear(els.packageList);
  clear(els.view);
}

function searchScore(row, query) {
  const title = row.title.toLowerCase();
  const text = row.text.toLowerCase();
  let score = 0;
  if (title === query) score += 100;
  if (title.includes(query)) score += 50;
  if (text.includes(query)) score += 10;
  return score;
}

function packageHref(id) {
  return `#/package/${encodeURIComponent(id)}`;
}

function entityHref(id) {
  return `#/entity/${encodeURIComponent(id)}`;
}

function decode(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function sourceLabel(source) {
  return source.line ? `${source.file}:${source.line}` : source.file;
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function stringValue(value) {
  return value == null ? "" : String(value);
}

function snippet(text, max) {
  const clean = stringValue(text).replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}...` : clean;
}

function firstSentence(text) {
  const clean = stringValue(text).trim();
  const [first] = clean.split(/\n\s*\n|\.\s+/);
  return first || "";
}

function clear(node) {
  node.replaceChildren();
}

function appendText(node, text) {
  node.append(document.createTextNode(text));
}
