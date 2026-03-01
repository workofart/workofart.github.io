(() => {
  const root = document.querySelector("[data-tags-page]");
  if (!root) return;

  const input = root.querySelector(".tagsSearch");
  const meta = root.querySelector(".tagsMeta");
  const hint = root.querySelector("[data-tags-hint]");
  const results = root.querySelector("[data-tags-results]");
  const clear = root.querySelector("[data-tags-clear]");
  const chips = Array.from(root.querySelectorAll(".tagChip"));
  const groups = Array.from(root.querySelectorAll(".tagGroup"));

  const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  let activeId = null;

  function setMeta(visible, total, query) {
    if (!meta) return;
    if (query) meta.textContent = `${visible} / ${total} topics`;
    else meta.textContent = `${total} topics`;
  }

  function setActiveChip(id) {
    for (const chip of chips) {
      const chipId = (chip.getAttribute("href") || "").slice(1);
      chip.classList.toggle("isActive", Boolean(id) && chipId === id);
    }
  }

  function closeAllExcept(exceptId) {
    for (const group of groups) {
      if (group.id !== exceptId) group.open = false;
    }
  }

  function showSelection(id) {
    const group = root.querySelector(`#${CSS.escape(id)}`);
    if (!group) return;

    activeId = id;

    if (results) results.hidden = false;
    if (hint) hint.hidden = true;

    for (const other of groups) {
      other.hidden = other.id !== id;
      if (other.id !== id) other.open = false;
    }

    group.hidden = false;
    group.open = true;
    setActiveChip(id);

    group.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block: "start" });
  }

  function clearSelection({ keepHash = false } = {}) {
    activeId = null;
    closeAllExcept("");
    setActiveChip(null);

    if (!keepHash) {
      const url = window.location.pathname + window.location.search;
      history.replaceState(null, "", url);
    }

    for (const group of groups) group.hidden = true;

    if (results) results.hidden = true;
    if (hint) hint.hidden = false;
  }

  function openFromHash() {
    const hash = window.location.hash || "";
    if (!hash.startsWith("#tag-")) return;
    const id = hash.slice(1);
    showSelection(id);
  }

  function applyFilter(raw) {
    const query = (raw || "").trim().toLowerCase();
    const total = chips.length;
    let visible = 0;

    for (const chip of chips) {
      const tag = chip.dataset.tag || "";
      const ok = !query || tag.includes(query);
      chip.hidden = !ok;
      if (ok) visible += 1;
    }

    if (activeId) {
      const active = root.querySelector(`#${CSS.escape(activeId)}`);
      const activeTag = active?.dataset?.tag || "";
      if (query && !activeTag.includes(query)) clearSelection();
    }

    setMeta(visible, total, query);
  }

  if (input) {
    applyFilter("");
    input.addEventListener("input", () => applyFilter(input.value));
  } else {
    setMeta(chips.length, chips.length, "");
  }

  clearSelection({ keepHash: true });

  for (const chip of chips) {
    chip.addEventListener("click", (event) => {
      const href = chip.getAttribute("href") || "";
      if (!href.startsWith("#")) return;

      event.preventDefault();

      const id = href.slice(1);
      history.replaceState(null, "", href);
      showSelection(id);
    });
  }

  for (const group of groups) {
    group.addEventListener("toggle", () => {
      if (activeId && group.id === activeId && !group.open) clearSelection();
    });
  }

  if (clear) clear.addEventListener("click", () => clearSelection());

  window.addEventListener("hashchange", openFromHash);
  openFromHash();
})();
