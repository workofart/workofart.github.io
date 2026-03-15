let mermaidInitialized = false;
const siteSansFont =
  '"Source Sans 3", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
const mermaidConfig = {
  startOnLoad: false,
  theme: "base",
  fontFamily: siteSansFont,
  themeVariables: {
    fontFamily: siteSansFont,
    fontSize: "18px",
  },
  flowchart: {
    nodeSpacing: 6,
    rankSpacing: 10,
    diagramPadding: 2,
  },
};

export async function enhanceMermaidDiagrams({
  mermaidApi = globalThis.mermaid,
  root = document,
} = {}) {
  if (!root?.querySelectorAll || !mermaidApi) {
    return 0;
  }

  const codeBlocks = [...root.querySelectorAll("pre > code.language-mermaid")];

  if (codeBlocks.length === 0) {
    return 0;
  }

  const documentRef = root.nodeType === 9 ? root : root.ownerDocument ?? document;

  for (const codeBlock of codeBlocks) {
    const sourcePre = codeBlock.parentElement;

    if (!sourcePre) {
      continue;
    }

    const mermaidBlock = documentRef.createElement("pre");
    mermaidBlock.className = "mermaid";
    mermaidBlock.textContent = codeBlock.textContent.trim();
    sourcePre.replaceWith(mermaidBlock);
  }

  if (!mermaidInitialized) {
    mermaidApi.initialize(mermaidConfig);
    mermaidInitialized = true;
  }

  await mermaidApi.run({ querySelector: "pre.mermaid" });
  return codeBlocks.length;
}

function bootMermaidDiagrams() {
  void enhanceMermaidDiagrams();
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootMermaidDiagrams, { once: true });
  } else {
    bootMermaidDiagrams();
  }
}
