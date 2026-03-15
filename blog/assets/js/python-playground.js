const WORKSPACE_ROOT = "/home/pyodide/workspace";
const scriptLoaders = new Map();

function loadScriptOnce(url) {
  if (!scriptLoaders.has(url)) {
    scriptLoaders.set(
      url,
      new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = url;
        script.async = true;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load script: ${url}`));
        document.head.append(script);
      })
    );
  }

  return scriptLoaders.get(url);
}

function dirname(path) {
  const parts = path.split("/");
  parts.pop();
  return parts.join("/") || "/";
}

function isExistingDirectoryError(error) {
  if (!error || typeof error !== "object") {
    return false;
  }

  const message = typeof error.message === "string" ? error.message : "";
  const code = typeof error.code === "string" ? error.code : "";
  const errno = typeof error.errno === "number" ? error.errno : null;

  return (
    code === "EEXIST" ||
    errno === 17 ||
    message.includes("File exists") ||
    String(error).includes("File exists")
  );
}

export function ensureFsPath(fs, path) {
  if (typeof fs.mkdirTree === "function") {
    fs.mkdirTree(path);
    return;
  }

  const parts = path.split("/").filter(Boolean);
  let current = "";

  for (const part of parts) {
    current += `/${part}`;
    try {
      fs.mkdir(current);
    } catch (error) {
      if (!isExistingDirectoryError(error)) {
        throw error;
      }
    }
  }
}

export function formatPlaygroundError(error) {
  if (error instanceof Error && error.message) {
    return error.message;
  }

  if (error && typeof error === "object") {
    const details = [];

    if (typeof error.name === "string" && error.name) {
      details.push(error.name);
    }
    if (typeof error.code === "string" && error.code) {
      details.push(`code=${error.code}`);
    }
    if (typeof error.errno === "number") {
      details.push(`errno=${error.errno}`);
    }
    if (typeof error.message === "string" && error.message) {
      details.push(error.message);
    }

    if (details.length > 0) {
      return details.join(" | ");
    }

    try {
      return JSON.stringify(error);
    } catch {
      return String(error);
    }
  }

  return String(error);
}

const jsonFetchers = new Map();
const textFetchers = new Map();
const assetVersion = new URL(import.meta.url).searchParams.get("v") || "";

function withAssetVersion(url) {
  if (!assetVersion) {
    return url;
  }

  const resolved = new URL(url, typeof window !== "undefined" ? window.location.href : "http://localhost");
  resolved.searchParams.set("v", assetVersion);
  return resolved.toString();
}

async function fetchWithCache(url, parser, cache) {
  if (!cache.has(url)) {
    cache.set(
      url,
      (async () => {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Request failed for ${url} with ${response.status}.`);
        }

        return parser(response);
      })().catch((error) => {
        cache.delete(url);
        throw error;
      })
    );
  }

  return cache.get(url);
}

async function fetchText(url) {
  return fetchWithCache(withAssetVersion(url), (response) => response.text(), textFetchers);
}

async function fetchJson(url) {
  return fetchWithCache(withAssetVersion(url), (response) => response.json(), jsonFetchers);
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

const PYTHON_KEYWORDS = new Set([
  "and",
  "as",
  "assert",
  "async",
  "await",
  "break",
  "class",
  "continue",
  "def",
  "del",
  "elif",
  "else",
  "except",
  "False",
  "finally",
  "for",
  "from",
  "global",
  "if",
  "import",
  "in",
  "is",
  "lambda",
  "None",
  "nonlocal",
  "not",
  "or",
  "pass",
  "raise",
  "return",
  "True",
  "try",
  "while",
  "with",
  "yield",
]);

const PYTHON_BUILTINS = new Set([
  "abs",
  "bool",
  "dict",
  "enumerate",
  "float",
  "int",
  "len",
  "list",
  "max",
  "min",
  "print",
  "range",
  "set",
  "str",
  "sum",
  "tuple",
  "zip",
]);

const PYTHON_TOKEN_PATTERN =
  /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|(?:[rRuUbBfF]{0,2})(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'))|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_][A-Za-z0-9_]*)\b/gm;

function highlightPython(code) {
  let highlighted = "";
  let lastIndex = 0;
  let match;

  PYTHON_TOKEN_PATTERN.lastIndex = 0;

  while ((match = PYTHON_TOKEN_PATTERN.exec(code)) !== null) {
    highlighted += escapeHtml(code.slice(lastIndex, match.index));
    lastIndex = PYTHON_TOKEN_PATTERN.lastIndex;

    if (match[1]) {
      highlighted += `<span class="tok-comment">${escapeHtml(match[1])}</span>`;
      continue;
    }

    if (match[2]) {
      highlighted += `<span class="tok-string">${escapeHtml(match[2])}</span>`;
      continue;
    }

    if (match[3]) {
      highlighted += `<span class="tok-number">${escapeHtml(match[3])}</span>`;
      continue;
    }

    const identifier = match[4];
    if (PYTHON_KEYWORDS.has(identifier)) {
      highlighted += `<span class="tok-keyword">${escapeHtml(identifier)}</span>`;
    } else if (PYTHON_BUILTINS.has(identifier)) {
      highlighted += `<span class="tok-builtin">${escapeHtml(identifier)}</span>`;
    } else {
      highlighted += escapeHtml(identifier);
    }
  }

  highlighted += escapeHtml(code.slice(lastIndex));
  return highlighted;
}

function normalizeInlineCode(text) {
  const trimmed = text.replace(/^\s*\n/, "").replace(/\n\s*$/, "");
  if (!trimmed.trim()) {
    return "";
  }

  const lines = trimmed.split("\n");
  const indents = lines
    .filter((line) => line.trim())
    .map((line) => line.match(/^[\t ]*/)[0].length);
  const commonIndent = indents.length ? Math.min(...indents) : 0;

  return lines.map((line) => line.slice(commonIndent)).join("\n");
}

async function readInlineCode(element) {
  let inlineCode = normalizeInlineCode(element.textContent || "");
  if (!inlineCode) {
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    inlineCode = normalizeInlineCode(element.textContent || "");
  }

  return inlineCode;
}

function stableList(value) {
  return Array.isArray(value) ? value.map((item) => `${item}`) : [];
}

function stableProjectFiles(files) {
  return Array.isArray(files)
    ? files.map((file) => ({
        path: file.path ?? "",
        url: file.url ?? "",
      }))
    : [];
}

function runtimeSignature(manifest) {
  return JSON.stringify({
    pyodideBaseUrl: manifest?.pyodideBaseUrl ?? "",
    pyodideVersion: manifest?.pyodideVersion ?? "",
    packages: stableList(manifest?.packages),
    runRequiresProjectSource: Boolean(manifest?.runRequiresProjectSource),
    projectSource: stableProjectFiles(manifest?.projectSource?.files),
  });
}

class SharedPlaygroundSession {
  constructor() {
    this.reset();
    this.clients = new Set();
  }

  subscribe(client) {
    this.clients.add(client);
    return () => {
      this.clients.delete(client);
    };
  }

  notify(event) {
    for (const client of this.clients) {
      if (typeof client.handleSharedSessionEvent === "function") {
        client.handleSharedSessionEvent(event);
      }
    }
  }

  reset() {
    this.runtimeConfigKey = null;
    this.runtimePromise = null;
    this.projectSourcePromise = null;
    this.projectSourceReady = false;
    this.pyodide = null;
    this.executionQueue = Promise.resolve();
  }

  assertCompatible(manifest) {
    const nextKey = runtimeSignature(manifest);
    if (!this.runtimeConfigKey) {
      this.runtimeConfigKey = nextKey;
      return;
    }

    if (this.runtimeConfigKey !== nextKey) {
      throw new Error(
        "All playgrounds on the same page must share one Python runtime configuration."
      );
    }
  }

  isRuntimeReady(manifest) {
    return this.pyodide !== null && (!manifest || this.runtimeConfigKey === runtimeSignature(manifest));
  }

  isProjectSourceReady(manifest) {
    if (!manifest?.runRequiresProjectSource) {
      return this.isRuntimeReady(manifest);
    }

    return this.projectSourcePromise !== null && !this.isProjectSourceLoading();
  }

  isProjectSourceLoading() {
    return this.projectSourcePromise !== null && this.pyodide !== null && !this.projectSourceReady;
  }

  async ensureRuntime(manifest) {
    this.assertCompatible(manifest);

    if (this.runtimePromise) {
      return this.runtimePromise;
    }

    this.runtimePromise = (async () => {
      const baseUrl = manifest?.pyodideBaseUrl;
      if (!baseUrl) {
        throw new Error("Manifest is missing pyodideBaseUrl.");
      }

      await loadScriptOnce(`${baseUrl}pyodide.js`);
      const pyodide = await globalThis.loadPyodide({ indexURL: baseUrl });

      pyodide.setStdout({ batched() {} });
      pyodide.setStderr({ batched() {} });

      if (manifest.packages?.length) {
        await pyodide.loadPackage(manifest.packages);
      }

      await pyodide.runPythonAsync(`
import os
import sys

workspace = "${WORKSPACE_ROOT}"
os.makedirs(workspace, exist_ok=True)
if workspace not in sys.path:
    sys.path.insert(0, workspace)
os.chdir(workspace)
      `);

      this.pyodide = pyodide;
      this.notify({ type: "runtime-ready" });
      return pyodide;
    })().catch((error) => {
      this.runtimePromise = null;
      this.runtimeConfigKey = null;
      this.pyodide = null;
      throw error;
    });

    return this.runtimePromise;
  }

  async ensureProjectSource(manifest) {
    this.assertCompatible(manifest);
    await this.ensureRuntime(manifest);

    if (!manifest?.projectSource?.files?.length) {
      this.projectSourceReady = true;
      return;
    }

    if (this.projectSourcePromise) {
      return this.projectSourcePromise;
    }

    this.projectSourceReady = false;
    this.projectSourcePromise = (async () => {
      const files = await Promise.all(
        manifest.projectSource.files.map(async (file) => ({
          path: file.path,
          content: await fetchText(file.url),
        }))
      );

      ensureFsPath(this.pyodide.FS, WORKSPACE_ROOT);

      for (const file of files) {
        const targetPath = `${WORKSPACE_ROOT}/${file.path}`;
        ensureFsPath(this.pyodide.FS, dirname(targetPath));
        this.pyodide.FS.writeFile(targetPath, file.content, { encoding: "utf8" });
      }

      await this.pyodide.runPythonAsync(`
import importlib
import sys

importlib.invalidate_caches()
for module_name in list(sys.modules):
    if module_name == "autograd" or module_name.startswith("autograd."):
        del sys.modules[module_name]
      `);

      this.projectSourceReady = true;
      this.notify({ type: "project-source-ready" });
    })().catch((error) => {
      this.projectSourcePromise = null;
      this.projectSourceReady = false;
      throw error;
    });

    return this.projectSourcePromise;
  }

  async runCode(manifest, code, handlers) {
    this.assertCompatible(manifest);
    await this.ensureRuntime(manifest);

    if (manifest.runRequiresProjectSource) {
      await this.ensureProjectSource(manifest);
    }

    const pyodide = this.pyodide;
    const run = async () => {
      pyodide.setStdout({
        batched: (message) => {
          handlers.stdout(`${message}\n`);
        },
      });
      pyodide.setStderr({
        batched: (message) => {
          handlers.stderr(`${message}\n`);
        },
      });

      await pyodide.runPythonAsync(code);
    };

    const result = this.executionQueue.then(run);
    this.executionQueue = result.catch(() => {});
    return result;
  }
}

const sharedPlaygroundSession = new SharedPlaygroundSession();

export function resetSharedPlaygroundStateForTests() {
  scriptLoaders.clear();
  jsonFetchers.clear();
  textFetchers.clear();
  sharedPlaygroundSession.reset();
  sharedPlaygroundSession.clients.clear();
}

class PythonPlayground extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });

    this.manifest = null;
    this.codeTemplate = "";
    this.loadingDefinition = false;
    this.loadingRuntime = false;
    this.loadingSource = false;
    this.runningCode = false;
    this.definitionPromise = null;
    this.unsubscribeSharedSession = null;
    this.enclosingDetailsEl = null;
    this.handleDetailsToggle = null;
    this.ready = false;
  }

  connectedCallback() {
    if (this.ready) {
      return;
    }

    this.ready = true;
    this.render();
    this.bindDetailsToggle();
    this.unsubscribeSharedSession = sharedPlaygroundSession.subscribe(this);
    this.primeInlineCodeTemplate().catch(() => {});
    this.syncUi();
  }

  disconnectedCallback() {
    if (this.enclosingDetailsEl && this.handleDetailsToggle) {
      this.enclosingDetailsEl.removeEventListener("toggle", this.handleDetailsToggle);
    }
    this.enclosingDetailsEl = null;
    this.handleDetailsToggle = null;
    this.unsubscribeSharedSession?.();
    this.unsubscribeSharedSession = null;
  }

  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          margin: 1.5rem 0;
          color: #1f2328;
        }

        *, *::before, *::after {
          box-sizing: border-box;
        }

        .playground {
          position: relative;
          border: 1px solid rgba(0, 0, 0, 0.1);
          border-left: 0.4375rem solid #444;
          border-radius: 0.25rem;
          overflow: hidden;
          background: #fff;
        }

        .toolbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 0.5rem;
          padding: 0.35rem 0.55rem 0.3rem;
          background: #fff;
        }

        .editorHint {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          min-height: 2rem;
          font: 700 0.64rem/1 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
            "Liberation Mono", "Courier New", monospace;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: rgba(0, 0, 0, 0.42);
          user-select: none;
        }

        .editorHint svg {
          width: 0.78rem;
          height: 0.78rem;
          opacity: 0.7;
        }

        button {
          appearance: none;
          border: 1px solid rgba(0, 0, 0, 0.14);
          border-radius: 0.35rem;
          padding: 0;
          background: #fff;
          color: #202428;
          cursor: pointer;
          transition:
            border-color 160ms ease,
            background 160ms ease,
            color 160ms ease,
            box-shadow 160ms ease;
        }

        button:hover:not(:disabled) {
          border-color: rgba(0, 0, 0, 0.26);
          box-shadow: 0 1px 0 rgba(0, 0, 0, 0.05);
        }

        button:focus-visible,
        .editorTextarea:focus-visible {
          outline: 2px solid rgba(0, 0, 0, 0.22);
          outline-offset: 2px;
        }

        button:disabled {
          cursor: not-allowed;
          opacity: 0.52;
          box-shadow: none;
        }

        .iconButton {
          width: 1.9rem;
          height: 1.9rem;
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }

        .iconButton svg {
          width: 0.95rem;
          height: 0.95rem;
        }

        .playButton {
          background: rgba(0, 0, 0, 0.02);
          border-color: rgba(0, 0, 0, 0.12);
          color: rgba(0, 0, 0, 0.72);
        }

        .playButton:hover:not(:disabled) {
          background: rgba(0, 0, 0, 0.05);
          border-color: rgba(0, 0, 0, 0.22);
        }

        .panel {
          background: #fff;
        }

        .panel[hidden] {
          display: none;
        }

        .paneLabel {
          display: block;
          padding: 0.48rem 0.625rem 0.44rem;
          border-top: 1px solid rgba(0, 0, 0, 0.08);
          font: 700 0.74rem/1.1 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
            "Liberation Mono", "Courier New", monospace;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: rgba(0, 0, 0, 0.72);
          background: #f4f4f4;
        }

        .panel > .paneLabel:first-child {
          border-top: 0;
        }

        .editorSurface {
          position: relative;
          min-height: 0;
          background-image: linear-gradient(
            rgba(0, 0, 0, 0.03),
            rgba(0, 0, 0, 0.03) 1.5em,
            rgba(0, 0, 0, 0.02) 1.5em,
            rgba(0, 0, 0, 0.02) 3em
          );
          background-size: auto 3em;
          background-position-y: 0.625rem;
          background-color: #fff;
        }

        .editorTextarea,
        .highlightLayer,
        #output {
          width: 100%;
          margin: 0;
          border: 0;
          border-radius: 0;
          font: 0.875rem/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }

        .editorTextarea,
        .highlightLayer {
          min-height: 0;
          padding: 0.625rem;
          tab-size: 4;
          white-space: pre;
        }

        .highlightLayer {
          position: absolute;
          inset: 0;
          overflow: hidden;
          pointer-events: none;
          color: #222;
        }

        .editorTextarea {
          position: relative;
          display: block;
          resize: none;
          overflow: hidden;
          outline: none;
          background: transparent;
          color: transparent;
          caret-color: #222;
          -webkit-text-fill-color: transparent;
        }

        .editorTextarea::selection {
          background: rgba(193, 45, 79, 0.18);
        }

        #highlighted-code {
          display: block;
          min-height: 0;
        }

        .tok-keyword {
          color: #9f1239;
          font-weight: 600;
        }

        .tok-builtin {
          color: #0f766e;
        }

        .tok-string {
          color: #b45309;
        }

        .tok-number {
          color: #1d4ed8;
        }

        .tok-comment {
          color: #6b7280;
          font-style: italic;
        }

        #output {
          min-height: 3rem;
          max-height: 18rem;
          padding: 0.625rem;
          overflow: auto;
          color: #222;
          white-space: pre;
          background-image: linear-gradient(
            rgba(0, 0, 0, 0.03),
            rgba(0, 0, 0, 0.03) 1.5em,
            rgba(0, 0, 0, 0.02) 1.5em,
            rgba(0, 0, 0, 0.02) 3em
          );
          background-size: auto 3em;
          background-position-y: 0.625rem;
          background-color: #fff;
        }

        pre[hidden] {
          display: none;
        }

        .outputSection[hidden] {
          display: none;
        }

        .statusText {
          margin: 0;
          padding: 0.45rem 0.625rem 0.55rem;
          font: 0.78rem/1.4 "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
          color: rgba(0, 0, 0, 0.62);
          white-space: pre-wrap;
          background: #fff;
        }

        .statusText.error {
          color: #8b1e1e;
        }

        .statusText[hidden] {
          display: none;
        }

        @media (max-width: 780px) {
          .editorSurface,
          .editorTextarea,
          .highlightLayer {
            min-height: 0;
          }

          #highlighted-code {
            min-height: 0;
          }
        }
      </style>

      <section class="playground">
          <div class="toolbar">
            <span class="editorHint" aria-hidden="true">
              <svg viewBox="0 0 16 16" focusable="false">
                <path
                  d="M11.96 2.46a1.5 1.5 0 0 1 2.12 2.12L6.5 12.16l-2.83.7.7-2.83 7.59-7.57Zm.7.7-7.2 7.19-.34 1.38 1.37-.34 7.18-7.2a.5.5 0 1 0-.71-.7Z"
                  fill="currentColor"
                />
              </svg>
              <span>Editable</span>
            </span>
            <button
              class="iconButton playButton"
              id="start-playground"
              type="button"
              aria-label="Run code"
              title="Run code"
            >
              <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                <path d="M5 3.75v8.5L12 8 5 3.75Z" fill="currentColor" />
              </svg>
            </button>
          </div>

          <div class="panel" id="panel">
            <div class="editorSurface">
              <pre class="highlightLayer" aria-hidden="true"><code id="highlighted-code"></code></pre>
              <textarea class="editorTextarea" id="editor" spellcheck="false" wrap="off"></textarea>
            </div>
            <div class="outputSection" id="output-section" hidden>
              <span class="paneLabel">Output</span>
              <pre id="output" aria-live="polite" hidden></pre>
              <p class="statusText" id="status-text" aria-live="polite" hidden></p>
            </div>
          </div>
      </section>
    `;

    this.playgroundEl = this.shadowRoot.querySelector(".playground");
    this.startPlaygroundButton = this.shadowRoot.querySelector("#start-playground");
    this.panelEl = this.shadowRoot.querySelector("#panel");
    this.editorSurfaceEl = this.shadowRoot.querySelector(".editorSurface");
    this.editorEl = this.shadowRoot.querySelector("#editor");
    this.highlightLayerEl = this.shadowRoot.querySelector(".highlightLayer");
    this.highlightCodeEl = this.shadowRoot.querySelector("#highlighted-code");
    this.outputSectionEl = this.shadowRoot.querySelector("#output-section");
    this.outputEl = this.shadowRoot.querySelector("#output");
    this.statusTextEl = this.shadowRoot.querySelector("#status-text");

    this.startPlaygroundButton.addEventListener("click", () => {
      this.runCode().catch((error) => {
        this.setStatus(`Run failed.\n${this.formatError(error)}`, "error");
      });
    });

    this.editorEl.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || !event.metaKey) {
        return;
      }

      event.preventDefault();
      this.runCode().catch((error) => {
        this.setStatus(`Run failed.\n${this.formatError(error)}`, "error");
      });
    });

    this.editorEl.addEventListener("input", () => {
      this.refreshHighlight();
      this.syncEditorHeight();
    });

    this.editorEl.addEventListener("scroll", () => {
      this.syncHighlightScroll();
    });
  }

  bindDetailsToggle() {
    this.enclosingDetailsEl = this.closest("details");
    if (!this.enclosingDetailsEl) {
      return;
    }

    this.handleDetailsToggle = () => {
      if (this.enclosingDetailsEl?.open) {
        this.syncEditorHeight();
        this.syncHighlightScroll();
      }
    };

    this.enclosingDetailsEl.addEventListener("toggle", this.handleDetailsToggle);
  }

  async primeInlineCodeTemplate() {
    const inlineCode = await readInlineCode(this);
    if (!inlineCode) {
      return;
    }

    this.codeTemplate = inlineCode;
    if (!this.editorEl.value) {
      this.setEditorValue(inlineCode);
    }
    this.syncUi();
  }

  async ensureDefinitionLoaded() {
    if (this.definitionPromise) {
      return this.definitionPromise;
    }

    const manifestUrl = this.dataset.manifestUrl;

    if (!manifestUrl) {
      throw new Error("data-manifest-url is required.");
    }

    this.loadingDefinition = true;
    this.setStatus("Loading the playground example.", "info");

    this.definitionPromise = Promise.all([fetchJson(manifestUrl), this.loadCodeTemplate()])
      .then(([manifest, codeTemplate]) => {
        this.manifest = manifest;
        this.codeTemplate = codeTemplate;
        if (!this.editorEl.value) {
          this.setEditorValue(codeTemplate);
        }
        this.setStatus("", "info");
        this.syncUi();
      })
      .catch((error) => {
        this.definitionPromise = null;
        throw error;
      })
      .finally(() => {
        this.loadingDefinition = false;
        this.syncUi();
      });

    return this.definitionPromise;
  }

  async loadCodeTemplate() {
    const inlineCode = await readInlineCode(this);
    if (inlineCode) {
      return inlineCode;
    }

    throw new Error("Provide playground code inline within the element content.");
  }

  runtimeReady() {
    return Boolean(this.manifest) && sharedPlaygroundSession.isRuntimeReady(this.manifest);
  }

  projectSourceReady() {
    return Boolean(this.manifest) && sharedPlaygroundSession.isProjectSourceReady(this.manifest);
  }

  syncUi() {
    const sourceRequired = Boolean(this.manifest?.runRequiresProjectSource);
    const runtimeReady = this.runtimeReady();
    const sourceReady = !sourceRequired || this.projectSourceReady();
    const busy = this.loadingDefinition || this.loadingRuntime || this.loadingSource || this.runningCode;

    this.startPlaygroundButton.disabled = busy;
    this.startPlaygroundButton.classList.toggle("isReady", runtimeReady && sourceReady);
    this.panelEl.hidden = false;
    this.editorEl.disabled = busy;
    this.outputEl.hidden = this.outputEl.textContent.length === 0;
    this.statusTextEl.hidden = this.statusTextEl.textContent.length === 0;
    this.outputSectionEl.hidden =
      this.outputEl.textContent.length === 0 && this.statusTextEl.textContent.length === 0;
  }

  setStatus(message, tone = "info") {
    this.statusTextEl.textContent = message;
    this.statusTextEl.classList.toggle("error", tone === "error");
    this.syncUi();
  }

  appendOutput(text) {
    this.outputEl.textContent += text;
    this.outputEl.hidden = false;
    this.syncUi();
  }

  clearOutput() {
    this.outputEl.textContent = "";
    this.outputEl.hidden = true;
    this.syncUi();
  }

  setEditorValue(code) {
    if (!this.editorEl) {
      return;
    }

    this.editorEl.value = code;
    this.refreshHighlight();
    this.syncEditorHeight();
    this.syncHighlightScroll();
  }

  refreshHighlight() {
    if (!this.highlightCodeEl || !this.editorEl) {
      return;
    }

    const code = this.editorEl.value || "";
    this.highlightCodeEl.innerHTML = highlightPython(code) || " ";
  }

  syncEditorHeight() {
    if (!this.editorEl || !this.editorSurfaceEl || !this.highlightLayerEl || !this.highlightCodeEl) {
      return;
    }

    const previousHeight = this.editorEl.style.height;
    this.editorEl.style.height = "0px";
    const nextHeight = this.editorEl.scrollHeight;

    if (nextHeight > 0) {
      const nextHeightPx = `${nextHeight}px`;
      this.editorEl.style.height = nextHeightPx;
      this.editorSurfaceEl.style.height = nextHeightPx;
      this.highlightLayerEl.style.height = nextHeightPx;
      this.highlightCodeEl.style.minHeight = nextHeightPx;
      return;
    }

    this.editorEl.style.height = previousHeight;
  }

  syncHighlightScroll() {
    if (!this.highlightLayerEl || !this.editorEl) {
      return;
    }

    this.highlightLayerEl.scrollTop = this.editorEl.scrollTop;
    this.highlightLayerEl.scrollLeft = this.editorEl.scrollLeft;
  }

  handleSharedSessionEvent(event) {
    if (!this.manifest) {
      return;
    }

    this.syncUi();
  }

  async startPlayground() {
    await this.ensureDefinitionLoaded();

    if (!this.runtimeReady()) {
      this.loadingRuntime = true;
      this.setStatus("Loading the shared Python runtime for this page.", "info");
      this.syncUi();
      try {
        await sharedPlaygroundSession.ensureRuntime(this.manifest);
        this.clearOutput();
        this.appendOutput("Python runtime loaded from local assets.\n");
      } finally {
        this.loadingRuntime = false;
        this.syncUi();
      }
    }

    if (this.manifest.projectSource?.files?.length && !this.projectSourceReady()) {
      this.loadingSource = true;
      this.setStatus("Loading the bundled ml-by-hand snapshot into the shared session.", "info");
      this.syncUi();
      try {
        await sharedPlaygroundSession.ensureProjectSource(this.manifest);
        this.appendOutput("Loaded local project snapshot.\n");
      } finally {
        this.loadingSource = false;
        this.syncUi();
      }
    }

    this.setStatus("", "info");
  }

  async runCode() {
    if (this.runningCode) {
      return;
    }

    await this.startPlayground();

    this.runningCode = true;
    this.clearOutput();
    this.setStatus("Running Python in the browser.", "info");

    try {
      await sharedPlaygroundSession.runCode(this.manifest, this.editorEl.value, {
        stdout: (text) => this.appendOutput(text),
        stderr: (text) => this.appendOutput(text),
      });
      this.setStatus("", "info");
    } finally {
      this.runningCode = false;
      this.syncUi();
    }
  }

  formatError(error) {
    return formatPlaygroundError(error);
  }
}

if (typeof globalThis.customElements !== "undefined" && !customElements.get("python-playground")) {
  customElements.define("python-playground", PythonPlayground);
}
