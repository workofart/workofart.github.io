const HEADING_SELECTOR = "h2[id], h3[id]";
const ACTIVE_OFFSET_PX = 140;

function cleanHeadingText(text) {
  return text.replace(/\s+/g, " ").trim();
}

function getStickyNavOffset(root = document) {
  const nav = root.querySelector(".navBar");

  if (!nav) {
    return 0;
  }

  return nav.getBoundingClientRect().height;
}

function scrollToAnchorTarget(target, { behavior = "smooth" } = {}) {
  if (!target) {
    return;
  }

  const top = Math.max(
    0,
    window.scrollY + target.getBoundingClientRect().top - getStickyNavOffset(target.ownerDocument)
  );

  window.scrollTo({
    top,
    behavior,
  });
}

function buildGeneratedTocList(article) {
  const headings = Array.from(article.querySelectorAll(HEADING_SELECTOR)).filter((heading) => {
    const text = cleanHeadingText(heading.textContent || "");
    return heading.id && text;
  });

  if (headings.length < 2) {
    return null;
  }

  const list = document.createElement("ul");
  list.className = "post-toc-list";

  let currentTopLevelItem = null;

  headings.forEach((heading) => {
    const item = document.createElement("li");
    const link = document.createElement("a");

    link.href = `#${heading.id}`;
    link.textContent = cleanHeadingText(heading.textContent || "");
    item.append(link);

    if (heading.tagName === "H3" && currentTopLevelItem) {
      let nestedList = currentTopLevelItem.querySelector("ul");

      if (!nestedList) {
        nestedList = document.createElement("ul");
        currentTopLevelItem.append(nestedList);
      }

      nestedList.append(item);
      return;
    }

    list.append(item);
    currentTopLevelItem = item;
  });

  return list;
}

function collectTocTargets(tocRoot) {
  return Array.from(tocRoot.querySelectorAll('a[href^="#"]'))
    .map((link) => {
      const id = decodeURIComponent(link.getAttribute("href").slice(1));

      if (!id) {
        return null;
      }

      return {
        id,
        link,
        target: document.getElementById(id),
      };
    })
    .filter((entry) => entry?.target);
}

function applyActiveState(entries) {
  if (!entries.length) {
    return;
  }

  let activeEntry = entries[0];

  entries.forEach((entry) => {
    if (entry.target.getBoundingClientRect().top - ACTIVE_OFFSET_PX <= 0) {
      activeEntry = entry;
    }
  });

  entries.forEach((entry) => {
    const isActive = entry === activeEntry;
    entry.link.classList.toggle("is-active", isActive);

    if (isActive) {
      entry.link.setAttribute("aria-current", "location");
    } else {
      entry.link.removeAttribute("aria-current");
    }
  });
}

function setupActiveTracking(tocRoot) {
  const entries = collectTocTargets(tocRoot);

  if (!entries.length) {
    return;
  }

  let rafId = null;
  const requestFrame = window.requestAnimationFrame || ((callback) => callback());

  const update = () => {
    rafId = null;
    applyActiveState(entries);
  };

  const scheduleUpdate = () => {
    if (rafId !== null) {
      return;
    }

    rafId = requestFrame(update);
  };

  window.addEventListener("scroll", scheduleUpdate, { passive: true });
  window.addEventListener("resize", scheduleUpdate);
  window.addEventListener("hashchange", scheduleUpdate);
  scheduleUpdate();
}

function setupAnchorNavigation(tocRoot) {
  const handleClick = (event) => {
    const link = event.target instanceof Element ? event.target.closest('a[href^="#"]') : null;

    if (
      !link ||
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    ) {
      return;
    }

    const hash = link.getAttribute("href");
    const id = decodeURIComponent(hash.slice(1));
    const target = document.getElementById(id);

    if (!target) {
      return;
    }

    event.preventDefault();
    scrollToAnchorTarget(target);

    if (window.history?.pushState) {
      window.history.pushState(null, "", hash);
    } else {
      window.location.hash = hash;
    }
  };

  tocRoot.addEventListener("click", handleClick);

  const alignCurrentHash = () => {
    if (!window.location.hash) {
      return;
    }

    const id = decodeURIComponent(window.location.hash.slice(1));
    const target = document.getElementById(id);

    if (!target) {
      return;
    }

    scrollToAnchorTarget(target, { behavior: "auto" });
  };

  window.addEventListener("hashchange", alignCurrentHash);
  const requestFrame = window.requestAnimationFrame || ((callback) => callback());
  requestFrame(alignCurrentHash);
}

function createTocChrome(tocList, index) {
  const heading = document.createElement("p");
  heading.className = "post-toc-title";
  heading.id = `post-toc-title-${index + 1}`;
  heading.textContent = "On this page";

  const nav = document.createElement("nav");
  nav.className = "post-toc-nav";
  nav.setAttribute("aria-labelledby", heading.id);
  nav.append(tocList);

  return { heading, nav };
}

export function enhancePostToc(root = document) {
  const postShells = Array.from(root.querySelectorAll(".post-shell"));

  postShells.forEach((postShell, index) => {
    if (postShell.classList.contains("post-shell--has-toc")) {
      return;
    }

    const article = postShell.querySelector(".blog-post");
    const tocShell = postShell.querySelector(".post-toc-shell");

    if (!article || !tocShell) {
      return;
    }

    let tocList = article.querySelector("#markdown-toc");

    if (!tocList) {
      tocList = buildGeneratedTocList(article);
    }

    if (!tocList) {
      return;
    }

    const { heading, nav } = createTocChrome(tocList, index);

    tocShell.replaceChildren(heading, nav);
    tocShell.hidden = false;
    postShell.classList.add("post-shell--has-toc");
    setupAnchorNavigation(tocShell);
    setupActiveTracking(tocShell);
  });
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => enhancePostToc(), { once: true });
  } else {
    enhancePostToc();
  }
}
