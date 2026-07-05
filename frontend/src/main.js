import "./style.css";

const STORAGE_KEY = "theme";

function applyTheme(theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

function currentTheme() {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

function setupThemeToggle() {
  const toggle = document.getElementById("theme-toggle");
  if (!toggle) return;

  toggle.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem(STORAGE_KEY, next);
  });
}

function highlightActiveNavLink() {
  const path = window.location.pathname;
  document.querySelectorAll(".site-nav a[data-nav]").forEach((link) => {
    const isBlogPost = path.startsWith("/blogs/") && link.dataset.nav === "blogs";
    const isExactMatch = link.getAttribute("href") === path;
    const isHome = path === "/" && link.getAttribute("href") === "/index.html";
    if (isExactMatch || isBlogPost || isHome) {
      link.setAttribute("aria-current", "page");
    }
  });
}

setupThemeToggle();
highlightActiveNavLink();
