# Portfolio Project Blueprint & Standards

This file contains the core governance principles, engineering standards, and architectural specifications for building the personal portfolio website.

## 1. System Architecture & Tech Stack
- **Architecture:** Static Site Generation (SSG) compiled locally or via CI/CD pipelines.
- **Backend Core:** Python (Modular, decoupled static site builder using Jinja2 templates, PyYAML metadata front-matter headers, and Python-Markdown text processors. Strict adherence to PEP 8, PEP 20, and explicit type hinting).
- **Frontend Engine:** JavaScript (Simple, elegant, minimalist design bundled into static, optimized HTML/CSS/JS web assets).
- **Dependency Management:** Monitored and locked natively via `pixi`.
- **Version Control:** Absolute Git branch isolation. Merge to `main` ONLY via verified GitHub Pull Requests. Day-to-day work occurs on the `dev` branch.

## 2. Frontend Specifications & Core Layout
- **Core Layout:** Minimalistic design featuring 4 mandatory core layout views.
- **Global Navigation:** The following titles MUST be located at the top right corner of every page view:
  1. About
  2. CV/Resume
  3. Projects
  4. Blogs
- **Visual Design - Sun/Moon Toggle:** A highly visible theme toggle must be positioned on the page (prefer top-right adjacent to navigation). It must cleanly transition the portfolio theme between light and dark backgrounds using global CSS variables or native class overrides (`.dark`).
- **Blogs Page Specifications:** The blog landing page must present a clean list of written blog post titles. Clicking a title must dynamically route the user to or load the respective markdown-compiled static content seamlessly without breaking full-site navigation.
- **Social Integrations:** Must embed distinct functional icons linking directly to:
  - Email (using standard `mailto:` links)
  - GitHub Page
  - LinkedIn
  - Google Scholar

## 3. Workspace Dependency Architecture (`pixi.toml`)
The project environment utilizes a strict, multi-environment layout via Pixi to protect production footprints while maximizing testing efficiency.

- **`[dependencies]` (Base Production Layer):** Contains lightweight markdown processing utilities, data parsers, and template engines (`jinja2`, `markdown`, `pyyaml`).
- **`[feature.dev]` (Developer Engine):** Houses local interactive servers, automated code formatters (`ruff`), and native JavaScript environments (`nodejs`).
- **`[feature.test]` (Lean Test Layer):** Dedicated strictly to test execution engines (`pytest`). Must remain completely free of large frontend build tool configurations.
- **Editable Package Mounting:** The root Python package must be mounted declaratively using `{ path = ".", editable = true }` within the common features block to guarantee seamless continuous integration pathing across all environmental slices.

## 4. Operational Commands for Claude Code

Use these explicit target commands when executing workspace automation steps:

* **Environment Syncing:** `pixi update`
* **Compile Local Markdown:** `pixi run compile-content`
* **Frontend Dev Server:** `pixi run start-frontend`
* **Compile Production Static Assets:** `pixi run build-static`
* **Testing Runner:** `pixi run test`
* **Linting & Formatting Execution:** `pixi run lint`

## 5. Development Constraints & Guardrails

* **No Global System Traps:** Do NOT install Python dependencies using global or un-isolated `pip install`. Everything must pass through the `pixi.toml` manifest file.
* **Static Type Safety:** All custom Python code written inside `src/` must contain explicit types using the `typing` module. Verify validation parameters cleanly before building.
* **Atomic Git Workflows:** Code adjustments must be written via micro-commits matching semantic descriptions. Never push unreviewed files straight to `main`.

## 6. Deployment & CI/CD

* **Remote:** [`paymantohidifar/paymantohidifar.github.io`](https://github.com/paymantohidifar/paymantohidifar.github.io), live at https://paymantohidifar.github.io/.
* **CI (`.github/workflows/ci.yml`):** Runs on every pull request and on pushes to `main`/`dev`. Executes `pixi run test` plus `ruff check .` / `ruff format --check .` (check-only, no auto-fix in CI).
* **Deploy (`.github/workflows/deploy.yml`):** Runs on push to `main`. Executes `pixi run build-static` and publishes the resulting `public/` directory to GitHub Pages via `actions/deploy-pages`. GitHub Pages is configured with `build_type: workflow` (not the legacy branch-build source).
* **Content assets:** Static files that aren't Markdown/YAML (icons, images) live in `content/assets/` and are copied verbatim into `public/static/assets/` by the Python builder before the frontend bundle step runs — keep `emptyOutDir: false` in `frontend/vite.config.js` so the JS/CSS build doesn't wipe them.