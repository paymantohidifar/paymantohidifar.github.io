# paymantohidifar.github.io

Personal portfolio site for Payman Tohidifar. Statically generated from Markdown/YAML
content and deployed to GitHub Pages.

Live at: https://paymantohidifar.github.io/

## Stack

- **Backend:** Python static site builder (`src/portfolio/compiler.py`) using
  Jinja2 templates, PyYAML front-matter, and Python-Markdown to render
  content into static HTML.
- **Frontend:** Vanilla JS/CSS (theme toggle, nav highlighting), bundled with
  Vite.
- **Environment:** Managed with [pixi](https://pixi.sh).
- **Deployment:** GitHub Actions builds the site and publishes it to GitHub
  Pages on every push to `main`.

## Project layout

```
content/            Markdown/YAML site content (about, cv, projects, blogs)
src/portfolio/       Python site builder + Jinja2 templates
frontend/            Vite-bundled JS/CSS source
public/              Generated static site output (gitignored)
tests/               Pytest suite for the site builder
.github/workflows/   CI and Pages deploy workflows
```

## Development

Day-to-day work happens on the `dev` branch; `main` is only updated via
reviewed pull requests.

```sh
pixi install              # sync the Python/Node environment
pixi run compile-content   # render content -> public/*.html
pixi run start-frontend    # local dev/preview server
pixi run build-static       # full production build (content + bundled assets)
pixi run test               # run the pytest suite
pixi run lint                # ruff check --fix && ruff format
```

## Content

- `content/about.md` — profile, tagline, and social links (front matter)
- `content/cv.yaml` — education, experience, skills, publications
- `content/projects.yaml` — project list
- `content/blogs/*.md` — blog posts (Markdown with YAML front matter:
  `title`, `date`, `description`)

Edit these files and re-run `pixi run build-static` to regenerate the site.

## Deployment

Pushing to `main` (via a merged PR from `dev`) triggers
`.github/workflows/deploy.yml`, which builds the site with pixi and publishes
`public/` to GitHub Pages. `.github/workflows/ci.yml` runs tests and lint
checks on every PR and push to `main`/`dev`.
