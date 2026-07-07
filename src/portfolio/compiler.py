"""Static site builder: renders Markdown/YAML content into static HTML via Jinja2 templates."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import markdown
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTENT_DIR = PROJECT_ROOT / "content"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
OUTPUT_DIR = PROJECT_ROOT / "public"

SITE_NAME = "Payman Tohidifar — Portfolio"
SITE_DESCRIPTION = (
    "Portfolio of Payman Tohidifar, computational & synthetic biologist "
    "and AIxBio enthusiast."
)


@dataclass(frozen=True)
class BlogPost:
    """A single parsed and rendered blog post."""

    slug: str
    title: str
    date: str
    description: str
    content: str

    def summary(self) -> Dict[str, Any]:
        """Metadata used for the blog listing page (excludes rendered HTML body)."""
        return {
            "slug": self.slug,
            "title": self.title,
            "date": self.date,
            "description": self.description,
        }


def parse_front_matter(raw_content: str) -> Tuple[Dict[str, Any], str]:
    """Splits a Markdown file with a YAML front-matter header into (metadata, body)."""
    if not raw_content.startswith("---"):
        return {}, raw_content
    parts = raw_content.split("---", 2)
    if len(parts) < 3:
        return {}, raw_content
    metadata = yaml.safe_load(parts[1]) or {}
    return metadata, parts[2]


def render_markdown(text: str) -> str:
    """Converts a Markdown body into HTML using the standard extension set."""
    return markdown.Markdown(extensions=["fenced_code", "tables"]).convert(text)


_DATE_FORMATS = ("%Y-%m-%d", "%B %Y", "%b %Y")


def _parse_post_date(raw_date: str) -> date:
    """Parses a front-matter date string (e.g. "March 2025", "2026-01-15") for sorting."""
    for fmt in _DATE_FORMATS:
        try:
            return date(*datetime.strptime(raw_date, fmt).timetuple()[:3])
        except ValueError:
            continue
    return date.min


def load_blog_posts(content_dir: Path) -> List[BlogPost]:
    """Loads and renders every Markdown post under content/blogs, newest first."""
    posts: List[BlogPost] = []
    blogs_dir = content_dir / "blogs"
    if not blogs_dir.exists():
        return posts

    for md_file in sorted(blogs_dir.glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        metadata, body = parse_front_matter(raw)
        posts.append(
            BlogPost(
                slug=md_file.stem,
                title=metadata.get("title", md_file.stem.replace("-", " ").title()),
                date=str(metadata.get("date", "2026-01-01")),
                description=metadata.get("description", ""),
                content=render_markdown(body),
            )
        )

    posts.sort(key=lambda post: _parse_post_date(post.date), reverse=True)
    return posts


def load_about(content_dir: Path) -> Dict[str, Any]:
    """Loads the About page front-matter (profile info) plus rendered body HTML."""
    raw = (content_dir / "about.md").read_text(encoding="utf-8")
    metadata, body = parse_front_matter(raw)
    metadata["about_html"] = render_markdown(body)
    return metadata


def load_yaml(path: Path) -> Any:
    """Loads a YAML content file (CV data, project list, etc.)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class SiteBuilder:
    """Renders all static portfolio pages from content files and Jinja2 templates."""

    def __init__(
        self, content_dir: Path, templates_dir: Path, output_dir: Path
    ) -> None:
        self.content_dir = content_dir
        self.output_dir = output_dir
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html"]),
        )

    def build(self) -> None:
        """Generates every static page (about/cv/projects/blogs + individual posts)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._copy_static_assets()

        about = load_about(self.content_dir)
        cv = load_yaml(self.content_dir / "cv.yaml")
        projects = load_yaml(self.content_dir / "projects.yaml")
        posts = load_blog_posts(self.content_dir)

        base_ctx = self._base_context(about)

        self._render(
            "about.html", "index.html", {**base_ctx, "about_html": about["about_html"]}
        )
        self._render("cv.html", "cv.html", {**base_ctx, "cv": cv})
        self._render(
            "projects.html", "projects.html", {**base_ctx, "projects": projects}
        )
        self._render(
            "blogs.html",
            "blogs.html",
            {**base_ctx, "posts": [post.summary() for post in posts]},
        )
        for post in posts:
            post_ctx = {**post.summary(), "content": post.content}
            self._render(
                "blog_post.html",
                f"blogs/{post.slug}.html",
                {**base_ctx, "post": post_ctx},
            )

        print(
            f"[+] Static site built: {len(posts)} blog post(s), output in {self.output_dir}"
        )

    def _copy_static_assets(self) -> None:
        """Copies non-templated content assets (images, icons) into public/static/assets."""
        assets_dir = self.content_dir / "assets"
        if not assets_dir.exists():
            return
        target_dir = self.output_dir / "static" / "assets"
        target_dir.mkdir(parents=True, exist_ok=True)
        for asset in assets_dir.iterdir():
            if asset.is_file():
                shutil.copy2(asset, target_dir / asset.name)

    @staticmethod
    def _base_context(about: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "site_name": SITE_NAME,
            "site_description": SITE_DESCRIPTION,
            "name": about.get("name", ""),
            "tagline": about.get("tagline", ""),
            "email": about.get("email", ""),
            "github": about.get("github", ""),
            "linkedin": about.get("linkedin", ""),
            "scholar": about.get("scholar", ""),
            "year": date.today().year,
        }

    def _render(
        self, template_name: str, output_relpath: str, context: Dict[str, Any]
    ) -> None:
        template = self.env.get_template(template_name)
        html = template.render(**context)
        out_path = self.output_dir / output_relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")


def main() -> None:
    builder = SiteBuilder(CONTENT_DIR, TEMPLATES_DIR, OUTPUT_DIR)
    builder.build()


if __name__ == "__main__":
    main()
