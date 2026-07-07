"""Tests for the static site builder (src/portfolio/compiler.py)."""

import datetime
from pathlib import Path

import pytest

from portfolio.compiler import (
    BlogPost,
    SiteBuilder,
    load_about,
    load_blog_posts,
    parse_front_matter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = REPO_ROOT / "src" / "portfolio" / "templates"


def test_parse_front_matter_extracts_metadata_and_body() -> None:
    raw = "---\ntitle: Hello\ndate: 2026-01-01\n---\nBody text here.\n"

    metadata, body = parse_front_matter(raw)

    assert metadata == {"title": "Hello", "date": datetime.date(2026, 1, 1)}
    assert body.strip() == "Body text here."


def test_parse_front_matter_without_delimiters_returns_raw_body() -> None:
    raw = "Just plain content, no front matter."

    metadata, body = parse_front_matter(raw)

    assert metadata == {}
    assert body == raw


def test_load_blog_posts_sorts_newest_first(tmp_path: Path) -> None:
    blogs_dir = tmp_path / "blogs"
    blogs_dir.mkdir()
    (blogs_dir / "old-post.md").write_text(
        "---\ntitle: Old Post\ndate: 2025-01-01\n---\nOld content.\n",
        encoding="utf-8",
    )
    (blogs_dir / "new-post.md").write_text(
        "---\ntitle: New Post\ndate: 2026-01-01\n---\nNew content.\n",
        encoding="utf-8",
    )

    posts = load_blog_posts(tmp_path)

    assert [post.slug for post in posts] == ["new-post", "old-post"]
    assert posts[0].title == "New Post"
    assert "<p>New content.</p>" in posts[0].content


def test_load_blog_posts_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert load_blog_posts(tmp_path) == []


def test_load_about_merges_metadata_and_rendered_html(tmp_path: Path) -> None:
    (tmp_path / "about.md").write_text(
        "---\nname: Jane Doe\n---\nHello **world**.\n",
        encoding="utf-8",
    )

    about = load_about(tmp_path)

    assert about["name"] == "Jane Doe"
    assert "<strong>world</strong>" in about["about_html"]


def test_blog_post_summary_excludes_rendered_content() -> None:
    post = BlogPost(
        slug="my-post",
        title="My Post",
        date="2026-01-01",
        description="A description.",
        content="<p>Full body</p>",
        tags=["Python", "Testing"],
    )

    summary = post.summary()

    assert summary == {
        "slug": "my-post",
        "title": "My Post",
        "date": "2026-01-01",
        "description": "A description.",
        "tags": ["Python", "Testing"],
    }
    assert "content" not in summary


@pytest.fixture
def sample_content_dir(tmp_path: Path) -> Path:
    content_dir = tmp_path / "content"
    (content_dir / "blogs").mkdir(parents=True)

    (content_dir / "about.md").write_text(
        "---\nname: Jane Doe\ntagline: Tester\nemail: jane@example.com\n"
        "github: https://github.com/jane\nlinkedin: https://linkedin.com/in/jane\n"
        "scholar: https://scholar.google.com/citations?user=jane\n---\nAbout body.\n",
        encoding="utf-8",
    )
    (content_dir / "cv.yaml").write_text(
        "education: []\nexperience: []\nskills: []\npublications: []\n"
        "resume_pdf: /static/assets/resume.pdf\n",
        encoding="utf-8",
    )
    (content_dir / "projects.yaml").write_text(
        "- title: Sample Project\n  description: A sample.\n  link: https://example.com\n"
        "  tags: [Python]\n",
        encoding="utf-8",
    )
    (content_dir / "blogs" / "hello.md").write_text(
        "---\ntitle: Hello\ndate: 2026-01-01\ndescription: Test post.\n---\nHi there.\n",
        encoding="utf-8",
    )
    return content_dir


def test_site_builder_generates_expected_pages(
    sample_content_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "public"
    builder = SiteBuilder(sample_content_dir, TEMPLATES_DIR, output_dir)

    builder.build()

    assert (output_dir / "index.html").exists()
    assert (output_dir / "cv.html").exists()
    assert (output_dir / "projects.html").exists()
    assert (output_dir / "blogs.html").exists()
    assert (output_dir / "blogs" / "hello.html").exists()

    about_html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Jane Doe" in about_html

    post_html = (output_dir / "blogs" / "hello.html").read_text(encoding="utf-8")
    assert "Hi there." in post_html
