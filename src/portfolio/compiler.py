"""Portfolio compiler engine for converting Markdown content to static assets."""

import json
from pathlib import Path
from typing import Any, Dict, List
import jinja2
import markdown
import yaml


class StaticSiteCompiler:
    """Core static compiler applying SOLID principles for portfolio resource generation."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.content_dir = root_dir / "content" / "blogs"
        self.output_dir = root_dir / "public" / "static"
        self.md_processor = markdown.Markdown(extensions=["fenced_code", "tables"])

    def setup_directories(self) -> None:
        """Ensures file structures are initialized safely without race conditions."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_markdown_file(self, file_path: Path) -> Dict[str, Any]:
        """Parses front-matter metadata and compiles markdown string blocks into clean HTML.

        Adheres to robust string isolation processing patterns.
        """
        with file_path.open("r", encoding="utf-8") as f:
            raw_content = f.read()

        # Isolate YAML front matter section bounded by triple hyphens
        if raw_content.startswith("---"):
            parts = raw_content.split("---", 2)
            metadata = yaml.safe_load(parts[1]) if len(parts) > 2 else {}
            markdown_text = parts[2] if len(parts) > 2 else parts[1]
        else:
            metadata = {}
            markdown_text = raw_content

        compiled_html = self.md_processor.convert(markdown_text)

        return {
            "slug": file_path.stem,
            "title": metadata.get("title", file_path.stem.replace("-", " ").title()),
            "date": str(metadata.get("date", "2026-01-01")),
            "description": metadata.get("description", ""),
            "content": compiled_html,
        }

    def compile(self) -> None:
        """Executes the complete generation loop across local raw text trees."""
        self.setup_directories()
        blog_posts: List[Dict[str, Any]] = []

        if not self.content_dir.exists():
            return

        # Core file stream processing using generators
        for md_file in self.content_dir.glob("*.md"):
            try:
                post_data = self.parse_markdown_file(md_file)
                blog_posts.append(post_data)

                # Write out individual compiled JSON file payloads for async frontend picking
                individual_output = self.output_dir / f"{post_data['slug']}.json"
                with individual_output.open("w", encoding="utf-8") as out_f:
                    json.dump(post_data, out_f, indent=2)

            except Exception as e:
                print(f"[-] Structural parsing failure on file node {md_file}: {e}")

        # Sort posts chronologically by date descending
        blog_posts.sort(key=lambda x: x["date"], reverse=True)

        # Write out a central database manifest mapping file for index generation
        manifest_path = self.output_dir / "blog_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as manifest_f:
            json.dump(blog_posts, manifest_f, indent=2)

        print(f"[+] Static processing complete. Generated {len(blog_posts)} blog artifacts.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    compiler = StaticSiteCompiler(project_root)
    compiler.compile()