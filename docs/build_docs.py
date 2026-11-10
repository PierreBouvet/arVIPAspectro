#!/usr/bin/env python3
"""
arVIPAspectro Documentation Builder
===================================
Automatically scans all markdown step files in `sections/`, extracts titles and anchors,
builds a dynamic Table of Contents, and stitches them into a unified master documentation
file (`arVIPAspectro/docs/README.md`).

Usage:
    python build_docs.py
    python build_docs.py --output CUSTOM_DOC.md
"""

import os
import sys
import re
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def generate_anchor(title: str) -> str:
    """
    Generates a GitHub-compatible markdown anchor link from a heading string.
    """
    clean = title.strip()
    # Remove leading markdown heading hashes
    clean = re.sub(r"^#+\s*", "", clean)
    # Lowercase
    clean = clean.lower()
    # Replace spaces and punctuation (excluding hyphens)
    clean = re.sub(r"[^\w\s-]", "", clean)
    clean = re.sub(r"[\s_]+", "-", clean)
    return clean


def extract_title(content: str, default: str) -> str:
    """
    Extracts the top-level H1 header from markdown content.
    """
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return default


def get_git_commit_hash(repo_dir: Path) -> str:
    """
    Retrieves the short git commit hash if git is available.
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "development"


def build_documentation(docs_dir: Path = None, output_filename: str = "README.md") -> Path:
    """
    Assembles all section markdown files into the master documentation file.
    """
    if docs_dir is None:
        docs_dir = Path(__file__).resolve().parent

    sections_dir = docs_dir / "sections"
    if not sections_dir.exists():
        raise FileNotFoundError(f"Sections directory not found: {sections_dir}")

    # Discover and sort section files
    section_files = sorted(
        [f for f in sections_dir.iterdir() if f.suffix == ".md" and not f.name.startswith(".")],
        key=lambda p: p.name
    )

    if not section_files:
        raise ValueError(f"No markdown files found in {sections_dir}")

    print(f"Discovered {len(section_files)} documentation sections in {sections_dir.name}/")

    parsed_sections = []
    toc_lines = []

    for file_path in section_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        title = extract_title(content, default=file_path.stem)
        anchor = generate_anchor(title)
        parsed_sections.append({
            "path": file_path,
            "filename": file_path.name,
            "title": title,
            "anchor": anchor,
            "content": content.strip()
        })
        toc_lines.append(f"- [{title}](#{anchor})")

    # Header and Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = get_git_commit_hash(docs_dir.parent)

    master_content = [
        '<p align="center">',
        '  <img src="../_static/icon.jpg" alt="arVIPAspectro Icon" width="160">',
        "</p>",
        "",
        "# arVIPAspectro — Complete Pipeline Documentation & Theory Guide",
        "",
        f"> **Automated Build**: Generated on `{timestamp}` (Commit `{git_hash}`).",
        "> This document is automatically compiled from individual step modules in [`sections/`](sections/).",
        "> To regenerate, run `python build_docs.py`.",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]
    master_content.extend(toc_lines)
    master_content.extend(["", "---", ""])

    # Append all sections separated by horizontal dividers
    for sec in parsed_sections:
        master_content.append(sec["content"])
        master_content.extend(["", "---", ""])

    output_path = docs_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(master_content) + "\n")

    print(f"Successfully compiled {len(parsed_sections)} sections into: {output_path}")
    print(f"Total size: {output_path.stat().st_size:,} bytes")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified arVIPAspectro documentation from sections.")
    parser.add_argument(
        "--output",
        "-o",
        default="README.md",
        help="Output filename in arVIPAspectro/docs/ (default: README.md)"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    build_documentation(docs_dir=current_dir, output_filename=args.output)
