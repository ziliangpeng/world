#!/usr/bin/env python3
"""
Alternative markdown viewer server with direct path routing.

This version maps URL paths directly to filesystem paths under ROOT_DIR:
  - "/" or "/some/dir" -> directory browser
  - "/path/to/file.md" -> rendered markdown viewer
  - "/path/to/other.ext" -> raw file content

Special API/static routes:
  - "/static/..."  -> CSS/JS assets
  - "/changes"     -> live reload polling
"""

from pathlib import Path
import threading
import os

from flask import Flask, render_template_string, send_file, jsonify, request
import markdown2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


app = Flask(__name__)

# Serve from parent directory (repo root), not from .server/
ROOT_DIR = Path(__file__).parent.parent.resolve()
SCRIPT_DIR = Path(__file__).parent.resolve()
changed_files = []
lock = threading.Lock()


class MarkdownChangeHandler(FileSystemEventHandler):
    """Watch for markdown file changes"""

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".md"):
            with lock:
                changed_files.append(str(Path(event.src_path).resolve()))


# HTML template for file browser
BROWSER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <button id="darkModeToggle" class="theme-toggle" aria-label="Toggle dark mode">
        <span class="theme-toggle-icon">🌕</span>
    </button>

    <div class="container">
        <div class="breadcrumb">
            {% for part in breadcrumb %}
                <a href="{{ part.url }}">{{ part.name }}</a>
                {% if not loop.last %}<span>/</span>{% endif %}
            {% endfor %}
        </div>

        <h1>{{ current_dir }}</h1>

        <div class="file-list">
            {% if parent_url %}
            <div class="file-item folder">
                <a href="{{ parent_url }}">..</a>
            </div>
            {% endif %}

            {% for item in items %}
            <div class="file-item {{ 'folder' if item.is_dir else 'file' }}">
                <a href="{{ item.url }}">
                    {{ item.name }}{% if item.is_dir %}/{% endif %}
                </a>
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        // Dark mode toggle
        const darkModeToggle = document.getElementById('darkModeToggle');
        const darkModeIcon = document.querySelector('.theme-toggle-icon');
        const htmlElement = document.documentElement;

        // Check for saved preference or default to dark mode
        const currentTheme = localStorage.getItem('theme') || 'dark';
        htmlElement.setAttribute('data-theme', currentTheme);
        darkModeIcon.textContent = currentTheme === 'dark' ? '💡' : '🌕';

        darkModeToggle.addEventListener('click', () => {
            const newTheme = htmlElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            htmlElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            darkModeIcon.textContent = newTheme === 'dark' ? '💡' : '🌕';
        });
    </script>
</body>
</html>
"""

# HTML template for markdown viewer
VIEWER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ filename }}</title>
    <link rel="stylesheet" href="/static/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
</head>
<body>
    <button id="darkModeToggle" class="theme-toggle" aria-label="Toggle dark mode">
        <span class="theme-toggle-icon">🌕</span>
    </button>

    <div class="container">
        <div class="breadcrumb">
            {% for part in breadcrumb %}
                <a href="{{ part.url }}">{{ part.name }}</a>
                {% if not loop.last %}<span>/</span>{% endif %}
            {% endfor %}
        </div>

        <div class="markdown-header">
            <h1>{{ filename }}</h1>
            <a href="{{ parent_url }}" class="back-link">← Back to folder</a>
        </div>

        {% if toc %}
        <div class="toc">
            <h3>Table of Contents</h3>
            {{ toc|safe }}
        </div>
        {% endif %}

        <div class="markdown-content">
            {{ content|safe }}
        </div>
    </div>

    <script>
        // Dark mode toggle
        const darkModeToggle = document.getElementById('darkModeToggle');
        const darkModeIcon = document.querySelector('.theme-toggle-icon');
        const htmlElement = document.documentElement;

        // Check for saved preference or default to dark mode
        const currentTheme = localStorage.getItem('theme') || 'dark';
        htmlElement.setAttribute('data-theme', currentTheme);
        darkModeIcon.textContent = currentTheme === 'dark' ? '💡' : '🌕';

        darkModeToggle.addEventListener('click', () => {
            const newTheme = htmlElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            htmlElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            darkModeIcon.textContent = newTheme === 'dark' ? '💡' : '🌕';
        });

        // Syntax highlighting
        hljs.highlightAll();

        // Live reload - polling based approach (checks every 2s)
        let lastCheck = Date.now();
        const currentPath = "{{ rel_path }}";

        setInterval(async () => {
            try {
                const response = await fetch('/changes?since=' + lastCheck + '&path=' + encodeURIComponent(currentPath));
                const data = await response.json();
                lastCheck = Date.now();

                if (data.changed) {
                    console.log('File changed, reloading...');
                    location.reload();
                }
            } catch (e) {
                console.error('Live reload check failed:', e);
            }
        }, 2000);
    </script>
</body>
</html>
"""


def get_breadcrumb_for_path(rel_path: str):
    """Generate breadcrumb navigation for a path-style URL."""
    breadcrumb = [{"name": "Home", "url": "/"}]

    if not rel_path:
        return breadcrumb

    parts = Path(rel_path).parts
    current = ""
    for part in parts:
        current = str(Path(current) / part) if current else part
        breadcrumb.append(
            {
                "name": part,
                "url": "/" + current,
            }
        )
    return breadcrumb


def list_directory(rel_path: str):
    """Render directory browser for given relative path."""
    abs_path = (ROOT_DIR / rel_path).resolve()

    # Security: ensure path is within ROOT_DIR
    if not str(abs_path).startswith(str(ROOT_DIR)):
        return "Access denied", 403

    if not abs_path.exists():
        return "Path not found", 404

    if not abs_path.is_dir():
        return "Not a directory", 400

    parent_url = None
    if abs_path != ROOT_DIR:
        parent_rel = str(abs_path.parent.relative_to(ROOT_DIR))
        parent_url = "/" + parent_rel if parent_rel else "/"

    items = []
    for item in sorted(abs_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        if item.name.startswith(".") or item.name in ["mdserver.py", "__pycache__", "venv"]:
            continue
        rel_item = item.relative_to(ROOT_DIR)
        url = "/" + str(rel_item)
        items.append(
            {
                "name": item.name,
                "url": url,
                "is_dir": item.is_dir(),
            }
        )

    breadcrumb = get_breadcrumb_for_path(rel_path)
    current_dir = abs_path.name if abs_path != ROOT_DIR else "Root"

    return render_template_string(
        BROWSER_TEMPLATE,
        title=current_dir,
        current_dir=current_dir,
        items=items,
        parent_url=parent_url,
        breadcrumb=breadcrumb,
    )


def render_markdown(rel_path: str):
    """Render a markdown file as HTML."""
    abs_path = (ROOT_DIR / rel_path).resolve()

    if not str(abs_path).startswith(str(ROOT_DIR)):
        return "Access denied", 403

    if not abs_path.exists():
        return "File not found", 404

    if not abs_path.is_file() or abs_path.suffix != ".md":
        return "Not a markdown file", 400

    with abs_path.open("r", encoding="utf-8") as f:
        md_content = f.read()

    html_content = markdown2.markdown(
        md_content, extras=["fenced-code-blocks", "tables", "header-ids", "toc"]
    )
    toc = getattr(html_content, "toc_html", None) if hasattr(html_content, "toc_html") else None

    if abs_path.parent == ROOT_DIR:
        parent_url = "/"
    else:
        parent_rel = str(abs_path.parent.relative_to(ROOT_DIR))
        parent_url = "/" + parent_rel

    rel_path_str = str(Path(rel_path))
    breadcrumb = get_breadcrumb_for_path(str(Path(rel_path).parent))
    breadcrumb.append(
        {
            "name": abs_path.name,
            "url": "/" + rel_path_str,
        }
    )

    return render_template_string(
        VIEWER_TEMPLATE,
        filename=abs_path.name,
        content=html_content,
        toc=toc,
        parent_url=parent_url,
        breadcrumb=breadcrumb,
        rel_path=rel_path_str,
    )


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files from .server/static/"""
    return send_file(SCRIPT_DIR / "static" / filename)


@app.route("/changes")
def changes():
    """Check for file changes (for live reload)"""
    path = request.args.get("path", "")
    abs_path = str((ROOT_DIR / path).resolve())

    with lock:
        changed = any(abs_path == changed_path for changed_path in changed_files)
        if changed:
            changed_files[:] = [f for f in changed_files if f != abs_path]

    return jsonify({"changed": changed})


@app.route("/", defaults={"req_path": ""})
@app.route("/<path:req_path>")
def serve(req_path):
    """
    Catch-all handler:
      - directory -> browser
      - .md file  -> rendered viewer
      - other file -> raw content
    """
    abs_path = (ROOT_DIR / req_path).resolve()

    # Security: ensure path is within ROOT_DIR
    if not str(abs_path).startswith(str(ROOT_DIR)):
        return "Access denied", 403

    if not abs_path.exists():
        return "Not found", 404

    rel_path = str(abs_path.relative_to(ROOT_DIR))

    if abs_path.is_dir():
        return list_directory(rel_path)

    if abs_path.is_file() and abs_path.suffix == ".md":
        return render_markdown(rel_path)

    # Other files (images, PDFs, etc.)
    return send_file(abs_path)


def start_file_watcher():
    """Start watching for file changes"""
    event_handler = MarkdownChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, str(ROOT_DIR), recursive=True)
    observer.start()
    return observer


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting markdown server2 (direct paths) for: {ROOT_DIR}")
    print(f"Server2 will be available at: http://localhost:{port}")

    observer = start_file_watcher()
    try:
        app.run(host="0.0.0.0", debug=True, port=port, use_reloader=False)
    finally:
        observer.stop()
        observer.join()
