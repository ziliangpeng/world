#!/usr/bin/env python3
"""
Simple markdown file browser and viewer web server.
Serves markdown files from the parent directory with live reload.
"""
import os
import mimetypes
from pathlib import Path
from flask import Flask, render_template_string, send_file, jsonify, request
import markdown2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time

app = Flask(__name__)
# Serve from parent directory (repo root), not from .server/
ROOT_DIR = Path(__file__).parent.parent.resolve()
SCRIPT_DIR = Path(__file__).parent.resolve()
changed_files = []
lock = threading.Lock()


class MarkdownChangeHandler(FileSystemEventHandler):
    """Watch for markdown file changes"""
    def on_modified(self, event):
        if event.src_path.endswith('.md'):
            with lock:
                changed_files.append(event.src_path)


# HTML template for file browser
BROWSER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <div class="breadcrumb">
            {% for part in breadcrumb %}
                <a href="{{ part.url }}">{{ part.name }}</a>
                {% if not loop.last %}<span>/</span>{% endif %}
            {% endfor %}
        </div>

        <h1>{{ current_dir }}</h1>

        <div class="file-list">
            {% if parent %}
            <div class="file-item folder">
                <a href="/?path={{ parent }}">..</a>
            </div>
            {% endif %}

            {% for item in items %}
            <div class="file-item {{ 'folder' if item.is_dir else 'file' }}">
                <a href="{% if item.is_dir %}/?path={{ item.path }}{% else %}/view?path={{ item.path }}{% endif %}">
                    {{ item.name }}{% if item.is_dir %}/{% endif %}
                </a>
            </div>
            {% endfor %}
        </div>
    </div>
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
    <div class="container">
        <div class="breadcrumb">
            {% for part in breadcrumb %}
                <a href="{{ part.url }}">{{ part.name }}</a>
                {% if not loop.last %}<span>/</span>{% endif %}
            {% endfor %}
        </div>

        <div class="markdown-header">
            <h1>{{ filename }}</h1>
            <a href="/?path={{ parent_path }}" class="back-link">← Back to folder</a>
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
        // Syntax highlighting
        hljs.highlightAll();

        // Live reload
        let lastCheck = Date.now();
        const currentPath = "{{ file_path }}";

        setInterval(async () => {
            try {
                const response = await fetch('/changes?since=' + lastCheck + '&path=' + encodeURIComponent(currentPath));
                const data = await response.json();
                lastCheck = Date.now();

                if (data.changed) {
                    location.reload();
                }
            } catch (e) {
                console.error('Live reload check failed:', e);
            }
        }, 1000);
    </script>
</body>
</html>
"""


def get_breadcrumb(path_str):
    """Generate breadcrumb navigation"""
    breadcrumb = [{'name': 'Home', 'url': '/'}]

    if not path_str or path_str == '.':
        return breadcrumb

    parts = Path(path_str).parts
    current_path = ''

    for part in parts:
        current_path = str(Path(current_path) / part)
        breadcrumb.append({
            'name': part,
            'url': f'/?path={current_path}'
        })

    return breadcrumb


@app.route('/')
def browse():
    """Browse files and folders"""
    rel_path = request.args.get('path', '.')
    abs_path = (ROOT_DIR / rel_path).resolve()

    # Security: ensure path is within ROOT_DIR
    if not str(abs_path).startswith(str(ROOT_DIR)):
        return "Access denied", 403

    if not abs_path.exists():
        return "Path not found", 404

    if not abs_path.is_dir():
        return "Not a directory", 400

    # Get parent path
    parent = None
    if abs_path != ROOT_DIR:
        parent = str(abs_path.parent.relative_to(ROOT_DIR))
        if parent == '.':
            parent = ''

    # List directory contents
    items = []
    for item in sorted(abs_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        # Skip hidden files, old server files, and python cache
        if item.name.startswith('.') or item.name in ['mdserver.py', '__pycache__', 'venv']:
            continue

        items.append({
            'name': item.name,
            'path': str(item.relative_to(ROOT_DIR)),
            'is_dir': item.is_dir()
        })

    breadcrumb = get_breadcrumb(rel_path)
    current_dir = abs_path.name if abs_path != ROOT_DIR else 'Root'

    return render_template_string(
        BROWSER_TEMPLATE,
        title=current_dir,
        current_dir=current_dir,
        items=items,
        parent=parent,
        breadcrumb=breadcrumb
    )


@app.route('/view')
def view():
    """View markdown file"""
    rel_path = request.args.get('path', '')
    abs_path = (ROOT_DIR / rel_path).resolve()

    # Security: ensure path is within ROOT_DIR
    if not str(abs_path).startswith(str(ROOT_DIR)):
        return "Access denied", 403

    if not abs_path.exists():
        return "File not found", 404

    if not abs_path.is_file() or not abs_path.suffix == '.md':
        return "Not a markdown file", 400

    # Read and render markdown
    with open(abs_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Render markdown with extras
    html_content = markdown2.markdown(
        md_content,
        extras=['fenced-code-blocks', 'tables', 'header-ids', 'toc']
    )

    # Extract TOC if available
    toc = getattr(html_content, 'toc_html', None) if hasattr(html_content, 'toc_html') else None

    # Get parent directory path
    parent_path = str(abs_path.parent.relative_to(ROOT_DIR))
    if parent_path == '.':
        parent_path = ''

    # Breadcrumb for file view
    breadcrumb = get_breadcrumb(str(abs_path.parent.relative_to(ROOT_DIR)))
    breadcrumb.append({
        'name': abs_path.name,
        'url': f'/view?path={rel_path}'
    })

    return render_template_string(
        VIEWER_TEMPLATE,
        filename=abs_path.name,
        content=html_content,
        toc=toc,
        parent_path=parent_path,
        breadcrumb=breadcrumb,
        file_path=rel_path
    )


@app.route('/changes')
def changes():
    """Check for file changes (for live reload)"""
    since = float(request.args.get('since', 0))
    path = request.args.get('path', '')
    abs_path = str((ROOT_DIR / path).resolve())

    with lock:
        # Check if the specific file was changed
        changed = any(abs_path == changed_path for changed_path in changed_files)
        # Clear old changes
        changed_files.clear()

    return jsonify({'changed': changed})


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files from .server/static/"""
    return send_file(SCRIPT_DIR / 'static' / filename)


def start_file_watcher():
    """Start watching for file changes"""
    event_handler = MarkdownChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, str(ROOT_DIR), recursive=True)
    observer.start()
    return observer


if __name__ == '__main__':
    print(f"Starting markdown server for: {ROOT_DIR}")
    print("Server will be available at: http://localhost:8000")

    # Start file watcher
    observer = start_file_watcher()

    try:
        app.run(debug=True, port=8000, use_reloader=False)
    finally:
        observer.stop()
        observer.join()
