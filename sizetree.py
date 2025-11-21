#!/usr/bin/env python3
"""Tree command with file sizes."""

import os
import sys


def format_size(size):
    """Format size in human-readable format."""
    for unit in ['B', 'K', 'M', 'G', 'T']:
        if size < 1024:
            return f"{size:>5.0f}{unit}" if unit == 'B' else f"{size:>5.1f}{unit}"
        size /= 1024
    return f"{size:>5.1f}P"


def sizetree(path, prefix=""):
    """Print directory tree with file sizes."""
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return

    # Filter hidden files
    entries = [e for e in entries if not e.startswith('.')]

    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "

        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            print(f"{prefix}{connector}({format_size(size)}) {entry}")
        else:
            print(f"{prefix}{connector}{entry}/")
            new_prefix = prefix + ("    " if is_last else "│   ")
            sizetree(full_path, new_prefix)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"{target}/")
    sizetree(target)
