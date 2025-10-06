# Python 0.9.0 - The First Public Release

## Overview

Python 0.9.0 was the **first public release** of the Python programming language, posted to the Usenet newsgroup `alt.sources` on **February 20, 1991** by Guido van Rossum at CWI (Centrum Wiskunde & Informatica) in Amsterdam, The Netherlands.

## Timeline

### Development History
- **December 1989**: Guido van Rossum started working on Python at CWI as a "hobby project" during Christmas week
- **Early 1990**: First working version completed (internal/private)
- **February 20, 1991**: Python 0.9.0 released publicly to alt.sources (actually labeled as 0.9.1 patchlevel 0)
- **February 1991**: Python 0.9.1 (patchlevel 1) released

### Python 0.9.x Release Dates

From the HISTORY file in Python 1.0.1:

- **0.9.0** - February 1991 (original posting to alt.sources)
- **0.9.1** - February 1991 (micro changes, added patchlevel.h)
- **0.9.2** - Autumn 1991 (major: continue statement, semicolons, dict constructors, .pyc files, long integers, sockets)
- **0.9.3** - Never made available outside CWI (internal release)
- **0.9.4** - December 24, 1991 (major: new argument handling, apply(), new exceptions, new class syntax, global statement)
- **0.9.6** - April 6, 1992 (major: try/except/finally, dynamic loading, varargs, debugging/profiling, `==` operator)
- **0.9.7beta** - Date unknown (major: special methods like `__getitem__`, Configure.py build system)
- **0.9.8** - January 9, 1993 (major: stricter argument checking, varargs `*args`, profile features, many stdlib additions)
- **0.9.9** - July 29, 1993 (major: X11 GUI support, speed improvements, `__init__`/`__del__`, string formatting `%`)

## First Release Description

From the original README file in Python 0.9.1:

> This is Python, an extensible interpreted programming language that combines remarkable power with very clear syntax.
>
> This is version 0.9 (the first beta release), patchlevel 1.
>
> Python can be used instead of shell, Awk or Perl scripts, to write prototypes of real applications, or as an extension language of large systems, you name it.

### Platform Support
- Most modern versions of UNIX
- Macintosh
- Potentially MS-DOS (untested)
- Primarily developed on SGI IRIS workstation (IRIX 3.1 and 3.2)
- Tested on SunOS 4.1 and BSD 4.3 (tahoe)

### Built-in Modules (0.9.0)
- Operating system interfaces
- X11 window system (via STDWIN)
- Mac window system (via STDWIN)
- Silicon Graphics GL library

## Historical Context

Python was created as a successor to the **ABC programming language**, which Guido van Rossum had worked on at CWI. ABC was inspired by SETL and designed for teaching programming. Python improved upon ABC by adding:
- Exception handling
- Ability to interface with the Amoeba operating system
- Appeal to Unix/C hackers

The first piece of code written for Python was a simple LL(1) parser generator called `pgen`, which is still part of Python's source distribution today.

## Source Materials

### Original Release
- **Usenet Archive**: Python 0.9.1 was posted to `alt.sources` as 21 shell archive (shar) parts
- **Reconstructed Source**: Andrew Dalke scraped the Usenet archives and assembled a tarball
- **Available at**: https://www.python.org/ftp/python/src/Python-0.9.1.tar.gz

### Documentation Sources
1. **README file**: `/tmp/Python-0.9.1/README` - Original introduction and overview
2. **Tutorial**: `/tmp/Python-0.9.1/doc/tut.tex` - LaTeX tutorial (incomplete in beta)
3. **Library Reference**: `/tmp/Python-0.9.1/doc/mod*.tex` - LaTeX library documentation (incomplete in beta)
4. **Man page**: `/tmp/Python-0.9.1/python.man` - UNIX manual page

### Historical References
- **Guido's Personal History**: http://python-history.blogspot.com/2009/01/personal-history-part-1-cwi.html
- **Python.org Early Releases**: https://www.python.org/download/releases/early/
- **Usenet Archive**: https://www.tuhs.org/Usenet/alt.sources/1991-February/001749.html

## Features Present in First Release

Based on the 0.9.1 HISTORY file, Python 0.9.0 already included:
- Classes with inheritance
- Exception handling
- Functions
- Core data types: list, dict, str, tuple
- Module system

The HISTORY file in later versions documents the evolution from 0.9.0 onwards, showing that even the first public release was already a sophisticated language.

## Note on Release Notes

Python 0.9.0 did **not have formal release notes** in the modern sense, as it was the first public release. The README file served as both an introduction and announcement. The only "change note" in the HISTORY file states simply:

> Original posting to alt.sources.

For detailed feature descriptions of Python 0.9.0, refer to the original LaTeX documentation in the source tarball or examine the preserved HISTORY file in `python_0.9.x_release.history.md`.
