# Python 2.1 Release Notes

**Released:** April 17, 2001
**EOL:** No formal EOL date (effectively ended with Python 2.2 release in December 2001)

## Major Highlights

Python 2.1 introduced fundamental scoping improvements and future compatibility mechanisms:

1. **Nested scopes (PEP 227)** - Static scoping allows inner functions to access outer function variables
2. **__future__ directives (PEP 236)** - Gradual feature migration mechanism for backward compatibility
3. **Weak references (PEP 205)** - New weakref module for references that don't prevent garbage collection
4. **Rich comparisons (PEP 207)** - Individual overloading of <, <=, >, >=, ==, != operations
5. **Warning framework (PEP 230)** - Structured deprecation system via warnings module
6. **Function attributes (PEP 232)** - Arbitrary attributes can be attached to functions
7. **Faster release cycle** - First release under new 6-9 month release schedule

## New Features

### Language Features

- 🔴 **Scoping** Nested scopes with static scoping (PEP 227) - Inner functions can access enclosing function variables
  - Optional in Python 2.1 via `from __future__ import nested_scopes`
  - Will become default in Python 2.2
  - Fixes lambda variable binding issues
- 🔴 **Syntax** __future__ directives (PEP 236) - Enable future features before they become default
  - Uses `from __future__ import feature_name` syntax
  - Must appear at top of module before other code
- 🔴 **Data Model** Rich comparisons (PEP 207) - Individual methods for each comparison operator
  - New magic methods: `__lt__()`, `__le__()`, `__gt__()`, `__ge__()`, `__eq__()`, `__ne__()`
  - Methods can return any type, not just boolean
  - Enables element-wise matrix comparisons for NumPy
- 🟡 **Functions** Function attributes (PEP 232) - Attach arbitrary attributes to functions
  - Access via `function.__dict__` or `function.attribute_name`
  - Eliminates need to overload docstrings for metadata
- 🟡 **Import** Case-sensitive imports on case-insensitive platforms (PEP 235)
  - `import file` won't import `FILE.PY` by default
  - Set `PYTHONCASEOK` environment variable to restore old behavior

### Standard Library

- 🟡 **inspect** New module for introspecting live Python objects
- 🟡 **pydoc** New module for converting docstrings to HTML/text documentation
  - Includes `pydoc` command-line tool for viewing documentation
  - Includes Tk-based interactive help browser
- 🟡 **weakref** New module for weak references (PEP 205)
  - `weakref.ref(obj)` creates weak reference that doesn't prevent garbage collection
  - `weakref.proxy(obj)` creates transparent proxy with automatic dereferencing
  - Useful for caches and avoiding circular references
- 🟡 **warnings** New module for deprecation warnings (PEP 230)
  - `warnings.warn()` issues warnings
  - `warnings.filterwarnings()` filters warnings by pattern
  - Used to deprecate `regex` module in favor of `re`
- 🟡 **doctest** New unit testing framework based on docstring examples
  - Contributed by Tim Peters
- 🟡 **unittest** PyUnit framework inspired by JUnit
  - Contributed by Steve Purcell
- 🟡 **difflib** Module for computing sequence differences
  - `SequenceMatcher` class computes transformations between sequences
  - Sample script `Tools/scripts/ndiff.py` demonstrates diff-like tool
- 🟢 **curses.panel** Wrapper for panel library (part of ncurses/SYSV curses)
  - Adds depth ordering to curses windows
- 🟡 **xml** Updated XML package with PyXML improvements
  - Support for Expat 1.2 and later
  - Expat parsers handle files in any Python-supported encoding
  - SAX, DOM, and minidom bugfixes
- 🟢 **pstats** Interactive statistics browser for profiling results
  - Invoked when module run as script

### Functions and APIs

- 🟡 **sys** `sys.displayhook()` customizes interactive output (PEP 217)
  - Can replace `repr()` with custom formatting like `pprint.pprint()`
- 🟡 **sys** `sys.excepthook()` handles uncaught exceptions
  - Can customize exception display (e.g., show local variables)
- 🟢 **sys** `sys._getframe([depth])` returns frame objects from call stack
  - Implementation-dependent, CPython only
  - For debugging use only
- 🟡 **time** Time functions like `asctime()` and `localtime()` now accept optional time argument
  - Defaults to current time if not provided
- 🟡 **dict** `dict.popitem()` method removes and returns random (key, value) pair
  - Enables destructive iteration without constructing list of keys
- 🟡 **file** `xreadlines()` method returns iterator over lines
  - Memory-efficient alternative to `readlines()`
  - Doesn't load entire file into memory
- 🟡 **module** `__all__` attribute controls `from module import *` behavior
  - List of names to import with `import *`
  - Prevents pollution from imported modules like `sys` or `string`

### C API

- 🟢 **C API** New `tp_richcmp` slot and rich comparison API (PEP 207)
- 🟢 **C API** New coercion model (PEP 208)
  - Extension types can set `Py_TPFLAGS_CHECKTYPES` flag
  - Numeric methods can return `Py_NotImplemented` to signal unsupported operation
- 🟢 **C API** Warning functions for issuing warnings from C (PEP 230)
- 🟢 **C API** `PyImport_ImportModule()` now respects import hooks
  - C extensions should use this instead of direct imports

## Breaking Changes

- 🟡 **Scoping** `from module import *` and `exec` illegal in function scope with nested scopes
  - Raises `SyntaxError` if function contains nested functions or lambdas with free variables
  - Compiler cannot determine variable bindings at compile time
  - Only applies when nested scopes are enabled via `__future__` import

## Improvements

### Performance

- 🟡 **I/O** Line-oriented file I/O significantly faster (66%+ speedup)
  - `readline()` method rewritten
  - Speedup varies by platform depending on C library `getc()` performance

### Build System

- 🟡 **Build** Distutils-based build system (PEP 229)
  - `setup.py` script autodetects available modules
  - No longer need to manually edit `Modules/Setup` file
  - Easier, more complete Python installations
- 🟡 **Build** Single non-recursive Makefile
  - Faster builds and simpler maintenance
  - Implemented by Neil Schemenauer

### Other Improvements

- 🟡 **Strings** `repr()` now uses hex escapes instead of octal
  - Also uses `\n`, `\t`, `\r` for common characters
- 🟡 **Errors** Syntax errors now include filename and line number
  - Result of compiler reorganization by Jeremy Hylton
- 🟢 **Unicode** Unicode character database shrunk by 340K
- 🟡 **ftplib** Defaults to passive mode for better firewall compatibility
  - Call `set_pasv(0)` to disable if needed
- 🟢 **socket** Raw socket access support
  - Contributed by Grant Edwards

## Deprecations

### Removing in Future Versions

- 🟡 **regex** Module deprecated in favor of `re` module
  - Importing triggers `DeprecationWarning`

## Implementation Details

### Memory Management

- 🟢 **Allocator** Optional specialized object allocator (--with-pymalloc)
  - Faster than system `malloc()` with less overhead
  - Uses memory pools for small allocations
  - C extensions must use correct allocation/deallocation pairs
  - Example: Memory from `PyMem_New()` must be freed with `PyMem_Del()`, not `free()`
  - Contributed by Vladimir Marangozov

### Compiler

- 🟢 **Bytecode** Compiler changes to support nested scopes
  - Different code generation for variables in containing scopes

## Platform & Environment

### New Platforms

- 🟢 **Platform** MacOS X support (Steven Majewski)
- 🟢 **Platform** Cygwin support (Jason Tishler)
- 🟢 **Platform** RISCOS support (Dietmar Schwertberger)
- 🟢 **Platform** Unixware 7 support (Billy G. Allie)

### Environment Variables

- 🟢 **Environment** `PYTHONCASEOK` controls case-sensitive imports
  - Set to enable case-insensitive imports on case-insensitive filesystems

## Release Process & Meta Changes

- 🟡 **Release** Accelerated release cycle to 6-9 months
  - Python 2.1 is first release under new schedule
  - First alpha in January, final release in April (3 months after Python 2.0)
- 🟡 **Release** PEP-driven development process
  - First release to use Python Enhancement Proposals
  - Major features documented with PEPs

### Distutils & Packaging

- 🟡 **Packaging** PKG-INFO metadata in packages (PEP 241)
  - Contains package name, version, author information
  - Foundation for future package catalogs
  - Distutils 1.0.2 includes this feature for earlier Python versions
