# Python 3.14 Release Notes

**Released:** October 7, 2025
**EOL:** October 2030 (security support - 2 years full support + 3 years security fixes per PEP 602)

## Major Highlights

Python 3.14 is a transformative release that fundamentally changes annotation handling and enables true parallelism:

1. **Deferred annotation evaluation (PEP 649/749)** - Annotations no longer evaluated at definition time, eliminates forward reference quotes, improves performance
2. **Multiple interpreters in stdlib (PEP 734)** - True multi-core parallelism without GIL via `concurrent.interpreters` module
3. **Template string literals (PEP 750)** - T-strings for safe SQL, HTML, and custom string processing
4. **Incremental garbage collection** - Order of magnitude reduction in GC pause times
5. **Free-threading performance** - Single-threaded overhead reduced to 5-10% (down from much higher in 3.13)
6. **Zstandard compression (PEP 784)** - Modern compression format throughout standard library
7. **Safe external debugger (PEP 768)** - Attach debuggers to running processes with zero overhead

## Experimental Features

- 🔴 **Interpreter** Tail-call interpreter - 3-5% performance improvement on benchmarks
  - Enable: Build with `--with-tail-call-interp`
  - Requirements: Clang 19+ on x86-64 or AArch64, profile-guided optimization recommended
  - Status: Production-ready optimization without semantic changes

- 🟡 **Interpreter** JIT compiler in official binaries - Windows and macOS releases include experimental JIT
  - Enable: `PYTHON_JIT=1` environment variable
  - Status: Experimental, modest performance gains currently

## Breaking Changes

- 🔴 **annotations** Deferred annotation evaluation (PEP 649) - Annotations stored as bytecode functions, not evaluated at definition
  - Use `annotationlib.get_annotations()` instead of `__annotations__` for reliable access
  - Forward references no longer need string quotes
  - Migration: Update code that directly accesses `__annotations__`

- 🔴 **multiprocessing** Default start method changed from 'fork' to 'forkserver' on Unix (except macOS)
  - Use `multiprocessing.set_start_method('fork')` if fork required
  - Improves safety and compatibility with libraries

- 🔴 **builtins** int() no longer delegates to `__trunc__()` - Use `__int__()` or `__index__()` instead

- 🔴 **builtins** NotImplemented in boolean context now raises TypeError
  - Use explicit comparison: `if obj.method() == NotImplemented`

- 🟡 **argparse** Removed deprecated parameters and aliases
- 🟡 **ast** Removed deprecated node types and visitors
- 🟡 **asyncio** Removed legacy subprocess APIs and child watcher classes
- 🟡 **email** Removed deprecated utility functions
- 🟡 **importlib.abc** Removed old loader methods - Use newer APIs
- 🟡 **itertools** Removed `izip_longest` - Use `zip_longest`
- 🟡 **pathlib** Removed deprecated `PurePath.is_relative()` method
- 🟡 **pkgutil** Removed legacy import emulation - Use `importlib.util.find_spec()`
- 🟡 **pty** Removed deprecated `master_open()` and `slave_open()` - Use `pty.openpty()`
- 🟡 **sqlite3** Removed deprecated parameter names and version attributes
- 🟡 **urllib** Removed legacy URL parsing methods

## Deprecations

### Removing in Python 3.15

- 🟡 **annotations** `from __future__ import annotations` behavior will change - Future behavior uses deferred evaluation
- 🟡 **Various modules** Additional deprecations scheduled for 3.15 removal

### Removing in Python 3.16

- 🟡 **Various APIs** Multiple deprecations across standard library scheduled for 3.16

### Removing in Python 3.17

- 🟡 **Various APIs** Long-term deprecations scheduled for 3.17

### Removing in Python 3.19

- 🟡 **Various APIs** Very long-term deprecations scheduled for 3.19

### Pending Removal in Future Versions

- 🟢 **Various** Additional soft deprecations with no specific removal timeline yet

## New Features

### Language & Interpreter

- 🔴 **Interpreter** Multiple interpreters in standard library (PEP 734) - `concurrent.interpreters` module for true multi-core parallelism
  - Each interpreter has its own GIL and isolated memory
  - `InterpreterPoolExecutor` for CPU-bound workloads
  - Lower overhead than multiprocessing, actor pattern support

- 🔴 **Syntax** Template string literals (PEP 750) - T-strings for custom string processing
  - Syntax: `t'Hello {name}'` returns Template object
  - Access static text and interpolated parts separately
  - Enables safe SQL queries, HTML escaping, shell commands

- 🔴 **Syntax** Exception expressions without brackets (PEP 758) - `except ValueError, TypeError:` syntax
  - Parentheses still required with `as` clause

- 🔴 **locals()** Deferred annotation evaluation (PEP 649 & 749) - Annotations stored as bytecode, evaluated on demand
  - New `annotationlib` module with `get_annotations()` function
  - Multiple format options: VALUE, FORWARDREF, STRING
  - Forward references no longer need string quotes
  - Significant import performance improvement

- 🔴 **Debugging** Safe external debugger interface (PEP 768) - Attach debuggers to running processes
  - `sys.remote_exec()` executes code in another process
  - Zero-overhead debugging capabilities
  - Security controls: `PYTHON_DISABLE_REMOTE_DEBUG`, `-X disable-remote-debug`, `--without-remote-debug`

- 🟡 **Syntax** Control flow in finally blocks warning (PEP 765) - SyntaxWarning for return/break/continue in finally
  - Helps catch bugs where exceptions are masked

- 🟡 **REPL** Python syntax highlighting enabled by default - 4-bit ANSI colors in interactive shell
  - Customizable via experimental `_colorize.set_theme()`
  - Import auto-completion support
  - Disable with `PYTHON_BASIC_REPL` or color environment variables

- 🟡 **Command-line** `-c` auto-dedents code arguments - Mirrors `textwrap.dedent()` behavior
- 🟡 **Command-line** `-X importtime=2` tracks cached module imports
- 🟡 **Command-line** `-X context_aware_warnings` controls context-aware warning filters
- 🟡 **Command-line** `-X disable-remote-debug` disables remote debugging

### Type Hints & Typing

- 🔴 **typing** Deferred annotation evaluation (PEP 649/749) - Major change to type hint handling
  - `annotationlib` module for controlled annotation evaluation
  - `inspect.signature()` accepts `annotation_format` parameter
  - `Signature.format()` supports `unquote_annotations`

### Error Messages

- 🟡 **Error Messages** Keyword typo suggestions - "Did you mean 'while'?"
- 🟡 **Error Messages** Control flow error detection - "elif cannot follow else"
- 🟡 **Error Messages** Missing expression detection in conditionals
- 🟡 **Error Messages** Incompatible string prefix errors - "u and b are incompatible"
- 🟡 **Error Messages** Invalid `as` target detection in imports/except/match
- 🟡 **Error Messages** Unhashable type errors - "cannot use 'dict' as a set element because it is not hashable"
- 🟡 **Error Messages** Context manager protocol mismatch errors

### Environment & Configuration

- 🟡 **Environment** Context-aware warnings with `-X context_aware_warnings` flag

## Improved Modules

### Standard Library - New Modules

- 🔴 **compression** New compression package with unified interface (PEP 784)
  - `compression.zstd` for Zstandard compression (Meta's high-performance algorithm)
  - `compression.gzip`, `compression.bz2`, `compression.lzma`, `compression.zlib` for existing formats
  - Integrated into `tarfile`, `zipfile`, and `shutil`

- 🔴 **annotationlib** New module for annotation introspection
  - `get_annotations()` function with multiple format options
  - Format.VALUE, Format.FORWARDREF, Format.STRING evaluation modes

- 🔴 **concurrent.interpreters** Multiple interpreter support for true parallelism
  - `interpreters.create()`, `interpreters.list()`, `interpreters.get_current()`
  - `Interpreter.exec()` for code execution in isolated interpreters

- 🟡 **string.templatelib** Template string literal support module

### Standard Library - Major Enhancements

- 🔴 **pathlib** Recursive copy and move operations
  - `Path.copy()` and `Path.copy_into()` for recursive copying
  - `Path.move()` and `Path.move_into()` for recursive moving
  - New `info` attribute caches file type and stat information

- 🔴 **asyncio** Command-line introspection tools
  - `python -m asyncio ps PID` for flat task listing
  - `python -m asyncio pstree PID` for hierarchical async call tree
  - `create_task()` accepts arbitrary keyword arguments
  - `capture_call_graph()` and `print_call_graph()` functions

- 🔴 **pdb** Remote debugging support
  - `python -m pdb -p PID` attaches to running process
  - `set_trace_async()` for coroutine debugging
  - `$_asynctask` variable for current asyncio task
  - 4-space indent and auto-indent support
  - Instance reuse preserves display and commands

- 🔴 **multiprocessing & concurrent.futures** Major enhancements
  - `InterpreterPoolExecutor` for subinterpreter-based parallelism
  - `ProcessPoolExecutor.terminate_workers()` and `kill_workers()` methods
  - `buffersize` parameter for `Executor.map()`
  - List proxies: `clear()`, `copy()` methods
  - Dict proxies: `fromkeys()`, `reversed()`, union operators
  - Set support via `SyncManager.set()`
  - `Process.interrupt()` for graceful termination

- 🟡 **heapq** Max-heap operations
  - `heapify_max()`, `heappush_max()`, `heappop_max()`
  - `heapreplace_max()`, `heappushpop_max()`

- 🟡 **http.server** HTTPS support
  - `HTTPSServer` class for TLS connections
  - Command-line options: `--tls-cert`, `--tls-key`, `--tls-password-file`
  - Dark mode support for directory listings

- 🟡 **ctypes** Enhanced functionality
  - Bit field layout improvements (GCC/Clang/MSVC alignment)
  - `Structure._layout_` for custom ABI matching
  - `CField` class exposure
  - Complex types: `c_float_complex`, `c_double_complex`, `c_longdouble_complex`
  - `ctypes.util.dllist()` lists loaded libraries
  - `memoryview_at()` function
  - `py_object` supports subscription
  - Free-threading build support

### Standard Library - Other Improvements

- 🟡 **argparse** Typo suggestions with `suggest_on_error` parameter, color output support
- 🟡 **datetime** `strptime()` method added to `date` and `time` classes
- 🟡 **decimal** `Decimal.from_number()` alternative constructor, `IEEEContext()` for IEEE 754 formats
- 🟡 **functools** `Placeholder` sentinel for positional argument reservation in `partial()`
- 🟡 **functools** `reduce()` accepts `initial` as keyword argument
- 🟡 **getopt** Support for options with optional arguments
- 🟡 **getpass** Keyboard feedback via optional `echo_char` parameter
- 🟡 **graphlib** `TopologicalSorter.prepare()` callable multiple times
- 🟡 **hmac** Built-in HMAC using HACL* formally verified code (RFC 2104)
- 🟡 **imaplib** `IMAP4.idle()` method implementing RFC 2177
- 🟡 **inspect** `signature()` accepts `annotation_format` parameter
- 🟡 **inspect** `Signature.format()` supports `unquote_annotations`
- 🟡 **inspect** New `ispackage()` function
- 🟡 **io** Non-blocking reads may raise `BlockingIOError`
- 🟡 **io** New `Reader` and `Writer` protocols (simpler than `typing.IO`)
- 🟡 **json** Exception notes for serialization errors, color output support
- 🟡 **json** Command-line: `python -m json` (preferred over `json.tool`)
- 🟡 **linecache** Retrieve source code for frozen modules
- 🟡 **logging.handlers** `QueueListener` supports context manager protocol
- 🟡 **math** Detailed error messages for domain errors
- 🟡 **mimetypes** Public CLI: `python -m mimetypes`
- 🟡 **mimetypes** Extensive new MIME types (fonts, Matroska, FLAC, YAML, etc.)
- 🟡 **operator** `is_none()` and `is_not_none()` functions
- 🟡 **os** `reload_environ()` updates `os.environ` with external changes
- 🟡 **os** `SCHED_DEADLINE` and `SCHED_NORMAL` scheduling constants
- 🟡 **os** `readinto()` function for buffer-based file reading
- 🟡 **os.path** `realpath()` accepts `ALLOW_MISSING` strict mode value (CVE-2025-4517 fix)
- 🟡 **contextvars** `Token` objects support context manager protocol
- 🟡 **curses** `assume_default_colors()` function
- 🟡 **difflib** Dark mode support for HTML comparisons
- 🟡 **dis** Position tracking and specialized bytecode display options
- 🟡 **errno** `EHWPOISON` error code
- 🟡 **faulthandler** C stack trace dumping capability
- 🟡 **fnmatch** `filterfalse()` function for pattern rejection
- 🟡 **fractions** `Fraction.from_number()` and `as_integer_ratio()` support
- 🟡 **threading** Context variable inheritance for threads
- 🟡 **warnings** Context-aware warning filtering with `warnings.catch_warnings()` (PEP 749)

### Built-in Types & Functions

- 🟡 **builtins** `float.from_number()` and `complex.from_number()` class methods for explicit conversion
- 🟡 **builtins** Thousands separators in fractional parts - `1_234.56_78`
- 🟡 **builtins** `pow(x, y, z)` now tries `__rpow__()` if needed
- 🟡 **builtins** Mixed-mode arithmetic (real/complex) follows C99 standard
- 🟡 **builtins** `map()` gains `strict` keyword argument (like `zip()`)
- 🟡 **builtins** `memoryview` supports subscription (generic type)
- 🟡 **builtins** `super` objects are copyable and pickleable
- 🟡 **bytes** `fromhex()` accepts ASCII bytes and bytes-like objects

## Improvements

### Performance

- 🔴 **GC** Incremental garbage collection - Order of magnitude reduction in max pause times
  - Two generations only (young/old)
  - Work spread across multiple operations
  - `gc.collect(1)` now performs increment instead of collecting generation 1

- 🔴 **Interpreter** Tail-call interpreter - 3-5% geometric mean improvement on benchmarks
  - Requires Clang 19+ on x86-64 or AArch64
  - Enable with `--with-tail-call-interp` build flag

- 🔴 **Free-threading** Performance penalty reduced to 5-10% on single-threaded code
  - Specializing adaptive interpreter (PEP 659) now enabled
  - Permanent optimizations replacing temporary workarounds

- 🟡 **Performance** Module-specific optimizations in asyncio, base64, bdb, difflib, gc, io, pathlib, pdb, uuid, zlib

## Implementation Details

### CPython Bytecode

- 🟢 **Bytecode** Disassembly improvements with position tracking
- 🟢 **Bytecode** Specialized bytecode display options
- 🟢 **Bytecode** Changes for incremental GC support
- 🟢 **Bytecode** Free-threading optimizations

### CPython Data Model

- 🟢 **Data Model** Annotations stored as bytecode in `__annotate__` functions
- 🟢 **Data Model** `__static_attributes__` and `__firstlineno__` attributes remain from 3.13

### C API

- 🟡 **C API** Python configuration C API (PEP 741) - Standardized initialization configuration
- 🟡 **C API** Limited C API enhancements - Additional stable ABI functions
- 🟡 **C API** New stable ABI functions for better ABI stability
- 🟡 **C API** Free-threading support for Windows extensions
  - Must specify `Py_GIL_DISABLED` preprocessor variable in build backend
- 🟢 **C API** Removed unsafe and deprecated APIs with migration guidance
- 🟢 **C API** Various functions with pending removal timelines

### Build System

- 🟡 **Build** Tail-call interpreter requires Clang 19+ for optimal performance
- 🟡 **Build** JIT compiler included in official Windows and macOS binaries
- 🟡 **Build** Free-threaded build officially supported (PEP 779)

## Platform & Environment

- 🔴 **Platform** Emscripten tier 3 official platform support (PEP 776)
- 🔴 **Platform** Free-threaded Python officially supported (PEP 779)
- 🔴 **Platform** Android binary releases now provided
- 🟡 **Platform** JIT compiler binaries available for Windows and macOS (experimental)

## Release Process & Meta Changes

- 🟡 **Release** PGP signatures discontinued (PEP 761) - Replaced with modern verification methods
- 🟡 **Release** New build-details.json file provides detailed build information
- 🟡 **Release** Python 3.14+ continues 2 years full support + 3 years security fixes (PEP 602)

## Security Improvements

- 🟡 **Security** CVE-2025-4517 fix - `os.path.realpath()` security enhancement with `ALLOW_MISSING` mode
- 🟡 **Security** Authenticated control socket for multiprocessing forkserver
- 🟡 **Security** HMAC using formally verified HACL* code as OpenSSL fallback
- 🟡 **Security** Remote debugging security controls (environment variable, command-line flag, build flag)
