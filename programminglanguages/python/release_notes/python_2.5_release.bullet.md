# Python 2.5 Release Notes

**Released:** September 19, 2006
**EOL:** May 26, 2011

## Major Highlights

Python 2.5 brings fundamental new control flow patterns and major standard library additions:

1. **with statement (PEP 343)** - Context managers for clean resource management (requires `from __future__ import with_statement`)
2. **Conditional expressions** - Ternary operator `x = true_value if condition else false_value` (PEP 308)
3. **Generator enhancements (PEP 342)** - Generators become coroutines with send(), throw(), and close() methods
4. **ElementTree in stdlib** - Fast XML processing with xml.etree package
5. **sqlite3 in stdlib** - Built-in SQLite database support
6. **ctypes in stdlib** - Call C functions from Python without writing C extensions
7. **Unified try/except/finally (PEP 341)** - Can now combine except blocks with finally in single statement

## New Features

### Language Syntax

- 🔴 **Syntax** Conditional expressions (PEP 308) - Ternary operator `x = true_value if condition else false_value`
- 🔴 **Syntax** with statement for context management (PEP 343) - Requires `from __future__ import with_statement`, automatic resource cleanup
- 🔴 **Syntax** Unified try/except/finally (PEP 341) - Can combine except blocks with finally block in single statement
- 🟡 **import** Absolute and relative imports (PEP 328) - `from __future__ import absolute_import` makes absolute imports default, relative imports use leading dots
- 🟡 **import** Relative imports with leading dots - `from .module import name` or `from ..package import name`

### Generator Enhancements

- 🔴 **Generators** Generators become coroutines (PEP 342) - yield is now an expression that can receive values
- 🔴 **Generators** send() method passes values into generators - `gen.send(value)` resumes generator with value
- 🟡 **Generators** throw() method raises exceptions inside generators - `gen.throw(exception)`
- 🟡 **Generators** close() method terminates generators cleanly - Called automatically by garbage collector
- 🟡 **Generators** yield in try/finally now allowed - Enables guaranteed cleanup code in generators

### Standard Library - New Modules

- 🔴 **ctypes** Foreign function library for calling C functions (PEP 3118) - Load shared libraries and call functions without writing C extensions
- 🔴 **xml.etree** ElementTree XML processing package - Fast, Pythonic XML parsing and generation
- 🔴 **sqlite3** SQLite database interface - Embedded SQL database with DB-API 2.0 interface
- 🔴 **hashlib** Unified cryptographic hashing - SHA-1, SHA-224, SHA-256, SHA-384, SHA-512, MD5 with consistent interface
- 🟡 **contextlib** Utilities for with statement - contextmanager decorator, nested(), closing() helpers
- 🟡 **functools** Functional programming tools (PEP 309) - partial() for partial function application, update_wrapper() and wraps() for decorators
- 🟡 **wsgiref** WSGI reference implementation - Tools for building WSGI servers and applications
- 🟡 **uuid** UUID generation (RFC 4122) - Generate universally unique identifiers
- 🟡 **runpy** Execute modules as scripts (PEP 338) - Implementation of python -m switch functionality

### collections Module

- 🔴 **collections** defaultdict class - Dictionary with default value factory for missing keys
- 🟡 **collections** deque improvements - Added remove() method and improved performance

### String Methods

- 🟡 **str** partition() and rpartition() methods - Split string into (before, separator, after) tuple
- 🟡 **str** startswith() and endswith() accept tuples - Check for multiple prefixes/suffixes at once

### Built-in Functions & Types

- 🟡 **dict** __missing__() hook for subclasses - Called when key not found, enables defaultdict
- 🟡 **builtins** any() and all() functions - Check if any/all elements in iterable are true
- 🟡 **builtins** min() and max() gain key parameter - Similar to sort(key=...) for custom comparisons

### Exception Handling

- 🔴 **Exceptions** New exception hierarchy with BaseException (PEP 352) - KeyboardInterrupt and SystemExit now under BaseException, not Exception
- 🔴 **Exceptions** All built-in exceptions now new-style classes - Exception class hierarchy modernized
- 🟡 **Exceptions** except Exception pattern recommended - Catches program errors but not KeyboardInterrupt/SystemExit

### Context Managers

- 🟡 **file** File objects support with statement - Automatic closing with `with open(...) as f:`
- 🟡 **threading** Locks and conditions support with statement - Automatic acquire/release
- 🟡 **decimal** localcontext() for temporary precision changes - Context manager for decimal arithmetic settings

## Improvements

### Performance

- 🟡 **Performance** Unicode string optimizations - casefold operations up to 10x faster
- 🟡 **Performance** Long integer optimizations - Faster arithmetic and string conversion
- 🟡 **Performance** Dictionary lookups optimized - Better performance for string keys

### Other Language Changes

- 🟡 **try/finally** try/finally blocks can contain yield - Enables cleanup code in generators
- 🟡 **builtins** Absolute value for complex numbers (abs()) - Returns magnitude of complex number
- 🟢 **-m switch** python -m now supports packages - Can run packages like `python -m package.module`

## Deprecations

- 🟡 **Exceptions** String exceptions deprecated - `raise "error"` now triggers DeprecationWarning, will be removed in Python 3.0
- 🟢 **contextlib** contextlib.nested() will be deprecated - Use multiple with statements instead (deprecated in 2.7)

## Implementation Details

### C API

- 🔴 **C API** Py_ssize_t type for sizes and indexes (PEP 353) - Enables handling >2GB data on 64-bit platforms
- 🟡 **C API** __index__() protocol for integer-like types (PEP 357) - Allows custom types to be used as slice indexes
- 🟡 **C API** PyArg_ParseTuple() gains 'n' format code - For Py_ssize_t arguments
- 🟢 **C API** nb_index slot added to PyNumberMethods - C-level support for __index__() protocol

### Interpreter

- 🟡 **Interpreter** with statement bytecode support - New opcodes for context management protocol
- 🟡 **Interpreter** Generator bytecode changes - Support for send(), throw(), close() methods
- 🟢 **Interpreter** Improved import system - Better support for absolute/relative imports

### Build System

- 🟢 **Build** Universal binary support for Mac OS X - Single binary runs on PowerPC and Intel Macs
- 🟢 **Build** Improved 64-bit platform support - Better handling of large data structures

## Platform & Environment

- 🟡 **Mac OS X** Universal binary support - Single Python binary for PowerPC and Intel architectures
- 🟢 **Windows** Visual Studio 2005 now supported - Better Windows build toolchain
- 🟢 **Unix** Improved platform detection - Better configuration for various Unix platforms

## Other Changes

### Interactive Interpreter

- 🟢 **REPL** Ctrl+C handling improved - Better KeyboardInterrupt behavior in interactive mode

### Distutils Enhancements

- 🟡 **distutils** Package metadata v1.1 (PEP 314) - Added requires, provides, obsoletes, download_url fields
- 🟡 **distutils** upload command for PyPI - Direct package upload to Python Package Index

### Standard Library Improvements

- 🟡 **mailbox** Module rewritten - Better performance and reliability
- 🟡 **mmap** Added rfind() method - Search from end of memory-mapped file
- 🟡 **operator** itemgetter() and attrgetter() support multiple items - Return tuple of values
- 🟡 **optparse** Option groups and conflict handling - Better command-line parsing organization
- 🟡 **os** Added SEEK_CUR and SEEK_END constants - Explicit constants for file seeking
- 🟡 **pprint** Better formatting for recursive data structures - Improved pretty-printing
- 🟡 **subprocess** Added communicate() timeout (Python 2.5.2+) - Prevent hanging on subprocess I/O
- 🟡 **tarfile** Better GNU tar compatibility - Improved archive handling
- 🟢 **webbrowser** Additional browser support - More web browsers recognized

### Minor Enhancements

- 🟢 **cPickle** Improved performance - Faster pickling for common types
- 🟢 **csv** Better Unicode support - Improved CSV file handling
- 🟢 **doctest** SKIP option for skipping examples - Conditional test execution
- 🟢 **locale** format_string() function - Locale-aware string formatting
- 🟢 **logging** Additional logging configuration options - More flexible logging setup
- 🟢 **re** Improved Unicode regular expressions - Better Unicode character class support
