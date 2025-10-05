# Python 3.3 Release Summary

**Released:** September 29, 2012
**Source:** [Official Python 3.3 Release Notes](https://docs.python.org/3.3/whatsnew/3.3.html)

## Overview

Python 3.3 represents a significant milestone in Python's evolution, focusing on improving Unicode handling, introducing virtual environments to the standard library, and making the language more accessible for migration from Python 2. Major changes include the yield from syntax for generator delegation, a completely rewritten import system based on importlib, flexible string representation for memory efficiency, namespace packages without __init__.py files, and a reorganized exception hierarchy that makes error handling more intuitive.

## Major Language Features

### PEP 380: yield from Expression

New syntax for delegating generator operations to subgenerators:

```python
def g(x):
    yield from range(x, 0, -1)
    yield from range(x)

list(g(5))  # [5, 4, 3, 2, 1, 0, 1, 2, 3, 4]
```

For complex generators, yield from allows subgenerators to receive sent and thrown values directly:

```python
def accumulate():
    tally = 0
    while 1:
        next = yield
        if next is None:
            return tally
        tally += next

def gather_tallies(tallies):
    while 1:
        tally = yield from accumulate()
        tallies.append(tally)
```

This enables splitting complex generators into manageable subgenerators, similar to how functions can be split into subfunctions.

### PEP 420: Implicit Namespace Packages

Package directories no longer require __init__.py marker files. Namespace packages can automatically span multiple path segments, making it easier to organize large projects and extend packages across different locations.

### PEP 414: Explicit Unicode Literals

The u"unicode" prefix is accepted again for string objects to ease migration from Python 2. This prefix has no semantic significance in Python 3 but reduces mechanical changes needed when porting code.

### PEP 409: Suppressing Exception Context

New syntax to suppress chained exception context with `raise ... from None`:

```python
class D:
    def __init__(self, extra):
        self._extra_attributes = extra
    def __getattr__(self, attr):
        try:
            return self._extra_attributes[attr]
        except KeyError:
            raise AttributeError(attr) from None
```

This provides cleaner error messages in applications that convert between exception types.

## Exception Hierarchy Improvements

### PEP 3151: Reworked OS and IO Exception Hierarchy

The exception hierarchy is now both simplified and more fine-grained. All OS-related exceptions (OSError, IOError, EnvironmentError, WindowsError, socket.error, etc.) are unified under OSError, with the old names kept as aliases.

New specific exception subclasses make error handling more intuitive:

```python
# Old way (Python 3.2)
from errno import ENOENT, EACCES, EPERM
try:
    with open("document.txt") as f:
        content = f.read()
except IOError as err:
    if err.errno == ENOENT:
        print("document.txt file is missing")
    elif err.errno in (EACCES, EPERM):
        print("You are not allowed to read document.txt")

# New way (Python 3.3)
try:
    with open("document.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("document.txt file is missing")
except PermissionError:
    print("You are not allowed to read document.txt")
```

New exception types include: BlockingIOError, ChildProcessError, ConnectionError (with subclasses BrokenPipeError, ConnectionAbortedError, ConnectionRefusedError, ConnectionResetError), FileExistsError, FileNotFoundError, InterruptedError, IsADirectoryError, NotADirectoryError, PermissionError, ProcessLookupError, and TimeoutError.

## Unicode and String Improvements

### PEP 393: Flexible String Representation

Unicode strings now use variable-width internal representation based on the widest character:
- Pure ASCII and Latin1 strings (U+0000-U+00FF): 1 byte per code point
- BMP strings (U+0000-U+FFFF): 2 bytes per code point
- Non-BMP strings (U+10000-U+10FFFF): 4 bytes per code point

This results in 2-3x memory reduction for most applications compared to Python 3.2 wide builds. Python now always supports the full Unicode range (U+0000 to U+10FFFF), eliminating the narrow/wide build distinction. All functions now correctly handle non-BMP characters, and sys.maxunicode is always 1114111.

## Virtual Environments

### PEP 405: Virtual Environments

The new venv module brings virtual environment functionality into the standard library, inspired by the popular virtualenv package. Virtual environments create separate Python setups while sharing a system-wide base install, with their own private site packages.

Command-line tool pyvenv and programmatic access via the venv module are now included. The Python interpreter checks for a pyvenv.cfg file to detect virtual environment directory trees.

## Import System Rewrite

### Using importlib as the Implementation of Import

The __import__() function is now powered by importlib.__import__(), completing "phase 2" of PEP 302. This exposes the machinery powering import, provides a single implementation for all Python VMs, and eases future maintenance.

Key changes:
- All modules now have a __loader__ attribute
- Loaders set the __package__ attribute
- sys.meta_path and sys.path_hooks now store all meta path finders and path entry hooks
- New abstract base classes: importlib.abc.MetaPathFinder and importlib.abc.PathEntryFinder
- Exposed classes: FileFinder, SourceFileLoader, SourcelessFileLoader, ExtensionFileLoader
- ImportError now has name and path attributes
- Per-module import lock replaces the global import lock, preventing deadlocks

## Interpreter and Runtime Improvements

### PEP 412: Key-Sharing Dictionary

Object attribute dictionaries now share internal storage for keys and hashes between instances. This significantly reduces memory consumption for programs creating many instances of non-builtin types.

### PEP 362: Function Signature Object

New inspect.signature() function provides easy introspection of Python callables. New classes inspect.Signature, inspect.Parameter, and inspect.BoundArguments hold information about call signatures, annotations, default values, and parameters, simplifying decorator writing and signature validation.

### PEP 3155: Qualified Names

Functions and class objects have a new __qualname__ attribute representing the path from module top-level to their definition:

```python
class C:
    class D:
        def meth(self):
            pass

C.D.meth.__qualname__  # 'C.D.meth'
```

### PEP 421: sys.implementation

New sys.implementation attribute exposes implementation-specific details (name, version, hexversion, cache_tag). This consolidates implementation-specific data in one namespace, making the standard library more portable across different Python implementations. Also introduces types.SimpleNamespace for attribute-based writable namespaces.

### A Finer-Grained Import Lock

Python 3.3 replaces the global import lock with per-module locks. This correctly serializes importation of a given module from multiple threads while eliminating deadlocks when importing triggers code execution in different threads.

## Standard Library Improvements

### New Modules

**faulthandler** - Debug module for dumping Python tracebacks on crashes, timeouts, or user signals. Enable with faulthandler.enable() or PYTHONFAULTHANDLER environment variable.

**ipaddress** - Tools for creating and manipulating IPv4 and IPv6 addresses, networks, and interfaces (PEP 3144).

**lzma** - Data compression using the LZMA algorithm, with support for .xz and .lzma file formats.

**unittest.mock** - Mock object library for replacing parts of systems under test (previously external package).

**venv** - Python virtual environments built into the standard library.

### Improved Modules

**decimal** - New C accelerator module provides dramatic performance improvements.

**email** - Better Unicode handling (provisional).

**bz2** - Complete rewrite with new bz2.open() function, support for arbitrary file-like objects, multi-stream decompression, and full io.BufferedIOBase API implementation.

**collections** - New ChainMap class for treating multiple mappings as a single unit. Abstract base classes moved to collections.abc module. Counter supports unary +/- operators and in-place operators.

**contextlib** - New ExitStack for programmatic manipulation of context managers, replacing the deprecated contextlib.nested API.

**PEP 3118: memoryview** - Completely rewritten with comprehensive fixes for ownership and lifetime issues. Supports all native single character format specifiers, casting, multi-dimensional operations, and arbitrary slicing.

**array** - Now supports long long type using 'q' and 'Q' type codes.

**os** - open() gets new opener parameter and 'x' exclusive creation mode.

**hash** - Hash randomization enabled by default for security.

**str** - New casefold() method for case-insensitive string matching.

**list and bytearray** - New copy() and clear() methods.

## Windows Improvements

### PEP 397: Python Launcher for Windows

Windows installer includes a new py launcher application for version-independent Python launching. Supports Unix-style shebang lines in scripts and command-line version selection (py -3, py -2.6). Installer now includes option to add Python to system PATH.

### Codec Improvements

The mbcs codec has been rewritten to handle replace and ignore error handlers correctly on all Windows versions. New cp65001 codec added (Windows UTF-8, CP_UTF8).

## Performance Optimizations

- Flexible string representation (PEP 393): 2-3x memory reduction for Unicode strings in most applications
- Key-sharing dictionaries (PEP 412): Reduced memory for object attributes
- dict.setdefault() now does only one lookup, making it atomic with built-in types
- Improved C accelerator for decimal module

## Security Improvements

Hash randomization is now enabled by default to protect against hash collision denial-of-service attacks. Controlled via PYTHONHASHSEED environment variable.

## Deprecations and Removals

- abc.abstractproperty, abc.abstractclassmethod, abc.abstractstaticmethod deprecated (use property/classmethod/staticmethod with abc.abstractmethod())
- unicode_internal codec deprecated
- contextlib.nested removed (use contextlib.ExitStack)
- configure flag --with-wide-unicode removed (Python now always behaves like wide build)

## Migration Notes

### Breaking Changes

Most code should work unchanged, but be aware of:
- Import system changes may affect code that manipulates import machinery
- sys.maxunicode is always 1114111 (was configurable in 3.2)
- memoryview API changes for advanced use cases
- Per-module import locks may expose previously hidden threading issues

### Compatibility

Python 3.3 continues the Python 3.x migration path while making it easier to port Python 2 code. The reintroduction of u"unicode" literals and improved exception hierarchy reduce porting friction.

## Key Takeaways

1. **yield from simplifies generator composition** - Makes it easy to delegate to subgenerators
2. **Virtual environments in stdlib** - No need for external virtualenv package
3. **Memory-efficient Unicode** - Flexible string representation saves 2-3x memory
4. **Better exception handling** - Fine-grained exception hierarchy eliminates errno checking
5. **Namespace packages** - No __init__.py required
6. **Import system modernized** - Fully exposed through importlib
7. **Python 2 migration support** - u"unicode" literals accepted again
8. **Windows improvements** - New launcher and PATH integration
9. **More robust threading** - Per-module import locks prevent deadlocks

Python 3.3 represents a maturation of Python 3, addressing performance concerns, improving developer experience, and making the language more accessible for teams migrating from Python 2. The foundation laid by the import system rewrite and flexible string representation enables future enhancements while the addition of virtual environments to the standard library simplifies Python environment management.
