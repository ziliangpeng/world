# Python 3.4 Release Summary

**Released:** March 16, 2014
**EOL:** March 2019 (reached)
**Source:** [Official Python 3.4 Release Notes](https://docs.python.org/3.4/whatsnew/3.4.html)

## Overview

Python 3.4 was a major quality-of-life release that brought several foundational improvements without adding new syntax features. The release introduced key infrastructure modules including asyncio for asynchronous I/O, pathlib for object-oriented filesystem paths, and enum for enumeration types. It also made pip available by default in Python installations, addressing one of the major pain points in Python package management. Security and stability improvements were significant, with better file descriptor handling and enhanced SSL/TLS support.

## Major New Modules

### asyncio - Asynchronous I/O Framework

The new `asyncio` module (PEP 3156) provides a standard pluggable event loop model for Python, bringing solid asynchronous IO support to the standard library. This was marked as a provisional API in 3.4:

```python
import asyncio

async def hello():
    print('Hello')
    await asyncio.sleep(1)
    print('World')

asyncio.run(hello())
```

The module includes event loops, coroutines, tasks, futures, and synchronization primitives. It enables writing concurrent code using the async/await syntax (though that syntax itself came later in 3.5). The asyncio module standardizes asynchronous programming patterns and makes it easier for different event loop implementations to interoperate.

### pathlib - Object-Oriented Filesystem Paths

The pathlib module (PEP 428) offers classes representing filesystem paths with semantics appropriate for different operating systems:

```python
from pathlib import Path

p = Path('/')
for child in p.iterdir():
    if child.is_dir():
        print(child)

# Path operations
config = Path.home() / 'config' / 'settings.ini'
if config.exists():
    content = config.read_text()
```

Path classes are divided between pure paths (computational operations without I/O) and concrete paths (inherit from pure paths but also provide I/O operations). This was marked as a provisional API in 3.4.

### enum - Enumeration Types

The enum module (PEP 435) provides standard enumeration types:

```python
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(Color.RED)  # Color.RED
print(Color.RED.name)  # 'RED'
print(Color.RED.value)  # 1
```

This allows modules to provide more informative error messages and better debugging support by replacing opaque integer constants with backwards compatible enumeration values.

### statistics - Basic Statistics Library

The statistics module (PEP 450) provides numerically stable statistics functions:

```python
import statistics

data = [1.5, 2.5, 3.5, 4.5, 5.5]
print(statistics.mean(data))  # 3.5
print(statistics.median(data))  # 3.5
print(statistics.stdev(data))  # 1.58...
```

Functions include mean, median, mode, variance, and standard deviation calculations.

### tracemalloc - Memory Allocation Tracing

The tracemalloc module (PEP 454) traces memory blocks allocated by Python:

```python
import tracemalloc

tracemalloc.start()

# ... run your code ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

This debug tool can trace where objects were allocated, provide statistics on allocated memory blocks, and compute differences between snapshots to detect memory leaks.

### selectors - High-Level I/O Multiplexing

The selectors module provides high-level and efficient I/O multiplexing built on select module primitives. It's part of the infrastructure supporting asyncio (PEP 3156).

## PIP Bootstrapping by Default

### PEP 453: Explicit Bootstrapping of PIP

One of the most significant changes in Python 3.4 was making pip available by default through the new `ensurepip` module:

- The `pip` command (pipX and pipX.Y versions) is now installed by default on all platforms
- Virtual environments created with `venv` include pip by default
- CPython installers on Windows and macOS now install pip by default
- The bundled pip version (1.5.4 in 3.4.0) gets updated in maintenance releases

This addressed a major pain point where new Python users struggled to install packages because pip wasn't included. The ensurepip module contains a bundled copy of pip and can bootstrap it without internet access.

## Single-Dispatch Generic Functions

### PEP 443: functools.singledispatch

The new `singledispatch` decorator enables function overloading based on the type of the first argument:

```python
from functools import singledispatch

@singledispatch
def process(data):
    raise NotImplementedError("Unsupported type")

@process.register(str)
def _(data):
    return data.upper()

@process.register(list)
def _(data):
    return [item.upper() for item in data]

print(process("hello"))  # "HELLO"
print(process(["a", "b"]))  # ["A", "B"]
```

This brings generic function support to Python, focusing on grouping multiple implementations of an operation that works with different kinds of data.

## Security Improvements

### PEP 446: Non-Inheritable File Descriptors

Newly created file descriptors are now non-inheritable by default. This prevents file descriptors from being accidentally inherited by child processes, which could lead to bugs and security issues. New functions support explicit inheritance control:

- `os.get_inheritable()`, `os.set_inheritable()`
- `os.get_handle_inheritable()`, `os.set_handle_inheritable()`
- `socket.socket.get_inheritable()`, `socket.socket.set_inheritable()`

### PEP 456: Secure Hash Algorithm

A new secure and interchangeable hash algorithm protects against hash collision denial-of-service attacks while maintaining performance.

### Enhanced SSL/TLS Support

- TLSv1.1 and TLSv1.2 support
- Server-side SNI (Server Name Indication) support
- Certificate retrieval from Windows system cert store
- SSLContext improvements
- All standard library modules with SSL support now support server certificate verification, including hostname matching and CRLs

### PBKDF2 Support

New `hashlib.pbkdf2_hmac()` function provides PKCS#5 password-based key derivation function 2 for secure password hashing.

### Isolated Mode

New `-I` command line option enables isolated mode (PEP 432), which:
- Doesn't add user site directory to sys.path
- Doesn't import site
- Doesn't set environment variables

This provides better security for running untrusted code.

## Language and Interpreter Improvements

### PEP 442: Safe Object Finalization

Finalizers can now be called even when reference cycles are involved, eliminating one of the major limitations of Python's garbage collector. As a result, module globals are no longer set to None during finalization in most cases, fixing many shutdown bugs.

### PEP 445: Configurable Memory Allocators

CPython now allows customizing memory allocators, useful for debugging memory issues and integrating with external memory profilers.

### PEP 451: ModuleSpec for Import System

A new ModuleSpec type encapsulates information about a module that the import machinery uses to load it. This simplifies the import implementation and makes it easier for importer authors.

### PEP 436: Argument Clinic

A new preprocessor for converting function parameter declarations into C code that parses those arguments. This improves performance and reduces boilerplate in CPython's C code.

## Improved Modules

### functools

- `singledispatch()`: Single-dispatch generic functions (PEP 443)
- `partialmethod()`: Partial argument application for descriptors

### contextlib

- `suppress()`: Context manager for suppressing exceptions
- `redirect_stdout()`: Context manager for redirecting stdout

### pickle

New protocol 4 (PEP 3154) provides:
- Support for very large objects
- More efficient pickling of small objects
- Better support for nested classes

### email

New `email.contentmanager` submodule and `EmailMessage` class simplify MIME handling:

```python
from email.message import EmailMessage

msg = EmailMessage()
msg['Subject'] = 'Hello'
msg['From'] = 'sender@example.com'
msg['To'] = 'recipient@example.com'
msg.set_content('This is the message body')
```

### dis

The disassembler module was rebuilt around an `Instruction` class providing object-oriented access to bytecode:

```python
import dis

def example():
    x = 1
    return x + 1

for instr in dis.get_instructions(example):
    print(f'{instr.opname}: {instr.argval}')
```

### multiprocessing

- New spawn and forkserver start methods avoid using os.fork on Unix, improving security
- Child processes on Windows no longer inherit all parent handles

### inspect and pydoc

Substantially improved introspection of callable objects, enhancing the `help()` system. Better support for custom `__dir__` methods and dynamic class attributes through metaclasses.

## Standard Library Improvements

### New Functions and Methods

- `min()` and `max()` accept a `default` keyword argument for empty iterables
- `glob.escape()`: Escape special characters in filenames for literal matching
- `html.unescape()`: Convert HTML5 character references to Unicode
- `base64.a85encode()`, `a85decode()`, `b85encode()`, `b85decode()`: Ascii85 and Base85 encoding
- `gc.get_stats()`: Get garbage collection statistics
- `abc.get_cache_token()`: Invalidate ABC caches when object graph changes

### Context Manager Support

Many file-like objects now support context managers:
- `aifc.open()`
- `dbm.open()`
- Wave file objects
- And more

### Better Bytes-Like Object Support

Many functions that previously required `bytes` or `bytearray` now accept any bytes-like object:
- `bytes.join()` and `bytearray.join()`
- Base64 encoding/decoding functions
- audioop functions

## Codec Handling Improvements

Better documentation and error messages for the codecs system:

- `codecs.encode()` and `codecs.decode()` are now properly documented
- Better error messages direct users to appropriate functions for non-text encodings:

```python
>>> "hello".encode("rot13")
LookupError: 'rot13' is not a text encoding; use codecs.encode() to handle arbitrary codecs
```

Chained exceptions now show which codec caused the error.

## Other Language Changes

- Unicode database updated to UCD 6.3
- Module objects are now weakly referenceable
- Module `__file__` attributes now contain absolute paths by default
- UTF-* codecs reject surrogates during encoding/decoding (except with surrogatepass handler)
- New German EBCDIC codec `cp273` and Ukrainian codec `cp1125`
- `int()` constructor accepts objects with `__index__` for the base argument
- Frame objects have a `clear()` method to clear local variable references
- `memoryview` registered as a Sequence and supports `reversed()`

## Marshal Format Improvements

The marshal format (used for .pyc files) was made more compact and efficient, reducing the size of compiled bytecode files.

## Performance Improvements

While Python 3.4 didn't have dramatic performance improvements, various optimizations were made throughout:
- More efficient .pyc format
- Various bytecode optimizations
- Improved memory usage patterns

## Deprecations and Removals

### Deprecated

- `imp` module deprecated in favor of `importlib`
- `formatter` module deprecated
- Various deprecated APIs in standard library modules

### Removed

- Several long-standing deprecated features were removed

## CPython Implementation Details

### PEP 442: Safe Object Finalization

Eliminated the problems with finalizers in reference cycles, allowing finalizers to be called reliably even when objects are part of cycles.

### PEP 445: Configurable Memory Allocators

Allows customization of memory allocation for debugging and profiling purposes.

### PEP 436: Argument Clinic

Preprocessor for converting function declarations to optimized C argument parsing code. Initially used in a subset of the standard library, with plans to expand usage.

## Migration Notes

### Breaking Changes

Most Python 3.3 code runs unchanged on 3.4. Notable changes:
- File descriptors are non-inheritable by default (may affect multiprocessing code)
- Some deprecated features removed
- Stricter surrogate handling in UTF codecs

### Recommended Updates

- Start using `pathlib` for filesystem path operations
- Consider using `asyncio` for asynchronous code
- Use `enum` for enumeration types instead of integer constants
- Replace direct `imp` usage with `importlib`

## Key Takeaways

1. **pip now included by default** - Major improvement for new Python users and package installation
2. **asyncio standardizes async I/O** - Foundation for modern asynchronous Python programming
3. **pathlib provides modern path handling** - Object-oriented alternative to os.path
4. **enum brings proper enumeration support** - Better than using raw integers or strings
5. **Security improvements** - Non-inheritable file descriptors, better SSL/TLS, hash improvements
6. **Safe object finalization** - Eliminates major garbage collection limitation
7. **statistics module** - Built-in support for common statistical calculations
8. **tracemalloc for debugging** - Track memory allocations and find leaks
9. **Better introspection** - Improved inspect and pydoc for better help() output
10. **No new syntax** - Focus on library improvements and quality of life

Python 3.4 was a foundational release that set the stage for modern Python development, particularly in asynchronous programming with asyncio and improved developer experience with pip bundling and better tooling.
