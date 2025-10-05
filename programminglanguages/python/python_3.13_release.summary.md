# Python 3.13 Release Summary

**Released:** October 7, 2024
**Source:** [Official Python 3.13 Release Notes](https://docs.python.org/3/whatsnew/3.13.html)

## Overview

Python 3.13 represents a major step forward with groundbreaking features and experimental capabilities. Headline features include experimental free-threaded CPython (PEP 703) that can run without the GIL, an experimental JIT compiler (PEP 744), a dramatically improved interactive REPL, and important type system enhancements. The release also includes defined semantics for `locals()`, mobile platform support (iOS and Android), continued error message improvements with color support, and removal of long-deprecated "dead batteries" from the standard library.

## Major Language Features

### PEP 703: Free-Threaded CPython (Experimental)

CPython now has experimental support for running without the Global Interpreter Lock (GIL):

- Available in special builds (`python3.13t` or `python3.13t.exe`)
- Allows true parallel execution of Python threads on multi-core systems
- Can be disabled at runtime via `PYTHON_GIL` environment variable or `-X gil=1`
- C extensions need explicit support via `Py_mod_gil` slot
- **Currently experimental** with performance tradeoffs: single-threaded code runs slower, but multi-threaded workloads can achieve parallelism

```python
# Check if free-threading is available
import sys
print("experimental free-threading build" in sys.version)

# Check if GIL is actually disabled
print(sys._is_gil_enabled())  # False in free-threaded mode
```

This opens the door to true multi-core Python without needing multiprocessing workarounds.

### PEP 744: Experimental JIT Compiler

A basic Just-In-Time compiler has been added:

- Build-time option: `--enable-experimental-jit`
- Uses copy-and-patch technique with LLVM as build dependency
- Tier 2 optimizer pipeline with micro-ops (uops)
- Currently modest performance gains (1-5%)
- Can be toggled at runtime: `PYTHON_JIT=0` or `PYTHON_JIT=1`
- Foundation for future performance improvements

The JIT translates hot Tier 1 bytecode to Tier 2 IR, optimizes it, then generates machine code.

### Better Interactive Interpreter

Python now includes a modern REPL based on PyPy:

- **Multiline editing** with proper history preservation
- **Direct commands**: Type `help`, `exit`, `quit` without parentheses
- **Color support** enabled by default for prompts and tracebacks
- **F1**: Interactive help browser
- **F2**: History browsing (skips output and prompts)
- **F3**: Paste mode for multi-line code blocks
- Disable with `PYTHON_BASIC_REPL` environment variable

This dramatically improves the interactive Python experience.

### PEP 667: Defined Semantics for `locals()`

The behavior of mutating `locals()` is now standardized:

**In optimized scopes** (functions, generators, comprehensions):
- `locals()` returns an **independent snapshot** of local variables
- Changes to the dict don't affect actual local variables
- `FrameType.f_locals` returns a **write-through proxy** for debuggers

**In module/class scopes**:
- Maintains previous behavior (modifications can affect the namespace)

```python
def example():
    x = 1
    loc = locals()
    loc['x'] = 2
    print(x)  # Still prints 1 (snapshot behavior)

# For debuggers:
import sys
frame = sys._getframe()
frame.f_locals['x'] = 2  # Works via write-through proxy
```

This fixes long-standing inconsistencies and enables better debugging tools.

## Type Hint Enhancements

### PEP 696: Type Parameter Defaults

Type parameters now support default values:

```python
from typing import TypeVar

T = TypeVar('T', default=int)

class Box[T = int]:
    def __init__(self, value: T):
        self.value = value

# Both valid:
box1: Box = Box(42)        # T defaults to int
box2: Box[str] = Box("hi") # T explicitly str
```

Applies to `TypeVar`, `ParamSpec`, and `TypeVarTuple`.

### PEP 702: Deprecation Decorator

New decorator for marking deprecations:

```python
from warnings import deprecated

@deprecated("Use new_function() instead")
def old_function():
    pass

@deprecated("Removed in v2.0", category=DeprecationWarning)
class OldClass:
    pass
```

- Visible to static type checkers
- Emits `DeprecationWarning` at runtime
- Standardized way to mark deprecated APIs

### PEP 705: ReadOnly TypedDict Items

Mark TypedDict fields as read-only:

```python
from typing import TypedDict, ReadOnly

class User(TypedDict):
    name: str
    id: ReadOnly[int]  # Cannot be modified after creation

user: User = {"name": "Alice", "id": 1}
user["name"] = "Bob"  # OK
user["id"] = 2        # Type checker error
```

### PEP 742: TypeIs for Better Type Narrowing

Alternative to `TypeGuard` with more intuitive narrowing:

```python
from typing import TypeIs

def is_string(val: str | int) -> TypeIs[str]:
    return isinstance(val, str)

def process(val: str | int):
    if is_string(val):
        # Type checker knows val is str here
        print(val.upper())
    else:
        # Type checker knows val is int here
        print(val + 1)
```

`TypeIs` provides more predictable narrowing than `TypeGuard`.

## Improved Error Messages

Color support and better diagnostics:

```python
# Color tracebacks by default (PYTHON_COLORS controls this)

# Script name collision detection:
$ python random.py
AttributeError: module 'random' has no attribute 'randint'
(consider renaming '/home/me/random.py' since it has the same
name as the standard library module named 'random' and prevents
importing that standard library module)

# Keyword argument suggestions:
>>> "text".split(max_split=1)
TypeError: split() got an unexpected keyword argument 'max_split'.
Did you mean 'maxsplit'?
```

## Platform Support

### PEP 730 & PEP 738: Mobile Platform Support

New tier 3 platform support:

- **iOS**: `arm64-apple-ios`, `arm64-apple-ios-simulator`
- **Android**: `aarch64-linux-android`, `x86_64-linux-android`
- Full Python standard library support
- Opens Python development on mobile devices

Also:
- `wasm32-wasi` promoted to tier 2
- `wasm32-emscripten` removed from official support

## Standard Library Improvements

### copy: New `replace()` Function

Create modified copies of immutable objects:

```python
from copy import replace
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: int
    y: int

p1 = Point(1, 2)
p2 = replace(p1, x=5)  # Point(5, 2)
```

Works with: `dataclass`, `namedtuple`, `datetime`, `SimpleNamespace`, code objects, and any class with `__replace__()`.

### dbm.sqlite3: New Default Backend

SQLite-based dbm backend is now the default:

```python
import dbm

# Automatically uses SQLite backend
with dbm.open('data.db', 'c') as db:
    db['key'] = 'value'
```

More reliable and portable than older dbm implementations.

### os: Timer File Descriptors (Linux)

Low-level interface to Linux timerfd:

```python
import os

# Create a timer that fires every second
tfd = os.timerfd_create(os.CLOCK_MONOTONIC)
os.timerfd_settime(tfd, 0, 1.0, 1.0)  # interval=1s, initial=1s

# Use with select/poll/epoll
```

### os: Process CPU Count

New function for actual available CPUs:

```python
import os

os.cpu_count()          # Total logical CPUs
os.process_cpu_count()  # CPUs available to this process

# Respects cgroup limits, CPU affinity
# Override with PYTHON_CPU_COUNT env var
```

Used by `concurrent.futures`, `multiprocessing`, and `compileall` for thread/process defaults.

### pathlib: Enhanced Functionality

```python
from pathlib import Path

# Create from file:// URI
p = Path.from_uri('file:///home/user/file.txt')

# Match with recursive wildcards
p.full_match('**/*.py')

# New parser attribute for path implementation
Path.parser  # posixpath or ntpath

# Follow symlinks options
p.is_file(follow_symlinks=False)
p.is_dir(follow_symlinks=True)
```

### asyncio: Queue Shutdown

Graceful queue termination:

```python
import asyncio

async def worker(queue):
    while True:
        try:
            item = await queue.get()
            process(item)
        except asyncio.QueueShutDown:
            break

queue = asyncio.Queue()
# ... add items ...
queue.shutdown()  # Unblocks all waiters
```

### base64: Z85 Encoding

```python
import base64

data = b"Hello"
encoded = base64.z85encode(data)
decoded = base64.z85decode(encoded)
```

### argparse: Deprecation Support

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--old', deprecated=True,
                   help='Use --new instead')
```

### random: Command-Line Interface

```python
# Terminal
$ python -m random
$ python -m random 1 10        # Random int between 1 and 10
$ python -m random --choice a b c
```

### statistics: Kernel Density Estimation

```python
from statistics import kde, kde_random

# Estimate continuous probability density from samples
samples = [1.2, 1.5, 2.1, 2.3, 2.8, 3.1, 3.5]
kernel = kde(samples, h=0.5)  # h is bandwidth

# Sample from estimated distribution
new_sample = kde_random(kernel)
```

### math: Fused Multiply-Add

```python
import math

# x * y + z with single rounding (no intermediate precision loss)
result = math.fma(2.0, 3.0, 1.0)  # 7.0

# Follows IEEE 754 fusedMultiplyAdd
```

### warnings.deprecated: Standardized Deprecations

```python
from warnings import deprecated

@deprecated("Use new_api() instead")
def old_api():
    pass

@deprecated("Removed in 2.0", category=FutureWarning)
class OldClass:
    pass
```

## Performance Optimizations

### Import Time Improvements

Significant speedups for standard library imports:

- **typing**: ~33% faster (removed `re` and `contextlib` dependencies)
- **email.utils**, **enum**, **functools**, **importlib.metadata**, **threading**: All faster

### textwrap.indent

~30% faster for large inputs.

### subprocess: More posix_spawn Usage

`subprocess` now uses `posix_spawn()` more extensively:

- Used when `close_fds=True` on Linux/FreeBSD/Solaris with modern libc
- Significant performance improvement for process spawning
- Fallback control: `subprocess._USE_POSIX_SPAWN = False`

### time Module (Windows)

Higher resolution clocks:

- `time.monotonic()`: Now uses `QueryPerformanceCounter()` (1μs vs 15.6ms)
- `time.time()`: Now uses `GetSystemTimePreciseAsFileTime()` (1μs vs 15.6ms)

## Security Improvements

### ssl: Stricter Default Verification

`ssl.create_default_context()` now enables:

- `VERIFY_X509_PARTIAL_CHAIN`: Accept partial certificate chains
- `VERIFY_X509_STRICT`: Reject malformed certificates

This may reject pre-RFC 5280 certificates. Disable if needed:

```python
import ssl
ctx = ssl.create_default_context()
ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
```

### email: Safer Header Handling

- Headers with embedded newlines are now quoted
- `getaddresses()` and `parseaddr()` reject malformed addresses by default
- New `strict` parameter for backward compatibility (CVE-2023-27043 fix)

### os.mkdir: Windows Permission Fix

`os.mkdir()` with `mode=0o700` on Windows now actually restricts access (CVE-2024-4030 mitigation).

## Important Removals

### PEP 594: Dead Batteries Removed

19 legacy modules removed:

- **Audio/Media**: `aifc`, `audioop`, `chunk`, `sndhdr`, `sunau`
- **Internet**: `cgi`, `cgitb`, `nntplib`, `telnetlib`
- **Unix**: `crypt`, `nis`, `ossaudiodev`, `pipes`, `spwd`
- **Other**: `imghdr`, `mailcap`, `msilib`, `uu`, `xdrlib`

See documentation for PyPI replacements.

### Other Removals

- **2to3**: Tool and `lib2to3` module removed
- **tkinter.tix**: Module removed
- **typing.io**, **typing.re**: Namespaces removed
- **locale.resetlocale()**: Function removed
- Chained `classmethod` descriptors: No longer supported

## CPython Implementation Changes

### Bytecode Changes

Significant internal restructuring:

- Removed cache entries as separate instructions (now part of `Instruction.cache_info`)
- Changed disassembly output to show logical labels instead of offsets
- New bytecode optimizations for Tier 2 IR

### Compiler Improvements

- Docstrings now have common leading whitespace stripped (~5% bytecode size reduction)
- Annotation scopes can contain lambdas and comprehensions
- New `__static_attributes__` on classes
- New `__firstlineno__` on classes

### Data Model Additions

```python
class Example:
    def __init__(self):
        self.x = 1
        self.y = 2

# New attributes:
Example.__static_attributes__  # ('x', 'y')
Example.__firstlineno__        # Line number of class definition
```

## C API Changes

### Free-Threading Support

- `Py_mod_gil` slot indicates GIL support
- `PyUnstable_Module_SetGIL()` for single-phase init modules
- Extensions without support will re-enable GIL

### PyTime API

New time-related functions:

- Access to system clocks from C
- Consistent time handling across platforms

### PyMutex

Lightweight mutex (single byte):

```c
PyMutex mutex = {0};
PyMutex_Lock(&mutex);
// critical section
PyMutex_Unlock(&mutex);
```

### Monitoring API

New functions for generating PEP 669 monitoring events from C.

## Migration Notes

### Breaking Changes

1. **PEP 594**: 19 modules removed - check PyPI for replacements
2. **locals() semantics**: Mutations in functions no longer affect locals
3. **2to3 removed**: Use other modernization tools
4. **Chained classmethod**: No longer supports descriptor chaining
5. **Stricter SSL verification**: May reject older certificates
6. **C extensions**: May need updates for free-threading

### Compatibility

Most Python 3.12 code will work in 3.13, except:

- Code depending on removed modules
- Code relying on `locals()` mutation in functions
- C extensions without free-threading support (will re-enable GIL)
- Code assuming specific bytecode structure

### Recommended Actions

1. Test with free-threaded build if you use threading
2. Update deprecated APIs before they're removed
3. Use `warnings.deprecated` for your own deprecations
4. Consider mobile platform support for new projects
5. Try the improved REPL for development

## Key Takeaways

1. **Free-threading is revolutionary** - first step toward GIL-free Python for true parallelism
2. **JIT compiler shows promise** - modest gains now, foundation for future performance
3. **Developer experience improved** - better REPL, colored output, enhanced error messages
4. **Type system keeps growing** - defaults, deprecations, readonly, better narrowing
5. **Mobile platforms now supported** - iOS and Android are official tier 3 platforms
6. **Standard library modernized** - dead batteries removed, useful additions like `copy.replace()`
7. **Better semantics** - `locals()` behavior now well-defined and consistent
8. **Performance gains** - faster imports, better subprocess, higher resolution timers
9. **Security hardened** - stricter SSL, email parsing fixes, Windows permissions

Python 3.13 is an ambitious release that lays critical groundwork for Python's future. The free-threaded mode and JIT compiler are experimental but represent fundamental shifts in Python's capabilities. Combined with practical improvements to the type system, standard library, and developer experience, Python 3.13 positions the language for the next decade of growth.
