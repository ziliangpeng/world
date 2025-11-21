# Python 3.14 Release Summary

**Released:** October 7, 2025
**Source:** [Official Python 3.14 Release Notes](https://docs.python.org/3/whatsnew/3.14.html)

## Overview

Python 3.14 represents a transformative release with fundamental changes to annotation handling, true multi-core parallelism through subinterpreters, and powerful new string processing capabilities. Headline features include deferred annotation evaluation (PEP 649/749) that eliminates forward reference string quotes and improves performance, template string literals (PEP 750) for safe string processing, multiple interpreters in the standard library (PEP 734) enabling true parallelism, and a safe external debugger interface (PEP 768). The release also includes incremental garbage collection with dramatically reduced pause times, Zstandard compression support throughout the standard library, extensive pathlib enhancements, a tail-call interpreter with 3-5% performance gains, and continued improvements to free-threaded mode reducing single-threaded overhead to just 5-10%.

## Major Language Features

### PEP 649 & 749: Deferred Annotation Evaluation

Python no longer eagerly evaluates annotations on functions, classes, and modules. Instead, annotations are stored in special `__annotate__` functions and evaluated only when needed:

**Key benefits:**
- Significant performance improvement for code with many annotations
- Forward references no longer require string quotes
- Controlled evaluation through new `annotationlib` module

```python
from annotationlib import get_annotations, Format

# Forward references work without quotes
def process(data: DataFrame) -> ResultSet:
    pass

# Multiple evaluation formats
get_annotations(process, format=Format.VALUE)      # Evaluates to actual types
get_annotations(process, format=Format.FORWARDREF) # Returns ForwardRef objects
get_annotations(process, format=Format.STRING)     # Returns string representations

# Handles undefined names gracefully
def func(arg: UndefinedType):
    pass

get_annotations(func, format=Format.VALUE)      # Raises NameError
get_annotations(func, format=Format.FORWARDREF) # Returns ForwardRef('UndefinedType')
get_annotations(func, format=Format.STRING)     # Returns 'UndefinedType'
```

The `__annotate__` function is stored as bytecode, improving import performance and reducing memory usage.

### PEP 750: Template String Literals

New t-string syntax enables custom string processing by providing access to both static text and interpolated parts:

```python
variety = 'Stilton'
template = t'Try some {variety} cheese!'

# Template is composed of parts
list(template)
# ['Try some ', Interpolation(expr='variety', value='Stilton', ...), ' cheese!']

# Safe SQL query building
def sql_query(template):
    parts = []
    params = []
    for part in template:
        if isinstance(part, str):
            parts.append(part)
        else:
            parts.append('?')
            params.append(part.value)
    return ''.join(parts), params

query, params = sql_query(t"SELECT * FROM users WHERE name = {username}")
# Safe parameterized query, prevents SQL injection

# HTML sanitization
attributes = {'src': 'image.jpg', 'alt': '<script>alert("xss")</script>'}
template = t'<img {attributes}>'
# Custom processor can safely escape the alt attribute value
```

T-strings return `Template` objects from the new `string.templatelib` module, enabling domain-specific string processing for SQL, shell commands, HTML, and more.

### PEP 734: Multiple Interpreters in Standard Library

The new `concurrent.interpreters` module enables running multiple isolated Python interpreters within a single process:

```python
from concurrent import interpreters

# Create isolated interpreter
interp = interpreters.create()

# Execute code in that interpreter
interp.exec('''
import sys
print(f"Running in interpreter {id(sys)}")
''')

# True multi-core parallelism without GIL
# Each interpreter has its own GIL and isolated memory
```

**Benefits:**
- True multi-core parallelism (each interpreter has its own GIL)
- Isolated memory spaces with opt-in sharing
- Lower resource overhead than multiprocessing
- Supports actor pattern and other concurrency models

**Current limitations:**
- Higher startup overhead than threads
- Increased memory usage per interpreter
- Limited third-party extension module compatibility

**New InterpreterPoolExecutor:**

```python
from concurrent.futures import InterpreterPoolExecutor

with InterpreterPoolExecutor(max_workers=4) as executor:
    # True parallel execution on multi-core systems
    results = executor.map(cpu_intensive_task, data)
```

This is a game-changer for CPU-bound Python workloads that previously required multiprocessing.

### PEP 768: Safe External Debugger Interface

Python now provides zero-overhead debugging capabilities through `sys.remote_exec()`:

```python
import sys

# Attach debugger to running process
sys.remote_exec(1234, '/path/to/debug_script.py')
```

**Security controls:**
- Environment variable: `PYTHON_DISABLE_REMOTE_DEBUG=1`
- Command-line flag: `-X disable-remote-debug`
- Build-time option: `--without-remote-debug`

**pdb remote debugging:**

```bash
# Attach pdb to running process
python -m pdb -p PID
```

This enables debugging production systems without requiring debug builds or prior instrumentation.

### PEP 758: Bracketless Exception Expressions

Multiple exception types can now be specified without parentheses when no `as` clause is present:

```python
# New syntax (no parentheses)
try:
    connect_to_server()
except TimeoutError, ConnectionRefusedError:
    print('Network error!')

# Still supported with parentheses
except (ValueError, TypeError):
    pass

# Parentheses required with 'as' clause
except (KeyError, AttributeError) as e:
    print(f'Error: {e}')
```

This provides a cleaner syntax for common exception handling patterns.

### PEP 765: Control Flow in Finally Blocks

New `SyntaxWarning` when `return`, `break`, or `continue` statements exit `finally` blocks:

```python
def problematic():
    try:
        return 1
    finally:
        return 2  # SyntaxWarning: 'return' statement in 'finally' block may swallow exceptions

# Warning helps catch bugs like:
for item in items:
    try:
        process(item)
    finally:
        break  # SyntaxWarning: probably not what you meant
```

This helps identify potentially problematic control flow that masks exceptions.

### Tail-Call Interpreter

New interpreter variant using tail calls between small C functions:

**Performance:** 3-5% geometric mean improvement on `pyperformance` benchmarks
**Requirements:** Clang 19+ on x86-64 or AArch64, profile-guided optimization recommended
**Build option:** `--with-tail-call-interp`

This optimization provides consistent performance gains without changing Python semantics.

### Free-Threaded Mode Improvements

Significant enhancements to free-threaded builds (PEP 703):

- **Specializing adaptive interpreter** (PEP 659) now enabled
- **Performance penalty reduced to 5-10%** on single-threaded code (down from much higher in 3.13)
- **Context-aware warnings** supported via `-X context_aware_warnings`
- **Thread context variable inheritance**
- Permanent solutions replace temporary workarounds from 3.13

Free-threaded mode is becoming production-ready for multi-threaded workloads.

## Type Hint Enhancements

### Deferred Annotation Evaluation

The primary type system enhancement is deferred annotation evaluation:

```python
# No more string quotes for forward references!
class Node:
    def add_child(self, child: Node) -> None:  # Just works
        pass

# Old way (still works but unnecessary):
class OldNode:
    def add_child(self, child: 'OldNode') -> None:
        pass
```

### Enhanced Introspection

```python
import inspect
from annotationlib import Format

def example(x: int, y: str) -> bool:
    pass

# Control annotation format during introspection
sig = inspect.signature(example, annotation_format=Format.FORWARDREF)

# Format signature with annotation control
sig.format(unquote_annotations=True)
```

The new `annotationlib` module provides fine-grained control over annotation evaluation for type checkers, runtime validation libraries, and documentation tools.

## Improved Error Messages

Enhanced diagnostics and suggestions:

### Keyword Typos

```python
>>> wile True:
  File "<stdin>", line 1
    wile True:
    ^^^^
SyntaxError: invalid syntax. Did you mean 'while'?
```

### Control Flow Errors

```python
>>> if x > 0:
...     print("positive")
... elif x < 0:
...     print("negative")
... else:
...     print("zero")
... elif x == -1:
          ^^^^
SyntaxError: 'elif' cannot follow 'else'
```

### Incompatible String Prefixes

```python
>>> text = ub"hello"
           ^^^^^^^^
SyntaxError: string prefixes 'u' and 'b' are incompatible
```

### Type Errors in Collections

```python
>>> {[1, 2, 3]}
TypeError: cannot use 'list' as a set element because it is not hashable

>>> {{'a': 1}: 'value'}
TypeError: cannot use 'dict' as a dictionary key because it is not hashable
```

Errors now provide specific guidance about what went wrong and often suggest corrections.

## Platform Support

### New Platform Support

- **Emscripten** promoted to tier 3 official platform (PEP 776)
- **Free-threaded Python** officially supported (PEP 779)
- **Android binary releases** now provided
- **JIT compiler** included in official Windows and macOS binaries (experimental)

### Build System Changes

- **PGP signatures discontinued** (PEP 761) - replaced with more modern verification methods
- **build-details.json** file provides detailed build information
- **Free-threaded Windows extensions** must specify `Py_GIL_DISABLED` preprocessor variable

## Standard Library Improvements

### compression: New Package with Zstandard Support (PEP 784)

Unified compression interface with new Zstandard (zstd) support:

```python
from compression import zstd, gzip, bz2, lzma

# Zstandard compression (Meta's high-performance algorithm)
data = b"Hello World" * 1000
compressed = zstd.compress(data, level=10)  # Levels 1-22
decompressed = zstd.decompress(compressed)

# Integrated into tarfile, zipfile, shutil
import tarfile
with tarfile.open('archive.tar.zst', 'w:zst') as tar:
    tar.add('myfile.txt')

import zipfile
with zipfile.ZipFile('archive.zip', 'w', compression=zipfile.ZIP_ZSTD) as zf:
    zf.write('myfile.txt')

import shutil
shutil.make_archive('backup', 'zsttar', '/path/to/dir')
```

Zstandard provides excellent compression ratios with very fast decompression.

### pathlib: Major Enhancements

Extensive new functionality for path operations:

```python
from pathlib import Path

# Recursive copy operations
src = Path('my_project')
dst = Path('backup')

src.copy(dst)                    # Copy file or entire directory tree
src.copy_into(dst)               # Copy into destination directory
src.move('/new/location')        # Move file or directory tree
src.move_into('/existing/dir')   # Move into destination directory

# New info attribute caches stat information
p = Path('file.txt')
print(p.info.type)   # Cached file type
print(p.info.size)   # Cached size
print(p.info.mtime)  # Cached modification time
```

These high-level operations simplify common filesystem tasks that previously required `shutil`.

### asyncio: Introspection Tools

New command-line tools for debugging async applications:

```bash
# Flat listing of all tasks in running process
python -m asyncio ps PID

# Hierarchical async call tree
python -m asyncio pstree PID
```

Output shows:
- Task names and states
- Coroutine stacks
- Awaiter chains
- Cycle detection in async graphs

**New API features:**

```python
import asyncio

# Create tasks with arbitrary keyword arguments
task = asyncio.create_task(my_coro(), name='worker', priority=1, custom_data='foo')

# Capture and print call graphs programmatically
graph = asyncio.capture_call_graph()
asyncio.print_call_graph(graph)
```

These tools make debugging complex async code much easier.

### pdb: Remote Debugging and Improvements

**Remote debugging:**

```bash
# Attach pdb to running process
python -m pdb -p PID
```

**Enhanced debugging experience:**

```python
import pdb

# New features:
# - 4-space indentation (matches code style)
# - Auto-indent support
# - $_asynctask variable for current asyncio task
# - set_trace_async() for debugging coroutines

async def my_coro():
    await pdb.set_trace_async()  # Debug async code
    # $asynctask variable available here
```

**Instance reuse:** pdb instances now preserve display and command history across debugging sessions.

### heapq: Max-Heap Support

Complete set of max-heap operations:

```python
import heapq

data = [3, 1, 4, 1, 5, 9, 2, 6]

# Max-heap operations
heapq.heapify_max(data)           # Convert to max-heap
largest = heapq.heappop_max(data)  # Pop largest element
heapq.heappush_max(data, 7)        # Push onto max-heap
heapq.heapreplace_max(data, 8)     # Pop and push in one operation
result = heapq.heappushpop_max(data, 10)  # Push and pop in one operation
```

Previously, max-heaps required manual negation of values.

### multiprocessing & concurrent.futures: Major Changes

**Default start method change on Unix (except macOS):**

Previously `'fork'`, now `'forkserver'` for better safety and compatibility.

```python
import multiprocessing as mp

# Explicitly request fork if needed
ctx = mp.get_context('fork')

# Or set globally
mp.set_start_method('fork')
```

**ProcessPoolExecutor enhancements:**

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    # New methods for resource management
    executor.terminate_workers()  # Graceful shutdown
    executor.kill_workers()       # Forceful shutdown

    # Buffer size control for map()
    results = executor.map(func, data, buffersize=10)
```

**InterpreterPoolExecutor:**

```python
from concurrent.futures import InterpreterPoolExecutor

# True multi-core parallelism using subinterpreters
with InterpreterPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_bound_task, data))
```

**Proxy improvements:**

```python
from multiprocessing import Manager

manager = Manager()

# List proxies: clear(), copy()
lst = manager.list([1, 2, 3])
lst.clear()
lst_copy = lst.copy()

# Dict proxies: fromkeys(), reversed(), union operators
d = manager.dict()
d.update(dict.fromkeys(['a', 'b'], 0))
d2 = manager.dict({'c': 3})
combined = d | d2  # Union operator

# Set support
s = manager.set([1, 2, 3])

# Graceful interruption
process = mp.Process(target=long_task)
process.start()
process.interrupt()  # Send SIGINT for graceful shutdown
```

### Built-in Type Enhancements

**Numeric conversions:**

```python
# New class methods for explicit conversion
x = float.from_number(42)      # float(42)
y = complex.from_number(3.14)  # complex(3.14)

# Thousands separators in fractional parts
price = 1_234.56_78  # Improved readability

# Mixed-mode arithmetic follows C99 standard
result = 3.5 + 2j  # Consistent complex arithmetic
```

**bytes/bytearray improvements:**

```python
# fromhex() accepts bytes and bytes-like objects
data = bytes.fromhex(b'48656c6c6f')  # Accepts bytes
data = bytes.fromhex(bytearray(b'48 65 6c 6c 6f'))  # Accepts bytearray
```

**Functional improvements:**

```python
# map() with strict mode (like zip)
result = map(func, seq1, seq2, strict=True)  # Raises if lengths differ

# memoryview is now a generic type
def process(data: memoryview[int]) -> None:
    pass

# super objects are copyable and pickleable
import copy
super_copy = copy.copy(super(MyClass, self))

# NotImplemented in boolean context raises TypeError
if obj.method() == NotImplemented:  # Still works
    pass
if obj.method():  # TypeError if returns NotImplemented
    pass
```

### functools: Placeholder for partial()

New `Placeholder` sentinel for positional argument reservation:

```python
from functools import partial, Placeholder as _

def format_log(level, message, timestamp):
    return f"[{timestamp}] {level}: {message}"

# Reserve position for timestamp (third argument)
log_info = partial(format_log, 'INFO', _, timestamp='2025-10-07')

# Call with just the message
log_info('Application started')
# '[2025-10-07] INFO: Application started'
```

### http.server: HTTPS Support

```bash
# HTTPS server from command line
python -m http.server 8443 \\
    --tls-cert server.crt \\
    --tls-key server.key \\
    --tls-password-file passphrase.txt
```

```python
from http.server import HTTPSServer, SimpleHTTPRequestHandler
import ssl

context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain('server.crt', 'server.key')

server = HTTPSServer(('localhost', 8443), SimpleHTTPRequestHandler)
server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()
```

Also includes dark mode support for directory listings.

### Other Notable Library Changes

**argparse:**
- Auto-suggest corrections for typos
- Color output support
- Program name reflects how Python found `__main__`

**datetime:**
```python
from datetime import date, time

# strptime() now available on date and time
d = date.strptime('2025-10-07', '%Y-%m-%d')
t = time.strptime('14:30:00', '%H:%M:%S')
```

**getpass:**
```python
import getpass

# Visual feedback during password entry
password = getpass.getpass(echo_char='*')
# Displays: Password: ******
```

**imaplib:**
```python
import imaplib

# RFC 2177 IDLE support
imap = imaplib.IMAP4('mail.example.com')
imap.login('user', 'pass')
imap.idle()  # Wait for new mail notifications
```

**json:**
```python
# Improved error messages
import json

try:
    json.dumps({'key': object()})
except TypeError as e:
    # Exception includes note about which object failed
    print(e.__notes__)  # Details about the non-serializable object

# Color output
python -m json data.json  # Colorized output in terminal
```

**os:**
```python
import os

# Reload environment variables
os.reload_environ()  # Pick up external changes

# Buffer-based file reading
with open('file.bin', 'rb') as f:
    buf = bytearray(1024)
    n = os.readinto(f.fileno(), buf)  # Read into existing buffer

# New scheduling constants
os.SCHED_DEADLINE  # Deadline scheduling
os.SCHED_NORMAL    # Normal scheduling
```

**operator:**
```python
from operator import is_none, is_not_none

# Functional identity checks
filter(is_not_none, [1, None, 2, None, 3])  # [1, 2, 3]
any(map(is_none, values))  # Check if any value is None
```

**warnings:**
```python
import warnings

# Context-aware warning filtering (PEP 749)
# Enable with: -X context_aware_warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    # Filters now use context variables, thread-safe in concurrent code
```

## Performance Optimizations

### Incremental Garbage Collection

Major improvement to garbage collection with order of magnitude reduction in max pause times:

- **Two generations only:** young and old (simplified from three)
- **Incremental collection:** Work spread across multiple operations
- **Less frequent invocation:** Reduces overhead
- **Changed API:** `gc.collect(1)` now performs increment instead of collecting generation 1

This dramatically improves responsiveness for applications with large object graphs.

### Tail-Call Interpreter

3-5% geometric mean performance improvement on `pyperformance` benchmarks when built with:
- Clang 19+ on x86-64 or AArch64
- Profile-guided optimization
- `--with-tail-call-interp` build flag

### Free-Threaded Mode Performance

Single-threaded performance penalty reduced from significant in 3.13 to just **5-10%** in 3.14, with:
- Specializing adaptive interpreter (PEP 659) now enabled
- True multi-core parallelism for multi-threaded workloads
- Permanent optimizations replacing temporary workarounds

### Module-Specific Optimizations

Performance improvements in: `asyncio`, `base64`, `bdb`, `difflib`, `gc`, `io`, `pathlib`, `pdb`, `uuid`, `zlib`

## Security Improvements

### CVE-2025-4517: os.path.realpath() Fix

```python
import os

# New ALLOW_MISSING strict mode value
path = os.path.realpath('/path/to/file', strict='ALLOW_MISSING')
# Safer handling of missing path components
```

### Authenticated Multiprocessing

Forkserver now uses authenticated control socket for better security against process injection attacks.

### HMAC with Verified Implementation

```python
import hmac

# Now uses HACL* formally verified implementation as fallback
# when OpenSSL unavailable (RFC 2104 compliant)
```

### Remote Debugging Security

Multiple layers of security control:
- `PYTHON_DISABLE_REMOTE_DEBUG=1` environment variable
- `-X disable-remote-debug` command-line flag
- `--without-remote-debug` build-time option

## Important Removals and Deprecations

### Major Removals

**argparse:** Deprecated aliases removed
**ast:** Various deprecated node visitors and methods
**asyncio:** Legacy subprocess APIs
**email:** Deprecated utility functions
**importlib.abc:** Old loader methods
**itertools:** Removed `izip_longest` (use `zip_longest`)
**pathlib:** Deprecated `PurePath.is_relative()` removed
**pkgutil:** Legacy import emulation
**pty:** Deprecated functions
**sqlite3:** Old parameter names
**urllib:** Legacy URL parsing methods

### New Deprecations

**PEP 749:** Behavior of `from __future__ import annotations` will change
- Currently makes all annotations strings
- Future behavior will use deferred evaluation
- Migration timeline announced

**Module-specific deprecations** scheduled for removal in Python 3.15, 3.16, 3.17, 3.19, and future versions.

### int() Conversion Change

```python
class OldStyle:
    def __trunc__(self):
        return 42

# No longer works:
# int(OldStyle())  # TypeError

# Use instead:
class NewStyle:
    def __int__(self):
        return 42

int(NewStyle())  # Works
```

`int()` no longer delegates to `__trunc__()`. Use `__int__()` or `__index__()` instead.

## CPython Implementation Changes

### Annotation Storage

Annotations stored as bytecode in `__annotate__` functions rather than evaluated at definition time:

```python
def example(x: int) -> str:
    pass

# Old: example.__annotations__ = {'x': int, 'return': str}
# New: example.__annotate__ = <function>  # Bytecode that computes annotations
```

### Bytecode Changes

- Disassembly improvements with position tracking
- Specialized bytecode display options
- Changes for incremental GC and free-threading support

### Data Model Additions

```python
class Example:
    """Example class."""
    x: int = 1

# Introspection attributes remain from 3.13:
Example.__static_attributes__  # ('x',)
Example.__firstlineno__        # Line number of class definition
```

## C API Changes

### Python Configuration API (PEP 741)

New standardized C API for configuring Python initialization.

### Limited C API Enhancements

Additional functions exposed in limited API for better ABI stability.

### Removed APIs

Numerous unsafe and deprecated C APIs removed with migration guidance provided in documentation.

### Free-Threading Requirements (Windows)

Windows extensions must specify `Py_GIL_DISABLED` preprocessor variable:

```c
#ifdef Py_GIL_DISABLED
// Free-threaded build code
#endif
```

## Migration Notes

### Breaking Changes

1. **Deferred annotation evaluation:** Code using `__annotations__` directly may need updates
2. **Multiprocessing start method:** Unix platforms (except macOS) default to `'forkserver'` instead of `'fork'`
3. **int() conversion:** No longer delegates to `__trunc__()`
4. **NotImplemented in boolean context:** Now raises `TypeError`
5. **Removed modules and functions:** Check deprecation timeline

### Compatibility

Most Python 3.13 code will work in 3.14, except:

- Code directly accessing `__annotations__` (use `annotationlib.get_annotations()`)
- Code relying on `'fork'` start method behavior
- Code using `__trunc__()` for int conversion
- Code using removed deprecated APIs

### Recommended Actions

1. **Update annotation handling:**
   ```python
   # Old way
   annotations = func.__annotations__

   # New way
   from annotationlib import get_annotations
   annotations = get_annotations(func)
   ```

2. **Review multiprocessing code:**
   ```python
   # Explicitly specify start method if fork required
   import multiprocessing
   multiprocessing.set_start_method('fork')
   ```

3. **Test with warnings enabled:**
   ```bash
   python -X context_aware_warnings -W error::DeprecationWarning script.py
   ```

4. **Try free-threaded mode for CPU-bound workloads:**
   ```bash
   python3.14t  # Free-threaded build
   ```

5. **Experiment with InterpreterPoolExecutor:**
   ```python
   from concurrent.futures import InterpreterPoolExecutor
   # May provide better performance than ProcessPoolExecutor
   ```

## Key Takeaways

1. **Deferred annotations are revolutionary** - eliminates forward reference quotes, improves performance, provides flexible evaluation
2. **True multi-core parallelism is here** - subinterpreters in stdlib enable genuine parallel Python execution without multiprocessing overhead
3. **Template strings unlock safe string processing** - t-strings provide foundation for safe SQL, shell commands, and HTML generation
4. **GC pause times dramatically reduced** - incremental collection makes Python more suitable for latency-sensitive applications
5. **Free-threading is production-ready** - 5-10% single-threaded overhead with true parallelism for multi-threaded code
6. **Zstandard compression throughout** - modern, fast compression available in tar, zip, and shutil
7. **Pathlib reaches maturity** - recursive copy/move operations complete the high-level file API
8. **Remote debugging enabled** - attach debuggers to production systems without prior instrumentation
9. **Developer experience enhanced** - syntax highlighting, import completion, better error messages
10. **Performance gains across the board** - tail-call interpreter, optimized modules, incremental GC

Python 3.14 is a landmark release that fundamentally changes how annotations work, enables true parallelism through subinterpreters, and provides powerful new tools for safe string processing. The dramatic reduction in GC pause times, continued free-threading improvements, and extensive standard library enhancements make this one of the most impactful Python releases in years. Combined with quality-of-life improvements to debugging, error messages, and the REPL, Python 3.14 positions the language for high-performance, concurrent applications while maintaining its reputation for developer productivity.
