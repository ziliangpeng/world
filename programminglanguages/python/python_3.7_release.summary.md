# Python 3.7 Release Summary

**Released:** June 27, 2018
**Source:** [Official Python 3.7 Release Notes](https://docs.python.org/3/whatsnew/3.7.html)

## Overview

Python 3.7 introduced dataclasses for simplified class definitions, context variables for async-aware state management, and the built-in breakpoint() function. The release made async and await reserved keywords, officially specified dict ordering as part of the language, and added nanosecond-resolution time functions. Notable improvements include better UTF-8 handling, hash-based .pyc files, and significant performance gains through method call optimizations.

## Major Language Features

### PEP 557: Data Classes

New `@dataclass` decorator automatically generates `__init__()`, `__repr__()`, `__eq__()`, and other methods:

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0

p = Point(1.5, 2.5)
print(p)  # Point(x=1.5, y=2.5, z=0.0)
```

### PEP 567: Context Variables

New contextvars module provides async-aware state management (similar to thread-local storage but works correctly with async code):

```python
from contextvars import ContextVar

request_id = ContextVar('request_id')

async def handle_request(req):
    request_id.set(req.id)
    # request_id maintained across async calls
    await process()
```

The asyncio and decimal modules were updated to use context variables, ensuring correct behavior in async contexts.

### PEP 563: Postponed Evaluation of Annotations

Annotations are now stored as strings and evaluated lazily, enabling forward references and improving startup time:

```python
from __future__ import annotations

class C:
    @classmethod
    def from_string(cls, source: str) -> C:  # Forward reference works
        ...

    def validate_b(self, obj: B) -> bool:  # Can reference B before it's defined
        ...

class B:
    ...
```

This becomes the default in Python 3.10.

### PEP 553: Built-in breakpoint()

Easy debugger entry:

```python
def complex_function():
    # ... some code ...
    breakpoint()  # Drops into debugger
    # ... more code ...
```

Calls `sys.breakpointhook()` which defaults to `pdb.set_trace()`. Customizable via `PYTHONBREAKPOINT` environment variable.

## Type System and Data Model Enhancements

### PEP 560: Core Support for typing Module

New `__class_getitem__()` and `__mro_entries__()` methods improve typing module performance:

- Type operations up to 7x faster
- Generic types work without metaclass conflicts
- Several long-standing typing bugs fixed

### PEP 562: Module __getattr__ and __dir__

Modules can now define `__getattr__()` and `__dir__()` for attribute access customization:

```python
# module.py
def __getattr__(name):
    if name == "deprecated_name":
        warnings.warn("deprecated")
        return new_implementation
    raise AttributeError(f"module has no attribute {name}")
```

Useful for deprecation warnings and lazy loading.

### Dict Ordering is Official

The insertion-order preservation of dict objects is now an official part of the Python language specification (was implementation detail in 3.6).

## Interpreter Improvements

### UTF-8 Handling Improvements

**PEP 538**: Legacy C locale coercion automatically switches to UTF-8 based locale when C/POSIX locale detected.

**PEP 540**: Forced UTF-8 runtime mode via `-X utf8` option or `PYTHONUTF8` environment variable, ignoring locale settings.

These changes avoid ASCII encoding issues on systems with misconfigured locales.

### PEP 552: Hash-based .pyc Files

Bytecode cache files can now use source file hash for validation instead of timestamp:

- Supports reproducible builds
- Two variants: checked (validates at runtime) and unchecked (build system manages validity)
- Generated with py_compile or compileall

### Python Development Mode

New `-X dev` option or `PYTHONDEVMODE` environment variable enables additional runtime checks:

- Shows ResourceWarning
- Enables faulthandler
- Enables asyncio debug mode
- Additional memory allocator checks

### PEP 565: DeprecationWarning in __main__

DeprecationWarnings are now shown by default when code runs in `__main__` module (scripts, interactive), helping developers catch deprecated API usage.

## Standard Library Improvements

### time: Nanosecond Resolution Functions (PEP 564)

Six new functions with nanosecond precision:

```python
import time

# Nanosecond resolution (returns int)
ns = time.time_ns()
monotonic_ns = time.monotonic_ns()
perf_counter_ns = time.perf_counter_ns()

# 3x better resolution than time.time()
```

Also added:
- `time.thread_time()` and `time.thread_time_ns()` for per-thread CPU time
- New clock identifiers: `CLOCK_BOOTTIME`, `CLOCK_PROF`, `CLOCK_UPTIME`

### asyncio: Major Enhancements

Significant usability and performance improvements:

- **asyncio.run()**: New provisional function to run coroutines from sync code
- **asyncio.create_task()**: Shortcut for creating tasks
- **asyncio.get_running_loop()**: Returns current loop (raises error if none)
- **asyncio.current_task()** and **asyncio.all_tasks()**: Replace deprecated Task methods
- **Context variables support**: Tasks automatically track their context
- **BufferedProtocol**: Manual control over receive buffer
- **Server improvements**: start_serving parameter, async context manager support
- **loop.start_tls()**: Upgrade existing connection to TLS
- **loop.sock_recv_into()**: Read data directly into buffer
- **loop.sock_sendfile()**: Use os.sendfile() when available
- TCP sockets now created with TCP_NODELAY by default on Linux
- Path-like object support for Unix socket paths

### importlib.resources

New module for accessing package resources (files bundled with packages):

```python
from importlib import resources

# Read resource files
text = resources.read_text('mypackage', 'data.txt')
binary = resources.read_binary('mypackage', 'data.bin')

# Get path to resource
with resources.path('mypackage', 'data.txt') as path:
    # Use path as needed
    ...
```

### Other Notable Improvements

**collections**: `namedtuple()` now supports default values

**datetime**:
- `datetime.fromisoformat()` parses ISO format strings
- `tzinfo` supports sub-minute offsets

**gc**: New `gc.freeze()`, `gc.unfreeze()`, and `gc.get_freeze_count()` for copy-on-write friendly GC

**os**:
- `os.register_at_fork()` for registering fork callbacks
- `os.preadv()` and `os.pwritev()` for vectored I/O
- `os.scandir()` now supports file descriptors

**pathlib**: `Path.is_mount()` checks if path is mount point

**re**:
- Flags can be set within group scope
- `re.split()` supports patterns matching empty strings
- Compiled regex objects can be copied

**socket**:
- `socket.getblocking()` returns blocking mode status
- `socket.close()` function for closing file descriptors
- `AF_VSOCK` support for VM-host communication
- Auto-detection of family, type, protocol from file descriptor

**sqlite3**: `Connection.backup()` method for database backup

**ssl**:
- OpenSSL built-in hostname verification (better performance and security)
- Preliminary TLS 1.3 support
- `SSLContext.minimum_version` and `SSLContext.maximum_version`
- `SSLContext.post_handshake_auth` for TLS 1.3
- No longer sends IP addresses in SNI

**subprocess**:
- `run()` accepts `capture_output` parameter
- `text` parameter as alias for `universal_newlines`
- Windows: `close_fds` now defaults to True when redirecting

**threading**: `threading.Thread.is_shutdown()` class method

**unittest**: New `-k` option for filtering tests by pattern

**unittest.mock**: `seal()` function to prevent further attribute mock creation

## Performance Optimizations

Major performance improvements across the board:

- **Method calls**: Up to 20% faster due to bytecode changes avoiding bound method creation
- **Startup time**: 10% faster on Linux, up to 30% faster on macOS
- **METH_FASTCALL**: Reduced calling overhead for many C methods
- **asyncio.get_event_loop()**: Up to 15x faster (reimplemented in C)
- **asyncio.Future** callbacks and **asyncio.gather()**: Optimized management
- **asyncio.sleep()**: 2x faster for zero/negative delays
- **typing module**: Import time reduced by factor of 7, operations faster
- **sorted() and list.sort()**: 40-75% faster for common cases
- **dict.copy()**: Up to 5.5x faster
- **hasattr() and getattr()**: ~4x faster when attribute not found
- **collections.namedtuple()**: Creation 4-6x faster
- **datetime**: fromordinal() and fromtimestamp() up to 30% faster
- **os.fwalk()**: 2x faster using os.scandir()
- **shutil.rmtree()**: 20-40% faster
- **re**: Case-insensitive matching up to 20x faster, compile() ~10% faster
- **abc module**: Most functions/methods rewritten in C, 1.5x faster isinstance/issubclass
- **hmac.digest()**: One-shot function up to 3x faster

## C API Changes

### PEP 539: New Thread-Local Storage API

New Thread Specific Storage (TSS) API replaces old TLS API:

- Uses `Py_tss_t` type instead of `int` for portability
- Old API deprecated but still available on compatible platforms

### Other C API Additions

- **Context variables**: New C API for contextvars
- **PyImport_GetModule()**: Returns previously imported module
- **Py_RETURN_RICHCOMPARE**: Macro for writing comparison functions
- **Py_UNREACHABLE**: Marks unreachable code paths
- **PyTraceMalloc_Track()** and **PyTraceMalloc_Untrack()**: Expose tracemalloc to C
- **PySlice_Unpack()** and **PySlice_AdjustIndices()**: Safer slice handling
- **PyOS_BeforeFork()**, **PyOS_AfterFork_Parent()**, **PyOS_AfterFork_Child()**: Replace PyOS_AfterFork()
- **PyTimeZone_FromOffset()** and **PyDateTime_TimeZone_UTC**: Timezone support
- Many struct fields changed to `const char*` from `char*`

## Important Removals and Deprecations

### Breaking Changes

- **async and await are now reserved keywords** (SyntaxError if used as identifiers)
- **os.stat_float_times()** removed
- **ntpath.splitunc()** removed (use `os.path.splitdrive()`)
- **collections.namedtuple()**: verbose parameter and _source attribute removed
- **bool(), float(), list(), tuple()** no longer accept keyword arguments
- **plistlib**: Removed Plist, Dict, _InternalDict classes
- **asyncio.windows_utils.socketpair()** removed (use `socket.socketpair()`)
- **ssl.SSLSocket and ssl.SSLObject**: Direct instantiation prohibited
- **fpectl module** removed entirely

### Deprecations (Python 3.8 removal)

- **aifc.openfp()**, **sunau.openfp()**, **wave.openfp()**: Use `.open()` instead
- **asyncio.Lock** direct await: Use async context manager
- **asyncio.Task.current_task()** and **Task.all_tasks()**: Use module-level functions
- **collections**: ABCs moving to collections.abc
- **dbm.dumb**: Warning on missing index file
- **enum**: Non-Enum/Flag membership tests will raise TypeError
- **gettext**: Non-integer plural form selection
- **importlib.abc.ResourceLoader**: Use ResourceReader
- **locale.format()**: Use `locale.format_string()`
- **macpath module**: Complete removal
- **dummy_threading and _dummy_thread**: Use threading instead
- **socket.htons()/ntohs()**: Argument truncation will raise exception
- **ssl.wrap_socket()**: Use `SSLContext.wrap_socket()`
- **sys.set_coroutine_wrapper()** and **sys.get_coroutine_wrapper()**
- **sys.callstats()**: Undocumented function

## Platform and Build Changes

### Platform Support

- FreeBSD 9 and older no longer supported
- **OpenSSL 1.0.2 or 1.1 required** (0.9.8 and 1.0.1 no longer supported)
- Affects Debian 8 and Ubuntu 14.04
- UTF-8 locale expected on *nix platforms

### Build Changes

- **--without-threads removed**: threading module always available
- **libffi**: Full copy no longer bundled (must be installed separately on non-OSX Unix)
- **Windows**: Python script downloads from GitHub instead of Subversion

## Other Notable Changes

### Language Changes

- More than 255 arguments/parameters now supported
- `bytes.fromhex()` and `bytearray.fromhex()` ignore all ASCII whitespace
- `str`, `bytes`, `bytearray` gained `isascii()` method
- `ImportError` shows module name and `__file__` path for import failures
- Circular imports with absolute imports now supported
- `TracebackType` can be instantiated from Python, `tb_next` is writable
- `-m` switch: `sys.path[0]` now eagerly expanded to full directory path
- `-X importtime`: Show module import timing
- Await expressions and async for allowed in f-strings
- Dict ordering now preserved in `locals()`

### Security

- XML modules (xml.dom.minidom, xml.sax) no longer process external entities by default

### Documentation

**PEP 545**: Python documentation now available in Japanese, French, and Korean

## Migration Notes

### Breaking Behavior Changes

1. **async/await keywords**: Code using these as identifiers must be updated
2. **StopIteration in generators/coroutines**: Now transformed to RuntimeError (PEP 479)
3. **__aiter__() cannot be async**: Must be regular method returning async iterator
4. **Generator expressions**: Require direct parentheses, no trailing comma
5. **-m switch**: sys.path[0] behavior changed (now full path, not empty string)
6. **socketserver**: server_close() now waits for threads/processes (set block_on_close=False for old behavior)
7. **os.makedirs()**: mode no longer affects intermediate directories
8. **struct.Struct.format**: Now str instead of bytes
9. **subprocess on Windows**: close_fds defaults to True when redirecting
10. **re.split()**: Now splits on patterns matching empty strings
11. **re.escape()**: Only escapes regex special characters (not all non-alphanumeric)

### Key Compatibility Notes

Most Python 3.6 code will run on 3.7, but watch for:
- Use of async/await as identifiers
- Reliance on removed modules/functions
- Code depending on changed behavior (socketserver, re.split, etc.)
- C extensions needing updates for new APIs

## Key Takeaways

1. **Dataclasses simplify class definitions** - automatic method generation for data-holding classes
2. **Context variables enable async-aware state** - better than thread-local for async code
3. **breakpoint() built-in** - standardized debugger entry point
4. **async and await now keywords** - reserved for async syntax
5. **Dict ordering is official** - guaranteed insertion-order preservation
6. **Time functions with nanosecond resolution** - better precision for timing measurements
7. **Significant performance improvements** - 20% faster method calls, faster startup, optimized stdlib
8. **Better UTF-8 support** - locale coercion and forced UTF-8 mode
9. **Hash-based .pyc files** - deterministic bytecode for reproducible builds
10. **asyncio maturity** - major API improvements and performance gains

Python 3.7 represents a significant evolution focusing on developer productivity (dataclasses, breakpoint), async programming improvements (context variables, asyncio enhancements), and performance (method calls, startup time, stdlib optimizations). The release also cleaned up technical debt by making dict ordering official and reserving async/await as keywords.
