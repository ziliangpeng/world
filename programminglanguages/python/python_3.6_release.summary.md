# Python 3.6 Release Summary

**Released:** December 23, 2016
**Source:** [Official Python 3.6 Release Notes](https://docs.python.org/3.6/whatsnew/3.6.html)

## Overview

Python 3.6 introduced transformative syntax features that fundamentally changed how Python code is written. The headline feature - formatted string literals (f-strings) - provided a readable, concise string formatting method that quickly became the preferred approach. Beyond syntax, Python 3.6 brought significant implementation improvements: dictionaries became 20-25% more memory-efficient while gaining insertion-order preservation as an implementation detail. The release also stabilized asyncio, making it production-ready, and introduced comprehensive type annotation syntax for variables alongside function parameters.

## Major Language Features

### PEP 498: Formatted String Literals (f-strings)

The most impactful feature in Python 3.6 was the introduction of f-strings, which transformed string formatting in Python:

```python
>>> name = "Fred"
>>> f"He said his name is {name}."
'He said his name is Fred.'

>>> width = 10
>>> precision = 4
>>> value = decimal.Decimal("12.34567")
>>> f"result: {value:{width}.{precision}}"  # nested fields
'result:      12.35'
```

F-strings evaluate expressions at runtime and format them using the format protocol. They're more readable than `str.format()` and more powerful than %-formatting, quickly becoming the standard way to construct strings in Python.

### PEP 526: Variable Type Annotations

Building on PEP 484's function parameter annotations, Python 3.6 added syntax for annotating variables:

```python
primes: List[int] = []
captain: str  # Note: no initial value!

class Starship:
    stats: Dict[str, int] = {}
```

Like function annotations, variable annotations don't affect runtime behavior - they're stored in `__annotations__` and used by type checkers like mypy and pytype.

### PEP 515: Underscores in Numeric Literals

For improved readability, underscores can now be used as visual separators in numeric literals:

```python
>>> 1_000_000_000_000_000
1000000000000000
>>> 0x_FF_FF_FF_FF
4294967295
>>> '{:_}'.format(1000000)
'1_000_000'
```

### PEP 525 & 530: Asynchronous Generators and Comprehensions

Python 3.6 completed the async/await story by allowing `await` and `yield` in the same function:

```python
async def ticker(delay, to):
    """Yield numbers from 0 to *to* every *delay* seconds."""
    for i in range(to):
        yield i
        await asyncio.sleep(delay)
```

Async comprehensions became possible:

```python
result = [i async for i in aiter() if i % 2]
result = [await fun() for fun in funcs if await condition()]
```

## Type Hint Enhancements

### Enhanced Variable Annotations

PEP 526 brought comprehensive variable annotation support, enabling type checkers to understand variable types at class, instance, and module levels. This closed a significant gap in Python's type hint ecosystem.

## Class and Object Model Improvements

### PEP 487: Simpler Class Customization

The new `__init_subclass__` method simplified subclass customization without requiring metaclasses:

```python
class PluginBase:
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

class Plugin1(PluginBase):
    pass  # Automatically registered
```

### PEP 487: Descriptor Protocol Enhancement

The new `__set_name__` method gives descriptors knowledge of their attribute name:

```python
class IntField:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]
```

### PEP 520: Preserved Class Attribute Order

Class definition order is now preserved in `__dict__`, and `type.__prepare__()` returns an insertion-order-preserving mapping by default.

### PEP 468: Preserved Keyword Argument Order

`**kwargs` now maintains insertion order, matching the order arguments were passed to the function.

## File System and Path Handling

### PEP 519: File System Path Protocol

A new protocol allows `pathlib.Path` and other path-like objects to work seamlessly with standard library functions:

```python
>>> import pathlib
>>> with open(pathlib.Path("README")) as f:
...     contents = f.read()
>>> os.path.splitext(pathlib.Path("some_file.txt"))
('some_file', '.txt')
>>> os.fspath(pathlib.Path("some_file.txt"))
'some_file.txt'
```

The `os.PathLike` interface and `os.fspath()` function enable this interoperability. All standard library path functions were updated to support path-like objects.

## Date and Time Improvements

### PEP 495: Local Time Disambiguation

The new `fold` attribute on `datetime` and `time` objects disambiguates times during daylight saving transitions:

```python
>>> u0 = datetime(2016, 11, 6, 4, tzinfo=timezone.utc)
>>> for i in range(4):
...     u = u0 + i*HOUR
...     t = u.astimezone(Eastern)
...     print(u.time(), 'UTC =', t.time(), t.tzname(), t.fold)
...
04:00:00 UTC = 00:00:00 EDT 0
05:00:00 UTC = 01:00:00 EDT 0
06:00:00 UTC = 01:00:00 EST 1  # fold=1 indicates second occurrence
07:00:00 UTC = 02:00:00 EST 0
```

## Interpreter Improvements

### Compact Dictionary Implementation

Dictionaries now use 20-25% less memory thanks to a compact representation based on Raymond Hettinger's proposal (first implemented in PyPy). As a side effect, dictionaries preserve insertion order, though this was considered an implementation detail in 3.6 (it became guaranteed in 3.7).

### PEP 523: Frame Evaluation API

A new C-level API allows intercepting frame evaluation, enabling tools like debuggers and JITs to customize Python code execution.

### PYTHONMALLOC Environment Variable

Debug hooks can now be installed on Python memory allocators in release builds using `PYTHONMALLOC=debug`. This detects buffer overflows, underflows, and API violations, with tracebacks showing where memory was allocated.

### DTrace and SystemTap Support

Python can be built with `--with-dtrace` to enable static markers for function calls, garbage collection, and line execution, allowing production instrumentation without debug builds.

## Error Messages and Debugging

Python 3.6 introduced several error message improvements:

- New `ModuleNotFoundError` exception (subclass of `ImportError`) when modules can't be found
- Abbreviated repeated traceback lines: `"[Previous line repeated {count} more times]"`
- Better debugging with `PYTHONMALLOC` and `tracemalloc` integration

## Standard Library Improvements

### asyncio Stabilized

The asyncio module graduated from provisional status to stable. Major improvements include:

- **30% performance boost** with C implementations of Future and Task
- `get_event_loop()` returns the running loop when called from coroutines
- New `run_coroutine_threadsafe()` for submitting coroutines from other threads
- `loop.create_future()` for custom event loop implementations
- `loop.shutdown_asyncgens()` for proper async generator cleanup
- TCP_NODELAY enabled by default for all TCP transports

### New secrets Module (PEP 506)

A dedicated module for cryptographically strong random numbers suitable for security:

```python
import secrets
token = secrets.token_hex(16)
password = secrets.token_urlsafe()
```

This provides a clear alternative to the `random` module, which should never be used for security purposes.

### hashlib Enhancements

Major cryptographic additions:

- **BLAKE2** hash functions: `blake2b()` and `blake2s()`
- **SHA-3** hash functions: `sha3_224()`, `sha3_256()`, `sha3_384()`, `sha3_512()`
- **SHAKE** hash functions: `shake_128()` and `shake_256()`
- **scrypt** key derivation function
- OpenSSL 1.1.0 support

### typing Module Improvements

New features for type hints:

- `typing.ContextManager` abstract base class
- `typing.Collection` ABC for sized iterable containers
- `typing.Reversible` ABC for reversible iterables
- `typing.AsyncGenerator` ABC
- Better integration with type checkers

### datetime Enhancements

- ISO 8601 date directives: `%G`, `%u`, and `%V` in `strftime()`
- `isoformat()` accepts `timespec` parameter for precision control
- `combine()` accepts optional `tzinfo` argument

### collections Additions

- New `Collection`, `Reversible`, and `AsyncGenerator` ABCs
- `namedtuple()` accepts `module` parameter for `__module__` attribute
- Recursive `deque` instances can be pickled

### enum Enhancements

New `Flag` and `IntFlag` base classes for bitwise-combinable constants:

```python
>>> from enum import Enum, auto
>>> class Color(Enum):
...     red = auto()
...     blue = auto()
...     green = auto()
>>> list(Color)
[<Color.red: 1>, <Color.blue: 2>, <Color.green: 3>]
```

The `auto()` function automatically assigns values to enum members.

## Windows Improvements

### PEP 528 & 529: UTF-8 Encoding

Windows console and filesystem encoding changed to UTF-8:

- Console (`sys.stdin`, `sys.stdout`, `sys.stderr`) defaults to UTF-8
- Filesystem encoding (`sys.getfilesystemencoding()`) returns `'utf-8'`
- Eliminates many encoding issues on Windows

### Other Windows Enhancements

- `py.exe` launcher no longer prefers Python 2 when used interactively
- `python.exe` and `pythonw.exe` marked long-path aware (bypasses 260-character limit)
- `._pth` files enable isolated mode with explicit search paths
- `python36.zip` can serve as a `PYTHONHOME` landmark

## Security Improvements

- New `secrets` module for cryptographically secure randomness
- `os.urandom()` on Linux blocks until entropy pool is initialized (PEP 524)
- OpenSSL 1.1.0 support in `ssl` and `hashlib`
- Improved default settings in `ssl` module
- BLAKE2, SHA-3, SHAKE, and scrypt support

## Performance Optimizations

Python 3.6 included numerous performance improvements:

- Dictionary operations are faster due to compact implementation
- asyncio up to 30% faster with C implementations
- `StreamReader.readexactly()` optimized
- `loop.getaddrinfo()` optimized to avoid redundant system calls
- Various bytecode optimizations

## Important Removals

Key APIs and features removed in 3.6:

- `inspect.getmoduleinfo()` - Use `inspect.getmodulename()`
- `inspect.getargspec()` - Use `inspect.signature()` or `inspect.getfullargspec()`
- Various deprecated `asyncio` APIs
- Several deprecated SSL/TLS features

## Deprecations

Notable deprecations in 3.6:

### Async keyword becomes reserved
`async` and `await` are now reserved keywords (not just soft keywords in async contexts).

### Deprecated modules and functions
- Importing from `collections` instead of `collections.abc` deprecated
- `imp` module deprecated in favor of `importlib`
- Various `asyncio` APIs deprecated
- Many C API functions deprecated

## CPython Implementation Changes

### Bytecode Changes

Several new opcodes were added to support new features and optimizations. The bytecode is not backwards compatible.

### C API Changes

- New frame evaluation API (PEP 523)
- New debug hooks for memory allocators
- PyArg_ParseTuple() now supports exception chaining
- Various deprecated C API functions

## Migration Notes

### Breaking Changes

1. `async` and `await` are now reserved keywords
2. Dictionary iteration order is an implementation detail (don't rely on it yet)
3. Some deprecated functions removed
4. Bytecode format changed (recompile .pyc files)

### Compatibility

Most Python 3.5 code works unchanged in 3.6. Main issues:

- Code using `async` or `await` as identifiers must rename them
- Code relying on specific dictionary iteration order might work accidentally in 3.6 (order preservation was not guaranteed until 3.7)
- Extension modules need recompilation due to bytecode changes

### Recommended Actions

1. **Adopt f-strings** - They're more readable and often faster than alternatives
2. **Add variable type annotations** - Improve code documentation and enable better tooling
3. **Use underscores in large numeric literals** - Improves readability
4. **Stabilize async code with asyncio** - The API is now stable
5. **Use `secrets` module** - For any security-sensitive random number generation
6. **Implement path protocol** - For custom path-like classes
7. **Update Windows code** - Benefit from UTF-8 encoding

## Key Takeaways

1. **F-strings revolutionized string formatting** - They became the preferred way to construct strings in Python
2. **Variable type annotations completed the type hint story** - Full static typing support for modern Python codebases
3. **Dictionaries got faster and smaller** - 20-25% memory reduction with order preservation as a bonus
4. **asyncio became production-ready** - Stable API with significant performance improvements
5. **Windows finally got UTF-8** - Eliminated major encoding headaches for Windows developers
6. **Security got easier** - Dedicated `secrets` module and extensive cryptographic hash support
7. **Class customization simplified** - `__init_subclass__` and `__set_name__` reduced metaclass needs
8. **Path handling modernized** - Path protocol unified str/bytes/pathlib usage

Python 3.6 was a landmark release that introduced features developers use daily. F-strings, variable annotations, and compact dictionaries fundamentally improved Python's ergonomics. The stabilization of asyncio and Windows UTF-8 support resolved long-standing pain points. This release set the stage for Python 3.7's guarantee of dictionary ordering and subsequent type system enhancements.
