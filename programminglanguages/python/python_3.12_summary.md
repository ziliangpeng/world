# Python 3.12 Release Summary

**Released:** October 2, 2023
**Source:** [Official Python 3.12 Release Notes](https://docs.python.org/3/whatsnew/3.12.html)

## Overview

Python 3.12 focuses on usability improvements, type system enhancements, and performance optimizations. Major changes include simplified type parameter syntax, relaxed f-string restrictions, per-interpreter GIL support, and significant performance gains across multiple modules.

## Major Language Features

### PEP 695: Type Parameter Syntax

New compact syntax for generic classes and functions:

```python
# Old way
from typing import TypeVar
T = TypeVar('T')
def max(args: Iterable[T]) -> T:
    ...

# New way in 3.12
def max[T](args: Iterable[T]) -> T:
    ...

class list[T]:
    def append(self, element: T) -> None:
        ...
```

New `type` statement for type aliases:

```python
type Point = tuple[float, float]
type Point[T] = tuple[T, T]  # Generic type alias
type HashableSequence[T: Hashable] = Sequence[T]  # With bound
```

### PEP 701: F-String Improvements

Lifted major restrictions on f-strings:

1. **Quote reuse** - Can now reuse the same quote type inside expressions:
```python
f"This is the playlist: {", ".join(songs)}"
```

2. **Multi-line expressions** - Expressions can span multiple lines:
```python
f"This is the playlist: {", ".join([
    'Take me back to Eden',
    'Alkaline',
])}"
```

3. **Comments inside expressions**:
```python
f"This is the playlist: {", ".join([
    'Take me back to Eden',  # My favorite song
    'Alkaline',
])}"
```

4. **Backslashes and Unicode escapes** now work in expressions

## Type Hint Enhancements

### PEP 692: TypedDict for **kwargs

More precise typing for keyword arguments:

```python
from typing import TypedDict, Unpack

class Movie(TypedDict):
    name: str
    year: int

def foo(**kwargs: Unpack[Movie]): ...
```

### PEP 698: @override Decorator

Helps catch method override mistakes:

```python
from typing import override

class Base:
    def get_color(self) -> str:
        return "blue"

class GoodChild(Base):
    @override  # OK: overrides Base.get_color
    def get_color(self) -> str:
        return "yellow"

class BadChild(Base):
    @override  # Type checker error: does not override
    def get_colour(self) -> str:  # Typo!
        return "red"
```

## Interpreter Improvements

### PEP 684: Per-Interpreter GIL

Each sub-interpreter can have its own GIL, enabling better isolation and parallelism for certain use cases.

### PEP 669: Low Impact Monitoring

New monitoring API for profilers and debuggers with minimal performance overhead.

### PEP 688: Buffer Protocol in Python

The buffer protocol is now accessible from Python code, not just C extensions.

### PEP 709: Comprehension Inlining

List/dict/set comprehensions are now inlined, improving performance and semantics.

## Improved Error Messages

Enhanced "Did you mean...?" suggestions:

```python
>>> sys.version_info
NameError: name 'sys' is not defined. Did you forget to import 'sys'?

>>> class A:
...     def __init__(self):
...         self.blech = 1
...     def foo(self):
...         somethin = blech
>>> A().foo()
NameError: name 'blech' is not defined. Did you mean: 'self.blech'?

>>> import a.y.z from b.y.z
SyntaxError: Did you mean to use 'from ... import ...' instead?

>>> from collections import chainmap
ImportError: cannot import name 'chainmap'. Did you mean: 'ChainMap'?
```

## Standard Library Improvements

### pathlib
- `Path` class now supports subclassing
- Better integration with the os module

### asyncio
- **75% performance improvement** in some benchmarks
- Faster socket writing (avoids unnecessary copying)
- Uses `sendmsg()` when available

### os
- Several Windows support improvements
- Better filesystem operations

### sqlite3
- New command-line interface

### uuid
- New command-line interface

### typing
- `isinstance()` checks against runtime-checkable protocols are **2-20x faster**

## Performance Optimizations

### Major Speed Improvements

- **asyncio**: Up to 75% faster in some benchmarks
- **tokenize**: Up to 64% faster (side effect of PEP 701)
- **isinstance()**: 2-20x faster for protocol checks
- **inspect.getattr_static()**: At least 2x faster
- **re.sub()/re.subn()**: 2-3x faster for group replacements
- **super()**: Faster method calls via new `LOAD_SUPER_ATTR` instruction

### Memory Optimizations

- **PEP 623**: Removed `wstr` from Unicode objects, saving 8-16 bytes per string on 64-bit platforms
- Incremental garbage collection reduces pause times

### Build Optimizations

- Experimental BOLT binary optimizer support (1-5% performance gain)
- ThinLTO as default link-time optimization with Clang

## Security Improvements

Replaced SHA1, SHA3, SHA2-384, SHA2-512, and MD5 implementations with formally verified code from the [HACL*](https://github.com/hacl-star/hacl-star/) project (used as fallback when OpenSSL doesn't provide them).

## Important Removals

### PEP 632: distutils Removed

The `distutils` package has been completely removed. Use `setuptools` instead. Migration guide: https://peps.python.org/pep-0632/#migration-advice

### setuptools No Longer Pre-installed in venv

Virtual environments created with `venv` no longer include `setuptools` by default. Install manually if needed:

```bash
pip install setuptools
```

### Removed Modules

- `asynchat`
- `asyncore`
- `imp`
- Several `unittest.TestCase` method aliases

## CPython Implementation Changes

### New Bytecode Instructions

- `BINARY_SLICE`, `STORE_SLICE`
- `CALL_INTRINSIC_1`, `CALL_INTRINSIC_2`
- `LOAD_SUPER_ATTR` (for faster `super()` calls)
- `RETURN_CONST`
- `LOAD_FAST_AND_CLEAR`, `LOAD_FAST_CHECK`

### Removed Bytecode Instructions

- `LOAD_METHOD` (merged into `LOAD_ATTR`)
- `JUMP_IF_FALSE_OR_POP`, `JUMP_IF_TRUE_OR_POP`
- `PRECALL`
- `LOAD_CLASSDEREF`

### perf Profiler Support

CPython now supports the Linux `perf` profiler for performance analysis.

### Stack Overflow Protection

Implemented on supported platforms to prevent crashes.

## C API Changes

### PEP 697: Unstable C API Tier

New tier for experimental C APIs that may change between minor versions.

### PEP 683: Immortal Objects

Objects that are never deallocated, improving performance in certain scenarios.

### Other C API Changes

- Extension modules should use `Py_TPFLAGS_MANAGED_DICT` and `Py_TPFLAGS_MANAGED_WEAKREF` instead of `tp_dictoffset` and `tp_weaklistoffset`
- `PyLongObject` internals changed for better performance (use public API functions)

## Migration Notes

### Breaking Changes

1. `distutils` removed - migrate to `setuptools` or other packaging tools
2. `setuptools` not in venv by default - install manually if needed
3. Several deprecated APIs and modules removed
4. C extension modules may need updates for new memory management flags

### Compatibility

Most Python 3.11 code should work in 3.12 with minimal changes, unless it:
- Depends on removed modules (`distutils`, `asynchat`, `asyncore`, `imp`)
- Uses removed `unittest.TestCase` aliases
- Relies on C API internals that changed

## Key Takeaways

1. **Type system is more ergonomic** with new syntax for generics and type aliases
2. **F-strings are much more flexible** - most previous limitations removed
3. **Significant performance gains** across asyncio, protocols, tokenize, and more
4. **Better error messages** continue to improve developer experience
5. **Cleaner standard library** with legacy code removed
6. **Foundation for future improvements** with per-interpreter GIL and monitoring API

Python 3.12 represents a substantial quality-of-life improvement for developers while laying groundwork for future performance and parallelism enhancements.
