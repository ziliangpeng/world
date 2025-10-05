# Python 3.10 Release Summary

**Released:** October 4, 2021
**Source:** [Official Python 3.10 Release Notes](https://docs.python.org/3/whatsnew/3.10.html)

## Overview

Python 3.10 introduces structural pattern matching as its marquee feature, alongside significant improvements to error messages, type hints, and developer ergonomics. The release emphasizes better debugging through precise line numbers, cleaner type annotation syntax with union operators, and numerous quality-of-life improvements across the standard library. Enhanced error messages and warnings help catch common mistakes earlier in development.

## Major Language Features

### PEP 634/635/636: Structural Pattern Matching

Python gains match/case statements for pattern matching, enabling declarative data structure handling:

```python
# Simple literal matching
def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case _:
            return "Something's wrong with the internet"

# Pattern matching with variable binding
match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y={y}")
    case (x, 0):
        print(f"X={x}")
    case (x, y):
        print(f"X={x}, Y={y}")

# Class pattern matching
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

match point:
    case Point(x=0, y=0):
        print("Origin")
    case Point(x=0, y=y):
        print(f"On Y-axis at {y}")
    case Point(x=x, y=0):
        print(f"On X-axis at {x}")

# Nested patterns
match points:
    case []:
        print("No points")
    case [Point(0, 0)]:
        print("Origin only")
    case [Point(x, y)]:
        print(f"Single point {x}, {y}")
    case [Point(0, y1), Point(0, y2)]:
        print(f"Two Y-axis points at {y1}, {y2}")

# Guards with if clause
match point:
    case Point(x, y) if x == y:
        print(f"Point on diagonal at {x}")
    case Point(x, y):
        print(f"Point not on diagonal")

# OR patterns
match status:
    case 401 | 403 | 404:
        return "Not allowed"
```

Key features:
- Literal, variable, sequence, mapping, and class patterns
- Wildcards (`_`) for catch-all cases
- Guards for conditional matching
- Variable capture with `as` keyword
- Nested patterns for complex data structures

### Parenthesized Context Managers

Context managers can now span multiple lines with parentheses:

```python
with (
    CtxManager1() as example1,
    CtxManager2() as example2,
    CtxManager3() as example3,
):
    ...
```

Trailing commas are supported for cleaner diffs and easier maintenance.

## Type Hint Enhancements

### PEP 604: Union Type Operator

New `|` syntax for union types provides cleaner type annotations:

```python
# Old way
def square(number: Union[int, float]) -> Union[int, float]:
    return number ** 2

# New way in 3.10
def square(number: int | float) -> int | float:
    return number ** 2

# Works with isinstance and issubclass
>>> isinstance(1, int | str)
True
```

### PEP 612: Parameter Specification Variables

Enables precise typing of decorators and higher-order functions:

```python
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec('P')
R = TypeVar('R')

def add_logging(f: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {f.__name__}")
        return f(*args, **kwargs)
    return wrapper
```

Use `Concatenate` to add or modify parameters while preserving the rest.

### PEP 613: Explicit TypeAlias

Disambiguates type aliases from regular assignments:

```python
# Now explicit
StrCache: TypeAlias = 'Cache[str]'  # type alias
LOG_PREFIX = 'LOG[DEBUG]'  # module constant
```

### PEP 647: User-Defined Type Guards

`TypeGuard` enables custom type narrowing functions:

```python
from typing import TypeGuard

def is_str_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

# Type checkers understand the narrowing
def process(val: list[object]) -> None:
    if is_str_list(val):
        # val is now known to be list[str]
        print(" ".join(val))
```

## Interpreter Improvements

### PEP 626: Precise Line Numbers

Tracing events now generate for every executed line with correct line numbers:
- `f_lineno` attribute always contains expected line number
- Better debugging, profiling, and coverage tool accuracy
- `co_lnotab` deprecated in favor of `co_lines()` method

### PEP 597: Optional EncodingWarning

Helps catch encoding bugs where locale-dependent defaults are used:

```python
# BUG: should specify encoding="utf-8"
with open("data.json") as f:
    data = json.load(f)
```

Enable with `-X warn_default_encoding` or `PYTHONWARNDEFAULTENCODING` environment variable to detect platform-dependent encoding issues.

## Improved Error Messages

### Enhanced SyntaxError Messages

**Unclosed brackets and quotes:**
```python
>>> expected = {9: 1, 18: 2,
...             19: 2, 27: 3,
SyntaxError: '{' was never closed
```

**Error range highlighting:**
```python
>>> foo(x, z for z in range(10), t, w)
SyntaxError: Generator expression must be parenthesized
               ^^^^^^^^^^^^^^^^^^^^
```

**Specific error hints:**
- Missing `:` before blocks
- Unparenthesized tuples in comprehensions
- Missing commas in collections
- Multiple exception types without parentheses
- Using `=` instead of `==` in comparisons
- Invalid starred expressions in f-strings

### Better AttributeError and NameError

```python
>>> collections.namedtoplo
AttributeError: module 'collections' has no attribute 'namedtoplo'.
Did you mean: 'namedtuple'?

>>> schwarzschild_black_hole = None
>>> schwarschild_black_hole
NameError: name 'schwarschild_black_hole' is not defined.
Did you mean: 'schwarzschild_black_hole'?
```

### Improved IndentationError

```python
>>> def foo():
...    if lel:
...    x = 2
IndentationError: expected an indented block after 'if' statement in line 2
```

## Standard Library Improvements

### dataclasses

**Slots support:**
```python
@dataclass(slots=True)
class Point:
    x: float
    y: float
```

**Keyword-only fields:**
```python
from dataclasses import dataclass, KW_ONLY

@dataclass
class Point:
    x: float
    y: float
    _: KW_ONLY
    z: float = 0.0
    t: float = 0.0

# x, y are positional; z, t are keyword-only
```

### bisect

Added `key` parameter to all functions for custom comparison:
```python
data = [('a', 3), ('b', 1), ('c', 2)]
bisect.insort(data, ('d', 1.5), key=lambda x: x[1])
```

### statistics

New functions for statistical analysis:
- `covariance()` - Calculate covariance between two variables
- `correlation()` - Pearson's correlation coefficient
- `linear_regression()` - Simple linear regression

### itertools

`itertools.pairwise()` returns successive overlapping pairs:
```python
>>> list(pairwise([1, 2, 3, 4]))
[(1, 2), (2, 3), (3, 4)]
```

### zip

Optional `strict` flag ensures equal-length iterables:
```python
>>> list(zip([1, 2], ['a', 'b', 'c'], strict=True))
ValueError: zip() argument 2 is longer than argument 1
```

### pathlib

- Slice and negative indexing support for `PurePath.parents`
- `Path.hardlink_to()` method with consistent argument order
- `follow_symlinks` parameter for `stat()` and `chmod()`

### os

- `os.eventfd()` for Linux eventfd support
- `os.splice()` for zero-copy data transfer between file descriptors
- `os.cpu_count()` support for VxWorks RTOS

### Async improvements

- `aiter()` and `anext()` builtin functions for async iteration
- `contextlib.aclosing()` for safely closing async generators
- `contextlib.AsyncContextDecorator` for async context managers as decorators

### Other improvements

- **int**: `bit_count()` method for population count
- **dict views**: New `mapping` attribute returns MappingProxyType
- **static/class methods**: Now inherit method attributes and have `__wrapped__`
- **enum.StrEnum**: New enum type where all members are strings
- **inspect.get_annotations()**: Safe annotation access with unstringizing support

## Performance Optimizations

- **NaN hashing**: Hash values now depend on object identity, preventing quadratic behavior in collections with multiple NaN values
- **Builtin functions**: No longer accept Decimal/Fraction for integer arguments (must use `__index__` not just `__int__`)

## Security Improvements

- **PEP 644**: Python now requires OpenSSL 1.1.1 or newer
- **SSL module**: More secure defaults with forward secrecy, SHA-1 MAC disabled, security level 2 enforced, minimum TLS 1.2
- OpenSSL 3.0.0 preliminary support

## Important Deprecations and Removals

### Deprecated

- **PEP 632**: `distutils` module deprecated (removed in 3.12)
- `asynchat`, `asyncore`, `smtpd`: Emit DeprecationWarning
- Various SSL constants and functions in favor of newer APIs

### Removed

- `parser` module (due to PEG parser switch)
- `formatter` module
- `PyParser_SimpleParseString*` and related C functions
- Deprecated collections ABC aliases
- `loop` parameter from most asyncio high-level APIs
- Complex number arithmetic special methods (`__int__`, `__floordiv__`, etc.)

## CPython Implementation Changes

### Bytecode changes

- `MAKE_FUNCTION` now accepts dict or tuple for annotations
- `SyntaxError` exceptions have `end_lineno` and `end_offset` attributes

### Build changes

- OpenSSL 1.1.1+ required
- C99 `snprintf()` and `vsnprintf()` required
- SQLite 3.7.15+ required

## C API Changes

### PEP 652: Stable ABI

Explicitly defined stable ABI for extension modules with documented stability guarantees.

### New features

- `PyConfig.orig_argv`: Original command line arguments
- `PyIter_Send()`: Send values to iterators without raising StopIteration
- `Py_NewRef()` and `Py_XNewRef()`: Increment refcount and return object
- `PyModule_AddObjectRef()`: Like PyModule_AddObject without stealing reference
- Type flag `Py_TPFLAGS_DISALLOW_INSTANTIATION` for non-instantiable types
- Type flag `Py_TPFLAGS_IMMUTABLETYPE` for immutable type objects

### Breaking changes

- `PY_SSIZE_T_CLEAN` now required for certain format codes
- `Py_REFCNT()` as inline function requires `Py_SET_REFCNT()` for assignment
- Many private/undocumented APIs removed or moved to internal C API

## Migration Notes

### Pattern matching adoption

Structural pattern matching is optional but recommended for:
- Parsing structured data (JSON, config files)
- State machines and event handlers
- Command/action dispatching
- Complex conditional logic on data shapes

### Type hint migration

The `|` union operator is backward compatible when used only in annotations:
```python
# Works in 3.10+, quoted string works in older versions
def func(x: int | str) -> int | str:  # 3.10+
    ...

def func(x: "int | str") -> "int | str":  # Works in older versions too
    ...
```

### asyncio loop parameter

Remove `loop` parameter from high-level asyncio APIs:
```python
# Old
async def foo(loop):
    await asyncio.sleep(1, loop=loop)

# New
async def foo():
    await asyncio.sleep(1)
```

### Error message testing

Tests checking exact error message text may need updates due to improved error messages.

## Key Takeaways

1. **Pattern matching brings declarative data handling** - Match statements simplify complex conditional logic on structured data
2. **Type hints are more ergonomic** - Union operator `|`, explicit TypeAlias, ParamSpec, and TypeGuard improve type annotation experience
3. **Error messages dramatically improved** - Better SyntaxError, AttributeError, and NameError suggestions help catch bugs faster
4. **Debugging is more accurate** - Precise line numbers from PEP 626 improve profiling and tracing tools
5. **Standard library maturation** - dataclasses keyword-only fields, statistical functions, and numerous quality-of-life improvements
6. **Security hardening** - Modern OpenSSL requirement and secure SSL defaults
7. **Cleaner APIs** - Removed deprecated modules and confusing legacy APIs

Python 3.10 represents a significant leap in language expressiveness with pattern matching while continuing Python's commitment to developer experience through better error messages and debugging support. The type system enhancements make static analysis more powerful without sacrificing Python's dynamic nature.
