# Python 3.8 Release Summary

**Released:** October 14, 2019
**Source:** [Official Python 3.8 Release Notes](https://docs.python.org/3/whatsnew/3.8.html)

## Overview

Python 3.8 introduces several powerful new language features including assignment expressions (the "walrus operator"), positional-only parameters, and self-documenting f-strings. The release brings major improvements to the type system with TypedDict, Literal types, and Protocol definitions. Performance optimizations are spread across the standard library, and the C API receives substantial enhancements for better embedding and extension capabilities.

## Major Language Features

### PEP 572: Assignment Expressions (Walrus Operator)

The walrus operator `:=` allows assignment as part of an expression, reducing code duplication and improving readability:

```python
# Avoid calling len() twice
if (n := len(a)) > 10:
    print(f"List is too long ({n} elements, expected <= 10)")

# Reuse match objects
if (mo := re.search(r'(\d+)% discount', advertisement)):
    discount = float(mo.group(1)) / 100.0

# Read blocks until EOF
while (block := f.read(256)) != '':
    process(block)

# Use computed values in list comprehensions
[clean_name.title() for name in names
 if (clean_name := normalize('NFC', name)) in allowed_names]
```

### PEP 570: Positional-Only Parameters

New syntax `/` allows functions to specify parameters that must be positional-only, preventing their use as keyword arguments:

```python
def f(a, b, /, c, d, *, e, f):
    print(a, b, c, d, e, f)

# Valid call
f(10, 20, 30, d=40, e=50, f=60)

# Invalid calls
f(10, b=20, c=30, d=40, e=50, f=60)   # b cannot be a keyword argument
f(10, 20, 30, 40, 50, f=60)           # e must be a keyword argument
```

Benefits include:
- Emulating C function signatures in pure Python
- Hiding parameter names that aren't helpful to users
- Allowing parameter names to be used in `**kwargs`
- Enabling future parameter name changes without breaking compatibility

```python
def f(a, b, /, **kwargs):
    print(a, b, kwargs)

>>> f(10, 20, a=1, b=2, c=3)  # a and b used both ways
10 20 {'a': 1, 'b': 2, 'c': 3}
```

### F-String Enhancements

F-strings now support `=` for self-documenting expressions (PEP 570 typo - should be no PEP), perfect for debugging:

```python
>>> user = 'eric_idle'
>>> member_since = date(1975, 7, 31)
>>> f'{user=} {member_since=}'
"user='eric_idle' member_since=datetime.date(1975, 7, 31)"

# With format specifiers
>>> delta = date.today() - member_since
>>> f'{user=!s}  {delta.days=:,d}'
'user=eric_idle  delta.days=16,075'

# Shows the full expression
>>> print(f'{theta=}  {cos(radians(theta))=:.3f}')
theta=30  cos(radians(theta))=0.866
```

## Type Hint Enhancements

### PEP 589: TypedDict

Dictionaries with per-key type annotations:

```python
from typing import TypedDict

class Location(TypedDict, total=False):
    lat_long: tuple
    grid_square: str
    xy_coordinate: tuple
```

### PEP 586: Literal Types

Type annotations for specific literal values:

```python
def get_status(port: int) -> Literal['connected', 'disconnected']:
    ...
```

### PEP 591: Final Qualifier

Prevent subclassing, overriding, and reassignment:

```python
from typing import Final

pi: Final[float] = 3.1415926536
```

### PEP 544: Protocol Definitions

Structural subtyping with runtime checking:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...
```

Additional type system improvements:
- New `typing.SupportsIndex` protocol
- New `typing.get_origin()` and `typing.get_args()` functions

## Interpreter Improvements

### PEP 578: Runtime Audit Hooks

Security-focused API allowing applications to monitor potentially dangerous operations:

```python
import sys

def audit_hook(event, args):
    if event == 'open':
        print(f"File opened: {args[0]}")

sys.addaudithook(audit_hook)
```

### PEP 587: Python Initialization Configuration

New C API for fine-grained control over Python initialization with better error reporting. Introduces `PyConfig`, `PyPreConfig`, and `PyStatus` structures.

### PEP 590: Vectorcall Protocol

Fast calling protocol for CPython improving performance of callable objects. Currently provisional, made stable in Python 3.9.

### Debug Build ABI Compatibility

Debug and release builds now use the same ABI on Unix, allowing:
- Loading C extensions built in release mode from debug Python
- Loading C extensions built with stable ABI from debug Python
- Statically linked Python to load C extensions from shared library Python

## Other Language Changes

### Enhanced Built-in Types

- `bool`, `int`, and `fractions.Fraction` now have `as_integer_ratio()` method
- `int`, `float`, and `complex` constructors use `__index__()` when available
- Regular expressions support `\N{name}` Unicode escapes
- Dictionaries and dict views support `reversed()` iteration
- Generalized iterable unpacking in `yield` and `return` no longer requires parentheses

### Math Enhancements

Three-argument `pow()` now supports negative exponents for modular multiplicative inverse:

```python
>>> pow(38, -1, 137)  # Modular inverse of 38 mod 137
119
>>> 119 * 38 % 137
1
```

### Dictionary Comprehension Order

Dict comprehensions now compute key before value, matching dict literals:

```python
>>> names = ['Martin von Löwis', 'Łukasz Langa', 'Walter Dörwald']
>>> {(n := normalize('NFC', name)).casefold() : n for name in names}
{'martin von löwis': 'Martin von Löwis',
 'łukasz langa': 'Łukasz Langa',
 'walter dörwald': 'Walter Dörwald'}
```

### Improved Error Messages

- Better SyntaxWarning when comma is missing: `[(10, 20) (30, 40)]`
- Warning for identity checks with literals: `if x is "hello"`

## New Modules

### importlib.metadata

Provides (provisional) support for reading metadata from third-party packages:

```python
>>> from importlib.metadata import version, requires, files
>>> version('requests')
'2.22.0'
>>> list(requires('requests'))
['chardet (<3.1.0,>=3.0.2)']
```

## Standard Library Improvements

### asyncio

Major upgrades to the async framework:

- `asyncio.run()` graduated from provisional to stable API
- Native async REPL: `python -m asyncio`
- `asyncio.CancelledError` now inherits from `BaseException` instead of `Exception`
- Windows: `ProactorEventLoop` is now the default, with UDP support
- Task naming support with `create_task(name=...)` and `set_name()`
- Happy Eyeballs support in `create_connection()`

### functools

Three significant additions:

1. **Direct decorator usage of `lru_cache`:**
```python
@lru_cache
def f(x):
    ...
```

2. **New `cached_property` decorator:**
```python
@functools.cached_property
def variance(self):
    return statistics.variance(self.data)
```

3. **New `singledispatchmethod` decorator:**
```python
@singledispatchmethod
def process(self, value):
    ...

@process.register(list)
def _(self, value):
    ...
```

### math

Substantial expansion of mathematical functions:

- `math.prod()` - product of numbers (analogous to `sum()`)
- `math.perm()` and `math.comb()` - permutations and combinations
- `math.dist()` - Euclidean distance between points
- `math.isqrt()` - accurate integer square root
- `math.hypot()` - now handles multiple dimensions

```python
>>> math.prod([1, 2, 3, 4])
24
>>> math.comb(10, 3)
120
>>> math.isqrt(650320426)
25506
```

### statistics

Major enhancements including normal distribution support:

```python
>>> from statistics import NormalDist
>>> temperature = NormalDist.from_samples([4, 12, -3, 2, 7, 14])
>>> temperature.mean
6.0
>>> temperature.cdf(3)  # Probability of being under 3 degrees
0.3184678262814532

# Arithmetic operations
>>> temperature * (9/5) + 32  # Convert to Fahrenheit
NormalDist(mu=50.0, sigma=12.294144947901014)
```

New functions:
- `statistics.fmean()` - fast floating-point mean
- `statistics.geometric_mean()`
- `statistics.multimode()` - returns all most common values
- `statistics.quantiles()` - divide data into equal probability intervals

### multiprocessing

- New `multiprocessing.shared_memory` module for direct cross-process memory access
- macOS now uses `spawn` start method by default

### Pickle Protocol 5

New protocol supporting out-of-band data buffers for efficient large data transfer between processes:

```python
# Reduces memory copies and allows custom compression
# for multi-core/multi-machine processing
```

### Other Notable Module Updates

**pathlib**
- `Path.link_to()` creates hard links

**os**
- `os.add_dll_directory()` on Windows for native dependency search paths
- `os.memfd_create()` wraps the memfd_create() syscall
- Better Windows reparse point handling

**shutil**
- `copytree()` accepts `dirs_exist_ok` parameter
- `make_archive()` defaults to modern pax format
- `rmtree()` handles directory junctions properly on Windows

**socket**
- `socket.create_server()` convenience function
- `socket.has_dualstack_ipv6()` checks IPv4/IPv6 support

**datetime**
- `fromisocalendar()` constructors (inverse of `isocalendar()`)

**itertools**
- `accumulate()` gains `initial` parameter

**logging**
- `basicConfig()` gains `force` parameter to remove/replace existing handlers

**unittest**
- `AsyncMock` for async testing
- `IsolatedAsyncioTestCase` for coroutine test cases
- Module and class-level cleanup with `addModuleCleanup()` and `addClassCleanup()`

**ast**
- AST nodes now have `end_lineno` and `end_col_offset` attributes
- `ast.get_source_segment()` returns source code for nodes
- Type comment parsing support

**typing**
- TypedDict, Literal, Final, Protocol all added (covered earlier)

## Performance Optimizations

### Standard Library Optimizations

- **subprocess**: Can use `os.posix_spawn()` for better performance on macOS/Linux
- **shutil**: Platform-specific fast-copy syscalls (26% faster on Linux, 50% on macOS, 40% on Windows)
- **shutil.copytree**: Uses `os.scandir()` with cached `stat()` (9% faster on Linux, 20-30% on Windows)
- **pickle**: Default protocol upgraded to Protocol 4 (better performance, smaller size)
- **operator.itemgetter**: 33% faster for common cases
- **collections.namedtuple**: Field lookups 2x faster
- **list** constructor: 12% smaller when input length is known
- **Class variable writes**: Doubled speed

### Interpreter Optimizations

- `LOAD_GLOBAL` instruction uses new per-opcode cache (40% faster)
- GC-tracked objects reduced by 4-8 bytes (removed `Py_ssize_t` from `PyGC_Head`)
- Reduced overhead for builtin function calls (20-50% faster in some cases)

## Security Improvements

- XML modules (`xml.dom.minidom`, `xml.sax`) no longer process external entities by default
- Windows DLL dependencies resolved more securely (only system paths, DLL directory, and explicitly added directories)

## Parallel Bytecode Cache

New `PYTHONPYCACHEPREFIX` setting allows bytecode cache in separate parallel filesystem tree instead of `__pycache__` subdirectories:

```bash
export PYTHONPYCACHEPREFIX=/tmp/pycache
```

## Important Removals

### Removed Features

- `macpath` module
- `platform.popen()` (use `os.popen()`)
- `time.clock()` (use `time.perf_counter()` or `time.process_time()`)
- `pyvenv` script (use `python -m venv`)
- `parse_qs`, `parse_qsl`, `escape` from `cgi` module
- `filemode` from `tarfile` module
- `unicode_internal` codec
- `sys.set_coroutine_wrapper()` and `sys.get_coroutine_wrapper()`

### Breaking Changes

- Importing ABCs from `collections` (delayed to 3.9, use `collections.abc`)
- `XMLParser` no longer accepts `html` argument
- `dbm.dumb` databases with flag `'r'` are now truly read-only

## Deprecations

Key deprecations include:

- `@asyncio.coroutine` decorator (use `async def`)
- Explicit `loop` parameter in many asyncio functions
- Various `gettext` functions returning bytes
- `threading.Thread.isAlive()` method
- Implicit integer conversion for `Decimal` and `Fraction`
- AST classes `Num`, `Str`, `Bytes`, `NameConstant`, `Ellipsis` (use `ast.Constant`)

## CPython Implementation Changes

### Build System

- `sys.abiflags` became empty string (pymalloc flag `m` removed as builds are ABI compatible)
- New `--embed` option for `python3-config` when embedding Python

### C API Changes

Major changes for extension authors:

- Extension modules no longer linked to libpython on Unix (except Android/Cygwin)
- `PyInterpreterState` moved to internal headers
- Heap-allocated types hold reference to type object
- `PyCompilerFlags` gained `cf_feature_version` field
- `PyEval_ReInitThreads()` removed
- Use of `#` format variants without `PY_SSIZE_T_CLEAN` now deprecated

### Python Initialization

New initialization API (PEP 587) provides:
- Fine-grained configuration control
- Better error reporting
- Structures: `PyConfig`, `PyPreConfig`, `PyStatus`, `PyWideStringList`
- Functions: `Py_InitializeFromConfig()`, `Py_PreInitialize()`, etc.

## Migration Notes

### Key Compatibility Issues

1. **asyncio.CancelledError** - Now inherits from `BaseException`, may affect exception handling
2. **Windows DLL loading** - More restrictive, may break code relying on current directory
3. **C extensions** - No longer linked to libpython on Unix
4. **Type objects** - Heap-allocated types now hold type references
5. **Positional-only parameters** - Some functions now enforce positional-only

### Updating Code

Most Python 3.7 code runs unchanged in 3.8, but watch for:

- Code depending on removed modules (`macpath`, `time.clock()`, etc.)
- XML processing code (external entities disabled by default)
- Windows DLL loading that relies on `PATH` or current directory
- C extensions that accessed `PyInterpreterState` fields directly
- Bare except clauses that caught `asyncio.CancelledError`

### C Extension Updates

For C extensions:
- Add `Py_DECREF` on type object in deallocator if manually incrementing
- Define `PY_SSIZE_T_CLEAN` before including Python.h
- Update `PyCompilerFlags` usage with `cf_feature_version`
- Replace `PyEval_ReInitThreads()` with `PyOS_AfterFork_Child()`

## Key Takeaways

1. **Walrus operator** brings significant expressiveness improvements for reducing code duplication
2. **Positional-only parameters** enable better API design and future compatibility
3. **Type system maturity** with TypedDict, Literal, Protocol, and Final
4. **asyncio stabilization** with `asyncio.run()` and better Windows support
5. **Math and statistics** substantially enhanced for scientific computing
6. **Performance gains** across standard library, especially file operations
7. **Security hardening** with audit hooks and safer XML/DLL handling
8. **C API improvements** for better embedding and extension capabilities

Python 3.8 represents a significant step forward in language expressiveness and type system capabilities, while maintaining strong backward compatibility. The focus on developer experience, performance, and security makes it a solid foundation for modern Python development.
