# Python 3.5 Release Summary

**Released:** September 13, 2015
**Source:** [Official Python 3.5 Release Notes](https://docs.python.org/3/whatsnew/3.5.html)

## Overview

Python 3.5 introduces revolutionary async programming with `async`/`await` syntax, the matrix multiplication operator, type hints, and major performance improvements across the standard library. This release represents a significant leap forward in Python's capabilities for asynchronous programming and scientific computing.

## Major Language Features

### PEP 492: Coroutines with async and await syntax

Python 3.5 introduces dedicated syntax for asynchronous programming, making async code as clear as synchronous code:

```python
async def http_get(domain):
    reader, writer = await asyncio.open_connection(domain, 80)

    writer.write(b'GET / HTTP/1.1\r\n...')

    async for line in reader:
        print('>>>', line)

    writer.close()
```

Key features:
- `async def` declares coroutine functions
- `await` expression suspends coroutine execution
- `async for` for asynchronous iteration
- `async with` for asynchronous context managers
- New awaitable protocol with `__await__()` method

This replaces the previous generator-based coroutines (`@asyncio.coroutine` and `yield from`) with cleaner, more explicit syntax.

### PEP 465: Matrix Multiplication Operator (@)

A dedicated `@` operator for matrix multiplication makes scientific computing code dramatically more readable:

```python
# Before: nested function calls
S = dot((dot(H, beta) - r).T,
        dot(inv(dot(dot(H, V), H.T)), dot(H, beta) - r))

# After: clean operator syntax
S = (H @ beta - r).T @ inv(H @ V @ H.T) @ (H @ beta - r)
```

The operator is defined via `__matmul__()`, `__rmatmul__()`, and `__imatmul__()` methods. NumPy 1.10+ supports this operator.

### PEP 448: Additional Unpacking Generalizations

Unpacking operators can now be used multiple times in function calls and collection literals:

```python
# Multiple unpacking in function calls
print(*[1], *[2], 3, *[4, 5])  # 1 2 3 4 5
fn(**{'a': 1, 'c': 3}, **{'b': 2, 'd': 4})

# Multiple unpacking in literals
[*range(4), 4]  # [0, 1, 2, 3, 4]
{*range(4), 4, *(5, 6, 7)}  # {0, 1, 2, 3, 4, 5, 6, 7}
{'x': 1, **{'y': 2}}  # {'x': 1, 'y': 2}
```

### PEP 461: Bytes Formatting

The `%` formatting operator now works with `bytes` and `bytearray`, critical for binary protocols:

```python
b'Hello %b!' % b'World'  # b'Hello World!'
b'x=%i y=%f' % (1, 2.5)  # b'x=1 y=2.500000'
b'price: %a' % '10€'  # b"price: '10\\u20ac'"
```

### PEP 484: Type Hints

The new `typing` module provides a standard framework for type annotations:

```python
def greeting(name: str) -> str:
    return 'Hello ' + name
```

Key features:
- Union types, generic types, and `Any` type
- Type annotations available at runtime via `__annotations__`
- No runtime type checking (use external tools like mypy)
- Provisional API status allows for evolution

## Interpreter Improvements

### PEP 471: os.scandir() - Fast Directory Iteration

New `os.scandir()` function provides 3-5x faster directory traversal on POSIX (7-20x on Windows):

```python
for entry in os.scandir(path):
    if not entry.name.startswith('.') and entry.is_file():
        print(entry.name)
```

Benefits:
- Returns iterator instead of list (better memory efficiency)
- Caches stat information from directory listing
- `os.walk()` reimplemented using `scandir()` for dramatic speedup

### PEP 475: Automatic EINTR Retry

System calls interrupted by signals are now automatically retried, eliminating boilerplate:

```python
# Before: manual retry loop needed
while True:
    try:
        print("Hello World")
        break
    except InterruptedError:
        continue

# After: just works
print("Hello World")
```

Affects: `open()`, file I/O, `socket` operations, `select()`, `time.sleep()`, and many more.

### PEP 479: StopIteration Handling in Generators

`StopIteration` raised inside a generator is now converted to `RuntimeError`, preventing silent bugs:

```python
from __future__ import generator_stop

def gen():
    next(iter([]))  # StopIteration raised
    yield

# Raises RuntimeError: generator raised StopIteration
```

Without `__future__` import, raises `PendingDeprecationWarning`. Becomes default in Python 3.7.

### PEP 488: Elimination of .pyo Files

`.pyo` files are gone. `.pyc` files now include optimization level in filename:

- Unoptimized: `module.cpython-35.pyc`
- Optimized (`-O`): `module.cpython-35.opt-1.pyc`
- Optimized (`-OO`): `module.cpython-35.opt-2.pyc`

Multiple optimization levels can now coexist.

### PEP 489: Multi-phase Extension Module Initialization

Extension modules can now use two-phase initialization similar to Python modules, allowing:
- Use of any valid identifier as module name (not just ASCII)
- Better import semantics matching Python modules
- More flexible module setup

## Improved Error Messages and Debugging

### New RecursionError Exception

Maximum recursion depth now raises `RecursionError` (subclass of `RuntimeError`) instead of generic `RuntimeError`, making it easier to catch specifically.

### Better Generator Attributes

Generators now have:
- `gi_yieldfrom` attribute showing the object being iterated by `yield from`
- `__name__` set from function name (not code name)
- `__qualname__` for qualified name

## Standard Library Improvements

### collections: OrderedDict in C

`OrderedDict` reimplemented in C, making it **4 to 100 times faster**. Views now support `reversed()`.

### collections.deque Enhancements

New methods make `deque` a complete `MutableSequence`:
- `index()`, `insert()`, `copy()` methods
- `+` and `*` operators

### functools.lru_cache() Optimization

Most of `lru_cache()` reimplemented in C for significantly better performance.

### subprocess.run()

New high-level API for running subprocesses:

```python
result = subprocess.run(['ls', '-l'], capture_output=True, text=True)
print(result.stdout)
```

### ssl: Memory BIO Support

SSL protocol handling decoupled from network I/O via Memory BIO, enabling SSL over non-socket transports.

### traceback Enhancements

New classes for programmatic traceback handling:
- `TracebackException`, `StackSummary`, `FrameSummary`
- `walk_stack()` and `walk_tb()` functions
- Negative `limit` arguments supported

### Other Notable Improvements

- **math.isclose()**: Test approximate equality with relative/absolute tolerances
- **bytes/bytearray.hex()**: Convert binary data to hex strings
- **memoryview**: Tuple indexing including multi-dimensional
- **glob**: Recursive search with `**` pattern
- **heapq.merge()**: `key` and `reverse` parameters
- **zipapp**: Create executable Python ZIP applications (PEP 441)

## Performance Optimizations

Major speedups across the standard library:

- **os.walk()**: 3-5x faster on POSIX, 7-20x faster on Windows
- **collections.OrderedDict**: 4-100x faster (C implementation)
- **functools.lru_cache()**: Much faster (mostly C implementation)
- **ipaddress operations**: 3-15x faster for subnet/supernet operations
- **io.BytesIO**: 50-100% faster
- **marshal.dumps()**: 20-85% faster depending on version
- **UTF-32 encoder**: 3-7x faster
- **regex parsing**: 10% faster
- **json.dumps()**: As fast with `ensure_ascii=False` as with `True`
- **property() getters**: 25% faster
- **String search operations**: Significantly faster for 1-character substrings

## Security Improvements

- **SSLv3 disabled** by default throughout standard library
- **HTTP cookie parsing** stricter to prevent injection attacks
- **POSIX locale handling** improved for stdin/stdout

## New Modules

### typing

Provisional module for type hints (PEP 484):
- Generic types: `List[T]`, `Dict[K, V]`, `Optional[T]`
- `Any`, `Union`, `Tuple`, `Callable` types
- Type variable support for generic functions/classes

### zipapp

Create executable Python ZIP applications:

```bash
python -m zipapp myapp
python myapp.pyz
```

## Platform Support

### Windows Improvements

- New installer replacing old MSI
- Microsoft Visual C++ 14.0 (Visual Studio 2015) now required
- Extension module filename tagging on Windows

### Windows XP

No longer officially supported (per PEP 11).

## Deprecations

### Soft Keywords (Future)

`async` and `await` introduced as soft keywords. They become full keywords in Python 3.7. Use `__future__` import to enable strict checking.

### Deprecated APIs

- **asyncio.async()** → use `ensure_future()`
- **formatter module** → deprecated
- **inspect.getargspec()** → use `signature()`
- **platform.dist()** and **platform.linux_distribution()** → use external package
- **re.LOCALE with str patterns** → deprecated
- **smtpd decode_data** default will change

## Important Removals

- **email.__version__** attribute removed
- **ftplib.Netrc** internal class removed
- **.pyo files** replaced by tagged .pyc files
- **asyncio.JoinableQueue** removed (use `Queue`)

## C API Changes

### New Functions

- `PyMem_RawCalloc()`, `PyMem_Calloc()`, `PyObject_Calloc()`
- `Py_DecodeLocale()`, `Py_EncodeLocale()`
- `PyCodec_NameReplaceErrors()`
- `PyErr_FormatV()`
- `PyExc_RecursionError` exception
- `PyModule_FromDefAndSpec()` for PEP 489
- `PyNumber_MatrixMultiply()`, `PyNumber_InPlaceMatrixMultiply()`

### Extension Module Tagging

Extension modules now include platform information in filenames:
- Linux: `.cpython-35m-x86_64-linux-gnu.so`
- Windows: `.cp35-win_amd64.pyd`
- macOS: `-darwin.so`

## Migration Notes

### Breaking Changes

1. **datetime.time**: Midnight UTC no longer considered false
2. **Generator syntax**: `f(1 for x in [1], *args)` now raises `SyntaxError`
3. **ssl.SSLSocket.send()**: Raises `SSLWantReadError`/`SSLWantWriteError` instead of returning 0
4. **HTMLParser**: Strict mode removed, `convert_charrefs` now `True` by default
5. **re.split()**: Warns/errors on patterns matching empty strings
6. **Buffer protocol errors**: Message format changed
7. **.pyo files**: No longer supported

### Compatibility

Most Python 3.4 code should work in 3.5 with minimal changes. Main areas requiring attention:
- Update `asyncio.async()` to `ensure_future()`
- Check for `.pyo` file usage
- Review deprecated inspect functions
- Test generator exception handling if relying on `StopIteration` behavior

## Key Takeaways

1. **Async/await syntax** makes asynchronous programming first-class and readable
2. **Matrix multiplication operator** (@) greatly improves scientific computing code
3. **Type hints** provide a standard way to annotate types for static analysis
4. **Major performance gains** across standard library (OrderedDict, os.walk, lru_cache)
5. **Better developer experience** with improved error messages and debugging tools
6. **os.scandir()** dramatically speeds up directory operations
7. **Automatic EINTR retry** eliminates signal handling boilerplate

Python 3.5 represents a major step forward for asynchronous programming and sets the foundation for Python's evolution as a systems programming and data science language.
