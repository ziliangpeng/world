# Python 3.9 Release Summary

**Released:** October 5, 2020
**Source:** [Official Python 3.9 Release Notes](https://docs.python.org/3/whatsnew/3.9.html)

## Overview

Python 3.9 represents a significant evolutionary step with powerful new syntax features, enhanced type hints, and foundational improvements to the interpreter. Major highlights include dictionary merge operators, relaxed decorator syntax, built-in generic types for type hints, a new PEG-based parser, and two new standard library modules for time zones and graph operations. The release also brings substantial performance optimizations through vectorcall, improved garbage collection, and faster builtin types.

## Major Language Features

### PEP 584: Dictionary Merge & Update Operators

Python 3.9 introduces new operators for merging dictionaries:

```python
>>> x = {"key1": "value1 from x", "key2": "value2 from x"}
>>> y = {"key2": "value2 from y", "key3": "value3 from y"}
>>> x | y
{'key1': 'value1 from x', 'key2': 'value2 from y', 'key3': 'value3 from y'}
>>> y | x
{'key2': 'value2 from x', 'key3': 'value3 from y', 'key1': 'value1 from x'}

# In-place update
>>> x |= y
```

The merge operator `|` creates a new dictionary with combined keys, where the rightmost dictionary's values take precedence. The update operator `|=` modifies the dictionary in place. These complement existing methods like `dict.update()` and `{**d1, **d2}` unpacking.

### PEP 616: String Methods to Remove Prefixes and Suffixes

New methods for cleaner string manipulation:

```python
>>> "HelloWorld".removeprefix("Hello")
'World'
>>> "HelloWorld".removesuffix("World")
'Hello'
>>> "HelloWorld".removeprefix("Hi")
'HelloWorld'  # No change if prefix not found
```

These methods are also available on `bytes`, `bytearray`, and `collections.UserString`. They provide a safer alternative to slicing, as they only remove the prefix/suffix if it actually exists.

### PEP 614: Relaxed Decorator Syntax

Any valid expression can now be used as a decorator, not just dotted names:

```python
# Now valid in Python 3.9
@buttons[0].clicked.connect
def handle_click():
    pass

@functions[key]
def decorated():
    pass

# Dictionary lookup decorator
@decorators["important"]
def my_function():
    pass
```

Previously, the grammar restricted decorators to simple names and attribute access. This change allows arbitrary expressions including subscripts, calls, and more.

### PEP 617: New PEG-Based Parser

Python 3.9 replaces the old LL(1) parser with a new PEG (Parsing Expression Grammar) based parser. The new parser has comparable performance but provides more flexibility for future language features. The PEG formalism allows for more natural expression of Python's grammar rules.

For Python 3.9 only, you can switch back to the old parser using `-X oldparser` or `PYTHONOLDPARSER=1` environment variable. The old parser will be completely removed in Python 3.10.

## Type Hint Enhancements

### PEP 585: Generic Types in Standard Collections

Built-in collection types can now be used directly in type annotations without importing from `typing`:

```python
# Old way (still works)
from typing import List, Dict, Tuple
def greet_all(names: List[str]) -> None:
    ...

# New way in 3.9
def greet_all(names: list[str]) -> None:
    for name in names:
        print("Hello", name)

# Works with many standard types
def process(data: dict[str, int]) -> tuple[str, ...]:
    ...

# Even works with types like queue.Queue
from queue import Queue
tasks: Queue[str] = Queue()
```

This simplification makes type hints more accessible and reduces imports. Types like `list`, `dict`, `set`, `frozenset`, `tuple`, and many others from the standard library now support generic syntax.

### PEP 593: Annotated Type Hints

The `typing.Annotated` type allows adding context-specific metadata to type hints:

```python
from typing import Annotated

# Add validation metadata
def process_id(user_id: Annotated[int, "Must be positive"]) -> None:
    ...

# Multiple annotations
UserId = Annotated[int, "range(1, 1000000)", "database key"]

# Can include arbitrary metadata for frameworks
from dataclasses import dataclass
from typing import Annotated

@dataclass
class User:
    name: Annotated[str, "max_length: 100"]
    age: Annotated[int, "minimum: 0", "maximum: 150"]
```

The first argument is the actual type, and subsequent arguments are metadata that can be accessed at runtime using `typing.get_type_hints()` with `include_extras=True`.

## New Standard Library Modules

### zoneinfo: IANA Time Zone Database Support

The `zoneinfo` module brings standardized time zone support:

```python
>>> from zoneinfo import ZoneInfo
>>> from datetime import datetime, timedelta

>>> # Daylight saving time
>>> dt = datetime(2020, 10, 31, 12, tzinfo=ZoneInfo("America/Los_Angeles"))
>>> print(dt)
2020-10-31 12:00:00-07:00
>>> dt.tzname()
'PDT'

>>> # Standard time
>>> dt += timedelta(days=7)
>>> print(dt)
2020-11-07 12:00:00-08:00
>>> print(dt.tzname())
PST
```

The module provides a concrete `datetime.tzinfo` implementation backed by the system's IANA time zone database. For platforms without the database, the `tzdata` package (distributed via PyPI) serves as a fallback.

### graphlib: Topological Sorting

The `graphlib` module provides `TopologicalSorter` for dependency resolution and graph ordering:

```python
from graphlib import TopologicalSorter

# Define dependencies
graph = {"D": {"B", "C"}, "C": {"A"}, "B": {"A"}}
ts = TopologicalSorter(graph)
print(list(ts.static_order()))
# Output: ['A', 'B', 'C', 'D']
```

This is useful for task scheduling, build systems, package dependency resolution, and any scenario requiring ordered processing of dependent items.

## Interpreter Improvements

### PEP 617: PEG Parser

The new PEG-based parser provides a more flexible foundation for future Python syntax improvements. While the performance is comparable to the old LL(1) parser, the PEG formalism makes it easier to express complex grammar rules and will enable new language features starting in Python 3.10.

### PEP 573: Module State Access from C Extensions

C extension types can now easily access their defining module and module state through new APIs. This improves encapsulation and enables better support for multiple interpreters:

```c
// New APIs for C extensions
PyType_FromModuleAndSpec()  // Associate module with class
PyType_GetModule()          // Retrieve the module
PyType_GetModuleState()     // Access module state
```

### Enhanced Multithreading

- **Signal handling optimization**: In multithreaded applications, only the main thread handles signals. Other threads no longer have their bytecode evaluation interrupted unnecessarily for signals they cannot handle.
- **Garbage collection improvement**: The garbage collector no longer blocks on resurrected objects, improving performance in scenarios where finalizers resurrect objects.

### PEP 590: Vectorcall Performance

Many Python builtins (`range`, `tuple`, `set`, `frozenset`, `list`, `dict`) now use the vectorcall protocol for significantly faster function calls. This reduces the overhead of calling these constructors and methods.

## Standard Library Improvements

### asyncio

- **New `asyncio.to_thread()`**: High-level API for running IO-bound functions in separate threads
- **`shutdown_default_executor()`**: New coroutine for clean executor shutdown
- **Linux-specific `PidfdChildWatcher`**: Better process management using file descriptors
- **Improved `wait_for()`**: Better cancellation handling even with timeout <= 0

### math

New mathematical functions:

```python
import math

# Multiple arguments for GCD
math.gcd(12, 18, 24)  # 6

# Least common multiple
math.lcm(4, 6, 8)  # 24

# Next representable float
math.nextafter(1.0, 2.0)  # 1.0000000000000002

# Unit in last place
math.ulp(1.0)  # 2.220446049250313e-16
```

### pathlib

- **`Path.readlink()`**: Read symbolic links directly from Path objects

### os

- **Linux process management**: `os.pidfd_open()` for race-free process management using file descriptors
- **`os.waitstatus_to_exitcode()`**: Convert wait status to exit code
- **`os.unsetenv()` on Windows**: Now available on all platforms

### ipaddress

- **IPv6 Scoped Addresses**: Support for IPv6 addresses with scope zone IDs (e.g., `fe80::1%eth0`)
- **Starting with Python 3.9.5**: Leading zeros in IPv4 addresses are no longer accepted to prevent octal interpretation ambiguity

### random

- **`random.randbytes(n)`**: Generate n random bytes efficiently

### ast

- **`ast.unparse()`**: Convert AST objects back to source code
- **Improved `ast.dump()`**: New indent parameter for readable multiline output

### concurrent.futures

- **`cancel_futures` parameter**: Cancel pending futures on executor shutdown
- **On-demand worker spawning**: `ProcessPoolExecutor` creates workers only when needed
- **No more daemon threads**: Better compatibility with subinterpreters

## Performance Optimizations

### Significant Speed Improvements

- **Builtins with vectorcall**: `range`, `tuple`, `set`, `frozenset`, `list`, `dict` are faster via PEP 590
- **Optimized comprehensions**: Temporary variables in comprehensions are as fast as simple assignment
- **Floor division**: Improved performance for float operations
- **ASCII decoding**: ~15% faster for short strings with UTF-8 and ASCII codecs
- **Signal handling**: Optimized in multithreaded applications
- **`PyLong_FromDouble()`**: Up to 1.87x faster for values fitting in long

### Memory Optimizations

- **Small object allocator**: Can retain one empty arena for reuse, preventing thrashing in simple loops
- **Garbage collection**: Improved handling of resurrected objects prevents unnecessary blocking

### Performance Benchmark Summary

Python 3.9 shows consistent performance improvements over 3.8 across many operations:

- Read operations: Global and builtin reads improved from ~7.6-7.5ns to ~7.8-7.8ns
- Write operations: Most operations improved, especially classvar writes (39.2ns → 39.8ns)
- Data structure access: Slight improvements across list, dict, and deque operations
- Stack operations: List append/pop improved from 50.8ns to 50.6ns

## Important Removals and Deprecations

### Removed

- **`unittest.mock.__version__`**: Erroneous version removed
- **Array methods**: `tostring()` and `fromstring()` (use `tobytes()` and `frombytes()`)
- **sys functions**: `callstats()`, `getcheckinterval()`, `setcheckinterval()`
- **Thread modules**: `_dummy_thread` and `dummy_threading`
- **Threading**: `Thread.isAlive()` (use `is_alive()`)
- **Parser functions**: Several old parser C API functions removed
- **base64 aliases**: `encodestring()` and `decodestring()`
- **fractions.gcd()**: Use `math.gcd()` instead
- **ElementTree methods**: `getchildren()` and `getiterator()`
- **Old plistlib API**: Use `load()`, `loads()`, `dump()`, `dumps()`
- **asyncio**: Old `with (await lock):` syntax, `Task.current_task()`, `Task.all_tasks()`
- **json.loads() encoding parameter**: No longer needed (JSON is always UTF-8)
- **bz2.BZ2File buffering parameter**: No longer supported

### Deprecated

- **`distutils.bdist_msi`**: Use `bdist_wheel` instead
- **`parser` and `symbol` modules**: Will be removed in Python 3.10
- **`NotImplemented` in boolean context**: Will become TypeError
- **`random` module**: Will restrict seeds to specific types (None, int, float, str, bytes, bytearray)
- **`lib2to3` module**: Emits PendingDeprecationWarning due to PEG parser
- **C API**: `PyEval_InitThreads()` and `PyEval_ThreadsInitialized()`
- **AST classes**: `slice`, `Index`, `ExtSlice`, `Suite`, `Param`, `AugLoad`, `AugStore`

### Python 2 Compatibility Cleanup

Python 3.9 is the **last version** providing Python 2 backward compatibility layers. Aliases in the `collections` module (like `collections.Mapping`) are kept for one final release but will be removed in Python 3.10. Test your code with `-W default` to see deprecation warnings.

## CPython Implementation Changes

### New Bytecode Instructions

- **`LOAD_ASSERTION_ERROR`**: Better handling of assert statements
- **`COMPARE_OP` split**: Divided into `COMPARE_OP`, `IS_OP`, `CONTAINS_OP`, and `JUMP_IF_NOT_EXC_MATCH` for more efficient comparisons

### C API Changes

#### New APIs

- **Frame access**: `PyFrame_GetCode()`, `PyFrame_GetBack()`, `PyFrame_GetLineNumber()`
- **Thread state**: `PyThreadState_GetInterpreter()`, `PyThreadState_GetFrame()`, `PyThreadState_GetID()`
- **Efficient calling**: `PyObject_CallNoArgs()`, `PyObject_CallOneArg()`
- **Module helpers**: `PyModule_AddType()`
- **GC queries**: `PyObject_GC_IsTracked()`, `PyObject_GC_IsFinalized()`

#### Changes

- **Limited API improvements**: `Py_EnterRecursiveCall()` and `Py_LeaveRecursiveCall()` now available as functions
- **Heap types**: Must visit their type in `tp_traverse` functions (starting 3.9)
- **Removed `tp_print`**: The long-deprecated slot has been removed from `PyTypeObject`

## Build Changes

- **`--with-platlibdir` option**: Customize platform library directory name (for distros using lib64)
- **Windows ARM64 support**: Python can now be built for Windows 10 ARM64
- **macOS Tcl/Tk**: Better framework detection in `/Library/Frameworks`
- **PGO optimization**: Faster PGO builds by skipping slow tests (15x speedup)
- **`setenv()` and `unsetenv()` required**: Now mandatory on non-Windows platforms

## Migration Notes

### Breaking Changes

1. **`__import__()` and `importlib.util.resolve_name()`**: Raise `ImportError` instead of `ValueError` for invalid relative imports
2. **IPv4 leading zeros**: No longer accepted (starting 3.9.5) due to ambiguity
3. **`select.epoll.unregister()`**: No longer ignores `EBADF` errors
4. **`bz2.BZ2File`**: `compresslevel` is now keyword-only
5. **AST changes**: Simplified subscription AST - `Index(value)` returns value directly
6. **ftplib encoding**: Default changed from Latin-1 to UTF-8
7. **logging.getLogger('root')**: Returns the actual root logger
8. **Array type 'u'**: Now uses `wchar_t` internally

### Compatibility Notes

Most Python 3.8 code will run unchanged in 3.9. Key areas requiring attention:

- Code using removed deprecated APIs
- C extensions with custom `tp_traverse` (must visit type object)
- Code relying on old parser internals
- IPv4 string parsing with leading zeros
- Direct use of removed bytecode instructions

### Testing Recommendations

- Run tests with `-W default` to expose deprecation warnings
- Use Python Development Mode (`-X dev`) to catch encoding issues
- Test with PEG parser (default) to ensure compatibility before 3.10
- For C extensions, review traverse functions for heap types

## Key Takeaways

1. **More expressive syntax**: Dictionary merge operators and relaxed decorators make code cleaner and more intuitive
2. **Type hints simplified**: Built-in generic types reduce boilerplate and improve readability
3. **World-class time zone support**: `zoneinfo` provides production-ready IANA database integration
4. **Cleaner string manipulation**: `removeprefix()` and `removesuffix()` are safer than slicing
5. **Performance gains across the board**: Vectorcall, optimized GC, and faster builtins provide measurable speedups
6. **Foundation for the future**: PEG parser enables upcoming syntax innovations in 3.10+
7. **End of Python 2 compatibility**: Last release maintaining backward compatibility layers

Python 3.9 delivers practical improvements that benefit everyday coding while establishing infrastructure for future enhancements. The combination of ergonomic syntax features, strengthened type system, and performance optimizations makes it a compelling upgrade from earlier 3.x versions.
