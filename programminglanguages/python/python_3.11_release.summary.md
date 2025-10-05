# Python 3.11 Release Summary

**Released:** October 24, 2022
**Source:** [Official Python 3.11 Release Notes](https://docs.python.org/3/whatsnew/3.11.html)

## Overview

Python 3.11 represents a major performance milestone, delivering 10-60% faster execution than Python 3.10 (25% average speedup) through the Faster CPython project. This release introduces powerful exception handling with exception groups, significantly improved error messages with fine-grained tracebacks, important type system enhancements including variadic generics and the Self type, and a new TOML parser in the standard library. The performance improvements come from adaptive specialization (PEP 659), cheaper frame objects, inlined function calls, and frozen imports for faster startup.

## Major Language Features

### PEP 654: Exception Groups and except*

New syntax for handling multiple unrelated exceptions simultaneously:

```python
# Raising multiple exceptions together
raise ExceptionGroup("multiple errors", [
    ValueError("invalid value"),
    TypeError("wrong type"),
    KeyError("missing key")
])

# Handling exception groups with except*
try:
    some_operation()
except* ValueError as e:
    # Handle all ValueError instances in the group
    print(f"Value errors: {e.exceptions}")
except* TypeError as e:
    # Handle all TypeError instances in the group
    print(f"Type errors: {e.exceptions}")
```

Exception groups enable better error handling for concurrent operations like asyncio task groups, where multiple operations can fail independently.

### PEP 678: Enriching Exceptions with Notes

Exceptions can now be annotated with additional context after creation:

```python
try:
    process_data(user_input)
except ValueError as e:
    e.add_note(f"Processing failed for user {user_id}")
    e.add_note(f"Input was: {user_input}")
    raise
```

Notes appear in the traceback, making debugging easier by providing context that wasn't available when the exception was originally raised.

### PEP 657: Fine-Grained Error Locations in Tracebacks

Tracebacks now point to the exact expression that caused an error:

```python
# Old behavior (Python 3.10):
Traceback (most recent call last):
  File "script.py", line 12, in <module>
    result = obj.method(x, y, z)
TypeError: 'NoneType' object has no attribute 'method'

# New behavior (Python 3.11):
Traceback (most recent call last):
  File "script.py", line 12, in <module>
    result = obj.method(x, y, z)
             ^^^
AttributeError: 'NoneType' object has no attribute 'method'
```

The enhanced tracebacks use carets (^) to highlight the specific sub-expression, making it immediately clear which variable is None or which operation failed.

## Type Hint Enhancements

### PEP 646: Variadic Generics

TypeVarTuple enables type-checking for array shapes and variable-length type parameters:

```python
from typing import TypeVarTuple

Ts = TypeVarTuple('Ts')

# Type-check array shapes
class Array(Generic[*Ts]):
    def __init__(self, shape: tuple[*Ts]):
        self.shape = shape

# Now type checkers can verify shape compatibility
arr1: Array[int, int] = Array((3, 4))  # 2D array
arr2: Array[int, int, int] = Array((2, 3, 4))  # 3D array

# Variable-length generic functions
def args_to_tuple(*args: *Ts) -> tuple[*Ts]: ...
```

This is particularly valuable for numerical computing libraries like NumPy and TensorFlow, enabling shape-aware type checking.

### PEP 655: Required and NotRequired for TypedDict

Fine-grained control over which TypedDict fields are required:

```python
from typing import TypedDict, Required, NotRequired

# All fields required by default
class Movie(TypedDict):
    title: str
    year: NotRequired[int]  # This field is optional

# With total=False, all fields are optional by default
class Config(TypedDict, total=False):
    debug: Required[bool]  # This field is required
    timeout: int  # This is optional
```

### PEP 673: Self Type

The Self annotation simplifies type hints for methods returning their own class:

```python
from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:
        self.name = name
        return self

    def set_age(self, age: int) -> Self:
        self.age = age
        return self

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls().set_name(data['name'])

# Fluent interfaces work correctly with type checkers
builder = Builder().set_name("Alice").set_age(30)
```

Previously, you'd need complex TypeVar-based solutions. Self makes this pattern simple and intuitive.

### PEP 675: LiteralString Type

Helps prevent injection attacks by ensuring only literal strings are used:

```python
from typing import LiteralString

def execute_query(sql: LiteralString) -> list:
    return database.execute(sql)

# Type checker allows this
execute_query("SELECT * FROM users WHERE id = 1")

# Type checker rejects this (prevents SQL injection)
user_input = input("Enter query: ")
execute_query(user_input)  # Error!

# But this is allowed (composed from literals)
table = "users"
execute_query(f"SELECT * FROM {table}")  # OK
```

### PEP 681: Dataclass Transforms

Decorator for marking functions/classes that create dataclass-like behavior:

```python
from typing import dataclass_transform

@dataclass_transform()
def create_model(cls):
    # Add __init__, __eq__, etc.
    return enhanced_cls

@create_model
class User:
    id: int
    name: str

# Type checkers understand User has dataclass-like behavior
user = User(id=1, name="Alice")  # Type-safe
```

## Interpreter Improvements

### Faster Startup (10-15% improvement)

Core modules are now "frozen" - their bytecode is statically allocated in the interpreter rather than loaded from disk:

**Python 3.10 startup:**
```
Read __pycache__ → Unmarshal → Heap allocated code object → Evaluate
```

**Python 3.11 startup:**
```
Statically allocated code object → Evaluate
```

This dramatically improves startup time for short-running scripts and CLI tools.

### Zero-Cost Exception Handling

Try blocks now have zero overhead when no exception is raised. Previously, just having a try block would slow code down; now exceptions are truly "exceptional" - they only impact performance when actually raised.

### -P Flag and PYTHONSAFEPATH

New security feature prevents automatically prepending potentially unsafe paths to sys.path:

```bash
# Prevents local directory from shadowing stdlib modules
python -P script.py

# Also available as environment variable
export PYTHONSAFEPATH=1
```

This prevents accidental or malicious module shadowing attacks.

## Performance Optimizations

### Faster CPython Project: 10-60% Speedup

Python 3.11 is **25% faster on average** than Python 3.10, with some workloads seeing 60% improvements.

### PEP 659: Adaptive Specializing Interpreter

The interpreter now adapts to your code at runtime through "inline caching" and "specialization":

**How it works:**
1. CPython monitors "hot" code (executed multiple times)
2. When types are stable, it replaces generic operations with specialized ones
3. Specialized operations use fast paths for specific types
4. If types change, it de-specializes and tries again

**Specialized operations:**

| Operation | Example | Speedup | Technique |
|-----------|---------|---------|-----------|
| Binary ops | `x + y` | 10% | Type-specific addition for int, float, str |
| Subscript | `lst[i]` | 10-25% | Direct indexing for list/tuple/dict |
| Attribute access | `obj.attr` | Major | Cache attribute location, zero lookups |
| Method calls | `obj.method()` | 10-20% | Cache method address |
| Builtin calls | `len(x)` | 20% | Direct C function call |
| Global lookup | `print` | Major | Cache index, zero namespace lookups |

### Cheaper, Lazy Python Frames (3-7% speedup)

Function call frames are now:
- Streamlined to contain only essential information
- Reused from C stack space (avoiding allocation)
- Created lazily (only when debugger or introspection needs them)

Most Python code never creates actual frame objects, resulting in faster function calls across the board.

### Inlined Python Function Calls (1-3% speedup)

When Python code calls Python code, the interpreter now "jumps" directly to the new function instead of calling through C:

```python
# Recursive fibonacci is 1.7x faster
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)  # These calls are now inlined
```

This also allows much deeper recursion (when recursion limit is increased).

### Other Optimizations

- **printf-style % formatting** is now as fast as f-strings (for simple cases)
- **Integer division** (`//`) is ~20% faster
- **sum()** is ~30% faster for small integers
- **list.append()** is ~15% faster
- **List comprehensions** are 20-30% faster
- **Dictionary size** reduced 23% when all keys are strings
- **math.comb() and math.perm()** are ~10x faster
- **statistics functions** now process iterators in one pass (2x faster, much less memory)
- **Unicode normalization** for ASCII strings is now constant time
- **re module** is ~10% faster with computed gotos

## Improved Error Messages

Building on Python 3.10's improved error messages:

```python
>>> sys.version_info
NameError: name 'sys' is not defined. Did you forget to import 'sys'?

>>> class A:
...     def __init__(self):
...         self.value = 1
...     def method(self):
...         print(val)
>>> A().method()
NameError: name 'val' is not defined. Did you mean: 'self.value'?

>>> from collections import chainmap
ImportError: cannot import name 'chainmap' from 'collections'. Did you mean: 'ChainMap'?
```

## Standard Library Improvements

### tomllib - New TOML Parser

```python
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)
```

TOML is now supported natively for configuration files (read-only parser).

### asyncio Enhancements

```python
# TaskGroup for structured concurrency
async with asyncio.TaskGroup() as tg:
    task1 = tg.create_task(fetch_data())
    task2 = tg.create_task(process_data())
# Waits for all tasks, propagates exceptions as ExceptionGroup

# Timeout context manager
async with asyncio.timeout(10):
    await long_operation()

# Runner for more control
with asyncio.Runner() as runner:
    runner.run(main())

# Barrier synchronization primitive
barrier = asyncio.Barrier(3)
```

### contextlib

```python
# Non-parallel-safe directory changer
with contextlib.chdir('/tmp'):
    # Do work in /tmp
    pass
# Back to original directory
```

### sqlite3

- New command-line interface: `python -m sqlite3`

### sys

New functions for int/str conversion limits (CVE-2020-10735 mitigation):
- `sys.set_int_max_str_digits()`
- `sys.get_int_max_str_digits()`

### typing

- `assert_never()` for exhaustiveness checking
- `assert_type()` for testing type checker understanding
- `reveal_type()` for debugging type inference
- `LiteralString`, `Self`, `TypeVarTuple`, `Unpack`
- `Required`, `NotRequired` for TypedDict
- `dataclass_transform()` decorator

## New Modules

- **tomllib**: TOML parser
- **wsgiref.types**: WSGI types for static type checking

## Important Removals and Deprecations

### Removed

- `@asyncio.coroutine()` decorator (use `async def`)
- `asyncio.coroutines.CoroWrapper`
- `binhex` module and related binascii functions
- `distutils.bdist_msi` command
- Various deprecated inspect functions (`getargspec`, `formatargspec`)
- `'U'` mode for file opening (universal newlines are always on in text mode)

### Deprecated (Removal in Python 3.13)

**PEP 594** - Many legacy modules deprecated:
- `aifc`, `audioop`, `cgi`, `cgitb`, `chunk`, `crypt`
- `imghdr`, `mailcap`, `msilib`, `nis`, `nntplib`
- `ossaudiodev`, `pipes`, `sndhdr`, `spwd`, `sunau`
- `telnetlib`, `uu`, `xdrlib`

These modules are outdated, unmaintained, or have better alternatives.

## CPython Implementation Changes

### Bytecode Changes

**New opcodes:**
- `ASYNC_GEN_WRAP`, `RESUME`, `RETURN_GENERATOR`
- `SEND`, `JUMP_BACKWARD_NO_INTERRUPT`
- `MAKE_CELL`, `COPY_FREE_VARS`
- `PUSH_NULL`, `PRECALL`, `CALL`, `KW_NAMES`
- `POP_JUMP_FORWARD_IF_*`, `POP_JUMP_BACKWARD_IF_*`
- `CHECK_EXC_MATCH`, `CHECK_EG_MATCH`

**Removed opcodes:**
- `COPY_DICT_WITHOUT_KEYS`, `GEN_START`, `POP_BLOCK`
- `SETUP_FINALLY`, `YIELD_FROM`

**Changed:**
- Many opcodes adapted for specialization (PEP 659)
- Instructions can have inline cache entries

### Internal Hash Algorithm

New `siphash13` hash algorithm for strings, bytes, and other types - similar security to `siphash24` but faster for long inputs.

### Frame Object Changes

- Frames are now "lazy" - only materialized when needed
- Internal frame structure streamlined significantly
- Re-raises preserve traceback correctly

### CVE-2020-10735 Mitigation

Converting between int and str in non-power-of-2 bases (like decimal) now has limits to prevent DoS attacks. Default limit is 4300 digits.

## C API Changes

### PEP 670: Macro to Static Inline Function

Many macros converted to static inline functions for type safety. When limited API >= 3.11, casts are not automatic.

### New Functions

- `PyType_GetName()`, `PyType_GetQualName()`
- `PyType_GetModuleByDef()`
- `PyThreadState_EnterTracing()`, `PyThreadState_LeaveTracing()`
- `PyErr_GetHandledException()`, `PyErr_SetHandledException()`
- `PyFrame_GetBuiltins()`, `PyFrame_GetGenerator()`, etc.
- `PyFloat_Pack2/4/8()`, `PyFloat_Unpack2/4/8()`
- Buffer protocol now in limited API

### Removed C APIs (PEP 624)

- `Py_UNICODE` encoder APIs completely removed
- Use UTF-8 encoding functions instead

## Build Changes

### WebAssembly Support (Tier 3)

CPython now has experimental support for:
- **Emscripten** (wasm32-unknown-emscripten) - Python in browsers
- **WASI** (wasm32-unknown-wasi) - WebAssembly System Interface

### Requirements

- C11 compiler now required
- IEEE 754 floating-point required
- Tcl/Tk 8.5.12+ required for tkinter

### Build Improvements

- ThinLTO support: `--with-lto=thin`
- Freelists can be disabled: `--without-freelists`
- Better pkg-config integration
- 30-bit digits now default for Python int (was 15-bit on some platforms)

## Migration Notes

### Breaking Changes

1. **'U' mode removed** from file opening - always use text mode
2. **asyncio.coroutine removed** - use `async def`
3. **Legacy modules removed** - binhex, various inspect functions
4. **random.shuffle()** no longer accepts custom random function
5. **random.sample()** no longer auto-converts sets to lists
6. **Global inline regex flags** must be at start of pattern
7. **Bytes on sys.path** no longer supported
8. **ThreadPoolExecutor required** for asyncio default executor

### Security Updates

- New `-P` flag prevents local directory in sys.path
- Int/str conversion limits prevent DoS attacks
- LiteralString type helps prevent injection vulnerabilities

### Compatibility

Most Python 3.10 code will work unchanged in 3.11, except for:
- Code depending on removed modules/functions
- C extensions needing updates for new frame internals
- Code relying on specific bytecode instructions

## Key Takeaways

1. **Massive performance gains** - 25% average speedup, up to 60% for some workloads through adaptive specialization
2. **Production-ready speed improvements** - no JIT needed, works with existing Python code
3. **Better error handling** - exception groups enable structured error handling for concurrent code
4. **Superior debugging** - fine-grained tracebacks pinpoint exact error locations
5. **Type system maturity** - variadic generics, Self, LiteralString, and Required/NotRequired fill major gaps
6. **Faster startup** - frozen imports make CLI tools and short scripts noticeably snappier
7. **Security hardening** - safe path mode, injection prevention types, DoS mitigations
8. **Modern infrastructure** - WebAssembly support, C11 requirement, legacy cleanup

Python 3.11 represents the most significant performance improvement in CPython history while maintaining full backwards compatibility. The Faster CPython project's adaptive specialization technique delivers substantial speed improvements without requiring any code changes. Combined with powerful new language features like exception groups and dramatic error message improvements, Python 3.11 is a compelling upgrade for all Python users.
