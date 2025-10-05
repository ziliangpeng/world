# Python Version History

A comprehensive guide to Python's evolution, major features by version, and practical guidance for version selection.

## Python 1.x (1994-2000)

**Python 1.0** - January 1994
- First official release
- Lambda, map, filter, reduce (functional programming)
- Exception handling
- Core data types: list, dict, str
- Modules and packages

**Python 1.5** - December 1997
- Standard library enhancements
- Initial Unicode support

## Python 2.x (2000-2020)

**Python 2.0** - October 2000
- List comprehensions
- Cycle-detecting garbage collector
- Reference counting and memory management
- Full Unicode support

**Python 2.2** - December 2001
- Unification of types and classes into one hierarchy
- Pure object-oriented model
- Generators introduced
- `//` floor division operator

**Python 2.5** - September 2006
- `with` statement (context managers)
- Ternary conditional expression: `x if condition else y`
- `try/except/finally` combined

**Python 2.7** - July 2010 | **EOL: January 1, 2020**
- Final 2.x release
- Dictionary comprehensions
- Set literals
- Multiple context managers in one `with` statement
- `collections.OrderedDict`
- Backported features from Python 3.x to ease migration

## Python 3.0-3.5 (2008-2015)

**Python 3.0** - December 3, 2008
- Major backwards-incompatible rewrite
- `print()` as a function instead of statement
- Text (str) vs data (bytes) separation
- Integer division returns float by default
- Iterators instead of lists (range, zip, map, dict methods)
- Unicode by default

**Python 3.4** - March 2014
- Asyncio framework for asynchronous programming
- Pathlib - object-oriented filesystem paths
- Enum type
- `statistics` module

**Python 3.5** - September 2015
- Type hints with `typing` module (PEP 484)
- `async` and `await` keywords for coroutines
- Matrix multiplication operator `@`
- Unpacking generalizations: `[*a, *b]`, `{**d1, **d2}`

## Python 3.6 (2016-2018)

**Python 3.6** - December 2016 | **EOL: December 23, 2021**

### Major Features

**F-strings (Formatted String Literals)**
```python
name = "Alice"
age = 30
f"Hello, {name}! You are {age} years old."
```

**Underscores in Numeric Literals**
```python
population = 1_000_000_000
hex_value = 0xFF_FF_FF
```

**Variable Annotations**
```python
primes: List[int] = []
captain: str  # Type annotation without assignment
```

**Async Generators and Comprehensions**
```python
async def ticker(delay, to):
    for i in range(to):
        yield i
        await asyncio.sleep(delay)

# Async comprehension
result = [i async for i in aiter() if i % 2]
```

**Preserves Dictionary Insertion Order** (implementation detail, became official in 3.7)

## Python 3.7 (2018-2023)

**Python 3.7** - June 2018 | **EOL: June 27, 2023**

- Dictionary order preservation guaranteed (language specification)
- Data classes with `@dataclass` decorator
- `breakpoint()` built-in for debugging
- Context variables for async contexts
- `async` and `await` became reserved keywords
- Postponed annotation evaluation (string annotations)

## Python 3.8 (2019-2024)

**Python 3.8** - October 2019 | **EOL: October 7, 2024**

### Major Features

**Walrus Operator `:=` (Assignment Expressions)**
```python
# Before
n = len(data)
if n > 10:
    print(f"List too long: {n}")

# After
if (n := len(data)) > 10:
    print(f"List too long: {n}")
```

**Positional-Only Parameters**
```python
def f(a, b, /, c, d, *, e, f):
    # a, b: positional-only
    # c, d: positional or keyword
    # e, f: keyword-only
    pass
```

**F-String Debugging**
```python
user = 'eric_idle'
f'{user=}'  # Outputs: "user='eric_idle'"
```

**`typing.TypedDict`, `typing.Literal`, `typing.Final`**

## Python 3.9 (2020-2025)

**Python 3.9** - October 2020 | **Security Support Ends: October 31, 2025**

### Major Features

**Dictionary Merge Operators**
```python
x = {"a": 1, "b": 2}
y = {"b": 3, "c": 4}
z = x | y  # {'a': 1, 'b': 3, 'c': 4}

x |= y  # In-place merge
```

**Built-in Generic Types**
```python
# Before
from typing import List, Dict, Tuple
def greet_all(names: List[str]) -> None:
    pass

# After - use built-ins directly
def greet_all(names: list[str]) -> None:
    pass
```

**String Methods: `removeprefix()` and `removesuffix()`**
```python
"HelloWorld".removeprefix("Hello")  # "World"
"test.txt".removesuffix(".txt")     # "test"
```

**New PEG Parser** - More flexible for future language features

## Python 3.10 (2021-2026)

**Python 3.10** - October 2021 | **Security Support Ends: October 31, 2026**

### Major Features

**Structural Pattern Matching**
```python
match status:
    case 400:
        return "Bad request"
    case 404:
        return "Not found"
    case 418:
        return "I'm a teapot"
    case _:
        return "Unknown"

# Pattern matching with data structures
match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y-axis at {y}")
    case (x, 0):
        print(f"X-axis at {x}")
    case (x, y):
        print(f"Point at ({x}, {y})")
```

**Parenthesized Context Managers**
```python
with (
    open('file1.txt') as f1,
    open('file2.txt') as f2,
):
    pass
```

**Better Error Messages**
- More precise syntax error locations
- Suggestions for typos in attribute/variable names

**Type Union Operator `|`**
```python
def square(number: int | float) -> int | float:
    return number ** 2
```

## Python 3.11 (2022-2027)

**Python 3.11** - October 2022 | **Security Support Ends: October 31, 2027**

### Major Features

**Performance Improvements: 10-60% Faster**
- Specialized adaptive interpreter
- Faster function calls
- Faster imports
- Zero-cost exceptions (try/except blocks)

**Enhanced Error Messages**
```python
# Shows exactly which part of expression failed
d = {"a": {"b": 1}}
print(d["a"]["c"]["d"])
# Error points specifically to the missing key 'c'
```

**Exception Groups and `except*`**
```python
try:
    # Code that might raise multiple exceptions
    pass
except* ValueError as eg:
    # Handle group of ValueErrors
    pass
except* KeyError as eg:
    # Handle group of KeyErrors
    pass
```

**`tomllib` for TOML parsing** (standard library)

**Fine-grained Error Locations** in tracebacks

## Python 3.12 (2023-2028)

**Python 3.12** - October 2023 | **Active Support Ended: April 2, 2025** | **Security Support Ends: October 31, 2028**

### Major Features

**Performance Improvements**
- Incremental garbage collection (reduces pause times for large heaps)
- 30% faster `textwrap.indent()` for large inputs
- Faster `import` for typing module (~33% reduction)

**Per-Interpreter GIL** (preparation for sub-interpreters)

**Improved Type System**
- `type` statement for type aliases
- Generic type parameter syntax

**Better Error Messages** continue to improve

**`sys.monitoring` API** for profiling and debugging tools

## Python 3.13 (2024-2029)

**Python 3.13** - October 7, 2024 | **Active Support Ends: October 1, 2026** | **Security Support Ends: October 31, 2029**

### Major Features

**Experimental Free-Threaded Mode (No-GIL)** - PEP 703
```bash
# Build Python without Global Interpreter Lock
python --disable-gil
```
- Allows true parallel execution of Python threads
- Experimental: May have compatibility issues
- Major step toward removing GIL entirely

**Experimental JIT Compiler** - PEP 744
- Just-In-Time compilation for performance
- Lays groundwork for significant future speed improvements
- Currently provides modest gains, expected to improve

**New Interactive Interpreter (REPL)**
- Based on PyPy code
- Multiline editing with history
- Color-enabled prompts and tracebacks by default
- Direct REPL commands: `help`, `exit`, `quit` (without parentheses)

**Improved Error Messages**
- Colored tracebacks by default
- Even more precise error locations

**Type System Enhancements**
- Type parameter defaults
- Better support for generic types

**`locals()` Semantics** - Defined behavior for modifying returned mapping

**Extended Support Timeline**: Starting with 3.13, versions receive 2 years of full support (up from 1.5 years) + 3 years of security support

---

**Note:** For practical guidance on version selection, migration strategies, and support timelines, see [Python Version Selection Guide](python_version_guide.md).
