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

## Python 3.0 (2008-2009)

**Python 3.0** - December 3, 2008 | **EOL: June 27, 2009**

Python 3.0 "Py3K" is the first ever intentionally backwards incompatible Python release - a comprehensive language redesign that fundamentally restructures Python around clean text/bytes separation, removes accumulated legacy features, and establishes a solid foundation for the language's future.

**Major Highlights:**
- `print()` as a function (not statement) with keyword arguments for flexibility
- Text vs. bytes separation - all text is Unicode (`str`), binary data is `bytes`, no implicit mixing
- Views and iterators everywhere - `dict.keys()`, `map()`, `filter()`, `range()`, `zip()` return views/iterators, not lists
- Integer division returns float - `1/2` → `0.5`, use `1//2` for floor division
- Integers unified - only `int` type with unlimited precision, `long` removed
- Strict ordering comparisons - `1 < ''` raises TypeError instead of arbitrary ordering
- Classic classes removed - all classes are new-style by default

```python
# Print as function with keyword arguments
print("There are <", 2**32, "> possibilities!", sep="")

# Text vs. binary data
s = "hello"           # str type (Unicode)
b = b"hello"          # bytes type
s.encode('utf-8')     # Returns bytes
b.decode('utf-8')     # Returns str

# Extended iterable unpacking
a, *rest = [1, 2, 3, 4]  # a = 1, rest = [2, 3, 4]
```

## Python 3.1 (2009-2012)

**Python 3.1** - June 27, 2009 | **EOL: April 9, 2012**

Python 3.1 made Python 3 practical for real-world use with critical performance fixes that addressed Python 3.0's biggest bottleneck - the I/O library was completely rewritten in C for 2-20x speedup.

**Major Highlights:**
- I/O library rewritten in C (PEP 3116) - 2-20x faster I/O operations
- `OrderedDict` (PEP 372) - dictionary that preserves insertion order
- Improved float representation - `repr(1.1)` now returns `'1.1'` instead of `'1.1000000000000001'`
- unittest improvements - test skipping, expected failures, and context manager assertions
- `importlib` module - pure Python reference implementation of import machinery
- `collections.Counter` - convenient counting of items in sequences
- Multiple context managers - `with open(...) as f1, open(...) as f2:` syntax

## Python 3.2 (2011-2016)

**Python 3.2** - February 20, 2011 | **EOL: February 20, 2016**

Python 3.2 is the release that made Python 3 production-ready, fixing critical bytes/text handling issues and adding essential infrastructure for Python packaging and deployment.

**Major Highlights:**
- `concurrent.futures` (PEP 3148) - high-level parallelism with ThreadPoolExecutor and ProcessPoolExecutor
- `argparse` module (PEP 389) - modern command-line parsing replacing optparse
- Stable ABI (PEP 384) - C extensions work across Python versions without recompilation
- PYC repository directories (PEP 3147) - bytecode cached in `__pycache__` with interpreter-specific names
- Email package fixed - first Python 3 release with working email/mailbox for bytes and mixed encodings
- SSL/TLS overhaul - proper certificate validation and modern SSL features throughout network modules
- Dictionary-based logging (PEP 391) - configure logging from JSON/YAML files

## Python 3.3 (2012-2017)

**Python 3.3** - September 29, 2012 | **EOL: September 29, 2017**

Python 3.3 modernizes the language with better Unicode handling, virtual environments in the standard library, and improved migration from Python 2.

**Major Highlights:**
- `yield from` expression (PEP 380) - delegate generator operations to subgenerators cleanly
- `venv` module (PEP 405) - virtual environments built into the standard library
- Flexible string representation (PEP 393) - 2-3x memory reduction for Unicode strings with variable-width encoding
- Implicit namespace packages (PEP 420) - no `__init__.py` required for namespace packages
- Reworked exception hierarchy (PEP 3151) - new specific exception types like FileNotFoundError, PermissionError
- Import system rewrite - import machinery now based on importlib with per-module locks
- `u'unicode'` literals return (PEP 414) - eases Python 2 to 3 migration

## Python 3.4 (2014-2019)

**Python 3.4** - March 16, 2014 | **EOL: March 18, 2019**

Python 3.4 focused on infrastructure improvements and developer experience with no new syntax, bringing pip by default and the foundational asyncio framework.

**Major Highlights:**
- pip bundled by default (PEP 453) - Python installations now include pip, solving a major pain point for newcomers
- `asyncio` module (PEP 3156) - standard asynchronous I/O framework, foundation for modern async Python
- `pathlib` module (PEP 428) - object-oriented filesystem paths, modern alternative to os.path
- `enum` module (PEP 435) - standard enumeration types for better code clarity
- Non-inheritable file descriptors (PEP 446) - security improvement preventing fd leaks to child processes
- Safe object finalization (PEP 442) - finalizers work with reference cycles, module globals no longer set to None on shutdown
- `statistics` module (PEP 450) - built-in numerically stable statistics functions

## Python 3.5 (2015-2020)

**Python 3.5** - September 13, 2015 | **EOL: September 30, 2020**

Python 3.5 is a groundbreaking release for asynchronous programming and scientific computing with revolutionary dedicated syntax for coroutines.

**Major Highlights:**
- `async`/`await` syntax (PEP 492) - dedicated syntax for coroutines making async code as readable as sync code
- Matrix multiplication operator `@` (PEP 465) - cleaner syntax for scientific computing: `S = (H @ beta - r).T @ inv(H @ V @ H.T)`
- Type hints (PEP 484) - standard framework for type annotations with new `typing` module
- `os.scandir()` (PEP 471) - 3-5x faster directory traversal on POSIX, 7-20x on Windows
- Unpacking generalizations (PEP 448) - multiple `*` and `**` unpacking in calls and literals
- OrderedDict 4-100x faster - reimplemented in C for dramatic performance improvements
- Automatic EINTR retry (PEP 475) - system calls automatically retry on signal interruption

```python
# async/await syntax
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async for chunk in session.get(url):
            await process(chunk)

# Matrix multiplication operator
S = (H @ beta - r).T @ inv(H @ V @ H.T)

# Multiple unpacking
print(*[1], *[2], 3)
{'x': 1, **{'y': 2}}
```

## Python 3.6 (2016-2021)

**Python 3.6** - December 23, 2016 | **EOL: December 23, 2021**

Python 3.6 introduced game-changing syntax features and critical infrastructure improvements that fundamentally changed how Python code is written.

**Major Highlights:**
- F-strings (PEP 498) - formatted string literals revolutionized string formatting: `f"Hello {name}"`
- Variable type annotations (PEP 526) - complete type hint support: `primes: List[int] = []`
- Async generators and comprehensions (PEP 525/530) - `async for` in comprehensions, `yield` in async functions
- Underscores in numeric literals (PEP 515) - improved readability: `1_000_000`
- Compact dictionaries - 20-25% memory reduction with insertion-order preservation (implementation detail)
- asyncio stabilized - graduated from provisional to stable API with 30% performance boost
- Windows UTF-8 (PEP 528/529) - console and filesystem encoding changed to UTF-8
- `secrets` module (PEP 506) - cryptographically secure random numbers for security-sensitive code

```python
# F-strings
name = "Alice"
age = 30
f"Hello, {name}! You are {age} years old."
f"result: {value:{width}.{precision}}"  # Nested expressions

# Variable type annotations
primes: List[int] = []
captain: str  # No initial value needed

# Underscores in numeric literals
population = 1_000_000_000
hex_value = 0xFF_FF_FF

# Async generators
async def ticker(delay, to):
    for i in range(to):
        yield i
        await asyncio.sleep(delay)
```

## Python 3.7 (2018-2023)

**Python 3.7** - June 27, 2018 | **EOL: June 27, 2023**

Python 3.7 focused on developer productivity, async programming, and performance with method calls 20% faster through bytecode optimizations.

**Major Highlights:**
- Dataclasses (PEP 557) - `@dataclass` decorator automatically generates `__init__`, `__repr__`, `__eq__`, and other methods
- Context variables (PEP 567) - async-aware state management (like thread-local but works correctly with async code)
- Built-in `breakpoint()` (PEP 553) - standardized debugger entry point
- `async` and `await` reserved keywords - breaking change requiring code updates if used as identifiers
- Dict ordering official - insertion-order preservation now part of language specification
- Nanosecond time functions (PEP 564) - `time.time_ns()` and 5 other functions with better precision
- Method calls 20% faster - bytecode optimizations avoiding bound method creation

```python
# Dataclasses
@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0

    def distance(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5
```

## Python 3.8 (2019-2024)

**Python 3.8** - October 14, 2019 | **EOL: October 7, 2024**

Python 3.8 introduces powerful language features for readability and expressiveness, with the walrus operator becoming one of Python's most useful recent additions.

**Major Highlights:**
- Walrus operator `:=` (PEP 572) - assignment expressions for cleaner code: `if (n := len(a)) > 10`
- Positional-only parameters (PEP 570) - function syntax with `/` separator for stricter APIs
- F-string `=` specifier - self-documenting expressions for debugging: `f'{user=} {value=}'`
- `typing.TypedDict` (PEP 589) - per-key type annotations for dictionaries
- `typing.Literal` (PEP 586) - constrain values to specific literals
- `typing.Final` (PEP 591) - mark variables/methods as final (no override/reassignment)
- Pickle protocol 5 (PEP 574) - out-of-band data buffers for better large-data serialization

```python
# Walrus operator
if (n := len(data)) > 10:
    print(f"List too long: {n}")

# Positional-only parameters
def f(a, b, /, c, d, *, e, f):
    # a, b: positional-only
    # c, d: positional or keyword
    # e, f: keyword-only
    pass

# F-string = specifier
user = 'eric_idle'
f'{user=}'  # Outputs: "user='eric_idle'"
```

## Python 3.9 (2020-2025)

**Python 3.9** - October 5, 2020 | **Security Support Ends: October 31, 2025**

Python 3.9 focuses on syntax improvements and standard library additions for common use cases, with dictionary merge operators and built-in generics making code cleaner.

**Major Highlights:**
- Dictionary merge operator `|=` and `|` (PEP 584) - convenient syntax for merging dicts: `x | y`
- Type hinting with built-in generics (PEP 585) - use `list[str]` instead of `typing.List[str]`
- String prefix/suffix removal methods (PEP 616) - `str.removeprefix()` and `str.removesuffix()`
- `zoneinfo` module (PEP 615) - IANA Time Zone Database in standard library
- `graphlib` module - topological sorting for graphs
- PEG parser (PEP 617) - new parser replacing LL(1), enables future syntax flexibility
- Annual release cycle (PEP 602) - CPython adopts yearly releases

```python
# Dictionary merge operators
x = {"a": 1, "b": 2}
y = {"b": 3, "c": 4}
z = x | y  # {'a': 1, 'b': 3, 'c': 4}
x |= y  # In-place merge

# Built-in generic types
def greet_all(names: list[str]) -> None:
    for name in names:
        print("Hello", name)

# String prefix/suffix removal
"HelloWorld".removeprefix("Hello")  # "World"
"test.txt".removesuffix(".txt")     # "test"
```

## Python 3.10 (2021-2026)

**Python 3.10** - October 4, 2021 | **Security Support Ends: October 31, 2026**

Python 3.10 brings major syntax improvements and enhanced developer experience with structural pattern matching as the headline feature.

**Major Highlights:**
- Structural Pattern Matching (PEP 634-636) - `match`/`case` statements for powerful pattern matching on data structures
- Union type operator (PEP 604) - write `int | str` instead of `Union[int, str]` for cleaner type hints
- Better error messages - precise location of syntax errors, suggestions for typos in NameError and AttributeError
- Parenthesized context managers - multi-line `with` statements now allowed with parentheses
- Precise line numbers (PEP 626) - reliable line numbers for debugging, profiling, and coverage tools
- Performance improvements - `LOAD_ATTR` 36-44% faster, str/bytes/bytearray constructors 30-40% faster
- `distutils` deprecated (PEP 632) - scheduled for removal in Python 3.12

```python
# Structural pattern matching
match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y-axis at {y}")
    case (x, 0):
        print(f"X-axis at {x}")
    case (x, y):
        print(f"Point at ({x}, {y}")

# Union type operator
def square(number: int | float) -> int | float:
    return number ** 2

# Better error messages
>>> collections.namedtoplo
AttributeError: module 'collections' has no attribute 'namedtoplo'.
Did you mean: 'namedtuple'?
```

## Python 3.11 (2022-2027)

**Python 3.11** - October 24, 2022 | **Security Support Ends: October 31, 2027**

Python 3.11 brings dramatic performance improvements and significantly better error messages with a 10-60% speedup over Python 3.10.

**Major Highlights:**
- 10-60% faster than Python 3.10 (PEP 659) - average 1.25x speedup through adaptive specializing interpreter
- Exception Groups and `except*` (PEP 654) - handle multiple unrelated exceptions simultaneously
- Fine-grained error locations (PEP 657) - tracebacks point to exact expressions, not just lines
- TOML parsing (PEP 680) - built-in `tomllib` module for TOML configuration files
- Faster startup (10-15%) - core modules frozen with statically allocated code objects
- Enhanced type hints - variadic generics, Self type, LiteralString, Required/NotRequired for TypedDict
- Exception notes (PEP 678) - add contextual information to exceptions with `add_note()`

```python
# Exception Groups
try:
    # Code that might raise multiple exceptions
    pass
except* ValueError as eg:
    # Handle group of ValueErrors
    pass
except* KeyError as eg:
    # Handle group of KeyErrors
    pass

# Fine-grained error locations
d = {"a": {"b": 1}}
print(d["a"]["c"]["d"])
# Error points specifically to the missing key 'c'
#           ~~~
```

## Python 3.12 (2023-2028)

**Python 3.12** - October 2, 2023 | **Security Support Ends: October 31, 2028**

Python 3.12 focuses on usability improvements for type hints, f-strings, and developer experience with cleaner generic syntax and comprehensions 2x faster.

**Major Highlights:**
- New type parameter syntax (PEP 695) - cleaner generic classes and functions: `def max[T](args: Iterable[T]) -> T`
- F-string restrictions removed (PEP 701) - can reuse quotes, use multiline expressions, and include backslashes
- Comprehensions 2x faster (PEP 709) - list/dict/set comprehensions inlined for major speedup
- `isinstance()` 2-20x faster - protocol checks dramatically accelerated
- Incremental garbage collection - reduced pause times for large heaps
- Better error messages - "Did you forget to import 'sys'?" and "Did you mean: 'self.blech'?"
- Per-interpreter GIL (PEP 684) - foundation for better parallelism (C-API only)
- `distutils` removed (PEP 632) - use setuptools or modern packaging tools

```python
# New type parameter syntax
def max[T](args: Iterable[T]) -> T:
    ...

type Point = tuple[float, float]  # Type alias

# F-string improvements
f"This is now valid: {', '.join(names)}"
f"Multiline expressions: {
    value if condition
    else other_value
}"
```

## Python 3.13 (2024-2029)

**Python 3.13** - October 7, 2024 | **Security Support Ends: October 31, 2029**

Python 3.13 is a groundbreaking release with two experimental game-changers and major quality-of-life improvements.

**Major Highlights:**
- Free-threaded mode (PEP 703) - run Python without the GIL for true parallelism (experimental)
- JIT compiler (PEP 744) - experimental just-in-time compilation for performance improvements
- Improved REPL - modern interactive interpreter with colors, multiline editing, and history browsing
- Defined `locals()` semantics (PEP 667) - clear mutation behavior for debugging and introspection
- Removed 19 "dead batteries" modules (PEP 594) - cleanup of legacy standard library modules
- iOS and Android tier 3 support - official mobile platform support
- Extended support timeline (PEP 602) - 2 years full support + 3 years security fixes (up from 1.5 + 3.5)

```python
# Free-threaded mode (experimental)
# Enable: Build with --disable-gil or use python3.13t
# Runtime control: PYTHON_GIL=0 or -X gil=0

# Improved REPL with colors and multiline editing
>>> def greet(name):
...     return f"Hello, {name}!"
...
>>> greet("World")
'Hello, World!'

# Check if GIL is enabled
>>> import sys
>>> sys._is_gil_enabled()
True
```

---

**Note:** For practical guidance on version selection, migration strategies, and support timelines, see [Python Version Selection Guide](python_version_guide.md).
