# Python 2.6 Release Notes

**Released:** October 1, 2008
**EOL:** October 29, 2013

## Major Highlights

Python 2.6 is the pivotal bridge release to Python 3.0, developed in parallel and designed to ease migration:

1. **Python 3.0 forward compatibility features** - Backported syntax and semantics to prepare codebases for migration
2. **multiprocessing package (PEP 371)** - True parallelism by circumventing the GIL with process-based execution
3. **json module** - Native JavaScript Object Notation encoding/decoding support
4. **Abstract Base Classes (PEP 3119)** - Formal interface definitions through the abc module
5. **Advanced string formatting (PEP 3101)** - str.format() method with powerful templating
6. **Migration tools** - -3 flag for Python 3.0 warnings, future_builtins, 2to3 converter
7. **with statement standard** - Context managers become core syntax (no __future__ import needed)

## New Features

### Language Syntax

- 🔴 **Syntax** with statement becomes standard syntax (PEP 343) - No longer requires __future__ import
- 🔴 **Syntax** Class decorators (PEP 3129) - Decorators now work on classes: `@decorator class A: pass`
- 🔴 **Syntax** Exception handling with 'as' keyword (PEP 3110) - `except Exception as exc:` replaces comma syntax
- 🔴 **str.format** Advanced string formatting (PEP 3101) - Powerful alternative to % operator with field names, format specifiers
- 🟡 **Syntax** Binary literals with 0b prefix (PEP 3127) - `0b101111` evaluates to 47
- 🟡 **Syntax** New octal literal 0o prefix (PEP 3127) - `0o21` alongside legacy leading-zero syntax
- 🟡 **Syntax** Byte literals b'' notation (PEP 3112) - Forward compatibility for Python 3.0's text/binary distinction
- 🟡 **builtins** bin() builtin returns binary string representations - `bin(173)` returns '0b10101101'
- 🟡 **builtins** bytes as synonym for str - Preparation for Python 3.0's bytes type
- 🟡 **builtins** bytearray type for mutable byte sequences - `bytearray([65, 66, 67])` returns bytearray(b'ABC')
- 🟡 **builtins** next() builtin retrieves next item from iterator with optional default
- 🟡 **Syntax** Directories and zip archives with __main__.py can be executed directly
- 🟡 **Syntax** Keyword arguments can follow *args in function calls
- 🟢 **Syntax** Function calls accept any mapping for **kwargs (not just dicts)

### Standard Library - New Modules

- 🔴 **multiprocessing** Process-based parallelism package (PEP 371) - Process, Pool, Manager classes with threading-like API
- 🔴 **json** JavaScript Object Notation encoding/decoding - dumps(), loads() with custom encoder/decoder support
- 🔴 **abc** Abstract Base Classes for interface definitions (PEP 3119) - ABCMeta, abstractmethod for formal protocols
- 🟡 **io** New I/O library with layered architecture (PEP 3116) - RawIOBase, BufferedIOBase, TextIOBase hierarchy
- 🟡 **numbers** Numeric type hierarchy (PEP 3141) - Number, Complex, Real, Rational, Integral ABCs
- 🟡 **fractions** Fraction class for rational number arithmetic - `Fraction(2, 3) + Fraction(2, 5)`
- 🟡 **ast** Abstract Syntax Tree representation and manipulation - parse(), dump(), literal_eval() for safe evaluation
- 🟡 **ssl** Comprehensive SSL/TLS support built on OpenSSL - Better protocol negotiation and X.509 certificate handling
- 🟡 **future_builtins** Python 3.0 versions of changed builtins - hex(), oct(), map(), filter() for migration testing

### collections Module

- 🔴 **collections** namedtuple() factory for tuple subclasses with named fields - Fields accessible by name or index
- 🔴 **collections** Abstract Base Classes for protocols - Iterable, Container, MutableMapping, Sequence, etc.
- 🟡 **collections** deque gains maxlen parameter for bounded queues
- 🟡 **collections** deque.rotate() method for rotating elements

### itertools Module

- 🟡 **itertools** product() for Cartesian products of iterables
- 🟡 **itertools** combinations() generates combinations of elements
- 🟡 **itertools** permutations() generates all permutations
- 🟡 **itertools** izip_longest() for zipping iterables of different lengths with fill values
- 🟡 **itertools** chain.from_iterable() chains iterables from a single iterable

### math Module

- 🟡 **math** isinf() and isnan() test for infinity and NaN values
- 🟡 **math** copysign() copies sign bits between numbers
- 🟡 **math** factorial() computes factorial values
- 🟡 **math** fsum() accurate floating-point summation
- 🟡 **math** acosh(), asinh(), atanh() inverse hyperbolic functions
- 🟡 **math** log1p() returns log(1+x) with better precision for small x
- 🟡 **math** trunc() truncates toward zero

### contextlib Module

- 🟡 **contextlib** contextmanager decorator for generator-based context managers
- 🟡 **contextlib** nested() for nesting multiple context managers (deprecated in 2.7)
- 🟡 **contextlib** closing() wrapper ensures close() is called

### Other Standard Library

- 🟡 **float** as_integer_ratio() method converts floats to rational (numerator, denominator) tuple
- 🟡 **tuple** index() and count() methods added
- 🟡 **property** getter, setter, and deleter decorators for convenient property definition
- 🟡 **set** Set methods accept multiple iterables - intersection(), union(), difference(), symmetric_difference()
- 🟡 **float** Conversions handle NaN, infinity on IEEE 754 platforms
- 🟡 **float** hex() method for hexadecimal representation without rounding errors
- 🟡 **complex** Constructor preserves signed zeros on supporting systems
- 🟡 **class** Classes can set __hash__ = None to indicate unhashable
- 🟡 **GeneratorExit** Now inherits from BaseException (not Exception)
- 🟡 **hasattr** No longer catches KeyboardInterrupt and SystemExit
- 🟡 **instance methods** __self__ and __func__ as synonyms for im_self and im_func

## Improvements

### Performance

- 🔴 **Performance** warnings module rewritten in C - Enables parser warnings, potentially faster startup
- 🔴 **Performance** struct module rewritten in C - Substantially faster binary data operations
- 🟡 **Performance** Type method cache reduces lookup overhead - Caches method resolutions to avoid repeated base class traversal
- 🟡 **Performance** Keyword argument optimization with pointer comparison - Faster than full string comparisons
- 🟡 **Performance** Unicode string split() 25% faster, splitlines() 35% faster
- 🟡 **Performance** with statement optimization stores __exit__ on stack
- 🟡 **Performance** Garbage collector clears internal free lists when collecting highest generation - Returns memory to OS sooner

### Language Changes

- 🟡 **Syntax** Properties support convenient decorator syntax for getter/setter/deleter
- 🟡 **Syntax** Set methods (intersection, union, etc.) accept multiple iterables
- 🟢 **float** Float conversions more robust with NaN and infinity handling
- 🟢 **complex** Better handling of signed zeros

## Migration Features (Python 3.0 Backports)

- 🔴 **Migration** -3 command-line switch enables Python 3.0 compatibility warnings
- 🔴 **Migration** sys.py3kwarning boolean indicates if Python 3.0 warnings enabled
- 🔴 **Migration** 2to3 conversion tool for automated migration assistance
- 🔴 **Migration** from __future__ import print_function converts print to function
- 🟡 **Migration** future_builtins module provides Python 3.0 versions of changed builtins
- 🟡 **Migration** Exception handling 'as' syntax (Python 3.0 compatible)
- 🟡 **Migration** Byte literals b'' and bytearray type
- 🟡 **Migration** Binary and new octal literal syntax
- 🟡 **Migration** Abstract base classes (abc module)
- 🟡 **Migration** I/O library (io module)
- 🟡 **Migration** Numbers type hierarchy (numbers module)

## Other Changes

### PEP 366: Explicit Relative Imports

- 🟡 **import** __package__ attribute fixes relative imports when using -m switch

### PEP 370: Per-user site-packages

- 🟡 **site** User-specific site directories for package installation without system access
- 🟡 **site** Unix/Mac: ~/.local/, Windows: %APPDATA%/Python

### Development Process

- 🟡 **Development** Migrated from SourceForge to Roundup issue tracker at bugs.python.org
- 🟡 **Documentation** Converted from LaTeX to reStructuredText using Sphinx
- 🟡 **Environment** -B switch and PYTHONDONTWRITEBYTECODE prevent .pyc/.pyo creation
- 🟡 **Environment** PYTHONIOENCODING controls stdin/stdout/stderr encoding

## Deprecations

### Future Removals

- 🟡 **string** String exceptions (already unsupported, will be SyntaxError in Python 3)
- 🟡 **__getslice__** __getslice__, __setslice__, __delslice__ methods (use __getitem__ with slice objects)
- 🟡 **float** float arguments to PyArgs_ParseTuple() with integer format codes (use integer types)
- 🟢 **octal** Leading-zero octal syntax will be removed in Python 3.0 - Use 0o prefix

## Implementation Details

### C API

- 🟡 **C API** Py_ssize_t used consistently for sizes and indexes
- 🟡 **C API** PyNumber_Index() and __index__() protocol for integer-like objects
- 🟡 **C API** Buffer protocol significantly revised (PEP 3118) - Multi-dimensional, typed buffers
- 🟢 **C API** PyCObject deprecated in favor of PyCapsule (in 2.7+)

### Interpreter

- 🟡 **Interpreter** Type objects gained tp_version_tag field for method caching
- 🟢 **Interpreter** Various internal refactorings and optimizations

### Build System

- 🟡 **Build** Improved support for building on various platforms
- 🟡 **Build** Better Windows build support with Visual Studio
- 🟢 **Build** Various configure script improvements

## Platform & Environment

- 🟡 **Platform** Improved platform detection and support
- 🟡 **Windows** Better Windows compatibility and native features
- 🟡 **Unix** Enhanced POSIX platform support
- 🟢 **Mac OS X** Improved Mac OS X integration
