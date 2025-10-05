# Python 2.4 Release Summary

**Released:** November 30, 2004
**Source:** [Official Python 2.4 Release Notes](https://docs.python.org/2/whatsnew/2.4.html)

## Overview

Python 2.4 represents a medium-sized but significant release in Python's evolution, introducing transformative language features while substantially enhancing the standard library. Unlike the radical changes in Python 2.2 or the conservative approach of 2.3, this release strikes a balance by delivering powerful new syntax alongside practical improvements to existing functionality. The development cycle saw 481 patches applied and 502 bugs fixed, though these figures likely underestimate the true scope of changes.

The release's most impactful contributions are function decorators and generator expressions, both addressing long-standing limitations in Python's expressiveness. Decorators provide elegant syntax for function and method transformation, while generator expressions extend Python's iterator capabilities with memory-efficient comprehension syntax. Beyond language features, Python 2.4 adds built-in set types for high-performance set operations, the decimal module for precise financial calculations, the subprocess module for unified process management, and numerous performance optimizations that deliver approximately 5% improvement over Python 2.3 and 35% over Python 2.2 on standard benchmarks.

## Major Language Features

### PEP 318: Function Decorators

Python 2.4 introduces decorator syntax, addressing the awkward pattern required for static and class methods since Python 2.2. Previously, defining a class method required writing a function definition followed by manual wrapping:

```python
class C:
   def meth (cls):
       ...
   meth = classmethod(meth)   # Rebind name to wrapped-up class method
```

The new decorator syntax uses the `@` character before function definitions, borrowing notation from Java:

```python
class C:
   @classmethod
   def meth (cls):
       ...
```

Multiple decorators can be stacked, applying transformations from bottom to top:

```python
@A
@B
@C
def f():
    ...
```

This is equivalent to `f = A(B(C(f)))`. Decorators must appear on separate lines before the `def` statement and can only decorate function definitions at module or class level.

Writing custom decorators is straightforward. A simple decorator that sets an attribute:

```python
>>> def deco(func):
...    func.attr = 'decorated'
...    return func
...
>>> @deco
... def f(): pass
...
>>> f.attr
'decorated'
```

A more practical example validates argument types:

```python
def require_int (func):
    def wrapper (arg):
        assert isinstance(arg, int)
        return func(arg)
    return wrapper

@require_int
def p1 (arg):
    print arg
```

Decorators can also accept arguments. When arguments are supplied, your decorator function must return another decorator function. The `func_name` attribute is now writable, allowing decorators to maintain proper function names in tracebacks.

### PEP 289: Generator Expressions

Generator expressions bring memory-efficient iteration to comprehension syntax. While list comprehensions produce complete lists in memory, generator expressions create iterators that yield elements one at a time:

```python
# List comprehension - materializes entire list
links = [link for link in get_all_links() if not link.followed]

# Generator expression - yields elements lazily
links = (link for link in get_all_links() if not link.followed)
```

Generator expressions always require parentheses, though function call parentheses suffice:

```python
print sum(obj.count for obj in list_all_objects())
```

An important difference from list comprehensions: the loop variable is not accessible outside generator expressions, eliminating namespace pollution. Future Python versions will make list comprehensions match this behavior.

### PEP 218: Built-in Set Objects

Python 2.3's `sets` module is now implemented as built-in types `set()` and `frozenset()` with C implementations for high performance:

```python
>>> a = set('abracadabra')              # form a set from a string
>>> 'z' in a                            # fast membership testing
False
>>> a                                   # unique letters in a
set(['a', 'r', 'b', 'c', 'd'])

>>> b = set('alacazam')                 # form a second set
>>> a - b                               # letters in a but not in b
set(['r', 'd', 'b'])
>>> a | b                               # letters in either a or b
set(['a', 'c', 'r', 'd', 'b', 'm', 'z', 'l'])
>>> a & b                               # letters in both a and b
set(['a', 'c'])
>>> a ^ b                               # letters in a or b but not both
set(['r', 'd', 'b', 'm', 'z', 'l'])

>>> a.add('z')                          # add a new element
>>> a.update('wxy')                     # add multiple new elements
```

The `frozenset()` type provides an immutable, hashable variant suitable for dictionary keys or set members. The original `sets` module remains available for subclassing purposes with no current deprecation plans.

### PEP 322: Reverse Iteration

The new `reversed(seq)` built-in function returns an iterator that traverses sequences in reverse order:

```python
>>> for i in reversed(xrange(1,4)):
...    print i
...
3
2
1
```

Compared to extended slicing like `range(1,4)[::-1]`, `reversed()` is more readable, faster, and uses substantially less memory. Note that `reversed()` requires sequences, not arbitrary iterators; convert iterators to lists first if needed.

### Enhanced Sorting Capabilities

The `list.sort()` method gained three powerful keyword parameters:

- **cmp**: A comparison function taking two parameters and returning -1, 0, or +1
- **key**: A function that extracts comparison keys from list elements
- **reverse**: A Boolean for reverse sorting

The `key` parameter enables efficient sorting transformations:

```python
>>> L = ['A', 'b', 'c', 'D']
>>> L.sort(key=str.lower)
>>> L
['A', 'b', 'c', 'D']
```

Using `key` is significantly faster than `cmp` because it calls the key function once per element rather than twice per comparison. For simple transformations, unbound methods work well: `L.sort(key=str.lower)`.

The `reverse` parameter simplifies descending sorts:

```python
>>> L.sort(reverse=True)
```

Sorting is now guaranteed stable, meaning elements with equal keys maintain their original relative order. This enables multi-level sorting: sort by name, then by age, producing an age-sorted list with name-sorted groups within each age.

A new `sorted(iterable)` built-in function returns a sorted copy without modifying the original, accepting any iterable and supporting the same keyword parameters:

```python
>>> sorted('Monty Python')
[' ', 'M', 'P', 'h', 'n', 'n', 'o', 'o', 't', 't', 'y', 'y']
>>> colormap = dict(red=1, blue=2, green=3, black=4, yellow=5)
>>> for k, v in sorted(colormap.iteritems()):
...     print k, v
...
black 4
blue 2
green 3
red 1
yellow 5
```

## Standard Library Improvements

### PEP 327: Decimal Data Type

The new `decimal` module provides arbitrary-precision decimal arithmetic, addressing floating-point representation limitations. IEEE 754 floating-point numbers cannot exactly represent many decimal fractions because they use binary representation with limited precision (52 bits for Python's float type):

```python
>>> 1.1
1.1000000000000001
```

The `Decimal` type represents numbers exactly using decimal arithmetic:

```python
>>> import decimal
>>> decimal.Decimal("1.1")
Decimal("1.1")
```

Create `Decimal` instances from integers, strings, or tuples specifying sign, mantissa digits, and exponent:

```python
>>> decimal.Decimal(1972)
Decimal("1972")
>>> decimal.Decimal((1, (1, 4, 7, 5), -2))
Decimal("-14.75")
```

Converting from floats is deliberately excluded from the API to avoid ambiguity; convert floats to strings first:

```python
>>> f = 1.1
>>> decimal.Decimal(str(f))
Decimal("1.1")
```

Decimal instances support standard arithmetic operations:

```python
>>> a = decimal.Decimal('35.72')
>>> b = decimal.Decimal('1.73')
>>> a + b
Decimal("37.45")
>>> a / b
Decimal("20.64739884393063583815028902")
>>> a ** 2
Decimal("1275.9184")
```

Decimals can combine with integers but not floats to prevent accuracy loss. The `Context` class encapsulates precision settings, rounding modes, and error handling:

```python
>>> decimal.getcontext().prec
28
>>> decimal.Decimal(1) / decimal.Decimal(7)
Decimal("0.1428571428571428571428571429")
>>> decimal.getcontext().prec = 9
>>> decimal.Decimal(1) / decimal.Decimal(7)
Decimal("0.142857143")
```

Contexts control error behavior through the `traps` dictionary:

```python
>>> decimal.Decimal(1) / decimal.Decimal(0)
Traceback (most recent call last):
  ...
decimal.DivisionByZero: x / 0
>>> decimal.getcontext().traps[decimal.DivisionByZero] = False
>>> decimal.Decimal(1) / decimal.Decimal(0)
Decimal("Infinity")
```

### PEP 324: subprocess Module

The `subprocess` module unifies Python's fragmented process execution APIs, replacing the confusing collection of `os.system()`, `popen2`, and related functions with a single `Popen` class:

```python
class Popen(args, bufsize=0, executable=None,
            stdin=None, stdout=None, stderr=None,
            preexec_fn=None, close_fds=False, shell=False,
            cwd=None, env=None, universal_newlines=False,
            startupinfo=None, creationflags=0):
```

The `args` parameter specifies the program and arguments as a sequence. The `stdin`, `stdout`, and `stderr` parameters control I/O streams, accepting file objects, file descriptors, or `subprocess.PIPE` for inter-process communication. Additional options include `cwd` for working directory, `env` for environment variables, and `preexec_fn` for pre-execution callbacks.

The `Popen` instance provides `wait()` to pause until completion, `poll()` to check status, and `communicate(data)` to send input and retrieve output.

The `call()` shortcut function simplifies common cases:

```python
sts = subprocess.call(['dpkg', '-i', '/tmp/new-package.deb'])
if sts == 0:
    # Success
    ...
else:
    # dpkg returned an error
    ...
```

By default, commands execute directly without shell interpretation, improving security. Use `shell=True` explicitly when shell features are needed.

### PEP 292: String Template Class

The `string.Template` class provides simpler variable substitution syntax suitable for user-edited templates:

```python
>>> import string
>>> t = string.Template('$page: $title')
>>> t.substitute({'page':2, 'title': 'The Best of Times'})
'2: The Best of Times'
```

The `$` syntax is simpler than `%` formatting, reducing user errors in applications like Mailman where non-programmers edit templates. The `safe_substitute()` method ignores missing keys rather than raising exceptions:

```python
>>> t = string.Template('$page: $title')
>>> t.safe_substitute({'page':3})
'3: $title'
```

### collections.deque

The new `collections` module introduces `deque`, a double-ended queue supporting efficient operations at both ends:

```python
>>> from collections import deque
>>> d = deque('ghi')        # make a new deque with three items
>>> d.append('j')           # add a new entry to the right side
>>> d.appendleft('f')       # add a new entry to the left side
>>> d                       # show the representation of the deque
deque(['f', 'g', 'h', 'i', 'j'])
>>> d.pop()                 # return and remove the rightmost item
'j'
>>> d.popleft()             # return and remove the leftmost item
'f'
>>> list(d)                 # list the contents of the deque
['g', 'h', 'i']
>>> 'h' in d                # search the deque
True
```

Several standard library modules, including `Queue` and `threading`, now use `deque` internally for improved performance.

### itertools Enhancements

The `itertools` module gained two significant functions:

**groupby(iterable[, func])** groups consecutive elements with matching keys:

```python
>>> import itertools
>>> L = [2, 4, 6, 7, 8, 9, 11, 12, 14]
>>> for key_val, it in itertools.groupby(L, lambda x: x % 2):
...    print key_val, list(it)
...
0 [2, 4, 6]
1 [7]
0 [8]
1 [9, 11]
0 [12, 14]
```

With sorted input, `groupby()` works like Unix's `uniq` filter for eliminating, counting, or identifying duplicates:

```python
>>> word = 'abracadabra'
>>> letters = sorted(word)
>>> [k for k, g in itertools.groupby(letters)]
['a', 'b', 'c', 'd', 'r']
>>> [(k, len(list(g))) for k, g in itertools.groupby(letters)]
[('a', 5), ('b', 2), ('c', 1), ('d', 1), ('r', 2)]
```

**tee(iterator, N)** returns N independent iterators replicating the input iterator:

```python
>>> L = [1,2,3]
>>> i1, i2 = itertools.tee(L)
>>> list(i1)
[1, 2, 3]
>>> list(i2)
[1, 2, 3]
```

Note that `tee()` must buffer yielded values, potentially consuming significant memory if iterators diverge widely. Use carefully when iterators track closely together.

### Other Notable Module Additions

**cookielib**: New module providing client-side HTTP cookie handling, mirroring the `Cookie` module's server-side support. Cookies are stored in jars with policy objects controlling acceptance. Implementations support both Netscape and Perl libwww cookie formats. `urllib2` now integrates with `cookielib` via `HTTPCookieProcessor`.

**doctest refactoring**: Extensive refactoring by Edward Loper and Tim Peters enables customization through new `DocTestFinder`, `DocTestRunner`, and `OutputChecker` classes. New features include `ELLIPSIS` flag for flexible output matching, `<BLANKLINE>` marker for blank lines, and diff-style output reporting with `REPORT_UDIFF`, `REPORT_CDIFF`, and `REPORT_NDIFF` flags.

**heapq improvements**: C implementation provides tenfold speed improvement. New `nlargest()` and `nsmallest()` functions find extreme values without full sorting, making the module suitable for high-volume data processing.

**logging enhancements**: The `basicConfig()` function gained keyword arguments for simplified configuration. New `TimedRotatingFileHandler` class complements existing `RotatingFileHandler`, both deriving from `BaseRotatingHandler` for custom implementations.

**threading.local**: Elegant thread-local storage through the `local` class, where attribute values are automatically thread-specific:

```python
import threading

data = threading.local()
data.number = 42
data.url = ('www.python.org', 80)
```

Other threads can assign and retrieve their own values for these attributes. Subclass `local` to add initialization or methods.

## Performance Improvements

Python 2.4 delivers substantial performance gains across multiple subsystems:

**List and tuple optimizations**:
- Slicing inner loops optimized, running approximately one-third faster
- List growth and shrinking machinery optimized for speed and space efficiency
- Appending and popping more efficient due to better code paths and less frequent `realloc()` calls
- List comprehensions benefit from new `LIST_APPEND` opcode, improving performance by about one-third

**Dictionary optimizations**:
- Inner loops optimized for `keys()`, `values()`, `items()`, and iterator variants
- Methods like `dict.__getitem__()` and `dict.__contains__()` implemented as `method_descriptor` objects, doubling performance

**Built-in function optimizations**:
- `list()`, `tuple()`, `map()`, `filter()`, and `zip()` run several times faster with non-sequence arguments providing `__len__()`

**String optimizations**:
- String concatenations in forms like `s = s + "abc"` and `s += "abc"` performed more efficiently
- Note: This optimization is implementation-specific; `join()` remains recommended for many concatenations

**Bytecode improvements**:
- Peephole optimizer enhanced to produce shorter, faster, more readable bytecode

**Overall improvement**: Python 2.4 runs the pystone benchmark approximately 5% faster than Python 2.3 and 35% faster than Python 2.2, though individual application results may vary.

## Other Changes

### Language Enhancements

**PEP 237: Integer/Long Unification**: The lengthy transition continues as expressions like `2 << 32` no longer produce `FutureWarning` and return correct long integer values instead of 32/64-bit limited results.

**PEP 328: Multi-line Imports**: Parentheses now enable clean multi-line import statements:

```python
from SimpleXMLRPCServer import (SimpleXMLRPCServer,
                                SimpleXMLRPCRequestHandler,
                                CGIXMLRPCRequestHandler,
                                resolve_dotted_attribute)
```

**PEP 331: Locale-Independent Float Conversions**: New C API functions `PyOS_ascii_strtod()`, `PyOS_ascii_atof()`, and `PyOS_ascii_formatd()` perform ASCII-only float conversions ignoring locale, enabling the `locale` module to safely change the numeric locale for extensions like GTK+.

**String method enhancements**:
- `ljust()`, `rjust()`, and `center()` now accept optional fill characters
- New `rsplit()` method splits from the string's end

**dict.update()**: Now accepts the same argument forms as the `dict` constructor, including mappings, iterables of key/value pairs, and keyword arguments.

**eval() and exec flexibility**: Now accept any mapping type for the `locals` parameter, not just dictionaries.

**zip() improvement**: Returns an empty list when called with no arguments instead of raising `TypeError`, enabling use with variable-length argument lists.

**-m interpreter switch**: New `-m` switch runs modules as scripts by name, searching `sys.path`. Example: `python -m profile`.

**None becomes constant**: Binding new values to `None` is now a syntax error.

**Import error handling**: Failed imports no longer leave partially-initialized module objects in `sys.modules`.

### Standard Library Updates

**CJKCodecs integration**: Comprehensive East Asian encodings added, including Chinese (gb2312, gbk, gb18030, big5), Japanese (shift-jis, euc-jp, iso-2022-jp variants), and Korean (euc-kr, iso-2022-kr) codecs.

**UTF codec improvements**: UTF-8 and UTF-16 codecs handle partial input better, allowing resumed decoding from streams.

**Module additions and enhancements**:
- `base64`: Complete RFC 3548 support for Base16, Base32, and Base64
- `bisect`: C implementation for improved performance
- `difflib`: New `HtmlDiff` class for side-by-side HTML comparisons
- `email`: Updated to version 3.0 with incremental MIME parser
- `httplib`: HTTP status code constants like `OK`, `CREATED`, `MOVED_PERMANENTLY`
- `locale`: Functions like `bind_textdomain_codeset()` for encoding control
- `operator`: New `attrgetter()` and `itemgetter()` functions for data extraction
- `os`: New `urandom(n)` function for cryptographically random bytes
- `os.path`: New `lexists()` function that returns true for broken symlinks
- `poplib`: POP over SSL support
- `profile`: Can now profile C extension functions
- `random`: New `getrandbits(N)` method for efficient large random numbers
- `re`: Conditional expressions `(?(group)A|B)` and non-recursive implementation preventing stack overflow
- `signal`: Tighter error-checking prevents invalid operations like setting handlers on `SIGKILL`
- `socket`: New `socketpair()` and `getservbyport()` functions
- `weakref`: Supports wider variety of objects including functions, sets, frozensets, deques, arrays, files, and regex patterns
- `xmlrpclib`: Multi-call extension for batching XML-RPC calls

**Deprecated modules removed**: `mpz`, `rotor`, and `xreadlines` modules have been removed.

### Build and C API Changes

**New convenience macros**:
- `Py_RETURN_NONE`, `Py_RETURN_TRUE`, `Py_RETURN_FALSE` for common return values
- `Py_CLEAR(obj)` decreases reference count and sets pointer to NULL

**New functions**:
- `PyTuple_Pack(N, obj1, obj2, ..., objN)` constructs tuples from variable arguments
- `PyDict_Contains(d, k)` performs fast dictionary lookups without masking exceptions
- `Py_IS_NAN(X)` macro tests for NaN values
- `PyEval_ThreadsInitialized()` checks if threading is active, avoiding unnecessary locking
- `PyArg_VaParseTupleAndKeywords()` accepts `va_list` arguments

**Method optimization**: New `METH_COEXISTS` flag allows functions in slots to coexist with `PyCFunction` of the same name, halving access time for methods like `set.__contains__()`.

**Profiling support**: Configure options `--enable-profiling` for gprof and `--with-tsc` for Time-Stamp-Counter profiling on x86 and PowerPC architectures.

## Key Takeaways

1. **Decorator syntax revolution** - The `@decorator` syntax transforms how Python developers apply function transformations, improving readability and enabling elegant metaprogramming patterns

2. **Generator expressions** - Memory-efficient comprehension syntax extends Python's iterator capabilities, enabling functional programming patterns on large datasets

3. **Built-in sets** - High-performance set types with C implementations bring mathematical set operations directly into Python's core, eliminating the need for separate modules

4. **Decimal arithmetic** - The decimal module addresses floating-point limitations, providing exact decimal arithmetic essential for financial and precise numerical applications

5. **Process management unification** - The subprocess module replaces Python's confusing collection of process execution functions with a single, powerful, secure interface

6. **Sorting sophistication** - Enhanced list sorting with key functions, comparison functions, and reverse parameters enables complex sorting operations with stable guarantees

7. **Performance across the board** - Systematic optimizations to lists, dictionaries, strings, and bytecode deliver 5-35% performance improvements over previous Python versions

Python 2.4 successfully balances language evolution with practical improvements, introducing decorator syntax that would become fundamental to Python programming while enhancing the standard library with production-ready modules like subprocess and decimal. The performance optimizations and iterator enhancements demonstrate Python's maturation as a language suitable for both rapid development and performance-critical applications.
