# Python 2.7 Release Summary

**Released:** July 3, 2010
**Source:** [Official Python 2.7 Release Notes](https://docs.python.org/2/whatsnew/2.7.html)

## Overview

Python 2.7 holds a unique position in Python's history as the final major release of the 2.x series. Released with an unprecedented extended support period until 2020, it serves as a bridge between Python 2 and Python 3, backporting several Python 3 features to ease migration while maintaining backward compatibility. The release focuses on making Python 2.7 a stable, long-term platform for production systems that haven't yet transitioned to Python 3.

The release brings substantial improvements across multiple areas: enhanced numeric handling for both floating-point and Decimal types, powerful standard library additions like OrderedDict and Counter, the argparse module for command-line parsing, and comprehensive unittest enhancements. Performance optimizations targeting garbage collection, long integer operations, and string methods make Python 2.7 faster than its predecessors. Most significantly, syntax features like set literals, dictionary comprehensions, and multiple context managers provide developers with more expressive tools while preparing codebases for eventual Python 3 migration.

## Major Language Features

### Set Literals

Python 2.7 backports the set literal syntax from Python 3, using curly brackets to create mutable sets directly:

```python
>>> {1, 2, 3, 4, 5}
set([1, 2, 3, 4, 5])
>>> set()  # empty set
set([])
>>> {}     # empty dict (unchanged)
{}
```

This syntax provides a more natural and readable way to create sets, distinguishing them from dictionaries by the absence of colons and key-value pairs.

### Dictionary and Set Comprehensions

Comprehensions now extend beyond lists to dictionaries and sets, generalizing the comprehension syntax:

```python
>>> {x: x*x for x in range(6)}
{0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
>>> {('a'*x) for x in range(6)}
set(['', 'a', 'aa', 'aaa', 'aaaa', 'aaaaa'])
```

These comprehensions provide concise, readable ways to construct dictionaries and sets, matching the expressiveness already available for lists.

### Multiple Context Managers

The `with` statement now supports multiple context managers in a single statement:

```python
with A() as a, B() as b:
    # suite of statements
```

This is equivalent to nested `with` statements but more readable. Context managers are processed left to right, each beginning a new context. This enhancement deprecates `contextlib.nested()`, which provided similar functionality with subtle semantic differences.

### String Formatting Enhancements

The `str.format()` method gains automatic numbering of replacement fields, reducing verbosity:

```python
>>> '{}:{}:{}'.format(2009, 04, 'Sunday')
'2009:4:Sunday'
>>> '{}:{}:{day}'.format(2009, 4, day='Sunday')
'2009:4:Sunday'
```

Format specifications also support the comma separator for thousands (PEP 378):

```python
>>> '{:20,.2f}'.format(18446744073709551616.0)
'18,446,744,073,709,551,616.00'
>>> '{:20,d}'.format(18446744073709551616)
'18,446,744,073,709,551,616'
```

## Standard Library Improvements

### PEP 372: OrderedDict

The `collections.OrderedDict` class remembers the insertion order of keys, providing predictable iteration:

```python
>>> from collections import OrderedDict
>>> d = OrderedDict([('first', 1), ('second', 2), ('third', 3)])
>>> d.items()
[('first', 1), ('second', 2), ('third', 3)]
```

Key features include:
- Maintains insertion order through a doubly-linked list
- `popitem(last=True)` retrieves most recent or oldest items
- Equality comparison checks both keys/values and insertion order
- O(1) operations maintained through secondary dictionary mapping keys to list nodes

The standard library integrates OrderedDict throughout: ConfigParser preserves configuration file order, `namedtuple._asdict()` returns ordered dictionaries, and json.JSONDecoder supports building OrderedDict instances.

### Counter Class

The `collections.Counter` class provides a dict subclass for tallying hashable objects:

```python
>>> from collections import Counter
>>> c = Counter()
>>> for letter in 'here is a sample of english text':
...     c[letter] += 1
>>> c
Counter({' ': 6, 'e': 5, 's': 3, 'a': 2, 'i': 2, 'h': 2, ...})
>>> c.most_common(5)
[(' ', 6), ('e', 5), ('s', 3), ('a', 2), ('i', 2)]
```

Three specialized methods enhance counting operations:
- `most_common(n)` returns the N most frequent elements
- `elements()` returns an iterator repeating elements by their count
- `subtract()` decreases counts (can result in negative values)

### PEP 389: argparse Module

The argparse module replaces optparse as the recommended command-line parsing solution:

```python
import argparse

parser = argparse.ArgumentParser(description='Command-line example.')
parser.add_argument('-v', action='store_true', dest='is_verbose',
                    help='produce verbose output')
parser.add_argument('-o', action='store', dest='output',
                    metavar='FILE',
                    help='direct output to FILE instead of stdout')
parser.add_argument(nargs='*', action='store', dest='inputs',
                    help='input filenames (default is stdin)')

args = parser.parse_args()
```

Advantages over optparse include:
- More flexible argument validation (exact count, 0+, 1+, optional)
- Subcommand support for complex applications
- FileType for automatic file handling
- Better help message formatting
- No automated migration from optparse, but superior for new code

### PEP 391: Dictionary-Based Logging Configuration

The logging module now supports `dictConfig()` for programmatic configuration:

```python
import logging.config

configdict = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(name)-15s %(levelname)-8s %(message)s'}},
    'handlers': {
        'netlog': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/logs/network.log',
            'formatter': 'standard',
            'maxBytes': 1000000,
            'backupCount': 10}},
    'loggers': {
        'network': {'handlers': ['netlog']}},
    'root': {'handlers': ['syslog']}
}

logging.config.dictConfig(configdict)
```

This approach is more maintainable than Python code and easier to generate than the fileConfig format. Dictionaries can be sourced from JSON, YAML, or constructed programmatically.

Additional logging enhancements:
- SysLogHandler supports TCP via `socktype` parameter
- `Logger.getChild()` retrieves descendant loggers with relative paths
- `LoggerAdapter.isEnabledFor()` checks if messages would be processed

### PEP 3106: Dictionary Views

Dictionary view methods backported from Python 3 provide memory-efficient alternatives to `keys()`, `values()`, and `items()`:

```python
>>> d = dict((i*10, chr(65+i)) for i in range(26))
>>> d.viewkeys()
dict_keys([0, 130, 10, 140, 20, 150, ...])
>>> d1.viewkeys() & d2.viewkeys()  # Set operations
set([0.0, 10.0, 20.0, 30.0])
```

Views are dynamic, reflecting dictionary changes, and key/item views support set operations. The 2to3 tool automatically converts view methods to standard Python 3 names.

### unittest Enhancements

The unittest module received substantial improvements:

**New assertion methods:**
- `assertIs()`, `assertIsNot()` - identity testing
- `assertIsNone()`, `assertIsNotNone()` - None checking
- `assertIn()`, `assertNotIn()` - membership testing
- `assertIsInstance()`, `assertNotIsInstance()` - type checking
- `assertGreater()`, `assertGreaterEqual()`, `assertLess()`, `assertLessEqual()` - ordering
- `assertMultiLineEqual()`, `assertSequenceEqual()`, `assertDictEqual()`, `assertSetEqual()` - type-specific comparisons
- `assertRegexpMatches()`, `assertNotRegexpMatches()` - pattern matching

**Test discovery and loading:**
- `TestLoader.discover()` finds test modules automatically
- Command-line interface: `python -m unittest discover`

**Enhanced failure reporting:**
- `assertMultiLineEqual()` uses difflib for clear diff output
- Type-specific comparison methods provide better error messages

**Test skipping and expected failures:**
```python
@unittest.skip("demonstrating skipping")
def test_nothing():
    pass

@unittest.skipIf(sys.version_info < (2, 7), "requires 2.7+")
def test_format():
    pass

@unittest.expectedFailure
def test_fail():
    self.assertEqual(1, 0)
```

**Cleanup functions:**
- `addCleanup(function, *args, **kwargs)` registers cleanup functions
- Guaranteed execution even if setUp() or test method raises

### datetime Enhancements

New `timedelta.total_seconds()` method returns the duration as a floating-point number of seconds:

```python
>>> from datetime import timedelta
>>> td = timedelta(days=1, hours=2, minutes=3, seconds=4)
>>> td.total_seconds()
93784.0
```

### Numeric Improvements

**Float representation and rounding:**
- `repr(float)` now returns the shortest decimal string that rounds back to the original value
- Correctly rounded float-to-string and string-to-float conversions
- `round()` function now correctly rounds
- Integer-to-float conversions use nearest-value rounding for large numbers

**Integer enhancements:**
- `bit_length()` method returns bits needed for binary representation:
```python
>>> n = 37
>>> bin(n)
'0b100101'
>>> n.bit_length()
6
```

### memoryview Objects

The `memoryview` type provides a view into another object's memory buffer:

```python
>>> m = memoryview(string.letters)
>>> len(m)
52
>>> m[0], m[25], m[26]
('a', 'z', 'A')
>>> m2 = m[0:26]  # Slicing creates another view
>>> m2.tobytes()
'abcdefghijklmnopqrstuvwxyz'
```

Memoryviews allow modifying mutable objects without copying:

```python
>>> b = bytearray(string.letters)
>>> mb = memoryview(b)
>>> mb[0] = '*'
>>> b[0:5]
bytearray(b'*bcde')
```

### Other Notable Module Updates

**collections:**
- `deque.count()` and `deque.reverse()` methods added
- `deque.maxlen` read-only attribute
- `namedtuple` gains `rename=True` for handling invalid field names

**io module:**
- Rewritten in C for performance
- Provides the foundation for Python 3's I/O system

**importlib:**
- Subset backported for import system functionality

**ElementTree 1.3:**
- Extended XPath support
- Improved iteration with `iter()` and `itertext()`
- ElementPath evaluation enhancements

**decimal:**
- Format specifier support for currency and thousands separators

**multiprocessing:**
- ProcessPoolExecutor support (concurrent.futures backport)

**gzip and bz2:**
- Context manager protocol support

## Performance Improvements

### Garbage Collection Optimizations

**Reduced full collection frequency:**
- Previously quadratic time for common allocation patterns
- Now performs full collection only when middle generation collected 10 times and survivors exceed 10% of oldest generation

**Smart container tracking:**
- Tuples and dicts containing only atomic types no longer tracked
- Transitively applied (dict of tuples of ints not tracked)
- Reduces objects considered during collection

### Long Integer Optimizations

**Variable base representation:**
- Base 2^30 on 64-bit systems (previously 2^15)
- Base 2^15 on 32-bit systems (configurable)
- Significant performance gains on 64-bit platforms
- 2-6 bytes smaller per long integer

**Faster algorithms:**
- Division algorithm optimized (50-150% faster)
- Bitwise operations significantly faster
- Shifts instead of multiplications in inner loops

### String and Bytes Optimizations

**Reverse-search algorithms:**
- `split()`, `replace()`, `rindex()`, `rpartition()`, `rsplit()` use fast reverse search
- Up to 10x faster in some cases

**String interning:**
- Pickle/cPickle automatically intern attribute name strings
- Reduces memory usage for unpickled objects

**Formatting optimization:**
- `%` operator special-cases string operands
- 1-3% faster for applications using frequent string formatting

### Other Performance Enhancements

**with statement:**
- New opcode for initial setup, faster `__enter__()` and `__exit__()` lookup

**List comprehensions:**
- Comprehensions with `if` conditions compiled to faster bytecode

**Integer-to-string conversion:**
- Base 10 special-cased, significantly faster

**cPickle improvements:**
- Special-cased dictionaries, nearly 2x faster pickling

## Migration Features

Python 2.7 backports several Python 3 features to facilitate migration:

**Syntax features:**
- Set literals: `{1, 2, 3}`
- Dictionary and set comprehensions
- Multiple context managers in `with`

**Standard library:**
- OrderedDict from collections
- Comma format specifier for thousands
- memoryview objects
- Dictionary views (`viewkeys()`, `viewvalues()`, `viewitems()`)
- importlib subset
- io module (C implementation)

**Built-in improvements:**
- PyCapsule C API type
- PyLong_AsLongAndOverflow() C function
- Improved float repr and rounding

**Python 3 warnings:**
- `-3` flag enables Python 3 compatibility warnings
- DeprecationWarnings for removed Python 3 features
- Warnings for extra parentheses in function definitions

## Other Changes

### Interpreter and Build System

**Environment variable:**
- `PYTHONWARNINGS` controls warning behavior

**Configure options:**
- `--with-system-expat` uses system Expat library
- `--with-valgrind` disables pymalloc for better memory debugging
- `--enable-big-digits` controls long integer base

**Threading improvements:**
- `os.fork()` acquires import lock before forking
- Cleans up threading module locks in child process
- `Py_AddPendingCall()` now thread-safe

### Security and Safety

**PySys_SetArgvEx():**
- Replaces PySys_SetArgv() to close security hole
- Prevents Trojan-horse modules in current directory
- Applications can disable sys.path updates

**Deprecation warning handling:**
- DeprecationWarnings silenced by default
- Only visible to developers, not end users
- Re-enable with `-Wd` flag or PYTHONWARNINGS

### Capsules

PyCapsule replaces PyCObject for C API exposure with type safety:

```c
void *vtable;
if (!PyCapsule_IsValid(capsule, "mymodule.CAPI")) {
    PyErr_SetString(PyExc_ValueError, "argument type invalid");
    return NULL;
}
vtable = PyCapsule_GetPointer(capsule, "mymodule.CAPI");
```

Name checking prevents accidental pointer substitution and segmentation faults.

### Platform-Specific Changes

**Windows:**
- `os.kill()` implemented with CTRL_C_EVENT, CTRL_BREAK_EVENT support
- Registry access via `_winreg` enhanced
- Native thread-local storage functions

**Mac OS X:**
- `/Library/Python/2.7/site-packages` added to sys.path (removed in 2.7.13)

**FreeBSD:**
- SO_SETFIB socket constant for alternate routing tables

## Maintenance Release Features

Python 2.7 uniquely received new features in maintenance releases:

**PEP 466 (2.7.7-2.7.9): Network Security**
- `hmac.compare_digest()` for timing-attack resistance
- `hashlib.pbkdf2_hmac()` for password hashing
- Most of Python 3.4's ssl module backported
- OpenSSL upgrades in official installers

**PEP 477 (2.7.9): ensurepip**
- pip bundled with Python installations
- `python -m ensurepip` bootstraps pip
- Simplified package management for users

**PEP 476 (2.7.9): Certificate Verification**
- HTTPS certificate verification enabled by default
- Matches hostname against certificate
- Checks against platform trust store

**PEP 493 (2.7.12): HTTPS Migration Tools**
- `PYTHONHTTPSVERIFY=0` environment variable
- `ssl._https_verify_certificates()` runtime control
- Assists gradual infrastructure upgrades

## Key Takeaways

1. **Final Python 2 release** with extended support until 2020, designed as stable long-term platform

2. **Migration bridge** backporting Python 3 syntax (set literals, comprehensions, multiple context managers) and features

3. **Standard library maturity** with OrderedDict, Counter, argparse, and comprehensive unittest enhancements

4. **Significant performance gains** in garbage collection, long integers, string operations, and pickling

5. **Enhanced numeric handling** with correctly rounded float conversions, improved repr(), and better precision

6. **Unique maintenance policy** allowing critical features like ssl improvements, pip bundling, and certificate verification in minor releases

7. **C API improvements** including PyCapsule for type-safe extension APIs, thread-safe operations, and security enhancements

Python 2.7 successfully served its dual purpose: providing a stable, performant platform for existing Python 2 codebases while incorporating enough Python 3 features to smooth the migration path. Its extended support period and careful feature backporting made it one of Python's most successful and longest-lived releases.
