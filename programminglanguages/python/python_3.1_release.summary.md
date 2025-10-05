# Python 3.1 Release Summary

**Released:** June 27, 2009
**Source:** [Official Python 3.1 Release Notes](https://docs.python.org/3/whatsnew/3.1.html)

## Overview

Python 3.1 was the first feature release following the groundbreaking Python 3.0. While 3.0 introduced fundamental breaking changes like the text/bytes distinction, 3.1 focused on making Python 3 practical for everyday use through major performance improvements and developer-friendly features. This release addressed the most critical performance bottlenecks in 3.0 while adding useful new capabilities to the standard library.

## Major Features

### PEP 372: Ordered Dictionaries

Python 3.1 introduced `collections.OrderedDict`, a dictionary that remembers the insertion order of keys. This addressed a long-standing need in Python applications that require deterministic iteration over key/value pairs.

```python
from collections import OrderedDict

# Regular dict has arbitrary order
d = {'banana': 3, 'apple': 4, 'pear': 1, 'orange': 2}

# OrderedDict maintains insertion order
od = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
# OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])
```

The standard library immediately adopted OrderedDict in several modules. The `configparser` module uses ordered dictionaries by default, allowing configuration files to maintain their original structure. Named tuples' `_asdict()` method returns an OrderedDict, and the `json` module gained an `object_pairs_hook` parameter to support ordered dictionaries during JSON decoding.

### PEP 378: Format Specifier for Thousands Separator

Python 3.1 added a simple, non-locale-aware way to format numbers with thousands separators using the comma format specifier:

```python
format(1234567, ',d')           # '1,234,567'
format(1234567.89, ',.2f')      # '1,234,567.89'
format(Decimal('1234567.89'), ',f')  # '1,234,567.89'
```

This works with `int`, `float`, `complex`, and `Decimal` types, making numeric output more readable without requiring locale configuration.

### Improved Floating-Point Representation

Python 3.1 adopted David Gay's algorithm for converting floating-point numbers to strings, producing much cleaner representations:

```python
# Python 3.0
repr(1.1)  # '1.1000000000000001'

# Python 3.1
repr(1.1)  # '1.1'
```

The algorithm finds the shortest string representation that round-trips correctly to the same float value. While the underlying binary representation hasn't changed (1.1 still isn't exactly representable in binary), the display is now much less confusing for users. This change helps reduce misconceptions about Python's floating-point implementation while maintaining IEEE-754 guarantees.

## Type Hints

Python 3.1 predates the introduction of type hints (PEP 484 in Python 3.5), so there were no type-related features in this release.

## Interpreter Improvements

### Multiple Context Managers

The `with` statement now supports multiple context managers in a single statement:

```python
with open('mylog.txt') as infile, open('a.out', 'w') as outfile:
    for line in infile:
        if '<critical>' in line:
            outfile.write(line)
```

This cleaner syntax eliminates the need for the `contextlib.nested()` function, which became deprecated.

### Executable Directories and Zip Files

Directories and zip archives containing a `__main__.py` file can now be executed directly by passing their name to the interpreter. The directory or zipfile is automatically inserted as the first entry in `sys.path`.

### Enhanced Integer Operations

The `int` type gained a `bit_length()` method that returns the number of bits needed to represent an integer in binary:

```python
n = 37
bin(37)         # '0b100101'
n.bit_length()  # 6

n = 2**123 - 1
n.bit_length()  # 123
```

### Automatic Format String Numbering

Format strings now support automatic field numbering, eliminating the need to explicitly number every placeholder:

```python
'Sir {} of {}'.format('Gallahad', 'Camelot')
# 'Sir Gallahad of Camelot'
```

## Error Messages

Python 3.1 focused primarily on performance and features rather than error message improvements. Error messages remained similar to Python 3.0.

## Standard Library Improvements

### collections.Counter

A new `Counter` class was added for convenient counting of unique items:

```python
from collections import Counter
Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])
# Counter({'blue': 3, 'red': 2, 'green': 1})
```

### unittest Enhancements

The `unittest` module received major improvements:

1. **Test skipping and expected failures:**
```python
class TestGizmo(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_gizmo_on_windows(self):
        ...

    @unittest.expectedFailure
    def test_gimzo_without_required_library(self):
        ...
```

2. **Context manager support for assertions:**
```python
def test_division_by_zero(self):
    with self.assertRaises(ZeroDivisionError):
        x / 0
```

3. **New assertion methods:** `assertSetEqual()`, `assertDictEqual()`, `assertListEqual()`, `assertTupleEqual()`, `assertSequenceEqual()`, `assertRaisesRegexp()`, `assertIsNone()`, and `assertIsNotNone()`.

### itertools Additions

The `itertools` module gained new functions:
- `combinations_with_replacement()` - Fourth combinatoric function
- `compress()` - Filters data based on selector values
- `count()` now accepts a `step` parameter and works with `Fraction` and `Decimal`

```python
from itertools import combinations_with_replacement, compress, count
from fractions import Fraction

list(combinations_with_replacement('LOVE', 2))
# ['LL', 'LO', 'LV', 'LE', 'OO', 'OV', 'OE', 'VV', 'VE', 'EE']

list(compress(data=range(10), selectors=[0,0,1,1,0,1,0,1,0,0]))
# [2, 3, 5, 7]

c = count(start=Fraction(1,2), step=Fraction(1,6))
[next(c), next(c), next(c)]
# [Fraction(1, 2), Fraction(2, 3), Fraction(5, 6)]
```

### importlib

A new module, `importlib`, provides a complete, portable, pure Python reference implementation of the `import` statement. This was a substantial step forward in documenting and defining import behavior.

### Other Notable Additions

- **tkinter.ttk** - Access to Tk themed widget set
- **Context managers** - `gzip.GzipFile` and `bz2.BZ2File` now support `with` statements
- **decimal** - New `from_float()` method for exact conversion from binary floats
- **logging.NullHandler** - For libraries that log but don't want to force logging configuration
- **runpy** - Now supports executing packages by finding `__main__` submodule
- **pdb** - Can access and display source code from zipimport
- **functools.partial** - Objects can now be pickled
- **re** - `sub()`, `subn()`, and `split()` now accept a flags parameter
- **sys.version_info** - Now a named tuple
- **nntplib and imaplib** - IPv6 support

## Performance

Python 3.1 delivered massive performance improvements that made Python 3 viable for production use:

### I/O Library Rewrite (PEP 3116)

The new I/O library in Python 3.0 was written mostly in Python and proved to be a severe bottleneck. Python 3.1 rewrote the entire I/O library in C, achieving **2-20x faster performance** depending on the operation. The pure Python version remained available as the `_pyio` module for experimentation.

This was arguably the single most important change in Python 3.1, transforming Python 3 from unusably slow for I/O-heavy workloads to competitive with Python 2.

### Other Performance Improvements

- **Garbage collection** - Tuples and dicts containing only untrackable objects are no longer tracked by the GC, reducing overhead on long-running programs
- **Bytecode evaluation** - New `--with-computed-gotos` configure option enables up to 20% faster bytecode dispatch on supported compilers (gcc, SunPro, icc)
- **Text encoding** - UTF-8, UTF-16, and LATIN-1 decoding is now 2-4x faster
- **json module** - New C extension provides substantial performance improvements
- **Unpickling** - Now interns attribute names, saving memory and reducing pickle size
- **Integer storage** - Now stored in base 2^30 on 64-bit machines (previously 2^15), providing significant performance gains

## Security

Python 3.1 didn't introduce specific security-focused features, though the general code quality improvements and bug fixes enhanced overall security.

## Removals and Deprecations

### Deprecations

- **string.maketrans()** - Deprecated and replaced by `bytes.maketrans()` and `bytearray.maketrans()` static methods
- **contextlib.nested()** - Deprecated in favor of the new multi-context-manager `with` syntax

### Behavior Changes

- **round(x, n)** - Now returns an integer when `x` is an integer (previously returned float)

## Implementation Changes

### Build and Internal Changes

- **Integer representation** - Integers stored internally in base 2^30 on 64-bit machines (vs. 2^15), with new `sys.int_info` attribute:
```python
sys.int_info
# sys.int_info(bits_per_digit=30, sizeof_digit=4)
```

## C API Changes

### New APIs

- **PyOS_string_to_double()** - Replaces deprecated `PyOS_ascii_strtod()` and `PyOS_ascii_atof()`
- **PyCapsule** - Replacement for the problematic `PyCObject` API with better type safety

### Deprecations

- **PyNumber_Int()** - Deprecated in favor of `PyNumber_Long()`

### Behavior Changes

- **PyLong_AsUnsignedLongLong()** - Now raises `OverflowError` for negative values instead of `TypeError`

## Migration Notes

### Potential Breaking Changes

1. **Floating-point string representation** - The new shorter float representations can break existing doctests that expect the old 17-digit format:

```python
def e():
    '''Compute the base of natural logarithms.

    >>> e()
    2.7182818284590451    # This will fail in 3.1

    '''
    return sum(1/math.factorial(x) for x in reversed(range(30)))
```

The actual output in 3.1 is `2.718281828459045` (shorter). Doctests need to be updated to expect the new format.

2. **pickle compatibility** - The automatic name remapping in pickle for protocol 2 or lower makes Python 3.1 pickles unreadable in Python 3.0. Solutions:
   - Use protocol 3 for Python 3.x to 3.x communication
   - Set `fix_imports=False` if you need 3.0 compatibility

   Protocol 2 pickles from Python 3.1 won't work in Python 3.0, but will work with Python 2.x.

3. **round() return type** - Code that depends on `round()` returning a float will need adjustment when passing integer arguments.

## Key Takeaways

1. **Python 3.1 made Python 3 practical** - The I/O performance improvements were critical for adoption
2. **OrderedDict filled a long-standing gap** - Deterministic dictionary iteration became possible
3. **Better floating-point display** - Reduced confusion about binary floating-point limitations
4. **unittest became production-ready** - Test skipping and context manager assertions modernized testing
5. **importlib documented import behavior** - Pure Python reference implementation clarified import semantics
6. **Performance across the board** - 2-20x I/O speedup, 2-4x encoding speedup, 20% bytecode speedup, faster GC

Python 3.1 transformed Python 3 from an interesting but slow experiment into a viable platform for real-world applications. The performance improvements, particularly to I/O operations, addressed the most serious criticism of Python 3.0 and laid the groundwork for broader Python 3 adoption.
