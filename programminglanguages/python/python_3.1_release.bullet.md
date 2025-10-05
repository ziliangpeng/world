# Python 3.1 Release Notes

**Released:** June 27, 2009
**EOL:** June 2012 (already reached)

## Major Highlights

Python 3.1 made Python 3 practical for real-world use with critical performance fixes and useful new features:

1. **I/O library rewritten in C (PEP 3116)** - 2-20x faster I/O operations, fixing Python 3.0's biggest bottleneck
2. **OrderedDict (PEP 372)** - Dictionary that preserves insertion order
3. **Improved float representation** - `repr(1.1)` now returns `'1.1'` instead of `'1.1000000000000001'` (David Gay's algorithm)
4. **unittest improvements** - Test skipping, expected failures, and context manager assertions
5. **importlib module** - Pure Python reference implementation of import machinery
6. **collections.Counter** - Convenient counting of items in sequences
7. **Multiple context managers** - `with open(...) as f1, open(...) as f2:` syntax

## New Features

### Language Syntax

- 🟡 **Syntax** Multiple context managers in single with statement - `with open('a') as f1, open('b') as f2:`
- 🟡 **Syntax** Automatic format string field numbering - `'{} {}'.format('a', 'b')` without explicit numbers
- 🟡 **Syntax** Directories and zip archives with `__main__.py` can be executed directly
- 🟢 **Syntax** `round(x, n)` now returns integer when x is an integer (previously returned float)

### Built-in Types

- 🟡 **int** New bit_length() method returns number of bits needed to represent integer in binary
- 🟡 **str** Format specifier for thousands separator (PEP 378) - `format(1234567, ',d')` → `'1,234,567'`
- 🟡 **float** Improved float-to-string conversion using David Gay's algorithm - `repr(1.1)` now `'1.1'` not `'1.1000000000000001'`
- 🟢 **bytes/bytearray** New maketrans() static methods replace deprecated string.maketrans()

### Standard Library - New Modules

- 🔴 **collections** OrderedDict class maintains key insertion order (PEP 372)
- 🔴 **collections** Counter class for counting items in sequences
- 🟡 **importlib** Pure Python reference implementation of import statement and __import__()
- 🟡 **tkinter.ttk** Access to Tk themed widget set

### Standard Library - Major Enhancements

- 🔴 **unittest** Test skipping with @skipUnless, @skipIf decorators
- 🔴 **unittest** Expected failure support with @expectedFailure
- 🔴 **unittest** Context manager for exception testing - `with self.assertRaises(ValueError):`
- 🔴 **unittest** New assertion methods: assertSetEqual, assertDictEqual, assertListEqual, assertTupleEqual, assertSequenceEqual, assertRaisesRegexp, assertIsNone, assertIsNotNone
- 🟡 **itertools** combinations_with_replacement() function
- 🟡 **itertools** compress() function for filtering by selector values
- 🟡 **itertools** count() now accepts step parameter and works with Fraction and Decimal
- 🟡 **collections.namedtuple** New rename parameter converts invalid field names to positional names (_0, _1, etc.)
- 🟡 **collections.namedtuple** _asdict() now returns OrderedDict
- 🟡 **decimal** New from_float() method for exact conversion from binary floats
- 🟡 **gzip, bz2** GzipFile and BZ2File now support context management protocol
- 🟡 **json** New object_pairs_hook parameter for building OrderedDicts during decoding
- 🟡 **logging** NullHandler class for libraries that don't want to force logging configuration
- 🟡 **configparser** Now uses OrderedDict by default to preserve configuration file order
- 🟡 **re** sub(), subn(), and split() functions now accept flags parameter
- 🟡 **runpy** Now supports executing packages by looking for __main__ submodule
- 🟡 **pdb** Can now access and display source code loaded via zipimport (PEP 302 loaders)
- 🟡 **functools** partial objects can now be pickled
- 🟡 **pydoc** Help topics for symbols - `help('@')` now works
- 🟡 **io** Three new constants for seek() method: SEEK_SET, SEEK_CUR, SEEK_END
- 🟡 **sys** version_info tuple is now a named tuple
- 🟢 **nntplib, imaplib** IPv6 support

### Standard Library - pickle Compatibility

- 🟡 **pickle** Better Python 2/3 interoperability with protocol 2 or lower
- 🟡 **pickle** Automatic name remapping for Python 2.x compatibility (e.g., `__builtin__.set` → `builtins.set`)
- 🟡 **pickle** New fix_imports parameter to control name remapping behavior
- 🟡 **pickle** Protocol 2 pickles from Python 3.1 won't be readable in Python 3.0

## Improvements

### Performance

- 🔴 **Performance** I/O library entirely rewritten in C (PEP 3116) - 2-20x faster depending on operation
- 🔴 **Performance** UTF-8, UTF-16, and LATIN-1 decoding now 2-4x faster
- 🟡 **Performance** json module now has C extension for substantial speedup
- 🟡 **Performance** Garbage collector optimization - tuples/dicts with only untrackable objects not tracked, reducing GC overhead
- 🟡 **Performance** New --with-computed-gotos configure option enables up to 20% bytecode speedup on gcc, SunPro, icc
- 🟡 **Performance** Unpickling now interns attribute names, saving memory and reducing pickle size
- 🟡 **Performance** Integers stored internally in base 2^30 on 64-bit machines (vs 2^15), significant performance gains

### Developer Experience

- 🟡 **IDLE** Format menu now provides option to strip trailing whitespace from source files

## Deprecations

### Removing in Future Versions

- 🟢 **string** string.maketrans() deprecated - Use bytes.maketrans() or bytearray.maketrans()
- 🟢 **contextlib** contextlib.nested() deprecated - Use multiple context managers in single with statement

## Implementation Details

### CPython Internals

- 🟡 **Interpreter** New sys.int_info attribute shows internal integer storage format
- 🟡 **Interpreter** Integers stored in base 2^30 on 64-bit (vs 2^15), base 2^15 on 32-bit
- 🟡 **Interpreter** Configure option --enable-big-digits to override default integer storage base
- 🟡 **Interpreter** Pure Python I/O implementation available as _pyio module for experimentation

### C API

- 🟡 **C API** New PyCapsule type replaces problematic PyCObject API - better type safety and simpler destructor
- 🟡 **C API** New PyOS_string_to_double() replaces deprecated PyOS_ascii_strtod() and PyOS_ascii_atof()
- 🟡 **C API** PyLong_AsUnsignedLongLong() now raises OverflowError for negative values (not TypeError)
- 🟢 **C API** PyNumber_Int() deprecated - Use PyNumber_Long() instead

## Porting Notes

### Breaking Changes

- 🟡 **Compatibility** New float representation breaks existing doctests expecting 17-digit format
  - Old: `2.7182818284590451`, New: `2.718281828459045`
  - Update doctests to expect shorter representations
- 🟡 **Compatibility** Protocol 2 pickles from Python 3.1 unreadable in Python 3.0
  - Use protocol 3 for Python 3.x communication
  - Set fix_imports=False if Python 3.0 compatibility needed
- 🟢 **Compatibility** round() return type changed for integer inputs (now returns int not float)
