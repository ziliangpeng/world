# Python 2.7 Release Notes

**Released:** July 3, 2010
**EOL:** January 1, 2020

## Major Highlights

Python 2.7 is the final major release of Python 2.x with an unprecedented 10-year support period, serving as a bridge to Python 3:

1. **Set literals and comprehensions (backported from Python 3.1)** - Set literals `{1,2,3}` and dict/set comprehensions for more expressive code
2. **OrderedDict (PEP 372)** - Dictionary that remembers insertion order, widely integrated across stdlib
3. **Counter class** - Elegant tallying and frequency counting in collections module
4. **argparse module (PEP 389)** - Modern command-line parsing replacing optparse
5. **Enhanced unittest** - Test discovery, 15+ new assertion methods, test skipping, and cleanup functions
6. **Correctly rounded float operations** - Improved numeric accuracy for float-to-string conversions and repr()
7. **Performance improvements** - Faster garbage collection, 2x faster long integer operations, optimized string methods

## New Features

### Language Syntax

- 🔴 **Syntax** Set literal syntax backported from Python 3.1 - `{1, 2, 3}` creates mutable sets
- 🔴 **Syntax** Dictionary comprehensions - `{x: x*x for x in range(6)}`
- 🔴 **Syntax** Set comprehensions - `{x*x for x in range(6)}`
- 🟡 **Syntax** Multiple context managers in single with statement - `with A() as a, B() as b:`
- 🟡 **str.format** Automatic numbering of replacement fields - `'{}:{}'.format(2009, 4)` instead of `'{0}:{1}'`
- 🟡 **str.format** Comma separator for thousands (PEP 378) - `'{:20,d}'.format(18446744073709551616)`
- 🟡 **Syntax** Complex numbers now support format() with right-alignment by default

### Standard Library - New Modules & Classes

- 🔴 **collections** OrderedDict class (PEP 372) - Remembers insertion order, integrated into ConfigParser, namedtuple._asdict(), and json.JSONDecoder
- 🔴 **collections** Counter class - Dict subclass for tallying with most_common(), elements(), and subtract() methods
- 🔴 **argparse** New command-line parsing module (PEP 389) - Replaces optparse with better validation, subcommands, and FileType support
- 🟡 **importlib** Subset backported from Python 3.1 for import system functionality
- 🟡 **io** Module rewritten in C for performance - Foundation for Python 3's I/O system

### Numeric & Type Improvements

- 🔴 **float** Correctly rounded float-to-string and string-to-float conversions on most platforms
- 🔴 **float** repr(float) now returns shortest decimal string that rounds back to original value
- 🔴 **float** round() function now correctly rounded
- 🟡 **int/long** bit_length() method returns number of bits needed for binary representation
- 🟡 **int/long** Integer-to-float conversions use nearest-value rounding for large numbers
- 🟡 **int/long** Integer division more accurate in rounding behaviors
- 🟡 **memoryview** New memoryview type (PEP 3137) - View into another object's memory buffer without copying

### collections Module

- 🟡 **collections** deque.count() and deque.reverse() methods added
- 🟡 **collections** deque.maxlen read-only attribute exposes maximum length
- 🟡 **collections** namedtuple gains rename=True parameter for handling invalid field names
- 🟡 **collections** Mapping ABC returns NotImplemented when compared to non-Mapping types

### Dictionary Enhancements

- 🟡 **dict** Dictionary view methods backported from Python 3 (PEP 3106) - viewkeys(), viewvalues(), viewitems()
- 🟡 **dict** Views are dynamic and reflect dictionary changes
- 🟡 **dict** Key/item views support set operations (intersection, union)
- 🟡 **dict** Views automatically converted to standard methods by 2to3 tool

### unittest Module

- 🔴 **unittest** Test discovery with TestLoader.discover() and command-line interface: `python -m unittest discover`
- 🔴 **unittest** 15+ new assertion methods: assertIs, assertIsNot, assertIsNone, assertIsNotNone, assertIn, assertNotIn, assertIsInstance, assertNotIsInstance, assertGreater, assertGreaterEqual, assertLess, assertLessEqual, assertMultiLineEqual, assertSequenceEqual, assertDictEqual, assertSetEqual, assertRegexpMatches, assertNotRegexpMatches
- 🔴 **unittest** Test skipping with @unittest.skip decorators and @unittest.skipIf, @unittest.skipUnless conditions
- 🔴 **unittest** Expected failures with @unittest.expectedFailure decorator
- 🟡 **unittest** addCleanup() for registering cleanup functions guaranteed to run
- 🟡 **unittest** assertMultiLineEqual() uses difflib for clear diff output
- 🟡 **unittest** Automatically re-enables deprecation warnings when running tests

### logging Module

- 🔴 **logging** Dictionary-based configuration with dictConfig() (PEP 391) - Alternative to fileConfig() using JSON/YAML-friendly dicts
- 🟡 **logging** SysLogHandler supports TCP via socktype parameter (socket.SOCK_STREAM)
- 🟡 **logging** Logger.getChild() retrieves descendant loggers with relative paths
- 🟡 **logging** LoggerAdapter.isEnabledFor() checks if messages would be processed

### datetime Module

- 🟡 **datetime** timedelta.total_seconds() returns duration as floating-point seconds

### Other Standard Library

- 🟡 **ElementTree** Updated to 1.3 with extended XPath support, improved iteration with iter() and itertext()
- 🟡 **decimal** Format specifier support for currency and thousands separators
- 🟡 **gzip** Context manager protocol support - `with gzip.open(...) as f:`
- 🟡 **bz2** BZ2File supports context management protocol
- 🟡 **ConfigParser** Constructors accept allow_no_value parameter for options without values
- 🟡 **ConfigParser** Uses OrderedDict by default, preserving configuration file order
- 🟡 **copy** deepcopy() now correctly copies bound instance methods
- 🟡 **ctypes** Always converts None to C NULL pointer for pointer arguments
- 🟡 **ctypes** libffi library updated to version 3.0.9
- 🟡 **binascii** Now supports buffer API (memoryview, buffer objects)
- 🟡 **bsddb** Updated to version 4.8.4 with better Python 3.x compatibility
- 🟡 **json** JSONDecoder constructor supports object_pairs_hook for building OrderedDict instances
- 🟢 **multiprocessing** ProcessPoolExecutor support (concurrent.futures backport)

## Improvements

### Performance

- 🔴 **Performance** Garbage collection optimized for common allocation patterns - Full collections only when middle generation collected 10x and survivors exceed 10% of oldest generation
- 🔴 **Performance** Garbage collector avoids tracking simple containers (tuples/dicts of atomic types) - Reduces GC overhead
- 🔴 **Performance** Long integers use base 2^30 on 64-bit systems (previously 2^15) - Significant performance gains and 2-6 bytes smaller
- 🔴 **Performance** Long integer division 50-150% faster with tightened inner loop and shift optimizations
- 🟡 **Performance** String methods (split, replace, rindex, rpartition, rsplit) use fast reverse-search - Up to 10x faster
- 🟡 **Performance** with statement initial setup uses new opcode for faster __enter__() and __exit__() lookup
- 🟡 **Performance** List comprehensions with if conditions compiled to faster bytecode
- 🟡 **Performance** Integer-to-string conversion (base 10) special-cased for significant speedup
- 🟡 **Performance** % operator special-cases string operands - 1-3% faster for templating applications
- 🟡 **Performance** cPickle special-cases dictionaries - Nearly 2x faster pickling
- 🟡 **Performance** pickle/cPickle automatically intern attribute name strings - Reduces memory usage
- 🟡 **Performance** Bitwise operations on long integers significantly faster

### Numeric Accuracy

- 🟡 **float** sys.float_repr_style attribute indicates 'short' (new) or 'legacy' repr style
- 🟡 **float** F format code always uses uppercase (INF, NAN)

## Deprecations

- 🟡 **contextlib** contextlib.nested() deprecated - Use multiple context managers in single with statement
- 🟡 **operator** operator.isCallable() and operator.sequenceIncludes() trigger warnings (not in Python 3)
- 🟡 **DeprecationWarning** Now silenced by default for end users - Re-enable with -Wd flag or PYTHONWARNINGS environment variable

## Migration Features (Python 3 Backports)

- 🔴 **Migration** Set literals, dict/set comprehensions, and multiple context managers backported
- 🔴 **Migration** -3 flag enables Python 3 compatibility warnings
- 🔴 **Migration** -3 flag automatically enables -Qwarn for classic division warnings
- 🟡 **Migration** io module C implementation (Python 3 I/O system foundation)
- 🟡 **Migration** Dictionary views (viewkeys, viewvalues, viewitems) automatically converted by 2to3
- 🟡 **Migration** memoryview objects backported
- 🟡 **Migration** importlib subset backported

## Implementation Details

### C API

- 🟡 **C API** PyCapsule type replaces PyCObject for type-safe C API exposure - Name checking prevents segmentation faults
- 🟡 **C API** PyLong_AsLongAndOverflow() C function added
- 🟡 **C API** PySys_SetArgvEx() replaces PySys_SetArgv() to close security hole - Prevents Trojan-horse modules in current directory
- 🟡 **C API** Py_AddPendingCall() now thread-safe
- 🟢 **C API** object.__format__() triggers PendingDeprecationWarning if passed format string

### Interpreter

- 🟡 **Interpreter** os.fork() acquires import lock before forking and cleans up threading module locks in child process
- 🟡 **Interpreter** compile() built-in accepts code with any line-ending convention and doesn't require trailing newline
- 🟡 **Interpreter** Python tokenizer translates line endings itself
- 🟡 **Interpreter** Implicit coercion for complex numbers removed - No more __coerce__() calls
- 🟡 **Interpreter** Module dictionaries only cleared on GC if no one else holds reference
- 🟢 **Interpreter** Extra parentheses in function definitions (def f((x)):) trigger warnings in Python 3 mode

### Build System

- 🟡 **Build** --with-system-expat configure option to use system Expat library
- 🟡 **Build** --with-valgrind disables pymalloc for better memory debugging
- 🟡 **Build** --enable-big-digits controls long integer base (2^15 vs 2^30)
- 🟡 **Build** sys.long_info structseq provides internal format information (bits_per_digit, sizeof_digit)

## Platform & Environment

- 🟡 **Environment** PYTHONWARNINGS environment variable controls warning behavior - Equivalent to -W switch settings
- 🟡 **Windows** os.kill() implemented with CTRL_C_EVENT and CTRL_BREAK_EVENT support
- 🟡 **Windows** Native thread-local storage functions used
- 🟡 **Windows** Registry access via _winreg enhanced
- 🟡 **Mac OS X** /Library/Python/2.7/site-packages added to sys.path (removed in 2.7.13)
- 🟡 **FreeBSD** SO_SETFIB socket constant for alternate routing tables
- 🟢 **Encodings** cp720 encoding for Arabic text and cp858 encoding (CP 850 variant with euro symbol)

## Other Changes

- 🟡 **unicode** Subclasses can now override __unicode__() method
- 🟡 **bytearray** translate() method now accepts None as first argument
- 🟡 **classmethod/staticmethod** Wrapper objects expose wrapped function as __func__ attribute
- 🟡 **__slots__** Deleting unset attribute now raises AttributeError as expected
- 🟡 **file** Sets filename attribute on IOError when opening directory on POSIX
- 🟡 **file** Explicitly checks for and forbids writing to read-only file objects
- 🟡 **import** No longer tries absolute import if relative import fails
- 🟡 **weak references** Old-style class objects now support weak references
- 🟢 **Syntax** Python 3 warning for extra parentheses in function definitions

## Maintenance Release Features (Unique to Python 2.7)

Python 2.7 uniquely received new features in maintenance releases due to its extended support:

- 🔴 **ssl** (PEP 466, 2.7.7-2.7.9) - Backported most of Python 3.4's ssl module for network security
- 🔴 **ssl** (PEP 476, 2.7.9) - HTTPS certificate verification enabled by default (hostname matching, platform trust store)
- 🔴 **ensurepip** (PEP 477, 2.7.9) - pip bundled with Python installations - `python -m ensurepip` bootstraps pip
- 🟡 **ssl** (PEP 493, 2.7.12) - HTTPS migration tools: PYTHONHTTPSVERIFY=0 environment variable and ssl._https_verify_certificates() runtime control
- 🟡 **hashlib** (PEP 466) - hashlib.pbkdf2_hmac() for password hashing
- 🟡 **hmac** (PEP 466) - hmac.compare_digest() for timing-attack resistance
- 🟡 **ssl** (PEP 466) - OpenSSL upgrades in official installers

## Long-Term Support

- 🔴 **Release** Python 2.7 supported for 10 years (until 2020) - Far longer than typical 18-24 month support period (PEP 373)
- 🔴 **Release** Final major release of Python 2.x series - No new feature releases for Python 2
- 🟡 **Release** Exceptional features may be added to standard library via PEP process for critical needs (especially network security)
