# Python 2.3 Release Notes

**Released:** July 29, 2003
**EOL:** March 11, 2008

## Major Highlights

Python 2.3 polishes the 2.2 object model and adds essential data structures and standard library improvements:

1. **Set datatype (PEP 218)** - Built-in set and frozenset types for fast membership testing and mathematical set operations
2. **Boolean type (PEP 285)** - True and False constants with bool() type for clearer code
3. **enumerate() built-in (PEP 279)** - Elegant loop counter pattern replacing range(len())
4. **datetime module** - Standard date/time types for calendar and time manipulation
5. **logging module (PEP 282)** - Professional logging system with handlers, filters, and formatters
6. **Import from ZIP archives (PEP 273)** - Add .zip files to sys.path for easier distribution
7. **Performance improvements** - 25% faster on pystone benchmark with new-style class optimizations, faster sorting, and Karatsuba multiplication

## New Features

### Language Syntax & Core

- 🔴 **Syntax** Generators no longer require `from __future__ import generators` - yield is now always a keyword (PEP 255)
- 🔴 **Syntax** Substring search with `in` operator - `"ab" in "abc"` returns True for any length substring
- 🔴 **Syntax** Extended slicing support for built-in types - `L[::2]`, `L[::-1]` now work on lists, tuples, strings
- 🔴 **Builtins** enumerate() function (PEP 279) - Returns (index, value) pairs for cleaner loop iteration
- 🔴 **Builtins** sum(iterable, start=0) function - Adds up numeric items
- 🔴 **Builtins** Boolean type with True and False constants (PEP 285) - Subclass of int for compatibility
- 🔴 **Builtins** bool() constructor converts any value to True or False
- 🟡 **int** int() constructor returns long integer instead of OverflowError when value too large
- 🟡 **slice** Extended slicing with third "step" argument - `L[1:10:2]`, assignment and deletion supported
- 🟡 **slice** slice.indices(length) method returns (start, stop, step) tuple for implementing sequences
- 🟡 **str** strip(), lstrip(), rstrip() accept optional characters parameter - Not just whitespace
- 🟡 **str** startswith() and endswith() accept negative start/end parameters
- 🟡 **str** zfill() method pads numeric strings with zeros on left
- 🟡 **str** basestring abstract type - Both str and unicode inherit from it for isinstance() checks
- 🟡 **list** list.insert(pos, value) with negative pos now consistent with slice indexing
- 🟡 **list** list.index(value, start, stop) accepts optional start/stop to limit search
- 🟡 **dict** dict.pop(key, default) returns and removes key, or returns default/raises KeyError
- 🟡 **dict** dict.fromkeys(iterable, value) class method creates dict with keys from iterable
- 🟡 **dict** dict() constructor accepts keyword arguments - `dict(red=1, blue=2)`
- 🟡 **file** File objects are their own iterator - xreadlines() method no longer necessary
- 🟡 **file** File objects have read-only encoding attribute for Unicode conversion
- 🟢 **type** Most type objects now callable for creating new objects (functions, classes, modules)
- 🟢 **type** Extension type names now include module prefix - `<type '_socket.socket'>` instead of `<type 'socket'>`
- 🟢 **class** Can now assign to __name__ and __bases__ attributes of new-style classes
- 🟢 **intern** Interned strings no longer immortal - Will be garbage-collected when unreferenced

### Standard Library - New Modules

- 🔴 **sets** Set and ImmutableSet classes (PEP 218) - Mathematical set operations (union, intersection, difference)
  - Built on dictionaries, elements must be hashable
  - ImmutableSet can be used as dictionary keys
  - Alternative notation: `|` for union, `&` for intersection, `^` for symmetric difference
- 🔴 **logging** Professional logging package (PEP 282) - Hierarchical loggers with handlers, filters, and formatters
  - Multiple handlers: console, file, socket, syslog, email
  - Convenience functions: debug(), info(), warning(), error(), critical()
  - Configuration via files or code
- 🔴 **datetime** Date and time types - date, time, datetime, timedelta, tzinfo classes
  - Replaces scattered date/time functionality
  - Calendar-aware and time-aware operations
- 🔴 **csv** CSV file parser and writer (PEP 305) - Handles quoting, escaping, dialects
  - Reader and writer classes for comma-separated files
  - Supports different dialects (Excel formats)
- 🔴 **zipimport** Import modules from ZIP archives (PEP 273) - Add .zip files to sys.path
  - Automatically enabled when .zip file in sys.path
  - Supports subdirectories within archives
- 🔴 **heapq** Heap queue algorithm - Priority queue implementation
  - heappush(), heappop() maintain heap property
  - O(log n) insertion and removal
- 🔴 **itertools** Iterator utility functions - Inspired by functional languages
  - ifilter(), imap(), izip(), repeat(), chain(), islice(), and more
  - Memory-efficient iteration patterns
- 🔴 **optparse** Command-line option parser - Modern replacement for getopt
  - Automatic type conversion and validation
  - Generates usage messages automatically
- 🔴 **textwrap** Text wrapping and formatting - wrap() and fill() functions
  - Breaks text into lines of specified width
  - Preserves paragraph structure
- 🔴 **tarfile** Read and write tar archives - Full tar format support
- 🔴 **bz2** Interface to bzip2 compression - Usually smaller than zlib
- 🔴 **platform** Platform identification utilities - Determine OS, architecture, distribution
- 🔴 **timeit** Benchmarking module - Measure execution time of code snippets
  - Command-line and programmatic interfaces
  - Timer class with repeat() method

### Import System & Hooks

- 🔴 **import** New import hooks system (PEP 302) - sys.path_hooks, sys.meta_path, sys.path_importer_cache
  - Unified framework for custom importers
  - Foundation for zipimport and other import mechanisms
  - Importer objects with find_module() and load_module() methods
- 🟡 **import** Failed imports no longer leave partially-initialized modules in sys.modules

### Encoding & Unicode

- 🔴 **Encoding** Source code encoding declarations (PEP 263) - `# -*- coding: UTF-8 -*-`
  - Declare encoding in first or second line
  - Default is 7-bit ASCII
  - Affects Unicode string literals only
- 🔴 **Unicode** Unicode filename support for Windows NT/2000/XP (PEP 277)
  - open() and os functions accept Unicode strings
  - os.listdir() returns Unicode strings when given Unicode path
  - os.getcwdu() returns current directory as Unicode
  - os.path.supports_unicode_filenames boolean for testing support
- 🔴 **codecs** Codec error handling callbacks (PEP 293) - Flexible error processing strategies
  - codecs.register_error() and codecs.lookup_error()
  - New handlers: "backslashreplace", "xmlcharrefreplace"
  - Custom error handlers for encoding/decoding

### File I/O

- 🔴 **file** Universal newline support (PEP 278) - 'U' or 'rU' mode handles all line endings
  - Translates Windows, Mac, Unix line endings to '\n'
  - Works with import and execfile()
  - Can be disabled with --without-universal-newlines

### Pickle Protocol

- 🟡 **pickle** New pickle protocol 2 (PEP 307) - More compact pickling of new-style classes
  - Protocol 0 (text), 1 (binary), 2 (new efficient format)
  - pickle.HIGHEST_PROTOCOL constant
  - New special methods: __getstate__(), __setstate__(), __getnewargs__()
  - Unpickling no longer considered safe - Don't unpickle untrusted data
  - Integer codes for classes (not yet standardized)

### Package Distribution

- 🟡 **distutils** Package Index support (PEP 301) - `python setup.py register` uploads to PyPI
  - Metadata includes name, version, description, maintainer
  - Classifiers keyword for Trove-style categorization
  - Central catalog at pypi.org
- 🟡 **distutils** Extension.depends parameter tracks header file dependencies
- 🟡 **distutils** Respects CC, CFLAGS, CPP, LDFLAGS, CPPFLAGS environment variables

### Standard Library Enhancements

- 🟡 **bsddb** Updated to PyBSDDB 4.1.6 - Full transactional BerkeleyDB interface
  - Old version renamed to bsddb185 (not built by default)
  - Requires database conversion when upgrading
- 🟡 **doctest** Now searches private methods and functions for test cases
  - DocTestSuite() creates unittest.TestSuite from doctests
- 🟡 **array** Supports Unicode character arrays with 'u' format
  - Supports += and *= operators
- 🟡 **getopt** gnu_getopt() function for GNU-style option parsing
  - Options and arguments can be mixed
- 🟡 **grp, pwd, resource** Enhanced tuples with named fields
- 🟡 **gzip** Handles files exceeding 2 GiB
- 🟡 **math** degrees() and radians() conversion functions
  - math.log() accepts optional base argument
- 🟡 **os** Several new POSIX functions: getpgid(), killpg(), lchown(), loadavg(), major(), makedev(), minor(), mknod()
- 🟡 **os** stat() family supports fractional seconds in timestamps
  - os.stat_float_times() enables float return values (default in 2.4)
  - Tuple interface maintains integers for compatibility
- 🟡 **random** sample(population, k) function for random sampling without replacement
  - New Mersenne Twister algorithm - Faster and better studied
- 🟡 **readline** New functions: get_history_item(), get_current_history_length(), redisplay()
- 🟡 **socket** Timeout support with settimeout(t) method
  - socket.timeout exception for operations exceeding timeout
- 🟡 **socket** SSL support on Windows
- 🟡 **time** Pure Python strptime() implementation - Consistent across platforms
- 🟡 **shutil** move(src, dest) function for recursive moves
- 🟡 **sys** sys.api_version exposes PYTHON_API_VERSION
- 🟡 **sys** sys.exc_clear() clears current exception
- 🟡 **gc** gc.get_referents(object) returns list of objects referenced by object
- 🟡 **IDLE** Updated with IDLEfork code - Executes code in subprocess
  - No more manual reload() operations
  - Core code in idlelib package
- 🟡 **imaplib** IMAP over SSL support
- 🟡 **ossaudiodev** Replaces linuxaudiodev - Works on non-Linux OSS platforms
- 🟡 **pyexpat** Parser objects support buffer_text attribute for character data buffering
- 🟡 **dummy_thread, dummy_threading** No-op implementations for platforms without thread support
- 🟡 **Tkinter** Thread-safe with thread-enabled Tcl - Automatic marshalling of cross-thread widget access
- 🟡 **Tkinter** _tkinter returns Python objects instead of only strings - wantobjects() method controls behavior
- 🟡 **UserDict** DictMixin class provides full dict interface from minimal mapping methods
- 🟢 **Tix** Various bug fixes and updates

## Improvements

### Performance

- 🔴 **Performance** 25% faster on pystone benchmark compared to Python 2.2
- 🔴 **Performance** New-style class instance creation faster than classic classes
- 🔴 **Performance** list.sort() extensively rewritten - Significantly faster
- 🔴 **Performance** Karatsuba multiplication for large long integers - Better than O(n*n) scaling
- 🔴 **Performance** SET_LINENO opcode removed - Small speed increase depending on compiler
- 🟡 **Performance** xrange() objects have dedicated iterator - Slightly faster than range()
- 🟡 **Performance** Various hotspot optimizations - Inlining and code removal

### Class & Object Model

- 🟡 **MRO** Method resolution order changed to C3 algorithm - More consistent linearization for complex hierarchies
  - Only affects new-style classes with complex inheritance
  - Classic classes unchanged

### Threading

- 🟡 **Threading** Check interval increased from 10 to 100 bytecodes - Faster single-threaded apps
  - sys.setcheckinterval(N) and sys.getcheckinterval() for tuning
  - Multithreaded apps may need lower value for better responsiveness

## Deprecations

- 🟡 **rexec, Bastion** Modules declared dead - Import raises RuntimeError
  - New-style classes break restricted execution
  - Known security bugs, rewrite applications immediately
- 🟡 **rotor** Deprecated - Encryption algorithm not secure
  - Use separate AES modules instead
- 🟡 **string exceptions** raise "Error" triggers PendingDeprecationWarning
- 🟡 **None** Using None as variable name triggers SyntaxWarning - May become keyword
- 🟢 **PendingDeprecationWarning** New warning class for features being deprecated
  - Not printed by default, use -Walways::PendingDeprecationWarning::

## Implementation Details

### Interpreter

- 🟡 **Interpreter** -m switch runs module as script - `python -m profile script.py`
- 🟡 **assert** No longer checks __debug__ flag - Can't disable by assigning to __debug__
  - -O switch still generates code without assertions

### C API

- 🟢 **C API** Various internal improvements and additions

### Build System

- 🟢 **Build** Various improvements for different platforms

## Platform & Environment

- 🟢 **MacOS** os.listdir() may return Unicode filenames
- 🟢 **Windows** Native thread-local storage functions used
- 🟢 **FreeBSD** SO_SETFIB socket constant for alternate routing tables
