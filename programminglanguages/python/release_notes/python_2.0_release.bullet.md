# Python 2.0 Release Notes

**Released:** October 16, 2000
**Superseded by Python 2.1:** April 2001 (~6 months later)
**Python 2.x series EOL:** January 1, 2020

## Major Highlights

Python 2.0 introduced several foundational features that shaped Python for the next two decades:

1. **Unicode support** - New fundamental data type for 16-bit Unicode strings, with encoding/decoding infrastructure
2. **List comprehensions** - Concise syntax for list transformations inspired by Haskell
3. **Augmented assignment operators** - +=, -=, *=, and other compound assignment operators
4. **Garbage collector for cycles** - Automatic detection and cleanup of circular references
5. **String methods** - String manipulation moved from string module to methods on string objects
6. **SourceForge migration** - Move to open development model with public CVS and bug tracking

## New Features

### Core Language

- 🔴 **Syntax** List comprehensions - `[x for x in seq if condition]` syntax for transforming lists
- 🔴 **Syntax** Augmented assignment operators - `+=`, `-=`, `*=`, `/=`, `%=`, `**=`, `&=`, `|=`, `^=`, `>>=`, `<<=`
- 🔴 **Unicode** New unicode string type with u"string" literal syntax
  - 16-bit characters supporting 65,536 distinct characters
  - New escape sequence `\uHHHH` for arbitrary Unicode characters
  - `unichr()`, `unicode()` built-in functions
  - Codec API for encoding/decoding
- 🟡 **Syntax** Function call unpacking - `f(*args, **kwargs)` as shorthand for `apply(f, args, kwargs)`
- 🟡 **Syntax** Print to file - `print >> file, "output"` syntax for redirecting output
- 🟡 **Syntax** Import renaming - `import module as name` and `from module import name as othername`
- 🟡 **Syntax** Format string %r - New '%r' format uses `repr()` for symmetry with '%s'
- 🟡 **Data Model** `__contains__()` magic method - Custom implementation of `in` operator
- 🟢 **Exceptions** New `UnboundLocalError` exception - Subclass of NameError for unassigned local variables
- 🟢 **Exceptions** New `TabError` and `IndentationError` exceptions - Both subclasses of SyntaxError

### Garbage Collection

- 🔴 **gc** New garbage collection module for cycle detection
  - Periodically detects and deletes inaccessible reference cycles
  - `gc` module provides functions to trigger collection, get statistics, tune parameters
  - Can be disabled at compile time with `--without-cycle-gc` flag
  - Fixes memory leaks from circular references that reference counting couldn't handle

### Built-in Functions & Types

- 🟡 **Built-ins** New `zip()` function - Combines sequences into list of tuples, truncates to shortest sequence
- 🟡 **Built-ins** `int()` and `long()` accept base parameter - `int('123', 16)` returns 291
- 🟡 **sys** New `sys.version_info` tuple - `(major, minor, micro, level, serial)` for programmatic version checks
- 🟡 **dict** New `setdefault(key, default)` method - Returns and inserts default if key missing
- 🟡 **sys** `sys.getrecursionlimit()` and `sys.setrecursionlimit()` - Dynamic recursion depth control (default 1000)
- 🟢 **Comparisons** Recursive data structures can now be compared without crashing
- 🟢 **Comparisons** Comparison operations can now raise exceptions from user-defined `__cmp__()`

### String Methods

- 🔴 **str** String manipulation moved to string methods
  - Methods like `capitalize()`, `replace()`, `find()` now available on string objects
  - Works for both 8-bit strings and Unicode strings
  - Old `string` module remains for backwards compatibility but acts as front-end
- 🟡 **str** New `startswith()` and `endswith()` methods - Cleaner than slice comparisons
- 🟡 **str** `join()` method - `s.join(seq)` equivalent to old `string.join(seq, s)`

## Improvements

### Performance & Internals

- 🟡 **Performance** Deletion of deeply nested structures no longer crashes - Rewrote recursive deletion algorithm
- 🟡 **Bytecode** Increased bytecode limit from 2^16 to 2^32 - Removes size limits on literal lists/dicts
- 🟢 **Threading** Windows threading performance improved - Only 10% slower than unthreaded (was 2x slower in 1.5.2)
- 🟢 **Threading** MacOS POSIX threading support via GUSI
- 🟢 **Threading** User-space GNU pth library support added

### Error Messages

- 🟢 **Error Messages** Better AttributeError and NameError messages - Now says "'Spam' instance has no attribute 'eggs'" instead of just "eggs"

### Platform Support

- 🟡 **Platform** Windows 64-bit (Itanium) support - Port to 64-bit Windows completed
- 🟡 **Platform** Windows CE support - Available at pythonce.sourceforge.net
- 🟡 **Platform** Darwin/MacOS X initial support - Dynamic loading works with `--with-dyld --with-suffix=.x`

## New Modules

- 🟡 **atexit** Register functions to call on interpreter exit - Replaces direct `sys.exitfunc` usage
- 🔴 **codecs** Codec registry and base classes for encodings - Part of Unicode support
- 🟡 **encodings** Standard encoding implementations - Part of Unicode support
- 🔴 **unicodedata** Unicode character properties database - Category, bidirectional properties, etc.
- 🟡 **filecmp** File and directory comparison - Supersedes `cmp`, `cmpcache`, `dircmp` modules
- 🟡 **gettext** Internationalization (I18N) and localization (L10N) - GNU gettext message catalog interface
- 🟢 **linuxaudiodev** Linux `/dev/audio` device support - Twin to existing `sunaudiodev`
- 🟡 **mmap** Memory-mapped file support - Works on both Windows and Unix
- 🟡 **pyexpat** Expat XML parser interface - Part of new XML support
- 🟢 **robotparser** Parse `robots.txt` files - For writing polite web spiders
- 🟢 **tabnanny** Check Python source for ambiguous indentation - Can be run as script
- 🟡 **UserString** Base class for string-like objects - Similar to UserDict and UserList
- 🟡 **webbrowser** Platform-independent browser launching - Open URLs in system browser
- 🟡 **_winreg** Windows registry interface - Unicode support, adapted from PythonWin
- 🟡 **zipfile** Read and write ZIP-format archives - Not to be confused with gzip
- 🟢 **imputil** Simplified custom import hooks - Simpler than existing `ihooks` module

## Module Changes

### XML Support

- 🔴 **xml** New XML package with SAX2 and DOM support
  - `xml.sax` module for event-driven SAX2 parsing
  - `xml.dom.minidom` for lightweight DOM Level 1 implementation with namespace support
  - Compatible with PyXML 0.6.0+ which provides additional features (4DOM, xmlproc, sgmlop)

### Standard Library Improvements

- 🟡 **socket** OpenSSL support for SSL/TLS encryption - New `socket.ssl()` function
- 🟡 **httplib** Rewritten to support HTTP/1.1 - Backward compatible with 1.5 version
- 🟡 **urllib** HTTPS URL support - Works with OpenSSL
- 🟡 **Tkinter** Updated to support Tcl/Tk 8.1, 8.2, 8.3 - Dropped support for 7.x versions
  - Unicode string display support
  - Performance optimization for `create_line` and `create_polygon`
- 🟡 **curses** Greatly extended with ncurses and SYSV curses features
  - Color support, alternative character sets, pads, mouse support
  - No longer compatible with BSD curses only systems
- 🔴 **re** New SRE (Secret Labs Regular Expression) engine
  - Supports both 8-bit and Unicode strings
  - Written by Fredrik Lundh

## Breaking Changes

- 🔴 **list** Methods like `append()` and `insert()` now require exactly one argument
  - Old: `L.append(1, 2)` appended tuple `(1,2)`
  - New: Must use `L.append((1, 2))` explicitly
  - Can restore old behavior by defining `NO_STRICT_LIST_APPEND` (not recommended)
- 🟡 **Syntax** `\x` escape now takes exactly 2 hex digits - Previously consumed all following hex digits
- 🟡 **long** `str()` of long integers no longer has trailing 'L' - `repr()` still includes it
- 🟡 **float** `repr()` of floats uses `%.17g` format - Shows more precision than `str()` which uses `%.12g`
- 🟡 **long** Long integers more interchangeable with regular integers
  - Can be used for sequence multiplication, slicing, file seeking
  - Accepted in `%d`, `%i`, `%x` format operators
- 🟡 **Exceptions** Standard exceptions always classes now - `-X` command-line option removed

## Deprecated Modules

- 🟢 **stdwin** Removed - Platform-independent windowing toolkit no longer developed
- 🟢 **lib-old** Multiple modules moved to `lib-old` subdirectory
  - Moved: `cmp`, `cmpcache`, `dircmp`, `dump`, `find`, `grep`, `packmail`, `poly`, `util`, `whatsound`, `zmod`
  - Can add `lib-old` to `sys.path` if needed but should update code

## IDLE Improvements

- 🟡 **IDLE** Updated to version 0.6 with numerous improvements
  - UI improvements and syntax highlighting optimizations
  - Enhanced class browser showing top-level functions
  - User-settable tab width with auto-detection of existing conventions
  - Browser integration for Python documentation
  - Command line interface similar to vanilla interpreter
  - Call tips in many places
  - Line/column bar in editor
  - New commands: Check module (Alt-F5), Import module (F5), Run script (Ctrl-F5)

## Development Process Changes

- 🔴 **Release** Moved to SourceForge for development
  - Public CVS repository
  - Bug tracking and patch management tools
  - Increased from ~7 to 27 developers with write access
  - Dramatically increased development velocity
- 🔴 **Release** Python Enhancement Proposal (PEP) process introduced
  - Formal design documents modeled on Internet RFC process
  - PEPs describe new features and collect community consensus
  - 25 PEPs created as of September 2000
- 🟡 **Release** New approval process with +1/+0/-0/-1 voting
  - Similar to Apache group model but advisory
  - Guido van Rossum retains Benevolent Dictator For Life authority

## Implementation Details

### C API Changes

- 🔴 **C API** Python C API version incremented - Extensions compiled for 1.5.2 must be recompiled
- 🟡 **C API** ExtensionClass now supported by `isinstance()` and `issubclass()`
- 🟡 **C API** Argument parsing modernized to use `PyArg_ParseTuple()` - Provides better error messages
- 🟡 **Build** Source now requires ANSI C compiler - K&R C no longer supported
  - Python source converted from K&R to ANSI C
- 🟡 **C API** New module convenience functions - `PyModule_AddObject()`, `PyModule_AddIntConstant()`, `PyModule_AddStringConstant()`
- 🟡 **C API** Wrapper API for Unix signal handlers - `PyOS_getsig()` and `PyOS_setsig()`
- 🟢 **Build** `Python/importdl.c` cleaned up - Platform-specific code moved to `Python/dynload_*.c` files
- 🟢 **Build** Portability headers merged - Multiple `my*.h` files merged into single `Include/pyport.h`
- 🟢 **Memory** Vladimir Marangozov's malloc restructuring - Easy to use custom allocators instead of C's `malloc()`
