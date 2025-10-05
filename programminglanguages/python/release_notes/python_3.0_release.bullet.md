# Python 3.0 Release Notes

**Released:** December 3, 2008
**EOL:** January 2009 (superseded by Python 3.1)

## Major Highlights

Python 3.0 "Py3K" is the first ever **intentionally backwards incompatible** Python release - a comprehensive language redesign:

1. **Print becomes a function (PEP 3105)** - `print "x"` → `print(x)` with keyword arguments for flexibility
2. **Text vs. bytes separation** - All text is Unicode (`str`), binary data is `bytes`, no implicit mixing
3. **Views and iterators everywhere** - `dict.keys()`, `map()`, `filter()`, `range()`, `zip()` return views/iterators, not lists
4. **Integer division returns float** - `1/2` → `0.5`, use `1//2` for floor division (PEP 238)
5. **Integers unified** - Only `int` type with unlimited precision, `long` removed (PEP 237)
6. **Strict ordering comparisons** - `1 < ''` raises TypeError instead of arbitrary ordering
7. **Classic classes removed** - All classes are new-style by default

## Breaking Changes

### Print Statement
- 🔴 **Syntax** `print` statement removed - Must use `print()` function (PEP 3105)
  - `print "x"` → `print(x)`
  - `print x,` → `print(x, end=" ")`
  - `print >>sys.stderr, "error"` → `print("error", file=sys.stderr)`
  - New: `sep` and `end` keyword arguments for customization

### Text and Binary Data
- 🔴 **Strings** All text is Unicode by default - `str` type is Unicode, `bytes` type for binary data
  - Mixing text and bytes raises TypeError immediately
  - Must explicitly convert: `str.encode()` → bytes, `bytes.decode()` → str
- 🔴 **Strings** `u"..."` prefix removed - All string literals are Unicode
- 🔴 **Strings** `b"..."` prefix required for bytes literals
- 🔴 **Strings** `basestring` type removed - Use `str` instead
- 🔴 **io** Files in text mode (default) require encoding - Binary mode uses `b` flag
- 🔴 **io** `sys.stdin`, `sys.stdout`, `sys.stderr` are Unicode text files - Use `.buffer` attribute for bytes
- 🟡 **io** `StringIO` and `cStringIO` modules removed - Use `io.StringIO` or `io.BytesIO`
- 🟡 **Strings** Source files default to UTF-8 encoding (PEP 3120)
- 🟡 **Strings** Non-ASCII letters allowed in identifiers (PEP 3131)
- 🟡 **Strings** `repr()` no longer escapes non-ASCII characters (PEP 3138)
- 🟡 **Strings** Raw string literals treat `\u` and `\U` literally

### Collections and Iterators
- 🔴 **dict** `dict.keys()`, `dict.items()`, `dict.values()` return views, not lists
  - Can't call `.sort()` on views - use `sorted(d)` instead
- 🔴 **dict** `dict.iterkeys()`, `dict.iteritems()`, `dict.itervalues()` removed - Views are already lazy
- 🔴 **dict** `dict.has_key()` removed - Use `in` operator
- 🔴 **builtins** `map()` and `filter()` return iterators, not lists - Wrap in `list()` if needed
- 🔴 **builtins** `range()` behaves like old `xrange()` - Returns iterator, not list
- 🔴 **builtins** `xrange()` removed - Use `range()`
- 🔴 **builtins** `zip()` returns iterator, not list
- 🔴 **sets** `sets` module removed - Use built-in `set()` type

### Ordering and Comparisons
- 🔴 **Syntax** Ordering operators raise TypeError for incomparable types - `1 < ''`, `None < None` are errors
- 🔴 **Syntax** Cannot sort heterogeneous lists - All elements must be comparable
- 🔴 **builtins** `cmp()` function removed - Use `(a > b) - (a < b)` or comparison key
- 🔴 **Data Model** `__cmp__()` method removed - Use `__lt__()`, `__eq__()`, etc.
- 🔴 **builtins** `sorted()` and `list.sort()` no longer accept `cmp` argument - Use `key` argument
  - `key` and `reverse` are now keyword-only arguments

### Integer Changes
- 🔴 **Integers** Integer division returns float (PEP 238) - `1/2` → `0.5`, use `1//2` for floor division
- 🔴 **Integers** `long` type removed - Only `int` with unlimited precision (PEP 237)
- 🔴 **Integers** `sys.maxint` removed - Use `sys.maxsize` for practical list/string index limit
- 🔴 **Integers** Integer literals cannot have trailing `L` or `l`
- 🔴 **Integers** `repr(long)` no longer includes trailing `L`
- 🔴 **Syntax** Old octal literals `0720` removed - Must use `0o720`

### Exception Handling
- 🔴 **Exceptions** All exceptions must derive from `BaseException` (PEP 352) - String exceptions dead
- 🔴 **Exceptions** `StandardError` removed
- 🔴 **Syntax** `except exc, var` syntax removed - Must use `except exc as var` (PEP 3110)
- 🔴 **Syntax** `raise Exception, args` syntax removed - Must use `raise Exception(args)` (PEP 3109)
- 🟡 **Exceptions** Variable in `except ... as var` is deleted when block exits
- 🟡 **Exceptions** Exceptions no longer behave as sequences - Use `.args` attribute

### Syntax Removals
- 🔴 **Syntax** Backticks removed - Use `repr()` instead of `` `x` ``
- 🔴 **Syntax** `<>` operator removed - Use `!=`
- 🔴 **Syntax** Tuple parameter unpacking removed (PEP 3113) - Can't write `def foo(a, (b, c)):`
- 🔴 **Syntax** Classic classes removed - All classes are new-style
- 🔴 **Syntax** `from module import *` only allowed at module level - Not in functions
- 🔴 **Syntax** Relative imports require explicit dot syntax - `from .module import name` (PEP 328)
- 🔴 **Syntax** List comprehension form `[... for x in a, b]` removed - Use `[... for x in (a, b)]`
- 🟡 **Syntax** `True`, `False`, `None` are now reserved words
- 🟡 **Syntax** `as` and `with` are now reserved words

### Metaclass Syntax
- 🔴 **Syntax** `__metaclass__` attribute removed - Use `class C(metaclass=M):` syntax (PEP 3115)
- 🔴 **Syntax** Module-global `__metaclass__` no longer supported

### Builtin Functions Removed
- 🔴 **builtins** `apply()` removed - Use `f(*args)`
- 🔴 **builtins** `callable()` removed - Use `isinstance(f, collections.Callable)`
- 🔴 **builtins** `coerce()` removed - No longer needed without classic classes
- 🔴 **builtins** `execfile()` removed - Use `exec(open(fn).read())`
- 🔴 **builtins** `file` type removed - Use `open()`
- 🔴 **builtins** `reduce()` removed - Moved to `functools.reduce()`
- 🔴 **builtins** `reload()` removed - Use `imp.reload()`
- 🔴 **builtins** `buffer()` removed - Use `memoryview()` (PEP 3118)
- 🟡 **builtins** `intern()` moved to `sys.intern()`
- 🟡 **builtins** `raw_input()` renamed to `input()` (PEP 3111) - Old `input()` removed

### Standard Library Reorganization
- 🔴 **stdlib** Many modules renamed for PEP 8 compliance:
  - `ConfigParser` → `configparser`
  - `Queue` → `queue`
  - `SocketServer` → `socketserver`
  - `_winreg` → `winreg`
  - `copy_reg` → `copyreg`
  - `repr` → `reprlib`
  - `__builtin__` → `builtins`
- 🔴 **stdlib** Modules grouped into packages:
  - `dbm` package (from `anydbm`, `dbhash`, `dbm`, `dumbdbm`, `gdbm`, `whichdb`)
  - `html` package (from `HTMLParser`, `htmlentitydefs`)
  - `http` package (from `httplib`, `BaseHTTPServer`, `CGIHTTPServer`, `SimpleHTTPServer`, `Cookie`, `cookielib`)
  - `tkinter` package (from all Tkinter modules)
  - `urllib` package (from `urllib`, `urllib2`, `urlparse`, `robotparse`)
  - `xmlrpc` package (from `xmlrpclib`, `DocXMLRPCServer`, `SimpleXMLRPCServer`)
- 🔴 **stdlib** Import unified modules directly - `pickle` (not `cPickle`), `profile` - auto-use C version if available
- 🟡 **stdlib** Many obsolete modules removed (PEP 3108) - `gopherlib`, `md5`, `sha`, `rgbimg`, `imageop`, etc.
- 🟡 **stdlib** `bsddb3` removed from stdlib - Available as external package
- 🟡 **stdlib** `new` module removed

### sys Module Changes
- 🔴 **sys** `sys.exitfunc()` removed - Use `atexit` module
- 🔴 **sys** `sys.exc_clear()` removed
- 🔴 **sys** `sys.exc_type`, `sys.exc_value`, `sys.exc_traceback` removed
  - `sys.last_type`, `sys.last_value`, `sys.last_traceback` still available

### Special Methods Renamed/Removed
- 🔴 **Data Model** `__nonzero__()` renamed to `__bool__()`
- 🔴 **Data Model** `next()` method renamed to `__next__()` (PEP 3114)
- 🔴 **Data Model** `func_*` attributes renamed to `__*__` (e.g., `func_name` → `__name__`)
- 🔴 **Data Model** `__getslice__()`, `__setslice__()`, `__delslice__()` removed - Use `__getitem__(slice(i,j))`
- 🔴 **Data Model** `__oct__()` and `__hex__()` removed - Use `__index__()`
- 🔴 **Data Model** `__members__` and `__methods__` removed
- 🟡 **Data Model** Unbound methods removed - Class attribute method access returns plain function
- 🟡 **Data Model** `!=` now returns opposite of `==` (unless `==` returns NotImplemented)

### Other Module Changes
- 🟡 **array** `array.read()` and `array.write()` removed - Use `fromfile()` and `tofile()`
- 🟡 **array** `'c'` typecode removed - Use `'b'` for bytes or `'u'` for Unicode
- 🟡 **operator** `operator.sequenceIncludes()` and `operator.isCallable()` removed
- 🟡 **thread** `acquire_lock()` and `release_lock()` removed - Use `acquire()` and `release()`
- 🟡 **random** `random.jumpahead()` removed
- 🟡 **os** `os.tmpnam()`, `os.tempnam()`, `os.tmpfile()` removed - Use `tempfile` module
- 🟡 **tokenize** `tokenize` module now works with bytes - Entry point is `tokenize.tokenize()`
- 🟡 **string** `string.letters`, `string.lowercase`, `string.uppercase` removed - Use `string.ascii_letters`, etc.

## New Features

### Language Syntax

- 🔴 **Syntax** Function annotations (PEP 3107) - Standardized parameter and return value annotations
  - `def foo(a: int, b: str) -> bool:`
  - Accessible via `__annotations__` attribute
  - No built-in semantics, for framework use
- 🔴 **Syntax** Keyword-only arguments (PEP 3102) - Parameters after `*args` or bare `*` must be keyword-only
  - `def foo(a, b, *, c, d):`
  - `def bar(a, *, b):` (no *args, just keyword-only)
- 🔴 **Syntax** Extended iterable unpacking (PEP 3132) - Star expressions in assignments
  - `a, *rest, b = sequence`
  - `*rest, a = sequence`
- 🔴 **Syntax** Dictionary comprehensions (PEP 274) - `{k: v for k, v in items}`
- 🔴 **Syntax** Set literals - `{1, 2, 3}` (note: `{}` is still empty dict, use `set()`)
- 🔴 **Syntax** Set comprehensions - `{x for x in items}`
- 🔴 **Syntax** `nonlocal` statement (PEP 3104) - Assign to variable in enclosing non-global scope
- 🟡 **Syntax** Binary literals - `0b1010`, `bin()` function
- 🟡 **Syntax** New octal syntax required - `0o720` (old `0720` removed)
- 🟡 **Syntax** Bytes literals - `b"hello"`, `b'world'`, `br"raw"`
- 🟡 **Syntax** Ellipsis `...` can be used as atomic expression anywhere (not just slices)
- 🟡 **Syntax** Keyword arguments allowed after base classes in class definition
- 🟡 **Syntax** `exec` is now a function, not keyword (syntax compatible with 2.x)

### Exception Features

- 🟡 **Exceptions** Exception chaining (PEP 3134) - Implicit and explicit chaining
  - Implicit: Exceptions in handler set `__context__`
  - Explicit: `raise NewException() from original` sets `__cause__`
  - Traceback walks chain and prints all exceptions
- 🟡 **Exceptions** Exception objects store traceback in `__traceback__` attribute
- 🟡 **Exceptions** Can no longer explicitly specify traceback in raise - Assign to `__traceback__` instead

### Builtin Changes

- 🔴 **builtins** `super()` can be called without arguments (PEP 3135) - Automatically determines class and instance
- 🔴 **builtins** `input()` replaces `raw_input()` (PEP 3111) - Reads line from stdin, returns string
  - Old `input()` behavior: use `eval(input())`
- 🔴 **builtins** `next()` builtin function added - Calls `__next__()` on iterator
- 🟡 **builtins** `round()` uses banker's rounding - Halfway cases round to even (e.g., `round(2.5)` → `2`)
- 🟡 **builtins** `round()` returns integer for single argument
- 🟡 **builtins** `bin()` builtin added for binary representation
- 🟡 **builtins** `memoryview()` replaces `buffer()` (PEP 3118)
- 🟡 **Data Model** `bytearray` type added - Mutable bytes-like object

### I/O System

- 🔴 **io** New I/O library (PEP 3116) - `io` module is now standard
  - `open()` is alias for `io.open()`
  - New keyword arguments: `encoding`, `errors`, `newline`, `closefd`
  - Invalid mode raises `ValueError`, not `IOError`
  - Text file `.buffer` attribute provides binary layer
- 🟡 **io** Buffer protocol revised (PEP 3118) - New buffer API for C extensions

### String Formatting

- 🟡 **Syntax** New `.format()` string method (PEP 3101) - Advanced string formatting
  - `"Hello {name}".format(name="World")`
  - `"{0} {1}".format(a, b)`
  - `%` operator still works but will be deprecated in 3.1

### List Comprehension Semantics

- 🟡 **Syntax** List comprehension variables no longer leak - Closer to generator expression semantics
  - `[x for x in range(10)]` - `x` not defined outside comprehension

## Improvements

### Better Error Messages

- 🟡 **Error Messages** Windows extension load errors improved - Better messages for error codes

## Features Already in Python 2.6

These were backported to Python 2.6 to ease migration:

- 🟢 **Syntax** `with` statement (PEP 343) - Now standard, no need to import from `__future__`
- 🟢 **Syntax** `print()` function available (PEP 3105) - Import from `__future__`
- 🟢 **Syntax** `except exc as var` syntax (PEP 3110) - `as` variant available
- 🟢 **Syntax** `b"..."` byte literals (PEP 3112)
- 🟢 **Syntax** Binary literals `0b1010` and `bin()` function (PEP 3127)
- 🟢 **Syntax** Class decorators (PEP 3129)
- 🟢 **abc** Abstract Base Classes (PEP 3119) - `abc` module and collection ABCs
- 🟢 **multiprocessing** `multiprocessing` module (PEP 3371)
- 🟢 **numbers** Numeric tower and `numbers` module (PEP 3141)
- 🟢 **fractions** `fractions` module for rational numbers
- 🟢 **stdlib** Relative imports with explicit syntax (PEP 366)
- 🟢 **stdlib** Per-user site-packages directory (PEP 370)
- 🟢 **Syntax** `.format()` string method (PEP 3101)

## Implementation Details

### C API

- 🟡 **C API** New buffer API (PEP 3118) - Revised buffer protocol
- 🟡 **C API** Extension module initialization & finalization (PEP 3121)
- 🟡 **C API** `PyObject_HEAD` conforms to standard C (PEP 3123)
- 🟡 **C API** `PyImport_ImportModuleNoBlock()` added - Non-blocking import
- 🟡 **C API** `nb_nonzero` slot renamed to `nb_bool`
- 🟡 **C API** `PyNumber_Coerce()`, `PyNumber_CoerceEx()`, `PyMember_Get()`, `PyMember_Set()` removed
- 🟡 **C API** `METH_OLDARGS` and `WITH_CYCLE_GC` removed
- 🟡 **C API** Restricted execution support removed

### Platform Support

- 🟡 **Platform** Support dropped for Mac OS 9, BeOS, RISCOS, Irix, Tru64

### Performance

- 🔴 **Performance** Python 3.0 runs ~10% slower than Python 2.5 on pystone benchmark
  - Main cause: removal of small integer special-casing
  - Performance improvements expected in future releases

## Migration Tools

- 🔴 **Tools** `2to3` source-to-source translator - Automates most Python 2.x → 3.0 conversions
  - Handles print statements, exception syntax, import renames, and more
  - Recommended: Keep 2.x as source, run 2to3 for 3.x distribution
- 🔴 **Tools** Python 2.6 `-3` flag - Warns about Python 3.0 incompatibilities
  - Use to prepare code before running 2to3

## Porting Strategy

Recommended migration path:

1. **Prerequisite:** Start with excellent test coverage
2. **Port to Python 2.6** - Ensure all tests pass
3. **Run with `-3` flag** - Fix all Python 3.0 warnings
4. **Run `2to3` translator** - Convert source tree
5. **Test under Python 3.0** - Fix remaining issues until tests pass

Not recommended: Writing code that runs on both 2.6 and 3.0 unchanged requires very contorted style.
