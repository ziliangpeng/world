# Python 3.0 Release Summary

**Released:** December 3, 2008
**Source:** [Official Python 3.0 Release Notes](https://docs.python.org/3/whatsnew/3.0.html)

## Overview

Python 3.0, also known as "Python 3000" or "Py3K", is the first ever **intentionally backwards incompatible** Python release. This landmark release fundamentally restructures Python around a clean text/bytes separation, removes legacy features accumulated over years, and establishes a solid foundation for the language's future. While most changes fix well-known annoyances and warts, the breaking changes are extensive enough that a source-to-source converter (`2to3`) is provided for migration. This is not an incremental improvement - it's a comprehensive language redesign that requires careful migration planning.

## Major Breaking Changes

### Print is a Function (PEP 3105)

The `print` statement has been replaced with a `print()` function:

```python
# Python 2.x
print "The answer is", 2*2
print x,                    # Trailing comma suppresses newline
print >>sys.stderr, "error"

# Python 3.0
print("The answer is", 2*2)
print(x, end=" ")           # Appends space instead of newline
print("error", file=sys.stderr)
```

The function provides keyword arguments for customization:

```python
print("There are <", 2**32, "> possibilities!", sep="")
# Output: There are <4294967296> possibilities!
```

### Text vs. Data: Unicode by Default

Everything about strings and bytes has fundamentally changed:

**Core Principles:**
- All text is Unicode (`str` type)
- Binary data uses `bytes` type
- Mixing text and data raises `TypeError` immediately
- No more implicit conversions that could hide bugs

```python
# Text (Unicode strings)
s = "hello"  # str type, Unicode
s.encode('utf-8')  # Returns bytes

# Binary data
b = b"hello"  # bytes type
b.decode('utf-8')  # Returns str

# Mutable binary data
ba = bytearray(b"hello")  # Like bytes but mutable
```

**Major implications:**
- File I/O now has explicit text vs. binary modes
- `open()` in text mode (default) uses encoding to convert between `str` and `bytes`
- `sys.stdin`, `sys.stdout`, `sys.stderr` are Unicode text files
- To read/write bytes, use `.buffer` attribute: `sys.stdout.buffer.write(b"data")`
- Filenames are Unicode strings, but APIs accept `bytes` for platform compatibility
- Default source encoding is now UTF-8 (PEP 3120)
- Non-ASCII letters allowed in identifiers (PEP 3131)
- `u"..."` literals removed (all strings are Unicode)
- `b"..."` literals required for bytes
- `StringIO`/`cStringIO` replaced by `io.StringIO`/`io.BytesIO`

### Views and Iterators Everywhere

Many APIs that returned lists now return views or iterators:

```python
# dict methods return views
d = {'a': 1, 'b': 2}
keys = d.keys()      # dict_keys view, not list
# Can't do: keys.sort()
# Instead: sorted(d)

# dict.iterkeys(), iteritems(), itervalues() removed

# map() and filter() return iterators
result = map(func, seq)  # Returns iterator, not list
# Wrap in list() if needed: list(map(func, seq))

# range() behaves like xrange()
r = range(10)  # Returns range object, not list
# xrange() removed

# zip() returns iterator
z = zip(seq1, seq2)  # Returns iterator
```

### Ordering Comparisons

Ordering comparisons have been simplified and made stricter:

```python
# Python 2.x allowed this
1 < ''        # No error in 2.x
None < None   # Returned False in 2.x

# Python 3.0 raises TypeError
1 < ''        # TypeError: '<' not supported
None < None   # TypeError: '<' not supported

# Can no longer sort heterogeneous lists
[1, 'a', None].sort()  # TypeError
```

Comparison functions replaced with key functions:

```python
# Python 2.x
sorted(items, cmp=lambda a, b: cmp(a.lower(), b.lower()))

# Python 3.0
sorted(items, key=str.lower)

# cmp() function removed
# __cmp__() method no longer supported
# Use __lt__(), __eq__(), __gt__(), etc. instead
```

### Integers Unified

Integer types have been unified (PEP 237, PEP 238):

```python
# Only one integer type: int
# Old 'long' type removed, but int behaves like long
x = 999999999999999999999999  # Just 'int', unlimited precision

# Division always returns float
1 / 2       # Returns 0.5, not 0
1 // 2      # Returns 0 (floor division)

# sys.maxint removed
# Use sys.maxsize instead (size of largest list/string)

# repr(long) no longer includes trailing 'L'
repr(100)  # '100', not '100L'

# Octal literals changed
0o720      # New syntax
# 0720     # Old syntax removed
```

## New Syntax Features

### Function Annotations (PEP 3107)

Standardized syntax for annotating function parameters and return values:

```python
def compile(source: str,
            filename: str = "",
            mode: str = "exec") -> CodeType:
    ...

# Annotations accessible via __annotations__
compile.__annotations__
# {'source': str, 'filename': str, 'mode': str, 'return': CodeType}
```

No built-in semantics - framework and tool authors can define meaning.

### Keyword-Only Arguments (PEP 3102)

Force certain arguments to be specified by keyword:

```python
def foo(a, b, *, c, d):
    # c and d must be specified by keyword
    pass

foo(1, 2, c=3, d=4)  # OK
foo(1, 2, 3, 4)      # TypeError

# Bare * for keyword-only args without *args
def bar(a, *, b):
    pass
```

### Extended Iterable Unpacking (PEP 3132)

Star expressions in assignments:

```python
a, *rest = [1, 2, 3, 4]
# a = 1, rest = [2, 3, 4]

*rest, a = [1, 2, 3, 4]
# rest = [1, 2, 3], a = 4

a, *middle, b = range(5)
# a = 0, middle = [1, 2, 3], b = 4
```

### Dictionary and Set Comprehensions

```python
# Dictionary comprehensions (PEP 274)
{k: v for k, v in items}

# Set literals
{1, 2, 3}  # set literal
{}         # Still empty dict, use set() for empty set

# Set comprehensions
{x for x in items}
```

### New Literal Formats

```python
# Binary literals
0b1010    # 10 in decimal
bin(10)   # '0b1010'

# New octal syntax (required)
0o720     # Octal
# 0720 no longer works

# Bytes literals
b"hello"
b'world'
br"raw\bytes"
```

### nonlocal Statement (PEP 3104)

Access variables in enclosing (non-global) scope:

```python
def outer():
    x = 0
    def inner():
        nonlocal x  # Refer to outer's x
        x += 1
    inner()
    print(x)  # 1
```

### Exception Handling Changes

New exception syntax (PEP 3109, PEP 3110):

```python
# Catching exceptions
# Old: except SomeException, e:
# New: except SomeException as e:
try:
    ...
except ValueError as e:
    ...

# Raising exceptions
# Old: raise Exception, args
# New: raise Exception(args)
raise ValueError("invalid value")

# Exception chaining (PEP 3134)
raise NewException() from original_exception
```

### New Metaclass Syntax (PEP 3115)

```python
# Python 2.x
class C:
    __metaclass__ = M
    ...

# Python 3.0
class C(metaclass=M):
    ...
```

## Syntax Removals

- **Backticks removed**: Use `repr()` instead of `` `x` ``
- **`<>` operator removed**: Use `!=` instead
- **Tuple parameter unpacking removed** (PEP 3113): Can't write `def foo(a, (b, c)):`
- **Classic classes removed**: All classes are new-style
- **`u"..."` string prefix removed**: All strings are Unicode
- **Integer literal `L` suffix removed**: `123L` is invalid
- **`from module import *` only at module level**: Not allowed in functions
- **Relative imports require explicit syntax**: Must use `from .module import name`
- **List comprehension old form removed**: `[... for x in item1, item2]` - use `[... for x in (item1, item2)]`

## Exception System Overhaul

### Exception Hierarchy Changes (PEP 352)

- All exceptions must derive from `BaseException`
- Most exceptions should derive from `Exception`
- `StandardError` removed
- String exceptions completely dead
- Exceptions no longer behave as sequences (use `.args` attribute)

### Exception Chaining (PEP 3134)

Implicit and explicit exception chaining:

```python
# Implicit: exception in handler
try:
    ...
except KeyError:
    # Exception here creates chain
    raise ValueError()  # Original KeyError stored in __context__

# Explicit: use 'from'
try:
    ...
except KeyError as e:
    raise ValueError() from e  # e stored in __cause__
```

Exception objects now have `__traceback__` attribute containing full traceback.

## Standard Library Changes

### Major Reorganizations (PEP 3108)

**Modules renamed for PEP 8 compliance:**

| Old Name | New Name |
|----------|----------|
| `_winreg` | `winreg` |
| `ConfigParser` | `configparser` |
| `copy_reg` | `copyreg` |
| `Queue` | `queue` |
| `SocketServer` | `socketserver` |
| `repr` | `reprlib` |
| `__builtin__` | `builtins` |

**Modules grouped into packages:**

- `dbm` package: Unified `anydbm`, `dbhash`, `dbm`, `dumbdbm`, `gdbm`, `whichdb`
- `html` package: `HTMLParser`, `htmlentitydefs`
- `http` package: `httplib`, `BaseHTTPServer`, `CGIHTTPServer`, `SimpleHTTPServer`, `Cookie`, `cookielib`
- `tkinter` package: All Tkinter-related modules
- `urllib` package: `urllib`, `urllib2`, `urlparse`, `robotparse`
- `xmlrpc` package: `xmlrpclib`, `DocXMLRPCServer`, `SimpleXMLRPCServer`

**Accelerated/pure Python unification:**
- Import `pickle` (not `cPickle`) - automatically uses C version if available
- `StringIO` is now `io.StringIO` class

### Major Removals

**Removed modules:**
- `gopherlib`, `md5`, `sha`, `rgbimg`, `imageop` (obsolete)
- `bsddb3` (moved to external package)
- `sets` module (use built-in `set()`)
- `new` module
- Platform-specific: Irix, BeOS, Mac OS 9 support

**Removed from sys:**
- `sys.exitfunc()` - use `atexit` module
- `sys.exc_clear()`
- `sys.exc_type`, `sys.exc_value`, `sys.exc_traceback`
- `sys.maxint` - use `sys.maxsize`

**Removed functions:**
- `apply()` - use `f(*args)` instead
- `callable()` - use `isinstance(f, collections.Callable)`
- `coerce()` - no longer needed without classic classes
- `execfile()` - use `exec(open(fn).read())`
- `file` type - use `open()`
- `reduce()` - moved to `functools.reduce()`
- `reload()` - use `imp.reload()`
- `dict.has_key()` - use `in` operator
- `os.tmpnam()`, `os.tempnam()`, `os.tmpfile()` - use `tempfile` module
- `string.letters`, `string.lowercase`, `string.uppercase` - use `string.ascii_letters`, etc.

## Operators and Special Methods

**Renamed/changed methods:**
- `__nonzero__()` → `__bool__()`
- `next()` method → `__next__()` (PEP 3114)
- `func_*` attributes → `__*__` (e.g., `func_name` → `__name__`)
- `__getslice__()`, `__setslice__()`, `__delslice__()` removed - use `__getitem__(slice(i, j))`

**Removed special methods:**
- `__cmp__()` - use rich comparison methods (`__lt__`, `__eq__`, etc.)
- `__oct__()`, `__hex__()` - use `__index__()`
- `__members__`, `__methods__` removed

**Behavior changes:**
- Unbound methods removed - referencing method as class attribute returns plain function
- `!=` now returns opposite of `==` (unless `==` returns `NotImplemented`)

## Builtin Function Changes

### New/Changed Builtins

- `super()` can be called without arguments (PEP 3135) - automatically determines class and instance
- `raw_input()` renamed to `input()` (PEP 3111) - old `input()` removed, use `eval(input())`
- `next()` builtin added to call `__next__()`
- `round()` now rounds halfway cases to even (banker's rounding): `round(2.5)` returns `2`
- `round()` returns integer for single argument
- `intern()` moved to `sys.intern()`
- `bin()` builtin added for binary representation

### New I/O System (PEP 3116)

The `io` module is now the standard I/O system:

```python
# open() is now io.open()
f = open('file.txt', 'r', encoding='utf-8')

# New keyword arguments:
# - encoding: text encoding
# - errors: error handling
# - newline: newline handling
# - closefd: close file descriptor

# Access binary layer of text file
f.buffer  # Binary file object

# Invalid mode raises ValueError, not IOError
```

**Buffer protocol:**
- `buffer()` builtin removed
- `memoryview()` provides similar functionality (PEP 3118)

## Features Already in Python 2.6

These features were backported to Python 2.6 to ease transition:

- `with` statement (PEP 343)
- Advanced string formatting with `.format()` (PEP 3101)
- `b"..."` byte literals (PEP 3112)
- Abstract Base Classes (PEP 3119) - `abc` module
- `multiprocessing` module (PEP 3371)
- Binary literals `0b1010` (PEP 3127)
- Class decorators (PEP 3129)
- Numbers module and numeric tower (PEP 3141)
- `fractions` module

## String Formatting

PEP 3101 introduces new `.format()` method:

```python
# Old % operator (still works but deprecated)
"Hello %s" % name

# New .format() method
"Hello {}".format(name)
"Hello {name}".format(name="World")
"{0} {1}".format(first, second)
```

The `%` operator will be deprecated in Python 3.1 and eventually removed.

## C API Changes

**Major C API changes (incomplete list):**

- PEP 3118: New buffer API
- PEP 3121: Extension module initialization & finalization
- PEP 3123: `PyObject_HEAD` conforms to standard C
- Removed: `PyNumber_Coerce()`, `PyNumber_CoerceEx()`, `PyMember_Get()`, `PyMember_Set()`
- Removed: `METH_OLDARGS`, `WITH_CYCLE_GC`
- Renamed: `nb_nonzero` slot → `nb_bool`
- New: `PyImport_ImportModuleNoBlock()` - non-blocking import
- No more restricted execution support

**Platform support dropped:**
- Mac OS 9, BeOS, RISCOS, Irix, Tru64

## Performance

Python 3.0 runs the pystone benchmark approximately **10% slower** than Python 2.5. The biggest cause is likely the removal of special-casing for small integers. Performance improvements are expected in future releases after 3.0.

## Migration Strategy

### Recommended Porting Process

1. **Start with excellent test coverage** (prerequisite)
2. **Port to Python 2.6** - ensure all tests pass
3. **Use `-3` flag in Python 2.6** - enables Python 3.0 warnings
4. **Fix all warnings** until tests pass without warnings
5. **Run `2to3` translator** on source tree
6. **Test under Python 3.0** and fix remaining issues

### The `2to3` Tool

A source-to-source converter that automatically translates Python 2.x code to Python 3.0. It handles most mechanical transformations:
- `print` statements → `print()` function calls
- Exception syntax updates
- Import renames
- Many more automatic transformations

**Not recommended:** Writing code that runs unchanged under both Python 2.6 and 3.0 requires very contorted style. Better to maintain 2.x version and run `2to3` for 3.x distribution.

## List Comprehension Semantics

List comprehensions are now closer to syntactic sugar for generator expressions inside `list()`:

```python
# Python 2.x: loop variable leaked
[x for x in range(10)]
print(x)  # x = 9

# Python 3.0: loop variable is local
[x for x in range(10)]
print(x)  # NameError: name 'x' is not defined
```

## Minor Changes

- `as` and `with` are now reserved words
- `True`, `False`, `None` are reserved words
- Ellipsis (`...`) can be used as atomic expression anywhere
- Ellipsis must be spelled as `...` (not `. . .`)
- Keyword arguments allowed after base classes in class definition
- `exec` is now a function, not a keyword (but function syntax worked in 2.x too)
- `tokenize` module now works with bytes
- Array type `'c'` removed - use `'b'` for bytes or `'u'` for Unicode
- `operator.sequenceIncludes()` and `operator.isCallable()` removed
- `thread` module: `acquire_lock()`/`release_lock()` → `acquire()`/`release()`
- `random.jumpahead()` removed

## Key Takeaways

1. **This is a breaking release** - Python 3.0 is intentionally incompatible with Python 2.x
2. **Text/bytes separation is fundamental** - All text is Unicode, binary data is bytes, never mixed
3. **Views and iterators replace lists** - More memory efficient but requires different patterns
4. **Print is a function** - Most visible change for beginners
5. **Integers unified** - No separate `int`/`long`, division returns float
6. **Exception handling modernized** - New syntax, chaining, better introspection
7. **Standard library reorganized** - More logical structure, PEP 8 compliance
8. **Migration path provided** - Python 2.6 compatibility and `2to3` tool
9. **Performance temporarily regressed** - About 10% slower than Python 2.5
10. **Foundation for future** - Clean base for future improvements

Python 3.0 represents a once-in-a-lifetime opportunity to fix fundamental design issues and remove accumulated cruft. While the transition requires effort, it establishes Python on a solid foundation for decades to come.
