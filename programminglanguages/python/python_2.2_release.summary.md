# Python 2.2 Release Summary

**Released:** December 21, 2001
**Source:** [Official Python 2.2 Release Notes](https://docs.python.org/2/whatsnew/2.2.html)

## Overview

Python 2.2 represents the most fundamental transformation of Python since the language's creation. At its core, this release unified types and classes, eliminating the artificial distinction between built-in types implemented in C and user-defined classes. This "cleanup release" removed irregularities that had accumulated over Python's first decade while introducing powerful new features like generators and iterators that would define modern Python programming.

The type/class unification breakthrough enabled subclassing built-in types such as lists, dictionaries, and even integers—something previously impossible. New-style classes introduced descriptors, properties, static methods, class methods, and slots, providing fine-grained control over attribute access and object behavior. Generators, inspired by the Icon programming language, offered a revolutionary way to write iterators that preserved local state between calls. The iterator protocol itself became formalized, allowing any object to define iteration behavior. Additional changes included nested scopes becoming mandatory, long and regular integers beginning their unification, Unicode enhancements, and the controversial start of changing division semantics. Together, these changes modernized Python's object model while maintaining remarkable backward compatibility.

## Major Language Features

### PEPs 252 and 253: Type and Class Unification

Python 2.2's most significant achievement was unifying types and classes. Previously, Python had two fundamentally different object systems: built-in types implemented in C (like lists, dictionaries, and integers) and user-defined classes. This created frustrating limitations—you couldn't subclass a list to add custom methods, forcing developers to use workaround classes like `UserList` that weren't accepted by C code expecting real lists.

**Old-style vs. New-style Classes:**

Python 2.2 introduced new-style classes while maintaining backward compatibility with classic classes. New-style classes are created by inheriting from `object` or any built-in type:

```python
class C(object):
    def __init__(self):
        ...
```

Classes without base classes remained old-style in Python 2.2 for compatibility. The new `object` type serves as the universal base class when no specific built-in type is appropriate.

**Subclassing Built-in Types:**

Built-in type names like `int()`, `float()`, `str()`, `dict()`, and `file()` now serve dual purposes—they're both type objects and factory functions:

```python
>>> int
<type 'int'>
>>> int('123')
123
```

This enables subclassing built-in types to add custom behavior:

```python
class LockableFile(file):
    def lock(self, operation, length=0, start=0, whence=0):
        import fcntl
        return fcntl.lockf(self.fileno(), operation,
                           length, start, whence)
```

The `LockableFile` class adds file locking while remaining usable everywhere built-in file objects are expected—something the obsolete `posixfile` module could never achieve.

### Descriptors

Descriptors formalize an API for controlling attribute access, forming the foundation for properties, static methods, class methods, and slots. A descriptor is an object living inside a class with special methods:

- `__name__`: The attribute's name
- `__doc__`: The attribute's docstring
- `__get__(object)`: Retrieves the attribute value
- `__set__(object, value)`: Sets the attribute value
- `__delete__(object, value)`: Deletes the attribute

When you access `obj.x`, Python actually performs:

```python
descriptor = obj.__class__.x
descriptor.__get__(obj)
```

**Static and Class Methods:**

Descriptors enable static and class methods. Static methods receive no implicit first argument, behaving like regular functions. Class methods receive the class rather than the instance:

```python
class C(object):
    def f(arg1, arg2):
        ...
    f = staticmethod(f)

    def g(cls, arg1, arg2):
        ...
    g = classmethod(g)
```

The descriptor mechanism wraps these methods appropriately, though special syntax for defining them was left for future Python versions.

### Properties

Properties provide a cleaner alternative to `__getattr__()` and `__setattr__()` for computed attributes:

```python
class C(object):
    def get_size(self):
        result = ... computation ...
        return result

    def set_size(self, size):
        ... compute something based on the size
        and set internal state appropriately ...

    # Define a property. The 'delete' method is None,
    # so the attribute can't be deleted.
    size = property(get_size, set_size,
                    None,
                    "Storage size of this instance")
```

Properties avoid the complexity and performance overhead of `__getattr__()`, which gets called for every attribute access and must carefully avoid recursion. Properties also support docstrings, allowing attributes to be documented.

### Slots

The `__slots__` class attribute restricts instance attributes to a predefined set:

```python
>>> class C(object):
...     __slots__ = ('template', 'name')
...
>>> obj = C()
>>> obj.template = 'Test'
>>> print obj.template
Test
>>> obj.newattr = None
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
AttributeError: 'C' object has no attribute 'newattr'
```

Slots prevent accidental attribute creation from typos and potentially enable future memory optimizations by eliminating per-instance dictionaries.

### Multiple Inheritance: The Diamond Rule

New-style classes use a smarter method resolution order for multiple inheritance. Consider this diamond inheritance pattern:

```python
      class A:
        ^   ^  def save(self): ...
       /     \
      /       \
     /         \
    /           \
class B       class C:
    ^         ^  def save(self): ...
     \       /
      \     /
       \   /
        \ /
      class D
```

Classic classes search depth-first left-to-right: D, B, A, C. This finds `A.save()` but never `C.save()`, which might contain critical state-saving logic for C.

New-style classes use a better algorithm:
1. List all base classes following classic order: [D, B, A, C, A]
2. Remove duplicates, keeping only the last occurrence: [D, B, C, A]

Now `D.save()` correctly finds `C.save()`. The `super()` built-in function navigates this resolution order:

```python
class D(B, C):
    def save(self):
        # Call superclass .save()
        super(D, self).save()
        # Save D's private information
        ...
```

### PEP 234: Iterators

Python 2.2 formalized iteration through a protocol, separating it from indexing. Previously, iteration required implementing `__getitem__()`, which conflated sequential access with random indexing.

**The Iterator Protocol:**

Objects implement `__iter__()` to return an iterator. The iterator's `next()` method returns successive values, raising `StopIteration` when exhausted:

```python
>>> L = [1, 2, 3]
>>> i = iter(L)
>>> i.next()
1
>>> i.next()
2
>>> i.next()
3
>>> i.next()
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
StopIteration
```

The `for` statement now uses iterators automatically, maintaining backward compatibility by creating iterators for sequences lacking `__iter__()`.

**Built-in Iterator Support:**

Dictionaries iterate over keys by default, with `iterkeys()`, `itervalues()`, and `iteritems()` providing explicit control:

```python
>>> m = {'Jan': 1, 'Feb': 2, 'Mar': 3}
>>> for key in m: print key, m[key]
...
Mar 3
Feb 2
Jan 1
```

Files iterate over lines:

```python
for line in file:
    # process each line
    ...
```

The `in` operator now works on dictionaries, making `key in dict` equivalent to `dict.has_key(key)`.

### PEP 255: Simple Generators

Generators provide resumable functions that preserve local state between calls. They're implemented using the new `yield` keyword:

```python
def generate_ints(N):
    for i in range(N):
        yield i
```

Functions containing `yield` become generator functions. In Python 2.2, generators require `from __future__ import generators` at the module top (this requirement was removed in Python 2.3).

**Generator Execution:**

Calling a generator function returns a generator object implementing the iterator protocol:

```python
>>> gen = generate_ints(3)
>>> gen
<generator object at 0x8117f90>
>>> gen.next()
0
>>> gen.next()
1
>>> gen.next()
2
>>> gen.next()
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
  File "<stdin>", line 2, in generate_ints
StopIteration
```

When execution reaches `yield`, the generator outputs the value and suspends, preserving all local variables. The next `next()` call resumes immediately after the `yield`. A `return` statement without a value signals completion.

**Practical Applications:**

Generators excel at complex iteration patterns. This recursive tree traversal elegantly yields nodes in order:

```python
def inorder(t):
    if t:
        for x in inorder(t.left):
            yield x
        yield t.label
        for x in inorder(t.right):
            yield x
```

Generators can solve intricate problems like the N-Queens problem and Knight's Tour, producing solutions lazily as needed. The concept comes from Icon, where generators are central to the language, though Python adopts them more conservatively as an optional feature.

### PEP 237: Unifying Long Integers and Integers

Python 2.2 began erasing the distinction between regular integers (32-bit on most platforms) and long integers (arbitrary size). Previously, mixing the two caused errors—for example, `'abc'[1L:]` raised `TypeError` because slice indices required regular integers.

```python
>>> 1234567890123
1234567890123L
>>> 2 ** 64
18446744073709551616L
```

Python now automatically promotes integers to longs when needed. The 'L' suffix remains supported but became unnecessary (Python 2.4 would warn about it; Python 3.0 would remove it entirely). Operations that previously raised `OverflowError` now return long integers. While `type()` can still distinguish them, this distinction rarely matters in practice.

### PEP 238: Changing the Division Operator

Python 2.2 began addressing division semantics, a controversial change that wouldn't complete until Python 3.0. The terminology:

- **True division**: Mathematical division (3/2 = 1.5)
- **Floor division**: Integer division (3/2 = 1)
- **Classic division**: Current Python 2.2 default behavior (integer operands floor-divide, floating-point operands true-divide)

**Changes introduced:**

The `//` operator always performs floor division regardless of operand types:

```python
>>> 1 // 2
0
>>> 1.0 // 2.0
0.0
```

Including `from __future__ import division` changes `/` to perform true division:

```python
>>> from __future__ import division
>>> 1 / 2
0.5
```

Without the `__future__` import, `/` continues classic division. Classes can define `__truediv__()` and `__floordiv__()` methods to overload both operators.

The `-Qwarn` command-line option warns about integer division, helping identify affected code. Python 2.3 would enable this warning by default; Python 3.0 would make true division the default behavior.

### PEP 227: Nested Scopes

Nested scopes, introduced as optional in Python 2.1, became mandatory in Python 2.2. Previously, Python used only three namespaces: local, module-level, and built-in. This created surprising limitations:

```python
def f():
    ...
    def g(value):
        ...
        return g(value-1) + 1  # NameError!
    ...
```

The inner `g()` couldn't reference its own name. Lambda expressions suffered similarly, forcing awkward workarounds:

```python
def find(self, name):
    # name=name captures the outer variable
    L = filter(lambda x, name=name: x == name,
               self.list_attribute)
    return L
```

With nested scopes, unassigned variables are looked up in enclosing scopes:

```python
def find(self, name):
    # name is naturally available
    L = filter(lambda x: x == name,
               self.list_attribute)
    return L
```

**Restrictions:**

Nested scopes made `from module import *` and `exec` illegal inside functions containing nested functions or lambdas with free variables. These statements introduce unknowable names at compile time, preventing the compiler from determining which variables are local versus free. Code violating this raises `SyntaxError`:

```python
x = 1
def f():
    exec 'x=2'  # Syntax error
    def g():
        return x
```

### Unicode Changes

**Wide Unicode Support:**

Python 2.2 can be compiled with `--enable-unicode=ucs4` to use 32-bit Unicode (UCS-4) instead of the default 16-bit (UCS-2). Wide Python handles Unicode characters from U+000000 to U+110000, expanding `unichr()`'s legal value range. Narrow Python restricts values to 65535. This choice affects `unichr()` behavior and available Unicode character ranges.

**Codec Enhancements:**

8-bit strings gained a `decode([encoding])` method symmetric to Unicode strings' `encode()` method:

```python
>>> s = """Here is a lengthy piece of redundant, overly verbose,
... and repetitive text.
... """
>>> data = s.encode('zlib')
>>> data
'x\x9c\r\xc9\xc1\r\x80 \x10\x04\xc0?Ul...'
>>> data.decode('zlib')
'Here is a lengthy piece of redundant, overly verbose,\nand repetitive text.\n'
```

Codecs were added for non-Unicode tasks like uu-encoding, base64, zlib compression, and ROT-13:

```python
>>> print s.encode('uu')
begin 666 <data>
M2&5R92!I<R!A(&QE;F=T:'D@<&EE8V4@;V8@<F5D=6YD86YT+"!O=F5R;'D@
>=F5R8F]S92P*86YD(')E<&5T:71I=F4@=&5X="X*

end
>>> "sheesh".encode('rot-13')
'furrfu'
```

Classes can define `__unicode__()` methods analogous to `__str__()` for Unicode conversion.

## Standard Library Improvements

### XML-RPC Support

Fredrik Lundh contributed the `xmlrpclib` module for XML-RPC clients. XML-RPC provides simple remote procedure calls over HTTP and XML:

```python
import xmlrpclib
s = xmlrpclib.Server(
    'http://www.oreillynet.com/meerkat/xml-rpc/server.php')
channels = s.meerkat.getChannels()
# channels is a list of dictionaries:
# [{'id': 4, 'title': 'Freshmeat Daily News'},
#  {'id': 190, 'title': '32Bits Online'}, ...]

items = s.meerkat.getItems({'channel': 4})
# items contains article information:
# [{'link': 'http://freshmeat.net/releases/52719/',
#   'description': 'A utility which converts HTML to XSL FO.',
#   'title': 'html2fo 0.3 (Default)'}, ...]
```

The companion `SimpleXMLRPCServer` module enables creating XML-RPC servers.

### HMAC Module

The `hmac` module implements HMAC (Keyed-Hashing for Message Authentication) as described in RFC 2104, useful for secure message authentication.

### Tuple-Like Return Values with Attributes

Functions returning tuples gained mnemonic attributes. Functions like `os.stat()`, `os.fstat()`, `os.statvfs()`, and time functions (`time.localtime()`, `time.gmtime()`, `time.strptime()`) now return pseudo-sequences with named attributes:

```python
# Old style
file_size = os.stat(filename)[stat.ST_SIZE]

# New style
file_size = os.stat(filename).st_size
```

This improves code clarity while maintaining backward compatibility—the objects still behave as tuples.

### Profiler Improvements

The Python profiler underwent extensive reworking to correct output errors and improve accuracy. The revised profiler provides more reliable performance analysis.

### Socket and Network Enhancements

**IPv6 Support:** The `socket` module can be compiled with `--enable-ipv6` to support IPv6 networking.

**SMTP Enhancements:** The `smtplib` module gained RFC 2487 "Secure SMTP over TLS" support, enabling encrypted SMTP traffic and SMTP authentication.

**IMAP Extensions:** The `imaplib` module added support for NAMESPACE (RFC 2342), SORT, GETACL, and SETACL extensions.

### String Module Additions

New constants `ascii_letters`, `ascii_lowercase`, and `ascii_uppercase` were added to the `string` module. These replaced incorrect uses of `string.letters` in locale-aware contexts, where `letters` varies by locale. The new constants consistently represent A-Za-z regardless of locale settings.

### Struct Module 64-bit Support

Two new format characters, `q` and `Q`, support 64-bit integers on platforms with C `long long`:

```python
>>> import struct
>>> struct.pack('q', 1234567890123)
# Packs as signed 64-bit integer
>>> struct.pack('Q', 1234567890123)
# Packs as unsigned 64-bit integer
```

### Interactive Help

The interpreter's interactive mode gained a `help()` function using the `pydoc` module:

```python
>>> help(dict)  # Displays dict documentation
>>> help()      # Enters interactive help utility
```

### Regular Expression Improvements

The SRE engine underlying the `re` module received bugfixes and performance improvements:

- `re.sub()` and `re.split()` rewritten in C for speed
- Unicode character range matching sped up by 2x
- New `finditer()` method returns an iterator over non-overlapping matches

### Email and RFC 822 Improvements

The `rfc822` module's email address parsing became RFC 2822 compliant. A comprehensive new `email` package was added for parsing and generating email messages, including MIME documents.

### Difflib Module

The `difflib` module gained a `Differ` class producing human-readable change deltas between text sequences. Generator functions `ndiff()` and `restore()` compute deltas and recover original sequences.

### Threading Timer

The `threading` module gained a `Timer` class for scheduling future activities:

```python
from threading import Timer

def delayed_task():
    print "Executing delayed task"

t = Timer(5.0, delayed_task)  # Execute after 5 seconds
t.start()
```

### Mimetypes Module

A `MimeTypes` class makes using alternative MIME-type databases easier by accepting a list of filenames to parse.

## Interpreter Changes and Fixes

### C-Level Profiling and Tracing

Profiling and tracing can now be implemented in C via `PyEval_SetProfile()` and `PyEval_SetTrace()`, operating at much higher speeds than Python-based functions. The existing `sys.setprofile()` and `sys.settrace()` now use this C-level interface internally.

### Interpreter State Inspection API

New C functions enable walking interpreter and thread states:

- `PyInterpreterState_Head()` and `PyInterpreterState_Next()` iterate through interpreters
- `PyInterpreterState_ThreadHead()` and `PyThreadState_Next()` iterate through threads

This low-level API assists debugger and development tool implementors.

### Garbage Collection API Changes

The garbage collection C API was redesigned to prevent misuse and debug errors more easily. Extension types must be updated:

1. Rename `Py_TPFLAGS_GC()` to `PyTPFLAGS_HAVE_GC()`
2. Use `PyObject_GC_New()` or `PyObject_GC_NewVar()` to allocate; `PyObject_GC_Del()` to deallocate
3. Rename `PyObject_GC_Init()` to `PyObject_GC_Track()` and `PyObject_GC_Fini()` to `PyObject_GC_UnTrack()`
4. Remove `PyGC_HEAD_SIZE()` from size calculations
5. Remove calls to `PyObject_AS_GC()` and `PyObject_FROM_GC()`

Extensions using the old API still compile but won't participate in garbage collection.

### Argument Parsing Enhancements

**New format sequence:** The `et` format for `PyArg_ParseTuple()` takes an encoding name and parameter, converting Unicode strings to the specified encoding while leaving 8-bit strings alone (assuming they're already encoded correctly). This differs from `es`, which assumes 8-bit strings are ASCII and converts them.

**Simplified function:** `PyArg_UnpackTuple()` provides a simpler, faster alternative to `PyArg_ParseTuple()` when format strings aren't needed—just specify minimum/maximum argument counts and pointers to receive values.

### Method Definition Improvements

**New flags:** `METH_NOARGS` and `METH_O` simplify methods with no arguments or single untyped arguments, calling them more efficiently than `METH_VARARGS`. The old `METH_OLDARGS` style became officially deprecated.

### Safe String Formatting

`PyOS_snprintf()` and `PyOS_vsnprintf()` provide cross-platform implementations of `snprintf()` and `vsnprintf()`, protecting against buffer overruns unlike `sprintf()` and `vsprintf()`.

### Tuple Resize Change

`_PyTuple_Resize()` lost its unused third parameter, now taking only two arguments.

## Other Changes and Fixes

### MacOS X Support

MacOS Python code, maintained by Jack Jansen, moved into the main Python CVS tree. The most significant addition was framework building support via `--enable-framework`, installing a self-contained Python into `/Library/Frameworks/Python.framework`. This laid groundwork for full Python applications, IDE ports, and OSA scripting language integration. MacPython toolbox modules (windowing, QuickTime, scripting APIs) were ported but left commented out in `setup.py` for experimental use.

### Weak References Integration

Weak references, added as an extension in Python 2.1, became part of the core for new-style class implementation. The `ReferenceError` exception consequently moved from the `weakref` module to built-in exceptions.

### Future Statements and Compilation

The `compile()` built-in gained a `flags` argument to properly observe `__future__` statement behavior in simulated shells and development environments (PEP 264).

A new script, `Tools/scripts/cleanfuture.py`, automatically removes obsolete `__future__` statements from source code.

### License Changes

The Python 1.6 license wasn't GPL-compatible. Minor textual changes to the 2.2 license fixed this, making Python embeddable in GPLed programs again. Python itself uses a BSD-equivalent license, not GPL. These license changes also applied to Python 2.0.1 and 2.1.1.

### Platform-Specific Improvements

**Windows:**
- Unicode filenames converted to MBCS for Microsoft file APIs (ASCII proved problematic)
- Large file support enabled
- `.pyw` files now importable (previously only runnable via PYTHONW.EXE)
- `os.kill()` implemented with CTRL_C_EVENT and CTRL_BREAK_EVENT
- Wise Solutions InstallerMaster 8.1 replaced aging Wise 5.0a for installers
- Borland C compilation support added (though not fully functional)

**Unix:**
- Unicode filename handling uses locale character set if `locale.nl_langinfo(CODESET)` available
- `sys.getdlopenflags()` and `sys.setdlopenflags()` control `dlopen()` flags for extension loading

**FreeBSD:**
- `SO_SETFIB` socket constant for alternate routing tables

### Built-in Function Changes

**Keyword argument checking:** Built-in functions now raise `TypeError` when receiving unexpected keyword arguments with the message "function takes no keyword arguments".

**pow() restriction:** `pow(x, y, z)` no longer accepts floating-point numbers since `(x**y) % z` produces unpredictable results for floats.

**xrange() deprecations:** Several `xrange()` features deprecated: slicing, sequence multiplication, the `in` operator, the `tolist()` method, and `start`/`stop`/`step` attributes. At the C level, `PyRange_New()`'s fourth `repeat` argument also deprecated. These rarely-used features were buggy and would be removed in Python 2.3.

### Dictionary Robustness

Numerous patches fixed core dumps when dictionaries contained objects that mutated themselves or the containing dictionary when their hash values were accessed—a challenging bug-hunting exercise between Michael Hudson finding crashes and Tim Peters fixing them.

### Miscellaneous Fixes

Python 2.2 applied 527 patches and fixed 683 bugs compared to Python 2.1. Maintenance releases 2.2.1 (139 patches, 143 bugs) and 2.2.2 (106 patches, 82 bugs) continued improvements—these figures likely underestimate the actual work.

## Key Takeaways

1. **Type/class unification** was Python's most fundamental change since creation, enabling subclassing built-in types and creating new-style classes with advanced features.

2. **Descriptors** provided the foundation for properties, static methods, class methods, and slots, offering unprecedented control over attribute access while remaining invisible to most users.

3. **Generators and iterators** introduced a new paradigm for writing iteration logic, separating sequential access from random indexing and enabling elegant solutions to complex iteration problems.

4. **Backward compatibility** was carefully maintained—classic classes remained available, iterators were automatically created for old-style sequences, and new features were opt-in through inheritance or `__future__` imports.

5. **Integer unification began** with automatic promotion from regular to long integers, eliminating `OverflowError` in many cases and beginning the decade-long path to Python 3's single integer type.

6. **Nested scopes became mandatory**, fixing longstanding issues with lambda expressions and recursive inner functions while restricting `exec` and `from module import *` in nested contexts.

7. **Standard library maturity** advanced with XML-RPC support, improved profiling, enhanced networking protocols, and better error handling across numerous modules.

Python 2.2 represented a pivotal moment in Python's evolution—it modernized the object system without breaking existing code, introduced powerful new programming paradigms, and laid the groundwork for features that would define Python development for the next two decades. The careful balance between revolutionary changes and pragmatic backward compatibility made this release both transformative and practical, setting the template for Python's future evolution.
