# Python 2.6 Release Summary

**Released:** October 1, 2008
**Source:** [Official Python 2.6 Release Notes](https://docs.python.org/2/whatsnew/2.6.html)

## Overview

Python 2.6 represents a pivotal moment in Python's evolution as the first major step toward Python 3.0. Released in synchronized development with Python 3.0, version 2.6 serves as a bridge between the 2.x and 3.x series, incorporating new features and syntax from 3.0 wherever possible while maintaining backward compatibility with existing 2.x code. The overarching philosophy was simple: prepare the migration path to Python 3.0 by backporting compatible features and providing tools to identify code that would break in 3.0.

This release introduces substantial additions to the standard library, most notably the multiprocessing package for process-based parallelism and the json module for JavaScript Object Notation support. Language enhancements focus heavily on Python 3.0 compatibility features, including the with statement becoming standard syntax, class decorators, advanced string formatting via str.format(), and abstract base classes for defining interfaces. New syntax for exception handling, byte literals, and binary/octal integer literals all pave the way for Python 3.0. A new -3 command-line switch enables warnings about features that will be removed or changed in Python 3.0, helping developers identify compatibility issues before migration.

## Major Language Features

### PEP 343: The 'with' Statement

Previously introduced as an optional feature in Python 2.5 requiring a `from __future__ import with_statement` directive, the with statement becomes standard syntax in Python 2.6. The statement clarifies code that previously required try...finally blocks for resource management:

```python
with open('/etc/passwd', 'r') as f:
    for line in f:
        print line
```

The file is automatically closed when the block completes, even if an exception occurs. Objects supporting the context management protocol implement `__enter__()` and `__exit__()` methods. The contextlib module provides utilities for writing context managers, including the contextmanager decorator that lets you write a generator function instead of defining a class:

```python
from contextlib import contextmanager

@contextmanager
def db_transaction(connection):
    cursor = connection.cursor()
    try:
        yield cursor
    except:
        connection.rollback()
        raise
    else:
        connection.commit()

db = DatabaseConnection()
with db_transaction(db) as cursor:
    cursor.execute('insert into ...')
```

### PEP 3101: Advanced String Formatting

Python 2.6 backports the str.format() method from Python 3.0, providing a powerful alternative to the % operator. The method treats strings as templates using curly brackets as placeholders:

```python
>>> "User ID: {0}".format("root")
'User ID: root'
>>> "User ID: {uid}   Last seen: {last_login}".format(
...     uid="root",
...     last_login = "5 Mar 2008 07:20")
'User ID: root   Last seen: 5 Mar 2008 07:20'
```

Field names can reference attributes or dictionary keys:

```python
>>> import sys
>>> print 'Platform: {0.platform}\nPython version: {0.version}'.format(sys)
Platform: darwin
Python version: 2.6a1+ (trunk:61261M, Mar  5 2008, 20:29:41)
```

Format specifiers control alignment and presentation:

```python
>>> fmt = '{0:15} ${1:>6}'
>>> fmt.format('Registration', 35)
'Registration    $    35'
>>> '{0:g}'.format(3.75)
'3.75'
>>> '{0:e}'.format(3.75)
'3.750000e+00'
```

### PEP 3110: Exception-Handling Changes

Python 2.6 introduces the new `except Exception as exc:` syntax alongside the older comma syntax. This eliminates ambiguity in exception handling:

```python
# Old syntax (still works in 2.6)
try:
    ...
except TypeError, exc:
    ...

# New syntax (Python 3.0 compatible)
try:
    ...
except TypeError as exc:
    ...
```

The new syntax makes it unambiguous when catching multiple exceptions:

```python
# Correct way to catch multiple exceptions
try:
    ...
except (TypeError, ValueError):
    ...
```

### PEP 3112: Byte Literals

For forward compatibility with Python 3.0's distinction between text and binary data, Python 2.6 adds `bytes` as a synonym for str and supports the `b''` notation for byte literals:

```python
>>> b'string'
'string'
>>> bytes([65, 66, 67])  # Different behavior than 3.0
'[65, 66, 67]'
```

A new bytearray type provides a mutable sequence of bytes:

```python
>>> bytearray([65, 66, 67])
bytearray(b'ABC')
>>> b = bytearray(u'\u21ef\u3244', 'utf-8')
>>> b
bytearray(b'\xe2\x87\xaf\xe3\x89\x84')
>>> b[0] = '\xe3'
>>> b
bytearray(b'\xe3\x87\xaf\xe3\x89\x84')
```

### PEP 3129: Class Decorators

Decorators, previously limited to functions, now work with classes:

```python
@foo
@bar
class A:
    pass

# Equivalent to:
class A:
    pass
A = foo(bar(A))
```

### PEP 3127: Integer Literal Support and Syntax

Python 2.6 adds support for binary literals with the `0b` prefix and the new `0o` prefix for octal numbers (though the old leading-zero syntax still works):

```python
>>> 0o21, 2*8 + 1
(17, 17)
>>> 0b101111
47
```

A new bin() builtin returns binary representations:

```python
>>> bin(173)
'0b10101101'
>>> oct(42)
'052'
```

## Standard Library Improvements

### PEP 371: The multiprocessing Package

The multiprocessing module enables true parallel execution by creating separate processes instead of threads, circumventing the Global Interpreter Lock (GIL). The API mirrors the threading module, making it familiar to developers:

```python
from multiprocessing import Process, Queue

def factorial(queue, N):
    fact = 1L
    for i in range(1, N+1):
        fact = fact * i
    queue.put(fact)

if __name__ == '__main__':
    queue = Queue()
    p = Process(target=factorial, args=(queue, 5))
    p.start()
    p.join()
    result = queue.get()
    print 'Factorial', 5, '=', result
```

The Pool class provides a higher-level interface for distributing work across multiple processes:

```python
from multiprocessing import Pool

p = Pool(5)
result = p.map(factorial, range(1, 1000, 10))
```

The Manager class creates a server process holding shared data structures accessible via proxy objects:

```python
from multiprocessing import Pool, Manager

mgr = Manager()
d = mgr.dict()  # Create shared dictionary

p = Pool(5)
for N in range(1, 1000, 10):
    p.apply_async(factorial, (N, d))

p.close()
p.join()

for k, v in sorted(d.items()):
    print k, v
```

### PEP 3116: New I/O Library

The io module provides a layered I/O library that separates buffering and text handling from raw I/O operations. Three levels of abstract base classes define the hierarchy:

- **RawIOBase** defines raw I/O operations: read(), write(), seek(), tell(), truncate(), and close()
- **BufferedIOBase** buffers data in memory to reduce system calls, with concrete implementations like BufferedWriter, BufferedReader, and BufferedRandom
- **TextIOBase** handles text I/O with Unicode support and universal newlines, including TextIOWrapper and StringIO implementations

In Python 2.6, the module primarily serves to ease forward compatibility with 3.0, as the underlying file implementations haven't been restructured to use the io module.

### PEP 3119: Abstract Base Classes

The abc module introduces Abstract Base Classes, Python's equivalent to Java interfaces. ABCs define protocols that classes can implement explicitly or register to support:

```python
import collections

class Storage(collections.MutableMapping):
    # Must implement __getitem__, __setitem__, __delitem__,
    # __iter__, and __len__
    ...
```

You can also register existing classes without inheritance:

```python
class Storage:
    ...

collections.MutableMapping.register(Storage)
```

Creating custom ABCs uses the ABCMeta metaclass and abstractmethod decorator:

```python
from abc import ABCMeta, abstractmethod

class Drawable():
    __metaclass__ = ABCMeta

    @abstractmethod
    def draw(self, x, y, scale=1.0):
        pass

    def draw_doubled(self, x, y):
        self.draw(x, y, scale=2.0)
```

The collections module provides ABCs for common protocols: Iterable, Container, MutableMapping, Sequence, and others.

### PEP 3141: A Type Hierarchy for Numbers

The numbers module backports Python 3.0's numeric tower defining abstract base classes for different numeric types:

- **Number**: The most general ABC, defines no operations
- **Complex**: Supports addition, subtraction, multiplication, division, exponentiation
- **Real**: Adds floor(), trunc(), rounding, and comparisons
- **Rational**: Includes numerator and denominator properties
- **Integral**: Supports bitwise operations and use as array indexes

The fractions module provides a Fraction class for rational numbers:

```python
>>> from fractions import Fraction
>>> a = Fraction(2, 3)
>>> b = Fraction(2, 5)
>>> a + b
Fraction(16, 15)
>>> a / b
Fraction(5, 3)
```

Floats gained an as_integer_ratio() method for conversion:

```python
>>> (2.5).as_integer_ratio()
(5, 2)
>>> (3.1415).as_integer_ratio()
(7074029114692207L, 2251799813685248L)
```

### The json Module

The new json module supports encoding and decoding JavaScript Object Notation:

```python
>>> import json
>>> data = {"spam": "foo", "parrot": 42}
>>> in_json = json.dumps(data)
>>> in_json
'{"parrot": 42, "spam": "foo"}'
>>> json.loads(in_json)
{"spam": "foo", "parrot": 42}
```

The module handles most built-in Python types and supports custom encoders/decoders for additional types.

### The ast Module

The ast module provides Abstract Syntax Tree representation of Python code with helper functions for analysis:

```python
import ast

t = ast.parse("""
d = {}
for i in 'abcdefghijklm':
    d[i + i] = ord(i) - ord('a') + 1
print d
""")
print ast.dump(t)
```

The literal_eval() function safely evaluates literal expressions without the security risks of eval():

```python
>>> literal = '("a", "b", {2:4, 3:8, 1:2})'
>>> print ast.literal_eval(literal)
('a', 'b', {1: 2, 2: 4, 3: 8})
>>> print ast.literal_eval('"a" + "b"')
ValueError: malformed string
```

### Other Notable Module Updates

**collections module:**
- namedtuple() factory creates tuple subclasses with named fields accessible by name or index
- deque now supports optional maxlen parameter to create bounded queues

**itertools module:**
- izip_longest() for zipping iterables of different lengths with fill values
- product() returns Cartesian products of iterables
- combinations() generates combinations of elements
- permutations() generates all permutations
- chain.from_iterable() chains iterables from a single iterable

**math module:**
- isinf() and isnan() test for infinity and NaN values
- copysign() copies sign bits
- factorial() computes factorials
- fsum() performs accurate floating-point summation
- acosh(), asinh(), atanh() compute inverse hyperbolic functions
- log1p() returns log(1+x) with better precision for small x
- trunc() truncates toward zero

**Improved SSL support:**
- New ssl module built atop OpenSSL provides more control over protocol negotiation and X.509 certificates
- Better support for writing SSL servers in Python

## Performance Improvements

Python 2.6 includes several performance optimizations:

- **warnings module rewritten in C** enables warnings from the parser and potentially faster interpreter startup
- **Type method cache** reduces method lookup overhead by caching method resolutions, avoiding repeated base class traversals
- **Keyword argument optimization** uses quick pointer comparison for keyword arguments, typically saving full string comparisons
- **struct module rewritten in C** substantially improves performance of binary data operations
- **Unicode string optimizations** make split() 25% faster and splitlines() 35% faster
- **with statement optimization** stores `__exit__` method on stack, speeding up context manager usage
- **Garbage collection improvement** clears internal free lists when garbage-collecting highest generation, potentially returning memory to OS sooner

## Migration Features

Python 2.6's primary purpose is facilitating migration to Python 3.0:

**Backported syntax:**
- with statement as standard syntax
- except Exception as exc syntax
- Binary (0b) and new octal (0o) literals
- Class decorators
- Advanced string formatting with str.format()

**Backported standard library:**
- Abstract base classes (abc module)
- I/O library (io module)
- Numbers type hierarchy (numbers module)
- Byte literals and bytearray type

**Migration tools:**
- `-3` command-line switch enables Python 3.0 compatibility warnings
- `sys.py3kwarning` boolean indicates if warnings are enabled
- future_builtins module provides Python 3.0 versions of modified builtins (hex, oct, map, filter)
- 2to3 conversion tool assists with automated migration

**Print as function:**
- `from __future__ import print_function` converts print to a function, matching Python 3.0

## Other Changes

### PEP 366: Explicit Relative Imports From a Main Module

Modules now have a `__package__` attribute that fixes relative imports when running modules with the -m switch.

### PEP 370: Per-user site-packages Directory

Python 2.6 introduces user-specific site directories for installing packages without system-wide access:
- Unix/Mac OS X: `~/.local/`
- Windows: `%APPDATA%/Python`

### Language Changes

Several smaller enhancements to the core language:

- Directories and zip archives with `__main__.py` can be executed directly
- hasattr() no longer catches KeyboardInterrupt and SystemExit
- Function calls accept any mapping for `**kwargs`, not just dictionaries
- Keyword arguments can follow `*args` in function calls
- New next() builtin for retrieving iterator values with optional default
- Tuples gained index() and count() methods
- Properties support getter, setter, and deleter decorators for convenient addition
- Set methods accept multiple iterables: intersection(), union(), difference(), etc.
- Float conversions handle NaN, positive/negative infinity on IEEE 754 platforms
- Floats have hex() method for hexadecimal representation without rounding errors
- Complex constructor preserves signed zeros on supporting systems
- Classes can set `__hash__ = None` to indicate they're not hashable
- GeneratorExit now inherits from BaseException instead of Exception
- Instance methods have `__self__` and `__func__` as synonyms for `im_self` and `im_func`

### Development Process Changes

- Migrated from SourceForge to Roundup issue tracker at bugs.python.org
- Documentation converted from LaTeX to reStructuredText using Sphinx
- New -B switch and PYTHONDONTWRITEBYTECODE environment variable prevent .pyc/.pyo file creation
- PYTHONIOENCODING environment variable controls encoding for stdin/stdout/stderr

## Key Takeaways

1. **Bridge to Python 3.0** designed specifically to ease migration by backporting compatible features and providing compatibility warnings

2. **Parallel processing capabilities** through the multiprocessing package enable true parallelism by circumventing the GIL

3. **Modern data interchange** with the json module for JavaScript Object Notation encoding/decoding

4. **Formal interface definitions** through Abstract Base Classes provide Pythonic equivalent to Java interfaces

5. **Enhanced numeric computing** with the numbers type hierarchy, fractions module, and comprehensive math module improvements

6. **Forward-compatible syntax** including class decorators, exception handling with 'as', binary/octal literals, and advanced string formatting

7. **Development infrastructure modernization** including new issue tracker, Sphinx documentation system, and improved build tools

Python 2.6 successfully established the migration path to Python 3.0 while maintaining full backward compatibility with Python 2.5 code. By introducing Python 3.0 features incrementally and providing tools to identify incompatibilities, it allowed developers to begin preparing their codebases for the future while continuing to run on the stable 2.x series.
