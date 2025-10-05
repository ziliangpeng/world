# Python 2.5 Release Summary

**Released:** September 19, 2006
**Source:** [Official Python 2.5 Release Notes](https://docs.python.org/2/whatsnew/2.5.html)

## Overview

Python 2.5 represents a significant milestone in Python's evolution, introducing powerful language features that fundamentally expanded the language's expressive capabilities. While the language changes may seem modest in number, they have profound implications for how Python code is written. The release is perhaps most notable for introducing the `with` statement for resource management and conditional expressions, both long-requested features that addressed real pain points in Python programming.

The standard library enhancements in Python 2.5 are substantial and transformative. Three major additions - ElementTree for XML processing, SQLite for embedded databases, and ctypes for calling C functions - brought enterprise-grade capabilities into the standard library without requiring external dependencies. These additions, combined with new modules like hashlib, uuid, and contextlib, significantly expanded Python's out-of-the-box functionality. The release also introduced enhanced generators that can receive values and function as coroutines, absolute and relative imports to clarify module dependencies, and unified try/except/finally blocks for cleaner error handling. Under the hood, Python 2.5 underwent major architectural changes including the new Abstract Syntax Tree (AST) compiler and memory management improvements.

## Major Language Features

### PEP 308: Conditional Expressions

Python 2.5 finally introduced conditional expressions (ternary operators) after years of community discussion. The syntax places the condition in the middle rather than at the beginning:

```python
x = true_value if condition else false_value
```

The unusual syntax was deliberately chosen to emphasize the common case. When tested against real code, most uses had a clear "normal" value and a rare "exceptional" value:

```python
contents = ((doc + '\n') if doc else '')
```

Evaluation is lazy - only the selected branch is evaluated. While parentheses aren't required, they're recommended for clarity:

```python
# Clearer with parentheses
level = (1 if logging else 0)
```

This syntax reads naturally when there's a common case that occasionally needs a fallback value, making the code's intent more obvious to readers.

### PEP 343: The 'with' Statement

The `with` statement provides a clean way to ensure cleanup code always executes, replacing many uses of try/finally blocks:

```python
from __future__ import with_statement  # Required in 2.5

with open('/etc/passwd', 'r') as f:
    for line in f:
        print line
```

The file is automatically closed when the block exits, even if an exception occurs. Any object with `__enter__()` and `__exit__()` methods supports the context management protocol. Standard library objects like file objects, threading locks, and decimal contexts now support `with`:

```python
# Locks are automatically released
lock = threading.Lock()
with lock:
    # Critical section
    ...

# Decimal precision temporarily changed
from decimal import Decimal, Context, localcontext

with localcontext(Context(prec=16)):
    # All code here uses 16-digit precision
    print Decimal('578').sqrt()
```

The `contextlib` module provides helpers for creating context managers. The `@contextmanager` decorator lets you write a generator function instead of defining a class:

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

with db_transaction(db) as cursor:
    cursor.execute('insert into ...')
```

The code before `yield` runs in `__enter__()`, the yielded value is returned to the `with` statement, and code after `yield` runs in `__exit__()`.

### PEP 342: New Generator Features

Generators evolved from one-way producers to bidirectional coroutines. In Python 2.5, `yield` became an expression that can receive values:

```python
def counter(maximum):
    i = 0
    while i < maximum:
        val = (yield i)
        # If value provided, change counter
        if val is not None:
            i = val
        else:
            i += 1
```

Values are sent into generators using `send(value)`:

```python
>>> it = counter(10)
>>> print it.next()
0
>>> print it.next()
1
>>> print it.send(8)  # Jump to 8
8
>>> print it.next()
9
```

Two additional methods were added:

- `throw(type, value, traceback)` raises an exception inside the generator at the yield point
- `close()` raises GeneratorExit to terminate the generator

These changes enable generators to function as coroutines - functions that can be entered, exited, and resumed at multiple points. The `close()` method means generators can now guarantee cleanup with try/finally blocks, which was previously restricted.

### PEP 341: Unified try/except/finally

Prior to Python 2.5, you couldn't combine except blocks with a finally block. Now you can write complete error handling in one structure:

```python
try:
    # code that might raise exceptions
    block_1()
except Exception1:
    handler_1()
except Exception2:
    handler_2()
else:
    # runs if no exception raised
    else_block()
finally:
    # always runs
    final_block()
```

The finally block executes no matter what happens - even if an exception handler raises a new exception. This simplified error handling code that previously required nested try statements.

### PEP 328: Absolute and Relative Imports

Python 2.5 introduced explicit relative imports to resolve ambiguity in package imports. Previously, `import string` in a package would import the package's `string.py` before the standard library's string module. With absolute imports (enabled via `from __future__ import absolute_import`):

```python
# Absolute import - finds standard library
import string

# Explicit relative imports using dot notation
from .string import name1, name2  # Import from pkg.string
from . import string              # Import pkg.string
from .. import E                  # Import from parent package
from ..F import G                 # Import from sibling package
```

Leading dots indicate relative imports from the current package. This clarifies whether code intends to import from the standard library or from within the package, making large codebases more maintainable.

## Standard Library Improvements

### The ctypes Package

The ctypes package, written by Thomas Heller, enables calling C functions in shared libraries without writing C extension modules:

```python
import ctypes

libc = ctypes.CDLL('libc.so.6')
result = libc.printf("Line of output\n")
```

Type constructors let you work with C data types:

```python
# C type constructors
i = ctypes.c_int(42)
f = ctypes.c_float(3.14)
s = ctypes.c_char_p("hello")

# Modifiable string buffer
buf = ctypes.create_string_buffer("this is a string")
libc.strfry(buf)

# Set return type for functions
libc.atof.restype = ctypes.c_double
result = libc.atof('2.71828')  # Returns 2.71828, not a random integer
```

The package also provides access to Python's C API as `ctypes.pythonapi`, though this requires careful use of `py_object()` to avoid segmentation faults. Now that ctypes is in the standard library, developers can write Python wrappers for C libraries without creating compiled extension modules.

### The ElementTree Package

A subset of Fredrik Lundh's ElementTree library for XML processing was added as `xml.etree`, including the fast C implementation cElementTree:

```python
from xml.etree import ElementTree as ET

# Parse XML from file or URL
tree = ET.parse('document.xml')
root = tree.getroot()

# Create XML from string literal
svg = ET.XML("""<svg width="10px" version="1.0">
             </svg>""")
svg.set('height', '320px')
```

Elements support dictionary-like operations for attributes and list-like operations for child nodes:

```python
# Dictionary operations for attributes
elem.get('name')          # Get attribute value
elem.set('name', 'value') # Set attribute
elem.keys()               # List attribute names

# List operations for children
elem[0]                   # First child
elem.append(child)        # Add child
len(elem)                 # Count children
for child in elem:        # Iterate children
    ...
```

ElementTree writes XML output efficiently:

```python
# Write with UTF-8 encoding
f = open('output.xml', 'w')
tree.write(f, encoding='utf-8')
```

This provides a modern, Pythonic alternative to the DOM for XML processing.

### The sqlite3 Package

The pysqlite wrapper for the SQLite embedded database was added as `sqlite3`, providing a lightweight disk-based database with SQL support:

```python
import sqlite3

# Connect to database file (or use ':memory:' for RAM)
conn = sqlite3.connect('/tmp/example')
c = conn.cursor()

# Execute SQL commands
c.execute('''create table stocks
(date text, trans text, symbol text, qty real, price real)''')

c.execute("insert into stocks values ('2006-01-05','BUY','RHAT',100,35.14)")
conn.commit()
```

The module follows the DB-API 2.0 specification. Always use parameter substitution to avoid SQL injection:

```python
# Secure parameter substitution
symbol = 'IBM'
c.execute('select * from stocks where symbol=?', (symbol,))

# Insert multiple records
for t in [('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
          ('2006-04-05', 'BUY', 'MSOFT', 1000, 72.00)]:
    c.execute('insert into stocks values (?,?,?,?,?)', t)
```

SQLite is perfect for applications that need structured storage without running a separate database server, or for prototyping before migrating to PostgreSQL or Oracle.

### The hashlib Package

The hashlib module replaced the md5 and sha modules, adding support for SHA-224, SHA-256, SHA-384, and SHA-512:

```python
import hashlib

# Create hash objects
h = hashlib.md5()
h = hashlib.sha1()
h = hashlib.sha256()
h = hashlib.sha512()

# Alternative form
h = hashlib.new('md5')

# Hash data
h.update('hello world')
digest = h.hexdigest()
```

When available, hashlib uses OpenSSL for optimized implementations. The interface is nearly identical to the old md5/sha modules except for constructor naming.

### collections Module Enhancements

The `collections` module gained `defaultdict`, which automatically creates missing keys:

```python
from collections import defaultdict

# Index words by initial letter
words = "Nel mezzo del cammin di nostra vita".lower().split()
index = defaultdict(list)

for w in words:
    init_letter = w[0]
    index[init_letter].append(w)  # No KeyError if key missing

print index
# defaultdict(<type 'list'>, {'c': ['cammin'], 'd': ['del', 'di'], ...})
```

The factory function (here `list`) is called with no arguments to create the default value. You can use any callable: `int` for counters, `set` for collections, or custom functions.

The `deque` type gained a `remove(value)` method to remove the first occurrence of a value.

### uuid Module

The new uuid module generates universally unique identifiers according to RFC 4122:

```python
import uuid

# UUID based on host ID and current time
uuid.uuid1()
# UUID('a8098c1a-f86e-11da-bd1a-00112444be1e')

# UUID using MD5 hash of namespace and name
uuid.uuid3(uuid.NAMESPACE_DNS, 'python.org')
# UUID('6fa459ea-ee8a-3ca4-894e-db77e160355e')

# Random UUID
uuid.uuid4()
# UUID('16fd2706-8baf-433b-82eb-8c7fada847da')

# UUID using SHA-1 hash of namespace and name
uuid.uuid5(uuid.NAMESPACE_DNS, 'python.org')
# UUID('886313e1-3b8a-5372-9b90-0c9aee199e5d')
```

This standardizes UUID generation for distributed systems, databases, and other applications requiring unique identifiers.

### functools Module

The new functools module provides tools for functional programming. The `partial()` function creates partially applied functions:

```python
import functools

def log(message, subsystem):
    print '%s: %s' % (subsystem, message)

server_log = functools.partial(log, subsystem='server')
server_log('Unable to open socket')  # Prints: server: Unable to open socket
```

The `update_wrapper()` function and `@wraps` decorator help create decorators that preserve function metadata:

```python
def my_decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        print 'Calling decorated function'
        return f(*args, **kwds)
    return wrapper
```

### Other Notable Module Updates

**operator module:** `itemgetter()` and `attrgetter()` now support multiple fields:

```python
# Sort by multiple attributes
import operator
data.sort(key=operator.attrgetter('last_name', 'first_name'))
```

**struct module:** Faster due to compiled `Struct` objects:

```python
import struct
s = struct.Struct('ih3s')
data = s.pack(1972, 187, 'abc')
year, number, name = s.unpack(data)
```

**mailbox module:** Complete rewrite supporting modification of mbox, MH, and Maildir formats with `add()`, `remove()`, and `lock()`/`unlock()` methods.

**Queue module:** New `join()` and `task_done()` methods enable coordinating producer/consumer threads.

**datetime module:** New `strptime()` method for parsing date strings.

**tarfile module:** New `extractall()` method and autodetection of compression format.

**zipfile module:** Support for ZIP64 format allowing archives and files larger than 4 GiB.

## Performance Improvements

### Memory Management

Evan Jones's patch improved Python's small object allocator (obmalloc). Python 2.4 allocated objects in 256K arenas but never freed them. Python 2.5 frees empty arenas, returning memory to the operating system when many objects are deleted. This significantly reduces memory usage in long-running applications.

The change requires extension modules to be more careful about mixing memory allocation families (`PyMem_*` vs `PyObject_*`). Previously all allocation functions used the platform's malloc/free, but now mismatches can cause segmentation faults.

### Struct Module

The struct module was optimized by compiling format strings into Struct objects with cached pack/unpack methods, similar to how the re module compiles regular expressions. This provides substantial performance improvements when packing/unpacking many values with the same format.

## Other Changes

### Language Changes

**String methods:** New `partition(sep)` and `rpartition(sep)` methods simplify splitting strings:

```python
>>> ('http://www.python.org').partition('://')
('http', '://', 'www.python.org')
>>> 'www.python.org'.rpartition('.')
('www.python', '.', 'org')
```

**String comparisons:** `startswith()` and `endswith()` accept tuples:

```python
def is_image_file(filename):
    return filename.endswith(('.gif', '.jpg', '.tiff'))
```

**min()/max() functions:** Added `key` parameter like `sort()`:

```python
# Find longest string
longest = max(strings, key=len)
```

**dict type:** New `__missing__(key)` hook called when key not found, enabling custom default values in dict subclasses.

### PEP 352: Exception Hierarchy Changes

All exceptions are now new-style classes. The hierarchy was reorganized with a new BaseException base class:

```python
BaseException
|- KeyboardInterrupt
|- SystemExit
|- Exception
   |- (all other built-in exceptions)
```

This allows `except Exception:` to catch program errors while letting KeyboardInterrupt and SystemExit propagate. Raising strings as exceptions is deprecated.

### PEP 353: ssize_t for 64-bit Platforms

The C API now uses `Py_ssize_t` instead of `int` for sizes and indices, allowing Python to handle more than 2^31-1 items on 64-bit platforms. This primarily affects C extension authors who must update their code to use `Py_ssize_t` and the new `n` format code in `PyArg_ParseTuple()`.

### PEP 357: The __index__ Method

The `__index__()` special method signals that an object can be used as a slice index. NumPy's specialized integer types use this to work with slice notation:

```python
class C:
    def __index__(self):
        return self.value

# Can now be used in slicing
sequence[C()]
```

### Build and C API Changes

**Abstract Syntax Tree compiler:** The bytecode compiler was completely redesigned to use an AST intermediate representation instead of directly traversing the parse tree. This makes the compiler more maintainable and enables tools to work with Python's AST.

**Subversion migration:** Python's source code moved from CVS to Subversion (PEP 347), supervised by Martin von Löwis.

**Coverity analysis:** Coverity's source code analysis tool found approximately 60 bugs, many refcounting errors in error-handling code, which were promptly fixed.

**Set API:** Built-in set types now have an official C API with functions like `PySet_New()`, `PySet_Add()`, `PySet_Contains()`.

**C++ compatibility:** CPython can now be compiled with C++ compilers without errors.

## Key Takeaways

1. **Resource management transformed** with the `with` statement providing clean, guaranteed cleanup for files, locks, and database transactions

2. **Major standard library additions** bringing ctypes (C function calling), ElementTree (XML processing), and SQLite (embedded database) into core Python

3. **Generator evolution** from simple iterators to bidirectional coroutines capable of receiving values and supporting sophisticated control flow

4. **Import system clarification** with absolute and relative imports eliminating ambiguity in package-local vs standard library imports

5. **Conditional expressions** finally added after years of debate, using the distinctive `true_value if condition else false_value` syntax

6. **Memory management improvements** with arena freeing returning memory to the OS and better small object allocation

7. **Compiler architecture overhaul** with the new AST-based compiler replacing direct parse tree traversal, enabling better tooling and optimization

Python 2.5 marked a significant maturation of Python's language design and standard library. The with statement and enhanced generators provided powerful new control flow mechanisms, while the three major library additions (ctypes, ElementTree, SQLite) brought capabilities that previously required third-party packages. The release laid important groundwork for Python 3.0 while maintaining backward compatibility, and introduced architectural improvements like the AST compiler that continue to benefit Python today.
