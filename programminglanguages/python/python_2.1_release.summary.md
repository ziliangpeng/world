# Python 2.1 Release Summary

**Released:** April 17, 2001
**Source:** [Official Python 2.1 Release Notes](https://docs.python.org/2/whatsnew/2.1.html)

## Overview

Python 2.1 marks a significant milestone in Python's evolution, introducing static nested scopes that fundamentally improved the language's treatment of closures and functional programming patterns. Released just six months after Python 2.0, this version demonstrated the development team's commitment to accelerating the release cycle to 6-9 months per major version. The release is notable for being the first to be guided by Python Enhancement Proposals (PEPs), establishing a formal process for documenting and discussing major language changes.

The most transformative change in Python 2.1 is the addition of nested scopes, which resolved long-standing frustrations with lambda expressions and inner functions by allowing them to access variables from enclosing scopes. To manage compatibility concerns, Python 2.1 introduced the `__future__` directive mechanism, allowing new features to be optional in one release before becoming mandatory in the next. Beyond scoping improvements, the release added rich comparisons for user-defined types, a warning framework for managing deprecations, weak references for cache implementations, function attributes for metadata storage, and numerous standard library enhancements including the pydoc documentation system and unittest framework.

## Major Language Features

### PEP 227: Nested Scopes

The largest and most significant change in Python 2.1 is the introduction of static nested scopes, fundamentally altering how Python resolves variable names. Prior to Python 2.1, Python used only three namespaces for variable lookup: local, module-level, and built-in. This created surprising behavior where inner functions couldn't access variables from enclosing function scopes.

**The problem with Python 2.0 scoping:**

```python
def f():
    ...
    def g(value):
        ...
        return g(value-1) + 1  # NameError: g is not defined
    ...
```

The nested function `g()` cannot reference itself recursively because `g` isn't in its local namespace or the module-level namespace. This limitation severely impacted functional programming patterns, particularly with lambda expressions:

```python
# Python 2.0 workaround - awkward default argument pattern
def find(self, name):
    "Return list of any entries equal to 'name'"
    L = filter(lambda x, name=name: x == name,
               self.list_attribute)
    return L
```

**Python 2.1 solution with nested scopes:**

```python
# Python 2.1 - clean and natural
def find(self, name):
    "Return list of any entries equal to 'name'"
    L = filter(lambda x: x == name, self.list_attribute)
    return L
```

With static scoping, when a variable name isn't assigned within a function, Python searches the local namespace of enclosing scopes before checking module-level and built-in namespaces. This makes closures work intuitively and eliminates the need for the awkward `name=name` default argument pattern.

**Restrictions introduced by nested scopes:**

The implementation of nested scopes required making `from module import *` and `exec` statements illegal inside functions containing nested function definitions or lambda expressions with free variables:

```python
x = 1
def f():
    exec 'x=2'  # SyntaxError in Python 2.1
    def g():
        return x
```

This restriction exists because the compiler must determine at compile time which variables come from enclosing scopes, but `exec` and `import *` add names to the local namespace that are unknowable until runtime.

**Gradual introduction:**

Due to compatibility concerns, nested scopes aren't enabled by default in Python 2.1. They must be explicitly enabled using a future statement (see PEP 236 below). In Python 2.2, nested scopes became the default with no way to disable them.

### PEP 236: __future__ Directives

The widespread concern about breaking existing code with nested scopes led to the creation of a new mechanism for gradually introducing incompatible changes. The `__future__` directive allows optional functionality in release N that will become mandatory in release N+1.

**Enabling nested scopes in Python 2.1:**

```python
from __future__ import nested_scopes
```

While it resembles a normal import statement, `__future__` imports have strict placement rules:
- Must appear at the top of the module
- Must precede any Python code or regular import statements
- Can only import features from the special `__future__` module

These restrictions exist because future statements affect how the Python bytecode compiler parses and generates code, so they must be processed before any bytecode generation occurs.

This mechanism provided a template for managing future language changes, giving users an entire release cycle to test and adapt their code before changes become mandatory.

### PEP 207: Rich Comparisons

Python 2.1 revolutionized comparison operations for user-defined classes and extension types. Previously, classes could only implement `__cmp__()`, which returned -1, 0, or +1 and couldn't raise exceptions or return non-Boolean values. This limitation frustrated users of Numeric Python, who needed to perform element-wise matrix comparisons returning result matrices.

**Rich comparison methods:**

Python classes can now individually overload each comparison operator:

| Operation | Method Name |
|-----------|-------------|
| `<`       | `__lt__()`  |
| `<=`      | `__le__()`  |
| `>`       | `__gt__()`  |
| `>=`      | `__ge__()`  |
| `==`      | `__eq__()`  |
| `!=`      | `__ne__()`  |

Each method has the signature `method(self, other)` and can return any Python object or raise exceptions:

```python
class Matrix:
    def __lt__(self, other):
        if self.shape != other.shape:
            raise ValueError("Matrices must have same dimensions")
        # Return a matrix of boolean comparison results
        return Matrix([[self.data[i][j] < other.data[i][j]
                       for j in range(self.cols)]
                       for i in range(self.rows)])
```

**Enhanced cmp() function:**

The built-in `cmp(A, B)` function gained an optional third argument specifying which comparison operation to use:

```python
cmp(A, B, "<")   # Equivalent to A < B
cmp(A, B, ">=")  # Equivalent to A >= B
```

Without the third argument, `cmp()` maintains backward compatibility by returning only -1, 0, or +1.

### PEP 230: Warning Framework

Python 2.1 introduced a systematic framework for managing deprecations and obsolete features. The new `warnings` module enables issuing warnings about deprecated features while giving users time to update their code before features are removed.

**Example of deprecation warning:**

```python
>>> import regex
__main__:1: DeprecationWarning: the regex module
         is deprecated; please use the re module
>>>
```

**Issuing warnings programmatically:**

```python
import warnings
warnings.warn("feature X no longer supported")
```

**Filtering warnings:**

The framework allows fine-grained control over which warnings to display:

```python
import warnings
warnings.filterwarnings(action='ignore',
                       message='.*regex module is deprecated',
                       category=DeprecationWarning,
                       module='__main__')
```

Warning filters support:
- Regular expression matching on message text
- Filtering by warning category
- Module-specific filtering
- Multiple actions: ignore, display once, display always, or raise as exceptions

This framework provides third-party modules with a standard mechanism to deprecate features while maintaining backward compatibility during transition periods.

## Standard Library Improvements

### PEP 205: Weak References

The new `weakref` module introduces weak references, enabling cache implementations and data structures that don't keep objects alive indefinitely. Normal references prevent objects from being garbage collected; weak references allow objects to be deallocated when only weak references remain.

**Problem: Cache keeping objects alive forever:**

```python
_cache = {}
def memoize(x):
    if _cache.has_key(x):
        return _cache[x]

    retval = f(x)
    _cache[x] = retval  # Object kept alive forever
    return retval
```

**Solution: Weak reference cache:**

```python
import weakref

_cache = {}
def memoize(x):
    if _cache.has_key(x):
        obj = _cache[x]()  # Call weak reference
        if obj is not None:
            return obj

    retval = f(x)
    _cache[x] = weakref.ref(retval)  # Store weak reference
    return retval
```

**Creating and using weak references:**

```python
import weakref

obj = SomeClass()
wr = weakref.ref(obj)

# Access referenced object by calling weak reference
referenced_obj = wr()
if referenced_obj is not None:
    # Object still exists
    referenced_obj.method()
else:
    # Object has been deallocated
    pass
```

**Proxy objects:**

Weak proxies provide transparent access to objects without explicit dereferencing:

```python
proxy = weakref.proxy(obj)
proxy.attr     # Equivalent to obj.attr
proxy.meth()   # Equivalent to obj.meth()

del obj
proxy.attr     # Raises weakref.ReferenceError
```

Proxies forward all operations transparently as long as the object exists, raising `ReferenceError` when the object is deallocated.

### PEP 232: Function Attributes

Functions can now have arbitrary attributes attached, eliminating the overloading of docstrings for metadata storage. Previously, frameworks like Zope used docstrings to mark functions as publicly accessible, preventing proper documentation.

**Setting function attributes:**

```python
def f():
    pass

f.publish = 1
f.secure = 1
f.grammar = "A ::= B (C D)*"
```

**Accessing the attribute dictionary:**

```python
# Read attributes
attrs = f.__dict__

# Replace the entire dictionary
f.__dict__ = {'new_attr': 'value'}
```

Unlike class instance `__dict__` attributes, function `__dict__` can be assigned a new dictionary, though it must be a standard Python dictionary, not a custom mapping type.

This feature enables cleaner separation between documentation and metadata, allowing frameworks to store configuration without polluting docstrings.

### Documentation and Testing Tools

**pydoc module:**

Ka-Ping Yee contributed the `pydoc` module, which converts docstrings to HTML or text interactively. The accompanying `pydoc` command-line tool displays documentation for any Python module, package, or class:

```bash
$ pydoc xml.dom
```

Output:
```
Python Library Documentation: package xml.dom in xml

NAME
    xml.dom - W3C Document Object Model implementation for Python.

FILE
    /usr/local/lib/python2.1/xml/dom/__init__.pyc

DESCRIPTION
    The Python mapping of the Document Object Model is documented in
    the Python Library Reference in the section on the xml.dom package.
```

The `pydoc` tool also includes a Tk-based interactive help browser for visual exploration of documentation.

**inspect module:**

Also contributed by Ka-Ping Yee, the `inspect` module provides functions for getting information about live Python code, including introspecting functions, classes, stack frames, and source code.

**unittest framework (PyUnit):**

Steve Purcell contributed PyUnit, a unit testing framework inspired by JUnit. It provides:
- Test case classes and assertion methods
- Test suites for organizing tests
- Test runners for executing tests
- Test discovery mechanisms

**doctest module:**

Tim Peters contributed `doctest`, which extracts examples from docstrings, executes them, and verifies the output matches expected results:

```python
def factorial(n):
    """Return the factorial of n.

    >>> factorial(5)
    120
    >>> factorial(0)
    1
    """
    if n == 0:
        return 1
    return n * factorial(n-1)
```

**difflib module:**

The new `difflib` module contains `SequenceMatcher`, which compares sequences and computes transformations. The sample script `Tools/scripts/ndiff.py` demonstrates building a diff-like tool.

### Other Module Enhancements

**time module improvements:**

Functions like `asctime()` and `localtime()` now make their time argument optional, defaulting to the current time:

```python
# Python 2.1 - concise
log_entry = time.asctime()

# Python 2.0 - verbose
log_entry = time.asctime(time.localtime(time.time()))
```

**ftplib passive mode:**

The `ftplib` module now defaults to passive mode for retrieving files, making it work correctly behind firewalls by default. Call `set_pasv(0)` to disable passive mode if needed.

**socket raw access:**

Grant Edwards contributed support for raw socket access, enabling low-level network programming.

**pstats interactive browser:**

Eric S. Raymond contributed an interactive statistics browser to the `pstats` module for displaying timing profiles, invoked when running the module as a script.

**sys._getframe():**

A new implementation-dependent function returns frame objects from the call stack:

```python
sys._getframe()     # Returns top frame
sys._getframe(1)    # Returns caller's frame
sys._getframe(2)    # Returns caller's caller's frame
```

This function is only available in CPython and intended for debugging, not production code.

**File I/O improvements:**

The `readline()` method was rewritten for dramatically improved performance (around 66% faster, platform-dependent). Jeff Epler contributed the `xreadlines()` method for memory-efficient line iteration:

```python
for line in sys.stdin.xreadlines():
    # Process line without loading entire file
    process(line)
```

**Dictionary popitem():**

A new `popitem()` method enables destructively iterating through dictionaries:

```python
while D:
    key, value = D.popitem()
    # Process key-value pair
    process(key, value)
```

This is faster for large dictionaries because it avoids constructing lists of keys or values.

**Module import control:**

Modules can now control `from module import *` behavior by defining `__all__`:

```python
# Only export these names
__all__ = ['Database', 'open']
```

This prevents imported modules like `sys` or `string` from being re-exported.

**String representation improvements:**

The `repr()` function now uses hex escapes and standard escape sequences instead of octal:

```python
# Python 2.0
>>> repr('\n')
"'\\012'"

# Python 2.1
>>> repr('\n')
"'\\n'"
```

## Build System and Platform Improvements

### PEP 229: New Build System

Python 2.1 revolutionized the build process by using Distutils to automatically detect and compile extension modules. Previously, users had to manually edit `Modules/Setup` to enable additional modules.

**Automatic module detection:**

A `setup.py` script in the Python source distribution runs at build time, examining the system for available modules and headers, then compiling all supported modules automatically. Modules explicitly configured in `Modules/Setup` are respected, allowing platform-specific customization.

**Non-recursive Makefile:**

Neil Schemenauer restructured the build system to use a single non-recursive Makefile instead of separate Makefiles in each subdirectory. This makes builds faster and Makefile modifications simpler.

### PEP 235: Case-Insensitive Import

On platforms with case-insensitive filesystems (MacOS, Windows), Python 2.1 simulates case-sensitivity for imports. By default, Python searches for case-sensitive matches, so `import file` won't import `FILE.PY`:

```python
# Raises ImportError if only FILE.PY exists
import file
```

Set the `PYTHONCASEOK` environment variable to enable case-insensitive matching for backward compatibility.

### Platform-Specific Ports

New platform ports contributed in Python 2.1:
- MacOS X (Steven Majewski)
- Cygwin (Jason Tishler)
- RISCOS (Dietmar Schwertberger)
- Unixware 7 (Billy G. Allie)

## Performance and Implementation Improvements

### Specialized Object Allocator

Python 2.1 introduced an optional specialized object allocator (enabled with `--with-pymalloc`) that's faster than system `malloc()` with less memory overhead. The allocator obtains large memory pools from the system, then fulfills small allocations from these pools.

**Important for C extension authors:**

The object allocator exposes incorrect memory management in C extensions. Previously, Python's allocation functions were simple aliases for `malloc()` and `free()`, masking mismatched function pairs. With the object allocator, using the wrong free function causes crashes:

```c
// Correct pattern
PyObject *obj = PyMem_New(PyObject, count);
PyMem_Del(obj);  // Must use matching free function

// Incorrect - will crash with object allocator
PyObject *obj = PyMem_New(PyObject, count);
free(obj);  // Wrong! Don't use system free()
```

### Line I/O Speed Improvements

Tim Peters rewrote file object's `readline()` method for dramatically improved performance (around 66% faster, varying by platform). This addressed a common complaint about Python's file I/O speed, especially important since file I/O is often used in naive benchmarks.

### Compiler Improvements

Jeremy Hylton's compiler reorganization enabled:
- Better error messages with filename and line numbers in syntax errors
- Foundation for implementing nested scopes
- Improved optimization opportunities

## Other Notable Changes

### PEP 217: Interactive Display Hook

The interactive interpreter can now use custom display functions instead of `repr()`:

```python
>>> L = [1, 2, 3]
>>> L.append(L)
>>> L  # Default output
[1, 2, 3, [...]]

>>> import sys, pprint
>>> sys.displayhook = pprint.pprint
>>> L  # Pretty-printed output
[1, 2, 3, <Recursion on list with id=135143996>]
```

Setting `sys.displayhook` to a callable object lets you customize how values are displayed in the interactive interpreter.

### PEP 208: New Coercion Model

The C-level numeric coercion mechanism was significantly modified for extension type authors. Extension types can set the `Py_TPFLAGS_CHECKTYPES` flag to indicate support for the new model, where numeric methods may receive arguments of different types:

```c
// New coercion model
static PyObject*
mytype_add(PyObject *self, PyObject *other) {
    if (!is_compatible_type(other)) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    // Perform operation
}
```

Returning `Py_NotImplemented` signals that the operation isn't supported for the given types, allowing Python to try the other object's methods or raise `TypeError`.

### PEP 241: Metadata in Python Packages

Packages built with Distutils now include a `PKG-INFO` file containing metadata (name, version, author, description). This represents the first step toward building automated Python package catalogs:

```
Name: MyPackage
Version: 1.0.0
Summary: A useful Python package
Author: John Doe
Author-email: john@example.com
```

While Python 2.1 doesn't automatically submit metadata to catalogs, the infrastructure enables future experimentation with package repositories and search systems.

### sys.excepthook

Ka-Ping Yee contributed another hook for customizing exception handling. Setting `sys.excepthook` to a callable object allows custom handling of uncaught exceptions:

```python
def detailed_exception_handler(type, value, tb):
    # Display extended traceback with local variables
    print_extended_traceback(type, value, tb)

sys.excepthook = detailed_exception_handler
```

At the Ninth Python Conference, Ping demonstrated using this hook to print extended tracebacks showing function arguments and local variables for each stack frame.

### C API Improvements

**PyImport_ImportModule:**

C extensions importing other modules should use `PyImport_ImportModule()`, which respects import hooks, enabling custom import mechanisms to work with C extensions.

**Memory allocation functions:**

The introduction of the specialized object allocator requires correct pairing of allocation and deallocation functions in C extensions.

### XML Package Updates

The PyXML package received multiple updates, included in Python 2.1:
- Support for Expat 1.2 and later
- Expat parsers handle files in any Python-supported encoding
- Various bugfixes for SAX, DOM, and minidom modules

### curses.panel

Thomas Gellekum contributed `curses.panel`, wrapping the panel library for managing overlapping windows with depth ordering. Panels can be moved higher or lower in the depth stack, with the library handling visibility calculations.

### Unicode Character Database

Fredrik Lundh shrunk the Unicode character database by an additional 340K through optimization efforts.

## Key Takeaways

1. **Nested scopes fundamentally improved Python's functional programming capabilities**, eliminating awkward workarounds for closures and lambda expressions while making code more intuitive and readable.

2. **The `__future__` directive mechanism established a template for managing incompatible changes**, giving users full release cycles to adapt code before new behavior becomes mandatory.

3. **Rich comparisons enabled sophisticated numeric computing**, allowing matrix libraries and other numeric types to implement element-wise comparisons and return non-Boolean results.

4. **The warning framework provided systematic deprecation management**, allowing gradual feature removal while giving users clear migration paths.

5. **Weak references solved memory management problems** in caches and circular data structures without requiring explicit cleanup code.

6. **Documentation and testing infrastructure matured significantly** with pydoc, inspect, unittest, and doctest, establishing Python's culture of documentation and testing.

7. **Build system automation improved user experience**, automatically detecting and compiling available extension modules instead of requiring manual configuration.

8. **Performance improvements in file I/O and the optional specialized allocator** addressed common performance complaints and improved memory efficiency.

9. **The PEP process proved effective** for managing complex language changes through documented proposals, community discussion, and implementation tracking.

Python 2.1 represents a pivotal release that addressed fundamental language limitations while establishing processes for managing future evolution. The nested scopes feature alone justified the release, but the combination of language improvements, standard library additions, and infrastructure enhancements made Python 2.1 a solid foundation for future development. The successful introduction of potentially breaking changes through the `__future__` mechanism set a precedent for managing Python's evolution while minimizing disruption to existing codebases.
