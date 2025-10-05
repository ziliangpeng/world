# Python 2.2 Release Notes

**Released:** December 21, 2001
**EOL:** May 30, 2003 (last bugfix release 2.2.3; superseded by Python 2.3 released July 29, 2003)

## Major Highlights

Python 2.2 is the "cleanup release" that fundamentally transforms Python's object model and adds revolutionary new language features:

1. **Type/class unification (PEP 252, 253)** - Can now subclass built-in types like list, dict, int; new-style classes eliminate classic class limitations
2. **Generators (PEP 255)** - New yield keyword for creating resumable functions; foundation for modern Python iteration patterns
3. **Iterators (PEP 234)** - Universal iteration protocol with __iter__() and next() methods; sequences can define custom iteration
4. **Descriptors** - New descriptor protocol enables properties, static methods, class methods, and slots
5. **Nested scopes (PEP 227)** - Static scoping now default; closures work intuitively without workarounds
6. **Integer/long unification begins (PEP 237)** - Automatic conversion between int and long; 'L' suffix becoming optional
7. **Multiple inheritance improvements** - New method resolution order (MRO) using "diamond rule" for correct superclass resolution

## New Features

### Type/Class Unification & New-Style Classes

- 🔴 **Type System** Can subclass built-in types (list, dict, int, str, file, etc.) - Subclasses work everywhere the original type is expected
- 🔴 **Type System** New object base class for creating new-style classes - `class C(object):` enables all new features
- 🔴 **Type System** Built-in type objects (int, float, str, dict, file) are now callable type constructors
- 🔴 **Descriptors** New descriptor protocol for attribute access - Foundation for properties, methods, slots
- 🔴 **Methods** Static methods via staticmethod() - Methods that don't receive instance or class
- 🔴 **Methods** Class methods via classmethod() - Methods that receive the class as first argument
- 🔴 **Properties** property() built-in for computed attributes - Define get/set/delete methods with optional docstring
  - Replaces most __getattr__()/__setattr__() use cases
  - Cleaner syntax and better performance than attribute hooks
- 🔴 **Slots** __slots__ class attribute restricts instance attributes - Prevents typos and enables future optimizations
  - Define as tuple/list of allowed attribute names
  - Instances without __slots__ won't have __dict__
- 🔴 **MRO** New method resolution order for multiple inheritance - "Diamond rule" ensures superclass methods called correctly
  - Follows depth-first left-to-right, removing all but last duplicate
  - Matches Common Lisp MRO
- 🔴 **super()** New super() built-in for calling superclass methods - super(Class, obj).method() follows MRO correctly
  - Avoids hardcoding parent class names
  - Essential for cooperative multiple inheritance
- 🟡 **Attributes** __getattribute__() method for new-style classes - Called for every attribute access (not just missing ones)
- 🟡 **Attributes** Attributes can now have docstrings via property()
- 🟡 **Metaclasses** __metaclass__ module/class attribute controls class creation
- 🟢 **Attributes** __truediv__() and __floordiv__() methods for division operator overloading

### Generators & Iterators

- 🔴 **Syntax** yield keyword creates generator functions (PEP 255) - Resumable functions that maintain state between calls
  - Requires `from __future__ import generators` in Python 2.2 (not needed in 2.3+)
  - Generator returns iterator object supporting next() method
  - Raises StopIteration when exhausted
  - Cannot use yield in try...finally blocks
- 🔴 **Iterators** Universal iteration protocol (PEP 234) - Objects define __iter__() returning iterator with next() method
  - for loops now use iteration protocol, not just __getitem__()
  - Backward compatible: sequences without __iter__() get automatic iterator
- 🔴 **Iterators** iter(obj) and iter(callable, sentinel) built-in functions
  - iter(obj) returns iterator for any iterable
  - iter(callable, sentinel) calls callable until it returns sentinel value
- 🔴 **dict** Dictionaries support iteration - for key in dict iterates over keys
  - iterkeys(), itervalues(), iteritems() methods return iterators
  - `key in dict` now works as alternative to dict.has_key(key)
- 🔴 **file** Files are iterable - `for line in file:` reads lines without readline()
  - More memory efficient than readlines()
  - Cannot reset or copy file iterators

### Integer/Long Unification

- 🔴 **int/long** Automatic shift from int to long when needed (PEP 237) - No more OverflowError for large integers
  - 'L' suffix no longer required for long integer literals
  - Operations automatically return long integers when result exceeds int range
  - type() can still distinguish int vs long, but rarely needed
  - 'L' suffix will trigger warnings in future (2.4+) and be removed in Python 3.0

### Division Operator Changes

- 🔴 **Syntax** New // operator for floor division (PEP 238) - Always returns floor result regardless of operand types
  - `1 // 2` is 0, `1.0 // 2.0` is 0.0
  - Available by default, no __future__ import needed
- 🟡 **Future** `from __future__ import division` makes / perform true division - `1/2` becomes 0.5 instead of 0
  - Without __future__ import, / still does "classic division" (int/int→int, float involved→float)
  - Default will change to true division in Python 3.0
  - Command-line flag `-Qwarn` warns on classic division of integers
  - `-Qwarn` will become default in Python 2.3

### Nested Scopes

- 🔴 **Scoping** Nested scopes now default (PEP 227) - Inner functions can reference variables from enclosing functions
  - No longer need `from __future__ import nested_scopes`
  - Fixes lambda default argument workaround: `lambda x, name=name:` no longer needed
  - Variables are local, enclosing function scope(s), module-level, or built-in
  - Enables true closures
- 🔴 **Scoping** `from module import *` and exec illegal in function with nested scopes - SyntaxError if function contains nested functions/lambdas with free variables
  - Compiler cannot determine names at compile time
  - exec and import * only legal at module level

### Unicode Improvements

- 🟡 **Unicode** Can compile with UCS-4 (32-bit) Unicode support - `--enable-unicode=ucs4` configure option
  - "Wide Python" supports U+000000 to U+110000
  - "Narrow Python" (default UCS-2) raises ValueError for unichr() values > 65535
  - See PEP 261
- 🟡 **Unicode** decode() method added to 8-bit strings - Symmetric with encode() on Unicode strings
  - `str.decode(encoding)` assumes str is in specified encoding
  - Enables non-Unicode codecs like uu-encoding, base64, zlib, rot-13
- 🟡 **Unicode** __unicode__() method for converting instances to Unicode - Analogous to __str__()
- 🟢 **Unicode** `--disable-unicode` configure option to completely disable Unicode support

## New and Improved Modules

### New Modules

- 🔴 **xmlrpclib** XML-RPC client support - Simple remote procedure call protocol over HTTP/XML
  - Server class for making RPC calls
  - Automatic marshaling of Python types
- 🔴 **SimpleXMLRPCServer** Create straightforward XML-RPC servers
- 🟡 **hmac** HMAC algorithm implementation (RFC 2104) - Keyed-hashing for message authentication

### Module Improvements

- 🔴 **os, time** Functions return pseudo-sequences with named attributes - stat(), fstat(), statvfs(), localtime(), gmtime(), strptime()
  - Still behave like tuples for backward compatibility
  - Access fields by name: `os.stat(filename).st_size` instead of `os.stat(filename)[stat.ST_SIZE]`
  - time tuples have tm_year, tm_mon, etc.
- 🔴 **help()** New interactive help built-in in interpreter - Uses pydoc module for documentation
  - `help(object)` displays help text for object
  - `help()` with no arguments enters interactive help mode
- 🟡 **socket** IPv6 support when compiled with `--enable-ipv6`
- 🟡 **struct** New 'q' and 'Q' format characters for 64-bit integers - Signed and unsigned long long on supporting platforms
- 🟡 **re** Performance improvements and bug fixes in SRE engine
  - re.sub() and re.split() rewritten in C
  - New finditer() method returns iterator over non-overlapping matches
  - BIGCHARSET patch speeds up certain Unicode ranges 2x
- 🟡 **smtplib** Secure SMTP over TLS support (RFC 2487) - Encrypt SMTP traffic
  - SMTP authentication support
- 🟡 **imaplib** New extensions: NAMESPACE (RFC 2342), SORT, GETACL, SETACL
- 🟡 **rfc822** Email address parsing now RFC 2822 compliant
- 🟡 **email** New package for parsing and generating email messages - Foundation for Mailman
- 🟡 **difflib** New Differ class for human-readable change lists ("deltas")
  - ndiff() and restore() generator functions
  - Produces deltas between two sequences
- 🟡 **string** New ascii_letters, ascii_lowercase, ascii_uppercase constants - Use instead of string.letters for A-Za-z
  - string.letters varies by locale (incorrect assumption in many modules)
- 🟡 **mimetypes** MimeTypes class for using alternative MIME-type databases
- 🟡 **threading** Timer class for scheduling future activities
- 🟡 **profiler** Extensively reworked with corrected output errors

## Improvements

### Performance

- 🟡 **Performance** Multiple inheritance lookups faster with new MRO algorithm
- 🟡 **Performance** New-style classes enable future optimizations
- 🟡 **Performance** __slots__ may enable memory optimizations in future versions

### Error Messages & Debugging

- 🟡 **Error Messages** Better error messages for attribute access and type mismatches

## Implementation Details

### C API & Interpreter Changes

- 🟡 **C API** PyEval_SetProfile() and PyEval_SetTrace() for C-based profiling/tracing - Much faster than Python-based profiling
  - sys.setprofile() and sys.settrace() now use C-level interface
- 🟡 **C API** PyInterpreterState_Head(), PyInterpreterState_Next(), PyThreadState_Next() for introspection
- 🟡 **C API** PyErr_NewException() accepts NULL base class
- 🟡 **C API** tp_iter and tp_iternext slots for extension types to support iteration protocol
- 🟡 **C API** PyNumberMethods slots for __truediv__() and __floordiv__()
- 🟡 **Build** Improved support for creating DLLs on Windows
- 🟡 **Build** Distutils gets new commands and options
- 🟢 **Bytecode** Bytecode changes to support new features (generators, iterators, nested scopes)
- 🟢 **Bytecode** Generator bytecode detected and handled specially by compiler
- 🟢 **Interpreter** Better cycle garbage collection of new-style classes

### Platform Support

- 🟡 **Platform** IPv6 support on platforms with IPv6 stack (`--enable-ipv6`)
- 🟡 **Platform** UCS-4 Unicode support on platforms supporting it (`--enable-unicode=ucs4`)
- 🟡 **Platform** Improved Windows DLL creation

## Compatibility Notes

### Classic vs New-Style Classes

- 🔴 **Compatibility** Two kinds of classes in Python 2.2: classic (old-style) and new-style
  - Classic classes: Don't inherit from anything or inherit from other classic classes
  - New-style classes: Inherit from object or any built-in type
  - All new features (properties, slots, descriptors, static/class methods, new MRO, super()) only work with new-style classes
  - Classic classes will be removed eventually, possibly in Python 3.0
  - Existing code using classic classes continues to work unchanged

### Future Changes

- 🟡 **Future** 'L' suffix for long integers will trigger warnings in Python 2.4 and be removed in Python 3.0
- 🟡 **Future** Division operator / will change to true division in Python 3.0 (use `from __future__ import division` to test)
- 🟡 **Future** Classic classes will be removed, possibly in Python 3.0
- 🟡 **Future** Generators will not require `from __future__ import generators` in Python 2.3+

### Breaking Changes

- 🟢 **Syntax** `from module import *` and exec now illegal in functions with nested scopes - SyntaxError when function contains nested functions/lambdas with free variables
  - Only affects code using these features together (rare)
  - Best practice: Avoid exec and import * in functions anyway

### Deprecations

- 🟢 **posixfile** Module made obsolete by ability to subclass file and add lock() method
- 🟢 **Long Integer** 'L' suffix will be discouraged in future versions (warnings in 2.4+, removed in 3.0)

## Related PEPs

- PEP 227: Statically Nested Scopes
- PEP 234: Iterators
- PEP 237: Unifying Long Integers and Integers
- PEP 238: Changing the Division Operator
- PEP 252: Making Types Look More Like Classes
- PEP 253: Subtyping Built-in Types
- PEP 255: Simple Generators
- PEP 261: Support for 'wide' Unicode characters
