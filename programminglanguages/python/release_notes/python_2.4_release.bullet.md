# Python 2.4 Release Notes

**Released:** November 30, 2004
**EOL:** December 19, 2008

## Major Highlights

Python 2.4 introduces powerful new syntax features and critical standard library additions:

1. **Function decorators (PEP 318)** - @ syntax for wrapping functions with metadata and behavior modifications
2. **Generator expressions (PEP 289)** - Memory-efficient lazy evaluation with (x for x in iterable) syntax
3. **Built-in set types (PEP 218)** - Fast set() and frozenset() for membership testing and mathematical operations
4. **decimal module (PEP 327)** - Arbitrary-precision decimal arithmetic for financial calculations
5. **subprocess module (PEP 324)** - Unified, safer interface for spawning and managing subprocesses
6. **collections.deque** - Double-ended queue with O(1) append/pop operations at both ends
7. **Integer/long unification (PEP 237)** - Seamless handling of large integers without overflow warnings

## New Features

### Language Syntax

- 🔴 **Syntax** Function decorators (PEP 318) - @ syntax for applying wrapper functions to methods and functions
  - Example: `@classmethod`, `@staticmethod`, or custom decorators
  - Replaces manual wrapping: `meth = classmethod(meth)`
- 🔴 **Syntax** Generator expressions (PEP 289) - Memory-efficient iteration with (x for x in iterable if condition)
  - Similar to list comprehensions but returns generator instead of materializing list
  - Parentheses required: `(x*x for x in range(10))`
- 🔴 **Syntax** Multi-line imports (PEP 328) - Parentheses allow splitting import lists across lines
  - Example: `from module import (name1, name2, name3)`
  - No backslash continuation needed
- 🟡 **str** ljust(), rjust(), and center() methods now accept optional fill character parameter
- 🟡 **str** rsplit() method splits from end of string - Useful for splitting on last occurrence
- 🟡 **list** sort() method gains key, cmp, and reverse parameters - More flexible and efficient sorting
  - key parameter for custom sort keys (faster than cmp)
  - reverse parameter for descending sort
  - Guaranteed stable sort (equal elements maintain original order)
- 🟡 **dict** update() method now accepts same argument forms as dict() constructor - Mappings, iterables, and keyword arguments
- 🟢 **None** None is now a constant - Assigning to None raises SyntaxError

### Built-in Types & Functions

- 🔴 **set** Built-in set() and frozenset() types (PEP 218) - Fast membership testing, deduplication, and set operations
  - set() is mutable, frozenset() is immutable and hashable
  - Mathematical operations: union (|), intersection (&), difference (-), symmetric difference (^)
  - Replaces sets.Set and sets.ImmutableSet from Python 2.3
- 🔴 **builtins** reversed(seq) function (PEP 322) - Iterate over sequence in reverse order
  - Faster and more memory-efficient than [::-1] slicing
- 🔴 **builtins** sorted(iterable) function - Returns new sorted list from any iterable
  - Accepts same key, cmp, and reverse parameters as list.sort()
  - Non-mutating alternative to list.sort()
- 🟡 **builtins** eval() and exec now accept any mapping type for locals parameter - Not just dict
- 🟡 **zip** zip() and itertools.izip() now return empty list when called with no arguments
- 🟡 **int/long** Integer operations no longer trigger OverflowWarning - Seamless int/long unification continues (PEP 237)
  - Expressions like 2 << 32 now return correct long integer instead of 0

### Standard Library - New Modules

- 🔴 **subprocess** New subprocess module (PEP 324) - Unified interface for spawning and managing subprocesses
  - Replaces os.system(), os.spawn*(), os.popen*(), popen2.*, and commands.*
  - Popen class with comprehensive control over stdin/stdout/stderr, environment, working directory
  - call() convenience function for simple subprocess execution
- 🔴 **decimal** Decimal data type (PEP 327) - Arbitrary-precision decimal arithmetic
  - Avoids binary floating-point inaccuracies (e.g., 0.1 + 0.2 != 0.3)
  - Critical for financial and monetary calculations
  - Configurable precision and rounding modes via Context objects
- 🔴 **cookielib** Client-side HTTP cookie handling - Stores and manages cookies like web browsers
  - Policy objects control cookie acceptance
  - Compatible with Netscape and Mozilla cookie file formats
  - Integrated with urllib2 via HTTPCookieProcessor
- 🟡 **string** Template class (PEP 292) - Simpler string substitution with $ syntax
  - Example: `Template('$name: $value').substitute(name='foo', value=42)`
  - safe_substitute() ignores missing keys instead of raising KeyError

### collections Module

- 🔴 **collections** deque class - Double-ended queue with O(1) append/pop at both ends
  - Methods: append(), appendleft(), pop(), popleft(), rotate()
  - Used internally by Queue and threading modules for improved performance

### itertools Module

- 🔴 **itertools** groupby(iterable, key=None) function - Groups consecutive elements by key value
  - Similar to SQL GROUP BY or Unix uniq command
  - Returns iterator of (key, group_iterator) tuples
- 🟡 **itertools** tee(iterator, N=2) function - Split one iterator into N independent iterators
  - All iterators return same values
  - Note: May need to buffer values if iterators diverge

### operator Module

- 🟡 **operator** attrgetter(attr) and itemgetter(index) functions - Extract attributes or items from objects
  - Excellent for use with map() and sorted()
  - Example: `sorted(L, key=operator.itemgetter(1))`

### Other Standard Library Enhancements

- 🔴 **heapq** Module rewritten in C - 10x performance improvement
  - New nlargest() and nsmallest() functions for finding top N values without full sort
- 🔴 **email** Updated to version 3.0 with incremental FeedParser - No longer requires loading entire message into memory
  - Records malformed message issues in defect attribute instead of raising exceptions
- 🟡 **doctest** Major refactoring with DocTestFinder and DocTestRunner classes - More flexible and customizable testing
- 🟡 **logging** basicConfig() function added for simple log configuration - Keyword arguments for filename, level, and format
  - New TimedRotatingFileHandler for time-based log rotation
- 🟡 **threading** threading.local() class for thread-local data - Simple attribute-based access to thread-specific values
- 🟡 **os** urandom(n) function returns n bytes of cryptographic-quality random data
  - Uses /dev/urandom on Linux, CryptoAPI on Windows
- 🟡 **os.path** lexists(path) returns True for broken symbolic links - Differs from exists() which returns False for broken symlinks
- 🟡 **random** getrandbits(N) method returns random N-bit long integer
  - Makes randrange() more efficient for large ranges
- 🟡 **socket** socketpair() returns pair of connected sockets
  - getservbyport(port) looks up service name for port number
- 🟡 **re** Conditional expressions added: (?(group)A|B) - Test if group matched and use pattern A or B accordingly
  - No longer recursive - Can match arbitrarily large patterns without stack overflow
- 🟡 **poplib** POP over SSL support added
- 🟡 **profile** Can now profile C extension functions
- 🟡 **weakref** Support expanded to functions, class instances, sets, frozensets, deques, arrays, files, sockets, and regex patterns
- 🟡 **xmlrpclib** Multi-call extension for batching multiple XML-RPC calls in single HTTP request
- 🟡 **timeit** Automatically disables garbage collection during timing loop - More consistent timing results
- 🟡 **tarfile** Now generates GNU-format tar files by default
- 🟢 **base64** Complete RFC 3548 support for Base64, Base32, and Base16 encoding
- 🟢 **bisect** Rewritten in C for improved performance
- 🟢 **difflib** HtmlDiff class creates HTML side-by-side comparison tables
- 🟢 **httplib** Constants for HTTP status codes (OK, CREATED, MOVED_PERMANENTLY, etc.)
- 🟢 **imaplib** THREAD command support, deleteacl() and myrights() methods
- 🟢 **nntplib** description() and descriptions() methods for newsgroup descriptions
- 🟢 **optparse** Messages now pass through gettext.gettext() for internationalization
  - Help messages support %default placeholder for default values
- 🟢 **locale** New functions like bind_textdomain_codeset() and l*gettext() family
- 🟢 **marshal** Shares interned strings on unpacking - Significantly smaller .pyc files

### Encodings

- 🟡 **Encodings** CJKCodecs integrated - Comprehensive East Asian character encoding support
  - Chinese: gb2312, gbk, gb18030, big5, big5hkscs, hz, cp950
  - Japanese: cp932, shift-jis, euc-jp, iso-2022-jp (and multiple variants)
  - Korean: cp949, euc-kr, johab, iso-2022-kr
- 🟢 **Encodings** Additional encodings: HP Roman8, ISO_8859-11, ISO_8859-16, PCTP-154, TIS-620
- 🟢 **Encodings** UTF-8 and UTF-16 codecs handle partial input better - StreamReader can resume decoding

## Improvements

### Performance

- 🔴 **Performance** ~5% faster than Python 2.3 on pystone benchmark (35% faster than Python 2.2)
- 🟡 **Performance** List and tuple slicing ~33% faster - Optimized inner loops
- 🟡 **Performance** Dictionary operations faster - Optimized keys(), values(), items(), and iter* variants
- 🟡 **Performance** List appending and popping faster - More efficient growth/shrink machinery, less realloc() calls
- 🟡 **Performance** list.extend() no longer creates temporary list
- 🟡 **Performance** List comprehensions ~33% faster - New LIST_APPEND opcode
- 🟡 **Performance** list(), tuple(), map(), filter(), zip() several times faster with objects that provide __len__()
- 🟡 **Performance** list.__getitem__(), dict.__getitem__(), dict.__contains__() 2x faster - Implemented as method_descriptor
- 🟡 **Performance** String concatenation (s = s + "abc", s += "abc") optimized in certain cases
- 🟡 **Performance** Peephole bytecode optimizer improved - Shorter, faster, more readable bytecode

### Other Improvements

- 🟡 **Interpreter** -m switch runs module as script - Example: `python -m profile script.py`
  - Searches sys.path for module and executes it
- 🟡 **Import** Failed imports no longer leave partially-initialized modules in sys.modules - Prevents confusing subsequent import errors
- 🟢 **Functions** func_name attribute now writable - Decorators can set function names for better tracebacks

## Deprecations

- 🟡 **sys** sys.exitfunc() deprecated - Use atexit module instead for exit handlers
- 🟡 **Modules** mpz, rotor, and xreadlines modules removed

## Implementation Details

### C API

- 🟡 **C API** Locale-independent float/string conversion functions added (PEP 331)
  - PyOS_ascii_strtod(), PyOS_ascii_atof(), PyOS_ascii_formatd()
  - Allows extensions to change numeric locale without breaking Python internals
- 🟢 **C API** func_name attribute made writable for decorator support

### Bytecode

- 🟢 **Bytecode** New LIST_APPEND opcode for faster list comprehensions
- 🟢 **Bytecode** Improved peephole optimizer produces shorter, faster code

## Platform & Environment

- 🟢 **Interpreter** Better compatibility with numeric locale settings (PEP 331) - Extensions like GTK+ can now set numeric locale
