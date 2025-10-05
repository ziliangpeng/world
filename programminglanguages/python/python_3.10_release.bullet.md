# Python 3.10 Release Notes

**Released:** October 4, 2021
**EOL:** October 2026 (security support)

## Major Highlights

Python 3.10 brings major syntax improvements and enhanced developer experience:

1. **Structural Pattern Matching (PEP 634-636)** - match/case statements for powerful pattern matching on data structures
2. **Union type operator (PEP 604)** - Write `int | str` instead of `Union[int, str]` for cleaner type hints
3. **Better error messages** - Precise location of syntax errors, suggestions for typos in NameError and AttributeError
4. **Parenthesized context managers** - Multi-line with statements now allowed with parentheses
5. **Precise line numbers (PEP 626)** - Reliable line numbers for debugging, profiling, and coverage tools
6. **Performance improvements** - LOAD_ATTR 36-44% faster, str/bytes/bytearray constructors 30-40% faster
7. **distutils deprecated (PEP 632)** - Scheduled for removal in Python 3.12

## Breaking Changes

- 🔴 **asyncio** Removed loop parameter from most high-level API functions - Use current thread's running loop implicitly
- 🟡 **collections** Removed deprecated aliases to ABCs (Iterable, Mapping, etc.) - Import from collections.abc
- 🟡 **complex** Removed special methods (__int__, __float__, __floordiv__, etc.) that always raised TypeError
- 🟡 **parser** Removed parser module and related C API functions (PyParser_*, PyNode_Compile) - Use Py_CompileString()
- 🟡 **formatter** Removed deprecated formatter module
- 🟡 **urllib.parse** Changed query parameter separator from `;` or `&` to `&` only for security - Use separator parameter if needed
- 🟡 **urllib.parse** Strip newline and tab characters from URLs for security (WHATWG spec compliance)
- 🟡 **Builtin Functions** No longer accept Decimal/Fraction for integer arguments - Only objects with __index__() method
- 🟢 **_markupbase** Removed ParserBase.error() method (private module)
- 🟢 **unicodedata** Removed internal ucnhash_CAPI attribute

## Deprecations

### Removing in Python 3.11

- 🟡 **ssl** Deprecated constants (OP_NO_SSLv2, OP_NO_TLSv1, etc.), protocols (PROTOCOL_TLSv1, etc.), wrap_socket(), match_hostname(), RAND_pseudo_bytes(), NPN features - Use SSLContext methods and ALPN

### Removing in Python 3.12

- 🔴 **distutils** Entire distutils package deprecated (PEP 632) - Use setuptools or modern build tools
- 🟡 **importlib** Many deprecated import APIs: find_loader(), find_module(), load_module(), module_repr(), __package__, __loader__, __cached__ - Use find_spec(), exec_module(), __spec__ attributes
- 🟡 **sqlite3** sqlite3.OptimizedUnicode deprecated (alias to str)
- 🟡 **sqlite3** sqlite3.enable_shared_cache() deprecated - Use URI mode with cache=shared parameter
- 🟡 **pathlib** Path.link_to() deprecated - Use Path.hardlink_to()
- 🟡 **cgi** cgi.log() deprecated
- 🟡 **threading** Old camelCase method names deprecated (currentThread, activeCount, notifyAll, isSet, setName, getName, isDaemon, setDaemon) - Use snake_case equivalents
- 🟡 **PYTHONTHREADDEBUG** Threading debug environment variable deprecated

### Removing in Future Versions

- 🟡 **Syntax** Numeric literals immediately followed by keywords (0in x, 1or x) - Add space between literal and keyword
- 🟡 **random** Non-integer arguments to random.randrange() deprecated - Use integer arguments
- 🟡 **typing** Importing from typing.io and typing.re submodules - Import directly from typing

## New Features

### Language Syntax

- 🔴 **Pattern Matching** Structural pattern matching with match/case statements (PEP 634-636)
  - Match literals, sequences, mappings, classes with powerful destructuring
  - Guards with if clauses
  - Wildcard patterns and named constants
  - Nested patterns and OR patterns with |

- 🔴 **Type Hints** Union type operator `X | Y` (PEP 604) - Alternative to typing.Union
  - Works in type hints: `def func(x: int | str) -> int | None`
  - Works with isinstance() and issubclass()

- 🟡 **Syntax** Parenthesized context managers - Multi-line with statements with trailing commas allowed

### Type Hints

- 🟡 **typing** Parameter Specification Variables (PEP 612) - ParamSpec and Concatenate for better decorator typing
- 🟡 **typing** TypeAlias annotation (PEP 613) - Explicitly mark type aliases: `StrCache: TypeAlias = 'Cache[str]'`
- 🟡 **typing** TypeGuard (PEP 647) - User-defined type guards for type narrowing
- 🟡 **typing** Literal behavior changes (PEP 586) - De-duplicates parameters, order-independent equality, respects types
- 🟡 **typing** is_typeddict() function to introspect TypedDict annotations

### Error Messages

- 🟡 **Error Messages** SyntaxError shows full error range with ^^ highlighting instead of single ^ marker
- 🟡 **Error Messages** SyntaxError points to unclosed bracket/parenthesis location instead of EOF
- 🟡 **Error Messages** Many new specialized SyntaxError messages: missing colon, missing comma, = vs ==, unparenthesized tuple comprehension targets, etc.
- 🟡 **Error Messages** IndentationError includes context about expected block type
- 🟡 **Error Messages** AttributeError suggests similar attribute names: "Did you mean: namedtuple?"
- 🟡 **Error Messages** NameError suggests similar variable names in scope: "Did you mean: schwarzschild_black_hole?"
- 🟡 **Error Messages** SyntaxError has end_lineno and end_offset attributes

### Interpreter & Runtime

- 🟡 **Debugging** PEP 626: Precise line numbers for debugging and profiling - Tracing events generated for all executed lines
- 🟡 **Encoding** Optional EncodingWarning (PEP 597) - Warn when locale-specific encoding used without explicit encoding parameter
- 🟢 **Encoding** encoding="locale" option to explicitly use locale encoding

### Built-in Functions & Types

- 🟡 **int** int.bit_count() method returns number of 1 bits (population count)
- 🟡 **dict** dict.keys(), dict.values(), dict.items() views have .mapping attribute returning MappingProxyType
- 🟡 **zip** zip() accepts strict=True flag to require equal-length iterables (PEP 618)
- 🟡 **Async** aiter() and anext() builtin functions for async iteration
- 🟡 **staticmethod/classmethod** Now inherit method attributes and have __wrapped__, callable as regular functions
- 🟡 **Functions** New __builtins__ attribute for builtin symbol lookup
- 🟡 **Assignment** Assignment expressions allowed unparenthesized in set literals and comprehensions

### Standard Library - Major Additions

- 🟡 **dataclasses** slots parameter to generate __slots__
- 🟡 **dataclasses** kw_only parameter and KW_ONLY marker for keyword-only fields
- 🟡 **asyncio** connect_accepted_socket() method added
- 🟡 **bisect** Key function support in all bisect APIs
- 🟡 **codecs** codecs.unregister() function to unregister codec search functions
- 🟡 **contextlib** aclosing() async context manager
- 🟡 **contextlib** nullcontext() now supports async context manager protocol
- 🟡 **contextlib** AsyncContextDecorator for async context managers as decorators
- 🟡 **enum** StrEnum for string enums
- 🟡 **enum** __repr__() returns enum_name.member_name, __str__() returns member_name
- 🟡 **inspect** get_annotations() function for safely retrieving annotations with un-stringizing support
- 🟡 **itertools** itertools.pairwise() function
- 🟡 **statistics** covariance(), correlation(), and linear_regression() functions

### Standard Library - OS & Path

- 🟡 **os** os.cpu_count() support for VxWorks
- 🟡 **os** os.eventfd() and related helpers for Linux eventfd2 syscall
- 🟡 **os** os.splice() for zero-copy data movement between file descriptors
- 🟡 **os** macOS flags: O_EVTONLY, O_FSYNC, O_SYMLINK, O_NOFOLLOW_ANY
- 🟡 **os.path** os.path.realpath() accepts strict parameter to raise on missing paths
- 🟡 **pathlib** PurePath.parents supports slicing and negative indexing
- 🟡 **pathlib** Path.hardlink_to() method (replaces link_to)
- 🟡 **pathlib** Path.stat() and chmod() accept follow_symlinks parameter
- 🟡 **platform** platform.freedesktop_os_release() to read os-release file

### Standard Library - Other

- 🟡 **argparse** "optional arguments" renamed to "options" in help text
- 🟡 **array** array.index() accepts start and stop parameters
- 🟡 **asynchat/asyncore/smtpd** Now emit DeprecationWarning on import
- 🟡 **base64** b32hexencode() and b32hexdecode() for Base32 with extended hex alphabet
- 🟡 **curses** Extended color support with ncurses 6.1, BUTTON5_* constants
- 🟡 **fileinput** encoding and errors parameters added
- 🟡 **glob** root_dir and dir_fd parameters for relative path searching
- 🟡 **hashlib** Requires OpenSSL 1.1.1+, preliminary OpenSSL 3.0.0 support (PEP 644)
- 🟡 **hmac** Now uses OpenSSL's HMAC implementation internally
- 🟡 **importlib.metadata** Feature parity with importlib_metadata 4.6, EntryPoints class, packages_distributions()
- 🟡 **pprint** underscore_numbers parameter, dataclass support
- 🟡 **shelve** Now uses pickle.DEFAULT_PROTOCOL instead of protocol 3
- 🟡 **socket** socket.timeout is now alias of TimeoutError
- 🟡 **socket** MPTCP support with IPPROTO_MPTCP, IP_RECVTOS option
- 🟡 **ssl** Requires OpenSSL 1.1.1+, preliminary OpenSSL 3.0.0 support (PEP 644)
- 🟡 **ssl** More secure defaults: TLS 1.2 minimum, disabled weak ciphers, security level 2
- 🟡 **ssl** TLS 1.0 and TLS 1.1 no longer officially supported
- 🟡 **ssl** OP_IGNORE_UNEXPECTED_EOF option, VERIFY_X509_PARTIAL_CHAIN flag
- 🟡 **sys** sys.orig_argv attribute with original command line arguments
- 🟡 **sys** sys.stdlib_module_names with list of standard library module names
- 🟡 **_thread** _thread.interrupt_main() accepts optional signal number
- 🟡 **threading** gettrace() and getprofile() functions, __excepthook__ attribute
- 🟡 **traceback** format_exception(), format_exception_only(), print_exception() accept exception object as positional arg
- 🟡 **types** Reintroduced EllipsisType, NoneType, NotImplementedType classes
- 🟡 **unittest** assertNoLogs() method
- 🟡 **xml.sax.handler** LexicalHandler class
- 🟡 **zipimport** PEP 451 methods: find_spec(), create_module(), exec_module(), invalidate_caches()

### IDLE & Tools

- 🟡 **IDLE** sys.excepthook() now invoked (when started without -n)
- 🟡 **IDLE** Shell sidebar with prompts, copy with prompts feature
- 🟡 **IDLE** Spaces instead of tabs for interactive code indentation
- 🟡 **IDLE** Syntax highlighting for match, case, and _ soft keywords
- 🟡 **IDLE** Settings dialog reorganized (Windows and Shell/Ed tabs)

## Improvements

### Performance

- 🟡 **Performance** str(), bytes(), bytearray() constructors 30-40% faster for small objects
- 🟡 **Performance** LOAD_ATTR instruction 36% faster for regular attributes, 44% faster for slots (per-opcode cache)
- 🟡 **Performance** runpy module imports fewer modules, python3 -m startup 1.4x faster
- 🟡 **Performance** --enable-optimizations with --enable-shared up to 30% faster with gcc (-fno-semantic-interposition)
- 🟡 **Performance** bz2/lzma/zlib decompression 1.09-1.32x faster (new buffer management)
- 🟡 **Performance** Stringized annotations lazy-loaded, halves CPU time for annotated function definitions
- 🟡 **Performance** Substring search uses Two-Way algorithm to avoid quadratic behavior
- 🟡 **Performance** _PyType_Lookup() micro-optimizations make interpreter 1.04x faster
- 🟡 **Performance** map(), filter(), reversed(), bool(), float() now use PEP 590 vectorcall
- 🟡 **Performance** BZ2File performance improved by removing internal RLock (now thread-unsafe)

### Other Improvements

- 🟡 **Annotations** Complex target annotations have no runtime effects with __future__.annotations
- 🟡 **Annotations** Classes and modules lazy-create empty __annotations__ dicts
- 🟡 **Annotations** yield, yield from, await, named expressions forbidden in annotations under __future__.annotations
- 🟡 **Hashing** NaN values now hash based on object identity instead of always 0 (avoids quadratic dict/set behavior)
- 🟡 **SyntaxError** Deleting __debug__ now raises SyntaxError instead of NameError
- 🟡 **__ipow__** Correctly falls back to __pow__ and __rpow__ when returning NotImplemented
- 🟡 **collections.abc.Callable** Generic now flattens type parameters like typing.Callable
- 🟡 **collections.abc.Callable** types.GenericAlias can now be subclassed

## Implementation Details

### CPython Bytecode

- 🟢 **Bytecode** MAKE_FUNCTION instruction accepts dict or tuple of strings for annotations
- 🟢 **Bytecode** frame.f_lasti now represents wordcode offset (multiply by 2 for byte offset)

### C API

- 🟡 **C API** PEP 652: Stable ABI explicitly defined with stability guarantees
- 🟡 **C API** co_lnotab deprecated, use co_lines() method (PEP 626)
- 🟢 **C API** Parser API removed (PyParser_SimpleParseStringFlags, etc.) - Use Py_CompileString()
- 🟢 **C API** PyModule_GetWarningsModule() removed

### Build System

- 🟡 **Build** OpenSSL 1.1.1 or newer required (PEP 644)
- 🟡 **Build** C99 snprintf() and vsnprintf() required
- 🟡 **Build** SQLite 3.7.15 or higher required
- 🟡 **Build** atexit module must be built-in
- 🟡 **Build** --disable-test-modules option to exclude test modules
- 🟡 **Build** --with-wheel-pkg-dir option for Linux distributions
- 🟡 **Build** --without-static-libpython option
- 🟡 **Build** configure uses pkg-config for Tcl/Tk detection
- 🟡 **Build** --with-openssl-rpath option

## Platform & Environment

- 🟡 **Platform** OpenSSL 1.1.1+ required on all platforms (PEP 644)
- 🟡 **Platform** Windows and macOS now use OpenSSL 3.0
- 🟡 **Environment** -X warn_default_encoding and PYTHONWARNDEFAULTENCODING for EncodingWarning

## Porting to Python 3.10

### Syntax Changes

- 🟡 **Syntax** Numeric literal followed by keyword emits DeprecationWarning (will become SyntaxError)

### API Changes

- 🟡 **traceback** Parameter etype renamed to exc in format_exception(), format_exception_only(), print_exception()
- 🟡 **atexit** All callback exceptions now logged (previously last exception silently ignored)
- 🟡 **collections.abc.Callable** Type parameters flattened - __args__ is (int, str, str) not ([int, str], str)
- 🟡 **socket** socket.htons() and socket.ntohs() raise OverflowError instead of DeprecationWarning for overflow
- 🟡 **asyncio** loop parameter removed from high-level API - Use implicit current thread loop
- 🟡 **types.FunctionType** Constructor inherits current builtins if globals has no __builtins__ key

### C API Changes

- 🟡 **C API** Parser API removed - Use Py_CompileString() to compile source directly to code object
- 🟡 **C API** frame.f_lasti is wordcode offset - Multiply by 2 for byte offset in APIs like PyCode_Addr2Line()
