# Python 3.8 Release Notes

**Released:** October 14, 2019
**EOL:** October 2024 (security support)

## Major Highlights

Python 3.8 introduces powerful language features for readability and expressiveness:

1. **Walrus operator := (PEP 572)** - Assignment expressions for cleaner code: `if (n := len(a)) > 10`
2. **Positional-only parameters (PEP 570)** - Function syntax with `/` separator for stricter APIs
3. **F-string = specifier** - Self-documenting expressions for debugging: `f'{user=} {value=}'`
4. **typing.TypedDict (PEP 589)** - Per-key type annotations for dictionaries
5. **typing.Literal (PEP 586)** - Constrain values to specific literals
6. **typing.Final (PEP 591)** - Mark variables/methods as final (no override/reassignment)
7. **Pickle protocol 5 (PEP 574)** - Out-of-band data buffers for better large-data serialization

## Breaking Changes

- 🔴 **collections** Importing ABCs from collections was delayed (planned for 3.8, delayed to 3.9) - Use collections.abc
- 🟡 **macpath** Removed macpath module - No longer needed for modern systems
- 🟡 **platform** Removed platform.popen() - Use os.popen()
- 🟡 **time** Removed time.clock() - Use time.perf_counter() or time.process_time()
- 🟡 **venv** Removed pyvenv script - Use `python3.8 -m venv` instead
- 🟡 **cgi** Removed parse_qs, parse_qsl, escape from cgi - Use urllib.parse and html modules
- 🟡 **tarfile** Removed undocumented filemode function
- 🟡 **xml.etree** XMLParser no longer accepts html argument, all parameters now keyword-only
- 🟡 **xml.etree** Removed XMLParser.doctype() method
- 🟡 **codecs** Removed unicode_internal codec
- 🟡 **sqlite3** Cache and Statement objects no longer exposed to users
- 🟡 **fileinput** Removed bufsize keyword argument (ignored since 3.6)
- 🟡 **sys** Removed sys.set_coroutine_wrapper() and sys.get_coroutine_wrapper()
- 🟡 **asyncio** asyncio.CancelledError now inherits from BaseException (not Exception)
- 🟡 **Windows** DLL dependencies now resolved more securely - PATH and CWD no longer searched, use os.add_dll_directory()
- 🟡 **Syntax** Yield expressions disallowed in comprehensions and generator expressions
- 🟡 **Syntax** SyntaxWarning for identity checks (is/is not) with literals
- 🟡 **Windows** os.getcwdb() now uses UTF-8 encoding (PEP 529)
- 🟢 **AIX** sys.platform always 'aix' (not 'aix3'..'aix7')
- 🟢 **math** math.factorial() no longer accepts non-int-like arguments

## Deprecations

### Removing in Python 3.9

- 🟡 **ElementTree** getchildren() and getiterator() methods
- 🟡 **asyncio** Passing non-ThreadPoolExecutor to loop.set_default_executor()

### Removing in Python 3.10

- 🔴 **asyncio** @asyncio.coroutine decorator - Use async def
- 🔴 **asyncio** Explicit loop argument in asyncio functions (sleep, gather, shield, wait_for, wait, as_completed, Task, Lock, Event, Condition, Semaphore, BoundedSemaphore, Queue, subprocess functions)
- 🟡 **C API** Use of # format variants without PY_SSIZE_T_CLEAN defined

### Removing in Python 3.11

- 🟡 **asyncio** Passing coroutine objects directly to asyncio.wait()

### Pending Removal (no specific version)

- 🟡 **distutils** bdist_wininst command - Use bdist_wheel
- 🟡 **gettext** lgettext(), ldgettext(), lngettext(), ldngettext() functions
- 🟡 **gettext** bind_textdomain_codeset(), output_charset(), set_output_charset()
- 🟡 **threading** isAlive() method - Use is_alive()
- 🟡 **typing** NamedTuple._field_types attribute - Use __annotations__
- 🟡 **ast** Num, Str, Bytes, NameConstant, Ellipsis classes - Use ast.Constant
- 🟡 **ast.NodeVisitor** visit_Num(), visit_Str(), visit_Bytes(), visit_NameConstant(), visit_Ellipsis() - Use visit_Constant()
- 🟡 **builtins** Decimal, Fraction for integer arguments (objects with __int__ but not __index__)
- 🟢 **xml.dom.pulldom** DOMEventStream.__getitem__()
- 🟢 **wsgiref** FileWrapper.__getitem__()
- 🟢 **fileinput** FileInput.__getitem__()
- 🟢 **Multiple modules** Various callback/function parameters deprecated as keyword arguments (will be positional-only)

## New Features

### Language Syntax

- 🔴 **Syntax** Assignment expressions with := "walrus operator" (PEP 572) - `if (n := len(a)) > 10`
- 🔴 **Syntax** Positional-only parameters with / separator (PEP 570) - `def f(a, b, /, c, d, *, e, f)`
- 🔴 **Syntax** F-string = specifier for debugging (PEP 367) - `f'{user=} {member_since=}'`
- 🟡 **Syntax** continue statement now allowed in finally clause
- 🟡 **Syntax** Dict and dictviews iterable in reversed insertion order with reversed()
- 🟡 **Syntax** Generalized iterable unpacking in yield/return without parentheses
- 🟡 **Syntax** Dict comprehensions now compute key first, then value (matches dict literals)
- 🟡 **re** Support for \N{name} Unicode escapes in regular expressions
- 🟢 **Syntax** SyntaxWarning for missing comma in tuples (e.g., [(10, 20) (30, 40)])

### Type System

- 🔴 **typing** TypedDict for per-key dictionary types (PEP 589) - `class Location(TypedDict): lat_long: tuple`
- 🔴 **typing** Literal types (PEP 586) - `def get_status() -> Literal['connected', 'disconnected']`
- 🔴 **typing** Final qualifier (PEP 591) - `pi: Final[float] = 3.14159`
- 🔴 **typing** Protocol definitions (PEP 544) - Structural subtyping with typing.Protocol
- 🟡 **typing** typing.SupportsIndex protocol
- 🟡 **typing** typing.get_origin() and typing.get_args() introspection functions

### Built-in Types

- 🟡 **builtins** bool, int, Fraction now have as_integer_ratio() method (like float/Decimal)
- 🟡 **builtins** int(), float(), complex() now use __index__() if available
- 🟡 **builtins** sum() uses Neumaier summation for better float accuracy
- 🟡 **builtins** memoryview supports half-float type ('e' format)
- 🟡 **builtins** pow(base, -1, mod) computes modular inverse
- 🟡 **types.CodeType** New replace() method for code objects with 19 parameters
- 🟡 **slice** Slice objects now hashable
- 🟡 **types** types.MappingProxyType instances now hashable

### Standard Library - New Modules

- 🔴 **importlib.metadata** (Provisional) Access installed package metadata (version, requirements, files)
- 🟡 **multiprocessing.shared_memory** Shared memory for direct cross-process access

### asyncio

- 🔴 **asyncio** asyncio.run() graduated from provisional to stable API
- 🟡 **asyncio** python -m asyncio launches native async REPL with top-level await
- 🟡 **asyncio** ProactorEventLoop now default on Windows (supports UDP, KeyboardInterrupt)
- 🟡 **asyncio** Task.get_coro() for getting wrapped coroutine
- 🟡 **asyncio** Tasks can be named (name parameter, set_name(), get_name())
- 🟡 **asyncio** Happy Eyeballs support in loop.create_connection() for IPv4/IPv6

### ast

- 🟡 **ast** AST nodes now have end_lineno and end_col_offset attributes
- 🟡 **ast** ast.get_source_segment() returns source code for specific AST node
- 🟡 **ast** ast.parse() type_comments=True for PEP 484/526 type comments
- 🟡 **ast** ast.parse() mode='func_type' for PEP 484 signature type comments
- 🟡 **ast** ast.parse() feature_version=(3, N) to parse as earlier Python version
- 🟡 **ast** All literals now represented as ast.Constant (Num, Str, Bytes, NameConstant, Ellipsis unified)

### functools

- 🟡 **functools** lru_cache() can be used as decorator without calling: `@lru_cache`
- 🟡 **functools** cached_property() decorator for cached computed properties
- 🟡 **functools** singledispatchmethod() for generic methods using single dispatch

### math

- 🟡 **math** math.dist() for Euclidean distance between two points
- 🟡 **math** math.hypot() expanded to handle multiple dimensions (not just 2-D)
- 🟡 **math** math.prod() returns product of iterable (analogous to sum())
- 🟡 **math** math.perm() and math.comb() for permutations and combinations
- 🟡 **math** math.isqrt() for accurate integer square roots

### statistics

- 🟡 **statistics** statistics.fmean() faster floating-point mean
- 🟡 **statistics** statistics.geometric_mean()
- 🟡 **statistics** statistics.multimode() returns list of most common values
- 🟡 **statistics** statistics.quantiles() divides data into equiprobable intervals
- 🟡 **statistics** statistics.NormalDist for normal distributions (mean, stdev, cdf, pdf, samples)
- 🟡 **statistics** statistics.mode() no longer raises on multimodal data (returns first mode)

### pathlib & os

- 🟡 **pathlib** Path and PurePath support subclassing via with_segments()
- 🟡 **pathlib** Path.walk() method (like os.walk())
- 🟡 **pathlib** PurePath.relative_to() supports walk_up parameter for .. entries
- 🟡 **pathlib** Path.is_junction() for Windows junctions
- 🟡 **pathlib** Path.glob/rglob/match support case_sensitive parameter
- 🟡 **pathlib** link_to() creates hard links (deprecated in 3.10, removed in 3.12)
- 🟡 **os** os.add_dll_directory() on Windows for extension module dependencies
- 🟡 **os** os.memfd_create() wraps memfd_create() syscall
- 🟡 **os** os.listdrives(), os.listvolumes(), os.listmounts() on Windows
- 🟡 **os** os.stat()/os.lstat() more accurate on Windows (st_birthtime, 64-bit st_dev, 128-bit st_ino)
- 🟡 **os** DirEntry.is_junction() for Windows junctions
- 🟡 **os** os.PIDFD_NONBLOCK for non-blocking pidfd_open()
- 🟡 **os.path** os.path.isjunction() for Windows junctions
- 🟡 **os.path** os.path.splitroot() splits path into (drive, root, tail)
- 🟡 **os.path** Boolean functions return False instead of raising for unrepresentable paths
- 🟡 **os.path** expanduser() on Windows prefers USERPROFILE (not HOME)
- 🟡 **os.path** realpath() on Windows resolves reparse points/symlinks/junctions
- 🟡 **Windows** os.readlink() reads directory junctions

### Other Standard Library

- 🟡 **datetime** datetime.date.fromisocalendar() and datetime.datetime.fromisocalendar() constructors
- 🟡 **datetime** Arithmetic with date/datetime subclasses returns subclass instances
- 🟡 **collections** namedtuple._asdict() returns dict (not OrderedDict)
- 🟡 **csv** DictReader returns dict instances (not OrderedDict), faster with less memory
- 🟡 **cProfile** Profile class can be used as context manager
- 🟡 **curses** ncurses_version variable for structured version info
- 🟡 **ctypes** CDLL accepts winmode parameter on Windows for LoadLibraryEx flags
- 🟡 **compile** Accepts ast.PyCF_ALLOW_TOP_LEVEL_AWAIT flag for top-level await/async
- 🟡 **gc** gc.get_objects() accepts optional generation parameter
- 🟡 **gettext** pgettext() and variants for context-aware translations
- 🟡 **gzip** gzip.compress() mtime parameter for reproducible output
- 🟡 **gzip** BadGzipFile exception for invalid gzip files (not OSError)
- 🟡 **inspect** getdoc() finds docstrings in __slots__ dict
- 🟡 **io** IOBase finalizer logs exceptions in dev mode/debug build
- 🟡 **itertools** itertools.accumulate() initial keyword argument
- 🟡 **json.tool** --json-lines option to parse each line as separate JSON
- 🟡 **logging** logging.basicConfig() force parameter to remove existing handlers
- 🟡 **mmap** mmap.madvise() method to access madvise() system call
- 🟡 **multiprocessing** spawn start method now default on macOS
- 🟡 **pickle** Pickler.reducer_override() method for custom pickling logic
- 🟡 **pickle** __reduce__() can return 6-element tuple with state-updater callable
- 🟡 **plistlib** plistlib.UID for NSKeyedArchiver-encoded binary plists
- 🟡 **pprint** sort_dicts parameter and pprint.pp() convenience function
- 🟡 **py_compile** compile() supports silent mode
- 🟡 **shlex** shlex.join() inverse of shlex.split()
- 🟡 **shutil** copytree() dirs_exist_ok keyword argument
- 🟡 **shutil** make_archive() defaults to modern pax (POSIX.1-2001) format
- 🟡 **shutil** rmtree() removes directory junctions on Windows without recursing
- 🟡 **socket** create_server() and has_dualstack_ipv6() convenience functions
- 🟡 **socket** if_nameindex(), if_nametoindex(), if_indextoname() on Windows
- 🟡 **ssl** post_handshake_auth and verify_client_post_handshake() for TLS 1.3
- 🟡 **sqlite3** Command-line interface: python -m sqlite3
- 🟡 **sqlite3** Connection.autocommit attribute, load_extension() entrypoint, getconfig()/setconfig()
- 🟡 **sys** sys.unraisablehook() for handling unraisable exceptions
- 🟡 **tarfile** Defaults to modern pax (POSIX.1-2001) format
- 🟡 **threading** threading.excepthook() for uncaught thread exceptions
- 🟡 **threading** get_native_id() function and Thread.native_id attribute
- 🟡 **tokenize** Implicitly emits NEWLINE token when input lacks trailing newline
- 🟡 **tkinter** Spinbox selection methods, Canvas.moveto(), PhotoImage transparency methods
- 🟡 **time** CLOCK_UPTIME_RAW clock for macOS 10.12
- 🟡 **unicodedata** Updated to Unicode 12.1.0
- 🟡 **unicodedata** is_normalized() to verify normalization without normalizing
- 🟡 **unittest** AsyncMock for async version of Mock
- 🟡 **unittest** addModuleCleanup() and addClassCleanup()
- 🟡 **unittest** Mock assert functions print actual calls on failure
- 🟡 **unittest** IsolatedAsyncioTestCase for coroutine test cases
- 🟡 **uuid** Command-line interface: python -m uuid
- 🟢 **IDLE** Output squeezing, "Run Customized", line numbers, emoji support, many UI improvements

## Improvements

### Performance

- 🟡 **Performance** asyncio socket writes 75% faster
- 🟡 **Performance** asyncio.current_task() 4-6x faster (C implementation)
- 🟡 **Performance** tokenize 64% faster
- 🟡 **Performance** inspect.getattr_static() at least 2x faster
- 🟡 **Performance** os.stat()/os.lstat() significantly faster on newer Windows
- 🟡 **Performance** Garbage collector runs on eval breaker (not object allocations)
- 🟢 **Performance** Experimental BOLT binary optimizer support (1-5% improvement)

### Error Messages

- 🟡 **Error Messages** Compiler suggests missing comma in malformed tuples
- 🟡 **Error Messages** KeyboardInterrupt exits via SIGINT with correct exit code

### Security

- 🟡 **Security** SHA1/SHA3/SHA2-384/SHA2-512/MD5 replaced with HACL* formally verified implementations
- 🟡 **Security** xml.dom.minidom and xml.sax no longer process external entities by default

### Developer Experience

- 🟡 **Dev** Debug/release builds now ABI compatible (Py_DEBUG doesn't imply Py_TRACE_REFS)
- 🟡 **Dev** C extensions not linked to libpython on Unix (except Android/Cygwin)
- 🟡 **Dev** PYTHONPYCACHEPREFIX environment variable for separate bytecode cache
- 🟡 **Dev** sys.pycache_prefix for bytecode cache location

## Implementation Details

### Runtime & Interpreter

- 🟡 **Interpreter** Stack overflow protection on supported platforms
- 🟡 **Interpreter** CPython support for Linux perf profiler (PYTHONPERFSUPPORT, -X perf)
- 🟡 **Runtime** Python Initialization Configuration API (PEP 587) - PyConfig, PyPreConfig, PyStatus
- 🟢 **Runtime** Python Runtime Audit Hooks (PEP 578) for security monitoring
- 🟢 **Performance** Vectorcall protocol (PEP 590) - Fast calling protocol for CPython (provisional)

### C API

- 🟡 **C API** New python3-config --embed option required for embedding Python
- 🟡 **C API** pkg-config python-3.8-embed module for embedding
- 🟡 **C API** PyEval_ReInitThreads() removed - Use PyOS_AfterFork_Child()
- 🟡 **C API** PyCompilerFlags.cf_feature_version field for version-specific parsing
- 🟡 **C API** Heap-allocated type instances hold reference to type object
- 🟡 **C API** Py_DEPRECATED() macro implemented for MSVC
- 🟡 **C API** PyInterpreterState moved to internal headers (opaque in public API)
- 🟡 **C API** PyGC_Head struct changed completely
- 🟢 **C API** pgen header files and functions removed (replaced by pure Python)

### Build System

- 🟡 **Build** python3-config --libs no longer contains -lpython3.8
- 🟢 **Build** va_start() with two parameters now required

### Porting Notes

- 🟡 **Porting** subprocess.Popen can use os.posix_spawn() for better performance
- 🟡 **Porting** subprocess.Popen preexec_fn no longer compatible with subinterpreters
- 🟡 **Porting** imaplib.IMAP4.logout() no longer silently ignores exceptions
- 🟡 **Porting** tkinter.ttk.Treeview.selection() no longer accepts arguments
- 🟡 **Porting** xml.dom.minidom and xml.etree preserve attribute order
- 🟡 **Porting** dbm.dumb opened with 'r' is read-only, 'r'/'w' don't create database
- 🟡 **Porting** dbm databases raise error (not KeyError) when deleting from read-only
- 🟡 **Porting** RuntimeError for metaclass missing __classcell__ (was DeprecationWarning)
- 🟡 **Porting** mmap.flush() returns None on success (was platform-dependent)
- 🟡 **Porting** shutil.copy* functions use platform-specific fast-copy syscalls
- 🟡 **Porting** shutil.copyfile() buffer size changed from 16 KiB to 1 MiB on Windows
- 🟡 **Porting** types.CodeType constructor has new posonlyargcount parameter (2nd position)
- 🟡 **Porting** hmac.new() digestmod parameter no longer defaults to MD5
- 🟢 **Porting** Exceptions raised when getting attributes no longer ignored
- 🟢 **Porting** Builtin types inherit __str__ from object (affects subclass repr)
- 🟢 **Porting** xml.etree.XMLParser.doctype() emits RuntimeWarning (not DeprecationWarning)
- 🟢 **Porting** PyEval_AcquireLock()/PyEval_AcquireThread() terminate if interpreter finalizing

## Platform & Environment

- 🟡 **Environment** PYTHONPYCACHEPREFIX for separate bytecode cache tree
- 🟡 **Environment** PYTHONPERFSUPPORT for perf profiler support
- 🟡 **Environment** -X pycache_prefix command-line option
- 🟡 **Environment** -X perf command-line option for perf profiler
