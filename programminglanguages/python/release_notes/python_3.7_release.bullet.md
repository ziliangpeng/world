# Python 3.7 Release Notes

**Released:** June 27, 2018
**EOL:** June 2023 (reached - no longer supported)

## Major Highlights

Python 3.7 focused on developer productivity, async programming, and performance:

1. **Dataclasses (PEP 557)** - @dataclass decorator automatically generates __init__, __repr__, __eq__, and other methods
2. **Context variables (PEP 567)** - Async-aware state management (like thread-local but works correctly with async code)
3. **Built-in breakpoint() (PEP 553)** - Standardized debugger entry point
4. **async and await reserved keywords** - Breaking change requiring code updates if used as identifiers
5. **Dict ordering official** - Insertion-order preservation now part of language specification
6. **Nanosecond time functions (PEP 564)** - time.time_ns() and 5 other functions with better precision
7. **Method calls 20% faster** - Bytecode optimizations avoiding bound method creation

## Breaking Changes

- 🔴 **Syntax** async and await are now reserved keywords - Code using them as identifiers raises SyntaxError
- 🔴 **stdlib** os.stat_float_times() removed - Was deprecated since Python 3.1
- 🟡 **stdlib** ntpath.splitunc() removed - Use os.path.splitdrive() instead
- 🟡 **collections** namedtuple() verbose parameter and _source attribute removed
- 🟡 **builtins** bool(), float(), list(), tuple() no longer accept keyword arguments - First int() argument must be positional
- 🟡 **plistlib** Removed Plist, Dict, _InternalDict classes - Dict values now normal dicts
- 🟡 **asyncio** Removed asyncio.windows_utils.socketpair() - Use socket.socketpair()
- 🟡 **asyncio** No longer exports selectors and _overlapped modules
- 🟡 **ssl** Direct instantiation of SSLSocket and SSLObject prohibited - Use SSLContext methods
- 🟡 **distutils** Removed install_misc command
- 🟡 **re** Unknown escapes in replacement templates now raise error - Were deprecated in 3.5
- 🟡 **tarfile** Removed exclude argument from TarFile.add() - Use filter parameter
- 🟢 **fpectl** Module completely removed
- 🟢 **Behavior** PEP 479 enabled for all code - StopIteration in generators/coroutines transformed to RuntimeError
- 🟢 **Behavior** object.__aiter__() cannot be async method - Must return async iterator
- 🟢 **Behavior** Generator expressions require direct parentheses - No comma on either side
- 🟢 **sys.path** With -m switch, sys.path[0] is now full directory path instead of empty string

## Deprecations

### Removing in Python 3.8

- 🔴 **asyncio** Direct await of asyncio.Lock and sync primitives - Use async context manager
- 🟡 **asyncio** asyncio.Task.current_task() and Task.all_tasks() - Use module-level asyncio.current_task() and all_tasks()
- 🟡 **collections** ABCs in collections module - Import from collections.abc instead
- 🟡 **dbm.dumb** Warning when index file is missing and recreated in 'r'/'w' modes
- 🟡 **enum** Checking non-Enum in Enum or non-Flag in Flag - Will raise TypeError
- 🟡 **gettext** Non-integer plural form selection
- 🟡 **importlib** MetaPathFinder.find_module() and PathEntryFinder.find_loader() - Use find_spec()
- 🟡 **importlib.abc** ResourceLoader ABC - Use ResourceReader
- 🟡 **locale** locale.format() - Use locale.format_string()
- 🟡 **macpath** Entire module deprecated
- 🟡 **threading** dummy_threading and _dummy_thread modules - Use threading
- 🟡 **socket** Argument truncation in socket.htons() and ntohs() - Will raise exception
- 🟡 **ssl** ssl.wrap_socket() - Use ssl.SSLContext.wrap_socket()
- 🟡 **sys** sys.set_coroutine_wrapper() and get_coroutine_wrapper()
- 🟡 **sys** sys.callstats() undocumented function
- 🟡 **aifc** aifc.openfp() - Use aifc.open()
- 🟡 **sunau** sunau.openfp() - Use sunau.open()
- 🟡 **wave** wave.openfp() - Use wave.open()

### Deprecation Behavior Changes

- 🟡 **Behavior** Yield expressions in comprehensions and generator expressions - SyntaxError in Python 3.8
- 🟡 **Behavior** Returning complex subclass from __complex__() - Will be error in future

### C API Deprecations

- 🟢 **C API** PySlice_GetIndicesEx() - Use PySlice_Unpack() and PySlice_AdjustIndices()
- 🟢 **C API** PyOS_AfterFork() - Use PyOS_BeforeFork(), PyOS_AfterFork_Parent(), PyOS_AfterFork_Child()

## New Features

### Language Syntax

- 🔴 **Type System** Postponed evaluation of annotations (PEP 563) - Enables forward references, improves startup time
  - Use `from __future__ import annotations`
  - Annotations stored as strings, evaluated via typing.get_type_hints()
  - Becomes default in Python 3.10

- 🟡 **Syntax** Await expressions and async for allowed in f-strings

- 🟢 **Syntax** Functions can now have more than 255 parameters/arguments

### Data Classes and Context Variables

- 🔴 **Data Model** Dataclasses module (PEP 557) - @dataclass decorator for automatic method generation
  - Generates __init__, __repr__, __eq__, __hash__ methods
  - `@dataclass` decorator with class variable annotations

- 🔴 **Data Model** Context variables module (PEP 567) - Async-aware state management
  - New contextvars module with ContextVar class
  - asyncio and decimal modules updated to use context variables
  - Replaces thread-local storage for async code

### Built-in and Core Features

- 🔴 **Debugging** Built-in breakpoint() function (PEP 553) - Calls sys.breakpointhook()
  - Defaults to pdb.set_trace()
  - Customizable via PYTHONBREAKPOINT environment variable
  - Set PYTHONBREAKPOINT=0 to disable

- 🟡 **Data Model** Module __getattr__ and __dir__ customization (PEP 562) - For deprecation and lazy loading

- 🟡 **Data Model** Core support for typing module and generic types (PEP 560)
  - New __class_getitem__() and __mro_entries__() methods
  - Type operations up to 7x faster
  - No more metaclass conflicts with generic types

- 🟡 **Syntax** Dict ordering preservation now official part of language spec - Was implementation detail in 3.6

- 🟡 **builtins** str, bytes, bytearray gained isascii() method

- 🟡 **types** TracebackType can be instantiated from Python, tb_next is writable

### Interpreter & Runtime

- 🟡 **Interpreter** Legacy C locale coercion (PEP 538) - Auto-switches to UTF-8 locale
  - PYTHONCOERCECLOCALE environment variable
  - Coerces C/POSIX locale to UTF-8 variants

- 🟡 **Interpreter** Forced UTF-8 runtime mode (PEP 540) - -X utf8 option or PYTHONUTF8
  - Ignores locale settings, uses UTF-8 by default
  - Works when UTF-8 locale unavailable

- 🟡 **Interpreter** Python Development Mode - -X dev option or PYTHONDEVMODE
  - Enables additional runtime checks
  - Shows ResourceWarning, enables faulthandler, asyncio debug mode

- 🟡 **Interpreter** Hash-based .pyc files (PEP 552) - Deterministic bytecode caching
  - Source hash instead of timestamp for validation
  - Supports reproducible builds
  - Two variants: checked and unchecked

- 🟡 **Warnings** DeprecationWarning shown in __main__ by default (PEP 565)

- 🟡 **Profiling** -X importtime option or PYTHONPROFILEIMPORTTIME - Shows module import timing

### Standard Library

- 🔴 **time** Six nanosecond-resolution time functions (PEP 564)
  - time.time_ns(), monotonic_ns(), perf_counter_ns(), process_time_ns(), clock_gettime_ns(), clock_settime_ns()
  - Return int nanoseconds, 3x better resolution than float versions

- 🟡 **time** time.thread_time() and thread_time_ns() - Per-thread CPU time
- 🟡 **time** New clock identifiers: CLOCK_BOOTTIME, CLOCK_PROF, CLOCK_UPTIME
- 🟡 **time** time.pthread_getcpuclockid() function

- 🔴 **asyncio** asyncio.run() function - Run coroutine from sync code (provisional)
- 🟡 **asyncio** asyncio.create_task() - Shortcut for get_event_loop().create_task()
- 🟡 **asyncio** asyncio.get_running_loop() - Returns current loop, raises error if none
- 🟡 **asyncio** asyncio.current_task() and all_tasks() - Replace Task methods
- 🟡 **asyncio** Context variables support - Tasks track context automatically
- 🟡 **asyncio** BufferedProtocol class - Manual receive buffer control
- 🟡 **asyncio** Server improvements - start_serving parameter, async context manager support
- 🟡 **asyncio** loop.start_tls() - Upgrade connection to TLS
- 🟡 **asyncio** loop.sock_recv_into() - Read directly into buffer
- 🟡 **asyncio** loop.sock_sendfile() - Use os.sendfile() when available
- 🟡 **asyncio** Future.get_loop(), Task.get_loop(), Server.get_loop() methods
- 🟡 **asyncio** Server.start_serving(), serve_forever(), is_serving() methods
- 🟡 **asyncio** StreamWriter.wait_closed() and is_closing() methods
- 🟡 **asyncio** TimerHandle.when() returns scheduled timestamp
- 🟡 **asyncio** create_datagram_endpoint() supports Unix sockets
- 🟡 **asyncio** ssl_handshake_timeout parameter for connection methods
- 🟡 **asyncio** Handle.cancelled() method
- 🟡 **asyncio** ReadTransport.is_reading() method, idempotent pause/resume
- 🟡 **asyncio** Path-like object support for Unix socket paths
- 🟡 **asyncio** TCP sockets created with TCP_NODELAY by default on Linux
- 🟡 **asyncio** WindowsSelectorEventLoopPolicy and WindowsProactorEventLoopPolicy classes

- 🟡 **importlib.resources** New module for accessing package resources
  - read_text(), read_binary(), path() functions
  - importlib.abc.ResourceReader ABC

- 🟡 **collections** namedtuple() supports default values
- 🟡 **datetime** datetime.fromisoformat() parses ISO format strings
- 🟡 **datetime** tzinfo supports sub-minute offsets
- 🟡 **gc** gc.freeze(), unfreeze(), get_freeze_count() - Copy-on-write friendly GC
- 🟡 **os** os.register_at_fork() - Register callbacks at process fork
- 🟡 **os** os.preadv() and pwritev() - Vectored I/O operations
- 🟡 **os** os.scandir() supports file descriptors
- 🟡 **os** os.fwalk() accepts bytes path argument
- 🟡 **os** os.dup2() now returns new file descriptor
- 🟡 **os** os.stat() contains st_fstype on Solaris
- 🟡 **pathlib** Path.is_mount() method for POSIX systems
- 🟡 **re** Flags can be set within group scope
- 🟡 **re** re.split() supports patterns matching empty strings
- 🟡 **re** Compiled regex objects can be copied with copy.copy()
- 🟡 **socket** socket.getblocking() method
- 🟡 **socket** socket.close() function for file descriptors
- 🟡 **socket** TCP_CONGESTION, TCP_USER_TIMEOUT, TCP_NOTSENT_LOWAT constants
- 🟡 **socket** AF_VSOCK support for VM-host communication
- 🟡 **socket** Auto-detect family, type, protocol from file descriptor
- 🟡 **sqlite3** Connection.backup() method for database backup
- 🟡 **sqlite3** Path-like object support for database parameter
- 🟡 **ssl** OpenSSL built-in hostname verification instead of match_hostname()
- 🟡 **ssl** Preliminary TLS 1.3 support (experimental)
- 🟡 **ssl** SSLContext.minimum_version and maximum_version properties
- 🟡 **ssl** SSLContext.post_handshake_auth for TLS 1.3
- 🟡 **ssl** SSLContext.hostname_checks_common_name customization
- 🟡 **ssl** No longer sends IP addresses in SNI extension
- 🟡 **ssl** IDN validation support
- 🟡 **subprocess** run() accepts capture_output parameter
- 🟡 **subprocess** text parameter as alias for universal_newlines
- 🟡 **subprocess** Windows: close_fds defaults to True when redirecting
- 🟡 **subprocess** Better KeyboardInterrupt handling
- 🟡 **sys** sys.breakpointhook() called by breakpoint()
- 🟡 **sys** sys.getandroidapilevel() on Android
- 🟡 **sys** sys.get/set_coroutine_origin_tracking_depth() - Replace set_coroutine_wrapper()
- 🟡 **argparse** ArgumentParser.parse_intermixed_args() method
- 🟡 **binascii** b2a_uu() accepts backtick parameter
- 🟡 **calendar** HTMLCalendar CSS class customization attributes
- 🟡 **compileall** compile_dir() invalidation_mode parameter for hash-based .pyc
- 🟡 **concurrent.futures** ProcessPoolExecutor and ThreadPoolExecutor accept initializer/initargs
- 🟡 **concurrent.futures** ProcessPoolExecutor accepts mp_context parameter
- 🟡 **contextlib** nullcontext() no-op context manager
- 🟡 **contextlib** asynccontextmanager(), AbstractAsyncContextManager, AsyncExitStack
- 🟡 **cProfile** Accepts -m module_name
- 🟡 **crypt** Blowfish hashing method support
- 🟡 **crypt** mksalt() rounds parameter
- 🟡 **dis** dis() can disassemble nested code objects with depth parameter
- 🟡 **distutils** README.rst included in source distributions
- 🟡 **enum** Enum._ignore_ class property
- 🟡 **functools** singledispatch() supports type annotation registration
- 🟡 **hmac** hmac.digest() one-shot function (up to 3x faster)
- 🟡 **http.client** HTTPConnection and HTTPSConnection accept blocksize parameter
- 🟡 **http.server** SimpleHTTPRequestHandler supports If-Modified-Since header
- 🟡 **http.server** SimpleHTTPRequestHandler accepts directory parameter
- 🟡 **http.server** ThreadingHTTPServer class
- 🟡 **importlib** importlib.abc.ResourceReader ABC
- 🟡 **importlib** importlib.reload() raises ModuleNotFoundError when module lacks spec
- 🟡 **importlib.util** find_spec() raises ModuleNotFoundError for non-package parent
- 🟡 **importlib.util** source_hash() computes hash for hash-based .pyc
- 🟡 **io** TextIOWrapper.reconfigure() method
- 🟡 **ipaddress** subnet_of() and supernet_of() methods for containment tests
- 🟡 **itertools** itertools.islice() accepts integer-like objects
- 🟡 **locale** locale.format_string() monetary parameter
- 🟡 **locale** getpreferredencoding() returns 'UTF-8' on Android or forced UTF-8 mode
- 🟡 **logging** Logger instances can be pickled
- 🟡 **logging** StreamHandler.setStream() method
- 🟡 **logging** fileConfig() supports keyword arguments for handlers
- 🟡 **math** math.remainder() function (IEEE 754-style)
- 🟡 **mimetypes** .bmp MIME type changed to 'image/bmp'
- 🟡 **msilib** Database.Close() method
- 🟡 **multiprocessing** Process.close() method
- 🟡 **multiprocessing** Process.kill() method using SIGKILL
- 🟡 **multiprocessing** Non-daemonic threads now joined on process exit
- 🟡 **pdb** pdb.set_trace() accepts header parameter
- 🟡 **pdb** Command line accepts -m module_name
- 🟡 **py_compile** Respects SOURCE_DATE_EPOCH for reproducible builds
- 🟡 **pydoc** Server can bind to arbitrary hostname with -n option
- 🟡 **queue** SimpleQueue class (unbounded FIFO)
- 🟡 **signal** signal.set_wakeup_fd() warn_on_full_buffer parameter
- 🟡 **string** Template now allows separate regex patterns for braced/non-braced placeholders
- 🟡 **threading** threading.Thread.is_shutdown() class method
- 🟡 **tkinter** tkinter.ttk.Spinbox class
- 🟡 **tracemalloc** Traceback.format() accepts negative limit and most_recent_first parameter
- 🟡 **types** WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, ClassMethodDescriptorType
- 🟡 **types** types.resolve_bases() resolves MRO entries dynamically
- 🟡 **unicodedata** Updated to Unicode 11
- 🟡 **unittest** -k option for filtering tests by pattern
- 🟡 **unittest.mock** sentinel attributes preserve identity when copied/pickled
- 🟡 **unittest.mock** seal() function seals Mock instances
- 🟡 **urllib.parse** urllib.parse.quote() updated from RFC 2396 to RFC 3986
- 🟡 **uu** uu.encode() accepts backtick parameter
- 🟡 **uuid** UUID.is_safe attribute
- 🟡 **uuid** uuid.getnode() prefers universally administered MAC addresses
- 🟡 **xml** xml.dom.minidom and xml.sax no longer process external entities by default
- 🟡 **xml.etree** ElementPath predicates can compare current node text with [.="text"]
- 🟡 **xmlrpc.server** SimpleXMLRPCDispatcher.register_function() usable as decorator
- 🟡 **zipapp** create_archive() accepts filter parameter
- 🟡 **zipapp** create_archive() accepts compressed parameter and --compress option
- 🟡 **zipfile** ZipFile accepts compresslevel parameter
- 🟡 **zipfile** Subdirectories sorted alphabetically

### Improved Error Messages

- 🟡 **Error Messages** ImportError displays module name and __file__ path for from...import failures

### Language Behavior

- 🟡 **Behavior** bytes.fromhex() and bytearray.fromhex() ignore all ASCII whitespace
- 🟡 **Behavior** Circular imports with absolute imports now supported
- 🟡 **Behavior** object.__format__(x, '') now equivalent to str(x)
- 🟡 **Behavior** locals() dictionary displays in lexical order

## Improvements

### Performance

- 🔴 **Performance** Method calls up to 20% faster - Bytecode changes avoid bound method creation
- 🔴 **Performance** Startup time 10% faster on Linux, 30% faster on macOS
- 🔴 **Performance** METH_FASTCALL reduces calling overhead for many C methods
- 🔴 **Performance** asyncio.get_event_loop() up to 15x faster (C implementation)
- 🟡 **Performance** asyncio.Future callback management optimized
- 🟡 **Performance** asyncio.gather() up to 15% faster
- 🟡 **Performance** asyncio.sleep() up to 2x faster for zero/negative delays
- 🟡 **Performance** asyncio debug mode overhead reduced
- 🟡 **Performance** typing module import 7x faster, operations faster
- 🟡 **Performance** sorted() and list.sort() 40-75% faster for common cases
- 🟡 **Performance** dict.copy() up to 5.5x faster
- 🟡 **Performance** hasattr() and getattr() ~4x faster when attribute not found
- 🟡 **Performance** Unicode character search up to 8x faster (worst case 3x slower)
- 🟡 **Performance** collections.namedtuple() creation 4-6x faster
- 🟡 **Performance** datetime.date.fromordinal() and fromtimestamp() up to 30% faster
- 🟡 **Performance** os.fwalk() 2x faster using os.scandir()
- 🟡 **Performance** shutil.rmtree() 20-40% faster
- 🟡 **Performance** re case-insensitive matching up to 20x faster
- 🟡 **Performance** re.compile() ~10% faster
- 🟡 **Performance** selectors modify() methods ~10% faster under heavy loads
- 🟡 **Performance** Constant folding moved to AST optimizer for consistent optimizations
- 🟡 **Performance** abc module functions/methods rewritten in C - 1.5x faster isinstance/issubclass, 10% faster startup
- 🟡 **Performance** datetime alternate constructors faster via fast-path
- 🟡 **Performance** array.array comparison 10x-70x faster for same integer types
- 🟡 **Performance** math.erf() and erfc() use C library implementation

### Other Improvements

- 🟡 **Security** xml modules no longer process external entities by default
- 🟡 **Warnings** Warning filter initialization changes - Command-line options have precedence
- 🟡 **Warnings** Debug builds display all warnings by default
- 🟡 **socketserver** ThreadingMixIn.server_close and ForkingMixIn.server_close wait for completion
- 🟡 **socketserver** block_on_close attribute to control blocking behavior
- 🟡 **dbm.dumb** Supports read-only files, doesn't write unchanged index

## Implementation Details

### CPython Bytecode

- 🟢 **Bytecode** New opcodes: LOAD_METHOD, CALL_METHOD for faster method calls
- 🟢 **Bytecode** Removed STORE_ANNOTATION opcode

### CPython Implementation

- 🟢 **Interpreter** Trace hooks can opt out of line events, opt into opcode events via f_trace_lines and f_trace_opcodes
- 🟢 **Interpreter** Namespace module objects: __file__ set to None, __spec__.origin set to None
- 🟢 **Interpreter** Distutils upload no longer changes CR to CRLF
- 🟢 **Interpreter** Exception state moved from frame to coroutine object
- 🟢 **Interpreter** Docstring return None marked as occurring on docstring line
- 🟢 **Interpreter** Internal startup/configuration management refactored (PEP 432 prep)

### C API

- 🟡 **C API** New Thread Specific Storage API (PEP 539) - Py_tss_t type for portability
- 🟡 **C API** Context variables C API exposed
- 🟡 **C API** PyImport_GetModule() function
- 🟡 **C API** Py_RETURN_RICHCOMPARE macro
- 🟡 **C API** Py_UNREACHABLE macro
- 🟡 **C API** PyTraceMalloc_Track() and PyTraceMalloc_Untrack()
- 🟡 **C API** import__find__load__start and import__find__load__done static markers
- 🟡 **C API** PyMemberDef, PyGetSetDef, PyStructSequence_Field fields now const char*
- 🟡 **C API** PyUnicode_AsUTF8AndSize() and AsUTF8() return const char*
- 🟡 **C API** PyMapping_Keys(), Values(), Items() always return list
- 🟡 **C API** PySlice_Unpack() and PySlice_AdjustIndices() functions
- 🟡 **C API** PyOS_BeforeFork(), AfterFork_Parent(), AfterFork_Child() replace PyOS_AfterFork()
- 🟡 **C API** PyExc_RecursionErrorInst singleton removed
- 🟡 **C API** PyTimeZone_FromOffset() and related timezone constructors
- 🟡 **C API** PyThread_start_new_thread() and get_thread_ident() return unsigned long
- 🟡 **C API** PyUnicode_AsWideCharString() raises ValueError on NULL with embedded nulls
- 🟡 **C API** Py_Initialize() requirement more strictly enforced
- 🟡 **C API** PyInterpreterState_GetID() function
- 🟡 **C API** Py_DecodeLocale()/EncodeLocale() use UTF-8 in UTF-8 mode
- 🟡 **C API** PyUnicode_DecodeLocaleAndSize()/EncodeLocale() use current locale for surrogateescape
- 🟡 **C API** PyUnicode_FindChar() start/end adjusted to behave like string slices

### Build System

- 🟡 **Build** --without-threads removed - threading always available
- 🟡 **Build** libffi no longer bundled on non-OSX Unix - Must be installed
- 🟡 **Build** Windows build uses Python script to download from GitHub instead of Subversion
- 🟡 **Build** ssl module requires OpenSSL 1.0.2 or 1.1 - 0.9.8 and 1.0.1 no longer supported

## Platform & Environment

- 🟡 **Platform** FreeBSD 9 and older no longer supported
- 🟡 **Platform** *nix platforms expected to provide C.UTF-8, C.utf8, or UTF-8 locale
- 🟡 **Platform** OpenSSL 1.0.2+ required - Affects Debian 8, Ubuntu 14.04
- 🟡 **Windows** Python launcher (py.exe) accepts -3-32 and -3-64 specifiers
- 🟡 **Windows** py -0 lists installed pythons, py -0p includes paths
- 🟡 **Windows** <python-executable>._pth file overrides sys.path (was 'sys.path')

## Release Process & Meta Changes

- 🟡 **Documentation** PEP 545: Python documentation now available in Japanese, French, Korean
