# Python 3.9 Release Notes

**Released:** October 5, 2020
**EOL:** October 2025 (security support)

## Major Highlights

Python 3.9 focuses on syntax improvements and standard library additions for common use cases:

1. **Dictionary merge operator |= and | (PEP 584)** - Convenient syntax for merging dicts: `x | y`
2. **Type hinting with built-in generics (PEP 585)** - Use `list[str]` instead of `typing.List[str]`
3. **String prefix/suffix removal methods (PEP 616)** - `str.removeprefix()` and `str.removesuffix()`
4. **zoneinfo module (PEP 615)** - IANA Time Zone Database in standard library
5. **graphlib module** - Topological sorting for graphs
6. **PEG parser (PEP 617)** - New parser replacing LL(1), enables future syntax flexibility
7. **Annual release cycle (PEP 602)** - CPython adopts yearly releases

## Breaking Changes

- 🟡 **collections** Last release with collections ABC aliases (e.g., collections.Mapping) - Use collections.abc versions
- 🟡 **nntplib** Removed xpath() and xgtitle() methods (deprecated since 3.3)
- 🟡 **array** Removed tostring() and fromstring() methods - Use tobytes() and frombytes()
- 🟡 **sys** Removed sys.callstats(), sys.getcheckinterval(), sys.setcheckinterval()
- 🟡 **threading** Removed threading.Thread.isAlive() method - Use is_alive()
- 🟡 **xml.etree.ElementTree** Removed getchildren() and getiterator() methods
- 🟡 **plistlib** Removed old API (readPlist, writePlist, etc.) - Use load(), loads(), dump(), dumps()
- 🟡 **base64** Removed base64.encodestring() and base64.decodestring() - Use encodebytes() and decodebytes()
- 🟡 **fractions** Removed fractions.gcd() - Use math.gcd()
- 🟡 **bz2** Removed buffering parameter from bz2.BZ2File
- 🟡 **json** Removed encoding parameter from json.loads()
- 🟡 **asyncio** Removed old context manager syntax for locks (with await asyncio.lock) - Use async with
- 🟡 **asyncio** Removed asyncio.Task.current_task() and asyncio.Task.all_tasks() - Use module-level functions
- 🟡 **html.parser** Removed HTMLParser.unescape() method - Use html.unescape()
- 🟡 **typing** Removed typing.NamedTuple._field_types attribute - Use __annotations__
- 🟡 **symtable** Removed symtable.SymbolTable.has_exec() method
- 🟡 **_dummy_thread, dummy_threading** Removed modules (deprecated since 3.7)
- 🟡 **aifc, sunau, wave** Removed openfp() aliases - Use open()
- 🟡 **C API** Removed PyImport_Cleanup(), PyGen_NeedsFinalizing, tp_print slot

## Deprecations

### Removing in Python 3.10

- 🟡 **parser, symbol** parser and symbol modules deprecated - Use ast module
- 🟡 **C API** PyParser_SimpleParseStringFlags and related parser functions deprecated

### Removing in Python 3.11

- 🟡 **asyncio** Passing coroutine objects to asyncio.wait() deprecated
- 🟡 **C API** PyEval_InitThreads() and PyEval_ThreadsInitialized() deprecated

### Removing in Future Versions

- 🟡 **distutils** bdist_msi command deprecated - Use bdist_wheel
- 🟡 **math** math.factorial() accepting float values deprecated - Will require int
- 🟡 **NotImplemented** Using NotImplemented in boolean context deprecated
- 🟡 **random** random module will restrict seed types to None, int, float, str, bytes, bytearray
- 🟡 **gzip** Opening GzipFile for writing without mode argument deprecated
- 🟡 **_tkinter** _tkinter.TkappType.split() deprecated - Use splitlist()
- 🟡 **binhex** binhex module and related binascii functions deprecated
- 🟡 **ast** ast.slice, ast.Index, ast.ExtSlice deprecated
- 🟡 **ast** ast.Suite, ast.Param, ast.AugLoad, ast.AugStore deprecated
- 🟡 **shlex** Passing None to shlex.split() deprecated
- 🟡 **smtpd** smtpd.MailmanProxy deprecated (unusable)
- 🟡 **lib2to3** lib2to3 module emits PendingDeprecationWarning - Consider LibCST or parso
- 🟡 **random** random parameter of random.shuffle() deprecated

## New Features

### Language Syntax

- 🔴 **Syntax** Dictionary merge and update operators (PEP 584)
  - Merge operator: `x | y` creates new dict
  - Update operator: `x |= y` updates in place
  - Works with dict and dict-like mappings

- 🔴 **Type System** Type hinting generics in standard collections (PEP 585)
  - Use `list[str]`, `dict[str, int]` instead of typing.List, typing.Dict
  - Also works with tuple, set, frozenset, queue.Queue, etc.
  - Simplifies type annotations without importing typing

- 🔴 **Syntax** Relaxed decorator grammar restrictions (PEP 614)
  - Any valid expression can now be used as decorator
  - Previously restricted to dotted names and function calls

- 🟡 **Syntax** Unparenthesized lambda in comprehension if clause no longer allowed

### Built-in Features

- 🔴 **str, bytes, bytearray** New removeprefix() and removesuffix() methods (PEP 616)
  - `'Hello World'.removeprefix('Hello ')` → `'World'`
  - `'test.py'.removesuffix('.py')` → `'test'`
  - More explicit than strip() or string slicing

### Standard Library - New Modules

- 🔴 **zoneinfo** IANA Time Zone Database support (PEP 615)
  - New zoneinfo.ZoneInfo class for timezone-aware datetime objects
  - Replaces third-party pytz for many use cases
  - Falls back to tzdata package on platforms without system timezone data

- 🟡 **graphlib** Topological sorting module
  - graphlib.TopologicalSorter class for ordering graph nodes
  - Useful for dependency resolution and task scheduling

### Standard Library - Improvements

- 🟡 **typing** typing.Annotated for context-specific metadata (PEP 593)
  - Decorate types with additional information
  - New include_extras parameter for typing.get_type_hints()

- 🟡 **ast** ast.unparse() function to convert AST back to source code
- 🟡 **ast** ast.dump() indent parameter for multiline output
- 🟡 **asyncio** asyncio.to_thread() coroutine for running functions in threads
- 🟡 **asyncio** asyncio.PidfdChildWatcher for Linux process management
- 🟡 **asyncio** loop.shutdown_default_executor() for cleaner shutdown
- 🟡 **asyncio** Removed reuse_address parameter for create_datagram_endpoint() (security)
- 🟡 **compileall** --hardlink-dupes option to hardlink duplicate .pyc files
- 🟡 **concurrent.futures** cancel_futures parameter for Executor.shutdown()
- 🟡 **concurrent.futures** ProcessPoolExecutor workers spawned on demand
- 🟡 **curses** get_escdelay(), set_escdelay(), get_tabsize(), set_tabsize() functions
- 🟡 **datetime** date.isocalendar() and datetime.isocalendar() return named tuples
- 🟡 **fcntl** F_OFD_GETLK, F_OFD_SETLK, F_OFD_SETLKW constants added
- 🟡 **ftplib, imaplib, nntplib, poplib, smtplib** Raise ValueError for zero timeout
- 🟡 **ftplib** Default encoding changed from Latin-1 to UTF-8 (RFC 2640)
- 🟡 **gc** gc.is_finalized() to check if object finalized
- 🟡 **gc** Garbage collection no longer blocks on resurrected objects
- 🟡 **hashlib** SHA3 and SHAKE support from OpenSSL when available
- 🟡 **http** Added status codes 103 EARLY_HINTS, 418 IM_A_TEAPOT, 425 TOO_EARLY
- 🟡 **imaplib** timeout parameter for IMAP4 and IMAP4_SSL constructors
- 🟡 **imaplib** imaplib.IMAP4.unselect() method added
- 🟡 **importlib** importlib.util.resolve_name() raises ImportError instead of ValueError
- 🟡 **importlib.resources** importlib.resources.files() with subdirectory support
- 🟡 **inspect** inspect.BoundArguments.arguments changed from OrderedDict to dict
- 🟡 **ipaddress** IPv6 Scoped Addresses support with scope_id attribute
- 🟡 **ipaddress** No longer accepts leading zeros in IPv4 addresses (since 3.9.5, security)
- 🟡 **math** math.gcd() now accepts multiple arguments
- 🟡 **math** math.lcm() for least common multiple
- 🟡 **math** math.nextafter() for next floating-point value
- 🟡 **math** math.ulp() for unit in last place of float
- 🟡 **multiprocessing** multiprocessing.SimpleQueue.close() method added
- 🟡 **os** os.pidfd_open() and os.P_PIDFD for Linux process management with file descriptors
- 🟡 **os** os.unsetenv() now available on Windows
- 🟡 **os** os.putenv() and os.unsetenv() always available
- 🟡 **os** os.waitstatus_to_exitcode() to convert wait status to exit code
- 🟡 **os** CLD_KILLED and CLD_STOPPED constants added
- 🟡 **pathlib** pathlib.Path.readlink() method added
- 🟡 **pdb** pdb now supports ~/.pdbrc on Windows
- 🟡 **pprint** Can now pretty-print types.SimpleNamespace
- 🟡 **pydoc** Shows __doc__ for any object with that attribute
- 🟡 **random** random.Random.randbytes() method for generating random bytes
- 🟡 **signal** signal.pidfd_send_signal() for Linux signal sending via file descriptors
- 🟡 **socket** CAN_RAW_JOIN_FILTERS constant on Linux 4.1+
- 🟡 **socket** CAN_J1939 protocol support
- 🟡 **socket** socket.send_fds() and socket.recv_fds() functions
- 🟡 **sys** sys.platlibdir attribute for platform-specific library directory
- 🟡 **sys** sys.stderr now line-buffered by default (was block-buffered when non-interactive)
- 🟡 **time** time.thread_time() on AIX uses thread_cputime() with nanosecond resolution
- 🟡 **tracemalloc** tracemalloc.reset_peak() to reset peak memory tracking
- 🟡 **unicodedata** Updated to Unicode 13.0.0
- 🟡 **venv** Activation scripts consistently use __VENV_PROMPT__
- 🟡 **xml.etree.ElementTree** White space in attributes now preserved when serializing
- 🟢 **typing** Improved help() for typing module special forms and generic aliases

## Improvements

### Performance

- 🟡 **Performance** Comprehension temporary variable assignment optimized (nearly as fast as simple assignment)
- 🟡 **Performance** Signal handling in multithreaded apps optimized
- 🟡 **Performance** Builtins (range, tuple, set, frozenset, list, dict) sped up using vectorcall (PEP 590)
- 🟡 **Performance** PyLong_FromDouble() up to 1.87x faster
- 🟡 **Performance** set.difference_update() optimized for large other sets
- 🟡 **Performance** Floor division of floats improved
- 🟡 **Performance** Short ASCII string decoding ~15% faster
- 🟡 **Performance** Small object allocator allows one empty arena for reuse
- 🟡 **Performance** subprocess module optimized on FreeBSD using closefrom()

### Error Messages

- 🟡 **Error Messages** Better handling of encoding/errors arguments in Development Mode
- 🟡 **Error Messages** Improved help for typing module
- 🟡 **Error Messages** Better traceback location for assert statement failures

### Language Changes

- 🟡 **Language** __file__ in __main__ module now absolute path
- 🟡 **Language** __import__() raises ImportError instead of ValueError for bad relative imports
- 🟡 **Language** "".replace("", s, n) now returns s for non-zero n
- 🟡 **Language** Parallel aclose()/asend()/athrow() on async generators now prohibited
- 🟡 **Language** Errors in __iter__ no longer masked by TypeError in 'in' operator
- 🟡 **Language** Improved help for typing module with docstrings for special forms
- 🟡 **IDLE** Toggle cursor blink option added
- 🟡 **IDLE** Escape key closes completion windows
- 🟡 **IDLE** Settings dialog rearranged and improved

### Multiphase Initialization

- 🟢 **Modules** Multiple modules now use multiphase initialization (PEP 489)
  - _abc, audioop, _bz2, _codecs, _contextvars, _crypt, _functools, _json, _locale, math, operator, resource, time, _weakref

### Stable ABI

- 🟢 **Modules** Multiple modules now use stable ABI (PEP 384)
  - audioop, ast, grp, _hashlib, pwd, _posixsubprocess, random, select, struct, termios, zlib

## Implementation Details

### Parser

- 🔴 **Parser** New PEG parser (PEP 617) replaces LL(1) parser
  - More flexible for future language features
  - Performance comparable to old parser
  - Old parser available via -X oldparser or PYTHONOLDPARSER=1 (removed in 3.10)

### CPython Bytecode

- 🟢 **Bytecode** LOAD_ASSERTION_ERROR opcode added for assert statement
- 🟢 **Bytecode** COMPARE_OP split into four instructions
  - COMPARE_OP (rich comparisons)
  - IS_OP ('is' and 'is not')
  - CONTAINS_OP ('in' and 'not in')
  - JUMP_IF_NOT_EXC_MATCH (exception matching in try-except)

### C API

- 🟡 **C API** PEP 573: Module state from extension type methods
  - PyType_FromModuleAndSpec(), PyType_GetModule(), PyType_GetModuleState()
  - PyCMethod and METH_METHOD for accessing defining class

- 🟡 **C API** Frame and thread state access functions
  - PyFrame_GetCode(), PyFrame_GetBack(), PyFrame_GetLineNumber()
  - PyThreadState_GetInterpreter(), PyThreadState_GetFrame(), PyThreadState_GetID()
  - PyInterpreterState_Get()

- 🟡 **C API** PyObject_CallNoArgs() for efficient no-argument calls
- 🟡 **C API** PyObject_CallOneArg() for single positional argument calls
- 🟡 **C API** Py_EnterRecursiveCall() and Py_LeaveRecursiveCall() now regular functions in limited API
- 🟡 **C API** PyModule_AddType() helper function added
- 🟡 **C API** PyObject_GC_IsTracked() and PyObject_GC_IsFinalized() added
- 🟡 **C API** PyInterpreterState.eval_frame (PEP 523) requires new tstate parameter
- 🟡 **C API** Py_AddPendingCall() now per-subinterpreter
- 🟡 **C API** PyStructSequence_UnnamedField now constant
- 🟡 **C API** PyGC_Head structure now opaque (internal C API only)
- 🟡 **C API** Py_FatalError() macro logs function name automatically
- 🟡 **C API** Vectorcall protocol requires string keyword names only
- 🟡 **C API** Multiple macros converted to functions (PyObject_IS_GC, PyObject_CheckBuffer, PyIndex_Check)
- 🟡 **C API** Heap-allocated type instances must visit their type in tp_traverse
- 🟡 **C API** PyEval_Call* functions deprecated - Use PyObject_Call variants

- 🟢 **C API** Py_UNICODE_COPY, Py_UNICODE_FILL, PyUnicode_WSTR_LENGTH deprecated
- 🟢 **C API** PyUnicode_FromUnicode(), PyUnicode_AsUnicode() marked deprecated
- 🟢 **C API** Removed PyFPE_START_PROTECT and PyFPE_END_PROTECT from limited API
- 🟢 **C API** Removed various free list clearing functions
- 🟢 **C API** Removed Py_UNICODE_MATCH (broken since 3.3)
- 🟢 **C API** Many internal symbols removed from public headers

## Platform & Environment

- 🟡 **Build** --with-platlibdir configure option added
- 🟡 **Build** setenv() and unsetenv() now required on non-Windows platforms
- 🟡 **Build** COUNT_ALLOCS special build macro removed
- 🟡 **Build** Python can be built for Windows 10 ARM64
- 🟡 **Build** _tkinter on macOS links with non-system Tcl/Tk frameworks in /Library/Frameworks
- 🟡 **Build** PGO build skips some slow tests for faster compilation
- 🟡 **Platform** Windows registry not used for sys.path when -E option used
- 🟢 **Build** bdist_wininst installers officially unsupported on non-Windows

## Release Process & Meta Changes

- 🟡 **Release** Python adopts annual release cycle (PEP 602)
  - New feature versions released yearly in October
  - Predictable schedule for planning upgrades

## Porting Notes

### API Changes

- 🟡 **API** __import__() and importlib.util.resolve_name() raise ImportError not ValueError
- 🟡 **API** venv activation no longer special-cases __VENV_PROMPT__ == ""
- 🟡 **API** select.epoll.unregister() no longer ignores EBADF
- 🟡 **API** bz2.BZ2File compresslevel parameter now keyword-only
- 🟡 **API** AST subscript simplification: Index(value) → value, ExtSlice(slices) → Tuple(slices, Load())
- 🟡 **API** importlib ignores PYTHONCASEOK when -E or -I options used
- 🟡 **API** ftplib.FTP and FTP_TLS encoding parameter added, default changed to UTF-8
- 🟡 **API** AbstractEventLoop must implement shutdown_default_executor()
- 🟡 **API** __future__ module constant values updated to prevent collision with compiler flags
- 🟡 **API** array('u') now uses wchar_t as C type
- 🟡 **API** logging.getLogger('root') returns root logger, not logger named 'root'
- 🟡 **API** pathlib.PurePath division returns NotImplemented for incompatible types
- 🟡 **API** codecs.lookup() normalizes encoding names consistently

### Notable Patch Release Changes

#### Python 3.9.1
- 🟡 **typing** typing.Literal behavior changed to match PEP 586
  - De-duplicates parameters, order-independent equality, respects types
- 🟡 **macOS** Full support for macOS 11.0 Big Sur and Apple Silicon (ARM64)
  - New universal2 build variant for ARM64 and Intel 64

#### Python 3.9.2
- 🟡 **collections.abc** collections.abc.Callable now flattens type parameters like typing.Callable
- 🟡 **urllib.parse** Changed to allow only '&' as query parameter separator (security, W3C compliance)

#### Python 3.9.3
- 🟡 **ftplib** FTP.* no longer trusts server's IPv4 address for passive data channel (security)

#### Python 3.9.5
- 🟡 **urllib.parse** Strips ASCII newline/tab characters from URLs (security, WHATWG spec)

#### Python 3.9.14
- 🟡 **Security** int/str conversion length limit added to prevent DoS (CVE-2020-10735)
  - Default limit: 4300 digits in string form
  - Configurable via environment variable, command line, or sys APIs

#### Python 3.9.17
- 🟡 **tarfile** Extraction filter argument added (PEP 706, security)
  - Limits dangerous tar features (files outside destination, etc.)
  - Will show DeprecationWarning in Python 3.12, default to 'data' in 3.14
