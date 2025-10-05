# Python 3.13 Release Notes

**Released:** October 7, 2024
**EOL:** October 2029 (security support - 2 years full support + 3 years security fixes per PEP 602)

## Major Highlights

Python 3.13 is a groundbreaking release with two experimental game-changers and major quality-of-life improvements:

1. **Free-threaded mode (PEP 703)** - Run Python without the GIL for true parallelism (experimental)
2. **JIT compiler (PEP 744)** - Experimental just-in-time compilation for performance improvements
3. **Improved REPL** - Modern interactive interpreter with colors, multiline editing, and history browsing
4. **Defined locals() semantics (PEP 667)** - Clear mutation behavior for debugging and introspection
5. **Removed 19 "dead batteries" modules (PEP 594)** - Cleanup of legacy standard library modules
6. **iOS and Android tier 3 support** - Official mobile platform support
7. **Extended support timeline (PEP 602)** - 2 years full support + 3 years security fixes (up from 1.5 + 3.5)

## Experimental Features

- 🔴 **Interpreter** Experimental free-threaded mode (PEP 703) - Run without GIL for true parallelism
  - Enable: Build with `--disable-gil` or use pre-built free-threaded binaries (`python3.13t`)
  - Runtime control: `PYTHON_GIL=0` or `-X gil=0` to disable GIL
  - Status: Experimental with substantial single-threaded performance hit, may have compatibility issues with C extensions
  - C extensions need `Py_mod_gil` slot or `PyUnstable_Module_SetGIL()` to indicate GIL support
  - Check with: `sys._is_gil_enabled()` or `python -VV` output

- 🟡 **Interpreter** Experimental JIT compiler (PEP 744) - Copy-and-patch JIT for potential speedups
  - Enable: Build with `--enable-experimental-jit` (disabled by default)
  - Runtime control: `PYTHON_JIT=1` to enable, `PYTHON_JIT=0` to disable
  - Status: Modest performance improvements currently, expect improvements in future releases
  - Uses Tier 2 IR optimization pipeline with machine code generation

## Breaking Changes

- 🔴 **stdlib** Removed 19 legacy "dead batteries" modules (PEP 594) - aifc, audioop, cgi, cgitb, chunk, crypt, imghdr, mailcap, msilib, nis, nntplib, ossaudiodev, pipes, sndhdr, spwd, sunau, telnetlib, uu, xdrlib
- 🔴 **2to3** Removed 2to3 tool and lib2to3 module - Use modern code migration tools instead
- 🟡 **tkinter.tix** Removed tkinter.tix module
- 🟡 **locale** Removed locale.resetlocale() function
- 🟡 **typing** Removed typing.io and typing.re namespaces
- 🟡 **builtins** Removed chained classmethod descriptors - Use __wrapped__ attribute instead
- 🟡 **configparser** Removed LegacyInterpolation class
- 🟡 **locale** Removed locale.getdefaultlocale() function - Use getlocale(), setlocale(), getencoding() instead
- 🟡 **tkinter** Removed Tkinter.Misc.tk_menuBar() method
- 🟡 **turtle** Removed turtle.RawTurtle.settiltangle() method
- 🟡 **typing** Removed typing.TypeAlias annotation (use `type` statement)
- 🟡 **unittest** Removed TestProgram.usageExit() method
- 🟡 **webbrowser** Removed MacOSX browser support

## Deprecations

### Removing in Python 3.14

- 🟡 **argparse** ArgumentParser.add_argument() parameters type, choices, metavar for BooleanOptionalAction
- 🟡 **ast** ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Ellipsis - Use ast.Constant instead
- 🟡 **asyncio** Child watcher classes (MultiLoopChildWatcher, FastChildWatcher, AbstractChildWatcher, SafeChildWatcher)
- 🟡 **asyncio** get_event_loop() emits DeprecationWarning when no event loop set
- 🟡 **asyncio** Child watcher getter/setter methods
- 🟡 **email** email.utils.localtime() isdst parameter
- 🟡 **importlib.abc** ResourceReader, Traversable, TraversableResources - Use importlib.resources.abc instead
- 🟡 **itertools** Undocumented support for copy, deepcopy, pickle operations
- 🟡 **multiprocessing** Default start method will change from 'fork' to safer option on Linux/BSDs
- 🟡 **pathlib** is_relative_to() and relative_to() passing additional arguments
- 🟡 **pkgutil** find_loader() and get_loader() - Use importlib.util.find_spec() instead
- 🟡 **pty** master_open(), slave_open() - Use pty.openpty()
- 🟡 **sqlite3** version and version_info attributes
- 🟡 **sqlite3** execute()/executemany() with named placeholders and sequence parameters
- 🟡 **urllib** urllib.parse.Quoter class

### Removing in Python 3.15

- 🟡 **import** Setting __cached__ without setting __spec__.cached
- 🟡 **import** Setting __package__ without setting __spec__.parent
- 🟡 **ctypes** Undocumented ctypes.SetPointerType() function
- 🟡 **http.server** CGIHTTPRequestHandler class and --cgi flag
- 🟡 **importlib** load_module() method - Use exec_module() instead
- 🟡 **locale** getdefaultlocale() - Use getlocale(), setlocale(), getencoding()
- 🟡 **pathlib** PurePath.is_reserved() - Use os.path.isreserved() on Windows
- 🟡 **platform** java_ver() function
- 🟡 **sysconfig** is_python_build() check_home argument
- 🟡 **threading** RLock() will take no arguments
- 🟡 **types** CodeType.co_lnotab attribute (deprecated in PEP 626)
- 🟡 **typing** NamedTuple keyword argument syntax
- 🟡 **typing** TypedDict functional syntax without fields parameter
- 🟡 **typing** no_type_check_decorator() function
- 🟡 **wave** getmark(), setmark(), getmarkers() methods

### Removing in Python 3.16

- 🟡 **import** Setting __loader__ without setting __spec__.loader
- 🟡 **array** 'u' format code (wchar_t) - Use 'w' format code (Py_UCS4) instead
- 🟡 **asyncio** asyncio.iscoroutinefunction() - Use inspect.iscoroutinefunction()
- 🟡 **builtins** Bitwise inversion on booleans (~True, ~False) - Use `not x` for logical negation
- 🟡 **shutil** ExecError exception (alias of RuntimeError)
- 🟡 **symtable** Class.get_methods() method
- 🟡 **sys** _enablelegacywindowsfsencoding() - Use PYTHONLEGACYWINDOWSFSENCODING env var
- 🟡 **tarfile** TarFile.tarfile attribute

### Removing in Python 3.17

- 🟡 **collections.abc** ByteString - Use collections.abc.Buffer or explicit union types
- 🟡 **typing** typing._UnionGenericAlias private class
- 🟡 **typing** ByteString - Use collections.abc.Buffer or explicit union types

### Pending Removal in Future Versions

- 🟢 **argparse** Nesting argument groups and mutually exclusive groups
- 🟢 **builtins** bool(NotImplemented), generator throw() 3-arg signature, numeric literals followed by keywords
- 🟢 **builtins** __index__/__int__/__float__/__complex__ returning non-standard types
- 🟢 **builtins** Delegation of int() to __trunc__(), passing complex as real/imag to complex()
- 🟢 **calendar** January and February constants - Use JANUARY and FEBRUARY
- 🟢 **codeobject** co_lnotab attribute - Use co_lines() method
- 🟢 **datetime** utcnow() and utcfromtimestamp() - Use timezone-aware alternatives
- 🟢 **gettext** Plural value must be integer
- 🟢 **importlib** cache_from_source() debug_override parameter
- 🟢 **logging** warn() method - Use warning()
- 🟢 **mailbox** StringIO input and text mode
- 🟢 **os** Calling register_at_fork() in multi-threaded process
- 🟢 **re** More strict rules for numerical group references and group names
- 🟢 **shutil** rmtree() onerror parameter - Use onexc
- 🟢 **ssl** SSLContext without protocol, set_npn_protocols(), SSL/TLS protocol versions
- 🟢 **threading** Old camelCase method names (notifyAll, isSet, isDaemon, etc.)
- 🟢 **typing** Text type
- 🟢 **unittest** IsolatedAsyncioTestCase returning non-None from test case
- 🟢 **urllib.parse** Many split functions - Use urlparse()
- 🟢 **urllib.request** URLopener and FancyURLopener
- 🟢 **xml.etree.ElementTree** Testing truth value of Element
- 🟢 **zipimport** zipimporter.load_module() - Use exec_module()

## New Features

### Language & Interpreter

- 🔴 **REPL** New interactive interpreter with multiline editing, color output, history browsing (F2), paste mode (F3), interactive help (F1)
  - Based on PyPy project code
  - Color enabled by default (control with PYTHON_COLORS, NO_COLOR, FORCE_COLOR)
  - Disable with: PYTHON_BASIC_REPL environment variable

- 🔴 **locals()** Defined mutation semantics for locals() (PEP 667)
  - Returns independent snapshots in optimized scopes (functions, generators, comprehensions)
  - frame.f_locals returns write-through proxy in optimized scopes
  - Changes exec()/eval() behavior - need explicit namespace to access changes

- 🔴 **Type System** Type parameter defaults (PEP 696) - TypeVar, ParamSpec, TypeVarTuple now support default values

- 🟡 **Data Model** __static_attributes__ stores class attribute names assigned through self.X
- 🟡 **Data Model** __firstlineno__ records first line number of class definition
- 🟡 **Compiler** Docstrings now have common leading whitespace stripped (~5% bytecode cache size reduction)
- 🟡 **Syntax** Annotation scopes can now contain lambdas and comprehensions
- 🟡 **Syntax** Future statements no longer triggered by relative imports of __future__
- 🟡 **Syntax** global declarations now permitted in except blocks when used in else block
- 🟡 **builtins** exec() and eval() now accept globals and locals as keyword arguments
- 🟡 **builtins** compile() accepts new ast.PyCF_OPTIMIZED_AST flag
- 🟡 **property** Add __name__ attribute to property objects
- 🟡 **PythonFinalizationError** New exception for operations blocked during finalization - used by _thread.start_new_thread(), os.fork(), os.forkpty(), subprocess.Popen
- 🟡 **str** str.replace() count argument can now be keyword argument
- 🟡 **warnings** Many functions now emit warning if boolean value passed as file descriptor
- 🟡 **archive** Added name and mode attributes for compressed file-like objects in bz2, lzma, tarfile, zipfile

### Type Hints & Typing

- 🔴 **typing** ReadOnly for TypedDict (PEP 705) - Mark TypedDict items as read-only for type checkers
- 🔴 **typing** TypeIs for type narrowing (PEP 742) - More intuitive alternative to TypeGuard
- 🟡 **typing** NoDefault sentinel object for typing module parameters
- 🟡 **typing** get_protocol_members() returns Protocol member names
- 🟡 **typing** is_protocol() checks if class is a Protocol
- 🟡 **typing** ClassVar can be nested in Final and vice versa
- 🟡 **warnings** warnings.deprecated() decorator (PEP 702) - Mark deprecations for static type checkers and runtime

### Error Messages

- 🟡 **Error Messages** Tracebacks now colored by default in terminal
- 🟡 **Error Messages** Better error for scripts with same name as stdlib/third-party modules
- 🟡 **Error Messages** Suggests correct keyword argument when incorrect one passed

### Environment & Configuration

- 🟡 **Environment** PYTHON_FROZEN_MODULES controls frozen module import
- 🟡 **Environment** PYTHON_PERF_JIT_SUPPORT for perf profiler without frame pointers
- 🟡 **Environment** PYTHON_HISTORY changes .python_history file location
- 🟡 **Environment** PYTHON_COLORS and NO_COLOR/FORCE_COLOR control color output
- 🟡 **Environment** PYTHON_CPU_COUNT override for os.cpu_count() and os.process_cpu_count()
- 🟡 **Environment** PYTHON_JIT control for experimental JIT compiler
- 🟡 **Environment** PYTHON_GIL control for free-threaded mode GIL behavior

## Improved Modules

### Standard Library

- 🟡 **argparse** Add deprecated parameter to add_argument() and add_parser() for deprecating CLI options
- 🟡 **array** Add 'w' type code (Py_UCS4) for Unicode characters - replaces deprecated 'u'
- 🟡 **array** Register array.array as MutableSequence by implementing clear()
- 🟡 **ast** AST node constructors now stricter - omitted optional fields default to None, errors for invalid arguments (deprecated, will error in 3.15)
- 🟡 **ast** ast.parse() accepts optimize argument for optimized AST
- 🟡 **asyncio** as_completed() returns object that's both async iterator and iterator, yielding original task/future objects
- 🟡 **asyncio** loop.create_unix_server() auto-removes Unix socket when server closed
- 🟡 **asyncio** DatagramTransport.sendto() sends zero-length datagrams
- 🟡 **asyncio** Queue.shutdown() and QueueShutDown for queue termination
- 🟡 **asyncio** Server.close_clients() and Server.abort_clients() forcefully close server
- 🟡 **asyncio** StreamReader.readuntil() accepts tuple of separators
- 🟡 **asyncio** TaskGroup improved external/internal cancellation handling, preserves cancellation count
- 🟡 **asyncio** TaskGroup.create_task() accepts **kwargs for Task constructor
- 🟡 **base64** z85encode() and z85decode() for Z85 data encoding/decoding
- 🟡 **compileall** Default worker count uses os.process_cpu_count() instead of os.cpu_count()
- 🟡 **concurrent.futures** Default worker count uses os.process_cpu_count() instead of os.cpu_count()
- 🟡 **configparser** ConfigParser supports unnamed sections with allow_unnamed_section parameter
- 🟡 **copy** copy.replace() function and __replace__() protocol for creating modified copies - supported by namedtuple, dataclass, datetime, inspect.Signature, types.SimpleNamespace, code objects
- 🟡 **ctypes** Structure objects have _align_ attribute for explicit alignment
- 🟡 **dbm** New dbm.sqlite3 module as default dbm backend
- 🟡 **dbm** Add clear() methods to GDBM and NDBM database objects
- 🟡 **dis** Output shows logical labels for jump targets, not offsets (use -O or show_offsets for offsets)
- 🟡 **dis** get_instructions() cache entries now part of Instruction.cache_info field
- 🟡 **doctest** Output colored by default (control with PYTHON_COLORS)
- 🟡 **doctest** DocTestRunner.run() counts skipped tests, added skips attribute
- 🟡 **email** Headers with embedded newlines now quoted on output
- 🟡 **email** getaddresses() and parseaddr() new strict parameter (default True) for safer parsing (CVE-2023-27043 fix)
- 🟡 **enum** EnumDict made public for subclassing EnumType
- 🟡 **fractions** Fraction objects support format specification mini-language
- 🟡 **glob** translate() converts glob pattern to regex
- 🟡 **importlib.resources** Functions now allow accessing directory trees with multiple positional arguments
- 🟡 **importlib.resources** contents() no longer planned for removal
- 🟡 **io** IOBase finalizer logs close() errors with sys.unraisablehook
- 🟡 **ipaddress** IPv4Address.ipv6_mapped property returns IPv4-mapped IPv6 address
- 🟡 **ipaddress** Fixed is_global and is_private behavior
- 🟡 **itertools** batched() has strict parameter to raise ValueError if final batch too short
- 🟡 **marshal** Add allow_code parameter to prevent code object serialization
- 🟡 **math** fma() performs fused multiply-add with single rounding
- 🟡 **mimetypes** guess_file_type() for filesystem paths - guess_type() with paths now soft deprecated
- 🟡 **mmap** Protected from crashes on Windows with inaccessible mapped memory
- 🟡 **mmap** Add seekable() method and seek() now returns absolute position
- 🟡 **mmap** New trackfd parameter (UNIX) to control file descriptor duplication
- 🟡 **multiprocessing** Default worker count uses os.process_cpu_count() instead of os.cpu_count()
- 🟡 **os** process_cpu_count() gets logical CPU cores usable by calling thread
- 🟡 **os** cpu_count() and process_cpu_count() can be overridden with PYTHON_CPU_COUNT
- 🟡 **os** Low-level interface to Linux timer file descriptors - timerfd_create(), timerfd_settime(), timerfd_settime_ns(), timerfd_gettime(), timerfd_gettime_ns() and related constants
- 🟡 **os** lchmod() and chmod() follow_symlinks now available on Windows
- 🟡 **os** fchmod() and chmod() with file descriptors now available on Windows
- 🟡 **os** mkdir() and makedirs() now support mode=0o700 on Windows for access control (CVE-2024-4030 mitigation)
- 🟡 **os** posix_spawn() accepts None for env argument
- 🟡 **os** posix_spawn() supports POSIX_SPAWN_CLOSEFROM attribute
- 🟡 **os.path** isreserved() checks if path is reserved on Windows
- 🟡 **os.path** isabs() on Windows no longer considers paths starting with single slash as absolute
- 🟡 **os.path** realpath() now resolves MS-DOS style file names even if file not accessible
- 🟡 **pathlib** UnsupportedOperation raised instead of NotImplementedError
- 🟡 **pathlib** Path.from_uri() creates Path from file:/// URIs
- 🟡 **pathlib** PurePath.full_match() matches with shell-style wildcards including recursive **
- 🟡 **pathlib** PurePath.parser class attribute stores os.path implementation
- 🟡 **pathlib** glob() and rglob() accept recurse_symlinks keyword argument
- 🟡 **pathlib** glob() and rglob() with pattern ending in ** now return files and directories
- 🟡 **pathlib** is_file(), is_dir(), owner(), group() accept follow_symlinks argument
- 🟡 **pdb** breakpoint() and set_trace() enter debugger immediately
- 🟡 **pdb** sys.path[0] no longer replaced when sys.flags.safe_path set
- 🟡 **pdb** zipapp now supported as debugging target
- 🟡 **pdb** exceptions command for moving between chained exceptions in post-mortem debugging
- 🟡 **queue** Queue.shutdown() and ShutDown for queue termination
- 🟡 **random** Command-line interface added (python -m random)
- 🟡 **re** Rename re.error to PatternError (re.error kept for compatibility)
- 🟡 **shutil** chown() supports dir_fd and follow_symlinks arguments
- 🟡 **site** .pth files decoded with UTF-8 first, then locale encoding
- 🟡 **sqlite3** ResourceWarning emitted if Connection not closed explicitly
- 🟡 **sqlite3** iterdump() filter parameter for filtering database objects
- 🟡 **ssl** create_default_context() includes VERIFY_X509_PARTIAL_CHAIN and VERIFY_X509_STRICT flags
- 🟡 **statistics** kde() for kernel density estimation
- 🟡 **statistics** kde_random() for sampling from estimated probability density
- 🟡 **subprocess** Uses posix_spawn() in more situations (when close_fds=True with posix_spawn_file_actions_addclosefrom_np support)
- 🟡 **sys** _is_interned() tests if string was interned
- 🟡 **tempfile** mkdtemp() on Windows uses mode=0o700 for access control (CVE-2024-4030 mitigation)
- 🟡 **time** monotonic() on Windows uses QueryPerformanceCounter() (1 microsecond resolution)
- 🟡 **time** time() on Windows uses GetSystemTimePreciseAsFileTime() (1 microsecond resolution)
- 🟡 **tkinter** Add tk_busy_* methods for busy state management
- 🟡 **tkinter** wm_attributes() improvements - accepts attribute names without minus, keyword arguments, return_python_dict parameter
- 🟡 **tkinter** Text.count() can return int with return_ints parameter
- 🟡 **tkinter** Support "vsapi" element type in element_create()
- 🟡 **tkinter** Add after_info(), copy_replace() for PhotoImage, read() and data() methods
- 🟡 **traceback** TracebackException.exc_type_str attribute for string display of exc_type
- 🟡 **traceback** format_exception_only() show_group parameter for recursive ExceptionGroup formatting
- 🟡 **types** SimpleNamespace can take single positional argument (mapping or iterable of key-value pairs)
- 🟡 **unicodedata** Updated to Unicode 15.1.0
- 🟡 **venv** Support for creating SCM ignore files (.gitignore by default) - opt-in via API, opt-out via --without-scm-ignore-files
- 🟡 **xml** Allow controlling Expat >=2.6.0 reparse deferral (CVE-2023-52425) with flush() and Get/SetReparseDeferralEnabled() methods
- 🟡 **xml** iterparse() iterator has close() method for explicit cleanup
- 🟡 **zipimport** Add support for ZIP64 format files

## Improvements

### Performance

- 🔴 **Import** Several stdlib modules have significantly improved import times - typing (~33% faster), email.utils, enum, functools, importlib.metadata, threading
- 🟡 **textwrap** textwrap.indent() ~30% faster for large input
- 🟡 **subprocess** Uses posix_spawn() more often for better performance on FreeBSD/Solaris

## Implementation Details

### CPython Bytecode

- 🟢 **Bytecode** YIELD_VALUE oparg indicates if yield is part of yield-from or await
- 🟢 **Bytecode** RESUME oparg changed to indicate except-depth for generator optimization

### C API

- 🟡 **C API** PyMonitoring C API for generating PEP 669 monitoring events
- 🟡 **C API** PyMutex lightweight mutex (single byte) with PyMutex_Lock() and PyMutex_Unlock()
- 🟡 **C API** PyTime C API for system clocks - PyTime_t, PyTime_Monotonic(), PyTime_Time(), etc.
- 🟡 **C API** Many new functions returning strong references instead of borrowed references
- 🟡 **C API** PyDict_GetItemRef(), PyDict_GetItemStringRef(), PyDict_SetDefaultRef(), PyDict_Pop(), PyDict_PopString()
- 🟡 **C API** PyMapping_GetOptionalItem(), PyMapping_GetOptionalItemString()
- 🟡 **C API** PyObject_GetOptionalAttr(), PyObject_GetOptionalAttrString()
- 🟡 **C API** PyErr_FormatUnraisable() for customized unraisable exception warnings
- 🟡 **C API** PyEval_GetFrameBuiltins(), PyEval_GetFrameGlobals(), PyEval_GetFrameLocals() (PEP 667)
- 🟡 **C API** Py_GetConstant() and Py_GetConstantBorrowed() for constant references
- 🟡 **C API** PyImport_AddModuleRef() returns strong reference
- 🟡 **C API** Py_IsFinalizing() checks if interpreter shutting down
- 🟡 **C API** PyList_GetItemRef() returns strong reference
- 🟡 **C API** PyList_Extend() and PyList_Clear() functions
- 🟡 **C API** PyLong_AsInt() stores result in C int
- 🟡 **C API** PyLong_AsNativeBytes(), PyLong_FromNativeBytes(), PyLong_FromUnsignedNativeBytes()
- 🟡 **C API** PyModule_Add() always steals reference to value
- 🟡 **C API** PyObject_GenericHash() implements default hashing
- 🟡 **C API** Py_HashPointer() hashes raw pointer
- 🟡 **C API** PyObject_VisitManagedDict() and PyObject_ClearManagedDict() for Py_TPFLAGS_MANAGED_DICT
- 🟡 **C API** PyRefTracer_SetTracer() and PyRefTracer_GetTracer() for tracking object creation/destruction
- 🟡 **C API** PySys_AuditTuple() alternative to PySys_Audit() with tuple arguments
- 🟡 **C API** PyThreadState_GetUnchecked() doesn't kill process if NULL
- 🟡 **C API** PyType_GetFullyQualifiedName() and PyType_GetModuleName()
- 🟡 **C API** PyUnicode_EqualToUTF8AndSize() and PyUnicode_EqualToUTF8() for string comparison
- 🟡 **C API** PyWeakref_GetRef() returns strong reference or NULL
- 🟡 **C API** PyObject_HasAttrWithError(), PyObject_HasAttrStringWithError(), PyMapping_HasKeyWithError(), PyMapping_HasKeyStringWithError()
- 🟡 **C API** PyArg_ParseTupleAndKeywords() keywords parameter now char * const * in C
- 🟡 **C API** PyArg_ParseTupleAndKeywords() supports non-ASCII keyword parameter names
- 🟡 **C API** PyUnicode_FromFormat() supports %T, %#T, %N, %#N formats for type names (PEP 737)
- 🟡 **C API** No longer need PY_SSIZE_T_CLEAN macro for # formats
- 🟡 **C API** Limited C API expanded - PyMem_Raw* functions, PySys_Audit functions, PyType_GetModuleByDef()
- 🟡 **C API** Python with --with-trace-refs now supports Limited API
- 🟢 **C API** Various deprecated C APIs with soft deprecations and pending removals
- 🟢 **C API** Removed old buffer protocols (PyObject_CheckReadBuffer, PyObject_AsCharBuffer, etc.)
- 🟢 **C API** Removed various functions deprecated in Python 3.9

### Build System

- 🟡 **Build** JIT compiler requires LLVM at build time (copy-and-patch technique)

## Platform & Environment

- 🔴 **Platform** iOS tier 3 support (PEP 730) - arm64-apple-ios and arm64-apple-ios-simulator
- 🔴 **Platform** Android tier 3 support (PEP 738) - aarch64-linux-android and x86_64-linux-android
- 🟡 **Platform** wasm32-wasi now tier 2 platform
- 🟡 **Platform** wasm32-emscripten no longer officially supported

## Release Process & Meta Changes

- 🟡 **Release** Python 3.13+ gets 2 years full support (up from 1.5 years) + 3 years security fixes (PEP 602)
- 🟡 **Release** Python 3.9-3.12 have 1.5 years full support + 3.5 years security fixes
