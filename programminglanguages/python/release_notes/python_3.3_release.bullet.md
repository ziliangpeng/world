# Python 3.3 Release Notes

**Released:** September 29, 2012
**EOL:** September 2017 (already reached)

## Major Highlights

Python 3.3 modernizes the language with better Unicode handling, virtual environments in the standard library, and improved migration from Python 2:

1. **yield from expression (PEP 380)** - Delegate generator operations to subgenerators cleanly
2. **venv module (PEP 405)** - Virtual environments built into the standard library
3. **Flexible string representation (PEP 393)** - 2-3x memory reduction for Unicode strings with variable-width encoding
4. **Implicit namespace packages (PEP 420)** - No __init__.py required for namespace packages
5. **Reworked exception hierarchy (PEP 3151)** - New specific exception types like FileNotFoundError, PermissionError
6. **Import system rewrite** - Import machinery now based on importlib with per-module locks
7. **u'unicode' literals return (PEP 414)** - Eases Python 2 to 3 migration

## Breaking Changes

- 🟡 **OS/Platform** OS/2 and VMS no longer supported due to lack of maintainer
- 🟡 **OS/Platform** Windows 2000 and platforms with command.com no longer supported
- 🟡 **OS/Platform** OSF support completely removed (deprecated in 3.2)
- 🟡 **Syntax** Null bytes in source code now raise SyntaxError
- 🟡 **pydoc** Tk GUI and serve() function removed (deprecated in 3.2)
- 🟡 **contextlib** contextlib.nested removed - Use contextlib.ExitStack instead
- 🟢 **Build** --with-wide-unicode configure flag removed (Python now always behaves like wide build)

## Deprecations

- 🟡 **abc** abc.abstractproperty deprecated - Use property with abc.abstractmethod()
- 🟡 **abc** abc.abstractclassmethod deprecated - Use classmethod with abc.abstractmethod()
- 🟡 **abc** abc.abstractstaticmethod deprecated - Use staticmethod with abc.abstractmethod()
- 🟡 **codecs** unicode_internal codec deprecated - Use UTF-8, UTF-16, or UTF-32
- 🟡 **ftplib** ftplib.FTP.nlst() and dir() deprecated - Use ftplib.FTP.mlsd()
- 🟡 **platform** platform.popen() deprecated - Use subprocess module
- 🟡 **os** Windows bytes API deprecated - Use Unicode filenames instead
- 🟡 **xml.etree** xml.etree.cElementTree deprecated - Accelerator used automatically
- 🟡 **time** time.clock() deprecated - Use time.perf_counter() or time.process_time() instead
- 🟡 **os** os.stat_float_times() deprecated
- 🟡 **importlib** importlib.abc.SourceLoader.path_mtime() deprecated - Use path_stats() instead
- 🟡 **builtins** Passing non-empty string to object.__format__() deprecated (will raise TypeError in 3.4)
- 🟢 **C API** Py_UNICODE type and related functions deprecated by PEP 393 (will be removed in Python 4)

## New Features

### Language Syntax

- 🔴 **Syntax** yield from expression for generator delegation (PEP 380) - Allows subgenerators to receive sent/thrown values
- 🟡 **Syntax** u'unicode' literal prefix accepted again for easier Python 2 migration (PEP 414)
- 🟡 **Syntax** raise ... from None suppresses exception context (PEP 409) - Cleaner error messages
- 🟡 **Syntax** Raw bytes literals can be written as rb"..." or br"..."
- 🟡 **Syntax** __qualname__ attribute for functions and classes shows qualified path (PEP 3155)

### Exception Handling

- 🔴 **Exceptions** Unified OS exception hierarchy (PEP 3151) - OSError, IOError, EnvironmentError, WindowsError are now one type
- 🔴 **Exceptions** New specific exception subclasses: FileNotFoundError, PermissionError, FileExistsError, IsADirectoryError, NotADirectoryError
- 🟡 **Exceptions** ConnectionError with subclasses: BrokenPipeError, ConnectionAbortedError, ConnectionRefusedError, ConnectionResetError
- 🟡 **Exceptions** Additional exception types: BlockingIOError, ChildProcessError, InterruptedError, ProcessLookupError, TimeoutError

### Import System

- 🔴 **importlib** Import system rewritten using importlib (completes PEP 302 phase 2)
- 🟡 **Import** Implicit namespace packages (PEP 420) - No __init__.py required for namespace packages
- 🟡 **Import** Per-module import locks replace global import lock - Prevents deadlocks
- 🟡 **Import** All modules have __loader__ attribute
- 🟡 **Import** ImportError has name and path attributes
- 🟡 **Import** New ABCs: importlib.abc.MetaPathFinder, importlib.abc.PathEntryFinder, importlib.abc.FileLoader
- 🟡 **Import** Exposed loaders: SourceFileLoader, SourcelessFileLoader, ExtensionFileLoader
- 🟡 **Import** importlib.machinery.FileFinder now exposed
- 🟡 **Import** importlib.invalidate_caches() function added

### Unicode and Strings

- 🔴 **Unicode** Flexible string representation (PEP 393) - Variable-width encoding: 1/2/4 bytes per character
- 🔴 **Unicode** Python now always supports full Unicode range U+0000 to U+10FFFF - No narrow/wide build distinction
- 🔴 **Unicode** sys.maxunicode always 1114111 (0x10FFFF)
- 🟡 **Unicode** All functions now correctly handle non-BMP characters
- 🟡 **Unicode** len() always returns 1 for non-BMP characters
- 🟡 **str** New str.casefold() method for caseless string matching
- 🟡 **unicodedata** Unicode database updated to UCD 6.1.0
- 🟡 **unicodedata** Support for Unicode name aliases and named sequences

### Built-in Types and Functions

- 🟡 **open** open() gets opener parameter for custom file descriptor handling
- 🟡 **open** New 'x' mode for exclusive creation (fails if file exists)
- 🟡 **print** print() gets flush keyword argument
- 🟡 **hash** Hash randomization enabled by default for security (PYTHONHASHSEED env var)
- 🟡 **list, bytearray** New copy() and clear() methods
- 🟡 **collections.abc.MutableSequence** Now defines clear() method
- 🟡 **dict** dict.setdefault() now does only one lookup (atomic with built-in types)
- 🟡 **range** range() equality comparisons now compare underlying sequences
- 🟡 **bytes, bytearray** count/find/rfind/index/rindex accept integers 0-255 as first argument
- 🟡 **bytes, bytearray** rjust/ljust/center accept bytearray for fill argument

### New Modules

- 🟡 **faulthandler** Debug module for dumping tracebacks on crashes/timeouts/signals
- 🟡 **ipaddress** Tools for IPv4/IPv6 addresses, networks, and interfaces (PEP 3144)
- 🟡 **lzma** Data compression using LZMA algorithm (.xz and .lzma files)
- 🟡 **unittest.mock** Mock object library (previously external package)
- 🔴 **venv** Virtual environments in standard library (PEP 405)

### Virtual Environments

- 🔴 **venv** venv module for programmatic virtual environment creation (PEP 405)
- 🔴 **venv** pyvenv command-line tool for virtual environment management
- 🟡 **venv** Python interpreter checks for pyvenv.cfg file to detect virtual environments

### Introspection

- 🟡 **inspect** inspect.signature() for easy callable introspection (PEP 362)
- 🟡 **inspect** New classes: Signature, Parameter, BoundArguments
- 🟡 **sys** sys.implementation attribute with name, version, hexversion, cache_tag (PEP 421)
- 🟡 **types** types.SimpleNamespace for attribute-based writable namespaces
- 🟡 **sys** sys.thread_info named tuple with thread implementation info

### Standard Library Enhancements

- 🟡 **collections** New ChainMap class for treating multiple mappings as single unit
- 🟡 **collections** Abstract base classes moved to collections.abc module (aliases remain)
- 🟡 **collections** Counter supports unary +/- and in-place operators (+=, -=, |=, &=)
- 🟡 **contextlib** New ExitStack for programmatic context manager manipulation
- 🟡 **bz2** Complete rewrite with bz2.open(), support for file-like objects, multi-stream decompression
- 🟡 **bz2** BZ2File implements full io.BufferedIOBase API
- 🟡 **array** array module supports long long type (q and Q codes)
- 🟡 **base64** ASCII-only Unicode strings accepted by decoding functions
- 🟡 **binascii** a2b_ functions accept ASCII-only strings
- 🟡 **codecs** mbcs codec rewritten to handle all error handlers on Windows
- 🟡 **codecs** New cp65001 codec (Windows UTF-8, CP_UTF8)
- 🟡 **codecs** Multibyte CJK decoders resynchronize faster
- 🟡 **codecs** Incremental CJK encoders no longer reset at each call
- 🟡 **crypt** Addition of salt, modular crypt format, and mksalt() function
- 🟡 **pickle** Pickler.dispatch_table attribute for per-pickler reduction functions
- 🟡 **re** str regular expressions support \u and \U escapes
- 🟡 **sched** scheduler.run() accepts blocking parameter for non-blocking use
- 🟡 **sched** scheduler class thread-safe and constructor parameters now optional
- 🟡 **select** New select.devpoll class for Solaris high-performance async sockets
- 🟡 **shlex** shlex.quote() function moved from pipes module and documented
- 🟡 **shutil** New disk_usage(), chown(), get_terminal_size() functions
- 🟡 **shutil** copy2() and copystat() preserve nanosecond timestamps and extended attributes
- 🟡 **shutil** Several functions accept symlinks argument
- 🟡 **shutil** move() handles symlinks correctly and returns dst
- 🟡 **shutil** rmtree() resistant to symlink attacks
- 🟡 **signal** New functions: pthread_sigmask(), pthread_kill(), sigpending(), sigwait(), sigwaitinfo(), sigtimedwait()
- 🟡 **signal** Signal handler writes signal number as byte into wakeup file descriptor
- 🟡 **smtpd** RFC 5321 (extended SMTP) and RFC 1870 (size extension) support
- 🟡 **smtplib** SMTP/SMTP_SSL/LMTP accept source_address parameter
- 🟡 **smtplib** SMTP supports context management protocol
- 🟡 **smtplib** SMTP_SSL and starttls() accept SSLContext parameter
- 🟡 **socket** New methods: sendmsg(), recvmsg(), recvmsg_into()
- 🟡 **socket** PF_CAN protocol family support on Linux
- 🟡 **socket** PF_RDS protocol family support
- 🟡 **socket** PF_SYSTEM protocol family support on OS X
- 🟡 **socket** sethostname() function for setting hostname on Unix
- 🟡 **socketserver** BaseServer.service_actions() method for cleanup in service loop
- 🟡 **sqlite3** Connection.set_trace_callback() for tracing SQL commands
- 🟡 **ssl** New RAND_bytes() and RAND_pseudo_bytes() functions
- 🟡 **ssl** Finer-grained exception hierarchy
- 🟡 **ssl** load_cert_chain() accepts password argument
- 🟡 **ssl** Diffie-Hellman key exchange support (load_dh_params(), set_ecdh_curve())
- 🟡 **ssl** SSLSocket.get_channel_binding() for SCRAM-SHA-1-PLUS authentication
- 🟡 **ssl** Query SSL compression algorithm (compression() method, OP_NO_COMPRESSION)
- 🟡 **ssl** Next Protocol Negotiation extension support
- 🟡 **ssl** SSLError.library and SSLError.reason attributes
- 🟡 **ssl** get_server_certificate() supports IPv6
- 🟡 **ssl** OP_CIPHER_SERVER_PREFERENCE attribute
- 🟡 **stat** stat.filemode() converts file mode to string form
- 🟡 **struct** struct module supports ssize_t and size_t (n and N codes)
- 🟡 **subprocess** Command strings can be bytes objects on POSIX
- 🟡 **subprocess** DEVNULL constant for platform-independent output suppression
- 🟡 **tarfile** tarfile supports lzma encoding
- 🟡 **tempfile** SpooledTemporaryFile.truncate() accepts size parameter
- 🟡 **textwrap** textwrap.indent() for adding common prefix to lines
- 🟡 **threading** Condition/Semaphore/BoundedSemaphore/Event/Timer now classes (were factory functions)
- 🟡 **threading** Thread constructor accepts daemon keyword argument
- 🟡 **threading** threading.get_ident() now public (was _thread.get_ident)
- 🟡 **time** New time functions (PEP 418): monotonic(), perf_counter(), process_time(), get_clock_info()
- 🟡 **time** New clock_getres(), clock_gettime(), clock_settime() functions
- 🟡 **time** sleep() raises ValueError for negative values (consistent across platforms)
- 🟡 **types** types.MappingProxyType read-only proxy of mapping
- 🟡 **types** types.new_class() and types.prepare_class() for dynamic type creation (PEP 3115)
- 🟡 **unittest** assertRaises/assertRaisesRegex/assertWarns/assertWarnsRegex accept msg keyword
- 🟡 **unittest** TestCase.run() returns TestResult object
- 🟡 **urllib** Request accepts method argument for HTTP method selection
- 🟡 **webbrowser** Support for Chrome, xdg-open, gvfs-open
- 🟡 **xml.etree.ElementTree** C accelerator imported by default
- 🟡 **xml.etree.ElementTree** Element.iter() family rewritten in C
- 🟡 **zlib** Decompress.eof attribute distinguishes complete vs incomplete streams
- 🟡 **zlib** ZLIB_RUNTIME_VERSION reports underlying zlib library version

### Windows Improvements

- 🟡 **Platform** Python Launcher for Windows (PEP 397) - py command for version-independent launching
- 🟡 **Platform** Windows installer option to add Python to system PATH
- 🟡 **Platform** Windows launcher supports Unix-style shebang lines
- 🟡 **Platform** Windows launcher supports version selection (py -3, py -2.6)

## Improvements

### Performance

- 🔴 **Performance** Flexible string representation (PEP 393): 2-3x memory reduction for Unicode strings
- 🔴 **Performance** UTF-8 encoding 2-4x faster, UTF-16 encoding up to 10x faster
- 🔴 **decimal** New C accelerator module for decimal (dramatic speedup)
- 🟡 **Performance** ASCII string operations much faster (substring 4x, repeating 4x)
- 🟡 **Performance** UTF-8 encoder optimized
- 🟡 **Performance** Encoding ASCII to UTF-8 shares representation (zero copy)
- 🟡 **Performance** Key-sharing dictionaries (PEP 412) reduce object attribute memory
- 🟡 **Performance** dict.setdefault() atomic with single lookup

### Error Messages

- 🟡 **Error Messages** Function call signature mismatch errors significantly improved

### Memory

- 🔴 **Memory** Unicode strings use 1/2/4 bytes per character depending on content (PEP 393)
- 🟡 **Memory** Object attribute dictionaries share keys between instances (PEP 412)

### Security

- 🟡 **Security** Hash randomization enabled by default (protects against DoS attacks)

## Implementation Details

### PEP 3118: memoryview

- 🟡 **memoryview** Complete rewrite fixing ownership and lifetime issues
- 🟡 **memoryview** All native single character format specifiers supported
- 🟡 **memoryview** cast() method for changing format/shape of C-contiguous arrays
- 🟡 **memoryview** Multi-dimensional list representations and comparisons
- 🟡 **memoryview** 1D memoryviews of hashable types (B, b, c) are hashable
- 🟡 **memoryview** Arbitrary slicing of 1D arrays (including negative step)
- 🟢 **memoryview** Maximum dimensions limited to 64
- 🟢 **memoryview** Empty shape/strides/suboffsets now empty tuple (not None)
- 🟢 **memoryview** Format 'B' returns integer (was bytes object)

### C API

- 🟡 **C API** New PEP 3118 function: PyMemoryView_FromMemory()
- 🟡 **C API** PEP 393 Unicode types: Py_UCS1, Py_UCS2, Py_UCS4
- 🟡 **C API** PEP 393 structures: PyASCIIObject, PyCompactUnicodeObject
- 🟡 **C API** PEP 393 high-level functions: PyUnicode_CopyCharacters(), PyUnicode_FindChar(), PyUnicode_GetLength()
- 🟡 **C API** PEP 393 creation: PyUnicode_New(), PyUnicode_FromKindAndData()
- 🟡 **C API** PEP 393 access: PyUnicode_ReadChar(), PyUnicode_WriteChar(), PyUnicode_Substring()
- 🟡 **C API** PEP 393 low-level: PyUnicode_READY, PyUnicode_KIND, PyUnicode_DATA, PyUnicode_READ, PyUnicode_WRITE
- 🟡 **C API** PyArg_ParseTuple accepts bytearray for 'c' format

### Platform & Environment

- 🟡 **Environment** PYTHONFAULTHANDLER enables faulthandler at startup
- 🟡 **Environment** -X faulthandler command-line option
- 🟡 **Environment** PYTHONHASHSEED controls hash randomization
