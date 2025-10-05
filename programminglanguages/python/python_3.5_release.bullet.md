# Python 3.5 Release Notes

**Released:** September 13, 2015
**EOL:** September 2020 (reached)

## Major Highlights

Python 3.5 is a groundbreaking release for asynchronous programming and scientific computing:

1. **async/await syntax (PEP 492)** - Revolutionary dedicated syntax for coroutines making async code as readable as sync code
2. **Matrix multiplication operator @ (PEP 465)** - Cleaner syntax for scientific computing: `S = (H @ beta - r).T @ inv(H @ V @ H.T)`
3. **Type hints (PEP 484)** - Standard framework for type annotations with new `typing` module
4. **os.scandir() (PEP 471)** - 3-5x faster directory traversal on POSIX, 7-20x on Windows
5. **Unpacking generalizations (PEP 448)** - Multiple `*` and `**` unpacking in calls and literals
6. **OrderedDict 4-100x faster** - Reimplemented in C for dramatic performance improvements
7. **Automatic EINTR retry (PEP 475)** - System calls automatically retry on signal interruption

## Breaking Changes

- 🔴 **Windows** Windows XP no longer supported (PEP 11)
- 🔴 **datetime** datetime.time representing midnight UTC no longer false (was obscure bug-prone behavior)
- 🟡 **Syntax** Generator expressions in function calls now require parentheses: `f((x for x in [1]), *args)` not `f(x for x in [1], *args)`
- 🟡 **ssl** ssl.SSLSocket.send() raises SSLWantReadError/SSLWantWriteError instead of returning 0 on non-blocking sockets
- 🟡 **html.parser** Removed deprecated strict mode, HTMLParser.error(), HTMLParserError exception
- 🟡 **html.parser** convert_charrefs argument now True by default
- 🟡 **email** Removed __version__ attribute from email package
- 🟡 **ftplib** Removed internal Netrc class
- 🟡 **asyncio** Removed JoinableQueue class (use Queue with join()/task_done())
- 🟡 **re** re.split() now warns/errors on patterns that can match empty strings
- 🟡 **http.cookies** Morsel comparison now considers key and value, copy() returns Morsel not dict
- 🟢 **Buffer protocol** Error message format changed: "a bytes-like object is required, not 'type'" instead of "'type' does not support buffer protocol"

## Deprecations

### Soft Keywords - Becoming Keywords in Python 3.7

- 🔴 **Syntax** async and await are soft keywords - will become full keywords in Python 3.7 (PEP 492)

### Deprecated Modules and Functions

- 🟡 **formatter** Fully deprecated (removal in Python 3.6)
- 🟡 **asyncio** asyncio.async() deprecated - Use ensure_future()
- 🟡 **inspect** inspect.getargspec() deprecated (removal in Python 3.6) - Use signature()
- 🟡 **inspect** inspect.getfullargspec(), getcallargs(), formatargspec() deprecated - Use signature()
- 🟡 **platform** platform.dist() and platform.linux_distribution() deprecated - Use external package
- 🟡 **re** re.LOCALE flag with str patterns deprecated
- 🟡 **re** Unrecognized special sequences with backslash now raise DeprecationWarning (error in Python 3.6)
- 🟡 **string** Passing format_string as keyword to string.Formatter.format() deprecated
- 🟡 **http.cookies** Directly assigning to Morsel.key, value, coded_value deprecated - Use set()
- 🟡 **unittest** use_load_tests argument of TestLoader.loadTestsFromModule() deprecated
- 🟡 **smtpd** decode_data parameter default will change - Specify explicitly to avoid deprecation warning
- 🟢 **inspect** Signature.from_function() and from_builtin() deprecated - Use from_callable()

### Future Behavior Changes

- 🔴 **Generators** StopIteration in generators raises PendingDeprecationWarning (becomes RuntimeError in Python 3.7 per PEP 479)

## New Features

### Language Syntax

- 🔴 **Async** async/await syntax for coroutines (PEP 492)
  - async def declares coroutine functions
  - await suspends execution for awaitable objects
  - async for for asynchronous iteration
  - async with for asynchronous context managers
  - New __await__() protocol for awaitable objects

- 🔴 **Syntax** Matrix multiplication operator @ (PEP 465)
  - Implements __matmul__(), __rmatmul__(), __imatmul__()
  - NumPy 1.10+ supports the new operator

- 🔴 **Syntax** Additional unpacking generalizations (PEP 448)
  - Multiple * and ** unpacking in function calls
  - Multiple unpacking in tuple/list/set/dict literals
  - Example: `print(*[1], *[2], 3)` and `{'x': 1, **{'y': 2}}`

- 🔴 **bytes** bytes % formatting (PEP 461)
  - % operator now works with bytes and bytearray
  - Supports %b (bytes), %a (ascii repr), numeric formats

### Type System

- 🔴 **typing** New typing module for type hints (PEP 484)
  - Function annotations for parameter and return types
  - Generic types, Union types, Any type
  - Type variables for generic functions/classes
  - No runtime type checking (use external tools)
  - Provisional API status

### Builtins

- 🟡 **bytes** bytes.hex(), bytearray.hex(), memoryview.hex() methods to convert to hex strings
- 🟡 **memoryview** Tuple indexing including multi-dimensional support
- 🟡 **generators** gi_yieldfrom attribute returns object being iterated by yield from
- 🟡 **generators** __name__ and __qualname__ attributes for better introspection
- 🟡 **exceptions** RecursionError exception for maximum recursion depth (subclass of RuntimeError)

### Interpreter & Runtime

- 🔴 **os** os.scandir() function for fast directory iteration (PEP 471)
  - 3-5x faster on POSIX, 7-20x faster on Windows
  - Returns iterator with cached stat information
  - os.walk() reimplemented using scandir()

- 🔴 **System** Automatic EINTR retry (PEP 475)
  - System calls automatically retry when interrupted by signals
  - Affects open(), file I/O, socket ops, select(), time.sleep(), etc.

- 🟡 **Bytecode** .pyo files eliminated (PEP 488)
  - .pyc files now include optimization level: module.cpython-35.opt-1.pyc
  - Multiple optimization levels can coexist

- 🟡 **Extensions** Multi-phase extension module initialization (PEP 489)
  - Two-step loading similar to Python modules
  - Any valid identifier as module name

- 🟡 **C API** Extension modules now include platform tags in filenames
  - Linux: .cpython-35m-x86_64-linux-gnu.so
  - Windows: .cp35-win_amd64.pyd

- 🟡 **locale** POSIX locale uses surrogateescape error handler for stdin/stdout

### Standard Library - New Modules

- 🔴 **typing** Type hints support (PEP 484) - Provisional API
- 🟡 **zipapp** Create executable Python ZIP applications (PEP 441)

### Standard Library - Major Improvements

- 🔴 **collections** OrderedDict reimplemented in C - 4 to 100 times faster
- 🔴 **collections** OrderedDict views support reversed() iteration
- 🔴 **functools** lru_cache() mostly reimplemented in C for significant speedup

- 🟡 **collections** deque.index(), insert(), copy() methods and + * operators
- 🟡 **collections** namedtuple docstrings now writable
- 🟡 **collections** UserString implements __getnewargs__(), __rmod__(), casefold(), format_map(), isprintable(), maketrans()

- 🟡 **collections.abc** Generator, Awaitable, Coroutine, AsyncIterator, AsyncIterable abstract base classes
- 🟡 **collections.abc** Sequence.index() accepts start and stop arguments

- 🟡 **subprocess** subprocess.run() function for streamlined subprocess execution

- 🟡 **traceback** TracebackException, StackSummary, FrameSummary lightweight classes
- 🟡 **traceback** walk_stack() and walk_tb() functions for frame/traceback traversal
- 🟡 **traceback** print_tb() and print_stack() support negative limit

- 🟡 **ssl** Memory BIO support decouples SSL protocol from network I/O
- 🟡 **ssl** SSLv3 disabled by default

- 🟡 **math** math.isclose() and cmath.isclose() for approximate equality testing
- 🟡 **glob** Recursive search with ** pattern
- 🟡 **heapq** heapq.merge() accepts key and reverse parameters

- 🟡 **asyncio** Many improvements (provisional module - backported to 3.4)
  - loop.set_debug() and get_debug() debugging APIs
  - Proactor event loop supports SSL
  - loop.is_closed(), loop.create_task()
  - transport.get_write_buffer_limits()
  - Queue.join() and task_done() methods
  - loop.set_task_factory() and get_task_factory()
  - 3.5.1: ensure_future() accepts awaitable objects, run_coroutine_threadsafe()
  - 3.5.2: loop.create_future(), StreamReader.readuntil()

- 🟡 **compileall** -j N option for parallel bytecode compilation
- 🟡 **compileall** -r option for recursion depth control

- 🟡 **concurrent.futures** Executor.map() accepts chunksize parameter
- 🟡 **concurrent.futures** ThreadPoolExecutor default workers is 5 * CPU count

- 🟡 **configparser** Custom value converters via converters parameter

- 🟡 **contextlib** redirect_stderr() context manager

- 🟡 **csv** writerow() supports arbitrary iterables

- 🟡 **curses** update_lines_cols() function

- 🟡 **difflib** HtmlDiff.make_file() charset parameter, default changed to utf-8
- 🟡 **difflib** diff_bytes() function for comparing byte strings

- 🟡 **distutils** build and build_ext accept -j for parallel building
- 🟡 **distutils** xz compression support via xztar format

- 🟡 **doctest** DocTestSuite() returns empty suite instead of raising ValueError

- 🟡 **email** Policy.mangle_from_ option for "From " line handling
- 🟡 **email** Message.get_content_disposition() method
- 🟡 **email** EmailPolicy.utf8 option for RFC 6532 support

- 🟡 **enum** Enum callable accepts start parameter

- 🟡 **faulthandler** Functions accept file descriptors in addition to file objects

- 🟡 **gzip** GzipFile mode accepts "x" for exclusive creation

- 🟡 **http** HTTPStatus enum for status codes and descriptions

- 🟡 **http.client** RemoteDisconnected exception for unexpected connection closure
- 🟡 **http.client** Automatic reconnection after ConnectionError

- 🟡 **imaplib** Context manager protocol support
- 🟡 **imaplib** RFC 5161 (ENABLE) and RFC 6855 (UTF-8) support via enable()
- 🟡 **imaplib** Automatic UTF-8 encoding for usernames and passwords

- 🟡 **imghdr** OpenEXR and WebP format recognition

- 🟡 **importlib** LazyLoader class for lazy module loading
- 🟡 **importlib** abc.InspectLoader.source_to_code() now static method
- 🟡 **importlib** util.module_from_spec() preferred way to create modules

- 🟡 **inspect** Signature and Parameter are picklable and hashable
- 🟡 **inspect** BoundArguments.apply_defaults() method
- 🟡 **inspect** Signature.from_callable() class method
- 🟡 **inspect** signature() accepts follow_wrapped parameter
- 🟡 **inspect** iscoroutine(), iscoroutinefunction(), isawaitable(), getcoroutinelocals(), getcoroutinestate()
- 🟡 **inspect** stack(), trace(), getouterframes(), getinnerframes() return named tuples

- 🟡 **io** BufferedIOBase.readinto1() method

- 🟡 **ipaddress** IPv4Network and IPv6Network accept (address, netmask) tuple
- 🟡 **ipaddress** reverse_pointer attribute for DNS PTR records

- 🟡 **json** JSONEncoder.sort_keys parameter

- 🟡 **linecache** checkcache() accepts module parameter

- 🟡 **locale** Kazakh kz1048 and Tajik koi8_t codecs

- 🟡 **os** POSIX locale supports "namereplace" and improved "backslashreplace" error handlers

- 🟡 **property** Property docstrings now writable

- 🟡 **shutil** make_archive() passes root_dir to custom archivers

- 🟡 **signal** Signals can be used with with statement

- 🟡 **smtplib** SMTP.auth() method

- 🟡 **socket** CAN_RAW_FD_FRAMES constant on Linux 3.6+

- 🟡 **sysconfig** get_config_var() accepts multiple vars

- 🟡 **tarfile** TarFile.extract() and extractall() accept numeric_owner parameter

- 🟡 **tempfile** TemporaryDirectory context manager

- 🟡 **time** time.monotonic() now always available

- 🟡 **timeit** -u/--unit option to specify time unit
- 🟡 **timeit** timeit() accepts globals parameter

- 🟡 **types** coroutine() function to transform generators into awaitables
- 🟡 **types** CoroutineType for coroutine objects

- 🟡 **unicodedata** Updated to Unicode 8.0.0

- 🟡 **unittest** TestLoader.loadTestsFromModule() accepts pattern parameter
- 🟡 **unittest** TestLoader.errors attribute exposes discovery errors
- 🟡 **unittest** --locals command line option

- 🟡 **unittest.mock** Mock constructor accepts unsafe parameter
- 🟡 **unittest.mock** Mock.assert_not_called() method
- 🟡 **unittest.mock** MagicMock supports __truediv__(), __divmod__(), __matmul__()
- 🟡 **unittest.mock** patch() no longer requires create=True for builtins

- 🟡 **urllib** HTTPPasswordMgrWithPriorAuth class for preemptive auth
- 🟡 **urllib** parse.urlencode() quote_via argument
- 🟡 **urllib** request.urlopen() accepts ssl.SSLContext as context
- 🟡 **urllib** parse.urljoin() updated to RFC 3986 semantics

- 🟡 **wsgiref** headers.Headers constructor headers argument optional

- 🟡 **xmlrpc** client.ServerProxy supports context manager protocol
- 🟡 **xmlrpc** client.ServerProxy accepts ssl.SSLContext

- 🟡 **xml.sax** SAX parsers support character streams
- 🟡 **xml.sax** parseString() accepts str

- 🟡 **zipfile** ZIP output to unseekable streams
- 🟡 **zipfile** ZipFile.open() mode accepts "x" for exclusive creation

- 🟢 **argparse** ArgumentParser.allow_abbrev parameter to disable abbreviations

- 🟢 **bz2** BZ2Decompressor.decompress() accepts max_length

- 🟢 **cgi** FieldStorage supports context manager protocol

- 🟢 **code** InteractiveInterpreter.showtraceback() prints full chained traceback

- 🟢 **dbm** dumb.open() with "n" always creates new database

- 🟢 **idlelib** Various IDLE improvements (see IDLE NEWS.txt)

- 🟢 **mmap, socket, ssl, codecs** Accept writable bytes-like objects

- 🟢 **Circular imports** Circular imports involving relative imports now supported

## Improvements

### Performance

- 🔴 **os** os.walk() 3-5x faster on POSIX, 7-20x faster on Windows (using scandir())
- 🔴 **collections** OrderedDict 4-100x faster (C implementation)
- 🔴 **functools** lru_cache() significantly faster (mostly C implementation)

- 🟡 **ipaddress** Operations 3-15x faster (subnets, supernet, summarize_address_range, collapse_addresses)
- 🟡 **ipaddress** Pickling optimized for smaller output
- 🟡 **io** io.BytesIO operations 50-100% faster
- 🟡 **marshal** marshal.dumps() 20-85% faster depending on version
- 🟡 **UTF-32** UTF-32 encoder 3-7x faster
- 🟡 **regex** Regular expression parsing 10% faster
- 🟡 **json** json.dumps() with ensure_ascii=False as fast as True
- 🟡 **isinstance** PyObject_IsInstance() and PyObject_IsSubclass() faster
- 🟡 **Method caching** 5% improvement in some benchmarks
- 🟡 **random** random module objects use 50% less memory on 64-bit
- 🟡 **property** property() getter calls up to 25% faster
- 🟡 **fractions** Fraction instantiation up to 30% faster
- 🟡 **str** String search methods significantly faster for 1-character substrings
- 🟡 **bytes** bytes(int) construction faster and uses less memory

### Security

- 🔴 **ssl** SSLv3 disabled by default
- 🟡 **http** HTTP cookie parsing stricter to prevent injection attacks

## Implementation Details

### CPython Bytecode

- 🟢 **Bytecode** Various bytecode optimizations and changes

### C API

- 🟡 **C API** PyMem_RawCalloc(), PyMem_Calloc(), PyObject_Calloc()
- 🟡 **C API** Py_DecodeLocale(), Py_EncodeLocale()
- 🟡 **C API** PyCodec_NameReplaceErrors()
- 🟡 **C API** PyErr_FormatV()
- 🟡 **C API** PyExc_RecursionError exception
- 🟡 **C API** PyModule_FromDefAndSpec(), PyModule_FromDefAndSpec2(), PyModule_ExecDef() for PEP 489
- 🟡 **C API** PyNumber_MatrixMultiply(), PyNumber_InPlaceMatrixMultiply()
- 🟡 **C API** PyTypeObject.tp_finalize now part of stable ABI

### Build System

- 🔴 **Build** Windows builds require Microsoft Visual C++ 14.0 (Visual Studio 2015)
- 🟡 **Build** Extension module filename tagging on Linux, Windows, macOS

## Platform & Environment

- 🔴 **Platform** Windows XP no longer supported (PEP 11)
- 🟡 **Platform** New Windows installer replacing MSI
