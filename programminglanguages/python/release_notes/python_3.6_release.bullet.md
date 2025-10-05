# Python 3.6 Release Notes

**Released:** December 23, 2016
**EOL:** December 2021 (reached)

## Major Highlights

Python 3.6 introduced game-changing syntax features and critical infrastructure improvements:

1. **F-strings (PEP 498)** - Formatted string literals revolutionized string formatting: `f"Hello {name}"`
2. **Variable type annotations (PEP 526)** - Complete type hint support: `primes: List[int] = []`
3. **Underscores in numeric literals (PEP 515)** - Improved readability: `1_000_000`
4. **Compact dictionaries** - 20-25% memory reduction with insertion-order preservation (implementation detail)
5. **asyncio stabilized** - Graduated from provisional to stable API with 30% performance boost
6. **Windows UTF-8 (PEP 528/529)** - Console and filesystem encoding changed to UTF-8
7. **secrets module (PEP 506)** - Cryptographically secure random numbers for security-sensitive code

## Breaking Changes

- 🔴 **Syntax** `async` and `await` are now reserved keywords (not just soft keywords)
- 🟡 **C API** Bytecode format changed - Extension modules need recompilation
- 🟡 **inspect** Removed `inspect.getmoduleinfo()` - Use `inspect.getmodulename()`
- 🟡 **inspect** Removed `inspect.getargspec()` - Use `inspect.signature()` or `inspect.getfullargspec()`
- 🟡 **importlib** Removed various deprecated APIs from `importlib.machinery` and `importlib.util`
- 🟡 **ssl** Removed SSL version 2 support (insecure protocol)
- 🟡 **asynchat, asyncore** Deprecated in favor of asyncio

## Deprecations

### General Deprecations

- 🔴 **collections** Importing ABCs from `collections` instead of `collections.abc` deprecated
- 🟡 **imp** Module deprecated in favor of `importlib`
- 🟡 **importlib** `importlib.find_loader()` deprecated - Use `importlib.util.find_spec()`
- 🟡 **pydoc** `pydoc.getpager()` deprecated
- 🟡 **re** Inline flags in middle of regex deprecated - Use at start or `(?i:...)` syntax
- 🟡 **ssl** OpenSSL 0.9.8, 1.0.0, 1.0.1 deprecated - Minimum supported version moving to 1.0.2
- 🟡 **ssl** `ssl.wrap_socket()` deprecated - Use `ssl.SSLContext.wrap_socket()`
- 🟡 **tkinter** `tkinter.tix` deprecated

### C API Deprecations

- 🟢 **C API** Undocumented private functions in `Modules/_sqlite/connection.c` deprecated
- 🟢 **C API** `PyErr_GetExcInfo()` and `PyErr_SetExcInfo()` deprecated - Use `PyErr_GetRaisedException()` and `PyErr_SetRaisedException()`

## New Features

### Language Syntax

- 🔴 **Syntax** F-strings for string formatting (PEP 498) - `f"Hello {name}!"` evaluates expressions at runtime
- 🔴 **Syntax** Variable type annotations (PEP 526) - `primes: List[int] = []`, `captain: str`
- 🟡 **Syntax** Underscores in numeric literals (PEP 515) - `1_000_000`, `0x_FF_FF`
- 🟡 **Syntax** Asynchronous generators (PEP 525) - `async def` functions can use `yield`
- 🟡 **Syntax** Asynchronous comprehensions (PEP 530) - `[i async for i in aiter()]`, `[await f() for f in funcs]`
- 🟡 **Syntax** Trailing commas allowed in function signatures and calls

### Class and Object Model

- 🟡 **Data Model** `__init_subclass__` classmethod for simpler class customization (PEP 487) - No metaclass needed
- 🟡 **Data Model** `__set_name__` descriptor protocol method (PEP 487) - Descriptors learn their attribute name
- 🟡 **Data Model** Class attribute definition order now preserved in `__dict__` (PEP 520)
- 🟡 **Data Model** `**kwargs` preserves keyword argument order (PEP 468)
- 🟡 **Data Model** Can set special methods to `None` to indicate operation not available (e.g., `__iter__ = None`)

### Type Hints

- 🟡 **Type System** `typing.ContextManager` and `typing.AsyncContextManager` ABCs
- 🟡 **Type System** `typing.Collection` ABC for sized iterable containers
- 🟡 **Type System** `typing.Reversible` ABC
- 🟡 **Type System** `typing.AsyncGenerator` ABC
- 🟡 **Type System** `typing.Text` as alias for `str`
- 🟡 **Type System** `typing.TYPE_CHECKING` constant for imports only used in type hints
- 🟡 **Type System** `typing.NewType` for creating distinct types
- 🟡 **Type System** Generic type aliases can be parameterized

### File System

- 🔴 **os** Path protocol (PEP 519) - `os.PathLike` interface, `os.fspath()` function
- 🔴 **pathlib** Path objects work with `open()` and all os/os.path functions (PEP 519)
- 🟡 **os** `os.scandir()` now supports context manager protocol
- 🟡 **os** `os.fsencode()` and `os.fsdecode()` support path-like objects
- 🟡 **os** All path functions updated to accept path-like objects

### New Module

- 🟡 **secrets** New module for cryptographically strong random numbers (PEP 506)
  - `secrets.token_bytes()`, `secrets.token_hex()`, `secrets.token_urlsafe()`
  - `secrets.choice()`, `secrets.randbelow()`, `secrets.SystemRandom`

### asyncio

- 🔴 **asyncio** Module no longer provisional - API is stable
- 🟡 **asyncio** `get_event_loop()` returns running loop when called from coroutines
- 🟡 **asyncio** `ensure_future()` accepts all awaitable objects
- 🟡 **asyncio** `run_coroutine_threadsafe()` submits coroutines from other threads
- 🟡 **asyncio** `loop.create_future()` for custom Future implementations
- 🟡 **asyncio** `loop.create_server()` accepts list of hosts
- 🟡 **asyncio** `loop.shutdown_asyncgens()` properly closes pending async generators
- 🟡 **asyncio** `StreamReader.readuntil()` reads until separator sequence
- 🟡 **asyncio** `Transport.is_closing()` checks transport state
- 🟡 **asyncio** `loop.connect_accepted_socket()` for servers accepting connections outside asyncio
- 🟡 **asyncio** TCP_NODELAY enabled by default for all TCP transports

### datetime

- 🟡 **datetime** `fold` attribute for local time disambiguation during DST transitions (PEP 495)
- 🟡 **datetime** `strftime()` supports ISO 8601 directives `%G`, `%u`, `%V`
- 🟡 **datetime** `isoformat()` accepts `timespec` parameter for precision
- 🟡 **datetime** `combine()` accepts optional `tzinfo` argument

### hashlib

- 🟡 **hashlib** BLAKE2 hash functions: `blake2b()`, `blake2s()`
- 🟡 **hashlib** SHA-3 hash functions: `sha3_224()`, `sha3_256()`, `sha3_384()`, `sha3_512()`
- 🟡 **hashlib** SHAKE hash functions: `shake_128()`, `shake_256()`
- 🟡 **hashlib** `scrypt()` key derivation function (OpenSSL 1.1.0+)
- 🟡 **hashlib** OpenSSL 1.1.0 support

### collections

- 🟡 **collections** `Collection` ABC for sized iterable containers
- 🟡 **collections** `Reversible` ABC for reversible iterables
- 🟡 **collections** `AsyncGenerator` ABC
- 🟡 **collections** `namedtuple()` accepts `module` parameter
- 🟡 **collections** `namedtuple()` `verbose` and `rename` are now keyword-only
- 🟡 **collections** Recursive `deque` instances can be pickled

### enum

- 🟡 **enum** `Flag` and `IntFlag` base classes for bitwise-combinable constants
- 🟡 **enum** `auto()` automatically assigns values to enum members
- 🟡 **enum** Many stdlib modules updated to use `IntFlag`

### email

- 🟡 **email** Email API no longer provisional
- 🟡 **email** `email.mime` classes accept `policy` keyword
- 🟡 **email** `DecodedGenerator` supports `policy` keyword
- 🟡 **email** New `message_factory` policy attribute

### contextlib

- 🟡 **contextlib** `AbstractContextManager` ABC with default `__enter__` implementation

### decimal

- 🟡 **decimal** `Decimal.as_integer_ratio()` returns fraction as (numerator, denominator)

### math

- 🟡 **math** `math.tau` constant (τ = 2π)

### cmath

- 🟡 **cmath** `cmath.tau` constant
- 🟡 **cmath** `cmath.inf`, `cmath.nan`, `cmath.infj`, `cmath.nanj` constants

### json

- 🟡 **json** `json.load()` and `json.loads()` accept binary input (UTF-8, UTF-16, UTF-32)

### os

- 🟡 **os** `os.urandom()` on Linux blocks until system urandom entropy pool initialized (PEP 524)
- 🟡 **os** `DirEntry` objects implement `os.PathLike`
- 🟡 **os** `os.scandir()` supports `with` statement

### pathlib

- 🟡 **pathlib** All `Path` methods support path-like objects
- 🟡 **pathlib** `Path.resolve()` no longer requires path to exist (strict parameter)

### pdb

- 🟡 **pdb** `pdb.Pdb` constructor accepts `readrc` argument to disable .pdbrc file reading

### pickle

- 🟡 **pickle** Protocol 4 improvements - 64-bit support for large data

### random

- 🟡 **random** `choices()` method for weighted random sampling
- 🟡 **random** `Random.choices()` performs sampling with replacement

### re

- 🟡 **re** `re.Match` object supports subscripting (e.g., `m[0]` equivalent to `m.group(0)`)

### readline

- 🟡 **readline** `append_history_file()` appends to history file

### shlex

- 🟡 **shlex** `shlex` class accepts `punctuation_chars` argument for improved shell-like parsing

### socket

- 🟡 **socket** `socket.close()` on Windows no longer cancels pending async operations
- 🟡 **socket** `socket.ioctl()` now supports SIO_LOOPBACK_FAST_PATH on Windows

### socketserver

- 🟡 **socketserver** All servers support context manager protocol
- 🟡 **socketserver** `server_close()` called on context manager exit

### ssl

- 🟡 **ssl** OpenSSL 1.1.0 support
- 🟡 **ssl** `SSLContext` defaults improved for better security
- 🟡 **ssl** `MemoryBIO` for in-memory SSL I/O
- 🟡 **ssl** `SSLObject` for separate SSL protocol instance
- 🟡 **ssl** `PROTOCOL_TLS` alias for auto-negotiating protocol version

### statistics

- 🟡 **statistics** `statistics.harmonic_mean()` function

### struct

- 🟡 **struct** Struct instances now support subscripting to extract field values

### subprocess

- 🟡 **subprocess** `subprocess.run()` default encoding matches `sys.getdefaultencoding()`
- 🟡 **subprocess** `encoding` and `errors` parameters added

### sys

- 🟡 **sys** `sys.getfilesystemencoding()` on Windows returns 'utf-8' (PEP 529)
- 🟡 **sys** `sys.stdin/stdout/stderr` on Windows default to utf-8 encoding (PEP 528)

### time

- 🟡 **time** `time.CLOCK_THREAD_CPUTIME_ID` and `time.pthread_getcpuclockid()` on relevant platforms

### timeit

- 🟡 **timeit** `timeit.timeit()` now accepts a `globals` argument

### tkinter

- 🟡 **tkinter** Various Tkinter improvements and new widget options

### traceback

- 🟡 **traceback** Repeated traceback lines abbreviated as "[Previous line repeated {count} more times]"

### tracemalloc

- 🟡 **tracemalloc** Integration with memory debug hooks for better diagnostics
- 🟡 **tracemalloc** `tracemalloc.Traceback` now has `format()` method

### typing

- 🟡 **typing** Generic type aliases: `Vector = List[T]`
- 🟡 **typing** `ContextManager`, `AsyncContextManager`, `Collection`, `Reversible`, `AsyncGenerator` ABCs

### unittest.mock

- 🟡 **unittest.mock** `Mock` and `MagicMock` accept `unsafe` parameter for dangerous attributes

### urllib.request

- 🟡 **urllib.request** `AbstractBasicAuthHandler` supports arbitrary realm for HTTP Basic Auth

### urllib.robotparser

- 🟡 **urllib.robotparser** `RobotFileParser.request_rate()` and `crawl_delay()` methods

### venv

- 🟡 **venv** `pyvenv` script deprecated - Use `python -m venv`

### warnings

- 🟡 **warnings** `ResourceWarning` now uses `tracemalloc` to provide better diagnostics

### zipfile

- 🟡 **zipfile** `ZipInfo.from_file()` and `ZipInfo.is_dir()` methods

### zlib

- 🟡 **zlib** Compression level 0 now uses Z_NO_COMPRESSION (no compression overhead)

### Windows-specific

- 🔴 **Platform** Console encoding changed to UTF-8 (PEP 528) - `sys.stdin/stdout/stderr` use utf-8
- 🔴 **Platform** Filesystem encoding changed to UTF-8 (PEP 529) - `sys.getfilesystemencoding()` returns 'utf-8'
- 🟡 **Platform** `py.exe` launcher no longer prefers Python 2 when used interactively
- 🟡 **Platform** `python.exe` and `pythonw.exe` marked long-path aware (bypasses 260-char limit)
- 🟡 **Platform** `._pth` files enable isolated mode with explicit search paths
- 🟡 **Platform** `python36.zip` can serve as PYTHONHOME landmark
- 🟡 **encodings** New 'oem' encoding and 'ansi' alias for 'mbcs' on Windows
- 🟡 **winreg** `REG_MULTI_SZ` values now support empty strings

## Improvements

### Performance

- 🔴 **Performance** Dictionaries 20-25% more memory efficient with compact representation
- 🔴 **Performance** asyncio Future and Task 30% faster with C implementations
- 🟡 **Performance** `StreamReader.readexactly()` performance improved
- 🟡 **Performance** `loop.getaddrinfo()` optimized to avoid redundant system calls
- 🟡 **Performance** Various bytecode optimizations
- 🟡 **Performance** Method calls and attribute access optimized
- 🟡 **Performance** Dictionary lookups faster due to compact layout
- 🟡 **Performance** Startup time improved

### Error Messages

- 🟡 **Error Messages** New `ModuleNotFoundError` exception (subclass of `ImportError`)
- 🟡 **Error Messages** Repeated traceback lines abbreviated for clarity
- 🟡 **Error Messages** Better messages for various syntax errors

### Security

- 🟡 **Security** New `secrets` module for cryptographically strong randomness
- 🟡 **Security** `os.urandom()` blocks on Linux until entropy available
- 🟡 **Security** OpenSSL 1.1.0 support with modern algorithms
- 🟡 **Security** Improved `ssl` module defaults

## Implementation Details

### CPython Interpreter

- 🟡 **Interpreter** Compact dictionary implementation preserves insertion order (implementation detail, not guaranteed)
- 🟡 **Interpreter** Frame evaluation API (PEP 523) - C-level pluggable frame evaluation
- 🟡 **Interpreter** PYTHONMALLOC environment variable for memory allocator control and debugging
- 🟡 **Interpreter** DTrace and SystemTap probing support (build with --with-dtrace)
- 🟡 **Interpreter** Memory allocator debug hooks can be enabled in release builds
- 🟡 **Interpreter** `global` and `nonlocal` must appear before first use (was SyntaxWarning, now SyntaxError)
- 🟡 **Interpreter** Zero-argument `super()` works correctly in `__init_subclass__`

### CPython Bytecode

- 🟢 **Bytecode** Many bytecode instructions added, modified, or removed
- 🟢 **Bytecode** Bytecode format is not backwards compatible
- 🟢 **Bytecode** `MAKE_FUNCTION` simplified
- 🟢 **Bytecode** `BUILD_MAP_UNPACK_WITH_CALL` added
- 🟢 **Bytecode** Various optimizations to instruction dispatch

### C API

- 🟡 **C API** Frame evaluation API (`_PyInterpreterState_GetEvalFrameFunc()`, `_PyInterpreterState_SetEvalFrameFunc()`)
- 🟡 **C API** `PyMem_SetupDebugHooks()` for debug hooks
- 🟡 **C API** `Py_FinalizeEx()` reports errors during finalization
- 🟡 **C API** `PyErr_ResourceWarning()` function
- 🟡 **C API** `PyArg_ParseTuple()` now supports exception chaining
- 🟡 **C API** Various new functions and improved error handling
- 🟢 **C API** Extensive internal changes to support new features

### Build System

- 🟡 **Build** `--with-optimizations` configure flag for profile-guided optimization
- 🟡 **Build** `--with-dtrace` configure flag for DTrace/SystemTap support
- 🟡 **Build** `--enable-loadable-sqlite-extensions` configure flag
- 🟢 **Build** Updated build requirements

## Platform & Environment

- 🔴 **Platform** Windows console and filesystem encoding changed to UTF-8
- 🟡 **Environment** PYTHONMALLOC environment variable controls memory allocator
- 🟡 **Environment** PYTHONLEGACYWINDOWSFSENCODING reverts to pre-3.6 Windows filesystem encoding
- 🟡 **Environment** PYTHONLEGACYWINDOWSSTDIO reverts to pre-3.6 Windows console encoding

## Other Improvements

- 🟡 **IDLE** Numerous improvements to the IDLE development environment
- 🟡 **Documentation** Extensive documentation updates and improvements
- 🟡 **Testing** Improved test coverage and test infrastructure
