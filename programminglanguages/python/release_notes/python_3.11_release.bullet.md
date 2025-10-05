# Python 3.11 Release Notes

**Released:** October 24, 2022
**EOL:** October 2027 (security support)

## Major Highlights

Python 3.11 brings dramatic performance improvements and significantly better error messages:

1. **10-60% faster than Python 3.10 (PEP 659)** - Average 1.25x speedup through adaptive specializing interpreter
2. **Exception Groups and except* (PEP 654)** - Handle multiple unrelated exceptions simultaneously
3. **Fine-grained error locations (PEP 657)** - Tracebacks point to exact expressions, not just lines
4. **TOML parsing (PEP 680)** - Built-in `tomllib` module for TOML configuration files
5. **Faster startup (10-15%)** - Core modules frozen with statically allocated code objects
6. **Enhanced type hints** - Variadic generics, Self type, LiteralString, Required/NotRequired for TypedDict
7. **Exception notes (PEP 678)** - Add contextual information to exceptions with `add_note()`

## Breaking Changes

- 🟡 **int/str conversion** CVE-2020-10735 mitigation - Converting int/str in bases other than 2/4/8/16/32 limited to 4300 digits by default (configurable)
- 🟢 **sys.path** Bytes no longer accepted on sys.path - Only str paths supported
- 🟢 **file mode** 'U' mode removed from open() - Universal newline mode is default in text mode
- 🟢 **asyncio** Removed reuse_address parameter from create_datagram_endpoint() - Security concerns
- 🟢 **asyncio** @asyncio.coroutine() decorator removed - Use async def instead
- 🟢 **binhex** Removed binhex module and related binascii functions
- 🟢 **distutils** Removed distutils bdist_msi command - Use bdist_wheel instead
- 🟢 **inspect** Removed getargspec(), formatargspec(), Signature.from_builtin() - Use inspect.signature()
- 🟢 **gettext** Removed lgettext(), ldgettext(), lngettext(), ldngettext() and codeset parameter
- 🟢 **unittest** Removed namespace package support from unittest discovery

## Deprecations

### Removing in Python 3.12

- 🔴 **asynchat, asyncore, smtpd** Modules deprecated for removal
- 🔴 **distutils** Entire package deprecated for removal - Use setuptools or modern build tools
- 🔴 **imp** Module deprecated for removal
- 🟡 **typing.io, typing.re** Namespaces deprecated for removal
- 🟡 **unittest** 15 TestCase method aliases deprecated (failUnless, assertEquals, etc.)

### Removing in Python 3.13

- 🔴 **PEP 594 "Dead Batteries"** 19 legacy modules deprecated - aifc, audioop, cgi, cgitb, chunk, crypt, imghdr, mailcap, msilib, nis, nntplib, ossaudiodev, pipes, sndhdr, spwd, sunau, telnetlib, uu, xdrlib
- 🟡 **lib2to3** Package and 2to3 tool deprecated - Cannot parse Python 3.10+ due to PEG parser
- 🟡 **configparser** SafeConfigParser, readfp(), filename property deprecated
- 🟡 **configparser** LegacyInterpolation deprecated
- 🟡 **importlib.resources** Older function set deprecated (contents(), is_resource(), open_binary(), etc.) - Use files() API
- 🟡 **turtle** settiltangle() deprecated - Use tiltangle()
- 🟡 **typing** Text deprecated - Use str instead
- 🟡 **typing** TypedDict keyword argument syntax deprecated
- 🟡 **unittest** findTestCases(), makeSuite(), getTestCaseNames() deprecated - Use TestLoader methods
- 🟡 **unittest** TestProgram.usageExit() deprecated
- 🟡 **unittest** Returning values from test methods deprecated
- 🟡 **webbrowser** MacOSX deprecated
- 🟡 **sre_compile, sre_constants, sre_parse** Undocumented modules deprecated

### Removing in Python 3.15

- 🟡 **locale** getdefaultlocale() deprecated - Use setlocale(), getpreferredencoding(False), getlocale()
- 🟡 **locale** resetlocale() deprecated - Use setlocale(LC_ALL, "")

### Pending Removal (Future Versions)

- 🟢 **Language** Chaining classmethod descriptors deprecated
- 🟢 **Language** Octal escapes >0o377 now produce DeprecationWarning, will become SyntaxError
- 🟢 **Language** int() delegation to __trunc__() deprecated - Implement __int__() or __index__()
- 🟢 **re** Stricter rules for numerical group references and group names
- 🟢 **re** re.template(), re.TEMPLATE, re.T deprecated

## New Features

### Language Syntax

- 🔴 **Syntax** Exception Groups and except* (PEP 654) - Handle multiple exceptions: `except* ValueError as e:`
- 🟡 **Syntax** Starred unpacking in for statements - `for x, *y in data:`
- 🟡 **Syntax** Async comprehensions inside comprehensions in async functions

### Exception Handling

- 🟡 **Exceptions** BaseException.add_note() (PEP 678) - Add contextual notes to exceptions
- 🟡 **Exceptions** ExceptionGroup and BaseExceptionGroup - Group multiple exceptions together

### Type Hints

- 🔴 **Type System** Variadic generics (PEP 646) - TypeVarTuple for arbitrary number of types (array shapes)
- 🟡 **Type System** Self type (PEP 673) - Annotate methods returning instances of their class
- 🟡 **Type System** LiteralString (PEP 675) - Prevent injection attacks by requiring literal strings
- 🟡 **Type System** Required and NotRequired (PEP 655) - Mark individual TypedDict items as required/optional
- 🟡 **Type System** dataclass_transform() (PEP 681) - Decorator for dataclass-like behavior
- 🟡 **typing** assert_never() and Never - Exhaustiveness checking for type checkers
- 🟡 **typing** reveal_type() - Ask type checker for inferred type
- 🟡 **typing** assert_type() - Confirm type checker inferred expected type
- 🟡 **typing** TypedDict and NamedTuple now support generics
- 🟡 **typing** Allow subclassing of Any
- 🟡 **typing** final() decorator sets __final__ attribute
- 🟡 **typing** get_overloads() and clear_overloads() for introspection

### Interpreter & Runtime

- 🟡 **Interpreter** Fine-grained error locations (PEP 657) - Tracebacks point to exact expression with ^^^ markers
- 🟡 **Interpreter** -P option and PYTHONSAFEPATH env var - Disable prepending script directory to sys.path
- 🟡 **Interpreter** --help-env, --help-xoptions, --help-all options
- 🟡 **Format** "z" option in format spec - Coerce negative zero to positive zero after rounding

### New Modules

- 🟡 **tomllib** TOML parsing support (PEP 680) - Read-only TOML parser in standard library
- 🟢 **wsgiref.types** WSGI types for static type checking

### Standard Library

- 🟡 **asyncio** TaskGroup - Context manager for managing task groups (replaces gather/create_task pattern)
- 🟡 **asyncio** timeout() - Context manager for timeouts (replaces wait_for)
- 🟡 **asyncio** Runner - Exposes machinery used by asyncio.run()
- 🟡 **asyncio** Barrier synchronization primitive
- 🟡 **asyncio** StreamWriter.start_tls() - Upgrade streams to TLS
- 🟡 **asyncio** sock_sendto(), sock_recvfrom(), sock_recvfrom_into() - Raw datagram socket functions
- 🟡 **asyncio** Task.cancelling() and Task.uncancel()
- 🟡 **contextlib** chdir() - Context manager for changing directory
- 🟡 **dataclasses** Fields must be hashable (stricter than dict/list/set check)
- 🟡 **datetime** datetime.UTC - Convenience alias for timezone.utc
- 🟡 **datetime** fromisoformat() now parses most ISO 8601 formats
- 🟡 **enum** EnumType (EnumMeta renamed), StrEnum, ReprEnum
- 🟡 **enum** __format__() now matches __str__() behavior
- 🟡 **enum** Flag boundary parameter and FlagBoundary enum
- 🟡 **enum** verify() decorator and EnumCheck enum
- 🟡 **enum** member() and nonmember() decorators
- 🟡 **enum** property() decorator for enums
- 🟡 **enum** global_enum() decorator
- 🟡 **enum** Flag supports len(), iteration, in/not in
- 🟡 **fractions** PEP 515 style initialization (underscores in numbers)
- 🟡 **fractions** Fraction implements __int__()
- 🟡 **functools** singledispatch() supports UnionType and typing.Union
- 🟡 **hashlib** file_digest() - Helper for hashing files efficiently
- 🟡 **hashlib** blake2b/s prefer libb2, internal _sha3 uses tiny_sha3
- 🟡 **inspect** getmembers_static() - Return members without triggering descriptors
- 🟡 **inspect** ismethodwrapper() - Check for MethodWrapperType
- 🟡 **inspect** Frame functions return FrameInfo/Traceback with PEP 657 position info
- 🟡 **locale** getencoding() - Get current locale encoding
- 🟡 **logging** getLevelNamesMapping() - Return level name to value mapping
- 🟡 **logging** SysLogHandler.createSocket()
- 🟡 **math** exp2() - 2 raised to power of x
- 🟡 **math** cbrt() - Cube root
- 🟡 **math** pow(0.0, -inf) and pow(-0.0, -inf) now return inf (IEEE 754 compliant)
- 🟡 **math** math.nan always available
- 🟡 **operator** operator.call() - Equivalent to obj(*args, **kwargs)
- 🟡 **pathlib** glob/rglob return only directories if pattern ends with separator
- 🟡 **re** Atomic grouping (?>...) and possessive quantifiers (*+, ++, ?+, {m,n}+)
- 🟡 **shutil** rmtree() dir_fd parameter
- 🟡 **socket** CAN Socket support for NetBSD
- 🟡 **socket** create_connection() can raise ExceptionGroup with all errors
- 🟡 **sqlite3** Disable authorizer with None
- 🟡 **sqlite3** Unicode characters in collation names
- 🟡 **sqlite3** sqlite_errorcode and sqlite_errorname on exceptions
- 🟡 **sqlite3** setlimit() and getlimit()
- 🟡 **sqlite3** threadsafety reflects underlying SQLite threading mode
- 🟡 **sqlite3** serialize() and deserialize()
- 🟡 **sqlite3** create_window_function()
- 🟡 **sqlite3** blobopen() and Blob for incremental I/O
- 🟡 **string** Template.get_identifiers() and Template.is_valid()
- 🟡 **sys** exception() - Return active exception (equivalent to exc_info()[1])
- 🟡 **sys** exc_info() derives type/traceback from value
- 🟡 **sys** sys.flags.safe_path
- 🟡 **sysconfig** posix_venv, nt_venv, venv installation schemes
- 🟡 **tempfile** SpooledTemporaryFile fully implements BufferedIOBase/TextIOBase
- 🟡 **threading** Lock.acquire() uses monotonic clock for timeout on Unix (with sem_clockwait)
- 🟡 **time** time.sleep() uses clock_nanosleep/nanosleep (1ns resolution) on Unix
- 🟡 **time** time.sleep() uses waitable timer (100ns resolution) on Windows 8.1+
- 🟡 **tkinter** info_patchlevel() returns exact Tcl version as named tuple
- 🟡 **traceback** StackSummary.format_frame_summary() - Override frame formatting
- 🟡 **traceback** TracebackException.print()
- 🟡 **unicodedata** Updated to Unicode 14.0.0
- 🟡 **unittest** enterContext(), enterClassContext(), enterAsyncContext(), enterModuleContext()
- 🟡 **venv** Uses venv sysconfig scheme
- 🟡 **warnings** catch_warnings() accepts arguments for simplefilter()
- 🟡 **zipfile** Member name encoding support
- 🟡 **zipfile** ZipFile.mkdir()
- 🟡 **zipfile** Path.stem, Path.suffix, Path.suffixes
- 🟢 **os** os.urandom() uses BCryptGenRandom() on Windows
- 🟢 **IDLE** Syntax highlighting for .pyi files
- 🟢 **IDLE** Include prompts when saving Shell

### Windows

- 🟡 **Windows** py.exe launcher supports company/tag syntax (PEP 514) - `-V:Company/Tag`
- 🟡 **Windows** py.exe -64 means "not 32-bit" (not just x86-64)
- 🟡 **Windows** Installer AppendPath option

## Improvements

### Performance - Faster CPython Project

- 🔴 **Performance** 10-60% faster overall (1.25x average speedup on pyperformance)
- 🔴 **Performance** PEP 659 Specializing Adaptive Interpreter - Inline caching and specialization for hot code
- 🔴 **Performance** 10-15% faster startup - Core modules frozen with static code objects
- 🔴 **Performance** Cheaper lazy frames - 3-7% speedup from streamlined frame creation
- 🔴 **Performance** Inlined Python function calls - 1.7x speedup for simple recursion, 1-3% overall
- 🟡 **Performance** Zero-cost exceptions - No overhead for try statements when no exception raised
- 🟡 **Performance** Binary operations specialized - 10% faster for int/float/str operations
- 🟡 **Performance** Subscripting specialized - 10-25% faster for list/tuple/dict
- 🟡 **Performance** Calls to builtin functions specialized - 20% faster
- 🟡 **Performance** Load global/attribute specialized with inline caching
- 🟡 **Performance** Method loading specialized - 10-20% faster
- 🟡 **Performance** Store attribute specialized - 2% improvement
- 🟡 **Performance** Unpack sequence specialized - 8% faster
- 🟡 **Performance** Objects require less memory - Lazily created namespaces
- 🟡 **Performance** Exception representation more concise - 10% faster catching
- 🟡 **Performance** re engine partially refactored with computed gotos - Up to 10% faster

### Other Performance

- 🟡 **Performance** printf-style % formatting optimized for %s/%r/%a - As fast as f-strings
- 🟡 **Performance** Integer division 20% faster on x86-64 for divisors <2**30
- 🟡 **Performance** sum() nearly 30% faster for integers <2**30
- 🟡 **Performance** list.append() 15% faster, list comprehensions 20-30% faster
- 🟡 **Performance** Dictionaries 23% smaller when all keys are Unicode (no hash storage)
- 🟡 **Performance** asyncio.DatagramProtocol 100x+ faster for large file transfers
- 🟡 **Performance** math.comb() and math.perm() ~10x faster for large arguments
- 🟡 **Performance** statistics mean/variance/stdev consume iterators in one pass (2x faster, less memory)
- 🟡 **Performance** unicodedata.normalize() constant time for pure-ASCII strings

### Error Messages

- 🟡 **Error Messages** NameError suggests stdlib modules - "Did you forget to import 'sys'?"
- 🟡 **Error Messages** NameError in methods suggests self.attribute - "Did you mean: 'self.blech'?"
- 🟡 **Error Messages** SyntaxError for "import x from y" suggests "from y import x"
- 🟡 **Error Messages** ImportError suggests correct names from module

### Other Improvements

- 🟡 **Syntax** TypeError (not AttributeError) for objects not supporting context manager protocol
- 🟡 **Data Model** object.__getstate__() default implementation - Copy/pickle slots for builtin subclasses
- 🟡 **Data Model** __complex__() for complex and __bytes__() for bytes implemented
- 🟡 **Hashing** siphash13 now default hash algorithm (faster than siphash24 for long inputs)
- 🟡 **Exceptions** Bare raise statement preserves traceback modifications
- 🟡 **Windows** os.stat()/lstat() more accurate (st_birthtime, 64-bit st_dev, 128-bit st_ino)
- 🟢 **builtins** All callables accepting bool parameters now accept any type
- 🟢 **builtins** memoryview supports half-float ('e' format)
- 🟢 **builtins** sum() uses Neumaier summation for better float accuracy

## Implementation Details

### CPython Bytecode

- 🟢 **Bytecode** Inline cache entries via CACHE instructions
- 🟢 **Bytecode** ASYNC_GEN_WRAP, RETURN_GENERATOR, SEND opcodes
- 🟢 **Bytecode** COPY_FREE_VARS, JUMP_BACKWARD_NO_INTERRUPT, MAKE_CELL opcodes
- 🟢 **Bytecode** CHECK_EG_MATCH, PREP_RERAISE_STAR for exception groups
- 🟢 **Bytecode** PUSH_EXC_INFO, RESUME opcodes
- 🟢 **Bytecode** BINARY_OP replaces all BINARY_*/INPLACE_* opcodes
- 🟢 **Bytecode** CALL/KW_NAMES/PRECALL/PUSH_NULL replace CALL_FUNCTION/CALL_METHOD variants
- 🟢 **Bytecode** COPY/SWAP replace DUP_TOP*/ROT_* opcodes
- 🟢 **Bytecode** CHECK_EXC_MATCH replaces JUMP_IF_NOT_EXC_MATCH
- 🟢 **Bytecode** JUMP_BACKWARD and POP_JUMP_*_IF_* variants replace absolute jumps
- 🟢 **Bytecode** BEFORE_WITH replaces SETUP_WITH/SETUP_ASYNC_WITH
- 🟢 **Bytecode** All jump opcodes now relative (not absolute)
- 🟢 **Bytecode** MATCH_CLASS/MATCH_KEYS push None on failure instead of boolean
- 🟢 **Bytecode** Exceptions represented as one stack item instead of three
- 🟢 **Bytecode** Removed COPY_DICT_WITHOUT_KEYS, GEN_START, POP_BLOCK, SETUP_FINALLY, YIELD_FROM

### C API & Interpreter

- 🟢 **Interpreter** Interpreter state exc_info simplified - Only exc_value field retained
- 🟢 **Interpreter** PyConfig.module_search_paths_set must be 1 to use module_search_paths
- 🟢 **C API** Frame objects (codeobject.co_positions(), PyCode_Addr2Location()) expose position info
- 🟢 **C API** PYTHONNODEBUGRANGES env var and -X no_debug_ranges to disable position storage

### Platform

- 🟢 **Platform** fcntl F_DUP2FD and F_DUP2FD_CLOEXEC on FreeBSD

## Porting to Python 3.11

- 🟡 **Porting** open() no longer accepts 'U' mode - Remove 'U' flag
- 🟡 **Porting** ast.AST node positions now validated when compiling - ValueError on invalid positions
- 🟡 **Porting** asyncio.loop.set_default_executor() only accepts ThreadPoolExecutor
- 🟡 **Porting** calendar.LocaleTextCalendar/LocaleHTMLCalendar use getlocale() not getdefaultlocale()
- 🟡 **Porting** pdb reads .pdbrc with UTF-8 encoding
- 🟡 **Porting** random.sample() population must be sequence (not set), ValueError if sample > population
- 🟢 **Porting** TypeError (not AttributeError) for context manager protocol violations
