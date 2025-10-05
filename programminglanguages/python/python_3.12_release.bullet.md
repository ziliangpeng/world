# Python 3.12 Release Notes

**Released:** October 2, 2023
**EOL:** October 2028 (security support)

## Major Highlights

Python 3.12 focuses on usability improvements for type hints, f-strings, and developer experience:

1. **New type parameter syntax (PEP 695)** - Cleaner generic classes and functions: `def max[T](args: Iterable[T]) -> T`
2. **F-string restrictions removed (PEP 701)** - Can reuse quotes, use multiline expressions, and include backslashes
3. **Comprehensions 2x faster (PEP 709)** - List/dict/set comprehensions inlined for major speedup
4. **isinstance() 2-20x faster** - Protocol checks dramatically accelerated
5. **Better error messages** - "Did you forget to import 'sys'?" and "Did you mean: 'self.blech'?"
6. **Per-interpreter GIL (PEP 684)** - Foundation for better parallelism (C-API only)
7. **distutils removed (PEP 632)** - Use setuptools or modern packaging tools

## Breaking Changes

- 🔴 **distutils** Removed distutils package (PEP 632) - Use setuptools or modern build tools instead
- 🔴 **venv** setuptools no longer pre-installed in virtual environments - Run `pip install setuptools` if needed
- 🟡 **asynchat, asyncore, imp** Removed deprecated modules
- 🟡 **configparser** Removed SafeConfigParser, readfp method, and filename attribute (deprecated in 3.2)
- 🟡 **enum** Removed EnumMeta.__getattr__
- 🟡 **ftplib** Removed FTP_TLS.ssl_version attribute - Use context parameter
- 🟡 **gzip** Removed GzipFile.filename attribute - Use name attribute
- 🟡 **hashlib** Removed pure Python pbkdf2_hmac implementation (OpenSSL required)
- 🟡 **io** Removed io.OpenWrapper and _pyio.OpenWrapper
- 🟡 **locale** Removed locale.format() - Use locale.format_string()
- 🟡 **smtpd** Removed smtpd module (PEP 594) - Use aiosmtpd PyPI package
- 🟡 **sqlite3** Removed sqlite3.enable_shared_cache() and sqlite3.OptimizedUnicode
- 🟡 **ssl** Removed ssl.RAND_pseudo_bytes(), ssl.match_hostname(), and ssl.wrap_socket()
- 🟡 **unittest** Removed 15 long-deprecated TestCase method aliases (failUnless, assertEquals, etc.)
- 🟡 **webbrowser** Removed support for obsolete browsers (Netscape, Mosaic, Galeon, etc.)
- 🟡 **xml.etree.ElementTree** Removed Element.copy() method - Use copy.copy()
- 🟡 **zipimport** Removed find_loader() and find_module() methods - Use find_spec()
- 🟡 **Syntax** Null bytes in source code now raise SyntaxError (not allowed)
- 🟡 **http.client, ftplib, imaplib, poplib, smtplib** Removed keyfile/certfile parameters - Use context parameter
- 🟡 **random** randrange() no longer accepts float arguments - Raises TypeError
- 🟡 **shlex** shlex.split() no longer accepts None for s argument
- 🟡 **os** No longer accepts bytes-like paths (bytearray, memoryview) - Only exact bytes type
- 🟡 **importlib** Removed module_repr(), set_package(), set_loader(), module_for_loader(), find_loader(), find_module(), importlib.abc.Finder, pkgutil.ImpImporter, pkgutil.ImpLoader

## Deprecations

### Removing in Python 3.14

- 🔴 **asyncio** asyncio.get_event_loop() will warn/error if no event loop exists
- 🟡 **ast** ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Ellipsis deprecated - Use ast.Constant
- 🟡 **asyncio** asyncio.get_event_loop() in coroutines without running loop
- 🟡 **pathlib** PurePath.is_reserved() method
- 🟡 **importlib.abc** Deprecated ResourceReader, Traversable classes (use importlib.resources.abc instead)
- 🟡 **tarfile** TarFile.tarfile attribute (use name attribute)
- 🟡 **typing** typing.no_type_check_decorator
- 🟡 **wave** Wave_read.getparams() and Wave_write.getparams() - Use individual methods

### Removing in Python 3.15

- 🟡 **importlib.resources** Deprecated methods in favor of files() API
- 🟡 **typing** typing.ByteString (use collections.abc.Buffer or typing.Union[bytes, bytearray, memoryview])
- 🟡 **C API** PyImport_ImportModuleNoBlock(), PyWeakref_GetObject(), PyWeakref_GET_OBJECT()

### Removing in Python 3.16

- 🟡 **threading** Passing keyword-only args group, target, name, args, kwargs, daemon to threading.Thread as positional args
- 🟡 **datetime** datetime.datetime.utcnow() and utcfromtimestamp() - Use timezone-aware alternatives
- 🟡 **C API** Bundled copy of libmpdec

### Removing in Python 3.17

- 🟡 **decimal** Decimal context manager protocol usage without exiting
- 🟡 **tarfile** TarInfo.type attribute (use TarInfo.isdir() and similar methods)
- 🟡 **typing** typing.Text (just use str)

### Pending Removal in Future Versions

- 🟡 **collections.abc** collections.abc.ByteString (PEP 688)
- 🟢 **argparse** Nesting argument groups and nesting mutually exclusive groups
- 🟢 **builtins** bool(NotImplemented), generator throw/athrow 3-arg signature, numeric literals followed by keywords (0in x), __index__/__int__/__float__/__complex__ returning non-standard types
- 🟢 **calendar** calendar.January and calendar.February constants (use calendar.JANUARY/FEBRUARY)
- 🟢 **codeobject** co_lnotab attribute (use co_lines() method)
- 🟢 **datetime** utcnow() and utcfromtimestamp()
- 🟢 **gettext** Plural value must be integer
- 🟢 **importlib** cache_from_source() debug_override parameter
- 🟢 **logging** warn() method (use warning())
- 🟢 **mailbox** StringIO input and text mode
- 🟢 **os** Calling os.register_at_fork() in multi-threaded process
- 🟢 **re** More strict rules for numerical group references and group names
- 🟢 **shutil** rmtree() onerror parameter (use onexc)
- 🟢 **ssl** SSLContext without protocol, set_npn_protocols(), SSL/TLS protocol versions
- 🟢 **threading** Old-style camelCase method names (notifyAll, isSet, isDaemon, etc.)
- 🟢 **urllib.parse** Many split functions (use urlparse)
- 🟢 **urllib.request** URLopener and FancyURLopener
- 🟢 **xml.etree.ElementTree** Testing truth value of Element
- 🟢 **zipimport** zipimporter.load_module()

## New Features

### Language Syntax

- 🔴 **Type System** New type parameter syntax for generics (PEP 695)
  - Simpler syntax: `def max[T](args: Iterable[T]) -> T`
  - New `type` statement for type aliases: `type Point = tuple[float, float]`
  - Support for TypeVarTuple and ParamSpec in new syntax
  - Lazy evaluation of type aliases

- 🔴 **Syntax** F-string restrictions removed (PEP 701)
  - Can reuse quotes: `f"Hello {', '.join(names)}"`
  - Multiline expressions and comments now allowed
  - Backslashes in expressions now supported
  - Arbitrary nesting depth now possible
  - Better error messages with exact location

- 🟢 **Syntax** Invalid escape sequences now raise SyntaxWarning instead of DeprecationWarning
- 🟢 **Syntax** Octal escapes >0o377 now produce SyntaxWarning
- 🟢 **Syntax** Assignment expressions allowed for comprehension target variables that aren't stored

### Type Hints

- 🔴 **Type System** TypedDict for **kwargs typing (PEP 692) - More precise function signatures
- 🟡 **Type System** typing.override() decorator (PEP 698) - Mark methods that override base class methods

### Interpreter & Runtime

- 🟡 **Interpreter** Per-interpreter GIL (PEP 684) - C-API only, enables true parallelism in sub-interpreters
- 🟡 **Interpreter** Low impact monitoring API (PEP 669) - sys.monitoring for profilers/debuggers with near-zero overhead
- 🟡 **Interpreter** Garbage collector now runs on eval breaker instead of object allocations
- 🟡 **Data Model** Buffer protocol accessible in Python (PEP 688) - Implement __buffer__() method
- 🟡 **Data Model** collections.abc.Buffer ABC for buffer types
- 🟡 **Performance** Comprehension inlining (PEP 709) - Dict/list/set comprehensions up to 2x faster
- 🟡 **Performance** Stack overflow protection on supported platforms
- 🟡 **Debugging** CPython support for Linux perf profiler (PYTHONPERFSUPPORT env var, -X perf option)

### Error Messages

- 🟡 **Error Messages** NameError suggests stdlib modules: "Did you forget to import 'sys'?"
- 🟡 **Error Messages** NameError in methods suggests self.attribute
- 🟡 **Error Messages** SyntaxError for "import x from y" suggests "from y import x"
- 🟡 **Error Messages** ImportError suggests correct names from module

### Standard Library

- 🟡 **pathlib** Path and PurePath now support subclassing via with_segments() method
- 🟡 **pathlib** Path.walk() method added (like os.walk())
- 🟡 **pathlib** PurePath.relative_to() supports walk_up parameter for .. entries
- 🟡 **pathlib** Path.is_junction() added for Windows junctions
- 🟡 **pathlib** Path.glob/rglob/match support case_sensitive parameter
- 🟡 **os** os.listdrives(), os.listvolumes(), os.listmounts() on Windows
- 🟡 **os** os.stat() and os.lstat() more accurate on Windows (st_birthtime, 64-bit st_dev, 128-bit st_ino)
- 🟡 **os** DirEntry.is_junction() for Windows junctions
- 🟡 **os** os.PIDFD_NONBLOCK for non-blocking pidfd_open()
- 🟡 **os.path** os.path.isjunction() for Windows junctions
- 🟡 **os.path** os.path.splitroot() splits path into (drive, root, tail)
- 🟡 **sqlite3** Command-line interface added (python -m sqlite3)
- 🟡 **sqlite3** Connection.autocommit attribute for transaction control
- 🟡 **sqlite3** Connection.load_extension() entrypoint parameter
- 🟡 **sqlite3** Connection.getconfig() and setconfig() methods
- 🟡 **uuid** Command-line interface added (python -m uuid)
- 🟡 **asyncio** eager_task_factory() and create_eager_task_factory() for eager task execution
- 🟡 **asyncio** asyncio.run() loop_factory parameter
- 🟡 **asyncio** PidfdChildWatcher default on Linux with pidfd_open support
- 🟡 **asyncio** wait() and as_completed() accept generators
- 🟡 **calendar** calendar.Month and calendar.Day enums
- 🟡 **csv** QUOTE_NOTNULL and QUOTE_STRINGS flags
- 🟡 **dis** Pseudo instruction opcodes exposed, dis.hasarg and dis.hasexc collections
- 🟡 **fractions** Fraction objects support float-style formatting
- 🟡 **importlib.resources** as_file() supports resource directories
- 🟡 **inspect** markcoroutinefunction() decorator
- 🟡 **inspect** getasyncgenstate() and getasyncgenlocals() functions
- 🟡 **inspect** BufferFlags enum for buffer protocol
- 🟡 **itertools** itertools.batched() for collecting into even-sized tuples
- 🟡 **math** math.sumprod() for sum of products
- 🟡 **math** math.nextafter() steps parameter
- 🟡 **platform** Detects Windows 11 and Windows Server 2012+
- 🟡 **pdb** Convenience variables for debug sessions
- 🟡 **random** random.binomialvariate()
- 🟡 **random** random.expovariate() default lambd=1.0
- 🟡 **shutil** make_archive() passes root_dir to custom archivers
- 🟡 **shutil** rmtree() onexc error handler (replaces onerror)
- 🟡 **shutil** which() improvements for Windows
- 🟡 **statistics** correlation() ranked method for Spearman correlation
- 🟡 **sys** sys.monitoring module for low-overhead event monitoring
- 🟡 **sys** activate_stack_trampoline(), deactivate_stack_trampoline(), is_stack_trampoline_active()
- 🟡 **tempfile** NamedTemporaryFile() delete_on_close parameter
- 🟡 **threading** threading.Thread.is_shutdown() class method
- 🟡 **tkinter** Tkinter widgets now support background tasks via event loop integration
- 🟡 **types** types.get_original_bases() function
- 🟡 **typing** typing.TypeAliasType for type statement
- 🟡 **typing** typing.get_protocol_members() and typing.is_protocol()
- 🟡 **unicodedata** Updated to Unicode 15.0.0
- 🟢 **array** array.array supports subscripting (generic type)
- 🟢 **tarfile** Extraction filter argument (PEP 706) - Default changes to 'data' in 3.14
- 🟢 **types** types.MappingProxyType instances now hashable
- 🟢 **slice** slice objects now hashable

### Built-in Improvements

- 🟡 **builtins** All callables accepting bool parameters now accept any type
- 🟡 **builtins** memoryview supports half-float type ('e' format)
- 🟡 **builtins** sum() uses Neumaier summation for better accuracy with floats
- 🟢 **builtins** ast.parse() raises SyntaxError instead of ValueError for null bytes
- 🟢 **builtins** Exceptions in __set_name__ no longer wrapped in RuntimeError

## Improvements

### Performance

- 🔴 **Performance** Comprehensions up to 2x faster (PEP 709 - inlining)
- 🔴 **Performance** isinstance() checks against runtime_checkable protocols 2-20x faster
- 🟡 **Performance** asyncio socket writes 75% faster, eager tasks 2-5x faster
- 🟡 **Performance** asyncio.current_task() 4-6x faster (C implementation)
- 🟡 **Performance** tokenize 64% faster due to PEP 701 changes
- 🟡 **Performance** inspect.getattr_static() at least 2x faster
- 🟡 **Performance** os.stat()/os.lstat() significantly faster on newer Windows
- 🟡 **Performance** Experimental BOLT binary optimizer support (1-5% improvement)
- 🟡 **Security** SHA1/SHA3/SHA2-384/SHA2-512/MD5 replaced with HACL* formally verified implementations

## Implementation Details

### CPython Bytecode

- 🟢 **Bytecode** LOAD_SUPER_ATTR instruction for faster super() calls
- 🟢 **Bytecode** Pseudo instructions now distinguished from real opcodes
- 🟢 **Bytecode** Many bytecode optimizations and changes

### C API

- 🟡 **C API** Unstable C API tier introduced (PEP 697) - PyUnstable_* prefix
- 🟡 **C API** API for extending types with opaque memory layout
- 🟡 **C API** PyType_FromMetaclass() for creating types with custom metaclass
- 🟡 **C API** Vectorcall protocol added to Limited API
- 🟡 **C API** PyEval_SetProfileAllThreads() and PyEval_SetTraceAllThreads()
- 🟡 **C API** Dictionary watchers (PyDict_AddWatcher, PyDict_Watch)
- 🟡 **C API** Type watchers (PyType_AddWatcher, PyType_Watch)
- 🟡 **C API** Code watchers (PyCode_AddWatcher, PyCode_ClearWatcher)
- 🟡 **C API** PyErr_GetRaisedException()/SetRaisedException() - Replace Fetch/Restore
- 🟡 **C API** Immortal objects (PEP 683) - Objects with constant refcount
- 🟡 **C API** Py_TPFLAGS_MANAGED_DICT and Py_TPFLAGS_MANAGED_WEAKREF flags
- 🟡 **C API** PyLongObject internals changed for performance
- 🟢 **C API** Legacy Unicode APIs removed (Py_UNICODE* based, PEP 623)
- 🟢 **C API** Global config variables deprecated (use PyConfig)
- 🟢 **C API** PyErr_Fetch/Restore deprecated (use PyErr_GetRaisedException/SetRaisedException)
- 🟢 **C API** structmember.h deprecated (use Python.h)
- 🟢 **C API** Creating types with metaclass overriding tp_new deprecated

### Build System

- 🟡 **Build** Python no longer uses setup.py for C extensions - Use configure and Makefile
- 🟡 **Build** va_start() with two parameters now required
- 🟡 **Build** ThinLTO default with Clang
- 🟡 **Build** Autoconf 2.71 and aclocal 1.16.4 required
- 🟡 **Build** Windows and macOS now use OpenSSL 3.0
- 🟢 **Build** LoongArch64 platform triplets added
- 🟢 **Build** PYTHON_FOR_REGEN requires Python 3.10+

## Platform & Environment

- 🟡 **Platform** iOS and Android tier 3 support tracking (not yet fully supported)
- 🟡 **Environment** PYTHONPERFSUPPORT environment variable for perf profiler
- 🟡 **Environment** -X perf command-line option