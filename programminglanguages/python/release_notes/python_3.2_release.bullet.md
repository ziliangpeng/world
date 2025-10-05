# Python 3.2 Release Notes

**Released:** February 20, 2011
**EOL:** February 2016 (ended)

## Major Highlights

Python 3.2 is the release that made Python 3 production-ready, fixing critical bytes/text handling issues:

1. **concurrent.futures (PEP 3148)** - High-level parallelism with ThreadPoolExecutor and ProcessPoolExecutor
2. **argparse module (PEP 389)** - Modern command-line parsing replacing optparse
3. **Stable ABI (PEP 384)** - C extensions work across Python versions without recompilation
4. **PYC repository directories (PEP 3147)** - Bytecode cached in __pycache__ with interpreter-specific names
5. **Email package fixed** - First Python 3 release with working email/mailbox for bytes and mixed encodings
6. **SSL/TLS overhaul** - Proper certificate validation and modern SSL features throughout network modules
7. **Dictionary-based logging (PEP 391)** - Configure logging from JSON/YAML files

## New Features

### Language Syntax

- 🟡 **Builtins** callable() function resurrected from Python 2.x - Concise alternative to isinstance(x, collections.Callable)
- 🟡 **Format** format() '#' character now works with floats/complex/Decimal - Always show decimal point
- 🟡 **Format** str.format_map() method for formatting with arbitrary mappings (defaultdict, shelve, ConfigParser, etc.)
- 🟡 **Interpreter** -q quiet option to suppress copyright/version in interactive mode
- 🟡 **Import** Import mechanism now handles non-ASCII characters in path names

### Major Modules

- 🔴 **concurrent.futures** New module for high-level thread/process parallelism (PEP 3148) - ThreadPoolExecutor and ProcessPoolExecutor with Future objects
- 🔴 **argparse** New command-line parsing module (PEP 389) - Replaces optparse with support for positional args, subcommands, required options
- 🟡 **logging** Dictionary-based configuration via logging.config.dictConfig() (PEP 391) - Configure from JSON/YAML
- 🟡 **html** New module with html.escape() function for escaping HTML reserved characters

### functools

- 🟡 **functools** lru_cache() decorator for memoization with maxsize parameter and cache_info() statistics
- 🟡 **functools** total_ordering() decorator fills in missing comparison methods (e.g., define __eq__ and __lt__, get others free)
- 🟡 **functools** cmp_to_key() converts old-style comparison functions to key functions
- 🟡 **functools** wraps() decorator now adds __wrapped__ attribute and copies __annotations__

### Email & Network

- 🔴 **email** Package overhauled to work with bytes and mixed encodings - message_from_bytes(), BytesParser, BytesGenerator classes
- 🔴 **ssl** SSLContext class for managing SSL configuration, certificates, and private keys
- 🟡 **ssl** ssl.match_hostname() for server identity verification (HTTPS/RFC 2818 rules)
- 🟡 **ssl** Support for ciphers parameter in wrap_socket()
- 🟡 **ssl** Server Name Indication (SNI) extension support for virtual hosts
- 🟡 **ssl** Module attributes for OpenSSL version: OPENSSL_VERSION, OPENSSL_VERSION_INFO, OPENSSL_VERSION_NUMBER
- 🟡 **nntplib** Complete rewrite with better bytes/text semantics - Breaking changes from 3.1
- 🟡 **nntplib** Support for secure connections via NNTP_SSL and starttls()
- 🟡 **smtplib** SMTP.sendmail() accepts byte strings, new send_message() method for Message objects
- 🟡 **imaplib** IMAP4.starttls() method for explicit TLS support
- 🟡 **ftplib** FTP and FTP_TLS now context managers, FTP_TLS accepts SSLContext
- 🟡 **poplib** POP3_SSL accepts SSLContext parameter
- 🟡 **http.client** Certificate checking support for HTTPSConnection
- 🟡 **http.client** HTTPConnection/HTTPSConnection have source_address parameter
- 🟡 **http.client** request() body parameter now accepts iterables
- 🟡 **http.client** set_tunnel() method for HTTP Connect tunneling through proxy
- 🟡 **urllib** HTTPSConnection, HTTPSHandler, urlopen() support certificate validation

### Collections & Itertools

- 🟡 **itertools** accumulate() function for cumulative sums (like APL scan, NumPy accumulate)
- 🟡 **collections** Counter.subtract() method for regular subtraction (vs saturating -= operator)
- 🟡 **collections** OrderedDict.move_to_end() method for resequencing entries
- 🟡 **collections** deque.count() and deque.reverse() methods for better list compatibility

### Threading & Multiprocessing

- 🟡 **threading** Barrier synchronization class for coordinating multiple threads at common barrier point
- 🟡 **threading** Barriers support timeouts and raise BrokenBarrierError on timeout

### Math & Numbers

- 🟡 **math** isfinite() function for detecting NaN/Infinity (returns True for regular numbers)
- 🟡 **math** expm1() for computing e**x-1 accurately for small x
- 🟡 **math** erf() and erfc() for error functions (Gaussian probability integrals)
- 🟡 **math** gamma() and lgamma() for gamma function and its natural logarithm

### File I/O & Archives

- 🟡 **io** BytesIO.getbuffer() method for zero-copy memory view editing
- 🟡 **gzip** GzipFile implements io.BufferedIOBase ABC, has peek() method
- 🟡 **gzip** compress() and decompress() functions for in-memory compression
- 🟡 **zipfile** ZipExtFile implementation significantly faster, can be wrapped in BufferedReader
- 🟡 **tarfile** TarFile now context manager, add() method has filter parameter (replaces deprecated exclude)
- 🟡 **shutil** copytree() has ignore_dangling_symlinks and copy_function parameters
- 🟡 **shutil** make_archive() and unpack_archive() for creating/extracting archives (zip, tar, gztar, bztar)
- 🟡 **shutil** register_archive_format() and register_unpack_format() for custom archive types
- 🟡 **mmap** mmap objects now context managers
- 🟡 **fileinput** fileinput.input() now context manager

### Date & Time

- 🟡 **datetime** timezone class implementing tzinfo interface for fixed UTC offsets
- 🟡 **datetime** timedelta supports multiplication by float, division by float/int
- 🟡 **datetime** timedelta objects can divide each other
- 🟡 **datetime** date.strftime() now supports years 1000-9999 (was restricted to 1900+)
- 🟡 **time** time.accept2dyear now triggers DeprecationWarning - Set to False for full date ranges

### XML

- 🟡 **ElementTree** Updated to version 1.3 with fromstringlist(), register_namespace(), tostringlist()
- 🟡 **ElementTree** Element.extend(), Element.iterfind(), Element.itertext() methods
- 🟡 **ElementTree** TreeBuilder.end() and TreeBuilder.doctype() methods

### Testing & Debugging

- 🟡 **unittest** Test discovery from command line: python -m unittest discover -s proj_dir -p _test.py
- 🟡 **unittest** TestCase can be instantiated without arguments for REPL experimentation
- 🟡 **unittest** assertWarns() and assertWarnsRegex() methods for verifying warnings
- 🟡 **unittest** assertCountEqual() compares iterables by element count regardless of order
- 🟡 **unittest** maxDiff attribute limits diff output length
- 🟡 **unittest** assertRegex() replaces assertRegexpMatches (better naming)
- 🟡 **inspect** getgeneratorstate() function returns GEN_CREATED/GEN_SUSPENDED/GEN_CLOSED
- 🟡 **inspect** getattr_static() for read-only attribute lookup without triggering properties/descriptors

### Other Standard Library

- 🟡 **tempfile** TemporaryDirectory() context manager for deterministic directory cleanup
- 🟡 **reprlib** recursive_repr() decorator for handling self-referential __repr__
- 🟡 **ast** literal_eval() now supports bytes and set literals (in addition to existing types)
- 🟡 **abc** abstractclassmethod() and abstractstaticmethod() decorators
- 🟡 **hashlib** algorithms_guaranteed and algorithms_available attributes listing hash algorithms
- 🟡 **os** fsencode() and fsdecode() for encoding/decoding filenames
- 🟡 **os** getenvb() and environb for bytes access to environment variables
- 🟡 **os** supports_bytes_environ constant indicates if OS allows bytes environ access
- 🟡 **select** PIPE_BUF attribute for minimum non-blocking pipe write size
- 🟡 **socket** socket.detach() method to close socket without closing file descriptor
- 🟡 **socket** create_connection() now context manager
- 🟡 **sqlite3** Updated to pysqlite 2.6.0 with in_transit attribute and load_extension() support
- 🟡 **subprocess** Popen() now context manager
- 🟡 **asyncore** dispatcher.handle_accepted() method returns (sock, addr) pair

### Built-in Improvements

- 🟡 **hasattr** Now only catches AttributeError, letting other exceptions propagate (was catching all exceptions)
- 🟡 **str/repr** str() of float/complex now equals repr() for consistency
- 🟡 **memoryview** Now context manager with release() method for timely resource cleanup
- 🟡 **range** range objects support index(), count(), slicing, and negative indices
- 🟡 **Struct sequences** os.stat(), time.gmtime(), sys.version_info are now tuple subclasses

## Improvements

### Performance

- 🟡 **Performance** random module integer methods now produce true uniform distributions (no power-of-two bias)
- 🟡 **Performance** Unicode memory optimization - Removed wstr attribute saving 8-16 bytes per string on 64-bit
- 🟢 **Performance** Various optimizations across stdlib modules

### Error Messages & Warnings

- 🟡 **Warnings** PYTHONWARNINGS environment variable as alternative to -W command line
- 🟡 **Warnings** New ResourceWarning category for unclosed files and resource leaks (silent by default)
- 🟡 **Warnings** ResourceWarning issued for unclosed file objects and gc.garbage at shutdown

## Deprecations

### Deprecated in 3.2

- 🟡 **ElementTree** getchildren() deprecated - Use list(elem)
- 🟡 **ElementTree** getiterator() deprecated - Use Element.iter
- 🟡 **time** Two-digit year interpretation with accept2dyear=True now triggers DeprecationWarning
- 🟡 **unittest** Method aliases deprecated: assert_(), assertEquals(), assertNotEquals(), assertAlmostEquals(), assertNotAlmostEquals()
- 🟡 **unittest** assertDictContainsSubset() deprecated - Arguments were in wrong order
- 🟡 **unittest** TestCase.fail* methods deprecated in 3.1, expected removal in 3.3
- 🟡 **http.client** strict parameter deprecated in all classes
- 🟡 **tarfile** exclude parameter deprecated - Use filter parameter instead

## Breaking Changes

- 🟡 **nntplib** Complete API overhaul with breaking changes from 3.1 - Better bytes/text handling
- 🟡 **http.client** HTTP 0.9 simple responses no longer supported
- 🟡 **Scoping** Exception targets in except clauses are implicitly deleted (existing behavior, now works with nested functions)

## Implementation Details

### C API & Stable ABI

- 🔴 **C API** Stable ABI introduced (PEP 384) - Extensions defining Py_LIMITED_API work across Python versions
- 🟡 **C API** PyLongObject internals changed for better performance - Use public API functions
- 🟢 **C API** Various API additions and improvements for extension authors

### Bytecode & Import

- 🔴 **Import** PYC files stored in __pycache__/ directories (PEP 3147)
- 🔴 **Import** PYC files have interpreter-specific names like mymodule.cpython-32.pyc
- 🔴 **Import** Modules have __cached__ attribute with actual imported file path
- 🟡 **Import** imp.get_tag() returns interpreter tag (e.g., 'cpython-32')
- 🟡 **Import** imp.source_from_cache() and cache_from_source() for path conversion
- 🟡 **Import** py_compile and compileall updated for new naming convention
- 🟡 **Import** compileall has -i (file list) and -b (legacy location) options
- 🟡 **Import** importlib.abc updated with new ABCs, PyLoader and PyPycLoader deprecated

### ABI Version Tagged .so Files

- 🟡 **Build** PEP 3149: Shared object files have ABI version tags (e.g., foo.cpython-32m.so)
- 🟡 **Build** sysconfig.get_config_var('SOABI') returns version tag
- 🟡 **Build** sysconfig.get_config_var('EXT_SUFFIX') returns full filename extension

### WSGI

- 🟡 **WSGI** PEP 3333 clarifies bytes/text handling in WSGI 1.0.1 for Python 3
- 🟡 **WSGI** Native strings (str type) for headers, byte strings for request/response bodies
- 🟡 **wsgiref** wsgiref.handlers.read_environ() transcodes CGI variables to native strings
