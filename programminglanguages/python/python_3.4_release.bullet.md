# Python 3.4 Release Notes

**Released:** March 16, 2014
**EOL:** March 2019 (reached)

## Major Highlights

Python 3.4 focused on infrastructure improvements and developer experience with no new syntax:

1. **pip bundled by default (PEP 453)** - Python installations now include pip, solving a major pain point for newcomers
2. **asyncio module (PEP 3156)** - Standard asynchronous I/O framework, foundation for modern async Python
3. **pathlib module (PEP 428)** - Object-oriented filesystem paths, modern alternative to os.path
4. **enum module (PEP 435)** - Standard enumeration types for better code clarity
5. **Non-inheritable file descriptors (PEP 446)** - Security improvement preventing fd leaks to child processes
6. **Safe object finalization (PEP 442)** - Finalizers work with reference cycles, module globals no longer set to None on shutdown
7. **statistics module (PEP 450)** - Built-in numerically stable statistics functions

## New Features

### New Modules

- 🔴 **asyncio** New provisional module for asynchronous I/O (PEP 3156) - Event loops, coroutines, futures, tasks
- 🔴 **pathlib** Object-oriented filesystem paths (PEP 428) - Pure paths and concrete paths with I/O operations (provisional API)
- 🔴 **enum** Enumeration types (PEP 435) - Enum, IntEnum, and unique decorator for type-safe constants
- 🔴 **ensurepip** Bootstrap pip installer (PEP 453) - Bundles pip for automatic installation
- 🟡 **statistics** Basic statistics library (PEP 450) - mean, median, mode, variance, stdev functions
- 🟡 **tracemalloc** Trace memory allocations (PEP 454) - Debug tool for tracking memory usage and finding leaks
- 🟡 **selectors** High-level I/O multiplexing - Built on select module, part of asyncio infrastructure

### Language Features

- 🟡 **min/max** Added default keyword argument for empty iterables
- 🟡 **Modules** Module objects now weakly referenceable
- 🟡 **Modules** Module __file__ attributes now contain absolute paths by default
- 🟡 **int** Constructor accepts objects with __index__ for base argument
- 🟡 **Frame** Frame objects have clear() method to clear local variable references
- 🟡 **memoryview** Now registered as Sequence and supports reversed()
- 🟡 **Unicode** Updated to UCD 6.3
- 🟡 **Codecs** New German EBCDIC codec cp273
- 🟡 **Codecs** New Ukrainian codec cp1125
- 🟡 **Codecs** UTF-* codecs reject surrogates unless surrogatepass error handler used
- 🟢 **Syntax** __length_hint__() now part of formal language specification (PEP 424)

### PIP Integration

- 🔴 **ensurepip** pip bundled by default in Python installations (PEP 453)
- 🔴 **venv** pyvenv and venv module bootstrap pip by default
- 🔴 **Installers** Windows and macOS installers include pip by default
- 🔴 **Build** make install and make altinstall bootstrap pip on POSIX systems

### functools

- 🔴 **functools** singledispatch() decorator for single-dispatch generic functions (PEP 443)
- 🟡 **functools** partialmethod() descriptor for partial argument application to methods
- 🟡 **functools** total_ordering() supports NotImplemented return value
- 🟢 **functools** Pure Python version of partial() available for other implementations

### contextlib

- 🟡 **contextlib** suppress() context manager for suppressing specific exceptions
- 🟡 **contextlib** redirect_stdout() context manager for redirecting sys.stdout

### pickle

- 🟡 **pickle** Protocol 4 (PEP 3154) - Support for very large objects, more efficient for small objects

### email

- 🟡 **email** New email.contentmanager submodule for simplified MIME handling
- 🟡 **email** EmailMessage and MIMEPart classes with improved API (provisional)
- 🟡 **email** Message.as_string() accepts policy argument
- 🟡 **email** Message.as_bytes() method for bytes representation
- 🟡 **email** Message.set_param() accepts replace keyword argument

### dis

- 🟡 **dis** Rebuilt around Instruction class for object-oriented bytecode access
- 🟡 **dis** get_instructions() provides iterator over bytecode
- 🟡 **dis** Bytecode class for human-readable and programmatic bytecode inspection
- 🟡 **dis** stack_effect() computes stack effect of opcodes
- 🟡 **dis** Functions accept file keyword argument for output redirection

### multiprocessing

- 🟡 **multiprocessing** spawn and forkserver start methods avoid os.fork on Unix for better security
- 🟡 **multiprocessing** Windows child processes no longer inherit all parent handles

### inspect

- 🟡 **inspect** Command line interface for displaying source and info
- 🟡 **inspect** unwrap() unravels wrapper function chains
- 🟡 **inspect** Better support for custom __dir__ and dynamic class attributes
- 🟡 **inspect** getfullargspec() and getargspec() use signature() API
- 🟡 **inspect** signature() supports Cython-compiled functions

### importlib

- 🟡 **importlib** reload() moved from imp to importlib
- 🟡 **importlib** importlib.util.MAGIC_NUMBER for bytecode version
- 🟡 **importlib** importlib.util.cache_from_source() and source_from_cache()
- 🟡 **importlib** importlib.util.decode_source() for universal newline processing
- 🟡 **importlib** NamespaceLoader conforms to InspectLoader ABC
- 🟡 **importlib** InspectLoader.source_to_code() method
- 🟡 **importlib** ExtensionFileLoader.get_filename() method

### Security Modules

- 🔴 **hashlib** pbkdf2_hmac() for PKCS#5 password-based key derivation
- 🟡 **hashlib** hash.name attribute now formally supported
- 🟡 **ssl** TLSv1.1 and TLSv1.2 support
- 🟡 **ssl** Server-side SNI (Server Name Indication) support
- 🟡 **ssl** Retrieve certificates from Windows system cert store
- 🟡 **ssl** SSLContext improvements
- 🟡 **ssl** All stdlib modules with SSL support certificate verification and hostname matching

### base64

- 🟡 **base64** a85encode(), a85decode() for Ascii85 encoding
- 🟡 **base64** b85encode(), b85decode() for Base85 (git/mercurial) encoding
- 🟡 **base64** Functions accept any bytes-like object

### Other Module Improvements

- 🟡 **abc** get_cache_token() for invalidating ABC caches
- 🟡 **abc** ABC class with ABCMeta as metaclass (simpler syntax)
- 🟡 **argparse** FileType accepts encoding and errors arguments
- 🟡 **audioop** 24-bit sample support and byteswap() function
- 🟡 **collections** ChainMap.new_child() accepts m argument
- 🟡 **doctest** FAIL_FAST option flag
- 🟡 **doctest** Command line -o and -f options
- 🟡 **doctest** Finds doctests in extension module __doc__ strings
- 🟡 **filecmp** clear_cache() function and DEFAULT_IGNORES attribute
- 🟡 **gc** get_stats() returns per-generation statistics
- 🟡 **glob** escape() for literal matching of special characters
- 🟡 **hmac** Accepts bytearray, digestmod can be any hashlib name
- 🟡 **hmac** block_size and name attributes (PEP 247 conformance)
- 🟡 **html** unescape() converts HTML5 character references
- 🟡 **html** HTMLParser accepts convert_charrefs argument
- 🟡 **http** BaseHTTPRequestHandler.send_error() accepts explain parameter
- 🟡 **http** http.server CLI has -b/--bind option
- 🟡 **ipaddress** API declared stable (was provisional in 3.3)
- 🟡 **ipaddress** is_global property for addresses
- 🟢 **aifc** getparams() returns namedtuple, supports context management
- 🟢 **audioop** Functions accept any bytes-like object
- 🟢 **colorsys** RGB-YIQ conversion coefficients expanded
- 🟢 **dbm** dbm.open() supports context management

## Improvements

### Security

- 🔴 **os** File descriptors non-inheritable by default (PEP 446) - Prevents leaks to child processes
- 🔴 **Hash** Secure and interchangeable hash algorithm (PEP 456) - Protection against collision DoS
- 🟡 **os** get_inheritable(), set_inheritable() for explicit fd inheritance control
- 🟡 **socket** get_inheritable(), set_inheritable() for socket inheritance
- 🟡 **CLI** -I option for isolated mode - No user site-packages, better security

### Performance

- 🟡 **marshal** More compact and efficient format for .pyc files
- 🟡 **bytecode** Various optimizations

### Error Messages

- 🟡 **codecs** Better error messages for non-text encodings - Directs to codecs.encode/decode
- 🟡 **codecs** Chained exceptions show which codec caused error

### Developer Experience

- 🟡 **help()** Improved signatures from inspect and pydoc improvements
- 🟡 **codecs** codecs.encode() and codecs.decode() now documented

### Bytes-Like Object Support

- 🟡 **bytes/bytearray** join() accepts arbitrary buffer objects
- 🟡 **base64** Encoding/decoding accepts any bytes-like object
- 🟡 **audioop** Functions accept any bytes-like object
- 🟡 **aifc** writeframesraw() and writeframes() accept any bytes-like object

## Implementation Details

### CPython Internals

- 🟡 **Finalization** Safe object finalization (PEP 442) - Finalizers work with reference cycles
- 🟡 **Finalization** Module globals no longer set to None during finalization in most cases
- 🟡 **Memory** Configurable memory allocators (PEP 445) - Custom allocators for debugging
- 🟡 **Import** ModuleSpec type for import system (PEP 451) - Encapsulates module import info
- 🟡 **C Code** Argument Clinic (PEP 436) - Preprocessor for C function parameter declarations

## Deprecations

### Deprecated in 3.4

- 🔴 **imp** Entire imp module deprecated - Use importlib instead
- 🟡 **formatter** formatter module deprecated
- 🟡 **hmac** Default digestmod of MD5 deprecated - Will require explicit value in future
- 🟡 **html** HTMLParser strict argument deprecated
- 🟢 **plistlib** readPlist/writePlist/readPlistFromBytes/writePlistToBytes deprecated

## Breaking Changes

- 🟡 **File Descriptors** Non-inheritable by default - May affect some multiprocessing code
- 🟡 **Codecs** UTF-* codecs reject surrogates unless surrogatepass handler used
- 🟢 **importlib** Several finder and loader methods deprecated (but still work)
