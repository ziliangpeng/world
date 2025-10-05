# Python 3.2 Release Summary

**Released:** February 20, 2011
**Source:** [Official Python 3.2 Release Notes](https://docs.python.org/3/whatsnew/3.2.html)

## Overview

Python 3.2 represents a significant maturation of the Python 3 line, focusing on making Python 3 practical for production use. The release addresses major pain points from earlier Python 3 versions, particularly around bytes/text handling in the standard library, and introduces important infrastructure improvements. Major highlights include the concurrent.futures module for high-level parallelism, argparse for modern command-line parsing, stable ABI for C extensions, and proper handling of bytecode caching in multi-interpreter environments.

## Major Features

### PEP 3148: concurrent.futures Module

A new high-level interface for managing threads and processes, inspired by Java's java.util.concurrent package. The module provides ThreadPoolExecutor and ProcessPoolExecutor for launching and managing parallel calls:

```python
import concurrent.futures, shutil
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
    e.submit(shutil.copy, 'src1.txt', 'dest1.txt')
    e.submit(shutil.copy, 'src2.txt', 'dest2.txt')
    e.submit(shutil.copy, 'src3.txt', 'dest3.txt')
    e.submit(shutil.copy, 'src4.txt', 'dest4.txt')
```

The design uses Future objects to abstract features common to threads, processes, and remote procedure calls, providing status checks, timeouts, cancellations, callbacks, and result access.

### PEP 389: argparse Module

The new argparse module supersedes optparse, providing support for positional arguments, subcommands, required options, and better help generation:

```python
import argparse
parser = argparse.ArgumentParser(description='Manage servers')
parser.add_argument('action', choices=['deploy', 'start', 'stop'])
parser.add_argument('targets', metavar='HOSTNAME', nargs='+')
parser.add_argument('-u', '--user', required=True)
```

The module includes sophisticated features like subparsers for git-style subcommands, automatic help generation, and extensive validation capabilities.

### PEP 384: Stable ABI for C Extensions

Extension modules can now opt into a limited API that remains stable across Python feature releases. By defining Py_LIMITED_API, extension modules built for Python 3.2 will work with 3.3, 3.4, and beyond without recompilation. This eliminates the long-standing problem of rebuilding all extensions for each Python release, particularly important on Windows.

### PEP 3147: PYC Repository Directories

Python now stores bytecode cache files in __pycache__ directories with interpreter-specific names like mymodule.cpython-32.pyc. This solves "pyc fights" where multiple Python interpreters on the same system would overwrite each other's cached bytecode. The change also declutters source directories by separating .pyc files from .py files:

```python
>>> import collections
>>> collections.__cached__
'c:/py32/lib/__pycache__/collections.cpython-32.pyc'
```

### PEP 391: Dictionary-Based Logging Configuration

The logging module now supports configuration through dictionaries, enabling configuration from JSON or YAML files:

```python
import json, logging.config
with open('conf.json') as f:
    conf = json.load(f)
logging.config.dictConfig(conf)
```

This flexible approach supports incremental configuration and programmatic manipulation, complementing the existing file-based configuration.

## Standard Library Improvements

### Email Package Overhaul

The email package finally works correctly with bytes and mixed encodings in Python 3. New functions like message_from_bytes() and classes like BytesParser and BytesGenerator handle email messages in their native bytes format. The smtplib.SMTP class now accepts byte strings and has a send_message() method for sending Message objects directly.

### SSL and Security

Major improvements to SSL support:
- New SSLContext class for managing SSL configuration, certificates, and keys
- ssl.match_hostname() for server identity verification (HTTPS rules)
- Support for cipher specification
- Server Name Indication (SNI) extension support
- Access to OpenSSL version information

HTTP, FTP, IMAP, POP3, and NNTP modules all gained TLS/SSL improvements.

### functools Enhancements

Three new powerful decorators:
- lru_cache() for memoization with statistics tracking
- total_ordering() to fill in missing comparison methods
- cmp_to_key() for converting old-style comparison functions

```python
@functools.lru_cache(maxsize=300)
def get_phone_number(name):
    c = conn.cursor()
    c.execute('SELECT phonenumber FROM phonelist WHERE name=?', (name,))
    return c.fetchone()[0]

>>> get_phone_number.cache_info()
CacheInfo(hits=4805, misses=980, maxsize=300, currsize=300)
```

### Math Module Additions

Six new functions from C99:
- isfinite() for detecting NaN and Infinity
- expm1() for computing e**x-1 accurately for small x
- erf() and erfc() for error functions
- gamma() and lgamma() for gamma function and its logarithm

### Collections Improvements

- Counter gained regular subtraction via subtract() method (vs saturating -= operator)
- OrderedDict gained move_to_end() for resequencing entries
- deque gained count() and reverse() methods for better list compatibility

### Threading Barrier

New Barrier synchronization class for coordinating multiple threads at a common barrier point:

```python
from threading import Barrier, Thread

def get_votes(site):
    ballots = conduct_election(site)
    all_polls_closed.wait()  # wait until all polls close
    totals = summarize(ballots)
    publish(site, totals)

all_polls_closed = Barrier(len(sites))
for site in sites:
    Thread(target=get_votes, args=(site,)).start()
```

### Context Managers Everywhere

Many more classes now support context management for reliable cleanup: FTP, FTP_TLS, mmap, fileinput, os.popen(), subprocess.Popen(), TarFile, socket.create_connection(), and memoryview.

### Other Notable Additions

- datetime.timezone class for creating timezone-aware datetimes
- itertools.accumulate() for cumulative sums
- io.BytesIO.getbuffer() for zero-copy editing
- reprlib.recursive_repr() decorator for handling self-referential containers
- tempfile.TemporaryDirectory() context manager
- shutil archive operations (make_archive, unpack_archive)
- gzip.compress() and gzip.decompress() functions
- html.escape() function
- ast.literal_eval() now supports bytes and set literals

## Language Changes

### Behavioral Improvements

- hasattr() now only catches AttributeError, allowing other exceptions to propagate
- str() of float/complex now equals repr() for consistency
- memoryview objects support context management and have release() method
- Deleting names from local namespace is now allowed even if used in nested blocks
- Struct sequence types (os.stat, time.gmtime, sys.version_info) are now tuple subclasses
- range objects support index(), count(), slicing, and negative indices
- callable() builtin function resurrected from Python 2
- Import mechanism handles non-ASCII characters in path names

### Format Improvements

- format() character '#' now works with floats, complex, and Decimal: `format(12.34, '#5.0f')` produces '  12.'
- str.format_map() method accepts arbitrary mapping objects (defaultdict, ConfigParser, etc.)

### Other Changes

- Interpreter has -q (quiet) option to suppress version information
- PYTHONWARNINGS environment variable for controlling warnings
- New ResourceWarning for detecting unclosed files and resource leaks
- Two-digit year interpretation now triggers DeprecationWarning

## Performance and Implementation

### Improvements

- Better uniform distribution in random module's integer methods
- Unicode optimization removing wstr attribute (memory savings)
- Various internal optimizations across stdlib modules

### Bytecode and Compilation

- PYC files stored in __pycache__ with interpreter-specific names
- ABI version tags for .so files (PEP 3149)
- Improved bytecode generation and caching

## Deprecations and Removals

### Deprecated

- ElementTree getchildren() and getiterator() methods
- time.accept2dyear default behavior
- unittest method aliases (assertEquals, etc.) in favor of assertEqual
- unittest assertDictContainsSubset() (misimplemented)
- Various old-style APIs across multiple modules

### Behavioral Changes

- HTTP 0.9 simple responses no longer supported
- http.client strict parameter deprecated
- nntplib completely revamped with breaking changes for better bytes/text handling

## Migration Notes

Python 3.2 is a major step forward in making Python 3 production-ready, particularly for:

- Email processing (now handles bytes and mixed encodings correctly)
- Network protocols (proper SSL/TLS support throughout)
- Parallel programming (concurrent.futures provides high-level abstractions)
- C extension development (stable ABI eliminates rebuild requirements)

Most Python 3.1 code should work in 3.2 with minimal changes. The main impact areas are:

1. Import statements may need updating if code relied on specific .pyc locations
2. Email processing code benefits from updating to new bytes-oriented APIs
3. SSL/TLS code should migrate to new SSLContext-based APIs
4. Command-line tools can migrate from optparse to argparse
5. Code using nntplib requires updates due to API overhaul

## Key Takeaways

1. **Production-ready Python 3**: First Python 3 release with properly functioning email, mailbox, and network protocol modules for bytes/text handling
2. **Stable ABI breakthrough**: C extensions can target one Python version and work across multiple releases without recompilation
3. **Modern concurrency primitives**: concurrent.futures provides high-level thread and process pools inspired by Java
4. **Better multi-interpreter support**: PYC repository directories eliminate bytecode cache conflicts
5. **Comprehensive SSL/TLS**: Proper certificate validation and modern SSL features across all network modules
6. **Developer experience improvements**: argparse, lru_cache, better format strings, callable() returns
7. **Infrastructure for the future**: Stable ABI and PYC directories lay groundwork for better Python deployment

Python 3.2 represents the point where Python 3 became genuinely usable for production applications, addressing many practical concerns that hindered adoption of earlier 3.x releases.
