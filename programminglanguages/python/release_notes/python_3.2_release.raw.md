<div class="body" role="main">

<div id="what-s-new-in-python-3-2" class="section">

# What’s New In Python 3.2<a href="#what-s-new-in-python-3-2" class="headerlink" title="Link to this heading">¶</a>

Author<span class="colon">:</span>  
Raymond Hettinger

This article explains the new features in Python 3.2 as compared to 3.1. Python 3.2 was released on February 20, 2011. It focuses on a few highlights and gives a few examples. For full details, see the <a href="https://github.com/python/cpython/blob/076ca6c3c8df3030307e548d9be792ce3c1c6eea/Misc/NEWS" class="reference external">Misc/NEWS</a> file.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0392/" class="pep reference external"><strong>PEP 392</strong></a> - Python 3.2 Release Schedule

</div>

<div id="pep-384-defining-a-stable-abi" class="section">

## PEP 384: Defining a Stable ABI<a href="#pep-384-defining-a-stable-abi" class="headerlink" title="Link to this heading">¶</a>

In the past, extension modules built for one Python version were often not usable with other Python versions. Particularly on Windows, every feature release of Python required rebuilding all extension modules that one wanted to use. This requirement was the result of the free access to Python interpreter internals that extension modules could use.

With Python 3.2, an alternative approach becomes available: extension modules which restrict themselves to a limited API (by defining Py_LIMITED_API) cannot use many of the internals, but are constrained to a set of API functions that are promised to be stable for several releases. As a consequence, extension modules built for 3.2 in that mode will also work with 3.3, 3.4, and so on. Extension modules that make use of details of memory structures can still be built, but will need to be recompiled for every feature release.

<div class="admonition seealso">

See also

<span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0384/" class="pep reference external"><strong>PEP 384</strong></a> - Defining a Stable ABI  
PEP written by Martin von Löwis.

</div>

</div>

<div id="pep-389-argparse-command-line-parsing-module" class="section">

## PEP 389: Argparse Command Line Parsing Module<a href="#pep-389-argparse-command-line-parsing-module" class="headerlink" title="Link to this heading">¶</a>

A new module for command line parsing, <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a>, was introduced to overcome the limitations of <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library."><span class="pre"><code class="sourceCode python">optparse</code></span></a> which did not provide support for positional arguments (not just options), subcommands, required options and other common patterns of specifying and validating options.

This module has already had widespread success in the community as a third-party module. Being more fully featured than its predecessor, the <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> module is now the preferred module for command-line processing. The older module is still being kept available because of the substantial amount of legacy code that depends on it.

Here’s an annotated example parser showing features like limiting results to a set of choices, specifying a *metavar* in the help screen, validating that one or more positional arguments is present, and making a required option:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import argparse
    parser = argparse.ArgumentParser(
                description = 'Manage servers',         # main description for help
                epilog = 'Tested on Solaris and Linux') # displayed after help
    parser.add_argument('action',                       # argument name
                choices = ['deploy', 'start', 'stop'],  # three allowed values
                help = 'action on each target')         # help msg
    parser.add_argument('targets',
                metavar = 'HOSTNAME',                   # var name used in help msg
                nargs = '+',                            # require one or more targets
                help = 'url for target machines')       # help msg explanation
    parser.add_argument('-u', '--user',                 # -u or --user option
                required = True,                        # make it a required argument
                help = 'login as user')

</div>

</div>

Example of calling the parser on a command string:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> cmd = 'deploy sneezy.example.com sleepy.example.com -u skycaptain'
    >>> result = parser.parse_args(cmd.split())
    >>> result.action
    'deploy'
    >>> result.targets
    ['sneezy.example.com', 'sleepy.example.com']
    >>> result.user
    'skycaptain'

</div>

</div>

Example of the parser’s automatically generated help:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> parser.parse_args('-h'.split())

    usage: manage_cloud.py [-h] -u USER
                           {deploy,start,stop} HOSTNAME [HOSTNAME ...]

    Manage servers

    positional arguments:
      {deploy,start,stop}   action on each target
      HOSTNAME              url for target machines

    optional arguments:
      -h, --help            show this help message and exit
      -u USER, --user USER  login as user

    Tested on Solaris and Linux

</div>

</div>

An especially nice <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> feature is the ability to define subparsers, each with their own argument patterns and help displays:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import argparse
    parser = argparse.ArgumentParser(prog='HELM')
    subparsers = parser.add_subparsers()

    parser_l = subparsers.add_parser('launch', help='Launch Control')   # first subgroup
    parser_l.add_argument('-m', '--missiles', action='store_true')
    parser_l.add_argument('-t', '--torpedos', action='store_true')

    parser_m = subparsers.add_parser('move', help='Move Vessel',        # second subgroup
                                     aliases=('steer', 'turn'))         # equivalent names
    parser_m.add_argument('-c', '--course', type=int, required=True)
    parser_m.add_argument('-s', '--speed', type=int, default=0)

</div>

</div>

<div class="highlight-shell-session notranslate">

<div class="highlight">

    $ ./helm.py --help                         # top level help (launch and move)
    $ ./helm.py launch --help                  # help for launch options
    $ ./helm.py launch --missiles              # set missiles=True and torpedos=False
    $ ./helm.py steer --course 180 --speed 5   # set movement parameters

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0389/" class="pep reference external"><strong>PEP 389</strong></a> - New Command Line Parsing Module  
PEP written by Steven Bethard.

<a href="../howto/argparse-optparse.html#upgrading-optparse-code" class="reference internal"><span class="std std-ref">Migrating optparse code to argparse</span></a> for details on the differences from <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library."><span class="pre"><code class="sourceCode python">optparse</code></span></a>.

</div>

</div>

<div id="pep-391-dictionary-based-configuration-for-logging" class="section">

## PEP 391: Dictionary Based Configuration for Logging<a href="#pep-391-dictionary-based-configuration-for-logging" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> module provided two kinds of configuration, one style with function calls for each option or another style driven by an external file saved in a <a href="../library/configparser.html#module-configparser" class="reference internal" title="configparser: Configuration file parser."><span class="pre"><code class="sourceCode python">configparser</code></span></a> format. Those options did not provide the flexibility to create configurations from JSON or YAML files, nor did they support incremental configuration, which is needed for specifying logger options from a command line.

To support a more flexible style, the module now offers <a href="../library/logging.config.html#logging.config.dictConfig" class="reference internal" title="logging.config.dictConfig"><span class="pre"><code class="sourceCode python">logging.config.dictConfig()</code></span></a> for specifying logging configuration with plain Python dictionaries. The configuration options include formatters, handlers, filters, and loggers. Here’s a working example of a configuration dictionary:

<div class="highlight-python3 notranslate">

<div class="highlight">

    {"version": 1,
     "formatters": {"brief": {"format": "%(levelname)-8s: %(name)-15s: %(message)s"},
                    "full": {"format": "%(asctime)s %(name)-15s %(levelname)-8s %(message)s"}
                    },
     "handlers": {"console": {
                       "class": "logging.StreamHandler",
                       "formatter": "brief",
                       "level": "INFO",
                       "stream": "ext://sys.stdout"},
                  "console_priority": {
                       "class": "logging.StreamHandler",
                       "formatter": "full",
                       "level": "ERROR",
                       "stream": "ext://sys.stderr"}
                  },
     "root": {"level": "DEBUG", "handlers": ["console", "console_priority"]}}

</div>

</div>

If that dictionary is stored in a file called <span class="pre">`conf.json`</span>, it can be loaded and called with code like this:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import json, logging.config
    >>> with open('conf.json') as f:
    ...     conf = json.load(f)
    ...
    >>> logging.config.dictConfig(conf)
    >>> logging.info("Transaction completed normally")
    INFO    : root           : Transaction completed normally
    >>> logging.critical("Abnormal termination")
    2011-02-17 11:14:36,694 root            CRITICAL Abnormal termination

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0391/" class="pep reference external"><strong>PEP 391</strong></a> - Dictionary Based Configuration for Logging  
PEP written by Vinay Sajip.

</div>

</div>

<div id="pep-3148-the-concurrent-futures-module" class="section">

## PEP 3148: The <span class="pre">`concurrent.futures`</span> module<a href="#pep-3148-the-concurrent-futures-module" class="headerlink" title="Link to this heading">¶</a>

Code for creating and managing concurrency is being collected in a new top-level namespace, *concurrent*. Its first member is a *futures* package which provides a uniform high-level interface for managing threads and processes.

The design for <a href="../library/concurrent.futures.html#module-concurrent.futures" class="reference internal" title="concurrent.futures: Execute computations concurrently using threads or processes."><span class="pre"><code class="sourceCode python">concurrent.futures</code></span></a> was inspired by the *java.util.concurrent* package. In that model, a running call and its result are represented by a <a href="../library/concurrent.futures.html#concurrent.futures.Future" class="reference internal" title="concurrent.futures.Future"><span class="pre"><code class="sourceCode python">Future</code></span></a> object that abstracts features common to threads, processes, and remote procedure calls. That object supports status checks (running or done), timeouts, cancellations, adding callbacks, and access to results or exceptions.

The primary offering of the new module is a pair of executor classes for launching and managing calls. The goal of the executors is to make it easier to use existing tools for making parallel calls. They save the effort needed to setup a pool of resources, launch the calls, create a results queue, add time-out handling, and limit the total number of threads, processes, or remote procedure calls.

Ideally, each application should share a single executor across multiple components so that process and thread limits can be centrally managed. This solves the design challenge that arises when each component has its own competing strategy for resource management.

Both classes share a common interface with three methods: <a href="../library/concurrent.futures.html#concurrent.futures.Executor.submit" class="reference internal" title="concurrent.futures.Executor.submit"><span class="pre"><code class="sourceCode python">submit()</code></span></a> for scheduling a callable and returning a <a href="../library/concurrent.futures.html#concurrent.futures.Future" class="reference internal" title="concurrent.futures.Future"><span class="pre"><code class="sourceCode python">Future</code></span></a> object; <a href="../library/concurrent.futures.html#concurrent.futures.Executor.map" class="reference internal" title="concurrent.futures.Executor.map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> for scheduling many asynchronous calls at a time, and <a href="../library/concurrent.futures.html#concurrent.futures.Executor.shutdown" class="reference internal" title="concurrent.futures.Executor.shutdown"><span class="pre"><code class="sourceCode python">shutdown()</code></span></a> for freeing resources. The class is a <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> and can be used in a <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement to assure that resources are automatically released when currently pending futures are done executing.

A simple of example of <a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">ThreadPoolExecutor</code></span></a> is a launch of four parallel threads for copying files:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import concurrent.futures, shutil
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
        e.submit(shutil.copy, 'src1.txt', 'dest1.txt')
        e.submit(shutil.copy, 'src2.txt', 'dest2.txt')
        e.submit(shutil.copy, 'src3.txt', 'dest3.txt')
        e.submit(shutil.copy, 'src3.txt', 'dest4.txt')

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-4" class="target"></span><a href="https://peps.python.org/pep-3148/" class="pep reference external"><strong>PEP 3148</strong></a> - Futures – Execute Computations Asynchronously  
PEP written by Brian Quinlan.

<a href="../library/concurrent.futures.html#threadpoolexecutor-example" class="reference internal"><span class="std std-ref">Code for Threaded Parallel URL reads</span></a>, an example using threads to fetch multiple web pages in parallel.

<a href="../library/concurrent.futures.html#processpoolexecutor-example" class="reference internal"><span class="std std-ref">Code for computing prime numbers in parallel</span></a>, an example demonstrating <a href="../library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor" class="reference internal" title="concurrent.futures.ProcessPoolExecutor"><span class="pre"><code class="sourceCode python">ProcessPoolExecutor</code></span></a>.

</div>

</div>

<div id="pep-3147-pyc-repository-directories" class="section">

## PEP 3147: PYC Repository Directories<a href="#pep-3147-pyc-repository-directories" class="headerlink" title="Link to this heading">¶</a>

Python’s scheme for caching bytecode in *.pyc* files did not work well in environments with multiple Python interpreters. If one interpreter encountered a cached file created by another interpreter, it would recompile the source and overwrite the cached file, thus losing the benefits of caching.

The issue of “pyc fights” has become more pronounced as it has become commonplace for Linux distributions to ship with multiple versions of Python. These conflicts also arise with CPython alternatives such as Unladen Swallow.

To solve this problem, Python’s import machinery has been extended to use distinct filenames for each interpreter. Instead of Python 3.2 and Python 3.3 and Unladen Swallow each competing for a file called “mymodule.pyc”, they will now look for “mymodule.cpython-32.pyc”, “mymodule.cpython-33.pyc”, and “mymodule.unladen10.pyc”. And to prevent all of these new files from cluttering source directories, the *pyc* files are now collected in a “\_\_pycache\_\_” directory stored under the package directory.

Aside from the filenames and target directories, the new scheme has a few aspects that are visible to the programmer:

- Imported modules now have a <a href="../reference/datamodel.html#module.__cached__" class="reference internal" title="module.__cached__"><span class="pre"><code class="sourceCode python">__cached__</code></span></a> attribute which stores the name of the actual file that was imported:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import collections
      >>> collections.__cached__
      'c:/py32/lib/__pycache__/collections.cpython-32.pyc'

  </div>

  </div>

- The tag that is unique to each interpreter is accessible from the <span class="pre">`imp`</span> module:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import imp
      >>> imp.get_tag()
      'cpython-32'

  </div>

  </div>

- Scripts that try to deduce source filename from the imported file now need to be smarter. It is no longer sufficient to simply strip the “c” from a “.pyc” filename. Instead, use the new functions in the <span class="pre">`imp`</span> module:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> imp.source_from_cache('c:/py32/lib/__pycache__/collections.cpython-32.pyc')
      'c:/py32/lib/collections.py'
      >>> imp.cache_from_source('c:/py32/lib/collections.py')
      'c:/py32/lib/__pycache__/collections.cpython-32.pyc'

  </div>

  </div>

- The <a href="../library/py_compile.html#module-py_compile" class="reference internal" title="py_compile: Generate byte-code files from Python source files."><span class="pre"><code class="sourceCode python">py_compile</code></span></a> and <a href="../library/compileall.html#module-compileall" class="reference internal" title="compileall: Tools for byte-compiling all Python source files in a directory tree."><span class="pre"><code class="sourceCode python">compileall</code></span></a> modules have been updated to reflect the new naming convention and target directory. The command-line invocation of *compileall* has new options: <span class="pre">`-i`</span> for specifying a list of files and directories to compile and <span class="pre">`-b`</span> which causes bytecode files to be written to their legacy location rather than *\_\_pycache\_\_*.

- The <a href="../library/importlib.html#module-importlib.abc" class="reference internal" title="importlib.abc: Abstract base classes related to import"><span class="pre"><code class="sourceCode python">importlib.abc</code></span></a> module has been updated with new <a href="../glossary.html#term-abstract-base-class" class="reference internal"><span class="xref std std-term">abstract base classes</span></a> for loading bytecode files. The obsolete ABCs, <span class="pre">`PyLoader`</span> and <span class="pre">`PyPycLoader`</span>, have been deprecated (instructions on how to stay Python 3.1 compatible are included with the documentation).

<div class="admonition seealso">

See also

<span id="index-5" class="target"></span><a href="https://peps.python.org/pep-3147/" class="pep reference external"><strong>PEP 3147</strong></a> - PYC Repository Directories  
PEP written by Barry Warsaw.

</div>

</div>

<div id="pep-3149-abi-version-tagged-so-files" class="section">

## PEP 3149: ABI Version Tagged .so Files<a href="#pep-3149-abi-version-tagged-so-files" class="headerlink" title="Link to this heading">¶</a>

The PYC repository directory allows multiple bytecode cache files to be co-located. This PEP implements a similar mechanism for shared object files by giving them a common directory and distinct names for each version.

The common directory is “pyshared” and the file names are made distinct by identifying the Python implementation (such as CPython, PyPy, Jython, etc.), the major and minor version numbers, and optional build flags (such as “d” for debug, “m” for pymalloc, “u” for wide-unicode). For an arbitrary package “foo”, you may see these files when the distribution package is installed:

<div class="highlight-python3 notranslate">

<div class="highlight">

    /usr/share/pyshared/foo.cpython-32m.so
    /usr/share/pyshared/foo.cpython-33md.so

</div>

</div>

In Python itself, the tags are accessible from functions in the <a href="../library/sysconfig.html#module-sysconfig" class="reference internal" title="sysconfig: Python&#39;s configuration information"><span class="pre"><code class="sourceCode python">sysconfig</code></span></a> module:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import sysconfig
    >>> sysconfig.get_config_var('SOABI')       # find the version tag
    'cpython-32mu'
    >>> sysconfig.get_config_var('EXT_SUFFIX')  # find the full filename extension
    '.cpython-32mu.so'

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-6" class="target"></span><a href="https://peps.python.org/pep-3149/" class="pep reference external"><strong>PEP 3149</strong></a> - ABI Version Tagged .so Files  
PEP written by Barry Warsaw.

</div>

</div>

<div id="pep-3333-python-web-server-gateway-interface-v1-0-1" class="section">

## PEP 3333: Python Web Server Gateway Interface v1.0.1<a href="#pep-3333-python-web-server-gateway-interface-v1-0-1" class="headerlink" title="Link to this heading">¶</a>

This informational PEP clarifies how bytes/text issues are to be handled by the WSGI protocol. The challenge is that string handling in Python 3 is most conveniently handled with the <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> type even though the HTTP protocol is itself bytes oriented.

The PEP differentiates so-called *native strings* that are used for request/response headers and metadata versus *byte strings* which are used for the bodies of requests and responses.

The *native strings* are always of type <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> but are restricted to code points between *U+0000* through *U+00FF* which are translatable to bytes using *Latin-1* encoding. These strings are used for the keys and values in the environment dictionary and for response headers and statuses in the <span class="pre">`start_response()`</span> function. They must follow <span id="index-7" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc2616.html" class="rfc reference external"><strong>RFC 2616</strong></a> with respect to encoding. That is, they must either be *ISO-8859-1* characters or use <span id="index-8" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc2047.html" class="rfc reference external"><strong>RFC 2047</strong></a> MIME encoding.

For developers porting WSGI applications from Python 2, here are the salient points:

- If the app already used strings for headers in Python 2, no change is needed.

- If instead, the app encoded output headers or decoded input headers, then the headers will need to be re-encoded to Latin-1. For example, an output header encoded in utf-8 was using <span class="pre">`h.encode('utf-8')`</span> now needs to convert from bytes to native strings using <span class="pre">`h.encode('utf-8').decode('latin-1')`</span>.

- Values yielded by an application or sent using the <span class="pre">`write()`</span> method must be byte strings. The <span class="pre">`start_response()`</span> function and environ must use native strings. The two cannot be mixed.

For server implementers writing CGI-to-WSGI pathways or other CGI-style protocols, the users must to be able access the environment using native strings even though the underlying platform may have a different convention. To bridge this gap, the <a href="../library/wsgiref.html#module-wsgiref" class="reference internal" title="wsgiref: WSGI Utilities and Reference Implementation."><span class="pre"><code class="sourceCode python">wsgiref</code></span></a> module has a new function, <a href="../library/wsgiref.html#wsgiref.handlers.read_environ" class="reference internal" title="wsgiref.handlers.read_environ"><span class="pre"><code class="sourceCode python">wsgiref.handlers.read_environ()</code></span></a> for transcoding CGI variables from <a href="../library/os.html#os.environ" class="reference internal" title="os.environ"><span class="pre"><code class="sourceCode python">os.environ</code></span></a> into native strings and returning a new dictionary.

<div class="admonition seealso">

See also

<span id="index-9" class="target"></span><a href="https://peps.python.org/pep-3333/" class="pep reference external"><strong>PEP 3333</strong></a> - Python Web Server Gateway Interface v1.0.1  
PEP written by Phillip Eby.

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

Some smaller changes made to the core Python language are:

- String formatting for <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> and <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a> gained new capabilities for the format character **\#**. Previously, for integers in binary, octal, or hexadecimal, it caused the output to be prefixed with ‘0b’, ‘0o’, or ‘0x’ respectively. Now it can also handle floats, complex, and Decimal, causing the output to always have a decimal point even when no digits follow it.

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> format(20, '#o')
      '0o24'
      >>> format(12.34, '#5.0f')
      '  12.'

  </div>

  </div>

  (Suggested by Mark Dickinson and implemented by Eric Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7094" class="reference external">bpo-7094</a>.)

- There is also a new <a href="../library/stdtypes.html#str.format_map" class="reference internal" title="str.format_map"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.format_map()</code></span></a> method that extends the capabilities of the existing <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a> method by accepting arbitrary <a href="../glossary.html#term-mapping" class="reference internal"><span class="xref std std-term">mapping</span></a> objects. This new method makes it possible to use string formatting with any of Python’s many dictionary-like objects such as <a href="../library/collections.html#collections.defaultdict" class="reference internal" title="collections.defaultdict"><span class="pre"><code class="sourceCode python">defaultdict</code></span></a>, <a href="../library/shelve.html#shelve.Shelf" class="reference internal" title="shelve.Shelf"><span class="pre"><code class="sourceCode python">Shelf</code></span></a>, <a href="../library/configparser.html#configparser.ConfigParser" class="reference internal" title="configparser.ConfigParser"><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a>, or <a href="../library/dbm.html#module-dbm" class="reference internal" title="dbm: Interfaces to various Unix &quot;database&quot; formats."><span class="pre"><code class="sourceCode python">dbm</code></span></a>. It is also useful with custom <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> subclasses that normalize keys before look-up or that supply a <span class="pre">`__missing__()`</span> method for unknown keys:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> import shelve
      >>> d = shelve.open('tmp.shl')
      >>> 'The {project_name} status is {status} as of {date}'.format_map(d)
      'The testing project status is green as of February 15, 2011'

      >>> class LowerCasedDict(dict):
      ...     def __getitem__(self, key):
      ...         return dict.__getitem__(self, key.lower())
      ...
      >>> lcd = LowerCasedDict(part='widgets', quantity=10)
      >>> 'There are {QUANTITY} {Part} in stock'.format_map(lcd)
      'There are 10 widgets in stock'

      >>> class PlaceholderDict(dict):
      ...     def __missing__(self, key):
      ...         return '<{}>'.format(key)
      ...
      >>> 'Hello {name}, welcome to {location}'.format_map(PlaceholderDict())
      'Hello <name>, welcome to <location>'

  </div>

  </div>

> <div>
>
> (Suggested by Raymond Hettinger and implemented by Eric Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6081" class="reference external">bpo-6081</a>.)
>
> </div>

- The interpreter can now be started with a quiet option, <span class="pre">`-q`</span>, to prevent the copyright and version information from being displayed in the interactive mode. The option can be introspected using the <a href="../library/sys.html#sys.flags" class="reference internal" title="sys.flags"><span class="pre"><code class="sourceCode python">sys.flags</code></span></a> attribute:

  <div class="highlight-shell-session notranslate">

  <div class="highlight">

      $ python -q
      >>> sys.flags
      sys.flags(debug=0, division_warning=0, inspect=0, interactive=0,
      optimize=0, dont_write_bytecode=0, no_user_site=0, no_site=0,
      ignore_environment=0, verbose=0, bytes_warning=0, quiet=1)

  </div>

  </div>

  (Contributed by Marcin Wojdyr in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1772833" class="reference external">bpo-1772833</a>).

- The <a href="../library/functions.html#hasattr" class="reference internal" title="hasattr"><span class="pre"><code class="sourceCode python"><span class="bu">hasattr</span>()</code></span></a> function works by calling <a href="../library/functions.html#getattr" class="reference internal" title="getattr"><span class="pre"><code class="sourceCode python"><span class="bu">getattr</span>()</code></span></a> and detecting whether an exception is raised. This technique allows it to detect methods created dynamically by <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> or <a href="../reference/datamodel.html#object.__getattribute__" class="reference internal" title="object.__getattribute__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattribute__</span>()</code></span></a> which would otherwise be absent from the class dictionary. Formerly, *hasattr* would catch any exception, possibly masking genuine errors. Now, *hasattr* has been tightened to only catch <a href="../library/exceptions.html#AttributeError" class="reference internal" title="AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> and let other exceptions pass through:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> class A:
      ...     @property
      ...     def f(self):
      ...         return 1 // 0
      ...
      >>> a = A()
      >>> hasattr(a, 'f')
      Traceback (most recent call last):
        ...
      ZeroDivisionError: integer division or modulo by zero

  </div>

  </div>

  (Discovered by Yury Selivanov and fixed by Benjamin Peterson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9666" class="reference external">bpo-9666</a>.)

- The <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> of a float or complex number is now the same as its <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a>. Previously, the <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> form was shorter but that just caused confusion and is no longer needed now that the shortest possible <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> is displayed by default:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import math
      >>> repr(math.pi)
      '3.141592653589793'
      >>> str(math.pi)
      '3.141592653589793'

  </div>

  </div>

  (Proposed and implemented by Mark Dickinson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9337" class="reference external">bpo-9337</a>.)

- <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> objects now have a <a href="../library/stdtypes.html#memoryview.release" class="reference internal" title="memoryview.release"><span class="pre"><code class="sourceCode python">release()</code></span></a> method and they also now support the context management protocol. This allows timely release of any resources that were acquired when requesting a buffer from the original object.

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> with memoryview(b'abcdefgh') as v:
      ...     print(v.tolist())
      [97, 98, 99, 100, 101, 102, 103, 104]

  </div>

  </div>

  (Added by Antoine Pitrou; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9757" class="reference external">bpo-9757</a>.)

- Previously it was illegal to delete a name from the local namespace if it occurs as a free variable in a nested block:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      def outer(x):
          def inner():
              return x
          inner()
          del x

  </div>

  </div>

  This is now allowed. Remember that the target of an <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> clause is cleared, so this code which used to work with Python 2.6, raised a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> with Python 3.1 and now works again:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      def f():
          def print_error():
              print(e)
          try:
              something
          except Exception as e:
              print_error()
              # implicit "del e" here

  </div>

  </div>

  (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4617" class="reference external">bpo-4617</a>.)

- <a href="../c-api/tuple.html#struct-sequence-objects" class="reference internal"><span class="std std-ref">Struct sequence types</span></a> are now subclasses of tuple. This means that C structures like those returned by <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a>, <a href="../library/time.html#time.gmtime" class="reference internal" title="time.gmtime"><span class="pre"><code class="sourceCode python">time.gmtime()</code></span></a>, and <a href="../library/sys.html#sys.version_info" class="reference internal" title="sys.version_info"><span class="pre"><code class="sourceCode python">sys.version_info</code></span></a> now work like a <a href="../glossary.html#term-named-tuple" class="reference internal"><span class="xref std std-term">named tuple</span></a> and now work with functions and methods that expect a tuple as an argument. This is a big step forward in making the C structures as flexible as their pure Python counterparts:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import sys
      >>> isinstance(sys.version_info, tuple)
      True
      >>> 'Version %d.%d.%d %s(%d)' % sys.version_info
      'Version 3.2.0 final(0)'

  </div>

  </div>

  (Suggested by Arfrever Frehtes Taifersar Arahesis and implemented by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8413" class="reference external">bpo-8413</a>.)

- Warnings are now easier to control using the <span id="index-10" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONWARNINGS" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONWARNINGS</code></span></a> environment variable as an alternative to using <span class="pre">`-W`</span> at the command line:

  <div class="highlight-shell-session notranslate">

  <div class="highlight">

      $ export PYTHONWARNINGS='ignore::RuntimeWarning::,once::UnicodeWarning::'

  </div>

  </div>

  (Suggested by Barry Warsaw and implemented by Philip Jenvey in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7301" class="reference external">bpo-7301</a>.)

- A new warning category, <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a>, has been added. It is emitted when potential issues with resource consumption or cleanup are detected. It is silenced by default in normal release builds but can be enabled through the means provided by the <a href="../library/warnings.html#module-warnings" class="reference internal" title="warnings: Issue warning messages and control their disposition."><span class="pre"><code class="sourceCode python">warnings</code></span></a> module, or on the command line.

  A <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a> is issued at interpreter shutdown if the <a href="../library/gc.html#gc.garbage" class="reference internal" title="gc.garbage"><span class="pre"><code class="sourceCode python">gc.garbage</code></span></a> list isn’t empty, and if <a href="../library/gc.html#gc.DEBUG_UNCOLLECTABLE" class="reference internal" title="gc.DEBUG_UNCOLLECTABLE"><span class="pre"><code class="sourceCode python">gc.DEBUG_UNCOLLECTABLE</code></span></a> is set, all uncollectable objects are printed. This is meant to make the programmer aware that their code contains object finalization issues.

  A <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a> is also issued when a <a href="../glossary.html#term-file-object" class="reference internal"><span class="xref std std-term">file object</span></a> is destroyed without having been explicitly closed. While the deallocator for such object ensures it closes the underlying operating system resource (usually, a file descriptor), the delay in deallocating the object could produce various issues, especially under Windows. Here is an example of enabling the warning from the command line:

  <div class="highlight-shell-session notranslate">

  <div class="highlight">

      $ python -q -Wdefault
      >>> f = open("foo", "wb")
      >>> del f
      __main__:1: ResourceWarning: unclosed file <_io.BufferedWriter name='foo'>

  </div>

  </div>

  (Added by Antoine Pitrou and Georg Brandl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10093" class="reference external">bpo-10093</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=477863" class="reference external">bpo-477863</a>.)

- <a href="../library/stdtypes.html#range" class="reference internal" title="range"><span class="pre"><code class="sourceCode python"><span class="bu">range</span></code></span></a> objects now support *index* and *count* methods. This is part of an effort to make more objects fully implement the <a href="../library/collections.abc.html#collections.abc.Sequence" class="reference internal" title="collections.abc.Sequence"><span class="pre"><code class="sourceCode python">collections.Sequence</code></span></a> <a href="../glossary.html#term-abstract-base-class" class="reference internal"><span class="xref std std-term">abstract base class</span></a>. As a result, the language will have a more uniform API. In addition, <a href="../library/stdtypes.html#range" class="reference internal" title="range"><span class="pre"><code class="sourceCode python"><span class="bu">range</span></code></span></a> objects now support slicing and negative indices, even with values larger than <a href="../library/sys.html#sys.maxsize" class="reference internal" title="sys.maxsize"><span class="pre"><code class="sourceCode python">sys.maxsize</code></span></a>. This makes *range* more interoperable with lists:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> range(0, 100, 2).count(10)
      1
      >>> range(0, 100, 2).index(10)
      5
      >>> range(0, 100, 2)[5]
      10
      >>> range(0, 100, 2)[0:5]
      range(0, 10, 2)

  </div>

  </div>

  (Contributed by Daniel Stutzbach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9213" class="reference external">bpo-9213</a>, by Alexander Belopolsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2690" class="reference external">bpo-2690</a>, and by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10889" class="reference external">bpo-10889</a>.)

- The <a href="../library/functions.html#callable" class="reference internal" title="callable"><span class="pre"><code class="sourceCode python"><span class="bu">callable</span>()</code></span></a> builtin function from Py2.x was resurrected. It provides a concise, readable alternative to using an <a href="../glossary.html#term-abstract-base-class" class="reference internal"><span class="xref std std-term">abstract base class</span></a> in an expression like <span class="pre">`isinstance(x,`</span>` `<span class="pre">`collections.Callable)`</span>:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> callable(max)
      True
      >>> callable(20)
      False

  </div>

  </div>

  (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10518" class="reference external">bpo-10518</a>.)

- Python’s import mechanism can now load modules installed in directories with non-ASCII characters in the path name. This solved an aggravating problem with home directories for users with non-ASCII characters in their usernames.

> <div>
>
> (Required extensive work by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9425" class="reference external">bpo-9425</a>.)
>
> </div>

</div>

<div id="new-improved-and-deprecated-modules" class="section">

## New, Improved, and Deprecated Modules<a href="#new-improved-and-deprecated-modules" class="headerlink" title="Link to this heading">¶</a>

Python’s standard library has undergone significant maintenance efforts and quality improvements.

The biggest news for Python 3.2 is that the <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages."><span class="pre"><code class="sourceCode python">email</code></span></a> package, <a href="../library/mailbox.html#module-mailbox" class="reference internal" title="mailbox: Manipulate mailboxes in various formats"><span class="pre"><code class="sourceCode python">mailbox</code></span></a> module, and <span class="pre">`nntplib`</span> modules now work correctly with the bytes/text model in Python 3. For the first time, there is correct handling of messages with mixed encodings.

Throughout the standard library, there has been more careful attention to encodings and text versus bytes issues. In particular, interactions with the operating system are now better able to exchange non-ASCII data using the Windows MBCS encoding, locale-aware encodings, or UTF-8.

Another significant win is the addition of substantially better support for *SSL* connections and security certificates.

In addition, more classes now implement a <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> to support convenient and reliable resource clean-up using a <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement.

<div id="email" class="section">

### email<a href="#email" class="headerlink" title="Link to this heading">¶</a>

The usability of the <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages."><span class="pre"><code class="sourceCode python">email</code></span></a> package in Python 3 has been mostly fixed by the extensive efforts of R. David Murray. The problem was that emails are typically read and stored in the form of <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> rather than <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> text, and they may contain multiple encodings within a single email. So, the email package had to be extended to parse and generate email messages in bytes format.

- New functions <a href="../library/email.parser.html#email.message_from_bytes" class="reference internal" title="email.message_from_bytes"><span class="pre"><code class="sourceCode python">message_from_bytes()</code></span></a> and <a href="../library/email.parser.html#email.message_from_binary_file" class="reference internal" title="email.message_from_binary_file"><span class="pre"><code class="sourceCode python">message_from_binary_file()</code></span></a>, and new classes <a href="../library/email.parser.html#email.parser.BytesFeedParser" class="reference internal" title="email.parser.BytesFeedParser"><span class="pre"><code class="sourceCode python">BytesFeedParser</code></span></a> and <a href="../library/email.parser.html#email.parser.BytesParser" class="reference internal" title="email.parser.BytesParser"><span class="pre"><code class="sourceCode python">BytesParser</code></span></a> allow binary message data to be parsed into model objects.

- Given bytes input to the model, <a href="../library/email.compat32-message.html#email.message.Message.get_payload" class="reference internal" title="email.message.Message.get_payload"><span class="pre"><code class="sourceCode python">get_payload()</code></span></a> will by default decode a message body that has a *Content-Transfer-Encoding* of *8bit* using the charset specified in the MIME headers and return the resulting string.

- Given bytes input to the model, <a href="../library/email.generator.html#email.generator.Generator" class="reference internal" title="email.generator.Generator"><span class="pre"><code class="sourceCode python">Generator</code></span></a> will convert message bodies that have a *Content-Transfer-Encoding* of *8bit* to instead have a *7bit* *Content-Transfer-Encoding*.

  Headers with unencoded non-ASCII bytes are deemed to be <span id="index-11" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc2047.html" class="rfc reference external"><strong>RFC 2047</strong></a>-encoded using the *unknown-8bit* character set.

- A new class <a href="../library/email.generator.html#email.generator.BytesGenerator" class="reference internal" title="email.generator.BytesGenerator"><span class="pre"><code class="sourceCode python">BytesGenerator</code></span></a> produces bytes as output, preserving any unchanged non-ASCII data that was present in the input used to build the model, including message bodies with a *Content-Transfer-Encoding* of *8bit*.

- The <a href="../library/smtplib.html#module-smtplib" class="reference internal" title="smtplib: SMTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">smtplib</code></span></a> <a href="../library/smtplib.html#smtplib.SMTP" class="reference internal" title="smtplib.SMTP"><span class="pre"><code class="sourceCode python">SMTP</code></span></a> class now accepts a byte string for the *msg* argument to the <a href="../library/smtplib.html#smtplib.SMTP.sendmail" class="reference internal" title="smtplib.SMTP.sendmail"><span class="pre"><code class="sourceCode python">sendmail()</code></span></a> method, and a new method, <a href="../library/smtplib.html#smtplib.SMTP.send_message" class="reference internal" title="smtplib.SMTP.send_message"><span class="pre"><code class="sourceCode python">send_message()</code></span></a> accepts a <a href="../library/email.compat32-message.html#email.message.Message" class="reference internal" title="email.message.Message"><span class="pre"><code class="sourceCode python">Message</code></span></a> object and can optionally obtain the *from_addr* and *to_addrs* addresses directly from the object.

(Proposed and implemented by R. David Murray, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4661" class="reference external">bpo-4661</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10321" class="reference external">bpo-10321</a>.)

</div>

<div id="elementtree" class="section">

### elementtree<a href="#elementtree" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> package and its <span class="pre">`xml.etree.cElementTree`</span> counterpart have been updated to version 1.3.

Several new and useful functions and methods have been added:

- <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.fromstringlist" class="reference internal" title="xml.etree.ElementTree.fromstringlist"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.fromstringlist()</code></span></a> which builds an XML document from a sequence of fragments

- <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.register_namespace" class="reference internal" title="xml.etree.ElementTree.register_namespace"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.register_namespace()</code></span></a> for registering a global namespace prefix

- <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.tostringlist" class="reference internal" title="xml.etree.ElementTree.tostringlist"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.tostringlist()</code></span></a> for string representation including all sublists

- <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element.extend" class="reference internal" title="xml.etree.ElementTree.Element.extend"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.Element.extend()</code></span></a> for appending a sequence of zero or more elements

- <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element.iterfind" class="reference internal" title="xml.etree.ElementTree.Element.iterfind"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.Element.iterfind()</code></span></a> searches an element and subelements

- <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element.itertext" class="reference internal" title="xml.etree.ElementTree.Element.itertext"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.Element.itertext()</code></span></a> creates a text iterator over an element and its subelements

- <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.TreeBuilder.end" class="reference internal" title="xml.etree.ElementTree.TreeBuilder.end"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.TreeBuilder.end()</code></span></a> closes the current element

- <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.TreeBuilder.doctype" class="reference internal" title="xml.etree.ElementTree.TreeBuilder.doctype"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.TreeBuilder.doctype()</code></span></a> handles a doctype declaration

Two methods have been deprecated:

- <span class="pre">`xml.etree.ElementTree.getchildren()`</span> use <span class="pre">`list(elem)`</span> instead.

- <span class="pre">`xml.etree.ElementTree.getiterator()`</span> use <span class="pre">`Element.iter`</span> instead.

For details of the update, see <a href="https://web.archive.org/web/20200703234532/http://effbot.org/zone/elementtree-13-intro.htm" class="reference external">Introducing ElementTree</a> on Fredrik Lundh’s website.

(Contributed by Florent Xicluna and Fredrik Lundh, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6472" class="reference external">bpo-6472</a>.)

</div>

<div id="functools" class="section">

### functools<a href="#functools" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/functools.html#module-functools" class="reference internal" title="functools: Higher-order functions and operations on callable objects."><span class="pre"><code class="sourceCode python">functools</code></span></a> module includes a new decorator for caching function calls. <a href="../library/functools.html#functools.lru_cache" class="reference internal" title="functools.lru_cache"><span class="pre"><code class="sourceCode python">functools.lru_cache()</code></span></a> can save repeated queries to an external resource whenever the results are expected to be the same.

  For example, adding a caching decorator to a database query function can save database accesses for popular searches:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import functools
      >>> @functools.lru_cache(maxsize=300)
      ... def get_phone_number(name):
      ...     c = conn.cursor()
      ...     c.execute('SELECT phonenumber FROM phonelist WHERE name=?', (name,))
      ...     return c.fetchone()[0]

  </div>

  </div>

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> for name in user_requests:
      ...     get_phone_number(name)        # cached lookup

  </div>

  </div>

  To help with choosing an effective cache size, the wrapped function is instrumented for tracking cache statistics:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> get_phone_number.cache_info()
      CacheInfo(hits=4805, misses=980, maxsize=300, currsize=300)

  </div>

  </div>

  If the phonelist table gets updated, the outdated contents of the cache can be cleared with:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> get_phone_number.cache_clear()

  </div>

  </div>

  (Contributed by Raymond Hettinger and incorporating design ideas from Jim Baker, Miki Tebeka, and Nick Coghlan; see <a href="https://code.activestate.com/recipes/498245-lru-and-lfu-cache-decorators/" class="reference external">recipe 498245</a>, <a href="https://code.activestate.com/recipes/577479-simple-caching-decorator/" class="reference external">recipe 577479</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10586" class="reference external">bpo-10586</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10593" class="reference external">bpo-10593</a>.)

- The <a href="../library/functools.html#functools.wraps" class="reference internal" title="functools.wraps"><span class="pre"><code class="sourceCode python">functools.wraps()</code></span></a> decorator now adds a <span class="pre">`__wrapped__`</span> attribute pointing to the original callable function. This allows wrapped functions to be introspected. It also copies <a href="../reference/datamodel.html#function.__annotations__" class="reference internal" title="function.__annotations__"><span class="pre"><code class="sourceCode python">__annotations__</code></span></a> if defined. And now it also gracefully skips over missing attributes such as <a href="../reference/datamodel.html#function.__doc__" class="reference internal" title="function.__doc__"><span class="pre"><code class="sourceCode python">__doc__</code></span></a> which might not be defined for the wrapped callable.

  In the above example, the cache can be removed by recovering the original function:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> get_phone_number = get_phone_number.__wrapped__    # uncached function

  </div>

  </div>

  (By Nick Coghlan and Terrence Cole; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9567" class="reference external">bpo-9567</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3445" class="reference external">bpo-3445</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8814" class="reference external">bpo-8814</a>.)

- To help write classes with rich comparison methods, a new decorator <a href="../library/functools.html#functools.total_ordering" class="reference internal" title="functools.total_ordering"><span class="pre"><code class="sourceCode python">functools.total_ordering()</code></span></a> will use existing equality and inequality methods to fill in the remaining methods.

  For example, supplying *\_\_eq\_\_* and *\_\_lt\_\_* will enable <a href="../library/functools.html#functools.total_ordering" class="reference internal" title="functools.total_ordering"><span class="pre"><code class="sourceCode python">total_ordering()</code></span></a> to fill-in *\_\_le\_\_*, *\_\_gt\_\_* and *\_\_ge\_\_*:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      @total_ordering
      class Student:
          def __eq__(self, other):
              return ((self.lastname.lower(), self.firstname.lower()) ==
                      (other.lastname.lower(), other.firstname.lower()))

          def __lt__(self, other):
              return ((self.lastname.lower(), self.firstname.lower()) <
                      (other.lastname.lower(), other.firstname.lower()))

  </div>

  </div>

  With the *total_ordering* decorator, the remaining comparison methods are filled in automatically.

  (Contributed by Raymond Hettinger.)

- To aid in porting programs from Python 2, the <a href="../library/functools.html#functools.cmp_to_key" class="reference internal" title="functools.cmp_to_key"><span class="pre"><code class="sourceCode python">functools.cmp_to_key()</code></span></a> function converts an old-style comparison function to modern <a href="../glossary.html#term-key-function" class="reference internal"><span class="xref std std-term">key function</span></a>:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> # locale-aware sort order
      >>> sorted(iterable, key=cmp_to_key(locale.strcoll))

  </div>

  </div>

  For sorting examples and a brief sorting tutorial, see the <a href="https://wiki.python.org/moin/HowTo/Sorting/" class="reference external">Sorting HowTo</a> tutorial.

  (Contributed by Raymond Hettinger.)

</div>

<div id="itertools" class="section">

### itertools<a href="#itertools" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> module has a new <a href="../library/itertools.html#itertools.accumulate" class="reference internal" title="itertools.accumulate"><span class="pre"><code class="sourceCode python">accumulate()</code></span></a> function modeled on APL’s *scan* operator and Numpy’s *accumulate* function:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> from itertools import accumulate
      >>> list(accumulate([8, 2, 50]))
      [8, 10, 60]

  </div>

  </div>

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> prob_dist = [0.1, 0.4, 0.2, 0.3]
      >>> list(accumulate(prob_dist))      # cumulative probability distribution
      [0.1, 0.5, 0.7, 1.0]

  </div>

  </div>

  For an example using <a href="../library/itertools.html#itertools.accumulate" class="reference internal" title="itertools.accumulate"><span class="pre"><code class="sourceCode python">accumulate()</code></span></a>, see the <a href="../library/random.html#random-examples" class="reference internal"><span class="std std-ref">examples for the random module</span></a>.

  (Contributed by Raymond Hettinger and incorporating design suggestions from Mark Dickinson.)

</div>

<div id="collections" class="section">

### collections<a href="#collections" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/collections.html#collections.Counter" class="reference internal" title="collections.Counter"><span class="pre"><code class="sourceCode python">collections.Counter</code></span></a> class now has two forms of in-place subtraction, the existing *-=* operator for <a href="https://en.wikipedia.org/wiki/Saturation_arithmetic" class="reference external">saturating subtraction</a> and the new <a href="../library/collections.html#collections.Counter.subtract" class="reference internal" title="collections.Counter.subtract"><span class="pre"><code class="sourceCode python">subtract()</code></span></a> method for regular subtraction. The former is suitable for <a href="https://en.wikipedia.org/wiki/Multiset" class="reference external">multisets</a> which only have positive counts, and the latter is more suitable for use cases that allow negative counts:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> from collections import Counter
      >>> tally = Counter(dogs=5, cats=3)
      >>> tally -= Counter(dogs=2, cats=8)    # saturating subtraction
      >>> tally
      Counter({'dogs': 3})

  </div>

  </div>

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> tally = Counter(dogs=5, cats=3)
      >>> tally.subtract(dogs=2, cats=8)      # regular subtraction
      >>> tally
      Counter({'dogs': 3, 'cats': -5})

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- The <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">collections.OrderedDict</code></span></a> class has a new method <a href="../library/collections.html#collections.OrderedDict.move_to_end" class="reference internal" title="collections.OrderedDict.move_to_end"><span class="pre"><code class="sourceCode python">move_to_end()</code></span></a> which takes an existing key and moves it to either the first or last position in the ordered sequence.

  The default is to move an item to the last position. This is equivalent of renewing an entry with <span class="pre">`od[k]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`od.pop(k)`</span>.

  A fast move-to-end operation is useful for resequencing entries. For example, an ordered dictionary can be used to track order of access by aging entries from the oldest to the most recently accessed.

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> from collections import OrderedDict
      >>> d = OrderedDict.fromkeys(['a', 'b', 'X', 'd', 'e'])
      >>> list(d)
      ['a', 'b', 'X', 'd', 'e']
      >>> d.move_to_end('X')
      >>> list(d)
      ['a', 'b', 'd', 'e', 'X']

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- The <a href="../library/collections.html#collections.deque" class="reference internal" title="collections.deque"><span class="pre"><code class="sourceCode python">collections.deque</code></span></a> class grew two new methods <a href="../library/collections.html#collections.deque.count" class="reference internal" title="collections.deque.count"><span class="pre"><code class="sourceCode python">count()</code></span></a> and <a href="../library/collections.html#collections.deque.reverse" class="reference internal" title="collections.deque.reverse"><span class="pre"><code class="sourceCode python">reverse()</code></span></a> that make them more substitutable for <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a> objects:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> from collections import deque
      >>> d = deque('simsalabim')
      >>> d.count('s')
      2
      >>> d.reverse()
      >>> d
      deque(['m', 'i', 'b', 'a', 'l', 'a', 's', 'm', 'i', 's'])

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

</div>

<div id="threading" class="section">

### threading<a href="#threading" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a> module has a new <a href="../library/threading.html#threading.Barrier" class="reference internal" title="threading.Barrier"><span class="pre"><code class="sourceCode python">Barrier</code></span></a> synchronization class for making multiple threads wait until all of them have reached a common barrier point. Barriers are useful for making sure that a task with multiple preconditions does not run until all of the predecessor tasks are complete.

Barriers can work with an arbitrary number of threads. This is a generalization of a <a href="https://en.wikipedia.org/wiki/Synchronous_rendezvous" class="reference external">Rendezvous</a> which is defined for only two threads.

Implemented as a two-phase cyclic barrier, <a href="../library/threading.html#threading.Barrier" class="reference internal" title="threading.Barrier"><span class="pre"><code class="sourceCode python">Barrier</code></span></a> objects are suitable for use in loops. The separate *filling* and *draining* phases assure that all threads get released (drained) before any one of them can loop back and re-enter the barrier. The barrier fully resets after each cycle.

Example of using barriers:

<div class="highlight-python3 notranslate">

<div class="highlight">

    from threading import Barrier, Thread

    def get_votes(site):
        ballots = conduct_election(site)
        all_polls_closed.wait()        # do not count until all polls are closed
        totals = summarize(ballots)
        publish(site, totals)

    all_polls_closed = Barrier(len(sites))
    for site in sites:
        Thread(target=get_votes, args=(site,)).start()

</div>

</div>

In this example, the barrier enforces a rule that votes cannot be counted at any polling site until all polls are closed. Notice how a solution with a barrier is similar to one with <a href="../library/threading.html#threading.Thread.join" class="reference internal" title="threading.Thread.join"><span class="pre"><code class="sourceCode python">threading.Thread.join()</code></span></a>, but the threads stay alive and continue to do work (summarizing ballots) after the barrier point is crossed.

If any of the predecessor tasks can hang or be delayed, a barrier can be created with an optional *timeout* parameter. Then if the timeout period elapses before all the predecessor tasks reach the barrier point, all waiting threads are released and a <a href="../library/threading.html#threading.BrokenBarrierError" class="reference internal" title="threading.BrokenBarrierError"><span class="pre"><code class="sourceCode python">BrokenBarrierError</code></span></a> exception is raised:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def get_votes(site):
        ballots = conduct_election(site)
        try:
            all_polls_closed.wait(timeout=midnight - time.now())
        except BrokenBarrierError:
            lockbox = seal_ballots(ballots)
            queue.put(lockbox)
        else:
            totals = summarize(ballots)
            publish(site, totals)

</div>

</div>

In this example, the barrier enforces a more robust rule. If some election sites do not finish before midnight, the barrier times-out and the ballots are sealed and deposited in a queue for later handling.

See <a href="https://osl.cs.illinois.edu/media/papers/karmani-2009-barrier_synchronization_pattern.pdf" class="reference external">Barrier Synchronization Patterns</a> for more examples of how barriers can be used in parallel computing. Also, there is a simple but thorough explanation of barriers in <a href="https://greenteapress.com/semaphores/LittleBookOfSemaphores.pdf" class="reference external">The Little Book of Semaphores</a>, *section 3.6*.

(Contributed by Kristján Valur Jónsson with an API review by Jeffrey Yasskin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8777" class="reference external">bpo-8777</a>.)

</div>

<div id="datetime-and-time" class="section">

### datetime and time<a href="#datetime-and-time" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> module has a new type <a href="../library/datetime.html#datetime.timezone" class="reference internal" title="datetime.timezone"><span class="pre"><code class="sourceCode python">timezone</code></span></a> that implements the <a href="../library/datetime.html#datetime.tzinfo" class="reference internal" title="datetime.tzinfo"><span class="pre"><code class="sourceCode python">tzinfo</code></span></a> interface by returning a fixed UTC offset and timezone name. This makes it easier to create timezone-aware datetime objects:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> from datetime import datetime, timezone

      >>> datetime.now(timezone.utc)
      datetime.datetime(2010, 12, 8, 21, 4, 2, 923754, tzinfo=datetime.timezone.utc)

      >>> datetime.strptime("01/01/2000 12:00 +0000", "%m/%d/%Y %H:%M %z")
      datetime.datetime(2000, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)

  </div>

  </div>

- Also, <a href="../library/datetime.html#datetime.timedelta" class="reference internal" title="datetime.timedelta"><span class="pre"><code class="sourceCode python">timedelta</code></span></a> objects can now be multiplied by <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> and divided by <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> and <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> objects. And <a href="../library/datetime.html#datetime.timedelta" class="reference internal" title="datetime.timedelta"><span class="pre"><code class="sourceCode python">timedelta</code></span></a> objects can now divide one another.

- The <a href="../library/datetime.html#datetime.date.strftime" class="reference internal" title="datetime.date.strftime"><span class="pre"><code class="sourceCode python">datetime.date.strftime()</code></span></a> method is no longer restricted to years after 1900. The new supported year range is from 1000 to 9999 inclusive.

- Whenever a two-digit year is used in a time tuple, the interpretation has been governed by <span class="pre">`time.accept2dyear`</span>. The default is <span class="pre">`True`</span> which means that for a two-digit year, the century is guessed according to the POSIX rules governing the <span class="pre">`%y`</span> strptime format.

  Starting with Py3.2, use of the century guessing heuristic will emit a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. Instead, it is recommended that <span class="pre">`time.accept2dyear`</span> be set to <span class="pre">`False`</span> so that large date ranges can be used without guesswork:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> import time, warnings
      >>> warnings.resetwarnings()      # remove the default warning filters

      >>> time.accept2dyear = True      # guess whether 11 means 11 or 2011
      >>> time.asctime((11, 1, 1, 12, 34, 56, 4, 1, 0))
      Warning (from warnings module):
        ...
      DeprecationWarning: Century info guessed for a 2-digit year.
      'Fri Jan  1 12:34:56 2011'

      >>> time.accept2dyear = False     # use the full range of allowable dates
      >>> time.asctime((11, 1, 1, 12, 34, 56, 4, 1, 0))
      'Fri Jan  1 12:34:56 11'

  </div>

  </div>

  Several functions now have significantly expanded date ranges. When <span class="pre">`time.accept2dyear`</span> is false, the <a href="../library/time.html#time.asctime" class="reference internal" title="time.asctime"><span class="pre"><code class="sourceCode python">time.asctime()</code></span></a> function will accept any year that fits in a C int, while the <a href="../library/time.html#time.mktime" class="reference internal" title="time.mktime"><span class="pre"><code class="sourceCode python">time.mktime()</code></span></a> and <a href="../library/time.html#time.strftime" class="reference internal" title="time.strftime"><span class="pre"><code class="sourceCode python">time.strftime()</code></span></a> functions will accept the full range supported by the corresponding operating system functions.

(Contributed by Alexander Belopolsky and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1289118" class="reference external">bpo-1289118</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5094" class="reference external">bpo-5094</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6641" class="reference external">bpo-6641</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2706" class="reference external">bpo-2706</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1777412" class="reference external">bpo-1777412</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8013" class="reference external">bpo-8013</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10827" class="reference external">bpo-10827</a>.)

</div>

<div id="math" class="section">

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> module has been updated with six new functions inspired by the C99 standard.

The <a href="../library/math.html#math.isfinite" class="reference internal" title="math.isfinite"><span class="pre"><code class="sourceCode python">isfinite()</code></span></a> function provides a reliable and fast way to detect special values. It returns <span class="pre">`True`</span> for regular numbers and <span class="pre">`False`</span> for *Nan* or *Infinity*:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> from math import isfinite
    >>> [isfinite(x) for x in (123, 4.56, float('Nan'), float('Inf'))]
    [True, True, False, False]

</div>

</div>

The <a href="../library/math.html#math.expm1" class="reference internal" title="math.expm1"><span class="pre"><code class="sourceCode python">expm1()</code></span></a> function computes <span class="pre">`e**x-1`</span> for small values of *x* without incurring the loss of precision that usually accompanies the subtraction of nearly equal quantities:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> from math import expm1
    >>> expm1(0.013671875)   # more accurate way to compute e**x-1 for a small x
    0.013765762467652909

</div>

</div>

The <a href="../library/math.html#math.erf" class="reference internal" title="math.erf"><span class="pre"><code class="sourceCode python">erf()</code></span></a> function computes a probability integral or <a href="https://en.wikipedia.org/wiki/Error_function" class="reference external">Gaussian error function</a>. The complementary error function, <a href="../library/math.html#math.erfc" class="reference internal" title="math.erfc"><span class="pre"><code class="sourceCode python">erfc()</code></span></a>, is <span class="pre">`1`</span>` `<span class="pre">`-`</span>` `<span class="pre">`erf(x)`</span>:

<div class="highlight-pycon notranslate">

<div class="highlight">

    >>> from math import erf, erfc, sqrt
    >>> erf(1.0/sqrt(2.0))   # portion of normal distribution within 1 standard deviation
    0.682689492137086
    >>> erfc(1.0/sqrt(2.0))  # portion of normal distribution outside 1 standard deviation
    0.31731050786291404
    >>> erf(1.0/sqrt(2.0)) + erfc(1.0/sqrt(2.0))
    1.0

</div>

</div>

The <a href="../library/math.html#math.gamma" class="reference internal" title="math.gamma"><span class="pre"><code class="sourceCode python">gamma()</code></span></a> function is a continuous extension of the factorial function. See <a href="https://en.wikipedia.org/wiki/Gamma_function" class="reference external">https://en.wikipedia.org/wiki/Gamma_function</a> for details. Because the function is related to factorials, it grows large even for small values of *x*, so there is also a <a href="../library/math.html#math.lgamma" class="reference internal" title="math.lgamma"><span class="pre"><code class="sourceCode python">lgamma()</code></span></a> function for computing the natural logarithm of the gamma function:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> from math import gamma, lgamma
    >>> gamma(7.0)           # six factorial
    720.0
    >>> lgamma(801.0)        # log(800 factorial)
    4551.950730698041

</div>

</div>

(Contributed by Mark Dickinson.)

</div>

<div id="abc" class="section">

### abc<a href="#abc" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/abc.html#module-abc" class="reference internal" title="abc: Abstract base classes according to :pep:`3119`."><span class="pre"><code class="sourceCode python">abc</code></span></a> module now supports <a href="../library/abc.html#abc.abstractclassmethod" class="reference internal" title="abc.abstractclassmethod"><span class="pre"><code class="sourceCode python">abstractclassmethod()</code></span></a> and <a href="../library/abc.html#abc.abstractstaticmethod" class="reference internal" title="abc.abstractstaticmethod"><span class="pre"><code class="sourceCode python">abstractstaticmethod()</code></span></a>.

These tools make it possible to define an <a href="../glossary.html#term-abstract-base-class" class="reference internal"><span class="xref std std-term">abstract base class</span></a> that requires a particular <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span>()</code></span></a> or <a href="../library/functions.html#staticmethod" class="reference internal" title="staticmethod"><span class="pre"><code class="sourceCode python"><span class="bu">staticmethod</span>()</code></span></a> to be implemented:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class Temperature(metaclass=abc.ABCMeta):
        @abc.abstractclassmethod
        def from_fahrenheit(cls, t):
            ...
        @abc.abstractclassmethod
        def from_celsius(cls, t):
            ...

</div>

</div>

(Patch submitted by Daniel Urban; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5867" class="reference external">bpo-5867</a>.)

</div>

<div id="io" class="section">

### io<a href="#io" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/io.html#io.BytesIO" class="reference internal" title="io.BytesIO"><span class="pre"><code class="sourceCode python">io.BytesIO</code></span></a> has a new method, <a href="../library/io.html#io.BytesIO.getbuffer" class="reference internal" title="io.BytesIO.getbuffer"><span class="pre"><code class="sourceCode python">getbuffer()</code></span></a>, which provides functionality similar to <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span>()</code></span></a>. It creates an editable view of the data without making a copy. The buffer’s random access and support for slice notation are well-suited to in-place editing:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> REC_LEN, LOC_START, LOC_LEN = 34, 7, 11

    >>> def change_location(buffer, record_number, location):
    ...     start = record_number * REC_LEN + LOC_START
    ...     buffer[start: start+LOC_LEN] = location

    >>> import io

    >>> byte_stream = io.BytesIO(
    ...     b'G3805  storeroom  Main chassis    '
    ...     b'X7899  shipping   Reserve cog     '
    ...     b'L6988  receiving  Primary sprocket'
    ... )
    >>> buffer = byte_stream.getbuffer()
    >>> change_location(buffer, 1, b'warehouse  ')
    >>> change_location(buffer, 0, b'showroom   ')
    >>> print(byte_stream.getvalue())
    b'G3805  showroom   Main chassis    '
    b'X7899  warehouse  Reserve cog     '
    b'L6988  receiving  Primary sprocket'

</div>

</div>

(Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5506" class="reference external">bpo-5506</a>.)

</div>

<div id="reprlib" class="section">

### reprlib<a href="#reprlib" class="headerlink" title="Link to this heading">¶</a>

When writing a <a href="../reference/datamodel.html#object.__repr__" class="reference internal" title="object.__repr__"><span class="pre"><code class="sourceCode python"><span class="fu">__repr__</span>()</code></span></a> method for a custom container, it is easy to forget to handle the case where a member refers back to the container itself. Python’s builtin objects such as <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a> and <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a> handle self-reference by displaying “…” in the recursive part of the representation string.

To help write such <a href="../reference/datamodel.html#object.__repr__" class="reference internal" title="object.__repr__"><span class="pre"><code class="sourceCode python"><span class="fu">__repr__</span>()</code></span></a> methods, the <a href="../library/reprlib.html#module-reprlib" class="reference internal" title="reprlib: Alternate repr() implementation with size limits."><span class="pre"><code class="sourceCode python">reprlib</code></span></a> module has a new decorator, <a href="../library/reprlib.html#reprlib.recursive_repr" class="reference internal" title="reprlib.recursive_repr"><span class="pre"><code class="sourceCode python">recursive_repr()</code></span></a>, for detecting recursive calls to <span class="pre">`__repr__()`</span> and substituting a placeholder string instead:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> class MyList(list):
    ...     @recursive_repr()
    ...     def __repr__(self):
    ...         return '<' + '|'.join(map(repr, self)) + '>'
    ...
    >>> m = MyList('abc')
    >>> m.append(m)
    >>> m.append('x')
    >>> print(m)
    <'a'|'b'|'c'|...|'x'>

</div>

</div>

(Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9826" class="reference external">bpo-9826</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9840" class="reference external">bpo-9840</a>.)

</div>

<div id="logging" class="section">

### logging<a href="#logging" class="headerlink" title="Link to this heading">¶</a>

In addition to dictionary-based configuration described above, the <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> package has many other improvements.

The logging documentation has been augmented by a <a href="../howto/logging.html#logging-basic-tutorial" class="reference internal"><span class="std std-ref">basic tutorial</span></a>, an <a href="../howto/logging.html#logging-advanced-tutorial" class="reference internal"><span class="std std-ref">advanced tutorial</span></a>, and a <a href="../howto/logging-cookbook.html#logging-cookbook" class="reference internal"><span class="std std-ref">cookbook</span></a> of logging recipes. These documents are the fastest way to learn about logging.

The <a href="../library/logging.html#logging.basicConfig" class="reference internal" title="logging.basicConfig"><span class="pre"><code class="sourceCode python">logging.basicConfig()</code></span></a> set-up function gained a *style* argument to support three different types of string formatting. It defaults to “%” for traditional %-formatting, can be set to “{” for the new <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a> style, or can be set to “\$” for the shell-style formatting provided by <a href="../library/string.html#string.Template" class="reference internal" title="string.Template"><span class="pre"><code class="sourceCode python">string.Template</code></span></a>. The following three configurations are equivalent:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from logging import basicConfig
    >>> basicConfig(style='%', format="%(name)s -> %(levelname)s: %(message)s")
    >>> basicConfig(style='{', format="{name} -> {levelname} {message}")
    >>> basicConfig(style='$', format="$name -> $levelname: $message")

</div>

</div>

If no configuration is set-up before a logging event occurs, there is now a default configuration using a <a href="../library/logging.handlers.html#logging.StreamHandler" class="reference internal" title="logging.StreamHandler"><span class="pre"><code class="sourceCode python">StreamHandler</code></span></a> directed to <a href="../library/sys.html#sys.stderr" class="reference internal" title="sys.stderr"><span class="pre"><code class="sourceCode python">sys.stderr</code></span></a> for events of <span class="pre">`WARNING`</span> level or higher. Formerly, an event occurring before a configuration was set-up would either raise an exception or silently drop the event depending on the value of <a href="../library/logging.html#logging.raiseExceptions" class="reference internal" title="logging.raiseExceptions"><span class="pre"><code class="sourceCode python">logging.raiseExceptions</code></span></a>. The new default handler is stored in <a href="../library/logging.html#logging.lastResort" class="reference internal" title="logging.lastResort"><span class="pre"><code class="sourceCode python">logging.lastResort</code></span></a>.

The use of filters has been simplified. Instead of creating a <a href="../library/logging.html#logging.Filter" class="reference internal" title="logging.Filter"><span class="pre"><code class="sourceCode python">Filter</code></span></a> object, the predicate can be any Python callable that returns <span class="pre">`True`</span> or <span class="pre">`False`</span>.

There were a number of other improvements that add flexibility and simplify configuration. See the module documentation for a full listing of changes in Python 3.2.

</div>

<div id="csv" class="section">

### csv<a href="#csv" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/csv.html#module-csv" class="reference internal" title="csv: Write and read tabular data to and from delimited files."><span class="pre"><code class="sourceCode python">csv</code></span></a> module now supports a new dialect, <a href="../library/csv.html#csv.unix_dialect" class="reference internal" title="csv.unix_dialect"><span class="pre"><code class="sourceCode python">unix_dialect</code></span></a>, which applies quoting for all fields and a traditional Unix style with <span class="pre">`'\n'`</span> as the line terminator. The registered dialect name is <span class="pre">`unix`</span>.

The <a href="../library/csv.html#csv.DictWriter" class="reference internal" title="csv.DictWriter"><span class="pre"><code class="sourceCode python">csv.DictWriter</code></span></a> has a new method, <a href="../library/csv.html#csv.DictWriter.writeheader" class="reference internal" title="csv.DictWriter.writeheader"><span class="pre"><code class="sourceCode python">writeheader()</code></span></a> for writing-out an initial row to document the field names:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import csv, sys
    >>> w = csv.DictWriter(sys.stdout, ['name', 'dept'], dialect='unix')
    >>> w.writeheader()
    "name","dept"
    >>> w.writerows([
    ...     {'name': 'tom', 'dept': 'accounting'},
    ...     {'name': 'susan', 'dept': 'Salesl'}])
    "tom","accounting"
    "susan","sales"

</div>

</div>

(New dialect suggested by Jay Talbot in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5975" class="reference external">bpo-5975</a>, and the new method suggested by Ed Abraham in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1537721" class="reference external">bpo-1537721</a>.)

</div>

<div id="contextlib" class="section">

### contextlib<a href="#contextlib" class="headerlink" title="Link to this heading">¶</a>

There is a new and slightly mind-blowing tool <a href="../library/contextlib.html#contextlib.ContextDecorator" class="reference internal" title="contextlib.ContextDecorator"><span class="pre"><code class="sourceCode python">ContextDecorator</code></span></a> that is helpful for creating a <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> that does double duty as a function decorator.

As a convenience, this new functionality is used by <a href="../library/contextlib.html#contextlib.contextmanager" class="reference internal" title="contextlib.contextmanager"><span class="pre"><code class="sourceCode python">contextmanager()</code></span></a> so that no extra effort is needed to support both roles.

The basic idea is that both context managers and function decorators can be used for pre-action and post-action wrappers. Context managers wrap a group of statements using a <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement, and function decorators wrap a group of statements enclosed in a function. So, occasionally there is a need to write a pre-action or post-action wrapper that can be used in either role.

For example, it is sometimes useful to wrap functions or groups of statements with a logger that can track the time of entry and time of exit. Rather than writing both a function decorator and a context manager for the task, the <a href="../library/contextlib.html#contextlib.contextmanager" class="reference internal" title="contextlib.contextmanager"><span class="pre"><code class="sourceCode python">contextmanager()</code></span></a> provides both capabilities in a single definition:

<div class="highlight-python3 notranslate">

<div class="highlight">

    from contextlib import contextmanager
    import logging

    logging.basicConfig(level=logging.INFO)

    @contextmanager
    def track_entry_and_exit(name):
        logging.info('Entering: %s', name)
        yield
        logging.info('Exiting: %s', name)

</div>

</div>

Formerly, this would have only been usable as a context manager:

<div class="highlight-python3 notranslate">

<div class="highlight">

    with track_entry_and_exit('widget loader'):
        print('Some time consuming activity goes here')
        load_widget()

</div>

</div>

Now, it can be used as a decorator as well:

<div class="highlight-python3 notranslate">

<div class="highlight">

    @track_entry_and_exit('widget loader')
    def activity():
        print('Some time consuming activity goes here')
        load_widget()

</div>

</div>

Trying to fulfill two roles at once places some limitations on the technique. Context managers normally have the flexibility to return an argument usable by a <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement, but there is no parallel for function decorators.

In the above example, there is not a clean way for the *track_entry_and_exit* context manager to return a logging instance for use in the body of enclosed statements.

(Contributed by Michael Foord in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9110" class="reference external">bpo-9110</a>.)

</div>

<div id="decimal-and-fractions" class="section">

### decimal and fractions<a href="#decimal-and-fractions" class="headerlink" title="Link to this heading">¶</a>

Mark Dickinson crafted an elegant and efficient scheme for assuring that different numeric datatypes will have the same hash value whenever their actual values are equal (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8188" class="reference external">bpo-8188</a>):

<div class="highlight-python3 notranslate">

<div class="highlight">

    assert hash(Fraction(3, 2)) == hash(1.5) == \
           hash(Decimal("1.5")) == hash(complex(1.5, 0))

</div>

</div>

Some of the hashing details are exposed through a new attribute, <a href="../library/sys.html#sys.hash_info" class="reference internal" title="sys.hash_info"><span class="pre"><code class="sourceCode python">sys.hash_info</code></span></a>, which describes the bit width of the hash value, the prime modulus, the hash values for *infinity* and *nan*, and the multiplier used for the imaginary part of a number:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> sys.hash_info
    sys.hash_info(width=64, modulus=2305843009213693951, inf=314159, nan=0, imag=1000003)

</div>

</div>

An early decision to limit the interoperability of various numeric types has been relaxed. It is still unsupported (and ill-advised) to have implicit mixing in arithmetic expressions such as <span class="pre">`Decimal('1.1')`</span>` `<span class="pre">`+`</span>` `<span class="pre">`float('1.1')`</span> because the latter loses information in the process of constructing the binary float. However, since existing floating-point value can be converted losslessly to either a decimal or rational representation, it makes sense to add them to the constructor and to support mixed-type comparisons.

- The <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">decimal.Decimal</code></span></a> constructor now accepts <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> objects directly so there in no longer a need to use the <a href="../library/decimal.html#decimal.Decimal.from_float" class="reference internal" title="decimal.Decimal.from_float"><span class="pre"><code class="sourceCode python">from_float()</code></span></a> method (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8257" class="reference external">bpo-8257</a>).

- Mixed type comparisons are now fully supported so that <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> objects can be directly compared with <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> and <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">fractions.Fraction</code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2531" class="reference external">bpo-2531</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8188" class="reference external">bpo-8188</a>).

Similar changes were made to <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">fractions.Fraction</code></span></a> so that the <a href="../library/fractions.html#fractions.Fraction.from_float" class="reference internal" title="fractions.Fraction.from_float"><span class="pre"><code class="sourceCode python">from_float()</code></span></a> and <a href="../library/fractions.html#fractions.Fraction.from_decimal" class="reference internal" title="fractions.Fraction.from_decimal"><span class="pre"><code class="sourceCode python">from_decimal()</code></span></a> methods are no longer needed (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8294" class="reference external">bpo-8294</a>):

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> from decimal import Decimal
    >>> from fractions import Fraction
    >>> Decimal(1.1)
    Decimal('1.100000000000000088817841970012523233890533447265625')
    >>> Fraction(1.1)
    Fraction(2476979795053773, 2251799813685248)

</div>

</div>

Another useful change for the <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> module is that the <a href="../library/decimal.html#decimal.Context.clamp" class="reference internal" title="decimal.Context.clamp"><span class="pre"><code class="sourceCode python">Context.clamp</code></span></a> attribute is now public. This is useful in creating contexts that correspond to the decimal interchange formats specified in IEEE 754 (see <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8540" class="reference external">bpo-8540</a>).

(Contributed by Mark Dickinson and Raymond Hettinger.)

</div>

<div id="ftp" class="section">

### ftp<a href="#ftp" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/ftplib.html#ftplib.FTP" class="reference internal" title="ftplib.FTP"><span class="pre"><code class="sourceCode python">ftplib.FTP</code></span></a> class now supports the context management protocol to unconditionally consume <a href="../library/socket.html#socket.error" class="reference internal" title="socket.error"><span class="pre"><code class="sourceCode python">socket.error</code></span></a> exceptions and to close the FTP connection when done:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from ftplib import FTP
    >>> with FTP("ftp1.at.proftpd.org") as ftp:
            ftp.login()
            ftp.dir()

    '230 Anonymous login ok, restrictions apply.'
    dr-xr-xr-x   9 ftp      ftp           154 May  6 10:43 .
    dr-xr-xr-x   9 ftp      ftp           154 May  6 10:43 ..
    dr-xr-xr-x   5 ftp      ftp          4096 May  6 10:43 CentOS
    dr-xr-xr-x   3 ftp      ftp            18 Jul 10  2008 Fedora

</div>

</div>

Other file-like objects such as <a href="../library/mmap.html#mmap.mmap" class="reference internal" title="mmap.mmap"><span class="pre"><code class="sourceCode python">mmap.mmap</code></span></a> and <a href="../library/fileinput.html#fileinput.input" class="reference internal" title="fileinput.input"><span class="pre"><code class="sourceCode python">fileinput.<span class="bu">input</span>()</code></span></a> also grew auto-closing context managers:

<div class="highlight-python3 notranslate">

<div class="highlight">

    with fileinput.input(files=('log1.txt', 'log2.txt')) as f:
        for line in f:
            process(line)

</div>

</div>

(Contributed by Tarek Ziadé and Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4972" class="reference external">bpo-4972</a>, and by Georg Brandl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8046" class="reference external">bpo-8046</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1286" class="reference external">bpo-1286</a>.)

The <a href="../library/ftplib.html#ftplib.FTP_TLS" class="reference internal" title="ftplib.FTP_TLS"><span class="pre"><code class="sourceCode python">FTP_TLS</code></span></a> class now accepts a *context* parameter, which is a <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a> object allowing bundling SSL configuration options, certificates and private keys into a single (potentially long-lived) structure.

(Contributed by Giampaolo Rodolà; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8806" class="reference external">bpo-8806</a>.)

</div>

<div id="popen" class="section">

### popen<a href="#popen" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/os.html#os.popen" class="reference internal" title="os.popen"><span class="pre"><code class="sourceCode python">os.popen()</code></span></a> and <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen()</code></span></a> functions now support <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statements for auto-closing of the file descriptors.

(Contributed by Antoine Pitrou and Brian Curtin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7461" class="reference external">bpo-7461</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10554" class="reference external">bpo-10554</a>.)

</div>

<div id="select" class="section">

### select<a href="#select" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/select.html#module-select" class="reference internal" title="select: Wait for I/O completion on multiple streams."><span class="pre"><code class="sourceCode python">select</code></span></a> module now exposes a new, constant attribute, <a href="../library/select.html#select.PIPE_BUF" class="reference internal" title="select.PIPE_BUF"><span class="pre"><code class="sourceCode python">PIPE_BUF</code></span></a>, which gives the minimum number of bytes which are guaranteed not to block when <a href="../library/select.html#select.select" class="reference internal" title="select.select"><span class="pre"><code class="sourceCode python">select.select()</code></span></a> says a pipe is ready for writing.

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import select
    >>> select.PIPE_BUF
    512

</div>

</div>

(Available on Unix systems. Patch by Sébastien Sablé in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9862" class="reference external">bpo-9862</a>)

</div>

<div id="gzip-and-zipfile" class="section">

### gzip and zipfile<a href="#gzip-and-zipfile" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/gzip.html#gzip.GzipFile" class="reference internal" title="gzip.GzipFile"><span class="pre"><code class="sourceCode python">gzip.GzipFile</code></span></a> now implements the <a href="../library/io.html#io.BufferedIOBase" class="reference internal" title="io.BufferedIOBase"><span class="pre"><code class="sourceCode python">io.BufferedIOBase</code></span></a> <a href="../glossary.html#term-abstract-base-class" class="reference internal"><span class="xref std std-term">abstract base class</span></a> (except for <span class="pre">`truncate()`</span>). It also has a <a href="../library/gzip.html#gzip.GzipFile.peek" class="reference internal" title="gzip.GzipFile.peek"><span class="pre"><code class="sourceCode python">peek()</code></span></a> method and supports unseekable as well as zero-padded file objects.

The <a href="../library/gzip.html#module-gzip" class="reference internal" title="gzip: Interfaces for gzip compression and decompression using file objects."><span class="pre"><code class="sourceCode python">gzip</code></span></a> module also gains the <a href="../library/gzip.html#gzip.compress" class="reference internal" title="gzip.compress"><span class="pre"><code class="sourceCode python">compress()</code></span></a> and <a href="../library/gzip.html#gzip.decompress" class="reference internal" title="gzip.decompress"><span class="pre"><code class="sourceCode python">decompress()</code></span></a> functions for easier in-memory compression and decompression. Keep in mind that text needs to be encoded as <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> before compressing and decompressing:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import gzip
    >>> s = 'Three shall be the number thou shalt count, '
    >>> s += 'and the number of the counting shall be three'
    >>> b = s.encode()                        # convert to utf-8
    >>> len(b)
    89
    >>> c = gzip.compress(b)
    >>> len(c)
    77
    >>> gzip.decompress(c).decode()[:42]      # decompress and convert to text
    'Three shall be the number thou shalt count'

</div>

</div>

(Contributed by Anand B. Pillai in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3488" class="reference external">bpo-3488</a>; and by Antoine Pitrou, Nir Aides and Brian Curtin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9962" class="reference external">bpo-9962</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1675951" class="reference external">bpo-1675951</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7471" class="reference external">bpo-7471</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2846" class="reference external">bpo-2846</a>.)

Also, the <a href="../library/zipfile.html#zipfile.ZipFile.open" class="reference internal" title="zipfile.ZipFile.open"><span class="pre"><code class="sourceCode python">zipfile.ZipExtFile</code></span></a> class was reworked internally to represent files stored inside an archive. The new implementation is significantly faster and can be wrapped in an <a href="../library/io.html#io.BufferedReader" class="reference internal" title="io.BufferedReader"><span class="pre"><code class="sourceCode python">io.BufferedReader</code></span></a> object for more speedups. It also solves an issue where interleaved calls to *read* and *readline* gave the wrong results.

(Patch submitted by Nir Aides in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7610" class="reference external">bpo-7610</a>.)

</div>

<div id="tarfile" class="section">

### tarfile<a href="#tarfile" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/tarfile.html#tarfile.TarFile" class="reference internal" title="tarfile.TarFile"><span class="pre"><code class="sourceCode python">TarFile</code></span></a> class can now be used as a context manager. In addition, its <a href="../library/tarfile.html#tarfile.TarFile.add" class="reference internal" title="tarfile.TarFile.add"><span class="pre"><code class="sourceCode python">add()</code></span></a> method has a new option, *filter*, that controls which files are added to the archive and allows the file metadata to be edited.

The new *filter* option replaces the older, less flexible *exclude* parameter which is now deprecated. If specified, the optional *filter* parameter needs to be a <a href="../glossary.html#term-keyword-argument" class="reference internal"><span class="xref std std-term">keyword argument</span></a>. The user-supplied filter function accepts a <a href="../library/tarfile.html#tarfile.TarInfo" class="reference internal" title="tarfile.TarInfo"><span class="pre"><code class="sourceCode python">TarInfo</code></span></a> object and returns an updated <a href="../library/tarfile.html#tarfile.TarInfo" class="reference internal" title="tarfile.TarInfo"><span class="pre"><code class="sourceCode python">TarInfo</code></span></a> object, or if it wants the file to be excluded, the function can return <span class="pre">`None`</span>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import tarfile, glob

    >>> def myfilter(tarinfo):
    ...     if tarinfo.isfile():             # only save real files
    ...         tarinfo.uname = 'monty'      # redact the user name
    ...         return tarinfo

    >>> with tarfile.open(name='myarchive.tar.gz', mode='w:gz') as tf:
    ...     for filename in glob.glob('*.txt'):
    ...         tf.add(filename, filter=myfilter)
    ...     tf.list()
    -rw-r--r-- monty/501        902 2011-01-26 17:59:11 annotations.txt
    -rw-r--r-- monty/501        123 2011-01-26 17:59:11 general_questions.txt
    -rw-r--r-- monty/501       3514 2011-01-26 17:59:11 prion.txt
    -rw-r--r-- monty/501        124 2011-01-26 17:59:11 py_todo.txt
    -rw-r--r-- monty/501       1399 2011-01-26 17:59:11 semaphore_notes.txt

</div>

</div>

(Proposed by Tarek Ziadé and implemented by Lars Gustäbel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6856" class="reference external">bpo-6856</a>.)

</div>

<div id="hashlib" class="section">

### hashlib<a href="#hashlib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module has two new constant attributes listing the hashing algorithms guaranteed to be present in all implementations and those available on the current implementation:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import hashlib

    >>> hashlib.algorithms_guaranteed
    {'sha1', 'sha224', 'sha384', 'sha256', 'sha512', 'md5'}

    >>> hashlib.algorithms_available
    {'md2', 'SHA256', 'SHA512', 'dsaWithSHA', 'mdc2', 'SHA224', 'MD4', 'sha256',
    'sha512', 'ripemd160', 'SHA1', 'MDC2', 'SHA', 'SHA384', 'MD2',
    'ecdsa-with-SHA1','md4', 'md5', 'sha1', 'DSA-SHA', 'sha224',
    'dsaEncryption', 'DSA', 'RIPEMD160', 'sha', 'MD5', 'sha384'}

</div>

</div>

(Suggested by Carl Chenet in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7418" class="reference external">bpo-7418</a>.)

</div>

<div id="ast" class="section">

### ast<a href="#ast" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> module has a wonderful a general-purpose tool for safely evaluating expression strings using the Python literal syntax. The <a href="../library/ast.html#ast.literal_eval" class="reference internal" title="ast.literal_eval"><span class="pre"><code class="sourceCode python">ast.literal_eval()</code></span></a> function serves as a secure alternative to the builtin <a href="../library/functions.html#eval" class="reference internal" title="eval"><span class="pre"><code class="sourceCode python"><span class="bu">eval</span>()</code></span></a> function which is easily abused. Python 3.2 adds <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a> literals to the list of supported types: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, and <span class="pre">`None`</span>.

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from ast import literal_eval

    >>> request = "{'req': 3, 'func': 'pow', 'args': (2, 0.5)}"
    >>> literal_eval(request)
    {'args': (2, 0.5), 'req': 3, 'func': 'pow'}

    >>> request = "os.system('do something harmful')"
    >>> literal_eval(request)
    Traceback (most recent call last):
      ...
    ValueError: malformed node or string: <_ast.Call object at 0x101739a10>

</div>

</div>

(Implemented by Benjamin Peterson and Georg Brandl.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

Different operating systems use various encodings for filenames and environment variables. The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module provides two new functions, <a href="../library/os.html#os.fsencode" class="reference internal" title="os.fsencode"><span class="pre"><code class="sourceCode python">fsencode()</code></span></a> and <a href="../library/os.html#os.fsdecode" class="reference internal" title="os.fsdecode"><span class="pre"><code class="sourceCode python">fsdecode()</code></span></a>, for encoding and decoding filenames:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import os
    >>> filename = 'Sehenswürdigkeiten'
    >>> os.fsencode(filename)
    b'Sehensw\xc3\xbcrdigkeiten'

</div>

</div>

Some operating systems allow direct access to encoded bytes in the environment. If so, the <a href="../library/os.html#os.supports_bytes_environ" class="reference internal" title="os.supports_bytes_environ"><span class="pre"><code class="sourceCode python">os.supports_bytes_environ</code></span></a> constant will be true.

For direct access to encoded environment variables (if available), use the new <a href="../library/os.html#os.getenvb" class="reference internal" title="os.getenvb"><span class="pre"><code class="sourceCode python">os.getenvb()</code></span></a> function or use <a href="../library/os.html#os.environb" class="reference internal" title="os.environb"><span class="pre"><code class="sourceCode python">os.environb</code></span></a> which is a bytes version of <a href="../library/os.html#os.environ" class="reference internal" title="os.environ"><span class="pre"><code class="sourceCode python">os.environ</code></span></a>.

(Contributed by Victor Stinner.)

</div>

<div id="shutil" class="section">

### shutil<a href="#shutil" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/shutil.html#shutil.copytree" class="reference internal" title="shutil.copytree"><span class="pre"><code class="sourceCode python">shutil.copytree()</code></span></a> function has two new options:

- *ignore_dangling_symlinks*: when <span class="pre">`symlinks=False`</span> so that the function copies a file pointed to by a symlink, not the symlink itself. This option will silence the error raised if the file doesn’t exist.

- *copy_function*: is a callable that will be used to copy files. <a href="../library/shutil.html#shutil.copy2" class="reference internal" title="shutil.copy2"><span class="pre"><code class="sourceCode python">shutil.copy2()</code></span></a> is used by default.

(Contributed by Tarek Ziadé.)

In addition, the <a href="../library/shutil.html#module-shutil" class="reference internal" title="shutil: High-level file operations, including copying."><span class="pre"><code class="sourceCode python">shutil</code></span></a> module now supports <a href="../library/shutil.html#archiving-operations" class="reference internal"><span class="std std-ref">archiving operations</span></a> for zipfiles, uncompressed tarfiles, gzipped tarfiles, and bzipped tarfiles. And there are functions for registering additional archiving file formats (such as xz compressed tarfiles or custom formats).

The principal functions are <a href="../library/shutil.html#shutil.make_archive" class="reference internal" title="shutil.make_archive"><span class="pre"><code class="sourceCode python">make_archive()</code></span></a> and <a href="../library/shutil.html#shutil.unpack_archive" class="reference internal" title="shutil.unpack_archive"><span class="pre"><code class="sourceCode python">unpack_archive()</code></span></a>. By default, both operate on the current directory (which can be set by <a href="../library/os.html#os.chdir" class="reference internal" title="os.chdir"><span class="pre"><code class="sourceCode python">os.chdir()</code></span></a>) and on any sub-directories. The archive filename needs to be specified with a full pathname. The archiving step is non-destructive (the original files are left unchanged).

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import shutil, pprint

    >>> os.chdir('mydata')  # change to the source directory
    >>> f = shutil.make_archive('/var/backup/mydata',
    ...                         'zip')      # archive the current directory
    >>> f                                   # show the name of archive
    '/var/backup/mydata.zip'
    >>> os.chdir('tmp')                     # change to an unpacking
    >>> shutil.unpack_archive('/var/backup/mydata.zip')  # recover the data

    >>> pprint.pprint(shutil.get_archive_formats())  # display known formats
    [('bztar', "bzip2'ed tar-file"),
     ('gztar', "gzip'ed tar-file"),
     ('tar', 'uncompressed tar file'),
     ('zip', 'ZIP file')]

    >>> shutil.register_archive_format(     # register a new archive format
    ...     name='xz',
    ...     function=xz.compress,           # callable archiving function
    ...     extra_args=[('level', 8)],      # arguments to the function
    ...     description='xz compression'
    ... )

</div>

</div>

(Contributed by Tarek Ziadé.)

</div>

<div id="sqlite3" class="section">

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> module was updated to pysqlite version 2.6.0. It has two new capabilities.

- The <span class="pre">`sqlite3.Connection.in_transit`</span> attribute is true if there is an active transaction for uncommitted changes.

- The <a href="../library/sqlite3.html#sqlite3.Connection.enable_load_extension" class="reference internal" title="sqlite3.Connection.enable_load_extension"><span class="pre"><code class="sourceCode python">sqlite3.Connection.enable_load_extension()</code></span></a> and <a href="../library/sqlite3.html#sqlite3.Connection.load_extension" class="reference internal" title="sqlite3.Connection.load_extension"><span class="pre"><code class="sourceCode python">sqlite3.Connection.load_extension()</code></span></a> methods allows you to load SQLite extensions from “.so” files. One well-known extension is the fulltext-search extension distributed with SQLite.

(Contributed by R. David Murray and Shashwat Anand; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8845" class="reference external">bpo-8845</a>.)

</div>

<div id="html" class="section">

### html<a href="#html" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/html.html#module-html" class="reference internal" title="html: Helpers for manipulating HTML."><span class="pre"><code class="sourceCode python">html</code></span></a> module was introduced with only a single function, <a href="../library/html.html#html.escape" class="reference internal" title="html.escape"><span class="pre"><code class="sourceCode python">escape()</code></span></a>, which is used for escaping reserved characters from HTML markup:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import html
    >>> html.escape('x > 2 && x < 7')
    'x &gt; 2 &amp;&amp; x &lt; 7'

</div>

</div>

</div>

<div id="socket" class="section">

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module has two new improvements.

- Socket objects now have a <a href="../library/socket.html#socket.socket.detach" class="reference internal" title="socket.socket.detach"><span class="pre"><code class="sourceCode python">detach()</code></span></a> method which puts the socket into closed state without actually closing the underlying file descriptor. The latter can then be reused for other purposes. (Added by Antoine Pitrou; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8524" class="reference external">bpo-8524</a>.)

- <a href="../library/socket.html#socket.create_connection" class="reference internal" title="socket.create_connection"><span class="pre"><code class="sourceCode python">socket.create_connection()</code></span></a> now supports the context management protocol to unconditionally consume <a href="../library/socket.html#socket.error" class="reference internal" title="socket.error"><span class="pre"><code class="sourceCode python">socket.error</code></span></a> exceptions and to close the socket when done. (Contributed by Giampaolo Rodolà; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9794" class="reference external">bpo-9794</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module added a number of features to satisfy common requirements for secure (encrypted, authenticated) internet connections:

- A new class, <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a>, serves as a container for persistent SSL data, such as protocol settings, certificates, private keys, and various other options. It includes a <a href="../library/ssl.html#ssl.SSLContext.wrap_socket" class="reference internal" title="ssl.SSLContext.wrap_socket"><span class="pre"><code class="sourceCode python">wrap_socket()</code></span></a> for creating an SSL socket from an SSL context.

- A new function, <span class="pre">`ssl.match_hostname()`</span>, supports server identity verification for higher-level protocols by implementing the rules of HTTPS (from <span id="index-12" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc2818.html" class="rfc reference external"><strong>RFC 2818</strong></a>) which are also suitable for other protocols.

- The <a href="../library/ssl.html#ssl.SSLContext.wrap_socket" class="reference internal" title="ssl.SSLContext.wrap_socket"><span class="pre"><code class="sourceCode python">ssl.wrap_socket()</code></span></a> constructor function now takes a *ciphers* argument. The *ciphers* string lists the allowed encryption algorithms using the format described in the <a href="https://docs.openssl.org/1.0.2/man1/ciphers/#cipher-list-format" class="reference external">OpenSSL documentation</a>.

- When linked against recent versions of OpenSSL, the <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module now supports the Server Name Indication extension to the TLS protocol, allowing multiple “virtual hosts” using different certificates on a single IP port. This extension is only supported in client mode, and is activated by passing the *server_hostname* argument to <a href="../library/ssl.html#ssl.SSLContext.wrap_socket" class="reference internal" title="ssl.SSLContext.wrap_socket"><span class="pre"><code class="sourceCode python">ssl.SSLContext.wrap_socket()</code></span></a>.

- Various options have been added to the <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module, such as <a href="../library/ssl.html#ssl.OP_NO_SSLv2" class="reference internal" title="ssl.OP_NO_SSLv2"><span class="pre"><code class="sourceCode python">OP_NO_SSLv2</code></span></a> which disables the insecure and obsolete SSLv2 protocol.

- The extension now loads all the OpenSSL ciphers and digest algorithms. If some SSL certificates cannot be verified, they are reported as an “unknown algorithm” error.

- The version of OpenSSL being used is now accessible using the module attributes <a href="../library/ssl.html#ssl.OPENSSL_VERSION" class="reference internal" title="ssl.OPENSSL_VERSION"><span class="pre"><code class="sourceCode python">ssl.OPENSSL_VERSION</code></span></a> (a string), <a href="../library/ssl.html#ssl.OPENSSL_VERSION_INFO" class="reference internal" title="ssl.OPENSSL_VERSION_INFO"><span class="pre"><code class="sourceCode python">ssl.OPENSSL_VERSION_INFO</code></span></a> (a 5-tuple), and <a href="../library/ssl.html#ssl.OPENSSL_VERSION_NUMBER" class="reference internal" title="ssl.OPENSSL_VERSION_NUMBER"><span class="pre"><code class="sourceCode python">ssl.OPENSSL_VERSION_NUMBER</code></span></a> (an integer).

(Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8850" class="reference external">bpo-8850</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1589" class="reference external">bpo-1589</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8322" class="reference external">bpo-8322</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5639" class="reference external">bpo-5639</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4870" class="reference external">bpo-4870</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8484" class="reference external">bpo-8484</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8321" class="reference external">bpo-8321</a>.)

</div>

<div id="nntp" class="section">

### nntp<a href="#nntp" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`nntplib`</span> module has a revamped implementation with better bytes and text semantics as well as more practical APIs. These improvements break compatibility with the nntplib version in Python 3.1, which was partly dysfunctional in itself.

Support for secure connections through both implicit (using <span class="pre">`nntplib.NNTP_SSL`</span>) and explicit (using <span class="pre">`nntplib.NNTP.starttls()`</span>) TLS has also been added.

(Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9360" class="reference external">bpo-9360</a> and Andrew Vant in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1926" class="reference external">bpo-1926</a>.)

</div>

<div id="certificates" class="section">

### certificates<a href="#certificates" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/http.client.html#http.client.HTTPSConnection" class="reference internal" title="http.client.HTTPSConnection"><span class="pre"><code class="sourceCode python">http.client.HTTPSConnection</code></span></a>, <a href="../library/urllib.request.html#urllib.request.HTTPSHandler" class="reference internal" title="urllib.request.HTTPSHandler"><span class="pre"><code class="sourceCode python">urllib.request.HTTPSHandler</code></span></a> and <a href="../library/urllib.request.html#urllib.request.urlopen" class="reference internal" title="urllib.request.urlopen"><span class="pre"><code class="sourceCode python">urllib.request.urlopen()</code></span></a> now take optional arguments to allow for server certificate checking against a set of Certificate Authorities, as recommended in public uses of HTTPS.

(Added by Antoine Pitrou, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9003" class="reference external">bpo-9003</a>.)

</div>

<div id="imaplib" class="section">

### imaplib<a href="#imaplib" class="headerlink" title="Link to this heading">¶</a>

Support for explicit TLS on standard IMAP4 connections has been added through the new <a href="../library/imaplib.html#imaplib.IMAP4.starttls" class="reference internal" title="imaplib.IMAP4.starttls"><span class="pre"><code class="sourceCode python">imaplib.IMAP4.starttls</code></span></a> method.

(Contributed by Lorenzo M. Catucci and Antoine Pitrou, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4471" class="reference external">bpo-4471</a>.)

</div>

<div id="http-client" class="section">

### http.client<a href="#http-client" class="headerlink" title="Link to this heading">¶</a>

There were a number of small API improvements in the <a href="../library/http.client.html#module-http.client" class="reference internal" title="http.client: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">http.client</code></span></a> module. The old-style HTTP 0.9 simple responses are no longer supported and the *strict* parameter is deprecated in all classes.

The <a href="../library/http.client.html#http.client.HTTPConnection" class="reference internal" title="http.client.HTTPConnection"><span class="pre"><code class="sourceCode python">HTTPConnection</code></span></a> and <a href="../library/http.client.html#http.client.HTTPSConnection" class="reference internal" title="http.client.HTTPSConnection"><span class="pre"><code class="sourceCode python">HTTPSConnection</code></span></a> classes now have a *source_address* parameter for a (host, port) tuple indicating where the HTTP connection is made from.

Support for certificate checking and HTTPS virtual hosts were added to <a href="../library/http.client.html#http.client.HTTPSConnection" class="reference internal" title="http.client.HTTPSConnection"><span class="pre"><code class="sourceCode python">HTTPSConnection</code></span></a>.

The <a href="../library/http.client.html#http.client.HTTPConnection.request" class="reference internal" title="http.client.HTTPConnection.request"><span class="pre"><code class="sourceCode python">request()</code></span></a> method on connection objects allowed an optional *body* argument so that a <a href="../glossary.html#term-file-object" class="reference internal"><span class="xref std std-term">file object</span></a> could be used to supply the content of the request. Conveniently, the *body* argument now also accepts an <a href="../glossary.html#term-iterable" class="reference internal"><span class="xref std std-term">iterable</span></a> object so long as it includes an explicit <span class="pre">`Content-Length`</span> header. This extended interface is much more flexible than before.

To establish an HTTPS connection through a proxy server, there is a new <a href="../library/http.client.html#http.client.HTTPConnection.set_tunnel" class="reference internal" title="http.client.HTTPConnection.set_tunnel"><span class="pre"><code class="sourceCode python">set_tunnel()</code></span></a> method that sets the host and port for HTTP Connect tunneling.

To match the behavior of <a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a>, the HTTP client library now also encodes headers with ISO-8859-1 (Latin-1) encoding. It was already doing that for incoming headers, so now the behavior is consistent for both incoming and outgoing traffic. (See work by Armin Ronacher in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10980" class="reference external">bpo-10980</a>.)

</div>

<div id="unittest" class="section">

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

The unittest module has a number of improvements supporting test discovery for packages, easier experimentation at the interactive prompt, new testcase methods, improved diagnostic messages for test failures, and better method names.

- The command-line call <span class="pre">`python`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`unittest`</span> can now accept file paths instead of module names for running specific tests (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10620" class="reference external">bpo-10620</a>). The new test discovery can find tests within packages, locating any test importable from the top-level directory. The top-level directory can be specified with the <span class="pre">`-t`</span> option, a pattern for matching files with <span class="pre">`-p`</span>, and a directory to start discovery with <span class="pre">`-s`</span>:

  <div class="highlight-shell-session notranslate">

  <div class="highlight">

      $ python -m unittest discover -s my_proj_dir -p _test.py

  </div>

  </div>

  (Contributed by Michael Foord.)

- Experimentation at the interactive prompt is now easier because the <a href="../library/unittest.html#unittest.TestCase" class="reference internal" title="unittest.TestCase"><span class="pre"><code class="sourceCode python">unittest.TestCase</code></span></a> class can now be instantiated without arguments:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> from unittest import TestCase
      >>> TestCase().assertEqual(pow(2, 3), 8)

  </div>

  </div>

  (Contributed by Michael Foord.)

- The <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> module has two new methods, <a href="../library/unittest.html#unittest.TestCase.assertWarns" class="reference internal" title="unittest.TestCase.assertWarns"><span class="pre"><code class="sourceCode python">assertWarns()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.assertWarnsRegex" class="reference internal" title="unittest.TestCase.assertWarnsRegex"><span class="pre"><code class="sourceCode python">assertWarnsRegex()</code></span></a> to verify that a given warning type is triggered by the code under test:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      with self.assertWarns(DeprecationWarning):
          legacy_function('XYZ')

  </div>

  </div>

  (Contributed by Antoine Pitrou, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9754" class="reference external">bpo-9754</a>.)

  Another new method, <a href="../library/unittest.html#unittest.TestCase.assertCountEqual" class="reference internal" title="unittest.TestCase.assertCountEqual"><span class="pre"><code class="sourceCode python">assertCountEqual()</code></span></a> is used to compare two iterables to determine if their element counts are equal (whether the same elements are present with the same number of occurrences regardless of order):

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      def test_anagram(self):
          self.assertCountEqual('algorithm', 'logarithm')

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- A principal feature of the unittest module is an effort to produce meaningful diagnostics when a test fails. When possible, the failure is recorded along with a diff of the output. This is especially helpful for analyzing log files of failed test runs. However, since diffs can sometime be voluminous, there is a new <a href="../library/unittest.html#unittest.TestCase.maxDiff" class="reference internal" title="unittest.TestCase.maxDiff"><span class="pre"><code class="sourceCode python">maxDiff</code></span></a> attribute that sets maximum length of diffs displayed.

- In addition, the method names in the module have undergone a number of clean-ups.

  For example, <a href="../library/unittest.html#unittest.TestCase.assertRegex" class="reference internal" title="unittest.TestCase.assertRegex"><span class="pre"><code class="sourceCode python">assertRegex()</code></span></a> is the new name for <span class="pre">`assertRegexpMatches()`</span> which was misnamed because the test uses <a href="../library/re.html#re.search" class="reference internal" title="re.search"><span class="pre"><code class="sourceCode python">re.search()</code></span></a>, not <a href="../library/re.html#re.match" class="reference internal" title="re.match"><span class="pre"><code class="sourceCode python">re.match()</code></span></a>. Other methods using regular expressions are now named using short form “Regex” in preference to “Regexp” – this matches the names used in other unittest implementations, matches Python’s old name for the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module, and it has unambiguous camel-casing.

  (Contributed by Raymond Hettinger and implemented by Ezio Melotti.)

- To improve consistency, some long-standing method aliases are being deprecated in favor of the preferred names:

  > <div>
  >
  > | Old Name | Preferred Name |
  > |----|----|
  > | <span class="pre">`assert_()`</span> | <a href="../library/unittest.html#unittest.TestCase.assertTrue" class="reference internal" title="unittest.TestCase.assertTrue"><span class="pre"><code class="sourceCode python">assertTrue()</code></span></a> |
  > | <span class="pre">`assertEquals()`</span> | <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a> |
  > | <span class="pre">`assertNotEquals()`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotEqual" class="reference internal" title="unittest.TestCase.assertNotEqual"><span class="pre"><code class="sourceCode python">assertNotEqual()</code></span></a> |
  > | <span class="pre">`assertAlmostEquals()`</span> | <a href="../library/unittest.html#unittest.TestCase.assertAlmostEqual" class="reference internal" title="unittest.TestCase.assertAlmostEqual"><span class="pre"><code class="sourceCode python">assertAlmostEqual()</code></span></a> |
  > | <span class="pre">`assertNotAlmostEquals()`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotAlmostEqual" class="reference internal" title="unittest.TestCase.assertNotAlmostEqual"><span class="pre"><code class="sourceCode python">assertNotAlmostEqual()</code></span></a> |
  >
  > </div>

  Likewise, the <span class="pre">`TestCase.fail*`</span> methods deprecated in Python 3.1 are expected to be removed in Python 3.3.

  (Contributed by Ezio Melotti; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9424" class="reference external">bpo-9424</a>.)

- The <span class="pre">`assertDictContainsSubset()`</span> method was deprecated because it was misimplemented with the arguments in the wrong order. This created hard-to-debug optical illusions where tests like <span class="pre">`TestCase().assertDictContainsSubset({'a':1,`</span>` `<span class="pre">`'b':2},`</span>` `<span class="pre">`{'a':1})`</span> would fail.

  (Contributed by Raymond Hettinger.)

</div>

<div id="random" class="section">

### random<a href="#random" class="headerlink" title="Link to this heading">¶</a>

The integer methods in the <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module now do a better job of producing uniform distributions. Previously, they computed selections with <span class="pre">`int(n*random())`</span> which had a slight bias whenever *n* was not a power of two. Now, multiple selections are made from a range up to the next power of two and a selection is kept only when it falls within the range <span class="pre">`0`</span>` `<span class="pre">`<=`</span>` `<span class="pre">`x`</span>` `<span class="pre">`<`</span>` `<span class="pre">`n`</span>. The functions and methods affected are <a href="../library/random.html#random.randrange" class="reference internal" title="random.randrange"><span class="pre"><code class="sourceCode python">randrange()</code></span></a>, <a href="../library/random.html#random.randint" class="reference internal" title="random.randint"><span class="pre"><code class="sourceCode python">randint()</code></span></a>, <a href="../library/random.html#random.choice" class="reference internal" title="random.choice"><span class="pre"><code class="sourceCode python">choice()</code></span></a>, <a href="../library/random.html#random.shuffle" class="reference internal" title="random.shuffle"><span class="pre"><code class="sourceCode python">shuffle()</code></span></a> and <a href="../library/random.html#random.sample" class="reference internal" title="random.sample"><span class="pre"><code class="sourceCode python">sample()</code></span></a>.

(Contributed by Raymond Hettinger; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9025" class="reference external">bpo-9025</a>.)

</div>

<div id="poplib" class="section">

### poplib<a href="#poplib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/poplib.html#poplib.POP3_SSL" class="reference internal" title="poplib.POP3_SSL"><span class="pre"><code class="sourceCode python">POP3_SSL</code></span></a> class now accepts a *context* parameter, which is a <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a> object allowing bundling SSL configuration options, certificates and private keys into a single (potentially long-lived) structure.

(Contributed by Giampaolo Rodolà; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8807" class="reference external">bpo-8807</a>.)

</div>

<div id="asyncore" class="section">

### asyncore<a href="#asyncore" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`asyncore.dispatcher`</span> now provides a <span class="pre">`handle_accepted()`</span> method returning a <span class="pre">`(sock,`</span>` `<span class="pre">`addr)`</span> pair which is called when a connection has actually been established with a new remote endpoint. This is supposed to be used as a replacement for old <span class="pre">`handle_accept()`</span> and avoids the user to call <span class="pre">`accept()`</span> directly.

(Contributed by Giampaolo Rodolà; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6706" class="reference external">bpo-6706</a>.)

</div>

<div id="tempfile" class="section">

### tempfile<a href="#tempfile" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/tempfile.html#module-tempfile" class="reference internal" title="tempfile: Generate temporary files and directories."><span class="pre"><code class="sourceCode python">tempfile</code></span></a> module has a new context manager, <a href="../library/tempfile.html#tempfile.TemporaryDirectory" class="reference internal" title="tempfile.TemporaryDirectory"><span class="pre"><code class="sourceCode python">TemporaryDirectory</code></span></a> which provides easy deterministic cleanup of temporary directories:

<div class="highlight-python3 notranslate">

<div class="highlight">

    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary dir:', tmpdirname)

</div>

</div>

(Contributed by Neil Schemenauer and Nick Coghlan; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5178" class="reference external">bpo-5178</a>.)

</div>

<div id="inspect" class="section">

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> module has a new function <a href="../library/inspect.html#inspect.getgeneratorstate" class="reference internal" title="inspect.getgeneratorstate"><span class="pre"><code class="sourceCode python">getgeneratorstate()</code></span></a> to easily identify the current state of a generator-iterator:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> from inspect import getgeneratorstate
      >>> def gen():
      ...     yield 'demo'
      ...
      >>> g = gen()
      >>> getgeneratorstate(g)
      'GEN_CREATED'
      >>> next(g)
      'demo'
      >>> getgeneratorstate(g)
      'GEN_SUSPENDED'
      >>> next(g, None)
      >>> getgeneratorstate(g)
      'GEN_CLOSED'

  </div>

  </div>

  (Contributed by Rodolpho Eckhardt and Nick Coghlan, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10220" class="reference external">bpo-10220</a>.)

- To support lookups without the possibility of activating a dynamic attribute, the <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> module has a new function, <a href="../library/inspect.html#inspect.getattr_static" class="reference internal" title="inspect.getattr_static"><span class="pre"><code class="sourceCode python">getattr_static()</code></span></a>. Unlike <a href="../library/functions.html#hasattr" class="reference internal" title="hasattr"><span class="pre"><code class="sourceCode python"><span class="bu">hasattr</span>()</code></span></a>, this is a true read-only search, guaranteed not to change state while it is searching:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> class A:
      ...     @property
      ...     def f(self):
      ...         print('Running')
      ...         return 10
      ...
      >>> a = A()
      >>> getattr(a, 'f')
      Running
      10
      >>> inspect.getattr_static(a, 'f')
      <property object at 0x1022bd788>

  </div>

  </div>

> <div>
>
> (Contributed by Michael Foord.)
>
> </div>

</div>

<div id="pydoc" class="section">

### pydoc<a href="#pydoc" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module now provides a much-improved web server interface, as well as a new command-line option <span class="pre">`-b`</span> to automatically open a browser window to display that server:

<div class="highlight-shell-session notranslate">

<div class="highlight">

    $ pydoc3.2 -b

</div>

</div>

(Contributed by Ron Adam; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2001" class="reference external">bpo-2001</a>.)

</div>

<div id="dis" class="section">

### dis<a href="#dis" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/dis.html#module-dis" class="reference internal" title="dis: Disassembler for Python bytecode."><span class="pre"><code class="sourceCode python">dis</code></span></a> module gained two new functions for inspecting code, <a href="../library/dis.html#dis.code_info" class="reference internal" title="dis.code_info"><span class="pre"><code class="sourceCode python">code_info()</code></span></a> and <a href="../library/dis.html#dis.show_code" class="reference internal" title="dis.show_code"><span class="pre"><code class="sourceCode python">show_code()</code></span></a>. Both provide detailed code object information for the supplied function, method, source code string or code object. The former returns a string and the latter prints it:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import dis, random
    >>> dis.show_code(random.choice)
    Name:              choice
    Filename:          /Library/Frameworks/Python.framework/Versions/3.2/lib/python3.2/random.py
    Argument count:    2
    Kw-only arguments: 0
    Number of locals:  3
    Stack size:        11
    Flags:             OPTIMIZED, NEWLOCALS, NOFREE
    Constants:
       0: 'Choose a random element from a non-empty sequence.'
       1: 'Cannot choose from an empty sequence'
    Names:
       0: _randbelow
       1: len
       2: ValueError
       3: IndexError
    Variable names:
       0: self
       1: seq
       2: i

</div>

</div>

In addition, the <a href="../library/dis.html#dis.dis" class="reference internal" title="dis.dis"><span class="pre"><code class="sourceCode python">dis()</code></span></a> function now accepts string arguments so that the common idiom <span class="pre">`dis(compile(s,`</span>` `<span class="pre">`'',`</span>` `<span class="pre">`'eval'))`</span> can be shortened to <span class="pre">`dis(s)`</span>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> dis('3*x+1 if x%2==1 else x//2')
      1           0 LOAD_NAME                0 (x)
                  3 LOAD_CONST               0 (2)
                  6 BINARY_MODULO
                  7 LOAD_CONST               1 (1)
                 10 COMPARE_OP               2 (==)
                 13 POP_JUMP_IF_FALSE       28
                 16 LOAD_CONST               2 (3)
                 19 LOAD_NAME                0 (x)
                 22 BINARY_MULTIPLY
                 23 LOAD_CONST               1 (1)
                 26 BINARY_ADD
                 27 RETURN_VALUE
            >>   28 LOAD_NAME                0 (x)
                 31 LOAD_CONST               0 (2)
                 34 BINARY_FLOOR_DIVIDE
                 35 RETURN_VALUE

</div>

</div>

Taken together, these improvements make it easier to explore how CPython is implemented and to see for yourself what the language syntax does under-the-hood.

(Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9147" class="reference external">bpo-9147</a>.)

</div>

<div id="dbm" class="section">

### dbm<a href="#dbm" class="headerlink" title="Link to this heading">¶</a>

All database modules now support the <span class="pre">`get()`</span> and <span class="pre">`setdefault()`</span> methods.

(Suggested by Ray Allen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9523" class="reference external">bpo-9523</a>.)

</div>

<div id="ctypes" class="section">

### ctypes<a href="#ctypes" class="headerlink" title="Link to this heading">¶</a>

A new type, <a href="../library/ctypes.html#ctypes.c_ssize_t" class="reference internal" title="ctypes.c_ssize_t"><span class="pre"><code class="sourceCode python">ctypes.c_ssize_t</code></span></a> represents the C <span class="pre">`ssize_t`</span> datatype.

</div>

<div id="site" class="section">

### site<a href="#site" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/site.html#module-site" class="reference internal" title="site: Module responsible for site-specific configuration."><span class="pre"><code class="sourceCode python">site</code></span></a> module has three new functions useful for reporting on the details of a given Python installation.

- <a href="../library/site.html#site.getsitepackages" class="reference internal" title="site.getsitepackages"><span class="pre"><code class="sourceCode python">getsitepackages()</code></span></a> lists all global site-packages directories.

- <a href="../library/site.html#site.getuserbase" class="reference internal" title="site.getuserbase"><span class="pre"><code class="sourceCode python">getuserbase()</code></span></a> reports on the user’s base directory where data can be stored.

- <a href="../library/site.html#site.getusersitepackages" class="reference internal" title="site.getusersitepackages"><span class="pre"><code class="sourceCode python">getusersitepackages()</code></span></a> reveals the user-specific site-packages directory path.

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import site
    >>> site.getsitepackages()
    ['/Library/Frameworks/Python.framework/Versions/3.2/lib/python3.2/site-packages',
     '/Library/Frameworks/Python.framework/Versions/3.2/lib/site-python',
     '/Library/Python/3.2/site-packages']
    >>> site.getuserbase()
    '/Users/raymondhettinger/Library/Python/3.2'
    >>> site.getusersitepackages()
    '/Users/raymondhettinger/Library/Python/3.2/lib/python/site-packages'

</div>

</div>

Conveniently, some of site’s functionality is accessible directly from the command-line:

<div class="highlight-shell-session notranslate">

<div class="highlight">

    $ python -m site --user-base
    /Users/raymondhettinger/.local
    $ python -m site --user-site
    /Users/raymondhettinger/.local/lib/python3.2/site-packages

</div>

</div>

(Contributed by Tarek Ziadé in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6693" class="reference external">bpo-6693</a>.)

</div>

<div id="sysconfig" class="section">

### sysconfig<a href="#sysconfig" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/sysconfig.html#module-sysconfig" class="reference internal" title="sysconfig: Python&#39;s configuration information"><span class="pre"><code class="sourceCode python">sysconfig</code></span></a> module makes it straightforward to discover installation paths and configuration variables that vary across platforms and installations.

The module offers access simple access functions for platform and version information:

- <a href="../library/sysconfig.html#sysconfig.get_platform" class="reference internal" title="sysconfig.get_platform"><span class="pre"><code class="sourceCode python">get_platform()</code></span></a> returning values like *linux-i586* or *macosx-10.6-ppc*.

- <a href="../library/sysconfig.html#sysconfig.get_python_version" class="reference internal" title="sysconfig.get_python_version"><span class="pre"><code class="sourceCode python">get_python_version()</code></span></a> returns a Python version string such as “3.2”.

It also provides access to the paths and variables corresponding to one of seven named schemes used by <span class="pre">`distutils`</span>. Those include *posix_prefix*, *posix_home*, *posix_user*, *nt*, *nt_user*, *os2*, *os2_home*:

- <a href="../library/sysconfig.html#sysconfig.get_paths" class="reference internal" title="sysconfig.get_paths"><span class="pre"><code class="sourceCode python">get_paths()</code></span></a> makes a dictionary containing installation paths for the current installation scheme.

- <a href="../library/sysconfig.html#sysconfig.get_config_vars" class="reference internal" title="sysconfig.get_config_vars"><span class="pre"><code class="sourceCode python">get_config_vars()</code></span></a> returns a dictionary of platform specific variables.

There is also a convenient command-line interface:

<div class="highlight-doscon notranslate">

<div class="highlight">

    C:\Python32>python -m sysconfig
    Platform: "win32"
    Python version: "3.2"
    Current installation scheme: "nt"

    Paths:
            data = "C:\Python32"
            include = "C:\Python32\Include"
            platinclude = "C:\Python32\Include"
            platlib = "C:\Python32\Lib\site-packages"
            platstdlib = "C:\Python32\Lib"
            purelib = "C:\Python32\Lib\site-packages"
            scripts = "C:\Python32\Scripts"
            stdlib = "C:\Python32\Lib"

    Variables:
            BINDIR = "C:\Python32"
            BINLIBDEST = "C:\Python32\Lib"
            EXE = ".exe"
            INCLUDEPY = "C:\Python32\Include"
            LIBDEST = "C:\Python32\Lib"
            SO = ".pyd"
            VERSION = "32"
            abiflags = ""
            base = "C:\Python32"
            exec_prefix = "C:\Python32"
            platbase = "C:\Python32"
            prefix = "C:\Python32"
            projectbase = "C:\Python32"
            py_version = "3.2"
            py_version_nodot = "32"
            py_version_short = "3.2"
            srcdir = "C:\Python32"
            userbase = "C:\Documents and Settings\Raymond\Application Data\Python"

</div>

</div>

(Moved out of Distutils by Tarek Ziadé.)

</div>

<div id="pdb" class="section">

### pdb<a href="#pdb" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a> debugger module gained a number of usability improvements:

- <span class="pre">`pdb.py`</span> now has a <span class="pre">`-c`</span> option that executes commands as given in a <span class="pre">`.pdbrc`</span> script file.

- A <span class="pre">`.pdbrc`</span> script file can contain <span class="pre">`continue`</span> and <span class="pre">`next`</span> commands that continue debugging.

- The <a href="../library/pdb.html#pdb.Pdb" class="reference internal" title="pdb.Pdb"><span class="pre"><code class="sourceCode python">Pdb</code></span></a> class constructor now accepts a *nosigint* argument.

- New commands: <span class="pre">`l(list)`</span>, <span class="pre">`ll(long`</span>` `<span class="pre">`list)`</span> and <span class="pre">`source`</span> for listing source code.

- New commands: <span class="pre">`display`</span> and <span class="pre">`undisplay`</span> for showing or hiding the value of an expression if it has changed.

- New command: <span class="pre">`interact`</span> for starting an interactive interpreter containing the global and local names found in the current scope.

- Breakpoints can be cleared by breakpoint number.

(Contributed by Georg Brandl, Antonio Cuni and Ilya Sandler.)

</div>

<div id="configparser" class="section">

### configparser<a href="#configparser" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/configparser.html#module-configparser" class="reference internal" title="configparser: Configuration file parser."><span class="pre"><code class="sourceCode python">configparser</code></span></a> module was modified to improve usability and predictability of the default parser and its supported INI syntax. The old <span class="pre">`ConfigParser`</span> class was removed in favor of <span class="pre">`SafeConfigParser`</span> which has in turn been renamed to <a href="../library/configparser.html#configparser.ConfigParser" class="reference internal" title="configparser.ConfigParser"><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a>. Support for inline comments is now turned off by default and section or option duplicates are not allowed in a single configuration source.

Config parsers gained a new API based on the mapping protocol:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> parser = ConfigParser()
    >>> parser.read_string("""
    ... [DEFAULT]
    ... location = upper left
    ... visible = yes
    ... editable = no
    ... color = blue
    ...
    ... [main]
    ... title = Main Menu
    ... color = green
    ...
    ... [options]
    ... title = Options
    ... """)
    >>> parser['main']['color']
    'green'
    >>> parser['main']['editable']
    'no'
    >>> section = parser['options']
    >>> section['title']
    'Options'
    >>> section['title'] = 'Options (editable: %(editable)s)'
    >>> section['title']
    'Options (editable: no)'

</div>

</div>

The new API is implemented on top of the classical API, so custom parser subclasses should be able to use it without modifications.

The INI file structure accepted by config parsers can now be customized. Users can specify alternative option/value delimiters and comment prefixes, change the name of the *DEFAULT* section or switch the interpolation syntax.

There is support for pluggable interpolation including an additional interpolation handler <a href="../library/configparser.html#configparser.ExtendedInterpolation" class="reference internal" title="configparser.ExtendedInterpolation"><span class="pre"><code class="sourceCode python">ExtendedInterpolation</code></span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> parser = ConfigParser(interpolation=ExtendedInterpolation())
    >>> parser.read_dict({'buildout': {'directory': '/home/ambv/zope9'},
    ...                   'custom': {'prefix': '/usr/local'}})
    >>> parser.read_string("""
    ... [buildout]
    ... parts =
    ...   zope9
    ...   instance
    ... find-links =
    ...   ${buildout:directory}/downloads/dist
    ...
    ... [zope9]
    ... recipe = plone.recipe.zope9install
    ... location = /opt/zope
    ...
    ... [instance]
    ... recipe = plone.recipe.zope9instance
    ... zope9-location = ${zope9:location}
    ... zope-conf = ${custom:prefix}/etc/zope.conf
    ... """)
    >>> parser['buildout']['find-links']
    '\n/home/ambv/zope9/downloads/dist'
    >>> parser['instance']['zope-conf']
    '/usr/local/etc/zope.conf'
    >>> instance = parser['instance']
    >>> instance['zope-conf']
    '/usr/local/etc/zope.conf'
    >>> instance['zope9-location']
    '/opt/zope'

</div>

</div>

A number of smaller features were also introduced, like support for specifying encoding in read operations, specifying fallback values for get-functions, or reading directly from dictionaries and strings.

(All changes contributed by Łukasz Langa.)

</div>

<div id="urllib-parse" class="section">

### urllib.parse<a href="#urllib-parse" class="headerlink" title="Link to this heading">¶</a>

A number of usability improvements were made for the <a href="../library/urllib.parse.html#module-urllib.parse" class="reference internal" title="urllib.parse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urllib.parse</code></span></a> module.

The <a href="../library/urllib.parse.html#urllib.parse.urlparse" class="reference internal" title="urllib.parse.urlparse"><span class="pre"><code class="sourceCode python">urlparse()</code></span></a> function now supports <a href="https://en.wikipedia.org/wiki/IPv6" class="reference external">IPv6</a> addresses as described in <span id="index-13" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc2732.html" class="rfc reference external"><strong>RFC 2732</strong></a>:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import urllib.parse
    >>> urllib.parse.urlparse('http://[dead:beef:cafe:5417:affe:8FA3:deaf:feed]/foo/')
    ParseResult(scheme='http',
                netloc='[dead:beef:cafe:5417:affe:8FA3:deaf:feed]',
                path='/foo/',
                params='',
                query='',
                fragment='')

</div>

</div>

The <a href="../library/urllib.parse.html#urllib.parse.urldefrag" class="reference internal" title="urllib.parse.urldefrag"><span class="pre"><code class="sourceCode python">urldefrag()</code></span></a> function now returns a <a href="../glossary.html#term-named-tuple" class="reference internal"><span class="xref std std-term">named tuple</span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> r = urllib.parse.urldefrag('http://python.org/about/#target')
    >>> r
    DefragResult(url='http://python.org/about/', fragment='target')
    >>> r[0]
    'http://python.org/about/'
    >>> r.fragment
    'target'

</div>

</div>

And, the <a href="../library/urllib.parse.html#urllib.parse.urlencode" class="reference internal" title="urllib.parse.urlencode"><span class="pre"><code class="sourceCode python">urlencode()</code></span></a> function is now much more flexible, accepting either a string or bytes type for the *query* argument. If it is a string, then the *safe*, *encoding*, and *error* parameters are sent to <a href="../library/urllib.parse.html#urllib.parse.quote_plus" class="reference internal" title="urllib.parse.quote_plus"><span class="pre"><code class="sourceCode python">quote_plus()</code></span></a> for encoding:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> urllib.parse.urlencode([
    ...      ('type', 'telenovela'),
    ...      ('name', '¿Dónde Está Elisa?')],
    ...      encoding='latin-1')
    'type=telenovela&name=%BFD%F3nde+Est%E1+Elisa%3F'

</div>

</div>

As detailed in <a href="../library/urllib.parse.html#parsing-ascii-encoded-bytes" class="reference internal"><span class="std std-ref">Parsing ASCII Encoded Bytes</span></a>, all the <a href="../library/urllib.parse.html#module-urllib.parse" class="reference internal" title="urllib.parse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urllib.parse</code></span></a> functions now accept ASCII-encoded byte strings as input, so long as they are not mixed with regular strings. If ASCII-encoded byte strings are given as parameters, the return types will also be an ASCII-encoded byte strings:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> urllib.parse.urlparse(b'http://www.python.org:80/about/')
    ParseResultBytes(scheme=b'http', netloc=b'www.python.org:80',
                     path=b'/about/', params=b'', query=b'', fragment=b'')

</div>

</div>

(Work by Nick Coghlan, Dan Mahn, and Senthil Kumaran in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2987" class="reference external">bpo-2987</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5468" class="reference external">bpo-5468</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9873" class="reference external">bpo-9873</a>.)

</div>

<div id="mailbox" class="section">

### mailbox<a href="#mailbox" class="headerlink" title="Link to this heading">¶</a>

Thanks to a concerted effort by R. David Murray, the <a href="../library/mailbox.html#module-mailbox" class="reference internal" title="mailbox: Manipulate mailboxes in various formats"><span class="pre"><code class="sourceCode python">mailbox</code></span></a> module has been fixed for Python 3.2. The challenge was that mailbox had been originally designed with a text interface, but email messages are best represented with <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> because various parts of a message may have different encodings.

The solution harnessed the <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages."><span class="pre"><code class="sourceCode python">email</code></span></a> package’s binary support for parsing arbitrary email messages. In addition, the solution required a number of API changes.

As expected, the <a href="../library/mailbox.html#mailbox.Mailbox.add" class="reference internal" title="mailbox.Mailbox.add"><span class="pre"><code class="sourceCode python">add()</code></span></a> method for <a href="../library/mailbox.html#mailbox.Mailbox" class="reference internal" title="mailbox.Mailbox"><span class="pre"><code class="sourceCode python">mailbox.Mailbox</code></span></a> objects now accepts binary input.

<a href="../library/io.html#io.StringIO" class="reference internal" title="io.StringIO"><span class="pre"><code class="sourceCode python">StringIO</code></span></a> and text file input are deprecated. Also, string input will fail early if non-ASCII characters are used. Previously it would fail when the email was processed in a later step.

There is also support for binary output. The <a href="../library/mailbox.html#mailbox.Mailbox.get_file" class="reference internal" title="mailbox.Mailbox.get_file"><span class="pre"><code class="sourceCode python">get_file()</code></span></a> method now returns a file in the binary mode (where it used to incorrectly set the file to text-mode). There is also a new <a href="../library/mailbox.html#mailbox.Mailbox.get_bytes" class="reference internal" title="mailbox.Mailbox.get_bytes"><span class="pre"><code class="sourceCode python">get_bytes()</code></span></a> method that returns a <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> representation of a message corresponding to a given *key*.

It is still possible to get non-binary output using the old API’s <a href="../library/mailbox.html#mailbox.Mailbox.get_string" class="reference internal" title="mailbox.Mailbox.get_string"><span class="pre"><code class="sourceCode python">get_string()</code></span></a> method, but that approach is not very useful. Instead, it is best to extract messages from a <a href="../library/mailbox.html#mailbox.Message" class="reference internal" title="mailbox.Message"><span class="pre"><code class="sourceCode python">Message</code></span></a> object or to load them from binary input.

(Contributed by R. David Murray, with efforts from Steffen Daode Nurpmeso and an initial patch by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9124" class="reference external">bpo-9124</a>.)

</div>

<div id="turtledemo" class="section">

### turtledemo<a href="#turtledemo" class="headerlink" title="Link to this heading">¶</a>

The demonstration code for the <a href="../library/turtle.html#module-turtle" class="reference internal" title="turtle: An educational framework for simple graphics applications"><span class="pre"><code class="sourceCode python">turtle</code></span></a> module was moved from the *Demo* directory to main library. It includes over a dozen sample scripts with lively displays. Being on <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a>, it can now be run directly from the command-line:

<div class="highlight-shell-session notranslate">

<div class="highlight">

    $ python -m turtledemo

</div>

</div>

(Moved from the Demo directory by Alexander Belopolsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10199" class="reference external">bpo-10199</a>.)

</div>

</div>

<div id="multi-threading" class="section">

## Multi-threading<a href="#multi-threading" class="headerlink" title="Link to this heading">¶</a>

- The mechanism for serializing execution of concurrently running Python threads (generally known as the <a href="../glossary.html#term-GIL" class="reference internal"><span class="xref std std-term">GIL</span></a> or Global Interpreter Lock) has been rewritten. Among the objectives were more predictable switching intervals and reduced overhead due to lock contention and the number of ensuing system calls. The notion of a “check interval” to allow thread switches has been abandoned and replaced by an absolute duration expressed in seconds. This parameter is tunable through <a href="../library/sys.html#sys.setswitchinterval" class="reference internal" title="sys.setswitchinterval"><span class="pre"><code class="sourceCode python">sys.setswitchinterval()</code></span></a>. It currently defaults to 5 milliseconds.

  Additional details about the implementation can be read from a <a href="https://mail.python.org/pipermail/python-dev/2009-October/093321.html" class="reference external">python-dev mailing-list message</a> (however, “priority requests” as exposed in this message have not been kept for inclusion).

  (Contributed by Antoine Pitrou.)

- Regular and recursive locks now accept an optional *timeout* argument to their <a href="../library/threading.html#threading.Lock.acquire" class="reference internal" title="threading.Lock.acquire"><span class="pre"><code class="sourceCode python">acquire()</code></span></a> method. (Contributed by Antoine Pitrou; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7316" class="reference external">bpo-7316</a>.)

- Similarly, <a href="../library/threading.html#threading.Semaphore.acquire" class="reference internal" title="threading.Semaphore.acquire"><span class="pre"><code class="sourceCode python">threading.Semaphore.acquire()</code></span></a> also gained a *timeout* argument. (Contributed by Torsten Landschoff; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=850728" class="reference external">bpo-850728</a>.)

- Regular and recursive lock acquisitions can now be interrupted by signals on platforms using Pthreads. This means that Python programs that deadlock while acquiring locks can be successfully killed by repeatedly sending SIGINT to the process (by pressing <span class="kbd kbd docutils literal notranslate">Ctrl</span>+<span class="kbd kbd docutils literal notranslate">C</span> in most shells). (Contributed by Reid Kleckner; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8844" class="reference external">bpo-8844</a>.)

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

A number of small performance enhancements have been added:

- Python’s peephole optimizer now recognizes patterns such <span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`{1,`</span>` `<span class="pre">`2,`</span>` `<span class="pre">`3}`</span> as being a test for membership in a set of constants. The optimizer recasts the <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a> as a <a href="../library/stdtypes.html#frozenset" class="reference internal" title="frozenset"><span class="pre"><code class="sourceCode python"><span class="bu">frozenset</span></code></span></a> and stores the pre-built constant.

  Now that the speed penalty is gone, it is practical to start writing membership tests using set-notation. This style is both semantically clear and operationally fast:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      extension = name.rpartition('.')[2]
      if extension in {'xml', 'html', 'xhtml', 'css'}:
          handle(name)

  </div>

  </div>

  (Patch and additional tests contributed by Dave Malcolm; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6690" class="reference external">bpo-6690</a>).

- Serializing and unserializing data using the <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> module is now several times faster.

  (Contributed by Alexandre Vassalotti, Antoine Pitrou and the Unladen Swallow team in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9410" class="reference external">bpo-9410</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3873" class="reference external">bpo-3873</a>.)

- The <a href="https://en.wikipedia.org/wiki/Timsort" class="reference external">Timsort algorithm</a> used in <a href="../library/stdtypes.html#list.sort" class="reference internal" title="list.sort"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>.sort()</code></span></a> and <a href="../library/functions.html#sorted" class="reference internal" title="sorted"><span class="pre"><code class="sourceCode python"><span class="bu">sorted</span>()</code></span></a> now runs faster and uses less memory when called with a <a href="../glossary.html#term-key-function" class="reference internal"><span class="xref std std-term">key function</span></a>. Previously, every element of a list was wrapped with a temporary object that remembered the key value associated with each element. Now, two arrays of keys and values are sorted in parallel. This saves the memory consumed by the sort wrappers, and it saves time lost to delegating comparisons.

  (Patch by Daniel Stutzbach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9915" class="reference external">bpo-9915</a>.)

- JSON decoding performance is improved and memory consumption is reduced whenever the same string is repeated for multiple keys. Also, JSON encoding now uses the C speedups when the <span class="pre">`sort_keys`</span> argument is true.

  (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7451" class="reference external">bpo-7451</a> and by Raymond Hettinger and Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10314" class="reference external">bpo-10314</a>.)

- Recursive locks (created with the <a href="../library/threading.html#threading.RLock" class="reference internal" title="threading.RLock"><span class="pre"><code class="sourceCode python">threading.RLock()</code></span></a> API) now benefit from a C implementation which makes them as fast as regular locks, and between 10x and 15x faster than their previous pure Python implementation.

  (Contributed by Antoine Pitrou; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3001" class="reference external">bpo-3001</a>.)

- The fast-search algorithm in stringlib is now used by the <a href="../library/stdtypes.html#str.split" class="reference internal" title="str.split"><span class="pre"><code class="sourceCode python">split()</code></span></a>, <a href="../library/stdtypes.html#str.rsplit" class="reference internal" title="str.rsplit"><span class="pre"><code class="sourceCode python">rsplit()</code></span></a>, <a href="../library/stdtypes.html#str.splitlines" class="reference internal" title="str.splitlines"><span class="pre"><code class="sourceCode python">splitlines()</code></span></a> and <a href="../library/stdtypes.html#str.replace" class="reference internal" title="str.replace"><span class="pre"><code class="sourceCode python">replace()</code></span></a> methods on <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>, <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> and <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> objects. Likewise, the algorithm is also used by <a href="../library/stdtypes.html#str.rfind" class="reference internal" title="str.rfind"><span class="pre"><code class="sourceCode python">rfind()</code></span></a>, <a href="../library/stdtypes.html#str.rindex" class="reference internal" title="str.rindex"><span class="pre"><code class="sourceCode python">rindex()</code></span></a>, <a href="../library/stdtypes.html#str.rsplit" class="reference internal" title="str.rsplit"><span class="pre"><code class="sourceCode python">rsplit()</code></span></a> and <a href="../library/stdtypes.html#str.rpartition" class="reference internal" title="str.rpartition"><span class="pre"><code class="sourceCode python">rpartition()</code></span></a>.

  (Patch by Florent Xicluna in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7622" class="reference external">bpo-7622</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7462" class="reference external">bpo-7462</a>.)

- Integer to string conversions now work two “digits” at a time, reducing the number of division and modulo operations.

  (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6713" class="reference external">bpo-6713</a> by Gawain Bolton, Mark Dickinson, and Victor Stinner.)

There were several other minor optimizations. Set differencing now runs faster when one operand is much larger than the other (patch by Andress Bennetts in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8685" class="reference external">bpo-8685</a>). The <span class="pre">`array.repeat()`</span> method has a faster implementation (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1569291" class="reference external">bpo-1569291</a> by Alexander Belopolsky). The <a href="../library/http.server.html#http.server.BaseHTTPRequestHandler" class="reference internal" title="http.server.BaseHTTPRequestHandler"><span class="pre"><code class="sourceCode python">BaseHTTPRequestHandler</code></span></a> has more efficient buffering (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3709" class="reference external">bpo-3709</a> by Andrew Schaaf). The <a href="../library/operator.html#operator.attrgetter" class="reference internal" title="operator.attrgetter"><span class="pre"><code class="sourceCode python">operator.attrgetter()</code></span></a> function has been sped-up (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10160" class="reference external">bpo-10160</a> by Christos Georgiou). And <a href="../library/configparser.html#configparser.ConfigParser" class="reference internal" title="configparser.ConfigParser"><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a> loads multi-line arguments a bit faster (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7113" class="reference external">bpo-7113</a> by Łukasz Langa).

</div>

<div id="unicode" class="section">

## Unicode<a href="#unicode" class="headerlink" title="Link to this heading">¶</a>

Python has been updated to <a href="https://unicode.org/versions/Unicode6.0.0/" class="reference external">Unicode 6.0.0</a>. The update to the standard adds over 2,000 new characters including <a href="https://en.wikipedia.org/wiki/Emoji" class="reference external">emoji</a> symbols which are important for mobile phones.

In addition, the updated standard has altered the character properties for two Kannada characters (U+0CF1, U+0CF2) and one New Tai Lue numeric character (U+19DA), making the former eligible for use in identifiers while disqualifying the latter. For more information, see <a href="https://www.unicode.org/versions/Unicode6.0.0/#Database_Changes" class="reference external">Unicode Character Database Changes</a>.

</div>

<div id="codecs" class="section">

## Codecs<a href="#codecs" class="headerlink" title="Link to this heading">¶</a>

Support was added for *cp720* Arabic DOS encoding (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1616979" class="reference external">bpo-1616979</a>).

MBCS encoding no longer ignores the error handler argument. In the default strict mode, it raises an <a href="../library/exceptions.html#UnicodeDecodeError" class="reference internal" title="UnicodeDecodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeDecodeError</span></code></span></a> when it encounters an undecodable byte sequence and an <a href="../library/exceptions.html#UnicodeEncodeError" class="reference internal" title="UnicodeEncodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeEncodeError</span></code></span></a> for an unencodable character.

The MBCS codec supports <span class="pre">`'strict'`</span> and <span class="pre">`'ignore'`</span> error handlers for decoding, and <span class="pre">`'strict'`</span> and <span class="pre">`'replace'`</span> for encoding.

To emulate Python3.1 MBCS encoding, select the <span class="pre">`'ignore'`</span> handler for decoding and the <span class="pre">`'replace'`</span> handler for encoding.

On Mac OS X, Python decodes command line arguments with <span class="pre">`'utf-8'`</span> rather than the locale encoding.

By default, <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> uses <span class="pre">`'utf-8'`</span> encoding on Windows (instead of <span class="pre">`'mbcs'`</span>) and the <span class="pre">`'surrogateescape'`</span> error handler on all operating systems.

</div>

<div id="documentation" class="section">

## Documentation<a href="#documentation" class="headerlink" title="Link to this heading">¶</a>

The documentation continues to be improved.

- A table of quick links has been added to the top of lengthy sections such as <a href="../library/functions.html#built-in-funcs" class="reference internal"><span class="std std-ref">Built-in Functions</span></a>. In the case of <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a>, the links are accompanied by tables of cheatsheet-style summaries to provide an overview and memory jog without having to read all of the docs.

- In some cases, the pure Python source code can be a helpful adjunct to the documentation, so now many modules now feature quick links to the latest version of the source code. For example, the <a href="../library/functools.html#module-functools" class="reference internal" title="functools: Higher-order functions and operations on callable objects."><span class="pre"><code class="sourceCode python">functools</code></span></a> module documentation has a quick link at the top labeled:

  > <div>
  >
  > **Source code** <a href="https://github.com/python/cpython/tree/3.13/Lib/functools.py" class="extlink-source reference external">Lib/functools.py</a>.
  >
  > </div>

  (Contributed by Raymond Hettinger; see <a href="https://rhettinger.wordpress.com/2011/01/28/open-your-source-more/" class="reference external">rationale</a>.)

- The docs now contain more examples and recipes. In particular, <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module has an extensive section, <a href="../library/re.html#re-examples" class="reference internal"><span class="std std-ref">Regular Expression Examples</span></a>. Likewise, the <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> module continues to be updated with new <a href="../library/itertools.html#itertools-recipes" class="reference internal"><span class="std std-ref">Itertools Recipes</span></a>.

- The <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> module now has an auxiliary implementation in pure Python. No functionality was changed. This just provides an easier-to-read alternate implementation.

  (Contributed by Alexander Belopolsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9528" class="reference external">bpo-9528</a>.)

- The unmaintained <span class="pre">`Demo`</span> directory has been removed. Some demos were integrated into the documentation, some were moved to the <span class="pre">`Tools/demo`</span> directory, and others were removed altogether.

  (Contributed by Georg Brandl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7962" class="reference external">bpo-7962</a>.)

</div>

<div id="idle" class="section">

## IDLE<a href="#idle" class="headerlink" title="Link to this heading">¶</a>

- The format menu now has an option to clean source files by stripping trailing whitespace.

  (Contributed by Raymond Hettinger; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5150" class="reference external">bpo-5150</a>.)

- IDLE on Mac OS X now works with both Carbon AquaTk and Cocoa AquaTk.

  (Contributed by Kevin Walzer, Ned Deily, and Ronald Oussoren; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6075" class="reference external">bpo-6075</a>.)

</div>

<div id="code-repository" class="section">

## Code Repository<a href="#code-repository" class="headerlink" title="Link to this heading">¶</a>

In addition to the existing Subversion code repository at <a href="https://svn.python.org" class="reference external">https://svn.python.org</a> there is now a <a href="https://www.mercurial-scm.org/" class="reference external">Mercurial</a> repository at <a href="https://hg.python.org/" class="reference external">https://hg.python.org/</a>.

After the 3.2 release, there are plans to switch to Mercurial as the primary repository. This distributed version control system should make it easier for members of the community to create and share external changesets. See <span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0385/" class="pep reference external"><strong>PEP 385</strong></a> for details.

To learn to use the new version control system, see the <a href="https://www.mercurial-scm.org/wiki/QuickStart" class="reference external">Quick Start</a> or the <a href="https://www.mercurial-scm.org/guide" class="reference external">Guide to Mercurial Workflows</a>.

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Link to this heading">¶</a>

Changes to Python’s build process and to the C API include:

- The *idle*, *pydoc* and *2to3* scripts are now installed with a version-specific suffix on <span class="pre">`make`</span>` `<span class="pre">`altinstall`</span> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10679" class="reference external">bpo-10679</a>).

- The C functions that access the Unicode Database now accept and return characters from the full Unicode range, even on narrow unicode builds (Py_UNICODE_TOLOWER, Py_UNICODE_ISDECIMAL, and others). A visible difference in Python is that <a href="../library/unicodedata.html#unicodedata.numeric" class="reference internal" title="unicodedata.numeric"><span class="pre"><code class="sourceCode python">unicodedata.numeric()</code></span></a> now returns the correct value for large code points, and <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> may consider more characters as printable.

  (Reported by Bupjoe Lee and fixed by Amaury Forgeot D’Arc; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5127" class="reference external">bpo-5127</a>.)

- Computed gotos are now enabled by default on supported compilers (which are detected by the configure script). They can still be disabled selectively by specifying <span class="pre">`--without-computed-gotos`</span>.

  (Contributed by Antoine Pitrou; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9203" class="reference external">bpo-9203</a>.)

- The option <span class="pre">`--with-wctype-functions`</span> was removed. The built-in unicode database is now used for all functions.

  (Contributed by Amaury Forgeot D’Arc; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9210" class="reference external">bpo-9210</a>.)

- Hash values are now values of a new type, <a href="../c-api/hash.html#c.Py_hash_t" class="reference internal" title="Py_hash_t"><span class="pre"><code class="sourceCode c">Py_hash_t</code></span></a>, which is defined to be the same size as a pointer. Previously they were of type long, which on some 64-bit operating systems is still only 32 bits long. As a result of this fix, <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a> and <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> can now hold more than <span class="pre">`2**32`</span> entries on builds with 64-bit pointers (previously, they could grow to that size but their performance degraded catastrophically).

  (Suggested by Raymond Hettinger and implemented by Benjamin Peterson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9778" class="reference external">bpo-9778</a>.)

- A new macro <span class="pre">`Py_VA_COPY`</span> copies the state of the variable argument list. It is equivalent to C99 *va_copy* but available on all Python platforms (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2443" class="reference external">bpo-2443</a>).

- A new C API function <span class="pre">`PySys_SetArgvEx()`</span> allows an embedded interpreter to set <a href="../library/sys.html#sys.argv" class="reference internal" title="sys.argv"><span class="pre"><code class="sourceCode python">sys.argv</code></span></a> without also modifying <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5753" class="reference external">bpo-5753</a>).

- <span class="pre">`PyEval_CallObject()`</span> is now only available in macro form. The function declaration, which was kept for backwards compatibility reasons, is now removed – the macro was introduced in 1997 (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8276" class="reference external">bpo-8276</a>).

- There is a new function <a href="../c-api/long.html#c.PyLong_AsLongLongAndOverflow" class="reference internal" title="PyLong_AsLongLongAndOverflow"><span class="pre"><code class="sourceCode c">PyLong_AsLongLongAndOverflow<span class="op">()</span></code></span></a> which is analogous to <a href="../c-api/long.html#c.PyLong_AsLongAndOverflow" class="reference internal" title="PyLong_AsLongAndOverflow"><span class="pre"><code class="sourceCode c">PyLong_AsLongAndOverflow<span class="op">()</span></code></span></a>. They both serve to convert Python <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> into a native fixed-width type while providing detection of cases where the conversion won’t fit (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7767" class="reference external">bpo-7767</a>).

- The <a href="../c-api/unicode.html#c.PyUnicode_CompareWithASCIIString" class="reference internal" title="PyUnicode_CompareWithASCIIString"><span class="pre"><code class="sourceCode c">PyUnicode_CompareWithASCIIString<span class="op">()</span></code></span></a> function now returns *not equal* if the Python string is *NUL* terminated.

- There is a new function <a href="../c-api/exceptions.html#c.PyErr_NewExceptionWithDoc" class="reference internal" title="PyErr_NewExceptionWithDoc"><span class="pre"><code class="sourceCode c">PyErr_NewExceptionWithDoc<span class="op">()</span></code></span></a> that is like <a href="../c-api/exceptions.html#c.PyErr_NewException" class="reference internal" title="PyErr_NewException"><span class="pre"><code class="sourceCode c">PyErr_NewException<span class="op">()</span></code></span></a> but allows a docstring to be specified. This lets C exceptions have the same self-documenting capabilities as their pure Python counterparts (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7033" class="reference external">bpo-7033</a>).

- When compiled with the <span class="pre">`--with-valgrind`</span> option, the pymalloc allocator will be automatically disabled when running under Valgrind. This gives improved memory leak detection when running under Valgrind, while taking advantage of pymalloc at other times (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2422" class="reference external">bpo-2422</a>).

- Removed the <span class="pre">`O?`</span> format from the *PyArg_Parse* functions. The format is no longer used and it had never been documented (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8837" class="reference external">bpo-8837</a>).

There were a number of other small changes to the C-API. See the <a href="https://github.com/python/cpython/blob/v3.2.6/Misc/NEWS" class="reference external">Misc/NEWS</a> file for a complete list.

Also, there were a number of updates to the Mac OS X build, see <a href="https://github.com/python/cpython/blob/v3.2.6/Mac/BuildScript/README.txt" class="reference external">Mac/BuildScript/README.txt</a> for details. For users running a 32/64-bit build, there is a known problem with the default Tcl/Tk on Mac OS X 10.6. Accordingly, we recommend installing an updated alternative such as <a href="https://web.archive.org/web/20101208191259/https://www.activestate.com/activetcl/downloads" class="reference external">ActiveState Tcl/Tk 8.5.9</a>. See <a href="https://www.python.org/download/mac/tcltk/" class="reference external">https://www.python.org/download/mac/tcltk/</a> for additional details.

</div>

<div id="porting-to-python-3-2" class="section">

## Porting to Python 3.2<a href="#porting-to-python-3-2" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code:

- The <a href="../library/configparser.html#module-configparser" class="reference internal" title="configparser: Configuration file parser."><span class="pre"><code class="sourceCode python">configparser</code></span></a> module has a number of clean-ups. The major change is to replace the old <span class="pre">`ConfigParser`</span> class with long-standing preferred alternative <span class="pre">`SafeConfigParser`</span>. In addition there are a number of smaller incompatibilities:

  - The interpolation syntax is now validated on <a href="../library/configparser.html#configparser.ConfigParser.get" class="reference internal" title="configparser.ConfigParser.get"><span class="pre"><code class="sourceCode python">get()</code></span></a> and <a href="../library/configparser.html#configparser.ConfigParser.set" class="reference internal" title="configparser.ConfigParser.set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a> operations. In the default interpolation scheme, only two tokens with percent signs are valid: <span class="pre">`%(name)s`</span> and <span class="pre">`%%`</span>, the latter being an escaped percent sign.

  - The <a href="../library/configparser.html#configparser.ConfigParser.set" class="reference internal" title="configparser.ConfigParser.set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a> and <a href="../library/configparser.html#configparser.ConfigParser.add_section" class="reference internal" title="configparser.ConfigParser.add_section"><span class="pre"><code class="sourceCode python">add_section()</code></span></a> methods now verify that values are actual strings. Formerly, unsupported types could be introduced unintentionally.

  - Duplicate sections or options from a single source now raise either <a href="../library/configparser.html#configparser.DuplicateSectionError" class="reference internal" title="configparser.DuplicateSectionError"><span class="pre"><code class="sourceCode python">DuplicateSectionError</code></span></a> or <a href="../library/configparser.html#configparser.DuplicateOptionError" class="reference internal" title="configparser.DuplicateOptionError"><span class="pre"><code class="sourceCode python">DuplicateOptionError</code></span></a>. Formerly, duplicates would silently overwrite a previous entry.

  - Inline comments are now disabled by default so now the **;** character can be safely used in values.

  - Comments now can be indented. Consequently, for **;** or **\#** to appear at the start of a line in multiline values, it has to be interpolated. This keeps comment prefix characters in values from being mistaken as comments.

  - <span class="pre">`""`</span> is now a valid value and is no longer automatically converted to an empty string. For empty strings, use <span class="pre">`"option`</span>` `<span class="pre">`="`</span> in a line.

- The <span class="pre">`nntplib`</span> module was reworked extensively, meaning that its APIs are often incompatible with the 3.1 APIs.

- <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> objects can no longer be used as filenames; instead, they should be converted to <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>.

- The <span class="pre">`array.tostring()`</span> and <span class="pre">`array.fromstring()`</span> have been renamed to <a href="../library/array.html#array.array.tobytes" class="reference internal" title="array.array.tobytes"><span class="pre"><code class="sourceCode python">array.tobytes()</code></span></a> and <a href="../library/array.html#array.array.frombytes" class="reference internal" title="array.array.frombytes"><span class="pre"><code class="sourceCode python">array.frombytes()</code></span></a> for clarity. The old names have been deprecated. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8990" class="reference external">bpo-8990</a>.)

- <span class="pre">`PyArg_Parse*()`</span> functions:

  - “t#” format has been removed: use “s#” or “s\*” instead

  - “w” and “w#” formats has been removed: use “w\*” instead

- The <span class="pre">`PyCObject`</span> type, deprecated in 3.1, has been removed. To wrap opaque C pointers in Python objects, the <a href="../c-api/capsule.html#c.PyCapsule" class="reference internal" title="PyCapsule"><span class="pre"><code class="sourceCode c">PyCapsule</code></span></a> API should be used instead; the new type has a well-defined interface for passing typing safety information and a less complicated signature for calling a destructor.

- The <span class="pre">`sys.setfilesystemencoding()`</span> function was removed because it had a flawed design.

- The <a href="../library/random.html#random.seed" class="reference internal" title="random.seed"><span class="pre"><code class="sourceCode python">random.seed()</code></span></a> function and method now salt string seeds with an sha512 hash function. To access the previous version of *seed* in order to reproduce Python 3.1 sequences, set the *version* argument to *1*, <span class="pre">`random.seed(s,`</span>` `<span class="pre">`version=1)`</span>.

- The previously deprecated <span class="pre">`string.maketrans()`</span> function has been removed in favor of the static methods <a href="../library/stdtypes.html#bytes.maketrans" class="reference internal" title="bytes.maketrans"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span>.maketrans()</code></span></a> and <a href="../library/stdtypes.html#bytearray.maketrans" class="reference internal" title="bytearray.maketrans"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span>.maketrans()</code></span></a>. This change solves the confusion around which types were supported by the <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module. Now, <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>, and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> each have their own **maketrans** and **translate** methods with intermediate translation tables of the appropriate type.

  (Contributed by Georg Brandl; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5675" class="reference external">bpo-5675</a>.)

- The previously deprecated <span class="pre">`contextlib.nested()`</span> function has been removed in favor of a plain <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement which can accept multiple context managers. The latter technique is faster (because it is built-in), and it does a better job finalizing multiple context managers when one of them raises an exception:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      with open('mylog.txt') as infile, open('a.out', 'w') as outfile:
          for line in infile:
              if '<critical>' in line:
                  outfile.write(line)

  </div>

  </div>

  (Contributed by Georg Brandl and Mattias Brändström; <a href="https://codereview.appspot.com/53094" class="reference external">appspot issue 53094</a>.)

- <a href="../library/struct.html#struct.pack" class="reference internal" title="struct.pack"><span class="pre"><code class="sourceCode python">struct.pack()</code></span></a> now only allows bytes for the <span class="pre">`s`</span> string pack code. Formerly, it would accept text arguments and implicitly encode them to bytes using UTF-8. This was problematic because it made assumptions about the correct encoding and because a variable-length encoding can fail when writing to fixed length segment of a structure.

  Code such as <span class="pre">`struct.pack('<6sHHBBB',`</span>` `<span class="pre">`'GIF87a',`</span>` `<span class="pre">`x,`</span>` `<span class="pre">`y)`</span> should be rewritten with to use bytes instead of text, <span class="pre">`struct.pack('<6sHHBBB',`</span>` `<span class="pre">`b'GIF87a',`</span>` `<span class="pre">`x,`</span>` `<span class="pre">`y)`</span>.

  (Discovered by David Beazley and fixed by Victor Stinner; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10783" class="reference external">bpo-10783</a>.)

- The <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> class now raises an <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.ParseError" class="reference internal" title="xml.etree.ElementTree.ParseError"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.ParseError</code></span></a> when a parse fails. Previously it raised an <a href="../library/pyexpat.html#xml.parsers.expat.ExpatError" class="reference internal" title="xml.parsers.expat.ExpatError"><span class="pre"><code class="sourceCode python">xml.parsers.expat.ExpatError</code></span></a>.

- The new, longer <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> value on floats may break doctests which rely on the old output format.

- In <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a>, the default value for *close_fds* is now <span class="pre">`True`</span> under Unix; under Windows, it is <span class="pre">`True`</span> if the three standard streams are set to <span class="pre">`None`</span>, <span class="pre">`False`</span> otherwise. Previously, *close_fds* was always <span class="pre">`False`</span> by default, which produced difficult to solve bugs or race conditions when open file descriptors would leak into the child process.

- Support for legacy HTTP 0.9 has been removed from <a href="../library/urllib.request.html#module-urllib.request" class="reference internal" title="urllib.request: Extensible library for opening URLs."><span class="pre"><code class="sourceCode python">urllib.request</code></span></a> and <a href="../library/http.client.html#module-http.client" class="reference internal" title="http.client: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">http.client</code></span></a>. Such support is still present on the server side (in <a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a>).

  (Contributed by Antoine Pitrou, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10711" class="reference external">bpo-10711</a>.)

- SSL sockets in timeout mode now raise <a href="../library/socket.html#socket.timeout" class="reference internal" title="socket.timeout"><span class="pre"><code class="sourceCode python">socket.timeout</code></span></a> when a timeout occurs, rather than a generic <a href="../library/ssl.html#ssl.SSLError" class="reference internal" title="ssl.SSLError"><span class="pre"><code class="sourceCode python">SSLError</code></span></a>.

  (Contributed by Antoine Pitrou, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10272" class="reference external">bpo-10272</a>.)

- The misleading functions <span class="pre">`PyEval_AcquireLock()`</span> and <span class="pre">`PyEval_ReleaseLock()`</span> have been officially deprecated. The thread-state aware APIs (such as <a href="../c-api/init.html#c.PyEval_SaveThread" class="reference internal" title="PyEval_SaveThread"><span class="pre"><code class="sourceCode c">PyEval_SaveThread<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyEval_RestoreThread" class="reference internal" title="PyEval_RestoreThread"><span class="pre"><code class="sourceCode c">PyEval_RestoreThread<span class="op">()</span></code></span></a>) should be used instead.

- Due to security risks, <span class="pre">`asyncore.handle_accept()`</span> has been deprecated, and a new function, <span class="pre">`asyncore.handle_accepted()`</span>, was added to replace it.

  (Contributed by Giampaolo Rodola in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6706" class="reference external">bpo-6706</a>.)

- Due to the new <a href="../glossary.html#term-GIL" class="reference internal"><span class="xref std std-term">GIL</span></a> implementation, <span class="pre">`PyEval_InitThreads()`</span> cannot be called before <a href="../c-api/init.html#c.Py_Initialize" class="reference internal" title="Py_Initialize"><span class="pre"><code class="sourceCode c">Py_Initialize<span class="op">()</span></code></span></a> anymore.

</div>

</div>

<div class="clearer">

</div>

</div>
