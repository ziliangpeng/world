<div class="body" role="main">

<div id="what-s-new-in-python-3-3" class="section">

# What’s New In Python 3.3<a href="#what-s-new-in-python-3-3" class="headerlink" title="Link to this heading">¶</a>

This article explains the new features in Python 3.3, compared to 3.2. Python 3.3 was released on September 29, 2012. For full details, see the <a href="https://docs.python.org/3.3/whatsnew/changelog.html" class="reference external">changelog</a>.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0398/" class="pep reference external"><strong>PEP 398</strong></a> - Python 3.3 Release Schedule

</div>

<div id="summary-release-highlights" class="section">

## Summary – Release highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

New syntax features:

- New <span class="pre">`yield`</span>` `<span class="pre">`from`</span> expression for <a href="#pep-380" class="reference internal"><span class="std std-ref">generator delegation</span></a>.

- The <span class="pre">`u'unicode'`</span> syntax is accepted again for <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> objects.

New library modules:

- <a href="../library/faulthandler.html#module-faulthandler" class="reference internal" title="faulthandler: Dump the Python traceback."><span class="pre"><code class="sourceCode python">faulthandler</code></span></a> (helps debugging low-level crashes)

- <a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> (high-level objects representing IP addresses and masks)

- <a href="../library/lzma.html#module-lzma" class="reference internal" title="lzma: A Python wrapper for the liblzma compression library."><span class="pre"><code class="sourceCode python">lzma</code></span></a> (compress data using the XZ / LZMA algorithm)

- <a href="../library/unittest.mock.html#module-unittest.mock" class="reference internal" title="unittest.mock: Mock object library."><span class="pre"><code class="sourceCode python">unittest.mock</code></span></a> (replace parts of your system under test with mock objects)

- <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> (Python <a href="#pep-405" class="reference internal"><span class="std std-ref">virtual environments</span></a>, as in the popular <span class="pre">`virtualenv`</span> package)

New built-in features:

- Reworked <a href="#pep-3151" class="reference internal"><span class="std std-ref">I/O exception hierarchy</span></a>.

Implementation improvements:

- Rewritten <a href="#importlib" class="reference internal"><span class="std std-ref">import machinery</span></a> based on <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a>.

- More compact <a href="#pep-393" class="reference internal"><span class="std std-ref">unicode strings</span></a>.

- More compact <a href="#pep-412" class="reference internal"><span class="std std-ref">attribute dictionaries</span></a>.

Significantly Improved Library Modules:

- C Accelerator for the <a href="#new-decimal" class="reference internal"><span class="std std-ref">decimal</span></a> module.

- Better unicode handling in the <a href="#new-email" class="reference internal"><span class="std std-ref">email</span></a> module (<a href="../glossary.html#term-provisional-package" class="reference internal"><span class="xref std std-term">provisional</span></a>).

Security improvements:

- Hash randomization is switched on by default.

Please read on for a comprehensive list of user-facing changes.

</div>

<div id="pep-405-virtual-environments" class="section">

<span id="pep-405"></span>

## PEP 405: Virtual Environments<a href="#pep-405-virtual-environments" class="headerlink" title="Link to this heading">¶</a>

Virtual environments help create separate Python setups while sharing a system-wide base install, for ease of maintenance. Virtual environments have their own set of private site packages (i.e. locally installed libraries), and are optionally segregated from the system-wide site packages. Their concept and implementation are inspired by the popular <span class="pre">`virtualenv`</span> third-party package, but benefit from tighter integration with the interpreter core.

This PEP adds the <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> module for programmatic access, and the <span class="pre">`pyvenv`</span> script for command-line access and administration. The Python interpreter checks for a <span class="pre">`pyvenv.cfg`</span>, file whose existence signals the base of a virtual environment’s directory tree.

<div class="admonition seealso">

See also

<span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0405/" class="pep reference external"><strong>PEP 405</strong></a> - Python Virtual Environments  
PEP written by Carl Meyer; implementation by Carl Meyer and Vinay Sajip

</div>

</div>

<div id="pep-420-implicit-namespace-packages" class="section">

## PEP 420: Implicit Namespace Packages<a href="#pep-420-implicit-namespace-packages" class="headerlink" title="Link to this heading">¶</a>

Native support for package directories that don’t require <span class="pre">`__init__.py`</span> marker files and can automatically span multiple path segments (inspired by various third party approaches to namespace packages, as described in <span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0420/" class="pep reference external"><strong>PEP 420</strong></a>)

<div class="admonition seealso">

See also

<span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0420/" class="pep reference external"><strong>PEP 420</strong></a> - Implicit Namespace Packages  
PEP written by Eric V. Smith; implementation by Eric V. Smith and Barry Warsaw

</div>

</div>

<div id="pep-3118-new-memoryview-implementation-and-buffer-protocol-documentation" class="section">

<span id="pep-3118-update"></span>

## PEP 3118: New memoryview implementation and buffer protocol documentation<a href="#pep-3118-new-memoryview-implementation-and-buffer-protocol-documentation" class="headerlink" title="Link to this heading">¶</a>

The implementation of <span id="index-4" class="target"></span><a href="https://peps.python.org/pep-3118/" class="pep reference external"><strong>PEP 3118</strong></a> has been significantly improved.

The new memoryview implementation comprehensively fixes all ownership and lifetime issues of dynamically allocated fields in the Py_buffer struct that led to multiple crash reports. Additionally, several functions that crashed or returned incorrect results for non-contiguous or multi-dimensional input have been fixed.

The memoryview object now has a PEP-3118 compliant getbufferproc() that checks the consumer’s request type. Many new features have been added, most of them work in full generality for non-contiguous arrays and arrays with suboffsets.

The documentation has been updated, clearly spelling out responsibilities for both exporters and consumers. Buffer request flags are grouped into basic and compound flags. The memory layout of non-contiguous and multi-dimensional NumPy-style arrays is explained.

<div id="features" class="section">

### Features<a href="#features" class="headerlink" title="Link to this heading">¶</a>

- All native single character format specifiers in struct module syntax (optionally prefixed with ‘@’) are now supported.

- With some restrictions, the cast() method allows changing of format and shape of C-contiguous arrays.

- Multi-dimensional list representations are supported for any array type.

- Multi-dimensional comparisons are supported for any array type.

- One-dimensional memoryviews of hashable (read-only) types with formats B, b or c are now hashable. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13411" class="reference external">bpo-13411</a>.)

- Arbitrary slicing of any 1-D arrays type is supported. For example, it is now possible to reverse a memoryview in *O*(1) by using a negative step.

</div>

<div id="api-changes" class="section">

### API changes<a href="#api-changes" class="headerlink" title="Link to this heading">¶</a>

- The maximum number of dimensions is officially limited to 64.

- The representation of empty shape, strides and suboffsets is now an empty tuple instead of <span class="pre">`None`</span>.

- Accessing a memoryview element with format ‘B’ (unsigned bytes) now returns an integer (in accordance with the struct module syntax). For returning a bytes object the view must be cast to ‘c’ first.

- memoryview comparisons now use the logical structure of the operands and compare all array elements by value. All format strings in struct module syntax are supported. Views with unrecognised format strings are still permitted, but will always compare as unequal, regardless of view contents.

- For further changes see <a href="#build-and-c-api-changes" class="reference internal">Build and C API Changes</a> and <a href="#porting-c-code" class="reference internal">Porting C code</a>.

(Contributed by Stefan Krah in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10181" class="reference external">bpo-10181</a>.)

<div class="admonition seealso">

See also

<span id="index-5" class="target"></span><a href="https://peps.python.org/pep-3118/" class="pep reference external"><strong>PEP 3118</strong></a> - Revising the Buffer Protocol

</div>

</div>

</div>

<div id="pep-393-flexible-string-representation" class="section">

<span id="pep-393"></span>

## PEP 393: Flexible String Representation<a href="#pep-393-flexible-string-representation" class="headerlink" title="Link to this heading">¶</a>

The Unicode string type is changed to support multiple internal representations, depending on the character with the largest Unicode ordinal (1, 2, or 4 bytes) in the represented string. This allows a space-efficient representation in common cases, but gives access to full UCS-4 on all systems. For compatibility with existing APIs, several representations may exist in parallel; over time, this compatibility should be phased out.

On the Python side, there should be no downside to this change.

On the C API side, <span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a> is fully backward compatible. The legacy API should remain available at least five years. Applications using the legacy API will not fully benefit of the memory reduction, or - worse - may use a bit more memory, because Python may have to maintain two versions of each string (in the legacy format and in the new efficient storage).

<div id="functionality" class="section">

### Functionality<a href="#functionality" class="headerlink" title="Link to this heading">¶</a>

Changes introduced by <span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a> are the following:

- Python now always supports the full range of Unicode code points, including non-BMP ones (i.e. from <span class="pre">`U+0000`</span> to <span class="pre">`U+10FFFF`</span>). The distinction between narrow and wide builds no longer exists and Python now behaves like a wide build, even under Windows.

- With the death of narrow builds, the problems specific to narrow builds have also been fixed, for example:

  - <a href="../library/functions.html#len" class="reference internal" title="len"><span class="pre"><code class="sourceCode python"><span class="bu">len</span>()</code></span></a> now always returns 1 for non-BMP characters, so <span class="pre">`len('\U0010FFFF')`</span>` `<span class="pre">`==`</span>` `<span class="pre">`1`</span>;

  - surrogate pairs are not recombined in string literals, so <span class="pre">`'\uDBFF\uDFFF'`</span>` `<span class="pre">`!=`</span>` `<span class="pre">`'\U0010FFFF'`</span>;

  - indexing or slicing non-BMP characters returns the expected value, so <span class="pre">`'\U0010FFFF'[0]`</span> now returns <span class="pre">`'\U0010FFFF'`</span> and not <span class="pre">`'\uDBFF'`</span>;

  - all other functions in the standard library now correctly handle non-BMP code points.

- The value of <a href="../library/sys.html#sys.maxunicode" class="reference internal" title="sys.maxunicode"><span class="pre"><code class="sourceCode python">sys.maxunicode</code></span></a> is now always <span class="pre">`1114111`</span> (<span class="pre">`0x10FFFF`</span> in hexadecimal). The <span class="pre">`PyUnicode_GetMax()`</span> function still returns either <span class="pre">`0xFFFF`</span> or <span class="pre">`0x10FFFF`</span> for backward compatibility, and it should not be used with the new Unicode API (see <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13054" class="reference external">bpo-13054</a>).

- The <span class="pre">`./configure`</span> flag <span class="pre">`--with-wide-unicode`</span> has been removed.

</div>

<div id="performance-and-resource-usage" class="section">

### Performance and resource usage<a href="#performance-and-resource-usage" class="headerlink" title="Link to this heading">¶</a>

The storage of Unicode strings now depends on the highest code point in the string:

- pure ASCII and Latin1 strings (<span class="pre">`U+0000-U+00FF`</span>) use 1 byte per code point;

- BMP strings (<span class="pre">`U+0000-U+FFFF`</span>) use 2 bytes per code point;

- non-BMP strings (<span class="pre">`U+10000-U+10FFFF`</span>) use 4 bytes per code point.

The net effect is that for most applications, memory usage of string storage should decrease significantly - especially compared to former wide unicode builds - as, in many cases, strings will be pure ASCII even in international contexts (because many strings store non-human language data, such as XML fragments, HTTP headers, JSON-encoded data, etc.). We also hope that it will, for the same reasons, increase CPU cache efficiency on non-trivial applications. The memory usage of Python 3.3 is two to three times smaller than Python 3.2, and a little bit better than Python 2.7, on a Django benchmark (see the PEP for details).

<div class="admonition seealso">

See also

<span id="index-8" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a> - Flexible String Representation  
PEP written by Martin von Löwis; implementation by Torsten Becker and Martin von Löwis.

</div>

</div>

</div>

<div id="pep-397-python-launcher-for-windows" class="section">

<span id="pep-397"></span>

## PEP 397: Python Launcher for Windows<a href="#pep-397-python-launcher-for-windows" class="headerlink" title="Link to this heading">¶</a>

The Python 3.3 Windows installer now includes a <span class="pre">`py`</span> launcher application that can be used to launch Python applications in a version independent fashion.

This launcher is invoked implicitly when double-clicking <span class="pre">`*.py`</span> files. If only a single Python version is installed on the system, that version will be used to run the file. If multiple versions are installed, the most recent version is used by default, but this can be overridden by including a Unix-style “shebang line” in the Python script.

The launcher can also be used explicitly from the command line as the <span class="pre">`py`</span> application. Running <span class="pre">`py`</span> follows the same version selection rules as implicitly launching scripts, but a more specific version can be selected by passing appropriate arguments (such as <span class="pre">`-3`</span> to request Python 3 when Python 2 is also installed, or <span class="pre">`-2.6`</span> to specifically request an earlier Python version when a more recent version is installed).

In addition to the launcher, the Windows installer now includes an option to add the newly installed Python to the system PATH. (Contributed by Brian Curtin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3561" class="reference external">bpo-3561</a>.)

<div class="admonition seealso">

See also

<span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0397/" class="pep reference external"><strong>PEP 397</strong></a> - Python Launcher for Windows  
PEP written by Mark Hammond and Martin v. Löwis; implementation by Vinay Sajip.

Launcher documentation: <a href="../using/windows.html#launcher" class="reference internal"><span class="std std-ref">Python Launcher for Windows</span></a>

Installer PATH modification: <a href="../using/windows.html#windows-path-mod" class="reference internal"><span class="std std-ref">Finding the Python executable</span></a>

</div>

</div>

<div id="pep-3151-reworking-the-os-and-io-exception-hierarchy" class="section">

<span id="pep-3151"></span>

## PEP 3151: Reworking the OS and IO exception hierarchy<a href="#pep-3151-reworking-the-os-and-io-exception-hierarchy" class="headerlink" title="Link to this heading">¶</a>

The hierarchy of exceptions raised by operating system errors is now both simplified and finer-grained.

You don’t have to worry anymore about choosing the appropriate exception type between <a href="../library/exceptions.html#OSError" class="reference internal" title="OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a>, <a href="../library/exceptions.html#IOError" class="reference internal" title="IOError"><span class="pre"><code class="sourceCode python"><span class="pp">IOError</span></code></span></a>, <a href="../library/exceptions.html#EnvironmentError" class="reference internal" title="EnvironmentError"><span class="pre"><code class="sourceCode python"><span class="pp">EnvironmentError</span></code></span></a>, <a href="../library/exceptions.html#WindowsError" class="reference internal" title="WindowsError"><span class="pre"><code class="sourceCode python"><span class="pp">WindowsError</span></code></span></a>, <span class="pre">`mmap.error`</span>, <a href="../library/socket.html#socket.error" class="reference internal" title="socket.error"><span class="pre"><code class="sourceCode python">socket.error</code></span></a> or <a href="../library/select.html#select.error" class="reference internal" title="select.error"><span class="pre"><code class="sourceCode python">select.error</code></span></a>. All these exception types are now only one: <a href="../library/exceptions.html#OSError" class="reference internal" title="OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a>. The other names are kept as aliases for compatibility reasons.

Also, it is now easier to catch a specific error condition. Instead of inspecting the <span class="pre">`errno`</span> attribute (or <span class="pre">`args[0]`</span>) for a particular constant from the <a href="../library/errno.html#module-errno" class="reference internal" title="errno: Standard errno system symbols."><span class="pre"><code class="sourceCode python">errno</code></span></a> module, you can catch the adequate <a href="../library/exceptions.html#OSError" class="reference internal" title="OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a> subclass. The available subclasses are the following:

- <a href="../library/exceptions.html#BlockingIOError" class="reference internal" title="BlockingIOError"><span class="pre"><code class="sourceCode python"><span class="pp">BlockingIOError</span></code></span></a>

- <a href="../library/exceptions.html#ChildProcessError" class="reference internal" title="ChildProcessError"><span class="pre"><code class="sourceCode python"><span class="pp">ChildProcessError</span></code></span></a>

- <a href="../library/exceptions.html#ConnectionError" class="reference internal" title="ConnectionError"><span class="pre"><code class="sourceCode python"><span class="pp">ConnectionError</span></code></span></a>

- <a href="../library/exceptions.html#FileExistsError" class="reference internal" title="FileExistsError"><span class="pre"><code class="sourceCode python"><span class="pp">FileExistsError</span></code></span></a>

- <a href="../library/exceptions.html#FileNotFoundError" class="reference internal" title="FileNotFoundError"><span class="pre"><code class="sourceCode python"><span class="pp">FileNotFoundError</span></code></span></a>

- <a href="../library/exceptions.html#InterruptedError" class="reference internal" title="InterruptedError"><span class="pre"><code class="sourceCode python"><span class="pp">InterruptedError</span></code></span></a>

- <a href="../library/exceptions.html#IsADirectoryError" class="reference internal" title="IsADirectoryError"><span class="pre"><code class="sourceCode python"><span class="pp">IsADirectoryError</span></code></span></a>

- <a href="../library/exceptions.html#NotADirectoryError" class="reference internal" title="NotADirectoryError"><span class="pre"><code class="sourceCode python"><span class="pp">NotADirectoryError</span></code></span></a>

- <a href="../library/exceptions.html#PermissionError" class="reference internal" title="PermissionError"><span class="pre"><code class="sourceCode python"><span class="pp">PermissionError</span></code></span></a>

- <a href="../library/exceptions.html#ProcessLookupError" class="reference internal" title="ProcessLookupError"><span class="pre"><code class="sourceCode python"><span class="pp">ProcessLookupError</span></code></span></a>

- <a href="../library/exceptions.html#TimeoutError" class="reference internal" title="TimeoutError"><span class="pre"><code class="sourceCode python"><span class="pp">TimeoutError</span></code></span></a>

And the <a href="../library/exceptions.html#ConnectionError" class="reference internal" title="ConnectionError"><span class="pre"><code class="sourceCode python"><span class="pp">ConnectionError</span></code></span></a> itself has finer-grained subclasses:

- <a href="../library/exceptions.html#BrokenPipeError" class="reference internal" title="BrokenPipeError"><span class="pre"><code class="sourceCode python"><span class="pp">BrokenPipeError</span></code></span></a>

- <a href="../library/exceptions.html#ConnectionAbortedError" class="reference internal" title="ConnectionAbortedError"><span class="pre"><code class="sourceCode python"><span class="pp">ConnectionAbortedError</span></code></span></a>

- <a href="../library/exceptions.html#ConnectionRefusedError" class="reference internal" title="ConnectionRefusedError"><span class="pre"><code class="sourceCode python"><span class="pp">ConnectionRefusedError</span></code></span></a>

- <a href="../library/exceptions.html#ConnectionResetError" class="reference internal" title="ConnectionResetError"><span class="pre"><code class="sourceCode python"><span class="pp">ConnectionResetError</span></code></span></a>

Thanks to the new exceptions, common usages of the <a href="../library/errno.html#module-errno" class="reference internal" title="errno: Standard errno system symbols."><span class="pre"><code class="sourceCode python">errno</code></span></a> can now be avoided. For example, the following code written for Python 3.2:

<div class="highlight-python3 notranslate">

<div class="highlight">

    from errno import ENOENT, EACCES, EPERM

    try:
        with open("document.txt") as f:
            content = f.read()
    except IOError as err:
        if err.errno == ENOENT:
            print("document.txt file is missing")
        elif err.errno in (EACCES, EPERM):
            print("You are not allowed to read document.txt")
        else:
            raise

</div>

</div>

can now be written without the <a href="../library/errno.html#module-errno" class="reference internal" title="errno: Standard errno system symbols."><span class="pre"><code class="sourceCode python">errno</code></span></a> import and without manual inspection of exception attributes:

<div class="highlight-python3 notranslate">

<div class="highlight">

    try:
        with open("document.txt") as f:
            content = f.read()
    except FileNotFoundError:
        print("document.txt file is missing")
    except PermissionError:
        print("You are not allowed to read document.txt")

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-10" class="target"></span><a href="https://peps.python.org/pep-3151/" class="pep reference external"><strong>PEP 3151</strong></a> - Reworking the OS and IO Exception Hierarchy  
PEP written and implemented by Antoine Pitrou

</div>

</div>

<div id="pep-380-syntax-for-delegating-to-a-subgenerator" class="section">

<span id="pep-380"></span><span id="index-11"></span>

## PEP 380: Syntax for Delegating to a Subgenerator<a href="#pep-380-syntax-for-delegating-to-a-subgenerator" class="headerlink" title="Link to this heading">¶</a>

PEP 380 adds the <span class="pre">`yield`</span>` `<span class="pre">`from`</span> expression, allowing a <a href="../glossary.html#term-generator" class="reference internal"><span class="xref std std-term">generator</span></a> to delegate part of its operations to another generator. This allows a section of code containing <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> to be factored out and placed in another generator. Additionally, the subgenerator is allowed to return with a value, and the value is made available to the delegating generator.

While designed primarily for use in delegating to a subgenerator, the <span class="pre">`yield`</span>` `<span class="pre">`from`</span> expression actually allows delegation to arbitrary subiterators.

For simple iterators, <span class="pre">`yield`</span>` `<span class="pre">`from`</span>` `<span class="pre">`iterable`</span> is essentially just a shortened form of <span class="pre">`for`</span>` `<span class="pre">`item`</span>` `<span class="pre">`in`</span>` `<span class="pre">`iterable:`</span>` `<span class="pre">`yield`</span>` `<span class="pre">`item`</span>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> def g(x):
    ...     yield from range(x, 0, -1)
    ...     yield from range(x)
    ...
    >>> list(g(5))
    [5, 4, 3, 2, 1, 0, 1, 2, 3, 4]

</div>

</div>

However, unlike an ordinary loop, <span class="pre">`yield`</span>` `<span class="pre">`from`</span> allows subgenerators to receive sent and thrown values directly from the calling scope, and return a final value to the outer generator:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> def accumulate():
    ...     tally = 0
    ...     while 1:
    ...         next = yield
    ...         if next is None:
    ...             return tally
    ...         tally += next
    ...
    >>> def gather_tallies(tallies):
    ...     while 1:
    ...         tally = yield from accumulate()
    ...         tallies.append(tally)
    ...
    >>> tallies = []
    >>> acc = gather_tallies(tallies)
    >>> next(acc)  # Ensure the accumulator is ready to accept values
    >>> for i in range(4):
    ...     acc.send(i)
    ...
    >>> acc.send(None)  # Finish the first tally
    >>> for i in range(5):
    ...     acc.send(i)
    ...
    >>> acc.send(None)  # Finish the second tally
    >>> tallies
    [6, 10]

</div>

</div>

The main principle driving this change is to allow even generators that are designed to be used with the <span class="pre">`send`</span> and <span class="pre">`throw`</span> methods to be split into multiple subgenerators as easily as a single large function can be split into multiple subfunctions.

<div class="admonition seealso">

See also

<span id="index-12" class="target"></span><a href="https://peps.python.org/pep-0380/" class="pep reference external"><strong>PEP 380</strong></a> - Syntax for Delegating to a Subgenerator  
PEP written by Greg Ewing; implementation by Greg Ewing, integrated into 3.3 by Renaud Blanch, Ryan Kelly and Nick Coghlan; documentation by Zbigniew Jędrzejewski-Szmek and Nick Coghlan

</div>

</div>

<div id="pep-409-suppressing-exception-context" class="section">

## PEP 409: Suppressing exception context<a href="#pep-409-suppressing-exception-context" class="headerlink" title="Link to this heading">¶</a>

PEP 409 introduces new syntax that allows the display of the chained exception context to be disabled. This allows cleaner error messages in applications that convert between exception types:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> class D:
    ...     def __init__(self, extra):
    ...         self._extra_attributes = extra
    ...     def __getattr__(self, attr):
    ...         try:
    ...             return self._extra_attributes[attr]
    ...         except KeyError:
    ...             raise AttributeError(attr) from None
    ...
    >>> D({}).x
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 8, in __getattr__
    AttributeError: x

</div>

</div>

Without the <span class="pre">`from`</span>` `<span class="pre">`None`</span> suffix to suppress the cause, the original exception would be displayed by default:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> class C:
    ...     def __init__(self, extra):
    ...         self._extra_attributes = extra
    ...     def __getattr__(self, attr):
    ...         try:
    ...             return self._extra_attributes[attr]
    ...         except KeyError:
    ...             raise AttributeError(attr)
    ...
    >>> C({}).x
    Traceback (most recent call last):
      File "<stdin>", line 6, in __getattr__
    KeyError: 'x'

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 8, in __getattr__
    AttributeError: x

</div>

</div>

No debugging capability is lost, as the original exception context remains available if needed (for example, if an intervening library has incorrectly suppressed valuable underlying details):

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> try:
    ...     D({}).x
    ... except AttributeError as exc:
    ...     print(repr(exc.__context__))
    ...
    KeyError('x',)

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0409/" class="pep reference external"><strong>PEP 409</strong></a> - Suppressing exception context  
PEP written by Ethan Furman; implemented by Ethan Furman and Nick Coghlan.

</div>

</div>

<div id="pep-414-explicit-unicode-literals" class="section">

## PEP 414: Explicit Unicode literals<a href="#pep-414-explicit-unicode-literals" class="headerlink" title="Link to this heading">¶</a>

To ease the transition from Python 2 for Unicode aware Python applications that make heavy use of Unicode literals, Python 3.3 once again supports the “<span class="pre">`u`</span>” prefix for string literals. This prefix has no semantic significance in Python 3, it is provided solely to reduce the number of purely mechanical changes in migrating to Python 3, making it easier for developers to focus on the more significant semantic changes (such as the stricter default separation of binary and text data).

<div class="admonition seealso">

See also

<span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0414/" class="pep reference external"><strong>PEP 414</strong></a> - Explicit Unicode literals  
PEP written by Armin Ronacher.

</div>

</div>

<div id="pep-3155-qualified-name-for-classes-and-functions" class="section">

## PEP 3155: Qualified name for classes and functions<a href="#pep-3155-qualified-name-for-classes-and-functions" class="headerlink" title="Link to this heading">¶</a>

Functions and class objects have a new <a href="../library/stdtypes.html#definition.__qualname__" class="reference internal" title="definition.__qualname__"><span class="pre"><code class="sourceCode python"><span class="va">__qualname__</span></code></span></a> attribute representing the “path” from the module top-level to their definition. For global functions and classes, this is the same as <a href="../library/stdtypes.html#definition.__name__" class="reference internal" title="definition.__name__"><span class="pre"><code class="sourceCode python"><span class="va">__name__</span></code></span></a>. For other functions and classes, it provides better information about where they were actually defined, and how they might be accessible from the global scope.

Example with (non-bound) methods:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> class C:
    ...     def meth(self):
    ...         pass
    ...
    >>> C.meth.__name__
    'meth'
    >>> C.meth.__qualname__
    'C.meth'

</div>

</div>

Example with nested classes:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> class C:
    ...     class D:
    ...         def meth(self):
    ...             pass
    ...
    >>> C.D.__name__
    'D'
    >>> C.D.__qualname__
    'C.D'
    >>> C.D.meth.__name__
    'meth'
    >>> C.D.meth.__qualname__
    'C.D.meth'

</div>

</div>

Example with nested functions:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> def outer():
    ...     def inner():
    ...         pass
    ...     return inner
    ...
    >>> outer().__name__
    'inner'
    >>> outer().__qualname__
    'outer.<locals>.inner'

</div>

</div>

The string representation of those objects is also changed to include the new, more precise information:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> str(C.D)
    "<class '__main__.C.D'>"
    >>> str(C.D.meth)
    '<function C.D.meth at 0x7f46b9fe31e0>'

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-15" class="target"></span><a href="https://peps.python.org/pep-3155/" class="pep reference external"><strong>PEP 3155</strong></a> - Qualified name for classes and functions  
PEP written and implemented by Antoine Pitrou.

</div>

</div>

<div id="pep-412-key-sharing-dictionary" class="section">

<span id="pep-412"></span>

## PEP 412: Key-Sharing Dictionary<a href="#pep-412-key-sharing-dictionary" class="headerlink" title="Link to this heading">¶</a>

Dictionaries used for the storage of objects’ attributes are now able to share part of their internal storage between each other (namely, the part which stores the keys and their respective hashes). This reduces the memory consumption of programs creating many instances of non-builtin types.

<div class="admonition seealso">

See also

<span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0412/" class="pep reference external"><strong>PEP 412</strong></a> - Key-Sharing Dictionary  
PEP written and implemented by Mark Shannon.

</div>

</div>

<div id="pep-362-function-signature-object" class="section">

## PEP 362: Function Signature Object<a href="#pep-362-function-signature-object" class="headerlink" title="Link to this heading">¶</a>

A new function <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a> makes introspection of python callables easy and straightforward. A broad range of callables is supported: python functions, decorated or not, classes, and <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">functools.partial()</code></span></a> objects. New classes <a href="../library/inspect.html#inspect.Signature" class="reference internal" title="inspect.Signature"><span class="pre"><code class="sourceCode python">inspect.Signature</code></span></a>, <a href="../library/inspect.html#inspect.Parameter" class="reference internal" title="inspect.Parameter"><span class="pre"><code class="sourceCode python">inspect.Parameter</code></span></a> and <a href="../library/inspect.html#inspect.BoundArguments" class="reference internal" title="inspect.BoundArguments"><span class="pre"><code class="sourceCode python">inspect.BoundArguments</code></span></a> hold information about the call signatures, such as, annotations, default values, parameters kinds, and bound arguments, which considerably simplifies writing decorators and any code that validates or amends calling signatures or arguments.

<div class="admonition seealso">

See also

<span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0362/" class="pep reference external"><strong>PEP 362</strong></a>: - Function Signature Object  
PEP written by Brett Cannon, Yury Selivanov, Larry Hastings, Jiwon Seo; implemented by Yury Selivanov.

</div>

</div>

<div id="pep-421-adding-sys-implementation" class="section">

## PEP 421: Adding sys.implementation<a href="#pep-421-adding-sys-implementation" class="headerlink" title="Link to this heading">¶</a>

A new attribute on the <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> module exposes details specific to the implementation of the currently running interpreter. The initial set of attributes on <a href="../library/sys.html#sys.implementation" class="reference internal" title="sys.implementation"><span class="pre"><code class="sourceCode python">sys.implementation</code></span></a> are <span class="pre">`name`</span>, <span class="pre">`version`</span>, <span class="pre">`hexversion`</span>, and <span class="pre">`cache_tag`</span>.

The intention of <span class="pre">`sys.implementation`</span> is to consolidate into one namespace the implementation-specific data used by the standard library. This allows different Python implementations to share a single standard library code base much more easily. In its initial state, <span class="pre">`sys.implementation`</span> holds only a small portion of the implementation-specific data. Over time that ratio will shift in order to make the standard library more portable.

One example of improved standard library portability is <span class="pre">`cache_tag`</span>. As of Python 3.3, <span class="pre">`sys.implementation.cache_tag`</span> is used by <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> to support <span id="index-18" class="target"></span><a href="https://peps.python.org/pep-3147/" class="pep reference external"><strong>PEP 3147</strong></a> compliance. Any Python implementation that uses <span class="pre">`importlib`</span> for its built-in import system may use <span class="pre">`cache_tag`</span> to control the caching behavior for modules.

<div id="simplenamespace" class="section">

### SimpleNamespace<a href="#simplenamespace" class="headerlink" title="Link to this heading">¶</a>

The implementation of <span class="pre">`sys.implementation`</span> also introduces a new type to Python: <a href="../library/types.html#types.SimpleNamespace" class="reference internal" title="types.SimpleNamespace"><span class="pre"><code class="sourceCode python">types.SimpleNamespace</code></span></a>. In contrast to a mapping-based namespace, like <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a>, <span class="pre">`SimpleNamespace`</span> is attribute-based, like <a href="../library/functions.html#object" class="reference internal" title="object"><span class="pre"><code class="sourceCode python"><span class="bu">object</span></code></span></a>. However, unlike <span class="pre">`object`</span>, <span class="pre">`SimpleNamespace`</span> instances are writable. This means that you can add, remove, and modify the namespace through normal attribute access.

<div class="admonition seealso">

See also

<span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0421/" class="pep reference external"><strong>PEP 421</strong></a> - Adding sys.implementation  
PEP written and implemented by Eric Snow.

</div>

</div>

</div>

<div id="using-importlib-as-the-implementation-of-import" class="section">

<span id="importlib"></span>

## Using importlib as the Implementation of Import<a href="#using-importlib-as-the-implementation-of-import" class="headerlink" title="Link to this heading">¶</a>

<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2377" class="reference external">bpo-2377</a> - Replace \_\_import\_\_ w/ importlib.\_\_import\_\_ <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13959" class="reference external">bpo-13959</a> - Re-implement parts of <span class="pre">`imp`</span> in pure Python <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14605" class="reference external">bpo-14605</a> - Make import machinery explicit <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14646" class="reference external">bpo-14646</a> - Require loaders set \_\_loader\_\_ and \_\_package\_\_

The <a href="../library/functions.html#import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> function is now powered by <a href="../library/importlib.html#importlib.__import__" class="reference internal" title="importlib.__import__"><span class="pre"><code class="sourceCode python">importlib.<span class="bu">__import__</span>()</code></span></a>. This work leads to the completion of “phase 2” of <span id="index-20" class="target"></span><a href="https://peps.python.org/pep-0302/" class="pep reference external"><strong>PEP 302</strong></a>. There are multiple benefits to this change. First, it has allowed for more of the machinery powering import to be exposed instead of being implicit and hidden within the C code. It also provides a single implementation for all Python VMs supporting Python 3.3 to use, helping to end any VM-specific deviations in import semantics. And finally it eases the maintenance of import, allowing for future growth to occur.

For the common user, there should be no visible change in semantics. For those whose code currently manipulates import or calls import programmatically, the code changes that might possibly be required are covered in the <a href="#porting-python-code" class="reference internal">Porting Python code</a> section of this document.

<div id="new-apis" class="section">

### New APIs<a href="#new-apis" class="headerlink" title="Link to this heading">¶</a>

One of the large benefits of this work is the exposure of what goes into making the import statement work. That means the various importers that were once implicit are now fully exposed as part of the <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> package.

The abstract base classes defined in <a href="../library/importlib.html#module-importlib.abc" class="reference internal" title="importlib.abc: Abstract base classes related to import"><span class="pre"><code class="sourceCode python">importlib.abc</code></span></a> have been expanded to properly delineate between <a href="../glossary.html#term-meta-path-finder" class="reference internal"><span class="xref std std-term">meta path finders</span></a> and <a href="../glossary.html#term-path-entry-finder" class="reference internal"><span class="xref std std-term">path entry finders</span></a> by introducing <a href="../library/importlib.html#importlib.abc.MetaPathFinder" class="reference internal" title="importlib.abc.MetaPathFinder"><span class="pre"><code class="sourceCode python">importlib.abc.MetaPathFinder</code></span></a> and <a href="../library/importlib.html#importlib.abc.PathEntryFinder" class="reference internal" title="importlib.abc.PathEntryFinder"><span class="pre"><code class="sourceCode python">importlib.abc.PathEntryFinder</code></span></a>, respectively. The old ABC of <span class="pre">`importlib.abc.Finder`</span> is now only provided for backwards-compatibility and does not enforce any method requirements.

In terms of finders, <a href="../library/importlib.html#importlib.machinery.FileFinder" class="reference internal" title="importlib.machinery.FileFinder"><span class="pre"><code class="sourceCode python">importlib.machinery.FileFinder</code></span></a> exposes the mechanism used to search for source and bytecode files of a module. Previously this class was an implicit member of <a href="../library/sys.html#sys.path_hooks" class="reference internal" title="sys.path_hooks"><span class="pre"><code class="sourceCode python">sys.path_hooks</code></span></a>.

For loaders, the new abstract base class <a href="../library/importlib.html#importlib.abc.FileLoader" class="reference internal" title="importlib.abc.FileLoader"><span class="pre"><code class="sourceCode python">importlib.abc.FileLoader</code></span></a> helps write a loader that uses the file system as the storage mechanism for a module’s code. The loader for source files (<a href="../library/importlib.html#importlib.machinery.SourceFileLoader" class="reference internal" title="importlib.machinery.SourceFileLoader"><span class="pre"><code class="sourceCode python">importlib.machinery.SourceFileLoader</code></span></a>), sourceless bytecode files (<a href="../library/importlib.html#importlib.machinery.SourcelessFileLoader" class="reference internal" title="importlib.machinery.SourcelessFileLoader"><span class="pre"><code class="sourceCode python">importlib.machinery.SourcelessFileLoader</code></span></a>), and extension modules (<a href="../library/importlib.html#importlib.machinery.ExtensionFileLoader" class="reference internal" title="importlib.machinery.ExtensionFileLoader"><span class="pre"><code class="sourceCode python">importlib.machinery.ExtensionFileLoader</code></span></a>) are now available for direct use.

<a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> now has <span class="pre">`name`</span> and <span class="pre">`path`</span> attributes which are set when there is relevant data to provide. The message for failed imports will also provide the full name of the module now instead of just the tail end of the module’s name.

The <a href="../library/importlib.html#importlib.invalidate_caches" class="reference internal" title="importlib.invalidate_caches"><span class="pre"><code class="sourceCode python">importlib.invalidate_caches()</code></span></a> function will now call the method with the same name on all finders cached in <a href="../library/sys.html#sys.path_importer_cache" class="reference internal" title="sys.path_importer_cache"><span class="pre"><code class="sourceCode python">sys.path_importer_cache</code></span></a> to help clean up any stored state as necessary.

</div>

<div id="visible-changes" class="section">

### Visible Changes<a href="#visible-changes" class="headerlink" title="Link to this heading">¶</a>

For potential required changes to code, see the <a href="#porting-python-code" class="reference internal">Porting Python code</a> section.

Beyond the expanse of what <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> now exposes, there are other visible changes to import. The biggest is that <a href="../library/sys.html#sys.meta_path" class="reference internal" title="sys.meta_path"><span class="pre"><code class="sourceCode python">sys.meta_path</code></span></a> and <a href="../library/sys.html#sys.path_hooks" class="reference internal" title="sys.path_hooks"><span class="pre"><code class="sourceCode python">sys.path_hooks</code></span></a> now store all of the meta path finders and path entry hooks used by import. Previously the finders were implicit and hidden within the C code of import instead of being directly exposed. This means that one can now easily remove or change the order of the various finders to fit one’s needs.

Another change is that all modules have a <span class="pre">`__loader__`</span> attribute, storing the loader used to create the module. <span id="index-21" class="target"></span><a href="https://peps.python.org/pep-0302/" class="pep reference external"><strong>PEP 302</strong></a> has been updated to make this attribute mandatory for loaders to implement, so in the future once 3rd-party loaders have been updated people will be able to rely on the existence of the attribute. Until such time, though, import is setting the module post-load.

Loaders are also now expected to set the <span class="pre">`__package__`</span> attribute from <span id="index-22" class="target"></span><a href="https://peps.python.org/pep-0366/" class="pep reference external"><strong>PEP 366</strong></a>. Once again, import itself is already setting this on all loaders from <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> and import itself is setting the attribute post-load.

<span class="pre">`None`</span> is now inserted into <a href="../library/sys.html#sys.path_importer_cache" class="reference internal" title="sys.path_importer_cache"><span class="pre"><code class="sourceCode python">sys.path_importer_cache</code></span></a> when no finder can be found on <a href="../library/sys.html#sys.path_hooks" class="reference internal" title="sys.path_hooks"><span class="pre"><code class="sourceCode python">sys.path_hooks</code></span></a>. Since <span class="pre">`imp.NullImporter`</span> is not directly exposed on <a href="../library/sys.html#sys.path_hooks" class="reference internal" title="sys.path_hooks"><span class="pre"><code class="sourceCode python">sys.path_hooks</code></span></a> it could no longer be relied upon to always be available to use as a value representing no finder found.

All other changes relate to semantic changes which should be taken into consideration when updating code for Python 3.3, and thus should be read about in the <a href="#porting-python-code" class="reference internal">Porting Python code</a> section of this document.

(Implementation by Brett Cannon)

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

Some smaller changes made to the core Python language are:

- Added support for Unicode name aliases and named sequences. Both <a href="../library/unicodedata.html#unicodedata.lookup" class="reference internal" title="unicodedata.lookup"><span class="pre"><code class="sourceCode python">unicodedata.lookup()</code></span></a> and <span class="pre">`'\N{...}'`</span> now resolve name aliases, and <a href="../library/unicodedata.html#unicodedata.lookup" class="reference internal" title="unicodedata.lookup"><span class="pre"><code class="sourceCode python">unicodedata.lookup()</code></span></a> resolves named sequences too.

  (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12753" class="reference external">bpo-12753</a>.)

- Unicode database updated to UCD version 6.1.0

- Equality comparisons on <a href="../library/stdtypes.html#range" class="reference internal" title="range"><span class="pre"><code class="sourceCode python"><span class="bu">range</span>()</code></span></a> objects now return a result reflecting the equality of the underlying sequences generated by those range objects. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13201" class="reference external">bpo-13201</a>)

- The <span class="pre">`count()`</span>, <span class="pre">`find()`</span>, <span class="pre">`rfind()`</span>, <span class="pre">`index()`</span> and <span class="pre">`rindex()`</span> methods of <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> objects now accept an integer between 0 and 255 as their first argument.

  (Contributed by Petri Lehtinen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12170" class="reference external">bpo-12170</a>.)

- The <span class="pre">`rjust()`</span>, <span class="pre">`ljust()`</span>, and <span class="pre">`center()`</span> methods of <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> now accept a <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> for the <span class="pre">`fill`</span> argument. (Contributed by Petri Lehtinen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12380" class="reference external">bpo-12380</a>.)

- New methods have been added to <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>: <span class="pre">`copy()`</span> and <span class="pre">`clear()`</span> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10516" class="reference external">bpo-10516</a>). Consequently, <a href="../library/collections.abc.html#collections.abc.MutableSequence" class="reference internal" title="collections.abc.MutableSequence"><span class="pre"><code class="sourceCode python">MutableSequence</code></span></a> now also defines a <span class="pre">`clear()`</span> method (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11388" class="reference external">bpo-11388</a>).

- Raw bytes literals can now be written <span class="pre">`rb"..."`</span> as well as <span class="pre">`br"..."`</span>.

  (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13748" class="reference external">bpo-13748</a>.)

- <a href="../library/stdtypes.html#dict.setdefault" class="reference internal" title="dict.setdefault"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.setdefault()</code></span></a> now does only one lookup for the given key, making it atomic when used with built-in types.

  (Contributed by Filip Gruszczyński in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13521" class="reference external">bpo-13521</a>.)

- The error messages produced when a function call does not match the function signature have been significantly improved.

  (Contributed by Benjamin Peterson.)

</div>

<div id="a-finer-grained-import-lock" class="section">

## A Finer-Grained Import Lock<a href="#a-finer-grained-import-lock" class="headerlink" title="Link to this heading">¶</a>

Previous versions of CPython have always relied on a global import lock. This led to unexpected annoyances, such as deadlocks when importing a module would trigger code execution in a different thread as a side-effect. Clumsy workarounds were sometimes employed, such as the <a href="../c-api/import.html#c.PyImport_ImportModuleNoBlock" class="reference internal" title="PyImport_ImportModuleNoBlock"><span class="pre"><code class="sourceCode c">PyImport_ImportModuleNoBlock<span class="op">()</span></code></span></a> C API function.

In Python 3.3, importing a module takes a per-module lock. This correctly serializes importation of a given module from multiple threads (preventing the exposure of incompletely initialized modules), while eliminating the aforementioned annoyances.

(Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9260" class="reference external">bpo-9260</a>.)

</div>

<div id="builtin-functions-and-types" class="section">

## Builtin functions and types<a href="#builtin-functions-and-types" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> gets a new *opener* parameter: the underlying file descriptor for the file object is then obtained by calling *opener* with (*file*, *flags*). It can be used to use custom flags like <a href="../library/os.html#os.O_CLOEXEC" class="reference internal" title="os.O_CLOEXEC"><span class="pre"><code class="sourceCode python">os.O_CLOEXEC</code></span></a> for example. The <span class="pre">`'x'`</span> mode was added: open for exclusive creation, failing if the file already exists.

- <a href="../library/functions.html#print" class="reference internal" title="print"><span class="pre"><code class="sourceCode python"><span class="bu">print</span>()</code></span></a>: added the *flush* keyword argument. If the *flush* keyword argument is true, the stream is forcibly flushed.

- <a href="../library/functions.html#hash" class="reference internal" title="hash"><span class="pre"><code class="sourceCode python"><span class="bu">hash</span>()</code></span></a>: hash randomization is enabled by default, see <a href="../reference/datamodel.html#object.__hash__" class="reference internal" title="object.__hash__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__hash__</span>()</code></span></a> and <span id="index-23" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONHASHSEED" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONHASHSEED</code></span></a>.

- The <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> type gets a new <a href="../library/stdtypes.html#str.casefold" class="reference internal" title="str.casefold"><span class="pre"><code class="sourceCode python">casefold()</code></span></a> method: return a casefolded copy of the string, casefolded strings may be used for caseless matching. For example, <span class="pre">`'ß'.casefold()`</span> returns <span class="pre">`'ss'`</span>.

- The sequence documentation has been substantially rewritten to better explain the binary/text sequence distinction and to provide specific documentation sections for the individual builtin sequence types (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4966" class="reference external">bpo-4966</a>).

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="faulthandler" class="section">

### faulthandler<a href="#faulthandler" class="headerlink" title="Link to this heading">¶</a>

This new debug module <a href="../library/faulthandler.html#module-faulthandler" class="reference internal" title="faulthandler: Dump the Python traceback."><span class="pre"><code class="sourceCode python">faulthandler</code></span></a> contains functions to dump Python tracebacks explicitly, on a fault (a crash like a segmentation fault), after a timeout, or on a user signal. Call <a href="../library/faulthandler.html#faulthandler.enable" class="reference internal" title="faulthandler.enable"><span class="pre"><code class="sourceCode python">faulthandler.enable()</code></span></a> to install fault handlers for the <a href="../library/signal.html#signal.SIGSEGV" class="reference internal" title="signal.SIGSEGV"><span class="pre"><code class="sourceCode python">SIGSEGV</code></span></a>, <a href="../library/signal.html#signal.SIGFPE" class="reference internal" title="signal.SIGFPE"><span class="pre"><code class="sourceCode python">SIGFPE</code></span></a>, <a href="../library/signal.html#signal.SIGABRT" class="reference internal" title="signal.SIGABRT"><span class="pre"><code class="sourceCode python">SIGABRT</code></span></a>, <a href="../library/signal.html#signal.SIGBUS" class="reference internal" title="signal.SIGBUS"><span class="pre"><code class="sourceCode python">SIGBUS</code></span></a>, and <a href="../library/signal.html#signal.SIGILL" class="reference internal" title="signal.SIGILL"><span class="pre"><code class="sourceCode python">SIGILL</code></span></a> signals. You can also enable them at startup by setting the <span id="index-24" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONFAULTHANDLER" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONFAULTHANDLER</code></span></a> environment variable or by using <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span></a> <span class="pre">`faulthandler`</span> command line option.

Example of a segmentation fault on Linux:

<div class="highlight-shell-session notranslate">

<div class="highlight">

    $ python -q -X faulthandler
    >>> import ctypes
    >>> ctypes.string_at(0)
    Fatal Python error: Segmentation fault

    Current thread 0x00007fb899f39700:
      File "/home/python/cpython/Lib/ctypes/__init__.py", line 486 in string_at
      File "<stdin>", line 1 in <module>
    Segmentation fault

</div>

</div>

</div>

<div id="ipaddress" class="section">

### ipaddress<a href="#ipaddress" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> module provides tools for creating and manipulating objects representing IPv4 and IPv6 addresses, networks and interfaces (i.e. an IP address associated with a specific IP subnet).

(Contributed by Google and Peter Moody in <span id="index-25" class="target"></span><a href="https://peps.python.org/pep-3144/" class="pep reference external"><strong>PEP 3144</strong></a>.)

</div>

<div id="lzma" class="section">

### lzma<a href="#lzma" class="headerlink" title="Link to this heading">¶</a>

The newly added <a href="../library/lzma.html#module-lzma" class="reference internal" title="lzma: A Python wrapper for the liblzma compression library."><span class="pre"><code class="sourceCode python">lzma</code></span></a> module provides data compression and decompression using the LZMA algorithm, including support for the <span class="pre">`.xz`</span> and <span class="pre">`.lzma`</span> file formats.

(Contributed by Nadeem Vawda and Per Øyvind Karlsen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6715" class="reference external">bpo-6715</a>.)

</div>

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="abc" class="section">

### abc<a href="#abc" class="headerlink" title="Link to this heading">¶</a>

Improved support for abstract base classes containing descriptors composed with abstract methods. The recommended approach to declaring abstract descriptors is now to provide <span class="pre">`__isabstractmethod__`</span> as a dynamically updated property. The built-in descriptors have been updated accordingly.

- <a href="../library/abc.html#abc.abstractproperty" class="reference internal" title="abc.abstractproperty"><span class="pre"><code class="sourceCode python">abc.abstractproperty</code></span></a> has been deprecated, use <a href="../library/functions.html#property" class="reference internal" title="property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span></code></span></a> with <a href="../library/abc.html#abc.abstractmethod" class="reference internal" title="abc.abstractmethod"><span class="pre"><code class="sourceCode python">abc.abstractmethod()</code></span></a> instead.

- <a href="../library/abc.html#abc.abstractclassmethod" class="reference internal" title="abc.abstractclassmethod"><span class="pre"><code class="sourceCode python">abc.abstractclassmethod</code></span></a> has been deprecated, use <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span></code></span></a> with <a href="../library/abc.html#abc.abstractmethod" class="reference internal" title="abc.abstractmethod"><span class="pre"><code class="sourceCode python">abc.abstractmethod()</code></span></a> instead.

- <a href="../library/abc.html#abc.abstractstaticmethod" class="reference internal" title="abc.abstractstaticmethod"><span class="pre"><code class="sourceCode python">abc.abstractstaticmethod</code></span></a> has been deprecated, use <a href="../library/functions.html#staticmethod" class="reference internal" title="staticmethod"><span class="pre"><code class="sourceCode python"><span class="bu">staticmethod</span></code></span></a> with <a href="../library/abc.html#abc.abstractmethod" class="reference internal" title="abc.abstractmethod"><span class="pre"><code class="sourceCode python">abc.abstractmethod()</code></span></a> instead.

(Contributed by Darren Dale in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11610" class="reference external">bpo-11610</a>.)

<a href="../library/abc.html#abc.ABCMeta.register" class="reference internal" title="abc.ABCMeta.register"><span class="pre"><code class="sourceCode python">abc.ABCMeta.register()</code></span></a> now returns the registered subclass, which means it can now be used as a class decorator (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10868" class="reference external">bpo-10868</a>).

</div>

<div id="array" class="section">

### array<a href="#array" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/array.html#module-array" class="reference internal" title="array: Space efficient arrays of uniformly typed numeric values."><span class="pre"><code class="sourceCode python">array</code></span></a> module supports the <span class="c-expr sig sig-inline c"><span class="kt">long</span><span class="w"> </span><span class="kt">long</span></span> type using <span class="pre">`q`</span> and <span class="pre">`Q`</span> type codes.

(Contributed by Oren Tirosh and Hirokazu Yamamoto in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1172711" class="reference external">bpo-1172711</a>.)

</div>

<div id="base64" class="section">

### base64<a href="#base64" class="headerlink" title="Link to this heading">¶</a>

ASCII-only Unicode strings are now accepted by the decoding functions of the <a href="../library/base64.html#module-base64" class="reference internal" title="base64: RFC 4648: Base16, Base32, Base64 Data Encodings; Base85 and Ascii85"><span class="pre"><code class="sourceCode python">base64</code></span></a> modern interface. For example, <span class="pre">`base64.b64decode('YWJj')`</span> returns <span class="pre">`b'abc'`</span>. (Contributed by Catalin Iacob in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13641" class="reference external">bpo-13641</a>.)

</div>

<div id="binascii" class="section">

### binascii<a href="#binascii" class="headerlink" title="Link to this heading">¶</a>

In addition to the binary objects they normally accept, the <span class="pre">`a2b_`</span> functions now all also accept ASCII-only strings as input. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13637" class="reference external">bpo-13637</a>.)

</div>

<div id="bz2" class="section">

### bz2<a href="#bz2" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/bz2.html#module-bz2" class="reference internal" title="bz2: Interfaces for bzip2 compression and decompression."><span class="pre"><code class="sourceCode python">bz2</code></span></a> module has been rewritten from scratch. In the process, several new features have been added:

- New <a href="../library/bz2.html#bz2.open" class="reference internal" title="bz2.open"><span class="pre"><code class="sourceCode python">bz2.<span class="bu">open</span>()</code></span></a> function: open a bzip2-compressed file in binary or text mode.

- <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">bz2.BZ2File</code></span></a> can now read from and write to arbitrary file-like objects, by means of its constructor’s *fileobj* argument.

  (Contributed by Nadeem Vawda in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5863" class="reference external">bpo-5863</a>.)

- <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">bz2.BZ2File</code></span></a> and <a href="../library/bz2.html#bz2.decompress" class="reference internal" title="bz2.decompress"><span class="pre"><code class="sourceCode python">bz2.decompress()</code></span></a> can now decompress multi-stream inputs (such as those produced by the **pbzip2** tool). <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">bz2.BZ2File</code></span></a> can now also be used to create this type of file, using the <span class="pre">`'a'`</span> (append) mode.

  (Contributed by Nir Aides in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1625" class="reference external">bpo-1625</a>.)

- <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">bz2.BZ2File</code></span></a> now implements all of the <a href="../library/io.html#io.BufferedIOBase" class="reference internal" title="io.BufferedIOBase"><span class="pre"><code class="sourceCode python">io.BufferedIOBase</code></span></a> API, except for the <span class="pre">`detach()`</span> and <span class="pre">`truncate()`</span> methods.

</div>

<div id="codecs" class="section">

### codecs<a href="#codecs" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/codecs.html#module-encodings.mbcs" class="reference internal" title="encodings.mbcs: Windows ANSI codepage"><span class="pre"><code class="sourceCode python">mbcs</code></span></a> codec has been rewritten to handle correctly <span class="pre">`replace`</span> and <span class="pre">`ignore`</span> error handlers on all Windows versions. The <a href="../library/codecs.html#module-encodings.mbcs" class="reference internal" title="encodings.mbcs: Windows ANSI codepage"><span class="pre"><code class="sourceCode python">mbcs</code></span></a> codec now supports all error handlers, instead of only <span class="pre">`replace`</span> to encode and <span class="pre">`ignore`</span> to decode.

A new Windows-only codec has been added: <span class="pre">`cp65001`</span> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13216" class="reference external">bpo-13216</a>). It is the Windows code page 65001 (Windows UTF-8, <span class="pre">`CP_UTF8`</span>). For example, it is used by <span class="pre">`sys.stdout`</span> if the console output code page is set to cp65001 (e.g., using <span class="pre">`chcp`</span>` `<span class="pre">`65001`</span> command).

Multibyte CJK decoders now resynchronize faster. They only ignore the first byte of an invalid byte sequence. For example, <span class="pre">`b'\xff\n'.decode('gb2312',`</span>` `<span class="pre">`'replace')`</span> now returns a <span class="pre">`\n`</span> after the replacement character.

(<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12016" class="reference external">bpo-12016</a>)

Incremental CJK codec encoders are no longer reset at each call to their encode() methods. For example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import codecs
    >>> encoder = codecs.getincrementalencoder('hz')('strict')
    >>> b''.join(encoder.encode(x) for x in '\u52ff\u65bd\u65bc\u4eba\u3002 Bye.')
    b'~{NpJ)l6HK!#~} Bye.'

</div>

</div>

This example gives <span class="pre">`b'~{Np~}~{J)~}~{l6~}~{HK~}~{!#~}`</span>` `<span class="pre">`Bye.'`</span> with older Python versions.

(<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12100" class="reference external">bpo-12100</a>)

The <span class="pre">`unicode_internal`</span> codec has been deprecated.

</div>

<div id="collections" class="section">

### collections<a href="#collections" class="headerlink" title="Link to this heading">¶</a>

Addition of a new <a href="../library/collections.html#collections.ChainMap" class="reference internal" title="collections.ChainMap"><span class="pre"><code class="sourceCode python">ChainMap</code></span></a> class to allow treating a number of mappings as a single unit. (Written by Raymond Hettinger for <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11089" class="reference external">bpo-11089</a>, made public in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11297" class="reference external">bpo-11297</a>.)

The abstract base classes have been moved in a new <a href="../library/collections.abc.html#module-collections.abc" class="reference internal" title="collections.abc: Abstract base classes for containers"><span class="pre"><code class="sourceCode python">collections.abc</code></span></a> module, to better differentiate between the abstract and the concrete collections classes. Aliases for ABCs are still present in the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: Container datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module to preserve existing imports. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11085" class="reference external">bpo-11085</a>)

The <a href="../library/collections.html#collections.Counter" class="reference internal" title="collections.Counter"><span class="pre"><code class="sourceCode python">Counter</code></span></a> class now supports the unary <span class="pre">`+`</span> and <span class="pre">`-`</span> operators, as well as the in-place operators <span class="pre">`+=`</span>, <span class="pre">`-=`</span>, <span class="pre">`|=`</span>, and <span class="pre">`&=`</span>. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13121" class="reference external">bpo-13121</a>.)

</div>

<div id="contextlib" class="section">

### contextlib<a href="#contextlib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/contextlib.html#contextlib.ExitStack" class="reference internal" title="contextlib.ExitStack"><span class="pre"><code class="sourceCode python">ExitStack</code></span></a> now provides a solid foundation for programmatic manipulation of context managers and similar cleanup functionality. Unlike the previous <span class="pre">`contextlib.nested`</span> API (which was deprecated and removed), the new API is designed to work correctly regardless of whether context managers acquire their resources in their <span class="pre">`__init__`</span> method (for example, file objects) or in their <span class="pre">`__enter__`</span> method (for example, synchronisation objects from the <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a> module).

(<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13585" class="reference external">bpo-13585</a>)

</div>

<div id="crypt" class="section">

### crypt<a href="#crypt" class="headerlink" title="Link to this heading">¶</a>

Addition of salt and modular crypt format (hashing method) and the <span class="pre">`mksalt()`</span> function to the <span class="pre">`crypt`</span> module.

(<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10924" class="reference external">bpo-10924</a>)

</div>

<div id="curses" class="section">

### curses<a href="#curses" class="headerlink" title="Link to this heading">¶</a>

- If the <a href="../library/curses.html#module-curses" class="reference internal" title="curses: An interface to the curses library, providing portable terminal handling. (Unix)"><span class="pre"><code class="sourceCode python">curses</code></span></a> module is linked to the ncursesw library, use Unicode functions when Unicode strings or characters are passed (e.g. <span class="pre">`waddwstr()`</span>), and bytes functions otherwise (e.g. <span class="pre">`waddstr()`</span>).

- Use the locale encoding instead of <span class="pre">`utf-8`</span> to encode Unicode strings.

- <a href="../library/curses.html#curses.window" class="reference internal" title="curses.window"><span class="pre"><code class="sourceCode python">curses.window</code></span></a> has a new <a href="../library/curses.html#curses.window.encoding" class="reference internal" title="curses.window.encoding"><span class="pre"><code class="sourceCode python">curses.window.encoding</code></span></a> attribute.

- The <a href="../library/curses.html#curses.window" class="reference internal" title="curses.window"><span class="pre"><code class="sourceCode python">curses.window</code></span></a> class has a new <a href="../library/curses.html#curses.window.get_wch" class="reference internal" title="curses.window.get_wch"><span class="pre"><code class="sourceCode python">get_wch()</code></span></a> method to get a wide character

- The <a href="../library/curses.html#module-curses" class="reference internal" title="curses: An interface to the curses library, providing portable terminal handling. (Unix)"><span class="pre"><code class="sourceCode python">curses</code></span></a> module has a new <a href="../library/curses.html#curses.unget_wch" class="reference internal" title="curses.unget_wch"><span class="pre"><code class="sourceCode python">unget_wch()</code></span></a> function to push a wide character so the next <a href="../library/curses.html#curses.window.get_wch" class="reference internal" title="curses.window.get_wch"><span class="pre"><code class="sourceCode python">get_wch()</code></span></a> will return it

(Contributed by Iñigo Serna in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6755" class="reference external">bpo-6755</a>.)

</div>

<div id="datetime" class="section">

### datetime<a href="#datetime" class="headerlink" title="Link to this heading">¶</a>

- Equality comparisons between naive and aware <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> instances now return <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a> instead of raising <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15006" class="reference external">bpo-15006</a>).

- New <a href="../library/datetime.html#datetime.datetime.timestamp" class="reference internal" title="datetime.datetime.timestamp"><span class="pre"><code class="sourceCode python">datetime.datetime.timestamp()</code></span></a> method: Return POSIX timestamp corresponding to the <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> instance.

- The <a href="../library/datetime.html#datetime.datetime.strftime" class="reference internal" title="datetime.datetime.strftime"><span class="pre"><code class="sourceCode python">datetime.datetime.strftime()</code></span></a> method supports formatting years older than 1000.

- The <a href="../library/datetime.html#datetime.datetime.astimezone" class="reference internal" title="datetime.datetime.astimezone"><span class="pre"><code class="sourceCode python">datetime.datetime.astimezone()</code></span></a> method can now be called without arguments to convert datetime instance to the system timezone.

</div>

<div id="decimal" class="section">

<span id="new-decimal"></span>

### decimal<a href="#decimal" class="headerlink" title="Link to this heading">¶</a>

<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7652" class="reference external">bpo-7652</a> - integrate fast native decimal arithmetic.  
C-module and libmpdec written by Stefan Krah.

The new C version of the decimal module integrates the high speed libmpdec library for arbitrary precision correctly rounded decimal floating-point arithmetic. libmpdec conforms to IBM’s General Decimal Arithmetic Specification.

Performance gains range from 10x for database applications to 100x for numerically intensive applications. These numbers are expected gains for standard precisions used in decimal floating-point arithmetic. Since the precision is user configurable, the exact figures may vary. For example, in integer bignum arithmetic the differences can be significantly higher.

The following table is meant as an illustration. Benchmarks are available at <a href="https://www.bytereef.org/mpdecimal/quickstart.html" class="reference external">https://www.bytereef.org/mpdecimal/quickstart.html</a>.

> <div>
>
> |         | decimal.py | \_decimal | speedup |
> |---------|------------|-----------|---------|
> | pi      | 42.02s     | 0.345s    | 120x    |
> | telco   | 172.19s    | 5.68s     | 30x     |
> | psycopg | 3.57s      | 0.29s     | 12x     |
>
> </div>

<div id="id1" class="section">

#### Features<a href="#id1" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/decimal.html#decimal.FloatOperation" class="reference internal" title="decimal.FloatOperation"><span class="pre"><code class="sourceCode python">FloatOperation</code></span></a> signal optionally enables stricter semantics for mixing floats and Decimals.

- If Python is compiled without threads, the C version automatically disables the expensive thread local context machinery. In this case, the variable <a href="../library/decimal.html#decimal.HAVE_THREADS" class="reference internal" title="decimal.HAVE_THREADS"><span class="pre"><code class="sourceCode python">HAVE_THREADS</code></span></a> is set to <span class="pre">`False`</span>.

</div>

<div id="id2" class="section">

#### API changes<a href="#id2" class="headerlink" title="Link to this heading">¶</a>

- The C module has the following context limits, depending on the machine architecture:

  > <div>
  >
  > |  | 32-bit | 64-bit |
  > |----|----|----|
  > | <a href="../library/decimal.html#decimal.MAX_PREC" class="reference internal" title="decimal.MAX_PREC"><span class="pre"><code class="sourceCode python">MAX_PREC</code></span></a> | <span class="pre">`425000000`</span> | <span class="pre">`999999999999999999`</span> |
  > | <a href="../library/decimal.html#decimal.MAX_EMAX" class="reference internal" title="decimal.MAX_EMAX"><span class="pre"><code class="sourceCode python">MAX_EMAX</code></span></a> | <span class="pre">`425000000`</span> | <span class="pre">`999999999999999999`</span> |
  > | <a href="../library/decimal.html#decimal.MIN_EMIN" class="reference internal" title="decimal.MIN_EMIN"><span class="pre"><code class="sourceCode python">MIN_EMIN</code></span></a> | <span class="pre">`-425000000`</span> | <span class="pre">`-999999999999999999`</span> |
  >
  > </div>

- In the context templates (<a href="../library/decimal.html#decimal.DefaultContext" class="reference internal" title="decimal.DefaultContext"><span class="pre"><code class="sourceCode python">DefaultContext</code></span></a>, <a href="../library/decimal.html#decimal.BasicContext" class="reference internal" title="decimal.BasicContext"><span class="pre"><code class="sourceCode python">BasicContext</code></span></a> and <a href="../library/decimal.html#decimal.ExtendedContext" class="reference internal" title="decimal.ExtendedContext"><span class="pre"><code class="sourceCode python">ExtendedContext</code></span></a>) the magnitude of <a href="../library/decimal.html#decimal.Context.Emax" class="reference internal" title="decimal.Context.Emax"><span class="pre"><code class="sourceCode python">Emax</code></span></a> and <a href="../library/decimal.html#decimal.Context.Emin" class="reference internal" title="decimal.Context.Emin"><span class="pre"><code class="sourceCode python">Emin</code></span></a> has changed to <span class="pre">`999999`</span>.

- The <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> constructor in decimal.py does not observe the context limits and converts values with arbitrary exponents or precision exactly. Since the C version has internal limits, the following scheme is used: If possible, values are converted exactly, otherwise <a href="../library/decimal.html#decimal.InvalidOperation" class="reference internal" title="decimal.InvalidOperation"><span class="pre"><code class="sourceCode python">InvalidOperation</code></span></a> is raised and the result is NaN. In the latter case it is always possible to use <a href="../library/decimal.html#decimal.Context.create_decimal" class="reference internal" title="decimal.Context.create_decimal"><span class="pre"><code class="sourceCode python">create_decimal()</code></span></a> in order to obtain a rounded or inexact value.

- The power function in decimal.py is always correctly rounded. In the C version, it is defined in terms of the correctly rounded <a href="../library/decimal.html#decimal.Decimal.exp" class="reference internal" title="decimal.Decimal.exp"><span class="pre"><code class="sourceCode python">exp()</code></span></a> and <a href="../library/decimal.html#decimal.Decimal.ln" class="reference internal" title="decimal.Decimal.ln"><span class="pre"><code class="sourceCode python">ln()</code></span></a> functions, but the final result is only “almost always correctly rounded”.

- In the C version, the context dictionary containing the signals is a <a href="../library/collections.abc.html#collections.abc.MutableMapping" class="reference internal" title="collections.abc.MutableMapping"><span class="pre"><code class="sourceCode python">MutableMapping</code></span></a>. For speed reasons, <a href="../library/decimal.html#decimal.Context.flags" class="reference internal" title="decimal.Context.flags"><span class="pre"><code class="sourceCode python">flags</code></span></a> and <a href="../library/decimal.html#decimal.Context.traps" class="reference internal" title="decimal.Context.traps"><span class="pre"><code class="sourceCode python">traps</code></span></a> always refer to the same <a href="../library/collections.abc.html#collections.abc.MutableMapping" class="reference internal" title="collections.abc.MutableMapping"><span class="pre"><code class="sourceCode python">MutableMapping</code></span></a> that the context was initialized with. If a new signal dictionary is assigned, <a href="../library/decimal.html#decimal.Context.flags" class="reference internal" title="decimal.Context.flags"><span class="pre"><code class="sourceCode python">flags</code></span></a> and <a href="../library/decimal.html#decimal.Context.traps" class="reference internal" title="decimal.Context.traps"><span class="pre"><code class="sourceCode python">traps</code></span></a> are updated with the new values, but they do not reference the RHS dictionary.

- Pickling a <a href="../library/decimal.html#decimal.Context" class="reference internal" title="decimal.Context"><span class="pre"><code class="sourceCode python">Context</code></span></a> produces a different output in order to have a common interchange format for the Python and C versions.

- The order of arguments in the <a href="../library/decimal.html#decimal.Context" class="reference internal" title="decimal.Context"><span class="pre"><code class="sourceCode python">Context</code></span></a> constructor has been changed to match the order displayed by <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a>.

- The <span class="pre">`watchexp`</span> parameter in the <a href="../library/decimal.html#decimal.Decimal.quantize" class="reference internal" title="decimal.Decimal.quantize"><span class="pre"><code class="sourceCode python">quantize()</code></span></a> method is deprecated.

</div>

</div>

<div id="email" class="section">

<span id="new-email"></span>

### email<a href="#email" class="headerlink" title="Link to this heading">¶</a>

<div id="policy-framework" class="section">

#### Policy Framework<a href="#policy-framework" class="headerlink" title="Link to this heading">¶</a>

The email package now has a <a href="../library/email.policy.html#module-email.policy" class="reference internal" title="email.policy: Controlling the parsing and generating of messages"><span class="pre"><code class="sourceCode python">policy</code></span></a> framework. A <a href="../library/email.policy.html#email.policy.Policy" class="reference internal" title="email.policy.Policy"><span class="pre"><code class="sourceCode python">Policy</code></span></a> is an object with several methods and properties that control how the email package behaves. The primary policy for Python 3.3 is the <a href="../library/email.policy.html#email.policy.Compat32" class="reference internal" title="email.policy.Compat32"><span class="pre"><code class="sourceCode python">Compat32</code></span></a> policy, which provides backward compatibility with the email package in Python 3.2. A <span class="pre">`policy`</span> can be specified when an email message is parsed by a <a href="../library/email.parser.html#module-email.parser" class="reference internal" title="email.parser: Parse flat text email messages to produce a message object structure."><span class="pre"><code class="sourceCode python">parser</code></span></a>, or when a <a href="../library/email.compat32-message.html#email.message.Message" class="reference internal" title="email.message.Message"><span class="pre"><code class="sourceCode python">Message</code></span></a> object is created, or when an email is serialized using a <a href="../library/email.generator.html#module-email.generator" class="reference internal" title="email.generator: Generate flat text email messages from a message structure."><span class="pre"><code class="sourceCode python">generator</code></span></a>. Unless overridden, a policy passed to a <span class="pre">`parser`</span> is inherited by all the <span class="pre">`Message`</span> object and sub-objects created by the <span class="pre">`parser`</span>. By default a <span class="pre">`generator`</span> will use the policy of the <span class="pre">`Message`</span> object it is serializing. The default policy is <a href="../library/email.policy.html#email.policy.compat32" class="reference internal" title="email.policy.compat32"><span class="pre"><code class="sourceCode python">compat32</code></span></a>.

The minimum set of controls implemented by all <span class="pre">`policy`</span> objects are:

|  |  |
|----|----|
| max_line_length | The maximum length, excluding the linesep character(s), individual lines may have when a <span class="pre">`Message`</span> is serialized. Defaults to 78. |
| linesep | The character used to separate individual lines when a <span class="pre">`Message`</span> is serialized. Defaults to <span class="pre">`\n`</span>. |
| cte_type | <span class="pre">`7bit`</span> or <span class="pre">`8bit`</span>. <span class="pre">`8bit`</span> applies only to a <span class="pre">`Bytes`</span> <span class="pre">`generator`</span>, and means that non-ASCII may be used where allowed by the protocol (or where it exists in the original input). |
| raise_on_defect | Causes a <span class="pre">`parser`</span> to raise error when defects are encountered instead of adding them to the <span class="pre">`Message`</span> object’s <span class="pre">`defects`</span> list. |

A new policy instance, with new settings, is created using the <a href="../library/email.policy.html#email.policy.Policy.clone" class="reference internal" title="email.policy.Policy.clone"><span class="pre"><code class="sourceCode python">clone()</code></span></a> method of policy objects. <span class="pre">`clone`</span> takes any of the above controls as keyword arguments. Any control not specified in the call retains its default value. Thus you can create a policy that uses <span class="pre">`\r\n`</span> linesep characters like this:

<div class="highlight-python3 notranslate">

<div class="highlight">

    mypolicy = compat32.clone(linesep='\r\n')

</div>

</div>

Policies can be used to make the generation of messages in the format needed by your application simpler. Instead of having to remember to specify <span class="pre">`linesep='\r\n'`</span> in all the places you call a <span class="pre">`generator`</span>, you can specify it once, when you set the policy used by the <span class="pre">`parser`</span> or the <span class="pre">`Message`</span>, whichever your program uses to create <span class="pre">`Message`</span> objects. On the other hand, if you need to generate messages in multiple forms, you can still specify the parameters in the appropriate <span class="pre">`generator`</span> call. Or you can have custom policy instances for your different cases, and pass those in when you create the <span class="pre">`generator`</span>.

</div>

<div id="provisional-policy-with-new-header-api" class="section">

#### Provisional Policy with New Header API<a href="#provisional-policy-with-new-header-api" class="headerlink" title="Link to this heading">¶</a>

While the policy framework is worthwhile all by itself, the main motivation for introducing it is to allow the creation of new policies that implement new features for the email package in a way that maintains backward compatibility for those who do not use the new policies. Because the new policies introduce a new API, we are releasing them in Python 3.3 as a <a href="../glossary.html#term-provisional-package" class="reference internal"><span class="xref std std-term">provisional policy</span></a>. Backwards incompatible changes (up to and including removal of the code) may occur if deemed necessary by the core developers.

The new policies are instances of <a href="../library/email.policy.html#email.policy.EmailPolicy" class="reference internal" title="email.policy.EmailPolicy"><span class="pre"><code class="sourceCode python">EmailPolicy</code></span></a>, and add the following additional controls:

|  |  |
|----|----|
| refold_source | Controls whether or not headers parsed by a <a href="../library/email.parser.html#module-email.parser" class="reference internal" title="email.parser: Parse flat text email messages to produce a message object structure."><span class="pre"><code class="sourceCode python">parser</code></span></a> are refolded by the <a href="../library/email.generator.html#module-email.generator" class="reference internal" title="email.generator: Generate flat text email messages from a message structure."><span class="pre"><code class="sourceCode python">generator</code></span></a>. It can be <span class="pre">`none`</span>, <span class="pre">`long`</span>, or <span class="pre">`all`</span>. The default is <span class="pre">`long`</span>, which means that source headers with a line longer than <span class="pre">`max_line_length`</span> get refolded. <span class="pre">`none`</span> means no line get refolded, and <span class="pre">`all`</span> means that all lines get refolded. |
| header_factory | A callable that take a <span class="pre">`name`</span> and <span class="pre">`value`</span> and produces a custom header object. |

The <span class="pre">`header_factory`</span> is the key to the new features provided by the new policies. When one of the new policies is used, any header retrieved from a <span class="pre">`Message`</span> object is an object produced by the <span class="pre">`header_factory`</span>, and any time you set a header on a <span class="pre">`Message`</span> it becomes an object produced by <span class="pre">`header_factory`</span>. All such header objects have a <span class="pre">`name`</span> attribute equal to the header name. Address and Date headers have additional attributes that give you access to the parsed data of the header. This means you can now do things like this:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> m = Message(policy=SMTP)
    >>> m['To'] = 'Éric <foo@example.com>'
    >>> m['to']
    'Éric <foo@example.com>'
    >>> m['to'].addresses
    (Address(display_name='Éric', username='foo', domain='example.com'),)
    >>> m['to'].addresses[0].username
    'foo'
    >>> m['to'].addresses[0].display_name
    'Éric'
    >>> m['Date'] = email.utils.localtime()
    >>> m['Date'].datetime
    datetime.datetime(2012, 5, 25, 21, 39, 24, 465484, tzinfo=datetime.timezone(datetime.timedelta(-1, 72000), 'EDT'))
    >>> m['Date']
    'Fri, 25 May 2012 21:44:27 -0400'
    >>> print(m)
    To: =?utf-8?q?=C3=89ric?= <foo@example.com>
    Date: Fri, 25 May 2012 21:44:27 -0400

</div>

</div>

You will note that the unicode display name is automatically encoded as <span class="pre">`utf-8`</span> when the message is serialized, but that when the header is accessed directly, you get the unicode version. This eliminates any need to deal with the <a href="../library/email.header.html#module-email.header" class="reference internal" title="email.header: Representing non-ASCII headers"><span class="pre"><code class="sourceCode python">email.header</code></span></a> <a href="../library/email.header.html#email.header.decode_header" class="reference internal" title="email.header.decode_header"><span class="pre"><code class="sourceCode python">decode_header()</code></span></a> or <a href="../library/email.header.html#email.header.make_header" class="reference internal" title="email.header.make_header"><span class="pre"><code class="sourceCode python">make_header()</code></span></a> functions.

You can also create addresses from parts:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> m['cc'] = [Group('pals', [Address('Bob', 'bob', 'example.com'),
    ...                           Address('Sally', 'sally', 'example.com')]),
    ...            Address('Bonzo', addr_spec='bonz@laugh.com')]
    >>> print(m)
    To: =?utf-8?q?=C3=89ric?= <foo@example.com>
    Date: Fri, 25 May 2012 21:44:27 -0400
    cc: pals: Bob <bob@example.com>, Sally <sally@example.com>;, Bonzo <bonz@laugh.com>

</div>

</div>

Decoding to unicode is done automatically:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> m2 = message_from_string(str(m))
    >>> m2['to']
    'Éric <foo@example.com>'

</div>

</div>

When you parse a message, you can use the <span class="pre">`addresses`</span> and <span class="pre">`groups`</span> attributes of the header objects to access the groups and individual addresses:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> m2['cc'].addresses
    (Address(display_name='Bob', username='bob', domain='example.com'), Address(display_name='Sally', username='sally', domain='example.com'), Address(display_name='Bonzo', username='bonz', domain='laugh.com'))
    >>> m2['cc'].groups
    (Group(display_name='pals', addresses=(Address(display_name='Bob', username='bob', domain='example.com'), Address(display_name='Sally', username='sally', domain='example.com')), Group(display_name=None, addresses=(Address(display_name='Bonzo', username='bonz', domain='laugh.com'),))

</div>

</div>

In summary, if you use one of the new policies, header manipulation works the way it ought to: your application works with unicode strings, and the email package transparently encodes and decodes the unicode to and from the RFC standard Content Transfer Encodings.

</div>

<div id="other-api-changes" class="section">

#### Other API Changes<a href="#other-api-changes" class="headerlink" title="Link to this heading">¶</a>

New <a href="../library/email.parser.html#email.parser.BytesHeaderParser" class="reference internal" title="email.parser.BytesHeaderParser"><span class="pre"><code class="sourceCode python">BytesHeaderParser</code></span></a>, added to the <a href="../library/email.parser.html#module-email.parser" class="reference internal" title="email.parser: Parse flat text email messages to produce a message object structure."><span class="pre"><code class="sourceCode python">parser</code></span></a> module to complement <a href="../library/email.parser.html#email.parser.HeaderParser" class="reference internal" title="email.parser.HeaderParser"><span class="pre"><code class="sourceCode python">HeaderParser</code></span></a> and complete the Bytes API.

New utility functions:

- <a href="../library/email.utils.html#email.utils.format_datetime" class="reference internal" title="email.utils.format_datetime"><span class="pre"><code class="sourceCode python">format_datetime()</code></span></a>: given a <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a>, produce a string formatted for use in an email header.

- <a href="../library/email.utils.html#email.utils.parsedate_to_datetime" class="reference internal" title="email.utils.parsedate_to_datetime"><span class="pre"><code class="sourceCode python">parsedate_to_datetime()</code></span></a>: given a date string from an email header, convert it into an aware <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a>, or a naive <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> if the offset is <span class="pre">`-0000`</span>.

- <a href="../library/email.utils.html#email.utils.localtime" class="reference internal" title="email.utils.localtime"><span class="pre"><code class="sourceCode python">localtime()</code></span></a>: With no argument, returns the current local time as an aware <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> using the local <a href="../library/datetime.html#datetime.timezone" class="reference internal" title="datetime.timezone"><span class="pre"><code class="sourceCode python">timezone</code></span></a>. Given an aware <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a>, converts it into an aware <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> using the local <a href="../library/datetime.html#datetime.timezone" class="reference internal" title="datetime.timezone"><span class="pre"><code class="sourceCode python">timezone</code></span></a>.

</div>

</div>

<div id="ftplib" class="section">

### ftplib<a href="#ftplib" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/ftplib.html#ftplib.FTP" class="reference internal" title="ftplib.FTP"><span class="pre"><code class="sourceCode python">ftplib.FTP</code></span></a> now accepts a <span class="pre">`source_address`</span> keyword argument to specify the <span class="pre">`(host,`</span>` `<span class="pre">`port)`</span> to use as the source address in the bind call when creating the outgoing socket. (Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8594" class="reference external">bpo-8594</a>.)

- The <a href="../library/ftplib.html#ftplib.FTP_TLS" class="reference internal" title="ftplib.FTP_TLS"><span class="pre"><code class="sourceCode python">FTP_TLS</code></span></a> class now provides a new <a href="../library/ftplib.html#ftplib.FTP_TLS.ccc" class="reference internal" title="ftplib.FTP_TLS.ccc"><span class="pre"><code class="sourceCode python">ccc()</code></span></a> function to revert control channel back to plaintext. This can be useful to take advantage of firewalls that know how to handle NAT with non-secure FTP without opening fixed ports. (Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12139" class="reference external">bpo-12139</a>.)

- Added <a href="../library/ftplib.html#ftplib.FTP.mlsd" class="reference internal" title="ftplib.FTP.mlsd"><span class="pre"><code class="sourceCode python">ftplib.FTP.mlsd()</code></span></a> method which provides a parsable directory listing format and deprecates <a href="../library/ftplib.html#ftplib.FTP.nlst" class="reference internal" title="ftplib.FTP.nlst"><span class="pre"><code class="sourceCode python">ftplib.FTP.nlst()</code></span></a> and <a href="../library/ftplib.html#ftplib.FTP.dir" class="reference internal" title="ftplib.FTP.dir"><span class="pre"><code class="sourceCode python">ftplib.FTP.<span class="bu">dir</span>()</code></span></a>. (Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11072" class="reference external">bpo-11072</a>.)

</div>

<div id="functools" class="section">

### functools<a href="#functools" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/functools.html#functools.lru_cache" class="reference internal" title="functools.lru_cache"><span class="pre"><code class="sourceCode python">functools.lru_cache()</code></span></a> decorator now accepts a <span class="pre">`typed`</span> keyword argument (that defaults to <span class="pre">`False`</span> to ensure that it caches values of different types that compare equal in separate cache slots. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13227" class="reference external">bpo-13227</a>.)

</div>

<div id="gc" class="section">

### gc<a href="#gc" class="headerlink" title="Link to this heading">¶</a>

It is now possible to register callbacks invoked by the garbage collector before and after collection using the new <a href="../library/gc.html#gc.callbacks" class="reference internal" title="gc.callbacks"><span class="pre"><code class="sourceCode python">callbacks</code></span></a> list.

</div>

<div id="hmac" class="section">

### hmac<a href="#hmac" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/hmac.html#hmac.compare_digest" class="reference internal" title="hmac.compare_digest"><span class="pre"><code class="sourceCode python">compare_digest()</code></span></a> function has been added to prevent side channel attacks on digests through timing analysis. (Contributed by Nick Coghlan and Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15061" class="reference external">bpo-15061</a>.)

</div>

<div id="http" class="section">

### http<a href="#http" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/http.server.html#http.server.BaseHTTPRequestHandler" class="reference internal" title="http.server.BaseHTTPRequestHandler"><span class="pre"><code class="sourceCode python">http.server.BaseHTTPRequestHandler</code></span></a> now buffers the headers and writes them all at once when <a href="../library/http.server.html#http.server.BaseHTTPRequestHandler.end_headers" class="reference internal" title="http.server.BaseHTTPRequestHandler.end_headers"><span class="pre"><code class="sourceCode python">end_headers()</code></span></a> is called. A new method <a href="../library/http.server.html#http.server.BaseHTTPRequestHandler.flush_headers" class="reference internal" title="http.server.BaseHTTPRequestHandler.flush_headers"><span class="pre"><code class="sourceCode python">flush_headers()</code></span></a> can be used to directly manage when the accumulated headers are sent. (Contributed by Andrew Schaaf in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3709" class="reference external">bpo-3709</a>.)

<a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a> now produces valid <span class="pre">`HTML`</span>` `<span class="pre">`4.01`</span>` `<span class="pre">`strict`</span> output. (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13295" class="reference external">bpo-13295</a>.)

<a href="../library/http.client.html#http.client.HTTPResponse" class="reference internal" title="http.client.HTTPResponse"><span class="pre"><code class="sourceCode python">http.client.HTTPResponse</code></span></a> now has a <a href="../library/http.client.html#http.client.HTTPResponse.readinto" class="reference internal" title="http.client.HTTPResponse.readinto"><span class="pre"><code class="sourceCode python">readinto()</code></span></a> method, which means it can be used as an <a href="../library/io.html#io.RawIOBase" class="reference internal" title="io.RawIOBase"><span class="pre"><code class="sourceCode python">io.RawIOBase</code></span></a> class. (Contributed by John Kuhn in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13464" class="reference external">bpo-13464</a>.)

</div>

<div id="html" class="section">

### html<a href="#html" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">html.parser.HTMLParser</code></span></a> is now able to parse broken markup without raising errors, therefore the *strict* argument of the constructor and the <span class="pre">`HTMLParseError`</span> exception are now deprecated. The ability to parse broken markup is the result of a number of bug fixes that are also available on the latest bug fix releases of Python 2.7/3.2. (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15114" class="reference external">bpo-15114</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14538" class="reference external">bpo-14538</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13993" class="reference external">bpo-13993</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13960" class="reference external">bpo-13960</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13358" class="reference external">bpo-13358</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1745761" class="reference external">bpo-1745761</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=755670" class="reference external">bpo-755670</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13357" class="reference external">bpo-13357</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12629" class="reference external">bpo-12629</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1200313" class="reference external">bpo-1200313</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=670664" class="reference external">bpo-670664</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13273" class="reference external">bpo-13273</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12888" class="reference external">bpo-12888</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7311" class="reference external">bpo-7311</a>.)

A new <a href="../library/html.entities.html#html.entities.html5" class="reference internal" title="html.entities.html5"><span class="pre"><code class="sourceCode python">html5</code></span></a> dictionary that maps HTML5 named character references to the equivalent Unicode character(s) (e.g. <span class="pre">`html5['gt;']`</span>` `<span class="pre">`==`</span>` `<span class="pre">`'>'`</span>) has been added to the <a href="../library/html.entities.html#module-html.entities" class="reference internal" title="html.entities: Definitions of HTML general entities."><span class="pre"><code class="sourceCode python">html.entities</code></span></a> module. The dictionary is now also used by <a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">HTMLParser</code></span></a>. (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11113" class="reference external">bpo-11113</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15156" class="reference external">bpo-15156</a>.)

</div>

<div id="imaplib" class="section">

### imaplib<a href="#imaplib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/imaplib.html#imaplib.IMAP4_SSL" class="reference internal" title="imaplib.IMAP4_SSL"><span class="pre"><code class="sourceCode python">IMAP4_SSL</code></span></a> constructor now accepts an SSLContext parameter to control parameters of the secure channel.

(Contributed by Sijin Joseph in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8808" class="reference external">bpo-8808</a>.)

</div>

<div id="inspect" class="section">

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/inspect.html#inspect.getclosurevars" class="reference internal" title="inspect.getclosurevars"><span class="pre"><code class="sourceCode python">getclosurevars()</code></span></a> function has been added. This function reports the current binding of all names referenced from the function body and where those names were resolved, making it easier to verify correct internal state when testing code that relies on stateful closures.

(Contributed by Meador Inge and Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13062" class="reference external">bpo-13062</a>.)

A new <a href="../library/inspect.html#inspect.getgeneratorlocals" class="reference internal" title="inspect.getgeneratorlocals"><span class="pre"><code class="sourceCode python">getgeneratorlocals()</code></span></a> function has been added. This function reports the current binding of local variables in the generator’s stack frame, making it easier to verify correct internal state when testing generators.

(Contributed by Meador Inge in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15153" class="reference external">bpo-15153</a>.)

</div>

<div id="io" class="section">

### io<a href="#io" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/io.html#io.open" class="reference internal" title="io.open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> function has a new <span class="pre">`'x'`</span> mode that can be used to exclusively create a new file, and raise a <a href="../library/exceptions.html#FileExistsError" class="reference internal" title="FileExistsError"><span class="pre"><code class="sourceCode python"><span class="pp">FileExistsError</span></code></span></a> if the file already exists. It is based on the C11 ‘x’ mode to fopen().

(Contributed by David Townshend in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12760" class="reference external">bpo-12760</a>.)

The constructor of the <a href="../library/io.html#io.TextIOWrapper" class="reference internal" title="io.TextIOWrapper"><span class="pre"><code class="sourceCode python">TextIOWrapper</code></span></a> class has a new *write_through* optional argument. If *write_through* is <span class="pre">`True`</span>, calls to <span class="pre">`write()`</span> are guaranteed not to be buffered: any data written on the <a href="../library/io.html#io.TextIOWrapper" class="reference internal" title="io.TextIOWrapper"><span class="pre"><code class="sourceCode python">TextIOWrapper</code></span></a> object is immediately handled to its underlying binary buffer.

</div>

<div id="itertools" class="section">

### itertools<a href="#itertools" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/itertools.html#itertools.accumulate" class="reference internal" title="itertools.accumulate"><span class="pre"><code class="sourceCode python">accumulate()</code></span></a> now takes an optional <span class="pre">`func`</span> argument for providing a user-supplied binary function.

</div>

<div id="logging" class="section">

### logging<a href="#logging" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/logging.html#logging.basicConfig" class="reference internal" title="logging.basicConfig"><span class="pre"><code class="sourceCode python">basicConfig()</code></span></a> function now supports an optional <span class="pre">`handlers`</span> argument taking an iterable of handlers to be added to the root logger.

A class level attribute <span class="pre">`append_nul`</span> has been added to <a href="../library/logging.handlers.html#logging.handlers.SysLogHandler" class="reference internal" title="logging.handlers.SysLogHandler"><span class="pre"><code class="sourceCode python">SysLogHandler</code></span></a> to allow control of the appending of the <span class="pre">`NUL`</span> (<span class="pre">`\000`</span>) byte to syslog records, since for some daemons it is required while for others it is passed through to the log.

</div>

<div id="math" class="section">

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> module has a new function, <a href="../library/math.html#math.log2" class="reference internal" title="math.log2"><span class="pre"><code class="sourceCode python">log2()</code></span></a>, which returns the base-2 logarithm of *x*.

(Written by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11888" class="reference external">bpo-11888</a>.)

</div>

<div id="mmap" class="section">

### mmap<a href="#mmap" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/mmap.html#mmap.mmap.read" class="reference internal" title="mmap.mmap.read"><span class="pre"><code class="sourceCode python">read()</code></span></a> method is now more compatible with other file-like objects: if the argument is omitted or specified as <span class="pre">`None`</span>, it returns the bytes from the current file position to the end of the mapping. (Contributed by Petri Lehtinen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12021" class="reference external">bpo-12021</a>.)

</div>

<div id="multiprocessing" class="section">

### multiprocessing<a href="#multiprocessing" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/multiprocessing.html#multiprocessing.connection.wait" class="reference internal" title="multiprocessing.connection.wait"><span class="pre"><code class="sourceCode python">multiprocessing.connection.wait()</code></span></a> function allows polling multiple objects (such as connections, sockets and pipes) with a timeout. (Contributed by Richard Oudkerk in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12328" class="reference external">bpo-12328</a>.)

<a href="../library/multiprocessing.html#multiprocessing.connection.Connection" class="reference internal" title="multiprocessing.connection.Connection"><span class="pre"><code class="sourceCode python">multiprocessing.connection.Connection</code></span></a> objects can now be transferred over multiprocessing connections. (Contributed by Richard Oudkerk in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4892" class="reference external">bpo-4892</a>.)

<a href="../library/multiprocessing.html#multiprocessing.Process" class="reference internal" title="multiprocessing.Process"><span class="pre"><code class="sourceCode python">multiprocessing.Process</code></span></a> now accepts a <span class="pre">`daemon`</span> keyword argument to override the default behavior of inheriting the <span class="pre">`daemon`</span> flag from the parent process (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6064" class="reference external">bpo-6064</a>).

New attribute <a href="../library/multiprocessing.html#multiprocessing.Process.sentinel" class="reference internal" title="multiprocessing.Process.sentinel"><span class="pre"><code class="sourceCode python">multiprocessing.Process.sentinel</code></span></a> allows a program to wait on multiple <a href="../library/multiprocessing.html#multiprocessing.Process" class="reference internal" title="multiprocessing.Process"><span class="pre"><code class="sourceCode python">Process</code></span></a> objects at one time using the appropriate OS primitives (for example, <a href="../library/select.html#module-select" class="reference internal" title="select: Wait for I/O completion on multiple streams."><span class="pre"><code class="sourceCode python">select</code></span></a> on posix systems).

New methods <a href="../library/multiprocessing.html#multiprocessing.pool.Pool.starmap" class="reference internal" title="multiprocessing.pool.Pool.starmap"><span class="pre"><code class="sourceCode python">multiprocessing.pool.Pool.starmap()</code></span></a> and <a href="../library/multiprocessing.html#multiprocessing.pool.Pool.starmap_async" class="reference internal" title="multiprocessing.pool.Pool.starmap_async"><span class="pre"><code class="sourceCode python">starmap_async()</code></span></a> provide <a href="../library/itertools.html#itertools.starmap" class="reference internal" title="itertools.starmap"><span class="pre"><code class="sourceCode python">itertools.starmap()</code></span></a> equivalents to the existing <a href="../library/multiprocessing.html#multiprocessing.pool.Pool.map" class="reference internal" title="multiprocessing.pool.Pool.map"><span class="pre"><code class="sourceCode python">multiprocessing.pool.Pool.<span class="bu">map</span>()</code></span></a> and <a href="../library/multiprocessing.html#multiprocessing.pool.Pool.map_async" class="reference internal" title="multiprocessing.pool.Pool.map_async"><span class="pre"><code class="sourceCode python">map_async()</code></span></a> functions. (Contributed by Hynek Schlawack in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12708" class="reference external">bpo-12708</a>.)

</div>

<div id="nntplib" class="section">

### nntplib<a href="#nntplib" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`nntplib.NNTP`</span> class now supports the context management protocol to unconditionally consume <a href="../library/socket.html#socket.error" class="reference internal" title="socket.error"><span class="pre"><code class="sourceCode python">socket.error</code></span></a> exceptions and to close the NNTP connection when done:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from nntplib import NNTP
    >>> with NNTP('news.gmane.org') as n:
    ...     n.group('gmane.comp.python.committers')
    ...
    ('211 1755 1 1755 gmane.comp.python.committers', 1755, 1, 1755, 'gmane.comp.python.committers')
    >>>

</div>

</div>

(Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9795" class="reference external">bpo-9795</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module has a new <a href="../library/os.html#os.pipe2" class="reference internal" title="os.pipe2"><span class="pre"><code class="sourceCode python">pipe2()</code></span></a> function that makes it possible to create a pipe with <a href="../library/os.html#os.O_CLOEXEC" class="reference internal" title="os.O_CLOEXEC"><span class="pre"><code class="sourceCode python">O_CLOEXEC</code></span></a> or <a href="../library/os.html#os.O_NONBLOCK" class="reference internal" title="os.O_NONBLOCK"><span class="pre"><code class="sourceCode python">O_NONBLOCK</code></span></a> flags set atomically. This is especially useful to avoid race conditions in multi-threaded programs.

- The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module has a new <a href="../library/os.html#os.sendfile" class="reference internal" title="os.sendfile"><span class="pre"><code class="sourceCode python">sendfile()</code></span></a> function which provides an efficient “zero-copy” way for copying data from one file (or socket) descriptor to another. The phrase “zero-copy” refers to the fact that all of the copying of data between the two descriptors is done entirely by the kernel, with no copying of data into userspace buffers. <a href="../library/os.html#os.sendfile" class="reference internal" title="os.sendfile"><span class="pre"><code class="sourceCode python">sendfile()</code></span></a> can be used to efficiently copy data from a file on disk to a network socket, e.g. for downloading a file.

  (Patch submitted by Ross Lagerwall and Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10882" class="reference external">bpo-10882</a>.)

- To avoid race conditions like symlink attacks and issues with temporary files and directories, it is more reliable (and also faster) to manipulate file descriptors instead of file names. Python 3.3 enhances existing functions and introduces new functions to work on file descriptors (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4761" class="reference external">bpo-4761</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10755" class="reference external">bpo-10755</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14626" class="reference external">bpo-14626</a>).

  - The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module has a new <a href="../library/os.html#os.fwalk" class="reference internal" title="os.fwalk"><span class="pre"><code class="sourceCode python">fwalk()</code></span></a> function similar to <a href="../library/os.html#os.walk" class="reference internal" title="os.walk"><span class="pre"><code class="sourceCode python">walk()</code></span></a> except that it also yields file descriptors referring to the directories visited. This is especially useful to avoid symlink races.

  - The following functions get new optional *dir_fd* (<a href="../library/os.html#dir-fd" class="reference internal"><span class="std std-ref">paths relative to directory descriptors</span></a>) and/or *follow_symlinks* (<a href="../library/os.html#follow-symlinks" class="reference internal"><span class="std std-ref">not following symlinks</span></a>): <a href="../library/os.html#os.access" class="reference internal" title="os.access"><span class="pre"><code class="sourceCode python">access()</code></span></a>, <a href="../library/os.html#os.chflags" class="reference internal" title="os.chflags"><span class="pre"><code class="sourceCode python">chflags()</code></span></a>, <a href="../library/os.html#os.chmod" class="reference internal" title="os.chmod"><span class="pre"><code class="sourceCode python">chmod()</code></span></a>, <a href="../library/os.html#os.chown" class="reference internal" title="os.chown"><span class="pre"><code class="sourceCode python">chown()</code></span></a>, <a href="../library/os.html#os.link" class="reference internal" title="os.link"><span class="pre"><code class="sourceCode python">link()</code></span></a>, <a href="../library/os.html#os.lstat" class="reference internal" title="os.lstat"><span class="pre"><code class="sourceCode python">lstat()</code></span></a>, <a href="../library/os.html#os.mkdir" class="reference internal" title="os.mkdir"><span class="pre"><code class="sourceCode python">mkdir()</code></span></a>, <a href="../library/os.html#os.mkfifo" class="reference internal" title="os.mkfifo"><span class="pre"><code class="sourceCode python">mkfifo()</code></span></a>, <a href="../library/os.html#os.mknod" class="reference internal" title="os.mknod"><span class="pre"><code class="sourceCode python">mknod()</code></span></a>, <a href="../library/os.html#os.open" class="reference internal" title="os.open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a>, <a href="../library/os.html#os.readlink" class="reference internal" title="os.readlink"><span class="pre"><code class="sourceCode python">readlink()</code></span></a>, <a href="../library/os.html#os.remove" class="reference internal" title="os.remove"><span class="pre"><code class="sourceCode python">remove()</code></span></a>, <a href="../library/os.html#os.rename" class="reference internal" title="os.rename"><span class="pre"><code class="sourceCode python">rename()</code></span></a>, <a href="../library/os.html#os.replace" class="reference internal" title="os.replace"><span class="pre"><code class="sourceCode python">replace()</code></span></a>, <a href="../library/os.html#os.rmdir" class="reference internal" title="os.rmdir"><span class="pre"><code class="sourceCode python">rmdir()</code></span></a>, <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">stat()</code></span></a>, <a href="../library/os.html#os.symlink" class="reference internal" title="os.symlink"><span class="pre"><code class="sourceCode python">symlink()</code></span></a>, <a href="../library/os.html#os.unlink" class="reference internal" title="os.unlink"><span class="pre"><code class="sourceCode python">unlink()</code></span></a>, <a href="../library/os.html#os.utime" class="reference internal" title="os.utime"><span class="pre"><code class="sourceCode python">utime()</code></span></a>. Platform support for using these parameters can be checked via the sets <a href="../library/os.html#os.supports_dir_fd" class="reference internal" title="os.supports_dir_fd"><span class="pre"><code class="sourceCode python">os.supports_dir_fd</code></span></a> and <a href="../library/os.html#os.supports_follow_symlinks" class="reference internal" title="os.supports_follow_symlinks"><span class="pre"><code class="sourceCode python">os.supports_follow_symlinks</code></span></a>.

  - The following functions now support a file descriptor for their path argument: <a href="../library/os.html#os.chdir" class="reference internal" title="os.chdir"><span class="pre"><code class="sourceCode python">chdir()</code></span></a>, <a href="../library/os.html#os.chmod" class="reference internal" title="os.chmod"><span class="pre"><code class="sourceCode python">chmod()</code></span></a>, <a href="../library/os.html#os.chown" class="reference internal" title="os.chown"><span class="pre"><code class="sourceCode python">chown()</code></span></a>, <a href="../library/os.html#os.execve" class="reference internal" title="os.execve"><span class="pre"><code class="sourceCode python">execve()</code></span></a>, <a href="../library/os.html#os.listdir" class="reference internal" title="os.listdir"><span class="pre"><code class="sourceCode python">listdir()</code></span></a>, <a href="../library/os.html#os.pathconf" class="reference internal" title="os.pathconf"><span class="pre"><code class="sourceCode python">pathconf()</code></span></a>, <a href="../library/os.path.html#os.path.exists" class="reference internal" title="os.path.exists"><span class="pre"><code class="sourceCode python">exists()</code></span></a>, <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">stat()</code></span></a>, <a href="../library/os.html#os.statvfs" class="reference internal" title="os.statvfs"><span class="pre"><code class="sourceCode python">statvfs()</code></span></a>, <a href="../library/os.html#os.utime" class="reference internal" title="os.utime"><span class="pre"><code class="sourceCode python">utime()</code></span></a>. Platform support for this can be checked via the <a href="../library/os.html#os.supports_fd" class="reference internal" title="os.supports_fd"><span class="pre"><code class="sourceCode python">os.supports_fd</code></span></a> set.

- <a href="../library/os.html#os.access" class="reference internal" title="os.access"><span class="pre"><code class="sourceCode python">access()</code></span></a> accepts an <span class="pre">`effective_ids`</span> keyword argument to turn on using the effective uid/gid rather than the real uid/gid in the access check. Platform support for this can be checked via the <a href="../library/os.html#os.supports_effective_ids" class="reference internal" title="os.supports_effective_ids"><span class="pre"><code class="sourceCode python">supports_effective_ids</code></span></a> set.

- The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module has two new functions: <a href="../library/os.html#os.getpriority" class="reference internal" title="os.getpriority"><span class="pre"><code class="sourceCode python">getpriority()</code></span></a> and <a href="../library/os.html#os.setpriority" class="reference internal" title="os.setpriority"><span class="pre"><code class="sourceCode python">setpriority()</code></span></a>. They can be used to get or set process niceness/priority in a fashion similar to <a href="../library/os.html#os.nice" class="reference internal" title="os.nice"><span class="pre"><code class="sourceCode python">os.nice()</code></span></a> but extended to all processes instead of just the current one.

  (Patch submitted by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10784" class="reference external">bpo-10784</a>.)

- The new <a href="../library/os.html#os.replace" class="reference internal" title="os.replace"><span class="pre"><code class="sourceCode python">os.replace()</code></span></a> function allows cross-platform renaming of a file with overwriting the destination. With <a href="../library/os.html#os.rename" class="reference internal" title="os.rename"><span class="pre"><code class="sourceCode python">os.rename()</code></span></a>, an existing destination file is overwritten under POSIX, but raises an error under Windows. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8828" class="reference external">bpo-8828</a>.)

- The stat family of functions (<a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">stat()</code></span></a>, <a href="../library/os.html#os.fstat" class="reference internal" title="os.fstat"><span class="pre"><code class="sourceCode python">fstat()</code></span></a>, and <a href="../library/os.html#os.lstat" class="reference internal" title="os.lstat"><span class="pre"><code class="sourceCode python">lstat()</code></span></a>) now support reading a file’s timestamps with nanosecond precision. Symmetrically, <a href="../library/os.html#os.utime" class="reference internal" title="os.utime"><span class="pre"><code class="sourceCode python">utime()</code></span></a> can now write file timestamps with nanosecond precision. (Contributed by Larry Hastings in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14127" class="reference external">bpo-14127</a>.)

- The new <a href="../library/os.html#os.get_terminal_size" class="reference internal" title="os.get_terminal_size"><span class="pre"><code class="sourceCode python">os.get_terminal_size()</code></span></a> function queries the size of the terminal attached to a file descriptor. See also <a href="../library/shutil.html#shutil.get_terminal_size" class="reference internal" title="shutil.get_terminal_size"><span class="pre"><code class="sourceCode python">shutil.get_terminal_size()</code></span></a>. (Contributed by Zbigniew Jędrzejewski-Szmek in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13609" class="reference external">bpo-13609</a>.)

<!-- -->

- New functions to support Linux extended attributes (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12720" class="reference external">bpo-12720</a>): <a href="../library/os.html#os.getxattr" class="reference internal" title="os.getxattr"><span class="pre"><code class="sourceCode python">getxattr()</code></span></a>, <a href="../library/os.html#os.listxattr" class="reference internal" title="os.listxattr"><span class="pre"><code class="sourceCode python">listxattr()</code></span></a>, <a href="../library/os.html#os.removexattr" class="reference internal" title="os.removexattr"><span class="pre"><code class="sourceCode python">removexattr()</code></span></a>, <a href="../library/os.html#os.setxattr" class="reference internal" title="os.setxattr"><span class="pre"><code class="sourceCode python">setxattr()</code></span></a>.

- New interface to the scheduler. These functions control how a process is allocated CPU time by the operating system. New functions: <a href="../library/os.html#os.sched_get_priority_max" class="reference internal" title="os.sched_get_priority_max"><span class="pre"><code class="sourceCode python">sched_get_priority_max()</code></span></a>, <a href="../library/os.html#os.sched_get_priority_min" class="reference internal" title="os.sched_get_priority_min"><span class="pre"><code class="sourceCode python">sched_get_priority_min()</code></span></a>, <a href="../library/os.html#os.sched_getaffinity" class="reference internal" title="os.sched_getaffinity"><span class="pre"><code class="sourceCode python">sched_getaffinity()</code></span></a>, <a href="../library/os.html#os.sched_getparam" class="reference internal" title="os.sched_getparam"><span class="pre"><code class="sourceCode python">sched_getparam()</code></span></a>, <a href="../library/os.html#os.sched_getscheduler" class="reference internal" title="os.sched_getscheduler"><span class="pre"><code class="sourceCode python">sched_getscheduler()</code></span></a>, <a href="../library/os.html#os.sched_rr_get_interval" class="reference internal" title="os.sched_rr_get_interval"><span class="pre"><code class="sourceCode python">sched_rr_get_interval()</code></span></a>, <a href="../library/os.html#os.sched_setaffinity" class="reference internal" title="os.sched_setaffinity"><span class="pre"><code class="sourceCode python">sched_setaffinity()</code></span></a>, <a href="../library/os.html#os.sched_setparam" class="reference internal" title="os.sched_setparam"><span class="pre"><code class="sourceCode python">sched_setparam()</code></span></a>, <a href="../library/os.html#os.sched_setscheduler" class="reference internal" title="os.sched_setscheduler"><span class="pre"><code class="sourceCode python">sched_setscheduler()</code></span></a>, <a href="../library/os.html#os.sched_yield" class="reference internal" title="os.sched_yield"><span class="pre"><code class="sourceCode python">sched_yield()</code></span></a>,

- New functions to control the file system:

  - <a href="../library/os.html#os.posix_fadvise" class="reference internal" title="os.posix_fadvise"><span class="pre"><code class="sourceCode python">posix_fadvise()</code></span></a>: Announces an intention to access data in a specific pattern thus allowing the kernel to make optimizations.

  - <a href="../library/os.html#os.posix_fallocate" class="reference internal" title="os.posix_fallocate"><span class="pre"><code class="sourceCode python">posix_fallocate()</code></span></a>: Ensures that enough disk space is allocated for a file.

  - <a href="../library/os.html#os.sync" class="reference internal" title="os.sync"><span class="pre"><code class="sourceCode python">sync()</code></span></a>: Force write of everything to disk.

- Additional new posix functions:

  - <a href="../library/os.html#os.lockf" class="reference internal" title="os.lockf"><span class="pre"><code class="sourceCode python">lockf()</code></span></a>: Apply, test or remove a POSIX lock on an open file descriptor.

  - <a href="../library/os.html#os.pread" class="reference internal" title="os.pread"><span class="pre"><code class="sourceCode python">pread()</code></span></a>: Read from a file descriptor at an offset, the file offset remains unchanged.

  - <a href="../library/os.html#os.pwrite" class="reference internal" title="os.pwrite"><span class="pre"><code class="sourceCode python">pwrite()</code></span></a>: Write to a file descriptor from an offset, leaving the file offset unchanged.

  - <a href="../library/os.html#os.readv" class="reference internal" title="os.readv"><span class="pre"><code class="sourceCode python">readv()</code></span></a>: Read from a file descriptor into a number of writable buffers.

  - <a href="../library/os.html#os.truncate" class="reference internal" title="os.truncate"><span class="pre"><code class="sourceCode python">truncate()</code></span></a>: Truncate the file corresponding to *path*, so that it is at most *length* bytes in size.

  - <a href="../library/os.html#os.waitid" class="reference internal" title="os.waitid"><span class="pre"><code class="sourceCode python">waitid()</code></span></a>: Wait for the completion of one or more child processes.

  - <a href="../library/os.html#os.writev" class="reference internal" title="os.writev"><span class="pre"><code class="sourceCode python">writev()</code></span></a>: Write the contents of *buffers* to a file descriptor, where *buffers* is an arbitrary sequence of buffers.

  - <a href="../library/os.html#os.getgrouplist" class="reference internal" title="os.getgrouplist"><span class="pre"><code class="sourceCode python">getgrouplist()</code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9344" class="reference external">bpo-9344</a>): Return list of group ids that specified user belongs to.

- <a href="../library/os.html#os.times" class="reference internal" title="os.times"><span class="pre"><code class="sourceCode python">times()</code></span></a> and <a href="../library/os.html#os.uname" class="reference internal" title="os.uname"><span class="pre"><code class="sourceCode python">uname()</code></span></a>: Return type changed from a tuple to a tuple-like object with named attributes.

- Some platforms now support additional constants for the <a href="../library/os.html#os.lseek" class="reference internal" title="os.lseek"><span class="pre"><code class="sourceCode python">lseek()</code></span></a> function, such as <span class="pre">`os.SEEK_HOLE`</span> and <span class="pre">`os.SEEK_DATA`</span>.

- New constants <a href="../library/os.html#os.RTLD_LAZY" class="reference internal" title="os.RTLD_LAZY"><span class="pre"><code class="sourceCode python">RTLD_LAZY</code></span></a>, <a href="../library/os.html#os.RTLD_NOW" class="reference internal" title="os.RTLD_NOW"><span class="pre"><code class="sourceCode python">RTLD_NOW</code></span></a>, <a href="../library/os.html#os.RTLD_GLOBAL" class="reference internal" title="os.RTLD_GLOBAL"><span class="pre"><code class="sourceCode python">RTLD_GLOBAL</code></span></a>, <a href="../library/os.html#os.RTLD_LOCAL" class="reference internal" title="os.RTLD_LOCAL"><span class="pre"><code class="sourceCode python">RTLD_LOCAL</code></span></a>, <a href="../library/os.html#os.RTLD_NODELETE" class="reference internal" title="os.RTLD_NODELETE"><span class="pre"><code class="sourceCode python">RTLD_NODELETE</code></span></a>, <a href="../library/os.html#os.RTLD_NOLOAD" class="reference internal" title="os.RTLD_NOLOAD"><span class="pre"><code class="sourceCode python">RTLD_NOLOAD</code></span></a>, and <a href="../library/os.html#os.RTLD_DEEPBIND" class="reference internal" title="os.RTLD_DEEPBIND"><span class="pre"><code class="sourceCode python">RTLD_DEEPBIND</code></span></a> are available on platforms that support them. These are for use with the <a href="../library/sys.html#sys.setdlopenflags" class="reference internal" title="sys.setdlopenflags"><span class="pre"><code class="sourceCode python">sys.setdlopenflags()</code></span></a> function, and supersede the similar constants defined in <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> and <span class="pre">`DLFCN`</span>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13226" class="reference external">bpo-13226</a>.)

- <a href="../library/os.html#os.symlink" class="reference internal" title="os.symlink"><span class="pre"><code class="sourceCode python">os.symlink()</code></span></a> now accepts (and ignores) the <span class="pre">`target_is_directory`</span> keyword argument on non-Windows platforms, to ease cross-platform support.

</div>

<div id="pdb" class="section">

### pdb<a href="#pdb" class="headerlink" title="Link to this heading">¶</a>

Tab-completion is now available not only for command names, but also their arguments. For example, for the <span class="pre">`break`</span> command, function and file names are completed.

(Contributed by Georg Brandl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14210" class="reference external">bpo-14210</a>)

</div>

<div id="pickle" class="section">

### pickle<a href="#pickle" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pickle.html#pickle.Pickler" class="reference internal" title="pickle.Pickler"><span class="pre"><code class="sourceCode python">pickle.Pickler</code></span></a> objects now have an optional <a href="../library/pickle.html#pickle.Pickler.dispatch_table" class="reference internal" title="pickle.Pickler.dispatch_table"><span class="pre"><code class="sourceCode python">dispatch_table</code></span></a> attribute allowing per-pickler reduction functions to be set.

(Contributed by Richard Oudkerk in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14166" class="reference external">bpo-14166</a>.)

</div>

<div id="pydoc" class="section">

### pydoc<a href="#pydoc" class="headerlink" title="Link to this heading">¶</a>

The Tk GUI and the <span class="pre">`serve()`</span> function have been removed from the <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module: <span class="pre">`pydoc`</span>` `<span class="pre">`-g`</span> and <span class="pre">`serve()`</span> have been deprecated in Python 3.2.

</div>

<div id="re" class="section">

### re<a href="#re" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> regular expressions now support <span class="pre">`\u`</span> and <span class="pre">`\U`</span> escapes.

(Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3665" class="reference external">bpo-3665</a>.)

</div>

<div id="sched" class="section">

### sched<a href="#sched" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/sched.html#sched.scheduler.run" class="reference internal" title="sched.scheduler.run"><span class="pre"><code class="sourceCode python">run()</code></span></a> now accepts a *blocking* parameter which when set to false makes the method execute the scheduled events due to expire soonest (if any) and then return immediately. This is useful in case you want to use the <a href="../library/sched.html#sched.scheduler" class="reference internal" title="sched.scheduler"><span class="pre"><code class="sourceCode python">scheduler</code></span></a> in non-blocking applications. (Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13449" class="reference external">bpo-13449</a>.)

- <a href="../library/sched.html#sched.scheduler" class="reference internal" title="sched.scheduler"><span class="pre"><code class="sourceCode python">scheduler</code></span></a> class can now be safely used in multi-threaded environments. (Contributed by Josiah Carlson and Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8684" class="reference external">bpo-8684</a>.)

- *timefunc* and *delayfunct* parameters of <a href="../library/sched.html#sched.scheduler" class="reference internal" title="sched.scheduler"><span class="pre"><code class="sourceCode python">scheduler</code></span></a> class constructor are now optional and defaults to <a href="../library/time.html#time.time" class="reference internal" title="time.time"><span class="pre"><code class="sourceCode python">time.time()</code></span></a> and <a href="../library/time.html#time.sleep" class="reference internal" title="time.sleep"><span class="pre"><code class="sourceCode python">time.sleep()</code></span></a> respectively. (Contributed by Chris Clark in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13245" class="reference external">bpo-13245</a>.)

- <a href="../library/sched.html#sched.scheduler.enter" class="reference internal" title="sched.scheduler.enter"><span class="pre"><code class="sourceCode python">enter()</code></span></a> and <a href="../library/sched.html#sched.scheduler.enterabs" class="reference internal" title="sched.scheduler.enterabs"><span class="pre"><code class="sourceCode python">enterabs()</code></span></a> *argument* parameter is now optional. (Contributed by Chris Clark in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13245" class="reference external">bpo-13245</a>.)

- <a href="../library/sched.html#sched.scheduler.enter" class="reference internal" title="sched.scheduler.enter"><span class="pre"><code class="sourceCode python">enter()</code></span></a> and <a href="../library/sched.html#sched.scheduler.enterabs" class="reference internal" title="sched.scheduler.enterabs"><span class="pre"><code class="sourceCode python">enterabs()</code></span></a> now accept a *kwargs* parameter. (Contributed by Chris Clark in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13245" class="reference external">bpo-13245</a>.)

</div>

<div id="select" class="section">

### select<a href="#select" class="headerlink" title="Link to this heading">¶</a>

Solaris and derivative platforms have a new class <a href="../library/select.html#select.devpoll" class="reference internal" title="select.devpoll"><span class="pre"><code class="sourceCode python">select.devpoll</code></span></a> for high performance asynchronous sockets via <span class="pre">`/dev/poll`</span>. (Contributed by Jesús Cea Avión in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6397" class="reference external">bpo-6397</a>.)

</div>

<div id="shlex" class="section">

### shlex<a href="#shlex" class="headerlink" title="Link to this heading">¶</a>

The previously undocumented helper function <span class="pre">`quote`</span> from the <span class="pre">`pipes`</span> modules has been moved to the <a href="../library/shlex.html#module-shlex" class="reference internal" title="shlex: Simple lexical analysis for Unix shell-like languages."><span class="pre"><code class="sourceCode python">shlex</code></span></a> module and documented. <a href="../library/shlex.html#shlex.quote" class="reference internal" title="shlex.quote"><span class="pre"><code class="sourceCode python">quote()</code></span></a> properly escapes all characters in a string that might be otherwise given special meaning by the shell.

</div>

<div id="shutil" class="section">

### shutil<a href="#shutil" class="headerlink" title="Link to this heading">¶</a>

- New functions:

  - <a href="../library/shutil.html#shutil.disk_usage" class="reference internal" title="shutil.disk_usage"><span class="pre"><code class="sourceCode python">disk_usage()</code></span></a>: provides total, used and free disk space statistics. (Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12442" class="reference external">bpo-12442</a>.)

  - <a href="../library/shutil.html#shutil.chown" class="reference internal" title="shutil.chown"><span class="pre"><code class="sourceCode python">chown()</code></span></a>: allows one to change user and/or group of the given path also specifying the user/group names and not only their numeric ids. (Contributed by Sandro Tosi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12191" class="reference external">bpo-12191</a>.)

  - <a href="../library/shutil.html#shutil.get_terminal_size" class="reference internal" title="shutil.get_terminal_size"><span class="pre"><code class="sourceCode python">shutil.get_terminal_size()</code></span></a>: returns the size of the terminal window to which the interpreter is attached. (Contributed by Zbigniew Jędrzejewski-Szmek in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13609" class="reference external">bpo-13609</a>.)

- <a href="../library/shutil.html#shutil.copy2" class="reference internal" title="shutil.copy2"><span class="pre"><code class="sourceCode python">copy2()</code></span></a> and <a href="../library/shutil.html#shutil.copystat" class="reference internal" title="shutil.copystat"><span class="pre"><code class="sourceCode python">copystat()</code></span></a> now preserve file timestamps with nanosecond precision on platforms that support it. They also preserve file “extended attributes” on Linux. (Contributed by Larry Hastings in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14127" class="reference external">bpo-14127</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15238" class="reference external">bpo-15238</a>.)

- Several functions now take an optional <span class="pre">`symlinks`</span> argument: when that parameter is true, symlinks aren’t dereferenced and the operation instead acts on the symlink itself (or creates one, if relevant). (Contributed by Hynek Schlawack in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12715" class="reference external">bpo-12715</a>.)

- When copying files to a different file system, <a href="../library/shutil.html#shutil.move" class="reference internal" title="shutil.move"><span class="pre"><code class="sourceCode python">move()</code></span></a> now handles symlinks the way the posix <span class="pre">`mv`</span> command does, recreating the symlink rather than copying the target file contents. (Contributed by Jonathan Niehof in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9993" class="reference external">bpo-9993</a>.) <a href="../library/shutil.html#shutil.move" class="reference internal" title="shutil.move"><span class="pre"><code class="sourceCode python">move()</code></span></a> now also returns the <span class="pre">`dst`</span> argument as its result.

- <a href="../library/shutil.html#shutil.rmtree" class="reference internal" title="shutil.rmtree"><span class="pre"><code class="sourceCode python">rmtree()</code></span></a> is now resistant to symlink attacks on platforms which support the new <span class="pre">`dir_fd`</span> parameter in <a href="../library/os.html#os.open" class="reference internal" title="os.open"><span class="pre"><code class="sourceCode python">os.<span class="bu">open</span>()</code></span></a> and <a href="../library/os.html#os.unlink" class="reference internal" title="os.unlink"><span class="pre"><code class="sourceCode python">os.unlink()</code></span></a>. (Contributed by Martin von Löwis and Hynek Schlawack in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4489" class="reference external">bpo-4489</a>.)

</div>

<div id="signal" class="section">

### signal<a href="#signal" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/signal.html#module-signal" class="reference internal" title="signal: Set handlers for asynchronous events."><span class="pre"><code class="sourceCode python">signal</code></span></a> module has new functions:

  - <a href="../library/signal.html#signal.pthread_sigmask" class="reference internal" title="signal.pthread_sigmask"><span class="pre"><code class="sourceCode python">pthread_sigmask()</code></span></a>: fetch and/or change the signal mask of the calling thread (Contributed by Jean-Paul Calderone in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8407" class="reference external">bpo-8407</a>);

  - <a href="../library/signal.html#signal.pthread_kill" class="reference internal" title="signal.pthread_kill"><span class="pre"><code class="sourceCode python">pthread_kill()</code></span></a>: send a signal to a thread;

  - <a href="../library/signal.html#signal.sigpending" class="reference internal" title="signal.sigpending"><span class="pre"><code class="sourceCode python">sigpending()</code></span></a>: examine pending functions;

  - <a href="../library/signal.html#signal.sigwait" class="reference internal" title="signal.sigwait"><span class="pre"><code class="sourceCode python">sigwait()</code></span></a>: wait a signal;

  - <a href="../library/signal.html#signal.sigwaitinfo" class="reference internal" title="signal.sigwaitinfo"><span class="pre"><code class="sourceCode python">sigwaitinfo()</code></span></a>: wait for a signal, returning detailed information about it;

  - <a href="../library/signal.html#signal.sigtimedwait" class="reference internal" title="signal.sigtimedwait"><span class="pre"><code class="sourceCode python">sigtimedwait()</code></span></a>: like <a href="../library/signal.html#signal.sigwaitinfo" class="reference internal" title="signal.sigwaitinfo"><span class="pre"><code class="sourceCode python">sigwaitinfo()</code></span></a> but with a timeout.

- The signal handler writes the signal number as a single byte instead of a nul byte into the wakeup file descriptor. So it is possible to wait more than one signal and know which signals were raised.

- <a href="../library/signal.html#signal.signal" class="reference internal" title="signal.signal"><span class="pre"><code class="sourceCode python">signal.signal()</code></span></a> and <a href="../library/signal.html#signal.siginterrupt" class="reference internal" title="signal.siginterrupt"><span class="pre"><code class="sourceCode python">signal.siginterrupt()</code></span></a> raise an OSError, instead of a RuntimeError: OSError has an errno attribute.

</div>

<div id="smtpd" class="section">

### smtpd<a href="#smtpd" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`smtpd`</span> module now supports <span id="index-26" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc5321.html" class="rfc reference external"><strong>RFC 5321</strong></a> (extended SMTP) and <span id="index-27" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc1870.html" class="rfc reference external"><strong>RFC 1870</strong></a> (size extension). Per the standard, these extensions are enabled if and only if the client initiates the session with an <span class="pre">`EHLO`</span> command.

(Initial <span class="pre">`ELHO`</span> support by Alberto Trevino. Size extension by Juhana Jauhiainen. Substantial additional work on the patch contributed by Michele Orrù and Dan Boswell. <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8739" class="reference external">bpo-8739</a>)

</div>

<div id="smtplib" class="section">

### smtplib<a href="#smtplib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/smtplib.html#smtplib.SMTP" class="reference internal" title="smtplib.SMTP"><span class="pre"><code class="sourceCode python">SMTP</code></span></a>, <a href="../library/smtplib.html#smtplib.SMTP_SSL" class="reference internal" title="smtplib.SMTP_SSL"><span class="pre"><code class="sourceCode python">SMTP_SSL</code></span></a>, and <a href="../library/smtplib.html#smtplib.LMTP" class="reference internal" title="smtplib.LMTP"><span class="pre"><code class="sourceCode python">LMTP</code></span></a> classes now accept a <span class="pre">`source_address`</span> keyword argument to specify the <span class="pre">`(host,`</span>` `<span class="pre">`port)`</span> to use as the source address in the bind call when creating the outgoing socket. (Contributed by Paulo Scardine in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11281" class="reference external">bpo-11281</a>.)

<a href="../library/smtplib.html#smtplib.SMTP" class="reference internal" title="smtplib.SMTP"><span class="pre"><code class="sourceCode python">SMTP</code></span></a> now supports the context management protocol, allowing an <span class="pre">`SMTP`</span> instance to be used in a <span class="pre">`with`</span> statement. (Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11289" class="reference external">bpo-11289</a>.)

The <a href="../library/smtplib.html#smtplib.SMTP_SSL" class="reference internal" title="smtplib.SMTP_SSL"><span class="pre"><code class="sourceCode python">SMTP_SSL</code></span></a> constructor and the <a href="../library/smtplib.html#smtplib.SMTP.starttls" class="reference internal" title="smtplib.SMTP.starttls"><span class="pre"><code class="sourceCode python">starttls()</code></span></a> method now accept an SSLContext parameter to control parameters of the secure channel. (Contributed by Kasun Herath in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8809" class="reference external">bpo-8809</a>.)

</div>

<div id="socket" class="section">

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/socket.html#socket.socket" class="reference internal" title="socket.socket"><span class="pre"><code class="sourceCode python">socket</code></span></a> class now exposes additional methods to process ancillary data when supported by the underlying platform:

  - <a href="../library/socket.html#socket.socket.sendmsg" class="reference internal" title="socket.socket.sendmsg"><span class="pre"><code class="sourceCode python">sendmsg()</code></span></a>

  - <a href="../library/socket.html#socket.socket.recvmsg" class="reference internal" title="socket.socket.recvmsg"><span class="pre"><code class="sourceCode python">recvmsg()</code></span></a>

  - <a href="../library/socket.html#socket.socket.recvmsg_into" class="reference internal" title="socket.socket.recvmsg_into"><span class="pre"><code class="sourceCode python">recvmsg_into()</code></span></a>

  (Contributed by David Watson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6560" class="reference external">bpo-6560</a>, based on an earlier patch by Heiko Wundram)

- The <a href="../library/socket.html#socket.socket" class="reference internal" title="socket.socket"><span class="pre"><code class="sourceCode python">socket</code></span></a> class now supports the PF_CAN protocol family (<a href="https://en.wikipedia.org/wiki/Socketcan" class="reference external">https://en.wikipedia.org/wiki/Socketcan</a>), on Linux (<a href="https://lwn.net/Articles/253425" class="reference external">https://lwn.net/Articles/253425</a>).

  (Contributed by Matthias Fuchs, updated by Tiago Gonçalves in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10141" class="reference external">bpo-10141</a>.)

- The <a href="../library/socket.html#socket.socket" class="reference internal" title="socket.socket"><span class="pre"><code class="sourceCode python">socket</code></span></a> class now supports the PF_RDS protocol family (<a href="https://en.wikipedia.org/wiki/Reliable_Datagram_Sockets" class="reference external">https://en.wikipedia.org/wiki/Reliable_Datagram_Sockets</a> and <a href="https://web.archive.org/web/20130115155505/https://oss.oracle.com/projects/rds/" class="reference external">https://oss.oracle.com/projects/rds</a>).

- The <a href="../library/socket.html#socket.socket" class="reference internal" title="socket.socket"><span class="pre"><code class="sourceCode python">socket</code></span></a> class now supports the <span class="pre">`PF_SYSTEM`</span> protocol family on OS X. (Contributed by Michael Goderbauer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13777" class="reference external">bpo-13777</a>.)

- New function <a href="../library/socket.html#socket.sethostname" class="reference internal" title="socket.sethostname"><span class="pre"><code class="sourceCode python">sethostname()</code></span></a> allows the hostname to be set on Unix systems if the calling process has sufficient privileges. (Contributed by Ross Lagerwall in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10866" class="reference external">bpo-10866</a>.)

</div>

<div id="socketserver" class="section">

### socketserver<a href="#socketserver" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/socketserver.html#socketserver.BaseServer" class="reference internal" title="socketserver.BaseServer"><span class="pre"><code class="sourceCode python">BaseServer</code></span></a> now has an overridable method <a href="../library/socketserver.html#socketserver.BaseServer.service_actions" class="reference internal" title="socketserver.BaseServer.service_actions"><span class="pre"><code class="sourceCode python">service_actions()</code></span></a> that is called by the <a href="../library/socketserver.html#socketserver.BaseServer.serve_forever" class="reference internal" title="socketserver.BaseServer.serve_forever"><span class="pre"><code class="sourceCode python">serve_forever()</code></span></a> method in the service loop. <a href="../library/socketserver.html#socketserver.ForkingMixIn" class="reference internal" title="socketserver.ForkingMixIn"><span class="pre"><code class="sourceCode python">ForkingMixIn</code></span></a> now uses this to clean up zombie child processes. (Contributed by Justin Warkentin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11109" class="reference external">bpo-11109</a>.)

</div>

<div id="sqlite3" class="section">

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

New <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">sqlite3.Connection</code></span></a> method <a href="../library/sqlite3.html#sqlite3.Connection.set_trace_callback" class="reference internal" title="sqlite3.Connection.set_trace_callback"><span class="pre"><code class="sourceCode python">set_trace_callback()</code></span></a> can be used to capture a trace of all sql commands processed by sqlite. (Contributed by Torsten Landschoff in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11688" class="reference external">bpo-11688</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module has two new random generation functions:

  - <a href="../library/ssl.html#ssl.RAND_bytes" class="reference internal" title="ssl.RAND_bytes"><span class="pre"><code class="sourceCode python">RAND_bytes()</code></span></a>: generate cryptographically strong pseudo-random bytes.

  - <span class="pre">`RAND_pseudo_bytes()`</span>: generate pseudo-random bytes.

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12049" class="reference external">bpo-12049</a>.)

- The <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module now exposes a finer-grained exception hierarchy in order to make it easier to inspect the various kinds of errors. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11183" class="reference external">bpo-11183</a>.)

- <a href="../library/ssl.html#ssl.SSLContext.load_cert_chain" class="reference internal" title="ssl.SSLContext.load_cert_chain"><span class="pre"><code class="sourceCode python">load_cert_chain()</code></span></a> now accepts a *password* argument to be used if the private key is encrypted. (Contributed by Adam Simpkins in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12803" class="reference external">bpo-12803</a>.)

- Diffie-Hellman key exchange, both regular and Elliptic Curve-based, is now supported through the <a href="../library/ssl.html#ssl.SSLContext.load_dh_params" class="reference internal" title="ssl.SSLContext.load_dh_params"><span class="pre"><code class="sourceCode python">load_dh_params()</code></span></a> and <a href="../library/ssl.html#ssl.SSLContext.set_ecdh_curve" class="reference internal" title="ssl.SSLContext.set_ecdh_curve"><span class="pre"><code class="sourceCode python">set_ecdh_curve()</code></span></a> methods. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13626" class="reference external">bpo-13626</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13627" class="reference external">bpo-13627</a>.)

- SSL sockets have a new <a href="../library/ssl.html#ssl.SSLSocket.get_channel_binding" class="reference internal" title="ssl.SSLSocket.get_channel_binding"><span class="pre"><code class="sourceCode python">get_channel_binding()</code></span></a> method allowing the implementation of certain authentication mechanisms such as SCRAM-SHA-1-PLUS. (Contributed by Jacek Konieczny in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12551" class="reference external">bpo-12551</a>.)

- You can query the SSL compression algorithm used by an SSL socket, thanks to its new <a href="../library/ssl.html#ssl.SSLSocket.compression" class="reference internal" title="ssl.SSLSocket.compression"><span class="pre"><code class="sourceCode python">compression()</code></span></a> method. The new attribute <a href="../library/ssl.html#ssl.OP_NO_COMPRESSION" class="reference internal" title="ssl.OP_NO_COMPRESSION"><span class="pre"><code class="sourceCode python">OP_NO_COMPRESSION</code></span></a> can be used to disable compression. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13634" class="reference external">bpo-13634</a>.)

- Support has been added for the Next Protocol Negotiation extension using the <a href="../library/ssl.html#ssl.SSLContext.set_npn_protocols" class="reference internal" title="ssl.SSLContext.set_npn_protocols"><span class="pre"><code class="sourceCode python">ssl.SSLContext.set_npn_protocols()</code></span></a> method. (Contributed by Colin Marc in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14204" class="reference external">bpo-14204</a>.)

- SSL errors can now be introspected more easily thanks to <a href="../library/ssl.html#ssl.SSLError.library" class="reference internal" title="ssl.SSLError.library"><span class="pre"><code class="sourceCode python">library</code></span></a> and <a href="../library/ssl.html#ssl.SSLError.reason" class="reference internal" title="ssl.SSLError.reason"><span class="pre"><code class="sourceCode python">reason</code></span></a> attributes. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14837" class="reference external">bpo-14837</a>.)

- The <a href="../library/ssl.html#ssl.get_server_certificate" class="reference internal" title="ssl.get_server_certificate"><span class="pre"><code class="sourceCode python">get_server_certificate()</code></span></a> function now supports IPv6. (Contributed by Charles-François Natali in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11811" class="reference external">bpo-11811</a>.)

- New attribute <a href="../library/ssl.html#ssl.OP_CIPHER_SERVER_PREFERENCE" class="reference internal" title="ssl.OP_CIPHER_SERVER_PREFERENCE"><span class="pre"><code class="sourceCode python">OP_CIPHER_SERVER_PREFERENCE</code></span></a> allows setting SSLv3 server sockets to use the server’s cipher ordering preference rather than the client’s (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13635" class="reference external">bpo-13635</a>).

</div>

<div id="stat" class="section">

### stat<a href="#stat" class="headerlink" title="Link to this heading">¶</a>

The undocumented tarfile.filemode function has been moved to <a href="../library/stat.html#stat.filemode" class="reference internal" title="stat.filemode"><span class="pre"><code class="sourceCode python">stat.filemode()</code></span></a>. It can be used to convert a file’s mode to a string of the form ‘-rwxrwxrwx’.

(Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14807" class="reference external">bpo-14807</a>.)

</div>

<div id="struct" class="section">

### struct<a href="#struct" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/struct.html#module-struct" class="reference internal" title="struct: Interpret bytes as packed binary data."><span class="pre"><code class="sourceCode python">struct</code></span></a> module now supports <span class="pre">`ssize_t`</span> and <span class="pre">`size_t`</span> via the new codes <span class="pre">`n`</span> and <span class="pre">`N`</span>, respectively. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3163" class="reference external">bpo-3163</a>.)

</div>

<div id="subprocess" class="section">

### subprocess<a href="#subprocess" class="headerlink" title="Link to this heading">¶</a>

Command strings can now be bytes objects on posix platforms. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8513" class="reference external">bpo-8513</a>.)

A new constant <a href="../library/subprocess.html#subprocess.DEVNULL" class="reference internal" title="subprocess.DEVNULL"><span class="pre"><code class="sourceCode python">DEVNULL</code></span></a> allows suppressing output in a platform-independent fashion. (Contributed by Ross Lagerwall in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5870" class="reference external">bpo-5870</a>.)

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> module has a new <a href="../library/sys.html#sys.thread_info" class="reference internal" title="sys.thread_info"><span class="pre"><code class="sourceCode python">thread_info</code></span></a> <a href="../glossary.html#term-named-tuple" class="reference internal"><span class="xref std std-term">named tuple</span></a> holding information about the thread implementation (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11223" class="reference external">bpo-11223</a>).

</div>

<div id="tarfile" class="section">

### tarfile<a href="#tarfile" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> now supports <span class="pre">`lzma`</span> encoding via the <a href="../library/lzma.html#module-lzma" class="reference internal" title="lzma: A Python wrapper for the liblzma compression library."><span class="pre"><code class="sourceCode python">lzma</code></span></a> module. (Contributed by Lars Gustäbel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5689" class="reference external">bpo-5689</a>.)

</div>

<div id="tempfile" class="section">

### tempfile<a href="#tempfile" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/tempfile.html#tempfile.SpooledTemporaryFile" class="reference internal" title="tempfile.SpooledTemporaryFile"><span class="pre"><code class="sourceCode python">tempfile.SpooledTemporaryFile</code></span></a>'s <span class="pre">`truncate()`</span> method now accepts a <span class="pre">`size`</span> parameter. (Contributed by Ryan Kelly in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9957" class="reference external">bpo-9957</a>.)

</div>

<div id="textwrap" class="section">

### textwrap<a href="#textwrap" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/textwrap.html#module-textwrap" class="reference internal" title="textwrap: Text wrapping and filling"><span class="pre"><code class="sourceCode python">textwrap</code></span></a> module has a new <a href="../library/textwrap.html#textwrap.indent" class="reference internal" title="textwrap.indent"><span class="pre"><code class="sourceCode python">indent()</code></span></a> that makes it straightforward to add a common prefix to selected lines in a block of text (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13857" class="reference external">bpo-13857</a>).

</div>

<div id="threading" class="section">

### threading<a href="#threading" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/threading.html#threading.Condition" class="reference internal" title="threading.Condition"><span class="pre"><code class="sourceCode python">threading.Condition</code></span></a>, <a href="../library/threading.html#threading.Semaphore" class="reference internal" title="threading.Semaphore"><span class="pre"><code class="sourceCode python">threading.Semaphore</code></span></a>, <a href="../library/threading.html#threading.BoundedSemaphore" class="reference internal" title="threading.BoundedSemaphore"><span class="pre"><code class="sourceCode python">threading.BoundedSemaphore</code></span></a>, <a href="../library/threading.html#threading.Event" class="reference internal" title="threading.Event"><span class="pre"><code class="sourceCode python">threading.Event</code></span></a>, and <a href="../library/threading.html#threading.Timer" class="reference internal" title="threading.Timer"><span class="pre"><code class="sourceCode python">threading.Timer</code></span></a>, all of which used to be factory functions returning a class instance, are now classes and may be subclassed. (Contributed by Éric Araujo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10968" class="reference external">bpo-10968</a>.)

The <a href="../library/threading.html#threading.Thread" class="reference internal" title="threading.Thread"><span class="pre"><code class="sourceCode python">threading.Thread</code></span></a> constructor now accepts a <span class="pre">`daemon`</span> keyword argument to override the default behavior of inheriting the <span class="pre">`daemon`</span> flag value from the parent thread (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6064" class="reference external">bpo-6064</a>).

The formerly private function <span class="pre">`_thread.get_ident`</span> is now available as the public function <a href="../library/threading.html#threading.get_ident" class="reference internal" title="threading.get_ident"><span class="pre"><code class="sourceCode python">threading.get_ident()</code></span></a>. This eliminates several cases of direct access to the <span class="pre">`_thread`</span> module in the stdlib. Third party code that used <span class="pre">`_thread.get_ident`</span> should likewise be changed to use the new public interface.

</div>

<div id="time" class="section">

### time<a href="#time" class="headerlink" title="Link to this heading">¶</a>

The <span id="index-28" class="target"></span><a href="https://peps.python.org/pep-0418/" class="pep reference external"><strong>PEP 418</strong></a> added new functions to the <a href="../library/time.html#module-time" class="reference internal" title="time: Time access and conversions."><span class="pre"><code class="sourceCode python">time</code></span></a> module:

- <a href="../library/time.html#time.get_clock_info" class="reference internal" title="time.get_clock_info"><span class="pre"><code class="sourceCode python">get_clock_info()</code></span></a>: Get information on a clock.

- <a href="../library/time.html#time.monotonic" class="reference internal" title="time.monotonic"><span class="pre"><code class="sourceCode python">monotonic()</code></span></a>: Monotonic clock (cannot go backward), not affected by system clock updates.

- <a href="../library/time.html#time.perf_counter" class="reference internal" title="time.perf_counter"><span class="pre"><code class="sourceCode python">perf_counter()</code></span></a>: Performance counter with the highest available resolution to measure a short duration.

- <a href="../library/time.html#time.process_time" class="reference internal" title="time.process_time"><span class="pre"><code class="sourceCode python">process_time()</code></span></a>: Sum of the system and user CPU time of the current process.

Other new functions:

- <a href="../library/time.html#time.clock_getres" class="reference internal" title="time.clock_getres"><span class="pre"><code class="sourceCode python">clock_getres()</code></span></a>, <a href="../library/time.html#time.clock_gettime" class="reference internal" title="time.clock_gettime"><span class="pre"><code class="sourceCode python">clock_gettime()</code></span></a> and <a href="../library/time.html#time.clock_settime" class="reference internal" title="time.clock_settime"><span class="pre"><code class="sourceCode python">clock_settime()</code></span></a> functions with <span class="pre">`CLOCK_`</span>*<span class="pre">`xxx`</span>* constants. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10278" class="reference external">bpo-10278</a>.)

To improve cross platform consistency, <a href="../library/time.html#time.sleep" class="reference internal" title="time.sleep"><span class="pre"><code class="sourceCode python">sleep()</code></span></a> now raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> when passed a negative sleep value. Previously this was an error on posix, but produced an infinite sleep on Windows.

</div>

<div id="types" class="section">

### types<a href="#types" class="headerlink" title="Link to this heading">¶</a>

Add a new <a href="../library/types.html#types.MappingProxyType" class="reference internal" title="types.MappingProxyType"><span class="pre"><code class="sourceCode python">types.MappingProxyType</code></span></a> class: Read-only proxy of a mapping. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14386" class="reference external">bpo-14386</a>)

The new functions <a href="../library/types.html#types.new_class" class="reference internal" title="types.new_class"><span class="pre"><code class="sourceCode python">types.new_class()</code></span></a> and <a href="../library/types.html#types.prepare_class" class="reference internal" title="types.prepare_class"><span class="pre"><code class="sourceCode python">types.prepare_class()</code></span></a> provide support for <span id="index-29" class="target"></span><a href="https://peps.python.org/pep-3115/" class="pep reference external"><strong>PEP 3115</strong></a> compliant dynamic type creation. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14588" class="reference external">bpo-14588</a>)

</div>

<div id="unittest" class="section">

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/unittest.html#unittest.TestCase.assertRaises" class="reference internal" title="unittest.TestCase.assertRaises"><span class="pre"><code class="sourceCode python">assertRaises()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertRaisesRegex" class="reference internal" title="unittest.TestCase.assertRaisesRegex"><span class="pre"><code class="sourceCode python">assertRaisesRegex()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertWarns" class="reference internal" title="unittest.TestCase.assertWarns"><span class="pre"><code class="sourceCode python">assertWarns()</code></span></a>, and <a href="../library/unittest.html#unittest.TestCase.assertWarnsRegex" class="reference internal" title="unittest.TestCase.assertWarnsRegex"><span class="pre"><code class="sourceCode python">assertWarnsRegex()</code></span></a> now accept a keyword argument *msg* when used as context managers. (Contributed by Ezio Melotti and Winston Ewert in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10775" class="reference external">bpo-10775</a>.)

<a href="../library/unittest.html#unittest.TestCase.run" class="reference internal" title="unittest.TestCase.run"><span class="pre"><code class="sourceCode python">unittest.TestCase.run()</code></span></a> now returns the <a href="../library/unittest.html#unittest.TestResult" class="reference internal" title="unittest.TestResult"><span class="pre"><code class="sourceCode python">TestResult</code></span></a> object.

</div>

<div id="urllib" class="section">

### urllib<a href="#urllib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/urllib.request.html#urllib.request.Request" class="reference internal" title="urllib.request.Request"><span class="pre"><code class="sourceCode python">Request</code></span></a> class, now accepts a *method* argument used by <a href="../library/urllib.request.html#urllib.request.Request.get_method" class="reference internal" title="urllib.request.Request.get_method"><span class="pre"><code class="sourceCode python">get_method()</code></span></a> to determine what HTTP method should be used. For example, this will send a <span class="pre">`'HEAD'`</span> request:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> urlopen(Request('https://www.python.org', method='HEAD'))

</div>

</div>

(<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1673007" class="reference external">bpo-1673007</a>)

</div>

<div id="webbrowser" class="section">

### webbrowser<a href="#webbrowser" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/webbrowser.html#module-webbrowser" class="reference internal" title="webbrowser: Easy-to-use controller for web browsers."><span class="pre"><code class="sourceCode python">webbrowser</code></span></a> module supports more “browsers”: Google Chrome (named **chrome**, **chromium**, **chrome-browser** or **chromium-browser** depending on the version and operating system), and the generic launchers **xdg-open**, from the FreeDesktop.org project, and **gvfs-open**, which is the default URI handler for GNOME 3. (The former contributed by Arnaud Calmettes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13620" class="reference external">bpo-13620</a>, the latter by Matthias Klose in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14493" class="reference external">bpo-14493</a>.)

</div>

<div id="xml-etree-elementtree" class="section">

### xml.etree.ElementTree<a href="#xml-etree-elementtree" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> module now imports its C accelerator by default; there is no longer a need to explicitly import <span class="pre">`xml.etree.cElementTree`</span> (this module stays for backwards compatibility, but is now deprecated). In addition, the <span class="pre">`iter`</span> family of methods of <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element" class="reference internal" title="xml.etree.ElementTree.Element"><span class="pre"><code class="sourceCode python">Element</code></span></a> has been optimized (rewritten in C). The module’s documentation has also been greatly improved with added examples and a more detailed reference.

</div>

<div id="zlib" class="section">

### zlib<a href="#zlib" class="headerlink" title="Link to this heading">¶</a>

New attribute <a href="../library/zlib.html#zlib.Decompress.eof" class="reference internal" title="zlib.Decompress.eof"><span class="pre"><code class="sourceCode python">zlib.Decompress.eof</code></span></a> makes it possible to distinguish between a properly formed compressed stream and an incomplete or truncated one. (Contributed by Nadeem Vawda in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12646" class="reference external">bpo-12646</a>.)

New attribute <a href="../library/zlib.html#zlib.ZLIB_RUNTIME_VERSION" class="reference internal" title="zlib.ZLIB_RUNTIME_VERSION"><span class="pre"><code class="sourceCode python">zlib.ZLIB_RUNTIME_VERSION</code></span></a> reports the version string of the underlying <span class="pre">`zlib`</span> library that is loaded at runtime. (Contributed by Torsten Landschoff in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12306" class="reference external">bpo-12306</a>.)

</div>

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

Major performance enhancements have been added:

- Thanks to <span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a>, some operations on Unicode strings have been optimized:

  - the memory footprint is divided by 2 to 4 depending on the text

  - encode an ASCII string to UTF-8 doesn’t need to encode characters anymore, the UTF-8 representation is shared with the ASCII representation

  - the UTF-8 encoder has been optimized

  - repeating a single ASCII letter and getting a substring of an ASCII string is 4 times faster

- UTF-8 is now 2x to 4x faster. UTF-16 encoding is now up to 10x faster.

  (Contributed by Serhiy Storchaka, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14624" class="reference external">bpo-14624</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14738" class="reference external">bpo-14738</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15026" class="reference external">bpo-15026</a>.)

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Link to this heading">¶</a>

Changes to Python’s build process and to the C API include:

- New <span id="index-31" class="target"></span><a href="https://peps.python.org/pep-3118/" class="pep reference external"><strong>PEP 3118</strong></a> related function:

  - <a href="../c-api/memoryview.html#c.PyMemoryView_FromMemory" class="reference internal" title="PyMemoryView_FromMemory"><span class="pre"><code class="sourceCode c">PyMemoryView_FromMemory<span class="op">()</span></code></span></a>

- <span id="index-32" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a> added new Unicode types, macros and functions:

  - High-level API:

    - <a href="../c-api/unicode.html#c.PyUnicode_CopyCharacters" class="reference internal" title="PyUnicode_CopyCharacters"><span class="pre"><code class="sourceCode c">PyUnicode_CopyCharacters<span class="op">()</span></code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_FindChar" class="reference internal" title="PyUnicode_FindChar"><span class="pre"><code class="sourceCode c">PyUnicode_FindChar<span class="op">()</span></code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_GetLength" class="reference internal" title="PyUnicode_GetLength"><span class="pre"><code class="sourceCode c">PyUnicode_GetLength<span class="op">()</span></code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_GET_LENGTH" class="reference internal" title="PyUnicode_GET_LENGTH"><span class="pre"><code class="sourceCode c">PyUnicode_GET_LENGTH</code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_New" class="reference internal" title="PyUnicode_New"><span class="pre"><code class="sourceCode c">PyUnicode_New<span class="op">()</span></code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_Substring" class="reference internal" title="PyUnicode_Substring"><span class="pre"><code class="sourceCode c">PyUnicode_Substring<span class="op">()</span></code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_ReadChar" class="reference internal" title="PyUnicode_ReadChar"><span class="pre"><code class="sourceCode c">PyUnicode_ReadChar<span class="op">()</span></code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_WriteChar" class="reference internal" title="PyUnicode_WriteChar"><span class="pre"><code class="sourceCode c">PyUnicode_WriteChar<span class="op">()</span></code></span></a>

  - Low-level API:

    - <a href="../c-api/unicode.html#c.Py_UCS1" class="reference internal" title="Py_UCS1"><span class="pre"><code class="sourceCode c">Py_UCS1</code></span></a>, <a href="../c-api/unicode.html#c.Py_UCS2" class="reference internal" title="Py_UCS2"><span class="pre"><code class="sourceCode c">Py_UCS2</code></span></a>, <a href="../c-api/unicode.html#c.Py_UCS4" class="reference internal" title="Py_UCS4"><span class="pre"><code class="sourceCode c">Py_UCS4</code></span></a> types

    - <a href="../c-api/unicode.html#c.PyASCIIObject" class="reference internal" title="PyASCIIObject"><span class="pre"><code class="sourceCode c">PyASCIIObject</code></span></a> and <a href="../c-api/unicode.html#c.PyCompactUnicodeObject" class="reference internal" title="PyCompactUnicodeObject"><span class="pre"><code class="sourceCode c">PyCompactUnicodeObject</code></span></a> structures

    - <a href="../c-api/unicode.html#c.PyUnicode_READY" class="reference internal" title="PyUnicode_READY"><span class="pre"><code class="sourceCode c">PyUnicode_READY</code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_FromKindAndData" class="reference internal" title="PyUnicode_FromKindAndData"><span class="pre"><code class="sourceCode c">PyUnicode_FromKindAndData<span class="op">()</span></code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_AsUCS4" class="reference internal" title="PyUnicode_AsUCS4"><span class="pre"><code class="sourceCode c">PyUnicode_AsUCS4<span class="op">()</span></code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_AsUCS4Copy" class="reference internal" title="PyUnicode_AsUCS4Copy"><span class="pre"><code class="sourceCode c">PyUnicode_AsUCS4Copy<span class="op">()</span></code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_DATA" class="reference internal" title="PyUnicode_DATA"><span class="pre"><code class="sourceCode c">PyUnicode_DATA</code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_1BYTE_DATA" class="reference internal" title="PyUnicode_1BYTE_DATA"><span class="pre"><code class="sourceCode c">PyUnicode_1BYTE_DATA</code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_2BYTE_DATA" class="reference internal" title="PyUnicode_2BYTE_DATA"><span class="pre"><code class="sourceCode c">PyUnicode_2BYTE_DATA</code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_4BYTE_DATA" class="reference internal" title="PyUnicode_4BYTE_DATA"><span class="pre"><code class="sourceCode c">PyUnicode_4BYTE_DATA</code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_KIND" class="reference internal" title="PyUnicode_KIND"><span class="pre"><code class="sourceCode c">PyUnicode_KIND</code></span></a> with <span class="pre">`PyUnicode_Kind`</span> enum: <span class="pre">`PyUnicode_WCHAR_KIND`</span>, <a href="../c-api/unicode.html#c.PyUnicode_1BYTE_KIND" class="reference internal" title="PyUnicode_1BYTE_KIND"><span class="pre"><code class="sourceCode c">PyUnicode_1BYTE_KIND</code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_2BYTE_KIND" class="reference internal" title="PyUnicode_2BYTE_KIND"><span class="pre"><code class="sourceCode c">PyUnicode_2BYTE_KIND</code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_4BYTE_KIND" class="reference internal" title="PyUnicode_4BYTE_KIND"><span class="pre"><code class="sourceCode c">PyUnicode_4BYTE_KIND</code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_READ" class="reference internal" title="PyUnicode_READ"><span class="pre"><code class="sourceCode c">PyUnicode_READ</code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_READ_CHAR" class="reference internal" title="PyUnicode_READ_CHAR"><span class="pre"><code class="sourceCode c">PyUnicode_READ_CHAR</code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_WRITE" class="reference internal" title="PyUnicode_WRITE"><span class="pre"><code class="sourceCode c">PyUnicode_WRITE</code></span></a>

    - <a href="../c-api/unicode.html#c.PyUnicode_MAX_CHAR_VALUE" class="reference internal" title="PyUnicode_MAX_CHAR_VALUE"><span class="pre"><code class="sourceCode c">PyUnicode_MAX_CHAR_VALUE</code></span></a>

- <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple</code></span></a> now accepts a <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> for the <span class="pre">`c`</span> format (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12380" class="reference external">bpo-12380</a>).

</div>

<div id="deprecated" class="section">

## Deprecated<a href="#deprecated" class="headerlink" title="Link to this heading">¶</a>

<div id="unsupported-operating-systems" class="section">

### Unsupported Operating Systems<a href="#unsupported-operating-systems" class="headerlink" title="Link to this heading">¶</a>

OS/2 and VMS are no longer supported due to the lack of a maintainer.

Windows 2000 and Windows platforms which set <span class="pre">`COMSPEC`</span> to <span class="pre">`command.com`</span> are no longer supported due to maintenance burden.

OSF support, which was deprecated in 3.2, has been completely removed.

</div>

<div id="deprecated-python-modules-functions-and-methods" class="section">

### Deprecated Python modules, functions and methods<a href="#deprecated-python-modules-functions-and-methods" class="headerlink" title="Link to this heading">¶</a>

- Passing a non-empty string to <span class="pre">`object.__format__()`</span> is deprecated, and will produce a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> in Python 3.4 (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9856" class="reference external">bpo-9856</a>).

- The <span class="pre">`unicode_internal`</span> codec has been deprecated because of the <span id="index-33" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a>, use UTF-8, UTF-16 (<span class="pre">`utf-16-le`</span> or <span class="pre">`utf-16-be`</span>), or UTF-32 (<span class="pre">`utf-32-le`</span> or <span class="pre">`utf-32-be`</span>)

- <a href="../library/ftplib.html#ftplib.FTP.nlst" class="reference internal" title="ftplib.FTP.nlst"><span class="pre"><code class="sourceCode python">ftplib.FTP.nlst()</code></span></a> and <a href="../library/ftplib.html#ftplib.FTP.dir" class="reference internal" title="ftplib.FTP.dir"><span class="pre"><code class="sourceCode python">ftplib.FTP.<span class="bu">dir</span>()</code></span></a>: use <a href="../library/ftplib.html#ftplib.FTP.mlsd" class="reference internal" title="ftplib.FTP.mlsd"><span class="pre"><code class="sourceCode python">ftplib.FTP.mlsd()</code></span></a>

- <span class="pre">`platform.popen()`</span>: use the <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module. Check especially the <a href="../library/subprocess.html#subprocess-replacements" class="reference internal"><span class="std std-ref">Replacing Older Functions with the subprocess Module</span></a> section (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11377" class="reference external">bpo-11377</a>).

- <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13374" class="reference external">bpo-13374</a>: The Windows bytes API has been deprecated in the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module. Use Unicode filenames, instead of bytes filenames, to not depend on the ANSI code page anymore and to support any filename.

- <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13988" class="reference external">bpo-13988</a>: The <span class="pre">`xml.etree.cElementTree`</span> module is deprecated. The accelerator is used automatically whenever available.

- The behaviour of <span class="pre">`time.clock()`</span> depends on the platform: use the new <a href="../library/time.html#time.perf_counter" class="reference internal" title="time.perf_counter"><span class="pre"><code class="sourceCode python">time.perf_counter()</code></span></a> or <a href="../library/time.html#time.process_time" class="reference internal" title="time.process_time"><span class="pre"><code class="sourceCode python">time.process_time()</code></span></a> function instead, depending on your requirements, to have a well defined behaviour.

- The <span class="pre">`os.stat_float_times()`</span> function is deprecated.

- <a href="../library/abc.html#module-abc" class="reference internal" title="abc: Abstract base classes according to :pep:`3119`."><span class="pre"><code class="sourceCode python">abc</code></span></a> module:

  - <a href="../library/abc.html#abc.abstractproperty" class="reference internal" title="abc.abstractproperty"><span class="pre"><code class="sourceCode python">abc.abstractproperty</code></span></a> has been deprecated, use <a href="../library/functions.html#property" class="reference internal" title="property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span></code></span></a> with <a href="../library/abc.html#abc.abstractmethod" class="reference internal" title="abc.abstractmethod"><span class="pre"><code class="sourceCode python">abc.abstractmethod()</code></span></a> instead.

  - <a href="../library/abc.html#abc.abstractclassmethod" class="reference internal" title="abc.abstractclassmethod"><span class="pre"><code class="sourceCode python">abc.abstractclassmethod</code></span></a> has been deprecated, use <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span></code></span></a> with <a href="../library/abc.html#abc.abstractmethod" class="reference internal" title="abc.abstractmethod"><span class="pre"><code class="sourceCode python">abc.abstractmethod()</code></span></a> instead.

  - <a href="../library/abc.html#abc.abstractstaticmethod" class="reference internal" title="abc.abstractstaticmethod"><span class="pre"><code class="sourceCode python">abc.abstractstaticmethod</code></span></a> has been deprecated, use <a href="../library/functions.html#staticmethod" class="reference internal" title="staticmethod"><span class="pre"><code class="sourceCode python"><span class="bu">staticmethod</span></code></span></a> with <a href="../library/abc.html#abc.abstractmethod" class="reference internal" title="abc.abstractmethod"><span class="pre"><code class="sourceCode python">abc.abstractmethod()</code></span></a> instead.

- <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> package:

  - <a href="../library/importlib.html#importlib.abc.SourceLoader.path_mtime" class="reference internal" title="importlib.abc.SourceLoader.path_mtime"><span class="pre"><code class="sourceCode python">importlib.abc.SourceLoader.path_mtime()</code></span></a> is now deprecated in favour of <a href="../library/importlib.html#importlib.abc.SourceLoader.path_stats" class="reference internal" title="importlib.abc.SourceLoader.path_stats"><span class="pre"><code class="sourceCode python">importlib.abc.SourceLoader.path_stats()</code></span></a> as bytecode files now store both the modification time and size of the source file the bytecode file was compiled from.

</div>

<div id="deprecated-functions-and-types-of-the-c-api" class="section">

### Deprecated functions and types of the C API<a href="#deprecated-functions-and-types-of-the-c-api" class="headerlink" title="Link to this heading">¶</a>

The <a href="../c-api/unicode.html#c.Py_UNICODE" class="reference internal" title="Py_UNICODE"><span class="pre"><code class="sourceCode c">Py_UNICODE</code></span></a> has been deprecated by <span id="index-34" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a> and will be removed in Python 4. All functions using this type are deprecated:

Unicode functions and methods using <a href="../c-api/unicode.html#c.Py_UNICODE" class="reference internal" title="Py_UNICODE"><span class="pre"><code class="sourceCode c">Py_UNICODE</code></span></a> and <span class="c-expr sig sig-inline c"><a href="../c-api/unicode.html#c.Py_UNICODE" class="reference internal" title="Py_UNICODE"><span class="n">Py_UNICODE</span></a><span class="p">\*</span></span> types:

- <span class="pre">`PyUnicode_FromUnicode`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_FromWideChar" class="reference internal" title="PyUnicode_FromWideChar"><span class="pre"><code class="sourceCode c">PyUnicode_FromWideChar<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_FromKindAndData" class="reference internal" title="PyUnicode_FromKindAndData"><span class="pre"><code class="sourceCode c">PyUnicode_FromKindAndData<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_AS_UNICODE`</span>, <span class="pre">`PyUnicode_AsUnicode()`</span>, <span class="pre">`PyUnicode_AsUnicodeAndSize()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_AsWideCharString" class="reference internal" title="PyUnicode_AsWideCharString"><span class="pre"><code class="sourceCode c">PyUnicode_AsWideCharString<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_AS_DATA`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_DATA" class="reference internal" title="PyUnicode_DATA"><span class="pre"><code class="sourceCode c">PyUnicode_DATA</code></span></a> with <a href="../c-api/unicode.html#c.PyUnicode_READ" class="reference internal" title="PyUnicode_READ"><span class="pre"><code class="sourceCode c">PyUnicode_READ</code></span></a> and <a href="../c-api/unicode.html#c.PyUnicode_WRITE" class="reference internal" title="PyUnicode_WRITE"><span class="pre"><code class="sourceCode c">PyUnicode_WRITE</code></span></a>

- <span class="pre">`PyUnicode_GET_SIZE`</span>, <span class="pre">`PyUnicode_GetSize()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_GET_LENGTH" class="reference internal" title="PyUnicode_GET_LENGTH"><span class="pre"><code class="sourceCode c">PyUnicode_GET_LENGTH</code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_GetLength" class="reference internal" title="PyUnicode_GetLength"><span class="pre"><code class="sourceCode c">PyUnicode_GetLength<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_GET_DATA_SIZE`</span>: use <span class="pre">`PyUnicode_GET_LENGTH(str)`</span>` `<span class="pre">`*`</span>` `<span class="pre">`PyUnicode_KIND(str)`</span> (only work on ready strings)

- <span class="pre">`PyUnicode_AsUnicodeCopy()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_AsUCS4Copy" class="reference internal" title="PyUnicode_AsUCS4Copy"><span class="pre"><code class="sourceCode c">PyUnicode_AsUCS4Copy<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_AsWideCharString" class="reference internal" title="PyUnicode_AsWideCharString"><span class="pre"><code class="sourceCode c">PyUnicode_AsWideCharString<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_GetMax()`</span>

Functions and macros manipulating Py_UNICODE\* strings:

- <span class="pre">`Py_UNICODE_strlen()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_GetLength" class="reference internal" title="PyUnicode_GetLength"><span class="pre"><code class="sourceCode c">PyUnicode_GetLength<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_GET_LENGTH" class="reference internal" title="PyUnicode_GET_LENGTH"><span class="pre"><code class="sourceCode c">PyUnicode_GET_LENGTH</code></span></a>

- <span class="pre">`Py_UNICODE_strcat()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_CopyCharacters" class="reference internal" title="PyUnicode_CopyCharacters"><span class="pre"><code class="sourceCode c">PyUnicode_CopyCharacters<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_FromFormat" class="reference internal" title="PyUnicode_FromFormat"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormat<span class="op">()</span></code></span></a>

- <span class="pre">`Py_UNICODE_strcpy()`</span>, <span class="pre">`Py_UNICODE_strncpy()`</span>, <span class="pre">`Py_UNICODE_COPY()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_CopyCharacters" class="reference internal" title="PyUnicode_CopyCharacters"><span class="pre"><code class="sourceCode c">PyUnicode_CopyCharacters<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_Substring" class="reference internal" title="PyUnicode_Substring"><span class="pre"><code class="sourceCode c">PyUnicode_Substring<span class="op">()</span></code></span></a>

- <span class="pre">`Py_UNICODE_strcmp()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_Compare" class="reference internal" title="PyUnicode_Compare"><span class="pre"><code class="sourceCode c">PyUnicode_Compare<span class="op">()</span></code></span></a>

- <span class="pre">`Py_UNICODE_strncmp()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_Tailmatch" class="reference internal" title="PyUnicode_Tailmatch"><span class="pre"><code class="sourceCode c">PyUnicode_Tailmatch<span class="op">()</span></code></span></a>

- <span class="pre">`Py_UNICODE_strchr()`</span>, <span class="pre">`Py_UNICODE_strrchr()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_FindChar" class="reference internal" title="PyUnicode_FindChar"><span class="pre"><code class="sourceCode c">PyUnicode_FindChar<span class="op">()</span></code></span></a>

- <span class="pre">`Py_UNICODE_FILL()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_Fill" class="reference internal" title="PyUnicode_Fill"><span class="pre"><code class="sourceCode c">PyUnicode_Fill<span class="op">()</span></code></span></a>

- <span class="pre">`Py_UNICODE_MATCH`</span>

Encoders:

- <span class="pre">`PyUnicode_Encode()`</span>: use <span class="pre">`PyUnicode_AsEncodedObject()`</span>

- <span class="pre">`PyUnicode_EncodeUTF7()`</span>

- <span class="pre">`PyUnicode_EncodeUTF8()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_AsUTF8" class="reference internal" title="PyUnicode_AsUTF8"><span class="pre"><code class="sourceCode c">PyUnicode_AsUTF8<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_AsUTF8String" class="reference internal" title="PyUnicode_AsUTF8String"><span class="pre"><code class="sourceCode c">PyUnicode_AsUTF8String<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_EncodeUTF32()`</span>

- <span class="pre">`PyUnicode_EncodeUTF16()`</span>

- <span class="pre">`PyUnicode_EncodeUnicodeEscape()`</span> use <a href="../c-api/unicode.html#c.PyUnicode_AsUnicodeEscapeString" class="reference internal" title="PyUnicode_AsUnicodeEscapeString"><span class="pre"><code class="sourceCode c">PyUnicode_AsUnicodeEscapeString<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_EncodeRawUnicodeEscape()`</span> use <a href="../c-api/unicode.html#c.PyUnicode_AsRawUnicodeEscapeString" class="reference internal" title="PyUnicode_AsRawUnicodeEscapeString"><span class="pre"><code class="sourceCode c">PyUnicode_AsRawUnicodeEscapeString<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_EncodeLatin1()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_AsLatin1String" class="reference internal" title="PyUnicode_AsLatin1String"><span class="pre"><code class="sourceCode c">PyUnicode_AsLatin1String<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_EncodeASCII()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_AsASCIIString" class="reference internal" title="PyUnicode_AsASCIIString"><span class="pre"><code class="sourceCode c">PyUnicode_AsASCIIString<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_EncodeCharmap()`</span>

- <span class="pre">`PyUnicode_TranslateCharmap()`</span>

- <span class="pre">`PyUnicode_EncodeMBCS()`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_AsMBCSString" class="reference internal" title="PyUnicode_AsMBCSString"><span class="pre"><code class="sourceCode c">PyUnicode_AsMBCSString<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_EncodeCodePage" class="reference internal" title="PyUnicode_EncodeCodePage"><span class="pre"><code class="sourceCode c">PyUnicode_EncodeCodePage<span class="op">()</span></code></span></a> (with <span class="pre">`CP_ACP`</span> code_page)

- <span class="pre">`PyUnicode_EncodeDecimal()`</span>, <span class="pre">`PyUnicode_TransformDecimalToASCII()`</span>

</div>

<div id="deprecated-features" class="section">

### Deprecated features<a href="#deprecated-features" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/array.html#module-array" class="reference internal" title="array: Space efficient arrays of uniformly typed numeric values."><span class="pre"><code class="sourceCode python">array</code></span></a> module’s <span class="pre">`'u'`</span> format code is now deprecated and will be removed in Python 4 together with the rest of the (<a href="../c-api/unicode.html#c.Py_UNICODE" class="reference internal" title="Py_UNICODE"><span class="pre"><code class="sourceCode c">Py_UNICODE</code></span></a>) API.

</div>

</div>

<div id="porting-to-python-3-3" class="section">

## Porting to Python 3.3<a href="#porting-to-python-3-3" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="porting-python-code" class="section">

<span id="portingpythoncode"></span>

### Porting Python code<a href="#porting-python-code" class="headerlink" title="Link to this heading">¶</a>

- Hash randomization is enabled by default. Set the <span id="index-35" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONHASHSEED" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONHASHSEED</code></span></a> environment variable to <span class="pre">`0`</span> to disable hash randomization. See also the <a href="../reference/datamodel.html#object.__hash__" class="reference internal" title="object.__hash__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__hash__</span>()</code></span></a> method.

- <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12326" class="reference external">bpo-12326</a>: On Linux, sys.platform doesn’t contain the major version anymore. It is now always ‘linux’, instead of ‘linux2’ or ‘linux3’ depending on the Linux version used to build Python. Replace sys.platform == ‘linux2’ with sys.platform.startswith(‘linux’), or directly sys.platform == ‘linux’ if you don’t need to support older Python versions.

- <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13847" class="reference external">bpo-13847</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14180" class="reference external">bpo-14180</a>: <a href="../library/time.html#module-time" class="reference internal" title="time: Time access and conversions."><span class="pre"><code class="sourceCode python">time</code></span></a> and <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a>: <a href="../library/exceptions.html#OverflowError" class="reference internal" title="OverflowError"><span class="pre"><code class="sourceCode python"><span class="pp">OverflowError</span></code></span></a> is now raised instead of <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if a timestamp is out of range. <a href="../library/exceptions.html#OSError" class="reference internal" title="OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a> is now raised if C functions <span class="pre">`gmtime()`</span> or <span class="pre">`localtime()`</span> failed.

- The default finders used by import now utilize a cache of what is contained within a specific directory. If you create a Python source file or sourceless bytecode file, make sure to call <a href="../library/importlib.html#importlib.invalidate_caches" class="reference internal" title="importlib.invalidate_caches"><span class="pre"><code class="sourceCode python">importlib.invalidate_caches()</code></span></a> to clear out the cache for the finders to notice the new file.

- <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> now uses the full name of the module that was attempted to be imported. Doctests that check ImportErrors’ message will need to be updated to use the full name of the module instead of just the tail of the name.

- The *index* argument to <a href="../library/functions.html#import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> now defaults to 0 instead of -1 and no longer support negative values. It was an oversight when <span id="index-36" class="target"></span><a href="https://peps.python.org/pep-0328/" class="pep reference external"><strong>PEP 328</strong></a> was implemented that the default value remained -1. If you need to continue to perform a relative import followed by an absolute import, then perform the relative import using an index of 1, followed by another import using an index of 0. It is preferred, though, that you use <a href="../library/importlib.html#importlib.import_module" class="reference internal" title="importlib.import_module"><span class="pre"><code class="sourceCode python">importlib.import_module()</code></span></a> rather than call <a href="../library/functions.html#import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> directly.

- <a href="../library/functions.html#import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> no longer allows one to use an index value other than 0 for top-level modules. E.g. <span class="pre">`__import__('sys',`</span>` `<span class="pre">`level=1)`</span> is now an error.

- Because <a href="../library/sys.html#sys.meta_path" class="reference internal" title="sys.meta_path"><span class="pre"><code class="sourceCode python">sys.meta_path</code></span></a> and <a href="../library/sys.html#sys.path_hooks" class="reference internal" title="sys.path_hooks"><span class="pre"><code class="sourceCode python">sys.path_hooks</code></span></a> now have finders on them by default, you will most likely want to use <a href="../library/stdtypes.html#list.insert" class="reference internal" title="list.insert"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>.insert()</code></span></a> instead of <a href="../library/stdtypes.html#list.append" class="reference internal" title="list.append"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>.append()</code></span></a> to add to those lists.

- Because <span class="pre">`None`</span> is now inserted into <a href="../library/sys.html#sys.path_importer_cache" class="reference internal" title="sys.path_importer_cache"><span class="pre"><code class="sourceCode python">sys.path_importer_cache</code></span></a>, if you are clearing out entries in the dictionary of paths that do not have a finder, you will need to remove keys paired with values of <span class="pre">`None`</span> **and** <span class="pre">`imp.NullImporter`</span> to be backwards-compatible. This will lead to extra overhead on older versions of Python that re-insert <span class="pre">`None`</span> into <a href="../library/sys.html#sys.path_importer_cache" class="reference internal" title="sys.path_importer_cache"><span class="pre"><code class="sourceCode python">sys.path_importer_cache</code></span></a> where it represents the use of implicit finders, but semantically it should not change anything.

- <span class="pre">`importlib.abc.Finder`</span> no longer specifies a <span class="pre">`find_module()`</span> abstract method that must be implemented. If you were relying on subclasses to implement that method, make sure to check for the method’s existence first. You will probably want to check for <span class="pre">`find_loader()`</span> first, though, in the case of working with <a href="../glossary.html#term-path-entry-finder" class="reference internal"><span class="xref std std-term">path entry finders</span></a>.

- <a href="../library/pkgutil.html#module-pkgutil" class="reference internal" title="pkgutil: Utilities for the import system."><span class="pre"><code class="sourceCode python">pkgutil</code></span></a> has been converted to use <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> internally. This eliminates many edge cases where the old behaviour of the <span id="index-37" class="target"></span><a href="https://peps.python.org/pep-0302/" class="pep reference external"><strong>PEP 302</strong></a> import emulation failed to match the behaviour of the real import system. The import emulation itself is still present, but is now deprecated. The <a href="../library/pkgutil.html#pkgutil.iter_importers" class="reference internal" title="pkgutil.iter_importers"><span class="pre"><code class="sourceCode python">pkgutil.iter_importers()</code></span></a> and <a href="../library/pkgutil.html#pkgutil.walk_packages" class="reference internal" title="pkgutil.walk_packages"><span class="pre"><code class="sourceCode python">pkgutil.walk_packages()</code></span></a> functions special case the standard import hooks so they are still supported even though they do not provide the non-standard <span class="pre">`iter_modules()`</span> method.

- A longstanding RFC-compliance bug (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1079" class="reference external">bpo-1079</a>) in the parsing done by <a href="../library/email.header.html#email.header.decode_header" class="reference internal" title="email.header.decode_header"><span class="pre"><code class="sourceCode python">email.header.decode_header()</code></span></a> has been fixed. Code that uses the standard idiom to convert encoded headers into unicode (<span class="pre">`str(make_header(decode_header(h))`</span>) will see no change, but code that looks at the individual tuples returned by decode_header will see that whitespace that precedes or follows <span class="pre">`ASCII`</span> sections is now included in the <span class="pre">`ASCII`</span> section. Code that builds headers using <span class="pre">`make_header`</span> should also continue to work without change, since <span class="pre">`make_header`</span> continues to add whitespace between <span class="pre">`ASCII`</span> and non-<span class="pre">`ASCII`</span> sections if it is not already present in the input strings.

- <a href="../library/email.utils.html#email.utils.formataddr" class="reference internal" title="email.utils.formataddr"><span class="pre"><code class="sourceCode python">email.utils.formataddr()</code></span></a> now does the correct content transfer encoding when passed non-<span class="pre">`ASCII`</span> display names. Any code that depended on the previous buggy behavior that preserved the non-<span class="pre">`ASCII`</span> unicode in the formatted output string will need to be changed (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1690608" class="reference external">bpo-1690608</a>).

- <a href="../library/poplib.html#poplib.POP3.quit" class="reference internal" title="poplib.POP3.quit"><span class="pre"><code class="sourceCode python">poplib.POP3.quit()</code></span></a> may now raise protocol errors like all other <span class="pre">`poplib`</span> methods. Code that assumes <span class="pre">`quit`</span> does not raise <a href="../library/poplib.html#poplib.error_proto" class="reference internal" title="poplib.error_proto"><span class="pre"><code class="sourceCode python">poplib.error_proto</code></span></a> errors may need to be changed if errors on <span class="pre">`quit`</span> are encountered by a particular application (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11291" class="reference external">bpo-11291</a>).

- The <span class="pre">`strict`</span> argument to <a href="../library/email.parser.html#email.parser.Parser" class="reference internal" title="email.parser.Parser"><span class="pre"><code class="sourceCode python">email.parser.Parser</code></span></a>, deprecated since Python 2.4, has finally been removed.

- The deprecated method <span class="pre">`unittest.TestCase.assertSameElements`</span> has been removed.

- The deprecated variable <span class="pre">`time.accept2dyear`</span> has been removed.

- The deprecated <span class="pre">`Context._clamp`</span> attribute has been removed from the <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> module. It was previously replaced by the public attribute <a href="../library/decimal.html#decimal.Context.clamp" class="reference internal" title="decimal.Context.clamp"><span class="pre"><code class="sourceCode python">clamp</code></span></a>. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8540" class="reference external">bpo-8540</a>.)

- The undocumented internal helper class <span class="pre">`SSLFakeFile`</span> has been removed from <a href="../library/smtplib.html#module-smtplib" class="reference internal" title="smtplib: SMTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">smtplib</code></span></a>, since its functionality has long been provided directly by <a href="../library/socket.html#socket.socket.makefile" class="reference internal" title="socket.socket.makefile"><span class="pre"><code class="sourceCode python">socket.socket.makefile()</code></span></a>.

- Passing a negative value to <a href="../library/time.html#time.sleep" class="reference internal" title="time.sleep"><span class="pre"><code class="sourceCode python">time.sleep()</code></span></a> on Windows now raises an error instead of sleeping forever. It has always raised an error on posix.

- The <span class="pre">`ast.__version__`</span> constant has been removed. If you need to make decisions affected by the AST version, use <a href="../library/sys.html#sys.version_info" class="reference internal" title="sys.version_info"><span class="pre"><code class="sourceCode python">sys.version_info</code></span></a> to make the decision.

- Code that used to work around the fact that the <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a> module used factory functions by subclassing the private classes will need to change to subclass the now-public classes.

- The undocumented debugging machinery in the threading module has been removed, simplifying the code. This should have no effect on production code, but is mentioned here in case any application debug frameworks were interacting with it (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13550" class="reference external">bpo-13550</a>).

</div>

<div id="porting-c-code" class="section">

### Porting C code<a href="#porting-c-code" class="headerlink" title="Link to this heading">¶</a>

- In the course of changes to the buffer API the undocumented <span class="pre">`smalltable`</span> member of the <a href="../c-api/buffer.html#c.Py_buffer" class="reference internal" title="Py_buffer"><span class="pre"><code class="sourceCode c">Py_buffer</code></span></a> structure has been removed and the layout of the <span class="pre">`PyMemoryViewObject`</span> has changed.

  All extensions relying on the relevant parts in <span class="pre">`memoryobject.h`</span> or <span class="pre">`object.h`</span> must be rebuilt.

- Due to <a href="#pep-393" class="reference internal"><span class="std std-ref">PEP 393</span></a>, the <a href="../c-api/unicode.html#c.Py_UNICODE" class="reference internal" title="Py_UNICODE"><span class="pre"><code class="sourceCode c">Py_UNICODE</code></span></a> type and all functions using this type are deprecated (but will stay available for at least five years). If you were using low-level Unicode APIs to construct and access unicode objects and you want to benefit of the memory footprint reduction provided by <span id="index-38" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a>, you have to convert your code to the new <a href="../c-api/unicode.html" class="reference internal"><span class="doc">Unicode API</span></a>.

  However, if you only have been using high-level functions such as <a href="../c-api/unicode.html#c.PyUnicode_Concat" class="reference internal" title="PyUnicode_Concat"><span class="pre"><code class="sourceCode c">PyUnicode_Concat<span class="op">()</span></code></span></a>, <a href="../c-api/unicode.html#c.PyUnicode_Join" class="reference internal" title="PyUnicode_Join"><span class="pre"><code class="sourceCode c">PyUnicode_Join<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_FromFormat" class="reference internal" title="PyUnicode_FromFormat"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormat<span class="op">()</span></code></span></a>, your code will automatically take advantage of the new unicode representations.

- <a href="../c-api/import.html#c.PyImport_GetMagicNumber" class="reference internal" title="PyImport_GetMagicNumber"><span class="pre"><code class="sourceCode c">PyImport_GetMagicNumber<span class="op">()</span></code></span></a> now returns <span class="pre">`-1`</span> upon failure.

- As a negative value for the *level* argument to <a href="../library/functions.html#import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> is no longer valid, the same now holds for <a href="../c-api/import.html#c.PyImport_ImportModuleLevel" class="reference internal" title="PyImport_ImportModuleLevel"><span class="pre"><code class="sourceCode c">PyImport_ImportModuleLevel<span class="op">()</span></code></span></a>. This also means that the value of *level* used by <a href="../c-api/import.html#c.PyImport_ImportModuleEx" class="reference internal" title="PyImport_ImportModuleEx"><span class="pre"><code class="sourceCode c">PyImport_ImportModuleEx<span class="op">()</span></code></span></a> is now <span class="pre">`0`</span> instead of <span class="pre">`-1`</span>.

</div>

<div id="building-c-extensions" class="section">

### Building C extensions<a href="#building-c-extensions" class="headerlink" title="Link to this heading">¶</a>

- The range of possible file names for C extensions has been narrowed. Very rarely used spellings have been suppressed: under POSIX, files named <span class="pre">`xxxmodule.so`</span>, <span class="pre">`xxxmodule.abi3.so`</span> and <span class="pre">`xxxmodule.cpython-*.so`</span> are no longer recognized as implementing the <span class="pre">`xxx`</span> module. If you had been generating such files, you have to switch to the other spellings (i.e., remove the <span class="pre">`module`</span> string from the file names).

  (implemented in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14040" class="reference external">bpo-14040</a>.)

</div>

<div id="command-line-switch-changes" class="section">

### Command Line Switch Changes<a href="#command-line-switch-changes" class="headerlink" title="Link to this heading">¶</a>

- The -Q command-line flag and related artifacts have been removed. Code checking sys.flags.division_warning will need updating.

  (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10998" class="reference external">bpo-10998</a>, contributed by Éric Araujo.)

- When **python** is started with <a href="../using/cmdline.html#cmdoption-S" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-S</code></span></a>, <span class="pre">`import`</span>` `<span class="pre">`site`</span> will no longer add site-specific paths to the module search paths. In previous versions, it did.

  (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11591" class="reference external">bpo-11591</a>, contributed by Carl Meyer with editions by Éric Araujo.)

</div>

</div>

</div>

<div class="clearer">

</div>

</div>
