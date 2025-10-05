<div class="body" role="main">

<div id="what-s-new-in-python-3-5" class="section">

# What’s New In Python 3.5<a href="#what-s-new-in-python-3-5" class="headerlink" title="Link to this heading">¶</a>

Editors<span class="colon">:</span>  
Elvis Pranskevichus \<<a href="mailto:elvis%40magic.io" class="reference external">elvis<span>@</span>magic<span>.</span>io</a>\>, Yury Selivanov \<<a href="mailto:yury%40magic.io" class="reference external">yury<span>@</span>magic<span>.</span>io</a>\>

This article explains the new features in Python 3.5, compared to 3.4. Python 3.5 was released on September 13, 2015.  See the <a href="https://docs.python.org/3.5/whatsnew/changelog.html" class="reference external">changelog</a> for a full list of changes.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0478/" class="pep reference external"><strong>PEP 478</strong></a> - Python 3.5 Release Schedule

</div>

<div id="summary-release-highlights" class="section">

## Summary – Release highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

New syntax features:

- <a href="#whatsnew-pep-492" class="reference internal"><span class="std std-ref">PEP 492</span></a>, coroutines with async and await syntax.

- <a href="#whatsnew-pep-465" class="reference internal"><span class="std std-ref">PEP 465</span></a>, a new matrix multiplication operator: <span class="pre">`a`</span>` `<span class="pre">`@`</span>` `<span class="pre">`b`</span>.

- <a href="#whatsnew-pep-448" class="reference internal"><span class="std std-ref">PEP 448</span></a>, additional unpacking generalizations.

New library modules:

- <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a>: <a href="#whatsnew-pep-484" class="reference internal"><span class="std std-ref">PEP 484 – Type Hints</span></a>.

- <a href="../library/zipapp.html#module-zipapp" class="reference internal" title="zipapp: Manage executable Python zip archives"><span class="pre"><code class="sourceCode python">zipapp</code></span></a>: <a href="#whatsnew-zipapp" class="reference internal"><span class="std std-ref">PEP 441 Improving Python ZIP Application Support</span></a>.

New built-in features:

- <span class="pre">`bytes`</span>` `<span class="pre">`%`</span>` `<span class="pre">`args`</span>, <span class="pre">`bytearray`</span>` `<span class="pre">`%`</span>` `<span class="pre">`args`</span>: <a href="#whatsnew-pep-461" class="reference internal"><span class="std std-ref">PEP 461</span></a> – Adding <span class="pre">`%`</span> formatting to bytes and bytearray.

- New <a href="../library/stdtypes.html#bytes.hex" class="reference internal" title="bytes.hex"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span>.<span class="bu">hex</span>()</code></span></a>, <a href="../library/stdtypes.html#bytearray.hex" class="reference internal" title="bytearray.hex"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span>.<span class="bu">hex</span>()</code></span></a> and <a href="../library/stdtypes.html#memoryview.hex" class="reference internal" title="memoryview.hex"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span>.<span class="bu">hex</span>()</code></span></a> methods. (Contributed by Arnon Yaari in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9951" class="reference external">bpo-9951</a>.)

- <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> now supports tuple indexing (including multi-dimensional). (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23632" class="reference external">bpo-23632</a>.)

- Generators have a new <span class="pre">`gi_yieldfrom`</span> attribute, which returns the object being iterated by <span class="pre">`yield`</span>` `<span class="pre">`from`</span> expressions. (Contributed by Benno Leslie and Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24450" class="reference external">bpo-24450</a>.)

- A new <a href="../library/exceptions.html#RecursionError" class="reference internal" title="RecursionError"><span class="pre"><code class="sourceCode python"><span class="pp">RecursionError</span></code></span></a> exception is now raised when maximum recursion depth is reached. (Contributed by Georg Brandl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19235" class="reference external">bpo-19235</a>.)

CPython implementation improvements:

- When the <span class="pre">`LC_TYPE`</span> locale is the POSIX locale (<span class="pre">`C`</span> locale), <a href="../library/sys.html#sys.stdin" class="reference internal" title="sys.stdin"><span class="pre"><code class="sourceCode python">sys.stdin</code></span></a> and <a href="../library/sys.html#sys.stdout" class="reference internal" title="sys.stdout"><span class="pre"><code class="sourceCode python">sys.stdout</code></span></a> now use the <span class="pre">`surrogateescape`</span> error handler, instead of the <span class="pre">`strict`</span> error handler. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19977" class="reference external">bpo-19977</a>.)

- <span class="pre">`.pyo`</span> files are no longer used and have been replaced by a more flexible scheme that includes the optimization level explicitly in <span class="pre">`.pyc`</span> name. (See <a href="#whatsnew-pep-488" class="reference internal"><span class="std std-ref">PEP 488 overview</span></a>.)

- Builtin and extension modules are now initialized in a multi-phase process, which is similar to how Python modules are loaded. (See <a href="#whatsnew-pep-489" class="reference internal"><span class="std std-ref">PEP 489 overview</span></a>.)

Significant improvements in the standard library:

- <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">collections.OrderedDict</code></span></a> is now <a href="#whatsnew-ordereddict" class="reference internal"><span class="std std-ref">implemented in C</span></a>, which makes it 4 to 100 times faster.

- The <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module gained <a href="#whatsnew-sslmemorybio" class="reference internal"><span class="std std-ref">support for Memory BIO</span></a>, which decouples SSL protocol handling from network IO.

- The new <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a> function provides a <a href="#whatsnew-pep-471" class="reference internal"><span class="std std-ref">better and significantly faster way</span></a> of directory traversal.

- <a href="../library/functools.html#functools.lru_cache" class="reference internal" title="functools.lru_cache"><span class="pre"><code class="sourceCode python">functools.lru_cache()</code></span></a> has been mostly <a href="#whatsnew-lrucache" class="reference internal"><span class="std std-ref">reimplemented in C</span></a>, yielding much better performance.

- The new <a href="../library/subprocess.html#subprocess.run" class="reference internal" title="subprocess.run"><span class="pre"><code class="sourceCode python">subprocess.run()</code></span></a> function provides a <a href="#whatsnew-subprocess" class="reference internal"><span class="std std-ref">streamlined way to run subprocesses</span></a>.

- The <a href="../library/traceback.html#module-traceback" class="reference internal" title="traceback: Print or retrieve a stack traceback."><span class="pre"><code class="sourceCode python">traceback</code></span></a> module has been significantly <a href="#whatsnew-traceback" class="reference internal"><span class="std std-ref">enhanced</span></a> for improved performance and developer convenience.

Security improvements:

- SSLv3 is now disabled throughout the standard library. It can still be enabled by instantiating a <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a> manually. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22638" class="reference external">bpo-22638</a> for more details; this change was backported to CPython 3.4 and 2.7.)

- HTTP cookie parsing is now stricter, in order to protect against potential injection attacks. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22796" class="reference external">bpo-22796</a>.)

Windows improvements:

- A new installer for Windows has replaced the old MSI. See <a href="../using/windows.html#using-on-windows" class="reference internal"><span class="std std-ref">Using Python on Windows</span></a> for more information.

- Windows builds now use Microsoft Visual C++ 14.0, and extension modules should use the same.

Please read on for a comprehensive list of user-facing changes, including many other smaller improvements, CPython optimizations, deprecations, and potential porting issues.

</div>

<div id="new-features" class="section">

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-492-coroutines-with-async-and-await-syntax" class="section">

<span id="whatsnew-pep-492"></span>

### PEP 492 - Coroutines with async and await syntax<a href="#pep-492-coroutines-with-async-and-await-syntax" class="headerlink" title="Link to this heading">¶</a>

<span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0492/" class="pep reference external"><strong>PEP 492</strong></a> greatly improves support for asynchronous programming in Python by adding <a href="../glossary.html#term-awaitable" class="reference internal"><span class="xref std std-term">awaitable objects</span></a>, <a href="../glossary.html#term-coroutine-function" class="reference internal"><span class="xref std std-term">coroutine functions</span></a>, <a href="../glossary.html#term-asynchronous-iterable" class="reference internal"><span class="xref std std-term">asynchronous iteration</span></a>, and <a href="../glossary.html#term-asynchronous-context-manager" class="reference internal"><span class="xref std std-term">asynchronous context managers</span></a>.

Coroutine functions are declared using the new <a href="../reference/compound_stmts.html#async-def" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">def</code></span></a> syntax:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> async def coro():
    ...     return 'spam'

</div>

</div>

Inside a coroutine function, the new <a href="../reference/expressions.html#await" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">await</code></span></a> expression can be used to suspend coroutine execution until the result is available. Any object can be *awaited*, as long as it implements the <a href="../glossary.html#term-awaitable" class="reference internal"><span class="xref std std-term">awaitable</span></a> protocol by defining the <a href="../reference/datamodel.html#object.__await__" class="reference internal" title="object.__await__"><span class="pre"><code class="sourceCode python"><span class="fu">__await__</span>()</code></span></a> method.

PEP 492 also adds <a href="../reference/compound_stmts.html#async-for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> statement for convenient iteration over asynchronous iterables.

An example of a rudimentary HTTP client written using the new syntax:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import asyncio

    async def http_get(domain):
        reader, writer = await asyncio.open_connection(domain, 80)

        writer.write(b'\r\n'.join([
            b'GET / HTTP/1.1',
            b'Host: %b' % domain.encode('latin-1'),
            b'Connection: close',
            b'', b''
        ]))

        async for line in reader:
            print('>>>', line)

        writer.close()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(http_get('example.com'))
    finally:
        loop.close()

</div>

</div>

Similarly to asynchronous iteration, there is a new syntax for asynchronous context managers. The following script:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import asyncio

    async def coro(name, lock):
        print('coro {}: waiting for lock'.format(name))
        async with lock:
            print('coro {}: holding the lock'.format(name))
            await asyncio.sleep(1)
            print('coro {}: releasing the lock'.format(name))

    loop = asyncio.get_event_loop()
    lock = asyncio.Lock()
    coros = asyncio.gather(coro(1, lock), coro(2, lock))
    try:
        loop.run_until_complete(coros)
    finally:
        loop.close()

</div>

</div>

will output:

<div class="highlight-python3 notranslate">

<div class="highlight">

    coro 2: waiting for lock
    coro 2: holding the lock
    coro 1: waiting for lock
    coro 2: releasing the lock
    coro 1: holding the lock
    coro 1: releasing the lock

</div>

</div>

Note that both <a href="../reference/compound_stmts.html#async-for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> and <a href="../reference/compound_stmts.html#async-with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> can only be used inside a coroutine function declared with <a href="../reference/compound_stmts.html#async-def" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">def</code></span></a>.

Coroutine functions are intended to be run inside a compatible event loop, such as the <a href="../library/asyncio-eventloop.html#asyncio-event-loop" class="reference internal"><span class="std std-ref">asyncio loop</span></a>.

<div class="admonition note">

Note

<div class="versionchanged">

<span class="versionmodified changed">Changed in version 3.5.2: </span>Starting with CPython 3.5.2, <span class="pre">`__aiter__`</span> can directly return <a href="../glossary.html#term-asynchronous-iterator" class="reference internal"><span class="xref std std-term">asynchronous iterators</span></a>. Returning an <a href="../glossary.html#term-awaitable" class="reference internal"><span class="xref std std-term">awaitable</span></a> object will result in a <a href="../library/exceptions.html#PendingDeprecationWarning" class="reference internal" title="PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a>.

See more details in the <a href="../reference/datamodel.html#async-iterators" class="reference internal"><span class="std std-ref">Asynchronous Iterators</span></a> documentation section.

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0492/" class="pep reference external"><strong>PEP 492</strong></a> – Coroutines with async and await syntax  
PEP written and implemented by Yury Selivanov.

</div>

</div>

<div id="pep-465-a-dedicated-infix-operator-for-matrix-multiplication" class="section">

<span id="whatsnew-pep-465"></span>

### PEP 465 - A dedicated infix operator for matrix multiplication<a href="#pep-465-a-dedicated-infix-operator-for-matrix-multiplication" class="headerlink" title="Link to this heading">¶</a>

<span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0465/" class="pep reference external"><strong>PEP 465</strong></a> adds the <span class="pre">`@`</span> infix operator for matrix multiplication. Currently, no builtin Python types implement the new operator, however, it can be implemented by defining <a href="../reference/datamodel.html#object.__matmul__" class="reference internal" title="object.__matmul__"><span class="pre"><code class="sourceCode python"><span class="fu">__matmul__</span>()</code></span></a>, <a href="../reference/datamodel.html#object.__rmatmul__" class="reference internal" title="object.__rmatmul__"><span class="pre"><code class="sourceCode python"><span class="fu">__rmatmul__</span>()</code></span></a>, and <a href="../reference/datamodel.html#object.__imatmul__" class="reference internal" title="object.__imatmul__"><span class="pre"><code class="sourceCode python"><span class="fu">__imatmul__</span>()</code></span></a> for regular, reflected, and in-place matrix multiplication. The semantics of these methods is similar to that of methods defining other infix arithmetic operators.

Matrix multiplication is a notably common operation in many fields of mathematics, science, engineering, and the addition of <span class="pre">`@`</span> allows writing cleaner code:

<div class="highlight-python3 notranslate">

<div class="highlight">

    S = (H @ beta - r).T @ inv(H @ V @ H.T) @ (H @ beta - r)

</div>

</div>

instead of:

<div class="highlight-python3 notranslate">

<div class="highlight">

    S = dot((dot(H, beta) - r).T,
            dot(inv(dot(dot(H, V), H.T)), dot(H, beta) - r))

</div>

</div>

NumPy 1.10 has support for the new operator:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import numpy

    >>> x = numpy.ones(3)
    >>> x
    array([ 1., 1., 1.])

    >>> m = numpy.eye(3)
    >>> m
    array([[ 1., 0., 0.],
           [ 0., 1., 0.],
           [ 0., 0., 1.]])

    >>> x @ m
    array([ 1., 1., 1.])

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-4" class="target"></span><a href="https://peps.python.org/pep-0465/" class="pep reference external"><strong>PEP 465</strong></a> – A dedicated infix operator for matrix multiplication  
PEP written by Nathaniel J. Smith; implemented by Benjamin Peterson.

</div>

</div>

<div id="pep-448-additional-unpacking-generalizations" class="section">

<span id="whatsnew-pep-448"></span>

### PEP 448 - Additional Unpacking Generalizations<a href="#pep-448-additional-unpacking-generalizations" class="headerlink" title="Link to this heading">¶</a>

<span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0448/" class="pep reference external"><strong>PEP 448</strong></a> extends the allowed uses of the <span class="pre">`*`</span> iterable unpacking operator and <span class="pre">`**`</span> dictionary unpacking operator. It is now possible to use an arbitrary number of unpackings in <a href="../reference/expressions.html#calls" class="reference internal"><span class="std std-ref">function calls</span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> print(*[1], *[2], 3, *[4, 5])
    1 2 3 4 5

    >>> def fn(a, b, c, d):
    ...     print(a, b, c, d)
    ...

    >>> fn(**{'a': 1, 'c': 3}, **{'b': 2, 'd': 4})
    1 2 3 4

</div>

</div>

Similarly, tuple, list, set, and dictionary displays allow multiple unpackings (see <a href="../reference/expressions.html#exprlists" class="reference internal"><span class="std std-ref">Expression lists</span></a> and <a href="../reference/expressions.html#dict" class="reference internal"><span class="std std-ref">Dictionary displays</span></a>):

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> *range(4), 4
    (0, 1, 2, 3, 4)

    >>> [*range(4), 4]
    [0, 1, 2, 3, 4]

    >>> {*range(4), 4, *(5, 6, 7)}
    {0, 1, 2, 3, 4, 5, 6, 7}

    >>> {'x': 1, **{'y': 2}}
    {'x': 1, 'y': 2}

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0448/" class="pep reference external"><strong>PEP 448</strong></a> – Additional Unpacking Generalizations  
PEP written by Joshua Landau; implemented by Neil Girdhar, Thomas Wouters, and Joshua Landau.

</div>

</div>

<div id="pep-461-percent-formatting-support-for-bytes-and-bytearray" class="section">

<span id="whatsnew-pep-461"></span>

### PEP 461 - percent formatting support for bytes and bytearray<a href="#pep-461-percent-formatting-support-for-bytes-and-bytearray" class="headerlink" title="Link to this heading">¶</a>

<span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0461/" class="pep reference external"><strong>PEP 461</strong></a> adds support for the <span class="pre">`%`</span> <a href="../library/stdtypes.html#bytes-formatting" class="reference internal"><span class="std std-ref">interpolation operator</span></a> to <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>.

While interpolation is usually thought of as a string operation, there are cases where interpolation on <span class="pre">`bytes`</span> or <span class="pre">`bytearrays`</span> makes sense, and the work needed to make up for this missing functionality detracts from the overall readability of the code. This issue is particularly important when dealing with wire format protocols, which are often a mixture of binary and ASCII compatible text.

Examples:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> b'Hello %b!' % b'World'
    b'Hello World!'

    >>> b'x=%i y=%f' % (1, 2.5)
    b'x=1 y=2.500000'

</div>

</div>

Unicode is not allowed for <span class="pre">`%b`</span>, but it is accepted by <span class="pre">`%a`</span> (equivalent of <span class="pre">`repr(obj).encode('ascii',`</span>` `<span class="pre">`'backslashreplace')`</span>):

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> b'Hello %b!' % 'World'
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: %b requires bytes, or an object that implements __bytes__, not 'str'

    >>> b'price: %a' % '10€'
    b"price: '10\\u20ac'"

</div>

</div>

Note that <span class="pre">`%s`</span> and <span class="pre">`%r`</span> conversion types, although supported, should only be used in codebases that need compatibility with Python 2.

<div class="admonition seealso">

See also

<span id="index-8" class="target"></span><a href="https://peps.python.org/pep-0461/" class="pep reference external"><strong>PEP 461</strong></a> – Adding % formatting to bytes and bytearray  
PEP written by Ethan Furman; implemented by Neil Schemenauer and Ethan Furman.

</div>

</div>

<div id="pep-484-type-hints" class="section">

<span id="whatsnew-pep-484"></span>

### PEP 484 - Type Hints<a href="#pep-484-type-hints" class="headerlink" title="Link to this heading">¶</a>

Function annotation syntax has been a Python feature since version 3.0 (<span id="index-9" class="target"></span><a href="https://peps.python.org/pep-3107/" class="pep reference external"><strong>PEP 3107</strong></a>), however the semantics of annotations has been left undefined.

Experience has shown that the majority of function annotation uses were to provide type hints to function parameters and return values. It became evident that it would be beneficial for Python users, if the standard library included the base definitions and tools for type annotations.

<span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> introduces a <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional module</span></a> to provide these standard definitions and tools, along with some conventions for situations where annotations are not available.

For example, here is a simple function whose argument and return type are declared in the annotations:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def greeting(name: str) -> str:
        return 'Hello ' + name

</div>

</div>

While these annotations are available at runtime through the usual <span class="pre">`__annotations__`</span> attribute, *no automatic type checking happens at runtime*. Instead, it is assumed that a separate off-line type checker (e.g. <a href="https://mypy-lang.org" class="reference external">mypy</a>) will be used for on-demand source code analysis.

The type system supports unions, generic types, and a special type named <a href="../library/typing.html#typing.Any" class="reference internal" title="typing.Any"><span class="pre"><code class="sourceCode python">Any</code></span></a> which is consistent with (i.e. assignable to and from) all types.

<div class="admonition seealso">

See also

- <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module documentation

- <span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> – Type Hints  
  PEP written by Guido van Rossum, Jukka Lehtosalo, and Łukasz Langa; implemented by Guido van Rossum.

- <span id="index-12" class="target"></span><a href="https://peps.python.org/pep-0483/" class="pep reference external"><strong>PEP 483</strong></a> – The Theory of Type Hints  
  PEP written by Guido van Rossum

</div>

</div>

<div id="pep-471-os-scandir-function-a-better-and-faster-directory-iterator" class="section">

<span id="whatsnew-pep-471"></span>

### PEP 471 - os.scandir() function – a better and faster directory iterator<a href="#pep-471-os-scandir-function-a-better-and-faster-directory-iterator" class="headerlink" title="Link to this heading">¶</a>

<span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0471/" class="pep reference external"><strong>PEP 471</strong></a> adds a new directory iteration function, <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a>, to the standard library. Additionally, <a href="../library/os.html#os.walk" class="reference internal" title="os.walk"><span class="pre"><code class="sourceCode python">os.walk()</code></span></a> is now implemented using <span class="pre">`scandir`</span>, which makes it 3 to 5 times faster on POSIX systems and 7 to 20 times faster on Windows systems. This is largely achieved by greatly reducing the number of calls to <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> required to walk a directory tree.

Additionally, <span class="pre">`scandir`</span> returns an iterator, as opposed to returning a list of file names, which improves memory efficiency when iterating over very large directories.

The following example shows a simple use of <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a> to display all the files (excluding directories) in the given *path* that don’t start with <span class="pre">`'.'`</span>. The <a href="../library/os.html#os.DirEntry.is_file" class="reference internal" title="os.DirEntry.is_file"><span class="pre"><code class="sourceCode python">entry.is_file()</code></span></a> call will generally not make an additional system call:

<div class="highlight-python3 notranslate">

<div class="highlight">

    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_file():
            print(entry.name)

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0471/" class="pep reference external"><strong>PEP 471</strong></a> – os.scandir() function – a better and faster directory iterator  
PEP written and implemented by Ben Hoyt with the help of Victor Stinner.

</div>

</div>

<div id="pep-475-retry-system-calls-failing-with-eintr" class="section">

<span id="whatsnew-pep-475"></span>

### PEP 475: Retry system calls failing with EINTR<a href="#pep-475-retry-system-calls-failing-with-eintr" class="headerlink" title="Link to this heading">¶</a>

An <a href="../library/errno.html#errno.EINTR" class="reference internal" title="errno.EINTR"><span class="pre"><code class="sourceCode python">errno.EINTR</code></span></a> error code is returned whenever a system call, that is waiting for I/O, is interrupted by a signal. Previously, Python would raise <a href="../library/exceptions.html#InterruptedError" class="reference internal" title="InterruptedError"><span class="pre"><code class="sourceCode python"><span class="pp">InterruptedError</span></code></span></a> in such cases. This meant that, when writing a Python application, the developer had two choices:

1.  Ignore the <span class="pre">`InterruptedError`</span>.

2.  Handle the <span class="pre">`InterruptedError`</span> and attempt to restart the interrupted system call at every call site.

The first option makes an application fail intermittently. The second option adds a large amount of boilerplate that makes the code nearly unreadable. Compare:

<div class="highlight-python3 notranslate">

<div class="highlight">

    print("Hello World")

</div>

</div>

and:

<div class="highlight-python3 notranslate">

<div class="highlight">

    while True:
        try:
            print("Hello World")
            break
        except InterruptedError:
            continue

</div>

</div>

<span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0475/" class="pep reference external"><strong>PEP 475</strong></a> implements automatic retry of system calls on <span class="pre">`EINTR`</span>. This removes the burden of dealing with <span class="pre">`EINTR`</span> or <a href="../library/exceptions.html#InterruptedError" class="reference internal" title="InterruptedError"><span class="pre"><code class="sourceCode python"><span class="pp">InterruptedError</span></code></span></a> in user code in most situations and makes Python programs, including the standard library, more robust. Note that the system call is only retried if the signal handler does not raise an exception.

Below is a list of functions which are now retried when interrupted by a signal:

- <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> and <a href="../library/io.html#io.open" class="reference internal" title="io.open"><span class="pre"><code class="sourceCode python">io.<span class="bu">open</span>()</code></span></a>;

- functions of the <a href="../library/faulthandler.html#module-faulthandler" class="reference internal" title="faulthandler: Dump the Python traceback."><span class="pre"><code class="sourceCode python">faulthandler</code></span></a> module;

- <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> functions: <a href="../library/os.html#os.fchdir" class="reference internal" title="os.fchdir"><span class="pre"><code class="sourceCode python">fchdir()</code></span></a>, <a href="../library/os.html#os.fchmod" class="reference internal" title="os.fchmod"><span class="pre"><code class="sourceCode python">fchmod()</code></span></a>, <a href="../library/os.html#os.fchown" class="reference internal" title="os.fchown"><span class="pre"><code class="sourceCode python">fchown()</code></span></a>, <a href="../library/os.html#os.fdatasync" class="reference internal" title="os.fdatasync"><span class="pre"><code class="sourceCode python">fdatasync()</code></span></a>, <a href="../library/os.html#os.fstat" class="reference internal" title="os.fstat"><span class="pre"><code class="sourceCode python">fstat()</code></span></a>, <a href="../library/os.html#os.fstatvfs" class="reference internal" title="os.fstatvfs"><span class="pre"><code class="sourceCode python">fstatvfs()</code></span></a>, <a href="../library/os.html#os.fsync" class="reference internal" title="os.fsync"><span class="pre"><code class="sourceCode python">fsync()</code></span></a>, <a href="../library/os.html#os.ftruncate" class="reference internal" title="os.ftruncate"><span class="pre"><code class="sourceCode python">ftruncate()</code></span></a>, <a href="../library/os.html#os.mkfifo" class="reference internal" title="os.mkfifo"><span class="pre"><code class="sourceCode python">mkfifo()</code></span></a>, <a href="../library/os.html#os.mknod" class="reference internal" title="os.mknod"><span class="pre"><code class="sourceCode python">mknod()</code></span></a>, <a href="../library/os.html#os.open" class="reference internal" title="os.open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a>, <a href="../library/os.html#os.posix_fadvise" class="reference internal" title="os.posix_fadvise"><span class="pre"><code class="sourceCode python">posix_fadvise()</code></span></a>, <a href="../library/os.html#os.posix_fallocate" class="reference internal" title="os.posix_fallocate"><span class="pre"><code class="sourceCode python">posix_fallocate()</code></span></a>, <a href="../library/os.html#os.pread" class="reference internal" title="os.pread"><span class="pre"><code class="sourceCode python">pread()</code></span></a>, <a href="../library/os.html#os.pwrite" class="reference internal" title="os.pwrite"><span class="pre"><code class="sourceCode python">pwrite()</code></span></a>, <a href="../library/os.html#os.read" class="reference internal" title="os.read"><span class="pre"><code class="sourceCode python">read()</code></span></a>, <a href="../library/os.html#os.readv" class="reference internal" title="os.readv"><span class="pre"><code class="sourceCode python">readv()</code></span></a>, <a href="../library/os.html#os.sendfile" class="reference internal" title="os.sendfile"><span class="pre"><code class="sourceCode python">sendfile()</code></span></a>, <a href="../library/os.html#os.wait3" class="reference internal" title="os.wait3"><span class="pre"><code class="sourceCode python">wait3()</code></span></a>, <a href="../library/os.html#os.wait4" class="reference internal" title="os.wait4"><span class="pre"><code class="sourceCode python">wait4()</code></span></a>, <a href="../library/os.html#os.wait" class="reference internal" title="os.wait"><span class="pre"><code class="sourceCode python">wait()</code></span></a>, <a href="../library/os.html#os.waitid" class="reference internal" title="os.waitid"><span class="pre"><code class="sourceCode python">waitid()</code></span></a>, <a href="../library/os.html#os.waitpid" class="reference internal" title="os.waitpid"><span class="pre"><code class="sourceCode python">waitpid()</code></span></a>, <a href="../library/os.html#os.write" class="reference internal" title="os.write"><span class="pre"><code class="sourceCode python">write()</code></span></a>, <a href="../library/os.html#os.writev" class="reference internal" title="os.writev"><span class="pre"><code class="sourceCode python">writev()</code></span></a>;

- special cases: <a href="../library/os.html#os.close" class="reference internal" title="os.close"><span class="pre"><code class="sourceCode python">os.close()</code></span></a> and <a href="../library/os.html#os.dup2" class="reference internal" title="os.dup2"><span class="pre"><code class="sourceCode python">os.dup2()</code></span></a> now ignore <a href="../library/errno.html#errno.EINTR" class="reference internal" title="errno.EINTR"><span class="pre"><code class="sourceCode python">EINTR</code></span></a> errors; the syscall is not retried (see the PEP for the rationale);

- <a href="../library/select.html#module-select" class="reference internal" title="select: Wait for I/O completion on multiple streams."><span class="pre"><code class="sourceCode python">select</code></span></a> functions: <a href="../library/select.html#select.devpoll.poll" class="reference internal" title="select.devpoll.poll"><span class="pre"><code class="sourceCode python">devpoll.poll()</code></span></a>, <a href="../library/select.html#select.epoll.poll" class="reference internal" title="select.epoll.poll"><span class="pre"><code class="sourceCode python">epoll.poll()</code></span></a>, <a href="../library/select.html#select.kqueue.control" class="reference internal" title="select.kqueue.control"><span class="pre"><code class="sourceCode python">kqueue.control()</code></span></a>, <a href="../library/select.html#select.poll.poll" class="reference internal" title="select.poll.poll"><span class="pre"><code class="sourceCode python">poll.poll()</code></span></a>, <a href="../library/select.html#select.select" class="reference internal" title="select.select"><span class="pre"><code class="sourceCode python">select()</code></span></a>;

- methods of the <a href="../library/socket.html#socket.socket" class="reference internal" title="socket.socket"><span class="pre"><code class="sourceCode python">socket</code></span></a> class: <a href="../library/socket.html#socket.socket.accept" class="reference internal" title="socket.socket.accept"><span class="pre"><code class="sourceCode python">accept()</code></span></a>, <a href="../library/socket.html#socket.socket.connect" class="reference internal" title="socket.socket.connect"><span class="pre"><code class="sourceCode python"><span class="ex">connect</span>()</code></span></a> (except for non-blocking sockets), <a href="../library/socket.html#socket.socket.recv" class="reference internal" title="socket.socket.recv"><span class="pre"><code class="sourceCode python">recv()</code></span></a>, <a href="../library/socket.html#socket.socket.recvfrom" class="reference internal" title="socket.socket.recvfrom"><span class="pre"><code class="sourceCode python">recvfrom()</code></span></a>, <a href="../library/socket.html#socket.socket.recvmsg" class="reference internal" title="socket.socket.recvmsg"><span class="pre"><code class="sourceCode python">recvmsg()</code></span></a>, <a href="../library/socket.html#socket.socket.send" class="reference internal" title="socket.socket.send"><span class="pre"><code class="sourceCode python">send()</code></span></a>, <a href="../library/socket.html#socket.socket.sendall" class="reference internal" title="socket.socket.sendall"><span class="pre"><code class="sourceCode python">sendall()</code></span></a>, <a href="../library/socket.html#socket.socket.sendmsg" class="reference internal" title="socket.socket.sendmsg"><span class="pre"><code class="sourceCode python">sendmsg()</code></span></a>, <a href="../library/socket.html#socket.socket.sendto" class="reference internal" title="socket.socket.sendto"><span class="pre"><code class="sourceCode python">sendto()</code></span></a>;

- <a href="../library/signal.html#signal.sigtimedwait" class="reference internal" title="signal.sigtimedwait"><span class="pre"><code class="sourceCode python">signal.sigtimedwait()</code></span></a> and <a href="../library/signal.html#signal.sigwaitinfo" class="reference internal" title="signal.sigwaitinfo"><span class="pre"><code class="sourceCode python">signal.sigwaitinfo()</code></span></a>;

- <a href="../library/time.html#time.sleep" class="reference internal" title="time.sleep"><span class="pre"><code class="sourceCode python">time.sleep()</code></span></a>.

<div class="admonition seealso">

See also

<span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0475/" class="pep reference external"><strong>PEP 475</strong></a> – Retry system calls failing with EINTR  
PEP and implementation written by Charles-François Natali and Victor Stinner, with the help of Antoine Pitrou (the French connection).

</div>

</div>

<div id="pep-479-change-stopiteration-handling-inside-generators" class="section">

<span id="whatsnew-pep-479"></span>

### PEP 479: Change StopIteration handling inside generators<a href="#pep-479-change-stopiteration-handling-inside-generators" class="headerlink" title="Link to this heading">¶</a>

The interaction of generators and <a href="../library/exceptions.html#StopIteration" class="reference internal" title="StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a> in Python 3.4 and earlier was sometimes surprising, and could conceal obscure bugs. Previously, <span class="pre">`StopIteration`</span> raised accidentally inside a generator function was interpreted as the end of the iteration by the loop construct driving the generator.

<span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0479/" class="pep reference external"><strong>PEP 479</strong></a> changes the behavior of generators: when a <span class="pre">`StopIteration`</span> exception is raised inside a generator, it is replaced with a <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> before it exits the generator frame. The main goal of this change is to ease debugging in the situation where an unguarded <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a> call raises <span class="pre">`StopIteration`</span> and causes the iteration controlled by the generator to terminate silently. This is particularly pernicious in combination with the <span class="pre">`yield`</span>` `<span class="pre">`from`</span> construct.

This is a backwards incompatible change, so to enable the new behavior, a <a href="../glossary.html#term-__future__" class="reference internal"><span class="xref std std-term">__future__</span></a> import is necessary:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from __future__ import generator_stop

    >>> def gen():
    ...     next(iter([]))
    ...     yield
    ...
    >>> next(gen())
    Traceback (most recent call last):
      File "<stdin>", line 2, in gen
    StopIteration

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    RuntimeError: generator raised StopIteration

</div>

</div>

Without a <span class="pre">`__future__`</span> import, a <a href="../library/exceptions.html#PendingDeprecationWarning" class="reference internal" title="PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a> will be raised whenever a <a href="../library/exceptions.html#StopIteration" class="reference internal" title="StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a> exception is raised inside a generator.

<div class="admonition seealso">

See also

<span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0479/" class="pep reference external"><strong>PEP 479</strong></a> – Change StopIteration handling inside generators  
PEP written by Chris Angelico and Guido van Rossum. Implemented by Chris Angelico, Yury Selivanov and Nick Coghlan.

</div>

</div>

<div id="pep-485-a-function-for-testing-approximate-equality" class="section">

<span id="whatsnew-pep-485"></span>

### PEP 485: A function for testing approximate equality<a href="#pep-485-a-function-for-testing-approximate-equality" class="headerlink" title="Link to this heading">¶</a>

<span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0485/" class="pep reference external"><strong>PEP 485</strong></a> adds the <a href="../library/math.html#math.isclose" class="reference internal" title="math.isclose"><span class="pre"><code class="sourceCode python">math.isclose()</code></span></a> and <a href="../library/cmath.html#cmath.isclose" class="reference internal" title="cmath.isclose"><span class="pre"><code class="sourceCode python">cmath.isclose()</code></span></a> functions which tell whether two values are approximately equal or “close” to each other. Whether or not two values are considered close is determined according to given absolute and relative tolerances. Relative tolerance is the maximum allowed difference between <span class="pre">`isclose`</span> arguments, relative to the larger absolute value:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import math
    >>> a = 5.0
    >>> b = 4.99998
    >>> math.isclose(a, b, rel_tol=1e-5)
    True
    >>> math.isclose(a, b, rel_tol=1e-6)
    False

</div>

</div>

It is also possible to compare two values using absolute tolerance, which must be a non-negative value:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import math
    >>> a = 5.0
    >>> b = 4.99998
    >>> math.isclose(a, b, abs_tol=0.00003)
    True
    >>> math.isclose(a, b, abs_tol=0.00001)
    False

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-20" class="target"></span><a href="https://peps.python.org/pep-0485/" class="pep reference external"><strong>PEP 485</strong></a> – A function for testing approximate equality  
PEP written by Christopher Barker; implemented by Chris Barker and Tal Einat.

</div>

</div>

<div id="pep-486-make-the-python-launcher-aware-of-virtual-environments" class="section">

<span id="whatsnew-pep-486"></span>

### PEP 486: Make the Python Launcher aware of virtual environments<a href="#pep-486-make-the-python-launcher-aware-of-virtual-environments" class="headerlink" title="Link to this heading">¶</a>

<span id="index-21" class="target"></span><a href="https://peps.python.org/pep-0486/" class="pep reference external"><strong>PEP 486</strong></a> makes the Windows launcher (see <span id="index-22" class="target"></span><a href="https://peps.python.org/pep-0397/" class="pep reference external"><strong>PEP 397</strong></a>) aware of an active virtual environment. When the default interpreter would be used and the <span class="pre">`VIRTUAL_ENV`</span> environment variable is set, the interpreter in the virtual environment will be used.

<div class="admonition seealso">

See also

<span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0486/" class="pep reference external"><strong>PEP 486</strong></a> – Make the Python Launcher aware of virtual environments  
PEP written and implemented by Paul Moore.

</div>

</div>

<div id="pep-488-elimination-of-pyo-files" class="section">

<span id="whatsnew-pep-488"></span>

### PEP 488: Elimination of PYO files<a href="#pep-488-elimination-of-pyo-files" class="headerlink" title="Link to this heading">¶</a>

<span id="index-24" class="target"></span><a href="https://peps.python.org/pep-0488/" class="pep reference external"><strong>PEP 488</strong></a> does away with the concept of <span class="pre">`.pyo`</span> files. This means that <span class="pre">`.pyc`</span> files represent both unoptimized and optimized bytecode. To prevent the need to constantly regenerate bytecode files, <span class="pre">`.pyc`</span> files now have an optional <span class="pre">`opt-`</span> tag in their name when the bytecode is optimized. This has the side-effect of no more bytecode file name clashes when running under either <a href="../using/cmdline.html#cmdoption-O" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-O</code></span></a> or <a href="../using/cmdline.html#cmdoption-OO" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-OO</code></span></a>. Consequently, bytecode files generated from <a href="../using/cmdline.html#cmdoption-O" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-O</code></span></a>, and <a href="../using/cmdline.html#cmdoption-OO" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-OO</code></span></a> may now exist simultaneously. <a href="../library/importlib.html#importlib.util.cache_from_source" class="reference internal" title="importlib.util.cache_from_source"><span class="pre"><code class="sourceCode python">importlib.util.cache_from_source()</code></span></a> has an updated API to help with this change.

<div class="admonition seealso">

See also

<span id="index-25" class="target"></span><a href="https://peps.python.org/pep-0488/" class="pep reference external"><strong>PEP 488</strong></a> – Elimination of PYO files  
PEP written and implemented by Brett Cannon.

</div>

</div>

<div id="pep-489-multi-phase-extension-module-initialization" class="section">

<span id="whatsnew-pep-489"></span>

### PEP 489: Multi-phase extension module initialization<a href="#pep-489-multi-phase-extension-module-initialization" class="headerlink" title="Link to this heading">¶</a>

<span id="index-26" class="target"></span><a href="https://peps.python.org/pep-0489/" class="pep reference external"><strong>PEP 489</strong></a> updates extension module initialization to take advantage of the two step module loading mechanism introduced by <span id="index-27" class="target"></span><a href="https://peps.python.org/pep-0451/" class="pep reference external"><strong>PEP 451</strong></a> in Python 3.4.

This change brings the import semantics of extension modules that opt-in to using the new mechanism much closer to those of Python source and bytecode modules, including the ability to use any valid identifier as a module name, rather than being restricted to ASCII.

<div class="admonition seealso">

See also

<span id="index-28" class="target"></span><a href="https://peps.python.org/pep-0489/" class="pep reference external"><strong>PEP 489</strong></a> – Multi-phase extension module initialization  
PEP written by Petr Viktorin, Stefan Behnel, and Nick Coghlan; implemented by Petr Viktorin.

</div>

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

Some smaller changes made to the core Python language are:

- Added the <span class="pre">`"namereplace"`</span> error handlers. The <span class="pre">`"backslashreplace"`</span> error handlers now work with decoding and translating. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19676" class="reference external">bpo-19676</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22286" class="reference external">bpo-22286</a>.)

- The <a href="../using/cmdline.html#cmdoption-b" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-b</code></span></a> option now affects comparisons of <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> with <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23681" class="reference external">bpo-23681</a>.)

- New Kazakh <span class="pre">`kz1048`</span> and Tajik <span class="pre">`koi8_t`</span> <a href="../library/codecs.html#standard-encodings" class="reference internal"><span class="std std-ref">codecs</span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22682" class="reference external">bpo-22682</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22681" class="reference external">bpo-22681</a>.)

- Property docstrings are now writable. This is especially useful for <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">collections.namedtuple()</code></span></a> docstrings. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24064" class="reference external">bpo-24064</a>.)

- Circular imports involving relative imports are now supported. (Contributed by Brett Cannon and Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17636" class="reference external">bpo-17636</a>.)

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="typing" class="section">

### typing<a href="#typing" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional</span></a> module provides standard definitions and tools for function type annotations. See <a href="#whatsnew-pep-484" class="reference internal"><span class="std std-ref">Type Hints</span></a> for more information.

</div>

<div id="zipapp" class="section">

<span id="whatsnew-zipapp"></span>

### zipapp<a href="#zipapp" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/zipapp.html#module-zipapp" class="reference internal" title="zipapp: Manage executable Python zip archives"><span class="pre"><code class="sourceCode python">zipapp</code></span></a> module (specified in <span id="index-29" class="target"></span><a href="https://peps.python.org/pep-0441/" class="pep reference external"><strong>PEP 441</strong></a>) provides an API and command line tool for creating executable Python Zip Applications, which were introduced in Python 2.6 in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1739468" class="reference external">bpo-1739468</a>, but which were not well publicized, either at the time or since.

With the new module, bundling your application is as simple as putting all the files, including a <span class="pre">`__main__.py`</span> file, into a directory <span class="pre">`myapp`</span> and running:

<div class="highlight-shell-session notranslate">

<div class="highlight">

    $ python -m zipapp myapp
    $ python myapp.pyz

</div>

</div>

The module implementation has been contributed by Paul Moore in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23491" class="reference external">bpo-23491</a>.

<div class="admonition seealso">

See also

<span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0441/" class="pep reference external"><strong>PEP 441</strong></a> – Improving Python ZIP Application Support

</div>

</div>

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="argparse" class="section">

### argparse<a href="#argparse" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/argparse.html#argparse.ArgumentParser" class="reference internal" title="argparse.ArgumentParser"><span class="pre"><code class="sourceCode python">ArgumentParser</code></span></a> class now allows disabling <a href="../library/argparse.html#prefix-matching" class="reference internal"><span class="std std-ref">abbreviated usage</span></a> of long options by setting <a href="../library/argparse.html#allow-abbrev" class="reference internal"><span class="std std-ref">allow_abbrev</span></a> to <span class="pre">`False`</span>. (Contributed by Jonathan Paugh, Steven Bethard, paul j3 and Daniel Eriksson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14910" class="reference external">bpo-14910</a>.)

</div>

<div id="asyncio" class="section">

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

Since the <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> module is <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional</span></a>, all changes introduced in Python 3.5 have also been backported to Python 3.4.x.

Notable changes in the <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> module since Python 3.4.0:

- New debugging APIs: <a href="../library/asyncio-eventloop.html#asyncio.loop.set_debug" class="reference internal" title="asyncio.loop.set_debug"><span class="pre"><code class="sourceCode python">loop.set_debug()</code></span></a> and <a href="../library/asyncio-eventloop.html#asyncio.loop.get_debug" class="reference internal" title="asyncio.loop.get_debug"><span class="pre"><code class="sourceCode python">loop.get_debug()</code></span></a> methods. (Contributed by Victor Stinner.)

- The proactor event loop now supports SSL. (Contributed by Antoine Pitrou and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22560" class="reference external">bpo-22560</a>.)

- A new <a href="../library/asyncio-eventloop.html#asyncio.loop.is_closed" class="reference internal" title="asyncio.loop.is_closed"><span class="pre"><code class="sourceCode python">loop.is_closed()</code></span></a> method to check if the event loop is closed. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21326" class="reference external">bpo-21326</a>.)

- A new <a href="../library/asyncio-eventloop.html#asyncio.loop.create_task" class="reference internal" title="asyncio.loop.create_task"><span class="pre"><code class="sourceCode python">loop.create_task()</code></span></a> to conveniently create and schedule a new <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">Task</code></span></a> for a coroutine. The <span class="pre">`create_task`</span> method is also used by all asyncio functions that wrap coroutines into tasks, such as <a href="../library/asyncio-task.html#asyncio.wait" class="reference internal" title="asyncio.wait"><span class="pre"><code class="sourceCode python">asyncio.wait()</code></span></a>, <a href="../library/asyncio-task.html#asyncio.gather" class="reference internal" title="asyncio.gather"><span class="pre"><code class="sourceCode python">asyncio.gather()</code></span></a>, etc. (Contributed by Victor Stinner.)

- A new <a href="../library/asyncio-protocol.html#asyncio.WriteTransport.get_write_buffer_limits" class="reference internal" title="asyncio.WriteTransport.get_write_buffer_limits"><span class="pre"><code class="sourceCode python">transport.get_write_buffer_limits()</code></span></a> method to inquire for *high-* and *low-* water limits of the flow control. (Contributed by Victor Stinner.)

- The <span class="pre">`async()`</span> function is deprecated in favor of <a href="../library/asyncio-future.html#asyncio.ensure_future" class="reference internal" title="asyncio.ensure_future"><span class="pre"><code class="sourceCode python">ensure_future()</code></span></a>. (Contributed by Yury Selivanov.)

- New <a href="../library/asyncio-eventloop.html#asyncio.loop.set_task_factory" class="reference internal" title="asyncio.loop.set_task_factory"><span class="pre"><code class="sourceCode python">loop.set_task_factory()</code></span></a> and <a href="../library/asyncio-eventloop.html#asyncio.loop.get_task_factory" class="reference internal" title="asyncio.loop.get_task_factory"><span class="pre"><code class="sourceCode python">loop.get_task_factory()</code></span></a> methods to customize the task factory that <a href="../library/asyncio-eventloop.html#asyncio.loop.create_task" class="reference internal" title="asyncio.loop.create_task"><span class="pre"><code class="sourceCode python">loop.create_task()</code></span></a> method uses. (Contributed by Yury Selivanov.)

- New <a href="../library/asyncio-queue.html#asyncio.Queue.join" class="reference internal" title="asyncio.Queue.join"><span class="pre"><code class="sourceCode python">Queue.join()</code></span></a> and <a href="../library/asyncio-queue.html#asyncio.Queue.task_done" class="reference internal" title="asyncio.Queue.task_done"><span class="pre"><code class="sourceCode python">Queue.task_done()</code></span></a> queue methods. (Contributed by Victor Stinner.)

- The <span class="pre">`JoinableQueue`</span> class was removed, in favor of the <a href="../library/asyncio-queue.html#asyncio.Queue" class="reference internal" title="asyncio.Queue"><span class="pre"><code class="sourceCode python">asyncio.Queue</code></span></a> class. (Contributed by Victor Stinner.)

Updates in 3.5.1:

- The <a href="../library/asyncio-future.html#asyncio.ensure_future" class="reference internal" title="asyncio.ensure_future"><span class="pre"><code class="sourceCode python">ensure_future()</code></span></a> function and all functions that use it, such as <a href="../library/asyncio-eventloop.html#asyncio.loop.run_until_complete" class="reference internal" title="asyncio.loop.run_until_complete"><span class="pre"><code class="sourceCode python">loop.run_until_complete()</code></span></a>, now accept all kinds of <a href="../glossary.html#term-awaitable" class="reference internal"><span class="xref std std-term">awaitable objects</span></a>. (Contributed by Yury Selivanov.)

- New <a href="../library/asyncio-task.html#asyncio.run_coroutine_threadsafe" class="reference internal" title="asyncio.run_coroutine_threadsafe"><span class="pre"><code class="sourceCode python">run_coroutine_threadsafe()</code></span></a> function to submit coroutines to event loops from other threads. (Contributed by Vincent Michel.)

- New <a href="../library/asyncio-protocol.html#asyncio.BaseTransport.is_closing" class="reference internal" title="asyncio.BaseTransport.is_closing"><span class="pre"><code class="sourceCode python">Transport.is_closing()</code></span></a> method to check if the transport is closing or closed. (Contributed by Yury Selivanov.)

- The <a href="../library/asyncio-eventloop.html#asyncio.loop.create_server" class="reference internal" title="asyncio.loop.create_server"><span class="pre"><code class="sourceCode python">loop.create_server()</code></span></a> method can now accept a list of hosts. (Contributed by Yann Sionneau.)

Updates in 3.5.2:

- New <a href="../library/asyncio-eventloop.html#asyncio.loop.create_future" class="reference internal" title="asyncio.loop.create_future"><span class="pre"><code class="sourceCode python">loop.create_future()</code></span></a> method to create Future objects. This allows alternative event loop implementations, such as <a href="https://github.com/MagicStack/uvloop" class="reference external">uvloop</a>, to provide a faster <a href="../library/asyncio-future.html#asyncio.Future" class="reference internal" title="asyncio.Future"><span class="pre"><code class="sourceCode python">asyncio.Future</code></span></a> implementation. (Contributed by Yury Selivanov.)

- New <a href="../library/asyncio-eventloop.html#asyncio.loop.get_exception_handler" class="reference internal" title="asyncio.loop.get_exception_handler"><span class="pre"><code class="sourceCode python">loop.get_exception_handler()</code></span></a> method to get the current exception handler. (Contributed by Yury Selivanov.)

- New <a href="../library/asyncio-stream.html#asyncio.StreamReader.readuntil" class="reference internal" title="asyncio.StreamReader.readuntil"><span class="pre"><code class="sourceCode python">StreamReader.readuntil()</code></span></a> method to read data from the stream until a separator bytes sequence appears. (Contributed by Mark Korenberg.)

- The <a href="../library/asyncio-eventloop.html#asyncio.loop.create_connection" class="reference internal" title="asyncio.loop.create_connection"><span class="pre"><code class="sourceCode python">loop.create_connection()</code></span></a> and <a href="../library/asyncio-eventloop.html#asyncio.loop.create_server" class="reference internal" title="asyncio.loop.create_server"><span class="pre"><code class="sourceCode python">loop.create_server()</code></span></a> methods are optimized to avoid calling the system <span class="pre">`getaddrinfo`</span> function if the address is already resolved. (Contributed by A. Jesse Jiryu Davis.)

- The <a href="../library/asyncio-eventloop.html#asyncio.loop.sock_connect" class="reference internal" title="asyncio.loop.sock_connect"><span class="pre"><code class="sourceCode python">loop.sock_connect(sock,</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">address)</code></span></a> no longer requires the *address* to be resolved prior to the call. (Contributed by A. Jesse Jiryu Davis.)

</div>

<div id="bz2" class="section">

### bz2<a href="#bz2" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/bz2.html#bz2.BZ2Decompressor.decompress" class="reference internal" title="bz2.BZ2Decompressor.decompress"><span class="pre"><code class="sourceCode python">BZ2Decompressor.decompress</code></span></a> method now accepts an optional *max_length* argument to limit the maximum size of decompressed data. (Contributed by Nikolaus Rath in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15955" class="reference external">bpo-15955</a>.)

</div>

<div id="cgi" class="section">

### cgi<a href="#cgi" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`FieldStorage`</span> class now supports the <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> protocol. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20289" class="reference external">bpo-20289</a>.)

</div>

<div id="cmath" class="section">

### cmath<a href="#cmath" class="headerlink" title="Link to this heading">¶</a>

A new function <a href="../library/cmath.html#cmath.isclose" class="reference internal" title="cmath.isclose"><span class="pre"><code class="sourceCode python">isclose()</code></span></a> provides a way to test for approximate equality. (Contributed by Chris Barker and Tal Einat in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24270" class="reference external">bpo-24270</a>.)

</div>

<div id="code" class="section">

### code<a href="#code" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/code.html#code.InteractiveInterpreter.showtraceback" class="reference internal" title="code.InteractiveInterpreter.showtraceback"><span class="pre"><code class="sourceCode python">InteractiveInterpreter.showtraceback()</code></span></a> method now prints the full chained traceback, just like the interactive interpreter. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17442" class="reference external">bpo-17442</a>.)

</div>

<div id="collections" class="section">

### collections<a href="#collections" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">OrderedDict</code></span></a> class is now implemented in C, which makes it 4 to 100 times faster. (Contributed by Eric Snow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16991" class="reference external">bpo-16991</a>.)

<span class="pre">`OrderedDict.items()`</span>, <span class="pre">`OrderedDict.keys()`</span>, and <span class="pre">`OrderedDict.values()`</span> views now support <a href="../library/functions.html#reversed" class="reference internal" title="reversed"><span class="pre"><code class="sourceCode python"><span class="bu">reversed</span>()</code></span></a> iteration. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19505" class="reference external">bpo-19505</a>.)

The <a href="../library/collections.html#collections.deque" class="reference internal" title="collections.deque"><span class="pre"><code class="sourceCode python">deque</code></span></a> class now defines <a href="../library/collections.html#collections.deque.index" class="reference internal" title="collections.deque.index"><span class="pre"><code class="sourceCode python">index()</code></span></a>, <a href="../library/collections.html#collections.deque.insert" class="reference internal" title="collections.deque.insert"><span class="pre"><code class="sourceCode python">insert()</code></span></a>, and <a href="../library/collections.html#collections.deque.copy" class="reference internal" title="collections.deque.copy"><span class="pre"><code class="sourceCode python">copy()</code></span></a>, and supports the <span class="pre">`+`</span> and <span class="pre">`*`</span> operators. This allows deques to be recognized as a <a href="../library/collections.abc.html#collections.abc.MutableSequence" class="reference internal" title="collections.abc.MutableSequence"><span class="pre"><code class="sourceCode python">MutableSequence</code></span></a> and improves their substitutability for lists. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23704" class="reference external">bpo-23704</a>.)

Docstrings produced by <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">namedtuple()</code></span></a> can now be updated:

<div class="highlight-python3 notranslate">

<div class="highlight">

    Point = namedtuple('Point', ['x', 'y'])
    Point.__doc__ += ': Cartesian coordinate'
    Point.x.__doc__ = 'abscissa'
    Point.y.__doc__ = 'ordinate'

</div>

</div>

(Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24064" class="reference external">bpo-24064</a>.)

The <a href="../library/collections.html#collections.UserString" class="reference internal" title="collections.UserString"><span class="pre"><code class="sourceCode python">UserString</code></span></a> class now implements the <a href="../library/pickle.html#object.__getnewargs__" class="reference internal" title="object.__getnewargs__"><span class="pre"><code class="sourceCode python">__getnewargs__()</code></span></a>, <a href="../reference/datamodel.html#object.__rmod__" class="reference internal" title="object.__rmod__"><span class="pre"><code class="sourceCode python"><span class="fu">__rmod__</span>()</code></span></a>, <a href="../library/stdtypes.html#str.casefold" class="reference internal" title="str.casefold"><span class="pre"><code class="sourceCode python">casefold()</code></span></a>, <a href="../library/stdtypes.html#str.format_map" class="reference internal" title="str.format_map"><span class="pre"><code class="sourceCode python">format_map()</code></span></a>, <a href="../library/stdtypes.html#str.isprintable" class="reference internal" title="str.isprintable"><span class="pre"><code class="sourceCode python">isprintable()</code></span></a>, and <a href="../library/stdtypes.html#str.maketrans" class="reference internal" title="str.maketrans"><span class="pre"><code class="sourceCode python">maketrans()</code></span></a> methods to match the corresponding methods of <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>. (Contributed by Joe Jevnik in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22189" class="reference external">bpo-22189</a>.)

</div>

<div id="collections-abc" class="section">

### collections.abc<a href="#collections-abc" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`Sequence.index()`</span> method now accepts *start* and *stop* arguments to match the corresponding methods of <a href="../library/stdtypes.html#tuple" class="reference internal" title="tuple"><span class="pre"><code class="sourceCode python"><span class="bu">tuple</span></code></span></a>, <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a>, etc. (Contributed by Devin Jeanpierre in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23086" class="reference external">bpo-23086</a>.)

A new <a href="../library/collections.abc.html#collections.abc.Generator" class="reference internal" title="collections.abc.Generator"><span class="pre"><code class="sourceCode python">Generator</code></span></a> abstract base class. (Contributed by Stefan Behnel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24018" class="reference external">bpo-24018</a>.)

New <a href="../library/collections.abc.html#collections.abc.Awaitable" class="reference internal" title="collections.abc.Awaitable"><span class="pre"><code class="sourceCode python">Awaitable</code></span></a>, <a href="../library/collections.abc.html#collections.abc.Coroutine" class="reference internal" title="collections.abc.Coroutine"><span class="pre"><code class="sourceCode python">Coroutine</code></span></a>, <a href="../library/collections.abc.html#collections.abc.AsyncIterator" class="reference internal" title="collections.abc.AsyncIterator"><span class="pre"><code class="sourceCode python">AsyncIterator</code></span></a>, and <a href="../library/collections.abc.html#collections.abc.AsyncIterable" class="reference internal" title="collections.abc.AsyncIterable"><span class="pre"><code class="sourceCode python">AsyncIterable</code></span></a> abstract base classes. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24184" class="reference external">bpo-24184</a>.)

For earlier Python versions, a backport of the new ABCs is available in an external <a href="https://pypi.org/project/backports_abc/" class="extlink-pypi reference external">PyPI package</a>.

</div>

<div id="compileall" class="section">

### compileall<a href="#compileall" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/compileall.html#module-compileall" class="reference internal" title="compileall: Tools for byte-compiling all Python source files in a directory tree."><span class="pre"><code class="sourceCode python">compileall</code></span></a> option, <span class="pre">`-j`</span>` `*<span class="pre">`N`</span>*, allows running *N* workers simultaneously to perform parallel bytecode compilation. The <a href="../library/compileall.html#compileall.compile_dir" class="reference internal" title="compileall.compile_dir"><span class="pre"><code class="sourceCode python">compile_dir()</code></span></a> function has a corresponding <span class="pre">`workers`</span> parameter. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16104" class="reference external">bpo-16104</a>.)

Another new option, <span class="pre">`-r`</span>, allows controlling the maximum recursion level for subdirectories. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19628" class="reference external">bpo-19628</a>.)

The <span class="pre">`-q`</span> command line option can now be specified more than once, in which case all output, including errors, will be suppressed. The corresponding <span class="pre">`quiet`</span> parameter in <a href="../library/compileall.html#compileall.compile_dir" class="reference internal" title="compileall.compile_dir"><span class="pre"><code class="sourceCode python">compile_dir()</code></span></a>, <a href="../library/compileall.html#compileall.compile_file" class="reference internal" title="compileall.compile_file"><span class="pre"><code class="sourceCode python">compile_file()</code></span></a>, and <a href="../library/compileall.html#compileall.compile_path" class="reference internal" title="compileall.compile_path"><span class="pre"><code class="sourceCode python">compile_path()</code></span></a> can now accept an integer value indicating the level of output suppression. (Contributed by Thomas Kluyver in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21338" class="reference external">bpo-21338</a>.)

</div>

<div id="concurrent-futures" class="section">

### concurrent.futures<a href="#concurrent-futures" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/concurrent.futures.html#concurrent.futures.Executor.map" class="reference internal" title="concurrent.futures.Executor.map"><span class="pre"><code class="sourceCode python">Executor.<span class="bu">map</span>()</code></span></a> method now accepts a *chunksize* argument to allow batching of tasks to improve performance when <a href="../library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor" class="reference internal" title="concurrent.futures.ProcessPoolExecutor"><span class="pre"><code class="sourceCode python">ProcessPoolExecutor()</code></span></a> is used. (Contributed by Dan O’Reilly in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11271" class="reference external">bpo-11271</a>.)

The number of workers in the <a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">ThreadPoolExecutor</code></span></a> constructor is optional now. The default value is 5 times the number of CPUs. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21527" class="reference external">bpo-21527</a>.)

</div>

<div id="configparser" class="section">

### configparser<a href="#configparser" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/configparser.html#module-configparser" class="reference internal" title="configparser: Configuration file parser."><span class="pre"><code class="sourceCode python">configparser</code></span></a> now provides a way to customize the conversion of values by specifying a dictionary of converters in the <a href="../library/configparser.html#configparser.ConfigParser" class="reference internal" title="configparser.ConfigParser"><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a> constructor, or by defining them as methods in <span class="pre">`ConfigParser`</span> subclasses. Converters defined in a parser instance are inherited by its section proxies.

Example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import configparser
    >>> conv = {}
    >>> conv['list'] = lambda v: [e.strip() for e in v.split() if e.strip()]
    >>> cfg = configparser.ConfigParser(converters=conv)
    >>> cfg.read_string("""
    ... [s]
    ... list = a b c d e f g
    ... """)
    >>> cfg.get('s', 'list')
    'a b c d e f g'
    >>> cfg.getlist('s', 'list')
    ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    >>> section = cfg['s']
    >>> section.getlist('list')
    ['a', 'b', 'c', 'd', 'e', 'f', 'g']

</div>

</div>

(Contributed by Łukasz Langa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18159" class="reference external">bpo-18159</a>.)

</div>

<div id="contextlib" class="section">

### contextlib<a href="#contextlib" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/contextlib.html#contextlib.redirect_stderr" class="reference internal" title="contextlib.redirect_stderr"><span class="pre"><code class="sourceCode python">redirect_stderr()</code></span></a> <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> (similar to <a href="../library/contextlib.html#contextlib.redirect_stdout" class="reference internal" title="contextlib.redirect_stdout"><span class="pre"><code class="sourceCode python">redirect_stdout()</code></span></a>) makes it easier for utility scripts to handle inflexible APIs that write their output to <a href="../library/sys.html#sys.stderr" class="reference internal" title="sys.stderr"><span class="pre"><code class="sourceCode python">sys.stderr</code></span></a> and don’t provide any options to redirect it:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import contextlib, io, logging
    >>> f = io.StringIO()
    >>> with contextlib.redirect_stderr(f):
    ...     logging.warning('warning')
    ...
    >>> f.getvalue()
    'WARNING:root:warning\n'

</div>

</div>

(Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22389" class="reference external">bpo-22389</a>.)

</div>

<div id="csv" class="section">

### csv<a href="#csv" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/csv.html#csv.csvwriter.writerow" class="reference internal" title="csv.csvwriter.writerow"><span class="pre"><code class="sourceCode python">writerow()</code></span></a> method now supports arbitrary iterables, not just sequences. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23171" class="reference external">bpo-23171</a>.)

</div>

<div id="curses" class="section">

### curses<a href="#curses" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/curses.html#curses.update_lines_cols" class="reference internal" title="curses.update_lines_cols"><span class="pre"><code class="sourceCode python">update_lines_cols()</code></span></a> function updates the <a href="../library/curses.html#curses.LINES" class="reference internal" title="curses.LINES"><span class="pre"><code class="sourceCode python">LINES</code></span></a> and <a href="../library/curses.html#curses.COLS" class="reference internal" title="curses.COLS"><span class="pre"><code class="sourceCode python">COLS</code></span></a> module variables. This is useful for detecting manual screen resizing. (Contributed by Arnon Yaari in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4254" class="reference external">bpo-4254</a>.)

</div>

<div id="dbm" class="section">

### dbm<a href="#dbm" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/dbm.html#dbm.dumb.open" class="reference internal" title="dbm.dumb.open"><span class="pre"><code class="sourceCode python">dumb.<span class="bu">open</span></code></span></a> always creates a new database when the flag has the value <span class="pre">`"n"`</span>. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18039" class="reference external">bpo-18039</a>.)

</div>

<div id="difflib" class="section">

### difflib<a href="#difflib" class="headerlink" title="Link to this heading">¶</a>

The charset of HTML documents generated by <a href="../library/difflib.html#difflib.HtmlDiff.make_file" class="reference internal" title="difflib.HtmlDiff.make_file"><span class="pre"><code class="sourceCode python">HtmlDiff.make_file()</code></span></a> can now be customized by using a new *charset* keyword-only argument. The default charset of HTML document changed from <span class="pre">`"ISO-8859-1"`</span> to <span class="pre">`"utf-8"`</span>. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2052" class="reference external">bpo-2052</a>.)

The <a href="../library/difflib.html#difflib.diff_bytes" class="reference internal" title="difflib.diff_bytes"><span class="pre"><code class="sourceCode python">diff_bytes()</code></span></a> function can now compare lists of byte strings. This fixes a regression from Python 2. (Contributed by Terry J. Reedy and Greg Ward in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17445" class="reference external">bpo-17445</a>.)

</div>

<div id="distutils" class="section">

### distutils<a href="#distutils" class="headerlink" title="Link to this heading">¶</a>

Both the <span class="pre">`build`</span> and <span class="pre">`build_ext`</span> commands now accept a <span class="pre">`-j`</span> option to enable parallel building of extension modules. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5309" class="reference external">bpo-5309</a>.)

The <span class="pre">`distutils`</span> module now supports <span class="pre">`xz`</span> compression, and can be enabled by passing <span class="pre">`xztar`</span> as an argument to <span class="pre">`bdist`</span>` `<span class="pre">`--format`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16314" class="reference external">bpo-16314</a>.)

</div>

<div id="doctest" class="section">

### doctest<a href="#doctest" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/doctest.html#doctest.DocTestSuite" class="reference internal" title="doctest.DocTestSuite"><span class="pre"><code class="sourceCode python">DocTestSuite()</code></span></a> function returns an empty <a href="../library/unittest.html#unittest.TestSuite" class="reference internal" title="unittest.TestSuite"><span class="pre"><code class="sourceCode python">unittest.TestSuite</code></span></a> if *module* contains no docstrings, instead of raising <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a>. (Contributed by Glenn Jones in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15916" class="reference external">bpo-15916</a>.)

</div>

<div id="email" class="section">

### email<a href="#email" class="headerlink" title="Link to this heading">¶</a>

A new policy option <a href="../library/email.policy.html#email.policy.Policy.mangle_from_" class="reference internal" title="email.policy.Policy.mangle_from_"><span class="pre"><code class="sourceCode python">Policy.mangle_from_</code></span></a> controls whether or not lines that start with <span class="pre">`"From`</span>` `<span class="pre">`"`</span> in email bodies are prefixed with a <span class="pre">`">"`</span> character by generators. The default is <span class="pre">`True`</span> for <a href="../library/email.policy.html#email.policy.compat32" class="reference internal" title="email.policy.compat32"><span class="pre"><code class="sourceCode python">compat32</code></span></a> and <span class="pre">`False`</span> for all other policies. (Contributed by Milan Oberkirch in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20098" class="reference external">bpo-20098</a>.)

A new <a href="../library/email.compat32-message.html#email.message.Message.get_content_disposition" class="reference internal" title="email.message.Message.get_content_disposition"><span class="pre"><code class="sourceCode python">Message.get_content_disposition()</code></span></a> method provides easy access to a canonical value for the *Content-Disposition* header. (Contributed by Abhilash Raj in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21083" class="reference external">bpo-21083</a>.)

A new policy option <a href="../library/email.policy.html#email.policy.EmailPolicy.utf8" class="reference internal" title="email.policy.EmailPolicy.utf8"><span class="pre"><code class="sourceCode python">EmailPolicy.utf8</code></span></a> can be set to <span class="pre">`True`</span> to encode email headers using the UTF-8 charset instead of using encoded words. This allows <span class="pre">`Messages`</span> to be formatted according to <span id="index-31" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc6532.html" class="rfc reference external"><strong>RFC 6532</strong></a> and used with an SMTP server that supports the <span id="index-32" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc6531.html" class="rfc reference external"><strong>RFC 6531</strong></a> <span class="pre">`SMTPUTF8`</span> extension. (Contributed by R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24211" class="reference external">bpo-24211</a>.)

The <a href="../library/email.mime.html#email.mime.text.MIMEText" class="reference internal" title="email.mime.text.MIMEText"><span class="pre"><code class="sourceCode python">mime.text.MIMEText</code></span></a> constructor now accepts a <a href="../library/email.charset.html#email.charset.Charset" class="reference internal" title="email.charset.Charset"><span class="pre"><code class="sourceCode python">charset.Charset</code></span></a> instance. (Contributed by Claude Paroz and Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16324" class="reference external">bpo-16324</a>.)

</div>

<div id="enum" class="section">

### enum<a href="#enum" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/enum.html#enum.Enum" class="reference internal" title="enum.Enum"><span class="pre"><code class="sourceCode python">Enum</code></span></a> callable has a new parameter *start* to specify the initial number of enum values if only *names* are provided:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> Animal = enum.Enum('Animal', 'cat dog', start=10)
    >>> Animal.cat
    <Animal.cat: 10>
    >>> Animal.dog
    <Animal.dog: 11>

</div>

</div>

(Contributed by Ethan Furman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21706" class="reference external">bpo-21706</a>.)

</div>

<div id="faulthandler" class="section">

### faulthandler<a href="#faulthandler" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/faulthandler.html#faulthandler.enable" class="reference internal" title="faulthandler.enable"><span class="pre"><code class="sourceCode python">enable()</code></span></a>, <a href="../library/faulthandler.html#faulthandler.register" class="reference internal" title="faulthandler.register"><span class="pre"><code class="sourceCode python">register()</code></span></a>, <a href="../library/faulthandler.html#faulthandler.dump_traceback" class="reference internal" title="faulthandler.dump_traceback"><span class="pre"><code class="sourceCode python">dump_traceback()</code></span></a> and <a href="../library/faulthandler.html#faulthandler.dump_traceback_later" class="reference internal" title="faulthandler.dump_traceback_later"><span class="pre"><code class="sourceCode python">dump_traceback_later()</code></span></a> functions now accept file descriptors in addition to file-like objects. (Contributed by Wei Wu in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23566" class="reference external">bpo-23566</a>.)

</div>

<div id="functools" class="section">

### functools<a href="#functools" class="headerlink" title="Link to this heading">¶</a>

Most of the <a href="../library/functools.html#functools.lru_cache" class="reference internal" title="functools.lru_cache"><span class="pre"><code class="sourceCode python">lru_cache()</code></span></a> machinery is now implemented in C, making it significantly faster. (Contributed by Matt Joiner, Alexey Kachayev, and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14373" class="reference external">bpo-14373</a>.)

</div>

<div id="glob" class="section">

### glob<a href="#glob" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/glob.html#glob.iglob" class="reference internal" title="glob.iglob"><span class="pre"><code class="sourceCode python">iglob()</code></span></a> and <a href="../library/glob.html#glob.glob" class="reference internal" title="glob.glob"><span class="pre"><code class="sourceCode python">glob()</code></span></a> functions now support recursive search in subdirectories, using the <span class="pre">`"**"`</span> pattern. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13968" class="reference external">bpo-13968</a>.)

</div>

<div id="gzip" class="section">

### gzip<a href="#gzip" class="headerlink" title="Link to this heading">¶</a>

The *mode* argument of the <a href="../library/gzip.html#gzip.GzipFile" class="reference internal" title="gzip.GzipFile"><span class="pre"><code class="sourceCode python">GzipFile</code></span></a> constructor now accepts <span class="pre">`"x"`</span> to request exclusive creation. (Contributed by Tim Heaney in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19222" class="reference external">bpo-19222</a>.)

</div>

<div id="heapq" class="section">

### heapq<a href="#heapq" class="headerlink" title="Link to this heading">¶</a>

Element comparison in <a href="../library/heapq.html#heapq.merge" class="reference internal" title="heapq.merge"><span class="pre"><code class="sourceCode python">merge()</code></span></a> can now be customized by passing a <a href="../glossary.html#term-key-function" class="reference internal"><span class="xref std std-term">key function</span></a> in a new optional *key* keyword argument, and a new optional *reverse* keyword argument can be used to reverse element comparison:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import heapq
    >>> a = ['9', '777', '55555']
    >>> b = ['88', '6666']
    >>> list(heapq.merge(a, b, key=len))
    ['9', '88', '777', '6666', '55555']
    >>> list(heapq.merge(reversed(a), reversed(b), key=len, reverse=True))
    ['55555', '6666', '777', '88', '9']

</div>

</div>

(Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13742" class="reference external">bpo-13742</a>.)

</div>

<div id="http" class="section">

### http<a href="#http" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/http.html#http.HTTPStatus" class="reference internal" title="http.HTTPStatus"><span class="pre"><code class="sourceCode python">HTTPStatus</code></span></a> enum that defines a set of HTTP status codes, reason phrases and long descriptions written in English. (Contributed by Demian Brecht in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21793" class="reference external">bpo-21793</a>.)

</div>

<div id="http-client" class="section">

### http.client<a href="#http-client" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/http.client.html#http.client.HTTPConnection.getresponse" class="reference internal" title="http.client.HTTPConnection.getresponse"><span class="pre"><code class="sourceCode python">HTTPConnection.getresponse()</code></span></a> now raises a <a href="../library/http.client.html#http.client.RemoteDisconnected" class="reference internal" title="http.client.RemoteDisconnected"><span class="pre"><code class="sourceCode python">RemoteDisconnected</code></span></a> exception when a remote server connection is closed unexpectedly. Additionally, if a <a href="../library/exceptions.html#ConnectionError" class="reference internal" title="ConnectionError"><span class="pre"><code class="sourceCode python"><span class="pp">ConnectionError</span></code></span></a> (of which <span class="pre">`RemoteDisconnected`</span> is a subclass) is raised, the client socket is now closed automatically, and will reconnect on the next request:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import http.client
    conn = http.client.HTTPConnection('www.python.org')
    for retries in range(3):
        try:
            conn.request('GET', '/')
            resp = conn.getresponse()
        except http.client.RemoteDisconnected:
            pass

</div>

</div>

(Contributed by Martin Panter in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3566" class="reference external">bpo-3566</a>.)

</div>

<div id="idlelib-and-idle" class="section">

### idlelib and IDLE<a href="#idlelib-and-idle" class="headerlink" title="Link to this heading">¶</a>

Since idlelib implements the IDLE shell and editor and is not intended for import by other programs, it gets improvements with every release. See <span class="pre">`Lib/idlelib/NEWS.txt`</span> for a cumulative list of changes since 3.4.0, as well as changes made in future 3.5.x releases. This file is also available from the IDLE <span class="menuselection">Help ‣ About IDLE</span> dialog.

</div>

<div id="imaplib" class="section">

### imaplib<a href="#imaplib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/imaplib.html#imaplib.IMAP4" class="reference internal" title="imaplib.IMAP4"><span class="pre"><code class="sourceCode python">IMAP4</code></span></a> class now supports the <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> protocol. When used in a <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement, the IMAP4 <span class="pre">`LOGOUT`</span> command will be called automatically at the end of the block. (Contributed by Tarek Ziadé and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4972" class="reference external">bpo-4972</a>.)

The <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a> module now supports <span id="index-33" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc5161.html" class="rfc reference external"><strong>RFC 5161</strong></a> (ENABLE Extension) and <span id="index-34" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc6855.html" class="rfc reference external"><strong>RFC 6855</strong></a> (UTF-8 Support) via the <a href="../library/imaplib.html#imaplib.IMAP4.enable" class="reference internal" title="imaplib.IMAP4.enable"><span class="pre"><code class="sourceCode python">IMAP4.enable()</code></span></a> method. A new <a href="../library/imaplib.html#imaplib.IMAP4.utf8_enabled" class="reference internal" title="imaplib.IMAP4.utf8_enabled"><span class="pre"><code class="sourceCode python">IMAP4.utf8_enabled</code></span></a> attribute tracks whether or not <span id="index-35" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc6855.html" class="rfc reference external"><strong>RFC 6855</strong></a> support is enabled. (Contributed by Milan Oberkirch, R. David Murray, and Maciej Szulik in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21800" class="reference external">bpo-21800</a>.)

The <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a> module now automatically encodes non-ASCII string usernames and passwords using UTF-8, as recommended by the RFCs. (Contributed by Milan Oberkirch in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21800" class="reference external">bpo-21800</a>.)

</div>

<div id="imghdr" class="section">

### imghdr<a href="#imghdr" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`what()`</span> function now recognizes the <a href="https://www.openexr.com" class="reference external">OpenEXR</a> format (contributed by Martin Vignali and Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20295" class="reference external">bpo-20295</a>), and the <a href="https://en.wikipedia.org/wiki/WebP" class="reference external">WebP</a> format (contributed by Fabrice Aneche and Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20197" class="reference external">bpo-20197</a>.)

</div>

<div id="importlib" class="section">

### importlib<a href="#importlib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/importlib.html#importlib.util.LazyLoader" class="reference internal" title="importlib.util.LazyLoader"><span class="pre"><code class="sourceCode python">util.LazyLoader</code></span></a> class allows for lazy loading of modules in applications where startup time is important. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17621" class="reference external">bpo-17621</a>.)

The <a href="../library/importlib.html#importlib.abc.InspectLoader.source_to_code" class="reference internal" title="importlib.abc.InspectLoader.source_to_code"><span class="pre"><code class="sourceCode python">abc.InspectLoader.source_to_code()</code></span></a> method is now a static method. This makes it easier to initialize a module object with code compiled from a string by running <span class="pre">`exec(code,`</span>` `<span class="pre">`module.__dict__)`</span>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21156" class="reference external">bpo-21156</a>.)

The new <a href="../library/importlib.html#importlib.util.module_from_spec" class="reference internal" title="importlib.util.module_from_spec"><span class="pre"><code class="sourceCode python">util.module_from_spec()</code></span></a> function is now the preferred way to create a new module. As opposed to creating a <a href="../library/types.html#types.ModuleType" class="reference internal" title="types.ModuleType"><span class="pre"><code class="sourceCode python">types.ModuleType</code></span></a> instance directly, this new function will set the various import-controlled attributes based on the passed-in spec object. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20383" class="reference external">bpo-20383</a>.)

</div>

<div id="inspect" class="section">

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

Both the <a href="../library/inspect.html#inspect.Signature" class="reference internal" title="inspect.Signature"><span class="pre"><code class="sourceCode python">Signature</code></span></a> and <a href="../library/inspect.html#inspect.Parameter" class="reference internal" title="inspect.Parameter"><span class="pre"><code class="sourceCode python">Parameter</code></span></a> classes are now picklable and hashable. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20726" class="reference external">bpo-20726</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20334" class="reference external">bpo-20334</a>.)

A new <a href="../library/inspect.html#inspect.BoundArguments.apply_defaults" class="reference internal" title="inspect.BoundArguments.apply_defaults"><span class="pre"><code class="sourceCode python">BoundArguments.apply_defaults()</code></span></a> method provides a way to set default values for missing arguments:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> def foo(a, b='ham', *args): pass
    >>> ba = inspect.signature(foo).bind('spam')
    >>> ba.apply_defaults()
    >>> ba.arguments
    OrderedDict([('a', 'spam'), ('b', 'ham'), ('args', ())])

</div>

</div>

(Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24190" class="reference external">bpo-24190</a>.)

A new class method <a href="../library/inspect.html#inspect.Signature.from_callable" class="reference internal" title="inspect.Signature.from_callable"><span class="pre"><code class="sourceCode python">Signature.from_callable()</code></span></a> makes subclassing of <a href="../library/inspect.html#inspect.Signature" class="reference internal" title="inspect.Signature"><span class="pre"><code class="sourceCode python">Signature</code></span></a> easier. (Contributed by Yury Selivanov and Eric Snow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17373" class="reference external">bpo-17373</a>.)

The <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">signature()</code></span></a> function now accepts a *follow_wrapped* optional keyword argument, which, when set to <span class="pre">`False`</span>, disables automatic following of <span class="pre">`__wrapped__`</span> links. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20691" class="reference external">bpo-20691</a>.)

A set of new functions to inspect <a href="../glossary.html#term-coroutine-function" class="reference internal"><span class="xref std std-term">coroutine functions</span></a> and <a href="../glossary.html#term-coroutine" class="reference internal"><span class="xref std std-term">coroutine objects</span></a> has been added: <a href="../library/inspect.html#inspect.iscoroutine" class="reference internal" title="inspect.iscoroutine"><span class="pre"><code class="sourceCode python">iscoroutine()</code></span></a>, <a href="../library/inspect.html#inspect.iscoroutinefunction" class="reference internal" title="inspect.iscoroutinefunction"><span class="pre"><code class="sourceCode python">iscoroutinefunction()</code></span></a>, <a href="../library/inspect.html#inspect.isawaitable" class="reference internal" title="inspect.isawaitable"><span class="pre"><code class="sourceCode python">isawaitable()</code></span></a>, <a href="../library/inspect.html#inspect.getcoroutinelocals" class="reference internal" title="inspect.getcoroutinelocals"><span class="pre"><code class="sourceCode python">getcoroutinelocals()</code></span></a>, and <a href="../library/inspect.html#inspect.getcoroutinestate" class="reference internal" title="inspect.getcoroutinestate"><span class="pre"><code class="sourceCode python">getcoroutinestate()</code></span></a>. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24017" class="reference external">bpo-24017</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24400" class="reference external">bpo-24400</a>.)

The <a href="../library/inspect.html#inspect.stack" class="reference internal" title="inspect.stack"><span class="pre"><code class="sourceCode python">stack()</code></span></a>, <a href="../library/inspect.html#inspect.trace" class="reference internal" title="inspect.trace"><span class="pre"><code class="sourceCode python">trace()</code></span></a>, <a href="../library/inspect.html#inspect.getouterframes" class="reference internal" title="inspect.getouterframes"><span class="pre"><code class="sourceCode python">getouterframes()</code></span></a>, and <a href="../library/inspect.html#inspect.getinnerframes" class="reference internal" title="inspect.getinnerframes"><span class="pre"><code class="sourceCode python">getinnerframes()</code></span></a> functions now return a list of named tuples. (Contributed by Daniel Shahaf in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16808" class="reference external">bpo-16808</a>.)

</div>

<div id="io" class="section">

### io<a href="#io" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/io.html#io.BufferedIOBase.readinto1" class="reference internal" title="io.BufferedIOBase.readinto1"><span class="pre"><code class="sourceCode python">BufferedIOBase.readinto1()</code></span></a> method, that uses at most one call to the underlying raw stream’s <a href="../library/io.html#io.RawIOBase.read" class="reference internal" title="io.RawIOBase.read"><span class="pre"><code class="sourceCode python">RawIOBase.read()</code></span></a> or <a href="../library/io.html#io.RawIOBase.readinto" class="reference internal" title="io.RawIOBase.readinto"><span class="pre"><code class="sourceCode python">RawIOBase.readinto()</code></span></a> methods. (Contributed by Nikolaus Rath in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20578" class="reference external">bpo-20578</a>.)

</div>

<div id="ipaddress" class="section">

### ipaddress<a href="#ipaddress" class="headerlink" title="Link to this heading">¶</a>

Both the <a href="../library/ipaddress.html#ipaddress.IPv4Network" class="reference internal" title="ipaddress.IPv4Network"><span class="pre"><code class="sourceCode python">IPv4Network</code></span></a> and <a href="../library/ipaddress.html#ipaddress.IPv6Network" class="reference internal" title="ipaddress.IPv6Network"><span class="pre"><code class="sourceCode python">IPv6Network</code></span></a> classes now accept an <span class="pre">`(address,`</span>` `<span class="pre">`netmask)`</span> tuple argument, so as to easily construct network objects from existing addresses:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import ipaddress
    >>> ipaddress.IPv4Network(('127.0.0.0', 8))
    IPv4Network('127.0.0.0/8')
    >>> ipaddress.IPv4Network(('127.0.0.0', '255.0.0.0'))
    IPv4Network('127.0.0.0/8')

</div>

</div>

(Contributed by Peter Moody and Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16531" class="reference external">bpo-16531</a>.)

A new <a href="../library/ipaddress.html#ipaddress.IPv4Address.reverse_pointer" class="reference internal" title="ipaddress.IPv4Address.reverse_pointer"><span class="pre"><code class="sourceCode python">reverse_pointer</code></span></a> attribute for the <a href="../library/ipaddress.html#ipaddress.IPv4Address" class="reference internal" title="ipaddress.IPv4Address"><span class="pre"><code class="sourceCode python">IPv4Address</code></span></a> and <a href="../library/ipaddress.html#ipaddress.IPv6Address" class="reference internal" title="ipaddress.IPv6Address"><span class="pre"><code class="sourceCode python">IPv6Address</code></span></a> classes returns the name of the reverse DNS PTR record:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import ipaddress
    >>> addr = ipaddress.IPv4Address('127.0.0.1')
    >>> addr.reverse_pointer
    '1.0.0.127.in-addr.arpa'
    >>> addr6 = ipaddress.IPv6Address('::1')
    >>> addr6.reverse_pointer
    '1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa'

</div>

</div>

(Contributed by Leon Weber in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20480" class="reference external">bpo-20480</a>.)

</div>

<div id="json" class="section">

### json<a href="#json" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/json.html#module-json.tool" class="reference internal" title="json.tool: A command line to validate and pretty-print JSON."><span class="pre"><code class="sourceCode python">json.tool</code></span></a> command line interface now preserves the order of keys in JSON objects passed in input. The new <span class="pre">`--sort-keys`</span> option can be used to sort the keys alphabetically. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21650" class="reference external">bpo-21650</a>.)

JSON decoder now raises <a href="../library/json.html#json.JSONDecodeError" class="reference internal" title="json.JSONDecodeError"><span class="pre"><code class="sourceCode python">JSONDecodeError</code></span></a> instead of <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> to provide better context information about the error. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19361" class="reference external">bpo-19361</a>.)

</div>

<div id="linecache" class="section">

### linecache<a href="#linecache" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/linecache.html#linecache.lazycache" class="reference internal" title="linecache.lazycache"><span class="pre"><code class="sourceCode python">lazycache()</code></span></a> function can be used to capture information about a non-file-based module to permit getting its lines later via <a href="../library/linecache.html#linecache.getline" class="reference internal" title="linecache.getline"><span class="pre"><code class="sourceCode python">getline()</code></span></a>. This avoids doing I/O until a line is actually needed, without having to carry the module globals around indefinitely. (Contributed by Robert Collins in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17911" class="reference external">bpo-17911</a>.)

</div>

<div id="locale" class="section">

### locale<a href="#locale" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/locale.html#locale.delocalize" class="reference internal" title="locale.delocalize"><span class="pre"><code class="sourceCode python">delocalize()</code></span></a> function can be used to convert a string into a normalized number string, taking the <span class="pre">`LC_NUMERIC`</span> settings into account:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import locale
    >>> locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')
    'de_DE.UTF-8'
    >>> locale.delocalize('1.234,56')
    '1234.56'
    >>> locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')
    'en_US.UTF-8'
    >>> locale.delocalize('1,234.56')
    '1234.56'

</div>

</div>

(Contributed by Cédric Krier in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13918" class="reference external">bpo-13918</a>.)

</div>

<div id="logging" class="section">

### logging<a href="#logging" class="headerlink" title="Link to this heading">¶</a>

All logging methods (<a href="../library/logging.html#logging.Logger" class="reference internal" title="logging.Logger"><span class="pre"><code class="sourceCode python">Logger</code></span></a> <a href="../library/logging.html#logging.Logger.log" class="reference internal" title="logging.Logger.log"><span class="pre"><code class="sourceCode python">log()</code></span></a>, <a href="../library/logging.html#logging.Logger.exception" class="reference internal" title="logging.Logger.exception"><span class="pre"><code class="sourceCode python">exception()</code></span></a>, <a href="../library/logging.html#logging.Logger.critical" class="reference internal" title="logging.Logger.critical"><span class="pre"><code class="sourceCode python">critical()</code></span></a>, <a href="../library/logging.html#logging.Logger.debug" class="reference internal" title="logging.Logger.debug"><span class="pre"><code class="sourceCode python">debug()</code></span></a>, etc.), now accept exception instances as an *exc_info* argument, in addition to boolean values and exception tuples:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import logging
    >>> try:
    ...     1/0
    ... except ZeroDivisionError as ex:
    ...     logging.error('exception', exc_info=ex)
    ERROR:root:exception

</div>

</div>

(Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20537" class="reference external">bpo-20537</a>.)

The <a href="../library/logging.handlers.html#logging.handlers.HTTPHandler" class="reference internal" title="logging.handlers.HTTPHandler"><span class="pre"><code class="sourceCode python">handlers.HTTPHandler</code></span></a> class now accepts an optional <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a> instance to configure SSL settings used in an HTTP connection. (Contributed by Alex Gaynor in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22788" class="reference external">bpo-22788</a>.)

The <a href="../library/logging.handlers.html#logging.handlers.QueueListener" class="reference internal" title="logging.handlers.QueueListener"><span class="pre"><code class="sourceCode python">handlers.QueueListener</code></span></a> class now takes a *respect_handler_level* keyword argument which, if set to <span class="pre">`True`</span>, will pass messages to handlers taking handler levels into account. (Contributed by Vinay Sajip.)

</div>

<div id="lzma" class="section">

### lzma<a href="#lzma" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/lzma.html#lzma.LZMADecompressor.decompress" class="reference internal" title="lzma.LZMADecompressor.decompress"><span class="pre"><code class="sourceCode python">LZMADecompressor.decompress()</code></span></a> method now accepts an optional *max_length* argument to limit the maximum size of decompressed data. (Contributed by Martin Panter in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15955" class="reference external">bpo-15955</a>.)

</div>

<div id="math" class="section">

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

Two new constants have been added to the <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> module: <a href="../library/math.html#math.inf" class="reference internal" title="math.inf"><span class="pre"><code class="sourceCode python">inf</code></span></a> and <a href="../library/math.html#math.nan" class="reference internal" title="math.nan"><span class="pre"><code class="sourceCode python">nan</code></span></a>. (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23185" class="reference external">bpo-23185</a>.)

A new function <a href="../library/math.html#math.isclose" class="reference internal" title="math.isclose"><span class="pre"><code class="sourceCode python">isclose()</code></span></a> provides a way to test for approximate equality. (Contributed by Chris Barker and Tal Einat in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24270" class="reference external">bpo-24270</a>.)

A new <a href="../library/math.html#math.gcd" class="reference internal" title="math.gcd"><span class="pre"><code class="sourceCode python">gcd()</code></span></a> function has been added. The <span class="pre">`fractions.gcd()`</span> function is now deprecated. (Contributed by Mark Dickinson and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22486" class="reference external">bpo-22486</a>.)

</div>

<div id="multiprocessing" class="section">

### multiprocessing<a href="#multiprocessing" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/multiprocessing.html#multiprocessing.sharedctypes.synchronized" class="reference internal" title="multiprocessing.sharedctypes.synchronized"><span class="pre"><code class="sourceCode python">sharedctypes.synchronized()</code></span></a> objects now support the <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> protocol. (Contributed by Charles-François Natali in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21565" class="reference external">bpo-21565</a>.)

</div>

<div id="operator" class="section">

### operator<a href="#operator" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/operator.html#operator.attrgetter" class="reference internal" title="operator.attrgetter"><span class="pre"><code class="sourceCode python">attrgetter()</code></span></a>, <a href="../library/operator.html#operator.itemgetter" class="reference internal" title="operator.itemgetter"><span class="pre"><code class="sourceCode python">itemgetter()</code></span></a>, and <a href="../library/operator.html#operator.methodcaller" class="reference internal" title="operator.methodcaller"><span class="pre"><code class="sourceCode python">methodcaller()</code></span></a> objects now support pickling. (Contributed by Josh Rosenberg and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22955" class="reference external">bpo-22955</a>.)

New <a href="../library/operator.html#operator.matmul" class="reference internal" title="operator.matmul"><span class="pre"><code class="sourceCode python">matmul()</code></span></a> and <a href="../library/operator.html#operator.imatmul" class="reference internal" title="operator.imatmul"><span class="pre"><code class="sourceCode python">imatmul()</code></span></a> functions to perform matrix multiplication. (Contributed by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21176" class="reference external">bpo-21176</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">scandir()</code></span></a> function returning an iterator of <a href="../library/os.html#os.DirEntry" class="reference internal" title="os.DirEntry"><span class="pre"><code class="sourceCode python">DirEntry</code></span></a> objects has been added. If possible, <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">scandir()</code></span></a> extracts file attributes while scanning a directory, removing the need to perform subsequent system calls to determine file type or attributes, which may significantly improve performance. (Contributed by Ben Hoyt with the help of Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22524" class="reference external">bpo-22524</a>.)

On Windows, a new <a href="../library/os.html#os.stat_result.st_file_attributes" class="reference internal" title="os.stat_result.st_file_attributes"><span class="pre"><code class="sourceCode python">stat_result.st_file_attributes</code></span></a> attribute is now available. It corresponds to the <span class="pre">`dwFileAttributes`</span> member of the <span class="pre">`BY_HANDLE_FILE_INFORMATION`</span> structure returned by <span class="pre">`GetFileInformationByHandle()`</span>. (Contributed by Ben Hoyt in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21719" class="reference external">bpo-21719</a>.)

The <a href="../library/os.html#os.urandom" class="reference internal" title="os.urandom"><span class="pre"><code class="sourceCode python">urandom()</code></span></a> function now uses the <span class="pre">`getrandom()`</span> syscall on Linux 3.17 or newer, and <span class="pre">`getentropy()`</span> on OpenBSD 5.6 and newer, removing the need to use <span class="pre">`/dev/urandom`</span> and avoiding failures due to potential file descriptor exhaustion. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22181" class="reference external">bpo-22181</a>.)

New <a href="../library/os.html#os.get_blocking" class="reference internal" title="os.get_blocking"><span class="pre"><code class="sourceCode python">get_blocking()</code></span></a> and <a href="../library/os.html#os.set_blocking" class="reference internal" title="os.set_blocking"><span class="pre"><code class="sourceCode python">set_blocking()</code></span></a> functions allow getting and setting a file descriptor’s blocking mode (<a href="../library/os.html#os.O_NONBLOCK" class="reference internal" title="os.O_NONBLOCK"><span class="pre"><code class="sourceCode python">O_NONBLOCK</code></span></a>.) (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22054" class="reference external">bpo-22054</a>.)

The <a href="../library/os.html#os.truncate" class="reference internal" title="os.truncate"><span class="pre"><code class="sourceCode python">truncate()</code></span></a> and <a href="../library/os.html#os.ftruncate" class="reference internal" title="os.ftruncate"><span class="pre"><code class="sourceCode python">ftruncate()</code></span></a> functions are now supported on Windows. (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23668" class="reference external">bpo-23668</a>.)

There is a new <a href="../library/os.path.html#os.path.commonpath" class="reference internal" title="os.path.commonpath"><span class="pre"><code class="sourceCode python">os.path.commonpath()</code></span></a> function returning the longest common sub-path of each passed pathname. Unlike the <a href="../library/os.path.html#os.path.commonprefix" class="reference internal" title="os.path.commonprefix"><span class="pre"><code class="sourceCode python">os.path.commonprefix()</code></span></a> function, it always returns a valid path:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> os.path.commonprefix(['/usr/lib', '/usr/local/lib'])
    '/usr/l'

    >>> os.path.commonpath(['/usr/lib', '/usr/local/lib'])
    '/usr'

</div>

</div>

(Contributed by Rafik Draoui and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10395" class="reference external">bpo-10395</a>.)

</div>

<div id="pathlib" class="section">

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/pathlib.html#pathlib.Path.samefile" class="reference internal" title="pathlib.Path.samefile"><span class="pre"><code class="sourceCode python">Path.samefile()</code></span></a> method can be used to check whether the path points to the same file as another path, which can be either another <a href="../library/pathlib.html#pathlib.Path" class="reference internal" title="pathlib.Path"><span class="pre"><code class="sourceCode python">Path</code></span></a> object, or a string:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import pathlib
    >>> p1 = pathlib.Path('/etc/hosts')
    >>> p2 = pathlib.Path('/etc/../etc/hosts')
    >>> p1.samefile(p2)
    True

</div>

</div>

(Contributed by Vajrasky Kok and Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19775" class="reference external">bpo-19775</a>.)

The <a href="../library/pathlib.html#pathlib.Path.mkdir" class="reference internal" title="pathlib.Path.mkdir"><span class="pre"><code class="sourceCode python">Path.mkdir()</code></span></a> method now accepts a new optional *exist_ok* argument to match <span class="pre">`mkdir`</span>` `<span class="pre">`-p`</span> and <a href="../library/os.html#os.makedirs" class="reference internal" title="os.makedirs"><span class="pre"><code class="sourceCode python">os.makedirs()</code></span></a> functionality. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21539" class="reference external">bpo-21539</a>.)

There is a new <a href="../library/pathlib.html#pathlib.Path.expanduser" class="reference internal" title="pathlib.Path.expanduser"><span class="pre"><code class="sourceCode python">Path.expanduser()</code></span></a> method to expand <span class="pre">`~`</span> and <span class="pre">`~user`</span> prefixes. (Contributed by Serhiy Storchaka and Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19776" class="reference external">bpo-19776</a>.)

A new <a href="../library/pathlib.html#pathlib.Path.home" class="reference internal" title="pathlib.Path.home"><span class="pre"><code class="sourceCode python">Path.home()</code></span></a> class method can be used to get a <a href="../library/pathlib.html#pathlib.Path" class="reference internal" title="pathlib.Path"><span class="pre"><code class="sourceCode python">Path</code></span></a> instance representing the user’s home directory. (Contributed by Victor Salgado and Mayank Tripathi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19777" class="reference external">bpo-19777</a>.)

New <a href="../library/pathlib.html#pathlib.Path.write_text" class="reference internal" title="pathlib.Path.write_text"><span class="pre"><code class="sourceCode python">Path.write_text()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.read_text" class="reference internal" title="pathlib.Path.read_text"><span class="pre"><code class="sourceCode python">Path.read_text()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.write_bytes" class="reference internal" title="pathlib.Path.write_bytes"><span class="pre"><code class="sourceCode python">Path.write_bytes()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.read_bytes" class="reference internal" title="pathlib.Path.read_bytes"><span class="pre"><code class="sourceCode python">Path.read_bytes()</code></span></a> methods to simplify read/write operations on files.

The following code snippet will create or rewrite existing file <span class="pre">`~/spam42`</span>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import pathlib
    >>> p = pathlib.Path('~/spam42')
    >>> p.expanduser().write_text('ham')
    3

</div>

</div>

(Contributed by Christopher Welborn in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20218" class="reference external">bpo-20218</a>.)

</div>

<div id="pickle" class="section">

### pickle<a href="#pickle" class="headerlink" title="Link to this heading">¶</a>

Nested objects, such as unbound methods or nested classes, can now be pickled using <a href="../library/pickle.html#pickle-protocols" class="reference internal"><span class="std std-ref">pickle protocols</span></a> older than protocol version 4. Protocol version 4 already supports these cases. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23611" class="reference external">bpo-23611</a>.)

</div>

<div id="poplib" class="section">

### poplib<a href="#poplib" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/poplib.html#poplib.POP3.utf8" class="reference internal" title="poplib.POP3.utf8"><span class="pre"><code class="sourceCode python">POP3.utf8()</code></span></a> command enables <span id="index-36" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc6856.html" class="rfc reference external"><strong>RFC 6856</strong></a> (Internationalized Email) support, if a POP server supports it. (Contributed by Milan OberKirch in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21804" class="reference external">bpo-21804</a>.)

</div>

<div id="re" class="section">

### re<a href="#re" class="headerlink" title="Link to this heading">¶</a>

References and conditional references to groups with fixed length are now allowed in lookbehind assertions:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import re
    >>> pat = re.compile(r'(a|b).(?<=\1)c')
    >>> pat.match('aac')
    <_sre.SRE_Match object; span=(0, 3), match='aac'>
    >>> pat.match('bbc')
    <_sre.SRE_Match object; span=(0, 3), match='bbc'>

</div>

</div>

(Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9179" class="reference external">bpo-9179</a>.)

The number of capturing groups in regular expressions is no longer limited to 100. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22437" class="reference external">bpo-22437</a>.)

The <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">sub()</code></span></a> and <a href="../library/re.html#re.subn" class="reference internal" title="re.subn"><span class="pre"><code class="sourceCode python">subn()</code></span></a> functions now replace unmatched groups with empty strings instead of raising an exception. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1519638" class="reference external">bpo-1519638</a>.)

The <a href="../library/re.html#re.PatternError" class="reference internal" title="re.PatternError"><span class="pre"><code class="sourceCode python">re.error</code></span></a> exceptions have new attributes, <a href="../library/re.html#re.PatternError.msg" class="reference internal" title="re.PatternError.msg"><span class="pre"><code class="sourceCode python">msg</code></span></a>, <a href="../library/re.html#re.PatternError.pattern" class="reference internal" title="re.PatternError.pattern"><span class="pre"><code class="sourceCode python">pattern</code></span></a>, <a href="../library/re.html#re.PatternError.pos" class="reference internal" title="re.PatternError.pos"><span class="pre"><code class="sourceCode python">pos</code></span></a>, <a href="../library/re.html#re.PatternError.lineno" class="reference internal" title="re.PatternError.lineno"><span class="pre"><code class="sourceCode python">lineno</code></span></a>, and <a href="../library/re.html#re.PatternError.colno" class="reference internal" title="re.PatternError.colno"><span class="pre"><code class="sourceCode python">colno</code></span></a>, that provide better context information about the error:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> re.compile("""
    ...     (?x)
    ...     .++
    ... """)
    Traceback (most recent call last):
       ...
    sre_constants.error: multiple repeat at position 16 (line 3, column 7)

</div>

</div>

(Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22578" class="reference external">bpo-22578</a>.)

</div>

<div id="readline" class="section">

### readline<a href="#readline" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/readline.html#readline.append_history_file" class="reference internal" title="readline.append_history_file"><span class="pre"><code class="sourceCode python">append_history_file()</code></span></a> function can be used to append the specified number of trailing elements in history to the given file. (Contributed by Bruno Cauet in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22940" class="reference external">bpo-22940</a>.)

</div>

<div id="selectors" class="section">

### selectors<a href="#selectors" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/selectors.html#selectors.DevpollSelector" class="reference internal" title="selectors.DevpollSelector"><span class="pre"><code class="sourceCode python">DevpollSelector</code></span></a> supports efficient <span class="pre">`/dev/poll`</span> polling on Solaris. (Contributed by Giampaolo Rodola’ in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18931" class="reference external">bpo-18931</a>.)

</div>

<div id="shutil" class="section">

### shutil<a href="#shutil" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/shutil.html#shutil.move" class="reference internal" title="shutil.move"><span class="pre"><code class="sourceCode python">move()</code></span></a> function now accepts a *copy_function* argument, allowing, for example, the <a href="../library/shutil.html#shutil.copy" class="reference internal" title="shutil.copy"><span class="pre"><code class="sourceCode python">copy()</code></span></a> function to be used instead of the default <a href="../library/shutil.html#shutil.copy2" class="reference internal" title="shutil.copy2"><span class="pre"><code class="sourceCode python">copy2()</code></span></a> if there is a need to ignore file metadata when moving. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19840" class="reference external">bpo-19840</a>.)

The <a href="../library/shutil.html#shutil.make_archive" class="reference internal" title="shutil.make_archive"><span class="pre"><code class="sourceCode python">make_archive()</code></span></a> function now supports the *xztar* format. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5411" class="reference external">bpo-5411</a>.)

</div>

<div id="signal" class="section">

### signal<a href="#signal" class="headerlink" title="Link to this heading">¶</a>

On Windows, the <a href="../library/signal.html#signal.set_wakeup_fd" class="reference internal" title="signal.set_wakeup_fd"><span class="pre"><code class="sourceCode python">set_wakeup_fd()</code></span></a> function now also supports socket handles. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22018" class="reference external">bpo-22018</a>.)

Various <span class="pre">`SIG*`</span> constants in the <a href="../library/signal.html#module-signal" class="reference internal" title="signal: Set handlers for asynchronous events."><span class="pre"><code class="sourceCode python">signal</code></span></a> module have been converted into <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">Enums</code></span></a>. This allows meaningful names to be printed during debugging, instead of integer “magic numbers”. (Contributed by Giampaolo Rodola’ in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21076" class="reference external">bpo-21076</a>.)

</div>

<div id="smtpd" class="section">

### smtpd<a href="#smtpd" class="headerlink" title="Link to this heading">¶</a>

Both the <span class="pre">`SMTPServer`</span> and <span class="pre">`SMTPChannel`</span> classes now accept a *decode_data* keyword argument to determine if the <span class="pre">`DATA`</span> portion of the SMTP transaction is decoded using the <span class="pre">`"utf-8"`</span> codec or is instead provided to the <span class="pre">`SMTPServer.process_message()`</span> method as a byte string. The default is <span class="pre">`True`</span> for backward compatibility reasons, but will change to <span class="pre">`False`</span> in Python 3.6. If *decode_data* is set to <span class="pre">`False`</span>, the <span class="pre">`process_message`</span> method must be prepared to accept keyword arguments. (Contributed by Maciej Szulik in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19662" class="reference external">bpo-19662</a>.)

The <span class="pre">`SMTPServer`</span> class now advertises the <span class="pre">`8BITMIME`</span> extension (<span id="index-37" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc6152.html" class="rfc reference external"><strong>RFC 6152</strong></a>) if *decode_data* has been set <span class="pre">`True`</span>. If the client specifies <span class="pre">`BODY=8BITMIME`</span> on the <span class="pre">`MAIL`</span> command, it is passed to <span class="pre">`SMTPServer.process_message()`</span> via the *mail_options* keyword. (Contributed by Milan Oberkirch and R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21795" class="reference external">bpo-21795</a>.)

The <span class="pre">`SMTPServer`</span> class now also supports the <span class="pre">`SMTPUTF8`</span> extension (<span id="index-38" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc6531.html" class="rfc reference external"><strong>RFC 6531</strong></a>: Internationalized Email). If the client specified <span class="pre">`SMTPUTF8`</span>` `<span class="pre">`BODY=8BITMIME`</span> on the <span class="pre">`MAIL`</span> command, they are passed to <span class="pre">`SMTPServer.process_message()`</span> via the *mail_options* keyword. It is the responsibility of the <span class="pre">`process_message`</span> method to correctly handle the <span class="pre">`SMTPUTF8`</span> data. (Contributed by Milan Oberkirch in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21725" class="reference external">bpo-21725</a>.)

It is now possible to provide, directly or via name resolution, IPv6 addresses in the <span class="pre">`SMTPServer`</span> constructor, and have it successfully connect. (Contributed by Milan Oberkirch in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14758" class="reference external">bpo-14758</a>.)

</div>

<div id="smtplib" class="section">

### smtplib<a href="#smtplib" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/smtplib.html#smtplib.SMTP.auth" class="reference internal" title="smtplib.SMTP.auth"><span class="pre"><code class="sourceCode python">SMTP.auth()</code></span></a> method provides a convenient way to implement custom authentication mechanisms. (Contributed by Milan Oberkirch in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15014" class="reference external">bpo-15014</a>.)

The <a href="../library/smtplib.html#smtplib.SMTP.set_debuglevel" class="reference internal" title="smtplib.SMTP.set_debuglevel"><span class="pre"><code class="sourceCode python">SMTP.set_debuglevel()</code></span></a> method now accepts an additional debuglevel (2), which enables timestamps in debug messages. (Contributed by Gavin Chappell and Maciej Szulik in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16914" class="reference external">bpo-16914</a>.)

Both the <a href="../library/smtplib.html#smtplib.SMTP.sendmail" class="reference internal" title="smtplib.SMTP.sendmail"><span class="pre"><code class="sourceCode python">SMTP.sendmail()</code></span></a> and <a href="../library/smtplib.html#smtplib.SMTP.send_message" class="reference internal" title="smtplib.SMTP.send_message"><span class="pre"><code class="sourceCode python">SMTP.send_message()</code></span></a> methods now support <span id="index-39" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc6531.html" class="rfc reference external"><strong>RFC 6531</strong></a> (SMTPUTF8). (Contributed by Milan Oberkirch and R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22027" class="reference external">bpo-22027</a>.)

</div>

<div id="sndhdr" class="section">

### sndhdr<a href="#sndhdr" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`what()`</span> and <span class="pre">`whathdr()`</span> functions now return a <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">namedtuple()</code></span></a>. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18615" class="reference external">bpo-18615</a>.)

</div>

<div id="socket" class="section">

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

Functions with timeouts now use a monotonic clock, instead of a system clock. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22043" class="reference external">bpo-22043</a>.)

A new <a href="../library/socket.html#socket.socket.sendfile" class="reference internal" title="socket.socket.sendfile"><span class="pre"><code class="sourceCode python">socket.sendfile()</code></span></a> method allows sending a file over a socket by using the high-performance <a href="../library/os.html#os.sendfile" class="reference internal" title="os.sendfile"><span class="pre"><code class="sourceCode python">os.sendfile()</code></span></a> function on UNIX, resulting in uploads being from 2 to 3 times faster than when using plain <a href="../library/socket.html#socket.socket.send" class="reference internal" title="socket.socket.send"><span class="pre"><code class="sourceCode python">socket.send()</code></span></a>. (Contributed by Giampaolo Rodola’ in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17552" class="reference external">bpo-17552</a>.)

The <a href="../library/socket.html#socket.socket.sendall" class="reference internal" title="socket.socket.sendall"><span class="pre"><code class="sourceCode python">socket.sendall()</code></span></a> method no longer resets the socket timeout every time bytes are received or sent. The socket timeout is now the maximum total duration to send all data. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23853" class="reference external">bpo-23853</a>.)

The *backlog* argument of the <a href="../library/socket.html#socket.socket.listen" class="reference internal" title="socket.socket.listen"><span class="pre"><code class="sourceCode python">socket.listen()</code></span></a> method is now optional. By default it is set to <a href="../library/socket.html#socket.SOMAXCONN" class="reference internal" title="socket.SOMAXCONN"><span class="pre"><code class="sourceCode python">SOMAXCONN</code></span></a> or to <span class="pre">`128`</span>, whichever is less. (Contributed by Charles-François Natali in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21455" class="reference external">bpo-21455</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

<div id="memory-bio-support" class="section">

<span id="whatsnew-sslmemorybio"></span>

#### Memory BIO Support<a href="#memory-bio-support" class="headerlink" title="Link to this heading">¶</a>

(Contributed by Geert Jansen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21965" class="reference external">bpo-21965</a>.)

The new <a href="../library/ssl.html#ssl.SSLObject" class="reference internal" title="ssl.SSLObject"><span class="pre"><code class="sourceCode python">SSLObject</code></span></a> class has been added to provide SSL protocol support for cases when the network I/O capabilities of <a href="../library/ssl.html#ssl.SSLSocket" class="reference internal" title="ssl.SSLSocket"><span class="pre"><code class="sourceCode python">SSLSocket</code></span></a> are not necessary or are suboptimal. <span class="pre">`SSLObject`</span> represents an SSL protocol instance, but does not implement any network I/O methods, and instead provides a memory buffer interface. The new <a href="../library/ssl.html#ssl.MemoryBIO" class="reference internal" title="ssl.MemoryBIO"><span class="pre"><code class="sourceCode python">MemoryBIO</code></span></a> class can be used to pass data between Python and an SSL protocol instance.

The memory BIO SSL support is primarily intended to be used in frameworks implementing asynchronous I/O for which <a href="../library/ssl.html#ssl.SSLSocket" class="reference internal" title="ssl.SSLSocket"><span class="pre"><code class="sourceCode python">SSLSocket</code></span></a>’s readiness model (“select/poll”) is inefficient.

A new <a href="../library/ssl.html#ssl.SSLContext.wrap_bio" class="reference internal" title="ssl.SSLContext.wrap_bio"><span class="pre"><code class="sourceCode python">SSLContext.wrap_bio()</code></span></a> method can be used to create a new <span class="pre">`SSLObject`</span> instance.

</div>

<div id="application-layer-protocol-negotiation-support" class="section">

#### Application-Layer Protocol Negotiation Support<a href="#application-layer-protocol-negotiation-support" class="headerlink" title="Link to this heading">¶</a>

(Contributed by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20188" class="reference external">bpo-20188</a>.)

Where OpenSSL support is present, the <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module now implements the *Application-Layer Protocol Negotiation* TLS extension as described in <span id="index-40" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc7301.html" class="rfc reference external"><strong>RFC 7301</strong></a>.

The new <a href="../library/ssl.html#ssl.SSLContext.set_alpn_protocols" class="reference internal" title="ssl.SSLContext.set_alpn_protocols"><span class="pre"><code class="sourceCode python">SSLContext.set_alpn_protocols()</code></span></a> can be used to specify which protocols a socket should advertise during the TLS handshake.

The new <a href="../library/ssl.html#ssl.SSLSocket.selected_alpn_protocol" class="reference internal" title="ssl.SSLSocket.selected_alpn_protocol"><span class="pre"><code class="sourceCode python">SSLSocket.selected_alpn_protocol()</code></span></a> returns the protocol that was selected during the TLS handshake. The <a href="../library/ssl.html#ssl.HAS_ALPN" class="reference internal" title="ssl.HAS_ALPN"><span class="pre"><code class="sourceCode python">HAS_ALPN</code></span></a> flag indicates whether ALPN support is present.

</div>

<div id="other-changes" class="section">

#### Other Changes<a href="#other-changes" class="headerlink" title="Link to this heading">¶</a>

There is a new <a href="../library/ssl.html#ssl.SSLSocket.version" class="reference internal" title="ssl.SSLSocket.version"><span class="pre"><code class="sourceCode python">SSLSocket.version()</code></span></a> method to query the actual protocol version in use. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20421" class="reference external">bpo-20421</a>.)

The <a href="../library/ssl.html#ssl.SSLSocket" class="reference internal" title="ssl.SSLSocket"><span class="pre"><code class="sourceCode python">SSLSocket</code></span></a> class now implements a <span class="pre">`SSLSocket.sendfile()`</span> method. (Contributed by Giampaolo Rodola’ in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17552" class="reference external">bpo-17552</a>.)

The <span class="pre">`SSLSocket.send()`</span> method now raises either the <a href="../library/ssl.html#ssl.SSLWantReadError" class="reference internal" title="ssl.SSLWantReadError"><span class="pre"><code class="sourceCode python">ssl.SSLWantReadError</code></span></a> or <a href="../library/ssl.html#ssl.SSLWantWriteError" class="reference internal" title="ssl.SSLWantWriteError"><span class="pre"><code class="sourceCode python">ssl.SSLWantWriteError</code></span></a> exception on a non-blocking socket if the operation would block. Previously, it would return <span class="pre">`0`</span>. (Contributed by Nikolaus Rath in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20951" class="reference external">bpo-20951</a>.)

The <a href="../library/ssl.html#ssl.cert_time_to_seconds" class="reference internal" title="ssl.cert_time_to_seconds"><span class="pre"><code class="sourceCode python">cert_time_to_seconds()</code></span></a> function now interprets the input time as UTC and not as local time, per <span id="index-41" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc5280.html" class="rfc reference external"><strong>RFC 5280</strong></a>. Additionally, the return value is always an <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>. (Contributed by Akira Li in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19940" class="reference external">bpo-19940</a>.)

New <span class="pre">`SSLObject.shared_ciphers()`</span> and <a href="../library/ssl.html#ssl.SSLSocket.shared_ciphers" class="reference internal" title="ssl.SSLSocket.shared_ciphers"><span class="pre"><code class="sourceCode python">SSLSocket.shared_ciphers()</code></span></a> methods return the list of ciphers sent by the client during the handshake. (Contributed by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23186" class="reference external">bpo-23186</a>.)

The <a href="../library/ssl.html#ssl.SSLSocket.do_handshake" class="reference internal" title="ssl.SSLSocket.do_handshake"><span class="pre"><code class="sourceCode python">SSLSocket.do_handshake()</code></span></a>, <a href="../library/ssl.html#ssl.SSLSocket.read" class="reference internal" title="ssl.SSLSocket.read"><span class="pre"><code class="sourceCode python">SSLSocket.read()</code></span></a>, <span class="pre">`SSLSocket.shutdown()`</span>, and <a href="../library/ssl.html#ssl.SSLSocket.write" class="reference internal" title="ssl.SSLSocket.write"><span class="pre"><code class="sourceCode python">SSLSocket.write()</code></span></a> methods of the <a href="../library/ssl.html#ssl.SSLSocket" class="reference internal" title="ssl.SSLSocket"><span class="pre"><code class="sourceCode python">SSLSocket</code></span></a> class no longer reset the socket timeout every time bytes are received or sent. The socket timeout is now the maximum total duration of the method. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23853" class="reference external">bpo-23853</a>.)

The <span class="pre">`match_hostname()`</span> function now supports matching of IP addresses. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23239" class="reference external">bpo-23239</a>.)

</div>

</div>

<div id="sqlite3" class="section">

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/sqlite3.html#sqlite3.Row" class="reference internal" title="sqlite3.Row"><span class="pre"><code class="sourceCode python">Row</code></span></a> class now fully supports the sequence protocol, in particular <a href="../library/functions.html#reversed" class="reference internal" title="reversed"><span class="pre"><code class="sourceCode python"><span class="bu">reversed</span>()</code></span></a> iteration and slice indexing. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10203" class="reference external">bpo-10203</a>; by Lucas Sinclair, Jessica McKellar, and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13583" class="reference external">bpo-13583</a>.)

</div>

<div id="subprocess" class="section">

<span id="whatsnew-subprocess"></span>

### subprocess<a href="#subprocess" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/subprocess.html#subprocess.run" class="reference internal" title="subprocess.run"><span class="pre"><code class="sourceCode python">run()</code></span></a> function has been added. It runs the specified command and returns a <a href="../library/subprocess.html#subprocess.CompletedProcess" class="reference internal" title="subprocess.CompletedProcess"><span class="pre"><code class="sourceCode python">CompletedProcess</code></span></a> object, which describes a finished process. The new API is more consistent and is the recommended approach to invoking subprocesses in Python code that does not need to maintain compatibility with earlier Python versions. (Contributed by Thomas Kluyver in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23342" class="reference external">bpo-23342</a>.)

Examples:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> subprocess.run(["ls", "-l"])  # doesn't capture output
    CompletedProcess(args=['ls', '-l'], returncode=0)

    >>> subprocess.run("exit 1", shell=True, check=True)
    Traceback (most recent call last):
      ...
    subprocess.CalledProcessError: Command 'exit 1' returned non-zero exit status 1

    >>> subprocess.run(["ls", "-l", "/dev/null"], stdout=subprocess.PIPE)
    CompletedProcess(args=['ls', '-l', '/dev/null'], returncode=0,
    stdout=b'crw-rw-rw- 1 root root 1, 3 Jan 23 16:23 /dev/null\n')

</div>

</div>

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

A new <span class="pre">`set_coroutine_wrapper()`</span> function allows setting a global hook that will be called whenever a <a href="../glossary.html#term-coroutine" class="reference internal"><span class="xref std std-term">coroutine object</span></a> is created by an <a href="../reference/compound_stmts.html#async-def" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">def</code></span></a> function. A corresponding <span class="pre">`get_coroutine_wrapper()`</span> can be used to obtain a currently set wrapper. Both functions are <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional</span></a>, and are intended for debugging purposes only. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24017" class="reference external">bpo-24017</a>.)

A new <a href="../library/sys.html#sys.is_finalizing" class="reference internal" title="sys.is_finalizing"><span class="pre"><code class="sourceCode python">is_finalizing()</code></span></a> function can be used to check if the Python interpreter is <a href="../glossary.html#term-interpreter-shutdown" class="reference internal"><span class="xref std std-term">shutting down</span></a>. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22696" class="reference external">bpo-22696</a>.)

</div>

<div id="sysconfig" class="section">

### sysconfig<a href="#sysconfig" class="headerlink" title="Link to this heading">¶</a>

The name of the user scripts directory on Windows now includes the first two components of the Python version. (Contributed by Paul Moore in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23437" class="reference external">bpo-23437</a>.)

</div>

<div id="tarfile" class="section">

### tarfile<a href="#tarfile" class="headerlink" title="Link to this heading">¶</a>

The *mode* argument of the <a href="../library/tarfile.html#tarfile.open" class="reference internal" title="tarfile.open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> function now accepts <span class="pre">`"x"`</span> to request exclusive creation. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21717" class="reference external">bpo-21717</a>.)

The <a href="../library/tarfile.html#tarfile.TarFile.extractall" class="reference internal" title="tarfile.TarFile.extractall"><span class="pre"><code class="sourceCode python">TarFile.extractall()</code></span></a> and <a href="../library/tarfile.html#tarfile.TarFile.extract" class="reference internal" title="tarfile.TarFile.extract"><span class="pre"><code class="sourceCode python">TarFile.extract()</code></span></a> methods now take a keyword argument *numeric_owner*. If set to <span class="pre">`True`</span>, the extracted files and directories will be owned by the numeric <span class="pre">`uid`</span> and <span class="pre">`gid`</span> from the tarfile. If set to <span class="pre">`False`</span> (the default, and the behavior in versions prior to 3.5), they will be owned by the named user and group in the tarfile. (Contributed by Michael Vogt and Eric Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23193" class="reference external">bpo-23193</a>.)

The <a href="../library/tarfile.html#tarfile.TarFile.list" class="reference internal" title="tarfile.TarFile.list"><span class="pre"><code class="sourceCode python">TarFile.<span class="bu">list</span>()</code></span></a> now accepts an optional *members* keyword argument that can be set to a subset of the list returned by <a href="../library/tarfile.html#tarfile.TarFile.getmembers" class="reference internal" title="tarfile.TarFile.getmembers"><span class="pre"><code class="sourceCode python">TarFile.getmembers()</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21549" class="reference external">bpo-21549</a>.)

</div>

<div id="threading" class="section">

### threading<a href="#threading" class="headerlink" title="Link to this heading">¶</a>

Both the <a href="../library/threading.html#threading.Lock.acquire" class="reference internal" title="threading.Lock.acquire"><span class="pre"><code class="sourceCode python">Lock.acquire()</code></span></a> and <a href="../library/threading.html#threading.RLock.acquire" class="reference internal" title="threading.RLock.acquire"><span class="pre"><code class="sourceCode python">RLock.acquire()</code></span></a> methods now use a monotonic clock for timeout management. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22043" class="reference external">bpo-22043</a>.)

</div>

<div id="time" class="section">

### time<a href="#time" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/time.html#time.monotonic" class="reference internal" title="time.monotonic"><span class="pre"><code class="sourceCode python">monotonic()</code></span></a> function is now always available. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22043" class="reference external">bpo-22043</a>.)

</div>

<div id="timeit" class="section">

### timeit<a href="#timeit" class="headerlink" title="Link to this heading">¶</a>

A new command line option <span class="pre">`-u`</span> or <span class="pre">`--unit=`</span>*<span class="pre">`U`</span>* can be used to specify the time unit for the timer output. Supported options are <span class="pre">`usec`</span>, <span class="pre">`msec`</span>, or <span class="pre">`sec`</span>. (Contributed by Julian Gindi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18983" class="reference external">bpo-18983</a>.)

The <a href="../library/timeit.html#timeit.timeit" class="reference internal" title="timeit.timeit"><span class="pre"><code class="sourceCode python">timeit()</code></span></a> function has a new *globals* parameter for specifying the namespace in which the code will be running. (Contributed by Ben Roberts in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2527" class="reference external">bpo-2527</a>.)

</div>

<div id="tkinter" class="section">

### tkinter<a href="#tkinter" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`tkinter._fix`</span> module used for setting up the Tcl/Tk environment on Windows has been replaced by a private function in the <span class="pre">`_tkinter`</span> module which makes no permanent changes to environment variables. (Contributed by Zachary Ware in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20035" class="reference external">bpo-20035</a>.)

</div>

<div id="traceback" class="section">

<span id="whatsnew-traceback"></span>

### traceback<a href="#traceback" class="headerlink" title="Link to this heading">¶</a>

New <a href="../library/traceback.html#traceback.walk_stack" class="reference internal" title="traceback.walk_stack"><span class="pre"><code class="sourceCode python">walk_stack()</code></span></a> and <a href="../library/traceback.html#traceback.walk_tb" class="reference internal" title="traceback.walk_tb"><span class="pre"><code class="sourceCode python">walk_tb()</code></span></a> functions to conveniently traverse frame and <a href="../reference/datamodel.html#traceback-objects" class="reference internal"><span class="std std-ref">traceback objects</span></a>. (Contributed by Robert Collins in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17911" class="reference external">bpo-17911</a>.)

New lightweight classes: <a href="../library/traceback.html#traceback.TracebackException" class="reference internal" title="traceback.TracebackException"><span class="pre"><code class="sourceCode python">TracebackException</code></span></a>, <a href="../library/traceback.html#traceback.StackSummary" class="reference internal" title="traceback.StackSummary"><span class="pre"><code class="sourceCode python">StackSummary</code></span></a>, and <a href="../library/traceback.html#traceback.FrameSummary" class="reference internal" title="traceback.FrameSummary"><span class="pre"><code class="sourceCode python">FrameSummary</code></span></a>. (Contributed by Robert Collins in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17911" class="reference external">bpo-17911</a>.)

Both the <a href="../library/traceback.html#traceback.print_tb" class="reference internal" title="traceback.print_tb"><span class="pre"><code class="sourceCode python">print_tb()</code></span></a> and <a href="../library/traceback.html#traceback.print_stack" class="reference internal" title="traceback.print_stack"><span class="pre"><code class="sourceCode python">print_stack()</code></span></a> functions now support negative values for the *limit* argument. (Contributed by Dmitry Kazakov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22619" class="reference external">bpo-22619</a>.)

</div>

<div id="types" class="section">

### types<a href="#types" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/types.html#types.coroutine" class="reference internal" title="types.coroutine"><span class="pre"><code class="sourceCode python">coroutine()</code></span></a> function to transform <a href="../glossary.html#term-generator-iterator" class="reference internal"><span class="xref std std-term">generator</span></a> and <a href="../library/collections.abc.html#collections.abc.Generator" class="reference internal" title="collections.abc.Generator"><span class="pre"><code class="sourceCode python">generator<span class="op">-</span>like</code></span></a> objects into <a href="../glossary.html#term-awaitable" class="reference internal"><span class="xref std std-term">awaitables</span></a>. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24017" class="reference external">bpo-24017</a>.)

A new type called <a href="../library/types.html#types.CoroutineType" class="reference internal" title="types.CoroutineType"><span class="pre"><code class="sourceCode python">CoroutineType</code></span></a>, which is used for <a href="../glossary.html#term-coroutine" class="reference internal"><span class="xref std std-term">coroutine</span></a> objects created by <a href="../reference/compound_stmts.html#async-def" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">def</code></span></a> functions. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24400" class="reference external">bpo-24400</a>.)

</div>

<div id="unicodedata" class="section">

### unicodedata<a href="#unicodedata" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a> module now uses data from <a href="https://unicode.org/versions/Unicode8.0.0/" class="reference external">Unicode 8.0.0</a>.

</div>

<div id="unittest" class="section">

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/unittest.html#unittest.TestLoader.loadTestsFromModule" class="reference internal" title="unittest.TestLoader.loadTestsFromModule"><span class="pre"><code class="sourceCode python">TestLoader.loadTestsFromModule()</code></span></a> method now accepts a keyword-only argument *pattern* which is passed to <span class="pre">`load_tests`</span> as the third argument. Found packages are now checked for <span class="pre">`load_tests`</span> regardless of whether their path matches *pattern*, because it is impossible for a package name to match the default pattern. (Contributed by Robert Collins and Barry A. Warsaw in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16662" class="reference external">bpo-16662</a>.)

Unittest discovery errors now are exposed in the <a href="../library/unittest.html#unittest.TestLoader.errors" class="reference internal" title="unittest.TestLoader.errors"><span class="pre"><code class="sourceCode python">TestLoader.errors</code></span></a> attribute of the <a href="../library/unittest.html#unittest.TestLoader" class="reference internal" title="unittest.TestLoader"><span class="pre"><code class="sourceCode python">TestLoader</code></span></a> instance. (Contributed by Robert Collins in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19746" class="reference external">bpo-19746</a>.)

A new command line option <span class="pre">`--locals`</span> to show local variables in tracebacks. (Contributed by Robert Collins in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22936" class="reference external">bpo-22936</a>.)

</div>

<div id="unittest-mock" class="section">

### unittest.mock<a href="#unittest-mock" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/unittest.mock.html#unittest.mock.Mock" class="reference internal" title="unittest.mock.Mock"><span class="pre"><code class="sourceCode python">Mock</code></span></a> class has the following improvements:

- The class constructor has a new *unsafe* parameter, which causes mock objects to raise <a href="../library/exceptions.html#AttributeError" class="reference internal" title="AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> on attribute names starting with <span class="pre">`"assert"`</span>. (Contributed by Kushal Das in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21238" class="reference external">bpo-21238</a>.)

- A new <a href="../library/unittest.mock.html#unittest.mock.Mock.assert_not_called" class="reference internal" title="unittest.mock.Mock.assert_not_called"><span class="pre"><code class="sourceCode python">Mock.assert_not_called()</code></span></a> method to check if the mock object was called. (Contributed by Kushal Das in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21262" class="reference external">bpo-21262</a>.)

The <a href="../library/unittest.mock.html#unittest.mock.MagicMock" class="reference internal" title="unittest.mock.MagicMock"><span class="pre"><code class="sourceCode python">MagicMock</code></span></a> class now supports <a href="../reference/datamodel.html#object.__truediv__" class="reference internal" title="object.__truediv__"><span class="pre"><code class="sourceCode python"><span class="fu">__truediv__</span>()</code></span></a>, <a href="../reference/datamodel.html#object.__divmod__" class="reference internal" title="object.__divmod__"><span class="pre"><code class="sourceCode python"><span class="fu">__divmod__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__matmul__" class="reference internal" title="object.__matmul__"><span class="pre"><code class="sourceCode python"><span class="fu">__matmul__</span>()</code></span></a> operators. (Contributed by Johannes Baiter in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20968" class="reference external">bpo-20968</a>, and Håkan Lövdahl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23581" class="reference external">bpo-23581</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23568" class="reference external">bpo-23568</a>.)

It is no longer necessary to explicitly pass <span class="pre">`create=True`</span> to the <a href="../library/unittest.mock.html#unittest.mock.patch" class="reference internal" title="unittest.mock.patch"><span class="pre"><code class="sourceCode python">patch()</code></span></a> function when patching builtin names. (Contributed by Kushal Das in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17660" class="reference external">bpo-17660</a>.)

</div>

<div id="urllib" class="section">

### urllib<a href="#urllib" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/urllib.request.html#urllib.request.HTTPPasswordMgrWithPriorAuth" class="reference internal" title="urllib.request.HTTPPasswordMgrWithPriorAuth"><span class="pre"><code class="sourceCode python">request.HTTPPasswordMgrWithPriorAuth</code></span></a> class allows HTTP Basic Authentication credentials to be managed so as to eliminate unnecessary <span class="pre">`401`</span> response handling, or to unconditionally send credentials on the first request in order to communicate with servers that return a <span class="pre">`404`</span> response instead of a <span class="pre">`401`</span> if the <span class="pre">`Authorization`</span> header is not sent. (Contributed by Matej Cepl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19494" class="reference external">bpo-19494</a> and Akshit Khurana in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7159" class="reference external">bpo-7159</a>.)

A new *quote_via* argument for the <a href="../library/urllib.parse.html#urllib.parse.urlencode" class="reference internal" title="urllib.parse.urlencode"><span class="pre"><code class="sourceCode python">parse.urlencode()</code></span></a> function provides a way to control the encoding of query parts if needed. (Contributed by Samwyse and Arnon Yaari in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13866" class="reference external">bpo-13866</a>.)

The <a href="../library/urllib.request.html#urllib.request.urlopen" class="reference internal" title="urllib.request.urlopen"><span class="pre"><code class="sourceCode python">request.urlopen()</code></span></a> function accepts an <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a> object as a *context* argument, which will be used for the HTTPS connection. (Contributed by Alex Gaynor in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22366" class="reference external">bpo-22366</a>.)

The <a href="../library/urllib.parse.html#urllib.parse.urljoin" class="reference internal" title="urllib.parse.urljoin"><span class="pre"><code class="sourceCode python">parse.urljoin()</code></span></a> was updated to use the <span id="index-42" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc3986.html" class="rfc reference external"><strong>RFC 3986</strong></a> semantics for the resolution of relative URLs, rather than <span id="index-43" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc1808.html" class="rfc reference external"><strong>RFC 1808</strong></a> and <span id="index-44" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc2396.html" class="rfc reference external"><strong>RFC 2396</strong></a>. (Contributed by Demian Brecht and Senthil Kumaran in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22118" class="reference external">bpo-22118</a>.)

</div>

<div id="wsgiref" class="section">

### wsgiref<a href="#wsgiref" class="headerlink" title="Link to this heading">¶</a>

The *headers* argument of the <a href="../library/wsgiref.html#wsgiref.headers.Headers" class="reference internal" title="wsgiref.headers.Headers"><span class="pre"><code class="sourceCode python">headers.Headers</code></span></a> class constructor is now optional. (Contributed by Pablo Torres Navarrete and SilentGhost in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5800" class="reference external">bpo-5800</a>.)

</div>

<div id="xmlrpc" class="section">

### xmlrpc<a href="#xmlrpc" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/xmlrpc.client.html#xmlrpc.client.ServerProxy" class="reference internal" title="xmlrpc.client.ServerProxy"><span class="pre"><code class="sourceCode python">client.ServerProxy</code></span></a> class now supports the <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> protocol. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20627" class="reference external">bpo-20627</a>.)

The <a href="../library/xmlrpc.client.html#xmlrpc.client.ServerProxy" class="reference internal" title="xmlrpc.client.ServerProxy"><span class="pre"><code class="sourceCode python">client.ServerProxy</code></span></a> constructor now accepts an optional <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a> instance. (Contributed by Alex Gaynor in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22960" class="reference external">bpo-22960</a>.)

</div>

<div id="xml-sax" class="section">

### xml.sax<a href="#xml-sax" class="headerlink" title="Link to this heading">¶</a>

SAX parsers now support a character stream of the <a href="../library/xml.sax.reader.html#xml.sax.xmlreader.InputSource" class="reference internal" title="xml.sax.xmlreader.InputSource"><span class="pre"><code class="sourceCode python">xmlreader.InputSource</code></span></a> object. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2175" class="reference external">bpo-2175</a>.)

<a href="../library/xml.sax.html#xml.sax.parseString" class="reference internal" title="xml.sax.parseString"><span class="pre"><code class="sourceCode python">parseString()</code></span></a> now accepts a <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> instance. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10590" class="reference external">bpo-10590</a>.)

</div>

<div id="zipfile" class="section">

### zipfile<a href="#zipfile" class="headerlink" title="Link to this heading">¶</a>

ZIP output can now be written to unseekable streams. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23252" class="reference external">bpo-23252</a>.)

The *mode* argument of <a href="../library/zipfile.html#zipfile.ZipFile.open" class="reference internal" title="zipfile.ZipFile.open"><span class="pre"><code class="sourceCode python">ZipFile.<span class="bu">open</span>()</code></span></a> method now accepts <span class="pre">`"x"`</span> to request exclusive creation. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21717" class="reference external">bpo-21717</a>.)

</div>

</div>

<div id="other-module-level-changes" class="section">

## Other module-level changes<a href="#other-module-level-changes" class="headerlink" title="Link to this heading">¶</a>

Many functions in the <a href="../library/mmap.html#module-mmap" class="reference internal" title="mmap: Interface to memory-mapped files for Unix and Windows."><span class="pre"><code class="sourceCode python">mmap</code></span></a>, <span class="pre">`ossaudiodev`</span>, <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a>, <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a>, and <a href="../library/codecs.html#module-codecs" class="reference internal" title="codecs: Encode and decode data and streams."><span class="pre"><code class="sourceCode python">codecs</code></span></a> modules now accept writable <a href="../glossary.html#term-bytes-like-object" class="reference internal"><span class="xref std std-term">bytes-like objects</span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23001" class="reference external">bpo-23001</a>.)

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/os.html#os.walk" class="reference internal" title="os.walk"><span class="pre"><code class="sourceCode python">os.walk()</code></span></a> function has been sped up by 3 to 5 times on POSIX systems, and by 7 to 20 times on Windows. This was done using the new <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a> function, which exposes file information from the underlying <span class="pre">`readdir`</span> or <span class="pre">`FindFirstFile`</span>/<span class="pre">`FindNextFile`</span> system calls. (Contributed by Ben Hoyt with help from Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23605" class="reference external">bpo-23605</a>.)

Construction of <span class="pre">`bytes(int)`</span> (filled by zero bytes) is faster and uses less memory for large objects. <span class="pre">`calloc()`</span> is used instead of <span class="pre">`malloc()`</span> to allocate memory for these objects. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21233" class="reference external">bpo-21233</a>.)

Some operations on <a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> <a href="../library/ipaddress.html#ipaddress.IPv4Network" class="reference internal" title="ipaddress.IPv4Network"><span class="pre"><code class="sourceCode python">IPv4Network</code></span></a> and <a href="../library/ipaddress.html#ipaddress.IPv6Network" class="reference internal" title="ipaddress.IPv6Network"><span class="pre"><code class="sourceCode python">IPv6Network</code></span></a> have been massively sped up, such as <a href="../library/ipaddress.html#ipaddress.IPv4Network.subnets" class="reference internal" title="ipaddress.IPv4Network.subnets"><span class="pre"><code class="sourceCode python">subnets()</code></span></a>, <a href="../library/ipaddress.html#ipaddress.IPv4Network.supernet" class="reference internal" title="ipaddress.IPv4Network.supernet"><span class="pre"><code class="sourceCode python">supernet()</code></span></a>, <a href="../library/ipaddress.html#ipaddress.summarize_address_range" class="reference internal" title="ipaddress.summarize_address_range"><span class="pre"><code class="sourceCode python">summarize_address_range()</code></span></a>, <a href="../library/ipaddress.html#ipaddress.collapse_addresses" class="reference internal" title="ipaddress.collapse_addresses"><span class="pre"><code class="sourceCode python">collapse_addresses()</code></span></a>. The speed up can range from 3 to 15 times. (Contributed by Antoine Pitrou, Michel Albert, and Markus in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21486" class="reference external">bpo-21486</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21487" class="reference external">bpo-21487</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20826" class="reference external">bpo-20826</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23266" class="reference external">bpo-23266</a>.)

Pickling of <a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> objects was optimized to produce significantly smaller output. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23133" class="reference external">bpo-23133</a>.)

Many operations on <a href="../library/io.html#io.BytesIO" class="reference internal" title="io.BytesIO"><span class="pre"><code class="sourceCode python">io.BytesIO</code></span></a> are now 50% to 100% faster. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15381" class="reference external">bpo-15381</a> and David Wilson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22003" class="reference external">bpo-22003</a>.)

The <a href="../library/marshal.html#marshal.dumps" class="reference internal" title="marshal.dumps"><span class="pre"><code class="sourceCode python">marshal.dumps()</code></span></a> function is now faster: 65–85% with versions 3 and 4, 20–25% with versions 0 to 2 on typical data, and up to 5 times in best cases. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20416" class="reference external">bpo-20416</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23344" class="reference external">bpo-23344</a>.)

The UTF-32 encoder is now 3 to 7 times faster. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15027" class="reference external">bpo-15027</a>.)

Regular expressions are now parsed up to 10% faster. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19380" class="reference external">bpo-19380</a>.)

The <a href="../library/json.html#json.dumps" class="reference internal" title="json.dumps"><span class="pre"><code class="sourceCode python">json.dumps()</code></span></a> function was optimized to run with <span class="pre">`ensure_ascii=False`</span> as fast as with <span class="pre">`ensure_ascii=True`</span>. (Contributed by Naoki Inada in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23206" class="reference external">bpo-23206</a>.)

The <a href="../c-api/object.html#c.PyObject_IsInstance" class="reference internal" title="PyObject_IsInstance"><span class="pre"><code class="sourceCode c">PyObject_IsInstance<span class="op">()</span></code></span></a> and <a href="../c-api/object.html#c.PyObject_IsSubclass" class="reference internal" title="PyObject_IsSubclass"><span class="pre"><code class="sourceCode c">PyObject_IsSubclass<span class="op">()</span></code></span></a> functions have been sped up in the common case that the second argument has <a href="../library/functions.html#type" class="reference internal" title="type"><span class="pre"><code class="sourceCode python"><span class="bu">type</span></code></span></a> as its metaclass. (Contributed Georg Brandl by in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22540" class="reference external">bpo-22540</a>.)

Method caching was slightly improved, yielding up to 5% performance improvement in some benchmarks. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22847" class="reference external">bpo-22847</a>.)

Objects from the <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module now use 50% less memory on 64-bit builds. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23488" class="reference external">bpo-23488</a>.)

The <a href="../library/functions.html#property" class="reference internal" title="property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span>()</code></span></a> getter calls are up to 25% faster. (Contributed by Joe Jevnik in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23910" class="reference external">bpo-23910</a>.)

Instantiation of <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">fractions.Fraction</code></span></a> is now up to 30% faster. (Contributed by Stefan Behnel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22464" class="reference external">bpo-22464</a>.)

String methods <a href="../library/stdtypes.html#str.find" class="reference internal" title="str.find"><span class="pre"><code class="sourceCode python">find()</code></span></a>, <a href="../library/stdtypes.html#str.rfind" class="reference internal" title="str.rfind"><span class="pre"><code class="sourceCode python">rfind()</code></span></a>, <a href="../library/stdtypes.html#str.split" class="reference internal" title="str.split"><span class="pre"><code class="sourceCode python">split()</code></span></a>, <a href="../library/stdtypes.html#str.partition" class="reference internal" title="str.partition"><span class="pre"><code class="sourceCode python">partition()</code></span></a> and the <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> string operator are now significantly faster for searching 1-character substrings. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23573" class="reference external">bpo-23573</a>.)

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Link to this heading">¶</a>

New <span class="pre">`calloc`</span> functions were added:

- <a href="../c-api/memory.html#c.PyMem_RawCalloc" class="reference internal" title="PyMem_RawCalloc"><span class="pre"><code class="sourceCode c">PyMem_RawCalloc<span class="op">()</span></code></span></a>,

- <a href="../c-api/memory.html#c.PyMem_Calloc" class="reference internal" title="PyMem_Calloc"><span class="pre"><code class="sourceCode c">PyMem_Calloc<span class="op">()</span></code></span></a>,

- <a href="../c-api/memory.html#c.PyObject_Calloc" class="reference internal" title="PyObject_Calloc"><span class="pre"><code class="sourceCode c">PyObject_Calloc<span class="op">()</span></code></span></a>.

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21233" class="reference external">bpo-21233</a>.)

New encoding/decoding helper functions:

- <a href="../c-api/sys.html#c.Py_DecodeLocale" class="reference internal" title="Py_DecodeLocale"><span class="pre"><code class="sourceCode c">Py_DecodeLocale<span class="op">()</span></code></span></a> (replaced <span class="pre">`_Py_char2wchar()`</span>),

- <a href="../c-api/sys.html#c.Py_EncodeLocale" class="reference internal" title="Py_EncodeLocale"><span class="pre"><code class="sourceCode c">Py_EncodeLocale<span class="op">()</span></code></span></a> (replaced <span class="pre">`_Py_wchar2char()`</span>).

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18395" class="reference external">bpo-18395</a>.)

A new <a href="../c-api/codec.html#c.PyCodec_NameReplaceErrors" class="reference internal" title="PyCodec_NameReplaceErrors"><span class="pre"><code class="sourceCode c">PyCodec_NameReplaceErrors<span class="op">()</span></code></span></a> function to replace the unicode encode error with <span class="pre">`\N{...}`</span> escapes. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19676" class="reference external">bpo-19676</a>.)

A new <a href="../c-api/exceptions.html#c.PyErr_FormatV" class="reference internal" title="PyErr_FormatV"><span class="pre"><code class="sourceCode c">PyErr_FormatV<span class="op">()</span></code></span></a> function similar to <a href="../c-api/exceptions.html#c.PyErr_Format" class="reference internal" title="PyErr_Format"><span class="pre"><code class="sourceCode c">PyErr_Format<span class="op">()</span></code></span></a>, but accepts a <span class="pre">`va_list`</span> argument. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18711" class="reference external">bpo-18711</a>.)

A new <a href="../c-api/exceptions.html#c.PyExc_RecursionError" class="reference internal" title="PyExc_RecursionError"><span class="pre"><code class="sourceCode c">PyExc_RecursionError</code></span></a> exception. (Contributed by Georg Brandl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19235" class="reference external">bpo-19235</a>.)

New <a href="../c-api/module.html#c.PyModule_FromDefAndSpec" class="reference internal" title="PyModule_FromDefAndSpec"><span class="pre"><code class="sourceCode c">PyModule_FromDefAndSpec<span class="op">()</span></code></span></a>, <a href="../c-api/module.html#c.PyModule_FromDefAndSpec2" class="reference internal" title="PyModule_FromDefAndSpec2"><span class="pre"><code class="sourceCode c">PyModule_FromDefAndSpec2<span class="op">()</span></code></span></a>, and <a href="../c-api/module.html#c.PyModule_ExecDef" class="reference internal" title="PyModule_ExecDef"><span class="pre"><code class="sourceCode c">PyModule_ExecDef<span class="op">()</span></code></span></a> functions introduced by <span id="index-45" class="target"></span><a href="https://peps.python.org/pep-0489/" class="pep reference external"><strong>PEP 489</strong></a> – multi-phase extension module initialization. (Contributed by Petr Viktorin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24268" class="reference external">bpo-24268</a>.)

New <a href="../c-api/number.html#c.PyNumber_MatrixMultiply" class="reference internal" title="PyNumber_MatrixMultiply"><span class="pre"><code class="sourceCode c">PyNumber_MatrixMultiply<span class="op">()</span></code></span></a> and <a href="../c-api/number.html#c.PyNumber_InPlaceMatrixMultiply" class="reference internal" title="PyNumber_InPlaceMatrixMultiply"><span class="pre"><code class="sourceCode c">PyNumber_InPlaceMatrixMultiply<span class="op">()</span></code></span></a> functions to perform matrix multiplication. (Contributed by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21176" class="reference external">bpo-21176</a>. See also <span id="index-46" class="target"></span><a href="https://peps.python.org/pep-0465/" class="pep reference external"><strong>PEP 465</strong></a> for details.)

The <a href="../c-api/typeobj.html#c.PyTypeObject.tp_finalize" class="reference internal" title="PyTypeObject.tp_finalize"><span class="pre"><code class="sourceCode c">PyTypeObject<span class="op">.</span>tp_finalize</code></span></a> slot is now part of the stable ABI.

Windows builds now require Microsoft Visual C++ 14.0, which is available as part of <a href="https://visualstudio.microsoft.com/en/vs/older-downloads/#visual-studio-2015-and-other-products" class="reference external">Visual Studio 2015</a>.

Extension modules now include a platform information tag in their filename on some platforms (the tag is optional, and CPython will import extensions without it, although if the tag is present and mismatched, the extension won’t be loaded):

- On Linux, extension module filenames end with <span class="pre">`.cpython-<major><minor>m-<architecture>-<os>.pyd`</span>:

  - <span class="pre">`<major>`</span> is the major number of the Python version; for Python 3.5 this is <span class="pre">`3`</span>.

  - <span class="pre">`<minor>`</span> is the minor number of the Python version; for Python 3.5 this is <span class="pre">`5`</span>.

  - <span class="pre">`<architecture>`</span> is the hardware architecture the extension module was built to run on. It’s most commonly either <span class="pre">`i386`</span> for 32-bit Intel platforms or <span class="pre">`x86_64`</span> for 64-bit Intel (and AMD) platforms.

  - <span class="pre">`<os>`</span> is always <span class="pre">`linux-gnu`</span>, except for extensions built to talk to the 32-bit ABI on 64-bit platforms, in which case it is <span class="pre">`linux-gnu32`</span> (and <span class="pre">`<architecture>`</span> will be <span class="pre">`x86_64`</span>).

- On Windows, extension module filenames end with <span class="pre">`<debug>.cp<major><minor>-<platform>.pyd`</span>:

  - <span class="pre">`<major>`</span> is the major number of the Python version; for Python 3.5 this is <span class="pre">`3`</span>.

  - <span class="pre">`<minor>`</span> is the minor number of the Python version; for Python 3.5 this is <span class="pre">`5`</span>.

  - <span class="pre">`<platform>`</span> is the platform the extension module was built for, either <span class="pre">`win32`</span> for Win32, <span class="pre">`win_amd64`</span> for Win64, <span class="pre">`win_ia64`</span> for Windows Itanium 64, and <span class="pre">`win_arm`</span> for Windows on ARM.

  - If built in debug mode, <span class="pre">`<debug>`</span> will be <span class="pre">`_d`</span>, otherwise it will be blank.

- On OS X platforms, extension module filenames now end with <span class="pre">`-darwin.so`</span>.

- On all other platforms, extension module filenames are the same as they were with Python 3.4.

</div>

<div id="deprecated" class="section">

## Deprecated<a href="#deprecated" class="headerlink" title="Link to this heading">¶</a>

<div id="new-keywords" class="section">

### New Keywords<a href="#new-keywords" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`async`</span> and <span class="pre">`await`</span> are not recommended to be used as variable, class, function or module names. Introduced by <span id="index-47" class="target"></span><a href="https://peps.python.org/pep-0492/" class="pep reference external"><strong>PEP 492</strong></a> in Python 3.5, they will become proper keywords in Python 3.7.

</div>

<div id="deprecated-python-behavior" class="section">

### Deprecated Python Behavior<a href="#deprecated-python-behavior" class="headerlink" title="Link to this heading">¶</a>

Raising the <a href="../library/exceptions.html#StopIteration" class="reference internal" title="StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a> exception inside a generator will now generate a silent <a href="../library/exceptions.html#PendingDeprecationWarning" class="reference internal" title="PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a>, which will become a non-silent deprecation warning in Python 3.6 and will trigger a <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> in Python 3.7. See <a href="#whatsnew-pep-479" class="reference internal"><span class="std std-ref">PEP 479: Change StopIteration handling inside generators</span></a> for details.

</div>

<div id="unsupported-operating-systems" class="section">

### Unsupported Operating Systems<a href="#unsupported-operating-systems" class="headerlink" title="Link to this heading">¶</a>

Windows XP is no longer supported by Microsoft, thus, per <span id="index-48" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a>, CPython 3.5 is no longer officially supported on this OS.

</div>

<div id="deprecated-python-modules-functions-and-methods" class="section">

### Deprecated Python modules, functions and methods<a href="#deprecated-python-modules-functions-and-methods" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`formatter`</span> module has now graduated to full deprecation and is still slated for removal in Python 3.6.

The <span class="pre">`asyncio.async()`</span> function is deprecated in favor of <a href="../library/asyncio-future.html#asyncio.ensure_future" class="reference internal" title="asyncio.ensure_future"><span class="pre"><code class="sourceCode python">ensure_future()</code></span></a>.

The <span class="pre">`smtpd`</span> module has in the past always decoded the DATA portion of email messages using the <span class="pre">`utf-8`</span> codec. This can now be controlled by the new *decode_data* keyword to <span class="pre">`SMTPServer`</span>. The default value is <span class="pre">`True`</span>, but this default is deprecated. Specify the *decode_data* keyword with an appropriate value to avoid the deprecation warning.

Directly assigning values to the <a href="../library/http.cookies.html#http.cookies.Morsel.key" class="reference internal" title="http.cookies.Morsel.key"><span class="pre"><code class="sourceCode python">key</code></span></a>, <a href="../library/http.cookies.html#http.cookies.Morsel.value" class="reference internal" title="http.cookies.Morsel.value"><span class="pre"><code class="sourceCode python">value</code></span></a> and <a href="../library/http.cookies.html#http.cookies.Morsel.coded_value" class="reference internal" title="http.cookies.Morsel.coded_value"><span class="pre"><code class="sourceCode python">coded_value</code></span></a> of <a href="../library/http.cookies.html#http.cookies.Morsel" class="reference internal" title="http.cookies.Morsel"><span class="pre"><code class="sourceCode python">http.cookies.Morsel</code></span></a> objects is deprecated. Use the <a href="../library/http.cookies.html#http.cookies.Morsel.set" class="reference internal" title="http.cookies.Morsel.set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a> method instead. In addition, the undocumented *LegalChars* parameter of <a href="../library/http.cookies.html#http.cookies.Morsel.set" class="reference internal" title="http.cookies.Morsel.set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a> is deprecated, and is now ignored.

Passing a format string as keyword argument *format_string* to the <a href="../library/string.html#string.Formatter.format" class="reference internal" title="string.Formatter.format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> method of the <a href="../library/string.html#string.Formatter" class="reference internal" title="string.Formatter"><span class="pre"><code class="sourceCode python">string.Formatter</code></span></a> class has been deprecated. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23671" class="reference external">bpo-23671</a>.)

The <span class="pre">`platform.dist()`</span> and <span class="pre">`platform.linux_distribution()`</span> functions are now deprecated. Linux distributions use too many different ways of describing themselves, so the functionality is left to a package. (Contributed by Vajrasky Kok and Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1322" class="reference external">bpo-1322</a>.)

The previously undocumented <span class="pre">`from_function`</span> and <span class="pre">`from_builtin`</span> methods of <a href="../library/inspect.html#inspect.Signature" class="reference internal" title="inspect.Signature"><span class="pre"><code class="sourceCode python">inspect.Signature</code></span></a> are deprecated. Use the new <a href="../library/inspect.html#inspect.Signature.from_callable" class="reference internal" title="inspect.Signature.from_callable"><span class="pre"><code class="sourceCode python">Signature.from_callable()</code></span></a> method instead. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24248" class="reference external">bpo-24248</a>.)

The <span class="pre">`inspect.getargspec()`</span> function is deprecated and scheduled to be removed in Python 3.6. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20438" class="reference external">bpo-20438</a> for details.)

The <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> <a href="../library/inspect.html#inspect.getfullargspec" class="reference internal" title="inspect.getfullargspec"><span class="pre"><code class="sourceCode python">getfullargspec()</code></span></a>, <a href="../library/inspect.html#inspect.getcallargs" class="reference internal" title="inspect.getcallargs"><span class="pre"><code class="sourceCode python">getcallargs()</code></span></a>, and <span class="pre">`formatargspec()`</span> functions are deprecated in favor of the <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a> API. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20438" class="reference external">bpo-20438</a>.)

<a href="../library/inspect.html#inspect.getargvalues" class="reference internal" title="inspect.getargvalues"><span class="pre"><code class="sourceCode python">getargvalues()</code></span></a> and <a href="../library/inspect.html#inspect.formatargvalues" class="reference internal" title="inspect.formatargvalues"><span class="pre"><code class="sourceCode python">formatargvalues()</code></span></a> functions were inadvertently marked as deprecated with the release of Python 3.5.0.

Use of <a href="../library/re.html#re.LOCALE" class="reference internal" title="re.LOCALE"><span class="pre"><code class="sourceCode python">re.LOCALE</code></span></a> flag with str patterns or <a href="../library/re.html#re.ASCII" class="reference internal" title="re.ASCII"><span class="pre"><code class="sourceCode python">re.ASCII</code></span></a> is now deprecated. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22407" class="reference external">bpo-22407</a>.)

Use of unrecognized special sequences consisting of <span class="pre">`'\'`</span> and an ASCII letter in regular expression patterns and replacement patterns now raises a deprecation warning and will be forbidden in Python 3.6. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23622" class="reference external">bpo-23622</a>.)

The undocumented and unofficial *use_load_tests* default argument of the <a href="../library/unittest.html#unittest.TestLoader.loadTestsFromModule" class="reference internal" title="unittest.TestLoader.loadTestsFromModule"><span class="pre"><code class="sourceCode python">unittest.TestLoader.loadTestsFromModule()</code></span></a> method now is deprecated and ignored. (Contributed by Robert Collins and Barry A. Warsaw in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16662" class="reference external">bpo-16662</a>.)

</div>

</div>

<div id="removed" class="section">

## Removed<a href="#removed" class="headerlink" title="Link to this heading">¶</a>

<div id="api-and-feature-removals" class="section">

### API and Feature Removals<a href="#api-and-feature-removals" class="headerlink" title="Link to this heading">¶</a>

The following obsolete and previously deprecated APIs and features have been removed:

- The <span class="pre">`__version__`</span> attribute has been dropped from the email package. The email code hasn’t been shipped separately from the stdlib for a long time, and the <span class="pre">`__version__`</span> string was not updated in the last few releases.

- The internal <span class="pre">`Netrc`</span> class in the <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a> module was deprecated in 3.4, and has now been removed. (Contributed by Matt Chaput in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6623" class="reference external">bpo-6623</a>.)

- The concept of <span class="pre">`.pyo`</span> files has been removed.

- The JoinableQueue class in the provisional <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> module was deprecated in 3.4.4 and is now removed. (Contributed by A. Jesse Jiryu Davis in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23464" class="reference external">bpo-23464</a>.)

</div>

</div>

<div id="porting-to-python-3-5" class="section">

## Porting to Python 3.5<a href="#porting-to-python-3-5" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="changes-in-python-behavior" class="section">

### Changes in Python behavior<a href="#changes-in-python-behavior" class="headerlink" title="Link to this heading">¶</a>

- Due to an oversight, earlier Python versions erroneously accepted the following syntax:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      f(1 for x in [1], *args)
      f(1 for x in [1], **kwargs)

  </div>

  </div>

  Python 3.5 now correctly raises a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>, as generator expressions must be put in parentheses if not a sole argument to a function.

</div>

<div id="changes-in-the-python-api" class="section">

### Changes in the Python API<a href="#changes-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-49" class="target"></span><a href="https://peps.python.org/pep-0475/" class="pep reference external"><strong>PEP 475</strong></a>: System calls are now retried when interrupted by a signal instead of raising <a href="../library/exceptions.html#InterruptedError" class="reference internal" title="InterruptedError"><span class="pre"><code class="sourceCode python"><span class="pp">InterruptedError</span></code></span></a> if the Python signal handler does not raise an exception.

- Before Python 3.5, a <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">datetime.time</code></span></a> object was considered to be false if it represented midnight in UTC. This behavior was considered obscure and error-prone and has been removed in Python 3.5. See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13936" class="reference external">bpo-13936</a> for full details.

- The <span class="pre">`ssl.SSLSocket.send()`</span> method now raises either <a href="../library/ssl.html#ssl.SSLWantReadError" class="reference internal" title="ssl.SSLWantReadError"><span class="pre"><code class="sourceCode python">ssl.SSLWantReadError</code></span></a> or <a href="../library/ssl.html#ssl.SSLWantWriteError" class="reference internal" title="ssl.SSLWantWriteError"><span class="pre"><code class="sourceCode python">ssl.SSLWantWriteError</code></span></a> on a non-blocking socket if the operation would block. Previously, it would return <span class="pre">`0`</span>. (Contributed by Nikolaus Rath in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20951" class="reference external">bpo-20951</a>.)

- The <span class="pre">`__name__`</span> attribute of generators is now set from the function name, instead of being set from the code name. Use <span class="pre">`gen.gi_code.co_name`</span> to retrieve the code name. Generators also have a new <span class="pre">`__qualname__`</span> attribute, the qualified name, which is now used for the representation of a generator (<span class="pre">`repr(gen)`</span>). (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21205" class="reference external">bpo-21205</a>.)

- The deprecated “strict” mode and argument of <a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">HTMLParser</code></span></a>, <span class="pre">`HTMLParser.error()`</span>, and the <span class="pre">`HTMLParserError`</span> exception have been removed. (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15114" class="reference external">bpo-15114</a>.) The *convert_charrefs* argument of <a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">HTMLParser</code></span></a> is now <span class="pre">`True`</span> by default. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21047" class="reference external">bpo-21047</a>.)

- Although it is not formally part of the API, it is worth noting for porting purposes (ie: fixing tests) that error messages that were previously of the form “‘sometype’ does not support the buffer protocol” are now of the form “a <a href="../glossary.html#term-bytes-like-object" class="reference internal"><span class="xref std std-term">bytes-like object</span></a> is required, not ‘sometype’”. (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16518" class="reference external">bpo-16518</a>.)

- If the current directory is set to a directory that no longer exists then <a href="../library/exceptions.html#FileNotFoundError" class="reference internal" title="FileNotFoundError"><span class="pre"><code class="sourceCode python"><span class="pp">FileNotFoundError</span></code></span></a> will no longer be raised and instead <a href="../library/importlib.html#importlib.machinery.FileFinder.find_spec" class="reference internal" title="importlib.machinery.FileFinder.find_spec"><span class="pre"><code class="sourceCode python">find_spec()</code></span></a> will return <span class="pre">`None`</span> **without** caching <span class="pre">`None`</span> in <a href="../library/sys.html#sys.path_importer_cache" class="reference internal" title="sys.path_importer_cache"><span class="pre"><code class="sourceCode python">sys.path_importer_cache</code></span></a>, which is different than the typical case (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22834" class="reference external">bpo-22834</a>).

- HTTP status code and messages from <a href="../library/http.client.html#module-http.client" class="reference internal" title="http.client: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">http.client</code></span></a> and <a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a> were refactored into a common <a href="../library/http.html#http.HTTPStatus" class="reference internal" title="http.HTTPStatus"><span class="pre"><code class="sourceCode python">HTTPStatus</code></span></a> enum. The values in <a href="../library/http.client.html#module-http.client" class="reference internal" title="http.client: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">http.client</code></span></a> and <a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a> remain available for backwards compatibility. (Contributed by Demian Brecht in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21793" class="reference external">bpo-21793</a>.)

- When an import loader defines <a href="../library/importlib.html#importlib.abc.Loader.exec_module" class="reference internal" title="importlib.abc.Loader.exec_module"><span class="pre"><code class="sourceCode python">exec_module()</code></span></a> it is now expected to also define <a href="../library/importlib.html#importlib.abc.Loader.create_module" class="reference internal" title="importlib.abc.Loader.create_module"><span class="pre"><code class="sourceCode python">create_module()</code></span></a> (raises a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> now, will be an error in Python 3.6). If the loader inherits from <a href="../library/importlib.html#importlib.abc.Loader" class="reference internal" title="importlib.abc.Loader"><span class="pre"><code class="sourceCode python">importlib.abc.Loader</code></span></a> then there is nothing to do, else simply define <a href="../library/importlib.html#importlib.abc.Loader.create_module" class="reference internal" title="importlib.abc.Loader.create_module"><span class="pre"><code class="sourceCode python">create_module()</code></span></a> to return <span class="pre">`None`</span>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23014" class="reference external">bpo-23014</a>.)

- The <a href="../library/re.html#re.split" class="reference internal" title="re.split"><span class="pre"><code class="sourceCode python">re.split()</code></span></a> function always ignored empty pattern matches, so the <span class="pre">`"x*"`</span> pattern worked the same as <span class="pre">`"x+"`</span>, and the <span class="pre">`"\b"`</span> pattern never worked. Now <a href="../library/re.html#re.split" class="reference internal" title="re.split"><span class="pre"><code class="sourceCode python">re.split()</code></span></a> raises a warning if the pattern could match an empty string. For compatibility, use patterns that never match an empty string (e.g. <span class="pre">`"x+"`</span> instead of <span class="pre">`"x*"`</span>). Patterns that could only match an empty string (such as <span class="pre">`"\b"`</span>) now raise an error. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22818" class="reference external">bpo-22818</a>.)

- The <a href="../library/http.cookies.html#http.cookies.Morsel" class="reference internal" title="http.cookies.Morsel"><span class="pre"><code class="sourceCode python">http.cookies.Morsel</code></span></a> dict-like interface has been made self consistent: morsel comparison now takes the <a href="../library/http.cookies.html#http.cookies.Morsel.key" class="reference internal" title="http.cookies.Morsel.key"><span class="pre"><code class="sourceCode python">key</code></span></a> and <a href="../library/http.cookies.html#http.cookies.Morsel.value" class="reference internal" title="http.cookies.Morsel.value"><span class="pre"><code class="sourceCode python">value</code></span></a> into account, <a href="../library/http.cookies.html#http.cookies.Morsel.copy" class="reference internal" title="http.cookies.Morsel.copy"><span class="pre"><code class="sourceCode python">copy()</code></span></a> now results in a <a href="../library/http.cookies.html#http.cookies.Morsel" class="reference internal" title="http.cookies.Morsel"><span class="pre"><code class="sourceCode python">Morsel</code></span></a> instance rather than a <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a>, and <a href="../library/http.cookies.html#http.cookies.Morsel.update" class="reference internal" title="http.cookies.Morsel.update"><span class="pre"><code class="sourceCode python">update()</code></span></a> will now raise an exception if any of the keys in the update dictionary are invalid. In addition, the undocumented *LegalChars* parameter of <a href="../library/http.cookies.html#http.cookies.Morsel.set" class="reference internal" title="http.cookies.Morsel.set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a> is deprecated and is now ignored. (Contributed by Demian Brecht in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2211" class="reference external">bpo-2211</a>.)

- <span id="index-50" class="target"></span><a href="https://peps.python.org/pep-0488/" class="pep reference external"><strong>PEP 488</strong></a> has removed <span class="pre">`.pyo`</span> files from Python and introduced the optional <span class="pre">`opt-`</span> tag in <span class="pre">`.pyc`</span> file names. The <a href="../library/importlib.html#importlib.util.cache_from_source" class="reference internal" title="importlib.util.cache_from_source"><span class="pre"><code class="sourceCode python">importlib.util.cache_from_source()</code></span></a> has gained an *optimization* parameter to help control the <span class="pre">`opt-`</span> tag. Because of this, the *debug_override* parameter of the function is now deprecated. <span class="pre">`.pyo`</span> files are also no longer supported as a file argument to the Python interpreter and thus serve no purpose when distributed on their own (i.e. sourceless code distribution). Due to the fact that the magic number for bytecode has changed in Python 3.5, all old <span class="pre">`.pyo`</span> files from previous versions of Python are invalid regardless of this PEP.

- The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module now exports the <a href="../library/socket.html#socket.CAN_RAW_FD_FRAMES" class="reference internal" title="socket.CAN_RAW_FD_FRAMES"><span class="pre"><code class="sourceCode python">CAN_RAW_FD_FRAMES</code></span></a> constant on linux 3.6 and greater.

- The <a href="../library/ssl.html#ssl.cert_time_to_seconds" class="reference internal" title="ssl.cert_time_to_seconds"><span class="pre"><code class="sourceCode python">ssl.cert_time_to_seconds()</code></span></a> function now interprets the input time as UTC and not as local time, per <span id="index-51" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc5280.html" class="rfc reference external"><strong>RFC 5280</strong></a>. Additionally, the return value is always an <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>. (Contributed by Akira Li in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19940" class="reference external">bpo-19940</a>.)

- The <span class="pre">`pygettext.py`</span> Tool now uses the standard +NNNN format for timezones in the POT-Creation-Date header.

- The <a href="../library/smtplib.html#module-smtplib" class="reference internal" title="smtplib: SMTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">smtplib</code></span></a> module now uses <a href="../library/sys.html#sys.stderr" class="reference internal" title="sys.stderr"><span class="pre"><code class="sourceCode python">sys.stderr</code></span></a> instead of the previous module-level <span class="pre">`stderr`</span> variable for debug output. If your (test) program depends on patching the module-level variable to capture the debug output, you will need to update it to capture sys.stderr instead.

- The <a href="../library/stdtypes.html#str.startswith" class="reference internal" title="str.startswith"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.startswith()</code></span></a> and <a href="../library/stdtypes.html#str.endswith" class="reference internal" title="str.endswith"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.endswith()</code></span></a> methods no longer return <span class="pre">`True`</span> when finding the empty string and the indexes are completely out of range. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24284" class="reference external">bpo-24284</a>.)

- The <a href="../library/inspect.html#inspect.getdoc" class="reference internal" title="inspect.getdoc"><span class="pre"><code class="sourceCode python">inspect.getdoc()</code></span></a> function now returns documentation strings inherited from base classes. Documentation strings no longer need to be duplicated if the inherited documentation is appropriate. To suppress an inherited string, an empty string must be specified (or the documentation may be filled in). This change affects the output of the <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module and the <a href="../library/functions.html#help" class="reference internal" title="help"><span class="pre"><code class="sourceCode python"><span class="bu">help</span>()</code></span></a> function. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15582" class="reference external">bpo-15582</a>.)

- Nested <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">functools.partial()</code></span></a> calls are now flattened. If you were relying on the previous behavior, you can now either add an attribute to a <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">functools.partial()</code></span></a> object or you can create a subclass of <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">functools.partial()</code></span></a>. (Contributed by Alexander Belopolsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7830" class="reference external">bpo-7830</a>.)

</div>

<div id="changes-in-the-c-api" class="section">

### Changes in the C API<a href="#changes-in-the-c-api" class="headerlink" title="Link to this heading">¶</a>

- The undocumented <span class="pre">`format`</span> member of the (non-public) <span class="pre">`PyMemoryViewObject`</span> structure has been removed. All extensions relying on the relevant parts in <span class="pre">`memoryobject.h`</span> must be rebuilt.

- The <span class="pre">`PyMemAllocator`</span> structure was renamed to <a href="../c-api/memory.html#c.PyMemAllocatorEx" class="reference internal" title="PyMemAllocatorEx"><span class="pre"><code class="sourceCode c">PyMemAllocatorEx</code></span></a> and a new <span class="pre">`calloc`</span> field was added.

- Removed non-documented macro <span class="pre">`PyObject_REPR()`</span> which leaked references. Use format character <span class="pre">`%R`</span> in <a href="../c-api/unicode.html#c.PyUnicode_FromFormat" class="reference internal" title="PyUnicode_FromFormat"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormat<span class="op">()</span></code></span></a>-like functions to format the <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> of the object. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22453" class="reference external">bpo-22453</a>.)

- Because the lack of the <a href="../reference/datamodel.html#type.__module__" class="reference internal" title="type.__module__"><span class="pre"><code class="sourceCode python">__module__</code></span></a> attribute breaks pickling and introspection, a deprecation warning is now raised for builtin types without the <a href="../reference/datamodel.html#type.__module__" class="reference internal" title="type.__module__"><span class="pre"><code class="sourceCode python">__module__</code></span></a> attribute. This will be an <a href="../library/exceptions.html#AttributeError" class="reference internal" title="AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> in the future. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20204" class="reference external">bpo-20204</a>.)

- As part of the <span id="index-52" class="target"></span><a href="https://peps.python.org/pep-0492/" class="pep reference external"><strong>PEP 492</strong></a> implementation, the <span class="pre">`tp_reserved`</span> slot of <a href="../c-api/type.html#c.PyTypeObject" class="reference internal" title="PyTypeObject"><span class="pre"><code class="sourceCode c">PyTypeObject</code></span></a> was replaced with a <a href="../c-api/typeobj.html#c.PyTypeObject.tp_as_async" class="reference internal" title="PyTypeObject.tp_as_async"><span class="pre"><code class="sourceCode c">tp_as_async</code></span></a> slot. Refer to <a href="../c-api/coro.html#coro-objects" class="reference internal"><span class="std std-ref">Coroutine Objects</span></a> for new types, structures and functions.

</div>

</div>

<div id="notable-changes-in-python-3-5-4" class="section">

## Notable changes in Python 3.5.4<a href="#notable-changes-in-python-3-5-4" class="headerlink" title="Link to this heading">¶</a>

<div id="new-make-regen-all-build-target" class="section">

### New <span class="pre">`make`</span>` `<span class="pre">`regen-all`</span> build target<a href="#new-make-regen-all-build-target" class="headerlink" title="Link to this heading">¶</a>

To simplify cross-compilation, and to ensure that CPython can reliably be compiled without requiring an existing version of Python to already be available, the autotools-based build system no longer attempts to implicitly recompile generated files based on file modification times.

Instead, a new <span class="pre">`make`</span>` `<span class="pre">`regen-all`</span> command has been added to force regeneration of these files when desired (e.g. after an initial version of Python has already been built based on the pregenerated versions).

More selective regeneration targets are also defined - see <a href="https://github.com/python/cpython/tree/3.13/Makefile.pre.in" class="extlink-source reference external">Makefile.pre.in</a> for details.

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23404" class="reference external">bpo-23404</a>.)

<div class="versionadded">

<span class="versionmodified added">Added in version 3.5.4.</span>

</div>

</div>

<div id="removal-of-make-touch-build-target" class="section">

### Removal of <span class="pre">`make`</span>` `<span class="pre">`touch`</span> build target<a href="#removal-of-make-touch-build-target" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`make`</span>` `<span class="pre">`touch`</span> build target previously used to request implicit regeneration of generated files by updating their modification times has been removed.

It has been replaced by the new <span class="pre">`make`</span>` `<span class="pre">`regen-all`</span> target.

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23404" class="reference external">bpo-23404</a>.)

<div class="versionchanged">

<span class="versionmodified changed">Changed in version 3.5.4.</span>

</div>

</div>

</div>

</div>

<div class="clearer">

</div>

</div>
