<div class="body" role="main">

<div id="what-s-new-in-python-3-8" class="section">

# What’s New In Python 3.8<a href="#what-s-new-in-python-3-8" class="headerlink" title="Link to this heading">¶</a>

Editor<span class="colon">:</span>  
Raymond Hettinger

This article explains the new features in Python 3.8, compared to 3.7. Python 3.8 was released on October 14, 2019. For full details, see the <a href="changelog.html#changelog" class="reference internal"><span class="std std-ref">changelog</span></a>.

<div id="summary-release-highlights" class="section">

## Summary – Release highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

</div>

<div id="new-features" class="section">

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="assignment-expressions" class="section">

### Assignment expressions<a href="#assignment-expressions" class="headerlink" title="Link to this heading">¶</a>

There is new syntax <span class="pre">`:=`</span> that assigns values to variables as part of a larger expression. It is affectionately known as “the walrus operator” due to its resemblance to <a href="https://en.wikipedia.org/wiki/Walrus#/media/File:Pacific_Walrus_-_Bull_(8247646168).jpg" class="reference external">the eyes and tusks of a walrus</a>.

In this example, the assignment expression helps avoid calling <a href="../library/functions.html#len" class="reference internal" title="len"><span class="pre"><code class="sourceCode python"><span class="bu">len</span>()</code></span></a> twice:

<div class="highlight-python3 notranslate">

<div class="highlight">

    if (n := len(a)) > 10:
        print(f"List is too long ({n} elements, expected <= 10)")

</div>

</div>

A similar benefit arises during regular expression matching where match objects are needed twice, once to test whether a match occurred and another to extract a subgroup:

<div class="highlight-python3 notranslate">

<div class="highlight">

    discount = 0.0
    if (mo := re.search(r'(\d+)% discount', advertisement)):
        discount = float(mo.group(1)) / 100.0

</div>

</div>

The operator is also useful with while-loops that compute a value to test loop termination and then need that same value again in the body of the loop:

<div class="highlight-python3 notranslate">

<div class="highlight">

    # Loop over fixed length blocks
    while (block := f.read(256)) != '':
        process(block)

</div>

</div>

Another motivating use case arises in list comprehensions where a value computed in a filtering condition is also needed in the expression body:

<div class="highlight-python3 notranslate">

<div class="highlight">

    [clean_name.title() for name in names
     if (clean_name := normalize('NFC', name)) in allowed_names]

</div>

</div>

Try to limit use of the walrus operator to clean cases that reduce complexity and improve readability.

See <span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0572/" class="pep reference external"><strong>PEP 572</strong></a> for a full description.

(Contributed by Emily Morehouse in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35224" class="reference external">bpo-35224</a>.)

</div>

<div id="positional-only-parameters" class="section">

### Positional-only parameters<a href="#positional-only-parameters" class="headerlink" title="Link to this heading">¶</a>

There is a new function parameter syntax <span class="pre">`/`</span> to indicate that some function parameters must be specified positionally and cannot be used as keyword arguments. This is the same notation shown by <span class="pre">`help()`</span> for C functions annotated with Larry Hastings’ <a href="https://devguide.python.org/development-tools/clinic/" class="reference external">Argument Clinic</a> tool.

In the following example, parameters *a* and *b* are positional-only, while *c* or *d* can be positional or keyword, and *e* or *f* are required to be keywords:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def f(a, b, /, c, d, *, e, f):
        print(a, b, c, d, e, f)

</div>

</div>

The following is a valid call:

<div class="highlight-python3 notranslate">

<div class="highlight">

    f(10, 20, 30, d=40, e=50, f=60)

</div>

</div>

However, these are invalid calls:

<div class="highlight-python3 notranslate">

<div class="highlight">

    f(10, b=20, c=30, d=40, e=50, f=60)   # b cannot be a keyword argument
    f(10, 20, 30, 40, 50, f=60)           # e must be a keyword argument

</div>

</div>

One use case for this notation is that it allows pure Python functions to fully emulate behaviors of existing C coded functions. For example, the built-in <a href="../library/functions.html#divmod" class="reference internal" title="divmod"><span class="pre"><code class="sourceCode python"><span class="bu">divmod</span>()</code></span></a> function does not accept keyword arguments:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def divmod(a, b, /):
        "Emulate the built in divmod() function"
        return (a // b, a % b)

</div>

</div>

Another use case is to preclude keyword arguments when the parameter name is not helpful. For example, the builtin <a href="../library/functions.html#len" class="reference internal" title="len"><span class="pre"><code class="sourceCode python"><span class="bu">len</span>()</code></span></a> function has the signature <span class="pre">`len(obj,`</span>` `<span class="pre">`/)`</span>. This precludes awkward calls such as:

<div class="highlight-python3 notranslate">

<div class="highlight">

    len(obj='hello')  # The "obj" keyword argument impairs readability

</div>

</div>

A further benefit of marking a parameter as positional-only is that it allows the parameter name to be changed in the future without risk of breaking client code. For example, in the <a href="../library/statistics.html#module-statistics" class="reference internal" title="statistics: Mathematical statistics functions"><span class="pre"><code class="sourceCode python">statistics</code></span></a> module, the parameter name *dist* may be changed in the future. This was made possible with the following function specification:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def quantiles(dist, /, *, n=4, method='exclusive')
        ...

</div>

</div>

Since the parameters to the left of <span class="pre">`/`</span> are not exposed as possible keywords, the parameters names remain available for use in <span class="pre">`**kwargs`</span>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> def f(a, b, /, **kwargs):
    ...     print(a, b, kwargs)
    ...
    >>> f(10, 20, a=1, b=2, c=3)         # a and b are used in two ways
    10 20 {'a': 1, 'b': 2, 'c': 3}

</div>

</div>

This greatly simplifies the implementation of functions and methods that need to accept arbitrary keyword arguments. For example, here is an excerpt from code in the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: Container datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class Counter(dict):

        def __init__(self, iterable=None, /, **kwds):
            # Note "iterable" is a possible keyword argument

</div>

</div>

See <span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0570/" class="pep reference external"><strong>PEP 570</strong></a> for a full description.

(Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36540" class="reference external">bpo-36540</a>.)

</div>

<div id="parallel-filesystem-cache-for-compiled-bytecode-files" class="section">

### Parallel filesystem cache for compiled bytecode files<a href="#parallel-filesystem-cache-for-compiled-bytecode-files" class="headerlink" title="Link to this heading">¶</a>

The new <span id="index-2" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONPYCACHEPREFIX" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONPYCACHEPREFIX</code></span></a> setting (also available as <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span></a> <span class="pre">`pycache_prefix`</span>) configures the implicit bytecode cache to use a separate parallel filesystem tree, rather than the default <span class="pre">`__pycache__`</span> subdirectories within each source directory.

The location of the cache is reported in <a href="../library/sys.html#sys.pycache_prefix" class="reference internal" title="sys.pycache_prefix"><span class="pre"><code class="sourceCode python">sys.pycache_prefix</code></span></a> (<a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a> indicates the default location in <span class="pre">`__pycache__`</span> subdirectories).

(Contributed by Carl Meyer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33499" class="reference external">bpo-33499</a>.)

</div>

<div id="debug-build-uses-the-same-abi-as-release-build" class="section">

### Debug build uses the same ABI as release build<a href="#debug-build-uses-the-same-abi-as-release-build" class="headerlink" title="Link to this heading">¶</a>

Python now uses the same ABI whether it’s built in release or debug mode. On Unix, when Python is built in debug mode, it is now possible to load C extensions built in release mode and C extensions built using the stable ABI.

Release builds and <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">debug builds</span></a> are now ABI compatible: defining the <span class="pre">`Py_DEBUG`</span> macro no longer implies the <span class="pre">`Py_TRACE_REFS`</span> macro, which introduces the only ABI incompatibility. The <span class="pre">`Py_TRACE_REFS`</span> macro, which adds the <a href="../library/sys.html#sys.getobjects" class="reference internal" title="sys.getobjects"><span class="pre"><code class="sourceCode python">sys.getobjects()</code></span></a> function and the <span id="index-3" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONDUMPREFS" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONDUMPREFS</code></span></a> environment variable, can be set using the new <a href="../using/configure.html#cmdoption-with-trace-refs" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">./configure</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">--with-trace-refs</code></span></a> build option. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36465" class="reference external">bpo-36465</a>.)

On Unix, C extensions are no longer linked to libpython except on Android and Cygwin. It is now possible for a statically linked Python to load a C extension built using a shared library Python. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21536" class="reference external">bpo-21536</a>.)

On Unix, when Python is built in debug mode, import now also looks for C extensions compiled in release mode and for C extensions compiled with the stable ABI. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36722" class="reference external">bpo-36722</a>.)

To embed Python into an application, a new <span class="pre">`--embed`</span> option must be passed to <span class="pre">`python3-config`</span>` `<span class="pre">`--libs`</span>` `<span class="pre">`--embed`</span> to get <span class="pre">`-lpython3.8`</span> (link the application to libpython). To support both 3.8 and older, try <span class="pre">`python3-config`</span>` `<span class="pre">`--libs`</span>` `<span class="pre">`--embed`</span> first and fallback to <span class="pre">`python3-config`</span>` `<span class="pre">`--libs`</span> (without <span class="pre">`--embed`</span>) if the previous command fails.

Add a pkg-config <span class="pre">`python-3.8-embed`</span> module to embed Python into an application: <span class="pre">`pkg-config`</span>` `<span class="pre">`python-3.8-embed`</span>` `<span class="pre">`--libs`</span> includes <span class="pre">`-lpython3.8`</span>. To support both 3.8 and older, try <span class="pre">`pkg-config`</span>` `<span class="pre">`python-X.Y-embed`</span>` `<span class="pre">`--libs`</span> first and fallback to <span class="pre">`pkg-config`</span>` `<span class="pre">`python-X.Y`</span>` `<span class="pre">`--libs`</span> (without <span class="pre">`--embed`</span>) if the previous command fails (replace <span class="pre">`X.Y`</span> with the Python version).

On the other hand, <span class="pre">`pkg-config`</span>` `<span class="pre">`python3.8`</span>` `<span class="pre">`--libs`</span> no longer contains <span class="pre">`-lpython3.8`</span>. C extensions must not be linked to libpython (except on Android and Cygwin, whose cases are handled by the script); this change is backward incompatible on purpose. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36721" class="reference external">bpo-36721</a>.)

</div>

<div id="f-strings-support-for-self-documenting-expressions-and-debugging" class="section">

<span id="bpo-36817-whatsnew"></span>

### f-strings support <span class="pre">`=`</span> for self-documenting expressions and debugging<a href="#f-strings-support-for-self-documenting-expressions-and-debugging" class="headerlink" title="Link to this heading">¶</a>

Added an <span class="pre">`=`</span> specifier to <a href="../glossary.html#term-f-string" class="reference internal"><span class="xref std std-term">f-string</span></a>s. An f-string such as <span class="pre">`f'{expr=}'`</span> will expand to the text of the expression, an equal sign, then the representation of the evaluated expression. For example:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> user = 'eric_idle'
    >>> member_since = date(1975, 7, 31)
    >>> f'{user=} {member_since=}'
    "user='eric_idle' member_since=datetime.date(1975, 7, 31)"

</div>

</div>

The usual <a href="../reference/lexical_analysis.html#f-strings" class="reference internal"><span class="std std-ref">f-string format specifiers</span></a> allow more control over how the result of the expression is displayed:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> delta = date.today() - member_since
    >>> f'{user=!s}  {delta.days=:,d}'
    'user=eric_idle  delta.days=16,075'

</div>

</div>

The <span class="pre">`=`</span> specifier will display the whole expression so that calculations can be shown:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> print(f'{theta=}  {cos(radians(theta))=:.3f}')
    theta=30  cos(radians(theta))=0.866

</div>

</div>

(Contributed by Eric V. Smith and Larry Hastings in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36817" class="reference external">bpo-36817</a>.)

</div>

<div id="pep-578-python-runtime-audit-hooks" class="section">

### PEP 578: Python Runtime Audit Hooks<a href="#pep-578-python-runtime-audit-hooks" class="headerlink" title="Link to this heading">¶</a>

The PEP adds an Audit Hook and Verified Open Hook. Both are available from Python and native code, allowing applications and frameworks written in pure Python code to take advantage of extra notifications, while also allowing embedders or system administrators to deploy builds of Python where auditing is always enabled.

See <span id="index-4" class="target"></span><a href="https://peps.python.org/pep-0578/" class="pep reference external"><strong>PEP 578</strong></a> for full details.

</div>

<div id="pep-587-python-initialization-configuration" class="section">

### PEP 587: Python Initialization Configuration<a href="#pep-587-python-initialization-configuration" class="headerlink" title="Link to this heading">¶</a>

The <span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0587/" class="pep reference external"><strong>PEP 587</strong></a> adds a new C API to configure the Python Initialization providing finer control on the whole configuration and better error reporting.

New structures:

- <a href="../c-api/init_config.html#c.PyConfig" class="reference internal" title="PyConfig"><span class="pre"><code class="sourceCode c">PyConfig</code></span></a>

- <a href="../c-api/init_config.html#c.PyPreConfig" class="reference internal" title="PyPreConfig"><span class="pre"><code class="sourceCode c">PyPreConfig</code></span></a>

- <a href="../c-api/init_config.html#c.PyStatus" class="reference internal" title="PyStatus"><span class="pre"><code class="sourceCode c">PyStatus</code></span></a>

- <a href="../c-api/init_config.html#c.PyWideStringList" class="reference internal" title="PyWideStringList"><span class="pre"><code class="sourceCode c">PyWideStringList</code></span></a>

New functions:

- <a href="../c-api/init_config.html#c.PyConfig_Clear" class="reference internal" title="PyConfig_Clear"><span class="pre"><code class="sourceCode c">PyConfig_Clear<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyConfig_InitIsolatedConfig" class="reference internal" title="PyConfig_InitIsolatedConfig"><span class="pre"><code class="sourceCode c">PyConfig_InitIsolatedConfig<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyConfig_InitPythonConfig" class="reference internal" title="PyConfig_InitPythonConfig"><span class="pre"><code class="sourceCode c">PyConfig_InitPythonConfig<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyConfig_Read" class="reference internal" title="PyConfig_Read"><span class="pre"><code class="sourceCode c">PyConfig_Read<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyConfig_SetArgv" class="reference internal" title="PyConfig_SetArgv"><span class="pre"><code class="sourceCode c">PyConfig_SetArgv<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyConfig_SetBytesArgv" class="reference internal" title="PyConfig_SetBytesArgv"><span class="pre"><code class="sourceCode c">PyConfig_SetBytesArgv<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyConfig_SetBytesString" class="reference internal" title="PyConfig_SetBytesString"><span class="pre"><code class="sourceCode c">PyConfig_SetBytesString<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyConfig_SetString" class="reference internal" title="PyConfig_SetString"><span class="pre"><code class="sourceCode c">PyConfig_SetString<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyPreConfig_InitIsolatedConfig" class="reference internal" title="PyPreConfig_InitIsolatedConfig"><span class="pre"><code class="sourceCode c">PyPreConfig_InitIsolatedConfig<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyPreConfig_InitPythonConfig" class="reference internal" title="PyPreConfig_InitPythonConfig"><span class="pre"><code class="sourceCode c">PyPreConfig_InitPythonConfig<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyStatus_Error" class="reference internal" title="PyStatus_Error"><span class="pre"><code class="sourceCode c">PyStatus_Error<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyStatus_Exception" class="reference internal" title="PyStatus_Exception"><span class="pre"><code class="sourceCode c">PyStatus_Exception<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyStatus_Exit" class="reference internal" title="PyStatus_Exit"><span class="pre"><code class="sourceCode c">PyStatus_Exit<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyStatus_IsError" class="reference internal" title="PyStatus_IsError"><span class="pre"><code class="sourceCode c">PyStatus_IsError<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyStatus_IsExit" class="reference internal" title="PyStatus_IsExit"><span class="pre"><code class="sourceCode c">PyStatus_IsExit<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyStatus_NoMemory" class="reference internal" title="PyStatus_NoMemory"><span class="pre"><code class="sourceCode c">PyStatus_NoMemory<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyStatus_Ok" class="reference internal" title="PyStatus_Ok"><span class="pre"><code class="sourceCode c">PyStatus_Ok<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyWideStringList_Append" class="reference internal" title="PyWideStringList_Append"><span class="pre"><code class="sourceCode c">PyWideStringList_Append<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.PyWideStringList_Insert" class="reference internal" title="PyWideStringList_Insert"><span class="pre"><code class="sourceCode c">PyWideStringList_Insert<span class="op">()</span></code></span></a>

- <a href="../c-api/init.html#c.Py_BytesMain" class="reference internal" title="Py_BytesMain"><span class="pre"><code class="sourceCode c">Py_BytesMain<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.Py_ExitStatusException" class="reference internal" title="Py_ExitStatusException"><span class="pre"><code class="sourceCode c">Py_ExitStatusException<span class="op">()</span></code></span></a>

- <a href="../c-api/init.html#c.Py_InitializeFromConfig" class="reference internal" title="Py_InitializeFromConfig"><span class="pre"><code class="sourceCode c">Py_InitializeFromConfig<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.Py_PreInitialize" class="reference internal" title="Py_PreInitialize"><span class="pre"><code class="sourceCode c">Py_PreInitialize<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.Py_PreInitializeFromArgs" class="reference internal" title="Py_PreInitializeFromArgs"><span class="pre"><code class="sourceCode c">Py_PreInitializeFromArgs<span class="op">()</span></code></span></a>

- <a href="../c-api/init_config.html#c.Py_PreInitializeFromBytesArgs" class="reference internal" title="Py_PreInitializeFromBytesArgs"><span class="pre"><code class="sourceCode c">Py_PreInitializeFromBytesArgs<span class="op">()</span></code></span></a>

- <a href="../c-api/init.html#c.Py_RunMain" class="reference internal" title="Py_RunMain"><span class="pre"><code class="sourceCode c">Py_RunMain<span class="op">()</span></code></span></a>

This PEP also adds <span class="pre">`_PyRuntimeState.preconfig`</span> (<a href="../c-api/init_config.html#c.PyPreConfig" class="reference internal" title="PyPreConfig"><span class="pre"><code class="sourceCode c">PyPreConfig</code></span></a> type) and <span class="pre">`PyInterpreterState.config`</span> (<a href="../c-api/init_config.html#c.PyConfig" class="reference internal" title="PyConfig"><span class="pre"><code class="sourceCode c">PyConfig</code></span></a> type) fields to these internal structures. <span class="pre">`PyInterpreterState.config`</span> becomes the new reference configuration, replacing global configuration variables and other private variables.

See <a href="../c-api/init_config.html#init-config" class="reference internal"><span class="std std-ref">Python Initialization Configuration</span></a> for the documentation.

See <span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0587/" class="pep reference external"><strong>PEP 587</strong></a> for a full description.

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36763" class="reference external">bpo-36763</a>.)

</div>

<div id="pep-590-vectorcall-a-fast-calling-protocol-for-cpython" class="section">

### PEP 590: Vectorcall: a fast calling protocol for CPython<a href="#pep-590-vectorcall-a-fast-calling-protocol-for-cpython" class="headerlink" title="Link to this heading">¶</a>

<a href="../c-api/call.html#vectorcall" class="reference internal"><span class="std std-ref">The Vectorcall Protocol</span></a> is added to the Python/C API. It is meant to formalize existing optimizations which were already done for various classes. Any <a href="../c-api/typeobj.html#static-types" class="reference internal"><span class="std std-ref">static type</span></a> implementing a callable can use this protocol.

This is currently provisional. The aim is to make it fully public in Python 3.9.

See <span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0590/" class="pep reference external"><strong>PEP 590</strong></a> for a full description.

(Contributed by Jeroen Demeyer, Mark Shannon and Petr Viktorin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36974" class="reference external">bpo-36974</a>.)

</div>

<div id="pickle-protocol-5-with-out-of-band-data-buffers" class="section">

### Pickle protocol 5 with out-of-band data buffers<a href="#pickle-protocol-5-with-out-of-band-data-buffers" class="headerlink" title="Link to this heading">¶</a>

When <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> is used to transfer large data between Python processes in order to take advantage of multi-core or multi-machine processing, it is important to optimize the transfer by reducing memory copies, and possibly by applying custom techniques such as data-dependent compression.

The <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> protocol 5 introduces support for out-of-band buffers where <span id="index-8" class="target"></span><a href="https://peps.python.org/pep-3118/" class="pep reference external"><strong>PEP 3118</strong></a>-compatible data can be transmitted separately from the main pickle stream, at the discretion of the communication layer.

See <span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0574/" class="pep reference external"><strong>PEP 574</strong></a> for a full description.

(Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36785" class="reference external">bpo-36785</a>.)

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

- A <a href="../reference/simple_stmts.html#continue" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">continue</code></span></a> statement was illegal in the <a href="../reference/compound_stmts.html#finally" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">finally</code></span></a> clause due to a problem with the implementation. In Python 3.8 this restriction was lifted. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32489" class="reference external">bpo-32489</a>.)

- The <a href="../library/functions.html#bool" class="reference internal" title="bool"><span class="pre"><code class="sourceCode python"><span class="bu">bool</span></code></span></a>, <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>, and <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">fractions.Fraction</code></span></a> types now have an <a href="../library/stdtypes.html#int.as_integer_ratio" class="reference internal" title="int.as_integer_ratio"><span class="pre"><code class="sourceCode python">as_integer_ratio()</code></span></a> method like that found in <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> and <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">decimal.Decimal</code></span></a>. This minor API extension makes it possible to write <span class="pre">`numerator,`</span>` `<span class="pre">`denominator`</span>` `<span class="pre">`=`</span>` `<span class="pre">`x.as_integer_ratio()`</span> and have it work across multiple numeric types. (Contributed by Lisa Roach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33073" class="reference external">bpo-33073</a> and Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37819" class="reference external">bpo-37819</a>.)

- Constructors of <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>, <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> and <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a> will now use the <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> special method, if available and the corresponding method <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="fu">__int__</span>()</code></span></a>, <a href="../reference/datamodel.html#object.__float__" class="reference internal" title="object.__float__"><span class="pre"><code class="sourceCode python"><span class="fu">__float__</span>()</code></span></a> or <a href="../reference/datamodel.html#object.__complex__" class="reference internal" title="object.__complex__"><span class="pre"><code class="sourceCode python"><span class="fu">__complex__</span>()</code></span></a> is not available. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20092" class="reference external">bpo-20092</a>.)

- Added support of <span class="pre">`\N{`</span>*<span class="pre">`name`</span>*<span class="pre">`}`</span> escapes in <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">regular</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">expressions</code></span></a>:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> notice = 'Copyright © 2019'
      >>> copyright_year_pattern = re.compile(r'\N{copyright sign}\s*(\d{4})')
      >>> int(copyright_year_pattern.search(notice).group(1))
      2019

  </div>

  </div>

  (Contributed by Jonathan Eunice and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30688" class="reference external">bpo-30688</a>.)

- Dict and dictviews are now iterable in reversed insertion order using <a href="../library/functions.html#reversed" class="reference internal" title="reversed"><span class="pre"><code class="sourceCode python"><span class="bu">reversed</span>()</code></span></a>. (Contributed by Rémi Lapeyre in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33462" class="reference external">bpo-33462</a>.)

- The syntax allowed for keyword names in function calls was further restricted. In particular, <span class="pre">`f((keyword)=arg)`</span> is no longer allowed. It was never intended to permit more than a bare name on the left-hand side of a keyword argument assignment term. (Contributed by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34641" class="reference external">bpo-34641</a>.)

- Generalized iterable unpacking in <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> and <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statements no longer requires enclosing parentheses. This brings the *yield* and *return* syntax into better agreement with normal assignment syntax:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> def parse(family):
              lastname, *members = family.split()
              return lastname.upper(), *members

      >>> parse('simpsons homer marge bart lisa maggie')
      ('SIMPSONS', 'homer', 'marge', 'bart', 'lisa', 'maggie')

  </div>

  </div>

  (Contributed by David Cuthbert and Jordan Chapman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32117" class="reference external">bpo-32117</a>.)

- When a comma is missed in code such as <span class="pre">`[(10,`</span>` `<span class="pre">`20)`</span>` `<span class="pre">`(30,`</span>` `<span class="pre">`40)]`</span>, the compiler displays a <a href="../library/exceptions.html#SyntaxWarning" class="reference internal" title="SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a> with a helpful suggestion. This improves on just having a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> indicating that the first tuple was not callable. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15248" class="reference external">bpo-15248</a>.)

- Arithmetic operations between subclasses of <a href="../library/datetime.html#datetime.date" class="reference internal" title="datetime.date"><span class="pre"><code class="sourceCode python">datetime.date</code></span></a> or <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime.datetime</code></span></a> and <a href="../library/datetime.html#datetime.timedelta" class="reference internal" title="datetime.timedelta"><span class="pre"><code class="sourceCode python">datetime.timedelta</code></span></a> objects now return an instance of the subclass, rather than the base class. This also affects the return type of operations whose implementation (directly or indirectly) uses <a href="../library/datetime.html#datetime.timedelta" class="reference internal" title="datetime.timedelta"><span class="pre"><code class="sourceCode python">datetime.timedelta</code></span></a> arithmetic, such as <a href="../library/datetime.html#datetime.datetime.astimezone" class="reference internal" title="datetime.datetime.astimezone"><span class="pre"><code class="sourceCode python">astimezone()</code></span></a>. (Contributed by Paul Ganssle in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32417" class="reference external">bpo-32417</a>.)

- When the Python interpreter is interrupted by Ctrl-C (SIGINT) and the resulting <a href="../library/exceptions.html#KeyboardInterrupt" class="reference internal" title="KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a> exception is not caught, the Python process now exits via a SIGINT signal or with the correct exit code such that the calling process can detect that it died due to a Ctrl-C. Shells on POSIX and Windows use this to properly terminate scripts in interactive sessions. (Contributed by Google via Gregory P. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1054041" class="reference external">bpo-1054041</a>.)

- Some advanced styles of programming require updating the <a href="../library/types.html#types.CodeType" class="reference internal" title="types.CodeType"><span class="pre"><code class="sourceCode python">types.CodeType</code></span></a> object for an existing function. Since code objects are immutable, a new code object needs to be created, one that is modeled on the existing code object. With 19 parameters, this was somewhat tedious. Now, the new <span class="pre">`replace()`</span> method makes it possible to create a clone with a few altered parameters.

  Here’s an example that alters the <a href="../library/statistics.html#statistics.mean" class="reference internal" title="statistics.mean"><span class="pre"><code class="sourceCode python">statistics.mean()</code></span></a> function to prevent the *data* parameter from being used as a keyword argument:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> from statistics import mean
      >>> mean(data=[10, 20, 90])
      40
      >>> mean.__code__ = mean.__code__.replace(co_posonlyargcount=1)
      >>> mean(data=[10, 20, 90])
      Traceback (most recent call last):
        ...
      TypeError: mean() got some positional-only arguments passed as keyword arguments: 'data'

  </div>

  </div>

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37032" class="reference external">bpo-37032</a>.)

- For integers, the three-argument form of the <a href="../library/functions.html#pow" class="reference internal" title="pow"><span class="pre"><code class="sourceCode python"><span class="bu">pow</span>()</code></span></a> function now permits the exponent to be negative in the case where the base is relatively prime to the modulus. It then computes a modular inverse to the base when the exponent is <span class="pre">`-1`</span>, and a suitable power of that inverse for other negative exponents. For example, to compute the <a href="https://en.wikipedia.org/wiki/Modular_multiplicative_inverse" class="reference external">modular multiplicative inverse</a> of 38 modulo 137, write:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> pow(38, -1, 137)
      119
      >>> 119 * 38 % 137
      1

  </div>

  </div>

  Modular inverses arise in the solution of <a href="https://en.wikipedia.org/wiki/Diophantine_equation" class="reference external">linear Diophantine equations</a>. For example, to find integer solutions for <span class="pre">`4258𝑥`</span>` `<span class="pre">`+`</span>` `<span class="pre">`147𝑦`</span>` `<span class="pre">`=`</span>` `<span class="pre">`369`</span>, first rewrite as <span class="pre">`4258𝑥`</span>` `<span class="pre">`≡`</span>` `<span class="pre">`369`</span>` `<span class="pre">`(mod`</span>` `<span class="pre">`147)`</span> then solve:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> x = 369 * pow(4258, -1, 147) % 147
      >>> y = (4258 * x - 369) // -147
      >>> 4258 * x + 147 * y
      369

  </div>

  </div>

  (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36027" class="reference external">bpo-36027</a>.)

- Dict comprehensions have been synced-up with dict literals so that the key is computed first and the value second:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> # Dict comprehension
      >>> cast = {input('role? '): input('actor? ') for i in range(2)}
      role? King Arthur
      actor? Chapman
      role? Black Knight
      actor? Cleese

      >>> # Dict literal
      >>> cast = {input('role? '): input('actor? ')}
      role? Sir Robin
      actor? Eric Idle

  </div>

  </div>

  The guaranteed execution order is helpful with assignment expressions because variables assigned in the key expression will be available in the value expression:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> names = ['Martin von Löwis', 'Łukasz Langa', 'Walter Dörwald']
      >>> {(n := normalize('NFC', name)).casefold() : n for name in names}
      {'martin von löwis': 'Martin von Löwis',
       'łukasz langa': 'Łukasz Langa',
       'walter dörwald': 'Walter Dörwald'}

  </div>

  </div>

  (Contributed by Jörn Heissler in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35224" class="reference external">bpo-35224</a>.)

- The <a href="../library/pickle.html#object.__reduce__" class="reference internal" title="object.__reduce__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.__reduce__()</code></span></a> method can now return a tuple from two to six elements long. Formerly, five was the limit. The new, optional sixth element is a callable with a <span class="pre">`(obj,`</span>` `<span class="pre">`state)`</span> signature. This allows the direct control over the state-updating behavior of a specific object. If not *None*, this callable will have priority over the object’s <a href="../library/pickle.html#object.__setstate__" class="reference internal" title="object.__setstate__"><span class="pre"><code class="sourceCode python">__setstate__()</code></span></a> method. (Contributed by Pierre Glaser and Olivier Grisel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35900" class="reference external">bpo-35900</a>.)

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

- The new <a href="../library/importlib.metadata.html#module-importlib.metadata" class="reference internal" title="importlib.metadata: Accessing package metadata"><span class="pre"><code class="sourceCode python">importlib.metadata</code></span></a> module provides (provisional) support for reading metadata from third-party packages. For example, it can extract an installed package’s version number, list of entry points, and more:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> # Note following example requires that the popular "requests"
      >>> # package has been installed.
      >>>
      >>> from importlib.metadata import version, requires, files
      >>> version('requests')
      '2.22.0'
      >>> list(requires('requests'))
      ['chardet (<3.1.0,>=3.0.2)']
      >>> list(files('requests'))[:5]
      [PackagePath('requests-2.22.0.dist-info/INSTALLER'),
       PackagePath('requests-2.22.0.dist-info/LICENSE'),
       PackagePath('requests-2.22.0.dist-info/METADATA'),
       PackagePath('requests-2.22.0.dist-info/RECORD'),
       PackagePath('requests-2.22.0.dist-info/WHEEL')]

  </div>

  </div>

  (Contributed by Barry Warsaw and Jason R. Coombs in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34632" class="reference external">bpo-34632</a>.)

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="ast" class="section">

### ast<a href="#ast" class="headerlink" title="Link to this heading">¶</a>

AST nodes now have <span class="pre">`end_lineno`</span> and <span class="pre">`end_col_offset`</span> attributes, which give the precise location of the end of the node. (This only applies to nodes that have <span class="pre">`lineno`</span> and <span class="pre">`col_offset`</span> attributes.)

New function <a href="../library/ast.html#ast.get_source_segment" class="reference internal" title="ast.get_source_segment"><span class="pre"><code class="sourceCode python">ast.get_source_segment()</code></span></a> returns the source code for a specific AST node.

(Contributed by Ivan Levkivskyi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33416" class="reference external">bpo-33416</a>.)

The <a href="../library/ast.html#ast.parse" class="reference internal" title="ast.parse"><span class="pre"><code class="sourceCode python">ast.parse()</code></span></a> function has some new flags:

- <span class="pre">`type_comments=True`</span> causes it to return the text of <span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> and <span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0526/" class="pep reference external"><strong>PEP 526</strong></a> type comments associated with certain AST nodes;

- <span class="pre">`mode='func_type'`</span> can be used to parse <span id="index-12" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> “signature type comments” (returned for function definition AST nodes);

- <span class="pre">`feature_version=(3,`</span>` `<span class="pre">`N)`</span> allows specifying an earlier Python 3 version. For example, <span class="pre">`feature_version=(3,`</span>` `<span class="pre">`4)`</span> will treat <a href="../reference/compound_stmts.html#async" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span></a> and <a href="../reference/expressions.html#await" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">await</code></span></a> as non-reserved words.

(Contributed by Guido van Rossum in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35766" class="reference external">bpo-35766</a>.)

</div>

<div id="asyncio" class="section">

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/asyncio-runner.html#asyncio.run" class="reference internal" title="asyncio.run"><span class="pre"><code class="sourceCode python">asyncio.run()</code></span></a> has graduated from the provisional to stable API. This function can be used to execute a <a href="../glossary.html#term-coroutine" class="reference internal"><span class="xref std std-term">coroutine</span></a> and return the result while automatically managing the event loop. For example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import asyncio

    async def main():
        await asyncio.sleep(0)
        return 42

    asyncio.run(main())

</div>

</div>

This is *roughly* equivalent to:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import asyncio

    async def main():
        await asyncio.sleep(0)
        return 42

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        asyncio.set_event_loop(None)
        loop.close()

</div>

</div>

The actual implementation is significantly more complex. Thus, <a href="../library/asyncio-runner.html#asyncio.run" class="reference internal" title="asyncio.run"><span class="pre"><code class="sourceCode python">asyncio.run()</code></span></a> should be the preferred way of running asyncio programs.

(Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32314" class="reference external">bpo-32314</a>.)

Running <span class="pre">`python`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`asyncio`</span> launches a natively async REPL. This allows rapid experimentation with code that has a top-level <a href="../reference/expressions.html#await" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">await</code></span></a>. There is no longer a need to directly call <span class="pre">`asyncio.run()`</span> which would spawn a new event loop on every invocation:

<div class="highlight-none notranslate">

<div class="highlight">

    $ python -m asyncio
    asyncio REPL 3.8.0
    Use "await" directly instead of "asyncio.run()".
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import asyncio
    >>> await asyncio.sleep(10, result='hello')
    hello

</div>

</div>

(Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37028" class="reference external">bpo-37028</a>.)

The exception <a href="../library/asyncio-exceptions.html#asyncio.CancelledError" class="reference internal" title="asyncio.CancelledError"><span class="pre"><code class="sourceCode python">asyncio.CancelledError</code></span></a> now inherits from <a href="../library/exceptions.html#BaseException" class="reference internal" title="BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a> rather than <a href="../library/exceptions.html#Exception" class="reference internal" title="Exception"><span class="pre"><code class="sourceCode python"><span class="pp">Exception</span></code></span></a> and no longer inherits from <a href="../library/concurrent.futures.html#concurrent.futures.CancelledError" class="reference internal" title="concurrent.futures.CancelledError"><span class="pre"><code class="sourceCode python">concurrent.futures.CancelledError</code></span></a>. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32528" class="reference external">bpo-32528</a>.)

On Windows, the default event loop is now <a href="../library/asyncio-eventloop.html#asyncio.ProactorEventLoop" class="reference internal" title="asyncio.ProactorEventLoop"><span class="pre"><code class="sourceCode python">ProactorEventLoop</code></span></a>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34687" class="reference external">bpo-34687</a>.)

<a href="../library/asyncio-eventloop.html#asyncio.ProactorEventLoop" class="reference internal" title="asyncio.ProactorEventLoop"><span class="pre"><code class="sourceCode python">ProactorEventLoop</code></span></a> now also supports UDP. (Contributed by Adam Meily and Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29883" class="reference external">bpo-29883</a>.)

<a href="../library/asyncio-eventloop.html#asyncio.ProactorEventLoop" class="reference internal" title="asyncio.ProactorEventLoop"><span class="pre"><code class="sourceCode python">ProactorEventLoop</code></span></a> can now be interrupted by <a href="../library/exceptions.html#KeyboardInterrupt" class="reference internal" title="KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a> (“CTRL+C”). (Contributed by Vladimir Matveev in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23057" class="reference external">bpo-23057</a>.)

Added <a href="../library/asyncio-task.html#asyncio.Task.get_coro" class="reference internal" title="asyncio.Task.get_coro"><span class="pre"><code class="sourceCode python">asyncio.Task.get_coro()</code></span></a> for getting the wrapped coroutine within an <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">asyncio.Task</code></span></a>. (Contributed by Alex Grönholm in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36999" class="reference external">bpo-36999</a>.)

Asyncio tasks can now be named, either by passing the <span class="pre">`name`</span> keyword argument to <a href="../library/asyncio-task.html#asyncio.create_task" class="reference internal" title="asyncio.create_task"><span class="pre"><code class="sourceCode python">asyncio.create_task()</code></span></a> or the <a href="../library/asyncio-eventloop.html#asyncio.loop.create_task" class="reference internal" title="asyncio.loop.create_task"><span class="pre"><code class="sourceCode python">create_task()</code></span></a> event loop method, or by calling the <a href="../library/asyncio-task.html#asyncio.Task.set_name" class="reference internal" title="asyncio.Task.set_name"><span class="pre"><code class="sourceCode python">set_name()</code></span></a> method on the task object. The task name is visible in the <span class="pre">`repr()`</span> output of <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">asyncio.Task</code></span></a> and can also be retrieved using the <a href="../library/asyncio-task.html#asyncio.Task.get_name" class="reference internal" title="asyncio.Task.get_name"><span class="pre"><code class="sourceCode python">get_name()</code></span></a> method. (Contributed by Alex Grönholm in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34270" class="reference external">bpo-34270</a>.)

Added support for <a href="https://en.wikipedia.org/wiki/Happy_Eyeballs" class="reference external">Happy Eyeballs</a> to <a href="../library/asyncio-eventloop.html#asyncio.loop.create_connection" class="reference internal" title="asyncio.loop.create_connection"><span class="pre"><code class="sourceCode python">asyncio.loop.create_connection()</code></span></a>. To specify the behavior, two new parameters have been added: *happy_eyeballs_delay* and *interleave*. The Happy Eyeballs algorithm improves responsiveness in applications that support IPv4 and IPv6 by attempting to simultaneously connect using both. (Contributed by twisteroid ambassador in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33530" class="reference external">bpo-33530</a>.)

</div>

<div id="builtins" class="section">

### builtins<a href="#builtins" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a> built-in has been improved to accept the <span class="pre">`ast.PyCF_ALLOW_TOP_LEVEL_AWAIT`</span> flag. With this new flag passed, <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a> will allow top-level <span class="pre">`await`</span>, <span class="pre">`async`</span>` `<span class="pre">`for`</span> and <span class="pre">`async`</span>` `<span class="pre">`with`</span> constructs that are usually considered invalid syntax. Asynchronous code object marked with the <span class="pre">`CO_COROUTINE`</span> flag may then be returned. (Contributed by Matthias Bussonnier in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34616" class="reference external">bpo-34616</a>)

</div>

<div id="collections" class="section">

### collections<a href="#collections" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/collections.html#collections.somenamedtuple._asdict" class="reference internal" title="collections.somenamedtuple._asdict"><span class="pre"><code class="sourceCode python">_asdict()</code></span></a> method for <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">collections.namedtuple()</code></span></a> now returns a <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> instead of a <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">collections.OrderedDict</code></span></a>. This works because regular dicts have guaranteed ordering since Python 3.7. If the extra features of <span class="pre">`OrderedDict`</span> are required, the suggested remediation is to cast the result to the desired type: <span class="pre">`OrderedDict(nt._asdict())`</span>. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35864" class="reference external">bpo-35864</a>.)

</div>

<div id="cprofile" class="section">

### cProfile<a href="#cprofile" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/profile.html#profile.Profile" class="reference internal" title="profile.Profile"><span class="pre"><code class="sourceCode python">cProfile.Profile</code></span></a> class can now be used as a context manager. Profile a block of code by running:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import cProfile

    with cProfile.Profile() as profiler:
          # code to be profiled
          ...

</div>

</div>

(Contributed by Scott Sanderson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29235" class="reference external">bpo-29235</a>.)

</div>

<div id="csv" class="section">

### csv<a href="#csv" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/csv.html#csv.DictReader" class="reference internal" title="csv.DictReader"><span class="pre"><code class="sourceCode python">csv.DictReader</code></span></a> now returns instances of <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> instead of a <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">collections.OrderedDict</code></span></a>. The tool is now faster and uses less memory while still preserving the field order. (Contributed by Michael Selik in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34003" class="reference external">bpo-34003</a>.)

</div>

<div id="curses" class="section">

### curses<a href="#curses" class="headerlink" title="Link to this heading">¶</a>

Added a new variable holding structured version information for the underlying ncurses library: <a href="../library/curses.html#curses.ncurses_version" class="reference internal" title="curses.ncurses_version"><span class="pre"><code class="sourceCode python">ncurses_version</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31680" class="reference external">bpo-31680</a>.)

</div>

<div id="ctypes" class="section">

### ctypes<a href="#ctypes" class="headerlink" title="Link to this heading">¶</a>

On Windows, <a href="../library/ctypes.html#ctypes.CDLL" class="reference internal" title="ctypes.CDLL"><span class="pre"><code class="sourceCode python">CDLL</code></span></a> and subclasses now accept a *winmode* parameter to specify flags for the underlying <span class="pre">`LoadLibraryEx`</span> call. The default flags are set to only load DLL dependencies from trusted locations, including the path where the DLL is stored (if a full or partial path is used to load the initial DLL) and paths added by <a href="../library/os.html#os.add_dll_directory" class="reference internal" title="os.add_dll_directory"><span class="pre"><code class="sourceCode python">add_dll_directory()</code></span></a>. (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36085" class="reference external">bpo-36085</a>.)

</div>

<div id="datetime" class="section">

### datetime<a href="#datetime" class="headerlink" title="Link to this heading">¶</a>

Added new alternate constructors <a href="../library/datetime.html#datetime.date.fromisocalendar" class="reference internal" title="datetime.date.fromisocalendar"><span class="pre"><code class="sourceCode python">datetime.date.fromisocalendar()</code></span></a> and <a href="../library/datetime.html#datetime.datetime.fromisocalendar" class="reference internal" title="datetime.datetime.fromisocalendar"><span class="pre"><code class="sourceCode python">datetime.datetime.fromisocalendar()</code></span></a>, which construct <a href="../library/datetime.html#datetime.date" class="reference internal" title="datetime.date"><span class="pre"><code class="sourceCode python">date</code></span></a> and <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> objects respectively from ISO year, week number, and weekday; these are the inverse of each class’s <span class="pre">`isocalendar`</span> method. (Contributed by Paul Ganssle in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36004" class="reference external">bpo-36004</a>.)

</div>

<div id="functools" class="section">

### functools<a href="#functools" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/functools.html#functools.lru_cache" class="reference internal" title="functools.lru_cache"><span class="pre"><code class="sourceCode python">functools.lru_cache()</code></span></a> can now be used as a straight decorator rather than as a function returning a decorator. So both of these are now supported:

<div class="highlight-python3 notranslate">

<div class="highlight">

    @lru_cache
    def f(x):
        ...

    @lru_cache(maxsize=256)
    def f(x):
        ...

</div>

</div>

(Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36772" class="reference external">bpo-36772</a>.)

Added a new <a href="../library/functools.html#functools.cached_property" class="reference internal" title="functools.cached_property"><span class="pre"><code class="sourceCode python">functools.cached_property()</code></span></a> decorator, for computed properties cached for the life of the instance.

<div class="highlight-python3 notranslate">

<div class="highlight">

    import functools
    import statistics

    class Dataset:
       def __init__(self, sequence_of_numbers):
          self.data = sequence_of_numbers

       @functools.cached_property
       def variance(self):
          return statistics.variance(self.data)

</div>

</div>

(Contributed by Carl Meyer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21145" class="reference external">bpo-21145</a>)

Added a new <a href="../library/functools.html#functools.singledispatchmethod" class="reference internal" title="functools.singledispatchmethod"><span class="pre"><code class="sourceCode python">functools.singledispatchmethod()</code></span></a> decorator that converts methods into <a href="../glossary.html#term-generic-function" class="reference internal"><span class="xref std std-term">generic functions</span></a> using <a href="../glossary.html#term-single-dispatch" class="reference internal"><span class="xref std std-term">single dispatch</span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    from functools import singledispatchmethod
    from contextlib import suppress

    class TaskManager:

        def __init__(self, tasks):
            self.tasks = list(tasks)

        @singledispatchmethod
        def discard(self, value):
            with suppress(ValueError):
                self.tasks.remove(value)

        @discard.register(list)
        def _(self, tasks):
            targets = set(tasks)
            self.tasks = [x for x in self.tasks if x not in targets]

</div>

</div>

(Contributed by Ethan Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32380" class="reference external">bpo-32380</a>)

</div>

<div id="gc" class="section">

### gc<a href="#gc" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/gc.html#gc.get_objects" class="reference internal" title="gc.get_objects"><span class="pre"><code class="sourceCode python">get_objects()</code></span></a> can now receive an optional *generation* parameter indicating a generation to get objects from. (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36016" class="reference external">bpo-36016</a>.)

</div>

<div id="gettext" class="section">

### gettext<a href="#gettext" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/gettext.html#gettext.pgettext" class="reference internal" title="gettext.pgettext"><span class="pre"><code class="sourceCode python">pgettext()</code></span></a> and its variants. (Contributed by Franz Glasner, Éric Araujo, and Cheryl Sabella in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2504" class="reference external">bpo-2504</a>.)

</div>

<div id="gzip" class="section">

### gzip<a href="#gzip" class="headerlink" title="Link to this heading">¶</a>

Added the *mtime* parameter to <a href="../library/gzip.html#gzip.compress" class="reference internal" title="gzip.compress"><span class="pre"><code class="sourceCode python">gzip.compress()</code></span></a> for reproducible output. (Contributed by Guo Ci Teo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34898" class="reference external">bpo-34898</a>.)

A <a href="../library/gzip.html#gzip.BadGzipFile" class="reference internal" title="gzip.BadGzipFile"><span class="pre"><code class="sourceCode python">BadGzipFile</code></span></a> exception is now raised instead of <a href="../library/exceptions.html#OSError" class="reference internal" title="OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a> for certain types of invalid or corrupt gzip files. (Contributed by Filip Gruszczyński, Michele Orrù, and Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6584" class="reference external">bpo-6584</a>.)

</div>

<div id="idle-and-idlelib" class="section">

### IDLE and idlelib<a href="#idle-and-idlelib" class="headerlink" title="Link to this heading">¶</a>

Output over N lines (50 by default) is squeezed down to a button. N can be changed in the PyShell section of the General page of the Settings dialog. Fewer, but possibly extra long, lines can be squeezed by right clicking on the output. Squeezed output can be expanded in place by double-clicking the button or into the clipboard or a separate window by right-clicking the button. (Contributed by Tal Einat in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1529353" class="reference external">bpo-1529353</a>.)

Add “Run Customized” to the Run menu to run a module with customized settings. Any command line arguments entered are added to sys.argv. They also re-appear in the box for the next customized run. One can also suppress the normal Shell main module restart. (Contributed by Cheryl Sabella, Terry Jan Reedy, and others in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5680" class="reference external">bpo-5680</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37627" class="reference external">bpo-37627</a>.)

Added optional line numbers for IDLE editor windows. Windows open without line numbers unless set otherwise in the General tab of the configuration dialog. Line numbers for an existing window are shown and hidden in the Options menu. (Contributed by Tal Einat and Saimadhav Heblikar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17535" class="reference external">bpo-17535</a>.)

OS native encoding is now used for converting between Python strings and Tcl objects. This allows IDLE to work with emoji and other non-BMP characters. These characters can be displayed or copied and pasted to or from the clipboard. Converting strings from Tcl to Python and back now never fails. (Many people worked on this for eight years but the problem was finally solved by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13153" class="reference external">bpo-13153</a>.)

New in 3.8.1:

Add option to toggle cursor blink off. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4603" class="reference external">bpo-4603</a>.)

Escape key now closes IDLE completion windows. (Contributed by Johnny Najera in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38944" class="reference external">bpo-38944</a>.)

The changes above have been backported to 3.7 maintenance releases.

Add keywords to module name completion list. (Contributed by Terry J. Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37765" class="reference external">bpo-37765</a>.)

</div>

<div id="inspect" class="section">

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/inspect.html#inspect.getdoc" class="reference internal" title="inspect.getdoc"><span class="pre"><code class="sourceCode python">inspect.getdoc()</code></span></a> function can now find docstrings for <span class="pre">`__slots__`</span> if that attribute is a <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> where the values are docstrings. This provides documentation options similar to what we already have for <a href="../library/functions.html#property" class="reference internal" title="property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span>()</code></span></a>, <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span>()</code></span></a>, and <a href="../library/functions.html#staticmethod" class="reference internal" title="staticmethod"><span class="pre"><code class="sourceCode python"><span class="bu">staticmethod</span>()</code></span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class AudioClip:
        __slots__ = {'bit_rate': 'expressed in kilohertz to one decimal place',
                     'duration': 'in seconds, rounded up to an integer'}
        def __init__(self, bit_rate, duration):
            self.bit_rate = round(bit_rate / 1000.0, 1)
            self.duration = ceil(duration)

</div>

</div>

(Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36326" class="reference external">bpo-36326</a>.)

</div>

<div id="io" class="section">

### io<a href="#io" class="headerlink" title="Link to this heading">¶</a>

In development mode (<a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span></a> <span class="pre">`env`</span>) and in <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">debug build</span></a>, the <a href="../library/io.html#io.IOBase" class="reference internal" title="io.IOBase"><span class="pre"><code class="sourceCode python">io.IOBase</code></span></a> finalizer now logs the exception if the <span class="pre">`close()`</span> method fails. The exception is ignored silently by default in release build. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18748" class="reference external">bpo-18748</a>.)

</div>

<div id="itertools" class="section">

### itertools<a href="#itertools" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/itertools.html#itertools.accumulate" class="reference internal" title="itertools.accumulate"><span class="pre"><code class="sourceCode python">itertools.accumulate()</code></span></a> function added an option *initial* keyword argument to specify an initial value:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from itertools import accumulate
    >>> list(accumulate([10, 5, 30, 15], initial=1000))
    [1000, 1010, 1015, 1045, 1060]

</div>

</div>

(Contributed by Lisa Roach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34659" class="reference external">bpo-34659</a>.)

</div>

<div id="json-tool" class="section">

### json.tool<a href="#json-tool" class="headerlink" title="Link to this heading">¶</a>

Add option <span class="pre">`--json-lines`</span> to parse every input line as a separate JSON object. (Contributed by Weipeng Hong in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31553" class="reference external">bpo-31553</a>.)

</div>

<div id="logging" class="section">

### logging<a href="#logging" class="headerlink" title="Link to this heading">¶</a>

Added a *force* keyword argument to <a href="../library/logging.html#logging.basicConfig" class="reference internal" title="logging.basicConfig"><span class="pre"><code class="sourceCode python">logging.basicConfig()</code></span></a>. When set to true, any existing handlers attached to the root logger are removed and closed before carrying out the configuration specified by the other arguments.

This solves a long-standing problem. Once a logger or *basicConfig()* had been called, subsequent calls to *basicConfig()* were silently ignored. This made it difficult to update, experiment with, or teach the various logging configuration options using the interactive prompt or a Jupyter notebook.

(Suggested by Raymond Hettinger, implemented by Donghee Na, and reviewed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33897" class="reference external">bpo-33897</a>.)

</div>

<div id="math" class="section">

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

Added new function <a href="../library/math.html#math.dist" class="reference internal" title="math.dist"><span class="pre"><code class="sourceCode python">math.dist()</code></span></a> for computing Euclidean distance between two points. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33089" class="reference external">bpo-33089</a>.)

Expanded the <a href="../library/math.html#math.hypot" class="reference internal" title="math.hypot"><span class="pre"><code class="sourceCode python">math.hypot()</code></span></a> function to handle multiple dimensions. Formerly, it only supported the 2-D case. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33089" class="reference external">bpo-33089</a>.)

Added new function, <a href="../library/math.html#math.prod" class="reference internal" title="math.prod"><span class="pre"><code class="sourceCode python">math.prod()</code></span></a>, as analogous function to <a href="../library/functions.html#sum" class="reference internal" title="sum"><span class="pre"><code class="sourceCode python"><span class="bu">sum</span>()</code></span></a> that returns the product of a ‘start’ value (default: 1) times an iterable of numbers:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> prior = 0.8
    >>> likelihoods = [0.625, 0.84, 0.30]
    >>> math.prod(likelihoods, start=prior)
    0.126

</div>

</div>

(Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35606" class="reference external">bpo-35606</a>.)

Added two new combinatoric functions <a href="../library/math.html#math.perm" class="reference internal" title="math.perm"><span class="pre"><code class="sourceCode python">math.perm()</code></span></a> and <a href="../library/math.html#math.comb" class="reference internal" title="math.comb"><span class="pre"><code class="sourceCode python">math.comb()</code></span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> math.perm(10, 3)    # Permutations of 10 things taken 3 at a time
    720
    >>> math.comb(10, 3)    # Combinations of 10 things taken 3 at a time
    120

</div>

</div>

(Contributed by Yash Aggarwal, Keller Fuchs, Serhiy Storchaka, and Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37128" class="reference external">bpo-37128</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37178" class="reference external">bpo-37178</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35431" class="reference external">bpo-35431</a>.)

Added a new function <a href="../library/math.html#math.isqrt" class="reference internal" title="math.isqrt"><span class="pre"><code class="sourceCode python">math.isqrt()</code></span></a> for computing accurate integer square roots without conversion to floating point. The new function supports arbitrarily large integers. It is faster than <span class="pre">`floor(sqrt(n))`</span> but slower than <a href="../library/math.html#math.sqrt" class="reference internal" title="math.sqrt"><span class="pre"><code class="sourceCode python">math.sqrt()</code></span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> r = 650320427
    >>> s = r ** 2
    >>> isqrt(s - 1)         # correct
    650320426
    >>> floor(sqrt(s - 1))   # incorrect
    650320427

</div>

</div>

(Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36887" class="reference external">bpo-36887</a>.)

The function <a href="../library/math.html#math.factorial" class="reference internal" title="math.factorial"><span class="pre"><code class="sourceCode python">math.factorial()</code></span></a> no longer accepts arguments that are not int-like. (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33083" class="reference external">bpo-33083</a>.)

</div>

<div id="mmap" class="section">

### mmap<a href="#mmap" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/mmap.html#mmap.mmap" class="reference internal" title="mmap.mmap"><span class="pre"><code class="sourceCode python">mmap.mmap</code></span></a> class now has an <a href="../library/mmap.html#mmap.mmap.madvise" class="reference internal" title="mmap.mmap.madvise"><span class="pre"><code class="sourceCode python">madvise()</code></span></a> method to access the <span class="pre">`madvise()`</span> system call. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32941" class="reference external">bpo-32941</a>.)

</div>

<div id="multiprocessing" class="section">

### multiprocessing<a href="#multiprocessing" class="headerlink" title="Link to this heading">¶</a>

Added new <a href="../library/multiprocessing.shared_memory.html#module-multiprocessing.shared_memory" class="reference internal" title="multiprocessing.shared_memory: Provides shared memory for direct access across processes."><span class="pre"><code class="sourceCode python">multiprocessing.shared_memory</code></span></a> module. (Contributed by Davin Potts in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35813" class="reference external">bpo-35813</a>.)

On macOS, the *spawn* start method is now used by default. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33725" class="reference external">bpo-33725</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

Added new function <a href="../library/os.html#os.add_dll_directory" class="reference internal" title="os.add_dll_directory"><span class="pre"><code class="sourceCode python">add_dll_directory()</code></span></a> on Windows for providing additional search paths for native dependencies when importing extension modules or loading DLLs using <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a>. (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36085" class="reference external">bpo-36085</a>.)

A new <a href="../library/os.html#os.memfd_create" class="reference internal" title="os.memfd_create"><span class="pre"><code class="sourceCode python">os.memfd_create()</code></span></a> function was added to wrap the <span class="pre">`memfd_create()`</span> syscall. (Contributed by Zackery Spytz and Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26836" class="reference external">bpo-26836</a>.)

On Windows, much of the manual logic for handling reparse points (including symlinks and directory junctions) has been delegated to the operating system. Specifically, <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> will now traverse anything supported by the operating system, while <a href="../library/os.html#os.lstat" class="reference internal" title="os.lstat"><span class="pre"><code class="sourceCode python">os.lstat()</code></span></a> will only open reparse points that identify as “name surrogates” while others are opened as for <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a>. In all cases, <a href="../library/os.html#os.stat_result.st_mode" class="reference internal" title="os.stat_result.st_mode"><span class="pre"><code class="sourceCode python">os.stat_result.st_mode</code></span></a> will only have <span class="pre">`S_IFLNK`</span> set for symbolic links and not other kinds of reparse points. To identify other kinds of reparse point, check the new <a href="../library/os.html#os.stat_result.st_reparse_tag" class="reference internal" title="os.stat_result.st_reparse_tag"><span class="pre"><code class="sourceCode python">os.stat_result.st_reparse_tag</code></span></a> attribute.

On Windows, <a href="../library/os.html#os.readlink" class="reference internal" title="os.readlink"><span class="pre"><code class="sourceCode python">os.readlink()</code></span></a> is now able to read directory junctions. Note that <a href="../library/os.path.html#os.path.islink" class="reference internal" title="os.path.islink"><span class="pre"><code class="sourceCode python">islink()</code></span></a> will return <span class="pre">`False`</span> for directory junctions, and so code that checks <span class="pre">`islink`</span> first will continue to treat junctions as directories, while code that handles errors from <a href="../library/os.html#os.readlink" class="reference internal" title="os.readlink"><span class="pre"><code class="sourceCode python">os.readlink()</code></span></a> may now treat junctions as links.

(Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37834" class="reference external">bpo-37834</a>.)

</div>

<div id="os-path" class="section">

### os.path<a href="#os-path" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/os.path.html#module-os.path" class="reference internal" title="os.path: Operations on pathnames."><span class="pre"><code class="sourceCode python">os.path</code></span></a> functions that return a boolean result like <a href="../library/os.path.html#os.path.exists" class="reference internal" title="os.path.exists"><span class="pre"><code class="sourceCode python">exists()</code></span></a>, <a href="../library/os.path.html#os.path.lexists" class="reference internal" title="os.path.lexists"><span class="pre"><code class="sourceCode python">lexists()</code></span></a>, <a href="../library/os.path.html#os.path.isdir" class="reference internal" title="os.path.isdir"><span class="pre"><code class="sourceCode python">isdir()</code></span></a>, <a href="../library/os.path.html#os.path.isfile" class="reference internal" title="os.path.isfile"><span class="pre"><code class="sourceCode python">isfile()</code></span></a>, <a href="../library/os.path.html#os.path.islink" class="reference internal" title="os.path.islink"><span class="pre"><code class="sourceCode python">islink()</code></span></a>, and <a href="../library/os.path.html#os.path.ismount" class="reference internal" title="os.path.ismount"><span class="pre"><code class="sourceCode python">ismount()</code></span></a> now return <span class="pre">`False`</span> instead of raising <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> or its subclasses <a href="../library/exceptions.html#UnicodeEncodeError" class="reference internal" title="UnicodeEncodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeEncodeError</span></code></span></a> and <a href="../library/exceptions.html#UnicodeDecodeError" class="reference internal" title="UnicodeDecodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeDecodeError</span></code></span></a> for paths that contain characters or bytes unrepresentable at the OS level. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33721" class="reference external">bpo-33721</a>.)

<a href="../library/os.path.html#os.path.expanduser" class="reference internal" title="os.path.expanduser"><span class="pre"><code class="sourceCode python">expanduser()</code></span></a> on Windows now prefers the <span id="index-13" class="target"></span><span class="pre">`USERPROFILE`</span> environment variable and does not use <span id="index-14" class="target"></span><span class="pre">`HOME`</span>, which is not normally set for regular user accounts. (Contributed by Anthony Sottile in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36264" class="reference external">bpo-36264</a>.)

<a href="../library/os.path.html#os.path.isdir" class="reference internal" title="os.path.isdir"><span class="pre"><code class="sourceCode python">isdir()</code></span></a> on Windows no longer returns <span class="pre">`True`</span> for a link to a non-existent directory.

<a href="../library/os.path.html#os.path.realpath" class="reference internal" title="os.path.realpath"><span class="pre"><code class="sourceCode python">realpath()</code></span></a> on Windows now resolves reparse points, including symlinks and directory junctions.

(Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37834" class="reference external">bpo-37834</a>.)

</div>

<div id="pathlib" class="section">

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pathlib.html#pathlib.Path" class="reference internal" title="pathlib.Path"><span class="pre"><code class="sourceCode python">pathlib.Path</code></span></a> methods that return a boolean result like <a href="../library/pathlib.html#pathlib.Path.exists" class="reference internal" title="pathlib.Path.exists"><span class="pre"><code class="sourceCode python">exists()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.is_dir" class="reference internal" title="pathlib.Path.is_dir"><span class="pre"><code class="sourceCode python">is_dir()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.is_file" class="reference internal" title="pathlib.Path.is_file"><span class="pre"><code class="sourceCode python">is_file()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.is_mount" class="reference internal" title="pathlib.Path.is_mount"><span class="pre"><code class="sourceCode python">is_mount()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.is_symlink" class="reference internal" title="pathlib.Path.is_symlink"><span class="pre"><code class="sourceCode python">is_symlink()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.is_block_device" class="reference internal" title="pathlib.Path.is_block_device"><span class="pre"><code class="sourceCode python">is_block_device()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.is_char_device" class="reference internal" title="pathlib.Path.is_char_device"><span class="pre"><code class="sourceCode python">is_char_device()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.is_fifo" class="reference internal" title="pathlib.Path.is_fifo"><span class="pre"><code class="sourceCode python">is_fifo()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.is_socket" class="reference internal" title="pathlib.Path.is_socket"><span class="pre"><code class="sourceCode python">is_socket()</code></span></a> now return <span class="pre">`False`</span> instead of raising <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> or its subclass <a href="../library/exceptions.html#UnicodeEncodeError" class="reference internal" title="UnicodeEncodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeEncodeError</span></code></span></a> for paths that contain characters unrepresentable at the OS level. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33721" class="reference external">bpo-33721</a>.)

Added <span class="pre">`pathlib.Path.link_to()`</span> which creates a hard link pointing to a path. (Contributed by Joannah Nanjekye in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26978" class="reference external">bpo-26978</a>) Note that <span class="pre">`link_to`</span> was deprecated in 3.10 and removed in 3.12 in favor of a <span class="pre">`hardlink_to`</span> method added in 3.10 which matches the semantics of the existing <span class="pre">`symlink_to`</span> method.

</div>

<div id="pickle" class="section">

### pickle<a href="#pickle" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> extensions subclassing the C-optimized <a href="../library/pickle.html#pickle.Pickler" class="reference internal" title="pickle.Pickler"><span class="pre"><code class="sourceCode python">Pickler</code></span></a> can now override the pickling logic of functions and classes by defining the special <a href="../library/pickle.html#pickle.Pickler.reducer_override" class="reference internal" title="pickle.Pickler.reducer_override"><span class="pre"><code class="sourceCode python">reducer_override()</code></span></a> method. (Contributed by Pierre Glaser and Olivier Grisel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35900" class="reference external">bpo-35900</a>.)

</div>

<div id="plistlib" class="section">

### plistlib<a href="#plistlib" class="headerlink" title="Link to this heading">¶</a>

Added new <a href="../library/plistlib.html#plistlib.UID" class="reference internal" title="plistlib.UID"><span class="pre"><code class="sourceCode python">plistlib.UID</code></span></a> and enabled support for reading and writing NSKeyedArchiver-encoded binary plists. (Contributed by Jon Janzen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26707" class="reference external">bpo-26707</a>.)

</div>

<div id="pprint" class="section">

### pprint<a href="#pprint" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/pprint.html#module-pprint" class="reference internal" title="pprint: Data pretty printer."><span class="pre"><code class="sourceCode python">pprint</code></span></a> module added a *sort_dicts* parameter to several functions. By default, those functions continue to sort dictionaries before rendering or printing. However, if *sort_dicts* is set to false, the dictionaries retain the order that keys were inserted. This can be useful for comparison to JSON inputs during debugging.

In addition, there is a convenience new function, <a href="../library/pprint.html#pprint.pp" class="reference internal" title="pprint.pp"><span class="pre"><code class="sourceCode python">pprint.pp()</code></span></a> that is like <a href="../library/pprint.html#pprint.pprint" class="reference internal" title="pprint.pprint"><span class="pre"><code class="sourceCode python">pprint.pprint()</code></span></a> but with *sort_dicts* defaulting to <span class="pre">`False`</span>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from pprint import pprint, pp
    >>> d = dict(source='input.txt', operation='filter', destination='output.txt')
    >>> pp(d, width=40)                  # Original order
    {'source': 'input.txt',
     'operation': 'filter',
     'destination': 'output.txt'}
    >>> pprint(d, width=40)              # Keys sorted alphabetically
    {'destination': 'output.txt',
     'operation': 'filter',
     'source': 'input.txt'}

</div>

</div>

(Contributed by Rémi Lapeyre in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30670" class="reference external">bpo-30670</a>.)

</div>

<div id="py-compile" class="section">

### py_compile<a href="#py-compile" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/py_compile.html#py_compile.compile" class="reference internal" title="py_compile.compile"><span class="pre"><code class="sourceCode python">py_compile.<span class="bu">compile</span>()</code></span></a> now supports silent mode. (Contributed by Joannah Nanjekye in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22640" class="reference external">bpo-22640</a>.)

</div>

<div id="shlex" class="section">

### shlex<a href="#shlex" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/shlex.html#shlex.join" class="reference internal" title="shlex.join"><span class="pre"><code class="sourceCode python">shlex.join()</code></span></a> function acts as the inverse of <a href="../library/shlex.html#shlex.split" class="reference internal" title="shlex.split"><span class="pre"><code class="sourceCode python">shlex.split()</code></span></a>. (Contributed by Bo Bayles in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32102" class="reference external">bpo-32102</a>.)

</div>

<div id="shutil" class="section">

### shutil<a href="#shutil" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/shutil.html#shutil.copytree" class="reference internal" title="shutil.copytree"><span class="pre"><code class="sourceCode python">shutil.copytree()</code></span></a> now accepts a new <span class="pre">`dirs_exist_ok`</span> keyword argument. (Contributed by Josh Bronson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20849" class="reference external">bpo-20849</a>.)

<a href="../library/shutil.html#shutil.make_archive" class="reference internal" title="shutil.make_archive"><span class="pre"><code class="sourceCode python">shutil.make_archive()</code></span></a> now defaults to the modern pax (POSIX.1-2001) format for new archives to improve portability and standards conformance, inherited from the corresponding change to the <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module. (Contributed by C.A.M. Gerlach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30661" class="reference external">bpo-30661</a>.)

<a href="../library/shutil.html#shutil.rmtree" class="reference internal" title="shutil.rmtree"><span class="pre"><code class="sourceCode python">shutil.rmtree()</code></span></a> on Windows now removes directory junctions without recursively removing their contents first. (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37834" class="reference external">bpo-37834</a>.)

</div>

<div id="socket" class="section">

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/socket.html#socket.create_server" class="reference internal" title="socket.create_server"><span class="pre"><code class="sourceCode python">create_server()</code></span></a> and <a href="../library/socket.html#socket.has_dualstack_ipv6" class="reference internal" title="socket.has_dualstack_ipv6"><span class="pre"><code class="sourceCode python">has_dualstack_ipv6()</code></span></a> convenience functions to automate the necessary tasks usually involved when creating a server socket, including accepting both IPv4 and IPv6 connections on the same socket. (Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17561" class="reference external">bpo-17561</a>.)

The <a href="../library/socket.html#socket.if_nameindex" class="reference internal" title="socket.if_nameindex"><span class="pre"><code class="sourceCode python">socket.if_nameindex()</code></span></a>, <a href="../library/socket.html#socket.if_nametoindex" class="reference internal" title="socket.if_nametoindex"><span class="pre"><code class="sourceCode python">socket.if_nametoindex()</code></span></a>, and <a href="../library/socket.html#socket.if_indextoname" class="reference internal" title="socket.if_indextoname"><span class="pre"><code class="sourceCode python">socket.if_indextoname()</code></span></a> functions have been implemented on Windows. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37007" class="reference external">bpo-37007</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/ssl.html#ssl.SSLContext.post_handshake_auth" class="reference internal" title="ssl.SSLContext.post_handshake_auth"><span class="pre"><code class="sourceCode python">post_handshake_auth</code></span></a> to enable and <a href="../library/ssl.html#ssl.SSLSocket.verify_client_post_handshake" class="reference internal" title="ssl.SSLSocket.verify_client_post_handshake"><span class="pre"><code class="sourceCode python">verify_client_post_handshake()</code></span></a> to initiate TLS 1.3 post-handshake authentication. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34670" class="reference external">bpo-34670</a>.)

</div>

<div id="statistics" class="section">

### statistics<a href="#statistics" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/statistics.html#statistics.fmean" class="reference internal" title="statistics.fmean"><span class="pre"><code class="sourceCode python">statistics.fmean()</code></span></a> as a faster, floating-point variant of <a href="../library/statistics.html#statistics.mean" class="reference internal" title="statistics.mean"><span class="pre"><code class="sourceCode python">statistics.mean()</code></span></a>. (Contributed by Raymond Hettinger and Steven D’Aprano in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35904" class="reference external">bpo-35904</a>.)

Added <a href="../library/statistics.html#statistics.geometric_mean" class="reference internal" title="statistics.geometric_mean"><span class="pre"><code class="sourceCode python">statistics.geometric_mean()</code></span></a> (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27181" class="reference external">bpo-27181</a>.)

Added <a href="../library/statistics.html#statistics.multimode" class="reference internal" title="statistics.multimode"><span class="pre"><code class="sourceCode python">statistics.multimode()</code></span></a> that returns a list of the most common values. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35892" class="reference external">bpo-35892</a>.)

Added <a href="../library/statistics.html#statistics.quantiles" class="reference internal" title="statistics.quantiles"><span class="pre"><code class="sourceCode python">statistics.quantiles()</code></span></a> that divides data or a distribution in to equiprobable intervals (e.g. quartiles, deciles, or percentiles). (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36546" class="reference external">bpo-36546</a>.)

Added <a href="../library/statistics.html#statistics.NormalDist" class="reference internal" title="statistics.NormalDist"><span class="pre"><code class="sourceCode python">statistics.NormalDist</code></span></a>, a tool for creating and manipulating normal distributions of a random variable. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36018" class="reference external">bpo-36018</a>.)

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> temperature_feb = NormalDist.from_samples([4, 12, -3, 2, 7, 14])
    >>> temperature_feb.mean
    6.0
    >>> temperature_feb.stdev
    6.356099432828281

    >>> temperature_feb.cdf(3)            # Chance of being under 3 degrees
    0.3184678262814532
    >>> # Relative chance of being 7 degrees versus 10 degrees
    >>> temperature_feb.pdf(7) / temperature_feb.pdf(10)
    1.2039930378537762

    >>> el_niño = NormalDist(4, 2.5)
    >>> temperature_feb += el_niño        # Add in a climate effect
    >>> temperature_feb
    NormalDist(mu=10.0, sigma=6.830080526611674)

    >>> temperature_feb * (9/5) + 32      # Convert to Fahrenheit
    NormalDist(mu=50.0, sigma=12.294144947901014)
    >>> temperature_feb.samples(3)        # Generate random samples
    [7.672102882379219, 12.000027119750287, 4.647488369766392]

</div>

</div>

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

Add new <a href="../library/sys.html#sys.unraisablehook" class="reference internal" title="sys.unraisablehook"><span class="pre"><code class="sourceCode python">sys.unraisablehook()</code></span></a> function which can be overridden to control how “unraisable exceptions” are handled. It is called when an exception has occurred but there is no way for Python to handle it. For example, when a destructor raises an exception or during garbage collection (<a href="../library/gc.html#gc.collect" class="reference internal" title="gc.collect"><span class="pre"><code class="sourceCode python">gc.collect()</code></span></a>). (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36829" class="reference external">bpo-36829</a>.)

</div>

<div id="tarfile" class="section">

### tarfile<a href="#tarfile" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module now defaults to the modern pax (POSIX.1-2001) format for new archives, instead of the previous GNU-specific one. This improves cross-platform portability with a consistent encoding (UTF-8) in a standardized and extensible format, and offers several other benefits. (Contributed by C.A.M. Gerlach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36268" class="reference external">bpo-36268</a>.)

</div>

<div id="threading" class="section">

### threading<a href="#threading" class="headerlink" title="Link to this heading">¶</a>

Add a new <a href="../library/threading.html#threading.excepthook" class="reference internal" title="threading.excepthook"><span class="pre"><code class="sourceCode python">threading.excepthook()</code></span></a> function which handles uncaught <a href="../library/threading.html#threading.Thread.run" class="reference internal" title="threading.Thread.run"><span class="pre"><code class="sourceCode python">threading.Thread.run()</code></span></a> exception. It can be overridden to control how uncaught <a href="../library/threading.html#threading.Thread.run" class="reference internal" title="threading.Thread.run"><span class="pre"><code class="sourceCode python">threading.Thread.run()</code></span></a> exceptions are handled. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1230540" class="reference external">bpo-1230540</a>.)

Add a new <a href="../library/threading.html#threading.get_native_id" class="reference internal" title="threading.get_native_id"><span class="pre"><code class="sourceCode python">threading.get_native_id()</code></span></a> function and a <a href="../library/threading.html#threading.Thread.native_id" class="reference internal" title="threading.Thread.native_id"><span class="pre"><code class="sourceCode python">native_id</code></span></a> attribute to the <a href="../library/threading.html#threading.Thread" class="reference internal" title="threading.Thread"><span class="pre"><code class="sourceCode python">threading.Thread</code></span></a> class. These return the native integral Thread ID of the current thread assigned by the kernel. This feature is only available on certain platforms, see <a href="../library/threading.html#threading.get_native_id" class="reference internal" title="threading.get_native_id"><span class="pre"><code class="sourceCode python">get_native_id</code></span></a> for more information. (Contributed by Jake Tesler in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36084" class="reference external">bpo-36084</a>.)

</div>

<div id="tokenize" class="section">

### tokenize<a href="#tokenize" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/tokenize.html#module-tokenize" class="reference internal" title="tokenize: Lexical scanner for Python source code."><span class="pre"><code class="sourceCode python">tokenize</code></span></a> module now implicitly emits a <span class="pre">`NEWLINE`</span> token when provided with input that does not have a trailing new line. This behavior now matches what the C tokenizer does internally. (Contributed by Ammar Askar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33899" class="reference external">bpo-33899</a>.)

</div>

<div id="tkinter" class="section">

### tkinter<a href="#tkinter" class="headerlink" title="Link to this heading">¶</a>

Added methods <span class="pre">`selection_from()`</span>, <span class="pre">`selection_present()`</span>, <span class="pre">`selection_range()`</span> and <span class="pre">`selection_to()`</span> in the <span class="pre">`tkinter.Spinbox`</span> class. (Contributed by Juliette Monsel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34829" class="reference external">bpo-34829</a>.)

Added method <span class="pre">`moveto()`</span> in the <span class="pre">`tkinter.Canvas`</span> class. (Contributed by Juliette Monsel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23831" class="reference external">bpo-23831</a>.)

The <span class="pre">`tkinter.PhotoImage`</span> class now has <span class="pre">`transparency_get()`</span> and <span class="pre">`transparency_set()`</span> methods. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25451" class="reference external">bpo-25451</a>.)

</div>

<div id="time" class="section">

### time<a href="#time" class="headerlink" title="Link to this heading">¶</a>

Added new clock <a href="../library/time.html#time.CLOCK_UPTIME_RAW" class="reference internal" title="time.CLOCK_UPTIME_RAW"><span class="pre"><code class="sourceCode python">CLOCK_UPTIME_RAW</code></span></a> for macOS 10.12. (Contributed by Joannah Nanjekye in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35702" class="reference external">bpo-35702</a>.)

</div>

<div id="typing" class="section">

### typing<a href="#typing" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module incorporates several new features:

- A dictionary type with per-key types. See <span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0589/" class="pep reference external"><strong>PEP 589</strong></a> and <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">typing.TypedDict</code></span></a>. TypedDict uses only string keys. By default, every key is required to be present. Specify “total=False” to allow keys to be optional:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      class Location(TypedDict, total=False):
          lat_long: tuple
          grid_square: str
          xy_coordinate: tuple

  </div>

  </div>

- Literal types. See <span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0586/" class="pep reference external"><strong>PEP 586</strong></a> and <a href="../library/typing.html#typing.Literal" class="reference internal" title="typing.Literal"><span class="pre"><code class="sourceCode python">typing.Literal</code></span></a>. Literal types indicate that a parameter or return value is constrained to one or more specific literal values:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      def get_status(port: int) -> Literal['connected', 'disconnected']:
          ...

  </div>

  </div>

- “Final” variables, functions, methods and classes. See <span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0591/" class="pep reference external"><strong>PEP 591</strong></a>, <a href="../library/typing.html#typing.Final" class="reference internal" title="typing.Final"><span class="pre"><code class="sourceCode python">typing.Final</code></span></a> and <a href="../library/typing.html#typing.final" class="reference internal" title="typing.final"><span class="pre"><code class="sourceCode python">typing.final()</code></span></a>. The final qualifier instructs a static type checker to restrict subclassing, overriding, or reassignment:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      pi: Final[float] = 3.1415926536

  </div>

  </div>

- Protocol definitions. See <span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0544/" class="pep reference external"><strong>PEP 544</strong></a>, <a href="../library/typing.html#typing.Protocol" class="reference internal" title="typing.Protocol"><span class="pre"><code class="sourceCode python">typing.Protocol</code></span></a> and <a href="../library/typing.html#typing.runtime_checkable" class="reference internal" title="typing.runtime_checkable"><span class="pre"><code class="sourceCode python">typing.runtime_checkable()</code></span></a>. Simple ABCs like <a href="../library/typing.html#typing.SupportsInt" class="reference internal" title="typing.SupportsInt"><span class="pre"><code class="sourceCode python">typing.SupportsInt</code></span></a> are now <span class="pre">`Protocol`</span> subclasses.

- New protocol class <a href="../library/typing.html#typing.SupportsIndex" class="reference internal" title="typing.SupportsIndex"><span class="pre"><code class="sourceCode python">typing.SupportsIndex</code></span></a>.

- New functions <a href="../library/typing.html#typing.get_origin" class="reference internal" title="typing.get_origin"><span class="pre"><code class="sourceCode python">typing.get_origin()</code></span></a> and <a href="../library/typing.html#typing.get_args" class="reference internal" title="typing.get_args"><span class="pre"><code class="sourceCode python">typing.get_args()</code></span></a>.

</div>

<div id="unicodedata" class="section">

### unicodedata<a href="#unicodedata" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a> module has been upgraded to use the <a href="https://blog.unicode.org/2019/05/unicode-12-1-en.html" class="reference external">Unicode 12.1.0</a> release.

New function <a href="../library/unicodedata.html#unicodedata.is_normalized" class="reference internal" title="unicodedata.is_normalized"><span class="pre"><code class="sourceCode python">is_normalized()</code></span></a> can be used to verify a string is in a specific normal form, often much faster than by actually normalizing the string. (Contributed by Max Belanger, David Euresti, and Greg Price in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32285" class="reference external">bpo-32285</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37966" class="reference external">bpo-37966</a>).

</div>

<div id="unittest" class="section">

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/unittest.mock.html#unittest.mock.AsyncMock" class="reference internal" title="unittest.mock.AsyncMock"><span class="pre"><code class="sourceCode python">AsyncMock</code></span></a> to support an asynchronous version of <a href="../library/unittest.mock.html#unittest.mock.Mock" class="reference internal" title="unittest.mock.Mock"><span class="pre"><code class="sourceCode python">Mock</code></span></a>. Appropriate new assert functions for testing have been added as well. (Contributed by Lisa Roach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26467" class="reference external">bpo-26467</a>).

Added <a href="../library/unittest.html#unittest.addModuleCleanup" class="reference internal" title="unittest.addModuleCleanup"><span class="pre"><code class="sourceCode python">addModuleCleanup()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.addClassCleanup" class="reference internal" title="unittest.TestCase.addClassCleanup"><span class="pre"><code class="sourceCode python">addClassCleanup()</code></span></a> to unittest to support cleanups for <a href="../library/unittest.html#unittest.setUpModule" class="reference internal" title="unittest.setUpModule"><span class="pre"><code class="sourceCode python">setUpModule()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.setUpClass" class="reference internal" title="unittest.TestCase.setUpClass"><span class="pre"><code class="sourceCode python">setUpClass()</code></span></a>. (Contributed by Lisa Roach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24412" class="reference external">bpo-24412</a>.)

Several mock assert functions now also print a list of actual calls upon failure. (Contributed by Petter Strandmark in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35047" class="reference external">bpo-35047</a>.)

<a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> module gained support for coroutines to be used as test cases with <a href="../library/unittest.html#unittest.IsolatedAsyncioTestCase" class="reference internal" title="unittest.IsolatedAsyncioTestCase"><span class="pre"><code class="sourceCode python">unittest.IsolatedAsyncioTestCase</code></span></a>. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32972" class="reference external">bpo-32972</a>.)

Example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import unittest


    class TestRequest(unittest.IsolatedAsyncioTestCase):

        async def asyncSetUp(self):
            self.connection = await AsyncConnection()

        async def test_get(self):
            response = await self.connection.get("https://example.com")
            self.assertEqual(response.status_code, 200)

        async def asyncTearDown(self):
            await self.connection.close()


    if __name__ == "__main__":
        unittest.main()

</div>

</div>

</div>

<div id="venv" class="section">

### venv<a href="#venv" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> now includes an <span class="pre">`Activate.ps1`</span> script on all platforms for activating virtual environments under PowerShell Core 6.1. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32718" class="reference external">bpo-32718</a>.)

</div>

<div id="weakref" class="section">

### weakref<a href="#weakref" class="headerlink" title="Link to this heading">¶</a>

The proxy objects returned by <a href="../library/weakref.html#weakref.proxy" class="reference internal" title="weakref.proxy"><span class="pre"><code class="sourceCode python">weakref.proxy()</code></span></a> now support the matrix multiplication operators <span class="pre">`@`</span> and <span class="pre">`@=`</span> in addition to the other numeric operators. (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36669" class="reference external">bpo-36669</a>.)

</div>

<div id="xml" class="section">

### xml<a href="#xml" class="headerlink" title="Link to this heading">¶</a>

As mitigation against DTD and external entity retrieval, the <a href="../library/xml.dom.minidom.html#module-xml.dom.minidom" class="reference internal" title="xml.dom.minidom: Minimal Document Object Model (DOM) implementation."><span class="pre"><code class="sourceCode python">xml.dom.minidom</code></span></a> and <a href="../library/xml.sax.html#module-xml.sax" class="reference internal" title="xml.sax: Package containing SAX2 base classes and convenience functions."><span class="pre"><code class="sourceCode python">xml.sax</code></span></a> modules no longer process external entities by default. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17239" class="reference external">bpo-17239</a>.)

The <span class="pre">`.find*()`</span> methods in the <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> module support wildcard searches like <span class="pre">`{*}tag`</span> which ignores the namespace and <span class="pre">`{namespace}*`</span> which returns all tags in the given namespace. (Contributed by Stefan Behnel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28238" class="reference external">bpo-28238</a>.)

The <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> module provides a new function <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.canonicalize" class="reference internal" title="xml.etree.ElementTree.canonicalize"><span class="pre"><code class="sourceCode python">canonicalize()</code></span></a> that implements C14N 2.0. (Contributed by Stefan Behnel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13611" class="reference external">bpo-13611</a>.)

The target object of <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLParser" class="reference internal" title="xml.etree.ElementTree.XMLParser"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.XMLParser</code></span></a> can receive namespace declaration events through the new callback methods <span class="pre">`start_ns()`</span> and <span class="pre">`end_ns()`</span>. Additionally, the <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.TreeBuilder" class="reference internal" title="xml.etree.ElementTree.TreeBuilder"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.TreeBuilder</code></span></a> target can be configured to process events about comments and processing instructions to include them in the generated tree. (Contributed by Stefan Behnel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36676" class="reference external">bpo-36676</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36673" class="reference external">bpo-36673</a>.)

</div>

<div id="xmlrpc" class="section">

### xmlrpc<a href="#xmlrpc" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/xmlrpc.client.html#xmlrpc.client.ServerProxy" class="reference internal" title="xmlrpc.client.ServerProxy"><span class="pre"><code class="sourceCode python">xmlrpc.client.ServerProxy</code></span></a> now supports an optional *headers* keyword argument for a sequence of HTTP headers to be sent with each request. Among other things, this makes it possible to upgrade from default basic authentication to faster session authentication. (Contributed by Cédric Krier in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35153" class="reference external">bpo-35153</a>.)

</div>

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module can now use the <a href="../library/os.html#os.posix_spawn" class="reference internal" title="os.posix_spawn"><span class="pre"><code class="sourceCode python">os.posix_spawn()</code></span></a> function in some cases for better performance. Currently, it is only used on macOS and Linux (using glibc 2.24 or newer) if all these conditions are met:

  - *close_fds* is false;

  - *preexec_fn*, *pass_fds*, *cwd* and *start_new_session* parameters are not set;

  - the *executable* path contains a directory.

  (Contributed by Joannah Nanjekye and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35537" class="reference external">bpo-35537</a>.)

- <a href="../library/shutil.html#shutil.copyfile" class="reference internal" title="shutil.copyfile"><span class="pre"><code class="sourceCode python">shutil.copyfile()</code></span></a>, <a href="../library/shutil.html#shutil.copy" class="reference internal" title="shutil.copy"><span class="pre"><code class="sourceCode python">shutil.copy()</code></span></a>, <a href="../library/shutil.html#shutil.copy2" class="reference internal" title="shutil.copy2"><span class="pre"><code class="sourceCode python">shutil.copy2()</code></span></a>, <a href="../library/shutil.html#shutil.copytree" class="reference internal" title="shutil.copytree"><span class="pre"><code class="sourceCode python">shutil.copytree()</code></span></a> and <a href="../library/shutil.html#shutil.move" class="reference internal" title="shutil.move"><span class="pre"><code class="sourceCode python">shutil.move()</code></span></a> use platform-specific “fast-copy” syscalls on Linux and macOS in order to copy the file more efficiently. “fast-copy” means that the copying operation occurs within the kernel, avoiding the use of userspace buffers in Python as in “<span class="pre">`outfd.write(infd.read())`</span>”. On Windows <a href="../library/shutil.html#shutil.copyfile" class="reference internal" title="shutil.copyfile"><span class="pre"><code class="sourceCode python">shutil.copyfile()</code></span></a> uses a bigger default buffer size (1 MiB instead of 16 KiB) and a <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span>()</code></span></a>-based variant of <a href="../library/shutil.html#shutil.copyfileobj" class="reference internal" title="shutil.copyfileobj"><span class="pre"><code class="sourceCode python">shutil.copyfileobj()</code></span></a> is used. The speedup for copying a 512 MiB file within the same partition is about +26% on Linux, +50% on macOS and +40% on Windows. Also, much less CPU cycles are consumed. See <a href="../library/shutil.html#shutil-platform-dependent-efficient-copy-operations" class="reference internal"><span class="std std-ref">Platform-dependent efficient copy operations</span></a> section. (Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33671" class="reference external">bpo-33671</a>.)

- <a href="../library/shutil.html#shutil.copytree" class="reference internal" title="shutil.copytree"><span class="pre"><code class="sourceCode python">shutil.copytree()</code></span></a> uses <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a> function and all copy functions depending from it use cached <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> values. The speedup for copying a directory with 8000 files is around +9% on Linux, +20% on Windows and +30% on a Windows SMB share. Also the number of <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> syscalls is reduced by 38% making <a href="../library/shutil.html#shutil.copytree" class="reference internal" title="shutil.copytree"><span class="pre"><code class="sourceCode python">shutil.copytree()</code></span></a> especially faster on network filesystems. (Contributed by Giampaolo Rodolà in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33695" class="reference external">bpo-33695</a>.)

- The default protocol in the <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> module is now Protocol 4, first introduced in Python 3.4. It offers better performance and smaller size compared to Protocol 3 available since Python 3.0.

- Removed one <a href="../c-api/intro.html#c.Py_ssize_t" class="reference internal" title="Py_ssize_t"><span class="pre"><code class="sourceCode c">Py_ssize_t</code></span></a> member from <span class="pre">`PyGC_Head`</span>. All GC tracked objects (e.g. tuple, list, dict) size is reduced 4 or 8 bytes. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33597" class="reference external">bpo-33597</a>.)

- <a href="../library/uuid.html#uuid.UUID" class="reference internal" title="uuid.UUID"><span class="pre"><code class="sourceCode python">uuid.UUID</code></span></a> now uses <span class="pre">`__slots__`</span> to reduce its memory footprint. (Contributed by Wouter Bolsterlee and Tal Einat in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30977" class="reference external">bpo-30977</a>)

- Improved performance of <a href="../library/operator.html#operator.itemgetter" class="reference internal" title="operator.itemgetter"><span class="pre"><code class="sourceCode python">operator.itemgetter()</code></span></a> by 33%. Optimized argument handling and added a fast path for the common case of a single non-negative integer index into a tuple (which is the typical use case in the standard library). (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35664" class="reference external">bpo-35664</a>.)

- Sped-up field lookups in <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">collections.namedtuple()</code></span></a>. They are now more than two times faster, making them the fastest form of instance variable lookup in Python. (Contributed by Raymond Hettinger, Pablo Galindo, and Joe Jevnik, Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32492" class="reference external">bpo-32492</a>.)

- The <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a> constructor does not overallocate the internal item buffer if the input iterable has a known length (the input implements <span class="pre">`__len__`</span>). This makes the created list 12% smaller on average. (Contributed by Raymond Hettinger and Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33234" class="reference external">bpo-33234</a>.)

- Doubled the speed of class variable writes. When a non-dunder attribute was updated, there was an unnecessary call to update slots. (Contributed by Stefan Behnel, Pablo Galindo Salgado, Raymond Hettinger, Neil Schemenauer, and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36012" class="reference external">bpo-36012</a>.)

- Reduced an overhead of converting arguments passed to many builtin functions and methods. This sped up calling some simple builtin functions and methods up to 20–50%. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23867" class="reference external">bpo-23867</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35582" class="reference external">bpo-35582</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36127" class="reference external">bpo-36127</a>.)

- <span class="pre">`LOAD_GLOBAL`</span> instruction now uses new “per opcode cache” mechanism. It is about 40% faster now. (Contributed by Yury Selivanov and Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26219" class="reference external">bpo-26219</a>.)

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Link to this heading">¶</a>

- Default <a href="../library/sys.html#sys.abiflags" class="reference internal" title="sys.abiflags"><span class="pre"><code class="sourceCode python">sys.abiflags</code></span></a> became an empty string: the <span class="pre">`m`</span> flag for pymalloc became useless (builds with and without pymalloc are ABI compatible) and so has been removed. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36707" class="reference external">bpo-36707</a>.)

  Example of changes:

  - Only <span class="pre">`python3.8`</span> program is installed, <span class="pre">`python3.8m`</span> program is gone.

  - Only <span class="pre">`python3.8-config`</span> script is installed, <span class="pre">`python3.8m-config`</span> script is gone.

  - The <span class="pre">`m`</span> flag has been removed from the suffix of dynamic library filenames: extension modules in the standard library as well as those produced and installed by third-party packages, like those downloaded from PyPI. On Linux, for example, the Python 3.7 suffix <span class="pre">`.cpython-37m-x86_64-linux-gnu.so`</span> became <span class="pre">`.cpython-38-x86_64-linux-gnu.so`</span> in Python 3.8.

- The header files have been reorganized to better separate the different kinds of APIs:

  - <span class="pre">`Include/*.h`</span> should be the portable public stable C API.

  - <span class="pre">`Include/cpython/*.h`</span> should be the unstable C API specific to CPython; public API, with some private API prefixed by <span class="pre">`_Py`</span> or <span class="pre">`_PY`</span>.

  - <span class="pre">`Include/internal/*.h`</span> is the private internal C API very specific to CPython. This API comes with no backward compatibility warranty and should not be used outside CPython. It is only exposed for very specific needs like debuggers and profiles which has to access to CPython internals without calling functions. This API is now installed by <span class="pre">`make`</span>` `<span class="pre">`install`</span>.

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35134" class="reference external">bpo-35134</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35081" class="reference external">bpo-35081</a>, work initiated by Eric Snow in Python 3.7.)

- Some macros have been converted to static inline functions: parameter types and return type are well defined, they don’t have issues specific to macros, variables have a local scopes. Examples:

  - <a href="../c-api/refcounting.html#c.Py_INCREF" class="reference internal" title="Py_INCREF"><span class="pre"><code class="sourceCode c">Py_INCREF<span class="op">()</span></code></span></a>, <a href="../c-api/refcounting.html#c.Py_DECREF" class="reference internal" title="Py_DECREF"><span class="pre"><code class="sourceCode c">Py_DECREF<span class="op">()</span></code></span></a>

  - <a href="../c-api/refcounting.html#c.Py_XINCREF" class="reference internal" title="Py_XINCREF"><span class="pre"><code class="sourceCode c">Py_XINCREF<span class="op">()</span></code></span></a>, <a href="../c-api/refcounting.html#c.Py_XDECREF" class="reference internal" title="Py_XDECREF"><span class="pre"><code class="sourceCode c">Py_XDECREF<span class="op">()</span></code></span></a>

  - <span class="pre">`PyObject_INIT`</span>, <span class="pre">`PyObject_INIT_VAR`</span>

  - Private functions: <span class="pre">`_PyObject_GC_TRACK()`</span>, <span class="pre">`_PyObject_GC_UNTRACK()`</span>, <span class="pre">`_Py_Dealloc()`</span>

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35059" class="reference external">bpo-35059</a>.)

- The <span class="pre">`PyByteArray_Init()`</span> and <span class="pre">`PyByteArray_Fini()`</span> functions have been removed. They did nothing since Python 2.7.4 and Python 3.2.0, were excluded from the limited API (stable ABI), and were not documented. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35713" class="reference external">bpo-35713</a>.)

- The result of <a href="../c-api/exceptions.html#c.PyExceptionClass_Name" class="reference internal" title="PyExceptionClass_Name"><span class="pre"><code class="sourceCode c">PyExceptionClass_Name<span class="op">()</span></code></span></a> is now of type <span class="pre">`const`</span>` `<span class="pre">`char`</span>` `<span class="pre">`*`</span> rather of <span class="pre">`char`</span>` `<span class="pre">`*`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33818" class="reference external">bpo-33818</a>.)

- The duality of <span class="pre">`Modules/Setup.dist`</span> and <span class="pre">`Modules/Setup`</span> has been removed. Previously, when updating the CPython source tree, one had to manually copy <span class="pre">`Modules/Setup.dist`</span> (inside the source tree) to <span class="pre">`Modules/Setup`</span> (inside the build tree) in order to reflect any changes upstream. This was of a small benefit to packagers at the expense of a frequent annoyance to developers following CPython development, as forgetting to copy the file could produce build failures.

  Now the build system always reads from <span class="pre">`Modules/Setup`</span> inside the source tree. People who want to customize that file are encouraged to maintain their changes in a git fork of CPython or as patch files, as they would do for any other change to the source tree.

  (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32430" class="reference external">bpo-32430</a>.)

- Functions that convert Python number to C integer like <a href="../c-api/long.html#c.PyLong_AsLong" class="reference internal" title="PyLong_AsLong"><span class="pre"><code class="sourceCode c">PyLong_AsLong<span class="op">()</span></code></span></a> and argument parsing functions like <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple<span class="op">()</span></code></span></a> with integer converting format units like <span class="pre">`'i'`</span> will now use the <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> special method instead of <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="fu">__int__</span>()</code></span></a>, if available. The deprecation warning will be emitted for objects with the <span class="pre">`__int__()`</span> method but without the <span class="pre">`__index__()`</span> method (like <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> and <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">Fraction</code></span></a>). <a href="../c-api/number.html#c.PyNumber_Check" class="reference internal" title="PyNumber_Check"><span class="pre"><code class="sourceCode c">PyNumber_Check<span class="op">()</span></code></span></a> will now return <span class="pre">`1`</span> for objects implementing <span class="pre">`__index__()`</span>. <a href="../c-api/number.html#c.PyNumber_Long" class="reference internal" title="PyNumber_Long"><span class="pre"><code class="sourceCode c">PyNumber_Long<span class="op">()</span></code></span></a>, <a href="../c-api/number.html#c.PyNumber_Float" class="reference internal" title="PyNumber_Float"><span class="pre"><code class="sourceCode c">PyNumber_Float<span class="op">()</span></code></span></a> and <a href="../c-api/float.html#c.PyFloat_AsDouble" class="reference internal" title="PyFloat_AsDouble"><span class="pre"><code class="sourceCode c">PyFloat_AsDouble<span class="op">()</span></code></span></a> also now use the <span class="pre">`__index__()`</span> method if available. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36048" class="reference external">bpo-36048</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20092" class="reference external">bpo-20092</a>.)

- Heap-allocated type objects will now increase their reference count in <a href="../c-api/allocation.html#c.PyObject_Init" class="reference internal" title="PyObject_Init"><span class="pre"><code class="sourceCode c">PyObject_Init<span class="op">()</span></code></span></a> (and its parallel macro <span class="pre">`PyObject_INIT`</span>) instead of in <a href="../c-api/type.html#c.PyType_GenericAlloc" class="reference internal" title="PyType_GenericAlloc"><span class="pre"><code class="sourceCode c">PyType_GenericAlloc<span class="op">()</span></code></span></a>. Types that modify instance allocation or deallocation may need to be adjusted. (Contributed by Eddie Elizondo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35810" class="reference external">bpo-35810</a>.)

- The new function <span class="pre">`PyCode_NewWithPosOnlyArgs()`</span> allows to create code objects like <span class="pre">`PyCode_New()`</span>, but with an extra *posonlyargcount* parameter for indicating the number of positional-only arguments. (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37221" class="reference external">bpo-37221</a>.)

- <span class="pre">`Py_SetPath()`</span> now sets <a href="../library/sys.html#sys.executable" class="reference internal" title="sys.executable"><span class="pre"><code class="sourceCode python">sys.executable</code></span></a> to the program full path (<a href="../c-api/init.html#c.Py_GetProgramFullPath" class="reference internal" title="Py_GetProgramFullPath"><span class="pre"><code class="sourceCode c">Py_GetProgramFullPath<span class="op">()</span></code></span></a>) rather than to the program name (<a href="../c-api/init.html#c.Py_GetProgramName" class="reference internal" title="Py_GetProgramName"><span class="pre"><code class="sourceCode c">Py_GetProgramName<span class="op">()</span></code></span></a>). (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38234" class="reference external">bpo-38234</a>.)

</div>

<div id="deprecated" class="section">

## Deprecated<a href="#deprecated" class="headerlink" title="Link to this heading">¶</a>

- The distutils <span class="pre">`bdist_wininst`</span> command is now deprecated, use <span class="pre">`bdist_wheel`</span> (wheel packages) instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37481" class="reference external">bpo-37481</a>.)

- Deprecated methods <span class="pre">`getchildren()`</span> and <span class="pre">`getiterator()`</span> in the <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">ElementTree</code></span></a> module now emit a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> instead of <a href="../library/exceptions.html#PendingDeprecationWarning" class="reference internal" title="PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a>. They will be removed in Python 3.9. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29209" class="reference external">bpo-29209</a>.)

- Passing an object that is not an instance of <a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">concurrent.futures.ThreadPoolExecutor</code></span></a> to <a href="../library/asyncio-eventloop.html#asyncio.loop.set_default_executor" class="reference internal" title="asyncio.loop.set_default_executor"><span class="pre"><code class="sourceCode python">loop.set_default_executor()</code></span></a> is deprecated and will be prohibited in Python 3.9. (Contributed by Elvis Pranskevichus in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34075" class="reference external">bpo-34075</a>.)

- The <a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a> methods of <a href="../library/xml.dom.pulldom.html#xml.dom.pulldom.DOMEventStream" class="reference internal" title="xml.dom.pulldom.DOMEventStream"><span class="pre"><code class="sourceCode python">xml.dom.pulldom.DOMEventStream</code></span></a>, <a href="../library/wsgiref.html#wsgiref.util.FileWrapper" class="reference internal" title="wsgiref.util.FileWrapper"><span class="pre"><code class="sourceCode python">wsgiref.util.FileWrapper</code></span></a> and <a href="../library/fileinput.html#fileinput.FileInput" class="reference internal" title="fileinput.FileInput"><span class="pre"><code class="sourceCode python">fileinput.FileInput</code></span></a> have been deprecated.

  Implementations of these methods have been ignoring their *index* parameter, and returning the next item instead. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9372" class="reference external">bpo-9372</a>.)

- The <a href="../library/typing.html#typing.NamedTuple" class="reference internal" title="typing.NamedTuple"><span class="pre"><code class="sourceCode python">typing.NamedTuple</code></span></a> class has deprecated the <span class="pre">`_field_types`</span> attribute in favor of the <span class="pre">`__annotations__`</span> attribute which has the same information. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36320" class="reference external">bpo-36320</a>.)

- <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> classes <span class="pre">`Num`</span>, <span class="pre">`Str`</span>, <span class="pre">`Bytes`</span>, <span class="pre">`NameConstant`</span> and <span class="pre">`Ellipsis`</span> are considered deprecated and will be removed in future Python versions. <a href="../library/ast.html#ast.Constant" class="reference internal" title="ast.Constant"><span class="pre"><code class="sourceCode python">Constant</code></span></a> should be used instead. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32892" class="reference external">bpo-32892</a>.)

- <a href="../library/ast.html#ast.NodeVisitor" class="reference internal" title="ast.NodeVisitor"><span class="pre"><code class="sourceCode python">ast.NodeVisitor</code></span></a> methods <span class="pre">`visit_Num()`</span>, <span class="pre">`visit_Str()`</span>, <span class="pre">`visit_Bytes()`</span>, <span class="pre">`visit_NameConstant()`</span> and <span class="pre">`visit_Ellipsis()`</span> are deprecated now and will not be called in future Python versions. Add the <a href="../library/ast.html#ast.NodeVisitor.visit_Constant" class="reference internal" title="ast.NodeVisitor.visit_Constant"><span class="pre"><code class="sourceCode python">visit_Constant()</code></span></a> method to handle all constant nodes. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36917" class="reference external">bpo-36917</a>.)

- The <span class="pre">`@asyncio.coroutine`</span> <a href="../glossary.html#term-decorator" class="reference internal"><span class="xref std std-term">decorator</span></a> is deprecated and will be removed in version 3.10. Instead of <span class="pre">`@asyncio.coroutine`</span>, use <a href="../reference/compound_stmts.html#async-def" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">def</code></span></a> instead. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36921" class="reference external">bpo-36921</a>.)

- In <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>, the explicit passing of a *loop* argument has been deprecated and will be removed in version 3.10 for the following: <a href="../library/asyncio-task.html#asyncio.sleep" class="reference internal" title="asyncio.sleep"><span class="pre"><code class="sourceCode python">asyncio.sleep()</code></span></a>, <a href="../library/asyncio-task.html#asyncio.gather" class="reference internal" title="asyncio.gather"><span class="pre"><code class="sourceCode python">asyncio.gather()</code></span></a>, <a href="../library/asyncio-task.html#asyncio.shield" class="reference internal" title="asyncio.shield"><span class="pre"><code class="sourceCode python">asyncio.shield()</code></span></a>, <a href="../library/asyncio-task.html#asyncio.wait_for" class="reference internal" title="asyncio.wait_for"><span class="pre"><code class="sourceCode python">asyncio.wait_for()</code></span></a>, <a href="../library/asyncio-task.html#asyncio.wait" class="reference internal" title="asyncio.wait"><span class="pre"><code class="sourceCode python">asyncio.wait()</code></span></a>, <a href="../library/asyncio-task.html#asyncio.as_completed" class="reference internal" title="asyncio.as_completed"><span class="pre"><code class="sourceCode python">asyncio.as_completed()</code></span></a>, <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">asyncio.Task</code></span></a>, <a href="../library/asyncio-sync.html#asyncio.Lock" class="reference internal" title="asyncio.Lock"><span class="pre"><code class="sourceCode python">asyncio.Lock</code></span></a>, <a href="../library/asyncio-sync.html#asyncio.Event" class="reference internal" title="asyncio.Event"><span class="pre"><code class="sourceCode python">asyncio.Event</code></span></a>, <a href="../library/asyncio-sync.html#asyncio.Condition" class="reference internal" title="asyncio.Condition"><span class="pre"><code class="sourceCode python">asyncio.Condition</code></span></a>, <a href="../library/asyncio-sync.html#asyncio.Semaphore" class="reference internal" title="asyncio.Semaphore"><span class="pre"><code class="sourceCode python">asyncio.Semaphore</code></span></a>, <a href="../library/asyncio-sync.html#asyncio.BoundedSemaphore" class="reference internal" title="asyncio.BoundedSemaphore"><span class="pre"><code class="sourceCode python">asyncio.BoundedSemaphore</code></span></a>, <a href="../library/asyncio-queue.html#asyncio.Queue" class="reference internal" title="asyncio.Queue"><span class="pre"><code class="sourceCode python">asyncio.Queue</code></span></a>, <a href="../library/asyncio-subprocess.html#asyncio.create_subprocess_exec" class="reference internal" title="asyncio.create_subprocess_exec"><span class="pre"><code class="sourceCode python">asyncio.create_subprocess_exec()</code></span></a>, and <a href="../library/asyncio-subprocess.html#asyncio.create_subprocess_shell" class="reference internal" title="asyncio.create_subprocess_shell"><span class="pre"><code class="sourceCode python">asyncio.create_subprocess_shell()</code></span></a>.

- The explicit passing of coroutine objects to <a href="../library/asyncio-task.html#asyncio.wait" class="reference internal" title="asyncio.wait"><span class="pre"><code class="sourceCode python">asyncio.wait()</code></span></a> has been deprecated and will be removed in version 3.11. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34790" class="reference external">bpo-34790</a>.)

- The following functions and methods are deprecated in the <a href="../library/gettext.html#module-gettext" class="reference internal" title="gettext: Multilingual internationalization services."><span class="pre"><code class="sourceCode python">gettext</code></span></a> module: <span class="pre">`lgettext()`</span>, <span class="pre">`ldgettext()`</span>, <span class="pre">`lngettext()`</span> and <span class="pre">`ldngettext()`</span>. They return encoded bytes, and it’s possible that you will get unexpected Unicode-related exceptions if there are encoding problems with the translated strings. It’s much better to use alternatives which return Unicode strings in Python 3. These functions have been broken for a long time.

  Function <span class="pre">`bind_textdomain_codeset()`</span>, methods <span class="pre">`NullTranslations.output_charset()`</span> and <span class="pre">`NullTranslations.set_output_charset()`</span>, and the *codeset* parameter of functions <a href="../library/gettext.html#gettext.translation" class="reference internal" title="gettext.translation"><span class="pre"><code class="sourceCode python">translation()</code></span></a> and <a href="../library/gettext.html#gettext.install" class="reference internal" title="gettext.install"><span class="pre"><code class="sourceCode python">install()</code></span></a> are also deprecated, since they are only used for the <span class="pre">`l*gettext()`</span> functions. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33710" class="reference external">bpo-33710</a>.)

- The <span class="pre">`isAlive()`</span> method of <a href="../library/threading.html#threading.Thread" class="reference internal" title="threading.Thread"><span class="pre"><code class="sourceCode python">threading.Thread</code></span></a> has been deprecated. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35283" class="reference external">bpo-35283</a>.)

- Many builtin and extension functions that take integer arguments will now emit a deprecation warning for <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a>s, <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">Fraction</code></span></a>s and any other objects that can be converted to integers only with a loss (e.g. that have the <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="fu">__int__</span>()</code></span></a> method but do not have the <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> method). In future version they will be errors. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36048" class="reference external">bpo-36048</a>.)

- Deprecated passing the following arguments as keyword arguments:

  - *func* in <a href="../library/functools.html#functools.partialmethod" class="reference internal" title="functools.partialmethod"><span class="pre"><code class="sourceCode python">functools.partialmethod()</code></span></a>, <a href="../library/weakref.html#weakref.finalize" class="reference internal" title="weakref.finalize"><span class="pre"><code class="sourceCode python">weakref.finalize()</code></span></a>, <a href="../library/profile.html#profile.Profile.runcall" class="reference internal" title="profile.Profile.runcall"><span class="pre"><code class="sourceCode python">profile.Profile.runcall()</code></span></a>, <span class="pre">`cProfile.Profile.runcall()`</span>, <a href="../library/bdb.html#bdb.Bdb.runcall" class="reference internal" title="bdb.Bdb.runcall"><span class="pre"><code class="sourceCode python">bdb.Bdb.runcall()</code></span></a>, <a href="../library/trace.html#trace.Trace.runfunc" class="reference internal" title="trace.Trace.runfunc"><span class="pre"><code class="sourceCode python">trace.Trace.runfunc()</code></span></a> and <a href="../library/curses.html#curses.wrapper" class="reference internal" title="curses.wrapper"><span class="pre"><code class="sourceCode python">curses.wrapper()</code></span></a>.

  - *function* in <a href="../library/unittest.html#unittest.TestCase.addCleanup" class="reference internal" title="unittest.TestCase.addCleanup"><span class="pre"><code class="sourceCode python">unittest.TestCase.addCleanup()</code></span></a>.

  - *fn* in the <a href="../library/concurrent.futures.html#concurrent.futures.Executor.submit" class="reference internal" title="concurrent.futures.Executor.submit"><span class="pre"><code class="sourceCode python">submit()</code></span></a> method of <a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">concurrent.futures.ThreadPoolExecutor</code></span></a> and <a href="../library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor" class="reference internal" title="concurrent.futures.ProcessPoolExecutor"><span class="pre"><code class="sourceCode python">concurrent.futures.ProcessPoolExecutor</code></span></a>.

  - *callback* in <a href="../library/contextlib.html#contextlib.ExitStack.callback" class="reference internal" title="contextlib.ExitStack.callback"><span class="pre"><code class="sourceCode python">contextlib.ExitStack.callback()</code></span></a>, <span class="pre">`contextlib.AsyncExitStack.callback()`</span> and <a href="../library/contextlib.html#contextlib.AsyncExitStack.push_async_callback" class="reference internal" title="contextlib.AsyncExitStack.push_async_callback"><span class="pre"><code class="sourceCode python">contextlib.AsyncExitStack.push_async_callback()</code></span></a>.

  - *c* and *typeid* in the <span class="pre">`create()`</span> method of <span class="pre">`multiprocessing.managers.Server`</span> and <span class="pre">`multiprocessing.managers.SharedMemoryServer`</span>.

  - *obj* in <a href="../library/weakref.html#weakref.finalize" class="reference internal" title="weakref.finalize"><span class="pre"><code class="sourceCode python">weakref.finalize()</code></span></a>.

  In future releases of Python, they will be <a href="../glossary.html#positional-only-parameter" class="reference internal"><span class="std std-ref">positional-only</span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36492" class="reference external">bpo-36492</a>.)

</div>

<div id="api-and-feature-removals" class="section">

## API and Feature Removals<a href="#api-and-feature-removals" class="headerlink" title="Link to this heading">¶</a>

The following features and APIs have been removed from Python 3.8:

- Starting with Python 3.3, importing ABCs from <a href="../library/collections.html#module-collections" class="reference internal" title="collections: Container datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> was deprecated, and importing should be done from <a href="../library/collections.abc.html#module-collections.abc" class="reference internal" title="collections.abc: Abstract base classes for containers"><span class="pre"><code class="sourceCode python">collections.abc</code></span></a>. Being able to import from collections was marked for removal in 3.8, but has been delayed to 3.9. (See <a href="https://github.com/python/cpython/issues/81134" class="reference external">gh-81134</a>.)

- The <span class="pre">`macpath`</span> module, deprecated in Python 3.7, has been removed. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35471" class="reference external">bpo-35471</a>.)

- The function <span class="pre">`platform.popen()`</span> has been removed, after having been deprecated since Python 3.3: use <a href="../library/os.html#os.popen" class="reference internal" title="os.popen"><span class="pre"><code class="sourceCode python">os.popen()</code></span></a> instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35345" class="reference external">bpo-35345</a>.)

- The function <span class="pre">`time.clock()`</span> has been removed, after having been deprecated since Python 3.3: use <a href="../library/time.html#time.perf_counter" class="reference internal" title="time.perf_counter"><span class="pre"><code class="sourceCode python">time.perf_counter()</code></span></a> or <a href="../library/time.html#time.process_time" class="reference internal" title="time.process_time"><span class="pre"><code class="sourceCode python">time.process_time()</code></span></a> instead, depending on your requirements, to have well-defined behavior. (Contributed by Matthias Bussonnier in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36895" class="reference external">bpo-36895</a>.)

- The <span class="pre">`pyvenv`</span> script has been removed in favor of <span class="pre">`python3.8`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`venv`</span> to help eliminate confusion as to what Python interpreter the <span class="pre">`pyvenv`</span> script is tied to. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25427" class="reference external">bpo-25427</a>.)

- <span class="pre">`parse_qs`</span>, <span class="pre">`parse_qsl`</span>, and <span class="pre">`escape`</span> are removed from the <span class="pre">`cgi`</span> module. They are deprecated in Python 3.2 or older. They should be imported from the <span class="pre">`urllib.parse`</span> and <span class="pre">`html`</span> modules instead.

- <span class="pre">`filemode`</span> function is removed from the <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module. It is not documented and deprecated since Python 3.3.

- The <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLParser" class="reference internal" title="xml.etree.ElementTree.XMLParser"><span class="pre"><code class="sourceCode python">XMLParser</code></span></a> constructor no longer accepts the *html* argument. It never had an effect and was deprecated in Python 3.4. All other parameters are now <a href="../glossary.html#keyword-only-parameter" class="reference internal"><span class="std std-ref">keyword-only</span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29209" class="reference external">bpo-29209</a>.)

- Removed the <span class="pre">`doctype()`</span> method of <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLParser" class="reference internal" title="xml.etree.ElementTree.XMLParser"><span class="pre"><code class="sourceCode python">XMLParser</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29209" class="reference external">bpo-29209</a>.)

- “unicode_internal” codec is removed. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36297" class="reference external">bpo-36297</a>.)

- The <span class="pre">`Cache`</span> and <span class="pre">`Statement`</span> objects of the <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> module are not exposed to the user. (Contributed by Aviv Palivoda in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30262" class="reference external">bpo-30262</a>.)

- The <span class="pre">`bufsize`</span> keyword argument of <a href="../library/fileinput.html#fileinput.input" class="reference internal" title="fileinput.input"><span class="pre"><code class="sourceCode python">fileinput.<span class="bu">input</span>()</code></span></a> and <a href="../library/fileinput.html#fileinput.FileInput" class="reference internal" title="fileinput.FileInput"><span class="pre"><code class="sourceCode python">fileinput.FileInput()</code></span></a> which was ignored and deprecated since Python 3.6 has been removed. <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36952" class="reference external">bpo-36952</a> (Contributed by Matthias Bussonnier.)

- The functions <span class="pre">`sys.set_coroutine_wrapper()`</span> and <span class="pre">`sys.get_coroutine_wrapper()`</span> deprecated in Python 3.7 have been removed; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36933" class="reference external">bpo-36933</a> (Contributed by Matthias Bussonnier.)

</div>

<div id="porting-to-python-3-8" class="section">

## Porting to Python 3.8<a href="#porting-to-python-3-8" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="changes-in-python-behavior" class="section">

### Changes in Python behavior<a href="#changes-in-python-behavior" class="headerlink" title="Link to this heading">¶</a>

- Yield expressions (both <span class="pre">`yield`</span> and <span class="pre">`yield`</span>` `<span class="pre">`from`</span> clauses) are now disallowed in comprehensions and generator expressions (aside from the iterable expression in the leftmost <span class="pre">`for`</span> clause). (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10544" class="reference external">bpo-10544</a>.)

- The compiler now produces a <a href="../library/exceptions.html#SyntaxWarning" class="reference internal" title="SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a> when identity checks (<span class="pre">`is`</span> and <span class="pre">`is`</span>` `<span class="pre">`not`</span>) are used with certain types of literals (e.g. strings, numbers). These can often work by accident in CPython, but are not guaranteed by the language spec. The warning advises users to use equality tests (<span class="pre">`==`</span> and <span class="pre">`!=`</span>) instead. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34850" class="reference external">bpo-34850</a>.)

- The CPython interpreter can swallow exceptions in some circumstances. In Python 3.8 this happens in fewer cases. In particular, exceptions raised when getting the attribute from the type dictionary are no longer ignored. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35459" class="reference external">bpo-35459</a>.)

- Removed <span class="pre">`__str__`</span> implementations from builtin types <a href="../library/functions.html#bool" class="reference internal" title="bool"><span class="pre"><code class="sourceCode python"><span class="bu">bool</span></code></span></a>, <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>, <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a>, <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a> and few classes from the standard library. They now inherit <span class="pre">`__str__()`</span> from <a href="../library/functions.html#object" class="reference internal" title="object"><span class="pre"><code class="sourceCode python"><span class="bu">object</span></code></span></a>. As result, defining the <span class="pre">`__repr__()`</span> method in the subclass of these classes will affect their string representation. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36793" class="reference external">bpo-36793</a>.)

- On AIX, <a href="../library/sys.html#sys.platform" class="reference internal" title="sys.platform"><span class="pre"><code class="sourceCode python">sys.platform</code></span></a> doesn’t contain the major version anymore. It is always <span class="pre">`'aix'`</span>, instead of <span class="pre">`'aix3'`</span> .. <span class="pre">`'aix7'`</span>. Since older Python versions include the version number, so it is recommended to always use <span class="pre">`sys.platform.startswith('aix')`</span>. (Contributed by M. Felt in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36588" class="reference external">bpo-36588</a>.)

- <span class="pre">`PyEval_AcquireLock()`</span> and <span class="pre">`PyEval_AcquireThread()`</span> now terminate the current thread if called while the interpreter is finalizing, making them consistent with <a href="../c-api/init.html#c.PyEval_RestoreThread" class="reference internal" title="PyEval_RestoreThread"><span class="pre"><code class="sourceCode c">PyEval_RestoreThread<span class="op">()</span></code></span></a>, <a href="../c-api/init.html#c.Py_END_ALLOW_THREADS" class="reference internal" title="Py_END_ALLOW_THREADS"><span class="pre"><code class="sourceCode c">Py_END_ALLOW_THREADS<span class="op">()</span></code></span></a>, and <a href="../c-api/init.html#c.PyGILState_Ensure" class="reference internal" title="PyGILState_Ensure"><span class="pre"><code class="sourceCode c">PyGILState_Ensure<span class="op">()</span></code></span></a>. If this behavior is not desired, guard the call by checking <span class="pre">`_Py_IsFinalizing()`</span> or <a href="../library/sys.html#sys.is_finalizing" class="reference internal" title="sys.is_finalizing"><span class="pre"><code class="sourceCode python">sys.is_finalizing()</code></span></a>. (Contributed by Joannah Nanjekye in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36475" class="reference external">bpo-36475</a>.)

</div>

<div id="changes-in-the-python-api" class="section">

### Changes in the Python API<a href="#changes-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/os.html#os.getcwdb" class="reference internal" title="os.getcwdb"><span class="pre"><code class="sourceCode python">os.getcwdb()</code></span></a> function now uses the UTF-8 encoding on Windows, rather than the ANSI code page: see <span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0529/" class="pep reference external"><strong>PEP 529</strong></a> for the rationale. The function is no longer deprecated on Windows. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37412" class="reference external">bpo-37412</a>.)

- <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a> can now use <a href="../library/os.html#os.posix_spawn" class="reference internal" title="os.posix_spawn"><span class="pre"><code class="sourceCode python">os.posix_spawn()</code></span></a> in some cases for better performance. On Windows Subsystem for Linux and QEMU User Emulation, the <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">Popen</code></span></a> constructor using <a href="../library/os.html#os.posix_spawn" class="reference internal" title="os.posix_spawn"><span class="pre"><code class="sourceCode python">os.posix_spawn()</code></span></a> no longer raises an exception on errors like “missing program”. Instead the child process fails with a non-zero <a href="../library/subprocess.html#subprocess.Popen.returncode" class="reference internal" title="subprocess.Popen.returncode"><span class="pre"><code class="sourceCode python">returncode</code></span></a>. (Contributed by Joannah Nanjekye and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35537" class="reference external">bpo-35537</a>.)

- The *preexec_fn* argument of \* <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a> is no longer compatible with subinterpreters. The use of the parameter in a subinterpreter now raises <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a>. (Contributed by Eric Snow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34651" class="reference external">bpo-34651</a>, modified by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37951" class="reference external">bpo-37951</a>.)

- The <a href="../library/imaplib.html#imaplib.IMAP4.logout" class="reference internal" title="imaplib.IMAP4.logout"><span class="pre"><code class="sourceCode python">imaplib.IMAP4.logout()</code></span></a> method no longer silently ignores arbitrary exceptions. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36348" class="reference external">bpo-36348</a>.)

- The function <span class="pre">`platform.popen()`</span> has been removed, after having been deprecated since Python 3.3: use <a href="../library/os.html#os.popen" class="reference internal" title="os.popen"><span class="pre"><code class="sourceCode python">os.popen()</code></span></a> instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35345" class="reference external">bpo-35345</a>.)

- The <a href="../library/statistics.html#statistics.mode" class="reference internal" title="statistics.mode"><span class="pre"><code class="sourceCode python">statistics.mode()</code></span></a> function no longer raises an exception when given multimodal data. Instead, it returns the first mode encountered in the input data. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35892" class="reference external">bpo-35892</a>.)

- The <a href="../library/tkinter.ttk.html#tkinter.ttk.Treeview.selection" class="reference internal" title="tkinter.ttk.Treeview.selection"><span class="pre"><code class="sourceCode python">selection()</code></span></a> method of the <a href="../library/tkinter.ttk.html#tkinter.ttk.Treeview" class="reference internal" title="tkinter.ttk.Treeview"><span class="pre"><code class="sourceCode python">tkinter.ttk.Treeview</code></span></a> class no longer takes arguments. Using it with arguments for changing the selection was deprecated in Python 3.6. Use specialized methods like <a href="../library/tkinter.ttk.html#tkinter.ttk.Treeview.selection_set" class="reference internal" title="tkinter.ttk.Treeview.selection_set"><span class="pre"><code class="sourceCode python">selection_set()</code></span></a> for changing the selection. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31508" class="reference external">bpo-31508</a>.)

- The <a href="../library/xml.dom.minidom.html#xml.dom.minidom.Node.writexml" class="reference internal" title="xml.dom.minidom.Node.writexml"><span class="pre"><code class="sourceCode python">writexml()</code></span></a>, <a href="../library/xml.dom.minidom.html#xml.dom.minidom.Node.toxml" class="reference internal" title="xml.dom.minidom.Node.toxml"><span class="pre"><code class="sourceCode python">toxml()</code></span></a> and <a href="../library/xml.dom.minidom.html#xml.dom.minidom.Node.toprettyxml" class="reference internal" title="xml.dom.minidom.Node.toprettyxml"><span class="pre"><code class="sourceCode python">toprettyxml()</code></span></a> methods of <a href="../library/xml.dom.minidom.html#module-xml.dom.minidom" class="reference internal" title="xml.dom.minidom: Minimal Document Object Model (DOM) implementation."><span class="pre"><code class="sourceCode python">xml.dom.minidom</code></span></a> and the <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.ElementTree.write" class="reference internal" title="xml.etree.ElementTree.ElementTree.write"><span class="pre"><code class="sourceCode python">write()</code></span></a> method of <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> now preserve the attribute order specified by the user. (Contributed by Diego Rojas and Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34160" class="reference external">bpo-34160</a>.)

- A <a href="../library/dbm.html#module-dbm.dumb" class="reference internal" title="dbm.dumb: Portable implementation of the simple DBM interface."><span class="pre"><code class="sourceCode python">dbm.dumb</code></span></a> database opened with flags <span class="pre">`'r'`</span> is now read-only. <a href="../library/dbm.html#dbm.dumb.open" class="reference internal" title="dbm.dumb.open"><span class="pre"><code class="sourceCode python">dbm.dumb.<span class="bu">open</span>()</code></span></a> with flags <span class="pre">`'r'`</span> and <span class="pre">`'w'`</span> no longer creates a database if it does not exist. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32749" class="reference external">bpo-32749</a>.)

- The <span class="pre">`doctype()`</span> method defined in a subclass of <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLParser" class="reference internal" title="xml.etree.ElementTree.XMLParser"><span class="pre"><code class="sourceCode python">XMLParser</code></span></a> will no longer be called and will emit a <a href="../library/exceptions.html#RuntimeWarning" class="reference internal" title="RuntimeWarning"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeWarning</span></code></span></a> instead of a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. Define the <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.TreeBuilder.doctype" class="reference internal" title="xml.etree.ElementTree.TreeBuilder.doctype"><span class="pre"><code class="sourceCode python">doctype()</code></span></a> method on a target for handling an XML doctype declaration. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29209" class="reference external">bpo-29209</a>.)

- A <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> is now raised when the custom metaclass doesn’t provide the <span class="pre">`__classcell__`</span> entry in the namespace passed to <span class="pre">`type.__new__`</span>. A <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> was emitted in Python 3.6–3.7. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23722" class="reference external">bpo-23722</a>.)

- The <a href="../library/profile.html#profile.Profile" class="reference internal" title="profile.Profile"><span class="pre"><code class="sourceCode python">cProfile.Profile</code></span></a> class can now be used as a context manager. (Contributed by Scott Sanderson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29235" class="reference external">bpo-29235</a>.)

- <a href="../library/shutil.html#shutil.copyfile" class="reference internal" title="shutil.copyfile"><span class="pre"><code class="sourceCode python">shutil.copyfile()</code></span></a>, <a href="../library/shutil.html#shutil.copy" class="reference internal" title="shutil.copy"><span class="pre"><code class="sourceCode python">shutil.copy()</code></span></a>, <a href="../library/shutil.html#shutil.copy2" class="reference internal" title="shutil.copy2"><span class="pre"><code class="sourceCode python">shutil.copy2()</code></span></a>, <a href="../library/shutil.html#shutil.copytree" class="reference internal" title="shutil.copytree"><span class="pre"><code class="sourceCode python">shutil.copytree()</code></span></a> and <a href="../library/shutil.html#shutil.move" class="reference internal" title="shutil.move"><span class="pre"><code class="sourceCode python">shutil.move()</code></span></a> use platform-specific “fast-copy” syscalls (see <a href="../library/shutil.html#shutil-platform-dependent-efficient-copy-operations" class="reference internal"><span class="std std-ref">Platform-dependent efficient copy operations</span></a> section).

- <a href="../library/shutil.html#shutil.copyfile" class="reference internal" title="shutil.copyfile"><span class="pre"><code class="sourceCode python">shutil.copyfile()</code></span></a> default buffer size on Windows was changed from 16 KiB to 1 MiB.

- The <span class="pre">`PyGC_Head`</span> struct has changed completely. All code that touched the struct member should be rewritten. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33597" class="reference external">bpo-33597</a>.)

- The <a href="../c-api/init.html#c.PyInterpreterState" class="reference internal" title="PyInterpreterState"><span class="pre"><code class="sourceCode c">PyInterpreterState</code></span></a> struct has been moved into the “internal” header files (specifically Include/internal/pycore_pystate.h). An opaque <span class="pre">`PyInterpreterState`</span> is still available as part of the public API (and stable ABI). The docs indicate that none of the struct’s fields are public, so we hope no one has been using them. However, if you do rely on one or more of those private fields and have no alternative then please open a BPO issue. We’ll work on helping you adjust (possibly including adding accessor functions to the public API). (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35886" class="reference external">bpo-35886</a>.)

- The <a href="../library/mmap.html#mmap.mmap.flush" class="reference internal" title="mmap.mmap.flush"><span class="pre"><code class="sourceCode python">mmap.flush()</code></span></a> method now returns <span class="pre">`None`</span> on success and raises an exception on error under all platforms. Previously, its behavior was platform-dependent: a nonzero value was returned on success; zero was returned on error under Windows. A zero value was returned on success; an exception was raised on error under Unix. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2122" class="reference external">bpo-2122</a>.)

- <a href="../library/xml.dom.minidom.html#module-xml.dom.minidom" class="reference internal" title="xml.dom.minidom: Minimal Document Object Model (DOM) implementation."><span class="pre"><code class="sourceCode python">xml.dom.minidom</code></span></a> and <a href="../library/xml.sax.html#module-xml.sax" class="reference internal" title="xml.sax: Package containing SAX2 base classes and convenience functions."><span class="pre"><code class="sourceCode python">xml.sax</code></span></a> modules no longer process external entities by default. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17239" class="reference external">bpo-17239</a>.)

- Deleting a key from a read-only <a href="../library/dbm.html#module-dbm" class="reference internal" title="dbm: Interfaces to various Unix &quot;database&quot; formats."><span class="pre"><code class="sourceCode python">dbm</code></span></a> database (<a href="../library/dbm.html#module-dbm.dumb" class="reference internal" title="dbm.dumb: Portable implementation of the simple DBM interface."><span class="pre"><code class="sourceCode python">dbm.dumb</code></span></a>, <a href="../library/dbm.html#module-dbm.gnu" class="reference internal" title="dbm.gnu: GNU database manager (Unix)"><span class="pre"><code class="sourceCode python">dbm.gnu</code></span></a> or <a href="../library/dbm.html#module-dbm.ndbm" class="reference internal" title="dbm.ndbm: The New Database Manager (Unix)"><span class="pre"><code class="sourceCode python">dbm.ndbm</code></span></a>) raises <span class="pre">`error`</span> (<a href="../library/dbm.html#dbm.dumb.error" class="reference internal" title="dbm.dumb.error"><span class="pre"><code class="sourceCode python">dbm.dumb.error</code></span></a>, <a href="../library/dbm.html#dbm.gnu.error" class="reference internal" title="dbm.gnu.error"><span class="pre"><code class="sourceCode python">dbm.gnu.error</code></span></a> or <a href="../library/dbm.html#dbm.ndbm.error" class="reference internal" title="dbm.ndbm.error"><span class="pre"><code class="sourceCode python">dbm.ndbm.error</code></span></a>) instead of <a href="../library/exceptions.html#KeyError" class="reference internal" title="KeyError"><span class="pre"><code class="sourceCode python"><span class="pp">KeyError</span></code></span></a>. (Contributed by Xiang Zhang in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33106" class="reference external">bpo-33106</a>.)

- Simplified AST for literals. All constants will be represented as <a href="../library/ast.html#ast.Constant" class="reference internal" title="ast.Constant"><span class="pre"><code class="sourceCode python">ast.Constant</code></span></a> instances. Instantiating old classes <span class="pre">`Num`</span>, <span class="pre">`Str`</span>, <span class="pre">`Bytes`</span>, <span class="pre">`NameConstant`</span> and <span class="pre">`Ellipsis`</span> will return an instance of <span class="pre">`Constant`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32892" class="reference external">bpo-32892</a>.)

- <a href="../library/os.path.html#os.path.expanduser" class="reference internal" title="os.path.expanduser"><span class="pre"><code class="sourceCode python">expanduser()</code></span></a> on Windows now prefers the <span id="index-20" class="target"></span><span class="pre">`USERPROFILE`</span> environment variable and does not use <span id="index-21" class="target"></span><span class="pre">`HOME`</span>, which is not normally set for regular user accounts. (Contributed by Anthony Sottile in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36264" class="reference external">bpo-36264</a>.)

- The exception <a href="../library/asyncio-exceptions.html#asyncio.CancelledError" class="reference internal" title="asyncio.CancelledError"><span class="pre"><code class="sourceCode python">asyncio.CancelledError</code></span></a> now inherits from <a href="../library/exceptions.html#BaseException" class="reference internal" title="BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a> rather than <a href="../library/exceptions.html#Exception" class="reference internal" title="Exception"><span class="pre"><code class="sourceCode python"><span class="pp">Exception</span></code></span></a> and no longer inherits from <a href="../library/concurrent.futures.html#concurrent.futures.CancelledError" class="reference internal" title="concurrent.futures.CancelledError"><span class="pre"><code class="sourceCode python">concurrent.futures.CancelledError</code></span></a>. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32528" class="reference external">bpo-32528</a>.)

- The function <a href="../library/asyncio-task.html#asyncio.wait_for" class="reference internal" title="asyncio.wait_for"><span class="pre"><code class="sourceCode python">asyncio.wait_for()</code></span></a> now correctly waits for cancellation when using an instance of <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">asyncio.Task</code></span></a>. Previously, upon reaching *timeout*, it was cancelled and immediately returned. (Contributed by Elvis Pranskevichus in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32751" class="reference external">bpo-32751</a>.)

- The function <a href="../library/asyncio-protocol.html#asyncio.BaseTransport.get_extra_info" class="reference internal" title="asyncio.BaseTransport.get_extra_info"><span class="pre"><code class="sourceCode python">asyncio.BaseTransport.get_extra_info()</code></span></a> now returns a safe to use socket object when ‘socket’ is passed to the *name* parameter. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37027" class="reference external">bpo-37027</a>.)

- <a href="../library/asyncio-protocol.html#asyncio.BufferedProtocol" class="reference internal" title="asyncio.BufferedProtocol"><span class="pre"><code class="sourceCode python">asyncio.BufferedProtocol</code></span></a> has graduated to the stable API.

<!-- -->

- DLL dependencies for extension modules and DLLs loaded with <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> on Windows are now resolved more securely. Only the system paths, the directory containing the DLL or PYD file, and directories added with <a href="../library/os.html#os.add_dll_directory" class="reference internal" title="os.add_dll_directory"><span class="pre"><code class="sourceCode python">add_dll_directory()</code></span></a> are searched for load-time dependencies. Specifically, <span id="index-22" class="target"></span><span class="pre">`PATH`</span> and the current working directory are no longer used, and modifications to these will no longer have any effect on normal DLL resolution. If your application relies on these mechanisms, you should check for <a href="../library/os.html#os.add_dll_directory" class="reference internal" title="os.add_dll_directory"><span class="pre"><code class="sourceCode python">add_dll_directory()</code></span></a> and if it exists, use it to add your DLLs directory while loading your library. Note that Windows 7 users will need to ensure that Windows Update KB2533623 has been installed (this is also verified by the installer). (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36085" class="reference external">bpo-36085</a>.)

- The header files and functions related to pgen have been removed after its replacement by a pure Python implementation. (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36623" class="reference external">bpo-36623</a>.)

- <a href="../library/types.html#types.CodeType" class="reference internal" title="types.CodeType"><span class="pre"><code class="sourceCode python">types.CodeType</code></span></a> has a new parameter in the second position of the constructor (*posonlyargcount*) to support positional-only arguments defined in <span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0570/" class="pep reference external"><strong>PEP 570</strong></a>. The first argument (*argcount*) now represents the total number of positional arguments (including positional-only arguments). The new <span class="pre">`replace()`</span> method of <a href="../library/types.html#types.CodeType" class="reference internal" title="types.CodeType"><span class="pre"><code class="sourceCode python">types.CodeType</code></span></a> can be used to make the code future-proof.

- The parameter <span class="pre">`digestmod`</span> for <a href="../library/hmac.html#hmac.new" class="reference internal" title="hmac.new"><span class="pre"><code class="sourceCode python">hmac.new()</code></span></a> no longer uses the MD5 digest by default.

</div>

<div id="changes-in-the-c-api" class="section">

### Changes in the C API<a href="#changes-in-the-c-api" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../c-api/veryhigh.html#c.PyCompilerFlags" class="reference internal" title="PyCompilerFlags"><span class="pre"><code class="sourceCode c">PyCompilerFlags</code></span></a> structure got a new *cf_feature_version* field. It should be initialized to <span class="pre">`PY_MINOR_VERSION`</span>. The field is ignored by default, and is used if and only if <span class="pre">`PyCF_ONLY_AST`</span> flag is set in *cf_flags*. (Contributed by Guido van Rossum in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35766" class="reference external">bpo-35766</a>.)

- The <span class="pre">`PyEval_ReInitThreads()`</span> function has been removed from the C API. It should not be called explicitly: use <a href="../c-api/sys.html#c.PyOS_AfterFork_Child" class="reference internal" title="PyOS_AfterFork_Child"><span class="pre"><code class="sourceCode c">PyOS_AfterFork_Child<span class="op">()</span></code></span></a> instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36728" class="reference external">bpo-36728</a>.)

- On Unix, C extensions are no longer linked to libpython except on Android and Cygwin. When Python is embedded, <span class="pre">`libpython`</span> must not be loaded with <span class="pre">`RTLD_LOCAL`</span>, but <span class="pre">`RTLD_GLOBAL`</span> instead. Previously, using <span class="pre">`RTLD_LOCAL`</span>, it was already not possible to load C extensions which were not linked to <span class="pre">`libpython`</span>, like C extensions of the standard library built by the <span class="pre">`*shared*`</span> section of <span class="pre">`Modules/Setup`</span>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21536" class="reference external">bpo-21536</a>.)

- Use of <span class="pre">`#`</span> variants of formats in parsing or building value (e.g. <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple<span class="op">()</span></code></span></a>, <a href="../c-api/arg.html#c.Py_BuildValue" class="reference internal" title="Py_BuildValue"><span class="pre"><code class="sourceCode c">Py_BuildValue<span class="op">()</span></code></span></a>, <a href="../c-api/call.html#c.PyObject_CallFunction" class="reference internal" title="PyObject_CallFunction"><span class="pre"><code class="sourceCode c">PyObject_CallFunction<span class="op">()</span></code></span></a>, etc.) without <span class="pre">`PY_SSIZE_T_CLEAN`</span> defined raises <span class="pre">`DeprecationWarning`</span> now. It will be removed in 3.10 or 4.0. Read <a href="../c-api/arg.html#arg-parsing" class="reference internal"><span class="std std-ref">Parsing arguments and building values</span></a> for detail. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36381" class="reference external">bpo-36381</a>.)

- Instances of heap-allocated types (such as those created with <a href="../c-api/type.html#c.PyType_FromSpec" class="reference internal" title="PyType_FromSpec"><span class="pre"><code class="sourceCode c">PyType_FromSpec<span class="op">()</span></code></span></a>) hold a reference to their type object. Increasing the reference count of these type objects has been moved from <a href="../c-api/type.html#c.PyType_GenericAlloc" class="reference internal" title="PyType_GenericAlloc"><span class="pre"><code class="sourceCode c">PyType_GenericAlloc<span class="op">()</span></code></span></a> to the more low-level functions, <a href="../c-api/allocation.html#c.PyObject_Init" class="reference internal" title="PyObject_Init"><span class="pre"><code class="sourceCode c">PyObject_Init<span class="op">()</span></code></span></a> and <span class="pre">`PyObject_INIT`</span>. This makes types created through <a href="../c-api/type.html#c.PyType_FromSpec" class="reference internal" title="PyType_FromSpec"><span class="pre"><code class="sourceCode c">PyType_FromSpec<span class="op">()</span></code></span></a> behave like other classes in managed code.

  <a href="../c-api/typeobj.html#static-types" class="reference internal"><span class="std std-ref">Statically allocated types</span></a> are not affected.

  For the vast majority of cases, there should be no side effect. However, types that manually increase the reference count after allocating an instance (perhaps to work around the bug) may now become immortal. To avoid this, these classes need to call Py_DECREF on the type object during instance deallocation.

  To correctly port these types into 3.8, please apply the following changes:

  - Remove <a href="../c-api/refcounting.html#c.Py_INCREF" class="reference internal" title="Py_INCREF"><span class="pre"><code class="sourceCode c">Py_INCREF</code></span></a> on the type object after allocating an instance - if any. This may happen after calling <a href="../c-api/allocation.html#c.PyObject_New" class="reference internal" title="PyObject_New"><span class="pre"><code class="sourceCode c">PyObject_New</code></span></a>, <a href="../c-api/allocation.html#c.PyObject_NewVar" class="reference internal" title="PyObject_NewVar"><span class="pre"><code class="sourceCode c">PyObject_NewVar</code></span></a>, <a href="../c-api/gcsupport.html#c.PyObject_GC_New" class="reference internal" title="PyObject_GC_New"><span class="pre"><code class="sourceCode c">PyObject_GC_New<span class="op">()</span></code></span></a>, <a href="../c-api/gcsupport.html#c.PyObject_GC_NewVar" class="reference internal" title="PyObject_GC_NewVar"><span class="pre"><code class="sourceCode c">PyObject_GC_NewVar<span class="op">()</span></code></span></a>, or any other custom allocator that uses <a href="../c-api/allocation.html#c.PyObject_Init" class="reference internal" title="PyObject_Init"><span class="pre"><code class="sourceCode c">PyObject_Init<span class="op">()</span></code></span></a> or <span class="pre">`PyObject_INIT`</span>.

    Example:

    <div class="highlight-c notranslate">

    <div class="highlight">

        static foo_struct *
        foo_new(PyObject *type) {
            foo_struct *foo = PyObject_GC_New(foo_struct, (PyTypeObject *) type);
            if (foo == NULL)
                return NULL;
        #if PY_VERSION_HEX < 0x03080000
            // Workaround for Python issue 35810; no longer necessary in Python 3.8
            PY_INCREF(type)
        #endif
            return foo;
        }

    </div>

    </div>

  - Ensure that all custom <span class="pre">`tp_dealloc`</span> functions of heap-allocated types decrease the type’s reference count.

    Example:

    <div class="highlight-c notranslate">

    <div class="highlight">

        static void
        foo_dealloc(foo_struct *instance) {
            PyObject *type = Py_TYPE(instance);
            PyObject_GC_Del(instance);
        #if PY_VERSION_HEX >= 0x03080000
            // This was not needed before Python 3.8 (Python issue 35810)
            Py_DECREF(type);
        #endif
        }

    </div>

    </div>

  (Contributed by Eddie Elizondo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35810" class="reference external">bpo-35810</a>.)

- The <a href="../c-api/intro.html#c.Py_DEPRECATED" class="reference internal" title="Py_DEPRECATED"><span class="pre"><code class="sourceCode c">Py_DEPRECATED<span class="op">()</span></code></span></a> macro has been implemented for MSVC. The macro now must be placed before the symbol name.

  Example:

  <div class="highlight-c notranslate">

  <div class="highlight">

      Py_DEPRECATED(3.8) PyAPI_FUNC(int) Py_OldFunction(void);

  </div>

  </div>

  (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33407" class="reference external">bpo-33407</a>.)

- The interpreter does not pretend to support binary compatibility of extension types across feature releases, anymore. A <a href="../c-api/type.html#c.PyTypeObject" class="reference internal" title="PyTypeObject"><span class="pre"><code class="sourceCode c">PyTypeObject</code></span></a> exported by a third-party extension module is supposed to have all the slots expected in the current Python version, including <a href="../c-api/typeobj.html#c.PyTypeObject.tp_finalize" class="reference internal" title="PyTypeObject.tp_finalize"><span class="pre"><code class="sourceCode c">tp_finalize</code></span></a> (<a href="../c-api/typeobj.html#c.Py_TPFLAGS_HAVE_FINALIZE" class="reference internal" title="Py_TPFLAGS_HAVE_FINALIZE"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_HAVE_FINALIZE</code></span></a> is not checked anymore before reading <a href="../c-api/typeobj.html#c.PyTypeObject.tp_finalize" class="reference internal" title="PyTypeObject.tp_finalize"><span class="pre"><code class="sourceCode c">tp_finalize</code></span></a>).

  (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32388" class="reference external">bpo-32388</a>.)

- The functions <span class="pre">`PyNode_AddChild()`</span> and <span class="pre">`PyParser_AddToken()`</span> now accept two additional <span class="pre">`int`</span> arguments *end_lineno* and *end_col_offset*.

- The <span class="pre">`libpython38.a`</span> file to allow MinGW tools to link directly against <span class="pre">`python38.dll`</span> is no longer included in the regular Windows distribution. If you require this file, it may be generated with the <span class="pre">`gendef`</span> and <span class="pre">`dlltool`</span> tools, which are part of the MinGW binutils package:

  <div class="highlight-shell notranslate">

  <div class="highlight">

      gendef - python38.dll > tmp.def
      dlltool --dllname python38.dll --def tmp.def --output-lib libpython38.a

  </div>

  </div>

  The location of an installed <span class="pre">`pythonXY.dll`</span> will depend on the installation options and the version and language of Windows. See <a href="../using/windows.html#using-on-windows" class="reference internal"><span class="std std-ref">Using Python on Windows</span></a> for more information. The resulting library should be placed in the same directory as <span class="pre">`pythonXY.lib`</span>, which is generally the <span class="pre">`libs`</span> directory under your Python installation.

  (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37351" class="reference external">bpo-37351</a>.)

</div>

<div id="cpython-bytecode-changes" class="section">

### CPython bytecode changes<a href="#cpython-bytecode-changes" class="headerlink" title="Link to this heading">¶</a>

- The interpreter loop has been simplified by moving the logic of unrolling the stack of blocks into the compiler. The compiler emits now explicit instructions for adjusting the stack of values and calling the cleaning-up code for <a href="../reference/simple_stmts.html#break" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">break</code></span></a>, <a href="../reference/simple_stmts.html#continue" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">continue</code></span></a> and <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a>.

  Removed opcodes <span class="pre">`BREAK_LOOP`</span>, <span class="pre">`CONTINUE_LOOP`</span>, <span class="pre">`SETUP_LOOP`</span> and <span class="pre">`SETUP_EXCEPT`</span>. Added new opcodes <span class="pre">`ROT_FOUR`</span>, <span class="pre">`BEGIN_FINALLY`</span>, <span class="pre">`CALL_FINALLY`</span> and <span class="pre">`POP_FINALLY`</span>. Changed the behavior of <span class="pre">`END_FINALLY`</span> and <span class="pre">`WITH_CLEANUP_START`</span>.

  (Contributed by Mark Shannon, Antoine Pitrou and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17611" class="reference external">bpo-17611</a>.)

- Added new opcode <a href="../library/dis.html#opcode-END_ASYNC_FOR" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">END_ASYNC_FOR</code></span></a> for handling exceptions raised when awaiting a next item in an <a href="../reference/compound_stmts.html#async-for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> loop. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33041" class="reference external">bpo-33041</a>.)

- The <a href="../library/dis.html#opcode-MAP_ADD" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">MAP_ADD</code></span></a> now expects the value as the first element in the stack and the key as the second element. This change was made so the key is always evaluated before the value in dictionary comprehensions, as proposed by <span id="index-24" class="target"></span><a href="https://peps.python.org/pep-0572/" class="pep reference external"><strong>PEP 572</strong></a>. (Contributed by Jörn Heissler in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35224" class="reference external">bpo-35224</a>.)

</div>

<div id="demos-and-tools" class="section">

### Demos and Tools<a href="#demos-and-tools" class="headerlink" title="Link to this heading">¶</a>

Added a benchmark script for timing various ways to access variables: <span class="pre">`Tools/scripts/var_access_benchmark.py`</span>. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35884" class="reference external">bpo-35884</a>.)

Here’s a summary of performance improvements since Python 3.3:

<div class="highlight-none notranslate">

<div class="highlight">

    Python version                       3.3     3.4     3.5     3.6     3.7     3.8
    --------------                       ---     ---     ---     ---     ---     ---

    Variable and attribute read access:
        read_local                       4.0     7.1     7.1     5.4     5.1     3.9
        read_nonlocal                    5.3     7.1     8.1     5.8     5.4     4.4
        read_global                     13.3    15.5    19.0    14.3    13.6     7.6
        read_builtin                    20.0    21.1    21.6    18.5    19.0     7.5
        read_classvar_from_class        20.5    25.6    26.5    20.7    19.5    18.4
        read_classvar_from_instance     18.5    22.8    23.5    18.8    17.1    16.4
        read_instancevar                26.8    32.4    33.1    28.0    26.3    25.4
        read_instancevar_slots          23.7    27.8    31.3    20.8    20.8    20.2
        read_namedtuple                 68.5    73.8    57.5    45.0    46.8    18.4
        read_boundmethod                29.8    37.6    37.9    29.6    26.9    27.7

    Variable and attribute write access:
        write_local                      4.6     8.7     9.3     5.5     5.3     4.3
        write_nonlocal                   7.3    10.5    11.1     5.6     5.5     4.7
        write_global                    15.9    19.7    21.2    18.0    18.0    15.8
        write_classvar                  81.9    92.9    96.0   104.6   102.1    39.2
        write_instancevar               36.4    44.6    45.8    40.0    38.9    35.5
        write_instancevar_slots         28.7    35.6    36.1    27.3    26.6    25.7

    Data structure read access:
        read_list                       19.2    24.2    24.5    20.8    20.8    19.0
        read_deque                      19.9    24.7    25.5    20.2    20.6    19.8
        read_dict                       19.7    24.3    25.7    22.3    23.0    21.0
        read_strdict                    17.9    22.6    24.3    19.5    21.2    18.9

    Data structure write access:
        write_list                      21.2    27.1    28.5    22.5    21.6    20.0
        write_deque                     23.8    28.7    30.1    22.7    21.8    23.5
        write_dict                      25.9    31.4    33.3    29.3    29.2    24.7
        write_strdict                   22.9    28.4    29.9    27.5    25.2    23.1

    Stack (or queue) operations:
        list_append_pop                144.2    93.4   112.7    75.4    74.2    50.8
        deque_append_pop                30.4    43.5    57.0    49.4    49.2    42.5
        deque_append_popleft            30.8    43.7    57.3    49.7    49.7    42.8

    Timing loop:
        loop_overhead                    0.3     0.5     0.6     0.4     0.3     0.3

</div>

</div>

The benchmarks were measured on an <a href="https://ark.intel.com/content/www/us/en/ark/products/76088/intel-core-i7-4960hq-processor-6m-cache-up-to-3-80-ghz.html" class="reference external">Intel® Core™ i7-4960HQ processor</a> running the macOS 64-bit builds found at <a href="https://www.python.org/downloads/macos/" class="reference external">python.org</a>. The benchmark script displays timings in nanoseconds.

</div>

</div>

<div id="notable-changes-in-python-3-8-1" class="section">

## Notable changes in Python 3.8.1<a href="#notable-changes-in-python-3-8-1" class="headerlink" title="Link to this heading">¶</a>

Due to significant security concerns, the *reuse_address* parameter of <a href="../library/asyncio-eventloop.html#asyncio.loop.create_datagram_endpoint" class="reference internal" title="asyncio.loop.create_datagram_endpoint"><span class="pre"><code class="sourceCode python">asyncio.loop.create_datagram_endpoint()</code></span></a> is no longer supported. This is because of the behavior of the socket option <span class="pre">`SO_REUSEADDR`</span> in UDP. For more details, see the documentation for <span class="pre">`loop.create_datagram_endpoint()`</span>. (Contributed by Kyle Stanley, Antoine Pitrou, and Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37228" class="reference external">bpo-37228</a>.)

</div>

<div id="notable-changes-in-python-3-8-2" class="section">

## Notable changes in Python 3.8.2<a href="#notable-changes-in-python-3-8-2" class="headerlink" title="Link to this heading">¶</a>

Fixed a regression with the <span class="pre">`ignore`</span> callback of <a href="../library/shutil.html#shutil.copytree" class="reference internal" title="shutil.copytree"><span class="pre"><code class="sourceCode python">shutil.copytree()</code></span></a>. The argument types are now str and List\[str\] again. (Contributed by Manuel Barkhau and Giampaolo Rodola in <a href="https://github.com/python/cpython/issues/83571" class="reference external">gh-83571</a>.)

</div>

<div id="notable-changes-in-python-3-8-3" class="section">

## Notable changes in Python 3.8.3<a href="#notable-changes-in-python-3-8-3" class="headerlink" title="Link to this heading">¶</a>

The constant values of future flags in the <a href="../library/__future__.html#module-__future__" class="reference internal" title="__future__: Future statement definitions"><span class="pre"><code class="sourceCode python">__future__</code></span></a> module are updated in order to prevent collision with compiler flags. Previously <span class="pre">`PyCF_ALLOW_TOP_LEVEL_AWAIT`</span> was clashing with <span class="pre">`CO_FUTURE_DIVISION`</span>. (Contributed by Batuhan Taskaya in <a href="https://github.com/python/cpython/issues/83743" class="reference external">gh-83743</a>)

</div>

<div id="notable-changes-in-python-3-8-8" class="section">

## Notable changes in Python 3.8.8<a href="#notable-changes-in-python-3-8-8" class="headerlink" title="Link to this heading">¶</a>

Earlier Python versions allowed using both <span class="pre">`;`</span> and <span class="pre">`&`</span> as query parameter separators in <a href="../library/urllib.parse.html#urllib.parse.parse_qs" class="reference internal" title="urllib.parse.parse_qs"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qs()</code></span></a> and <a href="../library/urllib.parse.html#urllib.parse.parse_qsl" class="reference internal" title="urllib.parse.parse_qsl"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qsl()</code></span></a>. Due to security concerns, and to conform with newer W3C recommendations, this has been changed to allow only a single separator key, with <span class="pre">`&`</span> as the default. This change also affects <span class="pre">`cgi.parse()`</span> and <span class="pre">`cgi.parse_multipart()`</span> as they use the affected functions internally. For more details, please see their respective documentation. (Contributed by Adam Goldschmidt, Senthil Kumaran and Ken Jin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42967" class="reference external">bpo-42967</a>.)

</div>

<div id="notable-changes-in-python-3-8-9" class="section">

## Notable changes in Python 3.8.9<a href="#notable-changes-in-python-3-8-9" class="headerlink" title="Link to this heading">¶</a>

A security fix alters the <a href="../library/ftplib.html#ftplib.FTP" class="reference internal" title="ftplib.FTP"><span class="pre"><code class="sourceCode python">ftplib.FTP</code></span></a> behavior to not trust the IPv4 address sent from the remote server when setting up a passive data channel. We reuse the ftp server IP address instead. For unusual code requiring the old behavior, set a <span class="pre">`trust_server_pasv_ipv4_address`</span> attribute on your FTP instance to <span class="pre">`True`</span>. (See <a href="https://github.com/python/cpython/issues/87451" class="reference external">gh-87451</a>)

</div>

<div id="notable-changes-in-python-3-8-10" class="section">

## Notable changes in Python 3.8.10<a href="#notable-changes-in-python-3-8-10" class="headerlink" title="Link to this heading">¶</a>

<div id="macos-11-0-big-sur-and-apple-silicon-mac-support" class="section">

### macOS 11.0 (Big Sur) and Apple Silicon Mac support<a href="#macos-11-0-big-sur-and-apple-silicon-mac-support" class="headerlink" title="Link to this heading">¶</a>

As of 3.8.10, Python now supports building and running on macOS 11 (Big Sur) and on Apple Silicon Macs (based on the <span class="pre">`ARM64`</span> architecture). A new universal build variant, <span class="pre">`universal2`</span>, is now available to natively support both <span class="pre">`ARM64`</span> and <span class="pre">`Intel`</span>` `<span class="pre">`64`</span> in one set of executables. Note that support for “weaklinking”, building binaries targeted for newer versions of macOS that will also run correctly on older versions by testing at runtime for missing features, is not included in this backport from Python 3.9; to support a range of macOS versions, continue to target for and build on the oldest version in the range.

(Originally contributed by Ronald Oussoren and Lawrence D’Anna in <a href="https://github.com/python/cpython/issues/85272" class="reference external">gh-85272</a>, with fixes by FX Coudert and Eli Rykoff, and backported to 3.8 by Maxime Bélanger and Ned Deily)

</div>

</div>

<div id="id1" class="section">

## Notable changes in Python 3.8.10<a href="#id1" class="headerlink" title="Link to this heading">¶</a>

<div id="urllib-parse" class="section">

### urllib.parse<a href="#urllib-parse" class="headerlink" title="Link to this heading">¶</a>

The presence of newline or tab characters in parts of a URL allows for some forms of attacks. Following the WHATWG specification that updates <span id="index-25" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc3986.html" class="rfc reference external"><strong>RFC 3986</strong></a>, ASCII newline <span class="pre">`\n`</span>, <span class="pre">`\r`</span> and tab <span class="pre">`\t`</span> characters are stripped from the URL by the parser in <a href="../library/urllib.parse.html#module-urllib.parse" class="reference internal" title="urllib.parse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urllib.parse</code></span></a> preventing such attacks. The removal characters are controlled by a new module level variable <span class="pre">`urllib.parse._UNSAFE_URL_BYTES_TO_REMOVE`</span>. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43882" class="reference external">bpo-43882</a>)

</div>

</div>

<div id="notable-changes-in-python-3-8-12" class="section">

## Notable changes in Python 3.8.12<a href="#notable-changes-in-python-3-8-12" class="headerlink" title="Link to this heading">¶</a>

<div id="id2" class="section">

### Changes in the Python API<a href="#id2" class="headerlink" title="Link to this heading">¶</a>

Starting with Python 3.8.12 the <a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> module no longer accepts any leading zeros in IPv4 address strings. Leading zeros are ambiguous and interpreted as octal notation by some libraries. For example the legacy function <a href="../library/socket.html#socket.inet_aton" class="reference internal" title="socket.inet_aton"><span class="pre"><code class="sourceCode python">socket.inet_aton()</code></span></a> treats leading zeros as octal notation. glibc implementation of modern <a href="../library/socket.html#socket.inet_pton" class="reference internal" title="socket.inet_pton"><span class="pre"><code class="sourceCode python">inet_pton()</code></span></a> does not accept any leading zeros.

(Originally contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36384" class="reference external">bpo-36384</a>, and backported to 3.8 by Achraf Merzouki.)

</div>

</div>

<div id="notable-security-feature-in-3-8-14" class="section">

## Notable security feature in 3.8.14<a href="#notable-security-feature-in-3-8-14" class="headerlink" title="Link to this heading">¶</a>

Converting between <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> and <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> in bases other than 2 (binary), 4, 8 (octal), 16 (hexadecimal), or 32 such as base 10 (decimal) now raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the number of digits in string form is above a limit to avoid potential denial of service attacks due to the algorithmic complexity. This is a mitigation for <span id="index-26" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2020-10735" class="cve reference external"><strong>CVE 2020-10735</strong></a>. This limit can be configured or disabled by environment variable, command line flag, or <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> APIs. See the <a href="../library/stdtypes.html#int-max-str-digits" class="reference internal"><span class="std std-ref">integer string conversion length limitation</span></a> documentation. The default limit is 4300 digits in string form.

</div>

<div id="notable-changes-in-3-8-17" class="section">

## Notable changes in 3.8.17<a href="#notable-changes-in-3-8-17" class="headerlink" title="Link to this heading">¶</a>

<div id="id3" class="section">

### tarfile<a href="#id3" class="headerlink" title="Link to this heading">¶</a>

- The extraction methods in <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>, and <a href="../library/shutil.html#shutil.unpack_archive" class="reference internal" title="shutil.unpack_archive"><span class="pre"><code class="sourceCode python">shutil.unpack_archive()</code></span></a>, have a new a *filter* argument that allows limiting tar features than may be surprising or dangerous, such as creating files outside the destination directory. See <a href="../library/tarfile.html#tarfile-extraction-filter" class="reference internal"><span class="std std-ref">Extraction filters</span></a> for details. In Python 3.12, use without the *filter* argument will show a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. In Python 3.14, the default will switch to <span class="pre">`'data'`</span>. (Contributed by Petr Viktorin in <span id="index-27" class="target"></span><a href="https://peps.python.org/pep-0706/" class="pep reference external"><strong>PEP 706</strong></a>.)

</div>

</div>

</div>

<div class="clearer">

</div>

</div>
