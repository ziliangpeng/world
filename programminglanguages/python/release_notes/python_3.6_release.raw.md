<div class="body" role="main">

<div id="what-s-new-in-python-3-6" class="section">

# What’s New In Python 3.6<a href="#what-s-new-in-python-3-6" class="headerlink" title="Link to this heading">¶</a>

Editors<span class="colon">:</span>  
Elvis Pranskevichus \<<a href="mailto:elvis%40magic.io" class="reference external">elvis<span>@</span>magic<span>.</span>io</a>\>, Yury Selivanov \<<a href="mailto:yury%40magic.io" class="reference external">yury<span>@</span>magic<span>.</span>io</a>\>

This article explains the new features in Python 3.6, compared to 3.5. Python 3.6 was released on December 23, 2016.  See the <a href="https://docs.python.org/3.6/whatsnew/changelog.html" class="reference external">changelog</a> for a full list of changes.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0494/" class="pep reference external"><strong>PEP 494</strong></a> - Python 3.6 Release Schedule

</div>

<div id="summary-release-highlights" class="section">

## Summary – Release highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

New syntax features:

- <a href="#whatsnew36-pep498" class="reference internal"><span class="std std-ref">PEP 498</span></a>, formatted string literals.

- <a href="#whatsnew36-pep515" class="reference internal"><span class="std std-ref">PEP 515</span></a>, underscores in numeric literals.

- <a href="#whatsnew36-pep526" class="reference internal"><span class="std std-ref">PEP 526</span></a>, syntax for variable annotations.

- <a href="#whatsnew36-pep525" class="reference internal"><span class="std std-ref">PEP 525</span></a>, asynchronous generators.

- <a href="#whatsnew36-pep530" class="reference internal"><span class="std std-ref">PEP 530</span></a>: asynchronous comprehensions.

New library modules:

- <a href="../library/secrets.html#module-secrets" class="reference internal" title="secrets: Generate secure random numbers for managing secrets."><span class="pre"><code class="sourceCode python">secrets</code></span></a>: <a href="#whatsnew36-pep506" class="reference internal"><span class="std std-ref">PEP 506 – Adding A Secrets Module To The Standard Library</span></a>.

CPython implementation improvements:

- The <a href="../library/stdtypes.html#typesmapping" class="reference internal"><span class="std std-ref">dict</span></a> type has been reimplemented to use a <a href="#whatsnew36-compactdict" class="reference internal"><span class="std std-ref">more compact representation</span></a> based on <a href="https://mail.python.org/pipermail/python-dev/2012-December/123028.html" class="reference external">a proposal by Raymond Hettinger</a> and similar to the <a href="https://morepypy.blogspot.com/2015/01/faster-more-memory-efficient-and-more.html" class="reference external">PyPy dict implementation</a>. This resulted in dictionaries using 20% to 25% less memory when compared to Python 3.5.

- Customization of class creation has been simplified with the <a href="#whatsnew36-pep487" class="reference internal"><span class="std std-ref">new protocol</span></a>.

- The class attribute definition order is <a href="#whatsnew36-pep520" class="reference internal"><span class="std std-ref">now preserved</span></a>.

- The order of elements in <span class="pre">`**kwargs`</span> now <a href="#whatsnew36-pep468" class="reference internal"><span class="std std-ref">corresponds to the order</span></a> in which keyword arguments were passed to the function.

- DTrace and SystemTap <a href="#whatsnew36-tracing" class="reference internal"><span class="std std-ref">probing support</span></a> has been added.

- The new <a href="#whatsnew36-pythonmalloc" class="reference internal"><span class="std std-ref">PYTHONMALLOC</span></a> environment variable can now be used to debug the interpreter memory allocation and access errors.

Significant improvements in the standard library:

- The <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> module has received new features, significant usability and performance improvements, and a fair amount of bug fixes. Starting with Python 3.6 the <span class="pre">`asyncio`</span> module is no longer provisional and its API is considered stable.

- A new <a href="#whatsnew36-pep519" class="reference internal"><span class="std std-ref">file system path protocol</span></a> has been implemented to support <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like objects</span></a>. All standard library functions operating on paths have been updated to work with the new protocol.

- The <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> module has gained support for <a href="#whatsnew36-pep495" class="reference internal"><span class="std std-ref">Local Time Disambiguation</span></a>.

- The <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module received a number of <a href="#whatsnew36-typing" class="reference internal"><span class="std std-ref">improvements</span></a>.

- The <a href="../library/tracemalloc.html#module-tracemalloc" class="reference internal" title="tracemalloc: Trace memory allocations."><span class="pre"><code class="sourceCode python">tracemalloc</code></span></a> module has been significantly reworked and is now used to provide better output for <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a> as well as provide better diagnostics for memory allocation errors. See the <a href="#whatsnew36-pythonmalloc" class="reference internal"><span class="std std-ref">PYTHONMALLOC section</span></a> for more information.

Security improvements:

- The new <a href="../library/secrets.html#module-secrets" class="reference internal" title="secrets: Generate secure random numbers for managing secrets."><span class="pre"><code class="sourceCode python">secrets</code></span></a> module has been added to simplify the generation of cryptographically strong pseudo-random numbers suitable for managing secrets such as account authentication, tokens, and similar.

- On Linux, <a href="../library/os.html#os.urandom" class="reference internal" title="os.urandom"><span class="pre"><code class="sourceCode python">os.urandom()</code></span></a> now blocks until the system urandom entropy pool is initialized to increase the security. See the <span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0524/" class="pep reference external"><strong>PEP 524</strong></a> for the rationale.

- The <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> and <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> modules now support OpenSSL 1.1.0.

- The default settings and feature set of the <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module have been improved.

- The <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module received support for the BLAKE2, SHA-3 and SHAKE hash algorithms and the <a href="../library/hashlib.html#hashlib.scrypt" class="reference internal" title="hashlib.scrypt"><span class="pre"><code class="sourceCode python">scrypt()</code></span></a> key derivation function.

Windows improvements:

- <a href="#whatsnew36-pep528" class="reference internal"><span class="std std-ref">PEP 528</span></a> and <a href="#whatsnew36-pep529" class="reference internal"><span class="std std-ref">PEP 529</span></a>, Windows filesystem and console encoding changed to UTF-8.

- The <span class="pre">`py.exe`</span> launcher, when used interactively, no longer prefers Python 2 over Python 3 when the user doesn’t specify a version (via command line arguments or a config file). Handling of shebang lines remains unchanged - “python” refers to Python 2 in that case.

- <span class="pre">`python.exe`</span> and <span class="pre">`pythonw.exe`</span> have been marked as long-path aware, which means that the 260 character path limit may no longer apply. See <a href="../using/windows.html#max-path" class="reference internal"><span class="std std-ref">removing the MAX_PATH limitation</span></a> for details.

- A <span class="pre">`._pth`</span> file can be added to force isolated mode and fully specify all search paths to avoid registry and environment lookup. See <a href="../using/windows.html#windows-finding-modules" class="reference internal"><span class="std std-ref">the documentation</span></a> for more information.

- A <span class="pre">`python36.zip`</span> file now works as a landmark to infer <span id="index-2" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONHOME" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONHOME</code></span></a>. See <a href="../using/windows.html#windows-finding-modules" class="reference internal"><span class="std std-ref">the documentation</span></a> for more information.

</div>

<div id="new-features" class="section">

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-498-formatted-string-literals" class="section">

<span id="whatsnew36-pep498"></span>

### PEP 498: Formatted string literals<a href="#pep-498-formatted-string-literals" class="headerlink" title="Link to this heading">¶</a>

<span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0498/" class="pep reference external"><strong>PEP 498</strong></a> introduces a new kind of string literals: *f-strings*, or <a href="../reference/lexical_analysis.html#f-strings" class="reference internal"><span class="std std-ref">formatted string literals</span></a>.

Formatted string literals are prefixed with <span class="pre">`'f'`</span> and are similar to the format strings accepted by <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a>. They contain replacement fields surrounded by curly braces. The replacement fields are expressions, which are evaluated at run time, and then formatted using the <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> protocol:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> name = "Fred"
    >>> f"He said his name is {name}."
    'He said his name is Fred.'
    >>> width = 10
    >>> precision = 4
    >>> value = decimal.Decimal("12.34567")
    >>> f"result: {value:{width}.{precision}}"  # nested fields
    'result:      12.35'

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-4" class="target"></span><a href="https://peps.python.org/pep-0498/" class="pep reference external"><strong>PEP 498</strong></a> – Literal String Interpolation.  
PEP written and implemented by Eric V. Smith.

<a href="../reference/lexical_analysis.html#f-strings" class="reference internal"><span class="std std-ref">Feature documentation</span></a>.

</div>

</div>

<div id="pep-526-syntax-for-variable-annotations" class="section">

<span id="whatsnew36-pep526"></span>

### PEP 526: Syntax for variable annotations<a href="#pep-526-syntax-for-variable-annotations" class="headerlink" title="Link to this heading">¶</a>

<span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> introduced the standard for type annotations of function parameters, a.k.a. type hints. This PEP adds syntax to Python for annotating the types of variables including class variables and instance variables:

<div class="highlight-python3 notranslate">

<div class="highlight">

    primes: List[int] = []

    captain: str  # Note: no initial value!

    class Starship:
        stats: Dict[str, int] = {}

</div>

</div>

Just as for function annotations, the Python interpreter does not attach any particular meaning to variable annotations and only stores them in the <span class="pre">`__annotations__`</span> attribute of a class or module.

In contrast to variable declarations in statically typed languages, the goal of annotation syntax is to provide an easy way to specify structured type metadata for third party tools and libraries via the abstract syntax tree and the <span class="pre">`__annotations__`</span> attribute.

<div class="admonition seealso">

See also

<span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0526/" class="pep reference external"><strong>PEP 526</strong></a> – Syntax for variable annotations.  
PEP written by Ryan Gonzalez, Philip House, Ivan Levkivskyi, Lisa Roach, and Guido van Rossum. Implemented by Ivan Levkivskyi.

Tools that use or will use the new syntax: <a href="https://www.mypy-lang.org/" class="reference external">mypy</a>, <a href="https://github.com/google/pytype" class="reference external">pytype</a>, PyCharm, etc.

</div>

</div>

<div id="pep-515-underscores-in-numeric-literals" class="section">

<span id="whatsnew36-pep515"></span>

### PEP 515: Underscores in Numeric Literals<a href="#pep-515-underscores-in-numeric-literals" class="headerlink" title="Link to this heading">¶</a>

<span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0515/" class="pep reference external"><strong>PEP 515</strong></a> adds the ability to use underscores in numeric literals for improved readability. For example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> 1_000_000_000_000_000
    1000000000000000
    >>> 0x_FF_FF_FF_FF
    4294967295

</div>

</div>

Single underscores are allowed between digits and after any base specifier. Leading, trailing, or multiple underscores in a row are not allowed.

The <a href="../library/string.html#formatspec" class="reference internal"><span class="std std-ref">string formatting</span></a> language also now has support for the <span class="pre">`'_'`</span> option to signal the use of an underscore for a thousands separator for floating-point presentation types and for integer presentation type <span class="pre">`'d'`</span>. For integer presentation types <span class="pre">`'b'`</span>, <span class="pre">`'o'`</span>, <span class="pre">`'x'`</span>, and <span class="pre">`'X'`</span>, underscores will be inserted every 4 digits:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> '{:_}'.format(1000000)
    '1_000_000'
    >>> '{:_x}'.format(0xFFFFFFFF)
    'ffff_ffff'

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-8" class="target"></span><a href="https://peps.python.org/pep-0515/" class="pep reference external"><strong>PEP 515</strong></a> – Underscores in Numeric Literals  
PEP written by Georg Brandl and Serhiy Storchaka.

</div>

</div>

<div id="pep-525-asynchronous-generators" class="section">

<span id="whatsnew36-pep525"></span>

### PEP 525: Asynchronous Generators<a href="#pep-525-asynchronous-generators" class="headerlink" title="Link to this heading">¶</a>

<span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0492/" class="pep reference external"><strong>PEP 492</strong></a> introduced support for native coroutines and <span class="pre">`async`</span> / <span class="pre">`await`</span> syntax to Python 3.5. A notable limitation of the Python 3.5 implementation is that it was not possible to use <span class="pre">`await`</span> and <span class="pre">`yield`</span> in the same function body. In Python 3.6 this restriction has been lifted, making it possible to define *asynchronous generators*:

<div class="highlight-python3 notranslate">

<div class="highlight">

    async def ticker(delay, to):
        """Yield numbers from 0 to *to* every *delay* seconds."""
        for i in range(to):
            yield i
            await asyncio.sleep(delay)

</div>

</div>

The new syntax allows for faster and more concise code.

<div class="admonition seealso">

See also

<span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0525/" class="pep reference external"><strong>PEP 525</strong></a> – Asynchronous Generators  
PEP written and implemented by Yury Selivanov.

</div>

</div>

<div id="pep-530-asynchronous-comprehensions" class="section">

<span id="whatsnew36-pep530"></span>

### PEP 530: Asynchronous Comprehensions<a href="#pep-530-asynchronous-comprehensions" class="headerlink" title="Link to this heading">¶</a>

<span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0530/" class="pep reference external"><strong>PEP 530</strong></a> adds support for using <span class="pre">`async`</span>` `<span class="pre">`for`</span> in list, set, dict comprehensions and generator expressions:

<div class="highlight-python3 notranslate">

<div class="highlight">

    result = [i async for i in aiter() if i % 2]

</div>

</div>

Additionally, <span class="pre">`await`</span> expressions are supported in all kinds of comprehensions:

<div class="highlight-python3 notranslate">

<div class="highlight">

    result = [await fun() for fun in funcs if await condition()]

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-12" class="target"></span><a href="https://peps.python.org/pep-0530/" class="pep reference external"><strong>PEP 530</strong></a> – Asynchronous Comprehensions  
PEP written and implemented by Yury Selivanov.

</div>

</div>

<div id="pep-487-simpler-customization-of-class-creation" class="section">

<span id="whatsnew36-pep487"></span>

### PEP 487: Simpler customization of class creation<a href="#pep-487-simpler-customization-of-class-creation" class="headerlink" title="Link to this heading">¶</a>

It is now possible to customize subclass creation without using a metaclass. The new <span class="pre">`__init_subclass__`</span> classmethod will be called on the base class whenever a new subclass is created:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class PluginBase:
        subclasses = []

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            cls.subclasses.append(cls)

    class Plugin1(PluginBase):
        pass

    class Plugin2(PluginBase):
        pass

</div>

</div>

In order to allow zero-argument <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a> calls to work correctly from <a href="../reference/datamodel.html#object.__init_subclass__" class="reference internal" title="object.__init_subclass__"><span class="pre"><code class="sourceCode python"><span class="fu">__init_subclass__</span>()</code></span></a> implementations, custom metaclasses must ensure that the new <span class="pre">`__classcell__`</span> namespace entry is propagated to <span class="pre">`type.__new__`</span> (as described in <a href="../reference/datamodel.html#class-object-creation" class="reference internal"><span class="std std-ref">Creating the class object</span></a>).

<div class="admonition seealso">

See also

<span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0487/" class="pep reference external"><strong>PEP 487</strong></a> – Simpler customization of class creation  
PEP written and implemented by Martin Teichmann.

<a href="../reference/datamodel.html#class-customization" class="reference internal"><span class="std std-ref">Feature documentation</span></a>

</div>

</div>

<div id="pep-487-descriptor-protocol-enhancements" class="section">

<span id="whatsnew36-pep487-descriptors"></span>

### PEP 487: Descriptor Protocol Enhancements<a href="#pep-487-descriptor-protocol-enhancements" class="headerlink" title="Link to this heading">¶</a>

<span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0487/" class="pep reference external"><strong>PEP 487</strong></a> extends the descriptor protocol to include the new optional <a href="../reference/datamodel.html#object.__set_name__" class="reference internal" title="object.__set_name__"><span class="pre"><code class="sourceCode python"><span class="fu">__set_name__</span>()</code></span></a> method. Whenever a new class is defined, the new method will be called on all descriptors included in the definition, providing them with a reference to the class being defined and the name given to the descriptor within the class namespace. In other words, instances of descriptors can now know the attribute name of the descriptor in the owner class:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class IntField:
        def __get__(self, instance, owner):
            return instance.__dict__[self.name]

        def __set__(self, instance, value):
            if not isinstance(value, int):
                raise ValueError(f'expecting integer in {self.name}')
            instance.__dict__[self.name] = value

        # this is the new initializer:
        def __set_name__(self, owner, name):
            self.name = name

    class Model:
        int_field = IntField()

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0487/" class="pep reference external"><strong>PEP 487</strong></a> – Simpler customization of class creation  
PEP written and implemented by Martin Teichmann.

<a href="../reference/datamodel.html#descriptors" class="reference internal"><span class="std std-ref">Feature documentation</span></a>

</div>

</div>

<div id="pep-519-adding-a-file-system-path-protocol" class="section">

<span id="whatsnew36-pep519"></span>

### PEP 519: Adding a file system path protocol<a href="#pep-519-adding-a-file-system-path-protocol" class="headerlink" title="Link to this heading">¶</a>

File system paths have historically been represented as <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> or <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> objects. This has led to people who write code which operate on file system paths to assume that such objects are only one of those two types (an <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> representing a file descriptor does not count as that is not a file path). Unfortunately that assumption prevents alternative object representations of file system paths like <a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a> from working with pre-existing code, including Python’s standard library.

To fix this situation, a new interface represented by <a href="../library/os.html#os.PathLike" class="reference internal" title="os.PathLike"><span class="pre"><code class="sourceCode python">os.PathLike</code></span></a> has been defined. By implementing the <a href="../library/os.html#os.PathLike.__fspath__" class="reference internal" title="os.PathLike.__fspath__"><span class="pre"><code class="sourceCode python">__fspath__()</code></span></a> method, an object signals that it represents a path. An object can then provide a low-level representation of a file system path as a <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> or <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> object. This means an object is considered <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like</span></a> if it implements <a href="../library/os.html#os.PathLike" class="reference internal" title="os.PathLike"><span class="pre"><code class="sourceCode python">os.PathLike</code></span></a> or is a <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> or <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> object which represents a file system path. Code can use <a href="../library/os.html#os.fspath" class="reference internal" title="os.fspath"><span class="pre"><code class="sourceCode python">os.fspath()</code></span></a>, <a href="../library/os.html#os.fsdecode" class="reference internal" title="os.fsdecode"><span class="pre"><code class="sourceCode python">os.fsdecode()</code></span></a>, or <a href="../library/os.html#os.fsencode" class="reference internal" title="os.fsencode"><span class="pre"><code class="sourceCode python">os.fsencode()</code></span></a> to explicitly get a <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> and/or <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> representation of a path-like object.

The built-in <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> function has been updated to accept <a href="../library/os.html#os.PathLike" class="reference internal" title="os.PathLike"><span class="pre"><code class="sourceCode python">os.PathLike</code></span></a> objects, as have all relevant functions in the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> and <a href="../library/os.path.html#module-os.path" class="reference internal" title="os.path: Operations on pathnames."><span class="pre"><code class="sourceCode python">os.path</code></span></a> modules, and most other functions and classes in the standard library. The <a href="../library/os.html#os.DirEntry" class="reference internal" title="os.DirEntry"><span class="pre"><code class="sourceCode python">os.DirEntry</code></span></a> class and relevant classes in <a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a> have also been updated to implement <a href="../library/os.html#os.PathLike" class="reference internal" title="os.PathLike"><span class="pre"><code class="sourceCode python">os.PathLike</code></span></a>.

The hope is that updating the fundamental functions for operating on file system paths will lead to third-party code to implicitly support all <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like objects</span></a> without any code changes, or at least very minimal ones (e.g. calling <a href="../library/os.html#os.fspath" class="reference internal" title="os.fspath"><span class="pre"><code class="sourceCode python">os.fspath()</code></span></a> at the beginning of code before operating on a path-like object).

Here are some examples of how the new interface allows for <a href="../library/pathlib.html#pathlib.Path" class="reference internal" title="pathlib.Path"><span class="pre"><code class="sourceCode python">pathlib.Path</code></span></a> to be used more easily and transparently with pre-existing code:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import pathlib
    >>> with open(pathlib.Path("README")) as f:
    ...     contents = f.read()
    ...
    >>> import os.path
    >>> os.path.splitext(pathlib.Path("some_file.txt"))
    ('some_file', '.txt')
    >>> os.path.join("/a/b", pathlib.Path("c"))
    '/a/b/c'
    >>> import os
    >>> os.fspath(pathlib.Path("some_file.txt"))
    'some_file.txt'

</div>

</div>

(Implemented by Brett Cannon, Ethan Furman, Dusty Phillips, and Jelle Zijlstra.)

<div class="admonition seealso">

See also

<span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0519/" class="pep reference external"><strong>PEP 519</strong></a> – Adding a file system path protocol  
PEP written by Brett Cannon and Koos Zevenhoven.

</div>

</div>

<div id="pep-495-local-time-disambiguation" class="section">

<span id="whatsnew36-pep495"></span>

### PEP 495: Local Time Disambiguation<a href="#pep-495-local-time-disambiguation" class="headerlink" title="Link to this heading">¶</a>

In most world locations, there have been and will be times when local clocks are moved back. In those times, intervals are introduced in which local clocks show the same time twice in the same day. In these situations, the information displayed on a local clock (or stored in a Python datetime instance) is insufficient to identify a particular moment in time.

<span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0495/" class="pep reference external"><strong>PEP 495</strong></a> adds the new *fold* attribute to instances of <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime.datetime</code></span></a> and <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">datetime.time</code></span></a> classes to differentiate between two moments in time for which local times are the same:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> u0 = datetime(2016, 11, 6, 4, tzinfo=timezone.utc)
    >>> for i in range(4):
    ...     u = u0 + i*HOUR
    ...     t = u.astimezone(Eastern)
    ...     print(u.time(), 'UTC =', t.time(), t.tzname(), t.fold)
    ...
    04:00:00 UTC = 00:00:00 EDT 0
    05:00:00 UTC = 01:00:00 EDT 0
    06:00:00 UTC = 01:00:00 EST 1
    07:00:00 UTC = 02:00:00 EST 0

</div>

</div>

The values of the <a href="../library/datetime.html#datetime.datetime.fold" class="reference internal" title="datetime.datetime.fold"><span class="pre"><code class="sourceCode python">fold</code></span></a> attribute have the value <span class="pre">`0`</span> for all instances except those that represent the second (chronologically) moment in time in an ambiguous case.

<div class="admonition seealso">

See also

<span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0495/" class="pep reference external"><strong>PEP 495</strong></a> – Local Time Disambiguation  
PEP written by Alexander Belopolsky and Tim Peters, implementation by Alexander Belopolsky.

</div>

</div>

<div id="pep-529-change-windows-filesystem-encoding-to-utf-8" class="section">

<span id="whatsnew36-pep529"></span>

### PEP 529: Change Windows filesystem encoding to UTF-8<a href="#pep-529-change-windows-filesystem-encoding-to-utf-8" class="headerlink" title="Link to this heading">¶</a>

Representing filesystem paths is best performed with str (Unicode) rather than bytes. However, there are some situations where using bytes is sufficient and correct.

Prior to Python 3.6, data loss could result when using bytes paths on Windows. With this change, using bytes to represent paths is now supported on Windows, provided those bytes are encoded with the encoding returned by <a href="../library/sys.html#sys.getfilesystemencoding" class="reference internal" title="sys.getfilesystemencoding"><span class="pre"><code class="sourceCode python">sys.getfilesystemencoding()</code></span></a>, which now defaults to <span class="pre">`'utf-8'`</span>.

Applications that do not use str to represent paths should use <a href="../library/os.html#os.fsencode" class="reference internal" title="os.fsencode"><span class="pre"><code class="sourceCode python">os.fsencode()</code></span></a> and <a href="../library/os.html#os.fsdecode" class="reference internal" title="os.fsdecode"><span class="pre"><code class="sourceCode python">os.fsdecode()</code></span></a> to ensure their bytes are correctly encoded. To revert to the previous behaviour, set <span id="index-19" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONLEGACYWINDOWSFSENCODING" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONLEGACYWINDOWSFSENCODING</code></span></a> or call <a href="../library/sys.html#sys._enablelegacywindowsfsencoding" class="reference internal" title="sys._enablelegacywindowsfsencoding"><span class="pre"><code class="sourceCode python">sys._enablelegacywindowsfsencoding()</code></span></a>.

See <span id="index-20" class="target"></span><a href="https://peps.python.org/pep-0529/" class="pep reference external"><strong>PEP 529</strong></a> for more information and discussion of code modifications that may be required.

</div>

<div id="pep-528-change-windows-console-encoding-to-utf-8" class="section">

<span id="whatsnew36-pep528"></span>

### PEP 528: Change Windows console encoding to UTF-8<a href="#pep-528-change-windows-console-encoding-to-utf-8" class="headerlink" title="Link to this heading">¶</a>

The default console on Windows will now accept all Unicode characters and provide correctly read str objects to Python code. <span class="pre">`sys.stdin`</span>, <span class="pre">`sys.stdout`</span> and <span class="pre">`sys.stderr`</span> now default to utf-8 encoding.

This change only applies when using an interactive console, and not when redirecting files or pipes. To revert to the previous behaviour for interactive console use, set <span id="index-21" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONLEGACYWINDOWSSTDIO" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONLEGACYWINDOWSSTDIO</code></span></a>.

<div class="admonition seealso">

See also

<span id="index-22" class="target"></span><a href="https://peps.python.org/pep-0528/" class="pep reference external"><strong>PEP 528</strong></a> – Change Windows console encoding to UTF-8  
PEP written and implemented by Steve Dower.

</div>

</div>

<div id="pep-520-preserving-class-attribute-definition-order" class="section">

<span id="whatsnew36-pep520"></span>

### PEP 520: Preserving Class Attribute Definition Order<a href="#pep-520-preserving-class-attribute-definition-order" class="headerlink" title="Link to this heading">¶</a>

Attributes in a class definition body have a natural ordering: the same order in which the names appear in the source. This order is now preserved in the new class’s <a href="../reference/datamodel.html#type.__dict__" class="reference internal" title="type.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a> attribute.

Also, the effective default class *execution* namespace (returned from <a href="../reference/datamodel.html#prepare" class="reference internal"><span class="std std-ref">type.__prepare__()</span></a>) is now an insertion-order-preserving mapping.

<div class="admonition seealso">

See also

<span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0520/" class="pep reference external"><strong>PEP 520</strong></a> – Preserving Class Attribute Definition Order  
PEP written and implemented by Eric Snow.

</div>

</div>

<div id="pep-468-preserving-keyword-argument-order" class="section">

<span id="whatsnew36-pep468"></span>

### PEP 468: Preserving Keyword Argument Order<a href="#pep-468-preserving-keyword-argument-order" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`**kwargs`</span> in a function signature is now guaranteed to be an insertion-order-preserving mapping.

<div class="admonition seealso">

See also

<span id="index-24" class="target"></span><a href="https://peps.python.org/pep-0468/" class="pep reference external"><strong>PEP 468</strong></a> – Preserving Keyword Argument Order  
PEP written and implemented by Eric Snow.

</div>

</div>

<div id="new-dict-implementation" class="section">

<span id="whatsnew36-compactdict"></span>

### New <a href="../library/stdtypes.html#typesmapping" class="reference internal"><span class="std std-ref">dict</span></a> implementation<a href="#new-dict-implementation" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/stdtypes.html#typesmapping" class="reference internal"><span class="std std-ref">dict</span></a> type now uses a “compact” representation based on <a href="https://mail.python.org/pipermail/python-dev/2012-December/123028.html" class="reference external">a proposal by Raymond Hettinger</a> which was <a href="https://morepypy.blogspot.com/2015/01/faster-more-memory-efficient-and-more.html" class="reference external">first implemented by PyPy</a>. The memory usage of the new <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>()</code></span></a> is between 20% and 25% smaller compared to Python 3.5.

The order-preserving aspect of this new implementation is considered an implementation detail and should not be relied upon (this may change in the future, but it is desired to have this new dict implementation in the language for a few releases before changing the language spec to mandate order-preserving semantics for all current and future Python implementations; this also helps preserve backwards-compatibility with older versions of the language where random iteration order is still in effect, e.g. Python 3.5).

(Contributed by INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27350" class="reference external">bpo-27350</a>. Idea <a href="https://mail.python.org/pipermail/python-dev/2012-December/123028.html" class="reference external">originally suggested by Raymond Hettinger</a>.)

</div>

<div id="pep-523-adding-a-frame-evaluation-api-to-cpython" class="section">

<span id="whatsnew36-pep523"></span>

### PEP 523: Adding a frame evaluation API to CPython<a href="#pep-523-adding-a-frame-evaluation-api-to-cpython" class="headerlink" title="Link to this heading">¶</a>

While Python provides extensive support to customize how code executes, one place it has not done so is in the evaluation of frame objects. If you wanted some way to intercept frame evaluation in Python there really wasn’t any way without directly manipulating function pointers for defined functions.

<span id="index-25" class="target"></span><a href="https://peps.python.org/pep-0523/" class="pep reference external"><strong>PEP 523</strong></a> changes this by providing an API to make frame evaluation pluggable at the C level. This will allow for tools such as debuggers and JITs to intercept frame evaluation before the execution of Python code begins. This enables the use of alternative evaluation implementations for Python code, tracking frame evaluation, etc.

This API is not part of the limited C API and is marked as private to signal that usage of this API is expected to be limited and only applicable to very select, low-level use-cases. Semantics of the API will change with Python as necessary.

<div class="admonition seealso">

See also

<span id="index-26" class="target"></span><a href="https://peps.python.org/pep-0523/" class="pep reference external"><strong>PEP 523</strong></a> – Adding a frame evaluation API to CPython  
PEP written by Brett Cannon and Dino Viehland.

</div>

</div>

<div id="pythonmalloc-environment-variable" class="section">

<span id="whatsnew36-pythonmalloc"></span>

### PYTHONMALLOC environment variable<a href="#pythonmalloc-environment-variable" class="headerlink" title="Link to this heading">¶</a>

The new <span id="index-27" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONMALLOC" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONMALLOC</code></span></a> environment variable allows setting the Python memory allocators and installing debug hooks.

It is now possible to install debug hooks on Python memory allocators on Python compiled in release mode using <span class="pre">`PYTHONMALLOC=debug`</span>. Effects of debug hooks:

- Newly allocated memory is filled with the byte <span class="pre">`0xCB`</span>

- Freed memory is filled with the byte <span class="pre">`0xDB`</span>

- Detect violations of the Python memory allocator API. For example, <a href="../c-api/memory.html#c.PyObject_Free" class="reference internal" title="PyObject_Free"><span class="pre"><code class="sourceCode c">PyObject_Free<span class="op">()</span></code></span></a> called on a memory block allocated by <a href="../c-api/memory.html#c.PyMem_Malloc" class="reference internal" title="PyMem_Malloc"><span class="pre"><code class="sourceCode c">PyMem_Malloc<span class="op">()</span></code></span></a>.

- Detect writes before the start of a buffer (buffer underflows)

- Detect writes after the end of a buffer (buffer overflows)

- Check that the <a href="../glossary.html#term-global-interpreter-lock" class="reference internal"><span class="xref std std-term">GIL</span></a> is held when allocator functions of <a href="../c-api/memory.html#c.PYMEM_DOMAIN_OBJ" class="reference internal" title="PYMEM_DOMAIN_OBJ"><span class="pre"><code class="sourceCode c">PYMEM_DOMAIN_OBJ</code></span></a> (ex: <a href="../c-api/memory.html#c.PyObject_Malloc" class="reference internal" title="PyObject_Malloc"><span class="pre"><code class="sourceCode c">PyObject_Malloc<span class="op">()</span></code></span></a>) and <a href="../c-api/memory.html#c.PYMEM_DOMAIN_MEM" class="reference internal" title="PYMEM_DOMAIN_MEM"><span class="pre"><code class="sourceCode c">PYMEM_DOMAIN_MEM</code></span></a> (ex: <a href="../c-api/memory.html#c.PyMem_Malloc" class="reference internal" title="PyMem_Malloc"><span class="pre"><code class="sourceCode c">PyMem_Malloc<span class="op">()</span></code></span></a>) domains are called.

Checking if the GIL is held is also a new feature of Python 3.6.

See the <a href="../c-api/memory.html#c.PyMem_SetupDebugHooks" class="reference internal" title="PyMem_SetupDebugHooks"><span class="pre"><code class="sourceCode c">PyMem_SetupDebugHooks<span class="op">()</span></code></span></a> function for debug hooks on Python memory allocators.

It is now also possible to force the usage of the <span class="pre">`malloc()`</span> allocator of the C library for all Python memory allocations using <span class="pre">`PYTHONMALLOC=malloc`</span>. This is helpful when using external memory debuggers like Valgrind on a Python compiled in release mode.

On error, the debug hooks on Python memory allocators now use the <a href="../library/tracemalloc.html#module-tracemalloc" class="reference internal" title="tracemalloc: Trace memory allocations."><span class="pre"><code class="sourceCode python">tracemalloc</code></span></a> module to get the traceback where a memory block was allocated.

Example of fatal error on buffer overflow using <span class="pre">`python3.6`</span>` `<span class="pre">`-X`</span>` `<span class="pre">`tracemalloc=5`</span> (store 5 frames in traces):

<div class="highlight-python3 notranslate">

<div class="highlight">

    Debug memory block at address p=0x7fbcd41666f8: API 'o'
        4 bytes originally requested
        The 7 pad bytes at p-7 are FORBIDDENBYTE, as expected.
        The 8 pad bytes at tail=0x7fbcd41666fc are not all FORBIDDENBYTE (0xfb):
            at tail+0: 0x02 *** OUCH
            at tail+1: 0xfb
            at tail+2: 0xfb
            at tail+3: 0xfb
            at tail+4: 0xfb
            at tail+5: 0xfb
            at tail+6: 0xfb
            at tail+7: 0xfb
        The block was made by call #1233329 to debug malloc/realloc.
        Data at p: 1a 2b 30 00

    Memory block allocated at (most recent call first):
      File "test/test_bytes.py", line 323
      File "unittest/case.py", line 600
      File "unittest/case.py", line 648
      File "unittest/suite.py", line 122
      File "unittest/suite.py", line 84

    Fatal Python error: bad trailing pad byte

    Current thread 0x00007fbcdbd32700 (most recent call first):
      File "test/test_bytes.py", line 323 in test_hex
      File "unittest/case.py", line 600 in run
      File "unittest/case.py", line 648 in __call__
      File "unittest/suite.py", line 122 in run
      File "unittest/suite.py", line 84 in __call__
      File "unittest/suite.py", line 122 in run
      File "unittest/suite.py", line 84 in __call__
      ...

</div>

</div>

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26516" class="reference external">bpo-26516</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26564" class="reference external">bpo-26564</a>.)

</div>

<div id="dtrace-and-systemtap-probing-support" class="section">

<span id="whatsnew36-tracing"></span>

### DTrace and SystemTap probing support<a href="#dtrace-and-systemtap-probing-support" class="headerlink" title="Link to this heading">¶</a>

Python can now be built <span class="pre">`--with-dtrace`</span> which enables static markers for the following events in the interpreter:

- function call/return

- garbage collection started/finished

- line of code executed.

This can be used to instrument running interpreters in production, without the need to recompile specific <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">debug builds</span></a> or providing application-specific profiling/debugging code.

More details in <a href="../howto/instrumentation.html#instrumentation" class="reference internal"><span class="std std-ref">Instrumenting CPython with DTrace and SystemTap</span></a>.

The current implementation is tested on Linux and macOS. Additional markers may be added in the future.

(Contributed by Łukasz Langa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21590" class="reference external">bpo-21590</a>, based on patches by Jesús Cea Avión, David Malcolm, and Nikhil Benesch.)

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

Some smaller changes made to the core Python language are:

- A <span class="pre">`global`</span> or <span class="pre">`nonlocal`</span> statement must now textually appear before the first use of the affected name in the same scope. Previously this was a <a href="../library/exceptions.html#SyntaxWarning" class="reference internal" title="SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a>.

- It is now possible to set a <a href="../reference/datamodel.html#specialnames" class="reference internal"><span class="std std-ref">special method</span></a> to <span class="pre">`None`</span> to indicate that the corresponding operation is not available. For example, if a class sets <a href="../reference/datamodel.html#object.__iter__" class="reference internal" title="object.__iter__"><span class="pre"><code class="sourceCode python"><span class="fu">__iter__</span>()</code></span></a> to <span class="pre">`None`</span>, the class is not iterable. (Contributed by Andrew Barnert and Ivan Levkivskyi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25958" class="reference external">bpo-25958</a>.)

- Long sequences of repeated traceback lines are now abbreviated as <span class="pre">`"[Previous`</span>` `<span class="pre">`line`</span>` `<span class="pre">`repeated`</span>` `<span class="pre">`{count}`</span>` `<span class="pre">`more`</span>` `<span class="pre">`times]"`</span> (see <a href="#whatsnew36-traceback" class="reference internal"><span class="std std-ref">traceback</span></a> for an example). (Contributed by Emanuel Barry in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26823" class="reference external">bpo-26823</a>.)

- Import now raises the new exception <a href="../library/exceptions.html#ModuleNotFoundError" class="reference internal" title="ModuleNotFoundError"><span class="pre"><code class="sourceCode python"><span class="pp">ModuleNotFoundError</span></code></span></a> (subclass of <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a>) when it cannot find a module. Code that currently checks for ImportError (in try-except) will still work. (Contributed by Eric Snow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15767" class="reference external">bpo-15767</a>.)

- Class methods relying on zero-argument <span class="pre">`super()`</span> will now work correctly when called from metaclass methods during class creation. (Contributed by Martin Teichmann in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23722" class="reference external">bpo-23722</a>.)

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="secrets" class="section">

<span id="whatsnew36-pep506"></span>

### secrets<a href="#secrets" class="headerlink" title="Link to this heading">¶</a>

The main purpose of the new <a href="../library/secrets.html#module-secrets" class="reference internal" title="secrets: Generate secure random numbers for managing secrets."><span class="pre"><code class="sourceCode python">secrets</code></span></a> module is to provide an obvious way to reliably generate cryptographically strong pseudo-random values suitable for managing secrets, such as account authentication, tokens, and similar.

<div class="admonition warning">

Warning

Note that the pseudo-random generators in the <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module should *NOT* be used for security purposes. Use <a href="../library/secrets.html#module-secrets" class="reference internal" title="secrets: Generate secure random numbers for managing secrets."><span class="pre"><code class="sourceCode python">secrets</code></span></a> on Python 3.6+ and <a href="../library/os.html#os.urandom" class="reference internal" title="os.urandom"><span class="pre"><code class="sourceCode python">os.urandom()</code></span></a> on Python 3.5 and earlier.

</div>

<div class="admonition seealso">

See also

<span id="index-28" class="target"></span><a href="https://peps.python.org/pep-0506/" class="pep reference external"><strong>PEP 506</strong></a> – Adding A Secrets Module To The Standard Library  
PEP written and implemented by Steven D’Aprano.

</div>

</div>

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="array" class="section">

### array<a href="#array" class="headerlink" title="Link to this heading">¶</a>

Exhausted iterators of <a href="../library/array.html#array.array" class="reference internal" title="array.array"><span class="pre"><code class="sourceCode python">array.array</code></span></a> will now stay exhausted even if the iterated array is extended. This is consistent with the behavior of other mutable sequences.

Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26492" class="reference external">bpo-26492</a>.

</div>

<div id="ast" class="section">

### ast<a href="#ast" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/ast.html#ast.Constant" class="reference internal" title="ast.Constant"><span class="pre"><code class="sourceCode python">ast.Constant</code></span></a> AST node has been added. It can be used by external AST optimizers for the purposes of constant folding.

Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26146" class="reference external">bpo-26146</a>.

</div>

<div id="asyncio" class="section">

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

Starting with Python 3.6 the <span class="pre">`asyncio`</span> module is no longer provisional and its API is considered stable.

Notable changes in the <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> module since Python 3.5.0 (all backported to 3.5.x due to the provisional status):

- The <a href="../library/asyncio-eventloop.html#asyncio.get_event_loop" class="reference internal" title="asyncio.get_event_loop"><span class="pre"><code class="sourceCode python">get_event_loop()</code></span></a> function has been changed to always return the currently running loop when called from coroutines and callbacks. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28613" class="reference external">bpo-28613</a>.)

- The <a href="../library/asyncio-future.html#asyncio.ensure_future" class="reference internal" title="asyncio.ensure_future"><span class="pre"><code class="sourceCode python">ensure_future()</code></span></a> function and all functions that use it, such as <a href="../library/asyncio-eventloop.html#asyncio.loop.run_until_complete" class="reference internal" title="asyncio.loop.run_until_complete"><span class="pre"><code class="sourceCode python">loop.run_until_complete()</code></span></a>, now accept all kinds of <a href="../glossary.html#term-awaitable" class="reference internal"><span class="xref std std-term">awaitable objects</span></a>. (Contributed by Yury Selivanov.)

- New <a href="../library/asyncio-task.html#asyncio.run_coroutine_threadsafe" class="reference internal" title="asyncio.run_coroutine_threadsafe"><span class="pre"><code class="sourceCode python">run_coroutine_threadsafe()</code></span></a> function to submit coroutines to event loops from other threads. (Contributed by Vincent Michel.)

- New <a href="../library/asyncio-protocol.html#asyncio.BaseTransport.is_closing" class="reference internal" title="asyncio.BaseTransport.is_closing"><span class="pre"><code class="sourceCode python">Transport.is_closing()</code></span></a> method to check if the transport is closing or closed. (Contributed by Yury Selivanov.)

- The <a href="../library/asyncio-eventloop.html#asyncio.loop.create_server" class="reference internal" title="asyncio.loop.create_server"><span class="pre"><code class="sourceCode python">loop.create_server()</code></span></a> method can now accept a list of hosts. (Contributed by Yann Sionneau.)

- New <a href="../library/asyncio-eventloop.html#asyncio.loop.create_future" class="reference internal" title="asyncio.loop.create_future"><span class="pre"><code class="sourceCode python">loop.create_future()</code></span></a> method to create Future objects. This allows alternative event loop implementations, such as <a href="https://github.com/MagicStack/uvloop" class="reference external">uvloop</a>, to provide a faster <a href="../library/asyncio-future.html#asyncio.Future" class="reference internal" title="asyncio.Future"><span class="pre"><code class="sourceCode python">asyncio.Future</code></span></a> implementation. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27041" class="reference external">bpo-27041</a>.)

- New <a href="../library/asyncio-eventloop.html#asyncio.loop.get_exception_handler" class="reference internal" title="asyncio.loop.get_exception_handler"><span class="pre"><code class="sourceCode python">loop.get_exception_handler()</code></span></a> method to get the current exception handler. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27040" class="reference external">bpo-27040</a>.)

- New <a href="../library/asyncio-stream.html#asyncio.StreamReader.readuntil" class="reference internal" title="asyncio.StreamReader.readuntil"><span class="pre"><code class="sourceCode python">StreamReader.readuntil()</code></span></a> method to read data from the stream until a separator bytes sequence appears. (Contributed by Mark Korenberg.)

- The performance of <a href="../library/asyncio-stream.html#asyncio.StreamReader.readexactly" class="reference internal" title="asyncio.StreamReader.readexactly"><span class="pre"><code class="sourceCode python">StreamReader.readexactly()</code></span></a> has been improved. (Contributed by Mark Korenberg in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28370" class="reference external">bpo-28370</a>.)

- The <a href="../library/asyncio-eventloop.html#asyncio.loop.getaddrinfo" class="reference internal" title="asyncio.loop.getaddrinfo"><span class="pre"><code class="sourceCode python">loop.getaddrinfo()</code></span></a> method is optimized to avoid calling the system <span class="pre">`getaddrinfo`</span> function if the address is already resolved. (Contributed by A. Jesse Jiryu Davis.)

- The <a href="../library/asyncio-eventloop.html#asyncio.loop.stop" class="reference internal" title="asyncio.loop.stop"><span class="pre"><code class="sourceCode python">loop.stop()</code></span></a> method has been changed to stop the loop immediately after the current iteration. Any new callbacks scheduled as a result of the last iteration will be discarded. (Contributed by Guido van Rossum in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25593" class="reference external">bpo-25593</a>.)

- <a href="../library/asyncio-future.html#asyncio.Future.set_exception" class="reference internal" title="asyncio.Future.set_exception"><span class="pre"><code class="sourceCode python">Future.set_exception</code></span></a> will now raise <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> when passed an instance of the <a href="../library/exceptions.html#StopIteration" class="reference internal" title="StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a> exception. (Contributed by Chris Angelico in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26221" class="reference external">bpo-26221</a>.)

- New <a href="../library/asyncio-eventloop.html#asyncio.loop.connect_accepted_socket" class="reference internal" title="asyncio.loop.connect_accepted_socket"><span class="pre"><code class="sourceCode python">loop.connect_accepted_socket()</code></span></a> method to be used by servers that accept connections outside of asyncio, but that use asyncio to handle them. (Contributed by Jim Fulton in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27392" class="reference external">bpo-27392</a>.)

- <span class="pre">`TCP_NODELAY`</span> flag is now set for all TCP transports by default. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27456" class="reference external">bpo-27456</a>.)

- New <a href="../library/asyncio-eventloop.html#asyncio.loop.shutdown_asyncgens" class="reference internal" title="asyncio.loop.shutdown_asyncgens"><span class="pre"><code class="sourceCode python">loop.shutdown_asyncgens()</code></span></a> to properly close pending asynchronous generators before closing the loop. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28003" class="reference external">bpo-28003</a>.)

- <a href="../library/asyncio-future.html#asyncio.Future" class="reference internal" title="asyncio.Future"><span class="pre"><code class="sourceCode python">Future</code></span></a> and <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">Task</code></span></a> classes now have an optimized C implementation which makes asyncio code up to 30% faster. (Contributed by Yury Selivanov and INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26081" class="reference external">bpo-26081</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28544" class="reference external">bpo-28544</a>.)

</div>

<div id="binascii" class="section">

### binascii<a href="#binascii" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/binascii.html#binascii.b2a_base64" class="reference internal" title="binascii.b2a_base64"><span class="pre"><code class="sourceCode python">b2a_base64()</code></span></a> function now accepts an optional *newline* keyword argument to control whether the newline character is appended to the return value. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25357" class="reference external">bpo-25357</a>.)

</div>

<div id="cmath" class="section">

### cmath<a href="#cmath" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/cmath.html#cmath.tau" class="reference internal" title="cmath.tau"><span class="pre"><code class="sourceCode python">cmath.tau</code></span></a> (*τ*) constant has been added. (Contributed by Lisa Roach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12345" class="reference external">bpo-12345</a>, see <span id="index-29" class="target"></span><a href="https://peps.python.org/pep-0628/" class="pep reference external"><strong>PEP 628</strong></a> for details.)

New constants: <a href="../library/cmath.html#cmath.inf" class="reference internal" title="cmath.inf"><span class="pre"><code class="sourceCode python">cmath.inf</code></span></a> and <a href="../library/cmath.html#cmath.nan" class="reference internal" title="cmath.nan"><span class="pre"><code class="sourceCode python">cmath.nan</code></span></a> to match <a href="../library/math.html#math.inf" class="reference internal" title="math.inf"><span class="pre"><code class="sourceCode python">math.inf</code></span></a> and <a href="../library/math.html#math.nan" class="reference internal" title="math.nan"><span class="pre"><code class="sourceCode python">math.nan</code></span></a>, and also <a href="../library/cmath.html#cmath.infj" class="reference internal" title="cmath.infj"><span class="pre"><code class="sourceCode python">cmath.infj</code></span></a> and <a href="../library/cmath.html#cmath.nanj" class="reference internal" title="cmath.nanj"><span class="pre"><code class="sourceCode python">cmath.nanj</code></span></a> to match the format used by complex repr. (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23229" class="reference external">bpo-23229</a>.)

</div>

<div id="collections" class="section">

### collections<a href="#collections" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/collections.abc.html#collections.abc.Collection" class="reference internal" title="collections.abc.Collection"><span class="pre"><code class="sourceCode python">Collection</code></span></a> abstract base class has been added to represent sized iterable container classes. (Contributed by Ivan Levkivskyi, docs by Neil Girdhar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27598" class="reference external">bpo-27598</a>.)

The new <a href="../library/collections.abc.html#collections.abc.Reversible" class="reference internal" title="collections.abc.Reversible"><span class="pre"><code class="sourceCode python">Reversible</code></span></a> abstract base class represents iterable classes that also provide the <a href="../reference/datamodel.html#object.__reversed__" class="reference internal" title="object.__reversed__"><span class="pre"><code class="sourceCode python"><span class="fu">__reversed__</span>()</code></span></a> method. (Contributed by Ivan Levkivskyi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25987" class="reference external">bpo-25987</a>.)

The new <a href="../library/collections.abc.html#collections.abc.AsyncGenerator" class="reference internal" title="collections.abc.AsyncGenerator"><span class="pre"><code class="sourceCode python">AsyncGenerator</code></span></a> abstract base class represents asynchronous generators. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28720" class="reference external">bpo-28720</a>.)

The <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">namedtuple()</code></span></a> function now accepts an optional keyword argument *module*, which, when specified, is used for the <a href="../reference/datamodel.html#type.__module__" class="reference internal" title="type.__module__"><span class="pre"><code class="sourceCode python">__module__</code></span></a> attribute of the returned named tuple class. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17941" class="reference external">bpo-17941</a>.)

The *verbose* and *rename* arguments for <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">namedtuple()</code></span></a> are now keyword-only. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25628" class="reference external">bpo-25628</a>.)

Recursive <a href="../library/collections.html#collections.deque" class="reference internal" title="collections.deque"><span class="pre"><code class="sourceCode python">collections.deque</code></span></a> instances can now be pickled. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26482" class="reference external">bpo-26482</a>.)

</div>

<div id="concurrent-futures" class="section">

### concurrent.futures<a href="#concurrent-futures" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">ThreadPoolExecutor</code></span></a> class constructor now accepts an optional *thread_name_prefix* argument to make it possible to customize the names of the threads created by the pool. (Contributed by Gregory P. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27664" class="reference external">bpo-27664</a>.)

</div>

<div id="contextlib" class="section">

### contextlib<a href="#contextlib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/contextlib.html#contextlib.AbstractContextManager" class="reference internal" title="contextlib.AbstractContextManager"><span class="pre"><code class="sourceCode python">contextlib.AbstractContextManager</code></span></a> class has been added to provide an abstract base class for context managers. It provides a sensible default implementation for <span class="pre">`__enter__()`</span> which returns <span class="pre">`self`</span> and leaves <span class="pre">`__exit__()`</span> an abstract method. A matching class has been added to the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module as <a href="../library/typing.html#typing.ContextManager" class="reference internal" title="typing.ContextManager"><span class="pre"><code class="sourceCode python">typing.ContextManager</code></span></a>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25609" class="reference external">bpo-25609</a>.)

</div>

<div id="datetime" class="section">

### datetime<a href="#datetime" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> and <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">time</code></span></a> classes have the new <a href="../library/datetime.html#datetime.time.fold" class="reference internal" title="datetime.time.fold"><span class="pre"><code class="sourceCode python">fold</code></span></a> attribute used to disambiguate local time when necessary. Many functions in the <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> have been updated to support local time disambiguation. See <a href="#whatsnew36-pep495" class="reference internal"><span class="std std-ref">Local Time Disambiguation</span></a> section for more information. (Contributed by Alexander Belopolsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24773" class="reference external">bpo-24773</a>.)

The <a href="../library/datetime.html#datetime.datetime.strftime" class="reference internal" title="datetime.datetime.strftime"><span class="pre"><code class="sourceCode python">datetime.strftime()</code></span></a> and <a href="../library/datetime.html#datetime.date.strftime" class="reference internal" title="datetime.date.strftime"><span class="pre"><code class="sourceCode python">date.strftime()</code></span></a> methods now support ISO 8601 date directives <span class="pre">`%G`</span>, <span class="pre">`%u`</span> and <span class="pre">`%V`</span>. (Contributed by Ashley Anderson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12006" class="reference external">bpo-12006</a>.)

The <a href="../library/datetime.html#datetime.datetime.isoformat" class="reference internal" title="datetime.datetime.isoformat"><span class="pre"><code class="sourceCode python">datetime.isoformat()</code></span></a> function now accepts an optional *timespec* argument that specifies the number of additional components of the time value to include. (Contributed by Alessandro Cucci and Alexander Belopolsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19475" class="reference external">bpo-19475</a>.)

The <a href="../library/datetime.html#datetime.datetime.combine" class="reference internal" title="datetime.datetime.combine"><span class="pre"><code class="sourceCode python">datetime.combine()</code></span></a> now accepts an optional *tzinfo* argument. (Contributed by Alexander Belopolsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27661" class="reference external">bpo-27661</a>.)

</div>

<div id="decimal" class="section">

### decimal<a href="#decimal" class="headerlink" title="Link to this heading">¶</a>

New <a href="../library/decimal.html#decimal.Decimal.as_integer_ratio" class="reference internal" title="decimal.Decimal.as_integer_ratio"><span class="pre"><code class="sourceCode python">Decimal.as_integer_ratio()</code></span></a> method that returns a pair <span class="pre">`(n,`</span>` `<span class="pre">`d)`</span> of integers that represent the given <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> instance as a fraction, in lowest terms and with a positive denominator:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> Decimal('-3.14').as_integer_ratio()
    (-157, 50)

</div>

</div>

(Contributed by Stefan Krah amd Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25928" class="reference external">bpo-25928</a>.)

</div>

<div id="distutils" class="section">

### distutils<a href="#distutils" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`default_format`</span> attribute has been removed from <span class="pre">`distutils.command.sdist.sdist`</span> and the <span class="pre">`formats`</span> attribute defaults to <span class="pre">`['gztar']`</span>. Although not anticipated, any code relying on the presence of <span class="pre">`default_format`</span> may need to be adapted. See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27819" class="reference external">bpo-27819</a> for more details.

</div>

<div id="email" class="section">

### email<a href="#email" class="headerlink" title="Link to this heading">¶</a>

The new email API, enabled via the *policy* keyword to various constructors, is no longer provisional. The <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages."><span class="pre"><code class="sourceCode python">email</code></span></a> documentation has been reorganized and rewritten to focus on the new API, while retaining the old documentation for the legacy API. (Contributed by R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24277" class="reference external">bpo-24277</a>.)

The <a href="../library/email.mime.html#module-email.mime" class="reference internal" title="email.mime: Build MIME messages."><span class="pre"><code class="sourceCode python">email.mime</code></span></a> classes now all accept an optional *policy* keyword. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27331" class="reference external">bpo-27331</a>.)

The <a href="../library/email.generator.html#email.generator.DecodedGenerator" class="reference internal" title="email.generator.DecodedGenerator"><span class="pre"><code class="sourceCode python">DecodedGenerator</code></span></a> now supports the *policy* keyword.

There is a new <a href="../library/email.policy.html#module-email.policy" class="reference internal" title="email.policy: Controlling the parsing and generating of messages"><span class="pre"><code class="sourceCode python">policy</code></span></a> attribute, <a href="../library/email.policy.html#email.policy.Policy.message_factory" class="reference internal" title="email.policy.Policy.message_factory"><span class="pre"><code class="sourceCode python">message_factory</code></span></a>, that controls what class is used by default when the parser creates new message objects. For the <a href="../library/email.policy.html#email.policy.compat32" class="reference internal" title="email.policy.compat32"><span class="pre"><code class="sourceCode python">email.policy.compat32</code></span></a> policy this is <a href="../library/email.compat32-message.html#email.message.Message" class="reference internal" title="email.message.Message"><span class="pre"><code class="sourceCode python">Message</code></span></a>, for the new policies it is <a href="../library/email.message.html#email.message.EmailMessage" class="reference internal" title="email.message.EmailMessage"><span class="pre"><code class="sourceCode python">EmailMessage</code></span></a>. (Contributed by R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20476" class="reference external">bpo-20476</a>.)

</div>

<div id="encodings" class="section">

### encodings<a href="#encodings" class="headerlink" title="Link to this heading">¶</a>

On Windows, added the <span class="pre">`'oem'`</span> encoding to use <span class="pre">`CP_OEMCP`</span>, and the <span class="pre">`'ansi'`</span> alias for the existing <span class="pre">`'mbcs'`</span> encoding, which uses the <span class="pre">`CP_ACP`</span> code page. (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27959" class="reference external">bpo-27959</a>.)

</div>

<div id="enum" class="section">

### enum<a href="#enum" class="headerlink" title="Link to this heading">¶</a>

Two new enumeration base classes have been added to the <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">enum</code></span></a> module: <a href="../library/enum.html#enum.Flag" class="reference internal" title="enum.Flag"><span class="pre"><code class="sourceCode python">Flag</code></span></a> and <a href="../library/enum.html#enum.IntFlag" class="reference internal" title="enum.IntFlag"><span class="pre"><code class="sourceCode python">IntFlag</code></span></a>. Both are used to define constants that can be combined using the bitwise operators. (Contributed by Ethan Furman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23591" class="reference external">bpo-23591</a>.)

Many standard library modules have been updated to use the <a href="../library/enum.html#enum.IntFlag" class="reference internal" title="enum.IntFlag"><span class="pre"><code class="sourceCode python">IntFlag</code></span></a> class for their constants.

The new <a href="../library/enum.html#enum.auto" class="reference internal" title="enum.auto"><span class="pre"><code class="sourceCode python">enum.auto</code></span></a> value can be used to assign values to enum members automatically:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from enum import Enum, auto
    >>> class Color(Enum):
    ...     red = auto()
    ...     blue = auto()
    ...     green = auto()
    ...
    >>> list(Color)
    [<Color.red: 1>, <Color.blue: 2>, <Color.green: 3>]

</div>

</div>

</div>

<div id="faulthandler" class="section">

### faulthandler<a href="#faulthandler" class="headerlink" title="Link to this heading">¶</a>

On Windows, the <a href="../library/faulthandler.html#module-faulthandler" class="reference internal" title="faulthandler: Dump the Python traceback."><span class="pre"><code class="sourceCode python">faulthandler</code></span></a> module now installs a handler for Windows exceptions: see <a href="../library/faulthandler.html#faulthandler.enable" class="reference internal" title="faulthandler.enable"><span class="pre"><code class="sourceCode python">faulthandler.enable()</code></span></a>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23848" class="reference external">bpo-23848</a>.)

</div>

<div id="fileinput" class="section">

### fileinput<a href="#fileinput" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/fileinput.html#fileinput.hook_encoded" class="reference internal" title="fileinput.hook_encoded"><span class="pre"><code class="sourceCode python">hook_encoded()</code></span></a> now supports the *errors* argument. (Contributed by Joseph Hackman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25788" class="reference external">bpo-25788</a>.)

</div>

<div id="hashlib" class="section">

### hashlib<a href="#hashlib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> supports OpenSSL 1.1.0. The minimum recommend version is 1.0.2. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26470" class="reference external">bpo-26470</a>.)

BLAKE2 hash functions were added to the module. <a href="../library/hashlib.html#hashlib.blake2b" class="reference internal" title="hashlib.blake2b"><span class="pre"><code class="sourceCode python">blake2b()</code></span></a> and <a href="../library/hashlib.html#hashlib.blake2s" class="reference internal" title="hashlib.blake2s"><span class="pre"><code class="sourceCode python">blake2s()</code></span></a> are always available and support the full feature set of BLAKE2. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26798" class="reference external">bpo-26798</a> based on code by Dmitry Chestnykh and Samuel Neves. Documentation written by Dmitry Chestnykh.)

The SHA-3 hash functions <a href="../library/hashlib.html#hashlib.sha3_224" class="reference internal" title="hashlib.sha3_224"><span class="pre"><code class="sourceCode python">sha3_224()</code></span></a>, <a href="../library/hashlib.html#hashlib.sha3_256" class="reference internal" title="hashlib.sha3_256"><span class="pre"><code class="sourceCode python">sha3_256()</code></span></a>, <a href="../library/hashlib.html#hashlib.sha3_384" class="reference internal" title="hashlib.sha3_384"><span class="pre"><code class="sourceCode python">sha3_384()</code></span></a>, <a href="../library/hashlib.html#hashlib.sha3_512" class="reference internal" title="hashlib.sha3_512"><span class="pre"><code class="sourceCode python">sha3_512()</code></span></a>, and SHAKE hash functions <a href="../library/hashlib.html#hashlib.shake_128" class="reference internal" title="hashlib.shake_128"><span class="pre"><code class="sourceCode python">shake_128()</code></span></a> and <a href="../library/hashlib.html#hashlib.shake_256" class="reference internal" title="hashlib.shake_256"><span class="pre"><code class="sourceCode python">shake_256()</code></span></a> were added. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16113" class="reference external">bpo-16113</a>. Keccak Code Package by Guido Bertoni, Joan Daemen, Michaël Peeters, Gilles Van Assche, and Ronny Van Keer.)

The password-based key derivation function <a href="../library/hashlib.html#hashlib.scrypt" class="reference internal" title="hashlib.scrypt"><span class="pre"><code class="sourceCode python">scrypt()</code></span></a> is now available with OpenSSL 1.1.0 and newer. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27928" class="reference external">bpo-27928</a>.)

</div>

<div id="http-client" class="section">

### http.client<a href="#http-client" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/http.client.html#http.client.HTTPConnection.request" class="reference internal" title="http.client.HTTPConnection.request"><span class="pre"><code class="sourceCode python">HTTPConnection.request()</code></span></a> and <a href="../library/http.client.html#http.client.HTTPConnection.endheaders" class="reference internal" title="http.client.HTTPConnection.endheaders"><span class="pre"><code class="sourceCode python">endheaders()</code></span></a> both now support chunked encoding request bodies. (Contributed by Demian Brecht and Rolf Krahl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12319" class="reference external">bpo-12319</a>.)

</div>

<div id="idlelib-and-idle" class="section">

### idlelib and IDLE<a href="#idlelib-and-idle" class="headerlink" title="Link to this heading">¶</a>

The idlelib package is being modernized and refactored to make IDLE look and work better and to make the code easier to understand, test, and improve. Part of making IDLE look better, especially on Linux and Mac, is using ttk widgets, mostly in the dialogs. As a result, IDLE no longer runs with tcl/tk 8.4. It now requires tcl/tk 8.5 or 8.6. We recommend running the latest release of either.

‘Modernizing’ includes renaming and consolidation of idlelib modules. The renaming of files with partial uppercase names is similar to the renaming of, for instance, Tkinter and TkFont to tkinter and tkinter.font in 3.0. As a result, imports of idlelib files that worked in 3.5 will usually not work in 3.6. At least a module name change will be needed (see idlelib/README.txt), sometimes more. (Name changes contributed by Al Swiegart and Terry Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24225" class="reference external">bpo-24225</a>. Most idlelib patches since have been and will be part of the process.)

In compensation, the eventual result with be that some idlelib classes will be easier to use, with better APIs and docstrings explaining them. Additional useful information will be added to idlelib when available.

New in 3.6.2:

Multiple fixes for autocompletion. (Contributed by Louie Lu in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15786" class="reference external">bpo-15786</a>.)

New in 3.6.3:

Module Browser (on the File menu, formerly called Class Browser), now displays nested functions and classes in addition to top-level functions and classes. (Contributed by Guilherme Polo, Cheryl Sabella, and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1612262" class="reference external">bpo-1612262</a>.)

The IDLE features formerly implemented as extensions have been reimplemented as normal features. Their settings have been moved from the Extensions tab to other dialog tabs. (Contributed by Charles Wohlganger and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27099" class="reference external">bpo-27099</a>.)

The Settings dialog (Options, Configure IDLE) has been partly rewritten to improve both appearance and function. (Contributed by Cheryl Sabella and Terry Jan Reedy in multiple issues.)

New in 3.6.4:

The font sample now includes a selection of non-Latin characters so that users can better see the effect of selecting a particular font. (Contributed by Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13802" class="reference external">bpo-13802</a>.) The sample can be edited to include other characters. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31860" class="reference external">bpo-31860</a>.)

New in 3.6.6:

Editor code context option revised. Box displays all context lines up to maxlines. Clicking on a context line jumps the editor to that line. Context colors for custom themes is added to Highlights tab of Settings dialog. (Contributed by Cheryl Sabella and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33642" class="reference external">bpo-33642</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33768" class="reference external">bpo-33768</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33679" class="reference external">bpo-33679</a>.)

On Windows, a new API call tells Windows that tk scales for DPI. On Windows 8.1+ or 10, with DPI compatibility properties of the Python binary unchanged, and a monitor resolution greater than 96 DPI, this should make text and lines sharper. It should otherwise have no effect. (Contributed by Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33656" class="reference external">bpo-33656</a>.)

New in 3.6.7:

Output over N lines (50 by default) is squeezed down to a button. N can be changed in the PyShell section of the General page of the Settings dialog. Fewer, but possibly extra long, lines can be squeezed by right clicking on the output. Squeezed output can be expanded in place by double-clicking the button or into the clipboard or a separate window by right-clicking the button. (Contributed by Tal Einat in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1529353" class="reference external">bpo-1529353</a>.)

</div>

<div id="importlib" class="section">

### importlib<a href="#importlib" class="headerlink" title="Link to this heading">¶</a>

Import now raises the new exception <a href="../library/exceptions.html#ModuleNotFoundError" class="reference internal" title="ModuleNotFoundError"><span class="pre"><code class="sourceCode python"><span class="pp">ModuleNotFoundError</span></code></span></a> (subclass of <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a>) when it cannot find a module. Code that current checks for <span class="pre">`ImportError`</span> (in try-except) will still work. (Contributed by Eric Snow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15767" class="reference external">bpo-15767</a>.)

<a href="../library/importlib.html#importlib.util.LazyLoader" class="reference internal" title="importlib.util.LazyLoader"><span class="pre"><code class="sourceCode python">importlib.util.LazyLoader</code></span></a> now calls <a href="../library/importlib.html#importlib.abc.Loader.create_module" class="reference internal" title="importlib.abc.Loader.create_module"><span class="pre"><code class="sourceCode python">create_module()</code></span></a> on the wrapped loader, removing the restriction that <a href="../library/importlib.html#importlib.machinery.BuiltinImporter" class="reference internal" title="importlib.machinery.BuiltinImporter"><span class="pre"><code class="sourceCode python">importlib.machinery.BuiltinImporter</code></span></a> and <a href="../library/importlib.html#importlib.machinery.ExtensionFileLoader" class="reference internal" title="importlib.machinery.ExtensionFileLoader"><span class="pre"><code class="sourceCode python">importlib.machinery.ExtensionFileLoader</code></span></a> couldn’t be used with <a href="../library/importlib.html#importlib.util.LazyLoader" class="reference internal" title="importlib.util.LazyLoader"><span class="pre"><code class="sourceCode python">importlib.util.LazyLoader</code></span></a>.

<a href="../library/importlib.html#importlib.util.cache_from_source" class="reference internal" title="importlib.util.cache_from_source"><span class="pre"><code class="sourceCode python">importlib.util.cache_from_source()</code></span></a>, <a href="../library/importlib.html#importlib.util.source_from_cache" class="reference internal" title="importlib.util.source_from_cache"><span class="pre"><code class="sourceCode python">importlib.util.source_from_cache()</code></span></a>, and <a href="../library/importlib.html#importlib.util.spec_from_file_location" class="reference internal" title="importlib.util.spec_from_file_location"><span class="pre"><code class="sourceCode python">importlib.util.spec_from_file_location()</code></span></a> now accept a <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like object</span></a>.

</div>

<div id="inspect" class="section">

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a> function now reports the implicit <span class="pre">`.0`</span> parameters generated by the compiler for comprehension and generator expression scopes as if they were positional-only parameters called <span class="pre">`implicit0`</span>. (Contributed by Jelle Zijlstra in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19611" class="reference external">bpo-19611</a>.)

To reduce code churn when upgrading from Python 2.7 and the legacy <span class="pre">`inspect.getargspec()`</span> API, the previously documented deprecation of <a href="../library/inspect.html#inspect.getfullargspec" class="reference internal" title="inspect.getfullargspec"><span class="pre"><code class="sourceCode python">inspect.getfullargspec()</code></span></a> has been reversed. While this function is convenient for single/source Python 2/3 code bases, the richer <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a> interface remains the recommended approach for new code. (Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27172" class="reference external">bpo-27172</a>)

</div>

<div id="json" class="section">

### json<a href="#json" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/json.html#json.load" class="reference internal" title="json.load"><span class="pre"><code class="sourceCode python">json.load()</code></span></a> and <a href="../library/json.html#json.loads" class="reference internal" title="json.loads"><span class="pre"><code class="sourceCode python">json.loads()</code></span></a> now support binary input. Encoded JSON should be represented using either UTF-8, UTF-16, or UTF-32. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17909" class="reference external">bpo-17909</a>.)

</div>

<div id="logging" class="section">

### logging<a href="#logging" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/logging.handlers.html#logging.handlers.WatchedFileHandler.reopenIfNeeded" class="reference internal" title="logging.handlers.WatchedFileHandler.reopenIfNeeded"><span class="pre"><code class="sourceCode python">WatchedFileHandler.reopenIfNeeded()</code></span></a> method has been added to add the ability to check if the log file needs to be reopened. (Contributed by Marian Horban in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24884" class="reference external">bpo-24884</a>.)

</div>

<div id="math" class="section">

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

The tau (*τ*) constant has been added to the <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> and <a href="../library/cmath.html#module-cmath" class="reference internal" title="cmath: Mathematical functions for complex numbers."><span class="pre"><code class="sourceCode python">cmath</code></span></a> modules. (Contributed by Lisa Roach in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12345" class="reference external">bpo-12345</a>, see <span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0628/" class="pep reference external"><strong>PEP 628</strong></a> for details.)

</div>

<div id="multiprocessing" class="section">

### multiprocessing<a href="#multiprocessing" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/multiprocessing.html#multiprocessing-proxy-objects" class="reference internal"><span class="std std-ref">Proxy Objects</span></a> returned by <a href="../library/multiprocessing.html#multiprocessing.Manager" class="reference internal" title="multiprocessing.Manager"><span class="pre"><code class="sourceCode python">multiprocessing.Manager()</code></span></a> can now be nested. (Contributed by Davin Potts in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6766" class="reference external">bpo-6766</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

See the summary of <a href="#whatsnew36-pep519" class="reference internal"><span class="std std-ref">PEP 519</span></a> for details on how the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> and <a href="../library/os.path.html#module-os.path" class="reference internal" title="os.path: Operations on pathnames."><span class="pre"><code class="sourceCode python">os.path</code></span></a> modules now support <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like objects</span></a>.

<a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">scandir()</code></span></a> now supports <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> paths on Windows.

A new <a href="../library/os.html#os.scandir.close" class="reference internal" title="os.scandir.close"><span class="pre"><code class="sourceCode python">close()</code></span></a> method allows explicitly closing a <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">scandir()</code></span></a> iterator. The <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">scandir()</code></span></a> iterator now supports the <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> protocol. If a <span class="pre">`scandir()`</span> iterator is neither exhausted nor explicitly closed a <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a> will be emitted in its destructor. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25994" class="reference external">bpo-25994</a>.)

On Linux, <a href="../library/os.html#os.urandom" class="reference internal" title="os.urandom"><span class="pre"><code class="sourceCode python">os.urandom()</code></span></a> now blocks until the system urandom entropy pool is initialized to increase the security. See the <span id="index-31" class="target"></span><a href="https://peps.python.org/pep-0524/" class="pep reference external"><strong>PEP 524</strong></a> for the rationale.

The Linux <span class="pre">`getrandom()`</span> syscall (get random bytes) is now exposed as the new <a href="../library/os.html#os.getrandom" class="reference internal" title="os.getrandom"><span class="pre"><code class="sourceCode python">os.getrandom()</code></span></a> function. (Contributed by Victor Stinner, part of the <span id="index-32" class="target"></span><a href="https://peps.python.org/pep-0524/" class="pep reference external"><strong>PEP 524</strong></a>)

</div>

<div id="pathlib" class="section">

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a> now supports <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like objects</span></a>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27186" class="reference external">bpo-27186</a>.)

See the summary of <a href="#whatsnew36-pep519" class="reference internal"><span class="std std-ref">PEP 519</span></a> for details.

</div>

<div id="pdb" class="section">

### pdb<a href="#pdb" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/pdb.html#pdb.Pdb" class="reference internal" title="pdb.Pdb"><span class="pre"><code class="sourceCode python">Pdb</code></span></a> class constructor has a new optional *readrc* argument to control whether <span class="pre">`.pdbrc`</span> files should be read.

</div>

<div id="pickle" class="section">

### pickle<a href="#pickle" class="headerlink" title="Link to this heading">¶</a>

Objects that need <span class="pre">`__new__`</span> called with keyword arguments can now be pickled using <a href="../library/pickle.html#pickle-protocols" class="reference internal"><span class="std std-ref">pickle protocols</span></a> older than protocol version 4. Protocol version 4 already supports this case. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24164" class="reference external">bpo-24164</a>.)

</div>

<div id="pickletools" class="section">

### pickletools<a href="#pickletools" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pickletools.html#pickletools.dis" class="reference internal" title="pickletools.dis"><span class="pre"><code class="sourceCode python">pickletools.dis()</code></span></a> now outputs the implicit memo index for the <span class="pre">`MEMOIZE`</span> opcode. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25382" class="reference external">bpo-25382</a>.)

</div>

<div id="pydoc" class="section">

### pydoc<a href="#pydoc" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module has learned to respect the <span class="pre">`MANPAGER`</span> environment variable. (Contributed by Matthias Klose in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8637" class="reference external">bpo-8637</a>.)

<a href="../library/functions.html#help" class="reference internal" title="help"><span class="pre"><code class="sourceCode python"><span class="bu">help</span>()</code></span></a> and <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> can now list named tuple fields in the order they were defined rather than alphabetically. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24879" class="reference external">bpo-24879</a>.)

</div>

<div id="random" class="section">

### random<a href="#random" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/random.html#random.choices" class="reference internal" title="random.choices"><span class="pre"><code class="sourceCode python">choices()</code></span></a> function returns a list of elements of specified size from the given population with optional weights. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18844" class="reference external">bpo-18844</a>.)

</div>

<div id="re" class="section">

### re<a href="#re" class="headerlink" title="Link to this heading">¶</a>

Added support of modifier spans in regular expressions. Examples: <span class="pre">`'(?i:p)ython'`</span> matches <span class="pre">`'python'`</span> and <span class="pre">`'Python'`</span>, but not <span class="pre">`'PYTHON'`</span>; <span class="pre">`'(?i)g(?-i:v)r'`</span> matches <span class="pre">`'GvR'`</span> and <span class="pre">`'gvr'`</span>, but not <span class="pre">`'GVR'`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=433028" class="reference external">bpo-433028</a>.)

Match object groups can be accessed by <span class="pre">`__getitem__`</span>, which is equivalent to <span class="pre">`group()`</span>. So <span class="pre">`mo['name']`</span> is now equivalent to <span class="pre">`mo.group('name')`</span>. (Contributed by Eric Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24454" class="reference external">bpo-24454</a>.)

<a href="../library/re.html#re.Match" class="reference internal" title="re.Match"><span class="pre"><code class="sourceCode python">Match</code></span></a> objects now support <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python">index<span class="op">-</span>like</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">objects</code></span></a> as group indices. (Contributed by Jeroen Demeyer and Xiang Zhang in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27177" class="reference external">bpo-27177</a>.)

</div>

<div id="readline" class="section">

### readline<a href="#readline" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/readline.html#readline.set_auto_history" class="reference internal" title="readline.set_auto_history"><span class="pre"><code class="sourceCode python">set_auto_history()</code></span></a> to enable or disable automatic addition of input to the history list. (Contributed by Tyler Crompton in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26870" class="reference external">bpo-26870</a>.)

</div>

<div id="rlcompleter" class="section">

### rlcompleter<a href="#rlcompleter" class="headerlink" title="Link to this heading">¶</a>

Private and special attribute names now are omitted unless the prefix starts with underscores. A space or a colon is added after some completed keywords. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25011" class="reference external">bpo-25011</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25209" class="reference external">bpo-25209</a>.)

</div>

<div id="shlex" class="section">

### shlex<a href="#shlex" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/shlex.html#shlex.shlex" class="reference internal" title="shlex.shlex"><span class="pre"><code class="sourceCode python">shlex</code></span></a> has much <a href="../library/shlex.html#improved-shell-compatibility" class="reference internal"><span class="std std-ref">improved shell compatibility</span></a> through the new *punctuation_chars* argument to control which characters are treated as punctuation. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1521950" class="reference external">bpo-1521950</a>.)

</div>

<div id="site" class="section">

### site<a href="#site" class="headerlink" title="Link to this heading">¶</a>

When specifying paths to add to <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> in a <span class="pre">`.pth`</span> file, you may now specify file paths on top of directories (e.g. zip files). (Contributed by Wolfgang Langner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26587" class="reference external">bpo-26587</a>).

</div>

<div id="sqlite3" class="section">

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/sqlite3.html#sqlite3.Cursor.lastrowid" class="reference internal" title="sqlite3.Cursor.lastrowid"><span class="pre"><code class="sourceCode python">sqlite3.Cursor.lastrowid</code></span></a> now supports the <span class="pre">`REPLACE`</span> statement. (Contributed by Alex LordThorsen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16864" class="reference external">bpo-16864</a>.)

</div>

<div id="socket" class="section">

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/socket.html#socket.socket.ioctl" class="reference internal" title="socket.socket.ioctl"><span class="pre"><code class="sourceCode python">ioctl()</code></span></a> function now supports the <a href="../library/socket.html#socket.SIO_LOOPBACK_FAST_PATH" class="reference internal" title="socket.SIO_LOOPBACK_FAST_PATH"><span class="pre"><code class="sourceCode python">SIO_LOOPBACK_FAST_PATH</code></span></a> control code. (Contributed by Daniel Stokes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26536" class="reference external">bpo-26536</a>.)

The <a href="../library/socket.html#socket.socket.getsockopt" class="reference internal" title="socket.socket.getsockopt"><span class="pre"><code class="sourceCode python">getsockopt()</code></span></a> constants <span class="pre">`SO_DOMAIN`</span>, <span class="pre">`SO_PROTOCOL`</span>, <span class="pre">`SO_PEERSEC`</span>, and <span class="pre">`SO_PASSSEC`</span> are now supported. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26907" class="reference external">bpo-26907</a>.)

The <a href="../library/socket.html#socket.socket.setsockopt" class="reference internal" title="socket.socket.setsockopt"><span class="pre"><code class="sourceCode python">setsockopt()</code></span></a> now supports the <span class="pre">`setsockopt(level,`</span>` `<span class="pre">`optname,`</span>` `<span class="pre">`None,`</span>` `<span class="pre">`optlen:`</span>` `<span class="pre">`int)`</span> form. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27744" class="reference external">bpo-27744</a>.)

The socket module now supports the address family <a href="../library/socket.html#socket.AF_ALG" class="reference internal" title="socket.AF_ALG"><span class="pre"><code class="sourceCode python">AF_ALG</code></span></a> to interface with Linux Kernel crypto API. <span class="pre">`ALG_*`</span>, <span class="pre">`SOL_ALG`</span> and <a href="../library/socket.html#socket.socket.sendmsg_afalg" class="reference internal" title="socket.socket.sendmsg_afalg"><span class="pre"><code class="sourceCode python">sendmsg_afalg()</code></span></a> were added. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27744" class="reference external">bpo-27744</a> with support from Victor Stinner.)

New Linux constants <span class="pre">`TCP_USER_TIMEOUT`</span> and <span class="pre">`TCP_CONGESTION`</span> were added. (Contributed by Omar Sandoval, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26273" class="reference external">bpo-26273</a>).

</div>

<div id="socketserver" class="section">

### socketserver<a href="#socketserver" class="headerlink" title="Link to this heading">¶</a>

Servers based on the <a href="../library/socketserver.html#module-socketserver" class="reference internal" title="socketserver: A framework for network servers."><span class="pre"><code class="sourceCode python">socketserver</code></span></a> module, including those defined in <a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a>, <a href="../library/xmlrpc.server.html#module-xmlrpc.server" class="reference internal" title="xmlrpc.server: Basic XML-RPC server implementations."><span class="pre"><code class="sourceCode python">xmlrpc.server</code></span></a> and <a href="../library/wsgiref.html#module-wsgiref.simple_server" class="reference internal" title="wsgiref.simple_server: A simple WSGI HTTP server."><span class="pre"><code class="sourceCode python">wsgiref.simple_server</code></span></a>, now support the <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> protocol. (Contributed by Aviv Palivoda in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26404" class="reference external">bpo-26404</a>.)

The <a href="../library/socketserver.html#socketserver.DatagramRequestHandler.wfile" class="reference internal" title="socketserver.DatagramRequestHandler.wfile"><span class="pre"><code class="sourceCode python">wfile</code></span></a> attribute of <a href="../library/socketserver.html#socketserver.StreamRequestHandler" class="reference internal" title="socketserver.StreamRequestHandler"><span class="pre"><code class="sourceCode python">StreamRequestHandler</code></span></a> classes now implements the <a href="../library/io.html#io.BufferedIOBase" class="reference internal" title="io.BufferedIOBase"><span class="pre"><code class="sourceCode python">io.BufferedIOBase</code></span></a> writable interface. In particular, calling <a href="../library/io.html#io.BufferedIOBase.write" class="reference internal" title="io.BufferedIOBase.write"><span class="pre"><code class="sourceCode python">write()</code></span></a> is now guaranteed to send the data in full. (Contributed by Martin Panter in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26721" class="reference external">bpo-26721</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> supports OpenSSL 1.1.0. The minimum recommend version is 1.0.2. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26470" class="reference external">bpo-26470</a>.)

3DES has been removed from the default cipher suites and ChaCha20 Poly1305 cipher suites have been added. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27850" class="reference external">bpo-27850</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27766" class="reference external">bpo-27766</a>.)

<a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> has better default configuration for options and ciphers. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28043" class="reference external">bpo-28043</a>.)

SSL session can be copied from one client-side connection to another with the new <a href="../library/ssl.html#ssl.SSLSession" class="reference internal" title="ssl.SSLSession"><span class="pre"><code class="sourceCode python">SSLSession</code></span></a> class. TLS session resumption can speed up the initial handshake, reduce latency and improve performance (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19500" class="reference external">bpo-19500</a> based on a draft by Alex Warhawk.)

The new <a href="../library/ssl.html#ssl.SSLContext.get_ciphers" class="reference internal" title="ssl.SSLContext.get_ciphers"><span class="pre"><code class="sourceCode python">get_ciphers()</code></span></a> method can be used to get a list of enabled ciphers in order of cipher priority.

All constants and flags have been converted to <a href="../library/enum.html#enum.IntEnum" class="reference internal" title="enum.IntEnum"><span class="pre"><code class="sourceCode python">IntEnum</code></span></a> and <a href="../library/enum.html#enum.IntFlag" class="reference internal" title="enum.IntFlag"><span class="pre"><code class="sourceCode python">IntFlag</code></span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28025" class="reference external">bpo-28025</a>.)

Server and client-side specific TLS protocols for <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> were added. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28085" class="reference external">bpo-28085</a>.)

Added <a href="../library/ssl.html#ssl.SSLContext.post_handshake_auth" class="reference internal" title="ssl.SSLContext.post_handshake_auth"><span class="pre"><code class="sourceCode python">ssl.SSLContext.post_handshake_auth</code></span></a> to enable and <a href="../library/ssl.html#ssl.SSLSocket.verify_client_post_handshake" class="reference internal" title="ssl.SSLSocket.verify_client_post_handshake"><span class="pre"><code class="sourceCode python">ssl.SSLSocket.verify_client_post_handshake()</code></span></a> to initiate TLS 1.3 post-handshake authentication. (Contributed by Christian Heimes in <a href="https://github.com/python/cpython/issues/78851" class="reference external">gh-78851</a>.)

</div>

<div id="statistics" class="section">

### statistics<a href="#statistics" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/statistics.html#statistics.harmonic_mean" class="reference internal" title="statistics.harmonic_mean"><span class="pre"><code class="sourceCode python">harmonic_mean()</code></span></a> function has been added. (Contributed by Steven D’Aprano in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27181" class="reference external">bpo-27181</a>.)

</div>

<div id="struct" class="section">

### struct<a href="#struct" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/struct.html#module-struct" class="reference internal" title="struct: Interpret bytes as packed binary data."><span class="pre"><code class="sourceCode python">struct</code></span></a> now supports IEEE 754 half-precision floats via the <span class="pre">`'e'`</span> format specifier. (Contributed by Eli Stevens, Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11734" class="reference external">bpo-11734</a>.)

</div>

<div id="subprocess" class="section">

### subprocess<a href="#subprocess" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a> destructor now emits a <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a> warning if the child process is still running. Use the context manager protocol (<span class="pre">`with`</span>` `<span class="pre">`proc:`</span>` `<span class="pre">`...`</span>) or explicitly call the <a href="../library/subprocess.html#subprocess.Popen.wait" class="reference internal" title="subprocess.Popen.wait"><span class="pre"><code class="sourceCode python">wait()</code></span></a> method to read the exit status of the child process. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26741" class="reference external">bpo-26741</a>.)

The <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a> constructor and all functions that pass arguments through to it now accept *encoding* and *errors* arguments. Specifying either of these will enable text mode for the *stdin*, *stdout* and *stderr* streams. (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6135" class="reference external">bpo-6135</a>.)

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/sys.html#sys.getfilesystemencodeerrors" class="reference internal" title="sys.getfilesystemencodeerrors"><span class="pre"><code class="sourceCode python">getfilesystemencodeerrors()</code></span></a> function returns the name of the error mode used to convert between Unicode filenames and bytes filenames. (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27781" class="reference external">bpo-27781</a>.)

On Windows the return value of the <a href="../library/sys.html#sys.getwindowsversion" class="reference internal" title="sys.getwindowsversion"><span class="pre"><code class="sourceCode python">getwindowsversion()</code></span></a> function now includes the *platform_version* field which contains the accurate major version, minor version and build number of the current operating system, rather than the version that is being emulated for the process (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27932" class="reference external">bpo-27932</a>.)

</div>

<div id="telnetlib" class="section">

### telnetlib<a href="#telnetlib" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`telnetlib.Telnet`</span> is now a context manager (contributed by Stéphane Wirtel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25485" class="reference external">bpo-25485</a>).

</div>

<div id="time" class="section">

### time<a href="#time" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/time.html#time.struct_time" class="reference internal" title="time.struct_time"><span class="pre"><code class="sourceCode python">struct_time</code></span></a> attributes <span class="pre">`tm_gmtoff`</span> and <span class="pre">`tm_zone`</span> are now available on all platforms.

</div>

<div id="timeit" class="section">

### timeit<a href="#timeit" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/timeit.html#timeit.Timer.autorange" class="reference internal" title="timeit.Timer.autorange"><span class="pre"><code class="sourceCode python">Timer.autorange()</code></span></a> convenience method has been added to call <a href="../library/timeit.html#timeit.Timer.timeit" class="reference internal" title="timeit.Timer.timeit"><span class="pre"><code class="sourceCode python">Timer.timeit()</code></span></a> repeatedly so that the total run time is greater or equal to 200 milliseconds. (Contributed by Steven D’Aprano in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6422" class="reference external">bpo-6422</a>.)

<a href="../library/timeit.html#module-timeit" class="reference internal" title="timeit: Measure the execution time of small code snippets."><span class="pre"><code class="sourceCode python">timeit</code></span></a> now warns when there is substantial (4x) variance between best and worst times. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23552" class="reference external">bpo-23552</a>.)

</div>

<div id="tkinter" class="section">

### tkinter<a href="#tkinter" class="headerlink" title="Link to this heading">¶</a>

Added methods <span class="pre">`Variable.trace_add()`</span>, <span class="pre">`Variable.trace_remove()`</span> and <span class="pre">`trace_info()`</span> in the <span class="pre">`tkinter.Variable`</span> class. They replace old methods <span class="pre">`trace_variable()`</span>, <span class="pre">`trace()`</span>, <span class="pre">`trace_vdelete()`</span> and <span class="pre">`trace_vinfo()`</span> that use obsolete Tcl commands and might not work in future versions of Tcl. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22115" class="reference external">bpo-22115</a>).

</div>

<div id="traceback" class="section">

<span id="whatsnew36-traceback"></span>

### traceback<a href="#traceback" class="headerlink" title="Link to this heading">¶</a>

Both the traceback module and the interpreter’s builtin exception display now abbreviate long sequences of repeated lines in tracebacks as shown in the following example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> def f(): f()
    ...
    >>> f()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 1, in f
      File "<stdin>", line 1, in f
      File "<stdin>", line 1, in f
      [Previous line repeated 995 more times]
    RecursionError: maximum recursion depth exceeded

</div>

</div>

(Contributed by Emanuel Barry in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26823" class="reference external">bpo-26823</a>.)

</div>

<div id="tracemalloc" class="section">

### tracemalloc<a href="#tracemalloc" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/tracemalloc.html#module-tracemalloc" class="reference internal" title="tracemalloc: Trace memory allocations."><span class="pre"><code class="sourceCode python">tracemalloc</code></span></a> module now supports tracing memory allocations in multiple different address spaces.

The new <a href="../library/tracemalloc.html#tracemalloc.DomainFilter" class="reference internal" title="tracemalloc.DomainFilter"><span class="pre"><code class="sourceCode python">DomainFilter</code></span></a> filter class has been added to filter block traces by their address space (domain).

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26588" class="reference external">bpo-26588</a>.)

</div>

<div id="typing" class="section">

<span id="whatsnew36-typing"></span>

### typing<a href="#typing" class="headerlink" title="Link to this heading">¶</a>

Since the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module is <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional</span></a>, all changes introduced in Python 3.6 have also been backported to Python 3.5.x.

The <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module has a much improved support for generic type aliases. For example <span class="pre">`Dict[str,`</span>` `<span class="pre">`Tuple[S,`</span>` `<span class="pre">`T]]`</span> is now a valid type annotation. (Contributed by Guido van Rossum in <a href="https://github.com/python/typing/pull/195" class="reference external">Github #195</a>.)

The <a href="../library/typing.html#typing.ContextManager" class="reference internal" title="typing.ContextManager"><span class="pre"><code class="sourceCode python">typing.ContextManager</code></span></a> class has been added for representing <a href="../library/contextlib.html#contextlib.AbstractContextManager" class="reference internal" title="contextlib.AbstractContextManager"><span class="pre"><code class="sourceCode python">contextlib.AbstractContextManager</code></span></a>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25609" class="reference external">bpo-25609</a>.)

The <a href="../library/typing.html#typing.Collection" class="reference internal" title="typing.Collection"><span class="pre"><code class="sourceCode python">typing.Collection</code></span></a> class has been added for representing <a href="../library/collections.abc.html#collections.abc.Collection" class="reference internal" title="collections.abc.Collection"><span class="pre"><code class="sourceCode python">collections.abc.Collection</code></span></a>. (Contributed by Ivan Levkivskyi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27598" class="reference external">bpo-27598</a>.)

The <a href="../library/typing.html#typing.ClassVar" class="reference internal" title="typing.ClassVar"><span class="pre"><code class="sourceCode python">typing.ClassVar</code></span></a> type construct has been added to mark class variables. As introduced in <span id="index-33" class="target"></span><a href="https://peps.python.org/pep-0526/" class="pep reference external"><strong>PEP 526</strong></a>, a variable annotation wrapped in ClassVar indicates that a given attribute is intended to be used as a class variable and should not be set on instances of that class. (Contributed by Ivan Levkivskyi in <a href="https://github.com/python/typing/pull/280" class="reference external">Github #280</a>.)

A new <a href="../library/typing.html#typing.TYPE_CHECKING" class="reference internal" title="typing.TYPE_CHECKING"><span class="pre"><code class="sourceCode python">TYPE_CHECKING</code></span></a> constant that is assumed to be <span class="pre">`True`</span> by the static type checkers, but is <span class="pre">`False`</span> at runtime. (Contributed by Guido van Rossum in <a href="https://github.com/python/typing/issues/230" class="reference external">Github #230</a>.)

A new <a href="../library/typing.html#typing.NewType" class="reference internal" title="typing.NewType"><span class="pre"><code class="sourceCode python">NewType()</code></span></a> helper function has been added to create lightweight distinct types for annotations:

<div class="highlight-python3 notranslate">

<div class="highlight">

    from typing import NewType

    UserId = NewType('UserId', int)
    some_id = UserId(524313)

</div>

</div>

The static type checker will treat the new type as if it were a subclass of the original type. (Contributed by Ivan Levkivskyi in <a href="https://github.com/python/typing/issues/189" class="reference external">Github #189</a>.)

</div>

<div id="unicodedata" class="section">

### unicodedata<a href="#unicodedata" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a> module now uses data from <a href="https://unicode.org/versions/Unicode9.0.0/" class="reference external">Unicode 9.0.0</a>. (Contributed by Benjamin Peterson.)

</div>

<div id="unittest-mock" class="section">

### unittest.mock<a href="#unittest-mock" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/unittest.mock.html#unittest.mock.Mock" class="reference internal" title="unittest.mock.Mock"><span class="pre"><code class="sourceCode python">Mock</code></span></a> class has the following improvements:

- Two new methods, <a href="../library/unittest.mock.html#unittest.mock.Mock.assert_called" class="reference internal" title="unittest.mock.Mock.assert_called"><span class="pre"><code class="sourceCode python">Mock.assert_called()</code></span></a> and <a href="../library/unittest.mock.html#unittest.mock.Mock.assert_called_once" class="reference internal" title="unittest.mock.Mock.assert_called_once"><span class="pre"><code class="sourceCode python">Mock.assert_called_once()</code></span></a> to check if the mock object was called. (Contributed by Amit Saha in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26323" class="reference external">bpo-26323</a>.)

- The <a href="../library/unittest.mock.html#unittest.mock.Mock.reset_mock" class="reference internal" title="unittest.mock.Mock.reset_mock"><span class="pre"><code class="sourceCode python">Mock.reset_mock()</code></span></a> method now has two optional keyword only arguments: *return_value* and *side_effect*. (Contributed by Kushal Das in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21271" class="reference external">bpo-21271</a>.)

</div>

<div id="urllib-request" class="section">

### urllib.request<a href="#urllib-request" class="headerlink" title="Link to this heading">¶</a>

If a HTTP request has a file or iterable body (other than a bytes object) but no <span class="pre">`Content-Length`</span> header, rather than throwing an error, <a href="../library/urllib.request.html#urllib.request.HTTPHandler" class="reference internal" title="urllib.request.HTTPHandler"><span class="pre"><code class="sourceCode python">AbstractHTTPHandler</code></span></a> now falls back to use chunked transfer encoding. (Contributed by Demian Brecht and Rolf Krahl in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12319" class="reference external">bpo-12319</a>.)

</div>

<div id="urllib-robotparser" class="section">

### urllib.robotparser<a href="#urllib-robotparser" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/urllib.robotparser.html#urllib.robotparser.RobotFileParser" class="reference internal" title="urllib.robotparser.RobotFileParser"><span class="pre"><code class="sourceCode python">RobotFileParser</code></span></a> now supports the <span class="pre">`Crawl-delay`</span> and <span class="pre">`Request-rate`</span> extensions. (Contributed by Nikolay Bogoychev in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16099" class="reference external">bpo-16099</a>.)

</div>

<div id="venv" class="section">

### venv<a href="#venv" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> accepts a new parameter <span class="pre">`--prompt`</span>. This parameter provides an alternative prefix for the virtual environment. (Proposed by Łukasz Balcerzak and ported to 3.6 by Stéphane Wirtel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22829" class="reference external">bpo-22829</a>.)

</div>

<div id="warnings" class="section">

### warnings<a href="#warnings" class="headerlink" title="Link to this heading">¶</a>

A new optional *source* parameter has been added to the <a href="../library/warnings.html#warnings.warn_explicit" class="reference internal" title="warnings.warn_explicit"><span class="pre"><code class="sourceCode python">warnings.warn_explicit()</code></span></a> function: the destroyed object which emitted a <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a>. A *source* attribute has also been added to <span class="pre">`warnings.WarningMessage`</span> (contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26568" class="reference external">bpo-26568</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26567" class="reference external">bpo-26567</a>).

When a <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a> warning is logged, the <a href="../library/tracemalloc.html#module-tracemalloc" class="reference internal" title="tracemalloc: Trace memory allocations."><span class="pre"><code class="sourceCode python">tracemalloc</code></span></a> module is now used to try to retrieve the traceback where the destroyed object was allocated.

Example with the script <span class="pre">`example.py`</span>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import warnings

    def func():
        return open(__file__)

    f = func()
    f = None

</div>

</div>

Output of the command <span class="pre">`python3.6`</span>` `<span class="pre">`-Wd`</span>` `<span class="pre">`-X`</span>` `<span class="pre">`tracemalloc=5`</span>` `<span class="pre">`example.py`</span>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    example.py:7: ResourceWarning: unclosed file <_io.TextIOWrapper name='example.py' mode='r' encoding='UTF-8'>
      f = None
    Object allocated at (most recent call first):
      File "example.py", lineno 4
        return open(__file__)
      File "example.py", lineno 6
        f = func()

</div>

</div>

The “Object allocated at” traceback is new and is only displayed if <a href="../library/tracemalloc.html#module-tracemalloc" class="reference internal" title="tracemalloc: Trace memory allocations."><span class="pre"><code class="sourceCode python">tracemalloc</code></span></a> is tracing Python memory allocations and if the <a href="../library/warnings.html#module-warnings" class="reference internal" title="warnings: Issue warning messages and control their disposition."><span class="pre"><code class="sourceCode python">warnings</code></span></a> module was already imported.

</div>

<div id="winreg" class="section">

### winreg<a href="#winreg" class="headerlink" title="Link to this heading">¶</a>

Added the 64-bit integer type <a href="../library/winreg.html#winreg.REG_QWORD" class="reference internal" title="winreg.REG_QWORD"><span class="pre"><code class="sourceCode python">REG_QWORD</code></span></a>. (Contributed by Clement Rouault in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23026" class="reference external">bpo-23026</a>.)

</div>

<div id="winsound" class="section">

### winsound<a href="#winsound" class="headerlink" title="Link to this heading">¶</a>

Allowed keyword arguments to be passed to <a href="../library/winsound.html#winsound.Beep" class="reference internal" title="winsound.Beep"><span class="pre"><code class="sourceCode python">Beep</code></span></a>, <a href="../library/winsound.html#winsound.MessageBeep" class="reference internal" title="winsound.MessageBeep"><span class="pre"><code class="sourceCode python">MessageBeep</code></span></a>, and <a href="../library/winsound.html#winsound.PlaySound" class="reference internal" title="winsound.PlaySound"><span class="pre"><code class="sourceCode python">PlaySound</code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27982" class="reference external">bpo-27982</a>).

</div>

<div id="xmlrpc-client" class="section">

### xmlrpc.client<a href="#xmlrpc-client" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/xmlrpc.client.html#module-xmlrpc.client" class="reference internal" title="xmlrpc.client: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpc.client</code></span></a> module now supports unmarshalling additional data types used by the Apache XML-RPC implementation for numerics and <span class="pre">`None`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26885" class="reference external">bpo-26885</a>.)

</div>

<div id="zipfile" class="section">

### zipfile<a href="#zipfile" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/zipfile.html#zipfile.ZipInfo.from_file" class="reference internal" title="zipfile.ZipInfo.from_file"><span class="pre"><code class="sourceCode python">ZipInfo.from_file()</code></span></a> class method allows making a <a href="../library/zipfile.html#zipfile.ZipInfo" class="reference internal" title="zipfile.ZipInfo"><span class="pre"><code class="sourceCode python">ZipInfo</code></span></a> instance from a filesystem file. A new <a href="../library/zipfile.html#zipfile.ZipInfo.is_dir" class="reference internal" title="zipfile.ZipInfo.is_dir"><span class="pre"><code class="sourceCode python">ZipInfo.is_dir()</code></span></a> method can be used to check if the <a href="../library/zipfile.html#zipfile.ZipInfo" class="reference internal" title="zipfile.ZipInfo"><span class="pre"><code class="sourceCode python">ZipInfo</code></span></a> instance represents a directory. (Contributed by Thomas Kluyver in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26039" class="reference external">bpo-26039</a>.)

The <a href="../library/zipfile.html#zipfile.ZipFile.open" class="reference internal" title="zipfile.ZipFile.open"><span class="pre"><code class="sourceCode python">ZipFile.<span class="bu">open</span>()</code></span></a> method can now be used to write data into a ZIP file, as well as for extracting data. (Contributed by Thomas Kluyver in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26039" class="reference external">bpo-26039</a>.)

</div>

<div id="zlib" class="section">

### zlib<a href="#zlib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/zlib.html#zlib.compress" class="reference internal" title="zlib.compress"><span class="pre"><code class="sourceCode python">compress()</code></span></a> and <a href="../library/zlib.html#zlib.decompress" class="reference internal" title="zlib.decompress"><span class="pre"><code class="sourceCode python">decompress()</code></span></a> functions now accept keyword arguments. (Contributed by Aviv Palivoda in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26243" class="reference external">bpo-26243</a> and Xiang Zhang in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16764" class="reference external">bpo-16764</a> respectively.)

</div>

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

- The Python interpreter now uses a 16-bit wordcode instead of bytecode which made a number of opcode optimizations possible. (Contributed by Demur Rumed with input and reviews from Serhiy Storchaka and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26647" class="reference external">bpo-26647</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28050" class="reference external">bpo-28050</a>.)

- The <a href="../library/asyncio-future.html#asyncio.Future" class="reference internal" title="asyncio.Future"><span class="pre"><code class="sourceCode python">asyncio.Future</code></span></a> class now has an optimized C implementation. (Contributed by Yury Selivanov and INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26081" class="reference external">bpo-26081</a>.)

- The <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">asyncio.Task</code></span></a> class now has an optimized C implementation. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28544" class="reference external">bpo-28544</a>.)

- Various implementation improvements in the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module (such as caching of generic types) allow up to 30 times performance improvements and reduced memory footprint.

- The ASCII decoder is now up to 60 times as fast for error handlers <span class="pre">`surrogateescape`</span>, <span class="pre">`ignore`</span> and <span class="pre">`replace`</span> (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24870" class="reference external">bpo-24870</a>).

- The ASCII and the Latin1 encoders are now up to 3 times as fast for the error handler <span class="pre">`surrogateescape`</span> (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25227" class="reference external">bpo-25227</a>).

- The UTF-8 encoder is now up to 75 times as fast for error handlers <span class="pre">`ignore`</span>, <span class="pre">`replace`</span>, <span class="pre">`surrogateescape`</span>, <span class="pre">`surrogatepass`</span> (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25267" class="reference external">bpo-25267</a>).

- The UTF-8 decoder is now up to 15 times as fast for error handlers <span class="pre">`ignore`</span>, <span class="pre">`replace`</span> and <span class="pre">`surrogateescape`</span> (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25301" class="reference external">bpo-25301</a>).

- <span class="pre">`bytes`</span>` `<span class="pre">`%`</span>` `<span class="pre">`args`</span> is now up to 2 times faster. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25349" class="reference external">bpo-25349</a>).

- <span class="pre">`bytearray`</span>` `<span class="pre">`%`</span>` `<span class="pre">`args`</span> is now between 2.5 and 5 times faster. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25399" class="reference external">bpo-25399</a>).

- Optimize <a href="../library/stdtypes.html#bytes.fromhex" class="reference internal" title="bytes.fromhex"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span>.fromhex()</code></span></a> and <a href="../library/stdtypes.html#bytearray.fromhex" class="reference internal" title="bytearray.fromhex"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span>.fromhex()</code></span></a>: they are now between 2x and 3.5x faster. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25401" class="reference external">bpo-25401</a>).

- Optimize <span class="pre">`bytes.replace(b'',`</span>` `<span class="pre">`b'.')`</span> and <span class="pre">`bytearray.replace(b'',`</span>` `<span class="pre">`b'.')`</span>: up to 80% faster. (Contributed by Josh Snider in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26574" class="reference external">bpo-26574</a>).

- Allocator functions of the <a href="../c-api/memory.html#c.PyMem_Malloc" class="reference internal" title="PyMem_Malloc"><span class="pre"><code class="sourceCode c">PyMem_Malloc<span class="op">()</span></code></span></a> domain (<a href="../c-api/memory.html#c.PYMEM_DOMAIN_MEM" class="reference internal" title="PYMEM_DOMAIN_MEM"><span class="pre"><code class="sourceCode c">PYMEM_DOMAIN_MEM</code></span></a>) now use the <a href="../c-api/memory.html#pymalloc" class="reference internal"><span class="std std-ref">pymalloc memory allocator</span></a> instead of <span class="pre">`malloc()`</span> function of the C library. The pymalloc allocator is optimized for objects smaller or equal to 512 bytes with a short lifetime, and use <span class="pre">`malloc()`</span> for larger memory blocks. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26249" class="reference external">bpo-26249</a>).

- <a href="../library/pickle.html#pickle.load" class="reference internal" title="pickle.load"><span class="pre"><code class="sourceCode python">pickle.load()</code></span></a> and <a href="../library/pickle.html#pickle.loads" class="reference internal" title="pickle.loads"><span class="pre"><code class="sourceCode python">pickle.loads()</code></span></a> are now up to 10% faster when deserializing many small objects (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27056" class="reference external">bpo-27056</a>).

- Passing <a href="../glossary.html#term-keyword-argument" class="reference internal"><span class="xref std std-term">keyword arguments</span></a> to a function has an overhead in comparison with passing <a href="../glossary.html#term-positional-argument" class="reference internal"><span class="xref std std-term">positional arguments</span></a>. Now in extension functions implemented with using Argument Clinic this overhead is significantly decreased. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27574" class="reference external">bpo-27574</a>).

- Optimized <a href="../library/glob.html#glob.glob" class="reference internal" title="glob.glob"><span class="pre"><code class="sourceCode python">glob()</code></span></a> and <a href="../library/glob.html#glob.iglob" class="reference internal" title="glob.iglob"><span class="pre"><code class="sourceCode python">iglob()</code></span></a> functions in the <a href="../library/glob.html#module-glob" class="reference internal" title="glob: Unix shell style pathname pattern expansion."><span class="pre"><code class="sourceCode python">glob</code></span></a> module; they are now about 3–6 times faster. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25596" class="reference external">bpo-25596</a>).

- Optimized globbing in <a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a> by using <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a>; it is now about 1.5–4 times faster. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26032" class="reference external">bpo-26032</a>).

- <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> parsing, iteration and deepcopy performance has been significantly improved. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25638" class="reference external">bpo-25638</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25873" class="reference external">bpo-25873</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25869" class="reference external">bpo-25869</a>.)

- Creation of <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">fractions.Fraction</code></span></a> instances from floats and decimals is now 2 to 3 times faster. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25971" class="reference external">bpo-25971</a>.)

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Link to this heading">¶</a>

- Python now requires some C99 support in the toolchain to build. Most notably, Python now uses standard integer types and macros in place of custom macros like <span class="pre">`PY_LONG_LONG`</span>. For more information, see <span id="index-34" class="target"></span><a href="https://peps.python.org/pep-0007/" class="pep reference external"><strong>PEP 7</strong></a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17884" class="reference external">bpo-17884</a>.

- Cross-compiling CPython with the Android NDK and the Android API level set to 21 (Android 5.0 Lollipop) or greater runs successfully. While Android is not yet a supported platform, the Python test suite runs on the Android emulator with only about 16 tests failures. See the Android meta-issue <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26865" class="reference external">bpo-26865</a>.

- The <span class="pre">`--enable-optimizations`</span> configure flag has been added. Turning it on will activate expensive optimizations like PGO. (Original patch by Alecsandru Patrascu of Intel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26359" class="reference external">bpo-26359</a>.)

- The <a href="../glossary.html#term-global-interpreter-lock" class="reference internal"><span class="xref std std-term">GIL</span></a> must now be held when allocator functions of <a href="../c-api/memory.html#c.PYMEM_DOMAIN_OBJ" class="reference internal" title="PYMEM_DOMAIN_OBJ"><span class="pre"><code class="sourceCode c">PYMEM_DOMAIN_OBJ</code></span></a> (ex: <a href="../c-api/memory.html#c.PyObject_Malloc" class="reference internal" title="PyObject_Malloc"><span class="pre"><code class="sourceCode c">PyObject_Malloc<span class="op">()</span></code></span></a>) and <a href="../c-api/memory.html#c.PYMEM_DOMAIN_MEM" class="reference internal" title="PYMEM_DOMAIN_MEM"><span class="pre"><code class="sourceCode c">PYMEM_DOMAIN_MEM</code></span></a> (ex: <a href="../c-api/memory.html#c.PyMem_Malloc" class="reference internal" title="PyMem_Malloc"><span class="pre"><code class="sourceCode c">PyMem_Malloc<span class="op">()</span></code></span></a>) domains are called.

- New <a href="../c-api/init.html#c.Py_FinalizeEx" class="reference internal" title="Py_FinalizeEx"><span class="pre"><code class="sourceCode c">Py_FinalizeEx<span class="op">()</span></code></span></a> API which indicates if flushing buffered data failed. (Contributed by Martin Panter in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5319" class="reference external">bpo-5319</a>.)

- <a href="../c-api/arg.html#c.PyArg_ParseTupleAndKeywords" class="reference internal" title="PyArg_ParseTupleAndKeywords"><span class="pre"><code class="sourceCode c">PyArg_ParseTupleAndKeywords<span class="op">()</span></code></span></a> now supports <a href="../glossary.html#positional-only-parameter" class="reference internal"><span class="std std-ref">positional-only parameters</span></a>. Positional-only parameters are defined by empty names. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26282" class="reference external">bpo-26282</a>).

- <span class="pre">`PyTraceback_Print`</span> method now abbreviates long sequences of repeated lines as <span class="pre">`"[Previous`</span>` `<span class="pre">`line`</span>` `<span class="pre">`repeated`</span>` `<span class="pre">`{count}`</span>` `<span class="pre">`more`</span>` `<span class="pre">`times]"`</span>. (Contributed by Emanuel Barry in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26823" class="reference external">bpo-26823</a>.)

- The new <a href="../c-api/exceptions.html#c.PyErr_SetImportErrorSubclass" class="reference internal" title="PyErr_SetImportErrorSubclass"><span class="pre"><code class="sourceCode c">PyErr_SetImportErrorSubclass<span class="op">()</span></code></span></a> function allows for specifying a subclass of <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> to raise. (Contributed by Eric Snow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15767" class="reference external">bpo-15767</a>.)

- The new <a href="../c-api/exceptions.html#c.PyErr_ResourceWarning" class="reference internal" title="PyErr_ResourceWarning"><span class="pre"><code class="sourceCode c">PyErr_ResourceWarning<span class="op">()</span></code></span></a> function can be used to generate a <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a> providing the source of the resource allocation. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26567" class="reference external">bpo-26567</a>.)

- The new <a href="../c-api/sys.html#c.PyOS_FSPath" class="reference internal" title="PyOS_FSPath"><span class="pre"><code class="sourceCode c">PyOS_FSPath<span class="op">()</span></code></span></a> function returns the file system representation of a <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like object</span></a>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27186" class="reference external">bpo-27186</a>.)

- The <a href="../c-api/unicode.html#c.PyUnicode_FSConverter" class="reference internal" title="PyUnicode_FSConverter"><span class="pre"><code class="sourceCode c">PyUnicode_FSConverter<span class="op">()</span></code></span></a> and <a href="../c-api/unicode.html#c.PyUnicode_FSDecoder" class="reference internal" title="PyUnicode_FSDecoder"><span class="pre"><code class="sourceCode c">PyUnicode_FSDecoder<span class="op">()</span></code></span></a> functions will now accept <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like objects</span></a>.

</div>

<div id="other-improvements" class="section">

## Other Improvements<a href="#other-improvements" class="headerlink" title="Link to this heading">¶</a>

- When <a href="../using/cmdline.html#cmdoption-version" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--version</code></span></a> (short form: <a href="../using/cmdline.html#cmdoption-V" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-V</code></span></a>) is supplied twice, Python prints <a href="../library/sys.html#sys.version" class="reference internal" title="sys.version"><span class="pre"><code class="sourceCode python">sys.version</code></span></a> for detailed information.

  <div class="highlight-shell-session notranslate">

  <div class="highlight">

      $ ./python -VV
      Python 3.6.0b4+ (3.6:223967b49e49+, Nov 21 2016, 20:55:04)
      [GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.42.1)]

  </div>

  </div>

</div>

<div id="deprecated" class="section">

## Deprecated<a href="#deprecated" class="headerlink" title="Link to this heading">¶</a>

<div id="new-keywords" class="section">

### New Keywords<a href="#new-keywords" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`async`</span> and <span class="pre">`await`</span> are not recommended to be used as variable, class, function or module names. Introduced by <span id="index-35" class="target"></span><a href="https://peps.python.org/pep-0492/" class="pep reference external"><strong>PEP 492</strong></a> in Python 3.5, they will become proper keywords in Python 3.7. Starting in Python 3.6, the use of <span class="pre">`async`</span> or <span class="pre">`await`</span> as names will generate a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>.

</div>

<div id="deprecated-python-behavior" class="section">

### Deprecated Python behavior<a href="#deprecated-python-behavior" class="headerlink" title="Link to this heading">¶</a>

Raising the <a href="../library/exceptions.html#StopIteration" class="reference internal" title="StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a> exception inside a generator will now generate a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>, and will trigger a <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> in Python 3.7. See <a href="3.5.html#whatsnew-pep-479" class="reference internal"><span class="std std-ref">PEP 479: Change StopIteration handling inside generators</span></a> for details.

The <a href="../reference/datamodel.html#object.__aiter__" class="reference internal" title="object.__aiter__"><span class="pre"><code class="sourceCode python"><span class="fu">__aiter__</span>()</code></span></a> method is now expected to return an asynchronous iterator directly instead of returning an awaitable as previously. Doing the former will trigger a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. Backward compatibility will be removed in Python 3.7. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27243" class="reference external">bpo-27243</a>.)

A backslash-character pair that is not a valid escape sequence now generates a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. Although this will eventually become a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>, that will not be for several Python releases. (Contributed by Emanuel Barry in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27364" class="reference external">bpo-27364</a>.)

When performing a relative import, falling back on <span class="pre">`__name__`</span> and <span class="pre">`__path__`</span> from the calling module when <span class="pre">`__spec__`</span> or <span class="pre">`__package__`</span> are not defined now raises an <a href="../library/exceptions.html#ImportWarning" class="reference internal" title="ImportWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ImportWarning</span></code></span></a>. (Contributed by Rose Ames in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25791" class="reference external">bpo-25791</a>.)

</div>

<div id="deprecated-python-modules-functions-and-methods" class="section">

### Deprecated Python modules, functions and methods<a href="#deprecated-python-modules-functions-and-methods" class="headerlink" title="Link to this heading">¶</a>

<div id="asynchat" class="section">

#### asynchat<a href="#asynchat" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`asynchat`</span> has been deprecated in favor of <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>. (Contributed by Mariatta in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25002" class="reference external">bpo-25002</a>.)

</div>

<div id="asyncore" class="section">

#### asyncore<a href="#asyncore" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`asyncore`</span> has been deprecated in favor of <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>. (Contributed by Mariatta in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25002" class="reference external">bpo-25002</a>.)

</div>

<div id="dbm" class="section">

#### dbm<a href="#dbm" class="headerlink" title="Link to this heading">¶</a>

Unlike other <a href="../library/dbm.html#module-dbm" class="reference internal" title="dbm: Interfaces to various Unix &quot;database&quot; formats."><span class="pre"><code class="sourceCode python">dbm</code></span></a> implementations, the <a href="../library/dbm.html#module-dbm.dumb" class="reference internal" title="dbm.dumb: Portable implementation of the simple DBM interface."><span class="pre"><code class="sourceCode python">dbm.dumb</code></span></a> module creates databases with the <span class="pre">`'rw'`</span> mode and allows modifying the database opened with the <span class="pre">`'r'`</span> mode. This behavior is now deprecated and will be removed in 3.8. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21708" class="reference external">bpo-21708</a>.)

</div>

<div id="id2" class="section">

#### distutils<a href="#id2" class="headerlink" title="Link to this heading">¶</a>

The undocumented <span class="pre">`extra_path`</span> argument to the <span class="pre">`distutils.Distribution`</span> constructor is now considered deprecated and will raise a warning if set. Support for this parameter will be removed in a future Python release. See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27919" class="reference external">bpo-27919</a> for details.

</div>

<div id="grp" class="section">

#### grp<a href="#grp" class="headerlink" title="Link to this heading">¶</a>

The support of non-integer arguments in <a href="../library/grp.html#grp.getgrgid" class="reference internal" title="grp.getgrgid"><span class="pre"><code class="sourceCode python">getgrgid()</code></span></a> has been deprecated. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26129" class="reference external">bpo-26129</a>.)

</div>

<div id="id3" class="section">

#### importlib<a href="#id3" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/importlib.html#importlib.machinery.SourceFileLoader.load_module" class="reference internal" title="importlib.machinery.SourceFileLoader.load_module"><span class="pre"><code class="sourceCode python">importlib.machinery.SourceFileLoader.load_module()</code></span></a> and <a href="../library/importlib.html#importlib.machinery.SourcelessFileLoader.load_module" class="reference internal" title="importlib.machinery.SourcelessFileLoader.load_module"><span class="pre"><code class="sourceCode python">importlib.machinery.SourcelessFileLoader.load_module()</code></span></a> methods are now deprecated. They were the only remaining implementations of <a href="../library/importlib.html#importlib.abc.Loader.load_module" class="reference internal" title="importlib.abc.Loader.load_module"><span class="pre"><code class="sourceCode python">importlib.abc.Loader.load_module()</code></span></a> in <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> that had not been deprecated in previous versions of Python in favour of <a href="../library/importlib.html#importlib.abc.Loader.exec_module" class="reference internal" title="importlib.abc.Loader.exec_module"><span class="pre"><code class="sourceCode python">importlib.abc.Loader.exec_module()</code></span></a>.

The <a href="../library/importlib.html#importlib.machinery.WindowsRegistryFinder" class="reference internal" title="importlib.machinery.WindowsRegistryFinder"><span class="pre"><code class="sourceCode python">importlib.machinery.WindowsRegistryFinder</code></span></a> class is now deprecated. As of 3.6.0, it is still added to <a href="../library/sys.html#sys.meta_path" class="reference internal" title="sys.meta_path"><span class="pre"><code class="sourceCode python">sys.meta_path</code></span></a> by default (on Windows), but this may change in future releases.

</div>

<div id="id4" class="section">

#### os<a href="#id4" class="headerlink" title="Link to this heading">¶</a>

Undocumented support of general <a href="../glossary.html#term-bytes-like-object" class="reference internal"><span class="xref std std-term">bytes-like objects</span></a> as paths in <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> functions, <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a> and similar functions is now deprecated. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25791" class="reference external">bpo-25791</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26754" class="reference external">bpo-26754</a>.)

</div>

<div id="id5" class="section">

#### re<a href="#id5" class="headerlink" title="Link to this heading">¶</a>

Support for inline flags <span class="pre">`(?letters)`</span> in the middle of the regular expression has been deprecated and will be removed in a future Python version. Flags at the start of a regular expression are still allowed. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22493" class="reference external">bpo-22493</a>.)

</div>

<div id="id6" class="section">

#### ssl<a href="#id6" class="headerlink" title="Link to this heading">¶</a>

OpenSSL 0.9.8, 1.0.0 and 1.0.1 are deprecated and no longer supported. In the future the <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module will require at least OpenSSL 1.0.2 or 1.1.0.

SSL-related arguments like <span class="pre">`certfile`</span>, <span class="pre">`keyfile`</span> and <span class="pre">`check_hostname`</span> in <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a>, <a href="../library/http.client.html#module-http.client" class="reference internal" title="http.client: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">http.client</code></span></a>, <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a>, <a href="../library/poplib.html#module-poplib" class="reference internal" title="poplib: POP3 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">poplib</code></span></a>, and <a href="../library/smtplib.html#module-smtplib" class="reference internal" title="smtplib: SMTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">smtplib</code></span></a> have been deprecated in favor of <span class="pre">`context`</span>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28022" class="reference external">bpo-28022</a>.)

A couple of protocols and functions of the <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module are now deprecated. Some features will no longer be available in future versions of OpenSSL. Other features are deprecated in favor of a different API. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28022" class="reference external">bpo-28022</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26470" class="reference external">bpo-26470</a>.)

</div>

<div id="id7" class="section">

#### tkinter<a href="#id7" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`tkinter.tix`</span> module is now deprecated. <a href="../library/tkinter.html#module-tkinter" class="reference internal" title="tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">tkinter</code></span></a> users should use <a href="../library/tkinter.ttk.html#module-tkinter.ttk" class="reference internal" title="tkinter.ttk: Tk themed widget set"><span class="pre"><code class="sourceCode python">tkinter.ttk</code></span></a> instead.

</div>

<div id="whatsnew36-venv" class="section">

<span id="id8"></span>

#### venv<a href="#whatsnew36-venv" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`pyvenv`</span> script has been deprecated in favour of <span class="pre">`python3`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`venv`</span>. This prevents confusion as to what Python interpreter <span class="pre">`pyvenv`</span> is connected to and thus what Python interpreter will be used by the virtual environment. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25154" class="reference external">bpo-25154</a>.)

</div>

</div>

<div id="xml" class="section">

### xml<a href="#xml" class="headerlink" title="Link to this heading">¶</a>

- As mitigation against DTD and external entity retrieval, the <a href="../library/xml.dom.minidom.html#module-xml.dom.minidom" class="reference internal" title="xml.dom.minidom: Minimal Document Object Model (DOM) implementation."><span class="pre"><code class="sourceCode python">xml.dom.minidom</code></span></a> and <a href="../library/xml.sax.html#module-xml.sax" class="reference internal" title="xml.sax: Package containing SAX2 base classes and convenience functions."><span class="pre"><code class="sourceCode python">xml.sax</code></span></a> modules no longer process external entities by default. (Contributed by Christian Heimes in <a href="https://github.com/python/cpython/issues/61441" class="reference external">gh-61441</a>.)

</div>

<div id="deprecated-functions-and-types-of-the-c-api" class="section">

### Deprecated functions and types of the C API<a href="#deprecated-functions-and-types-of-the-c-api" class="headerlink" title="Link to this heading">¶</a>

Undocumented functions <span class="pre">`PyUnicode_AsEncodedObject()`</span>, <span class="pre">`PyUnicode_AsDecodedObject()`</span>, <span class="pre">`PyUnicode_AsEncodedUnicode()`</span> and <span class="pre">`PyUnicode_AsDecodedUnicode()`</span> are deprecated now. Use the <a href="../c-api/codec.html#codec-registry" class="reference internal"><span class="std std-ref">generic codec based API</span></a> instead.

</div>

<div id="deprecated-build-options" class="section">

### Deprecated Build Options<a href="#deprecated-build-options" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`--with-system-ffi`</span> configure flag is now on by default on non-macOS UNIX platforms. It may be disabled by using <span class="pre">`--without-system-ffi`</span>, but using the flag is deprecated and will not be accepted in Python 3.7. macOS is unaffected by this change. Note that many OS distributors already use the <span class="pre">`--with-system-ffi`</span> flag when building their system Python.

</div>

</div>

<div id="removed" class="section">

## Removed<a href="#removed" class="headerlink" title="Link to this heading">¶</a>

<div id="api-and-feature-removals" class="section">

### API and Feature Removals<a href="#api-and-feature-removals" class="headerlink" title="Link to this heading">¶</a>

- Unknown escapes consisting of <span class="pre">`'\'`</span> and an ASCII letter in regular expressions will now cause an error. In replacement templates for <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">re.sub()</code></span></a> they are still allowed, but deprecated. The <a href="../library/re.html#re.LOCALE" class="reference internal" title="re.LOCALE"><span class="pre"><code class="sourceCode python">re.LOCALE</code></span></a> flag can now only be used with binary patterns.

- <span class="pre">`inspect.getmoduleinfo()`</span> was removed (was deprecated since CPython 3.3). <a href="../library/inspect.html#inspect.getmodulename" class="reference internal" title="inspect.getmodulename"><span class="pre"><code class="sourceCode python">inspect.getmodulename()</code></span></a> should be used for obtaining the module name for a given path. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13248" class="reference external">bpo-13248</a>.)

- <span class="pre">`traceback.Ignore`</span> class and <span class="pre">`traceback.usage`</span>, <span class="pre">`traceback.modname`</span>, <span class="pre">`traceback.fullmodname`</span>, <span class="pre">`traceback.find_lines_from_code`</span>, <span class="pre">`traceback.find_lines`</span>, <span class="pre">`traceback.find_strings`</span>, <span class="pre">`traceback.find_executable_lines`</span> methods were removed from the <a href="../library/traceback.html#module-traceback" class="reference internal" title="traceback: Print or retrieve a stack traceback."><span class="pre"><code class="sourceCode python">traceback</code></span></a> module. They were undocumented methods deprecated since Python 3.2 and equivalent functionality is available from private methods.

- The <span class="pre">`tk_menuBar()`</span> and <span class="pre">`tk_bindForTraversal()`</span> dummy methods in <a href="../library/tkinter.html#module-tkinter" class="reference internal" title="tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">tkinter</code></span></a> widget classes were removed (corresponding Tk commands were obsolete since Tk 4.0).

- The <a href="../library/zipfile.html#zipfile.ZipFile.open" class="reference internal" title="zipfile.ZipFile.open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> method of the <a href="../library/zipfile.html#zipfile.ZipFile" class="reference internal" title="zipfile.ZipFile"><span class="pre"><code class="sourceCode python">zipfile.ZipFile</code></span></a> class no longer supports the <span class="pre">`'U'`</span> mode (was deprecated since Python 3.4). Use <a href="../library/io.html#io.TextIOWrapper" class="reference internal" title="io.TextIOWrapper"><span class="pre"><code class="sourceCode python">io.TextIOWrapper</code></span></a> for reading compressed text files in <a href="../glossary.html#term-universal-newlines" class="reference internal"><span class="xref std std-term">universal newlines</span></a> mode.

- The undocumented <span class="pre">`IN`</span>, <span class="pre">`CDROM`</span>, <span class="pre">`DLFCN`</span>, <span class="pre">`TYPES`</span>, <span class="pre">`CDIO`</span>, and <span class="pre">`STROPTS`</span> modules have been removed. They had been available in the platform specific <span class="pre">`Lib/plat-*/`</span> directories, but were chronically out of date, inconsistently available across platforms, and unmaintained. The script that created these modules is still available in the source distribution at <a href="https://github.com/python/cpython/blob/v3.6.15/Tools/scripts/h2py.py" class="reference external">Tools/scripts/h2py.py</a>.

- The deprecated <span class="pre">`asynchat.fifo`</span> class has been removed.

</div>

</div>

<div id="porting-to-python-3-6" class="section">

## Porting to Python 3.6<a href="#porting-to-python-3-6" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="changes-in-python-command-behavior" class="section">

### Changes in ‘python’ Command Behavior<a href="#changes-in-python-command-behavior" class="headerlink" title="Link to this heading">¶</a>

- The output of a special Python build with defined <span class="pre">`COUNT_ALLOCS`</span>, <span class="pre">`SHOW_ALLOC_COUNT`</span> or <span class="pre">`SHOW_TRACK_COUNT`</span> macros is now off by default. It can be re-enabled using the <span class="pre">`-X`</span>` `<span class="pre">`showalloccount`</span> option. It now outputs to <span class="pre">`stderr`</span> instead of <span class="pre">`stdout`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23034" class="reference external">bpo-23034</a>.)

</div>

<div id="changes-in-the-python-api" class="section">

### Changes in the Python API<a href="#changes-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> will no longer allow combining the <span class="pre">`'U'`</span> mode flag with <span class="pre">`'+'`</span>. (Contributed by Jeff Balogh and John O’Connor in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2091" class="reference external">bpo-2091</a>.)

- <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> no longer implicitly commits an open transaction before DDL statements.

- On Linux, <a href="../library/os.html#os.urandom" class="reference internal" title="os.urandom"><span class="pre"><code class="sourceCode python">os.urandom()</code></span></a> now blocks until the system urandom entropy pool is initialized to increase the security.

- When <a href="../library/importlib.html#importlib.abc.Loader.exec_module" class="reference internal" title="importlib.abc.Loader.exec_module"><span class="pre"><code class="sourceCode python">importlib.abc.Loader.exec_module()</code></span></a> is defined, <a href="../library/importlib.html#importlib.abc.Loader.create_module" class="reference internal" title="importlib.abc.Loader.create_module"><span class="pre"><code class="sourceCode python">importlib.abc.Loader.create_module()</code></span></a> must also be defined.

- <a href="../c-api/exceptions.html#c.PyErr_SetImportError" class="reference internal" title="PyErr_SetImportError"><span class="pre"><code class="sourceCode c">PyErr_SetImportError<span class="op">()</span></code></span></a> now sets <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> when its **msg** argument is not set. Previously only <span class="pre">`NULL`</span> was returned.

- The format of the <a href="../reference/datamodel.html#codeobject.co_lnotab" class="reference internal" title="codeobject.co_lnotab"><span class="pre"><code class="sourceCode python">co_lnotab</code></span></a> attribute of code objects changed to support a negative line number delta. By default, Python does not emit bytecode with a negative line number delta. Functions using <a href="../reference/datamodel.html#frame.f_lineno" class="reference internal" title="frame.f_lineno"><span class="pre"><code class="sourceCode python">frame.f_lineno</code></span></a>, <span class="pre">`PyFrame_GetLineNumber()`</span> or <span class="pre">`PyCode_Addr2Line()`</span> are not affected. Functions directly decoding <span class="pre">`co_lnotab`</span> should be updated to use a signed 8-bit integer type for the line number delta, but this is only required to support applications using a negative line number delta. See <span class="pre">`Objects/lnotab_notes.txt`</span> for the <span class="pre">`co_lnotab`</span> format and how to decode it, and see the <span id="index-36" class="target"></span><a href="https://peps.python.org/pep-0511/" class="pep reference external"><strong>PEP 511</strong></a> for the rationale.

- The functions in the <a href="../library/compileall.html#module-compileall" class="reference internal" title="compileall: Tools for byte-compiling all Python source files in a directory tree."><span class="pre"><code class="sourceCode python">compileall</code></span></a> module now return booleans instead of <span class="pre">`1`</span> or <span class="pre">`0`</span> to represent success or failure, respectively. Thanks to booleans being a subclass of integers, this should only be an issue if you were doing identity checks for <span class="pre">`1`</span> or <span class="pre">`0`</span>. See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25768" class="reference external">bpo-25768</a>.

- Reading the <span class="pre">`port`</span> attribute of <a href="../library/urllib.parse.html#urllib.parse.urlsplit" class="reference internal" title="urllib.parse.urlsplit"><span class="pre"><code class="sourceCode python">urllib.parse.urlsplit()</code></span></a> and <a href="../library/urllib.parse.html#urllib.parse.urlparse" class="reference internal" title="urllib.parse.urlparse"><span class="pre"><code class="sourceCode python">urlparse()</code></span></a> results now raises <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> for out-of-range values, rather than returning <a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a>. See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20059" class="reference external">bpo-20059</a>.

- The <span class="pre">`imp`</span> module now raises a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> instead of <a href="../library/exceptions.html#PendingDeprecationWarning" class="reference internal" title="PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a>.

- The following modules have had missing APIs added to their <a href="../reference/simple_stmts.html#module.__all__" class="reference internal" title="module.__all__"><span class="pre"><code class="sourceCode python"><span class="va">__all__</span></code></span></a> attributes to match the documented APIs: <a href="../library/calendar.html#module-calendar" class="reference internal" title="calendar: Functions for working with calendars, including some emulation of the Unix cal program."><span class="pre"><code class="sourceCode python">calendar</code></span></a>, <span class="pre">`cgi`</span>, <a href="../library/csv.html#module-csv" class="reference internal" title="csv: Write and read tabular data to and from delimited files."><span class="pre"><code class="sourceCode python">csv</code></span></a>, <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">ElementTree</code></span></a>, <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">enum</code></span></a>, <a href="../library/fileinput.html#module-fileinput" class="reference internal" title="fileinput: Loop over standard input or a list of files."><span class="pre"><code class="sourceCode python">fileinput</code></span></a>, <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a>, <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a>, <a href="../library/mailbox.html#module-mailbox" class="reference internal" title="mailbox: Manipulate mailboxes in various formats"><span class="pre"><code class="sourceCode python">mailbox</code></span></a>, <a href="../library/mimetypes.html#module-mimetypes" class="reference internal" title="mimetypes: Mapping of filename extensions to MIME types."><span class="pre"><code class="sourceCode python">mimetypes</code></span></a>, <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library."><span class="pre"><code class="sourceCode python">optparse</code></span></a>, <a href="../library/plistlib.html#module-plistlib" class="reference internal" title="plistlib: Generate and parse Apple plist files."><span class="pre"><code class="sourceCode python">plistlib</code></span></a>, <span class="pre">`smtpd`</span>, <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a>, <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>, <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a> and <a href="../library/wave.html#module-wave" class="reference internal" title="wave: Provide an interface to the WAV sound format."><span class="pre"><code class="sourceCode python">wave</code></span></a>. This means they will export new symbols when <span class="pre">`import`</span>` `<span class="pre">`*`</span> is used. (Contributed by Joel Taddei and Jacek Kołodziej in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23883" class="reference external">bpo-23883</a>.)

- When performing a relative import, if <span class="pre">`__package__`</span> does not compare equal to <span class="pre">`__spec__.parent`</span> then <a href="../library/exceptions.html#ImportWarning" class="reference internal" title="ImportWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ImportWarning</span></code></span></a> is raised. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25791" class="reference external">bpo-25791</a>.)

- When a relative import is performed and no parent package is known, then <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> will be raised. Previously, <a href="../library/exceptions.html#SystemError" class="reference internal" title="SystemError"><span class="pre"><code class="sourceCode python"><span class="pp">SystemError</span></code></span></a> could be raised. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18018" class="reference external">bpo-18018</a>.)

- Servers based on the <a href="../library/socketserver.html#module-socketserver" class="reference internal" title="socketserver: A framework for network servers."><span class="pre"><code class="sourceCode python">socketserver</code></span></a> module, including those defined in <a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a>, <a href="../library/xmlrpc.server.html#module-xmlrpc.server" class="reference internal" title="xmlrpc.server: Basic XML-RPC server implementations."><span class="pre"><code class="sourceCode python">xmlrpc.server</code></span></a> and <a href="../library/wsgiref.html#module-wsgiref.simple_server" class="reference internal" title="wsgiref.simple_server: A simple WSGI HTTP server."><span class="pre"><code class="sourceCode python">wsgiref.simple_server</code></span></a>, now only catch exceptions derived from <a href="../library/exceptions.html#Exception" class="reference internal" title="Exception"><span class="pre"><code class="sourceCode python"><span class="pp">Exception</span></code></span></a>. Therefore if a request handler raises an exception like <a href="../library/exceptions.html#SystemExit" class="reference internal" title="SystemExit"><span class="pre"><code class="sourceCode python"><span class="pp">SystemExit</span></code></span></a> or <a href="../library/exceptions.html#KeyboardInterrupt" class="reference internal" title="KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a>, <a href="../library/socketserver.html#socketserver.BaseServer.handle_error" class="reference internal" title="socketserver.BaseServer.handle_error"><span class="pre"><code class="sourceCode python">handle_error()</code></span></a> is no longer called, and the exception will stop a single-threaded server. (Contributed by Martin Panter in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23430" class="reference external">bpo-23430</a>.)

- <span class="pre">`spwd.getspnam()`</span> now raises a <a href="../library/exceptions.html#PermissionError" class="reference internal" title="PermissionError"><span class="pre"><code class="sourceCode python"><span class="pp">PermissionError</span></code></span></a> instead of <a href="../library/exceptions.html#KeyError" class="reference internal" title="KeyError"><span class="pre"><code class="sourceCode python"><span class="pp">KeyError</span></code></span></a> if the user doesn’t have privileges.

- The <a href="../library/socket.html#socket.socket.close" class="reference internal" title="socket.socket.close"><span class="pre"><code class="sourceCode python">socket.socket.close()</code></span></a> method now raises an exception if an error (e.g. <span class="pre">`EBADF`</span>) was reported by the underlying system call. (Contributed by Martin Panter in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26685" class="reference external">bpo-26685</a>.)

- The *decode_data* argument for the <span class="pre">`smtpd.SMTPChannel`</span> and <span class="pre">`smtpd.SMTPServer`</span> constructors is now <span class="pre">`False`</span> by default. This means that the argument passed to <span class="pre">`process_message()`</span> is now a bytes object by default, and <span class="pre">`process_message()`</span> will be passed keyword arguments. Code that has already been updated in accordance with the deprecation warning generated by 3.5 will not be affected.

- All optional arguments of the <a href="../library/json.html#json.dump" class="reference internal" title="json.dump"><span class="pre"><code class="sourceCode python">dump()</code></span></a>, <a href="../library/json.html#json.dumps" class="reference internal" title="json.dumps"><span class="pre"><code class="sourceCode python">dumps()</code></span></a>, <a href="../library/json.html#json.load" class="reference internal" title="json.load"><span class="pre"><code class="sourceCode python">load()</code></span></a> and <a href="../library/json.html#json.loads" class="reference internal" title="json.loads"><span class="pre"><code class="sourceCode python">loads()</code></span></a> functions and <a href="../library/json.html#json.JSONEncoder" class="reference internal" title="json.JSONEncoder"><span class="pre"><code class="sourceCode python">JSONEncoder</code></span></a> and <a href="../library/json.html#json.JSONDecoder" class="reference internal" title="json.JSONDecoder"><span class="pre"><code class="sourceCode python">JSONDecoder</code></span></a> class constructors in the <a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> module are now <a href="../glossary.html#keyword-only-parameter" class="reference internal"><span class="std std-ref">keyword-only</span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18726" class="reference external">bpo-18726</a>.)

- Subclasses of <a href="../library/functions.html#type" class="reference internal" title="type"><span class="pre"><code class="sourceCode python"><span class="bu">type</span></code></span></a> which don’t override <span class="pre">`type.__new__`</span> may no longer use the one-argument form to get the type of an object.

- As part of <span id="index-37" class="target"></span><a href="https://peps.python.org/pep-0487/" class="pep reference external"><strong>PEP 487</strong></a>, the handling of keyword arguments passed to <a href="../library/functions.html#type" class="reference internal" title="type"><span class="pre"><code class="sourceCode python"><span class="bu">type</span></code></span></a> (other than the metaclass hint, <span class="pre">`metaclass`</span>) is now consistently delegated to <a href="../reference/datamodel.html#object.__init_subclass__" class="reference internal" title="object.__init_subclass__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__init_subclass__</span>()</code></span></a>. This means that <a href="../reference/datamodel.html#object.__new__" class="reference internal" title="object.__new__"><span class="pre"><code class="sourceCode python"><span class="bu">type</span>.<span class="fu">__new__</span></code></span></a> and <a href="../reference/datamodel.html#object.__init__" class="reference internal" title="object.__init__"><span class="pre"><code class="sourceCode python"><span class="bu">type</span>.<span class="fu">__init__</span></code></span></a> both now accept arbitrary keyword arguments, but <a href="../reference/datamodel.html#object.__init_subclass__" class="reference internal" title="object.__init_subclass__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__init_subclass__</span>()</code></span></a> (which is called from <a href="../reference/datamodel.html#object.__new__" class="reference internal" title="object.__new__"><span class="pre"><code class="sourceCode python"><span class="bu">type</span>.<span class="fu">__new__</span></code></span></a>) will reject them by default. Custom metaclasses accepting additional keyword arguments will need to adjust their calls to <a href="../reference/datamodel.html#object.__new__" class="reference internal" title="object.__new__"><span class="pre"><code class="sourceCode python"><span class="bu">type</span>.<span class="fu">__new__</span></code></span></a> (whether direct or via <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span></code></span></a>) accordingly.

- In <span class="pre">`distutils.command.sdist.sdist`</span>, the <span class="pre">`default_format`</span> attribute has been removed and is no longer honored. Instead, the gzipped tarfile format is the default on all platforms and no platform-specific selection is made. In environments where distributions are built on Windows and zip distributions are required, configure the project with a <span class="pre">`setup.cfg`</span> file containing the following:

  <div class="highlight-ini notranslate">

  <div class="highlight">

      [sdist]
      formats=zip

  </div>

  </div>

  This behavior has also been backported to earlier Python versions by Setuptools 26.0.0.

- In the <a href="../library/urllib.request.html#module-urllib.request" class="reference internal" title="urllib.request: Extensible library for opening URLs."><span class="pre"><code class="sourceCode python">urllib.request</code></span></a> module and the <a href="../library/http.client.html#http.client.HTTPConnection.request" class="reference internal" title="http.client.HTTPConnection.request"><span class="pre"><code class="sourceCode python">http.client.HTTPConnection.request()</code></span></a> method, if no Content-Length header field has been specified and the request body is a file object, it is now sent with HTTP 1.1 chunked encoding. If a file object has to be sent to a HTTP 1.0 server, the Content-Length value now has to be specified by the caller. (Contributed by Demian Brecht and Rolf Krahl with tweaks from Martin Panter in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12319" class="reference external">bpo-12319</a>.)

- The <a href="../library/csv.html#csv.DictReader" class="reference internal" title="csv.DictReader"><span class="pre"><code class="sourceCode python">DictReader</code></span></a> now returns rows of type <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">OrderedDict</code></span></a>. (Contributed by Steve Holden in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27842" class="reference external">bpo-27842</a>.)

- The <span class="pre">`crypt.METHOD_CRYPT`</span> will no longer be added to <span class="pre">`crypt.methods`</span> if unsupported by the platform. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25287" class="reference external">bpo-25287</a>.)

- The *verbose* and *rename* arguments for <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">namedtuple()</code></span></a> are now keyword-only. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25628" class="reference external">bpo-25628</a>.)

- On Linux, <a href="../library/ctypes.html#ctypes.util.find_library" class="reference internal" title="ctypes.util.find_library"><span class="pre"><code class="sourceCode python">ctypes.util.find_library()</code></span></a> now looks in <span class="pre">`LD_LIBRARY_PATH`</span> for shared libraries. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9998" class="reference external">bpo-9998</a>.)

- The <a href="../library/imaplib.html#imaplib.IMAP4" class="reference internal" title="imaplib.IMAP4"><span class="pre"><code class="sourceCode python">imaplib.IMAP4</code></span></a> class now handles flags containing the <span class="pre">`']'`</span> character in messages sent from the server to improve real-world compatibility. (Contributed by Lita Cho in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21815" class="reference external">bpo-21815</a>.)

- The <a href="../library/mmap.html#mmap.mmap.write" class="reference internal" title="mmap.mmap.write"><span class="pre"><code class="sourceCode python">mmap.mmap.write()</code></span></a> function now returns the number of bytes written like other write methods. (Contributed by Jakub Stasiak in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26335" class="reference external">bpo-26335</a>.)

- The <a href="../library/pkgutil.html#pkgutil.iter_modules" class="reference internal" title="pkgutil.iter_modules"><span class="pre"><code class="sourceCode python">pkgutil.iter_modules()</code></span></a> and <a href="../library/pkgutil.html#pkgutil.walk_packages" class="reference internal" title="pkgutil.walk_packages"><span class="pre"><code class="sourceCode python">pkgutil.walk_packages()</code></span></a> functions now return <a href="../library/pkgutil.html#pkgutil.ModuleInfo" class="reference internal" title="pkgutil.ModuleInfo"><span class="pre"><code class="sourceCode python">ModuleInfo</code></span></a> named tuples. (Contributed by Ramchandra Apte in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17211" class="reference external">bpo-17211</a>.)

- <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">re.sub()</code></span></a> now raises an error for invalid numerical group references in replacement templates even if the pattern is not found in the string. The error message for invalid group references now includes the group index and the position of the reference. (Contributed by SilentGhost, Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25953" class="reference external">bpo-25953</a>.)

- <a href="../library/zipfile.html#zipfile.ZipFile" class="reference internal" title="zipfile.ZipFile"><span class="pre"><code class="sourceCode python">zipfile.ZipFile</code></span></a> will now raise <a href="../library/exceptions.html#NotImplementedError" class="reference internal" title="NotImplementedError"><span class="pre"><code class="sourceCode python"><span class="pp">NotImplementedError</span></code></span></a> for unrecognized compression values. Previously a plain <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> was raised. Additionally, calling <a href="../library/zipfile.html#zipfile.ZipFile" class="reference internal" title="zipfile.ZipFile"><span class="pre"><code class="sourceCode python">ZipFile</code></span></a> methods on a closed ZipFile or calling the <a href="../library/zipfile.html#zipfile.ZipFile.write" class="reference internal" title="zipfile.ZipFile.write"><span class="pre"><code class="sourceCode python">write()</code></span></a> method on a ZipFile created with mode <span class="pre">`'r'`</span> will raise a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a>. Previously, a <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> was raised in those scenarios.

- when custom metaclasses are combined with zero-argument <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a> or direct references from methods to the implicit <span class="pre">`__class__`</span> closure variable, the implicit <span class="pre">`__classcell__`</span> namespace entry must now be passed up to <span class="pre">`type.__new__`</span> for initialisation. Failing to do so will result in a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> in Python 3.6 and a <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> in Python 3.8.

- With the introduction of <a href="../library/exceptions.html#ModuleNotFoundError" class="reference internal" title="ModuleNotFoundError"><span class="pre"><code class="sourceCode python"><span class="pp">ModuleNotFoundError</span></code></span></a>, import system consumers may start expecting import system replacements to raise that more specific exception when appropriate, rather than the less-specific <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a>. To provide future compatibility with such consumers, implementers of alternative import systems that completely replace <a href="../library/functions.html#import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> will need to update their implementations to raise the new subclass when a module can’t be found at all. Implementers of compliant plugins to the default import system shouldn’t need to make any changes, as the default import system will raise the new subclass when appropriate.

</div>

<div id="changes-in-the-c-api" class="section">

### Changes in the C API<a href="#changes-in-the-c-api" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../c-api/memory.html#c.PyMem_Malloc" class="reference internal" title="PyMem_Malloc"><span class="pre"><code class="sourceCode c">PyMem_Malloc<span class="op">()</span></code></span></a> allocator family now uses the <a href="../c-api/memory.html#pymalloc" class="reference internal"><span class="std std-ref">pymalloc allocator</span></a> rather than the system <span class="pre">`malloc()`</span>. Applications calling <a href="../c-api/memory.html#c.PyMem_Malloc" class="reference internal" title="PyMem_Malloc"><span class="pre"><code class="sourceCode c">PyMem_Malloc<span class="op">()</span></code></span></a> without holding the GIL can now crash. Set the <span id="index-38" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONMALLOC" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONMALLOC</code></span></a> environment variable to <span class="pre">`debug`</span> to validate the usage of memory allocators in your application. See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26249" class="reference external">bpo-26249</a>.

- <a href="../c-api/sys.html#c.Py_Exit" class="reference internal" title="Py_Exit"><span class="pre"><code class="sourceCode c">Py_Exit<span class="op">()</span></code></span></a> (and the main interpreter) now override the exit status with 120 if flushing buffered data failed. See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5319" class="reference external">bpo-5319</a>.

</div>

<div id="cpython-bytecode-changes" class="section">

### CPython bytecode changes<a href="#cpython-bytecode-changes" class="headerlink" title="Link to this heading">¶</a>

There have been several major changes to the <a href="../glossary.html#term-bytecode" class="reference internal"><span class="xref std std-term">bytecode</span></a> in Python 3.6.

- The Python interpreter now uses a 16-bit wordcode instead of bytecode. (Contributed by Demur Rumed with input and reviews from Serhiy Storchaka and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26647" class="reference external">bpo-26647</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28050" class="reference external">bpo-28050</a>.)

- The new <span class="pre">`FORMAT_VALUE`</span> and <a href="../library/dis.html#opcode-BUILD_STRING" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">BUILD_STRING</code></span></a> opcodes as part of the <a href="#whatsnew36-pep498" class="reference internal"><span class="std std-ref">formatted string literal</span></a> implementation. (Contributed by Eric Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25483" class="reference external">bpo-25483</a> and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27078" class="reference external">bpo-27078</a>.)

- The new <a href="../library/dis.html#opcode-BUILD_CONST_KEY_MAP" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">BUILD_CONST_KEY_MAP</code></span></a> opcode to optimize the creation of dictionaries with constant keys. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27140" class="reference external">bpo-27140</a>.)

- The function call opcodes have been heavily reworked for better performance and simpler implementation. The <a href="../library/dis.html#opcode-MAKE_FUNCTION" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">MAKE_FUNCTION</code></span></a>, <span class="pre">`CALL_FUNCTION`</span>, <span class="pre">`CALL_FUNCTION_KW`</span> and <span class="pre">`BUILD_MAP_UNPACK_WITH_CALL`</span> opcodes have been modified, the new <a href="../library/dis.html#opcode-CALL_FUNCTION_EX" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">CALL_FUNCTION_EX</code></span></a> and <span class="pre">`BUILD_TUPLE_UNPACK_WITH_CALL`</span> have been added, and <span class="pre">`CALL_FUNCTION_VAR`</span>, <span class="pre">`CALL_FUNCTION_VAR_KW`</span> and <span class="pre">`MAKE_CLOSURE`</span> opcodes have been removed. (Contributed by Demur Rumed in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27095" class="reference external">bpo-27095</a>, and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27213" class="reference external">bpo-27213</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28257" class="reference external">bpo-28257</a>.)

- The new <a href="../library/dis.html#opcode-SETUP_ANNOTATIONS" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">SETUP_ANNOTATIONS</code></span></a> and <span class="pre">`STORE_ANNOTATION`</span> opcodes have been added to support the new <a href="../glossary.html#term-variable-annotation" class="reference internal"><span class="xref std std-term">variable annotation</span></a> syntax. (Contributed by Ivan Levkivskyi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27985" class="reference external">bpo-27985</a>.)

</div>

</div>

<div id="notable-changes-in-python-3-6-2" class="section">

## Notable changes in Python 3.6.2<a href="#notable-changes-in-python-3-6-2" class="headerlink" title="Link to this heading">¶</a>

<div id="new-make-regen-all-build-target" class="section">

### New <span class="pre">`make`</span>` `<span class="pre">`regen-all`</span> build target<a href="#new-make-regen-all-build-target" class="headerlink" title="Link to this heading">¶</a>

To simplify cross-compilation, and to ensure that CPython can reliably be compiled without requiring an existing version of Python to already be available, the autotools-based build system no longer attempts to implicitly recompile generated files based on file modification times.

Instead, a new <span class="pre">`make`</span>` `<span class="pre">`regen-all`</span> command has been added to force regeneration of these files when desired (e.g. after an initial version of Python has already been built based on the pregenerated versions).

More selective regeneration targets are also defined - see <a href="https://github.com/python/cpython/tree/3.13/Makefile.pre.in" class="extlink-source reference external">Makefile.pre.in</a> for details.

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23404" class="reference external">bpo-23404</a>.)

<div class="versionadded">

<span class="versionmodified added">Added in version 3.6.2.</span>

</div>

</div>

<div id="removal-of-make-touch-build-target" class="section">

### Removal of <span class="pre">`make`</span>` `<span class="pre">`touch`</span> build target<a href="#removal-of-make-touch-build-target" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`make`</span>` `<span class="pre">`touch`</span> build target previously used to request implicit regeneration of generated files by updating their modification times has been removed.

It has been replaced by the new <span class="pre">`make`</span>` `<span class="pre">`regen-all`</span> target.

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23404" class="reference external">bpo-23404</a>.)

<div class="versionchanged">

<span class="versionmodified changed">Changed in version 3.6.2.</span>

</div>

</div>

</div>

<div id="notable-changes-in-python-3-6-4" class="section">

## Notable changes in Python 3.6.4<a href="#notable-changes-in-python-3-6-4" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`PyExc_RecursionErrorInst`</span> singleton that was part of the public API has been removed as its members being never cleared may cause a segfault during finalization of the interpreter. (Contributed by Xavier de Gaye in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22898" class="reference external">bpo-22898</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30697" class="reference external">bpo-30697</a>.)

</div>

<div id="notable-changes-in-python-3-6-5" class="section">

## Notable changes in Python 3.6.5<a href="#notable-changes-in-python-3-6-5" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/locale.html#locale.localeconv" class="reference internal" title="locale.localeconv"><span class="pre"><code class="sourceCode python">locale.localeconv()</code></span></a> function now sets temporarily the <span class="pre">`LC_CTYPE`</span> locale to the <span class="pre">`LC_NUMERIC`</span> locale in some cases. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31900" class="reference external">bpo-31900</a>.)

</div>

<div id="notable-changes-in-python-3-6-7" class="section">

## Notable changes in Python 3.6.7<a href="#notable-changes-in-python-3-6-7" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/xml.dom.minidom.html#module-xml.dom.minidom" class="reference internal" title="xml.dom.minidom: Minimal Document Object Model (DOM) implementation."><span class="pre"><code class="sourceCode python">xml.dom.minidom</code></span></a> and <a href="../library/xml.sax.html#module-xml.sax" class="reference internal" title="xml.sax: Package containing SAX2 base classes and convenience functions."><span class="pre"><code class="sourceCode python">xml.sax</code></span></a> modules no longer process external entities by default. See also <a href="https://github.com/python/cpython/issues/61441" class="reference external">gh-61441</a>.

In 3.6.7 the <a href="../library/tokenize.html#module-tokenize" class="reference internal" title="tokenize: Lexical scanner for Python source code."><span class="pre"><code class="sourceCode python">tokenize</code></span></a> module now implicitly emits a <span class="pre">`NEWLINE`</span> token when provided with input that does not have a trailing new line. This behavior now matches what the C tokenizer does internally. (Contributed by Ammar Askar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33899" class="reference external">bpo-33899</a>.)

</div>

<div id="notable-changes-in-python-3-6-10" class="section">

## Notable changes in Python 3.6.10<a href="#notable-changes-in-python-3-6-10" class="headerlink" title="Link to this heading">¶</a>

Due to significant security concerns, the *reuse_address* parameter of <a href="../library/asyncio-eventloop.html#asyncio.loop.create_datagram_endpoint" class="reference internal" title="asyncio.loop.create_datagram_endpoint"><span class="pre"><code class="sourceCode python">asyncio.loop.create_datagram_endpoint()</code></span></a> is no longer supported. This is because of the behavior of the socket option <span class="pre">`SO_REUSEADDR`</span> in UDP. For more details, see the documentation for <span class="pre">`loop.create_datagram_endpoint()`</span>. (Contributed by Kyle Stanley, Antoine Pitrou, and Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37228" class="reference external">bpo-37228</a>.)

</div>

<div id="notable-changes-in-python-3-6-13" class="section">

## Notable changes in Python 3.6.13<a href="#notable-changes-in-python-3-6-13" class="headerlink" title="Link to this heading">¶</a>

Earlier Python versions allowed using both <span class="pre">`;`</span> and <span class="pre">`&`</span> as query parameter separators in <a href="../library/urllib.parse.html#urllib.parse.parse_qs" class="reference internal" title="urllib.parse.parse_qs"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qs()</code></span></a> and <a href="../library/urllib.parse.html#urllib.parse.parse_qsl" class="reference internal" title="urllib.parse.parse_qsl"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qsl()</code></span></a>. Due to security concerns, and to conform with newer W3C recommendations, this has been changed to allow only a single separator key, with <span class="pre">`&`</span> as the default. This change also affects <span class="pre">`cgi.parse()`</span> and <span class="pre">`cgi.parse_multipart()`</span> as they use the affected functions internally. For more details, please see their respective documentation. (Contributed by Adam Goldschmidt, Senthil Kumaran and Ken Jin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42967" class="reference external">bpo-42967</a>.)

</div>

<div id="notable-changes-in-python-3-6-14" class="section">

## Notable changes in Python 3.6.14<a href="#notable-changes-in-python-3-6-14" class="headerlink" title="Link to this heading">¶</a>

A security fix alters the <a href="../library/ftplib.html#ftplib.FTP" class="reference internal" title="ftplib.FTP"><span class="pre"><code class="sourceCode python">ftplib.FTP</code></span></a> behavior to not trust the IPv4 address sent from the remote server when setting up a passive data channel. We reuse the ftp server IP address instead. For unusual code requiring the old behavior, set a <span class="pre">`trust_server_pasv_ipv4_address`</span> attribute on your FTP instance to <span class="pre">`True`</span>. (See <a href="https://github.com/python/cpython/issues/87451" class="reference external">gh-87451</a>)

The presence of newline or tab characters in parts of a URL allows for some forms of attacks. Following the WHATWG specification that updates RFC 3986, ASCII newline <span class="pre">`\n`</span>, <span class="pre">`\r`</span> and tab <span class="pre">`\t`</span> characters are stripped from the URL by the parser <a href="../library/urllib.parse.html#module-urllib.parse" class="reference internal" title="urllib.parse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urllib.parse()</code></span></a> preventing such attacks. The removal characters are controlled by a new module level variable <span class="pre">`urllib.parse._UNSAFE_URL_BYTES_TO_REMOVE`</span>. (See <a href="https://github.com/python/cpython/issues/88048" class="reference external">gh-88048</a>)

</div>

</div>

<div class="clearer">

</div>

</div>
