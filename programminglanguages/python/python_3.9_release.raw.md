<div class="body" role="main">

<div id="what-s-new-in-python-3-9" class="section">

# What’s New In Python 3.9<a href="#what-s-new-in-python-3-9" class="headerlink" title="Link to this heading">¶</a>

Editor<span class="colon">:</span>  
Łukasz Langa

This article explains the new features in Python 3.9, compared to 3.8. Python 3.9 was released on October 5, 2020. For full details, see the <a href="changelog.html#changelog" class="reference internal"><span class="std std-ref">changelog</span></a>.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0596/" class="pep reference external"><strong>PEP 596</strong></a> - Python 3.9 Release Schedule

</div>

<div id="summary-release-highlights" class="section">

## Summary – Release highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

New syntax features:

- <span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0584/" class="pep reference external"><strong>PEP 584</strong></a>, union operators added to <span class="pre">`dict`</span>;

- <span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0585/" class="pep reference external"><strong>PEP 585</strong></a>, type hinting generics in standard collections;

- <span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0614/" class="pep reference external"><strong>PEP 614</strong></a>, relaxed grammar restrictions on decorators.

New built-in features:

- <span id="index-4" class="target"></span><a href="https://peps.python.org/pep-0616/" class="pep reference external"><strong>PEP 616</strong></a>, string methods to remove prefixes and suffixes.

New features in the standard library:

- <span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0593/" class="pep reference external"><strong>PEP 593</strong></a>, flexible function and variable annotations;

- <a href="../library/os.html#os.pidfd_open" class="reference internal" title="os.pidfd_open"><span class="pre"><code class="sourceCode python">os.pidfd_open()</code></span></a> added that allows process management without races and signals.

Interpreter improvements:

- <span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0573/" class="pep reference external"><strong>PEP 573</strong></a>, fast access to module state from methods of C extension types;

- <span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0617/" class="pep reference external"><strong>PEP 617</strong></a>, CPython now uses a new parser based on PEG;

- a number of Python builtins (range, tuple, set, frozenset, list, dict) are now sped up using <span id="index-8" class="target"></span><a href="https://peps.python.org/pep-0590/" class="pep reference external"><strong>PEP 590</strong></a> vectorcall;

- garbage collection does not block on resurrected objects;

- a number of Python modules (<span class="pre">`_abc`</span>, <span class="pre">`audioop`</span>, <span class="pre">`_bz2`</span>, <span class="pre">`_codecs`</span>, <span class="pre">`_contextvars`</span>, <span class="pre">`_crypt`</span>, <span class="pre">`_functools`</span>, <span class="pre">`_json`</span>, <span class="pre">`_locale`</span>, <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a>, <a href="../library/operator.html#module-operator" class="reference internal" title="operator: Functions corresponding to the standard operators."><span class="pre"><code class="sourceCode python">operator</code></span></a>, <a href="../library/resource.html#module-resource" class="reference internal" title="resource: An interface to provide resource usage information on the current process. (Unix)"><span class="pre"><code class="sourceCode python">resource</code></span></a>, <a href="../library/time.html#module-time" class="reference internal" title="time: Time access and conversions."><span class="pre"><code class="sourceCode python">time</code></span></a>, <span class="pre">`_weakref`</span>) now use multiphase initialization as defined by PEP 489;

- a number of standard library modules (<span class="pre">`audioop`</span>, <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a>, <a href="../library/grp.html#module-grp" class="reference internal" title="grp: The group database (getgrnam() and friends). (Unix)"><span class="pre"><code class="sourceCode python">grp</code></span></a>, <span class="pre">`_hashlib`</span>, <a href="../library/pwd.html#module-pwd" class="reference internal" title="pwd: The password database (getpwnam() and friends). (Unix)"><span class="pre"><code class="sourceCode python">pwd</code></span></a>, <span class="pre">`_posixsubprocess`</span>, <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a>, <a href="../library/select.html#module-select" class="reference internal" title="select: Wait for I/O completion on multiple streams."><span class="pre"><code class="sourceCode python">select</code></span></a>, <a href="../library/struct.html#module-struct" class="reference internal" title="struct: Interpret bytes as packed binary data."><span class="pre"><code class="sourceCode python">struct</code></span></a>, <a href="../library/termios.html#module-termios" class="reference internal" title="termios: POSIX style tty control. (Unix)"><span class="pre"><code class="sourceCode python">termios</code></span></a>, <a href="../library/zlib.html#module-zlib" class="reference internal" title="zlib: Low-level interface to compression and decompression routines compatible with gzip."><span class="pre"><code class="sourceCode python">zlib</code></span></a>) are now using the stable ABI defined by PEP 384.

New library modules:

- <span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0615/" class="pep reference external"><strong>PEP 615</strong></a>, the IANA Time Zone Database is now present in the standard library in the <a href="../library/zoneinfo.html#module-zoneinfo" class="reference internal" title="zoneinfo: IANA time zone support"><span class="pre"><code class="sourceCode python">zoneinfo</code></span></a> module;

- an implementation of a topological sort of a graph is now provided in the new <a href="../library/graphlib.html#module-graphlib" class="reference internal" title="graphlib: Functionality to operate with graph-like structures"><span class="pre"><code class="sourceCode python">graphlib</code></span></a> module.

Release process changes:

- <span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0602/" class="pep reference external"><strong>PEP 602</strong></a>, CPython adopts an annual release cycle.

</div>

<div id="you-should-check-for-deprecationwarning-in-your-code" class="section">

## You should check for DeprecationWarning in your code<a href="#you-should-check-for-deprecationwarning-in-your-code" class="headerlink" title="Link to this heading">¶</a>

When Python 2.7 was still supported, a lot of functionality in Python 3 was kept for backward compatibility with Python 2.7. With the end of Python 2 support, these backward compatibility layers have been removed, or will be removed soon. Most of them emitted a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> warning for several years. For example, using <span class="pre">`collections.Mapping`</span> instead of <span class="pre">`collections.abc.Mapping`</span> emits a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> since Python 3.3, released in 2012.

Test your application with the <a href="../using/cmdline.html#cmdoption-W" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-W</code></span></a> <span class="pre">`default`</span> command-line option to see <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> and <a href="../library/exceptions.html#PendingDeprecationWarning" class="reference internal" title="PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a>, or even with <a href="../using/cmdline.html#cmdoption-W" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-W</code></span></a> <span class="pre">`error`</span> to treat them as errors. <a href="../library/warnings.html#warning-filter" class="reference internal"><span class="std std-ref">Warnings Filter</span></a> can be used to ignore warnings from third-party code.

Python 3.9 is the last version providing those Python 2 backward compatibility layers, to give more time to Python projects maintainers to organize the removal of the Python 2 support and add support for Python 3.9.

Aliases to <a href="../library/collections.abc.html#collections-abstract-base-classes" class="reference internal"><span class="std std-ref">Abstract Base Classes</span></a> in the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: Container datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module, like <span class="pre">`collections.Mapping`</span> alias to <a href="../library/collections.abc.html#collections.abc.Mapping" class="reference internal" title="collections.abc.Mapping"><span class="pre"><code class="sourceCode python">collections.abc.Mapping</code></span></a>, are kept for one last release for backward compatibility. They will be removed from Python 3.10.

More generally, try to run your tests in the <a href="../library/devmode.html#devmode" class="reference internal"><span class="std std-ref">Python Development Mode</span></a> which helps to prepare your code to make it compatible with the next Python version.

Note: a number of pre-existing deprecations were removed in this version of Python as well. Consult the <a href="#removed-in-python-39" class="reference internal"><span class="std std-ref">Removed</span></a> section.

</div>

<div id="new-features" class="section">

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="dictionary-merge-update-operators" class="section">

### Dictionary Merge & Update Operators<a href="#dictionary-merge-update-operators" class="headerlink" title="Link to this heading">¶</a>

Merge (<span class="pre">`|`</span>) and update (<span class="pre">`|=`</span>) operators have been added to the built-in <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> class. Those complement the existing <span class="pre">`dict.update`</span> and <span class="pre">`{**d1,`</span>` `<span class="pre">`**d2}`</span> methods of merging dictionaries.

Example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> x = {"key1": "value1 from x", "key2": "value2 from x"}
    >>> y = {"key2": "value2 from y", "key3": "value3 from y"}
    >>> x | y
    {'key1': 'value1 from x', 'key2': 'value2 from y', 'key3': 'value3 from y'}
    >>> y | x
    {'key2': 'value2 from x', 'key3': 'value3 from y', 'key1': 'value1 from x'}

</div>

</div>

See <span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0584/" class="pep reference external"><strong>PEP 584</strong></a> for a full description. (Contributed by Brandt Bucher in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36144" class="reference external">bpo-36144</a>.)

</div>

<div id="new-string-methods-to-remove-prefixes-and-suffixes" class="section">

### New String Methods to Remove Prefixes and Suffixes<a href="#new-string-methods-to-remove-prefixes-and-suffixes" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/stdtypes.html#str.removeprefix" class="reference internal" title="str.removeprefix"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.removeprefix(prefix)</code></span></a> and <a href="../library/stdtypes.html#str.removesuffix" class="reference internal" title="str.removesuffix"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.removesuffix(suffix)</code></span></a> have been added to easily remove an unneeded prefix or a suffix from a string. Corresponding <span class="pre">`bytes`</span>, <span class="pre">`bytearray`</span>, and <span class="pre">`collections.UserString`</span> methods have also been added. See <span id="index-12" class="target"></span><a href="https://peps.python.org/pep-0616/" class="pep reference external"><strong>PEP 616</strong></a> for a full description. (Contributed by Dennis Sweeney in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39939" class="reference external">bpo-39939</a>.)

</div>

<div id="type-hinting-generics-in-standard-collections" class="section">

### Type Hinting Generics in Standard Collections<a href="#type-hinting-generics-in-standard-collections" class="headerlink" title="Link to this heading">¶</a>

In type annotations you can now use built-in collection types such as <span class="pre">`list`</span> and <span class="pre">`dict`</span> as generic types instead of importing the corresponding capitalized types (e.g. <span class="pre">`List`</span> or <span class="pre">`Dict`</span>) from <span class="pre">`typing`</span>. Some other types in the standard library are also now generic, for example <span class="pre">`queue.Queue`</span>.

Example:

<div class="highlight-python notranslate">

<div class="highlight">

    def greet_all(names: list[str]) -> None:
        for name in names:
            print("Hello", name)

</div>

</div>

See <span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0585/" class="pep reference external"><strong>PEP 585</strong></a> for more details. (Contributed by Guido van Rossum, Ethan Smith, and Batuhan Taşkaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39481" class="reference external">bpo-39481</a>.)

</div>

<div id="new-parser" class="section">

### New Parser<a href="#new-parser" class="headerlink" title="Link to this heading">¶</a>

Python 3.9 uses a new parser, based on <a href="https://en.wikipedia.org/wiki/Parsing_expression_grammar" class="reference external">PEG</a> instead of <a href="https://en.wikipedia.org/wiki/LL_parser" class="reference external">LL(1)</a>. The new parser’s performance is roughly comparable to that of the old parser, but the PEG formalism is more flexible than LL(1) when it comes to designing new language features. We’ll start using this flexibility in Python 3.10 and later.

The <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> module uses the new parser and produces the same AST as the old parser.

In Python 3.10, the old parser will be deleted and so will all functionality that depends on it (primarily the <span class="pre">`parser`</span> module, which has long been deprecated). In Python 3.9 *only*, you can switch back to the LL(1) parser using a command line switch (<span class="pre">`-X`</span>` `<span class="pre">`oldparser`</span>) or an environment variable (<span class="pre">`PYTHONOLDPARSER=1`</span>).

See <span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0617/" class="pep reference external"><strong>PEP 617</strong></a> for more details. (Contributed by Guido van Rossum, Pablo Galindo and Lysandros Nikolaou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40334" class="reference external">bpo-40334</a>.)

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/functions.html#import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> now raises <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> instead of <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a>, which used to occur when a relative import went past its top-level package. (Contributed by Ngalim Siregar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37444" class="reference external">bpo-37444</a>.)

- Python now gets the absolute path of the script filename specified on the command line (ex: <span class="pre">`python3`</span>` `<span class="pre">`script.py`</span>): the <span class="pre">`__file__`</span> attribute of the <a href="../library/__main__.html#module-__main__" class="reference internal" title="__main__: The environment where top-level code is run. Covers command-line interfaces, import-time behavior, and ``__name__ == &#39;__main__&#39;``."><span class="pre"><code class="sourceCode python">__main__</code></span></a> module became an absolute path, rather than a relative path. These paths now remain valid after the current directory is changed by <a href="../library/os.html#os.chdir" class="reference internal" title="os.chdir"><span class="pre"><code class="sourceCode python">os.chdir()</code></span></a>. As a side effect, the traceback also displays the absolute path for <a href="../library/__main__.html#module-__main__" class="reference internal" title="__main__: The environment where top-level code is run. Covers command-line interfaces, import-time behavior, and ``__name__ == &#39;__main__&#39;``."><span class="pre"><code class="sourceCode python">__main__</code></span></a> module frames in this case. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20443" class="reference external">bpo-20443</a>.)

- In the <a href="../library/devmode.html#devmode" class="reference internal"><span class="std std-ref">Python Development Mode</span></a> and in <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">debug build</span></a>, the *encoding* and *errors* arguments are now checked for string encoding and decoding operations. Examples: <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a>, <a href="../library/stdtypes.html#str.encode" class="reference internal" title="str.encode"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.encode()</code></span></a> and <a href="../library/stdtypes.html#bytes.decode" class="reference internal" title="bytes.decode"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span>.decode()</code></span></a>.

  By default, for best performance, the *errors* argument is only checked at the first encoding/decoding error and the *encoding* argument is sometimes ignored for empty strings. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37388" class="reference external">bpo-37388</a>.)

- <span class="pre">`"".replace("",`</span>` `<span class="pre">`s,`</span>` `<span class="pre">`n)`</span> now returns <span class="pre">`s`</span> instead of an empty string for all non-zero <span class="pre">`n`</span>. It is now consistent with <span class="pre">`"".replace("",`</span>` `<span class="pre">`s)`</span>. There are similar changes for <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> objects. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28029" class="reference external">bpo-28029</a>.)

- Any valid expression can now be used as a <a href="../glossary.html#term-decorator" class="reference internal"><span class="xref std std-term">decorator</span></a>. Previously, the grammar was much more restrictive. See <span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0614/" class="pep reference external"><strong>PEP 614</strong></a> for details. (Contributed by Brandt Bucher in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39702" class="reference external">bpo-39702</a>.)

- Improved help for the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module. Docstrings are now shown for all special forms and special generic aliases (like <span class="pre">`Union`</span> and <span class="pre">`List`</span>). Using <a href="../library/functions.html#help" class="reference internal" title="help"><span class="pre"><code class="sourceCode python"><span class="bu">help</span>()</code></span></a> with generic alias like <span class="pre">`List[int]`</span> will show the help for the correspondent concrete type (<span class="pre">`list`</span> in this case). (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40257" class="reference external">bpo-40257</a>.)

- Parallel running of <a href="../reference/expressions.html#agen.aclose" class="reference internal" title="agen.aclose"><span class="pre"><code class="sourceCode python">aclose()</code></span></a> / <a href="../reference/expressions.html#agen.asend" class="reference internal" title="agen.asend"><span class="pre"><code class="sourceCode python">asend()</code></span></a> / <a href="../reference/expressions.html#agen.athrow" class="reference internal" title="agen.athrow"><span class="pre"><code class="sourceCode python">athrow()</code></span></a> is now prohibited, and <span class="pre">`ag_running`</span> now reflects the actual running status of the async generator. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30773" class="reference external">bpo-30773</a>.)

- Unexpected errors in calling the <span class="pre">`__iter__`</span> method are no longer masked by <span class="pre">`TypeError`</span> in the <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> operator and functions <a href="../library/operator.html#operator.contains" class="reference internal" title="operator.contains"><span class="pre"><code class="sourceCode python">contains()</code></span></a>, <a href="../library/operator.html#operator.indexOf" class="reference internal" title="operator.indexOf"><span class="pre"><code class="sourceCode python">indexOf()</code></span></a> and <a href="../library/operator.html#operator.countOf" class="reference internal" title="operator.countOf"><span class="pre"><code class="sourceCode python">countOf()</code></span></a> of the <a href="../library/operator.html#module-operator" class="reference internal" title="operator: Functions corresponding to the standard operators."><span class="pre"><code class="sourceCode python">operator</code></span></a> module. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40824" class="reference external">bpo-40824</a>.)

- Unparenthesized lambda expressions can no longer be the expression part in an <span class="pre">`if`</span> clause in comprehensions and generator expressions. See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41848" class="reference external">bpo-41848</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43755" class="reference external">bpo-43755</a> for details.

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="zoneinfo" class="section">

### zoneinfo<a href="#zoneinfo" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/zoneinfo.html#module-zoneinfo" class="reference internal" title="zoneinfo: IANA time zone support"><span class="pre"><code class="sourceCode python">zoneinfo</code></span></a> module brings support for the IANA time zone database to the standard library. It adds <a href="../library/zoneinfo.html#zoneinfo.ZoneInfo" class="reference internal" title="zoneinfo.ZoneInfo"><span class="pre"><code class="sourceCode python">zoneinfo.ZoneInfo</code></span></a>, a concrete <a href="../library/datetime.html#datetime.tzinfo" class="reference internal" title="datetime.tzinfo"><span class="pre"><code class="sourceCode python">datetime.tzinfo</code></span></a> implementation backed by the system’s time zone data.

Example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from zoneinfo import ZoneInfo
    >>> from datetime import datetime, timedelta

    >>> # Daylight saving time
    >>> dt = datetime(2020, 10, 31, 12, tzinfo=ZoneInfo("America/Los_Angeles"))
    >>> print(dt)
    2020-10-31 12:00:00-07:00
    >>> dt.tzname()
    'PDT'

    >>> # Standard time
    >>> dt += timedelta(days=7)
    >>> print(dt)
    2020-11-07 12:00:00-08:00
    >>> print(dt.tzname())
    PST

</div>

</div>

As a fall-back source of data for platforms that don’t ship the IANA database, the <a href="https://pypi.org/project/tzdata/" class="extlink-pypi reference external">tzdata</a> module was released as a first-party package – distributed via PyPI and maintained by the CPython core team.

<div class="admonition seealso">

See also

<span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0615/" class="pep reference external"><strong>PEP 615</strong></a> – Support for the IANA Time Zone Database in the Standard Library  
PEP written and implemented by Paul Ganssle

</div>

</div>

<div id="graphlib" class="section">

### graphlib<a href="#graphlib" class="headerlink" title="Link to this heading">¶</a>

A new module, <a href="../library/graphlib.html#module-graphlib" class="reference internal" title="graphlib: Functionality to operate with graph-like structures"><span class="pre"><code class="sourceCode python">graphlib</code></span></a>, was added that contains the <a href="../library/graphlib.html#graphlib.TopologicalSorter" class="reference internal" title="graphlib.TopologicalSorter"><span class="pre"><code class="sourceCode python">graphlib.TopologicalSorter</code></span></a> class to offer functionality to perform topological sorting of graphs. (Contributed by Pablo Galindo, Tim Peters and Larry Hastings in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17005" class="reference external">bpo-17005</a>.)

</div>

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="ast" class="section">

### ast<a href="#ast" class="headerlink" title="Link to this heading">¶</a>

Added the *indent* option to <a href="../library/ast.html#ast.dump" class="reference internal" title="ast.dump"><span class="pre"><code class="sourceCode python">dump()</code></span></a> which allows it to produce a multiline indented output. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37995" class="reference external">bpo-37995</a>.)

Added <a href="../library/ast.html#ast.unparse" class="reference internal" title="ast.unparse"><span class="pre"><code class="sourceCode python">ast.unparse()</code></span></a> as a function in the <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> module that can be used to unparse an <a href="../library/ast.html#ast.AST" class="reference internal" title="ast.AST"><span class="pre"><code class="sourceCode python">ast.AST</code></span></a> object and produce a string with code that would produce an equivalent <a href="../library/ast.html#ast.AST" class="reference internal" title="ast.AST"><span class="pre"><code class="sourceCode python">ast.AST</code></span></a> object when parsed. (Contributed by Pablo Galindo and Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38870" class="reference external">bpo-38870</a>.)

Added docstrings to AST nodes that contains the ASDL signature used to construct that node. (Contributed by Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39638" class="reference external">bpo-39638</a>.)

</div>

<div id="asyncio" class="section">

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

Due to significant security concerns, the *reuse_address* parameter of <a href="../library/asyncio-eventloop.html#asyncio.loop.create_datagram_endpoint" class="reference internal" title="asyncio.loop.create_datagram_endpoint"><span class="pre"><code class="sourceCode python">asyncio.loop.create_datagram_endpoint()</code></span></a> is no longer supported. This is because of the behavior of the socket option <span class="pre">`SO_REUSEADDR`</span> in UDP. For more details, see the documentation for <span class="pre">`loop.create_datagram_endpoint()`</span>. (Contributed by Kyle Stanley, Antoine Pitrou, and Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37228" class="reference external">bpo-37228</a>.)

Added a new <a href="../glossary.html#term-coroutine" class="reference internal"><span class="xref std std-term">coroutine</span></a> <a href="../library/asyncio-eventloop.html#asyncio.loop.shutdown_default_executor" class="reference internal" title="asyncio.loop.shutdown_default_executor"><span class="pre"><code class="sourceCode python">shutdown_default_executor()</code></span></a> that schedules a shutdown for the default executor that waits on the <a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">ThreadPoolExecutor</code></span></a> to finish closing. Also, <a href="../library/asyncio-runner.html#asyncio.run" class="reference internal" title="asyncio.run"><span class="pre"><code class="sourceCode python">asyncio.run()</code></span></a> has been updated to use the new <a href="../glossary.html#term-coroutine" class="reference internal"><span class="xref std std-term">coroutine</span></a>. (Contributed by Kyle Stanley in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34037" class="reference external">bpo-34037</a>.)

Added <a href="../library/asyncio-policy.html#asyncio.PidfdChildWatcher" class="reference internal" title="asyncio.PidfdChildWatcher"><span class="pre"><code class="sourceCode python">asyncio.PidfdChildWatcher</code></span></a>, a Linux-specific child watcher implementation that polls process file descriptors. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38692" class="reference external">bpo-38692</a>)

Added a new <a href="../glossary.html#term-coroutine" class="reference internal"><span class="xref std std-term">coroutine</span></a> <a href="../library/asyncio-task.html#asyncio.to_thread" class="reference internal" title="asyncio.to_thread"><span class="pre"><code class="sourceCode python">asyncio.to_thread()</code></span></a>. It is mainly used for running IO-bound functions in a separate thread to avoid blocking the event loop, and essentially works as a high-level version of <a href="../library/asyncio-eventloop.html#asyncio.loop.run_in_executor" class="reference internal" title="asyncio.loop.run_in_executor"><span class="pre"><code class="sourceCode python">run_in_executor()</code></span></a> that can directly take keyword arguments. (Contributed by Kyle Stanley and Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32309" class="reference external">bpo-32309</a>.)

When cancelling the task due to a timeout, <a href="../library/asyncio-task.html#asyncio.wait_for" class="reference internal" title="asyncio.wait_for"><span class="pre"><code class="sourceCode python">asyncio.wait_for()</code></span></a> will now wait until the cancellation is complete also in the case when *timeout* is \<= 0, like it does with positive timeouts. (Contributed by Elvis Pranskevichus in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32751" class="reference external">bpo-32751</a>.)

<a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> now raises <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> when calling incompatible methods with an <a href="../library/ssl.html#ssl.SSLSocket" class="reference internal" title="ssl.SSLSocket"><span class="pre"><code class="sourceCode python">ssl.SSLSocket</code></span></a> socket. (Contributed by Ido Michael in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37404" class="reference external">bpo-37404</a>.)

</div>

<div id="compileall" class="section">

### compileall<a href="#compileall" class="headerlink" title="Link to this heading">¶</a>

Added new possibility to use hardlinks for duplicated <span class="pre">`.pyc`</span> files: *hardlink_dupes* parameter and –hardlink-dupes command line option. (Contributed by Lumír ‘Frenzy’ Balhar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40495" class="reference external">bpo-40495</a>.)

Added new options for path manipulation in resulting <span class="pre">`.pyc`</span> files: *stripdir*, *prependdir*, *limit_sl_dest* parameters and -s, -p, -e command line options. Added the possibility to specify the option for an optimization level multiple times. (Contributed by Lumír ‘Frenzy’ Balhar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38112" class="reference external">bpo-38112</a>.)

</div>

<div id="concurrent-futures" class="section">

### concurrent.futures<a href="#concurrent-futures" class="headerlink" title="Link to this heading">¶</a>

Added a new *cancel_futures* parameter to <a href="../library/concurrent.futures.html#concurrent.futures.Executor.shutdown" class="reference internal" title="concurrent.futures.Executor.shutdown"><span class="pre"><code class="sourceCode python">concurrent.futures.Executor.shutdown()</code></span></a> that cancels all pending futures which have not started running, instead of waiting for them to complete before shutting down the executor. (Contributed by Kyle Stanley in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39349" class="reference external">bpo-39349</a>.)

Removed daemon threads from <a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">ThreadPoolExecutor</code></span></a> and <a href="../library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor" class="reference internal" title="concurrent.futures.ProcessPoolExecutor"><span class="pre"><code class="sourceCode python">ProcessPoolExecutor</code></span></a>. This improves compatibility with subinterpreters and predictability in their shutdown processes. (Contributed by Kyle Stanley in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39812" class="reference external">bpo-39812</a>.)

Workers in <a href="../library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor" class="reference internal" title="concurrent.futures.ProcessPoolExecutor"><span class="pre"><code class="sourceCode python">ProcessPoolExecutor</code></span></a> are now spawned on demand, only when there are no available idle workers to reuse. This optimizes startup overhead and reduces the amount of lost CPU time to idle workers. (Contributed by Kyle Stanley in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39207" class="reference external">bpo-39207</a>.)

</div>

<div id="curses" class="section">

### curses<a href="#curses" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/curses.html#curses.get_escdelay" class="reference internal" title="curses.get_escdelay"><span class="pre"><code class="sourceCode python">curses.get_escdelay()</code></span></a>, <a href="../library/curses.html#curses.set_escdelay" class="reference internal" title="curses.set_escdelay"><span class="pre"><code class="sourceCode python">curses.set_escdelay()</code></span></a>, <a href="../library/curses.html#curses.get_tabsize" class="reference internal" title="curses.get_tabsize"><span class="pre"><code class="sourceCode python">curses.get_tabsize()</code></span></a>, and <a href="../library/curses.html#curses.set_tabsize" class="reference internal" title="curses.set_tabsize"><span class="pre"><code class="sourceCode python">curses.set_tabsize()</code></span></a> functions. (Contributed by Anthony Sottile in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38312" class="reference external">bpo-38312</a>.)

</div>

<div id="datetime" class="section">

### datetime<a href="#datetime" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/datetime.html#datetime.date.isocalendar" class="reference internal" title="datetime.date.isocalendar"><span class="pre"><code class="sourceCode python">isocalendar()</code></span></a> of <a href="../library/datetime.html#datetime.date" class="reference internal" title="datetime.date"><span class="pre"><code class="sourceCode python">datetime.date</code></span></a> and <a href="../library/datetime.html#datetime.datetime.isocalendar" class="reference internal" title="datetime.datetime.isocalendar"><span class="pre"><code class="sourceCode python">isocalendar()</code></span></a> of <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime.datetime</code></span></a> methods now returns a <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">namedtuple()</code></span></a> instead of a <a href="../library/stdtypes.html#tuple" class="reference internal" title="tuple"><span class="pre"><code class="sourceCode python"><span class="bu">tuple</span></code></span></a>. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24416" class="reference external">bpo-24416</a>.)

</div>

<div id="distutils" class="section">

### distutils<a href="#distutils" class="headerlink" title="Link to this heading">¶</a>

The **upload** command now creates SHA2-256 and Blake2b-256 hash digests. It skips MD5 on platforms that block MD5 digest. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40698" class="reference external">bpo-40698</a>.)

</div>

<div id="fcntl" class="section">

### fcntl<a href="#fcntl" class="headerlink" title="Link to this heading">¶</a>

Added constants <span class="pre">`fcntl.F_OFD_GETLK`</span>, <span class="pre">`fcntl.F_OFD_SETLK`</span> and <span class="pre">`fcntl.F_OFD_SETLKW`</span>. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38602" class="reference external">bpo-38602</a>.)

</div>

<div id="ftplib" class="section">

### ftplib<a href="#ftplib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/ftplib.html#ftplib.FTP" class="reference internal" title="ftplib.FTP"><span class="pre"><code class="sourceCode python">FTP</code></span></a> and <a href="../library/ftplib.html#ftplib.FTP_TLS" class="reference internal" title="ftplib.FTP_TLS"><span class="pre"><code class="sourceCode python">FTP_TLS</code></span></a> now raise a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the given timeout for their constructor is zero to prevent the creation of a non-blocking socket. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39259" class="reference external">bpo-39259</a>.)

</div>

<div id="gc" class="section">

### gc<a href="#gc" class="headerlink" title="Link to this heading">¶</a>

When the garbage collector makes a collection in which some objects resurrect (they are reachable from outside the isolated cycles after the finalizers have been executed), do not block the collection of all objects that are still unreachable. (Contributed by Pablo Galindo and Tim Peters in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38379" class="reference external">bpo-38379</a>.)

Added a new function <a href="../library/gc.html#gc.is_finalized" class="reference internal" title="gc.is_finalized"><span class="pre"><code class="sourceCode python">gc.is_finalized()</code></span></a> to check if an object has been finalized by the garbage collector. (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39322" class="reference external">bpo-39322</a>.)

</div>

<div id="hashlib" class="section">

### hashlib<a href="#hashlib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module can now use SHA3 hashes and SHAKE XOF from OpenSSL when available. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37630" class="reference external">bpo-37630</a>.)

Builtin hash modules can now be disabled with <span class="pre">`./configure`</span>` `<span class="pre">`--without-builtin-hashlib-hashes`</span> or selectively enabled with e.g. <span class="pre">`./configure`</span>` `<span class="pre">`--with-builtin-hashlib-hashes=sha3,blake2`</span> to force use of OpenSSL based implementation. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40479" class="reference external">bpo-40479</a>)

</div>

<div id="http" class="section">

### http<a href="#http" class="headerlink" title="Link to this heading">¶</a>

HTTP status codes <span class="pre">`103`</span>` `<span class="pre">`EARLY_HINTS`</span>, <span class="pre">`418`</span>` `<span class="pre">`IM_A_TEAPOT`</span> and <span class="pre">`425`</span>` `<span class="pre">`TOO_EARLY`</span> are added to <a href="../library/http.html#http.HTTPStatus" class="reference internal" title="http.HTTPStatus"><span class="pre"><code class="sourceCode python">http.HTTPStatus</code></span></a>. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39509" class="reference external">bpo-39509</a> and Ross Rhodes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39507" class="reference external">bpo-39507</a>.)

</div>

<div id="idle-and-idlelib" class="section">

### IDLE and idlelib<a href="#idle-and-idlelib" class="headerlink" title="Link to this heading">¶</a>

Added option to toggle cursor blink off. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4603" class="reference external">bpo-4603</a>.)

Escape key now closes IDLE completion windows. (Contributed by Johnny Najera in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38944" class="reference external">bpo-38944</a>.)

Added keywords to module name completion list. (Contributed by Terry J. Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37765" class="reference external">bpo-37765</a>.)

New in 3.9 maintenance releases

Make IDLE invoke <a href="../library/sys.html#sys.excepthook" class="reference internal" title="sys.excepthook"><span class="pre"><code class="sourceCode python">sys.excepthook()</code></span></a> (when started without ‘-n’). User hooks were previously ignored. (Contributed by Ken Hilton in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43008" class="reference external">bpo-43008</a>.)

The changes above have been backported to 3.8 maintenance releases.

Rearrange the settings dialog. Split the General tab into Windows and Shell/Ed tabs. Move help sources, which extend the Help menu, to the Extensions tab. Make space for new options and shorten the dialog. The latter makes the dialog better fit small screens. (Contributed by Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40468" class="reference external">bpo-40468</a>.) Move the indent space setting from the Font tab to the new Windows tab. (Contributed by Mark Roseman and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33962" class="reference external">bpo-33962</a>.)

Apply syntax highlighting to <span class="pre">`.pyi`</span> files. (Contributed by Alex Waygood and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45447" class="reference external">bpo-45447</a>.)

</div>

<div id="imaplib" class="section">

### imaplib<a href="#imaplib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/imaplib.html#imaplib.IMAP4" class="reference internal" title="imaplib.IMAP4"><span class="pre"><code class="sourceCode python">IMAP4</code></span></a> and <a href="../library/imaplib.html#imaplib.IMAP4_SSL" class="reference internal" title="imaplib.IMAP4_SSL"><span class="pre"><code class="sourceCode python">IMAP4_SSL</code></span></a> now have an optional *timeout* parameter for their constructors. Also, the <a href="../library/imaplib.html#imaplib.IMAP4.open" class="reference internal" title="imaplib.IMAP4.open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> method now has an optional *timeout* parameter with this change. The overridden methods of <a href="../library/imaplib.html#imaplib.IMAP4_SSL" class="reference internal" title="imaplib.IMAP4_SSL"><span class="pre"><code class="sourceCode python">IMAP4_SSL</code></span></a> and <a href="../library/imaplib.html#imaplib.IMAP4_stream" class="reference internal" title="imaplib.IMAP4_stream"><span class="pre"><code class="sourceCode python">IMAP4_stream</code></span></a> were applied to this change. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38615" class="reference external">bpo-38615</a>.)

<a href="../library/imaplib.html#imaplib.IMAP4.unselect" class="reference internal" title="imaplib.IMAP4.unselect"><span class="pre"><code class="sourceCode python">imaplib.IMAP4.unselect()</code></span></a> is added. <a href="../library/imaplib.html#imaplib.IMAP4.unselect" class="reference internal" title="imaplib.IMAP4.unselect"><span class="pre"><code class="sourceCode python">imaplib.IMAP4.unselect()</code></span></a> frees server’s resources associated with the selected mailbox and returns the server to the authenticated state. This command performs the same actions as <a href="../library/imaplib.html#imaplib.IMAP4.close" class="reference internal" title="imaplib.IMAP4.close"><span class="pre"><code class="sourceCode python">imaplib.IMAP4.close()</code></span></a>, except that no messages are permanently removed from the currently selected mailbox. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40375" class="reference external">bpo-40375</a>.)

</div>

<div id="importlib" class="section">

### importlib<a href="#importlib" class="headerlink" title="Link to this heading">¶</a>

To improve consistency with import statements, <a href="../library/importlib.html#importlib.util.resolve_name" class="reference internal" title="importlib.util.resolve_name"><span class="pre"><code class="sourceCode python">importlib.util.resolve_name()</code></span></a> now raises <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> instead of <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> for invalid relative import attempts. (Contributed by Ngalim Siregar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37444" class="reference external">bpo-37444</a>.)

Import loaders which publish immutable module objects can now publish immutable packages in addition to individual modules. (Contributed by Dino Viehland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39336" class="reference external">bpo-39336</a>.)

Added <a href="../library/importlib.resources.html#importlib.resources.files" class="reference internal" title="importlib.resources.files"><span class="pre"><code class="sourceCode python">importlib.resources.files()</code></span></a> function with support for subdirectories in package data, matching backport in <span class="pre">`importlib_resources`</span> version 1.5. (Contributed by Jason R. Coombs in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39791" class="reference external">bpo-39791</a>.)

Refreshed <span class="pre">`importlib.metadata`</span> from <span class="pre">`importlib_metadata`</span> version 1.6.1.

</div>

<div id="inspect" class="section">

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/inspect.html#inspect.BoundArguments.arguments" class="reference internal" title="inspect.BoundArguments.arguments"><span class="pre"><code class="sourceCode python">inspect.BoundArguments.arguments</code></span></a> is changed from <span class="pre">`OrderedDict`</span> to regular dict. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36350" class="reference external">bpo-36350</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39775" class="reference external">bpo-39775</a>.)

</div>

<div id="ipaddress" class="section">

### ipaddress<a href="#ipaddress" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> now supports IPv6 Scoped Addresses (IPv6 address with suffix <span class="pre">`%<scope_id>`</span>).

Scoped IPv6 addresses can be parsed using <a href="../library/ipaddress.html#ipaddress.IPv6Address" class="reference internal" title="ipaddress.IPv6Address"><span class="pre"><code class="sourceCode python">ipaddress.IPv6Address</code></span></a>. If present, scope zone ID is available through the <a href="../library/ipaddress.html#ipaddress.IPv6Address.scope_id" class="reference internal" title="ipaddress.IPv6Address.scope_id"><span class="pre"><code class="sourceCode python">scope_id</code></span></a> attribute. (Contributed by Oleksandr Pavliuk in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34788" class="reference external">bpo-34788</a>.)

Starting with Python 3.9.5 the <a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> module no longer accepts any leading zeros in IPv4 address strings. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36384" class="reference external">bpo-36384</a>).

</div>

<div id="math" class="section">

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

Expanded the <a href="../library/math.html#math.gcd" class="reference internal" title="math.gcd"><span class="pre"><code class="sourceCode python">math.gcd()</code></span></a> function to handle multiple arguments. Formerly, it only supported two arguments. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39648" class="reference external">bpo-39648</a>.)

Added <a href="../library/math.html#math.lcm" class="reference internal" title="math.lcm"><span class="pre"><code class="sourceCode python">math.lcm()</code></span></a>: return the least common multiple of specified arguments. (Contributed by Mark Dickinson, Ananthakrishnan and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39479" class="reference external">bpo-39479</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39648" class="reference external">bpo-39648</a>.)

Added <a href="../library/math.html#math.nextafter" class="reference internal" title="math.nextafter"><span class="pre"><code class="sourceCode python">math.nextafter()</code></span></a>: return the next floating-point value after *x* towards *y*. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39288" class="reference external">bpo-39288</a>.)

Added <a href="../library/math.html#math.ulp" class="reference internal" title="math.ulp"><span class="pre"><code class="sourceCode python">math.ulp()</code></span></a>: return the value of the least significant bit of a float. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39310" class="reference external">bpo-39310</a>.)

</div>

<div id="multiprocessing" class="section">

### multiprocessing<a href="#multiprocessing" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/multiprocessing.html#multiprocessing.SimpleQueue" class="reference internal" title="multiprocessing.SimpleQueue"><span class="pre"><code class="sourceCode python">multiprocessing.SimpleQueue</code></span></a> class has a new <a href="../library/multiprocessing.html#multiprocessing.SimpleQueue.close" class="reference internal" title="multiprocessing.SimpleQueue.close"><span class="pre"><code class="sourceCode python">close()</code></span></a> method to explicitly close the queue. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30966" class="reference external">bpo-30966</a>.)

</div>

<div id="nntplib" class="section">

### nntplib<a href="#nntplib" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`NNTP`</span> and <span class="pre">`NNTP_SSL`</span> now raise a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the given timeout for their constructor is zero to prevent the creation of a non-blocking socket. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39259" class="reference external">bpo-39259</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/os.html#os.CLD_KILLED" class="reference internal" title="os.CLD_KILLED"><span class="pre"><code class="sourceCode python">CLD_KILLED</code></span></a> and <a href="../library/os.html#os.CLD_STOPPED" class="reference internal" title="os.CLD_STOPPED"><span class="pre"><code class="sourceCode python">CLD_STOPPED</code></span></a> for <span class="pre">`si_code`</span>. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38493" class="reference external">bpo-38493</a>.)

Exposed the Linux-specific <a href="../library/os.html#os.pidfd_open" class="reference internal" title="os.pidfd_open"><span class="pre"><code class="sourceCode python">os.pidfd_open()</code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38692" class="reference external">bpo-38692</a>) and <a href="../library/os.html#os.P_PIDFD" class="reference internal" title="os.P_PIDFD"><span class="pre"><code class="sourceCode python">os.P_PIDFD</code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38713" class="reference external">bpo-38713</a>) for process management with file descriptors.

The <a href="../library/os.html#os.unsetenv" class="reference internal" title="os.unsetenv"><span class="pre"><code class="sourceCode python">os.unsetenv()</code></span></a> function is now also available on Windows. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39413" class="reference external">bpo-39413</a>.)

The <a href="../library/os.html#os.putenv" class="reference internal" title="os.putenv"><span class="pre"><code class="sourceCode python">os.putenv()</code></span></a> and <a href="../library/os.html#os.unsetenv" class="reference internal" title="os.unsetenv"><span class="pre"><code class="sourceCode python">os.unsetenv()</code></span></a> functions are now always available. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39395" class="reference external">bpo-39395</a>.)

Added <a href="../library/os.html#os.waitstatus_to_exitcode" class="reference internal" title="os.waitstatus_to_exitcode"><span class="pre"><code class="sourceCode python">os.waitstatus_to_exitcode()</code></span></a> function: convert a wait status to an exit code. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40094" class="reference external">bpo-40094</a>.)

</div>

<div id="pathlib" class="section">

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/pathlib.html#pathlib.Path.readlink" class="reference internal" title="pathlib.Path.readlink"><span class="pre"><code class="sourceCode python">pathlib.Path.readlink()</code></span></a> which acts similarly to <a href="../library/os.html#os.readlink" class="reference internal" title="os.readlink"><span class="pre"><code class="sourceCode python">os.readlink()</code></span></a>. (Contributed by Girts Folkmanis in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30618" class="reference external">bpo-30618</a>)

</div>

<div id="pdb" class="section">

### pdb<a href="#pdb" class="headerlink" title="Link to this heading">¶</a>

On Windows now <a href="../library/pdb.html#pdb.Pdb" class="reference internal" title="pdb.Pdb"><span class="pre"><code class="sourceCode python">Pdb</code></span></a> supports <span class="pre">`~/.pdbrc`</span>. (Contributed by Tim Hopper and Dan Lidral-Porter in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20523" class="reference external">bpo-20523</a>.)

</div>

<div id="poplib" class="section">

### poplib<a href="#poplib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/poplib.html#poplib.POP3" class="reference internal" title="poplib.POP3"><span class="pre"><code class="sourceCode python">POP3</code></span></a> and <a href="../library/poplib.html#poplib.POP3_SSL" class="reference internal" title="poplib.POP3_SSL"><span class="pre"><code class="sourceCode python">POP3_SSL</code></span></a> now raise a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the given timeout for their constructor is zero to prevent the creation of a non-blocking socket. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39259" class="reference external">bpo-39259</a>.)

</div>

<div id="pprint" class="section">

### pprint<a href="#pprint" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pprint.html#module-pprint" class="reference internal" title="pprint: Data pretty printer."><span class="pre"><code class="sourceCode python">pprint</code></span></a> can now pretty-print <a href="../library/types.html#types.SimpleNamespace" class="reference internal" title="types.SimpleNamespace"><span class="pre"><code class="sourceCode python">types.SimpleNamespace</code></span></a>. (Contributed by Carl Bordum Hansen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37376" class="reference external">bpo-37376</a>.)

</div>

<div id="pydoc" class="section">

### pydoc<a href="#pydoc" class="headerlink" title="Link to this heading">¶</a>

The documentation string is now shown not only for class, function, method etc, but for any object that has its own <a href="../library/stdtypes.html#definition.__doc__" class="reference internal" title="definition.__doc__"><span class="pre"><code class="sourceCode python">__doc__</code></span></a> attribute. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40257" class="reference external">bpo-40257</a>.)

</div>

<div id="random" class="section">

### random<a href="#random" class="headerlink" title="Link to this heading">¶</a>

Added a new <a href="../library/random.html#random.Random.randbytes" class="reference internal" title="random.Random.randbytes"><span class="pre"><code class="sourceCode python">random.Random.randbytes()</code></span></a> method: generate random bytes. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40286" class="reference external">bpo-40286</a>.)

</div>

<div id="signal" class="section">

### signal<a href="#signal" class="headerlink" title="Link to this heading">¶</a>

Exposed the Linux-specific <a href="../library/signal.html#signal.pidfd_send_signal" class="reference internal" title="signal.pidfd_send_signal"><span class="pre"><code class="sourceCode python">signal.pidfd_send_signal()</code></span></a> for sending to signals to a process using a file descriptor instead of a pid. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38712" class="reference external">bpo-38712</a>)

</div>

<div id="smtplib" class="section">

### smtplib<a href="#smtplib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/smtplib.html#smtplib.SMTP" class="reference internal" title="smtplib.SMTP"><span class="pre"><code class="sourceCode python">SMTP</code></span></a> and <a href="../library/smtplib.html#smtplib.SMTP_SSL" class="reference internal" title="smtplib.SMTP_SSL"><span class="pre"><code class="sourceCode python">SMTP_SSL</code></span></a> now raise a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the given timeout for their constructor is zero to prevent the creation of a non-blocking socket. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39259" class="reference external">bpo-39259</a>.)

<a href="../library/smtplib.html#smtplib.LMTP" class="reference internal" title="smtplib.LMTP"><span class="pre"><code class="sourceCode python">LMTP</code></span></a> constructor now has an optional *timeout* parameter. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39329" class="reference external">bpo-39329</a>.)

</div>

<div id="socket" class="section">

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module now exports the <a href="../library/socket.html#socket.CAN_RAW_JOIN_FILTERS" class="reference internal" title="socket.CAN_RAW_JOIN_FILTERS"><span class="pre"><code class="sourceCode python">CAN_RAW_JOIN_FILTERS</code></span></a> constant on Linux 4.1 and greater. (Contributed by Stefan Tatschner and Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25780" class="reference external">bpo-25780</a>.)

The socket module now supports the <a href="../library/socket.html#socket.CAN_J1939" class="reference internal" title="socket.CAN_J1939"><span class="pre"><code class="sourceCode python">CAN_J1939</code></span></a> protocol on platforms that support it. (Contributed by Karl Ding in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40291" class="reference external">bpo-40291</a>.)

The socket module now has the <a href="../library/socket.html#socket.send_fds" class="reference internal" title="socket.send_fds"><span class="pre"><code class="sourceCode python">socket.send_fds()</code></span></a> and <a href="../library/socket.html#socket.recv_fds" class="reference internal" title="socket.recv_fds"><span class="pre"><code class="sourceCode python">socket.recv_fds()</code></span></a> functions. (Contributed by Joannah Nanjekye, Shinya Okano and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28724" class="reference external">bpo-28724</a>.)

</div>

<div id="time" class="section">

### time<a href="#time" class="headerlink" title="Link to this heading">¶</a>

On AIX, <a href="../library/time.html#time.thread_time" class="reference internal" title="time.thread_time"><span class="pre"><code class="sourceCode python">thread_time()</code></span></a> is now implemented with <span class="pre">`thread_cputime()`</span> which has nanosecond resolution, rather than <span class="pre">`clock_gettime(CLOCK_THREAD_CPUTIME_ID)`</span> which has a resolution of 10 milliseconds. (Contributed by Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40192" class="reference external">bpo-40192</a>)

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

Added a new <a href="../library/sys.html#sys.platlibdir" class="reference internal" title="sys.platlibdir"><span class="pre"><code class="sourceCode python">sys.platlibdir</code></span></a> attribute: name of the platform-specific library directory. It is used to build the path of standard library and the paths of installed extension modules. It is equal to <span class="pre">`"lib"`</span> on most platforms. On Fedora and SuSE, it is equal to <span class="pre">`"lib64"`</span> on 64-bit platforms. (Contributed by Jan Matějek, Matěj Cepl, Charalampos Stratakis and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1294959" class="reference external">bpo-1294959</a>.)

Previously, <a href="../library/sys.html#sys.stderr" class="reference internal" title="sys.stderr"><span class="pre"><code class="sourceCode python">sys.stderr</code></span></a> was block-buffered when non-interactive. Now <span class="pre">`stderr`</span> defaults to always being line-buffered. (Contributed by Jendrik Seipp in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13601" class="reference external">bpo-13601</a>.)

</div>

<div id="tracemalloc" class="section">

### tracemalloc<a href="#tracemalloc" class="headerlink" title="Link to this heading">¶</a>

Added <a href="../library/tracemalloc.html#tracemalloc.reset_peak" class="reference internal" title="tracemalloc.reset_peak"><span class="pre"><code class="sourceCode python">tracemalloc.reset_peak()</code></span></a> to set the peak size of traced memory blocks to the current size, to measure the peak of specific pieces of code. (Contributed by Huon Wilson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40630" class="reference external">bpo-40630</a>.)

</div>

<div id="typing" class="section">

### typing<a href="#typing" class="headerlink" title="Link to this heading">¶</a>

<span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0593/" class="pep reference external"><strong>PEP 593</strong></a> introduced an <a href="../library/typing.html#typing.Annotated" class="reference internal" title="typing.Annotated"><span class="pre"><code class="sourceCode python">typing.Annotated</code></span></a> type to decorate existing types with context-specific metadata and new <span class="pre">`include_extras`</span> parameter to <a href="../library/typing.html#typing.get_type_hints" class="reference internal" title="typing.get_type_hints"><span class="pre"><code class="sourceCode python">typing.get_type_hints()</code></span></a> to access the metadata at runtime. (Contributed by Till Varoquaux and Konstantin Kashin.)

</div>

<div id="unicodedata" class="section">

### unicodedata<a href="#unicodedata" class="headerlink" title="Link to this heading">¶</a>

The Unicode database has been updated to version 13.0.0. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39926" class="reference external">bpo-39926</a>).

</div>

<div id="venv" class="section">

### venv<a href="#venv" class="headerlink" title="Link to this heading">¶</a>

The activation scripts provided by <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> now all specify their prompt customization consistently by always using the value specified by <span class="pre">`__VENV_PROMPT__`</span>. Previously some scripts unconditionally used <span class="pre">`__VENV_PROMPT__`</span>, others only if it happened to be set (which was the default case), and one used <span class="pre">`__VENV_NAME__`</span> instead. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37663" class="reference external">bpo-37663</a>.)

</div>

<div id="xml" class="section">

### xml<a href="#xml" class="headerlink" title="Link to this heading">¶</a>

White space characters within attributes are now preserved when serializing <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> to XML file. EOLNs are no longer normalized to “n”. This is the result of discussion about how to interpret section 2.11 of XML spec. (Contributed by Mefistotelis in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39011" class="reference external">bpo-39011</a>.)

</div>

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

- Optimized the idiom for assignment a temporary variable in comprehensions. Now <span class="pre">`for`</span>` `<span class="pre">`y`</span>` `<span class="pre">`in`</span>` `<span class="pre">`[expr]`</span> in comprehensions is as fast as a simple assignment <span class="pre">`y`</span>` `<span class="pre">`=`</span>` `<span class="pre">`expr`</span>. For example:

  > <div>
  >
  > sums = \[s for s in \[0\] for x in data for s in \[s + x\]\]
  >
  > </div>

  Unlike the <span class="pre">`:=`</span> operator this idiom does not leak a variable to the outer scope.

  (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32856" class="reference external">bpo-32856</a>.)

- Optimized signal handling in multithreaded applications. If a thread different than the main thread gets a signal, the bytecode evaluation loop is no longer interrupted at each bytecode instruction to check for pending signals which cannot be handled. Only the main thread of the main interpreter can handle signals.

  Previously, the bytecode evaluation loop was interrupted at each instruction until the main thread handles signals. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40010" class="reference external">bpo-40010</a>.)

- Optimized the <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module on FreeBSD using <span class="pre">`closefrom()`</span>. (Contributed by Ed Maste, Conrad Meyer, Kyle Evans, Kubilay Kocak and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38061" class="reference external">bpo-38061</a>.)

- <a href="../c-api/long.html#c.PyLong_FromDouble" class="reference internal" title="PyLong_FromDouble"><span class="pre"><code class="sourceCode c">PyLong_FromDouble<span class="op">()</span></code></span></a> is now up to 1.87x faster for values that fit into <span class="c-expr sig sig-inline c"><span class="kt">long</span></span>. (Contributed by Sergey Fedoseev in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37986" class="reference external">bpo-37986</a>.)

- A number of Python builtins (<a href="../library/stdtypes.html#range" class="reference internal" title="range"><span class="pre"><code class="sourceCode python"><span class="bu">range</span></code></span></a>, <a href="../library/stdtypes.html#tuple" class="reference internal" title="tuple"><span class="pre"><code class="sourceCode python"><span class="bu">tuple</span></code></span></a>, <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a>, <a href="../library/stdtypes.html#frozenset" class="reference internal" title="frozenset"><span class="pre"><code class="sourceCode python"><span class="bu">frozenset</span></code></span></a>, <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a>, <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a>) are now sped up by using <span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0590/" class="pep reference external"><strong>PEP 590</strong></a> vectorcall protocol. (Contributed by Donghee Na, Mark Shannon, Jeroen Demeyer and Petr Viktorin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37207" class="reference external">bpo-37207</a>.)

- Optimized <span class="pre">`set.difference_update()`</span> for the case when the other set is much larger than the base set. (Suggested by Evgeny Kapun with code contributed by Michele Orrù in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8425" class="reference external">bpo-8425</a>.)

- Python’s small object allocator (<span class="pre">`obmalloc.c`</span>) now allows (no more than) one empty arena to remain available for immediate reuse, without returning it to the OS. This prevents thrashing in simple loops where an arena could be created and destroyed anew on each iteration. (Contributed by Tim Peters in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37257" class="reference external">bpo-37257</a>.)

- <a href="../glossary.html#term-floor-division" class="reference internal"><span class="xref std std-term">floor division</span></a> of float operation now has a better performance. Also the message of <a href="../library/exceptions.html#ZeroDivisionError" class="reference internal" title="ZeroDivisionError"><span class="pre"><code class="sourceCode python"><span class="pp">ZeroDivisionError</span></code></span></a> for this operation is updated. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39434" class="reference external">bpo-39434</a>.)

- Decoding short ASCII strings with UTF-8 and ascii codecs is now about 15% faster. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37348" class="reference external">bpo-37348</a>.)

Here’s a summary of performance improvements from Python 3.4 through Python 3.9:

<div class="highlight-none notranslate">

<div class="highlight">

    Python version                       3.4     3.5     3.6     3.7     3.8    3.9
    --------------                       ---     ---     ---     ---     ---    ---

    Variable and attribute read access:
        read_local                       7.1     7.1     5.4     5.1     3.9    3.9
        read_nonlocal                    7.1     8.1     5.8     5.4     4.4    4.5
        read_global                     15.5    19.0    14.3    13.6     7.6    7.8
        read_builtin                    21.1    21.6    18.5    19.0     7.5    7.8
        read_classvar_from_class        25.6    26.5    20.7    19.5    18.4   17.9
        read_classvar_from_instance     22.8    23.5    18.8    17.1    16.4   16.9
        read_instancevar                32.4    33.1    28.0    26.3    25.4   25.3
        read_instancevar_slots          27.8    31.3    20.8    20.8    20.2   20.5
        read_namedtuple                 73.8    57.5    45.0    46.8    18.4   18.7
        read_boundmethod                37.6    37.9    29.6    26.9    27.7   41.1

    Variable and attribute write access:
        write_local                      8.7     9.3     5.5     5.3     4.3    4.3
        write_nonlocal                  10.5    11.1     5.6     5.5     4.7    4.8
        write_global                    19.7    21.2    18.0    18.0    15.8   16.7
        write_classvar                  92.9    96.0   104.6   102.1    39.2   39.8
        write_instancevar               44.6    45.8    40.0    38.9    35.5   37.4
        write_instancevar_slots         35.6    36.1    27.3    26.6    25.7   25.8

    Data structure read access:
        read_list                       24.2    24.5    20.8    20.8    19.0   19.5
        read_deque                      24.7    25.5    20.2    20.6    19.8   20.2
        read_dict                       24.3    25.7    22.3    23.0    21.0   22.4
        read_strdict                    22.6    24.3    19.5    21.2    18.9   21.5

    Data structure write access:
        write_list                      27.1    28.5    22.5    21.6    20.0   20.0
        write_deque                     28.7    30.1    22.7    21.8    23.5   21.7
        write_dict                      31.4    33.3    29.3    29.2    24.7   25.4
        write_strdict                   28.4    29.9    27.5    25.2    23.1   24.5

    Stack (or queue) operations:
        list_append_pop                 93.4   112.7    75.4    74.2    50.8   50.6
        deque_append_pop                43.5    57.0    49.4    49.2    42.5   44.2
        deque_append_popleft            43.7    57.3    49.7    49.7    42.8   46.4

    Timing loop:
        loop_overhead                    0.5     0.6     0.4     0.3     0.3    0.3

</div>

</div>

These results were generated from the variable access benchmark script at: <span class="pre">`Tools/scripts/var_access_benchmark.py`</span>. The benchmark script displays timings in nanoseconds. The benchmarks were measured on an <a href="https://ark.intel.com/content/www/us/en/ark/products/76088/intel-core-i7-4960hq-processor-6m-cache-up-to-3-80-ghz.html" class="reference external">Intel® Core™ i7-4960HQ processor</a> running the macOS 64-bit builds found at <a href="https://www.python.org/downloads/macos/" class="reference external">python.org</a>.

</div>

<div id="deprecated" class="section">

## Deprecated<a href="#deprecated" class="headerlink" title="Link to this heading">¶</a>

- The distutils <span class="pre">`bdist_msi`</span> command is now deprecated, use <span class="pre">`bdist_wheel`</span> (wheel packages) instead. (Contributed by Hugo van Kemenade in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39586" class="reference external">bpo-39586</a>.)

- Currently <a href="../library/math.html#math.factorial" class="reference internal" title="math.factorial"><span class="pre"><code class="sourceCode python">math.factorial()</code></span></a> accepts <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> instances with non-negative integer values (like <span class="pre">`5.0`</span>). It raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> for non-integral and negative floats. It is now deprecated. In future Python versions it will raise a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> for all floats. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37315" class="reference external">bpo-37315</a>.)

- The <span class="pre">`parser`</span> and <span class="pre">`symbol`</span> modules are deprecated and will be removed in future versions of Python. For the majority of use cases, users can leverage the Abstract Syntax Tree (AST) generation and compilation stage, using the <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> module.

- The Public C API functions <span class="pre">`PyParser_SimpleParseStringFlags()`</span>, <span class="pre">`PyParser_SimpleParseStringFlagsFilename()`</span>, <span class="pre">`PyParser_SimpleParseFileFlags()`</span> and <span class="pre">`PyNode_Compile()`</span> are deprecated and will be removed in Python 3.10 together with the old parser.

- Using <a href="../library/constants.html#NotImplemented" class="reference internal" title="NotImplemented"><span class="pre"><code class="sourceCode python"><span class="va">NotImplemented</span></code></span></a> in a boolean context has been deprecated, as it is almost exclusively the result of incorrect rich comparator implementations. It will be made a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> in a future version of Python. (Contributed by Josh Rosenberg in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35712" class="reference external">bpo-35712</a>.)

- The <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module currently accepts any hashable type as a possible seed value. Unfortunately, some of those types are not guaranteed to have a deterministic hash value. After Python 3.9, the module will restrict its seeds to <a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a>, <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>, <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a>, <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>, and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>.

- Opening the <a href="../library/gzip.html#gzip.GzipFile" class="reference internal" title="gzip.GzipFile"><span class="pre"><code class="sourceCode python">GzipFile</code></span></a> file for writing without specifying the *mode* argument is deprecated. In future Python versions it will always be opened for reading by default. Specify the *mode* argument for opening it for writing and silencing a warning. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28286" class="reference external">bpo-28286</a>.)

- Deprecated the <span class="pre">`split()`</span> method of <span class="pre">`_tkinter.TkappType`</span> in favour of the <span class="pre">`splitlist()`</span> method which has more consistent and predictable behavior. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38371" class="reference external">bpo-38371</a>.)

- The explicit passing of coroutine objects to <a href="../library/asyncio-task.html#asyncio.wait" class="reference internal" title="asyncio.wait"><span class="pre"><code class="sourceCode python">asyncio.wait()</code></span></a> has been deprecated and will be removed in version 3.11. (Contributed by Yury Selivanov and Kyle Stanley in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34790" class="reference external">bpo-34790</a>.)

- binhex4 and hexbin4 standards are now deprecated. The <span class="pre">`binhex`</span> module and the following <a href="../library/binascii.html#module-binascii" class="reference internal" title="binascii: Tools for converting between binary and various ASCII-encoded binary representations."><span class="pre"><code class="sourceCode python">binascii</code></span></a> functions are now deprecated:

  - <span class="pre">`b2a_hqx()`</span>, <span class="pre">`a2b_hqx()`</span>

  - <span class="pre">`rlecode_hqx()`</span>, <span class="pre">`rledecode_hqx()`</span>

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39353" class="reference external">bpo-39353</a>.)

- <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> classes <span class="pre">`slice`</span>, <span class="pre">`Index`</span> and <span class="pre">`ExtSlice`</span> are considered deprecated and will be removed in future Python versions. <span class="pre">`value`</span> itself should be used instead of <span class="pre">`Index(value)`</span>. <span class="pre">`Tuple(slices,`</span>` `<span class="pre">`Load())`</span> should be used instead of <span class="pre">`ExtSlice(slices)`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34822" class="reference external">bpo-34822</a>.)

- <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> classes <span class="pre">`Suite`</span>, <span class="pre">`Param`</span>, <span class="pre">`AugLoad`</span> and <span class="pre">`AugStore`</span> are considered deprecated and will be removed in future Python versions. They were not generated by the parser and not accepted by the code generator in Python 3. (Contributed by Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39639" class="reference external">bpo-39639</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39969" class="reference external">bpo-39969</a> and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39988" class="reference external">bpo-39988</a>.)

- The <span class="pre">`PyEval_InitThreads()`</span> and <span class="pre">`PyEval_ThreadsInitialized()`</span> functions are now deprecated and will be removed in Python 3.11. Calling <span class="pre">`PyEval_InitThreads()`</span> now does nothing. The <a href="../glossary.html#term-GIL" class="reference internal"><span class="xref std std-term">GIL</span></a> is initialized by <a href="../c-api/init.html#c.Py_Initialize" class="reference internal" title="Py_Initialize"><span class="pre"><code class="sourceCode c">Py_Initialize<span class="op">()</span></code></span></a> since Python 3.7. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39877" class="reference external">bpo-39877</a>.)

- Passing <span class="pre">`None`</span> as the first argument to the <a href="../library/shlex.html#shlex.split" class="reference internal" title="shlex.split"><span class="pre"><code class="sourceCode python">shlex.split()</code></span></a> function has been deprecated. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33262" class="reference external">bpo-33262</a>.)

- <span class="pre">`smtpd.MailmanProxy()`</span> is now deprecated as it is unusable without an external module, <span class="pre">`mailman`</span>. (Contributed by Samuel Colvin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35800" class="reference external">bpo-35800</a>.)

- The <span class="pre">`lib2to3`</span> module now emits a <a href="../library/exceptions.html#PendingDeprecationWarning" class="reference internal" title="PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a>. Python 3.9 switched to a PEG parser (see <span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0617/" class="pep reference external"><strong>PEP 617</strong></a>), and Python 3.10 may include new language syntax that is not parsable by lib2to3’s LL(1) parser. The <span class="pre">`lib2to3`</span> module may be removed from the standard library in a future Python version. Consider third-party alternatives such as <a href="https://libcst.readthedocs.io/" class="reference external">LibCST</a> or <a href="https://parso.readthedocs.io/" class="reference external">parso</a>. (Contributed by Carl Meyer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40360" class="reference external">bpo-40360</a>.)

- The *random* parameter of <a href="../library/random.html#random.shuffle" class="reference internal" title="random.shuffle"><span class="pre"><code class="sourceCode python">random.shuffle()</code></span></a> has been deprecated. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40465" class="reference external">bpo-40465</a>)

</div>

<div id="removed" class="section">

<span id="removed-in-python-39"></span>

## Removed<a href="#removed" class="headerlink" title="Link to this heading">¶</a>

- The erroneous version at <span class="pre">`unittest.mock.__version__`</span> has been removed.

- <span class="pre">`nntplib.NNTP`</span>: <span class="pre">`xpath()`</span> and <span class="pre">`xgtitle()`</span> methods have been removed. These methods are deprecated since Python 3.3. Generally, these extensions are not supported or not enabled by NNTP server administrators. For <span class="pre">`xgtitle()`</span>, please use <span class="pre">`nntplib.NNTP.descriptions()`</span> or <span class="pre">`nntplib.NNTP.description()`</span> instead. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39366" class="reference external">bpo-39366</a>.)

- <a href="../library/array.html#array.array" class="reference internal" title="array.array"><span class="pre"><code class="sourceCode python">array.array</code></span></a>: <span class="pre">`tostring()`</span> and <span class="pre">`fromstring()`</span> methods have been removed. They were aliases to <span class="pre">`tobytes()`</span> and <span class="pre">`frombytes()`</span>, deprecated since Python 3.2. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38916" class="reference external">bpo-38916</a>.)

- The undocumented <span class="pre">`sys.callstats()`</span> function has been removed. Since Python 3.7, it was deprecated and always returned <a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a>. It required a special build option <span class="pre">`CALL_PROFILE`</span> which was already removed in Python 3.7. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37414" class="reference external">bpo-37414</a>.)

- The <span class="pre">`sys.getcheckinterval()`</span> and <span class="pre">`sys.setcheckinterval()`</span> functions have been removed. They were deprecated since Python 3.2. Use <a href="../library/sys.html#sys.getswitchinterval" class="reference internal" title="sys.getswitchinterval"><span class="pre"><code class="sourceCode python">sys.getswitchinterval()</code></span></a> and <a href="../library/sys.html#sys.setswitchinterval" class="reference internal" title="sys.setswitchinterval"><span class="pre"><code class="sourceCode python">sys.setswitchinterval()</code></span></a> instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37392" class="reference external">bpo-37392</a>.)

- The C function <span class="pre">`PyImport_Cleanup()`</span> has been removed. It was documented as: “Empty the module table. For internal use only.” (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36710" class="reference external">bpo-36710</a>.)

- <span class="pre">`_dummy_thread`</span> and <span class="pre">`dummy_threading`</span> modules have been removed. These modules were deprecated since Python 3.7 which requires threading support. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37312" class="reference external">bpo-37312</a>.)

- <span class="pre">`aifc.openfp()`</span> alias to <span class="pre">`aifc.open()`</span>, <span class="pre">`sunau.openfp()`</span> alias to <span class="pre">`sunau.open()`</span>, and <span class="pre">`wave.openfp()`</span> alias to <a href="../library/wave.html#wave.open" class="reference internal" title="wave.open"><span class="pre"><code class="sourceCode python">wave.<span class="bu">open</span>()</code></span></a> have been removed. They were deprecated since Python 3.7. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37320" class="reference external">bpo-37320</a>.)

- The <span class="pre">`isAlive()`</span> method of <a href="../library/threading.html#threading.Thread" class="reference internal" title="threading.Thread"><span class="pre"><code class="sourceCode python">threading.Thread</code></span></a> has been removed. It was deprecated since Python 3.8. Use <a href="../library/threading.html#threading.Thread.is_alive" class="reference internal" title="threading.Thread.is_alive"><span class="pre"><code class="sourceCode python">is_alive()</code></span></a> instead. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37804" class="reference external">bpo-37804</a>.)

- Methods <span class="pre">`getchildren()`</span> and <span class="pre">`getiterator()`</span> of classes <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.ElementTree" class="reference internal" title="xml.etree.ElementTree.ElementTree"><span class="pre"><code class="sourceCode python">ElementTree</code></span></a> and <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element" class="reference internal" title="xml.etree.ElementTree.Element"><span class="pre"><code class="sourceCode python">Element</code></span></a> in the <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">ElementTree</code></span></a> module have been removed. They were deprecated in Python 3.2. Use <span class="pre">`iter(x)`</span> or <span class="pre">`list(x)`</span> instead of <span class="pre">`x.getchildren()`</span> and <span class="pre">`x.iter()`</span> or <span class="pre">`list(x.iter())`</span> instead of <span class="pre">`x.getiterator()`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36543" class="reference external">bpo-36543</a>.)

- The old <a href="../library/plistlib.html#module-plistlib" class="reference internal" title="plistlib: Generate and parse Apple plist files."><span class="pre"><code class="sourceCode python">plistlib</code></span></a> API has been removed, it was deprecated since Python 3.4. Use the <a href="../library/plistlib.html#plistlib.load" class="reference internal" title="plistlib.load"><span class="pre"><code class="sourceCode python">load()</code></span></a>, <a href="../library/plistlib.html#plistlib.loads" class="reference internal" title="plistlib.loads"><span class="pre"><code class="sourceCode python">loads()</code></span></a>, <a href="../library/plistlib.html#plistlib.dump" class="reference internal" title="plistlib.dump"><span class="pre"><code class="sourceCode python">dump()</code></span></a>, and <a href="../library/plistlib.html#plistlib.dumps" class="reference internal" title="plistlib.dumps"><span class="pre"><code class="sourceCode python">dumps()</code></span></a> functions. Additionally, the *use_builtin_types* parameter was removed, standard <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> objects are always used instead. (Contributed by Jon Janzen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36409" class="reference external">bpo-36409</a>.)

- The C function <span class="pre">`PyGen_NeedsFinalizing`</span> has been removed. It was not documented, tested, or used anywhere within CPython after the implementation of <span id="index-20" class="target"></span><a href="https://peps.python.org/pep-0442/" class="pep reference external"><strong>PEP 442</strong></a>. Patch by Joannah Nanjekye. (Contributed by Joannah Nanjekye in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15088" class="reference external">bpo-15088</a>)

- <span class="pre">`base64.encodestring()`</span> and <span class="pre">`base64.decodestring()`</span>, aliases deprecated since Python 3.1, have been removed: use <a href="../library/base64.html#base64.encodebytes" class="reference internal" title="base64.encodebytes"><span class="pre"><code class="sourceCode python">base64.encodebytes()</code></span></a> and <a href="../library/base64.html#base64.decodebytes" class="reference internal" title="base64.decodebytes"><span class="pre"><code class="sourceCode python">base64.decodebytes()</code></span></a> instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39351" class="reference external">bpo-39351</a>.)

- <span class="pre">`fractions.gcd()`</span> function has been removed, it was deprecated since Python 3.5 (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22486" class="reference external">bpo-22486</a>): use <a href="../library/math.html#math.gcd" class="reference internal" title="math.gcd"><span class="pre"><code class="sourceCode python">math.gcd()</code></span></a> instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39350" class="reference external">bpo-39350</a>.)

- The *buffering* parameter of <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">bz2.BZ2File</code></span></a> has been removed. Since Python 3.0, it was ignored and using it emitted a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. Pass an open file object to control how the file is opened. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39357" class="reference external">bpo-39357</a>.)

- The *encoding* parameter of <a href="../library/json.html#json.loads" class="reference internal" title="json.loads"><span class="pre"><code class="sourceCode python">json.loads()</code></span></a> has been removed. As of Python 3.1, it was deprecated and ignored; using it has emitted a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> since Python 3.8. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39377" class="reference external">bpo-39377</a>)

- <span class="pre">`with`</span>` `<span class="pre">`(await`</span>` `<span class="pre">`asyncio.lock):`</span> and <span class="pre">`with`</span>` `<span class="pre">`(yield`</span>` `<span class="pre">`from`</span>` `<span class="pre">`asyncio.lock):`</span> statements are not longer supported, use <span class="pre">`async`</span>` `<span class="pre">`with`</span>` `<span class="pre">`lock`</span> instead. The same is correct for <span class="pre">`asyncio.Condition`</span> and <span class="pre">`asyncio.Semaphore`</span>. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34793" class="reference external">bpo-34793</a>.)

- The <span class="pre">`sys.getcounts()`</span> function, the <span class="pre">`-X`</span>` `<span class="pre">`showalloccount`</span> command line option and the <span class="pre">`show_alloc_count`</span> field of the C structure <a href="../c-api/init_config.html#c.PyConfig" class="reference internal" title="PyConfig"><span class="pre"><code class="sourceCode c">PyConfig</code></span></a> have been removed. They required a special Python build by defining <span class="pre">`COUNT_ALLOCS`</span> macro. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39489" class="reference external">bpo-39489</a>.)

- The <span class="pre">`_field_types`</span> attribute of the <a href="../library/typing.html#typing.NamedTuple" class="reference internal" title="typing.NamedTuple"><span class="pre"><code class="sourceCode python">typing.NamedTuple</code></span></a> class has been removed. It was deprecated since Python 3.8. Use the <span class="pre">`__annotations__`</span> attribute instead. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40182" class="reference external">bpo-40182</a>.)

- The <span class="pre">`symtable.SymbolTable.has_exec()`</span> method has been removed. It was deprecated since 2006, and only returning <span class="pre">`False`</span> when it’s called. (Contributed by Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40208" class="reference external">bpo-40208</a>)

- The <span class="pre">`asyncio.Task.current_task()`</span> and <span class="pre">`asyncio.Task.all_tasks()`</span> have been removed. They were deprecated since Python 3.7 and you can use <a href="../library/asyncio-task.html#asyncio.current_task" class="reference internal" title="asyncio.current_task"><span class="pre"><code class="sourceCode python">asyncio.current_task()</code></span></a> and <a href="../library/asyncio-task.html#asyncio.all_tasks" class="reference internal" title="asyncio.all_tasks"><span class="pre"><code class="sourceCode python">asyncio.all_tasks()</code></span></a> instead. (Contributed by Rémi Lapeyre in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40967" class="reference external">bpo-40967</a>)

- The <span class="pre">`unescape()`</span> method in the <a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">html.parser.HTMLParser</code></span></a> class has been removed (it was deprecated since Python 3.4). <a href="../library/html.html#html.unescape" class="reference internal" title="html.unescape"><span class="pre"><code class="sourceCode python">html.unescape()</code></span></a> should be used for converting character references to the corresponding unicode characters.

</div>

<div id="porting-to-python-3-9" class="section">

## Porting to Python 3.9<a href="#porting-to-python-3-9" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="changes-in-the-python-api" class="section">

### Changes in the Python API<a href="#changes-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/functions.html#import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> and <a href="../library/importlib.html#importlib.util.resolve_name" class="reference internal" title="importlib.util.resolve_name"><span class="pre"><code class="sourceCode python">importlib.util.resolve_name()</code></span></a> now raise <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> where it previously raised <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a>. Callers catching the specific exception type and supporting both Python 3.9 and earlier versions will need to catch both using <span class="pre">`except`</span>` `<span class="pre">`(ImportError,`</span>` `<span class="pre">`ValueError):`</span>.

- The <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> activation scripts no longer special-case when <span class="pre">`__VENV_PROMPT__`</span> is set to <span class="pre">`""`</span>.

- The <a href="../library/select.html#select.epoll.unregister" class="reference internal" title="select.epoll.unregister"><span class="pre"><code class="sourceCode python">select.epoll.unregister()</code></span></a> method no longer ignores the <a href="../library/errno.html#errno.EBADF" class="reference internal" title="errno.EBADF"><span class="pre"><code class="sourceCode python">EBADF</code></span></a> error. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39239" class="reference external">bpo-39239</a>.)

- The *compresslevel* parameter of <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">bz2.BZ2File</code></span></a> became keyword-only, since the *buffering* parameter has been removed. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39357" class="reference external">bpo-39357</a>.)

- Simplified AST for subscription. Simple indices will be represented by their value, extended slices will be represented as tuples. <span class="pre">`Index(value)`</span> will return a <span class="pre">`value`</span> itself, <span class="pre">`ExtSlice(slices)`</span> will return <span class="pre">`Tuple(slices,`</span>` `<span class="pre">`Load())`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34822" class="reference external">bpo-34822</a>.)

- The <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> module now ignores the <span id="index-21" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONCASEOK" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONCASEOK</code></span></a> environment variable when the <a href="../using/cmdline.html#cmdoption-E" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-E</code></span></a> or <a href="../using/cmdline.html#cmdoption-I" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-I</code></span></a> command line options are being used.

- The *encoding* parameter has been added to the classes <a href="../library/ftplib.html#ftplib.FTP" class="reference internal" title="ftplib.FTP"><span class="pre"><code class="sourceCode python">ftplib.FTP</code></span></a> and <a href="../library/ftplib.html#ftplib.FTP_TLS" class="reference internal" title="ftplib.FTP_TLS"><span class="pre"><code class="sourceCode python">ftplib.FTP_TLS</code></span></a> as a keyword-only parameter, and the default encoding is changed from Latin-1 to UTF-8 to follow <span id="index-22" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc2640.html" class="rfc reference external"><strong>RFC 2640</strong></a>.

- <a href="../library/asyncio-eventloop.html#asyncio.loop.shutdown_default_executor" class="reference internal" title="asyncio.loop.shutdown_default_executor"><span class="pre"><code class="sourceCode python">asyncio.loop.shutdown_default_executor()</code></span></a> has been added to <a href="../library/asyncio-eventloop.html#asyncio.AbstractEventLoop" class="reference internal" title="asyncio.AbstractEventLoop"><span class="pre"><code class="sourceCode python">AbstractEventLoop</code></span></a>, meaning alternative event loops that inherit from it should have this method defined. (Contributed by Kyle Stanley in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34037" class="reference external">bpo-34037</a>.)

- The constant values of future flags in the <a href="../library/__future__.html#module-__future__" class="reference internal" title="__future__: Future statement definitions"><span class="pre"><code class="sourceCode python">__future__</code></span></a> module is updated in order to prevent collision with compiler flags. Previously <span class="pre">`PyCF_ALLOW_TOP_LEVEL_AWAIT`</span> was clashing with <span class="pre">`CO_FUTURE_DIVISION`</span>. (Contributed by Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39562" class="reference external">bpo-39562</a>)

- <span class="pre">`array('u')`</span> now uses <span class="pre">`wchar_t`</span> as C type instead of <span class="pre">`Py_UNICODE`</span>. This change doesn’t affect to its behavior because <span class="pre">`Py_UNICODE`</span> is alias of <span class="pre">`wchar_t`</span> since Python 3.3. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34538" class="reference external">bpo-34538</a>.)

- The <a href="../library/logging.html#logging.getLogger" class="reference internal" title="logging.getLogger"><span class="pre"><code class="sourceCode python">logging.getLogger()</code></span></a> API now returns the root logger when passed the name <span class="pre">`'root'`</span>, whereas previously it returned a non-root logger named <span class="pre">`'root'`</span>. This could affect cases where user code explicitly wants a non-root logger named <span class="pre">`'root'`</span>, or instantiates a logger using <span class="pre">`logging.getLogger(__name__)`</span> in some top-level module called <span class="pre">`'root.py'`</span>. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37742" class="reference external">bpo-37742</a>.)

- Division handling of <a href="../library/pathlib.html#pathlib.PurePath" class="reference internal" title="pathlib.PurePath"><span class="pre"><code class="sourceCode python">PurePath</code></span></a> now returns <a href="../library/constants.html#NotImplemented" class="reference internal" title="NotImplemented"><span class="pre"><code class="sourceCode python"><span class="va">NotImplemented</span></code></span></a> instead of raising a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> when passed something other than an instance of <span class="pre">`str`</span> or <a href="../library/pathlib.html#pathlib.PurePath" class="reference internal" title="pathlib.PurePath"><span class="pre"><code class="sourceCode python">PurePath</code></span></a>. This allows creating compatible classes that don’t inherit from those mentioned types. (Contributed by Roger Aiudi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34775" class="reference external">bpo-34775</a>).

- Starting with Python 3.9.5 the <a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> module no longer accepts any leading zeros in IPv4 address strings. Leading zeros are ambiguous and interpreted as octal notation by some libraries. For example the legacy function <a href="../library/socket.html#socket.inet_aton" class="reference internal" title="socket.inet_aton"><span class="pre"><code class="sourceCode python">socket.inet_aton()</code></span></a> treats leading zeros as octal notatation. glibc implementation of modern <a href="../library/socket.html#socket.inet_pton" class="reference internal" title="socket.inet_pton"><span class="pre"><code class="sourceCode python">inet_pton()</code></span></a> does not accept any leading zeros. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36384" class="reference external">bpo-36384</a>).

- <a href="../library/codecs.html#codecs.lookup" class="reference internal" title="codecs.lookup"><span class="pre"><code class="sourceCode python">codecs.lookup()</code></span></a> now normalizes the encoding name the same way as <a href="../library/codecs.html#encodings.normalize_encoding" class="reference internal" title="encodings.normalize_encoding"><span class="pre"><code class="sourceCode python">encodings.normalize_encoding()</code></span></a>, except that <a href="../library/codecs.html#codecs.lookup" class="reference internal" title="codecs.lookup"><span class="pre"><code class="sourceCode python">codecs.lookup()</code></span></a> also converts the name to lower case. For example, <span class="pre">`"latex+latin1"`</span> encoding name is now normalized to <span class="pre">`"latex_latin1"`</span>. (Contributed by Jordon Xu in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37751" class="reference external">bpo-37751</a>.)

</div>

<div id="changes-in-the-c-api" class="section">

### Changes in the C API<a href="#changes-in-the-c-api" class="headerlink" title="Link to this heading">¶</a>

- Instances of <a href="../c-api/typeobj.html#heap-types" class="reference internal"><span class="std std-ref">heap-allocated types</span></a> (such as those created with <a href="../c-api/type.html#c.PyType_FromSpec" class="reference internal" title="PyType_FromSpec"><span class="pre"><code class="sourceCode c">PyType_FromSpec<span class="op">()</span></code></span></a> and similar APIs) hold a reference to their type object since Python 3.8. As indicated in the “Changes in the C API” of Python 3.8, for the vast majority of cases, there should be no side effect but for types that have a custom <a href="../c-api/typeobj.html#c.PyTypeObject.tp_traverse" class="reference internal" title="PyTypeObject.tp_traverse"><span class="pre"><code class="sourceCode c">tp_traverse</code></span></a> function, ensure that all custom <span class="pre">`tp_traverse`</span> functions of heap-allocated types visit the object’s type.

  > <div>
  >
  > Example:
  >
  > <div class="highlight-c notranslate">
  >
  > <div class="highlight">
  >
  >     int
  >     foo_traverse(foo_struct *self, visitproc visit, void *arg) {
  >     // Rest of the traverse function
  >     #if PY_VERSION_HEX >= 0x03090000
  >         // This was not needed before Python 3.9 (Python issue 35810 and 40217)
  >         Py_VISIT(Py_TYPE(self));
  >     #endif
  >     }
  >
  > </div>
  >
  > </div>
  >
  > </div>

  If your traverse function delegates to <span class="pre">`tp_traverse`</span> of its base class (or another type), ensure that <span class="pre">`Py_TYPE(self)`</span> is visited only once. Note that only <a href="../c-api/typeobj.html#heap-types" class="reference internal"><span class="std std-ref">heap type</span></a> are expected to visit the type in <span class="pre">`tp_traverse`</span>.

  > <div>
  >
  > For example, if your <span class="pre">`tp_traverse`</span> function includes:
  >
  > <div class="highlight-c notranslate">
  >
  > <div class="highlight">
  >
  >     base->tp_traverse(self, visit, arg)
  >
  > </div>
  >
  > </div>
  >
  > then add:
  >
  > <div class="highlight-c notranslate">
  >
  > <div class="highlight">
  >
  >     #if PY_VERSION_HEX >= 0x03090000
  >         // This was not needed before Python 3.9 (bpo-35810 and bpo-40217)
  >         if (base->tp_flags & Py_TPFLAGS_HEAPTYPE) {
  >             // a heap type's tp_traverse already visited Py_TYPE(self)
  >         } else {
  >             Py_VISIT(Py_TYPE(self));
  >         }
  >     #else
  >
  > </div>
  >
  > </div>
  >
  > </div>

  (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35810" class="reference external">bpo-35810</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40217" class="reference external">bpo-40217</a> for more information.)

- The functions <span class="pre">`PyEval_CallObject`</span>, <span class="pre">`PyEval_CallFunction`</span>, <span class="pre">`PyEval_CallMethod`</span> and <span class="pre">`PyEval_CallObjectWithKeywords`</span> are deprecated. Use <a href="../c-api/call.html#c.PyObject_Call" class="reference internal" title="PyObject_Call"><span class="pre"><code class="sourceCode c">PyObject_Call<span class="op">()</span></code></span></a> and its variants instead. (See more details in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29548" class="reference external">bpo-29548</a>.)

</div>

<div id="cpython-bytecode-changes" class="section">

### CPython bytecode changes<a href="#cpython-bytecode-changes" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/dis.html#opcode-LOAD_ASSERTION_ERROR" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_ASSERTION_ERROR</code></span></a> opcode was added for handling the <a href="../reference/simple_stmts.html#assert" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">assert</code></span></a> statement. Previously, the assert statement would not work correctly if the <a href="../library/exceptions.html#AssertionError" class="reference internal" title="AssertionError"><span class="pre"><code class="sourceCode python"><span class="pp">AssertionError</span></code></span></a> exception was being shadowed. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34880" class="reference external">bpo-34880</a>.)

- The <a href="../library/dis.html#opcode-COMPARE_OP" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">COMPARE_OP</code></span></a> opcode was split into four distinct instructions:

  - <span class="pre">`COMPARE_OP`</span> for rich comparisons

  - <span class="pre">`IS_OP`</span> for ‘is’ and ‘is not’ tests

  - <span class="pre">`CONTAINS_OP`</span> for ‘in’ and ‘not in’ tests

  - <span class="pre">`JUMP_IF_NOT_EXC_MATCH`</span> for checking exceptions in ‘try-except’ statements.

  (Contributed by Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39156" class="reference external">bpo-39156</a>.)

</div>

</div>

<div id="build-changes" class="section">

## Build Changes<a href="#build-changes" class="headerlink" title="Link to this heading">¶</a>

- Added <span class="pre">`--with-platlibdir`</span> option to the <span class="pre">`configure`</span> script: name of the platform-specific library directory, stored in the new <a href="../library/sys.html#sys.platlibdir" class="reference internal" title="sys.platlibdir"><span class="pre"><code class="sourceCode python">sys.platlibdir</code></span></a> attribute. See <a href="../library/sys.html#sys.platlibdir" class="reference internal" title="sys.platlibdir"><span class="pre"><code class="sourceCode python">sys.platlibdir</code></span></a> attribute for more information. (Contributed by Jan Matějek, Matěj Cepl, Charalampos Stratakis and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1294959" class="reference external">bpo-1294959</a>.)

- The <span class="pre">`COUNT_ALLOCS`</span> special build macro has been removed. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39489" class="reference external">bpo-39489</a>.)

- On non-Windows platforms, the <span class="pre">`setenv()`</span> and <span class="pre">`unsetenv()`</span> functions are now required to build Python. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39395" class="reference external">bpo-39395</a>.)

- On non-Windows platforms, creating <span class="pre">`bdist_wininst`</span> installers is now officially unsupported. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10945" class="reference external">bpo-10945</a> for more details.)

- When building Python on macOS from source, <span class="pre">`_tkinter`</span> now links with non-system Tcl and Tk frameworks if they are installed in <span class="pre">`/Library/Frameworks`</span>, as had been the case on older releases of macOS. If a macOS SDK is explicitly configured, by using <a href="../using/configure.html#cmdoption-enable-universalsdk" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--enable-universalsdk</code></span></a> or <span class="pre">`-isysroot`</span>, only the SDK itself is searched. The default behavior can still be overridden with <span class="pre">`--with-tcltk-includes`</span> and <span class="pre">`--with-tcltk-libs`</span>. (Contributed by Ned Deily in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34956" class="reference external">bpo-34956</a>.)

- Python can now be built for Windows 10 ARM64. (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33125" class="reference external">bpo-33125</a>.)

- Some individual tests are now skipped when <span class="pre">`--pgo`</span> is used. The tests in question increased the PGO task time significantly and likely didn’t help improve optimization of the final executable. This speeds up the task by a factor of about 15x. Running the full unit test suite is slow. This change may result in a slightly less optimized build since not as many code branches will be executed. If you are willing to wait for the much slower build, the old behavior can be restored using <span class="pre">`./configure`</span>` `<span class="pre">`[..]`</span>` `<span class="pre">`PROFILE_TASK="-m`</span>` `<span class="pre">`test`</span>` `<span class="pre">`--pgo-extended"`</span>. We make no guarantees as to which PGO task set produces a faster build. Users who care should run their own relevant benchmarks as results can depend on the environment, workload, and compiler tool chain. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36044" class="reference external">bpo-36044</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37707" class="reference external">bpo-37707</a> for more details.)

</div>

<div id="c-api-changes" class="section">

## C API Changes<a href="#c-api-changes" class="headerlink" title="Link to this heading">¶</a>

<div id="id1" class="section">

### New Features<a href="#id1" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0573/" class="pep reference external"><strong>PEP 573</strong></a>: Added <a href="../c-api/type.html#c.PyType_FromModuleAndSpec" class="reference internal" title="PyType_FromModuleAndSpec"><span class="pre"><code class="sourceCode c">PyType_FromModuleAndSpec<span class="op">()</span></code></span></a> to associate a module with a class; <a href="../c-api/type.html#c.PyType_GetModule" class="reference internal" title="PyType_GetModule"><span class="pre"><code class="sourceCode c">PyType_GetModule<span class="op">()</span></code></span></a> and <a href="../c-api/type.html#c.PyType_GetModuleState" class="reference internal" title="PyType_GetModuleState"><span class="pre"><code class="sourceCode c">PyType_GetModuleState<span class="op">()</span></code></span></a> to retrieve the module and its state; and <a href="../c-api/structures.html#c.PyCMethod" class="reference internal" title="PyCMethod"><span class="pre"><code class="sourceCode c">PyCMethod</code></span></a> and <a href="../c-api/structures.html#c.METH_METHOD" class="reference internal" title="METH_METHOD"><span class="pre"><code class="sourceCode c">METH_METHOD</code></span></a> to allow a method to access the class it was defined in. (Contributed by Marcel Plch and Petr Viktorin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38787" class="reference external">bpo-38787</a>.)

- Added <a href="../c-api/frame.html#c.PyFrame_GetCode" class="reference internal" title="PyFrame_GetCode"><span class="pre"><code class="sourceCode c">PyFrame_GetCode<span class="op">()</span></code></span></a> function: get a frame code. Added <a href="../c-api/frame.html#c.PyFrame_GetBack" class="reference internal" title="PyFrame_GetBack"><span class="pre"><code class="sourceCode c">PyFrame_GetBack<span class="op">()</span></code></span></a> function: get the frame next outer frame. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40421" class="reference external">bpo-40421</a>.)

- Added <a href="../c-api/frame.html#c.PyFrame_GetLineNumber" class="reference internal" title="PyFrame_GetLineNumber"><span class="pre"><code class="sourceCode c">PyFrame_GetLineNumber<span class="op">()</span></code></span></a> to the limited C API. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40421" class="reference external">bpo-40421</a>.)

- Added <a href="../c-api/init.html#c.PyThreadState_GetInterpreter" class="reference internal" title="PyThreadState_GetInterpreter"><span class="pre"><code class="sourceCode c">PyThreadState_GetInterpreter<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyInterpreterState_Get" class="reference internal" title="PyInterpreterState_Get"><span class="pre"><code class="sourceCode c">PyInterpreterState_Get<span class="op">()</span></code></span></a> functions to get the interpreter. Added <a href="../c-api/init.html#c.PyThreadState_GetFrame" class="reference internal" title="PyThreadState_GetFrame"><span class="pre"><code class="sourceCode c">PyThreadState_GetFrame<span class="op">()</span></code></span></a> function to get the current frame of a Python thread state. Added <a href="../c-api/init.html#c.PyThreadState_GetID" class="reference internal" title="PyThreadState_GetID"><span class="pre"><code class="sourceCode c">PyThreadState_GetID<span class="op">()</span></code></span></a> function: get the unique identifier of a Python thread state. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39947" class="reference external">bpo-39947</a>.)

- Added a new public <a href="../c-api/call.html#c.PyObject_CallNoArgs" class="reference internal" title="PyObject_CallNoArgs"><span class="pre"><code class="sourceCode c">PyObject_CallNoArgs<span class="op">()</span></code></span></a> function to the C API, which calls a callable Python object without any arguments. It is the most efficient way to call a callable Python object without any argument. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37194" class="reference external">bpo-37194</a>.)

- Changes in the limited C API (if <span class="pre">`Py_LIMITED_API`</span> macro is defined):

  - Provide <a href="../c-api/exceptions.html#c.Py_EnterRecursiveCall" class="reference internal" title="Py_EnterRecursiveCall"><span class="pre"><code class="sourceCode c">Py_EnterRecursiveCall<span class="op">()</span></code></span></a> and <a href="../c-api/exceptions.html#c.Py_LeaveRecursiveCall" class="reference internal" title="Py_LeaveRecursiveCall"><span class="pre"><code class="sourceCode c">Py_LeaveRecursiveCall<span class="op">()</span></code></span></a> as regular functions for the limited API. Previously, there were defined as macros, but these macros didn’t compile with the limited C API which cannot access <span class="pre">`PyThreadState.recursion_depth`</span> field (the structure is opaque in the limited C API).

  - <span class="pre">`PyObject_INIT()`</span> and <span class="pre">`PyObject_INIT_VAR()`</span> become regular “opaque” function to hide implementation details.

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38644" class="reference external">bpo-38644</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39542" class="reference external">bpo-39542</a>.)

- The <a href="../c-api/module.html#c.PyModule_AddType" class="reference internal" title="PyModule_AddType"><span class="pre"><code class="sourceCode c">PyModule_AddType<span class="op">()</span></code></span></a> function is added to help adding a type to a module. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40024" class="reference external">bpo-40024</a>.)

- Added the functions <a href="../c-api/gcsupport.html#c.PyObject_GC_IsTracked" class="reference internal" title="PyObject_GC_IsTracked"><span class="pre"><code class="sourceCode c">PyObject_GC_IsTracked<span class="op">()</span></code></span></a> and <a href="../c-api/gcsupport.html#c.PyObject_GC_IsFinalized" class="reference internal" title="PyObject_GC_IsFinalized"><span class="pre"><code class="sourceCode c">PyObject_GC_IsFinalized<span class="op">()</span></code></span></a> to the public API to allow to query if Python objects are being currently tracked or have been already finalized by the garbage collector respectively. (Contributed by Pablo Galindo Salgado in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40241" class="reference external">bpo-40241</a>.)

- Added <span class="pre">`_PyObject_FunctionStr()`</span> to get a user-friendly string representation of a function-like object. (Patch by Jeroen Demeyer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37645" class="reference external">bpo-37645</a>.)

- Added <a href="../c-api/call.html#c.PyObject_CallOneArg" class="reference internal" title="PyObject_CallOneArg"><span class="pre"><code class="sourceCode c">PyObject_CallOneArg<span class="op">()</span></code></span></a> for calling an object with one positional argument (Patch by Jeroen Demeyer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37483" class="reference external">bpo-37483</a>.)

</div>

<div id="id2" class="section">

### Porting to Python 3.9<a href="#id2" class="headerlink" title="Link to this heading">¶</a>

- <span class="pre">`PyInterpreterState.eval_frame`</span> (<span id="index-24" class="target"></span><a href="https://peps.python.org/pep-0523/" class="pep reference external"><strong>PEP 523</strong></a>) now requires a new mandatory *tstate* parameter (<span class="pre">`PyThreadState*`</span>). (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38500" class="reference external">bpo-38500</a>.)

- Extension modules: <a href="../c-api/module.html#c.PyModuleDef.m_traverse" class="reference internal" title="PyModuleDef.m_traverse"><span class="pre"><code class="sourceCode c">m_traverse</code></span></a>, <a href="../c-api/module.html#c.PyModuleDef.m_clear" class="reference internal" title="PyModuleDef.m_clear"><span class="pre"><code class="sourceCode c">m_clear</code></span></a> and <a href="../c-api/module.html#c.PyModuleDef.m_free" class="reference internal" title="PyModuleDef.m_free"><span class="pre"><code class="sourceCode c">m_free</code></span></a> functions of <a href="../c-api/module.html#c.PyModuleDef" class="reference internal" title="PyModuleDef"><span class="pre"><code class="sourceCode c">PyModuleDef</code></span></a> are no longer called if the module state was requested but is not allocated yet. This is the case immediately after the module is created and before the module is executed (<a href="../c-api/module.html#c.Py_mod_exec" class="reference internal" title="Py_mod_exec"><span class="pre"><code class="sourceCode c">Py_mod_exec</code></span></a> function). More precisely, these functions are not called if <a href="../c-api/module.html#c.PyModuleDef.m_size" class="reference internal" title="PyModuleDef.m_size"><span class="pre"><code class="sourceCode c">m_size</code></span></a> is greater than 0 and the module state (as returned by <a href="../c-api/module.html#c.PyModule_GetState" class="reference internal" title="PyModule_GetState"><span class="pre"><code class="sourceCode c">PyModule_GetState<span class="op">()</span></code></span></a>) is <span class="pre">`NULL`</span>.

  Extension modules without module state (<span class="pre">`m_size`</span>` `<span class="pre">`<=`</span>` `<span class="pre">`0`</span>) are not affected.

- If <a href="../c-api/init.html#c.Py_AddPendingCall" class="reference internal" title="Py_AddPendingCall"><span class="pre"><code class="sourceCode c">Py_AddPendingCall<span class="op">()</span></code></span></a> is called in a subinterpreter, the function is now scheduled to be called from the subinterpreter, rather than being called from the main interpreter. Each subinterpreter now has its own list of scheduled calls. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39984" class="reference external">bpo-39984</a>.)

- The Windows registry is no longer used to initialize <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> when the <span class="pre">`-E`</span> option is used (if <a href="../c-api/init_config.html#c.PyConfig.use_environment" class="reference internal" title="PyConfig.use_environment"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>use_environment</code></span></a> is set to <span class="pre">`0`</span>). This is significant when embedding Python on Windows. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8901" class="reference external">bpo-8901</a>.)

- The global variable <a href="../c-api/tuple.html#c.PyStructSequence_UnnamedField" class="reference internal" title="PyStructSequence_UnnamedField"><span class="pre"><code class="sourceCode c">PyStructSequence_UnnamedField</code></span></a> is now a constant and refers to a constant string. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38650" class="reference external">bpo-38650</a>.)

- The <span class="pre">`PyGC_Head`</span> structure is now opaque. It is only defined in the internal C API (<span class="pre">`pycore_gc.h`</span>). (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40241" class="reference external">bpo-40241</a>.)

- The <span class="pre">`Py_UNICODE_COPY`</span>, <span class="pre">`Py_UNICODE_FILL`</span>, <span class="pre">`PyUnicode_WSTR_LENGTH`</span>, <span class="pre">`PyUnicode_FromUnicode()`</span>, <span class="pre">`PyUnicode_AsUnicode()`</span>, <span class="pre">`_PyUnicode_AsUnicode`</span>, and <span class="pre">`PyUnicode_AsUnicodeAndSize()`</span> are marked as deprecated in C. They have been deprecated by <span id="index-25" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a> since Python 3.3. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36346" class="reference external">bpo-36346</a>.)

- The <a href="../c-api/sys.html#c.Py_FatalError" class="reference internal" title="Py_FatalError"><span class="pre"><code class="sourceCode c">Py_FatalError<span class="op">()</span></code></span></a> function is replaced with a macro which logs automatically the name of the current function, unless the <span class="pre">`Py_LIMITED_API`</span> macro is defined. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39882" class="reference external">bpo-39882</a>.)

- The vectorcall protocol now requires that the caller passes only strings as keyword names. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37540" class="reference external">bpo-37540</a> for more information.)

- Implementation details of a number of macros and functions are now hidden:

  - <a href="../c-api/gcsupport.html#c.PyObject_IS_GC" class="reference internal" title="PyObject_IS_GC"><span class="pre"><code class="sourceCode c">PyObject_IS_GC<span class="op">()</span></code></span></a> macro was converted to a function.

  - The <span class="pre">`PyObject_NEW()`</span> macro becomes an alias to the <a href="../c-api/allocation.html#c.PyObject_New" class="reference internal" title="PyObject_New"><span class="pre"><code class="sourceCode c">PyObject_New</code></span></a> macro, and the <span class="pre">`PyObject_NEW_VAR()`</span> macro becomes an alias to the <a href="../c-api/allocation.html#c.PyObject_NewVar" class="reference internal" title="PyObject_NewVar"><span class="pre"><code class="sourceCode c">PyObject_NewVar</code></span></a> macro. They no longer access directly the <a href="../c-api/typeobj.html#c.PyTypeObject.tp_basicsize" class="reference internal" title="PyTypeObject.tp_basicsize"><span class="pre"><code class="sourceCode c">PyTypeObject<span class="op">.</span>tp_basicsize</code></span></a> member.

  - <span class="pre">`PyObject_GET_WEAKREFS_LISTPTR()`</span> macro was converted to a function: the macro accessed directly the <a href="../c-api/typeobj.html#c.PyTypeObject.tp_weaklistoffset" class="reference internal" title="PyTypeObject.tp_weaklistoffset"><span class="pre"><code class="sourceCode c">PyTypeObject<span class="op">.</span>tp_weaklistoffset</code></span></a> member.

  - <a href="../c-api/buffer.html#c.PyObject_CheckBuffer" class="reference internal" title="PyObject_CheckBuffer"><span class="pre"><code class="sourceCode c">PyObject_CheckBuffer<span class="op">()</span></code></span></a> macro was converted to a function: the macro accessed directly the <a href="../c-api/typeobj.html#c.PyTypeObject.tp_as_buffer" class="reference internal" title="PyTypeObject.tp_as_buffer"><span class="pre"><code class="sourceCode c">PyTypeObject<span class="op">.</span>tp_as_buffer</code></span></a> member.

  - <a href="../c-api/number.html#c.PyIndex_Check" class="reference internal" title="PyIndex_Check"><span class="pre"><code class="sourceCode c">PyIndex_Check<span class="op">()</span></code></span></a> is now always declared as an opaque function to hide implementation details: removed the <span class="pre">`PyIndex_Check()`</span> macro. The macro accessed directly the <a href="../c-api/typeobj.html#c.PyTypeObject.tp_as_number" class="reference internal" title="PyTypeObject.tp_as_number"><span class="pre"><code class="sourceCode c">PyTypeObject<span class="op">.</span>tp_as_number</code></span></a> member.

  (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40170" class="reference external">bpo-40170</a> for more details.)

</div>

<div id="id3" class="section">

### Removed<a href="#id3" class="headerlink" title="Link to this heading">¶</a>

- Excluded <span class="pre">`PyFPE_START_PROTECT()`</span> and <span class="pre">`PyFPE_END_PROTECT()`</span> macros of <span class="pre">`pyfpe.h`</span> from the limited C API. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38835" class="reference external">bpo-38835</a>.)

- The <span class="pre">`tp_print`</span> slot of <a href="../c-api/typeobj.html#type-structs" class="reference internal"><span class="std std-ref">PyTypeObject</span></a> has been removed. It was used for printing objects to files in Python 2.7 and before. Since Python 3.0, it has been ignored and unused. (Contributed by Jeroen Demeyer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36974" class="reference external">bpo-36974</a>.)

- Changes in the limited C API (if <span class="pre">`Py_LIMITED_API`</span> macro is defined):

  - Excluded the following functions from the limited C API:

    - <span class="pre">`PyThreadState_DeleteCurrent()`</span> (Contributed by Joannah Nanjekye in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37878" class="reference external">bpo-37878</a>.)

    - <span class="pre">`_Py_CheckRecursionLimit`</span>

    - <span class="pre">`_Py_NewReference()`</span>

    - <span class="pre">`_Py_ForgetReference()`</span>

    - <span class="pre">`_PyTraceMalloc_NewReference()`</span>

    - <span class="pre">`_Py_GetRefTotal()`</span>

    - The trashcan mechanism which never worked in the limited C API.

    - <span class="pre">`PyTrash_UNWIND_LEVEL`</span>

    - <span class="pre">`Py_TRASHCAN_BEGIN_CONDITION`</span>

    - <span class="pre">`Py_TRASHCAN_BEGIN`</span>

    - <span class="pre">`Py_TRASHCAN_END`</span>

    - <span class="pre">`Py_TRASHCAN_SAFE_BEGIN`</span>

    - <span class="pre">`Py_TRASHCAN_SAFE_END`</span>

  - Moved following functions and definitions to the internal C API:

    - <span class="pre">`_PyDebug_PrintTotalRefs()`</span>

    - <span class="pre">`_Py_PrintReferences()`</span>

    - <span class="pre">`_Py_PrintReferenceAddresses()`</span>

    - <span class="pre">`_Py_tracemalloc_config`</span>

    - <span class="pre">`_Py_AddToAllObjects()`</span> (specific to <span class="pre">`Py_TRACE_REFS`</span> build)

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38644" class="reference external">bpo-38644</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39542" class="reference external">bpo-39542</a>.)

- Removed <span class="pre">`_PyRuntime.getframe`</span> hook and removed <span class="pre">`_PyThreadState_GetFrame`</span> macro which was an alias to <span class="pre">`_PyRuntime.getframe`</span>. They were only exposed by the internal C API. Removed also <span class="pre">`PyThreadFrameGetter`</span> type. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39946" class="reference external">bpo-39946</a>.)

- Removed the following functions from the C API. Call <a href="../c-api/gcsupport.html#c.PyGC_Collect" class="reference internal" title="PyGC_Collect"><span class="pre"><code class="sourceCode c">PyGC_Collect<span class="op">()</span></code></span></a> explicitly to clear all free lists. (Contributed by Inada Naoki and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37340" class="reference external">bpo-37340</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38896" class="reference external">bpo-38896</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40428" class="reference external">bpo-40428</a>.)

  - <span class="pre">`PyAsyncGen_ClearFreeLists()`</span>

  - <span class="pre">`PyContext_ClearFreeList()`</span>

  - <span class="pre">`PyDict_ClearFreeList()`</span>

  - <span class="pre">`PyFloat_ClearFreeList()`</span>

  - <span class="pre">`PyFrame_ClearFreeList()`</span>

  - <span class="pre">`PyList_ClearFreeList()`</span>

  - <span class="pre">`PyMethod_ClearFreeList()`</span> and <span class="pre">`PyCFunction_ClearFreeList()`</span>: the free lists of bound method objects have been removed.

  - <span class="pre">`PySet_ClearFreeList()`</span>: the set free list has been removed in Python 3.4.

  - <span class="pre">`PyTuple_ClearFreeList()`</span>

  - <span class="pre">`PyUnicode_ClearFreeList()`</span>: the Unicode free list has been removed in Python 3.3.

- Removed <span class="pre">`_PyUnicode_ClearStaticStrings()`</span> function. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39465" class="reference external">bpo-39465</a>.)

- Removed <span class="pre">`Py_UNICODE_MATCH`</span>. It has been deprecated by <span id="index-26" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a>, and broken since Python 3.3. The <a href="../c-api/unicode.html#c.PyUnicode_Tailmatch" class="reference internal" title="PyUnicode_Tailmatch"><span class="pre"><code class="sourceCode c">PyUnicode_Tailmatch<span class="op">()</span></code></span></a> function can be used instead. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36346" class="reference external">bpo-36346</a>.)

- Cleaned header files of interfaces defined but with no implementation. The public API symbols being removed are: <span class="pre">`_PyBytes_InsertThousandsGroupingLocale`</span>, <span class="pre">`_PyBytes_InsertThousandsGrouping`</span>, <span class="pre">`_Py_InitializeFromArgs`</span>, <span class="pre">`_Py_InitializeFromWideArgs`</span>, <span class="pre">`_PyFloat_Repr`</span>, <span class="pre">`_PyFloat_Digits`</span>, <span class="pre">`_PyFloat_DigitsInit`</span>, <span class="pre">`PyFrame_ExtendStack`</span>, <span class="pre">`_PyAIterWrapper_Type`</span>, <span class="pre">`PyNullImporter_Type`</span>, <span class="pre">`PyCmpWrapper_Type`</span>, <span class="pre">`PySortWrapper_Type`</span>, <span class="pre">`PyNoArgsFunction`</span>. (Contributed by Pablo Galindo Salgado in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39372" class="reference external">bpo-39372</a>.)

</div>

</div>

<div id="notable-changes-in-python-3-9-1" class="section">

## Notable changes in Python 3.9.1<a href="#notable-changes-in-python-3-9-1" class="headerlink" title="Link to this heading">¶</a>

<div id="id4" class="section">

### typing<a href="#id4" class="headerlink" title="Link to this heading">¶</a>

The behavior of <a href="../library/typing.html#typing.Literal" class="reference internal" title="typing.Literal"><span class="pre"><code class="sourceCode python">typing.Literal</code></span></a> was changed to conform with <span id="index-27" class="target"></span><a href="https://peps.python.org/pep-0586/" class="pep reference external"><strong>PEP 586</strong></a> and to match the behavior of static type checkers specified in the PEP.

1.  <span class="pre">`Literal`</span> now de-duplicates parameters.

2.  Equality comparisons between <span class="pre">`Literal`</span> objects are now order independent.

3.  <span class="pre">`Literal`</span> comparisons now respect types. For example, <span class="pre">`Literal[0]`</span>` `<span class="pre">`==`</span>` `<span class="pre">`Literal[False]`</span> previously evaluated to <span class="pre">`True`</span>. It is now <span class="pre">`False`</span>. To support this change, the internally used type cache now supports differentiating types.

4.  <span class="pre">`Literal`</span> objects will now raise a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception during equality comparisons if any of their parameters are not <a href="../glossary.html#term-hashable" class="reference internal"><span class="xref std std-term">hashable</span></a>. Note that declaring <span class="pre">`Literal`</span> with mutable parameters will not throw an error:

    <div class="highlight-python3 notranslate">

    <div class="highlight">

        >>> from typing import Literal
        >>> Literal[{0}]
        >>> Literal[{0}] == Literal[{False}]
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        TypeError: unhashable type: 'set'

    </div>

    </div>

(Contributed by Yurii Karabas in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42345" class="reference external">bpo-42345</a>.)

</div>

<div id="macos-11-0-big-sur-and-apple-silicon-mac-support" class="section">

### macOS 11.0 (Big Sur) and Apple Silicon Mac support<a href="#macos-11-0-big-sur-and-apple-silicon-mac-support" class="headerlink" title="Link to this heading">¶</a>

As of 3.9.1, Python now fully supports building and running on macOS 11.0 (Big Sur) and on Apple Silicon Macs (based on the <span class="pre">`ARM64`</span> architecture). A new universal build variant, <span class="pre">`universal2`</span>, is now available to natively support both <span class="pre">`ARM64`</span> and <span class="pre">`Intel`</span>` `<span class="pre">`64`</span> in one set of executables. Binaries can also now be built on current versions of macOS to be deployed on a range of older macOS versions (tested to 10.9) while making some newer OS functions and options conditionally available based on the operating system version in use at runtime (“weaklinking”).

(Contributed by Ronald Oussoren and Lawrence D’Anna in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41100" class="reference external">bpo-41100</a>.)

</div>

</div>

<div id="notable-changes-in-python-3-9-2" class="section">

## Notable changes in Python 3.9.2<a href="#notable-changes-in-python-3-9-2" class="headerlink" title="Link to this heading">¶</a>

<div id="collections-abc" class="section">

### collections.abc<a href="#collections-abc" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/collections.abc.html#collections.abc.Callable" class="reference internal" title="collections.abc.Callable"><span class="pre"><code class="sourceCode python">collections.abc.Callable</code></span></a> generic now flattens type parameters, similar to what <a href="../library/typing.html#typing.Callable" class="reference internal" title="typing.Callable"><span class="pre"><code class="sourceCode python">typing.Callable</code></span></a> currently does. This means that <span class="pre">`collections.abc.Callable[[int,`</span>` `<span class="pre">`str],`</span>` `<span class="pre">`str]`</span> will have <span class="pre">`__args__`</span> of <span class="pre">`(int,`</span>` `<span class="pre">`str,`</span>` `<span class="pre">`str)`</span>; previously this was <span class="pre">`([int,`</span>` `<span class="pre">`str],`</span>` `<span class="pre">`str)`</span>. To allow this change, <a href="../library/types.html#types.GenericAlias" class="reference internal" title="types.GenericAlias"><span class="pre"><code class="sourceCode python">types.GenericAlias</code></span></a> can now be subclassed, and a subclass will be returned when subscripting the <a href="../library/collections.abc.html#collections.abc.Callable" class="reference internal" title="collections.abc.Callable"><span class="pre"><code class="sourceCode python">collections.abc.Callable</code></span></a> type. Code which accesses the arguments via <a href="../library/typing.html#typing.get_args" class="reference internal" title="typing.get_args"><span class="pre"><code class="sourceCode python">typing.get_args()</code></span></a> or <span class="pre">`__args__`</span> need to account for this change. A <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> may be emitted for invalid forms of parameterizing <a href="../library/collections.abc.html#collections.abc.Callable" class="reference internal" title="collections.abc.Callable"><span class="pre"><code class="sourceCode python">collections.abc.Callable</code></span></a> which may have passed silently in Python 3.9.1. This <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> will become a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> in Python 3.10. (Contributed by Ken Jin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42195" class="reference external">bpo-42195</a>.)

</div>

<div id="urllib-parse" class="section">

### urllib.parse<a href="#urllib-parse" class="headerlink" title="Link to this heading">¶</a>

Earlier Python versions allowed using both <span class="pre">`;`</span> and <span class="pre">`&`</span> as query parameter separators in <a href="../library/urllib.parse.html#urllib.parse.parse_qs" class="reference internal" title="urllib.parse.parse_qs"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qs()</code></span></a> and <a href="../library/urllib.parse.html#urllib.parse.parse_qsl" class="reference internal" title="urllib.parse.parse_qsl"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qsl()</code></span></a>. Due to security concerns, and to conform with newer W3C recommendations, this has been changed to allow only a single separator key, with <span class="pre">`&`</span> as the default. This change also affects <span class="pre">`cgi.parse()`</span> and <span class="pre">`cgi.parse_multipart()`</span> as they use the affected functions internally. For more details, please see their respective documentation. (Contributed by Adam Goldschmidt, Senthil Kumaran and Ken Jin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42967" class="reference external">bpo-42967</a>.)

</div>

</div>

<div id="notable-changes-in-python-3-9-3" class="section">

## Notable changes in Python 3.9.3<a href="#notable-changes-in-python-3-9-3" class="headerlink" title="Link to this heading">¶</a>

A security fix alters the <a href="../library/ftplib.html#ftplib.FTP" class="reference internal" title="ftplib.FTP"><span class="pre"><code class="sourceCode python">ftplib.FTP</code></span></a> behavior to not trust the IPv4 address sent from the remote server when setting up a passive data channel. We reuse the ftp server IP address instead. For unusual code requiring the old behavior, set a <span class="pre">`trust_server_pasv_ipv4_address`</span> attribute on your FTP instance to <span class="pre">`True`</span>. (See <a href="https://github.com/python/cpython/issues/87451" class="reference external">gh-87451</a>)

</div>

<div id="notable-changes-in-python-3-9-5" class="section">

## Notable changes in Python 3.9.5<a href="#notable-changes-in-python-3-9-5" class="headerlink" title="Link to this heading">¶</a>

<div id="id5" class="section">

### urllib.parse<a href="#id5" class="headerlink" title="Link to this heading">¶</a>

The presence of newline or tab characters in parts of a URL allows for some forms of attacks. Following the WHATWG specification that updates <span id="index-28" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc3986.html" class="rfc reference external"><strong>RFC 3986</strong></a>, ASCII newline <span class="pre">`\n`</span>, <span class="pre">`\r`</span> and tab <span class="pre">`\t`</span> characters are stripped from the URL by the parser in <a href="../library/urllib.parse.html#module-urllib.parse" class="reference internal" title="urllib.parse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urllib.parse</code></span></a> preventing such attacks. The removal characters are controlled by a new module level variable <span class="pre">`urllib.parse._UNSAFE_URL_BYTES_TO_REMOVE`</span>. (See <a href="https://github.com/python/cpython/issues/88048" class="reference external">gh-88048</a>)

</div>

</div>

<div id="notable-security-feature-in-3-9-14" class="section">

## Notable security feature in 3.9.14<a href="#notable-security-feature-in-3-9-14" class="headerlink" title="Link to this heading">¶</a>

Converting between <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> and <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> in bases other than 2 (binary), 4, 8 (octal), 16 (hexadecimal), or 32 such as base 10 (decimal) now raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the number of digits in string form is above a limit to avoid potential denial of service attacks due to the algorithmic complexity. This is a mitigation for <span id="index-29" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2020-10735" class="cve reference external"><strong>CVE 2020-10735</strong></a>. This limit can be configured or disabled by environment variable, command line flag, or <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> APIs. See the <a href="../library/stdtypes.html#int-max-str-digits" class="reference internal"><span class="std std-ref">integer string conversion length limitation</span></a> documentation. The default limit is 4300 digits in string form.

</div>

<div id="notable-changes-in-3-9-17" class="section">

## Notable changes in 3.9.17<a href="#notable-changes-in-3-9-17" class="headerlink" title="Link to this heading">¶</a>

<div id="tarfile" class="section">

### tarfile<a href="#tarfile" class="headerlink" title="Link to this heading">¶</a>

- The extraction methods in <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>, and <a href="../library/shutil.html#shutil.unpack_archive" class="reference internal" title="shutil.unpack_archive"><span class="pre"><code class="sourceCode python">shutil.unpack_archive()</code></span></a>, have a new a *filter* argument that allows limiting tar features than may be surprising or dangerous, such as creating files outside the destination directory. See <a href="../library/tarfile.html#tarfile-extraction-filter" class="reference internal"><span class="std std-ref">Extraction filters</span></a> for details. In Python 3.12, use without the *filter* argument will show a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. In Python 3.14, the default will switch to <span class="pre">`'data'`</span>. (Contributed by Petr Viktorin in <span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0706/" class="pep reference external"><strong>PEP 706</strong></a>.)

</div>

</div>

</div>

<div class="clearer">

</div>

</div>
