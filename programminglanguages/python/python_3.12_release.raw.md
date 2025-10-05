<div class="body" role="main">

<div id="what-s-new-in-python-3-12" class="section">

# What’s New In Python 3.12<a href="#what-s-new-in-python-3-12" class="headerlink" title="Link to this heading">¶</a>

Editor<span class="colon">:</span>  
Adam Turner

This article explains the new features in Python 3.12, compared to 3.11. Python 3.12 was released on October 2, 2023. For full details, see the <a href="changelog.html#changelog" class="reference internal"><span class="std std-ref">changelog</span></a>.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0693/" class="pep reference external"><strong>PEP 693</strong></a> – Python 3.12 Release Schedule

</div>

<div id="summary-release-highlights" class="section">

## Summary – Release highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

Python 3.12 is a stable release of the Python programming language, with a mix of changes to the language and the standard library. The library changes focus on cleaning up deprecated APIs, usability, and correctness. Of note, the <span class="pre">`distutils`</span> package has been removed from the standard library. Filesystem support in <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> and <a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a> has seen a number of improvements, and several modules have better performance.

The language changes focus on usability, as <a href="../glossary.html#term-f-string" class="reference internal"><span class="xref std std-term">f-strings</span></a> have had many limitations removed and ‘Did you mean …’ suggestions continue to improve. The new <a href="#whatsnew312-pep695" class="reference internal"><span class="std std-ref">type parameter syntax</span></a> and <a href="../reference/simple_stmts.html#type" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">type</code></span></a> statement improve ergonomics for using <a href="../glossary.html#term-generic-type" class="reference internal"><span class="xref std std-term">generic types</span></a> and <a href="../glossary.html#term-type-alias" class="reference internal"><span class="xref std std-term">type aliases</span></a> with static type checkers.

This article doesn’t attempt to provide a complete specification of all new features, but instead gives a convenient overview. For full details, you should refer to the documentation, such as the <a href="../library/index.html#library-index" class="reference internal"><span class="std std-ref">Library Reference</span></a> and <a href="../reference/index.html#reference-index" class="reference internal"><span class="std std-ref">Language Reference</span></a>. If you want to understand the complete implementation and design rationale for a change, refer to the PEP for a particular new feature; but note that PEPs usually are not kept up-to-date once a feature has been fully implemented.

------------------------------------------------------------------------

New syntax features:

- <a href="#whatsnew312-pep695" class="reference internal"><span class="std std-ref">PEP 695</span></a>, type parameter syntax and the <a href="../reference/simple_stmts.html#type" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">type</code></span></a> statement

New grammar features:

- <a href="#whatsnew312-pep701" class="reference internal"><span class="std std-ref">PEP 701</span></a>, <a href="../glossary.html#term-f-string" class="reference internal"><span class="xref std std-term">f-strings</span></a> in the grammar

Interpreter improvements:

- <a href="#whatsnew312-pep684" class="reference internal"><span class="std std-ref">PEP 684</span></a>, a unique per-interpreter <a href="../glossary.html#term-global-interpreter-lock" class="reference internal"><span class="xref std std-term">GIL</span></a>

- <a href="#whatsnew312-pep669" class="reference internal"><span class="std std-ref">PEP 669</span></a>, low impact monitoring

- <a href="#improved-error-messages" class="reference internal">Improved ‘Did you mean …’ suggestions</a> for <a href="../library/exceptions.html#NameError" class="reference internal" title="NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a>, <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a>, and <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> exceptions

Python data model improvements:

- <a href="#whatsnew312-pep688" class="reference internal"><span class="std std-ref">PEP 688</span></a>, using the <a href="../c-api/buffer.html#bufferobjects" class="reference internal"><span class="std std-ref">buffer protocol</span></a> from Python

Significant improvements in the standard library:

- The <a href="../library/pathlib.html#pathlib.Path" class="reference internal" title="pathlib.Path"><span class="pre"><code class="sourceCode python">pathlib.Path</code></span></a> class now supports subclassing

- The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module received several improvements for Windows support

- A <a href="../library/sqlite3.html#sqlite3-cli" class="reference internal"><span class="std std-ref">command-line interface</span></a> has been added to the <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> module

- <a href="../library/functions.html#isinstance" class="reference internal" title="isinstance"><span class="pre"><code class="sourceCode python"><span class="bu">isinstance</span>()</code></span></a> checks against <a href="../library/typing.html#typing.runtime_checkable" class="reference internal" title="typing.runtime_checkable"><span class="pre"><code class="sourceCode python">runtime<span class="op">-</span>checkable</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">protocols</code></span></a> enjoy a speed up of between two and 20 times

- The <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> package has had a number of performance improvements, with some benchmarks showing a 75% speed up.

- A <a href="../library/uuid.html#uuid-cli" class="reference internal"><span class="std std-ref">command-line interface</span></a> has been added to the <a href="../library/uuid.html#module-uuid" class="reference internal" title="uuid: UUID objects (universally unique identifiers) according to RFC 4122"><span class="pre"><code class="sourceCode python">uuid</code></span></a> module

- Due to the changes in <a href="#whatsnew312-pep701" class="reference internal"><span class="std std-ref">PEP 701</span></a>, producing tokens via the <a href="../library/tokenize.html#module-tokenize" class="reference internal" title="tokenize: Lexical scanner for Python source code."><span class="pre"><code class="sourceCode python">tokenize</code></span></a> module is up to 64% faster.

Security improvements:

- Replace the builtin <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> implementations of SHA1, SHA3, SHA2-384, SHA2-512, and MD5 with formally verified code from the <a href="https://github.com/hacl-star/hacl-star/" class="reference external">HACL*</a> project. These builtin implementations remain as fallbacks that are only used when OpenSSL does not provide them.

C API improvements:

- <a href="#whatsnew312-pep697" class="reference internal"><span class="std std-ref">PEP 697</span></a>, unstable C API tier

- <a href="#whatsnew312-pep683" class="reference internal"><span class="std std-ref">PEP 683</span></a>, immortal objects

CPython implementation improvements:

- <a href="#whatsnew312-pep709" class="reference internal"><span class="std std-ref">PEP 709</span></a>, comprehension inlining

- <a href="../howto/perf_profiling.html#perf-profiling" class="reference internal"><span class="std std-ref">CPython support</span></a> for the Linux <span class="pre">`perf`</span> profiler

- Implement stack overflow protection on supported platforms

New typing features:

- <a href="#whatsnew312-pep692" class="reference internal"><span class="std std-ref">PEP 692</span></a>, using <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">TypedDict</code></span></a> to annotate <a href="../glossary.html#term-argument" class="reference internal"><span class="xref std std-term">**kwargs</span></a>

- <a href="#whatsnew312-pep698" class="reference internal"><span class="std std-ref">PEP 698</span></a>, <a href="../library/typing.html#typing.override" class="reference internal" title="typing.override"><span class="pre"><code class="sourceCode python">typing.override()</code></span></a> decorator

Important deprecations, removals or restrictions:

- <span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0623/" class="pep reference external"><strong>PEP 623</strong></a>: Remove <span class="pre">`wstr`</span> from Unicode objects in Python’s C API, reducing the size of every <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> object by at least 8 bytes.

- <span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0632/" class="pep reference external"><strong>PEP 632</strong></a>: Remove the <span class="pre">`distutils`</span> package. See <span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0632/#migration-advice" class="pep reference external"><strong>the migration guide</strong></a> for advice replacing the APIs it provided. The third-party <a href="https://setuptools.pypa.io/en/latest/deprecated/distutils-legacy.html" class="reference external">Setuptools</a> package continues to provide <span class="pre">`distutils`</span>, if you still require it in Python 3.12 and beyond.

- <a href="https://github.com/python/cpython/issues/95299" class="reference external">gh-95299</a>: Do not pre-install <span class="pre">`setuptools`</span> in virtual environments created with <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a>. This means that <span class="pre">`distutils`</span>, <span class="pre">`setuptools`</span>, <span class="pre">`pkg_resources`</span>, and <span class="pre">`easy_install`</span> will no longer available by default; to access these run <span class="pre">`pip`</span>` `<span class="pre">`install`</span>` `<span class="pre">`setuptools`</span> in the <a href="../library/venv.html#venv-explanation" class="reference internal"><span class="std std-ref">activated</span></a> virtual environment.

- The <span class="pre">`asynchat`</span>, <span class="pre">`asyncore`</span>, and <span class="pre">`imp`</span> modules have been removed, along with several <a href="../library/unittest.html#unittest.TestCase" class="reference internal" title="unittest.TestCase"><span class="pre"><code class="sourceCode python">unittest.TestCase</code></span></a> <a href="#unittest-testcase-removed-aliases" class="reference internal">method aliases</a>.

</div>

<div id="new-features" class="section">

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-695-type-parameter-syntax" class="section">

<span id="whatsnew312-pep695"></span>

### PEP 695: Type Parameter Syntax<a href="#pep-695-type-parameter-syntax" class="headerlink" title="Link to this heading">¶</a>

Generic classes and functions under <span id="index-4" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> were declared using a verbose syntax that left the scope of type parameters unclear and required explicit declarations of variance.

<span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0695/" class="pep reference external"><strong>PEP 695</strong></a> introduces a new, more compact and explicit way to create <a href="../reference/compound_stmts.html#generic-classes" class="reference internal"><span class="std std-ref">generic classes</span></a> and <a href="../reference/compound_stmts.html#generic-functions" class="reference internal"><span class="std std-ref">functions</span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def max[T](args: Iterable[T]) -> T:
        ...

    class list[T]:
        def __getitem__(self, index: int, /) -> T:
            ...

        def append(self, element: T) -> None:
            ...

</div>

</div>

In addition, the PEP introduces a new way to declare <a href="../library/typing.html#type-aliases" class="reference internal"><span class="std std-ref">type aliases</span></a> using the <a href="../reference/simple_stmts.html#type" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">type</code></span></a> statement, which creates an instance of <a href="../library/typing.html#typing.TypeAliasType" class="reference internal" title="typing.TypeAliasType"><span class="pre"><code class="sourceCode python">TypeAliasType</code></span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    type Point = tuple[float, float]

</div>

</div>

Type aliases can also be <a href="../reference/compound_stmts.html#generic-type-aliases" class="reference internal"><span class="std std-ref">generic</span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    type Point[T] = tuple[T, T]

</div>

</div>

The new syntax allows declaring <a href="../library/typing.html#typing.TypeVarTuple" class="reference internal" title="typing.TypeVarTuple"><span class="pre"><code class="sourceCode python">TypeVarTuple</code></span></a> and <a href="../library/typing.html#typing.ParamSpec" class="reference internal" title="typing.ParamSpec"><span class="pre"><code class="sourceCode python">ParamSpec</code></span></a> parameters, as well as <a href="../library/typing.html#typing.TypeVar" class="reference internal" title="typing.TypeVar"><span class="pre"><code class="sourceCode python">TypeVar</code></span></a> parameters with bounds or constraints:

<div class="highlight-python3 notranslate">

<div class="highlight">

    type IntFunc[**P] = Callable[P, int]  # ParamSpec
    type LabeledTuple[*Ts] = tuple[str, *Ts]  # TypeVarTuple
    type HashableSequence[T: Hashable] = Sequence[T]  # TypeVar with bound
    type IntOrStrSequence[T: (int, str)] = Sequence[T]  # TypeVar with constraints

</div>

</div>

The value of type aliases and the bound and constraints of type variables created through this syntax are evaluated only on demand (see <a href="../reference/executionmodel.html#lazy-evaluation" class="reference internal"><span class="std std-ref">lazy evaluation</span></a>). This means type aliases are able to refer to other types defined later in the file.

Type parameters declared through a type parameter list are visible within the scope of the declaration and any nested scopes, but not in the outer scope. For example, they can be used in the type annotations for the methods of a generic class or in the class body. However, they cannot be used in the module scope after the class is defined. See <a href="../reference/compound_stmts.html#type-params" class="reference internal"><span class="std std-ref">Type parameter lists</span></a> for a detailed description of the runtime semantics of type parameters.

In order to support these scoping semantics, a new kind of scope is introduced, the <a href="../reference/executionmodel.html#annotation-scopes" class="reference internal"><span class="std std-ref">annotation scope</span></a>. Annotation scopes behave for the most part like function scopes, but interact differently with enclosing class scopes. In Python 3.13, <a href="../glossary.html#term-annotation" class="reference internal"><span class="xref std std-term">annotations</span></a> will also be evaluated in annotation scopes.

See <span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0695/" class="pep reference external"><strong>PEP 695</strong></a> for more details.

(PEP written by Eric Traut. Implementation by Jelle Zijlstra, Eric Traut, and others in <a href="https://github.com/python/cpython/issues/103764" class="reference external">gh-103764</a>.)

</div>

<div id="pep-701-syntactic-formalization-of-f-strings" class="section">

<span id="whatsnew312-pep701"></span>

### PEP 701: Syntactic formalization of f-strings<a href="#pep-701-syntactic-formalization-of-f-strings" class="headerlink" title="Link to this heading">¶</a>

<span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0701/" class="pep reference external"><strong>PEP 701</strong></a> lifts some restrictions on the usage of <a href="../glossary.html#term-f-string" class="reference internal"><span class="xref std std-term">f-strings</span></a>. Expression components inside f-strings can now be any valid Python expression, including strings reusing the same quote as the containing f-string, multi-line expressions, comments, backslashes, and unicode escape sequences. Let’s cover these in detail:

- Quote reuse: in Python 3.11, reusing the same quotes as the enclosing f-string raises a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>, forcing the user to either use other available quotes (like using double quotes or triple quotes if the f-string uses single quotes). In Python 3.12, you can now do things like this:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> songs = ['Take me back to Eden', 'Alkaline', 'Ascensionism']
      >>> f"This is the playlist: {", ".join(songs)}"
      'This is the playlist: Take me back to Eden, Alkaline, Ascensionism'

  </div>

  </div>

  Note that before this change there was no explicit limit in how f-strings can be nested, but the fact that string quotes cannot be reused inside the expression component of f-strings made it impossible to nest f-strings arbitrarily. In fact, this is the most nested f-string that could be written:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> f"""{f'''{f'{f"{1+1}"}'}'''}"""
      '2'

  </div>

  </div>

  As now f-strings can contain any valid Python expression inside expression components, it is now possible to nest f-strings arbitrarily:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> f"{f"{f"{f"{f"{f"{1+1}"}"}"}"}"}"
      '2'

  </div>

  </div>

- Multi-line expressions and comments: In Python 3.11, f-string expressions must be defined in a single line, even if the expression within the f-string could normally span multiple lines (like literal lists being defined over multiple lines), making them harder to read. In Python 3.12 you can now define f-strings spanning multiple lines, and add inline comments:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> f"This is the playlist: {", ".join([
      ...     'Take me back to Eden',  # My, my, those eyes like fire
      ...     'Alkaline',              # Not acid nor alkaline
      ...     'Ascensionism'           # Take to the broken skies at last
      ... ])}"
      'This is the playlist: Take me back to Eden, Alkaline, Ascensionism'

  </div>

  </div>

- Backslashes and unicode characters: before Python 3.12 f-string expressions couldn’t contain any <span class="pre">`\`</span> character. This also affected unicode <a href="../reference/lexical_analysis.html#escape-sequences" class="reference internal"><span class="std std-ref">escape sequences</span></a> (such as <span class="pre">`\N{snowman}`</span>) as these contain the <span class="pre">`\N`</span> part that previously could not be part of expression components of f-strings. Now, you can define expressions like this:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> print(f"This is the playlist: {"\n".join(songs)}")
      This is the playlist: Take me back to Eden
      Alkaline
      Ascensionism
      >>> print(f"This is the playlist: {"\N{BLACK HEART SUIT}".join(songs)}")
      This is the playlist: Take me back to Eden♥Alkaline♥Ascensionism

  </div>

  </div>

See <span id="index-8" class="target"></span><a href="https://peps.python.org/pep-0701/" class="pep reference external"><strong>PEP 701</strong></a> for more details.

As a positive side-effect of how this feature has been implemented (by parsing f-strings with <span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0617/" class="pep reference external"><strong>the PEG parser</strong></a>), now error messages for f-strings are more precise and include the exact location of the error. For example, in Python 3.11, the following f-string raises a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>:

<div class="highlight-python notranslate">

<div class="highlight">

    >>> my_string = f"{x z y}" + f"{1 + 1}"
      File "<stdin>", line 1
        (x z y)
         ^^^
    SyntaxError: f-string: invalid syntax. Perhaps you forgot a comma?

</div>

</div>

but the error message doesn’t include the exact location of the error within the line and also has the expression artificially surrounded by parentheses. In Python 3.12, as f-strings are parsed with the PEG parser, error messages can be more precise and show the entire line:

<div class="highlight-python notranslate">

<div class="highlight">

    >>> my_string = f"{x z y}" + f"{1 + 1}"
      File "<stdin>", line 1
        my_string = f"{x z y}" + f"{1 + 1}"
                       ^^^
    SyntaxError: invalid syntax. Perhaps you forgot a comma?

</div>

</div>

(Contributed by Pablo Galindo, Batuhan Taskaya, Lysandros Nikolaou, Cristián Maureira-Fredes and Marta Gómez in <a href="https://github.com/python/cpython/issues/102856" class="reference external">gh-102856</a>. PEP written by Pablo Galindo, Batuhan Taskaya, Lysandros Nikolaou and Marta Gómez).

</div>

<div id="pep-684-a-per-interpreter-gil" class="section">

<span id="whatsnew312-pep684"></span>

### PEP 684: A Per-Interpreter GIL<a href="#pep-684-a-per-interpreter-gil" class="headerlink" title="Link to this heading">¶</a>

<span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0684/" class="pep reference external"><strong>PEP 684</strong></a> introduces a per-interpreter <a href="../glossary.html#term-global-interpreter-lock" class="reference internal"><span class="xref std std-term">GIL</span></a>, so that sub-interpreters may now be created with a unique GIL per interpreter. This allows Python programs to take full advantage of multiple CPU cores. This is currently only available through the C-API, though a Python API is <span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0554/" class="pep reference external"><strong>anticipated for 3.13</strong></a>.

Use the new <a href="../c-api/init.html#c.Py_NewInterpreterFromConfig" class="reference internal" title="Py_NewInterpreterFromConfig"><span class="pre"><code class="sourceCode c">Py_NewInterpreterFromConfig<span class="op">()</span></code></span></a> function to create an interpreter with its own GIL:

<div class="highlight-c notranslate">

<div class="highlight">

    PyInterpreterConfig config = {
        .check_multi_interp_extensions = 1,
        .gil = PyInterpreterConfig_OWN_GIL,
    };
    PyThreadState *tstate = NULL;
    PyStatus status = Py_NewInterpreterFromConfig(&tstate, &config);
    if (PyStatus_Exception(status)) {
        return -1;
    }
    /* The new interpreter is now active in the current thread. */

</div>

</div>

For further examples how to use the C-API for sub-interpreters with a per-interpreter GIL, see <span class="pre">`Modules/_xxsubinterpretersmodule.c`</span>.

(Contributed by Eric Snow in <a href="https://github.com/python/cpython/issues/104210" class="reference external">gh-104210</a>, etc.)

</div>

<div id="pep-669-low-impact-monitoring-for-cpython" class="section">

<span id="whatsnew312-pep669"></span>

### PEP 669: Low impact monitoring for CPython<a href="#pep-669-low-impact-monitoring-for-cpython" class="headerlink" title="Link to this heading">¶</a>

<span id="index-12" class="target"></span><a href="https://peps.python.org/pep-0669/" class="pep reference external"><strong>PEP 669</strong></a> defines a new <a href="../library/sys.monitoring.html#module-sys.monitoring" class="reference internal" title="sys.monitoring: Access and control event monitoring"><span class="pre"><code class="sourceCode python">API</code></span></a> for profilers, debuggers, and other tools to monitor events in CPython. It covers a wide range of events, including calls, returns, lines, exceptions, jumps, and more. This means that you only pay for what you use, providing support for near-zero overhead debuggers and coverage tools. See <a href="../library/sys.monitoring.html#module-sys.monitoring" class="reference internal" title="sys.monitoring: Access and control event monitoring"><span class="pre"><code class="sourceCode python">sys.monitoring</code></span></a> for details.

(Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/103082" class="reference external">gh-103082</a>.)

</div>

<div id="pep-688-making-the-buffer-protocol-accessible-in-python" class="section">

<span id="whatsnew312-pep688"></span>

### PEP 688: Making the buffer protocol accessible in Python<a href="#pep-688-making-the-buffer-protocol-accessible-in-python" class="headerlink" title="Link to this heading">¶</a>

<span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0688/" class="pep reference external"><strong>PEP 688</strong></a> introduces a way to use the <a href="../c-api/buffer.html#bufferobjects" class="reference internal"><span class="std std-ref">buffer protocol</span></a> from Python code. Classes that implement the <a href="../reference/datamodel.html#object.__buffer__" class="reference internal" title="object.__buffer__"><span class="pre"><code class="sourceCode python"><span class="fu">__buffer__</span>()</code></span></a> method are now usable as buffer types.

The new <a href="../library/collections.abc.html#collections.abc.Buffer" class="reference internal" title="collections.abc.Buffer"><span class="pre"><code class="sourceCode python">collections.abc.Buffer</code></span></a> ABC provides a standard way to represent buffer objects, for example in type annotations. The new <a href="../library/inspect.html#inspect.BufferFlags" class="reference internal" title="inspect.BufferFlags"><span class="pre"><code class="sourceCode python">inspect.BufferFlags</code></span></a> enum represents the flags that can be used to customize buffer creation. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/102500" class="reference external">gh-102500</a>.)

</div>

<div id="pep-709-comprehension-inlining" class="section">

<span id="whatsnew312-pep709"></span>

### PEP 709: Comprehension inlining<a href="#pep-709-comprehension-inlining" class="headerlink" title="Link to this heading">¶</a>

Dictionary, list, and set comprehensions are now inlined, rather than creating a new single-use function object for each execution of the comprehension. This speeds up execution of a comprehension by up to two times. See <span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0709/" class="pep reference external"><strong>PEP 709</strong></a> for further details.

Comprehension iteration variables remain isolated and don’t overwrite a variable of the same name in the outer scope, nor are they visible after the comprehension. Inlining does result in a few visible behavior changes:

- There is no longer a separate frame for the comprehension in tracebacks, and tracing/profiling no longer shows the comprehension as a function call.

- The <a href="../library/symtable.html#module-symtable" class="reference internal" title="symtable: Interface to the compiler&#39;s internal symbol tables."><span class="pre"><code class="sourceCode python">symtable</code></span></a> module will no longer produce child symbol tables for each comprehension; instead, the comprehension’s locals will be included in the parent function’s symbol table.

- Calling <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> inside a comprehension now includes variables from outside the comprehension, and no longer includes the synthetic <span class="pre">`.0`</span> variable for the comprehension “argument”.

- A comprehension iterating directly over <span class="pre">`locals()`</span> (e.g. <span class="pre">`[k`</span>` `<span class="pre">`for`</span>` `<span class="pre">`k`</span>` `<span class="pre">`in`</span>` `<span class="pre">`locals()]`</span>) may see “RuntimeError: dictionary changed size during iteration” when run under tracing (e.g. code coverage measurement). This is the same behavior already seen in e.g. <span class="pre">`for`</span>` `<span class="pre">`k`</span>` `<span class="pre">`in`</span>` `<span class="pre">`locals():`</span>. To avoid the error, first create a list of keys to iterate over: <span class="pre">`keys`</span>` `<span class="pre">`=`</span>` `<span class="pre">`list(locals());`</span>` `<span class="pre">`[k`</span>` `<span class="pre">`for`</span>` `<span class="pre">`k`</span>` `<span class="pre">`in`</span>` `<span class="pre">`keys]`</span>.

(Contributed by Carl Meyer and Vladimir Matveev in <span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0709/" class="pep reference external"><strong>PEP 709</strong></a>.)

</div>

<div id="improved-error-messages" class="section">

### Improved Error Messages<a href="#improved-error-messages" class="headerlink" title="Link to this heading">¶</a>

- Modules from the standard library are now potentially suggested as part of the error messages displayed by the interpreter when a <a href="../library/exceptions.html#NameError" class="reference internal" title="NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a> is raised to the top level. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/98254" class="reference external">gh-98254</a>.)

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> sys.version_info
      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
      NameError: name 'sys' is not defined. Did you forget to import 'sys'?

  </div>

  </div>

- Improve the error suggestion for <a href="../library/exceptions.html#NameError" class="reference internal" title="NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a> exceptions for instances. Now if a <a href="../library/exceptions.html#NameError" class="reference internal" title="NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a> is raised in a method and the instance has an attribute that’s exactly equal to the name in the exception, the suggestion will include <span class="pre">`self.<NAME>`</span> instead of the closest match in the method scope. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/99139" class="reference external">gh-99139</a>.)

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> class A:
      ...    def __init__(self):
      ...        self.blech = 1
      ...
      ...    def foo(self):
      ...        somethin = blech
      ...
      >>> A().foo()
      Traceback (most recent call last):
        File "<stdin>", line 1
          somethin = blech
                     ^^^^^
      NameError: name 'blech' is not defined. Did you mean: 'self.blech'?

  </div>

  </div>

- Improve the <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> error message when the user types <span class="pre">`import`</span>` `<span class="pre">`x`</span>` `<span class="pre">`from`</span>` `<span class="pre">`y`</span> instead of <span class="pre">`from`</span>` `<span class="pre">`y`</span>` `<span class="pre">`import`</span>` `<span class="pre">`x`</span>. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/98931" class="reference external">gh-98931</a>.)

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import a.y.z from b.y.z
      Traceback (most recent call last):
        File "<stdin>", line 1
          import a.y.z from b.y.z
          ^^^^^^^^^^^^^^^^^^^^^^^
      SyntaxError: Did you mean to use 'from ... import ...' instead?

  </div>

  </div>

- <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> exceptions raised from failed <span class="pre">`from`</span>` `<span class="pre">`<module>`</span>` `<span class="pre">`import`</span>` `<span class="pre">`<name>`</span> statements now include suggestions for the value of <span class="pre">`<name>`</span> based on the available names in <span class="pre">`<module>`</span>. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/91058" class="reference external">gh-91058</a>.)

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> from collections import chainmap
      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
      ImportError: cannot import name 'chainmap' from 'collections'. Did you mean: 'ChainMap'?

  </div>

  </div>

</div>

</div>

<div id="new-features-related-to-type-hints" class="section">

## New Features Related to Type Hints<a href="#new-features-related-to-type-hints" class="headerlink" title="Link to this heading">¶</a>

This section covers major changes affecting <span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>type hints</strong></a> and the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module.

<div id="pep-692-using-typeddict-for-more-precise-kwargs-typing" class="section">

<span id="whatsnew312-pep692"></span>

### PEP 692: Using <span class="pre">`TypedDict`</span> for more precise <span class="pre">`**kwargs`</span> typing<a href="#pep-692-using-typeddict-for-more-precise-kwargs-typing" class="headerlink" title="Link to this heading">¶</a>

Typing <span class="pre">`**kwargs`</span> in a function signature as introduced by <span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> allowed for valid annotations only in cases where all of the <span class="pre">`**kwargs`</span> were of the same type.

<span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0692/" class="pep reference external"><strong>PEP 692</strong></a> specifies a more precise way of typing <span class="pre">`**kwargs`</span> by relying on typed dictionaries:

<div class="highlight-python3 notranslate">

<div class="highlight">

    from typing import TypedDict, Unpack

    class Movie(TypedDict):
      name: str
      year: int

    def foo(**kwargs: Unpack[Movie]): ...

</div>

</div>

See <span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0692/" class="pep reference external"><strong>PEP 692</strong></a> for more details.

(Contributed by Franek Magiera in <a href="https://github.com/python/cpython/issues/103629" class="reference external">gh-103629</a>.)

</div>

<div id="pep-698-override-decorator-for-static-typing" class="section">

<span id="whatsnew312-pep698"></span>

### PEP 698: Override Decorator for Static Typing<a href="#pep-698-override-decorator-for-static-typing" class="headerlink" title="Link to this heading">¶</a>

A new decorator <a href="../library/typing.html#typing.override" class="reference internal" title="typing.override"><span class="pre"><code class="sourceCode python">typing.override()</code></span></a> has been added to the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module. It indicates to type checkers that the method is intended to override a method in a superclass. This allows type checkers to catch mistakes where a method that is intended to override something in a base class does not in fact do so.

Example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    from typing import override

    class Base:
      def get_color(self) -> str:
        return "blue"

    class GoodChild(Base):
      @override  # ok: overrides Base.get_color
      def get_color(self) -> str:
        return "yellow"

    class BadChild(Base):
      @override  # type checker error: does not override Base.get_color
      def get_colour(self) -> str:
        return "red"

</div>

</div>

See <span id="index-20" class="target"></span><a href="https://peps.python.org/pep-0698/" class="pep reference external"><strong>PEP 698</strong></a> for more details.

(Contributed by Steven Troxler in <a href="https://github.com/python/cpython/issues/101561" class="reference external">gh-101561</a>.)

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

- The parser now raises <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> when parsing source code containing null bytes. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/96670" class="reference external">gh-96670</a>.)

- A backslash-character pair that is not a valid escape sequence now generates a <a href="../library/exceptions.html#SyntaxWarning" class="reference internal" title="SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a>, instead of <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. For example, <span class="pre">`re.compile("\d+\.\d+")`</span> now emits a <a href="../library/exceptions.html#SyntaxWarning" class="reference internal" title="SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a> (<span class="pre">`"\d"`</span> is an invalid escape sequence, use raw strings for regular expression: <span class="pre">`re.compile(r"\d+\.\d+")`</span>). In a future Python version, <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> will eventually be raised, instead of <a href="../library/exceptions.html#SyntaxWarning" class="reference internal" title="SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/98401" class="reference external">gh-98401</a>.)

- Octal escapes with value larger than <span class="pre">`0o377`</span> (ex: <span class="pre">`"\477"`</span>), deprecated in Python 3.11, now produce a <a href="../library/exceptions.html#SyntaxWarning" class="reference internal" title="SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a>, instead of <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. In a future Python version they will be eventually a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/98401" class="reference external">gh-98401</a>.)

- Variables used in the target part of comprehensions that are not stored to can now be used in assignment expressions (<span class="pre">`:=`</span>). For example, in <span class="pre">`[(b`</span>` `<span class="pre">`:=`</span>` `<span class="pre">`1)`</span>` `<span class="pre">`for`</span>` `<span class="pre">`a,`</span>` `<span class="pre">`b.prop`</span>` `<span class="pre">`in`</span>` `<span class="pre">`some_iter]`</span>, the assignment to <span class="pre">`b`</span> is now allowed. Note that assigning to variables stored to in the target part of comprehensions (like <span class="pre">`a`</span>) is still disallowed, as per <span id="index-21" class="target"></span><a href="https://peps.python.org/pep-0572/" class="pep reference external"><strong>PEP 572</strong></a>. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/100581" class="reference external">gh-100581</a>.)

- Exceptions raised in a class or type’s <span class="pre">`__set_name__`</span> method are no longer wrapped by a <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a>. Context information is added to the exception as a <span id="index-22" class="target"></span><a href="https://peps.python.org/pep-0678/" class="pep reference external"><strong>PEP 678</strong></a> note. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/77757" class="reference external">gh-77757</a>.)

- When a <span class="pre">`try-except*`</span> construct handles the entire <a href="../library/exceptions.html#ExceptionGroup" class="reference internal" title="ExceptionGroup"><span class="pre"><code class="sourceCode python">ExceptionGroup</code></span></a> and raises one other exception, that exception is no longer wrapped in an <a href="../library/exceptions.html#ExceptionGroup" class="reference internal" title="ExceptionGroup"><span class="pre"><code class="sourceCode python">ExceptionGroup</code></span></a>. Also changed in version 3.11.4. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/103590" class="reference external">gh-103590</a>.)

- The Garbage Collector now runs only on the eval breaker mechanism of the Python bytecode evaluation loop instead of object allocations. The GC can also run when <a href="../c-api/exceptions.html#c.PyErr_CheckSignals" class="reference internal" title="PyErr_CheckSignals"><span class="pre"><code class="sourceCode c">PyErr_CheckSignals<span class="op">()</span></code></span></a> is called so C extensions that need to run for a long time without executing any Python code also have a chance to execute the GC periodically. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/97922" class="reference external">gh-97922</a>.)

- All builtin and extension callables expecting boolean parameters now accept arguments of any type instead of just <a href="../library/functions.html#bool" class="reference internal" title="bool"><span class="pre"><code class="sourceCode python"><span class="bu">bool</span></code></span></a> and <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/60203" class="reference external">gh-60203</a>.)

- <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> now supports the half-float type (the “e” format code). (Contributed by Donghee Na and Antoine Pitrou in <a href="https://github.com/python/cpython/issues/90751" class="reference external">gh-90751</a>.)

- <a href="../library/functions.html#slice" class="reference internal" title="slice"><span class="pre"><code class="sourceCode python"><span class="bu">slice</span></code></span></a> objects are now hashable, allowing them to be used as dict keys and set items. (Contributed by Will Bradshaw, Furkan Onder, and Raymond Hettinger in <a href="https://github.com/python/cpython/issues/101264" class="reference external">gh-101264</a>.)

- <a href="../library/functions.html#sum" class="reference internal" title="sum"><span class="pre"><code class="sourceCode python"><span class="bu">sum</span>()</code></span></a> now uses Neumaier summation to improve accuracy and commutativity when summing floats or mixed ints and floats. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/100425" class="reference external">gh-100425</a>.)

- <a href="../library/ast.html#ast.parse" class="reference internal" title="ast.parse"><span class="pre"><code class="sourceCode python">ast.parse()</code></span></a> now raises <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> instead of <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> when parsing source code containing null bytes. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/96670" class="reference external">gh-96670</a>.)

- The extraction methods in <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>, and <a href="../library/shutil.html#shutil.unpack_archive" class="reference internal" title="shutil.unpack_archive"><span class="pre"><code class="sourceCode python">shutil.unpack_archive()</code></span></a>, have a new a *filter* argument that allows limiting tar features than may be surprising or dangerous, such as creating files outside the destination directory. See <a href="../library/tarfile.html#tarfile-extraction-filter" class="reference internal"><span class="std std-ref">tarfile extraction filters</span></a> for details. In Python 3.14, the default will switch to <span class="pre">`'data'`</span>. (Contributed by Petr Viktorin in <span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0706/" class="pep reference external"><strong>PEP 706</strong></a>.)

- <a href="../library/types.html#types.MappingProxyType" class="reference internal" title="types.MappingProxyType"><span class="pre"><code class="sourceCode python">types.MappingProxyType</code></span></a> instances are now hashable if the underlying mapping is hashable. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/87995" class="reference external">gh-87995</a>.)

- Add <a href="../howto/perf_profiling.html#perf-profiling" class="reference internal"><span class="std std-ref">support for the perf profiler</span></a> through the new environment variable <span id="index-24" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONPERFSUPPORT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONPERFSUPPORT</code></span></a> and command-line option <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">perf</code></span></a>, as well as the new <a href="../library/sys.html#sys.activate_stack_trampoline" class="reference internal" title="sys.activate_stack_trampoline"><span class="pre"><code class="sourceCode python">sys.activate_stack_trampoline()</code></span></a>, <a href="../library/sys.html#sys.deactivate_stack_trampoline" class="reference internal" title="sys.deactivate_stack_trampoline"><span class="pre"><code class="sourceCode python">sys.deactivate_stack_trampoline()</code></span></a>, and <a href="../library/sys.html#sys.is_stack_trampoline_active" class="reference internal" title="sys.is_stack_trampoline_active"><span class="pre"><code class="sourceCode python">sys.is_stack_trampoline_active()</code></span></a> functions. (Design by Pablo Galindo. Contributed by Pablo Galindo and Christian Heimes with contributions from Gregory P. Smith \[Google\] and Mark Shannon in <a href="https://github.com/python/cpython/issues/96123" class="reference external">gh-96123</a>.)

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

- None.

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="array" class="section">

### array<a href="#array" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/array.html#array.array" class="reference internal" title="array.array"><span class="pre"><code class="sourceCode python">array.array</code></span></a> class now supports subscripting, making it a <a href="../glossary.html#term-generic-type" class="reference internal"><span class="xref std std-term">generic type</span></a>. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/98658" class="reference external">gh-98658</a>.)

</div>

<div id="asyncio" class="section">

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

- The performance of writing to sockets in <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> has been significantly improved. <span class="pre">`asyncio`</span> now avoids unnecessary copying when writing to sockets and uses <a href="../library/socket.html#socket.socket.sendmsg" class="reference internal" title="socket.socket.sendmsg"><span class="pre"><code class="sourceCode python">sendmsg()</code></span></a> if the platform supports it. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/91166" class="reference external">gh-91166</a>.)

- Add <a href="../library/asyncio-task.html#asyncio.eager_task_factory" class="reference internal" title="asyncio.eager_task_factory"><span class="pre"><code class="sourceCode python">asyncio.eager_task_factory()</code></span></a> and <a href="../library/asyncio-task.html#asyncio.create_eager_task_factory" class="reference internal" title="asyncio.create_eager_task_factory"><span class="pre"><code class="sourceCode python">asyncio.create_eager_task_factory()</code></span></a> functions to allow opting an event loop in to eager task execution, making some use-cases 2x to 5x faster. (Contributed by Jacob Bower & Itamar Oren in <a href="https://github.com/python/cpython/issues/102853" class="reference external">gh-102853</a>, <a href="https://github.com/python/cpython/issues/104140" class="reference external">gh-104140</a>, and <a href="https://github.com/python/cpython/issues/104138" class="reference external">gh-104138</a>)

- On Linux, <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> uses <a href="../library/asyncio-policy.html#asyncio.PidfdChildWatcher" class="reference internal" title="asyncio.PidfdChildWatcher"><span class="pre"><code class="sourceCode python">asyncio.PidfdChildWatcher</code></span></a> by default if <a href="../library/os.html#os.pidfd_open" class="reference internal" title="os.pidfd_open"><span class="pre"><code class="sourceCode python">os.pidfd_open()</code></span></a> is available and functional instead of <a href="../library/asyncio-policy.html#asyncio.ThreadedChildWatcher" class="reference internal" title="asyncio.ThreadedChildWatcher"><span class="pre"><code class="sourceCode python">asyncio.ThreadedChildWatcher</code></span></a>. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/98024" class="reference external">gh-98024</a>.)

- The event loop now uses the best available child watcher for each platform (<a href="../library/asyncio-policy.html#asyncio.PidfdChildWatcher" class="reference internal" title="asyncio.PidfdChildWatcher"><span class="pre"><code class="sourceCode python">asyncio.PidfdChildWatcher</code></span></a> if supported and <a href="../library/asyncio-policy.html#asyncio.ThreadedChildWatcher" class="reference internal" title="asyncio.ThreadedChildWatcher"><span class="pre"><code class="sourceCode python">asyncio.ThreadedChildWatcher</code></span></a> otherwise), so manually configuring a child watcher is not recommended. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/94597" class="reference external">gh-94597</a>.)

- Add *loop_factory* parameter to <a href="../library/asyncio-runner.html#asyncio.run" class="reference internal" title="asyncio.run"><span class="pre"><code class="sourceCode python">asyncio.run()</code></span></a> to allow specifying a custom event loop factory. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/99388" class="reference external">gh-99388</a>.)

- Add C implementation of <a href="../library/asyncio-task.html#asyncio.current_task" class="reference internal" title="asyncio.current_task"><span class="pre"><code class="sourceCode python">asyncio.current_task()</code></span></a> for 4x-6x speedup. (Contributed by Itamar Oren and Pranav Thulasiram Bhat in <a href="https://github.com/python/cpython/issues/100344" class="reference external">gh-100344</a>.)

- <a href="../library/asyncio-task.html#asyncio.iscoroutine" class="reference internal" title="asyncio.iscoroutine"><span class="pre"><code class="sourceCode python">asyncio.iscoroutine()</code></span></a> now returns <span class="pre">`False`</span> for generators as <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> does not support legacy generator-based coroutines. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/102748" class="reference external">gh-102748</a>.)

- <a href="../library/asyncio-task.html#asyncio.wait" class="reference internal" title="asyncio.wait"><span class="pre"><code class="sourceCode python">asyncio.wait()</code></span></a> and <a href="../library/asyncio-task.html#asyncio.as_completed" class="reference internal" title="asyncio.as_completed"><span class="pre"><code class="sourceCode python">asyncio.as_completed()</code></span></a> now accepts generators yielding tasks. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/78530" class="reference external">gh-78530</a>.)

</div>

<div id="calendar" class="section">

### calendar<a href="#calendar" class="headerlink" title="Link to this heading">¶</a>

- Add enums <a href="../library/calendar.html#calendar.Month" class="reference internal" title="calendar.Month"><span class="pre"><code class="sourceCode python">calendar.Month</code></span></a> and <a href="../library/calendar.html#calendar.Day" class="reference internal" title="calendar.Day"><span class="pre"><code class="sourceCode python">calendar.Day</code></span></a> defining months of the year and days of the week. (Contributed by Prince Roshan in <a href="https://github.com/python/cpython/issues/103636" class="reference external">gh-103636</a>.)

</div>

<div id="csv" class="section">

### csv<a href="#csv" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/csv.html#csv.QUOTE_NOTNULL" class="reference internal" title="csv.QUOTE_NOTNULL"><span class="pre"><code class="sourceCode python">csv.QUOTE_NOTNULL</code></span></a> and <a href="../library/csv.html#csv.QUOTE_STRINGS" class="reference internal" title="csv.QUOTE_STRINGS"><span class="pre"><code class="sourceCode python">csv.QUOTE_STRINGS</code></span></a> flags to provide finer grained control of <span class="pre">`None`</span> and empty strings by <a href="../library/csv.html#csv.reader" class="reference internal" title="csv.reader"><span class="pre"><code class="sourceCode python">reader</code></span></a> and <a href="../library/csv.html#csv.writer" class="reference internal" title="csv.writer"><span class="pre"><code class="sourceCode python">writer</code></span></a> objects.

</div>

<div id="dis" class="section">

### dis<a href="#dis" class="headerlink" title="Link to this heading">¶</a>

- Pseudo instruction opcodes (which are used by the compiler but do not appear in executable bytecode) are now exposed in the <a href="../library/dis.html#module-dis" class="reference internal" title="dis: Disassembler for Python bytecode."><span class="pre"><code class="sourceCode python">dis</code></span></a> module. <a href="../library/dis.html#opcode-HAVE_ARGUMENT" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">HAVE_ARGUMENT</code></span></a> is still relevant to real opcodes, but it is not useful for pseudo instructions. Use the new <a href="../library/dis.html#dis.hasarg" class="reference internal" title="dis.hasarg"><span class="pre"><code class="sourceCode python">dis.hasarg</code></span></a> collection instead. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/94216" class="reference external">gh-94216</a>.)

- Add the <a href="../library/dis.html#dis.hasexc" class="reference internal" title="dis.hasexc"><span class="pre"><code class="sourceCode python">dis.hasexc</code></span></a> collection to signify instructions that set an exception handler. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/94216" class="reference external">gh-94216</a>.)

</div>

<div id="fractions" class="section">

### fractions<a href="#fractions" class="headerlink" title="Link to this heading">¶</a>

- Objects of type <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">fractions.Fraction</code></span></a> now support float-style formatting. (Contributed by Mark Dickinson in <a href="https://github.com/python/cpython/issues/100161" class="reference external">gh-100161</a>.)

</div>

<div id="importlib-resources" class="section">

### importlib.resources<a href="#importlib-resources" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/importlib.resources.html#importlib.resources.as_file" class="reference internal" title="importlib.resources.as_file"><span class="pre"><code class="sourceCode python">importlib.resources.as_file()</code></span></a> now supports resource directories. (Contributed by Jason R. Coombs in <a href="https://github.com/python/cpython/issues/97930" class="reference external">gh-97930</a>.)

- Rename first parameter of <a href="../library/importlib.resources.html#importlib.resources.files" class="reference internal" title="importlib.resources.files"><span class="pre"><code class="sourceCode python">importlib.resources.files()</code></span></a> to *anchor*. (Contributed by Jason R. Coombs in <a href="https://github.com/python/cpython/issues/100598" class="reference external">gh-100598</a>.)

</div>

<div id="inspect" class="section">

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/inspect.html#inspect.markcoroutinefunction" class="reference internal" title="inspect.markcoroutinefunction"><span class="pre"><code class="sourceCode python">inspect.markcoroutinefunction()</code></span></a> to mark sync functions that return a <a href="../glossary.html#term-coroutine" class="reference internal"><span class="xref std std-term">coroutine</span></a> for use with <a href="../library/inspect.html#inspect.iscoroutinefunction" class="reference internal" title="inspect.iscoroutinefunction"><span class="pre"><code class="sourceCode python">inspect.iscoroutinefunction()</code></span></a>. (Contributed by Carlton Gibson in <a href="https://github.com/python/cpython/issues/99247" class="reference external">gh-99247</a>.)

- Add <a href="../library/inspect.html#inspect.getasyncgenstate" class="reference internal" title="inspect.getasyncgenstate"><span class="pre"><code class="sourceCode python">inspect.getasyncgenstate()</code></span></a> and <a href="../library/inspect.html#inspect.getasyncgenlocals" class="reference internal" title="inspect.getasyncgenlocals"><span class="pre"><code class="sourceCode python">inspect.getasyncgenlocals()</code></span></a> for determining the current state of asynchronous generators. (Contributed by Thomas Krennwallner in <a href="https://github.com/python/cpython/issues/79940" class="reference external">gh-79940</a>.)

- The performance of <a href="../library/inspect.html#inspect.getattr_static" class="reference internal" title="inspect.getattr_static"><span class="pre"><code class="sourceCode python">inspect.getattr_static()</code></span></a> has been considerably improved. Most calls to the function should be at least 2x faster than they were in Python 3.11. (Contributed by Alex Waygood in <a href="https://github.com/python/cpython/issues/103193" class="reference external">gh-103193</a>.)

</div>

<div id="itertools" class="section">

### itertools<a href="#itertools" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/itertools.html#itertools.batched" class="reference internal" title="itertools.batched"><span class="pre"><code class="sourceCode python">itertools.batched()</code></span></a> for collecting into even-sized tuples where the last batch may be shorter than the rest. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/98363" class="reference external">gh-98363</a>.)

</div>

<div id="math" class="section">

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/math.html#math.sumprod" class="reference internal" title="math.sumprod"><span class="pre"><code class="sourceCode python">math.sumprod()</code></span></a> for computing a sum of products. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/100485" class="reference external">gh-100485</a>.)

- Extend <a href="../library/math.html#math.nextafter" class="reference internal" title="math.nextafter"><span class="pre"><code class="sourceCode python">math.nextafter()</code></span></a> to include a *steps* argument for moving up or down multiple steps at a time. (Contributed by Matthias Goergens, Mark Dickinson, and Raymond Hettinger in <a href="https://github.com/python/cpython/issues/94906" class="reference external">gh-94906</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/os.html#os.PIDFD_NONBLOCK" class="reference internal" title="os.PIDFD_NONBLOCK"><span class="pre"><code class="sourceCode python">os.PIDFD_NONBLOCK</code></span></a> to open a file descriptor for a process with <a href="../library/os.html#os.pidfd_open" class="reference internal" title="os.pidfd_open"><span class="pre"><code class="sourceCode python">os.pidfd_open()</code></span></a> in non-blocking mode. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/93312" class="reference external">gh-93312</a>.)

- <a href="../library/os.html#os.DirEntry" class="reference internal" title="os.DirEntry"><span class="pre"><code class="sourceCode python">os.DirEntry</code></span></a> now includes an <a href="../library/os.html#os.DirEntry.is_junction" class="reference internal" title="os.DirEntry.is_junction"><span class="pre"><code class="sourceCode python">os.DirEntry.is_junction()</code></span></a> method to check if the entry is a junction. (Contributed by Charles Machalow in <a href="https://github.com/python/cpython/issues/99547" class="reference external">gh-99547</a>.)

- Add <a href="../library/os.html#os.listdrives" class="reference internal" title="os.listdrives"><span class="pre"><code class="sourceCode python">os.listdrives()</code></span></a>, <a href="../library/os.html#os.listvolumes" class="reference internal" title="os.listvolumes"><span class="pre"><code class="sourceCode python">os.listvolumes()</code></span></a> and <a href="../library/os.html#os.listmounts" class="reference internal" title="os.listmounts"><span class="pre"><code class="sourceCode python">os.listmounts()</code></span></a> functions on Windows for enumerating drives, volumes and mount points. (Contributed by Steve Dower in <a href="https://github.com/python/cpython/issues/102519" class="reference external">gh-102519</a>.)

- <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> and <a href="../library/os.html#os.lstat" class="reference internal" title="os.lstat"><span class="pre"><code class="sourceCode python">os.lstat()</code></span></a> are now more accurate on Windows. The <span class="pre">`st_birthtime`</span> field will now be filled with the creation time of the file, and <span class="pre">`st_ctime`</span> is deprecated but still contains the creation time (but in the future will return the last metadata change, for consistency with other platforms). <span class="pre">`st_dev`</span> may be up to 64 bits and <span class="pre">`st_ino`</span> up to 128 bits depending on your file system, and <span class="pre">`st_rdev`</span> is always set to zero rather than incorrect values. Both functions may be significantly faster on newer releases of Windows. (Contributed by Steve Dower in <a href="https://github.com/python/cpython/issues/99726" class="reference external">gh-99726</a>.)

</div>

<div id="os-path" class="section">

### os.path<a href="#os-path" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/os.path.html#os.path.isjunction" class="reference internal" title="os.path.isjunction"><span class="pre"><code class="sourceCode python">os.path.isjunction()</code></span></a> to check if a given path is a junction. (Contributed by Charles Machalow in <a href="https://github.com/python/cpython/issues/99547" class="reference external">gh-99547</a>.)

- Add <a href="../library/os.path.html#os.path.splitroot" class="reference internal" title="os.path.splitroot"><span class="pre"><code class="sourceCode python">os.path.splitroot()</code></span></a> to split a path into a triad <span class="pre">`(drive,`</span>` `<span class="pre">`root,`</span>` `<span class="pre">`tail)`</span>. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/101000" class="reference external">gh-101000</a>.)

</div>

<div id="pathlib" class="section">

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

- Add support for subclassing <a href="../library/pathlib.html#pathlib.PurePath" class="reference internal" title="pathlib.PurePath"><span class="pre"><code class="sourceCode python">pathlib.PurePath</code></span></a> and <a href="../library/pathlib.html#pathlib.Path" class="reference internal" title="pathlib.Path"><span class="pre"><code class="sourceCode python">pathlib.Path</code></span></a>, plus their Posix- and Windows-specific variants. Subclasses may override the <a href="../library/pathlib.html#pathlib.PurePath.with_segments" class="reference internal" title="pathlib.PurePath.with_segments"><span class="pre"><code class="sourceCode python">pathlib.PurePath.with_segments()</code></span></a> method to pass information between path instances.

- Add <a href="../library/pathlib.html#pathlib.Path.walk" class="reference internal" title="pathlib.Path.walk"><span class="pre"><code class="sourceCode python">pathlib.Path.walk()</code></span></a> for walking the directory trees and generating all file or directory names within them, similar to <a href="../library/os.html#os.walk" class="reference internal" title="os.walk"><span class="pre"><code class="sourceCode python">os.walk()</code></span></a>. (Contributed by Stanislav Zmiev in <a href="https://github.com/python/cpython/issues/90385" class="reference external">gh-90385</a>.)

- Add *walk_up* optional parameter to <a href="../library/pathlib.html#pathlib.PurePath.relative_to" class="reference internal" title="pathlib.PurePath.relative_to"><span class="pre"><code class="sourceCode python">pathlib.PurePath.relative_to()</code></span></a> to allow the insertion of <span class="pre">`..`</span> entries in the result; this behavior is more consistent with <a href="../library/os.path.html#os.path.relpath" class="reference internal" title="os.path.relpath"><span class="pre"><code class="sourceCode python">os.path.relpath()</code></span></a>. (Contributed by Domenico Ragusa in <a href="https://github.com/python/cpython/issues/84538" class="reference external">gh-84538</a>.)

- Add <a href="../library/pathlib.html#pathlib.Path.is_junction" class="reference internal" title="pathlib.Path.is_junction"><span class="pre"><code class="sourceCode python">pathlib.Path.is_junction()</code></span></a> as a proxy to <a href="../library/os.path.html#os.path.isjunction" class="reference internal" title="os.path.isjunction"><span class="pre"><code class="sourceCode python">os.path.isjunction()</code></span></a>. (Contributed by Charles Machalow in <a href="https://github.com/python/cpython/issues/99547" class="reference external">gh-99547</a>.)

- Add *case_sensitive* optional parameter to <a href="../library/pathlib.html#pathlib.Path.glob" class="reference internal" title="pathlib.Path.glob"><span class="pre"><code class="sourceCode python">pathlib.Path.glob()</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.rglob" class="reference internal" title="pathlib.Path.rglob"><span class="pre"><code class="sourceCode python">pathlib.Path.rglob()</code></span></a> and <a href="../library/pathlib.html#pathlib.PurePath.match" class="reference internal" title="pathlib.PurePath.match"><span class="pre"><code class="sourceCode python">pathlib.PurePath.match()</code></span></a> for matching the path’s case sensitivity, allowing for more precise control over the matching process.

</div>

<div id="platform" class="section">

### platform<a href="#platform" class="headerlink" title="Link to this heading">¶</a>

- Add support for detecting Windows 11 and Windows Server releases past 2012. Previously, lookups on Windows Server platforms newer than Windows Server 2012 and on Windows 11 would return <span class="pre">`Windows-10`</span>. (Contributed by Steve Dower in <a href="https://github.com/python/cpython/issues/89545" class="reference external">gh-89545</a>.)

</div>

<div id="pdb" class="section">

### pdb<a href="#pdb" class="headerlink" title="Link to this heading">¶</a>

- Add convenience variables to hold values temporarily for debug session and provide quick access to values like the current frame or the return value. (Contributed by Tian Gao in <a href="https://github.com/python/cpython/issues/103693" class="reference external">gh-103693</a>.)

</div>

<div id="random" class="section">

### random<a href="#random" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/random.html#random.binomialvariate" class="reference internal" title="random.binomialvariate"><span class="pre"><code class="sourceCode python">random.binomialvariate()</code></span></a>. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/81620" class="reference external">gh-81620</a>.)

- Add a default of <span class="pre">`lambd=1.0`</span> to <a href="../library/random.html#random.expovariate" class="reference internal" title="random.expovariate"><span class="pre"><code class="sourceCode python">random.expovariate()</code></span></a>. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/100234" class="reference external">gh-100234</a>.)

</div>

<div id="shutil" class="section">

### shutil<a href="#shutil" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/shutil.html#shutil.make_archive" class="reference internal" title="shutil.make_archive"><span class="pre"><code class="sourceCode python">shutil.make_archive()</code></span></a> now passes the *root_dir* argument to custom archivers which support it. In this case it no longer temporarily changes the current working directory of the process to *root_dir* to perform archiving. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/74696" class="reference external">gh-74696</a>.)

- <a href="../library/shutil.html#shutil.rmtree" class="reference internal" title="shutil.rmtree"><span class="pre"><code class="sourceCode python">shutil.rmtree()</code></span></a> now accepts a new argument *onexc* which is an error handler like *onerror* but which expects an exception instance rather than a *(typ, val, tb)* triplet. *onerror* is deprecated. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/102828" class="reference external">gh-102828</a>.)

- <a href="../library/shutil.html#shutil.which" class="reference internal" title="shutil.which"><span class="pre"><code class="sourceCode python">shutil.which()</code></span></a> now consults the *PATHEXT* environment variable to find matches within *PATH* on Windows even when the given *cmd* includes a directory component. (Contributed by Charles Machalow in <a href="https://github.com/python/cpython/issues/103179" class="reference external">gh-103179</a>.)

  <a href="../library/shutil.html#shutil.which" class="reference internal" title="shutil.which"><span class="pre"><code class="sourceCode python">shutil.which()</code></span></a> will call <span class="pre">`NeedCurrentDirectoryForExePathW`</span> when querying for executables on Windows to determine if the current working directory should be prepended to the search path. (Contributed by Charles Machalow in <a href="https://github.com/python/cpython/issues/103179" class="reference external">gh-103179</a>.)

  <a href="../library/shutil.html#shutil.which" class="reference internal" title="shutil.which"><span class="pre"><code class="sourceCode python">shutil.which()</code></span></a> will return a path matching the *cmd* with a component from <span class="pre">`PATHEXT`</span> prior to a direct match elsewhere in the search path on Windows. (Contributed by Charles Machalow in <a href="https://github.com/python/cpython/issues/103179" class="reference external">gh-103179</a>.)

</div>

<div id="sqlite3" class="section">

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

- Add a <a href="../library/sqlite3.html#sqlite3-cli" class="reference internal"><span class="std std-ref">command-line interface</span></a>. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/77617" class="reference external">gh-77617</a>.)

- Add the <a href="../library/sqlite3.html#sqlite3.Connection.autocommit" class="reference internal" title="sqlite3.Connection.autocommit"><span class="pre"><code class="sourceCode python">sqlite3.Connection.autocommit</code></span></a> attribute to <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">sqlite3.Connection</code></span></a> and the *autocommit* parameter to <a href="../library/sqlite3.html#sqlite3.connect" class="reference internal" title="sqlite3.connect"><span class="pre"><code class="sourceCode python">sqlite3.<span class="ex">connect</span>()</code></span></a> to control <span id="index-25" class="target"></span><a href="https://peps.python.org/pep-0249/" class="pep reference external"><strong>PEP 249</strong></a>-compliant <a href="../library/sqlite3.html#sqlite3-transaction-control-autocommit" class="reference internal"><span class="std std-ref">transaction handling</span></a>. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/83638" class="reference external">gh-83638</a>.)

- Add *entrypoint* keyword-only parameter to <a href="../library/sqlite3.html#sqlite3.Connection.load_extension" class="reference internal" title="sqlite3.Connection.load_extension"><span class="pre"><code class="sourceCode python">sqlite3.Connection.load_extension()</code></span></a>, for overriding the SQLite extension entry point. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/103015" class="reference external">gh-103015</a>.)

- Add <a href="../library/sqlite3.html#sqlite3.Connection.getconfig" class="reference internal" title="sqlite3.Connection.getconfig"><span class="pre"><code class="sourceCode python">sqlite3.Connection.getconfig()</code></span></a> and <a href="../library/sqlite3.html#sqlite3.Connection.setconfig" class="reference internal" title="sqlite3.Connection.setconfig"><span class="pre"><code class="sourceCode python">sqlite3.Connection.setconfig()</code></span></a> to <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">sqlite3.Connection</code></span></a> to make configuration changes to a database connection. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/103489" class="reference external">gh-103489</a>.)

</div>

<div id="statistics" class="section">

### statistics<a href="#statistics" class="headerlink" title="Link to this heading">¶</a>

- Extend <a href="../library/statistics.html#statistics.correlation" class="reference internal" title="statistics.correlation"><span class="pre"><code class="sourceCode python">statistics.correlation()</code></span></a> to include as a <span class="pre">`ranked`</span> method for computing the Spearman correlation of ranked data. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/95861" class="reference external">gh-95861</a>.)

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

- Add the <a href="../library/sys.monitoring.html#module-sys.monitoring" class="reference internal" title="sys.monitoring: Access and control event monitoring"><span class="pre"><code class="sourceCode python">sys.monitoring</code></span></a> namespace to expose the new <a href="#whatsnew312-pep669" class="reference internal"><span class="std std-ref">PEP 669</span></a> monitoring API. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/103082" class="reference external">gh-103082</a>.)

- Add <a href="../library/sys.html#sys.activate_stack_trampoline" class="reference internal" title="sys.activate_stack_trampoline"><span class="pre"><code class="sourceCode python">sys.activate_stack_trampoline()</code></span></a> and <a href="../library/sys.html#sys.deactivate_stack_trampoline" class="reference internal" title="sys.deactivate_stack_trampoline"><span class="pre"><code class="sourceCode python">sys.deactivate_stack_trampoline()</code></span></a> for activating and deactivating stack profiler trampolines, and <a href="../library/sys.html#sys.is_stack_trampoline_active" class="reference internal" title="sys.is_stack_trampoline_active"><span class="pre"><code class="sourceCode python">sys.is_stack_trampoline_active()</code></span></a> for querying if stack profiler trampolines are active. (Contributed by Pablo Galindo and Christian Heimes with contributions from Gregory P. Smith \[Google\] and Mark Shannon in <a href="https://github.com/python/cpython/issues/96123" class="reference external">gh-96123</a>.)

- Add <a href="../library/sys.html#sys.last_exc" class="reference internal" title="sys.last_exc"><span class="pre"><code class="sourceCode python">sys.last_exc</code></span></a> which holds the last unhandled exception that was raised (for post-mortem debugging use cases). Deprecate the three fields that have the same information in its legacy form: <a href="../library/sys.html#sys.last_type" class="reference internal" title="sys.last_type"><span class="pre"><code class="sourceCode python">sys.last_type</code></span></a>, <a href="../library/sys.html#sys.last_value" class="reference internal" title="sys.last_value"><span class="pre"><code class="sourceCode python">sys.last_value</code></span></a> and <a href="../library/sys.html#sys.last_traceback" class="reference internal" title="sys.last_traceback"><span class="pre"><code class="sourceCode python">sys.last_traceback</code></span></a>. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/102778" class="reference external">gh-102778</a>.)

- <a href="../library/sys.html#sys._current_exceptions" class="reference internal" title="sys._current_exceptions"><span class="pre"><code class="sourceCode python">sys._current_exceptions()</code></span></a> now returns a mapping from thread-id to an exception instance, rather than to a <span class="pre">`(typ,`</span>` `<span class="pre">`exc,`</span>` `<span class="pre">`tb)`</span> tuple. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/103176" class="reference external">gh-103176</a>.)

- <a href="../library/sys.html#sys.setrecursionlimit" class="reference internal" title="sys.setrecursionlimit"><span class="pre"><code class="sourceCode python">sys.setrecursionlimit()</code></span></a> and <a href="../library/sys.html#sys.getrecursionlimit" class="reference internal" title="sys.getrecursionlimit"><span class="pre"><code class="sourceCode python">sys.getrecursionlimit()</code></span></a>. The recursion limit now applies only to Python code. Builtin functions do not use the recursion limit, but are protected by a different mechanism that prevents recursion from causing a virtual machine crash.

</div>

<div id="tempfile" class="section">

### tempfile<a href="#tempfile" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/tempfile.html#tempfile.NamedTemporaryFile" class="reference internal" title="tempfile.NamedTemporaryFile"><span class="pre"><code class="sourceCode python">tempfile.NamedTemporaryFile</code></span></a> function has a new optional parameter *delete_on_close* (Contributed by Evgeny Zorin in <a href="https://github.com/python/cpython/issues/58451" class="reference external">gh-58451</a>.)

- <a href="../library/tempfile.html#tempfile.mkdtemp" class="reference internal" title="tempfile.mkdtemp"><span class="pre"><code class="sourceCode python">tempfile.mkdtemp()</code></span></a> now always returns an absolute path, even if the argument provided to the *dir* parameter is a relative path.

</div>

<div id="threading" class="section">

### threading<a href="#threading" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/threading.html#threading.settrace_all_threads" class="reference internal" title="threading.settrace_all_threads"><span class="pre"><code class="sourceCode python">threading.settrace_all_threads()</code></span></a> and <a href="../library/threading.html#threading.setprofile_all_threads" class="reference internal" title="threading.setprofile_all_threads"><span class="pre"><code class="sourceCode python">threading.setprofile_all_threads()</code></span></a> that allow to set tracing and profiling functions in all running threads in addition to the calling one. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/93503" class="reference external">gh-93503</a>.)

</div>

<div id="tkinter" class="section">

### tkinter<a href="#tkinter" class="headerlink" title="Link to this heading">¶</a>

- <span class="pre">`tkinter.Canvas.coords()`</span> now flattens its arguments. It now accepts not only coordinates as separate arguments (<span class="pre">`x1,`</span>` `<span class="pre">`y1,`</span>` `<span class="pre">`x2,`</span>` `<span class="pre">`y2,`</span>` `<span class="pre">`...`</span>) and a sequence of coordinates (<span class="pre">`[x1,`</span>` `<span class="pre">`y1,`</span>` `<span class="pre">`x2,`</span>` `<span class="pre">`y2,`</span>` `<span class="pre">`...]`</span>), but also coordinates grouped in pairs (<span class="pre">`(x1,`</span>` `<span class="pre">`y1),`</span>` `<span class="pre">`(x2,`</span>` `<span class="pre">`y2),`</span>` `<span class="pre">`...`</span> and <span class="pre">`[(x1,`</span>` `<span class="pre">`y1),`</span>` `<span class="pre">`(x2,`</span>` `<span class="pre">`y2),`</span>` `<span class="pre">`...]`</span>), like <span class="pre">`create_*()`</span> methods. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/94473" class="reference external">gh-94473</a>.)

</div>

<div id="tokenize" class="section">

### tokenize<a href="#tokenize" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/tokenize.html#module-tokenize" class="reference internal" title="tokenize: Lexical scanner for Python source code."><span class="pre"><code class="sourceCode python">tokenize</code></span></a> module includes the changes introduced in <span id="index-26" class="target"></span><a href="https://peps.python.org/pep-0701/" class="pep reference external"><strong>PEP 701</strong></a>. (Contributed by Marta Gómez Macías and Pablo Galindo in <a href="https://github.com/python/cpython/issues/102856" class="reference external">gh-102856</a>.) See <a href="#whatsnew312-porting-to-python312" class="reference internal"><span class="std std-ref">Porting to Python 3.12</span></a> for more information on the changes to the <a href="../library/tokenize.html#module-tokenize" class="reference internal" title="tokenize: Lexical scanner for Python source code."><span class="pre"><code class="sourceCode python">tokenize</code></span></a> module.

</div>

<div id="types" class="section">

### types<a href="#types" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/types.html#types.get_original_bases" class="reference internal" title="types.get_original_bases"><span class="pre"><code class="sourceCode python">types.get_original_bases()</code></span></a> to allow for further introspection of <a href="../library/typing.html#user-defined-generics" class="reference internal"><span class="std std-ref">User-defined generic types</span></a> when subclassed. (Contributed by James Hilton-Balfe and Alex Waygood in <a href="https://github.com/python/cpython/issues/101827" class="reference external">gh-101827</a>.)

</div>

<div id="typing" class="section">

<span id="whatsnew-typing-py312"></span>

### typing<a href="#typing" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/functions.html#isinstance" class="reference internal" title="isinstance"><span class="pre"><code class="sourceCode python"><span class="bu">isinstance</span>()</code></span></a> checks against <a href="../library/typing.html#typing.runtime_checkable" class="reference internal" title="typing.runtime_checkable"><span class="pre"><code class="sourceCode python">runtime<span class="op">-</span>checkable</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">protocols</code></span></a> now use <a href="../library/inspect.html#inspect.getattr_static" class="reference internal" title="inspect.getattr_static"><span class="pre"><code class="sourceCode python">inspect.getattr_static()</code></span></a> rather than <a href="../library/functions.html#hasattr" class="reference internal" title="hasattr"><span class="pre"><code class="sourceCode python"><span class="bu">hasattr</span>()</code></span></a> to lookup whether attributes exist. This means that descriptors and <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> methods are no longer unexpectedly evaluated during <span class="pre">`isinstance()`</span> checks against runtime-checkable protocols. However, it may also mean that some objects which used to be considered instances of a runtime-checkable protocol may no longer be considered instances of that protocol on Python 3.12+, and vice versa. Most users are unlikely to be affected by this change. (Contributed by Alex Waygood in <a href="https://github.com/python/cpython/issues/102433" class="reference external">gh-102433</a>.)

- The members of a runtime-checkable protocol are now considered “frozen” at runtime as soon as the class has been created. Monkey-patching attributes onto a runtime-checkable protocol will still work, but will have no impact on <a href="../library/functions.html#isinstance" class="reference internal" title="isinstance"><span class="pre"><code class="sourceCode python"><span class="bu">isinstance</span>()</code></span></a> checks comparing objects to the protocol. For example:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> from typing import Protocol, runtime_checkable
      >>> @runtime_checkable
      ... class HasX(Protocol):
      ...     x = 1
      ...
      >>> class Foo: ...
      ...
      >>> f = Foo()
      >>> isinstance(f, HasX)
      False
      >>> f.x = 1
      >>> isinstance(f, HasX)
      True
      >>> HasX.y = 2
      >>> isinstance(f, HasX)  # unchanged, even though HasX now also has a "y" attribute
      True

  </div>

  </div>

  This change was made in order to speed up <span class="pre">`isinstance()`</span> checks against runtime-checkable protocols.

- The performance profile of <a href="../library/functions.html#isinstance" class="reference internal" title="isinstance"><span class="pre"><code class="sourceCode python"><span class="bu">isinstance</span>()</code></span></a> checks against <a href="../library/typing.html#typing.runtime_checkable" class="reference internal" title="typing.runtime_checkable"><span class="pre"><code class="sourceCode python">runtime<span class="op">-</span>checkable</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">protocols</code></span></a> has changed significantly. Most <span class="pre">`isinstance()`</span> checks against protocols with only a few members should be at least 2x faster than in 3.11, and some may be 20x faster or more. However, <span class="pre">`isinstance()`</span> checks against protocols with many members may be slower than in Python 3.11. (Contributed by Alex Waygood in <a href="https://github.com/python/cpython/issues/74690" class="reference external">gh-74690</a> and <a href="https://github.com/python/cpython/issues/103193" class="reference external">gh-103193</a>.)

- All <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">typing.TypedDict</code></span></a> and <a href="../library/typing.html#typing.NamedTuple" class="reference internal" title="typing.NamedTuple"><span class="pre"><code class="sourceCode python">typing.NamedTuple</code></span></a> classes now have the <span class="pre">`__orig_bases__`</span> attribute. (Contributed by Adrian Garcia Badaracco in <a href="https://github.com/python/cpython/issues/103699" class="reference external">gh-103699</a>.)

- Add <span class="pre">`frozen_default`</span> parameter to <a href="../library/typing.html#typing.dataclass_transform" class="reference internal" title="typing.dataclass_transform"><span class="pre"><code class="sourceCode python">typing.dataclass_transform()</code></span></a>. (Contributed by Erik De Bonte in <a href="https://github.com/python/cpython/issues/99957" class="reference external">gh-99957</a>.)

</div>

<div id="unicodedata" class="section">

### unicodedata<a href="#unicodedata" class="headerlink" title="Link to this heading">¶</a>

- The Unicode database has been updated to version 15.0.0. (Contributed by Benjamin Peterson in <a href="https://github.com/python/cpython/issues/96734" class="reference external">gh-96734</a>).

</div>

<div id="unittest" class="section">

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

Add a <span class="pre">`--durations`</span> command line option, showing the N slowest test cases:

<div class="highlight-python3 notranslate">

<div class="highlight">

    python3 -m unittest --durations=3 lib.tests.test_threading
    .....
    Slowest test durations
    ----------------------------------------------------------------------
    1.210s     test_timeout (Lib.test.test_threading.BarrierTests)
    1.003s     test_default_timeout (Lib.test.test_threading.BarrierTests)
    0.518s     test_timeout (Lib.test.test_threading.EventTests)

    (0.000 durations hidden.  Use -v to show these durations.)
    ----------------------------------------------------------------------
    Ran 158 tests in 9.869s

    OK (skipped=3)

</div>

</div>

(Contributed by Giampaolo Rodola in <a href="https://github.com/python/cpython/issues/48330" class="reference external">gh-48330</a>)

</div>

<div id="uuid" class="section">

### uuid<a href="#uuid" class="headerlink" title="Link to this heading">¶</a>

- Add a <a href="../library/uuid.html#uuid-cli" class="reference internal"><span class="std std-ref">command-line interface</span></a>. (Contributed by Adam Chhina in <a href="https://github.com/python/cpython/issues/88597" class="reference external">gh-88597</a>.)

</div>

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

- Remove <span class="pre">`wstr`</span> and <span class="pre">`wstr_length`</span> members from Unicode objects. It reduces object size by 8 or 16 bytes on 64bit platform. (<span id="index-27" class="target"></span><a href="https://peps.python.org/pep-0623/" class="pep reference external"><strong>PEP 623</strong></a>) (Contributed by Inada Naoki in <a href="https://github.com/python/cpython/issues/92536" class="reference external">gh-92536</a>.)

- Add experimental support for using the BOLT binary optimizer in the build process, which improves performance by 1-5%. (Contributed by Kevin Modzelewski in <a href="https://github.com/python/cpython/issues/90536" class="reference external">gh-90536</a> and tuned by Donghee Na in <a href="https://github.com/python/cpython/issues/101525" class="reference external">gh-101525</a>)

- Speed up the regular expression substitution (functions <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">re.sub()</code></span></a> and <a href="../library/re.html#re.subn" class="reference internal" title="re.subn"><span class="pre"><code class="sourceCode python">re.subn()</code></span></a> and corresponding <span class="pre">`re.Pattern`</span> methods) for replacement strings containing group references by 2–3 times. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/91524" class="reference external">gh-91524</a>.)

- Speed up <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">asyncio.Task</code></span></a> creation by deferring expensive string formatting. (Contributed by Itamar Oren in <a href="https://github.com/python/cpython/issues/103793" class="reference external">gh-103793</a>.)

- The <a href="../library/tokenize.html#tokenize.tokenize" class="reference internal" title="tokenize.tokenize"><span class="pre"><code class="sourceCode python">tokenize.tokenize()</code></span></a> and <a href="../library/tokenize.html#tokenize.generate_tokens" class="reference internal" title="tokenize.generate_tokens"><span class="pre"><code class="sourceCode python">tokenize.generate_tokens()</code></span></a> functions are up to 64% faster as a side effect of the changes required to cover <span id="index-28" class="target"></span><a href="https://peps.python.org/pep-0701/" class="pep reference external"><strong>PEP 701</strong></a> in the <a href="../library/tokenize.html#module-tokenize" class="reference internal" title="tokenize: Lexical scanner for Python source code."><span class="pre"><code class="sourceCode python">tokenize</code></span></a> module. (Contributed by Marta Gómez Macías and Pablo Galindo in <a href="https://github.com/python/cpython/issues/102856" class="reference external">gh-102856</a>.)

- Speed up <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a> method calls and attribute loads via the new <a href="../library/dis.html#opcode-LOAD_SUPER_ATTR" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_SUPER_ATTR</code></span></a> instruction. (Contributed by Carl Meyer and Vladimir Matveev in <a href="https://github.com/python/cpython/issues/103497" class="reference external">gh-103497</a>.)

</div>

<div id="cpython-bytecode-changes" class="section">

## CPython bytecode changes<a href="#cpython-bytecode-changes" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`LOAD_METHOD`</span> instruction. It has been merged into <a href="../library/dis.html#opcode-LOAD_ATTR" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_ATTR</code></span></a>. <a href="../library/dis.html#opcode-LOAD_ATTR" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_ATTR</code></span></a> will now behave like the old <span class="pre">`LOAD_METHOD`</span> instruction if the low bit of its oparg is set. (Contributed by Ken Jin in <a href="https://github.com/python/cpython/issues/93429" class="reference external">gh-93429</a>.)

- Remove the <span class="pre">`JUMP_IF_FALSE_OR_POP`</span> and <span class="pre">`JUMP_IF_TRUE_OR_POP`</span> instructions. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/102859" class="reference external">gh-102859</a>.)

- Remove the <span class="pre">`PRECALL`</span> instruction. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/92925" class="reference external">gh-92925</a>.)

- Add the <a href="../library/dis.html#opcode-BINARY_SLICE" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">BINARY_SLICE</code></span></a> and <a href="../library/dis.html#opcode-STORE_SLICE" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">STORE_SLICE</code></span></a> instructions. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/94163" class="reference external">gh-94163</a>.)

- Add the <a href="../library/dis.html#opcode-CALL_INTRINSIC_1" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">CALL_INTRINSIC_1</code></span></a> instructions. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/99005" class="reference external">gh-99005</a>.)

- Add the <a href="../library/dis.html#opcode-CALL_INTRINSIC_2" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">CALL_INTRINSIC_2</code></span></a> instruction. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/101799" class="reference external">gh-101799</a>.)

- Add the <a href="../library/dis.html#opcode-CLEANUP_THROW" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">CLEANUP_THROW</code></span></a> instruction. (Contributed by Brandt Bucher in <a href="https://github.com/python/cpython/issues/90997" class="reference external">gh-90997</a>.)

- Add the <span class="pre">`END_SEND`</span> instruction. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/103082" class="reference external">gh-103082</a>.)

- Add the <a href="../library/dis.html#opcode-LOAD_FAST_AND_CLEAR" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_FAST_AND_CLEAR</code></span></a> instruction as part of the implementation of <span id="index-29" class="target"></span><a href="https://peps.python.org/pep-0709/" class="pep reference external"><strong>PEP 709</strong></a>. (Contributed by Carl Meyer in <a href="https://github.com/python/cpython/issues/101441" class="reference external">gh-101441</a>.)

- Add the <a href="../library/dis.html#opcode-LOAD_FAST_CHECK" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_FAST_CHECK</code></span></a> instruction. (Contributed by Dennis Sweeney in <a href="https://github.com/python/cpython/issues/93143" class="reference external">gh-93143</a>.)

- Add the <a href="../library/dis.html#opcode-LOAD_FROM_DICT_OR_DEREF" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_FROM_DICT_OR_DEREF</code></span></a>, <a href="../library/dis.html#opcode-LOAD_FROM_DICT_OR_GLOBALS" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_FROM_DICT_OR_GLOBALS</code></span></a>, and <a href="../library/dis.html#opcode-LOAD_LOCALS" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_LOCALS</code></span></a> opcodes as part of the implementation of <span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0695/" class="pep reference external"><strong>PEP 695</strong></a>. Remove the <span class="pre">`LOAD_CLASSDEREF`</span> opcode, which can be replaced with <a href="../library/dis.html#opcode-LOAD_LOCALS" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_LOCALS</code></span></a> plus <a href="../library/dis.html#opcode-LOAD_FROM_DICT_OR_DEREF" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_FROM_DICT_OR_DEREF</code></span></a>. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/103764" class="reference external">gh-103764</a>.)

- Add the <a href="../library/dis.html#opcode-LOAD_SUPER_ATTR" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_SUPER_ATTR</code></span></a> instruction. (Contributed by Carl Meyer and Vladimir Matveev in <a href="https://github.com/python/cpython/issues/103497" class="reference external">gh-103497</a>.)

- Add the <a href="../library/dis.html#opcode-RETURN_CONST" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">RETURN_CONST</code></span></a> instruction. (Contributed by Wenyang Wang in <a href="https://github.com/python/cpython/issues/101632" class="reference external">gh-101632</a>.)

</div>

<div id="demos-and-tools" class="section">

## Demos and Tools<a href="#demos-and-tools" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`Tools/demo/`</span> directory which contained old demo scripts. A copy can be found in the <a href="https://github.com/gvanrossum/old-demos" class="reference external">old-demos project</a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/97681" class="reference external">gh-97681</a>.)

- Remove outdated example scripts of the <span class="pre">`Tools/scripts/`</span> directory. A copy can be found in the <a href="https://github.com/gvanrossum/old-demos" class="reference external">old-demos project</a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/97669" class="reference external">gh-97669</a>.)

</div>

<div id="deprecated" class="section">

## Deprecated<a href="#deprecated" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a>: The *type*, *choices*, and *metavar* parameters of <span class="pre">`argparse.BooleanOptionalAction`</span> are deprecated and will be removed in 3.14. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/92248" class="reference external">gh-92248</a>.)

- <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a>: The following <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> features have been deprecated in documentation since Python 3.8, now cause a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> to be emitted at runtime when they are accessed or used, and will be removed in Python 3.14:

  - <span class="pre">`ast.Num`</span>

  - <span class="pre">`ast.Str`</span>

  - <span class="pre">`ast.Bytes`</span>

  - <span class="pre">`ast.NameConstant`</span>

  - <span class="pre">`ast.Ellipsis`</span>

  Use <a href="../library/ast.html#ast.Constant" class="reference internal" title="ast.Constant"><span class="pre"><code class="sourceCode python">ast.Constant</code></span></a> instead. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/90953" class="reference external">gh-90953</a>.)

- <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>:

  - The child watcher classes <a href="../library/asyncio-policy.html#asyncio.MultiLoopChildWatcher" class="reference internal" title="asyncio.MultiLoopChildWatcher"><span class="pre"><code class="sourceCode python">asyncio.MultiLoopChildWatcher</code></span></a>, <a href="../library/asyncio-policy.html#asyncio.FastChildWatcher" class="reference internal" title="asyncio.FastChildWatcher"><span class="pre"><code class="sourceCode python">asyncio.FastChildWatcher</code></span></a>, <a href="../library/asyncio-policy.html#asyncio.AbstractChildWatcher" class="reference internal" title="asyncio.AbstractChildWatcher"><span class="pre"><code class="sourceCode python">asyncio.AbstractChildWatcher</code></span></a> and <a href="../library/asyncio-policy.html#asyncio.SafeChildWatcher" class="reference internal" title="asyncio.SafeChildWatcher"><span class="pre"><code class="sourceCode python">asyncio.SafeChildWatcher</code></span></a> are deprecated and will be removed in Python 3.14. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/94597" class="reference external">gh-94597</a>.)

  - <a href="../library/asyncio-policy.html#asyncio.set_child_watcher" class="reference internal" title="asyncio.set_child_watcher"><span class="pre"><code class="sourceCode python">asyncio.set_child_watcher()</code></span></a>, <a href="../library/asyncio-policy.html#asyncio.get_child_watcher" class="reference internal" title="asyncio.get_child_watcher"><span class="pre"><code class="sourceCode python">asyncio.get_child_watcher()</code></span></a>, <a href="../library/asyncio-policy.html#asyncio.AbstractEventLoopPolicy.set_child_watcher" class="reference internal" title="asyncio.AbstractEventLoopPolicy.set_child_watcher"><span class="pre"><code class="sourceCode python">asyncio.AbstractEventLoopPolicy.set_child_watcher()</code></span></a> and <a href="../library/asyncio-policy.html#asyncio.AbstractEventLoopPolicy.get_child_watcher" class="reference internal" title="asyncio.AbstractEventLoopPolicy.get_child_watcher"><span class="pre"><code class="sourceCode python">asyncio.AbstractEventLoopPolicy.get_child_watcher()</code></span></a> are deprecated and will be removed in Python 3.14. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/94597" class="reference external">gh-94597</a>.)

  - The <a href="../library/asyncio-eventloop.html#asyncio.get_event_loop" class="reference internal" title="asyncio.get_event_loop"><span class="pre"><code class="sourceCode python">get_event_loop()</code></span></a> method of the default event loop policy now emits a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> if there is no current event loop set and it decides to create one. (Contributed by Serhiy Storchaka and Guido van Rossum in <a href="https://github.com/python/cpython/issues/100160" class="reference external">gh-100160</a>.)

- <a href="../library/calendar.html#module-calendar" class="reference internal" title="calendar: Functions for working with calendars, including some emulation of the Unix cal program."><span class="pre"><code class="sourceCode python">calendar</code></span></a>: <span class="pre">`calendar.January`</span> and <span class="pre">`calendar.February`</span> constants are deprecated and replaced by <a href="../library/calendar.html#calendar.JANUARY" class="reference internal" title="calendar.JANUARY"><span class="pre"><code class="sourceCode python">calendar.JANUARY</code></span></a> and <a href="../library/calendar.html#calendar.FEBRUARY" class="reference internal" title="calendar.FEBRUARY"><span class="pre"><code class="sourceCode python">calendar.FEBRUARY</code></span></a>. (Contributed by Prince Roshan in <a href="https://github.com/python/cpython/issues/103636" class="reference external">gh-103636</a>.)

- <a href="../library/collections.abc.html#module-collections.abc" class="reference internal" title="collections.abc: Abstract base classes for containers"><span class="pre"><code class="sourceCode python">collections.abc</code></span></a>: Deprecated <a href="../library/collections.abc.html#collections.abc.ByteString" class="reference internal" title="collections.abc.ByteString"><span class="pre"><code class="sourceCode python">collections.abc.ByteString</code></span></a>.

  Use <span class="pre">`isinstance(obj,`</span>` `<span class="pre">`collections.abc.Buffer)`</span> to test if <span class="pre">`obj`</span> implements the <a href="../c-api/buffer.html#bufferobjects" class="reference internal"><span class="std std-ref">buffer protocol</span></a> at runtime. For use in type annotations, either use <a href="../library/collections.abc.html#collections.abc.Buffer" class="reference internal" title="collections.abc.Buffer"><span class="pre"><code class="sourceCode python">Buffer</code></span></a> or a union that explicitly specifies the types your code supports (e.g., <span class="pre">`bytes`</span>` `<span class="pre">`|`</span>` `<span class="pre">`bytearray`</span>` `<span class="pre">`|`</span>` `<span class="pre">`memoryview`</span>).

  <span class="pre">`ByteString`</span> was originally intended to be an abstract class that would serve as a supertype of both <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>. However, since the ABC never had any methods, knowing that an object was an instance of <span class="pre">`ByteString`</span> never actually told you anything useful about the object. Other common buffer types such as <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> were also never understood as subtypes of <span class="pre">`ByteString`</span> (either at runtime or by static type checkers).

  See <span id="index-31" class="target"></span><a href="https://peps.python.org/pep-0688/#current-options" class="pep reference external"><strong>PEP 688</strong></a> for more details. (Contributed by Shantanu Jain in <a href="https://github.com/python/cpython/issues/91896" class="reference external">gh-91896</a>.)

- <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a>: <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime.datetime</code></span></a>’s <a href="../library/datetime.html#datetime.datetime.utcnow" class="reference internal" title="datetime.datetime.utcnow"><span class="pre"><code class="sourceCode python">utcnow()</code></span></a> and <a href="../library/datetime.html#datetime.datetime.utcfromtimestamp" class="reference internal" title="datetime.datetime.utcfromtimestamp"><span class="pre"><code class="sourceCode python">utcfromtimestamp()</code></span></a> are deprecated and will be removed in a future version. Instead, use timezone-aware objects to represent datetimes in UTC: respectively, call <a href="../library/datetime.html#datetime.datetime.now" class="reference internal" title="datetime.datetime.now"><span class="pre"><code class="sourceCode python">now()</code></span></a> and <a href="../library/datetime.html#datetime.datetime.fromtimestamp" class="reference internal" title="datetime.datetime.fromtimestamp"><span class="pre"><code class="sourceCode python">fromtimestamp()</code></span></a> with the *tz* parameter set to <a href="../library/datetime.html#datetime.UTC" class="reference internal" title="datetime.UTC"><span class="pre"><code class="sourceCode python">datetime.UTC</code></span></a>. (Contributed by Paul Ganssle in <a href="https://github.com/python/cpython/issues/103857" class="reference external">gh-103857</a>.)

- <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages."><span class="pre"><code class="sourceCode python">email</code></span></a>: Deprecate the *isdst* parameter in <a href="../library/email.utils.html#email.utils.localtime" class="reference internal" title="email.utils.localtime"><span class="pre"><code class="sourceCode python">email.utils.localtime()</code></span></a>. (Contributed by Alan Williams in <a href="https://github.com/python/cpython/issues/72346" class="reference external">gh-72346</a>.)

- <a href="../library/importlib.html#module-importlib.abc" class="reference internal" title="importlib.abc: Abstract base classes related to import"><span class="pre"><code class="sourceCode python">importlib.abc</code></span></a>: Deprecated the following classes, scheduled for removal in Python 3.14:

  - <span class="pre">`importlib.abc.ResourceReader`</span>

  - <span class="pre">`importlib.abc.Traversable`</span>

  - <span class="pre">`importlib.abc.TraversableResources`</span>

  Use <a href="../library/importlib.resources.abc.html#module-importlib.resources.abc" class="reference internal" title="importlib.resources.abc: Abstract base classes for resources"><span class="pre"><code class="sourceCode python">importlib.resources.abc</code></span></a> classes instead:

  - <a href="../library/importlib.resources.abc.html#importlib.resources.abc.Traversable" class="reference internal" title="importlib.resources.abc.Traversable"><span class="pre"><code class="sourceCode python">importlib.resources.abc.Traversable</code></span></a>

  - <a href="../library/importlib.resources.abc.html#importlib.resources.abc.TraversableResources" class="reference internal" title="importlib.resources.abc.TraversableResources"><span class="pre"><code class="sourceCode python">importlib.resources.abc.TraversableResources</code></span></a>

  (Contributed by Jason R. Coombs and Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/93963" class="reference external">gh-93963</a>.)

- <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a>: Deprecate the support for copy, deepcopy, and pickle operations, which is undocumented, inefficient, historically buggy, and inconsistent. This will be removed in 3.14 for a significant reduction in code volume and maintenance burden. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/101588" class="reference external">gh-101588</a>.)

- <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a>: In Python 3.14, the default <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> start method will change to a safer one on Linux, BSDs, and other non-macOS POSIX platforms where <span class="pre">`'fork'`</span> is currently the default (<a href="https://github.com/python/cpython/issues/84559" class="reference external">gh-84559</a>). Adding a runtime warning about this was deemed too disruptive as the majority of code is not expected to care. Use the <a href="../library/multiprocessing.html#multiprocessing.get_context" class="reference internal" title="multiprocessing.get_context"><span class="pre"><code class="sourceCode python">get_context()</code></span></a> or <a href="../library/multiprocessing.html#multiprocessing.set_start_method" class="reference internal" title="multiprocessing.set_start_method"><span class="pre"><code class="sourceCode python">set_start_method()</code></span></a> APIs to explicitly specify when your code *requires* <span class="pre">`'fork'`</span>. See <a href="../library/multiprocessing.html#multiprocessing-start-methods" class="reference internal"><span class="std std-ref">contexts and start methods</span></a>.

- <a href="../library/pkgutil.html#module-pkgutil" class="reference internal" title="pkgutil: Utilities for the import system."><span class="pre"><code class="sourceCode python">pkgutil</code></span></a>: <a href="../library/pkgutil.html#pkgutil.find_loader" class="reference internal" title="pkgutil.find_loader"><span class="pre"><code class="sourceCode python">pkgutil.find_loader()</code></span></a> and <a href="../library/pkgutil.html#pkgutil.get_loader" class="reference internal" title="pkgutil.get_loader"><span class="pre"><code class="sourceCode python">pkgutil.get_loader()</code></span></a> are deprecated and will be removed in Python 3.14; use <a href="../library/importlib.html#importlib.util.find_spec" class="reference internal" title="importlib.util.find_spec"><span class="pre"><code class="sourceCode python">importlib.util.find_spec()</code></span></a> instead. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/97850" class="reference external">gh-97850</a>.)

- <a href="../library/pty.html#module-pty" class="reference internal" title="pty: Pseudo-Terminal Handling for Unix. (Unix)"><span class="pre"><code class="sourceCode python">pty</code></span></a>: The module has two undocumented <span class="pre">`master_open()`</span> and <span class="pre">`slave_open()`</span> functions that have been deprecated since Python 2 but only gained a proper <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> in 3.12. Remove them in 3.14. (Contributed by Soumendra Ganguly and Gregory P. Smith in <a href="https://github.com/python/cpython/issues/85984" class="reference external">gh-85984</a>.)

- <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a>:

  - The <span class="pre">`st_ctime`</span> fields return by <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> and <a href="../library/os.html#os.lstat" class="reference internal" title="os.lstat"><span class="pre"><code class="sourceCode python">os.lstat()</code></span></a> on Windows are deprecated. In a future release, they will contain the last metadata change time, consistent with other platforms. For now, they still contain the creation time, which is also available in the new <span class="pre">`st_birthtime`</span> field. (Contributed by Steve Dower in <a href="https://github.com/python/cpython/issues/99726" class="reference external">gh-99726</a>.)

  - On POSIX platforms, <a href="../library/os.html#os.fork" class="reference internal" title="os.fork"><span class="pre"><code class="sourceCode python">os.fork()</code></span></a> can now raise a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> when it can detect being called from a multithreaded process. There has always been a fundamental incompatibility with the POSIX platform when doing so. Even if such code *appeared* to work. We added the warning to raise awareness as issues encountered by code doing this are becoming more frequent. See the <a href="../library/os.html#os.fork" class="reference internal" title="os.fork"><span class="pre"><code class="sourceCode python">os.fork()</code></span></a> documentation for more details along with <a href="https://discuss.python.org/t/concerns-regarding-deprecation-of-fork-with-alive-threads/33555" class="reference external">this discussion on fork being incompatible with threads</a> for *why* we’re now surfacing this longstanding platform compatibility problem to developers.

  When this warning appears due to usage of <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> or <a href="../library/concurrent.futures.html#module-concurrent.futures" class="reference internal" title="concurrent.futures: Execute computations concurrently using threads or processes."><span class="pre"><code class="sourceCode python">concurrent.futures</code></span></a> the fix is to use a different <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> start method such as <span class="pre">`"spawn"`</span> or <span class="pre">`"forkserver"`</span>.

- <a href="../library/shutil.html#module-shutil" class="reference internal" title="shutil: High-level file operations, including copying."><span class="pre"><code class="sourceCode python">shutil</code></span></a>: The *onerror* argument of <a href="../library/shutil.html#shutil.rmtree" class="reference internal" title="shutil.rmtree"><span class="pre"><code class="sourceCode python">shutil.rmtree()</code></span></a> is deprecated; use *onexc* instead. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/102828" class="reference external">gh-102828</a>.)

- <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a>:

  - <a href="../library/sqlite3.html#sqlite3-default-converters" class="reference internal"><span class="std std-ref">default adapters and converters</span></a> are now deprecated. Instead, use the <a href="../library/sqlite3.html#sqlite3-adapter-converter-recipes" class="reference internal"><span class="std std-ref">Adapter and converter recipes</span></a> and tailor them to your needs. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/90016" class="reference external">gh-90016</a>.)

  - In <a href="../library/sqlite3.html#sqlite3.Cursor.execute" class="reference internal" title="sqlite3.Cursor.execute"><span class="pre"><code class="sourceCode python">execute()</code></span></a>, <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> is now emitted when <a href="../library/sqlite3.html#sqlite3-placeholders" class="reference internal"><span class="std std-ref">named placeholders</span></a> are used together with parameters supplied as a <a href="../glossary.html#term-sequence" class="reference internal"><span class="xref std std-term">sequence</span></a> instead of as a <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a>. Starting from Python 3.14, using named placeholders with parameters supplied as a sequence will raise a <a href="../library/sqlite3.html#sqlite3.ProgrammingError" class="reference internal" title="sqlite3.ProgrammingError"><span class="pre"><code class="sourceCode python">ProgrammingError</code></span></a>. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/101698" class="reference external">gh-101698</a>.)

- <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a>: The <a href="../library/sys.html#sys.last_type" class="reference internal" title="sys.last_type"><span class="pre"><code class="sourceCode python">sys.last_type</code></span></a>, <a href="../library/sys.html#sys.last_value" class="reference internal" title="sys.last_value"><span class="pre"><code class="sourceCode python">sys.last_value</code></span></a> and <a href="../library/sys.html#sys.last_traceback" class="reference internal" title="sys.last_traceback"><span class="pre"><code class="sourceCode python">sys.last_traceback</code></span></a> fields are deprecated. Use <a href="../library/sys.html#sys.last_exc" class="reference internal" title="sys.last_exc"><span class="pre"><code class="sourceCode python">sys.last_exc</code></span></a> instead. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/102778" class="reference external">gh-102778</a>.)

- <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>: Extracting tar archives without specifying *filter* is deprecated until Python 3.14, when <span class="pre">`'data'`</span> filter will become the default. See <a href="../library/tarfile.html#tarfile-extraction-filter" class="reference internal"><span class="std std-ref">Extraction filters</span></a> for details.

- <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a>:

  - <a href="../library/typing.html#typing.Hashable" class="reference internal" title="typing.Hashable"><span class="pre"><code class="sourceCode python">typing.Hashable</code></span></a> and <a href="../library/typing.html#typing.Sized" class="reference internal" title="typing.Sized"><span class="pre"><code class="sourceCode python">typing.Sized</code></span></a>, aliases for <a href="../library/collections.abc.html#collections.abc.Hashable" class="reference internal" title="collections.abc.Hashable"><span class="pre"><code class="sourceCode python">collections.abc.Hashable</code></span></a> and <a href="../library/collections.abc.html#collections.abc.Sized" class="reference internal" title="collections.abc.Sized"><span class="pre"><code class="sourceCode python">collections.abc.Sized</code></span></a> respectively, are deprecated. (<a href="https://github.com/python/cpython/issues/94309" class="reference external">gh-94309</a>.)

  - <a href="../library/typing.html#typing.ByteString" class="reference internal" title="typing.ByteString"><span class="pre"><code class="sourceCode python">typing.ByteString</code></span></a>, deprecated since Python 3.9, now causes a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> to be emitted when it is used. (Contributed by Alex Waygood in <a href="https://github.com/python/cpython/issues/91896" class="reference external">gh-91896</a>.)

- <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a>: The module now emits <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> when testing the truth value of an <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element" class="reference internal" title="xml.etree.ElementTree.Element"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.Element</code></span></a>. Before, the Python implementation emitted <a href="../library/exceptions.html#FutureWarning" class="reference internal" title="FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a>, and the C implementation emitted nothing. (Contributed by Jacob Walls in <a href="https://github.com/python/cpython/issues/83122" class="reference external">gh-83122</a>.)

- The 3-arg signatures (type, value, traceback) of <a href="../reference/datamodel.html#coroutine.throw" class="reference internal" title="coroutine.throw"><span class="pre"><code class="sourceCode python">coroutine</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">throw()</code></span></a>, <a href="../reference/expressions.html#generator.throw" class="reference internal" title="generator.throw"><span class="pre"><code class="sourceCode python">generator</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">throw()</code></span></a> and <a href="../reference/expressions.html#agen.athrow" class="reference internal" title="agen.athrow"><span class="pre"><code class="sourceCode python"><span class="cf">async</span></code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">generator</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">throw()</code></span></a> are deprecated and may be removed in a future version of Python. Use the single-arg versions of these functions instead. (Contributed by Ofey Chan in <a href="https://github.com/python/cpython/issues/89874" class="reference external">gh-89874</a>.)

- <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> is now raised when <a href="../reference/datamodel.html#module.__package__" class="reference internal" title="module.__package__"><span class="pre"><code class="sourceCode python">__package__</code></span></a> on a module differs from <a href="../library/importlib.html#importlib.machinery.ModuleSpec.parent" class="reference internal" title="importlib.machinery.ModuleSpec.parent"><span class="pre"><code class="sourceCode python">__spec__.parent</code></span></a> (previously it was <a href="../library/exceptions.html#ImportWarning" class="reference internal" title="ImportWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ImportWarning</span></code></span></a>). (Contributed by Brett Cannon in <a href="https://github.com/python/cpython/issues/65961" class="reference external">gh-65961</a>.)

- Setting <a href="../reference/datamodel.html#module.__package__" class="reference internal" title="module.__package__"><span class="pre"><code class="sourceCode python">__package__</code></span></a> or <a href="../reference/datamodel.html#module.__cached__" class="reference internal" title="module.__cached__"><span class="pre"><code class="sourceCode python">__cached__</code></span></a> on a module is deprecated, and will cease to be set or taken into consideration by the import system in Python 3.14. (Contributed by Brett Cannon in <a href="https://github.com/python/cpython/issues/65961" class="reference external">gh-65961</a>.)

- The bitwise inversion operator (<span class="pre">`~`</span>) on bool is deprecated. It will throw an error in Python 3.16. Use <span class="pre">`not`</span> for logical negation of bools instead. In the rare case that you really need the bitwise inversion of the underlying <span class="pre">`int`</span>, convert to int explicitly: <span class="pre">`~int(x)`</span>. (Contributed by Tim Hoffmann in <a href="https://github.com/python/cpython/issues/103487" class="reference external">gh-103487</a>.)

- Accessing <a href="../reference/datamodel.html#codeobject.co_lnotab" class="reference internal" title="codeobject.co_lnotab"><span class="pre"><code class="sourceCode python">co_lnotab</code></span></a> on code objects was deprecated in Python 3.10 via <span id="index-32" class="target"></span><a href="https://peps.python.org/pep-0626/" class="pep reference external"><strong>PEP 626</strong></a>, but it only got a proper <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> in 3.12. May be removed in 3.15. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/101866" class="reference external">gh-101866</a>.)

<div id="pending-removal-in-python-3-13" class="section">

### Pending Removal in Python 3.13<a href="#pending-removal-in-python-3-13" class="headerlink" title="Link to this heading">¶</a>

Modules (see <span id="index-33" class="target"></span><a href="https://peps.python.org/pep-0594/" class="pep reference external"><strong>PEP 594</strong></a>):

- <span class="pre">`aifc`</span>

- <span class="pre">`audioop`</span>

- <span class="pre">`cgi`</span>

- <span class="pre">`cgitb`</span>

- <span class="pre">`chunk`</span>

- <span class="pre">`crypt`</span>

- <span class="pre">`imghdr`</span>

- <span class="pre">`mailcap`</span>

- <span class="pre">`msilib`</span>

- <span class="pre">`nis`</span>

- <span class="pre">`nntplib`</span>

- <span class="pre">`ossaudiodev`</span>

- <span class="pre">`pipes`</span>

- <span class="pre">`sndhdr`</span>

- <span class="pre">`spwd`</span>

- <span class="pre">`sunau`</span>

- <span class="pre">`telnetlib`</span>

- <span class="pre">`uu`</span>

- <span class="pre">`xdrlib`</span>

Other modules:

- <span class="pre">`lib2to3`</span>, and the **2to3** program (<a href="https://github.com/python/cpython/issues/84540" class="reference external">gh-84540</a>)

APIs:

- <span class="pre">`configparser.LegacyInterpolation`</span> (<a href="https://github.com/python/cpython/issues/90765" class="reference external">gh-90765</a>)

- <span class="pre">`locale.resetlocale()`</span> (<a href="https://github.com/python/cpython/issues/90817" class="reference external">gh-90817</a>)

- <span class="pre">`turtle.RawTurtle.settiltangle()`</span> (<a href="https://github.com/python/cpython/issues/50096" class="reference external">gh-50096</a>)

- <span class="pre">`unittest.findTestCases()`</span> (<a href="https://github.com/python/cpython/issues/50096" class="reference external">gh-50096</a>)

- <span class="pre">`unittest.getTestCaseNames()`</span> (<a href="https://github.com/python/cpython/issues/50096" class="reference external">gh-50096</a>)

- <span class="pre">`unittest.makeSuite()`</span> (<a href="https://github.com/python/cpython/issues/50096" class="reference external">gh-50096</a>)

- <span class="pre">`unittest.TestProgram.usageExit()`</span> (<a href="https://github.com/python/cpython/issues/67048" class="reference external">gh-67048</a>)

- <span class="pre">`webbrowser.MacOSX`</span> (<a href="https://github.com/python/cpython/issues/86421" class="reference external">gh-86421</a>)

- <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span></code></span></a> descriptor chaining (<a href="https://github.com/python/cpython/issues/89519" class="reference external">gh-89519</a>)

- <a href="../library/importlib.resources.html#module-importlib.resources" class="reference internal" title="importlib.resources: Package resource reading, opening, and access"><span class="pre"><code class="sourceCode python">importlib.resources</code></span></a> deprecated methods:

  - <span class="pre">`contents()`</span>

  - <span class="pre">`is_resource()`</span>

  - <span class="pre">`open_binary()`</span>

  - <span class="pre">`open_text()`</span>

  - <span class="pre">`path()`</span>

  - <span class="pre">`read_binary()`</span>

  - <span class="pre">`read_text()`</span>

  Use <a href="../library/importlib.resources.html#importlib.resources.files" class="reference internal" title="importlib.resources.files"><span class="pre"><code class="sourceCode python">importlib.resources.files()</code></span></a> instead. Refer to <a href="https://importlib-resources.readthedocs.io/en/latest/using.html#migrating-from-legacy" class="reference external">importlib-resources: Migrating from Legacy</a> (<a href="https://github.com/python/cpython/issues/106531" class="reference external">gh-106531</a>)

</div>

<div id="pending-removal-in-python-3-14" class="section">

### Pending Removal in Python 3.14<a href="#pending-removal-in-python-3-14" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a>: The *type*, *choices*, and *metavar* parameters of <span class="pre">`argparse.BooleanOptionalAction`</span> are deprecated and will be removed in 3.14. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/92248" class="reference external">gh-92248</a>.)

- <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a>: The following features have been deprecated in documentation since Python 3.8, now cause a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> to be emitted at runtime when they are accessed or used, and will be removed in Python 3.14:

  - <span class="pre">`ast.Num`</span>

  - <span class="pre">`ast.Str`</span>

  - <span class="pre">`ast.Bytes`</span>

  - <span class="pre">`ast.NameConstant`</span>

  - <span class="pre">`ast.Ellipsis`</span>

  Use <a href="../library/ast.html#ast.Constant" class="reference internal" title="ast.Constant"><span class="pre"><code class="sourceCode python">ast.Constant</code></span></a> instead. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/90953" class="reference external">gh-90953</a>.)

- <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>:

  - The child watcher classes <a href="../library/asyncio-policy.html#asyncio.MultiLoopChildWatcher" class="reference internal" title="asyncio.MultiLoopChildWatcher"><span class="pre"><code class="sourceCode python">MultiLoopChildWatcher</code></span></a>, <a href="../library/asyncio-policy.html#asyncio.FastChildWatcher" class="reference internal" title="asyncio.FastChildWatcher"><span class="pre"><code class="sourceCode python">FastChildWatcher</code></span></a>, <a href="../library/asyncio-policy.html#asyncio.AbstractChildWatcher" class="reference internal" title="asyncio.AbstractChildWatcher"><span class="pre"><code class="sourceCode python">AbstractChildWatcher</code></span></a> and <a href="../library/asyncio-policy.html#asyncio.SafeChildWatcher" class="reference internal" title="asyncio.SafeChildWatcher"><span class="pre"><code class="sourceCode python">SafeChildWatcher</code></span></a> are deprecated and will be removed in Python 3.14. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/94597" class="reference external">gh-94597</a>.)

  - <a href="../library/asyncio-policy.html#asyncio.set_child_watcher" class="reference internal" title="asyncio.set_child_watcher"><span class="pre"><code class="sourceCode python">asyncio.set_child_watcher()</code></span></a>, <a href="../library/asyncio-policy.html#asyncio.get_child_watcher" class="reference internal" title="asyncio.get_child_watcher"><span class="pre"><code class="sourceCode python">asyncio.get_child_watcher()</code></span></a>, <a href="../library/asyncio-policy.html#asyncio.AbstractEventLoopPolicy.set_child_watcher" class="reference internal" title="asyncio.AbstractEventLoopPolicy.set_child_watcher"><span class="pre"><code class="sourceCode python">asyncio.AbstractEventLoopPolicy.set_child_watcher()</code></span></a> and <a href="../library/asyncio-policy.html#asyncio.AbstractEventLoopPolicy.get_child_watcher" class="reference internal" title="asyncio.AbstractEventLoopPolicy.get_child_watcher"><span class="pre"><code class="sourceCode python">asyncio.AbstractEventLoopPolicy.get_child_watcher()</code></span></a> are deprecated and will be removed in Python 3.14. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/94597" class="reference external">gh-94597</a>.)

  - The <a href="../library/asyncio-eventloop.html#asyncio.get_event_loop" class="reference internal" title="asyncio.get_event_loop"><span class="pre"><code class="sourceCode python">get_event_loop()</code></span></a> method of the default event loop policy now emits a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> if there is no current event loop set and it decides to create one. (Contributed by Serhiy Storchaka and Guido van Rossum in <a href="https://github.com/python/cpython/issues/100160" class="reference external">gh-100160</a>.)

- <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages."><span class="pre"><code class="sourceCode python">email</code></span></a>: Deprecated the *isdst* parameter in <a href="../library/email.utils.html#email.utils.localtime" class="reference internal" title="email.utils.localtime"><span class="pre"><code class="sourceCode python">email.utils.localtime()</code></span></a>. (Contributed by Alan Williams in <a href="https://github.com/python/cpython/issues/72346" class="reference external">gh-72346</a>.)

- <a href="../library/importlib.html#module-importlib.abc" class="reference internal" title="importlib.abc: Abstract base classes related to import"><span class="pre"><code class="sourceCode python">importlib.abc</code></span></a> deprecated classes:

  - <span class="pre">`importlib.abc.ResourceReader`</span>

  - <span class="pre">`importlib.abc.Traversable`</span>

  - <span class="pre">`importlib.abc.TraversableResources`</span>

  Use <a href="../library/importlib.resources.abc.html#module-importlib.resources.abc" class="reference internal" title="importlib.resources.abc: Abstract base classes for resources"><span class="pre"><code class="sourceCode python">importlib.resources.abc</code></span></a> classes instead:

  - <a href="../library/importlib.resources.abc.html#importlib.resources.abc.Traversable" class="reference internal" title="importlib.resources.abc.Traversable"><span class="pre"><code class="sourceCode python">importlib.resources.abc.Traversable</code></span></a>

  - <a href="../library/importlib.resources.abc.html#importlib.resources.abc.TraversableResources" class="reference internal" title="importlib.resources.abc.TraversableResources"><span class="pre"><code class="sourceCode python">importlib.resources.abc.TraversableResources</code></span></a>

  (Contributed by Jason R. Coombs and Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/93963" class="reference external">gh-93963</a>.)

- <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> had undocumented, inefficient, historically buggy, and inconsistent support for copy, deepcopy, and pickle operations. This will be removed in 3.14 for a significant reduction in code volume and maintenance burden. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/101588" class="reference external">gh-101588</a>.)

- <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a>: The default start method will change to a safer one on Linux, BSDs, and other non-macOS POSIX platforms where <span class="pre">`'fork'`</span> is currently the default (<a href="https://github.com/python/cpython/issues/84559" class="reference external">gh-84559</a>). Adding a runtime warning about this was deemed too disruptive as the majority of code is not expected to care. Use the <a href="../library/multiprocessing.html#multiprocessing.get_context" class="reference internal" title="multiprocessing.get_context"><span class="pre"><code class="sourceCode python">get_context()</code></span></a> or <a href="../library/multiprocessing.html#multiprocessing.set_start_method" class="reference internal" title="multiprocessing.set_start_method"><span class="pre"><code class="sourceCode python">set_start_method()</code></span></a> APIs to explicitly specify when your code *requires* <span class="pre">`'fork'`</span>. See <a href="../library/multiprocessing.html#multiprocessing-start-methods" class="reference internal"><span class="std std-ref">Contexts and start methods</span></a>.

- <a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a>: <a href="../library/pathlib.html#pathlib.PurePath.is_relative_to" class="reference internal" title="pathlib.PurePath.is_relative_to"><span class="pre"><code class="sourceCode python">is_relative_to()</code></span></a> and <a href="../library/pathlib.html#pathlib.PurePath.relative_to" class="reference internal" title="pathlib.PurePath.relative_to"><span class="pre"><code class="sourceCode python">relative_to()</code></span></a>: passing additional arguments is deprecated.

- <a href="../library/pkgutil.html#module-pkgutil" class="reference internal" title="pkgutil: Utilities for the import system."><span class="pre"><code class="sourceCode python">pkgutil</code></span></a>: <a href="../library/pkgutil.html#pkgutil.find_loader" class="reference internal" title="pkgutil.find_loader"><span class="pre"><code class="sourceCode python">find_loader()</code></span></a> and <a href="../library/pkgutil.html#pkgutil.get_loader" class="reference internal" title="pkgutil.get_loader"><span class="pre"><code class="sourceCode python">get_loader()</code></span></a> now raise <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>; use <a href="../library/importlib.html#importlib.util.find_spec" class="reference internal" title="importlib.util.find_spec"><span class="pre"><code class="sourceCode python">importlib.util.find_spec()</code></span></a> instead. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/97850" class="reference external">gh-97850</a>.)

- <a href="../library/pty.html#module-pty" class="reference internal" title="pty: Pseudo-Terminal Handling for Unix. (Unix)"><span class="pre"><code class="sourceCode python">pty</code></span></a>:

  - <span class="pre">`master_open()`</span>: use <a href="../library/pty.html#pty.openpty" class="reference internal" title="pty.openpty"><span class="pre"><code class="sourceCode python">pty.openpty()</code></span></a>.

  - <span class="pre">`slave_open()`</span>: use <a href="../library/pty.html#pty.openpty" class="reference internal" title="pty.openpty"><span class="pre"><code class="sourceCode python">pty.openpty()</code></span></a>.

- <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a>:

  - <a href="../library/sqlite3.html#sqlite3.version" class="reference internal" title="sqlite3.version"><span class="pre"><code class="sourceCode python">version</code></span></a> and <a href="../library/sqlite3.html#sqlite3.version_info" class="reference internal" title="sqlite3.version_info"><span class="pre"><code class="sourceCode python">version_info</code></span></a>.

  - <a href="../library/sqlite3.html#sqlite3.Cursor.execute" class="reference internal" title="sqlite3.Cursor.execute"><span class="pre"><code class="sourceCode python">execute()</code></span></a> and <a href="../library/sqlite3.html#sqlite3.Cursor.executemany" class="reference internal" title="sqlite3.Cursor.executemany"><span class="pre"><code class="sourceCode python">executemany()</code></span></a> if <a href="../library/sqlite3.html#sqlite3-placeholders" class="reference internal"><span class="std std-ref">named placeholders</span></a> are used and *parameters* is a sequence instead of a <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a>.

- <a href="../library/urllib.html#module-urllib" class="reference internal" title="urllib"><span class="pre"><code class="sourceCode python">urllib</code></span></a>: <span class="pre">`urllib.parse.Quoter`</span> is deprecated: it was not intended to be a public API. (Contributed by Gregory P. Smith in <a href="https://github.com/python/cpython/issues/88168" class="reference external">gh-88168</a>.)

</div>

<div id="pending-removal-in-python-3-15" class="section">

### Pending Removal in Python 3.15<a href="#pending-removal-in-python-3-15" class="headerlink" title="Link to this heading">¶</a>

- The import system:

  - Setting <a href="../reference/datamodel.html#module.__cached__" class="reference internal" title="module.__cached__"><span class="pre"><code class="sourceCode python">__cached__</code></span></a> on a module while failing to set <a href="../library/importlib.html#importlib.machinery.ModuleSpec.cached" class="reference internal" title="importlib.machinery.ModuleSpec.cached"><span class="pre"><code class="sourceCode python">__spec__.cached</code></span></a> is deprecated. In Python 3.15, <span class="pre">`__cached__`</span> will cease to be set or take into consideration by the import system or standard library. (<a href="https://github.com/python/cpython/issues/97879" class="reference external">gh-97879</a>)

  - Setting <a href="../reference/datamodel.html#module.__package__" class="reference internal" title="module.__package__"><span class="pre"><code class="sourceCode python">__package__</code></span></a> on a module while failing to set <a href="../library/importlib.html#importlib.machinery.ModuleSpec.parent" class="reference internal" title="importlib.machinery.ModuleSpec.parent"><span class="pre"><code class="sourceCode python">__spec__.parent</code></span></a> is deprecated. In Python 3.15, <span class="pre">`__package__`</span> will cease to be set or take into consideration by the import system or standard library. (<a href="https://github.com/python/cpython/issues/97879" class="reference external">gh-97879</a>)

- <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a>:

  - The undocumented <span class="pre">`ctypes.SetPointerType()`</span> function has been deprecated since Python 3.13.

- <a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a>:

  - The obsolete and rarely used <a href="../library/http.server.html#http.server.CGIHTTPRequestHandler" class="reference internal" title="http.server.CGIHTTPRequestHandler"><span class="pre"><code class="sourceCode python">CGIHTTPRequestHandler</code></span></a> has been deprecated since Python 3.13. No direct replacement exists. *Anything* is better than CGI to interface a web server with a request handler.

  - The <span class="pre">`--cgi`</span> flag to the **python -m http.server** command-line interface has been deprecated since Python 3.13.

- <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a>:

  - <span class="pre">`load_module()`</span> method: use <span class="pre">`exec_module()`</span> instead.

- <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a>:

  - The <a href="../library/locale.html#locale.getdefaultlocale" class="reference internal" title="locale.getdefaultlocale"><span class="pre"><code class="sourceCode python">getdefaultlocale()</code></span></a> function has been deprecated since Python 3.11. Its removal was originally planned for Python 3.13 (<a href="https://github.com/python/cpython/issues/90817" class="reference external">gh-90817</a>), but has been postponed to Python 3.15. Use <a href="../library/locale.html#locale.getlocale" class="reference internal" title="locale.getlocale"><span class="pre"><code class="sourceCode python">getlocale()</code></span></a>, <a href="../library/locale.html#locale.setlocale" class="reference internal" title="locale.setlocale"><span class="pre"><code class="sourceCode python">setlocale()</code></span></a>, and <a href="../library/locale.html#locale.getencoding" class="reference internal" title="locale.getencoding"><span class="pre"><code class="sourceCode python">getencoding()</code></span></a> instead. (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/111187" class="reference external">gh-111187</a>.)

- <a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a>:

  - <a href="../library/pathlib.html#pathlib.PurePath.is_reserved" class="reference internal" title="pathlib.PurePath.is_reserved"><span class="pre"><code class="sourceCode python">PurePath.is_reserved()</code></span></a> has been deprecated since Python 3.13. Use <a href="../library/os.path.html#os.path.isreserved" class="reference internal" title="os.path.isreserved"><span class="pre"><code class="sourceCode python">os.path.isreserved()</code></span></a> to detect reserved paths on Windows.

- <a href="../library/platform.html#module-platform" class="reference internal" title="platform: Retrieves as much platform identifying data as possible."><span class="pre"><code class="sourceCode python">platform</code></span></a>:

  - <a href="../library/platform.html#platform.java_ver" class="reference internal" title="platform.java_ver"><span class="pre"><code class="sourceCode python">java_ver()</code></span></a> has been deprecated since Python 3.13. This function is only useful for Jython support, has a confusing API, and is largely untested.

- <a href="../library/sysconfig.html#module-sysconfig" class="reference internal" title="sysconfig: Python&#39;s configuration information"><span class="pre"><code class="sourceCode python">sysconfig</code></span></a>:

  - The *check_home* argument of <a href="../library/sysconfig.html#sysconfig.is_python_build" class="reference internal" title="sysconfig.is_python_build"><span class="pre"><code class="sourceCode python">sysconfig.is_python_build()</code></span></a> has been deprecated since Python 3.12.

- <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a>:

  - <a href="../library/threading.html#threading.RLock" class="reference internal" title="threading.RLock"><span class="pre"><code class="sourceCode python">RLock()</code></span></a> will take no arguments in Python 3.15. Passing any arguments has been deprecated since Python 3.14, as the Python version does not permit any arguments, but the C version allows any number of positional or keyword arguments, ignoring every argument.

- <a href="../library/types.html#module-types" class="reference internal" title="types: Names for built-in types."><span class="pre"><code class="sourceCode python">types</code></span></a>:

  - <a href="../library/types.html#types.CodeType" class="reference internal" title="types.CodeType"><span class="pre"><code class="sourceCode python">types.CodeType</code></span></a>: Accessing <a href="../reference/datamodel.html#codeobject.co_lnotab" class="reference internal" title="codeobject.co_lnotab"><span class="pre"><code class="sourceCode python">co_lnotab</code></span></a> was deprecated in <span id="index-34" class="target"></span><a href="https://peps.python.org/pep-0626/" class="pep reference external"><strong>PEP 626</strong></a> since 3.10 and was planned to be removed in 3.12, but it only got a proper <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> in 3.12. May be removed in 3.15. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/101866" class="reference external">gh-101866</a>.)

- <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a>:

  - The undocumented keyword argument syntax for creating <a href="../library/typing.html#typing.NamedTuple" class="reference internal" title="typing.NamedTuple"><span class="pre"><code class="sourceCode python">NamedTuple</code></span></a> classes (e.g. <span class="pre">`Point`</span>` `<span class="pre">`=`</span>` `<span class="pre">`NamedTuple("Point",`</span>` `<span class="pre">`x=int,`</span>` `<span class="pre">`y=int)`</span>) has been deprecated since Python 3.13. Use the class-based syntax or the functional syntax instead.

  - When using the functional syntax of <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">TypedDict</code></span></a>s, failing to pass a value to the *fields* parameter (<span class="pre">`TD`</span>` `<span class="pre">`=`</span>` `<span class="pre">`TypedDict("TD")`</span>) or passing <span class="pre">`None`</span> (<span class="pre">`TD`</span>` `<span class="pre">`=`</span>` `<span class="pre">`TypedDict("TD",`</span>` `<span class="pre">`None)`</span>) has been deprecated since Python 3.13. Use <span class="pre">`class`</span>` `<span class="pre">`TD(TypedDict):`</span>` `<span class="pre">`pass`</span> or <span class="pre">`TD`</span>` `<span class="pre">`=`</span>` `<span class="pre">`TypedDict("TD",`</span>` `<span class="pre">`{})`</span> to create a TypedDict with zero field.

  - The <a href="../library/typing.html#typing.no_type_check_decorator" class="reference internal" title="typing.no_type_check_decorator"><span class="pre"><code class="sourceCode python">typing.no_type_check_decorator()</code></span></a> decorator function has been deprecated since Python 3.13. After eight years in the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module, it has yet to be supported by any major type checker.

- <a href="../library/wave.html#module-wave" class="reference internal" title="wave: Provide an interface to the WAV sound format."><span class="pre"><code class="sourceCode python">wave</code></span></a>:

  - The <a href="../library/wave.html#wave.Wave_read.getmark" class="reference internal" title="wave.Wave_read.getmark"><span class="pre"><code class="sourceCode python">getmark()</code></span></a>, <span class="pre">`setmark()`</span>, and <a href="../library/wave.html#wave.Wave_read.getmarkers" class="reference internal" title="wave.Wave_read.getmarkers"><span class="pre"><code class="sourceCode python">getmarkers()</code></span></a> methods of the <a href="../library/wave.html#wave.Wave_read" class="reference internal" title="wave.Wave_read"><span class="pre"><code class="sourceCode python">Wave_read</code></span></a> and <a href="../library/wave.html#wave.Wave_write" class="reference internal" title="wave.Wave_write"><span class="pre"><code class="sourceCode python">Wave_write</code></span></a> classes have been deprecated since Python 3.13.

</div>

<div id="pending-removal-in-python-3-16" class="section">

### Pending removal in Python 3.16<a href="#pending-removal-in-python-3-16" class="headerlink" title="Link to this heading">¶</a>

- The import system:

  - Setting <a href="../reference/datamodel.html#module.__loader__" class="reference internal" title="module.__loader__"><span class="pre"><code class="sourceCode python">__loader__</code></span></a> on a module while failing to set <a href="../library/importlib.html#importlib.machinery.ModuleSpec.loader" class="reference internal" title="importlib.machinery.ModuleSpec.loader"><span class="pre"><code class="sourceCode python">__spec__.loader</code></span></a> is deprecated. In Python 3.16, <span class="pre">`__loader__`</span> will cease to be set or taken into consideration by the import system or the standard library.

- <a href="../library/array.html#module-array" class="reference internal" title="array: Space efficient arrays of uniformly typed numeric values."><span class="pre"><code class="sourceCode python">array</code></span></a>:

  - The <span class="pre">`'u'`</span> format code (<span class="pre">`wchar_t`</span>) has been deprecated in documentation since Python 3.3 and at runtime since Python 3.13. Use the <span class="pre">`'w'`</span> format code (<a href="../c-api/unicode.html#c.Py_UCS4" class="reference internal" title="Py_UCS4"><span class="pre"><code class="sourceCode c">Py_UCS4</code></span></a>) for Unicode characters instead.

- <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>:

  - <span class="pre">`asyncio.iscoroutinefunction()`</span> is deprecated and will be removed in Python 3.16, use <a href="../library/inspect.html#inspect.iscoroutinefunction" class="reference internal" title="inspect.iscoroutinefunction"><span class="pre"><code class="sourceCode python">inspect.iscoroutinefunction()</code></span></a> instead. (Contributed by Jiahao Li and Kumar Aditya in <a href="https://github.com/python/cpython/issues/122875" class="reference external">gh-122875</a>.)

- <a href="../library/builtins.html#module-builtins" class="reference internal" title="builtins: The module that provides the built-in namespace."><span class="pre"><code class="sourceCode python">builtins</code></span></a>:

  - Bitwise inversion on boolean types, <span class="pre">`~True`</span> or <span class="pre">`~False`</span> has been deprecated since Python 3.12, as it produces surprising and unintuitive results (<span class="pre">`-2`</span> and <span class="pre">`-1`</span>). Use <span class="pre">`not`</span>` `<span class="pre">`x`</span> instead for the logical negation of a Boolean. In the rare case that you need the bitwise inversion of the underlying integer, convert to <span class="pre">`int`</span> explicitly (<span class="pre">`~int(x)`</span>).

- <a href="../library/shutil.html#module-shutil" class="reference internal" title="shutil: High-level file operations, including copying."><span class="pre"><code class="sourceCode python">shutil</code></span></a>:

  - The <span class="pre">`ExecError`</span> exception has been deprecated since Python 3.14. It has not been used by any function in <span class="pre">`shutil`</span> since Python 3.4, and is now an alias of <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a>.

- <a href="../library/symtable.html#module-symtable" class="reference internal" title="symtable: Interface to the compiler&#39;s internal symbol tables."><span class="pre"><code class="sourceCode python">symtable</code></span></a>:

  - The <a href="../library/symtable.html#symtable.Class.get_methods" class="reference internal" title="symtable.Class.get_methods"><span class="pre"><code class="sourceCode python">Class.get_methods</code></span></a> method has been deprecated since Python 3.14.

- <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a>:

  - The <a href="../library/sys.html#sys._enablelegacywindowsfsencoding" class="reference internal" title="sys._enablelegacywindowsfsencoding"><span class="pre"><code class="sourceCode python">_enablelegacywindowsfsencoding()</code></span></a> function has been deprecated since Python 3.13. Use the <span id="index-35" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONLEGACYWINDOWSFSENCODING" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONLEGACYWINDOWSFSENCODING</code></span></a> environment variable instead.

- <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>:

  - The undocumented and unused <span class="pre">`TarFile.tarfile`</span> attribute has been deprecated since Python 3.13.

</div>

<div id="pending-removal-in-python-3-17" class="section">

### Pending removal in Python 3.17<a href="#pending-removal-in-python-3-17" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/collections.abc.html#module-collections.abc" class="reference internal" title="collections.abc: Abstract base classes for containers"><span class="pre"><code class="sourceCode python">collections.abc</code></span></a>:

  - <a href="../library/collections.abc.html#collections.abc.ByteString" class="reference internal" title="collections.abc.ByteString"><span class="pre"><code class="sourceCode python">collections.abc.ByteString</code></span></a> is scheduled for removal in Python 3.17.

    Use <span class="pre">`isinstance(obj,`</span>` `<span class="pre">`collections.abc.Buffer)`</span> to test if <span class="pre">`obj`</span> implements the <a href="../c-api/buffer.html#bufferobjects" class="reference internal"><span class="std std-ref">buffer protocol</span></a> at runtime. For use in type annotations, either use <a href="../library/collections.abc.html#collections.abc.Buffer" class="reference internal" title="collections.abc.Buffer"><span class="pre"><code class="sourceCode python">Buffer</code></span></a> or a union that explicitly specifies the types your code supports (e.g., <span class="pre">`bytes`</span>` `<span class="pre">`|`</span>` `<span class="pre">`bytearray`</span>` `<span class="pre">`|`</span>` `<span class="pre">`memoryview`</span>).

    <span class="pre">`ByteString`</span> was originally intended to be an abstract class that would serve as a supertype of both <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>. However, since the ABC never had any methods, knowing that an object was an instance of <span class="pre">`ByteString`</span> never actually told you anything useful about the object. Other common buffer types such as <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> were also never understood as subtypes of <span class="pre">`ByteString`</span> (either at runtime or by static type checkers).

    See <span id="index-36" class="target"></span><a href="https://peps.python.org/pep-0688/#current-options" class="pep reference external"><strong>PEP 688</strong></a> for more details. (Contributed by Shantanu Jain in <a href="https://github.com/python/cpython/issues/91896" class="reference external">gh-91896</a>.)

- <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a>:

  - Before Python 3.14, old-style unions were implemented using the private class <span class="pre">`typing._UnionGenericAlias`</span>. This class is no longer needed for the implementation, but it has been retained for backward compatibility, with removal scheduled for Python 3.17. Users should use documented introspection helpers like <a href="../library/typing.html#typing.get_origin" class="reference internal" title="typing.get_origin"><span class="pre"><code class="sourceCode python">typing.get_origin()</code></span></a> and <a href="../library/typing.html#typing.get_args" class="reference internal" title="typing.get_args"><span class="pre"><code class="sourceCode python">typing.get_args()</code></span></a> instead of relying on private implementation details.

  - <a href="../library/typing.html#typing.ByteString" class="reference internal" title="typing.ByteString"><span class="pre"><code class="sourceCode python">typing.ByteString</code></span></a>, deprecated since Python 3.9, is scheduled for removal in Python 3.17.

    Use <span class="pre">`isinstance(obj,`</span>` `<span class="pre">`collections.abc.Buffer)`</span> to test if <span class="pre">`obj`</span> implements the <a href="../c-api/buffer.html#bufferobjects" class="reference internal"><span class="std std-ref">buffer protocol</span></a> at runtime. For use in type annotations, either use <a href="../library/collections.abc.html#collections.abc.Buffer" class="reference internal" title="collections.abc.Buffer"><span class="pre"><code class="sourceCode python">Buffer</code></span></a> or a union that explicitly specifies the types your code supports (e.g., <span class="pre">`bytes`</span>` `<span class="pre">`|`</span>` `<span class="pre">`bytearray`</span>` `<span class="pre">`|`</span>` `<span class="pre">`memoryview`</span>).

    <span class="pre">`ByteString`</span> was originally intended to be an abstract class that would serve as a supertype of both <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>. However, since the ABC never had any methods, knowing that an object was an instance of <span class="pre">`ByteString`</span> never actually told you anything useful about the object. Other common buffer types such as <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> were also never understood as subtypes of <span class="pre">`ByteString`</span> (either at runtime or by static type checkers).

    See <span id="index-37" class="target"></span><a href="https://peps.python.org/pep-0688/#current-options" class="pep reference external"><strong>PEP 688</strong></a> for more details. (Contributed by Shantanu Jain in <a href="https://github.com/python/cpython/issues/91896" class="reference external">gh-91896</a>.)

</div>

<div id="pending-removal-in-future-versions" class="section">

### Pending Removal in Future Versions<a href="#pending-removal-in-future-versions" class="headerlink" title="Link to this heading">¶</a>

The following APIs will be removed in the future, although there is currently no date scheduled for their removal.

- <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a>: Nesting argument groups and nesting mutually exclusive groups are deprecated.

- <a href="../library/builtins.html#module-builtins" class="reference internal" title="builtins: The module that provides the built-in namespace."><span class="pre"><code class="sourceCode python">builtins</code></span></a>:

  - <span class="pre">`bool(NotImplemented)`</span>.

  - Generators: <span class="pre">`throw(type,`</span>` `<span class="pre">`exc,`</span>` `<span class="pre">`tb)`</span> and <span class="pre">`athrow(type,`</span>` `<span class="pre">`exc,`</span>` `<span class="pre">`tb)`</span> signature is deprecated: use <span class="pre">`throw(exc)`</span> and <span class="pre">`athrow(exc)`</span> instead, the single argument signature.

  - Currently Python accepts numeric literals immediately followed by keywords, for example <span class="pre">`0in`</span>` `<span class="pre">`x`</span>, <span class="pre">`1or`</span>` `<span class="pre">`x`</span>, <span class="pre">`0if`</span>` `<span class="pre">`1else`</span>` `<span class="pre">`2`</span>. It allows confusing and ambiguous expressions like <span class="pre">`[0x1for`</span>` `<span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`y]`</span> (which can be interpreted as <span class="pre">`[0x1`</span>` `<span class="pre">`for`</span>` `<span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`y]`</span> or <span class="pre">`[0x1f`</span>` `<span class="pre">`or`</span>` `<span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`y]`</span>). A syntax warning is raised if the numeric literal is immediately followed by one of keywords <a href="../reference/expressions.html#and" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">and</code></span></a>, <a href="../reference/compound_stmts.html#else" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">else</code></span></a>, <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a>, <a href="../reference/compound_stmts.html#if" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">if</code></span></a>, <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a>, <a href="../reference/expressions.html#is" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">is</code></span></a> and <a href="../reference/expressions.html#or" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">or</code></span></a>. In a future release it will be changed to a syntax error. (<a href="https://github.com/python/cpython/issues/87999" class="reference external">gh-87999</a>)

  - Support for <span class="pre">`__index__()`</span> and <span class="pre">`__int__()`</span> method returning non-int type: these methods will be required to return an instance of a strict subclass of <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>.

  - Support for <span class="pre">`__float__()`</span> method returning a strict subclass of <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a>: these methods will be required to return an instance of <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a>.

  - Support for <span class="pre">`__complex__()`</span> method returning a strict subclass of <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a>: these methods will be required to return an instance of <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a>.

  - Delegation of <span class="pre">`int()`</span> to <span class="pre">`__trunc__()`</span> method.

  - Passing a complex number as the *real* or *imag* argument in the <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span>()</code></span></a> constructor is now deprecated; it should only be passed as a single positional argument. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/109218" class="reference external">gh-109218</a>.)

- <a href="../library/calendar.html#module-calendar" class="reference internal" title="calendar: Functions for working with calendars, including some emulation of the Unix cal program."><span class="pre"><code class="sourceCode python">calendar</code></span></a>: <span class="pre">`calendar.January`</span> and <span class="pre">`calendar.February`</span> constants are deprecated and replaced by <a href="../library/calendar.html#calendar.JANUARY" class="reference internal" title="calendar.JANUARY"><span class="pre"><code class="sourceCode python">calendar.JANUARY</code></span></a> and <a href="../library/calendar.html#calendar.FEBRUARY" class="reference internal" title="calendar.FEBRUARY"><span class="pre"><code class="sourceCode python">calendar.FEBRUARY</code></span></a>. (Contributed by Prince Roshan in <a href="https://github.com/python/cpython/issues/103636" class="reference external">gh-103636</a>.)

- <a href="../reference/datamodel.html#codeobject.co_lnotab" class="reference internal" title="codeobject.co_lnotab"><span class="pre"><code class="sourceCode python">codeobject.co_lnotab</code></span></a>: use the <a href="../reference/datamodel.html#codeobject.co_lines" class="reference internal" title="codeobject.co_lines"><span class="pre"><code class="sourceCode python">codeobject.co_lines()</code></span></a> method instead.

- <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a>:

  - <a href="../library/datetime.html#datetime.datetime.utcnow" class="reference internal" title="datetime.datetime.utcnow"><span class="pre"><code class="sourceCode python">utcnow()</code></span></a>: use <span class="pre">`datetime.datetime.now(tz=datetime.UTC)`</span>.

  - <a href="../library/datetime.html#datetime.datetime.utcfromtimestamp" class="reference internal" title="datetime.datetime.utcfromtimestamp"><span class="pre"><code class="sourceCode python">utcfromtimestamp()</code></span></a>: use <span class="pre">`datetime.datetime.fromtimestamp(timestamp,`</span>` `<span class="pre">`tz=datetime.UTC)`</span>.

- <a href="../library/gettext.html#module-gettext" class="reference internal" title="gettext: Multilingual internationalization services."><span class="pre"><code class="sourceCode python">gettext</code></span></a>: Plural value must be an integer.

- <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a>:

  - <a href="../library/importlib.html#importlib.util.cache_from_source" class="reference internal" title="importlib.util.cache_from_source"><span class="pre"><code class="sourceCode python">cache_from_source()</code></span></a> *debug_override* parameter is deprecated: use the *optimization* parameter instead.

- <a href="../library/importlib.metadata.html#module-importlib.metadata" class="reference internal" title="importlib.metadata: Accessing package metadata"><span class="pre"><code class="sourceCode python">importlib.metadata</code></span></a>:

  - <span class="pre">`EntryPoints`</span> tuple interface.

  - Implicit <span class="pre">`None`</span> on return values.

- <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a>: the <span class="pre">`warn()`</span> method has been deprecated since Python 3.3, use <a href="../library/logging.html#logging.warning" class="reference internal" title="logging.warning"><span class="pre"><code class="sourceCode python">warning()</code></span></a> instead.

- <a href="../library/mailbox.html#module-mailbox" class="reference internal" title="mailbox: Manipulate mailboxes in various formats"><span class="pre"><code class="sourceCode python">mailbox</code></span></a>: Use of StringIO input and text mode is deprecated, use BytesIO and binary mode instead.

- <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a>: Calling <a href="../library/os.html#os.register_at_fork" class="reference internal" title="os.register_at_fork"><span class="pre"><code class="sourceCode python">os.register_at_fork()</code></span></a> in multi-threaded process.

- <span class="pre">`pydoc.ErrorDuringImport`</span>: A tuple value for *exc_info* parameter is deprecated, use an exception instance.

- <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a>: More strict rules are now applied for numerical group references and group names in regular expressions. Only sequence of ASCII digits is now accepted as a numerical reference. The group name in bytes patterns and replacement strings can now only contain ASCII letters and digits and underscore. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/91760" class="reference external">gh-91760</a>.)

- <span class="pre">`sre_compile`</span>, <span class="pre">`sre_constants`</span> and <span class="pre">`sre_parse`</span> modules.

- <a href="../library/shutil.html#module-shutil" class="reference internal" title="shutil: High-level file operations, including copying."><span class="pre"><code class="sourceCode python">shutil</code></span></a>: <a href="../library/shutil.html#shutil.rmtree" class="reference internal" title="shutil.rmtree"><span class="pre"><code class="sourceCode python">rmtree()</code></span></a>’s *onerror* parameter is deprecated in Python 3.12; use the *onexc* parameter instead.

- <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> options and protocols:

  - <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a> without protocol argument is deprecated.

  - <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a>: <a href="../library/ssl.html#ssl.SSLContext.set_npn_protocols" class="reference internal" title="ssl.SSLContext.set_npn_protocols"><span class="pre"><code class="sourceCode python">set_npn_protocols()</code></span></a> and <span class="pre">`selected_npn_protocol()`</span> are deprecated: use ALPN instead.

  - <span class="pre">`ssl.OP_NO_SSL*`</span> options

  - <span class="pre">`ssl.OP_NO_TLS*`</span> options

  - <span class="pre">`ssl.PROTOCOL_SSLv3`</span>

  - <span class="pre">`ssl.PROTOCOL_TLS`</span>

  - <span class="pre">`ssl.PROTOCOL_TLSv1`</span>

  - <span class="pre">`ssl.PROTOCOL_TLSv1_1`</span>

  - <span class="pre">`ssl.PROTOCOL_TLSv1_2`</span>

  - <span class="pre">`ssl.TLSVersion.SSLv3`</span>

  - <span class="pre">`ssl.TLSVersion.TLSv1`</span>

  - <span class="pre">`ssl.TLSVersion.TLSv1_1`</span>

- <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a> methods:

  - <span class="pre">`threading.Condition.notifyAll()`</span>: use <a href="../library/threading.html#threading.Condition.notify_all" class="reference internal" title="threading.Condition.notify_all"><span class="pre"><code class="sourceCode python">notify_all()</code></span></a>.

  - <span class="pre">`threading.Event.isSet()`</span>: use <a href="../library/threading.html#threading.Event.is_set" class="reference internal" title="threading.Event.is_set"><span class="pre"><code class="sourceCode python">is_set()</code></span></a>.

  - <span class="pre">`threading.Thread.isDaemon()`</span>, <a href="../library/threading.html#threading.Thread.setDaemon" class="reference internal" title="threading.Thread.setDaemon"><span class="pre"><code class="sourceCode python">threading.Thread.setDaemon()</code></span></a>: use <a href="../library/threading.html#threading.Thread.daemon" class="reference internal" title="threading.Thread.daemon"><span class="pre"><code class="sourceCode python">threading.Thread.daemon</code></span></a> attribute.

  - <span class="pre">`threading.Thread.getName()`</span>, <a href="../library/threading.html#threading.Thread.setName" class="reference internal" title="threading.Thread.setName"><span class="pre"><code class="sourceCode python">threading.Thread.setName()</code></span></a>: use <a href="../library/threading.html#threading.Thread.name" class="reference internal" title="threading.Thread.name"><span class="pre"><code class="sourceCode python">threading.Thread.name</code></span></a> attribute.

  - <span class="pre">`threading.currentThread()`</span>: use <a href="../library/threading.html#threading.current_thread" class="reference internal" title="threading.current_thread"><span class="pre"><code class="sourceCode python">threading.current_thread()</code></span></a>.

  - <span class="pre">`threading.activeCount()`</span>: use <a href="../library/threading.html#threading.active_count" class="reference internal" title="threading.active_count"><span class="pre"><code class="sourceCode python">threading.active_count()</code></span></a>.

- <a href="../library/typing.html#typing.Text" class="reference internal" title="typing.Text"><span class="pre"><code class="sourceCode python">typing.Text</code></span></a> (<a href="https://github.com/python/cpython/issues/92332" class="reference external">gh-92332</a>).

- <a href="../library/unittest.html#unittest.IsolatedAsyncioTestCase" class="reference internal" title="unittest.IsolatedAsyncioTestCase"><span class="pre"><code class="sourceCode python">unittest.IsolatedAsyncioTestCase</code></span></a>: it is deprecated to return a value that is not <span class="pre">`None`</span> from a test case.

- <a href="../library/urllib.parse.html#module-urllib.parse" class="reference internal" title="urllib.parse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urllib.parse</code></span></a> deprecated functions: <a href="../library/urllib.parse.html#urllib.parse.urlparse" class="reference internal" title="urllib.parse.urlparse"><span class="pre"><code class="sourceCode python">urlparse()</code></span></a> instead

  - <span class="pre">`splitattr()`</span>

  - <span class="pre">`splithost()`</span>

  - <span class="pre">`splitnport()`</span>

  - <span class="pre">`splitpasswd()`</span>

  - <span class="pre">`splitport()`</span>

  - <span class="pre">`splitquery()`</span>

  - <span class="pre">`splittag()`</span>

  - <span class="pre">`splittype()`</span>

  - <span class="pre">`splituser()`</span>

  - <span class="pre">`splitvalue()`</span>

  - <span class="pre">`to_bytes()`</span>

- <a href="../library/urllib.request.html#module-urllib.request" class="reference internal" title="urllib.request: Extensible library for opening URLs."><span class="pre"><code class="sourceCode python">urllib.request</code></span></a>: <a href="../library/urllib.request.html#urllib.request.URLopener" class="reference internal" title="urllib.request.URLopener"><span class="pre"><code class="sourceCode python">URLopener</code></span></a> and <a href="../library/urllib.request.html#urllib.request.FancyURLopener" class="reference internal" title="urllib.request.FancyURLopener"><span class="pre"><code class="sourceCode python">FancyURLopener</code></span></a> style of invoking requests is deprecated. Use newer <a href="../library/urllib.request.html#urllib.request.urlopen" class="reference internal" title="urllib.request.urlopen"><span class="pre"><code class="sourceCode python">urlopen()</code></span></a> functions and methods.

- <a href="../library/wsgiref.html#module-wsgiref" class="reference internal" title="wsgiref: WSGI Utilities and Reference Implementation."><span class="pre"><code class="sourceCode python">wsgiref</code></span></a>: <span class="pre">`SimpleHandler.stdout.write()`</span> should not do partial writes.

- <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a>: Testing the truth value of an <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element" class="reference internal" title="xml.etree.ElementTree.Element"><span class="pre"><code class="sourceCode python">Element</code></span></a> is deprecated. In a future release it will always return <span class="pre">`True`</span>. Prefer explicit <span class="pre">`len(elem)`</span> or <span class="pre">`elem`</span>` `<span class="pre">`is`</span>` `<span class="pre">`not`</span>` `<span class="pre">`None`</span> tests instead.

- <a href="../library/zipimport.html#zipimport.zipimporter.load_module" class="reference internal" title="zipimport.zipimporter.load_module"><span class="pre"><code class="sourceCode python">zipimport.zipimporter.load_module()</code></span></a> is deprecated: use <a href="../library/zipimport.html#zipimport.zipimporter.exec_module" class="reference internal" title="zipimport.zipimporter.exec_module"><span class="pre"><code class="sourceCode python">exec_module()</code></span></a> instead.

</div>

</div>

<div id="removed" class="section">

<span id="whatsnew312-removed"></span>

## Removed<a href="#removed" class="headerlink" title="Link to this heading">¶</a>

<div id="asynchat-and-asyncore" class="section">

### asynchat and asyncore<a href="#asynchat-and-asyncore" class="headerlink" title="Link to this heading">¶</a>

- These two modules have been removed according to the schedule in <span id="index-38" class="target"></span><a href="https://peps.python.org/pep-0594/" class="pep reference external"><strong>PEP 594</strong></a>, having been deprecated in Python 3.6. Use <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> instead. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/96580" class="reference external">gh-96580</a>.)

</div>

<div id="configparser" class="section">

### configparser<a href="#configparser" class="headerlink" title="Link to this heading">¶</a>

- Several names deprecated in the <a href="../library/configparser.html#module-configparser" class="reference internal" title="configparser: Configuration file parser."><span class="pre"><code class="sourceCode python">configparser</code></span></a> way back in 3.2 have been removed per <a href="https://github.com/python/cpython/issues/89336" class="reference external">gh-89336</a>:

  - <a href="../library/configparser.html#configparser.ParsingError" class="reference internal" title="configparser.ParsingError"><span class="pre"><code class="sourceCode python">configparser.ParsingError</code></span></a> no longer has a <span class="pre">`filename`</span> attribute or argument. Use the <span class="pre">`source`</span> attribute and argument instead.

  - <a href="../library/configparser.html#module-configparser" class="reference internal" title="configparser: Configuration file parser."><span class="pre"><code class="sourceCode python">configparser</code></span></a> no longer has a <span class="pre">`SafeConfigParser`</span> class. Use the shorter <a href="../library/configparser.html#configparser.ConfigParser" class="reference internal" title="configparser.ConfigParser"><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a> name instead.

  - <a href="../library/configparser.html#configparser.ConfigParser" class="reference internal" title="configparser.ConfigParser"><span class="pre"><code class="sourceCode python">configparser.ConfigParser</code></span></a> no longer has a <span class="pre">`readfp`</span> method. Use <a href="../library/configparser.html#configparser.ConfigParser.read_file" class="reference internal" title="configparser.ConfigParser.read_file"><span class="pre"><code class="sourceCode python">read_file()</code></span></a> instead.

</div>

<div id="distutils" class="section">

<span id="whatsnew312-removed-distutils"></span>

### distutils<a href="#distutils" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`distutils`</span> package. It was deprecated in Python 3.10 by <span id="index-39" class="target"></span><a href="https://peps.python.org/pep-0632/" class="pep reference external"><strong>PEP 632</strong></a> “Deprecate distutils module”. For projects still using <span class="pre">`distutils`</span> and cannot be updated to something else, the <span class="pre">`setuptools`</span> project can be installed: it still provides <span class="pre">`distutils`</span>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/92584" class="reference external">gh-92584</a>.)

</div>

<div id="ensurepip" class="section">

### ensurepip<a href="#ensurepip" class="headerlink" title="Link to this heading">¶</a>

- Remove the bundled setuptools wheel from <a href="../library/ensurepip.html#module-ensurepip" class="reference internal" title="ensurepip: Bootstrapping the &quot;pip&quot; installer into an existing Python installation or virtual environment."><span class="pre"><code class="sourceCode python">ensurepip</code></span></a>, and stop installing setuptools in environments created by <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a>.

  <span class="pre">`pip`</span>` `<span class="pre">`(>=`</span>` `<span class="pre">`22.1)`</span> does not require setuptools to be installed in the environment. <span class="pre">`setuptools`</span>-based (and <span class="pre">`distutils`</span>-based) packages can still be used with <span class="pre">`pip`</span>` `<span class="pre">`install`</span>, since pip will provide <span class="pre">`setuptools`</span> in the build environment it uses for building a package.

  <span class="pre">`easy_install`</span>, <span class="pre">`pkg_resources`</span>, <span class="pre">`setuptools`</span> and <span class="pre">`distutils`</span> are no longer provided by default in environments created with <span class="pre">`venv`</span> or bootstrapped with <span class="pre">`ensurepip`</span>, since they are part of the <span class="pre">`setuptools`</span> package. For projects relying on these at runtime, the <span class="pre">`setuptools`</span> project should be declared as a dependency and installed separately (typically, using pip).

  (Contributed by Pradyun Gedam in <a href="https://github.com/python/cpython/issues/95299" class="reference external">gh-95299</a>.)

</div>

<div id="enum" class="section">

### enum<a href="#enum" class="headerlink" title="Link to this heading">¶</a>

- Remove <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">enum</code></span></a>’s <span class="pre">`EnumMeta.__getattr__`</span>, which is no longer needed for enum attribute access. (Contributed by Ethan Furman in <a href="https://github.com/python/cpython/issues/95083" class="reference external">gh-95083</a>.)

</div>

<div id="ftplib" class="section">

### ftplib<a href="#ftplib" class="headerlink" title="Link to this heading">¶</a>

- Remove <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a>’s <span class="pre">`FTP_TLS.ssl_version`</span> class attribute: use the *context* parameter instead. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94172" class="reference external">gh-94172</a>.)

</div>

<div id="gzip" class="section">

### gzip<a href="#gzip" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`filename`</span> attribute of <a href="../library/gzip.html#module-gzip" class="reference internal" title="gzip: Interfaces for gzip compression and decompression using file objects."><span class="pre"><code class="sourceCode python">gzip</code></span></a>’s <a href="../library/gzip.html#gzip.GzipFile" class="reference internal" title="gzip.GzipFile"><span class="pre"><code class="sourceCode python">gzip.GzipFile</code></span></a>, deprecated since Python 2.6, use the <a href="../library/gzip.html#gzip.GzipFile.name" class="reference internal" title="gzip.GzipFile.name"><span class="pre"><code class="sourceCode python">name</code></span></a> attribute instead. In write mode, the <span class="pre">`filename`</span> attribute added <span class="pre">`'.gz'`</span> file extension if it was not present. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94196" class="reference external">gh-94196</a>.)

</div>

<div id="hashlib" class="section">

### hashlib<a href="#hashlib" class="headerlink" title="Link to this heading">¶</a>

- Remove the pure Python implementation of <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a>’s <a href="../library/hashlib.html#hashlib.pbkdf2_hmac" class="reference internal" title="hashlib.pbkdf2_hmac"><span class="pre"><code class="sourceCode python">hashlib.pbkdf2_hmac()</code></span></a>, deprecated in Python 3.10. Python 3.10 and newer requires OpenSSL 1.1.1 (<span id="index-40" class="target"></span><a href="https://peps.python.org/pep-0644/" class="pep reference external"><strong>PEP 644</strong></a>): this OpenSSL version provides a C implementation of <a href="../library/hashlib.html#hashlib.pbkdf2_hmac" class="reference internal" title="hashlib.pbkdf2_hmac"><span class="pre"><code class="sourceCode python">pbkdf2_hmac()</code></span></a> which is faster. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94199" class="reference external">gh-94199</a>.)

</div>

<div id="importlib" class="section">

### importlib<a href="#importlib" class="headerlink" title="Link to this heading">¶</a>

- Many previously deprecated cleanups in <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> have now been completed:

  - References to, and support for <span class="pre">`module_repr()`</span> has been removed. (Contributed by Barry Warsaw in <a href="https://github.com/python/cpython/issues/97850" class="reference external">gh-97850</a>.)

  - <span class="pre">`importlib.util.set_package`</span>, <span class="pre">`importlib.util.set_loader`</span> and <span class="pre">`importlib.util.module_for_loader`</span> have all been removed. (Contributed by Brett Cannon and Nikita Sobolev in <a href="https://github.com/python/cpython/issues/65961" class="reference external">gh-65961</a> and <a href="https://github.com/python/cpython/issues/97850" class="reference external">gh-97850</a>.)

  - Support for <span class="pre">`find_loader()`</span> and <span class="pre">`find_module()`</span> APIs have been removed. (Contributed by Barry Warsaw in <a href="https://github.com/python/cpython/issues/98040" class="reference external">gh-98040</a>.)

  - <span class="pre">`importlib.abc.Finder`</span>, <span class="pre">`pkgutil.ImpImporter`</span>, and <span class="pre">`pkgutil.ImpLoader`</span> have been removed. (Contributed by Barry Warsaw in <a href="https://github.com/python/cpython/issues/98040" class="reference external">gh-98040</a>.)

</div>

<div id="imp" class="section">

<span id="whatsnew312-removed-imp"></span>

### imp<a href="#imp" class="headerlink" title="Link to this heading">¶</a>

- The <span class="pre">`imp`</span> module has been removed. (Contributed by Barry Warsaw in <a href="https://github.com/python/cpython/issues/98040" class="reference external">gh-98040</a>.)

  To migrate, consult the following correspondence table:

  > <div>
  >
  > | imp | importlib |
  > |----|----|
  > | <span class="pre">`imp.NullImporter`</span> | Insert <span class="pre">`None`</span> into <span class="pre">`sys.path_importer_cache`</span> |
  > | <span class="pre">`imp.cache_from_source()`</span> | <a href="../library/importlib.html#importlib.util.cache_from_source" class="reference internal" title="importlib.util.cache_from_source"><span class="pre"><code class="sourceCode python">importlib.util.cache_from_source()</code></span></a> |
  > | <span class="pre">`imp.find_module()`</span> | <a href="../library/importlib.html#importlib.util.find_spec" class="reference internal" title="importlib.util.find_spec"><span class="pre"><code class="sourceCode python">importlib.util.find_spec()</code></span></a> |
  > | <span class="pre">`imp.get_magic()`</span> | <a href="../library/importlib.html#importlib.util.MAGIC_NUMBER" class="reference internal" title="importlib.util.MAGIC_NUMBER"><span class="pre"><code class="sourceCode python">importlib.util.MAGIC_NUMBER</code></span></a> |
  > | <span class="pre">`imp.get_suffixes()`</span> | <a href="../library/importlib.html#importlib.machinery.SOURCE_SUFFIXES" class="reference internal" title="importlib.machinery.SOURCE_SUFFIXES"><span class="pre"><code class="sourceCode python">importlib.machinery.SOURCE_SUFFIXES</code></span></a>, <a href="../library/importlib.html#importlib.machinery.EXTENSION_SUFFIXES" class="reference internal" title="importlib.machinery.EXTENSION_SUFFIXES"><span class="pre"><code class="sourceCode python">importlib.machinery.EXTENSION_SUFFIXES</code></span></a>, and <a href="../library/importlib.html#importlib.machinery.BYTECODE_SUFFIXES" class="reference internal" title="importlib.machinery.BYTECODE_SUFFIXES"><span class="pre"><code class="sourceCode python">importlib.machinery.BYTECODE_SUFFIXES</code></span></a> |
  > | <span class="pre">`imp.get_tag()`</span> | <a href="../library/sys.html#sys.implementation" class="reference internal" title="sys.implementation"><span class="pre"><code class="sourceCode python">sys.implementation.cache_tag</code></span></a> |
  > | <span class="pre">`imp.load_module()`</span> | <a href="../library/importlib.html#importlib.import_module" class="reference internal" title="importlib.import_module"><span class="pre"><code class="sourceCode python">importlib.import_module()</code></span></a> |
  > | <span class="pre">`imp.new_module(name)`</span> | <span class="pre">`types.ModuleType(name)`</span> |
  > | <span class="pre">`imp.reload()`</span> | <a href="../library/importlib.html#importlib.reload" class="reference internal" title="importlib.reload"><span class="pre"><code class="sourceCode python">importlib.<span class="bu">reload</span>()</code></span></a> |
  > | <span class="pre">`imp.source_from_cache()`</span> | <a href="../library/importlib.html#importlib.util.source_from_cache" class="reference internal" title="importlib.util.source_from_cache"><span class="pre"><code class="sourceCode python">importlib.util.source_from_cache()</code></span></a> |
  > | <span class="pre">`imp.load_source()`</span> | *See below* |
  >
  > </div>

  Replace <span class="pre">`imp.load_source()`</span> with:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      import importlib.util
      import importlib.machinery

      def load_source(modname, filename):
          loader = importlib.machinery.SourceFileLoader(modname, filename)
          spec = importlib.util.spec_from_file_location(modname, filename, loader=loader)
          module = importlib.util.module_from_spec(spec)
          # The module is always executed and not cached in sys.modules.
          # Uncomment the following line to cache the module.
          # sys.modules[module.__name__] = module
          loader.exec_module(module)
          return module

  </div>

  </div>

- Remove <span class="pre">`imp`</span> functions and attributes with no replacements:

  - Undocumented functions:

    - <span class="pre">`imp.init_builtin()`</span>

    - <span class="pre">`imp.load_compiled()`</span>

    - <span class="pre">`imp.load_dynamic()`</span>

    - <span class="pre">`imp.load_package()`</span>

  - <span class="pre">`imp.lock_held()`</span>, <span class="pre">`imp.acquire_lock()`</span>, <span class="pre">`imp.release_lock()`</span>: the locking scheme has changed in Python 3.3 to per-module locks.

  - <span class="pre">`imp.find_module()`</span> constants: <span class="pre">`SEARCH_ERROR`</span>, <span class="pre">`PY_SOURCE`</span>, <span class="pre">`PY_COMPILED`</span>, <span class="pre">`C_EXTENSION`</span>, <span class="pre">`PY_RESOURCE`</span>, <span class="pre">`PKG_DIRECTORY`</span>, <span class="pre">`C_BUILTIN`</span>, <span class="pre">`PY_FROZEN`</span>, <span class="pre">`PY_CODERESOURCE`</span>, <span class="pre">`IMP_HOOK`</span>.

</div>

<div id="io" class="section">

### io<a href="#io" class="headerlink" title="Link to this heading">¶</a>

- Remove <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a>’s <span class="pre">`io.OpenWrapper`</span> and <span class="pre">`_pyio.OpenWrapper`</span>, deprecated in Python 3.10: just use <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> instead. The <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> (<a href="../library/io.html#io.open" class="reference internal" title="io.open"><span class="pre"><code class="sourceCode python">io.<span class="bu">open</span>()</code></span></a>) function is a built-in function. Since Python 3.10, <span class="pre">`_pyio.open()`</span> is also a static method. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94169" class="reference external">gh-94169</a>.)

</div>

<div id="locale" class="section">

### locale<a href="#locale" class="headerlink" title="Link to this heading">¶</a>

- Remove <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a>’s <span class="pre">`locale.format()`</span> function, deprecated in Python 3.7: use <a href="../library/locale.html#locale.format_string" class="reference internal" title="locale.format_string"><span class="pre"><code class="sourceCode python">locale.format_string()</code></span></a> instead. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94226" class="reference external">gh-94226</a>.)

</div>

<div id="smtpd" class="section">

### smtpd<a href="#smtpd" class="headerlink" title="Link to this heading">¶</a>

- The <span class="pre">`smtpd`</span> module has been removed according to the schedule in <span id="index-41" class="target"></span><a href="https://peps.python.org/pep-0594/" class="pep reference external"><strong>PEP 594</strong></a>, having been deprecated in Python 3.4.7 and 3.5.4. Use the <a href="https://pypi.org/project/aiosmtpd/" class="extlink-pypi reference external">aiosmtpd</a> PyPI module or any other <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>-based server instead. (Contributed by Oleg Iarygin in <a href="https://github.com/python/cpython/issues/93243" class="reference external">gh-93243</a>.)

</div>

<div id="id2" class="section">

### sqlite3<a href="#id2" class="headerlink" title="Link to this heading">¶</a>

- The following undocumented <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> features, deprecated in Python 3.10, are now removed:

  - <span class="pre">`sqlite3.enable_shared_cache()`</span>

  - <span class="pre">`sqlite3.OptimizedUnicode`</span>

  If a shared cache must be used, open the database in URI mode using the <span class="pre">`cache=shared`</span> query parameter.

  The <span class="pre">`sqlite3.OptimizedUnicode`</span> text factory has been an alias for <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> since Python 3.3. Code that previously set the text factory to <span class="pre">`OptimizedUnicode`</span> can either use <span class="pre">`str`</span> explicitly, or rely on the default value which is also <span class="pre">`str`</span>.

  (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/92548" class="reference external">gh-92548</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

- Remove <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a>’s <span class="pre">`ssl.RAND_pseudo_bytes()`</span> function, deprecated in Python 3.6: use <a href="../library/os.html#os.urandom" class="reference internal" title="os.urandom"><span class="pre"><code class="sourceCode python">os.urandom()</code></span></a> or <a href="../library/ssl.html#ssl.RAND_bytes" class="reference internal" title="ssl.RAND_bytes"><span class="pre"><code class="sourceCode python">ssl.RAND_bytes()</code></span></a> instead. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94199" class="reference external">gh-94199</a>.)

- Remove the <span class="pre">`ssl.match_hostname()`</span> function. It was deprecated in Python 3.7. OpenSSL performs hostname matching since Python 3.7, Python no longer uses the <span class="pre">`ssl.match_hostname()`</span> function. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94199" class="reference external">gh-94199</a>.)

- Remove the <span class="pre">`ssl.wrap_socket()`</span> function, deprecated in Python 3.7: instead, create a <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a> object and call its <a href="../library/ssl.html#ssl.SSLContext.wrap_socket" class="reference internal" title="ssl.SSLContext.wrap_socket"><span class="pre"><code class="sourceCode python">ssl.SSLContext.wrap_socket</code></span></a> method. Any package that still uses <span class="pre">`ssl.wrap_socket()`</span> is broken and insecure. The function neither sends a SNI TLS extension nor validates the server hostname. Code is subject to <span id="index-42" class="target"></span><a href="https://cwe.mitre.org/data/definitions/295.html" class="cwe reference external"><strong>CWE 295</strong></a> (Improper Certificate Validation). (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94199" class="reference external">gh-94199</a>.)

</div>

<div id="id3" class="section">

### unittest<a href="#id3" class="headerlink" title="Link to this heading">¶</a>

- Remove many long-deprecated <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> features:

  - A number of <a href="../library/unittest.html#unittest.TestCase" class="reference internal" title="unittest.TestCase"><span class="pre"><code class="sourceCode python">TestCase</code></span></a> method aliases:

    | Deprecated alias | Method Name | Deprecated in |
    |----|----|----|
    | <span class="pre">`failUnless`</span> | <a href="../library/unittest.html#unittest.TestCase.assertTrue" class="reference internal" title="unittest.TestCase.assertTrue"><span class="pre"><code class="sourceCode python">assertTrue()</code></span></a> | 3.1 |
    | <span class="pre">`failIf`</span> | <a href="../library/unittest.html#unittest.TestCase.assertFalse" class="reference internal" title="unittest.TestCase.assertFalse"><span class="pre"><code class="sourceCode python">assertFalse()</code></span></a> | 3.1 |
    | <span class="pre">`failUnlessEqual`</span> | <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a> | 3.1 |
    | <span class="pre">`failIfEqual`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotEqual" class="reference internal" title="unittest.TestCase.assertNotEqual"><span class="pre"><code class="sourceCode python">assertNotEqual()</code></span></a> | 3.1 |
    | <span class="pre">`failUnlessAlmostEqual`</span> | <a href="../library/unittest.html#unittest.TestCase.assertAlmostEqual" class="reference internal" title="unittest.TestCase.assertAlmostEqual"><span class="pre"><code class="sourceCode python">assertAlmostEqual()</code></span></a> | 3.1 |
    | <span class="pre">`failIfAlmostEqual`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotAlmostEqual" class="reference internal" title="unittest.TestCase.assertNotAlmostEqual"><span class="pre"><code class="sourceCode python">assertNotAlmostEqual()</code></span></a> | 3.1 |
    | <span class="pre">`failUnlessRaises`</span> | <a href="../library/unittest.html#unittest.TestCase.assertRaises" class="reference internal" title="unittest.TestCase.assertRaises"><span class="pre"><code class="sourceCode python">assertRaises()</code></span></a> | 3.1 |
    | <span class="pre">`assert_`</span> | <a href="../library/unittest.html#unittest.TestCase.assertTrue" class="reference internal" title="unittest.TestCase.assertTrue"><span class="pre"><code class="sourceCode python">assertTrue()</code></span></a> | 3.2 |
    | <span class="pre">`assertEquals`</span> | <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a> | 3.2 |
    | <span class="pre">`assertNotEquals`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotEqual" class="reference internal" title="unittest.TestCase.assertNotEqual"><span class="pre"><code class="sourceCode python">assertNotEqual()</code></span></a> | 3.2 |
    | <span class="pre">`assertAlmostEquals`</span> | <a href="../library/unittest.html#unittest.TestCase.assertAlmostEqual" class="reference internal" title="unittest.TestCase.assertAlmostEqual"><span class="pre"><code class="sourceCode python">assertAlmostEqual()</code></span></a> | 3.2 |
    | <span class="pre">`assertNotAlmostEquals`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotAlmostEqual" class="reference internal" title="unittest.TestCase.assertNotAlmostEqual"><span class="pre"><code class="sourceCode python">assertNotAlmostEqual()</code></span></a> | 3.2 |
    | <span class="pre">`assertRegexpMatches`</span> | <a href="../library/unittest.html#unittest.TestCase.assertRegex" class="reference internal" title="unittest.TestCase.assertRegex"><span class="pre"><code class="sourceCode python">assertRegex()</code></span></a> | 3.2 |
    | <span class="pre">`assertRaisesRegexp`</span> | <a href="../library/unittest.html#unittest.TestCase.assertRaisesRegex" class="reference internal" title="unittest.TestCase.assertRaisesRegex"><span class="pre"><code class="sourceCode python">assertRaisesRegex()</code></span></a> | 3.2 |
    | <span class="pre">`assertNotRegexpMatches`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotRegex" class="reference internal" title="unittest.TestCase.assertNotRegex"><span class="pre"><code class="sourceCode python">assertNotRegex()</code></span></a> | 3.5 |

    You can use <a href="https://github.com/isidentical/teyit" class="reference external">https://github.com/isidentical/teyit</a> to automatically modernise your unit tests.

  - Undocumented and broken <a href="../library/unittest.html#unittest.TestCase" class="reference internal" title="unittest.TestCase"><span class="pre"><code class="sourceCode python">TestCase</code></span></a> method <span class="pre">`assertDictContainsSubset`</span> (deprecated in Python 3.2).

  - Undocumented <a href="../library/unittest.html#unittest.TestLoader.loadTestsFromModule" class="reference internal" title="unittest.TestLoader.loadTestsFromModule"><span class="pre"><code class="sourceCode python">TestLoader.loadTestsFromModule</code></span></a> parameter *use_load_tests* (deprecated and ignored since Python 3.5).

  - An alias of the <a href="../library/unittest.html#unittest.TextTestResult" class="reference internal" title="unittest.TextTestResult"><span class="pre"><code class="sourceCode python">TextTestResult</code></span></a> class: <span class="pre">`_TextTestResult`</span> (deprecated in Python 3.2).

  (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/89325" class="reference external">gh-89325</a>.)

</div>

<div id="webbrowser" class="section">

### webbrowser<a href="#webbrowser" class="headerlink" title="Link to this heading">¶</a>

- Remove support for obsolete browsers from <a href="../library/webbrowser.html#module-webbrowser" class="reference internal" title="webbrowser: Easy-to-use controller for web browsers."><span class="pre"><code class="sourceCode python">webbrowser</code></span></a>. The removed browsers include: Grail, Mosaic, Netscape, Galeon, Skipstone, Iceape, Firebird, and Firefox versions 35 and below (<a href="https://github.com/python/cpython/issues/102871" class="reference external">gh-102871</a>).

</div>

<div id="xml-etree-elementtree" class="section">

### xml.etree.ElementTree<a href="#xml-etree-elementtree" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`ElementTree.Element.copy()`</span> method of the pure Python implementation, deprecated in Python 3.10, use the <a href="../library/copy.html#copy.copy" class="reference internal" title="copy.copy"><span class="pre"><code class="sourceCode python">copy.copy()</code></span></a> function instead. The C implementation of <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> has no <span class="pre">`copy()`</span> method, only a <span class="pre">`__copy__()`</span> method. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94383" class="reference external">gh-94383</a>.)

</div>

<div id="zipimport" class="section">

### zipimport<a href="#zipimport" class="headerlink" title="Link to this heading">¶</a>

- Remove <a href="../library/zipimport.html#module-zipimport" class="reference internal" title="zipimport: Support for importing Python modules from ZIP archives."><span class="pre"><code class="sourceCode python">zipimport</code></span></a>’s <span class="pre">`find_loader()`</span> and <span class="pre">`find_module()`</span> methods, deprecated in Python 3.10: use the <span class="pre">`find_spec()`</span> method instead. See <span id="index-43" class="target"></span><a href="https://peps.python.org/pep-0451/" class="pep reference external"><strong>PEP 451</strong></a> for the rationale. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94379" class="reference external">gh-94379</a>.)

</div>

<div id="others" class="section">

### Others<a href="#others" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`suspicious`</span> rule from the documentation <span class="pre">`Makefile`</span> and <span class="pre">`Doc/tools/rstlint.py`</span>, both in favor of <a href="https://github.com/sphinx-contrib/sphinx-lint" class="reference external">sphinx-lint</a>. (Contributed by Julien Palard in <a href="https://github.com/python/cpython/issues/98179" class="reference external">gh-98179</a>.)

- Remove the *keyfile* and *certfile* parameters from the <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a>, <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a>, <a href="../library/poplib.html#module-poplib" class="reference internal" title="poplib: POP3 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">poplib</code></span></a> and <a href="../library/smtplib.html#module-smtplib" class="reference internal" title="smtplib: SMTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">smtplib</code></span></a> modules, and the *key_file*, *cert_file* and *check_hostname* parameters from the <a href="../library/http.client.html#module-http.client" class="reference internal" title="http.client: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">http.client</code></span></a> module, all deprecated since Python 3.6. Use the *context* parameter (*ssl_context* in <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a>) instead. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94172" class="reference external">gh-94172</a>.)

- Remove <span class="pre">`Jython`</span> compatibility hacks from several stdlib modules and tests. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/99482" class="reference external">gh-99482</a>.)

- Remove <span class="pre">`_use_broken_old_ctypes_structure_semantics_`</span> flag from <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> module. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/99285" class="reference external">gh-99285</a>.)

</div>

</div>

<div id="porting-to-python-3-12" class="section">

<span id="whatsnew312-porting-to-python312"></span>

## Porting to Python 3.12<a href="#porting-to-python-3-12" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="changes-in-the-python-api" class="section">

### Changes in the Python API<a href="#changes-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- More strict rules are now applied for numerical group references and group names in regular expressions. Only sequence of ASCII digits is now accepted as a numerical reference. The group name in bytes patterns and replacement strings can now only contain ASCII letters and digits and underscore. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/91760" class="reference external">gh-91760</a>.)

- Remove <span class="pre">`randrange()`</span> functionality deprecated since Python 3.10. Formerly, <span class="pre">`randrange(10.0)`</span> losslessly converted to <span class="pre">`randrange(10)`</span>. Now, it raises a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>. Also, the exception raised for non-integer values such as <span class="pre">`randrange(10.5)`</span> or <span class="pre">`randrange('10')`</span> has been changed from <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> to <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>. This also prevents bugs where <span class="pre">`randrange(1e25)`</span> would silently select from a larger range than <span class="pre">`randrange(10**25)`</span>. (Originally suggested by Serhiy Storchaka <a href="https://github.com/python/cpython/issues/86388" class="reference external">gh-86388</a>.)

- <a href="../library/argparse.html#argparse.ArgumentParser" class="reference internal" title="argparse.ArgumentParser"><span class="pre"><code class="sourceCode python">argparse.ArgumentParser</code></span></a> changed encoding and error handler for reading arguments from file (e.g. <span class="pre">`fromfile_prefix_chars`</span> option) from default text encoding (e.g. <a href="../library/locale.html#locale.getpreferredencoding" class="reference internal" title="locale.getpreferredencoding"><span class="pre"><code class="sourceCode python">locale.getpreferredencoding(<span class="va">False</span>)</code></span></a>) to <a href="../glossary.html#term-filesystem-encoding-and-error-handler" class="reference internal"><span class="xref std std-term">filesystem encoding and error handler</span></a>. Argument files should be encoded in UTF-8 instead of ANSI Codepage on Windows.

- Remove the <span class="pre">`asyncore`</span>-based <span class="pre">`smtpd`</span> module deprecated in Python 3.4.7 and 3.5.4. A recommended replacement is the <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>-based <a href="https://pypi.org/project/aiosmtpd/" class="extlink-pypi reference external">aiosmtpd</a> PyPI module.

- <a href="../library/shlex.html#shlex.split" class="reference internal" title="shlex.split"><span class="pre"><code class="sourceCode python">shlex.split()</code></span></a>: Passing <span class="pre">`None`</span> for *s* argument now raises an exception, rather than reading <a href="../library/sys.html#sys.stdin" class="reference internal" title="sys.stdin"><span class="pre"><code class="sourceCode python">sys.stdin</code></span></a>. The feature was deprecated in Python 3.9. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/94352" class="reference external">gh-94352</a>.)

- The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module no longer accepts bytes-like paths, like <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> and <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> types: only the exact <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> type is accepted for bytes strings. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/98393" class="reference external">gh-98393</a>.)

- <a href="../library/syslog.html#syslog.openlog" class="reference internal" title="syslog.openlog"><span class="pre"><code class="sourceCode python">syslog.openlog()</code></span></a> and <a href="../library/syslog.html#syslog.closelog" class="reference internal" title="syslog.closelog"><span class="pre"><code class="sourceCode python">syslog.closelog()</code></span></a> now fail if used in subinterpreters. <a href="../library/syslog.html#syslog.syslog" class="reference internal" title="syslog.syslog"><span class="pre"><code class="sourceCode python">syslog.syslog()</code></span></a> may still be used in subinterpreters, but now only if <a href="../library/syslog.html#syslog.openlog" class="reference internal" title="syslog.openlog"><span class="pre"><code class="sourceCode python">syslog.openlog()</code></span></a> has already been called in the main interpreter. These new restrictions do not apply to the main interpreter, so only a very small set of users might be affected. This change helps with interpreter isolation. Furthermore, <a href="../library/syslog.html#module-syslog" class="reference internal" title="syslog: An interface to the Unix syslog library routines. (Unix)"><span class="pre"><code class="sourceCode python">syslog</code></span></a> is a wrapper around process-global resources, which are best managed from the main interpreter. (Contributed by Donghee Na in <a href="https://github.com/python/cpython/issues/99127" class="reference external">gh-99127</a>.)

- The undocumented locking behavior of <a href="../library/functools.html#functools.cached_property" class="reference internal" title="functools.cached_property"><span class="pre"><code class="sourceCode python">cached_property()</code></span></a> is removed, because it locked across all instances of the class, leading to high lock contention. This means that a cached property getter function could now run more than once for a single instance, if two threads race. For most simple cached properties (e.g. those that are idempotent and simply calculate a value based on other attributes of the instance) this will be fine. If synchronization is needed, implement locking within the cached property getter function or around multi-threaded access points.

- <a href="../library/sys.html#sys._current_exceptions" class="reference internal" title="sys._current_exceptions"><span class="pre"><code class="sourceCode python">sys._current_exceptions()</code></span></a> now returns a mapping from thread-id to an exception instance, rather than to a <span class="pre">`(typ,`</span>` `<span class="pre">`exc,`</span>` `<span class="pre">`tb)`</span> tuple. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/103176" class="reference external">gh-103176</a>.)

- When extracting tar files using <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> or <a href="../library/shutil.html#shutil.unpack_archive" class="reference internal" title="shutil.unpack_archive"><span class="pre"><code class="sourceCode python">shutil.unpack_archive()</code></span></a>, pass the *filter* argument to limit features that may be surprising or dangerous. See <a href="../library/tarfile.html#tarfile-extraction-filter" class="reference internal"><span class="std std-ref">Extraction filters</span></a> for details.

- The output of the <a href="../library/tokenize.html#tokenize.tokenize" class="reference internal" title="tokenize.tokenize"><span class="pre"><code class="sourceCode python">tokenize.tokenize()</code></span></a> and <a href="../library/tokenize.html#tokenize.generate_tokens" class="reference internal" title="tokenize.generate_tokens"><span class="pre"><code class="sourceCode python">tokenize.generate_tokens()</code></span></a> functions is now changed due to the changes introduced in <span id="index-44" class="target"></span><a href="https://peps.python.org/pep-0701/" class="pep reference external"><strong>PEP 701</strong></a>. This means that <span class="pre">`STRING`</span> tokens are not emitted any more for f-strings and the tokens described in <span id="index-45" class="target"></span><a href="https://peps.python.org/pep-0701/" class="pep reference external"><strong>PEP 701</strong></a> are now produced instead: <span class="pre">`FSTRING_START`</span>, <span class="pre">`FSTRING_MIDDLE`</span> and <span class="pre">`FSTRING_END`</span> are now emitted for f-string “string” parts in addition to the appropriate tokens for the tokenization in the expression components. For example for the f-string <span class="pre">`f"start`</span>` `<span class="pre">`{1+1}`</span>` `<span class="pre">`end"`</span> the old version of the tokenizer emitted:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      1,0-1,18:           STRING         'f"start {1+1} end"'

  </div>

  </div>

  while the new version emits:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      1,0-1,2:            FSTRING_START  'f"'
      1,2-1,8:            FSTRING_MIDDLE 'start '
      1,8-1,9:            OP             '{'
      1,9-1,10:           NUMBER         '1'
      1,10-1,11:          OP             '+'
      1,11-1,12:          NUMBER         '1'
      1,12-1,13:          OP             '}'
      1,13-1,17:          FSTRING_MIDDLE ' end'
      1,17-1,18:          FSTRING_END    '"'

  </div>

  </div>

  Additionally, there may be some minor behavioral changes as a consequence of the changes required to support <span id="index-46" class="target"></span><a href="https://peps.python.org/pep-0701/" class="pep reference external"><strong>PEP 701</strong></a>. Some of these changes include:

  - The <span class="pre">`type`</span> attribute of the tokens emitted when tokenizing some invalid Python characters such as <span class="pre">`!`</span> has changed from <span class="pre">`ERRORTOKEN`</span> to <span class="pre">`OP`</span>.

  - Incomplete single-line strings now also raise <a href="../library/tokenize.html#tokenize.TokenError" class="reference internal" title="tokenize.TokenError"><span class="pre"><code class="sourceCode python">tokenize.TokenError</code></span></a> as incomplete multiline strings do.

  - Some incomplete or invalid Python code now raises <a href="../library/tokenize.html#tokenize.TokenError" class="reference internal" title="tokenize.TokenError"><span class="pre"><code class="sourceCode python">tokenize.TokenError</code></span></a> instead of returning arbitrary <span class="pre">`ERRORTOKEN`</span> tokens when tokenizing it.

  - Mixing tabs and spaces as indentation in the same file is not supported anymore and will raise a <a href="../library/exceptions.html#TabError" class="reference internal" title="TabError"><span class="pre"><code class="sourceCode python"><span class="pp">TabError</span></code></span></a>.

- The <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a> module now expects the <span class="pre">`_thread`</span> module to have an <span class="pre">`_is_main_interpreter`</span> attribute. It is a function with no arguments that returns <span class="pre">`True`</span> if the current interpreter is the main interpreter.

  Any library or application that provides a custom <span class="pre">`_thread`</span> module should provide <span class="pre">`_is_main_interpreter()`</span>. (See <a href="https://github.com/python/cpython/issues/112826" class="reference external">gh-112826</a>.)

</div>

</div>

<div id="build-changes" class="section">

## Build Changes<a href="#build-changes" class="headerlink" title="Link to this heading">¶</a>

- Python no longer uses <span class="pre">`setup.py`</span> to build shared C extension modules. Build parameters like headers and libraries are detected in <span class="pre">`configure`</span> script. Extensions are built by <span class="pre">`Makefile`</span>. Most extensions use <span class="pre">`pkg-config`</span> and fall back to manual detection. (Contributed by Christian Heimes in <a href="https://github.com/python/cpython/issues/93939" class="reference external">gh-93939</a>.)

- <span class="pre">`va_start()`</span> with two parameters, like <span class="pre">`va_start(args,`</span>` `<span class="pre">`format),`</span> is now required to build Python. <span class="pre">`va_start()`</span> is no longer called with a single parameter. (Contributed by Kumar Aditya in <a href="https://github.com/python/cpython/issues/93207" class="reference external">gh-93207</a>.)

- CPython now uses the ThinLTO option as the default link time optimization policy if the Clang compiler accepts the flag. (Contributed by Donghee Na in <a href="https://github.com/python/cpython/issues/89536" class="reference external">gh-89536</a>.)

- Add <span class="pre">`COMPILEALL_OPTS`</span> variable in <span class="pre">`Makefile`</span> to override <a href="../library/compileall.html#module-compileall" class="reference internal" title="compileall: Tools for byte-compiling all Python source files in a directory tree."><span class="pre"><code class="sourceCode python">compileall</code></span></a> options (default: <span class="pre">`-j0`</span>) in <span class="pre">`make`</span>` `<span class="pre">`install`</span>. Also merged the 3 <span class="pre">`compileall`</span> commands into a single command to build .pyc files for all optimization levels (0, 1, 2) at once. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/99289" class="reference external">gh-99289</a>.)

- Add platform triplets for 64-bit LoongArch:

  - loongarch64-linux-gnusf

  - loongarch64-linux-gnuf32

  - loongarch64-linux-gnu

  (Contributed by Zhang Na in <a href="https://github.com/python/cpython/issues/90656" class="reference external">gh-90656</a>.)

- <span class="pre">`PYTHON_FOR_REGEN`</span> now require Python 3.10 or newer.

- Autoconf 2.71 and aclocal 1.16.4 is now required to regenerate <span class="pre">`!configure`</span>. (Contributed by Christian Heimes in <a href="https://github.com/python/cpython/issues/89886" class="reference external">gh-89886</a>.)

- Windows builds and macOS installers from python.org now use OpenSSL 3.0.

</div>

<div id="c-api-changes" class="section">

## C API Changes<a href="#c-api-changes" class="headerlink" title="Link to this heading">¶</a>

<div id="id4" class="section">

### New Features<a href="#id4" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-47" class="target"></span><a href="https://peps.python.org/pep-0697/" class="pep reference external"><strong>PEP 697</strong></a>: Introduce the <a href="../c-api/stable.html#unstable-c-api" class="reference internal"><span class="std std-ref">Unstable C API tier</span></a>, intended for low-level tools like debuggers and JIT compilers. This API may change in each minor release of CPython without deprecation warnings. Its contents are marked by the <span class="pre">`PyUnstable_`</span> prefix in names.

  Code object constructors:

  - <span class="pre">`PyUnstable_Code_New()`</span> (renamed from <span class="pre">`PyCode_New`</span>)

  - <span class="pre">`PyUnstable_Code_NewWithPosOnlyArgs()`</span> (renamed from <span class="pre">`PyCode_NewWithPosOnlyArgs`</span>)

  Extra storage for code objects (<span id="index-48" class="target"></span><a href="https://peps.python.org/pep-0523/" class="pep reference external"><strong>PEP 523</strong></a>):

  - <span class="pre">`PyUnstable_Eval_RequestCodeExtraIndex()`</span> (renamed from <span class="pre">`_PyEval_RequestCodeExtraIndex`</span>)

  - <span class="pre">`PyUnstable_Code_GetExtra()`</span> (renamed from <span class="pre">`_PyCode_GetExtra`</span>)

  - <span class="pre">`PyUnstable_Code_SetExtra()`</span> (renamed from <span class="pre">`_PyCode_SetExtra`</span>)

  The original names will continue to be available until the respective API changes.

  (Contributed by Petr Viktorin in <a href="https://github.com/python/cpython/issues/101101" class="reference external">gh-101101</a>.)

- <span id="index-49" class="target"></span><a href="https://peps.python.org/pep-0697/" class="pep reference external"><strong>PEP 697</strong></a>: Add an API for extending types whose instance memory layout is opaque:

  - <a href="../c-api/type.html#c.PyType_Spec.basicsize" class="reference internal" title="PyType_Spec.basicsize"><span class="pre"><code class="sourceCode c">PyType_Spec<span class="op">.</span>basicsize</code></span></a> can be zero or negative to specify inheriting or extending the base class size.

  - <a href="../c-api/object.html#c.PyObject_GetTypeData" class="reference internal" title="PyObject_GetTypeData"><span class="pre"><code class="sourceCode c">PyObject_GetTypeData<span class="op">()</span></code></span></a> and <a href="../c-api/object.html#c.PyType_GetTypeDataSize" class="reference internal" title="PyType_GetTypeDataSize"><span class="pre"><code class="sourceCode c">PyType_GetTypeDataSize<span class="op">()</span></code></span></a> added to allow access to subclass-specific instance data.

  - <a href="../c-api/typeobj.html#c.Py_TPFLAGS_ITEMS_AT_END" class="reference internal" title="Py_TPFLAGS_ITEMS_AT_END"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_ITEMS_AT_END</code></span></a> and <a href="../c-api/object.html#c.PyObject_GetItemData" class="reference internal" title="PyObject_GetItemData"><span class="pre"><code class="sourceCode c">PyObject_GetItemData<span class="op">()</span></code></span></a> added to allow safely extending certain variable-sized types, including <a href="../c-api/type.html#c.PyType_Type" class="reference internal" title="PyType_Type"><span class="pre"><code class="sourceCode c">PyType_Type</code></span></a>.

  - <a href="../c-api/structures.html#c.Py_RELATIVE_OFFSET" class="reference internal" title="Py_RELATIVE_OFFSET"><span class="pre"><code class="sourceCode c">Py_RELATIVE_OFFSET</code></span></a> added to allow defining <a href="../c-api/structures.html#c.PyMemberDef" class="reference internal" title="PyMemberDef"><span class="pre"><code class="sourceCode c">members</code></span></a> in terms of a subclass-specific struct.

  (Contributed by Petr Viktorin in <a href="https://github.com/python/cpython/issues/103509" class="reference external">gh-103509</a>.)

- Add the new <a href="../c-api/stable.html#limited-c-api" class="reference internal"><span class="std std-ref">limited C API</span></a> function <a href="../c-api/type.html#c.PyType_FromMetaclass" class="reference internal" title="PyType_FromMetaclass"><span class="pre"><code class="sourceCode c">PyType_FromMetaclass<span class="op">()</span></code></span></a>, which generalizes the existing <a href="../c-api/type.html#c.PyType_FromModuleAndSpec" class="reference internal" title="PyType_FromModuleAndSpec"><span class="pre"><code class="sourceCode c">PyType_FromModuleAndSpec<span class="op">()</span></code></span></a> using an additional metaclass argument. (Contributed by Wenzel Jakob in <a href="https://github.com/python/cpython/issues/93012" class="reference external">gh-93012</a>.)

- API for creating objects that can be called using <a href="../c-api/call.html#vectorcall" class="reference internal"><span class="std std-ref">the vectorcall protocol</span></a> was added to the <a href="../c-api/stable.html#stable" class="reference internal"><span class="std std-ref">Limited API</span></a>:

  - <a href="../c-api/typeobj.html#c.Py_TPFLAGS_HAVE_VECTORCALL" class="reference internal" title="Py_TPFLAGS_HAVE_VECTORCALL"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_HAVE_VECTORCALL</code></span></a>

  - <a href="../c-api/call.html#c.PyVectorcall_NARGS" class="reference internal" title="PyVectorcall_NARGS"><span class="pre"><code class="sourceCode c">PyVectorcall_NARGS<span class="op">()</span></code></span></a>

  - <a href="../c-api/call.html#c.PyVectorcall_Call" class="reference internal" title="PyVectorcall_Call"><span class="pre"><code class="sourceCode c">PyVectorcall_Call<span class="op">()</span></code></span></a>

  - <a href="../c-api/call.html#c.vectorcallfunc" class="reference internal" title="vectorcallfunc"><span class="pre"><code class="sourceCode c">vectorcallfunc</code></span></a>

  The <a href="../c-api/typeobj.html#c.Py_TPFLAGS_HAVE_VECTORCALL" class="reference internal" title="Py_TPFLAGS_HAVE_VECTORCALL"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_HAVE_VECTORCALL</code></span></a> flag is now removed from a class when the class’s <a href="../reference/datamodel.html#object.__call__" class="reference internal" title="object.__call__"><span class="pre"><code class="sourceCode python"><span class="fu">__call__</span>()</code></span></a> method is reassigned. This makes vectorcall safe to use with mutable types (i.e. heap types without the immutable flag, <a href="../c-api/typeobj.html#c.Py_TPFLAGS_IMMUTABLETYPE" class="reference internal" title="Py_TPFLAGS_IMMUTABLETYPE"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_IMMUTABLETYPE</code></span></a>). Mutable types that do not override <a href="../c-api/typeobj.html#c.PyTypeObject.tp_call" class="reference internal" title="PyTypeObject.tp_call"><span class="pre"><code class="sourceCode c">tp_call</code></span></a> now inherit the <span class="pre">`Py_TPFLAGS_HAVE_VECTORCALL`</span> flag. (Contributed by Petr Viktorin in <a href="https://github.com/python/cpython/issues/93274" class="reference external">gh-93274</a>.)

  The <a href="../c-api/typeobj.html#c.Py_TPFLAGS_MANAGED_DICT" class="reference internal" title="Py_TPFLAGS_MANAGED_DICT"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_MANAGED_DICT</code></span></a> and <a href="../c-api/typeobj.html#c.Py_TPFLAGS_MANAGED_WEAKREF" class="reference internal" title="Py_TPFLAGS_MANAGED_WEAKREF"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_MANAGED_WEAKREF</code></span></a> flags have been added. This allows extensions classes to support object <a href="../reference/datamodel.html#object.__dict__" class="reference internal" title="object.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a> and weakrefs with less bookkeeping, using less memory and with faster access.

- API for performing calls using <a href="../c-api/call.html#vectorcall" class="reference internal"><span class="std std-ref">the vectorcall protocol</span></a> was added to the <a href="../c-api/stable.html#stable" class="reference internal"><span class="std std-ref">Limited API</span></a>:

  - <a href="../c-api/call.html#c.PyObject_Vectorcall" class="reference internal" title="PyObject_Vectorcall"><span class="pre"><code class="sourceCode c">PyObject_Vectorcall<span class="op">()</span></code></span></a>

  - <a href="../c-api/call.html#c.PyObject_VectorcallMethod" class="reference internal" title="PyObject_VectorcallMethod"><span class="pre"><code class="sourceCode c">PyObject_VectorcallMethod<span class="op">()</span></code></span></a>

  - <a href="../c-api/call.html#c.PY_VECTORCALL_ARGUMENTS_OFFSET" class="reference internal" title="PY_VECTORCALL_ARGUMENTS_OFFSET"><span class="pre"><code class="sourceCode c">PY_VECTORCALL_ARGUMENTS_OFFSET</code></span></a>

  This means that both the incoming and outgoing ends of the vector call protocol are now available in the <a href="../c-api/stable.html#stable" class="reference internal"><span class="std std-ref">Limited API</span></a>. (Contributed by Wenzel Jakob in <a href="https://github.com/python/cpython/issues/98586" class="reference external">gh-98586</a>.)

- Add two new public functions, <a href="../c-api/init.html#c.PyEval_SetProfileAllThreads" class="reference internal" title="PyEval_SetProfileAllThreads"><span class="pre"><code class="sourceCode c">PyEval_SetProfileAllThreads<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyEval_SetTraceAllThreads" class="reference internal" title="PyEval_SetTraceAllThreads"><span class="pre"><code class="sourceCode c">PyEval_SetTraceAllThreads<span class="op">()</span></code></span></a>, that allow to set tracing and profiling functions in all running threads in addition to the calling one. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/93503" class="reference external">gh-93503</a>.)

- Add new function <a href="../c-api/function.html#c.PyFunction_SetVectorcall" class="reference internal" title="PyFunction_SetVectorcall"><span class="pre"><code class="sourceCode c">PyFunction_SetVectorcall<span class="op">()</span></code></span></a> to the C API which sets the vectorcall field of a given <a href="../c-api/function.html#c.PyFunctionObject" class="reference internal" title="PyFunctionObject"><span class="pre"><code class="sourceCode c">PyFunctionObject</code></span></a>. (Contributed by Andrew Frost in <a href="https://github.com/python/cpython/issues/92257" class="reference external">gh-92257</a>.)

- The C API now permits registering callbacks via <a href="../c-api/dict.html#c.PyDict_AddWatcher" class="reference internal" title="PyDict_AddWatcher"><span class="pre"><code class="sourceCode c">PyDict_AddWatcher<span class="op">()</span></code></span></a>, <a href="../c-api/dict.html#c.PyDict_Watch" class="reference internal" title="PyDict_Watch"><span class="pre"><code class="sourceCode c">PyDict_Watch<span class="op">()</span></code></span></a> and related APIs to be called whenever a dictionary is modified. This is intended for use by optimizing interpreters, JIT compilers, or debuggers. (Contributed by Carl Meyer in <a href="https://github.com/python/cpython/issues/91052" class="reference external">gh-91052</a>.)

- Add <a href="../c-api/type.html#c.PyType_AddWatcher" class="reference internal" title="PyType_AddWatcher"><span class="pre"><code class="sourceCode c">PyType_AddWatcher<span class="op">()</span></code></span></a> and <a href="../c-api/type.html#c.PyType_Watch" class="reference internal" title="PyType_Watch"><span class="pre"><code class="sourceCode c">PyType_Watch<span class="op">()</span></code></span></a> API to register callbacks to receive notification on changes to a type. (Contributed by Carl Meyer in <a href="https://github.com/python/cpython/issues/91051" class="reference external">gh-91051</a>.)

- Add <a href="../c-api/code.html#c.PyCode_AddWatcher" class="reference internal" title="PyCode_AddWatcher"><span class="pre"><code class="sourceCode c">PyCode_AddWatcher<span class="op">()</span></code></span></a> and <a href="../c-api/code.html#c.PyCode_ClearWatcher" class="reference internal" title="PyCode_ClearWatcher"><span class="pre"><code class="sourceCode c">PyCode_ClearWatcher<span class="op">()</span></code></span></a> APIs to register callbacks to receive notification on creation and destruction of code objects. (Contributed by Itamar Oren in <a href="https://github.com/python/cpython/issues/91054" class="reference external">gh-91054</a>.)

- Add <a href="../c-api/frame.html#c.PyFrame_GetVar" class="reference internal" title="PyFrame_GetVar"><span class="pre"><code class="sourceCode c">PyFrame_GetVar<span class="op">()</span></code></span></a> and <a href="../c-api/frame.html#c.PyFrame_GetVarString" class="reference internal" title="PyFrame_GetVarString"><span class="pre"><code class="sourceCode c">PyFrame_GetVarString<span class="op">()</span></code></span></a> functions to get a frame variable by its name. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/91248" class="reference external">gh-91248</a>.)

- Add <a href="../c-api/exceptions.html#c.PyErr_GetRaisedException" class="reference internal" title="PyErr_GetRaisedException"><span class="pre"><code class="sourceCode c">PyErr_GetRaisedException<span class="op">()</span></code></span></a> and <a href="../c-api/exceptions.html#c.PyErr_SetRaisedException" class="reference internal" title="PyErr_SetRaisedException"><span class="pre"><code class="sourceCode c">PyErr_SetRaisedException<span class="op">()</span></code></span></a> for saving and restoring the current exception. These functions return and accept a single exception object, rather than the triple arguments of the now-deprecated <a href="../c-api/exceptions.html#c.PyErr_Fetch" class="reference internal" title="PyErr_Fetch"><span class="pre"><code class="sourceCode c">PyErr_Fetch<span class="op">()</span></code></span></a> and <a href="../c-api/exceptions.html#c.PyErr_Restore" class="reference internal" title="PyErr_Restore"><span class="pre"><code class="sourceCode c">PyErr_Restore<span class="op">()</span></code></span></a>. This is less error prone and a bit more efficient. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/101578" class="reference external">gh-101578</a>.)

- Add <span class="pre">`_PyErr_ChainExceptions1`</span>, which takes an exception instance, to replace the legacy-API <span class="pre">`_PyErr_ChainExceptions`</span>, which is now deprecated. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/101578" class="reference external">gh-101578</a>.)

- Add <a href="../c-api/exceptions.html#c.PyException_GetArgs" class="reference internal" title="PyException_GetArgs"><span class="pre"><code class="sourceCode c">PyException_GetArgs<span class="op">()</span></code></span></a> and <a href="../c-api/exceptions.html#c.PyException_SetArgs" class="reference internal" title="PyException_SetArgs"><span class="pre"><code class="sourceCode c">PyException_SetArgs<span class="op">()</span></code></span></a> as convenience functions for retrieving and modifying the <a href="../library/exceptions.html#BaseException.args" class="reference internal" title="BaseException.args"><span class="pre"><code class="sourceCode python">args</code></span></a> passed to the exception’s constructor. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/101578" class="reference external">gh-101578</a>.)

- Add <a href="../c-api/exceptions.html#c.PyErr_DisplayException" class="reference internal" title="PyErr_DisplayException"><span class="pre"><code class="sourceCode c">PyErr_DisplayException<span class="op">()</span></code></span></a>, which takes an exception instance, to replace the legacy-api <span class="pre">`PyErr_Display()`</span>. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/102755" class="reference external">gh-102755</a>).

<!-- -->

- <span id="index-50" class="target"></span><a href="https://peps.python.org/pep-0683/" class="pep reference external"><strong>PEP 683</strong></a>: Introduce *Immortal Objects*, which allows objects to bypass reference counts, and related changes to the C-API:

  - <span class="pre">`_Py_IMMORTAL_REFCNT`</span>: The reference count that defines an object  
    as immortal.

  - <span class="pre">`_Py_IsImmortal`</span> Checks if an object has the immortal reference count.

  - <span class="pre">`PyObject_HEAD_INIT`</span> This will now initialize reference count to  
    <span class="pre">`_Py_IMMORTAL_REFCNT`</span> when used with <span class="pre">`Py_BUILD_CORE`</span>.

  - <span class="pre">`SSTATE_INTERNED_IMMORTAL`</span> An identifier for interned unicode objects  
    that are immortal.

  - <span class="pre">`SSTATE_INTERNED_IMMORTAL_STATIC`</span> An identifier for interned unicode  
    objects that are immortal and static

  - <span class="pre">`sys.getunicodeinternedsize`</span> This returns the total number of unicode  
    objects that have been interned. This is now needed for <span class="pre">`refleak.py`</span> to correctly track reference counts and allocated blocks

  (Contributed by Eddie Elizondo in <a href="https://github.com/python/cpython/issues/84436" class="reference external">gh-84436</a>.)

- <span id="index-51" class="target"></span><a href="https://peps.python.org/pep-0684/" class="pep reference external"><strong>PEP 684</strong></a>: Add the new <a href="../c-api/init.html#c.Py_NewInterpreterFromConfig" class="reference internal" title="Py_NewInterpreterFromConfig"><span class="pre"><code class="sourceCode c">Py_NewInterpreterFromConfig<span class="op">()</span></code></span></a> function and <a href="../c-api/init.html#c.PyInterpreterConfig" class="reference internal" title="PyInterpreterConfig"><span class="pre"><code class="sourceCode c">PyInterpreterConfig</code></span></a>, which may be used to create sub-interpreters with their own GILs. (See <a href="#whatsnew312-pep684" class="reference internal"><span class="std std-ref">PEP 684: A Per-Interpreter GIL</span></a> for more info.) (Contributed by Eric Snow in <a href="https://github.com/python/cpython/issues/104110" class="reference external">gh-104110</a>.)

- In the limited C API version 3.12, <a href="../c-api/refcounting.html#c.Py_INCREF" class="reference internal" title="Py_INCREF"><span class="pre"><code class="sourceCode c">Py_INCREF<span class="op">()</span></code></span></a> and <a href="../c-api/refcounting.html#c.Py_DECREF" class="reference internal" title="Py_DECREF"><span class="pre"><code class="sourceCode c">Py_DECREF<span class="op">()</span></code></span></a> functions are now implemented as opaque function calls to hide implementation details. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105387" class="reference external">gh-105387</a>.)

</div>

<div id="id5" class="section">

### Porting to Python 3.12<a href="#id5" class="headerlink" title="Link to this heading">¶</a>

- Legacy Unicode APIs based on <span class="pre">`Py_UNICODE*`</span> representation has been removed. Please migrate to APIs based on UTF-8 or <span class="pre">`wchar_t*`</span>.

- Argument parsing functions like <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple<span class="op">()</span></code></span></a> doesn’t support <span class="pre">`Py_UNICODE*`</span> based format (e.g. <span class="pre">`u`</span>, <span class="pre">`Z`</span>) anymore. Please migrate to other formats for Unicode like <span class="pre">`s`</span>, <span class="pre">`z`</span>, <span class="pre">`es`</span>, and <span class="pre">`U`</span>.

- <span class="pre">`tp_weaklist`</span> for all static builtin types is always <span class="pre">`NULL`</span>. This is an internal-only field on <span class="pre">`PyTypeObject`</span> but we’re pointing out the change in case someone happens to be accessing the field directly anyway. To avoid breakage, consider using the existing public C-API instead, or, if necessary, the (internal-only) <span class="pre">`_PyObject_GET_WEAKREFS_LISTPTR()`</span> macro.

- This internal-only <a href="../c-api/typeobj.html#c.PyTypeObject.tp_subclasses" class="reference internal" title="PyTypeObject.tp_subclasses"><span class="pre"><code class="sourceCode c">PyTypeObject<span class="op">.</span>tp_subclasses</code></span></a> may now not be a valid object pointer. Its type was changed to <span class="c-expr sig sig-inline c"><span class="kt">void</span><span class="p">\*</span></span> to reflect this. We mention this in case someone happens to be accessing the internal-only field directly.

  To get a list of subclasses, call the Python method <a href="../reference/datamodel.html#type.__subclasses__" class="reference internal" title="type.__subclasses__"><span class="pre"><code class="sourceCode python">__subclasses__()</code></span></a> (using <a href="../c-api/call.html#c.PyObject_CallMethod" class="reference internal" title="PyObject_CallMethod"><span class="pre"><code class="sourceCode c">PyObject_CallMethod<span class="op">()</span></code></span></a>, for example).

- Add support of more formatting options (left aligning, octals, uppercase hexadecimals, <span class="pre">`intmax_t`</span>, <span class="pre">`ptrdiff_t`</span>, <span class="pre">`wchar_t`</span> C strings, variable width and precision) in <a href="../c-api/unicode.html#c.PyUnicode_FromFormat" class="reference internal" title="PyUnicode_FromFormat"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormat<span class="op">()</span></code></span></a> and <a href="../c-api/unicode.html#c.PyUnicode_FromFormatV" class="reference internal" title="PyUnicode_FromFormatV"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormatV<span class="op">()</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/98836" class="reference external">gh-98836</a>.)

- An unrecognized format character in <a href="../c-api/unicode.html#c.PyUnicode_FromFormat" class="reference internal" title="PyUnicode_FromFormat"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormat<span class="op">()</span></code></span></a> and <a href="../c-api/unicode.html#c.PyUnicode_FromFormatV" class="reference internal" title="PyUnicode_FromFormatV"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormatV<span class="op">()</span></code></span></a> now sets a <a href="../library/exceptions.html#SystemError" class="reference internal" title="SystemError"><span class="pre"><code class="sourceCode python"><span class="pp">SystemError</span></code></span></a>. In previous versions it caused all the rest of the format string to be copied as-is to the result string, and any extra arguments discarded. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/95781" class="reference external">gh-95781</a>.)

- Fix wrong sign placement in <a href="../c-api/unicode.html#c.PyUnicode_FromFormat" class="reference internal" title="PyUnicode_FromFormat"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormat<span class="op">()</span></code></span></a> and <a href="../c-api/unicode.html#c.PyUnicode_FromFormatV" class="reference internal" title="PyUnicode_FromFormatV"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormatV<span class="op">()</span></code></span></a>. (Contributed by Philip Georgi in <a href="https://github.com/python/cpython/issues/95504" class="reference external">gh-95504</a>.)

- Extension classes wanting to add a <a href="../reference/datamodel.html#object.__dict__" class="reference internal" title="object.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a> or weak reference slot should use <a href="../c-api/typeobj.html#c.Py_TPFLAGS_MANAGED_DICT" class="reference internal" title="Py_TPFLAGS_MANAGED_DICT"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_MANAGED_DICT</code></span></a> and <a href="../c-api/typeobj.html#c.Py_TPFLAGS_MANAGED_WEAKREF" class="reference internal" title="Py_TPFLAGS_MANAGED_WEAKREF"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_MANAGED_WEAKREF</code></span></a> instead of <span class="pre">`tp_dictoffset`</span> and <span class="pre">`tp_weaklistoffset`</span>, respectively. The use of <span class="pre">`tp_dictoffset`</span> and <span class="pre">`tp_weaklistoffset`</span> is still supported, but does not fully support multiple inheritance (<a href="https://github.com/python/cpython/issues/95589" class="reference external">gh-95589</a>), and performance may be worse. Classes declaring <a href="../c-api/typeobj.html#c.Py_TPFLAGS_MANAGED_DICT" class="reference internal" title="Py_TPFLAGS_MANAGED_DICT"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_MANAGED_DICT</code></span></a> must call <span class="pre">`_PyObject_VisitManagedDict()`</span> and <span class="pre">`_PyObject_ClearManagedDict()`</span> to traverse and clear their instance’s dictionaries. To clear weakrefs, call <a href="../c-api/weakref.html#c.PyObject_ClearWeakRefs" class="reference internal" title="PyObject_ClearWeakRefs"><span class="pre"><code class="sourceCode c">PyObject_ClearWeakRefs<span class="op">()</span></code></span></a>, as before.

- The <a href="../c-api/unicode.html#c.PyUnicode_FSDecoder" class="reference internal" title="PyUnicode_FSDecoder"><span class="pre"><code class="sourceCode c">PyUnicode_FSDecoder<span class="op">()</span></code></span></a> function no longer accepts bytes-like paths, like <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> and <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> types: only the exact <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> type is accepted for bytes strings. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/98393" class="reference external">gh-98393</a>.)

- The <a href="../c-api/refcounting.html#c.Py_CLEAR" class="reference internal" title="Py_CLEAR"><span class="pre"><code class="sourceCode c">Py_CLEAR</code></span></a>, <a href="../c-api/refcounting.html#c.Py_SETREF" class="reference internal" title="Py_SETREF"><span class="pre"><code class="sourceCode c">Py_SETREF</code></span></a> and <a href="../c-api/refcounting.html#c.Py_XSETREF" class="reference internal" title="Py_XSETREF"><span class="pre"><code class="sourceCode c">Py_XSETREF</code></span></a> macros now only evaluate their arguments once. If an argument has side effects, these side effects are no longer duplicated. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/98724" class="reference external">gh-98724</a>.)

- The interpreter’s error indicator is now always normalized. This means that <a href="../c-api/exceptions.html#c.PyErr_SetObject" class="reference internal" title="PyErr_SetObject"><span class="pre"><code class="sourceCode c">PyErr_SetObject<span class="op">()</span></code></span></a>, <a href="../c-api/exceptions.html#c.PyErr_SetString" class="reference internal" title="PyErr_SetString"><span class="pre"><code class="sourceCode c">PyErr_SetString<span class="op">()</span></code></span></a> and the other functions that set the error indicator now normalize the exception before storing it. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/101578" class="reference external">gh-101578</a>.)

- <span class="pre">`_Py_RefTotal`</span> is no longer authoritative and only kept around for ABI compatibility. Note that it is an internal global and only available on debug builds. If you happen to be using it then you’ll need to start using <span class="pre">`_Py_GetGlobalRefTotal()`</span>.

- The following functions now select an appropriate metaclass for the newly created type:

  - <a href="../c-api/type.html#c.PyType_FromSpec" class="reference internal" title="PyType_FromSpec"><span class="pre"><code class="sourceCode c">PyType_FromSpec<span class="op">()</span></code></span></a>

  - <a href="../c-api/type.html#c.PyType_FromSpecWithBases" class="reference internal" title="PyType_FromSpecWithBases"><span class="pre"><code class="sourceCode c">PyType_FromSpecWithBases<span class="op">()</span></code></span></a>

  - <a href="../c-api/type.html#c.PyType_FromModuleAndSpec" class="reference internal" title="PyType_FromModuleAndSpec"><span class="pre"><code class="sourceCode c">PyType_FromModuleAndSpec<span class="op">()</span></code></span></a>

  Creating classes whose metaclass overrides <a href="../c-api/typeobj.html#c.PyTypeObject.tp_new" class="reference internal" title="PyTypeObject.tp_new"><span class="pre"><code class="sourceCode c">tp_new</code></span></a> is deprecated, and in Python 3.14+ it will be disallowed. Note that these functions ignore <span class="pre">`tp_new`</span> of the metaclass, possibly allowing incomplete initialization.

  Note that <a href="../c-api/type.html#c.PyType_FromMetaclass" class="reference internal" title="PyType_FromMetaclass"><span class="pre"><code class="sourceCode c">PyType_FromMetaclass<span class="op">()</span></code></span></a> (added in Python 3.12) already disallows creating classes whose metaclass overrides <span class="pre">`tp_new`</span> (<a href="../reference/datamodel.html#object.__new__" class="reference internal" title="object.__new__"><span class="pre"><code class="sourceCode python"><span class="fu">__new__</span>()</code></span></a> in Python).

  Since <span class="pre">`tp_new`</span> overrides almost everything <span class="pre">`PyType_From*`</span> functions do, the two are incompatible with each other. The existing behavior – ignoring the metaclass for several steps of type creation – is unsafe in general, since (meta)classes assume that <span class="pre">`tp_new`</span> was called. There is no simple general workaround. One of the following may work for you:

  - If you control the metaclass, avoid using <span class="pre">`tp_new`</span> in it:

    - If initialization can be skipped, it can be done in <a href="../c-api/typeobj.html#c.PyTypeObject.tp_init" class="reference internal" title="PyTypeObject.tp_init"><span class="pre"><code class="sourceCode c">tp_init</code></span></a> instead.

    - If the metaclass doesn’t need to be instantiated from Python, set its <span class="pre">`tp_new`</span> to <span class="pre">`NULL`</span> using the <a href="../c-api/typeobj.html#c.Py_TPFLAGS_DISALLOW_INSTANTIATION" class="reference internal" title="Py_TPFLAGS_DISALLOW_INSTANTIATION"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_DISALLOW_INSTANTIATION</code></span></a> flag. This makes it acceptable for <span class="pre">`PyType_From*`</span> functions.

  - Avoid <span class="pre">`PyType_From*`</span> functions: if you don’t need C-specific features (slots or setting the instance size), create types by <a href="../c-api/call.html#call" class="reference internal"><span class="std std-ref">calling</span></a> the metaclass.

  - If you *know* the <span class="pre">`tp_new`</span> can be skipped safely, filter the deprecation warning out using <a href="../library/warnings.html#warnings.catch_warnings" class="reference internal" title="warnings.catch_warnings"><span class="pre"><code class="sourceCode python">warnings.catch_warnings()</code></span></a> from Python.

- <a href="../c-api/veryhigh.html#c.PyOS_InputHook" class="reference internal" title="PyOS_InputHook"><span class="pre"><code class="sourceCode c">PyOS_InputHook</code></span></a> and <a href="../c-api/veryhigh.html#c.PyOS_ReadlineFunctionPointer" class="reference internal" title="PyOS_ReadlineFunctionPointer"><span class="pre"><code class="sourceCode c">PyOS_ReadlineFunctionPointer</code></span></a> are no longer called in <a href="../c-api/init.html#sub-interpreter-support" class="reference internal"><span class="std std-ref">subinterpreters</span></a>. This is because clients generally rely on process-wide global state (since these callbacks have no way of recovering extension module state).

  This also avoids situations where extensions may find themselves running in a subinterpreter that they don’t support (or haven’t yet been loaded in). See <a href="https://github.com/python/cpython/issues/104668" class="reference external">gh-104668</a> for more info.

- <a href="../c-api/long.html#c.PyLongObject" class="reference internal" title="PyLongObject"><span class="pre"><code class="sourceCode c">PyLongObject</code></span></a> has had its internals changed for better performance. Although the internals of <a href="../c-api/long.html#c.PyLongObject" class="reference internal" title="PyLongObject"><span class="pre"><code class="sourceCode c">PyLongObject</code></span></a> are private, they are used by some extension modules. The internal fields should no longer be accessed directly, instead the API functions beginning <span class="pre">`PyLong_...`</span> should be used instead. Two new *unstable* API functions are provided for efficient access to the value of <a href="../c-api/long.html#c.PyLongObject" class="reference internal" title="PyLongObject"><span class="pre"><code class="sourceCode c">PyLongObject</code></span></a>s which fit into a single machine word:

  - <a href="../c-api/long.html#c.PyUnstable_Long_IsCompact" class="reference internal" title="PyUnstable_Long_IsCompact"><span class="pre"><code class="sourceCode c">PyUnstable_Long_IsCompact<span class="op">()</span></code></span></a>

  - <a href="../c-api/long.html#c.PyUnstable_Long_CompactValue" class="reference internal" title="PyUnstable_Long_CompactValue"><span class="pre"><code class="sourceCode c">PyUnstable_Long_CompactValue<span class="op">()</span></code></span></a>

- Custom allocators, set via <a href="../c-api/memory.html#c.PyMem_SetAllocator" class="reference internal" title="PyMem_SetAllocator"><span class="pre"><code class="sourceCode c">PyMem_SetAllocator<span class="op">()</span></code></span></a>, are now required to be thread-safe, regardless of memory domain. Allocators that don’t have their own state, including “hooks”, are not affected. If your custom allocator is not already thread-safe and you need guidance then please create a new GitHub issue and CC <span class="pre">`@ericsnowcurrently`</span>.

</div>

<div id="id6" class="section">

### Deprecated<a href="#id6" class="headerlink" title="Link to this heading">¶</a>

- In accordance with <span id="index-52" class="target"></span><a href="https://peps.python.org/pep-0699/" class="pep reference external"><strong>PEP 699</strong></a>, the <span class="pre">`ma_version_tag`</span> field in <a href="../c-api/dict.html#c.PyDictObject" class="reference internal" title="PyDictObject"><span class="pre"><code class="sourceCode c">PyDictObject</code></span></a> is deprecated for extension modules. Accessing this field will generate a compiler warning at compile time. This field will be removed in Python 3.14. (Contributed by Ramvikrams and Kumar Aditya in <a href="https://github.com/python/cpython/issues/101193" class="reference external">gh-101193</a>. PEP by Ken Jin.)

- Deprecate global configuration variable:

  - <a href="../c-api/init.html#c.Py_DebugFlag" class="reference internal" title="Py_DebugFlag"><span class="pre"><code class="sourceCode c">Py_DebugFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.parser_debug" class="reference internal" title="PyConfig.parser_debug"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>parser_debug</code></span></a>

  - <a href="../c-api/init.html#c.Py_VerboseFlag" class="reference internal" title="Py_VerboseFlag"><span class="pre"><code class="sourceCode c">Py_VerboseFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.verbose" class="reference internal" title="PyConfig.verbose"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>verbose</code></span></a>

  - <a href="../c-api/init.html#c.Py_QuietFlag" class="reference internal" title="Py_QuietFlag"><span class="pre"><code class="sourceCode c">Py_QuietFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.quiet" class="reference internal" title="PyConfig.quiet"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>quiet</code></span></a>

  - <a href="../c-api/init.html#c.Py_InteractiveFlag" class="reference internal" title="Py_InteractiveFlag"><span class="pre"><code class="sourceCode c">Py_InteractiveFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.interactive" class="reference internal" title="PyConfig.interactive"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>interactive</code></span></a>

  - <a href="../c-api/init.html#c.Py_InspectFlag" class="reference internal" title="Py_InspectFlag"><span class="pre"><code class="sourceCode c">Py_InspectFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.inspect" class="reference internal" title="PyConfig.inspect"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>inspect</code></span></a>

  - <a href="../c-api/init.html#c.Py_OptimizeFlag" class="reference internal" title="Py_OptimizeFlag"><span class="pre"><code class="sourceCode c">Py_OptimizeFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.optimization_level" class="reference internal" title="PyConfig.optimization_level"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>optimization_level</code></span></a>

  - <a href="../c-api/init.html#c.Py_NoSiteFlag" class="reference internal" title="Py_NoSiteFlag"><span class="pre"><code class="sourceCode c">Py_NoSiteFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.site_import" class="reference internal" title="PyConfig.site_import"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>site_import</code></span></a>

  - <a href="../c-api/init.html#c.Py_BytesWarningFlag" class="reference internal" title="Py_BytesWarningFlag"><span class="pre"><code class="sourceCode c">Py_BytesWarningFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.bytes_warning" class="reference internal" title="PyConfig.bytes_warning"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>bytes_warning</code></span></a>

  - <a href="../c-api/init.html#c.Py_FrozenFlag" class="reference internal" title="Py_FrozenFlag"><span class="pre"><code class="sourceCode c">Py_FrozenFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.pathconfig_warnings" class="reference internal" title="PyConfig.pathconfig_warnings"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>pathconfig_warnings</code></span></a>

  - <a href="../c-api/init.html#c.Py_IgnoreEnvironmentFlag" class="reference internal" title="Py_IgnoreEnvironmentFlag"><span class="pre"><code class="sourceCode c">Py_IgnoreEnvironmentFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.use_environment" class="reference internal" title="PyConfig.use_environment"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>use_environment</code></span></a>

  - <a href="../c-api/init.html#c.Py_DontWriteBytecodeFlag" class="reference internal" title="Py_DontWriteBytecodeFlag"><span class="pre"><code class="sourceCode c">Py_DontWriteBytecodeFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.write_bytecode" class="reference internal" title="PyConfig.write_bytecode"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>write_bytecode</code></span></a>

  - <a href="../c-api/init.html#c.Py_NoUserSiteDirectory" class="reference internal" title="Py_NoUserSiteDirectory"><span class="pre"><code class="sourceCode c">Py_NoUserSiteDirectory</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.user_site_directory" class="reference internal" title="PyConfig.user_site_directory"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>user_site_directory</code></span></a>

  - <a href="../c-api/init.html#c.Py_UnbufferedStdioFlag" class="reference internal" title="Py_UnbufferedStdioFlag"><span class="pre"><code class="sourceCode c">Py_UnbufferedStdioFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.buffered_stdio" class="reference internal" title="PyConfig.buffered_stdio"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>buffered_stdio</code></span></a>

  - <a href="../c-api/init.html#c.Py_HashRandomizationFlag" class="reference internal" title="Py_HashRandomizationFlag"><span class="pre"><code class="sourceCode c">Py_HashRandomizationFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.use_hash_seed" class="reference internal" title="PyConfig.use_hash_seed"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>use_hash_seed</code></span></a> and <a href="../c-api/init_config.html#c.PyConfig.hash_seed" class="reference internal" title="PyConfig.hash_seed"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>hash_seed</code></span></a>

  - <a href="../c-api/init.html#c.Py_IsolatedFlag" class="reference internal" title="Py_IsolatedFlag"><span class="pre"><code class="sourceCode c">Py_IsolatedFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.isolated" class="reference internal" title="PyConfig.isolated"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>isolated</code></span></a>

  - <a href="../c-api/init.html#c.Py_LegacyWindowsFSEncodingFlag" class="reference internal" title="Py_LegacyWindowsFSEncodingFlag"><span class="pre"><code class="sourceCode c">Py_LegacyWindowsFSEncodingFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyPreConfig.legacy_windows_fs_encoding" class="reference internal" title="PyPreConfig.legacy_windows_fs_encoding"><span class="pre"><code class="sourceCode c">PyPreConfig<span class="op">.</span>legacy_windows_fs_encoding</code></span></a>

  - <a href="../c-api/init.html#c.Py_LegacyWindowsStdioFlag" class="reference internal" title="Py_LegacyWindowsStdioFlag"><span class="pre"><code class="sourceCode c">Py_LegacyWindowsStdioFlag</code></span></a>: use <a href="../c-api/init_config.html#c.PyConfig.legacy_windows_stdio" class="reference internal" title="PyConfig.legacy_windows_stdio"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>legacy_windows_stdio</code></span></a>

  - <span class="pre">`Py_FileSystemDefaultEncoding`</span>: use <a href="../c-api/init_config.html#c.PyConfig.filesystem_encoding" class="reference internal" title="PyConfig.filesystem_encoding"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>filesystem_encoding</code></span></a>

  - <span class="pre">`Py_HasFileSystemDefaultEncoding`</span>: use <a href="../c-api/init_config.html#c.PyConfig.filesystem_encoding" class="reference internal" title="PyConfig.filesystem_encoding"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>filesystem_encoding</code></span></a>

  - <span class="pre">`Py_FileSystemDefaultEncodeErrors`</span>: use <a href="../c-api/init_config.html#c.PyConfig.filesystem_errors" class="reference internal" title="PyConfig.filesystem_errors"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>filesystem_errors</code></span></a>

  - <span class="pre">`Py_UTF8Mode`</span>: use <a href="../c-api/init_config.html#c.PyPreConfig.utf8_mode" class="reference internal" title="PyPreConfig.utf8_mode"><span class="pre"><code class="sourceCode c">PyPreConfig<span class="op">.</span>utf8_mode</code></span></a> (see <a href="../c-api/init_config.html#c.Py_PreInitialize" class="reference internal" title="Py_PreInitialize"><span class="pre"><code class="sourceCode c">Py_PreInitialize<span class="op">()</span></code></span></a>)

  The <a href="../c-api/init.html#c.Py_InitializeFromConfig" class="reference internal" title="Py_InitializeFromConfig"><span class="pre"><code class="sourceCode c">Py_InitializeFromConfig<span class="op">()</span></code></span></a> API should be used with <a href="../c-api/init_config.html#c.PyConfig" class="reference internal" title="PyConfig"><span class="pre"><code class="sourceCode c">PyConfig</code></span></a> instead. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/77782" class="reference external">gh-77782</a>.)

- Creating <a href="../c-api/typeobj.html#c.Py_TPFLAGS_IMMUTABLETYPE" class="reference internal" title="Py_TPFLAGS_IMMUTABLETYPE"><span class="pre"><code class="sourceCode c">immutable</code></span><code class="sourceCode c"> </code><span class="pre"><code class="sourceCode c">types</code></span></a> with mutable bases is deprecated and will be disabled in Python 3.14. (<a href="https://github.com/python/cpython/issues/95388" class="reference external">gh-95388</a>)

- The <span class="pre">`structmember.h`</span> header is deprecated, though it continues to be available and there are no plans to remove it.

  Its contents are now available just by including <span class="pre">`Python.h`</span>, with a <span class="pre">`Py`</span> prefix added if it was missing:

  - <a href="../c-api/structures.html#c.PyMemberDef" class="reference internal" title="PyMemberDef"><span class="pre"><code class="sourceCode c">PyMemberDef</code></span></a>, <a href="../c-api/structures.html#c.PyMember_GetOne" class="reference internal" title="PyMember_GetOne"><span class="pre"><code class="sourceCode c">PyMember_GetOne<span class="op">()</span></code></span></a> and <a href="../c-api/structures.html#c.PyMember_SetOne" class="reference internal" title="PyMember_SetOne"><span class="pre"><code class="sourceCode c">PyMember_SetOne<span class="op">()</span></code></span></a>

  - Type macros like <a href="../c-api/structures.html#c.Py_T_INT" class="reference internal" title="Py_T_INT"><span class="pre"><code class="sourceCode c">Py_T_INT</code></span></a>, <a href="../c-api/structures.html#c.Py_T_DOUBLE" class="reference internal" title="Py_T_DOUBLE"><span class="pre"><code class="sourceCode c">Py_T_DOUBLE</code></span></a>, etc. (previously <span class="pre">`T_INT`</span>, <span class="pre">`T_DOUBLE`</span>, etc.)

  - The flags <a href="../c-api/structures.html#c.Py_READONLY" class="reference internal" title="Py_READONLY"><span class="pre"><code class="sourceCode c">Py_READONLY</code></span></a> (previously <span class="pre">`READONLY`</span>) and <a href="../c-api/structures.html#c.Py_AUDIT_READ" class="reference internal" title="Py_AUDIT_READ"><span class="pre"><code class="sourceCode c">Py_AUDIT_READ</code></span></a> (previously all uppercase)

  Several items are not exposed from <span class="pre">`Python.h`</span>:

  - <a href="../c-api/structures.html#c.T_OBJECT" class="reference internal" title="T_OBJECT"><span class="pre"><code class="sourceCode c">T_OBJECT</code></span></a> (use <a href="../c-api/structures.html#c.Py_T_OBJECT_EX" class="reference internal" title="Py_T_OBJECT_EX"><span class="pre"><code class="sourceCode c">Py_T_OBJECT_EX</code></span></a>)

  - <a href="../c-api/structures.html#c.T_NONE" class="reference internal" title="T_NONE"><span class="pre"><code class="sourceCode c">T_NONE</code></span></a> (previously undocumented, and pretty quirky)

  - The macro <span class="pre">`WRITE_RESTRICTED`</span> which does nothing.

  - The macros <span class="pre">`RESTRICTED`</span> and <span class="pre">`READ_RESTRICTED`</span>, equivalents of <a href="../c-api/structures.html#c.Py_AUDIT_READ" class="reference internal" title="Py_AUDIT_READ"><span class="pre"><code class="sourceCode c">Py_AUDIT_READ</code></span></a>.

  - In some configurations, <span class="pre">`<stddef.h>`</span> is not included from <span class="pre">`Python.h`</span>. It should be included manually when using <span class="pre">`offsetof()`</span>.

  The deprecated header continues to provide its original contents under the original names. Your old code can stay unchanged, unless the extra include and non-namespaced macros bother you greatly.

  (Contributed in <a href="https://github.com/python/cpython/issues/47146" class="reference external">gh-47146</a> by Petr Viktorin, based on earlier work by Alexander Belopolsky and Matthias Braun.)

- <a href="../c-api/exceptions.html#c.PyErr_Fetch" class="reference internal" title="PyErr_Fetch"><span class="pre"><code class="sourceCode c">PyErr_Fetch<span class="op">()</span></code></span></a> and <a href="../c-api/exceptions.html#c.PyErr_Restore" class="reference internal" title="PyErr_Restore"><span class="pre"><code class="sourceCode c">PyErr_Restore<span class="op">()</span></code></span></a> are deprecated. Use <a href="../c-api/exceptions.html#c.PyErr_GetRaisedException" class="reference internal" title="PyErr_GetRaisedException"><span class="pre"><code class="sourceCode c">PyErr_GetRaisedException<span class="op">()</span></code></span></a> and <a href="../c-api/exceptions.html#c.PyErr_SetRaisedException" class="reference internal" title="PyErr_SetRaisedException"><span class="pre"><code class="sourceCode c">PyErr_SetRaisedException<span class="op">()</span></code></span></a> instead. (Contributed by Mark Shannon in <a href="https://github.com/python/cpython/issues/101578" class="reference external">gh-101578</a>.)

- <span class="pre">`PyErr_Display()`</span> is deprecated. Use <a href="../c-api/exceptions.html#c.PyErr_DisplayException" class="reference internal" title="PyErr_DisplayException"><span class="pre"><code class="sourceCode c">PyErr_DisplayException<span class="op">()</span></code></span></a> instead. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/102755" class="reference external">gh-102755</a>).

- <span class="pre">`_PyErr_ChainExceptions`</span> is deprecated. Use <span class="pre">`_PyErr_ChainExceptions1`</span> instead. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/102192" class="reference external">gh-102192</a>.)

- Using <a href="../c-api/type.html#c.PyType_FromSpec" class="reference internal" title="PyType_FromSpec"><span class="pre"><code class="sourceCode c">PyType_FromSpec<span class="op">()</span></code></span></a>, <a href="../c-api/type.html#c.PyType_FromSpecWithBases" class="reference internal" title="PyType_FromSpecWithBases"><span class="pre"><code class="sourceCode c">PyType_FromSpecWithBases<span class="op">()</span></code></span></a> or <a href="../c-api/type.html#c.PyType_FromModuleAndSpec" class="reference internal" title="PyType_FromModuleAndSpec"><span class="pre"><code class="sourceCode c">PyType_FromModuleAndSpec<span class="op">()</span></code></span></a> to create a class whose metaclass overrides <a href="../c-api/typeobj.html#c.PyTypeObject.tp_new" class="reference internal" title="PyTypeObject.tp_new"><span class="pre"><code class="sourceCode c">tp_new</code></span></a> is deprecated. Call the metaclass instead.

<div id="id7" class="section">

#### Pending Removal in Python 3.14<a href="#id7" class="headerlink" title="Link to this heading">¶</a>

- The <span class="pre">`ma_version_tag`</span> field in <a href="../c-api/dict.html#c.PyDictObject" class="reference internal" title="PyDictObject"><span class="pre"><code class="sourceCode c">PyDictObject</code></span></a> for extension modules (<span id="index-53" class="target"></span><a href="https://peps.python.org/pep-0699/" class="pep reference external"><strong>PEP 699</strong></a>; <a href="https://github.com/python/cpython/issues/101193" class="reference external">gh-101193</a>).

- Creating <a href="../c-api/typeobj.html#c.Py_TPFLAGS_IMMUTABLETYPE" class="reference internal" title="Py_TPFLAGS_IMMUTABLETYPE"><span class="pre"><code class="sourceCode c">immutable</code></span><code class="sourceCode c"> </code><span class="pre"><code class="sourceCode c">types</code></span></a> with mutable bases (<a href="https://github.com/python/cpython/issues/95388" class="reference external">gh-95388</a>).

- Functions to configure Python’s initialization, deprecated in Python 3.11:

  - <span class="pre">`PySys_SetArgvEx()`</span>: Set <a href="../c-api/init_config.html#c.PyConfig.argv" class="reference internal" title="PyConfig.argv"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>argv</code></span></a> instead.

  - <span class="pre">`PySys_SetArgv()`</span>: Set <a href="../c-api/init_config.html#c.PyConfig.argv" class="reference internal" title="PyConfig.argv"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>argv</code></span></a> instead.

  - <span class="pre">`Py_SetProgramName()`</span>: Set <a href="../c-api/init_config.html#c.PyConfig.program_name" class="reference internal" title="PyConfig.program_name"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>program_name</code></span></a> instead.

  - <span class="pre">`Py_SetPythonHome()`</span>: Set <a href="../c-api/init_config.html#c.PyConfig.home" class="reference internal" title="PyConfig.home"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>home</code></span></a> instead.

  The <a href="../c-api/init.html#c.Py_InitializeFromConfig" class="reference internal" title="Py_InitializeFromConfig"><span class="pre"><code class="sourceCode c">Py_InitializeFromConfig<span class="op">()</span></code></span></a> API should be used with <a href="../c-api/init_config.html#c.PyConfig" class="reference internal" title="PyConfig"><span class="pre"><code class="sourceCode c">PyConfig</code></span></a> instead.

- Global configuration variables:

  - <a href="../c-api/init.html#c.Py_DebugFlag" class="reference internal" title="Py_DebugFlag"><span class="pre"><code class="sourceCode c">Py_DebugFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.parser_debug" class="reference internal" title="PyConfig.parser_debug"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>parser_debug</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_VerboseFlag" class="reference internal" title="Py_VerboseFlag"><span class="pre"><code class="sourceCode c">Py_VerboseFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.verbose" class="reference internal" title="PyConfig.verbose"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>verbose</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_QuietFlag" class="reference internal" title="Py_QuietFlag"><span class="pre"><code class="sourceCode c">Py_QuietFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.quiet" class="reference internal" title="PyConfig.quiet"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>quiet</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_InteractiveFlag" class="reference internal" title="Py_InteractiveFlag"><span class="pre"><code class="sourceCode c">Py_InteractiveFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.interactive" class="reference internal" title="PyConfig.interactive"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>interactive</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_InspectFlag" class="reference internal" title="Py_InspectFlag"><span class="pre"><code class="sourceCode c">Py_InspectFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.inspect" class="reference internal" title="PyConfig.inspect"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>inspect</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_OptimizeFlag" class="reference internal" title="Py_OptimizeFlag"><span class="pre"><code class="sourceCode c">Py_OptimizeFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.optimization_level" class="reference internal" title="PyConfig.optimization_level"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>optimization_level</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_NoSiteFlag" class="reference internal" title="Py_NoSiteFlag"><span class="pre"><code class="sourceCode c">Py_NoSiteFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.site_import" class="reference internal" title="PyConfig.site_import"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>site_import</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_BytesWarningFlag" class="reference internal" title="Py_BytesWarningFlag"><span class="pre"><code class="sourceCode c">Py_BytesWarningFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.bytes_warning" class="reference internal" title="PyConfig.bytes_warning"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>bytes_warning</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_FrozenFlag" class="reference internal" title="Py_FrozenFlag"><span class="pre"><code class="sourceCode c">Py_FrozenFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.pathconfig_warnings" class="reference internal" title="PyConfig.pathconfig_warnings"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>pathconfig_warnings</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_IgnoreEnvironmentFlag" class="reference internal" title="Py_IgnoreEnvironmentFlag"><span class="pre"><code class="sourceCode c">Py_IgnoreEnvironmentFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.use_environment" class="reference internal" title="PyConfig.use_environment"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>use_environment</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_DontWriteBytecodeFlag" class="reference internal" title="Py_DontWriteBytecodeFlag"><span class="pre"><code class="sourceCode c">Py_DontWriteBytecodeFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.write_bytecode" class="reference internal" title="PyConfig.write_bytecode"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>write_bytecode</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_NoUserSiteDirectory" class="reference internal" title="Py_NoUserSiteDirectory"><span class="pre"><code class="sourceCode c">Py_NoUserSiteDirectory</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.user_site_directory" class="reference internal" title="PyConfig.user_site_directory"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>user_site_directory</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_UnbufferedStdioFlag" class="reference internal" title="Py_UnbufferedStdioFlag"><span class="pre"><code class="sourceCode c">Py_UnbufferedStdioFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.buffered_stdio" class="reference internal" title="PyConfig.buffered_stdio"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>buffered_stdio</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_HashRandomizationFlag" class="reference internal" title="Py_HashRandomizationFlag"><span class="pre"><code class="sourceCode c">Py_HashRandomizationFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.use_hash_seed" class="reference internal" title="PyConfig.use_hash_seed"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>use_hash_seed</code></span></a> and <a href="../c-api/init_config.html#c.PyConfig.hash_seed" class="reference internal" title="PyConfig.hash_seed"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>hash_seed</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_IsolatedFlag" class="reference internal" title="Py_IsolatedFlag"><span class="pre"><code class="sourceCode c">Py_IsolatedFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.isolated" class="reference internal" title="PyConfig.isolated"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>isolated</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_LegacyWindowsFSEncodingFlag" class="reference internal" title="Py_LegacyWindowsFSEncodingFlag"><span class="pre"><code class="sourceCode c">Py_LegacyWindowsFSEncodingFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyPreConfig.legacy_windows_fs_encoding" class="reference internal" title="PyPreConfig.legacy_windows_fs_encoding"><span class="pre"><code class="sourceCode c">PyPreConfig<span class="op">.</span>legacy_windows_fs_encoding</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_LegacyWindowsStdioFlag" class="reference internal" title="Py_LegacyWindowsStdioFlag"><span class="pre"><code class="sourceCode c">Py_LegacyWindowsStdioFlag</code></span></a>: Use <a href="../c-api/init_config.html#c.PyConfig.legacy_windows_stdio" class="reference internal" title="PyConfig.legacy_windows_stdio"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>legacy_windows_stdio</code></span></a> instead.

  - <span class="pre">`Py_FileSystemDefaultEncoding`</span>: Use <a href="../c-api/init_config.html#c.PyConfig.filesystem_encoding" class="reference internal" title="PyConfig.filesystem_encoding"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>filesystem_encoding</code></span></a> instead.

  - <span class="pre">`Py_HasFileSystemDefaultEncoding`</span>: Use <a href="../c-api/init_config.html#c.PyConfig.filesystem_encoding" class="reference internal" title="PyConfig.filesystem_encoding"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>filesystem_encoding</code></span></a> instead.

  - <span class="pre">`Py_FileSystemDefaultEncodeErrors`</span>: Use <a href="../c-api/init_config.html#c.PyConfig.filesystem_errors" class="reference internal" title="PyConfig.filesystem_errors"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>filesystem_errors</code></span></a> instead.

  - <span class="pre">`Py_UTF8Mode`</span>: Use <a href="../c-api/init_config.html#c.PyPreConfig.utf8_mode" class="reference internal" title="PyPreConfig.utf8_mode"><span class="pre"><code class="sourceCode c">PyPreConfig<span class="op">.</span>utf8_mode</code></span></a> instead. (see <a href="../c-api/init_config.html#c.Py_PreInitialize" class="reference internal" title="Py_PreInitialize"><span class="pre"><code class="sourceCode c">Py_PreInitialize<span class="op">()</span></code></span></a>)

  The <a href="../c-api/init.html#c.Py_InitializeFromConfig" class="reference internal" title="Py_InitializeFromConfig"><span class="pre"><code class="sourceCode c">Py_InitializeFromConfig<span class="op">()</span></code></span></a> API should be used with <a href="../c-api/init_config.html#c.PyConfig" class="reference internal" title="PyConfig"><span class="pre"><code class="sourceCode c">PyConfig</code></span></a> instead.

</div>

<div id="id8" class="section">

#### Pending Removal in Python 3.15<a href="#id8" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../c-api/import.html#c.PyImport_ImportModuleNoBlock" class="reference internal" title="PyImport_ImportModuleNoBlock"><span class="pre"><code class="sourceCode c">PyImport_ImportModuleNoBlock<span class="op">()</span></code></span></a>: Use <a href="../c-api/import.html#c.PyImport_ImportModule" class="reference internal" title="PyImport_ImportModule"><span class="pre"><code class="sourceCode c">PyImport_ImportModule<span class="op">()</span></code></span></a> instead.

- <a href="../c-api/weakref.html#c.PyWeakref_GetObject" class="reference internal" title="PyWeakref_GetObject"><span class="pre"><code class="sourceCode c">PyWeakref_GetObject<span class="op">()</span></code></span></a> and <a href="../c-api/weakref.html#c.PyWeakref_GET_OBJECT" class="reference internal" title="PyWeakref_GET_OBJECT"><span class="pre"><code class="sourceCode c">PyWeakref_GET_OBJECT<span class="op">()</span></code></span></a>: Use <a href="../c-api/weakref.html#c.PyWeakref_GetRef" class="reference internal" title="PyWeakref_GetRef"><span class="pre"><code class="sourceCode c">PyWeakref_GetRef<span class="op">()</span></code></span></a> instead.

- <a href="../c-api/unicode.html#c.Py_UNICODE" class="reference internal" title="Py_UNICODE"><span class="pre"><code class="sourceCode c">Py_UNICODE</code></span></a> type and the <span class="pre">`Py_UNICODE_WIDE`</span> macro: Use <span class="pre">`wchar_t`</span> instead.

- Python initialization functions:

  - <a href="../c-api/sys.html#c.PySys_ResetWarnOptions" class="reference internal" title="PySys_ResetWarnOptions"><span class="pre"><code class="sourceCode c">PySys_ResetWarnOptions<span class="op">()</span></code></span></a>: Clear <a href="../library/sys.html#sys.warnoptions" class="reference internal" title="sys.warnoptions"><span class="pre"><code class="sourceCode python">sys.warnoptions</code></span></a> and <span class="pre">`warnings.filters`</span> instead.

  - <a href="../c-api/init.html#c.Py_GetExecPrefix" class="reference internal" title="Py_GetExecPrefix"><span class="pre"><code class="sourceCode c">Py_GetExecPrefix<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.base_exec_prefix" class="reference internal" title="sys.base_exec_prefix"><span class="pre"><code class="sourceCode python">sys.base_exec_prefix</code></span></a> and <a href="../library/sys.html#sys.exec_prefix" class="reference internal" title="sys.exec_prefix"><span class="pre"><code class="sourceCode python">sys.exec_prefix</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetPath" class="reference internal" title="Py_GetPath"><span class="pre"><code class="sourceCode c">Py_GetPath<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetPrefix" class="reference internal" title="Py_GetPrefix"><span class="pre"><code class="sourceCode c">Py_GetPrefix<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.base_prefix" class="reference internal" title="sys.base_prefix"><span class="pre"><code class="sourceCode python">sys.base_prefix</code></span></a> and <a href="../library/sys.html#sys.prefix" class="reference internal" title="sys.prefix"><span class="pre"><code class="sourceCode python">sys.prefix</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetProgramFullPath" class="reference internal" title="Py_GetProgramFullPath"><span class="pre"><code class="sourceCode c">Py_GetProgramFullPath<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.executable" class="reference internal" title="sys.executable"><span class="pre"><code class="sourceCode python">sys.executable</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetProgramName" class="reference internal" title="Py_GetProgramName"><span class="pre"><code class="sourceCode c">Py_GetProgramName<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.executable" class="reference internal" title="sys.executable"><span class="pre"><code class="sourceCode python">sys.executable</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetPythonHome" class="reference internal" title="Py_GetPythonHome"><span class="pre"><code class="sourceCode c">Py_GetPythonHome<span class="op">()</span></code></span></a>: Get <a href="../c-api/init_config.html#c.PyConfig.home" class="reference internal" title="PyConfig.home"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>home</code></span></a> or the <span id="index-54" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONHOME" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONHOME</code></span></a> environment variable instead.

</div>

<div id="id9" class="section">

#### Pending removal in Python 3.16<a href="#id9" class="headerlink" title="Link to this heading">¶</a>

- The bundled copy of <span class="pre">`libmpdec`</span>.

</div>

<div id="id10" class="section">

#### Pending Removal in Future Versions<a href="#id10" class="headerlink" title="Link to this heading">¶</a>

The following APIs are deprecated and will be removed, although there is currently no date scheduled for their removal.

- <a href="../c-api/typeobj.html#c.Py_TPFLAGS_HAVE_FINALIZE" class="reference internal" title="Py_TPFLAGS_HAVE_FINALIZE"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_HAVE_FINALIZE</code></span></a>: Unneeded since Python 3.8.

- <a href="../c-api/exceptions.html#c.PyErr_Fetch" class="reference internal" title="PyErr_Fetch"><span class="pre"><code class="sourceCode c">PyErr_Fetch<span class="op">()</span></code></span></a>: Use <a href="../c-api/exceptions.html#c.PyErr_GetRaisedException" class="reference internal" title="PyErr_GetRaisedException"><span class="pre"><code class="sourceCode c">PyErr_GetRaisedException<span class="op">()</span></code></span></a> instead.

- <a href="../c-api/exceptions.html#c.PyErr_NormalizeException" class="reference internal" title="PyErr_NormalizeException"><span class="pre"><code class="sourceCode c">PyErr_NormalizeException<span class="op">()</span></code></span></a>: Use <a href="../c-api/exceptions.html#c.PyErr_GetRaisedException" class="reference internal" title="PyErr_GetRaisedException"><span class="pre"><code class="sourceCode c">PyErr_GetRaisedException<span class="op">()</span></code></span></a> instead.

- <a href="../c-api/exceptions.html#c.PyErr_Restore" class="reference internal" title="PyErr_Restore"><span class="pre"><code class="sourceCode c">PyErr_Restore<span class="op">()</span></code></span></a>: Use <a href="../c-api/exceptions.html#c.PyErr_SetRaisedException" class="reference internal" title="PyErr_SetRaisedException"><span class="pre"><code class="sourceCode c">PyErr_SetRaisedException<span class="op">()</span></code></span></a> instead.

- <a href="../c-api/module.html#c.PyModule_GetFilename" class="reference internal" title="PyModule_GetFilename"><span class="pre"><code class="sourceCode c">PyModule_GetFilename<span class="op">()</span></code></span></a>: Use <a href="../c-api/module.html#c.PyModule_GetFilenameObject" class="reference internal" title="PyModule_GetFilenameObject"><span class="pre"><code class="sourceCode c">PyModule_GetFilenameObject<span class="op">()</span></code></span></a> instead.

- <a href="../c-api/sys.html#c.PyOS_AfterFork" class="reference internal" title="PyOS_AfterFork"><span class="pre"><code class="sourceCode c">PyOS_AfterFork<span class="op">()</span></code></span></a>: Use <a href="../c-api/sys.html#c.PyOS_AfterFork_Child" class="reference internal" title="PyOS_AfterFork_Child"><span class="pre"><code class="sourceCode c">PyOS_AfterFork_Child<span class="op">()</span></code></span></a> instead.

- <a href="../c-api/slice.html#c.PySlice_GetIndicesEx" class="reference internal" title="PySlice_GetIndicesEx"><span class="pre"><code class="sourceCode c">PySlice_GetIndicesEx<span class="op">()</span></code></span></a>: Use <a href="../c-api/slice.html#c.PySlice_Unpack" class="reference internal" title="PySlice_Unpack"><span class="pre"><code class="sourceCode c">PySlice_Unpack<span class="op">()</span></code></span></a> and <a href="../c-api/slice.html#c.PySlice_AdjustIndices" class="reference internal" title="PySlice_AdjustIndices"><span class="pre"><code class="sourceCode c">PySlice_AdjustIndices<span class="op">()</span></code></span></a> instead.

- <span class="pre">`PyUnicode_AsDecodedObject()`</span>: Use <a href="../c-api/codec.html#c.PyCodec_Decode" class="reference internal" title="PyCodec_Decode"><span class="pre"><code class="sourceCode c">PyCodec_Decode<span class="op">()</span></code></span></a> instead.

- <span class="pre">`PyUnicode_AsDecodedUnicode()`</span>: Use <a href="../c-api/codec.html#c.PyCodec_Decode" class="reference internal" title="PyCodec_Decode"><span class="pre"><code class="sourceCode c">PyCodec_Decode<span class="op">()</span></code></span></a> instead.

- <span class="pre">`PyUnicode_AsEncodedObject()`</span>: Use <a href="../c-api/codec.html#c.PyCodec_Encode" class="reference internal" title="PyCodec_Encode"><span class="pre"><code class="sourceCode c">PyCodec_Encode<span class="op">()</span></code></span></a> instead.

- <span class="pre">`PyUnicode_AsEncodedUnicode()`</span>: Use <a href="../c-api/codec.html#c.PyCodec_Encode" class="reference internal" title="PyCodec_Encode"><span class="pre"><code class="sourceCode c">PyCodec_Encode<span class="op">()</span></code></span></a> instead.

- <a href="../c-api/unicode.html#c.PyUnicode_READY" class="reference internal" title="PyUnicode_READY"><span class="pre"><code class="sourceCode c">PyUnicode_READY<span class="op">()</span></code></span></a>: Unneeded since Python 3.12

- <span class="pre">`PyErr_Display()`</span>: Use <a href="../c-api/exceptions.html#c.PyErr_DisplayException" class="reference internal" title="PyErr_DisplayException"><span class="pre"><code class="sourceCode c">PyErr_DisplayException<span class="op">()</span></code></span></a> instead.

- <span class="pre">`_PyErr_ChainExceptions()`</span>: Use <span class="pre">`_PyErr_ChainExceptions1()`</span> instead.

- <span class="pre">`PyBytesObject.ob_shash`</span> member: call <a href="../c-api/object.html#c.PyObject_Hash" class="reference internal" title="PyObject_Hash"><span class="pre"><code class="sourceCode c">PyObject_Hash<span class="op">()</span></code></span></a> instead.

- <span class="pre">`PyDictObject.ma_version_tag`</span> member.

- Thread Local Storage (TLS) API:

  - <a href="../c-api/init.html#c.PyThread_create_key" class="reference internal" title="PyThread_create_key"><span class="pre"><code class="sourceCode c">PyThread_create_key<span class="op">()</span></code></span></a>: Use <a href="../c-api/init.html#c.PyThread_tss_alloc" class="reference internal" title="PyThread_tss_alloc"><span class="pre"><code class="sourceCode c">PyThread_tss_alloc<span class="op">()</span></code></span></a> instead.

  - <a href="../c-api/init.html#c.PyThread_delete_key" class="reference internal" title="PyThread_delete_key"><span class="pre"><code class="sourceCode c">PyThread_delete_key<span class="op">()</span></code></span></a>: Use <a href="../c-api/init.html#c.PyThread_tss_free" class="reference internal" title="PyThread_tss_free"><span class="pre"><code class="sourceCode c">PyThread_tss_free<span class="op">()</span></code></span></a> instead.

  - <a href="../c-api/init.html#c.PyThread_set_key_value" class="reference internal" title="PyThread_set_key_value"><span class="pre"><code class="sourceCode c">PyThread_set_key_value<span class="op">()</span></code></span></a>: Use <a href="../c-api/init.html#c.PyThread_tss_set" class="reference internal" title="PyThread_tss_set"><span class="pre"><code class="sourceCode c">PyThread_tss_set<span class="op">()</span></code></span></a> instead.

  - <a href="../c-api/init.html#c.PyThread_get_key_value" class="reference internal" title="PyThread_get_key_value"><span class="pre"><code class="sourceCode c">PyThread_get_key_value<span class="op">()</span></code></span></a>: Use <a href="../c-api/init.html#c.PyThread_tss_get" class="reference internal" title="PyThread_tss_get"><span class="pre"><code class="sourceCode c">PyThread_tss_get<span class="op">()</span></code></span></a> instead.

  - <a href="../c-api/init.html#c.PyThread_delete_key_value" class="reference internal" title="PyThread_delete_key_value"><span class="pre"><code class="sourceCode c">PyThread_delete_key_value<span class="op">()</span></code></span></a>: Use <a href="../c-api/init.html#c.PyThread_tss_delete" class="reference internal" title="PyThread_tss_delete"><span class="pre"><code class="sourceCode c">PyThread_tss_delete<span class="op">()</span></code></span></a> instead.

  - <a href="../c-api/init.html#c.PyThread_ReInitTLS" class="reference internal" title="PyThread_ReInitTLS"><span class="pre"><code class="sourceCode c">PyThread_ReInitTLS<span class="op">()</span></code></span></a>: Unneeded since Python 3.7.

</div>

</div>

<div id="id11" class="section">

### Removed<a href="#id11" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`token.h`</span> header file. There was never any public tokenizer C API. The <span class="pre">`token.h`</span> header file was only designed to be used by Python internals. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/92651" class="reference external">gh-92651</a>.)

- Legacy Unicode APIs have been removed. See <span id="index-55" class="target"></span><a href="https://peps.python.org/pep-0623/" class="pep reference external"><strong>PEP 623</strong></a> for detail.

  - <span class="pre">`PyUnicode_WCHAR_KIND`</span>

  - <span class="pre">`PyUnicode_AS_UNICODE()`</span>

  - <span class="pre">`PyUnicode_AsUnicode()`</span>

  - <span class="pre">`PyUnicode_AsUnicodeAndSize()`</span>

  - <span class="pre">`PyUnicode_AS_DATA()`</span>

  - <span class="pre">`PyUnicode_FromUnicode()`</span>

  - <span class="pre">`PyUnicode_GET_SIZE()`</span>

  - <span class="pre">`PyUnicode_GetSize()`</span>

  - <span class="pre">`PyUnicode_GET_DATA_SIZE()`</span>

- Remove the <span class="pre">`PyUnicode_InternImmortal()`</span> function macro. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/85858" class="reference external">gh-85858</a>.)

</div>

</div>

</div>

<div class="clearer">

</div>

</div>
