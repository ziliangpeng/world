<div class="body" role="main">

<div id="what-s-new-in-python-3-11" class="section">

# What’s New In Python 3.11<a href="#what-s-new-in-python-3-11" class="headerlink" title="Link to this heading">¶</a>

Editor<span class="colon">:</span>  
Pablo Galindo Salgado

This article explains the new features in Python 3.11, compared to 3.10. Python 3.11 was released on October 24, 2022. For full details, see the <a href="changelog.html#changelog" class="reference internal"><span class="std std-ref">changelog</span></a>.

<div id="summary-release-highlights" class="section">

<span id="whatsnew311-summary"></span>

## Summary – Release highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

- Python 3.11 is between 10-60% faster than Python 3.10. On average, we measured a 1.25x speedup on the standard benchmark suite. See <a href="#whatsnew311-faster-cpython" class="reference internal"><span class="std std-ref">Faster CPython</span></a> for details.

New syntax features:

- <a href="#whatsnew311-pep654" class="reference internal"><span class="std std-ref">PEP 654: Exception Groups and except*</span></a>

New built-in features:

- <a href="#whatsnew311-pep678" class="reference internal"><span class="std std-ref">PEP 678: Exceptions can be enriched with notes</span></a>

New standard library modules:

- <span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0680/" class="pep reference external"><strong>PEP 680</strong></a>: <a href="../library/tomllib.html#module-tomllib" class="reference internal" title="tomllib: Parse TOML files."><span class="pre"><code class="sourceCode python">tomllib</code></span></a> — Support for parsing <a href="https://toml.io/" class="reference external">TOML</a> in the Standard Library

Interpreter improvements:

- <a href="#whatsnew311-pep657" class="reference internal"><span class="std std-ref">PEP 657: Fine-grained error locations in tracebacks</span></a>

- New <a href="../using/cmdline.html#cmdoption-P" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-P</code></span></a> command line option and <span id="index-1" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONSAFEPATH" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONSAFEPATH</code></span></a> environment variable to <a href="#whatsnew311-pythonsafepath" class="reference internal"><span class="std std-ref">disable automatically prepending potentially unsafe paths</span></a> to <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a>

New typing features:

- <a href="#whatsnew311-pep646" class="reference internal"><span class="std std-ref">PEP 646: Variadic generics</span></a>

- <a href="#whatsnew311-pep655" class="reference internal"><span class="std std-ref">PEP 655: Marking individual TypedDict items as required or not-required</span></a>

- <a href="#whatsnew311-pep673" class="reference internal"><span class="std std-ref">PEP 673: Self type</span></a>

- <a href="#whatsnew311-pep675" class="reference internal"><span class="std std-ref">PEP 675: Arbitrary literal string type</span></a>

- <a href="#whatsnew311-pep681" class="reference internal"><span class="std std-ref">PEP 681: Data class transforms</span></a>

Important deprecations, removals and restrictions:

- <span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0594/" class="pep reference external"><strong>PEP 594</strong></a>: <a href="#whatsnew311-pep594" class="reference internal"><span class="std std-ref">Many legacy standard library modules have been deprecated</span></a> and will be removed in Python 3.13

- <span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0624/" class="pep reference external"><strong>PEP 624</strong></a>: <a href="#whatsnew311-pep624" class="reference internal"><span class="std std-ref">Py_UNICODE encoder APIs have been removed</span></a>

- <span id="index-4" class="target"></span><a href="https://peps.python.org/pep-0670/" class="pep reference external"><strong>PEP 670</strong></a>: <a href="#whatsnew311-pep670" class="reference internal"><span class="std std-ref">Macros converted to static inline functions</span></a>

</div>

<div id="new-features" class="section">

<span id="whatsnew311-features"></span>

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-657-fine-grained-error-locations-in-tracebacks" class="section">

<span id="whatsnew311-pep657"></span>

### PEP 657: Fine-grained error locations in tracebacks<a href="#pep-657-fine-grained-error-locations-in-tracebacks" class="headerlink" title="Link to this heading">¶</a>

When printing tracebacks, the interpreter will now point to the exact expression that caused the error, instead of just the line. For example:

<div class="highlight-python notranslate">

<div class="highlight">

    Traceback (most recent call last):
      File "distance.py", line 11, in <module>
        print(manhattan_distance(p1, p2))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "distance.py", line 6, in manhattan_distance
        return abs(point_1.x - point_2.x) + abs(point_1.y - point_2.y)
                               ^^^^^^^^^
    AttributeError: 'NoneType' object has no attribute 'x'

</div>

</div>

Previous versions of the interpreter would point to just the line, making it ambiguous which object was <span class="pre">`None`</span>. These enhanced errors can also be helpful when dealing with deeply nested <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> objects and multiple function calls:

<div class="highlight-python notranslate">

<div class="highlight">

    Traceback (most recent call last):
      File "query.py", line 37, in <module>
        magic_arithmetic('foo')
      File "query.py", line 18, in magic_arithmetic
        return add_counts(x) / 25
               ^^^^^^^^^^^^^
      File "query.py", line 24, in add_counts
        return 25 + query_user(user1) + query_user(user2)
                    ^^^^^^^^^^^^^^^^^
      File "query.py", line 32, in query_user
        return 1 + query_count(db, response['a']['b']['c']['user'], retry=True)
                                   ~~~~~~~~~~~~~~~~~~^^^^^
    TypeError: 'NoneType' object is not subscriptable

</div>

</div>

As well as complex arithmetic expressions:

<div class="highlight-python notranslate">

<div class="highlight">

    Traceback (most recent call last):
      File "calculation.py", line 54, in <module>
        result = (x / y / z) * (a / b / c)
                  ~~~~~~^~~
    ZeroDivisionError: division by zero

</div>

</div>

Additionally, the information used by the enhanced traceback feature is made available via a general API, that can be used to correlate <a href="../glossary.html#term-bytecode" class="reference internal"><span class="xref std std-term">bytecode</span></a> <a href="../library/dis.html#bytecodes" class="reference internal"><span class="std std-ref">instructions</span></a> with source code location. This information can be retrieved using:

- The <a href="../reference/datamodel.html#codeobject.co_positions" class="reference internal" title="codeobject.co_positions"><span class="pre"><code class="sourceCode python">codeobject.co_positions()</code></span></a> method in Python.

- The <a href="../c-api/code.html#c.PyCode_Addr2Location" class="reference internal" title="PyCode_Addr2Location"><span class="pre"><code class="sourceCode c">PyCode_Addr2Location<span class="op">()</span></code></span></a> function in the C API.

See <span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0657/" class="pep reference external"><strong>PEP 657</strong></a> for more details. (Contributed by Pablo Galindo, Batuhan Taskaya and Ammar Askar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43950" class="reference external">bpo-43950</a>.)

<div class="admonition note">

Note

This feature requires storing column positions in <a href="../c-api/code.html#codeobjects" class="reference internal"><span class="std std-ref">Code Objects</span></a>, which may result in a small increase in interpreter memory usage and disk usage for compiled Python files. To avoid storing the extra information and deactivate printing the extra traceback information, use the <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">no_debug_ranges</code></span></a> command line option or the <span id="index-6" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONNODEBUGRANGES" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONNODEBUGRANGES</code></span></a> environment variable.

</div>

</div>

<div id="pep-654-exception-groups-and-except" class="section">

<span id="whatsnew311-pep654"></span>

### PEP 654: Exception Groups and <span class="pre">`except*`</span><a href="#pep-654-exception-groups-and-except" class="headerlink" title="Link to this heading">¶</a>

<span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0654/" class="pep reference external"><strong>PEP 654</strong></a> introduces language features that enable a program to raise and handle multiple unrelated exceptions simultaneously. The builtin types <a href="../library/exceptions.html#ExceptionGroup" class="reference internal" title="ExceptionGroup"><span class="pre"><code class="sourceCode python">ExceptionGroup</code></span></a> and <a href="../library/exceptions.html#BaseExceptionGroup" class="reference internal" title="BaseExceptionGroup"><span class="pre"><code class="sourceCode python">BaseExceptionGroup</code></span></a> make it possible to group exceptions and raise them together, and the new <a href="../reference/compound_stmts.html#except-star" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except*</code></span></a> syntax generalizes <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> to match subgroups of exception groups.

See <span id="index-8" class="target"></span><a href="https://peps.python.org/pep-0654/" class="pep reference external"><strong>PEP 654</strong></a> for more details.

(Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45292" class="reference external">bpo-45292</a>. PEP written by Irit Katriel, Yury Selivanov and Guido van Rossum.)

</div>

<div id="pep-678-exceptions-can-be-enriched-with-notes" class="section">

<span id="whatsnew311-pep678"></span>

### PEP 678: Exceptions can be enriched with notes<a href="#pep-678-exceptions-can-be-enriched-with-notes" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/exceptions.html#BaseException.add_note" class="reference internal" title="BaseException.add_note"><span class="pre"><code class="sourceCode python">add_note()</code></span></a> method is added to <a href="../library/exceptions.html#BaseException" class="reference internal" title="BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a>. It can be used to enrich exceptions with context information that is not available at the time when the exception is raised. The added notes appear in the default traceback.

See <span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0678/" class="pep reference external"><strong>PEP 678</strong></a> for more details.

(Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45607" class="reference external">bpo-45607</a>. PEP written by Zac Hatfield-Dodds.)

</div>

<div id="windows-py-exe-launcher-improvements" class="section">

<span id="whatsnew311-windows-launcher"></span>

### Windows <span class="pre">`py.exe`</span> launcher improvements<a href="#windows-py-exe-launcher-improvements" class="headerlink" title="Link to this heading">¶</a>

The copy of the <a href="../using/windows.html#launcher" class="reference internal"><span class="std std-ref">Python Launcher for Windows</span></a> included with Python 3.11 has been significantly updated. It now supports company/tag syntax as defined in <span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0514/" class="pep reference external"><strong>PEP 514</strong></a> using the <span class="pre">`-V:`</span>*<span class="pre">`<company>`</span>*<span class="pre">`/`</span>*<span class="pre">`<tag>`</span>* argument instead of the limited <span class="pre">`-`</span>*<span class="pre">`<major>`</span>*<span class="pre">`.`</span>*<span class="pre">`<minor>`</span>*. This allows launching distributions other than <span class="pre">`PythonCore`</span>, the one hosted on <a href="https://www.python.org" class="reference external">python.org</a>.

When using <span class="pre">`-V:`</span> selectors, either company or tag can be omitted, but all installs will be searched. For example, <span class="pre">`-V:OtherPython/`</span> will select the “best” tag registered for <span class="pre">`OtherPython`</span>, while <span class="pre">`-V:3.11`</span> or <span class="pre">`-V:/3.11`</span> will select the “best” distribution with tag <span class="pre">`3.11`</span>.

When using the legacy <span class="pre">`-`</span>*<span class="pre">`<major>`</span>*, <span class="pre">`-`</span>*<span class="pre">`<major>`</span>*<span class="pre">`.`</span>*<span class="pre">`<minor>`</span>*, <span class="pre">`-`</span>*<span class="pre">`<major>`</span>*<span class="pre">`-`</span>*<span class="pre">`<bitness>`</span>* or <span class="pre">`-`</span>*<span class="pre">`<major>`</span>*<span class="pre">`.`</span>*<span class="pre">`<minor>`</span>*<span class="pre">`-`</span>*<span class="pre">`<bitness>`</span>* arguments, all existing behaviour should be preserved from past versions, and only releases from <span class="pre">`PythonCore`</span> will be selected. However, the <span class="pre">`-64`</span> suffix now implies “not 32-bit” (not necessarily x86-64), as there are multiple supported 64-bit platforms. 32-bit runtimes are detected by checking the runtime’s tag for a <span class="pre">`-32`</span> suffix. All releases of Python since 3.5 have included this in their 32-bit builds.

</div>

</div>

<div id="new-features-related-to-type-hints" class="section">

<span id="whatsnew311-typing-features"></span><span id="new-feat-related-type-hints-311"></span>

## New Features Related to Type Hints<a href="#new-features-related-to-type-hints" class="headerlink" title="Link to this heading">¶</a>

This section covers major changes affecting <span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> type hints and the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module.

<div id="pep-646-variadic-generics" class="section">

<span id="whatsnew311-pep646"></span>

### PEP 646: Variadic generics<a href="#pep-646-variadic-generics" class="headerlink" title="Link to this heading">¶</a>

<span id="index-12" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> previously introduced <a href="../library/typing.html#typing.TypeVar" class="reference internal" title="typing.TypeVar"><span class="pre"><code class="sourceCode python">TypeVar</code></span></a>, enabling creation of generics parameterised with a single type. <span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0646/" class="pep reference external"><strong>PEP 646</strong></a> adds <a href="../library/typing.html#typing.TypeVarTuple" class="reference internal" title="typing.TypeVarTuple"><span class="pre"><code class="sourceCode python">TypeVarTuple</code></span></a>, enabling parameterisation with an *arbitrary* number of types. In other words, a <a href="../library/typing.html#typing.TypeVarTuple" class="reference internal" title="typing.TypeVarTuple"><span class="pre"><code class="sourceCode python">TypeVarTuple</code></span></a> is a *variadic* type variable, enabling *variadic* generics.

This enables a wide variety of use cases. In particular, it allows the type of array-like structures in numerical computing libraries such as NumPy and TensorFlow to be parameterised with the array *shape*. Static type checkers will now be able to catch shape-related bugs in code that uses these libraries.

See <span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0646/" class="pep reference external"><strong>PEP 646</strong></a> for more details.

(Contributed by Matthew Rahtz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43224" class="reference external">bpo-43224</a>, with contributions by Serhiy Storchaka and Jelle Zijlstra. PEP written by Mark Mendoza, Matthew Rahtz, Pradeep Kumar Srinivasan, and Vincent Siles.)

</div>

<div id="pep-655-marking-individual-typeddict-items-as-required-or-not-required" class="section">

<span id="whatsnew311-pep655"></span>

### PEP 655: Marking individual <span class="pre">`TypedDict`</span> items as required or not-required<a href="#pep-655-marking-individual-typeddict-items-as-required-or-not-required" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/typing.html#typing.Required" class="reference internal" title="typing.Required"><span class="pre"><code class="sourceCode python">Required</code></span></a> and <a href="../library/typing.html#typing.NotRequired" class="reference internal" title="typing.NotRequired"><span class="pre"><code class="sourceCode python">NotRequired</code></span></a> provide a straightforward way to mark whether individual items in a <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">TypedDict</code></span></a> must be present. Previously, this was only possible using inheritance.

All fields are still required by default, unless the *total* parameter is set to <span class="pre">`False`</span>, in which case all fields are still not-required by default. For example, the following specifies a <span class="pre">`TypedDict`</span> with one required and one not-required key:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class Movie(TypedDict):
       title: str
       year: NotRequired[int]

    m1: Movie = {"title": "Black Panther", "year": 2018}  # OK
    m2: Movie = {"title": "Star Wars"}  # OK (year is not required)
    m3: Movie = {"year": 2022}  # ERROR (missing required field title)

</div>

</div>

The following definition is equivalent:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class Movie(TypedDict, total=False):
       title: Required[str]
       year: int

</div>

</div>

See <span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0655/" class="pep reference external"><strong>PEP 655</strong></a> for more details.

(Contributed by David Foster and Jelle Zijlstra in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=47087" class="reference external">bpo-47087</a>. PEP written by David Foster.)

</div>

<div id="pep-673-self-type" class="section">

<span id="whatsnew311-pep673"></span>

### PEP 673: <span class="pre">`Self`</span> type<a href="#pep-673-self-type" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/typing.html#typing.Self" class="reference internal" title="typing.Self"><span class="pre"><code class="sourceCode python">Self</code></span></a> annotation provides a simple and intuitive way to annotate methods that return an instance of their class. This behaves the same as the <a href="../library/typing.html#typing.TypeVar" class="reference internal" title="typing.TypeVar"><span class="pre"><code class="sourceCode python">TypeVar</code></span></a>-based approach <span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0484/#annotating-instance-and-class-methods" class="pep reference external"><strong>specified in PEP 484</strong></a>, but is more concise and easier to follow.

Common use cases include alternative constructors provided as <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span></code></span></a>s, and <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> methods that return <span class="pre">`self`</span>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class MyLock:
        def __enter__(self) -> Self:
            self.lock()
            return self

        ...

    class MyInt:
        @classmethod
        def fromhex(cls, s: str) -> Self:
            return cls(int(s, 16))

        ...

</div>

</div>

<a href="../library/typing.html#typing.Self" class="reference internal" title="typing.Self"><span class="pre"><code class="sourceCode python">Self</code></span></a> can also be used to annotate method parameters or attributes of the same type as their enclosing class.

See <span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0673/" class="pep reference external"><strong>PEP 673</strong></a> for more details.

(Contributed by James Hilton-Balfe in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46534" class="reference external">bpo-46534</a>. PEP written by Pradeep Kumar Srinivasan and James Hilton-Balfe.)

</div>

<div id="pep-675-arbitrary-literal-string-type" class="section">

<span id="whatsnew311-pep675"></span>

### PEP 675: Arbitrary literal string type<a href="#pep-675-arbitrary-literal-string-type" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/typing.html#typing.LiteralString" class="reference internal" title="typing.LiteralString"><span class="pre"><code class="sourceCode python">LiteralString</code></span></a> annotation may be used to indicate that a function parameter can be of any literal string type. This allows a function to accept arbitrary literal string types, as well as strings created from other literal strings. Type checkers can then enforce that sensitive functions, such as those that execute SQL statements or shell commands, are called only with static arguments, providing protection against injection attacks.

For example, a SQL query function could be annotated as follows:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def run_query(sql: LiteralString) -> ...
        ...

    def caller(
        arbitrary_string: str,
        query_string: LiteralString,
        table_name: LiteralString,
    ) -> None:
        run_query("SELECT * FROM students")       # ok
        run_query(query_string)                   # ok
        run_query("SELECT * FROM " + table_name)  # ok
        run_query(arbitrary_string)               # type checker error
        run_query(                                # type checker error
            f"SELECT * FROM students WHERE name = {arbitrary_string}"
        )

</div>

</div>

See <span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0675/" class="pep reference external"><strong>PEP 675</strong></a> for more details.

(Contributed by Jelle Zijlstra in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=47088" class="reference external">bpo-47088</a>. PEP written by Pradeep Kumar Srinivasan and Graham Bleaney.)

</div>

<div id="pep-681-data-class-transforms" class="section">

<span id="whatsnew311-pep681"></span>

### PEP 681: Data class transforms<a href="#pep-681-data-class-transforms" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/typing.html#typing.dataclass_transform" class="reference internal" title="typing.dataclass_transform"><span class="pre"><code class="sourceCode python">dataclass_transform</code></span></a> may be used to decorate a class, metaclass, or a function that is itself a decorator. The presence of <span class="pre">`@dataclass_transform()`</span> tells a static type checker that the decorated object performs runtime “magic” that transforms a class, giving it <a href="../library/dataclasses.html#dataclasses.dataclass" class="reference internal" title="dataclasses.dataclass"><span class="pre"><code class="sourceCode python">dataclass</code></span></a>-like behaviors.

For example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    # The create_model decorator is defined by a library.
    @typing.dataclass_transform()
    def create_model(cls: Type[T]) -> Type[T]:
        cls.__init__ = ...
        cls.__eq__ = ...
        cls.__ne__ = ...
        return cls

    # The create_model decorator can now be used to create new model classes:
    @create_model
    class CustomerModel:
        id: int
        name: str

    c = CustomerModel(id=327, name="Eric Idle")

</div>

</div>

See <span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0681/" class="pep reference external"><strong>PEP 681</strong></a> for more details.

(Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/91860" class="reference external">gh-91860</a>. PEP written by Erik De Bonte and Eric Traut.)

</div>

<div id="pep-563-may-not-be-the-future" class="section">

<span id="whatsnew311-pep563-deferred"></span>

### PEP 563 may not be the future<a href="#pep-563-may-not-be-the-future" class="headerlink" title="Link to this heading">¶</a>

<span id="index-20" class="target"></span><a href="https://peps.python.org/pep-0563/" class="pep reference external"><strong>PEP 563</strong></a> Postponed Evaluation of Annotations (the <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`annotations`</span> <a href="../reference/simple_stmts.html#future" class="reference internal"><span class="std std-ref">future statement</span></a>) that was originally planned for release in Python 3.10 has been put on hold indefinitely. See <a href="https://mail.python.org/archives/list/python-dev@python.org/message/VIZEBX5EYMSYIJNDBF6DMUMZOCWHARSO/" class="reference external">this message from the Steering Council</a> for more information.

</div>

</div>

<div id="other-language-changes" class="section">

<span id="whatsnew311-other-lang-changes"></span>

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

- Starred unpacking expressions can now be used in <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> statements. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46725" class="reference external">bpo-46725</a> for more details.)

- Asynchronous <a href="../reference/expressions.html#comprehensions" class="reference internal"><span class="std std-ref">comprehensions</span></a> are now allowed inside comprehensions in <a href="../reference/compound_stmts.html#async-def" class="reference internal"><span class="std std-ref">asynchronous functions</span></a>. Outer comprehensions implicitly become asynchronous in this case. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33346" class="reference external">bpo-33346</a>.)

- A <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> is now raised instead of an <a href="../library/exceptions.html#AttributeError" class="reference internal" title="AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> in <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statements and <a href="../library/contextlib.html#contextlib.ExitStack.enter_context" class="reference internal" title="contextlib.ExitStack.enter_context"><span class="pre"><code class="sourceCode python">contextlib.ExitStack.enter_context()</code></span></a> for objects that do not support the <a href="../glossary.html#term-context-manager" class="reference internal"><span class="xref std std-term">context manager</span></a> protocol, and in <a href="../reference/compound_stmts.html#async-with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statements and <a href="../library/contextlib.html#contextlib.AsyncExitStack.enter_async_context" class="reference internal" title="contextlib.AsyncExitStack.enter_async_context"><span class="pre"><code class="sourceCode python">contextlib.AsyncExitStack.enter_async_context()</code></span></a> for objects not supporting the <a href="../glossary.html#term-asynchronous-context-manager" class="reference internal"><span class="xref std std-term">asynchronous context manager</span></a> protocol. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12022" class="reference external">bpo-12022</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44471" class="reference external">bpo-44471</a>.)

- Added <a href="../library/pickle.html#object.__getstate__" class="reference internal" title="object.__getstate__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.__getstate__()</code></span></a>, which provides the default implementation of the <span class="pre">`__getstate__()`</span> method. <a href="../library/copy.html#module-copy" class="reference internal" title="copy: Shallow and deep copy operations."><span class="pre"><code class="sourceCode python">copy</code></span></a>ing and <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a>ing instances of subclasses of builtin types <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>, <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a>, <a href="../library/stdtypes.html#frozenset" class="reference internal" title="frozenset"><span class="pre"><code class="sourceCode python"><span class="bu">frozenset</span></code></span></a>, <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">collections.OrderedDict</code></span></a>, <a href="../library/collections.html#collections.deque" class="reference internal" title="collections.deque"><span class="pre"><code class="sourceCode python">collections.deque</code></span></a>, <a href="../library/weakref.html#weakref.WeakSet" class="reference internal" title="weakref.WeakSet"><span class="pre"><code class="sourceCode python">weakref.WeakSet</code></span></a>, and <a href="../library/datetime.html#datetime.tzinfo" class="reference internal" title="datetime.tzinfo"><span class="pre"><code class="sourceCode python">datetime.tzinfo</code></span></a> now copies and pickles instance attributes implemented as <a href="../glossary.html#term-__slots__" class="reference internal"><span class="xref std std-term">slots</span></a>. This change has an unintended side effect: It trips up a small minority of existing Python projects not expecting <a href="../library/pickle.html#object.__getstate__" class="reference internal" title="object.__getstate__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.__getstate__()</code></span></a> to exist. See the later comments on <a href="https://github.com/python/cpython/issues/70766" class="reference external">gh-70766</a> for discussions of what workarounds such code may need. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26579" class="reference external">bpo-26579</a>.)

<!-- -->

- Added a <a href="../using/cmdline.html#cmdoption-P" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-P</code></span></a> command line option and a <span id="index-21" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONSAFEPATH" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONSAFEPATH</code></span></a> environment variable, which disable the automatic prepending to <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> of the script’s directory when running a script, or the current directory when using <a href="../using/cmdline.html#cmdoption-c" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-c</code></span></a> and <a href="../using/cmdline.html#cmdoption-m" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-m</code></span></a>. This ensures only stdlib and installed modules are picked up by <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a>, and avoids unintentionally or maliciously shadowing modules with those in a local (and typically user-writable) directory. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/57684" class="reference external">gh-57684</a>.)

- A <span class="pre">`"z"`</span> option was added to the <a href="../library/string.html#formatspec" class="reference internal"><span class="std std-ref">Format Specification Mini-Language</span></a> that coerces negative to positive zero after rounding to the format precision. See <span id="index-22" class="target"></span><a href="https://peps.python.org/pep-0682/" class="pep reference external"><strong>PEP 682</strong></a> for more details. (Contributed by John Belmonte in <a href="https://github.com/python/cpython/issues/90153" class="reference external">gh-90153</a>.)

- Bytes are no longer accepted on <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a>. Support broke sometime between Python 3.2 and 3.6, with no one noticing until after Python 3.10.0 was released. In addition, bringing back support would be problematic due to interactions between <a href="../using/cmdline.html#cmdoption-b" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-b</code></span></a> and <a href="../library/sys.html#sys.path_importer_cache" class="reference internal" title="sys.path_importer_cache"><span class="pre"><code class="sourceCode python">sys.path_importer_cache</code></span></a> when there is a mixture of <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> and <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> keys. (Contributed by Thomas Grainger in <a href="https://github.com/python/cpython/issues/91181" class="reference external">gh-91181</a>.)

</div>

<div id="other-cpython-implementation-changes" class="section">

<span id="whatsnew311-other-implementation-changes"></span>

## Other CPython Implementation Changes<a href="#other-cpython-implementation-changes" class="headerlink" title="Link to this heading">¶</a>

- The special methods <a href="../reference/datamodel.html#object.__complex__" class="reference internal" title="object.__complex__"><span class="pre"><code class="sourceCode python"><span class="fu">__complex__</span>()</code></span></a> for <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a> and <a href="../reference/datamodel.html#object.__bytes__" class="reference internal" title="object.__bytes__"><span class="pre"><code class="sourceCode python"><span class="fu">__bytes__</span>()</code></span></a> for <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> are implemented to support the <a href="../library/typing.html#typing.SupportsComplex" class="reference internal" title="typing.SupportsComplex"><span class="pre"><code class="sourceCode python">typing.SupportsComplex</code></span></a> and <a href="../library/typing.html#typing.SupportsBytes" class="reference internal" title="typing.SupportsBytes"><span class="pre"><code class="sourceCode python">typing.SupportsBytes</code></span></a> protocols. (Contributed by Mark Dickinson and Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24234" class="reference external">bpo-24234</a>.)

- <span class="pre">`siphash13`</span> is added as a new internal hashing algorithm. It has similar security properties as <span class="pre">`siphash24`</span>, but it is slightly faster for long inputs. <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>, and some other types now use it as the default algorithm for <a href="../library/functions.html#hash" class="reference internal" title="hash"><span class="pre"><code class="sourceCode python"><span class="bu">hash</span>()</code></span></a>. <span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0552/" class="pep reference external"><strong>PEP 552</strong></a> <a href="../reference/import.html#pyc-invalidation" class="reference internal"><span class="std std-ref">hash-based .pyc files</span></a> now use <span class="pre">`siphash13`</span> too. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29410" class="reference external">bpo-29410</a>.)

- When an active exception is re-raised by a <a href="../reference/simple_stmts.html#raise" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">raise</code></span></a> statement with no parameters, the traceback attached to this exception is now always <span class="pre">`sys.exc_info()[1].__traceback__`</span>. This means that changes made to the traceback in the current <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> clause are reflected in the re-raised exception. (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45711" class="reference external">bpo-45711</a>.)

- The interpreter state’s representation of handled exceptions (aka <span class="pre">`exc_info`</span> or <span class="pre">`_PyErr_StackItem`</span>) now only has the <span class="pre">`exc_value`</span> field; <span class="pre">`exc_type`</span> and <span class="pre">`exc_traceback`</span> have been removed, as they can be derived from <span class="pre">`exc_value`</span>. (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45711" class="reference external">bpo-45711</a>.)

- A new <a href="../using/windows.html#install-quiet-option" class="reference internal"><span class="std std-ref">command line option</span></a>, <span class="pre">`AppendPath`</span>, has been added for the Windows installer. It behaves similarly to <span class="pre">`PrependPath`</span>, but appends the install and scripts directories instead of prepending them. (Contributed by Bastian Neuburger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44934" class="reference external">bpo-44934</a>.)

- The <a href="../c-api/init_config.html#c.PyConfig.module_search_paths_set" class="reference internal" title="PyConfig.module_search_paths_set"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>module_search_paths_set</code></span></a> field must now be set to <span class="pre">`1`</span> for initialization to use <a href="../c-api/init_config.html#c.PyConfig.module_search_paths" class="reference internal" title="PyConfig.module_search_paths"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>module_search_paths</code></span></a> to initialize <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a>. Otherwise, initialization will recalculate the path and replace any values added to <span class="pre">`module_search_paths`</span>.

- The output of the <a href="../using/cmdline.html#cmdoption-help" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--help</code></span></a> option now fits in 50 lines/80 columns. Information about <a href="../using/cmdline.html#using-on-envvars" class="reference internal"><span class="std std-ref">Python environment variables</span></a> and <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span></a> options is now available using the respective <a href="../using/cmdline.html#cmdoption-help-env" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--help-env</code></span></a> and <a href="../using/cmdline.html#cmdoption-help-xoptions" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--help-xoptions</code></span></a> flags, and with the new <a href="../using/cmdline.html#cmdoption-help-all" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--help-all</code></span></a>. (Contributed by Éric Araujo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46142" class="reference external">bpo-46142</a>.)

- Converting between <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> and <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> in bases other than 2 (binary), 4, 8 (octal), 16 (hexadecimal), or 32 such as base 10 (decimal) now raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the number of digits in string form is above a limit to avoid potential denial of service attacks due to the algorithmic complexity. This is a mitigation for <span id="index-24" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2020-10735" class="cve reference external"><strong>CVE 2020-10735</strong></a>. This limit can be configured or disabled by environment variable, command line flag, or <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> APIs. See the <a href="../library/stdtypes.html#int-max-str-digits" class="reference internal"><span class="std std-ref">integer string conversion length limitation</span></a> documentation. The default limit is 4300 digits in string form.

</div>

<div id="new-modules" class="section">

<span id="whatsnew311-new-modules"></span>

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/tomllib.html#module-tomllib" class="reference internal" title="tomllib: Parse TOML files."><span class="pre"><code class="sourceCode python">tomllib</code></span></a>: For parsing <a href="https://toml.io/" class="reference external">TOML</a>. See <span id="index-25" class="target"></span><a href="https://peps.python.org/pep-0680/" class="pep reference external"><strong>PEP 680</strong></a> for more details. (Contributed by Taneli Hukkinen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40059" class="reference external">bpo-40059</a>.)

- <a href="../library/wsgiref.html#module-wsgiref.types" class="reference internal" title="wsgiref.types: WSGI types for static type checking"><span class="pre"><code class="sourceCode python">wsgiref.types</code></span></a>: <span id="index-26" class="target"></span><a href="https://peps.python.org/pep-3333/" class="pep reference external"><strong>WSGI</strong></a>-specific types for static type checking. (Contributed by Sebastian Rittau in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42012" class="reference external">bpo-42012</a>.)

</div>

<div id="improved-modules" class="section">

<span id="whatsnew311-improved-modules"></span>

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="asyncio" class="section">

<span id="whatsnew311-asyncio"></span>

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

- Added the <a href="../library/asyncio-task.html#asyncio.TaskGroup" class="reference internal" title="asyncio.TaskGroup"><span class="pre"><code class="sourceCode python">TaskGroup</code></span></a> class, an <a href="../reference/datamodel.html#async-context-managers" class="reference internal"><span class="std std-ref">asynchronous context manager</span></a> holding a group of tasks that will wait for all of them upon exit. For new code this is recommended over using <a href="../library/asyncio-task.html#asyncio.create_task" class="reference internal" title="asyncio.create_task"><span class="pre"><code class="sourceCode python">create_task()</code></span></a> and <a href="../library/asyncio-task.html#asyncio.gather" class="reference internal" title="asyncio.gather"><span class="pre"><code class="sourceCode python">gather()</code></span></a> directly. (Contributed by Yury Selivanov and others in <a href="https://github.com/python/cpython/issues/90908" class="reference external">gh-90908</a>.)

- Added <a href="../library/asyncio-task.html#asyncio.timeout" class="reference internal" title="asyncio.timeout"><span class="pre"><code class="sourceCode python">timeout()</code></span></a>, an asynchronous context manager for setting a timeout on asynchronous operations. For new code this is recommended over using <a href="../library/asyncio-task.html#asyncio.wait_for" class="reference internal" title="asyncio.wait_for"><span class="pre"><code class="sourceCode python">wait_for()</code></span></a> directly. (Contributed by Andrew Svetlov in <a href="https://github.com/python/cpython/issues/90927" class="reference external">gh-90927</a>.)

- Added the <a href="../library/asyncio-runner.html#asyncio.Runner" class="reference internal" title="asyncio.Runner"><span class="pre"><code class="sourceCode python">Runner</code></span></a> class, which exposes the machinery used by <a href="../library/asyncio-runner.html#asyncio.run" class="reference internal" title="asyncio.run"><span class="pre"><code class="sourceCode python">run()</code></span></a>. (Contributed by Andrew Svetlov in <a href="https://github.com/python/cpython/issues/91218" class="reference external">gh-91218</a>.)

- Added the <a href="../library/asyncio-sync.html#asyncio.Barrier" class="reference internal" title="asyncio.Barrier"><span class="pre"><code class="sourceCode python">Barrier</code></span></a> class to the synchronization primitives in the asyncio library, and the related <a href="../library/asyncio-sync.html#asyncio.BrokenBarrierError" class="reference internal" title="asyncio.BrokenBarrierError"><span class="pre"><code class="sourceCode python">BrokenBarrierError</code></span></a> exception. (Contributed by Yves Duprat and Andrew Svetlov in <a href="https://github.com/python/cpython/issues/87518" class="reference external">gh-87518</a>.)

- Added keyword argument *all_errors* to <a href="../library/asyncio-eventloop.html#asyncio.loop.create_connection" class="reference internal" title="asyncio.loop.create_connection"><span class="pre"><code class="sourceCode python">asyncio.loop.create_connection()</code></span></a> so that multiple connection errors can be raised as an <a href="../library/exceptions.html#ExceptionGroup" class="reference internal" title="ExceptionGroup"><span class="pre"><code class="sourceCode python">ExceptionGroup</code></span></a>.

- Added the <a href="../library/asyncio-stream.html#asyncio.StreamWriter.start_tls" class="reference internal" title="asyncio.StreamWriter.start_tls"><span class="pre"><code class="sourceCode python">asyncio.StreamWriter.start_tls()</code></span></a> method for upgrading existing stream-based connections to TLS. (Contributed by Ian Good in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34975" class="reference external">bpo-34975</a>.)

- Added raw datagram socket functions to the event loop: <a href="../library/asyncio-eventloop.html#asyncio.loop.sock_sendto" class="reference internal" title="asyncio.loop.sock_sendto"><span class="pre"><code class="sourceCode python">sock_sendto()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.sock_recvfrom" class="reference internal" title="asyncio.loop.sock_recvfrom"><span class="pre"><code class="sourceCode python">sock_recvfrom()</code></span></a> and <a href="../library/asyncio-eventloop.html#asyncio.loop.sock_recvfrom_into" class="reference internal" title="asyncio.loop.sock_recvfrom_into"><span class="pre"><code class="sourceCode python">sock_recvfrom_into()</code></span></a>. These have implementations in <a href="../library/asyncio-eventloop.html#asyncio.SelectorEventLoop" class="reference internal" title="asyncio.SelectorEventLoop"><span class="pre"><code class="sourceCode python">SelectorEventLoop</code></span></a> and <a href="../library/asyncio-eventloop.html#asyncio.ProactorEventLoop" class="reference internal" title="asyncio.ProactorEventLoop"><span class="pre"><code class="sourceCode python">ProactorEventLoop</code></span></a>. (Contributed by Alex Grönholm in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46805" class="reference external">bpo-46805</a>.)

- Added <a href="../library/asyncio-task.html#asyncio.Task.cancelling" class="reference internal" title="asyncio.Task.cancelling"><span class="pre"><code class="sourceCode python">cancelling()</code></span></a> and <a href="../library/asyncio-task.html#asyncio.Task.uncancel" class="reference internal" title="asyncio.Task.uncancel"><span class="pre"><code class="sourceCode python">uncancel()</code></span></a> methods to <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">Task</code></span></a>. These are primarily intended for internal use, notably by <a href="../library/asyncio-task.html#asyncio.TaskGroup" class="reference internal" title="asyncio.TaskGroup"><span class="pre"><code class="sourceCode python">TaskGroup</code></span></a>.

</div>

<div id="contextlib" class="section">

<span id="whatsnew311-contextlib"></span>

### contextlib<a href="#contextlib" class="headerlink" title="Link to this heading">¶</a>

- Added non parallel-safe <a href="../library/contextlib.html#contextlib.chdir" class="reference internal" title="contextlib.chdir"><span class="pre"><code class="sourceCode python">chdir()</code></span></a> context manager to change the current working directory and then restore it on exit. Simple wrapper around <a href="../library/os.html#os.chdir" class="reference internal" title="os.chdir"><span class="pre"><code class="sourceCode python">chdir()</code></span></a>. (Contributed by Filipe Laíns in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25625" class="reference external">bpo-25625</a>)

</div>

<div id="dataclasses" class="section">

<span id="whatsnew311-dataclasses"></span>

### dataclasses<a href="#dataclasses" class="headerlink" title="Link to this heading">¶</a>

- Change field default mutability check, allowing only defaults which are <a href="../glossary.html#term-hashable" class="reference internal"><span class="xref std std-term">hashable</span></a> instead of any object which is not an instance of <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a>, <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a> or <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a>. (Contributed by Eric V. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44674" class="reference external">bpo-44674</a>.)

</div>

<div id="datetime" class="section">

<span id="whatsnew311-datetime"></span>

### datetime<a href="#datetime" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/datetime.html#datetime.UTC" class="reference internal" title="datetime.UTC"><span class="pre"><code class="sourceCode python">datetime.UTC</code></span></a>, a convenience alias for <a href="../library/datetime.html#datetime.timezone.utc" class="reference internal" title="datetime.timezone.utc"><span class="pre"><code class="sourceCode python">datetime.timezone.utc</code></span></a>. (Contributed by Kabir Kwatra in <a href="https://github.com/python/cpython/issues/91973" class="reference external">gh-91973</a>.)

- <a href="../library/datetime.html#datetime.date.fromisoformat" class="reference internal" title="datetime.date.fromisoformat"><span class="pre"><code class="sourceCode python">datetime.date.fromisoformat()</code></span></a>, <a href="../library/datetime.html#datetime.time.fromisoformat" class="reference internal" title="datetime.time.fromisoformat"><span class="pre"><code class="sourceCode python">datetime.time.fromisoformat()</code></span></a> and <a href="../library/datetime.html#datetime.datetime.fromisoformat" class="reference internal" title="datetime.datetime.fromisoformat"><span class="pre"><code class="sourceCode python">datetime.datetime.fromisoformat()</code></span></a> can now be used to parse most ISO 8601 formats (barring only those that support fractional hours and minutes). (Contributed by Paul Ganssle in <a href="https://github.com/python/cpython/issues/80010" class="reference external">gh-80010</a>.)

</div>

<div id="enum" class="section">

<span id="whatsnew311-enum"></span>

### enum<a href="#enum" class="headerlink" title="Link to this heading">¶</a>

- Renamed <span class="pre">`EnumMeta`</span> to <a href="../library/enum.html#enum.EnumType" class="reference internal" title="enum.EnumType"><span class="pre"><code class="sourceCode python">EnumType</code></span></a> (<span class="pre">`EnumMeta`</span> kept as an alias).

- Added <a href="../library/enum.html#enum.StrEnum" class="reference internal" title="enum.StrEnum"><span class="pre"><code class="sourceCode python">StrEnum</code></span></a>, with members that can be used as (and must be) strings.

- Added <a href="../library/enum.html#enum.ReprEnum" class="reference internal" title="enum.ReprEnum"><span class="pre"><code class="sourceCode python">ReprEnum</code></span></a>, which only modifies the <a href="../reference/datamodel.html#object.__repr__" class="reference internal" title="object.__repr__"><span class="pre"><code class="sourceCode python"><span class="fu">__repr__</span>()</code></span></a> of members while returning their literal values (rather than names) for <a href="../reference/datamodel.html#object.__str__" class="reference internal" title="object.__str__"><span class="pre"><code class="sourceCode python"><span class="fu">__str__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__format__" class="reference internal" title="object.__format__"><span class="pre"><code class="sourceCode python"><span class="fu">__format__</span>()</code></span></a> (used by <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a>, <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> and <a href="../glossary.html#term-f-string" class="reference internal"><span class="xref std std-term">f-string</span></a>s).

- Changed <a href="../library/enum.html#enum.Enum.__format__" class="reference internal" title="enum.Enum.__format__"><span class="pre"><code class="sourceCode python">Enum.<span class="fu">__format__</span>()</code></span></a> (the default for <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a>, <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a> and <a href="../glossary.html#term-f-string" class="reference internal"><span class="xref std std-term">f-string</span></a>s) to always produce the same result as <a href="../library/enum.html#enum.Enum.__str__" class="reference internal" title="enum.Enum.__str__"><span class="pre"><code class="sourceCode python">Enum.<span class="fu">__str__</span>()</code></span></a>: for enums inheriting from <a href="../library/enum.html#enum.ReprEnum" class="reference internal" title="enum.ReprEnum"><span class="pre"><code class="sourceCode python">ReprEnum</code></span></a> it will be the member’s value; for all other enums it will be the enum and member name (e.g. <span class="pre">`Color.RED`</span>).

- Added a new *boundary* class parameter to <a href="../library/enum.html#enum.Flag" class="reference internal" title="enum.Flag"><span class="pre"><code class="sourceCode python">Flag</code></span></a> enums and the <a href="../library/enum.html#enum.FlagBoundary" class="reference internal" title="enum.FlagBoundary"><span class="pre"><code class="sourceCode python">FlagBoundary</code></span></a> enum with its options, to control how to handle out-of-range flag values.

- Added the <a href="../library/enum.html#enum.verify" class="reference internal" title="enum.verify"><span class="pre"><code class="sourceCode python">verify()</code></span></a> enum decorator and the <a href="../library/enum.html#enum.EnumCheck" class="reference internal" title="enum.EnumCheck"><span class="pre"><code class="sourceCode python">EnumCheck</code></span></a> enum with its options, to check enum classes against several specific constraints.

- Added the <a href="../library/enum.html#enum.member" class="reference internal" title="enum.member"><span class="pre"><code class="sourceCode python">member()</code></span></a> and <a href="../library/enum.html#enum.nonmember" class="reference internal" title="enum.nonmember"><span class="pre"><code class="sourceCode python">nonmember()</code></span></a> decorators, to ensure the decorated object is/is not converted to an enum member.

- Added the <a href="../library/enum.html#enum.property" class="reference internal" title="enum.property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span>()</code></span></a> decorator, which works like <a href="../library/functions.html#property" class="reference internal" title="property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span>()</code></span></a> except for enums. Use this instead of <a href="../library/types.html#types.DynamicClassAttribute" class="reference internal" title="types.DynamicClassAttribute"><span class="pre"><code class="sourceCode python">types.DynamicClassAttribute()</code></span></a>.

- Added the <a href="../library/enum.html#enum.global_enum" class="reference internal" title="enum.global_enum"><span class="pre"><code class="sourceCode python">global_enum()</code></span></a> enum decorator, which adjusts <a href="../reference/datamodel.html#object.__repr__" class="reference internal" title="object.__repr__"><span class="pre"><code class="sourceCode python"><span class="fu">__repr__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__str__" class="reference internal" title="object.__str__"><span class="pre"><code class="sourceCode python"><span class="fu">__str__</span>()</code></span></a> to show values as members of their module rather than the enum class. For example, <span class="pre">`'re.ASCII'`</span> for the <a href="../library/re.html#re.ASCII" class="reference internal" title="re.ASCII"><span class="pre"><code class="sourceCode python">ASCII</code></span></a> member of <a href="../library/re.html#re.RegexFlag" class="reference internal" title="re.RegexFlag"><span class="pre"><code class="sourceCode python">re.RegexFlag</code></span></a> rather than <span class="pre">`'RegexFlag.ASCII'`</span>.

- Enhanced <a href="../library/enum.html#enum.Flag" class="reference internal" title="enum.Flag"><span class="pre"><code class="sourceCode python">Flag</code></span></a> to support <a href="../library/functions.html#len" class="reference internal" title="len"><span class="pre"><code class="sourceCode python"><span class="bu">len</span>()</code></span></a>, iteration and <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a>/<a href="../reference/expressions.html#not-in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">not</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> on its members. For example, the following now works: <span class="pre">`len(AFlag(3))`</span>` `<span class="pre">`==`</span>` `<span class="pre">`2`</span>` `<span class="pre">`and`</span>` `<span class="pre">`list(AFlag(3))`</span>` `<span class="pre">`==`</span>` `<span class="pre">`(AFlag.ONE,`</span>` `<span class="pre">`AFlag.TWO)`</span>

- Changed <a href="../library/enum.html#enum.Enum" class="reference internal" title="enum.Enum"><span class="pre"><code class="sourceCode python">Enum</code></span></a> and <a href="../library/enum.html#enum.Flag" class="reference internal" title="enum.Flag"><span class="pre"><code class="sourceCode python">Flag</code></span></a> so that members are now defined before <a href="../reference/datamodel.html#object.__init_subclass__" class="reference internal" title="object.__init_subclass__"><span class="pre"><code class="sourceCode python"><span class="fu">__init_subclass__</span>()</code></span></a> is called; <a href="../library/functions.html#dir" class="reference internal" title="dir"><span class="pre"><code class="sourceCode python"><span class="bu">dir</span>()</code></span></a> now includes methods, etc., from mixed-in data types.

- Changed <a href="../library/enum.html#enum.Flag" class="reference internal" title="enum.Flag"><span class="pre"><code class="sourceCode python">Flag</code></span></a> to only consider primary values (power of two) canonical while composite values (<span class="pre">`3`</span>, <span class="pre">`6`</span>, <span class="pre">`10`</span>, etc.) are considered aliases; inverted flags are coerced to their positive equivalent.

</div>

<div id="fcntl" class="section">

<span id="whatsnew311-fcntl"></span>

### fcntl<a href="#fcntl" class="headerlink" title="Link to this heading">¶</a>

- On FreeBSD, the <span class="pre">`F_DUP2FD`</span> and <span class="pre">`F_DUP2FD_CLOEXEC`</span> flags respectively are supported, the former equals to <span class="pre">`dup2`</span> usage while the latter set the <span class="pre">`FD_CLOEXEC`</span> flag in addition.

</div>

<div id="fractions" class="section">

<span id="whatsnew311-fractions"></span>

### fractions<a href="#fractions" class="headerlink" title="Link to this heading">¶</a>

- Support <span id="index-27" class="target"></span><a href="https://peps.python.org/pep-0515/" class="pep reference external"><strong>PEP 515</strong></a>-style initialization of <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">Fraction</code></span></a> from string. (Contributed by Sergey B Kirpichev in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44258" class="reference external">bpo-44258</a>.)

- <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">Fraction</code></span></a> now implements an <span class="pre">`__int__`</span> method, so that an <span class="pre">`isinstance(some_fraction,`</span>` `<span class="pre">`typing.SupportsInt)`</span> check passes. (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44547" class="reference external">bpo-44547</a>.)

</div>

<div id="functools" class="section">

<span id="whatsnew311-functools"></span>

### functools<a href="#functools" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/functools.html#functools.singledispatch" class="reference internal" title="functools.singledispatch"><span class="pre"><code class="sourceCode python">functools.singledispatch()</code></span></a> now supports <a href="../library/types.html#types.UnionType" class="reference internal" title="types.UnionType"><span class="pre"><code class="sourceCode python">types.UnionType</code></span></a> and <a href="../library/typing.html#typing.Union" class="reference internal" title="typing.Union"><span class="pre"><code class="sourceCode python">typing.Union</code></span></a> as annotations to the dispatch argument.:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> from functools import singledispatch
      >>> @singledispatch
      ... def fun(arg, verbose=False):
      ...     if verbose:
      ...         print("Let me just say,", end=" ")
      ...     print(arg)
      ...
      >>> @fun.register
      ... def _(arg: int | float, verbose=False):
      ...     if verbose:
      ...         print("Strength in numbers, eh?", end=" ")
      ...     print(arg)
      ...
      >>> from typing import Union
      >>> @fun.register
      ... def _(arg: Union[list, set], verbose=False):
      ...     if verbose:
      ...         print("Enumerate this:")
      ...     for i, elem in enumerate(arg):
      ...         print(i, elem)
      ...

  </div>

  </div>

  (Contributed by Yurii Karabas in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46014" class="reference external">bpo-46014</a>.)

</div>

<div id="gzip" class="section">

<span id="whatsnew311-gzip"></span>

### gzip<a href="#gzip" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/gzip.html#gzip.compress" class="reference internal" title="gzip.compress"><span class="pre"><code class="sourceCode python">gzip.compress()</code></span></a> function is now faster when used with the **mtime=0** argument as it delegates the compression entirely to a single <a href="../library/zlib.html#zlib.compress" class="reference internal" title="zlib.compress"><span class="pre"><code class="sourceCode python">zlib.compress()</code></span></a> operation. There is one side effect of this change: The gzip file header contains an “OS” byte in its header. That was traditionally always set to a value of 255 representing “unknown” by the <a href="../library/gzip.html#module-gzip" class="reference internal" title="gzip: Interfaces for gzip compression and decompression using file objects."><span class="pre"><code class="sourceCode python">gzip</code></span></a> module. Now, when using <a href="../library/gzip.html#gzip.compress" class="reference internal" title="gzip.compress"><span class="pre"><code class="sourceCode python">compress()</code></span></a> with **mtime=0**, it may be set to a different value by the underlying zlib C library Python was linked against. (See <a href="https://github.com/python/cpython/issues/112346" class="reference external">gh-112346</a> for details on the side effect.)

</div>

<div id="hashlib" class="section">

<span id="whatsnew311-hashlib"></span>

### hashlib<a href="#hashlib" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/hashlib.html#hashlib.blake2b" class="reference internal" title="hashlib.blake2b"><span class="pre"><code class="sourceCode python">hashlib.blake2b()</code></span></a> and <a href="../library/hashlib.html#hashlib.blake2s" class="reference internal" title="hashlib.blake2s"><span class="pre"><code class="sourceCode python">hashlib.blake2s()</code></span></a> now prefer <a href="https://www.blake2.net/" class="reference external">libb2</a> over Python’s vendored copy. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=47095" class="reference external">bpo-47095</a>.)

- The internal <span class="pre">`_sha3`</span> module with SHA3 and SHAKE algorithms now uses *tiny_sha3* instead of the *Keccak Code Package* to reduce code and binary size. The <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module prefers optimized SHA3 and SHAKE implementations from OpenSSL. The change affects only installations without OpenSSL support. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=47098" class="reference external">bpo-47098</a>.)

- Add <a href="../library/hashlib.html#hashlib.file_digest" class="reference internal" title="hashlib.file_digest"><span class="pre"><code class="sourceCode python">hashlib.file_digest()</code></span></a>, a helper function for efficient hashing of files or file-like objects. (Contributed by Christian Heimes in <a href="https://github.com/python/cpython/issues/89313" class="reference external">gh-89313</a>.)

</div>

<div id="whatsnew311-idle" class="section">

<span id="idle-and-idlelib"></span>

### IDLE and idlelib<a href="#whatsnew311-idle" class="headerlink" title="Link to this heading">¶</a>

- Apply syntax highlighting to <span class="pre">`.pyi`</span> files. (Contributed by Alex Waygood and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45447" class="reference external">bpo-45447</a>.)

- Include prompts when saving Shell with inputs and outputs. (Contributed by Terry Jan Reedy in <a href="https://github.com/python/cpython/issues/95191" class="reference external">gh-95191</a>.)

</div>

<div id="inspect" class="section">

<span id="whatsnew311-inspect"></span>

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/inspect.html#inspect.getmembers_static" class="reference internal" title="inspect.getmembers_static"><span class="pre"><code class="sourceCode python">getmembers_static()</code></span></a> to return all members without triggering dynamic lookup via the descriptor protocol. (Contributed by Weipeng Hong in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30533" class="reference external">bpo-30533</a>.)

- Add <a href="../library/inspect.html#inspect.ismethodwrapper" class="reference internal" title="inspect.ismethodwrapper"><span class="pre"><code class="sourceCode python">ismethodwrapper()</code></span></a> for checking if the type of an object is a <a href="../library/types.html#types.MethodWrapperType" class="reference internal" title="types.MethodWrapperType"><span class="pre"><code class="sourceCode python">MethodWrapperType</code></span></a>. (Contributed by Hakan Çelik in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29418" class="reference external">bpo-29418</a>.)

- Change the frame-related functions in the <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> module to return new <a href="../library/inspect.html#inspect.FrameInfo" class="reference internal" title="inspect.FrameInfo"><span class="pre"><code class="sourceCode python">FrameInfo</code></span></a> and <a href="../library/inspect.html#inspect.Traceback" class="reference internal" title="inspect.Traceback"><span class="pre"><code class="sourceCode python">Traceback</code></span></a> class instances (backwards compatible with the previous <a href="../glossary.html#term-named-tuple" class="reference internal"><span class="xref std std-term">named tuple</span></a>-like interfaces) that includes the extended <span id="index-28" class="target"></span><a href="https://peps.python.org/pep-0657/" class="pep reference external"><strong>PEP 657</strong></a> position information (end line number, column and end column). The affected functions are:

  - <a href="../library/inspect.html#inspect.getframeinfo" class="reference internal" title="inspect.getframeinfo"><span class="pre"><code class="sourceCode python">inspect.getframeinfo()</code></span></a>

  - <a href="../library/inspect.html#inspect.getouterframes" class="reference internal" title="inspect.getouterframes"><span class="pre"><code class="sourceCode python">inspect.getouterframes()</code></span></a>

  - <a href="../library/inspect.html#inspect.getinnerframes" class="reference internal" title="inspect.getinnerframes"><span class="pre"><code class="sourceCode python">inspect.getinnerframes()</code></span></a>,

  - <a href="../library/inspect.html#inspect.stack" class="reference internal" title="inspect.stack"><span class="pre"><code class="sourceCode python">inspect.stack()</code></span></a>

  - <a href="../library/inspect.html#inspect.trace" class="reference internal" title="inspect.trace"><span class="pre"><code class="sourceCode python">inspect.trace()</code></span></a>

  (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/88116" class="reference external">gh-88116</a>.)

</div>

<div id="locale" class="section">

<span id="whatsnew311-locale"></span>

### locale<a href="#locale" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/locale.html#locale.getencoding" class="reference internal" title="locale.getencoding"><span class="pre"><code class="sourceCode python">locale.getencoding()</code></span></a> to get the current locale encoding. It is similar to <span class="pre">`locale.getpreferredencoding(False)`</span> but ignores the <a href="../library/os.html#utf8-mode" class="reference internal"><span class="std std-ref">Python UTF-8 Mode</span></a>.

</div>

<div id="logging" class="section">

<span id="whatsnew311-logging"></span>

### logging<a href="#logging" class="headerlink" title="Link to this heading">¶</a>

- Added <a href="../library/logging.html#logging.getLevelNamesMapping" class="reference internal" title="logging.getLevelNamesMapping"><span class="pre"><code class="sourceCode python">getLevelNamesMapping()</code></span></a> to return a mapping from logging level names (e.g. <span class="pre">`'CRITICAL'`</span>) to the values of their corresponding <a href="../library/logging.html#levels" class="reference internal"><span class="std std-ref">Logging Levels</span></a> (e.g. <span class="pre">`50`</span>, by default). (Contributed by Andrei Kulakovin in <a href="https://github.com/python/cpython/issues/88024" class="reference external">gh-88024</a>.)

- Added a <a href="../library/logging.handlers.html#logging.handlers.SysLogHandler.createSocket" class="reference internal" title="logging.handlers.SysLogHandler.createSocket"><span class="pre"><code class="sourceCode python">createSocket()</code></span></a> method to <a href="../library/logging.handlers.html#logging.handlers.SysLogHandler" class="reference internal" title="logging.handlers.SysLogHandler"><span class="pre"><code class="sourceCode python">SysLogHandler</code></span></a>, to match <a href="../library/logging.handlers.html#logging.handlers.SocketHandler.createSocket" class="reference internal" title="logging.handlers.SocketHandler.createSocket"><span class="pre"><code class="sourceCode python">SocketHandler.createSocket()</code></span></a>. It is called automatically during handler initialization and when emitting an event, if there is no active socket. (Contributed by Kirill Pinchuk in <a href="https://github.com/python/cpython/issues/88457" class="reference external">gh-88457</a>.)

</div>

<div id="math" class="section">

<span id="whatsnew311-math"></span>

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/math.html#math.exp2" class="reference internal" title="math.exp2"><span class="pre"><code class="sourceCode python">math.exp2()</code></span></a>: return 2 raised to the power of x. (Contributed by Gideon Mitchell in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45917" class="reference external">bpo-45917</a>.)

- Add <a href="../library/math.html#math.cbrt" class="reference internal" title="math.cbrt"><span class="pre"><code class="sourceCode python">math.cbrt()</code></span></a>: return the cube root of x. (Contributed by Ajith Ramachandran in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44357" class="reference external">bpo-44357</a>.)

- The behaviour of two <a href="../library/math.html#math.pow" class="reference internal" title="math.pow"><span class="pre"><code class="sourceCode python">math.<span class="bu">pow</span>()</code></span></a> corner cases was changed, for consistency with the IEEE 754 specification. The operations <span class="pre">`math.pow(0.0,`</span>` `<span class="pre">`-math.inf)`</span> and <span class="pre">`math.pow(-0.0,`</span>` `<span class="pre">`-math.inf)`</span> now return <span class="pre">`inf`</span>. Previously they raised <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a>. (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44339" class="reference external">bpo-44339</a>.)

- The <a href="../library/math.html#math.nan" class="reference internal" title="math.nan"><span class="pre"><code class="sourceCode python">math.nan</code></span></a> value is now always available. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46917" class="reference external">bpo-46917</a>.)

</div>

<div id="operator" class="section">

<span id="whatsnew311-operator"></span>

### operator<a href="#operator" class="headerlink" title="Link to this heading">¶</a>

- A new function <span class="pre">`operator.call`</span> has been added, such that <span class="pre">`operator.call(obj,`</span>` `<span class="pre">`*args,`</span>` `<span class="pre">`**kwargs)`</span>` `<span class="pre">`==`</span>` `<span class="pre">`obj(*args,`</span>` `<span class="pre">`**kwargs)`</span>. (Contributed by Antony Lee in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44019" class="reference external">bpo-44019</a>.)

</div>

<div id="os" class="section">

<span id="whatsnew311-os"></span>

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

- On Windows, <a href="../library/os.html#os.urandom" class="reference internal" title="os.urandom"><span class="pre"><code class="sourceCode python">os.urandom()</code></span></a> now uses <span class="pre">`BCryptGenRandom()`</span>, instead of <span class="pre">`CryptGenRandom()`</span> which is deprecated. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44611" class="reference external">bpo-44611</a>.)

</div>

<div id="pathlib" class="section">

<span id="whatsnew311-pathlib"></span>

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/pathlib.html#pathlib.Path.glob" class="reference internal" title="pathlib.Path.glob"><span class="pre"><code class="sourceCode python">glob()</code></span></a> and <a href="../library/pathlib.html#pathlib.Path.rglob" class="reference internal" title="pathlib.Path.rglob"><span class="pre"><code class="sourceCode python">rglob()</code></span></a> return only directories if *pattern* ends with a pathname components separator: <a href="../library/os.html#os.sep" class="reference internal" title="os.sep"><span class="pre"><code class="sourceCode python">sep</code></span></a> or <a href="../library/os.html#os.altsep" class="reference internal" title="os.altsep"><span class="pre"><code class="sourceCode python">altsep</code></span></a>. (Contributed by Eisuke Kawasima in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22276" class="reference external">bpo-22276</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33392" class="reference external">bpo-33392</a>.)

</div>

<div id="re" class="section">

<span id="whatsnew311-re"></span>

### re<a href="#re" class="headerlink" title="Link to this heading">¶</a>

- Atomic grouping (<span class="pre">`(?>...)`</span>) and possessive quantifiers (<span class="pre">`*+`</span>, <span class="pre">`++`</span>, <span class="pre">`?+`</span>, <span class="pre">`{m,n}+`</span>) are now supported in regular expressions. (Contributed by Jeffrey C. Jacobs and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=433030" class="reference external">bpo-433030</a>.)

</div>

<div id="shutil" class="section">

<span id="whatsnew311-shutil"></span>

### shutil<a href="#shutil" class="headerlink" title="Link to this heading">¶</a>

- Add optional parameter *dir_fd* in <a href="../library/shutil.html#shutil.rmtree" class="reference internal" title="shutil.rmtree"><span class="pre"><code class="sourceCode python">shutil.rmtree()</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46245" class="reference external">bpo-46245</a>.)

</div>

<div id="socket" class="section">

<span id="whatsnew311-socket"></span>

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

- Add CAN Socket support for NetBSD. (Contributed by Thomas Klausner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30512" class="reference external">bpo-30512</a>.)

- <a href="../library/socket.html#socket.create_connection" class="reference internal" title="socket.create_connection"><span class="pre"><code class="sourceCode python">create_connection()</code></span></a> has an option to raise, in case of failure to connect, an <a href="../library/exceptions.html#ExceptionGroup" class="reference internal" title="ExceptionGroup"><span class="pre"><code class="sourceCode python">ExceptionGroup</code></span></a> containing all errors instead of only raising the last error. (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29980" class="reference external">bpo-29980</a>.)

</div>

<div id="sqlite3" class="section">

<span id="whatsnew311-sqlite3"></span>

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

- You can now disable the authorizer by passing <a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a> to <a href="../library/sqlite3.html#sqlite3.Connection.set_authorizer" class="reference internal" title="sqlite3.Connection.set_authorizer"><span class="pre"><code class="sourceCode python">set_authorizer()</code></span></a>. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44491" class="reference external">bpo-44491</a>.)

- Collation name <a href="../library/sqlite3.html#sqlite3.Connection.create_collation" class="reference internal" title="sqlite3.Connection.create_collation"><span class="pre"><code class="sourceCode python">create_collation()</code></span></a> can now contain any Unicode character. Collation names with invalid characters now raise <a href="../library/exceptions.html#UnicodeEncodeError" class="reference internal" title="UnicodeEncodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeEncodeError</span></code></span></a> instead of <a href="../library/sqlite3.html#sqlite3.ProgrammingError" class="reference internal" title="sqlite3.ProgrammingError"><span class="pre"><code class="sourceCode python">sqlite3.ProgrammingError</code></span></a>. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44688" class="reference external">bpo-44688</a>.)

- <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> exceptions now include the SQLite extended error code as <a href="../library/sqlite3.html#sqlite3.Error.sqlite_errorcode" class="reference internal" title="sqlite3.Error.sqlite_errorcode"><span class="pre"><code class="sourceCode python">sqlite_errorcode</code></span></a> and the SQLite error name as <a href="../library/sqlite3.html#sqlite3.Error.sqlite_errorname" class="reference internal" title="sqlite3.Error.sqlite_errorname"><span class="pre"><code class="sourceCode python">sqlite_errorname</code></span></a>. (Contributed by Aviv Palivoda, Daniel Shahaf, and Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16379" class="reference external">bpo-16379</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24139" class="reference external">bpo-24139</a>.)

- Add <a href="../library/sqlite3.html#sqlite3.Connection.setlimit" class="reference internal" title="sqlite3.Connection.setlimit"><span class="pre"><code class="sourceCode python">setlimit()</code></span></a> and <a href="../library/sqlite3.html#sqlite3.Connection.getlimit" class="reference internal" title="sqlite3.Connection.getlimit"><span class="pre"><code class="sourceCode python">getlimit()</code></span></a> to <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">sqlite3.Connection</code></span></a> for setting and getting SQLite limits by connection basis. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45243" class="reference external">bpo-45243</a>.)

- <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> now sets <a href="../library/sqlite3.html#sqlite3.threadsafety" class="reference internal" title="sqlite3.threadsafety"><span class="pre"><code class="sourceCode python">sqlite3.threadsafety</code></span></a> based on the default threading mode the underlying SQLite library has been compiled with. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45613" class="reference external">bpo-45613</a>.)

- <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> C callbacks now use unraisable exceptions if callback tracebacks are enabled. Users can now register an <a href="../library/sys.html#sys.unraisablehook" class="reference internal" title="sys.unraisablehook"><span class="pre"><code class="sourceCode python">unraisable</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">hook</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">handler</code></span></a> to improve their debug experience. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45828" class="reference external">bpo-45828</a>.)

- Fetch across rollback no longer raises <a href="../library/sqlite3.html#sqlite3.InterfaceError" class="reference internal" title="sqlite3.InterfaceError"><span class="pre"><code class="sourceCode python">InterfaceError</code></span></a>. Instead we leave it to the SQLite library to handle these cases. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44092" class="reference external">bpo-44092</a>.)

- Add <a href="../library/sqlite3.html#sqlite3.Connection.serialize" class="reference internal" title="sqlite3.Connection.serialize"><span class="pre"><code class="sourceCode python">serialize()</code></span></a> and <a href="../library/sqlite3.html#sqlite3.Connection.deserialize" class="reference internal" title="sqlite3.Connection.deserialize"><span class="pre"><code class="sourceCode python">deserialize()</code></span></a> to <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">sqlite3.Connection</code></span></a> for serializing and deserializing databases. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41930" class="reference external">bpo-41930</a>.)

- Add <a href="../library/sqlite3.html#sqlite3.Connection.create_window_function" class="reference internal" title="sqlite3.Connection.create_window_function"><span class="pre"><code class="sourceCode python">create_window_function()</code></span></a> to <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">sqlite3.Connection</code></span></a> for creating aggregate window functions. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34916" class="reference external">bpo-34916</a>.)

- Add <a href="../library/sqlite3.html#sqlite3.Connection.blobopen" class="reference internal" title="sqlite3.Connection.blobopen"><span class="pre"><code class="sourceCode python">blobopen()</code></span></a> to <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">sqlite3.Connection</code></span></a>. <a href="../library/sqlite3.html#sqlite3.Blob" class="reference internal" title="sqlite3.Blob"><span class="pre"><code class="sourceCode python">sqlite3.Blob</code></span></a> allows incremental I/O operations on blobs. (Contributed by Aviv Palivoda and Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24905" class="reference external">bpo-24905</a>.)

</div>

<div id="string" class="section">

<span id="whatsnew311-string"></span>

### string<a href="#string" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/string.html#string.Template.get_identifiers" class="reference internal" title="string.Template.get_identifiers"><span class="pre"><code class="sourceCode python">get_identifiers()</code></span></a> and <a href="../library/string.html#string.Template.is_valid" class="reference internal" title="string.Template.is_valid"><span class="pre"><code class="sourceCode python">is_valid()</code></span></a> to <a href="../library/string.html#string.Template" class="reference internal" title="string.Template"><span class="pre"><code class="sourceCode python">string.Template</code></span></a>, which respectively return all valid placeholders, and whether any invalid placeholders are present. (Contributed by Ben Kehoe in <a href="https://github.com/python/cpython/issues/90465" class="reference external">gh-90465</a>.)

</div>

<div id="sys" class="section">

<span id="whatsnew311-sys"></span>

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/sys.html#sys.exc_info" class="reference internal" title="sys.exc_info"><span class="pre"><code class="sourceCode python">sys.exc_info()</code></span></a> now derives the <span class="pre">`type`</span> and <span class="pre">`traceback`</span> fields from the <span class="pre">`value`</span> (the exception instance), so when an exception is modified while it is being handled, the changes are reflected in the results of subsequent calls to <span class="pre">`exc_info()`</span>. (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45711" class="reference external">bpo-45711</a>.)

- Add <a href="../library/sys.html#sys.exception" class="reference internal" title="sys.exception"><span class="pre"><code class="sourceCode python">sys.exception()</code></span></a> which returns the active exception instance (equivalent to <span class="pre">`sys.exc_info()[1]`</span>). (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46328" class="reference external">bpo-46328</a>.)

- Add the <a href="../library/sys.html#sys.flags" class="reference internal" title="sys.flags"><span class="pre"><code class="sourceCode python">sys.flags.safe_path</code></span></a> flag. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/57684" class="reference external">gh-57684</a>.)

</div>

<div id="sysconfig" class="section">

<span id="whatsnew311-sysconfig"></span>

### sysconfig<a href="#sysconfig" class="headerlink" title="Link to this heading">¶</a>

- Three new <a href="../library/sysconfig.html#installation-paths" class="reference internal"><span class="std std-ref">installation schemes</span></a> (*posix_venv*, *nt_venv* and *venv*) were added and are used when Python creates new virtual environments or when it is running from a virtual environment. The first two schemes (*posix_venv* and *nt_venv*) are OS-specific for non-Windows and Windows, the *venv* is essentially an alias to one of them according to the OS Python runs on. This is useful for downstream distributors who modify <a href="../library/sysconfig.html#sysconfig.get_preferred_scheme" class="reference internal" title="sysconfig.get_preferred_scheme"><span class="pre"><code class="sourceCode python">sysconfig.get_preferred_scheme()</code></span></a>. Third party code that creates new virtual environments should use the new *venv* installation scheme to determine the paths, as does <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a>. (Contributed by Miro Hrončok in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45413" class="reference external">bpo-45413</a>.)

</div>

<div id="tempfile" class="section">

<span id="whatsnew311-tempfile"></span>

### tempfile<a href="#tempfile" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/tempfile.html#tempfile.SpooledTemporaryFile" class="reference internal" title="tempfile.SpooledTemporaryFile"><span class="pre"><code class="sourceCode python">SpooledTemporaryFile</code></span></a> objects now fully implement the methods of <a href="../library/io.html#io.BufferedIOBase" class="reference internal" title="io.BufferedIOBase"><span class="pre"><code class="sourceCode python">io.BufferedIOBase</code></span></a> or <a href="../library/io.html#io.TextIOBase" class="reference internal" title="io.TextIOBase"><span class="pre"><code class="sourceCode python">io.TextIOBase</code></span></a> (depending on file mode). This lets them work correctly with APIs that expect file-like objects, such as compression modules. (Contributed by Carey Metcalfe in <a href="https://github.com/python/cpython/issues/70363" class="reference external">gh-70363</a>.)

</div>

<div id="threading" class="section">

<span id="whatsnew311-threading"></span>

### threading<a href="#threading" class="headerlink" title="Link to this heading">¶</a>

- On Unix, if the <span class="pre">`sem_clockwait()`</span> function is available in the C library (glibc 2.30 and newer), the <a href="../library/threading.html#threading.Lock.acquire" class="reference internal" title="threading.Lock.acquire"><span class="pre"><code class="sourceCode python">threading.Lock.acquire()</code></span></a> method now uses the monotonic clock (<a href="../library/time.html#time.CLOCK_MONOTONIC" class="reference internal" title="time.CLOCK_MONOTONIC"><span class="pre"><code class="sourceCode python">time.CLOCK_MONOTONIC</code></span></a>) for the timeout, rather than using the system clock (<a href="../library/time.html#time.CLOCK_REALTIME" class="reference internal" title="time.CLOCK_REALTIME"><span class="pre"><code class="sourceCode python">time.CLOCK_REALTIME</code></span></a>), to not be affected by system clock changes. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41710" class="reference external">bpo-41710</a>.)

</div>

<div id="time" class="section">

<span id="whatsnew311-time"></span>

### time<a href="#time" class="headerlink" title="Link to this heading">¶</a>

- On Unix, <a href="../library/time.html#time.sleep" class="reference internal" title="time.sleep"><span class="pre"><code class="sourceCode python">time.sleep()</code></span></a> now uses the <span class="pre">`clock_nanosleep()`</span> or <span class="pre">`nanosleep()`</span> function, if available, which has a resolution of 1 nanosecond (10<sup>-9</sup> seconds), rather than using <span class="pre">`select()`</span> which has a resolution of 1 microsecond (10<sup>-6</sup> seconds). (Contributed by Benjamin Szőke and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21302" class="reference external">bpo-21302</a>.)

- On Windows 8.1 and newer, <a href="../library/time.html#time.sleep" class="reference internal" title="time.sleep"><span class="pre"><code class="sourceCode python">time.sleep()</code></span></a> now uses a waitable timer based on <a href="https://docs.microsoft.com/en-us/windows-hardware/drivers/kernel/high-resolution-timers" class="reference external">high-resolution timers</a> which has a resolution of 100 nanoseconds (10<sup>-7</sup> seconds). Previously, it had a resolution of 1 millisecond (10<sup>-3</sup> seconds). (Contributed by Benjamin Szőke, Donghee Na, Eryk Sun and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21302" class="reference external">bpo-21302</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45429" class="reference external">bpo-45429</a>.)

</div>

<div id="tkinter" class="section">

<span id="whatsnew311-tkinter"></span>

### tkinter<a href="#tkinter" class="headerlink" title="Link to this heading">¶</a>

- Added method <span class="pre">`info_patchlevel()`</span> which returns the exact version of the Tcl library as a named tuple similar to <a href="../library/sys.html#sys.version_info" class="reference internal" title="sys.version_info"><span class="pre"><code class="sourceCode python">sys.version_info</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/91827" class="reference external">gh-91827</a>.)

</div>

<div id="traceback" class="section">

<span id="whatsnew311-traceback"></span>

### traceback<a href="#traceback" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/traceback.html#traceback.StackSummary.format_frame_summary" class="reference internal" title="traceback.StackSummary.format_frame_summary"><span class="pre"><code class="sourceCode python">traceback.StackSummary.format_frame_summary()</code></span></a> to allow users to override which frames appear in the traceback, and how they are formatted. (Contributed by Ammar Askar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44569" class="reference external">bpo-44569</a>.)

- Add <a href="../library/traceback.html#traceback.TracebackException.print" class="reference internal" title="traceback.TracebackException.print"><span class="pre"><code class="sourceCode python">traceback.TracebackException.<span class="bu">print</span>()</code></span></a>, which prints the formatted <a href="../library/traceback.html#traceback.TracebackException" class="reference internal" title="traceback.TracebackException"><span class="pre"><code class="sourceCode python">TracebackException</code></span></a> instance to a file. (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33809" class="reference external">bpo-33809</a>.)

</div>

<div id="typing" class="section">

<span id="whatsnew311-typing"></span>

### typing<a href="#typing" class="headerlink" title="Link to this heading">¶</a>

For major changes, see <a href="#new-feat-related-type-hints-311" class="reference internal"><span class="std std-ref">New Features Related to Type Hints</span></a>.

- Add <a href="../library/typing.html#typing.assert_never" class="reference internal" title="typing.assert_never"><span class="pre"><code class="sourceCode python">typing.assert_never()</code></span></a> and <a href="../library/typing.html#typing.Never" class="reference internal" title="typing.Never"><span class="pre"><code class="sourceCode python">typing.Never</code></span></a>. <a href="../library/typing.html#typing.assert_never" class="reference internal" title="typing.assert_never"><span class="pre"><code class="sourceCode python">typing.assert_never()</code></span></a> is useful for asking a type checker to confirm that a line of code is not reachable. At runtime, it raises an <a href="../library/exceptions.html#AssertionError" class="reference internal" title="AssertionError"><span class="pre"><code class="sourceCode python"><span class="pp">AssertionError</span></code></span></a>. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/90633" class="reference external">gh-90633</a>.)

- Add <a href="../library/typing.html#typing.reveal_type" class="reference internal" title="typing.reveal_type"><span class="pre"><code class="sourceCode python">typing.reveal_type()</code></span></a>. This is useful for asking a type checker what type it has inferred for a given expression. At runtime it prints the type of the received value. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/90572" class="reference external">gh-90572</a>.)

- Add <a href="../library/typing.html#typing.assert_type" class="reference internal" title="typing.assert_type"><span class="pre"><code class="sourceCode python">typing.assert_type()</code></span></a>. This is useful for asking a type checker to confirm that the type it has inferred for a given expression matches the given type. At runtime it simply returns the received value. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/90638" class="reference external">gh-90638</a>.)

- <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">typing.TypedDict</code></span></a> types can now be generic. (Contributed by Samodya Abeysiriwardane in <a href="https://github.com/python/cpython/issues/89026" class="reference external">gh-89026</a>.)

- <a href="../library/typing.html#typing.NamedTuple" class="reference internal" title="typing.NamedTuple"><span class="pre"><code class="sourceCode python">NamedTuple</code></span></a> types can now be generic. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43923" class="reference external">bpo-43923</a>.)

- Allow subclassing of <a href="../library/typing.html#typing.Any" class="reference internal" title="typing.Any"><span class="pre"><code class="sourceCode python">typing.Any</code></span></a>. This is useful for avoiding type checker errors related to highly dynamic class, such as mocks. (Contributed by Shantanu Jain in <a href="https://github.com/python/cpython/issues/91154" class="reference external">gh-91154</a>.)

- The <a href="../library/typing.html#typing.final" class="reference internal" title="typing.final"><span class="pre"><code class="sourceCode python">typing.final()</code></span></a> decorator now sets the <span class="pre">`__final__`</span> attributed on the decorated object. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/90500" class="reference external">gh-90500</a>.)

- The <a href="../library/typing.html#typing.get_overloads" class="reference internal" title="typing.get_overloads"><span class="pre"><code class="sourceCode python">typing.get_overloads()</code></span></a> function can be used for introspecting the overloads of a function. <a href="../library/typing.html#typing.clear_overloads" class="reference internal" title="typing.clear_overloads"><span class="pre"><code class="sourceCode python">typing.clear_overloads()</code></span></a> can be used to clear all registered overloads of a function. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/89263" class="reference external">gh-89263</a>.)

- The <a href="../reference/datamodel.html#object.__init__" class="reference internal" title="object.__init__"><span class="pre"><code class="sourceCode python"><span class="fu">__init__</span>()</code></span></a> method of <a href="../library/typing.html#typing.Protocol" class="reference internal" title="typing.Protocol"><span class="pre"><code class="sourceCode python">Protocol</code></span></a> subclasses is now preserved. (Contributed by Adrian Garcia Badarasco in <a href="https://github.com/python/cpython/issues/88970" class="reference external">gh-88970</a>.)

- The representation of empty tuple types (<span class="pre">`Tuple[()]`</span>) is simplified. This affects introspection, e.g. <span class="pre">`get_args(Tuple[()])`</span> now evaluates to <span class="pre">`()`</span> instead of <span class="pre">`((),)`</span>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/91137" class="reference external">gh-91137</a>.)

- Loosen runtime requirements for type annotations by removing the callable check in the private <span class="pre">`typing._type_check`</span> function. (Contributed by Gregory Beauregard in <a href="https://github.com/python/cpython/issues/90802" class="reference external">gh-90802</a>.)

- <a href="../library/typing.html#typing.get_type_hints" class="reference internal" title="typing.get_type_hints"><span class="pre"><code class="sourceCode python">typing.get_type_hints()</code></span></a> now supports evaluating strings as forward references in <a href="../library/stdtypes.html#types-genericalias" class="reference internal"><span class="std std-ref">PEP 585 generic aliases</span></a>. (Contributed by Niklas Rosenstein in <a href="https://github.com/python/cpython/issues/85542" class="reference external">gh-85542</a>.)

- <a href="../library/typing.html#typing.get_type_hints" class="reference internal" title="typing.get_type_hints"><span class="pre"><code class="sourceCode python">typing.get_type_hints()</code></span></a> no longer adds <a href="../library/typing.html#typing.Optional" class="reference internal" title="typing.Optional"><span class="pre"><code class="sourceCode python">Optional</code></span></a> to parameters with <span class="pre">`None`</span> as a default. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/90353" class="reference external">gh-90353</a>.)

- <a href="../library/typing.html#typing.get_type_hints" class="reference internal" title="typing.get_type_hints"><span class="pre"><code class="sourceCode python">typing.get_type_hints()</code></span></a> now supports evaluating bare stringified <a href="../library/typing.html#typing.ClassVar" class="reference internal" title="typing.ClassVar"><span class="pre"><code class="sourceCode python">ClassVar</code></span></a> annotations. (Contributed by Gregory Beauregard in <a href="https://github.com/python/cpython/issues/90711" class="reference external">gh-90711</a>.)

- <a href="../library/typing.html#typing.no_type_check" class="reference internal" title="typing.no_type_check"><span class="pre"><code class="sourceCode python">typing.no_type_check()</code></span></a> no longer modifies external classes and functions. It also now correctly marks classmethods as not to be type checked. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/90729" class="reference external">gh-90729</a>.)

</div>

<div id="unicodedata" class="section">

<span id="whatsnew311-unicodedata"></span>

### unicodedata<a href="#unicodedata" class="headerlink" title="Link to this heading">¶</a>

- The Unicode database has been updated to version 14.0.0. (Contributed by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45190" class="reference external">bpo-45190</a>).

</div>

<div id="unittest" class="section">

<span id="whatsnew311-unittest"></span>

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

- Added methods <a href="../library/unittest.html#unittest.TestCase.enterContext" class="reference internal" title="unittest.TestCase.enterContext"><span class="pre"><code class="sourceCode python">enterContext()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.enterClassContext" class="reference internal" title="unittest.TestCase.enterClassContext"><span class="pre"><code class="sourceCode python">enterClassContext()</code></span></a> of class <a href="../library/unittest.html#unittest.TestCase" class="reference internal" title="unittest.TestCase"><span class="pre"><code class="sourceCode python">TestCase</code></span></a>, method <a href="../library/unittest.html#unittest.IsolatedAsyncioTestCase.enterAsyncContext" class="reference internal" title="unittest.IsolatedAsyncioTestCase.enterAsyncContext"><span class="pre"><code class="sourceCode python">enterAsyncContext()</code></span></a> of class <a href="../library/unittest.html#unittest.IsolatedAsyncioTestCase" class="reference internal" title="unittest.IsolatedAsyncioTestCase"><span class="pre"><code class="sourceCode python">IsolatedAsyncioTestCase</code></span></a> and function <a href="../library/unittest.html#unittest.enterModuleContext" class="reference internal" title="unittest.enterModuleContext"><span class="pre"><code class="sourceCode python">unittest.enterModuleContext()</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45046" class="reference external">bpo-45046</a>.)

</div>

<div id="venv" class="section">

<span id="whatsnew311-venv"></span>

### venv<a href="#venv" class="headerlink" title="Link to this heading">¶</a>

- When new Python virtual environments are created, the *venv* <a href="../library/sysconfig.html#installation-paths" class="reference internal"><span class="std std-ref">sysconfig installation scheme</span></a> is used to determine the paths inside the environment. When Python runs in a virtual environment, the same installation scheme is the default. That means that downstream distributors can change the default sysconfig install scheme without changing behavior of virtual environments. Third party code that also creates new virtual environments should do the same. (Contributed by Miro Hrončok in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45413" class="reference external">bpo-45413</a>.)

</div>

<div id="warnings" class="section">

<span id="whatsnew311-warnings"></span>

### warnings<a href="#warnings" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/warnings.html#warnings.catch_warnings" class="reference internal" title="warnings.catch_warnings"><span class="pre"><code class="sourceCode python">warnings.catch_warnings()</code></span></a> now accepts arguments for <a href="../library/warnings.html#warnings.simplefilter" class="reference internal" title="warnings.simplefilter"><span class="pre"><code class="sourceCode python">warnings.simplefilter()</code></span></a>, providing a more concise way to locally ignore warnings or convert them to errors. (Contributed by Zac Hatfield-Dodds in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=47074" class="reference external">bpo-47074</a>.)

</div>

<div id="zipfile" class="section">

<span id="whatsnew311-zipfile"></span>

### zipfile<a href="#zipfile" class="headerlink" title="Link to this heading">¶</a>

- Added support for specifying member name encoding for reading metadata in a <a href="../library/zipfile.html#zipfile.ZipFile" class="reference internal" title="zipfile.ZipFile"><span class="pre"><code class="sourceCode python">ZipFile</code></span></a>’s directory and file headers. (Contributed by Stephen J. Turnbull and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28080" class="reference external">bpo-28080</a>.)

- Added <a href="../library/zipfile.html#zipfile.ZipFile.mkdir" class="reference internal" title="zipfile.ZipFile.mkdir"><span class="pre"><code class="sourceCode python">ZipFile.mkdir()</code></span></a> for creating new directories inside ZIP archives. (Contributed by Sam Ezeh in <a href="https://github.com/python/cpython/issues/49083" class="reference external">gh-49083</a>.)

- Added <a href="../library/zipfile.html#zipfile.Path.stem" class="reference internal" title="zipfile.Path.stem"><span class="pre"><code class="sourceCode python">stem</code></span></a>, <a href="../library/zipfile.html#zipfile.Path.suffix" class="reference internal" title="zipfile.Path.suffix"><span class="pre"><code class="sourceCode python">suffix</code></span></a> and <a href="../library/zipfile.html#zipfile.Path.suffixes" class="reference internal" title="zipfile.Path.suffixes"><span class="pre"><code class="sourceCode python">suffixes</code></span></a> to <a href="../library/zipfile.html#zipfile.Path" class="reference internal" title="zipfile.Path"><span class="pre"><code class="sourceCode python">zipfile.Path</code></span></a>. (Contributed by Miguel Brito in <a href="https://github.com/python/cpython/issues/88261" class="reference external">gh-88261</a>.)

</div>

</div>

<div id="optimizations" class="section">

<span id="whatsnew311-optimizations"></span>

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

This section covers specific optimizations independent of the <a href="#whatsnew311-faster-cpython" class="reference internal"><span class="std std-ref">Faster CPython</span></a> project, which is covered in its own section.

- The compiler now optimizes simple <a href="../library/stdtypes.html#old-string-formatting" class="reference internal"><span class="std std-ref">printf-style % formatting</span></a> on string literals containing only the format codes <span class="pre">`%s`</span>, <span class="pre">`%r`</span> and <span class="pre">`%a`</span> and makes it as fast as a corresponding <a href="../glossary.html#term-f-string" class="reference internal"><span class="xref std std-term">f-string</span></a> expression. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28307" class="reference external">bpo-28307</a>.)

- Integer division (<span class="pre">`//`</span>) is better tuned for optimization by compilers. It is now around 20% faster on x86-64 when dividing an <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> by a value smaller than <span class="pre">`2**30`</span>. (Contributed by Gregory P. Smith and Tim Peters in <a href="https://github.com/python/cpython/issues/90564" class="reference external">gh-90564</a>.)

- <a href="../library/functions.html#sum" class="reference internal" title="sum"><span class="pre"><code class="sourceCode python"><span class="bu">sum</span>()</code></span></a> is now nearly 30% faster for integers smaller than <span class="pre">`2**30`</span>. (Contributed by Stefan Behnel in <a href="https://github.com/python/cpython/issues/68264" class="reference external">gh-68264</a>.)

- Resizing lists is streamlined for the common case, speeding up <a href="../library/stdtypes.html#list.append" class="reference internal" title="list.append"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>.append()</code></span></a> by ≈15% and simple <a href="../glossary.html#term-list-comprehension" class="reference internal"><span class="xref std std-term">list comprehension</span></a>s by up to 20-30% (Contributed by Dennis Sweeney in <a href="https://github.com/python/cpython/issues/91165" class="reference external">gh-91165</a>.)

- Dictionaries don’t store hash values when all keys are Unicode objects, decreasing <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> size. For example, <span class="pre">`sys.getsizeof(dict.fromkeys("abcdefg"))`</span> is reduced from 352 bytes to 272 bytes (23% smaller) on 64-bit platforms. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46845" class="reference external">bpo-46845</a>.)

- Using <a href="../library/asyncio-protocol.html#asyncio.DatagramProtocol" class="reference internal" title="asyncio.DatagramProtocol"><span class="pre"><code class="sourceCode python">asyncio.DatagramProtocol</code></span></a> is now orders of magnitude faster when transferring large files over UDP, with speeds over 100 times higher for a ≈60 MiB file. (Contributed by msoxzw in <a href="https://github.com/python/cpython/issues/91487" class="reference external">gh-91487</a>.)

- <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> functions <a href="../library/math.html#math.comb" class="reference internal" title="math.comb"><span class="pre"><code class="sourceCode python">comb()</code></span></a> and <a href="../library/math.html#math.perm" class="reference internal" title="math.perm"><span class="pre"><code class="sourceCode python">perm()</code></span></a> are now ≈10 times faster for large arguments (with a larger speedup for larger *k*). (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37295" class="reference external">bpo-37295</a>.)

- The <a href="../library/statistics.html#module-statistics" class="reference internal" title="statistics: Mathematical statistics functions"><span class="pre"><code class="sourceCode python">statistics</code></span></a> functions <a href="../library/statistics.html#statistics.mean" class="reference internal" title="statistics.mean"><span class="pre"><code class="sourceCode python">mean()</code></span></a>, <a href="../library/statistics.html#statistics.variance" class="reference internal" title="statistics.variance"><span class="pre"><code class="sourceCode python">variance()</code></span></a> and <a href="../library/statistics.html#statistics.stdev" class="reference internal" title="statistics.stdev"><span class="pre"><code class="sourceCode python">stdev()</code></span></a> now consume iterators in one pass rather than converting them to a <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a> first. This is twice as fast and can save substantial memory. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/90415" class="reference external">gh-90415</a>.)

- <a href="../library/unicodedata.html#unicodedata.normalize" class="reference internal" title="unicodedata.normalize"><span class="pre"><code class="sourceCode python">unicodedata.normalize()</code></span></a> now normalizes pure-ASCII strings in constant time. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44987" class="reference external">bpo-44987</a>.)

</div>

<div id="faster-cpython" class="section">

<span id="whatsnew311-faster-cpython"></span>

## Faster CPython<a href="#faster-cpython" class="headerlink" title="Link to this heading">¶</a>

CPython 3.11 is an average of <a href="https://github.com/faster-cpython/ideas#published-results" class="reference external">25% faster</a> than CPython 3.10 as measured with the <a href="https://github.com/python/pyperformance" class="reference external">pyperformance</a> benchmark suite, when compiled with GCC on Ubuntu Linux. Depending on your workload, the overall speedup could be 10-60%.

This project focuses on two major areas in Python: <a href="#whatsnew311-faster-startup" class="reference internal"><span class="std std-ref">Faster Startup</span></a> and <a href="#whatsnew311-faster-runtime" class="reference internal"><span class="std std-ref">Faster Runtime</span></a>. Optimizations not covered by this project are listed separately under <a href="#whatsnew311-optimizations" class="reference internal"><span class="std std-ref">Optimizations</span></a>.

<div id="faster-startup" class="section">

<span id="whatsnew311-faster-startup"></span>

### Faster Startup<a href="#faster-startup" class="headerlink" title="Link to this heading">¶</a>

<div id="frozen-imports-static-code-objects" class="section">

<span id="whatsnew311-faster-imports"></span>

#### Frozen imports / Static code objects<a href="#frozen-imports-static-code-objects" class="headerlink" title="Link to this heading">¶</a>

Python caches <a href="../glossary.html#term-bytecode" class="reference internal"><span class="xref std std-term">bytecode</span></a> in the <a href="../tutorial/modules.html#tut-pycache" class="reference internal"><span class="std std-ref">__pycache__</span></a> directory to speed up module loading.

Previously in 3.10, Python module execution looked like this:

<div class="highlight-text notranslate">

<div class="highlight">

    Read __pycache__ -> Unmarshal -> Heap allocated code object -> Evaluate

</div>

</div>

In Python 3.11, the core modules essential for Python startup are “frozen”. This means that their <a href="../c-api/code.html#codeobjects" class="reference internal"><span class="std std-ref">Code Objects</span></a> (and bytecode) are statically allocated by the interpreter. This reduces the steps in module execution process to:

<div class="highlight-text notranslate">

<div class="highlight">

    Statically allocated code object -> Evaluate

</div>

</div>

Interpreter startup is now 10-15% faster in Python 3.11. This has a big impact for short-running programs using Python.

(Contributed by Eric Snow, Guido van Rossum and Kumar Aditya in many issues.)

</div>

</div>

<div id="faster-runtime" class="section">

<span id="whatsnew311-faster-runtime"></span>

### Faster Runtime<a href="#faster-runtime" class="headerlink" title="Link to this heading">¶</a>

<div id="cheaper-lazy-python-frames" class="section">

<span id="whatsnew311-lazy-python-frames"></span>

#### Cheaper, lazy Python frames<a href="#cheaper-lazy-python-frames" class="headerlink" title="Link to this heading">¶</a>

Python frames, holding execution information, are created whenever Python calls a Python function. The following are new frame optimizations:

- Streamlined the frame creation process.

- Avoided memory allocation by generously re-using frame space on the C stack.

- Streamlined the internal frame struct to contain only essential information. Frames previously held extra debugging and memory management information.

Old-style <a href="../reference/datamodel.html#frame-objects" class="reference internal"><span class="std std-ref">frame objects</span></a> are now created only when requested by debuggers or by Python introspection functions such as <a href="../library/sys.html#sys._getframe" class="reference internal" title="sys._getframe"><span class="pre"><code class="sourceCode python">sys._getframe()</code></span></a> and <a href="../library/inspect.html#inspect.currentframe" class="reference internal" title="inspect.currentframe"><span class="pre"><code class="sourceCode python">inspect.currentframe()</code></span></a>. For most user code, no frame objects are created at all. As a result, nearly all Python functions calls have sped up significantly. We measured a 3-7% speedup in pyperformance.

(Contributed by Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44590" class="reference external">bpo-44590</a>.)

</div>

<div id="inlined-python-function-calls" class="section">

<span id="whatsnew311-inline-calls"></span><span id="inline-calls"></span>

#### Inlined Python function calls<a href="#inlined-python-function-calls" class="headerlink" title="Link to this heading">¶</a>

During a Python function call, Python will call an evaluating C function to interpret that function’s code. This effectively limits pure Python recursion to what’s safe for the C stack.

In 3.11, when CPython detects Python code calling another Python function, it sets up a new frame, and “jumps” to the new code inside the new frame. This avoids calling the C interpreting function altogether.

Most Python function calls now consume no C stack space, speeding them up. In simple recursive functions like fibonacci or factorial, we observed a 1.7x speedup. This also means recursive functions can recurse significantly deeper (if the user increases the recursion limit with <a href="../library/sys.html#sys.setrecursionlimit" class="reference internal" title="sys.setrecursionlimit"><span class="pre"><code class="sourceCode python">sys.setrecursionlimit()</code></span></a>). We measured a 1-3% improvement in pyperformance.

(Contributed by Pablo Galindo and Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45256" class="reference external">bpo-45256</a>.)

</div>

<div id="pep-659-specializing-adaptive-interpreter" class="section">

<span id="whatsnew311-pep659"></span>

#### PEP 659: Specializing Adaptive Interpreter<a href="#pep-659-specializing-adaptive-interpreter" class="headerlink" title="Link to this heading">¶</a>

<span id="index-29" class="target"></span><a href="https://peps.python.org/pep-0659/" class="pep reference external"><strong>PEP 659</strong></a> is one of the key parts of the Faster CPython project. The general idea is that while Python is a dynamic language, most code has regions where objects and types rarely change. This concept is known as *type stability*.

At runtime, Python will try to look for common patterns and type stability in the executing code. Python will then replace the current operation with a more specialized one. This specialized operation uses fast paths available only to those use cases/types, which generally outperform their generic counterparts. This also brings in another concept called *inline caching*, where Python caches the results of expensive operations directly in the <a href="../glossary.html#term-bytecode" class="reference internal"><span class="xref std std-term">bytecode</span></a>.

The specializer will also combine certain common instruction pairs into one superinstruction, reducing the overhead during execution.

Python will only specialize when it sees code that is “hot” (executed multiple times). This prevents Python from wasting time on run-once code. Python can also de-specialize when code is too dynamic or when the use changes. Specialization is attempted periodically, and specialization attempts are not too expensive, allowing specialization to adapt to new circumstances.

(PEP written by Mark Shannon, with ideas inspired by Stefan Brunthaler. See <span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0659/" class="pep reference external"><strong>PEP 659</strong></a> for more information. Implementation by Mark Shannon and Brandt Bucher, with additional help from Irit Katriel and Dennis Sweeney.)

<table class="docutils align-default">
<colgroup>
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
</colgroup>
<thead>
<tr class="row-odd">
<th class="head"><p>Operation</p></th>
<th class="head"><p>Form</p></th>
<th class="head"><p>Specialization</p></th>
<th class="head"><p>Operation speedup (up to)</p></th>
<th class="head"><p>Contributor(s)</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even">
<td><p>Binary operations</p></td>
<td><p><span class="pre"><code class="docutils literal notranslate">x</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">+</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">x</code></span></p>
<p><span class="pre"><code class="docutils literal notranslate">x</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">-</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">x</code></span></p>
<p><span class="pre"><code class="docutils literal notranslate">x</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">*</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">x</code></span></p></td>
<td><p>Binary add, multiply and subtract for common types such as <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>, <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> and <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> take custom fast paths for their underlying types.</p></td>
<td><p>10%</p></td>
<td><p>Mark Shannon, Donghee Na, Brandt Bucher, Dennis Sweeney</p></td>
</tr>
<tr class="row-odd">
<td><p>Subscript</p></td>
<td><p><span class="pre"><code class="docutils literal notranslate">a[i]</code></span></p></td>
<td><p>Subscripting container types such as <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a>, <a href="../library/stdtypes.html#tuple" class="reference internal" title="tuple"><span class="pre"><code class="sourceCode python"><span class="bu">tuple</span></code></span></a> and <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> directly index the underlying data structures.</p>
<p>Subscripting custom <a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a> is also inlined similar to <a href="#inline-calls" class="reference internal"><span class="std std-ref">Inlined Python function calls</span></a>.</p></td>
<td><p>10-25%</p></td>
<td><p>Irit Katriel, Mark Shannon</p></td>
</tr>
<tr class="row-even">
<td><p>Store subscript</p></td>
<td><p><span class="pre"><code class="docutils literal notranslate">a[i]</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">=</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">z</code></span></p></td>
<td><p>Similar to subscripting specialization above.</p></td>
<td><p>10-25%</p></td>
<td><p>Dennis Sweeney</p></td>
</tr>
<tr class="row-odd">
<td><p>Calls</p></td>
<td><p><span class="pre"><code class="docutils literal notranslate">f(arg)</code></span></p>
<p><span class="pre"><code class="docutils literal notranslate">C(arg)</code></span></p></td>
<td><p>Calls to common builtin (C) functions and types such as <a href="../library/functions.html#len" class="reference internal" title="len"><span class="pre"><code class="sourceCode python"><span class="bu">len</span>()</code></span></a> and <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> directly call their underlying C version. This avoids going through the internal calling convention.</p></td>
<td><p>20%</p></td>
<td><p>Mark Shannon, Ken Jin</p></td>
</tr>
<tr class="row-even">
<td><p>Load global variable</p></td>
<td><p><span class="pre"><code class="docutils literal notranslate">print</code></span></p>
<p><span class="pre"><code class="docutils literal notranslate">len</code></span></p></td>
<td><p>The object’s index in the globals/builtins namespace is cached. Loading globals and builtins require zero namespace lookups.</p></td>
<td><p><a href="#fn1" class="footnote-ref" id="fnref1" role="doc-noteref"><sup>1</sup></a></p></td>
<td><p>Mark Shannon</p></td>
</tr>
<tr class="row-odd">
<td><p>Load attribute</p></td>
<td><p><span class="pre"><code class="docutils literal notranslate">o.attr</code></span></p></td>
<td><p>Similar to loading global variables. The attribute’s index inside the class/object’s namespace is cached. In most cases, attribute loading will require zero namespace lookups.</p></td>
<td><p><a href="#fn2" class="footnote-ref" id="fnref2" role="doc-noteref"><sup>2</sup></a></p></td>
<td><p>Mark Shannon</p></td>
</tr>
<tr class="row-even">
<td><p>Load methods for call</p></td>
<td><p><span class="pre"><code class="docutils literal notranslate">o.meth()</code></span></p></td>
<td><p>The actual address of the method is cached. Method loading now has no namespace lookups – even for classes with long inheritance chains.</p></td>
<td><p>10-20%</p></td>
<td><p>Ken Jin, Mark Shannon</p></td>
</tr>
<tr class="row-odd">
<td><p>Store attribute</p></td>
<td><p><span class="pre"><code class="docutils literal notranslate">o.attr</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">=</code></span><code class="docutils literal notranslate"> </code><span class="pre"><code class="docutils literal notranslate">z</code></span></p></td>
<td><p>Similar to load attribute optimization.</p></td>
<td><p>2% in pyperformance</p></td>
<td><p>Mark Shannon</p></td>
</tr>
<tr class="row-even">
<td><p>Unpack Sequence</p></td>
<td><p><span class="pre"><code class="docutils literal notranslate">*seq</code></span></p></td>
<td><p>Specialized for common containers such as <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a> and <a href="../library/stdtypes.html#tuple" class="reference internal" title="tuple"><span class="pre"><code class="sourceCode python"><span class="bu">tuple</span></code></span></a>. Avoids internal calling convention.</p></td>
<td><p>8%</p></td>
<td><p>Brandt Bucher</p></td>
</tr>
</tbody>
</table>
<section id="footnotes" class="footnotes footnotes-end-of-document" role="doc-endnotes">
<hr />
<ol>
<li id="fn1"></li>
<li id="fn2"></li>
</ol>
</section>

<span class="label"><span class="fn-bracket">\[</span><a href="#id2" role="doc-backlink">1</a><span class="fn-bracket">\]</span></span>

A similar optimization already existed since Python 3.8. 3.11 specializes for more forms and reduces some overhead.

<span class="label"><span class="fn-bracket">\[</span><a href="#id3" role="doc-backlink">2</a><span class="fn-bracket">\]</span></span>

A similar optimization already existed since Python 3.10. 3.11 specializes for more forms. Furthermore, all attribute loads should be sped up by <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45947" class="reference external">bpo-45947</a>.

</div>

</div>

<div id="misc" class="section">

<span id="whatsnew311-faster-cpython-misc"></span>

### Misc<a href="#misc" class="headerlink" title="Link to this heading">¶</a>

- Objects now require less memory due to lazily created object namespaces. Their namespace dictionaries now also share keys more freely. (Contributed Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45340" class="reference external">bpo-45340</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40116" class="reference external">bpo-40116</a>.)

- “Zero-cost” exceptions are implemented, eliminating the cost of <a href="../reference/compound_stmts.html#try" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">try</code></span></a> statements when no exception is raised. (Contributed by Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40222" class="reference external">bpo-40222</a>.)

- A more concise representation of exceptions in the interpreter reduced the time required for catching an exception by about 10%. (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45711" class="reference external">bpo-45711</a>.)

- <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a>’s regular expression matching engine has been partially refactored, and now uses computed gotos (or “threaded code”) on supported platforms. As a result, Python 3.11 executes the <a href="https://pyperformance.readthedocs.io/benchmarks.html#regex-dna" class="reference external">pyperformance regular expression benchmarks</a> up to 10% faster than Python 3.10. (Contributed by Brandt Bucher in <a href="https://github.com/python/cpython/issues/91404" class="reference external">gh-91404</a>.)

</div>

<div id="faq" class="section">

<span id="whatsnew311-faster-cpython-faq"></span>

### FAQ<a href="#faq" class="headerlink" title="Link to this heading">¶</a>

<div id="how-should-i-write-my-code-to-utilize-these-speedups" class="section">

<span id="faster-cpython-faq-my-code"></span>

#### How should I write my code to utilize these speedups?<a href="#how-should-i-write-my-code-to-utilize-these-speedups" class="headerlink" title="Link to this heading">¶</a>

Write Pythonic code that follows common best practices; you don’t have to change your code. The Faster CPython project optimizes for common code patterns we observe.

</div>

<div id="will-cpython-3-11-use-more-memory" class="section">

<span id="faster-cpython-faq-memory"></span>

#### Will CPython 3.11 use more memory?<a href="#will-cpython-3-11-use-more-memory" class="headerlink" title="Link to this heading">¶</a>

Maybe not; we don’t expect memory use to exceed 20% higher than 3.10. This is offset by memory optimizations for frame objects and object dictionaries as mentioned above.

</div>

<div id="i-don-t-see-any-speedups-in-my-workload-why" class="section">

<span id="faster-cpython-ymmv"></span>

#### I don’t see any speedups in my workload. Why?<a href="#i-don-t-see-any-speedups-in-my-workload-why" class="headerlink" title="Link to this heading">¶</a>

Certain code won’t have noticeable benefits. If your code spends most of its time on I/O operations, or already does most of its computation in a C extension library like NumPy, there won’t be significant speedups. This project currently benefits pure-Python workloads the most.

Furthermore, the pyperformance figures are a geometric mean. Even within the pyperformance benchmarks, certain benchmarks have slowed down slightly, while others have sped up by nearly 2x!

</div>

<div id="is-there-a-jit-compiler" class="section">

<span id="faster-cpython-jit"></span>

#### Is there a JIT compiler?<a href="#is-there-a-jit-compiler" class="headerlink" title="Link to this heading">¶</a>

No. We’re still exploring other optimizations.

</div>

</div>

<div id="about" class="section">

<span id="whatsnew311-faster-cpython-about"></span>

### About<a href="#about" class="headerlink" title="Link to this heading">¶</a>

Faster CPython explores optimizations for <a href="../glossary.html#term-CPython" class="reference internal"><span class="xref std std-term">CPython</span></a>. The main team is funded by Microsoft to work on this full-time. Pablo Galindo Salgado is also funded by Bloomberg LP to work on the project part-time. Finally, many contributors are volunteers from the community.

</div>

</div>

<div id="cpython-bytecode-changes" class="section">

<span id="whatsnew311-bytecode-changes"></span>

## CPython bytecode changes<a href="#cpython-bytecode-changes" class="headerlink" title="Link to this heading">¶</a>

The bytecode now contains inline cache entries, which take the form of the newly-added <a href="../library/dis.html#opcode-CACHE" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">CACHE</code></span></a> instructions. Many opcodes expect to be followed by an exact number of caches, and instruct the interpreter to skip over them at runtime. Populated caches can look like arbitrary instructions, so great care should be taken when reading or modifying raw, adaptive bytecode containing quickened data.

<div id="new-opcodes" class="section">

<span id="whatsnew311-added-opcodes"></span>

### New opcodes<a href="#new-opcodes" class="headerlink" title="Link to this heading">¶</a>

- <span class="pre">`ASYNC_GEN_WRAP`</span>, <a href="../library/dis.html#opcode-RETURN_GENERATOR" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">RETURN_GENERATOR</code></span></a> and <a href="../library/dis.html#opcode-SEND" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">SEND</code></span></a>, used in generators and co-routines.

- <a href="../library/dis.html#opcode-COPY_FREE_VARS" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">COPY_FREE_VARS</code></span></a>, which avoids needing special caller-side code for closures.

- <a href="../library/dis.html#opcode-JUMP_BACKWARD_NO_INTERRUPT" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">JUMP_BACKWARD_NO_INTERRUPT</code></span></a>, for use in certain loops where handling interrupts is undesirable.

- <a href="../library/dis.html#opcode-MAKE_CELL" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">MAKE_CELL</code></span></a>, to create <a href="../c-api/cell.html#cell-objects" class="reference internal"><span class="std std-ref">Cell Objects</span></a>.

- <a href="../library/dis.html#opcode-CHECK_EG_MATCH" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">CHECK_EG_MATCH</code></span></a> and <span class="pre">`PREP_RERAISE_STAR`</span>, to handle the <a href="#whatsnew311-pep654" class="reference internal"><span class="std std-ref">new exception groups and except*</span></a> added in <span id="index-31" class="target"></span><a href="https://peps.python.org/pep-0654/" class="pep reference external"><strong>PEP 654</strong></a>.

- <a href="../library/dis.html#opcode-PUSH_EXC_INFO" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">PUSH_EXC_INFO</code></span></a>, for use in exception handlers.

- <a href="../library/dis.html#opcode-RESUME" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">RESUME</code></span></a>, a no-op, for internal tracing, debugging and optimization checks.

</div>

<div id="replaced-opcodes" class="section">

<span id="whatsnew311-replaced-opcodes"></span>

### Replaced opcodes<a href="#replaced-opcodes" class="headerlink" title="Link to this heading">¶</a>

<table class="docutils align-default">
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr class="row-odd">
<th class="head"><p>Replaced Opcode(s)</p></th>
<th class="head"><p>New Opcode(s)</p></th>
<th class="head"><p>Notes</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even">
<td><div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">BINARY_*</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">INPLACE_*</code></span>
</div></td>
<td><p><a href="../library/dis.html#opcode-BINARY_OP" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">BINARY_OP</code></span></a></p></td>
<td><p>Replaced all numeric binary/in-place opcodes with a single opcode</p></td>
</tr>
<tr class="row-odd">
<td><div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">CALL_FUNCTION</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">CALL_FUNCTION_KW</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">CALL_METHOD</code></span>
</div></td>
<td><div class="line">
<a href="../library/dis.html#opcode-CALL" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">CALL</code></span></a>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">KW_NAMES</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">PRECALL</code></span>
</div>
<div class="line">
<a href="../library/dis.html#opcode-PUSH_NULL" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">PUSH_NULL</code></span></a>
</div></td>
<td><p>Decouples argument shifting for methods from handling of keyword arguments; allows better specialization of calls</p></td>
</tr>
<tr class="row-even">
<td><div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">DUP_TOP</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">DUP_TOP_TWO</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">ROT_TWO</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">ROT_THREE</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">ROT_FOUR</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">ROT_N</code></span>
</div></td>
<td><div class="line">
<a href="../library/dis.html#opcode-COPY" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">COPY</code></span></a>
</div>
<div class="line">
<a href="../library/dis.html#opcode-SWAP" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">SWAP</code></span></a>
</div></td>
<td><p>Stack manipulation instructions</p></td>
</tr>
<tr class="row-odd">
<td><div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">JUMP_IF_NOT_EXC_MATCH</code></span>
</div></td>
<td><div class="line">
<a href="../library/dis.html#opcode-CHECK_EXC_MATCH" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">CHECK_EXC_MATCH</code></span></a>
</div></td>
<td><p>Now performs check but doesn’t jump</p></td>
</tr>
<tr class="row-even">
<td><div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">JUMP_ABSOLUTE</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">POP_JUMP_IF_FALSE</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">POP_JUMP_IF_TRUE</code></span>
</div></td>
<td><div class="line">
<a href="../library/dis.html#opcode-JUMP_BACKWARD" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">JUMP_BACKWARD</code></span></a>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">POP_JUMP_BACKWARD_IF_*</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">POP_JUMP_FORWARD_IF_*</code></span>
</div></td>
<td><p>See <a href="#fn1" class="footnote-ref" id="fnref1" role="doc-noteref"><sup>1</sup></a>; <span class="pre"><code class="docutils literal notranslate">TRUE</code></span>, <span class="pre"><code class="docutils literal notranslate">FALSE</code></span>, <span class="pre"><code class="docutils literal notranslate">NONE</code></span> and <span class="pre"><code class="docutils literal notranslate">NOT_NONE</code></span> variants for each direction</p></td>
</tr>
<tr class="row-odd">
<td><div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">SETUP_WITH</code></span>
</div>
<div class="line">
<span class="pre"><code class="xref std std-opcode docutils literal notranslate">SETUP_ASYNC_WITH</code></span>
</div></td>
<td><p><a href="../library/dis.html#opcode-BEFORE_WITH" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">BEFORE_WITH</code></span></a></p></td>
<td><p><a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> block setup</p></td>
</tr>
</tbody>
</table>
<section id="footnotes" class="footnotes footnotes-end-of-document" role="doc-endnotes">
<hr />
<ol>
<li id="fn1"></li>
</ol>
</section>

<span class="label"><span class="fn-bracket">\[</span><a href="#id4" role="doc-backlink">3</a><span class="fn-bracket">\]</span></span>

All jump opcodes are now relative, including the existing <span class="pre">`JUMP_IF_TRUE_OR_POP`</span> and <span class="pre">`JUMP_IF_FALSE_OR_POP`</span>. The argument is now an offset from the current instruction rather than an absolute location.

</div>

<div id="changed-removed-opcodes" class="section">

<span id="whatsnew311-changed-removed-opcodes"></span><span id="whatsnew311-removed-opcodes"></span><span id="whatsnew311-changed-opcodes"></span>

### Changed/removed opcodes<a href="#changed-removed-opcodes" class="headerlink" title="Link to this heading">¶</a>

- Changed <a href="../library/dis.html#opcode-MATCH_CLASS" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">MATCH_CLASS</code></span></a> and <a href="../library/dis.html#opcode-MATCH_KEYS" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">MATCH_KEYS</code></span></a> to no longer push an additional boolean value to indicate success/failure. Instead, <span class="pre">`None`</span> is pushed on failure in place of the tuple of extracted values.

- Changed opcodes that work with exceptions to reflect them now being represented as one item on the stack instead of three (see <a href="https://github.com/python/cpython/issues/89874" class="reference external">gh-89874</a>).

- Removed <span class="pre">`COPY_DICT_WITHOUT_KEYS`</span>, <span class="pre">`GEN_START`</span>, <span class="pre">`POP_BLOCK`</span>, <span class="pre">`SETUP_FINALLY`</span> and <span class="pre">`YIELD_FROM`</span>.

</div>

</div>

<div id="deprecated" class="section">

<span id="whatsnew311-python-api-deprecated"></span><span id="whatsnew311-deprecated"></span>

## Deprecated<a href="#deprecated" class="headerlink" title="Link to this heading">¶</a>

This section lists Python APIs that have been deprecated in Python 3.11.

Deprecated C APIs are <a href="#whatsnew311-c-api-deprecated" class="reference internal"><span class="std std-ref">listed separately</span></a>.

<div id="language-builtins" class="section">

<span id="whatsnew311-deprecated-builtins"></span><span id="whatsnew311-deprecated-language"></span>

### Language/Builtins<a href="#language-builtins" class="headerlink" title="Link to this heading">¶</a>

- Chaining <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span></code></span></a> descriptors (introduced in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19072" class="reference external">bpo-19072</a>) is now deprecated. It can no longer be used to wrap other descriptors such as <a href="../library/functions.html#property" class="reference internal" title="property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span></code></span></a>. The core design of this feature was flawed and caused a number of downstream problems. To “pass-through” a <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span></code></span></a>, consider using the <span class="pre">`__wrapped__`</span> attribute that was added in Python 3.10. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/89519" class="reference external">gh-89519</a>.)

- Octal escapes in string and bytes literals with values larger than <span class="pre">`0o377`</span> (255 in decimal) now produce a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. In a future Python version, they will raise a <a href="../library/exceptions.html#SyntaxWarning" class="reference internal" title="SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a> and eventually a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/81548" class="reference external">gh-81548</a>.)

- The delegation of <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a> to <a href="../reference/datamodel.html#object.__trunc__" class="reference internal" title="object.__trunc__"><span class="pre"><code class="sourceCode python"><span class="fu">__trunc__</span>()</code></span></a> is now deprecated. Calling <span class="pre">`int(a)`</span> when <span class="pre">`type(a)`</span> implements <span class="pre">`__trunc__()`</span> but not <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="fu">__int__</span>()</code></span></a> or <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> now raises a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44977" class="reference external">bpo-44977</a>.)

</div>

<div id="modules" class="section">

<span id="whatsnew311-deprecated-modules"></span>

### Modules<a href="#modules" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-32" class="target"></span><a href="https://peps.python.org/pep-0594/" class="pep reference external"><strong>PEP 594</strong></a> led to the deprecations of the following modules slated for removal in Python 3.13:

  |  |  |  |  |  |
  |----|----|----|----|----|
  | <span class="pre">`aifc`</span> | <span class="pre">`chunk`</span> | <span class="pre">`msilib`</span> | <span class="pre">`pipes`</span> | <span class="pre">`telnetlib`</span> |
  | <span class="pre">`audioop`</span> | <span class="pre">`crypt`</span> | <span class="pre">`nis`</span> | <span class="pre">`sndhdr`</span> | <span class="pre">`uu`</span> |
  | <span class="pre">`cgi`</span> | <span class="pre">`imghdr`</span> | <span class="pre">`nntplib`</span> | <span class="pre">`spwd`</span> | <span class="pre">`xdrlib`</span> |
  | <span class="pre">`cgitb`</span> | <span class="pre">`mailcap`</span> | <span class="pre">`ossaudiodev`</span> | <span class="pre">`sunau`</span> |  |

  (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=47061" class="reference external">bpo-47061</a> and Victor Stinner in <a href="https://github.com/python/cpython/issues/68966" class="reference external">gh-68966</a>.)

- The <span class="pre">`asynchat`</span>, <span class="pre">`asyncore`</span> and <span class="pre">`smtpd`</span> modules have been deprecated since at least Python 3.6. Their documentation and deprecation warnings have now been updated to note they will be removed in Python 3.12. (Contributed by Hugo van Kemenade in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=47022" class="reference external">bpo-47022</a>.)

- The <span class="pre">`lib2to3`</span> package and <span class="pre">`2to3`</span> tool are now deprecated and may not be able to parse Python 3.10 or newer. See <span id="index-33" class="target"></span><a href="https://peps.python.org/pep-0617/" class="pep reference external"><strong>PEP 617</strong></a>, introducing the new PEG parser, for details. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40360" class="reference external">bpo-40360</a>.)

- Undocumented modules <span class="pre">`sre_compile`</span>, <span class="pre">`sre_constants`</span> and <span class="pre">`sre_parse`</span> are now deprecated. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=47152" class="reference external">bpo-47152</a>.)

</div>

<div id="standard-library" class="section">

<span id="whatsnew311-deprecated-stdlib"></span>

### Standard Library<a href="#standard-library" class="headerlink" title="Link to this heading">¶</a>

- The following have been deprecated in <a href="../library/configparser.html#module-configparser" class="reference internal" title="configparser: Configuration file parser."><span class="pre"><code class="sourceCode python">configparser</code></span></a> since Python 3.2. Their deprecation warnings have now been updated to note they will be removed in Python 3.12:

  - the <span class="pre">`configparser.SafeConfigParser`</span> class

  - the <span class="pre">`configparser.ParsingError.filename`</span> property

  - the <span class="pre">`configparser.RawConfigParser.readfp()`</span> method

  (Contributed by Hugo van Kemenade in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45173" class="reference external">bpo-45173</a>.)

- <span class="pre">`configparser.LegacyInterpolation`</span> has been deprecated in the docstring since Python 3.2, and is not listed in the <a href="../library/configparser.html#module-configparser" class="reference internal" title="configparser: Configuration file parser."><span class="pre"><code class="sourceCode python">configparser</code></span></a> documentation. It now emits a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> and will be removed in Python 3.13. Use <a href="../library/configparser.html#configparser.BasicInterpolation" class="reference internal" title="configparser.BasicInterpolation"><span class="pre"><code class="sourceCode python">configparser.BasicInterpolation</code></span></a> or <a href="../library/configparser.html#configparser.ExtendedInterpolation" class="reference internal" title="configparser.ExtendedInterpolation"><span class="pre"><code class="sourceCode python">configparser.ExtendedInterpolation</code></span></a> instead. (Contributed by Hugo van Kemenade in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46607" class="reference external">bpo-46607</a>.)

- The older set of <a href="../library/importlib.resources.html#module-importlib.resources" class="reference internal" title="importlib.resources: Package resource reading, opening, and access"><span class="pre"><code class="sourceCode python">importlib.resources</code></span></a> functions were deprecated in favor of the replacements added in Python 3.9 and will be removed in a future Python version, due to not supporting resources located within package subdirectories:

  - <span class="pre">`importlib.resources.contents()`</span>

  - <span class="pre">`importlib.resources.is_resource()`</span>

  - <span class="pre">`importlib.resources.open_binary()`</span>

  - <span class="pre">`importlib.resources.open_text()`</span>

  - <span class="pre">`importlib.resources.read_binary()`</span>

  - <span class="pre">`importlib.resources.read_text()`</span>

  - <span class="pre">`importlib.resources.path()`</span>

- The <a href="../library/locale.html#locale.getdefaultlocale" class="reference internal" title="locale.getdefaultlocale"><span class="pre"><code class="sourceCode python">locale.getdefaultlocale()</code></span></a> function is deprecated and will be removed in Python 3.15. Use <a href="../library/locale.html#locale.setlocale" class="reference internal" title="locale.setlocale"><span class="pre"><code class="sourceCode python">locale.setlocale()</code></span></a>, <a href="../library/locale.html#locale.getpreferredencoding" class="reference internal" title="locale.getpreferredencoding"><span class="pre"><code class="sourceCode python">locale.getpreferredencoding(<span class="va">False</span>)</code></span></a> and <a href="../library/locale.html#locale.getlocale" class="reference internal" title="locale.getlocale"><span class="pre"><code class="sourceCode python">locale.getlocale()</code></span></a> functions instead. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/90817" class="reference external">gh-90817</a>.)

- The <span class="pre">`locale.resetlocale()`</span> function is deprecated and will be removed in Python 3.13. Use <span class="pre">`locale.setlocale(locale.LC_ALL,`</span>` `<span class="pre">`"")`</span> instead. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/90817" class="reference external">gh-90817</a>.)

- Stricter rules will now be applied for numerical group references and group names in <a href="../library/re.html#re-syntax" class="reference internal"><span class="std std-ref">regular expressions</span></a>. Only sequences of ASCII digits will now be accepted as a numerical reference, and the group name in <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> patterns and replacement strings can only contain ASCII letters, digits and underscores. For now, a deprecation warning is raised for syntax violating these rules. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/91760" class="reference external">gh-91760</a>.)

- In the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module, the <span class="pre">`re.template()`</span> function and the corresponding <span class="pre">`re.TEMPLATE`</span> and <span class="pre">`re.T`</span> flags are deprecated, as they were undocumented and lacked an obvious purpose. They will be removed in Python 3.13. (Contributed by Serhiy Storchaka and Miro Hrončok in <a href="https://github.com/python/cpython/issues/92728" class="reference external">gh-92728</a>.)

- <span class="pre">`turtle.settiltangle()`</span> has been deprecated since Python 3.1; it now emits a deprecation warning and will be removed in Python 3.13. Use <a href="../library/turtle.html#turtle.tiltangle" class="reference internal" title="turtle.tiltangle"><span class="pre"><code class="sourceCode python">turtle.tiltangle()</code></span></a> instead (it was earlier incorrectly marked as deprecated, and its docstring is now corrected). (Contributed by Hugo van Kemenade in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45837" class="reference external">bpo-45837</a>.)

- <a href="../library/typing.html#typing.Text" class="reference internal" title="typing.Text"><span class="pre"><code class="sourceCode python">typing.Text</code></span></a>, which exists solely to provide compatibility support between Python 2 and Python 3 code, is now deprecated. Its removal is currently unplanned, but users are encouraged to use <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> instead wherever possible. (Contributed by Alex Waygood in <a href="https://github.com/python/cpython/issues/92332" class="reference external">gh-92332</a>.)

- The keyword argument syntax for constructing <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">typing.TypedDict</code></span></a> types is now deprecated. Support will be removed in Python 3.13. (Contributed by Jingchen Ye in <a href="https://github.com/python/cpython/issues/90224" class="reference external">gh-90224</a>.)

- <span class="pre">`webbrowser.MacOSX`</span> is deprecated and will be removed in Python 3.13. It is untested, undocumented, and not used by <a href="../library/webbrowser.html#module-webbrowser" class="reference internal" title="webbrowser: Easy-to-use controller for web browsers."><span class="pre"><code class="sourceCode python">webbrowser</code></span></a> itself. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42255" class="reference external">bpo-42255</a>.)

- The behavior of returning a value from a <a href="../library/unittest.html#unittest.TestCase" class="reference internal" title="unittest.TestCase"><span class="pre"><code class="sourceCode python">TestCase</code></span></a> and <a href="../library/unittest.html#unittest.IsolatedAsyncioTestCase" class="reference internal" title="unittest.IsolatedAsyncioTestCase"><span class="pre"><code class="sourceCode python">IsolatedAsyncioTestCase</code></span></a> test methods (other than the default <span class="pre">`None`</span> value) is now deprecated.

- Deprecated the following not-formally-documented <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> functions, scheduled for removal in Python 3.13:

  - <span class="pre">`unittest.findTestCases()`</span>

  - <span class="pre">`unittest.makeSuite()`</span>

  - <span class="pre">`unittest.getTestCaseNames()`</span>

  Use <a href="../library/unittest.html#unittest.TestLoader" class="reference internal" title="unittest.TestLoader"><span class="pre"><code class="sourceCode python">TestLoader</code></span></a> methods instead:

  - <a href="../library/unittest.html#unittest.TestLoader.loadTestsFromModule" class="reference internal" title="unittest.TestLoader.loadTestsFromModule"><span class="pre"><code class="sourceCode python">unittest.TestLoader.loadTestsFromModule()</code></span></a>

  - <a href="../library/unittest.html#unittest.TestLoader.loadTestsFromTestCase" class="reference internal" title="unittest.TestLoader.loadTestsFromTestCase"><span class="pre"><code class="sourceCode python">unittest.TestLoader.loadTestsFromTestCase()</code></span></a>

  - <a href="../library/unittest.html#unittest.TestLoader.getTestCaseNames" class="reference internal" title="unittest.TestLoader.getTestCaseNames"><span class="pre"><code class="sourceCode python">unittest.TestLoader.getTestCaseNames()</code></span></a>

  (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5846" class="reference external">bpo-5846</a>.)

- <span class="pre">`unittest.TestProgram.usageExit()`</span> is marked deprecated, to be removed in 3.13. (Contributed by Carlos Damázio in <a href="https://github.com/python/cpython/issues/67048" class="reference external">gh-67048</a>.)

</div>

</div>

<div id="pending-removal-in-python-3-12" class="section">

<span id="whatsnew311-python-api-pending-removal"></span><span id="whatsnew311-pending-removal"></span>

## Pending Removal in Python 3.12<a href="#pending-removal-in-python-3-12" class="headerlink" title="Link to this heading">¶</a>

The following Python APIs have been deprecated in earlier Python releases, and will be removed in Python 3.12.

C APIs pending removal are <a href="#whatsnew311-c-api-pending-removal" class="reference internal"><span class="std std-ref">listed separately</span></a>.

- The <span class="pre">`asynchat`</span> module

- The <span class="pre">`asyncore`</span> module

- The <a href="3.10.html#distutils-deprecated" class="reference internal"><span class="std std-ref">entire distutils package</span></a>

- The <span class="pre">`imp`</span> module

- The <a href="../library/typing.html#typing.IO" class="reference internal" title="typing.IO"><span class="pre"><code class="sourceCode python">typing.io</code></span></a> namespace

- The <a href="../library/typing.html#typing.Pattern" class="reference internal" title="typing.Pattern"><span class="pre"><code class="sourceCode python">typing.re</code></span></a> namespace

- <span class="pre">`cgi.log()`</span>

- <span class="pre">`importlib.find_loader()`</span>

- <span class="pre">`importlib.abc.Loader.module_repr()`</span>

- <span class="pre">`importlib.abc.MetaPathFinder.find_module()`</span>

- <span class="pre">`importlib.abc.PathEntryFinder.find_loader()`</span>

- <span class="pre">`importlib.abc.PathEntryFinder.find_module()`</span>

- <span class="pre">`importlib.machinery.BuiltinImporter.find_module()`</span>

- <span class="pre">`importlib.machinery.BuiltinLoader.module_repr()`</span>

- <span class="pre">`importlib.machinery.FileFinder.find_loader()`</span>

- <span class="pre">`importlib.machinery.FileFinder.find_module()`</span>

- <span class="pre">`importlib.machinery.FrozenImporter.find_module()`</span>

- <span class="pre">`importlib.machinery.FrozenLoader.module_repr()`</span>

- <span class="pre">`importlib.machinery.PathFinder.find_module()`</span>

- <span class="pre">`importlib.machinery.WindowsRegistryFinder.find_module()`</span>

- <span class="pre">`importlib.util.module_for_loader()`</span>

- <span class="pre">`importlib.util.set_loader_wrapper()`</span>

- <span class="pre">`importlib.util.set_package_wrapper()`</span>

- <span class="pre">`pkgutil.ImpImporter`</span>

- <span class="pre">`pkgutil.ImpLoader`</span>

- <span class="pre">`pathlib.Path.link_to()`</span>

- <span class="pre">`sqlite3.enable_shared_cache()`</span>

- <span class="pre">`sqlite3.OptimizedUnicode()`</span>

- <span class="pre">`PYTHONTHREADDEBUG`</span> environment variable

- The following deprecated aliases in <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a>:

  > <div>
  >
  > | Deprecated alias | Method Name | Deprecated in |
  > |----|----|----|
  > | <span class="pre">`failUnless`</span> | <a href="../library/unittest.html#unittest.TestCase.assertTrue" class="reference internal" title="unittest.TestCase.assertTrue"><span class="pre"><code class="sourceCode python">assertTrue()</code></span></a> | 3.1 |
  > | <span class="pre">`failIf`</span> | <a href="../library/unittest.html#unittest.TestCase.assertFalse" class="reference internal" title="unittest.TestCase.assertFalse"><span class="pre"><code class="sourceCode python">assertFalse()</code></span></a> | 3.1 |
  > | <span class="pre">`failUnlessEqual`</span> | <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a> | 3.1 |
  > | <span class="pre">`failIfEqual`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotEqual" class="reference internal" title="unittest.TestCase.assertNotEqual"><span class="pre"><code class="sourceCode python">assertNotEqual()</code></span></a> | 3.1 |
  > | <span class="pre">`failUnlessAlmostEqual`</span> | <a href="../library/unittest.html#unittest.TestCase.assertAlmostEqual" class="reference internal" title="unittest.TestCase.assertAlmostEqual"><span class="pre"><code class="sourceCode python">assertAlmostEqual()</code></span></a> | 3.1 |
  > | <span class="pre">`failIfAlmostEqual`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotAlmostEqual" class="reference internal" title="unittest.TestCase.assertNotAlmostEqual"><span class="pre"><code class="sourceCode python">assertNotAlmostEqual()</code></span></a> | 3.1 |
  > | <span class="pre">`failUnlessRaises`</span> | <a href="../library/unittest.html#unittest.TestCase.assertRaises" class="reference internal" title="unittest.TestCase.assertRaises"><span class="pre"><code class="sourceCode python">assertRaises()</code></span></a> | 3.1 |
  > | <span class="pre">`assert_`</span> | <a href="../library/unittest.html#unittest.TestCase.assertTrue" class="reference internal" title="unittest.TestCase.assertTrue"><span class="pre"><code class="sourceCode python">assertTrue()</code></span></a> | 3.2 |
  > | <span class="pre">`assertEquals`</span> | <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a> | 3.2 |
  > | <span class="pre">`assertNotEquals`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotEqual" class="reference internal" title="unittest.TestCase.assertNotEqual"><span class="pre"><code class="sourceCode python">assertNotEqual()</code></span></a> | 3.2 |
  > | <span class="pre">`assertAlmostEquals`</span> | <a href="../library/unittest.html#unittest.TestCase.assertAlmostEqual" class="reference internal" title="unittest.TestCase.assertAlmostEqual"><span class="pre"><code class="sourceCode python">assertAlmostEqual()</code></span></a> | 3.2 |
  > | <span class="pre">`assertNotAlmostEquals`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotAlmostEqual" class="reference internal" title="unittest.TestCase.assertNotAlmostEqual"><span class="pre"><code class="sourceCode python">assertNotAlmostEqual()</code></span></a> | 3.2 |
  > | <span class="pre">`assertRegexpMatches`</span> | <a href="../library/unittest.html#unittest.TestCase.assertRegex" class="reference internal" title="unittest.TestCase.assertRegex"><span class="pre"><code class="sourceCode python">assertRegex()</code></span></a> | 3.2 |
  > | <span class="pre">`assertRaisesRegexp`</span> | <a href="../library/unittest.html#unittest.TestCase.assertRaisesRegex" class="reference internal" title="unittest.TestCase.assertRaisesRegex"><span class="pre"><code class="sourceCode python">assertRaisesRegex()</code></span></a> | 3.2 |
  > | <span class="pre">`assertNotRegexpMatches`</span> | <a href="../library/unittest.html#unittest.TestCase.assertNotRegex" class="reference internal" title="unittest.TestCase.assertNotRegex"><span class="pre"><code class="sourceCode python">assertNotRegex()</code></span></a> | 3.5 |
  >
  > </div>

</div>

<div id="removed" class="section">

<span id="whatsnew311-python-api-removed"></span><span id="whatsnew311-removed"></span>

## Removed<a href="#removed" class="headerlink" title="Link to this heading">¶</a>

This section lists Python APIs that have been removed in Python 3.11.

Removed C APIs are <a href="#whatsnew311-c-api-removed" class="reference internal"><span class="std std-ref">listed separately</span></a>.

- Removed the <span class="pre">`@asyncio.coroutine()`</span> <a href="../glossary.html#term-decorator" class="reference internal"><span class="xref std std-term">decorator</span></a> enabling legacy generator-based coroutines to be compatible with <a href="../reference/compound_stmts.html#async" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span></a> / <a href="../reference/expressions.html#await" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">await</code></span></a> code. The function has been deprecated since Python 3.8 and the removal was initially scheduled for Python 3.10. Use <a href="../reference/compound_stmts.html#async-def" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">def</code></span></a> instead. (Contributed by Illia Volochii in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43216" class="reference external">bpo-43216</a>.)

- Removed <span class="pre">`asyncio.coroutines.CoroWrapper`</span> used for wrapping legacy generator-based coroutine objects in the debug mode. (Contributed by Illia Volochii in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43216" class="reference external">bpo-43216</a>.)

- Due to significant security concerns, the *reuse_address* parameter of <a href="../library/asyncio-eventloop.html#asyncio.loop.create_datagram_endpoint" class="reference internal" title="asyncio.loop.create_datagram_endpoint"><span class="pre"><code class="sourceCode python">asyncio.loop.create_datagram_endpoint()</code></span></a>, disabled in Python 3.9, is now entirely removed. This is because of the behavior of the socket option <span class="pre">`SO_REUSEADDR`</span> in UDP. (Contributed by Hugo van Kemenade in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45129" class="reference external">bpo-45129</a>.)

- Removed the <span class="pre">`binhex`</span> module, deprecated in Python 3.9. Also removed the related, similarly-deprecated <a href="../library/binascii.html#module-binascii" class="reference internal" title="binascii: Tools for converting between binary and various ASCII-encoded binary representations."><span class="pre"><code class="sourceCode python">binascii</code></span></a> functions:

  - <span class="pre">`binascii.a2b_hqx()`</span>

  - <span class="pre">`binascii.b2a_hqx()`</span>

  - <span class="pre">`binascii.rlecode_hqx()`</span>

  - <span class="pre">`binascii.rldecode_hqx()`</span>

  The <a href="../library/binascii.html#binascii.crc_hqx" class="reference internal" title="binascii.crc_hqx"><span class="pre"><code class="sourceCode python">binascii.crc_hqx()</code></span></a> function remains available.

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45085" class="reference external">bpo-45085</a>.)

- Removed the <span class="pre">`distutils`</span> <span class="pre">`bdist_msi`</span> command deprecated in Python 3.9. Use <span class="pre">`bdist_wheel`</span> (wheel packages) instead. (Contributed by Hugo van Kemenade in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45124" class="reference external">bpo-45124</a>.)

- Removed the <a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a> methods of <a href="../library/xml.dom.pulldom.html#xml.dom.pulldom.DOMEventStream" class="reference internal" title="xml.dom.pulldom.DOMEventStream"><span class="pre"><code class="sourceCode python">xml.dom.pulldom.DOMEventStream</code></span></a>, <a href="../library/wsgiref.html#wsgiref.util.FileWrapper" class="reference internal" title="wsgiref.util.FileWrapper"><span class="pre"><code class="sourceCode python">wsgiref.util.FileWrapper</code></span></a> and <a href="../library/fileinput.html#fileinput.FileInput" class="reference internal" title="fileinput.FileInput"><span class="pre"><code class="sourceCode python">fileinput.FileInput</code></span></a>, deprecated since Python 3.9. (Contributed by Hugo van Kemenade in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45132" class="reference external">bpo-45132</a>.)

- Removed the deprecated <a href="../library/gettext.html#module-gettext" class="reference internal" title="gettext: Multilingual internationalization services."><span class="pre"><code class="sourceCode python">gettext</code></span></a> functions <span class="pre">`lgettext()`</span>, <span class="pre">`ldgettext()`</span>, <span class="pre">`lngettext()`</span> and <span class="pre">`ldngettext()`</span>. Also removed the <span class="pre">`bind_textdomain_codeset()`</span> function, the <span class="pre">`NullTranslations.output_charset()`</span> and <span class="pre">`NullTranslations.set_output_charset()`</span> methods, and the *codeset* parameter of <span class="pre">`translation()`</span> and <span class="pre">`install()`</span>, since they are only used for the <span class="pre">`l*gettext()`</span> functions. (Contributed by Donghee Na and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44235" class="reference external">bpo-44235</a>.)

- Removed from the <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> module:

  - The <span class="pre">`getargspec()`</span> function, deprecated since Python 3.0; use <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a> or <a href="../library/inspect.html#inspect.getfullargspec" class="reference internal" title="inspect.getfullargspec"><span class="pre"><code class="sourceCode python">inspect.getfullargspec()</code></span></a> instead.

  - The <span class="pre">`formatargspec()`</span> function, deprecated since Python 3.5; use the <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a> function or the <a href="../library/inspect.html#inspect.Signature" class="reference internal" title="inspect.Signature"><span class="pre"><code class="sourceCode python">inspect.Signature</code></span></a> object directly.

  - The undocumented <span class="pre">`Signature.from_builtin()`</span> and <span class="pre">`Signature.from_function()`</span> methods, deprecated since Python 3.5; use the <a href="../library/inspect.html#inspect.Signature.from_callable" class="reference internal" title="inspect.Signature.from_callable"><span class="pre"><code class="sourceCode python">Signature.from_callable()</code></span></a> method instead.

  (Contributed by Hugo van Kemenade in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45320" class="reference external">bpo-45320</a>.)

- Removed the <a href="../reference/datamodel.html#object.__class_getitem__" class="reference internal" title="object.__class_getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__class_getitem__</span>()</code></span></a> method from <a href="../library/pathlib.html#pathlib.PurePath" class="reference internal" title="pathlib.PurePath"><span class="pre"><code class="sourceCode python">pathlib.PurePath</code></span></a>, because it was not used and added by mistake in previous versions. (Contributed by Nikita Sobolev in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46483" class="reference external">bpo-46483</a>.)

- Removed the <span class="pre">`MailmanProxy`</span> class in the <span class="pre">`smtpd`</span> module, as it is unusable without the external <span class="pre">`mailman`</span> package. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35800" class="reference external">bpo-35800</a>.)

- Removed the deprecated <span class="pre">`split()`</span> method of <span class="pre">`_tkinter.TkappType`</span>. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38371" class="reference external">bpo-38371</a>.)

- Removed namespace package support from <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> discovery. It was introduced in Python 3.4 but has been broken since Python 3.7. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23882" class="reference external">bpo-23882</a>.)

- Removed the undocumented private <span class="pre">`float.__set_format__()`</span> method, previously known as <span class="pre">`float.__setformat__()`</span> in Python 3.7. Its docstring said: “You probably don’t want to use this function. It exists mainly to be used in Python’s test suite.” (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46852" class="reference external">bpo-46852</a>.)

- The <span class="pre">`--experimental-isolated-subinterpreters`</span> configure flag (and corresponding <span class="pre">`EXPERIMENTAL_ISOLATED_SUBINTERPRETERS`</span> macro) have been removed.

- <a href="https://pypi.org/project/Pynche/" class="extlink-pypi reference external">Pynche</a> — The Pythonically Natural Color and Hue Editor — has been moved out of <span class="pre">`Tools/scripts`</span> and is <a href="https://gitlab.com/warsaw/pynche/-/tree/main" class="reference external">being developed independently</a> from the Python source tree.

</div>

<div id="porting-to-python-3-11" class="section">

<span id="whatsnew311-python-api-porting"></span><span id="whatsnew311-porting"></span>

## Porting to Python 3.11<a href="#porting-to-python-3-11" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes in the Python API that may require changes to your Python code.

Porting notes for the C API are <a href="#whatsnew311-c-api-porting" class="reference internal"><span class="std std-ref">listed separately</span></a>.

- <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a>, <a href="../library/io.html#io.open" class="reference internal" title="io.open"><span class="pre"><code class="sourceCode python">io.<span class="bu">open</span>()</code></span></a>, <a href="../library/codecs.html#codecs.open" class="reference internal" title="codecs.open"><span class="pre"><code class="sourceCode python">codecs.<span class="bu">open</span>()</code></span></a> and <a href="../library/fileinput.html#fileinput.FileInput" class="reference internal" title="fileinput.FileInput"><span class="pre"><code class="sourceCode python">fileinput.FileInput</code></span></a> no longer accept <span class="pre">`'U'`</span> (“universal newline”) in the file mode. In Python 3, “universal newline” mode is used by default whenever a file is opened in text mode, and the <span class="pre">`'U'`</span> flag has been deprecated since Python 3.3. The <a href="../library/functions.html#open-newline-parameter" class="reference internal"><span class="std std-ref">newline parameter</span></a> to these functions controls how universal newlines work. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37330" class="reference external">bpo-37330</a>.)

- <a href="../library/ast.html#ast.AST" class="reference internal" title="ast.AST"><span class="pre"><code class="sourceCode python">ast.AST</code></span></a> node positions are now validated when provided to <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a> and other related functions. If invalid positions are detected, a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> will be raised. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/93351" class="reference external">gh-93351</a>)

- Prohibited passing non-<a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">concurrent.futures.ThreadPoolExecutor</code></span></a> executors to <a href="../library/asyncio-eventloop.html#asyncio.loop.set_default_executor" class="reference internal" title="asyncio.loop.set_default_executor"><span class="pre"><code class="sourceCode python">asyncio.loop.set_default_executor()</code></span></a> following a deprecation in Python 3.8. (Contributed by Illia Volochii in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43234" class="reference external">bpo-43234</a>.)

- <a href="../library/calendar.html#module-calendar" class="reference internal" title="calendar: Functions for working with calendars, including some emulation of the Unix cal program."><span class="pre"><code class="sourceCode python">calendar</code></span></a>: The <a href="../library/calendar.html#calendar.LocaleTextCalendar" class="reference internal" title="calendar.LocaleTextCalendar"><span class="pre"><code class="sourceCode python">calendar.LocaleTextCalendar</code></span></a> and <a href="../library/calendar.html#calendar.LocaleHTMLCalendar" class="reference internal" title="calendar.LocaleHTMLCalendar"><span class="pre"><code class="sourceCode python">calendar.LocaleHTMLCalendar</code></span></a> classes now use <a href="../library/locale.html#locale.getlocale" class="reference internal" title="locale.getlocale"><span class="pre"><code class="sourceCode python">locale.getlocale()</code></span></a>, instead of using <a href="../library/locale.html#locale.getdefaultlocale" class="reference internal" title="locale.getdefaultlocale"><span class="pre"><code class="sourceCode python">locale.getdefaultlocale()</code></span></a>, if no locale is specified. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46659" class="reference external">bpo-46659</a>.)

- The <a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a> module now reads the <span class="pre">`.pdbrc`</span> configuration file with the <span class="pre">`'UTF-8'`</span> encoding. (Contributed by Srinivas Reddy Thatiparthy (శ్రీనివాస్ రెడ్డి తాటిపర్తి) in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41137" class="reference external">bpo-41137</a>.)

- The *population* parameter of <a href="../library/random.html#random.sample" class="reference internal" title="random.sample"><span class="pre"><code class="sourceCode python">random.sample()</code></span></a> must be a sequence, and automatic conversion of <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a>s to <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a>s is no longer supported. Also, if the sample size is larger than the population size, a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> is raised. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40465" class="reference external">bpo-40465</a>.)

- The *random* optional parameter of <a href="../library/random.html#random.shuffle" class="reference internal" title="random.shuffle"><span class="pre"><code class="sourceCode python">random.shuffle()</code></span></a> was removed. It was previously an arbitrary random function to use for the shuffle; now, <a href="../library/random.html#random.random" class="reference internal" title="random.random"><span class="pre"><code class="sourceCode python">random.random()</code></span></a> (its previous default) will always be used.

- In <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> <a href="../library/re.html#re-syntax" class="reference internal"><span class="std std-ref">Regular Expression Syntax</span></a>, global inline flags (e.g. <span class="pre">`(?i)`</span>) can now only be used at the start of regular expressions. Using them elsewhere has been deprecated since Python 3.6. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=47066" class="reference external">bpo-47066</a>.)

- In the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module, several long-standing bugs where fixed that, in rare cases, could cause capture groups to get the wrong result. Therefore, this could change the captured output in these cases. (Contributed by Ma Lin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35859" class="reference external">bpo-35859</a>.)

</div>

<div id="build-changes" class="section">

<span id="whatsnew311-build-changes"></span>

## Build Changes<a href="#build-changes" class="headerlink" title="Link to this heading">¶</a>

- CPython now has <span id="index-34" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a> <span id="index-35" class="target"></span><a href="https://peps.python.org/pep-0011/#tier-3" class="pep reference external"><strong>Tier 3 support</strong></a> for cross compiling to the <a href="https://webassembly.org/" class="reference external">WebAssembly</a> platforms <a href="https://emscripten.org/" class="reference external">Emscripten</a> (<span class="pre">`wasm32-unknown-emscripten`</span>, i.e. Python in the browser) and <a href="https://wasi.dev/" class="reference external">WebAssembly System Interface (WASI)</a> (<span class="pre">`wasm32-unknown-wasi`</span>). The effort is inspired by previous work like <a href="https://pyodide.org/" class="reference external">Pyodide</a>. These platforms provide a limited subset of POSIX APIs; Python standard libraries features and modules related to networking, processes, threading, signals, mmap, and users/groups are not available or don’t work. (Emscripten contributed by Christian Heimes and Ethan Smith in <a href="https://github.com/python/cpython/issues/84461" class="reference external">gh-84461</a> and WASI contributed by Christian Heimes in <a href="https://github.com/python/cpython/issues/90473" class="reference external">gh-90473</a>; platforms promoted in <a href="https://github.com/python/cpython/issues/95085" class="reference external">gh-95085</a>)

- Building CPython now requires:

  - A <a href="https://en.cppreference.com/w/c/11" class="reference external">C11</a> compiler and standard library. <a href="https://en.wikipedia.org/wiki/C11_(C_standard_revision)#Optional_features" class="reference external">Optional C11 features</a> are not required. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46656" class="reference external">bpo-46656</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45440" class="reference external">bpo-45440</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46640" class="reference external">bpo-46640</a>.)

  - Support for <a href="https://en.wikipedia.org/wiki/IEEE_754" class="reference external">IEEE 754</a> floating-point numbers. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46917" class="reference external">bpo-46917</a>.)

- The <span class="pre">`Py_NO_NAN`</span> macro has been removed. Since CPython now requires IEEE 754 floats, NaN values are always available. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46656" class="reference external">bpo-46656</a>.)

- The <a href="../library/tkinter.html#module-tkinter" class="reference internal" title="tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">tkinter</code></span></a> package now requires <a href="https://www.tcl.tk" class="reference external">Tcl/Tk</a> version 8.5.12 or newer. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46996" class="reference external">bpo-46996</a>.)

- Build dependencies, compiler flags, and linker flags for most stdlib extension modules are now detected by **configure**. libffi, libnsl, libsqlite3, zlib, bzip2, liblzma, libcrypt, Tcl/Tk, and uuid flags are detected by <a href="https://www.freedesktop.org/wiki/Software/pkg-config/" class="reference external">pkg-config</a> (when available). <a href="../library/tkinter.html#module-tkinter" class="reference internal" title="tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">tkinter</code></span></a> now requires a pkg-config command to detect development settings for <a href="https://www.tcl.tk" class="reference external">Tcl/Tk</a> headers and libraries. (Contributed by Christian Heimes and Erlend Egeberg Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45847" class="reference external">bpo-45847</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45747" class="reference external">bpo-45747</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45763" class="reference external">bpo-45763</a>.)

- libpython is no longer linked against libcrypt. (Contributed by Mike Gilbert in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45433" class="reference external">bpo-45433</a>.)

- CPython can now be built with the <a href="https://clang.llvm.org/docs/ThinLTO.html" class="reference external">ThinLTO</a> option via passing <span class="pre">`thin`</span> to <a href="../using/configure.html#cmdoption-with-lto" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--with-lto</code></span></a>, i.e. <span class="pre">`--with-lto=thin`</span>. (Contributed by Donghee Na and Brett Holman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44340" class="reference external">bpo-44340</a>.)

- Freelists for object structs can now be disabled. A new **configure** option <a href="../using/configure.html#cmdoption-without-freelists" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--without-freelists</code></span></a> can be used to disable all freelists except empty tuple singleton. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45522" class="reference external">bpo-45522</a>.)

- <span class="pre">`Modules/Setup`</span> and <span class="pre">`Modules/makesetup`</span> have been improved and tied up. Extension modules can now be built through <span class="pre">`makesetup`</span>. All except some test modules can be linked statically into a main binary or library. (Contributed by Brett Cannon and Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45548" class="reference external">bpo-45548</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45570" class="reference external">bpo-45570</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45571" class="reference external">bpo-45571</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43974" class="reference external">bpo-43974</a>.)

  <div class="admonition note">

  Note

  Use the environment variables <span class="pre">`TCLTK_CFLAGS`</span> and <span class="pre">`TCLTK_LIBS`</span> to manually specify the location of Tcl/Tk headers and libraries. The **configure** options <span class="pre">`--with-tcltk-includes`</span> and <span class="pre">`--with-tcltk-libs`</span> have been removed.

  On RHEL 7 and CentOS 7 the development packages do not provide <span class="pre">`tcl.pc`</span> and <span class="pre">`tk.pc`</span>; use <span class="pre">`TCLTK_LIBS="-ltk8.5`</span>` `<span class="pre">`-ltkstub8.5`</span>` `<span class="pre">`-ltcl8.5"`</span>. The directory <span class="pre">`Misc/rhel7`</span> contains <span class="pre">`.pc`</span> files and instructions on how to build Python with RHEL 7’s and CentOS 7’s Tcl/Tk and OpenSSL.

  </div>

- CPython will now use 30-bit digits by default for the Python <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> implementation. Previously, the default was to use 30-bit digits on platforms with <span class="pre">`SIZEOF_VOID_P`</span>` `<span class="pre">`>=`</span>` `<span class="pre">`8`</span>, and 15-bit digits otherwise. It’s still possible to explicitly request use of 15-bit digits via either the <a href="../using/configure.html#cmdoption-enable-big-digits" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--enable-big-digits</code></span></a> option to the configure script or (for Windows) the <span class="pre">`PYLONG_BITS_IN_DIGIT`</span> variable in <span class="pre">`PC/pyconfig.h`</span>, but this option may be removed at some point in the future. (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45569" class="reference external">bpo-45569</a>.)

</div>

<div id="c-api-changes" class="section">

<span id="whatsnew311-c-api"></span>

## C API Changes<a href="#c-api-changes" class="headerlink" title="Link to this heading">¶</a>

<div id="whatsnew311-c-api-new-features" class="section">

<span id="id5"></span>

### New Features<a href="#whatsnew311-c-api-new-features" class="headerlink" title="Link to this heading">¶</a>

- Add a new <a href="../c-api/type.html#c.PyType_GetName" class="reference internal" title="PyType_GetName"><span class="pre"><code class="sourceCode c">PyType_GetName<span class="op">()</span></code></span></a> function to get type’s short name. (Contributed by Hai Shi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42035" class="reference external">bpo-42035</a>.)

- Add a new <a href="../c-api/type.html#c.PyType_GetQualName" class="reference internal" title="PyType_GetQualName"><span class="pre"><code class="sourceCode c">PyType_GetQualName<span class="op">()</span></code></span></a> function to get type’s qualified name. (Contributed by Hai Shi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42035" class="reference external">bpo-42035</a>.)

- Add new <a href="../c-api/init.html#c.PyThreadState_EnterTracing" class="reference internal" title="PyThreadState_EnterTracing"><span class="pre"><code class="sourceCode c">PyThreadState_EnterTracing<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyThreadState_LeaveTracing" class="reference internal" title="PyThreadState_LeaveTracing"><span class="pre"><code class="sourceCode c">PyThreadState_LeaveTracing<span class="op">()</span></code></span></a> functions to the limited C API to suspend and resume tracing and profiling. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43760" class="reference external">bpo-43760</a>.)

- Added the <a href="../c-api/apiabiversion.html#c.Py_Version" class="reference internal" title="Py_Version"><span class="pre"><code class="sourceCode c">Py_Version</code></span></a> constant which bears the same value as <a href="../c-api/apiabiversion.html#c.PY_VERSION_HEX" class="reference internal" title="PY_VERSION_HEX"><span class="pre"><code class="sourceCode c">PY_VERSION_HEX</code></span></a>. (Contributed by Gabriele N. Tornetta in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43931" class="reference external">bpo-43931</a>.)

- <a href="../c-api/buffer.html#c.Py_buffer" class="reference internal" title="Py_buffer"><span class="pre"><code class="sourceCode c">Py_buffer</code></span></a> and APIs are now part of the limited API and the stable ABI:

  - <a href="../c-api/buffer.html#c.PyObject_CheckBuffer" class="reference internal" title="PyObject_CheckBuffer"><span class="pre"><code class="sourceCode c">PyObject_CheckBuffer<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyObject_GetBuffer" class="reference internal" title="PyObject_GetBuffer"><span class="pre"><code class="sourceCode c">PyObject_GetBuffer<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyBuffer_GetPointer" class="reference internal" title="PyBuffer_GetPointer"><span class="pre"><code class="sourceCode c">PyBuffer_GetPointer<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyBuffer_SizeFromFormat" class="reference internal" title="PyBuffer_SizeFromFormat"><span class="pre"><code class="sourceCode c">PyBuffer_SizeFromFormat<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyBuffer_ToContiguous" class="reference internal" title="PyBuffer_ToContiguous"><span class="pre"><code class="sourceCode c">PyBuffer_ToContiguous<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyBuffer_FromContiguous" class="reference internal" title="PyBuffer_FromContiguous"><span class="pre"><code class="sourceCode c">PyBuffer_FromContiguous<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyObject_CopyData" class="reference internal" title="PyObject_CopyData"><span class="pre"><code class="sourceCode c">PyObject_CopyData<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyBuffer_IsContiguous" class="reference internal" title="PyBuffer_IsContiguous"><span class="pre"><code class="sourceCode c">PyBuffer_IsContiguous<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyBuffer_FillContiguousStrides" class="reference internal" title="PyBuffer_FillContiguousStrides"><span class="pre"><code class="sourceCode c">PyBuffer_FillContiguousStrides<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyBuffer_FillInfo" class="reference internal" title="PyBuffer_FillInfo"><span class="pre"><code class="sourceCode c">PyBuffer_FillInfo<span class="op">()</span></code></span></a>

  - <a href="../c-api/buffer.html#c.PyBuffer_Release" class="reference internal" title="PyBuffer_Release"><span class="pre"><code class="sourceCode c">PyBuffer_Release<span class="op">()</span></code></span></a>

  - <a href="../c-api/memoryview.html#c.PyMemoryView_FromBuffer" class="reference internal" title="PyMemoryView_FromBuffer"><span class="pre"><code class="sourceCode c">PyMemoryView_FromBuffer<span class="op">()</span></code></span></a>

  - <a href="../c-api/typeobj.html#c.PyBufferProcs.bf_getbuffer" class="reference internal" title="PyBufferProcs.bf_getbuffer"><span class="pre"><code class="sourceCode c">bf_getbuffer</code></span></a> and <a href="../c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer" class="reference internal" title="PyBufferProcs.bf_releasebuffer"><span class="pre"><code class="sourceCode c">bf_releasebuffer</code></span></a> type slots

  (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45459" class="reference external">bpo-45459</a>.)

- Added the <a href="../c-api/type.html#c.PyType_GetModuleByDef" class="reference internal" title="PyType_GetModuleByDef"><span class="pre"><code class="sourceCode c">PyType_GetModuleByDef<span class="op">()</span></code></span></a> function, used to get the module in which a method was defined, in cases where this information is not available directly (via <a href="../c-api/structures.html#c.PyCMethod" class="reference internal" title="PyCMethod"><span class="pre"><code class="sourceCode c">PyCMethod</code></span></a>). (Contributed by Petr Viktorin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46613" class="reference external">bpo-46613</a>.)

- Add new functions to pack and unpack C double (serialize and deserialize): <a href="../c-api/float.html#c.PyFloat_Pack2" class="reference internal" title="PyFloat_Pack2"><span class="pre"><code class="sourceCode c">PyFloat_Pack2<span class="op">()</span></code></span></a>, <a href="../c-api/float.html#c.PyFloat_Pack4" class="reference internal" title="PyFloat_Pack4"><span class="pre"><code class="sourceCode c">PyFloat_Pack4<span class="op">()</span></code></span></a>, <a href="../c-api/float.html#c.PyFloat_Pack8" class="reference internal" title="PyFloat_Pack8"><span class="pre"><code class="sourceCode c">PyFloat_Pack8<span class="op">()</span></code></span></a>, <a href="../c-api/float.html#c.PyFloat_Unpack2" class="reference internal" title="PyFloat_Unpack2"><span class="pre"><code class="sourceCode c">PyFloat_Unpack2<span class="op">()</span></code></span></a>, <a href="../c-api/float.html#c.PyFloat_Unpack4" class="reference internal" title="PyFloat_Unpack4"><span class="pre"><code class="sourceCode c">PyFloat_Unpack4<span class="op">()</span></code></span></a> and <a href="../c-api/float.html#c.PyFloat_Unpack8" class="reference internal" title="PyFloat_Unpack8"><span class="pre"><code class="sourceCode c">PyFloat_Unpack8<span class="op">()</span></code></span></a>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46906" class="reference external">bpo-46906</a>.)

- Add new functions to get frame object attributes: <a href="../c-api/frame.html#c.PyFrame_GetBuiltins" class="reference internal" title="PyFrame_GetBuiltins"><span class="pre"><code class="sourceCode c">PyFrame_GetBuiltins<span class="op">()</span></code></span></a>, <a href="../c-api/frame.html#c.PyFrame_GetGenerator" class="reference internal" title="PyFrame_GetGenerator"><span class="pre"><code class="sourceCode c">PyFrame_GetGenerator<span class="op">()</span></code></span></a>, <a href="../c-api/frame.html#c.PyFrame_GetGlobals" class="reference internal" title="PyFrame_GetGlobals"><span class="pre"><code class="sourceCode c">PyFrame_GetGlobals<span class="op">()</span></code></span></a>, <a href="../c-api/frame.html#c.PyFrame_GetLasti" class="reference internal" title="PyFrame_GetLasti"><span class="pre"><code class="sourceCode c">PyFrame_GetLasti<span class="op">()</span></code></span></a>.

- Added two new functions to get and set the active exception instance: <a href="../c-api/exceptions.html#c.PyErr_GetHandledException" class="reference internal" title="PyErr_GetHandledException"><span class="pre"><code class="sourceCode c">PyErr_GetHandledException<span class="op">()</span></code></span></a> and <a href="../c-api/exceptions.html#c.PyErr_SetHandledException" class="reference internal" title="PyErr_SetHandledException"><span class="pre"><code class="sourceCode c">PyErr_SetHandledException<span class="op">()</span></code></span></a>. These are alternatives to <a href="../c-api/exceptions.html#c.PyErr_SetExcInfo" class="reference internal" title="PyErr_SetExcInfo"><span class="pre"><code class="sourceCode c">PyErr_SetExcInfo<span class="op">()</span></code></span></a> and <a href="../c-api/exceptions.html#c.PyErr_GetExcInfo" class="reference internal" title="PyErr_GetExcInfo"><span class="pre"><code class="sourceCode c">PyErr_GetExcInfo<span class="op">()</span></code></span></a> which work with the legacy 3-tuple representation of exceptions. (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46343" class="reference external">bpo-46343</a>.)

- Added the <a href="../c-api/init_config.html#c.PyConfig.safe_path" class="reference internal" title="PyConfig.safe_path"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>safe_path</code></span></a> member. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/57684" class="reference external">gh-57684</a>.)

</div>

<div id="whatsnew311-c-api-porting" class="section">

<span id="id6"></span>

### Porting to Python 3.11<a href="#whatsnew311-c-api-porting" class="headerlink" title="Link to this heading">¶</a>

- Some macros have been converted to static inline functions to avoid <a href="https://gcc.gnu.org/onlinedocs/cpp/Macro-Pitfalls.html" class="reference external">macro pitfalls</a>. The change should be mostly transparent to users, as the replacement functions will cast their arguments to the expected types to avoid compiler warnings due to static type checks. However, when the limited C API is set to \>=3.11, these casts are not done, and callers will need to cast arguments to their expected types. See <span id="index-36" class="target"></span><a href="https://peps.python.org/pep-0670/" class="pep reference external"><strong>PEP 670</strong></a> for more details. (Contributed by Victor Stinner and Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/89653" class="reference external">gh-89653</a>.)

- <a href="../c-api/exceptions.html#c.PyErr_SetExcInfo" class="reference internal" title="PyErr_SetExcInfo"><span class="pre"><code class="sourceCode c">PyErr_SetExcInfo<span class="op">()</span></code></span></a> no longer uses the <span class="pre">`type`</span> and <span class="pre">`traceback`</span> arguments, the interpreter now derives those values from the exception instance (the <span class="pre">`value`</span> argument). The function still steals references of all three arguments. (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45711" class="reference external">bpo-45711</a>.)

- <a href="../c-api/exceptions.html#c.PyErr_GetExcInfo" class="reference internal" title="PyErr_GetExcInfo"><span class="pre"><code class="sourceCode c">PyErr_GetExcInfo<span class="op">()</span></code></span></a> now derives the <span class="pre">`type`</span> and <span class="pre">`traceback`</span> fields of the result from the exception instance (the <span class="pre">`value`</span> field). (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45711" class="reference external">bpo-45711</a>.)

- <a href="../c-api/import.html#c._frozen" class="reference internal" title="_frozen"><span class="pre"><code class="sourceCode c">_frozen</code></span></a> has a new <span class="pre">`is_package`</span> field to indicate whether or not the frozen module is a package. Previously, a negative value in the <span class="pre">`size`</span> field was the indicator. Now only non-negative values be used for <span class="pre">`size`</span>. (Contributed by Kumar Aditya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46608" class="reference external">bpo-46608</a>.)

- <a href="../c-api/init.html#c._PyFrameEvalFunction" class="reference internal" title="_PyFrameEvalFunction"><span class="pre"><code class="sourceCode c">_PyFrameEvalFunction<span class="op">()</span></code></span></a> now takes <span class="pre">`_PyInterpreterFrame*`</span> as its second parameter, instead of <span class="pre">`PyFrameObject*`</span>. See <span id="index-37" class="target"></span><a href="https://peps.python.org/pep-0523/" class="pep reference external"><strong>PEP 523</strong></a> for more details of how to use this function pointer type.

- <span class="pre">`PyCode_New()`</span> and <span class="pre">`PyCode_NewWithPosOnlyArgs()`</span> now take an additional <span class="pre">`exception_table`</span> argument. Using these functions should be avoided, if at all possible. To get a custom code object: create a code object using the compiler, then get a modified version with the <span class="pre">`replace`</span> method.

- <a href="../c-api/code.html#c.PyCodeObject" class="reference internal" title="PyCodeObject"><span class="pre"><code class="sourceCode c">PyCodeObject</code></span></a> no longer has the <span class="pre">`co_code`</span>, <span class="pre">`co_varnames`</span>, <span class="pre">`co_cellvars`</span> and <span class="pre">`co_freevars`</span> fields. Instead, use <a href="../c-api/code.html#c.PyCode_GetCode" class="reference internal" title="PyCode_GetCode"><span class="pre"><code class="sourceCode c">PyCode_GetCode<span class="op">()</span></code></span></a>, <a href="../c-api/code.html#c.PyCode_GetVarnames" class="reference internal" title="PyCode_GetVarnames"><span class="pre"><code class="sourceCode c">PyCode_GetVarnames<span class="op">()</span></code></span></a>, <a href="../c-api/code.html#c.PyCode_GetCellvars" class="reference internal" title="PyCode_GetCellvars"><span class="pre"><code class="sourceCode c">PyCode_GetCellvars<span class="op">()</span></code></span></a> and <a href="../c-api/code.html#c.PyCode_GetFreevars" class="reference internal" title="PyCode_GetFreevars"><span class="pre"><code class="sourceCode c">PyCode_GetFreevars<span class="op">()</span></code></span></a> respectively to access them via the C API. (Contributed by Brandt Bucher in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46841" class="reference external">bpo-46841</a> and Ken Jin in <a href="https://github.com/python/cpython/issues/92154" class="reference external">gh-92154</a> and <a href="https://github.com/python/cpython/issues/94936" class="reference external">gh-94936</a>.)

- The old trashcan macros (<span class="pre">`Py_TRASHCAN_SAFE_BEGIN`</span>/<span class="pre">`Py_TRASHCAN_SAFE_END`</span>) are now deprecated. They should be replaced by the new macros <span class="pre">`Py_TRASHCAN_BEGIN`</span> and <span class="pre">`Py_TRASHCAN_END`</span>.

  A tp_dealloc function that has the old macros, such as:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      static void
      mytype_dealloc(mytype *p)
      {
          PyObject_GC_UnTrack(p);
          Py_TRASHCAN_SAFE_BEGIN(p);
          ...
          Py_TRASHCAN_SAFE_END
      }

  </div>

  </div>

  should migrate to the new macros as follows:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      static void
      mytype_dealloc(mytype *p)
      {
          PyObject_GC_UnTrack(p);
          Py_TRASHCAN_BEGIN(p, mytype_dealloc)
          ...
          Py_TRASHCAN_END
      }

  </div>

  </div>

  Note that <span class="pre">`Py_TRASHCAN_BEGIN`</span> has a second argument which should be the deallocation function it is in.

  To support older Python versions in the same codebase, you can define the following macros and use them throughout the code (credit: these were copied from the <span class="pre">`mypy`</span> codebase):

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      #if PY_VERSION_HEX >= 0x03080000
      #  define CPy_TRASHCAN_BEGIN(op, dealloc) Py_TRASHCAN_BEGIN(op, dealloc)
      #  define CPy_TRASHCAN_END(op) Py_TRASHCAN_END
      #else
      #  define CPy_TRASHCAN_BEGIN(op, dealloc) Py_TRASHCAN_SAFE_BEGIN(op)
      #  define CPy_TRASHCAN_END(op) Py_TRASHCAN_SAFE_END(op)
      #endif

  </div>

  </div>

- The <a href="../c-api/type.html#c.PyType_Ready" class="reference internal" title="PyType_Ready"><span class="pre"><code class="sourceCode c">PyType_Ready<span class="op">()</span></code></span></a> function now raises an error if a type is defined with the <a href="../c-api/typeobj.html#c.Py_TPFLAGS_HAVE_GC" class="reference internal" title="Py_TPFLAGS_HAVE_GC"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_HAVE_GC</code></span></a> flag set but has no traverse function (<a href="../c-api/typeobj.html#c.PyTypeObject.tp_traverse" class="reference internal" title="PyTypeObject.tp_traverse"><span class="pre"><code class="sourceCode c">PyTypeObject<span class="op">.</span>tp_traverse</code></span></a>). (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44263" class="reference external">bpo-44263</a>.)

- Heap types with the <a href="../c-api/typeobj.html#c.Py_TPFLAGS_IMMUTABLETYPE" class="reference internal" title="Py_TPFLAGS_IMMUTABLETYPE"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_IMMUTABLETYPE</code></span></a> flag can now inherit the <span id="index-38" class="target"></span><a href="https://peps.python.org/pep-0590/" class="pep reference external"><strong>PEP 590</strong></a> vectorcall protocol. Previously, this was only possible for <a href="../c-api/typeobj.html#static-types" class="reference internal"><span class="std std-ref">static types</span></a>. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43908" class="reference external">bpo-43908</a>)

- Since <a href="../c-api/structures.html#c.Py_TYPE" class="reference internal" title="Py_TYPE"><span class="pre"><code class="sourceCode c">Py_TYPE<span class="op">()</span></code></span></a> is changed to a inline static function, <span class="pre">`Py_TYPE(obj)`</span>` `<span class="pre">`=`</span>` `<span class="pre">`new_type`</span> must be replaced with <span class="pre">`Py_SET_TYPE(obj,`</span>` `<span class="pre">`new_type)`</span>: see the <a href="../c-api/structures.html#c.Py_SET_TYPE" class="reference internal" title="Py_SET_TYPE"><span class="pre"><code class="sourceCode c">Py_SET_TYPE<span class="op">()</span></code></span></a> function (available since Python 3.9). For backward compatibility, this macro can be used:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      #if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
      static inline void _Py_SET_TYPE(PyObject *ob, PyTypeObject *type)
      { ob->ob_type = type; }
      #define Py_SET_TYPE(ob, type) _Py_SET_TYPE((PyObject*)(ob), type)
      #endif

  </div>

  </div>

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39573" class="reference external">bpo-39573</a>.)

- Since <a href="../c-api/structures.html#c.Py_SIZE" class="reference internal" title="Py_SIZE"><span class="pre"><code class="sourceCode c">Py_SIZE<span class="op">()</span></code></span></a> is changed to a inline static function, <span class="pre">`Py_SIZE(obj)`</span>` `<span class="pre">`=`</span>` `<span class="pre">`new_size`</span> must be replaced with <span class="pre">`Py_SET_SIZE(obj,`</span>` `<span class="pre">`new_size)`</span>: see the <a href="../c-api/structures.html#c.Py_SET_SIZE" class="reference internal" title="Py_SET_SIZE"><span class="pre"><code class="sourceCode c">Py_SET_SIZE<span class="op">()</span></code></span></a> function (available since Python 3.9). For backward compatibility, this macro can be used:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      #if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_SIZE)
      static inline void _Py_SET_SIZE(PyVarObject *ob, Py_ssize_t size)
      { ob->ob_size = size; }
      #define Py_SET_SIZE(ob, size) _Py_SET_SIZE((PyVarObject*)(ob), size)
      #endif

  </div>

  </div>

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39573" class="reference external">bpo-39573</a>.)

- <span class="pre">`<Python.h>`</span> no longer includes the header files <span class="pre">`<stdlib.h>`</span>, <span class="pre">`<stdio.h>`</span>, <span class="pre">`<errno.h>`</span> and <span class="pre">`<string.h>`</span> when the <span class="pre">`Py_LIMITED_API`</span> macro is set to <span class="pre">`0x030b0000`</span> (Python 3.11) or higher. C extensions should explicitly include the header files after <span class="pre">`#include`</span>` `<span class="pre">`<Python.h>`</span>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45434" class="reference external">bpo-45434</a>.)

- The non-limited API files <span class="pre">`cellobject.h`</span>, <span class="pre">`classobject.h`</span>, <span class="pre">`code.h`</span>, <span class="pre">`context.h`</span>, <span class="pre">`funcobject.h`</span>, <span class="pre">`genobject.h`</span> and <span class="pre">`longintrepr.h`</span> have been moved to the <span class="pre">`Include/cpython`</span> directory. Moreover, the <span class="pre">`eval.h`</span> header file was removed. These files must not be included directly, as they are already included in <span class="pre">`Python.h`</span>: <a href="../c-api/intro.html#api-includes" class="reference internal"><span class="std std-ref">Include Files</span></a>. If they have been included directly, consider including <span class="pre">`Python.h`</span> instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35134" class="reference external">bpo-35134</a>.)

- The <span class="pre">`PyUnicode_CHECK_INTERNED()`</span> macro has been excluded from the limited C API. It was never usable there, because it used internal structures which are not available in the limited C API. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46007" class="reference external">bpo-46007</a>.)

- The following frame functions and type are now directly available with <span class="pre">`#include`</span>` `<span class="pre">`<Python.h>`</span>, it’s no longer needed to add <span class="pre">`#include`</span>` `<span class="pre">`<frameobject.h>`</span>:

  - <a href="../c-api/frame.html#c.PyFrame_Check" class="reference internal" title="PyFrame_Check"><span class="pre"><code class="sourceCode c">PyFrame_Check<span class="op">()</span></code></span></a>

  - <a href="../c-api/frame.html#c.PyFrame_GetBack" class="reference internal" title="PyFrame_GetBack"><span class="pre"><code class="sourceCode c">PyFrame_GetBack<span class="op">()</span></code></span></a>

  - <a href="../c-api/frame.html#c.PyFrame_GetBuiltins" class="reference internal" title="PyFrame_GetBuiltins"><span class="pre"><code class="sourceCode c">PyFrame_GetBuiltins<span class="op">()</span></code></span></a>

  - <a href="../c-api/frame.html#c.PyFrame_GetGenerator" class="reference internal" title="PyFrame_GetGenerator"><span class="pre"><code class="sourceCode c">PyFrame_GetGenerator<span class="op">()</span></code></span></a>

  - <a href="../c-api/frame.html#c.PyFrame_GetGlobals" class="reference internal" title="PyFrame_GetGlobals"><span class="pre"><code class="sourceCode c">PyFrame_GetGlobals<span class="op">()</span></code></span></a>

  - <a href="../c-api/frame.html#c.PyFrame_GetLasti" class="reference internal" title="PyFrame_GetLasti"><span class="pre"><code class="sourceCode c">PyFrame_GetLasti<span class="op">()</span></code></span></a>

  - <a href="../c-api/frame.html#c.PyFrame_GetLocals" class="reference internal" title="PyFrame_GetLocals"><span class="pre"><code class="sourceCode c">PyFrame_GetLocals<span class="op">()</span></code></span></a>

  - <a href="../c-api/frame.html#c.PyFrame_Type" class="reference internal" title="PyFrame_Type"><span class="pre"><code class="sourceCode c">PyFrame_Type</code></span></a>

  (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/93937" class="reference external">gh-93937</a>.)

<!-- -->

- The <a href="../c-api/frame.html#c.PyFrameObject" class="reference internal" title="PyFrameObject"><span class="pre"><code class="sourceCode c">PyFrameObject</code></span></a> structure members have been removed from the public C API.

  While the documentation notes that the <a href="../c-api/frame.html#c.PyFrameObject" class="reference internal" title="PyFrameObject"><span class="pre"><code class="sourceCode c">PyFrameObject</code></span></a> fields are subject to change at any time, they have been stable for a long time and were used in several popular extensions.

  In Python 3.11, the frame struct was reorganized to allow performance optimizations. Some fields were removed entirely, as they were details of the old implementation.

  <a href="../c-api/frame.html#c.PyFrameObject" class="reference internal" title="PyFrameObject"><span class="pre"><code class="sourceCode c">PyFrameObject</code></span></a> fields:

  - <span class="pre">`f_back`</span>: use <a href="../c-api/frame.html#c.PyFrame_GetBack" class="reference internal" title="PyFrame_GetBack"><span class="pre"><code class="sourceCode c">PyFrame_GetBack<span class="op">()</span></code></span></a>.

  - <span class="pre">`f_blockstack`</span>: removed.

  - <span class="pre">`f_builtins`</span>: use <a href="../c-api/frame.html#c.PyFrame_GetBuiltins" class="reference internal" title="PyFrame_GetBuiltins"><span class="pre"><code class="sourceCode c">PyFrame_GetBuiltins<span class="op">()</span></code></span></a>.

  - <span class="pre">`f_code`</span>: use <a href="../c-api/frame.html#c.PyFrame_GetCode" class="reference internal" title="PyFrame_GetCode"><span class="pre"><code class="sourceCode c">PyFrame_GetCode<span class="op">()</span></code></span></a>.

  - <span class="pre">`f_gen`</span>: use <a href="../c-api/frame.html#c.PyFrame_GetGenerator" class="reference internal" title="PyFrame_GetGenerator"><span class="pre"><code class="sourceCode c">PyFrame_GetGenerator<span class="op">()</span></code></span></a>.

  - <span class="pre">`f_globals`</span>: use <a href="../c-api/frame.html#c.PyFrame_GetGlobals" class="reference internal" title="PyFrame_GetGlobals"><span class="pre"><code class="sourceCode c">PyFrame_GetGlobals<span class="op">()</span></code></span></a>.

  - <span class="pre">`f_iblock`</span>: removed.

  - <span class="pre">`f_lasti`</span>: use <a href="../c-api/frame.html#c.PyFrame_GetLasti" class="reference internal" title="PyFrame_GetLasti"><span class="pre"><code class="sourceCode c">PyFrame_GetLasti<span class="op">()</span></code></span></a>. Code using <span class="pre">`f_lasti`</span> with <span class="pre">`PyCode_Addr2Line()`</span> should use <a href="../c-api/frame.html#c.PyFrame_GetLineNumber" class="reference internal" title="PyFrame_GetLineNumber"><span class="pre"><code class="sourceCode c">PyFrame_GetLineNumber<span class="op">()</span></code></span></a> instead; it may be faster.

  - <span class="pre">`f_lineno`</span>: use <a href="../c-api/frame.html#c.PyFrame_GetLineNumber" class="reference internal" title="PyFrame_GetLineNumber"><span class="pre"><code class="sourceCode c">PyFrame_GetLineNumber<span class="op">()</span></code></span></a>

  - <span class="pre">`f_locals`</span>: use <a href="../c-api/frame.html#c.PyFrame_GetLocals" class="reference internal" title="PyFrame_GetLocals"><span class="pre"><code class="sourceCode c">PyFrame_GetLocals<span class="op">()</span></code></span></a>.

  - <span class="pre">`f_stackdepth`</span>: removed.

  - <span class="pre">`f_state`</span>: no public API (renamed to <span class="pre">`f_frame.f_state`</span>).

  - <span class="pre">`f_trace`</span>: no public API.

  - <span class="pre">`f_trace_lines`</span>: use <span class="pre">`PyObject_GetAttrString((PyObject*)frame,`</span>` `<span class="pre">`"f_trace_lines")`</span>.

  - <span class="pre">`f_trace_opcodes`</span>: use <span class="pre">`PyObject_GetAttrString((PyObject*)frame,`</span>` `<span class="pre">`"f_trace_opcodes")`</span>.

  - <span class="pre">`f_localsplus`</span>: no public API (renamed to <span class="pre">`f_frame.localsplus`</span>).

  - <span class="pre">`f_valuestack`</span>: removed.

  The Python frame object is now created lazily. A side effect is that the <a href="../reference/datamodel.html#frame.f_back" class="reference internal" title="frame.f_back"><span class="pre"><code class="sourceCode python">f_back</code></span></a> member must not be accessed directly, since its value is now also computed lazily. The <a href="../c-api/frame.html#c.PyFrame_GetBack" class="reference internal" title="PyFrame_GetBack"><span class="pre"><code class="sourceCode c">PyFrame_GetBack<span class="op">()</span></code></span></a> function must be called instead.

  Debuggers that accessed the <a href="../reference/datamodel.html#frame.f_locals" class="reference internal" title="frame.f_locals"><span class="pre"><code class="sourceCode python">f_locals</code></span></a> directly *must* call <a href="../c-api/frame.html#c.PyFrame_GetLocals" class="reference internal" title="PyFrame_GetLocals"><span class="pre"><code class="sourceCode c">PyFrame_GetLocals<span class="op">()</span></code></span></a> instead. They no longer need to call <span class="pre">`PyFrame_FastToLocalsWithError()`</span> or <span class="pre">`PyFrame_LocalsToFast()`</span>, in fact they should not call those functions. The necessary updating of the frame is now managed by the virtual machine.

  Code defining <span class="pre">`PyFrame_GetCode()`</span> on Python 3.8 and older:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      #if PY_VERSION_HEX < 0x030900B1
      static inline PyCodeObject* PyFrame_GetCode(PyFrameObject *frame)
      {
          Py_INCREF(frame->f_code);
          return frame->f_code;
      }
      #endif

  </div>

  </div>

  Code defining <span class="pre">`PyFrame_GetBack()`</span> on Python 3.8 and older:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      #if PY_VERSION_HEX < 0x030900B1
      static inline PyFrameObject* PyFrame_GetBack(PyFrameObject *frame)
      {
          Py_XINCREF(frame->f_back);
          return frame->f_back;
      }
      #endif

  </div>

  </div>

  Or use the <a href="https://github.com/python/pythoncapi-compat" class="reference external">pythoncapi_compat project</a> to get these two functions on older Python versions.

- Changes of the <a href="../c-api/init.html#c.PyThreadState" class="reference internal" title="PyThreadState"><span class="pre"><code class="sourceCode c">PyThreadState</code></span></a> structure members:

  - <span class="pre">`frame`</span>: removed, use <a href="../c-api/init.html#c.PyThreadState_GetFrame" class="reference internal" title="PyThreadState_GetFrame"><span class="pre"><code class="sourceCode c">PyThreadState_GetFrame<span class="op">()</span></code></span></a> (function added to Python 3.9 by <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40429" class="reference external">bpo-40429</a>). Warning: the function returns a <a href="../glossary.html#term-strong-reference" class="reference internal"><span class="xref std std-term">strong reference</span></a>, need to call <a href="../c-api/refcounting.html#c.Py_XDECREF" class="reference internal" title="Py_XDECREF"><span class="pre"><code class="sourceCode c">Py_XDECREF<span class="op">()</span></code></span></a>.

  - <span class="pre">`tracing`</span>: changed, use <a href="../c-api/init.html#c.PyThreadState_EnterTracing" class="reference internal" title="PyThreadState_EnterTracing"><span class="pre"><code class="sourceCode c">PyThreadState_EnterTracing<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyThreadState_LeaveTracing" class="reference internal" title="PyThreadState_LeaveTracing"><span class="pre"><code class="sourceCode c">PyThreadState_LeaveTracing<span class="op">()</span></code></span></a> (functions added to Python 3.11 by <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43760" class="reference external">bpo-43760</a>).

  - <span class="pre">`recursion_depth`</span>: removed, use <span class="pre">`(tstate->recursion_limit`</span>` `<span class="pre">`-`</span>` `<span class="pre">`tstate->recursion_remaining)`</span> instead.

  - <span class="pre">`stackcheck_counter`</span>: removed.

  Code defining <span class="pre">`PyThreadState_GetFrame()`</span> on Python 3.8 and older:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      #if PY_VERSION_HEX < 0x030900B1
      static inline PyFrameObject* PyThreadState_GetFrame(PyThreadState *tstate)
      {
          Py_XINCREF(tstate->frame);
          return tstate->frame;
      }
      #endif

  </div>

  </div>

  Code defining <span class="pre">`PyThreadState_EnterTracing()`</span> and <span class="pre">`PyThreadState_LeaveTracing()`</span> on Python 3.10 and older:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      #if PY_VERSION_HEX < 0x030B00A2
      static inline void PyThreadState_EnterTracing(PyThreadState *tstate)
      {
          tstate->tracing++;
      #if PY_VERSION_HEX >= 0x030A00A1
          tstate->cframe->use_tracing = 0;
      #else
          tstate->use_tracing = 0;
      #endif
      }

      static inline void PyThreadState_LeaveTracing(PyThreadState *tstate)
      {
          int use_tracing = (tstate->c_tracefunc != NULL || tstate->c_profilefunc != NULL);
          tstate->tracing--;
      #if PY_VERSION_HEX >= 0x030A00A1
          tstate->cframe->use_tracing = use_tracing;
      #else
          tstate->use_tracing = use_tracing;
      #endif
      }
      #endif

  </div>

  </div>

  Or use <a href="https://github.com/python/pythoncapi-compat" class="reference external">the pythoncapi-compat project</a> to get these functions on old Python functions.

- Distributors are encouraged to build Python with the optimized Blake2 library <a href="https://www.blake2.net/" class="reference external">libb2</a>.

- The <a href="../c-api/init_config.html#c.PyConfig.module_search_paths_set" class="reference internal" title="PyConfig.module_search_paths_set"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>module_search_paths_set</code></span></a> field must now be set to 1 for initialization to use <a href="../c-api/init_config.html#c.PyConfig.module_search_paths" class="reference internal" title="PyConfig.module_search_paths"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>module_search_paths</code></span></a> to initialize <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a>. Otherwise, initialization will recalculate the path and replace any values added to <span class="pre">`module_search_paths`</span>.

- <a href="../c-api/init_config.html#c.PyConfig_Read" class="reference internal" title="PyConfig_Read"><span class="pre"><code class="sourceCode c">PyConfig_Read<span class="op">()</span></code></span></a> no longer calculates the initial search path, and will not fill any values into <a href="../c-api/init_config.html#c.PyConfig.module_search_paths" class="reference internal" title="PyConfig.module_search_paths"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>module_search_paths</code></span></a>. To calculate default paths and then modify them, finish initialization and use <a href="../c-api/sys.html#c.PySys_GetObject" class="reference internal" title="PySys_GetObject"><span class="pre"><code class="sourceCode c">PySys_GetObject<span class="op">()</span></code></span></a> to retrieve <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> as a Python list object and modify it directly.

</div>

<div id="whatsnew311-c-api-deprecated" class="section">

<span id="id7"></span>

### Deprecated<a href="#whatsnew311-c-api-deprecated" class="headerlink" title="Link to this heading">¶</a>

- Deprecate the following functions to configure the Python initialization:

  - <span class="pre">`PySys_AddWarnOptionUnicode()`</span>

  - <span class="pre">`PySys_AddWarnOption()`</span>

  - <span class="pre">`PySys_AddXOption()`</span>

  - <span class="pre">`PySys_HasWarnOptions()`</span>

  - <span class="pre">`PySys_SetArgvEx()`</span>

  - <span class="pre">`PySys_SetArgv()`</span>

  - <span class="pre">`PySys_SetPath()`</span>

  - <span class="pre">`Py_SetPath()`</span>

  - <span class="pre">`Py_SetProgramName()`</span>

  - <span class="pre">`Py_SetPythonHome()`</span>

  - <span class="pre">`Py_SetStandardStreamEncoding()`</span>

  - <span class="pre">`_Py_SetProgramFullPath()`</span>

  Use the new <a href="../c-api/init_config.html#c.PyConfig" class="reference internal" title="PyConfig"><span class="pre"><code class="sourceCode c">PyConfig</code></span></a> API of the <a href="../c-api/init_config.html#init-config" class="reference internal"><span class="std std-ref">Python Initialization Configuration</span></a> instead (<span id="index-39" class="target"></span><a href="https://peps.python.org/pep-0587/" class="pep reference external"><strong>PEP 587</strong></a>). (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/88279" class="reference external">gh-88279</a>.)

- Deprecate the <span class="pre">`ob_shash`</span> member of the <a href="../c-api/bytes.html#c.PyBytesObject" class="reference internal" title="PyBytesObject"><span class="pre"><code class="sourceCode c">PyBytesObject</code></span></a>. Use <a href="../c-api/object.html#c.PyObject_Hash" class="reference internal" title="PyObject_Hash"><span class="pre"><code class="sourceCode c">PyObject_Hash<span class="op">()</span></code></span></a> instead. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=46864" class="reference external">bpo-46864</a>.)

</div>

<div id="whatsnew311-c-api-pending-removal" class="section">

<span id="id8"></span>

### Pending Removal in Python 3.12<a href="#whatsnew311-c-api-pending-removal" class="headerlink" title="Link to this heading">¶</a>

The following C APIs have been deprecated in earlier Python releases, and will be removed in Python 3.12.

- <span class="pre">`PyUnicode_AS_DATA()`</span>

- <span class="pre">`PyUnicode_AS_UNICODE()`</span>

- <span class="pre">`PyUnicode_AsUnicodeAndSize()`</span>

- <span class="pre">`PyUnicode_AsUnicode()`</span>

- <span class="pre">`PyUnicode_FromUnicode()`</span>

- <span class="pre">`PyUnicode_GET_DATA_SIZE()`</span>

- <span class="pre">`PyUnicode_GET_SIZE()`</span>

- <span class="pre">`PyUnicode_GetSize()`</span>

- <span class="pre">`PyUnicode_IS_COMPACT()`</span>

- <span class="pre">`PyUnicode_IS_READY()`</span>

- <a href="../c-api/unicode.html#c.PyUnicode_READY" class="reference internal" title="PyUnicode_READY"><span class="pre"><code class="sourceCode c">PyUnicode_READY<span class="op">()</span></code></span></a>

- <span class="pre">`PyUnicode_WSTR_LENGTH()`</span>

- <span class="pre">`_PyUnicode_AsUnicode()`</span>

- <span class="pre">`PyUnicode_WCHAR_KIND`</span>

- <a href="../c-api/unicode.html#c.PyUnicodeObject" class="reference internal" title="PyUnicodeObject"><span class="pre"><code class="sourceCode c">PyUnicodeObject</code></span></a>

- <span class="pre">`PyUnicode_InternImmortal()`</span>

</div>

<div id="whatsnew311-c-api-removed" class="section">

<span id="id9"></span>

### Removed<a href="#whatsnew311-c-api-removed" class="headerlink" title="Link to this heading">¶</a>

- <span class="pre">`PyFrame_BlockSetup()`</span> and <span class="pre">`PyFrame_BlockPop()`</span> have been removed. (Contributed by Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40222" class="reference external">bpo-40222</a>.)

- Remove the following math macros using the <span class="pre">`errno`</span> variable:

  - <span class="pre">`Py_ADJUST_ERANGE1()`</span>

  - <span class="pre">`Py_ADJUST_ERANGE2()`</span>

  - <span class="pre">`Py_OVERFLOWED()`</span>

  - <span class="pre">`Py_SET_ERANGE_IF_OVERFLOW()`</span>

  - <span class="pre">`Py_SET_ERRNO_ON_MATH_ERROR()`</span>

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45412" class="reference external">bpo-45412</a>.)

- Remove <span class="pre">`Py_UNICODE_COPY()`</span> and <span class="pre">`Py_UNICODE_FILL()`</span> macros, deprecated since Python 3.3. Use <span class="pre">`PyUnicode_CopyCharacters()`</span> or <span class="pre">`memcpy()`</span> (<span class="pre">`wchar_t*`</span> string), and <span class="pre">`PyUnicode_Fill()`</span> functions instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41123" class="reference external">bpo-41123</a>.)

- Remove the <span class="pre">`pystrhex.h`</span> header file. It only contains private functions. C extensions should only include the main <span class="pre">`<Python.h>`</span> header file. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45434" class="reference external">bpo-45434</a>.)

- Remove the <span class="pre">`Py_FORCE_DOUBLE()`</span> macro. It was used by the <span class="pre">`Py_IS_INFINITY()`</span> macro. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45440" class="reference external">bpo-45440</a>.)

- The following items are no longer available when <a href="../c-api/stable.html#c.Py_LIMITED_API" class="reference internal" title="Py_LIMITED_API"><span class="pre"><code class="sourceCode c">Py_LIMITED_API</code></span></a> is defined:

  - <a href="../c-api/marshal.html#c.PyMarshal_WriteLongToFile" class="reference internal" title="PyMarshal_WriteLongToFile"><span class="pre"><code class="sourceCode c">PyMarshal_WriteLongToFile<span class="op">()</span></code></span></a>

  - <a href="../c-api/marshal.html#c.PyMarshal_WriteObjectToFile" class="reference internal" title="PyMarshal_WriteObjectToFile"><span class="pre"><code class="sourceCode c">PyMarshal_WriteObjectToFile<span class="op">()</span></code></span></a>

  - <a href="../c-api/marshal.html#c.PyMarshal_ReadObjectFromString" class="reference internal" title="PyMarshal_ReadObjectFromString"><span class="pre"><code class="sourceCode c">PyMarshal_ReadObjectFromString<span class="op">()</span></code></span></a>

  - <a href="../c-api/marshal.html#c.PyMarshal_WriteObjectToString" class="reference internal" title="PyMarshal_WriteObjectToString"><span class="pre"><code class="sourceCode c">PyMarshal_WriteObjectToString<span class="op">()</span></code></span></a>

  - the <span class="pre">`Py_MARSHAL_VERSION`</span> macro

  These are not part of the <a href="../c-api/stable.html#limited-api-list" class="reference internal"><span class="std std-ref">limited API</span></a>.

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45474" class="reference external">bpo-45474</a>.)

- Exclude <a href="../c-api/weakref.html#c.PyWeakref_GET_OBJECT" class="reference internal" title="PyWeakref_GET_OBJECT"><span class="pre"><code class="sourceCode c">PyWeakref_GET_OBJECT<span class="op">()</span></code></span></a> from the limited C API. It never worked since the <span class="pre">`PyWeakReference`</span> structure is opaque in the limited C API. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35134" class="reference external">bpo-35134</a>.)

- Remove the <span class="pre">`PyHeapType_GET_MEMBERS()`</span> macro. It was exposed in the public C API by mistake, it must only be used by Python internally. Use the <span class="pre">`PyTypeObject.tp_members`</span> member instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40170" class="reference external">bpo-40170</a>.)

- Remove the <span class="pre">`HAVE_PY_SET_53BIT_PRECISION`</span> macro (moved to the internal C API). (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45412" class="reference external">bpo-45412</a>.)

<!-- -->

- Remove the <a href="../c-api/unicode.html#c.Py_UNICODE" class="reference internal" title="Py_UNICODE"><span class="pre"><code class="sourceCode c">Py_UNICODE</code></span></a> encoder APIs, as they have been deprecated since Python 3.3, are little used and are inefficient relative to the recommended alternatives.

  The removed functions are:

  - <span class="pre">`PyUnicode_Encode()`</span>

  - <span class="pre">`PyUnicode_EncodeASCII()`</span>

  - <span class="pre">`PyUnicode_EncodeLatin1()`</span>

  - <span class="pre">`PyUnicode_EncodeUTF7()`</span>

  - <span class="pre">`PyUnicode_EncodeUTF8()`</span>

  - <span class="pre">`PyUnicode_EncodeUTF16()`</span>

  - <span class="pre">`PyUnicode_EncodeUTF32()`</span>

  - <span class="pre">`PyUnicode_EncodeUnicodeEscape()`</span>

  - <span class="pre">`PyUnicode_EncodeRawUnicodeEscape()`</span>

  - <span class="pre">`PyUnicode_EncodeCharmap()`</span>

  - <span class="pre">`PyUnicode_TranslateCharmap()`</span>

  - <span class="pre">`PyUnicode_EncodeDecimal()`</span>

  - <span class="pre">`PyUnicode_TransformDecimalToASCII()`</span>

  See <span id="index-40" class="target"></span><a href="https://peps.python.org/pep-0624/" class="pep reference external"><strong>PEP 624</strong></a> for details and <span id="index-41" class="target"></span><a href="https://peps.python.org/pep-0624/#alternative-apis" class="pep reference external"><strong>migration guidance</strong></a>. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44029" class="reference external">bpo-44029</a>.)

</div>

</div>

<div id="notable-changes-in-3-11-4" class="section">

## Notable changes in 3.11.4<a href="#notable-changes-in-3-11-4" class="headerlink" title="Link to this heading">¶</a>

<div id="tarfile" class="section">

### tarfile<a href="#tarfile" class="headerlink" title="Link to this heading">¶</a>

- The extraction methods in <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>, and <a href="../library/shutil.html#shutil.unpack_archive" class="reference internal" title="shutil.unpack_archive"><span class="pre"><code class="sourceCode python">shutil.unpack_archive()</code></span></a>, have a new a *filter* argument that allows limiting tar features than may be surprising or dangerous, such as creating files outside the destination directory. See <a href="../library/tarfile.html#tarfile-extraction-filter" class="reference internal"><span class="std std-ref">Extraction filters</span></a> for details. In Python 3.12, use without the *filter* argument will show a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. In Python 3.14, the default will switch to <span class="pre">`'data'`</span>. (Contributed by Petr Viktorin in <span id="index-42" class="target"></span><a href="https://peps.python.org/pep-0706/" class="pep reference external"><strong>PEP 706</strong></a>.)

</div>

</div>

<div id="notable-changes-in-3-11-5" class="section">

## Notable changes in 3.11.5<a href="#notable-changes-in-3-11-5" class="headerlink" title="Link to this heading">¶</a>

<div id="openssl" class="section">

### OpenSSL<a href="#openssl" class="headerlink" title="Link to this heading">¶</a>

- Windows builds and macOS installers from python.org now use OpenSSL 3.0.

</div>

</div>

</div>

<div class="clearer">

</div>

</div>
