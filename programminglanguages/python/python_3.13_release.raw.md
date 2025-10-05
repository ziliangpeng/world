<div class="body" role="main">

<div id="what-s-new-in-python-3-13" class="section">

# What’s New In Python 3.13<a href="#what-s-new-in-python-3-13" class="headerlink" title="Link to this heading">¶</a>

Editors<span class="colon">:</span>  
Adam Turner and Thomas Wouters

This article explains the new features in Python 3.13, compared to 3.12. Python 3.13 was released on October 7, 2024. For full details, see the <a href="changelog.html#changelog" class="reference internal"><span class="std std-ref">changelog</span></a>.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0719/" class="pep reference external"><strong>PEP 719</strong></a> – Python 3.13 Release Schedule

</div>

<div id="summary-release-highlights" class="section">

## Summary – Release Highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

Python 3.13 is the latest stable release of the Python programming language, with a mix of changes to the language, the implementation and the standard library. The biggest changes include a new <a href="#whatsnew313-better-interactive-interpreter" class="reference internal">interactive interpreter</a>, experimental support for running in a <a href="#whatsnew313-free-threaded-cpython" class="reference internal">free-threaded mode</a> (<span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0703/" class="pep reference external"><strong>PEP 703</strong></a>), and a <a href="#whatsnew313-jit-compiler" class="reference internal">Just-In-Time compiler</a> (<span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0744/" class="pep reference external"><strong>PEP 744</strong></a>).

Error messages continue to improve, with tracebacks now highlighted in color by default. The <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> builtin now has <a href="#whatsnew313-locals-semantics" class="reference internal"><span class="std std-ref">defined semantics</span></a> for changing the returned mapping, and type parameters now support default values.

The library changes contain removal of deprecated APIs and modules, as well as the usual improvements in user-friendliness and correctness. Several legacy standard library modules have now <a href="#whatsnew313-pep594" class="reference internal">been removed</a> following their deprecation in Python 3.11 (<span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0594/" class="pep reference external"><strong>PEP 594</strong></a>).

This article doesn’t attempt to provide a complete specification of all new features, but instead gives a convenient overview. For full details refer to the documentation, such as the <a href="../library/index.html#library-index" class="reference internal"><span class="std std-ref">Library Reference</span></a> and <a href="../reference/index.html#reference-index" class="reference internal"><span class="std std-ref">Language Reference</span></a>. To understand the complete implementation and design rationale for a change, refer to the PEP for a particular new feature; but note that PEPs usually are not kept up-to-date once a feature has been fully implemented. See <a href="#porting-to-python-3-13" class="reference internal">Porting to Python 3.13</a> for guidance on upgrading from earlier versions of Python.

------------------------------------------------------------------------

Interpreter improvements:

- A greatly improved <a href="#whatsnew313-better-interactive-interpreter" class="reference internal"><span class="std std-ref">interactive interpreter</span></a> and <a href="#whatsnew313-improved-error-messages" class="reference internal"><span class="std std-ref">improved error messages</span></a>.

- <span id="index-4" class="target"></span><a href="https://peps.python.org/pep-0667/" class="pep reference external"><strong>PEP 667</strong></a>: The <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> builtin now has <a href="#whatsnew313-locals-semantics" class="reference internal"><span class="std std-ref">defined semantics</span></a> when mutating the returned mapping. Python debuggers and similar tools may now more reliably update local variables in optimized scopes even during concurrent code execution.

- <span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0703/" class="pep reference external"><strong>PEP 703</strong></a>: CPython 3.13 has experimental support for running with the <a href="../glossary.html#term-global-interpreter-lock" class="reference internal"><span class="xref std std-term">global interpreter lock</span></a> disabled. See <a href="#whatsnew313-free-threaded-cpython" class="reference internal"><span class="std std-ref">Free-threaded CPython</span></a> for more details.

- <span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0744/" class="pep reference external"><strong>PEP 744</strong></a>: A basic <a href="#whatsnew313-jit-compiler" class="reference internal"><span class="std std-ref">JIT compiler</span></a> was added. It is currently disabled by default (though we may turn it on later). Performance improvements are modest – we expect to improve this over the next few releases.

- Color support in the new <a href="#whatsnew313-better-interactive-interpreter" class="reference internal"><span class="std std-ref">interactive interpreter</span></a>, as well as in <a href="#whatsnew313-improved-error-messages" class="reference internal"><span class="std std-ref">tracebacks</span></a> and <a href="#whatsnew313-doctest" class="reference internal"><span class="std std-ref">doctest</span></a> output. This can be disabled through the <span id="index-7" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_COLORS" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_COLORS</code></span></a> and <a href="https://no-color.org/" class="reference external"><span class="pre"><code class="docutils literal notranslate">NO_COLOR</code></span></a> environment variables.

Python data model improvements:

- <a href="../reference/datamodel.html#type.__static_attributes__" class="reference internal" title="type.__static_attributes__"><span class="pre"><code class="sourceCode python">__static_attributes__</code></span></a> stores the names of attributes accessed through <span class="pre">`self.X`</span> in any function in a class body.

- <a href="../reference/datamodel.html#type.__firstlineno__" class="reference internal" title="type.__firstlineno__"><span class="pre"><code class="sourceCode python">__firstlineno__</code></span></a> records the first line number of a class definition.

Significant improvements in the standard library:

- Add a new <a href="../library/exceptions.html#PythonFinalizationError" class="reference internal" title="PythonFinalizationError"><span class="pre"><code class="sourceCode python">PythonFinalizationError</code></span></a> exception, raised when an operation is blocked during <a href="../glossary.html#term-interpreter-shutdown" class="reference internal"><span class="xref std std-term">finalization</span></a>.

- The <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> module now supports deprecating command-line options, positional arguments, and subcommands.

- The new functions <a href="../library/base64.html#base64.z85encode" class="reference internal" title="base64.z85encode"><span class="pre"><code class="sourceCode python">base64.z85encode()</code></span></a> and <a href="../library/base64.html#base64.z85decode" class="reference internal" title="base64.z85decode"><span class="pre"><code class="sourceCode python">base64.z85decode()</code></span></a> support encoding and decoding <a href="https://rfc.zeromq.org/spec/32/" class="reference external">Z85 data</a>.

- The <a href="../library/copy.html#module-copy" class="reference internal" title="copy: Shallow and deep copy operations."><span class="pre"><code class="sourceCode python">copy</code></span></a> module now has a <a href="../library/copy.html#copy.replace" class="reference internal" title="copy.replace"><span class="pre"><code class="sourceCode python">copy.replace()</code></span></a> function, with support for many builtin types and any class defining the <a href="../library/copy.html#object.__replace__" class="reference internal" title="object.__replace__"><span class="pre"><code class="sourceCode python">__replace__()</code></span></a> method.

- The new <a href="../library/dbm.html#module-dbm.sqlite3" class="reference internal" title="dbm.sqlite3: SQLite backend for dbm (All)"><span class="pre"><code class="sourceCode python">dbm.sqlite3</code></span></a> module is now the default <a href="../library/dbm.html#module-dbm" class="reference internal" title="dbm: Interfaces to various Unix &quot;database&quot; formats."><span class="pre"><code class="sourceCode python">dbm</code></span></a> backend.

- The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module has a <a href="../library/os.html#os-timerfd" class="reference internal"><span class="std std-ref">suite of new functions</span></a> for working with Linux’s timer notification file descriptors.

- The <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module now has a <a href="../library/random.html#random-cli" class="reference internal"><span class="std std-ref">command-line interface</span></a>.

Security improvements:

- <a href="../library/ssl.html#ssl.create_default_context" class="reference internal" title="ssl.create_default_context"><span class="pre"><code class="sourceCode python">ssl.create_default_context()</code></span></a> sets <a href="../library/ssl.html#ssl.VERIFY_X509_PARTIAL_CHAIN" class="reference internal" title="ssl.VERIFY_X509_PARTIAL_CHAIN"><span class="pre"><code class="sourceCode python">ssl.VERIFY_X509_PARTIAL_CHAIN</code></span></a> and <a href="../library/ssl.html#ssl.VERIFY_X509_STRICT" class="reference internal" title="ssl.VERIFY_X509_STRICT"><span class="pre"><code class="sourceCode python">ssl.VERIFY_X509_STRICT</code></span></a> as default flags.

C API improvements:

- The <a href="../c-api/module.html#c.Py_mod_gil" class="reference internal" title="Py_mod_gil"><span class="pre"><code class="sourceCode c">Py_mod_gil</code></span></a> slot is now used to indicate that an extension module supports running with the <a href="../glossary.html#term-GIL" class="reference internal"><span class="xref std std-term">GIL</span></a> disabled.

- The <a href="../c-api/time.html" class="reference internal"><span class="doc">PyTime C API</span></a> has been added, providing access to system clocks.

- <a href="../c-api/init.html#c.PyMutex" class="reference internal" title="PyMutex"><span class="pre"><code class="sourceCode c">PyMutex</code></span></a> is a new lightweight mutex that occupies a single byte.

- There is a new <a href="../c-api/monitoring.html#c-api-monitoring" class="reference internal"><span class="std std-ref">suite of functions</span></a> for generating <span id="index-8" class="target"></span><a href="https://peps.python.org/pep-0669/" class="pep reference external"><strong>PEP 669</strong></a> monitoring events in the C API.

New typing features:

- <span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0696/" class="pep reference external"><strong>PEP 696</strong></a>: Type parameters (<a href="../library/typing.html#typing.TypeVar" class="reference internal" title="typing.TypeVar"><span class="pre"><code class="sourceCode python">typing.TypeVar</code></span></a>, <a href="../library/typing.html#typing.ParamSpec" class="reference internal" title="typing.ParamSpec"><span class="pre"><code class="sourceCode python">typing.ParamSpec</code></span></a>, and <a href="../library/typing.html#typing.TypeVarTuple" class="reference internal" title="typing.TypeVarTuple"><span class="pre"><code class="sourceCode python">typing.TypeVarTuple</code></span></a>) now support defaults.

- <span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0702/" class="pep reference external"><strong>PEP 702</strong></a>: The new <a href="../library/warnings.html#warnings.deprecated" class="reference internal" title="warnings.deprecated"><span class="pre"><code class="sourceCode python">warnings.deprecated()</code></span></a> decorator adds support for marking deprecations in the type system and at runtime.

- <span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0705/" class="pep reference external"><strong>PEP 705</strong></a>: <a href="../library/typing.html#typing.ReadOnly" class="reference internal" title="typing.ReadOnly"><span class="pre"><code class="sourceCode python">typing.ReadOnly</code></span></a> can be used to mark an item of a <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">typing.TypedDict</code></span></a> as read-only for type checkers.

- <span id="index-12" class="target"></span><a href="https://peps.python.org/pep-0742/" class="pep reference external"><strong>PEP 742</strong></a>: <a href="../library/typing.html#typing.TypeIs" class="reference internal" title="typing.TypeIs"><span class="pre"><code class="sourceCode python">typing.TypeIs</code></span></a> provides more intuitive type narrowing behavior, as an alternative to <a href="../library/typing.html#typing.TypeGuard" class="reference internal" title="typing.TypeGuard"><span class="pre"><code class="sourceCode python">typing.TypeGuard</code></span></a>.

Platform support:

- <span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0730/" class="pep reference external"><strong>PEP 730</strong></a>: Apple’s iOS is now an <a href="#whatsnew313-platform-support" class="reference internal"><span class="std std-ref">officially supported platform</span></a>, at <span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0011/#tier-3" class="pep reference external"><strong>tier 3</strong></a>.

- <span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0738/" class="pep reference external"><strong>PEP 738</strong></a>: Android is now an <a href="#whatsnew313-platform-support" class="reference internal"><span class="std std-ref">officially supported platform</span></a>, at <span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0011/#tier-3" class="pep reference external"><strong>tier 3</strong></a>.

- <span class="pre">`wasm32-wasi`</span> is now supported as a <span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0011/#tier-2" class="pep reference external"><strong>tier 2</strong></a> platform.

- <span class="pre">`wasm32-emscripten`</span> is no longer an officially supported platform.

Important removals:

- <a href="#whatsnew313-pep594" class="reference internal"><span class="std std-ref">PEP 594</span></a>: The remaining 19 “dead batteries” (legacy stdlib modules) have been removed from the standard library: <span class="pre">`aifc`</span>, <span class="pre">`audioop`</span>, <span class="pre">`cgi`</span>, <span class="pre">`cgitb`</span>, <span class="pre">`chunk`</span>, <span class="pre">`crypt`</span>, <span class="pre">`imghdr`</span>, <span class="pre">`mailcap`</span>, <span class="pre">`msilib`</span>, <span class="pre">`nis`</span>, <span class="pre">`nntplib`</span>, <span class="pre">`ossaudiodev`</span>, <span class="pre">`pipes`</span>, <span class="pre">`sndhdr`</span>, <span class="pre">`spwd`</span>, <span class="pre">`sunau`</span>, <span class="pre">`telnetlib`</span>, <span class="pre">`uu`</span> and <span class="pre">`xdrlib`</span>.

- Remove the **2to3** tool and <span class="pre">`lib2to3`</span> module (deprecated in Python 3.11).

- Remove the <span class="pre">`tkinter.tix`</span> module (deprecated in Python 3.6).

- Remove the <span class="pre">`locale.resetlocale()`</span> function.

- Remove the <span class="pre">`typing.io`</span> and <span class="pre">`typing.re`</span> namespaces.

- Remove chained <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span></code></span></a> descriptors.

Release schedule changes:

<span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0602/" class="pep reference external"><strong>PEP 602</strong></a> (“Annual Release Cycle for Python”) has been updated to extend the full support (‘bugfix’) period for new releases to two years. This updated policy means that:

- Python 3.9–3.12 have one and a half years of full support, followed by three and a half years of security fixes.

- Python 3.13 and later have two years of full support, followed by three years of security fixes.

</div>

<div id="new-features" class="section">

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="a-better-interactive-interpreter" class="section">

<span id="whatsnew313-better-interactive-interpreter"></span>

### A better interactive interpreter<a href="#a-better-interactive-interpreter" class="headerlink" title="Link to this heading">¶</a>

Python now uses a new <a href="../glossary.html#term-interactive" class="reference internal"><span class="xref std std-term">interactive</span></a> shell by default, based on code from the <a href="https://pypy.org/" class="reference external">PyPy project</a>. When the user starts the <a href="../glossary.html#term-REPL" class="reference internal"><span class="xref std std-term">REPL</span></a> from an interactive terminal, the following new features are now supported:

- Multiline editing with history preservation.

- Direct support for REPL-specific commands like <span class="kbd kbd docutils literal notranslate">help</span>, <span class="kbd kbd docutils literal notranslate">exit</span>, and <span class="kbd kbd docutils literal notranslate">quit</span>, without the need to call them as functions.

- Prompts and tracebacks with <a href="../using/cmdline.html#using-on-controlling-color" class="reference internal"><span class="std std-ref">color enabled by default</span></a>.

- Interactive help browsing using <span class="kbd kbd docutils literal notranslate">F1</span> with a separate command history.

- History browsing using <span class="kbd kbd docutils literal notranslate">F2</span> that skips output as well as the <a href="../glossary.html#term-0" class="reference internal"><span class="xref std std-term">&gt;&gt;&gt;</span></a> and <a href="../glossary.html#term-..." class="reference internal"><span class="xref std std-term">…</span></a> prompts.

- “Paste mode” with <span class="kbd kbd docutils literal notranslate">F3</span> that makes pasting larger blocks of code easier (press <span class="kbd kbd docutils literal notranslate">F3</span> again to return to the regular prompt).

To disable the new interactive shell, set the <span id="index-19" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_BASIC_REPL" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_BASIC_REPL</code></span></a> environment variable. For more on interactive mode, see <a href="../tutorial/appendix.html#tut-interac" class="reference internal"><span class="std std-ref">Interactive Mode</span></a>.

(Contributed by Pablo Galindo Salgado, Łukasz Langa, and Lysandros Nikolaou in <a href="https://github.com/python/cpython/issues/111201" class="reference external">gh-111201</a> based on code from the PyPy project. Windows support contributed by Dino Viehland and Anthony Shaw.)

</div>

<div id="improved-error-messages" class="section">

<span id="whatsnew313-improved-error-messages"></span>

### Improved error messages<a href="#improved-error-messages" class="headerlink" title="Link to this heading">¶</a>

- The interpreter now uses color by default when displaying tracebacks in the terminal. This feature <a href="../using/cmdline.html#using-on-controlling-color" class="reference internal"><span class="std std-ref">can be controlled</span></a> via the new <span id="index-20" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_COLORS" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_COLORS</code></span></a> environment variable as well as the canonical <a href="https://no-color.org/" class="reference external"><span class="pre"><code class="docutils literal notranslate">NO_COLOR</code></span></a> and <a href="https://force-color.org/" class="reference external"><span class="pre"><code class="docutils literal notranslate">FORCE_COLOR</code></span></a> environment variables. (Contributed by Pablo Galindo Salgado in <a href="https://github.com/python/cpython/issues/112730" class="reference external">gh-112730</a>.)

<!-- -->

- A common mistake is to write a script with the same name as a standard library module. When this results in errors, we now display a more helpful error message:

  <div class="highlight-pytb notranslate">

  <div class="highlight">

      $ python random.py
      Traceback (most recent call last):
        File "/home/me/random.py", line 1, in <module>
          import random
        File "/home/me/random.py", line 3, in <module>
          print(random.randint(5))
                ^^^^^^^^^^^^^^
      AttributeError: module 'random' has no attribute 'randint' (consider renaming '/home/me/random.py' since it has the same name as the standard library module named 'random' and prevents importing that standard library module)

  </div>

  </div>

  Similarly, if a script has the same name as a third-party module that it attempts to import and this results in errors, we also display a more helpful error message:

  <div class="highlight-pytb notranslate">

  <div class="highlight">

      $ python numpy.py
      Traceback (most recent call last):
        File "/home/me/numpy.py", line 1, in <module>
          import numpy as np
        File "/home/me/numpy.py", line 3, in <module>
          np.array([1, 2, 3])
          ^^^^^^^^
      AttributeError: module 'numpy' has no attribute 'array' (consider renaming '/home/me/numpy.py' if it has the same name as a library you intended to import)

  </div>

  </div>

  (Contributed by Shantanu Jain in <a href="https://github.com/python/cpython/issues/95754" class="reference external">gh-95754</a>.)

- The error message now tries to suggest the correct keyword argument when an incorrect keyword argument is passed to a function.

  <div class="highlight-pycon notranslate">

  <div class="highlight">

      >>> "Better error messages!".split(max_split=1)
      Traceback (most recent call last):
        File "<python-input-0>", line 1, in <module>
          "Better error messages!".split(max_split=1)
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
      TypeError: split() got an unexpected keyword argument 'max_split'. Did you mean 'maxsplit'?

  </div>

  </div>

  (Contributed by Pablo Galindo Salgado and Shantanu Jain in <a href="https://github.com/python/cpython/issues/107944" class="reference external">gh-107944</a>.)

</div>

<div id="free-threaded-cpython" class="section">

<span id="whatsnew313-free-threaded-cpython"></span>

### Free-threaded CPython<a href="#free-threaded-cpython" class="headerlink" title="Link to this heading">¶</a>

CPython now has experimental support for running in a free-threaded mode, with the <a href="../glossary.html#term-global-interpreter-lock" class="reference internal"><span class="xref std std-term">global interpreter lock</span></a> (GIL) disabled. This is an experimental feature and therefore is not enabled by default. The free-threaded mode requires a different executable, usually called <span class="pre">`python3.13t`</span> or <span class="pre">`python3.13t.exe`</span>. Pre-built binaries marked as *free-threaded* can be installed as part of the official <a href="../using/windows.html#install-freethreaded-windows" class="reference internal"><span class="std std-ref">Windows</span></a> and <a href="../using/mac.html#install-freethreaded-macos" class="reference internal"><span class="std std-ref">macOS</span></a> installers, or CPython can be built from source with the <a href="../using/configure.html#cmdoption-disable-gil" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--disable-gil</code></span></a> option.

Free-threaded execution allows for full utilization of the available processing power by running threads in parallel on available CPU cores. While not all software will benefit from this automatically, programs designed with threading in mind will run faster on multi-core hardware. **The free-threaded mode is experimental** and work is ongoing to improve it: expect some bugs and a substantial single-threaded performance hit. Free-threaded builds of CPython support optionally running with the GIL enabled at runtime using the environment variable <span id="index-21" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_GIL" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_GIL</code></span></a> or the command-line option <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">gil=1</code></span></a>.

To check if the current interpreter supports free-threading, <a href="../using/cmdline.html#cmdoption-V" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">python</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">-VV</code></span></a> and <a href="../library/sys.html#sys.version" class="reference internal" title="sys.version"><span class="pre"><code class="sourceCode python">sys.version</code></span></a> contain “experimental free-threading build”. The new <span class="pre">`sys._is_gil_enabled()`</span> function can be used to check whether the GIL is actually disabled in the running process.

C-API extension modules need to be built specifically for the free-threaded build. Extensions that support running with the <a href="../glossary.html#term-GIL" class="reference internal"><span class="xref std std-term">GIL</span></a> disabled should use the <a href="../c-api/module.html#c.Py_mod_gil" class="reference internal" title="Py_mod_gil"><span class="pre"><code class="sourceCode c">Py_mod_gil</code></span></a> slot. Extensions using single-phase init should use <a href="../c-api/module.html#c.PyUnstable_Module_SetGIL" class="reference internal" title="PyUnstable_Module_SetGIL"><span class="pre"><code class="sourceCode c">PyUnstable_Module_SetGIL<span class="op">()</span></code></span></a> to indicate whether they support running with the GIL disabled. Importing C extensions that don’t use these mechanisms will cause the GIL to be enabled, unless the GIL was explicitly disabled with the <span id="index-22" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_GIL" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_GIL</code></span></a> environment variable or the <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">gil=0</code></span></a> option. pip 24.1 or newer is required to install packages with C extensions in the free-threaded build.

This work was made possible thanks to many individuals and organizations, including the large community of contributors to Python and third-party projects to test and enable free-threading support. Notable contributors include: Sam Gross, Ken Jin, Donghee Na, Itamar Oren, Matt Page, Brett Simmers, Dino Viehland, Carl Meyer, Nathan Goldbaum, Ralf Gommers, Lysandros Nikolaou, and many others. Many of these contributors are employed by Meta, which has provided significant engineering resources to support this project.

<div class="admonition seealso">

See also

<span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0703/" class="pep reference external"><strong>PEP 703</strong></a> “Making the Global Interpreter Lock Optional in CPython” contains rationale and information surrounding this work.

<a href="https://py-free-threading.github.io/porting/" class="reference external">Porting Extension Modules to Support Free-Threading</a>: A community-maintained porting guide for extension authors.

</div>

</div>

<div id="an-experimental-just-in-time-jit-compiler" class="section">

<span id="whatsnew313-jit-compiler"></span>

### An experimental just-in-time (JIT) compiler<a href="#an-experimental-just-in-time-jit-compiler" class="headerlink" title="Link to this heading">¶</a>

When CPython is configured and built using the <span class="pre">`--enable-experimental-jit`</span> option, a just-in-time (JIT) compiler is added which may speed up some Python programs. On Windows, use <span class="pre">`PCbuild/build.bat`</span>` `<span class="pre">`--experimental-jit`</span> to enable the JIT or <span class="pre">`--experimental-jit-interpreter`</span> to enable the Tier 2 interpreter. Build requirements and further supporting information <a href="https://github.com/python/cpython/blob/main/Tools/jit/README.md" class="reference external">are contained at</a> <span class="pre">`Tools/jit/README.md`</span>.

The <span class="pre">`--enable-experimental-jit`</span> option takes these (optional) values, defaulting to <span class="pre">`yes`</span> if <span class="pre">`--enable-experimental-jit`</span> is present without the optional value.

- <span class="pre">`no`</span>: Disable the entire Tier 2 and JIT pipeline.

- <span class="pre">`yes`</span>: Enable the JIT. To disable the JIT at runtime, pass the environment variable <span id="index-24" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_JIT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_JIT=0</code></span></a>.

- <span class="pre">`yes-off`</span>: Build the JIT but disable it by default. To enable the JIT at runtime, pass the environment variable <span id="index-25" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_JIT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_JIT=1</code></span></a>.

- <span class="pre">`interpreter`</span>: Enable the Tier 2 interpreter but disable the JIT. The interpreter can be disabled by running with <span id="index-26" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_JIT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_JIT=0</code></span></a>.

The internal architecture is roughly as follows:

- We start with specialized *Tier 1 bytecode*. See <a href="3.11.html#whatsnew311-pep659" class="reference internal"><span class="std std-ref">What’s new in 3.11</span></a> for details.

- When the Tier 1 bytecode gets hot enough, it gets translated to a new purely internal intermediate representation (IR), called the *Tier 2 IR*, and sometimes referred to as micro-ops (“uops”).

- The Tier 2 IR uses the same stack-based virtual machine as Tier 1, but the instruction format is better suited to translation to machine code.

- We have several optimization passes for Tier 2 IR, which are applied before it is interpreted or translated to machine code.

- There is a Tier 2 interpreter, but it is mostly intended for debugging the earlier stages of the optimization pipeline. The Tier 2 interpreter can be enabled by configuring Python with <span class="pre">`--enable-experimental-jit=interpreter`</span>.

- When the JIT is enabled, the optimized Tier 2 IR is translated to machine code, which is then executed.

- The machine code translation process uses a technique called *copy-and-patch*. It has no runtime dependencies, but there is a new build-time dependency on LLVM.

<div class="admonition seealso">

See also

<span id="index-27" class="target"></span><a href="https://peps.python.org/pep-0744/" class="pep reference external"><strong>PEP 744</strong></a>

</div>

(JIT by Brandt Bucher, inspired by a paper by Haoran Xu and Fredrik Kjolstad. Tier 2 IR by Mark Shannon and Guido van Rossum. Tier 2 optimizer by Ken Jin.)

</div>

<div id="defined-mutation-semantics-for-locals" class="section">

<span id="whatsnew313-locals-semantics"></span>

### Defined mutation semantics for <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a><a href="#defined-mutation-semantics-for-locals" class="headerlink" title="Link to this heading">¶</a>

Historically, the expected result of mutating the return value of <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> has been left to individual Python implementations to define. Starting from Python 3.13, <span id="index-28" class="target"></span><a href="https://peps.python.org/pep-0667/" class="pep reference external"><strong>PEP 667</strong></a> standardises the historical behavior of CPython for most code execution scopes, but changes <a href="../glossary.html#term-optimized-scope" class="reference internal"><span class="xref std std-term">optimized scopes</span></a> (functions, generators, coroutines, comprehensions, and generator expressions) to explicitly return independent snapshots of the currently assigned local variables, including locally referenced nonlocal variables captured in closures.

This change to the semantics of <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> in optimized scopes also affects the default behavior of code execution functions that implicitly target <span class="pre">`locals()`</span> if no explicit namespace is provided (such as <a href="../library/functions.html#exec" class="reference internal" title="exec"><span class="pre"><code class="sourceCode python"><span class="bu">exec</span>()</code></span></a> and <a href="../library/functions.html#eval" class="reference internal" title="eval"><span class="pre"><code class="sourceCode python"><span class="bu">eval</span>()</code></span></a>). In previous versions, whether or not changes could be accessed by calling <span class="pre">`locals()`</span> after calling the code execution function was implementation-dependent. In CPython specifically, such code would typically appear to work as desired, but could sometimes fail in optimized scopes based on other code (including debuggers and code execution tracing tools) potentially resetting the shared snapshot in that scope. Now, the code will always run against an independent snapshot of the local variables in optimized scopes, and hence the changes will never be visible in subsequent calls to <span class="pre">`locals()`</span>. To access the changes made in these cases, an explicit namespace reference must now be passed to the relevant function. Alternatively, it may make sense to update affected code to use a higher level code execution API that returns the resulting code execution namespace (e.g. <a href="../library/runpy.html#runpy.run_path" class="reference internal" title="runpy.run_path"><span class="pre"><code class="sourceCode python">runpy.run_path()</code></span></a> when executing Python files from disk).

To ensure debuggers and similar tools can reliably update local variables in scopes affected by this change, <a href="../reference/datamodel.html#frame.f_locals" class="reference internal" title="frame.f_locals"><span class="pre"><code class="sourceCode python">FrameType.f_locals</code></span></a> now returns a write-through proxy to the frame’s local and locally referenced nonlocal variables in these scopes, rather than returning an inconsistently updated shared <span class="pre">`dict`</span> instance with undefined runtime semantics.

See <span id="index-29" class="target"></span><a href="https://peps.python.org/pep-0667/" class="pep reference external"><strong>PEP 667</strong></a> for more details, including related C API changes and deprecations. Porting notes are also provided below for the affected <a href="#pep667-porting-notes-py" class="reference internal"><span class="std std-ref">Python APIs</span></a> and <a href="#pep667-porting-notes-c" class="reference internal"><span class="std std-ref">C APIs</span></a>.

(PEP and implementation contributed by Mark Shannon and Tian Gao in <a href="https://github.com/python/cpython/issues/74929" class="reference external">gh-74929</a>. Documentation updates provided by Guido van Rossum and Alyssa Coghlan.)

</div>

<div id="support-for-mobile-platforms" class="section">

<span id="whatsnew313-platform-support"></span>

### Support for mobile platforms<a href="#support-for-mobile-platforms" class="headerlink" title="Link to this heading">¶</a>

<span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0730/" class="pep reference external"><strong>PEP 730</strong></a>: iOS is now a <span id="index-31" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a> supported platform, with the <span class="pre">`arm64-apple-ios`</span> and <span class="pre">`arm64-apple-ios-simulator`</span> targets at tier 3 (iPhone and iPad devices released after 2013 and the Xcode iOS simulator running on Apple silicon hardware, respectively). <span class="pre">`x86_64-apple-ios-simulator`</span> (the Xcode iOS simulator running on older <span class="pre">`x86_64`</span> hardware) is not a tier 3 supported platform, but will have best-effort support. (PEP written and implementation contributed by Russell Keith-Magee in <a href="https://github.com/python/cpython/issues/114099" class="reference external">gh-114099</a>.)

<span id="index-32" class="target"></span><a href="https://peps.python.org/pep-0738/" class="pep reference external"><strong>PEP 738</strong></a>: Android is now a <span id="index-33" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a> supported platform, with the <span class="pre">`aarch64-linux-android`</span> and <span class="pre">`x86_64-linux-android`</span> targets at tier 3. The 32-bit targets <span class="pre">`arm-linux-androideabi`</span> and <span class="pre">`i686-linux-android`</span> are not tier 3 supported platforms, but will have best-effort support. (PEP written and implementation contributed by Malcolm Smith in <a href="https://github.com/python/cpython/issues/116622" class="reference external">gh-116622</a>.)

<div class="admonition seealso">

See also

<span id="index-34" class="target"></span><a href="https://peps.python.org/pep-0730/" class="pep reference external"><strong>PEP 730</strong></a>, <span id="index-35" class="target"></span><a href="https://peps.python.org/pep-0738/" class="pep reference external"><strong>PEP 738</strong></a>

</div>

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

- The compiler now strips common leading whitespace from every line in a docstring. This reduces the size of the <a href="../glossary.html#term-bytecode" class="reference internal"><span class="xref std std-term">bytecode cache</span></a> (such as <span class="pre">`.pyc`</span> files), with reductions in file size of around 5%, for example in <span class="pre">`sqlalchemy.orm.session`</span> from SQLAlchemy 2.0. This change affects tools that use docstrings, such as <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a>.

  <div class="highlight-pycon notranslate">

  <div class="highlight">

      >>> def spam():
      ...     """
      ...         This is a docstring with
      ...           leading whitespace.
      ...
      ...         It even has multiple paragraphs!
      ...     """
      ...
      >>> spam.__doc__
      '\nThis is a docstring with\n  leading whitespace.\n\nIt even has multiple paragraphs!\n'

  </div>

  </div>

  (Contributed by Inada Naoki in <a href="https://github.com/python/cpython/issues/81283" class="reference external">gh-81283</a>.)

- <a href="../reference/executionmodel.html#annotation-scopes" class="reference internal"><span class="std std-ref">Annotation scopes</span></a> within class scopes can now contain lambdas and comprehensions. Comprehensions that are located within class scopes are not inlined into their parent scope.

  <div class="highlight-python notranslate">

  <div class="highlight">

      class C[T]:
          type Alias = lambda: T

  </div>

  </div>

  (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/109118" class="reference external">gh-109118</a> and <a href="https://github.com/python/cpython/issues/118160" class="reference external">gh-118160</a>.)

- <a href="../reference/simple_stmts.html#future" class="reference internal"><span class="std std-ref">Future statements</span></a> are no longer triggered by relative imports of the <a href="../library/__future__.html#module-__future__" class="reference internal" title="__future__: Future statement definitions"><span class="pre"><code class="sourceCode python">__future__</code></span></a> module, meaning that statements of the form <span class="pre">`from`</span>` `<span class="pre">`.__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`...`</span> are now simply standard relative imports, with no special features activated. (Contributed by Jeremiah Gabriel Pascual in <a href="https://github.com/python/cpython/issues/118216" class="reference external">gh-118216</a>.)

- <a href="../reference/simple_stmts.html#global" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">global</code></span></a> declarations are now permitted in <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> blocks when that global is used in the <a href="../reference/compound_stmts.html#else" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">else</code></span></a> block. Previously this raised an erroneous <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/111123" class="reference external">gh-111123</a>.)

- Add <span id="index-36" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_FROZEN_MODULES" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_FROZEN_MODULES</code></span></a>, a new environment variable that determines whether frozen modules are ignored by the import machinery, equivalent to the <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">frozen_modules</code></span></a> command-line option. (Contributed by Yilei Yang in <a href="https://github.com/python/cpython/issues/111374" class="reference external">gh-111374</a>.)

- Add <a href="../howto/perf_profiling.html#perf-profiling" class="reference internal"><span class="std std-ref">support for the perf profiler</span></a> working without <a href="https://en.wikipedia.org/wiki/Call_stack" class="reference external">frame pointers</a> through the new environment variable <span id="index-37" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_PERF_JIT_SUPPORT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_PERF_JIT_SUPPORT</code></span></a> and command-line option <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">perf_jit</code></span></a>. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/118518" class="reference external">gh-118518</a>.)

- The location of a <span class="pre">`.python_history`</span> file can be changed via the new <span id="index-38" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_HISTORY" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_HISTORY</code></span></a> environment variable. (Contributed by Levi Sabah, Zackery Spytz and Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/73965" class="reference external">gh-73965</a>.)

- Classes have a new <a href="../reference/datamodel.html#type.__static_attributes__" class="reference internal" title="type.__static_attributes__"><span class="pre"><code class="sourceCode python">__static_attributes__</code></span></a> attribute. This is populated by the compiler with a tuple of the class’s attribute names which are assigned through <span class="pre">`self.<name>`</span> from any function in its body. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/115775" class="reference external">gh-115775</a>.)

- The compiler now creates a <span class="pre">`__firstlineno__`</span> attribute on classes with the line number of the first line of the class definition. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/118465" class="reference external">gh-118465</a>.)

- The <a href="../library/functions.html#exec" class="reference internal" title="exec"><span class="pre"><code class="sourceCode python"><span class="bu">exec</span>()</code></span></a> and <a href="../library/functions.html#eval" class="reference internal" title="eval"><span class="pre"><code class="sourceCode python"><span class="bu">eval</span>()</code></span></a> builtins now accept the *globals* and *locals* arguments as keywords. (Contributed by Raphael Gaschignard in <a href="https://github.com/python/cpython/issues/105879" class="reference external">gh-105879</a>)

- The <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a> builtin now accepts a new flag, <span class="pre">`ast.PyCF_OPTIMIZED_AST`</span>, which is similar to <span class="pre">`ast.PyCF_ONLY_AST`</span> except that the returned AST is optimized according to the value of the *optimize* argument. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/108113" class="reference external">gh-108113</a>).

- Add a <a href="../library/functions.html#property.__name__" class="reference internal" title="property.__name__"><span class="pre"><code class="sourceCode python"><span class="va">__name__</span></code></span></a> attribute on <a href="../library/functions.html#property" class="reference internal" title="property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span></code></span></a> objects. (Contributed by Eugene Toder in <a href="https://github.com/python/cpython/issues/101860" class="reference external">gh-101860</a>.)

- Add <a href="../library/exceptions.html#PythonFinalizationError" class="reference internal" title="PythonFinalizationError"><span class="pre"><code class="sourceCode python">PythonFinalizationError</code></span></a>, a new exception derived from <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> and used to signal when operations are blocked during <a href="../glossary.html#term-interpreter-shutdown" class="reference internal"><span class="xref std std-term">finalization</span></a>. The following callables now raise <span class="pre">`PythonFinalizationError`</span>, instead of <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a>:

  - <a href="../library/_thread.html#thread.start_new_thread" class="reference internal" title="_thread.start_new_thread"><span class="pre"><code class="sourceCode python">_thread.start_new_thread()</code></span></a>

  - <a href="../library/os.html#os.fork" class="reference internal" title="os.fork"><span class="pre"><code class="sourceCode python">os.fork()</code></span></a>

  - <a href="../library/os.html#os.forkpty" class="reference internal" title="os.forkpty"><span class="pre"><code class="sourceCode python">os.forkpty()</code></span></a>

  - <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a>

  (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/114570" class="reference external">gh-114570</a>.)

- Allow the *count* argument of <a href="../library/stdtypes.html#str.replace" class="reference internal" title="str.replace"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.replace()</code></span></a> to be a keyword. (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/106487" class="reference external">gh-106487</a>.)

- Many functions now emit a warning if a boolean value is passed as a file descriptor argument. This can help catch some errors earlier. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/82626" class="reference external">gh-82626</a>.)

- Added <span class="pre">`name`</span> and <span class="pre">`mode`</span> attributes for compressed and archived file-like objects in the <a href="../library/bz2.html#module-bz2" class="reference internal" title="bz2: Interfaces for bzip2 compression and decompression."><span class="pre"><code class="sourceCode python">bz2</code></span></a>, <a href="../library/lzma.html#module-lzma" class="reference internal" title="lzma: A Python wrapper for the liblzma compression library."><span class="pre"><code class="sourceCode python">lzma</code></span></a>, <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>, and <a href="../library/zipfile.html#module-zipfile" class="reference internal" title="zipfile: Read and write ZIP-format archive files."><span class="pre"><code class="sourceCode python">zipfile</code></span></a> modules. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/115961" class="reference external">gh-115961</a>.)

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/dbm.html#module-dbm.sqlite3" class="reference internal" title="dbm.sqlite3: SQLite backend for dbm (All)"><span class="pre"><code class="sourceCode python">dbm.sqlite3</code></span></a>: An SQLite backend for <a href="../library/dbm.html#module-dbm" class="reference internal" title="dbm: Interfaces to various Unix &quot;database&quot; formats."><span class="pre"><code class="sourceCode python">dbm</code></span></a>. (Contributed by Raymond Hettinger and Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/100414" class="reference external">gh-100414</a>.)

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="argparse" class="section">

### argparse<a href="#argparse" class="headerlink" title="Link to this heading">¶</a>

- Add the *deprecated* parameter to the <a href="../library/argparse.html#argparse.ArgumentParser.add_argument" class="reference internal" title="argparse.ArgumentParser.add_argument"><span class="pre"><code class="sourceCode python">add_argument()</code></span></a> and <span class="pre">`add_parser()`</span> methods, to enable deprecating command-line options, positional arguments, and subcommands. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/83648" class="reference external">gh-83648</a>.)

</div>

<div id="array" class="section">

### array<a href="#array" class="headerlink" title="Link to this heading">¶</a>

- Add the <span class="pre">`'w'`</span> type code (<span class="pre">`Py_UCS4`</span>) for Unicode characters. It should be used instead of the deprecated <span class="pre">`'u'`</span> type code. (Contributed by Inada Naoki in <a href="https://github.com/python/cpython/issues/80480" class="reference external">gh-80480</a>.)

- Register <a href="../library/array.html#array.array" class="reference internal" title="array.array"><span class="pre"><code class="sourceCode python">array.array</code></span></a> as a <a href="../library/collections.abc.html#collections.abc.MutableSequence" class="reference internal" title="collections.abc.MutableSequence"><span class="pre"><code class="sourceCode python">MutableSequence</code></span></a> by implementing the <a href="../library/array.html#array.array.clear" class="reference internal" title="array.array.clear"><span class="pre"><code class="sourceCode python">clear()</code></span></a> method. (Contributed by Mike Zimin in <a href="https://github.com/python/cpython/issues/114894" class="reference external">gh-114894</a>.)

</div>

<div id="ast" class="section">

### ast<a href="#ast" class="headerlink" title="Link to this heading">¶</a>

- The constructors of node types in the <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> module are now stricter in the arguments they accept, with more intuitive behavior when arguments are omitted.

  If an optional field on an AST node is not included as an argument when constructing an instance, the field will now be set to <span class="pre">`None`</span>. Similarly, if a list field is omitted, that field will now be set to an empty list, and if an <span class="pre">`expr_context`</span> field is omitted, it defaults to <a href="../library/ast.html#ast.Load" class="reference internal" title="ast.Load"><span class="pre"><code class="sourceCode python">Load()</code></span></a>. (Previously, in all cases, the attribute would be missing on the newly constructed AST node instance.)

  In all other cases, where a required argument is omitted, the node constructor will emit a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. This will raise an exception in Python 3.15. Similarly, passing a keyword argument to the constructor that does not map to a field on the AST node is now deprecated, and will raise an exception in Python 3.15.

  These changes do not apply to user-defined subclasses of <a href="../library/ast.html#ast.AST" class="reference internal" title="ast.AST"><span class="pre"><code class="sourceCode python">ast.AST</code></span></a> unless the class opts in to the new behavior by defining the <a href="../library/ast.html#ast.AST._field_types" class="reference internal" title="ast.AST._field_types"><span class="pre"><code class="sourceCode python">AST._field_types</code></span></a> mapping.

  (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/105858" class="reference external">gh-105858</a>, <a href="https://github.com/python/cpython/issues/117486" class="reference external">gh-117486</a>, and <a href="https://github.com/python/cpython/issues/118851" class="reference external">gh-118851</a>.)

- <a href="../library/ast.html#ast.parse" class="reference internal" title="ast.parse"><span class="pre"><code class="sourceCode python">ast.parse()</code></span></a> now accepts an optional argument *optimize* which is passed on to <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a>. This makes it possible to obtain an optimized AST. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/108113" class="reference external">gh-108113</a>.)

</div>

<div id="asyncio" class="section">

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/asyncio-task.html#asyncio.as_completed" class="reference internal" title="asyncio.as_completed"><span class="pre"><code class="sourceCode python">asyncio.as_completed()</code></span></a> now returns an object that is both an <a href="../glossary.html#term-asynchronous-iterator" class="reference internal"><span class="xref std std-term">asynchronous iterator</span></a> and a plain <a href="../glossary.html#term-iterator" class="reference internal"><span class="xref std std-term">iterator</span></a> of <a href="../glossary.html#term-awaitable" class="reference internal"><span class="xref std std-term">awaitables</span></a>. The awaitables yielded by asynchronous iteration include original task or future objects that were passed in, making it easier to associate results with the tasks being completed. (Contributed by Justin Arthur in <a href="https://github.com/python/cpython/issues/77714" class="reference external">gh-77714</a>.)

- <a href="../library/asyncio-eventloop.html#asyncio.loop.create_unix_server" class="reference internal" title="asyncio.loop.create_unix_server"><span class="pre"><code class="sourceCode python">asyncio.loop.create_unix_server()</code></span></a> will now automatically remove the Unix socket when the server is closed. (Contributed by Pierre Ossman in <a href="https://github.com/python/cpython/issues/111246" class="reference external">gh-111246</a>.)

- <a href="../library/asyncio-protocol.html#asyncio.DatagramTransport.sendto" class="reference internal" title="asyncio.DatagramTransport.sendto"><span class="pre"><code class="sourceCode python">DatagramTransport.sendto()</code></span></a> will now send zero-length datagrams if called with an empty bytes object. The transport flow control also now accounts for the datagram header when calculating the buffer size. (Contributed by Jamie Phan in <a href="https://github.com/python/cpython/issues/115199" class="reference external">gh-115199</a>.)

- Add <a href="../library/asyncio-queue.html#asyncio.Queue.shutdown" class="reference internal" title="asyncio.Queue.shutdown"><span class="pre"><code class="sourceCode python">Queue.shutdown</code></span></a> and <a href="../library/asyncio-queue.html#asyncio.QueueShutDown" class="reference internal" title="asyncio.QueueShutDown"><span class="pre"><code class="sourceCode python">QueueShutDown</code></span></a> to manage queue termination. (Contributed by Laurie Opperman and Yves Duprat in <a href="https://github.com/python/cpython/issues/104228" class="reference external">gh-104228</a>.)

- Add the <a href="../library/asyncio-eventloop.html#asyncio.Server.close_clients" class="reference internal" title="asyncio.Server.close_clients"><span class="pre"><code class="sourceCode python">Server.close_clients()</code></span></a> and <a href="../library/asyncio-eventloop.html#asyncio.Server.abort_clients" class="reference internal" title="asyncio.Server.abort_clients"><span class="pre"><code class="sourceCode python">Server.abort_clients()</code></span></a> methods, which more forcefully close an asyncio server. (Contributed by Pierre Ossman in <a href="https://github.com/python/cpython/issues/113538" class="reference external">gh-113538</a>.)

- Accept a tuple of separators in <a href="../library/asyncio-stream.html#asyncio.StreamReader.readuntil" class="reference internal" title="asyncio.StreamReader.readuntil"><span class="pre"><code class="sourceCode python">StreamReader.readuntil()</code></span></a>, stopping when any one of them is encountered. (Contributed by Bruce Merry in <a href="https://github.com/python/cpython/issues/81322" class="reference external">gh-81322</a>.)

- Improve the behavior of <a href="../library/asyncio-task.html#asyncio.TaskGroup" class="reference internal" title="asyncio.TaskGroup"><span class="pre"><code class="sourceCode python">TaskGroup</code></span></a> when an external cancellation collides with an internal cancellation. For example, when two task groups are nested and both experience an exception in a child task simultaneously, it was possible that the outer task group would hang, because its internal cancellation was swallowed by the inner task group.

  In the case where a task group is cancelled externally and also must raise an <a href="../library/exceptions.html#ExceptionGroup" class="reference internal" title="ExceptionGroup"><span class="pre"><code class="sourceCode python">ExceptionGroup</code></span></a>, it will now call the parent task’s <a href="../library/asyncio-task.html#asyncio.Task.cancel" class="reference internal" title="asyncio.Task.cancel"><span class="pre"><code class="sourceCode python">cancel()</code></span></a> method. This ensures that a <a href="../library/asyncio-exceptions.html#asyncio.CancelledError" class="reference internal" title="asyncio.CancelledError"><span class="pre"><code class="sourceCode python">CancelledError</code></span></a> will be raised at the next <a href="../reference/expressions.html#await" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">await</code></span></a>, so the cancellation is not lost.

  An added benefit of these changes is that task groups now preserve the cancellation count (<a href="../library/asyncio-task.html#asyncio.Task.cancelling" class="reference internal" title="asyncio.Task.cancelling"><span class="pre"><code class="sourceCode python">cancelling()</code></span></a>).

  In order to handle some corner cases, <a href="../library/asyncio-task.html#asyncio.Task.uncancel" class="reference internal" title="asyncio.Task.uncancel"><span class="pre"><code class="sourceCode python">uncancel()</code></span></a> may now reset the undocumented <span class="pre">`_must_cancel`</span> flag when the cancellation count reaches zero.

  (Inspired by an issue reported by Arthur Tacca in <a href="https://github.com/python/cpython/issues/116720" class="reference external">gh-116720</a>.)

- When <a href="../library/asyncio-task.html#asyncio.TaskGroup.create_task" class="reference internal" title="asyncio.TaskGroup.create_task"><span class="pre"><code class="sourceCode python">TaskGroup.create_task()</code></span></a> is called on an inactive <a href="../library/asyncio-task.html#asyncio.TaskGroup" class="reference internal" title="asyncio.TaskGroup"><span class="pre"><code class="sourceCode python">TaskGroup</code></span></a>, the given coroutine will be closed (which prevents a <a href="../library/exceptions.html#RuntimeWarning" class="reference internal" title="RuntimeWarning"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeWarning</span></code></span></a> about the given coroutine being never awaited). (Contributed by Arthur Tacca and Jason Zhang in <a href="https://github.com/python/cpython/issues/115957" class="reference external">gh-115957</a>.)

- The function and methods named <span class="pre">`create_task`</span> have received a new <span class="pre">`**kwargs`</span> argument that is passed through to the task constructor. This change was accidentally added in 3.13.3, and broke the API contract for custom task factories. Several third-party task factories implemented workarounds for this. In 3.13.4 and later releases the old factory contract is honored once again (until 3.14). To keep the workarounds working, the extra <span class="pre">`**kwargs`</span> argument still allows passing additional keyword arguments to <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">Task</code></span></a> and to custom task factories.

  This affects the following function and methods: <a href="../library/asyncio-task.html#asyncio.create_task" class="reference internal" title="asyncio.create_task"><span class="pre"><code class="sourceCode python">asyncio.create_task()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.create_task" class="reference internal" title="asyncio.loop.create_task"><span class="pre"><code class="sourceCode python">asyncio.loop.create_task()</code></span></a>, <a href="../library/asyncio-task.html#asyncio.TaskGroup.create_task" class="reference internal" title="asyncio.TaskGroup.create_task"><span class="pre"><code class="sourceCode python">asyncio.TaskGroup.create_task()</code></span></a>. (Contributed by Thomas Grainger in <a href="https://github.com/python/cpython/issues/128307" class="reference external">gh-128307</a>.)

</div>

<div id="base64" class="section">

### base64<a href="#base64" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/base64.html#base64.z85encode" class="reference internal" title="base64.z85encode"><span class="pre"><code class="sourceCode python">z85encode()</code></span></a> and <a href="../library/base64.html#base64.z85decode" class="reference internal" title="base64.z85decode"><span class="pre"><code class="sourceCode python">z85decode()</code></span></a> functions for encoding <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> as <a href="https://rfc.zeromq.org/spec/32/" class="reference external">Z85 data</a> and decoding Z85-encoded data to <span class="pre">`bytes`</span>. (Contributed by Matan Perelman in <a href="https://github.com/python/cpython/issues/75299" class="reference external">gh-75299</a>.)

</div>

<div id="compileall" class="section">

### compileall<a href="#compileall" class="headerlink" title="Link to this heading">¶</a>

- The default number of worker threads and processes is now selected using <a href="../library/os.html#os.process_cpu_count" class="reference internal" title="os.process_cpu_count"><span class="pre"><code class="sourceCode python">os.process_cpu_count()</code></span></a> instead of <a href="../library/os.html#os.cpu_count" class="reference internal" title="os.cpu_count"><span class="pre"><code class="sourceCode python">os.cpu_count()</code></span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/109649" class="reference external">gh-109649</a>.)

</div>

<div id="concurrent-futures" class="section">

### concurrent.futures<a href="#concurrent-futures" class="headerlink" title="Link to this heading">¶</a>

- The default number of worker threads and processes is now selected using <a href="../library/os.html#os.process_cpu_count" class="reference internal" title="os.process_cpu_count"><span class="pre"><code class="sourceCode python">os.process_cpu_count()</code></span></a> instead of <a href="../library/os.html#os.cpu_count" class="reference internal" title="os.cpu_count"><span class="pre"><code class="sourceCode python">os.cpu_count()</code></span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/109649" class="reference external">gh-109649</a>.)

</div>

<div id="configparser" class="section">

### configparser<a href="#configparser" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/configparser.html#configparser.ConfigParser" class="reference internal" title="configparser.ConfigParser"><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a> now has support for unnamed sections, which allows for top-level key-value pairs. This can be enabled with the new *allow_unnamed_section* parameter. (Contributed by Pedro Sousa Lacerda in <a href="https://github.com/python/cpython/issues/66449" class="reference external">gh-66449</a>.)

</div>

<div id="copy" class="section">

### copy<a href="#copy" class="headerlink" title="Link to this heading">¶</a>

- The new <a href="../library/copy.html#copy.replace" class="reference internal" title="copy.replace"><span class="pre"><code class="sourceCode python">replace()</code></span></a> function and the <a href="../library/copy.html#object.__replace__" class="reference internal" title="object.__replace__"><span class="pre"><code class="sourceCode python">replace</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">protocol</code></span></a> make creating modified copies of objects much simpler. This is especially useful when working with immutable objects. The following types support the <a href="../library/copy.html#copy.replace" class="reference internal" title="copy.replace"><span class="pre"><code class="sourceCode python">replace()</code></span></a> function and implement the replace protocol:

  - <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">collections.namedtuple()</code></span></a>

  - <a href="../library/dataclasses.html#dataclasses.dataclass" class="reference internal" title="dataclasses.dataclass"><span class="pre"><code class="sourceCode python">dataclasses.dataclass</code></span></a>

  - <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime.datetime</code></span></a>, <a href="../library/datetime.html#datetime.date" class="reference internal" title="datetime.date"><span class="pre"><code class="sourceCode python">datetime.date</code></span></a>, <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">datetime.time</code></span></a>

  - <a href="../library/inspect.html#inspect.Signature" class="reference internal" title="inspect.Signature"><span class="pre"><code class="sourceCode python">inspect.Signature</code></span></a>, <a href="../library/inspect.html#inspect.Parameter" class="reference internal" title="inspect.Parameter"><span class="pre"><code class="sourceCode python">inspect.Parameter</code></span></a>

  - <a href="../library/types.html#types.SimpleNamespace" class="reference internal" title="types.SimpleNamespace"><span class="pre"><code class="sourceCode python">types.SimpleNamespace</code></span></a>

  - <a href="../reference/datamodel.html#code-objects" class="reference internal"><span class="std std-ref">code objects</span></a>

  Any user-defined class can also support <a href="../library/copy.html#copy.replace" class="reference internal" title="copy.replace"><span class="pre"><code class="sourceCode python">copy.replace()</code></span></a> by defining the <a href="../library/copy.html#object.__replace__" class="reference internal" title="object.__replace__"><span class="pre"><code class="sourceCode python">__replace__()</code></span></a> method. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/108751" class="reference external">gh-108751</a>.)

</div>

<div id="ctypes" class="section">

### ctypes<a href="#ctypes" class="headerlink" title="Link to this heading">¶</a>

- As a consequence of necessary internal refactoring, initialization of internal metaclasses now happens in <span class="pre">`__init__`</span> rather than in <span class="pre">`__new__`</span>. This affects projects that subclass these internal metaclasses to provide custom initialization. Generally:

  - Custom logic that was done in <span class="pre">`__new__`</span> after calling <span class="pre">`super().__new__`</span> should be moved to <span class="pre">`__init__`</span>.

  - To create a class, call the metaclass, not only the metaclass’s <span class="pre">`__new__`</span> method.

  See <a href="https://github.com/python/cpython/issues/124520" class="reference external">gh-124520</a> for discussion and links to changes in some affected projects.

- <a href="../library/ctypes.html#ctypes.Structure" class="reference internal" title="ctypes.Structure"><span class="pre"><code class="sourceCode python">ctypes.Structure</code></span></a> objects have a new <a href="../library/ctypes.html#ctypes.Structure._align_" class="reference internal" title="ctypes.Structure._align_"><span class="pre"><code class="sourceCode python">_align_</code></span></a> attribute which allows the alignment of the structure being packed to/from memory to be specified explicitly. (Contributed by Matt Sanderson in <a href="https://github.com/python/cpython/issues/112433" class="reference external">gh-112433</a>)

</div>

<div id="dbm" class="section">

### dbm<a href="#dbm" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/dbm.html#module-dbm.sqlite3" class="reference internal" title="dbm.sqlite3: SQLite backend for dbm (All)"><span class="pre"><code class="sourceCode python">dbm.sqlite3</code></span></a>, a new module which implements an SQLite backend, and make it the default <span class="pre">`dbm`</span> backend. (Contributed by Raymond Hettinger and Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/100414" class="reference external">gh-100414</a>.)

- Allow removing all items from the database through the new <span class="pre">`clear()`</span> methods of the GDBM and NDBM database objects. (Contributed by Donghee Na in <a href="https://github.com/python/cpython/issues/107122" class="reference external">gh-107122</a>.)

</div>

<div id="dis" class="section">

### dis<a href="#dis" class="headerlink" title="Link to this heading">¶</a>

- Change the output of <a href="../library/dis.html#module-dis" class="reference internal" title="dis: Disassembler for Python bytecode."><span class="pre"><code class="sourceCode python">dis</code></span></a> module functions to show logical labels for jump targets and exception handlers, rather than offsets. The offsets can be added with the new <a href="../library/dis.html#cmdoption-dis-O" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-O</code></span></a> command-line option or the *show_offsets* argument. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/112137" class="reference external">gh-112137</a>.)

- <a href="../library/dis.html#dis.get_instructions" class="reference internal" title="dis.get_instructions"><span class="pre"><code class="sourceCode python">get_instructions()</code></span></a> no longer represents cache entries as separate instructions. Instead, it returns them as part of the <a href="../library/dis.html#dis.Instruction" class="reference internal" title="dis.Instruction"><span class="pre"><code class="sourceCode python">Instruction</code></span></a>, in the new *cache_info* field. The *show_caches* argument to <a href="../library/dis.html#dis.get_instructions" class="reference internal" title="dis.get_instructions"><span class="pre"><code class="sourceCode python">get_instructions()</code></span></a> is deprecated and no longer has any effect. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/112962" class="reference external">gh-112962</a>.)

</div>

<div id="doctest" class="section">

<span id="whatsnew313-doctest"></span>

### doctest<a href="#doctest" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> output is now colored by default. This can be controlled via the new <span id="index-39" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_COLORS" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_COLORS</code></span></a> environment variable as well as the canonical <a href="https://no-color.org/" class="reference external"><span class="pre"><code class="docutils literal notranslate">NO_COLOR</code></span></a> and <a href="https://force-color.org/" class="reference external"><span class="pre"><code class="docutils literal notranslate">FORCE_COLOR</code></span></a> environment variables. See also <a href="../using/cmdline.html#using-on-controlling-color" class="reference internal"><span class="std std-ref">Controlling color</span></a>. (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/117225" class="reference external">gh-117225</a>.)

- The <a href="../library/doctest.html#doctest.DocTestRunner.run" class="reference internal" title="doctest.DocTestRunner.run"><span class="pre"><code class="sourceCode python">DocTestRunner.run()</code></span></a> method now counts the number of skipped tests. Add the <a href="../library/doctest.html#doctest.DocTestRunner.skips" class="reference internal" title="doctest.DocTestRunner.skips"><span class="pre"><code class="sourceCode python">DocTestRunner.skips</code></span></a> and <a href="../library/doctest.html#doctest.TestResults.skipped" class="reference internal" title="doctest.TestResults.skipped"><span class="pre"><code class="sourceCode python">TestResults.skipped</code></span></a> attributes. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108794" class="reference external">gh-108794</a>.)

</div>

<div id="email" class="section">

### email<a href="#email" class="headerlink" title="Link to this heading">¶</a>

- Headers with embedded newlines are now quoted on output. The <a href="../library/email.generator.html#module-email.generator" class="reference internal" title="email.generator: Generate flat text email messages from a message structure."><span class="pre"><code class="sourceCode python">generator</code></span></a> will now refuse to serialize (write) headers that are improperly folded or delimited, such that they would be parsed as multiple headers or joined with adjacent data. If you need to turn this safety feature off, set <a href="../library/email.policy.html#email.policy.Policy.verify_generated_headers" class="reference internal" title="email.policy.Policy.verify_generated_headers"><span class="pre"><code class="sourceCode python">verify_generated_headers</code></span></a>. (Contributed by Bas Bloemsaat and Petr Viktorin in <a href="https://github.com/python/cpython/issues/121650" class="reference external">gh-121650</a>.)

- <a href="../library/email.utils.html#email.utils.getaddresses" class="reference internal" title="email.utils.getaddresses"><span class="pre"><code class="sourceCode python">getaddresses()</code></span></a> and <a href="../library/email.utils.html#email.utils.parseaddr" class="reference internal" title="email.utils.parseaddr"><span class="pre"><code class="sourceCode python">parseaddr()</code></span></a> now return <span class="pre">`('',`</span>` `<span class="pre">`'')`</span> pairs in more situations where invalid email addresses are encountered instead of potentially inaccurate values. The two functions have a new optional *strict* parameter (default <span class="pre">`True`</span>). To get the old behavior (accepting malformed input), use <span class="pre">`strict=False`</span>. <span class="pre">`getattr(email.utils,`</span>` `<span class="pre">`'supports_strict_parsing',`</span>` `<span class="pre">`False)`</span> can be used to check if the *strict* parameter is available. (Contributed by Thomas Dwyer and Victor Stinner for <a href="https://github.com/python/cpython/issues/102988" class="reference external">gh-102988</a> to improve the <span id="index-40" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2023-27043" class="cve reference external"><strong>CVE 2023-27043</strong></a> fix.)

</div>

<div id="enum" class="section">

### enum<a href="#enum" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/enum.html#enum.EnumDict" class="reference internal" title="enum.EnumDict"><span class="pre"><code class="sourceCode python">EnumDict</code></span></a> has been made public to better support subclassing <a href="../library/enum.html#enum.EnumType" class="reference internal" title="enum.EnumType"><span class="pre"><code class="sourceCode python">EnumType</code></span></a>.

</div>

<div id="fractions" class="section">

### fractions<a href="#fractions" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">Fraction</code></span></a> objects now support the standard <a href="../library/string.html#formatspec" class="reference internal"><span class="std std-ref">format specification mini-language</span></a> rules for fill, alignment, sign handling, minimum width, and grouping. (Contributed by Mark Dickinson in <a href="https://github.com/python/cpython/issues/111320" class="reference external">gh-111320</a>.)

</div>

<div id="glob" class="section">

### glob<a href="#glob" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/glob.html#glob.translate" class="reference internal" title="glob.translate"><span class="pre"><code class="sourceCode python">translate()</code></span></a>, a function to convert a path specification with shell-style wildcards to a regular expression. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/72904" class="reference external">gh-72904</a>.)

</div>

<div id="importlib" class="section">

### importlib<a href="#importlib" class="headerlink" title="Link to this heading">¶</a>

- The following functions in <a href="../library/importlib.resources.html#module-importlib.resources" class="reference internal" title="importlib.resources: Package resource reading, opening, and access"><span class="pre"><code class="sourceCode python">importlib.resources</code></span></a> now allow accessing a directory (or tree) of resources, using multiple positional arguments (the *encoding* and *errors* arguments in the text-reading functions are now keyword-only):

  - <a href="../library/importlib.resources.html#importlib.resources.is_resource" class="reference internal" title="importlib.resources.is_resource"><span class="pre"><code class="sourceCode python">is_resource()</code></span></a>

  - <a href="../library/importlib.resources.html#importlib.resources.open_binary" class="reference internal" title="importlib.resources.open_binary"><span class="pre"><code class="sourceCode python">open_binary()</code></span></a>

  - <a href="../library/importlib.resources.html#importlib.resources.open_text" class="reference internal" title="importlib.resources.open_text"><span class="pre"><code class="sourceCode python">open_text()</code></span></a>

  - <a href="../library/importlib.resources.html#importlib.resources.path" class="reference internal" title="importlib.resources.path"><span class="pre"><code class="sourceCode python">path()</code></span></a>

  - <a href="../library/importlib.resources.html#importlib.resources.read_binary" class="reference internal" title="importlib.resources.read_binary"><span class="pre"><code class="sourceCode python">read_binary()</code></span></a>

  - <a href="../library/importlib.resources.html#importlib.resources.read_text" class="reference internal" title="importlib.resources.read_text"><span class="pre"><code class="sourceCode python">read_text()</code></span></a>

  These functions are no longer deprecated and are not scheduled for removal. (Contributed by Petr Viktorin in <a href="https://github.com/python/cpython/issues/116608" class="reference external">gh-116608</a>.)

- <a href="../library/importlib.resources.html#importlib.resources.contents" class="reference internal" title="importlib.resources.contents"><span class="pre"><code class="sourceCode python">contents()</code></span></a> remains deprecated in favor of the fully-featured <a href="../library/importlib.resources.abc.html#importlib.resources.abc.Traversable" class="reference internal" title="importlib.resources.abc.Traversable"><span class="pre"><code class="sourceCode python">Traversable</code></span></a> API. However, there is now no plan to remove it. (Contributed by Petr Viktorin in <a href="https://github.com/python/cpython/issues/116608" class="reference external">gh-116608</a>.)

</div>

<div id="io" class="section">

### io<a href="#io" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/io.html#io.IOBase" class="reference internal" title="io.IOBase"><span class="pre"><code class="sourceCode python">IOBase</code></span></a> finalizer now logs any errors raised by the <a href="../library/io.html#io.IOBase.close" class="reference internal" title="io.IOBase.close"><span class="pre"><code class="sourceCode python">close()</code></span></a> method with <a href="../library/sys.html#sys.unraisablehook" class="reference internal" title="sys.unraisablehook"><span class="pre"><code class="sourceCode python">sys.unraisablehook</code></span></a>. Previously, errors were ignored silently by default, and only logged in <a href="../library/devmode.html#devmode" class="reference internal"><span class="std std-ref">Python Development Mode</span></a> or when using a <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">Python debug build</span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/62948" class="reference external">gh-62948</a>.)

</div>

<div id="ipaddress" class="section">

### ipaddress<a href="#ipaddress" class="headerlink" title="Link to this heading">¶</a>

- Add the <a href="../library/ipaddress.html#ipaddress.IPv4Address.ipv6_mapped" class="reference internal" title="ipaddress.IPv4Address.ipv6_mapped"><span class="pre"><code class="sourceCode python">IPv4Address.ipv6_mapped</code></span></a> property, which returns the IPv4-mapped IPv6 address. (Contributed by Charles Machalow in <a href="https://github.com/python/cpython/issues/109466" class="reference external">gh-109466</a>.)

- Fix <span class="pre">`is_global`</span> and <span class="pre">`is_private`</span> behavior in <a href="../library/ipaddress.html#ipaddress.IPv4Address" class="reference internal" title="ipaddress.IPv4Address"><span class="pre"><code class="sourceCode python">IPv4Address</code></span></a>, <a href="../library/ipaddress.html#ipaddress.IPv6Address" class="reference internal" title="ipaddress.IPv6Address"><span class="pre"><code class="sourceCode python">IPv6Address</code></span></a>, <a href="../library/ipaddress.html#ipaddress.IPv4Network" class="reference internal" title="ipaddress.IPv4Network"><span class="pre"><code class="sourceCode python">IPv4Network</code></span></a>, and <a href="../library/ipaddress.html#ipaddress.IPv6Network" class="reference internal" title="ipaddress.IPv6Network"><span class="pre"><code class="sourceCode python">IPv6Network</code></span></a>. (Contributed by Jakub Stasiak in <a href="https://github.com/python/cpython/issues/113171" class="reference external">gh-113171</a>.)

</div>

<div id="itertools" class="section">

### itertools<a href="#itertools" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/itertools.html#itertools.batched" class="reference internal" title="itertools.batched"><span class="pre"><code class="sourceCode python">batched()</code></span></a> has a new *strict* parameter, which raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the final batch is shorter than the specified batch size. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/113202" class="reference external">gh-113202</a>.)

</div>

<div id="marshal" class="section">

### marshal<a href="#marshal" class="headerlink" title="Link to this heading">¶</a>

- Add the *allow_code* parameter in module functions. Passing <span class="pre">`allow_code=False`</span> prevents serialization and de-serialization of code objects which are incompatible between Python versions. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/113626" class="reference external">gh-113626</a>.)

</div>

<div id="math" class="section">

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

- The new function <a href="../library/math.html#math.fma" class="reference internal" title="math.fma"><span class="pre"><code class="sourceCode python">fma()</code></span></a> performs fused multiply-add operations. This computes <span class="pre">`x`</span>` `<span class="pre">`*`</span>` `<span class="pre">`y`</span>` `<span class="pre">`+`</span>` `<span class="pre">`z`</span> with only a single round, and so avoids any intermediate loss of precision. It wraps the <span class="pre">`fma()`</span> function provided by C99, and follows the specification of the IEEE 754 “fusedMultiplyAdd” operation for special cases. (Contributed by Mark Dickinson and Victor Stinner in <a href="https://github.com/python/cpython/issues/73468" class="reference external">gh-73468</a>.)

</div>

<div id="mimetypes" class="section">

### mimetypes<a href="#mimetypes" class="headerlink" title="Link to this heading">¶</a>

- Add the <a href="../library/mimetypes.html#mimetypes.guess_file_type" class="reference internal" title="mimetypes.guess_file_type"><span class="pre"><code class="sourceCode python">guess_file_type()</code></span></a> function to guess a MIME type from a filesystem path. Using paths with <a href="../library/mimetypes.html#mimetypes.guess_type" class="reference internal" title="mimetypes.guess_type"><span class="pre"><code class="sourceCode python">guess_type()</code></span></a> is now <a href="../glossary.html#term-soft-deprecated" class="reference internal"><span class="xref std std-term">soft deprecated</span></a>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/66543" class="reference external">gh-66543</a>.)

</div>

<div id="mmap" class="section">

### mmap<a href="#mmap" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/mmap.html#mmap.mmap" class="reference internal" title="mmap.mmap"><span class="pre"><code class="sourceCode python">mmap</code></span></a> is now protected from crashing on Windows when the mapped memory is inaccessible due to file system errors or access violations. (Contributed by Jannis Weigend in <a href="https://github.com/python/cpython/issues/118209" class="reference external">gh-118209</a>.)

- <a href="../library/mmap.html#mmap.mmap" class="reference internal" title="mmap.mmap"><span class="pre"><code class="sourceCode python">mmap</code></span></a> has a new <a href="../library/mmap.html#mmap.mmap.seekable" class="reference internal" title="mmap.mmap.seekable"><span class="pre"><code class="sourceCode python">seekable()</code></span></a> method that can be used when a seekable file-like object is required. The <a href="../library/mmap.html#mmap.mmap.seek" class="reference internal" title="mmap.mmap.seek"><span class="pre"><code class="sourceCode python">seek()</code></span></a> method now returns the new absolute position. (Contributed by Donghee Na and Sylvie Liberman in <a href="https://github.com/python/cpython/issues/111835" class="reference external">gh-111835</a>.)

- The new UNIX-only *trackfd* parameter for <a href="../library/mmap.html#mmap.mmap" class="reference internal" title="mmap.mmap"><span class="pre"><code class="sourceCode python">mmap</code></span></a> controls file descriptor duplication; if false, the file descriptor specified by *fileno* will not be duplicated. (Contributed by Zackery Spytz and Petr Viktorin in <a href="https://github.com/python/cpython/issues/78502" class="reference external">gh-78502</a>.)

</div>

<div id="multiprocessing" class="section">

### multiprocessing<a href="#multiprocessing" class="headerlink" title="Link to this heading">¶</a>

- The default number of worker threads and processes is now selected using <a href="../library/os.html#os.process_cpu_count" class="reference internal" title="os.process_cpu_count"><span class="pre"><code class="sourceCode python">os.process_cpu_count()</code></span></a> instead of <a href="../library/os.html#os.cpu_count" class="reference internal" title="os.cpu_count"><span class="pre"><code class="sourceCode python">os.cpu_count()</code></span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/109649" class="reference external">gh-109649</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/os.html#os.process_cpu_count" class="reference internal" title="os.process_cpu_count"><span class="pre"><code class="sourceCode python">process_cpu_count()</code></span></a> function to get the number of logical CPU cores usable by the calling thread of the current process. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/109649" class="reference external">gh-109649</a>.)

- <a href="../library/os.html#os.cpu_count" class="reference internal" title="os.cpu_count"><span class="pre"><code class="sourceCode python">cpu_count()</code></span></a> and <a href="../library/os.html#os.process_cpu_count" class="reference internal" title="os.process_cpu_count"><span class="pre"><code class="sourceCode python">process_cpu_count()</code></span></a> can be overridden through the new environment variable <span id="index-41" class="target"></span><a href="../using/cmdline.html#envvar-PYTHON_CPU_COUNT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHON_CPU_COUNT</code></span></a> or the new command-line option <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">cpu_count</code></span></a>. This option is useful for users who need to limit CPU resources of a container system without having to modify application code or the container itself. (Contributed by Donghee Na in <a href="https://github.com/python/cpython/issues/109595" class="reference external">gh-109595</a>.)

- Add a <a href="../library/os.html#os-timerfd" class="reference internal"><span class="std std-ref">low level interface</span></a> to Linux’s *<a href="https://manpages.debian.org/timerfd_create(2)" class="manpage reference external">timer file descriptors</a>* via <a href="../library/os.html#os.timerfd_create" class="reference internal" title="os.timerfd_create"><span class="pre"><code class="sourceCode python">timerfd_create()</code></span></a>, <a href="../library/os.html#os.timerfd_settime" class="reference internal" title="os.timerfd_settime"><span class="pre"><code class="sourceCode python">timerfd_settime()</code></span></a>, <a href="../library/os.html#os.timerfd_settime_ns" class="reference internal" title="os.timerfd_settime_ns"><span class="pre"><code class="sourceCode python">timerfd_settime_ns()</code></span></a>, <a href="../library/os.html#os.timerfd_gettime" class="reference internal" title="os.timerfd_gettime"><span class="pre"><code class="sourceCode python">timerfd_gettime()</code></span></a>, <a href="../library/os.html#os.timerfd_gettime_ns" class="reference internal" title="os.timerfd_gettime_ns"><span class="pre"><code class="sourceCode python">timerfd_gettime_ns()</code></span></a>, <a href="../library/os.html#os.TFD_NONBLOCK" class="reference internal" title="os.TFD_NONBLOCK"><span class="pre"><code class="sourceCode python">TFD_NONBLOCK</code></span></a>, <a href="../library/os.html#os.TFD_CLOEXEC" class="reference internal" title="os.TFD_CLOEXEC"><span class="pre"><code class="sourceCode python">TFD_CLOEXEC</code></span></a>, <a href="../library/os.html#os.TFD_TIMER_ABSTIME" class="reference internal" title="os.TFD_TIMER_ABSTIME"><span class="pre"><code class="sourceCode python">TFD_TIMER_ABSTIME</code></span></a>, and <a href="../library/os.html#os.TFD_TIMER_CANCEL_ON_SET" class="reference internal" title="os.TFD_TIMER_CANCEL_ON_SET"><span class="pre"><code class="sourceCode python">TFD_TIMER_CANCEL_ON_SET</code></span></a> (Contributed by Masaru Tsuchiyama in <a href="https://github.com/python/cpython/issues/108277" class="reference external">gh-108277</a>.)

- <a href="../library/os.html#os.lchmod" class="reference internal" title="os.lchmod"><span class="pre"><code class="sourceCode python">lchmod()</code></span></a> and the *follow_symlinks* argument of <a href="../library/os.html#os.chmod" class="reference internal" title="os.chmod"><span class="pre"><code class="sourceCode python">chmod()</code></span></a> are both now available on Windows. Note that the default value of *follow_symlinks* in <span class="pre">`lchmod()`</span> is <span class="pre">`False`</span> on Windows. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/59616" class="reference external">gh-59616</a>.)

- <a href="../library/os.html#os.fchmod" class="reference internal" title="os.fchmod"><span class="pre"><code class="sourceCode python">fchmod()</code></span></a> and support for file descriptors in <a href="../library/os.html#os.chmod" class="reference internal" title="os.chmod"><span class="pre"><code class="sourceCode python">chmod()</code></span></a> are both now available on Windows. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/113191" class="reference external">gh-113191</a>.)

- On Windows, <a href="../library/os.html#os.mkdir" class="reference internal" title="os.mkdir"><span class="pre"><code class="sourceCode python">mkdir()</code></span></a> and <a href="../library/os.html#os.makedirs" class="reference internal" title="os.makedirs"><span class="pre"><code class="sourceCode python">makedirs()</code></span></a> now support passing a *mode* value of <span class="pre">`0o700`</span> to apply access control to the new directory. This implicitly affects <a href="../library/tempfile.html#tempfile.mkdtemp" class="reference internal" title="tempfile.mkdtemp"><span class="pre"><code class="sourceCode python">tempfile.mkdtemp()</code></span></a> and is a mitigation for <span id="index-42" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2024-4030" class="cve reference external"><strong>CVE 2024-4030</strong></a>. Other values for *mode* continue to be ignored. (Contributed by Steve Dower in <a href="https://github.com/python/cpython/issues/118486" class="reference external">gh-118486</a>.)

- <a href="../library/os.html#os.posix_spawn" class="reference internal" title="os.posix_spawn"><span class="pre"><code class="sourceCode python">posix_spawn()</code></span></a> now accepts <span class="pre">`None`</span> for the *env* argument, which makes the newly spawned process use the current process environment. (Contributed by Jakub Kulik in <a href="https://github.com/python/cpython/issues/113119" class="reference external">gh-113119</a>.)

- <a href="../library/os.html#os.posix_spawn" class="reference internal" title="os.posix_spawn"><span class="pre"><code class="sourceCode python">posix_spawn()</code></span></a> can now use the <a href="../library/os.html#os.POSIX_SPAWN_CLOSEFROM" class="reference internal" title="os.POSIX_SPAWN_CLOSEFROM"><span class="pre"><code class="sourceCode python">POSIX_SPAWN_CLOSEFROM</code></span></a> attribute in the *file_actions* parameter on platforms that support <span class="pre">`posix_spawn_file_actions_addclosefrom_np()`</span>. (Contributed by Jakub Kulik in <a href="https://github.com/python/cpython/issues/113117" class="reference external">gh-113117</a>.)

</div>

<div id="os-path" class="section">

### os.path<a href="#os-path" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/os.path.html#os.path.isreserved" class="reference internal" title="os.path.isreserved"><span class="pre"><code class="sourceCode python">isreserved()</code></span></a> to check if a path is reserved on the current system. This function is only available on Windows. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/88569" class="reference external">gh-88569</a>.)

- On Windows, <a href="../library/os.path.html#os.path.isabs" class="reference internal" title="os.path.isabs"><span class="pre"><code class="sourceCode python">isabs()</code></span></a> no longer considers paths starting with exactly one slash (<span class="pre">`\`</span> or <span class="pre">`/`</span>) to be absolute. (Contributed by Barney Gale and Jon Foster in <a href="https://github.com/python/cpython/issues/44626" class="reference external">gh-44626</a>.)

- <a href="../library/os.path.html#os.path.realpath" class="reference internal" title="os.path.realpath"><span class="pre"><code class="sourceCode python">realpath()</code></span></a> now resolves MS-DOS style file names even if the file is not accessible. (Contributed by Moonsik Park in <a href="https://github.com/python/cpython/issues/82367" class="reference external">gh-82367</a>.)

</div>

<div id="pathlib" class="section">

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/pathlib.html#pathlib.UnsupportedOperation" class="reference internal" title="pathlib.UnsupportedOperation"><span class="pre"><code class="sourceCode python">UnsupportedOperation</code></span></a>, which is raised instead of <a href="../library/exceptions.html#NotImplementedError" class="reference internal" title="NotImplementedError"><span class="pre"><code class="sourceCode python"><span class="pp">NotImplementedError</span></code></span></a> when a path operation isn’t supported. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/89812" class="reference external">gh-89812</a>.)

- Add a new constructor for creating <a href="../library/pathlib.html#pathlib.Path" class="reference internal" title="pathlib.Path"><span class="pre"><code class="sourceCode python">Path</code></span></a> objects from ‘file’ URIs (<span class="pre">`file:///`</span>), <a href="../library/pathlib.html#pathlib.Path.from_uri" class="reference internal" title="pathlib.Path.from_uri"><span class="pre"><code class="sourceCode python">Path.from_uri()</code></span></a>. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/107465" class="reference external">gh-107465</a>.)

- Add <a href="../library/pathlib.html#pathlib.PurePath.full_match" class="reference internal" title="pathlib.PurePath.full_match"><span class="pre"><code class="sourceCode python">PurePath.full_match()</code></span></a> for matching paths with shell-style wildcards, including the recursive wildcard “<span class="pre">`**`</span>”. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/73435" class="reference external">gh-73435</a>.)

- Add the <a href="../library/pathlib.html#pathlib.PurePath.parser" class="reference internal" title="pathlib.PurePath.parser"><span class="pre"><code class="sourceCode python">PurePath.parser</code></span></a> class attribute to store the implementation of <a href="../library/os.path.html#module-os.path" class="reference internal" title="os.path: Operations on pathnames."><span class="pre"><code class="sourceCode python">os.path</code></span></a> used for low-level path parsing and joining. This will be either <span class="pre">`posixpath`</span> or <span class="pre">`ntpath`</span>.

- Add *recurse_symlinks* keyword-only argument to <a href="../library/pathlib.html#pathlib.Path.glob" class="reference internal" title="pathlib.Path.glob"><span class="pre"><code class="sourceCode python">Path.glob()</code></span></a> and <a href="../library/pathlib.html#pathlib.Path.rglob" class="reference internal" title="pathlib.Path.rglob"><span class="pre"><code class="sourceCode python">rglob()</code></span></a>. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/77609" class="reference external">gh-77609</a>.)

- <a href="../library/pathlib.html#pathlib.Path.glob" class="reference internal" title="pathlib.Path.glob"><span class="pre"><code class="sourceCode python">Path.glob()</code></span></a> and <a href="../library/pathlib.html#pathlib.Path.rglob" class="reference internal" title="pathlib.Path.rglob"><span class="pre"><code class="sourceCode python">rglob()</code></span></a> now return files and directories when given a pattern that ends with “<span class="pre">`**`</span>”. Previously, only directories were returned. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/70303" class="reference external">gh-70303</a>.)

- Add the *follow_symlinks* keyword-only argument to <a href="../library/pathlib.html#pathlib.Path.is_file" class="reference internal" title="pathlib.Path.is_file"><span class="pre"><code class="sourceCode python">Path.is_file</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.is_dir" class="reference internal" title="pathlib.Path.is_dir"><span class="pre"><code class="sourceCode python">Path.is_dir</code></span></a>, <a href="../library/pathlib.html#pathlib.Path.owner" class="reference internal" title="pathlib.Path.owner"><span class="pre"><code class="sourceCode python">Path.owner()</code></span></a>, and <a href="../library/pathlib.html#pathlib.Path.group" class="reference internal" title="pathlib.Path.group"><span class="pre"><code class="sourceCode python">Path.group()</code></span></a>. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/105793" class="reference external">gh-105793</a> and Kamil Turek in <a href="https://github.com/python/cpython/issues/107962" class="reference external">gh-107962</a>.)

</div>

<div id="pdb" class="section">

### pdb<a href="#pdb" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/functions.html#breakpoint" class="reference internal" title="breakpoint"><span class="pre"><code class="sourceCode python"><span class="bu">breakpoint</span>()</code></span></a> and <a href="../library/pdb.html#pdb.set_trace" class="reference internal" title="pdb.set_trace"><span class="pre"><code class="sourceCode python">set_trace()</code></span></a> now enter the debugger immediately rather than on the next line of code to be executed. This change prevents the debugger from breaking outside of the context when <span class="pre">`breakpoint()`</span> is positioned at the end of the context. (Contributed by Tian Gao in <a href="https://github.com/python/cpython/issues/118579" class="reference external">gh-118579</a>.)

- <span class="pre">`sys.path[0]`</span> is no longer replaced by the directory of the script being debugged when <a href="../library/sys.html#sys.flags.safe_path" class="reference internal" title="sys.flags.safe_path"><span class="pre"><code class="sourceCode python">sys.flags.safe_path</code></span></a> is set. (Contributed by Tian Gao and Christian Walther in <a href="https://github.com/python/cpython/issues/111762" class="reference external">gh-111762</a>.)

- <a href="../library/zipapp.html#module-zipapp" class="reference internal" title="zipapp: Manage executable Python zip archives"><span class="pre"><code class="sourceCode python">zipapp</code></span></a> is now supported as a debugging target. (Contributed by Tian Gao in <a href="https://github.com/python/cpython/issues/118501" class="reference external">gh-118501</a>.)

- Add ability to move between chained exceptions during post-mortem debugging in <a href="../library/pdb.html#pdb.pm" class="reference internal" title="pdb.pm"><span class="pre"><code class="sourceCode python">pm()</code></span></a> using the new <a href="../library/pdb.html#pdbcommand-exceptions" class="reference internal"><span class="pre"><code class="xref std std-pdbcmd docutils literal notranslate">exceptions</code></span><code class="xref std std-pdbcmd docutils literal notranslate"> </code><span class="pre"><code class="xref std std-pdbcmd docutils literal notranslate">[exc_number]</code></span></a> command for Pdb. (Contributed by Matthias Bussonnier in <a href="https://github.com/python/cpython/issues/106676" class="reference external">gh-106676</a>.)

- Expressions and statements whose prefix is a pdb command are now correctly identified and executed. (Contributed by Tian Gao in <a href="https://github.com/python/cpython/issues/108464" class="reference external">gh-108464</a>.)

</div>

<div id="queue" class="section">

### queue<a href="#queue" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/queue.html#queue.Queue.shutdown" class="reference internal" title="queue.Queue.shutdown"><span class="pre"><code class="sourceCode python">Queue.shutdown</code></span></a> and <a href="../library/queue.html#queue.ShutDown" class="reference internal" title="queue.ShutDown"><span class="pre"><code class="sourceCode python">ShutDown</code></span></a> to manage queue termination. (Contributed by Laurie Opperman and Yves Duprat in <a href="https://github.com/python/cpython/issues/104750" class="reference external">gh-104750</a>.)

</div>

<div id="random" class="section">

### random<a href="#random" class="headerlink" title="Link to this heading">¶</a>

- Add a <a href="../library/random.html#random-cli" class="reference internal"><span class="std std-ref">command-line interface</span></a>. (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/118131" class="reference external">gh-118131</a>.)

</div>

<div id="re" class="section">

### re<a href="#re" class="headerlink" title="Link to this heading">¶</a>

- Rename <span class="pre">`re.error`</span> to <a href="../library/re.html#re.PatternError" class="reference internal" title="re.PatternError"><span class="pre"><code class="sourceCode python">PatternError</code></span></a> for improved clarity. <span class="pre">`re.error`</span> is kept for backward compatibility.

</div>

<div id="shutil" class="section">

### shutil<a href="#shutil" class="headerlink" title="Link to this heading">¶</a>

- Support the *dir_fd* and *follow_symlinks* keyword arguments in <a href="../library/shutil.html#shutil.chown" class="reference internal" title="shutil.chown"><span class="pre"><code class="sourceCode python">chown()</code></span></a>. (Contributed by Berker Peksag and Tahia K in <a href="https://github.com/python/cpython/issues/62308" class="reference external">gh-62308</a>)

</div>

<div id="site" class="section">

### site<a href="#site" class="headerlink" title="Link to this heading">¶</a>

- <span class="pre">`.pth`</span> files are now decoded using UTF-8 first, and then with the <a href="../glossary.html#term-locale-encoding" class="reference internal"><span class="xref std std-term">locale encoding</span></a> if UTF-8 decoding fails. (Contributed by Inada Naoki in <a href="https://github.com/python/cpython/issues/117802" class="reference external">gh-117802</a>.)

</div>

<div id="sqlite3" class="section">

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

- A <a href="../library/exceptions.html#ResourceWarning" class="reference internal" title="ResourceWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ResourceWarning</span></code></span></a> is now emitted if a <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">Connection</code></span></a> object is not <a href="../library/sqlite3.html#sqlite3.Connection.close" class="reference internal" title="sqlite3.Connection.close"><span class="pre"><code class="sourceCode python">closed</code></span></a> explicitly. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/105539" class="reference external">gh-105539</a>.)

- Add the *filter* keyword-only parameter to <a href="../library/sqlite3.html#sqlite3.Connection.iterdump" class="reference internal" title="sqlite3.Connection.iterdump"><span class="pre"><code class="sourceCode python">Connection.iterdump()</code></span></a> for filtering database objects to dump. (Contributed by Mariusz Felisiak in <a href="https://github.com/python/cpython/issues/91602" class="reference external">gh-91602</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/ssl.html#ssl.create_default_context" class="reference internal" title="ssl.create_default_context"><span class="pre"><code class="sourceCode python">create_default_context()</code></span></a> API now includes <a href="../library/ssl.html#ssl.VERIFY_X509_PARTIAL_CHAIN" class="reference internal" title="ssl.VERIFY_X509_PARTIAL_CHAIN"><span class="pre"><code class="sourceCode python">VERIFY_X509_PARTIAL_CHAIN</code></span></a> and <a href="../library/ssl.html#ssl.VERIFY_X509_STRICT" class="reference internal" title="ssl.VERIFY_X509_STRICT"><span class="pre"><code class="sourceCode python">VERIFY_X509_STRICT</code></span></a> in its default flags.

  <div class="admonition note">

  Note

  <a href="../library/ssl.html#ssl.VERIFY_X509_STRICT" class="reference internal" title="ssl.VERIFY_X509_STRICT"><span class="pre"><code class="sourceCode python">VERIFY_X509_STRICT</code></span></a> may reject pre-<span id="index-43" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc5280.html" class="rfc reference external"><strong>RFC 5280</strong></a> or malformed certificates that the underlying OpenSSL implementation might otherwise accept. Whilst disabling this is not recommended, you can do so using:

  <div class="highlight-python notranslate">

  <div class="highlight">

      import ssl

      ctx = ssl.create_default_context()
      ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT

  </div>

  </div>

  </div>

  (Contributed by William Woodruff in <a href="https://github.com/python/cpython/issues/112389" class="reference external">gh-112389</a>.)

</div>

<div id="statistics" class="section">

### statistics<a href="#statistics" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/statistics.html#statistics.kde" class="reference internal" title="statistics.kde"><span class="pre"><code class="sourceCode python">kde()</code></span></a> for kernel density estimation. This makes it possible to estimate a continuous probability density function from a fixed number of discrete samples. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/115863" class="reference external">gh-115863</a>.)

- Add <a href="../library/statistics.html#statistics.kde_random" class="reference internal" title="statistics.kde_random"><span class="pre"><code class="sourceCode python">kde_random()</code></span></a> for sampling from an estimated probability density function created by <a href="../library/statistics.html#statistics.kde" class="reference internal" title="statistics.kde"><span class="pre"><code class="sourceCode python">kde()</code></span></a>. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/115863" class="reference external">gh-115863</a>.)

</div>

<div id="subprocess" class="section">

<span id="whatsnew313-subprocess"></span>

### subprocess<a href="#subprocess" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module now uses the <a href="../library/os.html#os.posix_spawn" class="reference internal" title="os.posix_spawn"><span class="pre"><code class="sourceCode python">posix_spawn()</code></span></a> function in more situations.

  Notably, when *close_fds* is <span class="pre">`True`</span> (the default), <a href="../library/os.html#os.posix_spawn" class="reference internal" title="os.posix_spawn"><span class="pre"><code class="sourceCode python">posix_spawn()</code></span></a> will be used when the C library provides <span class="pre">`posix_spawn_file_actions_addclosefrom_np()`</span>, which includes recent versions of Linux, FreeBSD, and Solaris. On Linux, this should perform similarly to the existing Linux <span class="pre">`vfork()`</span> based code.

  A private control knob <span class="pre">`subprocess._USE_POSIX_SPAWN`</span> can be set to <span class="pre">`False`</span> if you need to force <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> to never use <a href="../library/os.html#os.posix_spawn" class="reference internal" title="os.posix_spawn"><span class="pre"><code class="sourceCode python">posix_spawn()</code></span></a>. Please report your reason and platform details in the <a href="../bugs.html#using-the-tracker" class="reference internal"><span class="std std-ref">issue tracker</span></a> if you set this so that we can improve our API selection logic for everyone. (Contributed by Jakub Kulik in <a href="https://github.com/python/cpython/issues/113117" class="reference external">gh-113117</a>.)

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

- Add the <a href="../library/sys.html#sys._is_interned" class="reference internal" title="sys._is_interned"><span class="pre"><code class="sourceCode python">_is_interned()</code></span></a> function to test if a string was interned. This function is not guaranteed to exist in all implementations of Python. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/78573" class="reference external">gh-78573</a>.)

</div>

<div id="tempfile" class="section">

### tempfile<a href="#tempfile" class="headerlink" title="Link to this heading">¶</a>

- On Windows, the default mode <span class="pre">`0o700`</span> used by <a href="../library/tempfile.html#tempfile.mkdtemp" class="reference internal" title="tempfile.mkdtemp"><span class="pre"><code class="sourceCode python">tempfile.mkdtemp()</code></span></a> now limits access to the new directory due to changes to <a href="../library/os.html#os.mkdir" class="reference internal" title="os.mkdir"><span class="pre"><code class="sourceCode python">os.mkdir()</code></span></a>. This is a mitigation for <span id="index-44" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2024-4030" class="cve reference external"><strong>CVE 2024-4030</strong></a>. (Contributed by Steve Dower in <a href="https://github.com/python/cpython/issues/118486" class="reference external">gh-118486</a>.)

</div>

<div id="time" class="section">

### time<a href="#time" class="headerlink" title="Link to this heading">¶</a>

- On Windows, <a href="../library/time.html#time.monotonic" class="reference internal" title="time.monotonic"><span class="pre"><code class="sourceCode python">monotonic()</code></span></a> now uses the <span class="pre">`QueryPerformanceCounter()`</span> clock for a resolution of 1 microsecond, instead of the <span class="pre">`GetTickCount64()`</span> clock which has a resolution of 15.6 milliseconds. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/88494" class="reference external">gh-88494</a>.)

- On Windows, <a href="../library/time.html#time.time" class="reference internal" title="time.time"><span class="pre"><code class="sourceCode python">time()</code></span></a> now uses the <span class="pre">`GetSystemTimePreciseAsFileTime()`</span> clock for a resolution of 1 microsecond, instead of the <span class="pre">`GetSystemTimeAsFileTime()`</span> clock which has a resolution of 15.6 milliseconds. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/63207" class="reference external">gh-63207</a>.)

</div>

<div id="tkinter" class="section">

### tkinter<a href="#tkinter" class="headerlink" title="Link to this heading">¶</a>

- Add <a href="../library/tkinter.html#module-tkinter" class="reference internal" title="tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">tkinter</code></span></a> widget methods: <span class="pre">`tk_busy_hold()`</span>, <span class="pre">`tk_busy_configure()`</span>, <span class="pre">`tk_busy_cget()`</span>, <span class="pre">`tk_busy_forget()`</span>, <span class="pre">`tk_busy_current()`</span>, and <span class="pre">`tk_busy_status()`</span>. (Contributed by Miguel, klappnase and Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/72684" class="reference external">gh-72684</a>.)

- The <a href="../library/tkinter.html#module-tkinter" class="reference internal" title="tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">tkinter</code></span></a> widget method <span class="pre">`wm_attributes()`</span> now accepts the attribute name without the minus prefix to get window attributes, for example <span class="pre">`w.wm_attributes('alpha')`</span> and allows specifying attributes and values to set as keyword arguments, for example <span class="pre">`w.wm_attributes(alpha=0.5)`</span>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/43457" class="reference external">gh-43457</a>.)

- <span class="pre">`wm_attributes()`</span> can now return attributes as a <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a>, by using the new optional keyword-only parameter *return_python_dict*. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/43457" class="reference external">gh-43457</a>.)

- <span class="pre">`Text.count()`</span> can now return a simple <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> when the new optional keyword-only parameter *return_ints* is used. Otherwise, the single count is returned as a 1-tuple or <span class="pre">`None`</span>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/97928" class="reference external">gh-97928</a>.)

- Support the “vsapi” element type in the <a href="../library/tkinter.ttk.html#tkinter.ttk.Style.element_create" class="reference internal" title="tkinter.ttk.Style.element_create"><span class="pre"><code class="sourceCode python">element_create()</code></span></a> method of <a href="../library/tkinter.ttk.html#tkinter.ttk.Style" class="reference internal" title="tkinter.ttk.Style"><span class="pre"><code class="sourceCode python">tkinter.ttk.Style</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/68166" class="reference external">gh-68166</a>.)

- Add the <span class="pre">`after_info()`</span> method for Tkinter widgets. (Contributed by Cheryl Sabella in <a href="https://github.com/python/cpython/issues/77020" class="reference external">gh-77020</a>.)

- Add a new <span class="pre">`copy_replace()`</span> method to <span class="pre">`PhotoImage`</span> to copy a region from one image to another, possibly with pixel zooming, subsampling, or both. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/118225" class="reference external">gh-118225</a>.)

- Add *from_coords* parameter to the <span class="pre">`PhotoImage`</span> methods <span class="pre">`copy()`</span>, <span class="pre">`zoom()`</span> and <span class="pre">`subsample()`</span>. Add *zoom* and *subsample* parameters to the <span class="pre">`PhotoImage`</span> method <span class="pre">`copy()`</span>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/118225" class="reference external">gh-118225</a>.)

- Add the <span class="pre">`PhotoImage`</span> methods <span class="pre">`read()`</span> to read an image from a file and <span class="pre">`data()`</span> to get the image data. Add *background* and *grayscale* parameters to the <span class="pre">`write()`</span> method. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/118271" class="reference external">gh-118271</a>.)

</div>

<div id="traceback" class="section">

### traceback<a href="#traceback" class="headerlink" title="Link to this heading">¶</a>

- Add the <a href="../library/traceback.html#traceback.TracebackException.exc_type_str" class="reference internal" title="traceback.TracebackException.exc_type_str"><span class="pre"><code class="sourceCode python">exc_type_str</code></span></a> attribute to <a href="../library/traceback.html#traceback.TracebackException" class="reference internal" title="traceback.TracebackException"><span class="pre"><code class="sourceCode python">TracebackException</code></span></a>, which holds a string display of the *exc_type*. Deprecate the <a href="../library/traceback.html#traceback.TracebackException.exc_type" class="reference internal" title="traceback.TracebackException.exc_type"><span class="pre"><code class="sourceCode python">exc_type</code></span></a> attribute, which holds the type object itself. Add parameter *save_exc_type* (default <span class="pre">`True`</span>) to indicate whether <span class="pre">`exc_type`</span> should be saved. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/112332" class="reference external">gh-112332</a>.)

- Add a new *show_group* keyword-only parameter to <a href="../library/traceback.html#traceback.TracebackException.format_exception_only" class="reference internal" title="traceback.TracebackException.format_exception_only"><span class="pre"><code class="sourceCode python">TracebackException.format_exception_only()</code></span></a> to (recursively) format the nested exceptions of a <a href="../library/exceptions.html#BaseExceptionGroup" class="reference internal" title="BaseExceptionGroup"><span class="pre"><code class="sourceCode python">BaseExceptionGroup</code></span></a> instance. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/105292" class="reference external">gh-105292</a>.)

</div>

<div id="types" class="section">

### types<a href="#types" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/types.html#types.SimpleNamespace" class="reference internal" title="types.SimpleNamespace"><span class="pre"><code class="sourceCode python">SimpleNamespace</code></span></a> can now take a single positional argument to initialise the namespace’s arguments. This argument must either be a mapping or an iterable of key-value pairs. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/108191" class="reference external">gh-108191</a>.)

</div>

<div id="typing" class="section">

### typing<a href="#typing" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-45" class="target"></span><a href="https://peps.python.org/pep-0705/" class="pep reference external"><strong>PEP 705</strong></a>: Add <a href="../library/typing.html#typing.ReadOnly" class="reference internal" title="typing.ReadOnly"><span class="pre"><code class="sourceCode python">ReadOnly</code></span></a>, a special typing construct to mark a <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">TypedDict</code></span></a> item as read-only for type checkers.

- <span id="index-46" class="target"></span><a href="https://peps.python.org/pep-0742/" class="pep reference external"><strong>PEP 742</strong></a>: Add <a href="../library/typing.html#typing.TypeIs" class="reference internal" title="typing.TypeIs"><span class="pre"><code class="sourceCode python">TypeIs</code></span></a>, a typing construct that can be used to instruct a type checker how to narrow a type.

- Add <a href="../library/typing.html#typing.NoDefault" class="reference internal" title="typing.NoDefault"><span class="pre"><code class="sourceCode python">NoDefault</code></span></a>, a sentinel object used to represent the defaults of some parameters in the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/116126" class="reference external">gh-116126</a>.)

- Add <a href="../library/typing.html#typing.get_protocol_members" class="reference internal" title="typing.get_protocol_members"><span class="pre"><code class="sourceCode python">get_protocol_members()</code></span></a> to return the set of members defining a <a href="../library/typing.html#typing.Protocol" class="reference internal" title="typing.Protocol"><span class="pre"><code class="sourceCode python">typing.Protocol</code></span></a>. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/104873" class="reference external">gh-104873</a>.)

- Add <a href="../library/typing.html#typing.is_protocol" class="reference internal" title="typing.is_protocol"><span class="pre"><code class="sourceCode python">is_protocol()</code></span></a> to check whether a class is a <a href="../library/typing.html#typing.Protocol" class="reference internal" title="typing.Protocol"><span class="pre"><code class="sourceCode python">Protocol</code></span></a>. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/104873" class="reference external">gh-104873</a>.)

- <a href="../library/typing.html#typing.ClassVar" class="reference internal" title="typing.ClassVar"><span class="pre"><code class="sourceCode python">ClassVar</code></span></a> can now be nested in <a href="../library/typing.html#typing.Final" class="reference internal" title="typing.Final"><span class="pre"><code class="sourceCode python">Final</code></span></a>, and vice versa. (Contributed by Mehdi Drissi in <a href="https://github.com/python/cpython/issues/89547" class="reference external">gh-89547</a>.)

</div>

<div id="unicodedata" class="section">

### unicodedata<a href="#unicodedata" class="headerlink" title="Link to this heading">¶</a>

- Update the Unicode database to <a href="https://www.unicode.org/versions/Unicode15.1.0/" class="reference external">version 15.1.0</a>. (Contributed by James Gerity in <a href="https://github.com/python/cpython/issues/109559" class="reference external">gh-109559</a>.)

</div>

<div id="venv" class="section">

### venv<a href="#venv" class="headerlink" title="Link to this heading">¶</a>

- Add support for creating source control management (SCM) ignore files in a virtual environment’s directory. By default, Git is supported. This is implemented as opt-in via the API, which can be extended to support other SCMs (<a href="../library/venv.html#venv.EnvBuilder" class="reference internal" title="venv.EnvBuilder"><span class="pre"><code class="sourceCode python">EnvBuilder</code></span></a> and <a href="../library/venv.html#venv.create" class="reference internal" title="venv.create"><span class="pre"><code class="sourceCode python">create()</code></span></a>), and opt-out via the CLI, using <span class="pre">`--without-scm-ignore-files`</span>. (Contributed by Brett Cannon in <a href="https://github.com/python/cpython/issues/108125" class="reference external">gh-108125</a>.)

</div>

<div id="warnings" class="section">

### warnings<a href="#warnings" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-47" class="target"></span><a href="https://peps.python.org/pep-0702/" class="pep reference external"><strong>PEP 702</strong></a>: The new <a href="../library/warnings.html#warnings.deprecated" class="reference internal" title="warnings.deprecated"><span class="pre"><code class="sourceCode python">warnings.deprecated()</code></span></a> decorator provides a way to communicate deprecations to a <a href="../glossary.html#term-static-type-checker" class="reference internal"><span class="xref std std-term">static type checker</span></a> and to warn on usage of deprecated classes and functions. A <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> may also be emitted when a decorated function or class is used at runtime. (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/104003" class="reference external">gh-104003</a>.)

</div>

<div id="xml" class="section">

### xml<a href="#xml" class="headerlink" title="Link to this heading">¶</a>

- Allow controlling Expat \>=2.6.0 reparse deferral (<span id="index-48" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2023-52425" class="cve reference external"><strong>CVE 2023-52425</strong></a>) by adding five new methods:

  - <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLParser.flush" class="reference internal" title="xml.etree.ElementTree.XMLParser.flush"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.XMLParser.flush()</code></span></a>

  - <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLPullParser.flush" class="reference internal" title="xml.etree.ElementTree.XMLPullParser.flush"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.XMLPullParser.flush()</code></span></a>

  - <a href="../library/pyexpat.html#xml.parsers.expat.xmlparser.GetReparseDeferralEnabled" class="reference internal" title="xml.parsers.expat.xmlparser.GetReparseDeferralEnabled"><span class="pre"><code class="sourceCode python">xml.parsers.expat.xmlparser.GetReparseDeferralEnabled()</code></span></a>

  - <a href="../library/pyexpat.html#xml.parsers.expat.xmlparser.SetReparseDeferralEnabled" class="reference internal" title="xml.parsers.expat.xmlparser.SetReparseDeferralEnabled"><span class="pre"><code class="sourceCode python">xml.parsers.expat.xmlparser.SetReparseDeferralEnabled()</code></span></a>

  - <span class="pre">`xml.sax.expatreader.ExpatParser.flush()`</span>

  (Contributed by Sebastian Pipping in <a href="https://github.com/python/cpython/issues/115623" class="reference external">gh-115623</a>.)

- Add the <span class="pre">`close()`</span> method for the iterator returned by <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.iterparse" class="reference internal" title="xml.etree.ElementTree.iterparse"><span class="pre"><code class="sourceCode python">iterparse()</code></span></a> for explicit cleanup. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/69893" class="reference external">gh-69893</a>.)

</div>

<div id="zipimport" class="section">

### zipimport<a href="#zipimport" class="headerlink" title="Link to this heading">¶</a>

- Add support for <a href="https://en.wikipedia.org/wiki/Zip_(file_format)#ZIP64" class="reference external">ZIP64</a> format files. Everybody loves huge data, right? (Contributed by Tim Hatch in <a href="https://github.com/python/cpython/issues/94146" class="reference external">gh-94146</a>.)

</div>

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

- Several standard library modules have had their import times significantly improved. For example, the import time of the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module has been reduced by around a third by removing dependencies on <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> and <a href="../library/contextlib.html#module-contextlib" class="reference internal" title="contextlib: Utilities for with-statement contexts."><span class="pre"><code class="sourceCode python">contextlib</code></span></a>. Other modules to enjoy import-time speedups include <a href="../library/email.utils.html#module-email.utils" class="reference internal" title="email.utils: Miscellaneous email package utilities."><span class="pre"><code class="sourceCode python">email.utils</code></span></a>, <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">enum</code></span></a>, <a href="../library/functools.html#module-functools" class="reference internal" title="functools: Higher-order functions and operations on callable objects."><span class="pre"><code class="sourceCode python">functools</code></span></a>, <a href="../library/importlib.metadata.html#module-importlib.metadata" class="reference internal" title="importlib.metadata: Accessing package metadata"><span class="pre"><code class="sourceCode python">importlib.metadata</code></span></a>, and <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a>. (Contributed by Alex Waygood, Shantanu Jain, Adam Turner, Daniel Hollas, and others in <a href="https://github.com/python/cpython/issues/109653" class="reference external">gh-109653</a>.)

- <a href="../library/textwrap.html#textwrap.indent" class="reference internal" title="textwrap.indent"><span class="pre"><code class="sourceCode python">textwrap.indent()</code></span></a> is now around 30% faster than before for large input. (Contributed by Inada Naoki in <a href="https://github.com/python/cpython/issues/107369" class="reference external">gh-107369</a>.)

- The <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module now uses the <a href="../library/os.html#os.posix_spawn" class="reference internal" title="os.posix_spawn"><span class="pre"><code class="sourceCode python">posix_spawn()</code></span></a> function in more situations, including when *close_fds* is <span class="pre">`True`</span> (the default) on many modern platforms. This should provide a notable performance increase when launching processes on FreeBSD and Solaris. See the <a href="#whatsnew313-subprocess" class="reference internal"><span class="std std-ref">subprocess</span></a> section above for details. (Contributed by Jakub Kulik in <a href="https://github.com/python/cpython/issues/113117" class="reference external">gh-113117</a>.)

</div>

<div id="removed-modules-and-apis" class="section">

## Removed Modules And APIs<a href="#removed-modules-and-apis" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-594-remove-dead-batteries-from-the-standard-library" class="section">

<span id="whatsnew313-pep594"></span>

### PEP 594: Remove “dead batteries” from the standard library<a href="#pep-594-remove-dead-batteries-from-the-standard-library" class="headerlink" title="Link to this heading">¶</a>

<span id="index-49" class="target"></span><a href="https://peps.python.org/pep-0594/" class="pep reference external"><strong>PEP 594</strong></a> proposed removing 19 modules from the standard library, colloquially referred to as ‘dead batteries’ due to their historic, obsolete, or insecure status. All of the following modules were deprecated in Python 3.11, and are now removed:

- <span class="pre">`aifc`</span>

  - <a href="https://pypi.org/project/standard-aifc/" class="extlink-pypi reference external">standard-aifc</a>: Use the redistribution of <span class="pre">`aifc`</span> library from PyPI.

- <span class="pre">`audioop`</span>

  - <a href="https://pypi.org/project/audioop-lts/" class="extlink-pypi reference external">audioop-lts</a>: Use <span class="pre">`audioop-lts`</span> library from PyPI.

- <span class="pre">`chunk`</span>

  - <a href="https://pypi.org/project/standard-chunk/" class="extlink-pypi reference external">standard-chunk</a>: Use the redistribution of <span class="pre">`chunk`</span> library from PyPI.

- <span class="pre">`cgi`</span> and <span class="pre">`cgitb`</span>

  - <span class="pre">`cgi.FieldStorage`</span> can typically be replaced with <a href="../library/urllib.parse.html#urllib.parse.parse_qsl" class="reference internal" title="urllib.parse.parse_qsl"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qsl()</code></span></a> for <span class="pre">`GET`</span> and <span class="pre">`HEAD`</span> requests, and the <a href="../library/email.message.html#module-email.message" class="reference internal" title="email.message: The base class representing email messages."><span class="pre"><code class="sourceCode python">email.message</code></span></a> module or the <a href="https://pypi.org/project/multipart/" class="extlink-pypi reference external">multipart</a> library for <span class="pre">`POST`</span> and <span class="pre">`PUT`</span> requests.

  - <span class="pre">`cgi.parse()`</span> can be replaced by calling <a href="../library/urllib.parse.html#urllib.parse.parse_qs" class="reference internal" title="urllib.parse.parse_qs"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qs()</code></span></a> directly on the desired query string, unless the input is <span class="pre">`multipart/form-data`</span>, which should be replaced as described below for <span class="pre">`cgi.parse_multipart()`</span>.

  - <span class="pre">`cgi.parse_header()`</span> can be replaced with the functionality in the <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages."><span class="pre"><code class="sourceCode python">email</code></span></a> package, which implements the same MIME RFCs. For example, with <a href="../library/email.message.html#email.message.EmailMessage" class="reference internal" title="email.message.EmailMessage"><span class="pre"><code class="sourceCode python">email.message.EmailMessage</code></span></a>:

    <div class="highlight-python notranslate">

    <div class="highlight">

        from email.message import EmailMessage

        msg = EmailMessage()
        msg['content-type'] = 'application/json; charset="utf8"'
        main, params = msg.get_content_type(), msg['content-type'].params

    </div>

    </div>

  - <span class="pre">`cgi.parse_multipart()`</span> can be replaced with the functionality in the <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages."><span class="pre"><code class="sourceCode python">email</code></span></a> package, which implements the same MIME RFCs, or with the <a href="https://pypi.org/project/multipart/" class="extlink-pypi reference external">multipart</a> library. For example, the <a href="../library/email.message.html#email.message.EmailMessage" class="reference internal" title="email.message.EmailMessage"><span class="pre"><code class="sourceCode python">email.message.EmailMessage</code></span></a> and <a href="../library/email.compat32-message.html#email.message.Message" class="reference internal" title="email.message.Message"><span class="pre"><code class="sourceCode python">email.message.Message</code></span></a> classes.

  - <a href="https://pypi.org/project/standard-cgi/" class="extlink-pypi reference external">standard-cgi</a>: and <a href="https://pypi.org/project/standard-cgitb/" class="extlink-pypi reference external">standard-cgitb</a>: Use the redistribution of <span class="pre">`cgi`</span> and <span class="pre">`cgitb`</span> library from PyPI.

- <span class="pre">`crypt`</span> and the private <span class="pre">`_crypt`</span> extension. The <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module may be an appropriate replacement when simply hashing a value is required. Otherwise, various third-party libraries on PyPI are available:

  - <a href="https://pypi.org/project/bcrypt/" class="extlink-pypi reference external">bcrypt</a>: Modern password hashing for your software and your servers.

  - <a href="https://pypi.org/project/passlib/" class="extlink-pypi reference external">passlib</a>: Comprehensive password hashing framework supporting over 30 schemes.

  - <a href="https://pypi.org/project/argon2-cffi/" class="extlink-pypi reference external">argon2-cffi</a>: The secure Argon2 password hashing algorithm.

  - <a href="https://pypi.org/project/legacycrypt/" class="extlink-pypi reference external">legacycrypt</a>: <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> wrapper to the POSIX crypt library call and associated functionality.

  - <a href="https://pypi.org/project/crypt_r/" class="extlink-pypi reference external">crypt_r</a>: Fork of the <span class="pre">`crypt`</span> module, wrapper to the *<a href="https://manpages.debian.org/crypt_r(3)" class="manpage reference external">crypt_r(3)</a>* library call and associated functionality.

  - <a href="https://pypi.org/project/standard-crypt/" class="extlink-pypi reference external">standard-crypt</a> and <a href="https://pypi.org/project/deprecated-crypt-alternative/" class="extlink-pypi reference external">deprecated-crypt-alternative</a>: Use the redistribution of <span class="pre">`crypt`</span> and reimplementation of <span class="pre">`_crypt`</span> libraries from PyPI.

- <span class="pre">`imghdr`</span>: The <a href="https://pypi.org/project/filetype/" class="extlink-pypi reference external">filetype</a>, <a href="https://pypi.org/project/puremagic/" class="extlink-pypi reference external">puremagic</a>, or <a href="https://pypi.org/project/python-magic/" class="extlink-pypi reference external">python-magic</a> libraries should be used as replacements. For example, the <span class="pre">`puremagic.what()`</span> function can be used to replace the <span class="pre">`imghdr.what()`</span> function for all file formats that were supported by <span class="pre">`imghdr`</span>.

  - <a href="https://pypi.org/project/standard-imghdr/" class="extlink-pypi reference external">standard-imghdr</a>: Use the redistribution of <span class="pre">`imghdr`</span> library from PyPI.

- <span class="pre">`mailcap`</span>: Use the <a href="../library/mimetypes.html#module-mimetypes" class="reference internal" title="mimetypes: Mapping of filename extensions to MIME types."><span class="pre"><code class="sourceCode python">mimetypes</code></span></a> module instead.

  - <a href="https://pypi.org/project/standard-mailcap/" class="extlink-pypi reference external">standard-mailcap</a>: Use the redistribution of <span class="pre">`mailcap`</span> library from PyPI.

- <span class="pre">`msilib`</span>

- <span class="pre">`nis`</span>

- <span class="pre">`nntplib`</span>: Use the <a href="https://pypi.org/project/pynntp/" class="extlink-pypi reference external">pynntp</a> library from PyPI instead.

  - <a href="https://pypi.org/project/standard-nntplib/" class="extlink-pypi reference external">standard-nntplib</a>: Use the redistribution of <span class="pre">`nntplib`</span> library from PyPI.

- <span class="pre">`ossaudiodev`</span>: For audio playback, use the <a href="https://pypi.org/project/pygame/" class="extlink-pypi reference external">pygame</a> library from PyPI instead.

- <span class="pre">`pipes`</span>: Use the <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module instead. Use <a href="../library/shlex.html#shlex.quote" class="reference internal" title="shlex.quote"><span class="pre"><code class="sourceCode python">shlex.quote()</code></span></a> to replace the undocumented <span class="pre">`pipes.quote`</span> function.

  - <a href="https://pypi.org/project/standard-pipes/" class="extlink-pypi reference external">standard-pipes</a>: Use the redistribution of <span class="pre">`pipes`</span> library from PyPI.

- <span class="pre">`sndhdr`</span>: The <a href="https://pypi.org/project/filetype/" class="extlink-pypi reference external">filetype</a>, <a href="https://pypi.org/project/puremagic/" class="extlink-pypi reference external">puremagic</a>, or <a href="https://pypi.org/project/python-magic/" class="extlink-pypi reference external">python-magic</a> libraries should be used as replacements.

  - <a href="https://pypi.org/project/standard-sndhdr/" class="extlink-pypi reference external">standard-sndhdr</a>: Use the redistribution of <span class="pre">`sndhdr`</span> library from PyPI.

- <span class="pre">`spwd`</span>: Use the <a href="https://pypi.org/project/python-pam/" class="extlink-pypi reference external">python-pam</a> library from PyPI instead.

- <span class="pre">`sunau`</span>

  - <a href="https://pypi.org/project/standard-sunau/" class="extlink-pypi reference external">standard-sunau</a>: Use the redistribution of <span class="pre">`sunau`</span> library from PyPI.

- <span class="pre">`telnetlib`</span>, Use the <a href="https://pypi.org/project/telnetlib3/" class="extlink-pypi reference external">telnetlib3</a> or <a href="https://pypi.org/project/Exscript/" class="extlink-pypi reference external">Exscript</a> libraries from PyPI instead.

  - <a href="https://pypi.org/project/standard-telnetlib/" class="extlink-pypi reference external">standard-telnetlib</a>: Use the redistribution of <span class="pre">`telnetlib`</span> library from PyPI.

- <span class="pre">`uu`</span>: Use the <a href="../library/base64.html#module-base64" class="reference internal" title="base64: RFC 4648: Base16, Base32, Base64 Data Encodings; Base85 and Ascii85"><span class="pre"><code class="sourceCode python">base64</code></span></a> module instead, as a modern alternative.

  - <a href="https://pypi.org/project/standard-uu/" class="extlink-pypi reference external">standard-uu</a>: Use the redistribution of <span class="pre">`uu`</span> library from PyPI.

- <span class="pre">`xdrlib`</span>

  - <a href="https://pypi.org/project/standard-xdrlib/" class="extlink-pypi reference external">standard-xdrlib</a>: Use the redistribution of <span class="pre">`xdrlib`</span> library from PyPI.

(Contributed by Victor Stinner and Zachary Ware in <a href="https://github.com/python/cpython/issues/104773" class="reference external">gh-104773</a> and <a href="https://github.com/python/cpython/issues/104780" class="reference external">gh-104780</a>.)

</div>

<div id="to3" class="section">

### 2to3<a href="#to3" class="headerlink" title="Link to this heading">¶</a>

- Remove the **2to3** program and the <span class="pre">`lib2to3`</span> module, previously deprecated in Python 3.11. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/104780" class="reference external">gh-104780</a>.)

</div>

<div id="builtins" class="section">

### builtins<a href="#builtins" class="headerlink" title="Link to this heading">¶</a>

- Remove support for chained <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span></code></span></a> descriptors (introduced in <a href="https://github.com/python/cpython/issues/63272" class="reference external">gh-63272</a>). These can no longer be used to wrap other descriptors, such as <a href="../library/functions.html#property" class="reference internal" title="property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span></code></span></a>. The core design of this feature was flawed and led to several problems. To “pass-through” a <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span></code></span></a>, consider using the <span class="pre">`__wrapped__`</span> attribute that was added in Python 3.10. (Contributed by Raymond Hettinger in <a href="https://github.com/python/cpython/issues/89519" class="reference external">gh-89519</a>.)

- Raise a <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> when calling <a href="../reference/datamodel.html#frame.clear" class="reference internal" title="frame.clear"><span class="pre"><code class="sourceCode python">frame.clear()</code></span></a> on a suspended frame (as has always been the case for an executing frame). (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/79932" class="reference external">gh-79932</a>.)

</div>

<div id="id3" class="section">

### configparser<a href="#id3" class="headerlink" title="Link to this heading">¶</a>

- Remove the undocumented <span class="pre">`LegacyInterpolation`</span> class, deprecated in the docstring since Python 3.2, and at runtime since Python 3.11. (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/104886" class="reference external">gh-104886</a>.)

</div>

<div id="importlib-metadata" class="section">

### importlib.metadata<a href="#importlib-metadata" class="headerlink" title="Link to this heading">¶</a>

- Remove deprecated subscript (<a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a>) access for <a href="../library/importlib.metadata.html#entry-points" class="reference internal"><span class="std std-ref">EntryPoint</span></a> objects. (Contributed by Jason R. Coombs in <a href="https://github.com/python/cpython/issues/113175" class="reference external">gh-113175</a>.)

</div>

<div id="locale" class="section">

### locale<a href="#locale" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`locale.resetlocale()`</span> function, deprecated in Python 3.11. Use <span class="pre">`locale.setlocale(locale.LC_ALL,`</span>` `<span class="pre">`"")`</span> instead. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/104783" class="reference external">gh-104783</a>.)

</div>

<div id="opcode" class="section">

### opcode<a href="#opcode" class="headerlink" title="Link to this heading">¶</a>

- Move <span class="pre">`opcode.ENABLE_SPECIALIZATION`</span> to <span class="pre">`_opcode.ENABLE_SPECIALIZATION`</span>. This field was added in 3.12, it was never documented, and is not intended for external use. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/105481" class="reference external">gh-105481</a>.)

- Remove <span class="pre">`opcode.is_pseudo()`</span>, <span class="pre">`opcode.MIN_PSEUDO_OPCODE`</span>, and <span class="pre">`opcode.MAX_PSEUDO_OPCODE`</span>, which were added in Python 3.12, but were neither documented nor exposed through <a href="../library/dis.html#module-dis" class="reference internal" title="dis: Disassembler for Python bytecode."><span class="pre"><code class="sourceCode python">dis</code></span></a>, and were not intended to be used externally. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/105481" class="reference external">gh-105481</a>.)

</div>

<div id="optparse" class="section">

### optparse<a href="#optparse" class="headerlink" title="Link to this heading">¶</a>

- This module is no longer considered <a href="../glossary.html#term-soft-deprecated" class="reference internal"><span class="xref std std-term">soft deprecated</span></a>. While <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> remains preferred for new projects that aren’t using a third party command line argument processing library, there are aspects of the way <span class="pre">`argparse`</span> works that mean the lower level <span class="pre">`optparse`</span> module may provide a better foundation for *writing* argument processing libraries, and for implementing command line applications which adhere more strictly than <span class="pre">`argparse`</span> does to various Unix command line processing conventions that originate in the behaviour of the C <span class="pre">`getopt()`</span> function . (Contributed by Alyssa Coghlan and Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/126180" class="reference external">gh-126180</a>.)

</div>

<div id="id4" class="section">

### pathlib<a href="#id4" class="headerlink" title="Link to this heading">¶</a>

- Remove the ability to use <a href="../library/pathlib.html#pathlib.Path" class="reference internal" title="pathlib.Path"><span class="pre"><code class="sourceCode python">Path</code></span></a> objects as context managers. This functionality was deprecated and has had no effect since Python 3.9. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/83863" class="reference external">gh-83863</a>.)

</div>

<div id="id5" class="section">

### re<a href="#id5" class="headerlink" title="Link to this heading">¶</a>

- Remove the undocumented, deprecated, and broken <span class="pre">`re.template()`</span> function and <span class="pre">`re.TEMPLATE`</span> / <span class="pre">`re.T`</span> flag. (Contributed by Serhiy Storchaka and Nikita Sobolev in <a href="https://github.com/python/cpython/issues/105687" class="reference external">gh-105687</a>.)

</div>

<div id="tkinter-tix" class="section">

### tkinter.tix<a href="#tkinter-tix" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`tkinter.tix`</span> module, deprecated in Python 3.6. The third-party Tix library which the module wrapped is unmaintained. (Contributed by Zachary Ware in <a href="https://github.com/python/cpython/issues/75552" class="reference external">gh-75552</a>.)

</div>

<div id="turtle" class="section">

### turtle<a href="#turtle" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`RawTurtle.settiltangle()`</span> method, deprecated in the documentation since Python 3.1 and at runtime since Python 3.11. (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/104876" class="reference external">gh-104876</a>.)

</div>

<div id="id6" class="section">

### typing<a href="#id6" class="headerlink" title="Link to this heading">¶</a>

- Remove the <span class="pre">`typing.io`</span> and <span class="pre">`typing.re`</span> namespaces, deprecated since Python 3.8. The items in those namespaces can be imported directly from the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module. (Contributed by Sebastian Rittau in <a href="https://github.com/python/cpython/issues/92871" class="reference external">gh-92871</a>.)

- Remove the keyword-argument method of creating <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">TypedDict</code></span></a> types, deprecated in Python 3.11. (Contributed by Tomas Roun in <a href="https://github.com/python/cpython/issues/104786" class="reference external">gh-104786</a>.)

</div>

<div id="unittest" class="section">

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

- Remove the following <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> functions, deprecated in Python 3.11:

  - <span class="pre">`unittest.findTestCases()`</span>

  - <span class="pre">`unittest.makeSuite()`</span>

  - <span class="pre">`unittest.getTestCaseNames()`</span>

  Use <a href="../library/unittest.html#unittest.TestLoader" class="reference internal" title="unittest.TestLoader"><span class="pre"><code class="sourceCode python">TestLoader</code></span></a> methods instead:

  - <a href="../library/unittest.html#unittest.TestLoader.loadTestsFromModule" class="reference internal" title="unittest.TestLoader.loadTestsFromModule"><span class="pre"><code class="sourceCode python">loadTestsFromModule()</code></span></a>

  - <a href="../library/unittest.html#unittest.TestLoader.loadTestsFromTestCase" class="reference internal" title="unittest.TestLoader.loadTestsFromTestCase"><span class="pre"><code class="sourceCode python">loadTestsFromTestCase()</code></span></a>

  - <a href="../library/unittest.html#unittest.TestLoader.getTestCaseNames" class="reference internal" title="unittest.TestLoader.getTestCaseNames"><span class="pre"><code class="sourceCode python">getTestCaseNames()</code></span></a>

  (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/104835" class="reference external">gh-104835</a>.)

- Remove the untested and undocumented <span class="pre">`TestProgram.usageExit()`</span> method, deprecated in Python 3.11. (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/104992" class="reference external">gh-104992</a>.)

</div>

<div id="urllib" class="section">

### urllib<a href="#urllib" class="headerlink" title="Link to this heading">¶</a>

- Remove the *cafile*, *capath*, and *cadefault* parameters of the <a href="../library/urllib.request.html#urllib.request.urlopen" class="reference internal" title="urllib.request.urlopen"><span class="pre"><code class="sourceCode python">urllib.request.urlopen()</code></span></a> function, deprecated in Python 3.6. Use the *context* parameter instead with an <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> instance. The <a href="../library/ssl.html#ssl.SSLContext.load_cert_chain" class="reference internal" title="ssl.SSLContext.load_cert_chain"><span class="pre"><code class="sourceCode python">ssl.SSLContext.load_cert_chain()</code></span></a> function can be used to load specific certificates, or let <a href="../library/ssl.html#ssl.create_default_context" class="reference internal" title="ssl.create_default_context"><span class="pre"><code class="sourceCode python">ssl.create_default_context()</code></span></a> select the operating system’s trusted certificate authority (CA) certificates. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105382" class="reference external">gh-105382</a>.)

</div>

<div id="webbrowser" class="section">

### webbrowser<a href="#webbrowser" class="headerlink" title="Link to this heading">¶</a>

- Remove the untested and undocumented <span class="pre">`MacOSX`</span> class, deprecated in Python 3.11. Use the <span class="pre">`MacOSXOSAScript`</span> class (introduced in Python 3.2) instead. (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/104804" class="reference external">gh-104804</a>.)

- Remove the deprecated <span class="pre">`MacOSXOSAScript._name`</span> attribute. Use the <a href="../library/webbrowser.html#webbrowser.controller.name" class="reference internal" title="webbrowser.controller.name"><span class="pre"><code class="sourceCode python">MacOSXOSAScript.name</code></span></a> attribute instead. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/105546" class="reference external">gh-105546</a>.)

</div>

</div>

<div id="new-deprecations" class="section">

## New Deprecations<a href="#new-deprecations" class="headerlink" title="Link to this heading">¶</a>

- <a href="../reference/datamodel.html#user-defined-funcs" class="reference internal"><span class="std std-ref">User-defined functions</span></a>:

  - Deprecate assignment to a function’s <a href="../reference/datamodel.html#function.__code__" class="reference internal" title="function.__code__"><span class="pre"><code class="sourceCode python">__code__</code></span></a> attribute, where the new code object’s type does not match the function’s type. The different types are: plain function, generator, async generator, and coroutine. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/81137" class="reference external">gh-81137</a>.)

- <a href="../library/array.html#module-array" class="reference internal" title="array: Space efficient arrays of uniformly typed numeric values."><span class="pre"><code class="sourceCode python">array</code></span></a>:

  - Deprecate the <span class="pre">`'u'`</span> format code (<span class="pre">`wchar_t`</span>) at runtime. This format code has been deprecated in documentation since Python 3.3, and will be removed in Python 3.16. Use the <span class="pre">`'w'`</span> format code (<a href="../c-api/unicode.html#c.Py_UCS4" class="reference internal" title="Py_UCS4"><span class="pre"><code class="sourceCode c">Py_UCS4</code></span></a>) for Unicode characters instead. (Contributed by Hugo van Kemenade in <a href="https://github.com/python/cpython/issues/80480" class="reference external">gh-80480</a>.)

- <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a>:

  - Deprecate the undocumented <span class="pre">`SetPointerType()`</span> function, to be removed in Python 3.15. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105733" class="reference external">gh-105733</a>.)

  - <a href="../glossary.html#term-soft-deprecated" class="reference internal"><span class="xref std std-term">Soft-deprecate</span></a> the <a href="../library/ctypes.html#ctypes.ARRAY" class="reference internal" title="ctypes.ARRAY"><span class="pre"><code class="sourceCode python">ARRAY()</code></span></a> function in favour of <span class="pre">`type`</span>` `<span class="pre">`*`</span>` `<span class="pre">`length`</span> multiplication. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105733" class="reference external">gh-105733</a>.)

- <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a>:

  - Deprecate the non-standard and undocumented <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> format specifier <span class="pre">`'N'`</span>, which is only supported in the <span class="pre">`decimal`</span> module’s C implementation. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/89902" class="reference external">gh-89902</a>.)

- <a href="../library/dis.html#module-dis" class="reference internal" title="dis: Disassembler for Python bytecode."><span class="pre"><code class="sourceCode python">dis</code></span></a>:

  - Deprecate the <span class="pre">`HAVE_ARGUMENT`</span> separator. Check membership in <a href="../library/dis.html#dis.hasarg" class="reference internal" title="dis.hasarg"><span class="pre"><code class="sourceCode python">hasarg</code></span></a> instead. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/109319" class="reference external">gh-109319</a>.)

- <a href="../library/gettext.html#module-gettext" class="reference internal" title="gettext: Multilingual internationalization services."><span class="pre"><code class="sourceCode python">gettext</code></span></a>:

  - Deprecate non-integer numbers as arguments to functions and methods that consider plural forms in the <span class="pre">`gettext`</span> module, even if no translation was found. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/88434" class="reference external">gh-88434</a>.)

- <a href="../library/glob.html#module-glob" class="reference internal" title="glob: Unix shell style pathname pattern expansion."><span class="pre"><code class="sourceCode python">glob</code></span></a>:

  - Deprecate the undocumented <span class="pre">`glob0()`</span> and <span class="pre">`glob1()`</span> functions. Use <a href="../library/glob.html#glob.glob" class="reference internal" title="glob.glob"><span class="pre"><code class="sourceCode python">glob()</code></span></a> and pass a <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like object</span></a> specifying the root directory to the *root_dir* parameter instead. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/117337" class="reference external">gh-117337</a>.)

- <a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a>:

  - Deprecate <a href="../library/http.server.html#http.server.CGIHTTPRequestHandler" class="reference internal" title="http.server.CGIHTTPRequestHandler"><span class="pre"><code class="sourceCode python">CGIHTTPRequestHandler</code></span></a>, to be removed in Python 3.15. Process-based CGI HTTP servers have been out of favor for a very long time. This code was outdated, unmaintained, and rarely used. It has a high potential for both security and functionality bugs. (Contributed by Gregory P. Smith in <a href="https://github.com/python/cpython/issues/109096" class="reference external">gh-109096</a>.)

  - Deprecate the <span class="pre">`--cgi`</span> flag to the **python -m http.server** command-line interface, to be removed in Python 3.15. (Contributed by Gregory P. Smith in <a href="https://github.com/python/cpython/issues/109096" class="reference external">gh-109096</a>.)

- <a href="../library/mimetypes.html#module-mimetypes" class="reference internal" title="mimetypes: Mapping of filename extensions to MIME types."><span class="pre"><code class="sourceCode python">mimetypes</code></span></a>:

  - <a href="../glossary.html#term-soft-deprecated" class="reference internal"><span class="xref std std-term">Soft-deprecate</span></a> file path arguments to <a href="../library/mimetypes.html#mimetypes.guess_type" class="reference internal" title="mimetypes.guess_type"><span class="pre"><code class="sourceCode python">guess_type()</code></span></a>, use <a href="../library/mimetypes.html#mimetypes.guess_file_type" class="reference internal" title="mimetypes.guess_file_type"><span class="pre"><code class="sourceCode python">guess_file_type()</code></span></a> instead. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/66543" class="reference external">gh-66543</a>.)

- <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a>:

  - Deprecate passing the optional *maxsplit*, *count*, or *flags* arguments as positional arguments to the module-level <a href="../library/re.html#re.split" class="reference internal" title="re.split"><span class="pre"><code class="sourceCode python">split()</code></span></a>, <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">sub()</code></span></a>, and <a href="../library/re.html#re.subn" class="reference internal" title="re.subn"><span class="pre"><code class="sourceCode python">subn()</code></span></a> functions. These parameters will become <a href="../glossary.html#keyword-only-parameter" class="reference internal"><span class="std std-ref">keyword-only</span></a> in a future version of Python. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/56166" class="reference external">gh-56166</a>.)

- <a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a>:

  - Deprecate <a href="../library/pathlib.html#pathlib.PurePath.is_reserved" class="reference internal" title="pathlib.PurePath.is_reserved"><span class="pre"><code class="sourceCode python">PurePath.is_reserved()</code></span></a>, to be removed in Python 3.15. Use <a href="../library/os.path.html#os.path.isreserved" class="reference internal" title="os.path.isreserved"><span class="pre"><code class="sourceCode python">os.path.isreserved()</code></span></a> to detect reserved paths on Windows. (Contributed by Barney Gale in <a href="https://github.com/python/cpython/issues/88569" class="reference external">gh-88569</a>.)

- <a href="../library/platform.html#module-platform" class="reference internal" title="platform: Retrieves as much platform identifying data as possible."><span class="pre"><code class="sourceCode python">platform</code></span></a>:

  - Deprecate <a href="../library/platform.html#platform.java_ver" class="reference internal" title="platform.java_ver"><span class="pre"><code class="sourceCode python">java_ver()</code></span></a>, to be removed in Python 3.15. This function is only useful for Jython support, has a confusing API, and is largely untested. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/116349" class="reference external">gh-116349</a>.)

- <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a>:

  - Deprecate the undocumented <span class="pre">`ispackage()`</span> function. (Contributed by Zackery Spytz in <a href="https://github.com/python/cpython/issues/64020" class="reference external">gh-64020</a>.)

- <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a>:

  - Deprecate passing more than one positional argument to the <a href="../library/sqlite3.html#sqlite3.connect" class="reference internal" title="sqlite3.connect"><span class="pre"><code class="sourceCode python"><span class="ex">connect</span>()</code></span></a> function and the <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">Connection</code></span></a> constructor. The remaining parameters will become keyword-only in Python 3.15. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/107948" class="reference external">gh-107948</a>.)

  - Deprecate passing name, number of arguments, and the callable as keyword arguments for <a href="../library/sqlite3.html#sqlite3.Connection.create_function" class="reference internal" title="sqlite3.Connection.create_function"><span class="pre"><code class="sourceCode python">Connection.create_function()</code></span></a> and <a href="../library/sqlite3.html#sqlite3.Connection.create_aggregate" class="reference internal" title="sqlite3.Connection.create_aggregate"><span class="pre"><code class="sourceCode python">Connection.create_aggregate()</code></span></a> These parameters will become positional-only in Python 3.15. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/108278" class="reference external">gh-108278</a>.)

  - Deprecate passing the callback callable by keyword for the <a href="../library/sqlite3.html#sqlite3.Connection.set_authorizer" class="reference internal" title="sqlite3.Connection.set_authorizer"><span class="pre"><code class="sourceCode python">set_authorizer()</code></span></a>, <a href="../library/sqlite3.html#sqlite3.Connection.set_progress_handler" class="reference internal" title="sqlite3.Connection.set_progress_handler"><span class="pre"><code class="sourceCode python">set_progress_handler()</code></span></a>, and <a href="../library/sqlite3.html#sqlite3.Connection.set_trace_callback" class="reference internal" title="sqlite3.Connection.set_trace_callback"><span class="pre"><code class="sourceCode python">set_trace_callback()</code></span></a> <a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">Connection</code></span></a> methods. The callback callables will become positional-only in Python 3.15. (Contributed by Erlend E. Aasland in <a href="https://github.com/python/cpython/issues/108278" class="reference external">gh-108278</a>.)

- <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a>:

  - Deprecate the <a href="../library/sys.html#sys._enablelegacywindowsfsencoding" class="reference internal" title="sys._enablelegacywindowsfsencoding"><span class="pre"><code class="sourceCode python">_enablelegacywindowsfsencoding()</code></span></a> function, to be removed in Python 3.16. Use the <span id="index-50" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONLEGACYWINDOWSFSENCODING" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONLEGACYWINDOWSFSENCODING</code></span></a> environment variable instead. (Contributed by Inada Naoki in <a href="https://github.com/python/cpython/issues/73427" class="reference external">gh-73427</a>.)

- <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>:

  - Deprecate the undocumented and unused <span class="pre">`TarFile.tarfile`</span> attribute, to be removed in Python 3.16. (Contributed in <a href="https://github.com/python/cpython/issues/115256" class="reference external">gh-115256</a>.)

- <a href="../library/traceback.html#module-traceback" class="reference internal" title="traceback: Print or retrieve a stack traceback."><span class="pre"><code class="sourceCode python">traceback</code></span></a>:

  - Deprecate the <a href="../library/traceback.html#traceback.TracebackException.exc_type" class="reference internal" title="traceback.TracebackException.exc_type"><span class="pre"><code class="sourceCode python">TracebackException.exc_type</code></span></a> attribute. Use <a href="../library/traceback.html#traceback.TracebackException.exc_type_str" class="reference internal" title="traceback.TracebackException.exc_type_str"><span class="pre"><code class="sourceCode python">TracebackException.exc_type_str</code></span></a> instead. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/112332" class="reference external">gh-112332</a>.)

- <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a>:

  - Deprecate the undocumented keyword argument syntax for creating <a href="../library/typing.html#typing.NamedTuple" class="reference internal" title="typing.NamedTuple"><span class="pre"><code class="sourceCode python">NamedTuple</code></span></a> classes (e.g. <span class="pre">`Point`</span>` `<span class="pre">`=`</span>` `<span class="pre">`NamedTuple("Point",`</span>` `<span class="pre">`x=int,`</span>` `<span class="pre">`y=int)`</span>), to be removed in Python 3.15. Use the class-based syntax or the functional syntax instead. (Contributed by Alex Waygood in <a href="https://github.com/python/cpython/issues/105566" class="reference external">gh-105566</a>.)

  - Deprecate omitting the *fields* parameter when creating a <a href="../library/typing.html#typing.NamedTuple" class="reference internal" title="typing.NamedTuple"><span class="pre"><code class="sourceCode python">NamedTuple</code></span></a> or <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">typing.TypedDict</code></span></a> class, and deprecate passing <span class="pre">`None`</span> to the *fields* parameter of both types. Python 3.15 will require a valid sequence for the *fields* parameter. To create a NamedTuple class with zero fields, use <span class="pre">`class`</span>` `<span class="pre">`NT(NamedTuple):`</span>` `<span class="pre">`pass`</span> or <span class="pre">`NT`</span>` `<span class="pre">`=`</span>` `<span class="pre">`NamedTuple("NT",`</span>` `<span class="pre">`())`</span>. To create a TypedDict class with zero fields, use <span class="pre">`class`</span>` `<span class="pre">`TD(TypedDict):`</span>` `<span class="pre">`pass`</span> or <span class="pre">`TD`</span>` `<span class="pre">`=`</span>` `<span class="pre">`TypedDict("TD",`</span>` `<span class="pre">`{})`</span>. (Contributed by Alex Waygood in <a href="https://github.com/python/cpython/issues/105566" class="reference external">gh-105566</a> and <a href="https://github.com/python/cpython/issues/105570" class="reference external">gh-105570</a>.)

  - Deprecate the <a href="../library/typing.html#typing.no_type_check_decorator" class="reference internal" title="typing.no_type_check_decorator"><span class="pre"><code class="sourceCode python">typing.no_type_check_decorator()</code></span></a> decorator function, to be removed in Python 3.15. After eight years in the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module, it has yet to be supported by any major type checker. (Contributed by Alex Waygood in <a href="https://github.com/python/cpython/issues/106309" class="reference external">gh-106309</a>.)

  - Deprecate <a href="../library/typing.html#typing.AnyStr" class="reference internal" title="typing.AnyStr"><span class="pre"><code class="sourceCode python">typing.AnyStr</code></span></a>. In Python 3.16, it will be removed from <span class="pre">`typing.__all__`</span>, and a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> will be emitted at runtime when it is imported or accessed. It will be removed entirely in Python 3.18. Use the new <a href="../reference/compound_stmts.html#type-params" class="reference internal"><span class="std std-ref">type parameter syntax</span></a> instead. (Contributed by Michael The in <a href="https://github.com/python/cpython/issues/107116" class="reference external">gh-107116</a>.)

- <a href="../library/wave.html#module-wave" class="reference internal" title="wave: Provide an interface to the WAV sound format."><span class="pre"><code class="sourceCode python">wave</code></span></a>:

  - Deprecate the <a href="../library/wave.html#wave.Wave_read.getmark" class="reference internal" title="wave.Wave_read.getmark"><span class="pre"><code class="sourceCode python">getmark()</code></span></a>, <span class="pre">`setmark()`</span>, and <a href="../library/wave.html#wave.Wave_read.getmarkers" class="reference internal" title="wave.Wave_read.getmarkers"><span class="pre"><code class="sourceCode python">getmarkers()</code></span></a> methods of the <a href="../library/wave.html#wave.Wave_read" class="reference internal" title="wave.Wave_read"><span class="pre"><code class="sourceCode python">Wave_read</code></span></a> and <a href="../library/wave.html#wave.Wave_write" class="reference internal" title="wave.Wave_write"><span class="pre"><code class="sourceCode python">Wave_write</code></span></a> classes, to be removed in Python 3.15. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105096" class="reference external">gh-105096</a>.)

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

  - <a href="../library/types.html#types.CodeType" class="reference internal" title="types.CodeType"><span class="pre"><code class="sourceCode python">types.CodeType</code></span></a>: Accessing <a href="../reference/datamodel.html#codeobject.co_lnotab" class="reference internal" title="codeobject.co_lnotab"><span class="pre"><code class="sourceCode python">co_lnotab</code></span></a> was deprecated in <span id="index-51" class="target"></span><a href="https://peps.python.org/pep-0626/" class="pep reference external"><strong>PEP 626</strong></a> since 3.10 and was planned to be removed in 3.12, but it only got a proper <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> in 3.12. May be removed in 3.15. (Contributed by Nikita Sobolev in <a href="https://github.com/python/cpython/issues/101866" class="reference external">gh-101866</a>.)

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

  - The <a href="../library/sys.html#sys._enablelegacywindowsfsencoding" class="reference internal" title="sys._enablelegacywindowsfsencoding"><span class="pre"><code class="sourceCode python">_enablelegacywindowsfsencoding()</code></span></a> function has been deprecated since Python 3.13. Use the <span id="index-52" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONLEGACYWINDOWSFSENCODING" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONLEGACYWINDOWSFSENCODING</code></span></a> environment variable instead.

- <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>:

  - The undocumented and unused <span class="pre">`TarFile.tarfile`</span> attribute has been deprecated since Python 3.13.

</div>

<div id="pending-removal-in-python-3-17" class="section">

### Pending removal in Python 3.17<a href="#pending-removal-in-python-3-17" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/collections.abc.html#module-collections.abc" class="reference internal" title="collections.abc: Abstract base classes for containers"><span class="pre"><code class="sourceCode python">collections.abc</code></span></a>:

  - <a href="../library/collections.abc.html#collections.abc.ByteString" class="reference internal" title="collections.abc.ByteString"><span class="pre"><code class="sourceCode python">collections.abc.ByteString</code></span></a> is scheduled for removal in Python 3.17.

    Use <span class="pre">`isinstance(obj,`</span>` `<span class="pre">`collections.abc.Buffer)`</span> to test if <span class="pre">`obj`</span> implements the <a href="../c-api/buffer.html#bufferobjects" class="reference internal"><span class="std std-ref">buffer protocol</span></a> at runtime. For use in type annotations, either use <a href="../library/collections.abc.html#collections.abc.Buffer" class="reference internal" title="collections.abc.Buffer"><span class="pre"><code class="sourceCode python">Buffer</code></span></a> or a union that explicitly specifies the types your code supports (e.g., <span class="pre">`bytes`</span>` `<span class="pre">`|`</span>` `<span class="pre">`bytearray`</span>` `<span class="pre">`|`</span>` `<span class="pre">`memoryview`</span>).

    <span class="pre">`ByteString`</span> was originally intended to be an abstract class that would serve as a supertype of both <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>. However, since the ABC never had any methods, knowing that an object was an instance of <span class="pre">`ByteString`</span> never actually told you anything useful about the object. Other common buffer types such as <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> were also never understood as subtypes of <span class="pre">`ByteString`</span> (either at runtime or by static type checkers).

    See <span id="index-53" class="target"></span><a href="https://peps.python.org/pep-0688/#current-options" class="pep reference external"><strong>PEP 688</strong></a> for more details. (Contributed by Shantanu Jain in <a href="https://github.com/python/cpython/issues/91896" class="reference external">gh-91896</a>.)

- <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a>:

  - Before Python 3.14, old-style unions were implemented using the private class <span class="pre">`typing._UnionGenericAlias`</span>. This class is no longer needed for the implementation, but it has been retained for backward compatibility, with removal scheduled for Python 3.17. Users should use documented introspection helpers like <a href="../library/typing.html#typing.get_origin" class="reference internal" title="typing.get_origin"><span class="pre"><code class="sourceCode python">typing.get_origin()</code></span></a> and <a href="../library/typing.html#typing.get_args" class="reference internal" title="typing.get_args"><span class="pre"><code class="sourceCode python">typing.get_args()</code></span></a> instead of relying on private implementation details.

  - <a href="../library/typing.html#typing.ByteString" class="reference internal" title="typing.ByteString"><span class="pre"><code class="sourceCode python">typing.ByteString</code></span></a>, deprecated since Python 3.9, is scheduled for removal in Python 3.17.

    Use <span class="pre">`isinstance(obj,`</span>` `<span class="pre">`collections.abc.Buffer)`</span> to test if <span class="pre">`obj`</span> implements the <a href="../c-api/buffer.html#bufferobjects" class="reference internal"><span class="std std-ref">buffer protocol</span></a> at runtime. For use in type annotations, either use <a href="../library/collections.abc.html#collections.abc.Buffer" class="reference internal" title="collections.abc.Buffer"><span class="pre"><code class="sourceCode python">Buffer</code></span></a> or a union that explicitly specifies the types your code supports (e.g., <span class="pre">`bytes`</span>` `<span class="pre">`|`</span>` `<span class="pre">`bytearray`</span>` `<span class="pre">`|`</span>` `<span class="pre">`memoryview`</span>).

    <span class="pre">`ByteString`</span> was originally intended to be an abstract class that would serve as a supertype of both <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>. However, since the ABC never had any methods, knowing that an object was an instance of <span class="pre">`ByteString`</span> never actually told you anything useful about the object. Other common buffer types such as <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> were also never understood as subtypes of <span class="pre">`ByteString`</span> (either at runtime or by static type checkers).

    See <span id="index-54" class="target"></span><a href="https://peps.python.org/pep-0688/#current-options" class="pep reference external"><strong>PEP 688</strong></a> for more details. (Contributed by Shantanu Jain in <a href="https://github.com/python/cpython/issues/91896" class="reference external">gh-91896</a>.)

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

<div id="cpython-bytecode-changes" class="section">

## CPython Bytecode Changes<a href="#cpython-bytecode-changes" class="headerlink" title="Link to this heading">¶</a>

- The oparg of <a href="../library/dis.html#opcode-YIELD_VALUE" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">YIELD_VALUE</code></span></a> is now <span class="pre">`1`</span> if the yield is part of a yield-from or await, and <span class="pre">`0`</span> otherwise. The oparg of <a href="../library/dis.html#opcode-RESUME" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">RESUME</code></span></a> was changed to add a bit indicating if the except-depth is 1, which is needed to optimize closing of generators. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/111354" class="reference external">gh-111354</a>.)

</div>

<div id="c-api-changes" class="section">

## C API Changes<a href="#c-api-changes" class="headerlink" title="Link to this heading">¶</a>

<div id="id7" class="section">

### New Features<a href="#id7" class="headerlink" title="Link to this heading">¶</a>

- Add the <a href="../c-api/monitoring.html#c-api-monitoring" class="reference internal"><span class="std std-ref">PyMonitoring C API</span></a> for generating <span id="index-55" class="target"></span><a href="https://peps.python.org/pep-0669/" class="pep reference external"><strong>PEP 669</strong></a> monitoring events:

  - <a href="../c-api/monitoring.html#c.PyMonitoringState" class="reference internal" title="PyMonitoringState"><span class="pre"><code class="sourceCode c">PyMonitoringState</code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FirePyStartEvent" class="reference internal" title="PyMonitoring_FirePyStartEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FirePyStartEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FirePyResumeEvent" class="reference internal" title="PyMonitoring_FirePyResumeEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FirePyResumeEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FirePyReturnEvent" class="reference internal" title="PyMonitoring_FirePyReturnEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FirePyReturnEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FirePyYieldEvent" class="reference internal" title="PyMonitoring_FirePyYieldEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FirePyYieldEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireCallEvent" class="reference internal" title="PyMonitoring_FireCallEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireCallEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireLineEvent" class="reference internal" title="PyMonitoring_FireLineEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireLineEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireJumpEvent" class="reference internal" title="PyMonitoring_FireJumpEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireJumpEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireBranchEvent" class="reference internal" title="PyMonitoring_FireBranchEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireBranchEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireCReturnEvent" class="reference internal" title="PyMonitoring_FireCReturnEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireCReturnEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FirePyThrowEvent" class="reference internal" title="PyMonitoring_FirePyThrowEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FirePyThrowEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireRaiseEvent" class="reference internal" title="PyMonitoring_FireRaiseEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireRaiseEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireCRaiseEvent" class="reference internal" title="PyMonitoring_FireCRaiseEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireCRaiseEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireReraiseEvent" class="reference internal" title="PyMonitoring_FireReraiseEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireReraiseEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireExceptionHandledEvent" class="reference internal" title="PyMonitoring_FireExceptionHandledEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireExceptionHandledEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FirePyUnwindEvent" class="reference internal" title="PyMonitoring_FirePyUnwindEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FirePyUnwindEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_FireStopIterationEvent" class="reference internal" title="PyMonitoring_FireStopIterationEvent"><span class="pre"><code class="sourceCode c">PyMonitoring_FireStopIterationEvent<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_EnterScope" class="reference internal" title="PyMonitoring_EnterScope"><span class="pre"><code class="sourceCode c">PyMonitoring_EnterScope<span class="op">()</span></code></span></a>

  - <a href="../c-api/monitoring.html#c.PyMonitoring_ExitScope" class="reference internal" title="PyMonitoring_ExitScope"><span class="pre"><code class="sourceCode c">PyMonitoring_ExitScope<span class="op">()</span></code></span></a>

  (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/111997" class="reference external">gh-111997</a>).

- Add <a href="../c-api/init.html#c.PyMutex" class="reference internal" title="PyMutex"><span class="pre"><code class="sourceCode c">PyMutex</code></span></a>, a lightweight mutex that occupies a single byte, and the new <a href="../c-api/init.html#c.PyMutex_Lock" class="reference internal" title="PyMutex_Lock"><span class="pre"><code class="sourceCode c">PyMutex_Lock<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyMutex_Unlock" class="reference internal" title="PyMutex_Unlock"><span class="pre"><code class="sourceCode c">PyMutex_Unlock<span class="op">()</span></code></span></a> functions. <span class="pre">`PyMutex_Lock()`</span> will release the <a href="../glossary.html#term-GIL" class="reference internal"><span class="xref std std-term">GIL</span></a> (if currently held) if the operation needs to block. (Contributed by Sam Gross in <a href="https://github.com/python/cpython/issues/108724" class="reference external">gh-108724</a>.)

- Add the <a href="../c-api/time.html#c-api-time" class="reference internal"><span class="std std-ref">PyTime C API</span></a> to provide access to system clocks:

  - <a href="../c-api/time.html#c.PyTime_t" class="reference internal" title="PyTime_t"><span class="pre"><code class="sourceCode c">PyTime_t</code></span></a>.

  - <a href="../c-api/time.html#c.PyTime_MIN" class="reference internal" title="PyTime_MIN"><span class="pre"><code class="sourceCode c">PyTime_MIN</code></span></a> and <a href="../c-api/time.html#c.PyTime_MAX" class="reference internal" title="PyTime_MAX"><span class="pre"><code class="sourceCode c">PyTime_MAX</code></span></a>.

  - <a href="../c-api/time.html#c.PyTime_AsSecondsDouble" class="reference internal" title="PyTime_AsSecondsDouble"><span class="pre"><code class="sourceCode c">PyTime_AsSecondsDouble<span class="op">()</span></code></span></a>.

  - <a href="../c-api/time.html#c.PyTime_Monotonic" class="reference internal" title="PyTime_Monotonic"><span class="pre"><code class="sourceCode c">PyTime_Monotonic<span class="op">()</span></code></span></a>.

  - <a href="../c-api/time.html#c.PyTime_MonotonicRaw" class="reference internal" title="PyTime_MonotonicRaw"><span class="pre"><code class="sourceCode c">PyTime_MonotonicRaw<span class="op">()</span></code></span></a>.

  - <a href="../c-api/time.html#c.PyTime_PerfCounter" class="reference internal" title="PyTime_PerfCounter"><span class="pre"><code class="sourceCode c">PyTime_PerfCounter<span class="op">()</span></code></span></a>.

  - <a href="../c-api/time.html#c.PyTime_PerfCounterRaw" class="reference internal" title="PyTime_PerfCounterRaw"><span class="pre"><code class="sourceCode c">PyTime_PerfCounterRaw<span class="op">()</span></code></span></a>.

  - <a href="../c-api/time.html#c.PyTime_Time" class="reference internal" title="PyTime_Time"><span class="pre"><code class="sourceCode c">PyTime_Time<span class="op">()</span></code></span></a>.

  - <a href="../c-api/time.html#c.PyTime_TimeRaw" class="reference internal" title="PyTime_TimeRaw"><span class="pre"><code class="sourceCode c">PyTime_TimeRaw<span class="op">()</span></code></span></a>.

  (Contributed by Victor Stinner and Petr Viktorin in <a href="https://github.com/python/cpython/issues/110850" class="reference external">gh-110850</a>.)

- Add the <a href="../c-api/dict.html#c.PyDict_ContainsString" class="reference internal" title="PyDict_ContainsString"><span class="pre"><code class="sourceCode c">PyDict_ContainsString<span class="op">()</span></code></span></a> function with the same behavior as <a href="../c-api/dict.html#c.PyDict_Contains" class="reference internal" title="PyDict_Contains"><span class="pre"><code class="sourceCode c">PyDict_Contains<span class="op">()</span></code></span></a>, but *key* is specified as a <span class="c-expr sig sig-inline c"><span class="k">const</span><span class="w"> </span><span class="kt">char</span><span class="p">\*</span></span> UTF-8 encoded bytes string, rather than a <span class="c-expr sig sig-inline c"><a href="../c-api/structures.html#c.PyObject" class="reference internal" title="PyObject"><span class="n">PyObject</span></a><span class="p">\*</span></span>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108314" class="reference external">gh-108314</a>.)

- Add the <a href="../c-api/dict.html#c.PyDict_GetItemRef" class="reference internal" title="PyDict_GetItemRef"><span class="pre"><code class="sourceCode c">PyDict_GetItemRef<span class="op">()</span></code></span></a> and <a href="../c-api/dict.html#c.PyDict_GetItemStringRef" class="reference internal" title="PyDict_GetItemStringRef"><span class="pre"><code class="sourceCode c">PyDict_GetItemStringRef<span class="op">()</span></code></span></a> functions, which behave similarly to <a href="../c-api/dict.html#c.PyDict_GetItemWithError" class="reference internal" title="PyDict_GetItemWithError"><span class="pre"><code class="sourceCode c">PyDict_GetItemWithError<span class="op">()</span></code></span></a>, but return a <a href="../glossary.html#term-strong-reference" class="reference internal"><span class="xref std std-term">strong reference</span></a> instead of a <a href="../glossary.html#term-borrowed-reference" class="reference internal"><span class="xref std std-term">borrowed reference</span></a>. Moreover, these functions return <span class="pre">`-1`</span> on error, removing the need to check <span class="pre">`PyErr_Occurred()`</span>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/106004" class="reference external">gh-106004</a>.)

- Add the <a href="../c-api/dict.html#c.PyDict_SetDefaultRef" class="reference internal" title="PyDict_SetDefaultRef"><span class="pre"><code class="sourceCode c">PyDict_SetDefaultRef<span class="op">()</span></code></span></a> function, which behaves similarly to <a href="../c-api/dict.html#c.PyDict_SetDefault" class="reference internal" title="PyDict_SetDefault"><span class="pre"><code class="sourceCode c">PyDict_SetDefault<span class="op">()</span></code></span></a>, but returns a <a href="../glossary.html#term-strong-reference" class="reference internal"><span class="xref std std-term">strong reference</span></a> instead of a <a href="../glossary.html#term-borrowed-reference" class="reference internal"><span class="xref std std-term">borrowed reference</span></a>. This function returns <span class="pre">`-1`</span> on error, <span class="pre">`0`</span> on insertion, and <span class="pre">`1`</span> if the key was already present in the dictionary. (Contributed by Sam Gross in <a href="https://github.com/python/cpython/issues/112066" class="reference external">gh-112066</a>.)

- Add the <a href="../c-api/dict.html#c.PyDict_Pop" class="reference internal" title="PyDict_Pop"><span class="pre"><code class="sourceCode c">PyDict_Pop<span class="op">()</span></code></span></a> and <a href="../c-api/dict.html#c.PyDict_PopString" class="reference internal" title="PyDict_PopString"><span class="pre"><code class="sourceCode c">PyDict_PopString<span class="op">()</span></code></span></a> functions to remove a key from a dictionary and optionally return the removed value. This is similar to <a href="../library/stdtypes.html#dict.pop" class="reference internal" title="dict.pop"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.pop()</code></span></a>, though there is no default value, and <a href="../library/exceptions.html#KeyError" class="reference internal" title="KeyError"><span class="pre"><code class="sourceCode python"><span class="pp">KeyError</span></code></span></a> is not raised for missing keys. (Contributed by Stefan Behnel and Victor Stinner in <a href="https://github.com/python/cpython/issues/111262" class="reference external">gh-111262</a>.)

- Add the <a href="../c-api/mapping.html#c.PyMapping_GetOptionalItem" class="reference internal" title="PyMapping_GetOptionalItem"><span class="pre"><code class="sourceCode c">PyMapping_GetOptionalItem<span class="op">()</span></code></span></a> and <a href="../c-api/mapping.html#c.PyMapping_GetOptionalItemString" class="reference internal" title="PyMapping_GetOptionalItemString"><span class="pre"><code class="sourceCode c">PyMapping_GetOptionalItemString<span class="op">()</span></code></span></a> functions as alternatives to <a href="../c-api/object.html#c.PyObject_GetItem" class="reference internal" title="PyObject_GetItem"><span class="pre"><code class="sourceCode c">PyObject_GetItem<span class="op">()</span></code></span></a> and <a href="../c-api/mapping.html#c.PyMapping_GetItemString" class="reference internal" title="PyMapping_GetItemString"><span class="pre"><code class="sourceCode c">PyMapping_GetItemString<span class="op">()</span></code></span></a> respectively. The new functions do not raise <a href="../library/exceptions.html#KeyError" class="reference internal" title="KeyError"><span class="pre"><code class="sourceCode python"><span class="pp">KeyError</span></code></span></a> if the requested key is missing from the mapping. These variants are more convenient and faster if a missing key should not be treated as a failure. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/106307" class="reference external">gh-106307</a>.)

- Add the <a href="../c-api/object.html#c.PyObject_GetOptionalAttr" class="reference internal" title="PyObject_GetOptionalAttr"><span class="pre"><code class="sourceCode c">PyObject_GetOptionalAttr<span class="op">()</span></code></span></a> and <a href="../c-api/object.html#c.PyObject_GetOptionalAttrString" class="reference internal" title="PyObject_GetOptionalAttrString"><span class="pre"><code class="sourceCode c">PyObject_GetOptionalAttrString<span class="op">()</span></code></span></a> functions as alternatives to <a href="../c-api/object.html#c.PyObject_GetAttr" class="reference internal" title="PyObject_GetAttr"><span class="pre"><code class="sourceCode c">PyObject_GetAttr<span class="op">()</span></code></span></a> and <a href="../c-api/object.html#c.PyObject_GetAttrString" class="reference internal" title="PyObject_GetAttrString"><span class="pre"><code class="sourceCode c">PyObject_GetAttrString<span class="op">()</span></code></span></a> respectively. The new functions do not raise <a href="../library/exceptions.html#AttributeError" class="reference internal" title="AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> if the requested attribute is not found on the object. These variants are more convenient and faster if the missing attribute should not be treated as a failure. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/106521" class="reference external">gh-106521</a>.)

- Add the <a href="../c-api/exceptions.html#c.PyErr_FormatUnraisable" class="reference internal" title="PyErr_FormatUnraisable"><span class="pre"><code class="sourceCode c">PyErr_FormatUnraisable<span class="op">()</span></code></span></a> function as an extension to <a href="../c-api/exceptions.html#c.PyErr_WriteUnraisable" class="reference internal" title="PyErr_WriteUnraisable"><span class="pre"><code class="sourceCode c">PyErr_WriteUnraisable<span class="op">()</span></code></span></a> that allows customizing the warning message. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/108082" class="reference external">gh-108082</a>.)

- Add new functions that return a <a href="../glossary.html#term-strong-reference" class="reference internal"><span class="xref std std-term">strong reference</span></a> instead of a <a href="../glossary.html#term-borrowed-reference" class="reference internal"><span class="xref std std-term">borrowed reference</span></a> for frame locals, globals, and builtins, as part of <a href="#whatsnew313-locals-semantics" class="reference internal"><span class="std std-ref">PEP 667</span></a>:

  - <a href="../c-api/reflection.html#c.PyEval_GetFrameBuiltins" class="reference internal" title="PyEval_GetFrameBuiltins"><span class="pre"><code class="sourceCode c">PyEval_GetFrameBuiltins<span class="op">()</span></code></span></a> replaces <a href="../c-api/reflection.html#c.PyEval_GetBuiltins" class="reference internal" title="PyEval_GetBuiltins"><span class="pre"><code class="sourceCode c">PyEval_GetBuiltins<span class="op">()</span></code></span></a>

  - <a href="../c-api/reflection.html#c.PyEval_GetFrameGlobals" class="reference internal" title="PyEval_GetFrameGlobals"><span class="pre"><code class="sourceCode c">PyEval_GetFrameGlobals<span class="op">()</span></code></span></a> replaces <a href="../c-api/reflection.html#c.PyEval_GetGlobals" class="reference internal" title="PyEval_GetGlobals"><span class="pre"><code class="sourceCode c">PyEval_GetGlobals<span class="op">()</span></code></span></a>

  - <a href="../c-api/reflection.html#c.PyEval_GetFrameLocals" class="reference internal" title="PyEval_GetFrameLocals"><span class="pre"><code class="sourceCode c">PyEval_GetFrameLocals<span class="op">()</span></code></span></a> replaces <a href="../c-api/reflection.html#c.PyEval_GetLocals" class="reference internal" title="PyEval_GetLocals"><span class="pre"><code class="sourceCode c">PyEval_GetLocals<span class="op">()</span></code></span></a>

  (Contributed by Mark Shannon and Tian Gao in <a href="https://github.com/python/cpython/issues/74929" class="reference external">gh-74929</a>.)

- Add the <a href="../c-api/object.html#c.Py_GetConstant" class="reference internal" title="Py_GetConstant"><span class="pre"><code class="sourceCode c">Py_GetConstant<span class="op">()</span></code></span></a> and <a href="../c-api/object.html#c.Py_GetConstantBorrowed" class="reference internal" title="Py_GetConstantBorrowed"><span class="pre"><code class="sourceCode c">Py_GetConstantBorrowed<span class="op">()</span></code></span></a> functions to get <a href="../glossary.html#term-strong-reference" class="reference internal"><span class="xref std std-term">strong</span></a> or <a href="../glossary.html#term-borrowed-reference" class="reference internal"><span class="xref std std-term">borrowed</span></a> references to constants. For example, <span class="pre">`Py_GetConstant(Py_CONSTANT_ZERO)`</span> returns a strong reference to the constant zero. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/115754" class="reference external">gh-115754</a>.)

- Add the <a href="../c-api/import.html#c.PyImport_AddModuleRef" class="reference internal" title="PyImport_AddModuleRef"><span class="pre"><code class="sourceCode c">PyImport_AddModuleRef<span class="op">()</span></code></span></a> function as a replacement for <a href="../c-api/import.html#c.PyImport_AddModule" class="reference internal" title="PyImport_AddModule"><span class="pre"><code class="sourceCode c">PyImport_AddModule<span class="op">()</span></code></span></a> that returns a <a href="../glossary.html#term-strong-reference" class="reference internal"><span class="xref std std-term">strong reference</span></a> instead of a <a href="../glossary.html#term-borrowed-reference" class="reference internal"><span class="xref std std-term">borrowed reference</span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105922" class="reference external">gh-105922</a>.)

- Add the <a href="../c-api/init.html#c.Py_IsFinalizing" class="reference internal" title="Py_IsFinalizing"><span class="pre"><code class="sourceCode c">Py_IsFinalizing<span class="op">()</span></code></span></a> function to check whether the main Python interpreter is <a href="../glossary.html#term-interpreter-shutdown" class="reference internal"><span class="xref std std-term">shutting down</span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108014" class="reference external">gh-108014</a>.)

- Add the <a href="../c-api/list.html#c.PyList_GetItemRef" class="reference internal" title="PyList_GetItemRef"><span class="pre"><code class="sourceCode c">PyList_GetItemRef<span class="op">()</span></code></span></a> function as a replacement for <a href="../c-api/list.html#c.PyList_GetItem" class="reference internal" title="PyList_GetItem"><span class="pre"><code class="sourceCode c">PyList_GetItem<span class="op">()</span></code></span></a> that returns a <a href="../glossary.html#term-strong-reference" class="reference internal"><span class="xref std std-term">strong reference</span></a> instead of a <a href="../glossary.html#term-borrowed-reference" class="reference internal"><span class="xref std std-term">borrowed reference</span></a>. (Contributed by Sam Gross in <a href="https://github.com/python/cpython/issues/114329" class="reference external">gh-114329</a>.)

- Add the <a href="../c-api/list.html#c.PyList_Extend" class="reference internal" title="PyList_Extend"><span class="pre"><code class="sourceCode c">PyList_Extend<span class="op">()</span></code></span></a> and <a href="../c-api/list.html#c.PyList_Clear" class="reference internal" title="PyList_Clear"><span class="pre"><code class="sourceCode c">PyList_Clear<span class="op">()</span></code></span></a> functions, mirroring the Python <a href="../library/stdtypes.html#list.extend" class="reference internal" title="list.extend"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>.extend()</code></span></a> and <a href="../library/stdtypes.html#list.clear" class="reference internal" title="list.clear"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>.clear()</code></span></a> methods. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/111138" class="reference external">gh-111138</a>.)

- Add the <a href="../c-api/long.html#c.PyLong_AsInt" class="reference internal" title="PyLong_AsInt"><span class="pre"><code class="sourceCode c">PyLong_AsInt<span class="op">()</span></code></span></a> function. It behaves similarly to <a href="../c-api/long.html#c.PyLong_AsLong" class="reference internal" title="PyLong_AsLong"><span class="pre"><code class="sourceCode c">PyLong_AsLong<span class="op">()</span></code></span></a>, but stores the result in a C <span class="c-expr sig sig-inline c"><span class="kt">int</span></span> instead of a C <span class="c-expr sig sig-inline c"><span class="kt">long</span></span>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108014" class="reference external">gh-108014</a>.)

- Add the <a href="../c-api/long.html#c.PyLong_AsNativeBytes" class="reference internal" title="PyLong_AsNativeBytes"><span class="pre"><code class="sourceCode c">PyLong_AsNativeBytes<span class="op">()</span></code></span></a>, <a href="../c-api/long.html#c.PyLong_FromNativeBytes" class="reference internal" title="PyLong_FromNativeBytes"><span class="pre"><code class="sourceCode c">PyLong_FromNativeBytes<span class="op">()</span></code></span></a>, and <a href="../c-api/long.html#c.PyLong_FromUnsignedNativeBytes" class="reference internal" title="PyLong_FromUnsignedNativeBytes"><span class="pre"><code class="sourceCode c">PyLong_FromUnsignedNativeBytes<span class="op">()</span></code></span></a> functions to simplify converting between native integer types and Python <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> objects. (Contributed by Steve Dower in <a href="https://github.com/python/cpython/issues/111140" class="reference external">gh-111140</a>.)

- Add <a href="../c-api/module.html#c.PyModule_Add" class="reference internal" title="PyModule_Add"><span class="pre"><code class="sourceCode c">PyModule_Add<span class="op">()</span></code></span></a> function, which is similar to <a href="../c-api/module.html#c.PyModule_AddObjectRef" class="reference internal" title="PyModule_AddObjectRef"><span class="pre"><code class="sourceCode c">PyModule_AddObjectRef<span class="op">()</span></code></span></a> and <a href="../c-api/module.html#c.PyModule_AddObject" class="reference internal" title="PyModule_AddObject"><span class="pre"><code class="sourceCode c">PyModule_AddObject<span class="op">()</span></code></span></a>, but always steals a reference to the value. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/86493" class="reference external">gh-86493</a>.)

- Add the <a href="../c-api/hash.html#c.PyObject_GenericHash" class="reference internal" title="PyObject_GenericHash"><span class="pre"><code class="sourceCode c">PyObject_GenericHash<span class="op">()</span></code></span></a> function that implements the default hashing function of a Python object. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/113024" class="reference external">gh-113024</a>.)

- Add the <a href="../c-api/hash.html#c.Py_HashPointer" class="reference internal" title="Py_HashPointer"><span class="pre"><code class="sourceCode c">Py_HashPointer<span class="op">()</span></code></span></a> function to hash a raw pointer. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/111545" class="reference external">gh-111545</a>.)

- Add the <a href="../c-api/object.html#c.PyObject_VisitManagedDict" class="reference internal" title="PyObject_VisitManagedDict"><span class="pre"><code class="sourceCode c">PyObject_VisitManagedDict<span class="op">()</span></code></span></a> and <a href="../c-api/object.html#c.PyObject_ClearManagedDict" class="reference internal" title="PyObject_ClearManagedDict"><span class="pre"><code class="sourceCode c">PyObject_ClearManagedDict<span class="op">()</span></code></span></a> functions. which must be called by the traverse and clear functions of a type using the <a href="../c-api/typeobj.html#c.Py_TPFLAGS_MANAGED_DICT" class="reference internal" title="Py_TPFLAGS_MANAGED_DICT"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_MANAGED_DICT</code></span></a> flag. The <a href="https://github.com/python/pythoncapi-compat/" class="reference external">pythoncapi-compat project</a> can be used to use these functions with Python 3.11 and 3.12. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/107073" class="reference external">gh-107073</a>.)

- Add the <a href="../c-api/init.html#c.PyRefTracer_SetTracer" class="reference internal" title="PyRefTracer_SetTracer"><span class="pre"><code class="sourceCode c">PyRefTracer_SetTracer<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyRefTracer_GetTracer" class="reference internal" title="PyRefTracer_GetTracer"><span class="pre"><code class="sourceCode c">PyRefTracer_GetTracer<span class="op">()</span></code></span></a> functions, which enable tracking object creation and destruction in the same way that the <a href="../library/tracemalloc.html#module-tracemalloc" class="reference internal" title="tracemalloc: Trace memory allocations."><span class="pre"><code class="sourceCode python">tracemalloc</code></span></a> module does. (Contributed by Pablo Galindo in <a href="https://github.com/python/cpython/issues/93502" class="reference external">gh-93502</a>.)

- Add the <a href="../c-api/sys.html#c.PySys_AuditTuple" class="reference internal" title="PySys_AuditTuple"><span class="pre"><code class="sourceCode c">PySys_AuditTuple<span class="op">()</span></code></span></a> function as an alternative to <a href="../c-api/sys.html#c.PySys_Audit" class="reference internal" title="PySys_Audit"><span class="pre"><code class="sourceCode c">PySys_Audit<span class="op">()</span></code></span></a> that takes event arguments as a Python <a href="../library/stdtypes.html#tuple" class="reference internal" title="tuple"><span class="pre"><code class="sourceCode python"><span class="bu">tuple</span></code></span></a> object. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/85283" class="reference external">gh-85283</a>.)

- Add the <a href="../c-api/init.html#c.PyThreadState_GetUnchecked" class="reference internal" title="PyThreadState_GetUnchecked"><span class="pre"><code class="sourceCode c">PyThreadState_GetUnchecked<span class="op">()</span></code></span></a> function as an alternative to <a href="../c-api/init.html#c.PyThreadState_Get" class="reference internal" title="PyThreadState_Get"><span class="pre"><code class="sourceCode c">PyThreadState_Get<span class="op">()</span></code></span></a> that doesn’t kill the process with a fatal error if it is <span class="pre">`NULL`</span>. The caller is responsible for checking if the result is <span class="pre">`NULL`</span>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108867" class="reference external">gh-108867</a>.)

- Add the <a href="../c-api/type.html#c.PyType_GetFullyQualifiedName" class="reference internal" title="PyType_GetFullyQualifiedName"><span class="pre"><code class="sourceCode c">PyType_GetFullyQualifiedName<span class="op">()</span></code></span></a> function to get the type’s fully qualified name. The module name is prepended if <a href="../reference/datamodel.html#type.__module__" class="reference internal" title="type.__module__"><span class="pre"><code class="sourceCode python"><span class="bu">type</span>.__module__</code></span></a> is a string and is not equal to either <span class="pre">`'builtins'`</span> or <span class="pre">`'__main__'`</span>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/111696" class="reference external">gh-111696</a>.)

- Add the <a href="../c-api/type.html#c.PyType_GetModuleName" class="reference internal" title="PyType_GetModuleName"><span class="pre"><code class="sourceCode c">PyType_GetModuleName<span class="op">()</span></code></span></a> function to get the type’s module name. This is equivalent to getting the <a href="../reference/datamodel.html#type.__module__" class="reference internal" title="type.__module__"><span class="pre"><code class="sourceCode python"><span class="bu">type</span>.__module__</code></span></a> attribute. (Contributed by Eric Snow and Victor Stinner in <a href="https://github.com/python/cpython/issues/111696" class="reference external">gh-111696</a>.)

- Add the <a href="../c-api/unicode.html#c.PyUnicode_EqualToUTF8AndSize" class="reference internal" title="PyUnicode_EqualToUTF8AndSize"><span class="pre"><code class="sourceCode c">PyUnicode_EqualToUTF8AndSize<span class="op">()</span></code></span></a> and <a href="../c-api/unicode.html#c.PyUnicode_EqualToUTF8" class="reference internal" title="PyUnicode_EqualToUTF8"><span class="pre"><code class="sourceCode c">PyUnicode_EqualToUTF8<span class="op">()</span></code></span></a> functions to compare a Unicode object with a <span class="c-expr sig sig-inline c"><span class="k">const</span><span class="w"> </span><span class="kt">char</span><span class="p">\*</span></span> UTF-8 encoded string and <span class="pre">`1`</span> if they are equal or <span class="pre">`0`</span> otherwise. These functions do not raise exceptions. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/110289" class="reference external">gh-110289</a>.)

- Add the <a href="../c-api/weakref.html#c.PyWeakref_GetRef" class="reference internal" title="PyWeakref_GetRef"><span class="pre"><code class="sourceCode c">PyWeakref_GetRef<span class="op">()</span></code></span></a> function as an alternative to <a href="../c-api/weakref.html#c.PyWeakref_GetObject" class="reference internal" title="PyWeakref_GetObject"><span class="pre"><code class="sourceCode c">PyWeakref_GetObject<span class="op">()</span></code></span></a> that returns a <a href="../glossary.html#term-strong-reference" class="reference internal"><span class="xref std std-term">strong reference</span></a> or <span class="pre">`NULL`</span> if the referent is no longer live. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105927" class="reference external">gh-105927</a>.)

- Add fixed variants of functions which silently ignore errors:

  - <a href="../c-api/object.html#c.PyObject_HasAttrWithError" class="reference internal" title="PyObject_HasAttrWithError"><span class="pre"><code class="sourceCode c">PyObject_HasAttrWithError<span class="op">()</span></code></span></a> replaces <a href="../c-api/object.html#c.PyObject_HasAttr" class="reference internal" title="PyObject_HasAttr"><span class="pre"><code class="sourceCode c">PyObject_HasAttr<span class="op">()</span></code></span></a>.

  - <a href="../c-api/object.html#c.PyObject_HasAttrStringWithError" class="reference internal" title="PyObject_HasAttrStringWithError"><span class="pre"><code class="sourceCode c">PyObject_HasAttrStringWithError<span class="op">()</span></code></span></a> replaces <a href="../c-api/object.html#c.PyObject_HasAttrString" class="reference internal" title="PyObject_HasAttrString"><span class="pre"><code class="sourceCode c">PyObject_HasAttrString<span class="op">()</span></code></span></a>.

  - <a href="../c-api/mapping.html#c.PyMapping_HasKeyWithError" class="reference internal" title="PyMapping_HasKeyWithError"><span class="pre"><code class="sourceCode c">PyMapping_HasKeyWithError<span class="op">()</span></code></span></a> replaces <a href="../c-api/mapping.html#c.PyMapping_HasKey" class="reference internal" title="PyMapping_HasKey"><span class="pre"><code class="sourceCode c">PyMapping_HasKey<span class="op">()</span></code></span></a>.

  - <a href="../c-api/mapping.html#c.PyMapping_HasKeyStringWithError" class="reference internal" title="PyMapping_HasKeyStringWithError"><span class="pre"><code class="sourceCode c">PyMapping_HasKeyStringWithError<span class="op">()</span></code></span></a> replaces <a href="../c-api/mapping.html#c.PyMapping_HasKeyString" class="reference internal" title="PyMapping_HasKeyString"><span class="pre"><code class="sourceCode c">PyMapping_HasKeyString<span class="op">()</span></code></span></a>.

  The new functions return <span class="pre">`-1`</span> for errors and the standard <span class="pre">`1`</span> for true and <span class="pre">`0`</span> for false.

  (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/108511" class="reference external">gh-108511</a>.)

</div>

<div id="changed-c-apis" class="section">

### Changed C APIs<a href="#changed-c-apis" class="headerlink" title="Link to this heading">¶</a>

- The *keywords* parameter of <a href="../c-api/arg.html#c.PyArg_ParseTupleAndKeywords" class="reference internal" title="PyArg_ParseTupleAndKeywords"><span class="pre"><code class="sourceCode c">PyArg_ParseTupleAndKeywords<span class="op">()</span></code></span></a> and <a href="../c-api/arg.html#c.PyArg_VaParseTupleAndKeywords" class="reference internal" title="PyArg_VaParseTupleAndKeywords"><span class="pre"><code class="sourceCode c">PyArg_VaParseTupleAndKeywords<span class="op">()</span></code></span></a> now has type <span class="c-expr sig sig-inline c"><span class="kt">char</span><span class="w"> </span><span class="p">\*</span><span class="k">const</span><span class="p">\*</span></span> in C and <span class="c-expr sig sig-inline c"><span class="k">const</span><span class="w"> </span><span class="kt">char</span><span class="w"> </span><span class="p">\*</span><span class="k">const</span><span class="p">\*</span></span> in C++, instead of <span class="c-expr sig sig-inline c"><span class="kt">char</span><span class="p">\*</span><span class="p">\*</span></span>. In C++, this makes these functions compatible with arguments of type <span class="c-expr sig sig-inline c"><span class="k">const</span><span class="w"> </span><span class="kt">char</span><span class="w"> </span><span class="p">\*</span><span class="k">const</span><span class="p">\*</span></span>, <span class="c-expr sig sig-inline c"><span class="k">const</span><span class="w"> </span><span class="kt">char</span><span class="p">\*</span><span class="p">\*</span></span>, or <span class="c-expr sig sig-inline c"><span class="kt">char</span><span class="w"> </span><span class="p">\*</span><span class="k">const</span><span class="p">\*</span></span> without an explicit type cast. In C, the functions only support arguments of type <span class="c-expr sig sig-inline c"><span class="kt">char</span><span class="w"> </span><span class="p">\*</span><span class="k">const</span><span class="p">\*</span></span>. This can be overridden with the <a href="../c-api/arg.html#c.PY_CXX_CONST" class="reference internal" title="PY_CXX_CONST"><span class="pre"><code class="sourceCode c">PY_CXX_CONST</code></span></a> macro. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/65210" class="reference external">gh-65210</a>.)

- <a href="../c-api/arg.html#c.PyArg_ParseTupleAndKeywords" class="reference internal" title="PyArg_ParseTupleAndKeywords"><span class="pre"><code class="sourceCode c">PyArg_ParseTupleAndKeywords<span class="op">()</span></code></span></a> now supports non-ASCII keyword parameter names. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/110815" class="reference external">gh-110815</a>.)

- The <span class="pre">`PyCode_GetFirstFree()`</span> function is now unstable API and is now named <a href="../c-api/code.html#c.PyUnstable_Code_GetFirstFree" class="reference internal" title="PyUnstable_Code_GetFirstFree"><span class="pre"><code class="sourceCode c">PyUnstable_Code_GetFirstFree<span class="op">()</span></code></span></a>. (Contributed by Bogdan Romanyuk in <a href="https://github.com/python/cpython/issues/115781" class="reference external">gh-115781</a>.)

- The <a href="../c-api/dict.html#c.PyDict_GetItem" class="reference internal" title="PyDict_GetItem"><span class="pre"><code class="sourceCode c">PyDict_GetItem<span class="op">()</span></code></span></a>, <a href="../c-api/dict.html#c.PyDict_GetItemString" class="reference internal" title="PyDict_GetItemString"><span class="pre"><code class="sourceCode c">PyDict_GetItemString<span class="op">()</span></code></span></a>, <a href="../c-api/mapping.html#c.PyMapping_HasKey" class="reference internal" title="PyMapping_HasKey"><span class="pre"><code class="sourceCode c">PyMapping_HasKey<span class="op">()</span></code></span></a>, <a href="../c-api/mapping.html#c.PyMapping_HasKeyString" class="reference internal" title="PyMapping_HasKeyString"><span class="pre"><code class="sourceCode c">PyMapping_HasKeyString<span class="op">()</span></code></span></a>, <a href="../c-api/object.html#c.PyObject_HasAttr" class="reference internal" title="PyObject_HasAttr"><span class="pre"><code class="sourceCode c">PyObject_HasAttr<span class="op">()</span></code></span></a>, <a href="../c-api/object.html#c.PyObject_HasAttrString" class="reference internal" title="PyObject_HasAttrString"><span class="pre"><code class="sourceCode c">PyObject_HasAttrString<span class="op">()</span></code></span></a>, and <a href="../c-api/sys.html#c.PySys_GetObject" class="reference internal" title="PySys_GetObject"><span class="pre"><code class="sourceCode c">PySys_GetObject<span class="op">()</span></code></span></a> functions, each of which clears all errors which occurred when calling them now reports these errors using <a href="../library/sys.html#sys.unraisablehook" class="reference internal" title="sys.unraisablehook"><span class="pre"><code class="sourceCode python">sys.unraisablehook()</code></span></a>. You may replace them with other functions as recommended in the documentation. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/106672" class="reference external">gh-106672</a>.)

- Add support for the <span class="pre">`%T`</span>, <span class="pre">`%#T`</span>, <span class="pre">`%N`</span> and <span class="pre">`%#N`</span> formats to <a href="../c-api/unicode.html#c.PyUnicode_FromFormat" class="reference internal" title="PyUnicode_FromFormat"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormat<span class="op">()</span></code></span></a>:

  - <span class="pre">`%T`</span>: Get the fully qualified name of an object type

  - <span class="pre">`%#T`</span>: As above, but use a colon as the separator

  - <span class="pre">`%N`</span>: Get the fully qualified name of a type

  - <span class="pre">`%#N`</span>: As above, but use a colon as the separator

  See <span id="index-56" class="target"></span><a href="https://peps.python.org/pep-0737/" class="pep reference external"><strong>PEP 737</strong></a> for more information. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/111696" class="reference external">gh-111696</a>.)

- You no longer have to define the <span class="pre">`PY_SSIZE_T_CLEAN`</span> macro before including <span class="pre">`Python.h`</span> when using <span class="pre">`#`</span> formats in <a href="../c-api/arg.html#arg-parsing-string-and-buffers" class="reference internal"><span class="std std-ref">format codes</span></a>. APIs accepting the format codes always use <span class="pre">`Py_ssize_t`</span> for <span class="pre">`#`</span> formats. (Contributed by Inada Naoki in <a href="https://github.com/python/cpython/issues/104922" class="reference external">gh-104922</a>.)

- If Python is built in <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">debug mode</span></a> or <a href="../using/configure.html#cmdoption-with-assertions" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">with</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">assertions</code></span></a>, <a href="../c-api/tuple.html#c.PyTuple_SET_ITEM" class="reference internal" title="PyTuple_SET_ITEM"><span class="pre"><code class="sourceCode c">PyTuple_SET_ITEM<span class="op">()</span></code></span></a> and <a href="../c-api/list.html#c.PyList_SET_ITEM" class="reference internal" title="PyList_SET_ITEM"><span class="pre"><code class="sourceCode c">PyList_SET_ITEM<span class="op">()</span></code></span></a> now check the index argument with an assertion. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/106168" class="reference external">gh-106168</a>.)

</div>

<div id="limited-c-api-changes" class="section">

### Limited C API Changes<a href="#limited-c-api-changes" class="headerlink" title="Link to this heading">¶</a>

- The following functions are now included in the Limited C API:

  - <a href="../c-api/memory.html#c.PyMem_RawMalloc" class="reference internal" title="PyMem_RawMalloc"><span class="pre"><code class="sourceCode c">PyMem_RawMalloc<span class="op">()</span></code></span></a>

  - <a href="../c-api/memory.html#c.PyMem_RawCalloc" class="reference internal" title="PyMem_RawCalloc"><span class="pre"><code class="sourceCode c">PyMem_RawCalloc<span class="op">()</span></code></span></a>

  - <a href="../c-api/memory.html#c.PyMem_RawRealloc" class="reference internal" title="PyMem_RawRealloc"><span class="pre"><code class="sourceCode c">PyMem_RawRealloc<span class="op">()</span></code></span></a>

  - <a href="../c-api/memory.html#c.PyMem_RawFree" class="reference internal" title="PyMem_RawFree"><span class="pre"><code class="sourceCode c">PyMem_RawFree<span class="op">()</span></code></span></a>

  - <a href="../c-api/sys.html#c.PySys_Audit" class="reference internal" title="PySys_Audit"><span class="pre"><code class="sourceCode c">PySys_Audit<span class="op">()</span></code></span></a>

  - <a href="../c-api/sys.html#c.PySys_AuditTuple" class="reference internal" title="PySys_AuditTuple"><span class="pre"><code class="sourceCode c">PySys_AuditTuple<span class="op">()</span></code></span></a>

  - <a href="../c-api/type.html#c.PyType_GetModuleByDef" class="reference internal" title="PyType_GetModuleByDef"><span class="pre"><code class="sourceCode c">PyType_GetModuleByDef<span class="op">()</span></code></span></a>

  (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/85283" class="reference external">gh-85283</a>, <a href="https://github.com/python/cpython/issues/85283" class="reference external">gh-85283</a>, and <a href="https://github.com/python/cpython/issues/116936" class="reference external">gh-116936</a>.)

- Python built with <a href="../using/configure.html#cmdoption-with-trace-refs" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--with-trace-refs</code></span></a> (tracing references) now supports the <a href="../c-api/stable.html#limited-c-api" class="reference internal"><span class="std std-ref">Limited API</span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108634" class="reference external">gh-108634</a>.)

</div>

<div id="removed-c-apis" class="section">

### Removed C APIs<a href="#removed-c-apis" class="headerlink" title="Link to this heading">¶</a>

- Remove several functions, macros, variables, etc with names prefixed by <span class="pre">`_Py`</span> or <span class="pre">`_PY`</span> (which are considered private). If your project is affected by one of these removals and you believe that the removed API should remain available, please <a href="../bugs.html#using-the-tracker" class="reference internal"><span class="std std-ref">open a new issue</span></a> to request a public C API and add <span class="pre">`cc:`</span>` `<span class="pre">`@vstinner`</span> to the issue to notify Victor Stinner. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/106320" class="reference external">gh-106320</a>.)

- Remove old buffer protocols deprecated in Python 3.0. Use <a href="../c-api/buffer.html#bufferobjects" class="reference internal"><span class="std std-ref">Buffer Protocol</span></a> instead.

  - <span class="pre">`PyObject_CheckReadBuffer()`</span>: Use <a href="../c-api/buffer.html#c.PyObject_CheckBuffer" class="reference internal" title="PyObject_CheckBuffer"><span class="pre"><code class="sourceCode c">PyObject_CheckBuffer<span class="op">()</span></code></span></a> to test whether the object supports the buffer protocol. Note that <a href="../c-api/buffer.html#c.PyObject_CheckBuffer" class="reference internal" title="PyObject_CheckBuffer"><span class="pre"><code class="sourceCode c">PyObject_CheckBuffer<span class="op">()</span></code></span></a> doesn’t guarantee that <a href="../c-api/buffer.html#c.PyObject_GetBuffer" class="reference internal" title="PyObject_GetBuffer"><span class="pre"><code class="sourceCode c">PyObject_GetBuffer<span class="op">()</span></code></span></a> will succeed. To test if the object is actually readable, see the next example of <a href="../c-api/buffer.html#c.PyObject_GetBuffer" class="reference internal" title="PyObject_GetBuffer"><span class="pre"><code class="sourceCode c">PyObject_GetBuffer<span class="op">()</span></code></span></a>.

  - <span class="pre">`PyObject_AsCharBuffer()`</span>, <span class="pre">`PyObject_AsReadBuffer()`</span>: Use <a href="../c-api/buffer.html#c.PyObject_GetBuffer" class="reference internal" title="PyObject_GetBuffer"><span class="pre"><code class="sourceCode c">PyObject_GetBuffer<span class="op">()</span></code></span></a> and <a href="../c-api/buffer.html#c.PyBuffer_Release" class="reference internal" title="PyBuffer_Release"><span class="pre"><code class="sourceCode c">PyBuffer_Release<span class="op">()</span></code></span></a> instead:

    <div class="highlight-c notranslate">

    <div class="highlight">

        Py_buffer view;
        if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE) < 0) {
            return NULL;
        }
        // Use `view.buf` and `view.len` to read from the buffer.
        // You may need to cast buf as `(const char*)view.buf`.
        PyBuffer_Release(&view);

    </div>

    </div>

  - <span class="pre">`PyObject_AsWriteBuffer()`</span>: Use <a href="../c-api/buffer.html#c.PyObject_GetBuffer" class="reference internal" title="PyObject_GetBuffer"><span class="pre"><code class="sourceCode c">PyObject_GetBuffer<span class="op">()</span></code></span></a> and <a href="../c-api/buffer.html#c.PyBuffer_Release" class="reference internal" title="PyBuffer_Release"><span class="pre"><code class="sourceCode c">PyBuffer_Release<span class="op">()</span></code></span></a> instead:

    <div class="highlight-c notranslate">

    <div class="highlight">

        Py_buffer view;
        if (PyObject_GetBuffer(obj, &view, PyBUF_WRITABLE) < 0) {
            return NULL;
        }
        // Use `view.buf` and `view.len` to write to the buffer.
        PyBuffer_Release(&view);

    </div>

    </div>

  (Contributed by Inada Naoki in <a href="https://github.com/python/cpython/issues/85275" class="reference external">gh-85275</a>.)

- Remove various functions deprecated in Python 3.9:

  - <span class="pre">`PyEval_CallObject()`</span>, <span class="pre">`PyEval_CallObjectWithKeywords()`</span>: Use <a href="../c-api/call.html#c.PyObject_CallNoArgs" class="reference internal" title="PyObject_CallNoArgs"><span class="pre"><code class="sourceCode c">PyObject_CallNoArgs<span class="op">()</span></code></span></a> or <a href="../c-api/call.html#c.PyObject_Call" class="reference internal" title="PyObject_Call"><span class="pre"><code class="sourceCode c">PyObject_Call<span class="op">()</span></code></span></a> instead.

    <div class="admonition warning">

    Warning

    In <a href="../c-api/call.html#c.PyObject_Call" class="reference internal" title="PyObject_Call"><span class="pre"><code class="sourceCode c">PyObject_Call<span class="op">()</span></code></span></a>, positional arguments must be a <a href="../library/stdtypes.html#tuple" class="reference internal" title="tuple"><span class="pre"><code class="sourceCode python"><span class="bu">tuple</span></code></span></a> and must not be <span class="pre">`NULL`</span>, and keyword arguments must be a <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> or <span class="pre">`NULL`</span>, whereas the removed functions checked argument types and accepted <span class="pre">`NULL`</span> positional and keyword arguments. To replace <span class="pre">`PyEval_CallObjectWithKeywords(func,`</span>` `<span class="pre">`NULL,`</span>` `<span class="pre">`kwargs)`</span> with <a href="../c-api/call.html#c.PyObject_Call" class="reference internal" title="PyObject_Call"><span class="pre"><code class="sourceCode c">PyObject_Call<span class="op">()</span></code></span></a>, pass an empty tuple as positional arguments using <a href="../c-api/tuple.html#c.PyTuple_New" class="reference internal" title="PyTuple_New"><span class="pre"><code class="sourceCode c">PyTuple_New<span class="op">(</span><span class="dv">0</span><span class="op">)</span></code></span></a>.

    </div>

  - <span class="pre">`PyEval_CallFunction()`</span>: Use <a href="../c-api/call.html#c.PyObject_CallFunction" class="reference internal" title="PyObject_CallFunction"><span class="pre"><code class="sourceCode c">PyObject_CallFunction<span class="op">()</span></code></span></a> instead.

  - <span class="pre">`PyEval_CallMethod()`</span>: Use <a href="../c-api/call.html#c.PyObject_CallMethod" class="reference internal" title="PyObject_CallMethod"><span class="pre"><code class="sourceCode c">PyObject_CallMethod<span class="op">()</span></code></span></a> instead.

  - <span class="pre">`PyCFunction_Call()`</span>: Use <a href="../c-api/call.html#c.PyObject_Call" class="reference internal" title="PyObject_Call"><span class="pre"><code class="sourceCode c">PyObject_Call<span class="op">()</span></code></span></a> instead.

  (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105107" class="reference external">gh-105107</a>.)

- Remove the following old functions to configure the Python initialization, deprecated in Python 3.11:

  - <span class="pre">`PySys_AddWarnOptionUnicode()`</span>: Use <a href="../c-api/init_config.html#c.PyConfig.warnoptions" class="reference internal" title="PyConfig.warnoptions"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>warnoptions</code></span></a> instead.

  - <span class="pre">`PySys_AddWarnOption()`</span>: Use <a href="../c-api/init_config.html#c.PyConfig.warnoptions" class="reference internal" title="PyConfig.warnoptions"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>warnoptions</code></span></a> instead.

  - <span class="pre">`PySys_AddXOption()`</span>: Use <a href="../c-api/init_config.html#c.PyConfig.xoptions" class="reference internal" title="PyConfig.xoptions"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>xoptions</code></span></a> instead.

  - <span class="pre">`PySys_HasWarnOptions()`</span>: Use <a href="../c-api/init_config.html#c.PyConfig.xoptions" class="reference internal" title="PyConfig.xoptions"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>xoptions</code></span></a> instead.

  - <span class="pre">`PySys_SetPath()`</span>: Set <a href="../c-api/init_config.html#c.PyConfig.module_search_paths" class="reference internal" title="PyConfig.module_search_paths"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>module_search_paths</code></span></a> instead.

  - <span class="pre">`Py_SetPath()`</span>: Set <a href="../c-api/init_config.html#c.PyConfig.module_search_paths" class="reference internal" title="PyConfig.module_search_paths"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>module_search_paths</code></span></a> instead.

  - <span class="pre">`Py_SetStandardStreamEncoding()`</span>: Set <a href="../c-api/init_config.html#c.PyConfig.stdio_encoding" class="reference internal" title="PyConfig.stdio_encoding"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>stdio_encoding</code></span></a> instead, and set also maybe <a href="../c-api/init_config.html#c.PyConfig.legacy_windows_stdio" class="reference internal" title="PyConfig.legacy_windows_stdio"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>legacy_windows_stdio</code></span></a> (on Windows).

  - <span class="pre">`_Py_SetProgramFullPath()`</span>: Set <a href="../c-api/init_config.html#c.PyConfig.executable" class="reference internal" title="PyConfig.executable"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>executable</code></span></a> instead.

  Use the new <a href="../c-api/init_config.html#c.PyConfig" class="reference internal" title="PyConfig"><span class="pre"><code class="sourceCode c">PyConfig</code></span></a> API of the <a href="../c-api/init_config.html#init-config" class="reference internal"><span class="std std-ref">Python Initialization Configuration</span></a> instead (<span id="index-57" class="target"></span><a href="https://peps.python.org/pep-0587/" class="pep reference external"><strong>PEP 587</strong></a>), added to Python 3.8. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105145" class="reference external">gh-105145</a>.)

- Remove <span class="pre">`PyEval_AcquireLock()`</span> and <span class="pre">`PyEval_ReleaseLock()`</span> functions, deprecated in Python 3.2. They didn’t update the current thread state. They can be replaced with:

  - <a href="../c-api/init.html#c.PyEval_SaveThread" class="reference internal" title="PyEval_SaveThread"><span class="pre"><code class="sourceCode c">PyEval_SaveThread<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyEval_RestoreThread" class="reference internal" title="PyEval_RestoreThread"><span class="pre"><code class="sourceCode c">PyEval_RestoreThread<span class="op">()</span></code></span></a>;

  - low-level <a href="../c-api/init.html#c.PyEval_AcquireThread" class="reference internal" title="PyEval_AcquireThread"><span class="pre"><code class="sourceCode c">PyEval_AcquireThread<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyEval_RestoreThread" class="reference internal" title="PyEval_RestoreThread"><span class="pre"><code class="sourceCode c">PyEval_RestoreThread<span class="op">()</span></code></span></a>;

  - or <a href="../c-api/init.html#c.PyGILState_Ensure" class="reference internal" title="PyGILState_Ensure"><span class="pre"><code class="sourceCode c">PyGILState_Ensure<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyGILState_Release" class="reference internal" title="PyGILState_Release"><span class="pre"><code class="sourceCode c">PyGILState_Release<span class="op">()</span></code></span></a>.

  (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105182" class="reference external">gh-105182</a>.)

- Remove the <span class="pre">`PyEval_ThreadsInitialized()`</span> function, deprecated in Python 3.9. Since Python 3.7, <span class="pre">`Py_Initialize()`</span> always creates the GIL: calling <span class="pre">`PyEval_InitThreads()`</span> does nothing and <span class="pre">`PyEval_ThreadsInitialized()`</span> always returns non-zero. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105182" class="reference external">gh-105182</a>.)

- Remove the <span class="pre">`_PyInterpreterState_Get()`</span> alias to <a href="../c-api/init.html#c.PyInterpreterState_Get" class="reference internal" title="PyInterpreterState_Get"><span class="pre"><code class="sourceCode c">PyInterpreterState_Get<span class="op">()</span></code></span></a> which was kept for backward compatibility with Python 3.8. The <a href="https://github.com/python/pythoncapi-compat/" class="reference external">pythoncapi-compat project</a> can be used to get <a href="../c-api/init.html#c.PyInterpreterState_Get" class="reference internal" title="PyInterpreterState_Get"><span class="pre"><code class="sourceCode c">PyInterpreterState_Get<span class="op">()</span></code></span></a> on Python 3.8 and older. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/106320" class="reference external">gh-106320</a>.)

- Remove the private <span class="pre">`_PyObject_FastCall()`</span> function: use <span class="pre">`PyObject_Vectorcall()`</span> which is available since Python 3.8 (<span id="index-58" class="target"></span><a href="https://peps.python.org/pep-0590/" class="pep reference external"><strong>PEP 590</strong></a>). (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/106023" class="reference external">gh-106023</a>.)

- Remove the <span class="pre">`cpython/pytime.h`</span> header file, which only contained private functions. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/106316" class="reference external">gh-106316</a>.)

- Remove the undocumented <span class="pre">`PY_TIMEOUT_MAX`</span> constant from the limited C API. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/110014" class="reference external">gh-110014</a>.)

- Remove the old trashcan macros <span class="pre">`Py_TRASHCAN_SAFE_BEGIN`</span> and <span class="pre">`Py_TRASHCAN_SAFE_END`</span>. Replace both with the new macros <span class="pre">`Py_TRASHCAN_BEGIN`</span> and <span class="pre">`Py_TRASHCAN_END`</span>. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/105111" class="reference external">gh-105111</a>.)

</div>

<div id="deprecated-c-apis" class="section">

### Deprecated C APIs<a href="#deprecated-c-apis" class="headerlink" title="Link to this heading">¶</a>

- Deprecate old Python initialization functions:

  - <a href="../c-api/sys.html#c.PySys_ResetWarnOptions" class="reference internal" title="PySys_ResetWarnOptions"><span class="pre"><code class="sourceCode c">PySys_ResetWarnOptions<span class="op">()</span></code></span></a>: Clear <a href="../library/sys.html#sys.warnoptions" class="reference internal" title="sys.warnoptions"><span class="pre"><code class="sourceCode python">sys.warnoptions</code></span></a> and <span class="pre">`warnings.filters`</span> instead.

  - <a href="../c-api/init.html#c.Py_GetExecPrefix" class="reference internal" title="Py_GetExecPrefix"><span class="pre"><code class="sourceCode c">Py_GetExecPrefix<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.exec_prefix" class="reference internal" title="sys.exec_prefix"><span class="pre"><code class="sourceCode python">sys.exec_prefix</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetPath" class="reference internal" title="Py_GetPath"><span class="pre"><code class="sourceCode c">Py_GetPath<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetPrefix" class="reference internal" title="Py_GetPrefix"><span class="pre"><code class="sourceCode c">Py_GetPrefix<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.prefix" class="reference internal" title="sys.prefix"><span class="pre"><code class="sourceCode python">sys.prefix</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetProgramFullPath" class="reference internal" title="Py_GetProgramFullPath"><span class="pre"><code class="sourceCode c">Py_GetProgramFullPath<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.executable" class="reference internal" title="sys.executable"><span class="pre"><code class="sourceCode python">sys.executable</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetProgramName" class="reference internal" title="Py_GetProgramName"><span class="pre"><code class="sourceCode c">Py_GetProgramName<span class="op">()</span></code></span></a>: Get <a href="../library/sys.html#sys.executable" class="reference internal" title="sys.executable"><span class="pre"><code class="sourceCode python">sys.executable</code></span></a> instead.

  - <a href="../c-api/init.html#c.Py_GetPythonHome" class="reference internal" title="Py_GetPythonHome"><span class="pre"><code class="sourceCode c">Py_GetPythonHome<span class="op">()</span></code></span></a>: Get <a href="../c-api/init_config.html#c.PyConfig.home" class="reference internal" title="PyConfig.home"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>home</code></span></a> or the <span id="index-59" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONHOME" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONHOME</code></span></a> environment variable instead.

  (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105145" class="reference external">gh-105145</a>.)

- <a href="../glossary.html#term-soft-deprecated" class="reference internal"><span class="xref std std-term">Soft deprecate</span></a> the <a href="../c-api/reflection.html#c.PyEval_GetBuiltins" class="reference internal" title="PyEval_GetBuiltins"><span class="pre"><code class="sourceCode c">PyEval_GetBuiltins<span class="op">()</span></code></span></a>, <a href="../c-api/reflection.html#c.PyEval_GetGlobals" class="reference internal" title="PyEval_GetGlobals"><span class="pre"><code class="sourceCode c">PyEval_GetGlobals<span class="op">()</span></code></span></a>, and <a href="../c-api/reflection.html#c.PyEval_GetLocals" class="reference internal" title="PyEval_GetLocals"><span class="pre"><code class="sourceCode c">PyEval_GetLocals<span class="op">()</span></code></span></a> functions, which return a <a href="../glossary.html#term-borrowed-reference" class="reference internal"><span class="xref std std-term">borrowed reference</span></a>. (Soft deprecated as part of <span id="index-60" class="target"></span><a href="https://peps.python.org/pep-0667/" class="pep reference external"><strong>PEP 667</strong></a>.)

- Deprecate the <a href="../c-api/import.html#c.PyImport_ImportModuleNoBlock" class="reference internal" title="PyImport_ImportModuleNoBlock"><span class="pre"><code class="sourceCode c">PyImport_ImportModuleNoBlock<span class="op">()</span></code></span></a> function, which is just an alias to <a href="../c-api/import.html#c.PyImport_ImportModule" class="reference internal" title="PyImport_ImportModule"><span class="pre"><code class="sourceCode c">PyImport_ImportModule<span class="op">()</span></code></span></a> since Python 3.3. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105396" class="reference external">gh-105396</a>.)

- <a href="../glossary.html#term-soft-deprecated" class="reference internal"><span class="xref std std-term">Soft deprecate</span></a> the <a href="../c-api/module.html#c.PyModule_AddObject" class="reference internal" title="PyModule_AddObject"><span class="pre"><code class="sourceCode c">PyModule_AddObject<span class="op">()</span></code></span></a> function. It should be replaced with <a href="../c-api/module.html#c.PyModule_Add" class="reference internal" title="PyModule_Add"><span class="pre"><code class="sourceCode c">PyModule_Add<span class="op">()</span></code></span></a> or <a href="../c-api/module.html#c.PyModule_AddObjectRef" class="reference internal" title="PyModule_AddObjectRef"><span class="pre"><code class="sourceCode c">PyModule_AddObjectRef<span class="op">()</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/86493" class="reference external">gh-86493</a>.)

- Deprecate the old <span class="pre">`Py_UNICODE`</span> and <span class="pre">`PY_UNICODE_TYPE`</span> types and the <span class="pre">`Py_UNICODE_WIDE`</span> define. Use the <span class="pre">`wchar_t`</span> type directly instead. Since Python 3.3, <span class="pre">`Py_UNICODE`</span> and <span class="pre">`PY_UNICODE_TYPE`</span> are just aliases to <span class="pre">`wchar_t`</span>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105156" class="reference external">gh-105156</a>.)

- Deprecate the <a href="../c-api/weakref.html#c.PyWeakref_GetObject" class="reference internal" title="PyWeakref_GetObject"><span class="pre"><code class="sourceCode c">PyWeakref_GetObject<span class="op">()</span></code></span></a> and <a href="../c-api/weakref.html#c.PyWeakref_GET_OBJECT" class="reference internal" title="PyWeakref_GET_OBJECT"><span class="pre"><code class="sourceCode c">PyWeakref_GET_OBJECT<span class="op">()</span></code></span></a> functions, which return a <a href="../glossary.html#term-borrowed-reference" class="reference internal"><span class="xref std std-term">borrowed reference</span></a>. Replace them with the new <a href="../c-api/weakref.html#c.PyWeakref_GetRef" class="reference internal" title="PyWeakref_GetRef"><span class="pre"><code class="sourceCode c">PyWeakref_GetRef<span class="op">()</span></code></span></a> function, which returns a <a href="../glossary.html#term-strong-reference" class="reference internal"><span class="xref std std-term">strong reference</span></a>. The <a href="https://github.com/python/pythoncapi-compat/" class="reference external">pythoncapi-compat project</a> can be used to get <a href="../c-api/weakref.html#c.PyWeakref_GetRef" class="reference internal" title="PyWeakref_GetRef"><span class="pre"><code class="sourceCode c">PyWeakref_GetRef<span class="op">()</span></code></span></a> on Python 3.12 and older. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/105927" class="reference external">gh-105927</a>.)

<div id="id8" class="section">

#### Pending Removal in Python 3.14<a href="#id8" class="headerlink" title="Link to this heading">¶</a>

- The <span class="pre">`ma_version_tag`</span> field in <a href="../c-api/dict.html#c.PyDictObject" class="reference internal" title="PyDictObject"><span class="pre"><code class="sourceCode c">PyDictObject</code></span></a> for extension modules (<span id="index-61" class="target"></span><a href="https://peps.python.org/pep-0699/" class="pep reference external"><strong>PEP 699</strong></a>; <a href="https://github.com/python/cpython/issues/101193" class="reference external">gh-101193</a>).

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

<div id="id9" class="section">

#### Pending Removal in Python 3.15<a href="#id9" class="headerlink" title="Link to this heading">¶</a>

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

  - <a href="../c-api/init.html#c.Py_GetPythonHome" class="reference internal" title="Py_GetPythonHome"><span class="pre"><code class="sourceCode c">Py_GetPythonHome<span class="op">()</span></code></span></a>: Get <a href="../c-api/init_config.html#c.PyConfig.home" class="reference internal" title="PyConfig.home"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>home</code></span></a> or the <span id="index-62" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONHOME" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONHOME</code></span></a> environment variable instead.

</div>

<div id="id10" class="section">

#### Pending removal in Python 3.16<a href="#id10" class="headerlink" title="Link to this heading">¶</a>

- The bundled copy of <span class="pre">`libmpdec`</span>.

</div>

<div id="id11" class="section">

#### Pending Removal in Future Versions<a href="#id11" class="headerlink" title="Link to this heading">¶</a>

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

</div>

<div id="build-changes" class="section">

## Build Changes<a href="#build-changes" class="headerlink" title="Link to this heading">¶</a>

- <span class="pre">`arm64-apple-ios`</span> and <span class="pre">`arm64-apple-ios-simulator`</span> are both now <span id="index-63" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a> tier 3 platforms. (<a href="#whatsnew313-platform-support" class="reference internal"><span class="std std-ref">PEP 730</span></a> written and implementation contributed by Russell Keith-Magee in <a href="https://github.com/python/cpython/issues/114099" class="reference external">gh-114099</a>.)

- <span class="pre">`aarch64-linux-android`</span> and <span class="pre">`x86_64-linux-android`</span> are both now <span id="index-64" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a> tier 3 platforms. (<a href="#whatsnew313-platform-support" class="reference internal"><span class="std std-ref">PEP 738</span></a> written and implementation contributed by Malcolm Smith in <a href="https://github.com/python/cpython/issues/116622" class="reference external">gh-116622</a>.)

- <span class="pre">`wasm32-wasi`</span> is now a <span id="index-65" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a> tier 2 platform. (Contributed by Brett Cannon in <a href="https://github.com/python/cpython/issues/115192" class="reference external">gh-115192</a>.)

- <span class="pre">`wasm32-emscripten`</span> is no longer a <span id="index-66" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a> supported platform. (Contributed by Brett Cannon in <a href="https://github.com/python/cpython/issues/115192" class="reference external">gh-115192</a>.)

- Building CPython now requires a compiler with support for the C11 atomic library, GCC built-in atomic functions, or MSVC interlocked intrinsics.

- Autoconf 2.71 and aclocal 1.16.5 are now required to regenerate the <span class="pre">`configure`</span> script. (Contributed by Christian Heimes in <a href="https://github.com/python/cpython/issues/89886" class="reference external">gh-89886</a> and by Victor Stinner in <a href="https://github.com/python/cpython/issues/112090" class="reference external">gh-112090</a>.)

- SQLite 3.15.2 or newer is required to build the <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> extension module. (Contributed by Erlend Aasland in <a href="https://github.com/python/cpython/issues/105875" class="reference external">gh-105875</a>.)

- CPython now bundles the <a href="https://github.com/microsoft/mimalloc/" class="reference external">mimalloc library</a> by default. It is licensed under the MIT license; see <a href="../license.html#mimalloc-license" class="reference internal"><span class="std std-ref">mimalloc license</span></a>. The bundled mimalloc has custom changes, see <a href="https://github.com/python/cpython/issues/113141" class="reference external">gh-113141</a> for details. (Contributed by Dino Viehland in <a href="https://github.com/python/cpython/issues/109914" class="reference external">gh-109914</a>.)

- The <span class="pre">`configure`</span> option <a href="../using/configure.html#cmdoption-with-system-libmpdec" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--with-system-libmpdec</code></span></a> now defaults to <span class="pre">`yes`</span>. The bundled copy of <span class="pre">`libmpdec`</span> will be removed in Python 3.16.

- Python built with <span class="pre">`configure`</span> <a href="../using/configure.html#cmdoption-with-trace-refs" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--with-trace-refs</code></span></a> (tracing references) is now ABI compatible with the Python release build and <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">debug build</span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108634" class="reference external">gh-108634</a>.)

- On POSIX systems, the pkg-config (<span class="pre">`.pc`</span>) filenames now include the ABI flags. For example, the free-threaded build generates <span class="pre">`python-3.13t.pc`</span> and the debug build generates <span class="pre">`python-3.13d.pc`</span>.

- The <span class="pre">`errno`</span>, <span class="pre">`fcntl`</span>, <span class="pre">`grp`</span>, <span class="pre">`md5`</span>, <span class="pre">`pwd`</span>, <span class="pre">`resource`</span>, <span class="pre">`termios`</span>, <span class="pre">`winsound`</span>, <span class="pre">`_ctypes_test`</span>, <span class="pre">`_multiprocessing.posixshmem`</span>, <span class="pre">`_scproxy`</span>, <span class="pre">`_stat`</span>, <span class="pre">`_statistics`</span>, <span class="pre">`_testconsole`</span>, <span class="pre">`_testimportmultiple`</span> and <span class="pre">`_uuid`</span> C extensions are now built with the <a href="../c-api/stable.html#limited-c-api" class="reference internal"><span class="std std-ref">limited C API</span></a>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/85283" class="reference external">gh-85283</a>.)

</div>

<div id="porting-to-python-3-13" class="section">

## Porting to Python 3.13<a href="#porting-to-python-3-13" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="changes-in-the-python-api" class="section">

### Changes in the Python API<a href="#changes-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- <a href="#whatsnew313-locals-semantics" class="reference internal"><span class="std std-ref">PEP 667</span></a> introduces several changes to the semantics of <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> and <a href="../reference/datamodel.html#frame.f_locals" class="reference internal" title="frame.f_locals"><span class="pre"><code class="sourceCode python">f_locals</code></span></a>:

  - Calling <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> in an <a href="../glossary.html#term-optimized-scope" class="reference internal"><span class="xref std std-term">optimized scope</span></a> now produces an independent snapshot on each call, and hence no longer implicitly updates previously returned references. Obtaining the legacy CPython behavior now requires explicit calls to update the initially returned dictionary with the results of subsequent calls to <span class="pre">`locals()`</span>. Code execution functions that implicitly target <span class="pre">`locals()`</span> (such as <span class="pre">`exec`</span> and <span class="pre">`eval`</span>) must be passed an explicit namespace to access their results in an optimized scope. (Changed as part of <span id="index-67" class="target"></span><a href="https://peps.python.org/pep-0667/" class="pep reference external"><strong>PEP 667</strong></a>.)

  - Calling <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> from a comprehension at module or class scope (including via <span class="pre">`exec`</span> or <span class="pre">`eval`</span>) once more behaves as if the comprehension were running as an independent nested function (i.e. the local variables from the containing scope are not included). In Python 3.12, this had changed to include the local variables from the containing scope when implementing <span id="index-68" class="target"></span><a href="https://peps.python.org/pep-0709/" class="pep reference external"><strong>PEP 709</strong></a>. (Changed as part of <span id="index-69" class="target"></span><a href="https://peps.python.org/pep-0667/" class="pep reference external"><strong>PEP 667</strong></a>.)

  - Accessing <a href="../reference/datamodel.html#frame.f_locals" class="reference internal" title="frame.f_locals"><span class="pre"><code class="sourceCode python">FrameType.f_locals</code></span></a> in an <a href="../glossary.html#term-optimized-scope" class="reference internal"><span class="xref std std-term">optimized scope</span></a> now returns a write-through proxy rather than a snapshot that gets updated at ill-specified times. If a snapshot is desired, it must be created explicitly with <span class="pre">`dict`</span> or the proxy’s <span class="pre">`.copy()`</span> method. (Changed as part of <span id="index-70" class="target"></span><a href="https://peps.python.org/pep-0667/" class="pep reference external"><strong>PEP 667</strong></a>.)

- <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">functools.partial</code></span></a> now emits a <a href="../library/exceptions.html#FutureWarning" class="reference internal" title="FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a> when used as a method. The behavior will change in future Python versions. Wrap it in <a href="../library/functions.html#staticmethod" class="reference internal" title="staticmethod"><span class="pre"><code class="sourceCode python"><span class="bu">staticmethod</span>()</code></span></a> if you want to preserve the old behavior. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/121027" class="reference external">gh-121027</a>.)

- An <a href="../library/exceptions.html#OSError" class="reference internal" title="OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a> is now raised by <a href="../library/getpass.html#getpass.getuser" class="reference internal" title="getpass.getuser"><span class="pre"><code class="sourceCode python">getpass.getuser()</code></span></a> for any failure to retrieve a username, instead of <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> on non-Unix platforms or <a href="../library/exceptions.html#KeyError" class="reference internal" title="KeyError"><span class="pre"><code class="sourceCode python"><span class="pp">KeyError</span></code></span></a> on Unix platforms where the password database is empty.

- The value of the <span class="pre">`mode`</span> attribute of <a href="../library/gzip.html#gzip.GzipFile" class="reference internal" title="gzip.GzipFile"><span class="pre"><code class="sourceCode python">gzip.GzipFile</code></span></a> is now a string (<span class="pre">`'rb'`</span> or <span class="pre">`'wb'`</span>) instead of an integer (<span class="pre">`1`</span> or <span class="pre">`2`</span>). The value of the <span class="pre">`mode`</span> attribute of the readable file-like object returned by <a href="../library/zipfile.html#zipfile.ZipFile.open" class="reference internal" title="zipfile.ZipFile.open"><span class="pre"><code class="sourceCode python">zipfile.ZipFile.<span class="bu">open</span>()</code></span></a> is now <span class="pre">`'rb'`</span> instead of <span class="pre">`'r'`</span>. (Contributed by Serhiy Storchaka in <a href="https://github.com/python/cpython/issues/115961" class="reference external">gh-115961</a>.)

- <a href="../library/mailbox.html#mailbox.Maildir" class="reference internal" title="mailbox.Maildir"><span class="pre"><code class="sourceCode python">mailbox.Maildir</code></span></a> now ignores files with a leading dot (<span class="pre">`.`</span>). (Contributed by Zackery Spytz in <a href="https://github.com/python/cpython/issues/65559" class="reference external">gh-65559</a>.)

- <a href="../library/pathlib.html#pathlib.Path.glob" class="reference internal" title="pathlib.Path.glob"><span class="pre"><code class="sourceCode python">pathlib.Path.glob()</code></span></a> and <a href="../library/pathlib.html#pathlib.Path.rglob" class="reference internal" title="pathlib.Path.rglob"><span class="pre"><code class="sourceCode python">rglob()</code></span></a> now return both files and directories if a pattern that ends with “<span class="pre">`**`</span>” is given, rather than directories only. Add a trailing slash to keep the previous behavior and only match directories.

- The <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a> module now expects the <span class="pre">`_thread`</span> module to have an <span class="pre">`_is_main_interpreter()`</span> function. This function takes no arguments and returns <span class="pre">`True`</span> if the current interpreter is the main interpreter.

  Any library or application that provides a custom <span class="pre">`_thread`</span> module must provide <span class="pre">`_is_main_interpreter()`</span>, just like the module’s other “private” attributes. (<a href="https://github.com/python/cpython/issues/112826" class="reference external">gh-112826</a>.)

</div>

<div id="changes-in-the-c-api" class="section">

### Changes in the C API<a href="#changes-in-the-c-api" class="headerlink" title="Link to this heading">¶</a>

- <span class="pre">`Python.h`</span> no longer includes the <span class="pre">`<ieeefp.h>`</span> standard header. It was included for the <span class="pre">`finite()`</span> function which is now provided by the <span class="pre">`<math.h>`</span> header. It should now be included explicitly if needed. Remove also the <span class="pre">`HAVE_IEEEFP_H`</span> macro. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108765" class="reference external">gh-108765</a>.)

- <span class="pre">`Python.h`</span> no longer includes these standard header files: <span class="pre">`<time.h>`</span>, <span class="pre">`<sys/select.h>`</span> and <span class="pre">`<sys/time.h>`</span>. If needed, they should now be included explicitly. For example, <span class="pre">`<time.h>`</span> provides the <span class="pre">`clock()`</span> and <span class="pre">`gmtime()`</span> functions, <span class="pre">`<sys/select.h>`</span> provides the <span class="pre">`select()`</span> function, and <span class="pre">`<sys/time.h>`</span> provides the <span class="pre">`futimes()`</span>, <span class="pre">`gettimeofday()`</span> and <span class="pre">`setitimer()`</span> functions. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108765" class="reference external">gh-108765</a>.)

- On Windows, <span class="pre">`Python.h`</span> no longer includes the <span class="pre">`<stddef.h>`</span> standard header file. If needed, it should now be included explicitly. For example, it provides <span class="pre">`offsetof()`</span> function, and <span class="pre">`size_t`</span> and <span class="pre">`ptrdiff_t`</span> types. Including <span class="pre">`<stddef.h>`</span> explicitly was already needed by all other platforms, the <span class="pre">`HAVE_STDDEF_H`</span> macro is only defined on Windows. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/108765" class="reference external">gh-108765</a>.)

- If the <a href="../c-api/stable.html#c.Py_LIMITED_API" class="reference internal" title="Py_LIMITED_API"><span class="pre"><code class="sourceCode c">Py_LIMITED_API</code></span></a> macro is defined, <span class="pre">`Py_BUILD_CORE`</span>, <span class="pre">`Py_BUILD_CORE_BUILTIN`</span> and <span class="pre">`Py_BUILD_CORE_MODULE`</span> macros are now undefined by <span class="pre">`<Python.h>`</span>. (Contributed by Victor Stinner in <a href="https://github.com/python/cpython/issues/85283" class="reference external">gh-85283</a>.)

- The old trashcan macros <span class="pre">`Py_TRASHCAN_SAFE_BEGIN`</span> and <span class="pre">`Py_TRASHCAN_SAFE_END`</span> were removed. They should be replaced by the new macros <span class="pre">`Py_TRASHCAN_BEGIN`</span> and <span class="pre">`Py_TRASHCAN_END`</span>.

  A <span class="pre">`tp_dealloc`</span> function that has the old macros, such as:

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

  Note that <span class="pre">`Py_TRASHCAN_BEGIN`</span> has a second argument which should be the deallocation function it is in. The new macros were added in Python 3.8 and the old macros were deprecated in Python 3.11. (Contributed by Irit Katriel in <a href="https://github.com/python/cpython/issues/105111" class="reference external">gh-105111</a>.)

<!-- -->

- <a href="#whatsnew313-locals-semantics" class="reference internal"><span class="std std-ref">PEP 667</span></a> introduces several changes to frame-related functions:

  - The effects of mutating the dictionary returned from <a href="../c-api/reflection.html#c.PyEval_GetLocals" class="reference internal" title="PyEval_GetLocals"><span class="pre"><code class="sourceCode c">PyEval_GetLocals<span class="op">()</span></code></span></a> in an <a href="../glossary.html#term-optimized-scope" class="reference internal"><span class="xref std std-term">optimized scope</span></a> have changed. New dict entries added this way will now *only* be visible to subsequent <a href="../c-api/reflection.html#c.PyEval_GetLocals" class="reference internal" title="PyEval_GetLocals"><span class="pre"><code class="sourceCode c">PyEval_GetLocals<span class="op">()</span></code></span></a> calls in that frame, as <a href="../c-api/frame.html#c.PyFrame_GetLocals" class="reference internal" title="PyFrame_GetLocals"><span class="pre"><code class="sourceCode c">PyFrame_GetLocals<span class="op">()</span></code></span></a>, <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a>, and <a href="../reference/datamodel.html#frame.f_locals" class="reference internal" title="frame.f_locals"><span class="pre"><code class="sourceCode python">FrameType.f_locals</code></span></a> no longer access the same underlying cached dictionary. Changes made to entries for actual variable names and names added via the write-through proxy interfaces will be overwritten on subsequent calls to <a href="../c-api/reflection.html#c.PyEval_GetLocals" class="reference internal" title="PyEval_GetLocals"><span class="pre"><code class="sourceCode c">PyEval_GetLocals<span class="op">()</span></code></span></a> in that frame. The recommended code update depends on how the function was being used, so refer to the deprecation notice on the function for details.

  - Calling <a href="../c-api/frame.html#c.PyFrame_GetLocals" class="reference internal" title="PyFrame_GetLocals"><span class="pre"><code class="sourceCode c">PyFrame_GetLocals<span class="op">()</span></code></span></a> in an <a href="../glossary.html#term-optimized-scope" class="reference internal"><span class="xref std std-term">optimized scope</span></a> now returns a write-through proxy rather than a snapshot that gets updated at ill-specified times. If a snapshot is desired, it must be created explicitly (e.g. with <a href="../c-api/dict.html#c.PyDict_Copy" class="reference internal" title="PyDict_Copy"><span class="pre"><code class="sourceCode c">PyDict_Copy<span class="op">()</span></code></span></a>), or by calling the new <a href="../c-api/reflection.html#c.PyEval_GetFrameLocals" class="reference internal" title="PyEval_GetFrameLocals"><span class="pre"><code class="sourceCode c">PyEval_GetFrameLocals<span class="op">()</span></code></span></a> API.

  - <span class="pre">`PyFrame_FastToLocals()`</span> and <span class="pre">`PyFrame_FastToLocalsWithError()`</span> no longer have any effect. Calling these functions has been redundant since Python 3.11, when <a href="../c-api/frame.html#c.PyFrame_GetLocals" class="reference internal" title="PyFrame_GetLocals"><span class="pre"><code class="sourceCode c">PyFrame_GetLocals<span class="op">()</span></code></span></a> was first introduced.

  - <span class="pre">`PyFrame_LocalsToFast()`</span> no longer has any effect. Calling this function is redundant now that <a href="../c-api/frame.html#c.PyFrame_GetLocals" class="reference internal" title="PyFrame_GetLocals"><span class="pre"><code class="sourceCode c">PyFrame_GetLocals<span class="op">()</span></code></span></a> returns a write-through proxy for <a href="../glossary.html#term-optimized-scope" class="reference internal"><span class="xref std std-term">optimized scopes</span></a>.

- Python 3.13 removed many private functions. Some of them can be replaced using these alternatives:

  - <span class="pre">`_PyDict_Pop()`</span>: <a href="../c-api/dict.html#c.PyDict_Pop" class="reference internal" title="PyDict_Pop"><span class="pre"><code class="sourceCode c">PyDict_Pop<span class="op">()</span></code></span></a> or <a href="../c-api/dict.html#c.PyDict_PopString" class="reference internal" title="PyDict_PopString"><span class="pre"><code class="sourceCode c">PyDict_PopString<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyDict_GetItemWithError()`</span>: <a href="../c-api/dict.html#c.PyDict_GetItemRef" class="reference internal" title="PyDict_GetItemRef"><span class="pre"><code class="sourceCode c">PyDict_GetItemRef<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyErr_WriteUnraisableMsg()`</span>: <a href="../c-api/exceptions.html#c.PyErr_FormatUnraisable" class="reference internal" title="PyErr_FormatUnraisable"><span class="pre"><code class="sourceCode c">PyErr_FormatUnraisable<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyEval_SetTrace()`</span>: <a href="../c-api/init.html#c.PyEval_SetTrace" class="reference internal" title="PyEval_SetTrace"><span class="pre"><code class="sourceCode c">PyEval_SetTrace<span class="op">()</span></code></span></a> or <a href="../c-api/init.html#c.PyEval_SetTraceAllThreads" class="reference internal" title="PyEval_SetTraceAllThreads"><span class="pre"><code class="sourceCode c">PyEval_SetTraceAllThreads<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyList_Extend()`</span>: <a href="../c-api/list.html#c.PyList_Extend" class="reference internal" title="PyList_Extend"><span class="pre"><code class="sourceCode c">PyList_Extend<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyLong_AsInt()`</span>: <a href="../c-api/long.html#c.PyLong_AsInt" class="reference internal" title="PyLong_AsInt"><span class="pre"><code class="sourceCode c">PyLong_AsInt<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyMem_RawStrdup()`</span>: <span class="pre">`strdup()`</span>;

  - <span class="pre">`_PyMem_Strdup()`</span>: <span class="pre">`strdup()`</span>;

  - <span class="pre">`_PyObject_ClearManagedDict()`</span>: <a href="../c-api/object.html#c.PyObject_ClearManagedDict" class="reference internal" title="PyObject_ClearManagedDict"><span class="pre"><code class="sourceCode c">PyObject_ClearManagedDict<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyObject_VisitManagedDict()`</span>: <a href="../c-api/object.html#c.PyObject_VisitManagedDict" class="reference internal" title="PyObject_VisitManagedDict"><span class="pre"><code class="sourceCode c">PyObject_VisitManagedDict<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyThreadState_UncheckedGet()`</span>: <a href="../c-api/init.html#c.PyThreadState_GetUnchecked" class="reference internal" title="PyThreadState_GetUnchecked"><span class="pre"><code class="sourceCode c">PyThreadState_GetUnchecked<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyTime_AsSecondsDouble()`</span>: <a href="../c-api/time.html#c.PyTime_AsSecondsDouble" class="reference internal" title="PyTime_AsSecondsDouble"><span class="pre"><code class="sourceCode c">PyTime_AsSecondsDouble<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyTime_GetMonotonicClock()`</span>: <a href="../c-api/time.html#c.PyTime_Monotonic" class="reference internal" title="PyTime_Monotonic"><span class="pre"><code class="sourceCode c">PyTime_Monotonic<span class="op">()</span></code></span></a> or <a href="../c-api/time.html#c.PyTime_MonotonicRaw" class="reference internal" title="PyTime_MonotonicRaw"><span class="pre"><code class="sourceCode c">PyTime_MonotonicRaw<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyTime_GetPerfCounter()`</span>: <a href="../c-api/time.html#c.PyTime_PerfCounter" class="reference internal" title="PyTime_PerfCounter"><span class="pre"><code class="sourceCode c">PyTime_PerfCounter<span class="op">()</span></code></span></a> or <a href="../c-api/time.html#c.PyTime_PerfCounterRaw" class="reference internal" title="PyTime_PerfCounterRaw"><span class="pre"><code class="sourceCode c">PyTime_PerfCounterRaw<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyTime_GetSystemClock()`</span>: <a href="../c-api/time.html#c.PyTime_Time" class="reference internal" title="PyTime_Time"><span class="pre"><code class="sourceCode c">PyTime_Time<span class="op">()</span></code></span></a> or <a href="../c-api/time.html#c.PyTime_TimeRaw" class="reference internal" title="PyTime_TimeRaw"><span class="pre"><code class="sourceCode c">PyTime_TimeRaw<span class="op">()</span></code></span></a>;

  - <span class="pre">`_PyTime_MAX`</span>: <a href="../c-api/time.html#c.PyTime_MAX" class="reference internal" title="PyTime_MAX"><span class="pre"><code class="sourceCode c">PyTime_MAX</code></span></a>;

  - <span class="pre">`_PyTime_MIN`</span>: <a href="../c-api/time.html#c.PyTime_MIN" class="reference internal" title="PyTime_MIN"><span class="pre"><code class="sourceCode c">PyTime_MIN</code></span></a>;

  - <span class="pre">`_PyTime_t`</span>: <a href="../c-api/time.html#c.PyTime_t" class="reference internal" title="PyTime_t"><span class="pre"><code class="sourceCode c">PyTime_t</code></span></a>;

  - <span class="pre">`_Py_HashPointer()`</span>: <a href="../c-api/hash.html#c.Py_HashPointer" class="reference internal" title="Py_HashPointer"><span class="pre"><code class="sourceCode c">Py_HashPointer<span class="op">()</span></code></span></a>;

  - <span class="pre">`_Py_IsFinalizing()`</span>: <a href="../c-api/init.html#c.Py_IsFinalizing" class="reference internal" title="Py_IsFinalizing"><span class="pre"><code class="sourceCode c">Py_IsFinalizing<span class="op">()</span></code></span></a>.

  The <a href="https://github.com/python/pythoncapi-compat/" class="reference external">pythoncapi-compat project</a> can be used to get most of these new functions on Python 3.12 and older.

</div>

</div>

<div id="regression-test-changes" class="section">

## Regression Test Changes<a href="#regression-test-changes" class="headerlink" title="Link to this heading">¶</a>

- Python built with <span class="pre">`configure`</span> <a href="../using/configure.html#cmdoption-with-pydebug" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--with-pydebug</code></span></a> now supports a <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">presite=package.module</code></span></a> command-line option. If used, it specifies a module that should be imported early in the lifecycle of the interpreter, before <span class="pre">`site.py`</span> is executed. (Contributed by Łukasz Langa in <a href="https://github.com/python/cpython/issues/110769" class="reference external">gh-110769</a>.)

</div>

<div id="notable-changes-in-3-13-1" class="section">

## Notable changes in 3.13.1<a href="#notable-changes-in-3-13-1" class="headerlink" title="Link to this heading">¶</a>

<div id="id12" class="section">

### sys<a href="#id12" class="headerlink" title="Link to this heading">¶</a>

- The previously undocumented special function <a href="../library/sys.html#sys.getobjects" class="reference internal" title="sys.getobjects"><span class="pre"><code class="sourceCode python">sys.getobjects()</code></span></a>, which only exists in specialized builds of Python, may now return objects from other interpreters than the one it’s called in.

</div>

</div>

<div id="notable-changes-in-3-13-4" class="section">

## Notable changes in 3.13.4<a href="#notable-changes-in-3-13-4" class="headerlink" title="Link to this heading">¶</a>

<div id="id13" class="section">

### os.path<a href="#id13" class="headerlink" title="Link to this heading">¶</a>

- The *strict* parameter to <a href="../library/os.path.html#os.path.realpath" class="reference internal" title="os.path.realpath"><span class="pre"><code class="sourceCode python">os.path.realpath()</code></span></a> accepts a new value, <a href="../library/os.path.html#os.path.ALLOW_MISSING" class="reference internal" title="os.path.ALLOW_MISSING"><span class="pre"><code class="sourceCode python">os.path.ALLOW_MISSING</code></span></a>. If used, errors other than <a href="../library/exceptions.html#FileNotFoundError" class="reference internal" title="FileNotFoundError"><span class="pre"><code class="sourceCode python"><span class="pp">FileNotFoundError</span></code></span></a> will be re-raised; the resulting path can be missing but it will be free of symlinks. (Contributed by Petr Viktorin for <span id="index-71" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2025-4517" class="cve reference external"><strong>CVE 2025-4517</strong></a>.)

</div>

<div id="tarfile" class="section">

### tarfile<a href="#tarfile" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/tarfile.html#tarfile.data_filter" class="reference internal" title="tarfile.data_filter"><span class="pre"><code class="sourceCode python">data_filter()</code></span></a> now normalizes symbolic link targets in order to avoid path traversal attacks. (Contributed by Petr Viktorin in <a href="https://github.com/python/cpython/issues/127987" class="reference external">gh-127987</a> and <span id="index-72" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2025-4138" class="cve reference external"><strong>CVE 2025-4138</strong></a>.)

- <a href="../library/tarfile.html#tarfile.TarFile.extractall" class="reference internal" title="tarfile.TarFile.extractall"><span class="pre"><code class="sourceCode python">extractall()</code></span></a> now skips fixing up directory attributes when a directory was removed or replaced by another kind of file. (Contributed by Petr Viktorin in <a href="https://github.com/python/cpython/issues/127987" class="reference external">gh-127987</a> and <span id="index-73" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2024-12718" class="cve reference external"><strong>CVE 2024-12718</strong></a>.)

- <a href="../library/tarfile.html#tarfile.TarFile.extract" class="reference internal" title="tarfile.TarFile.extract"><span class="pre"><code class="sourceCode python">extract()</code></span></a> and <a href="../library/tarfile.html#tarfile.TarFile.extractall" class="reference internal" title="tarfile.TarFile.extractall"><span class="pre"><code class="sourceCode python">extractall()</code></span></a> now (re-)apply the extraction filter when substituting a link (hard or symbolic) with a copy of another archive member, and when fixing up directory attributes. The former raises a new exception, <a href="../library/tarfile.html#tarfile.LinkFallbackError" class="reference internal" title="tarfile.LinkFallbackError"><span class="pre"><code class="sourceCode python">LinkFallbackError</code></span></a>. (Contributed by Petr Viktorin for <span id="index-74" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2025-4330" class="cve reference external"><strong>CVE 2025-4330</strong></a> and <span id="index-75" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2024-12718" class="cve reference external"><strong>CVE 2024-12718</strong></a>.)

- <a href="../library/tarfile.html#tarfile.TarFile.extract" class="reference internal" title="tarfile.TarFile.extract"><span class="pre"><code class="sourceCode python">extract()</code></span></a> and <a href="../library/tarfile.html#tarfile.TarFile.extractall" class="reference internal" title="tarfile.TarFile.extractall"><span class="pre"><code class="sourceCode python">extractall()</code></span></a> no longer extract rejected members when <a href="../library/tarfile.html#tarfile.TarFile.errorlevel" class="reference internal" title="tarfile.TarFile.errorlevel"><span class="pre"><code class="sourceCode python">errorlevel()</code></span></a> is zero. (Contributed by Matt Prodani and Petr Viktorin in <a href="https://github.com/python/cpython/issues/112887" class="reference external">gh-112887</a> and <span id="index-76" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2025-4435" class="cve reference external"><strong>CVE 2025-4435</strong></a>.)

</div>

</div>

</div>

<div class="clearer">

</div>

</div>
