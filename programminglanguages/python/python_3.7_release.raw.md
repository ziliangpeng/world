<div class="body" role="main">

<div id="what-s-new-in-python-3-7" class="section">

# What’s New In Python 3.7<a href="#what-s-new-in-python-3-7" class="headerlink" title="Link to this heading">¶</a>

Editor<span class="colon">:</span>  
Elvis Pranskevichus \<<a href="mailto:elvis%40magic.io" class="reference external">elvis<span>@</span>magic<span>.</span>io</a>\>

This article explains the new features in Python 3.7, compared to 3.6. Python 3.7 was released on June 27, 2018. For full details, see the <a href="changelog.html#changelog" class="reference internal"><span class="std std-ref">changelog</span></a>.

<div id="summary-release-highlights" class="section">

## Summary – Release Highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

New syntax features:

- <a href="#whatsnew37-pep563" class="reference internal"><span class="std std-ref">PEP 563</span></a>, postponed evaluation of type annotations.

Backwards incompatible syntax changes:

- <a href="../reference/compound_stmts.html#async" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span></a> and <a href="../reference/expressions.html#await" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">await</code></span></a> are now reserved keywords.

New library modules:

- <a href="../library/contextvars.html#module-contextvars" class="reference internal" title="contextvars: Context Variables"><span class="pre"><code class="sourceCode python">contextvars</code></span></a>: <a href="#whatsnew37-pep567" class="reference internal"><span class="std std-ref">PEP 567 – Context Variables</span></a>

- <a href="../library/dataclasses.html#module-dataclasses" class="reference internal" title="dataclasses: Generate special methods on user-defined classes."><span class="pre"><code class="sourceCode python">dataclasses</code></span></a>: <a href="#whatsnew37-pep557" class="reference internal"><span class="std std-ref">PEP 557 – Data Classes</span></a>

- <a href="#whatsnew37-importlib-resources" class="reference internal"><span class="std std-ref">importlib.resources</span></a>

New built-in features:

- <a href="#whatsnew37-pep553" class="reference internal"><span class="std std-ref">PEP 553</span></a>, the new <a href="../library/functions.html#breakpoint" class="reference internal" title="breakpoint"><span class="pre"><code class="sourceCode python"><span class="bu">breakpoint</span>()</code></span></a> function.

Python data model improvements:

- <a href="#whatsnew37-pep562" class="reference internal"><span class="std std-ref">PEP 562</span></a>, customization of access to module attributes.

- <a href="#whatsnew37-pep560" class="reference internal"><span class="std std-ref">PEP 560</span></a>, core support for typing module and generic types.

- the insertion-order preservation nature of <a href="../library/stdtypes.html#typesmapping" class="reference internal"><span class="std std-ref">dict</span></a> objects <a href="https://mail.python.org/pipermail/python-dev/2017-December/151283.html" class="reference external">has been declared</a> to be an official part of the Python language spec.

Significant improvements in the standard library:

- The <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> module has received new features, significant <a href="#whatsnew37-asyncio" class="reference internal"><span class="std std-ref">usability and performance improvements</span></a>.

- The <a href="../library/time.html#module-time" class="reference internal" title="time: Time access and conversions."><span class="pre"><code class="sourceCode python">time</code></span></a> module gained support for <a href="#whatsnew37-pep564" class="reference internal"><span class="std std-ref">functions with nanosecond resolution</span></a>.

CPython implementation improvements:

- Avoiding the use of ASCII as a default text encoding:

  - <a href="#whatsnew37-pep538" class="reference internal"><span class="std std-ref">PEP 538</span></a>, legacy C locale coercion

  - <a href="#whatsnew37-pep540" class="reference internal"><span class="std std-ref">PEP 540</span></a>, forced UTF-8 runtime mode

- <a href="#whatsnew37-pep552" class="reference internal"><span class="std std-ref">PEP 552</span></a>, deterministic .pycs

- <a href="#whatsnew37-devmode" class="reference internal"><span class="std std-ref">New Python Development Mode</span></a>

- <a href="#whatsnew37-pep565" class="reference internal"><span class="std std-ref">PEP 565</span></a>, improved <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> handling

C API improvements:

- <a href="#whatsnew37-pep539" class="reference internal"><span class="std std-ref">PEP 539</span></a>, new C API for thread-local storage

Documentation improvements:

- <a href="#whatsnew37-pep545" class="reference internal"><span class="std std-ref">PEP 545</span></a>, Python documentation translations

- New documentation translations: <a href="https://docs.python.org/ja/" class="reference external">Japanese</a>, <a href="https://docs.python.org/fr/" class="reference external">French</a>, and <a href="https://docs.python.org/ko/" class="reference external">Korean</a>.

This release features notable performance improvements in many areas. The <a href="#whatsnew37-perf" class="reference internal"><span class="std std-ref">Optimizations</span></a> section lists them in detail.

For a list of changes that may affect compatibility with previous Python releases please refer to the <a href="#porting-to-python-37" class="reference internal"><span class="std std-ref">Porting to Python 3.7</span></a> section.

</div>

<div id="new-features" class="section">

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-563-postponed-evaluation-of-annotations" class="section">

<span id="whatsnew37-pep563"></span>

### PEP 563: Postponed Evaluation of Annotations<a href="#pep-563-postponed-evaluation-of-annotations" class="headerlink" title="Link to this heading">¶</a>

The advent of type hints in Python uncovered two glaring usability issues with the functionality of annotations added in <span id="index-0" class="target"></span><a href="https://peps.python.org/pep-3107/" class="pep reference external"><strong>PEP 3107</strong></a> and refined further in <span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0526/" class="pep reference external"><strong>PEP 526</strong></a>:

- annotations could only use names which were already available in the current scope, in other words they didn’t support forward references of any kind; and

- annotating source code had adverse effects on startup time of Python programs.

Both of these issues are fixed by postponing the evaluation of annotations. Instead of compiling code which executes expressions in annotations at their definition time, the compiler stores the annotation in a string form equivalent to the AST of the expression in question. If needed, annotations can be resolved at runtime using <a href="../library/typing.html#typing.get_type_hints" class="reference internal" title="typing.get_type_hints"><span class="pre"><code class="sourceCode python">typing.get_type_hints()</code></span></a>. In the common case where this is not required, the annotations are cheaper to store (since short strings are interned by the interpreter) and make startup time faster.

Usability-wise, annotations now support forward references, making the following syntax valid:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class C:
        @classmethod
        def from_string(cls, source: str) -> C:
            ...

        def validate_b(self, obj: B) -> bool:
            ...

    class B:
        ...

</div>

</div>

Since this change breaks compatibility, the new behavior needs to be enabled on a per-module basis in Python 3.7 using a <a href="../library/__future__.html#module-__future__" class="reference internal" title="__future__: Future statement definitions"><span class="pre"><code class="sourceCode python">__future__</code></span></a> import:

<div class="highlight-python3 notranslate">

<div class="highlight">

    from __future__ import annotations

</div>

</div>

It will become the default in Python 3.10.

<div class="admonition seealso">

See also

<span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0563/" class="pep reference external"><strong>PEP 563</strong></a> – Postponed evaluation of annotations  
PEP written and implemented by Łukasz Langa.

</div>

</div>

<div id="pep-538-legacy-c-locale-coercion" class="section">

<span id="whatsnew37-pep538"></span>

### PEP 538: Legacy C Locale Coercion<a href="#pep-538-legacy-c-locale-coercion" class="headerlink" title="Link to this heading">¶</a>

An ongoing challenge within the Python 3 series has been determining a sensible default strategy for handling the “7-bit ASCII” text encoding assumption currently implied by the use of the default C or POSIX locale on non-Windows platforms.

<span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0538/" class="pep reference external"><strong>PEP 538</strong></a> updates the default interpreter command line interface to automatically coerce that locale to an available UTF-8 based locale as described in the documentation of the new <span id="index-4" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONCOERCECLOCALE" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONCOERCECLOCALE</code></span></a> environment variable. Automatically setting <span class="pre">`LC_CTYPE`</span> this way means that both the core interpreter and locale-aware C extensions (such as <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline</code></span></a>) will assume the use of UTF-8 as the default text encoding, rather than ASCII.

The platform support definition in <span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a> has also been updated to limit full text handling support to suitably configured non-ASCII based locales.

As part of this change, the default error handler for <a href="../library/sys.html#sys.stdin" class="reference internal" title="sys.stdin"><span class="pre"><code class="sourceCode python">stdin</code></span></a> and <a href="../library/sys.html#sys.stdout" class="reference internal" title="sys.stdout"><span class="pre"><code class="sourceCode python">stdout</code></span></a> is now <span class="pre">`surrogateescape`</span> (rather than <span class="pre">`strict`</span>) when using any of the defined coercion target locales (currently <span class="pre">`C.UTF-8`</span>, <span class="pre">`C.utf8`</span>, and <span class="pre">`UTF-8`</span>). The default error handler for <a href="../library/sys.html#sys.stderr" class="reference internal" title="sys.stderr"><span class="pre"><code class="sourceCode python">stderr</code></span></a> continues to be <span class="pre">`backslashreplace`</span>, regardless of locale.

Locale coercion is silent by default, but to assist in debugging potentially locale related integration problems, explicit warnings (emitted directly on <a href="../library/sys.html#sys.stderr" class="reference internal" title="sys.stderr"><span class="pre"><code class="sourceCode python">stderr</code></span></a>) can be requested by setting <span class="pre">`PYTHONCOERCECLOCALE=warn`</span>. This setting will also cause the Python runtime to emit a warning if the legacy C locale remains active when the core interpreter is initialized.

While <span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0538/" class="pep reference external"><strong>PEP 538</strong></a>’s locale coercion has the benefit of also affecting extension modules (such as GNU <span class="pre">`readline`</span>), as well as child processes (including those running non-Python applications and older versions of Python), it has the downside of requiring that a suitable target locale be present on the running system. To better handle the case where no suitable target locale is available (as occurs on RHEL/CentOS 7, for example), Python 3.7 also implements <a href="#whatsnew37-pep540" class="reference internal"><span class="std std-ref">PEP 540: Forced UTF-8 Runtime Mode</span></a>.

<div class="admonition seealso">

See also

<span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0538/" class="pep reference external"><strong>PEP 538</strong></a> – Coercing the legacy C locale to a UTF-8 based locale  
PEP written and implemented by Nick Coghlan.

</div>

</div>

<div id="pep-540-forced-utf-8-runtime-mode" class="section">

<span id="whatsnew37-pep540"></span>

### PEP 540: Forced UTF-8 Runtime Mode<a href="#pep-540-forced-utf-8-runtime-mode" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span></a> <span class="pre">`utf8`</span> command line option and <span id="index-8" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONUTF8" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONUTF8</code></span></a> environment variable can be used to enable the <a href="../library/os.html#utf8-mode" class="reference internal"><span class="std std-ref">Python UTF-8 Mode</span></a>.

When in UTF-8 mode, CPython ignores the locale settings, and uses the UTF-8 encoding by default. The error handlers for <a href="../library/sys.html#sys.stdin" class="reference internal" title="sys.stdin"><span class="pre"><code class="sourceCode python">sys.stdin</code></span></a> and <a href="../library/sys.html#sys.stdout" class="reference internal" title="sys.stdout"><span class="pre"><code class="sourceCode python">sys.stdout</code></span></a> streams are set to <span class="pre">`surrogateescape`</span>.

The forced UTF-8 mode can be used to change the text handling behavior in an embedded Python interpreter without changing the locale settings of an embedding application.

While <span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0540/" class="pep reference external"><strong>PEP 540</strong></a>’s UTF-8 mode has the benefit of working regardless of which locales are available on the running system, it has the downside of having no effect on extension modules (such as GNU <span class="pre">`readline`</span>), child processes running non-Python applications, and child processes running older versions of Python. To reduce the risk of corrupting text data when communicating with such components, Python 3.7 also implements <a href="#whatsnew37-pep540" class="reference internal"><span class="std std-ref">PEP 540: Forced UTF-8 Runtime Mode</span></a>).

The UTF-8 mode is enabled by default when the locale is <span class="pre">`C`</span> or <span class="pre">`POSIX`</span>, and the <span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0538/" class="pep reference external"><strong>PEP 538</strong></a> locale coercion feature fails to change it to a UTF-8 based alternative (whether that failure is due to <span class="pre">`PYTHONCOERCECLOCALE=0`</span> being set, <span class="pre">`LC_ALL`</span> being set, or the lack of a suitable target locale).

<div class="admonition seealso">

See also

<span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0540/" class="pep reference external"><strong>PEP 540</strong></a> – Add a new UTF-8 mode  
PEP written and implemented by Victor Stinner

</div>

</div>

<div id="pep-553-built-in-breakpoint" class="section">

<span id="whatsnew37-pep553"></span>

### PEP 553: Built-in <span class="pre">`breakpoint()`</span><a href="#pep-553-built-in-breakpoint" class="headerlink" title="Link to this heading">¶</a>

Python 3.7 includes the new built-in <a href="../library/functions.html#breakpoint" class="reference internal" title="breakpoint"><span class="pre"><code class="sourceCode python"><span class="bu">breakpoint</span>()</code></span></a> function as an easy and consistent way to enter the Python debugger.

Built-in <span class="pre">`breakpoint()`</span> calls <a href="../library/sys.html#sys.breakpointhook" class="reference internal" title="sys.breakpointhook"><span class="pre"><code class="sourceCode python">sys.breakpointhook()</code></span></a>. By default, the latter imports <a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a> and then calls <span class="pre">`pdb.set_trace()`</span>, but by binding <span class="pre">`sys.breakpointhook()`</span> to the function of your choosing, <span class="pre">`breakpoint()`</span> can enter any debugger. Additionally, the environment variable <span id="index-12" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONBREAKPOINT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONBREAKPOINT</code></span></a> can be set to the callable of your debugger of choice. Set <span class="pre">`PYTHONBREAKPOINT=0`</span> to completely disable built-in <span class="pre">`breakpoint()`</span>.

<div class="admonition seealso">

See also

<span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0553/" class="pep reference external"><strong>PEP 553</strong></a> – Built-in breakpoint()  
PEP written and implemented by Barry Warsaw

</div>

</div>

<div id="pep-539-new-c-api-for-thread-local-storage" class="section">

<span id="whatsnew37-pep539"></span>

### PEP 539: New C API for Thread-Local Storage<a href="#pep-539-new-c-api-for-thread-local-storage" class="headerlink" title="Link to this heading">¶</a>

While Python provides a C API for thread-local storage support; the existing <a href="../c-api/init.html#thread-local-storage-api" class="reference internal"><span class="std std-ref">Thread Local Storage (TLS) API</span></a> has used <span class="c-expr sig sig-inline c"><span class="kt">int</span></span> to represent TLS keys across all platforms. This has not generally been a problem for officially support platforms, but that is neither POSIX-compliant, nor portable in any practical sense.

<span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0539/" class="pep reference external"><strong>PEP 539</strong></a> changes this by providing a new <a href="../c-api/init.html#thread-specific-storage-api" class="reference internal"><span class="std std-ref">Thread Specific Storage (TSS) API</span></a> to CPython which supersedes use of the existing TLS API within the CPython interpreter, while deprecating the existing API. The TSS API uses a new type <a href="../c-api/init.html#c.Py_tss_t" class="reference internal" title="Py_tss_t"><span class="pre"><code class="sourceCode c">Py_tss_t</code></span></a> instead of <span class="c-expr sig sig-inline c"><span class="kt">int</span></span> to represent TSS keys–an opaque type the definition of which may depend on the underlying TLS implementation. Therefore, this will allow to build CPython on platforms where the native TLS key is defined in a way that cannot be safely cast to <span class="c-expr sig sig-inline c"><span class="kt">int</span></span>.

Note that on platforms where the native TLS key is defined in a way that cannot be safely cast to <span class="c-expr sig sig-inline c"><span class="kt">int</span></span>, all functions of the existing TLS API will be no-op and immediately return failure. This indicates clearly that the old API is not supported on platforms where it cannot be used reliably, and that no effort will be made to add such support.

<div class="admonition seealso">

See also

<span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0539/" class="pep reference external"><strong>PEP 539</strong></a> – A New C-API for Thread-Local Storage in CPython  
PEP written by Erik M. Bray; implementation by Masayuki Yamamoto.

</div>

</div>

<div id="pep-562-customization-of-access-to-module-attributes" class="section">

<span id="whatsnew37-pep562"></span>

### PEP 562: Customization of Access to Module Attributes<a href="#pep-562-customization-of-access-to-module-attributes" class="headerlink" title="Link to this heading">¶</a>

Python 3.7 allows defining <a href="../reference/datamodel.html#module.__getattr__" class="reference internal" title="module.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> on modules and will call it whenever a module attribute is otherwise not found. Defining <a href="../reference/datamodel.html#module.__dir__" class="reference internal" title="module.__dir__"><span class="pre"><code class="sourceCode python"><span class="fu">__dir__</span>()</code></span></a> on modules is now also allowed.

A typical example of where this may be useful is module attribute deprecation and lazy loading.

<div class="admonition seealso">

See also

<span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0562/" class="pep reference external"><strong>PEP 562</strong></a> – Module <span class="pre">`__getattr__`</span> and <span class="pre">`__dir__`</span>  
PEP written and implemented by Ivan Levkivskyi

</div>

</div>

<div id="pep-564-new-time-functions-with-nanosecond-resolution" class="section">

<span id="whatsnew37-pep564"></span>

### PEP 564: New Time Functions With Nanosecond Resolution<a href="#pep-564-new-time-functions-with-nanosecond-resolution" class="headerlink" title="Link to this heading">¶</a>

The resolution of clocks in modern systems can exceed the limited precision of a floating-point number returned by the <a href="../library/time.html#time.time" class="reference internal" title="time.time"><span class="pre"><code class="sourceCode python">time.time()</code></span></a> function and its variants. To avoid loss of precision, <span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0564/" class="pep reference external"><strong>PEP 564</strong></a> adds six new “nanosecond” variants of the existing timer functions to the <a href="../library/time.html#module-time" class="reference internal" title="time: Time access and conversions."><span class="pre"><code class="sourceCode python">time</code></span></a> module:

- <a href="../library/time.html#time.clock_gettime_ns" class="reference internal" title="time.clock_gettime_ns"><span class="pre"><code class="sourceCode python">time.clock_gettime_ns()</code></span></a>

- <a href="../library/time.html#time.clock_settime_ns" class="reference internal" title="time.clock_settime_ns"><span class="pre"><code class="sourceCode python">time.clock_settime_ns()</code></span></a>

- <a href="../library/time.html#time.monotonic_ns" class="reference internal" title="time.monotonic_ns"><span class="pre"><code class="sourceCode python">time.monotonic_ns()</code></span></a>

- <a href="../library/time.html#time.perf_counter_ns" class="reference internal" title="time.perf_counter_ns"><span class="pre"><code class="sourceCode python">time.perf_counter_ns()</code></span></a>

- <a href="../library/time.html#time.process_time_ns" class="reference internal" title="time.process_time_ns"><span class="pre"><code class="sourceCode python">time.process_time_ns()</code></span></a>

- <a href="../library/time.html#time.time_ns" class="reference internal" title="time.time_ns"><span class="pre"><code class="sourceCode python">time.time_ns()</code></span></a>

The new functions return the number of nanoseconds as an integer value.

<span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0564/#annex-clocks-resolution-in-python" class="pep reference external"><strong>Measurements</strong></a> show that on Linux and Windows the resolution of <a href="../library/time.html#time.time_ns" class="reference internal" title="time.time_ns"><span class="pre"><code class="sourceCode python">time.time_ns()</code></span></a> is approximately 3 times better than that of <a href="../library/time.html#time.time" class="reference internal" title="time.time"><span class="pre"><code class="sourceCode python">time.time()</code></span></a>.

<div class="admonition seealso">

See also

<span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0564/" class="pep reference external"><strong>PEP 564</strong></a> – Add new time functions with nanosecond resolution  
PEP written and implemented by Victor Stinner

</div>

</div>

<div id="pep-565-show-deprecationwarning-in-main" class="section">

<span id="whatsnew37-pep565"></span>

### PEP 565: Show DeprecationWarning in <span class="pre">`__main__`</span><a href="#pep-565-show-deprecationwarning-in-main" class="headerlink" title="Link to this heading">¶</a>

The default handling of <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> has been changed such that these warnings are once more shown by default, but only when the code triggering them is running directly in the <a href="../library/__main__.html#module-__main__" class="reference internal" title="__main__: The environment where top-level code is run. Covers command-line interfaces, import-time behavior, and ``__name__ == &#39;__main__&#39;``."><span class="pre"><code class="sourceCode python">__main__</code></span></a> module. As a result, developers of single file scripts and those using Python interactively should once again start seeing deprecation warnings for the APIs they use, but deprecation warnings triggered by imported application, library and framework modules will continue to be hidden by default.

As a result of this change, the standard library now allows developers to choose between three different deprecation warning behaviours:

- <a href="../library/exceptions.html#FutureWarning" class="reference internal" title="FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a>: always displayed by default, recommended for warnings intended to be seen by application end users (e.g. for deprecated application configuration settings).

- <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>: displayed by default only in <a href="../library/__main__.html#module-__main__" class="reference internal" title="__main__: The environment where top-level code is run. Covers command-line interfaces, import-time behavior, and ``__name__ == &#39;__main__&#39;``."><span class="pre"><code class="sourceCode python">__main__</code></span></a> and when running tests, recommended for warnings intended to be seen by other Python developers where a version upgrade may result in changed behaviour or an error.

- <a href="../library/exceptions.html#PendingDeprecationWarning" class="reference internal" title="PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a>: displayed by default only when running tests, intended for cases where a future version upgrade will change the warning category to <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> or <a href="../library/exceptions.html#FutureWarning" class="reference internal" title="FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a>.

Previously both <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> and <a href="../library/exceptions.html#PendingDeprecationWarning" class="reference internal" title="PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a> were only visible when running tests, which meant that developers primarily writing single file scripts or using Python interactively could be surprised by breaking changes in the APIs they used.

<div class="admonition seealso">

See also

<span id="index-20" class="target"></span><a href="https://peps.python.org/pep-0565/" class="pep reference external"><strong>PEP 565</strong></a> – Show DeprecationWarning in <span class="pre">`__main__`</span>  
PEP written and implemented by Nick Coghlan

</div>

</div>

<div id="pep-560-core-support-for-typing-module-and-generic-types" class="section">

<span id="whatsnew37-pep560"></span>

### PEP 560: Core Support for <span class="pre">`typing`</span> module and Generic Types<a href="#pep-560-core-support-for-typing-module-and-generic-types" class="headerlink" title="Link to this heading">¶</a>

Initially <span id="index-21" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> was designed in such way that it would not introduce *any* changes to the core CPython interpreter. Now type hints and the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module are extensively used by the community, so this restriction is removed. The PEP introduces two special methods <a href="../reference/datamodel.html#object.__class_getitem__" class="reference internal" title="object.__class_getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__class_getitem__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__mro_entries__" class="reference internal" title="object.__mro_entries__"><span class="pre"><code class="sourceCode python">__mro_entries__()</code></span></a>, these methods are now used by most classes and special constructs in <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a>. As a result, the speed of various operations with types increased up to 7 times, the generic types can be used without metaclass conflicts, and several long standing bugs in <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module are fixed.

<div class="admonition seealso">

See also

<span id="index-22" class="target"></span><a href="https://peps.python.org/pep-0560/" class="pep reference external"><strong>PEP 560</strong></a> – Core support for typing module and generic types  
PEP written and implemented by Ivan Levkivskyi

</div>

</div>

<div id="pep-552-hash-based-pyc-files" class="section">

<span id="whatsnew37-pep552"></span>

### PEP 552: Hash-based .pyc Files<a href="#pep-552-hash-based-pyc-files" class="headerlink" title="Link to this heading">¶</a>

Python has traditionally checked the up-to-dateness of bytecode cache files (i.e., <span class="pre">`.pyc`</span> files) by comparing the source metadata (last-modified timestamp and size) with source metadata saved in the cache file header when it was generated. While effective, this invalidation method has its drawbacks. When filesystem timestamps are too coarse, Python can miss source updates, leading to user confusion. Additionally, having a timestamp in the cache file is problematic for <a href="https://reproducible-builds.org/" class="reference external">build reproducibility</a> and content-based build systems.

<span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0552/" class="pep reference external"><strong>PEP 552</strong></a> extends the pyc format to allow the hash of the source file to be used for invalidation instead of the source timestamp. Such <span class="pre">`.pyc`</span> files are called “hash-based”. By default, Python still uses timestamp-based invalidation and does not generate hash-based <span class="pre">`.pyc`</span> files at runtime. Hash-based <span class="pre">`.pyc`</span> files may be generated with <a href="../library/py_compile.html#module-py_compile" class="reference internal" title="py_compile: Generate byte-code files from Python source files."><span class="pre"><code class="sourceCode python">py_compile</code></span></a> or <a href="../library/compileall.html#module-compileall" class="reference internal" title="compileall: Tools for byte-compiling all Python source files in a directory tree."><span class="pre"><code class="sourceCode python">compileall</code></span></a>.

Hash-based <span class="pre">`.pyc`</span> files come in two variants: checked and unchecked. Python validates checked hash-based <span class="pre">`.pyc`</span> files against the corresponding source files at runtime but doesn’t do so for unchecked hash-based pycs. Unchecked hash-based <span class="pre">`.pyc`</span> files are a useful performance optimization for environments where a system external to Python (e.g., the build system) is responsible for keeping <span class="pre">`.pyc`</span> files up-to-date.

See <a href="../reference/import.html#pyc-invalidation" class="reference internal"><span class="std std-ref">Cached bytecode invalidation</span></a> for more information.

<div class="admonition seealso">

See also

<span id="index-24" class="target"></span><a href="https://peps.python.org/pep-0552/" class="pep reference external"><strong>PEP 552</strong></a> – Deterministic pycs  
PEP written and implemented by Benjamin Peterson

</div>

</div>

<div id="pep-545-python-documentation-translations" class="section">

<span id="whatsnew37-pep545"></span>

### PEP 545: Python Documentation Translations<a href="#pep-545-python-documentation-translations" class="headerlink" title="Link to this heading">¶</a>

<span id="index-25" class="target"></span><a href="https://peps.python.org/pep-0545/" class="pep reference external"><strong>PEP 545</strong></a> describes the process of creating and maintaining Python documentation translations.

Three new translations have been added:

- Japanese: <a href="https://docs.python.org/ja/" class="reference external">https://docs.python.org/ja/</a>

- French: <a href="https://docs.python.org/fr/" class="reference external">https://docs.python.org/fr/</a>

- Korean: <a href="https://docs.python.org/ko/" class="reference external">https://docs.python.org/ko/</a>

<div class="admonition seealso">

See also

<span id="index-26" class="target"></span><a href="https://peps.python.org/pep-0545/" class="pep reference external"><strong>PEP 545</strong></a> – Python Documentation Translations  
PEP written and implemented by Julien Palard, Inada Naoki, and Victor Stinner.

</div>

</div>

<div id="python-development-mode-x-dev" class="section">

<span id="whatsnew37-devmode"></span>

### Python Development Mode (-X dev)<a href="#python-development-mode-x-dev" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span></a> <span class="pre">`dev`</span> command line option or the new <span id="index-27" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONDEVMODE" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONDEVMODE</code></span></a> environment variable can be used to enable <a href="../library/devmode.html#devmode" class="reference internal"><span class="std std-ref">Python Development Mode</span></a>. When in development mode, Python performs additional runtime checks that are too expensive to be enabled by default. See <a href="../library/devmode.html#devmode" class="reference internal"><span class="std std-ref">Python Development Mode</span></a> documentation for the full description.

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

- An <a href="../reference/expressions.html#await" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">await</code></span></a> expression and comprehensions containing an <a href="../reference/compound_stmts.html#async-for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span><code class="xref std std-keyword docutils literal notranslate"> </code><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> clause were illegal in the expressions in <a href="../reference/lexical_analysis.html#f-strings" class="reference internal"><span class="std std-ref">formatted string literals</span></a> due to a problem with the implementation. In Python 3.7 this restriction was lifted.

- More than 255 arguments can now be passed to a function, and a function can now have more than 255 parameters. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12844" class="reference external">bpo-12844</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18896" class="reference external">bpo-18896</a>.)

- <a href="../library/stdtypes.html#bytes.fromhex" class="reference internal" title="bytes.fromhex"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span>.fromhex()</code></span></a> and <a href="../library/stdtypes.html#bytearray.fromhex" class="reference internal" title="bytearray.fromhex"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span>.fromhex()</code></span></a> now ignore all ASCII whitespace, not only spaces. (Contributed by Robert Xiao in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28927" class="reference external">bpo-28927</a>.)

- <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>, and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> gained support for the new <a href="../library/stdtypes.html#str.isascii" class="reference internal" title="str.isascii"><span class="pre"><code class="sourceCode python">isascii()</code></span></a> method, which can be used to test if a string or bytes contain only the ASCII characters. (Contributed by INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32677" class="reference external">bpo-32677</a>.)

- <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> now displays module name and module <span class="pre">`__file__`</span> path when <span class="pre">`from`</span>` `<span class="pre">`...`</span>` `<span class="pre">`import`</span>` `<span class="pre">`...`</span> fails. (Contributed by Matthias Bussonnier in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29546" class="reference external">bpo-29546</a>.)

- Circular imports involving absolute imports with binding a submodule to a name are now supported. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30024" class="reference external">bpo-30024</a>.)

- <span class="pre">`object.__format__(x,`</span>` `<span class="pre">`'')`</span> is now equivalent to <span class="pre">`str(x)`</span> rather than <span class="pre">`format(str(self),`</span>` `<span class="pre">`'')`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28974" class="reference external">bpo-28974</a>.)

- In order to better support dynamic creation of stack traces, <a href="../library/types.html#types.TracebackType" class="reference internal" title="types.TracebackType"><span class="pre"><code class="sourceCode python">types.TracebackType</code></span></a> can now be instantiated from Python code, and the <a href="../reference/datamodel.html#traceback.tb_next" class="reference internal" title="traceback.tb_next"><span class="pre"><code class="sourceCode python">tb_next</code></span></a> attribute on <a href="../reference/datamodel.html#traceback-objects" class="reference internal"><span class="std std-ref">tracebacks</span></a> is now writable. (Contributed by Nathaniel J. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30579" class="reference external">bpo-30579</a>.)

- When using the <a href="../using/cmdline.html#cmdoption-m" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-m</code></span></a> switch, <span class="pre">`sys.path[0]`</span> is now eagerly expanded to the full starting directory path, rather than being left as the empty directory (which allows imports from the *current* working directory at the time when an import occurs) (Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33053" class="reference external">bpo-33053</a>.)

- The new <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span></a> <span class="pre">`importtime`</span> option or the <span id="index-28" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONPROFILEIMPORTTIME" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONPROFILEIMPORTTIME</code></span></a> environment variable can be used to show the timing of each module import. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31415" class="reference external">bpo-31415</a>.)

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="contextvars" class="section">

<span id="whatsnew37-pep567"></span>

### contextvars<a href="#contextvars" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/contextvars.html#module-contextvars" class="reference internal" title="contextvars: Context Variables"><span class="pre"><code class="sourceCode python">contextvars</code></span></a> module and a set of <a href="../c-api/contextvars.html#contextvarsobjects" class="reference internal"><span class="std std-ref">new C APIs</span></a> introduce support for *context variables*. Context variables are conceptually similar to thread-local variables. Unlike TLS, context variables support asynchronous code correctly.

The <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> and <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> modules have been updated to use and support context variables out of the box. Particularly the active decimal context is now stored in a context variable, which allows decimal operations to work with the correct context in asynchronous code.

<div class="admonition seealso">

See also

<span id="index-29" class="target"></span><a href="https://peps.python.org/pep-0567/" class="pep reference external"><strong>PEP 567</strong></a> – Context Variables  
PEP written and implemented by Yury Selivanov

</div>

</div>

<div id="dataclasses" class="section">

<span id="whatsnew37-pep557"></span>

### dataclasses<a href="#dataclasses" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/dataclasses.html#dataclasses.dataclass" class="reference internal" title="dataclasses.dataclass"><span class="pre"><code class="sourceCode python">dataclass()</code></span></a> decorator provides a way to declare *data classes*. A data class describes its attributes using class variable annotations. Its constructor and other magic methods, such as <a href="../reference/datamodel.html#object.__repr__" class="reference internal" title="object.__repr__"><span class="pre"><code class="sourceCode python"><span class="fu">__repr__</span>()</code></span></a>, <a href="../reference/datamodel.html#object.__eq__" class="reference internal" title="object.__eq__"><span class="pre"><code class="sourceCode python"><span class="fu">__eq__</span>()</code></span></a>, and <a href="../reference/datamodel.html#object.__hash__" class="reference internal" title="object.__hash__"><span class="pre"><code class="sourceCode python"><span class="fu">__hash__</span>()</code></span></a> are generated automatically.

Example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    @dataclass
    class Point:
        x: float
        y: float
        z: float = 0.0

    p = Point(1.5, 2.5)
    print(p)   # produces "Point(x=1.5, y=2.5, z=0.0)"

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0557/" class="pep reference external"><strong>PEP 557</strong></a> – Data Classes  
PEP written and implemented by Eric V. Smith

</div>

</div>

<div id="importlib-resources" class="section">

<span id="whatsnew37-importlib-resources"></span>

### importlib.resources<a href="#importlib-resources" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/importlib.resources.html#module-importlib.resources" class="reference internal" title="importlib.resources: Package resource reading, opening, and access"><span class="pre"><code class="sourceCode python">importlib.resources</code></span></a> module provides several new APIs and one new ABC for access to, opening, and reading *resources* inside packages. Resources are roughly similar to files inside packages, but they needn’t be actual files on the physical file system. Module loaders can provide a <span class="pre">`get_resource_reader()`</span> function which returns a <a href="../library/importlib.html#importlib.abc.ResourceReader" class="reference internal" title="importlib.abc.ResourceReader"><span class="pre"><code class="sourceCode python">importlib.abc.ResourceReader</code></span></a> instance to support this new API. Built-in file path loaders and zip file loaders both support this.

Contributed by Barry Warsaw and Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32248" class="reference external">bpo-32248</a>.

<div class="admonition seealso">

See also

<a href="https://importlib-resources.readthedocs.io/en/latest/" class="reference external">importlib_resources</a> – a PyPI backport for earlier Python versions.

</div>

</div>

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="argparse" class="section">

### argparse<a href="#argparse" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/argparse.html#argparse.ArgumentParser.parse_intermixed_args" class="reference internal" title="argparse.ArgumentParser.parse_intermixed_args"><span class="pre"><code class="sourceCode python">ArgumentParser.parse_intermixed_args()</code></span></a> method allows intermixing options and positional arguments. (Contributed by paul.j3 in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14191" class="reference external">bpo-14191</a>.)

</div>

<div id="asyncio" class="section">

<span id="whatsnew37-asyncio"></span>

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> module has received many new features, usability and <a href="#whatsnew37-asyncio-perf" class="reference internal"><span class="std std-ref">performance improvements</span></a>. Notable changes include:

- The new <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional</span></a> <a href="../library/asyncio-runner.html#asyncio.run" class="reference internal" title="asyncio.run"><span class="pre"><code class="sourceCode python">asyncio.run()</code></span></a> function can be used to run a coroutine from synchronous code by automatically creating and destroying the event loop. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32314" class="reference external">bpo-32314</a>.)

- asyncio gained support for <a href="../library/contextvars.html#module-contextvars" class="reference internal" title="contextvars: Context Variables"><span class="pre"><code class="sourceCode python">contextvars</code></span></a>. <a href="../library/asyncio-eventloop.html#asyncio.loop.call_soon" class="reference internal" title="asyncio.loop.call_soon"><span class="pre"><code class="sourceCode python">loop.call_soon()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.call_soon_threadsafe" class="reference internal" title="asyncio.loop.call_soon_threadsafe"><span class="pre"><code class="sourceCode python">loop.call_soon_threadsafe()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.call_later" class="reference internal" title="asyncio.loop.call_later"><span class="pre"><code class="sourceCode python">loop.call_later()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.call_at" class="reference internal" title="asyncio.loop.call_at"><span class="pre"><code class="sourceCode python">loop.call_at()</code></span></a>, and <a href="../library/asyncio-future.html#asyncio.Future.add_done_callback" class="reference internal" title="asyncio.Future.add_done_callback"><span class="pre"><code class="sourceCode python">Future.add_done_callback()</code></span></a> have a new optional keyword-only *context* parameter. <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">Tasks</code></span></a> now track their context automatically. See <span id="index-31" class="target"></span><a href="https://peps.python.org/pep-0567/" class="pep reference external"><strong>PEP 567</strong></a> for more details. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32436" class="reference external">bpo-32436</a>.)

- The new <a href="../library/asyncio-task.html#asyncio.create_task" class="reference internal" title="asyncio.create_task"><span class="pre"><code class="sourceCode python">asyncio.create_task()</code></span></a> function has been added as a shortcut to <span class="pre">`asyncio.get_event_loop().create_task()`</span>. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32311" class="reference external">bpo-32311</a>.)

- The new <a href="../library/asyncio-eventloop.html#asyncio.loop.start_tls" class="reference internal" title="asyncio.loop.start_tls"><span class="pre"><code class="sourceCode python">loop.start_tls()</code></span></a> method can be used to upgrade an existing connection to TLS. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23749" class="reference external">bpo-23749</a>.)

- The new <a href="../library/asyncio-eventloop.html#asyncio.loop.sock_recv_into" class="reference internal" title="asyncio.loop.sock_recv_into"><span class="pre"><code class="sourceCode python">loop.sock_recv_into()</code></span></a> method allows reading data from a socket directly into a provided buffer making it possible to reduce data copies. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31819" class="reference external">bpo-31819</a>.)

- The new <a href="../library/asyncio-task.html#asyncio.current_task" class="reference internal" title="asyncio.current_task"><span class="pre"><code class="sourceCode python">asyncio.current_task()</code></span></a> function returns the currently running <a href="../library/asyncio-task.html#asyncio.Task" class="reference internal" title="asyncio.Task"><span class="pre"><code class="sourceCode python">Task</code></span></a> instance, and the new <a href="../library/asyncio-task.html#asyncio.all_tasks" class="reference internal" title="asyncio.all_tasks"><span class="pre"><code class="sourceCode python">asyncio.all_tasks()</code></span></a> function returns a set of all existing <span class="pre">`Task`</span> instances in a given loop. The <span class="pre">`Task.current_task()`</span> and <span class="pre">`Task.all_tasks()`</span> methods have been deprecated. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32250" class="reference external">bpo-32250</a>.)

- The new *provisional* <a href="../library/asyncio-protocol.html#asyncio.BufferedProtocol" class="reference internal" title="asyncio.BufferedProtocol"><span class="pre"><code class="sourceCode python">BufferedProtocol</code></span></a> class allows implementing streaming protocols with manual control over the receive buffer. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32251" class="reference external">bpo-32251</a>.)

- The new <a href="../library/asyncio-eventloop.html#asyncio.get_running_loop" class="reference internal" title="asyncio.get_running_loop"><span class="pre"><code class="sourceCode python">asyncio.get_running_loop()</code></span></a> function returns the currently running loop, and raises a <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> if no loop is running. This is in contrast with <a href="../library/asyncio-eventloop.html#asyncio.get_event_loop" class="reference internal" title="asyncio.get_event_loop"><span class="pre"><code class="sourceCode python">asyncio.get_event_loop()</code></span></a>, which will *create* a new event loop if none is running. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32269" class="reference external">bpo-32269</a>.)

- The new <a href="../library/asyncio-stream.html#asyncio.StreamWriter.wait_closed" class="reference internal" title="asyncio.StreamWriter.wait_closed"><span class="pre"><code class="sourceCode python">StreamWriter.wait_closed()</code></span></a> coroutine method allows waiting until the stream writer is closed. The new <a href="../library/asyncio-stream.html#asyncio.StreamWriter.is_closing" class="reference internal" title="asyncio.StreamWriter.is_closing"><span class="pre"><code class="sourceCode python">StreamWriter.is_closing()</code></span></a> method can be used to determine if the writer is closing. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32391" class="reference external">bpo-32391</a>.)

- The new <a href="../library/asyncio-eventloop.html#asyncio.loop.sock_sendfile" class="reference internal" title="asyncio.loop.sock_sendfile"><span class="pre"><code class="sourceCode python">loop.sock_sendfile()</code></span></a> coroutine method allows sending files using <a href="../library/os.html#os.sendfile" class="reference internal" title="os.sendfile"><span class="pre"><code class="sourceCode python">os.sendfile</code></span></a> when possible. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32410" class="reference external">bpo-32410</a>.)

- The new <a href="../library/asyncio-future.html#asyncio.Future.get_loop" class="reference internal" title="asyncio.Future.get_loop"><span class="pre"><code class="sourceCode python">Future.get_loop()</code></span></a> and <span class="pre">`Task.get_loop()`</span> methods return the instance of the loop on which a task or a future were created. <a href="../library/asyncio-eventloop.html#asyncio.Server.get_loop" class="reference internal" title="asyncio.Server.get_loop"><span class="pre"><code class="sourceCode python">Server.get_loop()</code></span></a> allows doing the same for <a href="../library/asyncio-eventloop.html#asyncio.Server" class="reference internal" title="asyncio.Server"><span class="pre"><code class="sourceCode python">asyncio.Server</code></span></a> objects. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32415" class="reference external">bpo-32415</a> and Srinivas Reddy Thatiparthy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32418" class="reference external">bpo-32418</a>.)

- It is now possible to control how instances of <a href="../library/asyncio-eventloop.html#asyncio.Server" class="reference internal" title="asyncio.Server"><span class="pre"><code class="sourceCode python">asyncio.Server</code></span></a> begin serving. Previously, the server would start serving immediately when created. The new *start_serving* keyword argument to <a href="../library/asyncio-eventloop.html#asyncio.loop.create_server" class="reference internal" title="asyncio.loop.create_server"><span class="pre"><code class="sourceCode python">loop.create_server()</code></span></a> and <a href="../library/asyncio-eventloop.html#asyncio.loop.create_unix_server" class="reference internal" title="asyncio.loop.create_unix_server"><span class="pre"><code class="sourceCode python">loop.create_unix_server()</code></span></a>, as well as <a href="../library/asyncio-eventloop.html#asyncio.Server.start_serving" class="reference internal" title="asyncio.Server.start_serving"><span class="pre"><code class="sourceCode python">Server.start_serving()</code></span></a>, and <a href="../library/asyncio-eventloop.html#asyncio.Server.serve_forever" class="reference internal" title="asyncio.Server.serve_forever"><span class="pre"><code class="sourceCode python">Server.serve_forever()</code></span></a> can be used to decouple server instantiation and serving. The new <a href="../library/asyncio-eventloop.html#asyncio.Server.is_serving" class="reference internal" title="asyncio.Server.is_serving"><span class="pre"><code class="sourceCode python">Server.is_serving()</code></span></a> method returns <span class="pre">`True`</span> if the server is serving. <a href="../library/asyncio-eventloop.html#asyncio.Server" class="reference internal" title="asyncio.Server"><span class="pre"><code class="sourceCode python">Server</code></span></a> objects are now asynchronous context managers:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      srv = await loop.create_server(...)

      async with srv:
          # some code

      # At this point, srv is closed and no longer accepts new connections.

  </div>

  </div>

  (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32662" class="reference external">bpo-32662</a>.)

- Callback objects returned by <a href="../library/asyncio-eventloop.html#asyncio.loop.call_later" class="reference internal" title="asyncio.loop.call_later"><span class="pre"><code class="sourceCode python">loop.call_later()</code></span></a> gained the new <a href="../library/asyncio-eventloop.html#asyncio.TimerHandle.when" class="reference internal" title="asyncio.TimerHandle.when"><span class="pre"><code class="sourceCode python">when()</code></span></a> method which returns an absolute scheduled callback timestamp. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32741" class="reference external">bpo-32741</a>.)

- The <a href="../library/asyncio-eventloop.html#asyncio.loop.create_datagram_endpoint" class="reference internal" title="asyncio.loop.create_datagram_endpoint"><span class="pre"><code class="sourceCode python">loop.create_datagram_endpoint()</code></span><code class="sourceCode python"> </code></a> method gained support for Unix sockets. (Contributed by Quentin Dawans in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31245" class="reference external">bpo-31245</a>.)

- The <a href="../library/asyncio-stream.html#asyncio.open_connection" class="reference internal" title="asyncio.open_connection"><span class="pre"><code class="sourceCode python">asyncio.open_connection()</code></span></a>, <a href="../library/asyncio-stream.html#asyncio.start_server" class="reference internal" title="asyncio.start_server"><span class="pre"><code class="sourceCode python">asyncio.start_server()</code></span></a> functions, <a href="../library/asyncio-eventloop.html#asyncio.loop.create_connection" class="reference internal" title="asyncio.loop.create_connection"><span class="pre"><code class="sourceCode python">loop.create_connection()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.create_server" class="reference internal" title="asyncio.loop.create_server"><span class="pre"><code class="sourceCode python">loop.create_server()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.connect_accepted_socket" class="reference internal" title="asyncio.loop.connect_accepted_socket"><span class="pre"><code class="sourceCode python">loop.create_accepted_socket()</code></span></a> methods and their corresponding UNIX socket variants now accept the *ssl_handshake_timeout* keyword argument. (Contributed by Neil Aspinall in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29970" class="reference external">bpo-29970</a>.)

- The new <a href="../library/asyncio-eventloop.html#asyncio.Handle.cancelled" class="reference internal" title="asyncio.Handle.cancelled"><span class="pre"><code class="sourceCode python">Handle.cancelled()</code></span></a> method returns <span class="pre">`True`</span> if the callback was cancelled. (Contributed by Marat Sharafutdinov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31943" class="reference external">bpo-31943</a>.)

- The asyncio source has been converted to use the <a href="../reference/compound_stmts.html#async" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span></a>/<a href="../reference/expressions.html#await" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">await</code></span></a> syntax. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32193" class="reference external">bpo-32193</a>.)

- The new <a href="../library/asyncio-protocol.html#asyncio.ReadTransport.is_reading" class="reference internal" title="asyncio.ReadTransport.is_reading"><span class="pre"><code class="sourceCode python">ReadTransport.is_reading()</code></span></a> method can be used to determine the reading state of the transport. Additionally, calls to <a href="../library/asyncio-protocol.html#asyncio.ReadTransport.resume_reading" class="reference internal" title="asyncio.ReadTransport.resume_reading"><span class="pre"><code class="sourceCode python">ReadTransport.resume_reading()</code></span></a> and <a href="../library/asyncio-protocol.html#asyncio.ReadTransport.pause_reading" class="reference internal" title="asyncio.ReadTransport.pause_reading"><span class="pre"><code class="sourceCode python">ReadTransport.pause_reading()</code></span></a> are now idempotent. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32356" class="reference external">bpo-32356</a>.)

- Loop methods which accept socket paths now support passing <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like objects</span></a>. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32066" class="reference external">bpo-32066</a>.)

- In <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> TCP sockets on Linux are now created with <span class="pre">`TCP_NODELAY`</span> flag set by default. (Contributed by Yury Selivanov and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27456" class="reference external">bpo-27456</a>.)

- Exceptions occurring in cancelled tasks are no longer logged. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30508" class="reference external">bpo-30508</a>.)

- New <span class="pre">`WindowsSelectorEventLoopPolicy`</span> and <span class="pre">`WindowsProactorEventLoopPolicy`</span> classes. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33792" class="reference external">bpo-33792</a>.)

Several <span class="pre">`asyncio`</span> APIs have been <a href="#whatsnew37-asyncio-deprecated" class="reference internal"><span class="std std-ref">deprecated</span></a>.

</div>

<div id="binascii" class="section">

### binascii<a href="#binascii" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/binascii.html#binascii.b2a_uu" class="reference internal" title="binascii.b2a_uu"><span class="pre"><code class="sourceCode python">b2a_uu()</code></span></a> function now accepts an optional *backtick* keyword argument. When it’s true, zeros are represented by <span class="pre">`` '`' ``</span> instead of spaces. (Contributed by Xiang Zhang in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30103" class="reference external">bpo-30103</a>.)

</div>

<div id="calendar" class="section">

### calendar<a href="#calendar" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/calendar.html#calendar.HTMLCalendar" class="reference internal" title="calendar.HTMLCalendar"><span class="pre"><code class="sourceCode python">HTMLCalendar</code></span></a> class has new class attributes which ease the customization of CSS classes in the produced HTML calendar. (Contributed by Oz Tiram in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30095" class="reference external">bpo-30095</a>.)

</div>

<div id="collections" class="section">

### collections<a href="#collections" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`collections.namedtuple()`</span> now supports default values. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32320" class="reference external">bpo-32320</a>.)

</div>

<div id="compileall" class="section">

### compileall<a href="#compileall" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/compileall.html#compileall.compile_dir" class="reference internal" title="compileall.compile_dir"><span class="pre"><code class="sourceCode python">compileall.compile_dir()</code></span></a> learned the new *invalidation_mode* parameter, which can be used to enable <a href="#whatsnew37-pep552" class="reference internal"><span class="std std-ref">hash-based .pyc invalidation</span></a>. The invalidation mode can also be specified on the command line using the new <span class="pre">`--invalidation-mode`</span> argument. (Contributed by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31650" class="reference external">bpo-31650</a>.)

</div>

<div id="concurrent-futures" class="section">

### concurrent.futures<a href="#concurrent-futures" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor" class="reference internal" title="concurrent.futures.ProcessPoolExecutor"><span class="pre"><code class="sourceCode python">ProcessPoolExecutor</code></span></a> and <a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">ThreadPoolExecutor</code></span></a> now support the new *initializer* and *initargs* constructor arguments. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21423" class="reference external">bpo-21423</a>.)

The <a href="../library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor" class="reference internal" title="concurrent.futures.ProcessPoolExecutor"><span class="pre"><code class="sourceCode python">ProcessPoolExecutor</code></span></a> can now take the multiprocessing context via the new *mp_context* argument. (Contributed by Thomas Moreau in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31540" class="reference external">bpo-31540</a>.)

</div>

<div id="contextlib" class="section">

### contextlib<a href="#contextlib" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/contextlib.html#contextlib.nullcontext" class="reference internal" title="contextlib.nullcontext"><span class="pre"><code class="sourceCode python">nullcontext()</code></span></a> is a simpler and faster no-op context manager than <a href="../library/contextlib.html#contextlib.ExitStack" class="reference internal" title="contextlib.ExitStack"><span class="pre"><code class="sourceCode python">ExitStack</code></span></a>. (Contributed by Jesse-Bakker in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10049" class="reference external">bpo-10049</a>.)

The new <a href="../library/contextlib.html#contextlib.asynccontextmanager" class="reference internal" title="contextlib.asynccontextmanager"><span class="pre"><code class="sourceCode python">asynccontextmanager()</code></span></a>, <a href="../library/contextlib.html#contextlib.AbstractAsyncContextManager" class="reference internal" title="contextlib.AbstractAsyncContextManager"><span class="pre"><code class="sourceCode python">AbstractAsyncContextManager</code></span></a>, and <a href="../library/contextlib.html#contextlib.AsyncExitStack" class="reference internal" title="contextlib.AsyncExitStack"><span class="pre"><code class="sourceCode python">AsyncExitStack</code></span></a> have been added to complement their synchronous counterparts. (Contributed by Jelle Zijlstra in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29679" class="reference external">bpo-29679</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30241" class="reference external">bpo-30241</a>, and by Alexander Mohr and Ilya Kulakov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29302" class="reference external">bpo-29302</a>.)

</div>

<div id="cprofile" class="section">

### cProfile<a href="#cprofile" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/profile.html#module-cProfile" class="reference internal" title="cProfile"><span class="pre"><code class="sourceCode python">cProfile</code></span></a> command line now accepts <span class="pre">`-m`</span>` `<span class="pre">`module_name`</span> as an alternative to script path. (Contributed by Sanyam Khurana in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21862" class="reference external">bpo-21862</a>.)

</div>

<div id="crypt" class="section">

### crypt<a href="#crypt" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`crypt`</span> module now supports the Blowfish hashing method. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31664" class="reference external">bpo-31664</a>.)

The <span class="pre">`mksalt()`</span> function now allows specifying the number of rounds for hashing. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31702" class="reference external">bpo-31702</a>.)

</div>

<div id="datetime" class="section">

### datetime<a href="#datetime" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/datetime.html#datetime.datetime.fromisoformat" class="reference internal" title="datetime.datetime.fromisoformat"><span class="pre"><code class="sourceCode python">datetime.fromisoformat()</code></span></a> method constructs a <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> object from a string in one of the formats output by <a href="../library/datetime.html#datetime.datetime.isoformat" class="reference internal" title="datetime.datetime.isoformat"><span class="pre"><code class="sourceCode python">datetime.isoformat()</code></span></a>. (Contributed by Paul Ganssle in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15873" class="reference external">bpo-15873</a>.)

The <a href="../library/datetime.html#datetime.tzinfo" class="reference internal" title="datetime.tzinfo"><span class="pre"><code class="sourceCode python">tzinfo</code></span></a> class now supports sub-minute offsets. (Contributed by Alexander Belopolsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5288" class="reference external">bpo-5288</a>.)

</div>

<div id="dbm" class="section">

### dbm<a href="#dbm" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/dbm.html#module-dbm.dumb" class="reference internal" title="dbm.dumb: Portable implementation of the simple DBM interface."><span class="pre"><code class="sourceCode python">dbm.dumb</code></span></a> now supports reading read-only files and no longer writes the index file when it is not changed.

</div>

<div id="decimal" class="section">

### decimal<a href="#decimal" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> module now uses <a href="#whatsnew37-pep567" class="reference internal"><span class="std std-ref">context variables</span></a> to store the decimal context. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32630" class="reference external">bpo-32630</a>.)

</div>

<div id="dis" class="section">

### dis<a href="#dis" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/dis.html#dis.dis" class="reference internal" title="dis.dis"><span class="pre"><code class="sourceCode python">dis()</code></span></a> function is now able to disassemble nested code objects (the code of comprehensions, generator expressions and nested functions, and the code used for building nested classes). The maximum depth of disassembly recursion is controlled by the new *depth* parameter. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11822" class="reference external">bpo-11822</a>.)

</div>

<div id="distutils" class="section">

### distutils<a href="#distutils" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`README.rst`</span> is now included in the list of distutils standard READMEs and therefore included in source distributions. (Contributed by Ryan Gonzalez in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11913" class="reference external">bpo-11913</a>.)

</div>

<div id="enum" class="section">

### enum<a href="#enum" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/enum.html#enum.Enum" class="reference internal" title="enum.Enum"><span class="pre"><code class="sourceCode python">Enum</code></span></a> learned the new <span class="pre">`_ignore_`</span> class property, which allows listing the names of properties which should not become enum members. (Contributed by Ethan Furman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31801" class="reference external">bpo-31801</a>.)

In Python 3.8, attempting to check for non-Enum objects in <a href="../library/enum.html#enum.Enum" class="reference internal" title="enum.Enum"><span class="pre"><code class="sourceCode python">Enum</code></span></a> classes will raise a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> (e.g. <span class="pre">`1`</span>` `<span class="pre">`in`</span>` `<span class="pre">`Color`</span>); similarly, attempting to check for non-Flag objects in a <a href="../library/enum.html#enum.Flag" class="reference internal" title="enum.Flag"><span class="pre"><code class="sourceCode python">Flag</code></span></a> member will raise <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> (e.g. <span class="pre">`1`</span>` `<span class="pre">`in`</span>` `<span class="pre">`Perm.RW`</span>); currently, both operations return <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a> instead and are deprecated. (Contributed by Ethan Furman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33217" class="reference external">bpo-33217</a>.)

</div>

<div id="functools" class="section">

### functools<a href="#functools" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/functools.html#functools.singledispatch" class="reference internal" title="functools.singledispatch"><span class="pre"><code class="sourceCode python">functools.singledispatch()</code></span></a> now supports registering implementations using type annotations. (Contributed by Łukasz Langa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32227" class="reference external">bpo-32227</a>.)

</div>

<div id="gc" class="section">

### gc<a href="#gc" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/gc.html#gc.freeze" class="reference internal" title="gc.freeze"><span class="pre"><code class="sourceCode python">gc.freeze()</code></span></a> function allows freezing all objects tracked by the garbage collector and excluding them from future collections. This can be used before a POSIX <span class="pre">`fork()`</span> call to make the GC copy-on-write friendly or to speed up collection. The new <a href="../library/gc.html#gc.unfreeze" class="reference internal" title="gc.unfreeze"><span class="pre"><code class="sourceCode python">gc.unfreeze()</code></span></a> functions reverses this operation. Additionally, <a href="../library/gc.html#gc.get_freeze_count" class="reference internal" title="gc.get_freeze_count"><span class="pre"><code class="sourceCode python">gc.get_freeze_count()</code></span></a> can be used to obtain the number of frozen objects. (Contributed by Li Zekun in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31558" class="reference external">bpo-31558</a>.)

</div>

<div id="hmac" class="section">

### hmac<a href="#hmac" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/hmac.html#module-hmac" class="reference internal" title="hmac: Keyed-Hashing for Message Authentication (HMAC) implementation"><span class="pre"><code class="sourceCode python">hmac</code></span></a> module now has an optimized one-shot <a href="../library/hmac.html#hmac.digest" class="reference internal" title="hmac.digest"><span class="pre"><code class="sourceCode python">digest()</code></span></a> function, which is up to three times faster than <a href="../library/hmac.html#hmac.HMAC" class="reference internal" title="hmac.HMAC"><span class="pre"><code class="sourceCode python">HMAC()</code></span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32433" class="reference external">bpo-32433</a>.)

</div>

<div id="http-client" class="section">

### http.client<a href="#http-client" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/http.client.html#http.client.HTTPConnection" class="reference internal" title="http.client.HTTPConnection"><span class="pre"><code class="sourceCode python">HTTPConnection</code></span></a> and <a href="../library/http.client.html#http.client.HTTPSConnection" class="reference internal" title="http.client.HTTPSConnection"><span class="pre"><code class="sourceCode python">HTTPSConnection</code></span></a> now support the new *blocksize* argument for improved upload throughput. (Contributed by Nir Soffer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31945" class="reference external">bpo-31945</a>.)

</div>

<div id="http-server" class="section">

### http.server<a href="#http-server" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/http.server.html#http.server.SimpleHTTPRequestHandler" class="reference internal" title="http.server.SimpleHTTPRequestHandler"><span class="pre"><code class="sourceCode python">SimpleHTTPRequestHandler</code></span></a> now supports the HTTP <span class="pre">`If-Modified-Since`</span> header. The server returns the 304 response status if the target file was not modified after the time specified in the header. (Contributed by Pierre Quentel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29654" class="reference external">bpo-29654</a>.)

<a href="../library/http.server.html#http.server.SimpleHTTPRequestHandler" class="reference internal" title="http.server.SimpleHTTPRequestHandler"><span class="pre"><code class="sourceCode python">SimpleHTTPRequestHandler</code></span></a> accepts the new *directory* argument, in addition to the new <span class="pre">`--directory`</span> command line argument. With this parameter, the server serves the specified directory, by default it uses the current working directory. (Contributed by Stéphane Wirtel and Julien Palard in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28707" class="reference external">bpo-28707</a>.)

The new <a href="../library/http.server.html#http.server.ThreadingHTTPServer" class="reference internal" title="http.server.ThreadingHTTPServer"><span class="pre"><code class="sourceCode python">ThreadingHTTPServer</code></span></a> class uses threads to handle requests using <a href="../library/socketserver.html#socketserver.ThreadingMixIn" class="reference internal" title="socketserver.ThreadingMixIn"><span class="pre"><code class="sourceCode python">ThreadingMixIn</code></span></a>. It is used when <span class="pre">`http.server`</span> is run with <span class="pre">`-m`</span>. (Contributed by Julien Palard in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31639" class="reference external">bpo-31639</a>.)

</div>

<div id="idlelib-and-idle" class="section">

### idlelib and IDLE<a href="#idlelib-and-idle" class="headerlink" title="Link to this heading">¶</a>

Multiple fixes for autocompletion. (Contributed by Louie Lu in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15786" class="reference external">bpo-15786</a>.)

Module Browser (on the File menu, formerly called Class Browser), now displays nested functions and classes in addition to top-level functions and classes. (Contributed by Guilherme Polo, Cheryl Sabella, and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1612262" class="reference external">bpo-1612262</a>.)

The Settings dialog (Options, Configure IDLE) has been partly rewritten to improve both appearance and function. (Contributed by Cheryl Sabella and Terry Jan Reedy in multiple issues.)

The font sample now includes a selection of non-Latin characters so that users can better see the effect of selecting a particular font. (Contributed by Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13802" class="reference external">bpo-13802</a>.) The sample can be edited to include other characters. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31860" class="reference external">bpo-31860</a>.)

The IDLE features formerly implemented as extensions have been reimplemented as normal features. Their settings have been moved from the Extensions tab to other dialog tabs. (Contributed by Charles Wohlganger and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27099" class="reference external">bpo-27099</a>.)

Editor code context option revised. Box displays all context lines up to maxlines. Clicking on a context line jumps the editor to that line. Context colors for custom themes is added to Highlights tab of Settings dialog. (Contributed by Cheryl Sabella and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33642" class="reference external">bpo-33642</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33768" class="reference external">bpo-33768</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33679" class="reference external">bpo-33679</a>.)

On Windows, a new API call tells Windows that tk scales for DPI. On Windows 8.1+ or 10, with DPI compatibility properties of the Python binary unchanged, and a monitor resolution greater than 96 DPI, this should make text and lines sharper. It should otherwise have no effect. (Contributed by Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33656" class="reference external">bpo-33656</a>.)

New in 3.7.1:

Output over N lines (50 by default) is squeezed down to a button. N can be changed in the PyShell section of the General page of the Settings dialog. Fewer, but possibly extra long, lines can be squeezed by right clicking on the output. Squeezed output can be expanded in place by double-clicking the button or into the clipboard or a separate window by right-clicking the button. (Contributed by Tal Einat in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1529353" class="reference external">bpo-1529353</a>.)

The changes above have been backported to 3.6 maintenance releases.

NEW in 3.7.4:

Add “Run Customized” to the Run menu to run a module with customized settings. Any command line arguments entered are added to sys.argv. They re-appear in the box for the next customized run. One can also suppress the normal Shell main module restart. (Contributed by Cheryl Sabella, Terry Jan Reedy, and others in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5680" class="reference external">bpo-5680</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37627" class="reference external">bpo-37627</a>.)

New in 3.7.5:

Add optional line numbers for IDLE editor windows. Windows open without line numbers unless set otherwise in the General tab of the configuration dialog. Line numbers for an existing window are shown and hidden in the Options menu. (Contributed by Tal Einat and Saimadhav Heblikar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17535" class="reference external">bpo-17535</a>.)

</div>

<div id="importlib" class="section">

### importlib<a href="#importlib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/importlib.html#importlib.abc.ResourceReader" class="reference internal" title="importlib.abc.ResourceReader"><span class="pre"><code class="sourceCode python">importlib.abc.ResourceReader</code></span></a> ABC was introduced to support the loading of resources from packages. See also <a href="#whatsnew37-importlib-resources" class="reference internal"><span class="std std-ref">importlib.resources</span></a>. (Contributed by Barry Warsaw, Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32248" class="reference external">bpo-32248</a>.)

<a href="../library/importlib.html#importlib.reload" class="reference internal" title="importlib.reload"><span class="pre"><code class="sourceCode python">importlib.<span class="bu">reload</span>()</code></span></a> now raises <a href="../library/exceptions.html#ModuleNotFoundError" class="reference internal" title="ModuleNotFoundError"><span class="pre"><code class="sourceCode python"><span class="pp">ModuleNotFoundError</span></code></span></a> if the module lacks a spec. (Contributed by Garvit Khatri in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29851" class="reference external">bpo-29851</a>.)

<a href="../library/importlib.html#importlib.util.find_spec" class="reference internal" title="importlib.util.find_spec"><span class="pre"><code class="sourceCode python">importlib.util.find_spec()</code></span></a> now raises <a href="../library/exceptions.html#ModuleNotFoundError" class="reference internal" title="ModuleNotFoundError"><span class="pre"><code class="sourceCode python"><span class="pp">ModuleNotFoundError</span></code></span></a> instead of <a href="../library/exceptions.html#AttributeError" class="reference internal" title="AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> if the specified parent module is not a package (i.e. lacks a <span class="pre">`__path__`</span> attribute). (Contributed by Milan Oberkirch in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30436" class="reference external">bpo-30436</a>.)

The new <a href="../library/importlib.html#importlib.util.source_hash" class="reference internal" title="importlib.util.source_hash"><span class="pre"><code class="sourceCode python">importlib.util.source_hash()</code></span></a> can be used to compute the hash of the passed source. A <a href="#whatsnew37-pep552" class="reference internal"><span class="std std-ref">hash-based .pyc file</span></a> embeds the value returned by this function.

</div>

<div id="io" class="section">

### io<a href="#io" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/io.html#io.TextIOWrapper.reconfigure" class="reference internal" title="io.TextIOWrapper.reconfigure"><span class="pre"><code class="sourceCode python">TextIOWrapper.reconfigure()</code></span></a> method can be used to reconfigure the text stream with the new settings. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30526" class="reference external">bpo-30526</a> and INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15216" class="reference external">bpo-15216</a>.)

</div>

<div id="ipaddress" class="section">

### ipaddress<a href="#ipaddress" class="headerlink" title="Link to this heading">¶</a>

The new <span class="pre">`subnet_of()`</span> and <span class="pre">`supernet_of()`</span> methods of <a href="../library/ipaddress.html#ipaddress.IPv6Network" class="reference internal" title="ipaddress.IPv6Network"><span class="pre"><code class="sourceCode python">ipaddress.IPv6Network</code></span></a> and <a href="../library/ipaddress.html#ipaddress.IPv4Network" class="reference internal" title="ipaddress.IPv4Network"><span class="pre"><code class="sourceCode python">ipaddress.IPv4Network</code></span></a> can be used for network containment tests. (Contributed by Michel Albert and Cheryl Sabella in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20825" class="reference external">bpo-20825</a>.)

</div>

<div id="itertools" class="section">

### itertools<a href="#itertools" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/itertools.html#itertools.islice" class="reference internal" title="itertools.islice"><span class="pre"><code class="sourceCode python">itertools.islice()</code></span></a> now accepts <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python">integer<span class="op">-</span>like</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">objects</code></span></a> as start, stop, and slice arguments. (Contributed by Will Roberts in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30537" class="reference external">bpo-30537</a>.)

</div>

<div id="locale" class="section">

### locale<a href="#locale" class="headerlink" title="Link to this heading">¶</a>

The new *monetary* argument to <a href="../library/locale.html#locale.format_string" class="reference internal" title="locale.format_string"><span class="pre"><code class="sourceCode python">locale.format_string()</code></span></a> can be used to make the conversion use monetary thousands separators and grouping strings. (Contributed by Garvit in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10379" class="reference external">bpo-10379</a>.)

The <a href="../library/locale.html#locale.getpreferredencoding" class="reference internal" title="locale.getpreferredencoding"><span class="pre"><code class="sourceCode python">locale.getpreferredencoding()</code></span></a> function now always returns <span class="pre">`'UTF-8'`</span> on Android or when in the <a href="#whatsnew37-pep540" class="reference internal"><span class="std std-ref">forced UTF-8 mode</span></a>.

</div>

<div id="logging" class="section">

### logging<a href="#logging" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/logging.html#logging.Logger" class="reference internal" title="logging.Logger"><span class="pre"><code class="sourceCode python">Logger</code></span></a> instances can now be pickled. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30520" class="reference external">bpo-30520</a>.)

The new <a href="../library/logging.handlers.html#logging.StreamHandler.setStream" class="reference internal" title="logging.StreamHandler.setStream"><span class="pre"><code class="sourceCode python">StreamHandler.setStream()</code></span></a> method can be used to replace the logger stream after handler creation. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30522" class="reference external">bpo-30522</a>.)

It is now possible to specify keyword arguments to handler constructors in configuration passed to <a href="../library/logging.config.html#logging.config.fileConfig" class="reference internal" title="logging.config.fileConfig"><span class="pre"><code class="sourceCode python">logging.config.fileConfig()</code></span></a>. (Contributed by Preston Landers in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31080" class="reference external">bpo-31080</a>.)

</div>

<div id="math" class="section">

### math<a href="#math" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/math.html#math.remainder" class="reference internal" title="math.remainder"><span class="pre"><code class="sourceCode python">math.remainder()</code></span></a> function implements the IEEE 754-style remainder operation. (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29962" class="reference external">bpo-29962</a>.)

</div>

<div id="mimetypes" class="section">

### mimetypes<a href="#mimetypes" class="headerlink" title="Link to this heading">¶</a>

The MIME type of .bmp has been changed from <span class="pre">`'image/x-ms-bmp'`</span> to <span class="pre">`'image/bmp'`</span>. (Contributed by Nitish Chandra in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22589" class="reference external">bpo-22589</a>.)

</div>

<div id="msilib" class="section">

### msilib<a href="#msilib" class="headerlink" title="Link to this heading">¶</a>

The new <span class="pre">`Database.Close()`</span> method can be used to close the <span class="abbr">MSI</span> database. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20486" class="reference external">bpo-20486</a>.)

</div>

<div id="multiprocessing" class="section">

### multiprocessing<a href="#multiprocessing" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/multiprocessing.html#multiprocessing.Process.close" class="reference internal" title="multiprocessing.Process.close"><span class="pre"><code class="sourceCode python">Process.close()</code></span></a> method explicitly closes the process object and releases all resources associated with it. <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> is raised if the underlying process is still running. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30596" class="reference external">bpo-30596</a>.)

The new <a href="../library/multiprocessing.html#multiprocessing.Process.kill" class="reference internal" title="multiprocessing.Process.kill"><span class="pre"><code class="sourceCode python">Process.kill()</code></span></a> method can be used to terminate the process using the <a href="../library/signal.html#signal.SIGKILL" class="reference internal" title="signal.SIGKILL"><span class="pre"><code class="sourceCode python">SIGKILL</code></span></a> signal on Unix. (Contributed by Vitor Pereira in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30794" class="reference external">bpo-30794</a>.)

Non-daemonic threads created by <a href="../library/multiprocessing.html#multiprocessing.Process" class="reference internal" title="multiprocessing.Process"><span class="pre"><code class="sourceCode python">Process</code></span></a> are now joined on process exit. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18966" class="reference external">bpo-18966</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/os.html#os.fwalk" class="reference internal" title="os.fwalk"><span class="pre"><code class="sourceCode python">os.fwalk()</code></span></a> now accepts the *path* argument as <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28682" class="reference external">bpo-28682</a>.)

<a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a> gained support for <a href="../library/os.html#path-fd" class="reference internal"><span class="std std-ref">file descriptors</span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25996" class="reference external">bpo-25996</a>.)

The new <a href="../library/os.html#os.register_at_fork" class="reference internal" title="os.register_at_fork"><span class="pre"><code class="sourceCode python">register_at_fork()</code></span></a> function allows registering Python callbacks to be executed at process fork. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16500" class="reference external">bpo-16500</a>.)

Added <a href="../library/os.html#os.preadv" class="reference internal" title="os.preadv"><span class="pre"><code class="sourceCode python">os.preadv()</code></span></a> (combine the functionality of <a href="../library/os.html#os.readv" class="reference internal" title="os.readv"><span class="pre"><code class="sourceCode python">os.readv()</code></span></a> and <a href="../library/os.html#os.pread" class="reference internal" title="os.pread"><span class="pre"><code class="sourceCode python">os.pread()</code></span></a>) and <a href="../library/os.html#os.pwritev" class="reference internal" title="os.pwritev"><span class="pre"><code class="sourceCode python">os.pwritev()</code></span></a> functions (combine the functionality of <a href="../library/os.html#os.writev" class="reference internal" title="os.writev"><span class="pre"><code class="sourceCode python">os.writev()</code></span></a> and <a href="../library/os.html#os.pwrite" class="reference internal" title="os.pwrite"><span class="pre"><code class="sourceCode python">os.pwrite()</code></span></a>). (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31368" class="reference external">bpo-31368</a>.)

The mode argument of <a href="../library/os.html#os.makedirs" class="reference internal" title="os.makedirs"><span class="pre"><code class="sourceCode python">os.makedirs()</code></span></a> no longer affects the file permission bits of newly created intermediate-level directories. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19930" class="reference external">bpo-19930</a>.)

<a href="../library/os.html#os.dup2" class="reference internal" title="os.dup2"><span class="pre"><code class="sourceCode python">os.dup2()</code></span></a> now returns the new file descriptor. Previously, <span class="pre">`None`</span> was always returned. (Contributed by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32441" class="reference external">bpo-32441</a>.)

The structure returned by <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> now contains the <a href="../library/os.html#os.stat_result.st_fstype" class="reference internal" title="os.stat_result.st_fstype"><span class="pre"><code class="sourceCode python">st_fstype</code></span></a> attribute on Solaris and its derivatives. (Contributed by Jesús Cea Avión in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32659" class="reference external">bpo-32659</a>.)

</div>

<div id="pathlib" class="section">

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/pathlib.html#pathlib.Path.is_mount" class="reference internal" title="pathlib.Path.is_mount"><span class="pre"><code class="sourceCode python">Path.is_mount()</code></span></a> method is now available on POSIX systems and can be used to determine whether a path is a mount point. (Contributed by Cooper Ry Lees in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30897" class="reference external">bpo-30897</a>.)

</div>

<div id="pdb" class="section">

### pdb<a href="#pdb" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pdb.html#pdb.set_trace" class="reference internal" title="pdb.set_trace"><span class="pre"><code class="sourceCode python">pdb.set_trace()</code></span></a> now takes an optional *header* keyword-only argument. If given, it is printed to the console just before debugging begins. (Contributed by Barry Warsaw in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31389" class="reference external">bpo-31389</a>.)

<a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a> command line now accepts <span class="pre">`-m`</span>` `<span class="pre">`module_name`</span> as an alternative to script file. (Contributed by Mario Corchero in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32206" class="reference external">bpo-32206</a>.)

</div>

<div id="py-compile" class="section">

### py_compile<a href="#py-compile" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/py_compile.html#py_compile.compile" class="reference internal" title="py_compile.compile"><span class="pre"><code class="sourceCode python">py_compile.<span class="bu">compile</span>()</code></span></a> – and by extension, <a href="../library/compileall.html#module-compileall" class="reference internal" title="compileall: Tools for byte-compiling all Python source files in a directory tree."><span class="pre"><code class="sourceCode python">compileall</code></span></a> – now respects the <span id="index-32" class="target"></span><span class="pre">`SOURCE_DATE_EPOCH`</span> environment variable by unconditionally creating <span class="pre">`.pyc`</span> files for hash-based validation. This allows for guaranteeing <a href="https://reproducible-builds.org/" class="reference external">reproducible builds</a> of <span class="pre">`.pyc`</span> files when they are created eagerly. (Contributed by Bernhard M. Wiedemann in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29708" class="reference external">bpo-29708</a>.)

</div>

<div id="pydoc" class="section">

### pydoc<a href="#pydoc" class="headerlink" title="Link to this heading">¶</a>

The pydoc server can now bind to an arbitrary hostname specified by the new <span class="pre">`-n`</span> command-line argument. (Contributed by Feanil Patel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31128" class="reference external">bpo-31128</a>.)

</div>

<div id="queue" class="section">

### queue<a href="#queue" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/queue.html#queue.SimpleQueue" class="reference internal" title="queue.SimpleQueue"><span class="pre"><code class="sourceCode python">SimpleQueue</code></span></a> class is an unbounded <span class="abbr">FIFO</span> queue. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14976" class="reference external">bpo-14976</a>.)

</div>

<div id="re" class="section">

### re<a href="#re" class="headerlink" title="Link to this heading">¶</a>

The flags <a href="../library/re.html#re.ASCII" class="reference internal" title="re.ASCII"><span class="pre"><code class="sourceCode python">re.ASCII</code></span></a>, <a href="../library/re.html#re.LOCALE" class="reference internal" title="re.LOCALE"><span class="pre"><code class="sourceCode python">re.LOCALE</code></span></a> and <a href="../library/re.html#re.UNICODE" class="reference internal" title="re.UNICODE"><span class="pre"><code class="sourceCode python">re.UNICODE</code></span></a> can be set within the scope of a group. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31690" class="reference external">bpo-31690</a>.)

<a href="../library/re.html#re.split" class="reference internal" title="re.split"><span class="pre"><code class="sourceCode python">re.split()</code></span></a> now supports splitting on a pattern like <span class="pre">`r'\b'`</span>, <span class="pre">`'^$'`</span> or <span class="pre">`(?=-)`</span> that matches an empty string. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25054" class="reference external">bpo-25054</a>.)

Regular expressions compiled with the <a href="../library/re.html#re.LOCALE" class="reference internal" title="re.LOCALE"><span class="pre"><code class="sourceCode python">re.LOCALE</code></span></a> flag no longer depend on the locale at compile time. Locale settings are applied only when the compiled regular expression is used. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30215" class="reference external">bpo-30215</a>.)

<a href="../library/exceptions.html#FutureWarning" class="reference internal" title="FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a> is now emitted if a regular expression contains character set constructs that will change semantically in the future, such as nested sets and set operations. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30349" class="reference external">bpo-30349</a>.)

Compiled regular expression and match objects can now be copied using <a href="../library/copy.html#copy.copy" class="reference internal" title="copy.copy"><span class="pre"><code class="sourceCode python">copy.copy()</code></span></a> and <a href="../library/copy.html#copy.deepcopy" class="reference internal" title="copy.deepcopy"><span class="pre"><code class="sourceCode python">copy.deepcopy()</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10076" class="reference external">bpo-10076</a>.)

</div>

<div id="signal" class="section">

### signal<a href="#signal" class="headerlink" title="Link to this heading">¶</a>

The new *warn_on_full_buffer* argument to the <a href="../library/signal.html#signal.set_wakeup_fd" class="reference internal" title="signal.set_wakeup_fd"><span class="pre"><code class="sourceCode python">signal.set_wakeup_fd()</code></span></a> function makes it possible to specify whether Python prints a warning on stderr when the wakeup buffer overflows. (Contributed by Nathaniel J. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30050" class="reference external">bpo-30050</a>.)

</div>

<div id="socket" class="section">

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/socket.html#socket.socket.getblocking" class="reference internal" title="socket.socket.getblocking"><span class="pre"><code class="sourceCode python">socket.getblocking()</code></span></a> method returns <span class="pre">`True`</span> if the socket is in blocking mode and <span class="pre">`False`</span> otherwise. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32373" class="reference external">bpo-32373</a>.)

The new <a href="../library/socket.html#socket.close" class="reference internal" title="socket.close"><span class="pre"><code class="sourceCode python">socket.close()</code></span></a> function closes the passed socket file descriptor. This function should be used instead of <a href="../library/os.html#os.close" class="reference internal" title="os.close"><span class="pre"><code class="sourceCode python">os.close()</code></span></a> for better compatibility across platforms. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32454" class="reference external">bpo-32454</a>.)

The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module now exposes the <a href="../library/socket.html#socket-unix-constants" class="reference internal"><span class="std std-ref">socket.TCP_CONGESTION</span></a> (Linux 2.6.13), <a href="../library/socket.html#socket-unix-constants" class="reference internal"><span class="std std-ref">socket.TCP_USER_TIMEOUT</span></a> (Linux 2.6.37), and <a href="../library/socket.html#socket-unix-constants" class="reference internal"><span class="std std-ref">socket.TCP_NOTSENT_LOWAT</span></a> (Linux 3.12) constants. (Contributed by Omar Sandoval in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26273" class="reference external">bpo-26273</a> and Nathaniel J. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29728" class="reference external">bpo-29728</a>.)

Support for <a href="../library/socket.html#socket.AF_VSOCK" class="reference internal" title="socket.AF_VSOCK"><span class="pre"><code class="sourceCode python">socket.AF_VSOCK</code></span></a> sockets has been added to allow communication between virtual machines and their hosts. (Contributed by Cathy Avery in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27584" class="reference external">bpo-27584</a>.)

Sockets now auto-detect family, type and protocol from file descriptor by default. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28134" class="reference external">bpo-28134</a>.)

</div>

<div id="socketserver" class="section">

### socketserver<a href="#socketserver" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/socketserver.html#socketserver.BaseServer.server_close" class="reference internal" title="socketserver.BaseServer.server_close"><span class="pre"><code class="sourceCode python">socketserver.ThreadingMixIn.server_close</code></span></a> now waits until all non-daemon threads complete. <a href="../library/socketserver.html#socketserver.BaseServer.server_close" class="reference internal" title="socketserver.BaseServer.server_close"><span class="pre"><code class="sourceCode python">socketserver.ForkingMixIn.server_close</code></span></a> now waits until all child processes complete.

Add a new <a href="../library/socketserver.html#socketserver.ThreadingMixIn.block_on_close" class="reference internal" title="socketserver.ThreadingMixIn.block_on_close"><span class="pre"><code class="sourceCode python">socketserver.ForkingMixIn.block_on_close</code></span></a> class attribute to <a href="../library/socketserver.html#socketserver.ForkingMixIn" class="reference internal" title="socketserver.ForkingMixIn"><span class="pre"><code class="sourceCode python">socketserver.ForkingMixIn</code></span></a> and <a href="../library/socketserver.html#socketserver.ThreadingMixIn" class="reference internal" title="socketserver.ThreadingMixIn"><span class="pre"><code class="sourceCode python">socketserver.ThreadingMixIn</code></span></a> classes. Set the class attribute to <span class="pre">`False`</span> to get the pre-3.7 behaviour.

</div>

<div id="sqlite3" class="section">

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/sqlite3.html#sqlite3.Connection" class="reference internal" title="sqlite3.Connection"><span class="pre"><code class="sourceCode python">sqlite3.Connection</code></span></a> now exposes the <a href="../library/sqlite3.html#sqlite3.Connection.backup" class="reference internal" title="sqlite3.Connection.backup"><span class="pre"><code class="sourceCode python">backup()</code></span></a> method when the underlying SQLite library is at version 3.6.11 or higher. (Contributed by Lele Gaifax in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27645" class="reference external">bpo-27645</a>.)

The *database* argument of <a href="../library/sqlite3.html#sqlite3.connect" class="reference internal" title="sqlite3.connect"><span class="pre"><code class="sourceCode python">sqlite3.<span class="ex">connect</span>()</code></span></a> now accepts any <a href="../glossary.html#term-path-like-object" class="reference internal"><span class="xref std std-term">path-like object</span></a>, instead of just a string. (Contributed by Anders Lorentsen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31843" class="reference external">bpo-31843</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module now uses OpenSSL’s builtin API instead of <span class="pre">`match_hostname()`</span> to check a host name or an IP address. Values are validated during TLS handshake. Any certificate validation error including failing the host name check now raises <a href="../library/ssl.html#ssl.SSLCertVerificationError" class="reference internal" title="ssl.SSLCertVerificationError"><span class="pre"><code class="sourceCode python">SSLCertVerificationError</code></span></a> and aborts the handshake with a proper TLS Alert message. The new exception contains additional information. Host name validation can be customized with <a href="../library/ssl.html#ssl.SSLContext.hostname_checks_common_name" class="reference internal" title="ssl.SSLContext.hostname_checks_common_name"><span class="pre"><code class="sourceCode python">SSLContext.hostname_checks_common_name</code></span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31399" class="reference external">bpo-31399</a>.)

<div class="admonition note">

Note

The improved host name check requires a *libssl* implementation compatible with OpenSSL 1.0.2 or 1.1. Consequently, OpenSSL 0.9.8 and 1.0.1 are no longer supported (see <a href="#platform-support-removals" class="reference internal"><span class="std std-ref">Platform Support Removals</span></a> for more details). The ssl module is mostly compatible with LibreSSL 2.7.2 and newer.

</div>

The <span class="pre">`ssl`</span> module no longer sends IP addresses in SNI TLS extension. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32185" class="reference external">bpo-32185</a>.)

<span class="pre">`match_hostname()`</span> no longer supports partial wildcards like <span class="pre">`www*.example.org`</span>. (Contributed by Mandeep Singh in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23033" class="reference external">bpo-23033</a> and Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31399" class="reference external">bpo-31399</a>.)

The default cipher suite selection of the <span class="pre">`ssl`</span> module now uses a blacklist approach rather than a hard-coded whitelist. Python no longer re-enables ciphers that have been blocked by OpenSSL security updates. Default cipher suite selection can be configured at compile time. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31429" class="reference external">bpo-31429</a>.)

Validation of server certificates containing internationalized domain names (IDNs) is now supported. As part of this change, the <a href="../library/ssl.html#ssl.SSLSocket.server_hostname" class="reference internal" title="ssl.SSLSocket.server_hostname"><span class="pre"><code class="sourceCode python">SSLSocket.server_hostname</code></span></a> attribute now stores the expected hostname in A-label form (<span class="pre">`"xn--pythn-mua.org"`</span>), rather than the U-label form (<span class="pre">`"pythön.org"`</span>). (Contributed by Nathaniel J. Smith and Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28414" class="reference external">bpo-28414</a>.)

The <span class="pre">`ssl`</span> module has preliminary and experimental support for TLS 1.3 and OpenSSL 1.1.1. At the time of Python 3.7.0 release, OpenSSL 1.1.1 is still under development and TLS 1.3 hasn’t been finalized yet. The TLS 1.3 handshake and protocol behaves slightly differently than TLS 1.2 and earlier, see <a href="../library/ssl.html#ssl-tlsv1-3" class="reference internal"><span class="std std-ref">TLS 1.3</span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32947" class="reference external">bpo-32947</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20995" class="reference external">bpo-20995</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29136" class="reference external">bpo-29136</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30622" class="reference external">bpo-30622</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33618" class="reference external">bpo-33618</a>)

<a href="../library/ssl.html#ssl.SSLSocket" class="reference internal" title="ssl.SSLSocket"><span class="pre"><code class="sourceCode python">SSLSocket</code></span></a> and <a href="../library/ssl.html#ssl.SSLObject" class="reference internal" title="ssl.SSLObject"><span class="pre"><code class="sourceCode python">SSLObject</code></span></a> no longer have a public constructor. Direct instantiation was never a documented and supported feature. Instances must be created with <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> methods <a href="../library/ssl.html#ssl.SSLContext.wrap_socket" class="reference internal" title="ssl.SSLContext.wrap_socket"><span class="pre"><code class="sourceCode python">wrap_socket()</code></span></a> and <a href="../library/ssl.html#ssl.SSLContext.wrap_bio" class="reference internal" title="ssl.SSLContext.wrap_bio"><span class="pre"><code class="sourceCode python">wrap_bio()</code></span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32951" class="reference external">bpo-32951</a>)

OpenSSL 1.1 APIs for setting the minimum and maximum TLS protocol version are available as <a href="../library/ssl.html#ssl.SSLContext.minimum_version" class="reference internal" title="ssl.SSLContext.minimum_version"><span class="pre"><code class="sourceCode python">SSLContext.minimum_version</code></span></a> and <a href="../library/ssl.html#ssl.SSLContext.maximum_version" class="reference internal" title="ssl.SSLContext.maximum_version"><span class="pre"><code class="sourceCode python">SSLContext.maximum_version</code></span></a>. Supported protocols are indicated by several new flags, such as <a href="../library/ssl.html#ssl.HAS_TLSv1_1" class="reference internal" title="ssl.HAS_TLSv1_1"><span class="pre"><code class="sourceCode python">HAS_TLSv1_1</code></span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32609" class="reference external">bpo-32609</a>.)

Added <a href="../library/ssl.html#ssl.SSLContext.post_handshake_auth" class="reference internal" title="ssl.SSLContext.post_handshake_auth"><span class="pre"><code class="sourceCode python">ssl.SSLContext.post_handshake_auth</code></span></a> to enable and <a href="../library/ssl.html#ssl.SSLSocket.verify_client_post_handshake" class="reference internal" title="ssl.SSLSocket.verify_client_post_handshake"><span class="pre"><code class="sourceCode python">ssl.SSLSocket.verify_client_post_handshake()</code></span></a> to initiate TLS 1.3 post-handshake authentication. (Contributed by Christian Heimes in <a href="https://github.com/python/cpython/issues/78851" class="reference external">gh-78851</a>.)

</div>

<div id="string" class="section">

### string<a href="#string" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/string.html#string.Template" class="reference internal" title="string.Template"><span class="pre"><code class="sourceCode python">string.Template</code></span></a> now lets you to optionally modify the regular expression pattern for braced placeholders and non-braced placeholders separately. (Contributed by Barry Warsaw in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1198569" class="reference external">bpo-1198569</a>.)

</div>

<div id="subprocess" class="section">

### subprocess<a href="#subprocess" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/subprocess.html#subprocess.run" class="reference internal" title="subprocess.run"><span class="pre"><code class="sourceCode python">subprocess.run()</code></span></a> function accepts the new *capture_output* keyword argument. When true, stdout and stderr will be captured. This is equivalent to passing <a href="../library/subprocess.html#subprocess.PIPE" class="reference internal" title="subprocess.PIPE"><span class="pre"><code class="sourceCode python">subprocess.PIPE</code></span></a> as *stdout* and *stderr* arguments. (Contributed by Bo Bayles in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32102" class="reference external">bpo-32102</a>.)

The <span class="pre">`subprocess.run`</span> function and the <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a> constructor now accept the *text* keyword argument as an alias to *universal_newlines*. (Contributed by Andrew Clegg in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31756" class="reference external">bpo-31756</a>.)

On Windows the default for *close_fds* was changed from <span class="pre">`False`</span> to <span class="pre">`True`</span> when redirecting the standard handles. It’s now possible to set *close_fds* to true when redirecting the standard handles. See <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a>. This means that *close_fds* now defaults to <span class="pre">`True`</span> on all supported platforms. (Contributed by Segev Finer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19764" class="reference external">bpo-19764</a>.)

The subprocess module is now more graceful when handling <a href="../library/exceptions.html#KeyboardInterrupt" class="reference internal" title="KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a> during <a href="../library/subprocess.html#subprocess.call" class="reference internal" title="subprocess.call"><span class="pre"><code class="sourceCode python">subprocess.call()</code></span></a>, <a href="../library/subprocess.html#subprocess.run" class="reference internal" title="subprocess.run"><span class="pre"><code class="sourceCode python">subprocess.run()</code></span></a>, or in a <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">Popen</code></span></a> context manager. It now waits a short amount of time for the child to exit, before continuing the handling of the <span class="pre">`KeyboardInterrupt`</span> exception. (Contributed by Gregory P. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25942" class="reference external">bpo-25942</a>.)

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/sys.html#sys.breakpointhook" class="reference internal" title="sys.breakpointhook"><span class="pre"><code class="sourceCode python">sys.breakpointhook()</code></span></a> hook function is called by the built-in <a href="../library/functions.html#breakpoint" class="reference internal" title="breakpoint"><span class="pre"><code class="sourceCode python"><span class="bu">breakpoint</span>()</code></span></a>. (Contributed by Barry Warsaw in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31353" class="reference external">bpo-31353</a>.)

On Android, the new <a href="../library/sys.html#sys.getandroidapilevel" class="reference internal" title="sys.getandroidapilevel"><span class="pre"><code class="sourceCode python">sys.getandroidapilevel()</code></span></a> returns the build-time Android API version. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28740" class="reference external">bpo-28740</a>.)

The new <a href="../library/sys.html#sys.get_coroutine_origin_tracking_depth" class="reference internal" title="sys.get_coroutine_origin_tracking_depth"><span class="pre"><code class="sourceCode python">sys.get_coroutine_origin_tracking_depth()</code></span></a> function returns the current coroutine origin tracking depth, as set by the new <a href="../library/sys.html#sys.set_coroutine_origin_tracking_depth" class="reference internal" title="sys.set_coroutine_origin_tracking_depth"><span class="pre"><code class="sourceCode python">sys.set_coroutine_origin_tracking_depth()</code></span></a>. <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> has been converted to use this new API instead of the deprecated <span class="pre">`sys.set_coroutine_wrapper()`</span>. (Contributed by Nathaniel J. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32591" class="reference external">bpo-32591</a>.)

</div>

<div id="time" class="section">

### time<a href="#time" class="headerlink" title="Link to this heading">¶</a>

<span id="index-33" class="target"></span><a href="https://peps.python.org/pep-0564/" class="pep reference external"><strong>PEP 564</strong></a> adds six new functions with nanosecond resolution to the <a href="../library/time.html#module-time" class="reference internal" title="time: Time access and conversions."><span class="pre"><code class="sourceCode python">time</code></span></a> module:

- <a href="../library/time.html#time.clock_gettime_ns" class="reference internal" title="time.clock_gettime_ns"><span class="pre"><code class="sourceCode python">time.clock_gettime_ns()</code></span></a>

- <a href="../library/time.html#time.clock_settime_ns" class="reference internal" title="time.clock_settime_ns"><span class="pre"><code class="sourceCode python">time.clock_settime_ns()</code></span></a>

- <a href="../library/time.html#time.monotonic_ns" class="reference internal" title="time.monotonic_ns"><span class="pre"><code class="sourceCode python">time.monotonic_ns()</code></span></a>

- <a href="../library/time.html#time.perf_counter_ns" class="reference internal" title="time.perf_counter_ns"><span class="pre"><code class="sourceCode python">time.perf_counter_ns()</code></span></a>

- <a href="../library/time.html#time.process_time_ns" class="reference internal" title="time.process_time_ns"><span class="pre"><code class="sourceCode python">time.process_time_ns()</code></span></a>

- <a href="../library/time.html#time.time_ns" class="reference internal" title="time.time_ns"><span class="pre"><code class="sourceCode python">time.time_ns()</code></span></a>

New clock identifiers have been added:

- <a href="../library/time.html#time.CLOCK_BOOTTIME" class="reference internal" title="time.CLOCK_BOOTTIME"><span class="pre"><code class="sourceCode python">time.CLOCK_BOOTTIME</code></span></a> (Linux): Identical to <a href="../library/time.html#time.CLOCK_MONOTONIC" class="reference internal" title="time.CLOCK_MONOTONIC"><span class="pre"><code class="sourceCode python">time.CLOCK_MONOTONIC</code></span></a>, except it also includes any time that the system is suspended.

- <a href="../library/time.html#time.CLOCK_PROF" class="reference internal" title="time.CLOCK_PROF"><span class="pre"><code class="sourceCode python">time.CLOCK_PROF</code></span></a> (FreeBSD, NetBSD and OpenBSD): High-resolution per-process CPU timer.

- <a href="../library/time.html#time.CLOCK_UPTIME" class="reference internal" title="time.CLOCK_UPTIME"><span class="pre"><code class="sourceCode python">time.CLOCK_UPTIME</code></span></a> (FreeBSD, OpenBSD): Time whose absolute value is the time the system has been running and not suspended, providing accurate uptime measurement.

The new <a href="../library/time.html#time.thread_time" class="reference internal" title="time.thread_time"><span class="pre"><code class="sourceCode python">time.thread_time()</code></span></a> and <a href="../library/time.html#time.thread_time_ns" class="reference internal" title="time.thread_time_ns"><span class="pre"><code class="sourceCode python">time.thread_time_ns()</code></span></a> functions can be used to get per-thread CPU time measurements. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32025" class="reference external">bpo-32025</a>.)

The new <a href="../library/time.html#time.pthread_getcpuclockid" class="reference internal" title="time.pthread_getcpuclockid"><span class="pre"><code class="sourceCode python">time.pthread_getcpuclockid()</code></span></a> function returns the clock ID of the thread-specific CPU-time clock.

</div>

<div id="tkinter" class="section">

### tkinter<a href="#tkinter" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/tkinter.ttk.html#tkinter.ttk.Spinbox" class="reference internal" title="tkinter.ttk.Spinbox"><span class="pre"><code class="sourceCode python">tkinter.ttk.Spinbox</code></span></a> class is now available. (Contributed by Alan Moore in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32585" class="reference external">bpo-32585</a>.)

</div>

<div id="tracemalloc" class="section">

### tracemalloc<a href="#tracemalloc" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/tracemalloc.html#tracemalloc.Traceback" class="reference internal" title="tracemalloc.Traceback"><span class="pre"><code class="sourceCode python">tracemalloc.Traceback</code></span></a> behaves more like regular tracebacks, sorting the frames from oldest to most recent. <a href="../library/tracemalloc.html#tracemalloc.Traceback.format" class="reference internal" title="tracemalloc.Traceback.format"><span class="pre"><code class="sourceCode python">Traceback.<span class="bu">format</span>()</code></span></a> now accepts negative *limit*, truncating the result to the <span class="pre">`abs(limit)`</span> oldest frames. To get the old behaviour, use the new *most_recent_first* argument to <span class="pre">`Traceback.format()`</span>. (Contributed by Jesse Bakker in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32121" class="reference external">bpo-32121</a>.)

</div>

<div id="types" class="section">

### types<a href="#types" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/types.html#types.WrapperDescriptorType" class="reference internal" title="types.WrapperDescriptorType"><span class="pre"><code class="sourceCode python">WrapperDescriptorType</code></span></a>, <a href="../library/types.html#types.MethodWrapperType" class="reference internal" title="types.MethodWrapperType"><span class="pre"><code class="sourceCode python">MethodWrapperType</code></span></a>, <a href="../library/types.html#types.MethodDescriptorType" class="reference internal" title="types.MethodDescriptorType"><span class="pre"><code class="sourceCode python">MethodDescriptorType</code></span></a>, and <a href="../library/types.html#types.ClassMethodDescriptorType" class="reference internal" title="types.ClassMethodDescriptorType"><span class="pre"><code class="sourceCode python">ClassMethodDescriptorType</code></span></a> classes are now available. (Contributed by Manuel Krebber and Guido van Rossum in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29377" class="reference external">bpo-29377</a>, and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32265" class="reference external">bpo-32265</a>.)

The new <a href="../library/types.html#types.resolve_bases" class="reference internal" title="types.resolve_bases"><span class="pre"><code class="sourceCode python">types.resolve_bases()</code></span></a> function resolves MRO entries dynamically as specified by <span id="index-34" class="target"></span><a href="https://peps.python.org/pep-0560/" class="pep reference external"><strong>PEP 560</strong></a>. (Contributed by Ivan Levkivskyi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32717" class="reference external">bpo-32717</a>.)

</div>

<div id="unicodedata" class="section">

### unicodedata<a href="#unicodedata" class="headerlink" title="Link to this heading">¶</a>

The internal <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a> database has been upgraded to use <a href="https://www.unicode.org/versions/Unicode11.0.0/" class="reference external">Unicode 11</a>. (Contributed by Benjamin Peterson.)

</div>

<div id="unittest" class="section">

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

The new <span class="pre">`-k`</span> command-line option allows filtering tests by a name substring or a Unix shell-like pattern. For example, <span class="pre">`python`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`unittest`</span>` `<span class="pre">`-k`</span>` `<span class="pre">`foo`</span> runs <span class="pre">`foo_tests.SomeTest.test_something`</span>, <span class="pre">`bar_tests.SomeTest.test_foo`</span>, but not <span class="pre">`bar_tests.FooTest.test_something`</span>. (Contributed by Jonas Haag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32071" class="reference external">bpo-32071</a>.)

</div>

<div id="unittest-mock" class="section">

### unittest.mock<a href="#unittest-mock" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/unittest.mock.html#unittest.mock.sentinel" class="reference internal" title="unittest.mock.sentinel"><span class="pre"><code class="sourceCode python">sentinel</code></span></a> attributes now preserve their identity when they are <a href="../library/copy.html#module-copy" class="reference internal" title="copy: Shallow and deep copy operations."><span class="pre"><code class="sourceCode python">copied</code></span></a> or <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickled</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20804" class="reference external">bpo-20804</a>.)

The new <a href="../library/unittest.mock.html#unittest.mock.seal" class="reference internal" title="unittest.mock.seal"><span class="pre"><code class="sourceCode python">seal()</code></span></a> function allows sealing <a href="../library/unittest.mock.html#unittest.mock.Mock" class="reference internal" title="unittest.mock.Mock"><span class="pre"><code class="sourceCode python">Mock</code></span></a> instances, which will disallow further creation of attribute mocks. The seal is applied recursively to all attributes that are themselves mocks. (Contributed by Mario Corchero in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30541" class="reference external">bpo-30541</a>.)

</div>

<div id="urllib-parse" class="section">

### urllib.parse<a href="#urllib-parse" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/urllib.parse.html#urllib.parse.quote" class="reference internal" title="urllib.parse.quote"><span class="pre"><code class="sourceCode python">urllib.parse.quote()</code></span></a> has been updated from <span id="index-35" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc2396.html" class="rfc reference external"><strong>RFC 2396</strong></a> to <span id="index-36" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc3986.html" class="rfc reference external"><strong>RFC 3986</strong></a>, adding <span class="pre">`~`</span> to the set of characters that are never quoted by default. (Contributed by Christian Theune and Ratnadeep Debnath in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16285" class="reference external">bpo-16285</a>.)

</div>

<div id="uu" class="section">

### uu<a href="#uu" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`uu.encode()`</span> function now accepts an optional *backtick* keyword argument. When it’s true, zeros are represented by <span class="pre">`` '`' ``</span> instead of spaces. (Contributed by Xiang Zhang in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30103" class="reference external">bpo-30103</a>.)

</div>

<div id="uuid" class="section">

### uuid<a href="#uuid" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/uuid.html#uuid.UUID.is_safe" class="reference internal" title="uuid.UUID.is_safe"><span class="pre"><code class="sourceCode python">UUID.is_safe</code></span></a> attribute relays information from the platform about whether generated UUIDs are generated with a multiprocessing-safe method. (Contributed by Barry Warsaw in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22807" class="reference external">bpo-22807</a>.)

<a href="../library/uuid.html#uuid.getnode" class="reference internal" title="uuid.getnode"><span class="pre"><code class="sourceCode python">uuid.getnode()</code></span></a> now prefers universally administered MAC addresses over locally administered MAC addresses. This makes a better guarantee for global uniqueness of UUIDs returned from <a href="../library/uuid.html#uuid.uuid1" class="reference internal" title="uuid.uuid1"><span class="pre"><code class="sourceCode python">uuid.uuid1()</code></span></a>. If only locally administered MAC addresses are available, the first such one found is returned. (Contributed by Barry Warsaw in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32107" class="reference external">bpo-32107</a>.)

</div>

<div id="warnings" class="section">

### warnings<a href="#warnings" class="headerlink" title="Link to this heading">¶</a>

The initialization of the default warnings filters has changed as follows:

- warnings enabled via command line options (including those for <a href="../using/cmdline.html#cmdoption-b" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-b</code></span></a> and the new CPython-specific <a href="../using/cmdline.html#cmdoption-X" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span></a> <span class="pre">`dev`</span> option) are always passed to the warnings machinery via the <a href="../library/sys.html#sys.warnoptions" class="reference internal" title="sys.warnoptions"><span class="pre"><code class="sourceCode python">sys.warnoptions</code></span></a> attribute.

- warnings filters enabled via the command line or the environment now have the following order of precedence:

  - the <span class="pre">`BytesWarning`</span> filter for <a href="../using/cmdline.html#cmdoption-b" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-b</code></span></a> (or <span class="pre">`-bb`</span>)

  - any filters specified with the <a href="../using/cmdline.html#cmdoption-W" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-W</code></span></a> option

  - any filters specified with the <span id="index-37" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONWARNINGS" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONWARNINGS</code></span></a> environment variable

  - any other CPython specific filters (e.g. the <span class="pre">`default`</span> filter added for the new <span class="pre">`-X`</span>` `<span class="pre">`dev`</span> mode)

  - any implicit filters defined directly by the warnings machinery

- in <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">CPython debug builds</span></a>, all warnings are now displayed by default (the implicit filter list is empty)

(Contributed by Nick Coghlan and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20361" class="reference external">bpo-20361</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32043" class="reference external">bpo-32043</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32230" class="reference external">bpo-32230</a>.)

Deprecation warnings are once again shown by default in single-file scripts and at the interactive prompt. See <a href="#whatsnew37-pep565" class="reference internal"><span class="std std-ref">PEP 565: Show DeprecationWarning in __main__</span></a> for details. (Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31975" class="reference external">bpo-31975</a>.)

</div>

<div id="xml" class="section">

### xml<a href="#xml" class="headerlink" title="Link to this heading">¶</a>

As mitigation against DTD and external entity retrieval, the <a href="../library/xml.dom.minidom.html#module-xml.dom.minidom" class="reference internal" title="xml.dom.minidom: Minimal Document Object Model (DOM) implementation."><span class="pre"><code class="sourceCode python">xml.dom.minidom</code></span></a> and <a href="../library/xml.sax.html#module-xml.sax" class="reference internal" title="xml.sax: Package containing SAX2 base classes and convenience functions."><span class="pre"><code class="sourceCode python">xml.sax</code></span></a> modules no longer process external entities by default. (Contributed by Christian Heimes in <a href="https://github.com/python/cpython/issues/61441" class="reference external">gh-61441</a>.)

</div>

<div id="xml-etree" class="section">

### xml.etree<a href="#xml-etree" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/xml.etree.elementtree.html#elementtree-xpath" class="reference internal"><span class="std std-ref">ElementPath</span></a> predicates in the <span class="pre">`find()`</span> methods can now compare text of the current node with <span class="pre">`[.`</span>` `<span class="pre">`=`</span>` `<span class="pre">`"text"]`</span>, not only text in children. Predicates also allow adding spaces for better readability. (Contributed by Stefan Behnel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31648" class="reference external">bpo-31648</a>.)

</div>

<div id="xmlrpc-server" class="section">

### xmlrpc.server<a href="#xmlrpc-server" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`SimpleXMLRPCDispatcher.register_function()`</span> can now be used as a decorator. (Contributed by Xiang Zhang in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7769" class="reference external">bpo-7769</a>.)

</div>

<div id="zipapp" class="section">

### zipapp<a href="#zipapp" class="headerlink" title="Link to this heading">¶</a>

Function <a href="../library/zipapp.html#zipapp.create_archive" class="reference internal" title="zipapp.create_archive"><span class="pre"><code class="sourceCode python">create_archive()</code></span></a> now accepts an optional *filter* argument to allow the user to select which files should be included in the archive. (Contributed by Irmen de Jong in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31072" class="reference external">bpo-31072</a>.)

Function <a href="../library/zipapp.html#zipapp.create_archive" class="reference internal" title="zipapp.create_archive"><span class="pre"><code class="sourceCode python">create_archive()</code></span></a> now accepts an optional *compressed* argument to generate a compressed archive. A command line option <span class="pre">`--compress`</span> has also been added to support compression. (Contributed by Zhiming Wang in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31638" class="reference external">bpo-31638</a>.)

</div>

<div id="zipfile" class="section">

### zipfile<a href="#zipfile" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/zipfile.html#zipfile.ZipFile" class="reference internal" title="zipfile.ZipFile"><span class="pre"><code class="sourceCode python">ZipFile</code></span></a> now accepts the new *compresslevel* parameter to control the compression level. (Contributed by Bo Bayles in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21417" class="reference external">bpo-21417</a>.)

Subdirectories in archives created by <span class="pre">`ZipFile`</span> are now stored in alphabetical order. (Contributed by Bernhard M. Wiedemann in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30693" class="reference external">bpo-30693</a>.)

</div>

</div>

<div id="c-api-changes" class="section">

## C API Changes<a href="#c-api-changes" class="headerlink" title="Link to this heading">¶</a>

A new API for thread-local storage has been implemented. See <a href="#whatsnew37-pep539" class="reference internal"><span class="std std-ref">PEP 539: New C API for Thread-Local Storage</span></a> for an overview and <a href="../c-api/init.html#thread-specific-storage-api" class="reference internal"><span class="std std-ref">Thread Specific Storage (TSS) API</span></a> for a complete reference. (Contributed by Masayuki Yamamoto in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25658" class="reference external">bpo-25658</a>.)

The new <a href="#whatsnew37-pep567" class="reference internal"><span class="std std-ref">context variables</span></a> functionality exposes a number of <a href="../c-api/contextvars.html#contextvarsobjects" class="reference internal"><span class="std std-ref">new C APIs</span></a>.

The new <a href="../c-api/import.html#c.PyImport_GetModule" class="reference internal" title="PyImport_GetModule"><span class="pre"><code class="sourceCode c">PyImport_GetModule<span class="op">()</span></code></span></a> function returns the previously imported module with the given name. (Contributed by Eric Snow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28411" class="reference external">bpo-28411</a>.)

The new <a href="../c-api/typeobj.html#c.Py_RETURN_RICHCOMPARE" class="reference internal" title="Py_RETURN_RICHCOMPARE"><span class="pre"><code class="sourceCode c">Py_RETURN_RICHCOMPARE</code></span></a> macro eases writing rich comparison functions. (Contributed by Petr Victorin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23699" class="reference external">bpo-23699</a>.)

The new <a href="../c-api/intro.html#c.Py_UNREACHABLE" class="reference internal" title="Py_UNREACHABLE"><span class="pre"><code class="sourceCode c">Py_UNREACHABLE</code></span></a> macro can be used to mark unreachable code paths. (Contributed by Barry Warsaw in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31338" class="reference external">bpo-31338</a>.)

The <a href="../library/tracemalloc.html#module-tracemalloc" class="reference internal" title="tracemalloc: Trace memory allocations."><span class="pre"><code class="sourceCode python">tracemalloc</code></span></a> now exposes a C API through the new <a href="../c-api/memory.html#c.PyTraceMalloc_Track" class="reference internal" title="PyTraceMalloc_Track"><span class="pre"><code class="sourceCode c">PyTraceMalloc_Track<span class="op">()</span></code></span></a> and <a href="../c-api/memory.html#c.PyTraceMalloc_Untrack" class="reference internal" title="PyTraceMalloc_Untrack"><span class="pre"><code class="sourceCode c">PyTraceMalloc_Untrack<span class="op">()</span></code></span></a> functions. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30054" class="reference external">bpo-30054</a>.)

The new <a href="../howto/instrumentation.html#static-markers" class="reference internal"><span class="std std-ref">import__find__load__start</span></a> and <a href="../howto/instrumentation.html#static-markers" class="reference internal"><span class="std std-ref">import__find__load__done</span></a> static markers can be used to trace module imports. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31574" class="reference external">bpo-31574</a>.)

The fields <span class="pre">`name`</span> and <span class="pre">`doc`</span> of structures <a href="../c-api/structures.html#c.PyMemberDef" class="reference internal" title="PyMemberDef"><span class="pre"><code class="sourceCode c">PyMemberDef</code></span></a>, <a href="../c-api/structures.html#c.PyGetSetDef" class="reference internal" title="PyGetSetDef"><span class="pre"><code class="sourceCode c">PyGetSetDef</code></span></a>, <a href="../c-api/tuple.html#c.PyStructSequence_Field" class="reference internal" title="PyStructSequence_Field"><span class="pre"><code class="sourceCode c">PyStructSequence_Field</code></span></a>, <a href="../c-api/tuple.html#c.PyStructSequence_Desc" class="reference internal" title="PyStructSequence_Desc"><span class="pre"><code class="sourceCode c">PyStructSequence_Desc</code></span></a>, and <span class="pre">`wrapperbase`</span> are now of type <span class="pre">`const`</span>` `<span class="pre">`char`</span>` `<span class="pre">`*`</span> rather of <span class="pre">`char`</span>` `<span class="pre">`*`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28761" class="reference external">bpo-28761</a>.)

The result of <a href="../c-api/unicode.html#c.PyUnicode_AsUTF8AndSize" class="reference internal" title="PyUnicode_AsUTF8AndSize"><span class="pre"><code class="sourceCode c">PyUnicode_AsUTF8AndSize<span class="op">()</span></code></span></a> and <a href="../c-api/unicode.html#c.PyUnicode_AsUTF8" class="reference internal" title="PyUnicode_AsUTF8"><span class="pre"><code class="sourceCode c">PyUnicode_AsUTF8<span class="op">()</span></code></span></a> is now of type <span class="pre">`const`</span>` `<span class="pre">`char`</span>` `<span class="pre">`*`</span> rather of <span class="pre">`char`</span>` `<span class="pre">`*`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28769" class="reference external">bpo-28769</a>.)

The result of <a href="../c-api/mapping.html#c.PyMapping_Keys" class="reference internal" title="PyMapping_Keys"><span class="pre"><code class="sourceCode c">PyMapping_Keys<span class="op">()</span></code></span></a>, <a href="../c-api/mapping.html#c.PyMapping_Values" class="reference internal" title="PyMapping_Values"><span class="pre"><code class="sourceCode c">PyMapping_Values<span class="op">()</span></code></span></a> and <a href="../c-api/mapping.html#c.PyMapping_Items" class="reference internal" title="PyMapping_Items"><span class="pre"><code class="sourceCode c">PyMapping_Items<span class="op">()</span></code></span></a> is now always a list, rather than a list or a tuple. (Contributed by Oren Milman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28280" class="reference external">bpo-28280</a>.)

Added functions <a href="../c-api/slice.html#c.PySlice_Unpack" class="reference internal" title="PySlice_Unpack"><span class="pre"><code class="sourceCode c">PySlice_Unpack<span class="op">()</span></code></span></a> and <a href="../c-api/slice.html#c.PySlice_AdjustIndices" class="reference internal" title="PySlice_AdjustIndices"><span class="pre"><code class="sourceCode c">PySlice_AdjustIndices<span class="op">()</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27867" class="reference external">bpo-27867</a>.)

<a href="../c-api/sys.html#c.PyOS_AfterFork" class="reference internal" title="PyOS_AfterFork"><span class="pre"><code class="sourceCode c">PyOS_AfterFork<span class="op">()</span></code></span></a> is deprecated in favour of the new functions <a href="../c-api/sys.html#c.PyOS_BeforeFork" class="reference internal" title="PyOS_BeforeFork"><span class="pre"><code class="sourceCode c">PyOS_BeforeFork<span class="op">()</span></code></span></a>, <a href="../c-api/sys.html#c.PyOS_AfterFork_Parent" class="reference internal" title="PyOS_AfterFork_Parent"><span class="pre"><code class="sourceCode c">PyOS_AfterFork_Parent<span class="op">()</span></code></span></a> and <a href="../c-api/sys.html#c.PyOS_AfterFork_Child" class="reference internal" title="PyOS_AfterFork_Child"><span class="pre"><code class="sourceCode c">PyOS_AfterFork_Child<span class="op">()</span></code></span></a>. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16500" class="reference external">bpo-16500</a>.)

The <span class="pre">`PyExc_RecursionErrorInst`</span> singleton that was part of the public API has been removed as its members being never cleared may cause a segfault during finalization of the interpreter. Contributed by Xavier de Gaye in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22898" class="reference external">bpo-22898</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30697" class="reference external">bpo-30697</a>.

Added C API support for timezones with timezone constructors <a href="../c-api/datetime.html#c.PyTimeZone_FromOffset" class="reference internal" title="PyTimeZone_FromOffset"><span class="pre"><code class="sourceCode c">PyTimeZone_FromOffset<span class="op">()</span></code></span></a> and <a href="../c-api/datetime.html#c.PyTimeZone_FromOffsetAndName" class="reference internal" title="PyTimeZone_FromOffsetAndName"><span class="pre"><code class="sourceCode c">PyTimeZone_FromOffsetAndName<span class="op">()</span></code></span></a>, and access to the UTC singleton with <a href="../c-api/datetime.html#c.PyDateTime_TimeZone_UTC" class="reference internal" title="PyDateTime_TimeZone_UTC"><span class="pre"><code class="sourceCode c">PyDateTime_TimeZone_UTC</code></span></a>. Contributed by Paul Ganssle in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10381" class="reference external">bpo-10381</a>.

The type of results of <span class="pre">`PyThread_start_new_thread()`</span> and <span class="pre">`PyThread_get_thread_ident()`</span>, and the *id* parameter of <a href="../c-api/init.html#c.PyThreadState_SetAsyncExc" class="reference internal" title="PyThreadState_SetAsyncExc"><span class="pre"><code class="sourceCode c">PyThreadState_SetAsyncExc<span class="op">()</span></code></span></a> changed from <span class="c-expr sig sig-inline c"><span class="kt">long</span></span> to <span class="c-expr sig sig-inline c"><span class="kt">unsigned</span><span class="w"> </span><span class="kt">long</span></span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6532" class="reference external">bpo-6532</a>.)

<a href="../c-api/unicode.html#c.PyUnicode_AsWideCharString" class="reference internal" title="PyUnicode_AsWideCharString"><span class="pre"><code class="sourceCode c">PyUnicode_AsWideCharString<span class="op">()</span></code></span></a> now raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the second argument is <span class="pre">`NULL`</span> and the <span class="c-expr sig sig-inline c"><span class="n">wchar_t</span><span class="p">\*</span></span> string contains null characters. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30708" class="reference external">bpo-30708</a>.)

Changes to the startup sequence and the management of dynamic memory allocators mean that the long documented requirement to call <a href="../c-api/init.html#c.Py_Initialize" class="reference internal" title="Py_Initialize"><span class="pre"><code class="sourceCode c">Py_Initialize<span class="op">()</span></code></span></a> before calling most C API functions is now relied on more heavily, and failing to abide by it may lead to segfaults in embedding applications. See the <a href="#porting-to-python-37" class="reference internal"><span class="std std-ref">Porting to Python 3.7</span></a> section in this document and the <a href="../c-api/init.html#pre-init-safe" class="reference internal"><span class="std std-ref">Before Python Initialization</span></a> section in the C API documentation for more details.

The new <a href="../c-api/init.html#c.PyInterpreterState_GetID" class="reference internal" title="PyInterpreterState_GetID"><span class="pre"><code class="sourceCode c">PyInterpreterState_GetID<span class="op">()</span></code></span></a> returns the unique ID for a given interpreter. (Contributed by Eric Snow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29102" class="reference external">bpo-29102</a>.)

<a href="../c-api/sys.html#c.Py_DecodeLocale" class="reference internal" title="Py_DecodeLocale"><span class="pre"><code class="sourceCode c">Py_DecodeLocale<span class="op">()</span></code></span></a>, <a href="../c-api/sys.html#c.Py_EncodeLocale" class="reference internal" title="Py_EncodeLocale"><span class="pre"><code class="sourceCode c">Py_EncodeLocale<span class="op">()</span></code></span></a> now use the UTF-8 encoding when the <a href="#whatsnew37-pep540" class="reference internal"><span class="std std-ref">UTF-8 mode</span></a> is enabled. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29240" class="reference external">bpo-29240</a>.)

<a href="../c-api/unicode.html#c.PyUnicode_DecodeLocaleAndSize" class="reference internal" title="PyUnicode_DecodeLocaleAndSize"><span class="pre"><code class="sourceCode c">PyUnicode_DecodeLocaleAndSize<span class="op">()</span></code></span></a> and <a href="../c-api/unicode.html#c.PyUnicode_EncodeLocale" class="reference internal" title="PyUnicode_EncodeLocale"><span class="pre"><code class="sourceCode c">PyUnicode_EncodeLocale<span class="op">()</span></code></span></a> now use the current locale encoding for <span class="pre">`surrogateescape`</span> error handler. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29240" class="reference external">bpo-29240</a>.)

The *start* and *end* parameters of <a href="../c-api/unicode.html#c.PyUnicode_FindChar" class="reference internal" title="PyUnicode_FindChar"><span class="pre"><code class="sourceCode c">PyUnicode_FindChar<span class="op">()</span></code></span></a> are now adjusted to behave like string slices. (Contributed by Xiang Zhang in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28822" class="reference external">bpo-28822</a>.)

</div>

<div id="build-changes" class="section">

## Build Changes<a href="#build-changes" class="headerlink" title="Link to this heading">¶</a>

Support for building <span class="pre">`--without-threads`</span> has been removed. The <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a> module is now always available. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31370" class="reference external">bpo-31370</a>.).

A full copy of libffi is no longer bundled for use when building the <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">_ctypes</code></span></a> module on non-OSX UNIX platforms. An installed copy of libffi is now required when building <span class="pre">`_ctypes`</span> on such platforms. (Contributed by Zachary Ware in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27979" class="reference external">bpo-27979</a>.)

The Windows build process no longer depends on Subversion to pull in external sources, a Python script is used to download zipfiles from GitHub instead. If Python 3.6 is not found on the system (via <span class="pre">`py`</span>` `<span class="pre">`-3.6`</span>), NuGet is used to download a copy of 32-bit Python for this purpose. (Contributed by Zachary Ware in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30450" class="reference external">bpo-30450</a>.)

The <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module requires OpenSSL 1.0.2 or 1.1 compatible libssl. OpenSSL 1.0.1 has reached end of lifetime on 2016-12-31 and is no longer supported. LibreSSL is temporarily not supported as well. LibreSSL releases up to version 2.6.4 are missing required OpenSSL 1.0.2 APIs.

</div>

<div id="optimizations" class="section">

<span id="whatsnew37-perf"></span>

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

The overhead of calling many methods of various standard library classes implemented in C has been significantly reduced by porting more code to use the <span class="pre">`METH_FASTCALL`</span> convention. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29300" class="reference external">bpo-29300</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29507" class="reference external">bpo-29507</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29452" class="reference external">bpo-29452</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29286" class="reference external">bpo-29286</a>.)

Various optimizations have reduced Python startup time by 10% on Linux and up to 30% on macOS. (Contributed by Victor Stinner, INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29585" class="reference external">bpo-29585</a>, and Ivan Levkivskyi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31333" class="reference external">bpo-31333</a>.)

Method calls are now up to 20% faster due to the bytecode changes which avoid creating bound method instances. (Contributed by Yury Selivanov and INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26110" class="reference external">bpo-26110</a>.)

The <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> module received a number of notable optimizations for commonly used functions:

- The <a href="../library/asyncio-eventloop.html#asyncio.get_event_loop" class="reference internal" title="asyncio.get_event_loop"><span class="pre"><code class="sourceCode python">asyncio.get_event_loop()</code></span></a> function has been reimplemented in C to make it up to 15 times faster. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32296" class="reference external">bpo-32296</a>.)

- <a href="../library/asyncio-future.html#asyncio.Future" class="reference internal" title="asyncio.Future"><span class="pre"><code class="sourceCode python">asyncio.Future</code></span></a> callback management has been optimized. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32348" class="reference external">bpo-32348</a>.)

- <a href="../library/asyncio-task.html#asyncio.gather" class="reference internal" title="asyncio.gather"><span class="pre"><code class="sourceCode python">asyncio.gather()</code></span></a> is now up to 15% faster. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32355" class="reference external">bpo-32355</a>.)

- <a href="../library/asyncio-task.html#asyncio.sleep" class="reference internal" title="asyncio.sleep"><span class="pre"><code class="sourceCode python">asyncio.sleep()</code></span></a> is now up to 2 times faster when the *delay* argument is zero or negative. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32351" class="reference external">bpo-32351</a>.)

- The performance overhead of asyncio debug mode has been reduced. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31970" class="reference external">bpo-31970</a>.)

As a result of <a href="#whatsnew37-pep560" class="reference internal"><span class="std std-ref">PEP 560 work</span></a>, the import time of <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> has been reduced by a factor of 7, and many typing operations are now faster. (Contributed by Ivan Levkivskyi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32226" class="reference external">bpo-32226</a>.)

<a href="../library/functions.html#sorted" class="reference internal" title="sorted"><span class="pre"><code class="sourceCode python"><span class="bu">sorted</span>()</code></span></a> and <a href="../library/stdtypes.html#list.sort" class="reference internal" title="list.sort"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>.sort()</code></span></a> have been optimized for common cases to be up to 40-75% faster. (Contributed by Elliot Gorokhovsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28685" class="reference external">bpo-28685</a>.)

<a href="../library/stdtypes.html#dict.copy" class="reference internal" title="dict.copy"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.copy()</code></span></a> is now up to 5.5 times faster. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31179" class="reference external">bpo-31179</a>.)

<a href="../library/functions.html#hasattr" class="reference internal" title="hasattr"><span class="pre"><code class="sourceCode python"><span class="bu">hasattr</span>()</code></span></a> and <a href="../library/functions.html#getattr" class="reference internal" title="getattr"><span class="pre"><code class="sourceCode python"><span class="bu">getattr</span>()</code></span></a> are now about 4 times faster when *name* is not found and *obj* does not override <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__getattr__</span>()</code></span></a> or <a href="../reference/datamodel.html#object.__getattribute__" class="reference internal" title="object.__getattribute__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__getattribute__</span>()</code></span></a>. (Contributed by INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32544" class="reference external">bpo-32544</a>.)

Searching for certain Unicode characters (like Ukrainian capital “Є”) in a string was up to 25 times slower than searching for other characters. It is now only 3 times slower in the worst case. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24821" class="reference external">bpo-24821</a>.)

The <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">collections.namedtuple()</code></span></a> factory has been reimplemented to make the creation of named tuples 4 to 6 times faster. (Contributed by Jelle Zijlstra with further improvements by INADA Naoki, Serhiy Storchaka, and Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28638" class="reference external">bpo-28638</a>.)

<a href="../library/datetime.html#datetime.date.fromordinal" class="reference internal" title="datetime.date.fromordinal"><span class="pre"><code class="sourceCode python">datetime.date.fromordinal()</code></span></a> and <a href="../library/datetime.html#datetime.date.fromtimestamp" class="reference internal" title="datetime.date.fromtimestamp"><span class="pre"><code class="sourceCode python">datetime.date.fromtimestamp()</code></span></a> are now up to 30% faster in the common case. (Contributed by Paul Ganssle in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32403" class="reference external">bpo-32403</a>.)

The <a href="../library/os.html#os.fwalk" class="reference internal" title="os.fwalk"><span class="pre"><code class="sourceCode python">os.fwalk()</code></span></a> function is now up to 2 times faster thanks to the use of <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25996" class="reference external">bpo-25996</a>.)

The speed of the <a href="../library/shutil.html#shutil.rmtree" class="reference internal" title="shutil.rmtree"><span class="pre"><code class="sourceCode python">shutil.rmtree()</code></span></a> function has been improved by 20–40% thanks to the use of the <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a> function. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28564" class="reference external">bpo-28564</a>.)

Optimized case-insensitive matching and searching of <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">regular</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">expressions</code></span></a>. Searching some patterns can now be up to 20 times faster. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30285" class="reference external">bpo-30285</a>.)

<a href="../library/re.html#re.compile" class="reference internal" title="re.compile"><span class="pre"><code class="sourceCode python">re.<span class="bu">compile</span>()</code></span></a> now converts <span class="pre">`flags`</span> parameter to int object if it is <span class="pre">`RegexFlag`</span>. It is now as fast as Python 3.5, and faster than Python 3.6 by about 10% depending on the pattern. (Contributed by INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31671" class="reference external">bpo-31671</a>.)

The <a href="../library/selectors.html#selectors.BaseSelector.modify" class="reference internal" title="selectors.BaseSelector.modify"><span class="pre"><code class="sourceCode python">modify()</code></span></a> methods of classes <a href="../library/selectors.html#selectors.EpollSelector" class="reference internal" title="selectors.EpollSelector"><span class="pre"><code class="sourceCode python">selectors.EpollSelector</code></span></a>, <a href="../library/selectors.html#selectors.PollSelector" class="reference internal" title="selectors.PollSelector"><span class="pre"><code class="sourceCode python">selectors.PollSelector</code></span></a> and <a href="../library/selectors.html#selectors.DevpollSelector" class="reference internal" title="selectors.DevpollSelector"><span class="pre"><code class="sourceCode python">selectors.DevpollSelector</code></span></a> may be around 10% faster under heavy loads. (Contributed by Giampaolo Rodola’ in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30014" class="reference external">bpo-30014</a>)

Constant folding has been moved from the peephole optimizer to the new AST optimizer, which is able perform optimizations more consistently. (Contributed by Eugene Toder and INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29469" class="reference external">bpo-29469</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11549" class="reference external">bpo-11549</a>.)

Most functions and methods in <a href="../library/abc.html#module-abc" class="reference internal" title="abc: Abstract base classes according to :pep:`3119`."><span class="pre"><code class="sourceCode python">abc</code></span></a> have been rewritten in C. This makes creation of abstract base classes, and calling <a href="../library/functions.html#isinstance" class="reference internal" title="isinstance"><span class="pre"><code class="sourceCode python"><span class="bu">isinstance</span>()</code></span></a> and <a href="../library/functions.html#issubclass" class="reference internal" title="issubclass"><span class="pre"><code class="sourceCode python"><span class="bu">issubclass</span>()</code></span></a> on them 1.5x faster. This also reduces Python start-up time by up to 10%. (Contributed by Ivan Levkivskyi and INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31333" class="reference external">bpo-31333</a>)

Significant speed improvements to alternate constructors for <a href="../library/datetime.html#datetime.date" class="reference internal" title="datetime.date"><span class="pre"><code class="sourceCode python">datetime.date</code></span></a> and <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime.datetime</code></span></a> by using fast-path constructors when not constructing subclasses. (Contributed by Paul Ganssle in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32403" class="reference external">bpo-32403</a>)

The speed of comparison of <a href="../library/array.html#array.array" class="reference internal" title="array.array"><span class="pre"><code class="sourceCode python">array.array</code></span></a> instances has been improved considerably in certain cases. It is now from 10x to 70x faster when comparing arrays holding values of the same integer type. (Contributed by Adrian Wielgosik in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24700" class="reference external">bpo-24700</a>.)

The <a href="../library/math.html#math.erf" class="reference internal" title="math.erf"><span class="pre"><code class="sourceCode python">math.erf()</code></span></a> and <a href="../library/math.html#math.erfc" class="reference internal" title="math.erfc"><span class="pre"><code class="sourceCode python">math.erfc()</code></span></a> functions now use the (faster) C library implementation on most platforms. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26121" class="reference external">bpo-26121</a>.)

</div>

<div id="other-cpython-implementation-changes" class="section">

## Other CPython Implementation Changes<a href="#other-cpython-implementation-changes" class="headerlink" title="Link to this heading">¶</a>

- Trace hooks may now opt out of receiving the <span class="pre">`line`</span> and opt into receiving the <span class="pre">`opcode`</span> events from the interpreter by setting the corresponding new <a href="../reference/datamodel.html#frame.f_trace_lines" class="reference internal" title="frame.f_trace_lines"><span class="pre"><code class="sourceCode python">f_trace_lines</code></span></a> and <a href="../reference/datamodel.html#frame.f_trace_opcodes" class="reference internal" title="frame.f_trace_opcodes"><span class="pre"><code class="sourceCode python">f_trace_opcodes</code></span></a> attributes on the frame being traced. (Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31344" class="reference external">bpo-31344</a>.)

- Fixed some consistency problems with namespace package module attributes. Namespace module objects now have an <span class="pre">`__file__`</span> that is set to <span class="pre">`None`</span> (previously unset), and their <span class="pre">`__spec__.origin`</span> is also set to <span class="pre">`None`</span> (previously the string <span class="pre">`"namespace"`</span>). See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32305" class="reference external">bpo-32305</a>. Also, the namespace module object’s <span class="pre">`__spec__.loader`</span> is set to the same value as <span class="pre">`__loader__`</span> (previously, the former was set to <span class="pre">`None`</span>). See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32303" class="reference external">bpo-32303</a>.

- The <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> dictionary now displays in the lexical order that variables were defined. Previously, the order was undefined. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32690" class="reference external">bpo-32690</a>.)

- The <span class="pre">`distutils`</span> <span class="pre">`upload`</span> command no longer tries to change CR end-of-line characters to CRLF. This fixes a corruption issue with sdists that ended with a byte equivalent to CR. (Contributed by Bo Bayles in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32304" class="reference external">bpo-32304</a>.)

</div>

<div id="deprecated-python-behavior" class="section">

## Deprecated Python Behavior<a href="#deprecated-python-behavior" class="headerlink" title="Link to this heading">¶</a>

Yield expressions (both <span class="pre">`yield`</span> and <span class="pre">`yield`</span>` `<span class="pre">`from`</span> clauses) are now deprecated in comprehensions and generator expressions (aside from the iterable expression in the leftmost <span class="pre">`for`</span> clause). This ensures that comprehensions always immediately return a container of the appropriate type (rather than potentially returning a <a href="../glossary.html#term-generator-iterator" class="reference internal"><span class="xref std std-term">generator iterator</span></a> object), while generator expressions won’t attempt to interleave their implicit output with the output from any explicit yield expressions. In Python 3.7, such expressions emit <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> when compiled, in Python 3.8 this will be a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10544" class="reference external">bpo-10544</a>.)

Returning a subclass of <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a> from <a href="../reference/datamodel.html#object.__complex__" class="reference internal" title="object.__complex__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__complex__</span>()</code></span></a> is deprecated and will be an error in future Python versions. This makes <span class="pre">`__complex__()`</span> consistent with <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__int__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__float__" class="reference internal" title="object.__float__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__float__</span>()</code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28894" class="reference external">bpo-28894</a>.)

</div>

<div id="deprecated-python-modules-functions-and-methods" class="section">

## Deprecated Python modules, functions and methods<a href="#deprecated-python-modules-functions-and-methods" class="headerlink" title="Link to this heading">¶</a>

<div id="aifc" class="section">

### aifc<a href="#aifc" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`aifc.openfp()`</span> has been deprecated and will be removed in Python 3.9. Use <span class="pre">`aifc.open()`</span> instead. (Contributed by Brian Curtin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31985" class="reference external">bpo-31985</a>.)

</div>

<div id="whatsnew37-asyncio-deprecated" class="section">

<span id="id2"></span>

### asyncio<a href="#whatsnew37-asyncio-deprecated" class="headerlink" title="Link to this heading">¶</a>

Support for directly <span class="pre">`await`</span>-ing instances of <a href="../library/asyncio-sync.html#asyncio.Lock" class="reference internal" title="asyncio.Lock"><span class="pre"><code class="sourceCode python">asyncio.Lock</code></span></a> and other asyncio synchronization primitives has been deprecated. An asynchronous context manager must be used in order to acquire and release the synchronization resource. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32253" class="reference external">bpo-32253</a>.)

The <span class="pre">`asyncio.Task.current_task()`</span> and <span class="pre">`asyncio.Task.all_tasks()`</span> methods have been deprecated. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32250" class="reference external">bpo-32250</a>.)

</div>

<div id="id3" class="section">

### collections<a href="#id3" class="headerlink" title="Link to this heading">¶</a>

In Python 3.8, the abstract base classes in <a href="../library/collections.abc.html#module-collections.abc" class="reference internal" title="collections.abc: Abstract base classes for containers"><span class="pre"><code class="sourceCode python">collections.abc</code></span></a> will no longer be exposed in the regular <a href="../library/collections.html#module-collections" class="reference internal" title="collections: Container datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module. This will help create a clearer distinction between the concrete classes and the abstract base classes. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25988" class="reference external">bpo-25988</a>.)

</div>

<div id="id4" class="section">

### dbm<a href="#id4" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/dbm.html#module-dbm.dumb" class="reference internal" title="dbm.dumb: Portable implementation of the simple DBM interface."><span class="pre"><code class="sourceCode python">dbm.dumb</code></span></a> now supports reading read-only files and no longer writes the index file when it is not changed. A deprecation warning is now emitted if the index file is missing and recreated in the <span class="pre">`'r'`</span> and <span class="pre">`'w'`</span> modes (this will be an error in future Python releases). (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28847" class="reference external">bpo-28847</a>.)

</div>

<div id="id5" class="section">

### enum<a href="#id5" class="headerlink" title="Link to this heading">¶</a>

In Python 3.8, attempting to check for non-Enum objects in <a href="../library/enum.html#enum.Enum" class="reference internal" title="enum.Enum"><span class="pre"><code class="sourceCode python">Enum</code></span></a> classes will raise a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> (e.g. <span class="pre">`1`</span>` `<span class="pre">`in`</span>` `<span class="pre">`Color`</span>); similarly, attempting to check for non-Flag objects in a <a href="../library/enum.html#enum.Flag" class="reference internal" title="enum.Flag"><span class="pre"><code class="sourceCode python">Flag</code></span></a> member will raise <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> (e.g. <span class="pre">`1`</span>` `<span class="pre">`in`</span>` `<span class="pre">`Perm.RW`</span>); currently, both operations return <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a> instead. (Contributed by Ethan Furman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33217" class="reference external">bpo-33217</a>.)

</div>

<div id="gettext" class="section">

### gettext<a href="#gettext" class="headerlink" title="Link to this heading">¶</a>

Using non-integer value for selecting a plural form in <a href="../library/gettext.html#module-gettext" class="reference internal" title="gettext: Multilingual internationalization services."><span class="pre"><code class="sourceCode python">gettext</code></span></a> is now deprecated. It never correctly worked. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28692" class="reference external">bpo-28692</a>.)

</div>

<div id="id6" class="section">

### importlib<a href="#id6" class="headerlink" title="Link to this heading">¶</a>

Methods <span class="pre">`MetaPathFinder.find_module()`</span> (replaced by <a href="../library/importlib.html#importlib.abc.MetaPathFinder.find_spec" class="reference internal" title="importlib.abc.MetaPathFinder.find_spec"><span class="pre"><code class="sourceCode python">MetaPathFinder.find_spec()</code></span></a>) and <span class="pre">`PathEntryFinder.find_loader()`</span> (replaced by <a href="../library/importlib.html#importlib.abc.PathEntryFinder.find_spec" class="reference internal" title="importlib.abc.PathEntryFinder.find_spec"><span class="pre"><code class="sourceCode python">PathEntryFinder.find_spec()</code></span></a>) both deprecated in Python 3.4 now emit <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. (Contributed by Matthias Bussonnier in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29576" class="reference external">bpo-29576</a>.)

The <a href="../library/importlib.html#importlib.abc.ResourceLoader" class="reference internal" title="importlib.abc.ResourceLoader"><span class="pre"><code class="sourceCode python">importlib.abc.ResourceLoader</code></span></a> ABC has been deprecated in favour of <a href="../library/importlib.html#importlib.abc.ResourceReader" class="reference internal" title="importlib.abc.ResourceReader"><span class="pre"><code class="sourceCode python">importlib.abc.ResourceReader</code></span></a>.

</div>

<div id="id7" class="section">

### locale<a href="#id7" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`locale.format()`</span> has been deprecated, use <a href="../library/locale.html#locale.format_string" class="reference internal" title="locale.format_string"><span class="pre"><code class="sourceCode python">locale.format_string()</code></span></a> instead. (Contributed by Garvit in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10379" class="reference external">bpo-10379</a>.)

</div>

<div id="macpath" class="section">

### macpath<a href="#macpath" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`macpath`</span> is now deprecated and will be removed in Python 3.8. (Contributed by Chi Hsuan Yen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9850" class="reference external">bpo-9850</a>.)

</div>

<div id="threading" class="section">

### threading<a href="#threading" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`dummy_threading`</span> and <span class="pre">`_dummy_thread`</span> have been deprecated. It is no longer possible to build Python with threading disabled. Use <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Thread-based parallelism."><span class="pre"><code class="sourceCode python">threading</code></span></a> instead. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31370" class="reference external">bpo-31370</a>.)

</div>

<div id="id8" class="section">

### socket<a href="#id8" class="headerlink" title="Link to this heading">¶</a>

The silent argument value truncation in <a href="../library/socket.html#socket.htons" class="reference internal" title="socket.htons"><span class="pre"><code class="sourceCode python">socket.htons()</code></span></a> and <a href="../library/socket.html#socket.ntohs" class="reference internal" title="socket.ntohs"><span class="pre"><code class="sourceCode python">socket.ntohs()</code></span></a> has been deprecated. In future versions of Python, if the passed argument is larger than 16 bits, an exception will be raised. (Contributed by Oren Milman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28332" class="reference external">bpo-28332</a>.)

</div>

<div id="id9" class="section">

### ssl<a href="#id9" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`ssl.wrap_socket()`</span> is deprecated. Use <a href="../library/ssl.html#ssl.SSLContext.wrap_socket" class="reference internal" title="ssl.SSLContext.wrap_socket"><span class="pre"><code class="sourceCode python">ssl.SSLContext.wrap_socket()</code></span></a> instead. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28124" class="reference external">bpo-28124</a>.)

</div>

<div id="sunau" class="section">

### sunau<a href="#sunau" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`sunau.openfp()`</span> has been deprecated and will be removed in Python 3.9. Use <span class="pre">`sunau.open()`</span> instead. (Contributed by Brian Curtin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31985" class="reference external">bpo-31985</a>.)

</div>

<div id="id10" class="section">

### sys<a href="#id10" class="headerlink" title="Link to this heading">¶</a>

Deprecated <span class="pre">`sys.set_coroutine_wrapper()`</span> and <span class="pre">`sys.get_coroutine_wrapper()`</span>.

The undocumented <span class="pre">`sys.callstats()`</span> function has been deprecated and will be removed in a future Python version. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28799" class="reference external">bpo-28799</a>.)

</div>

<div id="wave" class="section">

### wave<a href="#wave" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`wave.openfp()`</span> has been deprecated and will be removed in Python 3.9. Use <a href="../library/wave.html#wave.open" class="reference internal" title="wave.open"><span class="pre"><code class="sourceCode python">wave.<span class="bu">open</span>()</code></span></a> instead. (Contributed by Brian Curtin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31985" class="reference external">bpo-31985</a>.)

</div>

</div>

<div id="deprecated-functions-and-types-of-the-c-api" class="section">

## Deprecated functions and types of the C API<a href="#deprecated-functions-and-types-of-the-c-api" class="headerlink" title="Link to this heading">¶</a>

Function <a href="../c-api/slice.html#c.PySlice_GetIndicesEx" class="reference internal" title="PySlice_GetIndicesEx"><span class="pre"><code class="sourceCode c">PySlice_GetIndicesEx<span class="op">()</span></code></span></a> is deprecated and replaced with a macro if <span class="pre">`Py_LIMITED_API`</span> is not set or set to a value in the range between <span class="pre">`0x03050400`</span> and <span class="pre">`0x03060000`</span> (not inclusive), or is <span class="pre">`0x03060100`</span> or higher. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27867" class="reference external">bpo-27867</a>.)

<a href="../c-api/sys.html#c.PyOS_AfterFork" class="reference internal" title="PyOS_AfterFork"><span class="pre"><code class="sourceCode c">PyOS_AfterFork<span class="op">()</span></code></span></a> has been deprecated. Use <a href="../c-api/sys.html#c.PyOS_BeforeFork" class="reference internal" title="PyOS_BeforeFork"><span class="pre"><code class="sourceCode c">PyOS_BeforeFork<span class="op">()</span></code></span></a>, <a href="../c-api/sys.html#c.PyOS_AfterFork_Parent" class="reference internal" title="PyOS_AfterFork_Parent"><span class="pre"><code class="sourceCode c">PyOS_AfterFork_Parent<span class="op">()</span></code></span></a> or <a href="../c-api/sys.html#c.PyOS_AfterFork_Child" class="reference internal" title="PyOS_AfterFork_Child"><span class="pre"><code class="sourceCode c">PyOS_AfterFork_Child<span class="op">()</span></code></span></a> instead. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16500" class="reference external">bpo-16500</a>.)

</div>

<div id="platform-support-removals" class="section">

<span id="id11"></span>

## Platform Support Removals<a href="#platform-support-removals" class="headerlink" title="Link to this heading">¶</a>

- FreeBSD 9 and older are no longer officially supported.

- For full Unicode support, including within extension modules, \*nix platforms are now expected to provide at least one of <span class="pre">`C.UTF-8`</span> (full locale), <span class="pre">`C.utf8`</span> (full locale) or <span class="pre">`UTF-8`</span> (<span class="pre">`LC_CTYPE`</span>-only locale) as an alternative to the legacy <span class="pre">`ASCII`</span>-based <span class="pre">`C`</span> locale.

- OpenSSL 0.9.8 and 1.0.1 are no longer supported, which means building CPython 3.7 with SSL/TLS support on older platforms still using these versions requires custom build options that link to a more recent version of OpenSSL.

  Notably, this issue affects the Debian 8 (aka “jessie”) and Ubuntu 14.04 (aka “Trusty”) LTS Linux distributions, as they still use OpenSSL 1.0.1 by default.

  Debian 9 (“stretch”) and Ubuntu 16.04 (“xenial”), as well as recent releases of other LTS Linux releases (e.g. RHEL/CentOS 7.5, SLES 12-SP3), use OpenSSL 1.0.2 or later, and remain supported in the default build configuration.

  CPython’s own <a href="https://github.com/python/cpython/blob/v3.7.13/.travis.yml" class="reference external">CI configuration file</a> provides an example of using the SSL <a href="https://github.com/python/cpython/tree/3.13/Tools/ssl/multissltests.py" class="extlink-source reference external">compatibility testing infrastructure</a> in CPython’s test suite to build and link against OpenSSL 1.1.0 rather than an outdated system provided OpenSSL.

</div>

<div id="api-and-feature-removals" class="section">

## API and Feature Removals<a href="#api-and-feature-removals" class="headerlink" title="Link to this heading">¶</a>

The following features and APIs have been removed from Python 3.7:

- The <span class="pre">`os.stat_float_times()`</span> function has been removed. It was introduced in Python 2.3 for backward compatibility with Python 2.2, and was deprecated since Python 3.1.

- Unknown escapes consisting of <span class="pre">`'\'`</span> and an ASCII letter in replacement templates for <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">re.sub()</code></span></a> were deprecated in Python 3.5, and will now cause an error.

- Removed support of the *exclude* argument in <a href="../library/tarfile.html#tarfile.TarFile.add" class="reference internal" title="tarfile.TarFile.add"><span class="pre"><code class="sourceCode python">tarfile.TarFile.add()</code></span></a>. It was deprecated in Python 2.7 and 3.2. Use the *filter* argument instead.

- The <span class="pre">`ntpath.splitunc()`</span> function was deprecated in Python 3.1, and has now been removed. Use <a href="../library/os.path.html#os.path.splitdrive" class="reference internal" title="os.path.splitdrive"><span class="pre"><code class="sourceCode python">splitdrive()</code></span></a> instead.

- <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">collections.namedtuple()</code></span></a> no longer supports the *verbose* parameter or <span class="pre">`_source`</span> attribute which showed the generated source code for the named tuple class. This was part of an optimization designed to speed-up class creation. (Contributed by Jelle Zijlstra with further improvements by INADA Naoki, Serhiy Storchaka, and Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28638" class="reference external">bpo-28638</a>.)

- Functions <a href="../library/functions.html#bool" class="reference internal" title="bool"><span class="pre"><code class="sourceCode python"><span class="bu">bool</span>()</code></span></a>, <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span>()</code></span></a>, <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>()</code></span></a> and <a href="../library/stdtypes.html#tuple" class="reference internal" title="tuple"><span class="pre"><code class="sourceCode python"><span class="bu">tuple</span>()</code></span></a> no longer take keyword arguments. The first argument of <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a> can now be passed only as positional argument.

- Removed previously deprecated in Python 2.4 classes <span class="pre">`Plist`</span>, <span class="pre">`Dict`</span> and <span class="pre">`_InternalDict`</span> in the <a href="../library/plistlib.html#module-plistlib" class="reference internal" title="plistlib: Generate and parse Apple plist files."><span class="pre"><code class="sourceCode python">plistlib</code></span></a> module. Dict values in the result of functions <span class="pre">`readPlist()`</span> and <span class="pre">`readPlistFromBytes()`</span> are now normal dicts. You no longer can use attribute access to access items of these dictionaries.

- The <span class="pre">`asyncio.windows_utils.socketpair()`</span> function has been removed. Use the <a href="../library/socket.html#socket.socketpair" class="reference internal" title="socket.socketpair"><span class="pre"><code class="sourceCode python">socket.socketpair()</code></span></a> function instead, it is available on all platforms since Python 3.5. <span class="pre">`asyncio.windows_utils.socketpair`</span> was just an alias to <span class="pre">`socket.socketpair`</span> on Python 3.5 and newer.

- <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> no longer exports the <a href="../library/selectors.html#module-selectors" class="reference internal" title="selectors: High-level I/O multiplexing."><span class="pre"><code class="sourceCode python">selectors</code></span></a> and <span class="pre">`_overlapped`</span> modules as <span class="pre">`asyncio.selectors`</span> and <span class="pre">`asyncio._overlapped`</span>. Replace <span class="pre">`from`</span>` `<span class="pre">`asyncio`</span>` `<span class="pre">`import`</span>` `<span class="pre">`selectors`</span> with <span class="pre">`import`</span>` `<span class="pre">`selectors`</span>.

- Direct instantiation of <a href="../library/ssl.html#ssl.SSLSocket" class="reference internal" title="ssl.SSLSocket"><span class="pre"><code class="sourceCode python">ssl.SSLSocket</code></span></a> and <a href="../library/ssl.html#ssl.SSLObject" class="reference internal" title="ssl.SSLObject"><span class="pre"><code class="sourceCode python">ssl.SSLObject</code></span></a> objects is now prohibited. The constructors were never documented, tested, or designed as public constructors. Users were supposed to use <span class="pre">`ssl.wrap_socket()`</span> or <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32951" class="reference external">bpo-32951</a>.)

- The unused <span class="pre">`distutils`</span> <span class="pre">`install_misc`</span> command has been removed. (Contributed by Eric N. Vander Weele in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29218" class="reference external">bpo-29218</a>.)

</div>

<div id="module-removals" class="section">

## Module Removals<a href="#module-removals" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`fpectl`</span> module has been removed. It was never enabled by default, never worked correctly on x86-64, and it changed the Python ABI in ways that caused unexpected breakage of C extensions. (Contributed by Nathaniel J. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29137" class="reference external">bpo-29137</a>.)

</div>

<div id="windows-only-changes" class="section">

## Windows-only Changes<a href="#windows-only-changes" class="headerlink" title="Link to this heading">¶</a>

The python launcher, (py.exe), can accept 32 & 64 bit specifiers **without** having to specify a minor version as well. So <span class="pre">`py`</span>` `<span class="pre">`-3-32`</span> and <span class="pre">`py`</span>` `<span class="pre">`-3-64`</span> become valid as well as <span class="pre">`py`</span>` `<span class="pre">`-3.7-32`</span>, also the -*m*-64 and -*m.n*-64 forms are now accepted to force 64 bit python even if 32 bit would have otherwise been used. If the specified version is not available py.exe will error exit. (Contributed by Steve Barnes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30291" class="reference external">bpo-30291</a>.)

The launcher can be run as <span class="pre">`py`</span>` `<span class="pre">`-0`</span> to produce a list of the installed pythons, *with default marked with an asterisk*. Running <span class="pre">`py`</span>` `<span class="pre">`-0p`</span> will include the paths. If py is run with a version specifier that cannot be matched it will also print the *short form* list of available specifiers. (Contributed by Steve Barnes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30362" class="reference external">bpo-30362</a>.)

</div>

<div id="porting-to-python-3-7" class="section">

<span id="porting-to-python-37"></span>

## Porting to Python 3.7<a href="#porting-to-python-3-7" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="changes-in-python-behavior" class="section">

### Changes in Python Behavior<a href="#changes-in-python-behavior" class="headerlink" title="Link to this heading">¶</a>

- <a href="../reference/compound_stmts.html#async" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">async</code></span></a> and <a href="../reference/expressions.html#await" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">await</code></span></a> names are now reserved keywords. Code using these names as identifiers will now raise a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>. (Contributed by Jelle Zijlstra in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30406" class="reference external">bpo-30406</a>.)

- <span id="index-38" class="target"></span><a href="https://peps.python.org/pep-0479/" class="pep reference external"><strong>PEP 479</strong></a> is enabled for all code in Python 3.7, meaning that <a href="../library/exceptions.html#StopIteration" class="reference internal" title="StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a> exceptions raised directly or indirectly in coroutines and generators are transformed into <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> exceptions. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32670" class="reference external">bpo-32670</a>.)

- <a href="../reference/datamodel.html#object.__aiter__" class="reference internal" title="object.__aiter__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__aiter__</span>()</code></span></a> methods can no longer be declared as asynchronous. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31709" class="reference external">bpo-31709</a>.)

- Due to an oversight, earlier Python versions erroneously accepted the following syntax:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      f(1 for x in [1],)

      class C(1 for x in [1]):
          pass

  </div>

  </div>

  Python 3.7 now correctly raises a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>, as a generator expression always needs to be directly inside a set of parentheses and cannot have a comma on either side, and the duplication of the parentheses can be omitted only on calls. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32012" class="reference external">bpo-32012</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32023" class="reference external">bpo-32023</a>.)

- When using the <a href="../using/cmdline.html#cmdoption-m" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-m</code></span></a> switch, the initial working directory is now added to <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a>, rather than an empty string (which dynamically denoted the current working directory at the time of each import). Any programs that are checking for the empty string, or otherwise relying on the previous behaviour, will need to be updated accordingly (e.g. by also checking for <span class="pre">`os.getcwd()`</span> or <span class="pre">`os.path.dirname(__main__.__file__)`</span>, depending on why the code was checking for the empty string in the first place).

</div>

<div id="changes-in-the-python-api" class="section">

### Changes in the Python API<a href="#changes-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- <a href="../library/socketserver.html#socketserver.BaseServer.server_close" class="reference internal" title="socketserver.BaseServer.server_close"><span class="pre"><code class="sourceCode python">socketserver.ThreadingMixIn.server_close</code></span></a> now waits until all non-daemon threads complete. Set the new <a href="../library/socketserver.html#socketserver.ThreadingMixIn.block_on_close" class="reference internal" title="socketserver.ThreadingMixIn.block_on_close"><span class="pre"><code class="sourceCode python">socketserver.ThreadingMixIn.block_on_close</code></span></a> class attribute to <span class="pre">`False`</span> to get the pre-3.7 behaviour. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31233" class="reference external">bpo-31233</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33540" class="reference external">bpo-33540</a>.)

- <a href="../library/socketserver.html#socketserver.BaseServer.server_close" class="reference internal" title="socketserver.BaseServer.server_close"><span class="pre"><code class="sourceCode python">socketserver.ForkingMixIn.server_close</code></span></a> now waits until all child processes complete. Set the new <a href="../library/socketserver.html#socketserver.ThreadingMixIn.block_on_close" class="reference internal" title="socketserver.ThreadingMixIn.block_on_close"><span class="pre"><code class="sourceCode python">socketserver.ForkingMixIn.block_on_close</code></span></a> class attribute to <span class="pre">`False`</span> to get the pre-3.7 behaviour. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31151" class="reference external">bpo-31151</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33540" class="reference external">bpo-33540</a>.)

- The <a href="../library/locale.html#locale.localeconv" class="reference internal" title="locale.localeconv"><span class="pre"><code class="sourceCode python">locale.localeconv()</code></span></a> function now temporarily sets the <span class="pre">`LC_CTYPE`</span> locale to the value of <span class="pre">`LC_NUMERIC`</span> in some cases. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31900" class="reference external">bpo-31900</a>.)

- <a href="../library/pkgutil.html#pkgutil.walk_packages" class="reference internal" title="pkgutil.walk_packages"><span class="pre"><code class="sourceCode python">pkgutil.walk_packages()</code></span></a> now raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if *path* is a string. Previously an empty list was returned. (Contributed by Sanyam Khurana in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24744" class="reference external">bpo-24744</a>.)

- A format string argument for <a href="../library/string.html#string.Formatter.format" class="reference internal" title="string.Formatter.format"><span class="pre"><code class="sourceCode python">string.Formatter.<span class="bu">format</span>()</code></span></a> is now <a href="../glossary.html#positional-only-parameter" class="reference internal"><span class="std std-ref">positional-only</span></a>. Passing it as a keyword argument was deprecated in Python 3.5. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29193" class="reference external">bpo-29193</a>.)

- Attributes <a href="../library/http.cookies.html#http.cookies.Morsel.key" class="reference internal" title="http.cookies.Morsel.key"><span class="pre"><code class="sourceCode python">key</code></span></a>, <a href="../library/http.cookies.html#http.cookies.Morsel.value" class="reference internal" title="http.cookies.Morsel.value"><span class="pre"><code class="sourceCode python">value</code></span></a> and <a href="../library/http.cookies.html#http.cookies.Morsel.coded_value" class="reference internal" title="http.cookies.Morsel.coded_value"><span class="pre"><code class="sourceCode python">coded_value</code></span></a> of class <a href="../library/http.cookies.html#http.cookies.Morsel" class="reference internal" title="http.cookies.Morsel"><span class="pre"><code class="sourceCode python">http.cookies.Morsel</code></span></a> are now read-only. Assigning to them was deprecated in Python 3.5. Use the <a href="../library/http.cookies.html#http.cookies.Morsel.set" class="reference internal" title="http.cookies.Morsel.set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a> method for setting them. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29192" class="reference external">bpo-29192</a>.)

- The *mode* argument of <a href="../library/os.html#os.makedirs" class="reference internal" title="os.makedirs"><span class="pre"><code class="sourceCode python">os.makedirs()</code></span></a> no longer affects the file permission bits of newly created intermediate-level directories. To set their file permission bits you can set the umask before invoking <span class="pre">`makedirs()`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19930" class="reference external">bpo-19930</a>.)

- The <a href="../library/struct.html#struct.Struct.format" class="reference internal" title="struct.Struct.format"><span class="pre"><code class="sourceCode python">struct.Struct.<span class="bu">format</span></code></span></a> type is now <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> instead of <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21071" class="reference external">bpo-21071</a>.)

- <span class="pre">`cgi.parse_multipart()`</span> now accepts the *encoding* and *errors* arguments and returns the same results as <span class="pre">`FieldStorage`</span>: for non-file fields, the value associated to a key is a list of strings, not bytes. (Contributed by Pierre Quentel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29979" class="reference external">bpo-29979</a>.)

- Due to internal changes in <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a>, calling <a href="../library/socket.html#socket.fromshare" class="reference internal" title="socket.fromshare"><span class="pre"><code class="sourceCode python">socket.fromshare()</code></span></a> on a socket created by <a href="../library/socket.html#socket.socket.share" class="reference internal" title="socket.socket.share"><span class="pre"><code class="sourceCode python">socket.share</code></span></a> in older Python versions is not supported.

- <span class="pre">`repr`</span> for <a href="../library/exceptions.html#BaseException" class="reference internal" title="BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a> has changed to not include the trailing comma. Most exceptions are affected by this change. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30399" class="reference external">bpo-30399</a>.)

- <span class="pre">`repr`</span> for <a href="../library/datetime.html#datetime.timedelta" class="reference internal" title="datetime.timedelta"><span class="pre"><code class="sourceCode python">datetime.timedelta</code></span></a> has changed to include the keyword arguments in the output. (Contributed by Utkarsh Upadhyay in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30302" class="reference external">bpo-30302</a>.)

- Because <a href="../library/shutil.html#shutil.rmtree" class="reference internal" title="shutil.rmtree"><span class="pre"><code class="sourceCode python">shutil.rmtree()</code></span></a> is now implemented using the <a href="../library/os.html#os.scandir" class="reference internal" title="os.scandir"><span class="pre"><code class="sourceCode python">os.scandir()</code></span></a> function, the user specified handler *onerror* is now called with the first argument <span class="pre">`os.scandir`</span> instead of <span class="pre">`os.listdir`</span> when listing the directory is failed.

- Support for nested sets and set operations in regular expressions as in <a href="https://unicode.org/reports/tr18/" class="reference external">Unicode Technical Standard #18</a> might be added in the future. This would change the syntax. To facilitate this future change a <a href="../library/exceptions.html#FutureWarning" class="reference internal" title="FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a> will be raised in ambiguous cases for the time being. That include sets starting with a literal <span class="pre">`'['`</span> or containing literal character sequences <span class="pre">`'--'`</span>, <span class="pre">`'&&'`</span>, <span class="pre">`'~~'`</span>, and <span class="pre">`'||'`</span>. To avoid a warning, escape them with a backslash. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30349" class="reference external">bpo-30349</a>.)

- The result of splitting a string on a <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">regular</code></span><code class="sourceCode python"> </code><span class="pre"><code class="sourceCode python">expression</code></span></a> that could match an empty string has been changed. For example splitting on <span class="pre">`r'\s*'`</span> will now split not only on whitespaces as it did previously, but also on empty strings before all non-whitespace characters and just before the end of the string. The previous behavior can be restored by changing the pattern to <span class="pre">`r'\s+'`</span>. A <a href="../library/exceptions.html#FutureWarning" class="reference internal" title="FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a> was emitted for such patterns since Python 3.5.

  For patterns that match both empty and non-empty strings, the result of searching for all matches may also be changed in other cases. For example in the string <span class="pre">`'a\n\n'`</span>, the pattern <span class="pre">`r'(?m)^\s*?$'`</span> will not only match empty strings at positions 2 and 3, but also the string <span class="pre">`'\n'`</span> at positions 2–3. To match only blank lines, the pattern should be rewritten as <span class="pre">`r'(?m)^[^\S\n]*$'`</span>.

  <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">re.sub()</code></span></a> now replaces empty matches adjacent to a previous non-empty match. For example <span class="pre">`re.sub('x*',`</span>` `<span class="pre">`'-',`</span>` `<span class="pre">`'abxd')`</span> returns now <span class="pre">`'-a-b--d-'`</span> instead of <span class="pre">`'-a-b-d-'`</span> (the first minus between ‘b’ and ‘d’ replaces ‘x’, and the second minus replaces an empty string between ‘x’ and ‘d’).

  (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25054" class="reference external">bpo-25054</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32308" class="reference external">bpo-32308</a>.)

- Change <a href="../library/re.html#re.escape" class="reference internal" title="re.escape"><span class="pre"><code class="sourceCode python">re.escape()</code></span></a> to only escape regex special characters instead of escaping all characters other than ASCII letters, numbers, and <span class="pre">`'_'`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29995" class="reference external">bpo-29995</a>.)

- <a href="../library/tracemalloc.html#tracemalloc.Traceback" class="reference internal" title="tracemalloc.Traceback"><span class="pre"><code class="sourceCode python">tracemalloc.Traceback</code></span></a> frames are now sorted from oldest to most recent to be more consistent with <a href="../library/traceback.html#module-traceback" class="reference internal" title="traceback: Print or retrieve a stack traceback."><span class="pre"><code class="sourceCode python">traceback</code></span></a>. (Contributed by Jesse Bakker in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32121" class="reference external">bpo-32121</a>.)

- On OSes that support <a href="../library/socket.html#socket.SOCK_NONBLOCK" class="reference internal" title="socket.SOCK_NONBLOCK"><span class="pre"><code class="sourceCode python">socket.SOCK_NONBLOCK</code></span></a> or <a href="../library/socket.html#socket.SOCK_CLOEXEC" class="reference internal" title="socket.SOCK_CLOEXEC"><span class="pre"><code class="sourceCode python">socket.SOCK_CLOEXEC</code></span></a> bit flags, the <a href="../library/socket.html#socket.socket.type" class="reference internal" title="socket.socket.type"><span class="pre"><code class="sourceCode python">socket.<span class="bu">type</span></code></span></a> no longer has them applied. Therefore, checks like <span class="pre">`if`</span>` `<span class="pre">`sock.type`</span>` `<span class="pre">`==`</span>` `<span class="pre">`socket.SOCK_STREAM`</span> work as expected on all platforms. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32331" class="reference external">bpo-32331</a>.)

- On Windows the default for the *close_fds* argument of <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a> was changed from <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a> to <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> when redirecting the standard handles. If you previously depended on handles being inherited when using <a href="../library/subprocess.html#subprocess.Popen" class="reference internal" title="subprocess.Popen"><span class="pre"><code class="sourceCode python">subprocess.Popen</code></span></a> with standard io redirection, you will have to pass <span class="pre">`close_fds=False`</span> to preserve the previous behaviour, or use <a href="../library/subprocess.html#subprocess.STARTUPINFO.lpAttributeList" class="reference internal" title="subprocess.STARTUPINFO.lpAttributeList"><span class="pre"><code class="sourceCode python">STARTUPINFO.lpAttributeList</code></span></a>.

- <a href="../library/importlib.html#importlib.machinery.PathFinder.invalidate_caches" class="reference internal" title="importlib.machinery.PathFinder.invalidate_caches"><span class="pre"><code class="sourceCode python">importlib.machinery.PathFinder.invalidate_caches()</code></span></a> – which implicitly affects <a href="../library/importlib.html#importlib.invalidate_caches" class="reference internal" title="importlib.invalidate_caches"><span class="pre"><code class="sourceCode python">importlib.invalidate_caches()</code></span></a> – now deletes entries in <a href="../library/sys.html#sys.path_importer_cache" class="reference internal" title="sys.path_importer_cache"><span class="pre"><code class="sourceCode python">sys.path_importer_cache</code></span></a> which are set to <span class="pre">`None`</span>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33169" class="reference external">bpo-33169</a>.)

- In <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.sock_recv" class="reference internal" title="asyncio.loop.sock_recv"><span class="pre"><code class="sourceCode python">loop.sock_recv()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.sock_sendall" class="reference internal" title="asyncio.loop.sock_sendall"><span class="pre"><code class="sourceCode python">loop.sock_sendall()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.sock_accept" class="reference internal" title="asyncio.loop.sock_accept"><span class="pre"><code class="sourceCode python">loop.sock_accept()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.getaddrinfo" class="reference internal" title="asyncio.loop.getaddrinfo"><span class="pre"><code class="sourceCode python">loop.getaddrinfo()</code></span></a>, <a href="../library/asyncio-eventloop.html#asyncio.loop.getnameinfo" class="reference internal" title="asyncio.loop.getnameinfo"><span class="pre"><code class="sourceCode python">loop.getnameinfo()</code></span></a> have been changed to be proper coroutine methods to match their documentation. Previously, these methods returned <a href="../library/asyncio-future.html#asyncio.Future" class="reference internal" title="asyncio.Future"><span class="pre"><code class="sourceCode python">asyncio.Future</code></span></a> instances. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32327" class="reference external">bpo-32327</a>.)

- <a href="../library/asyncio-eventloop.html#asyncio.Server.sockets" class="reference internal" title="asyncio.Server.sockets"><span class="pre"><code class="sourceCode python">asyncio.Server.sockets</code></span></a> now returns a copy of the internal list of server sockets, instead of returning it directly. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32662" class="reference external">bpo-32662</a>.)

- <a href="../library/struct.html#struct.Struct.format" class="reference internal" title="struct.Struct.format"><span class="pre"><code class="sourceCode python">Struct.<span class="bu">format</span></code></span></a> is now a <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> instance instead of a <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> instance. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21071" class="reference external">bpo-21071</a>.)

- <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> subparsers can now be made mandatory by passing <span class="pre">`required=True`</span> to <a href="../library/argparse.html#argparse.ArgumentParser.add_subparsers" class="reference internal" title="argparse.ArgumentParser.add_subparsers"><span class="pre"><code class="sourceCode python">ArgumentParser.add_subparsers()</code></span></a>. (Contributed by Anthony Sottile in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26510" class="reference external">bpo-26510</a>.)

- <a href="../library/ast.html#ast.literal_eval" class="reference internal" title="ast.literal_eval"><span class="pre"><code class="sourceCode python">ast.literal_eval()</code></span></a> is now stricter. Addition and subtraction of arbitrary numbers are no longer allowed. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31778" class="reference external">bpo-31778</a>.)

- <a href="../library/calendar.html#calendar.Calendar.itermonthdates" class="reference internal" title="calendar.Calendar.itermonthdates"><span class="pre"><code class="sourceCode python">Calendar.itermonthdates</code></span></a> will now consistently raise an exception when a date falls outside of the <span class="pre">`0001-01-01`</span> through <span class="pre">`9999-12-31`</span> range. To support applications that cannot tolerate such exceptions, the new <a href="../library/calendar.html#calendar.Calendar.itermonthdays3" class="reference internal" title="calendar.Calendar.itermonthdays3"><span class="pre"><code class="sourceCode python">Calendar.itermonthdays3</code></span></a> and <a href="../library/calendar.html#calendar.Calendar.itermonthdays4" class="reference internal" title="calendar.Calendar.itermonthdays4"><span class="pre"><code class="sourceCode python">Calendar.itermonthdays4</code></span></a> can be used. The new methods return tuples and are not restricted by the range supported by <a href="../library/datetime.html#datetime.date" class="reference internal" title="datetime.date"><span class="pre"><code class="sourceCode python">datetime.date</code></span></a>. (Contributed by Alexander Belopolsky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28292" class="reference external">bpo-28292</a>.)

- <a href="../library/collections.html#collections.ChainMap" class="reference internal" title="collections.ChainMap"><span class="pre"><code class="sourceCode python">collections.ChainMap</code></span></a> now preserves the order of the underlying mappings. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32792" class="reference external">bpo-32792</a>.)

- The <span class="pre">`submit()`</span> method of <a href="../library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor" class="reference internal" title="concurrent.futures.ThreadPoolExecutor"><span class="pre"><code class="sourceCode python">concurrent.futures.ThreadPoolExecutor</code></span></a> and <a href="../library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor" class="reference internal" title="concurrent.futures.ProcessPoolExecutor"><span class="pre"><code class="sourceCode python">concurrent.futures.ProcessPoolExecutor</code></span></a> now raises a <a href="../library/exceptions.html#RuntimeError" class="reference internal" title="RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> if called during interpreter shutdown. (Contributed by Mark Nemec in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33097" class="reference external">bpo-33097</a>.)

- The <a href="../library/configparser.html#configparser.ConfigParser" class="reference internal" title="configparser.ConfigParser"><span class="pre"><code class="sourceCode python">configparser.ConfigParser</code></span></a> constructor now uses <span class="pre">`read_dict()`</span> to process the default values, making its behavior consistent with the rest of the parser. Non-string keys and values in the defaults dictionary are now being implicitly converted to strings. (Contributed by James Tocknell in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23835" class="reference external">bpo-23835</a>.)

- Several undocumented internal imports were removed. One example is that <span class="pre">`os.errno`</span> is no longer available; use <span class="pre">`import`</span>` `<span class="pre">`errno`</span> directly instead. Note that such undocumented internal imports may be removed any time without notice, even in micro version releases.

</div>

<div id="changes-in-the-c-api" class="section">

### Changes in the C API<a href="#changes-in-the-c-api" class="headerlink" title="Link to this heading">¶</a>

The function <a href="../c-api/slice.html#c.PySlice_GetIndicesEx" class="reference internal" title="PySlice_GetIndicesEx"><span class="pre"><code class="sourceCode c">PySlice_GetIndicesEx<span class="op">()</span></code></span></a> is considered unsafe for resizable sequences. If the slice indices are not instances of <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>, but objects that implement the <span class="pre">`__index__()`</span> method, the sequence can be resized after passing its length to <span class="pre">`PySlice_GetIndicesEx()`</span>. This can lead to returning indices out of the length of the sequence. For avoiding possible problems use new functions <a href="../c-api/slice.html#c.PySlice_Unpack" class="reference internal" title="PySlice_Unpack"><span class="pre"><code class="sourceCode c">PySlice_Unpack<span class="op">()</span></code></span></a> and <a href="../c-api/slice.html#c.PySlice_AdjustIndices" class="reference internal" title="PySlice_AdjustIndices"><span class="pre"><code class="sourceCode c">PySlice_AdjustIndices<span class="op">()</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27867" class="reference external">bpo-27867</a>.)

</div>

<div id="cpython-bytecode-changes" class="section">

### CPython bytecode changes<a href="#cpython-bytecode-changes" class="headerlink" title="Link to this heading">¶</a>

There are two new opcodes: <a href="../library/dis.html#opcode-LOAD_METHOD" class="reference internal"><span class="pre"><code class="xref std std-opcode docutils literal notranslate">LOAD_METHOD</code></span></a> and <span class="pre">`CALL_METHOD`</span>. (Contributed by Yury Selivanov and INADA Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26110" class="reference external">bpo-26110</a>.)

The <span class="pre">`STORE_ANNOTATION`</span> opcode has been removed. (Contributed by Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=32550" class="reference external">bpo-32550</a>.)

</div>

<div id="id12" class="section">

### Windows-only Changes<a href="#id12" class="headerlink" title="Link to this heading">¶</a>

The file used to override <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> is now called <span class="pre">`<python-executable>._pth`</span> instead of <span class="pre">`'sys.path'`</span>. See <a href="../using/windows.html#windows-finding-modules" class="reference internal"><span class="std std-ref">Finding modules</span></a> for more information. (Contributed by Steve Dower in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28137" class="reference external">bpo-28137</a>.)

</div>

<div id="id13" class="section">

### Other CPython implementation changes<a href="#id13" class="headerlink" title="Link to this heading">¶</a>

In preparation for potential future changes to the public CPython runtime initialization API (see <span id="index-39" class="target"></span><a href="https://peps.python.org/pep-0432/" class="pep reference external"><strong>PEP 432</strong></a> for an initial, but somewhat outdated, draft), CPython’s internal startup and configuration management logic has been significantly refactored. While these updates are intended to be entirely transparent to both embedding applications and users of the regular CPython CLI, they’re being mentioned here as the refactoring changes the internal order of various operations during interpreter startup, and hence may uncover previously latent defects, either in embedding applications, or in CPython itself. (Initially contributed by Nick Coghlan and Eric Snow as part of <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=22257" class="reference external">bpo-22257</a>, and further updated by Nick, Eric, and Victor Stinner in a number of other issues). Some known details affected:

- <span class="pre">`PySys_AddWarnOptionUnicode()`</span> is not currently usable by embedding applications due to the requirement to create a Unicode object prior to calling <span class="pre">`Py_Initialize`</span>. Use <span class="pre">`PySys_AddWarnOption()`</span> instead.

- warnings filters added by an embedding application with <span class="pre">`PySys_AddWarnOption()`</span> should now more consistently take precedence over the default filters set by the interpreter

Due to changes in the way the default warnings filters are configured, setting <a href="../c-api/init.html#c.Py_BytesWarningFlag" class="reference internal" title="Py_BytesWarningFlag"><span class="pre"><code class="sourceCode c">Py_BytesWarningFlag</code></span></a> to a value greater than one is no longer sufficient to both emit <a href="../library/exceptions.html#BytesWarning" class="reference internal" title="BytesWarning"><span class="pre"><code class="sourceCode python"><span class="pp">BytesWarning</span></code></span></a> messages and have them converted to exceptions. Instead, the flag must be set (to cause the warnings to be emitted in the first place), and an explicit <span class="pre">`error::BytesWarning`</span> warnings filter added to convert them to exceptions.

Due to a change in the way docstrings are handled by the compiler, the implicit <span class="pre">`return`</span>` `<span class="pre">`None`</span> in a function body consisting solely of a docstring is now marked as occurring on the same line as the docstring, not on the function’s header line.

The current exception state has been moved from the frame object to the co-routine. This simplified the interpreter and fixed a couple of obscure bugs caused by having swap exception state when entering or exiting a generator. (Contributed by Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=25612" class="reference external">bpo-25612</a>.)

</div>

</div>

<div id="notable-changes-in-python-3-7-1" class="section">

## Notable changes in Python 3.7.1<a href="#notable-changes-in-python-3-7-1" class="headerlink" title="Link to this heading">¶</a>

Starting in 3.7.1, <a href="../c-api/init.html#c.Py_Initialize" class="reference internal" title="Py_Initialize"><span class="pre"><code class="sourceCode c">Py_Initialize<span class="op">()</span></code></span></a> now consistently reads and respects all of the same environment settings as <a href="../c-api/init.html#c.Py_Main" class="reference internal" title="Py_Main"><span class="pre"><code class="sourceCode c">Py_Main<span class="op">()</span></code></span></a> (in earlier Python versions, it respected an ill-defined subset of those environment variables, while in Python 3.7.0 it didn’t read any of them due to <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34247" class="reference external">bpo-34247</a>). If this behavior is unwanted, set <a href="../c-api/init.html#c.Py_IgnoreEnvironmentFlag" class="reference internal" title="Py_IgnoreEnvironmentFlag"><span class="pre"><code class="sourceCode c">Py_IgnoreEnvironmentFlag</code></span></a> to 1 before calling <a href="../c-api/init.html#c.Py_Initialize" class="reference internal" title="Py_Initialize"><span class="pre"><code class="sourceCode c">Py_Initialize<span class="op">()</span></code></span></a>.

In 3.7.1 the C API for Context Variables <a href="../c-api/contextvars.html#contextvarsobjects-pointertype-change" class="reference internal"><span class="std std-ref">was updated</span></a> to use <a href="../c-api/structures.html#c.PyObject" class="reference internal" title="PyObject"><span class="pre"><code class="sourceCode c">PyObject</code></span></a> pointers. See also <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34762" class="reference external">bpo-34762</a>.

In 3.7.1 the <a href="../library/tokenize.html#module-tokenize" class="reference internal" title="tokenize: Lexical scanner for Python source code."><span class="pre"><code class="sourceCode python">tokenize</code></span></a> module now implicitly emits a <span class="pre">`NEWLINE`</span> token when provided with input that does not have a trailing new line. This behavior now matches what the C tokenizer does internally. (Contributed by Ammar Askar in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33899" class="reference external">bpo-33899</a>.)

</div>

<div id="notable-changes-in-python-3-7-2" class="section">

## Notable changes in Python 3.7.2<a href="#notable-changes-in-python-3-7-2" class="headerlink" title="Link to this heading">¶</a>

In 3.7.2, <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> on Windows no longer copies the original binaries, but creates redirector scripts named <span class="pre">`python.exe`</span> and <span class="pre">`pythonw.exe`</span> instead. This resolves a long standing issue where all virtual environments would have to be upgraded or recreated with each Python update. However, note that this release will still require recreation of virtual environments in order to get the new scripts.

</div>

<div id="notable-changes-in-python-3-7-6" class="section">

## Notable changes in Python 3.7.6<a href="#notable-changes-in-python-3-7-6" class="headerlink" title="Link to this heading">¶</a>

Due to significant security concerns, the *reuse_address* parameter of <a href="../library/asyncio-eventloop.html#asyncio.loop.create_datagram_endpoint" class="reference internal" title="asyncio.loop.create_datagram_endpoint"><span class="pre"><code class="sourceCode python">asyncio.loop.create_datagram_endpoint()</code></span></a> is no longer supported. This is because of the behavior of the socket option <span class="pre">`SO_REUSEADDR`</span> in UDP. For more details, see the documentation for <span class="pre">`loop.create_datagram_endpoint()`</span>. (Contributed by Kyle Stanley, Antoine Pitrou, and Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37228" class="reference external">bpo-37228</a>.)

</div>

<div id="notable-changes-in-python-3-7-10" class="section">

## Notable changes in Python 3.7.10<a href="#notable-changes-in-python-3-7-10" class="headerlink" title="Link to this heading">¶</a>

Earlier Python versions allowed using both <span class="pre">`;`</span> and <span class="pre">`&`</span> as query parameter separators in <a href="../library/urllib.parse.html#urllib.parse.parse_qs" class="reference internal" title="urllib.parse.parse_qs"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qs()</code></span></a> and <a href="../library/urllib.parse.html#urllib.parse.parse_qsl" class="reference internal" title="urllib.parse.parse_qsl"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qsl()</code></span></a>. Due to security concerns, and to conform with newer W3C recommendations, this has been changed to allow only a single separator key, with <span class="pre">`&`</span> as the default. This change also affects <span class="pre">`cgi.parse()`</span> and <span class="pre">`cgi.parse_multipart()`</span> as they use the affected functions internally. For more details, please see their respective documentation. (Contributed by Adam Goldschmidt, Senthil Kumaran and Ken Jin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42967" class="reference external">bpo-42967</a>.)

</div>

<div id="notable-changes-in-python-3-7-11" class="section">

## Notable changes in Python 3.7.11<a href="#notable-changes-in-python-3-7-11" class="headerlink" title="Link to this heading">¶</a>

A security fix alters the <a href="../library/ftplib.html#ftplib.FTP" class="reference internal" title="ftplib.FTP"><span class="pre"><code class="sourceCode python">ftplib.FTP</code></span></a> behavior to not trust the IPv4 address sent from the remote server when setting up a passive data channel. We reuse the ftp server IP address instead. For unusual code requiring the old behavior, set a <span class="pre">`trust_server_pasv_ipv4_address`</span> attribute on your FTP instance to <span class="pre">`True`</span>. (See <a href="https://github.com/python/cpython/issues/87451" class="reference external">gh-87451</a>)

The presence of newline or tab characters in parts of a URL allows for some forms of attacks. Following the WHATWG specification that updates RFC 3986, ASCII newline <span class="pre">`\n`</span>, <span class="pre">`\r`</span> and tab <span class="pre">`\t`</span> characters are stripped from the URL by the parser <a href="../library/urllib.parse.html#module-urllib.parse" class="reference internal" title="urllib.parse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urllib.parse()</code></span></a> preventing such attacks. The removal characters are controlled by a new module level variable <span class="pre">`urllib.parse._UNSAFE_URL_BYTES_TO_REMOVE`</span>. (See <a href="https://github.com/python/cpython/issues/88048" class="reference external">gh-88048</a>)

</div>

<div id="notable-security-feature-in-3-7-14" class="section">

## Notable security feature in 3.7.14<a href="#notable-security-feature-in-3-7-14" class="headerlink" title="Link to this heading">¶</a>

Converting between <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> and <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> in bases other than 2 (binary), 4, 8 (octal), 16 (hexadecimal), or 32 such as base 10 (decimal) now raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the number of digits in string form is above a limit to avoid potential denial of service attacks due to the algorithmic complexity. This is a mitigation for <span id="index-40" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2020-10735" class="cve reference external"><strong>CVE 2020-10735</strong></a>. This limit can be configured or disabled by environment variable, command line flag, or <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> APIs. See the <a href="../library/stdtypes.html#int-max-str-digits" class="reference internal"><span class="std std-ref">integer string conversion length limitation</span></a> documentation. The default limit is 4300 digits in string form.

</div>

</div>

<div class="clearer">

</div>

</div>
