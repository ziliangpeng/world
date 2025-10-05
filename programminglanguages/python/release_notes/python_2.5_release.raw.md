<div class="body" role="main">

<div id="what-s-new-in-python-2-5" class="section">

# What’s New in Python 2.5<a href="#what-s-new-in-python-2-5" class="headerlink" title="Permalink to this headline">¶</a>

Author  
A.M. Kuchling

This article explains the new features in Python 2.5. The final release of Python 2.5 is scheduled for August 2006; <span id="index-0" class="target"></span><a href="https://www.python.org/dev/peps/pep-0356" class="pep reference external"><strong>PEP 356</strong></a> describes the planned release schedule.

The changes in Python 2.5 are an interesting mix of language and library improvements. The library enhancements will be more important to Python’s user community, I think, because several widely-useful packages were added. New modules include ElementTree for XML processing (<span class="pre">`xml.etree`</span>), the SQLite database module (<span class="pre">`sqlite`</span>), and the <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> module for calling C functions.

The language changes are of middling significance. Some pleasant new features were added, but most of them aren’t features that you’ll use every day. Conditional expressions were finally added to the language using a novel syntax; see section <a href="#pep-308" class="reference internal"><span class="std std-ref">PEP 308: Conditional Expressions</span></a>. The new ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement will make writing cleanup code easier (section <a href="#pep-343" class="reference internal"><span class="std std-ref">PEP 343: The ‘with’ statement</span></a>). Values can now be passed into generators (section <a href="#pep-342" class="reference internal"><span class="std std-ref">PEP 342: New Generator Features</span></a>). Imports are now visible as either absolute or relative (section <a href="#pep-328" class="reference internal"><span class="std std-ref">PEP 328: Absolute and Relative Imports</span></a>). Some corner cases of exception handling are handled better (section <a href="#pep-341" class="reference internal"><span class="std std-ref">PEP 341: Unified try/except/finally</span></a>). All these improvements are worthwhile, but they’re improvements to one specific language feature or another; none of them are broad modifications to Python’s semantics.

As well as the language and library additions, other improvements and bugfixes were made throughout the source tree. A search through the SVN change logs finds there were 353 patches applied and 458 bugs fixed between Python 2.4 and 2.5. (Both figures are likely to be underestimates.)

This article doesn’t try to be a complete specification of the new features; instead changes are briefly introduced using helpful examples. For full details, you should always refer to the documentation for Python 2.5 at <a href="https://docs.python.org" class="reference external">https://docs.python.org</a>. If you want to understand the complete implementation and design rationale, refer to the PEP for a particular new feature.

Comments, suggestions, and error reports for this document are welcome; please e-mail them to the author or open a bug in the Python bug tracker.

<div id="pep-308-conditional-expressions" class="section">

<span id="pep-308"></span>

## PEP 308: Conditional Expressions<a href="#pep-308-conditional-expressions" class="headerlink" title="Permalink to this headline">¶</a>

For a long time, people have been requesting a way to write conditional expressions, which are expressions that return value A or value B depending on whether a Boolean value is true or false. A conditional expression lets you write a single assignment statement that has the same effect as the following:

<div class="highlight-default notranslate">

<div class="highlight">

    if condition:
        x = true_value
    else:
        x = false_value

</div>

</div>

There have been endless tedious discussions of syntax on both python-dev and comp.lang.python. A vote was even held that found the majority of voters wanted conditional expressions in some form, but there was no syntax that was preferred by a clear majority. Candidates included C’s <span class="pre">`cond`</span>` `<span class="pre">`?`</span>` `<span class="pre">`true_v`</span>` `<span class="pre">`:`</span>` `<span class="pre">`false_v`</span>, <span class="pre">`if`</span>` `<span class="pre">`cond`</span>` `<span class="pre">`then`</span>` `<span class="pre">`true_v`</span>` `<span class="pre">`else`</span>` `<span class="pre">`false_v`</span>, and 16 other variations.

Guido van Rossum eventually chose a surprising syntax:

<div class="highlight-default notranslate">

<div class="highlight">

    x = true_value if condition else false_value

</div>

</div>

Evaluation is still lazy as in existing Boolean expressions, so the order of evaluation jumps around a bit. The *condition* expression in the middle is evaluated first, and the *true_value* expression is evaluated only if the condition was true. Similarly, the *false_value* expression is only evaluated when the condition is false.

This syntax may seem strange and backwards; why does the condition go in the *middle* of the expression, and not in the front as in C’s <span class="pre">`c`</span>` `<span class="pre">`?`</span>` `<span class="pre">`x`</span>` `<span class="pre">`:`</span>` `<span class="pre">`y`</span>? The decision was checked by applying the new syntax to the modules in the standard library and seeing how the resulting code read. In many cases where a conditional expression is used, one value seems to be the ‘common case’ and one value is an ‘exceptional case’, used only on rarer occasions when the condition isn’t met. The conditional syntax makes this pattern a bit more obvious:

<div class="highlight-default notranslate">

<div class="highlight">

    contents = ((doc + '\n') if doc else '')

</div>

</div>

I read the above statement as meaning “here *contents* is usually assigned a value of <span class="pre">`doc+'\n'`</span>; sometimes *doc* is empty, in which special case an empty string is returned.” I doubt I will use conditional expressions very often where there isn’t a clear common and uncommon case.

There was some discussion of whether the language should require surrounding conditional expressions with parentheses. The decision was made to *not* require parentheses in the Python language’s grammar, but as a matter of style I think you should always use them. Consider these two statements:

<div class="highlight-default notranslate">

<div class="highlight">

    # First version -- no parens
    level = 1 if logging else 0

    # Second version -- with parens
    level = (1 if logging else 0)

</div>

</div>

In the first version, I think a reader’s eye might group the statement into ‘level = 1’, ‘if logging’, ‘else 0’, and think that the condition decides whether the assignment to *level* is performed. The second version reads better, in my opinion, because it makes it clear that the assignment is always performed and the choice is being made between two values.

Another reason for including the brackets: a few odd combinations of list comprehensions and lambdas could look like incorrect conditional expressions. See <span id="index-1" class="target"></span><a href="https://www.python.org/dev/peps/pep-0308" class="pep reference external"><strong>PEP 308</strong></a> for some examples. If you put parentheses around your conditional expressions, you won’t run into this case.

<div class="admonition seealso">

See also

<span id="index-2" class="target"></span><a href="https://www.python.org/dev/peps/pep-0308" class="pep reference external"><strong>PEP 308</strong></a> - Conditional Expressions  
PEP written by Guido van Rossum and Raymond D. Hettinger; implemented by Thomas Wouters.

</div>

</div>

<div id="pep-309-partial-function-application" class="section">

<span id="pep-309"></span>

## PEP 309: Partial Function Application<a href="#pep-309-partial-function-application" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/functools.html#module-functools" class="reference internal" title="functools: Higher-order functions and operations on callable objects."><span class="pre"><code class="sourceCode python">functools</code></span></a> module is intended to contain tools for functional-style programming.

One useful tool in this module is the <span class="pre">`partial()`</span> function. For programs written in a functional style, you’ll sometimes want to construct variants of existing functions that have some of the parameters filled in. Consider a Python function <span class="pre">`f(a,`</span>` `<span class="pre">`b,`</span>` `<span class="pre">`c)`</span>; you could create a new function <span class="pre">`g(b,`</span>` `<span class="pre">`c)`</span> that was equivalent to <span class="pre">`f(1,`</span>` `<span class="pre">`b,`</span>` `<span class="pre">`c)`</span>. This is called “partial function application”.

<span class="pre">`partial()`</span> takes the arguments <span class="pre">`(function,`</span>` `<span class="pre">`arg1,`</span>` `<span class="pre">`arg2,`</span>` `<span class="pre">`...`</span>` `<span class="pre">`kwarg1=value1,`</span>` `<span class="pre">`kwarg2=value2)`</span>. The resulting object is callable, so you can just call it to invoke *function* with the filled-in arguments.

Here’s a small but realistic example:

<div class="highlight-default notranslate">

<div class="highlight">

    import functools

    def log (message, subsystem):
        "Write the contents of 'message' to the specified subsystem."
        print '%s: %s' % (subsystem, message)
        ...

    server_log = functools.partial(log, subsystem='server')
    server_log('Unable to open socket')

</div>

</div>

Here’s another example, from a program that uses PyGTK. Here a context-sensitive pop-up menu is being constructed dynamically. The callback provided for the menu option is a partially applied version of the <span class="pre">`open_item()`</span> method, where the first argument has been provided.

<div class="highlight-default notranslate">

<div class="highlight">

    ...
    class Application:
        def open_item(self, path):
           ...
        def init (self):
            open_func = functools.partial(self.open_item, item_path)
            popup_menu.append( ("Open", open_func, 1) )

</div>

</div>

Another function in the <a href="../library/functools.html#module-functools" class="reference internal" title="functools: Higher-order functions and operations on callable objects."><span class="pre"><code class="sourceCode python">functools</code></span></a> module is the <span class="pre">`update_wrapper(wrapper,`</span>` `<span class="pre">`wrapped)`</span> function that helps you write well-behaved decorators. <span class="pre">`update_wrapper()`</span> copies the name, module, and docstring attribute to a wrapper function so that tracebacks inside the wrapped function are easier to understand. For example, you might write:

<div class="highlight-default notranslate">

<div class="highlight">

    def my_decorator(f):
        def wrapper(*args, **kwds):
            print 'Calling decorated function'
            return f(*args, **kwds)
        functools.update_wrapper(wrapper, f)
        return wrapper

</div>

</div>

<span class="pre">`wraps()`</span> is a decorator that can be used inside your own decorators to copy the wrapped function’s information. An alternate version of the previous example would be:

<div class="highlight-default notranslate">

<div class="highlight">

    def my_decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwds):
            print 'Calling decorated function'
            return f(*args, **kwds)
        return wrapper

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-3" class="target"></span><a href="https://www.python.org/dev/peps/pep-0309" class="pep reference external"><strong>PEP 309</strong></a> - Partial Function Application  
PEP proposed and written by Peter Harris; implemented by Hye-Shik Chang and Nick Coghlan, with adaptations by Raymond Hettinger.

</div>

</div>

<div id="pep-314-metadata-for-python-software-packages-v1-1" class="section">

<span id="pep-314"></span>

## PEP 314: Metadata for Python Software Packages v1.1<a href="#pep-314-metadata-for-python-software-packages-v1-1" class="headerlink" title="Permalink to this headline">¶</a>

Some simple dependency support was added to Distutils. The <span class="pre">`setup()`</span> function now has <span class="pre">`requires`</span>, <span class="pre">`provides`</span>, and <span class="pre">`obsoletes`</span> keyword parameters. When you build a source distribution using the <span class="pre">`sdist`</span> command, the dependency information will be recorded in the <span class="pre">`PKG-INFO`</span> file.

Another new keyword parameter is <span class="pre">`download_url`</span>, which should be set to a URL for the package’s source code. This means it’s now possible to look up an entry in the package index, determine the dependencies for a package, and download the required packages.

<div class="highlight-default notranslate">

<div class="highlight">

    VERSION = '1.0'
    setup(name='PyPackage',
          version=VERSION,
          requires=['numarray', 'zlib (>=1.1.4)'],
          obsoletes=['OldPackage']
          download_url=('http://www.example.com/pypackage/dist/pkg-%s.tar.gz'
                        % VERSION),
         )

</div>

</div>

Another new enhancement to the Python package index at <a href="https://pypi.org" class="reference external">https://pypi.org</a> is storing source and binary archives for a package. The new **upload** Distutils command will upload a package to the repository.

Before a package can be uploaded, you must be able to build a distribution using the **sdist** Distutils command. Once that works, you can run <span class="pre">`python`</span>` `<span class="pre">`setup.py`</span>` `<span class="pre">`upload`</span> to add your package to the PyPI archive. Optionally you can GPG-sign the package by supplying the <span class="pre">`--sign`</span> and <span class="pre">`--identity`</span> options.

Package uploading was implemented by Martin von Löwis and Richard Jones.

<div class="admonition seealso">

See also

<span id="index-4" class="target"></span><a href="https://www.python.org/dev/peps/pep-0314" class="pep reference external"><strong>PEP 314</strong></a> - Metadata for Python Software Packages v1.1  
PEP proposed and written by A.M. Kuchling, Richard Jones, and Fred Drake; implemented by Richard Jones and Fred Drake.

</div>

</div>

<div id="pep-328-absolute-and-relative-imports" class="section">

<span id="pep-328"></span>

## PEP 328: Absolute and Relative Imports<a href="#pep-328-absolute-and-relative-imports" class="headerlink" title="Permalink to this headline">¶</a>

The simpler part of PEP 328 was implemented in Python 2.4: parentheses could now be used to enclose the names imported from a module using the <span class="pre">`from`</span>` `<span class="pre">`...`</span>` `<span class="pre">`import`</span>` `<span class="pre">`...`</span> statement, making it easier to import many different names.

The more complicated part has been implemented in Python 2.5: importing a module can be specified to use absolute or package-relative imports. The plan is to move toward making absolute imports the default in future versions of Python.

Let’s say you have a package directory like this:

<div class="highlight-default notranslate">

<div class="highlight">

    pkg/
    pkg/__init__.py
    pkg/main.py
    pkg/string.py

</div>

</div>

This defines a package named <span class="pre">`pkg`</span> containing the <span class="pre">`pkg.main`</span> and <span class="pre">`pkg.string`</span> submodules.

Consider the code in the <span class="pre">`main.py`</span> module. What happens if it executes the statement <span class="pre">`import`</span>` `<span class="pre">`string`</span>? In Python 2.4 and earlier, it will first look in the package’s directory to perform a relative import, finds <span class="pre">`pkg/string.py`</span>, imports the contents of that file as the <span class="pre">`pkg.string`</span> module, and that module is bound to the name <span class="pre">`string`</span> in the <span class="pre">`pkg.main`</span> module’s namespace.

That’s fine if <span class="pre">`pkg.string`</span> was what you wanted. But what if you wanted Python’s standard <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module? There’s no clean way to ignore <span class="pre">`pkg.string`</span> and look for the standard module; generally you had to look at the contents of <span class="pre">`sys.modules`</span>, which is slightly unclean. Holger Krekel’s <span class="pre">`py.std`</span> package provides a tidier way to perform imports from the standard library, <span class="pre">`import`</span>` `<span class="pre">`py;`</span>` `<span class="pre">`py.std.string.join()`</span>, but that package isn’t available on all Python installations.

Reading code which relies on relative imports is also less clear, because a reader may be confused about which module, <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> or <span class="pre">`pkg.string`</span>, is intended to be used. Python users soon learned not to duplicate the names of standard library modules in the names of their packages’ submodules, but you can’t protect against having your submodule’s name being used for a new module added in a future version of Python.

In Python 2.5, you can switch <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a>’s behaviour to absolute imports using a <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`absolute_import`</span> directive. This absolute-import behaviour will become the default in a future version (probably Python 2.7). Once absolute imports are the default, <span class="pre">`import`</span>` `<span class="pre">`string`</span> will always find the standard library’s version. It’s suggested that users should begin using absolute imports as much as possible, so it’s preferable to begin writing <span class="pre">`from`</span>` `<span class="pre">`pkg`</span>` `<span class="pre">`import`</span>` `<span class="pre">`string`</span> in your code.

Relative imports are still possible by adding a leading period to the module name when using the <span class="pre">`from`</span>` `<span class="pre">`...`</span>` `<span class="pre">`import`</span> form:

<div class="highlight-default notranslate">

<div class="highlight">

    # Import names from pkg.string
    from .string import name1, name2
    # Import pkg.string
    from . import string

</div>

</div>

This imports the <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module relative to the current package, so in <span class="pre">`pkg.main`</span> this will import *name1* and *name2* from <span class="pre">`pkg.string`</span>. Additional leading periods perform the relative import starting from the parent of the current package. For example, code in the <span class="pre">`A.B.C`</span> module can do:

<div class="highlight-default notranslate">

<div class="highlight">

    from . import D                 # Imports A.B.D
    from .. import E                # Imports A.E
    from ..F import G               # Imports A.F.G

</div>

</div>

Leading periods cannot be used with the <span class="pre">`import`</span>` `<span class="pre">`modname`</span> form of the import statement, only the <span class="pre">`from`</span>` `<span class="pre">`...`</span>` `<span class="pre">`import`</span> form.

<div class="admonition seealso">

See also

<span id="index-5" class="target"></span><a href="https://www.python.org/dev/peps/pep-0328" class="pep reference external"><strong>PEP 328</strong></a> - Imports: Multi-Line and Absolute/Relative  
PEP written by Aahz; implemented by Thomas Wouters.

<a href="https://pylib.readthedocs.org/" class="reference external">https://pylib.readthedocs.org/</a>  
The py library by Holger Krekel, which contains the <span class="pre">`py.std`</span> package.

</div>

</div>

<div id="pep-338-executing-modules-as-scripts" class="section">

<span id="pep-338"></span>

## PEP 338: Executing Modules as Scripts<a href="#pep-338-executing-modules-as-scripts" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../using/cmdline.html#cmdoption-m" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-m</code></span></a> switch added in Python 2.4 to execute a module as a script gained a few more abilities. Instead of being implemented in C code inside the Python interpreter, the switch now uses an implementation in a new module, <a href="../library/runpy.html#module-runpy" class="reference internal" title="runpy: Locate and run Python modules without importing them first."><span class="pre"><code class="sourceCode python">runpy</code></span></a>.

The <a href="../library/runpy.html#module-runpy" class="reference internal" title="runpy: Locate and run Python modules without importing them first."><span class="pre"><code class="sourceCode python">runpy</code></span></a> module implements a more sophisticated import mechanism so that it’s now possible to run modules in a package such as <span class="pre">`pychecker.checker`</span>. The module also supports alternative import mechanisms such as the <a href="../library/zipimport.html#module-zipimport" class="reference internal" title="zipimport: support for importing Python modules from ZIP archives."><span class="pre"><code class="sourceCode python">zipimport</code></span></a> module. This means you can add a .zip archive’s path to <span class="pre">`sys.path`</span> and then use the <a href="../using/cmdline.html#cmdoption-m" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-m</code></span></a> switch to execute code from the archive.

<div class="admonition seealso">

See also

<span id="index-6" class="target"></span><a href="https://www.python.org/dev/peps/pep-0338" class="pep reference external"><strong>PEP 338</strong></a> - Executing modules as scripts  
PEP written and implemented by Nick Coghlan.

</div>

</div>

<div id="pep-341-unified-try-except-finally" class="section">

<span id="pep-341"></span>

## PEP 341: Unified try/except/finally<a href="#pep-341-unified-try-except-finally" class="headerlink" title="Permalink to this headline">¶</a>

Until Python 2.5, the <a href="../reference/compound_stmts.html#try" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">try</code></span></a> statement came in two flavours. You could use a <a href="../reference/compound_stmts.html#finally" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">finally</code></span></a> block to ensure that code is always executed, or one or more <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> blocks to catch specific exceptions. You couldn’t combine both <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> blocks and a <a href="../reference/compound_stmts.html#finally" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">finally</code></span></a> block, because generating the right bytecode for the combined version was complicated and it wasn’t clear what the semantics of the combined statement should be.

Guido van Rossum spent some time working with Java, which does support the equivalent of combining <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> blocks and a <a href="../reference/compound_stmts.html#finally" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">finally</code></span></a> block, and this clarified what the statement should mean. In Python 2.5, you can now write:

<div class="highlight-default notranslate">

<div class="highlight">

    try:
        block-1 ...
    except Exception1:
        handler-1 ...
    except Exception2:
        handler-2 ...
    else:
        else-block
    finally:
        final-block

</div>

</div>

The code in *block-1* is executed. If the code raises an exception, the various <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> blocks are tested: if the exception is of class <span class="pre">`Exception1`</span>, *handler-1* is executed; otherwise if it’s of class <span class="pre">`Exception2`</span>, *handler-2* is executed, and so forth. If no exception is raised, the *else-block* is executed.

No matter what happened previously, the *final-block* is executed once the code block is complete and any raised exceptions handled. Even if there’s an error in an exception handler or the *else-block* and a new exception is raised, the code in the *final-block* is still run.

<div class="admonition seealso">

See also

<span id="index-7" class="target"></span><a href="https://www.python.org/dev/peps/pep-0341" class="pep reference external"><strong>PEP 341</strong></a> - Unifying try-except and try-finally  
PEP written by Georg Brandl; implementation by Thomas Lee.

</div>

</div>

<div id="pep-342-new-generator-features" class="section">

<span id="pep-342"></span>

## PEP 342: New Generator Features<a href="#pep-342-new-generator-features" class="headerlink" title="Permalink to this headline">¶</a>

Python 2.5 adds a simple way to pass values *into* a generator. As introduced in Python 2.3, generators only produce output; once a generator’s code was invoked to create an iterator, there was no way to pass any new information into the function when its execution is resumed. Sometimes the ability to pass in some information would be useful. Hackish solutions to this include making the generator’s code look at a global variable and then changing the global variable’s value, or passing in some mutable object that callers then modify.

To refresh your memory of basic generators, here’s a simple example:

<div class="highlight-default notranslate">

<div class="highlight">

    def counter (maximum):
        i = 0
        while i < maximum:
            yield i
            i += 1

</div>

</div>

When you call <span class="pre">`counter(10)`</span>, the result is an iterator that returns the values from 0 up to 9. On encountering the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement, the iterator returns the provided value and suspends the function’s execution, preserving the local variables. Execution resumes on the following call to the iterator’s <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a> method, picking up after the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement.

In Python 2.3, <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> was a statement; it didn’t return any value. In 2.5, <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> is now an expression, returning a value that can be assigned to a variable or otherwise operated on:

<div class="highlight-default notranslate">

<div class="highlight">

    val = (yield i)

</div>

</div>

I recommend that you always put parentheses around a <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> expression when you’re doing something with the returned value, as in the above example. The parentheses aren’t always necessary, but it’s easier to always add them instead of having to remember when they’re needed.

(<span id="index-8" class="target"></span><a href="https://www.python.org/dev/peps/pep-0342" class="pep reference external"><strong>PEP 342</strong></a> explains the exact rules, which are that a <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a>-expression must always be parenthesized except when it occurs at the top-level expression on the right-hand side of an assignment. This means you can write <span class="pre">`val`</span>` `<span class="pre">`=`</span>` `<span class="pre">`yield`</span>` `<span class="pre">`i`</span> but have to use parentheses when there’s an operation, as in <span class="pre">`val`</span>` `<span class="pre">`=`</span>` `<span class="pre">`(yield`</span>` `<span class="pre">`i)`</span>` `<span class="pre">`+`</span>` `<span class="pre">`12`</span>.)

Values are sent into a generator by calling its <span class="pre">`send(value)`</span> method. The generator’s code is then resumed and the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> expression returns the specified *value*. If the regular <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a> method is called, the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> returns <a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a>.

Here’s the previous example, modified to allow changing the value of the internal counter.

<div class="highlight-default notranslate">

<div class="highlight">

    def counter (maximum):
        i = 0
        while i < maximum:
            val = (yield i)
            # If value provided, change counter
            if val is not None:
                i = val
            else:
                i += 1

</div>

</div>

And here’s an example of changing the counter:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> it = counter(10)
    >>> print it.next()
    0
    >>> print it.next()
    1
    >>> print it.send(8)
    8
    >>> print it.next()
    9
    >>> print it.next()
    Traceback (most recent call last):
      File "t.py", line 15, in ?
        print it.next()
    StopIteration

</div>

</div>

<a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> will usually return <a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a>, so you should always check for this case. Don’t just use its value in expressions unless you’re sure that the <span class="pre">`send()`</span> method will be the only method used to resume your generator function.

In addition to <span class="pre">`send()`</span>, there are two other new methods on generators:

- <span class="pre">`throw(type,`</span>` `<span class="pre">`value=None,`</span>` `<span class="pre">`traceback=None)`</span> is used to raise an exception inside the generator; the exception is raised by the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> expression where the generator’s execution is paused.

- <span class="pre">`close()`</span> raises a new <a href="../library/exceptions.html#exceptions.GeneratorExit" class="reference internal" title="exceptions.GeneratorExit"><span class="pre"><code class="sourceCode python"><span class="pp">GeneratorExit</span></code></span></a> exception inside the generator to terminate the iteration. On receiving this exception, the generator’s code must either raise <a href="../library/exceptions.html#exceptions.GeneratorExit" class="reference internal" title="exceptions.GeneratorExit"><span class="pre"><code class="sourceCode python"><span class="pp">GeneratorExit</span></code></span></a> or <a href="../library/exceptions.html#exceptions.StopIteration" class="reference internal" title="exceptions.StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a>. Catching the <a href="../library/exceptions.html#exceptions.GeneratorExit" class="reference internal" title="exceptions.GeneratorExit"><span class="pre"><code class="sourceCode python"><span class="pp">GeneratorExit</span></code></span></a> exception and returning a value is illegal and will trigger a <a href="../library/exceptions.html#exceptions.RuntimeError" class="reference internal" title="exceptions.RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a>; if the function raises some other exception, that exception is propagated to the caller. <span class="pre">`close()`</span> will also be called by Python’s garbage collector when the generator is garbage-collected.

  If you need to run cleanup code when a <a href="../library/exceptions.html#exceptions.GeneratorExit" class="reference internal" title="exceptions.GeneratorExit"><span class="pre"><code class="sourceCode python"><span class="pp">GeneratorExit</span></code></span></a> occurs, I suggest using a <span class="pre">`try:`</span>` `<span class="pre">`...`</span>` `<span class="pre">`finally:`</span> suite instead of catching <a href="../library/exceptions.html#exceptions.GeneratorExit" class="reference internal" title="exceptions.GeneratorExit"><span class="pre"><code class="sourceCode python"><span class="pp">GeneratorExit</span></code></span></a>.

The cumulative effect of these changes is to turn generators from one-way producers of information into both producers and consumers.

Generators also become *coroutines*, a more generalized form of subroutines. Subroutines are entered at one point and exited at another point (the top of the function, and a <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statement), but coroutines can be entered, exited, and resumed at many different points (the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statements). We’ll have to figure out patterns for using coroutines effectively in Python.

The addition of the <span class="pre">`close()`</span> method has one side effect that isn’t obvious. <span class="pre">`close()`</span> is called when a generator is garbage-collected, so this means the generator’s code gets one last chance to run before the generator is destroyed. This last chance means that <span class="pre">`try...finally`</span> statements in generators can now be guaranteed to work; the <a href="../reference/compound_stmts.html#finally" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">finally</code></span></a> clause will now always get a chance to run. The syntactic restriction that you couldn’t mix <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statements with a <span class="pre">`try...finally`</span> suite has therefore been removed. This seems like a minor bit of language trivia, but using generators and <span class="pre">`try...finally`</span> is actually necessary in order to implement the <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement described by PEP 343. I’ll look at this new statement in the following section.

Another even more esoteric effect of this change: previously, the <span class="pre">`gi_frame`</span> attribute of a generator was always a frame object. It’s now possible for <span class="pre">`gi_frame`</span> to be <span class="pre">`None`</span> once the generator has been exhausted.

<div class="admonition seealso">

See also

<span id="index-9" class="target"></span><a href="https://www.python.org/dev/peps/pep-0342" class="pep reference external"><strong>PEP 342</strong></a> - Coroutines via Enhanced Generators  
PEP written by Guido van Rossum and Phillip J. Eby; implemented by Phillip J. Eby. Includes examples of some fancier uses of generators as coroutines.

Earlier versions of these features were proposed in <span id="index-10" class="target"></span><a href="https://www.python.org/dev/peps/pep-0288" class="pep reference external"><strong>PEP 288</strong></a> by Raymond Hettinger and <span id="index-11" class="target"></span><a href="https://www.python.org/dev/peps/pep-0325" class="pep reference external"><strong>PEP 325</strong></a> by Samuele Pedroni.

<a href="https://en.wikipedia.org/wiki/Coroutine" class="reference external">https://en.wikipedia.org/wiki/Coroutine</a>  
The Wikipedia entry for coroutines.

<a href="http://www.sidhe.org/~dan/blog/archives/000178.html" class="reference external">http://www.sidhe.org/~dan/blog/archives/000178.html</a>  
An explanation of coroutines from a Perl point of view, written by Dan Sugalski.

</div>

</div>

<div id="pep-343-the-with-statement" class="section">

<span id="pep-343"></span>

## PEP 343: The ‘with’ statement<a href="#pep-343-the-with-statement" class="headerlink" title="Permalink to this headline">¶</a>

The ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement clarifies code that previously would use <span class="pre">`try...finally`</span> blocks to ensure that clean-up code is executed. In this section, I’ll discuss the statement as it will commonly be used. In the next section, I’ll examine the implementation details and show how to write objects for use with this statement.

The ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement is a new control-flow structure whose basic structure is:

<div class="highlight-default notranslate">

<div class="highlight">

    with expression [as variable]:
        with-block

</div>

</div>

The expression is evaluated, and it should result in an object that supports the context management protocol (that is, has <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> methods.

The object’s <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> is called before *with-block* is executed and therefore can run set-up code. It also may return a value that is bound to the name *variable*, if given. (Note carefully that *variable* is *not* assigned the result of *expression*.)

After execution of the *with-block* is finished, the object’s <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> method is called, even if the block raised an exception, and can therefore run clean-up code.

To enable the statement in Python 2.5, you need to add the following directive to your module:

<div class="highlight-default notranslate">

<div class="highlight">

    from __future__ import with_statement

</div>

</div>

The statement will always be enabled in Python 2.6.

Some standard Python objects now support the context management protocol and can be used with the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement. File objects are one example:

<div class="highlight-default notranslate">

<div class="highlight">

    with open('/etc/passwd', 'r') as f:
        for line in f:
            print line
            ... more processing code ...

</div>

</div>

After this statement has executed, the file object in *f* will have been automatically closed, even if the <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> loop raised an exception part-way through the block.

<div class="admonition note">

Note

In this case, *f* is the same object created by <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a>, because <span class="pre">`file.__enter__()`</span> returns *self*.

</div>

The <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> module’s locks and condition variables also support the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement:

<div class="highlight-default notranslate">

<div class="highlight">

    lock = threading.Lock()
    with lock:
        # Critical section of code
        ...

</div>

</div>

The lock is acquired before the block is executed and always released once the block is complete.

The new <span class="pre">`localcontext()`</span> function in the <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic  Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> module makes it easy to save and restore the current decimal context, which encapsulates the desired precision and rounding characteristics for computations:

<div class="highlight-default notranslate">

<div class="highlight">

    from decimal import Decimal, Context, localcontext

    # Displays with default precision of 28 digits
    v = Decimal('578')
    print v.sqrt()

    with localcontext(Context(prec=16)):
        # All code in this block uses a precision of 16 digits.
        # The original context is restored on exiting the block.
        print v.sqrt()

</div>

</div>

<div id="writing-context-managers" class="section">

<span id="new-25-context-managers"></span>

### Writing Context Managers<a href="#writing-context-managers" class="headerlink" title="Permalink to this headline">¶</a>

Under the hood, the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement is fairly complicated. Most people will only use ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ in company with existing objects and don’t need to know these details, so you can skip the rest of this section if you like. Authors of new objects will need to understand the details of the underlying implementation and should keep reading.

A high-level explanation of the context management protocol is:

- The expression is evaluated and should result in an object called a “context manager”. The context manager must have <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> methods.

- The context manager’s <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> method is called. The value returned is assigned to *VAR*. If no <span class="pre">`'as`</span>` `<span class="pre">`VAR'`</span> clause is present, the value is simply discarded.

- The code in *BLOCK* is executed.

- If *BLOCK* raises an exception, the <span class="pre">`__exit__(type,`</span>` `<span class="pre">`value,`</span>` `<span class="pre">`traceback)`</span> is called with the exception details, the same values returned by <a href="../library/sys.html#sys.exc_info" class="reference internal" title="sys.exc_info"><span class="pre"><code class="sourceCode python">sys.exc_info()</code></span></a>. The method’s return value controls whether the exception is re-raised: any false value re-raises the exception, and <span class="pre">`True`</span> will result in suppressing it. You’ll only rarely want to suppress the exception, because if you do the author of the code containing the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement will never realize anything went wrong.

- If *BLOCK* didn’t raise an exception, the <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> method is still called, but *type*, *value*, and *traceback* are all <span class="pre">`None`</span>.

Let’s think through an example. I won’t present detailed code but will only sketch the methods necessary for a database that supports transactions.

(For people unfamiliar with database terminology: a set of changes to the database are grouped into a transaction. Transactions can be either committed, meaning that all the changes are written into the database, or rolled back, meaning that the changes are all discarded and the database is unchanged. See any database textbook for more information.)

Let’s assume there’s an object representing a database connection. Our goal will be to let the user write code like this:

<div class="highlight-default notranslate">

<div class="highlight">

    db_connection = DatabaseConnection()
    with db_connection as cursor:
        cursor.execute('insert into ...')
        cursor.execute('delete from ...')
        # ... more operations ...

</div>

</div>

The transaction should be committed if the code in the block runs flawlessly or rolled back if there’s an exception. Here’s the basic interface for <span class="pre">`DatabaseConnection`</span> that I’ll assume:

<div class="highlight-default notranslate">

<div class="highlight">

    class DatabaseConnection:
        # Database interface
        def cursor (self):
            "Returns a cursor object and starts a new transaction"
        def commit (self):
            "Commits current transaction"
        def rollback (self):
            "Rolls back current transaction"

</div>

</div>

The <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> method is pretty easy, having only to start a new transaction. For this application the resulting cursor object would be a useful result, so the method will return it. The user can then add <span class="pre">`as`</span>` `<span class="pre">`cursor`</span> to their ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement to bind the cursor to a variable name.

<div class="highlight-default notranslate">

<div class="highlight">

    class DatabaseConnection:
        ...
        def __enter__ (self):
            # Code to start a new transaction
            cursor = self.cursor()
            return cursor

</div>

</div>

The <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> method is the most complicated because it’s where most of the work has to be done. The method has to check if an exception occurred. If there was no exception, the transaction is committed. The transaction is rolled back if there was an exception.

In the code below, execution will just fall off the end of the function, returning the default value of <span class="pre">`None`</span>. <span class="pre">`None`</span> is false, so the exception will be re-raised automatically. If you wished, you could be more explicit and add a <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statement at the marked location.

<div class="highlight-default notranslate">

<div class="highlight">

    class DatabaseConnection:
        ...
        def __exit__ (self, type, value, tb):
            if tb is None:
                # No exception, so commit
                self.commit()
            else:
                # Exception occurred, so rollback.
                self.rollback()
                # return False

</div>

</div>

</div>

<div id="the-contextlib-module" class="section">

<span id="contextlibmod"></span>

### The contextlib module<a href="#the-contextlib-module" class="headerlink" title="Permalink to this headline">¶</a>

The new <a href="../library/contextlib.html#module-contextlib" class="reference internal" title="contextlib: Utilities for with-statement contexts."><span class="pre"><code class="sourceCode python">contextlib</code></span></a> module provides some functions and a decorator that are useful for writing objects for use with the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement.

The decorator is called <span class="pre">`contextmanager()`</span>, and lets you write a single generator function instead of defining a new class. The generator should yield exactly one value. The code up to the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> will be executed as the <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> method, and the value yielded will be the method’s return value that will get bound to the variable in the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement’s <a href="../reference/compound_stmts.html#as" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">as</code></span></a> clause, if any. The code after the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> will be executed in the <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> method. Any exception raised in the block will be raised by the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement.

Our database example from the previous section could be written using this decorator as:

<div class="highlight-default notranslate">

<div class="highlight">

    from contextlib import contextmanager

    @contextmanager
    def db_transaction (connection):
        cursor = connection.cursor()
        try:
            yield cursor
        except:
            connection.rollback()
            raise
        else:
            connection.commit()

    db = DatabaseConnection()
    with db_transaction(db) as cursor:
        ...

</div>

</div>

The <a href="../library/contextlib.html#module-contextlib" class="reference internal" title="contextlib: Utilities for with-statement contexts."><span class="pre"><code class="sourceCode python">contextlib</code></span></a> module also has a <span class="pre">`nested(mgr1,`</span>` `<span class="pre">`mgr2,`</span>` `<span class="pre">`...)`</span> function that combines a number of context managers so you don’t need to write nested ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statements. In this example, the single ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement both starts a database transaction and acquires a thread lock:

<div class="highlight-default notranslate">

<div class="highlight">

    lock = threading.Lock()
    with nested (db_transaction(db), lock) as (cursor, locked):
        ...

</div>

</div>

Finally, the <span class="pre">`closing(object)`</span> function returns *object* so that it can be bound to a variable, and calls <span class="pre">`object.close`</span> at the end of the block.

<div class="highlight-default notranslate">

<div class="highlight">

    import urllib, sys
    from contextlib import closing

    with closing(urllib.urlopen('http://www.yahoo.com')) as f:
        for line in f:
            sys.stdout.write(line)

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-12" class="target"></span><a href="https://www.python.org/dev/peps/pep-0343" class="pep reference external"><strong>PEP 343</strong></a> - The “with” statement  
PEP written by Guido van Rossum and Nick Coghlan; implemented by Mike Bland, Guido van Rossum, and Neal Norwitz. The PEP shows the code generated for a ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement, which can be helpful in learning how the statement works.

The documentation for the <a href="../library/contextlib.html#module-contextlib" class="reference internal" title="contextlib: Utilities for with-statement contexts."><span class="pre"><code class="sourceCode python">contextlib</code></span></a> module.

</div>

</div>

</div>

<div id="pep-352-exceptions-as-new-style-classes" class="section">

<span id="pep-352"></span>

## PEP 352: Exceptions as New-Style Classes<a href="#pep-352-exceptions-as-new-style-classes" class="headerlink" title="Permalink to this headline">¶</a>

Exception classes can now be new-style classes, not just classic classes, and the built-in <a href="../library/exceptions.html#exceptions.Exception" class="reference internal" title="exceptions.Exception"><span class="pre"><code class="sourceCode python"><span class="pp">Exception</span></code></span></a> class and all the standard built-in exceptions (<a href="../library/exceptions.html#exceptions.NameError" class="reference internal" title="exceptions.NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a>, <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a>, etc.) are now new-style classes.

The inheritance hierarchy for exceptions has been rearranged a bit. In 2.5, the inheritance relationships are:

<div class="highlight-default notranslate">

<div class="highlight">

    BaseException       # New in Python 2.5
    |- KeyboardInterrupt
    |- SystemExit
    |- Exception
       |- (all other current built-in exceptions)

</div>

</div>

This rearrangement was done because people often want to catch all exceptions that indicate program errors. <a href="../library/exceptions.html#exceptions.KeyboardInterrupt" class="reference internal" title="exceptions.KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a> and <a href="../library/exceptions.html#exceptions.SystemExit" class="reference internal" title="exceptions.SystemExit"><span class="pre"><code class="sourceCode python"><span class="pp">SystemExit</span></code></span></a> aren’t errors, though, and usually represent an explicit action such as the user hitting <span class="kbd kbd docutils literal notranslate">Control-C</span> or code calling <a href="../library/sys.html#sys.exit" class="reference internal" title="sys.exit"><span class="pre"><code class="sourceCode python">sys.exit()</code></span></a>. A bare <span class="pre">`except:`</span> will catch all exceptions, so you commonly need to list <a href="../library/exceptions.html#exceptions.KeyboardInterrupt" class="reference internal" title="exceptions.KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a> and <a href="../library/exceptions.html#exceptions.SystemExit" class="reference internal" title="exceptions.SystemExit"><span class="pre"><code class="sourceCode python"><span class="pp">SystemExit</span></code></span></a> in order to re-raise them. The usual pattern is:

<div class="highlight-default notranslate">

<div class="highlight">

    try:
        ...
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        # Log error...
        # Continue running program...

</div>

</div>

In Python 2.5, you can now write <span class="pre">`except`</span>` `<span class="pre">`Exception`</span> to achieve the same result, catching all the exceptions that usually indicate errors but leaving <a href="../library/exceptions.html#exceptions.KeyboardInterrupt" class="reference internal" title="exceptions.KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a> and <a href="../library/exceptions.html#exceptions.SystemExit" class="reference internal" title="exceptions.SystemExit"><span class="pre"><code class="sourceCode python"><span class="pp">SystemExit</span></code></span></a> alone. As in previous versions, a bare <span class="pre">`except:`</span> still catches all exceptions.

The goal for Python 3.0 is to require any class raised as an exception to derive from <a href="../library/exceptions.html#exceptions.BaseException" class="reference internal" title="exceptions.BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a> or some descendant of <a href="../library/exceptions.html#exceptions.BaseException" class="reference internal" title="exceptions.BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a>, and future releases in the Python 2.x series may begin to enforce this constraint. Therefore, I suggest you begin making all your exception classes derive from <a href="../library/exceptions.html#exceptions.Exception" class="reference internal" title="exceptions.Exception"><span class="pre"><code class="sourceCode python"><span class="pp">Exception</span></code></span></a> now. It’s been suggested that the bare <span class="pre">`except:`</span> form should be removed in Python 3.0, but Guido van Rossum hasn’t decided whether to do this or not.

Raising of strings as exceptions, as in the statement <span class="pre">`raise`</span>` `<span class="pre">`"Error`</span>` `<span class="pre">`occurred"`</span>, is deprecated in Python 2.5 and will trigger a warning. The aim is to be able to remove the string-exception feature in a few releases.

<div class="admonition seealso">

See also

<span id="index-13" class="target"></span><a href="https://www.python.org/dev/peps/pep-0352" class="pep reference external"><strong>PEP 352</strong></a> - Required Superclass for Exceptions  
PEP written by Brett Cannon and Guido van Rossum; implemented by Brett Cannon.

</div>

</div>

<div id="pep-353-using-ssize-t-as-the-index-type" class="section">

<span id="pep-353"></span>

## PEP 353: Using ssize_t as the index type<a href="#pep-353-using-ssize-t-as-the-index-type" class="headerlink" title="Permalink to this headline">¶</a>

A wide-ranging change to Python’s C API, using a new <span class="pre">`Py_ssize_t`</span> type definition instead of <span class="pre">`int`</span>, will permit the interpreter to handle more data on 64-bit platforms. This change doesn’t affect Python’s capacity on 32-bit platforms.

Various pieces of the Python interpreter used C’s <span class="pre">`int`</span> type to store sizes or counts; for example, the number of items in a list or tuple were stored in an <span class="pre">`int`</span>. The C compilers for most 64-bit platforms still define <span class="pre">`int`</span> as a 32-bit type, so that meant that lists could only hold up to <span class="pre">`2**31`</span>` `<span class="pre">`-`</span>` `<span class="pre">`1`</span> = 2147483647 items. (There are actually a few different programming models that 64-bit C compilers can use – see <a href="http://www.unix.org/version2/whatsnew/lp64_wp.html" class="reference external">http://www.unix.org/version2/whatsnew/lp64_wp.html</a> for a discussion – but the most commonly available model leaves <span class="pre">`int`</span> as 32 bits.)

A limit of 2147483647 items doesn’t really matter on a 32-bit platform because you’ll run out of memory before hitting the length limit. Each list item requires space for a pointer, which is 4 bytes, plus space for a <a href="../c-api/structures.html#c.PyObject" class="reference internal" title="PyObject"><span class="pre"><code class="sourceCode c">PyObject</code></span></a> representing the item. 2147483647\*4 is already more bytes than a 32-bit address space can contain.

It’s possible to address that much memory on a 64-bit platform, however. The pointers for a list that size would only require 16 GiB of space, so it’s not unreasonable that Python programmers might construct lists that large. Therefore, the Python interpreter had to be changed to use some type other than <span class="pre">`int`</span>, and this will be a 64-bit type on 64-bit platforms. The change will cause incompatibilities on 64-bit machines, so it was deemed worth making the transition now, while the number of 64-bit users is still relatively small. (In 5 or 10 years, we may *all* be on 64-bit machines, and the transition would be more painful then.)

This change most strongly affects authors of C extension modules. Python strings and container types such as lists and tuples now use <span class="pre">`Py_ssize_t`</span> to store their size. Functions such as <a href="../c-api/list.html#c.PyList_Size" class="reference internal" title="PyList_Size"><span class="pre"><code class="sourceCode c">PyList_Size<span class="op">()</span></code></span></a> now return <span class="pre">`Py_ssize_t`</span>. Code in extension modules may therefore need to have some variables changed to <span class="pre">`Py_ssize_t`</span>.

The <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple<span class="op">()</span></code></span></a> and <a href="../c-api/arg.html#c.Py_BuildValue" class="reference internal" title="Py_BuildValue"><span class="pre"><code class="sourceCode c">Py_BuildValue<span class="op">()</span></code></span></a> functions have a new conversion code, <span class="pre">`n`</span>, for <span class="pre">`Py_ssize_t`</span>. <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple<span class="op">()</span></code></span></a>’s <span class="pre">`s#`</span> and <span class="pre">`t#`</span> still output <span class="pre">`int`</span> by default, but you can define the macro <span class="pre">`PY_SSIZE_T_CLEAN`</span> before including <span class="pre">`Python.h`</span> to make them return <span class="pre">`Py_ssize_t`</span>.

<span id="index-14" class="target"></span><a href="https://www.python.org/dev/peps/pep-0353" class="pep reference external"><strong>PEP 353</strong></a> has a section on conversion guidelines that extension authors should read to learn about supporting 64-bit platforms.

<div class="admonition seealso">

See also

<span id="index-15" class="target"></span><a href="https://www.python.org/dev/peps/pep-0353" class="pep reference external"><strong>PEP 353</strong></a> - Using ssize_t as the index type  
PEP written and implemented by Martin von Löwis.

</div>

</div>

<div id="pep-357-the-index-method" class="section">

<span id="pep-357"></span>

## PEP 357: The ‘\_\_index\_\_’ method<a href="#pep-357-the-index-method" class="headerlink" title="Permalink to this headline">¶</a>

The NumPy developers had a problem that could only be solved by adding a new special method, <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a>. When using slice notation, as in <span class="pre">`[start:stop:step]`</span>, the values of the *start*, *stop*, and *step* indexes must all be either integers or long integers. NumPy defines a variety of specialized integer types corresponding to unsigned and signed integers of 8, 16, 32, and 64 bits, but there was no way to signal that these types could be used as slice indexes.

Slicing can’t just use the existing <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="fu">__int__</span>()</code></span></a> method because that method is also used to implement coercion to integers. If slicing used <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="fu">__int__</span>()</code></span></a>, floating-point numbers would also become legal slice indexes and that’s clearly an undesirable behaviour.

Instead, a new special method called <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> was added. It takes no arguments and returns an integer giving the slice index to use. For example:

<div class="highlight-default notranslate">

<div class="highlight">

    class C:
        def __index__ (self):
            return self.value

</div>

</div>

The return value must be either a Python integer or long integer. The interpreter will check that the type returned is correct, and raises a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> if this requirement isn’t met.

A corresponding <span class="pre">`nb_index`</span> slot was added to the C-level <a href="../c-api/typeobj.html#c.PyNumberMethods" class="reference internal" title="PyNumberMethods"><span class="pre"><code class="sourceCode c">PyNumberMethods</code></span></a> structure to let C extensions implement this protocol. <span class="pre">`PyNumber_Index(obj)`</span> can be used in extension code to call the <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> function and retrieve its result.

<div class="admonition seealso">

See also

<span id="index-16" class="target"></span><a href="https://www.python.org/dev/peps/pep-0357" class="pep reference external"><strong>PEP 357</strong></a> - Allowing Any Object to be Used for Slicing  
PEP written and implemented by Travis Oliphant.

</div>

</div>

<div id="other-language-changes" class="section">

<span id="other-lang"></span>

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Permalink to this headline">¶</a>

Here are all of the changes that Python 2.5 makes to the core Python language.

- The <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> type has a new hook for letting subclasses provide a default value when a key isn’t contained in the dictionary. When a key isn’t found, the dictionary’s <span class="pre">`__missing__(key)`</span> method will be called. This hook is used to implement the new <span class="pre">`defaultdict`</span> class in the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module. The following example defines a dictionary that returns zero for any missing key:

  <div class="highlight-default notranslate">

  <div class="highlight">

      class zerodict (dict):
          def __missing__ (self, key):
              return 0

      d = zerodict({1:1, 2:2})
      print d[1], d[2]   # Prints 1, 2
      print d[3], d[4]   # Prints 0, 0

  </div>

  </div>

- Both 8-bit and Unicode strings have new <span class="pre">`partition(sep)`</span> and <span class="pre">`rpartition(sep)`</span> methods that simplify a common use case.

  The <span class="pre">`find(S)`</span> method is often used to get an index which is then used to slice the string and obtain the pieces that are before and after the separator. <span class="pre">`partition(sep)`</span> condenses this pattern into a single method call that returns a 3-tuple containing the substring before the separator, the separator itself, and the substring after the separator. If the separator isn’t found, the first element of the tuple is the entire string and the other two elements are empty. <span class="pre">`rpartition(sep)`</span> also returns a 3-tuple but starts searching from the end of the string; the <span class="pre">`r`</span> stands for ‘reverse’.

  Some examples:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> ('http://www.python.org').partition('://')
      ('http', '://', 'www.python.org')
      >>> ('file:/usr/share/doc/index.html').partition('://')
      ('file:/usr/share/doc/index.html', '', '')
      >>> (u'Subject: a quick question').partition(':')
      (u'Subject', u':', u' a quick question')
      >>> 'www.python.org'.rpartition('.')
      ('www.python', '.', 'org')
      >>> 'www.python.org'.rpartition(':')
      ('', '', 'www.python.org')

  </div>

  </div>

  (Implemented by Fredrik Lundh following a suggestion by Raymond Hettinger.)

- The <span class="pre">`startswith()`</span> and <span class="pre">`endswith()`</span> methods of string types now accept tuples of strings to check for.

  <div class="highlight-default notranslate">

  <div class="highlight">

      def is_image_file (filename):
          return filename.endswith(('.gif', '.jpg', '.tiff'))

  </div>

  </div>

  (Implemented by Georg Brandl following a suggestion by Tom Lynn.)

- The <a href="../library/functions.html#min" class="reference internal" title="min"><span class="pre"><code class="sourceCode python"><span class="bu">min</span>()</code></span></a> and <a href="../library/functions.html#max" class="reference internal" title="max"><span class="pre"><code class="sourceCode python"><span class="bu">max</span>()</code></span></a> built-in functions gained a <span class="pre">`key`</span> keyword parameter analogous to the <span class="pre">`key`</span> argument for <span class="pre">`sort()`</span>. This parameter supplies a function that takes a single argument and is called for every value in the list; <a href="../library/functions.html#min" class="reference internal" title="min"><span class="pre"><code class="sourceCode python"><span class="bu">min</span>()</code></span></a>/<a href="../library/functions.html#max" class="reference internal" title="max"><span class="pre"><code class="sourceCode python"><span class="bu">max</span>()</code></span></a> will return the element with the smallest/largest return value from this function. For example, to find the longest string in a list, you can do:

  <div class="highlight-default notranslate">

  <div class="highlight">

      L = ['medium', 'longest', 'short']
      # Prints 'longest'
      print max(L, key=len)
      # Prints 'short', because lexicographically 'short' has the largest value
      print max(L)

  </div>

  </div>

  (Contributed by Steven Bethard and Raymond Hettinger.)

- Two new built-in functions, <a href="../library/functions.html#any" class="reference internal" title="any"><span class="pre"><code class="sourceCode python"><span class="bu">any</span>()</code></span></a> and <a href="../library/functions.html#all" class="reference internal" title="all"><span class="pre"><code class="sourceCode python"><span class="bu">all</span>()</code></span></a>, evaluate whether an iterator contains any true or false values. <a href="../library/functions.html#any" class="reference internal" title="any"><span class="pre"><code class="sourceCode python"><span class="bu">any</span>()</code></span></a> returns <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> if any value returned by the iterator is true; otherwise it will return <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a>. <a href="../library/functions.html#all" class="reference internal" title="all"><span class="pre"><code class="sourceCode python"><span class="bu">all</span>()</code></span></a> returns <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> only if all of the values returned by the iterator evaluate as true. (Suggested by Guido van Rossum, and implemented by Raymond Hettinger.)

- The result of a class’s <a href="../reference/datamodel.html#object.__hash__" class="reference internal" title="object.__hash__"><span class="pre"><code class="sourceCode python"><span class="fu">__hash__</span>()</code></span></a> method can now be either a long integer or a regular integer. If a long integer is returned, the hash of that value is taken. In earlier versions the hash value was required to be a regular integer, but in 2.5 the <a href="../library/functions.html#id" class="reference internal" title="id"><span class="pre"><code class="sourceCode python"><span class="bu">id</span>()</code></span></a> built-in was changed to always return non-negative numbers, and users often seem to use <span class="pre">`id(self)`</span> in <a href="../reference/datamodel.html#object.__hash__" class="reference internal" title="object.__hash__"><span class="pre"><code class="sourceCode python"><span class="fu">__hash__</span>()</code></span></a> methods (though this is discouraged).

- ASCII is now the default encoding for modules. It’s now a syntax error if a module contains string literals with 8-bit characters but doesn’t have an encoding declaration. In Python 2.4 this triggered a warning, not a syntax error. See <span id="index-17" class="target"></span><a href="https://www.python.org/dev/peps/pep-0263" class="pep reference external"><strong>PEP 263</strong></a> for how to declare a module’s encoding; for example, you might add a line like this near the top of the source file:

  <div class="highlight-default notranslate">

  <div class="highlight">

      # -*- coding: latin1 -*-

  </div>

  </div>

- A new warning, <span class="pre">`UnicodeWarning`</span>, is triggered when you attempt to compare a Unicode string and an 8-bit string that can’t be converted to Unicode using the default ASCII encoding. The result of the comparison is false:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> chr(128) == unichr(128)   # Can't convert chr(128) to Unicode
      __main__:1: UnicodeWarning: Unicode equal comparison failed
        to convert both arguments to Unicode - interpreting them
        as being unequal
      False
      >>> chr(127) == unichr(127)   # chr(127) can be converted
      True

  </div>

  </div>

  Previously this would raise a <span class="pre">`UnicodeDecodeError`</span> exception, but in 2.5 this could result in puzzling problems when accessing a dictionary. If you looked up <span class="pre">`unichr(128)`</span> and <span class="pre">`chr(128)`</span> was being used as a key, you’d get a <span class="pre">`UnicodeDecodeError`</span> exception. Other changes in 2.5 resulted in this exception being raised instead of suppressed by the code in <span class="pre">`dictobject.c`</span> that implements dictionaries.

  Raising an exception for such a comparison is strictly correct, but the change might have broken code, so instead <span class="pre">`UnicodeWarning`</span> was introduced.

  (Implemented by Marc-André Lemburg.)

- One error that Python programmers sometimes make is forgetting to include an <span class="pre">`__init__.py`</span> module in a package directory. Debugging this mistake can be confusing, and usually requires running Python with the <a href="../using/cmdline.html#id3" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-v</code></span></a> switch to log all the paths searched. In Python 2.5, a new <a href="../library/exceptions.html#exceptions.ImportWarning" class="reference internal" title="exceptions.ImportWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ImportWarning</span></code></span></a> warning is triggered when an import would have picked up a directory as a package but no <span class="pre">`__init__.py`</span> was found. This warning is silently ignored by default; provide the <a href="../using/cmdline.html#cmdoption-w" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-Wd</code></span></a> option when running the Python executable to display the warning message. (Implemented by Thomas Wouters.)

- The list of base classes in a class definition can now be empty. As an example, this is now legal:

  <div class="highlight-default notranslate">

  <div class="highlight">

      class C():
          pass

  </div>

  </div>

  (Implemented by Brett Cannon.)

<div id="interactive-interpreter-changes" class="section">

<span id="interactive"></span>

### Interactive Interpreter Changes<a href="#interactive-interpreter-changes" class="headerlink" title="Permalink to this headline">¶</a>

In the interactive interpreter, <span class="pre">`quit`</span> and <span class="pre">`exit`</span> have long been strings so that new users get a somewhat helpful message when they try to quit:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> quit
    'Use Ctrl-D (i.e. EOF) to exit.'

</div>

</div>

In Python 2.5, <span class="pre">`quit`</span> and <span class="pre">`exit`</span> are now objects that still produce string representations of themselves, but are also callable. Newbies who try <span class="pre">`quit()`</span> or <span class="pre">`exit()`</span> will now exit the interpreter as they expect. (Implemented by Georg Brandl.)

The Python executable now accepts the standard long options <a href="../using/cmdline.html#cmdoption-help" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--help</code></span></a> and <a href="../using/cmdline.html#cmdoption-version" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--version</code></span></a>; on Windows, it also accepts the <a href="../using/cmdline.html#cmdoption" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">/?</code></span></a> option for displaying a help message. (Implemented by Georg Brandl.)

</div>

<div id="optimizations" class="section">

<span id="opts"></span>

### Optimizations<a href="#optimizations" class="headerlink" title="Permalink to this headline">¶</a>

Several of the optimizations were developed at the NeedForSpeed sprint, an event held in Reykjavik, Iceland, from May 21–28 2006. The sprint focused on speed enhancements to the CPython implementation and was funded by EWT LLC with local support from CCP Games. Those optimizations added at this sprint are specially marked in the following list.

- When they were introduced in Python 2.4, the built-in <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a> and <a href="../library/stdtypes.html#frozenset" class="reference internal" title="frozenset"><span class="pre"><code class="sourceCode python"><span class="bu">frozenset</span></code></span></a> types were built on top of Python’s dictionary type. In 2.5 the internal data structure has been customized for implementing sets, and as a result sets will use a third less memory and are somewhat faster. (Implemented by Raymond Hettinger.)

- The speed of some Unicode operations, such as finding substrings, string splitting, and character map encoding and decoding, has been improved. (Substring search and splitting improvements were added by Fredrik Lundh and Andrew Dalke at the NeedForSpeed sprint. Character maps were improved by Walter Dörwald and Martin von Löwis.)

- The <span class="pre">`long(str,`</span>` `<span class="pre">`base)`</span> function is now faster on long digit strings because fewer intermediate results are calculated. The peak is for strings of around 800–1000 digits where the function is 6 times faster. (Contributed by Alan McIntyre and committed at the NeedForSpeed sprint.)

- It’s now illegal to mix iterating over a file with <span class="pre">`for`</span>` `<span class="pre">`line`</span>` `<span class="pre">`in`</span>` `<span class="pre">`file`</span> and calling the file object’s <span class="pre">`read()`</span>/<a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline()</code></span></a>/<span class="pre">`readlines()`</span> methods. Iteration uses an internal buffer and the <span class="pre">`read*()`</span> methods don’t use that buffer. Instead they would return the data following the buffer, causing the data to appear out of order. Mixing iteration and these methods will now trigger a <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> from the <span class="pre">`read*()`</span> method. (Implemented by Thomas Wouters.)

- The <a href="../library/struct.html#module-struct" class="reference internal" title="struct: Interpret strings as packed binary data."><span class="pre"><code class="sourceCode python">struct</code></span></a> module now compiles structure format strings into an internal representation and caches this representation, yielding a 20% speedup. (Contributed by Bob Ippolito at the NeedForSpeed sprint.)

- The <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module got a 1 or 2% speedup by switching to Python’s allocator functions instead of the system’s <span class="pre">`malloc()`</span> and <span class="pre">`free()`</span>. (Contributed by Jack Diederich at the NeedForSpeed sprint.)

- The code generator’s peephole optimizer now performs simple constant folding in expressions. If you write something like <span class="pre">`a`</span>` `<span class="pre">`=`</span>` `<span class="pre">`2+3`</span>, the code generator will do the arithmetic and produce code corresponding to <span class="pre">`a`</span>` `<span class="pre">`=`</span>` `<span class="pre">`5`</span>. (Proposed and implemented by Raymond Hettinger.)

- Function calls are now faster because code objects now keep the most recently finished frame (a “zombie frame”) in an internal field of the code object, reusing it the next time the code object is invoked. (Original patch by Michael Hudson, modified by Armin Rigo and Richard Jones; committed at the NeedForSpeed sprint.) Frame objects are also slightly smaller, which may improve cache locality and reduce memory usage a bit. (Contributed by Neal Norwitz.)

- Python’s built-in exceptions are now new-style classes, a change that speeds up instantiation considerably. Exception handling in Python 2.5 is therefore about 30% faster than in 2.4. (Contributed by Richard Jones, Georg Brandl and Sean Reifschneider at the NeedForSpeed sprint.)

- Importing now caches the paths tried, recording whether they exist or not so that the interpreter makes fewer <span class="pre">`open()`</span> and <span class="pre">`stat()`</span> calls on startup. (Contributed by Martin von Löwis and Georg Brandl.)

</div>

</div>

<div id="new-improved-and-removed-modules" class="section">

<span id="modules"></span>

## New, Improved, and Removed Modules<a href="#new-improved-and-removed-modules" class="headerlink" title="Permalink to this headline">¶</a>

The standard library received many enhancements and bug fixes in Python 2.5. Here’s a partial list of the most notable changes, sorted alphabetically by module name. Consult the <span class="pre">`Misc/NEWS`</span> file in the source tree for a more complete list of changes, or look through the SVN logs for all the details.

- The <a href="../library/audioop.html#module-audioop" class="reference internal" title="audioop: Manipulate raw audio data."><span class="pre"><code class="sourceCode python">audioop</code></span></a> module now supports the a-LAW encoding, and the code for u-LAW encoding has been improved. (Contributed by Lars Immisch.)

- The <a href="../library/codecs.html#module-codecs" class="reference internal" title="codecs: Encode and decode data and streams."><span class="pre"><code class="sourceCode python">codecs</code></span></a> module gained support for incremental codecs. The <span class="pre">`codec.lookup()`</span> function now returns a <span class="pre">`CodecInfo`</span> instance instead of a tuple. <span class="pre">`CodecInfo`</span> instances behave like a 4-tuple to preserve backward compatibility but also have the attributes <span class="pre">`encode`</span>, <span class="pre">`decode`</span>, <span class="pre">`incrementalencoder`</span>, <span class="pre">`incrementaldecoder`</span>, <span class="pre">`streamwriter`</span>, and <span class="pre">`streamreader`</span>. Incremental codecs can receive input and produce output in multiple chunks; the output is the same as if the entire input was fed to the non-incremental codec. See the <a href="../library/codecs.html#module-codecs" class="reference internal" title="codecs: Encode and decode data and streams."><span class="pre"><code class="sourceCode python">codecs</code></span></a> module documentation for details. (Designed and implemented by Walter Dörwald.)

- The <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module gained a new type, <span class="pre">`defaultdict`</span>, that subclasses the standard <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> type. The new type mostly behaves like a dictionary but constructs a default value when a key isn’t present, automatically adding it to the dictionary for the requested key value.

  The first argument to <span class="pre">`defaultdict`</span>’s constructor is a factory function that gets called whenever a key is requested but not found. This factory function receives no arguments, so you can use built-in type constructors such as <span class="pre">`list()`</span> or <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a>. For example, you can make an index of words based on their initial letter like this:

  <div class="highlight-default notranslate">

  <div class="highlight">

      words = """Nel mezzo del cammin di nostra vita
      mi ritrovai per una selva oscura
      che la diritta via era smarrita""".lower().split()

      index = defaultdict(list)

      for w in words:
          init_letter = w[0]
          index[init_letter].append(w)

  </div>

  </div>

  Printing <span class="pre">`index`</span> results in the following output:

  <div class="highlight-default notranslate">

  <div class="highlight">

      defaultdict(<type 'list'>, {'c': ['cammin', 'che'], 'e': ['era'],
              'd': ['del', 'di', 'diritta'], 'm': ['mezzo', 'mi'],
              'l': ['la'], 'o': ['oscura'], 'n': ['nel', 'nostra'],
              'p': ['per'], 's': ['selva', 'smarrita'],
              'r': ['ritrovai'], 'u': ['una'], 'v': ['vita', 'via']}

  </div>

  </div>

  (Contributed by Guido van Rossum.)

- The <span class="pre">`deque`</span> double-ended queue type supplied by the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module now has a <span class="pre">`remove(value)`</span> method that removes the first occurrence of *value* in the queue, raising <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the value isn’t found. (Contributed by Raymond Hettinger.)

- New module: The <a href="../library/contextlib.html#module-contextlib" class="reference internal" title="contextlib: Utilities for with-statement contexts."><span class="pre"><code class="sourceCode python">contextlib</code></span></a> module contains helper functions for use with the new ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement. See section <a href="#contextlibmod" class="reference internal"><span class="std std-ref">The contextlib module</span></a> for more about this module.

- New module: The <a href="../library/profile.html#module-cProfile" class="reference internal" title="cProfile"><span class="pre"><code class="sourceCode python">cProfile</code></span></a> module is a C implementation of the existing <a href="../library/profile.html#module-profile" class="reference internal" title="profile: Python source profiler."><span class="pre"><code class="sourceCode python">profile</code></span></a> module that has much lower overhead. The module’s interface is the same as <a href="../library/profile.html#module-profile" class="reference internal" title="profile: Python source profiler."><span class="pre"><code class="sourceCode python">profile</code></span></a>: you run <span class="pre">`cProfile.run('main()')`</span> to profile a function, can save profile data to a file, etc. It’s not yet known if the Hotshot profiler, which is also written in C but doesn’t match the <a href="../library/profile.html#module-profile" class="reference internal" title="profile: Python source profiler."><span class="pre"><code class="sourceCode python">profile</code></span></a> module’s interface, will continue to be maintained in future versions of Python. (Contributed by Armin Rigo.)

  Also, the <a href="../library/profile.html#module-pstats" class="reference internal" title="pstats: Statistics object for use with the profiler."><span class="pre"><code class="sourceCode python">pstats</code></span></a> module for analyzing the data measured by the profiler now supports directing the output to any file object by supplying a *stream* argument to the <span class="pre">`Stats`</span> constructor. (Contributed by Skip Montanaro.)

- The <a href="../library/csv.html#module-csv" class="reference internal" title="csv: Write and read tabular data to and from delimited files."><span class="pre"><code class="sourceCode python">csv</code></span></a> module, which parses files in comma-separated value format, received several enhancements and a number of bugfixes. You can now set the maximum size in bytes of a field by calling the <span class="pre">`csv.field_size_limit(new_limit)`</span> function; omitting the *new_limit* argument will return the currently-set limit. The <span class="pre">`reader`</span> class now has a <span class="pre">`line_num`</span> attribute that counts the number of physical lines read from the source; records can span multiple physical lines, so <span class="pre">`line_num`</span> is not the same as the number of records read.

  The CSV parser is now stricter about multi-line quoted fields. Previously, if a line ended within a quoted field without a terminating newline character, a newline would be inserted into the returned field. This behavior caused problems when reading files that contained carriage return characters within fields, so the code was changed to return the field without inserting newlines. As a consequence, if newlines embedded within fields are important, the input should be split into lines in a manner that preserves the newline characters.

  (Contributed by Skip Montanaro and Andrew McNamara.)

- The <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> class in the <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> module now has a <span class="pre">`strptime(string,`</span>` `<span class="pre">`format)`</span> method for parsing date strings, contributed by Josh Spoerri. It uses the same format characters as <a href="../library/time.html#time.strptime" class="reference internal" title="time.strptime"><span class="pre"><code class="sourceCode python">time.strptime()</code></span></a> and <a href="../library/time.html#time.strftime" class="reference internal" title="time.strftime"><span class="pre"><code class="sourceCode python">time.strftime()</code></span></a>:

  <div class="highlight-default notranslate">

  <div class="highlight">

      from datetime import datetime

      ts = datetime.strptime('10:13:15 2006-03-07',
                             '%H:%M:%S %Y-%m-%d')

  </div>

  </div>

- The <span class="pre">`SequenceMatcher.get_matching_blocks()`</span> method in the <a href="../library/difflib.html#module-difflib" class="reference internal" title="difflib: Helpers for computing differences between objects."><span class="pre"><code class="sourceCode python">difflib</code></span></a> module now guarantees to return a minimal list of blocks describing matching subsequences. Previously, the algorithm would occasionally break a block of matching elements into two list entries. (Enhancement by Tim Peters.)

- The <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> module gained a <span class="pre">`SKIP`</span> option that keeps an example from being executed at all. This is intended for code snippets that are usage examples intended for the reader and aren’t actually test cases.

  An *encoding* parameter was added to the <span class="pre">`testfile()`</span> function and the <span class="pre">`DocFileSuite`</span> class to specify the file’s encoding. This makes it easier to use non-ASCII characters in tests contained within a docstring. (Contributed by Bjorn Tillenius.)

- The <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages, including MIME documents."><span class="pre"><code class="sourceCode python">email</code></span></a> package has been updated to version 4.0. (Contributed by Barry Warsaw.)

- <div id="index-18">

  The <a href="../library/fileinput.html#module-fileinput" class="reference internal" title="fileinput: Loop over standard input or a list of files."><span class="pre"><code class="sourceCode python">fileinput</code></span></a> module was made more flexible. Unicode filenames are now supported, and a *mode* parameter that defaults to <span class="pre">`"r"`</span> was added to the <a href="../library/functions.html#input" class="reference internal" title="input"><span class="pre"><code class="sourceCode python"><span class="bu">input</span>()</code></span></a> function to allow opening files in binary or <a href="../glossary.html#term-universal-newlines" class="reference internal"><span class="xref std std-term">universal newlines</span></a> mode. Another new parameter, *openhook*, lets you use a function other than <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> to open the input files. Once you’re iterating over the set of files, the <span class="pre">`FileInput`</span> object’s new <span class="pre">`fileno()`</span> returns the file descriptor for the currently opened file. (Contributed by Georg Brandl.)

  </div>

- In the <a href="../library/gc.html#module-gc" class="reference internal" title="gc: Interface to the cycle-detecting garbage collector."><span class="pre"><code class="sourceCode python">gc</code></span></a> module, the new <span class="pre">`get_count()`</span> function returns a 3-tuple containing the current collection counts for the three GC generations. This is accounting information for the garbage collector; when these counts reach a specified threshold, a garbage collection sweep will be made. The existing <a href="../library/gc.html#gc.collect" class="reference internal" title="gc.collect"><span class="pre"><code class="sourceCode python">gc.collect()</code></span></a> function now takes an optional *generation* argument of 0, 1, or 2 to specify which generation to collect. (Contributed by Barry Warsaw.)

- The <span class="pre">`nsmallest()`</span> and <span class="pre">`nlargest()`</span> functions in the <a href="../library/heapq.html#module-heapq" class="reference internal" title="heapq: Heap queue algorithm (a.k.a. priority queue)."><span class="pre"><code class="sourceCode python">heapq</code></span></a> module now support a <span class="pre">`key`</span> keyword parameter similar to the one provided by the <a href="../library/functions.html#min" class="reference internal" title="min"><span class="pre"><code class="sourceCode python"><span class="bu">min</span>()</code></span></a>/<a href="../library/functions.html#max" class="reference internal" title="max"><span class="pre"><code class="sourceCode python"><span class="bu">max</span>()</code></span></a> functions and the <span class="pre">`sort()`</span> methods. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import heapq
      >>> L = ["short", 'medium', 'longest', 'longer still']
      >>> heapq.nsmallest(2, L)  # Return two lowest elements, lexicographically
      ['longer still', 'longest']
      >>> heapq.nsmallest(2, L, key=len)   # Return two shortest elements
      ['short', 'medium']

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- The <a href="../library/itertools.html#itertools.islice" class="reference internal" title="itertools.islice"><span class="pre"><code class="sourceCode python">itertools.islice()</code></span></a> function now accepts <span class="pre">`None`</span> for the start and step arguments. This makes it more compatible with the attributes of slice objects, so that you can now write the following:

  <div class="highlight-default notranslate">

  <div class="highlight">

      s = slice(5)     # Create slice object
      itertools.islice(iterable, s.start, s.stop, s.step)

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- The <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> function in the <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a> module has been modified and two new functions were added, <span class="pre">`format_string()`</span> and <span class="pre">`currency()`</span>.

  The <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> function’s *val* parameter could previously be a string as long as no more than one %char specifier appeared; now the parameter must be exactly one %char specifier with no surrounding text. An optional *monetary* parameter was also added which, if <span class="pre">`True`</span>, will use the locale’s rules for formatting currency in placing a separator between groups of three digits.

  To format strings with multiple %char specifiers, use the new <span class="pre">`format_string()`</span> function that works like <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> but also supports mixing %char specifiers with arbitrary text.

  A new <span class="pre">`currency()`</span> function was also added that formats a number according to the current locale’s settings.

  (Contributed by Georg Brandl.)

- The <a href="../library/mailbox.html#module-mailbox" class="reference internal" title="mailbox: Manipulate mailboxes in various formats"><span class="pre"><code class="sourceCode python">mailbox</code></span></a> module underwent a massive rewrite to add the capability to modify mailboxes in addition to reading them. A new set of classes that include <span class="pre">`mbox`</span>, <span class="pre">`MH`</span>, and <span class="pre">`Maildir`</span> are used to read mailboxes, and have an <span class="pre">`add(message)`</span> method to add messages, <span class="pre">`remove(key)`</span> to remove messages, and <span class="pre">`lock()`</span>/<span class="pre">`unlock()`</span> to lock/unlock the mailbox. The following example converts a maildir-format mailbox into an mbox-format one:

  <div class="highlight-default notranslate">

  <div class="highlight">

      import mailbox

      # 'factory=None' uses email.Message.Message as the class representing
      # individual messages.
      src = mailbox.Maildir('maildir', factory=None)
      dest = mailbox.mbox('/tmp/mbox')

      for msg in src:
          dest.add(msg)

  </div>

  </div>

  (Contributed by Gregory K. Johnson. Funding was provided by Google’s 2005 Summer of Code.)

- New module: the <a href="../library/msilib.html#module-msilib" class="reference internal" title="msilib: Creation of Microsoft Installer files, and CAB files. (Windows)"><span class="pre"><code class="sourceCode python">msilib</code></span></a> module allows creating Microsoft Installer <span class="pre">`.msi`</span> files and CAB files. Some support for reading the <span class="pre">`.msi`</span> database is also included. (Contributed by Martin von Löwis.)

- The <a href="../library/nis.html#module-nis" class="reference internal" title="nis: Interface to Sun&#39;s NIS (Yellow Pages) library. (Unix)"><span class="pre"><code class="sourceCode python">nis</code></span></a> module now supports accessing domains other than the system default domain by supplying a *domain* argument to the <a href="../library/nis.html#nis.match" class="reference internal" title="nis.match"><span class="pre"><code class="sourceCode python">nis.match()</code></span></a> and <a href="../library/nis.html#nis.maps" class="reference internal" title="nis.maps"><span class="pre"><code class="sourceCode python">nis.maps()</code></span></a> functions. (Contributed by Ben Bell.)

- The <a href="../library/operator.html#module-operator" class="reference internal" title="operator: Functions corresponding to the standard operators."><span class="pre"><code class="sourceCode python">operator</code></span></a> module’s <span class="pre">`itemgetter()`</span> and <span class="pre">`attrgetter()`</span> functions now support multiple fields. A call such as <span class="pre">`operator.attrgetter('a',`</span>` `<span class="pre">`'b')`</span> will return a function that retrieves the <span class="pre">`a`</span> and <span class="pre">`b`</span> attributes. Combining this new feature with the <span class="pre">`sort()`</span> method’s <span class="pre">`key`</span> parameter lets you easily sort lists using multiple fields. (Contributed by Raymond Hettinger.)

- The <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a> module was updated to version 1.5.1 of the Optik library. The <span class="pre">`OptionParser`</span> class gained an <span class="pre">`epilog`</span> attribute, a string that will be printed after the help message, and a <span class="pre">`destroy()`</span> method to break reference cycles created by the object. (Contributed by Greg Ward.)

- The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module underwent several changes. The <span class="pre">`stat_float_times`</span> variable now defaults to true, meaning that <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> will now return time values as floats. (This doesn’t necessarily mean that <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> will return times that are precise to fractions of a second; not all systems support such precision.)

  Constants named <a href="../library/os.html#os.SEEK_SET" class="reference internal" title="os.SEEK_SET"><span class="pre"><code class="sourceCode python">os.SEEK_SET</code></span></a>, <a href="../library/os.html#os.SEEK_CUR" class="reference internal" title="os.SEEK_CUR"><span class="pre"><code class="sourceCode python">os.SEEK_CUR</code></span></a>, and <a href="../library/os.html#os.SEEK_END" class="reference internal" title="os.SEEK_END"><span class="pre"><code class="sourceCode python">os.SEEK_END</code></span></a> have been added; these are the parameters to the <a href="../library/os.html#os.lseek" class="reference internal" title="os.lseek"><span class="pre"><code class="sourceCode python">os.lseek()</code></span></a> function. Two new constants for locking are <a href="../library/os.html#os.O_SHLOCK" class="reference internal" title="os.O_SHLOCK"><span class="pre"><code class="sourceCode python">os.O_SHLOCK</code></span></a> and <a href="../library/os.html#os.O_EXLOCK" class="reference internal" title="os.O_EXLOCK"><span class="pre"><code class="sourceCode python">os.O_EXLOCK</code></span></a>.

  Two new functions, <span class="pre">`wait3()`</span> and <span class="pre">`wait4()`</span>, were added. They’re similar the <span class="pre">`waitpid()`</span> function which waits for a child process to exit and returns a tuple of the process ID and its exit status, but <span class="pre">`wait3()`</span> and <span class="pre">`wait4()`</span> return additional information. <span class="pre">`wait3()`</span> doesn’t take a process ID as input, so it waits for any child process to exit and returns a 3-tuple of *process-id*, *exit-status*, *resource-usage* as returned from the <a href="../library/resource.html#resource.getrusage" class="reference internal" title="resource.getrusage"><span class="pre"><code class="sourceCode python">resource.getrusage()</code></span></a> function. <span class="pre">`wait4(pid)`</span> does take a process ID. (Contributed by Chad J. Schroeder.)

  On FreeBSD, the <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> function now returns times with nanosecond resolution, and the returned object now has <span class="pre">`st_gen`</span> and <span class="pre">`st_birthtime`</span>. The <span class="pre">`st_flags`</span> attribute is also available, if the platform supports it. (Contributed by Antti Louko and Diego Pettenò.)

- The Python debugger provided by the <a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a> module can now store lists of commands to execute when a breakpoint is reached and execution stops. Once breakpoint \#1 has been created, enter <span class="pre">`commands`</span>` `<span class="pre">`1`</span> and enter a series of commands to be executed, finishing the list with <span class="pre">`end`</span>. The command list can include commands that resume execution, such as <span class="pre">`continue`</span> or <span class="pre">`next`</span>. (Contributed by Grégoire Dooms.)

- The <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> and <a href="../library/pickle.html#module-cPickle" class="reference internal" title="cPickle: Faster version of pickle, but not subclassable."><span class="pre"><code class="sourceCode python">cPickle</code></span></a> modules no longer accept a return value of <span class="pre">`None`</span> from the <a href="../library/pickle.html#object.__reduce__" class="reference internal" title="object.__reduce__"><span class="pre"><code class="sourceCode python">__reduce__()</code></span></a> method; the method must return a tuple of arguments instead. The ability to return <span class="pre">`None`</span> was deprecated in Python 2.4, so this completes the removal of the feature.

- The <a href="../library/pkgutil.html#module-pkgutil" class="reference internal" title="pkgutil: Utilities for the import system."><span class="pre"><code class="sourceCode python">pkgutil</code></span></a> module, containing various utility functions for finding packages, was enhanced to support PEP 302’s import hooks and now also works for packages stored in ZIP-format archives. (Contributed by Phillip J. Eby.)

- The pybench benchmark suite by Marc-André Lemburg is now included in the <span class="pre">`Tools/pybench`</span> directory. The pybench suite is an improvement on the commonly used <span class="pre">`pystone.py`</span> program because pybench provides a more detailed measurement of the interpreter’s speed. It times particular operations such as function calls, tuple slicing, method lookups, and numeric operations, instead of performing many different operations and reducing the result to a single number as <span class="pre">`pystone.py`</span> does.

- The <span class="pre">`pyexpat`</span> module now uses version 2.0 of the Expat parser. (Contributed by Trent Mick.)

- The <a href="../library/queue.html#Queue.Queue" class="reference internal" title="Queue.Queue"><span class="pre"><code class="sourceCode python">Queue</code></span></a> class provided by the <a href="../library/queue.html#module-Queue" class="reference internal" title="Queue: A synchronized queue class."><span class="pre"><code class="sourceCode python">Queue</code></span></a> module gained two new methods. <span class="pre">`join()`</span> blocks until all items in the queue have been retrieved and all processing work on the items have been completed. Worker threads call the other new method, <span class="pre">`task_done()`</span>, to signal that processing for an item has been completed. (Contributed by Raymond Hettinger.)

- The old <span class="pre">`regex`</span> and <span class="pre">`regsub`</span> modules, which have been deprecated ever since Python 2.0, have finally been deleted. Other deleted modules: <span class="pre">`statcache`</span>, <span class="pre">`tzparse`</span>, <span class="pre">`whrandom`</span>.

- Also deleted: the <span class="pre">`lib-old`</span> directory, which includes ancient modules such as <span class="pre">`dircmp`</span> and <span class="pre">`ni`</span>, was removed. <span class="pre">`lib-old`</span> wasn’t on the default <span class="pre">`sys.path`</span>, so unless your programs explicitly added the directory to <span class="pre">`sys.path`</span>, this removal shouldn’t affect your code.

- The <a href="../library/rlcompleter.html#module-rlcompleter" class="reference internal" title="rlcompleter: Python identifier completion, suitable for the GNU readline library."><span class="pre"><code class="sourceCode python">rlcompleter</code></span></a> module is no longer dependent on importing the <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline</code></span></a> module and therefore now works on non-Unix platforms. (Patch from Robert Kiendl.)

- The <a href="../library/simplexmlrpcserver.html#module-SimpleXMLRPCServer" class="reference internal" title="SimpleXMLRPCServer: Basic XML-RPC server implementation."><span class="pre"><code class="sourceCode python">SimpleXMLRPCServer</code></span></a> and <a href="../library/docxmlrpcserver.html#module-DocXMLRPCServer" class="reference internal" title="DocXMLRPCServer: Self-documenting XML-RPC server implementation."><span class="pre"><code class="sourceCode python">DocXMLRPCServer</code></span></a> classes now have a <span class="pre">`rpc_paths`</span> attribute that constrains XML-RPC operations to a limited set of URL paths; the default is to allow only <span class="pre">`'/'`</span> and <span class="pre">`'/RPC2'`</span>. Setting <span class="pre">`rpc_paths`</span> to <span class="pre">`None`</span> or an empty tuple disables this path checking.

- The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module now supports <span class="pre">`AF_NETLINK`</span> sockets on Linux, thanks to a patch from Philippe Biondi. Netlink sockets are a Linux-specific mechanism for communications between a user-space process and kernel code; an introductory article about them is at <a href="https://www.linuxjournal.com/article/7356" class="reference external">https://www.linuxjournal.com/article/7356</a>. In Python code, netlink addresses are represented as a tuple of 2 integers, <span class="pre">`(pid,`</span>` `<span class="pre">`group_mask)`</span>.

  Two new methods on socket objects, <span class="pre">`recv_into(buffer)`</span> and <span class="pre">`recvfrom_into(buffer)`</span>, store the received data in an object that supports the buffer protocol instead of returning the data as a string. This means you can put the data directly into an array or a memory-mapped file.

  Socket objects also gained <span class="pre">`getfamily()`</span>, <span class="pre">`gettype()`</span>, and <span class="pre">`getproto()`</span> accessor methods to retrieve the family, type, and protocol values for the socket.

- New module: the <a href="../library/spwd.html#module-spwd" class="reference internal" title="spwd: The shadow password database (getspnam() and friends). (Unix)"><span class="pre"><code class="sourceCode python">spwd</code></span></a> module provides functions for accessing the shadow password database on systems that support shadow passwords.

- The <a href="../library/struct.html#module-struct" class="reference internal" title="struct: Interpret strings as packed binary data."><span class="pre"><code class="sourceCode python">struct</code></span></a> is now faster because it compiles format strings into <span class="pre">`Struct`</span> objects with <span class="pre">`pack()`</span> and <span class="pre">`unpack()`</span> methods. This is similar to how the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module lets you create compiled regular expression objects. You can still use the module-level <span class="pre">`pack()`</span> and <span class="pre">`unpack()`</span> functions; they’ll create <span class="pre">`Struct`</span> objects and cache them. Or you can use <span class="pre">`Struct`</span> instances directly:

  <div class="highlight-default notranslate">

  <div class="highlight">

      s = struct.Struct('ih3s')

      data = s.pack(1972, 187, 'abc')
      year, number, name = s.unpack(data)

  </div>

  </div>

  You can also pack and unpack data to and from buffer objects directly using the <span class="pre">`pack_into(buffer,`</span>` `<span class="pre">`offset,`</span>` `<span class="pre">`v1,`</span>` `<span class="pre">`v2,`</span>` `<span class="pre">`...)`</span> and <span class="pre">`unpack_from(buffer,`</span>` `<span class="pre">`offset)`</span> methods. This lets you store data directly into an array or a memory-mapped file.

  (<span class="pre">`Struct`</span> objects were implemented by Bob Ippolito at the NeedForSpeed sprint. Support for buffer objects was added by Martin Blais, also at the NeedForSpeed sprint.)

- The Python developers switched from CVS to Subversion during the 2.5 development process. Information about the exact build version is available as the <span class="pre">`sys.subversion`</span> variable, a 3-tuple of <span class="pre">`(interpreter-name,`</span>` `<span class="pre">`branch-name,`</span>` `<span class="pre">`revision-range)`</span>. For example, at the time of writing my copy of 2.5 was reporting <span class="pre">`('CPython',`</span>` `<span class="pre">`'trunk',`</span>` `<span class="pre">`'45313:45315')`</span>.

  This information is also available to C extensions via the <a href="../c-api/init.html#c.Py_GetBuildInfo" class="reference internal" title="Py_GetBuildInfo"><span class="pre"><code class="sourceCode c">Py_GetBuildInfo<span class="op">()</span></code></span></a> function that returns a string of build information like this: <span class="pre">`"trunk:45355:45356M,`</span>` `<span class="pre">`Apr`</span>` `<span class="pre">`13`</span>` `<span class="pre">`2006,`</span>` `<span class="pre">`07:42:19"`</span>. (Contributed by Barry Warsaw.)

- Another new function, <a href="../library/sys.html#sys._current_frames" class="reference internal" title="sys._current_frames"><span class="pre"><code class="sourceCode python">sys._current_frames()</code></span></a>, returns the current stack frames for all running threads as a dictionary mapping thread identifiers to the topmost stack frame currently active in that thread at the time the function is called. (Contributed by Tim Peters.)

- The <span class="pre">`TarFile`</span> class in the <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module now has an <span class="pre">`extractall()`</span> method that extracts all members from the archive into the current working directory. It’s also possible to set a different directory as the extraction target, and to unpack only a subset of the archive’s members.

  The compression used for a tarfile opened in stream mode can now be autodetected using the mode <span class="pre">`'r|*'`</span>. (Contributed by Lars Gustäbel.)

- The <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> module now lets you set the stack size used when new threads are created. The <span class="pre">`stack_size([*size*])`</span> function returns the currently configured stack size, and supplying the optional *size* parameter sets a new value. Not all platforms support changing the stack size, but Windows, POSIX threading, and OS/2 all do. (Contributed by Andrew MacIntyre.)

- The <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a> module has been updated to use version 4.1.0 of the Unicode character database. Version 3.2.0 is required by some specifications, so it’s still available as <a href="../library/unicodedata.html#unicodedata.ucd_3_2_0" class="reference internal" title="unicodedata.ucd_3_2_0"><span class="pre"><code class="sourceCode python">unicodedata.ucd_3_2_0</code></span></a>.

- New module: the <a href="../library/uuid.html#module-uuid" class="reference internal" title="uuid: UUID objects (universally unique identifiers) according to RFC 4122"><span class="pre"><code class="sourceCode python">uuid</code></span></a> module generates universally unique identifiers (UUIDs) according to <span id="index-19" class="target"></span><a href="https://tools.ietf.org/html/rfc4122.html" class="rfc reference external"><strong>RFC 4122</strong></a>. The RFC defines several different UUID versions that are generated from a starting string, from system properties, or purely randomly. This module contains a <span class="pre">`UUID`</span> class and functions named <span class="pre">`uuid1()`</span>, <span class="pre">`uuid3()`</span>, <span class="pre">`uuid4()`</span>, and <span class="pre">`uuid5()`</span> to generate different versions of UUID. (Version 2 UUIDs are not specified in <span id="index-20" class="target"></span><a href="https://tools.ietf.org/html/rfc4122.html" class="rfc reference external"><strong>RFC 4122</strong></a> and are not supported by this module.)

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import uuid
      >>> # make a UUID based on the host ID and current time
      >>> uuid.uuid1()
      UUID('a8098c1a-f86e-11da-bd1a-00112444be1e')

      >>> # make a UUID using an MD5 hash of a namespace UUID and a name
      >>> uuid.uuid3(uuid.NAMESPACE_DNS, 'python.org')
      UUID('6fa459ea-ee8a-3ca4-894e-db77e160355e')

      >>> # make a random UUID
      >>> uuid.uuid4()
      UUID('16fd2706-8baf-433b-82eb-8c7fada847da')

      >>> # make a UUID using a SHA-1 hash of a namespace UUID and a name
      >>> uuid.uuid5(uuid.NAMESPACE_DNS, 'python.org')
      UUID('886313e1-3b8a-5372-9b90-0c9aee199e5d')

  </div>

  </div>

  (Contributed by Ka-Ping Yee.)

- The <a href="../library/weakref.html#module-weakref" class="reference internal" title="weakref: Support for weak references and weak dictionaries."><span class="pre"><code class="sourceCode python">weakref</code></span></a> module’s <span class="pre">`WeakKeyDictionary`</span> and <span class="pre">`WeakValueDictionary`</span> types gained new methods for iterating over the weak references contained in the dictionary. <span class="pre">`iterkeyrefs()`</span> and <span class="pre">`keyrefs()`</span> methods were added to <span class="pre">`WeakKeyDictionary`</span>, and <span class="pre">`itervaluerefs()`</span> and <span class="pre">`valuerefs()`</span> were added to <span class="pre">`WeakValueDictionary`</span>. (Contributed by Fred L. Drake, Jr.)

- The <a href="../library/webbrowser.html#module-webbrowser" class="reference internal" title="webbrowser: Easy-to-use controller for Web browsers."><span class="pre"><code class="sourceCode python">webbrowser</code></span></a> module received a number of enhancements. It’s now usable as a script with <span class="pre">`python`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`webbrowser`</span>, taking a URL as the argument; there are a number of switches to control the behaviour (<span class="pre">`-n`</span> for a new browser window, <span class="pre">`-t`</span> for a new tab). New module-level functions, <span class="pre">`open_new()`</span> and <span class="pre">`open_new_tab()`</span>, were added to support this. The module’s <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> function supports an additional feature, an *autoraise* parameter that signals whether to raise the open window when possible. A number of additional browsers were added to the supported list such as Firefox, Opera, Konqueror, and elinks. (Contributed by Oleg Broytmann and Georg Brandl.)

- The <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a> module now supports returning <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> objects for the XML-RPC date type. Supply <span class="pre">`use_datetime=True`</span> to the <span class="pre">`loads()`</span> function or the <span class="pre">`Unmarshaller`</span> class to enable this feature. (Contributed by Skip Montanaro.)

- The <a href="../library/zipfile.html#module-zipfile" class="reference internal" title="zipfile: Read and write ZIP-format archive files."><span class="pre"><code class="sourceCode python">zipfile</code></span></a> module now supports the ZIP64 version of the format, meaning that a .zip archive can now be larger than 4 GiB and can contain individual files larger than 4 GiB. (Contributed by Ronald Oussoren.)

- The <a href="../library/zlib.html#module-zlib" class="reference internal" title="zlib: Low-level interface to compression and decompression routines compatible with gzip."><span class="pre"><code class="sourceCode python">zlib</code></span></a> module’s <span class="pre">`Compress`</span> and <span class="pre">`Decompress`</span> objects now support a <a href="../library/copy.html#module-copy" class="reference internal" title="copy: Shallow and deep copy operations."><span class="pre"><code class="sourceCode python">copy()</code></span></a> method that makes a copy of the object’s internal state and returns a new <span class="pre">`Compress`</span> or <span class="pre">`Decompress`</span> object. (Contributed by Chris AtLee.)

<div id="the-ctypes-package" class="section">

<span id="module-ctypes"></span>

### The ctypes package<a href="#the-ctypes-package" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> package, written by Thomas Heller, has been added to the standard library. <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> lets you call arbitrary functions in shared libraries or DLLs. Long-time users may remember the <a href="../library/dl.html#module-dl" class="reference internal" title="dl: Call C functions in shared objects. (deprecated) (Unix)"><span class="pre"><code class="sourceCode python">dl</code></span></a> module, which provides functions for loading shared libraries and calling functions in them. The <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> package is much fancier.

To load a shared library or DLL, you must create an instance of the <span class="pre">`CDLL`</span> class and provide the name or path of the shared library or DLL. Once that’s done, you can call arbitrary functions by accessing them as attributes of the <span class="pre">`CDLL`</span> object.

<div class="highlight-default notranslate">

<div class="highlight">

    import ctypes

    libc = ctypes.CDLL('libc.so.6')
    result = libc.printf("Line of output\n")

</div>

</div>

Type constructors for the various C types are provided: <span class="pre">`c_int()`</span>, <span class="pre">`c_float()`</span>, <span class="pre">`c_double()`</span>, <span class="pre">`c_char_p()`</span> (equivalent to <span class="pre">`char`</span>` `<span class="pre">`*`</span>), and so forth. Unlike Python’s types, the C versions are all mutable; you can assign to their <span class="pre">`value`</span> attribute to change the wrapped value. Python integers and strings will be automatically converted to the corresponding C types, but for other types you must call the correct type constructor. (And I mean *must*; getting it wrong will often result in the interpreter crashing with a segmentation fault.)

You shouldn’t use <span class="pre">`c_char_p()`</span> with a Python string when the C function will be modifying the memory area, because Python strings are supposed to be immutable; breaking this rule will cause puzzling bugs. When you need a modifiable memory area, use <span class="pre">`create_string_buffer()`</span>:

<div class="highlight-default notranslate">

<div class="highlight">

    s = "this is a string"
    buf = ctypes.create_string_buffer(s)
    libc.strfry(buf)

</div>

</div>

C functions are assumed to return integers, but you can set the <span class="pre">`restype`</span> attribute of the function object to change this:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> libc.atof('2.71828')
    -1783957616
    >>> libc.atof.restype = ctypes.c_double
    >>> libc.atof('2.71828')
    2.71828

</div>

</div>

<a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> also provides a wrapper for Python’s C API as the <span class="pre">`ctypes.pythonapi`</span> object. This object does *not* release the global interpreter lock before calling a function, because the lock must be held when calling into the interpreter’s code. There’s a <span class="pre">`py_object()`</span> type constructor that will create a <a href="../c-api/structures.html#c.PyObject" class="reference internal" title="PyObject"><span class="pre"><code class="sourceCode c">PyObject</code></span><code class="sourceCode c"> </code><span class="pre"><code class="sourceCode c"><span class="op">*</span></code></span></a> pointer. A simple usage:

<div class="highlight-default notranslate">

<div class="highlight">

    import ctypes

    d = {}
    ctypes.pythonapi.PyObject_SetItem(ctypes.py_object(d),
              ctypes.py_object("abc"),  ctypes.py_object(1))
    # d is now {'abc', 1}.

</div>

</div>

Don’t forget to use <span class="pre">`py_object()`</span>; if it’s omitted you end up with a segmentation fault.

<a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> has been around for a while, but people still write and distribution hand-coded extension modules because you can’t rely on <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> being present. Perhaps developers will begin to write Python wrappers atop a library accessed through <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> instead of extension modules, now that <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> is included with core Python.

<div class="admonition seealso">

See also

<a href="http://starship.python.net/crew/theller/ctypes/" class="reference external">http://starship.python.net/crew/theller/ctypes/</a>  
The ctypes web page, with a tutorial, reference, and FAQ.

The documentation for the <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> module.

</div>

</div>

<div id="the-elementtree-package" class="section">

<span id="module-etree"></span>

### The ElementTree package<a href="#the-elementtree-package" class="headerlink" title="Permalink to this headline">¶</a>

A subset of Fredrik Lundh’s ElementTree library for processing XML has been added to the standard library as <span class="pre">`xml.etree`</span>. The available modules are <span class="pre">`ElementTree`</span>, <span class="pre">`ElementPath`</span>, and <span class="pre">`ElementInclude`</span> from ElementTree 1.2.6. The <span class="pre">`cElementTree`</span> accelerator module is also included.

The rest of this section will provide a brief overview of using ElementTree. Full documentation for ElementTree is available at <a href="http://effbot.org/zone/element-index.htm" class="reference external">http://effbot.org/zone/element-index.htm</a>.

ElementTree represents an XML document as a tree of element nodes. The text content of the document is stored as the <span class="pre">`text`</span> and <span class="pre">`tail`</span> attributes of (This is one of the major differences between ElementTree and the Document Object Model; in the DOM there are many different types of node, including <span class="pre">`TextNode`</span>.)

The most commonly used parsing function is <span class="pre">`parse()`</span>, that takes either a string (assumed to contain a filename) or a file-like object and returns an <span class="pre">`ElementTree`</span> instance:

<div class="highlight-default notranslate">

<div class="highlight">

    from xml.etree import ElementTree as ET

    tree = ET.parse('ex-1.xml')

    feed = urllib.urlopen(
              'http://planet.python.org/rss10.xml')
    tree = ET.parse(feed)

</div>

</div>

Once you have an <span class="pre">`ElementTree`</span> instance, you can call its <span class="pre">`getroot()`</span> method to get the root <span class="pre">`Element`</span> node.

There’s also an <span class="pre">`XML()`</span> function that takes a string literal and returns an <span class="pre">`Element`</span> node (not an <span class="pre">`ElementTree`</span>). This function provides a tidy way to incorporate XML fragments, approaching the convenience of an XML literal:

<div class="highlight-default notranslate">

<div class="highlight">

    svg = ET.XML("""<svg width="10px" version="1.0">
                 </svg>""")
    svg.set('height', '320px')
    svg.append(elem1)

</div>

</div>

Each XML element supports some dictionary-like and some list-like access methods. Dictionary-like operations are used to access attribute values, and list-like operations are used to access child nodes.

| Operation | Result |
|----|----|
| <span class="pre">`elem[n]`</span> | Returns n’th child element. |
| <span class="pre">`elem[m:n]`</span> | Returns list of m’th through n’th child elements. |
| <span class="pre">`len(elem)`</span> | Returns number of child elements. |
| <span class="pre">`list(elem)`</span> | Returns list of child elements. |
| <span class="pre">`elem.append(elem2)`</span> | Adds *elem2* as a child. |
| <span class="pre">`elem.insert(index,`</span>` `<span class="pre">`elem2)`</span> | Inserts *elem2* at the specified location. |
| <span class="pre">`del`</span>` `<span class="pre">`elem[n]`</span> | Deletes n’th child element. |
| <span class="pre">`elem.keys()`</span> | Returns list of attribute names. |
| <span class="pre">`elem.get(name)`</span> | Returns value of attribute *name*. |
| <span class="pre">`elem.set(name,`</span>` `<span class="pre">`value)`</span> | Sets new value for attribute *name*. |
| <span class="pre">`elem.attrib`</span> | Retrieves the dictionary containing attributes. |
| <span class="pre">`del`</span>` `<span class="pre">`elem.attrib[name]`</span> | Deletes attribute *name*. |

Comments and processing instructions are also represented as <span class="pre">`Element`</span> nodes. To check if a node is a comment or processing instructions:

<div class="highlight-default notranslate">

<div class="highlight">

    if elem.tag is ET.Comment:
        ...
    elif elem.tag is ET.ProcessingInstruction:
        ...

</div>

</div>

To generate XML output, you should call the <span class="pre">`ElementTree.write()`</span> method. Like <span class="pre">`parse()`</span>, it can take either a string or a file-like object:

<div class="highlight-default notranslate">

<div class="highlight">

    # Encoding is US-ASCII
    tree.write('output.xml')

    # Encoding is UTF-8
    f = open('output.xml', 'w')
    tree.write(f, encoding='utf-8')

</div>

</div>

(Caution: the default encoding used for output is ASCII. For general XML work, where an element’s name may contain arbitrary Unicode characters, ASCII isn’t a very useful encoding because it will raise an exception if an element’s name contains any characters with values greater than 127. Therefore, it’s best to specify a different encoding such as UTF-8 that can handle any Unicode character.)

This section is only a partial description of the ElementTree interfaces. Please read the package’s official documentation for more details.

<div class="admonition seealso">

See also

<a href="http://effbot.org/zone/element-index.htm" class="reference external">http://effbot.org/zone/element-index.htm</a>  
Official documentation for ElementTree.

</div>

</div>

<div id="the-hashlib-package" class="section">

<span id="module-hashlib"></span>

### The hashlib package<a href="#the-hashlib-package" class="headerlink" title="Permalink to this headline">¶</a>

A new <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module, written by Gregory P. Smith, has been added to replace the <a href="../library/md5.html#module-md5" class="reference internal" title="md5: RSA&#39;s MD5 message digest algorithm. (deprecated)"><span class="pre"><code class="sourceCode python">md5</code></span></a> and <a href="../library/sha.html#module-sha" class="reference internal" title="sha: NIST&#39;s secure hash algorithm, SHA. (deprecated)"><span class="pre"><code class="sourceCode python">sha</code></span></a> modules. <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> adds support for additional secure hashes (SHA-224, SHA-256, SHA-384, and SHA-512). When available, the module uses OpenSSL for fast platform optimized implementations of algorithms.

The old <a href="../library/md5.html#module-md5" class="reference internal" title="md5: RSA&#39;s MD5 message digest algorithm. (deprecated)"><span class="pre"><code class="sourceCode python">md5</code></span></a> and <a href="../library/sha.html#module-sha" class="reference internal" title="sha: NIST&#39;s secure hash algorithm, SHA. (deprecated)"><span class="pre"><code class="sourceCode python">sha</code></span></a> modules still exist as wrappers around hashlib to preserve backwards compatibility. The new module’s interface is very close to that of the old modules, but not identical. The most significant difference is that the constructor functions for creating new hashing objects are named differently.

<div class="highlight-default notranslate">

<div class="highlight">

    # Old versions
    h = md5.md5()
    h = md5.new()

    # New version
    h = hashlib.md5()

    # Old versions
    h = sha.sha()
    h = sha.new()

    # New version
    h = hashlib.sha1()

    # Hash that weren't previously available
    h = hashlib.sha224()
    h = hashlib.sha256()
    h = hashlib.sha384()
    h = hashlib.sha512()

    # Alternative form
    h = hashlib.new('md5')          # Provide algorithm as a string

</div>

</div>

Once a hash object has been created, its methods are the same as before: <span class="pre">`update(string)`</span> hashes the specified string into the current digest state, <span class="pre">`digest()`</span> and <span class="pre">`hexdigest()`</span> return the digest value as a binary string or a string of hex digits, and <a href="../library/copy.html#module-copy" class="reference internal" title="copy: Shallow and deep copy operations."><span class="pre"><code class="sourceCode python">copy()</code></span></a> returns a new hashing object with the same digest state.

<div class="admonition seealso">

See also

The documentation for the <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module.

</div>

</div>

<div id="the-sqlite3-package" class="section">

<span id="module-sqlite"></span>

### The sqlite3 package<a href="#the-sqlite3-package" class="headerlink" title="Permalink to this headline">¶</a>

The pysqlite module (<a href="http://www.pysqlite.org" class="reference external">http://www.pysqlite.org</a>), a wrapper for the SQLite embedded database, has been added to the standard library under the package name <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a>.

SQLite is a C library that provides a lightweight disk-based database that doesn’t require a separate server process and allows accessing the database using a nonstandard variant of the SQL query language. Some applications can use SQLite for internal data storage. It’s also possible to prototype an application using SQLite and then port the code to a larger database such as PostgreSQL or Oracle.

pysqlite was written by Gerhard Häring and provides a SQL interface compliant with the DB-API 2.0 specification described by <span id="index-21" class="target"></span><a href="https://www.python.org/dev/peps/pep-0249" class="pep reference external"><strong>PEP 249</strong></a>.

If you’re compiling the Python source yourself, note that the source tree doesn’t include the SQLite code, only the wrapper module. You’ll need to have the SQLite libraries and headers installed before compiling Python, and the build process will compile the module when the necessary headers are available.

To use the module, you must first create a <a href="../library/multiprocessing.html#Connection" class="reference internal" title="Connection"><span class="pre"><code class="sourceCode python">Connection</code></span></a> object that represents the database. Here the data will be stored in the <span class="pre">`/tmp/example`</span> file:

<div class="highlight-default notranslate">

<div class="highlight">

    conn = sqlite3.connect('/tmp/example')

</div>

</div>

You can also supply the special name <span class="pre">`:memory:`</span> to create a database in RAM.

Once you have a <a href="../library/multiprocessing.html#Connection" class="reference internal" title="Connection"><span class="pre"><code class="sourceCode python">Connection</code></span></a>, you can create a <span class="pre">`Cursor`</span> object and call its <span class="pre">`execute()`</span> method to perform SQL commands:

<div class="highlight-default notranslate">

<div class="highlight">

    c = conn.cursor()

    # Create table
    c.execute('''create table stocks
    (date text, trans text, symbol text,
     qty real, price real)''')

    # Insert a row of data
    c.execute("""insert into stocks
              values ('2006-01-05','BUY','RHAT',100,35.14)""")

</div>

</div>

Usually your SQL operations will need to use values from Python variables. You shouldn’t assemble your query using Python’s string operations because doing so is insecure; it makes your program vulnerable to an SQL injection attack.

Instead, use the DB-API’s parameter substitution. Put <span class="pre">`?`</span> as a placeholder wherever you want to use a value, and then provide a tuple of values as the second argument to the cursor’s <span class="pre">`execute()`</span> method. (Other database modules may use a different placeholder, such as <span class="pre">`%s`</span> or <span class="pre">`:1`</span>.) For example:

<div class="highlight-default notranslate">

<div class="highlight">

    # Never do this -- insecure!
    symbol = 'IBM'
    c.execute("... where symbol = '%s'" % symbol)

    # Do this instead
    t = (symbol,)
    c.execute('select * from stocks where symbol=?', t)

    # Larger example
    for t in (('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
              ('2006-04-05', 'BUY', 'MSOFT', 1000, 72.00),
              ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
             ):
        c.execute('insert into stocks values (?,?,?,?,?)', t)

</div>

</div>

To retrieve data after executing a SELECT statement, you can either treat the cursor as an iterator, call the cursor’s <span class="pre">`fetchone()`</span> method to retrieve a single matching row, or call <span class="pre">`fetchall()`</span> to get a list of the matching rows.

This example uses the iterator form:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> c = conn.cursor()
    >>> c.execute('select * from stocks order by price')
    >>> for row in c:
    ...    print row
    ...
    (u'2006-01-05', u'BUY', u'RHAT', 100, 35.140000000000001)
    (u'2006-03-28', u'BUY', u'IBM', 1000, 45.0)
    (u'2006-04-06', u'SELL', u'IBM', 500, 53.0)
    (u'2006-04-05', u'BUY', u'MSOFT', 1000, 72.0)
    >>>

</div>

</div>

For more information about the SQL dialect supported by SQLite, see <a href="https://www.sqlite.org" class="reference external">https://www.sqlite.org</a>.

<div class="admonition seealso">

See also

<a href="http://www.pysqlite.org" class="reference external">http://www.pysqlite.org</a>  
The pysqlite web page.

<a href="https://www.sqlite.org" class="reference external">https://www.sqlite.org</a>  
The SQLite web page; the documentation describes the syntax and the available data types for the supported SQL dialect.

The documentation for the <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> module.

<span id="index-22" class="target"></span><a href="https://www.python.org/dev/peps/pep-0249" class="pep reference external"><strong>PEP 249</strong></a> - Database API Specification 2.0  
PEP written by Marc-André Lemburg.

</div>

</div>

<div id="the-wsgiref-package" class="section">

<span id="module-wsgiref"></span>

### The wsgiref package<a href="#the-wsgiref-package" class="headerlink" title="Permalink to this headline">¶</a>

The Web Server Gateway Interface (WSGI) v1.0 defines a standard interface between web servers and Python web applications and is described in <span id="index-23" class="target"></span><a href="https://www.python.org/dev/peps/pep-0333" class="pep reference external"><strong>PEP 333</strong></a>. The <a href="../library/wsgiref.html#module-wsgiref" class="reference internal" title="wsgiref: WSGI Utilities and Reference Implementation."><span class="pre"><code class="sourceCode python">wsgiref</code></span></a> package is a reference implementation of the WSGI specification.

The package includes a basic HTTP server that will run a WSGI application; this server is useful for debugging but isn’t intended for production use. Setting up a server takes only a few lines of code:

<div class="highlight-default notranslate">

<div class="highlight">

    from wsgiref import simple_server

    wsgi_app = ...

    host = ''
    port = 8000
    httpd = simple_server.make_server(host, port, wsgi_app)
    httpd.serve_forever()

</div>

</div>

<div class="admonition seealso">

See also

<a href="http://www.wsgi.org" class="reference external">http://www.wsgi.org</a>  
A central web site for WSGI-related resources.

<span id="index-24" class="target"></span><a href="https://www.python.org/dev/peps/pep-0333" class="pep reference external"><strong>PEP 333</strong></a> - Python Web Server Gateway Interface v1.0  
PEP written by Phillip J. Eby.

</div>

</div>

</div>

<div id="build-and-c-api-changes" class="section">

<span id="build-api"></span>

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Permalink to this headline">¶</a>

Changes to Python’s build process and to the C API include:

- The Python source tree was converted from CVS to Subversion, in a complex migration procedure that was supervised and flawlessly carried out by Martin von Löwis. The procedure was developed as <span id="index-25" class="target"></span><a href="https://www.python.org/dev/peps/pep-0347" class="pep reference external"><strong>PEP 347</strong></a>.

- Coverity, a company that markets a source code analysis tool called Prevent, provided the results of their examination of the Python source code. The analysis found about 60 bugs that were quickly fixed. Many of the bugs were refcounting problems, often occurring in error-handling code. See <a href="https://scan.coverity.com" class="reference external">https://scan.coverity.com</a> for the statistics.

- The largest change to the C API came from <span id="index-26" class="target"></span><a href="https://www.python.org/dev/peps/pep-0353" class="pep reference external"><strong>PEP 353</strong></a>, which modifies the interpreter to use a <span class="pre">`Py_ssize_t`</span> type definition instead of <span class="pre">`int`</span>. See the earlier section <a href="#pep-353" class="reference internal"><span class="std std-ref">PEP 353: Using ssize_t as the index type</span></a> for a discussion of this change.

- The design of the bytecode compiler has changed a great deal, no longer generating bytecode by traversing the parse tree. Instead the parse tree is converted to an abstract syntax tree (or AST), and it is the abstract syntax tree that’s traversed to produce the bytecode.

  It’s possible for Python code to obtain AST objects by using the <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a> built-in and specifying <span class="pre">`_ast.PyCF_ONLY_AST`</span> as the value of the *flags* parameter:

  <div class="highlight-default notranslate">

  <div class="highlight">

      from _ast import PyCF_ONLY_AST
      ast = compile("""a=0
      for i in range(10):
          a += i
      """, "<string>", 'exec', PyCF_ONLY_AST)

      assignment = ast.body[0]
      for_loop = ast.body[1]

  </div>

  </div>

  No official documentation has been written for the AST code yet, but <span id="index-27" class="target"></span><a href="https://www.python.org/dev/peps/pep-0339" class="pep reference external"><strong>PEP 339</strong></a> discusses the design. To start learning about the code, read the definition of the various AST nodes in <span class="pre">`Parser/Python.asdl`</span>. A Python script reads this file and generates a set of C structure definitions in <span class="pre">`Include/Python-ast.h`</span>. The <span class="pre">`PyParser_ASTFromString()`</span> and <span class="pre">`PyParser_ASTFromFile()`</span>, defined in <span class="pre">`Include/pythonrun.h`</span>, take Python source as input and return the root of an AST representing the contents. This AST can then be turned into a code object by <span class="pre">`PyAST_Compile()`</span>. For more information, read the source code, and then ask questions on python-dev.

  The AST code was developed under Jeremy Hylton’s management, and implemented by (in alphabetical order) Brett Cannon, Nick Coghlan, Grant Edwards, John Ehresman, Kurt Kaiser, Neal Norwitz, Tim Peters, Armin Rigo, and Neil Schemenauer, plus the participants in a number of AST sprints at conferences such as PyCon.

- Evan Jones’s patch to obmalloc, first described in a talk at PyCon DC 2005, was applied. Python 2.4 allocated small objects in 256K-sized arenas, but never freed arenas. With this patch, Python will free arenas when they’re empty. The net effect is that on some platforms, when you allocate many objects, Python’s memory usage may actually drop when you delete them and the memory may be returned to the operating system. (Implemented by Evan Jones, and reworked by Tim Peters.)

  Note that this change means extension modules must be more careful when allocating memory. Python’s API has many different functions for allocating memory that are grouped into families. For example, <a href="../c-api/memory.html#c.PyMem_Malloc" class="reference internal" title="PyMem_Malloc"><span class="pre"><code class="sourceCode c">PyMem_Malloc<span class="op">()</span></code></span></a>, <a href="../c-api/memory.html#c.PyMem_Realloc" class="reference internal" title="PyMem_Realloc"><span class="pre"><code class="sourceCode c">PyMem_Realloc<span class="op">()</span></code></span></a>, and <a href="../c-api/memory.html#c.PyMem_Free" class="reference internal" title="PyMem_Free"><span class="pre"><code class="sourceCode c">PyMem_Free<span class="op">()</span></code></span></a> are one family that allocates raw memory, while <a href="../c-api/memory.html#c.PyObject_Malloc" class="reference internal" title="PyObject_Malloc"><span class="pre"><code class="sourceCode c">PyObject_Malloc<span class="op">()</span></code></span></a>, <a href="../c-api/memory.html#c.PyObject_Realloc" class="reference internal" title="PyObject_Realloc"><span class="pre"><code class="sourceCode c">PyObject_Realloc<span class="op">()</span></code></span></a>, and <a href="../c-api/memory.html#c.PyObject_Free" class="reference internal" title="PyObject_Free"><span class="pre"><code class="sourceCode c">PyObject_Free<span class="op">()</span></code></span></a> are another family that’s supposed to be used for creating Python objects.

  Previously these different families all reduced to the platform’s <span class="pre">`malloc()`</span> and <span class="pre">`free()`</span> functions. This meant it didn’t matter if you got things wrong and allocated memory with the <span class="pre">`PyMem()`</span> function but freed it with the <a href="../c-api/structures.html#c.PyObject" class="reference internal" title="PyObject"><span class="pre"><code class="sourceCode c">PyObject<span class="op">()</span></code></span></a> function. With 2.5’s changes to obmalloc, these families now do different things and mismatches will probably result in a segfault. You should carefully test your C extension modules with Python 2.5.

- The built-in set types now have an official C API. Call <a href="../c-api/set.html#c.PySet_New" class="reference internal" title="PySet_New"><span class="pre"><code class="sourceCode c">PySet_New<span class="op">()</span></code></span></a> and <a href="../c-api/set.html#c.PyFrozenSet_New" class="reference internal" title="PyFrozenSet_New"><span class="pre"><code class="sourceCode c">PyFrozenSet_New<span class="op">()</span></code></span></a> to create a new set, <a href="../c-api/set.html#c.PySet_Add" class="reference internal" title="PySet_Add"><span class="pre"><code class="sourceCode c">PySet_Add<span class="op">()</span></code></span></a> and <a href="../c-api/set.html#c.PySet_Discard" class="reference internal" title="PySet_Discard"><span class="pre"><code class="sourceCode c">PySet_Discard<span class="op">()</span></code></span></a> to add and remove elements, and <a href="../c-api/set.html#c.PySet_Contains" class="reference internal" title="PySet_Contains"><span class="pre"><code class="sourceCode c">PySet_Contains<span class="op">()</span></code></span></a> and <a href="../c-api/set.html#c.PySet_Size" class="reference internal" title="PySet_Size"><span class="pre"><code class="sourceCode c">PySet_Size<span class="op">()</span></code></span></a> to examine the set’s state. (Contributed by Raymond Hettinger.)

- C code can now obtain information about the exact revision of the Python interpreter by calling the <a href="../c-api/init.html#c.Py_GetBuildInfo" class="reference internal" title="Py_GetBuildInfo"><span class="pre"><code class="sourceCode c">Py_GetBuildInfo<span class="op">()</span></code></span></a> function that returns a string of build information like this: <span class="pre">`"trunk:45355:45356M,`</span>` `<span class="pre">`Apr`</span>` `<span class="pre">`13`</span>` `<span class="pre">`2006,`</span>` `<span class="pre">`07:42:19"`</span>. (Contributed by Barry Warsaw.)

- Two new macros can be used to indicate C functions that are local to the current file so that a faster calling convention can be used. <span class="pre">`Py_LOCAL(type)`</span> declares the function as returning a value of the specified *type* and uses a fast-calling qualifier. <span class="pre">`Py_LOCAL_INLINE(type)`</span> does the same thing and also requests the function be inlined. If <span class="pre">`PY_LOCAL_AGGRESSIVE()`</span> is defined before <span class="pre">`python.h`</span> is included, a set of more aggressive optimizations are enabled for the module; you should benchmark the results to find out if these optimizations actually make the code faster. (Contributed by Fredrik Lundh at the NeedForSpeed sprint.)

- <span class="pre">`PyErr_NewException(name,`</span>` `<span class="pre">`base,`</span>` `<span class="pre">`dict)`</span> can now accept a tuple of base classes as its *base* argument. (Contributed by Georg Brandl.)

- The <a href="../c-api/exceptions.html#c.PyErr_Warn" class="reference internal" title="PyErr_Warn"><span class="pre"><code class="sourceCode c">PyErr_Warn<span class="op">()</span></code></span></a> function for issuing warnings is now deprecated in favour of <span class="pre">`PyErr_WarnEx(category,`</span>` `<span class="pre">`message,`</span>` `<span class="pre">`stacklevel)`</span> which lets you specify the number of stack frames separating this function and the caller. A *stacklevel* of 1 is the function calling <a href="../c-api/exceptions.html#c.PyErr_WarnEx" class="reference internal" title="PyErr_WarnEx"><span class="pre"><code class="sourceCode c">PyErr_WarnEx<span class="op">()</span></code></span></a>, 2 is the function above that, and so forth. (Added by Neal Norwitz.)

- The CPython interpreter is still written in C, but the code can now be compiled with a C++ compiler without errors. (Implemented by Anthony Baxter, Martin von Löwis, Skip Montanaro.)

- The <span class="pre">`PyRange_New()`</span> function was removed. It was never documented, never used in the core code, and had dangerously lax error checking. In the unlikely case that your extensions were using it, you can replace it by something like the following:

  <div class="highlight-default notranslate">

  <div class="highlight">

      range = PyObject_CallFunction((PyObject*) &PyRange_Type, "lll",
                                    start, stop, step);

  </div>

  </div>

<div id="port-specific-changes" class="section">

<span id="ports"></span>

### Port-Specific Changes<a href="#port-specific-changes" class="headerlink" title="Permalink to this headline">¶</a>

- MacOS X (10.3 and higher): dynamic loading of modules now uses the <span class="pre">`dlopen()`</span> function instead of MacOS-specific functions.

- MacOS X: an <span class="pre">`--enable-universalsdk`</span> switch was added to the **configure** script that compiles the interpreter as a universal binary able to run on both PowerPC and Intel processors. (Contributed by Ronald Oussoren; <a href="https://bugs.python.org/issue2573" class="reference external">bpo-2573</a>.)

- Windows: <span class="pre">`.dll`</span> is no longer supported as a filename extension for extension modules. <span class="pre">`.pyd`</span> is now the only filename extension that will be searched for.

</div>

</div>

<div id="porting-to-python-2-5" class="section">

<span id="porting"></span>

## Porting to Python 2.5<a href="#porting-to-python-2-5" class="headerlink" title="Permalink to this headline">¶</a>

This section lists previously described changes that may require changes to your code:

- ASCII is now the default encoding for modules. It’s now a syntax error if a module contains string literals with 8-bit characters but doesn’t have an encoding declaration. In Python 2.4 this triggered a warning, not a syntax error.

- Previously, the <span class="pre">`gi_frame`</span> attribute of a generator was always a frame object. Because of the <span id="index-28" class="target"></span><a href="https://www.python.org/dev/peps/pep-0342" class="pep reference external"><strong>PEP 342</strong></a> changes described in section <a href="#pep-342" class="reference internal"><span class="std std-ref">PEP 342: New Generator Features</span></a>, it’s now possible for <span class="pre">`gi_frame`</span> to be <span class="pre">`None`</span>.

- A new warning, <span class="pre">`UnicodeWarning`</span>, is triggered when you attempt to compare a Unicode string and an 8-bit string that can’t be converted to Unicode using the default ASCII encoding. Previously such comparisons would raise a <span class="pre">`UnicodeDecodeError`</span> exception.

- Library: the <a href="../library/csv.html#module-csv" class="reference internal" title="csv: Write and read tabular data to and from delimited files."><span class="pre"><code class="sourceCode python">csv</code></span></a> module is now stricter about multi-line quoted fields. If your files contain newlines embedded within fields, the input should be split into lines in a manner which preserves the newline characters.

- Library: the <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a> module’s <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> function’s would previously accept any string as long as no more than one %char specifier appeared. In Python 2.5, the argument must be exactly one %char specifier with no surrounding text.

- Library: The <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> and <a href="../library/pickle.html#module-cPickle" class="reference internal" title="cPickle: Faster version of pickle, but not subclassable."><span class="pre"><code class="sourceCode python">cPickle</code></span></a> modules no longer accept a return value of <span class="pre">`None`</span> from the <a href="../library/pickle.html#object.__reduce__" class="reference internal" title="object.__reduce__"><span class="pre"><code class="sourceCode python">__reduce__()</code></span></a> method; the method must return a tuple of arguments instead. The modules also no longer accept the deprecated *bin* keyword parameter.

- Library: The <a href="../library/simplexmlrpcserver.html#module-SimpleXMLRPCServer" class="reference internal" title="SimpleXMLRPCServer: Basic XML-RPC server implementation."><span class="pre"><code class="sourceCode python">SimpleXMLRPCServer</code></span></a> and <a href="../library/docxmlrpcserver.html#module-DocXMLRPCServer" class="reference internal" title="DocXMLRPCServer: Self-documenting XML-RPC server implementation."><span class="pre"><code class="sourceCode python">DocXMLRPCServer</code></span></a> classes now have a <span class="pre">`rpc_paths`</span> attribute that constrains XML-RPC operations to a limited set of URL paths; the default is to allow only <span class="pre">`'/'`</span> and <span class="pre">`'/RPC2'`</span>. Setting <span class="pre">`rpc_paths`</span> to <span class="pre">`None`</span> or an empty tuple disables this path checking.

- C API: Many functions now use <span class="pre">`Py_ssize_t`</span> instead of <span class="pre">`int`</span> to allow processing more data on 64-bit machines. Extension code may need to make the same change to avoid warnings and to support 64-bit machines. See the earlier section <a href="#pep-353" class="reference internal"><span class="std std-ref">PEP 353: Using ssize_t as the index type</span></a> for a discussion of this change.

- C API: The obmalloc changes mean that you must be careful to not mix usage of the <span class="pre">`PyMem_*()`</span> and <span class="pre">`PyObject_*()`</span> families of functions. Memory allocated with one family’s <span class="pre">`*_Malloc()`</span> must be freed with the corresponding family’s <span class="pre">`*_Free()`</span> function.

</div>

<div id="acknowledgements" class="section">

## Acknowledgements<a href="#acknowledgements" class="headerlink" title="Permalink to this headline">¶</a>

The author would like to thank the following people for offering suggestions, corrections and assistance with various drafts of this article: Georg Brandl, Nick Coghlan, Phillip J. Eby, Lars Gustäbel, Raymond Hettinger, Ralf W. Grosse-Kunstleve, Kent Johnson, Iain Lowe, Martin von Löwis, Fredrik Lundh, Andrew McNamara, Skip Montanaro, Gustavo Niemeyer, Paul Prescod, James Pryor, Mike Rovner, Scott Weikart, Barry Warsaw, Thomas Wouters.

</div>

</div>

</div>
