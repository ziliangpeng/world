<div class="body" role="main">

<div id="what-s-new-in-python-2-6" class="section">

# What’s New in Python 2.6<a href="#what-s-new-in-python-2-6" class="headerlink" title="Permalink to this headline">¶</a>

Author  
A.M. Kuchling (amk at amk.ca)

This article explains the new features in Python 2.6, released on October 1 2008. The release schedule is described in <span id="index-0" class="target"></span><a href="https://www.python.org/dev/peps/pep-0361" class="pep reference external"><strong>PEP 361</strong></a>.

The major theme of Python 2.6 is preparing the migration path to Python 3.0, a major redesign of the language. Whenever possible, Python 2.6 incorporates new features and syntax from 3.0 while remaining compatible with existing code by not removing older features or syntax. When it’s not possible to do that, Python 2.6 tries to do what it can, adding compatibility functions in a <a href="../library/future_builtins.html#module-future_builtins" class="reference internal" title="future_builtins"><span class="pre"><code class="sourceCode python">future_builtins</code></span></a> module and a <a href="../using/cmdline.html#cmdoption-3" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-3</code></span></a> switch to warn about usages that will become unsupported in 3.0.

Some significant new packages have been added to the standard library, such as the <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based &quot;threading&quot; interface."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> and <a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> modules, but there aren’t many new features that aren’t related to Python 3.0 in some way.

Python 2.6 also sees a number of improvements and bugfixes throughout the source. A search through the change logs finds there were 259 patches applied and 612 bugs fixed between Python 2.5 and 2.6. Both figures are likely to be underestimates.

This article doesn’t attempt to provide a complete specification of the new features, but instead provides a convenient overview. For full details, you should refer to the documentation for Python 2.6. If you want to understand the rationale for the design and implementation, refer to the PEP for a particular new feature. Whenever possible, “What’s New in Python” links to the bug/patch item for each change.

<div id="python-3-0" class="section">

## Python 3.0<a href="#python-3-0" class="headerlink" title="Permalink to this headline">¶</a>

The development cycle for Python versions 2.6 and 3.0 was synchronized, with the alpha and beta releases for both versions being made on the same days. The development of 3.0 has influenced many features in 2.6.

Python 3.0 is a far-ranging redesign of Python that breaks compatibility with the 2.x series. This means that existing Python code will need some conversion in order to run on Python 3.0. However, not all the changes in 3.0 necessarily break compatibility. In cases where new features won’t cause existing code to break, they’ve been backported to 2.6 and are described in this document in the appropriate place. Some of the 3.0-derived features are:

- A <a href="../reference/datamodel.html#object.__complex__" class="reference internal" title="object.__complex__"><span class="pre"><code class="sourceCode python"><span class="fu">__complex__</span>()</code></span></a> method for converting objects to a complex number.

- Alternate syntax for catching exceptions: <span class="pre">`except`</span>` `<span class="pre">`TypeError`</span>` `<span class="pre">`as`</span>` `<span class="pre">`exc`</span>.

- The addition of <a href="../library/functools.html#functools.reduce" class="reference internal" title="functools.reduce"><span class="pre"><code class="sourceCode python">functools.<span class="bu">reduce</span>()</code></span></a> as a synonym for the built-in <a href="../library/functions.html#reduce" class="reference internal" title="reduce"><span class="pre"><code class="sourceCode python"><span class="bu">reduce</span>()</code></span></a> function.

Python 3.0 adds several new built-in functions and changes the semantics of some existing builtins. Functions that are new in 3.0 such as <a href="../library/functions.html#bin" class="reference internal" title="bin"><span class="pre"><code class="sourceCode python"><span class="bu">bin</span>()</code></span></a> have simply been added to Python 2.6, but existing builtins haven’t been changed; instead, the <a href="../library/future_builtins.html#module-future_builtins" class="reference internal" title="future_builtins"><span class="pre"><code class="sourceCode python">future_builtins</code></span></a> module has versions with the new 3.0 semantics. Code written to be compatible with 3.0 can do <span class="pre">`from`</span>` `<span class="pre">`future_builtins`</span>` `<span class="pre">`import`</span>` `<span class="pre">`hex,`</span>` `<span class="pre">`map`</span> as necessary.

A new command-line switch, <a href="../using/cmdline.html#cmdoption-3" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-3</code></span></a>, enables warnings about features that will be removed in Python 3.0. You can run code with this switch to see how much work will be necessary to port code to 3.0. The value of this switch is available to Python code as the boolean variable <a href="../library/sys.html#sys.py3kwarning" class="reference internal" title="sys.py3kwarning"><span class="pre"><code class="sourceCode python">sys.py3kwarning</code></span></a>, and to C extension code as <span class="pre">`Py_Py3kWarningFlag`</span>.

<div class="admonition seealso">

See also

The 3xxx series of PEPs, which contains proposals for Python 3.0. <span id="index-1" class="target"></span><a href="https://www.python.org/dev/peps/pep-3000" class="pep reference external"><strong>PEP 3000</strong></a> describes the development process for Python 3.0. Start with <span id="index-2" class="target"></span><a href="https://www.python.org/dev/peps/pep-3100" class="pep reference external"><strong>PEP 3100</strong></a> that describes the general goals for Python 3.0, and then explore the higher-numbered PEPS that propose specific features.

</div>

</div>

<div id="changes-to-the-development-process" class="section">

## Changes to the Development Process<a href="#changes-to-the-development-process" class="headerlink" title="Permalink to this headline">¶</a>

While 2.6 was being developed, the Python development process underwent two significant changes: we switched from SourceForge’s issue tracker to a customized Roundup installation, and the documentation was converted from LaTeX to reStructuredText.

<div id="new-issue-tracker-roundup" class="section">

### New Issue Tracker: Roundup<a href="#new-issue-tracker-roundup" class="headerlink" title="Permalink to this headline">¶</a>

For a long time, the Python developers had been growing increasingly annoyed by SourceForge’s bug tracker. SourceForge’s hosted solution doesn’t permit much customization; for example, it wasn’t possible to customize the life cycle of issues.

The infrastructure committee of the Python Software Foundation therefore posted a call for issue trackers, asking volunteers to set up different products and import some of the bugs and patches from SourceForge. Four different trackers were examined: <a href="https://www.atlassian.com/software/jira/" class="reference external">Jira</a>, <a href="https://launchpad.net/" class="reference external">Launchpad</a>, <a href="http://roundup.sourceforge.net/" class="reference external">Roundup</a>, and <a href="https://trac.edgewall.org/" class="reference external">Trac</a>. The committee eventually settled on Jira and Roundup as the two candidates. Jira is a commercial product that offers no-cost hosted instances to free-software projects; Roundup is an open-source project that requires volunteers to administer it and a server to host it.

After posting a call for volunteers, a new Roundup installation was set up at <a href="https://bugs.python.org" class="reference external">https://bugs.python.org</a>. One installation of Roundup can host multiple trackers, and this server now also hosts issue trackers for Jython and for the Python web site. It will surely find other uses in the future. Where possible, this edition of “What’s New in Python” links to the bug/patch item for each change.

Hosting of the Python bug tracker is kindly provided by <a href="http://www.upfrontsystems.co.za/" class="reference external">Upfront Systems</a> of Stellenbosch, South Africa. Martin von Loewis put a lot of effort into importing existing bugs and patches from SourceForge; his scripts for this import operation are at <a href="http://svn.python.org/view/tracker/importer/" class="reference external">http://svn.python.org/view/tracker/importer/</a> and may be useful to other projects wishing to move from SourceForge to Roundup.

<div class="admonition seealso">

See also

<a href="https://bugs.python.org" class="reference external">https://bugs.python.org</a>  
The Python bug tracker.

<a href="http://bugs.jython.org" class="reference external">http://bugs.jython.org</a>:  
The Jython bug tracker.

<a href="http://roundup.sourceforge.net/" class="reference external">http://roundup.sourceforge.net/</a>  
Roundup downloads and documentation.

<a href="http://svn.python.org/view/tracker/importer/" class="reference external">http://svn.python.org/view/tracker/importer/</a>  
Martin von Loewis’s conversion scripts.

</div>

</div>

<div id="new-documentation-format-restructuredtext-using-sphinx" class="section">

### New Documentation Format: reStructuredText Using Sphinx<a href="#new-documentation-format-restructuredtext-using-sphinx" class="headerlink" title="Permalink to this headline">¶</a>

The Python documentation was written using LaTeX since the project started around 1989. In the 1980s and early 1990s, most documentation was printed out for later study, not viewed online. LaTeX was widely used because it provided attractive printed output while remaining straightforward to write once the basic rules of the markup were learned.

Today LaTeX is still used for writing publications destined for printing, but the landscape for programming tools has shifted. We no longer print out reams of documentation; instead, we browse through it online and HTML has become the most important format to support. Unfortunately, converting LaTeX to HTML is fairly complicated and Fred L. Drake Jr., the long-time Python documentation editor, spent a lot of time maintaining the conversion process. Occasionally people would suggest converting the documentation into SGML and later XML, but performing a good conversion is a major task and no one ever committed the time required to finish the job.

During the 2.6 development cycle, Georg Brandl put a lot of effort into building a new toolchain for processing the documentation. The resulting package is called Sphinx, and is available from <a href="http://sphinx-doc.org/" class="reference external">http://sphinx-doc.org/</a>.

Sphinx concentrates on HTML output, producing attractively styled and modern HTML; printed output is still supported through conversion to LaTeX. The input format is reStructuredText, a markup syntax supporting custom extensions and directives that is commonly used in the Python community.

Sphinx is a standalone package that can be used for writing, and almost two dozen other projects (<a href="http://sphinx-doc.org/examples.html" class="reference external">listed on the Sphinx web site</a>) have adopted Sphinx as their documentation tool.

<div class="admonition seealso">

See also

<a href="https://docs.python.org/devguide/documenting.html" class="reference external">Documenting Python</a>  
Describes how to write for Python’s documentation.

<a href="http://sphinx-doc.org/" class="reference external">Sphinx</a>  
Documentation and code for the Sphinx toolchain.

<a href="http://docutils.sourceforge.net" class="reference external">Docutils</a>  
The underlying reStructuredText parser and toolset.

</div>

</div>

</div>

<div id="pep-343-the-with-statement" class="section">

## PEP 343: The ‘with’ statement<a href="#pep-343-the-with-statement" class="headerlink" title="Permalink to this headline">¶</a>

The previous version, Python 2.5, added the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement as an optional feature, to be enabled by a <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`with_statement`</span> directive. In 2.6 the statement no longer needs to be specially enabled; this means that <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> is now always a keyword. The rest of this section is a copy of the corresponding section from the “What’s New in Python 2.5” document; if you’re familiar with the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement from Python 2.5, you can skip this section.

The ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement clarifies code that previously would use <span class="pre">`try...finally`</span> blocks to ensure that clean-up code is executed. In this section, I’ll discuss the statement as it will commonly be used. In the next section, I’ll examine the implementation details and show how to write objects for use with this statement.

The ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement is a control-flow structure whose basic structure is:

<div class="highlight-default notranslate">

<div class="highlight">

    with expression [as variable]:
        with-block

</div>

</div>

The expression is evaluated, and it should result in an object that supports the context management protocol (that is, has <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> methods).

The object’s <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> is called before *with-block* is executed and therefore can run set-up code. It also may return a value that is bound to the name *variable*, if given. (Note carefully that *variable* is *not* assigned the result of *expression*.)

After execution of the *with-block* is finished, the object’s <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> method is called, even if the block raised an exception, and can therefore run clean-up code.

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

The <span class="pre">`localcontext()`</span> function in the <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic  Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> module makes it easy to save and restore the current decimal context, which encapsulates the desired precision and rounding characteristics for computations:

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

<span id="new-26-context-managers"></span>

### Writing Context Managers<a href="#writing-context-managers" class="headerlink" title="Permalink to this headline">¶</a>

Under the hood, the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement is fairly complicated. Most people will only use ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ in company with existing objects and don’t need to know these details, so you can skip the rest of this section if you like. Authors of new objects will need to understand the details of the underlying implementation and should keep reading.

A high-level explanation of the context management protocol is:

- The expression is evaluated and should result in an object called a “context manager”. The context manager must have <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> methods.

- The context manager’s <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> method is called. The value returned is assigned to *VAR*. If no <span class="pre">`as`</span>` `<span class="pre">`VAR`</span> clause is present, the value is simply discarded.

- The code in *BLOCK* is executed.

- If *BLOCK* raises an exception, the context manager’s <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> method is called with three arguments, the exception details (<span class="pre">`type,`</span>` `<span class="pre">`value,`</span>` `<span class="pre">`traceback`</span>, the same values returned by <a href="../library/sys.html#sys.exc_info" class="reference internal" title="sys.exc_info"><span class="pre"><code class="sourceCode python">sys.exc_info()</code></span></a>, which can also be <span class="pre">`None`</span> if no exception occurred). The method’s return value controls whether an exception is re-raised: any false value re-raises the exception, and <span class="pre">`True`</span> will result in suppressing it. You’ll only rarely want to suppress the exception, because if you do the author of the code containing the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement will never realize anything went wrong.

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
        def cursor(self):
            "Returns a cursor object and starts a new transaction"
        def commit(self):
            "Commits current transaction"
        def rollback(self):
            "Rolls back current transaction"

</div>

</div>

The <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> method is pretty easy, having only to start a new transaction. For this application the resulting cursor object would be a useful result, so the method will return it. The user can then add <span class="pre">`as`</span>` `<span class="pre">`cursor`</span> to their ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement to bind the cursor to a variable name.

<div class="highlight-default notranslate">

<div class="highlight">

    class DatabaseConnection:
        ...
        def __enter__(self):
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
        def __exit__(self, type, value, tb):
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

<span id="module-contextlib"></span>

### The contextlib module<a href="#the-contextlib-module" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/contextlib.html#module-contextlib" class="reference internal" title="contextlib: Utilities for with-statement contexts."><span class="pre"><code class="sourceCode python">contextlib</code></span></a> module provides some functions and a decorator that are useful when writing objects for use with the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement.

The decorator is called <span class="pre">`contextmanager()`</span>, and lets you write a single generator function instead of defining a new class. The generator should yield exactly one value. The code up to the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> will be executed as the <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> method, and the value yielded will be the method’s return value that will get bound to the variable in the ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement’s <a href="../reference/compound_stmts.html#as" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">as</code></span></a> clause, if any. The code after the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> will be executed in the <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> method. Any exception raised in the block will be raised by the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement.

Using this decorator, our database example from the previous section could be written as:

<div class="highlight-default notranslate">

<div class="highlight">

    from contextlib import contextmanager

    @contextmanager
    def db_transaction(connection):
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

Finally, the <span class="pre">`closing()`</span> function returns its argument so that it can be bound to a variable, and calls the argument’s <span class="pre">`.close()`</span> method at the end of the block.

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

<span id="index-3" class="target"></span><a href="https://www.python.org/dev/peps/pep-0343" class="pep reference external"><strong>PEP 343</strong></a> - The “with” statement  
PEP written by Guido van Rossum and Nick Coghlan; implemented by Mike Bland, Guido van Rossum, and Neal Norwitz. The PEP shows the code generated for a ‘<a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a>’ statement, which can be helpful in learning how the statement works.

The documentation for the <a href="../library/contextlib.html#module-contextlib" class="reference internal" title="contextlib: Utilities for with-statement contexts."><span class="pre"><code class="sourceCode python">contextlib</code></span></a> module.

</div>

</div>

</div>

<div id="pep-366-explicit-relative-imports-from-a-main-module" class="section">

<span id="pep-0366"></span>

## PEP 366: Explicit Relative Imports From a Main Module<a href="#pep-366-explicit-relative-imports-from-a-main-module" class="headerlink" title="Permalink to this headline">¶</a>

Python’s <a href="../using/cmdline.html#cmdoption-m" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-m</code></span></a> switch allows running a module as a script. When you ran a module that was located inside a package, relative imports didn’t work correctly.

The fix for Python 2.6 adds a <span class="pre">`__package__`</span> attribute to modules. When this attribute is present, relative imports will be relative to the value of this attribute instead of the <span class="pre">`__name__`</span> attribute.

PEP 302-style importers can then set <span class="pre">`__package__`</span> as necessary. The <a href="../library/runpy.html#module-runpy" class="reference internal" title="runpy: Locate and run Python modules without importing them first."><span class="pre"><code class="sourceCode python">runpy</code></span></a> module that implements the <a href="../using/cmdline.html#cmdoption-m" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-m</code></span></a> switch now does this, so relative imports will now work correctly in scripts running from inside a package.

</div>

<div id="pep-370-per-user-site-packages-directory" class="section">

<span id="pep-0370"></span>

## PEP 370: Per-user <span class="pre">`site-packages`</span> Directory<a href="#pep-370-per-user-site-packages-directory" class="headerlink" title="Permalink to this headline">¶</a>

When you run Python, the module search path <span class="pre">`sys.path`</span> usually includes a directory whose path ends in <span class="pre">`"site-packages"`</span>. This directory is intended to hold locally-installed packages available to all users using a machine or a particular site installation.

Python 2.6 introduces a convention for user-specific site directories. The directory varies depending on the platform:

- Unix and Mac OS X: <span class="pre">`~/.local/`</span>

- Windows: <span class="pre">`%APPDATA%/Python`</span>

Within this directory, there will be version-specific subdirectories, such as <span class="pre">`lib/python2.6/site-packages`</span> on Unix/Mac OS and <span class="pre">`Python26/site-packages`</span> on Windows.

If you don’t like the default directory, it can be overridden by an environment variable. <span id="index-4" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONUSERBASE" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONUSERBASE</code></span></a> sets the root directory used for all Python versions supporting this feature. On Windows, the directory for application-specific data can be changed by setting the <span id="index-5" class="target"></span><span class="pre">`APPDATA`</span> environment variable. You can also modify the <span class="pre">`site.py`</span> file for your Python installation.

The feature can be disabled entirely by running Python with the <a href="../using/cmdline.html#cmdoption-s" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-s</code></span></a> option or setting the <span id="index-6" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONNOUSERSITE" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONNOUSERSITE</code></span></a> environment variable.

<div class="admonition seealso">

See also

<span id="index-7" class="target"></span><a href="https://www.python.org/dev/peps/pep-0370" class="pep reference external"><strong>PEP 370</strong></a> - Per-user <span class="pre">`site-packages`</span> Directory  
PEP written and implemented by Christian Heimes.

</div>

</div>

<div id="pep-371-the-multiprocessing-package" class="section">

<span id="pep-0371"></span>

## PEP 371: The <span class="pre">`multiprocessing`</span> Package<a href="#pep-371-the-multiprocessing-package" class="headerlink" title="Permalink to this headline">¶</a>

The new <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based &quot;threading&quot; interface."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> package lets Python programs create new processes that will perform a computation and return a result to the parent. The parent and child processes can communicate using queues and pipes, synchronize their operations using locks and semaphores, and can share simple arrays of data.

The <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based &quot;threading&quot; interface."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> module started out as an exact emulation of the <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> module using processes instead of threads. That goal was discarded along the path to Python 2.6, but the general approach of the module is still similar. The fundamental class is the <span class="pre">`Process`</span>, which is passed a callable object and a collection of arguments. The <span class="pre">`start()`</span> method sets the callable running in a subprocess, after which you can call the <span class="pre">`is_alive()`</span> method to check whether the subprocess is still running and the <span class="pre">`join()`</span> method to wait for the process to exit.

Here’s a simple example where the subprocess will calculate a factorial. The function doing the calculation is written strangely so that it takes significantly longer when the input argument is a multiple of 4.

<div class="highlight-default notranslate">

<div class="highlight">

    import time
    from multiprocessing import Process, Queue


    def factorial(queue, N):
        "Compute a factorial."
        # If N is a multiple of 4, this function will take much longer.
        if (N % 4) == 0:
            time.sleep(.05 * N/4)

        # Calculate the result
        fact = 1L
        for i in range(1, N+1):
            fact = fact * i

        # Put the result on the queue
        queue.put(fact)

    if __name__ == '__main__':
        queue = Queue()

        N = 5

        p = Process(target=factorial, args=(queue, N))
        p.start()
        p.join()

        result = queue.get()
        print 'Factorial', N, '=', result

</div>

</div>

A <a href="../library/queue.html#Queue.Queue" class="reference internal" title="Queue.Queue"><span class="pre"><code class="sourceCode python">Queue</code></span></a> is used to communicate the result of the factorial. The <a href="../library/queue.html#Queue.Queue" class="reference internal" title="Queue.Queue"><span class="pre"><code class="sourceCode python">Queue</code></span></a> object is stored in a global variable. The child process will use the value of the variable when the child was created; because it’s a <a href="../library/queue.html#Queue.Queue" class="reference internal" title="Queue.Queue"><span class="pre"><code class="sourceCode python">Queue</code></span></a>, parent and child can use the object to communicate. (If the parent were to change the value of the global variable, the child’s value would be unaffected, and vice versa.)

Two other classes, <span class="pre">`Pool`</span> and <span class="pre">`Manager`</span>, provide higher-level interfaces. <span class="pre">`Pool`</span> will create a fixed number of worker processes, and requests can then be distributed to the workers by calling <a href="../library/functions.html#apply" class="reference internal" title="apply"><span class="pre"><code class="sourceCode python"><span class="bu">apply</span>()</code></span></a> or <span class="pre">`apply_async()`</span> to add a single request, and <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> or <span class="pre">`map_async()`</span> to add a number of requests. The following code uses a <span class="pre">`Pool`</span> to spread requests across 5 worker processes and retrieve a list of results:

<div class="highlight-default notranslate">

<div class="highlight">

    from multiprocessing import Pool

    def factorial(N, dictionary):
        "Compute a factorial."
        ...
    p = Pool(5)
    result = p.map(factorial, range(1, 1000, 10))
    for v in result:
        print v

</div>

</div>

This produces the following output:

<div class="highlight-default notranslate">

<div class="highlight">

    1
    39916800
    51090942171709440000
    8222838654177922817725562880000000
    33452526613163807108170062053440751665152000000000
    ...

</div>

</div>

The other high-level interface, the <span class="pre">`Manager`</span> class, creates a separate server process that can hold master copies of Python data structures. Other processes can then access and modify these data structures using proxy objects. The following example creates a shared dictionary by calling the <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>()</code></span></a> method; the worker processes then insert values into the dictionary. (Locking is not done for you automatically, which doesn’t matter in this example. <span class="pre">`Manager`</span>’s methods also include <span class="pre">`Lock()`</span>, <span class="pre">`RLock()`</span>, and <span class="pre">`Semaphore()`</span> to create shared locks.)

<div class="highlight-default notranslate">

<div class="highlight">

    import time
    from multiprocessing import Pool, Manager

    def factorial(N, dictionary):
        "Compute a factorial."
        # Calculate the result
        fact = 1L
        for i in range(1, N+1):
            fact = fact * i

        # Store result in dictionary
        dictionary[N] = fact

    if __name__ == '__main__':
        p = Pool(5)
        mgr = Manager()
        d = mgr.dict()         # Create shared dictionary

        # Run tasks using the pool
        for N in range(1, 1000, 10):
            p.apply_async(factorial, (N, d))

        # Mark pool as closed -- no more tasks can be added.
        p.close()

        # Wait for tasks to exit
        p.join()

        # Output results
        for k, v in sorted(d.items()):
            print k, v

</div>

</div>

This will produce the output:

<div class="highlight-default notranslate">

<div class="highlight">

    1 1
    11 39916800
    21 51090942171709440000
    31 8222838654177922817725562880000000
    41 33452526613163807108170062053440751665152000000000
    51 15511187532873822802242430164693032110632597200169861120000...

</div>

</div>

<div class="admonition seealso">

See also

The documentation for the <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based &quot;threading&quot; interface."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> module.

<span id="index-8" class="target"></span><a href="https://www.python.org/dev/peps/pep-0371" class="pep reference external"><strong>PEP 371</strong></a> - Addition of the multiprocessing package  
PEP written by Jesse Noller and Richard Oudkerk; implemented by Richard Oudkerk and Jesse Noller.

</div>

</div>

<div id="pep-3101-advanced-string-formatting" class="section">

<span id="pep-3101"></span>

## PEP 3101: Advanced String Formatting<a href="#pep-3101-advanced-string-formatting" class="headerlink" title="Permalink to this headline">¶</a>

In Python 3.0, the % operator is supplemented by a more powerful string formatting method, <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a>. Support for the <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a> method has been backported to Python 2.6.

In 2.6, both 8-bit and Unicode strings have a .format() method that treats the string as a template and takes the arguments to be formatted. The formatting template uses curly brackets ({, }) as special characters:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> # Substitute positional argument 0 into the string.
    >>> "User ID: {0}".format("root")
    'User ID: root'
    >>> # Use the named keyword arguments
    >>> "User ID: {uid}   Last seen: {last_login}".format(
    ...    uid="root",
    ...    last_login = "5 Mar 2008 07:20")
    'User ID: root   Last seen: 5 Mar 2008 07:20'

</div>

</div>

Curly brackets can be escaped by doubling them:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> "Empty dict: {{}}".format()
    "Empty dict: {}"

</div>

</div>

Field names can be integers indicating positional arguments, such as <span class="pre">`{0}`</span>, <span class="pre">`{1}`</span>, etc. or names of keyword arguments. You can also supply compound field names that read attributes or access dictionary keys:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> import sys
    >>> print 'Platform: {0.platform}\nPython version: {0.version}'.format(sys)
    Platform: darwin
    Python version: 2.6a1+ (trunk:61261M, Mar  5 2008, 20:29:41)
    [GCC 4.0.1 (Apple Computer, Inc. build 5367)]'

    >>> import mimetypes
    >>> 'Content-type: {0[.mp4]}'.format(mimetypes.types_map)
    'Content-type: video/mp4'

</div>

</div>

Note that when using dictionary-style notation such as <span class="pre">`[.mp4]`</span>, you don’t need to put any quotation marks around the string; it will look up the value using <span class="pre">`.mp4`</span> as the key. Strings beginning with a number will be converted to an integer. You can’t write more complicated expressions inside a format string.

So far we’ve shown how to specify which field to substitute into the resulting string. The precise formatting used is also controllable by adding a colon followed by a format specifier. For example:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> # Field 0: left justify, pad to 15 characters
    >>> # Field 1: right justify, pad to 6 characters
    >>> fmt = '{0:15} ${1:>6}'
    >>> fmt.format('Registration', 35)
    'Registration    $    35'
    >>> fmt.format('Tutorial', 50)
    'Tutorial        $    50'
    >>> fmt.format('Banquet', 125)
    'Banquet         $   125'

</div>

</div>

Format specifiers can reference other fields through nesting:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> fmt = '{0:{1}}'
    >>> width = 15
    >>> fmt.format('Invoice #1234', width)
    'Invoice #1234  '
    >>> width = 35
    >>> fmt.format('Invoice #1234', width)
    'Invoice #1234                      '

</div>

</div>

The alignment of a field within the desired width can be specified:

| Character    | Effect                                       |
|--------------|----------------------------------------------|
| \< (default) | Left-align                                   |
| \>           | Right-align                                  |
| ^            | Center                                       |
| =            | (For numeric types only) Pad after the sign. |

Format specifiers can also include a presentation type, which controls how the value is formatted. For example, floating-point numbers can be formatted as a general number or in exponential notation:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> '{0:g}'.format(3.75)
    '3.75'
    >>> '{0:e}'.format(3.75)
    '3.750000e+00'

</div>

</div>

A variety of presentation types are available. Consult the 2.6 documentation for a <a href="../library/string.html#formatstrings" class="reference internal"><span class="std std-ref">complete list</span></a>; here’s a sample:

|  |  |
|----|----|
| <span class="pre">`b`</span> | Binary. Outputs the number in base 2. |
| <span class="pre">`c`</span> | Character. Converts the integer to the corresponding Unicode character before printing. |
| <span class="pre">`d`</span> | Decimal Integer. Outputs the number in base 10. |
| <span class="pre">`o`</span> | Octal format. Outputs the number in base 8. |
| <span class="pre">`x`</span> | Hex format. Outputs the number in base 16, using lower-case letters for the digits above 9. |
| <span class="pre">`e`</span> | Exponent notation. Prints the number in scientific notation using the letter ‘e’ to indicate the exponent. |
| <span class="pre">`g`</span> | General format. This prints the number as a fixed-point number, unless the number is too large, in which case it switches to ‘e’ exponent notation. |
| <span class="pre">`n`</span> | Number. This is the same as ‘g’ (for floats) or ‘d’ (for integers), except that it uses the current locale setting to insert the appropriate number separator characters. |
| <span class="pre">`%`</span> | Percentage. Multiplies the number by 100 and displays in fixed (‘f’) format, followed by a percent sign. |

Classes and types can define a <span class="pre">`__format__()`</span> method to control how they’re formatted. It receives a single argument, the format specifier:

<div class="highlight-default notranslate">

<div class="highlight">

    def __format__(self, format_spec):
        if isinstance(format_spec, unicode):
            return unicode(str(self))
        else:
            return str(self)

</div>

</div>

There’s also a <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> builtin that will format a single value. It calls the type’s <span class="pre">`__format__()`</span> method with the provided specifier:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> format(75.6564, '.2f')
    '75.66'

</div>

</div>

<div class="admonition seealso">

See also

<a href="../library/string.html#formatstrings" class="reference internal"><span class="std std-ref">Format String Syntax</span></a>  
The reference documentation for format fields.

<span id="index-9" class="target"></span><a href="https://www.python.org/dev/peps/pep-3101" class="pep reference external"><strong>PEP 3101</strong></a> - Advanced String Formatting  
PEP written by Talin. Implemented by Eric Smith.

</div>

</div>

<div id="pep-3105-print-as-a-function" class="section">

<span id="pep-3105"></span>

## PEP 3105: <span class="pre">`print`</span> As a Function<a href="#pep-3105-print-as-a-function" class="headerlink" title="Permalink to this headline">¶</a>

The <span class="pre">`print`</span> statement becomes the <a href="../library/functions.html#print" class="reference internal" title="print"><span class="pre"><code class="sourceCode python"><span class="bu">print</span>()</code></span></a> function in Python 3.0. Making <a href="../library/functions.html#print" class="reference internal" title="print"><span class="pre"><code class="sourceCode python"><span class="bu">print</span>()</code></span></a> a function makes it possible to replace the function by doing <span class="pre">`def`</span>` `<span class="pre">`print(...)`</span> or importing a new function from somewhere else.

Python 2.6 has a <span class="pre">`__future__`</span> import that removes <span class="pre">`print`</span> as language syntax, letting you use the functional form instead. For example:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> from __future__ import print_function
    >>> print('# of entries', len(dictionary), file=sys.stderr)

</div>

</div>

The signature of the new function is:

<div class="highlight-default notranslate">

<div class="highlight">

    def print(*args, sep=' ', end='\n', file=None)

</div>

</div>

The parameters are:

> <div>
>
> - *args*: positional arguments whose values will be printed out.
>
> - *sep*: the separator, which will be printed between arguments.
>
> - *end*: the ending text, which will be printed after all of the arguments have been output.
>
> - *file*: the file object to which the output will be sent.
>
> </div>

<div class="admonition seealso">

See also

<span id="index-10" class="target"></span><a href="https://www.python.org/dev/peps/pep-3105" class="pep reference external"><strong>PEP 3105</strong></a> - Make print a function  
PEP written by Georg Brandl.

</div>

</div>

<div id="pep-3110-exception-handling-changes" class="section">

<span id="pep-3110"></span>

## PEP 3110: Exception-Handling Changes<a href="#pep-3110-exception-handling-changes" class="headerlink" title="Permalink to this headline">¶</a>

One error that Python programmers occasionally make is writing the following code:

<div class="highlight-default notranslate">

<div class="highlight">

    try:
        ...
    except TypeError, ValueError:  # Wrong!
        ...

</div>

</div>

The author is probably trying to catch both <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> and <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> exceptions, but this code actually does something different: it will catch <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> and bind the resulting exception object to the local name <span class="pre">`"ValueError"`</span>. The <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> exception will not be caught at all. The correct code specifies a tuple of exceptions:

<div class="highlight-default notranslate">

<div class="highlight">

    try:
        ...
    except (TypeError, ValueError):
        ...

</div>

</div>

This error happens because the use of the comma here is ambiguous: does it indicate two different nodes in the parse tree, or a single node that’s a tuple?

Python 3.0 makes this unambiguous by replacing the comma with the word “as”. To catch an exception and store the exception object in the variable <span class="pre">`exc`</span>, you must write:

<div class="highlight-default notranslate">

<div class="highlight">

    try:
        ...
    except TypeError as exc:
        ...

</div>

</div>

Python 3.0 will only support the use of “as”, and therefore interprets the first example as catching two different exceptions. Python 2.6 supports both the comma and “as”, so existing code will continue to work. We therefore suggest using “as” when writing new Python code that will only be executed with 2.6.

<div class="admonition seealso">

See also

<span id="index-11" class="target"></span><a href="https://www.python.org/dev/peps/pep-3110" class="pep reference external"><strong>PEP 3110</strong></a> - Catching Exceptions in Python 3000  
PEP written and implemented by Collin Winter.

</div>

</div>

<div id="pep-3112-byte-literals" class="section">

<span id="pep-3112"></span>

## PEP 3112: Byte Literals<a href="#pep-3112-byte-literals" class="headerlink" title="Permalink to this headline">¶</a>

Python 3.0 adopts Unicode as the language’s fundamental string type and denotes 8-bit literals differently, either as <span class="pre">`b'string'`</span> or using a <span class="pre">`bytes`</span> constructor. For future compatibility, Python 2.6 adds <span class="pre">`bytes`</span> as a synonym for the <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> type, and it also supports the <span class="pre">`b''`</span> notation.

The 2.6 <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> differs from 3.0’s <span class="pre">`bytes`</span> type in various ways; most notably, the constructor is completely different. In 3.0, <span class="pre">`bytes([65,`</span>` `<span class="pre">`66,`</span>` `<span class="pre">`67])`</span> is 3 elements long, containing the bytes representing <span class="pre">`ABC`</span>; in 2.6, <span class="pre">`bytes([65,`</span>` `<span class="pre">`66,`</span>` `<span class="pre">`67])`</span> returns the 12-byte string representing the <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> of the list.

The primary use of <span class="pre">`bytes`</span> in 2.6 will be to write tests of object type such as <span class="pre">`isinstance(x,`</span>` `<span class="pre">`bytes)`</span>. This will help the 2to3 converter, which can’t tell whether 2.x code intends strings to contain either characters or 8-bit bytes; you can now use either <span class="pre">`bytes`</span> or <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> to represent your intention exactly, and the resulting code will also be correct in Python 3.0.

There’s also a <span class="pre">`__future__`</span> import that causes all string literals to become Unicode strings. This means that <span class="pre">`\u`</span> escape sequences can be used to include Unicode characters:

<div class="highlight-default notranslate">

<div class="highlight">

    from __future__ import unicode_literals

    s = ('\u751f\u3080\u304e\u3000\u751f\u3054'
         '\u3081\u3000\u751f\u305f\u307e\u3054')

    print len(s)               # 12 Unicode characters

</div>

</div>

At the C level, Python 3.0 will rename the existing 8-bit string type, called <a href="../c-api/string.html#c.PyStringObject" class="reference internal" title="PyStringObject"><span class="pre"><code class="sourceCode c">PyStringObject</code></span></a> in Python 2.x, to <span class="pre">`PyBytesObject`</span>. Python 2.6 uses <span class="pre">`#define`</span> to support using the names <span class="pre">`PyBytesObject()`</span>, <span class="pre">`PyBytes_Check()`</span>, <span class="pre">`PyBytes_FromStringAndSize()`</span>, and all the other functions and macros used with strings.

Instances of the <span class="pre">`bytes`</span> type are immutable just as strings are. A new <a href="../library/functions.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> type stores a mutable sequence of bytes:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> bytearray([65, 66, 67])
    bytearray(b'ABC')
    >>> b = bytearray(u'\u21ef\u3244', 'utf-8')
    >>> b
    bytearray(b'\xe2\x87\xaf\xe3\x89\x84')
    >>> b[0] = '\xe3'
    >>> b
    bytearray(b'\xe3\x87\xaf\xe3\x89\x84')
    >>> unicode(str(b), 'utf-8')
    u'\u31ef \u3244'

</div>

</div>

Byte arrays support most of the methods of string types, such as <span class="pre">`startswith()`</span>/<span class="pre">`endswith()`</span>, <span class="pre">`find()`</span>/<span class="pre">`rfind()`</span>, and some of the methods of lists, such as <span class="pre">`append()`</span>, <span class="pre">`pop()`</span>, and <span class="pre">`reverse()`</span>.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> b = bytearray('ABC')
    >>> b.append('d')
    >>> b.append(ord('e'))
    >>> b
    bytearray(b'ABCde')

</div>

</div>

There’s also a corresponding C API, with <a href="../c-api/bytearray.html#c.PyByteArray_FromObject" class="reference internal" title="PyByteArray_FromObject"><span class="pre"><code class="sourceCode c">PyByteArray_FromObject<span class="op">()</span></code></span></a>, <a href="../c-api/bytearray.html#c.PyByteArray_FromStringAndSize" class="reference internal" title="PyByteArray_FromStringAndSize"><span class="pre"><code class="sourceCode c">PyByteArray_FromStringAndSize<span class="op">()</span></code></span></a>, and various other functions.

<div class="admonition seealso">

See also

<span id="index-12" class="target"></span><a href="https://www.python.org/dev/peps/pep-3112" class="pep reference external"><strong>PEP 3112</strong></a> - Bytes literals in Python 3000  
PEP written by Jason Orendorff; backported to 2.6 by Christian Heimes.

</div>

</div>

<div id="pep-3116-new-i-o-library" class="section">

<span id="pep-3116"></span>

## PEP 3116: New I/O Library<a href="#pep-3116-new-i-o-library" class="headerlink" title="Permalink to this headline">¶</a>

Python’s built-in file objects support a number of methods, but file-like objects don’t necessarily support all of them. Objects that imitate files usually support <span class="pre">`read()`</span> and <span class="pre">`write()`</span>, but they may not support <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline()</code></span></a>, for example. Python 3.0 introduces a layered I/O library in the <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module that separates buffering and text-handling features from the fundamental read and write operations.

There are three levels of abstract base classes provided by the <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module:

- <span class="pre">`RawIOBase`</span> defines raw I/O operations: <span class="pre">`read()`</span>, <span class="pre">`readinto()`</span>, <span class="pre">`write()`</span>, <span class="pre">`seek()`</span>, <span class="pre">`tell()`</span>, <span class="pre">`truncate()`</span>, and <span class="pre">`close()`</span>. Most of the methods of this class will often map to a single system call. There are also <span class="pre">`readable()`</span>, <span class="pre">`writable()`</span>, and <span class="pre">`seekable()`</span> methods for determining what operations a given object will allow.

  Python 3.0 has concrete implementations of this class for files and sockets, but Python 2.6 hasn’t restructured its file and socket objects in this way.

- <span class="pre">`BufferedIOBase`</span> is an abstract base class that buffers data in memory to reduce the number of system calls used, making I/O processing more efficient. It supports all of the methods of <span class="pre">`RawIOBase`</span>, and adds a <span class="pre">`raw`</span> attribute holding the underlying raw object.

  There are five concrete classes implementing this ABC. <span class="pre">`BufferedWriter`</span> and <span class="pre">`BufferedReader`</span> are for objects that support write-only or read-only usage that have a <span class="pre">`seek()`</span> method for random access. <span class="pre">`BufferedRandom`</span> objects support read and write access upon the same underlying stream, and <span class="pre">`BufferedRWPair`</span> is for objects such as TTYs that have both read and write operations acting upon unconnected streams of data. The <span class="pre">`BytesIO`</span> class supports reading, writing, and seeking over an in-memory buffer.

- <div id="index-13">

  <span class="pre">`TextIOBase`</span>: Provides functions for reading and writing strings (remember, strings will be Unicode in Python 3.0), and supporting <a href="../glossary.html#term-universal-newlines" class="reference internal"><span class="xref std std-term">universal newlines</span></a>. <span class="pre">`TextIOBase`</span> defines the <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline()</code></span></a> method and supports iteration upon objects.

  There are two concrete implementations. <span class="pre">`TextIOWrapper`</span> wraps a buffered I/O object, supporting all of the methods for text I/O and adding a <a href="../library/functions.html#buffer" class="reference internal" title="buffer"><span class="pre"><code class="sourceCode python"><span class="bu">buffer</span></code></span></a> attribute for access to the underlying object. <a href="../library/stringio.html#StringIO.StringIO" class="reference internal" title="StringIO.StringIO"><span class="pre"><code class="sourceCode python">StringIO</code></span></a> simply buffers everything in memory without ever writing anything to disk.

  (In Python 2.6, <a href="../library/io.html#io.StringIO" class="reference internal" title="io.StringIO"><span class="pre"><code class="sourceCode python">io.StringIO</code></span></a> is implemented in pure Python, so it’s pretty slow. You should therefore stick with the existing <a href="../library/stringio.html#module-StringIO" class="reference internal" title="StringIO: Read and write strings as if they were files."><span class="pre"><code class="sourceCode python">StringIO</code></span></a> module or <a href="../library/stringio.html#module-cStringIO" class="reference internal" title="cStringIO: Faster version of StringIO, but not subclassable."><span class="pre"><code class="sourceCode python">cStringIO</code></span></a> for now. At some point Python 3.0’s <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module will be rewritten into C for speed, and perhaps the C implementation will be backported to the 2.x releases.)

  </div>

In Python 2.6, the underlying implementations haven’t been restructured to build on top of the <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module’s classes. The module is being provided to make it easier to write code that’s forward-compatible with 3.0, and to save developers the effort of writing their own implementations of buffering and text I/O.

<div class="admonition seealso">

See also

<span id="index-14" class="target"></span><a href="https://www.python.org/dev/peps/pep-3116" class="pep reference external"><strong>PEP 3116</strong></a> - New I/O  
PEP written by Daniel Stutzbach, Mike Verdone, and Guido van Rossum. Code by Guido van Rossum, Georg Brandl, Walter Doerwald, Jeremy Hylton, Martin von Loewis, Tony Lownds, and others.

</div>

</div>

<div id="pep-3118-revised-buffer-protocol" class="section">

<span id="pep-3118"></span>

## PEP 3118: Revised Buffer Protocol<a href="#pep-3118-revised-buffer-protocol" class="headerlink" title="Permalink to this headline">¶</a>

The buffer protocol is a C-level API that lets Python types exchange pointers into their internal representations. A memory-mapped file can be viewed as a buffer of characters, for example, and this lets another module such as <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> treat memory-mapped files as a string of characters to be searched.

The primary users of the buffer protocol are numeric-processing packages such as NumPy, which expose the internal representation of arrays so that callers can write data directly into an array instead of going through a slower API. This PEP updates the buffer protocol in light of experience from NumPy development, adding a number of new features such as indicating the shape of an array or locking a memory region.

The most important new C API function is <span class="pre">`PyObject_GetBuffer(PyObject`</span>` `<span class="pre">`*obj,`</span>` `<span class="pre">`Py_buffer`</span>` `<span class="pre">`*view,`</span>` `<span class="pre">`int`</span>` `<span class="pre">`flags)`</span>, which takes an object and a set of flags, and fills in the <span class="pre">`Py_buffer`</span> structure with information about the object’s memory representation. Objects can use this operation to lock memory in place while an external caller could be modifying the contents, so there’s a corresponding <span class="pre">`PyBuffer_Release(Py_buffer`</span>` `<span class="pre">`*view)`</span> to indicate that the external caller is done.

The *flags* argument to <a href="../c-api/buffer.html#c.PyObject_GetBuffer" class="reference internal" title="PyObject_GetBuffer"><span class="pre"><code class="sourceCode c">PyObject_GetBuffer<span class="op">()</span></code></span></a> specifies constraints upon the memory returned. Some examples are:

> <div>
>
> - <span class="pre">`PyBUF_WRITABLE`</span> indicates that the memory must be writable.
>
> - <span class="pre">`PyBUF_LOCK`</span> requests a read-only or exclusive lock on the memory.
>
> - <span class="pre">`PyBUF_C_CONTIGUOUS`</span> and <span class="pre">`PyBUF_F_CONTIGUOUS`</span> requests a C-contiguous (last dimension varies the fastest) or Fortran-contiguous (first dimension varies the fastest) array layout.
>
> </div>

Two new argument codes for <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple<span class="op">()</span></code></span></a>, <span class="pre">`s*`</span> and <span class="pre">`z*`</span>, return locked buffer objects for a parameter.

<div class="admonition seealso">

See also

<span id="index-15" class="target"></span><a href="https://www.python.org/dev/peps/pep-3118" class="pep reference external"><strong>PEP 3118</strong></a> - Revising the buffer protocol  
PEP written by Travis Oliphant and Carl Banks; implemented by Travis Oliphant.

</div>

</div>

<div id="pep-3119-abstract-base-classes" class="section">

<span id="pep-3119"></span>

## PEP 3119: Abstract Base Classes<a href="#pep-3119-abstract-base-classes" class="headerlink" title="Permalink to this headline">¶</a>

Some object-oriented languages such as Java support interfaces, declaring that a class has a given set of methods or supports a given access protocol. Abstract Base Classes (or ABCs) are an equivalent feature for Python. The ABC support consists of an <a href="../library/abc.html#module-abc" class="reference internal" title="abc: Abstract base classes according to PEP 3119."><span class="pre"><code class="sourceCode python">abc</code></span></a> module containing a metaclass called <span class="pre">`ABCMeta`</span>, special handling of this metaclass by the <a href="../library/functions.html#isinstance" class="reference internal" title="isinstance"><span class="pre"><code class="sourceCode python"><span class="bu">isinstance</span>()</code></span></a> and <a href="../library/functions.html#issubclass" class="reference internal" title="issubclass"><span class="pre"><code class="sourceCode python"><span class="bu">issubclass</span>()</code></span></a> builtins, and a collection of basic ABCs that the Python developers think will be widely useful. Future versions of Python will probably add more ABCs.

Let’s say you have a particular class and wish to know whether it supports dictionary-style access. The phrase “dictionary-style” is vague, however. It probably means that accessing items with <span class="pre">`obj[1]`</span> works. Does it imply that setting items with <span class="pre">`obj[2]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`value`</span> works? Or that the object will have <span class="pre">`keys()`</span>, <span class="pre">`values()`</span>, and <span class="pre">`items()`</span> methods? What about the iterative variants such as <span class="pre">`iterkeys()`</span>? <a href="../library/copy.html#module-copy" class="reference internal" title="copy: Shallow and deep copy operations."><span class="pre"><code class="sourceCode python">copy()</code></span></a> and <span class="pre">`update()`</span>? Iterating over the object with <a href="../library/functions.html#iter" class="reference internal" title="iter"><span class="pre"><code class="sourceCode python"><span class="bu">iter</span>()</code></span></a>?

The Python 2.6 <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module includes a number of different ABCs that represent these distinctions. <span class="pre">`Iterable`</span> indicates that a class defines <a href="../reference/datamodel.html#object.__iter__" class="reference internal" title="object.__iter__"><span class="pre"><code class="sourceCode python"><span class="fu">__iter__</span>()</code></span></a>, and <span class="pre">`Container`</span> means the class defines a <a href="../reference/datamodel.html#object.__contains__" class="reference internal" title="object.__contains__"><span class="pre"><code class="sourceCode python"><span class="fu">__contains__</span>()</code></span></a> method and therefore supports <span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`y`</span> expressions. The basic dictionary interface of getting items, setting items, and <span class="pre">`keys()`</span>, <span class="pre">`values()`</span>, and <span class="pre">`items()`</span>, is defined by the <span class="pre">`MutableMapping`</span> ABC.

You can derive your own classes from a particular ABC to indicate they support that ABC’s interface:

<div class="highlight-default notranslate">

<div class="highlight">

    import collections

    class Storage(collections.MutableMapping):
        ...

</div>

</div>

Alternatively, you could write the class without deriving from the desired ABC and instead register the class by calling the ABC’s <span class="pre">`register()`</span> method:

<div class="highlight-default notranslate">

<div class="highlight">

    import collections

    class Storage:
        ...

    collections.MutableMapping.register(Storage)

</div>

</div>

For classes that you write, deriving from the ABC is probably clearer. The <span class="pre">`register()`</span> method is useful when you’ve written a new ABC that can describe an existing type or class, or if you want to declare that some third-party class implements an ABC. For example, if you defined a <span class="pre">`PrintableType`</span> ABC, it’s legal to do:

<div class="highlight-default notranslate">

<div class="highlight">

    # Register Python's types
    PrintableType.register(int)
    PrintableType.register(float)
    PrintableType.register(str)

</div>

</div>

Classes should obey the semantics specified by an ABC, but Python can’t check this; it’s up to the class author to understand the ABC’s requirements and to implement the code accordingly.

To check whether an object supports a particular interface, you can now write:

<div class="highlight-default notranslate">

<div class="highlight">

    def func(d):
        if not isinstance(d, collections.MutableMapping):
            raise ValueError("Mapping object expected, not %r" % d)

</div>

</div>

Don’t feel that you must now begin writing lots of checks as in the above example. Python has a strong tradition of duck-typing, where explicit type-checking is never done and code simply calls methods on an object, trusting that those methods will be there and raising an exception if they aren’t. Be judicious in checking for ABCs and only do it where it’s absolutely necessary.

You can write your own ABCs by using <span class="pre">`abc.ABCMeta`</span> as the metaclass in a class definition:

<div class="highlight-default notranslate">

<div class="highlight">

    from abc import ABCMeta, abstractmethod

    class Drawable():
        __metaclass__ = ABCMeta

        @abstractmethod
        def draw(self, x, y, scale=1.0):
            pass

        def draw_doubled(self, x, y):
            self.draw(x, y, scale=2.0)


    class Square(Drawable):
        def draw(self, x, y, scale):
            ...

</div>

</div>

In the <span class="pre">`Drawable`</span> ABC above, the <span class="pre">`draw_doubled()`</span> method renders the object at twice its size and can be implemented in terms of other methods described in <span class="pre">`Drawable`</span>. Classes implementing this ABC therefore don’t need to provide their own implementation of <span class="pre">`draw_doubled()`</span>, though they can do so. An implementation of <span class="pre">`draw()`</span> is necessary, though; the ABC can’t provide a useful generic implementation.

You can apply the <span class="pre">`@abstractmethod`</span> decorator to methods such as <span class="pre">`draw()`</span> that must be implemented; Python will then raise an exception for classes that don’t define the method. Note that the exception is only raised when you actually try to create an instance of a subclass lacking the method:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> class Circle(Drawable):
    ...     pass
    ...
    >>> c = Circle()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: Can't instantiate abstract class Circle with abstract methods draw
    >>>

</div>

</div>

Abstract data attributes can be declared using the <span class="pre">`@abstractproperty`</span> decorator:

<div class="highlight-default notranslate">

<div class="highlight">

    from abc import abstractproperty
    ...

    @abstractproperty
    def readonly(self):
       return self._x

</div>

</div>

Subclasses must then define a <span class="pre">`readonly()`</span> property.

<div class="admonition seealso">

See also

<span id="index-16" class="target"></span><a href="https://www.python.org/dev/peps/pep-3119" class="pep reference external"><strong>PEP 3119</strong></a> - Introducing Abstract Base Classes  
PEP written by Guido van Rossum and Talin. Implemented by Guido van Rossum. Backported to 2.6 by Benjamin Aranguren, with Alex Martelli.

</div>

</div>

<div id="pep-3127-integer-literal-support-and-syntax" class="section">

<span id="pep-3127"></span>

## PEP 3127: Integer Literal Support and Syntax<a href="#pep-3127-integer-literal-support-and-syntax" class="headerlink" title="Permalink to this headline">¶</a>

Python 3.0 changes the syntax for octal (base-8) integer literals, prefixing them with “0o” or “0O” instead of a leading zero, and adds support for binary (base-2) integer literals, signalled by a “0b” or “0B” prefix.

Python 2.6 doesn’t drop support for a leading 0 signalling an octal number, but it does add support for “0o” and “0b”:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> 0o21, 2*8 + 1
    (17, 17)
    >>> 0b101111
    47

</div>

</div>

The <a href="../library/functions.html#oct" class="reference internal" title="oct"><span class="pre"><code class="sourceCode python"><span class="bu">oct</span>()</code></span></a> builtin still returns numbers prefixed with a leading zero, and a new <a href="../library/functions.html#bin" class="reference internal" title="bin"><span class="pre"><code class="sourceCode python"><span class="bu">bin</span>()</code></span></a> builtin returns the binary representation for a number:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> oct(42)
    '052'
    >>> future_builtins.oct(42)
    '0o52'
    >>> bin(173)
    '0b10101101'

</div>

</div>

The <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a> and <a href="../library/functions.html#long" class="reference internal" title="long"><span class="pre"><code class="sourceCode python"><span class="bu">long</span>()</code></span></a> builtins will now accept the “0o” and “0b” prefixes when base-8 or base-2 are requested, or when the *base* argument is zero (signalling that the base used should be determined from the string):

<div class="highlight-default notranslate">

<div class="highlight">

    >>> int ('0o52', 0)
    42
    >>> int('1101', 2)
    13
    >>> int('0b1101', 2)
    13
    >>> int('0b1101', 0)
    13

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-17" class="target"></span><a href="https://www.python.org/dev/peps/pep-3127" class="pep reference external"><strong>PEP 3127</strong></a> - Integer Literal Support and Syntax  
PEP written by Patrick Maupin; backported to 2.6 by Eric Smith.

</div>

</div>

<div id="pep-3129-class-decorators" class="section">

<span id="pep-3129"></span>

## PEP 3129: Class Decorators<a href="#pep-3129-class-decorators" class="headerlink" title="Permalink to this headline">¶</a>

Decorators have been extended from functions to classes. It’s now legal to write:

<div class="highlight-default notranslate">

<div class="highlight">

    @foo
    @bar
    class A:
      pass

</div>

</div>

This is equivalent to:

<div class="highlight-default notranslate">

<div class="highlight">

    class A:
      pass

    A = foo(bar(A))

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-18" class="target"></span><a href="https://www.python.org/dev/peps/pep-3129" class="pep reference external"><strong>PEP 3129</strong></a> - Class Decorators  
PEP written by Collin Winter.

</div>

</div>

<div id="pep-3141-a-type-hierarchy-for-numbers" class="section">

<span id="pep-3141"></span>

## PEP 3141: A Type Hierarchy for Numbers<a href="#pep-3141-a-type-hierarchy-for-numbers" class="headerlink" title="Permalink to this headline">¶</a>

Python 3.0 adds several abstract base classes for numeric types inspired by Scheme’s numeric tower. These classes were backported to 2.6 as the <a href="../library/numbers.html#module-numbers" class="reference internal" title="numbers: Numeric abstract base classes (Complex, Real, Integral, etc.)."><span class="pre"><code class="sourceCode python">numbers</code></span></a> module.

The most general ABC is <span class="pre">`Number`</span>. It defines no operations at all, and only exists to allow checking if an object is a number by doing <span class="pre">`isinstance(obj,`</span>` `<span class="pre">`Number)`</span>.

<span class="pre">`Complex`</span> is a subclass of <span class="pre">`Number`</span>. Complex numbers can undergo the basic operations of addition, subtraction, multiplication, division, and exponentiation, and you can retrieve the real and imaginary parts and obtain a number’s conjugate. Python’s built-in complex type is an implementation of <span class="pre">`Complex`</span>.

<span class="pre">`Real`</span> further derives from <span class="pre">`Complex`</span>, and adds operations that only work on real numbers: <span class="pre">`floor()`</span>, <span class="pre">`trunc()`</span>, rounding, taking the remainder mod N, floor division, and comparisons.

<span class="pre">`Rational`</span> numbers derive from <span class="pre">`Real`</span>, have <span class="pre">`numerator`</span> and <span class="pre">`denominator`</span> properties, and can be converted to floats. Python 2.6 adds a simple rational-number class, <span class="pre">`Fraction`</span>, in the <a href="../library/fractions.html#module-fractions" class="reference internal" title="fractions: Rational numbers."><span class="pre"><code class="sourceCode python">fractions</code></span></a> module. (It’s called <span class="pre">`Fraction`</span> instead of <span class="pre">`Rational`</span> to avoid a name clash with <a href="../library/numbers.html#numbers.Rational" class="reference internal" title="numbers.Rational"><span class="pre"><code class="sourceCode python">numbers.Rational</code></span></a>.)

<span class="pre">`Integral`</span> numbers derive from <span class="pre">`Rational`</span>, and can be shifted left and right with <span class="pre">`<<`</span> and <span class="pre">`>>`</span>, combined using bitwise operations such as <span class="pre">`&`</span> and <span class="pre">`|`</span>, and can be used as array indexes and slice boundaries.

In Python 3.0, the PEP slightly redefines the existing builtins <a href="../library/functions.html#round" class="reference internal" title="round"><span class="pre"><code class="sourceCode python"><span class="bu">round</span>()</code></span></a>, <a href="../library/math.html#math.floor" class="reference internal" title="math.floor"><span class="pre"><code class="sourceCode python">math.floor()</code></span></a>, <a href="../library/math.html#math.ceil" class="reference internal" title="math.ceil"><span class="pre"><code class="sourceCode python">math.ceil()</code></span></a>, and adds a new one, <a href="../library/math.html#math.trunc" class="reference internal" title="math.trunc"><span class="pre"><code class="sourceCode python">math.trunc()</code></span></a>, that’s been backported to Python 2.6. <a href="../library/math.html#math.trunc" class="reference internal" title="math.trunc"><span class="pre"><code class="sourceCode python">math.trunc()</code></span></a> rounds toward zero, returning the closest <span class="pre">`Integral`</span> that’s between the function’s argument and zero.

<div class="admonition seealso">

See also

<span id="index-19" class="target"></span><a href="https://www.python.org/dev/peps/pep-3141" class="pep reference external"><strong>PEP 3141</strong></a> - A Type Hierarchy for Numbers  
PEP written by Jeffrey Yasskin.

<a href="https://www.gnu.org/software/guile/manual/html_node/Numerical-Tower.html#Numerical-Tower" class="reference external">Scheme’s numerical tower</a>, from the Guile manual.

<a href="http://schemers.org/Documents/Standards/R5RS/HTML/r5rs-Z-H-9.html#%_sec_6.2" class="reference external">Scheme’s number datatypes</a> from the R5RS Scheme specification.

</div>

<div id="the-fractions-module" class="section">

### The <a href="../library/fractions.html#module-fractions" class="reference internal" title="fractions: Rational numbers."><span class="pre"><code class="sourceCode python">fractions</code></span></a> Module<a href="#the-fractions-module" class="headerlink" title="Permalink to this headline">¶</a>

To fill out the hierarchy of numeric types, the <a href="../library/fractions.html#module-fractions" class="reference internal" title="fractions: Rational numbers."><span class="pre"><code class="sourceCode python">fractions</code></span></a> module provides a rational-number class. Rational numbers store their values as a numerator and denominator forming a fraction, and can exactly represent numbers such as <span class="pre">`2/3`</span> that floating-point numbers can only approximate.

The <span class="pre">`Fraction`</span> constructor takes two <span class="pre">`Integral`</span> values that will be the numerator and denominator of the resulting fraction.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> from fractions import Fraction
    >>> a = Fraction(2, 3)
    >>> b = Fraction(2, 5)
    >>> float(a), float(b)
    (0.66666666666666663, 0.40000000000000002)
    >>> a+b
    Fraction(16, 15)
    >>> a/b
    Fraction(5, 3)

</div>

</div>

For converting floating-point numbers to rationals, the float type now has an <span class="pre">`as_integer_ratio()`</span> method that returns the numerator and denominator for a fraction that evaluates to the same floating-point value:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> (2.5) .as_integer_ratio()
    (5, 2)
    >>> (3.1415) .as_integer_ratio()
    (7074029114692207L, 2251799813685248L)
    >>> (1./3) .as_integer_ratio()
    (6004799503160661L, 18014398509481984L)

</div>

</div>

Note that values that can only be approximated by floating-point numbers, such as 1./3, are not simplified to the number being approximated; the fraction attempts to match the floating-point value **exactly**.

The <a href="../library/fractions.html#module-fractions" class="reference internal" title="fractions: Rational numbers."><span class="pre"><code class="sourceCode python">fractions</code></span></a> module is based upon an implementation by Sjoerd Mullender that was in Python’s <span class="pre">`Demo/classes/`</span> directory for a long time. This implementation was significantly updated by Jeffrey Yasskin.

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Permalink to this headline">¶</a>

Some smaller changes made to the core Python language are:

- Directories and zip archives containing a <span class="pre">`__main__.py`</span> file can now be executed directly by passing their name to the interpreter. The directory or zip archive is automatically inserted as the first entry in sys.path. (Suggestion and initial patch by Andy Chu, subsequently revised by Phillip J. Eby and Nick Coghlan; <a href="https://bugs.python.org/issue1739468" class="reference external">bpo-1739468</a>.)

- The <a href="../library/functions.html#hasattr" class="reference internal" title="hasattr"><span class="pre"><code class="sourceCode python"><span class="bu">hasattr</span>()</code></span></a> function was catching and ignoring all errors, under the assumption that they meant a <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> method was failing somehow and the return value of <a href="../library/functions.html#hasattr" class="reference internal" title="hasattr"><span class="pre"><code class="sourceCode python"><span class="bu">hasattr</span>()</code></span></a> would therefore be <span class="pre">`False`</span>. This logic shouldn’t be applied to <a href="../library/exceptions.html#exceptions.KeyboardInterrupt" class="reference internal" title="exceptions.KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a> and <a href="../library/exceptions.html#exceptions.SystemExit" class="reference internal" title="exceptions.SystemExit"><span class="pre"><code class="sourceCode python"><span class="pp">SystemExit</span></code></span></a>, however; Python 2.6 will no longer discard such exceptions when <a href="../library/functions.html#hasattr" class="reference internal" title="hasattr"><span class="pre"><code class="sourceCode python"><span class="bu">hasattr</span>()</code></span></a> encounters them. (Fixed by Benjamin Peterson; <a href="https://bugs.python.org/issue2196" class="reference external">bpo-2196</a>.)

- When calling a function using the <span class="pre">`**`</span> syntax to provide keyword arguments, you are no longer required to use a Python dictionary; any mapping will now work:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> def f(**kw):
      ...    print sorted(kw)
      ...
      >>> ud=UserDict.UserDict()
      >>> ud['a'] = 1
      >>> ud['b'] = 'string'
      >>> f(**ud)
      ['a', 'b']

  </div>

  </div>

  (Contributed by Alexander Belopolsky; <a href="https://bugs.python.org/issue1686487" class="reference external">bpo-1686487</a>.)

  It’s also become legal to provide keyword arguments after a <span class="pre">`*args`</span> argument to a function call.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> def f(*args, **kw):
      ...     print args, kw
      ...
      >>> f(1,2,3, *(4,5,6), keyword=13)
      (1, 2, 3, 4, 5, 6) {'keyword': 13}

  </div>

  </div>

  Previously this would have been a syntax error. (Contributed by Amaury Forgeot d’Arc; <a href="https://bugs.python.org/issue3473" class="reference external">bpo-3473</a>.)

- A new builtin, <span class="pre">`next(iterator,`</span>` `<span class="pre">`[default])`</span> returns the next item from the specified iterator. If the *default* argument is supplied, it will be returned if *iterator* has been exhausted; otherwise, the <a href="../library/exceptions.html#exceptions.StopIteration" class="reference internal" title="exceptions.StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a> exception will be raised. (Backported in <a href="https://bugs.python.org/issue2719" class="reference external">bpo-2719</a>.)

- Tuples now have <span class="pre">`index()`</span> and <span class="pre">`count()`</span> methods matching the list type’s <span class="pre">`index()`</span> and <span class="pre">`count()`</span> methods:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> t = (0,1,2,3,4,0,1,2)
      >>> t.index(3)
      3
      >>> t.count(0)
      2

  </div>

  </div>

  (Contributed by Raymond Hettinger)

- The built-in types now have improved support for extended slicing syntax, accepting various combinations of <span class="pre">`(start,`</span>` `<span class="pre">`stop,`</span>` `<span class="pre">`step)`</span>. Previously, the support was partial and certain corner cases wouldn’t work. (Implemented by Thomas Wouters.)

- Properties now have three attributes, <span class="pre">`getter`</span>, <span class="pre">`setter`</span> and <span class="pre">`deleter`</span>, that are decorators providing useful shortcuts for adding a getter, setter or deleter function to an existing property. You would use them like this:

  <div class="highlight-default notranslate">

  <div class="highlight">

      class C(object):
          @property
          def x(self):
              return self._x

          @x.setter
          def x(self, value):
              self._x = value

          @x.deleter
          def x(self):
              del self._x

      class D(C):
          @C.x.getter
          def x(self):
              return self._x * 2

          @x.setter
          def x(self, value):
              self._x = value / 2

  </div>

  </div>

- Several methods of the built-in set types now accept multiple iterables: <span class="pre">`intersection()`</span>, <span class="pre">`intersection_update()`</span>, <span class="pre">`union()`</span>, <span class="pre">`update()`</span>, <span class="pre">`difference()`</span> and <span class="pre">`difference_update()`</span>.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> s=set('1234567890')
      >>> s.intersection('abc123', 'cdf246')  # Intersection between all inputs
      set(['2'])
      >>> s.difference('246', '789')
      set(['1', '0', '3', '5'])

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- Many floating-point features were added. The <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span>()</code></span></a> function will now turn the string <span class="pre">`nan`</span> into an IEEE 754 Not A Number value, and <span class="pre">`+inf`</span> and <span class="pre">`-inf`</span> into positive or negative infinity. This works on any platform with IEEE 754 semantics. (Contributed by Christian Heimes; <a href="https://bugs.python.org/issue1635" class="reference external">bpo-1635</a>.)

  Other functions in the <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> module, <span class="pre">`isinf()`</span> and <span class="pre">`isnan()`</span>, return true if their floating-point argument is infinite or Not A Number. (<a href="https://bugs.python.org/issue1640" class="reference external">bpo-1640</a>)

  Conversion functions were added to convert floating-point numbers into hexadecimal strings (<a href="https://bugs.python.org/issue3008" class="reference external">bpo-3008</a>). These functions convert floats to and from a string representation without introducing rounding errors from the conversion between decimal and binary. Floats have a <a href="../library/functions.html#hex" class="reference internal" title="hex"><span class="pre"><code class="sourceCode python"><span class="bu">hex</span>()</code></span></a> method that returns a string representation, and the <span class="pre">`float.fromhex()`</span> method converts a string back into a number:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> a = 3.75
      >>> a.hex()
      '0x1.e000000000000p+1'
      >>> float.fromhex('0x1.e000000000000p+1')
      3.75
      >>> b=1./3
      >>> b.hex()
      '0x1.5555555555555p-2'

  </div>

  </div>

- A numerical nicety: when creating a complex number from two floats on systems that support signed zeros (-0 and +0), the <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span>()</code></span></a> constructor will now preserve the sign of the zero. (Fixed by Mark T. Dickinson; <a href="https://bugs.python.org/issue1507" class="reference external">bpo-1507</a>.)

- Classes that inherit a <a href="../reference/datamodel.html#object.__hash__" class="reference internal" title="object.__hash__"><span class="pre"><code class="sourceCode python"><span class="fu">__hash__</span>()</code></span></a> method from a parent class can set <span class="pre">`__hash__`</span>` `<span class="pre">`=`</span>` `<span class="pre">`None`</span> to indicate that the class isn’t hashable. This will make <span class="pre">`hash(obj)`</span> raise a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> and the class will not be indicated as implementing the <span class="pre">`Hashable`</span> ABC.

  You should do this when you’ve defined a <a href="../reference/datamodel.html#object.__cmp__" class="reference internal" title="object.__cmp__"><span class="pre"><code class="sourceCode python"><span class="fu">__cmp__</span>()</code></span></a> or <a href="../reference/datamodel.html#object.__eq__" class="reference internal" title="object.__eq__"><span class="pre"><code class="sourceCode python"><span class="fu">__eq__</span>()</code></span></a> method that compares objects by their value rather than by identity. All objects have a default hash method that uses <span class="pre">`id(obj)`</span> as the hash value. There’s no tidy way to remove the <a href="../reference/datamodel.html#object.__hash__" class="reference internal" title="object.__hash__"><span class="pre"><code class="sourceCode python"><span class="fu">__hash__</span>()</code></span></a> method inherited from a parent class, so assigning <span class="pre">`None`</span> was implemented as an override. At the C level, extensions can set <span class="pre">`tp_hash`</span> to <a href="../c-api/object.html#c.PyObject_HashNotImplemented" class="reference internal" title="PyObject_HashNotImplemented"><span class="pre"><code class="sourceCode c">PyObject_HashNotImplemented<span class="op">()</span></code></span></a>. (Fixed by Nick Coghlan and Amaury Forgeot d’Arc; <a href="https://bugs.python.org/issue2235" class="reference external">bpo-2235</a>.)

- The <a href="../library/exceptions.html#exceptions.GeneratorExit" class="reference internal" title="exceptions.GeneratorExit"><span class="pre"><code class="sourceCode python"><span class="pp">GeneratorExit</span></code></span></a> exception now subclasses <a href="../library/exceptions.html#exceptions.BaseException" class="reference internal" title="exceptions.BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a> instead of <a href="../library/exceptions.html#exceptions.Exception" class="reference internal" title="exceptions.Exception"><span class="pre"><code class="sourceCode python"><span class="pp">Exception</span></code></span></a>. This means that an exception handler that does <span class="pre">`except`</span>` `<span class="pre">`Exception:`</span> will not inadvertently catch <a href="../library/exceptions.html#exceptions.GeneratorExit" class="reference internal" title="exceptions.GeneratorExit"><span class="pre"><code class="sourceCode python"><span class="pp">GeneratorExit</span></code></span></a>. (Contributed by Chad Austin; <a href="https://bugs.python.org/issue1537" class="reference external">bpo-1537</a>.)

- Generator objects now have a <span class="pre">`gi_code`</span> attribute that refers to the original code object backing the generator. (Contributed by Collin Winter; <a href="https://bugs.python.org/issue1473257" class="reference external">bpo-1473257</a>.)

- The <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a> built-in function now accepts keyword arguments as well as positional parameters. (Contributed by Thomas Wouters; <a href="https://bugs.python.org/issue1444529" class="reference external">bpo-1444529</a>.)

- The <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span>()</code></span></a> constructor now accepts strings containing parenthesized complex numbers, meaning that <span class="pre">`complex(repr(cplx))`</span> will now round-trip values. For example, <span class="pre">`complex('(3+4j)')`</span> now returns the value (3+4j). (<a href="https://bugs.python.org/issue1491866" class="reference external">bpo-1491866</a>)

- The string <span class="pre">`translate()`</span> method now accepts <span class="pre">`None`</span> as the translation table parameter, which is treated as the identity transformation. This makes it easier to carry out operations that only delete characters. (Contributed by Bengt Richter and implemented by Raymond Hettinger; <a href="https://bugs.python.org/issue1193128" class="reference external">bpo-1193128</a>.)

- The built-in <a href="../library/functions.html#dir" class="reference internal" title="dir"><span class="pre"><code class="sourceCode python"><span class="bu">dir</span>()</code></span></a> function now checks for a <span class="pre">`__dir__()`</span> method on the objects it receives. This method must return a list of strings containing the names of valid attributes for the object, and lets the object control the value that <a href="../library/functions.html#dir" class="reference internal" title="dir"><span class="pre"><code class="sourceCode python"><span class="bu">dir</span>()</code></span></a> produces. Objects that have <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> or <a href="../reference/datamodel.html#object.__getattribute__" class="reference internal" title="object.__getattribute__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattribute__</span>()</code></span></a> methods can use this to advertise pseudo-attributes they will honor. (<a href="https://bugs.python.org/issue1591665" class="reference external">bpo-1591665</a>)

- Instance method objects have new attributes for the object and function comprising the method; the new synonym for <span class="pre">`im_self`</span> is <span class="pre">`__self__`</span>, and <span class="pre">`im_func`</span> is also available as <span class="pre">`__func__`</span>. The old names are still supported in Python 2.6, but are gone in 3.0.

- An obscure change: when you use the <a href="../library/functions.html#locals" class="reference internal" title="locals"><span class="pre"><code class="sourceCode python"><span class="bu">locals</span>()</code></span></a> function inside a <a href="../reference/compound_stmts.html#class" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">class</code></span></a> statement, the resulting dictionary no longer returns free variables. (Free variables, in this case, are variables referenced in the <a href="../reference/compound_stmts.html#class" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">class</code></span></a> statement that aren’t attributes of the class.)

<div id="optimizations" class="section">

### Optimizations<a href="#optimizations" class="headerlink" title="Permalink to this headline">¶</a>

- The <a href="../library/warnings.html#module-warnings" class="reference internal" title="warnings: Issue warning messages and control their disposition."><span class="pre"><code class="sourceCode python">warnings</code></span></a> module has been rewritten in C. This makes it possible to invoke warnings from the parser, and may also make the interpreter’s startup faster. (Contributed by Neal Norwitz and Brett Cannon; <a href="https://bugs.python.org/issue1631171" class="reference external">bpo-1631171</a>.)

- Type objects now have a cache of methods that can reduce the work required to find the correct method implementation for a particular class; once cached, the interpreter doesn’t need to traverse base classes to figure out the right method to call. The cache is cleared if a base class or the class itself is modified, so the cache should remain correct even in the face of Python’s dynamic nature. (Original optimization implemented by Armin Rigo, updated for Python 2.6 by Kevin Jacobs; <a href="https://bugs.python.org/issue1700288" class="reference external">bpo-1700288</a>.)

  By default, this change is only applied to types that are included with the Python core. Extension modules may not necessarily be compatible with this cache, so they must explicitly add <span class="pre">`Py_TPFLAGS_HAVE_VERSION_TAG`</span> to the module’s <span class="pre">`tp_flags`</span> field to enable the method cache. (To be compatible with the method cache, the extension module’s code must not directly access and modify the <span class="pre">`tp_dict`</span> member of any of the types it implements. Most modules don’t do this, but it’s impossible for the Python interpreter to determine that. See <a href="https://bugs.python.org/issue1878" class="reference external">bpo-1878</a> for some discussion.)

- Function calls that use keyword arguments are significantly faster by doing a quick pointer comparison, usually saving the time of a full string comparison. (Contributed by Raymond Hettinger, after an initial implementation by Antoine Pitrou; <a href="https://bugs.python.org/issue1819" class="reference external">bpo-1819</a>.)

- All of the functions in the <a href="../library/struct.html#module-struct" class="reference internal" title="struct: Interpret strings as packed binary data."><span class="pre"><code class="sourceCode python">struct</code></span></a> module have been rewritten in C, thanks to work at the Need For Speed sprint. (Contributed by Raymond Hettinger.)

- Some of the standard built-in types now set a bit in their type objects. This speeds up checking whether an object is a subclass of one of these types. (Contributed by Neal Norwitz.)

- Unicode strings now use faster code for detecting whitespace and line breaks; this speeds up the <span class="pre">`split()`</span> method by about 25% and <span class="pre">`splitlines()`</span> by 35%. (Contributed by Antoine Pitrou.) Memory usage is reduced by using pymalloc for the Unicode string’s data.

- The <span class="pre">`with`</span> statement now stores the <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> method on the stack, producing a small speedup. (Implemented by Jeffrey Yasskin.)

- To reduce memory usage, the garbage collector will now clear internal free lists when garbage-collecting the highest generation of objects. This may return memory to the operating system sooner.

</div>

<div id="interpreter-changes" class="section">

<span id="new-26-interpreter"></span>

### Interpreter Changes<a href="#interpreter-changes" class="headerlink" title="Permalink to this headline">¶</a>

Two command-line options have been reserved for use by other Python implementations. The <a href="../using/cmdline.html#cmdoption-j" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-J</code></span></a> switch has been reserved for use by Jython for Jython-specific options, such as switches that are passed to the underlying JVM. <a href="../using/cmdline.html#id5" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-X</code></span></a> has been reserved for options specific to a particular implementation of Python such as CPython, Jython, or IronPython. If either option is used with Python 2.6, the interpreter will report that the option isn’t currently used.

Python can now be prevented from writing <span class="pre">`.pyc`</span> or <span class="pre">`.pyo`</span> files by supplying the <a href="../using/cmdline.html#id1" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-B</code></span></a> switch to the Python interpreter, or by setting the <span id="index-20" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONDONTWRITEBYTECODE" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONDONTWRITEBYTECODE</code></span></a> environment variable before running the interpreter. This setting is available to Python programs as the <span class="pre">`sys.dont_write_bytecode`</span> variable, and Python code can change the value to modify the interpreter’s behaviour. (Contributed by Neal Norwitz and Georg Brandl.)

The encoding used for standard input, output, and standard error can be specified by setting the <span id="index-21" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONIOENCODING" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONIOENCODING</code></span></a> environment variable before running the interpreter. The value should be a string in the form <span class="pre">`<encoding>`</span> or <span class="pre">`<encoding>:<errorhandler>`</span>. The *encoding* part specifies the encoding’s name, e.g. <span class="pre">`utf-8`</span> or <span class="pre">`latin-1`</span>; the optional *errorhandler* part specifies what to do with characters that can’t be handled by the encoding, and should be one of “error”, “ignore”, or “replace”. (Contributed by Martin von Loewis.)

</div>

</div>

<div id="new-and-improved-modules" class="section">

## New and Improved Modules<a href="#new-and-improved-modules" class="headerlink" title="Permalink to this headline">¶</a>

As in every release, Python’s standard library received a number of enhancements and bug fixes. Here’s a partial list of the most notable changes, sorted alphabetically by module name. Consult the <span class="pre">`Misc/NEWS`</span> file in the source tree for a more complete list of changes, or look through the Subversion logs for all the details.

- The <a href="../library/asyncore.html#module-asyncore" class="reference internal" title="asyncore: A base class for developing asynchronous socket handling services."><span class="pre"><code class="sourceCode python">asyncore</code></span></a> and <a href="../library/asynchat.html#module-asynchat" class="reference internal" title="asynchat: Support for asynchronous command/response protocols."><span class="pre"><code class="sourceCode python">asynchat</code></span></a> modules are being actively maintained again, and a number of patches and bugfixes were applied. (Maintained by Josiah Carlson; see <a href="https://bugs.python.org/issue1736190" class="reference external">bpo-1736190</a> for one patch.)

- The <a href="../library/bsddb.html#module-bsddb" class="reference internal" title="bsddb: Interface to Berkeley DB database library"><span class="pre"><code class="sourceCode python">bsddb</code></span></a> module also has a new maintainer, Jesús Cea Avion, and the package is now available as a standalone package. The web page for the package is <a href="https://www.jcea.es/programacion/pybsddb.htm" class="reference external">www.jcea.es/programacion/pybsddb.htm</a>. The plan is to remove the package from the standard library in Python 3.0, because its pace of releases is much more frequent than Python’s.

  The <span class="pre">`bsddb.dbshelve`</span> module now uses the highest pickling protocol available, instead of restricting itself to protocol 1. (Contributed by W. Barnes.)

- The <a href="../library/cgi.html#module-cgi" class="reference internal" title="cgi: Helpers for running Python scripts via the Common Gateway Interface."><span class="pre"><code class="sourceCode python">cgi</code></span></a> module will now read variables from the query string of an HTTP POST request. This makes it possible to use form actions with URLs that include query strings such as “/cgi-bin/add.py?category=1”. (Contributed by Alexandre Fiori and Nubis; <a href="https://bugs.python.org/issue1817" class="reference external">bpo-1817</a>.)

  The <span class="pre">`parse_qs()`</span> and <span class="pre">`parse_qsl()`</span> functions have been relocated from the <a href="../library/cgi.html#module-cgi" class="reference internal" title="cgi: Helpers for running Python scripts via the Common Gateway Interface."><span class="pre"><code class="sourceCode python">cgi</code></span></a> module to the <a href="../library/urlparse.html#module-urlparse" class="reference internal" title="urlparse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urlparse</code></span></a> module. The versions still available in the <a href="../library/cgi.html#module-cgi" class="reference internal" title="cgi: Helpers for running Python scripts via the Common Gateway Interface."><span class="pre"><code class="sourceCode python">cgi</code></span></a> module will trigger <a href="../library/exceptions.html#exceptions.PendingDeprecationWarning" class="reference internal" title="exceptions.PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a> messages in 2.6 (<a href="https://bugs.python.org/issue600362" class="reference external">bpo-600362</a>).

- The <a href="../library/cmath.html#module-cmath" class="reference internal" title="cmath: Mathematical functions for complex numbers."><span class="pre"><code class="sourceCode python">cmath</code></span></a> module underwent extensive revision, contributed by Mark Dickinson and Christian Heimes. Five new functions were added:

  - <span class="pre">`polar()`</span> converts a complex number to polar form, returning the modulus and argument of the complex number.

  - <span class="pre">`rect()`</span> does the opposite, turning a modulus, argument pair back into the corresponding complex number.

  - <span class="pre">`phase()`</span> returns the argument (also called the angle) of a complex number.

  - <span class="pre">`isnan()`</span> returns True if either the real or imaginary part of its argument is a NaN.

  - <span class="pre">`isinf()`</span> returns True if either the real or imaginary part of its argument is infinite.

  The revisions also improved the numerical soundness of the <a href="../library/cmath.html#module-cmath" class="reference internal" title="cmath: Mathematical functions for complex numbers."><span class="pre"><code class="sourceCode python">cmath</code></span></a> module. For all functions, the real and imaginary parts of the results are accurate to within a few units of least precision (ulps) whenever possible. See <a href="https://bugs.python.org/issue1381" class="reference external">bpo-1381</a> for the details. The branch cuts for <span class="pre">`asinh()`</span>, <span class="pre">`atanh()`</span>: and <span class="pre">`atan()`</span> have also been corrected.

  The tests for the module have been greatly expanded; nearly 2000 new test cases exercise the algebraic functions.

  On IEEE 754 platforms, the <a href="../library/cmath.html#module-cmath" class="reference internal" title="cmath: Mathematical functions for complex numbers."><span class="pre"><code class="sourceCode python">cmath</code></span></a> module now handles IEEE 754 special values and floating-point exceptions in a manner consistent with Annex ‘G’ of the C99 standard.

- A new data type in the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module: <span class="pre">`namedtuple(typename,`</span>` `<span class="pre">`fieldnames)`</span> is a factory function that creates subclasses of the standard tuple whose fields are accessible by name as well as index. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> var_type = collections.namedtuple('variable',
      ...             'id name type size')
      >>> # Names are separated by spaces or commas.
      >>> # 'id, name, type, size' would also work.
      >>> var_type._fields
      ('id', 'name', 'type', 'size')

      >>> var = var_type(1, 'frequency', 'int', 4)
      >>> print var[0], var.id    # Equivalent
      1 1
      >>> print var[2], var.type  # Equivalent
      int int
      >>> var._asdict()
      {'size': 4, 'type': 'int', 'id': 1, 'name': 'frequency'}
      >>> v2 = var._replace(name='amplitude')
      >>> v2
      variable(id=1, name='amplitude', type='int', size=4)

  </div>

  </div>

  Several places in the standard library that returned tuples have been modified to return <span class="pre">`namedtuple`</span> instances. For example, the <span class="pre">`Decimal.as_tuple()`</span> method now returns a named tuple with <span class="pre">`sign`</span>, <span class="pre">`digits`</span>, and <span class="pre">`exponent`</span> fields.

  (Contributed by Raymond Hettinger.)

- Another change to the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module is that the <span class="pre">`deque`</span> type now supports an optional *maxlen* parameter; if supplied, the deque’s size will be restricted to no more than *maxlen* items. Adding more items to a full deque causes old items to be discarded.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> from collections import deque
      >>> dq=deque(maxlen=3)
      >>> dq
      deque([], maxlen=3)
      >>> dq.append(1); dq.append(2); dq.append(3)
      >>> dq
      deque([1, 2, 3], maxlen=3)
      >>> dq.append(4)
      >>> dq
      deque([2, 3, 4], maxlen=3)

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- The <a href="../library/cookie.html#module-Cookie" class="reference internal" title="Cookie: Support for HTTP state management (cookies)."><span class="pre"><code class="sourceCode python">Cookie</code></span></a> module’s <span class="pre">`Morsel`</span> objects now support an <span class="pre">`httponly`</span> attribute. In some browsers. cookies with this attribute set cannot be accessed or manipulated by JavaScript code. (Contributed by Arvin Schnell; <a href="https://bugs.python.org/issue1638033" class="reference external">bpo-1638033</a>.)

- A new window method in the <a href="../library/curses.html#module-curses" class="reference internal" title="curses: An interface to the curses library, providing portable terminal handling. (Unix)"><span class="pre"><code class="sourceCode python">curses</code></span></a> module, <span class="pre">`chgat()`</span>, changes the display attributes for a certain number of characters on a single line. (Contributed by Fabian Kreutz.)

  <div class="highlight-default notranslate">

  <div class="highlight">

      # Boldface text starting at y=0,x=21
      # and affecting the rest of the line.
      stdscr.chgat(0, 21, curses.A_BOLD)

  </div>

  </div>

  The <span class="pre">`Textbox`</span> class in the <a href="../library/curses.html#module-curses.textpad" class="reference internal" title="curses.textpad: Emacs-like input editing in a curses window."><span class="pre"><code class="sourceCode python">curses.textpad</code></span></a> module now supports editing in insert mode as well as overwrite mode. Insert mode is enabled by supplying a true value for the *insert_mode* parameter when creating the <span class="pre">`Textbox`</span> instance.

- The <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> module’s <span class="pre">`strftime()`</span> methods now support a <span class="pre">`%f`</span> format code that expands to the number of microseconds in the object, zero-padded on the left to six places. (Contributed by Skip Montanaro; <a href="https://bugs.python.org/issue1158" class="reference external">bpo-1158</a>.)

- The <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic  Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> module was updated to version 1.66 of <a href="http://speleotrove.com/decimal/decarith.html" class="reference external">the General Decimal Specification</a>. New features include some methods for some basic mathematical functions such as <span class="pre">`exp()`</span> and <span class="pre">`log10()`</span>:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> Decimal(1).exp()
      Decimal("2.718281828459045235360287471")
      >>> Decimal("2.7182818").ln()
      Decimal("0.9999999895305022877376682436")
      >>> Decimal(1000).log10()
      Decimal("3")

  </div>

  </div>

  The <span class="pre">`as_tuple()`</span> method of <span class="pre">`Decimal`</span> objects now returns a named tuple with <span class="pre">`sign`</span>, <span class="pre">`digits`</span>, and <span class="pre">`exponent`</span> fields.

  (Implemented by Facundo Batista and Mark Dickinson. Named tuple support added by Raymond Hettinger.)

- The <a href="../library/difflib.html#module-difflib" class="reference internal" title="difflib: Helpers for computing differences between objects."><span class="pre"><code class="sourceCode python">difflib</code></span></a> module’s <span class="pre">`SequenceMatcher`</span> class now returns named tuples representing matches, with <span class="pre">`a`</span>, <span class="pre">`b`</span>, and <span class="pre">`size`</span> attributes. (Contributed by Raymond Hettinger.)

- An optional <span class="pre">`timeout`</span> parameter, specifying a timeout measured in seconds, was added to the <a href="../library/ftplib.html#ftplib.FTP" class="reference internal" title="ftplib.FTP"><span class="pre"><code class="sourceCode python">ftplib.FTP</code></span></a> class constructor as well as the <span class="pre">`connect()`</span> method. (Added by Facundo Batista.) Also, the <span class="pre">`FTP`</span> class’s <span class="pre">`storbinary()`</span> and <span class="pre">`storlines()`</span> now take an optional *callback* parameter that will be called with each block of data after the data has been sent. (Contributed by Phil Schwartz; <a href="https://bugs.python.org/issue1221598" class="reference external">bpo-1221598</a>.)

- The <a href="../library/functions.html#reduce" class="reference internal" title="reduce"><span class="pre"><code class="sourceCode python"><span class="bu">reduce</span>()</code></span></a> built-in function is also available in the <a href="../library/functools.html#module-functools" class="reference internal" title="functools: Higher-order functions and operations on callable objects."><span class="pre"><code class="sourceCode python">functools</code></span></a> module. In Python 3.0, the builtin has been dropped and <a href="../library/functions.html#reduce" class="reference internal" title="reduce"><span class="pre"><code class="sourceCode python"><span class="bu">reduce</span>()</code></span></a> is only available from <a href="../library/functools.html#module-functools" class="reference internal" title="functools: Higher-order functions and operations on callable objects."><span class="pre"><code class="sourceCode python">functools</code></span></a>; currently there are no plans to drop the builtin in the 2.x series. (Patched by Christian Heimes; <a href="https://bugs.python.org/issue1739906" class="reference external">bpo-1739906</a>.)

- When possible, the <a href="../library/getpass.html#module-getpass" class="reference internal" title="getpass: Portable reading of passwords and retrieval of the userid."><span class="pre"><code class="sourceCode python">getpass</code></span></a> module will now use <span class="pre">`/dev/tty`</span> to print a prompt message and read the password, falling back to standard error and standard input. If the password may be echoed to the terminal, a warning is printed before the prompt is displayed. (Contributed by Gregory P. Smith.)

- The <a href="../library/glob.html#glob.glob" class="reference internal" title="glob.glob"><span class="pre"><code class="sourceCode python">glob.glob()</code></span></a> function can now return Unicode filenames if a Unicode path was used and Unicode filenames are matched within the directory. (<a href="https://bugs.python.org/issue1001604" class="reference external">bpo-1001604</a>)

- A new function in the <a href="../library/heapq.html#module-heapq" class="reference internal" title="heapq: Heap queue algorithm (a.k.a. priority queue)."><span class="pre"><code class="sourceCode python">heapq</code></span></a> module, <span class="pre">`merge(iter1,`</span>` `<span class="pre">`iter2,`</span>` `<span class="pre">`...)`</span>, takes any number of iterables returning data in sorted order, and returns a new generator that returns the contents of all the iterators, also in sorted order. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> list(heapq.merge([1, 3, 5, 9], [2, 8, 16]))
      [1, 2, 3, 5, 8, 9, 16]

  </div>

  </div>

  Another new function, <span class="pre">`heappushpop(heap,`</span>` `<span class="pre">`item)`</span>, pushes *item* onto *heap*, then pops off and returns the smallest item. This is more efficient than making a call to <span class="pre">`heappush()`</span> and then <span class="pre">`heappop()`</span>.

  <a href="../library/heapq.html#module-heapq" class="reference internal" title="heapq: Heap queue algorithm (a.k.a. priority queue)."><span class="pre"><code class="sourceCode python">heapq</code></span></a> is now implemented to only use less-than comparison, instead of the less-than-or-equal comparison it previously used. This makes <a href="../library/heapq.html#module-heapq" class="reference internal" title="heapq: Heap queue algorithm (a.k.a. priority queue)."><span class="pre"><code class="sourceCode python">heapq</code></span></a>’s usage of a type match the <span class="pre">`list.sort()`</span> method. (Contributed by Raymond Hettinger.)

- An optional <span class="pre">`timeout`</span> parameter, specifying a timeout measured in seconds, was added to the <a href="../library/httplib.html#httplib.HTTPConnection" class="reference internal" title="httplib.HTTPConnection"><span class="pre"><code class="sourceCode python">httplib.HTTPConnection</code></span></a> and <span class="pre">`HTTPSConnection`</span> class constructors. (Added by Facundo Batista.)

- Most of the <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> module’s functions, such as <span class="pre">`getmoduleinfo()`</span> and <span class="pre">`getargs()`</span>, now return named tuples. In addition to behaving like tuples, the elements of the return value can also be accessed as attributes. (Contributed by Raymond Hettinger.)

  Some new functions in the module include <span class="pre">`isgenerator()`</span>, <span class="pre">`isgeneratorfunction()`</span>, and <span class="pre">`isabstract()`</span>.

- The <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> module gained several new functions.

  <span class="pre">`izip_longest(iter1,`</span>` `<span class="pre">`iter2,`</span>` `<span class="pre">`...[,`</span>` `<span class="pre">`fillvalue])`</span> makes tuples from each of the elements; if some of the iterables are shorter than others, the missing values are set to *fillvalue*. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> tuple(itertools.izip_longest([1,2,3], [1,2,3,4,5]))
      ((1, 1), (2, 2), (3, 3), (None, 4), (None, 5))

  </div>

  </div>

  <span class="pre">`product(iter1,`</span>` `<span class="pre">`iter2,`</span>` `<span class="pre">`...,`</span>` `<span class="pre">`[repeat=N])`</span> returns the Cartesian product of the supplied iterables, a set of tuples containing every possible combination of the elements returned from each iterable.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> list(itertools.product([1,2,3], [4,5,6]))
      [(1, 4), (1, 5), (1, 6),
       (2, 4), (2, 5), (2, 6),
       (3, 4), (3, 5), (3, 6)]

  </div>

  </div>

  The optional *repeat* keyword argument is used for taking the product of an iterable or a set of iterables with themselves, repeated *N* times. With a single iterable argument, *N*-tuples are returned:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> list(itertools.product([1,2], repeat=3))
      [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
       (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)]

  </div>

  </div>

  With two iterables, *2N*-tuples are returned.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> list(itertools.product([1,2], [3,4], repeat=2))
      [(1, 3, 1, 3), (1, 3, 1, 4), (1, 3, 2, 3), (1, 3, 2, 4),
       (1, 4, 1, 3), (1, 4, 1, 4), (1, 4, 2, 3), (1, 4, 2, 4),
       (2, 3, 1, 3), (2, 3, 1, 4), (2, 3, 2, 3), (2, 3, 2, 4),
       (2, 4, 1, 3), (2, 4, 1, 4), (2, 4, 2, 3), (2, 4, 2, 4)]

  </div>

  </div>

  <span class="pre">`combinations(iterable,`</span>` `<span class="pre">`r)`</span> returns sub-sequences of length *r* from the elements of *iterable*.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> list(itertools.combinations('123', 2))
      [('1', '2'), ('1', '3'), ('2', '3')]
      >>> list(itertools.combinations('123', 3))
      [('1', '2', '3')]
      >>> list(itertools.combinations('1234', 3))
      [('1', '2', '3'), ('1', '2', '4'),
       ('1', '3', '4'), ('2', '3', '4')]

  </div>

  </div>

  <span class="pre">`permutations(iter[,`</span>` `<span class="pre">`r])`</span> returns all the permutations of length *r* of the iterable’s elements. If *r* is not specified, it will default to the number of elements produced by the iterable.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> list(itertools.permutations([1,2,3,4], 2))
      [(1, 2), (1, 3), (1, 4),
       (2, 1), (2, 3), (2, 4),
       (3, 1), (3, 2), (3, 4),
       (4, 1), (4, 2), (4, 3)]

  </div>

  </div>

  <span class="pre">`itertools.chain(*iterables)`</span> is an existing function in <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> that gained a new constructor in Python 2.6. <span class="pre">`itertools.chain.from_iterable(iterable)`</span> takes a single iterable that should return other iterables. <span class="pre">`chain()`</span> will then return all the elements of the first iterable, then all the elements of the second, and so on.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> list(itertools.chain.from_iterable([[1,2,3], [4,5,6]]))
      [1, 2, 3, 4, 5, 6]

  </div>

  </div>

  (All contributed by Raymond Hettinger.)

- The <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> module’s <span class="pre">`FileHandler`</span> class and its subclasses <span class="pre">`WatchedFileHandler`</span>, <span class="pre">`RotatingFileHandler`</span>, and <span class="pre">`TimedRotatingFileHandler`</span> now have an optional *delay* parameter to their constructors. If *delay* is true, opening of the log file is deferred until the first <span class="pre">`emit()`</span> call is made. (Contributed by Vinay Sajip.)

  <span class="pre">`TimedRotatingFileHandler`</span> also has a *utc* constructor parameter. If the argument is true, UTC time will be used in determining when midnight occurs and in generating filenames; otherwise local time will be used.

- Several new functions were added to the <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> module:

  - <a href="../library/math.html#math.isinf" class="reference internal" title="math.isinf"><span class="pre"><code class="sourceCode python">isinf()</code></span></a> and <a href="../library/math.html#math.isnan" class="reference internal" title="math.isnan"><span class="pre"><code class="sourceCode python">isnan()</code></span></a> determine whether a given float is a (positive or negative) infinity or a NaN (Not a Number), respectively.

  - <a href="../library/math.html#math.copysign" class="reference internal" title="math.copysign"><span class="pre"><code class="sourceCode python">copysign()</code></span></a> copies the sign bit of an IEEE 754 number, returning the absolute value of *x* combined with the sign bit of *y*. For example, <span class="pre">`math.copysign(1,`</span>` `<span class="pre">`-0.0)`</span> returns -1.0. (Contributed by Christian Heimes.)

  - <a href="../library/math.html#math.factorial" class="reference internal" title="math.factorial"><span class="pre"><code class="sourceCode python">factorial()</code></span></a> computes the factorial of a number. (Contributed by Raymond Hettinger; <a href="https://bugs.python.org/issue2138" class="reference external">bpo-2138</a>.)

  - <a href="../library/math.html#math.fsum" class="reference internal" title="math.fsum"><span class="pre"><code class="sourceCode python">fsum()</code></span></a> adds up the stream of numbers from an iterable, and is careful to avoid loss of precision through using partial sums. (Contributed by Jean Brouwers, Raymond Hettinger, and Mark Dickinson; <a href="https://bugs.python.org/issue2819" class="reference external">bpo-2819</a>.)

  - <a href="../library/math.html#math.acosh" class="reference internal" title="math.acosh"><span class="pre"><code class="sourceCode python">acosh()</code></span></a>, <a href="../library/math.html#math.asinh" class="reference internal" title="math.asinh"><span class="pre"><code class="sourceCode python">asinh()</code></span></a> and <a href="../library/math.html#math.atanh" class="reference internal" title="math.atanh"><span class="pre"><code class="sourceCode python">atanh()</code></span></a> compute the inverse hyperbolic functions.

  - <a href="../library/math.html#math.log1p" class="reference internal" title="math.log1p"><span class="pre"><code class="sourceCode python">log1p()</code></span></a> returns the natural logarithm of *1+x* (base *e*).

  - <span class="pre">`trunc()`</span> rounds a number toward zero, returning the closest <span class="pre">`Integral`</span> that’s between the function’s argument and zero. Added as part of the backport of <a href="#pep-3141" class="reference external">PEP 3141’s type hierarchy for numbers</a>.

- The <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> module has been improved to give more consistent behaviour across platforms, especially with respect to handling of floating-point exceptions and IEEE 754 special values.

  Whenever possible, the module follows the recommendations of the C99 standard about 754’s special values. For example, <span class="pre">`sqrt(-1.)`</span> should now give a <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> across almost all platforms, while <span class="pre">`sqrt(float('NaN'))`</span> should return a NaN on all IEEE 754 platforms. Where Annex ‘F’ of the C99 standard recommends signaling ‘divide-by-zero’ or ‘invalid’, Python will raise <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a>. Where Annex ‘F’ of the C99 standard recommends signaling ‘overflow’, Python will raise <a href="../library/exceptions.html#exceptions.OverflowError" class="reference internal" title="exceptions.OverflowError"><span class="pre"><code class="sourceCode python"><span class="pp">OverflowError</span></code></span></a>. (See <a href="https://bugs.python.org/issue711019" class="reference external">bpo-711019</a> and <a href="https://bugs.python.org/issue1640" class="reference external">bpo-1640</a>.)

  (Contributed by Christian Heimes and Mark Dickinson.)

- <a href="../library/mmap.html#mmap.mmap" class="reference internal" title="mmap.mmap"><span class="pre"><code class="sourceCode python">mmap</code></span></a> objects now have a <span class="pre">`rfind()`</span> method that searches for a substring beginning at the end of the string and searching backwards. The <span class="pre">`find()`</span> method also gained an *end* parameter giving an index at which to stop searching. (Contributed by John Lenton.)

- The <a href="../library/operator.html#module-operator" class="reference internal" title="operator: Functions corresponding to the standard operators."><span class="pre"><code class="sourceCode python">operator</code></span></a> module gained a <span class="pre">`methodcaller()`</span> function that takes a name and an optional set of arguments, returning a callable that will call the named function on any arguments passed to it. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> # Equivalent to lambda s: s.replace('old', 'new')
      >>> replacer = operator.methodcaller('replace', 'old', 'new')
      >>> replacer('old wine in old bottles')
      'new wine in new bottles'

  </div>

  </div>

  (Contributed by Georg Brandl, after a suggestion by Gregory Petrosyan.)

  The <span class="pre">`attrgetter()`</span> function now accepts dotted names and performs the corresponding attribute lookups:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> inst_name = operator.attrgetter(
      ...        '__class__.__name__')
      >>> inst_name('')
      'str'
      >>> inst_name(help)
      '_Helper'

  </div>

  </div>

  (Contributed by Georg Brandl, after a suggestion by Barry Warsaw.)

- The <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module now wraps several new system calls. <span class="pre">`fchmod(fd,`</span>` `<span class="pre">`mode)`</span> and <span class="pre">`fchown(fd,`</span>` `<span class="pre">`uid,`</span>` `<span class="pre">`gid)`</span> change the mode and ownership of an opened file, and <span class="pre">`lchmod(path,`</span>` `<span class="pre">`mode)`</span> changes the mode of a symlink. (Contributed by Georg Brandl and Christian Heimes.)

  <span class="pre">`chflags()`</span> and <span class="pre">`lchflags()`</span> are wrappers for the corresponding system calls (where they’re available), changing the flags set on a file. Constants for the flag values are defined in the <a href="../library/stat.html#module-stat" class="reference internal" title="stat: Utilities for interpreting the results of os.stat(), os.lstat() and os.fstat()."><span class="pre"><code class="sourceCode python">stat</code></span></a> module; some possible values include <span class="pre">`UF_IMMUTABLE`</span> to signal the file may not be changed and <span class="pre">`UF_APPEND`</span> to indicate that data can only be appended to the file. (Contributed by M. Levinson.)

  <span class="pre">`os.closerange(low,`</span>` `<span class="pre">`high)`</span> efficiently closes all file descriptors from *low* to *high*, ignoring any errors and not including *high* itself. This function is now used by the <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module to make starting processes faster. (Contributed by Georg Brandl; <a href="https://bugs.python.org/issue1663329" class="reference external">bpo-1663329</a>.)

- The <span class="pre">`os.environ`</span> object’s <span class="pre">`clear()`</span> method will now unset the environment variables using <a href="../library/os.html#os.unsetenv" class="reference internal" title="os.unsetenv"><span class="pre"><code class="sourceCode python">os.unsetenv()</code></span></a> in addition to clearing the object’s keys. (Contributed by Martin Horcicka; <a href="https://bugs.python.org/issue1181" class="reference external">bpo-1181</a>.)

- The <a href="../library/os.html#os.walk" class="reference internal" title="os.walk"><span class="pre"><code class="sourceCode python">os.walk()</code></span></a> function now has a <span class="pre">`followlinks`</span> parameter. If set to True, it will follow symlinks pointing to directories and visit the directory’s contents. For backward compatibility, the parameter’s default value is false. Note that the function can fall into an infinite recursion if there’s a symlink that points to a parent directory. (<a href="https://bugs.python.org/issue1273829" class="reference external">bpo-1273829</a>)

- In the <a href="../library/os.path.html#module-os.path" class="reference internal" title="os.path: Operations on pathnames."><span class="pre"><code class="sourceCode python">os.path</code></span></a> module, the <span class="pre">`splitext()`</span> function has been changed to not split on leading period characters. This produces better results when operating on Unix’s dot-files. For example, <span class="pre">`os.path.splitext('.ipython')`</span> now returns <span class="pre">`('.ipython',`</span>` `<span class="pre">`'')`</span> instead of <span class="pre">`('',`</span>` `<span class="pre">`'.ipython')`</span>. (<a href="https://bugs.python.org/issue1115886" class="reference external">bpo-1115886</a>)

  A new function, <span class="pre">`os.path.relpath(path,`</span>` `<span class="pre">`start='.')`</span>, returns a relative path from the <span class="pre">`start`</span> path, if it’s supplied, or from the current working directory to the destination <span class="pre">`path`</span>. (Contributed by Richard Barran; <a href="https://bugs.python.org/issue1339796" class="reference external">bpo-1339796</a>.)

  On Windows, <a href="../library/os.path.html#os.path.expandvars" class="reference internal" title="os.path.expandvars"><span class="pre"><code class="sourceCode python">os.path.expandvars()</code></span></a> will now expand environment variables given in the form “%var%”, and “~user” will be expanded into the user’s home directory path. (Contributed by Josiah Carlson; <a href="https://bugs.python.org/issue957650" class="reference external">bpo-957650</a>.)

- The Python debugger provided by the <a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a> module gained a new command: “run” restarts the Python program being debugged and can optionally take new command-line arguments for the program. (Contributed by Rocky Bernstein; <a href="https://bugs.python.org/issue1393667" class="reference external">bpo-1393667</a>.)

- The <a href="../library/pdb.html#pdb.post_mortem" class="reference internal" title="pdb.post_mortem"><span class="pre"><code class="sourceCode python">pdb.post_mortem()</code></span></a> function, used to begin debugging a traceback, will now use the traceback returned by <a href="../library/sys.html#sys.exc_info" class="reference internal" title="sys.exc_info"><span class="pre"><code class="sourceCode python">sys.exc_info()</code></span></a> if no traceback is supplied. (Contributed by Facundo Batista; <a href="https://bugs.python.org/issue1106316" class="reference external">bpo-1106316</a>.)

- The <a href="../library/pickletools.html#module-pickletools" class="reference internal" title="pickletools: Contains extensive comments about the pickle protocols and pickle-machine opcodes, as well as some useful functions."><span class="pre"><code class="sourceCode python">pickletools</code></span></a> module now has an <span class="pre">`optimize()`</span> function that takes a string containing a pickle and removes some unused opcodes, returning a shorter pickle that contains the same data structure. (Contributed by Raymond Hettinger.)

- A <span class="pre">`get_data()`</span> function was added to the <a href="../library/pkgutil.html#module-pkgutil" class="reference internal" title="pkgutil: Utilities for the import system."><span class="pre"><code class="sourceCode python">pkgutil</code></span></a> module that returns the contents of resource files included with an installed Python package. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import pkgutil
      >>> print pkgutil.get_data('test', 'exception_hierarchy.txt')
      BaseException
       +-- SystemExit
       +-- KeyboardInterrupt
       +-- GeneratorExit
       +-- Exception
            +-- StopIteration
            +-- StandardError
       ...

  </div>

  </div>

  (Contributed by Paul Moore; <a href="https://bugs.python.org/issue2439" class="reference external">bpo-2439</a>.)

- The <span class="pre">`pyexpat`</span> module’s <span class="pre">`Parser`</span> objects now allow setting their <span class="pre">`buffer_size`</span> attribute to change the size of the buffer used to hold character data. (Contributed by Achim Gaedke; <a href="https://bugs.python.org/issue1137" class="reference external">bpo-1137</a>.)

- The <a href="../library/queue.html#module-Queue" class="reference internal" title="Queue: A synchronized queue class."><span class="pre"><code class="sourceCode python">Queue</code></span></a> module now provides queue variants that retrieve entries in different orders. The <span class="pre">`PriorityQueue`</span> class stores queued items in a heap and retrieves them in priority order, and <span class="pre">`LifoQueue`</span> retrieves the most recently added entries first, meaning that it behaves like a stack. (Contributed by Raymond Hettinger.)

- The <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module’s <span class="pre">`Random`</span> objects can now be pickled on a 32-bit system and unpickled on a 64-bit system, and vice versa. Unfortunately, this change also means that Python 2.6’s <span class="pre">`Random`</span> objects can’t be unpickled correctly on earlier versions of Python. (Contributed by Shawn Ligocki; <a href="https://bugs.python.org/issue1727780" class="reference external">bpo-1727780</a>.)

  The new <span class="pre">`triangular(low,`</span>` `<span class="pre">`high,`</span>` `<span class="pre">`mode)`</span> function returns random numbers following a triangular distribution. The returned values are between *low* and *high*, not including *high* itself, and with *mode* as the most frequently occurring value in the distribution. (Contributed by Wladmir van der Laan and Raymond Hettinger; <a href="https://bugs.python.org/issue1681432" class="reference external">bpo-1681432</a>.)

- Long regular expression searches carried out by the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module will check for signals being delivered, so time-consuming searches can now be interrupted. (Contributed by Josh Hoyt and Ralf Schmitt; <a href="https://bugs.python.org/issue846388" class="reference external">bpo-846388</a>.)

  The regular expression module is implemented by compiling bytecodes for a tiny regex-specific virtual machine. Untrusted code could create malicious strings of bytecode directly and cause crashes, so Python 2.6 includes a verifier for the regex bytecode. (Contributed by Guido van Rossum from work for Google App Engine; <a href="https://bugs.python.org/issue3487" class="reference external">bpo-3487</a>.)

- The <a href="../library/rlcompleter.html#module-rlcompleter" class="reference internal" title="rlcompleter: Python identifier completion, suitable for the GNU readline library."><span class="pre"><code class="sourceCode python">rlcompleter</code></span></a> module’s <span class="pre">`Completer.complete()`</span> method will now ignore exceptions triggered while evaluating a name. (Fixed by Lorenz Quack; <a href="https://bugs.python.org/issue2250" class="reference external">bpo-2250</a>.)

- The <a href="../library/sched.html#module-sched" class="reference internal" title="sched: General purpose event scheduler."><span class="pre"><code class="sourceCode python">sched</code></span></a> module’s <span class="pre">`scheduler`</span> instances now have a read-only <span class="pre">`queue`</span> attribute that returns the contents of the scheduler’s queue, represented as a list of named tuples with the fields <span class="pre">`(time,`</span>` `<span class="pre">`priority,`</span>` `<span class="pre">`action,`</span>` `<span class="pre">`argument)`</span>. (Contributed by Raymond Hettinger; <a href="https://bugs.python.org/issue1861" class="reference external">bpo-1861</a>.)

- The <a href="../library/select.html#module-select" class="reference internal" title="select: Wait for I/O completion on multiple streams."><span class="pre"><code class="sourceCode python">select</code></span></a> module now has wrapper functions for the Linux <span class="pre">`epoll()`</span> and BSD <span class="pre">`kqueue()`</span> system calls. <span class="pre">`modify()`</span> method was added to the existing <span class="pre">`poll`</span> objects; <span class="pre">`pollobj.modify(fd,`</span>` `<span class="pre">`eventmask)`</span> takes a file descriptor or file object and an event mask, modifying the recorded event mask for that file. (Contributed by Christian Heimes; <a href="https://bugs.python.org/issue1657" class="reference external">bpo-1657</a>.)

- The <a href="../library/shutil.html#shutil.copytree" class="reference internal" title="shutil.copytree"><span class="pre"><code class="sourceCode python">shutil.copytree()</code></span></a> function now has an optional *ignore* argument that takes a callable object. This callable will receive each directory path and a list of the directory’s contents, and returns a list of names that will be ignored, not copied.

  The <a href="../library/shutil.html#module-shutil" class="reference internal" title="shutil: High-level file operations, including copying."><span class="pre"><code class="sourceCode python">shutil</code></span></a> module also provides an <span class="pre">`ignore_patterns()`</span> function for use with this new parameter. <span class="pre">`ignore_patterns()`</span> takes an arbitrary number of glob-style patterns and returns a callable that will ignore any files and directories that match any of these patterns. The following example copies a directory tree, but skips both <span class="pre">`.svn`</span> directories and Emacs backup files, which have names ending with ‘~’:

  <div class="highlight-default notranslate">

  <div class="highlight">

      shutil.copytree('Doc/library', '/tmp/library',
                      ignore=shutil.ignore_patterns('*~', '.svn'))

  </div>

  </div>

  (Contributed by Tarek Ziadé; <a href="https://bugs.python.org/issue2663" class="reference external">bpo-2663</a>.)

- Integrating signal handling with GUI handling event loops like those used by Tkinter or GTk+ has long been a problem; most software ends up polling, waking up every fraction of a second to check if any GUI events have occurred. The <a href="../library/signal.html#module-signal" class="reference internal" title="signal: Set handlers for asynchronous events."><span class="pre"><code class="sourceCode python">signal</code></span></a> module can now make this more efficient. Calling <span class="pre">`signal.set_wakeup_fd(fd)`</span> sets a file descriptor to be used; when a signal is received, a byte is written to that file descriptor. There’s also a C-level function, <a href="../c-api/exceptions.html#c.PySignal_SetWakeupFd" class="reference internal" title="PySignal_SetWakeupFd"><span class="pre"><code class="sourceCode c">PySignal_SetWakeupFd<span class="op">()</span></code></span></a>, for setting the descriptor.

  Event loops will use this by opening a pipe to create two descriptors, one for reading and one for writing. The writable descriptor will be passed to <span class="pre">`set_wakeup_fd()`</span>, and the readable descriptor will be added to the list of descriptors monitored by the event loop via <span class="pre">`select()`</span> or <span class="pre">`poll()`</span>. On receiving a signal, a byte will be written and the main event loop will be woken up, avoiding the need to poll.

  (Contributed by Adam Olsen; <a href="https://bugs.python.org/issue1583" class="reference external">bpo-1583</a>.)

  The <span class="pre">`siginterrupt()`</span> function is now available from Python code, and allows changing whether signals can interrupt system calls or not. (Contributed by Ralf Schmitt.)

  The <span class="pre">`setitimer()`</span> and <span class="pre">`getitimer()`</span> functions have also been added (where they’re available). <span class="pre">`setitimer()`</span> allows setting interval timers that will cause a signal to be delivered to the process after a specified time, measured in wall-clock time, consumed process time, or combined process+system time. (Contributed by Guilherme Polo; <a href="https://bugs.python.org/issue2240" class="reference external">bpo-2240</a>.)

- The <a href="../library/smtplib.html#module-smtplib" class="reference internal" title="smtplib: SMTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">smtplib</code></span></a> module now supports SMTP over SSL thanks to the addition of the <span class="pre">`SMTP_SSL`</span> class. This class supports an interface identical to the existing <span class="pre">`SMTP`</span> class. (Contributed by Monty Taylor.) Both class constructors also have an optional <span class="pre">`timeout`</span> parameter that specifies a timeout for the initial connection attempt, measured in seconds. (Contributed by Facundo Batista.)

  An implementation of the LMTP protocol (<span id="index-22" class="target"></span><a href="https://tools.ietf.org/html/rfc2033.html" class="rfc reference external"><strong>RFC 2033</strong></a>) was also added to the module. LMTP is used in place of SMTP when transferring e-mail between agents that don’t manage a mail queue. (LMTP implemented by Leif Hedstrom; <a href="https://bugs.python.org/issue957003" class="reference external">bpo-957003</a>.)

  <span class="pre">`SMTP.starttls()`</span> now complies with <span id="index-23" class="target"></span><a href="https://tools.ietf.org/html/rfc3207.html" class="rfc reference external"><strong>RFC 3207</strong></a> and forgets any knowledge obtained from the server not obtained from the TLS negotiation itself. (Patch contributed by Bill Fenner; <a href="https://bugs.python.org/issue829951" class="reference external">bpo-829951</a>.)

- The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module now supports TIPC (<a href="http://tipc.sourceforge.net/" class="reference external">http://tipc.sourceforge.net/</a>), a high-performance non-IP-based protocol designed for use in clustered environments. TIPC addresses are 4- or 5-tuples. (Contributed by Alberto Bertogli; <a href="https://bugs.python.org/issue1646" class="reference external">bpo-1646</a>.)

  A new function, <span class="pre">`create_connection()`</span>, takes an address and connects to it using an optional timeout value, returning the connected socket object. This function also looks up the address’s type and connects to it using IPv4 or IPv6 as appropriate. Changing your code to use <span class="pre">`create_connection()`</span> instead of <span class="pre">`socket(socket.AF_INET,`</span>` `<span class="pre">`...)`</span> may be all that’s required to make your code work with IPv6.

- The base classes in the <a href="../library/socketserver.html#module-SocketServer" class="reference internal" title="SocketServer: A framework for network servers."><span class="pre"><code class="sourceCode python">SocketServer</code></span></a> module now support calling a <span class="pre">`handle_timeout()`</span> method after a span of inactivity specified by the server’s <span class="pre">`timeout`</span> attribute. (Contributed by Michael Pomraning.) The <span class="pre">`serve_forever()`</span> method now takes an optional poll interval measured in seconds, controlling how often the server will check for a shutdown request. (Contributed by Pedro Werneck and Jeffrey Yasskin; <a href="https://bugs.python.org/issue742598" class="reference external">bpo-742598</a>, <a href="https://bugs.python.org/issue1193577" class="reference external">bpo-1193577</a>.)

- The <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> module, maintained by Gerhard Haering, has been updated from version 2.3.2 in Python 2.5 to version 2.4.1.

- The <a href="../library/struct.html#module-struct" class="reference internal" title="struct: Interpret strings as packed binary data."><span class="pre"><code class="sourceCode python">struct</code></span></a> module now supports the C99 <span class="pre">`_Bool`</span> type, using the format character <span class="pre">`'?'`</span>. (Contributed by David Remahl.)

- The <span class="pre">`Popen`</span> objects provided by the <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module now have <span class="pre">`terminate()`</span>, <span class="pre">`kill()`</span>, and <span class="pre">`send_signal()`</span> methods. On Windows, <span class="pre">`send_signal()`</span> only supports the <span class="pre">`SIGTERM`</span> signal, and all these methods are aliases for the Win32 API function <span class="pre">`TerminateProcess()`</span>. (Contributed by Christian Heimes.)

- A new variable in the <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> module, <span class="pre">`float_info`</span>, is an object containing information derived from the <span class="pre">`float.h`</span> file about the platform’s floating-point support. Attributes of this object include <span class="pre">`mant_dig`</span> (number of digits in the mantissa), <span class="pre">`epsilon`</span> (smallest difference between 1.0 and the next largest value representable), and several others. (Contributed by Christian Heimes; <a href="https://bugs.python.org/issue1534" class="reference external">bpo-1534</a>.)

  Another new variable, <span class="pre">`dont_write_bytecode`</span>, controls whether Python writes any <span class="pre">`.pyc`</span> or <span class="pre">`.pyo`</span> files on importing a module. If this variable is true, the compiled files are not written. The variable is initially set on start-up by supplying the <a href="../using/cmdline.html#id1" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-B</code></span></a> switch to the Python interpreter, or by setting the <span id="index-24" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONDONTWRITEBYTECODE" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONDONTWRITEBYTECODE</code></span></a> environment variable before running the interpreter. Python code can subsequently change the value of this variable to control whether bytecode files are written or not. (Contributed by Neal Norwitz and Georg Brandl.)

  Information about the command-line arguments supplied to the Python interpreter is available by reading attributes of a named tuple available as <span class="pre">`sys.flags`</span>. For example, the <span class="pre">`verbose`</span> attribute is true if Python was executed in verbose mode, <span class="pre">`debug`</span> is true in debugging mode, etc. These attributes are all read-only. (Contributed by Christian Heimes.)

  A new function, <span class="pre">`getsizeof()`</span>, takes a Python object and returns the amount of memory used by the object, measured in bytes. Built-in objects return correct results; third-party extensions may not, but can define a <span class="pre">`__sizeof__()`</span> method to return the object’s size. (Contributed by Robert Schuppenies; <a href="https://bugs.python.org/issue2898" class="reference external">bpo-2898</a>.)

  It’s now possible to determine the current profiler and tracer functions by calling <a href="../library/sys.html#sys.getprofile" class="reference internal" title="sys.getprofile"><span class="pre"><code class="sourceCode python">sys.getprofile()</code></span></a> and <a href="../library/sys.html#sys.gettrace" class="reference internal" title="sys.gettrace"><span class="pre"><code class="sourceCode python">sys.gettrace()</code></span></a>. (Contributed by Georg Brandl; <a href="https://bugs.python.org/issue1648" class="reference external">bpo-1648</a>.)

- The <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module now supports POSIX.1-2001 (pax) tarfiles in addition to the POSIX.1-1988 (ustar) and GNU tar formats that were already supported. The default format is GNU tar; specify the <span class="pre">`format`</span> parameter to open a file using a different format:

  <div class="highlight-default notranslate">

  <div class="highlight">

      tar = tarfile.open("output.tar", "w",
                         format=tarfile.PAX_FORMAT)

  </div>

  </div>

  The new <span class="pre">`encoding`</span> and <span class="pre">`errors`</span> parameters specify an encoding and an error handling scheme for character conversions. <span class="pre">`'strict'`</span>, <span class="pre">`'ignore'`</span>, and <span class="pre">`'replace'`</span> are the three standard ways Python can handle errors,; <span class="pre">`'utf-8'`</span> is a special value that replaces bad characters with their UTF-8 representation. (Character conversions occur because the PAX format supports Unicode filenames, defaulting to UTF-8 encoding.)

  The <span class="pre">`TarFile.add()`</span> method now accepts an <span class="pre">`exclude`</span> argument that’s a function that can be used to exclude certain filenames from an archive. The function must take a filename and return true if the file should be excluded or false if it should be archived. The function is applied to both the name initially passed to <span class="pre">`add()`</span> and to the names of files in recursively-added directories.

  (All changes contributed by Lars Gustäbel).

- An optional <span class="pre">`timeout`</span> parameter was added to the <a href="../library/telnetlib.html#telnetlib.Telnet" class="reference internal" title="telnetlib.Telnet"><span class="pre"><code class="sourceCode python">telnetlib.Telnet</code></span></a> class constructor, specifying a timeout measured in seconds. (Added by Facundo Batista.)

- The <a href="../library/tempfile.html#tempfile.NamedTemporaryFile" class="reference internal" title="tempfile.NamedTemporaryFile"><span class="pre"><code class="sourceCode python">tempfile.NamedTemporaryFile</code></span></a> class usually deletes the temporary file it created when the file is closed. This behaviour can now be changed by passing <span class="pre">`delete=False`</span> to the constructor. (Contributed by Damien Miller; <a href="https://bugs.python.org/issue1537850" class="reference external">bpo-1537850</a>.)

  A new class, <span class="pre">`SpooledTemporaryFile`</span>, behaves like a temporary file but stores its data in memory until a maximum size is exceeded. On reaching that limit, the contents will be written to an on-disk temporary file. (Contributed by Dustin J. Mitchell.)

  The <span class="pre">`NamedTemporaryFile`</span> and <span class="pre">`SpooledTemporaryFile`</span> classes both work as context managers, so you can write <span class="pre">`with`</span>` `<span class="pre">`tempfile.NamedTemporaryFile()`</span>` `<span class="pre">`as`</span>` `<span class="pre">`tmp:`</span>` `<span class="pre">`...`</span>. (Contributed by Alexander Belopolsky; <a href="https://bugs.python.org/issue2021" class="reference external">bpo-2021</a>.)

- The <span class="pre">`test.test_support`</span> module gained a number of context managers useful for writing tests. <span class="pre">`EnvironmentVarGuard()`</span> is a context manager that temporarily changes environment variables and automatically restores them to their old values.

  Another context manager, <span class="pre">`TransientResource`</span>, can surround calls to resources that may or may not be available; it will catch and ignore a specified list of exceptions. For example, a network test may ignore certain failures when connecting to an external web site:

  <div class="highlight-default notranslate">

  <div class="highlight">

      with test_support.TransientResource(IOError,
                                      errno=errno.ETIMEDOUT):
          f = urllib.urlopen('https://sf.net')
          ...

  </div>

  </div>

  Finally, <span class="pre">`check_warnings()`</span> resets the <span class="pre">`warning`</span> module’s warning filters and returns an object that will record all warning messages triggered (<a href="https://bugs.python.org/issue3781" class="reference external">bpo-3781</a>):

  <div class="highlight-default notranslate">

  <div class="highlight">

      with test_support.check_warnings() as wrec:
          warnings.simplefilter("always")
          # ... code that triggers a warning ...
          assert str(wrec.message) == "function is outdated"
          assert len(wrec.warnings) == 1, "Multiple warnings raised"

  </div>

  </div>

  (Contributed by Brett Cannon.)

- The <a href="../library/textwrap.html#module-textwrap" class="reference internal" title="textwrap: Text wrapping and filling"><span class="pre"><code class="sourceCode python">textwrap</code></span></a> module can now preserve existing whitespace at the beginnings and ends of the newly-created lines by specifying <span class="pre">`drop_whitespace=False`</span> as an argument:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> S = """This  sentence  has a bunch   of
      ...   extra   whitespace."""
      >>> print textwrap.fill(S, width=15)
      This  sentence
      has a bunch
      of    extra
      whitespace.
      >>> print textwrap.fill(S, drop_whitespace=False, width=15)
      This  sentence
        has a bunch
         of    extra
         whitespace.
      >>>

  </div>

  </div>

  (Contributed by Dwayne Bailey; <a href="https://bugs.python.org/issue1581073" class="reference external">bpo-1581073</a>.)

- The <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> module API is being changed to use properties such as <span class="pre">`daemon`</span> instead of <span class="pre">`setDaemon()`</span> and <span class="pre">`isDaemon()`</span> methods, and some methods have been renamed to use underscores instead of camel-case; for example, the <span class="pre">`activeCount()`</span> method is renamed to <span class="pre">`active_count()`</span>. Both the 2.6 and 3.0 versions of the module support the same properties and renamed methods, but don’t remove the old methods. No date has been set for the deprecation of the old APIs in Python 3.x; the old APIs won’t be removed in any 2.x version. (Carried out by several people, most notably Benjamin Peterson.)

  The <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> module’s <span class="pre">`Thread`</span> objects gained an <span class="pre">`ident`</span> property that returns the thread’s identifier, a nonzero integer. (Contributed by Gregory P. Smith; <a href="https://bugs.python.org/issue2871" class="reference external">bpo-2871</a>.)

- The <a href="../library/timeit.html#module-timeit" class="reference internal" title="timeit: Measure the execution time of small code snippets."><span class="pre"><code class="sourceCode python">timeit</code></span></a> module now accepts callables as well as strings for the statement being timed and for the setup code. Two convenience functions were added for creating <span class="pre">`Timer`</span> instances: <span class="pre">`repeat(stmt,`</span>` `<span class="pre">`setup,`</span>` `<span class="pre">`time,`</span>` `<span class="pre">`repeat,`</span>` `<span class="pre">`number)`</span> and <span class="pre">`timeit(stmt,`</span>` `<span class="pre">`setup,`</span>` `<span class="pre">`time,`</span>` `<span class="pre">`number)`</span> create an instance and call the corresponding method. (Contributed by Erik Demaine; <a href="https://bugs.python.org/issue1533909" class="reference external">bpo-1533909</a>.)

- The <a href="../library/tkinter.html#module-Tkinter" class="reference internal" title="Tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">Tkinter</code></span></a> module now accepts lists and tuples for options, separating the elements by spaces before passing the resulting value to Tcl/Tk. (Contributed by Guilherme Polo; <a href="https://bugs.python.org/issue2906" class="reference external">bpo-2906</a>.)

- The <a href="../library/turtle.html#module-turtle" class="reference internal" title="turtle: Turtle graphics for Tk"><span class="pre"><code class="sourceCode python">turtle</code></span></a> module for turtle graphics was greatly enhanced by Gregor Lingl. New features in the module include:

  - Better animation of turtle movement and rotation.

  - Control over turtle movement using the new <span class="pre">`delay()`</span>, <span class="pre">`tracer()`</span>, and <span class="pre">`speed()`</span> methods.

  - The ability to set new shapes for the turtle, and to define a new coordinate system.

  - Turtles now have an <span class="pre">`undo()`</span> method that can roll back actions.

  - Simple support for reacting to input events such as mouse and keyboard activity, making it possible to write simple games.

  - A <span class="pre">`turtle.cfg`</span> file can be used to customize the starting appearance of the turtle’s screen.

  - The module’s docstrings can be replaced by new docstrings that have been translated into another language.

  (<a href="https://bugs.python.org/issue1513695" class="reference external">bpo-1513695</a>)

- An optional <span class="pre">`timeout`</span> parameter was added to the <a href="../library/urllib.html#urllib.urlopen" class="reference internal" title="urllib.urlopen"><span class="pre"><code class="sourceCode python">urllib.urlopen()</code></span></a> function and the <span class="pre">`urllib.ftpwrapper`</span> class constructor, as well as the <a href="../library/urllib2.html#urllib2.urlopen" class="reference internal" title="urllib2.urlopen"><span class="pre"><code class="sourceCode python">urllib2.urlopen()</code></span></a> function. The parameter specifies a timeout measured in seconds. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> u = urllib2.urlopen("http://slow.example.com",
                              timeout=3)
      Traceback (most recent call last):
        ...
      urllib2.URLError: <urlopen error timed out>
      >>>

  </div>

  </div>

  (Added by Facundo Batista.)

- The Unicode database provided by the <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a> module has been updated to version 5.1.0. (Updated by Martin von Loewis; <a href="https://bugs.python.org/issue3811" class="reference external">bpo-3811</a>.)

- The <a href="../library/warnings.html#module-warnings" class="reference internal" title="warnings: Issue warning messages and control their disposition."><span class="pre"><code class="sourceCode python">warnings</code></span></a> module’s <span class="pre">`formatwarning()`</span> and <span class="pre">`showwarning()`</span> gained an optional *line* argument that can be used to supply the line of source code. (Added as part of <a href="https://bugs.python.org/issue1631171" class="reference external">bpo-1631171</a>, which re-implemented part of the <a href="../library/warnings.html#module-warnings" class="reference internal" title="warnings: Issue warning messages and control their disposition."><span class="pre"><code class="sourceCode python">warnings</code></span></a> module in C code.)

  A new function, <span class="pre">`catch_warnings()`</span>, is a context manager intended for testing purposes that lets you temporarily modify the warning filters and then restore their original values (<a href="https://bugs.python.org/issue3781" class="reference external">bpo-3781</a>).

- The XML-RPC <a href="../library/simplexmlrpcserver.html#SimpleXMLRPCServer.SimpleXMLRPCServer" class="reference internal" title="SimpleXMLRPCServer.SimpleXMLRPCServer"><span class="pre"><code class="sourceCode python">SimpleXMLRPCServer</code></span></a> and <a href="../library/docxmlrpcserver.html#DocXMLRPCServer.DocXMLRPCServer" class="reference internal" title="DocXMLRPCServer.DocXMLRPCServer"><span class="pre"><code class="sourceCode python">DocXMLRPCServer</code></span></a> classes can now be prevented from immediately opening and binding to their socket by passing <span class="pre">`False`</span> as the *bind_and_activate* constructor parameter. This can be used to modify the instance’s <span class="pre">`allow_reuse_address`</span> attribute before calling the <span class="pre">`server_bind()`</span> and <span class="pre">`server_activate()`</span> methods to open the socket and begin listening for connections. (Contributed by Peter Parente; <a href="https://bugs.python.org/issue1599845" class="reference external">bpo-1599845</a>.)

  <a href="../library/simplexmlrpcserver.html#SimpleXMLRPCServer.SimpleXMLRPCServer" class="reference internal" title="SimpleXMLRPCServer.SimpleXMLRPCServer"><span class="pre"><code class="sourceCode python">SimpleXMLRPCServer</code></span></a> also has a <span class="pre">`_send_traceback_header`</span> attribute; if true, the exception and formatted traceback are returned as HTTP headers “X-Exception” and “X-Traceback”. This feature is for debugging purposes only and should not be used on production servers because the tracebacks might reveal passwords or other sensitive information. (Contributed by Alan McIntyre as part of his project for Google’s Summer of Code 2007.)

- The <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a> module no longer automatically converts <a href="../library/datetime.html#datetime.date" class="reference internal" title="datetime.date"><span class="pre"><code class="sourceCode python">datetime.date</code></span></a> and <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">datetime.time</code></span></a> to the <a href="../library/xmlrpclib.html#xmlrpclib.DateTime" class="reference internal" title="xmlrpclib.DateTime"><span class="pre"><code class="sourceCode python">xmlrpclib.DateTime</code></span></a> type; the conversion semantics were not necessarily correct for all applications. Code using <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a> should convert <span class="pre">`date`</span> and <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">time</code></span></a> instances. (<a href="https://bugs.python.org/issue1330538" class="reference external">bpo-1330538</a>) The code can also handle dates before 1900 (contributed by Ralf Schmitt; <a href="https://bugs.python.org/issue2014" class="reference external">bpo-2014</a>) and 64-bit integers represented by using <span class="pre">`<i8>`</span> in XML-RPC responses (contributed by Riku Lindblad; <a href="https://bugs.python.org/issue2985" class="reference external">bpo-2985</a>).

- The <a href="../library/zipfile.html#module-zipfile" class="reference internal" title="zipfile: Read and write ZIP-format archive files."><span class="pre"><code class="sourceCode python">zipfile</code></span></a> module’s <span class="pre">`ZipFile`</span> class now has <span class="pre">`extract()`</span> and <span class="pre">`extractall()`</span> methods that will unpack a single file or all the files in the archive to the current directory, or to a specified directory:

  <div class="highlight-default notranslate">

  <div class="highlight">

      z = zipfile.ZipFile('python-251.zip')

      # Unpack a single file, writing it relative
      # to the /tmp directory.
      z.extract('Python/sysmodule.c', '/tmp')

      # Unpack all the files in the archive.
      z.extractall()

  </div>

  </div>

  (Contributed by Alan McIntyre; <a href="https://bugs.python.org/issue467924" class="reference external">bpo-467924</a>.)

  The <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a>, <span class="pre">`read()`</span> and <span class="pre">`extract()`</span> methods can now take either a filename or a <span class="pre">`ZipInfo`</span> object. This is useful when an archive accidentally contains a duplicated filename. (Contributed by Graham Horler; <a href="https://bugs.python.org/issue1775025" class="reference external">bpo-1775025</a>.)

  Finally, <a href="../library/zipfile.html#module-zipfile" class="reference internal" title="zipfile: Read and write ZIP-format archive files."><span class="pre"><code class="sourceCode python">zipfile</code></span></a> now supports using Unicode filenames for archived files. (Contributed by Alexey Borzenkov; <a href="https://bugs.python.org/issue1734346" class="reference external">bpo-1734346</a>.)

<div id="the-ast-module" class="section">

### The <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> module<a href="#the-ast-module" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> module provides an Abstract Syntax Tree representation of Python code, and Armin Ronacher contributed a set of helper functions that perform a variety of common tasks. These will be useful for HTML templating packages, code analyzers, and similar tools that process Python code.

The <span class="pre">`parse()`</span> function takes an expression and returns an AST. The <span class="pre">`dump()`</span> function outputs a representation of a tree, suitable for debugging:

<div class="highlight-default notranslate">

<div class="highlight">

    import ast

    t = ast.parse("""
    d = {}
    for i in 'abcdefghijklm':
        d[i + i] = ord(i) - ord('a') + 1
    print d
    """)
    print ast.dump(t)

</div>

</div>

This outputs a deeply nested tree:

<div class="highlight-default notranslate">

<div class="highlight">

    Module(body=[
      Assign(targets=[
        Name(id='d', ctx=Store())
       ], value=Dict(keys=[], values=[]))
      For(target=Name(id='i', ctx=Store()),
          iter=Str(s='abcdefghijklm'), body=[
        Assign(targets=[
          Subscript(value=
            Name(id='d', ctx=Load()),
              slice=
              Index(value=
                BinOp(left=Name(id='i', ctx=Load()), op=Add(),
                 right=Name(id='i', ctx=Load()))), ctx=Store())
         ], value=
         BinOp(left=
          BinOp(left=
           Call(func=
            Name(id='ord', ctx=Load()), args=[
              Name(id='i', ctx=Load())
             ], keywords=[], starargs=None, kwargs=None),
           op=Sub(), right=Call(func=
            Name(id='ord', ctx=Load()), args=[
              Str(s='a')
             ], keywords=[], starargs=None, kwargs=None)),
           op=Add(), right=Num(n=1)))
        ], orelse=[])
       Print(dest=None, values=[
         Name(id='d', ctx=Load())
       ], nl=True)
     ])

</div>

</div>

The <span class="pre">`literal_eval()`</span> method takes a string or an AST representing a literal expression, parses and evaluates it, and returns the resulting value. A literal expression is a Python expression containing only strings, numbers, dictionaries, etc. but no statements or function calls. If you need to evaluate an expression but cannot accept the security risk of using an <a href="../library/functions.html#eval" class="reference internal" title="eval"><span class="pre"><code class="sourceCode python"><span class="bu">eval</span>()</code></span></a> call, <span class="pre">`literal_eval()`</span> will handle it safely:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> literal = '("a", "b", {2:4, 3:8, 1:2})'
    >>> print ast.literal_eval(literal)
    ('a', 'b', {1: 2, 2: 4, 3: 8})
    >>> print ast.literal_eval('"a" + "b"')
    Traceback (most recent call last):
      ...
    ValueError: malformed string

</div>

</div>

The module also includes <span class="pre">`NodeVisitor`</span> and <span class="pre">`NodeTransformer`</span> classes for traversing and modifying an AST, and functions for common transformations such as changing line numbers.

</div>

<div id="the-future-builtins-module" class="section">

### The <a href="../library/future_builtins.html#module-future_builtins" class="reference internal" title="future_builtins"><span class="pre"><code class="sourceCode python">future_builtins</code></span></a> module<a href="#the-future-builtins-module" class="headerlink" title="Permalink to this headline">¶</a>

Python 3.0 makes many changes to the repertoire of built-in functions, and most of the changes can’t be introduced in the Python 2.x series because they would break compatibility. The <a href="../library/future_builtins.html#module-future_builtins" class="reference internal" title="future_builtins"><span class="pre"><code class="sourceCode python">future_builtins</code></span></a> module provides versions of these built-in functions that can be imported when writing 3.0-compatible code.

The functions in this module currently include:

- <span class="pre">`ascii(obj)`</span>: equivalent to <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a>. In Python 3.0, <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> will return a Unicode string, while <span class="pre">`ascii()`</span> will return a pure ASCII bytestring.

- <span class="pre">`filter(predicate,`</span>` `<span class="pre">`iterable)`</span>, <span class="pre">`map(func,`</span>` `<span class="pre">`iterable1,`</span>` `<span class="pre">`...)`</span>: the 3.0 versions return iterators, unlike the 2.x builtins which return lists.

- <span class="pre">`hex(value)`</span>, <span class="pre">`oct(value)`</span>: instead of calling the <a href="../reference/datamodel.html#object.__hex__" class="reference internal" title="object.__hex__"><span class="pre"><code class="sourceCode python"><span class="fu">__hex__</span>()</code></span></a> or <a href="../reference/datamodel.html#object.__oct__" class="reference internal" title="object.__oct__"><span class="pre"><code class="sourceCode python"><span class="fu">__oct__</span>()</code></span></a> methods, these versions will call the <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> method and convert the result to hexadecimal or octal. <a href="../library/functions.html#oct" class="reference internal" title="oct"><span class="pre"><code class="sourceCode python"><span class="bu">oct</span>()</code></span></a> will use the new <span class="pre">`0o`</span> notation for its result.

</div>

<div id="the-json-module-javascript-object-notation" class="section">

### The <a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> module: JavaScript Object Notation<a href="#the-json-module-javascript-object-notation" class="headerlink" title="Permalink to this headline">¶</a>

The new <a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> module supports the encoding and decoding of Python types in JSON (Javascript Object Notation). JSON is a lightweight interchange format often used in web applications. For more information about JSON, see <a href="http://www.json.org" class="reference external">http://www.json.org</a>.

<a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> comes with support for decoding and encoding most built-in Python types. The following example encodes and decodes a dictionary:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> import json
    >>> data = {"spam": "foo", "parrot": 42}
    >>> in_json = json.dumps(data) # Encode the data
    >>> in_json
    '{"parrot": 42, "spam": "foo"}'
    >>> json.loads(in_json) # Decode into a Python object
    {"spam": "foo", "parrot": 42}

</div>

</div>

It’s also possible to write your own decoders and encoders to support more types. Pretty-printing of the JSON strings is also supported.

<a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> (originally called simplejson) was written by Bob Ippolito.

</div>

<div id="the-plistlib-module-a-property-list-parser" class="section">

### The <a href="../library/plistlib.html#module-plistlib" class="reference internal" title="plistlib: Generate and parse Mac OS X plist files."><span class="pre"><code class="sourceCode python">plistlib</code></span></a> module: A Property-List Parser<a href="#the-plistlib-module-a-property-list-parser" class="headerlink" title="Permalink to this headline">¶</a>

The <span class="pre">`.plist`</span> format is commonly used on Mac OS X to store basic data types (numbers, strings, lists, and dictionaries) by serializing them into an XML-based format. It resembles the XML-RPC serialization of data types.

Despite being primarily used on Mac OS X, the format has nothing Mac-specific about it and the Python implementation works on any platform that Python supports, so the <a href="../library/plistlib.html#module-plistlib" class="reference internal" title="plistlib: Generate and parse Mac OS X plist files."><span class="pre"><code class="sourceCode python">plistlib</code></span></a> module has been promoted to the standard library.

Using the module is simple:

<div class="highlight-default notranslate">

<div class="highlight">

    import sys
    import plistlib
    import datetime

    # Create data structure
    data_struct = dict(lastAccessed=datetime.datetime.now(),
                       version=1,
                       categories=('Personal','Shared','Private'))

    # Create string containing XML.
    plist_str = plistlib.writePlistToString(data_struct)
    new_struct = plistlib.readPlistFromString(plist_str)
    print data_struct
    print new_struct

    # Write data structure to a file and read it back.
    plistlib.writePlist(data_struct, '/tmp/customizations.plist')
    new_struct = plistlib.readPlist('/tmp/customizations.plist')

    # read/writePlist accepts file-like objects as well as paths.
    plistlib.writePlist(data_struct, sys.stdout)

</div>

</div>

</div>

<div id="ctypes-enhancements" class="section">

### ctypes Enhancements<a href="#ctypes-enhancements" class="headerlink" title="Permalink to this headline">¶</a>

Thomas Heller continued to maintain and enhance the <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> module.

<a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> now supports a <span class="pre">`c_bool`</span> datatype that represents the C99 <span class="pre">`bool`</span> type. (Contributed by David Remahl; <a href="https://bugs.python.org/issue1649190" class="reference external">bpo-1649190</a>.)

The <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> string, buffer and array types have improved support for extended slicing syntax, where various combinations of <span class="pre">`(start,`</span>` `<span class="pre">`stop,`</span>` `<span class="pre">`step)`</span> are supplied. (Implemented by Thomas Wouters.)

All <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> data types now support <span class="pre">`from_buffer()`</span> and <span class="pre">`from_buffer_copy()`</span> methods that create a ctypes instance based on a provided buffer object. <span class="pre">`from_buffer_copy()`</span> copies the contents of the object, while <span class="pre">`from_buffer()`</span> will share the same memory area.

A new calling convention tells <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> to clear the <span class="pre">`errno`</span> or Win32 LastError variables at the outset of each wrapped call. (Implemented by Thomas Heller; <a href="https://bugs.python.org/issue1798" class="reference external">bpo-1798</a>.)

You can now retrieve the Unix <span class="pre">`errno`</span> variable after a function call. When creating a wrapped function, you can supply <span class="pre">`use_errno=True`</span> as a keyword parameter to the <span class="pre">`DLL()`</span> function and then call the module-level methods <span class="pre">`set_errno()`</span> and <span class="pre">`get_errno()`</span> to set and retrieve the error value.

The Win32 LastError variable is similarly supported by the <span class="pre">`DLL()`</span>, <span class="pre">`OleDLL()`</span>, and <span class="pre">`WinDLL()`</span> functions. You supply <span class="pre">`use_last_error=True`</span> as a keyword parameter and then call the module-level methods <span class="pre">`set_last_error()`</span> and <span class="pre">`get_last_error()`</span>.

The <span class="pre">`byref()`</span> function, used to retrieve a pointer to a ctypes instance, now has an optional *offset* parameter that is a byte count that will be added to the returned pointer.

</div>

<div id="improved-ssl-support" class="section">

### Improved SSL Support<a href="#improved-ssl-support" class="headerlink" title="Permalink to this headline">¶</a>

Bill Janssen made extensive improvements to Python 2.6’s support for the Secure Sockets Layer by adding a new module, <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a>, that’s built atop the <a href="https://www.openssl.org/" class="reference external">OpenSSL</a> library. This new module provides more control over the protocol negotiated, the X.509 certificates used, and has better support for writing SSL servers (as opposed to clients) in Python. The existing SSL support in the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module hasn’t been removed and continues to work, though it will be removed in Python 3.0.

To use the new module, you must first create a TCP connection in the usual way and then pass it to the <a href="../library/ssl.html#ssl.wrap_socket" class="reference internal" title="ssl.wrap_socket"><span class="pre"><code class="sourceCode python">ssl.wrap_socket()</code></span></a> function. It’s possible to specify whether a certificate is required, and to obtain certificate info by calling the <span class="pre">`getpeercert()`</span> method.

<div class="admonition seealso">

See also

The documentation for the <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module.

</div>

</div>

</div>

<div id="deprecations-and-removals" class="section">

## Deprecations and Removals<a href="#deprecations-and-removals" class="headerlink" title="Permalink to this headline">¶</a>

- String exceptions have been removed. Attempting to use them raises a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>.

- Changes to the <span class="pre">`Exception`</span> interface as dictated by <span id="index-25" class="target"></span><a href="https://www.python.org/dev/peps/pep-0352" class="pep reference external"><strong>PEP 352</strong></a> continue to be made. For 2.6, the <span class="pre">`message`</span> attribute is being deprecated in favor of the <span class="pre">`args`</span> attribute.

- (3.0-warning mode) Python 3.0 will feature a reorganized standard library that will drop many outdated modules and rename others. Python 2.6 running in 3.0-warning mode will warn about these modules when they are imported.

  The list of deprecated modules is: <span class="pre">`audiodev`</span>, <span class="pre">`bgenlocations`</span>, <a href="../library/undoc.html#module-buildtools" class="reference internal" title="buildtools: Helper module for BuildApplet, BuildApplication and macfreeze. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">buildtools</code></span></a>, <span class="pre">`bundlebuilder`</span>, <span class="pre">`Canvas`</span>, <a href="../library/compiler.html#module-compiler" class="reference internal" title="compiler: Python code compiler written in Python. (deprecated)"><span class="pre"><code class="sourceCode python">compiler</code></span></a>, <a href="../library/dircache.html#module-dircache" class="reference internal" title="dircache: Return directory listing, with cache mechanism. (deprecated)"><span class="pre"><code class="sourceCode python">dircache</code></span></a>, <a href="../library/dl.html#module-dl" class="reference internal" title="dl: Call C functions in shared objects. (deprecated) (Unix)"><span class="pre"><code class="sourceCode python">dl</code></span></a>, <a href="../library/fpformat.html#module-fpformat" class="reference internal" title="fpformat: General floating point formatting functions. (deprecated)"><span class="pre"><code class="sourceCode python">fpformat</code></span></a>, <a href="../library/gensuitemodule.html#module-gensuitemodule" class="reference internal" title="gensuitemodule: Create a stub package from an OSA dictionary (Mac)"><span class="pre"><code class="sourceCode python">gensuitemodule</code></span></a>, <span class="pre">`ihooks`</span>, <a href="../library/imageop.html#module-imageop" class="reference internal" title="imageop: Manipulate raw image data. (deprecated)"><span class="pre"><code class="sourceCode python">imageop</code></span></a>, <a href="../library/imgfile.html#module-imgfile" class="reference internal" title="imgfile: Support for SGI imglib files. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">imgfile</code></span></a>, <span class="pre">`linuxaudiodev`</span>, <a href="../library/mhlib.html#module-mhlib" class="reference internal" title="mhlib: Manipulate MH mailboxes from Python. (deprecated)"><span class="pre"><code class="sourceCode python">mhlib</code></span></a>, <a href="../library/mimetools.html#module-mimetools" class="reference internal" title="mimetools: Tools for parsing MIME-style message bodies. (deprecated)"><span class="pre"><code class="sourceCode python">mimetools</code></span></a>, <a href="../library/multifile.html#module-multifile" class="reference internal" title="multifile: Support for reading files which contain distinct parts, such as some MIME data. (deprecated)"><span class="pre"><code class="sourceCode python">multifile</code></span></a>, <a href="../library/new.html#module-new" class="reference internal" title="new: Interface to the creation of runtime implementation objects. (deprecated)"><span class="pre"><code class="sourceCode python">new</code></span></a>, <span class="pre">`pure`</span>, <a href="../library/statvfs.html#module-statvfs" class="reference internal" title="statvfs: Constants for interpreting the result of os.statvfs(). (deprecated)"><span class="pre"><code class="sourceCode python">statvfs</code></span></a>, <a href="../library/sunaudio.html#module-sunaudiodev" class="reference internal" title="sunaudiodev: Access to Sun audio hardware. (deprecated) (SunOS)"><span class="pre"><code class="sourceCode python">sunaudiodev</code></span></a>, <span class="pre">`test.testall`</span>, and <span class="pre">`toaiff`</span>.

- The <span class="pre">`gopherlib`</span> module has been removed.

- The <a href="../library/mimewriter.html#module-MimeWriter" class="reference internal" title="MimeWriter: Write MIME format files. (deprecated)"><span class="pre"><code class="sourceCode python">MimeWriter</code></span></a> module and <a href="../library/mimify.html#module-mimify" class="reference internal" title="mimify: Mimification and unmimification of mail messages. (deprecated)"><span class="pre"><code class="sourceCode python">mimify</code></span></a> module have been deprecated; use the <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages, including MIME documents."><span class="pre"><code class="sourceCode python">email</code></span></a> package instead.

- The <a href="../library/md5.html#module-md5" class="reference internal" title="md5: RSA&#39;s MD5 message digest algorithm. (deprecated)"><span class="pre"><code class="sourceCode python">md5</code></span></a> module has been deprecated; use the <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module instead.

- The <a href="../library/posixfile.html#module-posixfile" class="reference internal" title="posixfile: A file-like object with support for locking. (deprecated) (Unix)"><span class="pre"><code class="sourceCode python">posixfile</code></span></a> module has been deprecated; <a href="../library/fcntl.html#fcntl.lockf" class="reference internal" title="fcntl.lockf"><span class="pre"><code class="sourceCode python">fcntl.lockf()</code></span></a> provides better locking.

- The <a href="../library/popen2.html#module-popen2" class="reference internal" title="popen2: Subprocesses with accessible standard I/O streams. (deprecated)"><span class="pre"><code class="sourceCode python">popen2</code></span></a> module has been deprecated; use the <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module.

- The <span class="pre">`rgbimg`</span> module has been removed.

- The <a href="../library/sets.html#module-sets" class="reference internal" title="sets: Implementation of sets of unique elements. (deprecated)"><span class="pre"><code class="sourceCode python">sets</code></span></a> module has been deprecated; it’s better to use the built-in <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a> and <a href="../library/stdtypes.html#frozenset" class="reference internal" title="frozenset"><span class="pre"><code class="sourceCode python"><span class="bu">frozenset</span></code></span></a> types.

- The <a href="../library/sha.html#module-sha" class="reference internal" title="sha: NIST&#39;s secure hash algorithm, SHA. (deprecated)"><span class="pre"><code class="sourceCode python">sha</code></span></a> module has been deprecated; use the <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module instead.

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Permalink to this headline">¶</a>

Changes to Python’s build process and to the C API include:

- Python now must be compiled with C89 compilers (after 19 years!). This means that the Python source tree has dropped its own implementations of <span class="pre">`memmove()`</span> and <span class="pre">`strerror()`</span>, which are in the C89 standard library.

- Python 2.6 can be built with Microsoft Visual Studio 2008 (version 9.0), and this is the new default compiler. See the <span class="pre">`PCbuild`</span> directory for the build files. (Implemented by Christian Heimes.)

- On Mac OS X, Python 2.6 can be compiled as a 4-way universal build. The **configure** script can take a <span class="pre">`--with-universal-archs=[32-bit|64-bit|all]`</span> switch, controlling whether the binaries are built for 32-bit architectures (x86, PowerPC), 64-bit (x86-64 and PPC-64), or both. (Contributed by Ronald Oussoren.)

- The BerkeleyDB module now has a C API object, available as <span class="pre">`bsddb.db.api`</span>. This object can be used by other C extensions that wish to use the <a href="../library/bsddb.html#module-bsddb" class="reference internal" title="bsddb: Interface to Berkeley DB database library"><span class="pre"><code class="sourceCode python">bsddb</code></span></a> module for their own purposes. (Contributed by Duncan Grisby.)

- The new buffer interface, previously described in <a href="#pep-3118-revised-buffer-protocol" class="reference external">the PEP 3118 section</a>, adds <a href="../c-api/buffer.html#c.PyObject_GetBuffer" class="reference internal" title="PyObject_GetBuffer"><span class="pre"><code class="sourceCode c">PyObject_GetBuffer<span class="op">()</span></code></span></a> and <a href="../c-api/buffer.html#c.PyBuffer_Release" class="reference internal" title="PyBuffer_Release"><span class="pre"><code class="sourceCode c">PyBuffer_Release<span class="op">()</span></code></span></a>, as well as a few other functions.

- Python’s use of the C stdio library is now thread-safe, or at least as thread-safe as the underlying library is. A long-standing potential bug occurred if one thread closed a file object while another thread was reading from or writing to the object. In 2.6 file objects have a reference count, manipulated by the <a href="../c-api/file.html#c.PyFile_IncUseCount" class="reference internal" title="PyFile_IncUseCount"><span class="pre"><code class="sourceCode c">PyFile_IncUseCount<span class="op">()</span></code></span></a> and <a href="../c-api/file.html#c.PyFile_DecUseCount" class="reference internal" title="PyFile_DecUseCount"><span class="pre"><code class="sourceCode c">PyFile_DecUseCount<span class="op">()</span></code></span></a> functions. File objects can’t be closed unless the reference count is zero. <a href="../c-api/file.html#c.PyFile_IncUseCount" class="reference internal" title="PyFile_IncUseCount"><span class="pre"><code class="sourceCode c">PyFile_IncUseCount<span class="op">()</span></code></span></a> should be called while the GIL is still held, before carrying out an I/O operation using the <span class="pre">`FILE`</span>` `<span class="pre">`*`</span> pointer, and <a href="../c-api/file.html#c.PyFile_DecUseCount" class="reference internal" title="PyFile_DecUseCount"><span class="pre"><code class="sourceCode c">PyFile_DecUseCount<span class="op">()</span></code></span></a> should be called immediately after the GIL is re-acquired. (Contributed by Antoine Pitrou and Gregory P. Smith.)

- Importing modules simultaneously in two different threads no longer deadlocks; it will now raise an <a href="../library/exceptions.html#exceptions.ImportError" class="reference internal" title="exceptions.ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a>. A new API function, <a href="../c-api/import.html#c.PyImport_ImportModuleNoBlock" class="reference internal" title="PyImport_ImportModuleNoBlock"><span class="pre"><code class="sourceCode c">PyImport_ImportModuleNoBlock<span class="op">()</span></code></span></a>, will look for a module in <span class="pre">`sys.modules`</span> first, then try to import it after acquiring an import lock. If the import lock is held by another thread, an <a href="../library/exceptions.html#exceptions.ImportError" class="reference internal" title="exceptions.ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> is raised. (Contributed by Christian Heimes.)

- Several functions return information about the platform’s floating-point support. <a href="../c-api/float.html#c.PyFloat_GetMax" class="reference internal" title="PyFloat_GetMax"><span class="pre"><code class="sourceCode c">PyFloat_GetMax<span class="op">()</span></code></span></a> returns the maximum representable floating point value, and <a href="../c-api/float.html#c.PyFloat_GetMin" class="reference internal" title="PyFloat_GetMin"><span class="pre"><code class="sourceCode c">PyFloat_GetMin<span class="op">()</span></code></span></a> returns the minimum positive value. <a href="../c-api/float.html#c.PyFloat_GetInfo" class="reference internal" title="PyFloat_GetInfo"><span class="pre"><code class="sourceCode c">PyFloat_GetInfo<span class="op">()</span></code></span></a> returns an object containing more information from the <span class="pre">`float.h`</span> file, such as <span class="pre">`"mant_dig"`</span> (number of digits in the mantissa), <span class="pre">`"epsilon"`</span> (smallest difference between 1.0 and the next largest value representable), and several others. (Contributed by Christian Heimes; <a href="https://bugs.python.org/issue1534" class="reference external">bpo-1534</a>.)

- C functions and methods that use <a href="../c-api/complex.html#c.PyComplex_AsCComplex" class="reference internal" title="PyComplex_AsCComplex"><span class="pre"><code class="sourceCode c">PyComplex_AsCComplex<span class="op">()</span></code></span></a> will now accept arguments that have a <a href="../reference/datamodel.html#object.__complex__" class="reference internal" title="object.__complex__"><span class="pre"><code class="sourceCode python"><span class="fu">__complex__</span>()</code></span></a> method. In particular, the functions in the <a href="../library/cmath.html#module-cmath" class="reference internal" title="cmath: Mathematical functions for complex numbers."><span class="pre"><code class="sourceCode python">cmath</code></span></a> module will now accept objects with this method. This is a backport of a Python 3.0 change. (Contributed by Mark Dickinson; <a href="https://bugs.python.org/issue1675423" class="reference external">bpo-1675423</a>.)

- Python’s C API now includes two functions for case-insensitive string comparisons, <span class="pre">`PyOS_stricmp(char*,`</span>` `<span class="pre">`char*)`</span> and <span class="pre">`PyOS_strnicmp(char*,`</span>` `<span class="pre">`char*,`</span>` `<span class="pre">`Py_ssize_t)`</span>. (Contributed by Christian Heimes; <a href="https://bugs.python.org/issue1635" class="reference external">bpo-1635</a>.)

- Many C extensions define their own little macro for adding integers and strings to the module’s dictionary in the <span class="pre">`init*`</span> function. Python 2.6 finally defines standard macros for adding values to a module, <a href="../c-api/module.html#c.PyModule_AddStringMacro" class="reference internal" title="PyModule_AddStringMacro"><span class="pre"><code class="sourceCode c">PyModule_AddStringMacro</code></span></a> and <span class="pre">`PyModule_AddIntMacro()`</span>. (Contributed by Christian Heimes.)

- Some macros were renamed in both 3.0 and 2.6 to make it clearer that they are macros, not functions. <span class="pre">`Py_Size()`</span> became <span class="pre">`Py_SIZE()`</span>, <span class="pre">`Py_Type()`</span> became <span class="pre">`Py_TYPE()`</span>, and <span class="pre">`Py_Refcnt()`</span> became <span class="pre">`Py_REFCNT()`</span>. The mixed-case macros are still available in Python 2.6 for backward compatibility. (<a href="https://bugs.python.org/issue1629" class="reference external">bpo-1629</a>)

- Distutils now places C extensions it builds in a different directory when running on a debug version of Python. (Contributed by Collin Winter; <a href="https://bugs.python.org/issue1530959" class="reference external">bpo-1530959</a>.)

- Several basic data types, such as integers and strings, maintain internal free lists of objects that can be re-used. The data structures for these free lists now follow a naming convention: the variable is always named <span class="pre">`free_list`</span>, the counter is always named <span class="pre">`numfree`</span>, and a macro <span class="pre">`Py<typename>_MAXFREELIST`</span> is always defined.

- A new Makefile target, “make patchcheck”, prepares the Python source tree for making a patch: it fixes trailing whitespace in all modified <span class="pre">`.py`</span> files, checks whether the documentation has been changed, and reports whether the <span class="pre">`Misc/ACKS`</span> and <span class="pre">`Misc/NEWS`</span> files have been updated. (Contributed by Brett Cannon.)

  Another new target, “make profile-opt”, compiles a Python binary using GCC’s profile-guided optimization. It compiles Python with profiling enabled, runs the test suite to obtain a set of profiling results, and then compiles using these results for optimization. (Contributed by Gregory P. Smith.)

<div id="port-specific-changes-windows" class="section">

### Port-Specific Changes: Windows<a href="#port-specific-changes-windows" class="headerlink" title="Permalink to this headline">¶</a>

- The support for Windows 95, 98, ME and NT4 has been dropped. Python 2.6 requires at least Windows 2000 SP4.

- The new default compiler on Windows is Visual Studio 2008 (version 9.0). The build directories for Visual Studio 2003 (version 7.1) and 2005 (version 8.0) were moved into the PC/ directory. The new <span class="pre">`PCbuild`</span> directory supports cross compilation for X64, debug builds and Profile Guided Optimization (PGO). PGO builds are roughly 10% faster than normal builds. (Contributed by Christian Heimes with help from Amaury Forgeot d’Arc and Martin von Loewis.)

- The <a href="../library/msvcrt.html#module-msvcrt" class="reference internal" title="msvcrt: Miscellaneous useful routines from the MS VC++ runtime. (Windows)"><span class="pre"><code class="sourceCode python">msvcrt</code></span></a> module now supports both the normal and wide char variants of the console I/O API. The <span class="pre">`getwch()`</span> function reads a keypress and returns a Unicode value, as does the <span class="pre">`getwche()`</span> function. The <span class="pre">`putwch()`</span> function takes a Unicode character and writes it to the console. (Contributed by Christian Heimes.)

- <a href="../library/os.path.html#os.path.expandvars" class="reference internal" title="os.path.expandvars"><span class="pre"><code class="sourceCode python">os.path.expandvars()</code></span></a> will now expand environment variables in the form “%var%”, and “~user” will be expanded into the user’s home directory path. (Contributed by Josiah Carlson; <a href="https://bugs.python.org/issue957650" class="reference external">bpo-957650</a>.)

- The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module’s socket objects now have an <span class="pre">`ioctl()`</span> method that provides a limited interface to the <span class="pre">`WSAIoctl()`</span> system interface.

- The <a href="../library/_winreg.html#module-_winreg" class="reference internal" title="_winreg: Routines and objects for manipulating the Windows registry. (Windows)"><span class="pre"><code class="sourceCode python">_winreg</code></span></a> module now has a function, <span class="pre">`ExpandEnvironmentStrings()`</span>, that expands environment variable references such as <span class="pre">`%NAME%`</span> in an input string. The handle objects provided by this module now support the context protocol, so they can be used in <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statements. (Contributed by Christian Heimes.)

  <a href="../library/_winreg.html#module-_winreg" class="reference internal" title="_winreg: Routines and objects for manipulating the Windows registry. (Windows)"><span class="pre"><code class="sourceCode python">_winreg</code></span></a> also has better support for x64 systems, exposing the <span class="pre">`DisableReflectionKey()`</span>, <span class="pre">`EnableReflectionKey()`</span>, and <span class="pre">`QueryReflectionKey()`</span> functions, which enable and disable registry reflection for 32-bit processes running on 64-bit systems. (<a href="https://bugs.python.org/issue1753245" class="reference external">bpo-1753245</a>)

- The <a href="../library/msilib.html#module-msilib" class="reference internal" title="msilib: Creation of Microsoft Installer files, and CAB files. (Windows)"><span class="pre"><code class="sourceCode python">msilib</code></span></a> module’s <span class="pre">`Record`</span> object gained <span class="pre">`GetInteger()`</span> and <span class="pre">`GetString()`</span> methods that return field values as an integer or a string. (Contributed by Floris Bruynooghe; <a href="https://bugs.python.org/issue2125" class="reference external">bpo-2125</a>.)

</div>

<div id="port-specific-changes-mac-os-x" class="section">

### Port-Specific Changes: Mac OS X<a href="#port-specific-changes-mac-os-x" class="headerlink" title="Permalink to this headline">¶</a>

- When compiling a framework build of Python, you can now specify the framework name to be used by providing the <span class="pre">`--with-framework-name=`</span> option to the **configure** script.

- The <span class="pre">`macfs`</span> module has been removed. This in turn required the <a href="../library/macostools.html#macostools.touched" class="reference internal" title="macostools.touched"><span class="pre"><code class="sourceCode python">macostools.touched()</code></span></a> function to be removed because it depended on the <span class="pre">`macfs`</span> module. (<a href="https://bugs.python.org/issue1490190" class="reference external">bpo-1490190</a>)

- Many other Mac OS modules have been deprecated and will be removed in Python 3.0: <span class="pre">`_builtinSuites`</span>, <a href="../library/aepack.html#module-aepack" class="reference internal" title="aepack: Conversion between Python variables and AppleEvent data containers. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">aepack</code></span></a>, <a href="../library/aetools.html#module-aetools" class="reference internal" title="aetools: Basic support for sending Apple Events (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">aetools</code></span></a>, <a href="../library/aetypes.html#module-aetypes" class="reference internal" title="aetypes: Python representation of the Apple Event Object Model. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">aetypes</code></span></a>, <a href="../library/undoc.html#module-applesingle" class="reference internal" title="applesingle: Rudimentary decoder for AppleSingle format files. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">applesingle</code></span></a>, <span class="pre">`appletrawmain`</span>, <span class="pre">`appletrunner`</span>, <span class="pre">`argvemulator`</span>, <span class="pre">`Audio_mac`</span>, <a href="../library/autogil.html#module-autoGIL" class="reference internal" title="autoGIL: Global Interpreter Lock handling in event loops. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">autoGIL</code></span></a>, <span class="pre">`Carbon`</span>, <a href="../library/undoc.html#module-cfmfile" class="reference internal" title="cfmfile: Code Fragment Resource module. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">cfmfile</code></span></a>, <span class="pre">`CodeWarrior`</span>, <a href="../library/colorpicker.html#module-ColorPicker" class="reference internal" title="ColorPicker: Interface to the standard color selection dialog. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">ColorPicker</code></span></a>, <a href="../library/easydialogs.html#module-EasyDialogs" class="reference internal" title="EasyDialogs: Basic Macintosh dialogs. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">EasyDialogs</code></span></a>, <span class="pre">`Explorer`</span>, <span class="pre">`Finder`</span>, <a href="../library/framework.html#module-FrameWork" class="reference internal" title="FrameWork: Interactive application framework. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">FrameWork</code></span></a>, <a href="../library/macostools.html#module-findertools" class="reference internal" title="findertools: Wrappers around the finder&#39;s Apple Events interface. (Mac)"><span class="pre"><code class="sourceCode python">findertools</code></span></a>, <a href="../library/ic.html#module-ic" class="reference internal" title="ic: Access to the Mac OS X Internet Config. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">ic</code></span></a>, <span class="pre">`icglue`</span>, <a href="../library/undoc.html#module-icopen" class="reference internal" title="icopen: Internet Config replacement for open(). (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">icopen</code></span></a>, <a href="../library/undoc.html#module-macerrors" class="reference internal" title="macerrors: Constant definitions for many Mac OS error codes. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">macerrors</code></span></a>, <a href="../library/macos.html#module-MacOS" class="reference internal" title="MacOS: Access to Mac OS-specific interpreter features. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">MacOS</code></span></a>, <span class="pre">`macfs`</span>, <a href="../library/macostools.html#module-macostools" class="reference internal" title="macostools: Convenience routines for file manipulation. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">macostools</code></span></a>, <a href="../library/undoc.html#module-macresource" class="reference internal" title="macresource: Locate script resources. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">macresource</code></span></a>, <a href="../library/miniaeframe.html#module-MiniAEFrame" class="reference internal" title="MiniAEFrame: Support to act as an Open Scripting Architecture (OSA) server (&quot;Apple Events&quot;). (Mac)"><span class="pre"><code class="sourceCode python">MiniAEFrame</code></span></a>, <a href="../library/undoc.html#module-Nav" class="reference internal" title="Nav: Interface to Navigation Services. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">Nav</code></span></a>, <span class="pre">`Netscape`</span>, <span class="pre">`OSATerminology`</span>, <span class="pre">`pimp`</span>, <a href="../library/undoc.html#module-PixMapWrapper" class="reference internal" title="PixMapWrapper: Wrapper for PixMap objects. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">PixMapWrapper</code></span></a>, <span class="pre">`StdSuites`</span>, <span class="pre">`SystemEvents`</span>, <span class="pre">`Terminal`</span>, and <span class="pre">`terminalcommand`</span>.

</div>

<div id="port-specific-changes-irix" class="section">

### Port-Specific Changes: IRIX<a href="#port-specific-changes-irix" class="headerlink" title="Permalink to this headline">¶</a>

A number of old IRIX-specific modules were deprecated and will be removed in Python 3.0: <a href="../library/al.html#module-al" class="reference internal" title="al: Audio functions on the SGI. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">al</code></span></a> and <a href="../library/al.html#module-AL" class="reference internal" title="AL: Constants used with the al module. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">AL</code></span></a>, <a href="../library/cd.html#module-cd" class="reference internal" title="cd: Interface to the CD-ROM on Silicon Graphics systems. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">cd</code></span></a>, <span class="pre">`cddb`</span>, <span class="pre">`cdplayer`</span>, <span class="pre">`CL`</span> and <span class="pre">`cl`</span>, <a href="../library/gl.html#module-DEVICE" class="reference internal" title="DEVICE: Constants used with the gl module. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">DEVICE</code></span></a>, <span class="pre">`ERRNO`</span>, <span class="pre">`FILE`</span>, <a href="../library/fl.html#module-FL" class="reference internal" title="FL: Constants used with the fl module. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">FL</code></span></a> and <a href="../library/fl.html#module-fl" class="reference internal" title="fl: FORMS library for applications with graphical user interfaces. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">fl</code></span></a>, <a href="../library/fl.html#module-flp" class="reference internal" title="flp: Functions for loading stored FORMS designs. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">flp</code></span></a>, <a href="../library/fm.html#module-fm" class="reference internal" title="fm: Font Manager interface for SGI workstations. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">fm</code></span></a>, <span class="pre">`GET`</span>, <span class="pre">`GLWS`</span>, <a href="../library/gl.html#module-GL" class="reference internal" title="GL: Constants used with the gl module. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">GL</code></span></a> and <a href="../library/gl.html#module-gl" class="reference internal" title="gl: Functions from the Silicon Graphics Graphics Library. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">gl</code></span></a>, <span class="pre">`IN`</span>, <span class="pre">`IOCTL`</span>, <a href="../library/jpeg.html#module-jpeg" class="reference internal" title="jpeg: Read and write image files in compressed JPEG format. (deprecated) (IRIX)"><span class="pre"><code class="sourceCode python">jpeg</code></span></a>, <span class="pre">`panelparser`</span>, <span class="pre">`readcd`</span>, <span class="pre">`SV`</span> and <span class="pre">`sv`</span>, <span class="pre">`torgb`</span>, <a href="../library/undoc.html#module-videoreader" class="reference internal" title="videoreader: Read QuickTime movies frame by frame for further processing. (deprecated) (Mac)"><span class="pre"><code class="sourceCode python">videoreader</code></span></a>, and <span class="pre">`WAIT`</span>.

</div>

</div>

<div id="porting-to-python-2-6" class="section">

## Porting to Python 2.6<a href="#porting-to-python-2-6" class="headerlink" title="Permalink to this headline">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code:

- Classes that aren’t supposed to be hashable should set <span class="pre">`__hash__`</span>` `<span class="pre">`=`</span>` `<span class="pre">`None`</span> in their definitions to indicate the fact.

- String exceptions have been removed. Attempting to use them raises a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>.

- The <a href="../reference/datamodel.html#object.__init__" class="reference internal" title="object.__init__"><span class="pre"><code class="sourceCode python"><span class="fu">__init__</span>()</code></span></a> method of <a href="../library/collections.html#collections.deque" class="reference internal" title="collections.deque"><span class="pre"><code class="sourceCode python">collections.deque</code></span></a> now clears any existing contents of the deque before adding elements from the iterable. This change makes the behavior match <span class="pre">`list.__init__()`</span>.

- <a href="../reference/datamodel.html#object.__init__" class="reference internal" title="object.__init__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__init__</span>()</code></span></a> previously accepted arbitrary arguments and keyword arguments, ignoring them. In Python 2.6, this is no longer allowed and will result in a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>. This will affect <a href="../reference/datamodel.html#object.__init__" class="reference internal" title="object.__init__"><span class="pre"><code class="sourceCode python"><span class="fu">__init__</span>()</code></span></a> methods that end up calling the corresponding method on <a href="../library/functions.html#object" class="reference internal" title="object"><span class="pre"><code class="sourceCode python"><span class="bu">object</span></code></span></a> (perhaps through using <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a>). See <a href="https://bugs.python.org/issue1683368" class="reference external">bpo-1683368</a> for discussion.

- The <span class="pre">`Decimal`</span> constructor now accepts leading and trailing whitespace when passed a string. Previously it would raise an <span class="pre">`InvalidOperation`</span> exception. On the other hand, the <span class="pre">`create_decimal()`</span> method of <span class="pre">`Context`</span> objects now explicitly disallows extra whitespace, raising a <span class="pre">`ConversionSyntax`</span> exception.

- Due to an implementation accident, if you passed a file path to the built-in <a href="../library/functions.html#__import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> function, it would actually import the specified file. This was never intended to work, however, and the implementation now explicitly checks for this case and raises an <a href="../library/exceptions.html#exceptions.ImportError" class="reference internal" title="exceptions.ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a>.

- C API: the <a href="../c-api/import.html#c.PyImport_Import" class="reference internal" title="PyImport_Import"><span class="pre"><code class="sourceCode c">PyImport_Import<span class="op">()</span></code></span></a> and <a href="../c-api/import.html#c.PyImport_ImportModule" class="reference internal" title="PyImport_ImportModule"><span class="pre"><code class="sourceCode c">PyImport_ImportModule<span class="op">()</span></code></span></a> functions now default to absolute imports, not relative imports. This will affect C extensions that import other modules.

- C API: extension data types that shouldn’t be hashable should define their <span class="pre">`tp_hash`</span> slot to <a href="../c-api/object.html#c.PyObject_HashNotImplemented" class="reference internal" title="PyObject_HashNotImplemented"><span class="pre"><code class="sourceCode c">PyObject_HashNotImplemented<span class="op">()</span></code></span></a>.

- The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module exception <a href="../library/socket.html#socket.error" class="reference internal" title="socket.error"><span class="pre"><code class="sourceCode python">socket.error</code></span></a> now inherits from <a href="../library/exceptions.html#exceptions.IOError" class="reference internal" title="exceptions.IOError"><span class="pre"><code class="sourceCode python"><span class="pp">IOError</span></code></span></a>. Previously it wasn’t a subclass of <a href="../library/exceptions.html#exceptions.StandardError" class="reference internal" title="exceptions.StandardError"><span class="pre"><code class="sourceCode python"><span class="pp">StandardError</span></code></span></a> but now it is, through <a href="../library/exceptions.html#exceptions.IOError" class="reference internal" title="exceptions.IOError"><span class="pre"><code class="sourceCode python"><span class="pp">IOError</span></code></span></a>. (Implemented by Gregory P. Smith; <a href="https://bugs.python.org/issue1706815" class="reference external">bpo-1706815</a>.)

- The <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a> module no longer automatically converts <a href="../library/datetime.html#datetime.date" class="reference internal" title="datetime.date"><span class="pre"><code class="sourceCode python">datetime.date</code></span></a> and <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">datetime.time</code></span></a> to the <a href="../library/xmlrpclib.html#xmlrpclib.DateTime" class="reference internal" title="xmlrpclib.DateTime"><span class="pre"><code class="sourceCode python">xmlrpclib.DateTime</code></span></a> type; the conversion semantics were not necessarily correct for all applications. Code using <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a> should convert <span class="pre">`date`</span> and <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">time</code></span></a> instances. (<a href="https://bugs.python.org/issue1330538" class="reference external">bpo-1330538</a>)

- (3.0-warning mode) The <span class="pre">`Exception`</span> class now warns when accessed using slicing or index access; having <span class="pre">`Exception`</span> behave like a tuple is being phased out.

- (3.0-warning mode) inequality comparisons between two dictionaries or two objects that don’t implement comparison methods are reported as warnings. <span class="pre">`dict1`</span>` `<span class="pre">`==`</span>` `<span class="pre">`dict2`</span> still works, but <span class="pre">`dict1`</span>` `<span class="pre">`<`</span>` `<span class="pre">`dict2`</span> is being phased out.

  Comparisons between cells, which are an implementation detail of Python’s scoping rules, also cause warnings because such comparisons are forbidden entirely in 3.0.

</div>

<div id="acknowledgements" class="section">

<span id="acks"></span>

## Acknowledgements<a href="#acknowledgements" class="headerlink" title="Permalink to this headline">¶</a>

The author would like to thank the following people for offering suggestions, corrections and assistance with various drafts of this article: Georg Brandl, Steve Brown, Nick Coghlan, Ralph Corderoy, Jim Jewett, Kent Johnson, Chris Lambacher, Martin Michlmayr, Antoine Pitrou, Brian Warner.

</div>

</div>

</div>
