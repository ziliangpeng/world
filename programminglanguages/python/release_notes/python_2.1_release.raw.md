<div class="body" role="main">

<div id="what-s-new-in-python-2-1" class="section">

# What’s New in Python 2.1<a href="#what-s-new-in-python-2-1" class="headerlink" title="Permalink to this headline">¶</a>

Author  
A.M. Kuchling

<div id="introduction" class="section">

## Introduction<a href="#introduction" class="headerlink" title="Permalink to this headline">¶</a>

This article explains the new features in Python 2.1. While there aren’t as many changes in 2.1 as there were in Python 2.0, there are still some pleasant surprises in store. 2.1 is the first release to be steered through the use of Python Enhancement Proposals, or PEPs, so most of the sizable changes have accompanying PEPs that provide more complete documentation and a design rationale for the change. This article doesn’t attempt to document the new features completely, but simply provides an overview of the new features for Python programmers. Refer to the Python 2.1 documentation, or to the specific PEP, for more details about any new feature that particularly interests you.

One recent goal of the Python development team has been to accelerate the pace of new releases, with a new release coming every 6 to 9 months. 2.1 is the first release to come out at this faster pace, with the first alpha appearing in January, 3 months after the final version of 2.0 was released.

The final release of Python 2.1 was made on April 17, 2001.

</div>

<div id="pep-227-nested-scopes" class="section">

## PEP 227: Nested Scopes<a href="#pep-227-nested-scopes" class="headerlink" title="Permalink to this headline">¶</a>

The largest change in Python 2.1 is to Python’s scoping rules. In Python 2.0, at any given time there are at most three namespaces used to look up variable names: local, module-level, and the built-in namespace. This often surprised people because it didn’t match their intuitive expectations. For example, a nested recursive function definition doesn’t work:

<div class="highlight-default notranslate">

<div class="highlight">

    def f():
        ...
        def g(value):
            ...
            return g(value-1) + 1
        ...

</div>

</div>

The function <span class="pre">`g()`</span> will always raise a <a href="../library/exceptions.html#exceptions.NameError" class="reference internal" title="exceptions.NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a> exception, because the binding of the name <span class="pre">`g`</span> isn’t in either its local namespace or in the module-level namespace. This isn’t much of a problem in practice (how often do you recursively define interior functions like this?), but this also made using the <a href="../reference/expressions.html#lambda" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">lambda</code></span></a> statement clumsier, and this was a problem in practice. In code which uses <a href="../reference/expressions.html#lambda" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">lambda</code></span></a> you can often find local variables being copied by passing them as the default values of arguments.

<div class="highlight-default notranslate">

<div class="highlight">

    def find(self, name):
        "Return list of any entries equal to 'name'"
        L = filter(lambda x, name=name: x == name,
                   self.list_attribute)
        return L

</div>

</div>

The readability of Python code written in a strongly functional style suffers greatly as a result.

The most significant change to Python 2.1 is that static scoping has been added to the language to fix this problem. As a first effect, the <span class="pre">`name=name`</span> default argument is now unnecessary in the above example. Put simply, when a given variable name is not assigned a value within a function (by an assignment, or the <a href="../reference/compound_stmts.html#def" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">def</code></span></a>, <a href="../reference/compound_stmts.html#class" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">class</code></span></a>, or <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statements), references to the variable will be looked up in the local namespace of the enclosing scope. A more detailed explanation of the rules, and a dissection of the implementation, can be found in the PEP.

This change may cause some compatibility problems for code where the same variable name is used both at the module level and as a local variable within a function that contains further function definitions. This seems rather unlikely though, since such code would have been pretty confusing to read in the first place.

One side effect of the change is that the <span class="pre">`from`</span>` `<span class="pre">`module`</span>` `<span class="pre">`import`</span>` `<span class="pre">`*`</span> and <a href="../reference/simple_stmts.html#exec" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">exec</code></span></a> statements have been made illegal inside a function scope under certain conditions. The Python reference manual has said all along that <span class="pre">`from`</span>` `<span class="pre">`module`</span>` `<span class="pre">`import`</span>` `<span class="pre">`*`</span> is only legal at the top level of a module, but the CPython interpreter has never enforced this before. As part of the implementation of nested scopes, the compiler which turns Python source into bytecodes has to generate different code to access variables in a containing scope. <span class="pre">`from`</span>` `<span class="pre">`module`</span>` `<span class="pre">`import`</span>` `<span class="pre">`*`</span> and <a href="../reference/simple_stmts.html#exec" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">exec</code></span></a> make it impossible for the compiler to figure this out, because they add names to the local namespace that are unknowable at compile time. Therefore, if a function contains function definitions or <a href="../reference/expressions.html#lambda" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">lambda</code></span></a> expressions with free variables, the compiler will flag this by raising a <a href="../library/exceptions.html#exceptions.SyntaxError" class="reference internal" title="exceptions.SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> exception.

To make the preceding explanation a bit clearer, here’s an example:

<div class="highlight-default notranslate">

<div class="highlight">

    x = 1
    def f():
        # The next line is a syntax error
        exec 'x=2'
        def g():
            return x

</div>

</div>

Line 4 containing the <a href="../reference/simple_stmts.html#exec" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">exec</code></span></a> statement is a syntax error, since <a href="../reference/simple_stmts.html#exec" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">exec</code></span></a> would define a new local variable named <span class="pre">`x`</span> whose value should be accessed by <span class="pre">`g()`</span>.

This shouldn’t be much of a limitation, since <a href="../reference/simple_stmts.html#exec" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">exec</code></span></a> is rarely used in most Python code (and when it is used, it’s often a sign of a poor design anyway).

Compatibility concerns have led to nested scopes being introduced gradually; in Python 2.1, they aren’t enabled by default, but can be turned on within a module by using a future statement as described in PEP 236. (See the following section for further discussion of PEP 236.) In Python 2.2, nested scopes will become the default and there will be no way to turn them off, but users will have had all of 2.1’s lifetime to fix any breakage resulting from their introduction.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://www.python.org/dev/peps/pep-0227" class="pep reference external"><strong>PEP 227</strong></a> - Statically Nested Scopes  
Written and implemented by Jeremy Hylton.

</div>

</div>

<div id="pep-236-future-directives" class="section">

## PEP 236: \_\_future\_\_ Directives<a href="#pep-236-future-directives" class="headerlink" title="Permalink to this headline">¶</a>

The reaction to nested scopes was widespread concern about the dangers of breaking code with the 2.1 release, and it was strong enough to make the Pythoneers take a more conservative approach. This approach consists of introducing a convention for enabling optional functionality in release N that will become compulsory in release N+1.

The syntax uses a <span class="pre">`from...import`</span> statement using the reserved module name <a href="../library/__future__.html#module-__future__" class="reference internal" title="__future__: Future statement definitions"><span class="pre"><code class="sourceCode python">__future__</code></span></a>. Nested scopes can be enabled by the following statement:

<div class="highlight-default notranslate">

<div class="highlight">

    from __future__ import nested_scopes

</div>

</div>

While it looks like a normal <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statement, it’s not; there are strict rules on where such a future statement can be put. They can only be at the top of a module, and must precede any Python code or regular <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statements. This is because such statements can affect how the Python bytecode compiler parses code and generates bytecode, so they must precede any statement that will result in bytecodes being produced.

<div class="admonition seealso">

See also

<span id="index-1" class="target"></span><a href="https://www.python.org/dev/peps/pep-0236" class="pep reference external"><strong>PEP 236</strong></a> - Back to the <a href="../library/__future__.html#module-__future__" class="reference internal" title="__future__: Future statement definitions"><span class="pre"><code class="sourceCode python">__future__</code></span></a>  
Written by Tim Peters, and primarily implemented by Jeremy Hylton.

</div>

</div>

<div id="pep-207-rich-comparisons" class="section">

## PEP 207: Rich Comparisons<a href="#pep-207-rich-comparisons" class="headerlink" title="Permalink to this headline">¶</a>

In earlier versions, Python’s support for implementing comparisons on user-defined classes and extension types was quite simple. Classes could implement a <a href="../reference/datamodel.html#object.__cmp__" class="reference internal" title="object.__cmp__"><span class="pre"><code class="sourceCode python"><span class="fu">__cmp__</span>()</code></span></a> method that was given two instances of a class, and could only return 0 if they were equal or +1 or -1 if they weren’t; the method couldn’t raise an exception or return anything other than a Boolean value. Users of Numeric Python often found this model too weak and restrictive, because in the number-crunching programs that numeric Python is used for, it would be more useful to be able to perform elementwise comparisons of two matrices, returning a matrix containing the results of a given comparison for each element. If the two matrices are of different sizes, then the compare has to be able to raise an exception to signal the error.

In Python 2.1, rich comparisons were added in order to support this need. Python classes can now individually overload each of the <span class="pre">`<`</span>, <span class="pre">`<=`</span>, <span class="pre">`>`</span>, <span class="pre">`>=`</span>, <span class="pre">`==`</span>, and <span class="pre">`!=`</span> operations. The new magic method names are:

| Operation | Method name |
|----|----|
| <span class="pre">`<`</span> | <a href="../reference/datamodel.html#object.__lt__" class="reference internal" title="object.__lt__"><span class="pre"><code class="sourceCode python"><span class="fu">__lt__</span>()</code></span></a> |
| <span class="pre">`<=`</span> | <a href="../reference/datamodel.html#object.__le__" class="reference internal" title="object.__le__"><span class="pre"><code class="sourceCode python"><span class="fu">__le__</span>()</code></span></a> |
| <span class="pre">`>`</span> | <a href="../reference/datamodel.html#object.__gt__" class="reference internal" title="object.__gt__"><span class="pre"><code class="sourceCode python"><span class="fu">__gt__</span>()</code></span></a> |
| <span class="pre">`>=`</span> | <a href="../reference/datamodel.html#object.__ge__" class="reference internal" title="object.__ge__"><span class="pre"><code class="sourceCode python"><span class="fu">__ge__</span>()</code></span></a> |
| <span class="pre">`==`</span> | <a href="../reference/datamodel.html#object.__eq__" class="reference internal" title="object.__eq__"><span class="pre"><code class="sourceCode python"><span class="fu">__eq__</span>()</code></span></a> |
| <span class="pre">`!=`</span> | <a href="../reference/datamodel.html#object.__ne__" class="reference internal" title="object.__ne__"><span class="pre"><code class="sourceCode python"><span class="fu">__ne__</span>()</code></span></a> |

(The magic methods are named after the corresponding Fortran operators <span class="pre">`.LT.`</span>. <span class="pre">`.LE.`</span>, &c. Numeric programmers are almost certainly quite familiar with these names and will find them easy to remember.)

Each of these magic methods is of the form <span class="pre">`method(self,`</span>` `<span class="pre">`other)`</span>, where <span class="pre">`self`</span> will be the object on the left-hand side of the operator, while <span class="pre">`other`</span> will be the object on the right-hand side. For example, the expression <span class="pre">`A`</span>` `<span class="pre">`<`</span>` `<span class="pre">`B`</span> will cause <span class="pre">`A.__lt__(B)`</span> to be called.

Each of these magic methods can return anything at all: a Boolean, a matrix, a list, or any other Python object. Alternatively they can raise an exception if the comparison is impossible, inconsistent, or otherwise meaningless.

The built-in <span class="pre">`cmp(A,B)`</span> function can use the rich comparison machinery, and now accepts an optional argument specifying which comparison operation to use; this is given as one of the strings <span class="pre">`"<"`</span>, <span class="pre">`"<="`</span>, <span class="pre">`">"`</span>, <span class="pre">`">="`</span>, <span class="pre">`"=="`</span>, or <span class="pre">`"!="`</span>. If called without the optional third argument, <a href="../library/functions.html#cmp" class="reference internal" title="cmp"><span class="pre"><code class="sourceCode python"><span class="bu">cmp</span>()</code></span></a> will only return -1, 0, or +1 as in previous versions of Python; otherwise it will call the appropriate method and can return any Python object.

There are also corresponding changes of interest to C programmers; there’s a new slot <span class="pre">`tp_richcmp`</span> in type objects and an API for performing a given rich comparison. I won’t cover the C API here, but will refer you to PEP 207, or to 2.1’s C API documentation, for the full list of related functions.

<div class="admonition seealso">

See also

<span id="index-2" class="target"></span><a href="https://www.python.org/dev/peps/pep-0207" class="pep reference external"><strong>PEP 207</strong></a> - Rich Comparisons  
Written by Guido van Rossum, heavily based on earlier work by David Ascher, and implemented by Guido van Rossum.

</div>

</div>

<div id="pep-230-warning-framework" class="section">

## PEP 230: Warning Framework<a href="#pep-230-warning-framework" class="headerlink" title="Permalink to this headline">¶</a>

Over its 10 years of existence, Python has accumulated a certain number of obsolete modules and features along the way. It’s difficult to know when a feature is safe to remove, since there’s no way of knowing how much code uses it — perhaps no programs depend on the feature, or perhaps many do. To enable removing old features in a more structured way, a warning framework was added. When the Python developers want to get rid of a feature, it will first trigger a warning in the next version of Python. The following Python version can then drop the feature, and users will have had a full release cycle to remove uses of the old feature.

Python 2.1 adds the warning framework to be used in this scheme. It adds a <a href="../library/warnings.html#module-warnings" class="reference internal" title="warnings: Issue warning messages and control their disposition."><span class="pre"><code class="sourceCode python">warnings</code></span></a> module that provide functions to issue warnings, and to filter out warnings that you don’t want to be displayed. Third-party modules can also use this framework to deprecate old features that they no longer wish to support.

For example, in Python 2.1 the <span class="pre">`regex`</span> module is deprecated, so importing it causes a warning to be printed:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> import regex
    __main__:1: DeprecationWarning: the regex module
             is deprecated; please use the re module
    >>>

</div>

</div>

Warnings can be issued by calling the <a href="../library/warnings.html#warnings.warn" class="reference internal" title="warnings.warn"><span class="pre"><code class="sourceCode python">warnings.warn()</code></span></a> function:

<div class="highlight-default notranslate">

<div class="highlight">

    warnings.warn("feature X no longer supported")

</div>

</div>

The first parameter is the warning message; an additional optional parameters can be used to specify a particular warning category.

Filters can be added to disable certain warnings; a regular expression pattern can be applied to the message or to the module name in order to suppress a warning. For example, you may have a program that uses the <span class="pre">`regex`</span> module and not want to spare the time to convert it to use the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module right now. The warning can be suppressed by calling

<div class="highlight-default notranslate">

<div class="highlight">

    import warnings
    warnings.filterwarnings(action = 'ignore',
                            message='.*regex module is deprecated',
                            category=DeprecationWarning,
                            module = '__main__')

</div>

</div>

This adds a filter that will apply only to warnings of the class <span class="pre">`DeprecationWarning`</span> triggered in the <a href="../library/__main__.html#module-__main__" class="reference internal" title="__main__: The environment where the top-level script is run."><span class="pre"><code class="sourceCode python">__main__</code></span></a> module, and applies a regular expression to only match the message about the <span class="pre">`regex`</span> module being deprecated, and will cause such warnings to be ignored. Warnings can also be printed only once, printed every time the offending code is executed, or turned into exceptions that will cause the program to stop (unless the exceptions are caught in the usual way, of course).

Functions were also added to Python’s C API for issuing warnings; refer to PEP 230 or to Python’s API documentation for the details.

<div class="admonition seealso">

See also

<span id="index-3" class="target"></span><a href="https://www.python.org/dev/peps/pep-0005" class="pep reference external"><strong>PEP 5</strong></a> - Guidelines for Language Evolution  
Written by Paul Prescod, to specify procedures to be followed when removing old features from Python. The policy described in this PEP hasn’t been officially adopted, but the eventual policy probably won’t be too different from Prescod’s proposal.

<span id="index-4" class="target"></span><a href="https://www.python.org/dev/peps/pep-0230" class="pep reference external"><strong>PEP 230</strong></a> - Warning Framework  
Written and implemented by Guido van Rossum.

</div>

</div>

<div id="pep-229-new-build-system" class="section">

## PEP 229: New Build System<a href="#pep-229-new-build-system" class="headerlink" title="Permalink to this headline">¶</a>

When compiling Python, the user had to go in and edit the <span class="pre">`Modules/Setup`</span> file in order to enable various additional modules; the default set is relatively small and limited to modules that compile on most Unix platforms. This means that on Unix platforms with many more features, most notably Linux, Python installations often don’t contain all useful modules they could.

Python 2.0 added the Distutils, a set of modules for distributing and installing extensions. In Python 2.1, the Distutils are used to compile much of the standard library of extension modules, autodetecting which ones are supported on the current machine. It’s hoped that this will make Python installations easier and more featureful.

Instead of having to edit the <span class="pre">`Modules/Setup`</span> file in order to enable modules, a <span class="pre">`setup.py`</span> script in the top directory of the Python source distribution is run at build time, and attempts to discover which modules can be enabled by examining the modules and header files on the system. If a module is configured in <span class="pre">`Modules/Setup`</span>, the <span class="pre">`setup.py`</span> script won’t attempt to compile that module and will defer to the <span class="pre">`Modules/Setup`</span> file’s contents. This provides a way to specific any strange command-line flags or libraries that are required for a specific platform.

In another far-reaching change to the build mechanism, Neil Schemenauer restructured things so Python now uses a single makefile that isn’t recursive, instead of makefiles in the top directory and in each of the <span class="pre">`Python/`</span>, <span class="pre">`Parser/`</span>, <span class="pre">`Objects/`</span>, and <span class="pre">`Modules/`</span> subdirectories. This makes building Python faster and also makes hacking the Makefiles clearer and simpler.

<div class="admonition seealso">

See also

<span id="index-5" class="target"></span><a href="https://www.python.org/dev/peps/pep-0229" class="pep reference external"><strong>PEP 229</strong></a> - Using Distutils to Build Python  
Written and implemented by A.M. Kuchling.

</div>

</div>

<div id="pep-205-weak-references" class="section">

## PEP 205: Weak References<a href="#pep-205-weak-references" class="headerlink" title="Permalink to this headline">¶</a>

Weak references, available through the <a href="../library/weakref.html#module-weakref" class="reference internal" title="weakref: Support for weak references and weak dictionaries."><span class="pre"><code class="sourceCode python">weakref</code></span></a> module, are a minor but useful new data type in the Python programmer’s toolbox.

Storing a reference to an object (say, in a dictionary or a list) has the side effect of keeping that object alive forever. There are a few specific cases where this behaviour is undesirable, object caches being the most common one, and another being circular references in data structures such as trees.

For example, consider a memoizing function that caches the results of another function <span class="pre">`f(x)`</span> by storing the function’s argument and its result in a dictionary:

<div class="highlight-default notranslate">

<div class="highlight">

    _cache = {}
    def memoize(x):
        if _cache.has_key(x):
            return _cache[x]

        retval = f(x)

        # Cache the returned object
        _cache[x] = retval

        return retval

</div>

</div>

This version works for simple things such as integers, but it has a side effect; the <span class="pre">`_cache`</span> dictionary holds a reference to the return values, so they’ll never be deallocated until the Python process exits and cleans up This isn’t very noticeable for integers, but if <span class="pre">`f()`</span> returns an object, or a data structure that takes up a lot of memory, this can be a problem.

Weak references provide a way to implement a cache that won’t keep objects alive beyond their time. If an object is only accessible through weak references, the object will be deallocated and the weak references will now indicate that the object it referred to no longer exists. A weak reference to an object *obj* is created by calling <span class="pre">`wr`</span>` `<span class="pre">`=`</span>` `<span class="pre">`weakref.ref(obj)`</span>. The object being referred to is returned by calling the weak reference as if it were a function: <span class="pre">`wr()`</span>. It will return the referenced object, or <span class="pre">`None`</span> if the object no longer exists.

This makes it possible to write a <span class="pre">`memoize()`</span> function whose cache doesn’t keep objects alive, by storing weak references in the cache.

<div class="highlight-default notranslate">

<div class="highlight">

    _cache = {}
    def memoize(x):
        if _cache.has_key(x):
            obj = _cache[x]()
            # If weak reference object still exists,
            # return it
            if obj is not None: return obj

        retval = f(x)

        # Cache a weak reference
        _cache[x] = weakref.ref(retval)

        return retval

</div>

</div>

The <a href="../library/weakref.html#module-weakref" class="reference internal" title="weakref: Support for weak references and weak dictionaries."><span class="pre"><code class="sourceCode python">weakref</code></span></a> module also allows creating proxy objects which behave like weak references — an object referenced only by proxy objects is deallocated – but instead of requiring an explicit call to retrieve the object, the proxy transparently forwards all operations to the object as long as the object still exists. If the object is deallocated, attempting to use a proxy will cause a <a href="../library/weakref.html#weakref.ReferenceError" class="reference internal" title="weakref.ReferenceError"><span class="pre"><code class="sourceCode python">weakref.<span class="pp">ReferenceError</span></code></span></a> exception to be raised.

<div class="highlight-default notranslate">

<div class="highlight">

    proxy = weakref.proxy(obj)
    proxy.attr   # Equivalent to obj.attr
    proxy.meth() # Equivalent to obj.meth()
    del obj
    proxy.attr   # raises weakref.ReferenceError

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-6" class="target"></span><a href="https://www.python.org/dev/peps/pep-0205" class="pep reference external"><strong>PEP 205</strong></a> - Weak References  
Written and implemented by Fred L. Drake, Jr.

</div>

</div>

<div id="pep-232-function-attributes" class="section">

## PEP 232: Function Attributes<a href="#pep-232-function-attributes" class="headerlink" title="Permalink to this headline">¶</a>

In Python 2.1, functions can now have arbitrary information attached to them. People were often using docstrings to hold information about functions and methods, because the <span class="pre">`__doc__`</span> attribute was the only way of attaching any information to a function. For example, in the Zope Web application server, functions are marked as safe for public access by having a docstring, and in John Aycock’s SPARK parsing framework, docstrings hold parts of the BNF grammar to be parsed. This overloading is unfortunate, since docstrings are really intended to hold a function’s documentation; for example, it means you can’t properly document functions intended for private use in Zope.

Arbitrary attributes can now be set and retrieved on functions using the regular Python syntax:

<div class="highlight-default notranslate">

<div class="highlight">

    def f(): pass

    f.publish = 1
    f.secure = 1
    f.grammar = "A ::= B (C D)*"

</div>

</div>

The dictionary containing attributes can be accessed as the function’s <a href="../library/stdtypes.html#object.__dict__" class="reference internal" title="object.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a>. Unlike the <a href="../library/stdtypes.html#object.__dict__" class="reference internal" title="object.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a> attribute of class instances, in functions you can actually assign a new dictionary to <a href="../library/stdtypes.html#object.__dict__" class="reference internal" title="object.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a>, though the new value is restricted to a regular Python dictionary; you *can’t* be tricky and set it to a <a href="../library/userdict.html#UserDict.UserDict" class="reference internal" title="UserDict.UserDict"><span class="pre"><code class="sourceCode python">UserDict</code></span></a> instance, or any other random object that behaves like a mapping.

<div class="admonition seealso">

See also

<span id="index-7" class="target"></span><a href="https://www.python.org/dev/peps/pep-0232" class="pep reference external"><strong>PEP 232</strong></a> - Function Attributes  
Written and implemented by Barry Warsaw.

</div>

</div>

<div id="pep-235-importing-modules-on-case-insensitive-platforms" class="section">

## PEP 235: Importing Modules on Case-Insensitive Platforms<a href="#pep-235-importing-modules-on-case-insensitive-platforms" class="headerlink" title="Permalink to this headline">¶</a>

Some operating systems have filesystems that are case-insensitive, MacOS and Windows being the primary examples; on these systems, it’s impossible to distinguish the filenames <span class="pre">`FILE.PY`</span> and <span class="pre">`file.py`</span>, even though they do store the file’s name in its original case (they’re case-preserving, too).

In Python 2.1, the <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statement will work to simulate case-sensitivity on case-insensitive platforms. Python will now search for the first case-sensitive match by default, raising an <a href="../library/exceptions.html#exceptions.ImportError" class="reference internal" title="exceptions.ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> if no such file is found, so <span class="pre">`import`</span>` `<span class="pre">`file`</span> will not import a module named <span class="pre">`FILE.PY`</span>. Case-insensitive matching can be requested by setting the <span id="index-8" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONCASEOK" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONCASEOK</code></span></a> environment variable before starting the Python interpreter.

</div>

<div id="pep-217-interactive-display-hook" class="section">

## PEP 217: Interactive Display Hook<a href="#pep-217-interactive-display-hook" class="headerlink" title="Permalink to this headline">¶</a>

When using the Python interpreter interactively, the output of commands is displayed using the built-in <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> function. In Python 2.1, the variable <a href="../library/sys.html#sys.displayhook" class="reference internal" title="sys.displayhook"><span class="pre"><code class="sourceCode python">sys.displayhook()</code></span></a> can be set to a callable object which will be called instead of <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a>. For example, you can set it to a special pretty-printing function:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> # Create a recursive data structure
    ... L = [1,2,3]
    >>> L.append(L)
    >>> L # Show Python's default output
    [1, 2, 3, [...]]
    >>> # Use pprint.pprint() as the display function
    ... import sys, pprint
    >>> sys.displayhook = pprint.pprint
    >>> L
    [1, 2, 3,  <Recursion on list with id=135143996>]
    >>>

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-9" class="target"></span><a href="https://www.python.org/dev/peps/pep-0217" class="pep reference external"><strong>PEP 217</strong></a> - Display Hook for Interactive Use  
Written and implemented by Moshe Zadka.

</div>

</div>

<div id="pep-208-new-coercion-model" class="section">

## PEP 208: New Coercion Model<a href="#pep-208-new-coercion-model" class="headerlink" title="Permalink to this headline">¶</a>

How numeric coercion is done at the C level was significantly modified. This will only affect the authors of C extensions to Python, allowing them more flexibility in writing extension types that support numeric operations.

Extension types can now set the type flag <span class="pre">`Py_TPFLAGS_CHECKTYPES`</span> in their <span class="pre">`PyTypeObject`</span> structure to indicate that they support the new coercion model. In such extension types, the numeric slot functions can no longer assume that they’ll be passed two arguments of the same type; instead they may be passed two arguments of differing types, and can then perform their own internal coercion. If the slot function is passed a type it can’t handle, it can indicate the failure by returning a reference to the <span class="pre">`Py_NotImplemented`</span> singleton value. The numeric functions of the other type will then be tried, and perhaps they can handle the operation; if the other type also returns <span class="pre">`Py_NotImplemented`</span>, then a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> will be raised. Numeric methods written in Python can also return <span class="pre">`Py_NotImplemented`</span>, causing the interpreter to act as if the method did not exist (perhaps raising a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>, perhaps trying another object’s numeric methods).

<div class="admonition seealso">

See also

<span id="index-10" class="target"></span><a href="https://www.python.org/dev/peps/pep-0208" class="pep reference external"><strong>PEP 208</strong></a> - Reworking the Coercion Model  
Written and implemented by Neil Schemenauer, heavily based upon earlier work by Marc-André Lemburg. Read this to understand the fine points of how numeric operations will now be processed at the C level.

</div>

</div>

<div id="pep-241-metadata-in-python-packages" class="section">

## PEP 241: Metadata in Python Packages<a href="#pep-241-metadata-in-python-packages" class="headerlink" title="Permalink to this headline">¶</a>

A common complaint from Python users is that there’s no single catalog of all the Python modules in existence. T. Middleton’s Vaults of Parnassus at <a href="http://www.vex.net/parnassus/" class="reference external">http://www.vex.net/parnassus/</a> are the largest catalog of Python modules, but registering software at the Vaults is optional, and many people don’t bother.

As a first small step toward fixing the problem, Python software packaged using the Distutils **sdist** command will include a file named <span class="pre">`PKG-INFO`</span> containing information about the package such as its name, version, and author (metadata, in cataloguing terminology). PEP 241 contains the full list of fields that can be present in the <span class="pre">`PKG-INFO`</span> file. As people began to package their software using Python 2.1, more and more packages will include metadata, making it possible to build automated cataloguing systems and experiment with them. With the result experience, perhaps it’ll be possible to design a really good catalog and then build support for it into Python 2.2. For example, the Distutils **sdist** and **bdist\_\*** commands could support an <span class="pre">`upload`</span> option that would automatically upload your package to a catalog server.

You can start creating packages containing <span class="pre">`PKG-INFO`</span> even if you’re not using Python 2.1, since a new release of the Distutils will be made for users of earlier Python versions. Version 1.0.2 of the Distutils includes the changes described in PEP 241, as well as various bugfixes and enhancements. It will be available from the Distutils SIG at <a href="https://www.python.org/community/sigs/current/distutils-sig/" class="reference external">https://www.python.org/community/sigs/current/distutils-sig/</a>.

<div class="admonition seealso">

See also

<span id="index-11" class="target"></span><a href="https://www.python.org/dev/peps/pep-0241" class="pep reference external"><strong>PEP 241</strong></a> - Metadata for Python Software Packages  
Written and implemented by A.M. Kuchling.

<span id="index-12" class="target"></span><a href="https://www.python.org/dev/peps/pep-0243" class="pep reference external"><strong>PEP 243</strong></a> - Module Repository Upload Mechanism  
Written by Sean Reifschneider, this draft PEP describes a proposed mechanism for uploading Python packages to a central server.

</div>

</div>

<div id="new-and-improved-modules" class="section">

## New and Improved Modules<a href="#new-and-improved-modules" class="headerlink" title="Permalink to this headline">¶</a>

- Ka-Ping Yee contributed two new modules: <span class="pre">`inspect.py`</span>, a module for getting information about live Python code, and <span class="pre">`pydoc.py`</span>, a module for interactively converting docstrings to HTML or text. As a bonus, <span class="pre">`Tools/scripts/pydoc`</span>, which is now automatically installed, uses <span class="pre">`pydoc.py`</span> to display documentation given a Python module, package, or class name. For example, <span class="pre">`pydoc`</span>` `<span class="pre">`xml.dom`</span> displays the following:

  <div class="highlight-default notranslate">

  <div class="highlight">

      Python Library Documentation: package xml.dom in xml

      NAME
          xml.dom - W3C Document Object Model implementation for Python.

      FILE
          /usr/local/lib/python2.1/xml/dom/__init__.pyc

      DESCRIPTION
          The Python mapping of the Document Object Model is documented in the
          Python Library Reference in the section on the xml.dom package.

          This package contains the following modules:
            ...

  </div>

  </div>

  <span class="pre">`pydoc`</span> also includes a Tk-based interactive help browser. <span class="pre">`pydoc`</span> quickly becomes addictive; try it out!

- Two different modules for unit testing were added to the standard library. The <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> module, contributed by Tim Peters, provides a testing framework based on running embedded examples in docstrings and comparing the results against the expected output. PyUnit, contributed by Steve Purcell, is a unit testing framework inspired by JUnit, which was in turn an adaptation of Kent Beck’s Smalltalk testing framework. See <a href="http://pyunit.sourceforge.net/" class="reference external">http://pyunit.sourceforge.net/</a> for more information about PyUnit.

- The <a href="../library/difflib.html#module-difflib" class="reference internal" title="difflib: Helpers for computing differences between objects."><span class="pre"><code class="sourceCode python">difflib</code></span></a> module contains a class, <span class="pre">`SequenceMatcher`</span>, which compares two sequences and computes the changes required to transform one sequence into the other. For example, this module can be used to write a tool similar to the Unix **diff** program, and in fact the sample program <span class="pre">`Tools/scripts/ndiff.py`</span> demonstrates how to write such a script.

- <a href="../library/curses.panel.html#module-curses.panel" class="reference internal" title="curses.panel: A panel stack extension that adds depth to  curses windows."><span class="pre"><code class="sourceCode python">curses.panel</code></span></a>, a wrapper for the panel library, part of ncurses and of SYSV curses, was contributed by Thomas Gellekum. The panel library provides windows with the additional feature of depth. Windows can be moved higher or lower in the depth ordering, and the panel library figures out where panels overlap and which sections are visible.

- The PyXML package has gone through a few releases since Python 2.0, and Python 2.1 includes an updated version of the <a href="../library/xml.html#module-xml" class="reference internal" title="xml: Package containing XML processing modules"><span class="pre"><code class="sourceCode python">xml</code></span></a> package. Some of the noteworthy changes include support for Expat 1.2 and later versions, the ability for Expat parsers to handle files in any encoding supported by Python, and various bugfixes for SAX, DOM, and the <span class="pre">`minidom`</span> module.

- Ping also contributed another hook for handling uncaught exceptions. <a href="../library/sys.html#sys.excepthook" class="reference internal" title="sys.excepthook"><span class="pre"><code class="sourceCode python">sys.excepthook()</code></span></a> can be set to a callable object. When an exception isn’t caught by any <a href="../reference/compound_stmts.html#try" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">try</code></span></a>…<a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> blocks, the exception will be passed to <a href="../library/sys.html#sys.excepthook" class="reference internal" title="sys.excepthook"><span class="pre"><code class="sourceCode python">sys.excepthook()</code></span></a>, which can then do whatever it likes. At the Ninth Python Conference, Ping demonstrated an application for this hook: printing an extended traceback that not only lists the stack frames, but also lists the function arguments and the local variables for each frame.

- Various functions in the <a href="../library/time.html#module-time" class="reference internal" title="time: Time access and conversions."><span class="pre"><code class="sourceCode python">time</code></span></a> module, such as <span class="pre">`asctime()`</span> and <span class="pre">`localtime()`</span>, require a floating point argument containing the time in seconds since the epoch. The most common use of these functions is to work with the current time, so the floating point argument has been made optional; when a value isn’t provided, the current time will be used. For example, log file entries usually need a string containing the current time; in Python 2.1, <span class="pre">`time.asctime()`</span> can be used, instead of the lengthier <span class="pre">`time.asctime(time.localtime(time.time()))`</span> that was previously required.

  This change was proposed and implemented by Thomas Wouters.

- The <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a> module now defaults to retrieving files in passive mode, because passive mode is more likely to work from behind a firewall. This request came from the Debian bug tracking system, since other Debian packages use <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a> to retrieve files and then don’t work from behind a firewall. It’s deemed unlikely that this will cause problems for anyone, because Netscape defaults to passive mode and few people complain, but if passive mode is unsuitable for your application or network setup, call <span class="pre">`set_pasv(0)`</span> on FTP objects to disable passive mode.

- Support for raw socket access has been added to the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module, contributed by Grant Edwards.

- The <a href="../library/profile.html#module-pstats" class="reference internal" title="pstats: Statistics object for use with the profiler."><span class="pre"><code class="sourceCode python">pstats</code></span></a> module now contains a simple interactive statistics browser for displaying timing profiles for Python programs, invoked when the module is run as a script. Contributed by Eric S. Raymond.

- A new implementation-dependent function, <span class="pre">`sys._getframe([depth])`</span>, has been added to return a given frame object from the current call stack. <a href="../library/sys.html#sys._getframe" class="reference internal" title="sys._getframe"><span class="pre"><code class="sourceCode python">sys._getframe()</code></span></a> returns the frame at the top of the call stack; if the optional integer argument *depth* is supplied, the function returns the frame that is *depth* calls below the top of the stack. For example, <span class="pre">`sys._getframe(1)`</span> returns the caller’s frame object.

  This function is only present in CPython, not in Jython or the .NET implementation. Use it for debugging, and resist the temptation to put it into production code.

</div>

<div id="other-changes-and-fixes" class="section">

## Other Changes and Fixes<a href="#other-changes-and-fixes" class="headerlink" title="Permalink to this headline">¶</a>

There were relatively few smaller changes made in Python 2.1 due to the shorter release cycle. A search through the CVS change logs turns up 117 patches applied, and 136 bugs fixed; both figures are likely to be underestimates. Some of the more notable changes are:

- A specialized object allocator is now optionally available, that should be faster than the system <span class="pre">`malloc()`</span> and have less memory overhead. The allocator uses C’s <span class="pre">`malloc()`</span> function to get large pools of memory, and then fulfills smaller memory requests from these pools. It can be enabled by providing the <span class="pre">`--with-pymalloc`</span> option to the **configure** script; see <span class="pre">`Objects/obmalloc.c`</span> for the implementation details.

  Authors of C extension modules should test their code with the object allocator enabled, because some incorrect code may break, causing core dumps at runtime. There are a bunch of memory allocation functions in Python’s C API that have previously been just aliases for the C library’s <span class="pre">`malloc()`</span> and <span class="pre">`free()`</span>, meaning that if you accidentally called mismatched functions, the error wouldn’t be noticeable. When the object allocator is enabled, these functions aren’t aliases of <span class="pre">`malloc()`</span> and <span class="pre">`free()`</span> any more, and calling the wrong function to free memory will get you a core dump. For example, if memory was allocated using <span class="pre">`PyMem_New()`</span>, it has to be freed using <span class="pre">`PyMem_Del()`</span>, not <span class="pre">`free()`</span>. A few modules included with Python fell afoul of this and had to be fixed; doubtless there are more third-party modules that will have the same problem.

  The object allocator was contributed by Vladimir Marangozov.

- The speed of line-oriented file I/O has been improved because people often complain about its lack of speed, and because it’s often been used as a naïve benchmark. The <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline()</code></span></a> method of file objects has therefore been rewritten to be much faster. The exact amount of the speedup will vary from platform to platform depending on how slow the C library’s <span class="pre">`getc()`</span> was, but is around 66%, and potentially much faster on some particular operating systems. Tim Peters did much of the benchmarking and coding for this change, motivated by a discussion in comp.lang.python.

  A new module and method for file objects was also added, contributed by Jeff Epler. The new method, <span class="pre">`xreadlines()`</span>, is similar to the existing <a href="../library/functions.html#xrange" class="reference internal" title="xrange"><span class="pre"><code class="sourceCode python"><span class="bu">xrange</span>()</code></span></a> built-in. <span class="pre">`xreadlines()`</span> returns an opaque sequence object that only supports being iterated over, reading a line on every iteration but not reading the entire file into memory as the existing <span class="pre">`readlines()`</span> method does. You’d use it like this:

  <div class="highlight-default notranslate">

  <div class="highlight">

      for line in sys.stdin.xreadlines():
          # ... do something for each line ...
          ...

  </div>

  </div>

  For a fuller discussion of the line I/O changes, see the python-dev summary for January 1–15, 2001 at <a href="https://mail.python.org/pipermail/python-dev/2001-January/" class="reference external">https://mail.python.org/pipermail/python-dev/2001-January/</a>.

- A new method, <span class="pre">`popitem()`</span>, was added to dictionaries to enable destructively iterating through the contents of a dictionary; this can be faster for large dictionaries because there’s no need to construct a list containing all the keys or values. <span class="pre">`D.popitem()`</span> removes a random <span class="pre">`(key,`</span>` `<span class="pre">`value)`</span> pair from the dictionary <span class="pre">`D`</span> and returns it as a 2-tuple. This was implemented mostly by Tim Peters and Guido van Rossum, after a suggestion and preliminary patch by Moshe Zadka.

- Modules can now control which names are imported when <span class="pre">`from`</span>` `<span class="pre">`module`</span>` `<span class="pre">`import`</span>` `<span class="pre">`*`</span> is used, by defining an <span class="pre">`__all__`</span> attribute containing a list of names that will be imported. One common complaint is that if the module imports other modules such as <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> or <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a>, <span class="pre">`from`</span>` `<span class="pre">`module`</span>` `<span class="pre">`import`</span>` `<span class="pre">`*`</span> will add them to the importing module’s namespace. To fix this, simply list the public names in <span class="pre">`__all__`</span>:

  <div class="highlight-default notranslate">

  <div class="highlight">

      # List public names
      __all__ = ['Database', 'open']

  </div>

  </div>

  A stricter version of this patch was first suggested and implemented by Ben Wolfson, but after some python-dev discussion, a weaker final version was checked in.

- Applying <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> to strings previously used octal escapes for non-printable characters; for example, a newline was <span class="pre">`'\012'`</span>. This was a vestigial trace of Python’s C ancestry, but today octal is of very little practical use. Ka-Ping Yee suggested using hex escapes instead of octal ones, and using the <span class="pre">`\n`</span>, <span class="pre">`\t`</span>, <span class="pre">`\r`</span> escapes for the appropriate characters, and implemented this new formatting.

- Syntax errors detected at compile-time can now raise exceptions containing the filename and line number of the error, a pleasant side effect of the compiler reorganization done by Jeremy Hylton.

- C extensions which import other modules have been changed to use <span class="pre">`PyImport_ImportModule()`</span>, which means that they will use any import hooks that have been installed. This is also encouraged for third-party extensions that need to import some other module from C code.

- The size of the Unicode character database was shrunk by another 340K thanks to Fredrik Lundh.

- Some new ports were contributed: MacOS X (by Steven Majewski), Cygwin (by Jason Tishler); RISCOS (by Dietmar Schwertberger); Unixware 7 (by Billy G. Allie).

And there’s the usual list of minor bugfixes, minor memory leaks, docstring edits, and other tweaks, too lengthy to be worth itemizing; see the CVS logs for the full details if you want them.

</div>

<div id="acknowledgements" class="section">

## Acknowledgements<a href="#acknowledgements" class="headerlink" title="Permalink to this headline">¶</a>

The author would like to thank the following people for offering suggestions on various drafts of this article: Graeme Cross, David Goodger, Jay Graves, Michael Hudson, Marc-André Lemburg, Fredrik Lundh, Neil Schemenauer, Thomas Wouters.

</div>

</div>

</div>
