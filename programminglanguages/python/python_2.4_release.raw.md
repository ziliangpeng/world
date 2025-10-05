<div class="body" role="main">

<div id="what-s-new-in-python-2-4" class="section">

# What’s New in Python 2.4<a href="#what-s-new-in-python-2-4" class="headerlink" title="Permalink to this headline">¶</a>

Author  
A.M. Kuchling

This article explains the new features in Python 2.4.1, released on March 30, 2005.

Python 2.4 is a medium-sized release. It doesn’t introduce as many changes as the radical Python 2.2, but introduces more features than the conservative 2.3 release. The most significant new language features are function decorators and generator expressions; most other changes are to the standard library.

According to the CVS change logs, there were 481 patches applied and 502 bugs fixed between Python 2.3 and 2.4. Both figures are likely to be underestimates.

This article doesn’t attempt to provide a complete specification of every single new feature, but instead provides a brief introduction to each feature. For full details, you should refer to the documentation for Python 2.4, such as the Python Library Reference and the Python Reference Manual. Often you will be referred to the PEP for a particular new feature for explanations of the implementation and design rationale.

<div id="pep-218-built-in-set-objects" class="section">

## PEP 218: Built-In Set Objects<a href="#pep-218-built-in-set-objects" class="headerlink" title="Permalink to this headline">¶</a>

Python 2.3 introduced the <a href="../library/sets.html#module-sets" class="reference internal" title="sets: Implementation of sets of unique elements. (deprecated)"><span class="pre"><code class="sourceCode python">sets</code></span></a> module. C implementations of set data types have now been added to the Python core as two new built-in types, <span class="pre">`set(iterable)`</span> and <span class="pre">`frozenset(iterable)`</span>. They provide high speed operations for membership testing, for eliminating duplicates from sequences, and for mathematical operations like unions, intersections, differences, and symmetric differences.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> a = set('abracadabra')              # form a set from a string
    >>> 'z' in a                            # fast membership testing
    False
    >>> a                                   # unique letters in a
    set(['a', 'r', 'b', 'c', 'd'])
    >>> ''.join(a)                          # convert back into a string
    'arbcd'

    >>> b = set('alacazam')                 # form a second set
    >>> a - b                               # letters in a but not in b
    set(['r', 'd', 'b'])
    >>> a | b                               # letters in either a or b
    set(['a', 'c', 'r', 'd', 'b', 'm', 'z', 'l'])
    >>> a & b                               # letters in both a and b
    set(['a', 'c'])
    >>> a ^ b                               # letters in a or b but not both
    set(['r', 'd', 'b', 'm', 'z', 'l'])

    >>> a.add('z')                          # add a new element
    >>> a.update('wxy')                     # add multiple new elements
    >>> a
    set(['a', 'c', 'b', 'd', 'r', 'w', 'y', 'x', 'z'])
    >>> a.remove('x')                       # take one element out
    >>> a
    set(['a', 'c', 'b', 'd', 'r', 'w', 'y', 'z'])

</div>

</div>

The <a href="../library/stdtypes.html#frozenset" class="reference internal" title="frozenset"><span class="pre"><code class="sourceCode python"><span class="bu">frozenset</span>()</code></span></a> type is an immutable version of <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a>. Since it is immutable and hashable, it may be used as a dictionary key or as a member of another set.

The <a href="../library/sets.html#module-sets" class="reference internal" title="sets: Implementation of sets of unique elements. (deprecated)"><span class="pre"><code class="sourceCode python">sets</code></span></a> module remains in the standard library, and may be useful if you wish to subclass the <span class="pre">`Set`</span> or <span class="pre">`ImmutableSet`</span> classes. There are currently no plans to deprecate the module.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://www.python.org/dev/peps/pep-0218" class="pep reference external"><strong>PEP 218</strong></a> - Adding a Built-In Set Object Type  
Originally proposed by Greg Wilson and ultimately implemented by Raymond Hettinger.

</div>

</div>

<div id="pep-237-unifying-long-integers-and-integers" class="section">

## PEP 237: Unifying Long Integers and Integers<a href="#pep-237-unifying-long-integers-and-integers" class="headerlink" title="Permalink to this headline">¶</a>

The lengthy transition process for this PEP, begun in Python 2.2, takes another step forward in Python 2.4. In 2.3, certain integer operations that would behave differently after int/long unification triggered <a href="../library/exceptions.html#exceptions.FutureWarning" class="reference internal" title="exceptions.FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a> warnings and returned values limited to 32 or 64 bits (depending on your platform). In 2.4, these expressions no longer produce a warning and instead produce a different result that’s usually a long integer.

The problematic expressions are primarily left shifts and lengthy hexadecimal and octal constants. For example, <span class="pre">`2`</span>` `<span class="pre">`<<`</span>` `<span class="pre">`32`</span> results in a warning in 2.3, evaluating to 0 on 32-bit platforms. In Python 2.4, this expression now returns the correct answer, 8589934592.

<div class="admonition seealso">

See also

<span id="index-1" class="target"></span><a href="https://www.python.org/dev/peps/pep-0237" class="pep reference external"><strong>PEP 237</strong></a> - Unifying Long Integers and Integers  
Original PEP written by Moshe Zadka and GvR. The changes for 2.4 were implemented by Kalle Svensson.

</div>

</div>

<div id="pep-289-generator-expressions" class="section">

## PEP 289: Generator Expressions<a href="#pep-289-generator-expressions" class="headerlink" title="Permalink to this headline">¶</a>

The iterator feature introduced in Python 2.2 and the <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> module make it easier to write programs that loop through large data sets without having the entire data set in memory at one time. List comprehensions don’t fit into this picture very well because they produce a Python list object containing all of the items. This unavoidably pulls all of the objects into memory, which can be a problem if your data set is very large. When trying to write a functionally-styled program, it would be natural to write something like:

<div class="highlight-default notranslate">

<div class="highlight">

    links = [link for link in get_all_links() if not link.followed]
    for link in links:
        ...

</div>

</div>

instead of

<div class="highlight-default notranslate">

<div class="highlight">

    for link in get_all_links():
        if link.followed:
            continue
        ...

</div>

</div>

The first form is more concise and perhaps more readable, but if you’re dealing with a large number of link objects you’d have to write the second form to avoid having all link objects in memory at the same time.

Generator expressions work similarly to list comprehensions but don’t materialize the entire list; instead they create a generator that will return elements one by one. The above example could be written as:

<div class="highlight-default notranslate">

<div class="highlight">

    links = (link for link in get_all_links() if not link.followed)
    for link in links:
        ...

</div>

</div>

Generator expressions always have to be written inside parentheses, as in the above example. The parentheses signalling a function call also count, so if you want to create an iterator that will be immediately passed to a function you could write:

<div class="highlight-default notranslate">

<div class="highlight">

    print sum(obj.count for obj in list_all_objects())

</div>

</div>

Generator expressions differ from list comprehensions in various small ways. Most notably, the loop variable (*obj* in the above example) is not accessible outside of the generator expression. List comprehensions leave the variable assigned to its last value; future versions of Python will change this, making list comprehensions match generator expressions in this respect.

<div class="admonition seealso">

See also

<span id="index-2" class="target"></span><a href="https://www.python.org/dev/peps/pep-0289" class="pep reference external"><strong>PEP 289</strong></a> - Generator Expressions  
Proposed by Raymond Hettinger and implemented by Jiwon Seo with early efforts steered by Hye-Shik Chang.

</div>

</div>

<div id="pep-292-simpler-string-substitutions" class="section">

## PEP 292: Simpler String Substitutions<a href="#pep-292-simpler-string-substitutions" class="headerlink" title="Permalink to this headline">¶</a>

Some new classes in the standard library provide an alternative mechanism for substituting variables into strings; this style of substitution may be better for applications where untrained users need to edit templates.

The usual way of substituting variables by name is the <span class="pre">`%`</span> operator:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> '%(page)i: %(title)s' % {'page':2, 'title': 'The Best of Times'}
    '2: The Best of Times'

</div>

</div>

When writing the template string, it can be easy to forget the <span class="pre">`i`</span> or <span class="pre">`s`</span> after the closing parenthesis. This isn’t a big problem if the template is in a Python module, because you run the code, get an “Unsupported format character” <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a>, and fix the problem. However, consider an application such as Mailman where template strings or translations are being edited by users who aren’t aware of the Python language. The format string’s syntax is complicated to explain to such users, and if they make a mistake, it’s difficult to provide helpful feedback to them.

PEP 292 adds a <span class="pre">`Template`</span> class to the <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module that uses <span class="pre">`$`</span> to indicate a substitution:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> import string
    >>> t = string.Template('$page: $title')
    >>> t.substitute({'page':2, 'title': 'The Best of Times'})
    '2: The Best of Times'

</div>

</div>

If a key is missing from the dictionary, the <span class="pre">`substitute()`</span> method will raise a <a href="../library/exceptions.html#exceptions.KeyError" class="reference internal" title="exceptions.KeyError"><span class="pre"><code class="sourceCode python"><span class="pp">KeyError</span></code></span></a>. There’s also a <span class="pre">`safe_substitute()`</span> method that ignores missing keys:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> t = string.Template('$page: $title')
    >>> t.safe_substitute({'page':3})
    '3: $title'

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-3" class="target"></span><a href="https://www.python.org/dev/peps/pep-0292" class="pep reference external"><strong>PEP 292</strong></a> - Simpler String Substitutions  
Written and implemented by Barry Warsaw.

</div>

</div>

<div id="pep-318-decorators-for-functions-and-methods" class="section">

## PEP 318: Decorators for Functions and Methods<a href="#pep-318-decorators-for-functions-and-methods" class="headerlink" title="Permalink to this headline">¶</a>

Python 2.2 extended Python’s object model by adding static methods and class methods, but it didn’t extend Python’s syntax to provide any new way of defining static or class methods. Instead, you had to write a <a href="../reference/compound_stmts.html#def" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">def</code></span></a> statement in the usual way, and pass the resulting method to a <a href="../library/functions.html#staticmethod" class="reference internal" title="staticmethod"><span class="pre"><code class="sourceCode python"><span class="bu">staticmethod</span>()</code></span></a> or <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span>()</code></span></a> function that would wrap up the function as a method of the new type. Your code would look like this:

<div class="highlight-default notranslate">

<div class="highlight">

    class C:
       def meth (cls):
           ...

       meth = classmethod(meth)   # Rebind name to wrapped-up class method

</div>

</div>

If the method was very long, it would be easy to miss or forget the <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span>()</code></span></a> invocation after the function body.

The intention was always to add some syntax to make such definitions more readable, but at the time of 2.2’s release a good syntax was not obvious. Today a good syntax *still* isn’t obvious but users are asking for easier access to the feature; a new syntactic feature has been added to meet this need.

The new feature is called “function decorators”. The name comes from the idea that <a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="bu">classmethod</span>()</code></span></a>, <a href="../library/functions.html#staticmethod" class="reference internal" title="staticmethod"><span class="pre"><code class="sourceCode python"><span class="bu">staticmethod</span>()</code></span></a>, and friends are storing additional information on a function object; they’re *decorating* functions with more details.

The notation borrows from Java and uses the <span class="pre">`'@'`</span> character as an indicator. Using the new syntax, the example above would be written:

<div class="highlight-default notranslate">

<div class="highlight">

    class C:

       @classmethod
       def meth (cls):
           ...

</div>

</div>

The <span class="pre">`@classmethod`</span> is shorthand for the <span class="pre">`meth=classmethod(meth)`</span> assignment. More generally, if you have the following:

<div class="highlight-default notranslate">

<div class="highlight">

    @A
    @B
    @C
    def f ():
        ...

</div>

</div>

It’s equivalent to the following pre-decorator code:

<div class="highlight-default notranslate">

<div class="highlight">

    def f(): ...
    f = A(B(C(f)))

</div>

</div>

Decorators must come on the line before a function definition, one decorator per line, and can’t be on the same line as the def statement, meaning that <span class="pre">`@A`</span>` `<span class="pre">`def`</span>` `<span class="pre">`f():`</span>` `<span class="pre">`...`</span> is illegal. You can only decorate function definitions, either at the module level or inside a class; you can’t decorate class definitions.

A decorator is just a function that takes the function to be decorated as an argument and returns either the same function or some new object. The return value of the decorator need not be callable (though it typically is), unless further decorators will be applied to the result. It’s easy to write your own decorators. The following simple example just sets an attribute on the function object:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> def deco(func):
    ...    func.attr = 'decorated'
    ...    return func
    ...
    >>> @deco
    ... def f(): pass
    ...
    >>> f
    <function f at 0x402ef0d4>
    >>> f.attr
    'decorated'
    >>>

</div>

</div>

As a slightly more realistic example, the following decorator checks that the supplied argument is an integer:

<div class="highlight-default notranslate">

<div class="highlight">

    def require_int (func):
        def wrapper (arg):
            assert isinstance(arg, int)
            return func(arg)

        return wrapper

    @require_int
    def p1 (arg):
        print arg

    @require_int
    def p2(arg):
        print arg*2

</div>

</div>

An example in <span id="index-4" class="target"></span><a href="https://www.python.org/dev/peps/pep-0318" class="pep reference external"><strong>PEP 318</strong></a> contains a fancier version of this idea that lets you both specify the required type and check the returned type.

Decorator functions can take arguments. If arguments are supplied, your decorator function is called with only those arguments and must return a new decorator function; this function must take a single function and return a function, as previously described. In other words, <span class="pre">`@A`</span>` `<span class="pre">`@B`</span>` `<span class="pre">`@C(args)`</span> becomes:

<div class="highlight-default notranslate">

<div class="highlight">

    def f(): ...
    _deco = C(args)
    f = A(B(_deco(f)))

</div>

</div>

Getting this right can be slightly brain-bending, but it’s not too difficult.

A small related change makes the <span class="pre">`func_name`</span> attribute of functions writable. This attribute is used to display function names in tracebacks, so decorators should change the name of any new function that’s constructed and returned.

<div class="admonition seealso">

See also

<span id="index-5" class="target"></span><a href="https://www.python.org/dev/peps/pep-0318" class="pep reference external"><strong>PEP 318</strong></a> - Decorators for Functions, Methods and Classes  
Written by Kevin D. Smith, Jim Jewett, and Skip Montanaro. Several people wrote patches implementing function decorators, but the one that was actually checked in was patch \#979728, written by Mark Russell.

<a href="https://wiki.python.org/moin/PythonDecoratorLibrary" class="reference external">https://wiki.python.org/moin/PythonDecoratorLibrary</a>  
This Wiki page contains several examples of decorators.

</div>

</div>

<div id="pep-322-reverse-iteration" class="section">

## PEP 322: Reverse Iteration<a href="#pep-322-reverse-iteration" class="headerlink" title="Permalink to this headline">¶</a>

A new built-in function, <span class="pre">`reversed(seq)`</span>, takes a sequence and returns an iterator that loops over the elements of the sequence in reverse order.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> for i in reversed(xrange(1,4)):
    ...    print i
    ...
    3
    2
    1

</div>

</div>

Compared to extended slicing, such as <span class="pre">`range(1,4)[::-1]`</span>, <a href="../library/functions.html#reversed" class="reference internal" title="reversed"><span class="pre"><code class="sourceCode python"><span class="bu">reversed</span>()</code></span></a> is easier to read, runs faster, and uses substantially less memory.

Note that <a href="../library/functions.html#reversed" class="reference internal" title="reversed"><span class="pre"><code class="sourceCode python"><span class="bu">reversed</span>()</code></span></a> only accepts sequences, not arbitrary iterators. If you want to reverse an iterator, first convert it to a list with <span class="pre">`list()`</span>.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> input = open('/etc/passwd', 'r')
    >>> for line in reversed(list(input)):
    ...   print line
    ...
    root:*:0:0:System Administrator:/var/root:/bin/tcsh
      ...

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-6" class="target"></span><a href="https://www.python.org/dev/peps/pep-0322" class="pep reference external"><strong>PEP 322</strong></a> - Reverse Iteration  
Written and implemented by Raymond Hettinger.

</div>

</div>

<div id="pep-324-new-subprocess-module" class="section">

## PEP 324: New subprocess Module<a href="#pep-324-new-subprocess-module" class="headerlink" title="Permalink to this headline">¶</a>

The standard library provides a number of ways to execute a subprocess, offering different features and different levels of complexity. <span class="pre">`os.system(command)`</span> is easy to use, but slow (it runs a shell process which executes the command) and dangerous (you have to be careful about escaping the shell’s metacharacters). The <a href="../library/popen2.html#module-popen2" class="reference internal" title="popen2: Subprocesses with accessible standard I/O streams. (deprecated)"><span class="pre"><code class="sourceCode python">popen2</code></span></a> module offers classes that can capture standard output and standard error from the subprocess, but the naming is confusing. The <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module cleans this up, providing a unified interface that offers all the features you might need.

Instead of <a href="../library/popen2.html#module-popen2" class="reference internal" title="popen2: Subprocesses with accessible standard I/O streams. (deprecated)"><span class="pre"><code class="sourceCode python">popen2</code></span></a>’s collection of classes, <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> contains a single class called <span class="pre">`Popen`</span> whose constructor supports a number of different keyword arguments.

<div class="highlight-default notranslate">

<div class="highlight">

    class Popen(args, bufsize=0, executable=None,
                stdin=None, stdout=None, stderr=None,
                preexec_fn=None, close_fds=False, shell=False,
                cwd=None, env=None, universal_newlines=False,
                startupinfo=None, creationflags=0):

</div>

</div>

*args* is commonly a sequence of strings that will be the arguments to the program executed as the subprocess. (If the *shell* argument is true, *args* can be a string which will then be passed on to the shell for interpretation, just as <a href="../library/os.html#os.system" class="reference internal" title="os.system"><span class="pre"><code class="sourceCode python">os.system()</code></span></a> does.)

*stdin*, *stdout*, and *stderr* specify what the subprocess’s input, output, and error streams will be. You can provide a file object or a file descriptor, or you can use the constant <span class="pre">`subprocess.PIPE`</span> to create a pipe between the subprocess and the parent.

The constructor has a number of handy options:

- *close_fds* requests that all file descriptors be closed before running the subprocess.

- *cwd* specifies the working directory in which the subprocess will be executed (defaulting to whatever the parent’s working directory is).

- *env* is a dictionary specifying environment variables.

- *preexec_fn* is a function that gets called before the child is started.

- *universal_newlines* opens the child’s input and output using Python’s <a href="../glossary.html#term-universal-newlines" class="reference internal"><span class="xref std std-term">universal newlines</span></a> feature.

Once you’ve created the <span class="pre">`Popen`</span> instance, you can call its <span class="pre">`wait()`</span> method to pause until the subprocess has exited, <span class="pre">`poll()`</span> to check if it’s exited without pausing, or <span class="pre">`communicate(data)`</span> to send the string *data* to the subprocess’s standard input. <span class="pre">`communicate(data)`</span> then reads any data that the subprocess has sent to its standard output or standard error, returning a tuple <span class="pre">`(stdout_data,`</span>` `<span class="pre">`stderr_data)`</span>.

<span class="pre">`call()`</span> is a shortcut that passes its arguments along to the <span class="pre">`Popen`</span> constructor, waits for the command to complete, and returns the status code of the subprocess. It can serve as a safer analog to <a href="../library/os.html#os.system" class="reference internal" title="os.system"><span class="pre"><code class="sourceCode python">os.system()</code></span></a>:

<div class="highlight-default notranslate">

<div class="highlight">

    sts = subprocess.call(['dpkg', '-i', '/tmp/new-package.deb'])
    if sts == 0:
        # Success
        ...
    else:
        # dpkg returned an error
        ...

</div>

</div>

The command is invoked without use of the shell. If you really do want to use the shell, you can add <span class="pre">`shell=True`</span> as a keyword argument and provide a string instead of a sequence:

<div class="highlight-default notranslate">

<div class="highlight">

    sts = subprocess.call('dpkg -i /tmp/new-package.deb', shell=True)

</div>

</div>

The PEP takes various examples of shell and Python code and shows how they’d be translated into Python code that uses <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a>. Reading this section of the PEP is highly recommended.

<div class="admonition seealso">

See also

<span id="index-8" class="target"></span><a href="https://www.python.org/dev/peps/pep-0324" class="pep reference external"><strong>PEP 324</strong></a> - subprocess - New process module  
Written and implemented by Peter Åstrand, with assistance from Fredrik Lundh and others.

</div>

</div>

<div id="pep-327-decimal-data-type" class="section">

## PEP 327: Decimal Data Type<a href="#pep-327-decimal-data-type" class="headerlink" title="Permalink to this headline">¶</a>

Python has always supported floating-point (FP) numbers, based on the underlying C <span class="pre">`double`</span> type, as a data type. However, while most programming languages provide a floating-point type, many people (even programmers) are unaware that floating-point numbers don’t represent certain decimal fractions accurately. The new <span class="pre">`Decimal`</span> type can represent these fractions accurately, up to a user-specified precision limit.

<div id="why-is-decimal-needed" class="section">

### Why is Decimal needed?<a href="#why-is-decimal-needed" class="headerlink" title="Permalink to this headline">¶</a>

The limitations arise from the representation used for floating-point numbers. FP numbers are made up of three components:

- The sign, which is positive or negative.

- The mantissa, which is a single-digit binary number followed by a fractional part. For example, <span class="pre">`1.01`</span> in base-2 notation is <span class="pre">`1`</span>` `<span class="pre">`+`</span>` `<span class="pre">`0/2`</span>` `<span class="pre">`+`</span>` `<span class="pre">`1/4`</span>, or 1.25 in decimal notation.

- The exponent, which tells where the decimal point is located in the number represented.

For example, the number 1.25 has positive sign, a mantissa value of 1.01 (in binary), and an exponent of 0 (the decimal point doesn’t need to be shifted). The number 5 has the same sign and mantissa, but the exponent is 2 because the mantissa is multiplied by 4 (2 to the power of the exponent 2); 1.25 \* 4 equals 5.

Modern systems usually provide floating-point support that conforms to a standard called IEEE 754. C’s <span class="pre">`double`</span> type is usually implemented as a 64-bit IEEE 754 number, which uses 52 bits of space for the mantissa. This means that numbers can only be specified to 52 bits of precision. If you’re trying to represent numbers whose expansion repeats endlessly, the expansion is cut off after 52 bits. Unfortunately, most software needs to produce output in base 10, and common fractions in base 10 are often repeating decimals in binary. For example, 1.1 decimal is binary <span class="pre">`1.0001100110011`</span>` `<span class="pre">`...`</span>; .1 = 1/16 + 1/32 + 1/256 plus an infinite number of additional terms. IEEE 754 has to chop off that infinitely repeated decimal after 52 digits, so the representation is slightly inaccurate.

Sometimes you can see this inaccuracy when the number is printed:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> 1.1
    1.1000000000000001

</div>

</div>

The inaccuracy isn’t always visible when you print the number because the FP-to-decimal-string conversion is provided by the C library, and most C libraries try to produce sensible output. Even if it’s not displayed, however, the inaccuracy is still there and subsequent operations can magnify the error.

For many applications this doesn’t matter. If I’m plotting points and displaying them on my monitor, the difference between 1.1 and 1.1000000000000001 is too small to be visible. Reports often limit output to a certain number of decimal places, and if you round the number to two or three or even eight decimal places, the error is never apparent. However, for applications where it does matter, it’s a lot of work to implement your own custom arithmetic routines.

Hence, the <span class="pre">`Decimal`</span> type was created.

</div>

<div id="the-decimal-type" class="section">

### The <span class="pre">`Decimal`</span> type<a href="#the-decimal-type" class="headerlink" title="Permalink to this headline">¶</a>

A new module, <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic  Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a>, was added to Python’s standard library. It contains two classes, <span class="pre">`Decimal`</span> and <span class="pre">`Context`</span>. <span class="pre">`Decimal`</span> instances represent numbers, and <span class="pre">`Context`</span> instances are used to wrap up various settings such as the precision and default rounding mode.

<span class="pre">`Decimal`</span> instances are immutable, like regular Python integers and FP numbers; once it’s been created, you can’t change the value an instance represents. <span class="pre">`Decimal`</span> instances can be created from integers or strings:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> import decimal
    >>> decimal.Decimal(1972)
    Decimal("1972")
    >>> decimal.Decimal("1.1")
    Decimal("1.1")

</div>

</div>

You can also provide tuples containing the sign, the mantissa represented as a tuple of decimal digits, and the exponent:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> decimal.Decimal((1, (1, 4, 7, 5), -2))
    Decimal("-14.75")

</div>

</div>

Cautionary note: the sign bit is a Boolean value, so 0 is positive and 1 is negative.

Converting from floating-point numbers poses a bit of a problem: should the FP number representing 1.1 turn into the decimal number for exactly 1.1, or for 1.1 plus whatever inaccuracies are introduced? The decision was to dodge the issue and leave such a conversion out of the API. Instead, you should convert the floating-point number into a string using the desired precision and pass the string to the <span class="pre">`Decimal`</span> constructor:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> f = 1.1
    >>> decimal.Decimal(str(f))
    Decimal("1.1")
    >>> decimal.Decimal('%.12f' % f)
    Decimal("1.100000000000")

</div>

</div>

Once you have <span class="pre">`Decimal`</span> instances, you can perform the usual mathematical operations on them. One limitation: exponentiation requires an integer exponent:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> a = decimal.Decimal('35.72')
    >>> b = decimal.Decimal('1.73')
    >>> a+b
    Decimal("37.45")
    >>> a-b
    Decimal("33.99")
    >>> a*b
    Decimal("61.7956")
    >>> a/b
    Decimal("20.64739884393063583815028902")
    >>> a ** 2
    Decimal("1275.9184")
    >>> a**b
    Traceback (most recent call last):
      ...
    decimal.InvalidOperation: x ** (non-integer)

</div>

</div>

You can combine <span class="pre">`Decimal`</span> instances with integers, but not with floating-point numbers:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> a + 4
    Decimal("39.72")
    >>> a + 4.5
    Traceback (most recent call last):
      ...
    TypeError: You can interact Decimal only with int, long or Decimal data types.
    >>>

</div>

</div>

<span class="pre">`Decimal`</span> numbers can be used with the <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> and <a href="../library/cmath.html#module-cmath" class="reference internal" title="cmath: Mathematical functions for complex numbers."><span class="pre"><code class="sourceCode python">cmath</code></span></a> modules, but note that they’ll be immediately converted to floating-point numbers before the operation is performed, resulting in a possible loss of precision and accuracy. You’ll also get back a regular floating-point number and not a <span class="pre">`Decimal`</span>.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> import math, cmath
    >>> d = decimal.Decimal('123456789012.345')
    >>> math.sqrt(d)
    351364.18288201344
    >>> cmath.sqrt(-d)
    351364.18288201344j

</div>

</div>

<span class="pre">`Decimal`</span> instances have a <span class="pre">`sqrt()`</span> method that returns a <span class="pre">`Decimal`</span>, but if you need other things such as trigonometric functions you’ll have to implement them.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> d.sqrt()
    Decimal("351364.1828820134592177245001")

</div>

</div>

</div>

<div id="the-context-type" class="section">

### The <span class="pre">`Context`</span> type<a href="#the-context-type" class="headerlink" title="Permalink to this headline">¶</a>

Instances of the <span class="pre">`Context`</span> class encapsulate several settings for decimal operations:

- <span class="pre">`prec`</span> is the precision, the number of decimal places.

- <span class="pre">`rounding`</span> specifies the rounding mode. The <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic  Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> module has constants for the various possibilities: <span class="pre">`ROUND_DOWN`</span>, <span class="pre">`ROUND_CEILING`</span>, <span class="pre">`ROUND_HALF_EVEN`</span>, and various others.

- <span class="pre">`traps`</span> is a dictionary specifying what happens on encountering certain error conditions: either an exception is raised or a value is returned. Some examples of error conditions are division by zero, loss of precision, and overflow.

There’s a thread-local default context available by calling <span class="pre">`getcontext()`</span>; you can change the properties of this context to alter the default precision, rounding, or trap handling. The following example shows the effect of changing the precision of the default context:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> decimal.getcontext().prec
    28
    >>> decimal.Decimal(1) / decimal.Decimal(7)
    Decimal("0.1428571428571428571428571429")
    >>> decimal.getcontext().prec = 9
    >>> decimal.Decimal(1) / decimal.Decimal(7)
    Decimal("0.142857143")

</div>

</div>

The default action for error conditions is selectable; the module can either return a special value such as infinity or not-a-number, or exceptions can be raised:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> decimal.Decimal(1) / decimal.Decimal(0)
    Traceback (most recent call last):
      ...
    decimal.DivisionByZero: x / 0
    >>> decimal.getcontext().traps[decimal.DivisionByZero] = False
    >>> decimal.Decimal(1) / decimal.Decimal(0)
    Decimal("Infinity")
    >>>

</div>

</div>

The <span class="pre">`Context`</span> instance also has various methods for formatting numbers such as <span class="pre">`to_eng_string()`</span> and <span class="pre">`to_sci_string()`</span>.

For more information, see the documentation for the <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic  Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> module, which includes a quick-start tutorial and a reference.

<div class="admonition seealso">

See also

<span id="index-9" class="target"></span><a href="https://www.python.org/dev/peps/pep-0327" class="pep reference external"><strong>PEP 327</strong></a> - Decimal Data Type  
Written by Facundo Batista and implemented by Facundo Batista, Eric Price, Raymond Hettinger, Aahz, and Tim Peters.

<a href="http://www.lahey.com/float.htm" class="reference external">http://www.lahey.com/float.htm</a>  
The article uses Fortran code to illustrate many of the problems that floating-point inaccuracy can cause.

<a href="http://speleotrove.com/decimal/" class="reference external">http://speleotrove.com/decimal/</a>  
A description of a decimal-based representation. This representation is being proposed as a standard, and underlies the new Python decimal type. Much of this material was written by Mike Cowlishaw, designer of the Rexx language.

</div>

</div>

</div>

<div id="pep-328-multi-line-imports" class="section">

## PEP 328: Multi-line Imports<a href="#pep-328-multi-line-imports" class="headerlink" title="Permalink to this headline">¶</a>

One language change is a small syntactic tweak aimed at making it easier to import many names from a module. In a <span class="pre">`from`</span>` `<span class="pre">`module`</span>` `<span class="pre">`import`</span>` `<span class="pre">`names`</span> statement, *names* is a sequence of names separated by commas. If the sequence is very long, you can either write multiple imports from the same module, or you can use backslashes to escape the line endings like this:

<div class="highlight-default notranslate">

<div class="highlight">

    from SimpleXMLRPCServer import SimpleXMLRPCServer,\
                SimpleXMLRPCRequestHandler,\
                CGIXMLRPCRequestHandler,\
                resolve_dotted_attribute

</div>

</div>

The syntactic change in Python 2.4 simply allows putting the names within parentheses. Python ignores newlines within a parenthesized expression, so the backslashes are no longer needed:

<div class="highlight-default notranslate">

<div class="highlight">

    from SimpleXMLRPCServer import (SimpleXMLRPCServer,
                                    SimpleXMLRPCRequestHandler,
                                    CGIXMLRPCRequestHandler,
                                    resolve_dotted_attribute)

</div>

</div>

The PEP also proposes that all <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statements be absolute imports, with a leading <span class="pre">`.`</span> character to indicate a relative import. This part of the PEP was not implemented for Python 2.4, but was completed for Python 2.5.

<div class="admonition seealso">

See also

<span id="index-10" class="target"></span><a href="https://www.python.org/dev/peps/pep-0328" class="pep reference external"><strong>PEP 328</strong></a> - Imports: Multi-Line and Absolute/Relative  
Written by Aahz. Multi-line imports were implemented by Dima Dorfman.

</div>

</div>

<div id="pep-331-locale-independent-float-string-conversions" class="section">

## PEP 331: Locale-Independent Float/String Conversions<a href="#pep-331-locale-independent-float-string-conversions" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a> modules lets Python software select various conversions and display conventions that are localized to a particular country or language. However, the module was careful to not change the numeric locale because various functions in Python’s implementation required that the numeric locale remain set to the <span class="pre">`'C'`</span> locale. Often this was because the code was using the C library’s <span class="pre">`atof()`</span> function.

Not setting the numeric locale caused trouble for extensions that used third-party C libraries, however, because they wouldn’t have the correct locale set. The motivating example was GTK+, whose user interface widgets weren’t displaying numbers in the current locale.

The solution described in the PEP is to add three new functions to the Python API that perform ASCII-only conversions, ignoring the locale setting:

- <span class="pre">`PyOS_ascii_strtod(str,`</span>` `<span class="pre">`ptr)`</span> and <span class="pre">`PyOS_ascii_atof(str,`</span>` `<span class="pre">`ptr)`</span> both convert a string to a C <span class="pre">`double`</span>.

- <span class="pre">`PyOS_ascii_formatd(buffer,`</span>` `<span class="pre">`buf_len,`</span>` `<span class="pre">`format,`</span>` `<span class="pre">`d)`</span> converts a <span class="pre">`double`</span> to an ASCII string.

The code for these functions came from the GLib library (<a href="https://developer.gnome.org/glib/stable/" class="reference external">https://developer.gnome.org/glib/stable/</a>), whose developers kindly relicensed the relevant functions and donated them to the Python Software Foundation. The <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a> module can now change the numeric locale, letting extensions such as GTK+ produce the correct results.

<div class="admonition seealso">

See also

<span id="index-11" class="target"></span><a href="https://www.python.org/dev/peps/pep-0331" class="pep reference external"><strong>PEP 331</strong></a> - Locale-Independent Float/String Conversions  
Written by Christian R. Reis, and implemented by Gustavo Carneiro.

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Permalink to this headline">¶</a>

Here are all of the changes that Python 2.4 makes to the core Python language.

- Decorators for functions and methods were added (<span id="index-12" class="target"></span><a href="https://www.python.org/dev/peps/pep-0318" class="pep reference external"><strong>PEP 318</strong></a>).

- Built-in <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a> and <a href="../library/stdtypes.html#frozenset" class="reference internal" title="frozenset"><span class="pre"><code class="sourceCode python"><span class="bu">frozenset</span>()</code></span></a> types were added (<span id="index-13" class="target"></span><a href="https://www.python.org/dev/peps/pep-0218" class="pep reference external"><strong>PEP 218</strong></a>). Other new built-ins include the <span class="pre">`reversed(seq)`</span> function (<span id="index-14" class="target"></span><a href="https://www.python.org/dev/peps/pep-0322" class="pep reference external"><strong>PEP 322</strong></a>).

- Generator expressions were added (<span id="index-15" class="target"></span><a href="https://www.python.org/dev/peps/pep-0289" class="pep reference external"><strong>PEP 289</strong></a>).

- Certain numeric expressions no longer return values restricted to 32 or 64 bits (<span id="index-16" class="target"></span><a href="https://www.python.org/dev/peps/pep-0237" class="pep reference external"><strong>PEP 237</strong></a>).

- You can now put parentheses around the list of names in a <span class="pre">`from`</span>` `<span class="pre">`module`</span>` `<span class="pre">`import`</span>` `<span class="pre">`names`</span> statement (<span id="index-17" class="target"></span><a href="https://www.python.org/dev/peps/pep-0328" class="pep reference external"><strong>PEP 328</strong></a>).

- The <a href="../library/stdtypes.html#dict.update" class="reference internal" title="dict.update"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.update()</code></span></a> method now accepts the same argument forms as the <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> constructor. This includes any mapping, any iterable of key/value pairs, and keyword arguments. (Contributed by Raymond Hettinger.)

- The string methods <span class="pre">`ljust()`</span>, <span class="pre">`rjust()`</span>, and <span class="pre">`center()`</span> now take an optional argument for specifying a fill character other than a space. (Contributed by Raymond Hettinger.)

- Strings also gained an <span class="pre">`rsplit()`</span> method that works like the <span class="pre">`split()`</span> method but splits from the end of the string. (Contributed by Sean Reifschneider.)

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> 'www.python.org'.split('.', 1)
      ['www', 'python.org']
      'www.python.org'.rsplit('.', 1)
      ['www.python', 'org']

  </div>

  </div>

- Three keyword parameters, *cmp*, *key*, and *reverse*, were added to the <span class="pre">`sort()`</span> method of lists. These parameters make some common usages of <span class="pre">`sort()`</span> simpler. All of these parameters are optional.

  For the *cmp* parameter, the value should be a comparison function that takes two parameters and returns -1, 0, or +1 depending on how the parameters compare. This function will then be used to sort the list. Previously this was the only parameter that could be provided to <span class="pre">`sort()`</span>.

  *key* should be a single-parameter function that takes a list element and returns a comparison key for the element. The list is then sorted using the comparison keys. The following example sorts a list case-insensitively:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> L = ['A', 'b', 'c', 'D']
      >>> L.sort()                 # Case-sensitive sort
      >>> L
      ['A', 'D', 'b', 'c']
      >>> # Using 'key' parameter to sort list
      >>> L.sort(key=lambda x: x.lower())
      >>> L
      ['A', 'b', 'c', 'D']
      >>> # Old-fashioned way
      >>> L.sort(cmp=lambda x,y: cmp(x.lower(), y.lower()))
      >>> L
      ['A', 'b', 'c', 'D']

  </div>

  </div>

  The last example, which uses the *cmp* parameter, is the old way to perform a case-insensitive sort. It works but is slower than using a *key* parameter. Using *key* calls <span class="pre">`lower()`</span> method once for each element in the list while using *cmp* will call it twice for each comparison, so using *key* saves on invocations of the <span class="pre">`lower()`</span> method.

  For simple key functions and comparison functions, it is often possible to avoid a <a href="../reference/expressions.html#lambda" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">lambda</code></span></a> expression by using an unbound method instead. For example, the above case-insensitive sort is best written as:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> L.sort(key=str.lower)
      >>> L
      ['A', 'b', 'c', 'D']

  </div>

  </div>

  Finally, the *reverse* parameter takes a Boolean value. If the value is true, the list will be sorted into reverse order. Instead of <span class="pre">`L.sort();`</span>` `<span class="pre">`L.reverse()`</span>, you can now write <span class="pre">`L.sort(reverse=True)`</span>.

  The results of sorting are now guaranteed to be stable. This means that two entries with equal keys will be returned in the same order as they were input. For example, you can sort a list of people by name, and then sort the list by age, resulting in a list sorted by age where people with the same age are in name-sorted order.

  (All changes to <span class="pre">`sort()`</span> contributed by Raymond Hettinger.)

- There is a new built-in function <span class="pre">`sorted(iterable)`</span> that works like the in-place <span class="pre">`list.sort()`</span> method but can be used in expressions. The differences are:

- the input may be any iterable;

- a newly formed copy is sorted, leaving the original intact; and

- the expression returns the new sorted copy

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> L = [9,7,8,3,2,4,1,6,5]
      >>> [10+i for i in sorted(L)]       # usable in a list comprehension
      [11, 12, 13, 14, 15, 16, 17, 18, 19]
      >>> L                               # original is left unchanged
      [9,7,8,3,2,4,1,6,5]
      >>> sorted('Monty Python')          # any iterable may be an input
      [' ', 'M', 'P', 'h', 'n', 'n', 'o', 'o', 't', 't', 'y', 'y']

      >>> # List the contents of a dict sorted by key values
      >>> colormap = dict(red=1, blue=2, green=3, black=4, yellow=5)
      >>> for k, v in sorted(colormap.iteritems()):
      ...     print k, v
      ...
      black 4
      blue 2
      green 3
      red 1
      yellow 5

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- Integer operations will no longer trigger an <span class="pre">`OverflowWarning`</span>. The <span class="pre">`OverflowWarning`</span> warning will disappear in Python 2.5.

- The interpreter gained a new switch, <a href="../using/cmdline.html#cmdoption-m" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-m</code></span></a>, that takes a name, searches for the corresponding module on <span class="pre">`sys.path`</span>, and runs the module as a script. For example, you can now run the Python profiler with <span class="pre">`python`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`profile`</span>. (Contributed by Nick Coghlan.)

- The <span class="pre">`eval(expr,`</span>` `<span class="pre">`globals,`</span>` `<span class="pre">`locals)`</span> and <span class="pre">`execfile(filename,`</span>` `<span class="pre">`globals,`</span>` `<span class="pre">`locals)`</span> functions and the <a href="../reference/simple_stmts.html#exec" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">exec</code></span></a> statement now accept any mapping type for the *locals* parameter. Previously this had to be a regular Python dictionary. (Contributed by Raymond Hettinger.)

- The <a href="../library/functions.html#zip" class="reference internal" title="zip"><span class="pre"><code class="sourceCode python"><span class="bu">zip</span>()</code></span></a> built-in function and <a href="../library/itertools.html#itertools.izip" class="reference internal" title="itertools.izip"><span class="pre"><code class="sourceCode python">itertools.izip()</code></span></a> now return an empty list if called with no arguments. Previously they raised a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception. This makes them more suitable for use with variable length argument lists:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> def transpose(array):
      ...    return zip(*array)
      ...
      >>> transpose([(1,2,3), (4,5,6)])
      [(1, 4), (2, 5), (3, 6)]
      >>> transpose([])
      []

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- Encountering a failure while importing a module no longer leaves a partially-initialized module object in <span class="pre">`sys.modules`</span>. The incomplete module object left behind would fool further imports of the same module into succeeding, leading to confusing errors. (Fixed by Tim Peters.)

- <a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a> is now a constant; code that binds a new value to the name <span class="pre">`None`</span> is now a syntax error. (Contributed by Raymond Hettinger.)

<div id="optimizations" class="section">

### Optimizations<a href="#optimizations" class="headerlink" title="Permalink to this headline">¶</a>

- The inner loops for list and tuple slicing were optimized and now run about one-third faster. The inner loops for dictionaries were also optimized, resulting in performance boosts for <span class="pre">`keys()`</span>, <span class="pre">`values()`</span>, <span class="pre">`items()`</span>, <span class="pre">`iterkeys()`</span>, <span class="pre">`itervalues()`</span>, and <span class="pre">`iteritems()`</span>. (Contributed by Raymond Hettinger.)

- The machinery for growing and shrinking lists was optimized for speed and for space efficiency. Appending and popping from lists now runs faster due to more efficient code paths and less frequent use of the underlying system <span class="pre">`realloc()`</span>. List comprehensions also benefit. <span class="pre">`list.extend()`</span> was also optimized and no longer converts its argument into a temporary list before extending the base list. (Contributed by Raymond Hettinger.)

- <span class="pre">`list()`</span>, <a href="../library/functions.html#tuple" class="reference internal" title="tuple"><span class="pre"><code class="sourceCode python"><span class="bu">tuple</span>()</code></span></a>, <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a>, <a href="../library/functions.html#filter" class="reference internal" title="filter"><span class="pre"><code class="sourceCode python"><span class="bu">filter</span>()</code></span></a>, and <a href="../library/functions.html#zip" class="reference internal" title="zip"><span class="pre"><code class="sourceCode python"><span class="bu">zip</span>()</code></span></a> now run several times faster with non-sequence arguments that supply a <a href="../reference/datamodel.html#object.__len__" class="reference internal" title="object.__len__"><span class="pre"><code class="sourceCode python"><span class="fu">__len__</span>()</code></span></a> method. (Contributed by Raymond Hettinger.)

- The methods <span class="pre">`list.__getitem__()`</span>, <span class="pre">`dict.__getitem__()`</span>, and <span class="pre">`dict.__contains__()`</span> are now implemented as <span class="pre">`method_descriptor`</span> objects rather than <span class="pre">`wrapper_descriptor`</span> objects. This form of access doubles their performance and makes them more suitable for use as arguments to functionals: <span class="pre">`map(mydict.__getitem__,`</span>` `<span class="pre">`keylist)`</span>. (Contributed by Raymond Hettinger.)

- Added a new opcode, <span class="pre">`LIST_APPEND`</span>, that simplifies the generated bytecode for list comprehensions and speeds them up by about a third. (Contributed by Raymond Hettinger.)

- The peephole bytecode optimizer has been improved to produce shorter, faster bytecode; remarkably, the resulting bytecode is more readable. (Enhanced by Raymond Hettinger.)

- String concatenations in statements of the form <span class="pre">`s`</span>` `<span class="pre">`=`</span>` `<span class="pre">`s`</span>` `<span class="pre">`+`</span>` `<span class="pre">`"abc"`</span> and <span class="pre">`s`</span>` `<span class="pre">`+=`</span>` `<span class="pre">`"abc"`</span> are now performed more efficiently in certain circumstances. This optimization won’t be present in other Python implementations such as Jython, so you shouldn’t rely on it; using the <span class="pre">`join()`</span> method of strings is still recommended when you want to efficiently glue a large number of strings together. (Contributed by Armin Rigo.)

The net result of the 2.4 optimizations is that Python 2.4 runs the pystone benchmark around 5% faster than Python 2.3 and 35% faster than Python 2.2. (pystone is not a particularly good benchmark, but it’s the most commonly used measurement of Python’s performance. Your own applications may show greater or smaller benefits from Python 2.4.)

</div>

</div>

<div id="new-improved-and-deprecated-modules" class="section">

## New, Improved, and Deprecated Modules<a href="#new-improved-and-deprecated-modules" class="headerlink" title="Permalink to this headline">¶</a>

As usual, Python’s standard library received a number of enhancements and bug fixes. Here’s a partial list of the most notable changes, sorted alphabetically by module name. Consult the <span class="pre">`Misc/NEWS`</span> file in the source tree for a more complete list of changes, or look through the CVS logs for all the details.

- The <a href="../library/asyncore.html#module-asyncore" class="reference internal" title="asyncore: A base class for developing asynchronous socket handling services."><span class="pre"><code class="sourceCode python">asyncore</code></span></a> module’s <span class="pre">`loop()`</span> function now has a *count* parameter that lets you perform a limited number of passes through the polling loop. The default is still to loop forever.

- The <a href="../library/base64.html#module-base64" class="reference internal" title="base64: RFC 3548: Base16, Base32, Base64 Data Encodings"><span class="pre"><code class="sourceCode python">base64</code></span></a> module now has more complete RFC 3548 support for Base64, Base32, and Base16 encoding and decoding, including optional case folding and optional alternative alphabets. (Contributed by Barry Warsaw.)

- The <a href="../library/bisect.html#module-bisect" class="reference internal" title="bisect: Array bisection algorithms for binary searching."><span class="pre"><code class="sourceCode python">bisect</code></span></a> module now has an underlying C implementation for improved performance. (Contributed by Dmitry Vasiliev.)

- The CJKCodecs collections of East Asian codecs, maintained by Hye-Shik Chang, was integrated into 2.4. The new encodings are:

- Chinese (PRC): gb2312, gbk, gb18030, big5hkscs, hz

- Chinese (ROC): big5, cp950

- Japanese: cp932, euc-jis-2004, euc-jp, euc-jisx0213, iso-2022-jp,  
  iso-2022-jp-1, iso-2022-jp-2, iso-2022-jp-3, iso-2022-jp-ext, iso-2022-jp-2004, shift-jis, shift-jisx0213, shift-jis-2004

- Korean: cp949, euc-kr, johab, iso-2022-kr

- Some other new encodings were added: HP Roman8, ISO_8859-11, ISO_8859-16, PCTP-154, and TIS-620.

- The UTF-8 and UTF-16 codecs now cope better with receiving partial input. Previously the <span class="pre">`StreamReader`</span> class would try to read more data, making it impossible to resume decoding from the stream. The <span class="pre">`read()`</span> method will now return as much data as it can and future calls will resume decoding where previous ones left off. (Implemented by Walter Dörwald.)

- There is a new <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module for various specialized collection datatypes. Currently it contains just one type, <span class="pre">`deque`</span>, a double-ended queue that supports efficiently adding and removing elements from either end:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> from collections import deque
      >>> d = deque('ghi')        # make a new deque with three items
      >>> d.append('j')           # add a new entry to the right side
      >>> d.appendleft('f')       # add a new entry to the left side
      >>> d                       # show the representation of the deque
      deque(['f', 'g', 'h', 'i', 'j'])
      >>> d.pop()                 # return and remove the rightmost item
      'j'
      >>> d.popleft()             # return and remove the leftmost item
      'f'
      >>> list(d)                 # list the contents of the deque
      ['g', 'h', 'i']
      >>> 'h' in d                # search the deque
      True

  </div>

  </div>

  Several modules, such as the <a href="../library/queue.html#module-Queue" class="reference internal" title="Queue: A synchronized queue class."><span class="pre"><code class="sourceCode python">Queue</code></span></a> and <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> modules, now take advantage of <a href="../library/collections.html#collections.deque" class="reference internal" title="collections.deque"><span class="pre"><code class="sourceCode python">collections.deque</code></span></a> for improved performance. (Contributed by Raymond Hettinger.)

- The <a href="../library/configparser.html#module-ConfigParser" class="reference internal" title="ConfigParser: Configuration file parser."><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a> classes have been enhanced slightly. The <span class="pre">`read()`</span> method now returns a list of the files that were successfully parsed, and the <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a> method raises <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> if passed a *value* argument that isn’t a string. (Contributed by John Belmonte and David Goodger.)

- The <a href="../library/curses.html#module-curses" class="reference internal" title="curses: An interface to the curses library, providing portable terminal handling. (Unix)"><span class="pre"><code class="sourceCode python">curses</code></span></a> module now supports the ncurses extension <span class="pre">`use_default_colors()`</span>. On platforms where the terminal supports transparency, this makes it possible to use a transparent background. (Contributed by Jörg Lehmann.)

- The <a href="../library/difflib.html#module-difflib" class="reference internal" title="difflib: Helpers for computing differences between objects."><span class="pre"><code class="sourceCode python">difflib</code></span></a> module now includes an <span class="pre">`HtmlDiff`</span> class that creates an HTML table showing a side by side comparison of two versions of a text. (Contributed by Dan Gass.)

- The <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages, including MIME documents."><span class="pre"><code class="sourceCode python">email</code></span></a> package was updated to version 3.0, which dropped various deprecated APIs and removes support for Python versions earlier than 2.3. The 3.0 version of the package uses a new incremental parser for MIME messages, available in the <span class="pre">`email.FeedParser`</span> module. The new parser doesn’t require reading the entire message into memory, and doesn’t raise exceptions if a message is malformed; instead it records any problems in the <span class="pre">`defect`</span> attribute of the message. (Developed by Anthony Baxter, Barry Warsaw, Thomas Wouters, and others.)

- The <a href="../library/heapq.html#module-heapq" class="reference internal" title="heapq: Heap queue algorithm (a.k.a. priority queue)."><span class="pre"><code class="sourceCode python">heapq</code></span></a> module has been converted to C. The resulting tenfold improvement in speed makes the module suitable for handling high volumes of data. In addition, the module has two new functions <span class="pre">`nlargest()`</span> and <span class="pre">`nsmallest()`</span> that use heaps to find the N largest or smallest values in a dataset without the expense of a full sort. (Contributed by Raymond Hettinger.)

- The <a href="../library/httplib.html#module-httplib" class="reference internal" title="httplib: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">httplib</code></span></a> module now contains constants for HTTP status codes defined in various HTTP-related RFC documents. Constants have names such as <span class="pre">`OK`</span>, <span class="pre">`CREATED`</span>, <span class="pre">`CONTINUE`</span>, and <span class="pre">`MOVED_PERMANENTLY`</span>; use pydoc to get a full list. (Contributed by Andrew Eland.)

- The <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a> module now supports IMAP’s THREAD command (contributed by Yves Dionne) and new <span class="pre">`deleteacl()`</span> and <span class="pre">`myrights()`</span> methods (contributed by Arnaud Mazin).

- The <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> module gained a <span class="pre">`groupby(iterable[,`</span>` `<span class="pre">`*func*])`</span> function. *iterable* is something that can be iterated over to return a stream of elements, and the optional *func* parameter is a function that takes an element and returns a key value; if omitted, the key is simply the element itself. <span class="pre">`groupby()`</span> then groups the elements into subsequences which have matching values of the key, and returns a series of 2-tuples containing the key value and an iterator over the subsequence.

  Here’s an example to make this clearer. The *key* function simply returns whether a number is even or odd, so the result of <span class="pre">`groupby()`</span> is to return consecutive runs of odd or even numbers.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import itertools
      >>> L = [2, 4, 6, 7, 8, 9, 11, 12, 14]
      >>> for key_val, it in itertools.groupby(L, lambda x: x % 2):
      ...    print key_val, list(it)
      ...
      0 [2, 4, 6]
      1 [7]
      0 [8]
      1 [9, 11]
      0 [12, 14]
      >>>

  </div>

  </div>

  <span class="pre">`groupby()`</span> is typically used with sorted input. The logic for <span class="pre">`groupby()`</span> is similar to the Unix <span class="pre">`uniq`</span> filter which makes it handy for eliminating, counting, or identifying duplicate elements:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> word = 'abracadabra'
      >>> letters = sorted(word)   # Turn string into a sorted list of letters
      >>> letters
      ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'c', 'd', 'r', 'r']
      >>> for k, g in itertools.groupby(letters):
      ...    print k, list(g)
      ...
      a ['a', 'a', 'a', 'a', 'a']
      b ['b', 'b']
      c ['c']
      d ['d']
      r ['r', 'r']
      >>> # List unique letters
      >>> [k for k, g in groupby(letters)]
      ['a', 'b', 'c', 'd', 'r']
      >>> # Count letter occurrences
      >>> [(k, len(list(g))) for k, g in groupby(letters)]
      [('a', 5), ('b', 2), ('c', 1), ('d', 1), ('r', 2)]

  </div>

  </div>

  (Contributed by Hye-Shik Chang.)

- <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> also gained a function named <span class="pre">`tee(iterator,`</span>` `<span class="pre">`N)`</span> that returns *N* independent iterators that replicate *iterator*. If *N* is omitted, the default is 2.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> L = [1,2,3]
      >>> i1, i2 = itertools.tee(L)
      >>> i1,i2
      (<itertools.tee object at 0x402c2080>, <itertools.tee object at 0x402c2090>)
      >>> list(i1)               # Run the first iterator to exhaustion
      [1, 2, 3]
      >>> list(i2)               # Run the second iterator to exhaustion
      [1, 2, 3]

  </div>

  </div>

  Note that <span class="pre">`tee()`</span> has to keep copies of the values returned by the iterator; in the worst case, it may need to keep all of them. This should therefore be used carefully if the leading iterator can run far ahead of the trailing iterator in a long stream of inputs. If the separation is large, then you might as well use <span class="pre">`list()`</span> instead. When the iterators track closely with one another, <span class="pre">`tee()`</span> is ideal. Possible applications include bookmarking, windowing, or lookahead iterators. (Contributed by Raymond Hettinger.)

- A number of functions were added to the <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a> module, such as <span class="pre">`bind_textdomain_codeset()`</span> to specify a particular encoding and a family of <span class="pre">`l*gettext()`</span> functions that return messages in the chosen encoding. (Contributed by Gustavo Niemeyer.)

- Some keyword arguments were added to the <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> package’s <span class="pre">`basicConfig()`</span> function to simplify log configuration. The default behavior is to log messages to standard error, but various keyword arguments can be specified to log to a particular file, change the logging format, or set the logging level. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      import logging
      logging.basicConfig(filename='/var/log/application.log',
          level=0,  # Log all messages
          format='%(levelname):%(process):%(thread):%(message)')

  </div>

  </div>

  Other additions to the <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> package include a <span class="pre">`log(level,`</span>` `<span class="pre">`msg)`</span> convenience method, as well as a <span class="pre">`TimedRotatingFileHandler`</span> class that rotates its log files at a timed interval. The module already had <span class="pre">`RotatingFileHandler`</span>, which rotated logs once the file exceeded a certain size. Both classes derive from a new <span class="pre">`BaseRotatingHandler`</span> class that can be used to implement other rotating handlers.

  (Changes implemented by Vinay Sajip.)

- The <a href="../library/marshal.html#module-marshal" class="reference internal" title="marshal: Convert Python objects to streams of bytes and back (with different constraints)."><span class="pre"><code class="sourceCode python">marshal</code></span></a> module now shares interned strings on unpacking a data structure. This may shrink the size of certain pickle strings, but the primary effect is to make <span class="pre">`.pyc`</span> files significantly smaller. (Contributed by Martin von Löwis.)

- The <a href="../library/nntplib.html#module-nntplib" class="reference internal" title="nntplib: NNTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">nntplib</code></span></a> module’s <span class="pre">`NNTP`</span> class gained <span class="pre">`description()`</span> and <span class="pre">`descriptions()`</span> methods to retrieve newsgroup descriptions for a single group or for a range of groups. (Contributed by Jürgen A. Erhard.)

- Two new functions were added to the <a href="../library/operator.html#module-operator" class="reference internal" title="operator: Functions corresponding to the standard operators."><span class="pre"><code class="sourceCode python">operator</code></span></a> module, <span class="pre">`attrgetter(attr)`</span> and <span class="pre">`itemgetter(index)`</span>. Both functions return callables that take a single argument and return the corresponding attribute or item; these callables make excellent data extractors when used with <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> or <a href="../library/functions.html#sorted" class="reference internal" title="sorted"><span class="pre"><code class="sourceCode python"><span class="bu">sorted</span>()</code></span></a>. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> L = [('c', 2), ('d', 1), ('a', 4), ('b', 3)]
      >>> map(operator.itemgetter(0), L)
      ['c', 'd', 'a', 'b']
      >>> map(operator.itemgetter(1), L)
      [2, 1, 4, 3]
      >>> sorted(L, key=operator.itemgetter(1)) # Sort list by second tuple item
      [('d', 1), ('c', 2), ('b', 3), ('a', 4)]

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- The <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a> module was updated in various ways. The module now passes its messages through <a href="../library/gettext.html#gettext.gettext" class="reference internal" title="gettext.gettext"><span class="pre"><code class="sourceCode python">gettext.gettext()</code></span></a>, making it possible to internationalize Optik’s help and error messages. Help messages for options can now include the string <span class="pre">`'%default'`</span>, which will be replaced by the option’s default value. (Contributed by Greg Ward.)

- The long-term plan is to deprecate the <a href="../library/rfc822.html#module-rfc822" class="reference internal" title="rfc822: Parse 2822 style mail messages. (deprecated)"><span class="pre"><code class="sourceCode python">rfc822</code></span></a> module in some future Python release in favor of the <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages, including MIME documents."><span class="pre"><code class="sourceCode python">email</code></span></a> package. To this end, the <span class="pre">`email.Utils.formatdate()`</span> function has been changed to make it usable as a replacement for <span class="pre">`rfc822.formatdate()`</span>. You may want to write new e-mail processing code with this in mind. (Change implemented by Anthony Baxter.)

- A new <span class="pre">`urandom(n)`</span> function was added to the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module, returning a string containing *n* bytes of random data. This function provides access to platform-specific sources of randomness such as <span class="pre">`/dev/urandom`</span> on Linux or the Windows CryptoAPI. (Contributed by Trevor Perrin.)

- Another new function: <span class="pre">`os.path.lexists(path)`</span> returns true if the file specified by *path* exists, whether or not it’s a symbolic link. This differs from the existing <span class="pre">`os.path.exists(path)`</span> function, which returns false if *path* is a symlink that points to a destination that doesn’t exist. (Contributed by Beni Cherniavsky.)

- A new <span class="pre">`getsid()`</span> function was added to the <a href="../library/posix.html#module-posix" class="reference internal" title="posix: The most common POSIX system calls (normally used via module os). (Unix)"><span class="pre"><code class="sourceCode python">posix</code></span></a> module that underlies the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module. (Contributed by J. Raynor.)

- The <a href="../library/poplib.html#module-poplib" class="reference internal" title="poplib: POP3 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">poplib</code></span></a> module now supports POP over SSL. (Contributed by Hector Urtubia.)

- The <a href="../library/profile.html#module-profile" class="reference internal" title="profile: Python source profiler."><span class="pre"><code class="sourceCode python">profile</code></span></a> module can now profile C extension functions. (Contributed by Nick Bastin.)

- The <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module has a new method called <span class="pre">`getrandbits(N)`</span> that returns a long integer *N* bits in length. The existing <span class="pre">`randrange()`</span> method now uses <span class="pre">`getrandbits()`</span> where appropriate, making generation of arbitrarily large random numbers more efficient. (Contributed by Raymond Hettinger.)

- The regular expression language accepted by the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module was extended with simple conditional expressions, written as <span class="pre">`(?(group)A|B)`</span>. *group* is either a numeric group ID or a group name defined with <span class="pre">`(?P<group>...)`</span> earlier in the expression. If the specified group matched, the regular expression pattern *A* will be tested against the string; if the group didn’t match, the pattern *B* will be used instead. (Contributed by Gustavo Niemeyer.)

- The <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module is also no longer recursive, thanks to a massive amount of work by Gustavo Niemeyer. In a recursive regular expression engine, certain patterns result in a large amount of C stack space being consumed, and it was possible to overflow the stack. For example, if you matched a 30000-byte string of <span class="pre">`a`</span> characters against the expression <span class="pre">`(a|b)+`</span>, one stack frame was consumed per character. Python 2.3 tried to check for stack overflow and raise a <a href="../library/exceptions.html#exceptions.RuntimeError" class="reference internal" title="exceptions.RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> exception, but certain patterns could sidestep the checking and if you were unlucky Python could segfault. Python 2.4’s regular expression engine can match this pattern without problems.

- The <a href="../library/signal.html#module-signal" class="reference internal" title="signal: Set handlers for asynchronous events."><span class="pre"><code class="sourceCode python">signal</code></span></a> module now performs tighter error-checking on the parameters to the <a href="../library/signal.html#signal.signal" class="reference internal" title="signal.signal"><span class="pre"><code class="sourceCode python">signal.signal()</code></span></a> function. For example, you can’t set a handler on the <span class="pre">`SIGKILL`</span> signal; previous versions of Python would quietly accept this, but 2.4 will raise a <a href="../library/exceptions.html#exceptions.RuntimeError" class="reference internal" title="exceptions.RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> exception.

- Two new functions were added to the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module. <span class="pre">`socketpair()`</span> returns a pair of connected sockets and <span class="pre">`getservbyport(port)`</span> looks up the service name for a given port number. (Contributed by Dave Cole and Barry Warsaw.)

- The <a href="../library/sys.html#sys.exitfunc" class="reference internal" title="sys.exitfunc"><span class="pre"><code class="sourceCode python">sys.exitfunc()</code></span></a> function has been deprecated. Code should be using the existing <a href="../library/atexit.html#module-atexit" class="reference internal" title="atexit: Register and execute cleanup functions."><span class="pre"><code class="sourceCode python">atexit</code></span></a> module, which correctly handles calling multiple exit functions. Eventually <a href="../library/sys.html#sys.exitfunc" class="reference internal" title="sys.exitfunc"><span class="pre"><code class="sourceCode python">sys.exitfunc()</code></span></a> will become a purely internal interface, accessed only by <a href="../library/atexit.html#module-atexit" class="reference internal" title="atexit: Register and execute cleanup functions."><span class="pre"><code class="sourceCode python">atexit</code></span></a>.

- The <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module now generates GNU-format tar files by default. (Contributed by Lars Gustaebel.)

- The <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> module now has an elegantly simple way to support thread-local data. The module contains a <span class="pre">`local`</span> class whose attribute values are local to different threads.

  <div class="highlight-default notranslate">

  <div class="highlight">

      import threading

      data = threading.local()
      data.number = 42
      data.url = ('www.python.org', 80)

  </div>

  </div>

  Other threads can assign and retrieve their own values for the <span class="pre">`number`</span> and <span class="pre">`url`</span> attributes. You can subclass <span class="pre">`local`</span> to initialize attributes or to add methods. (Contributed by Jim Fulton.)

- The <a href="../library/timeit.html#module-timeit" class="reference internal" title="timeit: Measure the execution time of small code snippets."><span class="pre"><code class="sourceCode python">timeit</code></span></a> module now automatically disables periodic garbage collection during the timing loop. This change makes consecutive timings more comparable. (Contributed by Raymond Hettinger.)

- The <a href="../library/weakref.html#module-weakref" class="reference internal" title="weakref: Support for weak references and weak dictionaries."><span class="pre"><code class="sourceCode python">weakref</code></span></a> module now supports a wider variety of objects including Python functions, class instances, sets, frozensets, deques, arrays, files, sockets, and regular expression pattern objects. (Contributed by Raymond Hettinger.)

- The <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a> module now supports a multi-call extension for transmitting multiple XML-RPC calls in a single HTTP operation. (Contributed by Brian Quinlan.)

- The <span class="pre">`mpz`</span>, <span class="pre">`rotor`</span>, and <span class="pre">`xreadlines`</span> modules have been removed.

<div id="cookielib" class="section">

### cookielib<a href="#cookielib" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/cookielib.html#module-cookielib" class="reference internal" title="cookielib: Classes for automatic handling of HTTP cookies."><span class="pre"><code class="sourceCode python">cookielib</code></span></a> library supports client-side handling for HTTP cookies, mirroring the <a href="../library/cookie.html#module-Cookie" class="reference internal" title="Cookie: Support for HTTP state management (cookies)."><span class="pre"><code class="sourceCode python">Cookie</code></span></a> module’s server-side cookie support. Cookies are stored in cookie jars; the library transparently stores cookies offered by the web server in the cookie jar, and fetches the cookie from the jar when connecting to the server. As in web browsers, policy objects control whether cookies are accepted or not.

In order to store cookies across sessions, two implementations of cookie jars are provided: one that stores cookies in the Netscape format so applications can use the Mozilla or Lynx cookie files, and one that stores cookies in the same format as the Perl libwww library.

<a href="../library/urllib2.html#module-urllib2" class="reference internal" title="urllib2: Next generation URL opening library."><span class="pre"><code class="sourceCode python">urllib2</code></span></a> has been changed to interact with <a href="../library/cookielib.html#module-cookielib" class="reference internal" title="cookielib: Classes for automatic handling of HTTP cookies."><span class="pre"><code class="sourceCode python">cookielib</code></span></a>: <span class="pre">`HTTPCookieProcessor`</span> manages a cookie jar that is used when accessing URLs.

This module was contributed by John J. Lee.

</div>

<div id="doctest" class="section">

### doctest<a href="#doctest" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> module underwent considerable refactoring thanks to Edward Loper and Tim Peters. Testing can still be as simple as running <a href="../library/doctest.html#doctest.testmod" class="reference internal" title="doctest.testmod"><span class="pre"><code class="sourceCode python">doctest.testmod()</code></span></a>, but the refactorings allow customizing the module’s operation in various ways

The new <span class="pre">`DocTestFinder`</span> class extracts the tests from a given object’s docstrings:

<div class="highlight-default notranslate">

<div class="highlight">

    def f (x, y):
        """>>> f(2,2)
    4
    >>> f(3,2)
    6
        """
        return x*y

    finder = doctest.DocTestFinder()

    # Get list of DocTest instances
    tests = finder.find(f)

</div>

</div>

The new <span class="pre">`DocTestRunner`</span> class then runs individual tests and can produce a summary of the results:

<div class="highlight-default notranslate">

<div class="highlight">

    runner = doctest.DocTestRunner()
    for t in tests:
        tried, failed = runner.run(t)

    runner.summarize(verbose=1)

</div>

</div>

The above example produces the following output:

<div class="highlight-default notranslate">

<div class="highlight">

    1 items passed all tests:
       2 tests in f
    2 tests in 1 items.
    2 passed and 0 failed.
    Test passed.

</div>

</div>

<span class="pre">`DocTestRunner`</span> uses an instance of the <span class="pre">`OutputChecker`</span> class to compare the expected output with the actual output. This class takes a number of different flags that customize its behaviour; ambitious users can also write a completely new subclass of <span class="pre">`OutputChecker`</span>.

The default output checker provides a number of handy features. For example, with the <a href="../library/doctest.html#doctest.ELLIPSIS" class="reference internal" title="doctest.ELLIPSIS"><span class="pre"><code class="sourceCode python">doctest.ELLIPSIS</code></span></a> option flag, an ellipsis (<span class="pre">`...`</span>) in the expected output matches any substring, making it easier to accommodate outputs that vary in minor ways:

<div class="highlight-default notranslate">

<div class="highlight">

    def o (n):
        """>>> o(1)
    <__main__.C instance at 0x...>
    >>>
    """

</div>

</div>

Another special string, <span class="pre">`<BLANKLINE>`</span>, matches a blank line:

<div class="highlight-default notranslate">

<div class="highlight">

    def p (n):
        """>>> p(1)
    <BLANKLINE>
    >>>
    """

</div>

</div>

Another new capability is producing a diff-style display of the output by specifying the <a href="../library/doctest.html#doctest.REPORT_UDIFF" class="reference internal" title="doctest.REPORT_UDIFF"><span class="pre"><code class="sourceCode python">doctest.REPORT_UDIFF</code></span></a> (unified diffs), <a href="../library/doctest.html#doctest.REPORT_CDIFF" class="reference internal" title="doctest.REPORT_CDIFF"><span class="pre"><code class="sourceCode python">doctest.REPORT_CDIFF</code></span></a> (context diffs), or <a href="../library/doctest.html#doctest.REPORT_NDIFF" class="reference internal" title="doctest.REPORT_NDIFF"><span class="pre"><code class="sourceCode python">doctest.REPORT_NDIFF</code></span></a> (delta-style) option flags. For example:

<div class="highlight-default notranslate">

<div class="highlight">

    def g (n):
        """>>> g(4)
    here
    is
    a
    lengthy
    >>>"""
        L = 'here is a rather lengthy list of words'.split()
        for word in L[:n]:
            print word

</div>

</div>

Running the above function’s tests with <a href="../library/doctest.html#doctest.REPORT_UDIFF" class="reference internal" title="doctest.REPORT_UDIFF"><span class="pre"><code class="sourceCode python">doctest.REPORT_UDIFF</code></span></a> specified, you get the following output:

<div class="highlight-none notranslate">

<div class="highlight">

    **********************************************************************
    File "t.py", line 15, in g
    Failed example:
        g(4)
    Differences (unified diff with -expected +actual):
        @@ -2,3 +2,3 @@
         is
         a
        -lengthy
        +rather
    **********************************************************************

</div>

</div>

</div>

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Permalink to this headline">¶</a>

Some of the changes to Python’s build process and to the C API are:

- Three new convenience macros were added for common return values from extension functions: <a href="../c-api/none.html#c.Py_RETURN_NONE" class="reference internal" title="Py_RETURN_NONE"><span class="pre"><code class="sourceCode c">Py_RETURN_NONE</code></span></a>, <a href="../c-api/bool.html#c.Py_RETURN_TRUE" class="reference internal" title="Py_RETURN_TRUE"><span class="pre"><code class="sourceCode c">Py_RETURN_TRUE</code></span></a>, and <a href="../c-api/bool.html#c.Py_RETURN_FALSE" class="reference internal" title="Py_RETURN_FALSE"><span class="pre"><code class="sourceCode c">Py_RETURN_FALSE</code></span></a>. (Contributed by Brett Cannon.)

- Another new macro, <span class="pre">`Py_CLEAR(obj)`</span>, decreases the reference count of *obj* and sets *obj* to the null pointer. (Contributed by Jim Fulton.)

- A new function, <span class="pre">`PyTuple_Pack(N,`</span>` `<span class="pre">`obj1,`</span>` `<span class="pre">`obj2,`</span>` `<span class="pre">`...,`</span>` `<span class="pre">`objN)`</span>, constructs tuples from a variable length argument list of Python objects. (Contributed by Raymond Hettinger.)

- A new function, <span class="pre">`PyDict_Contains(d,`</span>` `<span class="pre">`k)`</span>, implements fast dictionary lookups without masking exceptions raised during the look-up process. (Contributed by Raymond Hettinger.)

- The <span class="pre">`Py_IS_NAN(X)`</span> macro returns 1 if its float or double argument *X* is a NaN. (Contributed by Tim Peters.)

- C code can avoid unnecessary locking by using the new <a href="../c-api/init.html#c.PyEval_ThreadsInitialized" class="reference internal" title="PyEval_ThreadsInitialized"><span class="pre"><code class="sourceCode c">PyEval_ThreadsInitialized<span class="op">()</span></code></span></a> function to tell if any thread operations have been performed. If this function returns false, no lock operations are needed. (Contributed by Nick Coghlan.)

- A new function, <a href="../c-api/arg.html#c.PyArg_VaParseTupleAndKeywords" class="reference internal" title="PyArg_VaParseTupleAndKeywords"><span class="pre"><code class="sourceCode c">PyArg_VaParseTupleAndKeywords<span class="op">()</span></code></span></a>, is the same as <a href="../c-api/arg.html#c.PyArg_ParseTupleAndKeywords" class="reference internal" title="PyArg_ParseTupleAndKeywords"><span class="pre"><code class="sourceCode c">PyArg_ParseTupleAndKeywords<span class="op">()</span></code></span></a> but takes a <span class="pre">`va_list`</span> instead of a number of arguments. (Contributed by Greg Chapman.)

- A new method flag, <span class="pre">`METH_COEXISTS`</span>, allows a function defined in slots to co-exist with a <a href="../c-api/structures.html#c.PyCFunction" class="reference internal" title="PyCFunction"><span class="pre"><code class="sourceCode c">PyCFunction</code></span></a> having the same name. This can halve the access time for a method such as <span class="pre">`set.__contains__()`</span>. (Contributed by Raymond Hettinger.)

- Python can now be built with additional profiling for the interpreter itself, intended as an aid to people developing the Python core. Providing <span class="pre">`--enable-profiling`</span> to the **configure** script will let you profile the interpreter with **gprof**, and providing the <span class="pre">`--with-tsc`</span> switch enables profiling using the Pentium’s Time-Stamp-Counter register. Note that the <span class="pre">`--with-tsc`</span> switch is slightly misnamed, because the profiling feature also works on the PowerPC platform, though that processor architecture doesn’t call that register “the TSC register”. (Contributed by Jeremy Hylton.)

- The <span class="pre">`tracebackobject`</span> type has been renamed to <span class="pre">`PyTracebackObject`</span>.

<div id="port-specific-changes" class="section">

### Port-Specific Changes<a href="#port-specific-changes" class="headerlink" title="Permalink to this headline">¶</a>

- The Windows port now builds under MSVC++ 7.1 as well as version 6. (Contributed by Martin von Löwis.)

</div>

</div>

<div id="porting-to-python-2-4" class="section">

## Porting to Python 2.4<a href="#porting-to-python-2-4" class="headerlink" title="Permalink to this headline">¶</a>

This section lists previously described changes that may require changes to your code:

- Left shifts and hexadecimal/octal constants that are too large no longer trigger a <a href="../library/exceptions.html#exceptions.FutureWarning" class="reference internal" title="exceptions.FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a> and return a value limited to 32 or 64 bits; instead they return a long integer.

- Integer operations will no longer trigger an <span class="pre">`OverflowWarning`</span>. The <span class="pre">`OverflowWarning`</span> warning will disappear in Python 2.5.

- The <a href="../library/functions.html#zip" class="reference internal" title="zip"><span class="pre"><code class="sourceCode python"><span class="bu">zip</span>()</code></span></a> built-in function and <a href="../library/itertools.html#itertools.izip" class="reference internal" title="itertools.izip"><span class="pre"><code class="sourceCode python">itertools.izip()</code></span></a> now return an empty list instead of raising a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception if called with no arguments.

- You can no longer compare the <span class="pre">`date`</span> and <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> instances provided by the <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> module. Two instances of different classes will now always be unequal, and relative comparisons (<span class="pre">`<`</span>, <span class="pre">`>`</span>) will raise a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>.

- <a href="../library/dircache.html#dircache.listdir" class="reference internal" title="dircache.listdir"><span class="pre"><code class="sourceCode python">dircache.listdir()</code></span></a> now passes exceptions to the caller instead of returning empty lists.

- <span class="pre">`LexicalHandler.startDTD()`</span> used to receive the public and system IDs in the wrong order. This has been corrected; applications relying on the wrong order need to be fixed.

- <a href="../library/fcntl.html#fcntl.ioctl" class="reference internal" title="fcntl.ioctl"><span class="pre"><code class="sourceCode python">fcntl.ioctl()</code></span></a> now warns if the *mutate* argument is omitted and relevant.

- The <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module now generates GNU-format tar files by default.

- Encountering a failure while importing a module no longer leaves a partially-initialized module object in <span class="pre">`sys.modules`</span>.

- <a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a> is now a constant; code that binds a new value to the name <span class="pre">`None`</span> is now a syntax error.

- The <span class="pre">`signals.signal()`</span> function now raises a <a href="../library/exceptions.html#exceptions.RuntimeError" class="reference internal" title="exceptions.RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a> exception for certain illegal values; previously these errors would pass silently. For example, you can no longer set a handler on the <span class="pre">`SIGKILL`</span> signal.

</div>

<div id="acknowledgements" class="section">

<span id="acks"></span>

## Acknowledgements<a href="#acknowledgements" class="headerlink" title="Permalink to this headline">¶</a>

The author would like to thank the following people for offering suggestions, corrections and assistance with various drafts of this article: Koray Can, Hye-Shik Chang, Michael Dyck, Raymond Hettinger, Brian Hurt, Hamish Lawson, Fredrik Lundh, Sean Reifschneider, Sadruddin Rejeb.

</div>

</div>

</div>
