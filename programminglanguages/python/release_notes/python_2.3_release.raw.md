<div class="body" role="main">

<div id="what-s-new-in-python-2-3" class="section">

# What’s New in Python 2.3<a href="#what-s-new-in-python-2-3" class="headerlink" title="Permalink to this headline">¶</a>

Author  
A.M. Kuchling

This article explains the new features in Python 2.3. Python 2.3 was released on July 29, 2003.

The main themes for Python 2.3 are polishing some of the features added in 2.2, adding various small but useful enhancements to the core language, and expanding the standard library. The new object model introduced in the previous version has benefited from 18 months of bugfixes and from optimization efforts that have improved the performance of new-style classes. A few new built-in functions have been added such as <a href="../library/functions.html#sum" class="reference internal" title="sum"><span class="pre"><code class="sourceCode python"><span class="bu">sum</span>()</code></span></a> and <a href="../library/functions.html#enumerate" class="reference internal" title="enumerate"><span class="pre"><code class="sourceCode python"><span class="bu">enumerate</span>()</code></span></a>. The <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> operator can now be used for substring searches (e.g. <span class="pre">`"ab"`</span>` `<span class="pre">`in`</span>` `<span class="pre">`"abc"`</span> returns <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a>).

Some of the many new library features include Boolean, set, heap, and date/time data types, the ability to import modules from ZIP-format archives, metadata support for the long-awaited Python catalog, an updated version of IDLE, and modules for logging messages, wrapping text, parsing CSV files, processing command-line options, using BerkeleyDB databases… the list of new and enhanced modules is lengthy.

This article doesn’t attempt to provide a complete specification of the new features, but instead provides a convenient overview. For full details, you should refer to the documentation for Python 2.3, such as the Python Library Reference and the Python Reference Manual. If you want to understand the complete implementation and design rationale, refer to the PEP for a particular new feature.

<div id="pep-218-a-standard-set-datatype" class="section">

## PEP 218: A Standard Set Datatype<a href="#pep-218-a-standard-set-datatype" class="headerlink" title="Permalink to this headline">¶</a>

The new <a href="../library/sets.html#module-sets" class="reference internal" title="sets: Implementation of sets of unique elements. (deprecated)"><span class="pre"><code class="sourceCode python">sets</code></span></a> module contains an implementation of a set datatype. The <span class="pre">`Set`</span> class is for mutable sets, sets that can have members added and removed. The <span class="pre">`ImmutableSet`</span> class is for sets that can’t be modified, and instances of <span class="pre">`ImmutableSet`</span> can therefore be used as dictionary keys. Sets are built on top of dictionaries, so the elements within a set must be hashable.

Here’s a simple example:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> import sets
    >>> S = sets.Set([1,2,3])
    >>> S
    Set([1, 2, 3])
    >>> 1 in S
    True
    >>> 0 in S
    False
    >>> S.add(5)
    >>> S.remove(3)
    >>> S
    Set([1, 2, 5])
    >>>

</div>

</div>

The union and intersection of sets can be computed with the <span class="pre">`union()`</span> and <span class="pre">`intersection()`</span> methods; an alternative notation uses the bitwise operators <span class="pre">`&`</span> and <span class="pre">`|`</span>. Mutable sets also have in-place versions of these methods, <span class="pre">`union_update()`</span> and <span class="pre">`intersection_update()`</span>.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> S1 = sets.Set([1,2,3])
    >>> S2 = sets.Set([4,5,6])
    >>> S1.union(S2)
    Set([1, 2, 3, 4, 5, 6])
    >>> S1 | S2                  # Alternative notation
    Set([1, 2, 3, 4, 5, 6])
    >>> S1.intersection(S2)
    Set([])
    >>> S1 & S2                  # Alternative notation
    Set([])
    >>> S1.union_update(S2)
    >>> S1
    Set([1, 2, 3, 4, 5, 6])
    >>>

</div>

</div>

It’s also possible to take the symmetric difference of two sets. This is the set of all elements in the union that aren’t in the intersection. Another way of putting it is that the symmetric difference contains all elements that are in exactly one set. Again, there’s an alternative notation (<span class="pre">`^`</span>), and an in-place version with the ungainly name <span class="pre">`symmetric_difference_update()`</span>.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> S1 = sets.Set([1,2,3,4])
    >>> S2 = sets.Set([3,4,5,6])
    >>> S1.symmetric_difference(S2)
    Set([1, 2, 5, 6])
    >>> S1 ^ S2
    Set([1, 2, 5, 6])
    >>>

</div>

</div>

There are also <span class="pre">`issubset()`</span> and <span class="pre">`issuperset()`</span> methods for checking whether one set is a subset or superset of another:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> S1 = sets.Set([1,2,3])
    >>> S2 = sets.Set([2,3])
    >>> S2.issubset(S1)
    True
    >>> S1.issubset(S2)
    False
    >>> S1.issuperset(S2)
    True
    >>>

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://www.python.org/dev/peps/pep-0218" class="pep reference external"><strong>PEP 218</strong></a> - Adding a Built-In Set Object Type  
PEP written by Greg V. Wilson. Implemented by Greg V. Wilson, Alex Martelli, and GvR.

</div>

</div>

<div id="pep-255-simple-generators" class="section">

<span id="section-generators"></span>

## PEP 255: Simple Generators<a href="#pep-255-simple-generators" class="headerlink" title="Permalink to this headline">¶</a>

In Python 2.2, generators were added as an optional feature, to be enabled by a <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`generators`</span> directive. In 2.3 generators no longer need to be specially enabled, and are now always present; this means that <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> is now always a keyword. The rest of this section is a copy of the description of generators from the “What’s New in Python 2.2” document; if you read it back when Python 2.2 came out, you can skip the rest of this section.

You’re doubtless familiar with how function calls work in Python or C. When you call a function, it gets a private namespace where its local variables are created. When the function reaches a <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statement, the local variables are destroyed and the resulting value is returned to the caller. A later call to the same function will get a fresh new set of local variables. But, what if the local variables weren’t thrown away on exiting a function? What if you could later resume the function where it left off? This is what generators provide; they can be thought of as resumable functions.

Here’s the simplest example of a generator function:

<div class="highlight-default notranslate">

<div class="highlight">

    def generate_ints(N):
        for i in range(N):
            yield i

</div>

</div>

A new keyword, <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a>, was introduced for generators. Any function containing a <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement is a generator function; this is detected by Python’s bytecode compiler which compiles the function specially as a result.

When you call a generator function, it doesn’t return a single value; instead it returns a generator object that supports the iterator protocol. On executing the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement, the generator outputs the value of <span class="pre">`i`</span>, similar to a <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statement. The big difference between <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> and a <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statement is that on reaching a <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> the generator’s state of execution is suspended and local variables are preserved. On the next call to the generator’s <span class="pre">`.next()`</span> method, the function will resume executing immediately after the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement. (For complicated reasons, the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement isn’t allowed inside the <a href="../reference/compound_stmts.html#try" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">try</code></span></a> block of a <a href="../reference/compound_stmts.html#try" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">try</code></span></a>…<a href="../reference/compound_stmts.html#finally" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">finally</code></span></a> statement; read <span id="index-1" class="target"></span><a href="https://www.python.org/dev/peps/pep-0255" class="pep reference external"><strong>PEP 255</strong></a> for a full explanation of the interaction between <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> and exceptions.)

Here’s a sample usage of the <span class="pre">`generate_ints()`</span> generator:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> gen = generate_ints(3)
    >>> gen
    <generator object at 0x8117f90>
    >>> gen.next()
    0
    >>> gen.next()
    1
    >>> gen.next()
    2
    >>> gen.next()
    Traceback (most recent call last):
      File "stdin", line 1, in ?
      File "stdin", line 2, in generate_ints
    StopIteration

</div>

</div>

You could equally write <span class="pre">`for`</span>` `<span class="pre">`i`</span>` `<span class="pre">`in`</span>` `<span class="pre">`generate_ints(5)`</span>, or <span class="pre">`a,b,c`</span>` `<span class="pre">`=`</span>` `<span class="pre">`generate_ints(3)`</span>.

Inside a generator function, the <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statement can only be used without a value, and signals the end of the procession of values; afterwards the generator cannot return any further values. <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> with a value, such as <span class="pre">`return`</span>` `<span class="pre">`5`</span>, is a syntax error inside a generator function. The end of the generator’s results can also be indicated by raising <a href="../library/exceptions.html#exceptions.StopIteration" class="reference internal" title="exceptions.StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a> manually, or by just letting the flow of execution fall off the bottom of the function.

You could achieve the effect of generators manually by writing your own class and storing all the local variables of the generator as instance variables. For example, returning a list of integers could be done by setting <span class="pre">`self.count`</span> to 0, and having the <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a> method increment <span class="pre">`self.count`</span> and return it. However, for a moderately complicated generator, writing a corresponding class would be much messier. <span class="pre">`Lib/test/test_generators.py`</span> contains a number of more interesting examples. The simplest one implements an in-order traversal of a tree using generators recursively.

<div class="highlight-default notranslate">

<div class="highlight">

    # A recursive generator that generates Tree leaves in in-order.
    def inorder(t):
        if t:
            for x in inorder(t.left):
                yield x
            yield t.label
            for x in inorder(t.right):
                yield x

</div>

</div>

Two other examples in <span class="pre">`Lib/test/test_generators.py`</span> produce solutions for the N-Queens problem (placing \$N\$ queens on an \$NxN\$ chess board so that no queen threatens another) and the Knight’s Tour (a route that takes a knight to every square of an \$NxN\$ chessboard without visiting any square twice).

The idea of generators comes from other programming languages, especially Icon (<a href="https://www.cs.arizona.edu/icon/" class="reference external">https://www.cs.arizona.edu/icon/</a>), where the idea of generators is central. In Icon, every expression and function call behaves like a generator. One example from “An Overview of the Icon Programming Language” at <a href="https://www.cs.arizona.edu/icon/docs/ipd266.htm" class="reference external">https://www.cs.arizona.edu/icon/docs/ipd266.htm</a> gives an idea of what this looks like:

<div class="highlight-default notranslate">

<div class="highlight">

    sentence := "Store it in the neighboring harbor"
    if (i := find("or", sentence)) > 5 then write(i)

</div>

</div>

In Icon the <span class="pre">`find()`</span> function returns the indexes at which the substring “or” is found: 3, 23, 33. In the <a href="../reference/compound_stmts.html#if" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">if</code></span></a> statement, <span class="pre">`i`</span> is first assigned a value of 3, but 3 is less than 5, so the comparison fails, and Icon retries it with the second value of 23. 23 is greater than 5, so the comparison now succeeds, and the code prints the value 23 to the screen.

Python doesn’t go nearly as far as Icon in adopting generators as a central concept. Generators are considered part of the core Python language, but learning or using them isn’t compulsory; if they don’t solve any problems that you have, feel free to ignore them. One novel feature of Python’s interface as compared to Icon’s is that a generator’s state is represented as a concrete object (the iterator) that can be passed around to other functions or stored in a data structure.

<div class="admonition seealso">

See also

<span id="index-2" class="target"></span><a href="https://www.python.org/dev/peps/pep-0255" class="pep reference external"><strong>PEP 255</strong></a> - Simple Generators  
Written by Neil Schemenauer, Tim Peters, Magnus Lie Hetland. Implemented mostly by Neil Schemenauer and Tim Peters, with other fixes from the Python Labs crew.

</div>

</div>

<div id="pep-263-source-code-encodings" class="section">

<span id="section-encodings"></span>

## PEP 263: Source Code Encodings<a href="#pep-263-source-code-encodings" class="headerlink" title="Permalink to this headline">¶</a>

Python source files can now be declared as being in different character set encodings. Encodings are declared by including a specially formatted comment in the first or second line of the source file. For example, a UTF-8 file can be declared with:

<div class="highlight-default notranslate">

<div class="highlight">

    #!/usr/bin/env python
    # -*- coding: UTF-8 -*-

</div>

</div>

Without such an encoding declaration, the default encoding used is 7-bit ASCII. Executing or importing modules that contain string literals with 8-bit characters and have no encoding declaration will result in a <a href="../library/exceptions.html#exceptions.DeprecationWarning" class="reference internal" title="exceptions.DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> being signalled by Python 2.3; in 2.4 this will be a syntax error.

The encoding declaration only affects Unicode string literals, which will be converted to Unicode using the specified encoding. Note that Python identifiers are still restricted to ASCII characters, so you can’t have variable names that use characters outside of the usual alphanumerics.

<div class="admonition seealso">

See also

<span id="index-3" class="target"></span><a href="https://www.python.org/dev/peps/pep-0263" class="pep reference external"><strong>PEP 263</strong></a> - Defining Python Source Code Encodings  
Written by Marc-André Lemburg and Martin von Löwis; implemented by Suzuki Hisao and Martin von Löwis.

</div>

</div>

<div id="pep-273-importing-modules-from-zip-archives" class="section">

## PEP 273: Importing Modules from ZIP Archives<a href="#pep-273-importing-modules-from-zip-archives" class="headerlink" title="Permalink to this headline">¶</a>

The new <a href="../library/zipimport.html#module-zipimport" class="reference internal" title="zipimport: support for importing Python modules from ZIP archives."><span class="pre"><code class="sourceCode python">zipimport</code></span></a> module adds support for importing modules from a ZIP-format archive. You don’t need to import the module explicitly; it will be automatically imported if a ZIP archive’s filename is added to <span class="pre">`sys.path`</span>. For example:

<div class="highlight-shell-session notranslate">

<div class="highlight">

    amk@nyman:~/src/python$ unzip -l /tmp/example.zip
    Archive:  /tmp/example.zip
      Length     Date   Time    Name
     --------    ----   ----    ----
         8467  11-26-02 22:30   jwzthreading.py
     --------                   -------
         8467                   1 file
    amk@nyman:~/src/python$ ./python
    Python 2.3 (#1, Aug 1 2003, 19:54:32)
    >>> import sys
    >>> sys.path.insert(0, '/tmp/example.zip')  # Add .zip file to front of path
    >>> import jwzthreading
    >>> jwzthreading.__file__
    '/tmp/example.zip/jwzthreading.py'
    >>>

</div>

</div>

An entry in <span class="pre">`sys.path`</span> can now be the filename of a ZIP archive. The ZIP archive can contain any kind of files, but only files named <span class="pre">`*.py`</span>, <span class="pre">`*.pyc`</span>, or <span class="pre">`*.pyo`</span> can be imported. If an archive only contains <span class="pre">`*.py`</span> files, Python will not attempt to modify the archive by adding the corresponding <span class="pre">`*.pyc`</span> file, meaning that if a ZIP archive doesn’t contain <span class="pre">`*.pyc`</span> files, importing may be rather slow.

A path within the archive can also be specified to only import from a subdirectory; for example, the path <span class="pre">`/tmp/example.zip/lib/`</span> would only import from the <span class="pre">`lib/`</span> subdirectory within the archive.

<div class="admonition seealso">

See also

<span id="index-4" class="target"></span><a href="https://www.python.org/dev/peps/pep-0273" class="pep reference external"><strong>PEP 273</strong></a> - Import Modules from Zip Archives  
Written by James C. Ahlstrom, who also provided an implementation. Python 2.3 follows the specification in <span id="index-5" class="target"></span><a href="https://www.python.org/dev/peps/pep-0273" class="pep reference external"><strong>PEP 273</strong></a>, but uses an implementation written by Just van Rossum that uses the import hooks described in <span id="index-6" class="target"></span><a href="https://www.python.org/dev/peps/pep-0302" class="pep reference external"><strong>PEP 302</strong></a>. See section <a href="#section-pep302" class="reference internal"><span class="std std-ref">PEP 302: New Import Hooks</span></a> for a description of the new import hooks.

</div>

</div>

<div id="pep-277-unicode-file-name-support-for-windows-nt" class="section">

## PEP 277: Unicode file name support for Windows NT<a href="#pep-277-unicode-file-name-support-for-windows-nt" class="headerlink" title="Permalink to this headline">¶</a>

On Windows NT, 2000, and XP, the system stores file names as Unicode strings. Traditionally, Python has represented file names as byte strings, which is inadequate because it renders some file names inaccessible.

Python now allows using arbitrary Unicode strings (within the limitations of the file system) for all functions that expect file names, most notably the <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> built-in function. If a Unicode string is passed to <a href="../library/os.html#os.listdir" class="reference internal" title="os.listdir"><span class="pre"><code class="sourceCode python">os.listdir()</code></span></a>, Python now returns a list of Unicode strings. A new function, <a href="../library/os.html#os.getcwdu" class="reference internal" title="os.getcwdu"><span class="pre"><code class="sourceCode python">os.getcwdu()</code></span></a>, returns the current directory as a Unicode string.

Byte strings still work as file names, and on Windows Python will transparently convert them to Unicode using the <span class="pre">`mbcs`</span> encoding.

Other systems also allow Unicode strings as file names but convert them to byte strings before passing them to the system, which can cause a <a href="../library/exceptions.html#exceptions.UnicodeError" class="reference internal" title="exceptions.UnicodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeError</span></code></span></a> to be raised. Applications can test whether arbitrary Unicode strings are supported as file names by checking <a href="../library/os.path.html#os.path.supports_unicode_filenames" class="reference internal" title="os.path.supports_unicode_filenames"><span class="pre"><code class="sourceCode python">os.path.supports_unicode_filenames</code></span></a>, a Boolean value.

Under MacOS, <a href="../library/os.html#os.listdir" class="reference internal" title="os.listdir"><span class="pre"><code class="sourceCode python">os.listdir()</code></span></a> may now return Unicode filenames.

<div class="admonition seealso">

See also

<span id="index-7" class="target"></span><a href="https://www.python.org/dev/peps/pep-0277" class="pep reference external"><strong>PEP 277</strong></a> - Unicode file name support for Windows NT  
Written by Neil Hodgson; implemented by Neil Hodgson, Martin von Löwis, and Mark Hammond.

</div>

</div>

<div id="pep-278-universal-newline-support" class="section">

<span id="index-8"></span>

## PEP 278: Universal Newline Support<a href="#pep-278-universal-newline-support" class="headerlink" title="Permalink to this headline">¶</a>

The three major operating systems used today are Microsoft Windows, Apple’s Macintosh OS, and the various Unix derivatives. A minor irritation of cross-platform work is that these three platforms all use different characters to mark the ends of lines in text files. Unix uses the linefeed (ASCII character 10), MacOS uses the carriage return (ASCII character 13), and Windows uses a two-character sequence of a carriage return plus a newline.

Python’s file objects can now support end of line conventions other than the one followed by the platform on which Python is running. Opening a file with the mode <span class="pre">`'U'`</span> or <span class="pre">`'rU'`</span> will open a file for reading in <a href="../glossary.html#term-universal-newlines" class="reference internal"><span class="xref std std-term">universal newlines</span></a> mode. All three line ending conventions will be translated to a <span class="pre">`'\n'`</span> in the strings returned by the various file methods such as <span class="pre">`read()`</span> and <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline()</code></span></a>.

Universal newline support is also used when importing modules and when executing a file with the <a href="../library/functions.html#execfile" class="reference internal" title="execfile"><span class="pre"><code class="sourceCode python"><span class="bu">execfile</span>()</code></span></a> function. This means that Python modules can be shared between all three operating systems without needing to convert the line-endings.

This feature can be disabled when compiling Python by specifying the <span class="pre">`--without-universal-newlines`</span> switch when running Python’s **configure** script.

<div class="admonition seealso">

See also

<span id="index-9" class="target"></span><a href="https://www.python.org/dev/peps/pep-0278" class="pep reference external"><strong>PEP 278</strong></a> - Universal Newline Support  
Written and implemented by Jack Jansen.

</div>

</div>

<div id="pep-279-enumerate" class="section">

<span id="section-enumerate"></span>

## PEP 279: enumerate()<a href="#pep-279-enumerate" class="headerlink" title="Permalink to this headline">¶</a>

A new built-in function, <a href="../library/functions.html#enumerate" class="reference internal" title="enumerate"><span class="pre"><code class="sourceCode python"><span class="bu">enumerate</span>()</code></span></a>, will make certain loops a bit clearer. <span class="pre">`enumerate(thing)`</span>, where *thing* is either an iterator or a sequence, returns an iterator that will return <span class="pre">`(0,`</span>` `<span class="pre">`thing[0])`</span>, <span class="pre">`(1,`</span>` `<span class="pre">`thing[1])`</span>, <span class="pre">`(2,`</span>` `<span class="pre">`thing[2])`</span>, and so forth.

A common idiom to change every element of a list looks like this:

<div class="highlight-default notranslate">

<div class="highlight">

    for i in range(len(L)):
        item = L[i]
        # ... compute some result based on item ...
        L[i] = result

</div>

</div>

This can be rewritten using <a href="../library/functions.html#enumerate" class="reference internal" title="enumerate"><span class="pre"><code class="sourceCode python"><span class="bu">enumerate</span>()</code></span></a> as:

<div class="highlight-default notranslate">

<div class="highlight">

    for i, item in enumerate(L):
        # ... compute some result based on item ...
        L[i] = result

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-10" class="target"></span><a href="https://www.python.org/dev/peps/pep-0279" class="pep reference external"><strong>PEP 279</strong></a> - The enumerate() built-in function  
Written and implemented by Raymond D. Hettinger.

</div>

</div>

<div id="pep-282-the-logging-package" class="section">

## PEP 282: The logging Package<a href="#pep-282-the-logging-package" class="headerlink" title="Permalink to this headline">¶</a>

A standard package for writing logs, <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a>, has been added to Python 2.3. It provides a powerful and flexible mechanism for generating logging output which can then be filtered and processed in various ways. A configuration file written in a standard format can be used to control the logging behavior of a program. Python includes handlers that will write log records to standard error or to a file or socket, send them to the system log, or even e-mail them to a particular address; of course, it’s also possible to write your own handler classes.

The <span class="pre">`Logger`</span> class is the primary class. Most application code will deal with one or more <span class="pre">`Logger`</span> objects, each one used by a particular subsystem of the application. Each <span class="pre">`Logger`</span> is identified by a name, and names are organized into a hierarchy using <span class="pre">`.`</span> as the component separator. For example, you might have <span class="pre">`Logger`</span> instances named <span class="pre">`server`</span>, <span class="pre">`server.auth`</span> and <span class="pre">`server.network`</span>. The latter two instances are below <span class="pre">`server`</span> in the hierarchy. This means that if you turn up the verbosity for <span class="pre">`server`</span> or direct <span class="pre">`server`</span> messages to a different handler, the changes will also apply to records logged to <span class="pre">`server.auth`</span> and <span class="pre">`server.network`</span>. There’s also a root <span class="pre">`Logger`</span> that’s the parent of all other loggers.

For simple uses, the <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> package contains some convenience functions that always use the root log:

<div class="highlight-default notranslate">

<div class="highlight">

    import logging

    logging.debug('Debugging information')
    logging.info('Informational message')
    logging.warning('Warning:config file %s not found', 'server.conf')
    logging.error('Error occurred')
    logging.critical('Critical error -- shutting down')

</div>

</div>

This produces the following output:

<div class="highlight-default notranslate">

<div class="highlight">

    WARNING:root:Warning:config file server.conf not found
    ERROR:root:Error occurred
    CRITICAL:root:Critical error -- shutting down

</div>

</div>

In the default configuration, informational and debugging messages are suppressed and the output is sent to standard error. You can enable the display of informational and debugging messages by calling the <span class="pre">`setLevel()`</span> method on the root logger.

Notice the <span class="pre">`warning()`</span> call’s use of string formatting operators; all of the functions for logging messages take the arguments <span class="pre">`(msg,`</span>` `<span class="pre">`arg1,`</span>` `<span class="pre">`arg2,`</span>` `<span class="pre">`...)`</span> and log the string resulting from <span class="pre">`msg`</span>` `<span class="pre">`%`</span>` `<span class="pre">`(arg1,`</span>` `<span class="pre">`arg2,`</span>` `<span class="pre">`...)`</span>.

There’s also an <span class="pre">`exception()`</span> function that records the most recent traceback. Any of the other functions will also record the traceback if you specify a true value for the keyword argument *exc_info*.

<div class="highlight-default notranslate">

<div class="highlight">

    def f():
        try:    1/0
        except: logging.exception('Problem recorded')

    f()

</div>

</div>

This produces the following output:

<div class="highlight-default notranslate">

<div class="highlight">

    ERROR:root:Problem recorded
    Traceback (most recent call last):
      File "t.py", line 6, in f
        1/0
    ZeroDivisionError: integer division or modulo by zero

</div>

</div>

Slightly more advanced programs will use a logger other than the root logger. The <span class="pre">`getLogger(name)`</span> function is used to get a particular log, creating it if it doesn’t exist yet. <span class="pre">`getLogger(None)`</span> returns the root logger.

<div class="highlight-default notranslate">

<div class="highlight">

    log = logging.getLogger('server')
     ...
    log.info('Listening on port %i', port)
     ...
    log.critical('Disk full')
     ...

</div>

</div>

Log records are usually propagated up the hierarchy, so a message logged to <span class="pre">`server.auth`</span> is also seen by <span class="pre">`server`</span> and <span class="pre">`root`</span>, but a <span class="pre">`Logger`</span> can prevent this by setting its <span class="pre">`propagate`</span> attribute to <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a>.

There are more classes provided by the <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> package that can be customized. When a <span class="pre">`Logger`</span> instance is told to log a message, it creates a <span class="pre">`LogRecord`</span> instance that is sent to any number of different <span class="pre">`Handler`</span> instances. Loggers and handlers can also have an attached list of filters, and each filter can cause the <span class="pre">`LogRecord`</span> to be ignored or can modify the record before passing it along. When they’re finally output, <span class="pre">`LogRecord`</span> instances are converted to text by a <span class="pre">`Formatter`</span> class. All of these classes can be replaced by your own specially-written classes.

With all of these features the <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> package should provide enough flexibility for even the most complicated applications. This is only an incomplete overview of its features, so please see the package’s reference documentation for all of the details. Reading <span id="index-11" class="target"></span><a href="https://www.python.org/dev/peps/pep-0282" class="pep reference external"><strong>PEP 282</strong></a> will also be helpful.

<div class="admonition seealso">

See also

<span id="index-12" class="target"></span><a href="https://www.python.org/dev/peps/pep-0282" class="pep reference external"><strong>PEP 282</strong></a> - A Logging System  
Written by Vinay Sajip and Trent Mick; implemented by Vinay Sajip.

</div>

</div>

<div id="pep-285-a-boolean-type" class="section">

<span id="section-bool"></span>

## PEP 285: A Boolean Type<a href="#pep-285-a-boolean-type" class="headerlink" title="Permalink to this headline">¶</a>

A Boolean type was added to Python 2.3. Two new constants were added to the <a href="../library/__builtin__.html#module-__builtin__" class="reference internal" title="__builtin__: The module that provides the built-in namespace."><span class="pre"><code class="sourceCode python">__builtin__</code></span></a> module, <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> and <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a>. (<a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> and <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a> constants were added to the built-ins in Python 2.2.1, but the 2.2.1 versions are simply set to integer values of 1 and 0 and aren’t a different type.)

The type object for this new type is named <a href="../library/functions.html#bool" class="reference internal" title="bool"><span class="pre"><code class="sourceCode python"><span class="bu">bool</span></code></span></a>; the constructor for it takes any Python value and converts it to <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> or <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a>.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> bool(1)
    True
    >>> bool(0)
    False
    >>> bool([])
    False
    >>> bool( (1,) )
    True

</div>

</div>

Most of the standard library modules and built-in functions have been changed to return Booleans.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> obj = []
    >>> hasattr(obj, 'append')
    True
    >>> isinstance(obj, list)
    True
    >>> isinstance(obj, tuple)
    False

</div>

</div>

Python’s Booleans were added with the primary goal of making code clearer. For example, if you’re reading a function and encounter the statement <span class="pre">`return`</span>` `<span class="pre">`1`</span>, you might wonder whether the <span class="pre">`1`</span> represents a Boolean truth value, an index, or a coefficient that multiplies some other quantity. If the statement is <span class="pre">`return`</span>` `<span class="pre">`True`</span>, however, the meaning of the return value is quite clear.

Python’s Booleans were *not* added for the sake of strict type-checking. A very strict language such as Pascal would also prevent you performing arithmetic with Booleans, and would require that the expression in an <a href="../reference/compound_stmts.html#if" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">if</code></span></a> statement always evaluate to a Boolean result. Python is not this strict and never will be, as <span id="index-13" class="target"></span><a href="https://www.python.org/dev/peps/pep-0285" class="pep reference external"><strong>PEP 285</strong></a> explicitly says. This means you can still use any expression in an <a href="../reference/compound_stmts.html#if" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">if</code></span></a> statement, even ones that evaluate to a list or tuple or some random object. The Boolean type is a subclass of the <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> class so that arithmetic using a Boolean still works.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> True + 1
    2
    >>> False + 1
    1
    >>> False * 75
    0
    >>> True * 75
    75

</div>

</div>

To sum up <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> and <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a> in a sentence: they’re alternative ways to spell the integer values 1 and 0, with the single difference that <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> and <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> return the strings <span class="pre">`'True'`</span> and <span class="pre">`'False'`</span> instead of <span class="pre">`'1'`</span> and <span class="pre">`'0'`</span>.

<div class="admonition seealso">

See also

<span id="index-14" class="target"></span><a href="https://www.python.org/dev/peps/pep-0285" class="pep reference external"><strong>PEP 285</strong></a> - Adding a bool type  
Written and implemented by GvR.

</div>

</div>

<div id="pep-293-codec-error-handling-callbacks" class="section">

## PEP 293: Codec Error Handling Callbacks<a href="#pep-293-codec-error-handling-callbacks" class="headerlink" title="Permalink to this headline">¶</a>

When encoding a Unicode string into a byte string, unencodable characters may be encountered. So far, Python has allowed specifying the error processing as either “strict” (raising <a href="../library/exceptions.html#exceptions.UnicodeError" class="reference internal" title="exceptions.UnicodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeError</span></code></span></a>), “ignore” (skipping the character), or “replace” (using a question mark in the output string), with “strict” being the default behavior. It may be desirable to specify alternative processing of such errors, such as inserting an XML character reference or HTML entity reference into the converted string.

Python now has a flexible framework to add different processing strategies. New error handlers can be added with <a href="../library/codecs.html#codecs.register_error" class="reference internal" title="codecs.register_error"><span class="pre"><code class="sourceCode python">codecs.register_error()</code></span></a>, and codecs then can access the error handler with <a href="../library/codecs.html#codecs.lookup_error" class="reference internal" title="codecs.lookup_error"><span class="pre"><code class="sourceCode python">codecs.lookup_error()</code></span></a>. An equivalent C API has been added for codecs written in C. The error handler gets the necessary state information such as the string being converted, the position in the string where the error was detected, and the target encoding. The handler can then either raise an exception or return a replacement string.

Two additional error handlers have been implemented using this framework: “backslashreplace” uses Python backslash quoting to represent unencodable characters and “xmlcharrefreplace” emits XML character references.

<div class="admonition seealso">

See also

<span id="index-15" class="target"></span><a href="https://www.python.org/dev/peps/pep-0293" class="pep reference external"><strong>PEP 293</strong></a> - Codec Error Handling Callbacks  
Written and implemented by Walter Dörwald.

</div>

</div>

<div id="pep-301-package-index-and-metadata-for-distutils" class="section">

<span id="section-pep301"></span>

## PEP 301: Package Index and Metadata for Distutils<a href="#pep-301-package-index-and-metadata-for-distutils" class="headerlink" title="Permalink to this headline">¶</a>

Support for the long-requested Python catalog makes its first appearance in 2.3.

The heart of the catalog is the new Distutils **register** command. Running <span class="pre">`python`</span>` `<span class="pre">`setup.py`</span>` `<span class="pre">`register`</span> will collect the metadata describing a package, such as its name, version, maintainer, description, &c., and send it to a central catalog server. The resulting catalog is available from <a href="https://pypi.org" class="reference external">https://pypi.org</a>.

To make the catalog a bit more useful, a new optional *classifiers* keyword argument has been added to the Distutils <span class="pre">`setup()`</span> function. A list of <a href="http://catb.org/~esr/trove/" class="reference external">Trove</a>-style strings can be supplied to help classify the software.

Here’s an example <span class="pre">`setup.py`</span> with classifiers, written to be compatible with older versions of the Distutils:

<div class="highlight-default notranslate">

<div class="highlight">

    from distutils import core
    kw = {'name': "Quixote",
          'version': "0.5.1",
          'description': "A highly Pythonic Web application framework",
          # ...
          }

    if (hasattr(core, 'setup_keywords') and
        'classifiers' in core.setup_keywords):
        kw['classifiers'] = \
            ['Topic :: Internet :: WWW/HTTP :: Dynamic Content',
             'Environment :: No Input/Output (Daemon)',
             'Intended Audience :: Developers'],

    core.setup(**kw)

</div>

</div>

The full list of classifiers can be obtained by running <span class="pre">`python`</span>` `<span class="pre">`setup.py`</span>` `<span class="pre">`register`</span>` `<span class="pre">`--list-classifiers`</span>.

<div class="admonition seealso">

See also

<span id="index-16" class="target"></span><a href="https://www.python.org/dev/peps/pep-0301" class="pep reference external"><strong>PEP 301</strong></a> - Package Index and Metadata for Distutils  
Written and implemented by Richard Jones.

</div>

</div>

<div id="pep-302-new-import-hooks" class="section">

<span id="section-pep302"></span>

## PEP 302: New Import Hooks<a href="#pep-302-new-import-hooks" class="headerlink" title="Permalink to this headline">¶</a>

While it’s been possible to write custom import hooks ever since the <span class="pre">`ihooks`</span> module was introduced in Python 1.3, no one has ever been really happy with it because writing new import hooks is difficult and messy. There have been various proposed alternatives such as the <a href="../library/imputil.html#module-imputil" class="reference internal" title="imputil: Manage and augment the import process. (deprecated)"><span class="pre"><code class="sourceCode python">imputil</code></span></a> and <span class="pre">`iu`</span> modules, but none of them has ever gained much acceptance, and none of them were easily usable from C code.

<span id="index-17" class="target"></span><a href="https://www.python.org/dev/peps/pep-0302" class="pep reference external"><strong>PEP 302</strong></a> borrows ideas from its predecessors, especially from Gordon McMillan’s <span class="pre">`iu`</span> module. Three new items are added to the <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> module:

- <span class="pre">`sys.path_hooks`</span> is a list of callable objects; most often they’ll be classes. Each callable takes a string containing a path and either returns an importer object that will handle imports from this path or raises an <a href="../library/exceptions.html#exceptions.ImportError" class="reference internal" title="exceptions.ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> exception if it can’t handle this path.

- <span class="pre">`sys.path_importer_cache`</span> caches importer objects for each path, so <span class="pre">`sys.path_hooks`</span> will only need to be traversed once for each path.

- <span class="pre">`sys.meta_path`</span> is a list of importer objects that will be traversed before <span class="pre">`sys.path`</span> is checked. This list is initially empty, but user code can add objects to it. Additional built-in and frozen modules can be imported by an object added to this list.

Importer objects must have a single method, <span class="pre">`find_module(fullname,`</span>` `<span class="pre">`path=None)`</span>. *fullname* will be a module or package name, e.g. <span class="pre">`string`</span> or <span class="pre">`distutils.core`</span>. <span class="pre">`find_module()`</span> must return a loader object that has a single method, <span class="pre">`load_module(fullname)`</span>, that creates and returns the corresponding module object.

Pseudo-code for Python’s new import logic, therefore, looks something like this (simplified a bit; see <span id="index-18" class="target"></span><a href="https://www.python.org/dev/peps/pep-0302" class="pep reference external"><strong>PEP 302</strong></a> for the full details):

<div class="highlight-default notranslate">

<div class="highlight">

    for mp in sys.meta_path:
        loader = mp(fullname)
        if loader is not None:
            <module> = loader.load_module(fullname)

    for path in sys.path:
        for hook in sys.path_hooks:
            try:
                importer = hook(path)
            except ImportError:
                # ImportError, so try the other path hooks
                pass
            else:
                loader = importer.find_module(fullname)
                <module> = loader.load_module(fullname)

    # Not found!
    raise ImportError

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-19" class="target"></span><a href="https://www.python.org/dev/peps/pep-0302" class="pep reference external"><strong>PEP 302</strong></a> - New Import Hooks  
Written by Just van Rossum and Paul Moore. Implemented by Just van Rossum.

</div>

</div>

<div id="pep-305-comma-separated-files" class="section">

<span id="section-pep305"></span>

## PEP 305: Comma-separated Files<a href="#pep-305-comma-separated-files" class="headerlink" title="Permalink to this headline">¶</a>

Comma-separated files are a format frequently used for exporting data from databases and spreadsheets. Python 2.3 adds a parser for comma-separated files.

Comma-separated format is deceptively simple at first glance:

<div class="highlight-default notranslate">

<div class="highlight">

    Costs,150,200,3.95

</div>

</div>

Read a line and call <span class="pre">`line.split(',')`</span>: what could be simpler? But toss in string data that can contain commas, and things get more complicated:

<div class="highlight-default notranslate">

<div class="highlight">

    "Costs",150,200,3.95,"Includes taxes, shipping, and sundry items"

</div>

</div>

A big ugly regular expression can parse this, but using the new <a href="../library/csv.html#module-csv" class="reference internal" title="csv: Write and read tabular data to and from delimited files."><span class="pre"><code class="sourceCode python">csv</code></span></a> package is much simpler:

<div class="highlight-default notranslate">

<div class="highlight">

    import csv

    input = open('datafile', 'rb')
    reader = csv.reader(input)
    for line in reader:
        print line

</div>

</div>

The <span class="pre">`reader()`</span> function takes a number of different options. The field separator isn’t limited to the comma and can be changed to any character, and so can the quoting and line-ending characters.

Different dialects of comma-separated files can be defined and registered; currently there are two dialects, both used by Microsoft Excel. A separate <a href="../library/csv.html#csv.writer" class="reference internal" title="csv.writer"><span class="pre"><code class="sourceCode python">csv.writer</code></span></a> class will generate comma-separated files from a succession of tuples or lists, quoting strings that contain the delimiter.

<div class="admonition seealso">

See also

<span id="index-20" class="target"></span><a href="https://www.python.org/dev/peps/pep-0305" class="pep reference external"><strong>PEP 305</strong></a> - CSV File API  
Written and implemented by Kevin Altis, Dave Cole, Andrew McNamara, Skip Montanaro, Cliff Wells.

</div>

</div>

<div id="pep-307-pickle-enhancements" class="section">

<span id="section-pep307"></span>

## PEP 307: Pickle Enhancements<a href="#pep-307-pickle-enhancements" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> and <a href="../library/pickle.html#module-cPickle" class="reference internal" title="cPickle: Faster version of pickle, but not subclassable."><span class="pre"><code class="sourceCode python">cPickle</code></span></a> modules received some attention during the 2.3 development cycle. In 2.2, new-style classes could be pickled without difficulty, but they weren’t pickled very compactly; <span id="index-21" class="target"></span><a href="https://www.python.org/dev/peps/pep-0307" class="pep reference external"><strong>PEP 307</strong></a> quotes a trivial example where a new-style class results in a pickled string three times longer than that for a classic class.

The solution was to invent a new pickle protocol. The <a href="../library/pickle.html#pickle.dumps" class="reference internal" title="pickle.dumps"><span class="pre"><code class="sourceCode python">pickle.dumps()</code></span></a> function has supported a text-or-binary flag for a long time. In 2.3, this flag is redefined from a Boolean to an integer: 0 is the old text-mode pickle format, 1 is the old binary format, and now 2 is a new 2.3-specific format. A new constant, <a href="../library/pickle.html#pickle.HIGHEST_PROTOCOL" class="reference internal" title="pickle.HIGHEST_PROTOCOL"><span class="pre"><code class="sourceCode python">pickle.HIGHEST_PROTOCOL</code></span></a>, can be used to select the fanciest protocol available.

Unpickling is no longer considered a safe operation. 2.2’s <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> provided hooks for trying to prevent unsafe classes from being unpickled (specifically, a <span class="pre">`__safe_for_unpickling__`</span> attribute), but none of this code was ever audited and therefore it’s all been ripped out in 2.3. You should not unpickle untrusted data in any version of Python.

To reduce the pickling overhead for new-style classes, a new interface for customizing pickling was added using three special methods: <a href="../library/pickle.html#object.__getstate__" class="reference internal" title="object.__getstate__"><span class="pre"><code class="sourceCode python">__getstate__()</code></span></a>, <a href="../library/pickle.html#object.__setstate__" class="reference internal" title="object.__setstate__"><span class="pre"><code class="sourceCode python">__setstate__()</code></span></a>, and <a href="../library/pickle.html#object.__getnewargs__" class="reference internal" title="object.__getnewargs__"><span class="pre"><code class="sourceCode python">__getnewargs__()</code></span></a>. Consult <span id="index-22" class="target"></span><a href="https://www.python.org/dev/peps/pep-0307" class="pep reference external"><strong>PEP 307</strong></a> for the full semantics of these methods.

As a way to compress pickles yet further, it’s now possible to use integer codes instead of long strings to identify pickled classes. The Python Software Foundation will maintain a list of standardized codes; there’s also a range of codes for private use. Currently no codes have been specified.

<div class="admonition seealso">

See also

<span id="index-23" class="target"></span><a href="https://www.python.org/dev/peps/pep-0307" class="pep reference external"><strong>PEP 307</strong></a> - Extensions to the pickle protocol  
Written and implemented by Guido van Rossum and Tim Peters.

</div>

</div>

<div id="extended-slices" class="section">

<span id="section-slices"></span>

## Extended Slices<a href="#extended-slices" class="headerlink" title="Permalink to this headline">¶</a>

Ever since Python 1.4, the slicing syntax has supported an optional third “step” or “stride” argument. For example, these are all legal Python syntax: <span class="pre">`L[1:10:2]`</span>, <span class="pre">`L[:-1:1]`</span>, <span class="pre">`L[::-1]`</span>. This was added to Python at the request of the developers of Numerical Python, which uses the third argument extensively. However, Python’s built-in list, tuple, and string sequence types have never supported this feature, raising a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> if you tried it. Michael Hudson contributed a patch to fix this shortcoming.

For example, you can now easily extract the elements of a list that have even indexes:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> L = range(10)
    >>> L[::2]
    [0, 2, 4, 6, 8]

</div>

</div>

Negative values also work to make a copy of the same list in reverse order:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> L[::-1]
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

</div>

</div>

This also works for tuples, arrays, and strings:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> s='abcd'
    >>> s[::2]
    'ac'
    >>> s[::-1]
    'dcba'

</div>

</div>

If you have a mutable sequence such as a list or an array you can assign to or delete an extended slice, but there are some differences between assignment to extended and regular slices. Assignment to a regular slice can be used to change the length of the sequence:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> a = range(3)
    >>> a
    [0, 1, 2]
    >>> a[1:3] = [4, 5, 6]
    >>> a
    [0, 4, 5, 6]

</div>

</div>

Extended slices aren’t this flexible. When assigning to an extended slice, the list on the right hand side of the statement must contain the same number of items as the slice it is replacing:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> a = range(4)
    >>> a
    [0, 1, 2, 3]
    >>> a[::2]
    [0, 2]
    >>> a[::2] = [0, -1]
    >>> a
    [0, 1, -1, 3]
    >>> a[::2] = [0,1,2]
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    ValueError: attempt to assign sequence of size 3 to extended slice of size 2

</div>

</div>

Deletion is more straightforward:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> a = range(4)
    >>> a
    [0, 1, 2, 3]
    >>> a[::2]
    [0, 2]
    >>> del a[::2]
    >>> a
    [1, 3]

</div>

</div>

One can also now pass slice objects to the <a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a> methods of the built-in sequences:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> range(10).__getitem__(slice(0, 5, 2))
    [0, 2, 4]

</div>

</div>

Or use slice objects directly in subscripts:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> range(10)[slice(0, 5, 2)]
    [0, 2, 4]

</div>

</div>

To simplify implementing sequences that support extended slicing, slice objects now have a method <span class="pre">`indices(length)`</span> which, given the length of a sequence, returns a <span class="pre">`(start,`</span>` `<span class="pre">`stop,`</span>` `<span class="pre">`step)`</span> tuple that can be passed directly to <a href="../library/functions.html#range" class="reference internal" title="range"><span class="pre"><code class="sourceCode python"><span class="bu">range</span>()</code></span></a>. <span class="pre">`indices()`</span> handles omitted and out-of-bounds indices in a manner consistent with regular slices (and this innocuous phrase hides a welter of confusing details!). The method is intended to be used like this:

<div class="highlight-default notranslate">

<div class="highlight">

    class FakeSeq:
        ...
        def calc_item(self, i):
            ...
        def __getitem__(self, item):
            if isinstance(item, slice):
                indices = item.indices(len(self))
                return FakeSeq([self.calc_item(i) for i in range(*indices)])
            else:
                return self.calc_item(i)

</div>

</div>

From this example you can also see that the built-in <a href="../library/functions.html#slice" class="reference internal" title="slice"><span class="pre"><code class="sourceCode python"><span class="bu">slice</span></code></span></a> object is now the type object for the slice type, and is no longer a function. This is consistent with Python 2.2, where <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>, <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, etc., underwent the same change.

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Permalink to this headline">¶</a>

Here are all of the changes that Python 2.3 makes to the core Python language.

- The <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement is now always a keyword, as described in section <a href="#section-generators" class="reference internal"><span class="std std-ref">PEP 255: Simple Generators</span></a> of this document.

- A new built-in function <a href="../library/functions.html#enumerate" class="reference internal" title="enumerate"><span class="pre"><code class="sourceCode python"><span class="bu">enumerate</span>()</code></span></a> was added, as described in section <a href="#section-enumerate" class="reference internal"><span class="std std-ref">PEP 279: enumerate()</span></a> of this document.

- Two new constants, <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> and <a href="../library/constants.html#False" class="reference internal" title="False"><span class="pre"><code class="sourceCode python"><span class="va">False</span></code></span></a> were added along with the built-in <a href="../library/functions.html#bool" class="reference internal" title="bool"><span class="pre"><code class="sourceCode python"><span class="bu">bool</span></code></span></a> type, as described in section <a href="#section-bool" class="reference internal"><span class="std std-ref">PEP 285: A Boolean Type</span></a> of this document.

- The <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a> type constructor will now return a long integer instead of raising an <a href="../library/exceptions.html#exceptions.OverflowError" class="reference internal" title="exceptions.OverflowError"><span class="pre"><code class="sourceCode python"><span class="pp">OverflowError</span></code></span></a> when a string or floating-point number is too large to fit into an integer. This can lead to the paradoxical result that <span class="pre">`isinstance(int(expression),`</span>` `<span class="pre">`int)`</span> is false, but that seems unlikely to cause problems in practice.

- Built-in types now support the extended slicing syntax, as described in section <a href="#section-slices" class="reference internal"><span class="std std-ref">Extended Slices</span></a> of this document.

- A new built-in function, <span class="pre">`sum(iterable,`</span>` `<span class="pre">`start=0)`</span>, adds up the numeric items in the iterable object and returns their sum. <a href="../library/functions.html#sum" class="reference internal" title="sum"><span class="pre"><code class="sourceCode python"><span class="bu">sum</span>()</code></span></a> only accepts numbers, meaning that you can’t use it to concatenate a bunch of strings. (Contributed by Alex Martelli.)

- <span class="pre">`list.insert(pos,`</span>` `<span class="pre">`value)`</span> used to insert *value* at the front of the list when *pos* was negative. The behaviour has now been changed to be consistent with slice indexing, so when *pos* is -1 the value will be inserted before the last element, and so forth.

- <span class="pre">`list.index(value)`</span>, which searches for *value* within the list and returns its index, now takes optional *start* and *stop* arguments to limit the search to only part of the list.

- Dictionaries have a new method, <span class="pre">`pop(key[,`</span>` `<span class="pre">`*default*])`</span>, that returns the value corresponding to *key* and removes that key/value pair from the dictionary. If the requested key isn’t present in the dictionary, *default* is returned if it’s specified and <a href="../library/exceptions.html#exceptions.KeyError" class="reference internal" title="exceptions.KeyError"><span class="pre"><code class="sourceCode python"><span class="pp">KeyError</span></code></span></a> raised if it isn’t.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> d = {1:2}
      >>> d
      {1: 2}
      >>> d.pop(4)
      Traceback (most recent call last):
        File "stdin", line 1, in ?
      KeyError: 4
      >>> d.pop(1)
      2
      >>> d.pop(1)
      Traceback (most recent call last):
        File "stdin", line 1, in ?
      KeyError: 'pop(): dictionary is empty'
      >>> d
      {}
      >>>

  </div>

  </div>

  There’s also a new class method, <span class="pre">`dict.fromkeys(iterable,`</span>` `<span class="pre">`value)`</span>, that creates a dictionary with keys taken from the supplied iterator *iterable* and all values set to *value*, defaulting to <span class="pre">`None`</span>.

  (Patches contributed by Raymond Hettinger.)

  Also, the <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>()</code></span></a> constructor now accepts keyword arguments to simplify creating small dictionaries:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> dict(red=1, blue=2, green=3, black=4)
      {'blue': 2, 'black': 4, 'green': 3, 'red': 1}

  </div>

  </div>

  (Contributed by Just van Rossum.)

- The <a href="../reference/simple_stmts.html#assert" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">assert</code></span></a> statement no longer checks the <span class="pre">`__debug__`</span> flag, so you can no longer disable assertions by assigning to <span class="pre">`__debug__`</span>. Running Python with the <a href="../using/cmdline.html#cmdoption-o" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-O</code></span></a> switch will still generate code that doesn’t execute any assertions.

- Most type objects are now callable, so you can use them to create new objects such as functions, classes, and modules. (This means that the <a href="../library/new.html#module-new" class="reference internal" title="new: Interface to the creation of runtime implementation objects. (deprecated)"><span class="pre"><code class="sourceCode python">new</code></span></a> module can be deprecated in a future Python version, because you can now use the type objects available in the <a href="../library/types.html#module-types" class="reference internal" title="types: Names for built-in types."><span class="pre"><code class="sourceCode python">types</code></span></a> module.) For example, you can create a new module object with the following code:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import types
      >>> m = types.ModuleType('abc','docstring')
      >>> m
      <module 'abc' (built-in)>
      >>> m.__doc__
      'docstring'

  </div>

  </div>

- A new warning, <a href="../library/exceptions.html#exceptions.PendingDeprecationWarning" class="reference internal" title="exceptions.PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a> was added to indicate features which are in the process of being deprecated. The warning will *not* be printed by default. To check for use of features that will be deprecated in the future, supply <a href="../using/cmdline.html#cmdoption-w" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-Walways::PendingDeprecationWarning::</code></span></a> on the command line or use <a href="../library/warnings.html#warnings.filterwarnings" class="reference internal" title="warnings.filterwarnings"><span class="pre"><code class="sourceCode python">warnings.filterwarnings()</code></span></a>.

- The process of deprecating string-based exceptions, as in <span class="pre">`raise`</span>` `<span class="pre">`"Error`</span>` `<span class="pre">`occurred"`</span>, has begun. Raising a string will now trigger <a href="../library/exceptions.html#exceptions.PendingDeprecationWarning" class="reference internal" title="exceptions.PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a>.

- Using <span class="pre">`None`</span> as a variable name will now result in a <a href="../library/exceptions.html#exceptions.SyntaxWarning" class="reference internal" title="exceptions.SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a> warning. In a future version of Python, <span class="pre">`None`</span> may finally become a keyword.

- The <span class="pre">`xreadlines()`</span> method of file objects, introduced in Python 2.1, is no longer necessary because files now behave as their own iterator. <span class="pre">`xreadlines()`</span> was originally introduced as a faster way to loop over all the lines in a file, but now you can simply write <span class="pre">`for`</span>` `<span class="pre">`line`</span>` `<span class="pre">`in`</span>` `<span class="pre">`file_obj`</span>. File objects also have a new read-only <span class="pre">`encoding`</span> attribute that gives the encoding used by the file; Unicode strings written to the file will be automatically converted to bytes using the given encoding.

- The method resolution order used by new-style classes has changed, though you’ll only notice the difference if you have a really complicated inheritance hierarchy. Classic classes are unaffected by this change. Python 2.2 originally used a topological sort of a class’s ancestors, but 2.3 now uses the C3 algorithm as described in the paper <a href="http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.3910" class="reference external">“A Monotonic Superclass Linearization for Dylan”</a>. To understand the motivation for this change, read Michele Simionato’s article <a href="http://www.phyast.pitt.edu/~micheles/mro.html" class="reference external">“Python 2.3 Method Resolution Order”</a>, or read the thread on python-dev starting with the message at <a href="https://mail.python.org/pipermail/python-dev/2002-October/029035.html" class="reference external">https://mail.python.org/pipermail/python-dev/2002-October/029035.html</a>. Samuele Pedroni first pointed out the problem and also implemented the fix by coding the C3 algorithm.

- Python runs multithreaded programs by switching between threads after executing N bytecodes. The default value for N has been increased from 10 to 100 bytecodes, speeding up single-threaded applications by reducing the switching overhead. Some multithreaded applications may suffer slower response time, but that’s easily fixed by setting the limit back to a lower number using <span class="pre">`sys.setcheckinterval(N)`</span>. The limit can be retrieved with the new <a href="../library/sys.html#sys.getcheckinterval" class="reference internal" title="sys.getcheckinterval"><span class="pre"><code class="sourceCode python">sys.getcheckinterval()</code></span></a> function.

- One minor but far-reaching change is that the names of extension types defined by the modules included with Python now contain the module and a <span class="pre">`'.'`</span> in front of the type name. For example, in Python 2.2, if you created a socket and printed its <span class="pre">`__class__`</span>, you’d get this output:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> s = socket.socket()
      >>> s.__class__
      <type 'socket'>

  </div>

  </div>

  In 2.3, you get this:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> s.__class__
      <type '_socket.socket'>

  </div>

  </div>

- One of the noted incompatibilities between old- and new-style classes has been removed: you can now assign to the <a href="../library/stdtypes.html#definition.__name__" class="reference internal" title="definition.__name__"><span class="pre"><code class="sourceCode python"><span class="va">__name__</span></code></span></a> and <a href="../library/stdtypes.html#class.__bases__" class="reference internal" title="class.__bases__"><span class="pre"><code class="sourceCode python">__bases__</code></span></a> attributes of new-style classes. There are some restrictions on what can be assigned to <a href="../library/stdtypes.html#class.__bases__" class="reference internal" title="class.__bases__"><span class="pre"><code class="sourceCode python">__bases__</code></span></a> along the lines of those relating to assigning to an instance’s <a href="../library/stdtypes.html#instance.__class__" class="reference internal" title="instance.__class__"><span class="pre"><code class="sourceCode python"><span class="va">__class__</span></code></span></a> attribute.

<div id="string-changes" class="section">

### String Changes<a href="#string-changes" class="headerlink" title="Permalink to this headline">¶</a>

- The <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> operator now works differently for strings. Previously, when evaluating <span class="pre">`X`</span>` `<span class="pre">`in`</span>` `<span class="pre">`Y`</span> where *X* and *Y* are strings, *X* could only be a single character. That’s now changed; *X* can be a string of any length, and <span class="pre">`X`</span>` `<span class="pre">`in`</span>` `<span class="pre">`Y`</span> will return <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> if *X* is a substring of *Y*. If *X* is the empty string, the result is always <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a>.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> 'ab' in 'abcd'
      True
      >>> 'ad' in 'abcd'
      False
      >>> '' in 'abcd'
      True

  </div>

  </div>

  Note that this doesn’t tell you where the substring starts; if you need that information, use the <span class="pre">`find()`</span> string method.

- The <span class="pre">`strip()`</span>, <span class="pre">`lstrip()`</span>, and <span class="pre">`rstrip()`</span> string methods now have an optional argument for specifying the characters to strip. The default is still to remove all whitespace characters:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> '   abc '.strip()
      'abc'
      >>> '><><abc<><><>'.strip('<>')
      'abc'
      >>> '><><abc<><><>\n'.strip('<>')
      'abc<><><>\n'
      >>> u'\u4000\u4001abc\u4000'.strip(u'\u4000')
      u'\u4001abc'
      >>>

  </div>

  </div>

  (Suggested by Simon Brunning and implemented by Walter Dörwald.)

- The <span class="pre">`startswith()`</span> and <span class="pre">`endswith()`</span> string methods now accept negative numbers for the *start* and *end* parameters.

- Another new string method is <span class="pre">`zfill()`</span>, originally a function in the <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module. <span class="pre">`zfill()`</span> pads a numeric string with zeros on the left until it’s the specified width. Note that the <span class="pre">`%`</span> operator is still more flexible and powerful than <span class="pre">`zfill()`</span>.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> '45'.zfill(4)
      '0045'
      >>> '12345'.zfill(4)
      '12345'
      >>> 'goofy'.zfill(6)
      '0goofy'

  </div>

  </div>

  (Contributed by Walter Dörwald.)

- A new type object, <a href="../library/functions.html#basestring" class="reference internal" title="basestring"><span class="pre"><code class="sourceCode python"><span class="bu">basestring</span></code></span></a>, has been added. Both 8-bit strings and Unicode strings inherit from this type, so <span class="pre">`isinstance(obj,`</span>` `<span class="pre">`basestring)`</span> will return <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> for either kind of string. It’s a completely abstract type, so you can’t create <a href="../library/functions.html#basestring" class="reference internal" title="basestring"><span class="pre"><code class="sourceCode python"><span class="bu">basestring</span></code></span></a> instances.

- Interned strings are no longer immortal and will now be garbage-collected in the usual way when the only reference to them is from the internal dictionary of interned strings. (Implemented by Oren Tirosh.)

</div>

<div id="optimizations" class="section">

### Optimizations<a href="#optimizations" class="headerlink" title="Permalink to this headline">¶</a>

- The creation of new-style class instances has been made much faster; they’re now faster than classic classes!

- The <span class="pre">`sort()`</span> method of list objects has been extensively rewritten by Tim Peters, and the implementation is significantly faster.

- Multiplication of large long integers is now much faster thanks to an implementation of Karatsuba multiplication, an algorithm that scales better than the O(n\*n) required for the grade-school multiplication algorithm. (Original patch by Christopher A. Craig, and significantly reworked by Tim Peters.)

- The <span class="pre">`SET_LINENO`</span> opcode is now gone. This may provide a small speed increase, depending on your compiler’s idiosyncrasies. See section <a href="#section-other" class="reference internal"><span class="std std-ref">Other Changes and Fixes</span></a> for a longer explanation. (Removed by Michael Hudson.)

- <a href="../library/functions.html#xrange" class="reference internal" title="xrange"><span class="pre"><code class="sourceCode python"><span class="bu">xrange</span>()</code></span></a> objects now have their own iterator, making <span class="pre">`for`</span>` `<span class="pre">`i`</span>` `<span class="pre">`in`</span>` `<span class="pre">`xrange(n)`</span> slightly faster than <span class="pre">`for`</span>` `<span class="pre">`i`</span>` `<span class="pre">`in`</span>` `<span class="pre">`range(n)`</span>. (Patch by Raymond Hettinger.)

- A number of small rearrangements have been made in various hotspots to improve performance, such as inlining a function or removing some code. (Implemented mostly by GvR, but lots of people have contributed single changes.)

The net result of the 2.3 optimizations is that Python 2.3 runs the pystone benchmark around 25% faster than Python 2.2.

</div>

</div>

<div id="new-improved-and-deprecated-modules" class="section">

## New, Improved, and Deprecated Modules<a href="#new-improved-and-deprecated-modules" class="headerlink" title="Permalink to this headline">¶</a>

As usual, Python’s standard library received a number of enhancements and bug fixes. Here’s a partial list of the most notable changes, sorted alphabetically by module name. Consult the <span class="pre">`Misc/NEWS`</span> file in the source tree for a more complete list of changes, or look through the CVS logs for all the details.

- The <a href="../library/array.html#module-array" class="reference internal" title="array: Space efficient arrays of uniformly typed numeric values."><span class="pre"><code class="sourceCode python">array</code></span></a> module now supports arrays of Unicode characters using the <span class="pre">`'u'`</span> format character. Arrays also now support using the <span class="pre">`+=`</span> assignment operator to add another array’s contents, and the <span class="pre">`*=`</span> assignment operator to repeat an array. (Contributed by Jason Orendorff.)

- The <a href="../library/bsddb.html#module-bsddb" class="reference internal" title="bsddb: Interface to Berkeley DB database library"><span class="pre"><code class="sourceCode python">bsddb</code></span></a> module has been replaced by version 4.1.6 of the <a href="http://pybsddb.sourceforge.net" class="reference external">PyBSDDB</a> package, providing a more complete interface to the transactional features of the BerkeleyDB library.

  The old version of the module has been renamed to <span class="pre">`bsddb185`</span> and is no longer built automatically; you’ll have to edit <span class="pre">`Modules/Setup`</span> to enable it. Note that the new <a href="../library/bsddb.html#module-bsddb" class="reference internal" title="bsddb: Interface to Berkeley DB database library"><span class="pre"><code class="sourceCode python">bsddb</code></span></a> package is intended to be compatible with the old module, so be sure to file bugs if you discover any incompatibilities. When upgrading to Python 2.3, if the new interpreter is compiled with a new version of the underlying BerkeleyDB library, you will almost certainly have to convert your database files to the new version. You can do this fairly easily with the new scripts <span class="pre">`db2pickle.py`</span> and <span class="pre">`pickle2db.py`</span> which you will find in the distribution’s <span class="pre">`Tools/scripts`</span> directory. If you’ve already been using the PyBSDDB package and importing it as <span class="pre">`bsddb3`</span>, you will have to change your <span class="pre">`import`</span> statements to import it as <a href="../library/bsddb.html#module-bsddb" class="reference internal" title="bsddb: Interface to Berkeley DB database library"><span class="pre"><code class="sourceCode python">bsddb</code></span></a>.

- The new <a href="../library/bz2.html#module-bz2" class="reference internal" title="bz2: Interface to compression and decompression routines compatible with bzip2."><span class="pre"><code class="sourceCode python">bz2</code></span></a> module is an interface to the bz2 data compression library. bz2-compressed data is usually smaller than corresponding <a href="../library/zlib.html#module-zlib" class="reference internal" title="zlib: Low-level interface to compression and decompression routines compatible with gzip."><span class="pre"><code class="sourceCode python">zlib</code></span></a>-compressed data. (Contributed by Gustavo Niemeyer.)

- A set of standard date/time types has been added in the new <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> module. See the following section for more details.

- The Distutils <span class="pre">`Extension`</span> class now supports an extra constructor argument named *depends* for listing additional source files that an extension depends on. This lets Distutils recompile the module if any of the dependency files are modified. For example, if <span class="pre">`sampmodule.c`</span> includes the header file <span class="pre">`sample.h`</span>, you would create the <span class="pre">`Extension`</span> object like this:

  <div class="highlight-default notranslate">

  <div class="highlight">

      ext = Extension("samp",
                      sources=["sampmodule.c"],
                      depends=["sample.h"])

  </div>

  </div>

  Modifying <span class="pre">`sample.h`</span> would then cause the module to be recompiled. (Contributed by Jeremy Hylton.)

- Other minor changes to Distutils: it now checks for the <span id="index-24" class="target"></span><span class="pre">`CC`</span>, <span id="index-25" class="target"></span><span class="pre">`CFLAGS`</span>, <span id="index-26" class="target"></span><span class="pre">`CPP`</span>, <span id="index-27" class="target"></span><span class="pre">`LDFLAGS`</span>, and <span id="index-28" class="target"></span><span class="pre">`CPPFLAGS`</span> environment variables, using them to override the settings in Python’s configuration (contributed by Robert Weber).

- Previously the <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> module would only search the docstrings of public methods and functions for test cases, but it now also examines private ones as well. The <span class="pre">`DocTestSuite()`</span> function creates a <a href="../library/unittest.html#unittest.TestSuite" class="reference internal" title="unittest.TestSuite"><span class="pre"><code class="sourceCode python">unittest.TestSuite</code></span></a> object from a set of <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> tests.

- The new <span class="pre">`gc.get_referents(object)`</span> function returns a list of all the objects referenced by *object*.

- The <a href="../library/getopt.html#module-getopt" class="reference internal" title="getopt: Portable parser for command line options; support both short and long option names."><span class="pre"><code class="sourceCode python">getopt</code></span></a> module gained a new function, <span class="pre">`gnu_getopt()`</span>, that supports the same arguments as the existing <a href="../library/getopt.html#module-getopt" class="reference internal" title="getopt: Portable parser for command line options; support both short and long option names."><span class="pre"><code class="sourceCode python">getopt()</code></span></a> function but uses GNU-style scanning mode. The existing <a href="../library/getopt.html#module-getopt" class="reference internal" title="getopt: Portable parser for command line options; support both short and long option names."><span class="pre"><code class="sourceCode python">getopt()</code></span></a> stops processing options as soon as a non-option argument is encountered, but in GNU-style mode processing continues, meaning that options and arguments can be mixed. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> getopt.getopt(['-f', 'filename', 'output', '-v'], 'f:v')
      ([('-f', 'filename')], ['output', '-v'])
      >>> getopt.gnu_getopt(['-f', 'filename', 'output', '-v'], 'f:v')
      ([('-f', 'filename'), ('-v', '')], ['output'])

  </div>

  </div>

  (Contributed by Peter Åstrand.)

- The <a href="../library/grp.html#module-grp" class="reference internal" title="grp: The group database (getgrnam() and friends). (Unix)"><span class="pre"><code class="sourceCode python">grp</code></span></a>, <a href="../library/pwd.html#module-pwd" class="reference internal" title="pwd: The password database (getpwnam() and friends). (Unix)"><span class="pre"><code class="sourceCode python">pwd</code></span></a>, and <a href="../library/resource.html#module-resource" class="reference internal" title="resource: An interface to provide resource usage information on the current process. (Unix)"><span class="pre"><code class="sourceCode python">resource</code></span></a> modules now return enhanced tuples:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import grp
      >>> g = grp.getgrnam('amk')
      >>> g.gr_name, g.gr_gid
      ('amk', 500)

  </div>

  </div>

- The <a href="../library/gzip.html#module-gzip" class="reference internal" title="gzip: Interfaces for gzip compression and decompression using file objects."><span class="pre"><code class="sourceCode python">gzip</code></span></a> module can now handle files exceeding 2 GiB.

- The new <a href="../library/heapq.html#module-heapq" class="reference internal" title="heapq: Heap queue algorithm (a.k.a. priority queue)."><span class="pre"><code class="sourceCode python">heapq</code></span></a> module contains an implementation of a heap queue algorithm. A heap is an array-like data structure that keeps items in a partially sorted order such that, for every index *k*, <span class="pre">`heap[k]`</span>` `<span class="pre">`<=`</span>` `<span class="pre">`heap[2*k+1]`</span> and <span class="pre">`heap[k]`</span>` `<span class="pre">`<=`</span>` `<span class="pre">`heap[2*k+2]`</span>. This makes it quick to remove the smallest item, and inserting a new item while maintaining the heap property is O(lg n). (See <a href="https://xlinux.nist.gov/dads//HTML/priorityque.html" class="reference external">https://xlinux.nist.gov/dads//HTML/priorityque.html</a> for more information about the priority queue data structure.)

  The <a href="../library/heapq.html#module-heapq" class="reference internal" title="heapq: Heap queue algorithm (a.k.a. priority queue)."><span class="pre"><code class="sourceCode python">heapq</code></span></a> module provides <span class="pre">`heappush()`</span> and <span class="pre">`heappop()`</span> functions for adding and removing items while maintaining the heap property on top of some other mutable Python sequence type. Here’s an example that uses a Python list:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import heapq
      >>> heap = []
      >>> for item in [3, 7, 5, 11, 1]:
      ...    heapq.heappush(heap, item)
      ...
      >>> heap
      [1, 3, 5, 11, 7]
      >>> heapq.heappop(heap)
      1
      >>> heapq.heappop(heap)
      3
      >>> heap
      [5, 7, 11]

  </div>

  </div>

  (Contributed by Kevin O’Connor.)

- The IDLE integrated development environment has been updated using the code from the IDLEfork project (<a href="http://idlefork.sourceforge.net" class="reference external">http://idlefork.sourceforge.net</a>). The most notable feature is that the code being developed is now executed in a subprocess, meaning that there’s no longer any need for manual <span class="pre">`reload()`</span> operations. IDLE’s core code has been incorporated into the standard library as the <span class="pre">`idlelib`</span> package.

- The <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a> module now supports IMAP over SSL. (Contributed by Piers Lauder and Tino Lange.)

- The <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> contains a number of useful functions for use with iterators, inspired by various functions provided by the ML and Haskell languages. For example, <span class="pre">`itertools.ifilter(predicate,`</span>` `<span class="pre">`iterator)`</span> returns all elements in the iterator for which the function <span class="pre">`predicate()`</span> returns <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a>, and <span class="pre">`itertools.repeat(obj,`</span>` `<span class="pre">`N)`</span> returns <span class="pre">`obj`</span> *N* times. There are a number of other functions in the module; see the package’s reference documentation for details. (Contributed by Raymond Hettinger.)

- Two new functions in the <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> module, <span class="pre">`degrees(rads)`</span> and <span class="pre">`radians(degs)`</span>, convert between radians and degrees. Other functions in the <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> module such as <a href="../library/math.html#math.sin" class="reference internal" title="math.sin"><span class="pre"><code class="sourceCode python">math.sin()</code></span></a> and <a href="../library/math.html#math.cos" class="reference internal" title="math.cos"><span class="pre"><code class="sourceCode python">math.cos()</code></span></a> have always required input values measured in radians. Also, an optional *base* argument was added to <a href="../library/math.html#math.log" class="reference internal" title="math.log"><span class="pre"><code class="sourceCode python">math.log()</code></span></a> to make it easier to compute logarithms for bases other than <span class="pre">`e`</span> and <span class="pre">`10`</span>. (Contributed by Raymond Hettinger.)

- Several new POSIX functions (<span class="pre">`getpgid()`</span>, <span class="pre">`killpg()`</span>, <span class="pre">`lchown()`</span>, <span class="pre">`loadavg()`</span>, <span class="pre">`major()`</span>, <span class="pre">`makedev()`</span>, <span class="pre">`minor()`</span>, and <span class="pre">`mknod()`</span>) were added to the <a href="../library/posix.html#module-posix" class="reference internal" title="posix: The most common POSIX system calls (normally used via module os). (Unix)"><span class="pre"><code class="sourceCode python">posix</code></span></a> module that underlies the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module. (Contributed by Gustavo Niemeyer, Geert Jansen, and Denis S. Otkidach.)

- In the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module, the <span class="pre">`*stat()`</span> family of functions can now report fractions of a second in a timestamp. Such time stamps are represented as floats, similar to the value returned by <a href="../library/time.html#time.time" class="reference internal" title="time.time"><span class="pre"><code class="sourceCode python">time.time()</code></span></a>.

  During testing, it was found that some applications will break if time stamps are floats. For compatibility, when using the tuple interface of the <span class="pre">`stat_result`</span> time stamps will be represented as integers. When using named fields (a feature first introduced in Python 2.2), time stamps are still represented as integers, unless <a href="../library/os.html#os.stat_float_times" class="reference internal" title="os.stat_float_times"><span class="pre"><code class="sourceCode python">os.stat_float_times()</code></span></a> is invoked to enable float return values:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> os.stat("/tmp").st_mtime
      1034791200
      >>> os.stat_float_times(True)
      >>> os.stat("/tmp").st_mtime
      1034791200.6335014

  </div>

  </div>

  In Python 2.4, the default will change to always returning floats.

  Application developers should enable this feature only if all their libraries work properly when confronted with floating point time stamps, or if they use the tuple API. If used, the feature should be activated on an application level instead of trying to enable it on a per-use basis.

- The <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a> module contains a new parser for command-line arguments that can convert option values to a particular Python type and will automatically generate a usage message. See the following section for more details.

- The old and never-documented <span class="pre">`linuxaudiodev`</span> module has been deprecated, and a new version named <a href="../library/ossaudiodev.html#module-ossaudiodev" class="reference internal" title="ossaudiodev: Access to OSS-compatible audio devices. (Linux, FreeBSD)"><span class="pre"><code class="sourceCode python">ossaudiodev</code></span></a> has been added. The module was renamed because the OSS sound drivers can be used on platforms other than Linux, and the interface has also been tidied and brought up to date in various ways. (Contributed by Greg Ward and Nicholas FitzRoy-Dale.)

- The new <a href="../library/platform.html#module-platform" class="reference internal" title="platform: Retrieves as much platform identifying data as possible."><span class="pre"><code class="sourceCode python">platform</code></span></a> module contains a number of functions that try to determine various properties of the platform you’re running on. There are functions for getting the architecture, CPU type, the Windows OS version, and even the Linux distribution version. (Contributed by Marc-André Lemburg.)

- The parser objects provided by the <span class="pre">`pyexpat`</span> module can now optionally buffer character data, resulting in fewer calls to your character data handler and therefore faster performance. Setting the parser object’s <span class="pre">`buffer_text`</span> attribute to <a href="../library/constants.html#True" class="reference internal" title="True"><span class="pre"><code class="sourceCode python"><span class="va">True</span></code></span></a> will enable buffering.

- The <span class="pre">`sample(population,`</span>` `<span class="pre">`k)`</span> function was added to the <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module. *population* is a sequence or <a href="../library/functions.html#xrange" class="reference internal" title="xrange"><span class="pre"><code class="sourceCode python"><span class="bu">xrange</span></code></span></a> object containing the elements of a population, and <span class="pre">`sample()`</span> chooses *k* elements from the population without replacing chosen elements. *k* can be any value up to <span class="pre">`len(population)`</span>. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> days = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'St', 'Sn']
      >>> random.sample(days, 3)      # Choose 3 elements
      ['St', 'Sn', 'Th']
      >>> random.sample(days, 7)      # Choose 7 elements
      ['Tu', 'Th', 'Mo', 'We', 'St', 'Fr', 'Sn']
      >>> random.sample(days, 7)      # Choose 7 again
      ['We', 'Mo', 'Sn', 'Fr', 'Tu', 'St', 'Th']
      >>> random.sample(days, 8)      # Can't choose eight
      Traceback (most recent call last):
        File "<stdin>", line 1, in ?
        File "random.py", line 414, in sample
            raise ValueError, "sample larger than population"
      ValueError: sample larger than population
      >>> random.sample(xrange(1,10000,2), 10)   # Choose ten odd nos. under 10000
      [3407, 3805, 1505, 7023, 2401, 2267, 9733, 3151, 8083, 9195]

  </div>

  </div>

  The <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module now uses a new algorithm, the Mersenne Twister, implemented in C. It’s faster and more extensively studied than the previous algorithm.

  (All changes contributed by Raymond Hettinger.)

- The <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline</code></span></a> module also gained a number of new functions: <span class="pre">`get_history_item()`</span>, <span class="pre">`get_current_history_length()`</span>, and <span class="pre">`redisplay()`</span>.

- The <a href="../library/rexec.html#module-rexec" class="reference internal" title="rexec: Basic restricted execution framework. (deprecated)"><span class="pre"><code class="sourceCode python">rexec</code></span></a> and <a href="../library/bastion.html#module-Bastion" class="reference internal" title="Bastion: Providing restricted access to objects. (deprecated)"><span class="pre"><code class="sourceCode python">Bastion</code></span></a> modules have been declared dead, and attempts to import them will fail with a <a href="../library/exceptions.html#exceptions.RuntimeError" class="reference internal" title="exceptions.RuntimeError"><span class="pre"><code class="sourceCode python"><span class="pp">RuntimeError</span></code></span></a>. New-style classes provide new ways to break out of the restricted execution environment provided by <a href="../library/rexec.html#module-rexec" class="reference internal" title="rexec: Basic restricted execution framework. (deprecated)"><span class="pre"><code class="sourceCode python">rexec</code></span></a>, and no one has interest in fixing them or time to do so. If you have applications using <a href="../library/rexec.html#module-rexec" class="reference internal" title="rexec: Basic restricted execution framework. (deprecated)"><span class="pre"><code class="sourceCode python">rexec</code></span></a>, rewrite them to use something else.

  (Sticking with Python 2.2 or 2.1 will not make your applications any safer because there are known bugs in the <a href="../library/rexec.html#module-rexec" class="reference internal" title="rexec: Basic restricted execution framework. (deprecated)"><span class="pre"><code class="sourceCode python">rexec</code></span></a> module in those versions. To repeat: if you’re using <a href="../library/rexec.html#module-rexec" class="reference internal" title="rexec: Basic restricted execution framework. (deprecated)"><span class="pre"><code class="sourceCode python">rexec</code></span></a>, stop using it immediately.)

- The <span class="pre">`rotor`</span> module has been deprecated because the algorithm it uses for encryption is not believed to be secure. If you need encryption, use one of the several AES Python modules that are available separately.

- The <a href="../library/shutil.html#module-shutil" class="reference internal" title="shutil: High-level file operations, including copying."><span class="pre"><code class="sourceCode python">shutil</code></span></a> module gained a <span class="pre">`move(src,`</span>` `<span class="pre">`dest)`</span> function that recursively moves a file or directory to a new location.

- Support for more advanced POSIX signal handling was added to the <a href="../library/signal.html#module-signal" class="reference internal" title="signal: Set handlers for asynchronous events."><span class="pre"><code class="sourceCode python">signal</code></span></a> but then removed again as it proved impossible to make it work reliably across platforms.

- The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module now supports timeouts. You can call the <span class="pre">`settimeout(t)`</span> method on a socket object to set a timeout of *t* seconds. Subsequent socket operations that take longer than *t* seconds to complete will abort and raise a <a href="../library/socket.html#socket.timeout" class="reference internal" title="socket.timeout"><span class="pre"><code class="sourceCode python">socket.timeout</code></span></a> exception.

  The original timeout implementation was by Tim O’Malley. Michael Gilfix integrated it into the Python <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module and shepherded it through a lengthy review. After the code was checked in, Guido van Rossum rewrote parts of it. (This is a good example of a collaborative development process in action.)

- On Windows, the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module now ships with Secure Sockets Layer (SSL) support.

- The value of the C <span class="pre">`PYTHON_API_VERSION`</span> macro is now exposed at the Python level as <span class="pre">`sys.api_version`</span>. The current exception can be cleared by calling the new <a href="../library/sys.html#sys.exc_clear" class="reference internal" title="sys.exc_clear"><span class="pre"><code class="sourceCode python">sys.exc_clear()</code></span></a> function.

- The new <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module allows reading from and writing to **tar**-format archive files. (Contributed by Lars Gustäbel.)

- The new <a href="../library/textwrap.html#module-textwrap" class="reference internal" title="textwrap: Text wrapping and filling"><span class="pre"><code class="sourceCode python">textwrap</code></span></a> module contains functions for wrapping strings containing paragraphs of text. The <span class="pre">`wrap(text,`</span>` `<span class="pre">`width)`</span> function takes a string and returns a list containing the text split into lines of no more than the chosen width. The <span class="pre">`fill(text,`</span>` `<span class="pre">`width)`</span> function returns a single string, reformatted to fit into lines no longer than the chosen width. (As you can guess, <span class="pre">`fill()`</span> is built on top of <span class="pre">`wrap()`</span>. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import textwrap
      >>> paragraph = "Not a whit, we defy augury: ... more text ..."
      >>> textwrap.wrap(paragraph, 60)
      ["Not a whit, we defy augury: there's a special providence in",
       "the fall of a sparrow. If it be now, 'tis not to come; if it",
       ...]
      >>> print textwrap.fill(paragraph, 35)
      Not a whit, we defy augury: there's
      a special providence in the fall of
      a sparrow. If it be now, 'tis not
      to come; if it be not to come, it
      will be now; if it be not now, yet
      it will come: the readiness is all.
      >>>

  </div>

  </div>

  The module also contains a <span class="pre">`TextWrapper`</span> class that actually implements the text wrapping strategy. Both the <span class="pre">`TextWrapper`</span> class and the <span class="pre">`wrap()`</span> and <span class="pre">`fill()`</span> functions support a number of additional keyword arguments for fine-tuning the formatting; consult the module’s documentation for details. (Contributed by Greg Ward.)

- The <a href="../library/thread.html#module-thread" class="reference internal" title="thread: Create multiple threads of control within one interpreter."><span class="pre"><code class="sourceCode python">thread</code></span></a> and <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> modules now have companion modules, <a href="../library/dummy_thread.html#module-dummy_thread" class="reference internal" title="dummy_thread: Drop-in replacement for the thread module."><span class="pre"><code class="sourceCode python">dummy_thread</code></span></a> and <a href="../library/dummy_threading.html#module-dummy_threading" class="reference internal" title="dummy_threading: Drop-in replacement for the threading module."><span class="pre"><code class="sourceCode python">dummy_threading</code></span></a>, that provide a do-nothing implementation of the <a href="../library/thread.html#module-thread" class="reference internal" title="thread: Create multiple threads of control within one interpreter."><span class="pre"><code class="sourceCode python">thread</code></span></a> module’s interface for platforms where threads are not supported. The intention is to simplify thread-aware modules (ones that *don’t* rely on threads to run) by putting the following code at the top:

  <div class="highlight-default notranslate">

  <div class="highlight">

      try:
          import threading as _threading
      except ImportError:
          import dummy_threading as _threading

  </div>

  </div>

  In this example, <span class="pre">`_threading`</span> is used as the module name to make it clear that the module being used is not necessarily the actual <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> module. Code can call functions and use classes in <span class="pre">`_threading`</span> whether or not threads are supported, avoiding an <a href="../reference/compound_stmts.html#if" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">if</code></span></a> statement and making the code slightly clearer. This module will not magically make multithreaded code run without threads; code that waits for another thread to return or to do something will simply hang forever.

- The <a href="../library/time.html#module-time" class="reference internal" title="time: Time access and conversions."><span class="pre"><code class="sourceCode python">time</code></span></a> module’s <span class="pre">`strptime()`</span> function has long been an annoyance because it uses the platform C library’s <span class="pre">`strptime()`</span> implementation, and different platforms sometimes have odd bugs. Brett Cannon contributed a portable implementation that’s written in pure Python and should behave identically on all platforms.

- The new <a href="../library/timeit.html#module-timeit" class="reference internal" title="timeit: Measure the execution time of small code snippets."><span class="pre"><code class="sourceCode python">timeit</code></span></a> module helps measure how long snippets of Python code take to execute. The <span class="pre">`timeit.py`</span> file can be run directly from the command line, or the module’s <span class="pre">`Timer`</span> class can be imported and used directly. Here’s a short example that figures out whether it’s faster to convert an 8-bit string to Unicode by appending an empty Unicode string to it or by using the <a href="../library/functions.html#unicode" class="reference internal" title="unicode"><span class="pre"><code class="sourceCode python"><span class="bu">unicode</span>()</code></span></a> function:

  <div class="highlight-default notranslate">

  <div class="highlight">

      import timeit

      timer1 = timeit.Timer('unicode("abc")')
      timer2 = timeit.Timer('"abc" + u""')

      # Run three trials
      print timer1.repeat(repeat=3, number=100000)
      print timer2.repeat(repeat=3, number=100000)

      # On my laptop this outputs:
      # [0.36831796169281006, 0.37441694736480713, 0.35304892063140869]
      # [0.17574405670166016, 0.18193507194519043, 0.17565798759460449]

  </div>

  </div>

- The <a href="../library/tix.html#module-Tix" class="reference internal" title="Tix: Tk Extension Widgets for Tkinter"><span class="pre"><code class="sourceCode python">Tix</code></span></a> module has received various bug fixes and updates for the current version of the Tix package.

- The <a href="../library/tkinter.html#module-Tkinter" class="reference internal" title="Tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">Tkinter</code></span></a> module now works with a thread-enabled version of Tcl. Tcl’s threading model requires that widgets only be accessed from the thread in which they’re created; accesses from another thread can cause Tcl to panic. For certain Tcl interfaces, <a href="../library/tkinter.html#module-Tkinter" class="reference internal" title="Tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">Tkinter</code></span></a> will now automatically avoid this when a widget is accessed from a different thread by marshalling a command, passing it to the correct thread, and waiting for the results. Other interfaces can’t be handled automatically but <a href="../library/tkinter.html#module-Tkinter" class="reference internal" title="Tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">Tkinter</code></span></a> will now raise an exception on such an access so that you can at least find out about the problem. See <a href="https://mail.python.org/pipermail/python-dev/2002-December/031107.html" class="reference external">https://mail.python.org/pipermail/python-dev/2002-December/031107.html</a> for a more detailed explanation of this change. (Implemented by Martin von Löwis.)

- Calling Tcl methods through <span class="pre">`_tkinter`</span> no longer returns only strings. Instead, if Tcl returns other objects those objects are converted to their Python equivalent, if one exists, or wrapped with a <span class="pre">`_tkinter.Tcl_Obj`</span> object if no Python equivalent exists. This behavior can be controlled through the <span class="pre">`wantobjects()`</span> method of <span class="pre">`tkapp`</span> objects.

  When using <span class="pre">`_tkinter`</span> through the <a href="../library/tkinter.html#module-Tkinter" class="reference internal" title="Tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">Tkinter</code></span></a> module (as most Tkinter applications will), this feature is always activated. It should not cause compatibility problems, since Tkinter would always convert string results to Python types where possible.

  If any incompatibilities are found, the old behavior can be restored by setting the <span class="pre">`wantobjects`</span> variable in the <a href="../library/tkinter.html#module-Tkinter" class="reference internal" title="Tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">Tkinter</code></span></a> module to false before creating the first <span class="pre">`tkapp`</span> object.

  <div class="highlight-default notranslate">

  <div class="highlight">

      import Tkinter
      Tkinter.wantobjects = 0

  </div>

  </div>

  Any breakage caused by this change should be reported as a bug.

- The <a href="../library/userdict.html#module-UserDict" class="reference internal" title="UserDict: Class wrapper for dictionary objects."><span class="pre"><code class="sourceCode python">UserDict</code></span></a> module has a new <span class="pre">`DictMixin`</span> class which defines all dictionary methods for classes that already have a minimum mapping interface. This greatly simplifies writing classes that need to be substitutable for dictionaries, such as the classes in the <a href="../library/shelve.html#module-shelve" class="reference internal" title="shelve: Python object persistence."><span class="pre"><code class="sourceCode python">shelve</code></span></a> module.

  Adding the mix-in as a superclass provides the full dictionary interface whenever the class defines <a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a>, <a href="../reference/datamodel.html#object.__setitem__" class="reference internal" title="object.__setitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__setitem__</span>()</code></span></a>, <a href="../reference/datamodel.html#object.__delitem__" class="reference internal" title="object.__delitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__delitem__</span>()</code></span></a>, and <span class="pre">`keys()`</span>. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import UserDict
      >>> class SeqDict(UserDict.DictMixin):
      ...     """Dictionary lookalike implemented with lists."""
      ...     def __init__(self):
      ...         self.keylist = []
      ...         self.valuelist = []
      ...     def __getitem__(self, key):
      ...         try:
      ...             i = self.keylist.index(key)
      ...         except ValueError:
      ...             raise KeyError
      ...         return self.valuelist[i]
      ...     def __setitem__(self, key, value):
      ...         try:
      ...             i = self.keylist.index(key)
      ...             self.valuelist[i] = value
      ...         except ValueError:
      ...             self.keylist.append(key)
      ...             self.valuelist.append(value)
      ...     def __delitem__(self, key):
      ...         try:
      ...             i = self.keylist.index(key)
      ...         except ValueError:
      ...             raise KeyError
      ...         self.keylist.pop(i)
      ...         self.valuelist.pop(i)
      ...     def keys(self):
      ...         return list(self.keylist)
      ...
      >>> s = SeqDict()
      >>> dir(s)      # See that other dictionary methods are implemented
      ['__cmp__', '__contains__', '__delitem__', '__doc__', '__getitem__',
       '__init__', '__iter__', '__len__', '__module__', '__repr__',
       '__setitem__', 'clear', 'get', 'has_key', 'items', 'iteritems',
       'iterkeys', 'itervalues', 'keylist', 'keys', 'pop', 'popitem',
       'setdefault', 'update', 'valuelist', 'values']

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- The DOM implementation in <a href="../library/xml.dom.minidom.html#module-xml.dom.minidom" class="reference internal" title="xml.dom.minidom: Minimal Document Object Model (DOM) implementation."><span class="pre"><code class="sourceCode python">xml.dom.minidom</code></span></a> can now generate XML output in a particular encoding by providing an optional encoding argument to the <span class="pre">`toxml()`</span> and <span class="pre">`toprettyxml()`</span> methods of DOM nodes.

- The <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a> module now supports an XML-RPC extension for handling nil data values such as Python’s <span class="pre">`None`</span>. Nil values are always supported on unmarshalling an XML-RPC response. To generate requests containing <span class="pre">`None`</span>, you must supply a true value for the *allow_none* parameter when creating a <span class="pre">`Marshaller`</span> instance.

- The new <a href="../library/docxmlrpcserver.html#module-DocXMLRPCServer" class="reference internal" title="DocXMLRPCServer: Self-documenting XML-RPC server implementation."><span class="pre"><code class="sourceCode python">DocXMLRPCServer</code></span></a> module allows writing self-documenting XML-RPC servers. Run it in demo mode (as a program) to see it in action. Pointing the Web browser to the RPC server produces pydoc-style documentation; pointing xmlrpclib to the server allows invoking the actual methods. (Contributed by Brian Quinlan.)

- Support for internationalized domain names (RFCs 3454, 3490, 3491, and 3492) has been added. The “idna” encoding can be used to convert between a Unicode domain name and the ASCII-compatible encoding (ACE) of that name.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >{}>{}> u"www.Alliancefrançaise.nu".encode("idna")
      'www.xn--alliancefranaise-npb.nu'

  </div>

  </div>

  The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module has also been extended to transparently convert Unicode hostnames to the ACE version before passing them to the C library. Modules that deal with hostnames such as <a href="../library/httplib.html#module-httplib" class="reference internal" title="httplib: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">httplib</code></span></a> and <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a>) also support Unicode host names; <a href="../library/httplib.html#module-httplib" class="reference internal" title="httplib: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">httplib</code></span></a> also sends HTTP <span class="pre">`Host`</span> headers using the ACE version of the domain name. <a href="../library/urllib.html#module-urllib" class="reference internal" title="urllib: Open an arbitrary network resource by URL (requires sockets)."><span class="pre"><code class="sourceCode python">urllib</code></span></a> supports Unicode URLs with non-ASCII host names as long as the <span class="pre">`path`</span> part of the URL is ASCII only.

  To implement this change, the <a href="../library/stringprep.html#module-stringprep" class="reference internal" title="stringprep: String preparation, as per RFC 3453"><span class="pre"><code class="sourceCode python">stringprep</code></span></a> module, the <span class="pre">`mkstringprep`</span> tool and the <span class="pre">`punycode`</span> encoding have been added.

<div id="date-time-type" class="section">

### Date/Time Type<a href="#date-time-type" class="headerlink" title="Permalink to this headline">¶</a>

Date and time types suitable for expressing timestamps were added as the <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> module. The types don’t support different calendars or many fancy features, and just stick to the basics of representing time.

The three primary types are: <span class="pre">`date`</span>, representing a day, month, and year; <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">time</code></span></a>, consisting of hour, minute, and second; and <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a>, which contains all the attributes of both <span class="pre">`date`</span> and <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">time</code></span></a>. There’s also a <span class="pre">`timedelta`</span> class representing differences between two points in time, and time zone logic is implemented by classes inheriting from the abstract <span class="pre">`tzinfo`</span> class.

You can create instances of <span class="pre">`date`</span> and <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">time</code></span></a> by either supplying keyword arguments to the appropriate constructor, e.g. <span class="pre">`datetime.date(year=1972,`</span>` `<span class="pre">`month=10,`</span>` `<span class="pre">`day=15)`</span>, or by using one of a number of class methods. For example, the <span class="pre">`date.today()`</span> class method returns the current local date.

Once created, instances of the date/time classes are all immutable. There are a number of methods for producing formatted strings from objects:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> import datetime
    >>> now = datetime.datetime.now()
    >>> now.isoformat()
    '2002-12-30T21:27:03.994956'
    >>> now.ctime()  # Only available on date, datetime
    'Mon Dec 30 21:27:03 2002'
    >>> now.strftime('%Y %d %b')
    '2002 30 Dec'

</div>

</div>

The <span class="pre">`replace()`</span> method allows modifying one or more fields of a <span class="pre">`date`</span> or <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> instance, returning a new instance:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> d = datetime.datetime.now()
    >>> d
    datetime.datetime(2002, 12, 30, 22, 15, 38, 827738)
    >>> d.replace(year=2001, hour = 12)
    datetime.datetime(2001, 12, 30, 12, 15, 38, 827738)
    >>>

</div>

</div>

Instances can be compared, hashed, and converted to strings (the result is the same as that of <span class="pre">`isoformat()`</span>). <span class="pre">`date`</span> and <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> instances can be subtracted from each other, and added to <span class="pre">`timedelta`</span> instances. The largest missing feature is that there’s no standard library support for parsing strings and getting back a <span class="pre">`date`</span> or <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a>.

For more information, refer to the module’s reference documentation. (Contributed by Tim Peters.)

</div>

<div id="the-optparse-module" class="section">

### The optparse Module<a href="#the-optparse-module" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/getopt.html#module-getopt" class="reference internal" title="getopt: Portable parser for command line options; support both short and long option names."><span class="pre"><code class="sourceCode python">getopt</code></span></a> module provides simple parsing of command-line arguments. The new <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a> module (originally named Optik) provides more elaborate command-line parsing that follows the Unix conventions, automatically creates the output for <span class="pre">`--help`</span>, and can perform different actions for different options.

You start by creating an instance of <span class="pre">`OptionParser`</span> and telling it what your program’s options are.

<div class="highlight-default notranslate">

<div class="highlight">

    import sys
    from optparse import OptionParser

    op = OptionParser()
    op.add_option('-i', '--input',
                  action='store', type='string', dest='input',
                  help='set input filename')
    op.add_option('-l', '--length',
                  action='store', type='int', dest='length',
                  help='set maximum length of output')

</div>

</div>

Parsing a command line is then done by calling the <span class="pre">`parse_args()`</span> method.

<div class="highlight-default notranslate">

<div class="highlight">

    options, args = op.parse_args(sys.argv[1:])
    print options
    print args

</div>

</div>

This returns an object containing all of the option values, and a list of strings containing the remaining arguments.

Invoking the script with the various arguments now works as you’d expect it to. Note that the length argument is automatically converted to an integer.

<div class="highlight-shell-session notranslate">

<div class="highlight">

    $ ./python opt.py -i data arg1
    <Values at 0x400cad4c: {'input': 'data', 'length': None}>
    ['arg1']
    $ ./python opt.py --input=data --length=4
    <Values at 0x400cad2c: {'input': 'data', 'length': 4}>
    []
    $

</div>

</div>

The help message is automatically generated for you:

<div class="highlight-shell-session notranslate">

<div class="highlight">

    $ ./python opt.py --help
    usage: opt.py [options]

    options:
      -h, --help            show this help message and exit
      -iINPUT, --input=INPUT
                            set input filename
      -lLENGTH, --length=LENGTH
                            set maximum length of output
    $

</div>

</div>

See the module’s documentation for more details.

Optik was written by Greg Ward, with suggestions from the readers of the Getopt SIG.

</div>

</div>

<div id="pymalloc-a-specialized-object-allocator" class="section">

<span id="section-pymalloc"></span>

## Pymalloc: A Specialized Object Allocator<a href="#pymalloc-a-specialized-object-allocator" class="headerlink" title="Permalink to this headline">¶</a>

Pymalloc, a specialized object allocator written by Vladimir Marangozov, was a feature added to Python 2.1. Pymalloc is intended to be faster than the system <span class="pre">`malloc()`</span> and to have less memory overhead for allocation patterns typical of Python programs. The allocator uses C’s <span class="pre">`malloc()`</span> function to get large pools of memory and then fulfills smaller memory requests from these pools.

In 2.1 and 2.2, pymalloc was an experimental feature and wasn’t enabled by default; you had to explicitly enable it when compiling Python by providing the <span class="pre">`--with-pymalloc`</span> option to the **configure** script. In 2.3, pymalloc has had further enhancements and is now enabled by default; you’ll have to supply <span class="pre">`--without-pymalloc`</span> to disable it.

This change is transparent to code written in Python; however, pymalloc may expose bugs in C extensions. Authors of C extension modules should test their code with pymalloc enabled, because some incorrect code may cause core dumps at runtime.

There’s one particularly common error that causes problems. There are a number of memory allocation functions in Python’s C API that have previously just been aliases for the C library’s <span class="pre">`malloc()`</span> and <span class="pre">`free()`</span>, meaning that if you accidentally called mismatched functions the error wouldn’t be noticeable. When the object allocator is enabled, these functions aren’t aliases of <span class="pre">`malloc()`</span> and <span class="pre">`free()`</span> any more, and calling the wrong function to free memory may get you a core dump. For example, if memory was allocated using <a href="../c-api/memory.html#c.PyObject_Malloc" class="reference internal" title="PyObject_Malloc"><span class="pre"><code class="sourceCode c">PyObject_Malloc<span class="op">()</span></code></span></a>, it has to be freed using <a href="../c-api/memory.html#c.PyObject_Free" class="reference internal" title="PyObject_Free"><span class="pre"><code class="sourceCode c">PyObject_Free<span class="op">()</span></code></span></a>, not <span class="pre">`free()`</span>. A few modules included with Python fell afoul of this and had to be fixed; doubtless there are more third-party modules that will have the same problem.

As part of this change, the confusing multiple interfaces for allocating memory have been consolidated down into two API families. Memory allocated with one family must not be manipulated with functions from the other family. There is one family for allocating chunks of memory and another family of functions specifically for allocating Python objects.

- To allocate and free an undistinguished chunk of memory use the “raw memory” family: <a href="../c-api/memory.html#c.PyMem_Malloc" class="reference internal" title="PyMem_Malloc"><span class="pre"><code class="sourceCode c">PyMem_Malloc<span class="op">()</span></code></span></a>, <a href="../c-api/memory.html#c.PyMem_Realloc" class="reference internal" title="PyMem_Realloc"><span class="pre"><code class="sourceCode c">PyMem_Realloc<span class="op">()</span></code></span></a>, and <a href="../c-api/memory.html#c.PyMem_Free" class="reference internal" title="PyMem_Free"><span class="pre"><code class="sourceCode c">PyMem_Free<span class="op">()</span></code></span></a>.

- The “object memory” family is the interface to the pymalloc facility described above and is biased towards a large number of “small” allocations: <a href="../c-api/memory.html#c.PyObject_Malloc" class="reference internal" title="PyObject_Malloc"><span class="pre"><code class="sourceCode c">PyObject_Malloc<span class="op">()</span></code></span></a>, <a href="../c-api/memory.html#c.PyObject_Realloc" class="reference internal" title="PyObject_Realloc"><span class="pre"><code class="sourceCode c">PyObject_Realloc<span class="op">()</span></code></span></a>, and <a href="../c-api/memory.html#c.PyObject_Free" class="reference internal" title="PyObject_Free"><span class="pre"><code class="sourceCode c">PyObject_Free<span class="op">()</span></code></span></a>.

- To allocate and free Python objects, use the “object” family <a href="../c-api/allocation.html#c.PyObject_New" class="reference internal" title="PyObject_New"><span class="pre"><code class="sourceCode c">PyObject_New<span class="op">()</span></code></span></a>, <a href="../c-api/allocation.html#c.PyObject_NewVar" class="reference internal" title="PyObject_NewVar"><span class="pre"><code class="sourceCode c">PyObject_NewVar<span class="op">()</span></code></span></a>, and <a href="../c-api/allocation.html#c.PyObject_Del" class="reference internal" title="PyObject_Del"><span class="pre"><code class="sourceCode c">PyObject_Del<span class="op">()</span></code></span></a>.

Thanks to lots of work by Tim Peters, pymalloc in 2.3 also provides debugging features to catch memory overwrites and doubled frees in both extension modules and in the interpreter itself. To enable this support, compile a debugging version of the Python interpreter by running **configure** with <span class="pre">`--with-pydebug`</span>.

To aid extension writers, a header file <span class="pre">`Misc/pymemcompat.h`</span> is distributed with the source to Python 2.3 that allows Python extensions to use the 2.3 interfaces to memory allocation while compiling against any version of Python since 1.5.2. You would copy the file from Python’s source distribution and bundle it with the source of your extension.

<div class="admonition seealso">

See also

<a href="https://hg.python.org/cpython/file/default/Objects/obmalloc.c" class="reference external">https://hg.python.org/cpython/file/default/Objects/obmalloc.c</a>  
For the full details of the pymalloc implementation, see the comments at the top of the file <span class="pre">`Objects/obmalloc.c`</span> in the Python source code. The above link points to the file within the python.org SVN browser.

</div>

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Permalink to this headline">¶</a>

Changes to Python’s build process and to the C API include:

- The cycle detection implementation used by the garbage collection has proven to be stable, so it’s now been made mandatory. You can no longer compile Python without it, and the <span class="pre">`--with-cycle-gc`</span> switch to **configure** has been removed.

- Python can now optionally be built as a shared library (<span class="pre">`libpython2.3.so`</span>) by supplying <span class="pre">`--enable-shared`</span> when running Python’s **configure** script. (Contributed by Ondrej Palkovsky.)

- The <span class="pre">`DL_EXPORT`</span> and <span class="pre">`DL_IMPORT`</span> macros are now deprecated. Initialization functions for Python extension modules should now be declared using the new macro <span class="pre">`PyMODINIT_FUNC`</span>, while the Python core will generally use the <span class="pre">`PyAPI_FUNC`</span> and <span class="pre">`PyAPI_DATA`</span> macros.

- The interpreter can be compiled without any docstrings for the built-in functions and modules by supplying <span class="pre">`--without-doc-strings`</span> to the **configure** script. This makes the Python executable about 10% smaller, but will also mean that you can’t get help for Python’s built-ins. (Contributed by Gustavo Niemeyer.)

- The <span class="pre">`PyArg_NoArgs()`</span> macro is now deprecated, and code that uses it should be changed. For Python 2.2 and later, the method definition table can specify the <a href="../c-api/structures.html#METH_NOARGS" class="reference internal" title="METH_NOARGS"><span class="pre"><code class="sourceCode python">METH_NOARGS</code></span></a> flag, signalling that there are no arguments, and the argument checking can then be removed. If compatibility with pre-2.2 versions of Python is important, the code could use <span class="pre">`PyArg_ParseTuple(args,`</span>` `<span class="pre">`"")`</span> instead, but this will be slower than using <a href="../c-api/structures.html#METH_NOARGS" class="reference internal" title="METH_NOARGS"><span class="pre"><code class="sourceCode python">METH_NOARGS</code></span></a>.

- <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple<span class="op">()</span></code></span></a> accepts new format characters for various sizes of unsigned integers: <span class="pre">`B`</span> for <span class="pre">`unsigned`</span>` `<span class="pre">`char`</span>, <span class="pre">`H`</span> for <span class="pre">`unsigned`</span>` `<span class="pre">`short`</span>` `<span class="pre">`int`</span>, <span class="pre">`I`</span> for <span class="pre">`unsigned`</span>` `<span class="pre">`int`</span>, and <span class="pre">`K`</span> for <span class="pre">`unsigned`</span>` `<span class="pre">`long`</span>` `<span class="pre">`long`</span>.

- A new function, <span class="pre">`PyObject_DelItemString(mapping,`</span>` `<span class="pre">`char`</span>` `<span class="pre">`*key)`</span> was added as shorthand for <span class="pre">`PyObject_DelItem(mapping,`</span>` `<span class="pre">`PyString_New(key))`</span>.

- File objects now manage their internal string buffer differently, increasing it exponentially when needed. This results in the benchmark tests in <span class="pre">`Lib/test/test_bufio.py`</span> speeding up considerably (from 57 seconds to 1.7 seconds, according to one measurement).

- It’s now possible to define class and static methods for a C extension type by setting either the <a href="../c-api/structures.html#METH_CLASS" class="reference internal" title="METH_CLASS"><span class="pre"><code class="sourceCode python">METH_CLASS</code></span></a> or <a href="../c-api/structures.html#METH_STATIC" class="reference internal" title="METH_STATIC"><span class="pre"><code class="sourceCode python">METH_STATIC</code></span></a> flags in a method’s <a href="../c-api/structures.html#c.PyMethodDef" class="reference internal" title="PyMethodDef"><span class="pre"><code class="sourceCode c">PyMethodDef</code></span></a> structure.

- Python now includes a copy of the Expat XML parser’s source code, removing any dependence on a system version or local installation of Expat.

- If you dynamically allocate type objects in your extension, you should be aware of a change in the rules relating to the <span class="pre">`__module__`</span> and <a href="../library/stdtypes.html#definition.__name__" class="reference internal" title="definition.__name__"><span class="pre"><code class="sourceCode python"><span class="va">__name__</span></code></span></a> attributes. In summary, you will want to ensure the type’s dictionary contains a <span class="pre">`'__module__'`</span> key; making the module name the part of the type name leading up to the final period will no longer have the desired effect. For more detail, read the API reference documentation or the source.

<div id="port-specific-changes" class="section">

### Port-Specific Changes<a href="#port-specific-changes" class="headerlink" title="Permalink to this headline">¶</a>

Support for a port to IBM’s OS/2 using the EMX runtime environment was merged into the main Python source tree. EMX is a POSIX emulation layer over the OS/2 system APIs. The Python port for EMX tries to support all the POSIX-like capability exposed by the EMX runtime, and mostly succeeds; <span class="pre">`fork()`</span> and <a href="../library/fcntl.html#module-fcntl" class="reference internal" title="fcntl: The fcntl() and ioctl() system calls. (Unix)"><span class="pre"><code class="sourceCode python">fcntl()</code></span></a> are restricted by the limitations of the underlying emulation layer. The standard OS/2 port, which uses IBM’s Visual Age compiler, also gained support for case-sensitive import semantics as part of the integration of the EMX port into CVS. (Contributed by Andrew MacIntyre.)

On MacOS, most toolbox modules have been weaklinked to improve backward compatibility. This means that modules will no longer fail to load if a single routine is missing on the current OS version. Instead calling the missing routine will raise an exception. (Contributed by Jack Jansen.)

The RPM spec files, found in the <span class="pre">`Misc/RPM/`</span> directory in the Python source distribution, were updated for 2.3. (Contributed by Sean Reifschneider.)

Other new platforms now supported by Python include AtheOS (<a href="http://atheos.cx/" class="reference external">http://atheos.cx/</a>), GNU/Hurd, and OpenVMS.

</div>

</div>

<div id="other-changes-and-fixes" class="section">

<span id="section-other"></span>

## Other Changes and Fixes<a href="#other-changes-and-fixes" class="headerlink" title="Permalink to this headline">¶</a>

As usual, there were a bunch of other improvements and bugfixes scattered throughout the source tree. A search through the CVS change logs finds there were 523 patches applied and 514 bugs fixed between Python 2.2 and 2.3. Both figures are likely to be underestimates.

Some of the more notable changes are:

- If the <span id="index-29" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONINSPECT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONINSPECT</code></span></a> environment variable is set, the Python interpreter will enter the interactive prompt after running a Python program, as if Python had been invoked with the <a href="../using/cmdline.html#cmdoption-i" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-i</code></span></a> option. The environment variable can be set before running the Python interpreter, or it can be set by the Python program as part of its execution.

- The <span class="pre">`regrtest.py`</span> script now provides a way to allow “all resources except *foo*.” A resource name passed to the <span class="pre">`-u`</span> option can now be prefixed with a hyphen (<span class="pre">`'-'`</span>) to mean “remove this resource.” For example, the option ‘<span class="pre">`-uall,-bsddb`</span>’ could be used to enable the use of all resources except <span class="pre">`bsddb`</span>.

- The tools used to build the documentation now work under Cygwin as well as Unix.

- The <span class="pre">`SET_LINENO`</span> opcode has been removed. Back in the mists of time, this opcode was needed to produce line numbers in tracebacks and support trace functions (for, e.g., <a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a>). Since Python 1.5, the line numbers in tracebacks have been computed using a different mechanism that works with “python -O”. For Python 2.3 Michael Hudson implemented a similar scheme to determine when to call the trace function, removing the need for <span class="pre">`SET_LINENO`</span> entirely.

  It would be difficult to detect any resulting difference from Python code, apart from a slight speed up when Python is run without <a href="../using/cmdline.html#cmdoption-o" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-O</code></span></a>.

  C extensions that access the <span class="pre">`f_lineno`</span> field of frame objects should instead call <span class="pre">`PyCode_Addr2Line(f->f_code,`</span>` `<span class="pre">`f->f_lasti)`</span>. This will have the added effect of making the code work as desired under “python -O” in earlier versions of Python.

  A nifty new feature is that trace functions can now assign to the <span class="pre">`f_lineno`</span> attribute of frame objects, changing the line that will be executed next. A <span class="pre">`jump`</span> command has been added to the <a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a> debugger taking advantage of this new feature. (Implemented by Richie Hindle.)

</div>

<div id="porting-to-python-2-3" class="section">

## Porting to Python 2.3<a href="#porting-to-python-2-3" class="headerlink" title="Permalink to this headline">¶</a>

This section lists previously described changes that may require changes to your code:

- <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> is now always a keyword; if it’s used as a variable name in your code, a different name must be chosen.

- For strings *X* and *Y*, <span class="pre">`X`</span>` `<span class="pre">`in`</span>` `<span class="pre">`Y`</span> now works if *X* is more than one character long.

- The <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a> type constructor will now return a long integer instead of raising an <a href="../library/exceptions.html#exceptions.OverflowError" class="reference internal" title="exceptions.OverflowError"><span class="pre"><code class="sourceCode python"><span class="pp">OverflowError</span></code></span></a> when a string or floating-point number is too large to fit into an integer.

- If you have Unicode strings that contain 8-bit characters, you must declare the file’s encoding (UTF-8, Latin-1, or whatever) by adding a comment to the top of the file. See section <a href="#section-encodings" class="reference internal"><span class="std std-ref">PEP 263: Source Code Encodings</span></a> for more information.

- Calling Tcl methods through <span class="pre">`_tkinter`</span> no longer returns only strings. Instead, if Tcl returns other objects those objects are converted to their Python equivalent, if one exists, or wrapped with a <span class="pre">`_tkinter.Tcl_Obj`</span> object if no Python equivalent exists.

- Large octal and hex literals such as <span class="pre">`0xffffffff`</span> now trigger a <a href="../library/exceptions.html#exceptions.FutureWarning" class="reference internal" title="exceptions.FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a>. Currently they’re stored as 32-bit numbers and result in a negative value, but in Python 2.4 they’ll become positive long integers.

  There are a few ways to fix this warning. If you really need a positive number, just add an <span class="pre">`L`</span> to the end of the literal. If you’re trying to get a 32-bit integer with low bits set and have previously used an expression such as <span class="pre">`~(1`</span>` `<span class="pre">`<<`</span>` `<span class="pre">`31)`</span>, it’s probably clearest to start with all bits set and clear the desired upper bits. For example, to clear just the top bit (bit 31), you could write <span class="pre">`0xffffffffL`</span>` `<span class="pre">`&~(1L<<31)`</span>.

- You can no longer disable assertions by assigning to <span class="pre">`__debug__`</span>.

- The Distutils <span class="pre">`setup()`</span> function has gained various new keyword arguments such as *depends*. Old versions of the Distutils will abort if passed unknown keywords. A solution is to check for the presence of the new <span class="pre">`get_distutil_options()`</span> function in your <span class="pre">`setup.py`</span> and only uses the new keywords with a version of the Distutils that supports them:

  <div class="highlight-default notranslate">

  <div class="highlight">

      from distutils import core

      kw = {'sources': 'foo.c', ...}
      if hasattr(core, 'get_distutil_options'):
          kw['depends'] = ['foo.h']
      ext = Extension(**kw)

  </div>

  </div>

- Using <span class="pre">`None`</span> as a variable name will now result in a <a href="../library/exceptions.html#exceptions.SyntaxWarning" class="reference internal" title="exceptions.SyntaxWarning"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxWarning</span></code></span></a> warning.

- Names of extension types defined by the modules included with Python now contain the module and a <span class="pre">`'.'`</span> in front of the type name.

</div>

<div id="acknowledgements" class="section">

<span id="acks"></span>

## Acknowledgements<a href="#acknowledgements" class="headerlink" title="Permalink to this headline">¶</a>

The author would like to thank the following people for offering suggestions, corrections and assistance with various drafts of this article: Jeff Bauer, Simon Brunning, Brett Cannon, Michael Chermside, Andrew Dalke, Scott David Daniels, Fred L. Drake, Jr., David Fraser, Kelly Gerber, Raymond Hettinger, Michael Hudson, Chris Lambert, Detlef Lannert, Martin von Löwis, Andrew MacIntyre, Lalo Martins, Chad Netzer, Gustavo Niemeyer, Neal Norwitz, Hans Nowak, Chris Reedy, Francesco Ricciardi, Vinay Sajip, Neil Schemenauer, Roman Suzi, Jason Tishler, Just van Rossum.

</div>

</div>

</div>
