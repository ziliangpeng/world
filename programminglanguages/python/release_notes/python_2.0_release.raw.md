<div class="body" role="main">

<div id="what-s-new-in-python-2-0" class="section">

# What’s New in Python 2.0<a href="#what-s-new-in-python-2-0" class="headerlink" title="Permalink to this headline">¶</a>

Author  
A.M. Kuchling and Moshe Zadka

<div id="introduction" class="section">

## Introduction<a href="#introduction" class="headerlink" title="Permalink to this headline">¶</a>

A new release of Python, version 2.0, was released on October 16, 2000. This article covers the exciting new features in 2.0, highlights some other useful changes, and points out a few incompatible changes that may require rewriting code.

Python’s development never completely stops between releases, and a steady flow of bug fixes and improvements are always being submitted. A host of minor fixes, a few optimizations, additional docstrings, and better error messages went into 2.0; to list them all would be impossible, but they’re certainly significant. Consult the publicly-available CVS logs if you want to see the full list. This progress is due to the five developers working for PythonLabs are now getting paid to spend their days fixing bugs, and also due to the improved communication resulting from moving to SourceForge.

</div>

<div id="what-about-python-1-6" class="section">

## What About Python 1.6?<a href="#what-about-python-1-6" class="headerlink" title="Permalink to this headline">¶</a>

Python 1.6 can be thought of as the Contractual Obligations Python release. After the core development team left CNRI in May 2000, CNRI requested that a 1.6 release be created, containing all the work on Python that had been performed at CNRI. Python 1.6 therefore represents the state of the CVS tree as of May 2000, with the most significant new feature being Unicode support. Development continued after May, of course, so the 1.6 tree received a few fixes to ensure that it’s forward-compatible with Python 2.0. 1.6 is therefore part of Python’s evolution, and not a side branch.

So, should you take much interest in Python 1.6? Probably not. The 1.6final and 2.0beta1 releases were made on the same day (September 5, 2000), the plan being to finalize Python 2.0 within a month or so. If you have applications to maintain, there seems little point in breaking things by moving to 1.6, fixing them, and then having another round of breakage within a month by moving to 2.0; you’re better off just going straight to 2.0. Most of the really interesting features described in this document are only in 2.0, because a lot of work was done between May and September.

</div>

<div id="new-development-process" class="section">

## New Development Process<a href="#new-development-process" class="headerlink" title="Permalink to this headline">¶</a>

The most important change in Python 2.0 may not be to the code at all, but to how Python is developed: in May 2000 the Python developers began using the tools made available by SourceForge for storing source code, tracking bug reports, and managing the queue of patch submissions. To report bugs or submit patches for Python 2.0, use the bug tracking and patch manager tools available from Python’s project page, located at <a href="https://sourceforge.net/projects/python/" class="reference external">https://sourceforge.net/projects/python/</a>.

The most important of the services now hosted at SourceForge is the Python CVS tree, the version-controlled repository containing the source code for Python. Previously, there were roughly 7 or so people who had write access to the CVS tree, and all patches had to be inspected and checked in by one of the people on this short list. Obviously, this wasn’t very scalable. By moving the CVS tree to SourceForge, it became possible to grant write access to more people; as of September 2000 there were 27 people able to check in changes, a fourfold increase. This makes possible large-scale changes that wouldn’t be attempted if they’d have to be filtered through the small group of core developers. For example, one day Peter Schneider-Kamp took it into his head to drop K&R C compatibility and convert the C source for Python to ANSI C. After getting approval on the python-dev mailing list, he launched into a flurry of checkins that lasted about a week, other developers joined in to help, and the job was done. If there were only 5 people with write access, probably that task would have been viewed as “nice, but not worth the time and effort needed” and it would never have gotten done.

The shift to using SourceForge’s services has resulted in a remarkable increase in the speed of development. Patches now get submitted, commented on, revised by people other than the original submitter, and bounced back and forth between people until the patch is deemed worth checking in. Bugs are tracked in one central location and can be assigned to a specific person for fixing, and we can count the number of open bugs to measure progress. This didn’t come without a cost: developers now have more e-mail to deal with, more mailing lists to follow, and special tools had to be written for the new environment. For example, SourceForge sends default patch and bug notification e-mail messages that are completely unhelpful, so Ka-Ping Yee wrote an HTML screen-scraper that sends more useful messages.

The ease of adding code caused a few initial growing pains, such as code was checked in before it was ready or without getting clear agreement from the developer group. The approval process that has emerged is somewhat similar to that used by the Apache group. Developers can vote +1, +0, -0, or -1 on a patch; +1 and -1 denote acceptance or rejection, while +0 and -0 mean the developer is mostly indifferent to the change, though with a slight positive or negative slant. The most significant change from the Apache model is that the voting is essentially advisory, letting Guido van Rossum, who has Benevolent Dictator For Life status, know what the general opinion is. He can still ignore the result of a vote, and approve or reject a change even if the community disagrees with him.

Producing an actual patch is the last step in adding a new feature, and is usually easy compared to the earlier task of coming up with a good design. Discussions of new features can often explode into lengthy mailing list threads, making the discussion hard to follow, and no one can read every posting to python-dev. Therefore, a relatively formal process has been set up to write Python Enhancement Proposals (PEPs), modelled on the Internet RFC process. PEPs are draft documents that describe a proposed new feature, and are continually revised until the community reaches a consensus, either accepting or rejecting the proposal. Quoting from the introduction to PEP 1, “PEP Purpose and Guidelines”:

> <div>
>
> PEP stands for Python Enhancement Proposal. A PEP is a design document providing information to the Python community, or describing a new feature for Python. The PEP should provide a concise technical specification of the feature and a rationale for the feature.
>
> We intend PEPs to be the primary mechanisms for proposing new features, for collecting community input on an issue, and for documenting the design decisions that have gone into Python. The PEP author is responsible for building consensus within the community and documenting dissenting opinions.
>
> </div>

Read the rest of PEP 1 for the details of the PEP editorial process, style, and format. PEPs are kept in the Python CVS tree on SourceForge, though they’re not part of the Python 2.0 distribution, and are also available in HTML form from <a href="https://www.python.org/dev/peps/" class="reference external">https://www.python.org/dev/peps/</a>. As of September 2000, there are 25 PEPS, ranging from PEP 201, “Lockstep Iteration”, to PEP 225, “Elementwise/Objectwise Operators”.

</div>

<div id="unicode" class="section">

## Unicode<a href="#unicode" class="headerlink" title="Permalink to this headline">¶</a>

The largest new feature in Python 2.0 is a new fundamental data type: Unicode strings. Unicode uses 16-bit numbers to represent characters instead of the 8-bit number used by ASCII, meaning that 65,536 distinct characters can be supported.

The final interface for Unicode support was arrived at through countless often-stormy discussions on the python-dev mailing list, and mostly implemented by Marc-André Lemburg, based on a Unicode string type implementation by Fredrik Lundh. A detailed explanation of the interface was written up as <span id="index-0" class="target"></span><a href="https://www.python.org/dev/peps/pep-0100" class="pep reference external"><strong>PEP 100</strong></a>, “Python Unicode Integration”. This article will simply cover the most significant points about the Unicode interfaces.

In Python source code, Unicode strings are written as <span class="pre">`u"string"`</span>. Arbitrary Unicode characters can be written using a new escape sequence, <span class="pre">`\uHHHH`</span>, where *HHHH* is a 4-digit hexadecimal number from 0000 to FFFF. The existing <span class="pre">`\xHHHH`</span> escape sequence can also be used, and octal escapes can be used for characters up to U+01FF, which is represented by <span class="pre">`\777`</span>.

Unicode strings, just like regular strings, are an immutable sequence type. They can be indexed and sliced, but not modified in place. Unicode strings have an <span class="pre">`encode(`</span>` `<span class="pre">`[encoding]`</span>` `<span class="pre">`)`</span> method that returns an 8-bit string in the desired encoding. Encodings are named by strings, such as <span class="pre">`'ascii'`</span>, <span class="pre">`'utf-8'`</span>, <span class="pre">`'iso-8859-1'`</span>, or whatever. A codec API is defined for implementing and registering new encodings that are then available throughout a Python program. If an encoding isn’t specified, the default encoding is usually 7-bit ASCII, though it can be changed for your Python installation by calling the <span class="pre">`sys.setdefaultencoding(encoding)`</span> function in a customised version of <span class="pre">`site.py`</span>.

Combining 8-bit and Unicode strings always coerces to Unicode, using the default ASCII encoding; the result of <span class="pre">`'a'`</span>` `<span class="pre">`+`</span>` `<span class="pre">`u'bc'`</span> is <span class="pre">`u'abc'`</span>.

New built-in functions have been added, and existing built-ins modified to support Unicode:

- <span class="pre">`unichr(ch)`</span> returns a Unicode string 1 character long, containing the character *ch*.

- <span class="pre">`ord(u)`</span>, where *u* is a 1-character regular or Unicode string, returns the number of the character as an integer.

- <span class="pre">`unicode(string`</span>` `<span class="pre">`[,`</span>` `<span class="pre">`encoding]`</span>`  `<span class="pre">`[,`</span>` `<span class="pre">`errors]`</span>` `<span class="pre">`)`</span> creates a Unicode string from an 8-bit string. <span class="pre">`encoding`</span> is a string naming the encoding to use. The <span class="pre">`errors`</span> parameter specifies the treatment of characters that are invalid for the current encoding; passing <span class="pre">`'strict'`</span> as the value causes an exception to be raised on any encoding error, while <span class="pre">`'ignore'`</span> causes errors to be silently ignored and <span class="pre">`'replace'`</span> uses U+FFFD, the official replacement character, in case of any problems.

- The <a href="../reference/simple_stmts.html#exec" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">exec</code></span></a> statement, and various built-ins such as <span class="pre">`eval()`</span>, <span class="pre">`getattr()`</span>, and <span class="pre">`setattr()`</span> will also accept Unicode strings as well as regular strings. (It’s possible that the process of fixing this missed some built-ins; if you find a built-in function that accepts strings but doesn’t accept Unicode strings at all, please report it as a bug.)

A new module, <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a>, provides an interface to Unicode character properties. For example, <span class="pre">`unicodedata.category(u'A')`</span> returns the 2-character string ‘Lu’, the ‘L’ denoting it’s a letter, and ‘u’ meaning that it’s uppercase. <span class="pre">`unicodedata.bidirectional(u'\u0660')`</span> returns ‘AN’, meaning that U+0660 is an Arabic number.

The <a href="../library/codecs.html#module-codecs" class="reference internal" title="codecs: Encode and decode data and streams."><span class="pre"><code class="sourceCode python">codecs</code></span></a> module contains functions to look up existing encodings and register new ones. Unless you want to implement a new encoding, you’ll most often use the <span class="pre">`codecs.lookup(encoding)`</span> function, which returns a 4-element tuple: <span class="pre">`(encode_func,`</span>` `<span class="pre">`decode_func,`</span>` `<span class="pre">`stream_reader,`</span>` `<span class="pre">`stream_writer)`</span>.

- *encode_func* is a function that takes a Unicode string, and returns a 2-tuple <span class="pre">`(string,`</span>` `<span class="pre">`length)`</span>. *string* is an 8-bit string containing a portion (perhaps all) of the Unicode string converted into the given encoding, and *length* tells you how much of the Unicode string was converted.

- *decode_func* is the opposite of *encode_func*, taking an 8-bit string and returning a 2-tuple <span class="pre">`(ustring,`</span>` `<span class="pre">`length)`</span>, consisting of the resulting Unicode string *ustring* and the integer *length* telling how much of the 8-bit string was consumed.

- *stream_reader* is a class that supports decoding input from a stream. *stream_reader(file_obj)* returns an object that supports the <span class="pre">`read()`</span>, <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline()</code></span></a>, and <span class="pre">`readlines()`</span> methods. These methods will all translate from the given encoding and return Unicode strings.

- *stream_writer*, similarly, is a class that supports encoding output to a stream. *stream_writer(file_obj)* returns an object that supports the <span class="pre">`write()`</span> and <span class="pre">`writelines()`</span> methods. These methods expect Unicode strings, translating them to the given encoding on output.

For example, the following code writes a Unicode string into a file, encoding it as UTF-8:

<div class="highlight-default notranslate">

<div class="highlight">

    import codecs

    unistr = u'\u0660\u2000ab ...'

    (UTF8_encode, UTF8_decode,
     UTF8_streamreader, UTF8_streamwriter) = codecs.lookup('UTF-8')

    output = UTF8_streamwriter( open( '/tmp/output', 'wb') )
    output.write( unistr )
    output.close()

</div>

</div>

The following code would then read UTF-8 input from the file:

<div class="highlight-default notranslate">

<div class="highlight">

    input = UTF8_streamreader( open( '/tmp/output', 'rb') )
    print repr(input.read())
    input.close()

</div>

</div>

Unicode-aware regular expressions are available through the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module, which has a new underlying implementation called SRE written by Fredrik Lundh of Secret Labs AB.

A <span class="pre">`-U`</span> command line option was added which causes the Python compiler to interpret all string literals as Unicode string literals. This is intended to be used in testing and future-proofing your Python code, since some future version of Python may drop support for 8-bit strings and provide only Unicode strings.

</div>

<div id="list-comprehensions" class="section">

## List Comprehensions<a href="#list-comprehensions" class="headerlink" title="Permalink to this headline">¶</a>

Lists are a workhorse data type in Python, and many programs manipulate a list at some point. Two common operations on lists are to loop over them, and either pick out the elements that meet a certain criterion, or apply some function to each element. For example, given a list of strings, you might want to pull out all the strings containing a given substring, or strip off trailing whitespace from each line.

The existing <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> and <a href="../library/functions.html#filter" class="reference internal" title="filter"><span class="pre"><code class="sourceCode python"><span class="bu">filter</span>()</code></span></a> functions can be used for this purpose, but they require a function as one of their arguments. This is fine if there’s an existing built-in function that can be passed directly, but if there isn’t, you have to create a little function to do the required work, and Python’s scoping rules make the result ugly if the little function needs additional information. Take the first example in the previous paragraph, finding all the strings in the list containing a given substring. You could write the following to do it:

<div class="highlight-default notranslate">

<div class="highlight">

    # Given the list L, make a list of all strings
    # containing the substring S.
    sublist = filter( lambda s, substring=S:
                         string.find(s, substring) != -1,
                      L)

</div>

</div>

Because of Python’s scoping rules, a default argument is used so that the anonymous function created by the <a href="../reference/expressions.html#lambda" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">lambda</code></span></a> statement knows what substring is being searched for. List comprehensions make this cleaner:

<div class="highlight-default notranslate">

<div class="highlight">

    sublist = [ s for s in L if string.find(s, S) != -1 ]

</div>

</div>

List comprehensions have the form:

<div class="highlight-default notranslate">

<div class="highlight">

    [ expression for expr in sequence1
                 for expr2 in sequence2 ...
                 for exprN in sequenceN
                 if condition ]

</div>

</div>

The <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a>…<a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> clauses contain the sequences to be iterated over. The sequences do not have to be the same length, because they are *not* iterated over in parallel, but from left to right; this is explained more clearly in the following paragraphs. The elements of the generated list will be the successive values of *expression*. The final <a href="../reference/compound_stmts.html#if" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">if</code></span></a> clause is optional; if present, *expression* is only evaluated and added to the result if *condition* is true.

To make the semantics very clear, a list comprehension is equivalent to the following Python code:

<div class="highlight-default notranslate">

<div class="highlight">

    for expr1 in sequence1:
        for expr2 in sequence2:
        ...
            for exprN in sequenceN:
                 if (condition):
                      # Append the value of
                      # the expression to the
                      # resulting list.

</div>

</div>

This means that when there are multiple <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a>…<a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> clauses, the resulting list will be equal to the product of the lengths of all the sequences. If you have two lists of length 3, the output list is 9 elements long:

<div class="highlight-default notranslate">

<div class="highlight">

    seq1 = 'abc'
    seq2 = (1,2,3)
    >>> [ (x,y) for x in seq1 for y in seq2]
    [('a', 1), ('a', 2), ('a', 3), ('b', 1), ('b', 2), ('b', 3), ('c', 1),
    ('c', 2), ('c', 3)]

</div>

</div>

To avoid introducing an ambiguity into Python’s grammar, if *expression* is creating a tuple, it must be surrounded with parentheses. The first list comprehension below is a syntax error, while the second one is correct:

<div class="highlight-default notranslate">

<div class="highlight">

    # Syntax error
    [ x,y for x in seq1 for y in seq2]
    # Correct
    [ (x,y) for x in seq1 for y in seq2]

</div>

</div>

The idea of list comprehensions originally comes from the functional programming language Haskell (<a href="https://www.haskell.org" class="reference external">https://www.haskell.org</a>). Greg Ewing argued most effectively for adding them to Python and wrote the initial list comprehension patch, which was then discussed for a seemingly endless time on the python-dev mailing list and kept up-to-date by Skip Montanaro.

</div>

<div id="augmented-assignment" class="section">

## Augmented Assignment<a href="#augmented-assignment" class="headerlink" title="Permalink to this headline">¶</a>

Augmented assignment operators, another long-requested feature, have been added to Python 2.0. Augmented assignment operators include <span class="pre">`+=`</span>, <span class="pre">`-=`</span>, <span class="pre">`*=`</span>, and so forth. For example, the statement <span class="pre">`a`</span>` `<span class="pre">`+=`</span>` `<span class="pre">`2`</span> increments the value of the variable <span class="pre">`a`</span> by 2, equivalent to the slightly lengthier <span class="pre">`a`</span>` `<span class="pre">`=`</span>` `<span class="pre">`a`</span>` `<span class="pre">`+`</span>` `<span class="pre">`2`</span>.

The full list of supported assignment operators is <span class="pre">`+=`</span>, <span class="pre">`-=`</span>, <span class="pre">`*=`</span>, <span class="pre">`/=`</span>, <span class="pre">`%=`</span>, <span class="pre">`**=`</span>, <span class="pre">`&=`</span>, <span class="pre">`|=`</span>, <span class="pre">`^=`</span>, <span class="pre">`>>=`</span>, and <span class="pre">`<<=`</span>. Python classes can override the augmented assignment operators by defining methods named <a href="../reference/datamodel.html#object.__iadd__" class="reference internal" title="object.__iadd__"><span class="pre"><code class="sourceCode python"><span class="fu">__iadd__</span>()</code></span></a>, <a href="../reference/datamodel.html#object.__isub__" class="reference internal" title="object.__isub__"><span class="pre"><code class="sourceCode python"><span class="fu">__isub__</span>()</code></span></a>, etc. For example, the following <span class="pre">`Number`</span> class stores a number and supports using += to create a new instance with an incremented value.

<div class="highlight-default notranslate">

<div class="highlight">

    class Number:
        def __init__(self, value):
            self.value = value
        def __iadd__(self, increment):
            return Number( self.value + increment)

    n = Number(5)
    n += 3
    print n.value

</div>

</div>

The <a href="../reference/datamodel.html#object.__iadd__" class="reference internal" title="object.__iadd__"><span class="pre"><code class="sourceCode python"><span class="fu">__iadd__</span>()</code></span></a> special method is called with the value of the increment, and should return a new instance with an appropriately modified value; this return value is bound as the new value of the variable on the left-hand side.

Augmented assignment operators were first introduced in the C programming language, and most C-derived languages, such as **awk**, C++, Java, Perl, and PHP also support them. The augmented assignment patch was implemented by Thomas Wouters.

</div>

<div id="string-methods" class="section">

## String Methods<a href="#string-methods" class="headerlink" title="Permalink to this headline">¶</a>

Until now string-manipulation functionality was in the <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module, which was usually a front-end for the <span class="pre">`strop`</span> module written in C. The addition of Unicode posed a difficulty for the <span class="pre">`strop`</span> module, because the functions would all need to be rewritten in order to accept either 8-bit or Unicode strings. For functions such as <a href="../library/string.html#string.replace" class="reference internal" title="string.replace"><span class="pre"><code class="sourceCode python">string.replace()</code></span></a>, which takes 3 string arguments, that means eight possible permutations, and correspondingly complicated code.

Instead, Python 2.0 pushes the problem onto the string type, making string manipulation functionality available through methods on both 8-bit strings and Unicode strings.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> 'andrew'.capitalize()
    'Andrew'
    >>> 'hostname'.replace('os', 'linux')
    'hlinuxtname'
    >>> 'moshe'.find('sh')
    2

</div>

</div>

One thing that hasn’t changed, a noteworthy April Fools’ joke notwithstanding, is that Python strings are immutable. Thus, the string methods return new strings, and do not modify the string on which they operate.

The old <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module is still around for backwards compatibility, but it mostly acts as a front-end to the new string methods.

Two methods which have no parallel in pre-2.0 versions, although they did exist in JPython for quite some time, are <span class="pre">`startswith()`</span> and <span class="pre">`endswith()`</span>. <span class="pre">`s.startswith(t)`</span> is equivalent to <span class="pre">`s[:len(t)]`</span>` `<span class="pre">`==`</span>` `<span class="pre">`t`</span>, while <span class="pre">`s.endswith(t)`</span> is equivalent to <span class="pre">`s[-len(t):]`</span>` `<span class="pre">`==`</span>` `<span class="pre">`t`</span>.

One other method which deserves special mention is <span class="pre">`join()`</span>. The <span class="pre">`join()`</span> method of a string receives one parameter, a sequence of strings, and is equivalent to the <a href="../library/string.html#string.join" class="reference internal" title="string.join"><span class="pre"><code class="sourceCode python">string.join()</code></span></a> function from the old <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module, with the arguments reversed. In other words, <span class="pre">`s.join(seq)`</span> is equivalent to the old <span class="pre">`string.join(seq,`</span>` `<span class="pre">`s)`</span>.

</div>

<div id="garbage-collection-of-cycles" class="section">

## Garbage Collection of Cycles<a href="#garbage-collection-of-cycles" class="headerlink" title="Permalink to this headline">¶</a>

The C implementation of Python uses reference counting to implement garbage collection. Every Python object maintains a count of the number of references pointing to itself, and adjusts the count as references are created or destroyed. Once the reference count reaches zero, the object is no longer accessible, since you need to have a reference to an object to access it, and if the count is zero, no references exist any longer.

Reference counting has some pleasant properties: it’s easy to understand and implement, and the resulting implementation is portable, fairly fast, and reacts well with other libraries that implement their own memory handling schemes. The major problem with reference counting is that it sometimes doesn’t realise that objects are no longer accessible, resulting in a memory leak. This happens when there are cycles of references.

Consider the simplest possible cycle, a class instance which has a reference to itself:

<div class="highlight-default notranslate">

<div class="highlight">

    instance = SomeClass()
    instance.myself = instance

</div>

</div>

After the above two lines of code have been executed, the reference count of <span class="pre">`instance`</span> is 2; one reference is from the variable named <span class="pre">`'instance'`</span>, and the other is from the <span class="pre">`myself`</span> attribute of the instance.

If the next line of code is <span class="pre">`del`</span>` `<span class="pre">`instance`</span>, what happens? The reference count of <span class="pre">`instance`</span> is decreased by 1, so it has a reference count of 1; the reference in the <span class="pre">`myself`</span> attribute still exists. Yet the instance is no longer accessible through Python code, and it could be deleted. Several objects can participate in a cycle if they have references to each other, causing all of the objects to be leaked.

Python 2.0 fixes this problem by periodically executing a cycle detection algorithm which looks for inaccessible cycles and deletes the objects involved. A new <a href="../library/gc.html#module-gc" class="reference internal" title="gc: Interface to the cycle-detecting garbage collector."><span class="pre"><code class="sourceCode python">gc</code></span></a> module provides functions to perform a garbage collection, obtain debugging statistics, and tuning the collector’s parameters.

Running the cycle detection algorithm takes some time, and therefore will result in some additional overhead. It is hoped that after we’ve gotten experience with the cycle collection from using 2.0, Python 2.1 will be able to minimize the overhead with careful tuning. It’s not yet obvious how much performance is lost, because benchmarking this is tricky and depends crucially on how often the program creates and destroys objects. The detection of cycles can be disabled when Python is compiled, if you can’t afford even a tiny speed penalty or suspect that the cycle collection is buggy, by specifying the <span class="pre">`--without-cycle-gc`</span> switch when running the **configure** script.

Several people tackled this problem and contributed to a solution. An early implementation of the cycle detection approach was written by Toby Kelsey. The current algorithm was suggested by Eric Tiedemann during a visit to CNRI, and Guido van Rossum and Neil Schemenauer wrote two different implementations, which were later integrated by Neil. Lots of other people offered suggestions along the way; the March 2000 archives of the python-dev mailing list contain most of the relevant discussion, especially in the threads titled “Reference cycle collection for Python” and “Finalization again”.

</div>

<div id="other-core-changes" class="section">

## Other Core Changes<a href="#other-core-changes" class="headerlink" title="Permalink to this headline">¶</a>

Various minor changes have been made to Python’s syntax and built-in functions. None of the changes are very far-reaching, but they’re handy conveniences.

<div id="minor-language-changes" class="section">

### Minor Language Changes<a href="#minor-language-changes" class="headerlink" title="Permalink to this headline">¶</a>

A new syntax makes it more convenient to call a given function with a tuple of arguments and/or a dictionary of keyword arguments. In Python 1.5 and earlier, you’d use the <a href="../library/functions.html#apply" class="reference internal" title="apply"><span class="pre"><code class="sourceCode python"><span class="bu">apply</span>()</code></span></a> built-in function: <span class="pre">`apply(f,`</span>` `<span class="pre">`args,`</span>` `<span class="pre">`kw)`</span> calls the function <span class="pre">`f()`</span> with the argument tuple *args* and the keyword arguments in the dictionary *kw*. <a href="../library/functions.html#apply" class="reference internal" title="apply"><span class="pre"><code class="sourceCode python"><span class="bu">apply</span>()</code></span></a> is the same in 2.0, but thanks to a patch from Greg Ewing, <span class="pre">`f(*args,`</span>` `<span class="pre">`**kw)`</span> is a shorter and clearer way to achieve the same effect. This syntax is symmetrical with the syntax for defining functions:

<div class="highlight-default notranslate">

<div class="highlight">

    def f(*args, **kw):
        # args is a tuple of positional args,
        # kw is a dictionary of keyword args
        ...

</div>

</div>

The <a href="../reference/simple_stmts.html#print" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">print</code></span></a> statement can now have its output directed to a file-like object by following the <a href="../reference/simple_stmts.html#print" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">print</code></span></a> with <span class="pre">`>>`</span>` `<span class="pre">`file`</span>, similar to the redirection operator in Unix shells. Previously you’d either have to use the <span class="pre">`write()`</span> method of the file-like object, which lacks the convenience and simplicity of <a href="../reference/simple_stmts.html#print" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">print</code></span></a>, or you could assign a new value to <span class="pre">`sys.stdout`</span> and then restore the old value. For sending output to standard error, it’s much easier to write this:

<div class="highlight-default notranslate">

<div class="highlight">

    print >> sys.stderr, "Warning: action field not supplied"

</div>

</div>

Modules can now be renamed on importing them, using the syntax <span class="pre">`import`</span>` `<span class="pre">`module`</span>` `<span class="pre">`as`</span>` `<span class="pre">`name`</span> or <span class="pre">`from`</span>` `<span class="pre">`module`</span>` `<span class="pre">`import`</span>` `<span class="pre">`name`</span>` `<span class="pre">`as`</span>` `<span class="pre">`othername`</span>. The patch was submitted by Thomas Wouters.

A new format style is available when using the <span class="pre">`%`</span> operator; ‘%r’ will insert the <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> of its argument. This was also added from symmetry considerations, this time for symmetry with the existing ‘%s’ format style, which inserts the <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> of its argument. For example, <span class="pre">`'%r`</span>` `<span class="pre">`%s'`</span>` `<span class="pre">`%`</span>` `<span class="pre">`('abc',`</span>` `<span class="pre">`'abc')`</span> returns a string containing <span class="pre">`'abc'`</span>` `<span class="pre">`abc`</span>.

Previously there was no way to implement a class that overrode Python’s built-in <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> operator and implemented a custom version. <span class="pre">`obj`</span>` `<span class="pre">`in`</span>` `<span class="pre">`seq`</span> returns true if *obj* is present in the sequence *seq*; Python computes this by simply trying every index of the sequence until either *obj* is found or an <a href="../library/exceptions.html#exceptions.IndexError" class="reference internal" title="exceptions.IndexError"><span class="pre"><code class="sourceCode python"><span class="pp">IndexError</span></code></span></a> is encountered. Moshe Zadka contributed a patch which adds a <a href="../reference/datamodel.html#object.__contains__" class="reference internal" title="object.__contains__"><span class="pre"><code class="sourceCode python"><span class="fu">__contains__</span>()</code></span></a> magic method for providing a custom implementation for <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a>. Additionally, new built-in objects written in C can define what <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> means for them via a new slot in the sequence protocol.

Earlier versions of Python used a recursive algorithm for deleting objects. Deeply nested data structures could cause the interpreter to fill up the C stack and crash; Christian Tismer rewrote the deletion logic to fix this problem. On a related note, comparing recursive objects recursed infinitely and crashed; Jeremy Hylton rewrote the code to no longer crash, producing a useful result instead. For example, after this code:

<div class="highlight-default notranslate">

<div class="highlight">

    a = []
    b = []
    a.append(a)
    b.append(b)

</div>

</div>

The comparison <span class="pre">`a==b`</span> returns true, because the two recursive data structures are isomorphic. See the thread “trashcan and PR#7” in the April 2000 archives of the python-dev mailing list for the discussion leading up to this implementation, and some useful relevant links. Note that comparisons can now also raise exceptions. In earlier versions of Python, a comparison operation such as <span class="pre">`cmp(a,b)`</span> would always produce an answer, even if a user-defined <a href="../reference/datamodel.html#object.__cmp__" class="reference internal" title="object.__cmp__"><span class="pre"><code class="sourceCode python"><span class="fu">__cmp__</span>()</code></span></a> method encountered an error, since the resulting exception would simply be silently swallowed.

Work has been done on porting Python to 64-bit Windows on the Itanium processor, mostly by Trent Mick of ActiveState. (Confusingly, <span class="pre">`sys.platform`</span> is still <span class="pre">`'win32'`</span> on Win64 because it seems that for ease of porting, MS Visual C++ treats code as 32 bit on Itanium.) PythonWin also supports Windows CE; see the Python CE page at <a href="http://pythonce.sourceforge.net/" class="reference external">http://pythonce.sourceforge.net/</a> for more information.

Another new platform is Darwin/MacOS X; initial support for it is in Python 2.0. Dynamic loading works, if you specify “configure –with-dyld –with-suffix=.x”. Consult the README in the Python source distribution for more instructions.

An attempt has been made to alleviate one of Python’s warts, the often-confusing <a href="../library/exceptions.html#exceptions.NameError" class="reference internal" title="exceptions.NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a> exception when code refers to a local variable before the variable has been assigned a value. For example, the following code raises an exception on the <a href="../reference/simple_stmts.html#print" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">print</code></span></a> statement in both 1.5.2 and 2.0; in 1.5.2 a <a href="../library/exceptions.html#exceptions.NameError" class="reference internal" title="exceptions.NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a> exception is raised, while 2.0 raises a new <a href="../library/exceptions.html#exceptions.UnboundLocalError" class="reference internal" title="exceptions.UnboundLocalError"><span class="pre"><code class="sourceCode python"><span class="pp">UnboundLocalError</span></code></span></a> exception. <a href="../library/exceptions.html#exceptions.UnboundLocalError" class="reference internal" title="exceptions.UnboundLocalError"><span class="pre"><code class="sourceCode python"><span class="pp">UnboundLocalError</span></code></span></a> is a subclass of <a href="../library/exceptions.html#exceptions.NameError" class="reference internal" title="exceptions.NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a>, so any existing code that expects <a href="../library/exceptions.html#exceptions.NameError" class="reference internal" title="exceptions.NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a> to be raised should still work.

<div class="highlight-default notranslate">

<div class="highlight">

    def f():
        print "i=",i
        i = i + 1
    f()

</div>

</div>

Two new exceptions, <a href="../library/exceptions.html#exceptions.TabError" class="reference internal" title="exceptions.TabError"><span class="pre"><code class="sourceCode python"><span class="pp">TabError</span></code></span></a> and <a href="../library/exceptions.html#exceptions.IndentationError" class="reference internal" title="exceptions.IndentationError"><span class="pre"><code class="sourceCode python"><span class="pp">IndentationError</span></code></span></a>, have been introduced. They’re both subclasses of <a href="../library/exceptions.html#exceptions.SyntaxError" class="reference internal" title="exceptions.SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>, and are raised when Python code is found to be improperly indented.

</div>

<div id="changes-to-built-in-functions" class="section">

### Changes to Built-in Functions<a href="#changes-to-built-in-functions" class="headerlink" title="Permalink to this headline">¶</a>

A new built-in, <span class="pre">`zip(seq1,`</span>` `<span class="pre">`seq2,`</span>` `<span class="pre">`...)`</span>, has been added. <a href="../library/functions.html#zip" class="reference internal" title="zip"><span class="pre"><code class="sourceCode python"><span class="bu">zip</span>()</code></span></a> returns a list of tuples where each tuple contains the i-th element from each of the argument sequences. The difference between <a href="../library/functions.html#zip" class="reference internal" title="zip"><span class="pre"><code class="sourceCode python"><span class="bu">zip</span>()</code></span></a> and <span class="pre">`map(None,`</span>` `<span class="pre">`seq1,`</span>` `<span class="pre">`seq2)`</span> is that <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> pads the sequences with <span class="pre">`None`</span> if the sequences aren’t all of the same length, while <a href="../library/functions.html#zip" class="reference internal" title="zip"><span class="pre"><code class="sourceCode python"><span class="bu">zip</span>()</code></span></a> truncates the returned list to the length of the shortest argument sequence.

The <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a> and <a href="../library/functions.html#long" class="reference internal" title="long"><span class="pre"><code class="sourceCode python"><span class="bu">long</span>()</code></span></a> functions now accept an optional “base” parameter when the first argument is a string. <span class="pre">`int('123',`</span>` `<span class="pre">`10)`</span> returns 123, while <span class="pre">`int('123',`</span>` `<span class="pre">`16)`</span> returns 291. <span class="pre">`int(123,`</span>` `<span class="pre">`16)`</span> raises a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception with the message “can’t convert non-string with explicit base”.

A new variable holding more detailed version information has been added to the <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> module. <span class="pre">`sys.version_info`</span> is a tuple <span class="pre">`(major,`</span>` `<span class="pre">`minor,`</span>` `<span class="pre">`micro,`</span>` `<span class="pre">`level,`</span>` `<span class="pre">`serial)`</span> For example, in a hypothetical 2.0.1beta1, <span class="pre">`sys.version_info`</span> would be <span class="pre">`(2,`</span>` `<span class="pre">`0,`</span>` `<span class="pre">`1,`</span>` `<span class="pre">`'beta',`</span>` `<span class="pre">`1)`</span>. *level* is a string such as <span class="pre">`"alpha"`</span>, <span class="pre">`"beta"`</span>, or <span class="pre">`"final"`</span> for a final release.

Dictionaries have an odd new method, <span class="pre">`setdefault(key,`</span>` `<span class="pre">`default)`</span>, which behaves similarly to the existing <span class="pre">`get()`</span> method. However, if the key is missing, <span class="pre">`setdefault()`</span> both returns the value of *default* as <span class="pre">`get()`</span> would do, and also inserts it into the dictionary as the value for *key*. Thus, the following lines of code:

<div class="highlight-default notranslate">

<div class="highlight">

    if dict.has_key( key ): return dict[key]
    else:
        dict[key] = []
        return dict[key]

</div>

</div>

can be reduced to a single <span class="pre">`return`</span>` `<span class="pre">`dict.setdefault(key,`</span>` `<span class="pre">`[])`</span> statement.

The interpreter sets a maximum recursion depth in order to catch runaway recursion before filling the C stack and causing a core dump or GPF.. Previously this limit was fixed when you compiled Python, but in 2.0 the maximum recursion depth can be read and modified using <a href="../library/sys.html#sys.getrecursionlimit" class="reference internal" title="sys.getrecursionlimit"><span class="pre"><code class="sourceCode python">sys.getrecursionlimit()</code></span></a> and <a href="../library/sys.html#sys.setrecursionlimit" class="reference internal" title="sys.setrecursionlimit"><span class="pre"><code class="sourceCode python">sys.setrecursionlimit()</code></span></a>. The default value is 1000, and a rough maximum value for a given platform can be found by running a new script, <span class="pre">`Misc/find_recursionlimit.py`</span>.

</div>

</div>

<div id="porting-to-2-0" class="section">

## Porting to 2.0<a href="#porting-to-2-0" class="headerlink" title="Permalink to this headline">¶</a>

New Python releases try hard to be compatible with previous releases, and the record has been pretty good. However, some changes are considered useful enough, usually because they fix initial design decisions that turned out to be actively mistaken, that breaking backward compatibility can’t always be avoided. This section lists the changes in Python 2.0 that may cause old Python code to break.

The change which will probably break the most code is tightening up the arguments accepted by some methods. Some methods would take multiple arguments and treat them as a tuple, particularly various list methods such as <span class="pre">`append()`</span> and <span class="pre">`insert()`</span>. In earlier versions of Python, if <span class="pre">`L`</span> is a list, <span class="pre">`L.append(`</span>` `<span class="pre">`1,2`</span>` `<span class="pre">`)`</span> appends the tuple <span class="pre">`(1,2)`</span> to the list. In Python 2.0 this causes a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception to be raised, with the message: ‘append requires exactly 1 argument; 2 given’. The fix is to simply add an extra set of parentheses to pass both values as a tuple: <span class="pre">`L.append(`</span>` `<span class="pre">`(1,2)`</span>` `<span class="pre">`)`</span>.

The earlier versions of these methods were more forgiving because they used an old function in Python’s C interface to parse their arguments; 2.0 modernizes them to use <span class="pre">`PyArg_ParseTuple()`</span>, the current argument parsing function, which provides more helpful error messages and treats multi-argument calls as errors. If you absolutely must use 2.0 but can’t fix your code, you can edit <span class="pre">`Objects/listobject.c`</span> and define the preprocessor symbol <span class="pre">`NO_STRICT_LIST_APPEND`</span> to preserve the old behaviour; this isn’t recommended.

Some of the functions in the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module are still forgiving in this way. For example, <span class="pre">`socket.connect(`</span>` `<span class="pre">`('hostname',`</span>` `<span class="pre">`25)`</span>` `<span class="pre">`)()`</span> is the correct form, passing a tuple representing an IP address, but <span class="pre">`socket.connect(`</span>` `<span class="pre">`'hostname',`</span>` `<span class="pre">`25`</span>` `<span class="pre">`)()`</span> also works. <span class="pre">`socket.connect_ex()`</span> and <span class="pre">`socket.bind()`</span> are similarly easy-going. 2.0alpha1 tightened these functions up, but because the documentation actually used the erroneous multiple argument form, many people wrote code which would break with the stricter checking. GvR backed out the changes in the face of public reaction, so for the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module, the documentation was fixed and the multiple argument form is simply marked as deprecated; it *will* be tightened up again in a future Python version.

The <span class="pre">`\x`</span> escape in string literals now takes exactly 2 hex digits. Previously it would consume all the hex digits following the ‘x’ and take the lowest 8 bits of the result, so <span class="pre">`\x123456`</span> was equivalent to <span class="pre">`\x56`</span>.

The <a href="../library/exceptions.html#exceptions.AttributeError" class="reference internal" title="exceptions.AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> and <a href="../library/exceptions.html#exceptions.NameError" class="reference internal" title="exceptions.NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a> exceptions have a more friendly error message, whose text will be something like <span class="pre">`'Spam'`</span>` `<span class="pre">`instance`</span>` `<span class="pre">`has`</span>` `<span class="pre">`no`</span>` `<span class="pre">`attribute`</span>` `<span class="pre">`'eggs'`</span> or <span class="pre">`name`</span>` `<span class="pre">`'eggs'`</span>` `<span class="pre">`is`</span>` `<span class="pre">`not`</span>` `<span class="pre">`defined`</span>. Previously the error message was just the missing attribute name <span class="pre">`eggs`</span>, and code written to take advantage of this fact will break in 2.0.

Some work has been done to make integers and long integers a bit more interchangeable. In 1.5.2, large-file support was added for Solaris, to allow reading files larger than 2 GiB; this made the <span class="pre">`tell()`</span> method of file objects return a long integer instead of a regular integer. Some code would subtract two file offsets and attempt to use the result to multiply a sequence or slice a string, but this raised a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>. In 2.0, long integers can be used to multiply or slice a sequence, and it’ll behave as you’d intuitively expect it to; <span class="pre">`3L`</span>` `<span class="pre">`*`</span>` `<span class="pre">`'abc'`</span> produces ‘abcabcabc’, and <span class="pre">`(0,1,2,3)[2L:4L]`</span> produces (2,3). Long integers can also be used in various contexts where previously only integers were accepted, such as in the <span class="pre">`seek()`</span> method of file objects, and in the formats supported by the <span class="pre">`%`</span> operator (<span class="pre">`%d`</span>, <span class="pre">`%i`</span>, <span class="pre">`%x`</span>, etc.). For example, <span class="pre">`"%d"`</span>` `<span class="pre">`%`</span>` `<span class="pre">`2L**64`</span> will produce the string <span class="pre">`18446744073709551616`</span>.

The subtlest long integer change of all is that the <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> of a long integer no longer has a trailing ‘L’ character, though <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> still includes it. The ‘L’ annoyed many people who wanted to print long integers that looked just like regular integers, since they had to go out of their way to chop off the character. This is no longer a problem in 2.0, but code which does <span class="pre">`str(longval)[:-1]`</span> and assumes the ‘L’ is there, will now lose the final digit.

Taking the <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> of a float now uses a different formatting precision than <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a>. <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> uses <span class="pre">`%.17g`</span> format string for C’s <span class="pre">`sprintf()`</span>, while <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> uses <span class="pre">`%.12g`</span> as before. The effect is that <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> may occasionally show more decimal places than <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a>, for certain numbers. For example, the number 8.1 can’t be represented exactly in binary, so <span class="pre">`repr(8.1)`</span> is <span class="pre">`'8.0999999999999996'`</span>, while str(8.1) is <span class="pre">`'8.1'`</span>.

The <span class="pre">`-X`</span> command-line option, which turned all standard exceptions into strings instead of classes, has been removed; the standard exceptions will now always be classes. The <a href="../library/exceptions.html#module-exceptions" class="reference internal" title="exceptions: Standard exception classes."><span class="pre"><code class="sourceCode python">exceptions</code></span></a> module containing the standard exceptions was translated from Python to a built-in C module, written by Barry Warsaw and Fredrik Lundh.

</div>

<div id="extending-embedding-changes" class="section">

## Extending/Embedding Changes<a href="#extending-embedding-changes" class="headerlink" title="Permalink to this headline">¶</a>

Some of the changes are under the covers, and will only be apparent to people writing C extension modules or embedding a Python interpreter in a larger application. If you aren’t dealing with Python’s C API, you can safely skip this section.

The version number of the Python C API was incremented, so C extensions compiled for 1.5.2 must be recompiled in order to work with 2.0. On Windows, it’s not possible for Python 2.0 to import a third party extension built for Python 1.5.x due to how Windows DLLs work, so Python will raise an exception and the import will fail.

Users of Jim Fulton’s ExtensionClass module will be pleased to find out that hooks have been added so that ExtensionClasses are now supported by <a href="../library/functions.html#isinstance" class="reference internal" title="isinstance"><span class="pre"><code class="sourceCode python"><span class="bu">isinstance</span>()</code></span></a> and <a href="../library/functions.html#issubclass" class="reference internal" title="issubclass"><span class="pre"><code class="sourceCode python"><span class="bu">issubclass</span>()</code></span></a>. This means you no longer have to remember to write code such as <span class="pre">`if`</span>` `<span class="pre">`type(obj)`</span>` `<span class="pre">`==`</span>` `<span class="pre">`myExtensionClass`</span>, but can use the more natural <span class="pre">`if`</span>` `<span class="pre">`isinstance(obj,`</span>` `<span class="pre">`myExtensionClass)`</span>.

The <span class="pre">`Python/importdl.c`</span> file, which was a mass of \#ifdefs to support dynamic loading on many different platforms, was cleaned up and reorganised by Greg Stein. <span class="pre">`importdl.c`</span> is now quite small, and platform-specific code has been moved into a bunch of <span class="pre">`Python/dynload_*.c`</span> files. Another cleanup: there were also a number of <span class="pre">`my*.h`</span> files in the Include/ directory that held various portability hacks; they’ve been merged into a single file, <span class="pre">`Include/pyport.h`</span>.

Vladimir Marangozov’s long-awaited malloc restructuring was completed, to make it easy to have the Python interpreter use a custom allocator instead of C’s standard <span class="pre">`malloc()`</span>. For documentation, read the comments in <span class="pre">`Include/pymem.h`</span> and <span class="pre">`Include/objimpl.h`</span>. For the lengthy discussions during which the interface was hammered out, see the Web archives of the ‘patches’ and ‘python-dev’ lists at python.org.

Recent versions of the GUSI development environment for MacOS support POSIX threads. Therefore, Python’s POSIX threading support now works on the Macintosh. Threading support using the user-space GNU <span class="pre">`pth`</span> library was also contributed.

Threading support on Windows was enhanced, too. Windows supports thread locks that use kernel objects only in case of contention; in the common case when there’s no contention, they use simpler functions which are an order of magnitude faster. A threaded version of Python 1.5.2 on NT is twice as slow as an unthreaded version; with the 2.0 changes, the difference is only 10%. These improvements were contributed by Yakov Markovitch.

Python 2.0’s source now uses only ANSI C prototypes, so compiling Python now requires an ANSI C compiler, and can no longer be done using a compiler that only supports K&R C.

Previously the Python virtual machine used 16-bit numbers in its bytecode, limiting the size of source files. In particular, this affected the maximum size of literal lists and dictionaries in Python source; occasionally people who are generating Python code would run into this limit. A patch by Charles G. Waldman raises the limit from <span class="pre">`2^16`</span> to <span class="pre">`2^{32}`</span>.

Three new convenience functions intended for adding constants to a module’s dictionary at module initialization time were added: <span class="pre">`PyModule_AddObject()`</span>, <span class="pre">`PyModule_AddIntConstant()`</span>, and <span class="pre">`PyModule_AddStringConstant()`</span>. Each of these functions takes a module object, a null-terminated C string containing the name to be added, and a third argument for the value to be assigned to the name. This third argument is, respectively, a Python object, a C long, or a C string.

A wrapper API was added for Unix-style signal handlers. <span class="pre">`PyOS_getsig()`</span> gets a signal handler and <span class="pre">`PyOS_setsig()`</span> will set a new handler.

</div>

<div id="distutils-making-modules-easy-to-install" class="section">

## Distutils: Making Modules Easy to Install<a href="#distutils-making-modules-easy-to-install" class="headerlink" title="Permalink to this headline">¶</a>

Before Python 2.0, installing modules was a tedious affair – there was no way to figure out automatically where Python is installed, or what compiler options to use for extension modules. Software authors had to go through an arduous ritual of editing Makefiles and configuration files, which only really work on Unix and leave Windows and MacOS unsupported. Python users faced wildly differing installation instructions which varied between different extension packages, which made administering a Python installation something of a chore.

The SIG for distribution utilities, shepherded by Greg Ward, has created the Distutils, a system to make package installation much easier. They form the <a href="../library/distutils.html#module-distutils" class="reference internal" title="distutils: Support for building and installing Python modules into an existing Python installation."><span class="pre"><code class="sourceCode python">distutils</code></span></a> package, a new part of Python’s standard library. In the best case, installing a Python module from source will require the same steps: first you simply mean unpack the tarball or zip archive, and the run “<span class="pre">`python`</span>` `<span class="pre">`setup.py`</span>` `<span class="pre">`install`</span>”. The platform will be automatically detected, the compiler will be recognized, C extension modules will be compiled, and the distribution installed into the proper directory. Optional command-line arguments provide more control over the installation process, the distutils package offers many places to override defaults – separating the build from the install, building or installing in non-default directories, and more.

In order to use the Distutils, you need to write a <span class="pre">`setup.py`</span> script. For the simple case, when the software contains only .py files, a minimal <span class="pre">`setup.py`</span> can be just a few lines long:

<div class="highlight-default notranslate">

<div class="highlight">

    from distutils.core import setup
    setup (name = "foo", version = "1.0",
           py_modules = ["module1", "module2"])

</div>

</div>

The <span class="pre">`setup.py`</span> file isn’t much more complicated if the software consists of a few packages:

<div class="highlight-default notranslate">

<div class="highlight">

    from distutils.core import setup
    setup (name = "foo", version = "1.0",
           packages = ["package", "package.subpackage"])

</div>

</div>

A C extension can be the most complicated case; here’s an example taken from the PyXML package:

<div class="highlight-default notranslate">

<div class="highlight">

    from distutils.core import setup, Extension

    expat_extension = Extension('xml.parsers.pyexpat',
         define_macros = [('XML_NS', None)],
         include_dirs = [ 'extensions/expat/xmltok',
                          'extensions/expat/xmlparse' ],
         sources = [ 'extensions/pyexpat.c',
                     'extensions/expat/xmltok/xmltok.c',
                     'extensions/expat/xmltok/xmlrole.c', ]
           )
    setup (name = "PyXML", version = "0.5.4",
           ext_modules =[ expat_extension ] )

</div>

</div>

The Distutils can also take care of creating source and binary distributions. The “sdist” command, run by “<span class="pre">`python`</span>` `<span class="pre">`setup.py`</span>` `<span class="pre">`sdist`</span>’, builds a source distribution such as <span class="pre">`foo-1.0.tar.gz`</span>. Adding new commands isn’t difficult, “bdist_rpm” and “bdist_wininst” commands have already been contributed to create an RPM distribution and a Windows installer for the software, respectively. Commands to create other distribution formats such as Debian packages and Solaris <span class="pre">`.pkg`</span> files are in various stages of development.

All this is documented in a new manual, *Distributing Python Modules*, that joins the basic set of Python documentation.

</div>

<div id="xml-modules" class="section">

## XML Modules<a href="#xml-modules" class="headerlink" title="Permalink to this headline">¶</a>

Python 1.5.2 included a simple XML parser in the form of the <span class="pre">`xmllib`</span> module, contributed by Sjoerd Mullender. Since 1.5.2’s release, two different interfaces for processing XML have become common: SAX2 (version 2 of the Simple API for XML) provides an event-driven interface with some similarities to <span class="pre">`xmllib`</span>, and the DOM (Document Object Model) provides a tree-based interface, transforming an XML document into a tree of nodes that can be traversed and modified. Python 2.0 includes a SAX2 interface and a stripped-down DOM interface as part of the <a href="../library/xml.html#module-xml" class="reference internal" title="xml: Package containing XML processing modules"><span class="pre"><code class="sourceCode python">xml</code></span></a> package. Here we will give a brief overview of these new interfaces; consult the Python documentation or the source code for complete details. The Python XML SIG is also working on improved documentation.

<div id="sax2-support" class="section">

### SAX2 Support<a href="#sax2-support" class="headerlink" title="Permalink to this headline">¶</a>

SAX defines an event-driven interface for parsing XML. To use SAX, you must write a SAX handler class. Handler classes inherit from various classes provided by SAX, and override various methods that will then be called by the XML parser. For example, the <span class="pre">`startElement()`</span> and <span class="pre">`endElement()`</span> methods are called for every starting and end tag encountered by the parser, the <span class="pre">`characters()`</span> method is called for every chunk of character data, and so forth.

The advantage of the event-driven approach is that the whole document doesn’t have to be resident in memory at any one time, which matters if you are processing really huge documents. However, writing the SAX handler class can get very complicated if you’re trying to modify the document structure in some elaborate way.

For example, this little example program defines a handler that prints a message for every starting and ending tag, and then parses the file <span class="pre">`hamlet.xml`</span> using it:

<div class="highlight-default notranslate">

<div class="highlight">

    from xml import sax

    class SimpleHandler(sax.ContentHandler):
        def startElement(self, name, attrs):
            print 'Start of element:', name, attrs.keys()

        def endElement(self, name):
            print 'End of element:', name

    # Create a parser object
    parser = sax.make_parser()

    # Tell it what handler to use
    handler = SimpleHandler()
    parser.setContentHandler( handler )

    # Parse a file!
    parser.parse( 'hamlet.xml' )

</div>

</div>

For more information, consult the Python documentation, or the XML HOWTO at <a href="http://pyxml.sourceforge.net/topics/howto/xml-howto.html" class="reference external">http://pyxml.sourceforge.net/topics/howto/xml-howto.html</a>.

</div>

<div id="dom-support" class="section">

### DOM Support<a href="#dom-support" class="headerlink" title="Permalink to this headline">¶</a>

The Document Object Model is a tree-based representation for an XML document. A top-level <span class="pre">`Document`</span> instance is the root of the tree, and has a single child which is the top-level <span class="pre">`Element`</span> instance. This <span class="pre">`Element`</span> has children nodes representing character data and any sub-elements, which may have further children of their own, and so forth. Using the DOM you can traverse the resulting tree any way you like, access element and attribute values, insert and delete nodes, and convert the tree back into XML.

The DOM is useful for modifying XML documents, because you can create a DOM tree, modify it by adding new nodes or rearranging subtrees, and then produce a new XML document as output. You can also construct a DOM tree manually and convert it to XML, which can be a more flexible way of producing XML output than simply writing <span class="pre">`<tag1>`</span>…<span class="pre">`</tag1>`</span> to a file.

The DOM implementation included with Python lives in the <a href="../library/xml.dom.minidom.html#module-xml.dom.minidom" class="reference internal" title="xml.dom.minidom: Minimal Document Object Model (DOM) implementation."><span class="pre"><code class="sourceCode python">xml.dom.minidom</code></span></a> module. It’s a lightweight implementation of the Level 1 DOM with support for XML namespaces. The <span class="pre">`parse()`</span> and <span class="pre">`parseString()`</span> convenience functions are provided for generating a DOM tree:

<div class="highlight-default notranslate">

<div class="highlight">

    from xml.dom import minidom
    doc = minidom.parse('hamlet.xml')

</div>

</div>

<span class="pre">`doc`</span> is a <span class="pre">`Document`</span> instance. <span class="pre">`Document`</span>, like all the other DOM classes such as <span class="pre">`Element`</span> and <span class="pre">`Text`</span>, is a subclass of the <span class="pre">`Node`</span> base class. All the nodes in a DOM tree therefore support certain common methods, such as <span class="pre">`toxml()`</span> which returns a string containing the XML representation of the node and its children. Each class also has special methods of its own; for example, <span class="pre">`Element`</span> and <span class="pre">`Document`</span> instances have a method to find all child elements with a given tag name. Continuing from the previous 2-line example:

<div class="highlight-default notranslate">

<div class="highlight">

    perslist = doc.getElementsByTagName( 'PERSONA' )
    print perslist[0].toxml()
    print perslist[1].toxml()

</div>

</div>

For the *Hamlet* XML file, the above few lines output:

<div class="highlight-default notranslate">

<div class="highlight">

    <PERSONA>CLAUDIUS, king of Denmark. </PERSONA>
    <PERSONA>HAMLET, son to the late, and nephew to the present king.</PERSONA>

</div>

</div>

The root element of the document is available as <span class="pre">`doc.documentElement`</span>, and its children can be easily modified by deleting, adding, or removing nodes:

<div class="highlight-default notranslate">

<div class="highlight">

    root = doc.documentElement

    # Remove the first child
    root.removeChild( root.childNodes[0] )

    # Move the new first child to the end
    root.appendChild( root.childNodes[0] )

    # Insert the new first child (originally,
    # the third child) before the 20th child.
    root.insertBefore( root.childNodes[0], root.childNodes[20] )

</div>

</div>

Again, I will refer you to the Python documentation for a complete listing of the different <span class="pre">`Node`</span> classes and their various methods.

</div>

<div id="relationship-to-pyxml" class="section">

### Relationship to PyXML<a href="#relationship-to-pyxml" class="headerlink" title="Permalink to this headline">¶</a>

The XML Special Interest Group has been working on XML-related Python code for a while. Its code distribution, called PyXML, is available from the SIG’s Web pages at <a href="https://www.python.org/community/sigs/current/xml-sig" class="reference external">https://www.python.org/community/sigs/current/xml-sig</a>. The PyXML distribution also used the package name <span class="pre">`xml`</span>. If you’ve written programs that used PyXML, you’re probably wondering about its compatibility with the 2.0 <a href="../library/xml.html#module-xml" class="reference internal" title="xml: Package containing XML processing modules"><span class="pre"><code class="sourceCode python">xml</code></span></a> package.

The answer is that Python 2.0’s <a href="../library/xml.html#module-xml" class="reference internal" title="xml: Package containing XML processing modules"><span class="pre"><code class="sourceCode python">xml</code></span></a> package isn’t compatible with PyXML, but can be made compatible by installing a recent version PyXML. Many applications can get by with the XML support that is included with Python 2.0, but more complicated applications will require that the full PyXML package will be installed. When installed, PyXML versions 0.6.0 or greater will replace the <a href="../library/xml.html#module-xml" class="reference internal" title="xml: Package containing XML processing modules"><span class="pre"><code class="sourceCode python">xml</code></span></a> package shipped with Python, and will be a strict superset of the standard package, adding a bunch of additional features. Some of the additional features in PyXML include:

- 4DOM, a full DOM implementation from FourThought, Inc.

- The xmlproc validating parser, written by Lars Marius Garshol.

- The <span class="pre">`sgmlop`</span> parser accelerator module, written by Fredrik Lundh.

</div>

</div>

<div id="module-changes" class="section">

## Module changes<a href="#module-changes" class="headerlink" title="Permalink to this headline">¶</a>

Lots of improvements and bugfixes were made to Python’s extensive standard library; some of the affected modules include <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline</code></span></a>, <a href="../library/configparser.html#module-ConfigParser" class="reference internal" title="ConfigParser: Configuration file parser."><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a>, <a href="../library/cgi.html#module-cgi" class="reference internal" title="cgi: Helpers for running Python scripts via the Common Gateway Interface."><span class="pre"><code class="sourceCode python">cgi</code></span></a>, <a href="../library/calendar.html#module-calendar" class="reference internal" title="calendar: Functions for working with calendars, including some emulation of the Unix cal program."><span class="pre"><code class="sourceCode python">calendar</code></span></a>, <a href="../library/posix.html#module-posix" class="reference internal" title="posix: The most common POSIX system calls (normally used via module os). (Unix)"><span class="pre"><code class="sourceCode python">posix</code></span></a>, <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline</code></span></a>, <span class="pre">`xmllib`</span>, <a href="../library/aifc.html#module-aifc" class="reference internal" title="aifc: Read and write audio files in AIFF or AIFC format."><span class="pre"><code class="sourceCode python">aifc</code></span></a>, <span class="pre">`chunk,`</span>` `<span class="pre">`wave`</span>, <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a>, <a href="../library/shelve.html#module-shelve" class="reference internal" title="shelve: Python object persistence."><span class="pre"><code class="sourceCode python">shelve</code></span></a>, and <a href="../library/nntplib.html#module-nntplib" class="reference internal" title="nntplib: NNTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">nntplib</code></span></a>. Consult the CVS logs for the exact patch-by-patch details.

Brian Gallew contributed OpenSSL support for the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module. OpenSSL is an implementation of the Secure Socket Layer, which encrypts the data being sent over a socket. When compiling Python, you can edit <span class="pre">`Modules/Setup`</span> to include SSL support, which adds an additional function to the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module: <span class="pre">`socket.ssl(socket,`</span>` `<span class="pre">`keyfile,`</span>` `<span class="pre">`certfile)`</span>, which takes a socket object and returns an SSL socket. The <a href="../library/httplib.html#module-httplib" class="reference internal" title="httplib: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">httplib</code></span></a> and <a href="../library/urllib.html#module-urllib" class="reference internal" title="urllib: Open an arbitrary network resource by URL (requires sockets)."><span class="pre"><code class="sourceCode python">urllib</code></span></a> modules were also changed to support <span class="pre">`https://`</span> URLs, though no one has implemented FTP or SMTP over SSL.

The <a href="../library/httplib.html#module-httplib" class="reference internal" title="httplib: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">httplib</code></span></a> module has been rewritten by Greg Stein to support HTTP/1.1. Backward compatibility with the 1.5 version of <a href="../library/httplib.html#module-httplib" class="reference internal" title="httplib: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">httplib</code></span></a> is provided, though using HTTP/1.1 features such as pipelining will require rewriting code to use a different set of interfaces.

The <a href="../library/tkinter.html#module-Tkinter" class="reference internal" title="Tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">Tkinter</code></span></a> module now supports Tcl/Tk version 8.1, 8.2, or 8.3, and support for the older 7.x versions has been dropped. The Tkinter module now supports displaying Unicode strings in Tk widgets. Also, Fredrik Lundh contributed an optimization which makes operations like <span class="pre">`create_line`</span> and <span class="pre">`create_polygon`</span> much faster, especially when using lots of coordinates.

The <a href="../library/curses.html#module-curses" class="reference internal" title="curses: An interface to the curses library, providing portable terminal handling. (Unix)"><span class="pre"><code class="sourceCode python">curses</code></span></a> module has been greatly extended, starting from Oliver Andrich’s enhanced version, to provide many additional functions from ncurses and SYSV curses, such as colour, alternative character set support, pads, and mouse support. This means the module is no longer compatible with operating systems that only have BSD curses, but there don’t seem to be any currently maintained OSes that fall into this category.

As mentioned in the earlier discussion of 2.0’s Unicode support, the underlying implementation of the regular expressions provided by the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module has been changed. SRE, a new regular expression engine written by Fredrik Lundh and partially funded by Hewlett Packard, supports matching against both 8-bit strings and Unicode strings.

</div>

<div id="new-modules" class="section">

## New modules<a href="#new-modules" class="headerlink" title="Permalink to this headline">¶</a>

A number of new modules were added. We’ll simply list them with brief descriptions; consult the 2.0 documentation for the details of a particular module.

- <a href="../library/atexit.html#module-atexit" class="reference internal" title="atexit: Register and execute cleanup functions."><span class="pre"><code class="sourceCode python">atexit</code></span></a>: For registering functions to be called before the Python interpreter exits. Code that currently sets <span class="pre">`sys.exitfunc`</span> directly should be changed to use the <a href="../library/atexit.html#module-atexit" class="reference internal" title="atexit: Register and execute cleanup functions."><span class="pre"><code class="sourceCode python">atexit</code></span></a> module instead, importing <a href="../library/atexit.html#module-atexit" class="reference internal" title="atexit: Register and execute cleanup functions."><span class="pre"><code class="sourceCode python">atexit</code></span></a> and calling <a href="../library/atexit.html#atexit.register" class="reference internal" title="atexit.register"><span class="pre"><code class="sourceCode python">atexit.register()</code></span></a> with the function to be called on exit. (Contributed by Skip Montanaro.)

- <a href="../library/codecs.html#module-codecs" class="reference internal" title="codecs: Encode and decode data and streams."><span class="pre"><code class="sourceCode python">codecs</code></span></a>, <span class="pre">`encodings`</span>, <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a>: Added as part of the new Unicode support.

- <a href="../library/filecmp.html#module-filecmp" class="reference internal" title="filecmp: Compare files efficiently."><span class="pre"><code class="sourceCode python">filecmp</code></span></a>: Supersedes the old <a href="../library/functions.html#cmp" class="reference internal" title="cmp"><span class="pre"><code class="sourceCode python"><span class="bu">cmp</span></code></span></a>, <span class="pre">`cmpcache`</span> and <span class="pre">`dircmp`</span> modules, which have now become deprecated. (Contributed by Gordon MacMillan and Moshe Zadka.)

- <a href="../library/gettext.html#module-gettext" class="reference internal" title="gettext: Multilingual internationalization services."><span class="pre"><code class="sourceCode python">gettext</code></span></a>: This module provides internationalization (I18N) and localization (L10N) support for Python programs by providing an interface to the GNU gettext message catalog library. (Integrated by Barry Warsaw, from separate contributions by Martin von Löwis, Peter Funk, and James Henstridge.)

- <span class="pre">`linuxaudiodev`</span>: Support for the <span class="pre">`/dev/audio`</span> device on Linux, a twin to the existing <a href="../library/sunaudio.html#module-sunaudiodev" class="reference internal" title="sunaudiodev: Access to Sun audio hardware. (deprecated) (SunOS)"><span class="pre"><code class="sourceCode python">sunaudiodev</code></span></a> module. (Contributed by Peter Bosch, with fixes by Jeremy Hylton.)

- <a href="../library/mmap.html#module-mmap" class="reference internal" title="mmap: Interface to memory-mapped files for Unix and Windows."><span class="pre"><code class="sourceCode python">mmap</code></span></a>: An interface to memory-mapped files on both Windows and Unix. A file’s contents can be mapped directly into memory, at which point it behaves like a mutable string, so its contents can be read and modified. They can even be passed to functions that expect ordinary strings, such as the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module. (Contributed by Sam Rushing, with some extensions by A.M. Kuchling.)

- <span class="pre">`pyexpat`</span>: An interface to the Expat XML parser. (Contributed by Paul Prescod.)

- <a href="../library/robotparser.html#module-robotparser" class="reference internal" title="robotparser: Loads a robots.txt file and answers questions about fetchability of other URLs."><span class="pre"><code class="sourceCode python">robotparser</code></span></a>: Parse a <span class="pre">`robots.txt`</span> file, which is used for writing Web spiders that politely avoid certain areas of a Web site. The parser accepts the contents of a <span class="pre">`robots.txt`</span> file, builds a set of rules from it, and can then answer questions about the fetchability of a given URL. (Contributed by Skip Montanaro.)

- <a href="../library/tabnanny.html#module-tabnanny" class="reference internal" title="tabnanny: Tool for detecting white space related problems in Python source files in a directory tree."><span class="pre"><code class="sourceCode python">tabnanny</code></span></a>: A module/script to check Python source code for ambiguous indentation. (Contributed by Tim Peters.)

- <a href="../library/userdict.html#module-UserString" class="reference internal" title="UserString: Class wrapper for string objects."><span class="pre"><code class="sourceCode python">UserString</code></span></a>: A base class useful for deriving objects that behave like strings.

- <a href="../library/webbrowser.html#module-webbrowser" class="reference internal" title="webbrowser: Easy-to-use controller for Web browsers."><span class="pre"><code class="sourceCode python">webbrowser</code></span></a>: A module that provides a platform independent way to launch a web browser on a specific URL. For each platform, various browsers are tried in a specific order. The user can alter which browser is launched by setting the *BROWSER* environment variable. (Originally inspired by Eric S. Raymond’s patch to <a href="../library/urllib.html#module-urllib" class="reference internal" title="urllib: Open an arbitrary network resource by URL (requires sockets)."><span class="pre"><code class="sourceCode python">urllib</code></span></a> which added similar functionality, but the final module comes from code originally implemented by Fred Drake as <span class="pre">`Tools/idle/BrowserControl.py`</span>, and adapted for the standard library by Fred.)

- <a href="../library/_winreg.html#module-_winreg" class="reference internal" title="_winreg: Routines and objects for manipulating the Windows registry. (Windows)"><span class="pre"><code class="sourceCode python">_winreg</code></span></a>: An interface to the Windows registry. <a href="../library/_winreg.html#module-_winreg" class="reference internal" title="_winreg: Routines and objects for manipulating the Windows registry. (Windows)"><span class="pre"><code class="sourceCode python">_winreg</code></span></a> is an adaptation of functions that have been part of PythonWin since 1995, but has now been added to the core distribution, and enhanced to support Unicode. <a href="../library/_winreg.html#module-_winreg" class="reference internal" title="_winreg: Routines and objects for manipulating the Windows registry. (Windows)"><span class="pre"><code class="sourceCode python">_winreg</code></span></a> was written by Bill Tutt and Mark Hammond.

- <a href="../library/zipfile.html#module-zipfile" class="reference internal" title="zipfile: Read and write ZIP-format archive files."><span class="pre"><code class="sourceCode python">zipfile</code></span></a>: A module for reading and writing ZIP-format archives. These are archives produced by **PKZIP** on DOS/Windows or **zip** on Unix, not to be confused with **gzip**-format files (which are supported by the <a href="../library/gzip.html#module-gzip" class="reference internal" title="gzip: Interfaces for gzip compression and decompression using file objects."><span class="pre"><code class="sourceCode python">gzip</code></span></a> module) (Contributed by James C. Ahlstrom.)

- <a href="../library/imputil.html#module-imputil" class="reference internal" title="imputil: Manage and augment the import process. (deprecated)"><span class="pre"><code class="sourceCode python">imputil</code></span></a>: A module that provides a simpler way for writing customised import hooks, in comparison to the existing <span class="pre">`ihooks`</span> module. (Implemented by Greg Stein, with much discussion on python-dev along the way.)

</div>

<div id="idle-improvements" class="section">

## IDLE Improvements<a href="#idle-improvements" class="headerlink" title="Permalink to this headline">¶</a>

IDLE is the official Python cross-platform IDE, written using Tkinter. Python 2.0 includes IDLE 0.6, which adds a number of new features and improvements. A partial list:

- UI improvements and optimizations, especially in the area of syntax highlighting and auto-indentation.

- The class browser now shows more information, such as the top level functions in a module.

- Tab width is now a user settable option. When opening an existing Python file, IDLE automatically detects the indentation conventions, and adapts.

- There is now support for calling browsers on various platforms, used to open the Python documentation in a browser.

- IDLE now has a command line, which is largely similar to the vanilla Python interpreter.

- Call tips were added in many places.

- IDLE can now be installed as a package.

- In the editor window, there is now a line/column bar at the bottom.

- Three new keystroke commands: Check module (<span class="kbd kbd docutils literal notranslate">Alt-F5</span>), Import module (<span class="kbd kbd docutils literal notranslate">F5</span>) and Run script (<span class="kbd kbd docutils literal notranslate">Ctrl-F5</span>).

</div>

<div id="deleted-and-deprecated-modules" class="section">

## Deleted and Deprecated Modules<a href="#deleted-and-deprecated-modules" class="headerlink" title="Permalink to this headline">¶</a>

A few modules have been dropped because they’re obsolete, or because there are now better ways to do the same thing. The <span class="pre">`stdwin`</span> module is gone; it was for a platform-independent windowing toolkit that’s no longer developed.

A number of modules have been moved to the <span class="pre">`lib-old`</span> subdirectory: <a href="../library/functions.html#cmp" class="reference internal" title="cmp"><span class="pre"><code class="sourceCode python"><span class="bu">cmp</span></code></span></a>, <span class="pre">`cmpcache`</span>, <span class="pre">`dircmp`</span>, <span class="pre">`dump`</span>, <span class="pre">`find`</span>, <span class="pre">`grep`</span>, <span class="pre">`packmail`</span>, <span class="pre">`poly`</span>, <span class="pre">`util`</span>, <span class="pre">`whatsound`</span>, <span class="pre">`zmod`</span>. If you have code which relies on a module that’s been moved to <span class="pre">`lib-old`</span>, you can simply add that directory to <span class="pre">`sys.path`</span> to get them back, but you’re encouraged to update any code that uses these modules.

</div>

<div id="acknowledgements" class="section">

## Acknowledgements<a href="#acknowledgements" class="headerlink" title="Permalink to this headline">¶</a>

The authors would like to thank the following people for offering suggestions on various drafts of this article: David Bolen, Mark Hammond, Gregg Hauser, Jeremy Hylton, Fredrik Lundh, Detlef Lannert, Aahz Maruch, Skip Montanaro, Vladimir Marangozov, Tobias Polzin, Guido van Rossum, Neil Schemenauer, and Russ Schmidt.

</div>

</div>

</div>
