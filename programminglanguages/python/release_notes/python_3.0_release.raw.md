<div class="body" role="main">

<div id="what-s-new-in-python-3-0" class="section">

# What’s New In Python 3.0<a href="#what-s-new-in-python-3-0" class="headerlink" title="Link to this heading">¶</a>

Author<span class="colon">:</span>  
Guido van Rossum

This article explains the new features in Python 3.0, compared to 2.6. Python 3.0, also known as “Python 3000” or “Py3K”, is the first ever *intentionally backwards incompatible* Python release. Python 3.0 was released on December 3, 2008. There are more changes than in a typical release, and more that are important for all Python users. Nevertheless, after digesting the changes, you’ll find that Python really hasn’t changed all that much – by and large, we’re mostly fixing well-known annoyances and warts, and removing a lot of old cruft.

This article doesn’t attempt to provide a complete specification of all new features, but instead tries to give a convenient overview. For full details, you should refer to the documentation for Python 3.0, and/or the many PEPs referenced in the text. If you want to understand the complete implementation and design rationale for a particular feature, PEPs usually have more details than the regular documentation; but note that PEPs usually are not kept up-to-date once a feature has been fully implemented.

Due to time constraints this document is not as complete as it should have been. As always for a new release, the <span class="pre">`Misc/NEWS`</span> file in the source distribution contains a wealth of detailed information about every small thing that was changed.

<div id="common-stumbling-blocks" class="section">

## Common Stumbling Blocks<a href="#common-stumbling-blocks" class="headerlink" title="Link to this heading">¶</a>

This section lists those few changes that are most likely to trip you up if you’re used to Python 2.5.

<div id="print-is-a-function" class="section">

### Print Is A Function<a href="#print-is-a-function" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`print`</span> statement has been replaced with a <a href="../library/functions.html#print" class="reference internal" title="print"><span class="pre"><code class="sourceCode python"><span class="bu">print</span>()</code></span></a> function, with keyword arguments to replace most of the special syntax of the old <span class="pre">`print`</span> statement (<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-3105/" class="pep reference external"><strong>PEP 3105</strong></a>). Examples:

<div class="highlight-python3 notranslate">

<div class="highlight">

    Old: print "The answer is", 2*2
    New: print("The answer is", 2*2)

    Old: print x,           # Trailing comma suppresses newline
    New: print(x, end=" ")  # Appends a space instead of a newline

    Old: print              # Prints a newline
    New: print()            # You must call the function!

    Old: print >>sys.stderr, "fatal error"
    New: print("fatal error", file=sys.stderr)

    Old: print (x, y)       # prints repr((x, y))
    New: print((x, y))      # Not the same as print(x, y)!

</div>

</div>

You can also customize the separator between items, e.g.:

<div class="highlight-python3 notranslate">

<div class="highlight">

    print("There are <", 2**32, "> possibilities!", sep="")

</div>

</div>

which produces:

<div class="highlight-none notranslate">

<div class="highlight">

    There are <4294967296> possibilities!

</div>

</div>

Note:

- The <a href="../library/functions.html#print" class="reference internal" title="print"><span class="pre"><code class="sourceCode python"><span class="bu">print</span>()</code></span></a> function doesn’t support the “softspace” feature of the old <span class="pre">`print`</span> statement. For example, in Python 2.x, <span class="pre">`print`</span>` `<span class="pre">`"A\n",`</span>` `<span class="pre">`"B"`</span> would write <span class="pre">`"A\nB\n"`</span>; but in Python 3.0, <span class="pre">`print("A\n",`</span>` `<span class="pre">`"B")`</span> writes <span class="pre">`"A\n`</span>` `<span class="pre">`B\n"`</span>.

- Initially, you’ll be finding yourself typing the old <span class="pre">`print`</span>` `<span class="pre">`x`</span> a lot in interactive mode. Time to retrain your fingers to type <span class="pre">`print(x)`</span> instead!

- When using the <span class="pre">`2to3`</span> source-to-source conversion tool, all <span class="pre">`print`</span> statements are automatically converted to <a href="../library/functions.html#print" class="reference internal" title="print"><span class="pre"><code class="sourceCode python"><span class="bu">print</span>()</code></span></a> function calls, so this is mostly a non-issue for larger projects.

</div>

<div id="views-and-iterators-instead-of-lists" class="section">

### Views And Iterators Instead Of Lists<a href="#views-and-iterators-instead-of-lists" class="headerlink" title="Link to this heading">¶</a>

Some well-known APIs no longer return lists:

- <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> methods <a href="../library/stdtypes.html#dict.keys" class="reference internal" title="dict.keys"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.keys()</code></span></a>, <a href="../library/stdtypes.html#dict.items" class="reference internal" title="dict.items"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.items()</code></span></a> and <a href="../library/stdtypes.html#dict.values" class="reference internal" title="dict.values"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.values()</code></span></a> return “views” instead of lists. For example, this no longer works: <span class="pre">`k`</span>` `<span class="pre">`=`</span>` `<span class="pre">`d.keys();`</span>` `<span class="pre">`k.sort()`</span>. Use <span class="pre">`k`</span>` `<span class="pre">`=`</span>` `<span class="pre">`sorted(d)`</span> instead (this works in Python 2.5 too and is just as efficient).

- Also, the <span class="pre">`dict.iterkeys()`</span>, <span class="pre">`dict.iteritems()`</span> and <span class="pre">`dict.itervalues()`</span> methods are no longer supported.

- <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> and <a href="../library/functions.html#filter" class="reference internal" title="filter"><span class="pre"><code class="sourceCode python"><span class="bu">filter</span>()</code></span></a> return iterators. If you really need a list and the input sequences are all of equal length, a quick fix is to wrap <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> in <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>()</code></span></a>, e.g. <span class="pre">`list(map(...))`</span>, but a better fix is often to use a list comprehension (especially when the original code uses <a href="../reference/expressions.html#lambda" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">lambda</code></span></a>), or rewriting the code so it doesn’t need a list at all. Particularly tricky is <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> invoked for the side effects of the function; the correct transformation is to use a regular <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> loop (since creating a list would just be wasteful).

  If the input sequences are not of equal length, <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> will stop at the termination of the shortest of the sequences. For full compatibility with <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a> from Python 2.x, also wrap the sequences in <a href="../library/itertools.html#itertools.zip_longest" class="reference internal" title="itertools.zip_longest"><span class="pre"><code class="sourceCode python">itertools.zip_longest()</code></span></a>, e.g. <span class="pre">`map(func,`</span>` `<span class="pre">`*sequences)`</span> becomes <span class="pre">`list(map(func,`</span>` `<span class="pre">`itertools.zip_longest(*sequences)))`</span>.

- <a href="../library/stdtypes.html#range" class="reference internal" title="range"><span class="pre"><code class="sourceCode python"><span class="bu">range</span>()</code></span></a> now behaves like <span class="pre">`xrange()`</span> used to behave, except it works with values of arbitrary size. The latter no longer exists.

- <a href="../library/functions.html#zip" class="reference internal" title="zip"><span class="pre"><code class="sourceCode python"><span class="bu">zip</span>()</code></span></a> now returns an iterator.

</div>

<div id="ordering-comparisons" class="section">

### Ordering Comparisons<a href="#ordering-comparisons" class="headerlink" title="Link to this heading">¶</a>

Python 3.0 has simplified the rules for ordering comparisons:

- The ordering comparison operators (<span class="pre">`<`</span>, <span class="pre">`<=`</span>, <span class="pre">`>=`</span>, <span class="pre">`>`</span>) raise a TypeError exception when the operands don’t have a meaningful natural ordering. Thus, expressions like <span class="pre">`1`</span>` `<span class="pre">`<`</span>` `<span class="pre">`''`</span>, <span class="pre">`0`</span>` `<span class="pre">`>`</span>` `<span class="pre">`None`</span> or <span class="pre">`len`</span>` `<span class="pre">`<=`</span>` `<span class="pre">`len`</span> are no longer valid, and e.g. <span class="pre">`None`</span>` `<span class="pre">`<`</span>` `<span class="pre">`None`</span> raises <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> instead of returning <span class="pre">`False`</span>. A corollary is that sorting a heterogeneous list no longer makes sense – all the elements must be comparable to each other. Note that this does not apply to the <span class="pre">`==`</span> and <span class="pre">`!=`</span> operators: objects of different incomparable types always compare unequal to each other.

- <a href="../library/functions.html#sorted" class="reference internal" title="sorted"><span class="pre"><code class="sourceCode python"><span class="bu">sorted</span>()</code></span></a> and <a href="../library/stdtypes.html#list.sort" class="reference internal" title="list.sort"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>.sort()</code></span></a> no longer accept the *cmp* argument providing a comparison function. Use the *key* argument instead. N.B. the *key* and *reverse* arguments are now “keyword-only”.

- The <span class="pre">`cmp()`</span> function should be treated as gone, and the <span class="pre">`__cmp__()`</span> special method is no longer supported. Use <a href="../reference/datamodel.html#object.__lt__" class="reference internal" title="object.__lt__"><span class="pre"><code class="sourceCode python"><span class="fu">__lt__</span>()</code></span></a> for sorting, <a href="../reference/datamodel.html#object.__eq__" class="reference internal" title="object.__eq__"><span class="pre"><code class="sourceCode python"><span class="fu">__eq__</span>()</code></span></a> with <a href="../reference/datamodel.html#object.__hash__" class="reference internal" title="object.__hash__"><span class="pre"><code class="sourceCode python"><span class="fu">__hash__</span>()</code></span></a>, and other rich comparisons as needed. (If you really need the <span class="pre">`cmp()`</span> functionality, you could use the expression <span class="pre">`(a`</span>` `<span class="pre">`>`</span>` `<span class="pre">`b)`</span>` `<span class="pre">`-`</span>` `<span class="pre">`(a`</span>` `<span class="pre">`<`</span>` `<span class="pre">`b)`</span> as the equivalent for <span class="pre">`cmp(a,`</span>` `<span class="pre">`b)`</span>.)

</div>

<div id="integers" class="section">

### Integers<a href="#integers" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0237/" class="pep reference external"><strong>PEP 237</strong></a>: Essentially, <span class="pre">`long`</span> renamed to <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>. That is, there is only one built-in integral type, named <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>; but it behaves mostly like the old <span class="pre">`long`</span> type.

- <span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0238/" class="pep reference external"><strong>PEP 238</strong></a>: An expression like <span class="pre">`1/2`</span> returns a float. Use <span class="pre">`1//2`</span> to get the truncating behavior. (The latter syntax has existed for years, at least since Python 2.2.)

- The <span class="pre">`sys.maxint`</span> constant was removed, since there is no longer a limit to the value of integers. However, <a href="../library/sys.html#sys.maxsize" class="reference internal" title="sys.maxsize"><span class="pre"><code class="sourceCode python">sys.maxsize</code></span></a> can be used as an integer larger than any practical list or string index. It conforms to the implementation’s “natural” integer size and is typically the same as <span class="pre">`sys.maxint`</span> in previous releases on the same platform (assuming the same build options).

- The <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> of a long integer doesn’t include the trailing <span class="pre">`L`</span> anymore, so code that unconditionally strips that character will chop off the last digit instead. (Use <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> instead.)

- Octal literals are no longer of the form <span class="pre">`0720`</span>; use <span class="pre">`0o720`</span> instead.

</div>

<div id="text-vs-data-instead-of-unicode-vs-8-bit" class="section">

### Text Vs. Data Instead Of Unicode Vs. 8-bit<a href="#text-vs-data-instead-of-unicode-vs-8-bit" class="headerlink" title="Link to this heading">¶</a>

Everything you thought you knew about binary data and Unicode has changed.

- Python 3.0 uses the concepts of *text* and (binary) *data* instead of Unicode strings and 8-bit strings. All text is Unicode; however *encoded* Unicode is represented as binary data. The type used to hold text is <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, the type used to hold data is <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>. The biggest difference with the 2.x situation is that any attempt to mix text and data in Python 3.0 raises <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>, whereas if you were to mix Unicode and 8-bit strings in Python 2.x, it would work if the 8-bit string happened to contain only 7-bit (ASCII) bytes, but you would get <a href="../library/exceptions.html#UnicodeDecodeError" class="reference internal" title="UnicodeDecodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeDecodeError</span></code></span></a> if it contained non-ASCII values. This value-specific behavior has caused numerous sad faces over the years.

- As a consequence of this change in philosophy, pretty much all code that uses Unicode, encodings or binary data most likely has to change. The change is for the better, as in the 2.x world there were numerous bugs having to do with mixing encoded and unencoded text. To be prepared in Python 2.x, start using <span class="pre">`unicode`</span> for all unencoded text, and <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> for binary or encoded data only. Then the <span class="pre">`2to3`</span> tool will do most of the work for you.

- You can no longer use <span class="pre">`u"..."`</span> literals for Unicode text. However, you must use <span class="pre">`b"..."`</span> literals for binary data.

- As the <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> and <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> types cannot be mixed, you must always explicitly convert between them. Use <a href="../library/stdtypes.html#str.encode" class="reference internal" title="str.encode"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.encode()</code></span></a> to go from <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> to <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>, and <a href="../library/stdtypes.html#bytes.decode" class="reference internal" title="bytes.decode"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span>.decode()</code></span></a> to go from <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> to <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>. You can also use <span class="pre">`bytes(s,`</span>` `<span class="pre">`encoding=...)`</span> and <span class="pre">`str(b,`</span>` `<span class="pre">`encoding=...)`</span>, respectively.

- Like <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, the <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> type is immutable. There is a separate *mutable* type to hold buffered binary data, <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>. Nearly all APIs that accept <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> also accept <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>. The mutable API is based on <a href="../library/collections.abc.html#collections.abc.MutableSequence" class="reference internal" title="collections.abc.MutableSequence"><span class="pre"><code class="sourceCode python">collections.MutableSequence</code></span></a>.

- All backslashes in raw string literals are interpreted literally. This means that <span class="pre">`'\U'`</span> and <span class="pre">`'\u'`</span> escapes in raw strings are not treated specially. For example, <span class="pre">`r'\u20ac'`</span> is a string of 6 characters in Python 3.0, whereas in 2.6, <span class="pre">`ur'\u20ac'`</span> was the single “euro” character. (Of course, this change only affects raw string literals; the euro character is <span class="pre">`'\u20ac'`</span> in Python 3.0.)

- The built-in <span class="pre">`basestring`</span> abstract type was removed. Use <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> instead. The <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> and <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> types don’t have functionality enough in common to warrant a shared base class. The <span class="pre">`2to3`</span> tool (see below) replaces every occurrence of <span class="pre">`basestring`</span> with <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>.

- Files opened as text files (still the default mode for <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a>) always use an encoding to map between strings (in memory) and bytes (on disk). Binary files (opened with a <span class="pre">`b`</span> in the mode argument) always use bytes in memory. This means that if a file is opened using an incorrect mode or encoding, I/O will likely fail loudly, instead of silently producing incorrect data. It also means that even Unix users will have to specify the correct mode (text or binary) when opening a file. There is a platform-dependent default encoding, which on Unixy platforms can be set with the <span class="pre">`LANG`</span> environment variable (and sometimes also with some other platform-specific locale-related environment variables). In many cases, but not all, the system default is UTF-8; you should never count on this default. Any application reading or writing more than pure ASCII text should probably have a way to override the encoding. There is no longer any need for using the encoding-aware streams in the <a href="../library/codecs.html#module-codecs" class="reference internal" title="codecs: Encode and decode data and streams."><span class="pre"><code class="sourceCode python">codecs</code></span></a> module.

- The initial values of <a href="../library/sys.html#sys.stdin" class="reference internal" title="sys.stdin"><span class="pre"><code class="sourceCode python">sys.stdin</code></span></a>, <a href="../library/sys.html#sys.stdout" class="reference internal" title="sys.stdout"><span class="pre"><code class="sourceCode python">sys.stdout</code></span></a> and <a href="../library/sys.html#sys.stderr" class="reference internal" title="sys.stderr"><span class="pre"><code class="sourceCode python">sys.stderr</code></span></a> are now unicode-only text files (i.e., they are instances of <a href="../library/io.html#io.TextIOBase" class="reference internal" title="io.TextIOBase"><span class="pre"><code class="sourceCode python">io.TextIOBase</code></span></a>). To read and write bytes data with these streams, you need to use their <a href="../library/io.html#io.TextIOBase.buffer" class="reference internal" title="io.TextIOBase.buffer"><span class="pre"><code class="sourceCode python">io.TextIOBase.<span class="bu">buffer</span></code></span></a> attribute.

- Filenames are passed to and returned from APIs as (Unicode) strings. This can present platform-specific problems because on some platforms filenames are arbitrary byte strings. (On the other hand, on Windows filenames are natively stored as Unicode.) As a work-around, most APIs (e.g. <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> and many functions in the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module) that take filenames accept <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> objects as well as strings, and a few APIs have a way to ask for a <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> return value. Thus, <a href="../library/os.html#os.listdir" class="reference internal" title="os.listdir"><span class="pre"><code class="sourceCode python">os.listdir()</code></span></a> returns a list of <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> instances if the argument is a <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> instance, and <a href="../library/os.html#os.getcwdb" class="reference internal" title="os.getcwdb"><span class="pre"><code class="sourceCode python">os.getcwdb()</code></span></a> returns the current working directory as a <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> instance. Note that when <a href="../library/os.html#os.listdir" class="reference internal" title="os.listdir"><span class="pre"><code class="sourceCode python">os.listdir()</code></span></a> returns a list of strings, filenames that cannot be decoded properly are omitted rather than raising <a href="../library/exceptions.html#UnicodeError" class="reference internal" title="UnicodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeError</span></code></span></a>.

- Some system APIs like <a href="../library/os.html#os.environ" class="reference internal" title="os.environ"><span class="pre"><code class="sourceCode python">os.environ</code></span></a> and <a href="../library/sys.html#sys.argv" class="reference internal" title="sys.argv"><span class="pre"><code class="sourceCode python">sys.argv</code></span></a> can also present problems when the bytes made available by the system is not interpretable using the default encoding. Setting the <span class="pre">`LANG`</span> variable and rerunning the program is probably the best approach.

- <span id="index-3" class="target"></span><a href="https://peps.python.org/pep-3138/" class="pep reference external"><strong>PEP 3138</strong></a>: The <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> of a string no longer escapes non-ASCII characters. It still escapes control characters and code points with non-printable status in the Unicode standard, however.

- <span id="index-4" class="target"></span><a href="https://peps.python.org/pep-3120/" class="pep reference external"><strong>PEP 3120</strong></a>: The default source encoding is now UTF-8.

- <span id="index-5" class="target"></span><a href="https://peps.python.org/pep-3131/" class="pep reference external"><strong>PEP 3131</strong></a>: Non-ASCII letters are now allowed in identifiers. (However, the standard library remains ASCII-only with the exception of contributor names in comments.)

- The <span class="pre">`StringIO`</span> and <span class="pre">`cStringIO`</span> modules are gone. Instead, import the <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module and use <a href="../library/io.html#io.StringIO" class="reference internal" title="io.StringIO"><span class="pre"><code class="sourceCode python">io.StringIO</code></span></a> or <a href="../library/io.html#io.BytesIO" class="reference internal" title="io.BytesIO"><span class="pre"><code class="sourceCode python">io.BytesIO</code></span></a> for text and data respectively.

- See also the <a href="../howto/unicode.html#unicode-howto" class="reference internal"><span class="std std-ref">Unicode HOWTO</span></a>, which was updated for Python 3.0.

</div>

</div>

<div id="overview-of-syntax-changes" class="section">

## Overview Of Syntax Changes<a href="#overview-of-syntax-changes" class="headerlink" title="Link to this heading">¶</a>

This section gives a brief overview of every *syntactic* change in Python 3.0.

<div id="new-syntax" class="section">

### New Syntax<a href="#new-syntax" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-6" class="target"></span><a href="https://peps.python.org/pep-3107/" class="pep reference external"><strong>PEP 3107</strong></a>: Function argument and return value annotations. This provides a standardized way of annotating a function’s parameters and return value. There are no semantics attached to such annotations except that they can be introspected at runtime using the <span class="pre">`__annotations__`</span> attribute. The intent is to encourage experimentation through metaclasses, decorators or frameworks.

- <span id="index-7" class="target"></span><a href="https://peps.python.org/pep-3102/" class="pep reference external"><strong>PEP 3102</strong></a>: Keyword-only arguments. Named parameters occurring after <span class="pre">`*args`</span> in the parameter list *must* be specified using keyword syntax in the call. You can also use a bare <span class="pre">`*`</span> in the parameter list to indicate that you don’t accept a variable-length argument list, but you do have keyword-only arguments.

- Keyword arguments are allowed after the list of base classes in a class definition. This is used by the new convention for specifying a metaclass (see next section), but can be used for other purposes as well, as long as the metaclass supports it.

- <span id="index-8" class="target"></span><a href="https://peps.python.org/pep-3104/" class="pep reference external"><strong>PEP 3104</strong></a>: <a href="../reference/simple_stmts.html#nonlocal" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">nonlocal</code></span></a> statement. Using <span class="pre">`nonlocal`</span>` `<span class="pre">`x`</span> you can now assign directly to a variable in an outer (but non-global) scope. <span class="pre">`nonlocal`</span> is a new reserved word.

- <span id="index-9" class="target"></span><a href="https://peps.python.org/pep-3132/" class="pep reference external"><strong>PEP 3132</strong></a>: Extended Iterable Unpacking. You can now write things like <span class="pre">`a,`</span>` `<span class="pre">`b,`</span>` `<span class="pre">`*rest`</span>` `<span class="pre">`=`</span>` `<span class="pre">`some_sequence`</span>. And even <span class="pre">`*rest,`</span>` `<span class="pre">`a`</span>` `<span class="pre">`=`</span>` `<span class="pre">`stuff`</span>. The <span class="pre">`rest`</span> object is always a (possibly empty) list; the right-hand side may be any iterable. Example:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      (a, *rest, b) = range(5)

  </div>

  </div>

  This sets *a* to <span class="pre">`0`</span>, *b* to <span class="pre">`4`</span>, and *rest* to <span class="pre">`[1,`</span>` `<span class="pre">`2,`</span>` `<span class="pre">`3]`</span>.

- Dictionary comprehensions: <span class="pre">`{k:`</span>` `<span class="pre">`v`</span>` `<span class="pre">`for`</span>` `<span class="pre">`k,`</span>` `<span class="pre">`v`</span>` `<span class="pre">`in`</span>` `<span class="pre">`stuff}`</span> means the same thing as <span class="pre">`dict(stuff)`</span> but is more flexible. (This is <span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0274/" class="pep reference external"><strong>PEP 274</strong></a> vindicated. :-)

- Set literals, e.g. <span class="pre">`{1,`</span>` `<span class="pre">`2}`</span>. Note that <span class="pre">`{}`</span> is an empty dictionary; use <span class="pre">`set()`</span> for an empty set. Set comprehensions are also supported; e.g., <span class="pre">`{x`</span>` `<span class="pre">`for`</span>` `<span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`stuff}`</span> means the same thing as <span class="pre">`set(stuff)`</span> but is more flexible.

- New octal literals, e.g. <span class="pre">`0o720`</span> (already in 2.6). The old octal literals (<span class="pre">`0720`</span>) are gone.

- New binary literals, e.g. <span class="pre">`0b1010`</span> (already in 2.6), and there is a new corresponding built-in function, <a href="../library/functions.html#bin" class="reference internal" title="bin"><span class="pre"><code class="sourceCode python"><span class="bu">bin</span>()</code></span></a>.

- Bytes literals are introduced with a leading <span class="pre">`b`</span> or <span class="pre">`B`</span>, and there is a new corresponding built-in function, <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span>()</code></span></a>.

</div>

<div id="changed-syntax" class="section">

### Changed Syntax<a href="#changed-syntax" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-11" class="target"></span><a href="https://peps.python.org/pep-3109/" class="pep reference external"><strong>PEP 3109</strong></a> and <span id="index-12" class="target"></span><a href="https://peps.python.org/pep-3134/" class="pep reference external"><strong>PEP 3134</strong></a>: new <a href="../reference/simple_stmts.html#raise" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">raise</code></span></a> statement syntax: <span class="pre">`raise`</span>` `<span class="pre">`[`</span>*<span class="pre">`expr`</span>*` `<span class="pre">`[from`</span>` `*<span class="pre">`expr`</span>*<span class="pre">`]]`</span>. See below.

- <span class="pre">`as`</span> and <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> are now reserved words. (Since 2.6, actually.)

- <span class="pre">`True`</span>, <span class="pre">`False`</span>, and <span class="pre">`None`</span> are reserved words. (2.6 partially enforced the restrictions on <span class="pre">`None`</span> already.)

- Change from <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> *exc*, *var* to <span class="pre">`except`</span> *exc* <span class="pre">`as`</span> *var*. See <span id="index-13" class="target"></span><a href="https://peps.python.org/pep-3110/" class="pep reference external"><strong>PEP 3110</strong></a>.

- <span id="index-14" class="target"></span><a href="https://peps.python.org/pep-3115/" class="pep reference external"><strong>PEP 3115</strong></a>: New Metaclass Syntax. Instead of:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      class C:
          __metaclass__ = M
          ...

  </div>

  </div>

  you must now use:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      class C(metaclass=M):
          ...

  </div>

  </div>

  The module-global <span class="pre">`__metaclass__`</span> variable is no longer supported. (It was a crutch to make it easier to default to new-style classes without deriving every class from <a href="../library/functions.html#object" class="reference internal" title="object"><span class="pre"><code class="sourceCode python"><span class="bu">object</span></code></span></a>.)

- List comprehensions no longer support the syntactic form <span class="pre">`[...`</span>` `<span class="pre">`for`</span>` `*<span class="pre">`var`</span>*` `<span class="pre">`in`</span>` `*<span class="pre">`item1`</span>*<span class="pre">`,`</span>` `*<span class="pre">`item2`</span>*<span class="pre">`,`</span>` `<span class="pre">`...]`</span>. Use <span class="pre">`[...`</span>` `<span class="pre">`for`</span>` `*<span class="pre">`var`</span>*` `<span class="pre">`in`</span>` `<span class="pre">`(`</span>*<span class="pre">`item1`</span>*<span class="pre">`,`</span>` `*<span class="pre">`item2`</span>*<span class="pre">`,`</span>` `<span class="pre">`...)]`</span> instead. Also note that list comprehensions have different semantics: they are closer to syntactic sugar for a generator expression inside a <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span>()</code></span></a> constructor, and in particular the loop control variables are no longer leaked into the surrounding scope.

- The *ellipsis* (<span class="pre">`...`</span>) can be used as an atomic expression anywhere. (Previously it was only allowed in slices.) Also, it *must* now be spelled as <span class="pre">`...`</span>. (Previously it could also be spelled as <span class="pre">`.`</span>` `<span class="pre">`.`</span>` `<span class="pre">`.`</span>, by a mere accident of the grammar.)

</div>

<div id="removed-syntax" class="section">

### Removed Syntax<a href="#removed-syntax" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-15" class="target"></span><a href="https://peps.python.org/pep-3113/" class="pep reference external"><strong>PEP 3113</strong></a>: Tuple parameter unpacking removed. You can no longer write <span class="pre">`def`</span>` `<span class="pre">`foo(a,`</span>` `<span class="pre">`(b,`</span>` `<span class="pre">`c)):`</span>` `<span class="pre">`...`</span>. Use <span class="pre">`def`</span>` `<span class="pre">`foo(a,`</span>` `<span class="pre">`b_c):`</span>` `<span class="pre">`b,`</span>` `<span class="pre">`c`</span>` `<span class="pre">`=`</span>` `<span class="pre">`b_c`</span> instead.

- Removed backticks (use <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> instead).

- Removed <span class="pre">`<>`</span> (use <span class="pre">`!=`</span> instead).

- Removed keyword: <a href="../library/functions.html#exec" class="reference internal" title="exec"><span class="pre"><code class="sourceCode python"><span class="bu">exec</span>()</code></span></a> is no longer a keyword; it remains as a function. (Fortunately the function syntax was also accepted in 2.x.) Also note that <a href="../library/functions.html#exec" class="reference internal" title="exec"><span class="pre"><code class="sourceCode python"><span class="bu">exec</span>()</code></span></a> no longer takes a stream argument; instead of <span class="pre">`exec(f)`</span> you can use <span class="pre">`exec(f.read())`</span>.

- Integer literals no longer support a trailing <span class="pre">`l`</span> or <span class="pre">`L`</span>.

- String literals no longer support a leading <span class="pre">`u`</span> or <span class="pre">`U`</span>.

- The <a href="../reference/simple_stmts.html#from" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">from</code></span></a> *module* <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> <span class="pre">`*`</span> syntax is only allowed at the module level, no longer inside functions.

- The only acceptable syntax for relative imports is <span class="pre">`from`</span>` `<span class="pre">`.[`</span>*<span class="pre">`module`</span>*<span class="pre">`]`</span>` `<span class="pre">`import`</span>` `*<span class="pre">`name`</span>*. All <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> forms not starting with <span class="pre">`.`</span> are interpreted as absolute imports. (<span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0328/" class="pep reference external"><strong>PEP 328</strong></a>)

- Classic classes are gone.

</div>

</div>

<div id="changes-already-present-in-python-2-6" class="section">

## Changes Already Present In Python 2.6<a href="#changes-already-present-in-python-2-6" class="headerlink" title="Link to this heading">¶</a>

Since many users presumably make the jump straight from Python 2.5 to Python 3.0, this section reminds the reader of new features that were originally designed for Python 3.0 but that were back-ported to Python 2.6. The corresponding sections in <a href="2.6.html#whats-new-in-2-6" class="reference internal"><span class="std std-ref">What’s New in Python 2.6</span></a> should be consulted for longer descriptions.

- <a href="2.6.html#pep-0343" class="reference internal"><span class="std std-ref">PEP 343: The ‘with’ statement</span></a>. The <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement is now a standard feature and no longer needs to be imported from the <a href="../library/__future__.html#module-__future__" class="reference internal" title="__future__: Future statement definitions"><span class="pre"><code class="sourceCode python">__future__</code></span></a>. Also check out <a href="2.6.html#new-26-context-managers" class="reference internal"><span class="std std-ref">Writing Context Managers</span></a> and <a href="2.6.html#new-module-contextlib" class="reference internal"><span class="std std-ref">The contextlib module</span></a>.

- <a href="2.6.html#pep-0366" class="reference internal"><span class="std std-ref">PEP 366: Explicit Relative Imports From a Main Module</span></a>. This enhances the usefulness of the <a href="../using/cmdline.html#cmdoption-m" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-m</code></span></a> option when the referenced module lives in a package.

- <a href="2.6.html#pep-0370" class="reference internal"><span class="std std-ref">PEP 370: Per-user site-packages Directory</span></a>.

- <a href="2.6.html#pep-0371" class="reference internal"><span class="std std-ref">PEP 371: The multiprocessing Package</span></a>.

- <a href="2.6.html#pep-3101" class="reference internal"><span class="std std-ref">PEP 3101: Advanced String Formatting</span></a>. Note: the 2.6 description mentions the <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> method for both 8-bit and Unicode strings. In 3.0, only the <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> type (text strings with Unicode support) supports this method; the <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> type does not. The plan is to eventually make this the only API for string formatting, and to start deprecating the <span class="pre">`%`</span> operator in Python 3.1.

- <a href="2.6.html#pep-3105" class="reference internal"><span class="std std-ref">PEP 3105: print As a Function</span></a>. This is now a standard feature and no longer needs to be imported from <a href="../library/__future__.html#module-__future__" class="reference internal" title="__future__: Future statement definitions"><span class="pre"><code class="sourceCode python">__future__</code></span></a>. More details were given above.

- <a href="2.6.html#pep-3110" class="reference internal"><span class="std std-ref">PEP 3110: Exception-Handling Changes</span></a>. The <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> *exc* <span class="pre">`as`</span> *var* syntax is now standard and <span class="pre">`except`</span> *exc*, *var* is no longer supported. (Of course, the <span class="pre">`as`</span> *var* part is still optional.)

- <a href="2.6.html#pep-3112" class="reference internal"><span class="std std-ref">PEP 3112: Byte Literals</span></a>. The <span class="pre">`b"..."`</span> string literal notation (and its variants like <span class="pre">`b'...'`</span>, <span class="pre">`b"""..."""`</span>, and <span class="pre">`br"..."`</span>) now produces a literal of type <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>.

- <a href="2.6.html#pep-3116" class="reference internal"><span class="std std-ref">PEP 3116: New I/O Library</span></a>. The <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module is now the standard way of doing file I/O. The built-in <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> function is now an alias for <a href="../library/io.html#io.open" class="reference internal" title="io.open"><span class="pre"><code class="sourceCode python">io.<span class="bu">open</span>()</code></span></a> and has additional keyword arguments *encoding*, *errors*, *newline* and *closefd*. Also note that an invalid *mode* argument now raises <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a>, not <a href="../library/exceptions.html#IOError" class="reference internal" title="IOError"><span class="pre"><code class="sourceCode python"><span class="pp">IOError</span></code></span></a>. The binary file object underlying a text file object can be accessed as <span class="pre">`f.buffer`</span> (but beware that the text object maintains a buffer of itself in order to speed up the encoding and decoding operations).

- <a href="2.6.html#pep-3118" class="reference internal"><span class="std std-ref">PEP 3118: Revised Buffer Protocol</span></a>. The old builtin <span class="pre">`buffer()`</span> is now really gone; the new builtin <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span>()</code></span></a> provides (mostly) similar functionality.

- <a href="2.6.html#pep-3119" class="reference internal"><span class="std std-ref">PEP 3119: Abstract Base Classes</span></a>. The <a href="../library/abc.html#module-abc" class="reference internal" title="abc: Abstract base classes according to :pep:`3119`."><span class="pre"><code class="sourceCode python">abc</code></span></a> module and the ABCs defined in the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: Container datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module plays a somewhat more prominent role in the language now, and built-in collection types like <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span></code></span></a> and <a href="../library/stdtypes.html#list" class="reference internal" title="list"><span class="pre"><code class="sourceCode python"><span class="bu">list</span></code></span></a> conform to the <a href="../library/collections.abc.html#collections.abc.MutableMapping" class="reference internal" title="collections.abc.MutableMapping"><span class="pre"><code class="sourceCode python">collections.MutableMapping</code></span></a> and <a href="../library/collections.abc.html#collections.abc.MutableSequence" class="reference internal" title="collections.abc.MutableSequence"><span class="pre"><code class="sourceCode python">collections.MutableSequence</code></span></a> ABCs, respectively.

- <a href="2.6.html#pep-3127" class="reference internal"><span class="std std-ref">PEP 3127: Integer Literal Support and Syntax</span></a>. As mentioned above, the new octal literal notation is the only one supported, and binary literals have been added.

- <a href="2.6.html#pep-3129" class="reference internal"><span class="std std-ref">PEP 3129: Class Decorators</span></a>.

- <a href="2.6.html#pep-3141" class="reference internal"><span class="std std-ref">PEP 3141: A Type Hierarchy for Numbers</span></a>. The <a href="../library/numbers.html#module-numbers" class="reference internal" title="numbers: Numeric abstract base classes (Complex, Real, Integral, etc.)."><span class="pre"><code class="sourceCode python">numbers</code></span></a> module is another new use of ABCs, defining Python’s “numeric tower”. Also note the new <a href="../library/fractions.html#module-fractions" class="reference internal" title="fractions: Rational numbers."><span class="pre"><code class="sourceCode python">fractions</code></span></a> module which implements <a href="../library/numbers.html#numbers.Rational" class="reference internal" title="numbers.Rational"><span class="pre"><code class="sourceCode python">numbers.Rational</code></span></a>.

</div>

<div id="library-changes" class="section">

## Library Changes<a href="#library-changes" class="headerlink" title="Link to this heading">¶</a>

Due to time constraints, this document does not exhaustively cover the very extensive changes to the standard library. <span id="index-17" class="target"></span><a href="https://peps.python.org/pep-3108/" class="pep reference external"><strong>PEP 3108</strong></a> is the reference for the major changes to the library. Here’s a capsule review:

- Many old modules were removed. Some, like <span class="pre">`gopherlib`</span> (no longer used) and <span class="pre">`md5`</span> (replaced by <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a>), were already deprecated by <span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0004/" class="pep reference external"><strong>PEP 4</strong></a>. Others were removed as a result of the removal of support for various platforms such as Irix, BeOS and Mac OS 9 (see <span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0011/" class="pep reference external"><strong>PEP 11</strong></a>). Some modules were also selected for removal in Python 3.0 due to lack of use or because a better replacement exists. See <span id="index-20" class="target"></span><a href="https://peps.python.org/pep-3108/" class="pep reference external"><strong>PEP 3108</strong></a> for an exhaustive list.

- The <span class="pre">`bsddb3`</span> package was removed because its presence in the core standard library has proved over time to be a particular burden for the core developers due to testing instability and Berkeley DB’s release schedule. However, the package is alive and well, externally maintained at <a href="https://www.jcea.es/programacion/pybsddb.htm" class="reference external">https://www.jcea.es/programacion/pybsddb.htm</a>.

- Some modules were renamed because their old name disobeyed <span id="index-21" class="target"></span><a href="https://peps.python.org/pep-0008/" class="pep reference external"><strong>PEP 8</strong></a>, or for various other reasons. Here’s the list:

  | Old Name          | New Name     |
  |-------------------|--------------|
  | \_winreg          | winreg       |
  | ConfigParser      | configparser |
  | copy_reg          | copyreg      |
  | Queue             | queue        |
  | SocketServer      | socketserver |
  | markupbase        | \_markupbase |
  | repr              | reprlib      |
  | test.test_support | test.support |

- A common pattern in Python 2.x is to have one version of a module implemented in pure Python, with an optional accelerated version implemented as a C extension; for example, <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> and <span class="pre">`cPickle`</span>. This places the burden of importing the accelerated version and falling back on the pure Python version on each user of these modules. In Python 3.0, the accelerated versions are considered implementation details of the pure Python versions. Users should always import the standard version, which attempts to import the accelerated version and falls back to the pure Python version. The <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> / <span class="pre">`cPickle`</span> pair received this treatment. The <a href="../library/profile.html#module-profile" class="reference internal" title="profile: Python source profiler."><span class="pre"><code class="sourceCode python">profile</code></span></a> module is on the list for 3.1. The <span class="pre">`StringIO`</span> module has been turned into a class in the <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module.

- Some related modules have been grouped into packages, and usually the submodule names have been simplified. The resulting new packages are:

  - <a href="../library/dbm.html#module-dbm" class="reference internal" title="dbm: Interfaces to various Unix &quot;database&quot; formats."><span class="pre"><code class="sourceCode python">dbm</code></span></a> (<span class="pre">`anydbm`</span>, <span class="pre">`dbhash`</span>, <span class="pre">`dbm`</span>, <span class="pre">`dumbdbm`</span>, <span class="pre">`gdbm`</span>, <span class="pre">`whichdb`</span>).

  - <a href="../library/html.html#module-html" class="reference internal" title="html: Helpers for manipulating HTML."><span class="pre"><code class="sourceCode python">html</code></span></a> (<span class="pre">`HTMLParser`</span>, <span class="pre">`htmlentitydefs`</span>).

  - <a href="../library/http.html#module-http" class="reference internal" title="http: HTTP status codes and messages"><span class="pre"><code class="sourceCode python">http</code></span></a> (<span class="pre">`httplib`</span>, <span class="pre">`BaseHTTPServer`</span>, <span class="pre">`CGIHTTPServer`</span>, <span class="pre">`SimpleHTTPServer`</span>, <span class="pre">`Cookie`</span>, <span class="pre">`cookielib`</span>).

  - <a href="../library/tkinter.html#module-tkinter" class="reference internal" title="tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">tkinter</code></span></a> (all <span class="pre">`Tkinter`</span>-related modules except <a href="../library/turtle.html#module-turtle" class="reference internal" title="turtle: An educational framework for simple graphics applications"><span class="pre"><code class="sourceCode python">turtle</code></span></a>). The target audience of <a href="../library/turtle.html#module-turtle" class="reference internal" title="turtle: An educational framework for simple graphics applications"><span class="pre"><code class="sourceCode python">turtle</code></span></a> doesn’t really care about <a href="../library/tkinter.html#module-tkinter" class="reference internal" title="tkinter: Interface to Tcl/Tk for graphical user interfaces"><span class="pre"><code class="sourceCode python">tkinter</code></span></a>. Also note that as of Python 2.6, the functionality of <a href="../library/turtle.html#module-turtle" class="reference internal" title="turtle: An educational framework for simple graphics applications"><span class="pre"><code class="sourceCode python">turtle</code></span></a> has been greatly enhanced.

  - <a href="../library/urllib.html#module-urllib" class="reference internal" title="urllib"><span class="pre"><code class="sourceCode python">urllib</code></span></a> (<span class="pre">`urllib`</span>, <span class="pre">`urllib2`</span>, <span class="pre">`urlparse`</span>, <span class="pre">`robotparse`</span>).

  - <a href="../library/xmlrpc.html#module-xmlrpc" class="reference internal" title="xmlrpc: Server and client modules implementing XML-RPC."><span class="pre"><code class="sourceCode python">xmlrpc</code></span></a> (<span class="pre">`xmlrpclib`</span>, <span class="pre">`DocXMLRPCServer`</span>, <span class="pre">`SimpleXMLRPCServer`</span>).

Some other changes to standard library modules, not covered by <span id="index-22" class="target"></span><a href="https://peps.python.org/pep-3108/" class="pep reference external"><strong>PEP 3108</strong></a>:

- Killed <span class="pre">`sets`</span>. Use the built-in <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span>()</code></span></a> class.

- Cleanup of the <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> module: removed <span class="pre">`sys.exitfunc()`</span>, <span class="pre">`sys.exc_clear()`</span>, <span class="pre">`sys.exc_type`</span>, <span class="pre">`sys.exc_value`</span>, <span class="pre">`sys.exc_traceback`</span>. (Note that <a href="../library/sys.html#sys.last_type" class="reference internal" title="sys.last_type"><span class="pre"><code class="sourceCode python">sys.last_type</code></span></a> etc. remain.)

- Cleanup of the <a href="../library/array.html#array.array" class="reference internal" title="array.array"><span class="pre"><code class="sourceCode python">array.array</code></span></a> type: the <span class="pre">`read()`</span> and <span class="pre">`write()`</span> methods are gone; use <a href="../library/array.html#array.array.fromfile" class="reference internal" title="array.array.fromfile"><span class="pre"><code class="sourceCode python">fromfile()</code></span></a> and <a href="../library/array.html#array.array.tofile" class="reference internal" title="array.array.tofile"><span class="pre"><code class="sourceCode python">tofile()</code></span></a> instead. Also, the <span class="pre">`'c'`</span> typecode for array is gone – use either <span class="pre">`'b'`</span> for bytes or <span class="pre">`'u'`</span> for Unicode characters.

- Cleanup of the <a href="../library/operator.html#module-operator" class="reference internal" title="operator: Functions corresponding to the standard operators."><span class="pre"><code class="sourceCode python">operator</code></span></a> module: removed <span class="pre">`sequenceIncludes()`</span> and <span class="pre">`isCallable()`</span>.

- Cleanup of the <span class="pre">`thread`</span> module: <span class="pre">`acquire_lock()`</span> and <span class="pre">`release_lock()`</span> are gone; use <a href="../library/threading.html#threading.Lock.acquire" class="reference internal" title="threading.Lock.acquire"><span class="pre"><code class="sourceCode python">acquire()</code></span></a> and <a href="../library/threading.html#threading.Lock.release" class="reference internal" title="threading.Lock.release"><span class="pre"><code class="sourceCode python">release()</code></span></a> instead.

- Cleanup of the <a href="../library/random.html#module-random" class="reference internal" title="random: Generate pseudo-random numbers with various common distributions."><span class="pre"><code class="sourceCode python">random</code></span></a> module: removed the <span class="pre">`jumpahead()`</span> API.

- The <span class="pre">`new`</span> module is gone.

- The functions <span class="pre">`os.tmpnam()`</span>, <span class="pre">`os.tempnam()`</span> and <span class="pre">`os.tmpfile()`</span> have been removed in favor of the <a href="../library/tempfile.html#module-tempfile" class="reference internal" title="tempfile: Generate temporary files and directories."><span class="pre"><code class="sourceCode python">tempfile</code></span></a> module.

- The <a href="../library/tokenize.html#module-tokenize" class="reference internal" title="tokenize: Lexical scanner for Python source code."><span class="pre"><code class="sourceCode python">tokenize</code></span></a> module has been changed to work with bytes. The main entry point is now <a href="../library/tokenize.html#tokenize.tokenize" class="reference internal" title="tokenize.tokenize"><span class="pre"><code class="sourceCode python">tokenize.tokenize()</code></span></a>, instead of generate_tokens.

- <span class="pre">`string.letters`</span> and its friends (<span class="pre">`string.lowercase`</span> and <span class="pre">`string.uppercase`</span>) are gone. Use <a href="../library/string.html#string.ascii_letters" class="reference internal" title="string.ascii_letters"><span class="pre"><code class="sourceCode python">string.ascii_letters</code></span></a> etc. instead. (The reason for the removal is that <span class="pre">`string.letters`</span> and friends had locale-specific behavior, which is a bad idea for such attractively named global “constants”.)

- Renamed module <span class="pre">`__builtin__`</span> to <a href="../library/builtins.html#module-builtins" class="reference internal" title="builtins: The module that provides the built-in namespace."><span class="pre"><code class="sourceCode python">builtins</code></span></a> (removing the underscores, adding an ‘s’). The <span class="pre">`__builtins__`</span> variable found in most global namespaces is unchanged. To modify a builtin, you should use <a href="../library/builtins.html#module-builtins" class="reference internal" title="builtins: The module that provides the built-in namespace."><span class="pre"><code class="sourceCode python">builtins</code></span></a>, not <span class="pre">`__builtins__`</span>!

</div>

<div id="pep-3101-a-new-approach-to-string-formatting" class="section">

## <span id="index-23" class="target"></span><a href="https://peps.python.org/pep-3101/" class="pep reference external"><strong>PEP 3101</strong></a>: A New Approach To String Formatting<a href="#pep-3101-a-new-approach-to-string-formatting" class="headerlink" title="Link to this heading">¶</a>

- A new system for built-in string formatting operations replaces the <span class="pre">`%`</span> string formatting operator. (However, the <span class="pre">`%`</span> operator is still supported; it will be deprecated in Python 3.1 and removed from the language at some later time.) Read <span id="index-24" class="target"></span><a href="https://peps.python.org/pep-3101/" class="pep reference external"><strong>PEP 3101</strong></a> for the full scoop.

</div>

<div id="changes-to-exceptions" class="section">

## Changes To Exceptions<a href="#changes-to-exceptions" class="headerlink" title="Link to this heading">¶</a>

The APIs for raising and catching exception have been cleaned up and new powerful features added:

- <span id="index-25" class="target"></span><a href="https://peps.python.org/pep-0352/" class="pep reference external"><strong>PEP 352</strong></a>: All exceptions must be derived (directly or indirectly) from <a href="../library/exceptions.html#BaseException" class="reference internal" title="BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a>. This is the root of the exception hierarchy. This is not new as a recommendation, but the *requirement* to inherit from <a href="../library/exceptions.html#BaseException" class="reference internal" title="BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a> is new. (Python 2.6 still allowed classic classes to be raised, and placed no restriction on what you can catch.) As a consequence, string exceptions are finally truly and utterly dead.

- Almost all exceptions should actually derive from <a href="../library/exceptions.html#Exception" class="reference internal" title="Exception"><span class="pre"><code class="sourceCode python"><span class="pp">Exception</span></code></span></a>; <a href="../library/exceptions.html#BaseException" class="reference internal" title="BaseException"><span class="pre"><code class="sourceCode python"><span class="pp">BaseException</span></code></span></a> should only be used as a base class for exceptions that should only be handled at the top level, such as <a href="../library/exceptions.html#SystemExit" class="reference internal" title="SystemExit"><span class="pre"><code class="sourceCode python"><span class="pp">SystemExit</span></code></span></a> or <a href="../library/exceptions.html#KeyboardInterrupt" class="reference internal" title="KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a>. The recommended idiom for handling all exceptions except for this latter category is to use <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> <a href="../library/exceptions.html#Exception" class="reference internal" title="Exception"><span class="pre"><code class="sourceCode python"><span class="pp">Exception</span></code></span></a>.

- <span class="pre">`StandardError`</span> was removed.

- Exceptions no longer behave as sequences. Use the <a href="../library/exceptions.html#BaseException.args" class="reference internal" title="BaseException.args"><span class="pre"><code class="sourceCode python">args</code></span></a> attribute instead.

- <span id="index-26" class="target"></span><a href="https://peps.python.org/pep-3109/" class="pep reference external"><strong>PEP 3109</strong></a>: Raising exceptions. You must now use <span class="pre">`raise`</span>` `*<span class="pre">`Exception`</span>*<span class="pre">`(`</span>*<span class="pre">`args`</span>*<span class="pre">`)`</span> instead of <span class="pre">`raise`</span>` `*<span class="pre">`Exception`</span>*<span class="pre">`,`</span>` `*<span class="pre">`args`</span>*. Additionally, you can no longer explicitly specify a traceback; instead, if you *have* to do this, you can assign directly to the <a href="../library/exceptions.html#BaseException.__traceback__" class="reference internal" title="BaseException.__traceback__"><span class="pre"><code class="sourceCode python">__traceback__</code></span></a> attribute (see below).

- <span id="index-27" class="target"></span><a href="https://peps.python.org/pep-3110/" class="pep reference external"><strong>PEP 3110</strong></a>: Catching exceptions. You must now use <span class="pre">`except`</span>` `*<span class="pre">`SomeException`</span>*` `<span class="pre">`as`</span>` `*<span class="pre">`variable`</span>* instead of <span class="pre">`except`</span>` `*<span class="pre">`SomeException`</span>*<span class="pre">`,`</span>` `*<span class="pre">`variable`</span>*. Moreover, the *variable* is explicitly deleted when the <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> block is left.

- <span id="index-28" class="target"></span><a href="https://peps.python.org/pep-3134/" class="pep reference external"><strong>PEP 3134</strong></a>: Exception chaining. There are two cases: implicit chaining and explicit chaining. Implicit chaining happens when an exception is raised in an <a href="../reference/compound_stmts.html#except" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">except</code></span></a> or <a href="../reference/compound_stmts.html#finally" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">finally</code></span></a> handler block. This usually happens due to a bug in the handler block; we call this a *secondary* exception. In this case, the original exception (that was being handled) is saved as the <a href="../library/exceptions.html#BaseException.__context__" class="reference internal" title="BaseException.__context__"><span class="pre"><code class="sourceCode python">__context__</code></span></a> attribute of the secondary exception. Explicit chaining is invoked with this syntax:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      raise SecondaryException() from primary_exception

  </div>

  </div>

  (where *primary_exception* is any expression that produces an exception object, probably an exception that was previously caught). In this case, the primary exception is stored on the <a href="../library/exceptions.html#BaseException.__cause__" class="reference internal" title="BaseException.__cause__"><span class="pre"><code class="sourceCode python">__cause__</code></span></a> attribute of the secondary exception. The traceback printed when an unhandled exception occurs walks the chain of <span class="pre">`__cause__`</span> and <a href="../library/exceptions.html#BaseException.__context__" class="reference internal" title="BaseException.__context__"><span class="pre"><code class="sourceCode python">__context__</code></span></a> attributes and prints a separate traceback for each component of the chain, with the primary exception at the top. (Java users may recognize this behavior.)

- <span id="index-29" class="target"></span><a href="https://peps.python.org/pep-3134/" class="pep reference external"><strong>PEP 3134</strong></a>: Exception objects now store their traceback as the <a href="../library/exceptions.html#BaseException.__traceback__" class="reference internal" title="BaseException.__traceback__"><span class="pre"><code class="sourceCode python">__traceback__</code></span></a> attribute. This means that an exception object now contains all the information pertaining to an exception, and there are fewer reasons to use <a href="../library/sys.html#sys.exc_info" class="reference internal" title="sys.exc_info"><span class="pre"><code class="sourceCode python">sys.exc_info()</code></span></a> (though the latter is not removed).

- A few exception messages are improved when Windows fails to load an extension module. For example, <span class="pre">`error`</span>` `<span class="pre">`code`</span>` `<span class="pre">`193`</span> is now <span class="pre">`%1`</span>` `<span class="pre">`is`</span>` `<span class="pre">`not`</span>` `<span class="pre">`a`</span>` `<span class="pre">`valid`</span>` `<span class="pre">`Win32`</span>` `<span class="pre">`application`</span>. Strings now deal with non-English locales.

</div>

<div id="miscellaneous-other-changes" class="section">

## Miscellaneous Other Changes<a href="#miscellaneous-other-changes" class="headerlink" title="Link to this heading">¶</a>

<div id="operators-and-special-methods" class="section">

### Operators And Special Methods<a href="#operators-and-special-methods" class="headerlink" title="Link to this heading">¶</a>

- <span class="pre">`!=`</span> now returns the opposite of <span class="pre">`==`</span>, unless <span class="pre">`==`</span> returns <a href="../library/constants.html#NotImplemented" class="reference internal" title="NotImplemented"><span class="pre"><code class="sourceCode python"><span class="va">NotImplemented</span></code></span></a>.

- The concept of “unbound methods” has been removed from the language. When referencing a method as a class attribute, you now get a plain function object.

- <span class="pre">`__getslice__()`</span>, <span class="pre">`__setslice__()`</span> and <span class="pre">`__delslice__()`</span> were killed. The syntax <span class="pre">`a[i:j]`</span> now translates to <span class="pre">`a.__getitem__(slice(i,`</span>` `<span class="pre">`j))`</span> (or <a href="../reference/datamodel.html#object.__setitem__" class="reference internal" title="object.__setitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__setitem__</span>()</code></span></a> or <a href="../reference/datamodel.html#object.__delitem__" class="reference internal" title="object.__delitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__delitem__</span>()</code></span></a>, when used as an assignment or deletion target, respectively).

- <span id="index-30" class="target"></span><a href="https://peps.python.org/pep-3114/" class="pep reference external"><strong>PEP 3114</strong></a>: the standard <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a> method has been renamed to <a href="../library/stdtypes.html#iterator.__next__" class="reference internal" title="iterator.__next__"><span class="pre"><code class="sourceCode python"><span class="fu">__next__</span>()</code></span></a>.

- The <span class="pre">`__oct__()`</span> and <span class="pre">`__hex__()`</span> special methods are removed – <a href="../library/functions.html#oct" class="reference internal" title="oct"><span class="pre"><code class="sourceCode python"><span class="bu">oct</span>()</code></span></a> and <a href="../library/functions.html#hex" class="reference internal" title="hex"><span class="pre"><code class="sourceCode python"><span class="bu">hex</span>()</code></span></a> use <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> now to convert the argument to an integer.

- Removed support for <span class="pre">`__members__`</span> and <span class="pre">`__methods__`</span>.

- The function attributes named <span class="pre">`func_X`</span> have been renamed to use the <span class="pre">`__X__`</span> form, freeing up these names in the function attribute namespace for user-defined attributes. To wit, <span class="pre">`func_closure`</span>, <span class="pre">`func_code`</span>, <span class="pre">`func_defaults`</span>, <span class="pre">`func_dict`</span>, <span class="pre">`func_doc`</span>, <span class="pre">`func_globals`</span>, <span class="pre">`func_name`</span> were renamed to <a href="../reference/datamodel.html#function.__closure__" class="reference internal" title="function.__closure__"><span class="pre"><code class="sourceCode python">__closure__</code></span></a>, <a href="../reference/datamodel.html#function.__code__" class="reference internal" title="function.__code__"><span class="pre"><code class="sourceCode python">__code__</code></span></a>, <a href="../reference/datamodel.html#function.__defaults__" class="reference internal" title="function.__defaults__"><span class="pre"><code class="sourceCode python">__defaults__</code></span></a>, <a href="../reference/datamodel.html#function.__dict__" class="reference internal" title="function.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a>, <a href="../reference/datamodel.html#function.__doc__" class="reference internal" title="function.__doc__"><span class="pre"><code class="sourceCode python">__doc__</code></span></a>, <a href="../reference/datamodel.html#function.__globals__" class="reference internal" title="function.__globals__"><span class="pre"><code class="sourceCode python">__globals__</code></span></a>, <a href="../reference/datamodel.html#function.__name__" class="reference internal" title="function.__name__"><span class="pre"><code class="sourceCode python"><span class="va">__name__</span></code></span></a>, respectively.

- <span class="pre">`__nonzero__()`</span> is now <a href="../reference/datamodel.html#object.__bool__" class="reference internal" title="object.__bool__"><span class="pre"><code class="sourceCode python"><span class="fu">__bool__</span>()</code></span></a>.

</div>

<div id="builtins" class="section">

### Builtins<a href="#builtins" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-31" class="target"></span><a href="https://peps.python.org/pep-3135/" class="pep reference external"><strong>PEP 3135</strong></a>: New <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a>. You can now invoke <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a> without arguments and (assuming this is in a regular instance method defined inside a <a href="../reference/compound_stmts.html#class" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">class</code></span></a> statement) the right class and instance will automatically be chosen. With arguments, the behavior of <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a> is unchanged.

- <span id="index-32" class="target"></span><a href="https://peps.python.org/pep-3111/" class="pep reference external"><strong>PEP 3111</strong></a>: <span class="pre">`raw_input()`</span> was renamed to <a href="../library/functions.html#input" class="reference internal" title="input"><span class="pre"><code class="sourceCode python"><span class="bu">input</span>()</code></span></a>. That is, the new <a href="../library/functions.html#input" class="reference internal" title="input"><span class="pre"><code class="sourceCode python"><span class="bu">input</span>()</code></span></a> function reads a line from <a href="../library/sys.html#sys.stdin" class="reference internal" title="sys.stdin"><span class="pre"><code class="sourceCode python">sys.stdin</code></span></a> and returns it with the trailing newline stripped. It raises <a href="../library/exceptions.html#EOFError" class="reference internal" title="EOFError"><span class="pre"><code class="sourceCode python"><span class="pp">EOFError</span></code></span></a> if the input is terminated prematurely. To get the old behavior of <a href="../library/functions.html#input" class="reference internal" title="input"><span class="pre"><code class="sourceCode python"><span class="bu">input</span>()</code></span></a>, use <span class="pre">`eval(input())`</span>.

- A new built-in function <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a> was added to call the <a href="../library/stdtypes.html#iterator.__next__" class="reference internal" title="iterator.__next__"><span class="pre"><code class="sourceCode python"><span class="fu">__next__</span>()</code></span></a> method on an object.

- The <a href="../library/functions.html#round" class="reference internal" title="round"><span class="pre"><code class="sourceCode python"><span class="bu">round</span>()</code></span></a> function rounding strategy and return type have changed. Exact halfway cases are now rounded to the nearest even result instead of away from zero. (For example, <span class="pre">`round(2.5)`</span> now returns <span class="pre">`2`</span> rather than <span class="pre">`3`</span>.) <span class="pre">`round(x[,`</span>` `<span class="pre">`n])`</span> now delegates to <span class="pre">`x.__round__([n])`</span> instead of always returning a float. It generally returns an integer when called with a single argument and a value of the same type as <span class="pre">`x`</span> when called with two arguments.

- Moved <span class="pre">`intern()`</span> to <a href="../library/sys.html#sys.intern" class="reference internal" title="sys.intern"><span class="pre"><code class="sourceCode python">sys.<span class="bu">intern</span>()</code></span></a>.

- Removed: <span class="pre">`apply()`</span>. Instead of <span class="pre">`apply(f,`</span>` `<span class="pre">`args)`</span> use <span class="pre">`f(*args)`</span>.

- Removed <a href="../library/functions.html#callable" class="reference internal" title="callable"><span class="pre"><code class="sourceCode python"><span class="bu">callable</span>()</code></span></a>. Instead of <span class="pre">`callable(f)`</span> you can use <span class="pre">`isinstance(f,`</span>` `<span class="pre">`collections.Callable)`</span>. The <span class="pre">`operator.isCallable()`</span> function is also gone.

- Removed <span class="pre">`coerce()`</span>. This function no longer serves a purpose now that classic classes are gone.

- Removed <span class="pre">`execfile()`</span>. Instead of <span class="pre">`execfile(fn)`</span> use <span class="pre">`exec(open(fn).read())`</span>.

- Removed the <span class="pre">`file`</span> type. Use <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a>. There are now several different kinds of streams that open can return in the <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module.

- Removed <span class="pre">`reduce()`</span>. Use <a href="../library/functools.html#functools.reduce" class="reference internal" title="functools.reduce"><span class="pre"><code class="sourceCode python">functools.<span class="bu">reduce</span>()</code></span></a> if you really need it; however, 99 percent of the time an explicit <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> loop is more readable.

- Removed <span class="pre">`reload()`</span>. Use <span class="pre">`imp.reload()`</span>.

- Removed. <span class="pre">`dict.has_key()`</span> – use the <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> operator instead.

</div>

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Link to this heading">¶</a>

Due to time constraints, here is a *very* incomplete list of changes to the C API.

- Support for several platforms was dropped, including but not limited to Mac OS 9, BeOS, RISCOS, Irix, and Tru64.

- <span id="index-33" class="target"></span><a href="https://peps.python.org/pep-3118/" class="pep reference external"><strong>PEP 3118</strong></a>: New Buffer API.

- <span id="index-34" class="target"></span><a href="https://peps.python.org/pep-3121/" class="pep reference external"><strong>PEP 3121</strong></a>: Extension Module Initialization & Finalization.

- <span id="index-35" class="target"></span><a href="https://peps.python.org/pep-3123/" class="pep reference external"><strong>PEP 3123</strong></a>: Making <a href="../c-api/structures.html#c.PyObject_HEAD" class="reference internal" title="PyObject_HEAD"><span class="pre"><code class="sourceCode c">PyObject_HEAD</code></span></a> conform to standard C.

- No more C API support for restricted execution.

- <span class="pre">`PyNumber_Coerce()`</span>, <span class="pre">`PyNumber_CoerceEx()`</span>, <span class="pre">`PyMember_Get()`</span>, and <span class="pre">`PyMember_Set()`</span> C APIs are removed.

- New C API <a href="../c-api/import.html#c.PyImport_ImportModuleNoBlock" class="reference internal" title="PyImport_ImportModuleNoBlock"><span class="pre"><code class="sourceCode c">PyImport_ImportModuleNoBlock<span class="op">()</span></code></span></a>, works like <a href="../c-api/import.html#c.PyImport_ImportModule" class="reference internal" title="PyImport_ImportModule"><span class="pre"><code class="sourceCode c">PyImport_ImportModule<span class="op">()</span></code></span></a> but won’t block on the import lock (returning an error instead).

- Renamed the boolean conversion C-level slot and method: <span class="pre">`nb_nonzero`</span> is now <span class="pre">`nb_bool`</span>.

- Removed <span class="pre">`METH_OLDARGS`</span> and <span class="pre">`WITH_CYCLE_GC`</span> from the C API.

</div>

<div id="performance" class="section">

## Performance<a href="#performance" class="headerlink" title="Link to this heading">¶</a>

The net result of the 3.0 generalizations is that Python 3.0 runs the pystone benchmark around 10% slower than Python 2.5. Most likely the biggest cause is the removal of special-casing for small integers. There’s room for improvement, but it will happen after 3.0 is released!

</div>

<div id="porting-to-python-3-0" class="section">

## Porting To Python 3.0<a href="#porting-to-python-3-0" class="headerlink" title="Link to this heading">¶</a>

For porting existing Python 2.5 or 2.6 source code to Python 3.0, the best strategy is the following:

0.  (Prerequisite:) Start with excellent test coverage.

1.  Port to Python 2.6. This should be no more work than the average port from Python 2.x to Python 2.(x+1). Make sure all your tests pass.

2.  (Still using 2.6:) Turn on the <span class="pre">`-3`</span> command line switch. This enables warnings about features that will be removed (or change) in 3.0. Run your test suite again, and fix code that you get warnings about until there are no warnings left, and all your tests still pass.

3.  Run the <span class="pre">`2to3`</span> source-to-source translator over your source code tree. Run the result of the translation under Python 3.0. Manually fix up any remaining issues, fixing problems until all tests pass again.

It is not recommended to try to write source code that runs unchanged under both Python 2.6 and 3.0; you’d have to use a very contorted coding style, e.g. avoiding <span class="pre">`print`</span> statements, metaclasses, and much more. If you are maintaining a library that needs to support both Python 2.6 and Python 3.0, the best approach is to modify step 3 above by editing the 2.6 version of the source code and running the <span class="pre">`2to3`</span> translator again, rather than editing the 3.0 version of the source code.

For porting C extensions to Python 3.0, please see <a href="../howto/cporting.html#cporting-howto" class="reference internal"><span class="std std-ref">Porting Extension Modules to Python 3</span></a>.

</div>

</div>

<div class="clearer">

</div>

</div>
