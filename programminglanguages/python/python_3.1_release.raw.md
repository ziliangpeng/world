<div class="body" role="main">

<div id="what-s-new-in-python-3-1" class="section">

# What’s New In Python 3.1<a href="#what-s-new-in-python-3-1" class="headerlink" title="Link to this heading">¶</a>

Author<span class="colon">:</span>  
Raymond Hettinger

This article explains the new features in Python 3.1, compared to 3.0. Python 3.1 was released on June 27, 2009.

<div id="pep-372-ordered-dictionaries" class="section">

## PEP 372: Ordered Dictionaries<a href="#pep-372-ordered-dictionaries" class="headerlink" title="Link to this heading">¶</a>

Regular Python dictionaries iterate over key/value pairs in arbitrary order. Over the years, a number of authors have written alternative implementations that remember the order that the keys were originally inserted. Based on the experiences from those implementations, a new <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">collections.OrderedDict</code></span></a> class has been introduced.

The OrderedDict API is substantially the same as regular dictionaries but will iterate over keys and values in a guaranteed order depending on when a key was first inserted. If a new entry overwrites an existing entry, the original insertion position is left unchanged. Deleting an entry and reinserting it will move it to the end.

The standard library now supports use of ordered dictionaries in several modules. The <a href="../library/configparser.html#module-configparser" class="reference internal" title="configparser: Configuration file parser."><span class="pre"><code class="sourceCode python">configparser</code></span></a> module uses them by default. This lets configuration files be read, modified, and then written back in their original order. The *\_asdict()* method for <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">collections.namedtuple()</code></span></a> now returns an ordered dictionary with the values appearing in the same order as the underlying tuple indices. The <a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> module is being built-out with an *object_pairs_hook* to allow OrderedDicts to be built by the decoder. Support was also added for third-party tools like <a href="https://pyyaml.org/" class="reference external">PyYAML</a>.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0372/" class="pep reference external"><strong>PEP 372</strong></a> - Ordered Dictionaries  
PEP written by Armin Ronacher and Raymond Hettinger. Implementation written by Raymond Hettinger.

</div>

Since an ordered dictionary remembers its insertion order, it can be used in conjunction with sorting to make a sorted dictionary:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> # regular unsorted dictionary
    >>> d = {'banana': 3, 'apple':4, 'pear': 1, 'orange': 2}

    >>> # dictionary sorted by key
    >>> OrderedDict(sorted(d.items(), key=lambda t: t[0]))
    OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])

    >>> # dictionary sorted by value
    >>> OrderedDict(sorted(d.items(), key=lambda t: t[1]))
    OrderedDict([('pear', 1), ('orange', 2), ('banana', 3), ('apple', 4)])

    >>> # dictionary sorted by length of the key string
    >>> OrderedDict(sorted(d.items(), key=lambda t: len(t[0])))
    OrderedDict([('pear', 1), ('apple', 4), ('orange', 2), ('banana', 3)])

</div>

</div>

The new sorted dictionaries maintain their sort order when entries are deleted. But when new keys are added, the keys are appended to the end and the sort is not maintained.

</div>

<div id="pep-378-format-specifier-for-thousands-separator" class="section">

## PEP 378: Format Specifier for Thousands Separator<a href="#pep-378-format-specifier-for-thousands-separator" class="headerlink" title="Link to this heading">¶</a>

The built-in <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> function and the <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a> method use a mini-language that now includes a simple, non-locale aware way to format a number with a thousands separator. That provides a way to humanize a program’s output, improving its professional appearance and readability:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> format(1234567, ',d')
    '1,234,567'
    >>> format(1234567.89, ',.2f')
    '1,234,567.89'
    >>> format(12345.6 + 8901234.12j, ',f')
    '12,345.600000+8,901,234.120000j'
    >>> format(Decimal('1234567.89'), ',f')
    '1,234,567.89'

</div>

</div>

The supported types are <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>, <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a>, <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a> and <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">decimal.Decimal</code></span></a>.

Discussions are underway about how to specify alternative separators like dots, spaces, apostrophes, or underscores. Locale-aware applications should use the existing *n* format specifier which already has some support for thousands separators.

<div class="admonition seealso">

See also

<span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0378/" class="pep reference external"><strong>PEP 378</strong></a> - Format Specifier for Thousands Separator  
PEP written by Raymond Hettinger and implemented by Eric Smith and Mark Dickinson.

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

Some smaller changes made to the core Python language are:

- Directories and zip archives containing a <span class="pre">`__main__.py`</span> file can now be executed directly by passing their name to the interpreter. The directory/zipfile is automatically inserted as the first entry in sys.path. (Suggestion and initial patch by Andy Chu; revised patch by Phillip J. Eby and Nick Coghlan; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1739468" class="reference external">bpo-1739468</a>.)

- The <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a> type gained a <span class="pre">`bit_length`</span> method that returns the number of bits necessary to represent its argument in binary:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> n = 37
      >>> bin(37)
      '0b100101'
      >>> n.bit_length()
      6
      >>> n = 2**123-1
      >>> n.bit_length()
      123
      >>> (n+1).bit_length()
      124

  </div>

  </div>

  (Contributed by Fredrik Johansson, Victor Stinner, Raymond Hettinger, and Mark Dickinson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3439" class="reference external">bpo-3439</a>.)

- The fields in <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> strings can now be automatically numbered:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> 'Sir {} of {}'.format('Gallahad', 'Camelot')
      'Sir Gallahad of Camelot'

  </div>

  </div>

  Formerly, the string would have required numbered fields such as: <span class="pre">`'Sir`</span>` `<span class="pre">`{0}`</span>` `<span class="pre">`of`</span>` `<span class="pre">`{1}'`</span>.

  (Contributed by Eric Smith; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5237" class="reference external">bpo-5237</a>.)

- The <span class="pre">`string.maketrans()`</span> function is deprecated and is replaced by new static methods, <a href="../library/stdtypes.html#bytes.maketrans" class="reference internal" title="bytes.maketrans"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span>.maketrans()</code></span></a> and <a href="../library/stdtypes.html#bytearray.maketrans" class="reference internal" title="bytearray.maketrans"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span>.maketrans()</code></span></a>. This change solves the confusion around which types were supported by the <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module. Now, <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>, and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> each have their own **maketrans** and **translate** methods with intermediate translation tables of the appropriate type.

  (Contributed by Georg Brandl; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5675" class="reference external">bpo-5675</a>.)

- The syntax of the <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement now allows multiple context managers in a single statement:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> with open('mylog.txt') as infile, open('a.out', 'w') as outfile:
      ...     for line in infile:
      ...         if '<critical>' in line:
      ...             outfile.write(line)

  </div>

  </div>

  With the new syntax, the <span class="pre">`contextlib.nested()`</span> function is no longer needed and is now deprecated.

  (Contributed by Georg Brandl and Mattias Brändström; <a href="https://codereview.appspot.com/53094" class="reference external">appspot issue 53094</a>.)

- <span class="pre">`round(x,`</span>` `<span class="pre">`n)`</span> now returns an integer if *x* is an integer. Previously it returned a float:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> round(1123, -2)
      1100

  </div>

  </div>

  (Contributed by Mark Dickinson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4707" class="reference external">bpo-4707</a>.)

- Python now uses David Gay’s algorithm for finding the shortest floating-point representation that doesn’t change its value. This should help mitigate some of the confusion surrounding binary floating-point numbers.

  The significance is easily seen with a number like <span class="pre">`1.1`</span> which does not have an exact equivalent in binary floating point. Since there is no exact equivalent, an expression like <span class="pre">`float('1.1')`</span> evaluates to the nearest representable value which is <span class="pre">`0x1.199999999999ap+0`</span> in hex or <span class="pre">`1.100000000000000088817841970012523233890533447265625`</span> in decimal. That nearest value was and still is used in subsequent floating-point calculations.

  What is new is how the number gets displayed. Formerly, Python used a simple approach. The value of <span class="pre">`repr(1.1)`</span> was computed as <span class="pre">`format(1.1,`</span>` `<span class="pre">`'.17g')`</span> which evaluated to <span class="pre">`'1.1000000000000001'`</span>. The advantage of using 17 digits was that it relied on IEEE-754 guarantees to assure that <span class="pre">`eval(repr(1.1))`</span> would round-trip exactly to its original value. The disadvantage is that many people found the output to be confusing (mistaking intrinsic limitations of binary floating-point representation as being a problem with Python itself).

  The new algorithm for <span class="pre">`repr(1.1)`</span> is smarter and returns <span class="pre">`'1.1'`</span>. Effectively, it searches all equivalent string representations (ones that get stored with the same underlying float value) and returns the shortest representation.

  The new algorithm tends to emit cleaner representations when possible, but it does not change the underlying values. So, it is still the case that <span class="pre">`1.1`</span>` `<span class="pre">`+`</span>` `<span class="pre">`2.2`</span>` `<span class="pre">`!=`</span>` `<span class="pre">`3.3`</span> even though the representations may suggest otherwise.

  The new algorithm depends on certain features in the underlying floating-point implementation. If the required features are not found, the old algorithm will continue to be used. Also, the text pickle protocols assure cross-platform portability by using the old algorithm.

  (Contributed by Eric Smith and Mark Dickinson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1580" class="reference external">bpo-1580</a>)

</div>

<div id="new-improved-and-deprecated-modules" class="section">

## New, Improved, and Deprecated Modules<a href="#new-improved-and-deprecated-modules" class="headerlink" title="Link to this heading">¶</a>

- Added a <a href="../library/collections.html#collections.Counter" class="reference internal" title="collections.Counter"><span class="pre"><code class="sourceCode python">collections.Counter</code></span></a> class to support convenient counting of unique items in a sequence or iterable:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])
      Counter({'blue': 3, 'red': 2, 'green': 1})

  </div>

  </div>

  (Contributed by Raymond Hettinger; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1696199" class="reference external">bpo-1696199</a>.)

- Added a new module, <a href="../library/tkinter.ttk.html#module-tkinter.ttk" class="reference internal" title="tkinter.ttk: Tk themed widget set"><span class="pre"><code class="sourceCode python">tkinter.ttk</code></span></a> for access to the Tk themed widget set. The basic idea of ttk is to separate, to the extent possible, the code implementing a widget’s behavior from the code implementing its appearance.

  (Contributed by Guilherme Polo; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2983" class="reference external">bpo-2983</a>.)

- The <a href="../library/gzip.html#gzip.GzipFile" class="reference internal" title="gzip.GzipFile"><span class="pre"><code class="sourceCode python">gzip.GzipFile</code></span></a> and <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">bz2.BZ2File</code></span></a> classes now support the context management protocol:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> # Automatically close file after writing
      >>> with gzip.GzipFile(filename, "wb") as f:
      ...     f.write(b"xxx")

  </div>

  </div>

  (Contributed by Antoine Pitrou.)

- The <a href="../library/decimal.html#module-decimal" class="reference internal" title="decimal: Implementation of the General Decimal Arithmetic Specification."><span class="pre"><code class="sourceCode python">decimal</code></span></a> module now supports methods for creating a decimal object from a binary <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a>. The conversion is exact but can sometimes be surprising:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> Decimal.from_float(1.1)
      Decimal('1.100000000000000088817841970012523233890533447265625')

  </div>

  </div>

  The long decimal result shows the actual binary fraction being stored for *1.1*. The fraction has many digits because *1.1* cannot be exactly represented in binary.

  (Contributed by Raymond Hettinger and Mark Dickinson.)

- The <a href="../library/itertools.html#module-itertools" class="reference internal" title="itertools: Functions creating iterators for efficient looping."><span class="pre"><code class="sourceCode python">itertools</code></span></a> module grew two new functions. The <a href="../library/itertools.html#itertools.combinations_with_replacement" class="reference internal" title="itertools.combinations_with_replacement"><span class="pre"><code class="sourceCode python">itertools.combinations_with_replacement()</code></span></a> function is one of four for generating combinatorics including permutations and Cartesian products. The <a href="../library/itertools.html#itertools.compress" class="reference internal" title="itertools.compress"><span class="pre"><code class="sourceCode python">itertools.compress()</code></span></a> function mimics its namesake from APL. Also, the existing <a href="../library/itertools.html#itertools.count" class="reference internal" title="itertools.count"><span class="pre"><code class="sourceCode python">itertools.count()</code></span></a> function now has an optional *step* argument and can accept any type of counting sequence including <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">fractions.Fraction</code></span></a> and <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">decimal.Decimal</code></span></a>:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> [p+q for p,q in combinations_with_replacement('LOVE', 2)]
      ['LL', 'LO', 'LV', 'LE', 'OO', 'OV', 'OE', 'VV', 'VE', 'EE']

      >>> list(compress(data=range(10), selectors=[0,0,1,1,0,1,0,1,0,0]))
      [2, 3, 5, 7]

      >>> c = count(start=Fraction(1,2), step=Fraction(1,6))
      >>> [next(c), next(c), next(c), next(c)]
      [Fraction(1, 2), Fraction(2, 3), Fraction(5, 6), Fraction(1, 1)]

  </div>

  </div>

  (Contributed by Raymond Hettinger.)

- <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">collections.namedtuple()</code></span></a> now supports a keyword argument *rename* which lets invalid fieldnames be automatically converted to positional names in the form \_0, \_1, etc. This is useful when the field names are being created by an external source such as a CSV header, SQL field list, or user input:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> query = input()
      SELECT region, dept, count(*) FROM main GROUPBY region, dept

      >>> cursor.execute(query)
      >>> query_fields = [desc[0] for desc in cursor.description]
      >>> UserQuery = namedtuple('UserQuery', query_fields, rename=True)
      >>> pprint.pprint([UserQuery(*row) for row in cursor])
      [UserQuery(region='South', dept='Shipping', _2=185),
       UserQuery(region='North', dept='Accounting', _2=37),
       UserQuery(region='West', dept='Sales', _2=419)]

  </div>

  </div>

  (Contributed by Raymond Hettinger; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1818" class="reference external">bpo-1818</a>.)

- The <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">re.sub()</code></span></a>, <a href="../library/re.html#re.subn" class="reference internal" title="re.subn"><span class="pre"><code class="sourceCode python">re.subn()</code></span></a> and <a href="../library/re.html#re.split" class="reference internal" title="re.split"><span class="pre"><code class="sourceCode python">re.split()</code></span></a> functions now accept a flags parameter.

  (Contributed by Gregory Smith.)

- The <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> module now implements a simple <a href="../library/logging.handlers.html#logging.NullHandler" class="reference internal" title="logging.NullHandler"><span class="pre"><code class="sourceCode python">logging.NullHandler</code></span></a> class for applications that are not using logging but are calling library code that does. Setting-up a null handler will suppress spurious warnings such as “No handlers could be found for logger foo”:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> h = logging.NullHandler()
      >>> logging.getLogger("foo").addHandler(h)

  </div>

  </div>

  (Contributed by Vinay Sajip; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4384" class="reference external">bpo-4384</a>).

- The <a href="../library/runpy.html#module-runpy" class="reference internal" title="runpy: Locate and run Python modules without importing them first."><span class="pre"><code class="sourceCode python">runpy</code></span></a> module which supports the <span class="pre">`-m`</span> command line switch now supports the execution of packages by looking for and executing a <span class="pre">`__main__`</span> submodule when a package name is supplied.

  (Contributed by Andi Vajda; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4195" class="reference external">bpo-4195</a>.)

- The <a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a> module can now access and display source code loaded via <a href="../library/zipimport.html#module-zipimport" class="reference internal" title="zipimport: Support for importing Python modules from ZIP archives."><span class="pre"><code class="sourceCode python">zipimport</code></span></a> (or any other conformant <span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0302/" class="pep reference external"><strong>PEP 302</strong></a> loader).

  (Contributed by Alexander Belopolsky; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4201" class="reference external">bpo-4201</a>.)

- <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">functools.partial</code></span></a> objects can now be pickled.

> <div>
>
> (Suggested by Antoine Pitrou and Jesse Noller. Implemented by Jack Diederich; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5228" class="reference external">bpo-5228</a>.)
>
> </div>

- Add <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> help topics for symbols so that <span class="pre">`help('@')`</span> works as expected in the interactive environment.

  (Contributed by David Laban; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4739" class="reference external">bpo-4739</a>.)

- The <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> module now supports skipping individual tests or classes of tests. And it supports marking a test as an expected failure, a test that is known to be broken, but shouldn’t be counted as a failure on a TestResult:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      class TestGizmo(unittest.TestCase):

          @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
          def test_gizmo_on_windows(self):
              ...

          @unittest.expectedFailure
          def test_gimzo_without_required_library(self):
              ...

  </div>

  </div>

  Also, tests for exceptions have been builtout to work with context managers using the <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      def test_division_by_zero(self):
          with self.assertRaises(ZeroDivisionError):
              x / 0

  </div>

  </div>

  In addition, several new assertion methods were added including <a href="../library/unittest.html#unittest.TestCase.assertSetEqual" class="reference internal" title="unittest.TestCase.assertSetEqual"><span class="pre"><code class="sourceCode python">assertSetEqual()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertDictEqual" class="reference internal" title="unittest.TestCase.assertDictEqual"><span class="pre"><code class="sourceCode python">assertDictEqual()</code></span></a>, <span class="pre">`assertDictContainsSubset()`</span>, <a href="../library/unittest.html#unittest.TestCase.assertListEqual" class="reference internal" title="unittest.TestCase.assertListEqual"><span class="pre"><code class="sourceCode python">assertListEqual()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertTupleEqual" class="reference internal" title="unittest.TestCase.assertTupleEqual"><span class="pre"><code class="sourceCode python">assertTupleEqual()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertSequenceEqual" class="reference internal" title="unittest.TestCase.assertSequenceEqual"><span class="pre"><code class="sourceCode python">assertSequenceEqual()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertRaisesRegex" class="reference internal" title="unittest.TestCase.assertRaisesRegex"><span class="pre"><code class="sourceCode python">assertRaisesRegexp()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertIsNone" class="reference internal" title="unittest.TestCase.assertIsNone"><span class="pre"><code class="sourceCode python">assertIsNone()</code></span></a>, and <a href="../library/unittest.html#unittest.TestCase.assertIsNotNone" class="reference internal" title="unittest.TestCase.assertIsNotNone"><span class="pre"><code class="sourceCode python">assertIsNotNone()</code></span></a>.

  (Contributed by Benjamin Peterson and Antoine Pitrou.)

- The <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module has three new constants for the <a href="../library/io.html#io.IOBase.seek" class="reference internal" title="io.IOBase.seek"><span class="pre"><code class="sourceCode python">seek()</code></span></a> method: <a href="../library/os.html#os.SEEK_SET" class="reference internal" title="os.SEEK_SET"><span class="pre"><code class="sourceCode python">SEEK_SET</code></span></a>, <a href="../library/os.html#os.SEEK_CUR" class="reference internal" title="os.SEEK_CUR"><span class="pre"><code class="sourceCode python">SEEK_CUR</code></span></a>, and <a href="../library/os.html#os.SEEK_END" class="reference internal" title="os.SEEK_END"><span class="pre"><code class="sourceCode python">SEEK_END</code></span></a>.

- The <a href="../library/sys.html#sys.version_info" class="reference internal" title="sys.version_info"><span class="pre"><code class="sourceCode python">sys.version_info</code></span></a> tuple is now a named tuple:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> sys.version_info
      sys.version_info(major=3, minor=1, micro=0, releaselevel='alpha', serial=2)

  </div>

  </div>

  (Contributed by Ross Light; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4285" class="reference external">bpo-4285</a>.)

- The <span class="pre">`nntplib`</span> and <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a> modules now support IPv6.

  (Contributed by Derek Morr; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1655" class="reference external">bpo-1655</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1664" class="reference external">bpo-1664</a>.)

- The <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> module has been adapted for better interoperability with Python 2.x when used with protocol 2 or lower. The reorganization of the standard library changed the formal reference for many objects. For example, <span class="pre">`__builtin__.set`</span> in Python 2 is called <span class="pre">`builtins.set`</span> in Python 3. This change confounded efforts to share data between different versions of Python. But now when protocol 2 or lower is selected, the pickler will automatically use the old Python 2 names for both loading and dumping. This remapping is turned-on by default but can be disabled with the *fix_imports* option:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> s = {1, 2, 3}
      >>> pickle.dumps(s, protocol=0)
      b'c__builtin__\nset\np0\n((lp1\nL1L\naL2L\naL3L\natp2\nRp3\n.'
      >>> pickle.dumps(s, protocol=0, fix_imports=False)
      b'cbuiltins\nset\np0\n((lp1\nL1L\naL2L\naL3L\natp2\nRp3\n.'

  </div>

  </div>

  An unfortunate but unavoidable side-effect of this change is that protocol 2 pickles produced by Python 3.1 won’t be readable with Python 3.0. The latest pickle protocol, protocol 3, should be used when migrating data between Python 3.x implementations, as it doesn’t attempt to remain compatible with Python 2.x.

  (Contributed by Alexandre Vassalotti and Antoine Pitrou, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=6137" class="reference external">bpo-6137</a>.)

- A new module, <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> was added. It provides a complete, portable, pure Python reference implementation of the <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statement and its counterpart, the <a href="../library/functions.html#import__" class="reference internal" title="__import__"><span class="pre"><code class="sourceCode python"><span class="bu">__import__</span>()</code></span></a> function. It represents a substantial step forward in documenting and defining the actions that take place during imports.

  (Contributed by Brett Cannon.)

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

Major performance enhancements have been added:

- The new I/O library (as defined in <span id="index-3" class="target"></span><a href="https://peps.python.org/pep-3116/" class="pep reference external"><strong>PEP 3116</strong></a>) was mostly written in Python and quickly proved to be a problematic bottleneck in Python 3.0. In Python 3.1, the I/O library has been entirely rewritten in C and is 2 to 20 times faster depending on the task at hand. The pure Python version is still available for experimentation purposes through the <span class="pre">`_pyio`</span> module.

  (Contributed by Amaury Forgeot d’Arc and Antoine Pitrou.)

- Added a heuristic so that tuples and dicts containing only untrackable objects are not tracked by the garbage collector. This can reduce the size of collections and therefore the garbage collection overhead on long-running programs, depending on their particular use of datatypes.

  (Contributed by Antoine Pitrou, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4688" class="reference external">bpo-4688</a>.)

- Enabling a configure option named <span class="pre">`--with-computed-gotos`</span> on compilers that support it (notably: gcc, SunPro, icc), the bytecode evaluation loop is compiled with a new dispatch mechanism which gives speedups of up to 20%, depending on the system, the compiler, and the benchmark.

  (Contributed by Antoine Pitrou along with a number of other participants, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4753" class="reference external">bpo-4753</a>).

- The decoding of UTF-8, UTF-16 and LATIN-1 is now two to four times faster.

  (Contributed by Antoine Pitrou and Amaury Forgeot d’Arc, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4868" class="reference external">bpo-4868</a>.)

- The <a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> module now has a C extension to substantially improve its performance. In addition, the API was modified so that json works only with <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, not with <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>. That change makes the module closely match the <a href="https://json.org/" class="reference external">JSON specification</a> which is defined in terms of Unicode.

  (Contributed by Bob Ippolito and converted to Py3.1 by Antoine Pitrou and Benjamin Peterson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4136" class="reference external">bpo-4136</a>.)

- Unpickling now interns the attribute names of pickled objects. This saves memory and allows pickles to be smaller.

  (Contributed by Jake McGuire and Antoine Pitrou; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5084" class="reference external">bpo-5084</a>.)

</div>

<div id="idle" class="section">

## IDLE<a href="#idle" class="headerlink" title="Link to this heading">¶</a>

- IDLE’s format menu now provides an option to strip trailing whitespace from a source file.

  (Contributed by Roger D. Serwy; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5150" class="reference external">bpo-5150</a>.)

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Link to this heading">¶</a>

Changes to Python’s build process and to the C API include:

- Integers are now stored internally either in base <span class="pre">`2**15`</span> or in base <span class="pre">`2**30`</span>, the base being determined at build time. Previously, they were always stored in base <span class="pre">`2**15`</span>. Using base <span class="pre">`2**30`</span> gives significant performance improvements on 64-bit machines, but benchmark results on 32-bit machines have been mixed. Therefore, the default is to use base <span class="pre">`2**30`</span> on 64-bit machines and base <span class="pre">`2**15`</span> on 32-bit machines; on Unix, there’s a new configure option <span class="pre">`--enable-big-digits`</span> that can be used to override this default.

  Apart from the performance improvements this change should be invisible to end users, with one exception: for testing and debugging purposes there’s a new <a href="../library/sys.html#sys.int_info" class="reference internal" title="sys.int_info"><span class="pre"><code class="sourceCode python">sys.int_info</code></span></a> that provides information about the internal format, giving the number of bits per digit and the size in bytes of the C type used to store each digit:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      >>> import sys
      >>> sys.int_info
      sys.int_info(bits_per_digit=30, sizeof_digit=4)

  </div>

  </div>

  (Contributed by Mark Dickinson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4258" class="reference external">bpo-4258</a>.)

- The <a href="../c-api/long.html#c.PyLong_AsUnsignedLongLong" class="reference internal" title="PyLong_AsUnsignedLongLong"><span class="pre"><code class="sourceCode c">PyLong_AsUnsignedLongLong<span class="op">()</span></code></span></a> function now handles a negative *pylong* by raising <a href="../library/exceptions.html#OverflowError" class="reference internal" title="OverflowError"><span class="pre"><code class="sourceCode python"><span class="pp">OverflowError</span></code></span></a> instead of <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>.

  (Contributed by Mark Dickinson and Lisandro Dalcrin; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5175" class="reference external">bpo-5175</a>.)

- Deprecated <span class="pre">`PyNumber_Int()`</span>. Use <a href="../c-api/number.html#c.PyNumber_Long" class="reference internal" title="PyNumber_Long"><span class="pre"><code class="sourceCode c">PyNumber_Long<span class="op">()</span></code></span></a> instead.

  (Contributed by Mark Dickinson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4910" class="reference external">bpo-4910</a>.)

- Added a new <a href="../c-api/conversion.html#c.PyOS_string_to_double" class="reference internal" title="PyOS_string_to_double"><span class="pre"><code class="sourceCode c">PyOS_string_to_double<span class="op">()</span></code></span></a> function to replace the deprecated functions <span class="pre">`PyOS_ascii_strtod()`</span> and <span class="pre">`PyOS_ascii_atof()`</span>.

  (Contributed by Mark Dickinson; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5914" class="reference external">bpo-5914</a>.)

- Added <a href="../c-api/capsule.html#c.PyCapsule" class="reference internal" title="PyCapsule"><span class="pre"><code class="sourceCode c">PyCapsule</code></span></a> as a replacement for the <span class="pre">`PyCObject`</span> API. The principal difference is that the new type has a well defined interface for passing typing safety information and a less complicated signature for calling a destructor. The old type had a problematic API and is now deprecated.

  (Contributed by Larry Hastings; <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5630" class="reference external">bpo-5630</a>.)

</div>

<div id="porting-to-python-3-1" class="section">

## Porting to Python 3.1<a href="#porting-to-python-3-1" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code:

- The new floating-point string representations can break existing doctests. For example:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      def e():
          '''Compute the base of natural logarithms.

          >>> e()
          2.7182818284590451

          '''
          return sum(1/math.factorial(x) for x in reversed(range(30)))

      doctest.testmod()

      **********************************************************************
      Failed example:
          e()
      Expected:
          2.7182818284590451
      Got:
          2.718281828459045
      **********************************************************************

  </div>

  </div>

- The automatic name remapping in the pickle module for protocol 2 or lower can make Python 3.1 pickles unreadable in Python 3.0. One solution is to use protocol 3. Another solution is to set the *fix_imports* option to <span class="pre">`False`</span>. See the discussion above for more details.

</div>

</div>

<div class="clearer">

</div>

</div>
