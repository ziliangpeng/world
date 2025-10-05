<div class="body" role="main">

<div id="what-s-new-in-python-2-7" class="section">

# What’s New in Python 2.7<a href="#what-s-new-in-python-2-7" class="headerlink" title="Permalink to this headline">¶</a>

Author  
A.M. Kuchling (amk at amk.ca)

This article explains the new features in Python 2.7. Python 2.7 was released on July 3, 2010.

Numeric handling has been improved in many ways, for both floating-point numbers and for the <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> class. There are some useful additions to the standard library, such as a greatly enhanced <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> module, the <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> module for parsing command-line options, convenient <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">OrderedDict</code></span></a> and <a href="../library/collections.html#collections.Counter" class="reference internal" title="collections.Counter"><span class="pre"><code class="sourceCode python">Counter</code></span></a> classes in the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module, and many other improvements.

Python 2.7 is planned to be the last of the 2.x releases, so we worked on making it a good release for the long term. To help with porting to Python 3, several new features from the Python 3.x series have been included in 2.7.

This article doesn’t attempt to provide a complete specification of the new features, but instead provides a convenient overview. For full details, you should refer to the documentation for Python 2.7 at <a href="https://docs.python.org" class="reference external">https://docs.python.org</a>. If you want to understand the rationale for the design and implementation, refer to the PEP for a particular new feature or the issue on <a href="https://bugs.python.org" class="reference external">https://bugs.python.org</a> in which a change was discussed. Whenever possible, “What’s New in Python” links to the bug/patch item for each change.

<div id="the-future-for-python-2-x" class="section">

<span id="whatsnew27-python31"></span>

## The Future for Python 2.x<a href="#the-future-for-python-2-x" class="headerlink" title="Permalink to this headline">¶</a>

Python 2.7 is the last major release in the 2.x series, as the Python maintainers have shifted the focus of their new feature development efforts to the Python 3.x series. This means that while Python 2 continues to receive bug fixes, and to be updated to build correctly on new hardware and versions of supported operated systems, there will be no new full feature releases for the language or standard library.

However, while there is a large common subset between Python 2.7 and Python 3, and many of the changes involved in migrating to that common subset, or directly to Python 3, can be safely automated, some other changes (notably those associated with Unicode handling) may require careful consideration, and preferably robust automated regression test suites, to migrate effectively.

This means that Python 2.7 will remain in place for a long time, providing a stable and supported base platform for production systems that have not yet been ported to Python 3. The full expected lifecycle of the Python 2.7 series is detailed in <span id="index-0" class="target"></span><a href="https://www.python.org/dev/peps/pep-0373" class="pep reference external"><strong>PEP 373</strong></a>.

Some key consequences of the long-term significance of 2.7 are:

- As noted above, the 2.7 release has a much longer period of maintenance when compared to earlier 2.x versions. Python 2.7 is currently expected to remain supported by the core development team (receiving security updates and other bug fixes) until at least 2020 (10 years after its initial release, compared to the more typical support period of 18–24 months).

- As the Python 2.7 standard library ages, making effective use of the Python Package Index (either directly or via a redistributor) becomes more important for Python 2 users. In addition to a wide variety of third party packages for various tasks, the available packages include backports of new modules and features from the Python 3 standard library that are compatible with Python 2, as well as various tools and libraries that can make it easier to migrate to Python 3. The <a href="https://packaging.python.org" class="reference external">Python Packaging User Guide</a> provides guidance on downloading and installing software from the Python Package Index.

- While the preferred approach to enhancing Python 2 is now the publication of new packages on the Python Package Index, this approach doesn’t necessarily work in all cases, especially those related to network security. In exceptional cases that cannot be handled adequately by publishing new or updated packages on PyPI, the Python Enhancement Proposal process may be used to make the case for adding new features directly to the Python 2 standard library. Any such additions, and the maintenance releases where they were added, will be noted in the <a href="#py27-maintenance-enhancements" class="reference internal"><span class="std std-ref">New Features Added to Python 2.7 Maintenance Releases</span></a> section below.

For projects wishing to migrate from Python 2 to Python 3, or for library and framework developers wishing to support users on both Python 2 and Python 3, there are a variety of tools and guides available to help decide on a suitable approach and manage some of the technical details involved. The recommended starting point is the <a href="../howto/pyporting.html#pyporting-howto" class="reference internal"><span class="std std-ref">Porting Python 2 Code to Python 3</span></a> HOWTO guide.

</div>

<div id="changes-to-the-handling-of-deprecation-warnings" class="section">

## Changes to the Handling of Deprecation Warnings<a href="#changes-to-the-handling-of-deprecation-warnings" class="headerlink" title="Permalink to this headline">¶</a>

For Python 2.7, a policy decision was made to silence warnings only of interest to developers by default. <a href="../library/exceptions.html#exceptions.DeprecationWarning" class="reference internal" title="exceptions.DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> and its descendants are now ignored unless otherwise requested, preventing users from seeing warnings triggered by an application. This change was also made in the branch that became Python 3.2. (Discussed on stdlib-sig and carried out in <a href="https://bugs.python.org/issue7319" class="reference external">bpo-7319</a>.)

In previous releases, <a href="../library/exceptions.html#exceptions.DeprecationWarning" class="reference internal" title="exceptions.DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> messages were enabled by default, providing Python developers with a clear indication of where their code may break in a future major version of Python.

However, there are increasingly many users of Python-based applications who are not directly involved in the development of those applications. <a href="../library/exceptions.html#exceptions.DeprecationWarning" class="reference internal" title="exceptions.DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> messages are irrelevant to such users, making them worry about an application that’s actually working correctly and burdening application developers with responding to these concerns.

You can re-enable display of <a href="../library/exceptions.html#exceptions.DeprecationWarning" class="reference internal" title="exceptions.DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> messages by running Python with the <a href="../using/cmdline.html#cmdoption-w" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-Wdefault</code></span></a> (short form: <a href="../using/cmdline.html#cmdoption-w" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-Wd</code></span></a>) switch, or by setting the <span id="index-1" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONWARNINGS" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONWARNINGS</code></span></a> environment variable to <span class="pre">`"default"`</span> (or <span class="pre">`"d"`</span>) before running Python. Python code can also re-enable them by calling <span class="pre">`warnings.simplefilter('default')`</span>.

The <span class="pre">`unittest`</span> module also automatically reenables deprecation warnings when running tests.

</div>

<div id="python-3-1-features" class="section">

## Python 3.1 Features<a href="#python-3-1-features" class="headerlink" title="Permalink to this headline">¶</a>

Much as Python 2.6 incorporated features from Python 3.0, version 2.7 incorporates some of the new features in Python 3.1. The 2.x series continues to provide tools for migrating to the 3.x series.

A partial list of 3.1 features that were backported to 2.7:

- The syntax for set literals (<span class="pre">`{1,2,3}`</span> is a mutable set).

- Dictionary and set comprehensions (<span class="pre">`{i:`</span>` `<span class="pre">`i*2`</span>` `<span class="pre">`for`</span>` `<span class="pre">`i`</span>` `<span class="pre">`in`</span>` `<span class="pre">`range(3)}`</span>).

- Multiple context managers in a single <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement.

- A new version of the <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> library, rewritten in C for performance.

- The ordered-dictionary type described in <a href="#pep-0372" class="reference internal"><span class="std std-ref">PEP 372: Adding an Ordered Dictionary to collections</span></a>.

- The new <span class="pre">`","`</span> format specifier described in <a href="#pep-0378" class="reference internal"><span class="std std-ref">PEP 378: Format Specifier for Thousands Separator</span></a>.

- The <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> object.

- A small subset of the <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: Convenience wrappers for __import__"><span class="pre"><code class="sourceCode python">importlib</code></span></a> module, <a href="#importlib-section" class="reference external">described below</a>.

- The <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> of a float <span class="pre">`x`</span> is shorter in many cases: it’s now based on the shortest decimal string that’s guaranteed to round back to <span class="pre">`x`</span>. As in previous versions of Python, it’s guaranteed that <span class="pre">`float(repr(x))`</span> recovers <span class="pre">`x`</span>.

- Float-to-string and string-to-float conversions are correctly rounded. The <a href="../library/functions.html#round" class="reference internal" title="round"><span class="pre"><code class="sourceCode python"><span class="bu">round</span>()</code></span></a> function is also now correctly rounded.

- The <a href="../c-api/capsule.html#c.PyCapsule" class="reference internal" title="PyCapsule"><span class="pre"><code class="sourceCode c">PyCapsule</code></span></a> type, used to provide a C API for extension modules.

- The <a href="../c-api/long.html#c.PyLong_AsLongAndOverflow" class="reference internal" title="PyLong_AsLongAndOverflow"><span class="pre"><code class="sourceCode c">PyLong_AsLongAndOverflow<span class="op">()</span></code></span></a> C API function.

Other new Python3-mode warnings include:

- <a href="../library/operator.html#operator.isCallable" class="reference internal" title="operator.isCallable"><span class="pre"><code class="sourceCode python">operator.isCallable()</code></span></a> and <a href="../library/operator.html#operator.sequenceIncludes" class="reference internal" title="operator.sequenceIncludes"><span class="pre"><code class="sourceCode python">operator.sequenceIncludes()</code></span></a>, which are not supported in 3.x, now trigger warnings.

- The <a href="../using/cmdline.html#cmdoption-3" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-3</code></span></a> switch now automatically enables the <a href="../using/cmdline.html#cmdoption-q" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-Qwarn</code></span></a> switch that causes warnings about using classic division with integers and long integers.

</div>

<div id="pep-372-adding-an-ordered-dictionary-to-collections" class="section">

<span id="pep-0372"></span>

## PEP 372: Adding an Ordered Dictionary to collections<a href="#pep-372-adding-an-ordered-dictionary-to-collections" class="headerlink" title="Permalink to this headline">¶</a>

Regular Python dictionaries iterate over key/value pairs in arbitrary order. Over the years, a number of authors have written alternative implementations that remember the order that the keys were originally inserted. Based on the experiences from those implementations, 2.7 introduces a new <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">OrderedDict</code></span></a> class in the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module.

The <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">OrderedDict</code></span></a> API provides the same interface as regular dictionaries but iterates over keys and values in a guaranteed order depending on when a key was first inserted:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> from collections import OrderedDict
    >>> d = OrderedDict([('first', 1),
    ...                  ('second', 2),
    ...                  ('third', 3)])
    >>> d.items()
    [('first', 1), ('second', 2), ('third', 3)]

</div>

</div>

If a new entry overwrites an existing entry, the original insertion position is left unchanged:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> d['second'] = 4
    >>> d.items()
    [('first', 1), ('second', 4), ('third', 3)]

</div>

</div>

Deleting an entry and reinserting it will move it to the end:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> del d['second']
    >>> d['second'] = 5
    >>> d.items()
    [('first', 1), ('third', 3), ('second', 5)]

</div>

</div>

The <a href="../library/collections.html#collections.OrderedDict.popitem" class="reference internal" title="collections.OrderedDict.popitem"><span class="pre"><code class="sourceCode python">popitem()</code></span></a> method has an optional *last* argument that defaults to <span class="pre">`True`</span>. If *last* is true, the most recently added key is returned and removed; if it’s false, the oldest key is selected:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> od = OrderedDict([(x,0) for x in range(20)])
    >>> od.popitem()
    (19, 0)
    >>> od.popitem()
    (18, 0)
    >>> od.popitem(last=False)
    (0, 0)
    >>> od.popitem(last=False)
    (1, 0)

</div>

</div>

Comparing two ordered dictionaries checks both the keys and values, and requires that the insertion order was the same:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> od1 = OrderedDict([('first', 1),
    ...                    ('second', 2),
    ...                    ('third', 3)])
    >>> od2 = OrderedDict([('third', 3),
    ...                    ('first', 1),
    ...                    ('second', 2)])
    >>> od1 == od2
    False
    >>> # Move 'third' key to the end
    >>> del od2['third']; od2['third'] = 3
    >>> od1 == od2
    True

</div>

</div>

Comparing an <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">OrderedDict</code></span></a> with a regular dictionary ignores the insertion order and just compares the keys and values.

How does the <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">OrderedDict</code></span></a> work? It maintains a doubly-linked list of keys, appending new keys to the list as they’re inserted. A secondary dictionary maps keys to their corresponding list node, so deletion doesn’t have to traverse the entire linked list and therefore remains O(1).

The standard library now supports use of ordered dictionaries in several modules.

- The <a href="../library/configparser.html#module-ConfigParser" class="reference internal" title="ConfigParser: Configuration file parser."><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a> module uses them by default, meaning that configuration files can now be read, modified, and then written back in their original order.

- The <a href="../library/collections.html#collections.somenamedtuple._asdict" class="reference internal" title="collections.somenamedtuple._asdict"><span class="pre"><code class="sourceCode python">_asdict()</code></span></a> method for <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">collections.namedtuple()</code></span></a> now returns an ordered dictionary with the values appearing in the same order as the underlying tuple indices.

- The <a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> module’s <a href="../library/json.html#json.JSONDecoder" class="reference internal" title="json.JSONDecoder"><span class="pre"><code class="sourceCode python">JSONDecoder</code></span></a> class constructor was extended with an *object_pairs_hook* parameter to allow <span class="pre">`OrderedDict`</span> instances to be built by the decoder. Support was also added for third-party tools like <a href="http://pyyaml.org/" class="reference external">PyYAML</a>.

<div class="admonition seealso">

See also

<span id="index-2" class="target"></span><a href="https://www.python.org/dev/peps/pep-0372" class="pep reference external"><strong>PEP 372</strong></a> - Adding an ordered dictionary to collections  
PEP written by Armin Ronacher and Raymond Hettinger; implemented by Raymond Hettinger.

</div>

</div>

<div id="pep-378-format-specifier-for-thousands-separator" class="section">

<span id="pep-0378"></span>

## PEP 378: Format Specifier for Thousands Separator<a href="#pep-378-format-specifier-for-thousands-separator" class="headerlink" title="Permalink to this headline">¶</a>

To make program output more readable, it can be useful to add separators to large numbers, rendering them as 18,446,744,073,709,551,616 instead of 18446744073709551616.

The fully general solution for doing this is the <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a> module, which can use different separators (“,” in North America, “.” in Europe) and different grouping sizes, but <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a> is complicated to use and unsuitable for multi-threaded applications where different threads are producing output for different locales.

Therefore, a simple comma-grouping mechanism has been added to the mini-language used by the <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a> method. When formatting a floating-point number, simply include a comma between the width and the precision:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> '{:20,.2f}'.format(18446744073709551616.0)
    '18,446,744,073,709,551,616.00'

</div>

</div>

When formatting an integer, include the comma after the width:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> '{:20,d}'.format(18446744073709551616)
    '18,446,744,073,709,551,616'

</div>

</div>

This mechanism is not adaptable at all; commas are always used as the separator and the grouping is always into three-digit groups. The comma-formatting mechanism isn’t as general as the <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a> module, but it’s easier to use.

<div class="admonition seealso">

See also

<span id="index-3" class="target"></span><a href="https://www.python.org/dev/peps/pep-0378" class="pep reference external"><strong>PEP 378</strong></a> - Format Specifier for Thousands Separator  
PEP written by Raymond Hettinger; implemented by Eric Smith.

</div>

</div>

<div id="pep-389-the-argparse-module-for-parsing-command-lines" class="section">

## PEP 389: The argparse Module for Parsing Command Lines<a href="#pep-389-the-argparse-module-for-parsing-command-lines" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> module for parsing command-line arguments was added as a more powerful replacement for the <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a> module.

This means Python now supports three different modules for parsing command-line arguments: <a href="../library/getopt.html#module-getopt" class="reference internal" title="getopt: Portable parser for command line options; support both short and long option names."><span class="pre"><code class="sourceCode python">getopt</code></span></a>, <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a>, and <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a>. The <a href="../library/getopt.html#module-getopt" class="reference internal" title="getopt: Portable parser for command line options; support both short and long option names."><span class="pre"><code class="sourceCode python">getopt</code></span></a> module closely resembles the C library’s <span class="pre">`getopt()`</span> function, so it remains useful if you’re writing a Python prototype that will eventually be rewritten in C. <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a> becomes redundant, but there are no plans to remove it because there are many scripts still using it, and there’s no automated way to update these scripts. (Making the <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> API consistent with <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a>’s interface was discussed but rejected as too messy and difficult.)

In short, if you’re writing a new script and don’t need to worry about compatibility with earlier versions of Python, use <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> instead of <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a>.

Here’s an example:

<div class="highlight-default notranslate">

<div class="highlight">

    import argparse

    parser = argparse.ArgumentParser(description='Command-line example.')

    # Add optional switches
    parser.add_argument('-v', action='store_true', dest='is_verbose',
                        help='produce verbose output')
    parser.add_argument('-o', action='store', dest='output',
                        metavar='FILE',
                        help='direct output to FILE instead of stdout')
    parser.add_argument('-C', action='store', type=int, dest='context',
                        metavar='NUM', default=0,
                        help='display NUM lines of added context')

    # Allow any number of additional arguments.
    parser.add_argument(nargs='*', action='store', dest='inputs',
                        help='input filenames (default is stdin)')

    args = parser.parse_args()
    print args.__dict__

</div>

</div>

Unless you override it, <span class="pre">`-h`</span> and <span class="pre">`--help`</span> switches are automatically added, and produce neatly formatted output:

<div class="highlight-default notranslate">

<div class="highlight">

    -> ./python.exe argparse-example.py --help
    usage: argparse-example.py [-h] [-v] [-o FILE] [-C NUM] [inputs [inputs ...]]

    Command-line example.

    positional arguments:
      inputs      input filenames (default is stdin)

    optional arguments:
      -h, --help  show this help message and exit
      -v          produce verbose output
      -o FILE     direct output to FILE instead of stdout
      -C NUM      display NUM lines of added context

</div>

</div>

As with <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a>, the command-line switches and arguments are returned as an object with attributes named by the *dest* parameters:

<div class="highlight-default notranslate">

<div class="highlight">

    -> ./python.exe argparse-example.py -v
    {'output': None,
     'is_verbose': True,
     'context': 0,
     'inputs': []}

    -> ./python.exe argparse-example.py -v -o /tmp/output -C 4 file1 file2
    {'output': '/tmp/output',
     'is_verbose': True,
     'context': 4,
     'inputs': ['file1', 'file2']}

</div>

</div>

<a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> has much fancier validation than <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a>; you can specify an exact number of arguments as an integer, 0 or more arguments by passing <span class="pre">`'*'`</span>, 1 or more by passing <span class="pre">`'+'`</span>, or an optional argument with <span class="pre">`'?'`</span>. A top-level parser can contain sub-parsers to define subcommands that have different sets of switches, as in <span class="pre">`svn`</span>` `<span class="pre">`commit`</span>, <span class="pre">`svn`</span>` `<span class="pre">`checkout`</span>, etc. You can specify an argument’s type as <a href="../library/argparse.html#argparse.FileType" class="reference internal" title="argparse.FileType"><span class="pre"><code class="sourceCode python">FileType</code></span></a>, which will automatically open files for you and understands that <span class="pre">`'-'`</span> means standard input or output.

<div class="admonition seealso">

See also

<a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> documentation  
The documentation page of the argparse module.

<a href="../library/argparse.html#argparse-from-optparse" class="reference internal"><span class="std std-ref">Upgrading optparse code</span></a>  
Part of the Python documentation, describing how to convert code that uses <a href="../library/optparse.html#module-optparse" class="reference internal" title="optparse: Command-line option parsing library. (deprecated)"><span class="pre"><code class="sourceCode python">optparse</code></span></a>.

<span id="index-4" class="target"></span><a href="https://www.python.org/dev/peps/pep-0389" class="pep reference external"><strong>PEP 389</strong></a> - argparse - New Command Line Parsing Module  
PEP written and implemented by Steven Bethard.

</div>

</div>

<div id="pep-391-dictionary-based-configuration-for-logging" class="section">

## PEP 391: Dictionary-Based Configuration For Logging<a href="#pep-391-dictionary-based-configuration-for-logging" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> module is very flexible; applications can define a tree of logging subsystems, and each logger in this tree can filter out certain messages, format them differently, and direct messages to a varying number of handlers.

All this flexibility can require a lot of configuration. You can write Python statements to create objects and set their properties, but a complex set-up requires verbose but boring code. <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> also supports a <span class="pre">`fileConfig()`</span> function that parses a file, but the file format doesn’t support configuring filters, and it’s messier to generate programmatically.

Python 2.7 adds a <span class="pre">`dictConfig()`</span> function that uses a dictionary to configure logging. There are many ways to produce a dictionary from different sources: construct one with code; parse a file containing JSON; or use a YAML parsing library if one is installed. For more information see <a href="../library/logging.config.html#logging-config-api" class="reference internal"><span class="std std-ref">Configuration functions</span></a>.

The following example configures two loggers, the root logger and a logger named “network”. Messages sent to the root logger will be sent to the system log using the syslog protocol, and messages to the “network” logger will be written to a <span class="pre">`network.log`</span> file that will be rotated once the log reaches 1MB.

<div class="highlight-default notranslate">

<div class="highlight">

    import logging
    import logging.config

    configdict = {
     'version': 1,    # Configuration schema in use; must be 1 for now
     'formatters': {
         'standard': {
             'format': ('%(asctime)s %(name)-15s '
                        '%(levelname)-8s %(message)s')}},

     'handlers': {'netlog': {'backupCount': 10,
                         'class': 'logging.handlers.RotatingFileHandler',
                         'filename': '/logs/network.log',
                         'formatter': 'standard',
                         'level': 'INFO',
                         'maxBytes': 1000000},
                  'syslog': {'class': 'logging.handlers.SysLogHandler',
                             'formatter': 'standard',
                             'level': 'ERROR'}},

     # Specify all the subordinate loggers
     'loggers': {
                 'network': {
                             'handlers': ['netlog']
                 }
     },
     # Specify properties of the root logger
     'root': {
              'handlers': ['syslog']
     },
    }

    # Set up configuration
    logging.config.dictConfig(configdict)

    # As an example, log two error messages
    logger = logging.getLogger('/')
    logger.error('Database not found')

    netlogger = logging.getLogger('network')
    netlogger.error('Connection failed')

</div>

</div>

Three smaller enhancements to the <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> module, all implemented by Vinay Sajip, are:

- The <a href="../library/logging.handlers.html#logging.handlers.SysLogHandler" class="reference internal" title="logging.handlers.SysLogHandler"><span class="pre"><code class="sourceCode python">SysLogHandler</code></span></a> class now supports syslogging over TCP. The constructor has a *socktype* parameter giving the type of socket to use, either <a href="../library/socket.html#socket.SOCK_DGRAM" class="reference internal" title="socket.SOCK_DGRAM"><span class="pre"><code class="sourceCode python">socket.SOCK_DGRAM</code></span></a> for UDP or <a href="../library/socket.html#socket.SOCK_STREAM" class="reference internal" title="socket.SOCK_STREAM"><span class="pre"><code class="sourceCode python">socket.SOCK_STREAM</code></span></a> for TCP. The default protocol remains UDP.

- <a href="../library/logging.html#logging.Logger" class="reference internal" title="logging.Logger"><span class="pre"><code class="sourceCode python">Logger</code></span></a> instances gained a <a href="../library/logging.html#logging.Logger.getChild" class="reference internal" title="logging.Logger.getChild"><span class="pre"><code class="sourceCode python">getChild()</code></span></a> method that retrieves a descendant logger using a relative path. For example, once you retrieve a logger by doing <span class="pre">`log`</span>` `<span class="pre">`=`</span>` `<span class="pre">`getLogger('app')`</span>, calling <span class="pre">`log.getChild('network.listen')`</span> is equivalent to <span class="pre">`getLogger('app.network.listen')`</span>.

- The <a href="../library/logging.html#logging.LoggerAdapter" class="reference internal" title="logging.LoggerAdapter"><span class="pre"><code class="sourceCode python">LoggerAdapter</code></span></a> class gained an <span class="pre">`isEnabledFor()`</span> method that takes a *level* and returns whether the underlying logger would process a message of that level of importance.

<div class="admonition seealso">

See also

<span id="index-5" class="target"></span><a href="https://www.python.org/dev/peps/pep-0391" class="pep reference external"><strong>PEP 391</strong></a> - Dictionary-Based Configuration For Logging  
PEP written and implemented by Vinay Sajip.

</div>

</div>

<div id="pep-3106-dictionary-views" class="section">

## PEP 3106: Dictionary Views<a href="#pep-3106-dictionary-views" class="headerlink" title="Permalink to this headline">¶</a>

The dictionary methods <a href="../library/stdtypes.html#dict.keys" class="reference internal" title="dict.keys"><span class="pre"><code class="sourceCode python">keys()</code></span></a>, <a href="../library/stdtypes.html#dict.values" class="reference internal" title="dict.values"><span class="pre"><code class="sourceCode python">values()</code></span></a>, and <a href="../library/stdtypes.html#dict.items" class="reference internal" title="dict.items"><span class="pre"><code class="sourceCode python">items()</code></span></a> are different in Python 3.x. They return an object called a *view* instead of a fully materialized list.

It’s not possible to change the return values of <a href="../library/stdtypes.html#dict.keys" class="reference internal" title="dict.keys"><span class="pre"><code class="sourceCode python">keys()</code></span></a>, <a href="../library/stdtypes.html#dict.values" class="reference internal" title="dict.values"><span class="pre"><code class="sourceCode python">values()</code></span></a>, and <a href="../library/stdtypes.html#dict.items" class="reference internal" title="dict.items"><span class="pre"><code class="sourceCode python">items()</code></span></a> in Python 2.7 because too much code would break. Instead the 3.x versions were added under the new names <a href="../library/stdtypes.html#dict.viewkeys" class="reference internal" title="dict.viewkeys"><span class="pre"><code class="sourceCode python">viewkeys()</code></span></a>, <a href="../library/stdtypes.html#dict.viewvalues" class="reference internal" title="dict.viewvalues"><span class="pre"><code class="sourceCode python">viewvalues()</code></span></a>, and <a href="../library/stdtypes.html#dict.viewitems" class="reference internal" title="dict.viewitems"><span class="pre"><code class="sourceCode python">viewitems()</code></span></a>.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> d = dict((i*10, chr(65+i)) for i in range(26))
    >>> d
    {0: 'A', 130: 'N', 10: 'B', 140: 'O', 20: ..., 250: 'Z'}
    >>> d.viewkeys()
    dict_keys([0, 130, 10, 140, 20, 150, 30, ..., 250])

</div>

</div>

Views can be iterated over, but the key and item views also behave like sets. The <span class="pre">`&`</span> operator performs intersection, and <span class="pre">`|`</span> performs a union:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> d1 = dict((i*10, chr(65+i)) for i in range(26))
    >>> d2 = dict((i**.5, i) for i in range(1000))
    >>> d1.viewkeys() & d2.viewkeys()
    set([0.0, 10.0, 20.0, 30.0])
    >>> d1.viewkeys() | range(0, 30)
    set([0, 1, 130, 3, 4, 5, 6, ..., 120, 250])

</div>

</div>

The view keeps track of the dictionary and its contents change as the dictionary is modified:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> vk = d.viewkeys()
    >>> vk
    dict_keys([0, 130, 10, ..., 250])
    >>> d[260] = '&'
    >>> vk
    dict_keys([0, 130, 260, 10, ..., 250])

</div>

</div>

However, note that you can’t add or remove keys while you’re iterating over the view:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> for k in vk:
    ...     d[k*2] = k
    ...
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    RuntimeError: dictionary changed size during iteration

</div>

</div>

You can use the view methods in Python 2.x code, and the 2to3 converter will change them to the standard <a href="../library/stdtypes.html#dict.keys" class="reference internal" title="dict.keys"><span class="pre"><code class="sourceCode python">keys()</code></span></a>, <a href="../library/stdtypes.html#dict.values" class="reference internal" title="dict.values"><span class="pre"><code class="sourceCode python">values()</code></span></a>, and <a href="../library/stdtypes.html#dict.items" class="reference internal" title="dict.items"><span class="pre"><code class="sourceCode python">items()</code></span></a> methods.

<div class="admonition seealso">

See also

<span id="index-6" class="target"></span><a href="https://www.python.org/dev/peps/pep-3106" class="pep reference external"><strong>PEP 3106</strong></a> - Revamping dict.keys(), .values() and .items()  
PEP written by Guido van Rossum. Backported to 2.7 by Alexandre Vassalotti; <a href="https://bugs.python.org/issue1967" class="reference external">bpo-1967</a>.

</div>

</div>

<div id="pep-3137-the-memoryview-object" class="section">

## PEP 3137: The memoryview Object<a href="#pep-3137-the-memoryview-object" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> object provides a view of another object’s memory content that matches the <span class="pre">`bytes`</span> type’s interface.

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import string
    >>> m = memoryview(string.letters)
    >>> m
    <memory at 0x37f850>
    >>> len(m)           # Returns length of underlying object
    52
    >>> m[0], m[25], m[26]   # Indexing returns one byte
    ('a', 'z', 'A')
    >>> m2 = m[0:26]         # Slicing returns another memoryview
    >>> m2
    <memory at 0x37f080>

</div>

</div>

The content of the view can be converted to a string of bytes or a list of integers:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> m2.tobytes()
    'abcdefghijklmnopqrstuvwxyz'
    >>> m2.tolist()
    [97, 98, 99, 100, 101, 102, 103, ... 121, 122]
    >>>

</div>

</div>

<a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> objects allow modifying the underlying object if it’s a mutable object.

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> m2[0] = 75
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: cannot modify read-only memory
    >>> b = bytearray(string.letters)  # Creating a mutable object
    >>> b
    bytearray(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    >>> mb = memoryview(b)
    >>> mb[0] = '*'         # Assign to view, changing the bytearray.
    >>> b[0:5]              # The bytearray has been changed.
    bytearray(b'*bcde')
    >>>

</div>

</div>

<div class="admonition seealso">

See also

<span id="index-7" class="target"></span><a href="https://www.python.org/dev/peps/pep-3137" class="pep reference external"><strong>PEP 3137</strong></a> - Immutable Bytes and Mutable Buffer  
PEP written by Guido van Rossum. Implemented by Travis Oliphant, Antoine Pitrou and others. Backported to 2.7 by Antoine Pitrou; <a href="https://bugs.python.org/issue2396" class="reference external">bpo-2396</a>.

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Permalink to this headline">¶</a>

Some smaller changes made to the core Python language are:

- The syntax for set literals has been backported from Python 3.x. Curly brackets are used to surround the contents of the resulting mutable set; set literals are distinguished from dictionaries by not containing colons and values. <span class="pre">`{}`</span> continues to represent an empty dictionary; use <span class="pre">`set()`</span> for an empty set.

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> {1, 2, 3, 4, 5}
      set([1, 2, 3, 4, 5])
      >>> set() # empty set
      set([])
      >>> {}    # empty dict
      {}

  </div>

  </div>

  Backported by Alexandre Vassalotti; <a href="https://bugs.python.org/issue2335" class="reference external">bpo-2335</a>.

- Dictionary and set comprehensions are another feature backported from 3.x, generalizing list/generator comprehensions to use the literal syntax for sets and dictionaries.

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> {x: x*x for x in range(6)}
      {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
      >>> {('a'*x) for x in range(6)}
      set(['', 'a', 'aa', 'aaa', 'aaaa', 'aaaaa'])

  </div>

  </div>

  Backported by Alexandre Vassalotti; <a href="https://bugs.python.org/issue2333" class="reference external">bpo-2333</a>.

- The <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement can now use multiple context managers in one statement. Context managers are processed from left to right and each one is treated as beginning a new <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement. This means that:

  <div class="highlight-default notranslate">

  <div class="highlight">

      with A() as a, B() as b:
          ... suite of statements ...

  </div>

  </div>

  is equivalent to:

  <div class="highlight-default notranslate">

  <div class="highlight">

      with A() as a:
          with B() as b:
              ... suite of statements ...

  </div>

  </div>

  The <a href="../library/contextlib.html#contextlib.nested" class="reference internal" title="contextlib.nested"><span class="pre"><code class="sourceCode python">contextlib.nested()</code></span></a> function provides a very similar function, so it’s no longer necessary and has been deprecated.

  (Proposed in <a href="https://codereview.appspot.com/53094" class="reference external">https://codereview.appspot.com/53094</a>; implemented by Georg Brandl.)

- Conversions between floating-point numbers and strings are now correctly rounded on most platforms. These conversions occur in many different places: <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a> on floats and complex numbers; the <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> and <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a> constructors; numeric formatting; serializing and deserializing floats and complex numbers using the <a href="../library/marshal.html#module-marshal" class="reference internal" title="marshal: Convert Python objects to streams of bytes and back (with different constraints)."><span class="pre"><code class="sourceCode python">marshal</code></span></a>, <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> and <a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> modules; parsing of float and imaginary literals in Python code; and <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a>-to-float conversion.

  Related to this, the <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> of a floating-point number *x* now returns a result based on the shortest decimal string that’s guaranteed to round back to *x* under correct rounding (with round-half-to-even rounding mode). Previously it gave a string based on rounding x to 17 decimal digits.

  The rounding library responsible for this improvement works on Windows and on Unix platforms using the gcc, icc, or suncc compilers. There may be a small number of platforms where correct operation of this code cannot be guaranteed, so the code is not used on such systems. You can find out which code is being used by checking <a href="../library/sys.html#sys.float_repr_style" class="reference internal" title="sys.float_repr_style"><span class="pre"><code class="sourceCode python">sys.float_repr_style</code></span></a>, which will be <span class="pre">`short`</span> if the new code is in use and <span class="pre">`legacy`</span> if it isn’t.

  Implemented by Eric Smith and Mark Dickinson, using David Gay’s <span class="pre">`dtoa.c`</span> library; <a href="https://bugs.python.org/issue7117" class="reference external">bpo-7117</a>.

- Conversions from long integers and regular integers to floating point now round differently, returning the floating-point number closest to the number. This doesn’t matter for small integers that can be converted exactly, but for large numbers that will unavoidably lose precision, Python 2.7 now approximates more closely. For example, Python 2.6 computed the following:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> n = 295147905179352891391
      >>> float(n)
      2.9514790517935283e+20
      >>> n - long(float(n))
      65535L

  </div>

  </div>

  Python 2.7’s floating-point result is larger, but much closer to the true value:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> n = 295147905179352891391
      >>> float(n)
      2.9514790517935289e+20
      >>> n - long(float(n))
      -1L

  </div>

  </div>

  (Implemented by Mark Dickinson; <a href="https://bugs.python.org/issue3166" class="reference external">bpo-3166</a>.)

  Integer division is also more accurate in its rounding behaviours. (Also implemented by Mark Dickinson; <a href="https://bugs.python.org/issue1811" class="reference external">bpo-1811</a>.)

- Implicit coercion for complex numbers has been removed; the interpreter will no longer ever attempt to call a <a href="../reference/datamodel.html#object.__coerce__" class="reference internal" title="object.__coerce__"><span class="pre"><code class="sourceCode python"><span class="fu">__coerce__</span>()</code></span></a> method on complex objects. (Removed by Meador Inge and Mark Dickinson; <a href="https://bugs.python.org/issue5211" class="reference external">bpo-5211</a>.)

- The <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a> method now supports automatic numbering of the replacement fields. This makes using <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a> more closely resemble using <span class="pre">`%s`</span> formatting:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> '{}:{}:{}'.format(2009, 04, 'Sunday')
      '2009:4:Sunday'
      >>> '{}:{}:{day}'.format(2009, 4, day='Sunday')
      '2009:4:Sunday'

  </div>

  </div>

  The auto-numbering takes the fields from left to right, so the first <span class="pre">`{...}`</span> specifier will use the first argument to <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>.<span class="bu">format</span>()</code></span></a>, the next specifier will use the next argument, and so on. You can’t mix auto-numbering and explicit numbering – either number all of your specifier fields or none of them – but you can mix auto-numbering and named fields, as in the second example above. (Contributed by Eric Smith; <a href="https://bugs.python.org/issue5237" class="reference external">bpo-5237</a>.)

  Complex numbers now correctly support usage with <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a>, and default to being right-aligned. Specifying a precision or comma-separation applies to both the real and imaginary parts of the number, but a specified field width and alignment is applied to the whole of the resulting <span class="pre">`1.5+3j`</span> output. (Contributed by Eric Smith; <a href="https://bugs.python.org/issue1588" class="reference external">bpo-1588</a> and <a href="https://bugs.python.org/issue7988" class="reference external">bpo-7988</a>.)

  The ‘F’ format code now always formats its output using uppercase characters, so it will now produce ‘INF’ and ‘NAN’. (Contributed by Eric Smith; <a href="https://bugs.python.org/issue3382" class="reference external">bpo-3382</a>.)

  A low-level change: the <span class="pre">`object.__format__()`</span> method now triggers a <a href="../library/exceptions.html#exceptions.PendingDeprecationWarning" class="reference internal" title="exceptions.PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a> if it’s passed a format string, because the <span class="pre">`__format__()`</span> method for <a href="../library/functions.html#object" class="reference internal" title="object"><span class="pre"><code class="sourceCode python"><span class="bu">object</span></code></span></a> converts the object to a string representation and formats that. Previously the method silently applied the format string to the string representation, but that could hide mistakes in Python code. If you’re supplying formatting information such as an alignment or precision, presumably you’re expecting the formatting to be applied in some object-specific way. (Fixed by Eric Smith; <a href="https://bugs.python.org/issue7994" class="reference external">bpo-7994</a>.)

- The <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a> and <a href="../library/functions.html#long" class="reference internal" title="long"><span class="pre"><code class="sourceCode python"><span class="bu">long</span>()</code></span></a> types gained a <span class="pre">`bit_length`</span> method that returns the number of bits necessary to represent its argument in binary:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> n = 37
      >>> bin(n)
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

  (Contributed by Fredrik Johansson and Victor Stinner; <a href="https://bugs.python.org/issue3439" class="reference external">bpo-3439</a>.)

- The <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statement will no longer try an absolute import if a relative import (e.g. <span class="pre">`from`</span>` `<span class="pre">`.os`</span>` `<span class="pre">`import`</span>` `<span class="pre">`sep`</span>) fails. This fixes a bug, but could possibly break certain <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statements that were only working by accident. (Fixed by Meador Inge; <a href="https://bugs.python.org/issue7902" class="reference external">bpo-7902</a>.)

- It’s now possible for a subclass of the built-in <a href="../library/functions.html#unicode" class="reference internal" title="unicode"><span class="pre"><code class="sourceCode python"><span class="bu">unicode</span></code></span></a> type to override the <a href="../reference/datamodel.html#object.__unicode__" class="reference internal" title="object.__unicode__"><span class="pre"><code class="sourceCode python"><span class="fu">__unicode__</span>()</code></span></a> method. (Implemented by Victor Stinner; <a href="https://bugs.python.org/issue1583863" class="reference external">bpo-1583863</a>.)

- The <a href="../library/functions.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> type’s <span class="pre">`translate()`</span> method now accepts <span class="pre">`None`</span> as its first argument. (Fixed by Georg Brandl; <a href="https://bugs.python.org/issue4759" class="reference external">bpo-4759</a>.)

- When using <span class="pre">`@classmethod`</span> and <span class="pre">`@staticmethod`</span> to wrap methods as class or static methods, the wrapper object now exposes the wrapped function as their <span class="pre">`__func__`</span> attribute. (Contributed by Amaury Forgeot d’Arc, after a suggestion by George Sakkis; <a href="https://bugs.python.org/issue5982" class="reference external">bpo-5982</a>.)

- When a restricted set of attributes were set using <span class="pre">`__slots__`</span>, deleting an unset attribute would not raise <a href="../library/exceptions.html#exceptions.AttributeError" class="reference internal" title="exceptions.AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> as you would expect. Fixed by Benjamin Peterson; <a href="https://bugs.python.org/issue7604" class="reference external">bpo-7604</a>.)

- Two new encodings are now supported: “cp720”, used primarily for Arabic text; and “cp858”, a variant of CP 850 that adds the euro symbol. (CP720 contributed by Alexander Belchenko and Amaury Forgeot d’Arc in <a href="https://bugs.python.org/issue1616979" class="reference external">bpo-1616979</a>; CP858 contributed by Tim Hatch in <a href="https://bugs.python.org/issue8016" class="reference external">bpo-8016</a>.)

- The <a href="../library/functions.html#file" class="reference internal" title="file"><span class="pre"><code class="sourceCode python"><span class="bu">file</span></code></span></a> object will now set the <span class="pre">`filename`</span> attribute on the <a href="../library/exceptions.html#exceptions.IOError" class="reference internal" title="exceptions.IOError"><span class="pre"><code class="sourceCode python"><span class="pp">IOError</span></code></span></a> exception when trying to open a directory on POSIX platforms (noted by Jan Kaliszewski; <a href="https://bugs.python.org/issue4764" class="reference external">bpo-4764</a>), and now explicitly checks for and forbids writing to read-only file objects instead of trusting the C library to catch and report the error (fixed by Stefan Krah; <a href="https://bugs.python.org/issue5677" class="reference external">bpo-5677</a>).

- The Python tokenizer now translates line endings itself, so the <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a> built-in function now accepts code using any line-ending convention. Additionally, it no longer requires that the code end in a newline.

- Extra parentheses in function definitions are illegal in Python 3.x, meaning that you get a syntax error from <span class="pre">`def`</span>` `<span class="pre">`f((x)):`</span>` `<span class="pre">`pass`</span>. In Python3-warning mode, Python 2.7 will now warn about this odd usage. (Noted by James Lingard; <a href="https://bugs.python.org/issue7362" class="reference external">bpo-7362</a>.)

- It’s now possible to create weak references to old-style class objects. New-style classes were always weak-referenceable. (Fixed by Antoine Pitrou; <a href="https://bugs.python.org/issue8268" class="reference external">bpo-8268</a>.)

- When a module object is garbage-collected, the module’s dictionary is now only cleared if no one else is holding a reference to the dictionary (<a href="https://bugs.python.org/issue7140" class="reference external">bpo-7140</a>).

<div id="interpreter-changes" class="section">

<span id="new-27-interpreter"></span>

### Interpreter Changes<a href="#interpreter-changes" class="headerlink" title="Permalink to this headline">¶</a>

A new environment variable, <span id="index-8" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONWARNINGS" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONWARNINGS</code></span></a>, allows controlling warnings. It should be set to a string containing warning settings, equivalent to those used with the <a href="../using/cmdline.html#cmdoption-w" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-W</code></span></a> switch, separated by commas. (Contributed by Brian Curtin; <a href="https://bugs.python.org/issue7301" class="reference external">bpo-7301</a>.)

For example, the following setting will print warnings every time they occur, but turn warnings from the <a href="../library/cookie.html#module-Cookie" class="reference internal" title="Cookie: Support for HTTP state management (cookies)."><span class="pre"><code class="sourceCode python">Cookie</code></span></a> module into an error. (The exact syntax for setting an environment variable varies across operating systems and shells.)

<div class="highlight-default notranslate">

<div class="highlight">

    export PYTHONWARNINGS=all,error:::Cookie:0

</div>

</div>

</div>

<div id="optimizations" class="section">

### Optimizations<a href="#optimizations" class="headerlink" title="Permalink to this headline">¶</a>

Several performance enhancements have been added:

- A new opcode was added to perform the initial setup for <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statements, looking up the <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> methods. (Contributed by Benjamin Peterson.)

- The garbage collector now performs better for one common usage pattern: when many objects are being allocated without deallocating any of them. This would previously take quadratic time for garbage collection, but now the number of full garbage collections is reduced as the number of objects on the heap grows. The new logic only performs a full garbage collection pass when the middle generation has been collected 10 times and when the number of survivor objects from the middle generation exceeds 10% of the number of objects in the oldest generation. (Suggested by Martin von Löwis and implemented by Antoine Pitrou; <a href="https://bugs.python.org/issue4074" class="reference external">bpo-4074</a>.)

- The garbage collector tries to avoid tracking simple containers which can’t be part of a cycle. In Python 2.7, this is now true for tuples and dicts containing atomic types (such as ints, strings, etc.). Transitively, a dict containing tuples of atomic types won’t be tracked either. This helps reduce the cost of each garbage collection by decreasing the number of objects to be considered and traversed by the collector. (Contributed by Antoine Pitrou; <a href="https://bugs.python.org/issue4688" class="reference external">bpo-4688</a>.)

- Long integers are now stored internally either in base 2\*\*15 or in base 2\*\*30, the base being determined at build time. Previously, they were always stored in base 2\*\*15. Using base 2\*\*30 gives significant performance improvements on 64-bit machines, but benchmark results on 32-bit machines have been mixed. Therefore, the default is to use base 2\*\*30 on 64-bit machines and base 2\*\*15 on 32-bit machines; on Unix, there’s a new configure option <span class="pre">`--enable-big-digits`</span> that can be used to override this default.

  Apart from the performance improvements this change should be invisible to end users, with one exception: for testing and debugging purposes there’s a new structseq <a href="../library/sys.html#sys.long_info" class="reference internal" title="sys.long_info"><span class="pre"><code class="sourceCode python">sys.long_info</code></span></a> that provides information about the internal format, giving the number of bits per digit and the size in bytes of the C type used to store each digit:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import sys
      >>> sys.long_info
      sys.long_info(bits_per_digit=30, sizeof_digit=4)

  </div>

  </div>

  (Contributed by Mark Dickinson; <a href="https://bugs.python.org/issue4258" class="reference external">bpo-4258</a>.)

  Another set of changes made long objects a few bytes smaller: 2 bytes smaller on 32-bit systems and 6 bytes on 64-bit. (Contributed by Mark Dickinson; <a href="https://bugs.python.org/issue5260" class="reference external">bpo-5260</a>.)

- The division algorithm for long integers has been made faster by tightening the inner loop, doing shifts instead of multiplications, and fixing an unnecessary extra iteration. Various benchmarks show speedups of between 50% and 150% for long integer divisions and modulo operations. (Contributed by Mark Dickinson; <a href="https://bugs.python.org/issue5512" class="reference external">bpo-5512</a>.) Bitwise operations are also significantly faster (initial patch by Gregory Smith; <a href="https://bugs.python.org/issue1087418" class="reference external">bpo-1087418</a>).

- The implementation of <span class="pre">`%`</span> checks for the left-side operand being a Python string and special-cases it; this results in a 1–3% performance increase for applications that frequently use <span class="pre">`%`</span> with strings, such as templating libraries. (Implemented by Collin Winter; <a href="https://bugs.python.org/issue5176" class="reference external">bpo-5176</a>.)

- List comprehensions with an <span class="pre">`if`</span> condition are compiled into faster bytecode. (Patch by Antoine Pitrou, back-ported to 2.7 by Jeffrey Yasskin; <a href="https://bugs.python.org/issue4715" class="reference external">bpo-4715</a>.)

- Converting an integer or long integer to a decimal string was made faster by special-casing base 10 instead of using a generalized conversion function that supports arbitrary bases. (Patch by Gawain Bolton; <a href="https://bugs.python.org/issue6713" class="reference external">bpo-6713</a>.)

- The <span class="pre">`split()`</span>, <span class="pre">`replace()`</span>, <span class="pre">`rindex()`</span>, <span class="pre">`rpartition()`</span>, and <span class="pre">`rsplit()`</span> methods of string-like types (strings, Unicode strings, and <a href="../library/functions.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> objects) now use a fast reverse-search algorithm instead of a character-by-character scan. This is sometimes faster by a factor of 10. (Added by Florent Xicluna; <a href="https://bugs.python.org/issue7462" class="reference external">bpo-7462</a> and <a href="https://bugs.python.org/issue7622" class="reference external">bpo-7622</a>.)

- The <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> and <a href="../library/pickle.html#module-cPickle" class="reference internal" title="cPickle: Faster version of pickle, but not subclassable."><span class="pre"><code class="sourceCode python">cPickle</code></span></a> modules now automatically intern the strings used for attribute names, reducing memory usage of the objects resulting from unpickling. (Contributed by Jake McGuire; <a href="https://bugs.python.org/issue5084" class="reference external">bpo-5084</a>.)

- The <a href="../library/pickle.html#module-cPickle" class="reference internal" title="cPickle: Faster version of pickle, but not subclassable."><span class="pre"><code class="sourceCode python">cPickle</code></span></a> module now special-cases dictionaries, nearly halving the time required to pickle them. (Contributed by Collin Winter; <a href="https://bugs.python.org/issue5670" class="reference external">bpo-5670</a>.)

</div>

</div>

<div id="new-and-improved-modules" class="section">

## New and Improved Modules<a href="#new-and-improved-modules" class="headerlink" title="Permalink to this headline">¶</a>

As in every release, Python’s standard library received a number of enhancements and bug fixes. Here’s a partial list of the most notable changes, sorted alphabetically by module name. Consult the <span class="pre">`Misc/NEWS`</span> file in the source tree for a more complete list of changes, or look through the Subversion logs for all the details.

- The <a href="../library/bdb.html#module-bdb" class="reference internal" title="bdb: Debugger framework."><span class="pre"><code class="sourceCode python">bdb</code></span></a> module’s base debugging class <a href="../library/bdb.html#bdb.Bdb" class="reference internal" title="bdb.Bdb"><span class="pre"><code class="sourceCode python">Bdb</code></span></a> gained a feature for skipping modules. The constructor now takes an iterable containing glob-style patterns such as <span class="pre">`django.*`</span>; the debugger will not step into stack frames from a module that matches one of these patterns. (Contributed by Maru Newby after a suggestion by Senthil Kumaran; <a href="https://bugs.python.org/issue5142" class="reference external">bpo-5142</a>.)

- The <a href="../library/binascii.html#module-binascii" class="reference internal" title="binascii: Tools for converting between binary and various ASCII-encoded binary representations."><span class="pre"><code class="sourceCode python">binascii</code></span></a> module now supports the buffer API, so it can be used with <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> instances and other similar buffer objects. (Backported from 3.x by Florent Xicluna; <a href="https://bugs.python.org/issue7703" class="reference external">bpo-7703</a>.)

- Updated module: the <a href="../library/bsddb.html#module-bsddb" class="reference internal" title="bsddb: Interface to Berkeley DB database library"><span class="pre"><code class="sourceCode python">bsddb</code></span></a> module has been updated from 4.7.2devel9 to version 4.8.4 of <a href="https://www.jcea.es/programacion/pybsddb.htm" class="reference external">the pybsddb package</a>. The new version features better Python 3.x compatibility, various bug fixes, and adds several new BerkeleyDB flags and methods. (Updated by Jesús Cea Avión; <a href="https://bugs.python.org/issue8156" class="reference external">bpo-8156</a>. The pybsddb changelog can be read at <a href="http://hg.jcea.es/pybsddb/file/tip/ChangeLog" class="reference external">http://hg.jcea.es/pybsddb/file/tip/ChangeLog</a>.)

- The <a href="../library/bz2.html#module-bz2" class="reference internal" title="bz2: Interface to compression and decompression routines compatible with bzip2."><span class="pre"><code class="sourceCode python">bz2</code></span></a> module’s <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">BZ2File</code></span></a> now supports the context management protocol, so you can write <span class="pre">`with`</span>` `<span class="pre">`bz2.BZ2File(...)`</span>` `<span class="pre">`as`</span>` `<span class="pre">`f:`</span>. (Contributed by Hagen Fürstenau; <a href="https://bugs.python.org/issue3860" class="reference external">bpo-3860</a>.)

- New class: the <a href="../library/collections.html#collections.Counter" class="reference internal" title="collections.Counter"><span class="pre"><code class="sourceCode python">Counter</code></span></a> class in the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: High-performance datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module is useful for tallying data. <a href="../library/collections.html#collections.Counter" class="reference internal" title="collections.Counter"><span class="pre"><code class="sourceCode python">Counter</code></span></a> instances behave mostly like dictionaries but return zero for missing keys instead of raising a <a href="../library/exceptions.html#exceptions.KeyError" class="reference internal" title="exceptions.KeyError"><span class="pre"><code class="sourceCode python"><span class="pp">KeyError</span></code></span></a>:

  <div class="highlight-pycon3 notranslate">

  <div class="highlight">

      >>> from collections import Counter
      >>> c = Counter()
      >>> for letter in 'here is a sample of english text':
      ...   c[letter] += 1
      ...
      >>> c
      Counter({' ': 6, 'e': 5, 's': 3, 'a': 2, 'i': 2, 'h': 2,
      'l': 2, 't': 2, 'g': 1, 'f': 1, 'm': 1, 'o': 1, 'n': 1,
      'p': 1, 'r': 1, 'x': 1})
      >>> c['e']
      5
      >>> c['z']
      0

  </div>

  </div>

  There are three additional <a href="../library/collections.html#collections.Counter" class="reference internal" title="collections.Counter"><span class="pre"><code class="sourceCode python">Counter</code></span></a> methods. <a href="../library/collections.html#collections.Counter.most_common" class="reference internal" title="collections.Counter.most_common"><span class="pre"><code class="sourceCode python">most_common()</code></span></a> returns the N most common elements and their counts. <a href="../library/collections.html#collections.Counter.elements" class="reference internal" title="collections.Counter.elements"><span class="pre"><code class="sourceCode python">elements()</code></span></a> returns an iterator over the contained elements, repeating each element as many times as its count. <a href="../library/collections.html#collections.Counter.subtract" class="reference internal" title="collections.Counter.subtract"><span class="pre"><code class="sourceCode python">subtract()</code></span></a> takes an iterable and subtracts one for each element instead of adding; if the argument is a dictionary or another <span class="pre">`Counter`</span>, the counts are subtracted.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> c.most_common(5)
      [(' ', 6), ('e', 5), ('s', 3), ('a', 2), ('i', 2)]
      >>> c.elements() ->
         'a', 'a', ' ', ' ', ' ', ' ', ' ', ' ',
         'e', 'e', 'e', 'e', 'e', 'g', 'f', 'i', 'i',
         'h', 'h', 'm', 'l', 'l', 'o', 'n', 'p', 's',
         's', 's', 'r', 't', 't', 'x'
      >>> c['e']
      5
      >>> c.subtract('very heavy on the letter e')
      >>> c['e']    # Count is now lower
      -1

  </div>

  </div>

  Contributed by Raymond Hettinger; <a href="https://bugs.python.org/issue1696199" class="reference external">bpo-1696199</a>.

  New class: <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">OrderedDict</code></span></a> is described in the earlier section <a href="#pep-0372" class="reference internal"><span class="std std-ref">PEP 372: Adding an Ordered Dictionary to collections</span></a>.

  New method: The <a href="../library/collections.html#collections.deque" class="reference internal" title="collections.deque"><span class="pre"><code class="sourceCode python">deque</code></span></a> data type now has a <a href="../library/collections.html#collections.deque.count" class="reference internal" title="collections.deque.count"><span class="pre"><code class="sourceCode python">count()</code></span></a> method that returns the number of contained elements equal to the supplied argument *x*, and a <a href="../library/collections.html#collections.deque.reverse" class="reference internal" title="collections.deque.reverse"><span class="pre"><code class="sourceCode python">reverse()</code></span></a> method that reverses the elements of the deque in-place. <a href="../library/collections.html#collections.deque" class="reference internal" title="collections.deque"><span class="pre"><code class="sourceCode python">deque</code></span></a> also exposes its maximum length as the read-only <a href="../library/collections.html#collections.deque.maxlen" class="reference internal" title="collections.deque.maxlen"><span class="pre"><code class="sourceCode python">maxlen</code></span></a> attribute. (Both features added by Raymond Hettinger.)

  The <a href="../library/collections.html#collections.namedtuple" class="reference internal" title="collections.namedtuple"><span class="pre"><code class="sourceCode python">namedtuple</code></span></a> class now has an optional *rename* parameter. If *rename* is true, field names that are invalid because they’ve been repeated or aren’t legal Python identifiers will be renamed to legal names that are derived from the field’s position within the list of fields:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> from collections import namedtuple
      >>> T = namedtuple('T', ['field1', '$illegal', 'for', 'field2'], rename=True)
      >>> T._fields
      ('field1', '_1', '_2', 'field2')

  </div>

  </div>

  (Added by Raymond Hettinger; <a href="https://bugs.python.org/issue1818" class="reference external">bpo-1818</a>.)

  Finally, the <a href="../library/collections.html#collections.Mapping" class="reference internal" title="collections.Mapping"><span class="pre"><code class="sourceCode python">Mapping</code></span></a> abstract base class now returns <a href="../library/constants.html#NotImplemented" class="reference internal" title="NotImplemented"><span class="pre"><code class="sourceCode python"><span class="va">NotImplemented</span></code></span></a> if a mapping is compared to another type that isn’t a <span class="pre">`Mapping`</span>. (Fixed by Daniel Stutzbach; <a href="https://bugs.python.org/issue8729" class="reference external">bpo-8729</a>.)

- Constructors for the parsing classes in the <a href="../library/configparser.html#module-ConfigParser" class="reference internal" title="ConfigParser: Configuration file parser."><span class="pre"><code class="sourceCode python">ConfigParser</code></span></a> module now take an *allow_no_value* parameter, defaulting to false; if true, options without values will be allowed. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> import ConfigParser, StringIO
      >>> sample_config = """
      ... [mysqld]
      ... user = mysql
      ... pid-file = /var/run/mysqld/mysqld.pid
      ... skip-bdb
      ... """
      >>> config = ConfigParser.RawConfigParser(allow_no_value=True)
      >>> config.readfp(StringIO.StringIO(sample_config))
      >>> config.get('mysqld', 'user')
      'mysql'
      >>> print config.get('mysqld', 'skip-bdb')
      None
      >>> print config.get('mysqld', 'unknown')
      Traceback (most recent call last):
        ...
      NoOptionError: No option 'unknown' in section: 'mysqld'

  </div>

  </div>

  (Contributed by Mats Kindahl; <a href="https://bugs.python.org/issue7005" class="reference external">bpo-7005</a>.)

- Deprecated function: <a href="../library/contextlib.html#contextlib.nested" class="reference internal" title="contextlib.nested"><span class="pre"><code class="sourceCode python">contextlib.nested()</code></span></a>, which allows handling more than one context manager with a single <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement, has been deprecated, because the <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement now supports multiple context managers.

- The <a href="../library/cookielib.html#module-cookielib" class="reference internal" title="cookielib: Classes for automatic handling of HTTP cookies."><span class="pre"><code class="sourceCode python">cookielib</code></span></a> module now ignores cookies that have an invalid version field, one that doesn’t contain an integer value. (Fixed by John J. Lee; <a href="https://bugs.python.org/issue3924" class="reference external">bpo-3924</a>.)

- The <a href="../library/copy.html#module-copy" class="reference internal" title="copy: Shallow and deep copy operations."><span class="pre"><code class="sourceCode python">copy</code></span></a> module’s <a href="../library/copy.html#copy.deepcopy" class="reference internal" title="copy.deepcopy"><span class="pre"><code class="sourceCode python">deepcopy()</code></span></a> function will now correctly copy bound instance methods. (Implemented by Robert Collins; <a href="https://bugs.python.org/issue1515" class="reference external">bpo-1515</a>.)

- The <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> module now always converts <span class="pre">`None`</span> to a C NULL pointer for arguments declared as pointers. (Changed by Thomas Heller; <a href="https://bugs.python.org/issue4606" class="reference external">bpo-4606</a>.) The underlying <a href="https://sourceware.org/libffi/" class="reference external">libffi library</a> has been updated to version 3.0.9, containing various fixes for different platforms. (Updated by Matthias Klose; <a href="https://bugs.python.org/issue8142" class="reference external">bpo-8142</a>.)

- New method: the <a href="../library/datetime.html#module-datetime" class="reference internal" title="datetime: Basic date and time types."><span class="pre"><code class="sourceCode python">datetime</code></span></a> module’s <a href="../library/datetime.html#datetime.timedelta" class="reference internal" title="datetime.timedelta"><span class="pre"><code class="sourceCode python">timedelta</code></span></a> class gained a <a href="../library/datetime.html#datetime.timedelta.total_seconds" class="reference internal" title="datetime.timedelta.total_seconds"><span class="pre"><code class="sourceCode python">total_seconds()</code></span></a> method that returns the number of seconds in the duration. (Contributed by Brian Quinlan; <a href="https://bugs.python.org/issue5788" class="reference external">bpo-5788</a>.)

- New method: the <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> class gained a <a href="../library/decimal.html#decimal.Decimal.from_float" class="reference internal" title="decimal.Decimal.from_float"><span class="pre"><code class="sourceCode python">from_float()</code></span></a> class method that performs an exact conversion of a floating-point number to a <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a>. This exact conversion strives for the closest decimal approximation to the floating-point representation’s value; the resulting decimal value will therefore still include the inaccuracy, if any. For example, <span class="pre">`Decimal.from_float(0.1)`</span> returns <span class="pre">`Decimal('0.1000000000000000055511151231257827021181583404541015625')`</span>. (Implemented by Raymond Hettinger; <a href="https://bugs.python.org/issue4796" class="reference external">bpo-4796</a>.)

  Comparing instances of <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> with floating-point numbers now produces sensible results based on the numeric values of the operands. Previously such comparisons would fall back to Python’s default rules for comparing objects, which produced arbitrary results based on their type. Note that you still cannot combine <span class="pre">`Decimal`</span> and floating-point in other operations such as addition, since you should be explicitly choosing how to convert between float and <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a>. (Fixed by Mark Dickinson; <a href="https://bugs.python.org/issue2531" class="reference external">bpo-2531</a>.)

  The constructor for <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> now accepts floating-point numbers (added by Raymond Hettinger; <a href="https://bugs.python.org/issue8257" class="reference external">bpo-8257</a>) and non-European Unicode characters such as Arabic-Indic digits (contributed by Mark Dickinson; <a href="https://bugs.python.org/issue6595" class="reference external">bpo-6595</a>).

  Most of the methods of the <a href="../library/decimal.html#decimal.Context" class="reference internal" title="decimal.Context"><span class="pre"><code class="sourceCode python">Context</code></span></a> class now accept integers as well as <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> instances; the only exceptions are the <a href="../library/decimal.html#decimal.Context.canonical" class="reference internal" title="decimal.Context.canonical"><span class="pre"><code class="sourceCode python">canonical()</code></span></a> and <a href="../library/decimal.html#decimal.Context.is_canonical" class="reference internal" title="decimal.Context.is_canonical"><span class="pre"><code class="sourceCode python">is_canonical()</code></span></a> methods. (Patch by Juan José Conti; <a href="https://bugs.python.org/issue7633" class="reference external">bpo-7633</a>.)

  When using <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> instances with a string’s <a href="../library/stdtypes.html#str.format" class="reference internal" title="str.format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> method, the default alignment was previously left-alignment. This has been changed to right-alignment, which is more sensible for numeric types. (Changed by Mark Dickinson; <a href="https://bugs.python.org/issue6857" class="reference external">bpo-6857</a>.)

  Comparisons involving a signaling NaN value (or <span class="pre">`sNAN`</span>) now signal <span class="pre">`InvalidOperation`</span> instead of silently returning a true or false value depending on the comparison operator. Quiet NaN values (or <span class="pre">`NaN`</span>) are now hashable. (Fixed by Mark Dickinson; <a href="https://bugs.python.org/issue7279" class="reference external">bpo-7279</a>.)

- The <a href="../library/difflib.html#module-difflib" class="reference internal" title="difflib: Helpers for computing differences between objects."><span class="pre"><code class="sourceCode python">difflib</code></span></a> module now produces output that is more compatible with modern **diff**/**patch** tools through one small change, using a tab character instead of spaces as a separator in the header giving the filename. (Fixed by Anatoly Techtonik; <a href="https://bugs.python.org/issue7585" class="reference external">bpo-7585</a>.)

- The Distutils <span class="pre">`sdist`</span> command now always regenerates the <span class="pre">`MANIFEST`</span> file, since even if the <span class="pre">`MANIFEST.in`</span> or <span class="pre">`setup.py`</span> files haven’t been modified, the user might have created some new files that should be included. (Fixed by Tarek Ziadé; <a href="https://bugs.python.org/issue8688" class="reference external">bpo-8688</a>.)

- The <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> module’s <span class="pre">`IGNORE_EXCEPTION_DETAIL`</span> flag will now ignore the name of the module containing the exception being tested. (Patch by Lennart Regebro; <a href="https://bugs.python.org/issue7490" class="reference external">bpo-7490</a>.)

- The <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages, including MIME documents."><span class="pre"><code class="sourceCode python">email</code></span></a> module’s <a href="../library/email.message.html#email.message.Message" class="reference internal" title="email.message.Message"><span class="pre"><code class="sourceCode python">Message</code></span></a> class will now accept a Unicode-valued payload, automatically converting the payload to the encoding specified by <span class="pre">`output_charset`</span>. (Added by R. David Murray; <a href="https://bugs.python.org/issue1368247" class="reference external">bpo-1368247</a>.)

- The <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">Fraction</code></span></a> class now accepts a single float or <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> instance, or two rational numbers, as arguments to its constructor. (Implemented by Mark Dickinson; rationals added in <a href="https://bugs.python.org/issue5812" class="reference external">bpo-5812</a>, and float/decimal in <a href="https://bugs.python.org/issue8294" class="reference external">bpo-8294</a>.)

  Ordering comparisons (<span class="pre">`<`</span>, <span class="pre">`<=`</span>, <span class="pre">`>`</span>, <span class="pre">`>=`</span>) between fractions and complex numbers now raise a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>. This fixes an oversight, making the <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">Fraction</code></span></a> match the other numeric types.

- New class: <a href="../library/ftplib.html#ftplib.FTP_TLS" class="reference internal" title="ftplib.FTP_TLS"><span class="pre"><code class="sourceCode python">FTP_TLS</code></span></a> in the <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a> module provides secure FTP connections using TLS encapsulation of authentication as well as subsequent control and data transfers. (Contributed by Giampaolo Rodola; <a href="https://bugs.python.org/issue2054" class="reference external">bpo-2054</a>.)

  The <a href="../library/ftplib.html#ftplib.FTP.storbinary" class="reference internal" title="ftplib.FTP.storbinary"><span class="pre"><code class="sourceCode python">storbinary()</code></span></a> method for binary uploads can now restart uploads thanks to an added *rest* parameter (patch by Pablo Mouzo; <a href="https://bugs.python.org/issue6845" class="reference external">bpo-6845</a>.)

- New class decorator: <a href="../library/functools.html#functools.total_ordering" class="reference internal" title="functools.total_ordering"><span class="pre"><code class="sourceCode python">total_ordering()</code></span></a> in the <a href="../library/functools.html#module-functools" class="reference internal" title="functools: Higher-order functions and operations on callable objects."><span class="pre"><code class="sourceCode python">functools</code></span></a> module takes a class that defines an <a href="../reference/datamodel.html#object.__eq__" class="reference internal" title="object.__eq__"><span class="pre"><code class="sourceCode python"><span class="fu">__eq__</span>()</code></span></a> method and one of <a href="../reference/datamodel.html#object.__lt__" class="reference internal" title="object.__lt__"><span class="pre"><code class="sourceCode python"><span class="fu">__lt__</span>()</code></span></a>, <a href="../reference/datamodel.html#object.__le__" class="reference internal" title="object.__le__"><span class="pre"><code class="sourceCode python"><span class="fu">__le__</span>()</code></span></a>, <a href="../reference/datamodel.html#object.__gt__" class="reference internal" title="object.__gt__"><span class="pre"><code class="sourceCode python"><span class="fu">__gt__</span>()</code></span></a>, or <a href="../reference/datamodel.html#object.__ge__" class="reference internal" title="object.__ge__"><span class="pre"><code class="sourceCode python"><span class="fu">__ge__</span>()</code></span></a>, and generates the missing comparison methods. Since the <a href="../reference/datamodel.html#object.__cmp__" class="reference internal" title="object.__cmp__"><span class="pre"><code class="sourceCode python"><span class="fu">__cmp__</span>()</code></span></a> method is being deprecated in Python 3.x, this decorator makes it easier to define ordered classes. (Added by Raymond Hettinger; <a href="https://bugs.python.org/issue5479" class="reference external">bpo-5479</a>.)

  New function: <a href="../library/functools.html#functools.cmp_to_key" class="reference internal" title="functools.cmp_to_key"><span class="pre"><code class="sourceCode python">cmp_to_key()</code></span></a> will take an old-style comparison function that expects two arguments and return a new callable that can be used as the *key* parameter to functions such as <a href="../library/functions.html#sorted" class="reference internal" title="sorted"><span class="pre"><code class="sourceCode python"><span class="bu">sorted</span>()</code></span></a>, <a href="../library/functions.html#min" class="reference internal" title="min"><span class="pre"><code class="sourceCode python"><span class="bu">min</span>()</code></span></a> and <a href="../library/functions.html#max" class="reference internal" title="max"><span class="pre"><code class="sourceCode python"><span class="bu">max</span>()</code></span></a>, etc. The primary intended use is to help with making code compatible with Python 3.x. (Added by Raymond Hettinger.)

- New function: the <a href="../library/gc.html#module-gc" class="reference internal" title="gc: Interface to the cycle-detecting garbage collector."><span class="pre"><code class="sourceCode python">gc</code></span></a> module’s <a href="../library/gc.html#gc.is_tracked" class="reference internal" title="gc.is_tracked"><span class="pre"><code class="sourceCode python">is_tracked()</code></span></a> returns true if a given instance is tracked by the garbage collector, false otherwise. (Contributed by Antoine Pitrou; <a href="https://bugs.python.org/issue4688" class="reference external">bpo-4688</a>.)

- The <a href="../library/gzip.html#module-gzip" class="reference internal" title="gzip: Interfaces for gzip compression and decompression using file objects."><span class="pre"><code class="sourceCode python">gzip</code></span></a> module’s <a href="../library/gzip.html#gzip.GzipFile" class="reference internal" title="gzip.GzipFile"><span class="pre"><code class="sourceCode python">GzipFile</code></span></a> now supports the context management protocol, so you can write <span class="pre">`with`</span>` `<span class="pre">`gzip.GzipFile(...)`</span>` `<span class="pre">`as`</span>` `<span class="pre">`f:`</span> (contributed by Hagen Fürstenau; <a href="https://bugs.python.org/issue3860" class="reference external">bpo-3860</a>), and it now implements the <a href="../library/io.html#io.BufferedIOBase" class="reference internal" title="io.BufferedIOBase"><span class="pre"><code class="sourceCode python">io.BufferedIOBase</code></span></a> ABC, so you can wrap it with <a href="../library/io.html#io.BufferedReader" class="reference internal" title="io.BufferedReader"><span class="pre"><code class="sourceCode python">io.BufferedReader</code></span></a> for faster processing (contributed by Nir Aides; <a href="https://bugs.python.org/issue7471" class="reference external">bpo-7471</a>). It’s also now possible to override the modification time recorded in a gzipped file by providing an optional timestamp to the constructor. (Contributed by Jacques Frechet; <a href="https://bugs.python.org/issue4272" class="reference external">bpo-4272</a>.)

  Files in gzip format can be padded with trailing zero bytes; the <a href="../library/gzip.html#module-gzip" class="reference internal" title="gzip: Interfaces for gzip compression and decompression using file objects."><span class="pre"><code class="sourceCode python">gzip</code></span></a> module will now consume these trailing bytes. (Fixed by Tadek Pietraszek and Brian Curtin; <a href="https://bugs.python.org/issue2846" class="reference external">bpo-2846</a>.)

- New attribute: the <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module now has an <a href="../library/hashlib.html#hashlib.hashlib.algorithms" class="reference internal" title="hashlib.hashlib.algorithms"><span class="pre"><code class="sourceCode python">algorithms</code></span></a> attribute containing a tuple naming the supported algorithms. In Python 2.7, <span class="pre">`hashlib.algorithms`</span> contains <span class="pre">`('md5',`</span>` `<span class="pre">`'sha1',`</span>` `<span class="pre">`'sha224',`</span>` `<span class="pre">`'sha256',`</span>` `<span class="pre">`'sha384',`</span>` `<span class="pre">`'sha512')`</span>. (Contributed by Carl Chenet; <a href="https://bugs.python.org/issue7418" class="reference external">bpo-7418</a>.)

- The default <a href="../library/httplib.html#httplib.HTTPResponse" class="reference internal" title="httplib.HTTPResponse"><span class="pre"><code class="sourceCode python">HTTPResponse</code></span></a> class used by the <a href="../library/httplib.html#module-httplib" class="reference internal" title="httplib: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">httplib</code></span></a> module now supports buffering, resulting in much faster reading of HTTP responses. (Contributed by Kristján Valur Jónsson; <a href="https://bugs.python.org/issue4879" class="reference external">bpo-4879</a>.)

  The <a href="../library/httplib.html#httplib.HTTPConnection" class="reference internal" title="httplib.HTTPConnection"><span class="pre"><code class="sourceCode python">HTTPConnection</code></span></a> and <a href="../library/httplib.html#httplib.HTTPSConnection" class="reference internal" title="httplib.HTTPSConnection"><span class="pre"><code class="sourceCode python">HTTPSConnection</code></span></a> classes now support a *source_address* parameter, a <span class="pre">`(host,`</span>` `<span class="pre">`port)`</span> 2-tuple giving the source address that will be used for the connection. (Contributed by Eldon Ziegler; <a href="https://bugs.python.org/issue3972" class="reference external">bpo-3972</a>.)

- The <span class="pre">`ihooks`</span> module now supports relative imports. Note that <span class="pre">`ihooks`</span> is an older module for customizing imports, superseded by the <a href="../library/imputil.html#module-imputil" class="reference internal" title="imputil: Manage and augment the import process. (deprecated)"><span class="pre"><code class="sourceCode python">imputil</code></span></a> module added in Python 2.0. (Relative import support added by Neil Schemenauer.)

- The <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a> module now supports IPv6 addresses. (Contributed by Derek Morr; <a href="https://bugs.python.org/issue1655" class="reference external">bpo-1655</a>.)

- New function: the <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> module’s <a href="../library/inspect.html#inspect.getcallargs" class="reference internal" title="inspect.getcallargs"><span class="pre"><code class="sourceCode python">getcallargs()</code></span></a> takes a callable and its positional and keyword arguments, and figures out which of the callable’s parameters will receive each argument, returning a dictionary mapping argument names to their values. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> from inspect import getcallargs
      >>> def f(a, b=1, *pos, **named):
      ...     pass
      >>> getcallargs(f, 1, 2, 3)
      {'a': 1, 'b': 2, 'pos': (3,), 'named': {}}
      >>> getcallargs(f, a=2, x=4)
      {'a': 2, 'b': 1, 'pos': (), 'named': {'x': 4}}
      >>> getcallargs(f)
      Traceback (most recent call last):
      ...
      TypeError: f() takes at least 1 argument (0 given)

  </div>

  </div>

  Contributed by George Sakkis; <a href="https://bugs.python.org/issue3135" class="reference external">bpo-3135</a>.

- Updated module: The <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> library has been upgraded to the version shipped with Python 3.1. For 3.1, the I/O library was entirely rewritten in C and is 2 to 20 times faster depending on the task being performed. The original Python version was renamed to the <span class="pre">`_pyio`</span> module.

  One minor resulting change: the <a href="../library/io.html#io.TextIOBase" class="reference internal" title="io.TextIOBase"><span class="pre"><code class="sourceCode python">io.TextIOBase</code></span></a> class now has an <span class="pre">`errors`</span> attribute giving the error setting used for encoding and decoding errors (one of <span class="pre">`'strict'`</span>, <span class="pre">`'replace'`</span>, <span class="pre">`'ignore'`</span>).

  The <a href="../library/io.html#io.FileIO" class="reference internal" title="io.FileIO"><span class="pre"><code class="sourceCode python">io.FileIO</code></span></a> class now raises an <a href="../library/exceptions.html#exceptions.OSError" class="reference internal" title="exceptions.OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a> when passed an invalid file descriptor. (Implemented by Benjamin Peterson; <a href="https://bugs.python.org/issue4991" class="reference external">bpo-4991</a>.) The <a href="../library/io.html#io.IOBase.truncate" class="reference internal" title="io.IOBase.truncate"><span class="pre"><code class="sourceCode python">truncate()</code></span></a> method now preserves the file position; previously it would change the file position to the end of the new file. (Fixed by Pascal Chambon; <a href="https://bugs.python.org/issue6939" class="reference external">bpo-6939</a>.)

- New function: <span class="pre">`itertools.compress(data,`</span>` `<span class="pre">`selectors)`</span> takes two iterators. Elements of *data* are returned if the corresponding value in *selectors* is true:

  <div class="highlight-default notranslate">

  <div class="highlight">

      itertools.compress('ABCDEF', [1,0,1,0,1,1]) =>
        A, C, E, F

  </div>

  </div>

  New function: <span class="pre">`itertools.combinations_with_replacement(iter,`</span>` `<span class="pre">`r)`</span> returns all the possible *r*-length combinations of elements from the iterable *iter*. Unlike <a href="../library/itertools.html#itertools.combinations" class="reference internal" title="itertools.combinations"><span class="pre"><code class="sourceCode python">combinations()</code></span></a>, individual elements can be repeated in the generated combinations:

  <div class="highlight-default notranslate">

  <div class="highlight">

      itertools.combinations_with_replacement('abc', 2) =>
        ('a', 'a'), ('a', 'b'), ('a', 'c'),
        ('b', 'b'), ('b', 'c'), ('c', 'c')

  </div>

  </div>

  Note that elements are treated as unique depending on their position in the input, not their actual values.

  The <a href="../library/itertools.html#itertools.count" class="reference internal" title="itertools.count"><span class="pre"><code class="sourceCode python">itertools.count()</code></span></a> function now has a *step* argument that allows incrementing by values other than 1. <a href="../library/itertools.html#itertools.count" class="reference internal" title="itertools.count"><span class="pre"><code class="sourceCode python">count()</code></span></a> also now allows keyword arguments, and using non-integer values such as floats or <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> instances. (Implemented by Raymond Hettinger; <a href="https://bugs.python.org/issue5032" class="reference external">bpo-5032</a>.)

  <a href="../library/itertools.html#itertools.combinations" class="reference internal" title="itertools.combinations"><span class="pre"><code class="sourceCode python">itertools.combinations()</code></span></a> and <a href="../library/itertools.html#itertools.product" class="reference internal" title="itertools.product"><span class="pre"><code class="sourceCode python">itertools.product()</code></span></a> previously raised <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> for values of *r* larger than the input iterable. This was deemed a specification error, so they now return an empty iterator. (Fixed by Raymond Hettinger; <a href="https://bugs.python.org/issue4816" class="reference external">bpo-4816</a>.)

- Updated module: The <a href="../library/json.html#module-json" class="reference internal" title="json: Encode and decode the JSON format."><span class="pre"><code class="sourceCode python">json</code></span></a> module was upgraded to version 2.0.9 of the simplejson package, which includes a C extension that makes encoding and decoding faster. (Contributed by Bob Ippolito; <a href="https://bugs.python.org/issue4136" class="reference external">bpo-4136</a>.)

  To support the new <a href="../library/collections.html#collections.OrderedDict" class="reference internal" title="collections.OrderedDict"><span class="pre"><code class="sourceCode python">collections.OrderedDict</code></span></a> type, <a href="../library/json.html#json.load" class="reference internal" title="json.load"><span class="pre"><code class="sourceCode python">json.load()</code></span></a> now has an optional *object_pairs_hook* parameter that will be called with any object literal that decodes to a list of pairs. (Contributed by Raymond Hettinger; <a href="https://bugs.python.org/issue5381" class="reference external">bpo-5381</a>.)

- The <a href="../library/mailbox.html#module-mailbox" class="reference internal" title="mailbox: Manipulate mailboxes in various formats"><span class="pre"><code class="sourceCode python">mailbox</code></span></a> module’s <a href="../library/mailbox.html#mailbox.Maildir" class="reference internal" title="mailbox.Maildir"><span class="pre"><code class="sourceCode python">Maildir</code></span></a> class now records the timestamp on the directories it reads, and only re-reads them if the modification time has subsequently changed. This improves performance by avoiding unneeded directory scans. (Fixed by A.M. Kuchling and Antoine Pitrou; <a href="https://bugs.python.org/issue1607951" class="reference external">bpo-1607951</a>, <a href="https://bugs.python.org/issue6896" class="reference external">bpo-6896</a>.)

- New functions: the <a href="../library/math.html#module-math" class="reference internal" title="math: Mathematical functions (sin() etc.)."><span class="pre"><code class="sourceCode python">math</code></span></a> module gained <a href="../library/math.html#math.erf" class="reference internal" title="math.erf"><span class="pre"><code class="sourceCode python">erf()</code></span></a> and <a href="../library/math.html#math.erfc" class="reference internal" title="math.erfc"><span class="pre"><code class="sourceCode python">erfc()</code></span></a> for the error function and the complementary error function, <a href="../library/math.html#math.expm1" class="reference internal" title="math.expm1"><span class="pre"><code class="sourceCode python">expm1()</code></span></a> which computes <span class="pre">`e**x`</span>` `<span class="pre">`-`</span>` `<span class="pre">`1`</span> with more precision than using <a href="../library/math.html#math.exp" class="reference internal" title="math.exp"><span class="pre"><code class="sourceCode python">exp()</code></span></a> and subtracting 1, <a href="../library/math.html#math.gamma" class="reference internal" title="math.gamma"><span class="pre"><code class="sourceCode python">gamma()</code></span></a> for the Gamma function, and <a href="../library/math.html#math.lgamma" class="reference internal" title="math.lgamma"><span class="pre"><code class="sourceCode python">lgamma()</code></span></a> for the natural log of the Gamma function. (Contributed by Mark Dickinson and nirinA raseliarison; <a href="https://bugs.python.org/issue3366" class="reference external">bpo-3366</a>.)

- The <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based &quot;threading&quot; interface."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> module’s <span class="pre">`Manager*`</span> classes can now be passed a callable that will be called whenever a subprocess is started, along with a set of arguments that will be passed to the callable. (Contributed by lekma; <a href="https://bugs.python.org/issue5585" class="reference external">bpo-5585</a>.)

  The <span class="pre">`Pool`</span> class, which controls a pool of worker processes, now has an optional *maxtasksperchild* parameter. Worker processes will perform the specified number of tasks and then exit, causing the <span class="pre">`Pool`</span> to start a new worker. This is useful if tasks may leak memory or other resources, or if some tasks will cause the worker to become very large. (Contributed by Charles Cazabon; <a href="https://bugs.python.org/issue6963" class="reference external">bpo-6963</a>.)

- The <a href="../library/nntplib.html#module-nntplib" class="reference internal" title="nntplib: NNTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">nntplib</code></span></a> module now supports IPv6 addresses. (Contributed by Derek Morr; <a href="https://bugs.python.org/issue1664" class="reference external">bpo-1664</a>.)

- New functions: the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module wraps the following POSIX system calls: <a href="../library/os.html#os.getresgid" class="reference internal" title="os.getresgid"><span class="pre"><code class="sourceCode python">getresgid()</code></span></a> and <a href="../library/os.html#os.getresuid" class="reference internal" title="os.getresuid"><span class="pre"><code class="sourceCode python">getresuid()</code></span></a>, which return the real, effective, and saved GIDs and UIDs; <a href="../library/os.html#os.setresgid" class="reference internal" title="os.setresgid"><span class="pre"><code class="sourceCode python">setresgid()</code></span></a> and <a href="../library/os.html#os.setresuid" class="reference internal" title="os.setresuid"><span class="pre"><code class="sourceCode python">setresuid()</code></span></a>, which set real, effective, and saved GIDs and UIDs to new values; <a href="../library/os.html#os.initgroups" class="reference internal" title="os.initgroups"><span class="pre"><code class="sourceCode python">initgroups()</code></span></a>, which initialize the group access list for the current process. (GID/UID functions contributed by Travis H.; <a href="https://bugs.python.org/issue6508" class="reference external">bpo-6508</a>. Support for initgroups added by Jean-Paul Calderone; <a href="https://bugs.python.org/issue7333" class="reference external">bpo-7333</a>.)

  The <a href="../library/os.html#os.fork" class="reference internal" title="os.fork"><span class="pre"><code class="sourceCode python">os.fork()</code></span></a> function now re-initializes the import lock in the child process; this fixes problems on Solaris when <a href="../library/os.html#os.fork" class="reference internal" title="os.fork"><span class="pre"><code class="sourceCode python">fork()</code></span></a> is called from a thread. (Fixed by Zsolt Cserna; <a href="https://bugs.python.org/issue7242" class="reference external">bpo-7242</a>.)

- In the <a href="../library/os.path.html#module-os.path" class="reference internal" title="os.path: Operations on pathnames."><span class="pre"><code class="sourceCode python">os.path</code></span></a> module, the <a href="../library/os.path.html#os.path.normpath" class="reference internal" title="os.path.normpath"><span class="pre"><code class="sourceCode python">normpath()</code></span></a> and <a href="../library/os.path.html#os.path.abspath" class="reference internal" title="os.path.abspath"><span class="pre"><code class="sourceCode python">abspath()</code></span></a> functions now preserve Unicode; if their input path is a Unicode string, the return value is also a Unicode string. (<a href="../library/os.path.html#os.path.normpath" class="reference internal" title="os.path.normpath"><span class="pre"><code class="sourceCode python">normpath()</code></span></a> fixed by Matt Giuca in <a href="https://bugs.python.org/issue5827" class="reference external">bpo-5827</a>; <a href="../library/os.path.html#os.path.abspath" class="reference internal" title="os.path.abspath"><span class="pre"><code class="sourceCode python">abspath()</code></span></a> fixed by Ezio Melotti in <a href="https://bugs.python.org/issue3426" class="reference external">bpo-3426</a>.)

- The <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module now has help for the various symbols that Python uses. You can now do <span class="pre">`help('<<')`</span> or <span class="pre">`help('@')`</span>, for example. (Contributed by David Laban; <a href="https://bugs.python.org/issue4739" class="reference external">bpo-4739</a>.)

- The <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module’s <a href="../library/re.html#re.split" class="reference internal" title="re.split"><span class="pre"><code class="sourceCode python">split()</code></span></a>, <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">sub()</code></span></a>, and <a href="../library/re.html#re.subn" class="reference internal" title="re.subn"><span class="pre"><code class="sourceCode python">subn()</code></span></a> now accept an optional *flags* argument, for consistency with the other functions in the module. (Added by Gregory P. Smith.)

- New function: <a href="../library/runpy.html#runpy.run_path" class="reference internal" title="runpy.run_path"><span class="pre"><code class="sourceCode python">run_path()</code></span></a> in the <a href="../library/runpy.html#module-runpy" class="reference internal" title="runpy: Locate and run Python modules without importing them first."><span class="pre"><code class="sourceCode python">runpy</code></span></a> module will execute the code at a provided *path* argument. *path* can be the path of a Python source file (<span class="pre">`example.py`</span>), a compiled bytecode file (<span class="pre">`example.pyc`</span>), a directory (<span class="pre">`./package/`</span>), or a zip archive (<span class="pre">`example.zip`</span>). If a directory or zip path is provided, it will be added to the front of <span class="pre">`sys.path`</span> and the module <a href="../library/__main__.html#module-__main__" class="reference internal" title="__main__: The environment where the top-level script is run."><span class="pre"><code class="sourceCode python">__main__</code></span></a> will be imported. It’s expected that the directory or zip contains a <span class="pre">`__main__.py`</span>; if it doesn’t, some other <span class="pre">`__main__.py`</span> might be imported from a location later in <span class="pre">`sys.path`</span>. This makes more of the machinery of <a href="../library/runpy.html#module-runpy" class="reference internal" title="runpy: Locate and run Python modules without importing them first."><span class="pre"><code class="sourceCode python">runpy</code></span></a> available to scripts that want to mimic the way Python’s command line processes an explicit path name. (Added by Nick Coghlan; <a href="https://bugs.python.org/issue6816" class="reference external">bpo-6816</a>.)

- New function: in the <a href="../library/shutil.html#module-shutil" class="reference internal" title="shutil: High-level file operations, including copying."><span class="pre"><code class="sourceCode python">shutil</code></span></a> module, <a href="../library/shutil.html#shutil.make_archive" class="reference internal" title="shutil.make_archive"><span class="pre"><code class="sourceCode python">make_archive()</code></span></a> takes a filename, archive type (zip or tar-format), and a directory path, and creates an archive containing the directory’s contents. (Added by Tarek Ziadé.)

  <a href="../library/shutil.html#module-shutil" class="reference internal" title="shutil: High-level file operations, including copying."><span class="pre"><code class="sourceCode python">shutil</code></span></a>’s <a href="../library/shutil.html#shutil.copyfile" class="reference internal" title="shutil.copyfile"><span class="pre"><code class="sourceCode python">copyfile()</code></span></a> and <a href="../library/shutil.html#shutil.copytree" class="reference internal" title="shutil.copytree"><span class="pre"><code class="sourceCode python">copytree()</code></span></a> functions now raise a <span class="pre">`SpecialFileError`</span> exception when asked to copy a named pipe. Previously the code would treat named pipes like a regular file by opening them for reading, and this would block indefinitely. (Fixed by Antoine Pitrou; <a href="https://bugs.python.org/issue3002" class="reference external">bpo-3002</a>.)

- The <a href="../library/signal.html#module-signal" class="reference internal" title="signal: Set handlers for asynchronous events."><span class="pre"><code class="sourceCode python">signal</code></span></a> module no longer re-installs the signal handler unless this is truly necessary, which fixes a bug that could make it impossible to catch the EINTR signal robustly. (Fixed by Charles-Francois Natali; <a href="https://bugs.python.org/issue8354" class="reference external">bpo-8354</a>.)

- New functions: in the <a href="../library/site.html#module-site" class="reference internal" title="site: Module responsible for site-specific configuration."><span class="pre"><code class="sourceCode python">site</code></span></a> module, three new functions return various site- and user-specific paths. <a href="../library/site.html#site.getsitepackages" class="reference internal" title="site.getsitepackages"><span class="pre"><code class="sourceCode python">getsitepackages()</code></span></a> returns a list containing all global site-packages directories, <a href="../library/site.html#site.getusersitepackages" class="reference internal" title="site.getusersitepackages"><span class="pre"><code class="sourceCode python">getusersitepackages()</code></span></a> returns the path of the user’s site-packages directory, and <a href="../library/site.html#site.getuserbase" class="reference internal" title="site.getuserbase"><span class="pre"><code class="sourceCode python">getuserbase()</code></span></a> returns the value of the <span id="index-9" class="target"></span><span class="pre">`USER_BASE`</span> environment variable, giving the path to a directory that can be used to store data. (Contributed by Tarek Ziadé; <a href="https://bugs.python.org/issue6693" class="reference external">bpo-6693</a>.)

  The <a href="../library/site.html#module-site" class="reference internal" title="site: Module responsible for site-specific configuration."><span class="pre"><code class="sourceCode python">site</code></span></a> module now reports exceptions occurring when the <span class="pre">`sitecustomize`</span> module is imported, and will no longer catch and swallow the <a href="../library/exceptions.html#exceptions.KeyboardInterrupt" class="reference internal" title="exceptions.KeyboardInterrupt"><span class="pre"><code class="sourceCode python"><span class="pp">KeyboardInterrupt</span></code></span></a> exception. (Fixed by Victor Stinner; <a href="https://bugs.python.org/issue3137" class="reference external">bpo-3137</a>.)

- The <a href="../library/socket.html#socket.create_connection" class="reference internal" title="socket.create_connection"><span class="pre"><code class="sourceCode python">create_connection()</code></span></a> function gained a *source_address* parameter, a <span class="pre">`(host,`</span>` `<span class="pre">`port)`</span> 2-tuple giving the source address that will be used for the connection. (Contributed by Eldon Ziegler; <a href="https://bugs.python.org/issue3972" class="reference external">bpo-3972</a>.)

  The <a href="../library/socket.html#socket.socket.recv_into" class="reference internal" title="socket.socket.recv_into"><span class="pre"><code class="sourceCode python">recv_into()</code></span></a> and <a href="../library/socket.html#socket.socket.recvfrom_into" class="reference internal" title="socket.socket.recvfrom_into"><span class="pre"><code class="sourceCode python">recvfrom_into()</code></span></a> methods will now write into objects that support the buffer API, most usefully the <a href="../library/functions.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> and <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> objects. (Implemented by Antoine Pitrou; <a href="https://bugs.python.org/issue8104" class="reference external">bpo-8104</a>.)

- The <a href="../library/socketserver.html#module-SocketServer" class="reference internal" title="SocketServer: A framework for network servers."><span class="pre"><code class="sourceCode python">SocketServer</code></span></a> module’s <a href="../library/socketserver.html#SocketServer.TCPServer" class="reference internal" title="SocketServer.TCPServer"><span class="pre"><code class="sourceCode python">TCPServer</code></span></a> class now supports socket timeouts and disabling the Nagle algorithm. The <span class="pre">`disable_nagle_algorithm`</span> class attribute defaults to <span class="pre">`False`</span>; if overridden to be true, new request connections will have the TCP_NODELAY option set to prevent buffering many small sends into a single TCP packet. The <a href="../library/socketserver.html#SocketServer.BaseServer.timeout" class="reference internal" title="SocketServer.BaseServer.timeout"><span class="pre"><code class="sourceCode python">timeout</code></span></a> class attribute can hold a timeout in seconds that will be applied to the request socket; if no request is received within that time, <a href="../library/socketserver.html#SocketServer.BaseServer.handle_timeout" class="reference internal" title="SocketServer.BaseServer.handle_timeout"><span class="pre"><code class="sourceCode python">handle_timeout()</code></span></a> will be called and <a href="../library/socketserver.html#SocketServer.BaseServer.handle_request" class="reference internal" title="SocketServer.BaseServer.handle_request"><span class="pre"><code class="sourceCode python">handle_request()</code></span></a> will return. (Contributed by Kristján Valur Jónsson; <a href="https://bugs.python.org/issue6192" class="reference external">bpo-6192</a> and <a href="https://bugs.python.org/issue6267" class="reference external">bpo-6267</a>.)

- Updated module: the <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> module has been updated to version 2.6.0 of the <a href="https://github.com/ghaering/pysqlite" class="reference external">pysqlite package</a>. Version 2.6.0 includes a number of bugfixes, and adds the ability to load SQLite extensions from shared libraries. Call the <span class="pre">`enable_load_extension(True)`</span> method to enable extensions, and then call <a href="../library/sqlite3.html#sqlite3.Connection.load_extension" class="reference internal" title="sqlite3.Connection.load_extension"><span class="pre"><code class="sourceCode python">load_extension()</code></span></a> to load a particular shared library. (Updated by Gerhard Häring.)

- The <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module’s <span class="pre">`SSLSocket`</span> objects now support the buffer API, which fixed a test suite failure (fix by Antoine Pitrou; <a href="https://bugs.python.org/issue7133" class="reference external">bpo-7133</a>) and automatically set OpenSSL’s <span class="pre">`SSL_MODE_AUTO_RETRY`</span>, which will prevent an error code being returned from <span class="pre">`recv()`</span> operations that trigger an SSL renegotiation (fix by Antoine Pitrou; <a href="https://bugs.python.org/issue8222" class="reference external">bpo-8222</a>).

  The <a href="../library/ssl.html#ssl.wrap_socket" class="reference internal" title="ssl.wrap_socket"><span class="pre"><code class="sourceCode python">ssl.wrap_socket()</code></span></a> constructor function now takes a *ciphers* argument that’s a string listing the encryption algorithms to be allowed; the format of the string is described <a href="https://www.openssl.org/docs/manmaster/man1/ciphers.html#CIPHER-LIST-FORMAT" class="reference external">in the OpenSSL documentation</a>. (Added by Antoine Pitrou; <a href="https://bugs.python.org/issue8322" class="reference external">bpo-8322</a>.)

  Another change makes the extension load all of OpenSSL’s ciphers and digest algorithms so that they’re all available. Some SSL certificates couldn’t be verified, reporting an “unknown algorithm” error. (Reported by Beda Kosata, and fixed by Antoine Pitrou; <a href="https://bugs.python.org/issue8484" class="reference external">bpo-8484</a>.)

  The version of OpenSSL being used is now available as the module attributes <a href="../library/ssl.html#ssl.OPENSSL_VERSION" class="reference internal" title="ssl.OPENSSL_VERSION"><span class="pre"><code class="sourceCode python">ssl.OPENSSL_VERSION</code></span></a> (a string), <a href="../library/ssl.html#ssl.OPENSSL_VERSION_INFO" class="reference internal" title="ssl.OPENSSL_VERSION_INFO"><span class="pre"><code class="sourceCode python">ssl.OPENSSL_VERSION_INFO</code></span></a> (a 5-tuple), and <a href="../library/ssl.html#ssl.OPENSSL_VERSION_NUMBER" class="reference internal" title="ssl.OPENSSL_VERSION_NUMBER"><span class="pre"><code class="sourceCode python">ssl.OPENSSL_VERSION_NUMBER</code></span></a> (an integer). (Added by Antoine Pitrou; <a href="https://bugs.python.org/issue8321" class="reference external">bpo-8321</a>.)

- The <a href="../library/struct.html#module-struct" class="reference internal" title="struct: Interpret strings as packed binary data."><span class="pre"><code class="sourceCode python">struct</code></span></a> module will no longer silently ignore overflow errors when a value is too large for a particular integer format code (one of <span class="pre">`bBhHiIlLqQ`</span>); it now always raises a <a href="../library/struct.html#struct.error" class="reference internal" title="struct.error"><span class="pre"><code class="sourceCode python">struct.error</code></span></a> exception. (Changed by Mark Dickinson; <a href="https://bugs.python.org/issue1523" class="reference external">bpo-1523</a>.) The <a href="../library/struct.html#struct.pack" class="reference internal" title="struct.pack"><span class="pre"><code class="sourceCode python">pack()</code></span></a> function will also attempt to use <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> to convert and pack non-integers before trying the <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="fu">__int__</span>()</code></span></a> method or reporting an error. (Changed by Mark Dickinson; <a href="https://bugs.python.org/issue8300" class="reference external">bpo-8300</a>.)

- New function: the <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module’s <a href="../library/subprocess.html#subprocess.check_output" class="reference internal" title="subprocess.check_output"><span class="pre"><code class="sourceCode python">check_output()</code></span></a> runs a command with a specified set of arguments and returns the command’s output as a string when the command runs without error, or raises a <a href="../library/subprocess.html#subprocess.CalledProcessError" class="reference internal" title="subprocess.CalledProcessError"><span class="pre"><code class="sourceCode python">CalledProcessError</code></span></a> exception otherwise.

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> subprocess.check_output(['df', '-h', '.'])
      'Filesystem     Size   Used  Avail Capacity  Mounted on\n
      /dev/disk0s2    52G    49G   3.0G    94%    /\n'

      >>> subprocess.check_output(['df', '-h', '/bogus'])
        ...
      subprocess.CalledProcessError: Command '['df', '-h', '/bogus']' returned non-zero exit status 1

  </div>

  </div>

  (Contributed by Gregory P. Smith.)

  The <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> module will now retry its internal system calls on receiving an <span class="pre">`EINTR`</span> signal. (Reported by several people; final patch by Gregory P. Smith in <a href="https://bugs.python.org/issue1068268" class="reference external">bpo-1068268</a>.)

- New function: <a href="../library/symtable.html#symtable.Symbol.is_declared_global" class="reference internal" title="symtable.Symbol.is_declared_global"><span class="pre"><code class="sourceCode python">is_declared_global()</code></span></a> in the <a href="../library/symtable.html#module-symtable" class="reference internal" title="symtable: Interface to the compiler&#39;s internal symbol tables."><span class="pre"><code class="sourceCode python">symtable</code></span></a> module returns true for variables that are explicitly declared to be global, false for ones that are implicitly global. (Contributed by Jeremy Hylton.)

- The <a href="../library/syslog.html#module-syslog" class="reference internal" title="syslog: An interface to the Unix syslog library routines. (Unix)"><span class="pre"><code class="sourceCode python">syslog</code></span></a> module will now use the value of <span class="pre">`sys.argv[0]`</span> as the identifier instead of the previous default value of <span class="pre">`'python'`</span>. (Changed by Sean Reifschneider; <a href="https://bugs.python.org/issue8451" class="reference external">bpo-8451</a>.)

- The <span class="pre">`sys.version_info`</span> value is now a named tuple, with attributes named <span class="pre">`major`</span>, <span class="pre">`minor`</span>, <span class="pre">`micro`</span>, <span class="pre">`releaselevel`</span>, and <span class="pre">`serial`</span>. (Contributed by Ross Light; <a href="https://bugs.python.org/issue4285" class="reference external">bpo-4285</a>.)

  <a href="../library/sys.html#sys.getwindowsversion" class="reference internal" title="sys.getwindowsversion"><span class="pre"><code class="sourceCode python">sys.getwindowsversion()</code></span></a> also returns a named tuple, with attributes named <span class="pre">`major`</span>, <span class="pre">`minor`</span>, <span class="pre">`build`</span>, <a href="../library/platform.html#module-platform" class="reference internal" title="platform: Retrieves as much platform identifying data as possible."><span class="pre"><code class="sourceCode python">platform</code></span></a>, <span class="pre">`service_pack`</span>, <span class="pre">`service_pack_major`</span>, <span class="pre">`service_pack_minor`</span>, <span class="pre">`suite_mask`</span>, and <span class="pre">`product_type`</span>. (Contributed by Brian Curtin; <a href="https://bugs.python.org/issue7766" class="reference external">bpo-7766</a>.)

- The <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module’s default error handling has changed, to no longer suppress fatal errors. The default error level was previously 0, which meant that errors would only result in a message being written to the debug log, but because the debug log is not activated by default, these errors go unnoticed. The default error level is now 1, which raises an exception if there’s an error. (Changed by Lars Gustäbel; <a href="https://bugs.python.org/issue7357" class="reference external">bpo-7357</a>.)

  <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> now supports filtering the <a href="../library/tarfile.html#tarfile.TarInfo" class="reference internal" title="tarfile.TarInfo"><span class="pre"><code class="sourceCode python">TarInfo</code></span></a> objects being added to a tar file. When you call <a href="../library/tarfile.html#tarfile.TarFile.add" class="reference internal" title="tarfile.TarFile.add"><span class="pre"><code class="sourceCode python">add()</code></span></a>, you may supply an optional *filter* argument that’s a callable. The *filter* callable will be passed the <a href="../library/tarfile.html#tarfile.TarInfo" class="reference internal" title="tarfile.TarInfo"><span class="pre"><code class="sourceCode python">TarInfo</code></span></a> for every file being added, and can modify and return it. If the callable returns <span class="pre">`None`</span>, the file will be excluded from the resulting archive. This is more powerful than the existing *exclude* argument, which has therefore been deprecated. (Added by Lars Gustäbel; <a href="https://bugs.python.org/issue6856" class="reference external">bpo-6856</a>.) The <a href="../library/tarfile.html#tarfile.TarFile" class="reference internal" title="tarfile.TarFile"><span class="pre"><code class="sourceCode python">TarFile</code></span></a> class also now supports the context management protocol. (Added by Lars Gustäbel; <a href="https://bugs.python.org/issue7232" class="reference external">bpo-7232</a>.)

- The <a href="../library/threading.html#threading.Event.wait" class="reference internal" title="threading.Event.wait"><span class="pre"><code class="sourceCode python">wait()</code></span></a> method of the <a href="../library/threading.html#threading.Event" class="reference internal" title="threading.Event"><span class="pre"><code class="sourceCode python">threading.Event</code></span></a> class now returns the internal flag on exit. This means the method will usually return true because <a href="../library/threading.html#threading.Event.wait" class="reference internal" title="threading.Event.wait"><span class="pre"><code class="sourceCode python">wait()</code></span></a> is supposed to block until the internal flag becomes true. The return value will only be false if a timeout was provided and the operation timed out. (Contributed by Tim Lesher; <a href="https://bugs.python.org/issue1674032" class="reference external">bpo-1674032</a>.)

- The Unicode database provided by the <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a> module is now used internally to determine which characters are numeric, whitespace, or represent line breaks. The database also includes information from the <span class="pre">`Unihan.txt`</span> data file (patch by Anders Chrigström and Amaury Forgeot d’Arc; <a href="https://bugs.python.org/issue1571184" class="reference external">bpo-1571184</a>) and has been updated to version 5.2.0 (updated by Florent Xicluna; <a href="https://bugs.python.org/issue8024" class="reference external">bpo-8024</a>).

- The <a href="../library/urlparse.html#module-urlparse" class="reference internal" title="urlparse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urlparse</code></span></a> module’s <a href="../library/urlparse.html#urlparse.urlsplit" class="reference internal" title="urlparse.urlsplit"><span class="pre"><code class="sourceCode python">urlsplit()</code></span></a> now handles unknown URL schemes in a fashion compliant with <span id="index-10" class="target"></span><a href="https://tools.ietf.org/html/rfc3986.html" class="rfc reference external"><strong>RFC 3986</strong></a>: if the URL is of the form <span class="pre">`"<something>://..."`</span>, the text before the <span class="pre">`://`</span> is treated as the scheme, even if it’s a made-up scheme that the module doesn’t know about. This change may break code that worked around the old behaviour. For example, Python 2.6.4 or 2.5 will return the following:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import urlparse
      >>> urlparse.urlsplit('invented://host/filename?query')
      ('invented', '', '//host/filename?query', '', '')

  </div>

  </div>

  Python 2.7 (and Python 2.6.5) will return:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import urlparse
      >>> urlparse.urlsplit('invented://host/filename?query')
      ('invented', 'host', '/filename?query', '', '')

  </div>

  </div>

  (Python 2.7 actually produces slightly different output, since it returns a named tuple instead of a standard tuple.)

  The <a href="../library/urlparse.html#module-urlparse" class="reference internal" title="urlparse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urlparse</code></span></a> module also supports IPv6 literal addresses as defined by <span id="index-11" class="target"></span><a href="https://tools.ietf.org/html/rfc2732.html" class="rfc reference external"><strong>RFC 2732</strong></a> (contributed by Senthil Kumaran; <a href="https://bugs.python.org/issue2987" class="reference external">bpo-2987</a>).

  <div class="highlight-default notranslate">

  <div class="highlight">

      >>> urlparse.urlparse('http://[1080::8:800:200C:417A]/foo')
      ParseResult(scheme='http', netloc='[1080::8:800:200C:417A]',
                  path='/foo', params='', query='', fragment='')

  </div>

  </div>

- New class: the <a href="../library/weakref.html#weakref.WeakSet" class="reference internal" title="weakref.WeakSet"><span class="pre"><code class="sourceCode python">WeakSet</code></span></a> class in the <a href="../library/weakref.html#module-weakref" class="reference internal" title="weakref: Support for weak references and weak dictionaries."><span class="pre"><code class="sourceCode python">weakref</code></span></a> module is a set that only holds weak references to its elements; elements will be removed once there are no references pointing to them. (Originally implemented in Python 3.x by Raymond Hettinger, and backported to 2.7 by Michael Foord.)

- The ElementTree library, <span class="pre">`xml.etree`</span>, no longer escapes ampersands and angle brackets when outputting an XML processing instruction (which looks like <span class="pre">`<?xml-stylesheet`</span>` `<span class="pre">`href="#style1"?>`</span>) or comment (which looks like <span class="pre">`<!--`</span>` `<span class="pre">`comment`</span>` `<span class="pre">`-->`</span>). (Patch by Neil Muller; <a href="https://bugs.python.org/issue2746" class="reference external">bpo-2746</a>.)

- The XML-RPC client and server, provided by the <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a> and <a href="../library/simplexmlrpcserver.html#module-SimpleXMLRPCServer" class="reference internal" title="SimpleXMLRPCServer: Basic XML-RPC server implementation."><span class="pre"><code class="sourceCode python">SimpleXMLRPCServer</code></span></a> modules, have improved performance by supporting HTTP/1.1 keep-alive and by optionally using gzip encoding to compress the XML being exchanged. The gzip compression is controlled by the <span class="pre">`encode_threshold`</span> attribute of <span class="pre">`SimpleXMLRPCRequestHandler`</span>, which contains a size in bytes; responses larger than this will be compressed. (Contributed by Kristján Valur Jónsson; <a href="https://bugs.python.org/issue6267" class="reference external">bpo-6267</a>.)

- The <a href="../library/zipfile.html#module-zipfile" class="reference internal" title="zipfile: Read and write ZIP-format archive files."><span class="pre"><code class="sourceCode python">zipfile</code></span></a> module’s <a href="../library/zipfile.html#zipfile.ZipFile" class="reference internal" title="zipfile.ZipFile"><span class="pre"><code class="sourceCode python">ZipFile</code></span></a> now supports the context management protocol, so you can write <span class="pre">`with`</span>` `<span class="pre">`zipfile.ZipFile(...)`</span>` `<span class="pre">`as`</span>` `<span class="pre">`f:`</span>. (Contributed by Brian Curtin; <a href="https://bugs.python.org/issue5511" class="reference external">bpo-5511</a>.)

  <a href="../library/zipfile.html#module-zipfile" class="reference internal" title="zipfile: Read and write ZIP-format archive files."><span class="pre"><code class="sourceCode python">zipfile</code></span></a> now also supports archiving empty directories and extracts them correctly. (Fixed by Kuba Wieczorek; <a href="https://bugs.python.org/issue4710" class="reference external">bpo-4710</a>.) Reading files out of an archive is faster, and interleaving <a href="../library/zipfile.html#zipfile.ZipFile.read" class="reference internal" title="zipfile.ZipFile.read"><span class="pre"><code class="sourceCode python">read()</code></span></a> and <span class="pre">`readline()`</span> now works correctly. (Contributed by Nir Aides; <a href="https://bugs.python.org/issue7610" class="reference external">bpo-7610</a>.)

  The <a href="../library/zipfile.html#zipfile.is_zipfile" class="reference internal" title="zipfile.is_zipfile"><span class="pre"><code class="sourceCode python">is_zipfile()</code></span></a> function now accepts a file object, in addition to the path names accepted in earlier versions. (Contributed by Gabriel Genellina; <a href="https://bugs.python.org/issue4756" class="reference external">bpo-4756</a>.)

  The <a href="../library/zipfile.html#zipfile.ZipFile.writestr" class="reference internal" title="zipfile.ZipFile.writestr"><span class="pre"><code class="sourceCode python">writestr()</code></span></a> method now has an optional *compress_type* parameter that lets you override the default compression method specified in the <a href="../library/zipfile.html#zipfile.ZipFile" class="reference internal" title="zipfile.ZipFile"><span class="pre"><code class="sourceCode python">ZipFile</code></span></a> constructor. (Contributed by Ronald Oussoren; <a href="https://bugs.python.org/issue6003" class="reference external">bpo-6003</a>.)

<div id="new-module-importlib" class="section">

<span id="importlib-section"></span>

### New module: importlib<a href="#new-module-importlib" class="headerlink" title="Permalink to this headline">¶</a>

Python 3.1 includes the <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: Convenience wrappers for __import__"><span class="pre"><code class="sourceCode python">importlib</code></span></a> package, a re-implementation of the logic underlying Python’s <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statement. <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: Convenience wrappers for __import__"><span class="pre"><code class="sourceCode python">importlib</code></span></a> is useful for implementors of Python interpreters and to users who wish to write new importers that can participate in the import process. Python 2.7 doesn’t contain the complete <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: Convenience wrappers for __import__"><span class="pre"><code class="sourceCode python">importlib</code></span></a> package, but instead has a tiny subset that contains a single function, <a href="../library/importlib.html#importlib.import_module" class="reference internal" title="importlib.import_module"><span class="pre"><code class="sourceCode python">import_module()</code></span></a>.

<span class="pre">`import_module(name,`</span>` `<span class="pre">`package=None)`</span> imports a module. *name* is a string containing the module or package’s name. It’s possible to do relative imports by providing a string that begins with a <span class="pre">`.`</span> character, such as <span class="pre">`..utils.errors`</span>. For relative imports, the *package* argument must be provided and is the name of the package that will be used as the anchor for the relative import. <a href="../library/importlib.html#importlib.import_module" class="reference internal" title="importlib.import_module"><span class="pre"><code class="sourceCode python">import_module()</code></span></a> both inserts the imported module into <span class="pre">`sys.modules`</span> and returns the module object.

Here are some examples:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> from importlib import import_module
    >>> anydbm = import_module('anydbm')  # Standard absolute import
    >>> anydbm
    <module 'anydbm' from '/p/python/Lib/anydbm.py'>
    >>> # Relative import
    >>> file_util = import_module('..file_util', 'distutils.command')
    >>> file_util
    <module 'distutils.file_util' from '/python/Lib/distutils/file_util.pyc'>

</div>

</div>

<a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: Convenience wrappers for __import__"><span class="pre"><code class="sourceCode python">importlib</code></span></a> was implemented by Brett Cannon and introduced in Python 3.1.

</div>

<div id="new-module-sysconfig" class="section">

### New module: sysconfig<a href="#new-module-sysconfig" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/sysconfig.html#module-sysconfig" class="reference internal" title="sysconfig: Python&#39;s configuration information"><span class="pre"><code class="sourceCode python">sysconfig</code></span></a> module has been pulled out of the Distutils package, becoming a new top-level module in its own right. <a href="../library/sysconfig.html#module-sysconfig" class="reference internal" title="sysconfig: Python&#39;s configuration information"><span class="pre"><code class="sourceCode python">sysconfig</code></span></a> provides functions for getting information about Python’s build process: compiler switches, installation paths, the platform name, and whether Python is running from its source directory.

Some of the functions in the module are:

- <a href="../library/sysconfig.html#sysconfig.get_config_var" class="reference internal" title="sysconfig.get_config_var"><span class="pre"><code class="sourceCode python">get_config_var()</code></span></a> returns variables from Python’s Makefile and the <span class="pre">`pyconfig.h`</span> file.

- <a href="../library/sysconfig.html#sysconfig.get_config_vars" class="reference internal" title="sysconfig.get_config_vars"><span class="pre"><code class="sourceCode python">get_config_vars()</code></span></a> returns a dictionary containing all of the configuration variables.

- <a href="../library/sysconfig.html#sysconfig.get_path" class="reference internal" title="sysconfig.get_path"><span class="pre"><code class="sourceCode python">get_path()</code></span></a> returns the configured path for a particular type of module: the standard library, site-specific modules, platform-specific modules, etc.

- <a href="../library/sysconfig.html#sysconfig.is_python_build" class="reference internal" title="sysconfig.is_python_build"><span class="pre"><code class="sourceCode python">is_python_build()</code></span></a> returns true if you’re running a binary from a Python source tree, and false otherwise.

Consult the <a href="../library/sysconfig.html#module-sysconfig" class="reference internal" title="sysconfig: Python&#39;s configuration information"><span class="pre"><code class="sourceCode python">sysconfig</code></span></a> documentation for more details and for a complete list of functions.

The Distutils package and <a href="../library/sysconfig.html#module-sysconfig" class="reference internal" title="sysconfig: Python&#39;s configuration information"><span class="pre"><code class="sourceCode python">sysconfig</code></span></a> are now maintained by Tarek Ziadé, who has also started a Distutils2 package (source repository at <a href="https://hg.python.org/distutils2/" class="reference external">https://hg.python.org/distutils2/</a>) for developing a next-generation version of Distutils.

</div>

<div id="ttk-themed-widgets-for-tk" class="section">

### ttk: Themed Widgets for Tk<a href="#ttk-themed-widgets-for-tk" class="headerlink" title="Permalink to this headline">¶</a>

Tcl/Tk 8.5 includes a set of themed widgets that re-implement basic Tk widgets but have a more customizable appearance and can therefore more closely resemble the native platform’s widgets. This widget set was originally called Tile, but was renamed to Ttk (for “themed Tk”) on being added to Tcl/Tck release 8.5.

To learn more, read the <a href="../library/ttk.html#module-ttk" class="reference internal" title="ttk: Tk themed widget set"><span class="pre"><code class="sourceCode python">ttk</code></span></a> module documentation. You may also wish to read the Tcl/Tk manual page describing the Ttk theme engine, available at <a href="https://www.tcl.tk/man/tcl8.5/TkCmd/ttk_intro.htm" class="reference external">https://www.tcl.tk/man/tcl8.5/TkCmd/ttk_intro.htm</a>. Some screenshots of the Python/Ttk code in use are at <a href="https://code.google.com/archive/p/python-ttk/wikis/Screenshots.wiki" class="reference external">https://code.google.com/archive/p/python-ttk/wikis/Screenshots.wiki</a>.

The <a href="../library/ttk.html#module-ttk" class="reference internal" title="ttk: Tk themed widget set"><span class="pre"><code class="sourceCode python">ttk</code></span></a> module was written by Guilherme Polo and added in <a href="https://bugs.python.org/issue2983" class="reference external">bpo-2983</a>. An alternate version called <span class="pre">`Tile.py`</span>, written by Martin Franklin and maintained by Kevin Walzer, was proposed for inclusion in <a href="https://bugs.python.org/issue2618" class="reference external">bpo-2618</a>, but the authors argued that Guilherme Polo’s work was more comprehensive.

</div>

<div id="updated-module-unittest" class="section">

<span id="unittest-section"></span>

### Updated module: unittest<a href="#updated-module-unittest" class="headerlink" title="Permalink to this headline">¶</a>

The <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> module was greatly enhanced; many new features were added. Most of these features were implemented by Michael Foord, unless otherwise noted. The enhanced version of the module is downloadable separately for use with Python versions 2.4 to 2.6, packaged as the <span class="pre">`unittest2`</span> package, from <a href="https://pypi.org/project/unittest2" class="reference external">https://pypi.org/project/unittest2</a>.

When used from the command line, the module can automatically discover tests. It’s not as fancy as <a href="http://pytest.org" class="reference external">py.test</a> or <a href="https://nose.readthedocs.io/" class="reference external">nose</a>, but provides a simple way to run tests kept within a set of package directories. For example, the following command will search the <span class="pre">`test/`</span> subdirectory for any importable test files named <span class="pre">`test*.py`</span>:

<div class="highlight-default notranslate">

<div class="highlight">

    python -m unittest discover -s test

</div>

</div>

Consult the <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> module documentation for more details. (Developed in <a href="https://bugs.python.org/issue6001" class="reference external">bpo-6001</a>.)

The <a href="../library/unittest.html#unittest.main" class="reference internal" title="unittest.main"><span class="pre"><code class="sourceCode python">main()</code></span></a> function supports some other new options:

- <a href="../library/unittest.html#cmdoption-unittest-b" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-b</code></span></a> or <span class="pre">`--buffer`</span> will buffer the standard output and standard error streams during each test. If the test passes, any resulting output will be discarded; on failure, the buffered output will be displayed.

- <a href="../library/unittest.html#cmdoption-unittest-c" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-c</code></span></a> or <span class="pre">`--catch`</span> will cause the control-C interrupt to be handled more gracefully. Instead of interrupting the test process immediately, the currently running test will be completed and then the partial results up to the interruption will be reported. If you’re impatient, a second press of control-C will cause an immediate interruption.

  This control-C handler tries to avoid causing problems when the code being tested or the tests being run have defined a signal handler of their own, by noticing that a signal handler was already set and calling it. If this doesn’t work for you, there’s a <a href="../library/unittest.html#unittest.removeHandler" class="reference internal" title="unittest.removeHandler"><span class="pre"><code class="sourceCode python">removeHandler()</code></span></a> decorator that can be used to mark tests that should have the control-C handling disabled.

- <a href="../library/unittest.html#cmdoption-unittest-f" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-f</code></span></a> or <span class="pre">`--failfast`</span> makes test execution stop immediately when a test fails instead of continuing to execute further tests. (Suggested by Cliff Dyer and implemented by Michael Foord; <a href="https://bugs.python.org/issue8074" class="reference external">bpo-8074</a>.)

The progress messages now show ‘x’ for expected failures and ‘u’ for unexpected successes when run in verbose mode. (Contributed by Benjamin Peterson.)

Test cases can raise the <a href="../library/unittest.html#unittest.SkipTest" class="reference internal" title="unittest.SkipTest"><span class="pre"><code class="sourceCode python">SkipTest</code></span></a> exception to skip a test (<a href="https://bugs.python.org/issue1034053" class="reference external">bpo-1034053</a>).

The error messages for <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertTrue" class="reference internal" title="unittest.TestCase.assertTrue"><span class="pre"><code class="sourceCode python">assertTrue()</code></span></a>, and <a href="../library/unittest.html#unittest.TestCase.assertFalse" class="reference internal" title="unittest.TestCase.assertFalse"><span class="pre"><code class="sourceCode python">assertFalse()</code></span></a> failures now provide more information. If you set the <a href="../library/unittest.html#unittest.TestCase.longMessage" class="reference internal" title="unittest.TestCase.longMessage"><span class="pre"><code class="sourceCode python">longMessage</code></span></a> attribute of your <a href="../library/unittest.html#unittest.TestCase" class="reference internal" title="unittest.TestCase"><span class="pre"><code class="sourceCode python">TestCase</code></span></a> classes to true, both the standard error message and any additional message you provide will be printed for failures. (Added by Michael Foord; <a href="https://bugs.python.org/issue5663" class="reference external">bpo-5663</a>.)

The <a href="../library/unittest.html#unittest.TestCase.assertRaises" class="reference internal" title="unittest.TestCase.assertRaises"><span class="pre"><code class="sourceCode python">assertRaises()</code></span></a> method now returns a context handler when called without providing a callable object to run. For example, you can write this:

<div class="highlight-default notranslate">

<div class="highlight">

    with self.assertRaises(KeyError):
        {}['foo']

</div>

</div>

(Implemented by Antoine Pitrou; <a href="https://bugs.python.org/issue4444" class="reference external">bpo-4444</a>.)

Module- and class-level setup and teardown fixtures are now supported. Modules can contain <span class="pre">`setUpModule()`</span> and <span class="pre">`tearDownModule()`</span> functions. Classes can have <a href="../library/unittest.html#unittest.TestCase.setUpClass" class="reference internal" title="unittest.TestCase.setUpClass"><span class="pre"><code class="sourceCode python">setUpClass()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.tearDownClass" class="reference internal" title="unittest.TestCase.tearDownClass"><span class="pre"><code class="sourceCode python">tearDownClass()</code></span></a> methods that must be defined as class methods (using <span class="pre">`@classmethod`</span> or equivalent). These functions and methods are invoked when the test runner switches to a test case in a different module or class.

The methods <a href="../library/unittest.html#unittest.TestCase.addCleanup" class="reference internal" title="unittest.TestCase.addCleanup"><span class="pre"><code class="sourceCode python">addCleanup()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.doCleanups" class="reference internal" title="unittest.TestCase.doCleanups"><span class="pre"><code class="sourceCode python">doCleanups()</code></span></a> were added. <a href="../library/unittest.html#unittest.TestCase.addCleanup" class="reference internal" title="unittest.TestCase.addCleanup"><span class="pre"><code class="sourceCode python">addCleanup()</code></span></a> lets you add cleanup functions that will be called unconditionally (after <a href="../library/unittest.html#unittest.TestCase.setUp" class="reference internal" title="unittest.TestCase.setUp"><span class="pre"><code class="sourceCode python">setUp()</code></span></a> if <a href="../library/unittest.html#unittest.TestCase.setUp" class="reference internal" title="unittest.TestCase.setUp"><span class="pre"><code class="sourceCode python">setUp()</code></span></a> fails, otherwise after <a href="../library/unittest.html#unittest.TestCase.tearDown" class="reference internal" title="unittest.TestCase.tearDown"><span class="pre"><code class="sourceCode python">tearDown()</code></span></a>). This allows for much simpler resource allocation and deallocation during tests (<a href="https://bugs.python.org/issue5679" class="reference external">bpo-5679</a>).

A number of new methods were added that provide more specialized tests. Many of these methods were written by Google engineers for use in their test suites; Gregory P. Smith, Michael Foord, and GvR worked on merging them into Python’s version of <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a>.

- <a href="../library/unittest.html#unittest.TestCase.assertIsNone" class="reference internal" title="unittest.TestCase.assertIsNone"><span class="pre"><code class="sourceCode python">assertIsNone()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.assertIsNotNone" class="reference internal" title="unittest.TestCase.assertIsNotNone"><span class="pre"><code class="sourceCode python">assertIsNotNone()</code></span></a> take one expression and verify that the result is or is not <span class="pre">`None`</span>.

- <a href="../library/unittest.html#unittest.TestCase.assertIs" class="reference internal" title="unittest.TestCase.assertIs"><span class="pre"><code class="sourceCode python">assertIs()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.assertIsNot" class="reference internal" title="unittest.TestCase.assertIsNot"><span class="pre"><code class="sourceCode python">assertIsNot()</code></span></a> take two values and check whether the two values evaluate to the same object or not. (Added by Michael Foord; <a href="https://bugs.python.org/issue2578" class="reference external">bpo-2578</a>.)

- <a href="../library/unittest.html#unittest.TestCase.assertIsInstance" class="reference internal" title="unittest.TestCase.assertIsInstance"><span class="pre"><code class="sourceCode python">assertIsInstance()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.assertNotIsInstance" class="reference internal" title="unittest.TestCase.assertNotIsInstance"><span class="pre"><code class="sourceCode python">assertNotIsInstance()</code></span></a> check whether the resulting object is an instance of a particular class, or of one of a tuple of classes. (Added by Georg Brandl; <a href="https://bugs.python.org/issue7031" class="reference external">bpo-7031</a>.)

- <a href="../library/unittest.html#unittest.TestCase.assertGreater" class="reference internal" title="unittest.TestCase.assertGreater"><span class="pre"><code class="sourceCode python">assertGreater()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertGreaterEqual" class="reference internal" title="unittest.TestCase.assertGreaterEqual"><span class="pre"><code class="sourceCode python">assertGreaterEqual()</code></span></a>, <a href="../library/unittest.html#unittest.TestCase.assertLess" class="reference internal" title="unittest.TestCase.assertLess"><span class="pre"><code class="sourceCode python">assertLess()</code></span></a>, and <a href="../library/unittest.html#unittest.TestCase.assertLessEqual" class="reference internal" title="unittest.TestCase.assertLessEqual"><span class="pre"><code class="sourceCode python">assertLessEqual()</code></span></a> compare two quantities.

- <a href="../library/unittest.html#unittest.TestCase.assertMultiLineEqual" class="reference internal" title="unittest.TestCase.assertMultiLineEqual"><span class="pre"><code class="sourceCode python">assertMultiLineEqual()</code></span></a> compares two strings, and if they’re not equal, displays a helpful comparison that highlights the differences in the two strings. This comparison is now used by default when Unicode strings are compared with <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a>.

- <a href="../library/unittest.html#unittest.TestCase.assertRegexpMatches" class="reference internal" title="unittest.TestCase.assertRegexpMatches"><span class="pre"><code class="sourceCode python">assertRegexpMatches()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.assertNotRegexpMatches" class="reference internal" title="unittest.TestCase.assertNotRegexpMatches"><span class="pre"><code class="sourceCode python">assertNotRegexpMatches()</code></span></a> checks whether the first argument is a string matching or not matching the regular expression provided as the second argument (<a href="https://bugs.python.org/issue8038" class="reference external">bpo-8038</a>).

- <a href="../library/unittest.html#unittest.TestCase.assertRaisesRegexp" class="reference internal" title="unittest.TestCase.assertRaisesRegexp"><span class="pre"><code class="sourceCode python">assertRaisesRegexp()</code></span></a> checks whether a particular exception is raised, and then also checks that the string representation of the exception matches the provided regular expression.

- <a href="../library/unittest.html#unittest.TestCase.assertIn" class="reference internal" title="unittest.TestCase.assertIn"><span class="pre"><code class="sourceCode python">assertIn()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.assertNotIn" class="reference internal" title="unittest.TestCase.assertNotIn"><span class="pre"><code class="sourceCode python">assertNotIn()</code></span></a> tests whether *first* is or is not in *second*.

- <a href="../library/unittest.html#unittest.TestCase.assertItemsEqual" class="reference internal" title="unittest.TestCase.assertItemsEqual"><span class="pre"><code class="sourceCode python">assertItemsEqual()</code></span></a> tests whether two provided sequences contain the same elements.

- <a href="../library/unittest.html#unittest.TestCase.assertSetEqual" class="reference internal" title="unittest.TestCase.assertSetEqual"><span class="pre"><code class="sourceCode python">assertSetEqual()</code></span></a> compares whether two sets are equal, and only reports the differences between the sets in case of error.

- Similarly, <a href="../library/unittest.html#unittest.TestCase.assertListEqual" class="reference internal" title="unittest.TestCase.assertListEqual"><span class="pre"><code class="sourceCode python">assertListEqual()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.assertTupleEqual" class="reference internal" title="unittest.TestCase.assertTupleEqual"><span class="pre"><code class="sourceCode python">assertTupleEqual()</code></span></a> compare the specified types and explain any differences without necessarily printing their full values; these methods are now used by default when comparing lists and tuples using <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a>. More generally, <a href="../library/unittest.html#unittest.TestCase.assertSequenceEqual" class="reference internal" title="unittest.TestCase.assertSequenceEqual"><span class="pre"><code class="sourceCode python">assertSequenceEqual()</code></span></a> compares two sequences and can optionally check whether both sequences are of a particular type.

- <a href="../library/unittest.html#unittest.TestCase.assertDictEqual" class="reference internal" title="unittest.TestCase.assertDictEqual"><span class="pre"><code class="sourceCode python">assertDictEqual()</code></span></a> compares two dictionaries and reports the differences; it’s now used by default when you compare two dictionaries using <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a>. <a href="../library/unittest.html#unittest.TestCase.assertDictContainsSubset" class="reference internal" title="unittest.TestCase.assertDictContainsSubset"><span class="pre"><code class="sourceCode python">assertDictContainsSubset()</code></span></a> checks whether all of the key/value pairs in *first* are found in *second*.

- <a href="../library/unittest.html#unittest.TestCase.assertAlmostEqual" class="reference internal" title="unittest.TestCase.assertAlmostEqual"><span class="pre"><code class="sourceCode python">assertAlmostEqual()</code></span></a> and <a href="../library/unittest.html#unittest.TestCase.assertNotAlmostEqual" class="reference internal" title="unittest.TestCase.assertNotAlmostEqual"><span class="pre"><code class="sourceCode python">assertNotAlmostEqual()</code></span></a> test whether *first* and *second* are approximately equal. This method can either round their difference to an optionally-specified number of *places* (the default is 7) and compare it to zero, or require the difference to be smaller than a supplied *delta* value.

- <a href="../library/unittest.html#unittest.TestLoader.loadTestsFromName" class="reference internal" title="unittest.TestLoader.loadTestsFromName"><span class="pre"><code class="sourceCode python">loadTestsFromName()</code></span></a> properly honors the <a href="../library/unittest.html#unittest.TestLoader.suiteClass" class="reference internal" title="unittest.TestLoader.suiteClass"><span class="pre"><code class="sourceCode python">suiteClass</code></span></a> attribute of the <a href="../library/unittest.html#unittest.TestLoader" class="reference internal" title="unittest.TestLoader"><span class="pre"><code class="sourceCode python">TestLoader</code></span></a>. (Fixed by Mark Roddy; <a href="https://bugs.python.org/issue6866" class="reference external">bpo-6866</a>.)

- A new hook lets you extend the <a href="../library/unittest.html#unittest.TestCase.assertEqual" class="reference internal" title="unittest.TestCase.assertEqual"><span class="pre"><code class="sourceCode python">assertEqual()</code></span></a> method to handle new data types. The <a href="../library/unittest.html#unittest.TestCase.addTypeEqualityFunc" class="reference internal" title="unittest.TestCase.addTypeEqualityFunc"><span class="pre"><code class="sourceCode python">addTypeEqualityFunc()</code></span></a> method takes a type object and a function. The function will be used when both of the objects being compared are of the specified type. This function should compare the two objects and raise an exception if they don’t match; it’s a good idea for the function to provide additional information about why the two objects aren’t matching, much as the new sequence comparison methods do.

<a href="../library/unittest.html#unittest.main" class="reference internal" title="unittest.main"><span class="pre"><code class="sourceCode python">unittest.main()</code></span></a> now takes an optional <span class="pre">`exit`</span> argument. If false, <a href="../library/unittest.html#unittest.main" class="reference internal" title="unittest.main"><span class="pre"><code class="sourceCode python">main()</code></span></a> doesn’t call <a href="../library/sys.html#sys.exit" class="reference internal" title="sys.exit"><span class="pre"><code class="sourceCode python">sys.exit()</code></span></a>, allowing <a href="../library/unittest.html#unittest.main" class="reference internal" title="unittest.main"><span class="pre"><code class="sourceCode python">main()</code></span></a> to be used from the interactive interpreter. (Contributed by J. Pablo Fernández; <a href="https://bugs.python.org/issue3379" class="reference external">bpo-3379</a>.)

<a href="../library/unittest.html#unittest.TestResult" class="reference internal" title="unittest.TestResult"><span class="pre"><code class="sourceCode python">TestResult</code></span></a> has new <a href="../library/unittest.html#unittest.TestResult.startTestRun" class="reference internal" title="unittest.TestResult.startTestRun"><span class="pre"><code class="sourceCode python">startTestRun()</code></span></a> and <a href="../library/unittest.html#unittest.TestResult.stopTestRun" class="reference internal" title="unittest.TestResult.stopTestRun"><span class="pre"><code class="sourceCode python">stopTestRun()</code></span></a> methods that are called immediately before and after a test run. (Contributed by Robert Collins; <a href="https://bugs.python.org/issue5728" class="reference external">bpo-5728</a>.)

With all these changes, the <span class="pre">`unittest.py`</span> was becoming awkwardly large, so the module was turned into a package and the code split into several files (by Benjamin Peterson). This doesn’t affect how the module is imported or used.

<div class="admonition seealso">

See also

<a href="http://www.voidspace.org.uk/python/articles/unittest2.shtml" class="reference external">http://www.voidspace.org.uk/python/articles/unittest2.shtml</a>  
Describes the new features, how to use them, and the rationale for various design decisions. (By Michael Foord.)

</div>

</div>

<div id="updated-module-elementtree-1-3" class="section">

<span id="elementtree-section"></span>

### Updated module: ElementTree 1.3<a href="#updated-module-elementtree-1-3" class="headerlink" title="Permalink to this headline">¶</a>

The version of the ElementTree library included with Python was updated to version 1.3. Some of the new features are:

- The various parsing functions now take a *parser* keyword argument giving an <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLParser" class="reference internal" title="xml.etree.ElementTree.XMLParser"><span class="pre"><code class="sourceCode python">XMLParser</code></span></a> instance that will be used. This makes it possible to override the file’s internal encoding:

  <div class="highlight-default notranslate">

  <div class="highlight">

      p = ET.XMLParser(encoding='utf-8')
      t = ET.XML("""<root/>""", parser=p)

  </div>

  </div>

  Errors in parsing XML now raise a <span class="pre">`ParseError`</span> exception, whose instances have a <span class="pre">`position`</span> attribute containing a (*line*, *column*) tuple giving the location of the problem.

- ElementTree’s code for converting trees to a string has been significantly reworked, making it roughly twice as fast in many cases. The <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.ElementTree.write" class="reference internal" title="xml.etree.ElementTree.ElementTree.write"><span class="pre"><code class="sourceCode python">ElementTree.write()</code></span></a> and <span class="pre">`Element.write()`</span> methods now have a *method* parameter that can be “xml” (the default), “html”, or “text”. HTML mode will output empty elements as <span class="pre">`<empty></empty>`</span> instead of <span class="pre">`<empty/>`</span>, and text mode will skip over elements and only output the text chunks. If you set the <span class="pre">`tag`</span> attribute of an element to <span class="pre">`None`</span> but leave its children in place, the element will be omitted when the tree is written out, so you don’t need to do more extensive rearrangement to remove a single element.

  Namespace handling has also been improved. All <span class="pre">`xmlns:<whatever>`</span> declarations are now output on the root element, not scattered throughout the resulting XML. You can set the default namespace for a tree by setting the <span class="pre">`default_namespace`</span> attribute and can register new prefixes with <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.register_namespace" class="reference internal" title="xml.etree.ElementTree.register_namespace"><span class="pre"><code class="sourceCode python">register_namespace()</code></span></a>. In XML mode, you can use the true/false *xml_declaration* parameter to suppress the XML declaration.

- New <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element" class="reference internal" title="xml.etree.ElementTree.Element"><span class="pre"><code class="sourceCode python">Element</code></span></a> method: <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element.extend" class="reference internal" title="xml.etree.ElementTree.Element.extend"><span class="pre"><code class="sourceCode python">extend()</code></span></a> appends the items from a sequence to the element’s children. Elements themselves behave like sequences, so it’s easy to move children from one element to another:

  <div class="highlight-default notranslate">

  <div class="highlight">

      from xml.etree import ElementTree as ET

      t = ET.XML("""<list>
        <item>1</item> <item>2</item>  <item>3</item>
      </list>""")
      new = ET.XML('<root/>')
      new.extend(t)

      # Outputs <root><item>1</item>...</root>
      print ET.tostring(new)

  </div>

  </div>

- New <span class="pre">`Element`</span> method: <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element.iter" class="reference internal" title="xml.etree.ElementTree.Element.iter"><span class="pre"><code class="sourceCode python"><span class="bu">iter</span>()</code></span></a> yields the children of the element as a generator. It’s also possible to write <span class="pre">`for`</span>` `<span class="pre">`child`</span>` `<span class="pre">`in`</span>` `<span class="pre">`elem:`</span> to loop over an element’s children. The existing method <span class="pre">`getiterator()`</span> is now deprecated, as is <span class="pre">`getchildren()`</span> which constructs and returns a list of children.

- New <span class="pre">`Element`</span> method: <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.Element.itertext" class="reference internal" title="xml.etree.ElementTree.Element.itertext"><span class="pre"><code class="sourceCode python">itertext()</code></span></a> yields all chunks of text that are descendants of the element. For example:

  <div class="highlight-default notranslate">

  <div class="highlight">

      t = ET.XML("""<list>
        <item>1</item> <item>2</item>  <item>3</item>
      </list>""")

      # Outputs ['\n  ', '1', ' ', '2', '  ', '3', '\n']
      print list(t.itertext())

  </div>

  </div>

- Deprecated: using an element as a Boolean (i.e., <span class="pre">`if`</span>` `<span class="pre">`elem:`</span>) would return true if the element had any children, or false if there were no children. This behaviour is confusing – <span class="pre">`None`</span> is false, but so is a childless element? – so it will now trigger a <a href="../library/exceptions.html#exceptions.FutureWarning" class="reference internal" title="exceptions.FutureWarning"><span class="pre"><code class="sourceCode python"><span class="pp">FutureWarning</span></code></span></a>. In your code, you should be explicit: write <span class="pre">`len(elem)`</span>` `<span class="pre">`!=`</span>` `<span class="pre">`0`</span> if you’re interested in the number of children, or <span class="pre">`elem`</span>` `<span class="pre">`is`</span>` `<span class="pre">`not`</span>` `<span class="pre">`None`</span>.

Fredrik Lundh develops ElementTree and produced the 1.3 version; you can read his article describing 1.3 at <a href="http://effbot.org/zone/elementtree-13-intro.htm" class="reference external">http://effbot.org/zone/elementtree-13-intro.htm</a>. Florent Xicluna updated the version included with Python, after discussions on python-dev and in <a href="https://bugs.python.org/issue6472" class="reference external">bpo-6472</a>.)

</div>

</div>

<div id="build-and-c-api-changes" class="section">

## Build and C API Changes<a href="#build-and-c-api-changes" class="headerlink" title="Permalink to this headline">¶</a>

Changes to Python’s build process and to the C API include:

- The latest release of the GNU Debugger, GDB 7, can be <a href="https://sourceware.org/gdb/current/onlinedocs/gdb/Python.html" class="reference external">scripted using Python</a>. When you begin debugging an executable program P, GDB will look for a file named <span class="pre">`P-gdb.py`</span> and automatically read it. Dave Malcolm contributed a <span class="pre">`python-gdb.py`</span> that adds a number of commands useful when debugging Python itself. For example, <span class="pre">`py-up`</span> and <span class="pre">`py-down`</span> go up or down one Python stack frame, which usually corresponds to several C stack frames. <span class="pre">`py-print`</span> prints the value of a Python variable, and <span class="pre">`py-bt`</span> prints the Python stack trace. (Added as a result of <a href="https://bugs.python.org/issue8032" class="reference external">bpo-8032</a>.)

- If you use the <span class="pre">`.gdbinit`</span> file provided with Python, the “pyo” macro in the 2.7 version now works correctly when the thread being debugged doesn’t hold the GIL; the macro now acquires it before printing. (Contributed by Victor Stinner; <a href="https://bugs.python.org/issue3632" class="reference external">bpo-3632</a>.)

- <a href="../c-api/init.html#c.Py_AddPendingCall" class="reference internal" title="Py_AddPendingCall"><span class="pre"><code class="sourceCode c">Py_AddPendingCall<span class="op">()</span></code></span></a> is now thread-safe, letting any worker thread submit notifications to the main Python thread. This is particularly useful for asynchronous IO operations. (Contributed by Kristján Valur Jónsson; <a href="https://bugs.python.org/issue4293" class="reference external">bpo-4293</a>.)

- New function: <a href="../c-api/code.html#c.PyCode_NewEmpty" class="reference internal" title="PyCode_NewEmpty"><span class="pre"><code class="sourceCode c">PyCode_NewEmpty<span class="op">()</span></code></span></a> creates an empty code object; only the filename, function name, and first line number are required. This is useful for extension modules that are attempting to construct a more useful traceback stack. Previously such extensions needed to call <a href="../c-api/code.html#c.PyCode_New" class="reference internal" title="PyCode_New"><span class="pre"><code class="sourceCode c">PyCode_New<span class="op">()</span></code></span></a>, which had many more arguments. (Added by Jeffrey Yasskin.)

- New function: <a href="../c-api/exceptions.html#c.PyErr_NewExceptionWithDoc" class="reference internal" title="PyErr_NewExceptionWithDoc"><span class="pre"><code class="sourceCode c">PyErr_NewExceptionWithDoc<span class="op">()</span></code></span></a> creates a new exception class, just as the existing <a href="../c-api/exceptions.html#c.PyErr_NewException" class="reference internal" title="PyErr_NewException"><span class="pre"><code class="sourceCode c">PyErr_NewException<span class="op">()</span></code></span></a> does, but takes an extra <span class="pre">`char`</span>` `<span class="pre">`*`</span> argument containing the docstring for the new exception class. (Added by ‘lekma’ on the Python bug tracker; <a href="https://bugs.python.org/issue7033" class="reference external">bpo-7033</a>.)

- New function: <a href="../c-api/reflection.html#c.PyFrame_GetLineNumber" class="reference internal" title="PyFrame_GetLineNumber"><span class="pre"><code class="sourceCode c">PyFrame_GetLineNumber<span class="op">()</span></code></span></a> takes a frame object and returns the line number that the frame is currently executing. Previously code would need to get the index of the bytecode instruction currently executing, and then look up the line number corresponding to that address. (Added by Jeffrey Yasskin.)

- New functions: <a href="../c-api/long.html#c.PyLong_AsLongAndOverflow" class="reference internal" title="PyLong_AsLongAndOverflow"><span class="pre"><code class="sourceCode c">PyLong_AsLongAndOverflow<span class="op">()</span></code></span></a> and <a href="../c-api/long.html#c.PyLong_AsLongLongAndOverflow" class="reference internal" title="PyLong_AsLongLongAndOverflow"><span class="pre"><code class="sourceCode c">PyLong_AsLongLongAndOverflow<span class="op">()</span></code></span></a> approximates a Python long integer as a C <span class="pre">`long`</span> or <span class="pre">`long`</span>` `<span class="pre">`long`</span>. If the number is too large to fit into the output type, an *overflow* flag is set and returned to the caller. (Contributed by Case Van Horsen; <a href="https://bugs.python.org/issue7528" class="reference external">bpo-7528</a> and <a href="https://bugs.python.org/issue7767" class="reference external">bpo-7767</a>.)

- New function: stemming from the rewrite of string-to-float conversion, a new <a href="../c-api/conversion.html#c.PyOS_string_to_double" class="reference internal" title="PyOS_string_to_double"><span class="pre"><code class="sourceCode c">PyOS_string_to_double<span class="op">()</span></code></span></a> function was added. The old <a href="../c-api/conversion.html#c.PyOS_ascii_strtod" class="reference internal" title="PyOS_ascii_strtod"><span class="pre"><code class="sourceCode c">PyOS_ascii_strtod<span class="op">()</span></code></span></a> and <a href="../c-api/conversion.html#c.PyOS_ascii_atof" class="reference internal" title="PyOS_ascii_atof"><span class="pre"><code class="sourceCode c">PyOS_ascii_atof<span class="op">()</span></code></span></a> functions are now deprecated.

- New function: <a href="../c-api/init.html#c.PySys_SetArgvEx" class="reference internal" title="PySys_SetArgvEx"><span class="pre"><code class="sourceCode c">PySys_SetArgvEx<span class="op">()</span></code></span></a> sets the value of <span class="pre">`sys.argv`</span> and can optionally update <span class="pre">`sys.path`</span> to include the directory containing the script named by <span class="pre">`sys.argv[0]`</span> depending on the value of an *updatepath* parameter.

  This function was added to close a security hole for applications that embed Python. The old function, <a href="../c-api/init.html#c.PySys_SetArgv" class="reference internal" title="PySys_SetArgv"><span class="pre"><code class="sourceCode c">PySys_SetArgv<span class="op">()</span></code></span></a>, would always update <span class="pre">`sys.path`</span>, and sometimes it would add the current directory. This meant that, if you ran an application embedding Python in a directory controlled by someone else, attackers could put a Trojan-horse module in the directory (say, a file named <span class="pre">`os.py`</span>) that your application would then import and run.

  If you maintain a C/C++ application that embeds Python, check whether you’re calling <a href="../c-api/init.html#c.PySys_SetArgv" class="reference internal" title="PySys_SetArgv"><span class="pre"><code class="sourceCode c">PySys_SetArgv<span class="op">()</span></code></span></a> and carefully consider whether the application should be using <a href="../c-api/init.html#c.PySys_SetArgvEx" class="reference internal" title="PySys_SetArgvEx"><span class="pre"><code class="sourceCode c">PySys_SetArgvEx<span class="op">()</span></code></span></a> with *updatepath* set to false.

  Security issue reported as <a href="https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2008-5983" class="reference external">CVE-2008-5983</a>; discussed in <a href="https://bugs.python.org/issue5753" class="reference external">bpo-5753</a>, and fixed by Antoine Pitrou.

- New macros: the Python header files now define the following macros: <span class="pre">`Py_ISALNUM`</span>, <span class="pre">`Py_ISALPHA`</span>, <span class="pre">`Py_ISDIGIT`</span>, <span class="pre">`Py_ISLOWER`</span>, <span class="pre">`Py_ISSPACE`</span>, <span class="pre">`Py_ISUPPER`</span>, <span class="pre">`Py_ISXDIGIT`</span>, <span class="pre">`Py_TOLOWER`</span>, and <span class="pre">`Py_TOUPPER`</span>. All of these functions are analogous to the C standard macros for classifying characters, but ignore the current locale setting, because in several places Python needs to analyze characters in a locale-independent way. (Added by Eric Smith; <a href="https://bugs.python.org/issue5793" class="reference external">bpo-5793</a>.)

- Removed function: <span class="pre">`PyEval_CallObject`</span> is now only available as a macro. A function version was being kept around to preserve ABI linking compatibility, but that was in 1997; it can certainly be deleted by now. (Removed by Antoine Pitrou; <a href="https://bugs.python.org/issue8276" class="reference external">bpo-8276</a>.)

- New format codes: the <span class="pre">`PyFormat_FromString()`</span>, <span class="pre">`PyFormat_FromStringV()`</span>, and <a href="../c-api/exceptions.html#c.PyErr_Format" class="reference internal" title="PyErr_Format"><span class="pre"><code class="sourceCode c">PyErr_Format<span class="op">()</span></code></span></a> functions now accept <span class="pre">`%lld`</span> and <span class="pre">`%llu`</span> format codes for displaying C’s <span class="pre">`long`</span>` `<span class="pre">`long`</span> types. (Contributed by Mark Dickinson; <a href="https://bugs.python.org/issue7228" class="reference external">bpo-7228</a>.)

- The complicated interaction between threads and process forking has been changed. Previously, the child process created by <a href="../library/os.html#os.fork" class="reference internal" title="os.fork"><span class="pre"><code class="sourceCode python">os.fork()</code></span></a> might fail because the child is created with only a single thread running, the thread performing the <a href="../library/os.html#os.fork" class="reference internal" title="os.fork"><span class="pre"><code class="sourceCode python">os.fork()</code></span></a>. If other threads were holding a lock, such as Python’s import lock, when the fork was performed, the lock would still be marked as “held” in the new process. But in the child process nothing would ever release the lock, since the other threads weren’t replicated, and the child process would no longer be able to perform imports.

  Python 2.7 acquires the import lock before performing an <a href="../library/os.html#os.fork" class="reference internal" title="os.fork"><span class="pre"><code class="sourceCode python">os.fork()</code></span></a>, and will also clean up any locks created using the <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> module. C extension modules that have internal locks, or that call <span class="pre">`fork()`</span> themselves, will not benefit from this clean-up.

  (Fixed by Thomas Wouters; <a href="https://bugs.python.org/issue1590864" class="reference external">bpo-1590864</a>.)

- The <a href="../c-api/init.html#c.Py_Finalize" class="reference internal" title="Py_Finalize"><span class="pre"><code class="sourceCode c">Py_Finalize<span class="op">()</span></code></span></a> function now calls the internal <span class="pre">`threading._shutdown()`</span> function; this prevents some exceptions from being raised when an interpreter shuts down. (Patch by Adam Olsen; <a href="https://bugs.python.org/issue1722344" class="reference external">bpo-1722344</a>.)

- When using the <a href="../c-api/structures.html#c.PyMemberDef" class="reference internal" title="PyMemberDef"><span class="pre"><code class="sourceCode c">PyMemberDef</code></span></a> structure to define attributes of a type, Python will no longer let you try to delete or set a <span class="pre">`T_STRING_INPLACE`</span> attribute.

- Global symbols defined by the <a href="../library/ctypes.html#module-ctypes" class="reference internal" title="ctypes: A foreign function library for Python."><span class="pre"><code class="sourceCode python">ctypes</code></span></a> module are now prefixed with <span class="pre">`Py`</span>, or with <span class="pre">`_ctypes`</span>. (Implemented by Thomas Heller; <a href="https://bugs.python.org/issue3102" class="reference external">bpo-3102</a>.)

- New configure option: the <span class="pre">`--with-system-expat`</span> switch allows building the <span class="pre">`pyexpat`</span> module to use the system Expat library. (Contributed by Arfrever Frehtes Taifersar Arahesis; <a href="https://bugs.python.org/issue7609" class="reference external">bpo-7609</a>.)

- New configure option: the <span class="pre">`--with-valgrind`</span> option will now disable the pymalloc allocator, which is difficult for the Valgrind memory-error detector to analyze correctly. Valgrind will therefore be better at detecting memory leaks and overruns. (Contributed by James Henstridge; <a href="https://bugs.python.org/issue2422" class="reference external">bpo-2422</a>.)

- New configure option: you can now supply an empty string to <span class="pre">`--with-dbmliborder=`</span> in order to disable all of the various DBM modules. (Added by Arfrever Frehtes Taifersar Arahesis; <a href="https://bugs.python.org/issue6491" class="reference external">bpo-6491</a>.)

- The **configure** script now checks for floating-point rounding bugs on certain 32-bit Intel chips and defines a <span class="pre">`X87_DOUBLE_ROUNDING`</span> preprocessor definition. No code currently uses this definition, but it’s available if anyone wishes to use it. (Added by Mark Dickinson; <a href="https://bugs.python.org/issue2937" class="reference external">bpo-2937</a>.)

  **configure** also now sets a <span id="index-12" class="target"></span><span class="pre">`LDCXXSHARED`</span> Makefile variable for supporting C++ linking. (Contributed by Arfrever Frehtes Taifersar Arahesis; <a href="https://bugs.python.org/issue1222585" class="reference external">bpo-1222585</a>.)

- The build process now creates the necessary files for pkg-config support. (Contributed by Clinton Roy; <a href="https://bugs.python.org/issue3585" class="reference external">bpo-3585</a>.)

- The build process now supports Subversion 1.7. (Contributed by Arfrever Frehtes Taifersar Arahesis; <a href="https://bugs.python.org/issue6094" class="reference external">bpo-6094</a>.)

<div id="capsules" class="section">

<span id="whatsnew27-capsules"></span>

### Capsules<a href="#capsules" class="headerlink" title="Permalink to this headline">¶</a>

Python 3.1 adds a new C datatype, <a href="../c-api/capsule.html#c.PyCapsule" class="reference internal" title="PyCapsule"><span class="pre"><code class="sourceCode c">PyCapsule</code></span></a>, for providing a C API to an extension module. A capsule is essentially the holder of a C <span class="pre">`void`</span>` `<span class="pre">`*`</span> pointer, and is made available as a module attribute; for example, the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module’s API is exposed as <span class="pre">`socket.CAPI`</span>, and <a href="../library/unicodedata.html#module-unicodedata" class="reference internal" title="unicodedata: Access the Unicode Database."><span class="pre"><code class="sourceCode python">unicodedata</code></span></a> exposes <span class="pre">`ucnhash_CAPI`</span>. Other extensions can import the module, access its dictionary to get the capsule object, and then get the <span class="pre">`void`</span>` `<span class="pre">`*`</span> pointer, which will usually point to an array of pointers to the module’s various API functions.

There is an existing data type already used for this, <a href="../c-api/cobject.html#c.PyCObject" class="reference internal" title="PyCObject"><span class="pre"><code class="sourceCode c">PyCObject</code></span></a>, but it doesn’t provide type safety. Evil code written in pure Python could cause a segmentation fault by taking a <a href="../c-api/cobject.html#c.PyCObject" class="reference internal" title="PyCObject"><span class="pre"><code class="sourceCode c">PyCObject</code></span></a> from module A and somehow substituting it for the <a href="../c-api/cobject.html#c.PyCObject" class="reference internal" title="PyCObject"><span class="pre"><code class="sourceCode c">PyCObject</code></span></a> in module B. Capsules know their own name, and getting the pointer requires providing the name:

<div class="highlight-c notranslate">

<div class="highlight">

    void *vtable;

    if (!PyCapsule_IsValid(capsule, "mymodule.CAPI") {
            PyErr_SetString(PyExc_ValueError, "argument type invalid");
            return NULL;
    }

    vtable = PyCapsule_GetPointer(capsule, "mymodule.CAPI");

</div>

</div>

You are assured that <span class="pre">`vtable`</span> points to whatever you’re expecting. If a different capsule was passed in, <a href="../c-api/capsule.html#c.PyCapsule_IsValid" class="reference internal" title="PyCapsule_IsValid"><span class="pre"><code class="sourceCode c">PyCapsule_IsValid<span class="op">()</span></code></span></a> would detect the mismatched name and return false. Refer to <a href="../extending/extending.html#using-capsules" class="reference internal"><span class="std std-ref">Providing a C API for an Extension Module</span></a> for more information on using these objects.

Python 2.7 now uses capsules internally to provide various extension-module APIs, but the <a href="../c-api/cobject.html#c.PyCObject_AsVoidPtr" class="reference internal" title="PyCObject_AsVoidPtr"><span class="pre"><code class="sourceCode c">PyCObject_AsVoidPtr<span class="op">()</span></code></span></a> was modified to handle capsules, preserving compile-time compatibility with the <span class="pre">`CObject`</span> interface. Use of <a href="../c-api/cobject.html#c.PyCObject_AsVoidPtr" class="reference internal" title="PyCObject_AsVoidPtr"><span class="pre"><code class="sourceCode c">PyCObject_AsVoidPtr<span class="op">()</span></code></span></a> will signal a <a href="../library/exceptions.html#exceptions.PendingDeprecationWarning" class="reference internal" title="exceptions.PendingDeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">PendingDeprecationWarning</span></code></span></a>, which is silent by default.

Implemented in Python 3.1 and backported to 2.7 by Larry Hastings; discussed in <a href="https://bugs.python.org/issue5630" class="reference external">bpo-5630</a>.

</div>

<div id="port-specific-changes-windows" class="section">

### Port-Specific Changes: Windows<a href="#port-specific-changes-windows" class="headerlink" title="Permalink to this headline">¶</a>

- The <a href="../library/msvcrt.html#module-msvcrt" class="reference internal" title="msvcrt: Miscellaneous useful routines from the MS VC++ runtime. (Windows)"><span class="pre"><code class="sourceCode python">msvcrt</code></span></a> module now contains some constants from the <span class="pre">`crtassem.h`</span> header file: <span class="pre">`CRT_ASSEMBLY_VERSION`</span>, <span class="pre">`VC_ASSEMBLY_PUBLICKEYTOKEN`</span>, and <span class="pre">`LIBRARIES_ASSEMBLY_NAME_PREFIX`</span>. (Contributed by David Cournapeau; <a href="https://bugs.python.org/issue4365" class="reference external">bpo-4365</a>.)

- The <a href="../library/_winreg.html#module-_winreg" class="reference internal" title="_winreg: Routines and objects for manipulating the Windows registry. (Windows)"><span class="pre"><code class="sourceCode python">_winreg</code></span></a> module for accessing the registry now implements the <a href="../library/_winreg.html#_winreg.CreateKeyEx" class="reference internal" title="_winreg.CreateKeyEx"><span class="pre"><code class="sourceCode python">CreateKeyEx()</code></span></a> and <a href="../library/_winreg.html#_winreg.DeleteKeyEx" class="reference internal" title="_winreg.DeleteKeyEx"><span class="pre"><code class="sourceCode python">DeleteKeyEx()</code></span></a> functions, extended versions of previously-supported functions that take several extra arguments. The <a href="../library/_winreg.html#_winreg.DisableReflectionKey" class="reference internal" title="_winreg.DisableReflectionKey"><span class="pre"><code class="sourceCode python">DisableReflectionKey()</code></span></a>, <a href="../library/_winreg.html#_winreg.EnableReflectionKey" class="reference internal" title="_winreg.EnableReflectionKey"><span class="pre"><code class="sourceCode python">EnableReflectionKey()</code></span></a>, and <a href="../library/_winreg.html#_winreg.QueryReflectionKey" class="reference internal" title="_winreg.QueryReflectionKey"><span class="pre"><code class="sourceCode python">QueryReflectionKey()</code></span></a> were also tested and documented. (Implemented by Brian Curtin: <a href="https://bugs.python.org/issue7347" class="reference external">bpo-7347</a>.)

- The new <span class="pre">`_beginthreadex()`</span> API is used to start threads, and the native thread-local storage functions are now used. (Contributed by Kristján Valur Jónsson; <a href="https://bugs.python.org/issue3582" class="reference external">bpo-3582</a>.)

- The <a href="../library/os.html#os.kill" class="reference internal" title="os.kill"><span class="pre"><code class="sourceCode python">os.kill()</code></span></a> function now works on Windows. The signal value can be the constants <span class="pre">`CTRL_C_EVENT`</span>, <span class="pre">`CTRL_BREAK_EVENT`</span>, or any integer. The first two constants will send <span class="kbd kbd docutils literal notranslate">Control-C</span> and <span class="kbd kbd docutils literal notranslate">Control-Break</span> keystroke events to subprocesses; any other value will use the <span class="pre">`TerminateProcess()`</span> API. (Contributed by Miki Tebeka; <a href="https://bugs.python.org/issue1220212" class="reference external">bpo-1220212</a>.)

- The <a href="../library/os.html#os.listdir" class="reference internal" title="os.listdir"><span class="pre"><code class="sourceCode python">os.listdir()</code></span></a> function now correctly fails for an empty path. (Fixed by Hirokazu Yamamoto; <a href="https://bugs.python.org/issue5913" class="reference external">bpo-5913</a>.)

- The <span class="pre">`mimelib`</span> module will now read the MIME database from the Windows registry when initializing. (Patch by Gabriel Genellina; <a href="https://bugs.python.org/issue4969" class="reference external">bpo-4969</a>.)

</div>

<div id="port-specific-changes-mac-os-x" class="section">

### Port-Specific Changes: Mac OS X<a href="#port-specific-changes-mac-os-x" class="headerlink" title="Permalink to this headline">¶</a>

- The path <span class="pre">`/Library/Python/2.7/site-packages`</span> is now appended to <span class="pre">`sys.path`</span>, in order to share added packages between the system installation and a user-installed copy of the same version. (Changed by Ronald Oussoren; <a href="https://bugs.python.org/issue4865" class="reference external">bpo-4865</a>.)

  > <div>
  >
  > <div class="versionchanged">
  >
  > <span class="versionmodified changed">Changed in version 2.7.13: </span>As of 2.7.13, this change was removed. <span class="pre">`/Library/Python/2.7/site-packages`</span>, the site-packages directory used by the Apple-supplied system Python 2.7 is no longer appended to <span class="pre">`sys.path`</span> for user-installed Pythons such as from the python.org installers. As of macOS 10.12, Apple changed how the system site-packages directory is configured, which could cause installation of pip components, like setuptools, to fail. Packages installed for the system Python will no longer be shared with user-installed Pythons. (<a href="https://bugs.python.org/issue28440" class="reference external">bpo-28440</a>)
  >
  > </div>
  >
  > </div>

</div>

<div id="port-specific-changes-freebsd" class="section">

### Port-Specific Changes: FreeBSD<a href="#port-specific-changes-freebsd" class="headerlink" title="Permalink to this headline">¶</a>

- FreeBSD 7.1’s <span class="pre">`SO_SETFIB`</span> constant, used with <span class="pre">`getsockopt()`</span>/<span class="pre">`setsockopt()`</span> to select an alternate routing table, is now available in the <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module. (Added by Kyle VanderBeek; <a href="https://bugs.python.org/issue8235" class="reference external">bpo-8235</a>.)

</div>

</div>

<div id="other-changes-and-fixes" class="section">

## Other Changes and Fixes<a href="#other-changes-and-fixes" class="headerlink" title="Permalink to this headline">¶</a>

- Two benchmark scripts, <span class="pre">`iobench`</span> and <span class="pre">`ccbench`</span>, were added to the <span class="pre">`Tools`</span> directory. <span class="pre">`iobench`</span> measures the speed of the built-in file I/O objects returned by <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> while performing various operations, and <span class="pre">`ccbench`</span> is a concurrency benchmark that tries to measure computing throughput, thread switching latency, and IO processing bandwidth when performing several tasks using a varying number of threads.

- The <span class="pre">`Tools/i18n/msgfmt.py`</span> script now understands plural forms in <span class="pre">`.po`</span> files. (Fixed by Martin von Löwis; <a href="https://bugs.python.org/issue5464" class="reference external">bpo-5464</a>.)

- When importing a module from a <span class="pre">`.pyc`</span> or <span class="pre">`.pyo`</span> file with an existing <span class="pre">`.py`</span> counterpart, the <span class="pre">`co_filename`</span> attributes of the resulting code objects are overwritten when the original filename is obsolete. This can happen if the file has been renamed, moved, or is accessed through different paths. (Patch by Ziga Seilnacht and Jean-Paul Calderone; <a href="https://bugs.python.org/issue1180193" class="reference external">bpo-1180193</a>.)

- The <span class="pre">`regrtest.py`</span> script now takes a <span class="pre">`--randseed=`</span> switch that takes an integer that will be used as the random seed for the <span class="pre">`-r`</span> option that executes tests in random order. The <span class="pre">`-r`</span> option also reports the seed that was used (Added by Collin Winter.)

- Another <span class="pre">`regrtest.py`</span> switch is <span class="pre">`-j`</span>, which takes an integer specifying how many tests run in parallel. This allows reducing the total runtime on multi-core machines. This option is compatible with several other options, including the <span class="pre">`-R`</span> switch which is known to produce long runtimes. (Added by Antoine Pitrou, <a href="https://bugs.python.org/issue6152" class="reference external">bpo-6152</a>.) This can also be used with a new <span class="pre">`-F`</span> switch that runs selected tests in a loop until they fail. (Added by Antoine Pitrou; <a href="https://bugs.python.org/issue7312" class="reference external">bpo-7312</a>.)

- When executed as a script, the <span class="pre">`py_compile.py`</span> module now accepts <span class="pre">`'-'`</span> as an argument, which will read standard input for the list of filenames to be compiled. (Contributed by Piotr Ożarowski; <a href="https://bugs.python.org/issue8233" class="reference external">bpo-8233</a>.)

</div>

<div id="porting-to-python-2-7" class="section">

## Porting to Python 2.7<a href="#porting-to-python-2-7" class="headerlink" title="Permalink to this headline">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code:

- The <a href="../library/functions.html#range" class="reference internal" title="range"><span class="pre"><code class="sourceCode python"><span class="bu">range</span>()</code></span></a> function processes its arguments more consistently; it will now call <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="fu">__int__</span>()</code></span></a> on non-float, non-integer arguments that are supplied to it. (Fixed by Alexander Belopolsky; <a href="https://bugs.python.org/issue1533" class="reference external">bpo-1533</a>.)

- The string <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> method changed the default precision used for floating-point and complex numbers from 6 decimal places to 12, which matches the precision used by <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a>. (Changed by Eric Smith; <a href="https://bugs.python.org/issue5920" class="reference external">bpo-5920</a>.)

- Because of an optimization for the <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement, the special methods <a href="../reference/datamodel.html#object.__enter__" class="reference internal" title="object.__enter__"><span class="pre"><code class="sourceCode python"><span class="fu">__enter__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> must belong to the object’s type, and cannot be directly attached to the object’s instance. This affects new-style classes (derived from <a href="../library/functions.html#object" class="reference internal" title="object"><span class="pre"><code class="sourceCode python"><span class="bu">object</span></code></span></a>) and C extension types. (<a href="https://bugs.python.org/issue6101" class="reference external">bpo-6101</a>.)

- Due to a bug in Python 2.6, the *exc_value* parameter to <a href="../reference/datamodel.html#object.__exit__" class="reference internal" title="object.__exit__"><span class="pre"><code class="sourceCode python"><span class="fu">__exit__</span>()</code></span></a> methods was often the string representation of the exception, not an instance. This was fixed in 2.7, so *exc_value* will be an instance as expected. (Fixed by Florent Xicluna; <a href="https://bugs.python.org/issue7853" class="reference external">bpo-7853</a>.)

- When a restricted set of attributes were set using <span class="pre">`__slots__`</span>, deleting an unset attribute would not raise <a href="../library/exceptions.html#exceptions.AttributeError" class="reference internal" title="exceptions.AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> as you would expect. Fixed by Benjamin Peterson; <a href="https://bugs.python.org/issue7604" class="reference external">bpo-7604</a>.)

In the standard library:

- Operations with <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime</code></span></a> instances that resulted in a year falling outside the supported range didn’t always raise <a href="../library/exceptions.html#exceptions.OverflowError" class="reference internal" title="exceptions.OverflowError"><span class="pre"><code class="sourceCode python"><span class="pp">OverflowError</span></code></span></a>. Such errors are now checked more carefully and will now raise the exception. (Reported by Mark Leander, patch by Anand B. Pillai and Alexander Belopolsky; <a href="https://bugs.python.org/issue7150" class="reference external">bpo-7150</a>.)

- When using <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a> instances with a string’s <a href="../library/functions.html#format" class="reference internal" title="format"><span class="pre"><code class="sourceCode python"><span class="bu">format</span>()</code></span></a> method, the default alignment was previously left-alignment. This has been changed to right-alignment, which might change the output of your programs. (Changed by Mark Dickinson; <a href="https://bugs.python.org/issue6857" class="reference external">bpo-6857</a>.)

  Comparisons involving a signaling NaN value (or <span class="pre">`sNAN`</span>) now signal <a href="../library/decimal.html#decimal.InvalidOperation" class="reference internal" title="decimal.InvalidOperation"><span class="pre"><code class="sourceCode python">InvalidOperation</code></span></a> instead of silently returning a true or false value depending on the comparison operator. Quiet NaN values (or <span class="pre">`NaN`</span>) are now hashable. (Fixed by Mark Dickinson; <a href="https://bugs.python.org/issue7279" class="reference external">bpo-7279</a>.)

- The ElementTree library, <span class="pre">`xml.etree`</span>, no longer escapes ampersands and angle brackets when outputting an XML processing instruction (which looks like \<?xml-stylesheet href=”#style1”?\>) or comment (which looks like \<!– comment –\>). (Patch by Neil Muller; <a href="https://bugs.python.org/issue2746" class="reference external">bpo-2746</a>.)

- The <span class="pre">`readline()`</span> method of <a href="../library/stringio.html#StringIO.StringIO" class="reference internal" title="StringIO.StringIO"><span class="pre"><code class="sourceCode python">StringIO</code></span></a> objects now does nothing when a negative length is requested, as other file-like objects do. (<a href="https://bugs.python.org/issue7348" class="reference external">bpo-7348</a>).

- The <a href="../library/syslog.html#module-syslog" class="reference internal" title="syslog: An interface to the Unix syslog library routines. (Unix)"><span class="pre"><code class="sourceCode python">syslog</code></span></a> module will now use the value of <span class="pre">`sys.argv[0]`</span> as the identifier instead of the previous default value of <span class="pre">`'python'`</span>. (Changed by Sean Reifschneider; <a href="https://bugs.python.org/issue8451" class="reference external">bpo-8451</a>.)

- The <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module’s default error handling has changed, to no longer suppress fatal errors. The default error level was previously 0, which meant that errors would only result in a message being written to the debug log, but because the debug log is not activated by default, these errors go unnoticed. The default error level is now 1, which raises an exception if there’s an error. (Changed by Lars Gustäbel; <a href="https://bugs.python.org/issue7357" class="reference external">bpo-7357</a>.)

- The <a href="../library/urlparse.html#module-urlparse" class="reference internal" title="urlparse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urlparse</code></span></a> module’s <a href="../library/urlparse.html#urlparse.urlsplit" class="reference internal" title="urlparse.urlsplit"><span class="pre"><code class="sourceCode python">urlsplit()</code></span></a> now handles unknown URL schemes in a fashion compliant with <span id="index-13" class="target"></span><a href="https://tools.ietf.org/html/rfc3986.html" class="rfc reference external"><strong>RFC 3986</strong></a>: if the URL is of the form <span class="pre">`"<something>://..."`</span>, the text before the <span class="pre">`://`</span> is treated as the scheme, even if it’s a made-up scheme that the module doesn’t know about. This change may break code that worked around the old behaviour. For example, Python 2.6.4 or 2.5 will return the following:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import urlparse
      >>> urlparse.urlsplit('invented://host/filename?query')
      ('invented', '', '//host/filename?query', '', '')

  </div>

  </div>

  Python 2.7 (and Python 2.6.5) will return:

  <div class="doctest highlight-default notranslate">

  <div class="highlight">

      >>> import urlparse
      >>> urlparse.urlsplit('invented://host/filename?query')
      ('invented', 'host', '/filename?query', '', '')

  </div>

  </div>

  (Python 2.7 actually produces slightly different output, since it returns a named tuple instead of a standard tuple.)

For C extensions:

- C extensions that use integer format codes with the <span class="pre">`PyArg_Parse*`</span> family of functions will now raise a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception instead of triggering a <a href="../library/exceptions.html#exceptions.DeprecationWarning" class="reference internal" title="exceptions.DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> (<a href="https://bugs.python.org/issue5080" class="reference external">bpo-5080</a>).

- Use the new <a href="../c-api/conversion.html#c.PyOS_string_to_double" class="reference internal" title="PyOS_string_to_double"><span class="pre"><code class="sourceCode c">PyOS_string_to_double<span class="op">()</span></code></span></a> function instead of the old <a href="../c-api/conversion.html#c.PyOS_ascii_strtod" class="reference internal" title="PyOS_ascii_strtod"><span class="pre"><code class="sourceCode c">PyOS_ascii_strtod<span class="op">()</span></code></span></a> and <a href="../c-api/conversion.html#c.PyOS_ascii_atof" class="reference internal" title="PyOS_ascii_atof"><span class="pre"><code class="sourceCode c">PyOS_ascii_atof<span class="op">()</span></code></span></a> functions, which are now deprecated.

For applications that embed Python:

- The <a href="../c-api/init.html#c.PySys_SetArgvEx" class="reference internal" title="PySys_SetArgvEx"><span class="pre"><code class="sourceCode c">PySys_SetArgvEx<span class="op">()</span></code></span></a> function was added, letting applications close a security hole when the existing <a href="../c-api/init.html#c.PySys_SetArgv" class="reference internal" title="PySys_SetArgv"><span class="pre"><code class="sourceCode c">PySys_SetArgv<span class="op">()</span></code></span></a> function was used. Check whether you’re calling <a href="../c-api/init.html#c.PySys_SetArgv" class="reference internal" title="PySys_SetArgv"><span class="pre"><code class="sourceCode c">PySys_SetArgv<span class="op">()</span></code></span></a> and carefully consider whether the application should be using <a href="../c-api/init.html#c.PySys_SetArgvEx" class="reference internal" title="PySys_SetArgvEx"><span class="pre"><code class="sourceCode c">PySys_SetArgvEx<span class="op">()</span></code></span></a> with *updatepath* set to false.

</div>

<div id="new-features-added-to-python-2-7-maintenance-releases" class="section">

<span id="py27-maintenance-enhancements"></span>

## New Features Added to Python 2.7 Maintenance Releases<a href="#new-features-added-to-python-2-7-maintenance-releases" class="headerlink" title="Permalink to this headline">¶</a>

New features may be added to Python 2.7 maintenance releases when the situation genuinely calls for it. Any such additions must go through the Python Enhancement Proposal process, and make a compelling case for why they can’t be adequately addressed by either adding the new feature solely to Python 3, or else by publishing it on the Python Package Index.

In addition to the specific proposals listed below, there is a general exemption allowing new <span class="pre">`-3`</span> warnings to be added in any Python 2.7 maintenance release.

<div id="two-new-environment-variables-for-debug-mode" class="section">

### Two new environment variables for debug mode<a href="#two-new-environment-variables-for-debug-mode" class="headerlink" title="Permalink to this headline">¶</a>

In debug mode, the <span class="pre">`[xxx`</span>` `<span class="pre">`refs]`</span> statistic is not written by default, the <span id="index-14" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONSHOWREFCOUNT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONSHOWREFCOUNT</code></span></a> environment variable now must also be set. (Contributed by Victor Stinner; <a href="https://bugs.python.org/issue31733" class="reference external">bpo-31733</a>.)

When Python is compiled with <span class="pre">`COUNT_ALLOC`</span> defined, allocation counts are no longer dumped by default anymore: the <span id="index-15" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONSHOWALLOCCOUNT" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONSHOWALLOCCOUNT</code></span></a> environment variable must now also be set. Moreover, allocation counts are now dumped into stderr, rather than stdout. (Contributed by Victor Stinner; <a href="https://bugs.python.org/issue31692" class="reference external">bpo-31692</a>.)

<div class="versionadded">

<span class="versionmodified added">New in version 2.7.15.</span>

</div>

</div>

<div id="pep-434-idle-enhancement-exception-for-all-branches" class="section">

### PEP 434: IDLE Enhancement Exception for All Branches<a href="#pep-434-idle-enhancement-exception-for-all-branches" class="headerlink" title="Permalink to this headline">¶</a>

<span id="index-16" class="target"></span><a href="https://www.python.org/dev/peps/pep-0434" class="pep reference external"><strong>PEP 434</strong></a> describes a general exemption for changes made to the IDLE development environment shipped along with Python. This exemption makes it possible for the IDLE developers to provide a more consistent user experience across all supported versions of Python 2 and 3.

For details of any IDLE changes, refer to the NEWS file for the specific release.

</div>

<div id="pep-466-network-security-enhancements-for-python-2-7" class="section">

### PEP 466: Network Security Enhancements for Python 2.7<a href="#pep-466-network-security-enhancements-for-python-2-7" class="headerlink" title="Permalink to this headline">¶</a>

<span id="index-17" class="target"></span><a href="https://www.python.org/dev/peps/pep-0466" class="pep reference external"><strong>PEP 466</strong></a> describes a number of network security enhancement proposals that have been approved for inclusion in Python 2.7 maintenance releases, with the first of those changes appearing in the Python 2.7.7 release.

<span id="index-18" class="target"></span><a href="https://www.python.org/dev/peps/pep-0466" class="pep reference external"><strong>PEP 466</strong></a> related features added in Python 2.7.7:

- <a href="../library/hmac.html#hmac.compare_digest" class="reference internal" title="hmac.compare_digest"><span class="pre"><code class="sourceCode python">hmac.compare_digest()</code></span></a> was backported from Python 3 to make a timing attack resistant comparison operation available to Python 2 applications. (Contributed by Alex Gaynor; <a href="https://bugs.python.org/issue21306" class="reference external">bpo-21306</a>.)

- OpenSSL 1.0.1g was upgraded in the official Windows installers published on python.org. (Contributed by Zachary Ware; <a href="https://bugs.python.org/issue21462" class="reference external">bpo-21462</a>.)

<span id="index-19" class="target"></span><a href="https://www.python.org/dev/peps/pep-0466" class="pep reference external"><strong>PEP 466</strong></a> related features added in Python 2.7.8:

- <a href="../library/hashlib.html#hashlib.pbkdf2_hmac" class="reference internal" title="hashlib.pbkdf2_hmac"><span class="pre"><code class="sourceCode python">hashlib.pbkdf2_hmac()</code></span></a> was backported from Python 3 to make a hashing algorithm suitable for secure password storage broadly available to Python 2 applications. (Contributed by Alex Gaynor; <a href="https://bugs.python.org/issue21304" class="reference external">bpo-21304</a>.)

- OpenSSL 1.0.1h was upgraded for the official Windows installers published on python.org. (contributed by Zachary Ware in <a href="https://bugs.python.org/issue21671" class="reference external">bpo-21671</a> for CVE-2014-0224)

<span id="index-20" class="target"></span><a href="https://www.python.org/dev/peps/pep-0466" class="pep reference external"><strong>PEP 466</strong></a> related features added in Python 2.7.9:

- Most of Python 3.4’s <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> module was backported. This means <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> now supports Server Name Indication, TLS1.x settings, access to the platform certificate store, the <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> class, and other features. (Contributed by Alex Gaynor and David Reid; <a href="https://bugs.python.org/issue21308" class="reference external">bpo-21308</a>.)

  Refer to the “Version added: 2.7.9” notes in the module documentation for specific details.

- <a href="../library/os.html#os.urandom" class="reference internal" title="os.urandom"><span class="pre"><code class="sourceCode python">os.urandom()</code></span></a> was changed to cache a file descriptor to <span class="pre">`/dev/urandom`</span> instead of reopening <span class="pre">`/dev/urandom`</span> on every call. (Contributed by Alex Gaynor; <a href="https://bugs.python.org/issue21305" class="reference external">bpo-21305</a>.)

- <a href="../library/hashlib.html#hashlib.algorithms_guaranteed" class="reference internal" title="hashlib.algorithms_guaranteed"><span class="pre"><code class="sourceCode python">hashlib.algorithms_guaranteed</code></span></a> and <a href="../library/hashlib.html#hashlib.algorithms_available" class="reference internal" title="hashlib.algorithms_available"><span class="pre"><code class="sourceCode python">hashlib.algorithms_available</code></span></a> were backported from Python 3 to make it easier for Python 2 applications to select the strongest available hash algorithm. (Contributed by Alex Gaynor in <a href="https://bugs.python.org/issue21307" class="reference external">bpo-21307</a>)

</div>

<div id="pep-477-backport-ensurepip-pep-453-to-python-2-7" class="section">

### PEP 477: Backport ensurepip (PEP 453) to Python 2.7<a href="#pep-477-backport-ensurepip-pep-453-to-python-2-7" class="headerlink" title="Permalink to this headline">¶</a>

<span id="index-21" class="target"></span><a href="https://www.python.org/dev/peps/pep-0477" class="pep reference external"><strong>PEP 477</strong></a> approves the inclusion of the <span id="index-22" class="target"></span><a href="https://www.python.org/dev/peps/pep-0453" class="pep reference external"><strong>PEP 453</strong></a> ensurepip module and the improved documentation that was enabled by it in the Python 2.7 maintenance releases, appearing first in the Python 2.7.9 release.

<div id="bootstrapping-pip-by-default" class="section">

#### Bootstrapping pip By Default<a href="#bootstrapping-pip-by-default" class="headerlink" title="Permalink to this headline">¶</a>

The new <a href="../library/ensurepip.html#module-ensurepip" class="reference internal" title="ensurepip: Bootstrapping the ``pip`` installer into an existing Python installation or virtual environment."><span class="pre"><code class="sourceCode python">ensurepip</code></span></a> module (defined in <span id="index-23" class="target"></span><a href="https://www.python.org/dev/peps/pep-0453" class="pep reference external"><strong>PEP 453</strong></a>) provides a standard cross-platform mechanism to bootstrap the pip installer into Python installations. The version of <span class="pre">`pip`</span> included with Python 2.7.9 is <span class="pre">`pip`</span> 1.5.6, and future 2.7.x maintenance releases will update the bundled version to the latest version of <span class="pre">`pip`</span> that is available at the time of creating the release candidate.

By default, the commands <span class="pre">`pip`</span>, <span class="pre">`pipX`</span> and <span class="pre">`pipX.Y`</span> will be installed on all platforms (where X.Y stands for the version of the Python installation), along with the <span class="pre">`pip`</span> Python package and its dependencies.

For CPython <a href="../using/unix.html#building-python-on-unix" class="reference internal"><span class="std std-ref">source builds on POSIX systems</span></a>, the <span class="pre">`make`</span>` `<span class="pre">`install`</span> and <span class="pre">`make`</span>` `<span class="pre">`altinstall`</span> commands do not bootstrap <span class="pre">`pip`</span> by default. This behaviour can be controlled through configure options, and overridden through Makefile options.

On Windows and Mac OS X, the CPython installers now default to installing <span class="pre">`pip`</span> along with CPython itself (users may opt out of installing it during the installation process). Window users will need to opt in to the automatic <span class="pre">`PATH`</span> modifications to have <span class="pre">`pip`</span> available from the command line by default, otherwise it can still be accessed through the Python launcher for Windows as <span class="pre">`py`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`pip`</span>.

As <a href="https://www.python.org/dev/peps/pep-0477/#disabling-ensurepip-by-downstream-distributors" class="reference external">discussed in the PEP</a>, platform packagers may choose not to install these commands by default, as long as, when invoked, they provide clear and simple directions on how to install them on that platform (usually using the system package manager).

</div>

<div id="documentation-changes" class="section">

#### Documentation Changes<a href="#documentation-changes" class="headerlink" title="Permalink to this headline">¶</a>

As part of this change, the <a href="../installing/index.html#installing-index" class="reference internal"><span class="std std-ref">Installing Python Modules</span></a> and <a href="../distributing/index.html#distributing-index" class="reference internal"><span class="std std-ref">Distributing Python Modules</span></a> sections of the documentation have been completely redesigned as short getting started and FAQ documents. Most packaging documentation has now been moved out to the Python Packaging Authority maintained <a href="http://packaging.python.org" class="reference external">Python Packaging User Guide</a> and the documentation of the individual projects.

However, as this migration is currently still incomplete, the legacy versions of those guides remaining available as <a href="../install/index.html#install-index" class="reference internal"><span class="std std-ref">Installing Python Modules (Legacy version)</span></a> and <a href="../distutils/index.html#distutils-index" class="reference internal"><span class="std std-ref">Distributing Python Modules (Legacy version)</span></a>.

<div class="admonition seealso">

See also

<span id="index-24" class="target"></span><a href="https://www.python.org/dev/peps/pep-0453" class="pep reference external"><strong>PEP 453</strong></a> – Explicit bootstrapping of pip in Python installations  
PEP written by Donald Stufft and Nick Coghlan, implemented by Donald Stufft, Nick Coghlan, Martin von Löwis and Ned Deily.

</div>

</div>

</div>

<div id="pep-476-enabling-certificate-verification-by-default-for-stdlib-http-clients" class="section">

### PEP 476: Enabling certificate verification by default for stdlib http clients<a href="#pep-476-enabling-certificate-verification-by-default-for-stdlib-http-clients" class="headerlink" title="Permalink to this headline">¶</a>

<span id="index-25" class="target"></span><a href="https://www.python.org/dev/peps/pep-0476" class="pep reference external"><strong>PEP 476</strong></a> updated <a href="../library/httplib.html#module-httplib" class="reference internal" title="httplib: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">httplib</code></span></a> and modules which use it, such as <a href="../library/urllib2.html#module-urllib2" class="reference internal" title="urllib2: Next generation URL opening library."><span class="pre"><code class="sourceCode python">urllib2</code></span></a> and <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a>, to now verify that the server presents a certificate which is signed by a Certificate Authority in the platform trust store and whose hostname matches the hostname being requested by default, significantly improving security for many applications. This change was made in the Python 2.7.9 release.

For applications which require the old previous behavior, they can pass an alternate context:

<div class="highlight-default notranslate">

<div class="highlight">

    import urllib2
    import ssl

    # This disables all verification
    context = ssl._create_unverified_context()

    # This allows using a specific certificate for the host, which doesn't need
    # to be in the trust store
    context = ssl.create_default_context(cafile="/path/to/file.crt")

    urllib2.urlopen("https://invalid-cert", context=context)

</div>

</div>

</div>

<div id="pep-493-https-verification-migration-tools-for-python-2-7" class="section">

### PEP 493: HTTPS verification migration tools for Python 2.7<a href="#pep-493-https-verification-migration-tools-for-python-2-7" class="headerlink" title="Permalink to this headline">¶</a>

<span id="index-26" class="target"></span><a href="https://www.python.org/dev/peps/pep-0493" class="pep reference external"><strong>PEP 493</strong></a> provides additional migration tools to support a more incremental infrastructure upgrade process for environments containing applications and services relying on the historically permissive processing of server certificates when establishing client HTTPS connections. These additions were made in the Python 2.7.12 release.

These tools are intended for use in cases where affected applications and services can’t be modified to explicitly pass a more permissive SSL context when establishing the connection.

For applications and services which can’t be modified at all, the new <span class="pre">`PYTHONHTTPSVERIFY`</span> environment variable may be set to <span class="pre">`0`</span> to revert an entire Python process back to the default permissive behaviour of Python 2.7.8 and earlier.

For cases where the connection establishment code can’t be modified, but the overall application can be, the new <a href="../library/ssl.html#ssl._https_verify_certificates" class="reference internal" title="ssl._https_verify_certificates"><span class="pre"><code class="sourceCode python">ssl._https_verify_certificates()</code></span></a> function can be used to adjust the default behaviour at runtime.

</div>

<div id="new-make-regen-all-build-target" class="section">

### New <span class="pre">`make`</span>` `<span class="pre">`regen-all`</span> build target<a href="#new-make-regen-all-build-target" class="headerlink" title="Permalink to this headline">¶</a>

To simplify cross-compilation, and to ensure that CPython can reliably be compiled without requiring an existing version of Python to already be available, the autotools-based build system no longer attempts to implicitly recompile generated files based on file modification times.

Instead, a new <span class="pre">`make`</span>` `<span class="pre">`regen-all`</span> command has been added to force regeneration of these files when desired (e.g. after an initial version of Python has already been built based on the pregenerated versions).

More selective regeneration targets are also defined - see <a href="https://github.com/python/cpython/tree/2.7/Makefile.pre.in" class="reference external">Makefile.pre.in</a> for details.

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue23404" class="reference external">bpo-23404</a>.)

<div class="versionadded">

<span class="versionmodified added">New in version 2.7.14.</span>

</div>

</div>

<div id="removal-of-make-touch-build-target" class="section">

### Removal of <span class="pre">`make`</span>` `<span class="pre">`touch`</span> build target<a href="#removal-of-make-touch-build-target" class="headerlink" title="Permalink to this headline">¶</a>

The <span class="pre">`make`</span>` `<span class="pre">`touch`</span> build target previously used to request implicit regeneration of generated files by updating their modification times has been removed.

It has been replaced by the new <span class="pre">`make`</span>` `<span class="pre">`regen-all`</span> target.

(Contributed by Victor Stinner in <a href="https://bugs.python.org/issue23404" class="reference external">bpo-23404</a>.)

<div class="versionchanged">

<span class="versionmodified changed">Changed in version 2.7.14.</span>

</div>

</div>

</div>

<div id="acknowledgements" class="section">

<span id="acks27"></span>

## Acknowledgements<a href="#acknowledgements" class="headerlink" title="Permalink to this headline">¶</a>

The author would like to thank the following people for offering suggestions, corrections and assistance with various drafts of this article: Nick Coghlan, Philip Jenvey, Ryan Lovett, R. David Murray, Hugh Secker-Walker.

</div>

</div>

</div>
