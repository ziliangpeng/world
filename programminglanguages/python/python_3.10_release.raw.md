<div class="body" role="main">

<div id="what-s-new-in-python-3-10" class="section">

# What’s New In Python 3.10<a href="#what-s-new-in-python-3-10" class="headerlink" title="Link to this heading">¶</a>

Editor<span class="colon">:</span>  
Pablo Galindo Salgado

This article explains the new features in Python 3.10, compared to 3.9. Python 3.10 was released on October 4, 2021. For full details, see the <a href="changelog.html#changelog" class="reference internal"><span class="std std-ref">changelog</span></a>.

<div id="summary-release-highlights" class="section">

## Summary – Release highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

New syntax features:

- <span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0634/" class="pep reference external"><strong>PEP 634</strong></a>, Structural Pattern Matching: Specification

- <span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0635/" class="pep reference external"><strong>PEP 635</strong></a>, Structural Pattern Matching: Motivation and Rationale

- <span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0636/" class="pep reference external"><strong>PEP 636</strong></a>, Structural Pattern Matching: Tutorial

- <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12782" class="reference external">bpo-12782</a>, Parenthesized context managers are now officially allowed.

New features in the standard library:

- <span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0618/" class="pep reference external"><strong>PEP 618</strong></a>, Add Optional Length-Checking To zip.

Interpreter improvements:

- <span id="index-4" class="target"></span><a href="https://peps.python.org/pep-0626/" class="pep reference external"><strong>PEP 626</strong></a>, Precise line numbers for debugging and other tools.

New typing features:

- <span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0604/" class="pep reference external"><strong>PEP 604</strong></a>, Allow writing union types as X \| Y

- <span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0612/" class="pep reference external"><strong>PEP 612</strong></a>, Parameter Specification Variables

- <span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0613/" class="pep reference external"><strong>PEP 613</strong></a>, Explicit Type Aliases

- <span id="index-8" class="target"></span><a href="https://peps.python.org/pep-0647/" class="pep reference external"><strong>PEP 647</strong></a>, User-Defined Type Guards

Important deprecations, removals or restrictions:

- <span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0644/" class="pep reference external"><strong>PEP 644</strong></a>, Require OpenSSL 1.1.1 or newer

- <span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0632/" class="pep reference external"><strong>PEP 632</strong></a>, Deprecate distutils module.

- <span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0623/" class="pep reference external"><strong>PEP 623</strong></a>, Deprecate and prepare for the removal of the wstr member in PyUnicodeObject.

- <span id="index-12" class="target"></span><a href="https://peps.python.org/pep-0624/" class="pep reference external"><strong>PEP 624</strong></a>, Remove Py_UNICODE encoder APIs

- <span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0597/" class="pep reference external"><strong>PEP 597</strong></a>, Add optional EncodingWarning

</div>

<div id="new-features" class="section">

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="parenthesized-context-managers" class="section">

<span id="whatsnew310-pep563"></span>

### Parenthesized context managers<a href="#parenthesized-context-managers" class="headerlink" title="Link to this heading">¶</a>

Using enclosing parentheses for continuation across multiple lines in context managers is now supported. This allows formatting a long collection of context managers in multiple lines in a similar way as it was previously possible with import statements. For instance, all these examples are now valid:

<div class="highlight-python notranslate">

<div class="highlight">

    with (CtxManager() as example):
        ...

    with (
        CtxManager1(),
        CtxManager2()
    ):
        ...

    with (CtxManager1() as example,
          CtxManager2()):
        ...

    with (CtxManager1(),
          CtxManager2() as example):
        ...

    with (
        CtxManager1() as example1,
        CtxManager2() as example2
    ):
        ...

</div>

</div>

it is also possible to use a trailing comma at the end of the enclosed group:

<div class="highlight-python notranslate">

<div class="highlight">

    with (
        CtxManager1() as example1,
        CtxManager2() as example2,
        CtxManager3() as example3,
    ):
        ...

</div>

</div>

This new syntax uses the non LL(1) capacities of the new parser. Check <span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0617/" class="pep reference external"><strong>PEP 617</strong></a> for more details.

(Contributed by Guido van Rossum, Pablo Galindo and Lysandros Nikolaou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12782" class="reference external">bpo-12782</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40334" class="reference external">bpo-40334</a>.)

</div>

<div id="better-error-messages" class="section">

### Better error messages<a href="#better-error-messages" class="headerlink" title="Link to this heading">¶</a>

<div id="syntaxerrors" class="section">

#### SyntaxErrors<a href="#syntaxerrors" class="headerlink" title="Link to this heading">¶</a>

When parsing code that contains unclosed parentheses or brackets the interpreter now includes the location of the unclosed bracket of parentheses instead of displaying *SyntaxError: unexpected EOF while parsing* or pointing to some incorrect location. For instance, consider the following code (notice the unclosed ‘{‘):

<div class="highlight-python notranslate">

<div class="highlight">

    expected = {9: 1, 18: 2, 19: 2, 27: 3, 28: 3, 29: 3, 36: 4, 37: 4,
                38: 4, 39: 4, 45: 5, 46: 5, 47: 5, 48: 5, 49: 5, 54: 6,
    some_other_code = foo()

</div>

</div>

Previous versions of the interpreter reported confusing places as the location of the syntax error:

<div class="highlight-python notranslate">

<div class="highlight">

    File "example.py", line 3
        some_other_code = foo()
                        ^
    SyntaxError: invalid syntax

</div>

</div>

but in Python 3.10 a more informative error is emitted:

<div class="highlight-python notranslate">

<div class="highlight">

    File "example.py", line 1
        expected = {9: 1, 18: 2, 19: 2, 27: 3, 28: 3, 29: 3, 36: 4, 37: 4,
                   ^
    SyntaxError: '{' was never closed

</div>

</div>

In a similar way, errors involving unclosed string literals (single and triple quoted) now point to the start of the string instead of reporting EOF/EOL.

These improvements are inspired by previous work in the PyPy interpreter.

(Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42864" class="reference external">bpo-42864</a> and Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40176" class="reference external">bpo-40176</a>.)

<a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> exceptions raised by the interpreter will now highlight the full error range of the expression that constitutes the syntax error itself, instead of just where the problem is detected. In this way, instead of displaying (before Python 3.10):

<div class="highlight-python notranslate">

<div class="highlight">

    >>> foo(x, z for z in range(10), t, w)
      File "<stdin>", line 1
        foo(x, z for z in range(10), t, w)
               ^
    SyntaxError: Generator expression must be parenthesized

</div>

</div>

now Python 3.10 will display the exception as:

<div class="highlight-python notranslate">

<div class="highlight">

    >>> foo(x, z for z in range(10), t, w)
      File "<stdin>", line 1
        foo(x, z for z in range(10), t, w)
               ^^^^^^^^^^^^^^^^^^^^
    SyntaxError: Generator expression must be parenthesized

</div>

</div>

This improvement was contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43914" class="reference external">bpo-43914</a>.

A considerable amount of new specialized messages for <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> exceptions have been incorporated. Some of the most notable ones are as follows:

- Missing <span class="pre">`:`</span> before blocks:

  <div class="highlight-python notranslate">

  <div class="highlight">

      >>> if rocket.position > event_horizon
        File "<stdin>", line 1
          if rocket.position > event_horizon
                                            ^
      SyntaxError: expected ':'

  </div>

  </div>

  (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42997" class="reference external">bpo-42997</a>.)

- Unparenthesised tuples in comprehensions targets:

  <div class="highlight-python notranslate">

  <div class="highlight">

      >>> {x,y for x,y in zip('abcd', '1234')}
        File "<stdin>", line 1
          {x,y for x,y in zip('abcd', '1234')}
           ^
      SyntaxError: did you forget parentheses around the comprehension target?

  </div>

  </div>

  (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43017" class="reference external">bpo-43017</a>.)

- Missing commas in collection literals and between expressions:

  <div class="highlight-python notranslate">

  <div class="highlight">

      >>> items = {
      ... x: 1,
      ... y: 2
      ... z: 3,
        File "<stdin>", line 3
          y: 2
             ^
      SyntaxError: invalid syntax. Perhaps you forgot a comma?

  </div>

  </div>

  (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43822" class="reference external">bpo-43822</a>.)

- Multiple Exception types without parentheses:

  <div class="highlight-python notranslate">

  <div class="highlight">

      >>> try:
      ...     build_dyson_sphere()
      ... except NotEnoughScienceError, NotEnoughResourcesError:
        File "<stdin>", line 3
          except NotEnoughScienceError, NotEnoughResourcesError:
                 ^
      SyntaxError: multiple exception types must be parenthesized

  </div>

  </div>

  (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43149" class="reference external">bpo-43149</a>.)

- Missing <span class="pre">`:`</span> and values in dictionary literals:

  <div class="highlight-python notranslate">

  <div class="highlight">

      >>> values = {
      ... x: 1,
      ... y: 2,
      ... z:
      ... }
        File "<stdin>", line 4
          z:
           ^
      SyntaxError: expression expected after dictionary key and ':'

      >>> values = {x:1, y:2, z w:3}
        File "<stdin>", line 1
          values = {x:1, y:2, z w:3}
                              ^
      SyntaxError: ':' expected after dictionary key

  </div>

  </div>

  (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43823" class="reference external">bpo-43823</a>.)

- <span class="pre">`try`</span> blocks without <span class="pre">`except`</span> or <span class="pre">`finally`</span> blocks:

  <div class="highlight-python notranslate">

  <div class="highlight">

      >>> try:
      ...     x = 2
      ... something = 3
        File "<stdin>", line 3
          something  = 3
          ^^^^^^^^^
      SyntaxError: expected 'except' or 'finally' block

  </div>

  </div>

  (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44305" class="reference external">bpo-44305</a>.)

- Usage of <span class="pre">`=`</span> instead of <span class="pre">`==`</span> in comparisons:

  <div class="highlight-python notranslate">

  <div class="highlight">

      >>> if rocket.position = event_horizon:
        File "<stdin>", line 1
          if rocket.position = event_horizon:
                             ^
      SyntaxError: cannot assign to attribute here. Maybe you meant '==' instead of '='?

  </div>

  </div>

  (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43797" class="reference external">bpo-43797</a>.)

- Usage of <span class="pre">`*`</span> in f-strings:

  <div class="highlight-python notranslate">

  <div class="highlight">

      >>> f"Black holes {*all_black_holes} and revelations"
        File "<stdin>", line 1
          (*all_black_holes)
           ^
      SyntaxError: f-string: cannot use starred expression here

  </div>

  </div>

  (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41064" class="reference external">bpo-41064</a>.)

</div>

<div id="indentationerrors" class="section">

#### IndentationErrors<a href="#indentationerrors" class="headerlink" title="Link to this heading">¶</a>

Many <a href="../library/exceptions.html#IndentationError" class="reference internal" title="IndentationError"><span class="pre"><code class="sourceCode python"><span class="pp">IndentationError</span></code></span></a> exceptions now have more context regarding what kind of block was expecting an indentation, including the location of the statement:

<div class="highlight-python notranslate">

<div class="highlight">

    >>> def foo():
    ...    if lel:
    ...    x = 2
      File "<stdin>", line 3
        x = 2
        ^
    IndentationError: expected an indented block after 'if' statement in line 2

</div>

</div>

</div>

<div id="attributeerrors" class="section">

#### AttributeErrors<a href="#attributeerrors" class="headerlink" title="Link to this heading">¶</a>

When printing <a href="../library/exceptions.html#AttributeError" class="reference internal" title="AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a>, <span class="pre">`PyErr_Display()`</span> will offer suggestions of similar attribute names in the object that the exception was raised from:

<div class="highlight-python notranslate">

<div class="highlight">

    >>> collections.namedtoplo
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    AttributeError: module 'collections' has no attribute 'namedtoplo'. Did you mean: namedtuple?

</div>

</div>

(Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38530" class="reference external">bpo-38530</a>.)

<div class="admonition warning">

Warning

Notice this won’t work if <span class="pre">`PyErr_Display()`</span> is not called to display the error which can happen if some other custom error display function is used. This is a common scenario in some REPLs like IPython.

</div>

</div>

<div id="nameerrors" class="section">

#### NameErrors<a href="#nameerrors" class="headerlink" title="Link to this heading">¶</a>

When printing <a href="../library/exceptions.html#NameError" class="reference internal" title="NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a> raised by the interpreter, <span class="pre">`PyErr_Display()`</span> will offer suggestions of similar variable names in the function that the exception was raised from:

<div class="highlight-python notranslate">

<div class="highlight">

    >>> schwarzschild_black_hole = None
    >>> schwarschild_black_hole
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    NameError: name 'schwarschild_black_hole' is not defined. Did you mean: schwarzschild_black_hole?

</div>

</div>

(Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38530" class="reference external">bpo-38530</a>.)

<div class="admonition warning">

Warning

Notice this won’t work if <span class="pre">`PyErr_Display()`</span> is not called to display the error, which can happen if some other custom error display function is used. This is a common scenario in some REPLs like IPython.

</div>

</div>

</div>

<div id="pep-626-precise-line-numbers-for-debugging-and-other-tools" class="section">

### PEP 626: Precise line numbers for debugging and other tools<a href="#pep-626-precise-line-numbers-for-debugging-and-other-tools" class="headerlink" title="Link to this heading">¶</a>

PEP 626 brings more precise and reliable line numbers for debugging, profiling and coverage tools. Tracing events, with the correct line number, are generated for all lines of code executed and only for lines of code that are executed.

The <a href="../reference/datamodel.html#frame.f_lineno" class="reference internal" title="frame.f_lineno"><span class="pre"><code class="sourceCode python">f_lineno</code></span></a> attribute of frame objects will always contain the expected line number.

The <a href="../reference/datamodel.html#codeobject.co_lnotab" class="reference internal" title="codeobject.co_lnotab"><span class="pre"><code class="sourceCode python">co_lnotab</code></span></a> attribute of <a href="../reference/datamodel.html#code-objects" class="reference internal"><span class="std std-ref">code objects</span></a> is deprecated and will be removed in 3.12. Code that needs to convert from offset to line number should use the new <a href="../reference/datamodel.html#codeobject.co_lines" class="reference internal" title="codeobject.co_lines"><span class="pre"><code class="sourceCode python">co_lines()</code></span></a> method instead.

</div>

<div id="pep-634-structural-pattern-matching" class="section">

### PEP 634: Structural Pattern Matching<a href="#pep-634-structural-pattern-matching" class="headerlink" title="Link to this heading">¶</a>

Structural pattern matching has been added in the form of a *match statement* and *case statements* of patterns with associated actions. Patterns consist of sequences, mappings, primitive data types as well as class instances. Pattern matching enables programs to extract information from complex data types, branch on the structure of data, and apply specific actions based on different forms of data.

<div id="syntax-and-operations" class="section">

#### Syntax and operations<a href="#syntax-and-operations" class="headerlink" title="Link to this heading">¶</a>

The generic syntax of pattern matching is:

<div class="highlight-python3 notranslate">

<div class="highlight">

    match subject:
        case <pattern_1>:
            <action_1>
        case <pattern_2>:
            <action_2>
        case <pattern_3>:
            <action_3>
        case _:
            <action_wildcard>

</div>

</div>

A match statement takes an expression and compares its value to successive patterns given as one or more case blocks. Specifically, pattern matching operates by:

1.  using data with type and shape (the <span class="pre">`subject`</span>)

2.  evaluating the <span class="pre">`subject`</span> in the <span class="pre">`match`</span> statement

3.  comparing the subject with each pattern in a <span class="pre">`case`</span> statement from top to bottom until a match is confirmed.

4.  executing the action associated with the pattern of the confirmed match

5.  If an exact match is not confirmed, the last case, a wildcard <span class="pre">`_`</span>, if provided, will be used as the matching case. If an exact match is not confirmed and a wildcard case does not exist, the entire match block is a no-op.

</div>

<div id="declarative-approach" class="section">

#### Declarative approach<a href="#declarative-approach" class="headerlink" title="Link to this heading">¶</a>

Readers may be aware of pattern matching through the simple example of matching a subject (data object) to a literal (pattern) with the switch statement found in C, Java or JavaScript (and many other languages). Often the switch statement is used for comparison of an object/expression with case statements containing literals.

More powerful examples of pattern matching can be found in languages such as Scala and Elixir. With structural pattern matching, the approach is “declarative” and explicitly states the conditions (the patterns) for data to match.

While an “imperative” series of instructions using nested “if” statements could be used to accomplish something similar to structural pattern matching, it is less clear than the “declarative” approach. Instead the “declarative” approach states the conditions to meet for a match and is more readable through its explicit patterns. While structural pattern matching can be used in its simplest form comparing a variable to a literal in a case statement, its true value for Python lies in its handling of the subject’s type and shape.

</div>

<div id="simple-pattern-match-to-a-literal" class="section">

#### Simple pattern: match to a literal<a href="#simple-pattern-match-to-a-literal" class="headerlink" title="Link to this heading">¶</a>

Let’s look at this example as pattern matching in its simplest form: a value, the subject, being matched to several literals, the patterns. In the example below, <span class="pre">`status`</span> is the subject of the match statement. The patterns are each of the case statements, where literals represent request status codes. The associated action to the case is executed after a match:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def http_error(status):
        match status:
            case 400:
                return "Bad request"
            case 404:
                return "Not found"
            case 418:
                return "I'm a teapot"
            case _:
                return "Something's wrong with the internet"

</div>

</div>

If the above function is passed a <span class="pre">`status`</span> of 418, “I’m a teapot” is returned. If the above function is passed a <span class="pre">`status`</span> of 500, the case statement with <span class="pre">`_`</span> will match as a wildcard, and “Something’s wrong with the internet” is returned. Note the last block: the variable name, <span class="pre">`_`</span>, acts as a *wildcard* and insures the subject will always match. The use of <span class="pre">`_`</span> is optional.

You can combine several literals in a single pattern using <span class="pre">`|`</span> (“or”):

<div class="highlight-python3 notranslate">

<div class="highlight">

    case 401 | 403 | 404:
        return "Not allowed"

</div>

</div>

<div id="behavior-without-the-wildcard" class="section">

##### Behavior without the wildcard<a href="#behavior-without-the-wildcard" class="headerlink" title="Link to this heading">¶</a>

If we modify the above example by removing the last case block, the example becomes:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def http_error(status):
        match status:
            case 400:
                return "Bad request"
            case 404:
                return "Not found"
            case 418:
                return "I'm a teapot"

</div>

</div>

Without the use of <span class="pre">`_`</span> in a case statement, a match may not exist. If no match exists, the behavior is a no-op. For example, if <span class="pre">`status`</span> of 500 is passed, a no-op occurs.

</div>

</div>

<div id="patterns-with-a-literal-and-variable" class="section">

#### Patterns with a literal and variable<a href="#patterns-with-a-literal-and-variable" class="headerlink" title="Link to this heading">¶</a>

Patterns can look like unpacking assignments, and a pattern may be used to bind variables. In this example, a data point can be unpacked to its x-coordinate and y-coordinate:

<div class="highlight-python3 notranslate">

<div class="highlight">

    # point is an (x, y) tuple
    match point:
        case (0, 0):
            print("Origin")
        case (0, y):
            print(f"Y={y}")
        case (x, 0):
            print(f"X={x}")
        case (x, y):
            print(f"X={x}, Y={y}")
        case _:
            raise ValueError("Not a point")

</div>

</div>

The first pattern has two literals, <span class="pre">`(0,`</span>` `<span class="pre">`0)`</span>, and may be thought of as an extension of the literal pattern shown above. The next two patterns combine a literal and a variable, and the variable *binds* a value from the subject (<span class="pre">`point`</span>). The fourth pattern captures two values, which makes it conceptually similar to the unpacking assignment <span class="pre">`(x,`</span>` `<span class="pre">`y)`</span>` `<span class="pre">`=`</span>` `<span class="pre">`point`</span>.

</div>

<div id="patterns-and-classes" class="section">

#### Patterns and classes<a href="#patterns-and-classes" class="headerlink" title="Link to this heading">¶</a>

If you are using classes to structure your data, you can use as a pattern the class name followed by an argument list resembling a constructor. This pattern has the ability to capture instance attributes into variables:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def location(point):
        match point:
            case Point(x=0, y=0):
                print("Origin is the point's location.")
            case Point(x=0, y=y):
                print(f"Y={y} and the point is on the y-axis.")
            case Point(x=x, y=0):
                print(f"X={x} and the point is on the x-axis.")
            case Point():
                print("The point is located somewhere else on the plane.")
            case _:
                print("Not a point")

</div>

</div>

<div id="patterns-with-positional-parameters" class="section">

##### Patterns with positional parameters<a href="#patterns-with-positional-parameters" class="headerlink" title="Link to this heading">¶</a>

You can use positional parameters with some builtin classes that provide an ordering for their attributes (e.g. dataclasses). You can also define a specific position for attributes in patterns by setting the <span class="pre">`__match_args__`</span> special attribute in your classes. If it’s set to (“x”, “y”), the following patterns are all equivalent (and all bind the <span class="pre">`y`</span> attribute to the <span class="pre">`var`</span> variable):

<div class="highlight-python3 notranslate">

<div class="highlight">

    Point(1, var)
    Point(1, y=var)
    Point(x=1, y=var)
    Point(y=var, x=1)

</div>

</div>

</div>

</div>

<div id="nested-patterns" class="section">

#### Nested patterns<a href="#nested-patterns" class="headerlink" title="Link to this heading">¶</a>

Patterns can be arbitrarily nested. For example, if our data is a short list of points, it could be matched like this:

<div class="highlight-python3 notranslate">

<div class="highlight">

    match points:
        case []:
            print("No points in the list.")
        case [Point(0, 0)]:
            print("The origin is the only point in the list.")
        case [Point(x, y)]:
            print(f"A single point {x}, {y} is in the list.")
        case [Point(0, y1), Point(0, y2)]:
            print(f"Two points on the Y axis at {y1}, {y2} are in the list.")
        case _:
            print("Something else is found in the list.")

</div>

</div>

</div>

<div id="complex-patterns-and-the-wildcard" class="section">

#### Complex patterns and the wildcard<a href="#complex-patterns-and-the-wildcard" class="headerlink" title="Link to this heading">¶</a>

To this point, the examples have used <span class="pre">`_`</span> alone in the last case statement. A wildcard can be used in more complex patterns, such as <span class="pre">`('error',`</span>` `<span class="pre">`code,`</span>` `<span class="pre">`_)`</span>. For example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    match test_variable:
        case ('warning', code, 40):
            print("A warning has been received.")
        case ('error', code, _):
            print(f"An error {code} occurred.")

</div>

</div>

In the above case, <span class="pre">`test_variable`</span> will match for (‘error’, code, 100) and (‘error’, code, 800).

</div>

<div id="guard" class="section">

#### Guard<a href="#guard" class="headerlink" title="Link to this heading">¶</a>

We can add an <span class="pre">`if`</span> clause to a pattern, known as a “guard”. If the guard is false, <span class="pre">`match`</span> goes on to try the next case block. Note that value capture happens before the guard is evaluated:

<div class="highlight-python3 notranslate">

<div class="highlight">

    match point:
        case Point(x, y) if x == y:
            print(f"The point is located on the diagonal Y=X at {x}.")
        case Point(x, y):
            print(f"Point is not on the diagonal.")

</div>

</div>

</div>

<div id="other-key-features" class="section">

#### Other Key Features<a href="#other-key-features" class="headerlink" title="Link to this heading">¶</a>

Several other key features:

- Like unpacking assignments, tuple and list patterns have exactly the same meaning and actually match arbitrary sequences. Technically, the subject must be a sequence. Therefore, an important exception is that patterns don’t match iterators. Also, to prevent a common mistake, sequence patterns don’t match strings.

- Sequence patterns support wildcards: <span class="pre">`[x,`</span>` `<span class="pre">`y,`</span>` `<span class="pre">`*rest]`</span> and <span class="pre">`(x,`</span>` `<span class="pre">`y,`</span>` `<span class="pre">`*rest)`</span> work similar to wildcards in unpacking assignments. The name after <span class="pre">`*`</span> may also be <span class="pre">`_`</span>, so <span class="pre">`(x,`</span>` `<span class="pre">`y,`</span>` `<span class="pre">`*_)`</span> matches a sequence of at least two items without binding the remaining items.

- Mapping patterns: <span class="pre">`{"bandwidth":`</span>` `<span class="pre">`b,`</span>` `<span class="pre">`"latency":`</span>` `<span class="pre">`l}`</span> captures the <span class="pre">`"bandwidth"`</span> and <span class="pre">`"latency"`</span> values from a dict. Unlike sequence patterns, extra keys are ignored. A wildcard <span class="pre">`**rest`</span> is also supported. (But <span class="pre">`**_`</span> would be redundant, so is not allowed.)

- Subpatterns may be captured using the <span class="pre">`as`</span> keyword:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      case (Point(x1, y1), Point(x2, y2) as p2): ...

  </div>

  </div>

  This binds x1, y1, x2, y2 like you would expect without the <span class="pre">`as`</span> clause, and p2 to the entire second item of the subject.

- Most literals are compared by equality. However, the singletons <span class="pre">`True`</span>, <span class="pre">`False`</span> and <span class="pre">`None`</span> are compared by identity.

- Named constants may be used in patterns. These named constants must be dotted names to prevent the constant from being interpreted as a capture variable:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      from enum import Enum
      class Color(Enum):
          RED = 0
          GREEN = 1
          BLUE = 2

      color = Color.GREEN
      match color:
          case Color.RED:
              print("I see red!")
          case Color.GREEN:
              print("Grass is green")
          case Color.BLUE:
              print("I'm feeling the blues :(")

  </div>

  </div>

For the full specification see <span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0634/" class="pep reference external"><strong>PEP 634</strong></a>. Motivation and rationale are in <span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0635/" class="pep reference external"><strong>PEP 635</strong></a>, and a longer tutorial is in <span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0636/" class="pep reference external"><strong>PEP 636</strong></a>.

</div>

</div>

<div id="optional-encodingwarning-and-encoding-locale-option" class="section">

<span id="whatsnew310-pep597"></span>

### Optional <span class="pre">`EncodingWarning`</span> and <span class="pre">`encoding="locale"`</span> option<a href="#optional-encodingwarning-and-encoding-locale-option" class="headerlink" title="Link to this heading">¶</a>

The default encoding of <a href="../library/io.html#io.TextIOWrapper" class="reference internal" title="io.TextIOWrapper"><span class="pre"><code class="sourceCode python">TextIOWrapper</code></span></a> and <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a> is platform and locale dependent. Since UTF-8 is used on most Unix platforms, omitting <span class="pre">`encoding`</span> option when opening UTF-8 files (e.g. JSON, YAML, TOML, Markdown) is a very common bug. For example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    # BUG: "rb" mode or encoding="utf-8" should be used.
    with open("data.json") as f:
        data = json.load(f)

</div>

</div>

To find this type of bug, an optional <span class="pre">`EncodingWarning`</span> is added. It is emitted when <a href="../library/sys.html#sys.flags" class="reference internal" title="sys.flags"><span class="pre"><code class="sourceCode python">sys.flags.warn_default_encoding</code></span></a> is true and locale-specific default encoding is used.

<span class="pre">`-X`</span>` `<span class="pre">`warn_default_encoding`</span> option and <span id="index-18" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONWARNDEFAULTENCODING" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONWARNDEFAULTENCODING</code></span></a> are added to enable the warning.

See <a href="../library/io.html#io-text-encoding" class="reference internal"><span class="std std-ref">Text Encoding</span></a> for more information.

</div>

</div>

<div id="new-features-related-to-type-hints" class="section">

<span id="new-feat-related-type-hints"></span>

## New Features Related to Type Hints<a href="#new-features-related-to-type-hints" class="headerlink" title="Link to this heading">¶</a>

This section covers major changes affecting <span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> type hints and the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module.

<div id="pep-604-new-type-union-operator" class="section">

### PEP 604: New Type Union Operator<a href="#pep-604-new-type-union-operator" class="headerlink" title="Link to this heading">¶</a>

A new type union operator was introduced which enables the syntax <span class="pre">`X`</span>` `<span class="pre">`|`</span>` `<span class="pre">`Y`</span>. This provides a cleaner way of expressing ‘either type X or type Y’ instead of using <a href="../library/typing.html#typing.Union" class="reference internal" title="typing.Union"><span class="pre"><code class="sourceCode python">typing.Union</code></span></a>, especially in type hints.

In previous versions of Python, to apply a type hint for functions accepting arguments of multiple types, <a href="../library/typing.html#typing.Union" class="reference internal" title="typing.Union"><span class="pre"><code class="sourceCode python">typing.Union</code></span></a> was used:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def square(number: Union[int, float]) -> Union[int, float]:
        return number ** 2

</div>

</div>

Type hints can now be written in a more succinct manner:

<div class="highlight-python3 notranslate">

<div class="highlight">

    def square(number: int | float) -> int | float:
        return number ** 2

</div>

</div>

This new syntax is also accepted as the second argument to <a href="../library/functions.html#isinstance" class="reference internal" title="isinstance"><span class="pre"><code class="sourceCode python"><span class="bu">isinstance</span>()</code></span></a> and <a href="../library/functions.html#issubclass" class="reference internal" title="issubclass"><span class="pre"><code class="sourceCode python"><span class="bu">issubclass</span>()</code></span></a>:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> isinstance(1, int | str)
    True

</div>

</div>

See <a href="../library/stdtypes.html#types-union" class="reference internal"><span class="std std-ref">Union Type</span></a> and <span id="index-20" class="target"></span><a href="https://peps.python.org/pep-0604/" class="pep reference external"><strong>PEP 604</strong></a> for more details.

(Contributed by Maggie Moss and Philippe Prados in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41428" class="reference external">bpo-41428</a>, with additions by Yurii Karabas and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44490" class="reference external">bpo-44490</a>.)

</div>

<div id="pep-612-parameter-specification-variables" class="section">

### PEP 612: Parameter Specification Variables<a href="#pep-612-parameter-specification-variables" class="headerlink" title="Link to this heading">¶</a>

Two new options to improve the information provided to static type checkers for <span id="index-21" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a>‘s <span class="pre">`Callable`</span> have been added to the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module.

The first is the parameter specification variable. They are used to forward the parameter types of one callable to another callable – a pattern commonly found in higher order functions and decorators. Examples of usage can be found in <a href="../library/typing.html#typing.ParamSpec" class="reference internal" title="typing.ParamSpec"><span class="pre"><code class="sourceCode python">typing.ParamSpec</code></span></a>. Previously, there was no easy way to type annotate dependency of parameter types in such a precise manner.

The second option is the new <span class="pre">`Concatenate`</span> operator. It’s used in conjunction with parameter specification variables to type annotate a higher order callable which adds or removes parameters of another callable. Examples of usage can be found in <a href="../library/typing.html#typing.Concatenate" class="reference internal" title="typing.Concatenate"><span class="pre"><code class="sourceCode python">typing.Concatenate</code></span></a>.

See <a href="../library/typing.html#typing.Callable" class="reference internal" title="typing.Callable"><span class="pre"><code class="sourceCode python">typing.Callable</code></span></a>, <a href="../library/typing.html#typing.ParamSpec" class="reference internal" title="typing.ParamSpec"><span class="pre"><code class="sourceCode python">typing.ParamSpec</code></span></a>, <a href="../library/typing.html#typing.Concatenate" class="reference internal" title="typing.Concatenate"><span class="pre"><code class="sourceCode python">typing.Concatenate</code></span></a>, <a href="../library/typing.html#typing.ParamSpecArgs" class="reference internal" title="typing.ParamSpecArgs"><span class="pre"><code class="sourceCode python">typing.ParamSpecArgs</code></span></a>, <a href="../library/typing.html#typing.ParamSpecKwargs" class="reference internal" title="typing.ParamSpecKwargs"><span class="pre"><code class="sourceCode python">typing.ParamSpecKwargs</code></span></a>, and <span id="index-22" class="target"></span><a href="https://peps.python.org/pep-0612/" class="pep reference external"><strong>PEP 612</strong></a> for more details.

(Contributed by Ken Jin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41559" class="reference external">bpo-41559</a>, with minor enhancements by Jelle Zijlstra in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43783" class="reference external">bpo-43783</a>. PEP written by Mark Mendoza.)

</div>

<div id="pep-613-typealias" class="section">

### PEP 613: TypeAlias<a href="#pep-613-typealias" class="headerlink" title="Link to this heading">¶</a>

<span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0484/" class="pep reference external"><strong>PEP 484</strong></a> introduced the concept of type aliases, only requiring them to be top-level unannotated assignments. This simplicity sometimes made it difficult for type checkers to distinguish between type aliases and ordinary assignments, especially when forward references or invalid types were involved. Compare:

<div class="highlight-python3 notranslate">

<div class="highlight">

    StrCache = 'Cache[str]'  # a type alias
    LOG_PREFIX = 'LOG[DEBUG]'  # a module constant

</div>

</div>

Now the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module has a special value <a href="../library/typing.html#typing.TypeAlias" class="reference internal" title="typing.TypeAlias"><span class="pre"><code class="sourceCode python">TypeAlias</code></span></a> which lets you declare type aliases more explicitly:

<div class="highlight-python3 notranslate">

<div class="highlight">

    StrCache: TypeAlias = 'Cache[str]'  # a type alias
    LOG_PREFIX = 'LOG[DEBUG]'  # a module constant

</div>

</div>

See <span id="index-24" class="target"></span><a href="https://peps.python.org/pep-0613/" class="pep reference external"><strong>PEP 613</strong></a> for more details.

(Contributed by Mikhail Golubev in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41923" class="reference external">bpo-41923</a>.)

</div>

<div id="pep-647-user-defined-type-guards" class="section">

### PEP 647: User-Defined Type Guards<a href="#pep-647-user-defined-type-guards" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/typing.html#typing.TypeGuard" class="reference internal" title="typing.TypeGuard"><span class="pre"><code class="sourceCode python">TypeGuard</code></span></a> has been added to the <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> module to annotate type guard functions and improve information provided to static type checkers during type narrowing. For more information, please see <a href="../library/typing.html#typing.TypeGuard" class="reference internal" title="typing.TypeGuard"><span class="pre"><code class="sourceCode python">TypeGuard</code></span></a>‘s documentation, and <span id="index-25" class="target"></span><a href="https://peps.python.org/pep-0647/" class="pep reference external"><strong>PEP 647</strong></a>.

(Contributed by Ken Jin and Guido van Rossum in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43766" class="reference external">bpo-43766</a>. PEP written by Eric Traut.)

</div>

</div>

<div id="other-language-changes" class="section">

## Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> type has a new method <a href="../library/stdtypes.html#int.bit_count" class="reference internal" title="int.bit_count"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>.bit_count()</code></span></a>, returning the number of ones in the binary expansion of a given integer, also known as the population count. (Contributed by Niklas Fiekas in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=29882" class="reference external">bpo-29882</a>.)

- The views returned by <a href="../library/stdtypes.html#dict.keys" class="reference internal" title="dict.keys"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.keys()</code></span></a>, <a href="../library/stdtypes.html#dict.values" class="reference internal" title="dict.values"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.values()</code></span></a> and <a href="../library/stdtypes.html#dict.items" class="reference internal" title="dict.items"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>.items()</code></span></a> now all have a <span class="pre">`mapping`</span> attribute that gives a <a href="../library/types.html#types.MappingProxyType" class="reference internal" title="types.MappingProxyType"><span class="pre"><code class="sourceCode python">types.MappingProxyType</code></span></a> object wrapping the original dictionary. (Contributed by Dennis Sweeney in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40890" class="reference external">bpo-40890</a>.)

- <span id="index-26" class="target"></span><a href="https://peps.python.org/pep-0618/" class="pep reference external"><strong>PEP 618</strong></a>: The <a href="../library/functions.html#zip" class="reference internal" title="zip"><span class="pre"><code class="sourceCode python"><span class="bu">zip</span>()</code></span></a> function now has an optional <span class="pre">`strict`</span> flag, used to require that all the iterables have an equal length.

- Builtin and extension functions that take integer arguments no longer accept <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">Decimal</code></span></a>s, <a href="../library/fractions.html#fractions.Fraction" class="reference internal" title="fractions.Fraction"><span class="pre"><code class="sourceCode python">Fraction</code></span></a>s and other objects that can be converted to integers only with a loss (e.g. that have the <a href="../reference/datamodel.html#object.__int__" class="reference internal" title="object.__int__"><span class="pre"><code class="sourceCode python"><span class="fu">__int__</span>()</code></span></a> method but do not have the <a href="../reference/datamodel.html#object.__index__" class="reference internal" title="object.__index__"><span class="pre"><code class="sourceCode python"><span class="fu">__index__</span>()</code></span></a> method). (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37999" class="reference external">bpo-37999</a>.)

- If <a href="../reference/datamodel.html#object.__ipow__" class="reference internal" title="object.__ipow__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__ipow__</span>()</code></span></a> returns <a href="../library/constants.html#NotImplemented" class="reference internal" title="NotImplemented"><span class="pre"><code class="sourceCode python"><span class="va">NotImplemented</span></code></span></a>, the operator will correctly fall back to <a href="../reference/datamodel.html#object.__pow__" class="reference internal" title="object.__pow__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__pow__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__rpow__" class="reference internal" title="object.__rpow__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__rpow__</span>()</code></span></a> as expected. (Contributed by Alex Shkop in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38302" class="reference external">bpo-38302</a>.)

- Assignment expressions can now be used unparenthesized within set literals and set comprehensions, as well as in sequence indexes (but not slices).

- Functions have a new <span class="pre">`__builtins__`</span> attribute which is used to look for builtin symbols when a function is executed, instead of looking into <span class="pre">`__globals__['__builtins__']`</span>. The attribute is initialized from <span class="pre">`__globals__["__builtins__"]`</span> if it exists, else from the current builtins. (Contributed by Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42990" class="reference external">bpo-42990</a>.)

- Two new builtin functions – <a href="../library/functions.html#aiter" class="reference internal" title="aiter"><span class="pre"><code class="sourceCode python"><span class="bu">aiter</span>()</code></span></a> and <a href="../library/functions.html#anext" class="reference internal" title="anext"><span class="pre"><code class="sourceCode python"><span class="bu">anext</span>()</code></span></a> have been added to provide asynchronous counterparts to <a href="../library/functions.html#iter" class="reference internal" title="iter"><span class="pre"><code class="sourceCode python"><span class="bu">iter</span>()</code></span></a> and <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a>, respectively. (Contributed by Joshua Bronson, Daniel Pope, and Justin Wang in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31861" class="reference external">bpo-31861</a>.)

- Static methods (<a href="../library/functions.html#staticmethod" class="reference internal" title="staticmethod"><span class="pre"><code class="sourceCode python"><span class="at">@staticmethod</span></code></span></a>) and class methods (<a href="../library/functions.html#classmethod" class="reference internal" title="classmethod"><span class="pre"><code class="sourceCode python"><span class="at">@classmethod</span></code></span></a>) now inherit the method attributes (<span class="pre">`__module__`</span>, <span class="pre">`__name__`</span>, <span class="pre">`__qualname__`</span>, <span class="pre">`__doc__`</span>, <span class="pre">`__annotations__`</span>) and have a new <span class="pre">`__wrapped__`</span> attribute. Moreover, static methods are now callable as regular functions. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43682" class="reference external">bpo-43682</a>.)

- Annotations for complex targets (everything beside <span class="pre">`simple`</span>` `<span class="pre">`name`</span> targets defined by <span id="index-27" class="target"></span><a href="https://peps.python.org/pep-0526/" class="pep reference external"><strong>PEP 526</strong></a>) no longer cause any runtime effects with <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`annotations`</span>. (Contributed by Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42737" class="reference external">bpo-42737</a>.)

- Class and module objects now lazy-create empty annotations dicts on demand. The annotations dicts are stored in the object’s <span class="pre">`__dict__`</span> for backwards compatibility. This improves the best practices for working with <span class="pre">`__annotations__`</span>; for more information, please see <a href="../howto/annotations.html#annotations-howto" class="reference internal"><span class="std std-ref">Annotations Best Practices</span></a>. (Contributed by Larry Hastings in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43901" class="reference external">bpo-43901</a>.)

- Annotations consist of <span class="pre">`yield`</span>, <span class="pre">`yield`</span>` `<span class="pre">`from`</span>, <span class="pre">`await`</span> or named expressions are now forbidden under <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`annotations`</span> due to their side effects. (Contributed by Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42725" class="reference external">bpo-42725</a>.)

- Usage of unbound variables, <span class="pre">`super()`</span> and other expressions that might alter the processing of symbol table as annotations are now rendered effectless under <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`annotations`</span>. (Contributed by Batuhan Taskaya in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42725" class="reference external">bpo-42725</a>.)

- Hashes of NaN values of both <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> type and <a href="../library/decimal.html#decimal.Decimal" class="reference internal" title="decimal.Decimal"><span class="pre"><code class="sourceCode python">decimal.Decimal</code></span></a> type now depend on object identity. Formerly, they always hashed to <span class="pre">`0`</span> even though NaN values are not equal to one another. This caused potentially quadratic runtime behavior due to excessive hash collisions when creating dictionaries and sets containing multiple NaNs. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43475" class="reference external">bpo-43475</a>.)

- A <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> (instead of a <a href="../library/exceptions.html#NameError" class="reference internal" title="NameError"><span class="pre"><code class="sourceCode python"><span class="pp">NameError</span></code></span></a>) will be raised when deleting the <a href="../library/constants.html#debug__" class="reference internal" title="__debug__"><span class="pre"><code class="sourceCode python"><span class="va">__debug__</span></code></span></a> constant. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45000" class="reference external">bpo-45000</a>.)

- <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> exceptions now have <span class="pre">`end_lineno`</span> and <span class="pre">`end_offset`</span> attributes. They will be <span class="pre">`None`</span> if not determined. (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43914" class="reference external">bpo-43914</a>.)

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

- None.

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="asyncio" class="section">

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

Add missing <a href="../library/asyncio-eventloop.html#asyncio.loop.connect_accepted_socket" class="reference internal" title="asyncio.loop.connect_accepted_socket"><span class="pre"><code class="sourceCode python">connect_accepted_socket()</code></span></a> method. (Contributed by Alex Grönholm in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41332" class="reference external">bpo-41332</a>.)

</div>

<div id="argparse" class="section">

### argparse<a href="#argparse" class="headerlink" title="Link to this heading">¶</a>

Misleading phrase “optional arguments” was replaced with “options” in argparse help. Some tests might require adaptation if they rely on exact output match. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9694" class="reference external">bpo-9694</a>.)

</div>

<div id="array" class="section">

### array<a href="#array" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/array.html#array.array.index" class="reference internal" title="array.array.index"><span class="pre"><code class="sourceCode python">index()</code></span></a> method of <a href="../library/array.html#array.array" class="reference internal" title="array.array"><span class="pre"><code class="sourceCode python">array.array</code></span></a> now has optional *start* and *stop* parameters. (Contributed by Anders Lorentsen and Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31956" class="reference external">bpo-31956</a>.)

</div>

<div id="asynchat-asyncore-smtpd" class="section">

### asynchat, asyncore, smtpd<a href="#asynchat-asyncore-smtpd" class="headerlink" title="Link to this heading">¶</a>

These modules have been marked as deprecated in their module documentation since Python 3.6. An import-time <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> has now been added to all three of these modules.

</div>

<div id="base64" class="section">

### base64<a href="#base64" class="headerlink" title="Link to this heading">¶</a>

Add <a href="../library/base64.html#base64.b32hexencode" class="reference internal" title="base64.b32hexencode"><span class="pre"><code class="sourceCode python">base64.b32hexencode()</code></span></a> and <a href="../library/base64.html#base64.b32hexdecode" class="reference internal" title="base64.b32hexdecode"><span class="pre"><code class="sourceCode python">base64.b32hexdecode()</code></span></a> to support the Base32 Encoding with Extended Hex Alphabet.

</div>

<div id="bdb" class="section">

### bdb<a href="#bdb" class="headerlink" title="Link to this heading">¶</a>

Add <span class="pre">`clearBreakpoints()`</span> to reset all set breakpoints. (Contributed by Irit Katriel in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24160" class="reference external">bpo-24160</a>.)

</div>

<div id="bisect" class="section">

### bisect<a href="#bisect" class="headerlink" title="Link to this heading">¶</a>

Added the possibility of providing a *key* function to the APIs in the <a href="../library/bisect.html#module-bisect" class="reference internal" title="bisect: Array bisection algorithms for binary searching."><span class="pre"><code class="sourceCode python">bisect</code></span></a> module. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4356" class="reference external">bpo-4356</a>.)

</div>

<div id="codecs" class="section">

### codecs<a href="#codecs" class="headerlink" title="Link to this heading">¶</a>

Add a <a href="../library/codecs.html#codecs.unregister" class="reference internal" title="codecs.unregister"><span class="pre"><code class="sourceCode python">codecs.unregister()</code></span></a> function to unregister a codec search function. (Contributed by Hai Shi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41842" class="reference external">bpo-41842</a>.)

</div>

<div id="collections-abc" class="section">

### collections.abc<a href="#collections-abc" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`__args__`</span> of the <a href="../library/stdtypes.html#types-genericalias" class="reference internal"><span class="std std-ref">parameterized generic</span></a> for <a href="../library/collections.abc.html#collections.abc.Callable" class="reference internal" title="collections.abc.Callable"><span class="pre"><code class="sourceCode python">collections.abc.Callable</code></span></a> are now consistent with <a href="../library/typing.html#typing.Callable" class="reference internal" title="typing.Callable"><span class="pre"><code class="sourceCode python">typing.Callable</code></span></a>. <a href="../library/collections.abc.html#collections.abc.Callable" class="reference internal" title="collections.abc.Callable"><span class="pre"><code class="sourceCode python">collections.abc.Callable</code></span></a> generic now flattens type parameters, similar to what <a href="../library/typing.html#typing.Callable" class="reference internal" title="typing.Callable"><span class="pre"><code class="sourceCode python">typing.Callable</code></span></a> currently does. This means that <span class="pre">`collections.abc.Callable[[int,`</span>` `<span class="pre">`str],`</span>` `<span class="pre">`str]`</span> will have <span class="pre">`__args__`</span> of <span class="pre">`(int,`</span>` `<span class="pre">`str,`</span>` `<span class="pre">`str)`</span>; previously this was <span class="pre">`([int,`</span>` `<span class="pre">`str],`</span>` `<span class="pre">`str)`</span>. To allow this change, <a href="../library/types.html#types.GenericAlias" class="reference internal" title="types.GenericAlias"><span class="pre"><code class="sourceCode python">types.GenericAlias</code></span></a> can now be subclassed, and a subclass will be returned when subscripting the <a href="../library/collections.abc.html#collections.abc.Callable" class="reference internal" title="collections.abc.Callable"><span class="pre"><code class="sourceCode python">collections.abc.Callable</code></span></a> type. Note that a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> may be raised for invalid forms of parameterizing <a href="../library/collections.abc.html#collections.abc.Callable" class="reference internal" title="collections.abc.Callable"><span class="pre"><code class="sourceCode python">collections.abc.Callable</code></span></a> which may have passed silently in Python 3.9. (Contributed by Ken Jin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42195" class="reference external">bpo-42195</a>.)

</div>

<div id="contextlib" class="section">

### contextlib<a href="#contextlib" class="headerlink" title="Link to this heading">¶</a>

Add a <a href="../library/contextlib.html#contextlib.aclosing" class="reference internal" title="contextlib.aclosing"><span class="pre"><code class="sourceCode python">contextlib.aclosing()</code></span></a> context manager to safely close async generators and objects representing asynchronously released resources. (Contributed by Joongi Kim and John Belmonte in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41229" class="reference external">bpo-41229</a>.)

Add asynchronous context manager support to <a href="../library/contextlib.html#contextlib.nullcontext" class="reference internal" title="contextlib.nullcontext"><span class="pre"><code class="sourceCode python">contextlib.nullcontext()</code></span></a>. (Contributed by Tom Gringauz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41543" class="reference external">bpo-41543</a>.)

Add <a href="../library/contextlib.html#contextlib.AsyncContextDecorator" class="reference internal" title="contextlib.AsyncContextDecorator"><span class="pre"><code class="sourceCode python">AsyncContextDecorator</code></span></a>, for supporting usage of async context managers as decorators.

</div>

<div id="curses" class="section">

### curses<a href="#curses" class="headerlink" title="Link to this heading">¶</a>

The extended color functions added in ncurses 6.1 will be used transparently by <a href="../library/curses.html#curses.color_content" class="reference internal" title="curses.color_content"><span class="pre"><code class="sourceCode python">curses.color_content()</code></span></a>, <a href="../library/curses.html#curses.init_color" class="reference internal" title="curses.init_color"><span class="pre"><code class="sourceCode python">curses.init_color()</code></span></a>, <a href="../library/curses.html#curses.init_pair" class="reference internal" title="curses.init_pair"><span class="pre"><code class="sourceCode python">curses.init_pair()</code></span></a>, and <a href="../library/curses.html#curses.pair_content" class="reference internal" title="curses.pair_content"><span class="pre"><code class="sourceCode python">curses.pair_content()</code></span></a>. A new function, <a href="../library/curses.html#curses.has_extended_color_support" class="reference internal" title="curses.has_extended_color_support"><span class="pre"><code class="sourceCode python">curses.has_extended_color_support()</code></span></a>, indicates whether extended color support is provided by the underlying ncurses library. (Contributed by Jeffrey Kintscher and Hans Petter Jansson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36982" class="reference external">bpo-36982</a>.)

The <span class="pre">`BUTTON5_*`</span> constants are now exposed in the <a href="../library/curses.html#module-curses" class="reference internal" title="curses: An interface to the curses library, providing portable terminal handling. (Unix)"><span class="pre"><code class="sourceCode python">curses</code></span></a> module if they are provided by the underlying curses library. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39273" class="reference external">bpo-39273</a>.)

</div>

<div id="dataclasses" class="section">

### dataclasses<a href="#dataclasses" class="headerlink" title="Link to this heading">¶</a>

<div id="slots" class="section">

#### \_\_slots\_\_<a href="#slots" class="headerlink" title="Link to this heading">¶</a>

Added <span class="pre">`slots`</span> parameter in <a href="../library/dataclasses.html#dataclasses.dataclass" class="reference internal" title="dataclasses.dataclass"><span class="pre"><code class="sourceCode python">dataclasses.dataclass()</code></span></a> decorator. (Contributed by Yurii Karabas in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42269" class="reference external">bpo-42269</a>)

</div>

<div id="keyword-only-fields" class="section">

#### Keyword-only fields<a href="#keyword-only-fields" class="headerlink" title="Link to this heading">¶</a>

dataclasses now supports fields that are keyword-only in the generated \_\_init\_\_ method. There are a number of ways of specifying keyword-only fields.

You can say that every field is keyword-only:

<div class="highlight-python notranslate">

<div class="highlight">

    from dataclasses import dataclass

    @dataclass(kw_only=True)
    class Birthday:
        name: str
        birthday: datetime.date

</div>

</div>

Both <span class="pre">`name`</span> and <span class="pre">`birthday`</span> are keyword-only parameters to the generated \_\_init\_\_ method.

You can specify keyword-only on a per-field basis:

<div class="highlight-python notranslate">

<div class="highlight">

    from dataclasses import dataclass, field

    @dataclass
    class Birthday:
        name: str
        birthday: datetime.date = field(kw_only=True)

</div>

</div>

Here only <span class="pre">`birthday`</span> is keyword-only. If you set <span class="pre">`kw_only`</span> on individual fields, be aware that there are rules about re-ordering fields due to keyword-only fields needing to follow non-keyword-only fields. See the full dataclasses documentation for details.

You can also specify that all fields following a KW_ONLY marker are keyword-only. This will probably be the most common usage:

<div class="highlight-python notranslate">

<div class="highlight">

    from dataclasses import dataclass, KW_ONLY

    @dataclass
    class Point:
        x: float
        y: float
        _: KW_ONLY
        z: float = 0.0
        t: float = 0.0

</div>

</div>

Here, <span class="pre">`z`</span> and <span class="pre">`t`</span> are keyword-only parameters, while <span class="pre">`x`</span> and <span class="pre">`y`</span> are not. (Contributed by Eric V. Smith in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43532" class="reference external">bpo-43532</a>.)

</div>

</div>

<div id="distutils" class="section">

<span id="distutils-deprecated"></span>

### distutils<a href="#distutils" class="headerlink" title="Link to this heading">¶</a>

The entire <span class="pre">`distutils`</span> package is deprecated, to be removed in Python 3.12. Its functionality for specifying package builds has already been completely replaced by third-party packages <span class="pre">`setuptools`</span> and <span class="pre">`packaging`</span>, and most other commonly used APIs are available elsewhere in the standard library (such as <a href="../library/platform.html#module-platform" class="reference internal" title="platform: Retrieves as much platform identifying data as possible."><span class="pre"><code class="sourceCode python">platform</code></span></a>, <a href="../library/shutil.html#module-shutil" class="reference internal" title="shutil: High-level file operations, including copying."><span class="pre"><code class="sourceCode python">shutil</code></span></a>, <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a> or <a href="../library/sysconfig.html#module-sysconfig" class="reference internal" title="sysconfig: Python&#39;s configuration information"><span class="pre"><code class="sourceCode python">sysconfig</code></span></a>). There are no plans to migrate any other functionality from <span class="pre">`distutils`</span>, and applications that are using other functions should plan to make private copies of the code. Refer to <span id="index-28" class="target"></span><a href="https://peps.python.org/pep-0632/" class="pep reference external"><strong>PEP 632</strong></a> for discussion.

The <span class="pre">`bdist_wininst`</span> command deprecated in Python 3.8 has been removed. The <span class="pre">`bdist_wheel`</span> command is now recommended to distribute binary packages on Windows. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42802" class="reference external">bpo-42802</a>.)

</div>

<div id="doctest" class="section">

### doctest<a href="#doctest" class="headerlink" title="Link to this heading">¶</a>

When a module does not define <span class="pre">`__loader__`</span>, fall back to <span class="pre">`__spec__.loader`</span>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42133" class="reference external">bpo-42133</a>.)

</div>

<div id="encodings" class="section">

### encodings<a href="#encodings" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/codecs.html#encodings.normalize_encoding" class="reference internal" title="encodings.normalize_encoding"><span class="pre"><code class="sourceCode python">encodings.normalize_encoding()</code></span></a> now ignores non-ASCII characters. (Contributed by Hai Shi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39337" class="reference external">bpo-39337</a>.)

</div>

<div id="enum" class="section">

### enum<a href="#enum" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/enum.html#enum.Enum" class="reference internal" title="enum.Enum"><span class="pre"><code class="sourceCode python">Enum</code></span></a> <a href="../reference/datamodel.html#object.__repr__" class="reference internal" title="object.__repr__"><span class="pre"><code class="sourceCode python"><span class="fu">__repr__</span>()</code></span></a> now returns <span class="pre">`enum_name.member_name`</span> and <a href="../reference/datamodel.html#object.__str__" class="reference internal" title="object.__str__"><span class="pre"><code class="sourceCode python"><span class="fu">__str__</span>()</code></span></a> now returns <span class="pre">`member_name`</span>. Stdlib enums available as module constants have a <a href="../library/functions.html#repr" class="reference internal" title="repr"><span class="pre"><code class="sourceCode python"><span class="bu">repr</span>()</code></span></a> of <span class="pre">`module_name.member_name`</span>. (Contributed by Ethan Furman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40066" class="reference external">bpo-40066</a>.)

Add <a href="../library/enum.html#enum.StrEnum" class="reference internal" title="enum.StrEnum"><span class="pre"><code class="sourceCode python">enum.StrEnum</code></span></a> for enums where all members are strings. (Contributed by Ethan Furman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41816" class="reference external">bpo-41816</a>.)

</div>

<div id="fileinput" class="section">

### fileinput<a href="#fileinput" class="headerlink" title="Link to this heading">¶</a>

Add *encoding* and *errors* parameters in <a href="../library/fileinput.html#fileinput.input" class="reference internal" title="fileinput.input"><span class="pre"><code class="sourceCode python">fileinput.<span class="bu">input</span>()</code></span></a> and <a href="../library/fileinput.html#fileinput.FileInput" class="reference internal" title="fileinput.FileInput"><span class="pre"><code class="sourceCode python">fileinput.FileInput</code></span></a>. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43712" class="reference external">bpo-43712</a>.)

<a href="../library/fileinput.html#fileinput.hook_compressed" class="reference internal" title="fileinput.hook_compressed"><span class="pre"><code class="sourceCode python">fileinput.hook_compressed()</code></span></a> now returns <a href="../library/io.html#io.TextIOWrapper" class="reference internal" title="io.TextIOWrapper"><span class="pre"><code class="sourceCode python">TextIOWrapper</code></span></a> object when *mode* is “r” and file is compressed, like uncompressed files. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5758" class="reference external">bpo-5758</a>.)

</div>

<div id="faulthandler" class="section">

### faulthandler<a href="#faulthandler" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/faulthandler.html#module-faulthandler" class="reference internal" title="faulthandler: Dump the Python traceback."><span class="pre"><code class="sourceCode python">faulthandler</code></span></a> module now detects if a fatal error occurs during a garbage collector collection. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44466" class="reference external">bpo-44466</a>.)

</div>

<div id="gc" class="section">

### gc<a href="#gc" class="headerlink" title="Link to this heading">¶</a>

Add audit hooks for <a href="../library/gc.html#gc.get_objects" class="reference internal" title="gc.get_objects"><span class="pre"><code class="sourceCode python">gc.get_objects()</code></span></a>, <a href="../library/gc.html#gc.get_referrers" class="reference internal" title="gc.get_referrers"><span class="pre"><code class="sourceCode python">gc.get_referrers()</code></span></a> and <a href="../library/gc.html#gc.get_referents" class="reference internal" title="gc.get_referents"><span class="pre"><code class="sourceCode python">gc.get_referents()</code></span></a>. (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43439" class="reference external">bpo-43439</a>.)

</div>

<div id="glob" class="section">

### glob<a href="#glob" class="headerlink" title="Link to this heading">¶</a>

Add the *root_dir* and *dir_fd* parameters in <a href="../library/glob.html#glob.glob" class="reference internal" title="glob.glob"><span class="pre"><code class="sourceCode python">glob()</code></span></a> and <a href="../library/glob.html#glob.iglob" class="reference internal" title="glob.iglob"><span class="pre"><code class="sourceCode python">iglob()</code></span></a> which allow to specify the root directory for searching. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38144" class="reference external">bpo-38144</a>.)

</div>

<div id="hashlib" class="section">

### hashlib<a href="#hashlib" class="headerlink" title="Link to this heading">¶</a>

The hashlib module requires OpenSSL 1.1.1 or newer. (Contributed by Christian Heimes in <span id="index-29" class="target"></span><a href="https://peps.python.org/pep-0644/" class="pep reference external"><strong>PEP 644</strong></a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43669" class="reference external">bpo-43669</a>.)

The hashlib module has preliminary support for OpenSSL 3.0.0. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38820" class="reference external">bpo-38820</a> and other issues.)

The pure-Python fallback of <a href="../library/hashlib.html#hashlib.pbkdf2_hmac" class="reference internal" title="hashlib.pbkdf2_hmac"><span class="pre"><code class="sourceCode python">pbkdf2_hmac()</code></span></a> is deprecated. In the future PBKDF2-HMAC will only be available when Python has been built with OpenSSL support. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43880" class="reference external">bpo-43880</a>.)

</div>

<div id="hmac" class="section">

### hmac<a href="#hmac" class="headerlink" title="Link to this heading">¶</a>

The hmac module now uses OpenSSL’s HMAC implementation internally. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40645" class="reference external">bpo-40645</a>.)

</div>

<div id="idle-and-idlelib" class="section">

### IDLE and idlelib<a href="#idle-and-idlelib" class="headerlink" title="Link to this heading">¶</a>

Make IDLE invoke <a href="../library/sys.html#sys.excepthook" class="reference internal" title="sys.excepthook"><span class="pre"><code class="sourceCode python">sys.excepthook()</code></span></a> (when started without ‘-n’). User hooks were previously ignored. (Contributed by Ken Hilton in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43008" class="reference external">bpo-43008</a>.)

Rearrange the settings dialog. Split the General tab into Windows and Shell/Ed tabs. Move help sources, which extend the Help menu, to the Extensions tab. Make space for new options and shorten the dialog. The latter makes the dialog better fit small screens. (Contributed by Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40468" class="reference external">bpo-40468</a>.) Move the indent space setting from the Font tab to the new Windows tab. (Contributed by Mark Roseman and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=33962" class="reference external">bpo-33962</a>.)

The changes above were backported to a 3.9 maintenance release.

Add a Shell sidebar. Move the primary prompt (‘\>\>\>’) to the sidebar. Add secondary prompts (’…’) to the sidebar. Left click and optional drag selects one or more lines of text, as with the editor line number sidebar. Right click after selecting text lines displays a context menu with ‘copy with prompts’. This zips together prompts from the sidebar with lines from the selected text. This option also appears on the context menu for the text. (Contributed by Tal Einat in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37903" class="reference external">bpo-37903</a>.)

Use spaces instead of tabs to indent interactive code. This makes interactive code entries ‘look right’. Making this feasible was a major motivation for adding the shell sidebar. (Contributed by Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37892" class="reference external">bpo-37892</a>.)

Highlight the new <a href="../reference/lexical_analysis.html#soft-keywords" class="reference internal"><span class="std std-ref">soft keywords</span></a> <a href="../reference/compound_stmts.html#match" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">match</code></span></a>, <a href="../reference/compound_stmts.html#match" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">case</code></span></a>, and <a href="../reference/compound_stmts.html#wildcard-patterns" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">_</code></span></a> in pattern-matching statements. However, this highlighting is not perfect and will be incorrect in some rare cases, including some <span class="pre">`_`</span>-s in <span class="pre">`case`</span> patterns. (Contributed by Tal Einat in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44010" class="reference external">bpo-44010</a>.)

New in 3.10 maintenance releases.

Apply syntax highlighting to <span class="pre">`.pyi`</span> files. (Contributed by Alex Waygood and Terry Jan Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=45447" class="reference external">bpo-45447</a>.)

Include prompts when saving Shell with inputs and outputs. (Contributed by Terry Jan Reedy in <a href="https://github.com/python/cpython/issues/95191" class="reference external">gh-95191</a>.)

</div>

<div id="importlib-metadata" class="section">

### importlib.metadata<a href="#importlib-metadata" class="headerlink" title="Link to this heading">¶</a>

Feature parity with <span class="pre">`importlib_metadata`</span> 4.6 (<a href="https://importlib-metadata.readthedocs.io/en/latest/history.html" class="reference external">history</a>).

<a href="../library/importlib.metadata.html#entry-points" class="reference internal"><span class="std std-ref">importlib.metadata entry points</span></a> now provide a nicer experience for selecting entry points by group and name through a new <a href="../library/importlib.metadata.html#entry-points" class="reference internal"><span class="std std-ref">importlib.metadata.EntryPoints</span></a> class. See the Compatibility Note in the docs for more info on the deprecation and usage.

Added <a href="../library/importlib.metadata.html#package-distributions" class="reference internal"><span class="std std-ref">importlib.metadata.packages_distributions()</span></a> for resolving top-level Python modules and packages to their <a href="../library/importlib.metadata.html#distributions" class="reference internal"><span class="std std-ref">importlib.metadata.Distribution</span></a>.

</div>

<div id="inspect" class="section">

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

When a module does not define <span class="pre">`__loader__`</span>, fall back to <span class="pre">`__spec__.loader`</span>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42133" class="reference external">bpo-42133</a>.)

Add <a href="../library/inspect.html#inspect.get_annotations" class="reference internal" title="inspect.get_annotations"><span class="pre"><code class="sourceCode python">inspect.get_annotations()</code></span></a>, which safely computes the annotations defined on an object. It works around the quirks of accessing the annotations on various types of objects, and makes very few assumptions about the object it examines. <a href="../library/inspect.html#inspect.get_annotations" class="reference internal" title="inspect.get_annotations"><span class="pre"><code class="sourceCode python">inspect.get_annotations()</code></span></a> can also correctly un-stringize stringized annotations. <a href="../library/inspect.html#inspect.get_annotations" class="reference internal" title="inspect.get_annotations"><span class="pre"><code class="sourceCode python">inspect.get_annotations()</code></span></a> is now considered best practice for accessing the annotations dict defined on any Python object; for more information on best practices for working with annotations, please see <a href="../howto/annotations.html#annotations-howto" class="reference internal"><span class="std std-ref">Annotations Best Practices</span></a>. Relatedly, <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a>, <a href="../library/inspect.html#inspect.Signature.from_callable" class="reference internal" title="inspect.Signature.from_callable"><span class="pre"><code class="sourceCode python">inspect.Signature.from_callable()</code></span></a>, and <span class="pre">`inspect.Signature.from_function()`</span> now call <a href="../library/inspect.html#inspect.get_annotations" class="reference internal" title="inspect.get_annotations"><span class="pre"><code class="sourceCode python">inspect.get_annotations()</code></span></a> to retrieve annotations. This means <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a> and <a href="../library/inspect.html#inspect.Signature.from_callable" class="reference internal" title="inspect.Signature.from_callable"><span class="pre"><code class="sourceCode python">inspect.Signature.from_callable()</code></span></a> can also now un-stringize stringized annotations. (Contributed by Larry Hastings in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43817" class="reference external">bpo-43817</a>.)

</div>

<div id="itertools" class="section">

### itertools<a href="#itertools" class="headerlink" title="Link to this heading">¶</a>

Add <a href="../library/itertools.html#itertools.pairwise" class="reference internal" title="itertools.pairwise"><span class="pre"><code class="sourceCode python">itertools.pairwise()</code></span></a>. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38200" class="reference external">bpo-38200</a>.)

</div>

<div id="linecache" class="section">

### linecache<a href="#linecache" class="headerlink" title="Link to this heading">¶</a>

When a module does not define <span class="pre">`__loader__`</span>, fall back to <span class="pre">`__spec__.loader`</span>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42133" class="reference external">bpo-42133</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

Add <a href="../library/os.html#os.cpu_count" class="reference internal" title="os.cpu_count"><span class="pre"><code class="sourceCode python">os.cpu_count()</code></span></a> support for VxWorks RTOS. (Contributed by Peixing Xin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41440" class="reference external">bpo-41440</a>.)

Add a new function <a href="../library/os.html#os.eventfd" class="reference internal" title="os.eventfd"><span class="pre"><code class="sourceCode python">os.eventfd()</code></span></a> and related helpers to wrap the <span class="pre">`eventfd2`</span> syscall on Linux. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41001" class="reference external">bpo-41001</a>.)

Add <a href="../library/os.html#os.splice" class="reference internal" title="os.splice"><span class="pre"><code class="sourceCode python">os.splice()</code></span></a> that allows to move data between two file descriptors without copying between kernel address space and user address space, where one of the file descriptors must refer to a pipe. (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41625" class="reference external">bpo-41625</a>.)

Add <a href="../library/os.html#os.O_EVTONLY" class="reference internal" title="os.O_EVTONLY"><span class="pre"><code class="sourceCode python">O_EVTONLY</code></span></a>, <a href="../library/os.html#os.O_FSYNC" class="reference internal" title="os.O_FSYNC"><span class="pre"><code class="sourceCode python">O_FSYNC</code></span></a>, <a href="../library/os.html#os.O_SYMLINK" class="reference internal" title="os.O_SYMLINK"><span class="pre"><code class="sourceCode python">O_SYMLINK</code></span></a> and <a href="../library/os.html#os.O_NOFOLLOW_ANY" class="reference internal" title="os.O_NOFOLLOW_ANY"><span class="pre"><code class="sourceCode python">O_NOFOLLOW_ANY</code></span></a> for macOS. (Contributed by Donghee Na in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43106" class="reference external">bpo-43106</a>.)

</div>

<div id="os-path" class="section">

### os.path<a href="#os-path" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/os.path.html#os.path.realpath" class="reference internal" title="os.path.realpath"><span class="pre"><code class="sourceCode python">os.path.realpath()</code></span></a> now accepts a *strict* keyword-only argument. When set to <span class="pre">`True`</span>, <a href="../library/exceptions.html#OSError" class="reference internal" title="OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a> is raised if a path doesn’t exist or a symlink loop is encountered. (Contributed by Barney Gale in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43757" class="reference external">bpo-43757</a>.)

</div>

<div id="pathlib" class="section">

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

Add slice support to <a href="../library/pathlib.html#pathlib.PurePath.parents" class="reference internal" title="pathlib.PurePath.parents"><span class="pre"><code class="sourceCode python">PurePath.parents</code></span></a>. (Contributed by Joshua Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35498" class="reference external">bpo-35498</a>.)

Add negative indexing support to <a href="../library/pathlib.html#pathlib.PurePath.parents" class="reference internal" title="pathlib.PurePath.parents"><span class="pre"><code class="sourceCode python">PurePath.parents</code></span></a>. (Contributed by Yaroslav Pankovych in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=21041" class="reference external">bpo-21041</a>.)

Add <a href="../library/pathlib.html#pathlib.Path.hardlink_to" class="reference internal" title="pathlib.Path.hardlink_to"><span class="pre"><code class="sourceCode python">Path.hardlink_to</code></span></a> method that supersedes <span class="pre">`link_to()`</span>. The new method has the same argument order as <a href="../library/pathlib.html#pathlib.Path.symlink_to" class="reference internal" title="pathlib.Path.symlink_to"><span class="pre"><code class="sourceCode python">symlink_to()</code></span></a>. (Contributed by Barney Gale in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39950" class="reference external">bpo-39950</a>.)

<a href="../library/pathlib.html#pathlib.Path.stat" class="reference internal" title="pathlib.Path.stat"><span class="pre"><code class="sourceCode python">pathlib.Path.stat()</code></span></a> and <a href="../library/pathlib.html#pathlib.Path.chmod" class="reference internal" title="pathlib.Path.chmod"><span class="pre"><code class="sourceCode python">chmod()</code></span></a> now accept a *follow_symlinks* keyword-only argument for consistency with corresponding functions in the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module. (Contributed by Barney Gale in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39906" class="reference external">bpo-39906</a>.)

</div>

<div id="platform" class="section">

### platform<a href="#platform" class="headerlink" title="Link to this heading">¶</a>

Add <a href="../library/platform.html#platform.freedesktop_os_release" class="reference internal" title="platform.freedesktop_os_release"><span class="pre"><code class="sourceCode python">platform.freedesktop_os_release()</code></span></a> to retrieve operation system identification from <a href="https://www.freedesktop.org/software/systemd/man/os-release.html" class="reference external">freedesktop.org os-release</a> standard file. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=28468" class="reference external">bpo-28468</a>.)

</div>

<div id="pprint" class="section">

### pprint<a href="#pprint" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pprint.html#pprint.pprint" class="reference internal" title="pprint.pprint"><span class="pre"><code class="sourceCode python">pprint.pprint()</code></span></a> now accepts a new <span class="pre">`underscore_numbers`</span> keyword argument. (Contributed by sblondon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42914" class="reference external">bpo-42914</a>.)

<a href="../library/pprint.html#module-pprint" class="reference internal" title="pprint: Data pretty printer."><span class="pre"><code class="sourceCode python">pprint</code></span></a> can now pretty-print <a href="../library/dataclasses.html#dataclasses.dataclass" class="reference internal" title="dataclasses.dataclass"><span class="pre"><code class="sourceCode python">dataclasses.dataclass</code></span></a> instances. (Contributed by Lewis Gaul in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43080" class="reference external">bpo-43080</a>.)

</div>

<div id="py-compile" class="section">

### py_compile<a href="#py-compile" class="headerlink" title="Link to this heading">¶</a>

Add <span class="pre">`--quiet`</span> option to command-line interface of <a href="../library/py_compile.html#module-py_compile" class="reference internal" title="py_compile: Generate byte-code files from Python source files."><span class="pre"><code class="sourceCode python">py_compile</code></span></a>. (Contributed by Gregory Schevchenko in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38731" class="reference external">bpo-38731</a>.)

</div>

<div id="pyclbr" class="section">

### pyclbr<a href="#pyclbr" class="headerlink" title="Link to this heading">¶</a>

Add an <span class="pre">`end_lineno`</span> attribute to the <span class="pre">`Function`</span> and <span class="pre">`Class`</span> objects in the tree returned by <a href="../library/pyclbr.html#pyclbr.readmodule" class="reference internal" title="pyclbr.readmodule"><span class="pre"><code class="sourceCode python">pyclbr.readmodule()</code></span></a> and <a href="../library/pyclbr.html#pyclbr.readmodule_ex" class="reference internal" title="pyclbr.readmodule_ex"><span class="pre"><code class="sourceCode python">pyclbr.readmodule_ex()</code></span></a>. It matches the existing (start) <span class="pre">`lineno`</span>. (Contributed by Aviral Srivastava in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38307" class="reference external">bpo-38307</a>.)

</div>

<div id="shelve" class="section">

### shelve<a href="#shelve" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/shelve.html#module-shelve" class="reference internal" title="shelve: Python object persistence."><span class="pre"><code class="sourceCode python">shelve</code></span></a> module now uses <a href="../library/pickle.html#pickle.DEFAULT_PROTOCOL" class="reference internal" title="pickle.DEFAULT_PROTOCOL"><span class="pre"><code class="sourceCode python">pickle.DEFAULT_PROTOCOL</code></span></a> by default instead of <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> protocol <span class="pre">`3`</span> when creating shelves. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=34204" class="reference external">bpo-34204</a>.)

</div>

<div id="statistics" class="section">

### statistics<a href="#statistics" class="headerlink" title="Link to this heading">¶</a>

Add <a href="../library/statistics.html#statistics.covariance" class="reference internal" title="statistics.covariance"><span class="pre"><code class="sourceCode python">covariance()</code></span></a>, Pearson’s <a href="../library/statistics.html#statistics.correlation" class="reference internal" title="statistics.correlation"><span class="pre"><code class="sourceCode python">correlation()</code></span></a>, and simple <a href="../library/statistics.html#statistics.linear_regression" class="reference internal" title="statistics.linear_regression"><span class="pre"><code class="sourceCode python">linear_regression()</code></span></a> functions. (Contributed by Tymoteusz Wołodźko in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38490" class="reference external">bpo-38490</a>.)

</div>

<div id="site" class="section">

### site<a href="#site" class="headerlink" title="Link to this heading">¶</a>

When a module does not define <span class="pre">`__loader__`</span>, fall back to <span class="pre">`__spec__.loader`</span>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42133" class="reference external">bpo-42133</a>.)

</div>

<div id="socket" class="section">

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

The exception <a href="../library/socket.html#socket.timeout" class="reference internal" title="socket.timeout"><span class="pre"><code class="sourceCode python">socket.timeout</code></span></a> is now an alias of <a href="../library/exceptions.html#TimeoutError" class="reference internal" title="TimeoutError"><span class="pre"><code class="sourceCode python"><span class="pp">TimeoutError</span></code></span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42413" class="reference external">bpo-42413</a>.)

Add option to create MPTCP sockets with <span class="pre">`IPPROTO_MPTCP`</span> (Contributed by Rui Cunha in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43571" class="reference external">bpo-43571</a>.)

Add <span class="pre">`IP_RECVTOS`</span> option to receive the type of service (ToS) or DSCP/ECN fields (Contributed by Georg Sauthoff in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44077" class="reference external">bpo-44077</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

The ssl module requires OpenSSL 1.1.1 or newer. (Contributed by Christian Heimes in <span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0644/" class="pep reference external"><strong>PEP 644</strong></a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43669" class="reference external">bpo-43669</a>.)

The ssl module has preliminary support for OpenSSL 3.0.0 and new option <a href="../library/ssl.html#ssl.OP_IGNORE_UNEXPECTED_EOF" class="reference internal" title="ssl.OP_IGNORE_UNEXPECTED_EOF"><span class="pre"><code class="sourceCode python">OP_IGNORE_UNEXPECTED_EOF</code></span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38820" class="reference external">bpo-38820</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43794" class="reference external">bpo-43794</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43788" class="reference external">bpo-43788</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43791" class="reference external">bpo-43791</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43799" class="reference external">bpo-43799</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43920" class="reference external">bpo-43920</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43789" class="reference external">bpo-43789</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43811" class="reference external">bpo-43811</a>.)

Deprecated function and use of deprecated constants now result in a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. <a href="../library/ssl.html#ssl.SSLContext.options" class="reference internal" title="ssl.SSLContext.options"><span class="pre"><code class="sourceCode python">ssl.SSLContext.options</code></span></a> has <a href="../library/ssl.html#ssl.OP_NO_SSLv2" class="reference internal" title="ssl.OP_NO_SSLv2"><span class="pre"><code class="sourceCode python">OP_NO_SSLv2</code></span></a> and <a href="../library/ssl.html#ssl.OP_NO_SSLv3" class="reference internal" title="ssl.OP_NO_SSLv3"><span class="pre"><code class="sourceCode python">OP_NO_SSLv3</code></span></a> set by default and therefore cannot warn about setting the flag again. The <a href="#whatsnew310-deprecated" class="reference internal"><span class="std std-ref">deprecation section</span></a> has a list of deprecated features. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43880" class="reference external">bpo-43880</a>.)

The ssl module now has more secure default settings. Ciphers without forward secrecy or SHA-1 MAC are disabled by default. Security level 2 prohibits weak RSA, DH, and ECC keys with less than 112 bits of security. <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> defaults to minimum protocol version TLS 1.2. Settings are based on Hynek Schlawack’s research. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43998" class="reference external">bpo-43998</a>.)

The deprecated protocols SSL 3.0, TLS 1.0, and TLS 1.1 are no longer officially supported. Python does not block them actively. However OpenSSL build options, distro configurations, vendor patches, and cipher suites may prevent a successful handshake.

Add a *timeout* parameter to the <a href="../library/ssl.html#ssl.get_server_certificate" class="reference internal" title="ssl.get_server_certificate"><span class="pre"><code class="sourceCode python">ssl.get_server_certificate()</code></span></a> function. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31870" class="reference external">bpo-31870</a>.)

The ssl module uses heap-types and multi-phase initialization. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42333" class="reference external">bpo-42333</a>.)

A new verify flag <a href="../library/ssl.html#ssl.VERIFY_X509_PARTIAL_CHAIN" class="reference internal" title="ssl.VERIFY_X509_PARTIAL_CHAIN"><span class="pre"><code class="sourceCode python">VERIFY_X509_PARTIAL_CHAIN</code></span></a> has been added. (Contributed by l0x in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40849" class="reference external">bpo-40849</a>.)

</div>

<div id="sqlite3" class="section">

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

Add audit events for <a href="../library/sqlite3.html#sqlite3.connect" class="reference internal" title="sqlite3.connect"><span class="pre"><code class="sourceCode python"><span class="ex">connect</span>()</code></span></a>, <a href="../library/sqlite3.html#sqlite3.Connection.enable_load_extension" class="reference internal" title="sqlite3.Connection.enable_load_extension"><span class="pre"><code class="sourceCode python">enable_load_extension()</code></span></a>, and <a href="../library/sqlite3.html#sqlite3.Connection.load_extension" class="reference internal" title="sqlite3.Connection.load_extension"><span class="pre"><code class="sourceCode python">load_extension()</code></span></a>. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43762" class="reference external">bpo-43762</a>.)

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

Add <a href="../library/sys.html#sys.orig_argv" class="reference internal" title="sys.orig_argv"><span class="pre"><code class="sourceCode python">sys.orig_argv</code></span></a> attribute: the list of the original command line arguments passed to the Python executable. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23427" class="reference external">bpo-23427</a>.)

Add <a href="../library/sys.html#sys.stdlib_module_names" class="reference internal" title="sys.stdlib_module_names"><span class="pre"><code class="sourceCode python">sys.stdlib_module_names</code></span></a>, containing the list of the standard library module names. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42955" class="reference external">bpo-42955</a>.)

</div>

<div id="thread" class="section">

### \_thread<a href="#thread" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/_thread.html#thread.interrupt_main" class="reference internal" title="_thread.interrupt_main"><span class="pre"><code class="sourceCode python">_thread.interrupt_main()</code></span></a> now takes an optional signal number to simulate (the default is still <a href="../library/signal.html#signal.SIGINT" class="reference internal" title="signal.SIGINT"><span class="pre"><code class="sourceCode python">signal.SIGINT</code></span></a>). (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43356" class="reference external">bpo-43356</a>.)

</div>

<div id="threading" class="section">

### threading<a href="#threading" class="headerlink" title="Link to this heading">¶</a>

Add <a href="../library/threading.html#threading.gettrace" class="reference internal" title="threading.gettrace"><span class="pre"><code class="sourceCode python">threading.gettrace()</code></span></a> and <a href="../library/threading.html#threading.getprofile" class="reference internal" title="threading.getprofile"><span class="pre"><code class="sourceCode python">threading.getprofile()</code></span></a> to retrieve the functions set by <a href="../library/threading.html#threading.settrace" class="reference internal" title="threading.settrace"><span class="pre"><code class="sourceCode python">threading.settrace()</code></span></a> and <a href="../library/threading.html#threading.setprofile" class="reference internal" title="threading.setprofile"><span class="pre"><code class="sourceCode python">threading.setprofile()</code></span></a> respectively. (Contributed by Mario Corchero in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42251" class="reference external">bpo-42251</a>.)

Add <a href="../library/threading.html#threading.__excepthook__" class="reference internal" title="threading.__excepthook__"><span class="pre"><code class="sourceCode python">threading.__excepthook__</code></span></a> to allow retrieving the original value of <a href="../library/threading.html#threading.excepthook" class="reference internal" title="threading.excepthook"><span class="pre"><code class="sourceCode python">threading.excepthook()</code></span></a> in case it is set to a broken or a different value. (Contributed by Mario Corchero in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42308" class="reference external">bpo-42308</a>.)

</div>

<div id="traceback" class="section">

### traceback<a href="#traceback" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/traceback.html#traceback.format_exception" class="reference internal" title="traceback.format_exception"><span class="pre"><code class="sourceCode python">format_exception()</code></span></a>, <a href="../library/traceback.html#traceback.format_exception_only" class="reference internal" title="traceback.format_exception_only"><span class="pre"><code class="sourceCode python">format_exception_only()</code></span></a>, and <a href="../library/traceback.html#traceback.print_exception" class="reference internal" title="traceback.print_exception"><span class="pre"><code class="sourceCode python">print_exception()</code></span></a> functions can now take an exception object as a positional-only argument. (Contributed by Zackery Spytz and Matthias Bussonnier in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26389" class="reference external">bpo-26389</a>.)

</div>

<div id="types" class="section">

### types<a href="#types" class="headerlink" title="Link to this heading">¶</a>

Reintroduce the <a href="../library/types.html#types.EllipsisType" class="reference internal" title="types.EllipsisType"><span class="pre"><code class="sourceCode python">types.EllipsisType</code></span></a>, <a href="../library/types.html#types.NoneType" class="reference internal" title="types.NoneType"><span class="pre"><code class="sourceCode python">types.NoneType</code></span></a> and <a href="../library/types.html#types.NotImplementedType" class="reference internal" title="types.NotImplementedType"><span class="pre"><code class="sourceCode python">types.NotImplementedType</code></span></a> classes, providing a new set of types readily interpretable by type checkers. (Contributed by Bas van Beek in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41810" class="reference external">bpo-41810</a>.)

</div>

<div id="typing" class="section">

### typing<a href="#typing" class="headerlink" title="Link to this heading">¶</a>

For major changes, see <a href="#new-feat-related-type-hints" class="reference internal"><span class="std std-ref">New Features Related to Type Hints</span></a>.

The behavior of <a href="../library/typing.html#typing.Literal" class="reference internal" title="typing.Literal"><span class="pre"><code class="sourceCode python">typing.Literal</code></span></a> was changed to conform with <span id="index-31" class="target"></span><a href="https://peps.python.org/pep-0586/" class="pep reference external"><strong>PEP 586</strong></a> and to match the behavior of static type checkers specified in the PEP.

1.  <span class="pre">`Literal`</span> now de-duplicates parameters.

2.  Equality comparisons between <span class="pre">`Literal`</span> objects are now order independent.

3.  <span class="pre">`Literal`</span> comparisons now respect types. For example, <span class="pre">`Literal[0]`</span>` `<span class="pre">`==`</span>` `<span class="pre">`Literal[False]`</span> previously evaluated to <span class="pre">`True`</span>. It is now <span class="pre">`False`</span>. To support this change, the internally used type cache now supports differentiating types.

4.  <span class="pre">`Literal`</span> objects will now raise a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception during equality comparisons if any of their parameters are not <a href="../glossary.html#term-hashable" class="reference internal"><span class="xref std std-term">hashable</span></a>. Note that declaring <span class="pre">`Literal`</span> with unhashable parameters will not throw an error:

    <div class="highlight-python3 notranslate">

    <div class="highlight">

        >>> from typing import Literal
        >>> Literal[{0}]
        >>> Literal[{0}] == Literal[{False}]
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        TypeError: unhashable type: 'set'

    </div>

    </div>

(Contributed by Yurii Karabas in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42345" class="reference external">bpo-42345</a>.)

Add new function <a href="../library/typing.html#typing.is_typeddict" class="reference internal" title="typing.is_typeddict"><span class="pre"><code class="sourceCode python">typing.is_typeddict()</code></span></a> to introspect if an annotation is a <a href="../library/typing.html#typing.TypedDict" class="reference internal" title="typing.TypedDict"><span class="pre"><code class="sourceCode python">typing.TypedDict</code></span></a>. (Contributed by Patrick Reader in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41792" class="reference external">bpo-41792</a>.)

Subclasses of <span class="pre">`typing.Protocol`</span> which only have data variables declared will now raise a <span class="pre">`TypeError`</span> when checked with <span class="pre">`isinstance`</span> unless they are decorated with <a href="../library/typing.html#typing.runtime_checkable" class="reference internal" title="typing.runtime_checkable"><span class="pre"><code class="sourceCode python">runtime_checkable()</code></span></a>. Previously, these checks passed silently. Users should decorate their subclasses with the <span class="pre">`runtime_checkable()`</span> decorator if they want runtime protocols. (Contributed by Yurii Karabas in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38908" class="reference external">bpo-38908</a>.)

Importing from the <span class="pre">`typing.io`</span> and <span class="pre">`typing.re`</span> submodules will now emit <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. These submodules have been deprecated since Python 3.8 and will be removed in a future version of Python. Anything belonging to those submodules should be imported directly from <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> instead. (Contributed by Sebastian Rittau in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38291" class="reference external">bpo-38291</a>.)

</div>

<div id="unittest" class="section">

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

Add new method <a href="../library/unittest.html#unittest.TestCase.assertNoLogs" class="reference internal" title="unittest.TestCase.assertNoLogs"><span class="pre"><code class="sourceCode python">assertNoLogs()</code></span></a> to complement the existing <a href="../library/unittest.html#unittest.TestCase.assertLogs" class="reference internal" title="unittest.TestCase.assertLogs"><span class="pre"><code class="sourceCode python">assertLogs()</code></span></a>. (Contributed by Kit Yan Choi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39385" class="reference external">bpo-39385</a>.)

</div>

<div id="urllib-parse" class="section">

### urllib.parse<a href="#urllib-parse" class="headerlink" title="Link to this heading">¶</a>

Python versions earlier than Python 3.10 allowed using both <span class="pre">`;`</span> and <span class="pre">`&`</span> as query parameter separators in <a href="../library/urllib.parse.html#urllib.parse.parse_qs" class="reference internal" title="urllib.parse.parse_qs"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qs()</code></span></a> and <a href="../library/urllib.parse.html#urllib.parse.parse_qsl" class="reference internal" title="urllib.parse.parse_qsl"><span class="pre"><code class="sourceCode python">urllib.parse.parse_qsl()</code></span></a>. Due to security concerns, and to conform with newer W3C recommendations, this has been changed to allow only a single separator key, with <span class="pre">`&`</span> as the default. This change also affects <span class="pre">`cgi.parse()`</span> and <span class="pre">`cgi.parse_multipart()`</span> as they use the affected functions internally. For more details, please see their respective documentation. (Contributed by Adam Goldschmidt, Senthil Kumaran and Ken Jin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42967" class="reference external">bpo-42967</a>.)

The presence of newline or tab characters in parts of a URL allows for some forms of attacks. Following the WHATWG specification that updates <span id="index-32" class="target"></span><a href="https://datatracker.ietf.org/doc/html/rfc3986.html" class="rfc reference external"><strong>RFC 3986</strong></a>, ASCII newline <span class="pre">`\n`</span>, <span class="pre">`\r`</span> and tab <span class="pre">`\t`</span> characters are stripped from the URL by the parser in <a href="../library/urllib.parse.html#module-urllib.parse" class="reference internal" title="urllib.parse: Parse URLs into or assemble them from components."><span class="pre"><code class="sourceCode python">urllib.parse</code></span></a> preventing such attacks. The removal characters are controlled by a new module level variable <span class="pre">`urllib.parse._UNSAFE_URL_BYTES_TO_REMOVE`</span>. (See <a href="https://github.com/python/cpython/issues/88048" class="reference external">gh-88048</a>)

</div>

<div id="xml" class="section">

### xml<a href="#xml" class="headerlink" title="Link to this heading">¶</a>

Add a <a href="../library/xml.sax.handler.html#xml.sax.handler.LexicalHandler" class="reference internal" title="xml.sax.handler.LexicalHandler"><span class="pre"><code class="sourceCode python">LexicalHandler</code></span></a> class to the <a href="../library/xml.sax.handler.html#module-xml.sax.handler" class="reference internal" title="xml.sax.handler: Base classes for SAX event handlers."><span class="pre"><code class="sourceCode python">xml.sax.handler</code></span></a> module. (Contributed by Jonathan Gossage and Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35018" class="reference external">bpo-35018</a>.)

</div>

<div id="zipimport" class="section">

### zipimport<a href="#zipimport" class="headerlink" title="Link to this heading">¶</a>

Add methods related to <span id="index-33" class="target"></span><a href="https://peps.python.org/pep-0451/" class="pep reference external"><strong>PEP 451</strong></a>: <a href="../library/zipimport.html#zipimport.zipimporter.find_spec" class="reference internal" title="zipimport.zipimporter.find_spec"><span class="pre"><code class="sourceCode python">find_spec()</code></span></a>, <a href="../library/zipimport.html#zipimport.zipimporter.create_module" class="reference internal" title="zipimport.zipimporter.create_module"><span class="pre"><code class="sourceCode python">zipimport.zipimporter.create_module()</code></span></a>, and <a href="../library/zipimport.html#zipimport.zipimporter.exec_module" class="reference internal" title="zipimport.zipimporter.exec_module"><span class="pre"><code class="sourceCode python">zipimport.zipimporter.exec_module()</code></span></a>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42131" class="reference external">bpo-42131</a>.)

Add <a href="../library/zipimport.html#zipimport.zipimporter.invalidate_caches" class="reference internal" title="zipimport.zipimporter.invalidate_caches"><span class="pre"><code class="sourceCode python">invalidate_caches()</code></span></a> method. (Contributed by Desmond Cheong in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14678" class="reference external">bpo-14678</a>.)

</div>

</div>

<div id="optimizations" class="section">

## Optimizations<a href="#optimizations" class="headerlink" title="Link to this heading">¶</a>

- Constructors <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a>, <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span>()</code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span>()</code></span></a> are now faster (around 30–40% for small objects). (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41334" class="reference external">bpo-41334</a>.)

- The <a href="../library/runpy.html#module-runpy" class="reference internal" title="runpy: Locate and run Python modules without importing them first."><span class="pre"><code class="sourceCode python">runpy</code></span></a> module now imports fewer modules. The <span class="pre">`python3`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`module-name`</span> command startup time is 1.4x faster in average. On Linux, <span class="pre">`python3`</span>` `<span class="pre">`-I`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`module-name`</span> imports 69 modules on Python 3.9, whereas it only imports 51 modules (-18) on Python 3.10. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41006" class="reference external">bpo-41006</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41718" class="reference external">bpo-41718</a>.)

- The <span class="pre">`LOAD_ATTR`</span> instruction now uses new “per opcode cache” mechanism. It is about 36% faster now for regular attributes and 44% faster for slots. (Contributed by Pablo Galindo and Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42093" class="reference external">bpo-42093</a> and Guido van Rossum in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42927" class="reference external">bpo-42927</a>, based on ideas implemented originally in PyPy and MicroPython.)

- When building Python with <a href="../using/configure.html#cmdoption-enable-optimizations" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--enable-optimizations</code></span></a> now <span class="pre">`-fno-semantic-interposition`</span> is added to both the compile and link line. This speeds builds of the Python interpreter created with <a href="../using/configure.html#cmdoption-enable-shared" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--enable-shared</code></span></a> with <span class="pre">`gcc`</span> by up to 30%. See <a href="https://developers.redhat.com/blog/2020/06/25/red-hat-enterprise-linux-8-2-brings-faster-python-3-8-run-speeds/" class="reference external">this article</a> for more details. (Contributed by Victor Stinner and Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38980" class="reference external">bpo-38980</a>.)

- Use a new output buffer management code for <a href="../library/bz2.html#module-bz2" class="reference internal" title="bz2: Interfaces for bzip2 compression and decompression."><span class="pre"><code class="sourceCode python">bz2</code></span></a> / <a href="../library/lzma.html#module-lzma" class="reference internal" title="lzma: A Python wrapper for the liblzma compression library."><span class="pre"><code class="sourceCode python">lzma</code></span></a> / <a href="../library/zlib.html#module-zlib" class="reference internal" title="zlib: Low-level interface to compression and decompression routines compatible with gzip."><span class="pre"><code class="sourceCode python">zlib</code></span></a> modules, and add <span class="pre">`.readall()`</span> function to <span class="pre">`_compression.DecompressReader`</span> class. bz2 decompression is now 1.09x ~ 1.17x faster, lzma decompression 1.20x ~ 1.32x faster, <span class="pre">`GzipFile.read(-1)`</span> 1.11x ~ 1.18x faster. (Contributed by Ma Lin, reviewed by Gregory P. Smith, in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41486" class="reference external">bpo-41486</a>)

- When using stringized annotations, annotations dicts for functions are no longer created when the function is created. Instead, they are stored as a tuple of strings, and the function object lazily converts this into the annotations dict on demand. This optimization cuts the CPU time needed to define an annotated function by half. (Contributed by Yurii Karabas and Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42202" class="reference external">bpo-42202</a>.)

- Substring search functions such as <span class="pre">`str1`</span>` `<span class="pre">`in`</span>` `<span class="pre">`str2`</span> and <span class="pre">`str2.find(str1)`</span> now sometimes use Crochemore & Perrin’s “Two-Way” string searching algorithm to avoid quadratic behavior on long strings. (Contributed by Dennis Sweeney in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41972" class="reference external">bpo-41972</a>)

- Add micro-optimizations to <span class="pre">`_PyType_Lookup()`</span> to improve type attribute cache lookup performance in the common case of cache hits. This makes the interpreter 1.04 times faster on average. (Contributed by Dino Viehland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43452" class="reference external">bpo-43452</a>.)

- The following built-in functions now support the faster <span id="index-34" class="target"></span><a href="https://peps.python.org/pep-0590/" class="pep reference external"><strong>PEP 590</strong></a> vectorcall calling convention: <a href="../library/functions.html#map" class="reference internal" title="map"><span class="pre"><code class="sourceCode python"><span class="bu">map</span>()</code></span></a>, <a href="../library/functions.html#filter" class="reference internal" title="filter"><span class="pre"><code class="sourceCode python"><span class="bu">filter</span>()</code></span></a>, <a href="../library/functions.html#reversed" class="reference internal" title="reversed"><span class="pre"><code class="sourceCode python"><span class="bu">reversed</span>()</code></span></a>, <a href="../library/functions.html#bool" class="reference internal" title="bool"><span class="pre"><code class="sourceCode python"><span class="bu">bool</span>()</code></span></a> and <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span>()</code></span></a>. (Contributed by Donghee Na and Jeroen Demeyer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43575" class="reference external">bpo-43575</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43287" class="reference external">bpo-43287</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41922" class="reference external">bpo-41922</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41873" class="reference external">bpo-41873</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41870" class="reference external">bpo-41870</a>.)

- <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">BZ2File</code></span></a> performance is improved by removing internal <span class="pre">`RLock`</span>. This makes <span class="pre">`BZ2File`</span> thread unsafe in the face of multiple simultaneous readers or writers, just like its equivalent classes in <a href="../library/gzip.html#module-gzip" class="reference internal" title="gzip: Interfaces for gzip compression and decompression using file objects."><span class="pre"><code class="sourceCode python">gzip</code></span></a> and <a href="../library/lzma.html#module-lzma" class="reference internal" title="lzma: A Python wrapper for the liblzma compression library."><span class="pre"><code class="sourceCode python">lzma</code></span></a> have always been. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43785" class="reference external">bpo-43785</a>.)

</div>

<div id="deprecated" class="section">

<span id="whatsnew310-deprecated"></span>

## Deprecated<a href="#deprecated" class="headerlink" title="Link to this heading">¶</a>

- Currently Python accepts numeric literals immediately followed by keywords, for example <span class="pre">`0in`</span>` `<span class="pre">`x`</span>, <span class="pre">`1or`</span>` `<span class="pre">`x`</span>, <span class="pre">`0if`</span>` `<span class="pre">`1else`</span>` `<span class="pre">`2`</span>. It allows confusing and ambiguous expressions like <span class="pre">`[0x1for`</span>` `<span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`y]`</span> (which can be interpreted as <span class="pre">`[0x1`</span>` `<span class="pre">`for`</span>` `<span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`y]`</span> or <span class="pre">`[0x1f`</span>` `<span class="pre">`or`</span>` `<span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`y]`</span>). Starting in this release, a deprecation warning is raised if the numeric literal is immediately followed by one of keywords <a href="../reference/expressions.html#and" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">and</code></span></a>, <a href="../reference/compound_stmts.html#else" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">else</code></span></a>, <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a>, <a href="../reference/compound_stmts.html#if" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">if</code></span></a>, <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a>, <a href="../reference/expressions.html#is" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">is</code></span></a> and <a href="../reference/expressions.html#or" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">or</code></span></a>. In future releases it will be changed to syntax warning, and finally to syntax error. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43833" class="reference external">bpo-43833</a>.)

- Starting in this release, there will be a concerted effort to begin cleaning up old import semantics that were kept for Python 2.7 compatibility. Specifically, <span class="pre">`find_loader()`</span>/<span class="pre">`find_module()`</span> (superseded by <a href="../library/importlib.html#importlib.abc.MetaPathFinder.find_spec" class="reference internal" title="importlib.abc.MetaPathFinder.find_spec"><span class="pre"><code class="sourceCode python">find_spec()</code></span></a>), <a href="../library/importlib.html#importlib.abc.Loader.load_module" class="reference internal" title="importlib.abc.Loader.load_module"><span class="pre"><code class="sourceCode python">load_module()</code></span></a> (superseded by <a href="../library/importlib.html#importlib.abc.Loader.exec_module" class="reference internal" title="importlib.abc.Loader.exec_module"><span class="pre"><code class="sourceCode python">exec_module()</code></span></a>), <span class="pre">`module_repr()`</span> (which the import system takes care of for you), the <span class="pre">`__package__`</span> attribute (superseded by <span class="pre">`__spec__.parent`</span>), the <span class="pre">`__loader__`</span> attribute (superseded by <span class="pre">`__spec__.loader`</span>), and the <span class="pre">`__cached__`</span> attribute (superseded by <span class="pre">`__spec__.cached`</span>) will slowly be removed (as well as other classes and methods in <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a>). <a href="../library/exceptions.html#ImportWarning" class="reference internal" title="ImportWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ImportWarning</span></code></span></a> and/or <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> will be raised as appropriate to help identify code which needs updating during this transition.

- The entire <span class="pre">`distutils`</span> namespace is deprecated, to be removed in Python 3.12. Refer to the <a href="#distutils-deprecated" class="reference internal"><span class="std std-ref">module changes</span></a> section for more information.

- Non-integer arguments to <a href="../library/random.html#random.randrange" class="reference internal" title="random.randrange"><span class="pre"><code class="sourceCode python">random.randrange()</code></span></a> are deprecated. The <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> is deprecated in favor of a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>. (Contributed by Serhiy Storchaka and Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37319" class="reference external">bpo-37319</a>.)

- The various <span class="pre">`load_module()`</span> methods of <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> have been documented as deprecated since Python 3.6, but will now also trigger a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. Use <a href="../library/importlib.html#importlib.abc.Loader.exec_module" class="reference internal" title="importlib.abc.Loader.exec_module"><span class="pre"><code class="sourceCode python">exec_module()</code></span></a> instead. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26131" class="reference external">bpo-26131</a>.)

- <span class="pre">`zimport.zipimporter.load_module()`</span> has been deprecated in preference for <a href="../library/zipimport.html#zipimport.zipimporter.exec_module" class="reference internal" title="zipimport.zipimporter.exec_module"><span class="pre"><code class="sourceCode python">exec_module()</code></span></a>. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26131" class="reference external">bpo-26131</a>.)

- The use of <a href="../library/importlib.html#importlib.abc.Loader.load_module" class="reference internal" title="importlib.abc.Loader.load_module"><span class="pre"><code class="sourceCode python">load_module()</code></span></a> by the import system now triggers an <a href="../library/exceptions.html#ImportWarning" class="reference internal" title="ImportWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ImportWarning</span></code></span></a> as <a href="../library/importlib.html#importlib.abc.Loader.exec_module" class="reference internal" title="importlib.abc.Loader.exec_module"><span class="pre"><code class="sourceCode python">exec_module()</code></span></a> is preferred. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26131" class="reference external">bpo-26131</a>.)

- The use of <span class="pre">`importlib.abc.MetaPathFinder.find_module()`</span> and <span class="pre">`importlib.abc.PathEntryFinder.find_module()`</span> by the import system now trigger an <a href="../library/exceptions.html#ImportWarning" class="reference internal" title="ImportWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ImportWarning</span></code></span></a> as <a href="../library/importlib.html#importlib.abc.MetaPathFinder.find_spec" class="reference internal" title="importlib.abc.MetaPathFinder.find_spec"><span class="pre"><code class="sourceCode python">importlib.abc.MetaPathFinder.find_spec()</code></span></a> and <a href="../library/importlib.html#importlib.abc.PathEntryFinder.find_spec" class="reference internal" title="importlib.abc.PathEntryFinder.find_spec"><span class="pre"><code class="sourceCode python">importlib.abc.PathEntryFinder.find_spec()</code></span></a> are preferred, respectively. You can use <a href="../library/importlib.html#importlib.util.spec_from_loader" class="reference internal" title="importlib.util.spec_from_loader"><span class="pre"><code class="sourceCode python">importlib.util.spec_from_loader()</code></span></a> to help in porting. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42134" class="reference external">bpo-42134</a>.)

- The use of <span class="pre">`importlib.abc.PathEntryFinder.find_loader()`</span> by the import system now triggers an <a href="../library/exceptions.html#ImportWarning" class="reference internal" title="ImportWarning"><span class="pre"><code class="sourceCode python"><span class="pp">ImportWarning</span></code></span></a> as <a href="../library/importlib.html#importlib.abc.PathEntryFinder.find_spec" class="reference internal" title="importlib.abc.PathEntryFinder.find_spec"><span class="pre"><code class="sourceCode python">importlib.abc.PathEntryFinder.find_spec()</code></span></a> is preferred. You can use <a href="../library/importlib.html#importlib.util.spec_from_loader" class="reference internal" title="importlib.util.spec_from_loader"><span class="pre"><code class="sourceCode python">importlib.util.spec_from_loader()</code></span></a> to help in porting. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43672" class="reference external">bpo-43672</a>.)

- The various implementations of <span class="pre">`importlib.abc.MetaPathFinder.find_module()`</span> ( <span class="pre">`importlib.machinery.BuiltinImporter.find_module()`</span>, <span class="pre">`importlib.machinery.FrozenImporter.find_module()`</span>, <span class="pre">`importlib.machinery.WindowsRegistryFinder.find_module()`</span>, <span class="pre">`importlib.machinery.PathFinder.find_module()`</span>, <span class="pre">`importlib.abc.MetaPathFinder.find_module()`</span> ), <span class="pre">`importlib.abc.PathEntryFinder.find_module()`</span> ( <span class="pre">`importlib.machinery.FileFinder.find_module()`</span> ), and <span class="pre">`importlib.abc.PathEntryFinder.find_loader()`</span> ( <span class="pre">`importlib.machinery.FileFinder.find_loader()`</span> ) now raise <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> and are slated for removal in Python 3.12 (previously they were documented as deprecated in Python 3.4). (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42135" class="reference external">bpo-42135</a>.)

- <span class="pre">`importlib.abc.Finder`</span> is deprecated (including its sole method, <span class="pre">`find_module()`</span>). Both <a href="../library/importlib.html#importlib.abc.MetaPathFinder" class="reference internal" title="importlib.abc.MetaPathFinder"><span class="pre"><code class="sourceCode python">importlib.abc.MetaPathFinder</code></span></a> and <a href="../library/importlib.html#importlib.abc.PathEntryFinder" class="reference internal" title="importlib.abc.PathEntryFinder"><span class="pre"><code class="sourceCode python">importlib.abc.PathEntryFinder</code></span></a> no longer inherit from the class. Users should inherit from one of these two classes as appropriate instead. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42135" class="reference external">bpo-42135</a>.)

- The deprecations of <span class="pre">`imp`</span>, <span class="pre">`importlib.find_loader()`</span>, <span class="pre">`importlib.util.set_package_wrapper()`</span>, <span class="pre">`importlib.util.set_loader_wrapper()`</span>, <span class="pre">`importlib.util.module_for_loader()`</span>, <span class="pre">`pkgutil.ImpImporter`</span>, and <span class="pre">`pkgutil.ImpLoader`</span> have all been updated to list Python 3.12 as the slated version of removal (they began raising <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> in previous versions of Python). (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43720" class="reference external">bpo-43720</a>.)

- The import system now uses the <span class="pre">`__spec__`</span> attribute on modules before falling back on <span class="pre">`module_repr()`</span> for a module’s <span class="pre">`__repr__()`</span> method. Removal of the use of <span class="pre">`module_repr()`</span> is scheduled for Python 3.12. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42137" class="reference external">bpo-42137</a>.)

- <span class="pre">`importlib.abc.Loader.module_repr()`</span>, <span class="pre">`importlib.machinery.FrozenLoader.module_repr()`</span>, and <span class="pre">`importlib.machinery.BuiltinLoader.module_repr()`</span> are deprecated and slated for removal in Python 3.12. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42136" class="reference external">bpo-42136</a>.)

- <span class="pre">`sqlite3.OptimizedUnicode`</span> has been undocumented and obsolete since Python 3.3, when it was made an alias to <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>. It is now deprecated, scheduled for removal in Python 3.12. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42264" class="reference external">bpo-42264</a>.)

- The undocumented built-in function <span class="pre">`sqlite3.enable_shared_cache`</span> is now deprecated, scheduled for removal in Python 3.12. Its use is strongly discouraged by the SQLite3 documentation. See <a href="https://sqlite.org/c3ref/enable_shared_cache.html" class="reference external">the SQLite3 docs</a> for more details. If a shared cache must be used, open the database in URI mode using the <span class="pre">`cache=shared`</span> query parameter. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=24464" class="reference external">bpo-24464</a>.)

- The following <span class="pre">`threading`</span> methods are now deprecated:

  - <span class="pre">`threading.currentThread`</span> =\> <a href="../library/threading.html#threading.current_thread" class="reference internal" title="threading.current_thread"><span class="pre"><code class="sourceCode python">threading.current_thread()</code></span></a>

  - <span class="pre">`threading.activeCount`</span> =\> <a href="../library/threading.html#threading.active_count" class="reference internal" title="threading.active_count"><span class="pre"><code class="sourceCode python">threading.active_count()</code></span></a>

  - <span class="pre">`threading.Condition.notifyAll`</span> =\> <a href="../library/threading.html#threading.Condition.notify_all" class="reference internal" title="threading.Condition.notify_all"><span class="pre"><code class="sourceCode python">threading.Condition.notify_all()</code></span></a>

  - <span class="pre">`threading.Event.isSet`</span> =\> <a href="../library/threading.html#threading.Event.is_set" class="reference internal" title="threading.Event.is_set"><span class="pre"><code class="sourceCode python">threading.Event.is_set()</code></span></a>

  - <span class="pre">`threading.Thread.setName`</span> =\> <a href="../library/threading.html#threading.Thread.name" class="reference internal" title="threading.Thread.name"><span class="pre"><code class="sourceCode python">threading.Thread.name</code></span></a>

  - <span class="pre">`threading.thread.getName`</span> =\> <a href="../library/threading.html#threading.Thread.name" class="reference internal" title="threading.Thread.name"><span class="pre"><code class="sourceCode python">threading.Thread.name</code></span></a>

  - <span class="pre">`threading.Thread.isDaemon`</span> =\> <a href="../library/threading.html#threading.Thread.daemon" class="reference internal" title="threading.Thread.daemon"><span class="pre"><code class="sourceCode python">threading.Thread.daemon</code></span></a>

  - <span class="pre">`threading.Thread.setDaemon`</span> =\> <a href="../library/threading.html#threading.Thread.daemon" class="reference internal" title="threading.Thread.daemon"><span class="pre"><code class="sourceCode python">threading.Thread.daemon</code></span></a>

  (Contributed by Jelle Zijlstra in <a href="https://github.com/python/cpython/issues/87889" class="reference external">gh-87889</a>.)

- <span class="pre">`pathlib.Path.link_to()`</span> is deprecated and slated for removal in Python 3.12. Use <a href="../library/pathlib.html#pathlib.Path.hardlink_to" class="reference internal" title="pathlib.Path.hardlink_to"><span class="pre"><code class="sourceCode python">pathlib.Path.hardlink_to()</code></span></a> instead. (Contributed by Barney Gale in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39950" class="reference external">bpo-39950</a>.)

- <span class="pre">`cgi.log()`</span> is deprecated and slated for removal in Python 3.12. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41139" class="reference external">bpo-41139</a>.)

- The following <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a> features have been deprecated since Python 3.6, Python 3.7, or OpenSSL 1.1.0 and will be removed in 3.11:

  - <span class="pre">`OP_NO_SSLv2`</span>, <span class="pre">`OP_NO_SSLv3`</span>, <span class="pre">`OP_NO_TLSv1`</span>, <span class="pre">`OP_NO_TLSv1_1`</span>, <span class="pre">`OP_NO_TLSv1_2`</span>, and <span class="pre">`OP_NO_TLSv1_3`</span> are replaced by <a href="../library/ssl.html#ssl.SSLContext.minimum_version" class="reference internal" title="ssl.SSLContext.minimum_version"><span class="pre"><code class="sourceCode python">minimum_version</code></span></a> and <a href="../library/ssl.html#ssl.SSLContext.maximum_version" class="reference internal" title="ssl.SSLContext.maximum_version"><span class="pre"><code class="sourceCode python">maximum_version</code></span></a>.

  - <span class="pre">`PROTOCOL_SSLv2`</span>, <span class="pre">`PROTOCOL_SSLv3`</span>, <span class="pre">`PROTOCOL_SSLv23`</span>, <span class="pre">`PROTOCOL_TLSv1`</span>, <span class="pre">`PROTOCOL_TLSv1_1`</span>, <span class="pre">`PROTOCOL_TLSv1_2`</span>, and <span class="pre">`PROTOCOL_TLS`</span> are deprecated in favor of <a href="../library/ssl.html#ssl.PROTOCOL_TLS_CLIENT" class="reference internal" title="ssl.PROTOCOL_TLS_CLIENT"><span class="pre"><code class="sourceCode python">PROTOCOL_TLS_CLIENT</code></span></a> and <a href="../library/ssl.html#ssl.PROTOCOL_TLS_SERVER" class="reference internal" title="ssl.PROTOCOL_TLS_SERVER"><span class="pre"><code class="sourceCode python">PROTOCOL_TLS_SERVER</code></span></a>

  - <span class="pre">`wrap_socket()`</span> is replaced by <a href="../library/ssl.html#ssl.SSLContext.wrap_socket" class="reference internal" title="ssl.SSLContext.wrap_socket"><span class="pre"><code class="sourceCode python">ssl.SSLContext.wrap_socket()</code></span></a>

  - <span class="pre">`match_hostname()`</span>

  - <span class="pre">`RAND_pseudo_bytes()`</span>, <span class="pre">`RAND_egd()`</span>

  - NPN features like <a href="../library/ssl.html#ssl.SSLSocket.selected_npn_protocol" class="reference internal" title="ssl.SSLSocket.selected_npn_protocol"><span class="pre"><code class="sourceCode python">ssl.SSLSocket.selected_npn_protocol()</code></span></a> and <a href="../library/ssl.html#ssl.SSLContext.set_npn_protocols" class="reference internal" title="ssl.SSLContext.set_npn_protocols"><span class="pre"><code class="sourceCode python">ssl.SSLContext.set_npn_protocols()</code></span></a> are replaced by ALPN.

- The threading debug (<span class="pre">`PYTHONTHREADDEBUG`</span> environment variable) is deprecated in Python 3.10 and will be removed in Python 3.12. This feature requires a <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">debug build of Python</span></a>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=44584" class="reference external">bpo-44584</a>.)

- Importing from the <span class="pre">`typing.io`</span> and <span class="pre">`typing.re`</span> submodules will now emit <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. These submodules will be removed in a future version of Python. Anything belonging to these submodules should be imported directly from <a href="../library/typing.html#module-typing" class="reference internal" title="typing: Support for type hints (see :pep:`484`)."><span class="pre"><code class="sourceCode python">typing</code></span></a> instead. (Contributed by Sebastian Rittau in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=38291" class="reference external">bpo-38291</a>.)

</div>

<div id="removed" class="section">

<span id="whatsnew310-removed"></span>

## Removed<a href="#removed" class="headerlink" title="Link to this heading">¶</a>

- Removed special methods <span class="pre">`__int__`</span>, <span class="pre">`__float__`</span>, <span class="pre">`__floordiv__`</span>, <span class="pre">`__mod__`</span>, <span class="pre">`__divmod__`</span>, <span class="pre">`__rfloordiv__`</span>, <span class="pre">`__rmod__`</span> and <span class="pre">`__rdivmod__`</span> of the <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a> class. They always raised a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41974" class="reference external">bpo-41974</a>.)

- The <span class="pre">`ParserBase.error()`</span> method from the private and undocumented <span class="pre">`_markupbase`</span> module has been removed. <a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">html.parser.HTMLParser</code></span></a> is the only subclass of <span class="pre">`ParserBase`</span> and its <span class="pre">`error()`</span> implementation was already removed in Python 3.5. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=31844" class="reference external">bpo-31844</a>.)

- Removed the <span class="pre">`unicodedata.ucnhash_CAPI`</span> attribute which was an internal PyCapsule object. The related private <span class="pre">`_PyUnicode_Name_CAPI`</span> structure was moved to the internal C API. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42157" class="reference external">bpo-42157</a>.)

- Removed the <span class="pre">`parser`</span> module, which was deprecated in 3.9 due to the switch to the new PEG parser, as well as all the C source and header files that were only being used by the old parser, including <span class="pre">`node.h`</span>, <span class="pre">`parser.h`</span>, <span class="pre">`graminit.h`</span> and <span class="pre">`grammar.h`</span>.

- Removed the Public C API functions <span class="pre">`PyParser_SimpleParseStringFlags`</span>, <span class="pre">`PyParser_SimpleParseStringFlagsFilename`</span>, <span class="pre">`PyParser_SimpleParseFileFlags`</span> and <span class="pre">`PyNode_Compile`</span> that were deprecated in 3.9 due to the switch to the new PEG parser.

- Removed the <span class="pre">`formatter`</span> module, which was deprecated in Python 3.4. It is somewhat obsolete, little used, and not tested. It was originally scheduled to be removed in Python 3.6, but such removals were delayed until after Python 2.7 EOL. Existing users should copy whatever classes they use into their code. (Contributed by Donghee Na and Terry J. Reedy in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42299" class="reference external">bpo-42299</a>.)

- Removed the <span class="pre">`PyModule_GetWarningsModule()`</span> function that was useless now due to the <span class="pre">`_warnings`</span> module was converted to a builtin module in 2.6. (Contributed by Hai Shi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42599" class="reference external">bpo-42599</a>.)

- Remove deprecated aliases to <a href="../library/collections.abc.html#collections-abstract-base-classes" class="reference internal"><span class="std std-ref">Collections Abstract Base Classes</span></a> from the <a href="../library/collections.html#module-collections" class="reference internal" title="collections: Container datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> module. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=37324" class="reference external">bpo-37324</a>.)

- The <span class="pre">`loop`</span> parameter has been removed from most of <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>‘s <a href="../library/asyncio-api-index.html" class="reference internal"><span class="doc">high-level API</span></a> following deprecation in Python 3.8. The motivation behind this change is multifold:

  1.  This simplifies the high-level API.

  2.  The functions in the high-level API have been implicitly getting the current thread’s running event loop since Python 3.7. There isn’t a need to pass the event loop to the API in most normal use cases.

  3.  Event loop passing is error-prone especially when dealing with loops running in different threads.

  Note that the low-level API will still accept <span class="pre">`loop`</span>. See <a href="#changes-python-api" class="reference internal"><span class="std std-ref">Changes in the Python API</span></a> for examples of how to replace existing code.

  (Contributed by Yurii Karabas, Andrew Svetlov, Yury Selivanov and Kyle Stanley in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42392" class="reference external">bpo-42392</a>.)

</div>

<div id="porting-to-python-3-10" class="section">

## Porting to Python 3.10<a href="#porting-to-python-3-10" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="changes-in-the-python-syntax" class="section">

### Changes in the Python syntax<a href="#changes-in-the-python-syntax" class="headerlink" title="Link to this heading">¶</a>

- Deprecation warning is now emitted when compiling previously valid syntax if the numeric literal is immediately followed by a keyword (like in <span class="pre">`0in`</span>` `<span class="pre">`x`</span>). In future releases it will be changed to syntax warning, and finally to a syntax error. To get rid of the warning and make the code compatible with future releases just add a space between the numeric literal and the following keyword. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43833" class="reference external">bpo-43833</a>.)

</div>

<div id="changes-in-the-python-api" class="section">

<span id="changes-python-api"></span>

### Changes in the Python API<a href="#changes-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- The *etype* parameters of the <a href="../library/traceback.html#traceback.format_exception" class="reference internal" title="traceback.format_exception"><span class="pre"><code class="sourceCode python">format_exception()</code></span></a>, <a href="../library/traceback.html#traceback.format_exception_only" class="reference internal" title="traceback.format_exception_only"><span class="pre"><code class="sourceCode python">format_exception_only()</code></span></a>, and <a href="../library/traceback.html#traceback.print_exception" class="reference internal" title="traceback.print_exception"><span class="pre"><code class="sourceCode python">print_exception()</code></span></a> functions in the <a href="../library/traceback.html#module-traceback" class="reference internal" title="traceback: Print or retrieve a stack traceback."><span class="pre"><code class="sourceCode python">traceback</code></span></a> module have been renamed to *exc*. (Contributed by Zackery Spytz and Matthias Bussonnier in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26389" class="reference external">bpo-26389</a>.)

- <a href="../library/atexit.html#module-atexit" class="reference internal" title="atexit: Register and execute cleanup functions."><span class="pre"><code class="sourceCode python">atexit</code></span></a>: At Python exit, if a callback registered with <a href="../library/atexit.html#atexit.register" class="reference internal" title="atexit.register"><span class="pre"><code class="sourceCode python">atexit.register()</code></span></a> fails, its exception is now logged. Previously, only some exceptions were logged, and the last exception was always silently ignored. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42639" class="reference external">bpo-42639</a>.)

- <a href="../library/collections.abc.html#collections.abc.Callable" class="reference internal" title="collections.abc.Callable"><span class="pre"><code class="sourceCode python">collections.abc.Callable</code></span></a> generic now flattens type parameters, similar to what <a href="../library/typing.html#typing.Callable" class="reference internal" title="typing.Callable"><span class="pre"><code class="sourceCode python">typing.Callable</code></span></a> currently does. This means that <span class="pre">`collections.abc.Callable[[int,`</span>` `<span class="pre">`str],`</span>` `<span class="pre">`str]`</span> will have <span class="pre">`__args__`</span> of <span class="pre">`(int,`</span>` `<span class="pre">`str,`</span>` `<span class="pre">`str)`</span>; previously this was <span class="pre">`([int,`</span>` `<span class="pre">`str],`</span>` `<span class="pre">`str)`</span>. Code which accesses the arguments via <a href="../library/typing.html#typing.get_args" class="reference internal" title="typing.get_args"><span class="pre"><code class="sourceCode python">typing.get_args()</code></span></a> or <span class="pre">`__args__`</span> need to account for this change. Furthermore, <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> may be raised for invalid forms of parameterizing <a href="../library/collections.abc.html#collections.abc.Callable" class="reference internal" title="collections.abc.Callable"><span class="pre"><code class="sourceCode python">collections.abc.Callable</code></span></a> which may have passed silently in Python 3.9. (Contributed by Ken Jin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42195" class="reference external">bpo-42195</a>.)

- <a href="../library/socket.html#socket.htons" class="reference internal" title="socket.htons"><span class="pre"><code class="sourceCode python">socket.htons()</code></span></a> and <a href="../library/socket.html#socket.ntohs" class="reference internal" title="socket.ntohs"><span class="pre"><code class="sourceCode python">socket.ntohs()</code></span></a> now raise <a href="../library/exceptions.html#OverflowError" class="reference internal" title="OverflowError"><span class="pre"><code class="sourceCode python"><span class="pp">OverflowError</span></code></span></a> instead of <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> if the given parameter will not fit in a 16-bit unsigned integer. (Contributed by Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42393" class="reference external">bpo-42393</a>.)

- The <span class="pre">`loop`</span> parameter has been removed from most of <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>‘s <a href="../library/asyncio-api-index.html" class="reference internal"><span class="doc">high-level API</span></a> following deprecation in Python 3.8.

  A coroutine that currently looks like this:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      async def foo(loop):
          await asyncio.sleep(1, loop=loop)

  </div>

  </div>

  Should be replaced with this:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      async def foo():
          await asyncio.sleep(1)

  </div>

  </div>

  If <span class="pre">`foo()`</span> was specifically designed *not* to run in the current thread’s running event loop (e.g. running in another thread’s event loop), consider using <a href="../library/asyncio-task.html#asyncio.run_coroutine_threadsafe" class="reference internal" title="asyncio.run_coroutine_threadsafe"><span class="pre"><code class="sourceCode python">asyncio.run_coroutine_threadsafe()</code></span></a> instead.

  (Contributed by Yurii Karabas, Andrew Svetlov, Yury Selivanov and Kyle Stanley in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42392" class="reference external">bpo-42392</a>.)

- The <a href="../library/types.html#types.FunctionType" class="reference internal" title="types.FunctionType"><span class="pre"><code class="sourceCode python">types.FunctionType</code></span></a> constructor now inherits the current builtins if the *globals* dictionary has no <span class="pre">`"__builtins__"`</span> key, rather than using <span class="pre">`{"None":`</span>` `<span class="pre">`None}`</span> as builtins: same behavior as <a href="../library/functions.html#eval" class="reference internal" title="eval"><span class="pre"><code class="sourceCode python"><span class="bu">eval</span>()</code></span></a> and <a href="../library/functions.html#exec" class="reference internal" title="exec"><span class="pre"><code class="sourceCode python"><span class="bu">exec</span>()</code></span></a> functions. Defining a function with <span class="pre">`def`</span>` `<span class="pre">`function(...):`</span>` `<span class="pre">`...`</span> in Python is not affected, globals cannot be overridden with this syntax: it also inherits the current builtins. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42990" class="reference external">bpo-42990</a>.)

</div>

<div id="changes-in-the-c-api" class="section">

### Changes in the C API<a href="#changes-in-the-c-api" class="headerlink" title="Link to this heading">¶</a>

- The C API functions <span class="pre">`PyParser_SimpleParseStringFlags`</span>, <span class="pre">`PyParser_SimpleParseStringFlagsFilename`</span>, <span class="pre">`PyParser_SimpleParseFileFlags`</span>, <span class="pre">`PyNode_Compile`</span> and the type used by these functions, <span class="pre">`struct`</span>` `<span class="pre">`_node`</span>, were removed due to the switch to the new PEG parser.

  Source should be now be compiled directly to a code object using, for example, <a href="../c-api/veryhigh.html#c.Py_CompileString" class="reference internal" title="Py_CompileString"><span class="pre"><code class="sourceCode c">Py_CompileString<span class="op">()</span></code></span></a>. The resulting code object can then be evaluated using, for example, <a href="../c-api/veryhigh.html#c.PyEval_EvalCode" class="reference internal" title="PyEval_EvalCode"><span class="pre"><code class="sourceCode c">PyEval_EvalCode<span class="op">()</span></code></span></a>.

  Specifically:

  - A call to <span class="pre">`PyParser_SimpleParseStringFlags`</span> followed by <span class="pre">`PyNode_Compile`</span> can be replaced by calling <a href="../c-api/veryhigh.html#c.Py_CompileString" class="reference internal" title="Py_CompileString"><span class="pre"><code class="sourceCode c">Py_CompileString<span class="op">()</span></code></span></a>.

  - There is no direct replacement for <span class="pre">`PyParser_SimpleParseFileFlags`</span>. To compile code from a <span class="pre">`FILE`</span>` `<span class="pre">`*`</span> argument, you will need to read the file in C and pass the resulting buffer to <a href="../c-api/veryhigh.html#c.Py_CompileString" class="reference internal" title="Py_CompileString"><span class="pre"><code class="sourceCode c">Py_CompileString<span class="op">()</span></code></span></a>.

  - To compile a file given a <span class="pre">`char`</span>` `<span class="pre">`*`</span> filename, explicitly open the file, read it and compile the result. One way to do this is using the <a href="../library/io.html#module-io" class="reference internal" title="io: Core tools for working with streams."><span class="pre"><code class="sourceCode python">io</code></span></a> module with <a href="../c-api/import.html#c.PyImport_ImportModule" class="reference internal" title="PyImport_ImportModule"><span class="pre"><code class="sourceCode c">PyImport_ImportModule<span class="op">()</span></code></span></a>, <a href="../c-api/call.html#c.PyObject_CallMethod" class="reference internal" title="PyObject_CallMethod"><span class="pre"><code class="sourceCode c">PyObject_CallMethod<span class="op">()</span></code></span></a>, <a href="../c-api/bytes.html#c.PyBytes_AsString" class="reference internal" title="PyBytes_AsString"><span class="pre"><code class="sourceCode c">PyBytes_AsString<span class="op">()</span></code></span></a> and <a href="../c-api/veryhigh.html#c.Py_CompileString" class="reference internal" title="Py_CompileString"><span class="pre"><code class="sourceCode c">Py_CompileString<span class="op">()</span></code></span></a>, as sketched below. (Declarations and error handling are omitted.)

    <div class="highlight-python3 notranslate">

    <div class="highlight">

        io_module = Import_ImportModule("io");
        fileobject = PyObject_CallMethod(io_module, "open", "ss", filename, "rb");
        source_bytes_object = PyObject_CallMethod(fileobject, "read", "");
        result = PyObject_CallMethod(fileobject, "close", "");
        source_buf = PyBytes_AsString(source_bytes_object);
        code = Py_CompileString(source_buf, filename, Py_file_input);

    </div>

    </div>

  - For <span class="pre">`FrameObject`</span> objects, the <a href="../reference/datamodel.html#frame.f_lasti" class="reference internal" title="frame.f_lasti"><span class="pre"><code class="sourceCode python">f_lasti</code></span></a> member now represents a wordcode offset instead of a simple offset into the bytecode string. This means that this number needs to be multiplied by 2 to be used with APIs that expect a byte offset instead (like <a href="../c-api/code.html#c.PyCode_Addr2Line" class="reference internal" title="PyCode_Addr2Line"><span class="pre"><code class="sourceCode c">PyCode_Addr2Line<span class="op">()</span></code></span></a> for example). Notice as well that the <span class="pre">`f_lasti`</span> member of <span class="pre">`FrameObject`</span> objects is not considered stable: please use <a href="../c-api/frame.html#c.PyFrame_GetLineNumber" class="reference internal" title="PyFrame_GetLineNumber"><span class="pre"><code class="sourceCode c">PyFrame_GetLineNumber<span class="op">()</span></code></span></a> instead.

</div>

</div>

<div id="cpython-bytecode-changes" class="section">

## CPython bytecode changes<a href="#cpython-bytecode-changes" class="headerlink" title="Link to this heading">¶</a>

- The <span class="pre">`MAKE_FUNCTION`</span> instruction now accepts either a dict or a tuple of strings as the function’s annotations. (Contributed by Yurii Karabas and Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42202" class="reference external">bpo-42202</a>.)

</div>

<div id="build-changes" class="section">

## Build Changes<a href="#build-changes" class="headerlink" title="Link to this heading">¶</a>

- <span id="index-35" class="target"></span><a href="https://peps.python.org/pep-0644/" class="pep reference external"><strong>PEP 644</strong></a>: Python now requires OpenSSL 1.1.1 or newer. OpenSSL 1.0.2 is no longer supported. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43669" class="reference external">bpo-43669</a>.)

- The C99 functions <span class="pre">`snprintf()`</span> and <span class="pre">`vsnprintf()`</span> are now required to build Python. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36020" class="reference external">bpo-36020</a>.)

- <a href="../library/sqlite3.html#module-sqlite3" class="reference internal" title="sqlite3: A DB-API 2.0 implementation using SQLite 3.x."><span class="pre"><code class="sourceCode python">sqlite3</code></span></a> requires SQLite 3.7.15 or higher. (Contributed by Sergey Fedoseev and Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40744" class="reference external">bpo-40744</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40810" class="reference external">bpo-40810</a>.)

- The <a href="../library/atexit.html#module-atexit" class="reference internal" title="atexit: Register and execute cleanup functions."><span class="pre"><code class="sourceCode python">atexit</code></span></a> module must now always be built as a built-in module. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42639" class="reference external">bpo-42639</a>.)

- Add <a href="../using/configure.html#cmdoption-disable-test-modules" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--disable-test-modules</code></span></a> option to the <span class="pre">`configure`</span> script: don’t build nor install test modules. (Contributed by Xavier de Gaye, Thomas Petazzoni and Peixing Xin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=27640" class="reference external">bpo-27640</a>.)

- Add <a href="../using/configure.html#cmdoption-with-wheel-pkg-dir" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--with-wheel-pkg-dir=PATH</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">option</code></span></a> to the <span class="pre">`./configure`</span> script. If specified, the <a href="../library/ensurepip.html#module-ensurepip" class="reference internal" title="ensurepip: Bootstrapping the &quot;pip&quot; installer into an existing Python installation or virtual environment."><span class="pre"><code class="sourceCode python">ensurepip</code></span></a> module looks for <span class="pre">`setuptools`</span> and <span class="pre">`pip`</span> wheel packages in this directory: if both are present, these wheel packages are used instead of ensurepip bundled wheel packages.

  Some Linux distribution packaging policies recommend against bundling dependencies. For example, Fedora installs wheel packages in the <span class="pre">`/usr/share/python-wheels/`</span> directory and don’t install the <span class="pre">`ensurepip._bundled`</span> package.

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42856" class="reference external">bpo-42856</a>.)

- Add a new <a href="../using/configure.html#cmdoption-without-static-libpython" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">configure</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">--without-static-libpython</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">option</code></span></a> to not build the <span class="pre">`libpythonMAJOR.MINOR.a`</span> static library and not install the <span class="pre">`python.o`</span> object file.

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43103" class="reference external">bpo-43103</a>.)

- The <span class="pre">`configure`</span> script now uses the <span class="pre">`pkg-config`</span> utility, if available, to detect the location of Tcl/Tk headers and libraries. As before, those locations can be explicitly specified with the <span class="pre">`--with-tcltk-includes`</span> and <span class="pre">`--with-tcltk-libs`</span> configuration options. (Contributed by Manolis Stamatogiannakis in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42603" class="reference external">bpo-42603</a>.)

- Add <a href="../using/configure.html#cmdoption-with-openssl-rpath" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--with-openssl-rpath</code></span></a> option to <span class="pre">`configure`</span> script. The option simplifies building Python with a custom OpenSSL installation, e.g. <span class="pre">`./configure`</span>` `<span class="pre">`--with-openssl=/path/to/openssl`</span>` `<span class="pre">`--with-openssl-rpath=auto`</span>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43466" class="reference external">bpo-43466</a>.)

</div>

<div id="c-api-changes" class="section">

## C API Changes<a href="#c-api-changes" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-652-maintaining-the-stable-abi" class="section">

### PEP 652: Maintaining the Stable ABI<a href="#pep-652-maintaining-the-stable-abi" class="headerlink" title="Link to this heading">¶</a>

The Stable ABI (Application Binary Interface) for extension modules or embedding Python is now explicitly defined. <a href="../c-api/stable.html#stable" class="reference internal"><span class="std std-ref">C API Stability</span></a> describes C API and ABI stability guarantees along with best practices for using the Stable ABI.

(Contributed by Petr Viktorin in <span id="index-36" class="target"></span><a href="https://peps.python.org/pep-0652/" class="pep reference external"><strong>PEP 652</strong></a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43795" class="reference external">bpo-43795</a>.)

</div>

<div id="id1" class="section">

### New Features<a href="#id1" class="headerlink" title="Link to this heading">¶</a>

- The result of <a href="../c-api/number.html#c.PyNumber_Index" class="reference internal" title="PyNumber_Index"><span class="pre"><code class="sourceCode c">PyNumber_Index<span class="op">()</span></code></span></a> now always has exact type <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a>. Previously, the result could have been an instance of a subclass of <span class="pre">`int`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40792" class="reference external">bpo-40792</a>.)

- Add a new <a href="../c-api/init_config.html#c.PyConfig.orig_argv" class="reference internal" title="PyConfig.orig_argv"><span class="pre"><code class="sourceCode c">orig_argv</code></span></a> member to the <a href="../c-api/init_config.html#c.PyConfig" class="reference internal" title="PyConfig"><span class="pre"><code class="sourceCode c">PyConfig</code></span></a> structure: the list of the original command line arguments passed to the Python executable. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=23427" class="reference external">bpo-23427</a>.)

- The <a href="../c-api/datetime.html#c.PyDateTime_DATE_GET_TZINFO" class="reference internal" title="PyDateTime_DATE_GET_TZINFO"><span class="pre"><code class="sourceCode c">PyDateTime_DATE_GET_TZINFO<span class="op">()</span></code></span></a> and <a href="../c-api/datetime.html#c.PyDateTime_TIME_GET_TZINFO" class="reference internal" title="PyDateTime_TIME_GET_TZINFO"><span class="pre"><code class="sourceCode c">PyDateTime_TIME_GET_TZINFO<span class="op">()</span></code></span></a> macros have been added for accessing the <span class="pre">`tzinfo`</span> attributes of <a href="../library/datetime.html#datetime.datetime" class="reference internal" title="datetime.datetime"><span class="pre"><code class="sourceCode python">datetime.datetime</code></span></a> and <a href="../library/datetime.html#datetime.time" class="reference internal" title="datetime.time"><span class="pre"><code class="sourceCode python">datetime.time</code></span></a> objects. (Contributed by Zackery Spytz in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30155" class="reference external">bpo-30155</a>.)

- Add a <a href="../c-api/codec.html#c.PyCodec_Unregister" class="reference internal" title="PyCodec_Unregister"><span class="pre"><code class="sourceCode c">PyCodec_Unregister<span class="op">()</span></code></span></a> function to unregister a codec search function. (Contributed by Hai Shi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41842" class="reference external">bpo-41842</a>.)

- The <a href="../c-api/iter.html#c.PyIter_Send" class="reference internal" title="PyIter_Send"><span class="pre"><code class="sourceCode c">PyIter_Send<span class="op">()</span></code></span></a> function was added to allow sending value into iterator without raising <span class="pre">`StopIteration`</span> exception. (Contributed by Vladimir Matveev in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41756" class="reference external">bpo-41756</a>.)

- Add <a href="../c-api/unicode.html#c.PyUnicode_AsUTF8AndSize" class="reference internal" title="PyUnicode_AsUTF8AndSize"><span class="pre"><code class="sourceCode c">PyUnicode_AsUTF8AndSize<span class="op">()</span></code></span></a> to the limited C API. (Contributed by Alex Gaynor in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41784" class="reference external">bpo-41784</a>.)

- Add <a href="../c-api/module.html#c.PyModule_AddObjectRef" class="reference internal" title="PyModule_AddObjectRef"><span class="pre"><code class="sourceCode c">PyModule_AddObjectRef<span class="op">()</span></code></span></a> function: similar to <a href="../c-api/module.html#c.PyModule_AddObject" class="reference internal" title="PyModule_AddObject"><span class="pre"><code class="sourceCode c">PyModule_AddObject<span class="op">()</span></code></span></a> but don’t steal a reference to the value on success. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1635741" class="reference external">bpo-1635741</a>.)

- Add <a href="../c-api/refcounting.html#c.Py_NewRef" class="reference internal" title="Py_NewRef"><span class="pre"><code class="sourceCode c">Py_NewRef<span class="op">()</span></code></span></a> and <a href="../c-api/refcounting.html#c.Py_XNewRef" class="reference internal" title="Py_XNewRef"><span class="pre"><code class="sourceCode c">Py_XNewRef<span class="op">()</span></code></span></a> functions to increment the reference count of an object and return the object. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42262" class="reference external">bpo-42262</a>.)

- The <a href="../c-api/type.html#c.PyType_FromSpecWithBases" class="reference internal" title="PyType_FromSpecWithBases"><span class="pre"><code class="sourceCode c">PyType_FromSpecWithBases<span class="op">()</span></code></span></a> and <a href="../c-api/type.html#c.PyType_FromModuleAndSpec" class="reference internal" title="PyType_FromModuleAndSpec"><span class="pre"><code class="sourceCode c">PyType_FromModuleAndSpec<span class="op">()</span></code></span></a> functions now accept a single class as the *bases* argument. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42423" class="reference external">bpo-42423</a>.)

- The <a href="../c-api/type.html#c.PyType_FromModuleAndSpec" class="reference internal" title="PyType_FromModuleAndSpec"><span class="pre"><code class="sourceCode c">PyType_FromModuleAndSpec<span class="op">()</span></code></span></a> function now accepts NULL <span class="pre">`tp_doc`</span> slot. (Contributed by Hai Shi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41832" class="reference external">bpo-41832</a>.)

- The <a href="../c-api/type.html#c.PyType_GetSlot" class="reference internal" title="PyType_GetSlot"><span class="pre"><code class="sourceCode c">PyType_GetSlot<span class="op">()</span></code></span></a> function can accept <a href="../c-api/typeobj.html#static-types" class="reference internal"><span class="std std-ref">static types</span></a>. (Contributed by Hai Shi and Petr Viktorin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41073" class="reference external">bpo-41073</a>.)

- Add a new <a href="../c-api/set.html#c.PySet_CheckExact" class="reference internal" title="PySet_CheckExact"><span class="pre"><code class="sourceCode c">PySet_CheckExact<span class="op">()</span></code></span></a> function to the C-API to check if an object is an instance of <a href="../library/stdtypes.html#set" class="reference internal" title="set"><span class="pre"><code class="sourceCode python"><span class="bu">set</span></code></span></a> but not an instance of a subtype. (Contributed by Pablo Galindo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43277" class="reference external">bpo-43277</a>.)

- Add <a href="../c-api/exceptions.html#c.PyErr_SetInterruptEx" class="reference internal" title="PyErr_SetInterruptEx"><span class="pre"><code class="sourceCode c">PyErr_SetInterruptEx<span class="op">()</span></code></span></a> which allows passing a signal number to simulate. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43356" class="reference external">bpo-43356</a>.)

- The limited C API is now supported if <a href="../using/configure.html#debug-build" class="reference internal"><span class="std std-ref">Python is built in debug mode</span></a> (if the <span class="pre">`Py_DEBUG`</span> macro is defined). In the limited C API, the <a href="../c-api/refcounting.html#c.Py_INCREF" class="reference internal" title="Py_INCREF"><span class="pre"><code class="sourceCode c">Py_INCREF<span class="op">()</span></code></span></a> and <a href="../c-api/refcounting.html#c.Py_DECREF" class="reference internal" title="Py_DECREF"><span class="pre"><code class="sourceCode c">Py_DECREF<span class="op">()</span></code></span></a> functions are now implemented as opaque function calls, rather than accessing directly the <a href="../c-api/typeobj.html#c.PyObject.ob_refcnt" class="reference internal" title="PyObject.ob_refcnt"><span class="pre"><code class="sourceCode c">PyObject<span class="op">.</span>ob_refcnt</code></span></a> member, if Python is built in debug mode and the <span class="pre">`Py_LIMITED_API`</span> macro targets Python 3.10 or newer. It became possible to support the limited C API in debug mode because the <a href="../c-api/structures.html#c.PyObject" class="reference internal" title="PyObject"><span class="pre"><code class="sourceCode c">PyObject</code></span></a> structure is the same in release and debug mode since Python 3.8 (see <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36465" class="reference external">bpo-36465</a>).

  The limited C API is still not supported in the <a href="../using/configure.html#cmdoption-with-trace-refs" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">--with-trace-refs</code></span></a> special build (<span class="pre">`Py_TRACE_REFS`</span> macro). (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43688" class="reference external">bpo-43688</a>.)

- Add the <a href="../c-api/structures.html#c.Py_Is" class="reference internal" title="Py_Is"><span class="pre"><code class="sourceCode c">Py_Is<span class="op">(</span>x<span class="op">,</span></code></span><code class="sourceCode c"> </code><span class="pre"><code class="sourceCode c">y<span class="op">)</span></code></span></a> function to test if the *x* object is the *y* object, the same as <span class="pre">`x`</span>` `<span class="pre">`is`</span>` `<span class="pre">`y`</span> in Python. Add also the <a href="../c-api/structures.html#c.Py_IsNone" class="reference internal" title="Py_IsNone"><span class="pre"><code class="sourceCode c">Py_IsNone<span class="op">()</span></code></span></a>, <a href="../c-api/structures.html#c.Py_IsTrue" class="reference internal" title="Py_IsTrue"><span class="pre"><code class="sourceCode c">Py_IsTrue<span class="op">()</span></code></span></a>, <a href="../c-api/structures.html#c.Py_IsFalse" class="reference internal" title="Py_IsFalse"><span class="pre"><code class="sourceCode c">Py_IsFalse<span class="op">()</span></code></span></a> functions to test if an object is, respectively, the <span class="pre">`None`</span> singleton, the <span class="pre">`True`</span> singleton or the <span class="pre">`False`</span> singleton. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43753" class="reference external">bpo-43753</a>.)

- Add new functions to control the garbage collector from C code: <a href="../c-api/gcsupport.html#c.PyGC_Enable" class="reference internal" title="PyGC_Enable"><span class="pre"><code class="sourceCode c">PyGC_Enable<span class="op">()</span></code></span></a>, <a href="../c-api/gcsupport.html#c.PyGC_Disable" class="reference internal" title="PyGC_Disable"><span class="pre"><code class="sourceCode c">PyGC_Disable<span class="op">()</span></code></span></a>, <a href="../c-api/gcsupport.html#c.PyGC_IsEnabled" class="reference internal" title="PyGC_IsEnabled"><span class="pre"><code class="sourceCode c">PyGC_IsEnabled<span class="op">()</span></code></span></a>. These functions allow to activate, deactivate and query the state of the garbage collector from C code without having to import the <a href="../library/gc.html#module-gc" class="reference internal" title="gc: Interface to the cycle-detecting garbage collector."><span class="pre"><code class="sourceCode python">gc</code></span></a> module.

- Add a new <a href="../c-api/typeobj.html#c.Py_TPFLAGS_DISALLOW_INSTANTIATION" class="reference internal" title="Py_TPFLAGS_DISALLOW_INSTANTIATION"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_DISALLOW_INSTANTIATION</code></span></a> type flag to disallow creating type instances. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43916" class="reference external">bpo-43916</a>.)

- Add a new <a href="../c-api/typeobj.html#c.Py_TPFLAGS_IMMUTABLETYPE" class="reference internal" title="Py_TPFLAGS_IMMUTABLETYPE"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_IMMUTABLETYPE</code></span></a> type flag for creating immutable type objects: type attributes cannot be set nor deleted. (Contributed by Victor Stinner and Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43908" class="reference external">bpo-43908</a>.)

</div>

<div id="id2" class="section">

### Porting to Python 3.10<a href="#id2" class="headerlink" title="Link to this heading">¶</a>

- The <span class="pre">`PY_SSIZE_T_CLEAN`</span> macro must now be defined to use <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple<span class="op">()</span></code></span></a> and <a href="../c-api/arg.html#c.Py_BuildValue" class="reference internal" title="Py_BuildValue"><span class="pre"><code class="sourceCode c">Py_BuildValue<span class="op">()</span></code></span></a> formats which use <span class="pre">`#`</span>: <span class="pre">`es#`</span>, <span class="pre">`et#`</span>, <span class="pre">`s#`</span>, <span class="pre">`u#`</span>, <span class="pre">`y#`</span>, <span class="pre">`z#`</span>, <span class="pre">`U#`</span> and <span class="pre">`Z#`</span>. See <a href="../c-api/arg.html#arg-parsing" class="reference internal"><span class="std std-ref">Parsing arguments and building values</span></a> and <span id="index-37" class="target"></span><a href="https://peps.python.org/pep-0353/" class="pep reference external"><strong>PEP 353</strong></a>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40943" class="reference external">bpo-40943</a>.)

- Since <a href="../c-api/refcounting.html#c.Py_REFCNT" class="reference internal" title="Py_REFCNT"><span class="pre"><code class="sourceCode c">Py_REFCNT<span class="op">()</span></code></span></a> is changed to the inline static function, <span class="pre">`Py_REFCNT(obj)`</span>` `<span class="pre">`=`</span>` `<span class="pre">`new_refcnt`</span> must be replaced with <span class="pre">`Py_SET_REFCNT(obj,`</span>` `<span class="pre">`new_refcnt)`</span>: see <a href="../c-api/refcounting.html#c.Py_SET_REFCNT" class="reference internal" title="Py_SET_REFCNT"><span class="pre"><code class="sourceCode c">Py_SET_REFCNT<span class="op">()</span></code></span></a> (available since Python 3.9). For backward compatibility, this macro can be used:

  <div class="highlight-python3 notranslate">

  <div class="highlight">

      #if PY_VERSION_HEX < 0x030900A4
      #  define Py_SET_REFCNT(obj, refcnt) ((Py_REFCNT(obj) = (refcnt)), (void)0)
      #endif

  </div>

  </div>

  (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=39573" class="reference external">bpo-39573</a>.)

- Calling <a href="../c-api/dict.html#c.PyDict_GetItem" class="reference internal" title="PyDict_GetItem"><span class="pre"><code class="sourceCode c">PyDict_GetItem<span class="op">()</span></code></span></a> without <a href="../glossary.html#term-GIL" class="reference internal"><span class="xref std std-term">GIL</span></a> held had been allowed for historical reason. It is no longer allowed. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=40839" class="reference external">bpo-40839</a>.)

- <span class="pre">`PyUnicode_FromUnicode(NULL,`</span>` `<span class="pre">`size)`</span> and <span class="pre">`PyUnicode_FromStringAndSize(NULL,`</span>` `<span class="pre">`size)`</span> raise <span class="pre">`DeprecationWarning`</span> now. Use <a href="../c-api/unicode.html#c.PyUnicode_New" class="reference internal" title="PyUnicode_New"><span class="pre"><code class="sourceCode c">PyUnicode_New<span class="op">()</span></code></span></a> to allocate Unicode object without initial data. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=36346" class="reference external">bpo-36346</a>.)

- The private <span class="pre">`_PyUnicode_Name_CAPI`</span> structure of the PyCapsule API <span class="pre">`unicodedata.ucnhash_CAPI`</span> has been moved to the internal C API. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42157" class="reference external">bpo-42157</a>.)

- <a href="../c-api/init.html#c.Py_GetPath" class="reference internal" title="Py_GetPath"><span class="pre"><code class="sourceCode c">Py_GetPath<span class="op">()</span></code></span></a>, <a href="../c-api/init.html#c.Py_GetPrefix" class="reference internal" title="Py_GetPrefix"><span class="pre"><code class="sourceCode c">Py_GetPrefix<span class="op">()</span></code></span></a>, <a href="../c-api/init.html#c.Py_GetExecPrefix" class="reference internal" title="Py_GetExecPrefix"><span class="pre"><code class="sourceCode c">Py_GetExecPrefix<span class="op">()</span></code></span></a>, <a href="../c-api/init.html#c.Py_GetProgramFullPath" class="reference internal" title="Py_GetProgramFullPath"><span class="pre"><code class="sourceCode c">Py_GetProgramFullPath<span class="op">()</span></code></span></a>, <a href="../c-api/init.html#c.Py_GetPythonHome" class="reference internal" title="Py_GetPythonHome"><span class="pre"><code class="sourceCode c">Py_GetPythonHome<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.Py_GetProgramName" class="reference internal" title="Py_GetProgramName"><span class="pre"><code class="sourceCode c">Py_GetProgramName<span class="op">()</span></code></span></a> functions now return <span class="pre">`NULL`</span> if called before <a href="../c-api/init.html#c.Py_Initialize" class="reference internal" title="Py_Initialize"><span class="pre"><code class="sourceCode c">Py_Initialize<span class="op">()</span></code></span></a> (before Python is initialized). Use the new <a href="../c-api/init_config.html#init-config" class="reference internal"><span class="std std-ref">Python Initialization Configuration</span></a> API to get the <a href="../c-api/init_config.html#init-path-config" class="reference internal"><span class="std std-ref">Python Path Configuration</span></a>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=42260" class="reference external">bpo-42260</a>.)

- <a href="../c-api/list.html#c.PyList_SET_ITEM" class="reference internal" title="PyList_SET_ITEM"><span class="pre"><code class="sourceCode c">PyList_SET_ITEM<span class="op">()</span></code></span></a>, <a href="../c-api/tuple.html#c.PyTuple_SET_ITEM" class="reference internal" title="PyTuple_SET_ITEM"><span class="pre"><code class="sourceCode c">PyTuple_SET_ITEM<span class="op">()</span></code></span></a> and <a href="../c-api/cell.html#c.PyCell_SET" class="reference internal" title="PyCell_SET"><span class="pre"><code class="sourceCode c">PyCell_SET<span class="op">()</span></code></span></a> macros can no longer be used as l-value or r-value. For example, <span class="pre">`x`</span>` `<span class="pre">`=`</span>` `<span class="pre">`PyList_SET_ITEM(a,`</span>` `<span class="pre">`b,`</span>` `<span class="pre">`c)`</span> and <span class="pre">`PyList_SET_ITEM(a,`</span>` `<span class="pre">`b,`</span>` `<span class="pre">`c)`</span>` `<span class="pre">`=`</span>` `<span class="pre">`x`</span> now fail with a compiler error. It prevents bugs like <span class="pre">`if`</span>` `<span class="pre">`(PyList_SET_ITEM`</span>` `<span class="pre">`(a,`</span>` `<span class="pre">`b,`</span>` `<span class="pre">`c)`</span>` `<span class="pre">`<`</span>` `<span class="pre">`0)`</span>` `<span class="pre">`...`</span> test. (Contributed by Zackery Spytz and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=30459" class="reference external">bpo-30459</a>.)

- The non-limited API files <span class="pre">`odictobject.h`</span>, <span class="pre">`parser_interface.h`</span>, <span class="pre">`picklebufobject.h`</span>, <span class="pre">`pyarena.h`</span>, <span class="pre">`pyctype.h`</span>, <span class="pre">`pydebug.h`</span>, <span class="pre">`pyfpe.h`</span>, and <span class="pre">`pytime.h`</span> have been moved to the <span class="pre">`Include/cpython`</span> directory. These files must not be included directly, as they are already included in <span class="pre">`Python.h`</span>; see <a href="../c-api/intro.html#api-includes" class="reference internal"><span class="std std-ref">Include Files</span></a>. If they have been included directly, consider including <span class="pre">`Python.h`</span> instead. (Contributed by Nicholas Sim in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=35134" class="reference external">bpo-35134</a>.)

- Use the <a href="../c-api/typeobj.html#c.Py_TPFLAGS_IMMUTABLETYPE" class="reference internal" title="Py_TPFLAGS_IMMUTABLETYPE"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_IMMUTABLETYPE</code></span></a> type flag to create immutable type objects. Do not rely on <a href="../c-api/typeobj.html#c.Py_TPFLAGS_HEAPTYPE" class="reference internal" title="Py_TPFLAGS_HEAPTYPE"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_HEAPTYPE</code></span></a> to decide if a type object is mutable or not; check if <a href="../c-api/typeobj.html#c.Py_TPFLAGS_IMMUTABLETYPE" class="reference internal" title="Py_TPFLAGS_IMMUTABLETYPE"><span class="pre"><code class="sourceCode c">Py_TPFLAGS_IMMUTABLETYPE</code></span></a> is set instead. (Contributed by Victor Stinner and Erlend E. Aasland in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43908" class="reference external">bpo-43908</a>.)

- The undocumented function <span class="pre">`Py_FrozenMain`</span> has been removed from the limited API. The function is mainly useful for custom builds of Python. (Contributed by Petr Viktorin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=26241" class="reference external">bpo-26241</a>.)

</div>

<div id="id3" class="section">

### Deprecated<a href="#id3" class="headerlink" title="Link to this heading">¶</a>

- The <span class="pre">`PyUnicode_InternImmortal()`</span> function is now deprecated and will be removed in Python 3.12: use <a href="../c-api/unicode.html#c.PyUnicode_InternInPlace" class="reference internal" title="PyUnicode_InternInPlace"><span class="pre"><code class="sourceCode c">PyUnicode_InternInPlace<span class="op">()</span></code></span></a> instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41692" class="reference external">bpo-41692</a>.)

</div>

<div id="id4" class="section">

### Removed<a href="#id4" class="headerlink" title="Link to this heading">¶</a>

- Removed <span class="pre">`Py_UNICODE_str*`</span> functions manipulating <span class="pre">`Py_UNICODE*`</span> strings. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41123" class="reference external">bpo-41123</a>.)

  - <span class="pre">`Py_UNICODE_strlen`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_GetLength" class="reference internal" title="PyUnicode_GetLength"><span class="pre"><code class="sourceCode c">PyUnicode_GetLength<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_GET_LENGTH" class="reference internal" title="PyUnicode_GET_LENGTH"><span class="pre"><code class="sourceCode c">PyUnicode_GET_LENGTH</code></span></a>

  - <span class="pre">`Py_UNICODE_strcat`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_CopyCharacters" class="reference internal" title="PyUnicode_CopyCharacters"><span class="pre"><code class="sourceCode c">PyUnicode_CopyCharacters<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_FromFormat" class="reference internal" title="PyUnicode_FromFormat"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormat<span class="op">()</span></code></span></a>

  - <span class="pre">`Py_UNICODE_strcpy`</span>, <span class="pre">`Py_UNICODE_strncpy`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_CopyCharacters" class="reference internal" title="PyUnicode_CopyCharacters"><span class="pre"><code class="sourceCode c">PyUnicode_CopyCharacters<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_Substring" class="reference internal" title="PyUnicode_Substring"><span class="pre"><code class="sourceCode c">PyUnicode_Substring<span class="op">()</span></code></span></a>

  - <span class="pre">`Py_UNICODE_strcmp`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_Compare" class="reference internal" title="PyUnicode_Compare"><span class="pre"><code class="sourceCode c">PyUnicode_Compare<span class="op">()</span></code></span></a>

  - <span class="pre">`Py_UNICODE_strncmp`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_Tailmatch" class="reference internal" title="PyUnicode_Tailmatch"><span class="pre"><code class="sourceCode c">PyUnicode_Tailmatch<span class="op">()</span></code></span></a>

  - <span class="pre">`Py_UNICODE_strchr`</span>, <span class="pre">`Py_UNICODE_strrchr`</span>: use <a href="../c-api/unicode.html#c.PyUnicode_FindChar" class="reference internal" title="PyUnicode_FindChar"><span class="pre"><code class="sourceCode c">PyUnicode_FindChar<span class="op">()</span></code></span></a>

- Removed <span class="pre">`PyUnicode_GetMax()`</span>. Please migrate to new (<span id="index-38" class="target"></span><a href="https://peps.python.org/pep-0393/" class="pep reference external"><strong>PEP 393</strong></a>) APIs. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41103" class="reference external">bpo-41103</a>.)

- Removed <span class="pre">`PyLong_FromUnicode()`</span>. Please migrate to <a href="../c-api/long.html#c.PyLong_FromUnicodeObject" class="reference internal" title="PyLong_FromUnicodeObject"><span class="pre"><code class="sourceCode c">PyLong_FromUnicodeObject<span class="op">()</span></code></span></a>. (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41103" class="reference external">bpo-41103</a>.)

- Removed <span class="pre">`PyUnicode_AsUnicodeCopy()`</span>. Please use <a href="../c-api/unicode.html#c.PyUnicode_AsUCS4Copy" class="reference internal" title="PyUnicode_AsUCS4Copy"><span class="pre"><code class="sourceCode c">PyUnicode_AsUCS4Copy<span class="op">()</span></code></span></a> or <a href="../c-api/unicode.html#c.PyUnicode_AsWideCharString" class="reference internal" title="PyUnicode_AsWideCharString"><span class="pre"><code class="sourceCode c">PyUnicode_AsWideCharString<span class="op">()</span></code></span></a> (Contributed by Inada Naoki in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41103" class="reference external">bpo-41103</a>.)

- Removed <span class="pre">`_Py_CheckRecursionLimit`</span> variable: it has been replaced by <span class="pre">`ceval.recursion_limit`</span> of the <a href="../c-api/init.html#c.PyInterpreterState" class="reference internal" title="PyInterpreterState"><span class="pre"><code class="sourceCode c">PyInterpreterState</code></span></a> structure. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41834" class="reference external">bpo-41834</a>.)

- Removed undocumented macros <span class="pre">`Py_ALLOW_RECURSION`</span> and <span class="pre">`Py_END_ALLOW_RECURSION`</span> and the <span class="pre">`recursion_critical`</span> field of the <a href="../c-api/init.html#c.PyInterpreterState" class="reference internal" title="PyInterpreterState"><span class="pre"><code class="sourceCode c">PyInterpreterState</code></span></a> structure. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41936" class="reference external">bpo-41936</a>.)

- Removed the undocumented <span class="pre">`PyOS_InitInterrupts()`</span> function. Initializing Python already implicitly installs signal handlers: see <a href="../c-api/init_config.html#c.PyConfig.install_signal_handlers" class="reference internal" title="PyConfig.install_signal_handlers"><span class="pre"><code class="sourceCode c">PyConfig<span class="op">.</span>install_signal_handlers</code></span></a>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=41713" class="reference external">bpo-41713</a>.)

- Remove the <span class="pre">`PyAST_Validate()`</span> function. It is no longer possible to build a AST object (<span class="pre">`mod_ty`</span> type) with the public C API. The function was already excluded from the limited C API (<span id="index-39" class="target"></span><a href="https://peps.python.org/pep-0384/" class="pep reference external"><strong>PEP 384</strong></a>). (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43244" class="reference external">bpo-43244</a>.)

- Remove the <span class="pre">`symtable.h`</span> header file and the undocumented functions:

  - <span class="pre">`PyST_GetScope()`</span>

  - <span class="pre">`PySymtable_Build()`</span>

  - <span class="pre">`PySymtable_BuildObject()`</span>

  - <span class="pre">`PySymtable_Free()`</span>

  - <span class="pre">`Py_SymtableString()`</span>

  - <span class="pre">`Py_SymtableStringObject()`</span>

  The <span class="pre">`Py_SymtableString()`</span> function was part the stable ABI by mistake but it could not be used, because the <span class="pre">`symtable.h`</span> header file was excluded from the limited C API.

  Use Python <a href="../library/symtable.html#module-symtable" class="reference internal" title="symtable: Interface to the compiler&#39;s internal symbol tables."><span class="pre"><code class="sourceCode python">symtable</code></span></a> module instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43244" class="reference external">bpo-43244</a>.)

- Remove <a href="../c-api/veryhigh.html#c.PyOS_ReadlineFunctionPointer" class="reference internal" title="PyOS_ReadlineFunctionPointer"><span class="pre"><code class="sourceCode c">PyOS_ReadlineFunctionPointer<span class="op">()</span></code></span></a> from the limited C API headers and from <span class="pre">`python3.dll`</span>, the library that provides the stable ABI on Windows. Since the function takes a <span class="pre">`FILE*`</span> argument, its ABI stability cannot be guaranteed. (Contributed by Petr Viktorin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43868" class="reference external">bpo-43868</a>.)

- Remove <span class="pre">`ast.h`</span>, <span class="pre">`asdl.h`</span>, and <span class="pre">`Python-ast.h`</span> header files. These functions were undocumented and excluded from the limited C API. Most names defined by these header files were not prefixed by <span class="pre">`Py`</span> and so could create names conflicts. For example, <span class="pre">`Python-ast.h`</span> defined a <span class="pre">`Yield`</span> macro which was conflict with the <span class="pre">`Yield`</span> name used by the Windows <span class="pre">`<winbase.h>`</span> header. Use the Python <a href="../library/ast.html#module-ast" class="reference internal" title="ast: Abstract Syntax Tree classes and manipulation."><span class="pre"><code class="sourceCode python">ast</code></span></a> module instead. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43244" class="reference external">bpo-43244</a>.)

- Remove the compiler and parser functions using <span class="pre">`struct`</span>` `<span class="pre">`_mod`</span> type, because the public AST C API was removed:

  - <span class="pre">`PyAST_Compile()`</span>

  - <span class="pre">`PyAST_CompileEx()`</span>

  - <span class="pre">`PyAST_CompileObject()`</span>

  - <span class="pre">`PyFuture_FromAST()`</span>

  - <span class="pre">`PyFuture_FromASTObject()`</span>

  - <span class="pre">`PyParser_ASTFromFile()`</span>

  - <span class="pre">`PyParser_ASTFromFileObject()`</span>

  - <span class="pre">`PyParser_ASTFromFilename()`</span>

  - <span class="pre">`PyParser_ASTFromString()`</span>

  - <span class="pre">`PyParser_ASTFromStringObject()`</span>

  These functions were undocumented and excluded from the limited C API. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43244" class="reference external">bpo-43244</a>.)

- Remove the <span class="pre">`pyarena.h`</span> header file with functions:

  - <span class="pre">`PyArena_New()`</span>

  - <span class="pre">`PyArena_Free()`</span>

  - <span class="pre">`PyArena_Malloc()`</span>

  - <span class="pre">`PyArena_AddPyObject()`</span>

  These functions were undocumented, excluded from the limited C API, and were only used internally by the compiler. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43244" class="reference external">bpo-43244</a>.)

- The <span class="pre">`PyThreadState.use_tracing`</span> member has been removed to optimize Python. (Contributed by Mark Shannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=43760" class="reference external">bpo-43760</a>.)

</div>

</div>

<div id="notable-security-feature-in-3-10-7" class="section">

## Notable security feature in 3.10.7<a href="#notable-security-feature-in-3-10-7" class="headerlink" title="Link to this heading">¶</a>

Converting between <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> and <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a> in bases other than 2 (binary), 4, 8 (octal), 16 (hexadecimal), or 32 such as base 10 (decimal) now raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the number of digits in string form is above a limit to avoid potential denial of service attacks due to the algorithmic complexity. This is a mitigation for <span id="index-40" class="target"></span><a href="https://www.cve.org/CVERecord?id=CVE-2020-10735" class="cve reference external"><strong>CVE 2020-10735</strong></a>. This limit can be configured or disabled by environment variable, command line flag, or <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> APIs. See the <a href="../library/stdtypes.html#int-max-str-digits" class="reference internal"><span class="std std-ref">integer string conversion length limitation</span></a> documentation. The default limit is 4300 digits in string form.

</div>

<div id="notable-security-feature-in-3-10-8" class="section">

## Notable security feature in 3.10.8<a href="#notable-security-feature-in-3-10-8" class="headerlink" title="Link to this heading">¶</a>

The deprecated <span class="pre">`mailcap`</span> module now refuses to inject unsafe text (filenames, MIME types, parameters) into shell commands. Instead of using such text, it will warn and act as if a match was not found (or for test commands, as if the test failed). (Contributed by Petr Viktorin in <a href="https://github.com/python/cpython/issues/98966" class="reference external">gh-98966</a>.)

</div>

<div id="notable-changes-in-3-10-12" class="section">

## Notable changes in 3.10.12<a href="#notable-changes-in-3-10-12" class="headerlink" title="Link to this heading">¶</a>

<div id="tarfile" class="section">

### tarfile<a href="#tarfile" class="headerlink" title="Link to this heading">¶</a>

- The extraction methods in <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a>, and <a href="../library/shutil.html#shutil.unpack_archive" class="reference internal" title="shutil.unpack_archive"><span class="pre"><code class="sourceCode python">shutil.unpack_archive()</code></span></a>, have a new a *filter* argument that allows limiting tar features than may be surprising or dangerous, such as creating files outside the destination directory. See <a href="../library/tarfile.html#tarfile-extraction-filter" class="reference internal"><span class="std std-ref">Extraction filters</span></a> for details. In Python 3.12, use without the *filter* argument will show a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a>. In Python 3.14, the default will switch to <span class="pre">`'data'`</span>. (Contributed by Petr Viktorin in <span id="index-41" class="target"></span><a href="https://peps.python.org/pep-0706/" class="pep reference external"><strong>PEP 706</strong></a>.)

</div>

</div>

</div>

<div class="clearer">

</div>

</div>
