<div class="body" role="main">

<div id="what-s-new-in-python-2-2" class="section">

# What’s New in Python 2.2<a href="#what-s-new-in-python-2-2" class="headerlink" title="Permalink to this headline">¶</a>

Author  
A.M. Kuchling

<div id="introduction" class="section">

## Introduction<a href="#introduction" class="headerlink" title="Permalink to this headline">¶</a>

This article explains the new features in Python 2.2.2, released on October 14, 2002. Python 2.2.2 is a bugfix release of Python 2.2, originally released on December 21, 2001.

Python 2.2 can be thought of as the “cleanup release”. There are some features such as generators and iterators that are completely new, but most of the changes, significant and far-reaching though they may be, are aimed at cleaning up irregularities and dark corners of the language design.

This article doesn’t attempt to provide a complete specification of the new features, but instead provides a convenient overview. For full details, you should refer to the documentation for Python 2.2, such as the <a href="https://docs.python.org/2.2/lib/lib.html" class="reference external">Python Library Reference</a> and the <a href="https://docs.python.org/2.2/ref/ref.html" class="reference external">Python Reference Manual</a>. If you want to understand the complete implementation and design rationale for a change, refer to the PEP for a particular new feature.

</div>

<div id="peps-252-and-253-type-and-class-changes" class="section">

## PEPs 252 and 253: Type and Class Changes<a href="#peps-252-and-253-type-and-class-changes" class="headerlink" title="Permalink to this headline">¶</a>

The largest and most far-reaching changes in Python 2.2 are to Python’s model of objects and classes. The changes should be backward compatible, so it’s likely that your code will continue to run unchanged, but the changes provide some amazing new capabilities. Before beginning this, the longest and most complicated section of this article, I’ll provide an overview of the changes and offer some comments.

A long time ago I wrote a Web page listing flaws in Python’s design. One of the most significant flaws was that it’s impossible to subclass Python types implemented in C. In particular, it’s not possible to subclass built-in types, so you can’t just subclass, say, lists in order to add a single useful method to them. The <a href="../library/userdict.html#module-UserList" class="reference internal" title="UserList: Class wrapper for list objects."><span class="pre"><code class="sourceCode python">UserList</code></span></a> module provides a class that supports all of the methods of lists and that can be subclassed further, but there’s lots of C code that expects a regular Python list and won’t accept a <a href="../library/userdict.html#UserList.UserList" class="reference internal" title="UserList.UserList"><span class="pre"><code class="sourceCode python">UserList</code></span></a> instance.

Python 2.2 fixes this, and in the process adds some exciting new capabilities. A brief summary:

- You can subclass built-in types such as lists and even integers, and your subclasses should work in every place that requires the original type.

- It’s now possible to define static and class methods, in addition to the instance methods available in previous versions of Python.

- It’s also possible to automatically call methods on accessing or setting an instance attribute by using a new mechanism called *properties*. Many uses of <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> can be rewritten to use properties instead, making the resulting code simpler and faster. As a small side benefit, attributes can now have docstrings, too.

- The list of legal attributes for an instance can be limited to a particular set using *slots*, making it possible to safeguard against typos and perhaps make more optimizations possible in future versions of Python.

Some users have voiced concern about all these changes. Sure, they say, the new features are neat and lend themselves to all sorts of tricks that weren’t possible in previous versions of Python, but they also make the language more complicated. Some people have said that they’ve always recommended Python for its simplicity, and feel that its simplicity is being lost.

Personally, I think there’s no need to worry. Many of the new features are quite esoteric, and you can write a lot of Python code without ever needed to be aware of them. Writing a simple class is no more difficult than it ever was, so you don’t need to bother learning or teaching them unless they’re actually needed. Some very complicated tasks that were previously only possible from C will now be possible in pure Python, and to my mind that’s all for the better.

I’m not going to attempt to cover every single corner case and small change that were required to make the new features work. Instead this section will paint only the broad strokes. See section <a href="#sect-rellinks" class="reference internal"><span class="std std-ref">Related Links</span></a>, “Related Links”, for further sources of information about Python 2.2’s new object model.

<div id="old-and-new-classes" class="section">

### Old and New Classes<a href="#old-and-new-classes" class="headerlink" title="Permalink to this headline">¶</a>

First, you should know that Python 2.2 really has two kinds of classes: classic or old-style classes, and new-style classes. The old-style class model is exactly the same as the class model in earlier versions of Python. All the new features described in this section apply only to new-style classes. This divergence isn’t intended to last forever; eventually old-style classes will be dropped, possibly in Python 3.0.

So how do you define a new-style class? You do it by subclassing an existing new-style class. Most of Python’s built-in types, such as integers, lists, dictionaries, and even files, are new-style classes now. A new-style class named <a href="../library/functions.html#object" class="reference internal" title="object"><span class="pre"><code class="sourceCode python"><span class="bu">object</span></code></span></a>, the base class for all built-in types, has also been added so if no built-in type is suitable, you can just subclass <a href="../library/functions.html#object" class="reference internal" title="object"><span class="pre"><code class="sourceCode python"><span class="bu">object</span></code></span></a>:

<div class="highlight-default notranslate">

<div class="highlight">

    class C(object):
        def __init__ (self):
            ...
        ...

</div>

</div>

This means that <a href="../reference/compound_stmts.html#class" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">class</code></span></a> statements that don’t have any base classes are always classic classes in Python 2.2. (Actually you can also change this by setting a module-level variable named <a href="../reference/datamodel.html#__metaclass__" class="reference internal" title="__metaclass__"><span class="pre"><code class="sourceCode python">__metaclass__</code></span></a> — see <span id="index-0" class="target"></span><a href="https://www.python.org/dev/peps/pep-0253" class="pep reference external"><strong>PEP 253</strong></a> for the details — but it’s easier to just subclass <a href="../c-api/object.html#object" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">object</code></span></a>.)

The type objects for the built-in types are available as built-ins, named using a clever trick. Python has always had built-in functions named <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span>()</code></span></a>, <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span>()</code></span></a>, and <a href="../library/functions.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span>()</code></span></a>. In 2.2, they aren’t functions any more, but type objects that behave as factories when called.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> int
    <type 'int'>
    >>> int('123')
    123

</div>

</div>

To make the set of types complete, new type objects such as <a href="../library/stdtypes.html#dict" class="reference internal" title="dict"><span class="pre"><code class="sourceCode python"><span class="bu">dict</span>()</code></span></a> and <a href="../library/functions.html#file" class="reference internal" title="file"><span class="pre"><code class="sourceCode python"><span class="bu">file</span>()</code></span></a> have been added. Here’s a more interesting example, adding a <span class="pre">`lock()`</span> method to file objects:

<div class="highlight-default notranslate">

<div class="highlight">

    class LockableFile(file):
        def lock (self, operation, length=0, start=0, whence=0):
            import fcntl
            return fcntl.lockf(self.fileno(), operation,
                               length, start, whence)

</div>

</div>

The now-obsolete <a href="../library/posixfile.html#module-posixfile" class="reference internal" title="posixfile: A file-like object with support for locking. (deprecated) (Unix)"><span class="pre"><code class="sourceCode python">posixfile</code></span></a> module contained a class that emulated all of a file object’s methods and also added a <span class="pre">`lock()`</span> method, but this class couldn’t be passed to internal functions that expected a built-in file, something which is possible with our new <span class="pre">`LockableFile`</span>.

</div>

<div id="descriptors" class="section">

### Descriptors<a href="#descriptors" class="headerlink" title="Permalink to this headline">¶</a>

In previous versions of Python, there was no consistent way to discover what attributes and methods were supported by an object. There were some informal conventions, such as defining <span class="pre">`__members__`</span> and <span class="pre">`__methods__`</span> attributes that were lists of names, but often the author of an extension type or a class wouldn’t bother to define them. You could fall back on inspecting the <a href="../library/stdtypes.html#object.__dict__" class="reference internal" title="object.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a> of an object, but when class inheritance or an arbitrary <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> hook were in use this could still be inaccurate.

The one big idea underlying the new class model is that an API for describing the attributes of an object using *descriptors* has been formalized. Descriptors specify the value of an attribute, stating whether it’s a method or a field. With the descriptor API, static methods and class methods become possible, as well as more exotic constructs.

Attribute descriptors are objects that live inside class objects, and have a few attributes of their own:

- <a href="../library/stdtypes.html#definition.__name__" class="reference internal" title="definition.__name__"><span class="pre"><code class="sourceCode python"><span class="va">__name__</span></code></span></a> is the attribute’s name.

- <span class="pre">`__doc__`</span> is the attribute’s docstring.

- <span class="pre">`__get__(object)`</span> is a method that retrieves the attribute value from *object*.

- <span class="pre">`__set__(object,`</span>` `<span class="pre">`value)`</span> sets the attribute on *object* to *value*.

- <span class="pre">`__delete__(object,`</span>` `<span class="pre">`value)`</span> deletes the *value* attribute of *object*.

For example, when you write <span class="pre">`obj.x`</span>, the steps that Python actually performs are:

<div class="highlight-default notranslate">

<div class="highlight">

    descriptor = obj.__class__.x
    descriptor.__get__(obj)

</div>

</div>

For methods, <span class="pre">`descriptor.__get__()`</span> returns a temporary object that’s callable, and wraps up the instance and the method to be called on it. This is also why static methods and class methods are now possible; they have descriptors that wrap up just the method, or the method and the class. As a brief explanation of these new kinds of methods, static methods aren’t passed the instance, and therefore resemble regular functions. Class methods are passed the class of the object, but not the object itself. Static and class methods are defined like this:

<div class="highlight-default notranslate">

<div class="highlight">

    class C(object):
        def f(arg1, arg2):
            ...
        f = staticmethod(f)

        def g(cls, arg1, arg2):
            ...
        g = classmethod(g)

</div>

</div>

The <a href="../library/functions.html#staticmethod" class="reference internal" title="staticmethod"><span class="pre"><code class="sourceCode python"><span class="bu">staticmethod</span>()</code></span></a> function takes the function <span class="pre">`f()`</span>, and returns it wrapped up in a descriptor so it can be stored in the class object. You might expect there to be special syntax for creating such methods (<span class="pre">`def`</span>` `<span class="pre">`static`</span>` `<span class="pre">`f`</span>, <span class="pre">`defstatic`</span>` `<span class="pre">`f()`</span>, or something like that) but no such syntax has been defined yet; that’s been left for future versions of Python.

More new features, such as slots and properties, are also implemented as new kinds of descriptors, and it’s not difficult to write a descriptor class that does something novel. For example, it would be possible to write a descriptor class that made it possible to write Eiffel-style preconditions and postconditions for a method. A class that used this feature might be defined like this:

<div class="highlight-default notranslate">

<div class="highlight">

    from eiffel import eiffelmethod

    class C(object):
        def f(self, arg1, arg2):
            # The actual function
            ...
        def pre_f(self):
            # Check preconditions
            ...
        def post_f(self):
            # Check postconditions
            ...

        f = eiffelmethod(f, pre_f, post_f)

</div>

</div>

Note that a person using the new <span class="pre">`eiffelmethod()`</span> doesn’t have to understand anything about descriptors. This is why I think the new features don’t increase the basic complexity of the language. There will be a few wizards who need to know about it in order to write <span class="pre">`eiffelmethod()`</span> or the ZODB or whatever, but most users will just write code on top of the resulting libraries and ignore the implementation details.

</div>

<div id="multiple-inheritance-the-diamond-rule" class="section">

### Multiple Inheritance: The Diamond Rule<a href="#multiple-inheritance-the-diamond-rule" class="headerlink" title="Permalink to this headline">¶</a>

Multiple inheritance has also been made more useful through changing the rules under which names are resolved. Consider this set of classes (diagram taken from <span id="index-1" class="target"></span><a href="https://www.python.org/dev/peps/pep-0253" class="pep reference external"><strong>PEP 253</strong></a> by Guido van Rossum):

<div class="highlight-default notranslate">

<div class="highlight">

          class A:
            ^ ^  def save(self): ...
           /   \
          /     \
         /       \
        /         \
    class B     class C:
        ^         ^  def save(self): ...
         \       /
          \     /
           \   /
            \ /
          class D

</div>

</div>

The lookup rule for classic classes is simple but not very smart; the base classes are searched depth-first, going from left to right. A reference to <span class="pre">`D.save()`</span> will search the classes <span class="pre">`D`</span>, <span class="pre">`B`</span>, and then <span class="pre">`A`</span>, where <span class="pre">`save()`</span> would be found and returned. <span class="pre">`C.save()`</span> would never be found at all. This is bad, because if <span class="pre">`C`</span>’s <span class="pre">`save()`</span> method is saving some internal state specific to <span class="pre">`C`</span>, not calling it will result in that state never getting saved.

New-style classes follow a different algorithm that’s a bit more complicated to explain, but does the right thing in this situation. (Note that Python 2.3 changes this algorithm to one that produces the same results in most cases, but produces more useful results for really complicated inheritance graphs.)

1.  List all the base classes, following the classic lookup rule and include a class multiple times if it’s visited repeatedly. In the above example, the list of visited classes is \[<span class="pre">`D`</span>, <span class="pre">`B`</span>, <span class="pre">`A`</span>, <span class="pre">`C`</span>, <span class="pre">`A`</span>\].

2.  Scan the list for duplicated classes. If any are found, remove all but one occurrence, leaving the *last* one in the list. In the above example, the list becomes \[<span class="pre">`D`</span>, <span class="pre">`B`</span>, <span class="pre">`C`</span>, <span class="pre">`A`</span>\] after dropping duplicates.

Following this rule, referring to <span class="pre">`D.save()`</span> will return <span class="pre">`C.save()`</span>, which is the behaviour we’re after. This lookup rule is the same as the one followed by Common Lisp. A new built-in function, <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a>, provides a way to get at a class’s superclasses without having to reimplement Python’s algorithm. The most commonly used form will be <span class="pre">`super(class,`</span>` `<span class="pre">`obj)`</span>, which returns a bound superclass object (not the actual class object). This form will be used in methods to call a method in the superclass; for example, <span class="pre">`D`</span>’s <span class="pre">`save()`</span> method would look like this:

<div class="highlight-default notranslate">

<div class="highlight">

    class D (B,C):
        def save (self):
            # Call superclass .save()
            super(D, self).save()
            # Save D's private information here
            ...

</div>

</div>

<a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a> can also return unbound superclass objects when called as <span class="pre">`super(class)`</span> or <span class="pre">`super(class1,`</span>` `<span class="pre">`class2)`</span>, but this probably won’t often be useful.

</div>

<div id="attribute-access" class="section">

### Attribute Access<a href="#attribute-access" class="headerlink" title="Permalink to this headline">¶</a>

A fair number of sophisticated Python classes define hooks for attribute access using <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a>; most commonly this is done for convenience, to make code more readable by automatically mapping an attribute access such as <span class="pre">`obj.parent`</span> into a method call such as <span class="pre">`obj.get_parent`</span>. Python 2.2 adds some new ways of controlling attribute access.

First, <span class="pre">`__getattr__(attr_name)`</span> is still supported by new-style classes, and nothing about it has changed. As before, it will be called when an attempt is made to access <span class="pre">`obj.foo`</span> and no attribute named <span class="pre">`foo`</span> is found in the instance’s dictionary.

New-style classes also support a new method, <span class="pre">`__getattribute__(attr_name)`</span>. The difference between the two methods is that <a href="../reference/datamodel.html#object.__getattribute__" class="reference internal" title="object.__getattribute__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattribute__</span>()</code></span></a> is *always* called whenever any attribute is accessed, while the old <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> is only called if <span class="pre">`foo`</span> isn’t found in the instance’s dictionary.

However, Python 2.2’s support for *properties* will often be a simpler way to trap attribute references. Writing a <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> method is complicated because to avoid recursion you can’t use regular attribute accesses inside them, and instead have to mess around with the contents of <a href="../library/stdtypes.html#object.__dict__" class="reference internal" title="object.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a>. <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a> methods also end up being called by Python when it checks for other methods such as <a href="../reference/datamodel.html#object.__repr__" class="reference internal" title="object.__repr__"><span class="pre"><code class="sourceCode python"><span class="fu">__repr__</span>()</code></span></a> or <a href="../reference/datamodel.html#object.__coerce__" class="reference internal" title="object.__coerce__"><span class="pre"><code class="sourceCode python"><span class="fu">__coerce__</span>()</code></span></a>, and so have to be written with this in mind. Finally, calling a function on every attribute access results in a sizable performance loss.

<a href="../library/functions.html#property" class="reference internal" title="property"><span class="pre"><code class="sourceCode python"><span class="bu">property</span></code></span></a> is a new built-in type that packages up three functions that get, set, or delete an attribute, and a docstring. For example, if you want to define a <span class="pre">`size`</span> attribute that’s computed, but also settable, you could write:

<div class="highlight-default notranslate">

<div class="highlight">

    class C(object):
        def get_size (self):
            result = ... computation ...
            return result
        def set_size (self, size):
            ... compute something based on the size
            and set internal state appropriately ...

        # Define a property.  The 'delete this attribute'
        # method is defined as None, so the attribute
        # can't be deleted.
        size = property(get_size, set_size,
                        None,
                        "Storage size of this instance")

</div>

</div>

That is certainly clearer and easier to write than a pair of <a href="../reference/datamodel.html#object.__getattr__" class="reference internal" title="object.__getattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__getattr__</span>()</code></span></a>/<a href="../reference/datamodel.html#object.__setattr__" class="reference internal" title="object.__setattr__"><span class="pre"><code class="sourceCode python"><span class="fu">__setattr__</span>()</code></span></a> methods that check for the <span class="pre">`size`</span> attribute and handle it specially while retrieving all other attributes from the instance’s <a href="../library/stdtypes.html#object.__dict__" class="reference internal" title="object.__dict__"><span class="pre"><code class="sourceCode python">__dict__</code></span></a>. Accesses to <span class="pre">`size`</span> are also the only ones which have to perform the work of calling a function, so references to other attributes run at their usual speed.

Finally, it’s possible to constrain the list of attributes that can be referenced on an object using the new <span class="pre">`__slots__`</span> class attribute. Python objects are usually very dynamic; at any time it’s possible to define a new attribute on an instance by just doing <span class="pre">`obj.new_attr=1`</span>. A new-style class can define a class attribute named <span class="pre">`__slots__`</span> to limit the legal attributes to a particular set of names. An example will make this clear:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> class C(object):
    ...     __slots__ = ('template', 'name')
    ...
    >>> obj = C()
    >>> print obj.template
    None
    >>> obj.template = 'Test'
    >>> print obj.template
    Test
    >>> obj.newattr = None
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    AttributeError: 'C' object has no attribute 'newattr'

</div>

</div>

Note how you get an <a href="../library/exceptions.html#exceptions.AttributeError" class="reference internal" title="exceptions.AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> on the attempt to assign to an attribute not listed in <span class="pre">`__slots__`</span>.

</div>

<div id="related-links" class="section">

<span id="sect-rellinks"></span>

### Related Links<a href="#related-links" class="headerlink" title="Permalink to this headline">¶</a>

This section has just been a quick overview of the new features, giving enough of an explanation to start you programming, but many details have been simplified or ignored. Where should you go to get a more complete picture?

<a href="https://docs.python.org/dev/howto/descriptor.html" class="reference external">https://docs.python.org/dev/howto/descriptor.html</a> is a lengthy tutorial introduction to the descriptor features, written by Guido van Rossum. If my description has whetted your appetite, go read this tutorial next, because it goes into much more detail about the new features while still remaining quite easy to read.

Next, there are two relevant PEPs, <span id="index-2" class="target"></span><a href="https://www.python.org/dev/peps/pep-0252" class="pep reference external"><strong>PEP 252</strong></a> and <span id="index-3" class="target"></span><a href="https://www.python.org/dev/peps/pep-0253" class="pep reference external"><strong>PEP 253</strong></a>. <span id="index-4" class="target"></span><a href="https://www.python.org/dev/peps/pep-0252" class="pep reference external"><strong>PEP 252</strong></a> is titled “Making Types Look More Like Classes”, and covers the descriptor API. <span id="index-5" class="target"></span><a href="https://www.python.org/dev/peps/pep-0253" class="pep reference external"><strong>PEP 253</strong></a> is titled “Subtyping Built-in Types”, and describes the changes to type objects that make it possible to subtype built-in objects. <span id="index-6" class="target"></span><a href="https://www.python.org/dev/peps/pep-0253" class="pep reference external"><strong>PEP 253</strong></a> is the more complicated PEP of the two, and at a few points the necessary explanations of types and meta-types may cause your head to explode. Both PEPs were written and implemented by Guido van Rossum, with substantial assistance from the rest of the Zope Corp. team.

Finally, there’s the ultimate authority: the source code. Most of the machinery for the type handling is in <span class="pre">`Objects/typeobject.c`</span>, but you should only resort to it after all other avenues have been exhausted, including posting a question to python-list or python-dev.

</div>

</div>

<div id="pep-234-iterators" class="section">

## PEP 234: Iterators<a href="#pep-234-iterators" class="headerlink" title="Permalink to this headline">¶</a>

Another significant addition to 2.2 is an iteration interface at both the C and Python levels. Objects can define how they can be looped over by callers.

In Python versions up to 2.1, the usual way to make <span class="pre">`for`</span>` `<span class="pre">`item`</span>` `<span class="pre">`in`</span>` `<span class="pre">`obj`</span> work is to define a <a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a> method that looks something like this:

<div class="highlight-default notranslate">

<div class="highlight">

    def __getitem__(self, index):
        return <next item>

</div>

</div>

<a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a> is more properly used to define an indexing operation on an object so that you can write <span class="pre">`obj[5]`</span> to retrieve the sixth element. It’s a bit misleading when you’re using this only to support <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> loops. Consider some file-like object that wants to be looped over; the *index* parameter is essentially meaningless, as the class probably assumes that a series of <a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a> calls will be made with *index* incrementing by one each time. In other words, the presence of the <a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a> method doesn’t mean that using <span class="pre">`file[5]`</span> to randomly access the sixth element will work, though it really should.

In Python 2.2, iteration can be implemented separately, and <a href="../reference/datamodel.html#object.__getitem__" class="reference internal" title="object.__getitem__"><span class="pre"><code class="sourceCode python"><span class="fu">__getitem__</span>()</code></span></a> methods can be limited to classes that really do support random access. The basic idea of iterators is simple. A new built-in function, <span class="pre">`iter(obj)`</span> or <span class="pre">`iter(C,`</span>` `<span class="pre">`sentinel)`</span>, is used to get an iterator. <span class="pre">`iter(obj)`</span> returns an iterator for the object *obj*, while <span class="pre">`iter(C,`</span>` `<span class="pre">`sentinel)`</span> returns an iterator that will invoke the callable object *C* until it returns *sentinel* to signal that the iterator is done.

Python classes can define an <a href="../reference/datamodel.html#object.__iter__" class="reference internal" title="object.__iter__"><span class="pre"><code class="sourceCode python"><span class="fu">__iter__</span>()</code></span></a> method, which should create and return a new iterator for the object; if the object is its own iterator, this method can just return <span class="pre">`self`</span>. In particular, iterators will usually be their own iterators. Extension types implemented in C can implement a <a href="../c-api/typeobj.html#c.PyTypeObject.tp_iter" class="reference internal" title="PyTypeObject.tp_iter"><span class="pre"><code class="sourceCode c">tp_iter</code></span></a> function in order to return an iterator, and extension types that want to behave as iterators can define a <a href="../c-api/typeobj.html#c.PyTypeObject.tp_iternext" class="reference internal" title="PyTypeObject.tp_iternext"><span class="pre"><code class="sourceCode c">tp_iternext</code></span></a> function.

So, after all this, what do iterators actually do? They have one required method, <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a>, which takes no arguments and returns the next value. When there are no more values to be returned, calling <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a> should raise the <a href="../library/exceptions.html#exceptions.StopIteration" class="reference internal" title="exceptions.StopIteration"><span class="pre"><code class="sourceCode python"><span class="pp">StopIteration</span></code></span></a> exception.

<div class="highlight-default notranslate">

<div class="highlight">

    >>> L = [1,2,3]
    >>> i = iter(L)
    >>> print i
    <iterator object at 0x8116870>
    >>> i.next()
    1
    >>> i.next()
    2
    >>> i.next()
    3
    >>> i.next()
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    StopIteration
    >>>

</div>

</div>

In 2.2, Python’s <a href="../reference/compound_stmts.html#for" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">for</code></span></a> statement no longer expects a sequence; it expects something for which <a href="../library/functions.html#iter" class="reference internal" title="iter"><span class="pre"><code class="sourceCode python"><span class="bu">iter</span>()</code></span></a> will return an iterator. For backward compatibility and convenience, an iterator is automatically constructed for sequences that don’t implement <a href="../reference/datamodel.html#object.__iter__" class="reference internal" title="object.__iter__"><span class="pre"><code class="sourceCode python"><span class="fu">__iter__</span>()</code></span></a> or a <a href="../c-api/typeobj.html#c.PyTypeObject.tp_iter" class="reference internal" title="PyTypeObject.tp_iter"><span class="pre"><code class="sourceCode c">tp_iter</code></span></a> slot, so <span class="pre">`for`</span>` `<span class="pre">`i`</span>` `<span class="pre">`in`</span>` `<span class="pre">`[1,2,3]`</span> will still work. Wherever the Python interpreter loops over a sequence, it’s been changed to use the iterator protocol. This means you can do things like this:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> L = [1,2,3]
    >>> i = iter(L)
    >>> a,b,c = i
    >>> a,b,c
    (1, 2, 3)

</div>

</div>

Iterator support has been added to some of Python’s basic types. Calling <a href="../library/functions.html#iter" class="reference internal" title="iter"><span class="pre"><code class="sourceCode python"><span class="bu">iter</span>()</code></span></a> on a dictionary will return an iterator which loops over its keys:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> m = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    ...      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    >>> for key in m: print key, m[key]
    ...
    Mar 3
    Feb 2
    Aug 8
    Sep 9
    May 5
    Jun 6
    Jul 7
    Jan 1
    Apr 4
    Nov 11
    Dec 12
    Oct 10

</div>

</div>

That’s just the default behaviour. If you want to iterate over keys, values, or key/value pairs, you can explicitly call the <span class="pre">`iterkeys()`</span>, <span class="pre">`itervalues()`</span>, or <span class="pre">`iteritems()`</span> methods to get an appropriate iterator. In a minor related change, the <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> operator now works on dictionaries, so <span class="pre">`key`</span>` `<span class="pre">`in`</span>` `<span class="pre">`dict`</span> is now equivalent to <span class="pre">`dict.has_key(key)`</span>.

Files also provide an iterator, which calls the <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline()</code></span></a> method until there are no more lines in the file. This means you can now read each line of a file using code like this:

<div class="highlight-default notranslate">

<div class="highlight">

    for line in file:
        # do something for each line
        ...

</div>

</div>

Note that you can only go forward in an iterator; there’s no way to get the previous element, reset the iterator, or make a copy of it. An iterator object could provide such additional capabilities, but the iterator protocol only requires a <a href="../library/functions.html#next" class="reference internal" title="next"><span class="pre"><code class="sourceCode python"><span class="bu">next</span>()</code></span></a> method.

<div class="admonition seealso">

See also

<span id="index-7" class="target"></span><a href="https://www.python.org/dev/peps/pep-0234" class="pep reference external"><strong>PEP 234</strong></a> - Iterators  
Written by Ka-Ping Yee and GvR; implemented by the Python Labs crew, mostly by GvR and Tim Peters.

</div>

</div>

<div id="pep-255-simple-generators" class="section">

## PEP 255: Simple Generators<a href="#pep-255-simple-generators" class="headerlink" title="Permalink to this headline">¶</a>

Generators are another new feature, one that interacts with the introduction of iterators.

You’re doubtless familiar with how function calls work in Python or C. When you call a function, it gets a private namespace where its local variables are created. When the function reaches a <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statement, the local variables are destroyed and the resulting value is returned to the caller. A later call to the same function will get a fresh new set of local variables. But, what if the local variables weren’t thrown away on exiting a function? What if you could later resume the function where it left off? This is what generators provide; they can be thought of as resumable functions.

Here’s the simplest example of a generator function:

<div class="highlight-default notranslate">

<div class="highlight">

    def generate_ints(N):
        for i in range(N):
            yield i

</div>

</div>

A new keyword, <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a>, was introduced for generators. Any function containing a <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement is a generator function; this is detected by Python’s bytecode compiler which compiles the function specially as a result. Because a new keyword was introduced, generators must be explicitly enabled in a module by including a <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`generators`</span> statement near the top of the module’s source code. In Python 2.3 this statement will become unnecessary.

When you call a generator function, it doesn’t return a single value; instead it returns a generator object that supports the iterator protocol. On executing the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement, the generator outputs the value of <span class="pre">`i`</span>, similar to a <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statement. The big difference between <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> and a <a href="../reference/simple_stmts.html#return" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">return</code></span></a> statement is that on reaching a <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> the generator’s state of execution is suspended and local variables are preserved. On the next call to the generator’s <span class="pre">`next()`</span> method, the function will resume executing immediately after the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement. (For complicated reasons, the <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> statement isn’t allowed inside the <a href="../reference/compound_stmts.html#try" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">try</code></span></a> block of a <a href="../reference/compound_stmts.html#try" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">try</code></span></a>…<a href="../reference/compound_stmts.html#finally" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">finally</code></span></a> statement; read <span id="index-8" class="target"></span><a href="https://www.python.org/dev/peps/pep-0255" class="pep reference external"><strong>PEP 255</strong></a> for a full explanation of the interaction between <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a> and exceptions.)

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
      File "<stdin>", line 1, in ?
      File "<stdin>", line 2, in generate_ints
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

Python doesn’t go nearly as far as Icon in adopting generators as a central concept. Generators are considered a new part of the core Python language, but learning or using them isn’t compulsory; if they don’t solve any problems that you have, feel free to ignore them. One novel feature of Python’s interface as compared to Icon’s is that a generator’s state is represented as a concrete object (the iterator) that can be passed around to other functions or stored in a data structure.

<div class="admonition seealso">

See also

<span id="index-9" class="target"></span><a href="https://www.python.org/dev/peps/pep-0255" class="pep reference external"><strong>PEP 255</strong></a> - Simple Generators  
Written by Neil Schemenauer, Tim Peters, Magnus Lie Hetland. Implemented mostly by Neil Schemenauer and Tim Peters, with other fixes from the Python Labs crew.

</div>

</div>

<div id="pep-237-unifying-long-integers-and-integers" class="section">

## PEP 237: Unifying Long Integers and Integers<a href="#pep-237-unifying-long-integers-and-integers" class="headerlink" title="Permalink to this headline">¶</a>

In recent versions, the distinction between regular integers, which are 32-bit values on most machines, and long integers, which can be of arbitrary size, was becoming an annoyance. For example, on platforms that support files larger than <span class="pre">`2**32`</span> bytes, the <span class="pre">`tell()`</span> method of file objects has to return a long integer. However, there were various bits of Python that expected plain integers and would raise an error if a long integer was provided instead. For example, in Python 1.5, only regular integers could be used as a slice index, and <span class="pre">`'abc'[1L:]`</span> would raise a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception with the message ‘slice index must be int’.

Python 2.2 will shift values from short to long integers as required. The ‘L’ suffix is no longer needed to indicate a long integer literal, as now the compiler will choose the appropriate type. (Using the ‘L’ suffix will be discouraged in future 2.x versions of Python, triggering a warning in Python 2.4, and probably dropped in Python 3.0.) Many operations that used to raise an <a href="../library/exceptions.html#exceptions.OverflowError" class="reference internal" title="exceptions.OverflowError"><span class="pre"><code class="sourceCode python"><span class="pp">OverflowError</span></code></span></a> will now return a long integer as their result. For example:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> 1234567890123
    1234567890123L
    >>> 2 ** 64
    18446744073709551616L

</div>

</div>

In most cases, integers and long integers will now be treated identically. You can still distinguish them with the <a href="../library/functions.html#type" class="reference internal" title="type"><span class="pre"><code class="sourceCode python"><span class="bu">type</span>()</code></span></a> built-in function, but that’s rarely needed.

<div class="admonition seealso">

See also

<span id="index-10" class="target"></span><a href="https://www.python.org/dev/peps/pep-0237" class="pep reference external"><strong>PEP 237</strong></a> - Unifying Long Integers and Integers  
Written by Moshe Zadka and Guido van Rossum. Implemented mostly by Guido van Rossum.

</div>

</div>

<div id="pep-238-changing-the-division-operator" class="section">

## PEP 238: Changing the Division Operator<a href="#pep-238-changing-the-division-operator" class="headerlink" title="Permalink to this headline">¶</a>

The most controversial change in Python 2.2 heralds the start of an effort to fix an old design flaw that’s been in Python from the beginning. Currently Python’s division operator, <span class="pre">`/`</span>, behaves like C’s division operator when presented with two integer arguments: it returns an integer result that’s truncated down when there would be a fractional part. For example, <span class="pre">`3/2`</span> is 1, not 1.5, and <span class="pre">`(-1)/2`</span> is -1, not -0.5. This means that the results of division can vary unexpectedly depending on the type of the two operands and because Python is dynamically typed, it can be difficult to determine the possible types of the operands.

(The controversy is over whether this is *really* a design flaw, and whether it’s worth breaking existing code to fix this. It’s caused endless discussions on python-dev, and in July 2001 erupted into a storm of acidly sarcastic postings on *comp.lang.python*. I won’t argue for either side here and will stick to describing what’s implemented in 2.2. Read <span id="index-11" class="target"></span><a href="https://www.python.org/dev/peps/pep-0238" class="pep reference external"><strong>PEP 238</strong></a> for a summary of arguments and counter-arguments.)

Because this change might break code, it’s being introduced very gradually. Python 2.2 begins the transition, but the switch won’t be complete until Python 3.0.

First, I’ll borrow some terminology from <span id="index-12" class="target"></span><a href="https://www.python.org/dev/peps/pep-0238" class="pep reference external"><strong>PEP 238</strong></a>. “True division” is the division that most non-programmers are familiar with: 3/2 is 1.5, 1/4 is 0.25, and so forth. “Floor division” is what Python’s <span class="pre">`/`</span> operator currently does when given integer operands; the result is the floor of the value returned by true division. “Classic division” is the current mixed behaviour of <span class="pre">`/`</span>; it returns the result of floor division when the operands are integers, and returns the result of true division when one of the operands is a floating-point number.

Here are the changes 2.2 introduces:

- A new operator, <span class="pre">`//`</span>, is the floor division operator. (Yes, we know it looks like C++’s comment symbol.) <span class="pre">`//`</span> *always* performs floor division no matter what the types of its operands are, so <span class="pre">`1`</span>` `<span class="pre">`//`</span>` `<span class="pre">`2`</span> is 0 and <span class="pre">`1.0`</span>` `<span class="pre">`//`</span>` `<span class="pre">`2.0`</span> is also 0.0.

  <span class="pre">`//`</span> is always available in Python 2.2; you don’t need to enable it using a <span class="pre">`__future__`</span> statement.

- By including a <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`division`</span> in a module, the <span class="pre">`/`</span> operator will be changed to return the result of true division, so <span class="pre">`1/2`</span> is 0.5. Without the <span class="pre">`__future__`</span> statement, <span class="pre">`/`</span> still means classic division. The default meaning of <span class="pre">`/`</span> will not change until Python 3.0.

- Classes can define methods called <a href="../reference/datamodel.html#object.__truediv__" class="reference internal" title="object.__truediv__"><span class="pre"><code class="sourceCode python"><span class="fu">__truediv__</span>()</code></span></a> and <a href="../reference/datamodel.html#object.__floordiv__" class="reference internal" title="object.__floordiv__"><span class="pre"><code class="sourceCode python"><span class="fu">__floordiv__</span>()</code></span></a> to overload the two division operators. At the C level, there are also slots in the <a href="../c-api/typeobj.html#c.PyNumberMethods" class="reference internal" title="PyNumberMethods"><span class="pre"><code class="sourceCode c">PyNumberMethods</code></span></a> structure so extension types can define the two operators.

- Python 2.2 supports some command-line arguments for testing whether code will work with the changed division semantics. Running python with <a href="../using/cmdline.html#cmdoption-q" class="reference internal"><span class="pre"><code class="xref std std-option docutils literal notranslate">-Q</code></span><code class="xref std std-option docutils literal notranslate"> </code><span class="pre"><code class="xref std std-option docutils literal notranslate">warn</code></span></a> will cause a warning to be issued whenever division is applied to two integers. You can use this to find code that’s affected by the change and fix it. By default, Python 2.2 will simply perform classic division without a warning; the warning will be turned on by default in Python 2.3.

<div class="admonition seealso">

See also

<span id="index-13" class="target"></span><a href="https://www.python.org/dev/peps/pep-0238" class="pep reference external"><strong>PEP 238</strong></a> - Changing the Division Operator  
Written by Moshe Zadka and Guido van Rossum. Implemented by Guido van Rossum..

</div>

</div>

<div id="unicode-changes" class="section">

## Unicode Changes<a href="#unicode-changes" class="headerlink" title="Permalink to this headline">¶</a>

Python’s Unicode support has been enhanced a bit in 2.2. Unicode strings are usually stored as UCS-2, as 16-bit unsigned integers. Python 2.2 can also be compiled to use UCS-4, 32-bit unsigned integers, as its internal encoding by supplying <span class="pre">`--enable-unicode=ucs4`</span> to the configure script. (It’s also possible to specify <span class="pre">`--disable-unicode`</span> to completely disable Unicode support.)

When built to use UCS-4 (a “wide Python”), the interpreter can natively handle Unicode characters from U+000000 to U+110000, so the range of legal values for the <a href="../library/functions.html#unichr" class="reference internal" title="unichr"><span class="pre"><code class="sourceCode python"><span class="bu">unichr</span>()</code></span></a> function is expanded accordingly. Using an interpreter compiled to use UCS-2 (a “narrow Python”), values greater than 65535 will still cause <a href="../library/functions.html#unichr" class="reference internal" title="unichr"><span class="pre"><code class="sourceCode python"><span class="bu">unichr</span>()</code></span></a> to raise a <a href="../library/exceptions.html#exceptions.ValueError" class="reference internal" title="exceptions.ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> exception. This is all described in <span id="index-14" class="target"></span><a href="https://www.python.org/dev/peps/pep-0261" class="pep reference external"><strong>PEP 261</strong></a>, “Support for ‘wide’ Unicode characters”; consult it for further details.

Another change is simpler to explain. Since their introduction, Unicode strings have supported an <span class="pre">`encode()`</span> method to convert the string to a selected encoding such as UTF-8 or Latin-1. A symmetric <span class="pre">`decode([*encoding*])`</span> method has been added to 8-bit strings (though not to Unicode strings) in 2.2. <span class="pre">`decode()`</span> assumes that the string is in the specified encoding and decodes it, returning whatever is returned by the codec.

Using this new feature, codecs have been added for tasks not directly related to Unicode. For example, codecs have been added for uu-encoding, MIME’s base64 encoding, and compression with the <a href="../library/zlib.html#module-zlib" class="reference internal" title="zlib: Low-level interface to compression and decompression routines compatible with gzip."><span class="pre"><code class="sourceCode python">zlib</code></span></a> module:

<div class="highlight-default notranslate">

<div class="highlight">

    >>> s = """Here is a lengthy piece of redundant, overly verbose,
    ... and repetitive text.
    ... """
    >>> data = s.encode('zlib')
    >>> data
    'x\x9c\r\xc9\xc1\r\x80 \x10\x04\xc0?Ul...'
    >>> data.decode('zlib')
    'Here is a lengthy piece of redundant, overly verbose,\nand repetitive text.\n'
    >>> print s.encode('uu')
    begin 666 <data>
    M2&5R92!I<R!A(&QE;F=T:'D@<&EE8V4@;V8@<F5D=6YD86YT+"!O=F5R;'D@
    >=F5R8F]S92P*86YD(')E<&5T:71I=F4@=&5X="X*

    end
    >>> "sheesh".encode('rot-13')
    'furrfu'

</div>

</div>

To convert a class instance to Unicode, a <a href="../reference/datamodel.html#object.__unicode__" class="reference internal" title="object.__unicode__"><span class="pre"><code class="sourceCode python"><span class="fu">__unicode__</span>()</code></span></a> method can be defined by a class, analogous to <a href="../reference/datamodel.html#object.__str__" class="reference internal" title="object.__str__"><span class="pre"><code class="sourceCode python"><span class="fu">__str__</span>()</code></span></a>.

<span class="pre">`encode()`</span>, <span class="pre">`decode()`</span>, and <a href="../reference/datamodel.html#object.__unicode__" class="reference internal" title="object.__unicode__"><span class="pre"><code class="sourceCode python"><span class="fu">__unicode__</span>()</code></span></a> were implemented by Marc-André Lemburg. The changes to support using UCS-4 internally were implemented by Fredrik Lundh and Martin von Löwis.

<div class="admonition seealso">

See also

<span id="index-15" class="target"></span><a href="https://www.python.org/dev/peps/pep-0261" class="pep reference external"><strong>PEP 261</strong></a> - Support for ‘wide’ Unicode characters  
Written by Paul Prescod.

</div>

</div>

<div id="pep-227-nested-scopes" class="section">

## PEP 227: Nested Scopes<a href="#pep-227-nested-scopes" class="headerlink" title="Permalink to this headline">¶</a>

In Python 2.1, statically nested scopes were added as an optional feature, to be enabled by a <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`nested_scopes`</span> directive. In 2.2 nested scopes no longer need to be specially enabled, and are now always present. The rest of this section is a copy of the description of nested scopes from my “What’s New in Python 2.1” document; if you read it when 2.1 came out, you can skip the rest of this section.

The largest change introduced in Python 2.1, and made complete in 2.2, is to Python’s scoping rules. In Python 2.0, at any given time there are at most three namespaces used to look up variable names: local, module-level, and the built-in namespace. This often surprised people because it didn’t match their intuitive expectations. For example, a nested recursive function definition doesn’t work:

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

The most significant change to Python 2.2 is that static scoping has been added to the language to fix this problem. As a first effect, the <span class="pre">`name=name`</span> default argument is now unnecessary in the above example. Put simply, when a given variable name is not assigned a value within a function (by an assignment, or the <a href="../reference/compound_stmts.html#def" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">def</code></span></a>, <a href="../reference/compound_stmts.html#class" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">class</code></span></a>, or <a href="../reference/simple_stmts.html#import" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">import</code></span></a> statements), references to the variable will be looked up in the local namespace of the enclosing scope. A more detailed explanation of the rules, and a dissection of the implementation, can be found in the PEP.

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

<div class="admonition seealso">

See also

<span id="index-16" class="target"></span><a href="https://www.python.org/dev/peps/pep-0227" class="pep reference external"><strong>PEP 227</strong></a> - Statically Nested Scopes  
Written and implemented by Jeremy Hylton.

</div>

</div>

<div id="new-and-improved-modules" class="section">

## New and Improved Modules<a href="#new-and-improved-modules" class="headerlink" title="Permalink to this headline">¶</a>

- The <a href="../library/xmlrpclib.html#module-xmlrpclib" class="reference internal" title="xmlrpclib: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpclib</code></span></a> module was contributed to the standard library by Fredrik Lundh, providing support for writing XML-RPC clients. XML-RPC is a simple remote procedure call protocol built on top of HTTP and XML. For example, the following snippet retrieves a list of RSS channels from the O’Reilly Network, and then lists the recent headlines for one channel:

  <div class="highlight-default notranslate">

  <div class="highlight">

      import xmlrpclib
      s = xmlrpclib.Server(
            'http://www.oreillynet.com/meerkat/xml-rpc/server.php')
      channels = s.meerkat.getChannels()
      # channels is a list of dictionaries, like this:
      # [{'id': 4, 'title': 'Freshmeat Daily News'}
      #  {'id': 190, 'title': '32Bits Online'},
      #  {'id': 4549, 'title': '3DGamers'}, ... ]

      # Get the items for one channel
      items = s.meerkat.getItems( {'channel': 4} )

      # 'items' is another list of dictionaries, like this:
      # [{'link': 'http://freshmeat.net/releases/52719/',
      #   'description': 'A utility which converts HTML to XSL FO.',
      #   'title': 'html2fo 0.3 (Default)'}, ... ]

  </div>

  </div>

  The <a href="../library/simplexmlrpcserver.html#module-SimpleXMLRPCServer" class="reference internal" title="SimpleXMLRPCServer: Basic XML-RPC server implementation."><span class="pre"><code class="sourceCode python">SimpleXMLRPCServer</code></span></a> module makes it easy to create straightforward XML-RPC servers. See <a href="http://www.xmlrpc.com/" class="reference external">http://www.xmlrpc.com/</a> for more information about XML-RPC.

- The new <a href="../library/hmac.html#module-hmac" class="reference internal" title="hmac: Keyed-Hashing for Message Authentication (HMAC) implementation"><span class="pre"><code class="sourceCode python">hmac</code></span></a> module implements the HMAC algorithm described by <span id="index-17" class="target"></span><a href="https://tools.ietf.org/html/rfc2104.html" class="rfc reference external"><strong>RFC 2104</strong></a>. (Contributed by Gerhard Häring.)

- Several functions that originally returned lengthy tuples now return pseudo-sequences that still behave like tuples but also have mnemonic attributes such as memberst_mtime or <span class="pre">`tm_year`</span>. The enhanced functions include <a href="../library/stat.html#module-stat" class="reference internal" title="stat: Utilities for interpreting the results of os.stat(), os.lstat() and os.fstat()."><span class="pre"><code class="sourceCode python">stat()</code></span></a>, <span class="pre">`fstat()`</span>, <a href="../library/statvfs.html#module-statvfs" class="reference internal" title="statvfs: Constants for interpreting the result of os.statvfs(). (deprecated)"><span class="pre"><code class="sourceCode python">statvfs()</code></span></a>, and <span class="pre">`fstatvfs()`</span> in the <a href="../library/os.html#module-os" class="reference internal" title="os: Miscellaneous operating system interfaces."><span class="pre"><code class="sourceCode python">os</code></span></a> module, and <span class="pre">`localtime()`</span>, <span class="pre">`gmtime()`</span>, and <span class="pre">`strptime()`</span> in the <a href="../library/time.html#module-time" class="reference internal" title="time: Time access and conversions."><span class="pre"><code class="sourceCode python">time</code></span></a> module.

  For example, to obtain a file’s size using the old tuples, you’d end up writing something like <span class="pre">`file_size`</span>` `<span class="pre">`=`</span>` `<span class="pre">`os.stat(filename)[stat.ST_SIZE]`</span>, but now this can be written more clearly as <span class="pre">`file_size`</span>` `<span class="pre">`=`</span>` `<span class="pre">`os.stat(filename).st_size`</span>.

  The original patch for this feature was contributed by Nick Mathewson.

- The Python profiler has been extensively reworked and various errors in its output have been corrected. (Contributed by Fred L. Drake, Jr. and Tim Peters.)

- The <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a> module can be compiled to support IPv6; specify the <span class="pre">`--enable-ipv6`</span> option to Python’s configure script. (Contributed by Jun-ichiro “itojun” Hagino.)

- Two new format characters were added to the <a href="../library/struct.html#module-struct" class="reference internal" title="struct: Interpret strings as packed binary data."><span class="pre"><code class="sourceCode python">struct</code></span></a> module for 64-bit integers on platforms that support the C <span class="pre">`long`</span>` `<span class="pre">`long`</span> type. <span class="pre">`q`</span> is for a signed 64-bit integer, and <span class="pre">`Q`</span> is for an unsigned one. The value is returned in Python’s long integer type. (Contributed by Tim Peters.)

- In the interpreter’s interactive mode, there’s a new built-in function <a href="../library/functions.html#help" class="reference internal" title="help"><span class="pre"><code class="sourceCode python"><span class="bu">help</span>()</code></span></a> that uses the <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module introduced in Python 2.1 to provide interactive help. <span class="pre">`help(object)`</span> displays any available help text about *object*. <a href="../library/functions.html#help" class="reference internal" title="help"><span class="pre"><code class="sourceCode python"><span class="bu">help</span>()</code></span></a> with no argument puts you in an online help utility, where you can enter the names of functions, classes, or modules to read their help text. (Contributed by Guido van Rossum, using Ka-Ping Yee’s <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module.)

- Various bugfixes and performance improvements have been made to the SRE engine underlying the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a> module. For example, the <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">re.sub()</code></span></a> and <a href="../library/re.html#re.split" class="reference internal" title="re.split"><span class="pre"><code class="sourceCode python">re.split()</code></span></a> functions have been rewritten in C. Another contributed patch speeds up certain Unicode character ranges by a factor of two, and a new <span class="pre">`finditer()`</span> method that returns an iterator over all the non-overlapping matches in a given string. (SRE is maintained by Fredrik Lundh. The BIGCHARSET patch was contributed by Martin von Löwis.)

- The <a href="../library/smtplib.html#module-smtplib" class="reference internal" title="smtplib: SMTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">smtplib</code></span></a> module now supports <span id="index-18" class="target"></span><a href="https://tools.ietf.org/html/rfc2487.html" class="rfc reference external"><strong>RFC 2487</strong></a>, “Secure SMTP over TLS”, so it’s now possible to encrypt the SMTP traffic between a Python program and the mail transport agent being handed a message. <a href="../library/smtplib.html#module-smtplib" class="reference internal" title="smtplib: SMTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">smtplib</code></span></a> also supports SMTP authentication. (Contributed by Gerhard Häring.)

- The <a href="../library/imaplib.html#module-imaplib" class="reference internal" title="imaplib: IMAP4 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">imaplib</code></span></a> module, maintained by Piers Lauder, has support for several new extensions: the NAMESPACE extension defined in <span id="index-19" class="target"></span><a href="https://tools.ietf.org/html/rfc2342.html" class="rfc reference external"><strong>RFC 2342</strong></a>, SORT, GETACL and SETACL. (Contributed by Anthony Baxter and Michel Pelletier.)

- The <a href="../library/rfc822.html#module-rfc822" class="reference internal" title="rfc822: Parse 2822 style mail messages. (deprecated)"><span class="pre"><code class="sourceCode python">rfc822</code></span></a> module’s parsing of email addresses is now compliant with <span id="index-20" class="target"></span><a href="https://tools.ietf.org/html/rfc2822.html" class="rfc reference external"><strong>RFC 2822</strong></a>, an update to <span id="index-21" class="target"></span><a href="https://tools.ietf.org/html/rfc822.html" class="rfc reference external"><strong>RFC 822</strong></a>. (The module’s name is *not* going to be changed to <span class="pre">`rfc2822`</span>.) A new package, <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages, including MIME documents."><span class="pre"><code class="sourceCode python">email</code></span></a>, has also been added for parsing and generating e-mail messages. (Contributed by Barry Warsaw, and arising out of his work on Mailman.)

- The <a href="../library/difflib.html#module-difflib" class="reference internal" title="difflib: Helpers for computing differences between objects."><span class="pre"><code class="sourceCode python">difflib</code></span></a> module now contains a new <span class="pre">`Differ`</span> class for producing human-readable lists of changes (a “delta”) between two sequences of lines of text. There are also two generator functions, <span class="pre">`ndiff()`</span> and <span class="pre">`restore()`</span>, which respectively return a delta from two sequences, or one of the original sequences from a delta. (Grunt work contributed by David Goodger, from ndiff.py code by Tim Peters who then did the generatorization.)

- New constants <span class="pre">`ascii_letters`</span>, <span class="pre">`ascii_lowercase`</span>, and <span class="pre">`ascii_uppercase`</span> were added to the <a href="../library/string.html#module-string" class="reference internal" title="string: Common string operations."><span class="pre"><code class="sourceCode python">string</code></span></a> module. There were several modules in the standard library that used <a href="../library/string.html#string.letters" class="reference internal" title="string.letters"><span class="pre"><code class="sourceCode python">string.letters</code></span></a> to mean the ranges A-Za-z, but that assumption is incorrect when locales are in use, because <a href="../library/string.html#string.letters" class="reference internal" title="string.letters"><span class="pre"><code class="sourceCode python">string.letters</code></span></a> varies depending on the set of legal characters defined by the current locale. The buggy modules have all been fixed to use <span class="pre">`ascii_letters`</span> instead. (Reported by an unknown person; fixed by Fred L. Drake, Jr.)

- The <a href="../library/mimetypes.html#module-mimetypes" class="reference internal" title="mimetypes: Mapping of filename extensions to MIME types."><span class="pre"><code class="sourceCode python">mimetypes</code></span></a> module now makes it easier to use alternative MIME-type databases by the addition of a <span class="pre">`MimeTypes`</span> class, which takes a list of filenames to be parsed. (Contributed by Fred L. Drake, Jr.)

- A <span class="pre">`Timer`</span> class was added to the <a href="../library/threading.html#module-threading" class="reference internal" title="threading: Higher-level threading interface."><span class="pre"><code class="sourceCode python">threading</code></span></a> module that allows scheduling an activity to happen at some future time. (Contributed by Itamar Shtull-Trauring.)

</div>

<div id="interpreter-changes-and-fixes" class="section">

## Interpreter Changes and Fixes<a href="#interpreter-changes-and-fixes" class="headerlink" title="Permalink to this headline">¶</a>

Some of the changes only affect people who deal with the Python interpreter at the C level because they’re writing Python extension modules, embedding the interpreter, or just hacking on the interpreter itself. If you only write Python code, none of the changes described here will affect you very much.

- Profiling and tracing functions can now be implemented in C, which can operate at much higher speeds than Python-based functions and should reduce the overhead of profiling and tracing. This will be of interest to authors of development environments for Python. Two new C functions were added to Python’s API, <a href="../c-api/init.html#c.PyEval_SetProfile" class="reference internal" title="PyEval_SetProfile"><span class="pre"><code class="sourceCode c">PyEval_SetProfile<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyEval_SetTrace" class="reference internal" title="PyEval_SetTrace"><span class="pre"><code class="sourceCode c">PyEval_SetTrace<span class="op">()</span></code></span></a>. The existing <a href="../library/sys.html#sys.setprofile" class="reference internal" title="sys.setprofile"><span class="pre"><code class="sourceCode python">sys.setprofile()</code></span></a> and <a href="../library/sys.html#sys.settrace" class="reference internal" title="sys.settrace"><span class="pre"><code class="sourceCode python">sys.settrace()</code></span></a> functions still exist, and have simply been changed to use the new C-level interface. (Contributed by Fred L. Drake, Jr.)

- Another low-level API, primarily of interest to implementors of Python debuggers and development tools, was added. <a href="../c-api/init.html#c.PyInterpreterState_Head" class="reference internal" title="PyInterpreterState_Head"><span class="pre"><code class="sourceCode c">PyInterpreterState_Head<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyInterpreterState_Next" class="reference internal" title="PyInterpreterState_Next"><span class="pre"><code class="sourceCode c">PyInterpreterState_Next<span class="op">()</span></code></span></a> let a caller walk through all the existing interpreter objects; <a href="../c-api/init.html#c.PyInterpreterState_ThreadHead" class="reference internal" title="PyInterpreterState_ThreadHead"><span class="pre"><code class="sourceCode c">PyInterpreterState_ThreadHead<span class="op">()</span></code></span></a> and <a href="../c-api/init.html#c.PyThreadState_Next" class="reference internal" title="PyThreadState_Next"><span class="pre"><code class="sourceCode c">PyThreadState_Next<span class="op">()</span></code></span></a> allow looping over all the thread states for a given interpreter. (Contributed by David Beazley.)

- The C-level interface to the garbage collector has been changed to make it easier to write extension types that support garbage collection and to debug misuses of the functions. Various functions have slightly different semantics, so a bunch of functions had to be renamed. Extensions that use the old API will still compile but will *not* participate in garbage collection, so updating them for 2.2 should be considered fairly high priority.

  To upgrade an extension module to the new API, perform the following steps:

- Rename <span class="pre">`Py_TPFLAGS_GC()`</span> to <span class="pre">`PyTPFLAGS_HAVE_GC()`</span>.

- Use <a href="../c-api/gcsupport.html#c.PyObject_GC_New" class="reference internal" title="PyObject_GC_New"><span class="pre"><code class="sourceCode c">PyObject_GC_New<span class="op">()</span></code></span></a> or <a href="../c-api/gcsupport.html#c.PyObject_GC_NewVar" class="reference internal" title="PyObject_GC_NewVar"><span class="pre"><code class="sourceCode c">PyObject_GC_NewVar<span class="op">()</span></code></span></a> to allocate  
  objects, and <a href="../c-api/gcsupport.html#c.PyObject_GC_Del" class="reference internal" title="PyObject_GC_Del"><span class="pre"><code class="sourceCode c">PyObject_GC_Del<span class="op">()</span></code></span></a> to deallocate them.

- Rename <span class="pre">`PyObject_GC_Init()`</span> to <a href="../c-api/gcsupport.html#c.PyObject_GC_Track" class="reference internal" title="PyObject_GC_Track"><span class="pre"><code class="sourceCode c">PyObject_GC_Track<span class="op">()</span></code></span></a> and  
  <span class="pre">`PyObject_GC_Fini()`</span> to <a href="../c-api/gcsupport.html#c.PyObject_GC_UnTrack" class="reference internal" title="PyObject_GC_UnTrack"><span class="pre"><code class="sourceCode c">PyObject_GC_UnTrack<span class="op">()</span></code></span></a>.

- Remove <span class="pre">`PyGC_HEAD_SIZE()`</span> from object size calculations.

- Remove calls to <span class="pre">`PyObject_AS_GC()`</span> and <span class="pre">`PyObject_FROM_GC()`</span>.

- A new <span class="pre">`et`</span> format sequence was added to <a href="../c-api/arg.html#c.PyArg_ParseTuple" class="reference internal" title="PyArg_ParseTuple"><span class="pre"><code class="sourceCode c">PyArg_ParseTuple<span class="op">()</span></code></span></a>; <span class="pre">`et`</span> takes both a parameter and an encoding name, and converts the parameter to the given encoding if the parameter turns out to be a Unicode string, or leaves it alone if it’s an 8-bit string, assuming it to already be in the desired encoding. This differs from the <span class="pre">`es`</span> format character, which assumes that 8-bit strings are in Python’s default ASCII encoding and converts them to the specified new encoding. (Contributed by M.-A. Lemburg, and used for the MBCS support on Windows described in the following section.)

- A different argument parsing function, <a href="../c-api/arg.html#c.PyArg_UnpackTuple" class="reference internal" title="PyArg_UnpackTuple"><span class="pre"><code class="sourceCode c">PyArg_UnpackTuple<span class="op">()</span></code></span></a>, has been added that’s simpler and presumably faster. Instead of specifying a format string, the caller simply gives the minimum and maximum number of arguments expected, and a set of pointers to <a href="../c-api/structures.html#c.PyObject" class="reference internal" title="PyObject"><span class="pre"><code class="sourceCode c">PyObject<span class="op">*</span></code></span></a> variables that will be filled in with argument values.

- Two new flags <a href="../c-api/structures.html#METH_NOARGS" class="reference internal" title="METH_NOARGS"><span class="pre"><code class="sourceCode python">METH_NOARGS</code></span></a> and <a href="../c-api/structures.html#METH_O" class="reference internal" title="METH_O"><span class="pre"><code class="sourceCode python">METH_O</code></span></a> are available in method definition tables to simplify implementation of methods with no arguments or a single untyped argument. Calling such methods is more efficient than calling a corresponding method that uses <a href="../c-api/structures.html#METH_VARARGS" class="reference internal" title="METH_VARARGS"><span class="pre"><code class="sourceCode python">METH_VARARGS</code></span></a>. Also, the old <a href="../c-api/structures.html#METH_OLDARGS" class="reference internal" title="METH_OLDARGS"><span class="pre"><code class="sourceCode python">METH_OLDARGS</code></span></a> style of writing C methods is now officially deprecated.

- Two new wrapper functions, <a href="../c-api/conversion.html#c.PyOS_snprintf" class="reference internal" title="PyOS_snprintf"><span class="pre"><code class="sourceCode c">PyOS_snprintf<span class="op">()</span></code></span></a> and <a href="../c-api/conversion.html#c.PyOS_vsnprintf" class="reference internal" title="PyOS_vsnprintf"><span class="pre"><code class="sourceCode c">PyOS_vsnprintf<span class="op">()</span></code></span></a> were added to provide cross-platform implementations for the relatively new <span class="pre">`snprintf()`</span> and <span class="pre">`vsnprintf()`</span> C lib APIs. In contrast to the standard <span class="pre">`sprintf()`</span> and <span class="pre">`vsprintf()`</span> functions, the Python versions check the bounds of the buffer used to protect against buffer overruns. (Contributed by M.-A. Lemburg.)

- The <a href="../c-api/tuple.html#c._PyTuple_Resize" class="reference internal" title="_PyTuple_Resize"><span class="pre"><code class="sourceCode c">_PyTuple_Resize<span class="op">()</span></code></span></a> function has lost an unused parameter, so now it takes 2 parameters instead of 3. The third argument was never used, and can simply be discarded when porting code from earlier versions to Python 2.2.

</div>

<div id="other-changes-and-fixes" class="section">

## Other Changes and Fixes<a href="#other-changes-and-fixes" class="headerlink" title="Permalink to this headline">¶</a>

As usual there were a bunch of other improvements and bugfixes scattered throughout the source tree. A search through the CVS change logs finds there were 527 patches applied and 683 bugs fixed between Python 2.1 and 2.2; 2.2.1 applied 139 patches and fixed 143 bugs; 2.2.2 applied 106 patches and fixed 82 bugs. These figures are likely to be underestimates.

Some of the more notable changes are:

- The code for the MacOS port for Python, maintained by Jack Jansen, is now kept in the main Python CVS tree, and many changes have been made to support MacOS X.

  The most significant change is the ability to build Python as a framework, enabled by supplying the <span class="pre">`--enable-framework`</span> option to the configure script when compiling Python. According to Jack Jansen, “This installs a self-contained Python installation plus the OS X framework “glue” into <span class="pre">`/Library/Frameworks/Python.framework`</span> (or another location of choice). For now there is little immediate added benefit to this (actually, there is the disadvantage that you have to change your PATH to be able to find Python), but it is the basis for creating a full-blown Python application, porting the MacPython IDE, possibly using Python as a standard OSA scripting language and much more.”

  Most of the MacPython toolbox modules, which interface to MacOS APIs such as windowing, QuickTime, scripting, etc. have been ported to OS X, but they’ve been left commented out in <span class="pre">`setup.py`</span>. People who want to experiment with these modules can uncomment them manually.

- Keyword arguments passed to built-in functions that don’t take them now cause a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception to be raised, with the message “*function* takes no keyword arguments”.

- Weak references, added in Python 2.1 as an extension module, are now part of the core because they’re used in the implementation of new-style classes. The <a href="../library/exceptions.html#exceptions.ReferenceError" class="reference internal" title="exceptions.ReferenceError"><span class="pre"><code class="sourceCode python"><span class="pp">ReferenceError</span></code></span></a> exception has therefore moved from the <a href="../library/weakref.html#module-weakref" class="reference internal" title="weakref: Support for weak references and weak dictionaries."><span class="pre"><code class="sourceCode python">weakref</code></span></a> module to become a built-in exception.

- A new script, <span class="pre">`Tools/scripts/cleanfuture.py`</span> by Tim Peters, automatically removes obsolete <span class="pre">`__future__`</span> statements from Python source code.

- An additional *flags* argument has been added to the built-in function <a href="../library/functions.html#compile" class="reference internal" title="compile"><span class="pre"><code class="sourceCode python"><span class="bu">compile</span>()</code></span></a>, so the behaviour of <span class="pre">`__future__`</span> statements can now be correctly observed in simulated shells, such as those presented by IDLE and other development environments. This is described in <span id="index-22" class="target"></span><a href="https://www.python.org/dev/peps/pep-0264" class="pep reference external"><strong>PEP 264</strong></a>. (Contributed by Michael Hudson.)

- The new license introduced with Python 1.6 wasn’t GPL-compatible. This is fixed by some minor textual changes to the 2.2 license, so it’s now legal to embed Python inside a GPLed program again. Note that Python itself is not GPLed, but instead is under a license that’s essentially equivalent to the BSD license, same as it always was. The license changes were also applied to the Python 2.0.1 and 2.1.1 releases.

- When presented with a Unicode filename on Windows, Python will now convert it to an MBCS encoded string, as used by the Microsoft file APIs. As MBCS is explicitly used by the file APIs, Python’s choice of ASCII as the default encoding turns out to be an annoyance. On Unix, the locale’s character set is used if <span class="pre">`locale.nl_langinfo(CODESET)`</span> is available. (Windows support was contributed by Mark Hammond with assistance from Marc-André Lemburg. Unix support was added by Martin von Löwis.)

- Large file support is now enabled on Windows. (Contributed by Tim Peters.)

- The <span class="pre">`Tools/scripts/ftpmirror.py`</span> script now parses a <span class="pre">`.netrc`</span> file, if you have one. (Contributed by Mike Romberg.)

- Some features of the object returned by the <a href="../library/functions.html#xrange" class="reference internal" title="xrange"><span class="pre"><code class="sourceCode python"><span class="bu">xrange</span>()</code></span></a> function are now deprecated, and trigger warnings when they’re accessed; they’ll disappear in Python 2.3. <a href="../library/functions.html#xrange" class="reference internal" title="xrange"><span class="pre"><code class="sourceCode python"><span class="bu">xrange</span></code></span></a> objects tried to pretend they were full sequence types by supporting slicing, sequence multiplication, and the <a href="../reference/expressions.html#in" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">in</code></span></a> operator, but these features were rarely used and therefore buggy. The <span class="pre">`tolist()`</span> method and the <span class="pre">`start`</span>, <span class="pre">`stop`</span>, and <span class="pre">`step`</span> attributes are also being deprecated. At the C level, the fourth argument to the <span class="pre">`PyRange_New()`</span> function, <span class="pre">`repeat`</span>, has also been deprecated.

- There were a bunch of patches to the dictionary implementation, mostly to fix potential core dumps if a dictionary contains objects that sneakily changed their hash value, or mutated the dictionary they were contained in. For a while python-dev fell into a gentle rhythm of Michael Hudson finding a case that dumped core, Tim Peters fixing the bug, Michael finding another case, and round and round it went.

- On Windows, Python can now be compiled with Borland C thanks to a number of patches contributed by Stephen Hansen, though the result isn’t fully functional yet. (But this *is* progress…)

- Another Windows enhancement: Wise Solutions generously offered PythonLabs use of their InstallerMaster 8.1 system. Earlier PythonLabs Windows installers used Wise 5.0a, which was beginning to show its age. (Packaged up by Tim Peters.)

- Files ending in <span class="pre">`.pyw`</span> can now be imported on Windows. <span class="pre">`.pyw`</span> is a Windows-only thing, used to indicate that a script needs to be run using PYTHONW.EXE instead of PYTHON.EXE in order to prevent a DOS console from popping up to display the output. This patch makes it possible to import such scripts, in case they’re also usable as modules. (Implemented by David Bolen.)

- On platforms where Python uses the C <span class="pre">`dlopen()`</span> function to load extension modules, it’s now possible to set the flags used by <span class="pre">`dlopen()`</span> using the <a href="../library/sys.html#sys.getdlopenflags" class="reference internal" title="sys.getdlopenflags"><span class="pre"><code class="sourceCode python">sys.getdlopenflags()</code></span></a> and <a href="../library/sys.html#sys.setdlopenflags" class="reference internal" title="sys.setdlopenflags"><span class="pre"><code class="sourceCode python">sys.setdlopenflags()</code></span></a> functions. (Contributed by Bram Stolk.)

- The <a href="../library/functions.html#pow" class="reference internal" title="pow"><span class="pre"><code class="sourceCode python"><span class="bu">pow</span>()</code></span></a> built-in function no longer supports 3 arguments when floating-point numbers are supplied. <span class="pre">`pow(x,`</span>` `<span class="pre">`y,`</span>` `<span class="pre">`z)`</span> returns <span class="pre">`(x**y)`</span>` `<span class="pre">`%`</span>` `<span class="pre">`z`</span>, but this is never useful for floating point numbers, and the final result varies unpredictably depending on the platform. A call such as <span class="pre">`pow(2.0,`</span>` `<span class="pre">`8.0,`</span>` `<span class="pre">`7.0)`</span> will now raise a <a href="../library/exceptions.html#exceptions.TypeError" class="reference internal" title="exceptions.TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> exception.

</div>

<div id="acknowledgements" class="section">

## Acknowledgements<a href="#acknowledgements" class="headerlink" title="Permalink to this headline">¶</a>

The author would like to thank the following people for offering suggestions, corrections and assistance with various drafts of this article: Fred Bremmer, Keith Briggs, Andrew Dalke, Fred L. Drake, Jr., Carel Fellinger, David Goodger, Mark Hammond, Stephen Hansen, Michael Hudson, Jack Jansen, Marc-André Lemburg, Martin von Löwis, Fredrik Lundh, Michael McLay, Nick Mathewson, Paul Moore, Gustavo Niemeyer, Don O’Donnell, Joonas Paalasma, Tim Peters, Jens Quade, Tom Reinhardt, Neil Schemenauer, Guido van Rossum, Greg Ward, Edward Welbourne.

</div>

</div>

</div>
