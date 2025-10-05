<div class="body" role="main">

<div id="what-s-new-in-python-3-4" class="section">

# What’s New In Python 3.4<a href="#what-s-new-in-python-3-4" class="headerlink" title="Link to this heading">¶</a>

Author<span class="colon">:</span>  
R. David Murray \<<a href="mailto:rdmurray%40bitdance.com" class="reference external">rdmurray<span>@</span>bitdance<span>.</span>com</a>\> (Editor)

This article explains the new features in Python 3.4, compared to 3.3. Python 3.4 was released on March 16, 2014. For full details, see the <a href="https://docs.python.org/3.4/whatsnew/changelog.html" class="reference external">changelog</a>.

<div class="admonition seealso">

See also

<span id="index-0" class="target"></span><a href="https://peps.python.org/pep-0429/" class="pep reference external"><strong>PEP 429</strong></a> – Python 3.4 Release Schedule

</div>

<div id="summary-release-highlights" class="section">

## Summary – Release Highlights<a href="#summary-release-highlights" class="headerlink" title="Link to this heading">¶</a>

New syntax features:

- No new syntax features were added in Python 3.4.

Other new features:

- <a href="#whatsnew-pep-453" class="reference internal"><span class="std std-ref">pip should always be available</span></a> (<span id="index-1" class="target"></span><a href="https://peps.python.org/pep-0453/" class="pep reference external"><strong>PEP 453</strong></a>).

- <a href="#whatsnew-pep-446" class="reference internal"><span class="std std-ref">Newly created file descriptors are non-inheritable</span></a> (<span id="index-2" class="target"></span><a href="https://peps.python.org/pep-0446/" class="pep reference external"><strong>PEP 446</strong></a>).

- command line option for <a href="#whatsnew-isolated-mode" class="reference internal"><span class="std std-ref">isolated mode</span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16499" class="reference external">bpo-16499</a>).

- <a href="#codec-handling-improvements" class="reference internal"><span class="std std-ref">improvements in the handling of codecs</span></a> that are not text encodings (multiple issues).

- <a href="#whatsnew-pep-451" class="reference internal"><span class="std std-ref">A ModuleSpec Type</span></a> for the Import System (<span id="index-3" class="target"></span><a href="https://peps.python.org/pep-0451/" class="pep reference external"><strong>PEP 451</strong></a>). (Affects importer authors.)

- The <a href="../library/marshal.html#module-marshal" class="reference internal" title="marshal: Convert Python objects to streams of bytes and back (with different constraints)."><span class="pre"><code class="sourceCode python">marshal</code></span></a> format has been made <a href="#whatsnew-marshal-3" class="reference internal"><span class="std std-ref">more compact and efficient</span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16475" class="reference external">bpo-16475</a>).

New library modules:

- <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a>: <a href="#whatsnew-asyncio" class="reference internal"><span class="std std-ref">New provisional API for asynchronous IO</span></a> (<span id="index-4" class="target"></span><a href="https://peps.python.org/pep-3156/" class="pep reference external"><strong>PEP 3156</strong></a>).

- <a href="../library/ensurepip.html#module-ensurepip" class="reference internal" title="ensurepip: Bootstrapping the &quot;pip&quot; installer into an existing Python installation or virtual environment."><span class="pre"><code class="sourceCode python">ensurepip</code></span></a>: <a href="#whatsnew-ensurepip" class="reference internal"><span class="std std-ref">Bootstrapping the pip installer</span></a> (<span id="index-5" class="target"></span><a href="https://peps.python.org/pep-0453/" class="pep reference external"><strong>PEP 453</strong></a>).

- <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">enum</code></span></a>: <a href="#whatsnew-enum" class="reference internal"><span class="std std-ref">Support for enumeration types</span></a> (<span id="index-6" class="target"></span><a href="https://peps.python.org/pep-0435/" class="pep reference external"><strong>PEP 435</strong></a>).

- <a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a>: <a href="#whatsnew-pathlib" class="reference internal"><span class="std std-ref">Object-oriented filesystem paths</span></a> (<span id="index-7" class="target"></span><a href="https://peps.python.org/pep-0428/" class="pep reference external"><strong>PEP 428</strong></a>).

- <a href="../library/selectors.html#module-selectors" class="reference internal" title="selectors: High-level I/O multiplexing."><span class="pre"><code class="sourceCode python">selectors</code></span></a>: <a href="#whatsnew-selectors" class="reference internal"><span class="std std-ref">High-level and efficient I/O multiplexing</span></a>, built upon the <a href="../library/select.html#module-select" class="reference internal" title="select: Wait for I/O completion on multiple streams."><span class="pre"><code class="sourceCode python">select</code></span></a> module primitives (part of <span id="index-8" class="target"></span><a href="https://peps.python.org/pep-3156/" class="pep reference external"><strong>PEP 3156</strong></a>).

- <a href="../library/statistics.html#module-statistics" class="reference internal" title="statistics: Mathematical statistics functions"><span class="pre"><code class="sourceCode python">statistics</code></span></a>: A basic <a href="#whatsnew-statistics" class="reference internal"><span class="std std-ref">numerically stable statistics library</span></a> (<span id="index-9" class="target"></span><a href="https://peps.python.org/pep-0450/" class="pep reference external"><strong>PEP 450</strong></a>).

- <a href="../library/tracemalloc.html#module-tracemalloc" class="reference internal" title="tracemalloc: Trace memory allocations."><span class="pre"><code class="sourceCode python">tracemalloc</code></span></a>: <a href="#whatsnew-tracemalloc" class="reference internal"><span class="std std-ref">Trace Python memory allocations</span></a> (<span id="index-10" class="target"></span><a href="https://peps.python.org/pep-0454/" class="pep reference external"><strong>PEP 454</strong></a>).

Significantly improved library modules:

- <a href="#whatsnew-singledispatch" class="reference internal"><span class="std std-ref">Single-dispatch generic functions</span></a> in <a href="../library/functools.html#module-functools" class="reference internal" title="functools: Higher-order functions and operations on callable objects."><span class="pre"><code class="sourceCode python">functools</code></span></a> (<span id="index-11" class="target"></span><a href="https://peps.python.org/pep-0443/" class="pep reference external"><strong>PEP 443</strong></a>).

- New <a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> <a href="#whatsnew-protocol-4" class="reference internal"><span class="std std-ref">protocol 4</span></a> (<span id="index-12" class="target"></span><a href="https://peps.python.org/pep-3154/" class="pep reference external"><strong>PEP 3154</strong></a>).

- <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> now has <a href="#whatsnew-multiprocessing-no-fork" class="reference internal"><span class="std std-ref">an option to avoid using os.fork on Unix</span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8713" class="reference external">bpo-8713</a>).

- <a href="../library/email.html#module-email" class="reference internal" title="email: Package supporting the parsing, manipulating, and generating email messages."><span class="pre"><code class="sourceCode python">email</code></span></a> has a new submodule, <a href="../library/email.contentmanager.html#module-email.contentmanager" class="reference internal" title="email.contentmanager: Storing and Retrieving Content from MIME Parts"><span class="pre"><code class="sourceCode python">contentmanager</code></span></a>, and a new <a href="../library/email.compat32-message.html#email.message.Message" class="reference internal" title="email.message.Message"><span class="pre"><code class="sourceCode python">Message</code></span></a> subclass (<a href="../library/email.message.html#email.message.EmailMessage" class="reference internal" title="email.message.EmailMessage"><span class="pre"><code class="sourceCode python">EmailMessage</code></span></a>) that <a href="#whatsnew-email-contentmanager" class="reference internal"><span class="std std-ref">simplify MIME handling</span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18891" class="reference external">bpo-18891</a>).

- The <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> and <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> modules are now capable of correct introspection of a much wider variety of callable objects, which improves the output of the Python <a href="../library/functions.html#help" class="reference internal" title="help"><span class="pre"><code class="sourceCode python"><span class="bu">help</span>()</code></span></a> system.

- The <a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> module API has been declared stable

Security improvements:

- <a href="#whatsnew-pep-456" class="reference internal"><span class="std std-ref">Secure and interchangeable hash algorithm</span></a> (<span id="index-13" class="target"></span><a href="https://peps.python.org/pep-0456/" class="pep reference external"><strong>PEP 456</strong></a>).

- <a href="#whatsnew-pep-446" class="reference internal"><span class="std std-ref">Make newly created file descriptors non-inheritable</span></a> (<span id="index-14" class="target"></span><a href="https://peps.python.org/pep-0446/" class="pep reference external"><strong>PEP 446</strong></a>) to avoid leaking file descriptors to child processes.

- New command line option for <a href="#whatsnew-isolated-mode" class="reference internal"><span class="std std-ref">isolated mode</span></a>, (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16499" class="reference external">bpo-16499</a>).

- <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> now has <a href="#whatsnew-multiprocessing-no-fork" class="reference internal"><span class="std std-ref">an option to avoid using os.fork on Unix</span></a>. *spawn* and *forkserver* are more secure because they avoid sharing data with child processes.

- <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> child processes on Windows no longer inherit all of the parent’s inheritable handles, only the necessary ones.

- A new <a href="../library/hashlib.html#hashlib.pbkdf2_hmac" class="reference internal" title="hashlib.pbkdf2_hmac"><span class="pre"><code class="sourceCode python">hashlib.pbkdf2_hmac()</code></span></a> function provides the <a href="https://en.wikipedia.org/wiki/PBKDF2" class="reference external">PKCS#5 password-based key derivation function 2</a>.

- <a href="#whatsnew-tls-11-12" class="reference internal"><span class="std std-ref">TLSv1.1 and TLSv1.2 support</span></a> for <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a>.

- <a href="#whatsnew34-win-cert-store" class="reference internal"><span class="std std-ref">Retrieving certificates from the Windows system cert store support</span></a> for <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a>.

- <a href="#whatsnew34-sni" class="reference internal"><span class="std std-ref">Server-side SNI (Server Name Indication) support</span></a> for <a href="../library/ssl.html#module-ssl" class="reference internal" title="ssl: TLS/SSL wrapper for socket objects"><span class="pre"><code class="sourceCode python">ssl</code></span></a>.

- The <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">ssl.SSLContext</code></span></a> class has a <a href="#whatsnew34-sslcontext" class="reference internal"><span class="std std-ref">lot of improvements</span></a>.

- All modules in the standard library that support SSL now support server certificate verification, including hostname matching (<span class="pre">`ssl.match_hostname()`</span>) and CRLs (Certificate Revocation lists, see <a href="../library/ssl.html#ssl.SSLContext.load_verify_locations" class="reference internal" title="ssl.SSLContext.load_verify_locations"><span class="pre"><code class="sourceCode python">ssl.SSLContext.load_verify_locations()</code></span></a>).

CPython implementation improvements:

- <a href="#whatsnew-pep-442" class="reference internal"><span class="std std-ref">Safe object finalization</span></a> (<span id="index-15" class="target"></span><a href="https://peps.python.org/pep-0442/" class="pep reference external"><strong>PEP 442</strong></a>).

- Leveraging <span id="index-16" class="target"></span><a href="https://peps.python.org/pep-0442/" class="pep reference external"><strong>PEP 442</strong></a>, in most cases <a href="#whatsnew-pep-442" class="reference internal"><span class="std std-ref">module globals are no longer set to None during finalization</span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18214" class="reference external">bpo-18214</a>).

- <a href="#whatsnew-pep-445" class="reference internal"><span class="std std-ref">Configurable memory allocators</span></a> (<span id="index-17" class="target"></span><a href="https://peps.python.org/pep-0445/" class="pep reference external"><strong>PEP 445</strong></a>).

- <a href="#whatsnew-pep-436" class="reference internal"><span class="std std-ref">Argument Clinic</span></a> (<span id="index-18" class="target"></span><a href="https://peps.python.org/pep-0436/" class="pep reference external"><strong>PEP 436</strong></a>).

Please read on for a comprehensive list of user-facing changes, including many other smaller improvements, CPython optimizations, deprecations, and potential porting issues.

</div>

<div id="new-features" class="section">

## New Features<a href="#new-features" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-453-explicit-bootstrapping-of-pip-in-python-installations" class="section">

<span id="whatsnew-pep-453"></span>

### PEP 453: Explicit Bootstrapping of PIP in Python Installations<a href="#pep-453-explicit-bootstrapping-of-pip-in-python-installations" class="headerlink" title="Link to this heading">¶</a>

<div id="bootstrapping-pip-by-default" class="section">

#### Bootstrapping pip By Default<a href="#bootstrapping-pip-by-default" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/ensurepip.html#module-ensurepip" class="reference internal" title="ensurepip: Bootstrapping the &quot;pip&quot; installer into an existing Python installation or virtual environment."><span class="pre"><code class="sourceCode python">ensurepip</code></span></a> module (defined in <span id="index-19" class="target"></span><a href="https://peps.python.org/pep-0453/" class="pep reference external"><strong>PEP 453</strong></a>) provides a standard cross-platform mechanism to bootstrap the pip installer into Python installations and virtual environments. The version of <span class="pre">`pip`</span> included with Python 3.4.0 is <span class="pre">`pip`</span> 1.5.4, and future 3.4.x maintenance releases will update the bundled version to the latest version of <span class="pre">`pip`</span> that is available at the time of creating the release candidate.

By default, the commands <span class="pre">`pipX`</span> and <span class="pre">`pipX.Y`</span> will be installed on all platforms (where X.Y stands for the version of the Python installation), along with the <span class="pre">`pip`</span> Python package and its dependencies. On Windows and in virtual environments on all platforms, the unversioned <span class="pre">`pip`</span> command will also be installed. On other platforms, the system wide unversioned <span class="pre">`pip`</span> command typically refers to the separately installed Python 2 version.

The <span class="pre">`pyvenv`</span> command line utility and the <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> module make use of the <a href="../library/ensurepip.html#module-ensurepip" class="reference internal" title="ensurepip: Bootstrapping the &quot;pip&quot; installer into an existing Python installation or virtual environment."><span class="pre"><code class="sourceCode python">ensurepip</code></span></a> module to make <span class="pre">`pip`</span> readily available in virtual environments. When using the command line utility, <span class="pre">`pip`</span> is installed by default, while when using the <a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> module <a href="../library/venv.html#venv-api" class="reference internal"><span class="std std-ref">API</span></a> installation of <span class="pre">`pip`</span> must be requested explicitly.

For CPython <a href="../using/unix.html#building-python-on-unix" class="reference internal"><span class="std std-ref">source builds on POSIX systems</span></a>, the <span class="pre">`make`</span>` `<span class="pre">`install`</span> and <span class="pre">`make`</span>` `<span class="pre">`altinstall`</span> commands bootstrap <span class="pre">`pip`</span> by default. This behaviour can be controlled through configure options, and overridden through Makefile options.

On Windows and Mac OS X, the CPython installers now default to installing <span class="pre">`pip`</span> along with CPython itself (users may opt out of installing it during the installation process). Window users will need to opt in to the automatic <span class="pre">`PATH`</span> modifications to have <span class="pre">`pip`</span> available from the command line by default, otherwise it can still be accessed through the Python launcher for Windows as <span class="pre">`py`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`pip`</span>.

As <span id="index-20" class="target"></span><a href="https://peps.python.org/pep-0453/#recommendations-for-downstream-distributors" class="pep reference external"><strong>discussed in the PEP</strong></a> platform packagers may choose not to install these commands by default, as long as, when invoked, they provide clear and simple directions on how to install them on that platform (usually using the system package manager).

<div class="admonition note">

Note

To avoid conflicts between parallel Python 2 and Python 3 installations, only the versioned <span class="pre">`pip3`</span> and <span class="pre">`pip3.4`</span> commands are bootstrapped by default when <span class="pre">`ensurepip`</span> is invoked directly - the <span class="pre">`--default-pip`</span> option is needed to also request the unversioned <span class="pre">`pip`</span> command. <span class="pre">`pyvenv`</span> and the Windows installer ensure that the unqualified <span class="pre">`pip`</span> command is made available in those environments, and <span class="pre">`pip`</span> can always be invoked via the <span class="pre">`-m`</span> switch rather than directly to avoid ambiguity on systems with multiple Python installations.

</div>

</div>

<div id="documentation-changes" class="section">

#### Documentation Changes<a href="#documentation-changes" class="headerlink" title="Link to this heading">¶</a>

As part of this change, the <a href="../installing/index.html#installing-index" class="reference internal"><span class="std std-ref">Installing Python Modules</span></a> and <a href="../distributing/index.html#distributing-index" class="reference internal"><span class="std std-ref">Distributing Python Modules</span></a> sections of the documentation have been completely redesigned as short getting started and FAQ documents. Most packaging documentation has now been moved out to the Python Packaging Authority maintained <a href="https://packaging.python.org" class="reference external">Python Packaging User Guide</a> and the documentation of the individual projects.

However, as this migration is currently still incomplete, the legacy versions of those guides remaining available as <a href="../extending/building.html#install-index" class="reference internal"><span class="std std-ref">Building C and C++ Extensions with setuptools</span></a> and <a href="../extending/building.html#setuptools-index" class="reference internal"><span class="std std-ref">Building C and C++ Extensions with setuptools</span></a>.

<div class="admonition seealso">

See also

<span id="index-21" class="target"></span><a href="https://peps.python.org/pep-0453/" class="pep reference external"><strong>PEP 453</strong></a> – Explicit bootstrapping of pip in Python installations  
PEP written by Donald Stufft and Nick Coghlan, implemented by Donald Stufft, Nick Coghlan, Martin von Löwis and Ned Deily.

</div>

</div>

</div>

<div id="pep-446-newly-created-file-descriptors-are-non-inheritable" class="section">

<span id="whatsnew-pep-446"></span>

### PEP 446: Newly Created File Descriptors Are Non-Inheritable<a href="#pep-446-newly-created-file-descriptors-are-non-inheritable" class="headerlink" title="Link to this heading">¶</a>

<span id="index-22" class="target"></span><a href="https://peps.python.org/pep-0446/" class="pep reference external"><strong>PEP 446</strong></a> makes newly created file descriptors <a href="../library/os.html#fd-inheritance" class="reference internal"><span class="std std-ref">non-inheritable</span></a>. In general, this is the behavior an application will want: when launching a new process, having currently open files also open in the new process can lead to all sorts of hard to find bugs, and potentially to security issues.

However, there are occasions when inheritance is desired. To support these cases, the following new functions and methods are available:

- <a href="../library/os.html#os.get_inheritable" class="reference internal" title="os.get_inheritable"><span class="pre"><code class="sourceCode python">os.get_inheritable()</code></span></a>, <a href="../library/os.html#os.set_inheritable" class="reference internal" title="os.set_inheritable"><span class="pre"><code class="sourceCode python">os.set_inheritable()</code></span></a>

- <a href="../library/os.html#os.get_handle_inheritable" class="reference internal" title="os.get_handle_inheritable"><span class="pre"><code class="sourceCode python">os.get_handle_inheritable()</code></span></a>, <a href="../library/os.html#os.set_handle_inheritable" class="reference internal" title="os.set_handle_inheritable"><span class="pre"><code class="sourceCode python">os.set_handle_inheritable()</code></span></a>

- <a href="../library/socket.html#socket.socket.get_inheritable" class="reference internal" title="socket.socket.get_inheritable"><span class="pre"><code class="sourceCode python">socket.socket.get_inheritable()</code></span></a>, <a href="../library/socket.html#socket.socket.set_inheritable" class="reference internal" title="socket.socket.set_inheritable"><span class="pre"><code class="sourceCode python">socket.socket.set_inheritable()</code></span></a>

<div class="admonition seealso">

See also

<span id="index-23" class="target"></span><a href="https://peps.python.org/pep-0446/" class="pep reference external"><strong>PEP 446</strong></a> – Make newly created file descriptors non-inheritable  
PEP written and implemented by Victor Stinner.

</div>

</div>

<div id="improvements-to-codec-handling" class="section">

<span id="codec-handling-improvements"></span>

### Improvements to Codec Handling<a href="#improvements-to-codec-handling" class="headerlink" title="Link to this heading">¶</a>

Since it was first introduced, the <a href="../library/codecs.html#module-codecs" class="reference internal" title="codecs: Encode and decode data and streams."><span class="pre"><code class="sourceCode python">codecs</code></span></a> module has always been intended to operate as a type-neutral dynamic encoding and decoding system. However, its close coupling with the Python text model, especially the type restricted convenience methods on the builtin <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> types, has historically obscured that fact.

As a key step in clarifying the situation, the <a href="../library/codecs.html#codecs.encode" class="reference internal" title="codecs.encode"><span class="pre"><code class="sourceCode python">codecs.encode()</code></span></a> and <a href="../library/codecs.html#codecs.decode" class="reference internal" title="codecs.decode"><span class="pre"><code class="sourceCode python">codecs.decode()</code></span></a> convenience functions are now properly documented in Python 2.7, 3.3 and 3.4. These functions have existed in the <a href="../library/codecs.html#module-codecs" class="reference internal" title="codecs: Encode and decode data and streams."><span class="pre"><code class="sourceCode python">codecs</code></span></a> module (and have been covered by the regression test suite) since Python 2.4, but were previously only discoverable through runtime introspection.

Unlike the convenience methods on <a href="../library/stdtypes.html#str" class="reference internal" title="str"><span class="pre"><code class="sourceCode python"><span class="bu">str</span></code></span></a>, <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>, the <a href="../library/codecs.html#module-codecs" class="reference internal" title="codecs: Encode and decode data and streams."><span class="pre"><code class="sourceCode python">codecs</code></span></a> convenience functions support arbitrary codecs in both Python 2 and Python 3, rather than being limited to Unicode text encodings (in Python 3) or <span class="pre">`basestring`</span> \<-\> <span class="pre">`basestring`</span> conversions (in Python 2).

In Python 3.4, the interpreter is able to identify the known non-text encodings provided in the standard library and direct users towards these general purpose convenience functions when appropriate:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> b"abcdef".decode("hex")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    LookupError: 'hex' is not a text encoding; use codecs.decode() to handle arbitrary codecs

    >>> "hello".encode("rot13")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    LookupError: 'rot13' is not a text encoding; use codecs.encode() to handle arbitrary codecs

    >>> open("foo.txt", encoding="hex")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    LookupError: 'hex' is not a text encoding; use codecs.open() to handle arbitrary codecs

</div>

</div>

In a related change, whenever it is feasible without breaking backwards compatibility, exceptions raised during encoding and decoding operations are wrapped in a chained exception of the same type that mentions the name of the codec responsible for producing the error:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import codecs

    >>> codecs.decode(b"abcdefgh", "hex")
    Traceback (most recent call last):
      File "/usr/lib/python3.4/encodings/hex_codec.py", line 20, in hex_decode
        return (binascii.a2b_hex(input), len(input))
    binascii.Error: Non-hexadecimal digit found

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    binascii.Error: decoding with 'hex' codec failed (Error: Non-hexadecimal digit found)

    >>> codecs.encode("hello", "bz2")
    Traceback (most recent call last):
      File "/usr/lib/python3.4/encodings/bz2_codec.py", line 17, in bz2_encode
        return (bz2.compress(input), len(input))
      File "/usr/lib/python3.4/bz2.py", line 498, in compress
        return comp.compress(data) + comp.flush()
    TypeError: 'str' does not support the buffer interface

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: encoding with 'bz2' codec failed (TypeError: 'str' does not support the buffer interface)

</div>

</div>

Finally, as the examples above show, these improvements have permitted the restoration of the convenience aliases for the non-Unicode codecs that were themselves restored in Python 3.2. This means that encoding binary data to and from its hexadecimal representation (for example) can now be written as:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> from codecs import encode, decode
    >>> encode(b"hello", "hex")
    b'68656c6c6f'
    >>> decode(b"68656c6c6f", "hex")
    b'hello'

</div>

</div>

The binary and text transforms provided in the standard library are detailed in <a href="../library/codecs.html#binary-transforms" class="reference internal"><span class="std std-ref">Binary Transforms</span></a> and <a href="../library/codecs.html#text-transforms" class="reference internal"><span class="std std-ref">Text Transforms</span></a>.

(Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7475" class="reference external">bpo-7475</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17827" class="reference external">bpo-17827</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17828" class="reference external">bpo-17828</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19619" class="reference external">bpo-19619</a>.)

</div>

<div id="pep-451-a-modulespec-type-for-the-import-system" class="section">

<span id="whatsnew-pep-451"></span>

### PEP 451: A ModuleSpec Type for the Import System<a href="#pep-451-a-modulespec-type-for-the-import-system" class="headerlink" title="Link to this heading">¶</a>

<span id="index-24" class="target"></span><a href="https://peps.python.org/pep-0451/" class="pep reference external"><strong>PEP 451</strong></a> provides an encapsulation of the information about a module that the import machinery will use to load it (that is, a module specification). This helps simplify both the import implementation and several import-related APIs. The change is also a stepping stone for <a href="https://mail.python.org/pipermail/python-dev/2013-November/130111.html" class="reference external">several future import-related improvements</a>.

The public-facing changes from the PEP are entirely backward-compatible. Furthermore, they should be transparent to everyone but importer authors. Key finder and loader methods have been deprecated, but they will continue working. New importers should use the new methods described in the PEP. Existing importers should be updated to implement the new methods. See the <a href="#deprecated-3-4" class="reference internal"><span class="std std-ref">Deprecated</span></a> section for a list of methods that should be replaced and their replacements.

</div>

<div id="other-language-changes" class="section">

### Other Language Changes<a href="#other-language-changes" class="headerlink" title="Link to this heading">¶</a>

Some smaller changes made to the core Python language are:

- Unicode database updated to UCD version 6.3.

- <a href="../library/functions.html#min" class="reference internal" title="min"><span class="pre"><code class="sourceCode python"><span class="bu">min</span>()</code></span></a> and <a href="../library/functions.html#max" class="reference internal" title="max"><span class="pre"><code class="sourceCode python"><span class="bu">max</span>()</code></span></a> now accept a *default* keyword-only argument that can be used to specify the value they return if the iterable they are evaluating has no elements. (Contributed by Julian Berman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18111" class="reference external">bpo-18111</a>.)

- Module objects are now <a href="../library/weakref.html#mod-weakref" class="reference internal"><span class="std std-ref">weakly referenceable</span></a>.

- Module <span class="pre">`__file__`</span> attributes (and related values) should now always contain absolute paths by default, with the sole exception of <span class="pre">`__main__.__file__`</span> when a script has been executed directly using a relative path. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18416" class="reference external">bpo-18416</a>.)

- All the UTF-\* codecs (except UTF-7) now reject surrogates during both encoding and decoding unless the <span class="pre">`surrogatepass`</span> error handler is used, with the exception of the UTF-16 decoder (which accepts valid surrogate pairs) and the UTF-16 encoder (which produces them while encoding non-BMP characters). (Contributed by Victor Stinner, Kang-Hao (Kenny) Lu and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12892" class="reference external">bpo-12892</a>.)

- New German EBCDIC <a href="../library/codecs.html#standard-encodings" class="reference internal"><span class="std std-ref">codec</span></a> <span class="pre">`cp273`</span>. (Contributed by Michael Bierenfeld and Andrew Kuchling in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1097797" class="reference external">bpo-1097797</a>.)

- New Ukrainian <a href="../library/codecs.html#standard-encodings" class="reference internal"><span class="std std-ref">codec</span></a> <span class="pre">`cp1125`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19668" class="reference external">bpo-19668</a>.)

- <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a>.join() and <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a>.join() now accept arbitrary buffer objects as arguments. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15958" class="reference external">bpo-15958</a>.)

- The <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> constructor now accepts any object that has an <span class="pre">`__index__`</span> method for its *base* argument. (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16772" class="reference external">bpo-16772</a>.)

- Frame objects now have a <a href="../reference/datamodel.html#frame.clear" class="reference internal" title="frame.clear"><span class="pre"><code class="sourceCode python">clear()</code></span></a> method that clears all references to local variables from the frame. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17934" class="reference external">bpo-17934</a>.)

- <a href="../library/stdtypes.html#memoryview" class="reference internal" title="memoryview"><span class="pre"><code class="sourceCode python"><span class="bu">memoryview</span></code></span></a> is now registered as a <a href="../library/collections.abc.html#module-collections.abc" class="reference internal" title="collections.abc: Abstract base classes for containers"><span class="pre"><code class="sourceCode python">Sequence</code></span></a>, and supports the <a href="../library/functions.html#reversed" class="reference internal" title="reversed"><span class="pre"><code class="sourceCode python"><span class="bu">reversed</span>()</code></span></a> builtin. (Contributed by Nick Coghlan and Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18690" class="reference external">bpo-18690</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19078" class="reference external">bpo-19078</a>.)

- Signatures reported by <a href="../library/functions.html#help" class="reference internal" title="help"><span class="pre"><code class="sourceCode python"><span class="bu">help</span>()</code></span></a> have been modified and improved in several cases as a result of the introduction of Argument Clinic and other changes to the <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> and <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> modules.

- <a href="../reference/datamodel.html#object.__length_hint__" class="reference internal" title="object.__length_hint__"><span class="pre"><code class="sourceCode python"><span class="fu">__length_hint__</span>()</code></span></a> is now part of the formal language specification (see <span id="index-25" class="target"></span><a href="https://peps.python.org/pep-0424/" class="pep reference external"><strong>PEP 424</strong></a>). (Contributed by Armin Ronacher in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16148" class="reference external">bpo-16148</a>.)

</div>

</div>

<div id="new-modules" class="section">

## New Modules<a href="#new-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="asyncio" class="section">

<span id="whatsnew-asyncio"></span>

### asyncio<a href="#asyncio" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> module (defined in <span id="index-26" class="target"></span><a href="https://peps.python.org/pep-3156/" class="pep reference external"><strong>PEP 3156</strong></a>) provides a standard pluggable event loop model for Python, providing solid asynchronous IO support in the standard library, and making it easier for other event loop implementations to interoperate with the standard library and each other.

For Python 3.4, this module is considered a <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional API</span></a>.

<div class="admonition seealso">

See also

<span id="index-27" class="target"></span><a href="https://peps.python.org/pep-3156/" class="pep reference external"><strong>PEP 3156</strong></a> – Asynchronous IO Support Rebooted: the “asyncio” Module  
PEP written and implementation led by Guido van Rossum.

</div>

</div>

<div id="ensurepip" class="section">

<span id="whatsnew-ensurepip"></span>

### ensurepip<a href="#ensurepip" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/ensurepip.html#module-ensurepip" class="reference internal" title="ensurepip: Bootstrapping the &quot;pip&quot; installer into an existing Python installation or virtual environment."><span class="pre"><code class="sourceCode python">ensurepip</code></span></a> module is the primary infrastructure for the <span id="index-28" class="target"></span><a href="https://peps.python.org/pep-0453/" class="pep reference external"><strong>PEP 453</strong></a> implementation. In the normal course of events end users will not need to interact with this module, but it can be used to manually bootstrap <span class="pre">`pip`</span> if the automated bootstrapping into an installation or virtual environment was declined.

<a href="../library/ensurepip.html#module-ensurepip" class="reference internal" title="ensurepip: Bootstrapping the &quot;pip&quot; installer into an existing Python installation or virtual environment."><span class="pre"><code class="sourceCode python">ensurepip</code></span></a> includes a bundled copy of <span class="pre">`pip`</span>, up-to-date as of the first release candidate of the release of CPython with which it ships (this applies to both maintenance releases and feature releases). <span class="pre">`ensurepip`</span> does not access the internet. If the installation has internet access, after <span class="pre">`ensurepip`</span> is run the bundled <span class="pre">`pip`</span> can be used to upgrade <span class="pre">`pip`</span> to a more recent release than the bundled one. (Note that such an upgraded version of <span class="pre">`pip`</span> is considered to be a separately installed package and will not be removed if Python is uninstalled.)

The module is named *ensure*pip because if called when <span class="pre">`pip`</span> is already installed, it does nothing. It also has an <span class="pre">`--upgrade`</span> option that will cause it to install the bundled copy of <span class="pre">`pip`</span> if the existing installed version of <span class="pre">`pip`</span> is older than the bundled copy.

</div>

<div id="enum" class="section">

<span id="whatsnew-enum"></span>

### enum<a href="#enum" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">enum</code></span></a> module (defined in <span id="index-29" class="target"></span><a href="https://peps.python.org/pep-0435/" class="pep reference external"><strong>PEP 435</strong></a>) provides a standard implementation of enumeration types, allowing other modules (such as <a href="../library/socket.html#module-socket" class="reference internal" title="socket: Low-level networking interface."><span class="pre"><code class="sourceCode python">socket</code></span></a>) to provide more informative error messages and better debugging support by replacing opaque integer constants with backwards compatible enumeration values.

<div class="admonition seealso">

See also

<span id="index-30" class="target"></span><a href="https://peps.python.org/pep-0435/" class="pep reference external"><strong>PEP 435</strong></a> – Adding an Enum type to the Python standard library  
PEP written by Barry Warsaw, Eli Bendersky and Ethan Furman, implemented by Ethan Furman.

</div>

</div>

<div id="pathlib" class="section">

<span id="whatsnew-pathlib"></span>

### pathlib<a href="#pathlib" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/pathlib.html#module-pathlib" class="reference internal" title="pathlib: Object-oriented filesystem paths"><span class="pre"><code class="sourceCode python">pathlib</code></span></a> module offers classes representing filesystem paths with semantics appropriate for different operating systems. Path classes are divided between *pure paths*, which provide purely computational operations without I/O, and *concrete paths*, which inherit from pure paths but also provide I/O operations.

For Python 3.4, this module is considered a <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional API</span></a>.

<div class="admonition seealso">

See also

<span id="index-31" class="target"></span><a href="https://peps.python.org/pep-0428/" class="pep reference external"><strong>PEP 428</strong></a> – The pathlib module – object-oriented filesystem paths  
PEP written and implemented by Antoine Pitrou.

</div>

</div>

<div id="selectors" class="section">

<span id="whatsnew-selectors"></span>

### selectors<a href="#selectors" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/selectors.html#module-selectors" class="reference internal" title="selectors: High-level I/O multiplexing."><span class="pre"><code class="sourceCode python">selectors</code></span></a> module (created as part of implementing <span id="index-32" class="target"></span><a href="https://peps.python.org/pep-3156/" class="pep reference external"><strong>PEP 3156</strong></a>) allows high-level and efficient I/O multiplexing, built upon the <a href="../library/select.html#module-select" class="reference internal" title="select: Wait for I/O completion on multiple streams."><span class="pre"><code class="sourceCode python">select</code></span></a> module primitives.

</div>

<div id="statistics" class="section">

<span id="whatsnew-statistics"></span>

### statistics<a href="#statistics" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/statistics.html#module-statistics" class="reference internal" title="statistics: Mathematical statistics functions"><span class="pre"><code class="sourceCode python">statistics</code></span></a> module (defined in <span id="index-33" class="target"></span><a href="https://peps.python.org/pep-0450/" class="pep reference external"><strong>PEP 450</strong></a>) offers some core statistics functionality directly in the standard library. This module supports calculation of the mean, median, mode, variance and standard deviation of a data series.

<div class="admonition seealso">

See also

<span id="index-34" class="target"></span><a href="https://peps.python.org/pep-0450/" class="pep reference external"><strong>PEP 450</strong></a> – Adding A Statistics Module To The Standard Library  
PEP written and implemented by Steven D’Aprano

</div>

</div>

<div id="tracemalloc" class="section">

<span id="whatsnew-tracemalloc"></span>

### tracemalloc<a href="#tracemalloc" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/tracemalloc.html#module-tracemalloc" class="reference internal" title="tracemalloc: Trace memory allocations."><span class="pre"><code class="sourceCode python">tracemalloc</code></span></a> module (defined in <span id="index-35" class="target"></span><a href="https://peps.python.org/pep-0454/" class="pep reference external"><strong>PEP 454</strong></a>) is a debug tool to trace memory blocks allocated by Python. It provides the following information:

- Trace where an object was allocated

- Statistics on allocated memory blocks per filename and per line number: total size, number and average size of allocated memory blocks

- Compute the differences between two snapshots to detect memory leaks

<div class="admonition seealso">

See also

<span id="index-36" class="target"></span><a href="https://peps.python.org/pep-0454/" class="pep reference external"><strong>PEP 454</strong></a> – Add a new tracemalloc module to trace Python memory allocations  
PEP written and implemented by Victor Stinner

</div>

</div>

</div>

<div id="improved-modules" class="section">

## Improved Modules<a href="#improved-modules" class="headerlink" title="Link to this heading">¶</a>

<div id="abc" class="section">

### abc<a href="#abc" class="headerlink" title="Link to this heading">¶</a>

New function <a href="../library/abc.html#abc.get_cache_token" class="reference internal" title="abc.get_cache_token"><span class="pre"><code class="sourceCode python">abc.get_cache_token()</code></span></a> can be used to know when to invalidate caches that are affected by changes in the object graph. (Contributed by Łukasz Langa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16832" class="reference external">bpo-16832</a>.)

New class <a href="../library/abc.html#abc.ABC" class="reference internal" title="abc.ABC"><span class="pre"><code class="sourceCode python">ABC</code></span></a> has <a href="../library/abc.html#abc.ABCMeta" class="reference internal" title="abc.ABCMeta"><span class="pre"><code class="sourceCode python">ABCMeta</code></span></a> as its meta class. Using <span class="pre">`ABC`</span> as a base class has essentially the same effect as specifying <span class="pre">`metaclass=abc.ABCMeta`</span>, but is simpler to type and easier to read. (Contributed by Bruno Dupuis in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16049" class="reference external">bpo-16049</a>.)

</div>

<div id="aifc" class="section">

### aifc<a href="#aifc" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`getparams()`</span> method now returns a namedtuple rather than a plain tuple. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17818" class="reference external">bpo-17818</a>.)

<span class="pre">`aifc.open()`</span> now supports the context management protocol: when used in a <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> block, the <span class="pre">`close()`</span> method of the returned object will be called automatically at the end of the block. (Contributed by Serhiy Storchacha in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16486" class="reference external">bpo-16486</a>.)

The <span class="pre">`writeframesraw()`</span> and <span class="pre">`writeframes()`</span> methods now accept any <a href="../glossary.html#term-bytes-like-object" class="reference internal"><span class="xref std std-term">bytes-like object</span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8311" class="reference external">bpo-8311</a>.)

</div>

<div id="argparse" class="section">

### argparse<a href="#argparse" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/argparse.html#argparse.FileType" class="reference internal" title="argparse.FileType"><span class="pre"><code class="sourceCode python">FileType</code></span></a> class now accepts *encoding* and *errors* arguments, which are passed through to <a href="../library/functions.html#open" class="reference internal" title="open"><span class="pre"><code class="sourceCode python"><span class="bu">open</span>()</code></span></a>. (Contributed by Lucas Maystre in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11175" class="reference external">bpo-11175</a>.)

</div>

<div id="audioop" class="section">

### audioop<a href="#audioop" class="headerlink" title="Link to this heading">¶</a>

<span class="pre">`audioop`</span> now supports 24-bit samples. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12866" class="reference external">bpo-12866</a>.)

New <span class="pre">`byteswap()`</span> function converts big-endian samples to little-endian and vice versa. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19641" class="reference external">bpo-19641</a>.)

All <span class="pre">`audioop`</span> functions now accept any <a href="../glossary.html#term-bytes-like-object" class="reference internal"><span class="xref std std-term">bytes-like object</span></a>. Strings are not accepted: they didn’t work before, now they raise an error right away. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16685" class="reference external">bpo-16685</a>.)

</div>

<div id="base64" class="section">

### base64<a href="#base64" class="headerlink" title="Link to this heading">¶</a>

The encoding and decoding functions in <a href="../library/base64.html#module-base64" class="reference internal" title="base64: RFC 4648: Base16, Base32, Base64 Data Encodings; Base85 and Ascii85"><span class="pre"><code class="sourceCode python">base64</code></span></a> now accept any <a href="../glossary.html#term-bytes-like-object" class="reference internal"><span class="xref std std-term">bytes-like object</span></a> in cases where it previously required a <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> or <a href="../library/stdtypes.html#bytearray" class="reference internal" title="bytearray"><span class="pre"><code class="sourceCode python"><span class="bu">bytearray</span></code></span></a> instance. (Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17839" class="reference external">bpo-17839</a>.)

New functions <a href="../library/base64.html#base64.a85encode" class="reference internal" title="base64.a85encode"><span class="pre"><code class="sourceCode python">a85encode()</code></span></a>, <a href="../library/base64.html#base64.a85decode" class="reference internal" title="base64.a85decode"><span class="pre"><code class="sourceCode python">a85decode()</code></span></a>, <a href="../library/base64.html#base64.b85encode" class="reference internal" title="base64.b85encode"><span class="pre"><code class="sourceCode python">b85encode()</code></span></a>, and <a href="../library/base64.html#base64.b85decode" class="reference internal" title="base64.b85decode"><span class="pre"><code class="sourceCode python">b85decode()</code></span></a> provide the ability to encode and decode binary data from and to <span class="pre">`Ascii85`</span> and the git/mercurial <span class="pre">`Base85`</span> formats, respectively. The <span class="pre">`a85`</span> functions have options that can be used to make them compatible with the variants of the <span class="pre">`Ascii85`</span> encoding, including the Adobe variant. (Contributed by Martin Morrison, the Mercurial project, Serhiy Storchaka, and Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17618" class="reference external">bpo-17618</a>.)

</div>

<div id="collections" class="section">

### collections<a href="#collections" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/collections.html#collections.ChainMap.new_child" class="reference internal" title="collections.ChainMap.new_child"><span class="pre"><code class="sourceCode python">ChainMap.new_child()</code></span></a> method now accepts an *m* argument specifying the child map to add to the chain. This allows an existing mapping and/or a custom mapping type to be used for the child. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16613" class="reference external">bpo-16613</a>.)

</div>

<div id="colorsys" class="section">

### colorsys<a href="#colorsys" class="headerlink" title="Link to this heading">¶</a>

The number of digits in the coefficients for the RGB — YIQ conversions have been expanded so that they match the FCC NTSC versions. The change in results should be less than 1% and may better match results found elsewhere. (Contributed by Brian Landers and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14323" class="reference external">bpo-14323</a>.)

</div>

<div id="contextlib" class="section">

### contextlib<a href="#contextlib" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/contextlib.html#contextlib.suppress" class="reference internal" title="contextlib.suppress"><span class="pre"><code class="sourceCode python">contextlib.suppress</code></span></a> context manager helps to clarify the intent of code that deliberately suppresses exceptions from a single statement. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15806" class="reference external">bpo-15806</a> and Zero Piraeus in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19266" class="reference external">bpo-19266</a>.)

The new <a href="../library/contextlib.html#contextlib.redirect_stdout" class="reference internal" title="contextlib.redirect_stdout"><span class="pre"><code class="sourceCode python">contextlib.redirect_stdout()</code></span></a> context manager makes it easier for utility scripts to handle inflexible APIs that write their output to <a href="../library/sys.html#sys.stdout" class="reference internal" title="sys.stdout"><span class="pre"><code class="sourceCode python">sys.stdout</code></span></a> and don’t provide any options to redirect it. Using the context manager, the <a href="../library/sys.html#sys.stdout" class="reference internal" title="sys.stdout"><span class="pre"><code class="sourceCode python">sys.stdout</code></span></a> output can be redirected to any other stream or, in conjunction with <a href="../library/io.html#io.StringIO" class="reference internal" title="io.StringIO"><span class="pre"><code class="sourceCode python">io.StringIO</code></span></a>, to a string. The latter can be especially useful, for example, to capture output from a function that was written to implement a command line interface. It is recommended only for utility scripts because it affects the global state of <a href="../library/sys.html#sys.stdout" class="reference internal" title="sys.stdout"><span class="pre"><code class="sourceCode python">sys.stdout</code></span></a>. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15805" class="reference external">bpo-15805</a>.)

The <a href="../library/contextlib.html#module-contextlib" class="reference internal" title="contextlib: Utilities for with-statement contexts."><span class="pre"><code class="sourceCode python">contextlib</code></span></a> documentation has also been updated to include a <a href="../library/contextlib.html#single-use-reusable-and-reentrant-cms" class="reference internal"><span class="std std-ref">discussion</span></a> of the differences between single use, reusable and reentrant context managers.

</div>

<div id="dbm" class="section">

### dbm<a href="#dbm" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/dbm.html#dbm.open" class="reference internal" title="dbm.open"><span class="pre"><code class="sourceCode python">dbm.<span class="bu">open</span>()</code></span></a> objects now support the context management protocol. When used in a <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement, the <span class="pre">`close`</span> method of the database object will be called automatically at the end of the block. (Contributed by Claudiu Popa and Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19282" class="reference external">bpo-19282</a>.)

</div>

<div id="dis" class="section">

### dis<a href="#dis" class="headerlink" title="Link to this heading">¶</a>

Functions <a href="../library/dis.html#dis.show_code" class="reference internal" title="dis.show_code"><span class="pre"><code class="sourceCode python">show_code()</code></span></a>, <a href="../library/dis.html#dis.dis" class="reference internal" title="dis.dis"><span class="pre"><code class="sourceCode python">dis()</code></span></a>, <a href="../library/dis.html#dis.distb" class="reference internal" title="dis.distb"><span class="pre"><code class="sourceCode python">distb()</code></span></a>, and <a href="../library/dis.html#dis.disassemble" class="reference internal" title="dis.disassemble"><span class="pre"><code class="sourceCode python">disassemble()</code></span></a> now accept a keyword-only *file* argument that controls where they write their output.

The <a href="../library/dis.html#module-dis" class="reference internal" title="dis: Disassembler for Python bytecode."><span class="pre"><code class="sourceCode python">dis</code></span></a> module is now built around an <a href="../library/dis.html#dis.Instruction" class="reference internal" title="dis.Instruction"><span class="pre"><code class="sourceCode python">Instruction</code></span></a> class that provides object oriented access to the details of each individual bytecode operation.

A new method, <a href="../library/dis.html#dis.get_instructions" class="reference internal" title="dis.get_instructions"><span class="pre"><code class="sourceCode python">get_instructions()</code></span></a>, provides an iterator that emits the Instruction stream for a given piece of Python code. Thus it is now possible to write a program that inspects and manipulates a bytecode object in ways different from those provided by the <a href="../library/dis.html#module-dis" class="reference internal" title="dis: Disassembler for Python bytecode."><span class="pre"><code class="sourceCode python">dis</code></span></a> module itself. For example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> import dis
    >>> for instr in dis.get_instructions(lambda x: x + 1):
    ...     print(instr.opname)
    LOAD_FAST
    LOAD_CONST
    BINARY_ADD
    RETURN_VALUE

</div>

</div>

The various display tools in the <a href="../library/dis.html#module-dis" class="reference internal" title="dis: Disassembler for Python bytecode."><span class="pre"><code class="sourceCode python">dis</code></span></a> module have been rewritten to use these new components.

In addition, a new application-friendly class <a href="../library/dis.html#dis.Bytecode" class="reference internal" title="dis.Bytecode"><span class="pre"><code class="sourceCode python">Bytecode</code></span></a> provides an object-oriented API for inspecting bytecode in both in human-readable form and for iterating over instructions. The <a href="../library/dis.html#dis.Bytecode" class="reference internal" title="dis.Bytecode"><span class="pre"><code class="sourceCode python">Bytecode</code></span></a> constructor takes the same arguments that <a href="../library/dis.html#dis.get_instructions" class="reference internal" title="dis.get_instructions"><span class="pre"><code class="sourceCode python">get_instructions()</code></span></a> does (plus an optional *current_offset*), and the resulting object can be iterated to produce <a href="../library/dis.html#dis.Instruction" class="reference internal" title="dis.Instruction"><span class="pre"><code class="sourceCode python">Instruction</code></span></a> objects. But it also has a <a href="../library/dis.html#dis.Bytecode.dis" class="reference internal" title="dis.Bytecode.dis"><span class="pre"><code class="sourceCode python">dis</code></span></a> method, equivalent to calling <a href="../library/dis.html#dis.dis" class="reference internal" title="dis.dis"><span class="pre"><code class="sourceCode python">dis</code></span></a> on the constructor argument, but returned as a multi-line string:

<div class="highlight-python3 notranslate">

<div class="highlight">

    >>> bytecode = dis.Bytecode(lambda x: x + 1, current_offset=3)
    >>> for instr in bytecode:
    ...     print('{} ({})'.format(instr.opname, instr.opcode))
    LOAD_FAST (124)
    LOAD_CONST (100)
    BINARY_ADD (23)
    RETURN_VALUE (83)
    >>> bytecode.dis().splitlines()
    ['  1           0 LOAD_FAST                0 (x)',
     '      -->     3 LOAD_CONST               1 (1)',
     '              6 BINARY_ADD',
     '              7 RETURN_VALUE']

</div>

</div>

<a href="../library/dis.html#dis.Bytecode" class="reference internal" title="dis.Bytecode"><span class="pre"><code class="sourceCode python">Bytecode</code></span></a> also has a class method, <a href="../library/dis.html#dis.Bytecode.from_traceback" class="reference internal" title="dis.Bytecode.from_traceback"><span class="pre"><code class="sourceCode python">from_traceback()</code></span></a>, that provides the ability to manipulate a traceback (that is, <span class="pre">`print(Bytecode.from_traceback(tb).dis())`</span> is equivalent to <span class="pre">`distb(tb)`</span>).

(Contributed by Nick Coghlan, Ryan Kelly and Thomas Kluyver in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11816" class="reference external">bpo-11816</a> and Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17916" class="reference external">bpo-17916</a>.)

New function <a href="../library/dis.html#dis.stack_effect" class="reference internal" title="dis.stack_effect"><span class="pre"><code class="sourceCode python">stack_effect()</code></span></a> computes the effect on the Python stack of a given opcode and argument, information that is not otherwise available. (Contributed by Larry Hastings in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19722" class="reference external">bpo-19722</a>.)

</div>

<div id="doctest" class="section">

### doctest<a href="#doctest" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/doctest.html#doctest-options" class="reference internal"><span class="std std-ref">option flag</span></a>, <a href="../library/doctest.html#doctest.FAIL_FAST" class="reference internal" title="doctest.FAIL_FAST"><span class="pre"><code class="sourceCode python">FAIL_FAST</code></span></a>, halts test running as soon as the first failure is detected. (Contributed by R. David Murray and Daniel Urban in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16522" class="reference external">bpo-16522</a>.)

The <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> command line interface now uses <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a>, and has two new options, <span class="pre">`-o`</span> and <span class="pre">`-f`</span>. <span class="pre">`-o`</span> allows <a href="../library/doctest.html#doctest-options" class="reference internal"><span class="std std-ref">doctest options</span></a> to be specified on the command line, and <span class="pre">`-f`</span> is a shorthand for <span class="pre">`-o`</span>` `<span class="pre">`FAIL_FAST`</span> (to parallel the similar option supported by the <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> CLI). (Contributed by R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11390" class="reference external">bpo-11390</a>.)

<a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> will now find doctests in extension module <span class="pre">`__doc__`</span> strings. (Contributed by Zachary Ware in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3158" class="reference external">bpo-3158</a>.)

</div>

<div id="email" class="section">

### email<a href="#email" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/email.compat32-message.html#email.message.Message.as_string" class="reference internal" title="email.message.Message.as_string"><span class="pre"><code class="sourceCode python">as_string()</code></span></a> now accepts a *policy* argument to override the default policy of the message when generating a string representation of it. This means that <span class="pre">`as_string`</span> can now be used in more circumstances, instead of having to create and use a <a href="../library/email.generator.html#module-email.generator" class="reference internal" title="email.generator: Generate flat text email messages from a message structure."><span class="pre"><code class="sourceCode python">generator</code></span></a> in order to pass formatting parameters to its <span class="pre">`flatten`</span> method. (Contributed by R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18600" class="reference external">bpo-18600</a>.)

New method <a href="../library/email.compat32-message.html#email.message.Message.as_bytes" class="reference internal" title="email.message.Message.as_bytes"><span class="pre"><code class="sourceCode python">as_bytes()</code></span></a> added to produce a bytes representation of the message in a fashion similar to how <span class="pre">`as_string`</span> produces a string representation. It does not accept the *maxheaderlen* argument, but does accept the *unixfrom* and *policy* arguments. The <a href="../library/email.compat32-message.html#email.message.Message" class="reference internal" title="email.message.Message"><span class="pre"><code class="sourceCode python">Message</code></span></a> <a href="../library/email.compat32-message.html#email.message.Message.__bytes__" class="reference internal" title="email.message.Message.__bytes__"><span class="pre"><code class="sourceCode python"><span class="fu">__bytes__</span>()</code></span></a> method calls it, meaning that <span class="pre">`bytes(mymsg)`</span> will now produce the intuitive result: a bytes object containing the fully formatted message. (Contributed by R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18600" class="reference external">bpo-18600</a>.)

The <a href="../library/email.compat32-message.html#email.message.Message.set_param" class="reference internal" title="email.message.Message.set_param"><span class="pre"><code class="sourceCode python">Message.set_param()</code></span></a> message now accepts a *replace* keyword argument. When specified, the associated header will be updated without changing its location in the list of headers. For backward compatibility, the default is <span class="pre">`False`</span>. (Contributed by R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18891" class="reference external">bpo-18891</a>.)

A pair of new subclasses of <a href="../library/email.compat32-message.html#email.message.Message" class="reference internal" title="email.message.Message"><span class="pre"><code class="sourceCode python">Message</code></span></a> have been added (<a href="../library/email.message.html#email.message.EmailMessage" class="reference internal" title="email.message.EmailMessage"><span class="pre"><code class="sourceCode python">EmailMessage</code></span></a> and <a href="../library/email.message.html#email.message.MIMEPart" class="reference internal" title="email.message.MIMEPart"><span class="pre"><code class="sourceCode python">MIMEPart</code></span></a>), along with a new sub-module, <a href="../library/email.contentmanager.html#module-email.contentmanager" class="reference internal" title="email.contentmanager: Storing and Retrieving Content from MIME Parts"><span class="pre"><code class="sourceCode python">contentmanager</code></span></a> and a new <a href="../library/email.policy.html#module-email.policy" class="reference internal" title="email.policy: Controlling the parsing and generating of messages"><span class="pre"><code class="sourceCode python">policy</code></span></a> attribute <a href="../library/email.policy.html#email.policy.EmailPolicy.content_manager" class="reference internal" title="email.policy.EmailPolicy.content_manager"><span class="pre"><code class="sourceCode python">content_manager</code></span></a>. All documentation is currently in the new module, which is being added as part of email’s new <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional API</span></a>. These classes provide a number of new methods that make extracting content from and inserting content into email messages much easier. For details, see the <a href="../library/email.contentmanager.html#module-email.contentmanager" class="reference internal" title="email.contentmanager: Storing and Retrieving Content from MIME Parts"><span class="pre"><code class="sourceCode python">contentmanager</code></span></a> documentation and the <a href="../library/email.examples.html#email-examples" class="reference internal"><span class="std std-ref">email: Examples</span></a>. These API additions complete the bulk of the work that was planned as part of the email6 project. The currently provisional API is scheduled to become final in Python 3.5 (possibly with a few minor additions in the area of error handling). (Contributed by R. David Murray in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18891" class="reference external">bpo-18891</a>.)

</div>

<div id="filecmp" class="section">

### filecmp<a href="#filecmp" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/filecmp.html#filecmp.clear_cache" class="reference internal" title="filecmp.clear_cache"><span class="pre"><code class="sourceCode python">clear_cache()</code></span></a> function provides the ability to clear the <a href="../library/filecmp.html#module-filecmp" class="reference internal" title="filecmp: Compare files efficiently."><span class="pre"><code class="sourceCode python">filecmp</code></span></a> comparison cache, which uses <a href="../library/os.html#os.stat" class="reference internal" title="os.stat"><span class="pre"><code class="sourceCode python">os.stat()</code></span></a> information to determine if the file has changed since the last compare. This can be used, for example, if the file might have been changed and re-checked in less time than the resolution of a particular filesystem’s file modification time field. (Contributed by Mark Levitt in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18149" class="reference external">bpo-18149</a>.)

New module attribute <a href="../library/filecmp.html#filecmp.DEFAULT_IGNORES" class="reference internal" title="filecmp.DEFAULT_IGNORES"><span class="pre"><code class="sourceCode python">DEFAULT_IGNORES</code></span></a> provides the list of directories that are used as the default value for the *ignore* parameter of the <a href="../library/filecmp.html#filecmp.dircmp" class="reference internal" title="filecmp.dircmp"><span class="pre"><code class="sourceCode python">dircmp()</code></span></a> function. (Contributed by Eli Bendersky in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15442" class="reference external">bpo-15442</a>.)

</div>

<div id="functools" class="section">

### functools<a href="#functools" class="headerlink" title="Link to this heading">¶</a>

The new <a href="../library/functools.html#functools.partialmethod" class="reference internal" title="functools.partialmethod"><span class="pre"><code class="sourceCode python">partialmethod()</code></span></a> descriptor brings partial argument application to descriptors, just as <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">partial()</code></span></a> provides for normal callables. The new descriptor also makes it easier to get arbitrary callables (including <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">partial()</code></span></a> instances) to behave like normal instance methods when included in a class definition. (Contributed by Alon Horev and Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4331" class="reference external">bpo-4331</a>.)

The new <a href="../library/functools.html#functools.singledispatch" class="reference internal" title="functools.singledispatch"><span class="pre"><code class="sourceCode python">singledispatch()</code></span></a> decorator brings support for single-dispatch generic functions to the Python standard library. Where object oriented programming focuses on grouping multiple operations on a common set of data into a class, a generic function focuses on grouping multiple implementations of an operation that allows it to work with *different* kinds of data.

<div class="admonition seealso">

See also

<span id="index-37" class="target"></span><a href="https://peps.python.org/pep-0443/" class="pep reference external"><strong>PEP 443</strong></a> – Single-dispatch generic functions  
PEP written and implemented by Łukasz Langa.

</div>

<a href="../library/functools.html#functools.total_ordering" class="reference internal" title="functools.total_ordering"><span class="pre"><code class="sourceCode python">total_ordering()</code></span></a> now supports a return value of <a href="../library/constants.html#NotImplemented" class="reference internal" title="NotImplemented"><span class="pre"><code class="sourceCode python"><span class="va">NotImplemented</span></code></span></a> from the underlying comparison function. (Contributed by Katie Miller in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10042" class="reference external">bpo-10042</a>.)

A pure-python version of the <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">partial()</code></span></a> function is now in the stdlib; in CPython it is overridden by the C accelerated version, but it is available for other implementations to use. (Contributed by Brian Thorne in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12428" class="reference external">bpo-12428</a>.)

</div>

<div id="gc" class="section">

### gc<a href="#gc" class="headerlink" title="Link to this heading">¶</a>

New function <a href="../library/gc.html#gc.get_stats" class="reference internal" title="gc.get_stats"><span class="pre"><code class="sourceCode python">get_stats()</code></span></a> returns a list of three per-generation dictionaries containing the collections statistics since interpreter startup. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16351" class="reference external">bpo-16351</a>.)

</div>

<div id="glob" class="section">

### glob<a href="#glob" class="headerlink" title="Link to this heading">¶</a>

A new function <a href="../library/glob.html#glob.escape" class="reference internal" title="glob.escape"><span class="pre"><code class="sourceCode python">escape()</code></span></a> provides a way to escape special characters in a filename so that they do not become part of the globbing expansion but are instead matched literally. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8402" class="reference external">bpo-8402</a>.)

</div>

<div id="hashlib" class="section">

### hashlib<a href="#hashlib" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/hashlib.html#hashlib.pbkdf2_hmac" class="reference internal" title="hashlib.pbkdf2_hmac"><span class="pre"><code class="sourceCode python">hashlib.pbkdf2_hmac()</code></span></a> function provides the <a href="https://en.wikipedia.org/wiki/PBKDF2" class="reference external">PKCS#5 password-based key derivation function 2</a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18582" class="reference external">bpo-18582</a>.)

The <a href="../library/hashlib.html#hashlib.hash.name" class="reference internal" title="hashlib.hash.name"><span class="pre"><code class="sourceCode python">name</code></span></a> attribute of <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> hash objects is now a formally supported interface. It has always existed in CPython’s <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> (although it did not return lower case names for all supported hashes), but it was not a public interface and so some other Python implementations have not previously supported it. (Contributed by Jason R. Coombs in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18532" class="reference external">bpo-18532</a>.)

</div>

<div id="hmac" class="section">

### hmac<a href="#hmac" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/hmac.html#module-hmac" class="reference internal" title="hmac: Keyed-Hashing for Message Authentication (HMAC) implementation"><span class="pre"><code class="sourceCode python">hmac</code></span></a> now accepts <span class="pre">`bytearray`</span> as well as <span class="pre">`bytes`</span> for the *key* argument to the <a href="../library/hmac.html#hmac.new" class="reference internal" title="hmac.new"><span class="pre"><code class="sourceCode python">new()</code></span></a> function, and the *msg* parameter to both the <a href="../library/hmac.html#hmac.new" class="reference internal" title="hmac.new"><span class="pre"><code class="sourceCode python">new()</code></span></a> function and the <a href="../library/hmac.html#hmac.HMAC.update" class="reference internal" title="hmac.HMAC.update"><span class="pre"><code class="sourceCode python">update()</code></span></a> method now accepts any type supported by the <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a> module. (Contributed by Jonas Borgström in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18240" class="reference external">bpo-18240</a>.)

The *digestmod* argument to the <a href="../library/hmac.html#hmac.new" class="reference internal" title="hmac.new"><span class="pre"><code class="sourceCode python">hmac.new()</code></span></a> function may now be any hash digest name recognized by <a href="../library/hashlib.html#module-hashlib" class="reference internal" title="hashlib: Secure hash and message digest algorithms."><span class="pre"><code class="sourceCode python">hashlib</code></span></a>. In addition, the current behavior in which the value of *digestmod* defaults to <span class="pre">`MD5`</span> is deprecated: in a future version of Python there will be no default value. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17276" class="reference external">bpo-17276</a>.)

With the addition of <a href="../library/hmac.html#hmac.HMAC.block_size" class="reference internal" title="hmac.HMAC.block_size"><span class="pre"><code class="sourceCode python">block_size</code></span></a> and <a href="../library/hmac.html#hmac.HMAC.name" class="reference internal" title="hmac.HMAC.name"><span class="pre"><code class="sourceCode python">name</code></span></a> attributes (and the formal documentation of the <a href="../library/hmac.html#hmac.HMAC.digest_size" class="reference internal" title="hmac.HMAC.digest_size"><span class="pre"><code class="sourceCode python">digest_size</code></span></a> attribute), the <a href="../library/hmac.html#module-hmac" class="reference internal" title="hmac: Keyed-Hashing for Message Authentication (HMAC) implementation"><span class="pre"><code class="sourceCode python">hmac</code></span></a> module now conforms fully to the <span id="index-38" class="target"></span><a href="https://peps.python.org/pep-0247/" class="pep reference external"><strong>PEP 247</strong></a> API. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18775" class="reference external">bpo-18775</a>.)

</div>

<div id="html" class="section">

### html<a href="#html" class="headerlink" title="Link to this heading">¶</a>

New function <a href="../library/html.html#html.unescape" class="reference internal" title="html.unescape"><span class="pre"><code class="sourceCode python">unescape()</code></span></a> function converts HTML5 character references to the corresponding Unicode characters. (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2927" class="reference external">bpo-2927</a>.)

<a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">HTMLParser</code></span></a> accepts a new keyword argument *convert_charrefs* that, when <span class="pre">`True`</span>, automatically converts all character references. For backward-compatibility, its value defaults to <span class="pre">`False`</span>, but it will change to <span class="pre">`True`</span> in a future version of Python, so you are invited to set it explicitly and update your code to use this new feature. (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13633" class="reference external">bpo-13633</a>.)

The *strict* argument of <a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">HTMLParser</code></span></a> is now deprecated. (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15114" class="reference external">bpo-15114</a>.)

</div>

<div id="http" class="section">

### http<a href="#http" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/http.server.html#http.server.BaseHTTPRequestHandler.send_error" class="reference internal" title="http.server.BaseHTTPRequestHandler.send_error"><span class="pre"><code class="sourceCode python">send_error()</code></span></a> now accepts an optional additional *explain* parameter which can be used to provide an extended error description, overriding the hardcoded default if there is one. This extended error description will be formatted using the <a href="../library/http.server.html#http.server.BaseHTTPRequestHandler.error_message_format" class="reference internal" title="http.server.BaseHTTPRequestHandler.error_message_format"><span class="pre"><code class="sourceCode python">error_message_format</code></span></a> attribute and sent as the body of the error response. (Contributed by Karl Cow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=12921" class="reference external">bpo-12921</a>.)

The <a href="../library/http.server.html#module-http.server" class="reference internal" title="http.server: HTTP server and request handlers."><span class="pre"><code class="sourceCode python">http.server</code></span></a> <a href="../library/http.server.html#http-server-cli" class="reference internal"><span class="std std-ref">command line interface</span></a> now has a <span class="pre">`-b/--bind`</span> option that causes the server to listen on a specific address. (Contributed by Malte Swart in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17764" class="reference external">bpo-17764</a>.)

</div>

<div id="idlelib-and-idle" class="section">

### idlelib and IDLE<a href="#idlelib-and-idle" class="headerlink" title="Link to this heading">¶</a>

Since idlelib implements the IDLE shell and editor and is not intended for import by other programs, it gets improvements with every release. See <span class="pre">`Lib/idlelib/NEWS.txt`</span> for a cumulative list of changes since 3.3.0, as well as changes made in future 3.4.x releases. This file is also available from the IDLE <span class="menuselection">Help ‣ About IDLE</span> dialog.

</div>

<div id="importlib" class="section">

### importlib<a href="#importlib" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/importlib.html#importlib.abc.InspectLoader" class="reference internal" title="importlib.abc.InspectLoader"><span class="pre"><code class="sourceCode python">InspectLoader</code></span></a> ABC defines a new method, <a href="../library/importlib.html#importlib.abc.InspectLoader.source_to_code" class="reference internal" title="importlib.abc.InspectLoader.source_to_code"><span class="pre"><code class="sourceCode python">source_to_code()</code></span></a> that accepts source data and a path and returns a code object. The default implementation is equivalent to <span class="pre">`compile(data,`</span>` `<span class="pre">`path,`</span>` `<span class="pre">`'exec',`</span>` `<span class="pre">`dont_inherit=True)`</span>. (Contributed by Eric Snow and Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15627" class="reference external">bpo-15627</a>.)

<a href="../library/importlib.html#importlib.abc.InspectLoader" class="reference internal" title="importlib.abc.InspectLoader"><span class="pre"><code class="sourceCode python">InspectLoader</code></span></a> also now has a default implementation for the <a href="../library/importlib.html#importlib.abc.InspectLoader.get_code" class="reference internal" title="importlib.abc.InspectLoader.get_code"><span class="pre"><code class="sourceCode python">get_code()</code></span></a> method. However, it will normally be desirable to override the default implementation for performance reasons. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18072" class="reference external">bpo-18072</a>.)

The <a href="../library/importlib.html#importlib.reload" class="reference internal" title="importlib.reload"><span class="pre"><code class="sourceCode python"><span class="bu">reload</span>()</code></span></a> function has been moved from <span class="pre">`imp`</span> to <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> as part of the <span class="pre">`imp`</span> module deprecation. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18193" class="reference external">bpo-18193</a>.)

<a href="../library/importlib.html#module-importlib.util" class="reference internal" title="importlib.util: Utility code for importers"><span class="pre"><code class="sourceCode python">importlib.util</code></span></a> now has a <a href="../library/importlib.html#importlib.util.MAGIC_NUMBER" class="reference internal" title="importlib.util.MAGIC_NUMBER"><span class="pre"><code class="sourceCode python">MAGIC_NUMBER</code></span></a> attribute providing access to the bytecode version number. This replaces the <span class="pre">`get_magic()`</span> function in the deprecated <span class="pre">`imp`</span> module. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18192" class="reference external">bpo-18192</a>.)

New <a href="../library/importlib.html#module-importlib.util" class="reference internal" title="importlib.util: Utility code for importers"><span class="pre"><code class="sourceCode python">importlib.util</code></span></a> functions <a href="../library/importlib.html#importlib.util.cache_from_source" class="reference internal" title="importlib.util.cache_from_source"><span class="pre"><code class="sourceCode python">cache_from_source()</code></span></a> and <a href="../library/importlib.html#importlib.util.source_from_cache" class="reference internal" title="importlib.util.source_from_cache"><span class="pre"><code class="sourceCode python">source_from_cache()</code></span></a> replace the same-named functions in the deprecated <span class="pre">`imp`</span> module. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18194" class="reference external">bpo-18194</a>.)

The <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> bootstrap <a href="../library/importlib.html#importlib.machinery.NamespaceLoader" class="reference internal" title="importlib.machinery.NamespaceLoader"><span class="pre"><code class="sourceCode python">NamespaceLoader</code></span></a> now conforms to the <a href="../library/importlib.html#importlib.abc.InspectLoader" class="reference internal" title="importlib.abc.InspectLoader"><span class="pre"><code class="sourceCode python">InspectLoader</code></span></a> ABC, which means that <span class="pre">`runpy`</span> and <span class="pre">`python`</span>` `<span class="pre">`-m`</span> can now be used with namespace packages. (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18058" class="reference external">bpo-18058</a>.)

<a href="../library/importlib.html#module-importlib.util" class="reference internal" title="importlib.util: Utility code for importers"><span class="pre"><code class="sourceCode python">importlib.util</code></span></a> has a new function <a href="../library/importlib.html#importlib.util.decode_source" class="reference internal" title="importlib.util.decode_source"><span class="pre"><code class="sourceCode python">decode_source()</code></span></a> that decodes source from bytes using universal newline processing. This is useful for implementing <a href="../library/importlib.html#importlib.abc.InspectLoader.get_source" class="reference internal" title="importlib.abc.InspectLoader.get_source"><span class="pre"><code class="sourceCode python">InspectLoader.get_source()</code></span></a> methods.

<a href="../library/importlib.html#importlib.machinery.ExtensionFileLoader" class="reference internal" title="importlib.machinery.ExtensionFileLoader"><span class="pre"><code class="sourceCode python">importlib.machinery.ExtensionFileLoader</code></span></a> now has a <a href="../library/importlib.html#importlib.machinery.ExtensionFileLoader.get_filename" class="reference internal" title="importlib.machinery.ExtensionFileLoader.get_filename"><span class="pre"><code class="sourceCode python">get_filename()</code></span></a> method. This was inadvertently omitted in the original implementation. (Contributed by Eric Snow in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19152" class="reference external">bpo-19152</a>.)

</div>

<div id="inspect" class="section">

### inspect<a href="#inspect" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> module now offers a basic <a href="../library/inspect.html#inspect-module-cli" class="reference internal"><span class="std std-ref">command line interface</span></a> to quickly display source code and other information for modules, classes and functions. (Contributed by Claudiu Popa and Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18626" class="reference external">bpo-18626</a>.)

<a href="../library/inspect.html#inspect.unwrap" class="reference internal" title="inspect.unwrap"><span class="pre"><code class="sourceCode python">unwrap()</code></span></a> makes it easy to unravel wrapper function chains created by <a href="../library/functools.html#functools.wraps" class="reference internal" title="functools.wraps"><span class="pre"><code class="sourceCode python">functools.wraps()</code></span></a> (and any other API that sets the <span class="pre">`__wrapped__`</span> attribute on a wrapper function). (Contributed by Daniel Urban, Aaron Iles and Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13266" class="reference external">bpo-13266</a>.)

As part of the implementation of the new <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">enum</code></span></a> module, the <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> module now has substantially better support for custom <span class="pre">`__dir__`</span> methods and dynamic class attributes provided through metaclasses. (Contributed by Ethan Furman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18929" class="reference external">bpo-18929</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19030" class="reference external">bpo-19030</a>.)

<a href="../library/inspect.html#inspect.getfullargspec" class="reference internal" title="inspect.getfullargspec"><span class="pre"><code class="sourceCode python">getfullargspec()</code></span></a> and <span class="pre">`getargspec()`</span> now use the <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">signature()</code></span></a> API. This allows them to support a much broader range of callables, including those with <span class="pre">`__signature__`</span> attributes, those with metadata provided by argument clinic, <a href="../library/functools.html#functools.partial" class="reference internal" title="functools.partial"><span class="pre"><code class="sourceCode python">functools.partial()</code></span></a> objects and more. Note that, unlike <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">signature()</code></span></a>, these functions still ignore <span class="pre">`__wrapped__`</span> attributes, and report the already bound first argument for bound methods, so it is still necessary to update your code to use <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">signature()</code></span></a> directly if those features are desired. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17481" class="reference external">bpo-17481</a>.)

<a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">signature()</code></span></a> now supports duck types of CPython functions, which adds support for functions compiled with Cython. (Contributed by Stefan Behnel and Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17159" class="reference external">bpo-17159</a>.)

</div>

<div id="ipaddress" class="section">

### ipaddress<a href="#ipaddress" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> was added to the standard library in Python 3.3 as a <a href="../glossary.html#term-provisional-API" class="reference internal"><span class="xref std std-term">provisional API</span></a>. With the release of Python 3.4, this qualification has been removed: <a href="../library/ipaddress.html#module-ipaddress" class="reference internal" title="ipaddress: IPv4/IPv6 manipulation library."><span class="pre"><code class="sourceCode python">ipaddress</code></span></a> is now considered a stable API, covered by the normal standard library requirements to maintain backwards compatibility.

A new <a href="../library/ipaddress.html#ipaddress.IPv4Address.is_global" class="reference internal" title="ipaddress.IPv4Address.is_global"><span class="pre"><code class="sourceCode python">is_global</code></span></a> property is <span class="pre">`True`</span> if an address is globally routeable. (Contributed by Peter Moody in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17400" class="reference external">bpo-17400</a>.)

</div>

<div id="logging" class="section">

### logging<a href="#logging" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/logging.handlers.html#logging.handlers.TimedRotatingFileHandler" class="reference internal" title="logging.handlers.TimedRotatingFileHandler"><span class="pre"><code class="sourceCode python">TimedRotatingFileHandler</code></span></a> has a new *atTime* parameter that can be used to specify the time of day when rollover should happen. (Contributed by Ronald Oussoren in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9556" class="reference external">bpo-9556</a>.)

<a href="../library/logging.handlers.html#logging.handlers.SocketHandler" class="reference internal" title="logging.handlers.SocketHandler"><span class="pre"><code class="sourceCode python">SocketHandler</code></span></a> and <a href="../library/logging.handlers.html#logging.handlers.DatagramHandler" class="reference internal" title="logging.handlers.DatagramHandler"><span class="pre"><code class="sourceCode python">DatagramHandler</code></span></a> now support Unix domain sockets (by setting *port* to <span class="pre">`None`</span>). (Contributed by Vinay Sajip in commit ce46195b56a9.)

<a href="../library/logging.config.html#logging.config.fileConfig" class="reference internal" title="logging.config.fileConfig"><span class="pre"><code class="sourceCode python">fileConfig()</code></span></a> now accepts a <a href="../library/configparser.html#configparser.RawConfigParser" class="reference internal" title="configparser.RawConfigParser"><span class="pre"><code class="sourceCode python">configparser.RawConfigParser</code></span></a> subclass instance for the *fname* parameter. This facilitates using a configuration file when logging configuration is just a part of the overall application configuration, or where the application modifies the configuration before passing it to <a href="../library/logging.config.html#logging.config.fileConfig" class="reference internal" title="logging.config.fileConfig"><span class="pre"><code class="sourceCode python">fileConfig()</code></span></a>. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16110" class="reference external">bpo-16110</a>.)

Logging configuration data received from a socket via the <a href="../library/logging.config.html#logging.config.listen" class="reference internal" title="logging.config.listen"><span class="pre"><code class="sourceCode python">logging.config.listen()</code></span></a> function can now be validated before being processed by supplying a verification function as the argument to the new *verify* keyword argument. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15452" class="reference external">bpo-15452</a>.)

</div>

<div id="marshal" class="section">

<span id="whatsnew-marshal-3"></span>

### marshal<a href="#marshal" class="headerlink" title="Link to this heading">¶</a>

The default <a href="../library/marshal.html#module-marshal" class="reference internal" title="marshal: Convert Python objects to streams of bytes and back (with different constraints)."><span class="pre"><code class="sourceCode python">marshal</code></span></a> version has been bumped to 3. The code implementing the new version restores the Python2 behavior of recording only one copy of interned strings and preserving the interning on deserialization, and extends this “one copy” ability to any object type (including handling recursive references). This reduces both the size of <span class="pre">`.pyc`</span> files and the amount of memory a module occupies in memory when it is loaded from a <span class="pre">`.pyc`</span> (or <span class="pre">`.pyo`</span>) file. (Contributed by Kristján Valur Jónsson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16475" class="reference external">bpo-16475</a>, with additional speedups by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19219" class="reference external">bpo-19219</a>.)

</div>

<div id="mmap" class="section">

### mmap<a href="#mmap" class="headerlink" title="Link to this heading">¶</a>

mmap objects are now <a href="../library/weakref.html#mod-weakref" class="reference internal"><span class="std std-ref">weakly referenceable</span></a>. (Contributed by Valerie Lambert in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4885" class="reference external">bpo-4885</a>.)

</div>

<div id="multiprocessing" class="section">

### multiprocessing<a href="#multiprocessing" class="headerlink" title="Link to this heading">¶</a>

On Unix two new <a href="../library/multiprocessing.html#multiprocessing-start-methods" class="reference internal"><span class="std std-ref">start methods</span></a>, <span class="pre">`spawn`</span> and <span class="pre">`forkserver`</span>, have been added for starting processes using <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a>. These make the mixing of processes with threads more robust, and the <span class="pre">`spawn`</span> method matches the semantics that multiprocessing has always used on Windows. New function <a href="../library/multiprocessing.html#multiprocessing.get_all_start_methods" class="reference internal" title="multiprocessing.get_all_start_methods"><span class="pre"><code class="sourceCode python">get_all_start_methods()</code></span></a> reports all start methods available on the platform, <a href="../library/multiprocessing.html#multiprocessing.get_start_method" class="reference internal" title="multiprocessing.get_start_method"><span class="pre"><code class="sourceCode python">get_start_method()</code></span></a> reports the current start method, and <a href="../library/multiprocessing.html#multiprocessing.set_start_method" class="reference internal" title="multiprocessing.set_start_method"><span class="pre"><code class="sourceCode python">set_start_method()</code></span></a> sets the start method. (Contributed by Richard Oudkerk in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8713" class="reference external">bpo-8713</a>.)

<a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> also now has the concept of a <span class="pre">`context`</span>, which determines how child processes are created. New function <a href="../library/multiprocessing.html#multiprocessing.get_context" class="reference internal" title="multiprocessing.get_context"><span class="pre"><code class="sourceCode python">get_context()</code></span></a> returns a context that uses a specified start method. It has the same API as the <a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> module itself, so you can use it to create <a href="../library/multiprocessing.html#multiprocessing.pool.Pool" class="reference internal" title="multiprocessing.pool.Pool"><span class="pre"><code class="sourceCode python">Pool</code></span></a>s and other objects that will operate within that context. This allows a framework and an application or different parts of the same application to use multiprocessing without interfering with each other. (Contributed by Richard Oudkerk in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18999" class="reference external">bpo-18999</a>.)

Except when using the old *fork* start method, child processes no longer inherit unneeded handles/file descriptors from their parents (part of <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8713" class="reference external">bpo-8713</a>).

<a href="../library/multiprocessing.html#module-multiprocessing" class="reference internal" title="multiprocessing: Process-based parallelism."><span class="pre"><code class="sourceCode python">multiprocessing</code></span></a> now relies on <a href="../library/runpy.html#module-runpy" class="reference internal" title="runpy: Locate and run Python modules without importing them first."><span class="pre"><code class="sourceCode python">runpy</code></span></a> (which implements the <span class="pre">`-m`</span> switch) to initialise <span class="pre">`__main__`</span> appropriately in child processes when using the <span class="pre">`spawn`</span> or <span class="pre">`forkserver`</span> start methods. This resolves some edge cases where combining multiprocessing, the <span class="pre">`-m`</span> command line switch, and explicit relative imports could cause obscure failures in child processes. (Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19946" class="reference external">bpo-19946</a>.)

</div>

<div id="operator" class="section">

### operator<a href="#operator" class="headerlink" title="Link to this heading">¶</a>

New function <a href="../library/operator.html#operator.length_hint" class="reference internal" title="operator.length_hint"><span class="pre"><code class="sourceCode python">length_hint()</code></span></a> provides an implementation of the specification for how the <a href="../reference/datamodel.html#object.__length_hint__" class="reference internal" title="object.__length_hint__"><span class="pre"><code class="sourceCode python"><span class="fu">__length_hint__</span>()</code></span></a> special method should be used, as part of the <span id="index-39" class="target"></span><a href="https://peps.python.org/pep-0424/" class="pep reference external"><strong>PEP 424</strong></a> formal specification of this language feature. (Contributed by Armin Ronacher in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16148" class="reference external">bpo-16148</a>.)

There is now a pure-python version of the <a href="../library/operator.html#module-operator" class="reference internal" title="operator: Functions corresponding to the standard operators."><span class="pre"><code class="sourceCode python">operator</code></span></a> module available for reference and for use by alternate implementations of Python. (Contributed by Zachary Ware in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16694" class="reference external">bpo-16694</a>.)

</div>

<div id="os" class="section">

### os<a href="#os" class="headerlink" title="Link to this heading">¶</a>

There are new functions to get and set the <a href="../library/os.html#fd-inheritance" class="reference internal"><span class="std std-ref">inheritable flag</span></a> of a file descriptor (<a href="../library/os.html#os.get_inheritable" class="reference internal" title="os.get_inheritable"><span class="pre"><code class="sourceCode python">os.get_inheritable()</code></span></a>, <a href="../library/os.html#os.set_inheritable" class="reference internal" title="os.set_inheritable"><span class="pre"><code class="sourceCode python">os.set_inheritable()</code></span></a>) or a Windows handle (<a href="../library/os.html#os.get_handle_inheritable" class="reference internal" title="os.get_handle_inheritable"><span class="pre"><code class="sourceCode python">os.get_handle_inheritable()</code></span></a>, <a href="../library/os.html#os.set_handle_inheritable" class="reference internal" title="os.set_handle_inheritable"><span class="pre"><code class="sourceCode python">os.set_handle_inheritable()</code></span></a>).

New function <a href="../library/os.html#os.cpu_count" class="reference internal" title="os.cpu_count"><span class="pre"><code class="sourceCode python">cpu_count()</code></span></a> reports the number of CPUs available on the platform on which Python is running (or <span class="pre">`None`</span> if the count can’t be determined). The <a href="../library/multiprocessing.html#multiprocessing.cpu_count" class="reference internal" title="multiprocessing.cpu_count"><span class="pre"><code class="sourceCode python">multiprocessing.cpu_count()</code></span></a> function is now implemented in terms of this function). (Contributed by Trent Nelson, Yogesh Chaudhari, Victor Stinner, and Charles-François Natali in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17914" class="reference external">bpo-17914</a>.)

<a href="../library/os.path.html#os.path.samestat" class="reference internal" title="os.path.samestat"><span class="pre"><code class="sourceCode python">os.path.samestat()</code></span></a> is now available on the Windows platform (and the <a href="../library/os.path.html#os.path.samefile" class="reference internal" title="os.path.samefile"><span class="pre"><code class="sourceCode python">os.path.samefile()</code></span></a> implementation is now shared between Unix and Windows). (Contributed by Brian Curtin in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11939" class="reference external">bpo-11939</a>.)

<a href="../library/os.path.html#os.path.ismount" class="reference internal" title="os.path.ismount"><span class="pre"><code class="sourceCode python">os.path.ismount()</code></span></a> now recognizes volumes mounted below a drive root on Windows. (Contributed by Tim Golden in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9035" class="reference external">bpo-9035</a>.)

<a href="../library/os.html#os.open" class="reference internal" title="os.open"><span class="pre"><code class="sourceCode python">os.<span class="bu">open</span>()</code></span></a> supports two new flags on platforms that provide them, <a href="../library/os.html#os.O_PATH" class="reference internal" title="os.O_PATH"><span class="pre"><code class="sourceCode python">O_PATH</code></span></a> (un-opened file descriptor), and <a href="../library/os.html#os.O_TMPFILE" class="reference internal" title="os.O_TMPFILE"><span class="pre"><code class="sourceCode python">O_TMPFILE</code></span></a> (unnamed temporary file; as of 3.4.0 release available only on Linux systems with a kernel version of 3.11 or newer that have uapi headers). (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18673" class="reference external">bpo-18673</a> and Benjamin Peterson, respectively.)

</div>

<div id="pdb" class="section">

### pdb<a href="#pdb" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a> has been enhanced to handle generators, <a href="../reference/simple_stmts.html#yield" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">yield</code></span></a>, and <span class="pre">`yield`</span>` `<span class="pre">`from`</span> in a more useful fashion. This is especially helpful when debugging <a href="../library/asyncio.html#module-asyncio" class="reference internal" title="asyncio: Asynchronous I/O."><span class="pre"><code class="sourceCode python">asyncio</code></span></a> based programs. (Contributed by Andrew Svetlov and Xavier de Gaye in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16596" class="reference external">bpo-16596</a>.)

The <span class="pre">`print`</span> command has been removed from <a href="../library/pdb.html#module-pdb" class="reference internal" title="pdb: The Python debugger for interactive interpreters."><span class="pre"><code class="sourceCode python">pdb</code></span></a>, restoring access to the Python <a href="../library/functions.html#print" class="reference internal" title="print"><span class="pre"><code class="sourceCode python"><span class="bu">print</span>()</code></span></a> function from the pdb command line. Python2’s <span class="pre">`pdb`</span> did not have a <span class="pre">`print`</span> command; instead, entering <span class="pre">`print`</span> executed the <span class="pre">`print`</span> statement. In Python3 <span class="pre">`print`</span> was mistakenly made an alias for the pdb <a href="../library/pdb.html#pdbcommand-p" class="reference internal"><span class="pre"><code class="xref std std-pdbcmd docutils literal notranslate">p</code></span></a> command. <span class="pre">`p`</span>, however, prints the <span class="pre">`repr`</span> of its argument, not the <span class="pre">`str`</span> like the Python2 <span class="pre">`print`</span> command did. Worse, the Python3 <span class="pre">`pdb`</span>` `<span class="pre">`print`</span> command shadowed the Python3 <span class="pre">`print`</span> function, making it inaccessible at the <span class="pre">`pdb`</span> prompt. (Contributed by Connor Osborn in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18764" class="reference external">bpo-18764</a>.)

</div>

<div id="pickle" class="section">

<span id="whatsnew-protocol-4"></span>

### pickle<a href="#pickle" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pickle.html#module-pickle" class="reference internal" title="pickle: Convert Python objects to streams of bytes and back."><span class="pre"><code class="sourceCode python">pickle</code></span></a> now supports (but does not use by default) a new pickle protocol, protocol 4. This new protocol addresses a number of issues that were present in previous protocols, such as the serialization of nested classes, very large strings and containers, and classes whose <a href="../reference/datamodel.html#object.__new__" class="reference internal" title="object.__new__"><span class="pre"><code class="sourceCode python"><span class="fu">__new__</span>()</code></span></a> method takes keyword-only arguments. It also provides some efficiency improvements.

<div class="admonition seealso">

See also

<span id="index-40" class="target"></span><a href="https://peps.python.org/pep-3154/" class="pep reference external"><strong>PEP 3154</strong></a> – Pickle protocol 4  
PEP written by Antoine Pitrou and implemented by Alexandre Vassalotti.

</div>

</div>

<div id="plistlib" class="section">

### plistlib<a href="#plistlib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/plistlib.html#module-plistlib" class="reference internal" title="plistlib: Generate and parse Apple plist files."><span class="pre"><code class="sourceCode python">plistlib</code></span></a> now has an API that is similar to the standard pattern for stdlib serialization protocols, with new <a href="../library/plistlib.html#plistlib.load" class="reference internal" title="plistlib.load"><span class="pre"><code class="sourceCode python">load()</code></span></a>, <a href="../library/plistlib.html#plistlib.dump" class="reference internal" title="plistlib.dump"><span class="pre"><code class="sourceCode python">dump()</code></span></a>, <a href="../library/plistlib.html#plistlib.loads" class="reference internal" title="plistlib.loads"><span class="pre"><code class="sourceCode python">loads()</code></span></a>, and <a href="../library/plistlib.html#plistlib.dumps" class="reference internal" title="plistlib.dumps"><span class="pre"><code class="sourceCode python">dumps()</code></span></a> functions. (The older API is now deprecated.) In addition to the already supported XML plist format (<a href="../library/plistlib.html#plistlib.FMT_XML" class="reference internal" title="plistlib.FMT_XML"><span class="pre"><code class="sourceCode python">FMT_XML</code></span></a>), it also now supports the binary plist format (<a href="../library/plistlib.html#plistlib.FMT_BINARY" class="reference internal" title="plistlib.FMT_BINARY"><span class="pre"><code class="sourceCode python">FMT_BINARY</code></span></a>). (Contributed by Ronald Oussoren and others in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14455" class="reference external">bpo-14455</a>.)

</div>

<div id="poplib" class="section">

### poplib<a href="#poplib" class="headerlink" title="Link to this heading">¶</a>

Two new methods have been added to <a href="../library/poplib.html#module-poplib" class="reference internal" title="poplib: POP3 protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">poplib</code></span></a>: <a href="../library/poplib.html#poplib.POP3.capa" class="reference internal" title="poplib.POP3.capa"><span class="pre"><code class="sourceCode python">capa()</code></span></a>, which returns the list of capabilities advertised by the POP server, and <a href="../library/poplib.html#poplib.POP3.stls" class="reference internal" title="poplib.POP3.stls"><span class="pre"><code class="sourceCode python">stls()</code></span></a>, which switches a clear-text POP3 session into an encrypted POP3 session if the POP server supports it. (Contributed by Lorenzo Catucci in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=4473" class="reference external">bpo-4473</a>.)

</div>

<div id="pprint" class="section">

### pprint<a href="#pprint" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/pprint.html#module-pprint" class="reference internal" title="pprint: Data pretty printer."><span class="pre"><code class="sourceCode python">pprint</code></span></a> module’s <a href="../library/pprint.html#pprint.PrettyPrinter" class="reference internal" title="pprint.PrettyPrinter"><span class="pre"><code class="sourceCode python">PrettyPrinter</code></span></a> class and its <a href="../library/pprint.html#pprint.pformat" class="reference internal" title="pprint.pformat"><span class="pre"><code class="sourceCode python">pformat()</code></span></a>, and <a href="../library/pprint.html#pprint.pprint" class="reference internal" title="pprint.pprint"><span class="pre"><code class="sourceCode python">pprint()</code></span></a> functions have a new option, *compact*, that controls how the output is formatted. Currently setting *compact* to <span class="pre">`True`</span> means that sequences will be printed with as many sequence elements as will fit within *width* on each (indented) line. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19132" class="reference external">bpo-19132</a>.)

Long strings are now wrapped using Python’s normal line continuation syntax. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17150" class="reference external">bpo-17150</a>.)

</div>

<div id="pty" class="section">

### pty<a href="#pty" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/pty.html#pty.spawn" class="reference internal" title="pty.spawn"><span class="pre"><code class="sourceCode python">pty.spawn()</code></span></a> now returns the status value from <a href="../library/os.html#os.waitpid" class="reference internal" title="os.waitpid"><span class="pre"><code class="sourceCode python">os.waitpid()</code></span></a> on the child process, instead of <span class="pre">`None`</span>. (Contributed by Gregory P. Smith.)

</div>

<div id="pydoc" class="section">

### pydoc<a href="#pydoc" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module is now based directly on the <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a> introspection API, allowing it to provide signature information for a wider variety of callable objects. This change also means that <span class="pre">`__wrapped__`</span> attributes are now taken into account when displaying help information. (Contributed by Larry Hastings in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19674" class="reference external">bpo-19674</a>.)

The <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module no longer displays the <span class="pre">`self`</span> parameter for already bound methods. Instead, it aims to always display the exact current signature of the supplied callable. (Contributed by Larry Hastings in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20710" class="reference external">bpo-20710</a>.)

In addition to the changes that have been made to <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> directly, its handling of custom <span class="pre">`__dir__`</span> methods and various descriptor behaviours has also been improved substantially by the underlying changes in the <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> module.

As the <a href="../library/functions.html#help" class="reference internal" title="help"><span class="pre"><code class="sourceCode python"><span class="bu">help</span>()</code></span></a> builtin is based on <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a>, the above changes also affect the behaviour of <a href="../library/functions.html#help" class="reference internal" title="help"><span class="pre"><code class="sourceCode python"><span class="bu">help</span>()</code></span></a>.

</div>

<div id="re" class="section">

### re<a href="#re" class="headerlink" title="Link to this heading">¶</a>

New <a href="../library/re.html#re.fullmatch" class="reference internal" title="re.fullmatch"><span class="pre"><code class="sourceCode python">fullmatch()</code></span></a> function and <a href="../library/re.html#re.Pattern.fullmatch" class="reference internal" title="re.Pattern.fullmatch"><span class="pre"><code class="sourceCode python">Pattern.fullmatch()</code></span></a> method anchor the pattern at both ends of the string to match. This provides a way to be explicit about the goal of the match, which avoids a class of subtle bugs where <span class="pre">`$`</span> characters get lost during code changes or the addition of alternatives to an existing regular expression. (Contributed by Matthew Barnett in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16203" class="reference external">bpo-16203</a>.)

The repr of <a href="../library/re.html#re-objects" class="reference internal"><span class="std std-ref">regex objects</span></a> now includes the pattern and the flags; the repr of <a href="../library/re.html#match-objects" class="reference internal"><span class="std std-ref">match objects</span></a> now includes the start, end, and the part of the string that matched. (Contributed by Hugo Lopes Tavares and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13592" class="reference external">bpo-13592</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17087" class="reference external">bpo-17087</a>.)

</div>

<div id="resource" class="section">

### resource<a href="#resource" class="headerlink" title="Link to this heading">¶</a>

New <a href="../library/resource.html#resource.prlimit" class="reference internal" title="resource.prlimit"><span class="pre"><code class="sourceCode python">prlimit()</code></span></a> function, available on Linux platforms with a kernel version of 2.6.36 or later and glibc of 2.13 or later, provides the ability to query or set the resource limits for processes other than the one making the call. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16595" class="reference external">bpo-16595</a>.)

On Linux kernel version 2.6.36 or later, there are also some new Linux specific constants: <a href="../library/resource.html#resource.RLIMIT_MSGQUEUE" class="reference internal" title="resource.RLIMIT_MSGQUEUE"><span class="pre"><code class="sourceCode python">RLIMIT_MSGQUEUE</code></span></a>, <a href="../library/resource.html#resource.RLIMIT_NICE" class="reference internal" title="resource.RLIMIT_NICE"><span class="pre"><code class="sourceCode python">RLIMIT_NICE</code></span></a>, <a href="../library/resource.html#resource.RLIMIT_RTPRIO" class="reference internal" title="resource.RLIMIT_RTPRIO"><span class="pre"><code class="sourceCode python">RLIMIT_RTPRIO</code></span></a>, <a href="../library/resource.html#resource.RLIMIT_RTTIME" class="reference internal" title="resource.RLIMIT_RTTIME"><span class="pre"><code class="sourceCode python">RLIMIT_RTTIME</code></span></a>, and <a href="../library/resource.html#resource.RLIMIT_SIGPENDING" class="reference internal" title="resource.RLIMIT_SIGPENDING"><span class="pre"><code class="sourceCode python">RLIMIT_SIGPENDING</code></span></a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19324" class="reference external">bpo-19324</a>.)

On FreeBSD version 9 and later, there some new FreeBSD specific constants: <a href="../library/resource.html#resource.RLIMIT_SBSIZE" class="reference internal" title="resource.RLIMIT_SBSIZE"><span class="pre"><code class="sourceCode python">RLIMIT_SBSIZE</code></span></a>, <a href="../library/resource.html#resource.RLIMIT_SWAP" class="reference internal" title="resource.RLIMIT_SWAP"><span class="pre"><code class="sourceCode python">RLIMIT_SWAP</code></span></a>, and <a href="../library/resource.html#resource.RLIMIT_NPTS" class="reference internal" title="resource.RLIMIT_NPTS"><span class="pre"><code class="sourceCode python">RLIMIT_NPTS</code></span></a>. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19343" class="reference external">bpo-19343</a>.)

</div>

<div id="select" class="section">

### select<a href="#select" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/select.html#select.epoll" class="reference internal" title="select.epoll"><span class="pre"><code class="sourceCode python">epoll</code></span></a> objects now support the context management protocol. When used in a <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statement, the <a href="../library/select.html#select.epoll.close" class="reference internal" title="select.epoll.close"><span class="pre"><code class="sourceCode python">close()</code></span></a> method will be called automatically at the end of the block. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16488" class="reference external">bpo-16488</a>.)

<a href="../library/select.html#select.devpoll" class="reference internal" title="select.devpoll"><span class="pre"><code class="sourceCode python">devpoll</code></span></a> objects now have <a href="../library/select.html#select.devpoll.fileno" class="reference internal" title="select.devpoll.fileno"><span class="pre"><code class="sourceCode python">fileno()</code></span></a> and <a href="../library/select.html#select.devpoll.close" class="reference internal" title="select.devpoll.close"><span class="pre"><code class="sourceCode python">close()</code></span></a> methods, as well as a new attribute <a href="../library/select.html#select.devpoll.closed" class="reference internal" title="select.devpoll.closed"><span class="pre"><code class="sourceCode python">closed</code></span></a>. (Contributed by Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18794" class="reference external">bpo-18794</a>.)

</div>

<div id="shelve" class="section">

### shelve<a href="#shelve" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/shelve.html#shelve.Shelf" class="reference internal" title="shelve.Shelf"><span class="pre"><code class="sourceCode python">Shelf</code></span></a> instances may now be used in <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> statements, and will be automatically closed at the end of the <span class="pre">`with`</span> block. (Contributed by Filip Gruszczyński in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13896" class="reference external">bpo-13896</a>.)

</div>

<div id="shutil" class="section">

### shutil<a href="#shutil" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/shutil.html#shutil.copyfile" class="reference internal" title="shutil.copyfile"><span class="pre"><code class="sourceCode python">copyfile()</code></span></a> now raises a specific <a href="../library/shutil.html#shutil.Error" class="reference internal" title="shutil.Error"><span class="pre"><code class="sourceCode python">Error</code></span></a> subclass, <a href="../library/shutil.html#shutil.SameFileError" class="reference internal" title="shutil.SameFileError"><span class="pre"><code class="sourceCode python">SameFileError</code></span></a>, when the source and destination are the same file, which allows an application to take appropriate action on this specific error. (Contributed by Atsuo Ishimoto and Hynek Schlawack in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1492704" class="reference external">bpo-1492704</a>.)

</div>

<div id="smtpd" class="section">

### smtpd<a href="#smtpd" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`SMTPServer`</span> and <span class="pre">`SMTPChannel`</span> classes now accept a *map* keyword argument which, if specified, is passed in to <span class="pre">`asynchat.async_chat`</span> as its *map* argument. This allows an application to avoid affecting the global socket map. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11959" class="reference external">bpo-11959</a>.)

</div>

<div id="smtplib" class="section">

### smtplib<a href="#smtplib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/smtplib.html#smtplib.SMTPException" class="reference internal" title="smtplib.SMTPException"><span class="pre"><code class="sourceCode python">SMTPException</code></span></a> is now a subclass of <a href="../library/exceptions.html#OSError" class="reference internal" title="OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a>, which allows both socket level errors and SMTP protocol level errors to be caught in one try/except statement by code that only cares whether or not an error occurred. (Contributed by Ned Jackson Lovely in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=2118" class="reference external">bpo-2118</a>.)

</div>

<div id="socket" class="section">

### socket<a href="#socket" class="headerlink" title="Link to this heading">¶</a>

The socket module now supports the <a href="../library/socket.html#socket.CAN_BCM" class="reference internal" title="socket.CAN_BCM"><span class="pre"><code class="sourceCode python">CAN_BCM</code></span></a> protocol on platforms that support it. (Contributed by Brian Thorne in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15359" class="reference external">bpo-15359</a>.)

Socket objects have new methods to get or set their <a href="../library/os.html#fd-inheritance" class="reference internal"><span class="std std-ref">inheritable flag</span></a>, <a href="../library/socket.html#socket.socket.get_inheritable" class="reference internal" title="socket.socket.get_inheritable"><span class="pre"><code class="sourceCode python">get_inheritable()</code></span></a> and <a href="../library/socket.html#socket.socket.set_inheritable" class="reference internal" title="socket.socket.set_inheritable"><span class="pre"><code class="sourceCode python">set_inheritable()</code></span></a>.

The <span class="pre">`socket.AF_*`</span> and <span class="pre">`socket.SOCK_*`</span> constants are now enumeration values using the new <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">enum</code></span></a> module. This allows meaningful names to be printed during debugging, instead of integer “magic numbers”.

The <a href="../library/socket.html#socket.AF_LINK" class="reference internal" title="socket.AF_LINK"><span class="pre"><code class="sourceCode python">AF_LINK</code></span></a> constant is now available on BSD and OSX.

<a href="../library/socket.html#socket.inet_pton" class="reference internal" title="socket.inet_pton"><span class="pre"><code class="sourceCode python">inet_pton()</code></span></a> and <a href="../library/socket.html#socket.inet_ntop" class="reference internal" title="socket.inet_ntop"><span class="pre"><code class="sourceCode python">inet_ntop()</code></span></a> are now supported on Windows. (Contributed by Atsuo Ishimoto in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7171" class="reference external">bpo-7171</a>.)

</div>

<div id="sqlite3" class="section">

### sqlite3<a href="#sqlite3" class="headerlink" title="Link to this heading">¶</a>

A new boolean parameter to the <a href="../library/sqlite3.html#sqlite3.connect" class="reference internal" title="sqlite3.connect"><span class="pre"><code class="sourceCode python"><span class="ex">connect</span>()</code></span></a> function, *uri*, can be used to indicate that the *database* parameter is a <span class="pre">`uri`</span> (see the <a href="https://www.sqlite.org/uri.html" class="reference external">SQLite URI documentation</a>). (Contributed by poq in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13773" class="reference external">bpo-13773</a>.)

</div>

<div id="ssl" class="section">

### ssl<a href="#ssl" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/ssl.html#ssl.PROTOCOL_TLSv1_1" class="reference internal" title="ssl.PROTOCOL_TLSv1_1"><span class="pre"><code class="sourceCode python">PROTOCOL_TLSv1_1</code></span></a> and <a href="../library/ssl.html#ssl.PROTOCOL_TLSv1_2" class="reference internal" title="ssl.PROTOCOL_TLSv1_2"><span class="pre"><code class="sourceCode python">PROTOCOL_TLSv1_2</code></span></a> (TLSv1.1 and TLSv1.2 support) have been added; support for these protocols is only available if Python is linked with OpenSSL 1.0.1 or later. (Contributed by Michele Orrù and Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16692" class="reference external">bpo-16692</a>.)

New function <a href="../library/ssl.html#ssl.create_default_context" class="reference internal" title="ssl.create_default_context"><span class="pre"><code class="sourceCode python">create_default_context()</code></span></a> provides a standard way to obtain an <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> whose settings are intended to be a reasonable balance between compatibility and security. These settings are more stringent than the defaults provided by the <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> constructor, and may be adjusted in the future, without prior deprecation, if best-practice security requirements change. The new recommended best practice for using stdlib libraries that support SSL is to use <a href="../library/ssl.html#ssl.create_default_context" class="reference internal" title="ssl.create_default_context"><span class="pre"><code class="sourceCode python">create_default_context()</code></span></a> to obtain an <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> object, modify it if needed, and then pass it as the *context* argument of the appropriate stdlib API. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19689" class="reference external">bpo-19689</a>.)

<a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> method <a href="../library/ssl.html#ssl.SSLContext.load_verify_locations" class="reference internal" title="ssl.SSLContext.load_verify_locations"><span class="pre"><code class="sourceCode python">load_verify_locations()</code></span></a> accepts a new optional argument *cadata*, which can be used to provide PEM or DER encoded certificates directly via strings or bytes, respectively. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18138" class="reference external">bpo-18138</a>.)

New function <a href="../library/ssl.html#ssl.get_default_verify_paths" class="reference internal" title="ssl.get_default_verify_paths"><span class="pre"><code class="sourceCode python">get_default_verify_paths()</code></span></a> returns a named tuple of the paths and environment variables that the <a href="../library/ssl.html#ssl.SSLContext.set_default_verify_paths" class="reference internal" title="ssl.SSLContext.set_default_verify_paths"><span class="pre"><code class="sourceCode python">set_default_verify_paths()</code></span></a> method uses to set OpenSSL’s default <span class="pre">`cafile`</span> and <span class="pre">`capath`</span>. This can be an aid in debugging default verification issues. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18143" class="reference external">bpo-18143</a>.)

<a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> has a new method, <a href="../library/ssl.html#ssl.SSLContext.cert_store_stats" class="reference internal" title="ssl.SSLContext.cert_store_stats"><span class="pre"><code class="sourceCode python">cert_store_stats()</code></span></a>, that reports the number of loaded <span class="pre">`X.509`</span> certs, <span class="pre">`X.509`</span>` `<span class="pre">`CA`</span> certs, and certificate revocation lists (<span class="pre">`crl`</span>s), as well as a <a href="../library/ssl.html#ssl.SSLContext.get_ca_certs" class="reference internal" title="ssl.SSLContext.get_ca_certs"><span class="pre"><code class="sourceCode python">get_ca_certs()</code></span></a> method that returns a list of the loaded <span class="pre">`CA`</span> certificates. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18147" class="reference external">bpo-18147</a>.)

If OpenSSL 0.9.8 or later is available, <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> has a new attribute <a href="../library/ssl.html#ssl.SSLContext.verify_flags" class="reference internal" title="ssl.SSLContext.verify_flags"><span class="pre"><code class="sourceCode python">verify_flags</code></span></a> that can be used to control the certificate verification process by setting it to some combination of the new constants <a href="../library/ssl.html#ssl.VERIFY_DEFAULT" class="reference internal" title="ssl.VERIFY_DEFAULT"><span class="pre"><code class="sourceCode python">VERIFY_DEFAULT</code></span></a>, <a href="../library/ssl.html#ssl.VERIFY_CRL_CHECK_LEAF" class="reference internal" title="ssl.VERIFY_CRL_CHECK_LEAF"><span class="pre"><code class="sourceCode python">VERIFY_CRL_CHECK_LEAF</code></span></a>, <a href="../library/ssl.html#ssl.VERIFY_CRL_CHECK_CHAIN" class="reference internal" title="ssl.VERIFY_CRL_CHECK_CHAIN"><span class="pre"><code class="sourceCode python">VERIFY_CRL_CHECK_CHAIN</code></span></a>, or <a href="../library/ssl.html#ssl.VERIFY_X509_STRICT" class="reference internal" title="ssl.VERIFY_X509_STRICT"><span class="pre"><code class="sourceCode python">VERIFY_X509_STRICT</code></span></a>. OpenSSL does not do any CRL verification by default. (Contributed by Christien Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8813" class="reference external">bpo-8813</a>.)

New <a href="../library/ssl.html#ssl.SSLContext" class="reference internal" title="ssl.SSLContext"><span class="pre"><code class="sourceCode python">SSLContext</code></span></a> method <a href="../library/ssl.html#ssl.SSLContext.load_default_certs" class="reference internal" title="ssl.SSLContext.load_default_certs"><span class="pre"><code class="sourceCode python">load_default_certs()</code></span></a> loads a set of default “certificate authority” (CA) certificates from default locations, which vary according to the platform. It can be used to load both TLS web server authentication certificates (<span class="pre">`purpose=`</span><a href="../library/ssl.html#ssl.Purpose.SERVER_AUTH" class="reference internal" title="ssl.Purpose.SERVER_AUTH"><span class="pre"><code class="sourceCode python">SERVER_AUTH</code></span></a>) for a client to use to verify a server, and certificates for a server to use in verifying client certificates (<span class="pre">`purpose=`</span><a href="../library/ssl.html#ssl.Purpose.CLIENT_AUTH" class="reference internal" title="ssl.Purpose.CLIENT_AUTH"><span class="pre"><code class="sourceCode python">CLIENT_AUTH</code></span></a>). (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19292" class="reference external">bpo-19292</a>.)

Two new windows-only functions, <a href="../library/ssl.html#ssl.enum_certificates" class="reference internal" title="ssl.enum_certificates"><span class="pre"><code class="sourceCode python">enum_certificates()</code></span></a> and <a href="../library/ssl.html#ssl.enum_crls" class="reference internal" title="ssl.enum_crls"><span class="pre"><code class="sourceCode python">enum_crls()</code></span></a> provide the ability to retrieve certificates, certificate information, and CRLs from the Windows cert store. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17134" class="reference external">bpo-17134</a>.)

Support for server-side SNI (Server Name Indication) using the new <a href="../library/ssl.html#ssl.SSLContext.set_servername_callback" class="reference internal" title="ssl.SSLContext.set_servername_callback"><span class="pre"><code class="sourceCode python">ssl.SSLContext.set_servername_callback()</code></span></a> method. (Contributed by Daniel Black in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8109" class="reference external">bpo-8109</a>.)

The dictionary returned by <a href="../library/ssl.html#ssl.SSLSocket.getpeercert" class="reference internal" title="ssl.SSLSocket.getpeercert"><span class="pre"><code class="sourceCode python">SSLSocket.getpeercert()</code></span></a> contains additional <span class="pre">`X509v3`</span> extension items: <span class="pre">`crlDistributionPoints`</span>, <span class="pre">`calIssuers`</span>, and <span class="pre">`OCSP`</span> URIs. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18379" class="reference external">bpo-18379</a>.)

</div>

<div id="stat" class="section">

### stat<a href="#stat" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/stat.html#module-stat" class="reference internal" title="stat: Utilities for interpreting the results of os.stat(), os.lstat() and os.fstat()."><span class="pre"><code class="sourceCode python">stat</code></span></a> module is now backed by a C implementation in <span class="pre">`_stat`</span>. A C implementation is required as most of the values aren’t standardized and are platform-dependent. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11016" class="reference external">bpo-11016</a>.)

The module supports new <a href="../library/stat.html#stat.ST_MODE" class="reference internal" title="stat.ST_MODE"><span class="pre"><code class="sourceCode python">ST_MODE</code></span></a> flags, <a href="../library/stat.html#stat.S_IFDOOR" class="reference internal" title="stat.S_IFDOOR"><span class="pre"><code class="sourceCode python">S_IFDOOR</code></span></a>, <a href="../library/stat.html#stat.S_IFPORT" class="reference internal" title="stat.S_IFPORT"><span class="pre"><code class="sourceCode python">S_IFPORT</code></span></a>, and <a href="../library/stat.html#stat.S_IFWHT" class="reference internal" title="stat.S_IFWHT"><span class="pre"><code class="sourceCode python">S_IFWHT</code></span></a>. (Contributed by Christian Hiemes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11016" class="reference external">bpo-11016</a>.)

</div>

<div id="struct" class="section">

### struct<a href="#struct" class="headerlink" title="Link to this heading">¶</a>

New function <a href="../library/struct.html#struct.iter_unpack" class="reference internal" title="struct.iter_unpack"><span class="pre"><code class="sourceCode python">iter_unpack</code></span></a> and a new <a href="../library/struct.html#struct.Struct.iter_unpack" class="reference internal" title="struct.Struct.iter_unpack"><span class="pre"><code class="sourceCode python">struct.Struct.iter_unpack()</code></span></a> method on compiled formats provide streamed unpacking of a buffer containing repeated instances of a given format of data. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17804" class="reference external">bpo-17804</a>.)

</div>

<div id="subprocess" class="section">

### subprocess<a href="#subprocess" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/subprocess.html#subprocess.check_output" class="reference internal" title="subprocess.check_output"><span class="pre"><code class="sourceCode python">check_output()</code></span></a> now accepts an *input* argument that can be used to provide the contents of <span class="pre">`stdin`</span> for the command that is run. (Contributed by Zack Weinberg in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16624" class="reference external">bpo-16624</a>.)

<a href="../library/subprocess.html#subprocess.getoutput" class="reference internal" title="subprocess.getoutput"><span class="pre"><code class="sourceCode python">getoutput()</code></span></a> and <a href="../library/subprocess.html#subprocess.getstatusoutput" class="reference internal" title="subprocess.getstatusoutput"><span class="pre"><code class="sourceCode python">getstatusoutput()</code></span></a> now work on Windows. This change was actually inadvertently made in 3.3.4. (Contributed by Tim Golden in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=10197" class="reference external">bpo-10197</a>.)

</div>

<div id="sunau" class="section">

### sunau<a href="#sunau" class="headerlink" title="Link to this heading">¶</a>

The <span class="pre">`getparams()`</span> method now returns a namedtuple rather than a plain tuple. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18901" class="reference external">bpo-18901</a>.)

<span class="pre">`sunau.open()`</span> now supports the context management protocol: when used in a <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> block, the <span class="pre">`close`</span> method of the returned object will be called automatically at the end of the block. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18878" class="reference external">bpo-18878</a>.)

<span class="pre">`AU_write.setsampwidth()`</span> now supports 24 bit samples, thus adding support for writing 24 sample using the module. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19261" class="reference external">bpo-19261</a>.)

The <span class="pre">`writeframesraw()`</span> and <span class="pre">`writeframes()`</span> methods now accept any <a href="../glossary.html#term-bytes-like-object" class="reference internal"><span class="xref std std-term">bytes-like object</span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8311" class="reference external">bpo-8311</a>.)

</div>

<div id="sys" class="section">

### sys<a href="#sys" class="headerlink" title="Link to this heading">¶</a>

New function <a href="../library/sys.html#sys.getallocatedblocks" class="reference internal" title="sys.getallocatedblocks"><span class="pre"><code class="sourceCode python">sys.getallocatedblocks()</code></span></a> returns the current number of blocks allocated by the interpreter. (In CPython with the default <span class="pre">`--with-pymalloc`</span> setting, this is allocations made through the <a href="../c-api/memory.html#c.PyObject_Malloc" class="reference internal" title="PyObject_Malloc"><span class="pre"><code class="sourceCode c">PyObject_Malloc<span class="op">()</span></code></span></a> API.) This can be useful for tracking memory leaks, especially if automated via a test suite. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13390" class="reference external">bpo-13390</a>.)

When the Python interpreter starts in <a href="../tutorial/interpreter.html#tut-interactive" class="reference internal"><span class="std std-ref">interactive mode</span></a>, it checks for an <a href="../library/sys.html#sys.__interactivehook__" class="reference internal" title="sys.__interactivehook__"><span class="pre"><code class="sourceCode python">__interactivehook__</code></span></a> attribute on the <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> module. If the attribute exists, its value is called with no arguments just before interactive mode is started. The check is made after the <span id="index-41" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONSTARTUP" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONSTARTUP</code></span></a> file is read, so it can be set there. The <a href="../library/site.html#module-site" class="reference internal" title="site: Module responsible for site-specific configuration."><span class="pre"><code class="sourceCode python">site</code></span></a> module <a href="../library/site.html#rlcompleter-config" class="reference internal"><span class="std std-ref">sets it</span></a> to a function that enables tab completion and history saving (in <span class="pre">`~/.python-history`</span>) if the platform supports <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline</code></span></a>. If you do not want this (new) behavior, you can override it in <span id="index-42" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONSTARTUP" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONSTARTUP</code></span></a>, <a href="../library/site.html#module-sitecustomize" class="reference internal" title="sitecustomize"><span class="pre"><code class="sourceCode python">sitecustomize</code></span></a>, or <a href="../library/site.html#module-usercustomize" class="reference internal" title="usercustomize"><span class="pre"><code class="sourceCode python">usercustomize</code></span></a> by deleting this attribute from <a href="../library/sys.html#module-sys" class="reference internal" title="sys: Access system-specific parameters and functions."><span class="pre"><code class="sourceCode python">sys</code></span></a> (or setting it to some other callable). (Contributed by Éric Araujo and Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5845" class="reference external">bpo-5845</a>.)

</div>

<div id="tarfile" class="section">

### tarfile<a href="#tarfile" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module now supports a simple <a href="../library/tarfile.html#tarfile-commandline" class="reference internal"><span class="std std-ref">Command-Line Interface</span></a> when called as a script directly or via <span class="pre">`-m`</span>. This can be used to create and extract tarfile archives. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13477" class="reference external">bpo-13477</a>.)

</div>

<div id="textwrap" class="section">

### textwrap<a href="#textwrap" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/textwrap.html#textwrap.TextWrapper" class="reference internal" title="textwrap.TextWrapper"><span class="pre"><code class="sourceCode python">TextWrapper</code></span></a> class has two new attributes/constructor arguments: <a href="../library/textwrap.html#textwrap.TextWrapper.max_lines" class="reference internal" title="textwrap.TextWrapper.max_lines"><span class="pre"><code class="sourceCode python">max_lines</code></span></a>, which limits the number of lines in the output, and <a href="../library/textwrap.html#textwrap.TextWrapper.placeholder" class="reference internal" title="textwrap.TextWrapper.placeholder"><span class="pre"><code class="sourceCode python">placeholder</code></span></a>, which is a string that will appear at the end of the output if it has been truncated because of *max_lines*. Building on these capabilities, a new convenience function <a href="../library/textwrap.html#textwrap.shorten" class="reference internal" title="textwrap.shorten"><span class="pre"><code class="sourceCode python">shorten()</code></span></a> collapses all of the whitespace in the input to single spaces and produces a single line of a given *width* that ends with the *placeholder* (by default, <span class="pre">`[...]`</span>). (Contributed by Antoine Pitrou and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18585" class="reference external">bpo-18585</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18725" class="reference external">bpo-18725</a>.)

</div>

<div id="threading" class="section">

### threading<a href="#threading" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/threading.html#threading.Thread" class="reference internal" title="threading.Thread"><span class="pre"><code class="sourceCode python">Thread</code></span></a> object representing the main thread can be obtained from the new <a href="../library/threading.html#threading.main_thread" class="reference internal" title="threading.main_thread"><span class="pre"><code class="sourceCode python">main_thread()</code></span></a> function. In normal conditions this will be the thread from which the Python interpreter was started. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18882" class="reference external">bpo-18882</a>.)

</div>

<div id="traceback" class="section">

### traceback<a href="#traceback" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/traceback.html#traceback.clear_frames" class="reference internal" title="traceback.clear_frames"><span class="pre"><code class="sourceCode python">traceback.clear_frames()</code></span></a> function takes a traceback object and clears the local variables in all of the frames it references, reducing the amount of memory consumed. (Contributed by Andrew Kuchling in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1565525" class="reference external">bpo-1565525</a>.)

</div>

<div id="types" class="section">

### types<a href="#types" class="headerlink" title="Link to this heading">¶</a>

A new <a href="../library/types.html#types.DynamicClassAttribute" class="reference internal" title="types.DynamicClassAttribute"><span class="pre"><code class="sourceCode python">DynamicClassAttribute()</code></span></a> descriptor provides a way to define an attribute that acts normally when looked up through an instance object, but which is routed to the *class* <span class="pre">`__getattr__`</span> when looked up through the class. This allows one to have properties active on a class, and have virtual attributes on the class with the same name (see <a href="../library/enum.html#module-enum" class="reference internal" title="enum: Implementation of an enumeration class."><span class="pre"><code class="sourceCode python">enum</code></span></a> for an example). (Contributed by Ethan Furman in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19030" class="reference external">bpo-19030</a>.)

</div>

<div id="urllib" class="section">

### urllib<a href="#urllib" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/urllib.request.html#module-urllib.request" class="reference internal" title="urllib.request: Extensible library for opening URLs."><span class="pre"><code class="sourceCode python">urllib.request</code></span></a> now supports <span class="pre">`data:`</span> URLs via the <a href="../library/urllib.request.html#urllib.request.DataHandler" class="reference internal" title="urllib.request.DataHandler"><span class="pre"><code class="sourceCode python">DataHandler</code></span></a> class. (Contributed by Mathias Panzenböck in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16423" class="reference external">bpo-16423</a>.)

The http method that will be used by a <a href="../library/urllib.request.html#urllib.request.Request" class="reference internal" title="urllib.request.Request"><span class="pre"><code class="sourceCode python">Request</code></span></a> class can now be specified by setting a <a href="../library/urllib.request.html#urllib.request.Request.method" class="reference internal" title="urllib.request.Request.method"><span class="pre"><code class="sourceCode python">method</code></span></a> class attribute on the subclass. (Contributed by Jason R Coombs in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18978" class="reference external">bpo-18978</a>.)

<a href="../library/urllib.request.html#urllib.request.Request" class="reference internal" title="urllib.request.Request"><span class="pre"><code class="sourceCode python">Request</code></span></a> objects are now reusable: if the <a href="../library/urllib.request.html#urllib.request.Request.full_url" class="reference internal" title="urllib.request.Request.full_url"><span class="pre"><code class="sourceCode python">full_url</code></span></a> or <a href="../library/urllib.request.html#urllib.request.Request.data" class="reference internal" title="urllib.request.Request.data"><span class="pre"><code class="sourceCode python">data</code></span></a> attributes are modified, all relevant internal properties are updated. This means, for example, that it is now possible to use the same <a href="../library/urllib.request.html#urllib.request.Request" class="reference internal" title="urllib.request.Request"><span class="pre"><code class="sourceCode python">Request</code></span></a> object in more than one <a href="../library/urllib.request.html#urllib.request.OpenerDirector.open" class="reference internal" title="urllib.request.OpenerDirector.open"><span class="pre"><code class="sourceCode python">OpenerDirector.<span class="bu">open</span>()</code></span></a> call with different *data* arguments, or to modify a <a href="../library/urllib.request.html#urllib.request.Request" class="reference internal" title="urllib.request.Request"><span class="pre"><code class="sourceCode python">Request</code></span></a>‘s <span class="pre">`url`</span> rather than recomputing it from scratch. There is also a new <a href="../library/urllib.request.html#urllib.request.Request.remove_header" class="reference internal" title="urllib.request.Request.remove_header"><span class="pre"><code class="sourceCode python">remove_header()</code></span></a> method that can be used to remove headers from a <a href="../library/urllib.request.html#urllib.request.Request" class="reference internal" title="urllib.request.Request"><span class="pre"><code class="sourceCode python">Request</code></span></a>. (Contributed by Alexey Kachayev in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16464" class="reference external">bpo-16464</a>, Daniel Wozniak in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17485" class="reference external">bpo-17485</a>, and Damien Brecht and Senthil Kumaran in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17272" class="reference external">bpo-17272</a>.)

<a href="../library/urllib.error.html#urllib.error.HTTPError" class="reference internal" title="urllib.error.HTTPError"><span class="pre"><code class="sourceCode python">HTTPError</code></span></a> objects now have a <a href="../library/urllib.error.html#urllib.error.HTTPError.headers" class="reference internal" title="urllib.error.HTTPError.headers"><span class="pre"><code class="sourceCode python">headers</code></span></a> attribute that provides access to the HTTP response headers associated with the error. (Contributed by Berker Peksag in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15701" class="reference external">bpo-15701</a>.)

</div>

<div id="unittest" class="section">

### unittest<a href="#unittest" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/unittest.html#unittest.TestCase" class="reference internal" title="unittest.TestCase"><span class="pre"><code class="sourceCode python">TestCase</code></span></a> class has a new method, <a href="../library/unittest.html#unittest.TestCase.subTest" class="reference internal" title="unittest.TestCase.subTest"><span class="pre"><code class="sourceCode python">subTest()</code></span></a>, that produces a context manager whose <a href="../reference/compound_stmts.html#with" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">with</code></span></a> block becomes a “sub-test”. This context manager allows a test method to dynamically generate subtests by, say, calling the <span class="pre">`subTest`</span> context manager inside a loop. A single test method can thereby produce an indefinite number of separately identified and separately counted tests, all of which will run even if one or more of them fail. For example:

<div class="highlight-python3 notranslate">

<div class="highlight">

    class NumbersTest(unittest.TestCase):
        def test_even(self):
            for i in range(6):
                with self.subTest(i=i):
                    self.assertEqual(i % 2, 0)

</div>

</div>

will result in six subtests, each identified in the unittest verbose output with a label consisting of the variable name <span class="pre">`i`</span> and a particular value for that variable (<span class="pre">`i=0`</span>, <span class="pre">`i=1`</span>, etc). See <a href="../library/unittest.html#subtests" class="reference internal"><span class="std std-ref">Distinguishing test iterations using subtests</span></a> for the full version of this example. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16997" class="reference external">bpo-16997</a>.)

<a href="../library/unittest.html#unittest.main" class="reference internal" title="unittest.main"><span class="pre"><code class="sourceCode python">unittest.main()</code></span></a> now accepts an iterable of test names for *defaultTest*, where previously it only accepted a single test name as a string. (Contributed by Jyrki Pulliainen in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15132" class="reference external">bpo-15132</a>.)

If <a href="../library/unittest.html#unittest.SkipTest" class="reference internal" title="unittest.SkipTest"><span class="pre"><code class="sourceCode python">SkipTest</code></span></a> is raised during test discovery (that is, at the module level in the test file), it is now reported as a skip instead of an error. (Contributed by Zach Ware in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16935" class="reference external">bpo-16935</a>.)

<a href="../library/unittest.html#unittest.TestLoader.discover" class="reference internal" title="unittest.TestLoader.discover"><span class="pre"><code class="sourceCode python">discover()</code></span></a> now sorts the discovered files to provide consistent test ordering. (Contributed by Martin Melin and Jeff Ramnani in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16709" class="reference external">bpo-16709</a>.)

<a href="../library/unittest.html#unittest.TestSuite" class="reference internal" title="unittest.TestSuite"><span class="pre"><code class="sourceCode python">TestSuite</code></span></a> now drops references to tests as soon as the test has been run, if the test is successful. On Python interpreters that do garbage collection, this allows the tests to be garbage collected if nothing else is holding a reference to the test. It is possible to override this behavior by creating a <a href="../library/unittest.html#unittest.TestSuite" class="reference internal" title="unittest.TestSuite"><span class="pre"><code class="sourceCode python">TestSuite</code></span></a> subclass that defines a custom <span class="pre">`_removeTestAtIndex`</span> method. (Contributed by Tom Wardill, Matt McClure, and Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11798" class="reference external">bpo-11798</a>.)

A new test assertion context-manager, <a href="../library/unittest.html#unittest.TestCase.assertLogs" class="reference internal" title="unittest.TestCase.assertLogs"><span class="pre"><code class="sourceCode python">assertLogs()</code></span></a>, will ensure that a given block of code emits a log message using the <a href="../library/logging.html#module-logging" class="reference internal" title="logging: Flexible event logging system for applications."><span class="pre"><code class="sourceCode python">logging</code></span></a> module. By default the message can come from any logger and have a priority of <span class="pre">`INFO`</span> or higher, but both the logger name and an alternative minimum logging level may be specified. The object returned by the context manager can be queried for the <a href="../library/logging.html#logging.LogRecord" class="reference internal" title="logging.LogRecord"><span class="pre"><code class="sourceCode python">LogRecord</code></span></a>s and/or formatted messages that were logged. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18937" class="reference external">bpo-18937</a>.)

Test discovery now works with namespace packages (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17457" class="reference external">bpo-17457</a>.)

<a href="../library/unittest.mock.html#module-unittest.mock" class="reference internal" title="unittest.mock: Mock object library."><span class="pre"><code class="sourceCode python">unittest.mock</code></span></a> objects now inspect their specification signatures when matching calls, which means an argument can now be matched by either position or name, instead of only by position. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17015" class="reference external">bpo-17015</a>.)

<a href="../library/unittest.mock.html#unittest.mock.mock_open" class="reference internal" title="unittest.mock.mock_open"><span class="pre"><code class="sourceCode python">mock_open()</code></span></a> objects now have <span class="pre">`readline`</span> and <span class="pre">`readlines`</span> methods. (Contributed by Toshio Kuratomi in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17467" class="reference external">bpo-17467</a>.)

</div>

<div id="venv" class="section">

### venv<a href="#venv" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/venv.html#module-venv" class="reference internal" title="venv: Creation of virtual environments."><span class="pre"><code class="sourceCode python">venv</code></span></a> now includes activation scripts for the <span class="pre">`csh`</span> and <span class="pre">`fish`</span> shells. (Contributed by Andrew Svetlov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15417" class="reference external">bpo-15417</a>.)

<a href="../library/venv.html#venv.EnvBuilder" class="reference internal" title="venv.EnvBuilder"><span class="pre"><code class="sourceCode python">EnvBuilder</code></span></a> and the <a href="../library/venv.html#venv.create" class="reference internal" title="venv.create"><span class="pre"><code class="sourceCode python">create()</code></span></a> convenience function take a new keyword argument *with_pip*, which defaults to <span class="pre">`False`</span>, that controls whether or not <a href="../library/venv.html#venv.EnvBuilder" class="reference internal" title="venv.EnvBuilder"><span class="pre"><code class="sourceCode python">EnvBuilder</code></span></a> ensures that <span class="pre">`pip`</span> is installed in the virtual environment. (Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19552" class="reference external">bpo-19552</a> as part of the <span id="index-43" class="target"></span><a href="https://peps.python.org/pep-0453/" class="pep reference external"><strong>PEP 453</strong></a> implementation.)

</div>

<div id="wave" class="section">

### wave<a href="#wave" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/wave.html#wave.Wave_read.getparams" class="reference internal" title="wave.Wave_read.getparams"><span class="pre"><code class="sourceCode python">getparams()</code></span></a> method now returns a namedtuple rather than a plain tuple. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17487" class="reference external">bpo-17487</a>.)

<a href="../library/wave.html#wave.open" class="reference internal" title="wave.open"><span class="pre"><code class="sourceCode python">wave.<span class="bu">open</span>()</code></span></a> now supports the context management protocol. (Contributed by Claudiu Popa in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17616" class="reference external">bpo-17616</a>.)

<a href="../library/wave.html#module-wave" class="reference internal" title="wave: Provide an interface to the WAV sound format."><span class="pre"><code class="sourceCode python">wave</code></span></a> can now <a href="../library/wave.html#wave-write-objects" class="reference internal"><span class="std std-ref">write output to unseekable files</span></a>. (Contributed by David Jones, Guilherme Polo, and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5202" class="reference external">bpo-5202</a>.)

The <a href="../library/wave.html#wave.Wave_write.writeframesraw" class="reference internal" title="wave.Wave_write.writeframesraw"><span class="pre"><code class="sourceCode python">writeframesraw()</code></span></a> and <a href="../library/wave.html#wave.Wave_write.writeframes" class="reference internal" title="wave.Wave_write.writeframes"><span class="pre"><code class="sourceCode python">writeframes()</code></span></a> methods now accept any <a href="../glossary.html#term-bytes-like-object" class="reference internal"><span class="xref std std-term">bytes-like object</span></a>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=8311" class="reference external">bpo-8311</a>.)

</div>

<div id="weakref" class="section">

### weakref<a href="#weakref" class="headerlink" title="Link to this heading">¶</a>

New <a href="../library/weakref.html#weakref.WeakMethod" class="reference internal" title="weakref.WeakMethod"><span class="pre"><code class="sourceCode python">WeakMethod</code></span></a> class simulates weak references to bound methods. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14631" class="reference external">bpo-14631</a>.)

New <a href="../library/weakref.html#weakref.finalize" class="reference internal" title="weakref.finalize"><span class="pre"><code class="sourceCode python">finalize</code></span></a> class makes it possible to register a callback to be invoked when an object is garbage collected, without needing to carefully manage the lifecycle of the weak reference itself. (Contributed by Richard Oudkerk in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15528" class="reference external">bpo-15528</a>.)

The callback, if any, associated with a <a href="../library/weakref.html#weakref.ref" class="reference internal" title="weakref.ref"><span class="pre"><code class="sourceCode python">ref</code></span></a> is now exposed via the <a href="../library/weakref.html#weakref.ref.__callback__" class="reference internal" title="weakref.ref.__callback__"><span class="pre"><code class="sourceCode python">__callback__</code></span></a> attribute. (Contributed by Mark Dickinson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17643" class="reference external">bpo-17643</a>.)

</div>

<div id="xml-etree" class="section">

### xml.etree<a href="#xml-etree" class="headerlink" title="Link to this heading">¶</a>

A new parser, <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLPullParser" class="reference internal" title="xml.etree.ElementTree.XMLPullParser"><span class="pre"><code class="sourceCode python">XMLPullParser</code></span></a>, allows a non-blocking applications to parse XML documents. An example can be seen at <a href="../library/xml.etree.elementtree.html#elementtree-pull-parsing" class="reference internal"><span class="std std-ref">Pull API for non-blocking parsing</span></a>. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17741" class="reference external">bpo-17741</a>.)

The <a href="../library/xml.etree.elementtree.html#module-xml.etree.ElementTree" class="reference internal" title="xml.etree.ElementTree: Implementation of the ElementTree API."><span class="pre"><code class="sourceCode python">xml.etree.ElementTree</code></span></a> <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.tostring" class="reference internal" title="xml.etree.ElementTree.tostring"><span class="pre"><code class="sourceCode python">tostring()</code></span></a> and <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.tostringlist" class="reference internal" title="xml.etree.ElementTree.tostringlist"><span class="pre"><code class="sourceCode python">tostringlist()</code></span></a> functions, and the <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.ElementTree" class="reference internal" title="xml.etree.ElementTree.ElementTree"><span class="pre"><code class="sourceCode python">ElementTree</code></span></a> <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.ElementTree.write" class="reference internal" title="xml.etree.ElementTree.ElementTree.write"><span class="pre"><code class="sourceCode python">write()</code></span></a> method, now have a *short_empty_elements* <a href="../glossary.html#keyword-only-parameter" class="reference internal"><span class="std std-ref">keyword-only parameter</span></a> providing control over whether elements with no content are written in abbreviated (<span class="pre">`<tag`</span>` `<span class="pre">`/>`</span>) or expanded (<span class="pre">`<tag></tag>`</span>) form. (Contributed by Ariel Poliak and Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14377" class="reference external">bpo-14377</a>.)

</div>

<div id="zipfile" class="section">

### zipfile<a href="#zipfile" class="headerlink" title="Link to this heading">¶</a>

The <a href="../library/zipfile.html#zipfile.PyZipFile.writepy" class="reference internal" title="zipfile.PyZipFile.writepy"><span class="pre"><code class="sourceCode python">writepy()</code></span></a> method of the <a href="../library/zipfile.html#zipfile.PyZipFile" class="reference internal" title="zipfile.PyZipFile"><span class="pre"><code class="sourceCode python">PyZipFile</code></span></a> class has a new *filterfunc* option that can be used to control which directories and files are added to the archive. For example, this could be used to exclude test files from the archive. (Contributed by Christian Tismer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19274" class="reference external">bpo-19274</a>.)

The *allowZip64* parameter to <a href="../library/zipfile.html#zipfile.ZipFile" class="reference internal" title="zipfile.ZipFile"><span class="pre"><code class="sourceCode python">ZipFile</code></span></a> and <a href="../library/zipfile.html#zipfile.PyZipFile" class="reference internal" title="zipfile.PyZipFile"><span class="pre"><code class="sourceCode python">PyZipFile</code></span></a> is now <span class="pre">`True`</span> by default. (Contributed by William Mallard in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17201" class="reference external">bpo-17201</a>.)

</div>

</div>

<div id="cpython-implementation-changes" class="section">

## CPython Implementation Changes<a href="#cpython-implementation-changes" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-445-customization-of-cpython-memory-allocators" class="section">

<span id="whatsnew-pep-445"></span>

### PEP 445: Customization of CPython Memory Allocators<a href="#pep-445-customization-of-cpython-memory-allocators" class="headerlink" title="Link to this heading">¶</a>

<span id="index-44" class="target"></span><a href="https://peps.python.org/pep-0445/" class="pep reference external"><strong>PEP 445</strong></a> adds new C level interfaces to customize memory allocation in the CPython interpreter.

<div class="admonition seealso">

See also

<span id="index-45" class="target"></span><a href="https://peps.python.org/pep-0445/" class="pep reference external"><strong>PEP 445</strong></a> – Add new APIs to customize Python memory allocators  
PEP written and implemented by Victor Stinner.

</div>

</div>

<div id="pep-442-safe-object-finalization" class="section">

<span id="whatsnew-pep-442"></span>

### PEP 442: Safe Object Finalization<a href="#pep-442-safe-object-finalization" class="headerlink" title="Link to this heading">¶</a>

<span id="index-46" class="target"></span><a href="https://peps.python.org/pep-0442/" class="pep reference external"><strong>PEP 442</strong></a> removes the current limitations and quirks of object finalization in CPython. With it, objects with <a href="../reference/datamodel.html#object.__del__" class="reference internal" title="object.__del__"><span class="pre"><code class="sourceCode python"><span class="fu">__del__</span>()</code></span></a> methods, as well as generators with <a href="../reference/compound_stmts.html#finally" class="reference internal"><span class="pre"><code class="xref std std-keyword docutils literal notranslate">finally</code></span></a> clauses, can be finalized when they are part of a reference cycle.

As part of this change, module globals are no longer forcibly set to <a href="../library/constants.html#None" class="reference internal" title="None"><span class="pre"><code class="sourceCode python"><span class="va">None</span></code></span></a> during interpreter shutdown in most cases, instead relying on the normal operation of the cyclic garbage collector. This avoids a whole class of interpreter-shutdown-time errors, usually involving <span class="pre">`__del__`</span> methods, that have plagued Python since the cyclic GC was first introduced.

<div class="admonition seealso">

See also

<span id="index-47" class="target"></span><a href="https://peps.python.org/pep-0442/" class="pep reference external"><strong>PEP 442</strong></a> – Safe object finalization  
PEP written and implemented by Antoine Pitrou.

</div>

</div>

<div id="pep-456-secure-and-interchangeable-hash-algorithm" class="section">

<span id="whatsnew-pep-456"></span>

### PEP 456: Secure and Interchangeable Hash Algorithm<a href="#pep-456-secure-and-interchangeable-hash-algorithm" class="headerlink" title="Link to this heading">¶</a>

<span id="index-48" class="target"></span><a href="https://peps.python.org/pep-0456/" class="pep reference external"><strong>PEP 456</strong></a> follows up on earlier security fix work done on Python’s hash algorithm to address certain DOS attacks to which public facing APIs backed by dictionary lookups may be subject. (See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14621" class="reference external">bpo-14621</a> for the start of the current round of improvements.) The PEP unifies CPython’s hash code to make it easier for a packager to substitute a different hash algorithm, and switches Python’s default implementation to a SipHash implementation on platforms that have a 64 bit data type. Any performance differences in comparison with the older FNV algorithm are trivial.

The PEP adds additional fields to the <a href="../library/sys.html#sys.hash_info" class="reference internal" title="sys.hash_info"><span class="pre"><code class="sourceCode python">sys.hash_info</code></span></a> named tuple to describe the hash algorithm in use by the currently executing binary. Otherwise, the PEP does not alter any existing CPython APIs.

</div>

<div id="pep-436-argument-clinic" class="section">

<span id="whatsnew-pep-436"></span>

### PEP 436: Argument Clinic<a href="#pep-436-argument-clinic" class="headerlink" title="Link to this heading">¶</a>

“Argument Clinic” (<span id="index-49" class="target"></span><a href="https://peps.python.org/pep-0436/" class="pep reference external"><strong>PEP 436</strong></a>) is now part of the CPython build process and can be used to simplify the process of defining and maintaining accurate signatures for builtins and standard library extension modules implemented in C.

Some standard library extension modules have been converted to use Argument Clinic in Python 3.4, and <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> and <a href="../library/inspect.html#module-inspect" class="reference internal" title="inspect: Extract information and source code from live objects."><span class="pre"><code class="sourceCode python">inspect</code></span></a> have been updated accordingly.

It is expected that signature metadata for programmatic introspection will be added to additional callables implemented in C as part of Python 3.4 maintenance releases.

<div class="admonition note">

Note

The Argument Clinic PEP is not fully up to date with the state of the implementation. This has been deemed acceptable by the release manager and core development team in this case, as Argument Clinic will not be made available as a public API for third party use in Python 3.4.

</div>

<div class="admonition seealso">

See also

<span id="index-50" class="target"></span><a href="https://peps.python.org/pep-0436/" class="pep reference external"><strong>PEP 436</strong></a> – The Argument Clinic DSL  
PEP written and implemented by Larry Hastings.

</div>

</div>

<div id="other-build-and-c-api-changes" class="section">

### Other Build and C API Changes<a href="#other-build-and-c-api-changes" class="headerlink" title="Link to this heading">¶</a>

- The new <a href="../c-api/type.html#c.PyType_GetSlot" class="reference internal" title="PyType_GetSlot"><span class="pre"><code class="sourceCode c">PyType_GetSlot<span class="op">()</span></code></span></a> function has been added to the stable ABI, allowing retrieval of function pointers from named type slots when using the limited API. (Contributed by Martin von Löwis in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17162" class="reference external">bpo-17162</a>.)

- The new <span class="pre">`Py_SetStandardStreamEncoding()`</span> pre-initialization API allows applications embedding the CPython interpreter to reliably force a particular encoding and error handler for the standard streams. (Contributed by Bastien Montagne and Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16129" class="reference external">bpo-16129</a>.)

- Most Python C APIs that don’t mutate string arguments are now correctly marked as accepting <span class="pre">`const`</span>` `<span class="pre">`char`</span>` `<span class="pre">`*`</span> rather than <span class="pre">`char`</span>` `<span class="pre">`*`</span>. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=1772673" class="reference external">bpo-1772673</a>.)

- A new shell version of <span class="pre">`python-config`</span> can be used even when a python interpreter is not available (for example, in cross compilation scenarios).

- <a href="../c-api/unicode.html#c.PyUnicode_FromFormat" class="reference internal" title="PyUnicode_FromFormat"><span class="pre"><code class="sourceCode c">PyUnicode_FromFormat<span class="op">()</span></code></span></a> now supports width and precision specifications for <span class="pre">`%s`</span>, <span class="pre">`%A`</span>, <span class="pre">`%U`</span>, <span class="pre">`%V`</span>, <span class="pre">`%S`</span>, and <span class="pre">`%R`</span>. (Contributed by Ysj Ray and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7330" class="reference external">bpo-7330</a>.)

- New function <a href="../c-api/tuple.html#c.PyStructSequence_InitType2" class="reference internal" title="PyStructSequence_InitType2"><span class="pre"><code class="sourceCode c">PyStructSequence_InitType2<span class="op">()</span></code></span></a> supplements the existing <a href="../c-api/tuple.html#c.PyStructSequence_InitType" class="reference internal" title="PyStructSequence_InitType"><span class="pre"><code class="sourceCode c">PyStructSequence_InitType<span class="op">()</span></code></span></a> function. The difference is that it returns <span class="pre">`0`</span> on success and <span class="pre">`-1`</span> on failure.

- The CPython source can now be compiled using the address sanity checking features of recent versions of GCC and clang: the false alarms in the small object allocator have been silenced. (Contributed by Dhiru Kholia in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18596" class="reference external">bpo-18596</a>.)

- The Windows build now uses <a href="https://en.wikipedia.org/wiki/Address_space_layout_randomization" class="reference external">Address Space Layout Randomization</a> and <a href="https://en.wikipedia.org/wiki/Data_Execution_Prevention" class="reference external">Data Execution Prevention</a>. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16632" class="reference external">bpo-16632</a>.)

- New function <a href="../c-api/object.html#c.PyObject_LengthHint" class="reference internal" title="PyObject_LengthHint"><span class="pre"><code class="sourceCode c">PyObject_LengthHint<span class="op">()</span></code></span></a> is the C API equivalent of <a href="../library/operator.html#operator.length_hint" class="reference internal" title="operator.length_hint"><span class="pre"><code class="sourceCode python">operator.length_hint()</code></span></a>. (Contributed by Armin Ronacher in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16148" class="reference external">bpo-16148</a>.)

</div>

<div id="other-improvements" class="section">

<span id="other-improvements-3-4"></span>

### Other Improvements<a href="#other-improvements" class="headerlink" title="Link to this heading">¶</a>

- The <a href="../using/cmdline.html#using-on-cmdline" class="reference internal"><span class="std std-ref">python</span></a> command has a new <a href="../using/cmdline.html#using-on-misc-options" class="reference internal"><span class="std std-ref">option</span></a>, <span class="pre">`-I`</span>, which causes it to run in “isolated mode”, which means that <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> contains neither the script’s directory nor the user’s <span class="pre">`site-packages`</span> directory, and all <span class="pre">`PYTHON*`</span> environment variables are ignored (it implies both <span class="pre">`-s`</span> and <span class="pre">`-E`</span>). Other restrictions may also be applied in the future, with the goal being to isolate the execution of a script from the user’s environment. This is appropriate, for example, when Python is used to run a system script. On most POSIX systems it can and should be used in the <span class="pre">`#!`</span> line of system scripts. (Contributed by Christian Heimes in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16499" class="reference external">bpo-16499</a>.)

- Tab-completion is now enabled by default in the interactive interpreter on systems that support <a href="../library/readline.html#module-readline" class="reference internal" title="readline: GNU readline support for Python. (Unix)"><span class="pre"><code class="sourceCode python">readline</code></span></a>. History is also enabled by default, and is written to (and read from) the file <span class="pre">`~/.python-history`</span>. (Contributed by Antoine Pitrou and Éric Araujo in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=5845" class="reference external">bpo-5845</a>.)

- Invoking the Python interpreter with <span class="pre">`--version`</span> now outputs the version to standard output instead of standard error (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18338" class="reference external">bpo-18338</a>). Similar changes were made to <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18920" class="reference external">bpo-18920</a>) and other modules that have script-like invocation capabilities (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18922" class="reference external">bpo-18922</a>).

- The CPython Windows installer now adds <span class="pre">`.py`</span> to the <span id="index-51" class="target"></span><span class="pre">`PATHEXT`</span> variable when extensions are registered, allowing users to run a python script at the windows command prompt by just typing its name without the <span class="pre">`.py`</span> extension. (Contributed by Paul Moore in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18569" class="reference external">bpo-18569</a>.)

- A new <span class="pre">`make`</span> target <a href="https://devguide.python.org/coverage/#measuring-coverage-of-c-code-with-gcov-and-lcov" class="reference external">coverage-report</a> will build python, run the test suite, and generate an HTML coverage report for the C codebase using <span class="pre">`gcov`</span> and <a href="https://github.com/linux-test-project/lcov" class="reference external">lcov</a>.

- The <span class="pre">`-R`</span> option to the <a href="../library/test.html#regrtest" class="reference internal"><span class="std std-ref">python regression test suite</span></a> now also checks for memory allocation leaks, using <a href="../library/sys.html#sys.getallocatedblocks" class="reference internal" title="sys.getallocatedblocks"><span class="pre"><code class="sourceCode python">sys.getallocatedblocks()</code></span></a>. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13390" class="reference external">bpo-13390</a>.)

- <span class="pre">`python`</span>` `<span class="pre">`-m`</span> now works with namespace packages.

- The <a href="../library/stat.html#module-stat" class="reference internal" title="stat: Utilities for interpreting the results of os.stat(), os.lstat() and os.fstat()."><span class="pre"><code class="sourceCode python">stat</code></span></a> module is now implemented in C, which means it gets the values for its constants from the C header files, instead of having the values hard-coded in the python module as was previously the case.

- Loading multiple python modules from a single OS module (<span class="pre">`.so`</span>, <span class="pre">`.dll`</span>) now works correctly (previously it silently returned the first python module in the file). (Contributed by Václav Šmilauer in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16421" class="reference external">bpo-16421</a>.)

- A new opcode, <span class="pre">`LOAD_CLASSDEREF`</span>, has been added to fix a bug in the loading of free variables in class bodies that could be triggered by certain uses of <a href="../reference/datamodel.html#prepare" class="reference internal"><span class="std std-ref">__prepare__</span></a>. (Contributed by Benjamin Peterson in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17853" class="reference external">bpo-17853</a>.)

- A number of MemoryError-related crashes were identified and fixed by Victor Stinner using his <span id="index-52" class="target"></span><a href="https://peps.python.org/pep-0445/" class="pep reference external"><strong>PEP 445</strong></a>-based <span class="pre">`pyfailmalloc`</span> tool (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18408" class="reference external">bpo-18408</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18520" class="reference external">bpo-18520</a>).

- The <span class="pre">`pyvenv`</span> command now accepts a <span class="pre">`--copies`</span> option to use copies rather than symlinks even on systems where symlinks are the default. (Contributed by Vinay Sajip in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18807" class="reference external">bpo-18807</a>.)

- The <span class="pre">`pyvenv`</span> command also accepts a <span class="pre">`--without-pip`</span> option to suppress the otherwise-automatic bootstrapping of pip into the virtual environment. (Contributed by Nick Coghlan in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19552" class="reference external">bpo-19552</a> as part of the <span id="index-53" class="target"></span><a href="https://peps.python.org/pep-0453/" class="pep reference external"><strong>PEP 453</strong></a> implementation.)

- The encoding name is now optional in the value set for the <span id="index-54" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONIOENCODING" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONIOENCODING</code></span></a> environment variable. This makes it possible to set just the error handler, without changing the default encoding. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18818" class="reference external">bpo-18818</a>.)

- The <a href="../library/bz2.html#module-bz2" class="reference internal" title="bz2: Interfaces for bzip2 compression and decompression."><span class="pre"><code class="sourceCode python">bz2</code></span></a>, <a href="../library/lzma.html#module-lzma" class="reference internal" title="lzma: A Python wrapper for the liblzma compression library."><span class="pre"><code class="sourceCode python">lzma</code></span></a>, and <a href="../library/gzip.html#module-gzip" class="reference internal" title="gzip: Interfaces for gzip compression and decompression using file objects."><span class="pre"><code class="sourceCode python">gzip</code></span></a> module <span class="pre">`open`</span> functions now support <span class="pre">`x`</span> (exclusive creation) mode. (Contributed by Tim Heaney and Vajrasky Kok in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19201" class="reference external">bpo-19201</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19222" class="reference external">bpo-19222</a>, and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19223" class="reference external">bpo-19223</a>.)

</div>

<div id="significant-optimizations" class="section">

### Significant Optimizations<a href="#significant-optimizations" class="headerlink" title="Link to this heading">¶</a>

- The UTF-32 decoder is now 3x to 4x faster. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14625" class="reference external">bpo-14625</a>.)

- The cost of hash collisions for sets is now reduced. Each hash table probe now checks a series of consecutive, adjacent key/hash pairs before continuing to make random probes through the hash table. This exploits cache locality to make collision resolution less expensive. The collision resolution scheme can be described as a hybrid of linear probing and open addressing. The number of additional linear probes defaults to nine. This can be changed at compile-time by defining LINEAR_PROBES to be any value. Set LINEAR_PROBES=0 to turn-off linear probing entirely. (Contributed by Raymond Hettinger in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18771" class="reference external">bpo-18771</a>.)

- The interpreter starts about 30% faster. A couple of measures lead to the speedup. The interpreter loads fewer modules on startup, e.g. the <a href="../library/re.html#module-re" class="reference internal" title="re: Regular expression operations."><span class="pre"><code class="sourceCode python">re</code></span></a>, <a href="../library/collections.html#module-collections" class="reference internal" title="collections: Container datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> and <a href="../library/locale.html#module-locale" class="reference internal" title="locale: Internationalization services."><span class="pre"><code class="sourceCode python">locale</code></span></a> modules and their dependencies are no longer imported by default. The marshal module has been improved to load compiled Python code faster. (Contributed by Antoine Pitrou, Christian Heimes and Victor Stinner in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19219" class="reference external">bpo-19219</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19218" class="reference external">bpo-19218</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19209" class="reference external">bpo-19209</a>, <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19205" class="reference external">bpo-19205</a> and <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9548" class="reference external">bpo-9548</a>.)

- <a href="../library/bz2.html#bz2.BZ2File" class="reference internal" title="bz2.BZ2File"><span class="pre"><code class="sourceCode python">bz2.BZ2File</code></span></a> is now as fast or faster than the Python2 version for most cases. <a href="../library/lzma.html#lzma.LZMAFile" class="reference internal" title="lzma.LZMAFile"><span class="pre"><code class="sourceCode python">lzma.LZMAFile</code></span></a> has also been optimized. (Contributed by Serhiy Storchaka and Nadeem Vawda in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16034" class="reference external">bpo-16034</a>.)

- <a href="../library/random.html#random.getrandbits" class="reference internal" title="random.getrandbits"><span class="pre"><code class="sourceCode python">random.getrandbits()</code></span></a> is 20%-40% faster for small integers (the most common use case). (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16674" class="reference external">bpo-16674</a>.)

- By taking advantage of the new storage format for strings, pickling of strings is now significantly faster. (Contributed by Victor Stinner and Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15596" class="reference external">bpo-15596</a>.)

- A performance issue in <span class="pre">`io.FileIO.readall()`</span> has been solved. This particularly affects Windows, and significantly speeds up the case of piping significant amounts of data through <a href="../library/subprocess.html#module-subprocess" class="reference internal" title="subprocess: Subprocess management."><span class="pre"><code class="sourceCode python">subprocess</code></span></a>. (Contributed by Richard Oudkerk in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15758" class="reference external">bpo-15758</a>.)

- <a href="../library/html.html#html.escape" class="reference internal" title="html.escape"><span class="pre"><code class="sourceCode python">html.escape()</code></span></a> is now 10x faster. (Contributed by Matt Bryant in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18020" class="reference external">bpo-18020</a>.)

- On Windows, the native <span class="pre">`VirtualAlloc`</span> is now used instead of the CRT <span class="pre">`malloc`</span> in <span class="pre">`obmalloc`</span>. Artificial benchmarks show about a 3% memory savings.

- <a href="../library/os.html#os.urandom" class="reference internal" title="os.urandom"><span class="pre"><code class="sourceCode python">os.urandom()</code></span></a> now uses a lazily opened persistent file descriptor so as to avoid using many file descriptors when run in parallel from multiple threads. (Contributed by Antoine Pitrou in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18756" class="reference external">bpo-18756</a>.)

</div>

</div>

<div id="deprecated" class="section">

<span id="deprecated-3-4"></span>

## Deprecated<a href="#deprecated" class="headerlink" title="Link to this heading">¶</a>

This section covers various APIs and other features that have been deprecated in Python 3.4, and will be removed in Python 3.5 or later. In most (but not all) cases, using the deprecated APIs will produce a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> when the interpreter is run with deprecation warnings enabled (for example, by using <span class="pre">`-Wd`</span>).

<div id="deprecations-in-the-python-api" class="section">

### Deprecations in the Python API<a href="#deprecations-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- As mentioned in <a href="#whatsnew-pep-451" class="reference internal"><span class="std std-ref">PEP 451: A ModuleSpec Type for the Import System</span></a>, a number of <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a> methods and functions are deprecated: <span class="pre">`importlib.find_loader()`</span> is replaced by <a href="../library/importlib.html#importlib.util.find_spec" class="reference internal" title="importlib.util.find_spec"><span class="pre"><code class="sourceCode python">importlib.util.find_spec()</code></span></a>; <span class="pre">`importlib.machinery.PathFinder.find_module()`</span> is replaced by <a href="../library/importlib.html#importlib.machinery.PathFinder.find_spec" class="reference internal" title="importlib.machinery.PathFinder.find_spec"><span class="pre"><code class="sourceCode python">importlib.machinery.PathFinder.find_spec()</code></span></a>; <span class="pre">`importlib.abc.MetaPathFinder.find_module()`</span> is replaced by <a href="../library/importlib.html#importlib.abc.MetaPathFinder.find_spec" class="reference internal" title="importlib.abc.MetaPathFinder.find_spec"><span class="pre"><code class="sourceCode python">importlib.abc.MetaPathFinder.find_spec()</code></span></a>; <span class="pre">`importlib.abc.PathEntryFinder.find_loader()`</span> and <span class="pre">`find_module()`</span> are replaced by <a href="../library/importlib.html#importlib.abc.PathEntryFinder.find_spec" class="reference internal" title="importlib.abc.PathEntryFinder.find_spec"><span class="pre"><code class="sourceCode python">importlib.abc.PathEntryFinder.find_spec()</code></span></a>; all of the *<span class="pre">`xxx`</span>*<span class="pre">`Loader`</span> ABC <span class="pre">`load_module`</span> methods (<span class="pre">`importlib.abc.Loader.load_module()`</span>, <span class="pre">`importlib.abc.InspectLoader.load_module()`</span>, <span class="pre">`importlib.abc.FileLoader.load_module()`</span>, <span class="pre">`importlib.abc.SourceLoader.load_module()`</span>) should no longer be implemented, instead loaders should implement an <span class="pre">`exec_module`</span> method (<a href="../library/importlib.html#importlib.abc.Loader.exec_module" class="reference internal" title="importlib.abc.Loader.exec_module"><span class="pre"><code class="sourceCode python">importlib.abc.Loader.exec_module()</code></span></a>, <a href="../library/importlib.html#importlib.abc.InspectLoader.exec_module" class="reference internal" title="importlib.abc.InspectLoader.exec_module"><span class="pre"><code class="sourceCode python">importlib.abc.InspectLoader.exec_module()</code></span></a> <a href="../library/importlib.html#importlib.abc.SourceLoader.exec_module" class="reference internal" title="importlib.abc.SourceLoader.exec_module"><span class="pre"><code class="sourceCode python">importlib.abc.SourceLoader.exec_module()</code></span></a>) and let the import system take care of the rest; and <span class="pre">`importlib.abc.Loader.module_repr()`</span>, <span class="pre">`importlib.util.module_for_loader()`</span>, <span class="pre">`importlib.util.set_loader()`</span>, and <span class="pre">`importlib.util.set_package()`</span> are no longer needed because their functions are now handled automatically by the import system.

- The <span class="pre">`imp`</span> module is pending deprecation. To keep compatibility with Python 2/3 code bases, the module’s removal is currently not scheduled.

- The <span class="pre">`formatter`</span> module is pending deprecation and is slated for removal in Python 3.6.

- <span class="pre">`MD5`</span> as the default *digestmod* for the <a href="../library/hmac.html#hmac.new" class="reference internal" title="hmac.new"><span class="pre"><code class="sourceCode python">hmac.new()</code></span></a> function is deprecated. Python 3.6 will require an explicit digest name or constructor as *digestmod* argument.

- The internal <span class="pre">`Netrc`</span> class in the <a href="../library/ftplib.html#module-ftplib" class="reference internal" title="ftplib: FTP protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">ftplib</code></span></a> module has been documented as deprecated in its docstring for quite some time. It now emits a <a href="../library/exceptions.html#DeprecationWarning" class="reference internal" title="DeprecationWarning"><span class="pre"><code class="sourceCode python"><span class="pp">DeprecationWarning</span></code></span></a> and will be removed completely in Python 3.5.

- The undocumented *endtime* argument to <a href="../library/subprocess.html#subprocess.Popen.wait" class="reference internal" title="subprocess.Popen.wait"><span class="pre"><code class="sourceCode python">subprocess.Popen.wait()</code></span></a> should not have been exposed and is hopefully not in use; it is deprecated and will mostly likely be removed in Python 3.5.

- The *strict* argument of <a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">HTMLParser</code></span></a> is deprecated.

- The <a href="../library/plistlib.html#module-plistlib" class="reference internal" title="plistlib: Generate and parse Apple plist files."><span class="pre"><code class="sourceCode python">plistlib</code></span></a> <span class="pre">`readPlist()`</span>, <span class="pre">`writePlist()`</span>, <span class="pre">`readPlistFromBytes()`</span>, and <span class="pre">`writePlistToBytes()`</span> functions are deprecated in favor of the corresponding new functions <a href="../library/plistlib.html#plistlib.load" class="reference internal" title="plistlib.load"><span class="pre"><code class="sourceCode python">load()</code></span></a>, <a href="../library/plistlib.html#plistlib.dump" class="reference internal" title="plistlib.dump"><span class="pre"><code class="sourceCode python">dump()</code></span></a>, <a href="../library/plistlib.html#plistlib.loads" class="reference internal" title="plistlib.loads"><span class="pre"><code class="sourceCode python">loads()</code></span></a>, and <a href="../library/plistlib.html#plistlib.dumps" class="reference internal" title="plistlib.dumps"><span class="pre"><code class="sourceCode python">dumps()</code></span></a>. <span class="pre">`Data()`</span> is deprecated in favor of just using the <a href="../library/stdtypes.html#bytes" class="reference internal" title="bytes"><span class="pre"><code class="sourceCode python"><span class="bu">bytes</span></code></span></a> constructor.

- The <a href="../library/sysconfig.html#module-sysconfig" class="reference internal" title="sysconfig: Python&#39;s configuration information"><span class="pre"><code class="sourceCode python">sysconfig</code></span></a> key <span class="pre">`SO`</span> is deprecated, it has been replaced by <span class="pre">`EXT_SUFFIX`</span>.

- The <span class="pre">`U`</span> mode accepted by various <span class="pre">`open`</span> functions is deprecated. In Python3 it does not do anything useful, and should be replaced by appropriate uses of <a href="../library/io.html#io.TextIOWrapper" class="reference internal" title="io.TextIOWrapper"><span class="pre"><code class="sourceCode python">io.TextIOWrapper</code></span></a> (if needed) and its *newline* argument.

- The *parser* argument of <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.iterparse" class="reference internal" title="xml.etree.ElementTree.iterparse"><span class="pre"><code class="sourceCode python">xml.etree.ElementTree.iterparse()</code></span></a> has been deprecated, as has the *html* argument of <a href="../library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLParser" class="reference internal" title="xml.etree.ElementTree.XMLParser"><span class="pre"><code class="sourceCode python">XMLParser()</code></span></a>. To prepare for the removal of the latter, all arguments to <span class="pre">`XMLParser`</span> should be passed by keyword.

</div>

<div id="deprecated-features" class="section">

### Deprecated Features<a href="#deprecated-features" class="headerlink" title="Link to this heading">¶</a>

- Running <a href="../library/idle.html#idle" class="reference internal"><span class="std std-ref">IDLE — Python editor and shell</span></a> with the <span class="pre">`-n`</span> flag (no subprocess) is deprecated. However, the feature will not be removed until <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18823" class="reference external">bpo-18823</a> is resolved.

- The site module adding a “site-python” directory to sys.path, if it exists, is deprecated (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19375" class="reference external">bpo-19375</a>).

</div>

</div>

<div id="removed" class="section">

## Removed<a href="#removed" class="headerlink" title="Link to this heading">¶</a>

<div id="operating-systems-no-longer-supported" class="section">

### Operating Systems No Longer Supported<a href="#operating-systems-no-longer-supported" class="headerlink" title="Link to this heading">¶</a>

Support for the following operating systems has been removed from the source and build tools:

- OS/2 (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16135" class="reference external">bpo-16135</a>).

- Windows 2000 (changeset e52df05b496a).

- Windows systems where <span class="pre">`COMSPEC`</span> points to <span class="pre">`command.com`</span> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14470" class="reference external">bpo-14470</a>).

- VMS (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16136" class="reference external">bpo-16136</a>).

</div>

<div id="api-and-feature-removals" class="section">

### API and Feature Removals<a href="#api-and-feature-removals" class="headerlink" title="Link to this heading">¶</a>

The following obsolete and previously deprecated APIs and features have been removed:

- The unmaintained <span class="pre">`Misc/TextMate`</span> and <span class="pre">`Misc/vim`</span> directories have been removed (see the <a href="https://devguide.python.org" class="reference external">devguide</a> for suggestions on what to use instead).

- The <span class="pre">`SO`</span> makefile macro is removed (it was replaced by the <span class="pre">`SHLIB_SUFFIX`</span> and <span class="pre">`EXT_SUFFIX`</span> macros) (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16754" class="reference external">bpo-16754</a>).

- The <span class="pre">`PyThreadState.tick_counter`</span> field has been removed; its value has been meaningless since Python 3.2, when the “new GIL” was introduced (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19199" class="reference external">bpo-19199</a>).

- <span class="pre">`PyLoader`</span> and <span class="pre">`PyPycLoader`</span> have been removed from <a href="../library/importlib.html#module-importlib" class="reference internal" title="importlib: The implementation of the import machinery."><span class="pre"><code class="sourceCode python">importlib</code></span></a>. (Contributed by Taras Lyapun in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15641" class="reference external">bpo-15641</a>.)

- The *strict* argument to <a href="../library/http.client.html#http.client.HTTPConnection" class="reference internal" title="http.client.HTTPConnection"><span class="pre"><code class="sourceCode python">HTTPConnection</code></span></a> and <a href="../library/http.client.html#http.client.HTTPSConnection" class="reference internal" title="http.client.HTTPSConnection"><span class="pre"><code class="sourceCode python">HTTPSConnection</code></span></a> has been removed. HTTP 0.9-style “Simple Responses” are no longer supported.

- The deprecated <a href="../library/urllib.request.html#urllib.request.Request" class="reference internal" title="urllib.request.Request"><span class="pre"><code class="sourceCode python">urllib.request.Request</code></span></a> getter and setter methods <span class="pre">`add_data`</span>, <span class="pre">`has_data`</span>, <span class="pre">`get_data`</span>, <span class="pre">`get_type`</span>, <span class="pre">`get_host`</span>, <span class="pre">`get_selector`</span>, <span class="pre">`set_proxy`</span>, <span class="pre">`get_origin_req_host`</span>, and <span class="pre">`is_unverifiable`</span> have been removed (use direct attribute access instead).

- Support for loading the deprecated <span class="pre">`TYPE_INT64`</span> has been removed from <a href="../library/marshal.html#module-marshal" class="reference internal" title="marshal: Convert Python objects to streams of bytes and back (with different constraints)."><span class="pre"><code class="sourceCode python">marshal</code></span></a>. (Contributed by Dan Riti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15480" class="reference external">bpo-15480</a>.)

- <a href="../library/inspect.html#inspect.Signature" class="reference internal" title="inspect.Signature"><span class="pre"><code class="sourceCode python">inspect.Signature</code></span></a>: positional-only parameters are now required to have a valid name.

- <a href="../reference/datamodel.html#object.__format__" class="reference internal" title="object.__format__"><span class="pre"><code class="sourceCode python"><span class="bu">object</span>.<span class="fu">__format__</span>()</code></span></a> no longer accepts non-empty format strings, it now raises a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> instead. Using a non-empty string has been deprecated since Python 3.2. This change has been made to prevent a situation where previously working (but incorrect) code would start failing if an object gained a \_\_format\_\_ method, which means that your code may now raise a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> if you are using an <span class="pre">`'s'`</span> format code with objects that do not have a \_\_format\_\_ method that handles it. See <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=7994" class="reference external">bpo-7994</a> for background.

- <span class="pre">`difflib.SequenceMatcher.isbjunk()`</span> and <span class="pre">`difflib.SequenceMatcher.isbpopular()`</span> were deprecated in 3.2, and have now been removed: use <span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`sm.bjunk`</span> and <span class="pre">`x`</span>` `<span class="pre">`in`</span>` `<span class="pre">`sm.bpopular`</span>, where *sm* is a <a href="../library/difflib.html#difflib.SequenceMatcher" class="reference internal" title="difflib.SequenceMatcher"><span class="pre"><code class="sourceCode python">SequenceMatcher</code></span></a> object (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13248" class="reference external">bpo-13248</a>).

</div>

<div id="code-cleanups" class="section">

### Code Cleanups<a href="#code-cleanups" class="headerlink" title="Link to this heading">¶</a>

- The unused and undocumented internal <span class="pre">`Scanner`</span> class has been removed from the <a href="../library/pydoc.html#module-pydoc" class="reference internal" title="pydoc: Documentation generator and online help system."><span class="pre"><code class="sourceCode python">pydoc</code></span></a> module.

- The private and effectively unused <span class="pre">`_gestalt`</span> module has been removed, along with the private <a href="../library/platform.html#module-platform" class="reference internal" title="platform: Retrieves as much platform identifying data as possible."><span class="pre"><code class="sourceCode python">platform</code></span></a> functions <span class="pre">`_mac_ver_lookup`</span>, <span class="pre">`_mac_ver_gstalt`</span>, and <span class="pre">`_bcd2str`</span>, which would only have ever been called on badly broken OSX systems (see <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18393" class="reference external">bpo-18393</a>).

- The hardcoded copies of certain <a href="../library/stat.html#module-stat" class="reference internal" title="stat: Utilities for interpreting the results of os.stat(), os.lstat() and os.fstat()."><span class="pre"><code class="sourceCode python">stat</code></span></a> constants that were included in the <a href="../library/tarfile.html#module-tarfile" class="reference internal" title="tarfile: Read and write tar-format archive files."><span class="pre"><code class="sourceCode python">tarfile</code></span></a> module namespace have been removed.

</div>

</div>

<div id="porting-to-python-3-4" class="section">

## Porting to Python 3.4<a href="#porting-to-python-3-4" class="headerlink" title="Link to this heading">¶</a>

This section lists previously described changes and other bugfixes that may require changes to your code.

<div id="changes-in-python-command-behavior" class="section">

### Changes in ‘python’ Command Behavior<a href="#changes-in-python-command-behavior" class="headerlink" title="Link to this heading">¶</a>

- In a posix shell, setting the <span id="index-55" class="target"></span><span class="pre">`PATH`</span> environment variable to an empty value is equivalent to not setting it at all. However, setting <span id="index-56" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONPATH" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONPATH</code></span></a> to an empty value was *not* equivalent to not setting it at all: setting <span id="index-57" class="target"></span><a href="../using/cmdline.html#envvar-PYTHONPATH" class="reference internal"><span class="pre"><code class="xref std std-envvar docutils literal notranslate">PYTHONPATH</code></span></a> to an empty value was equivalent to setting it to <span class="pre">`.`</span>, which leads to confusion when reasoning by analogy to how <span id="index-58" class="target"></span><span class="pre">`PATH`</span> works. The behavior now conforms to the posix convention for <span id="index-59" class="target"></span><span class="pre">`PATH`</span>.

- The \[X refs, Y blocks\] output of a debug (<span class="pre">`--with-pydebug`</span>) build of the CPython interpreter is now off by default. It can be re-enabled using the <span class="pre">`-X`</span>` `<span class="pre">`showrefcount`</span> option. (Contributed by Ezio Melotti in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17323" class="reference external">bpo-17323</a>.)

- The python command and most stdlib scripts (as well as <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a>) now output <span class="pre">`--version`</span> information to <span class="pre">`stdout`</span> instead of <span class="pre">`stderr`</span> (for issue list see <a href="#other-improvements-3-4" class="reference internal"><span class="std std-ref">Other Improvements</span></a> above).

</div>

<div id="changes-in-the-python-api" class="section">

### Changes in the Python API<a href="#changes-in-the-python-api" class="headerlink" title="Link to this heading">¶</a>

- The ABCs defined in <a href="../library/importlib.html#module-importlib.abc" class="reference internal" title="importlib.abc: Abstract base classes related to import"><span class="pre"><code class="sourceCode python">importlib.abc</code></span></a> now either raise the appropriate exception or return a default value instead of raising <a href="../library/exceptions.html#NotImplementedError" class="reference internal" title="NotImplementedError"><span class="pre"><code class="sourceCode python"><span class="pp">NotImplementedError</span></code></span></a> blindly. This will only affect code calling <a href="../library/functions.html#super" class="reference internal" title="super"><span class="pre"><code class="sourceCode python"><span class="bu">super</span>()</code></span></a> and falling through all the way to the ABCs. For compatibility, catch both <a href="../library/exceptions.html#NotImplementedError" class="reference internal" title="NotImplementedError"><span class="pre"><code class="sourceCode python"><span class="pp">NotImplementedError</span></code></span></a> or the appropriate exception as needed.

- The module type now initializes the <a href="../reference/datamodel.html#module.__package__" class="reference internal" title="module.__package__"><span class="pre"><code class="sourceCode python">__package__</code></span></a> and <a href="../reference/datamodel.html#module.__loader__" class="reference internal" title="module.__loader__"><span class="pre"><code class="sourceCode python">__loader__</code></span></a> attributes to <span class="pre">`None`</span> by default. To determine if these attributes were set in a backwards-compatible fashion, use e.g. <span class="pre">`getattr(module,`</span>` `<span class="pre">`'__loader__',`</span>` `<span class="pre">`None)`</span>` `<span class="pre">`is`</span>` `<span class="pre">`not`</span>` `<span class="pre">`None`</span>. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17115" class="reference external">bpo-17115</a>.)

- <span class="pre">`importlib.util.module_for_loader()`</span> now sets <span class="pre">`__loader__`</span> and <span class="pre">`__package__`</span> unconditionally to properly support reloading. If this is not desired then you will need to set these attributes manually. You can use <span class="pre">`importlib.util.module_to_load()`</span> for module management.

- Import now resets relevant attributes (e.g. <span class="pre">`__name__`</span>, <span class="pre">`__loader__`</span>, <span class="pre">`__package__`</span>, <span class="pre">`__file__`</span>, <span class="pre">`__cached__`</span>) unconditionally when reloading. Note that this restores a pre-3.3 behavior in that it means a module is re-found when re-loaded (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19413" class="reference external">bpo-19413</a>).

- Frozen packages no longer set <span class="pre">`__path__`</span> to a list containing the package name, they now set it to an empty list. The previous behavior could cause the import system to do the wrong thing on submodule imports if there was also a directory with the same name as the frozen package. The correct way to determine if a module is a package or not is to use <span class="pre">`hasattr(module,`</span>` `<span class="pre">`'__path__')`</span> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18065" class="reference external">bpo-18065</a>).

- Frozen modules no longer define a <span class="pre">`__file__`</span> attribute. It’s semantically incorrect for frozen modules to set the attribute as they are not loaded from any explicit location. If you must know that a module comes from frozen code then you can see if the module’s <span class="pre">`__spec__.location`</span> is set to <span class="pre">`'frozen'`</span>, check if the loader is a subclass of <a href="../library/importlib.html#importlib.machinery.FrozenImporter" class="reference internal" title="importlib.machinery.FrozenImporter"><span class="pre"><code class="sourceCode python">importlib.machinery.FrozenImporter</code></span></a>, or if Python 2 compatibility is necessary you can use <span class="pre">`imp.is_frozen()`</span>.

- <a href="../library/py_compile.html#py_compile.compile" class="reference internal" title="py_compile.compile"><span class="pre"><code class="sourceCode python">py_compile.<span class="bu">compile</span>()</code></span></a> now raises <a href="../library/exceptions.html#FileExistsError" class="reference internal" title="FileExistsError"><span class="pre"><code class="sourceCode python"><span class="pp">FileExistsError</span></code></span></a> if the file path it would write to is a symlink or a non-regular file. This is to act as a warning that import will overwrite those files with a regular file regardless of what type of file path they were originally.

- <a href="../library/importlib.html#importlib.abc.SourceLoader.get_source" class="reference internal" title="importlib.abc.SourceLoader.get_source"><span class="pre"><code class="sourceCode python">importlib.abc.SourceLoader.get_source()</code></span></a> no longer raises <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> when the source code being loaded triggers a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a> or <a href="../library/exceptions.html#UnicodeDecodeError" class="reference internal" title="UnicodeDecodeError"><span class="pre"><code class="sourceCode python"><span class="pp">UnicodeDecodeError</span></code></span></a>. As <a href="../library/exceptions.html#ImportError" class="reference internal" title="ImportError"><span class="pre"><code class="sourceCode python"><span class="pp">ImportError</span></code></span></a> is meant to be raised only when source code cannot be found but it should, it was felt to be over-reaching/overloading of that meaning when the source code is found but improperly structured. If you were catching ImportError before and wish to continue to ignore syntax or decoding issues, catch all three exceptions now.

- <a href="../library/functools.html#functools.update_wrapper" class="reference internal" title="functools.update_wrapper"><span class="pre"><code class="sourceCode python">functools.update_wrapper()</code></span></a> and <a href="../library/functools.html#functools.wraps" class="reference internal" title="functools.wraps"><span class="pre"><code class="sourceCode python">functools.wraps()</code></span></a> now correctly set the <span class="pre">`__wrapped__`</span> attribute to the function being wrapped, even if that function also had its <span class="pre">`__wrapped__`</span> attribute set. This means <span class="pre">`__wrapped__`</span> attributes now correctly link a stack of decorated functions rather than every <span class="pre">`__wrapped__`</span> attribute in the chain referring to the innermost function. Introspection libraries that assumed the previous behaviour was intentional can use <a href="../library/inspect.html#inspect.unwrap" class="reference internal" title="inspect.unwrap"><span class="pre"><code class="sourceCode python">inspect.unwrap()</code></span></a> to access the first function in the chain that has no <span class="pre">`__wrapped__`</span> attribute.

- <a href="../library/inspect.html#inspect.getfullargspec" class="reference internal" title="inspect.getfullargspec"><span class="pre"><code class="sourceCode python">inspect.getfullargspec()</code></span></a> has been reimplemented on top of <a href="../library/inspect.html#inspect.signature" class="reference internal" title="inspect.signature"><span class="pre"><code class="sourceCode python">inspect.signature()</code></span></a> and hence handles a much wider variety of callable objects than it did in the past. It is expected that additional builtin and extension module callables will gain signature metadata over the course of the Python 3.4 series. Code that assumes that <a href="../library/inspect.html#inspect.getfullargspec" class="reference internal" title="inspect.getfullargspec"><span class="pre"><code class="sourceCode python">inspect.getfullargspec()</code></span></a> will fail on non-Python callables may need to be adjusted accordingly.

- <a href="../library/importlib.html#importlib.machinery.PathFinder" class="reference internal" title="importlib.machinery.PathFinder"><span class="pre"><code class="sourceCode python">importlib.machinery.PathFinder</code></span></a> now passes on the current working directory to objects in <a href="../library/sys.html#sys.path_hooks" class="reference internal" title="sys.path_hooks"><span class="pre"><code class="sourceCode python">sys.path_hooks</code></span></a> for the empty string. This results in <a href="../library/sys.html#sys.path_importer_cache" class="reference internal" title="sys.path_importer_cache"><span class="pre"><code class="sourceCode python">sys.path_importer_cache</code></span></a> never containing <span class="pre">`''`</span>, thus iterating through <a href="../library/sys.html#sys.path_importer_cache" class="reference internal" title="sys.path_importer_cache"><span class="pre"><code class="sourceCode python">sys.path_importer_cache</code></span></a> based on <a href="../library/sys.html#sys.path" class="reference internal" title="sys.path"><span class="pre"><code class="sourceCode python">sys.path</code></span></a> will not find all keys. A module’s <span class="pre">`__file__`</span> when imported in the current working directory will also now have an absolute path, including when using <span class="pre">`-m`</span> with the interpreter (except for <span class="pre">`__main__.__file__`</span> when a script has been executed directly using a relative path) (Contributed by Brett Cannon in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18416" class="reference external">bpo-18416</a>). is specified on the command-line) (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18416" class="reference external">bpo-18416</a>).

- The removal of the *strict* argument to <a href="../library/http.client.html#http.client.HTTPConnection" class="reference internal" title="http.client.HTTPConnection"><span class="pre"><code class="sourceCode python">HTTPConnection</code></span></a> and <a href="../library/http.client.html#http.client.HTTPSConnection" class="reference internal" title="http.client.HTTPSConnection"><span class="pre"><code class="sourceCode python">HTTPSConnection</code></span></a> changes the meaning of the remaining arguments if you are specifying them positionally rather than by keyword. If you’ve been paying attention to deprecation warnings your code should already be specifying any additional arguments via keywords.

- Strings between <span class="pre">`from`</span>` `<span class="pre">`__future__`</span>` `<span class="pre">`import`</span>` `<span class="pre">`...`</span> statements now *always* raise a <a href="../library/exceptions.html#SyntaxError" class="reference internal" title="SyntaxError"><span class="pre"><code class="sourceCode python"><span class="pp">SyntaxError</span></code></span></a>. Previously if there was no leading docstring, an interstitial string would sometimes be ignored. This brings CPython into compliance with the language spec; Jython and PyPy already were. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17434" class="reference external">bpo-17434</a>).

- <a href="../library/ssl.html#ssl.SSLSocket.getpeercert" class="reference internal" title="ssl.SSLSocket.getpeercert"><span class="pre"><code class="sourceCode python">ssl.SSLSocket.getpeercert()</code></span></a> and <a href="../library/ssl.html#ssl.SSLSocket.do_handshake" class="reference internal" title="ssl.SSLSocket.do_handshake"><span class="pre"><code class="sourceCode python">ssl.SSLSocket.do_handshake()</code></span></a> now raise an <a href="../library/exceptions.html#OSError" class="reference internal" title="OSError"><span class="pre"><code class="sourceCode python"><span class="pp">OSError</span></code></span></a> with <span class="pre">`ENOTCONN`</span> when the <span class="pre">`SSLSocket`</span> is not connected, instead of the previous behavior of raising an <a href="../library/exceptions.html#AttributeError" class="reference internal" title="AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a>. In addition, <a href="../library/ssl.html#ssl.SSLSocket.getpeercert" class="reference internal" title="ssl.SSLSocket.getpeercert"><span class="pre"><code class="sourceCode python">getpeercert()</code></span></a> will raise a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if the handshake has not yet been done.

- <a href="../library/base64.html#base64.b32decode" class="reference internal" title="base64.b32decode"><span class="pre"><code class="sourceCode python">base64.b32decode()</code></span></a> now raises a <a href="../library/binascii.html#binascii.Error" class="reference internal" title="binascii.Error"><span class="pre"><code class="sourceCode python">binascii.Error</code></span></a> when the input string contains non-b32-alphabet characters, instead of a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>. This particular <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> was missed when the other <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>s were converted. (Contributed by Serhiy Storchaka in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18011" class="reference external">bpo-18011</a>.) Note: this change was also inadvertently applied in Python 3.3.3.

- The <span class="pre">`file`</span> attribute is now automatically closed when the creating <span class="pre">`cgi.FieldStorage`</span> instance is garbage collected. If you were pulling the file object out separately from the <span class="pre">`cgi.FieldStorage`</span> instance and not keeping the instance alive, then you should either store the entire <span class="pre">`cgi.FieldStorage`</span> instance or read the contents of the file before the <span class="pre">`cgi.FieldStorage`</span> instance is garbage collected.

- Calling <span class="pre">`read`</span> or <span class="pre">`write`</span> on a closed SSL socket now raises an informative <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> rather than the previous more mysterious <a href="../library/exceptions.html#AttributeError" class="reference internal" title="AttributeError"><span class="pre"><code class="sourceCode python"><span class="pp">AttributeError</span></code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=9177" class="reference external">bpo-9177</a>).

- <a href="../reference/datamodel.html#slice.indices" class="reference internal" title="slice.indices"><span class="pre"><code class="sourceCode python"><span class="bu">slice</span>.indices()</code></span></a> no longer produces an <a href="../library/exceptions.html#OverflowError" class="reference internal" title="OverflowError"><span class="pre"><code class="sourceCode python"><span class="pp">OverflowError</span></code></span></a> for huge values. As a consequence of this fix, <a href="../reference/datamodel.html#slice.indices" class="reference internal" title="slice.indices"><span class="pre"><code class="sourceCode python"><span class="bu">slice</span>.indices()</code></span></a> now raises a <a href="../library/exceptions.html#ValueError" class="reference internal" title="ValueError"><span class="pre"><code class="sourceCode python"><span class="pp">ValueError</span></code></span></a> if given a negative length; previously it returned nonsense values (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14794" class="reference external">bpo-14794</a>).

- The <a href="../library/functions.html#complex" class="reference internal" title="complex"><span class="pre"><code class="sourceCode python"><span class="bu">complex</span></code></span></a> constructor, unlike the <a href="../library/cmath.html#module-cmath" class="reference internal" title="cmath: Mathematical functions for complex numbers."><span class="pre"><code class="sourceCode python">cmath</code></span></a> functions, was incorrectly accepting <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> values if an object’s <span class="pre">`__complex__`</span> special method returned one. This now raises a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a>. (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16290" class="reference external">bpo-16290</a>.)

- The <a href="../library/functions.html#int" class="reference internal" title="int"><span class="pre"><code class="sourceCode python"><span class="bu">int</span></code></span></a> constructor in 3.2 and 3.3 erroneously accepts <a href="../library/functions.html#float" class="reference internal" title="float"><span class="pre"><code class="sourceCode python"><span class="bu">float</span></code></span></a> values for the *base* parameter. It is unlikely anyone was doing this, but if so, it will now raise a <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16772" class="reference external">bpo-16772</a>).

- Defaults for keyword-only arguments are now evaluated *after* defaults for regular keyword arguments, instead of before. Hopefully no one wrote any code that depends on the previous buggy behavior (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16967" class="reference external">bpo-16967</a>).

- Stale thread states are now cleared after <a href="../library/os.html#os.fork" class="reference internal" title="os.fork"><span class="pre"><code class="sourceCode python">fork()</code></span></a>. This may cause some system resources to be released that previously were incorrectly kept perpetually alive (for example, database connections kept in thread-local storage). (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17094" class="reference external">bpo-17094</a>.)

- Parameter names in <span class="pre">`__annotations__`</span> dicts are now mangled properly, similarly to <a href="../reference/datamodel.html#function.__kwdefaults__" class="reference internal" title="function.__kwdefaults__"><span class="pre"><code class="sourceCode python">__kwdefaults__</code></span></a>. (Contributed by Yury Selivanov in <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20625" class="reference external">bpo-20625</a>.)

- <a href="../library/hashlib.html#hashlib.hash.name" class="reference internal" title="hashlib.hash.name"><span class="pre"><code class="sourceCode python">hashlib.<span class="bu">hash</span>.name</code></span></a> now always returns the identifier in lower case. Previously some builtin hashes had uppercase names, but now that it is a formal public interface the naming has been made consistent (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=18532" class="reference external">bpo-18532</a>).

- Because <a href="../library/unittest.html#unittest.TestSuite" class="reference internal" title="unittest.TestSuite"><span class="pre"><code class="sourceCode python">unittest.TestSuite</code></span></a> now drops references to tests after they are run, test harnesses that reuse a <a href="../library/unittest.html#unittest.TestSuite" class="reference internal" title="unittest.TestSuite"><span class="pre"><code class="sourceCode python">TestSuite</code></span></a> to re-run a set of tests may fail. Test suites should not be re-used in this fashion since it means state is retained between test runs, breaking the test isolation that <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> is designed to provide. However, if the lack of isolation is considered acceptable, the old behavior can be restored by creating a <a href="../library/unittest.html#unittest.TestSuite" class="reference internal" title="unittest.TestSuite"><span class="pre"><code class="sourceCode python">TestSuite</code></span></a> subclass that defines a <span class="pre">`_removeTestAtIndex`</span> method that does nothing (see <a href="../library/unittest.html#unittest.TestSuite.__iter__" class="reference internal" title="unittest.TestSuite.__iter__"><span class="pre"><code class="sourceCode python">TestSuite.<span class="fu">__iter__</span>()</code></span></a>) (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=11798" class="reference external">bpo-11798</a>).

- <a href="../library/unittest.html#module-unittest" class="reference internal" title="unittest: Unit testing framework for Python."><span class="pre"><code class="sourceCode python">unittest</code></span></a> now uses <a href="../library/argparse.html#module-argparse" class="reference internal" title="argparse: Command-line option and argument parsing library."><span class="pre"><code class="sourceCode python">argparse</code></span></a> for command line parsing. There are certain invalid command forms that used to work that are no longer allowed; in theory this should not cause backward compatibility issues since the disallowed command forms didn’t make any sense and are unlikely to be in use.

- The <a href="../library/re.html#re.split" class="reference internal" title="re.split"><span class="pre"><code class="sourceCode python">re.split()</code></span></a>, <a href="../library/re.html#re.findall" class="reference internal" title="re.findall"><span class="pre"><code class="sourceCode python">re.findall()</code></span></a>, and <a href="../library/re.html#re.sub" class="reference internal" title="re.sub"><span class="pre"><code class="sourceCode python">re.sub()</code></span></a> functions, and the <a href="../library/re.html#re.Match.group" class="reference internal" title="re.Match.group"><span class="pre"><code class="sourceCode python">group()</code></span></a> and <a href="../library/re.html#re.Match.groups" class="reference internal" title="re.Match.groups"><span class="pre"><code class="sourceCode python">groups()</code></span></a> methods of <span class="pre">`match`</span> objects now always return a *bytes* object when the string to be matched is a <a href="../glossary.html#term-bytes-like-object" class="reference internal"><span class="xref std std-term">bytes-like object</span></a>. Previously the return type matched the input type, so if your code was depending on the return value being, say, a <span class="pre">`bytearray`</span>, you will need to change your code.

- <span class="pre">`audioop`</span> functions now raise an error immediately if passed string input, instead of failing randomly later on (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16685" class="reference external">bpo-16685</a>).

- The new *convert_charrefs* argument to <a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">HTMLParser</code></span></a> currently defaults to <span class="pre">`False`</span> for backward compatibility, but will eventually be changed to default to <span class="pre">`True`</span>. It is recommended that you add this keyword, with the appropriate value, to any <a href="../library/html.parser.html#html.parser.HTMLParser" class="reference internal" title="html.parser.HTMLParser"><span class="pre"><code class="sourceCode python">HTMLParser</code></span></a> calls in your code (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=13633" class="reference external">bpo-13633</a>).

- Since the *digestmod* argument to the <a href="../library/hmac.html#hmac.new" class="reference internal" title="hmac.new"><span class="pre"><code class="sourceCode python">hmac.new()</code></span></a> function will in the future have no default, all calls to <a href="../library/hmac.html#hmac.new" class="reference internal" title="hmac.new"><span class="pre"><code class="sourceCode python">hmac.new()</code></span></a> should be changed to explicitly specify a *digestmod* (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=17276" class="reference external">bpo-17276</a>).

- Calling <a href="../library/sysconfig.html#sysconfig.get_config_var" class="reference internal" title="sysconfig.get_config_var"><span class="pre"><code class="sourceCode python">sysconfig.get_config_var()</code></span></a> with the <span class="pre">`SO`</span> key, or looking <span class="pre">`SO`</span> up in the results of a call to <a href="../library/sysconfig.html#sysconfig.get_config_vars" class="reference internal" title="sysconfig.get_config_vars"><span class="pre"><code class="sourceCode python">sysconfig.get_config_vars()</code></span></a> is deprecated. This key should be replaced by <span class="pre">`EXT_SUFFIX`</span> or <span class="pre">`SHLIB_SUFFIX`</span>, depending on the context (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=19555" class="reference external">bpo-19555</a>).

- Any calls to <span class="pre">`open`</span> functions that specify <span class="pre">`U`</span> should be modified. <span class="pre">`U`</span> is ineffective in Python3 and will eventually raise an error if used. Depending on the function, the equivalent of its old Python2 behavior can be achieved using either a *newline* argument, or if necessary by wrapping the stream in <a href="../library/io.html#io.TextIOWrapper" class="reference internal" title="io.TextIOWrapper"><span class="pre"><code class="sourceCode python">TextIOWrapper</code></span></a> to use its *newline* argument (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=15204" class="reference external">bpo-15204</a>).

- If you use <span class="pre">`pyvenv`</span> in a script and desire that pip *not* be installed, you must add <span class="pre">`--without-pip`</span> to your command invocation.

- The default behavior of <a href="../library/json.html#json.dump" class="reference internal" title="json.dump"><span class="pre"><code class="sourceCode python">json.dump()</code></span></a> and <a href="../library/json.html#json.dumps" class="reference internal" title="json.dumps"><span class="pre"><code class="sourceCode python">json.dumps()</code></span></a> when an indent is specified has changed: it no longer produces trailing spaces after the item separating commas at the ends of lines. This will matter only if you have tests that are doing white-space-sensitive comparisons of such output (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16333" class="reference external">bpo-16333</a>).

- <a href="../library/doctest.html#module-doctest" class="reference internal" title="doctest: Test pieces of code within docstrings."><span class="pre"><code class="sourceCode python">doctest</code></span></a> now looks for doctests in extension module <span class="pre">`__doc__`</span> strings, so if your doctest test discovery includes extension modules that have things that look like doctests in them you may see test failures you’ve never seen before when running your tests (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=3158" class="reference external">bpo-3158</a>).

- The <a href="../library/collections.abc.html#module-collections.abc" class="reference internal" title="collections.abc: Abstract base classes for containers"><span class="pre"><code class="sourceCode python">collections.abc</code></span></a> module has been slightly refactored as part of the Python startup improvements. As a consequence of this, it is no longer the case that importing <a href="../library/collections.html#module-collections" class="reference internal" title="collections: Container datatypes"><span class="pre"><code class="sourceCode python">collections</code></span></a> automatically imports <a href="../library/collections.abc.html#module-collections.abc" class="reference internal" title="collections.abc: Abstract base classes for containers"><span class="pre"><code class="sourceCode python">collections.abc</code></span></a>. If your program depended on the (undocumented) implicit import, you will need to add an explicit <span class="pre">`import`</span>` `<span class="pre">`collections.abc`</span> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=20784" class="reference external">bpo-20784</a>).

</div>

<div id="changes-in-the-c-api" class="section">

### Changes in the C API<a href="#changes-in-the-c-api" class="headerlink" title="Link to this heading">¶</a>

- <a href="../c-api/veryhigh.html#c.PyEval_EvalFrameEx" class="reference internal" title="PyEval_EvalFrameEx"><span class="pre"><code class="sourceCode c">PyEval_EvalFrameEx<span class="op">()</span></code></span></a>, <a href="../c-api/object.html#c.PyObject_Repr" class="reference internal" title="PyObject_Repr"><span class="pre"><code class="sourceCode c">PyObject_Repr<span class="op">()</span></code></span></a>, and <a href="../c-api/object.html#c.PyObject_Str" class="reference internal" title="PyObject_Str"><span class="pre"><code class="sourceCode c">PyObject_Str<span class="op">()</span></code></span></a>, along with some other internal C APIs, now include a debugging assertion that ensures they are not used in situations where they may silently discard a currently active exception. In cases where discarding the active exception is expected and desired (for example, because it has already been saved locally with <a href="../c-api/exceptions.html#c.PyErr_Fetch" class="reference internal" title="PyErr_Fetch"><span class="pre"><code class="sourceCode c">PyErr_Fetch<span class="op">()</span></code></span></a> or is being deliberately replaced with a different exception), an explicit <a href="../c-api/exceptions.html#c.PyErr_Clear" class="reference internal" title="PyErr_Clear"><span class="pre"><code class="sourceCode c">PyErr_Clear<span class="op">()</span></code></span></a> call will be needed to avoid triggering the assertion when invoking these operations (directly or indirectly) and running against a version of Python that is compiled with assertions enabled.

- <a href="../c-api/exceptions.html#c.PyErr_SetImportError" class="reference internal" title="PyErr_SetImportError"><span class="pre"><code class="sourceCode c">PyErr_SetImportError<span class="op">()</span></code></span></a> now sets <a href="../library/exceptions.html#TypeError" class="reference internal" title="TypeError"><span class="pre"><code class="sourceCode python"><span class="pp">TypeError</span></code></span></a> when its **msg** argument is not set. Previously only <span class="pre">`NULL`</span> was returned with no exception set.

- The result of the <a href="../c-api/veryhigh.html#c.PyOS_ReadlineFunctionPointer" class="reference internal" title="PyOS_ReadlineFunctionPointer"><span class="pre"><code class="sourceCode c">PyOS_ReadlineFunctionPointer</code></span></a> callback must now be a string allocated by <a href="../c-api/memory.html#c.PyMem_RawMalloc" class="reference internal" title="PyMem_RawMalloc"><span class="pre"><code class="sourceCode c">PyMem_RawMalloc<span class="op">()</span></code></span></a> or <a href="../c-api/memory.html#c.PyMem_RawRealloc" class="reference internal" title="PyMem_RawRealloc"><span class="pre"><code class="sourceCode c">PyMem_RawRealloc<span class="op">()</span></code></span></a>, or <span class="pre">`NULL`</span> if an error occurred, instead of a string allocated by <a href="../c-api/memory.html#c.PyMem_Malloc" class="reference internal" title="PyMem_Malloc"><span class="pre"><code class="sourceCode c">PyMem_Malloc<span class="op">()</span></code></span></a> or <a href="../c-api/memory.html#c.PyMem_Realloc" class="reference internal" title="PyMem_Realloc"><span class="pre"><code class="sourceCode c">PyMem_Realloc<span class="op">()</span></code></span></a> (<a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=16742" class="reference external">bpo-16742</a>)

- <a href="../c-api/init.html#c.PyThread_set_key_value" class="reference internal" title="PyThread_set_key_value"><span class="pre"><code class="sourceCode c">PyThread_set_key_value<span class="op">()</span></code></span></a> now always set the value. In Python 3.3, the function did nothing if the key already exists (if the current value is a non-<span class="pre">`NULL`</span> pointer).

- The <span class="pre">`f_tstate`</span> (thread state) field of the <a href="../c-api/frame.html#c.PyFrameObject" class="reference internal" title="PyFrameObject"><span class="pre"><code class="sourceCode c">PyFrameObject</code></span></a> structure has been removed to fix a bug: see <a href="https://bugs.python.org/issue?@action=redirect&amp;bpo=14432" class="reference external">bpo-14432</a> for the rationale.

</div>

</div>

<div id="changed-in-3-4-3" class="section">

## Changed in 3.4.3<a href="#changed-in-3-4-3" class="headerlink" title="Link to this heading">¶</a>

<div id="pep-476-enabling-certificate-verification-by-default-for-stdlib-http-clients" class="section">

<span id="pep-476"></span>

### PEP 476: Enabling certificate verification by default for stdlib http clients<a href="#pep-476-enabling-certificate-verification-by-default-for-stdlib-http-clients" class="headerlink" title="Link to this heading">¶</a>

<a href="../library/http.client.html#module-http.client" class="reference internal" title="http.client: HTTP and HTTPS protocol client (requires sockets)."><span class="pre"><code class="sourceCode python">http.client</code></span></a> and modules which use it, such as <a href="../library/urllib.request.html#module-urllib.request" class="reference internal" title="urllib.request: Extensible library for opening URLs."><span class="pre"><code class="sourceCode python">urllib.request</code></span></a> and <a href="../library/xmlrpc.client.html#module-xmlrpc.client" class="reference internal" title="xmlrpc.client: XML-RPC client access."><span class="pre"><code class="sourceCode python">xmlrpc.client</code></span></a>, will now verify that the server presents a certificate which is signed by a CA in the platform trust store and whose hostname matches the hostname being requested by default, significantly improving security for many applications.

For applications which require the old previous behavior, they can pass an alternate context:

<div class="highlight-python3 notranslate">

<div class="highlight">

    import urllib.request
    import ssl

    # This disables all verification
    context = ssl._create_unverified_context()

    # This allows using a specific certificate for the host, which doesn't need
    # to be in the trust store
    context = ssl.create_default_context(cafile="/path/to/file.crt")

    urllib.request.urlopen("https://invalid-cert", context=context)

</div>

</div>

</div>

</div>

</div>

<div class="clearer">

</div>

</div>
