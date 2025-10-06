::::: {#main-content .SiteContent .SiteContent--default role="main"}
# Go 1.14 Release Notes

::: {#nav .TOC}
:::

::: markdown
## Introduction to Go 1.14 {#introduction}

The latest Go release, version 1.14, arrives six months after [Go
1.13](go1.13). Most of its changes are in the implementation of the
toolchain, runtime, and libraries. As always, the release maintains the
Go 1 [promise of compatibility](/doc/go1compat.html). We expect almost
all Go programs to continue to compile and run as before.

Module support in the `go` command is now ready for production use, and
we encourage all users to [migrate to Go modules for dependency
management](/blog/migrating-to-go-modules). If you are unable to migrate
due to a problem in the Go toolchain, please ensure that the problem has
an [open issue](/issue?q=is%3Aissue+is%3Aopen+label%3Amodules) filed.
(If the issue is not on the `Go1.15` milestone, please let us know why
it prevents you from migrating so that we can prioritize it
appropriately.)

## Changes to the language {#language}

Per the [overlapping interfaces
proposal](https://github.com/golang/proposal/blob/master/design/6977-overlapping-interfaces.md){rel="noreferrer"
target="_blank"}, Go 1.14 now permits embedding of interfaces with
overlapping method sets: methods from an embedded interface may have the
same names and identical signatures as methods already present in the
(embedding) interface. This solves problems that typically (but not
exclusively) occur with diamond-shaped embedding graphs. Explicitly
declared methods in an interface must remain
[unique](https://tip.golang.org/ref/spec#Uniqueness_of_identifiers){rel="noreferrer"
target="_blank"}, as before.

## Ports

### Darwin

Go 1.14 is the last release that will run on macOS 10.11 El Capitan. Go
1.15 will require macOS 10.12 Sierra or later.

Go 1.14 is the last Go release to support 32-bit binaries on macOS (the
`darwin/386` port). They are no longer supported by macOS, starting with
macOS 10.15 (Catalina). Go continues to support the 64-bit
`darwin/amd64` port.

Go 1.14 will likely be the last Go release to support 32-bit binaries on
iOS, iPadOS, watchOS, and tvOS (the `darwin/arm` port). Go continues to
support the 64-bit `darwin/arm64` port.

### Windows

Go binaries on Windows now have [DEP (Data Execution
Prevention)](https://docs.microsoft.com/en-us/windows/win32/memory/data-execution-prevention){rel="noreferrer"
target="_blank"} enabled.

On Windows, creating a file via [`os.OpenFile`](/pkg/os#CreateFile) with
the [`os.O_CREATE`](/pkg/os/#O_CREATE) flag, or via
[`syscall.Open`](/pkg/syscall#Open) with the
[`syscall.O_CREAT`](/pkg/syscall#O_CREAT) flag, will now create the file
as read-only if the bit `0o200` (owner write permission) is not set in
the permission argument. This makes the behavior on Windows more like
that on Unix systems.

### WebAssembly {#wasm}

JavaScript values referenced from Go via `js.Value` objects can now be
garbage collected.

`js.Value` values can no longer be compared using the `==` operator, and
instead must be compared using their `Equal` method.

`js.Value` now has `IsUndefined`, `IsNull`, and `IsNaN` methods.

### RISC-V {#riscv}

Go 1.14 contains experimental support for 64-bit RISC-V on Linux
(`GOOS=linux`, `GOARCH=riscv64`). Be aware that performance, assembly
syntax stability, and possibly correctness are a work in progress.

### FreeBSD

Go now supports the 64-bit ARM architecture on FreeBSD 12.0 or later
(the `freebsd/arm64` port).

### Native Client (NaCl) {#nacl}

As [announced](go1.13#ports) in the Go 1.13 release notes, Go 1.14 drops
support for the Native Client platform (`GOOS=nacl`).

### Illumos

The runtime now respects zone CPU caps (the `zone.cpu-cap` resource
control) for `runtime.NumCPU` and the default value of `GOMAXPROCS`.

## Tools

### Go command

#### Vendoring {#vendor}

When the main module contains a top-level `vendor` directory and its
`go.mod` file specifies `go` `1.14` or higher, the `go` command now
defaults to `-mod=vendor` for operations that accept that flag. A new
value for that flag, `-mod=mod`, causes the `go` command to instead load
modules from the module cache (as when no `vendor` directory is
present).

When `-mod=vendor` is set (explicitly or by default), the `go` command
now verifies that the main module's `vendor/modules.txt` file is
consistent with its `go.mod` file.

`go` `list` `-m` no longer silently omits transitive dependencies that
do not provide packages in the `vendor` directory. It now fails
explicitly if `-mod=vendor` is set and information is requested for a
module not mentioned in `vendor/modules.txt`.

#### Flags {#go-flags}

The `go` `get` command no longer accepts the `-mod` flag. Previously,
the flag's setting either [was ignored](/issue/30345) or [caused the
build to fail](/issue/32502).

`-mod=readonly` is now set by default when the `go.mod` file is
read-only and no top-level `vendor` directory is present.

`-modcacherw` is a new flag that instructs the `go` command to leave
newly-created directories in the module cache at their default
permissions rather than making them read-only. The use of this flag
makes it more likely that tests or other tools will accidentally add
files not included in the module's verified checksum. However, it allows
the use of `rm` `-rf` (instead of `go` `clean` `-modcache`) to remove
the module cache.

`-modfile=file` is a new flag that instructs the `go` command to read
(and possibly write) an alternate `go.mod` file instead of the one in
the module root directory. A file named `go.mod` must still be present
in order to determine the module root directory, but it is not accessed.
When `-modfile` is specified, an alternate `go.sum` file is also used:
its path is derived from the `-modfile` flag by trimming the `.mod`
extension and appending `.sum`.

#### Environment variables {#go-env-vars}

`GOINSECURE` is a new environment variable that instructs the `go`
command to not require an HTTPS connection, and to skip certificate
validation, when fetching certain modules directly from their origins.
Like the existing `GOPRIVATE` variable, the value of `GOINSECURE` is a
comma-separated list of glob patterns.

#### Commands outside modules

When module-aware mode is enabled explicitly (by setting
`GO111MODULE=on`), most module commands have more limited functionality
if no `go.mod` file is present. For example, `go` `build`, `go` `run`,
and other build commands can only build packages in the standard library
and packages specified as `.go` files on the command line.

Previously, the `go` command would resolve each package path to the
latest version of a module but would not record the module path or
version. This resulted in [slow, non-reproducible builds](/issue/32027).

`go` `get` continues to work as before, as do `go` `mod` `download` and
`go` `list` `-m` with explicit versions.

#### `+incompatible` versions

If the latest version of a module contains a `go.mod` file, `go` `get`
will no longer upgrade to an
[incompatible](/cmd/go/#hdr-Module_compatibility_and_semantic_versioning)
major version of that module unless such a version is requested
explicitly or is already required. `go` `list` also omits incompatible
major versions for such a module when fetching directly from version
control, but may include them if reported by a proxy.

#### `go.mod` file maintenance {#go.mod}

`go` commands other than `go` `mod` `tidy` no longer remove a `require`
directive that specifies a version of an indirect dependency that is
already implied by other (transitive) dependencies of the main module.

`go` commands other than `go` `mod` `tidy` no longer edit the `go.mod`
file if the changes are only cosmetic.

When `-mod=readonly` is set, `go` commands will no longer fail due to a
missing `go` directive or an erroneous `// indirect` comment.

#### Module downloading

The `go` command now supports Subversion repositories in module mode.

The `go` command now includes snippets of plain-text error messages from
module proxies and other HTTP servers. An error message will only be
shown if it is valid UTF-8 and consists of only graphic characters and
spaces.

#### Testing {#go-test}

`go test -v` now streams `t.Log` output as it happens, rather than at
the end of all tests.

## Runtime

This release improves the performance of most uses of `defer` to incur
almost zero overhead compared to calling the deferred function directly.
As a result, `defer` can now be used in performance-critical code
without overhead concerns.

Goroutines are now asynchronously preemptible. As a result, loops
without function calls no longer potentially deadlock the scheduler or
significantly delay garbage collection. This is supported on all
platforms except `windows/arm`, `darwin/arm`, `js/wasm`, and `plan9/*`.

A consequence of the implementation of preemption is that on Unix
systems, including Linux and macOS systems, programs built with Go 1.14
will receive more signals than programs built with earlier releases.
This means that programs that use packages like
[`syscall`](/pkg/syscall/) or
[`golang.org/x/sys/unix`](https://godoc.org/golang.org/x/sys/unix){rel="noreferrer"
target="_blank"} will see more slow system calls fail with `EINTR`
errors. Those programs will have to handle those errors in some way,
most likely looping to try the system call again. For more information
about this see
[`man 7 signal`](https://man7.org/linux/man-pages/man7/signal.7.html){rel="noreferrer"
target="_blank"} for Linux systems or similar documentation for other
systems.

The page allocator is more efficient and incurs significantly less lock
contention at high values of `GOMAXPROCS`. This is most noticeable as
lower latency and higher throughput for large allocations being done in
parallel and at a high rate.

Internal timers, used by [`time.After`](/pkg/time/#After),
[`time.Tick`](/pkg/time/#Tick),
[`net.Conn.SetDeadline`](/pkg/net/#Conn), and friends, are more
efficient, with less lock contention and fewer context switches. This is
a performance improvement that should not cause any user visible
changes.

## Compiler

This release adds `-d=checkptr` as a compile-time option for adding
instrumentation to check that Go code is following `unsafe.Pointer`
safety rules dynamically. This option is enabled by default (except on
Windows) with the `-race` or `-msan` flags, and can be disabled with
`-gcflags=all=-d=checkptr=0`. Specifically, `-d=checkptr` checks the
following:

1.  When converting `unsafe.Pointer` to `*T`, the resulting pointer must
    be aligned appropriately for `T`.
2.  If the result of pointer arithmetic points into a Go heap object,
    one of the `unsafe.Pointer`-typed operands must point into the same
    object.

Using `-d=checkptr` is not currently recommended on Windows because it
causes false alerts in the standard library.

The compiler can now emit machine-readable logs of key optimizations
using the `-json` flag, including inlining, escape analysis,
bounds-check elimination, and nil-check elimination.

Detailed escape analysis diagnostics (`-m=2`) now work again. This had
been dropped from the new escape analysis implementation in the previous
release.

All Go symbols in macOS binaries now begin with an underscore, following
platform conventions.

This release includes experimental support for compiler-inserted
coverage instrumentation for fuzzing. See [issue 14565](/issue/14565)
for more details. This API may change in future releases.

Bounds check elimination now uses information from slice creation and
can eliminate checks for indexes with types smaller than `int`.

## Standard library {#library}

### New byte sequence hashing package {#hash_maphash}

Go 1.14 includes a new package, [`hash/maphash`](/pkg/hash/maphash/),
which provides hash functions on byte sequences. These hash functions
are intended to be used to implement hash tables or other data
structures that need to map arbitrary strings or byte sequences to a
uniform distribution on unsigned 64-bit integers.

The hash functions are collision-resistant but not cryptographically
secure.

The hash value of a given byte sequence is consistent within a single
process, but will be different in different processes.

### Minor changes to the library {#minor_library_changes}

As always, there are various minor changes and updates to the library,
made with the Go 1 [promise of compatibility](/doc/go1compat) in mind.

#### [crypto/tls](/pkg/crypto/tls/) {#cryptotlspkgcryptotls}

Support for SSL version 3.0 (SSLv3) has been removed. Note that SSLv3 is
the [cryptographically
broken](https://tools.ietf.org/html/rfc7568){rel="noreferrer"
target="_blank"} protocol predating TLS.

TLS 1.3 can't be disabled via the `GODEBUG` environment variable
anymore. Use the
[`Config.MaxVersion`](/pkg/crypto/tls/#Config.MaxVersion) field to
configure TLS versions.

When multiple certificate chains are provided through the
[`Config.Certificates`](/pkg/crypto/tls/#Config.Certificates) field, the
first one compatible with the peer is now automatically selected. This
allows for example providing an ECDSA and an RSA certificate, and
letting the package automatically select the best one. Note that the
performance of this selection is going to be poor unless the
[`Certificate.Leaf`](/pkg/crypto/tls/#Certificate.Leaf) field is set.
The
[`Config.NameToCertificate`](/pkg/crypto/tls/#Config.NameToCertificate)
field, which only supports associating a single certificate with a give
name, is now deprecated and should be left as `nil`. Similarly the
[`Config.BuildNameToCertificate`](/pkg/crypto/tls/#Config.BuildNameToCertificate)
method, which builds the `NameToCertificate` field from the leaf
certificates, is now deprecated and should not be called.

The new [`CipherSuites`](/pkg/crypto/tls/#CipherSuites) and
[`InsecureCipherSuites`](/pkg/crypto/tls/#InsecureCipherSuites)
functions return a list of currently implemented cipher suites. The new
[`CipherSuiteName`](/pkg/crypto/tls/#CipherSuiteName) function returns a
name for a cipher suite ID.

The new
[`(*ClientHelloInfo).SupportsCertificate`](/pkg/crypto/tls/#ClientHelloInfo.SupportsCertificate)
and
[`(*CertificateRequestInfo).SupportsCertificate`](/pkg/crypto/tls/#CertificateRequestInfo.SupportsCertificate)
methods expose whether a peer supports a certain certificate.

The `tls` package no longer supports the legacy Next Protocol
Negotiation (NPN) extension and now only supports ALPN. In previous
releases it supported both. There are no API changes and applications
should function identically as before. Most other clients and servers
have already removed NPN support in favor of the standardized ALPN.

RSA-PSS signatures are now used when supported in TLS 1.2 handshakes.
This won't affect most applications, but custom
[`Certificate.PrivateKey`](/pkg/crypto/tls/#Certificate.PrivateKey)
implementations that don't support RSA-PSS signatures will need to use
the new
[`Certificate.SupportedSignatureAlgorithms`](/pkg/crypto/tls/#Certificate.SupportedSignatureAlgorithms)
field to disable them.

[`Config.Certificates`](/pkg/crypto/tls/#Config.Certificates) and
[`Config.GetCertificate`](/pkg/crypto/tls/#Config.GetCertificate) can
now both be nil if
[`Config.GetConfigForClient`](/pkg/crypto/tls/#Config.GetConfigForClient)
is set. If the callbacks return neither certificates nor an error, the
`unrecognized_name` is now sent.

The new
[`CertificateRequestInfo.Version`](/pkg/crypto/tls/#CertificateRequestInfo.Version)
field provides the TLS version to client certificates callbacks.

The new `TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256` and
`TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256` constants use the final
names for the cipher suites previously referred to as
`TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305` and
`TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305`.

#### [crypto/x509](/pkg/crypto/x509/) {#cryptox509pkgcryptox509}

[`Certificate.CreateCRL`](/pkg/crypto/x509/#Certificate.CreateCRL) now
supports Ed25519 issuers.

#### [debug/dwarf](/pkg/debug/dwarf/) {#debugdwarfpkgdebugdwarf}

The `debug/dwarf` package now supports reading DWARF version 5.

The new method [`(*Data).AddSection`](/pkg/debug/dwarf/#Data.AddSection)
supports adding arbitrary new DWARF sections from the input file to the
DWARF `Data`.

The new method
[`(*Reader).ByteOrder`](/pkg/debug/dwarf/#Reader.ByteOrder) returns the
byte order of the current compilation unit. This may be used to
interpret attributes that are encoded in the native ordering, such as
location descriptions.

The new method
[`(*LineReader).Files`](/pkg/debug/dwarf/#LineReader.Files) returns the
file name table from a line reader. This may be used to interpret the
value of DWARF attributes such as `AttrDeclFile`.

#### [encoding/asn1](/pkg/encoding/asn1/) {#encodingasn1pkgencodingasn1}

[`Unmarshal`](/pkg/encoding/asn1/#Unmarshal) now supports ASN.1 string
type BMPString, represented by the new
[`TagBMPString`](/pkg/encoding/asn1/#TagBMPString) constant.

#### [encoding/json](/pkg/encoding/json/) {#encodingjsonpkgencodingjson}

The [`Decoder`](/pkg/encoding/json/#Decoder) type supports a new method
[`InputOffset`](/pkg/encoding/json/#Decoder.InputOffset) that returns
the input stream byte offset of the current decoder position.

[`Compact`](/pkg/encoding/json/#Compact) no longer escapes the `U+2028`
and `U+2029` characters, which was never a documented feature. For
proper escaping, see [`HTMLEscape`](/pkg/encoding/json/#HTMLEscape).

[`Number`](/pkg/encoding/json/#Number) no longer accepts invalid
numbers, to follow the documented behavior more closely. If a program
needs to accept invalid numbers like the empty string, consider wrapping
the type with [`Unmarshaler`](/pkg/encoding/json/#Unmarshaler).

[`Unmarshal`](/pkg/encoding/json/#Unmarshal) can now support map keys
with string underlying type which implement
[`encoding.TextUnmarshaler`](/pkg/encoding/#TextUnmarshaler).

#### [go/build](/pkg/go/build/) {#gobuildpkggobuild}

The [`Context`](/pkg/go/build/#Context) type has a new field `Dir` which
may be used to set the working directory for the build. The default is
the current directory of the running process. In module mode, this is
used to locate the main module.

#### [go/doc](/pkg/go/doc/) {#godocpkggodoc}

The new function [`NewFromFiles`](/pkg/go/doc/#NewFromFiles) computes
package documentation from a list of `*ast.File`'s and associates
examples with the appropriate package elements. The new information is
available in a new `Examples` field in the
[`Package`](/pkg/go/doc/#Package), [`Type`](/pkg/go/doc/#Type), and
[`Func`](/pkg/go/doc/#Func) types, and a new
[`Suffix`](/pkg/go/doc/#Example.Suffix) field in the
[`Example`](/pkg/go/doc/#Example) type.

#### [io/ioutil](/pkg/io/ioutil/) {#ioioutilpkgioioutil}

[`TempDir`](/pkg/io/ioutil/#TempDir) can now create directories whose
names have predictable prefixes and suffixes. As with
[`TempFile`](/pkg/io/ioutil/#TempFile), if the pattern contains a '\*',
the random string replaces the last '\*'.

#### [log](/pkg/log/) {#logpkglog}

The new
[`Lmsgprefix`](https://tip.golang.org/pkg/log/#pkg-constants){rel="noreferrer"
target="_blank"} flag may be used to tell the logging functions to emit
the optional output prefix immediately before the log message rather
than at the start of the line.

#### [math](/pkg/math/) {#mathpkgmath}

The new [`FMA`](/pkg/math/#FMA) function computes `x*y+z` in floating
point with no intermediate rounding of the `x*y` computation. Several
architectures implement this computation using dedicated hardware
instructions for additional performance.

#### [math/big](/pkg/math/big/) {#mathbigpkgmathbig}

The [`GCD`](/pkg/math/big/#Int.GCD) method now allows the inputs `a` and
`b` to be zero or negative.

#### [math/bits](/pkg/math/bits/) {#mathbitspkgmathbits}

The new functions [`Rem`](/pkg/math/bits/#Rem),
[`Rem32`](/pkg/math/bits/#Rem32), and [`Rem64`](/pkg/math/bits/#Rem64)
support computing a remainder even when the quotient overflows.

#### [mime](/pkg/mime/) {#mimepkgmime}

The default type of `.js` and `.mjs` files is now `text/javascript`
rather than `application/javascript`. This is in accordance with [an
IETF
draft](https://datatracker.ietf.org/doc/draft-ietf-dispatch-javascript-mjs/){rel="noreferrer"
target="_blank"} that treats `application/javascript` as obsolete.

#### [mime/multipart](/pkg/mime/multipart/) {#mimemultipartpkgmimemultipart}

The new [`Reader`](/pkg/mime/multipart/#Reader) method
[`NextRawPart`](/pkg/mime/multipart/#Reader.NextRawPart) supports
fetching the next MIME part without transparently decoding
`quoted-printable` data.

#### [net/http](/pkg/net/http/) {#nethttppkgnethttp}

The new [`Header`](/pkg/net/http/#Header) method
[`Values`](/pkg/net/http/#Header.Values) can be used to fetch all values
associated with a canonicalized key.

The new [`Transport`](/pkg/net/http/#Transport) field
[`DialTLSContext`](/pkg/net/http/#Transport.DialTLSContext) can be used
to specify an optional dial function for creating TLS connections for
non-proxied HTTPS requests. This new field can be used instead of
[`DialTLS`](/pkg/net/http/#Transport.DialTLS), which is now considered
deprecated; `DialTLS` will continue to work, but new code should use
`DialTLSContext`, which allows the transport to cancel dials as soon as
they are no longer needed.

On Windows, [`ServeFile`](/pkg/net/http/#ServeFile) now correctly serves
files larger than 2GB.

#### [net/http/httptest](/pkg/net/http/httptest/) {#nethttphttptestpkgnethttphttptest}

The new [`Server`](/pkg/net/http/httptest/#Server) field
[`EnableHTTP2`](/pkg/net/http/httptest/#Server.EnableHTTP2) supports
enabling HTTP/2 on the test server.

#### [net/textproto](/pkg/net/textproto/) {#nettextprotopkgnettextproto}

The new [`MIMEHeader`](/pkg/net/textproto/#MIMEHeader) method
[`Values`](/pkg/net/textproto/#MIMEHeader.Values) can be used to fetch
all values associated with a canonicalized key.

#### [net/url](/pkg/net/url/) {#neturlpkgneturl}

When parsing of a URL fails (for example by
[`Parse`](/pkg/net/url/#Parse) or
[`ParseRequestURI`](/pkg/net/url/#ParseRequestURI)), the resulting
[`Error`](/pkg/net/url/#Error.Error) message will now quote the
unparsable URL. This provides clearer structure and consistency with
other parsing errors.

#### [os/signal](/pkg/os/signal/) {#ossignalpkgossignal}

On Windows, the `CTRL_CLOSE_EVENT`, `CTRL_LOGOFF_EVENT`, and
`CTRL_SHUTDOWN_EVENT` events now generate a `syscall.SIGTERM` signal,
similar to how Control-C and Control-Break generate a `syscall.SIGINT`
signal.

#### [plugin](/pkg/plugin/) {#pluginpkgplugin}

The `plugin` package now supports `freebsd/amd64`.

#### [reflect](/pkg/reflect/) {#reflectpkgreflect}

[`StructOf`](/pkg/reflect#StructOf) now supports creating struct types
with unexported fields, by setting the `PkgPath` field in a
`StructField` element.

#### [runtime](/pkg/runtime/) {#runtimepkgruntime}

`runtime.Goexit` can no longer be aborted by a recursive
`panic`/`recover`.

On macOS, `SIGPIPE` is no longer forwarded to signal handlers installed
before the Go runtime is initialized. This is necessary because macOS
delivers `SIGPIPE` [to the main thread](/issue/33384) rather than the
thread writing to the closed pipe.

#### [runtime/pprof](/pkg/runtime/pprof/) {#runtimepprofpkgruntimepprof}

The generated profile no longer includes the pseudo-PCs used for inline
marks. Symbol information of inlined functions is encoded in [the
format](https://github.com/google/pprof/blob/5e96527/proto/profile.proto#L177-L184){rel="noreferrer"
target="_blank"} the pprof tool expects. This is a fix for the
regression introduced during recent releases.

#### [strconv](/pkg/strconv/) {#strconvpkgstrconv}

The [`NumError`](/pkg/strconv/#NumError) type now has an
[`Unwrap`](/pkg/strconv/#NumError.Unwrap) method that may be used to
retrieve the reason that a conversion failed. This supports using
`NumError` values with [`errors.Is`](/pkg/errors/#Is) to see if the
underlying error is [`strconv.ErrRange`](/pkg/strconv/#pkg-variables) or
[`strconv.ErrSyntax`](/pkg/strconv/#pkg-variables).

#### [sync](/pkg/sync/) {#syncpkgsync}

Unlocking a highly contended `Mutex` now directly yields the CPU to the
next goroutine waiting for that `Mutex`. This significantly improves the
performance of highly contended mutexes on high CPU count machines.

#### [testing](/pkg/testing/) {#testingpkgtesting}

The testing package now supports cleanup functions, called after a test
or benchmark has finished, by calling
[`T.Cleanup`](/pkg/testing#T.Cleanup) or
[`B.Cleanup`](/pkg/testing#B.Cleanup) respectively.

#### [text/template](/pkg/text/template/) {#texttemplatepkgtexttemplate}

The text/template package now correctly reports errors when a
parenthesized argument is used as a function. This most commonly shows
up in erroneous cases like `{{if (eq .F "a") or (eq .F "b")}}`. This
should be written as `{{if or (eq .F "a") (eq .F "b")}}`. The erroneous
case never worked as expected, and will now be reported with an error
`can't give argument to non-function`.

[`JSEscape`](/pkg/text/template/#JSEscape) now escapes the `&` and `=`
characters to mitigate the impact of its output being misused in HTML
contexts.

#### [unicode](/pkg/unicode/) {#unicodepkgunicode}

The [`unicode`](/pkg/unicode/) package and associated support throughout
the system has been upgraded from Unicode 11.0 to [Unicode
12.0](https://www.unicode.org/versions/Unicode12.0.0/){rel="noreferrer"
target="_blank"}, which adds 554 new characters, including four new
scripts, and 61 new emoji.
:::
:::::
