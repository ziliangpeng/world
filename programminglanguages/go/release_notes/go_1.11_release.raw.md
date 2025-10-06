::::: {#main-content .SiteContent .SiteContent--default role="main"}
# Go 1.11 Release Notes

::: {#nav .TOC}
:::

::: markdown
## Introduction to Go 1.11 {#introduction}

The latest Go release, version 1.11, arrives six months after [Go
1.10](go1.10). Most of its changes are in the implementation of the
toolchain, runtime, and libraries. As always, the release maintains the
Go 1 [promise of compatibility](/doc/go1compat.html). We expect almost
all Go programs to continue to compile and run as before.

## Changes to the language {#language}

There are no changes to the language specification.

## Ports

As [announced in the Go 1.10 release notes](go1.10#ports), Go 1.11 now
requires OpenBSD 6.2 or later, macOS 10.10 Yosemite or later, or Windows
7 or later; support for previous versions of these operating systems has
been removed.

Go 1.11 supports the upcoming OpenBSD 6.4 release. Due to changes in the
OpenBSD kernel, older versions of Go will not work on OpenBSD 6.4.

There are [known issues](/issue/25206) with NetBSD on i386 hardware.

The race detector is now supported on `linux/ppc64le` and, to a lesser
extent, on `netbsd/amd64`. The NetBSD race detector support has [known
issues](/issue/26403).

The memory sanitizer (`-msan`) is now supported on `linux/arm64`.

The build modes `c-shared` and `c-archive` are now supported on
`freebsd/amd64`.

On 64-bit MIPS systems, the new environment variable settings
`GOMIPS64=hardfloat` (the default) and `GOMIPS64=softfloat` select
whether to use hardware instructions or software emulation for
floating-point computations. For 32-bit systems, the environment
variable is still `GOMIPS`, as [added in Go 1.10](go1.10#mips).

On soft-float ARM systems (`GOARM=5`), Go now uses a more efficient
software floating point interface. This is transparent to Go code, but
ARM assembly that uses floating-point instructions not guarded on GOARM
will break and must be ported to the [new interface](/cl/107475).

Go 1.11 on ARMv7 no longer requires a Linux kernel configured with
`KUSER_HELPERS`. This setting is enabled in default kernel
configurations, but is sometimes disabled in stripped-down
configurations.

### WebAssembly {#wasm}

Go 1.11 adds an experimental port to
[WebAssembly](https://webassembly.org){rel="noreferrer" target="_blank"}
(`js/wasm`).

Go programs currently compile to one WebAssembly module that includes
the Go runtime for goroutine scheduling, garbage collection, maps, etc.
As a result, the resulting size is at minimum around 2 MB, or 500 KB
compressed. Go programs can call into JavaScript using the new
experimental [`syscall/js`](/pkg/syscall/js/) package. Binary size and
interop with other languages has not yet been a priority but may be
addressed in future releases.

As a result of the addition of the new `GOOS` value "`js`" and `GOARCH`
value "`wasm`", Go files named `*_js.go` or `*_wasm.go` will now be
[ignored by Go tools](/pkg/go/build/#hdr-Build_Constraints) except when
those GOOS/GOARCH values are being used. If you have existing filenames
matching those patterns, you will need to rename them.

More information can be found on the [WebAssembly wiki
page](/wiki/WebAssembly).

### RISC-V GOARCH values reserved {#riscv}

The main Go compiler does not yet support the RISC-V architecture but
we've reserved the `GOARCH` values "`riscv`" and "`riscv64`", as used by
Gccgo, which does support RISC-V. This means that Go files named
`*_riscv.go` will now also be [ignored by Go
tools](/pkg/go/build/#hdr-Build_Constraints) except when those
GOOS/GOARCH values are being used.

## Tools

### Modules, package versioning, and dependency management {#modules}

Go 1.11 adds preliminary support for a [new concept called
"modules,"](/cmd/go/#hdr-Modules__module_versions__and_more) an
alternative to GOPATH with integrated support for versioning and package
distribution. Using modules, developers are no longer confined to
working inside GOPATH, version dependency information is explicit yet
lightweight, and builds are more reliable and reproducible.

Module support is considered experimental. Details are likely to change
in response to feedback from Go 1.11 users, and we have more tools
planned. Although the details of module support may change, projects
that convert to modules using Go 1.11 will continue to work with Go 1.12
and later. If you encounter bugs using modules, please [file
issues](/issue/new) so we can fix them. For more information, see the
[`go` command
documentation](/cmd/go#hdr-Modules__module_versions__and_more).

### Import path restriction {#importpath}

Because Go module support assigns special meaning to the `@` symbol in
command line operations, the `go` command now disallows the use of
import paths containing `@` symbols. Such import paths were never
allowed by `go` `get`, so this restriction can only affect users
building custom GOPATH trees by other means.

### Package loading {#gopackages}

The new package
[`golang.org/x/tools/go/packages`](https://godoc.org/golang.org/x/tools/go/packages){rel="noreferrer"
target="_blank"} provides a simple API for locating and loading packages
of Go source code. Although not yet part of the standard library, for
many tasks it effectively replaces the [`go/build`](/pkg/go/build)
package, whose API is unable to fully support modules. Because it runs
an external query command such as
[`go list`](/cmd/go/#hdr-List_packages) to obtain information about Go
packages, it enables the construction of analysis tools that work
equally well with alternative build systems such as
[Bazel](https://bazel.build){rel="noreferrer" target="_blank"} and
[Buck](https://buckbuild.com){rel="noreferrer" target="_blank"}.

### Build cache requirement {#gocache}

Go 1.11 will be the last release to support setting the environment
variable `GOCACHE=off` to disable the [build
cache](/cmd/go/#hdr-Build_and_test_caching), introduced in Go 1.10.
Starting in Go 1.12, the build cache will be required, as a step toward
eliminating `$GOPATH/pkg`. The module and package loading support
described above already require that the build cache be enabled. If you
have disabled the build cache to avoid problems you encountered, please
[file an issue](/issue/new) to let us know about them.

### Compiler toolchain {#compiler}

More functions are now eligible for inlining by default, including
functions that call `panic`.

The compiler toolchain now supports column information in [line
directives](/cmd/compile/#hdr-Compiler_Directives).

A new package export data format has been introduced. This should be
transparent to end users, except for speeding up build times for large
Go projects. If it does cause problems, it can be turned off again by
passing `-gcflags=all=-iexport=false` to the `go` tool when building a
binary.

The compiler now rejects unused variables declared in a type switch
guard, such as `x` in the following example:

    func f(v interface{}) {
        switch x := v.(type) {
        }
    }

This was already rejected by both `gccgo` and
[go/types](/pkg/go/types/).

### Assembler

The assembler for `amd64` now accepts AVX512 instructions.

### Debugging

The compiler now produces significantly more accurate debug information
for optimized binaries, including variable location information, line
numbers, and breakpoint locations. This should make it possible to debug
binaries compiled *without* `-N` `-l`. There are still limitations to
the quality of the debug information, some of which are fundamental, and
some of which will continue to improve with future releases.

DWARF sections are now compressed by default because of the expanded and
more accurate debug information produced by the compiler. This is
transparent to most ELF tools (such as debuggers on Linux and \*BSD) and
is supported by the Delve debugger on all platforms, but has limited
support in the native tools on macOS and Windows. To disable DWARF
compression, pass `-ldflags=-compressdwarf=false` to the `go` tool when
building a binary.

Go 1.11 adds experimental support for calling Go functions from within a
debugger. This is useful, for example, to call `String` methods when
paused at a breakpoint. This is currently only supported by Delve
(version 1.1.0 and up).

### Test

Since Go 1.10, the `go` `test` command runs `go` `vet` on the package
being tested, to identify problems before running the test. Since `vet`
typechecks the code with [go/types](/pkg/go/types/) before running,
tests that do not typecheck will now fail. In particular, tests that
contain an unused variable inside a closure compiled with Go 1.10,
because the Go compiler incorrectly accepted them ([Issue
#3059](/issues/3059)), but will now fail, since `go/types` correctly
reports an "unused variable" error in this case.

The `-memprofile` flag to `go` `test` now defaults to the "allocs"
profile, which records the total bytes allocated since the test began
(including garbage-collected bytes).

### Vet

The [`go` `vet`](/cmd/vet/) command now reports a fatal error when the
package under analysis does not typecheck. Previously, a type checking
error simply caused a warning to be printed, and `vet` to exit with
status 1.

Additionally, [`go` `vet`](/cmd/vet) has become more robust when
format-checking `printf` wrappers. Vet now detects the mistake in this
example:

    func wrapper(s string, args ...interface{}) {
        fmt.Printf(s, args...)
    }

    func main() {
        wrapper("%s", 42)
    }

### Trace

With the new `runtime/trace` package's [user annotation
API](/pkg/runtime/trace/#hdr-User_annotation), users can record
application-level information in execution traces and create groups of
related goroutines. The `go` `tool` `trace` command visualizes this
information in the trace view and the new user task/region analysis
page.

### Cgo

Since Go 1.10, cgo has translated some C pointer types to the Go type
`uintptr`. These types include the `CFTypeRef` hierarchy in Darwin's
CoreFoundation framework and the `jobject` hierarchy in Java's JNI
interface. In Go 1.11, several improvements have been made to the code
that detects these types. Code that uses these types may need some
updating. See the [Go 1.10 release notes](go1.10.html#cgo) for details.

### Go command {#go_command}

The environment variable `GOFLAGS` may now be used to set default flags
for the `go` command. This is useful in certain situations. Linking can
be noticeably slower on underpowered systems due to DWARF, and users may
want to set `-ldflags=-w` by default. For modules, some users and CI
systems will want vendoring always, so they should set `-mod=vendor` by
default. For more information, see the [`go` command
documentation](/cmd/go/#hdr-Environment_variables).

### Godoc

Go 1.11 will be the last release to support `godoc`'s command-line
interface. In future releases, `godoc` will only be a web server. Users
should use `go` `doc` for command-line help output instead.

The `godoc` web server now shows which version of Go introduced new API
features. The initial Go version of types, funcs, and methods are shown
right-aligned. For example, see [`UserCacheDir`](/pkg/os/#UserCacheDir),
with "1.11" on the right side. For struct fields, inline comments are
added when the struct field was added in a Go version other than when
the type itself was introduced. For a struct field example, see
[`ClientTrace.Got1xxResponse`](/pkg/net/http/httptrace/#ClientTrace.Got1xxResponse).

### Gofmt

One minor detail of the default formatting of Go source code has
changed. When formatting expression lists with inline comments, the
comments were aligned according to a heuristic. However, in some cases
the alignment would be split up too easily, or introduce too much
whitespace. The heuristic has been changed to behave better for
human-written code.

Note that these kinds of minor updates to gofmt are expected from time
to time. In general, systems that need consistent formatting of Go
source code should use a specific version of the `gofmt` binary. See the
[go/format](/pkg/go/format/) package documentation for more information.

### Run

The [`go` `run`](/cmd/go/) command now allows a single import path, a
directory name or a pattern matching a single package. This allows
`go` `run` `pkg` or `go` `run` `dir`, most importantly `go` `run` `.`

## Runtime

The runtime now uses a sparse heap layout so there is no longer a limit
to the size of the Go heap (previously, the limit was 512GiB). This also
fixes rare "address space conflict" failures in mixed Go/C binaries or
binaries compiled with `-race`.

On macOS and iOS, the runtime now uses `libSystem.dylib` instead of
calling the kernel directly. This should make Go binaries more
compatible with future versions of macOS and iOS. The
[syscall](/pkg/syscall) package still makes direct system calls; fixing
this is planned for a future release.

## Performance

As always, the changes are so general and varied that precise statements
about performance are difficult to make. Most programs should run a bit
faster, due to better generated code and optimizations in the core
library.

There were multiple performance changes to the `math/big` package as
well as many changes across the tree specific to `GOARCH=arm64`.

### Compiler toolchain {#performance-compiler}

The compiler now optimizes map clearing operations of the form:

    for k := range m {
        delete(m, k)
    }

The compiler now optimizes slice extension of the form
`append(s,` `make([]T,` `n)...)`.

The compiler now performs significantly more aggressive bounds-check and
branch elimination. Notably, it now recognizes transitive relations, so
if `i<j` and `j<len(s)`, it can use these facts to eliminate the bounds
check for `s[i]`. It also understands simple arithmetic such as
`s[i-10]` and can recognize more inductive cases in loops. Furthermore,
the compiler now uses bounds information to more aggressively optimize
shift operations.

## Standard library {#library}

All of the changes to the standard library are minor.

### Minor changes to the library {#minor_library_changes}

As always, there are various minor changes and updates to the library,
made with the Go 1 [promise of compatibility](/doc/go1compat) in mind.

#### [crypto](/pkg/crypto/) {#cryptopkgcrypto}

Certain crypto operations, including
[`ecdsa.Sign`](/pkg/crypto/ecdsa/#Sign),
[`rsa.EncryptPKCS1v15`](/pkg/crypto/rsa/#EncryptPKCS1v15) and
[`rsa.GenerateKey`](/pkg/crypto/rsa/#GenerateKey), now randomly read an
extra byte of randomness to ensure tests don't rely on internal
behavior.

#### [crypto/cipher](/pkg/crypto/cipher/) {#cryptocipherpkgcryptocipher}

The new function
[`NewGCMWithTagSize`](/pkg/crypto/cipher/#NewGCMWithTagSize) implements
Galois Counter Mode with non-standard tag lengths for compatibility with
existing cryptosystems.

#### [crypto/rsa](/pkg/crypto/rsa/) {#cryptorsapkgcryptorsa}

[`PublicKey`](/pkg/crypto/rsa/#PublicKey) now implements a
[`Size`](/pkg/crypto/rsa/#PublicKey.Size) method that returns the
modulus size in bytes.

#### [crypto/tls](/pkg/crypto/tls/) {#cryptotlspkgcryptotls}

[`ConnectionState`](/pkg/crypto/tls/#ConnectionState)'s new
[`ExportKeyingMaterial`](/pkg/crypto/tls/#ConnectionState.ExportKeyingMaterial)
method allows exporting keying material bound to the connection
according to RFC 5705.

#### [crypto/x509](/pkg/crypto/x509/) {#cryptox509pkgcryptox509}

The deprecated, legacy behavior of treating the `CommonName` field as a
hostname when no Subject Alternative Names are present is now disabled
when the CN is not a valid hostname. The `CommonName` can be completely
ignored by adding the experimental value `x509ignoreCN=1` to the
`GODEBUG` environment variable. When the CN is ignored, certificates
without SANs validate under chains with name constraints instead of
returning `NameConstraintsWithoutSANs`.

Extended key usage restrictions are again checked only if they appear in
the `KeyUsages` field of
[`VerifyOptions`](/pkg/crypto/x509/#VerifyOptions), instead of always
being checked. This matches the behavior of Go 1.9 and earlier.

The value returned by
[`SystemCertPool`](/pkg/crypto/x509/#SystemCertPool) is now cached and
might not reflect system changes between invocations.

#### [debug/elf](/pkg/debug/elf/) {#debugelfpkgdebugelf}

More [`ELFOSABI`](/pkg/debug/elf/#ELFOSABI_NONE) and
[`EM`](/pkg/debug/elf/#EM_NONE) constants have been added.

#### [encoding/asn1](/pkg/encoding/asn1/) {#encodingasn1pkgencodingasn1}

`Marshal` and [`Unmarshal`](/pkg/encoding/asn1/#Unmarshal) now support
"private" class annotations for fields.

#### [encoding/base32](/pkg/encoding/base32/) {#encodingbase32pkgencodingbase32}

The decoder now consistently returns `io.ErrUnexpectedEOF` for an
incomplete chunk. Previously it would return `io.EOF` in some cases.

#### [encoding/csv](/pkg/encoding/csv/) {#encodingcsvpkgencodingcsv}

The `Reader` now rejects attempts to set the
[`Comma`](/pkg/encoding/csv/#Reader.Comma) field to a double-quote
character, as double-quote characters already have a special meaning in
CSV.

#### [html/template](/pkg/html/template/) {#htmltemplatepkghtmltemplate}

The package has changed its behavior when a typed interface value is
passed to an implicit escaper function. Previously such a value was
written out as (an escaped form) of `<nil>`. Now such values are
ignored, just as an untyped `nil` value is (and always has been)
ignored.

#### [image/gif](/pkg/image/gif/) {#imagegifpkgimagegif}

Non-looping animated GIFs are now supported. They are denoted by having
a [`LoopCount`](/pkg/image/gif/#GIF.LoopCount) of -1.

#### [io/ioutil](/pkg/io/ioutil/) {#ioioutilpkgioioutil}

The [`TempFile`](/pkg/io/ioutil/#TempFile) function now supports
specifying where the random characters in the filename are placed. If
the `prefix` argument includes a "`*`", the random string replaces the
"`*`". For example, a `prefix` argument of "`myname.*.bat`" will result
in a random filename such as "`myname.123456.bat`". If no "`*`" is
included the old behavior is retained, and the random digits are
appended to the end.

#### [math/big](/pkg/math/big/) {#mathbigpkgmathbig}

[`ModInverse`](/pkg/math/big/#Int.ModInverse) now returns nil when g and
n are not relatively prime. The result was previously undefined.

#### [mime/multipart](/pkg/mime/multipart/) {#mimemultipartpkgmimemultipart}

The handling of form-data with missing/empty file names has been
restored to the behavior in Go 1.9: in the
[`Form`](/pkg/mime/multipart/#Form) for the form-data part the value is
available in the `Value` field rather than the `File` field. In Go
releases 1.10 through 1.10.3 a form-data part with a missing/empty file
name and a non-empty "Content-Type" field was stored in the `File`
field. This change was a mistake in 1.10 and has been reverted to the
1.9 behavior.

#### [mime/quotedprintable](/pkg/mime/quotedprintable/) {#mimequotedprintablepkgmimequotedprintable}

To support invalid input found in the wild, the package now permits
non-ASCII bytes but does not validate their encoding.

#### [net](/pkg/net/) {#netpkgnet}

The new [`ListenConfig`](/pkg/net/#ListenConfig) type and the new
[`Dialer.Control`](/pkg/net/#Dialer.Control) field permit setting socket
options before accepting and creating connections, respectively.

The [`syscall.RawConn`](/pkg/syscall/#RawConn) `Read` and `Write`
methods now work correctly on Windows.

The `net` package now automatically uses the [`splice` system
call](https://man7.org/linux/man-pages/man2/splice.2.html){rel="noreferrer"
target="_blank"} on Linux when copying data between TCP connections in
[`TCPConn.ReadFrom`](/pkg/net/#TCPConn.ReadFrom), as called by
[`io.Copy`](/pkg/io/#Copy). The result is faster, more efficient TCP
proxying.

The [`TCPConn.File`](/pkg/net/#TCPConn.File),
[`UDPConn.File`](/pkg/net/#UDPConn.File),
[`UnixConn.File`](/pkg/net/#UnixCOnn.File), and
[`IPConn.File`](/pkg/net/#IPConn.File) methods no longer put the
returned `*os.File` into blocking mode.

#### [net/http](/pkg/net/http/) {#nethttppkgnethttp}

The [`Transport`](/pkg/net/http/#Transport) type has a new
[`MaxConnsPerHost`](/pkg/net/http/#Transport.MaxConnsPerHost) option
that permits limiting the maximum number of connections per host.

The [`Cookie`](/pkg/net/http/#Cookie) type has a new
[`SameSite`](/pkg/net/http/#Cookie.SameSite) field (of new type also
named [`SameSite`](/pkg/net/http/#SameSite)) to represent the new cookie
attribute recently supported by most browsers. The `net/http`'s
`Transport` does not use the `SameSite` attribute itself, but the
package supports parsing and serializing the attribute for browsers to
use.

It is no longer allowed to reuse a [`Server`](/pkg/net/http/#Server)
after a call to [`Shutdown`](/pkg/net/http/#Server.Shutdown) or
[`Close`](/pkg/net/http/#Server.Close). It was never officially
supported in the past and had often surprising behavior. Now, all future
calls to the server's `Serve` methods will return errors after a
shutdown or close.

The constant `StatusMisdirectedRequest` is now defined for HTTP status
code 421.

The HTTP server will no longer cancel contexts or send on
[`CloseNotifier`](/pkg/net/http/#CloseNotifier) channels upon receiving
pipelined HTTP/1.1 requests. Browsers do not use HTTP pipelining, but
some clients (such as Debian's `apt`) may be configured to do so.

[`ProxyFromEnvironment`](/pkg/net/http/#ProxyFromEnvironment), which is
used by the [`DefaultTransport`](/pkg/net/http/#DefaultTransport), now
supports CIDR notation and ports in the `NO_PROXY` environment variable.

#### [net/http/httputil](/pkg/net/http/httputil/) {#nethttphttputilpkgnethttphttputil}

The [`ReverseProxy`](/pkg/net/http/httputil/#ReverseProxy) has a new
[`ErrorHandler`](/pkg/net/http/httputil/#ReverseProxy.ErrorHandler)
option to permit changing how errors are handled.

The `ReverseProxy` now also passes "`TE:` `trailers`" request headers
through to the backend, as required by the gRPC protocol.

#### [os](/pkg/os/) {#ospkgos}

The new [`UserCacheDir`](/pkg/os/#UserCacheDir) function returns the
default root directory to use for user-specific cached data.

The new [`ModeIrregular`](/pkg/os/#ModeIrregular) is a
[`FileMode`](/pkg/os/#FileMode) bit to represent that a file is not a
regular file, but nothing else is known about it, or that it's not a
socket, device, named pipe, symlink, or other file type for which Go has
a defined mode bit.

[`Symlink`](/pkg/os/#Symlink) now works for unprivileged users on
Windows 10 on machines with Developer Mode enabled.

When a non-blocking descriptor is passed to
[`NewFile`](/pkg/os#NewFile), the resulting `*File` will be kept in
non-blocking mode. This means that I/O for that `*File` will use the
runtime poller rather than a separate thread, and that the
[`SetDeadline`](/pkg/os/#File.SetDeadline) methods will work.

#### [os/signal](/pkg/os/signal/) {#ossignalpkgossignal}

The new [`Ignored`](/pkg/os/signal/#Ignored) function reports whether a
signal is currently ignored.

#### [os/user](/pkg/os/user/) {#osuserpkgosuser}

The `os/user` package can now be built in pure Go mode using the build
tag "`osusergo`", independent of the use of the environment variable
`CGO_ENABLED=0`. Previously the only way to use the package's pure Go
implementation was to disable `cgo` support across the entire program.

#### [runtime](/pkg/runtime/) {#runtimepkgruntime}

Setting the `GODEBUG=tracebackancestors=`*`N`* environment variable now
extends tracebacks with the stacks at which goroutines were created,
where *N* limits the number of ancestor goroutines to report.

#### [runtime/pprof](/pkg/runtime/pprof/) {#runtimepprofpkgruntimepprof}

This release adds a new "allocs" profile type that profiles total number
of bytes allocated since the program began (including garbage-collected
bytes). This is identical to the existing "heap" profile viewed in
`-alloc_space` mode. Now `go test -memprofile=...` reports an "allocs"
profile instead of "heap" profile.

#### [sync](/pkg/sync/) {#syncpkgsync}

The mutex profile now includes reader/writer contention for
[`RWMutex`](/pkg/sync/#RWMutex). Writer/writer contention was already
included in the mutex profile.

#### [syscall](/pkg/syscall/) {#syscallpkgsyscall}

On Windows, several fields were changed from `uintptr` to a new
[`Pointer`](/pkg/syscall/?GOOS=windows&GOARCH=amd64#Pointer) type to
avoid problems with Go's garbage collector. The same change was made to
the
[`golang.org/x/sys/windows`](https://godoc.org/golang.org/x/sys/windows){rel="noreferrer"
target="_blank"} package. For any code affected, users should first
migrate away from the `syscall` package to the
`golang.org/x/sys/windows` package, and then change to using the
`Pointer`, while obeying the [`unsafe.Pointer` conversion
rules](/pkg/unsafe/#Pointer).

On Linux, the `flags` parameter to
[`Faccessat`](/pkg/syscall/?GOOS=linux&GOARCH=amd64#Faccessat) is now
implemented just as in glibc. In earlier Go releases the flags parameter
was ignored.

On Linux, the `flags` parameter to
[`Fchmodat`](/pkg/syscall/?GOOS=linux&GOARCH=amd64#Fchmodat) is now
validated. Linux's `fchmodat` doesn't support the `flags` parameter so
we now mimic glibc's behavior and return an error if it's non-zero.

#### [text/scanner](/pkg/text/scanner/) {#textscannerpkgtextscanner}

The [`Scanner.Scan`](/pkg/text/scanner/#Scanner.Scan) method now returns
the [`RawString`](/pkg/text/scanner/#RawString) token instead of
[`String`](/pkg/text/scanner/#String) for raw string literals.

#### [text/template](/pkg/text/template/) {#texttemplatepkgtexttemplate}

Modifying template variables via assignments is now permitted via the
`=` token:

      {{ $v := "init" }}
      {{ if true }}
        {{ $v = "changed" }}
      {{ end }}
      v: {{ $v }} {{/* "changed" */}}

In previous versions untyped `nil` values passed to template functions
were ignored. They are now passed as normal arguments.

#### [time](/pkg/time/) {#timepkgtime}

Parsing of timezones denoted by sign and offset is now supported. In
previous versions, numeric timezone names (such as `+03`) were not
considered valid, and only three-letter abbreviations (such as `MST`)
were accepted when expecting a timezone name.
:::
:::::
