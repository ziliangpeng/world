# Go 1.11 Release Notes

**Released:** August 24, 2018
**EOL:** February 2020 (expected)

## Major Highlights

Go 1.11 introduces experimental modules for dependency management and brings WebAssembly support:

1. **Experimental module support** - Alternative to GOPATH with integrated versioning and package distribution (experimental)
2. **WebAssembly port** - Experimental support for compiling Go to WebAssembly (`js/wasm`)
3. **Sparse heap layout** - Runtime now supports unlimited heap size (was 512GiB limit)
4. **Map clearing optimization** - Compiler optimizes `for k := range m { delete(m, k) }` pattern
5. **Aggressive bounds-check elimination** - Compiler recognizes transitive relations and inductive cases
6. **Better debug info** - Significantly more accurate debug information for optimized binaries

## Breaking Changes

- 🟡 **go/types** Compiler now rejects unused variables in type switch guards (matches gccgo behavior)

## Deprecations

- 🟡 **Go Command** `go get` is deprecated in legacy GOPATH mode (`GO111MODULE=off`)

## New Features

- 🔴 **Modules** Experimental module support with versioning and reproducible builds
- 🔴 **WebAssembly** Experimental `js/wasm` port with new `syscall/js` package
- 🟡 **Language** Files named `*_js.go` or `*_wasm.go` now ignored except when those GOOS/GOARCH values used
- 🟡 **Language** `*_riscv.go` filename pattern reserved for future RISC-V support
- 🟡 **go/packages** New external package for locating and loading Go packages (replaces `go/build` for many tasks)
- 🟡 **Compiler** Column information now supported in line directives
- 🟢 **Compiler** More functions eligible for inlining, including those that call `panic`
- 🟢 **Compiler** New package export data format speeds up build times

## Improvements

- 🟢 **Runtime** Sparse heap layout removes 512GiB heap size limit
- 🟢 **Runtime** macOS/iOS now use `libSystem.dylib` instead of direct kernel calls for better compatibility
- 🟢 **Compiler** Map clearing operations optimized: `for k := range m { delete(m, k) }`
- 🟢 **Compiler** Slice extension optimized: `append(s, make([]T, n)...)`
- 🟢 **Compiler** Aggressive bounds-check and branch elimination with transitive relations
- 🟢 **Debugging** Significantly more accurate debug info for optimized binaries (variable locations, line numbers, breakpoints)
- 🟢 **Debugging** DWARF sections now compressed by default (disable with `-ldflags=-compressdwarf=false`)
- 🟢 **Debugging** Experimental support for calling Go functions from debugger (Delve 1.1.0+)
- 🟢 **Assembler** AVX512 instructions now accepted on amd64

## Tooling & Developer Experience

- 🟡 **go command** `GOFLAGS` environment variable for setting default flags
- 🟡 **go command** Import paths containing `@` symbols now disallowed
- 🟡 **go test** `-memprofile` now defaults to "allocs" profile
- 🟡 **go test** Tests with unused variables in closures now fail (go/types correctly reports error)
- 🟡 **vet** Now reports fatal error when package doesn't typecheck (was warning)
- 🟡 **vet** Better printf wrapper format checking
- 🟢 **trace** New user annotation API for application-level information in traces
- 🟡 **godoc** Command-line interface deprecated - will be web-server only in Go 1.12
- 🟡 **godoc** Web server now shows which Go version introduced new API features
- 🟡 **gofmt** Improved alignment heuristic for inline comments in expression lists
- 🟡 **go run** Now accepts single import path, directory name, or pattern matching single package

## Implementation Details

- 🟢 **Compiler** New `-iexport` flag controls new export data format (use `-gcflags=all=-iexport=false` to disable)

## Platform & Environment

- 🟡 **Platform** OpenBSD 6.2+ now required (was announced in Go 1.10)
- 🟡 **Platform** macOS 10.10 Yosemite+ now required (was announced in Go 1.10)
- 🟡 **Platform** Windows 7+ now required (was announced in Go 1.10)
- 🟡 **Platform** OpenBSD 6.4+ supported (requires kernel changes)
- 🟡 **Platform** Race detector now supported on linux/ppc64le and netbsd/amd64 (with known issues)
- 🟡 **Platform** Memory sanitizer (`-msan`) supported on linux/arm64
- 🟡 **Platform** Build modes `c-shared` and `c-archive` on freebsd/amd64
- 🟡 **Platform** 64-bit MIPS: `GOMIPS64=hardfloat` (default) vs `GOMIPS64=softfloat`
- 🟡 **Platform** Soft-float ARM (`GOARM=5`) now uses more efficient floating point interface
- 🟡 **Platform** ARMv7 no longer requires `KUSER_HELPERS` kernel config

## Standard Library Changes

### Major Changes

- 🟢 **crypto/cipher** New `NewGCMWithTagSize` for non-standard GCM tag lengths
- 🟡 **crypto/rsa** `PublicKey.Size` returns modulus size in bytes
- 🟡 **crypto/tls** `ConnectionState.ExportKeyingMaterial` for RFC 5705 keying material export
- 🟡 **crypto/x509** CommonName field deprecated as hostname (use `x509ignoreCN=1` to disable)
- 🟡 **crypto/x509** Extended key usage checks match Go 1.9 behavior (only if in `VerifyOptions.KeyUsages`)
- 🟡 **crypto/x509** `SystemCertPool` now cached (may not reflect system changes)

### net

- 🟡 **net** New `ListenConfig` type and `Dialer.Control` for setting socket options before connections
- 🟡 **net** `syscall.RawConn` Read/Write now work correctly on Windows
- 🟡 **net** Automatic use of `splice` syscall on Linux for TCP connection copying
- 🟡 **net** `TCPConn.File`/`UDPConn.File`/`UnixConn.File`/`IPConn.File` no longer put files in blocking mode

### net/http

- 🟡 **net/http** `Transport.MaxConnsPerHost` limits maximum connections per host
- 🟡 **net/http** `Cookie.SameSite` field for SameSite cookie attribute
- 🟡 **net/http** `Server` cannot be reused after `Shutdown` or `Close`
- 🟡 **net/http** `StatusMisdirectedRequest` constant (421)
- 🟡 **net/http** HTTP server no longer cancels contexts for pipelined HTTP/1.1 requests
- 🟡 **net/http** `ProxyFromEnvironment` supports CIDR notation and ports in `NO_PROXY`
- 🟡 **net/http/httputil** `ReverseProxy.ErrorHandler` for custom error handling
- 🟡 **net/http/httputil** `ReverseProxy` passes `TE: trailers` headers through to backend

### Other packages

- 🟡 **os** `UserCacheDir` returns default cache directory
- 🟡 **os** `ModeIrregular` bit for non-regular files
- 🟡 **os** `Symlink` works for unprivileged users on Windows 10 Developer Mode
- 🟡 **os** Non-blocking descriptors passed to `NewFile` stay non-blocking
- 🟡 **os/signal** `Ignored` reports if signal is currently ignored
- 🟡 **os/user** Pure Go mode with build tag `osusergo`
- 🟡 **runtime** `GODEBUG=tracebackancestors=N` extends tracebacks with goroutine creation stacks
- 🟡 **runtime/pprof** New "allocs" profile type for total bytes allocated
- 🟡 **sync** Mutex profile includes reader/writer contention for `RWMutex`
- 🟡 **syscall** Windows: Multiple fields changed to `Pointer` type (use `golang.org/x/sys/windows`)
- 🟡 **syscall** Linux: `Faccessat` flags parameter now implemented
- 🟡 **syscall** Linux: `Fchmodat` validates flags parameter
- 🟢 **text/scanner** `Scanner.Scan` returns `RawString` token for raw string literals
- 🟡 **text/template** Template variable assignment via `=` token: `{{ $v = "changed" }}`
- 🟡 **text/template** Untyped nil values now passed to template functions as normal arguments
- 🟡 **time** Parsing of numeric timezone offsets (e.g., `+03`) now supported

### Minor library changes

- 🟡 **crypto** Extra randomness byte read in `ecdsa.Sign`, `rsa.EncryptPKCS1v15`, `rsa.GenerateKey`
- 🟡 **encoding/asn1** `Marshal` and `Unmarshal` support "private" class annotations
- 🟡 **encoding/base32** Decoder consistently returns `io.ErrUnexpectedEOF` for incomplete chunks
- 🟡 **encoding/csv** `Reader` rejects double-quote as `Comma` field
- 🟡 **html/template** Typed interface nil values ignored (not printed as `<nil>`)
- 🟡 **image/gif** Non-looping animated GIFs supported (`LoopCount` of -1)
- 🟡 **io/ioutil** `TempFile` supports `*` for random string placement in prefix
- 🟡 **math/big** `ModInverse` returns nil when g and n not relatively prime
- 🟡 **mime/multipart** Form-data with missing/empty filename reverted to Go 1.9 behavior
- 🟡 **mime/quotedprintable** Permits non-ASCII bytes without validating encoding
