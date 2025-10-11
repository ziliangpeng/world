# Go 1.16 Release Notes

**Released:** February 16, 2021
**EOL:** August 2022 (expected)

## Major Highlights

Go 1.16 brings embedded files, file system interfaces, and modules-by-default:

1. **Modules on by default** - `GO111MODULE=on` is now default (was `auto`)
2. **Embedded files** - New `//go:embed` directive and `embed` package for embedding files at compile time
3. **File system abstraction** - New `io/fs` package with `fs.FS` interface for read-only file trees
4. **Apple Silicon support** - Native darwin/arm64 support for Apple M1 Macs
5. **Linker improvements continued** - 20-25% faster, 5-15% less memory across all platforms
6. **io/ioutil deprecation** - Functions moved to `io` and `os` packages

## Changes to the Language

None.

## Breaking Changes

- 🔴 **Modules** `GO111MODULE` defaults to `on` (was `auto`) - module mode enabled everywhere
- 🔴 **Modules** Build commands no longer modify `go.mod`/`go.sum` by default (as if `-mod=readonly`)
- 🟡 **Platform** iOS now uses `GOOS=ios` (was `GOOS=darwin` with arm64) - affects `*_ios.go` files
- 🟡 **386** x87-only mode (`GO386=387`) dropped - soft float mode (`GO386=softfloat`) for non-SSE2

## Deprecations

- 🟡 **io/ioutil** Package deprecated - functions moved to `io` and `os`
- 🟡 **crypto/dsa** Package deprecated (see issue #40337)
- 🟡 **go command** `-i` flag deprecated (build cache made it unnecessary)

## New Features

- 🔴 **Embedded Files** `//go:embed` directive embeds files into binaries
- 🔴 **embed** New package provides access to embedded files
- 🔴 **io/fs** New file system abstraction with `fs.FS` interface
- 🔴 **testing/fstest** New package for testing `fs.FS` implementations with `TestFS` and `MapFS`
- 🔴 **Modules** Module retraction with `retract` directives in `go.mod`
- 🟡 **go install** Accepts version suffixes (e.g., `go install example.com/cmd@v1.0.0`)
- 🟡 **go command** New `-overlay` flag for file path replacements (for editor tooling)
- 🟡 **GOVCS** New environment variable limits version control tools for security
- 🟡 **Vet** Warning for `testing.T` methods called from goroutines
- 🟡 **Vet** Warning for clobbering BP register in amd64 assembly
- 🟡 **Vet** Warning for incorrect `asn1.Unmarshal` usage

## Improvements

- 🟢 **Linker** 20-25% faster, 5-15% less memory on linux/amd64 (more for other platforms)
- 🟢 **Linker** Smaller binaries from aggressive symbol pruning
- 🟢 **Runtime** New `runtime/metrics` package for stable metrics interface
- 🟢 **Runtime** `GODEBUG=inittrace=1` traces package init timing and memory
- 🟢 **Runtime** Linux: Promptly releases memory to OS with `MADV_DONTNEED` (was `MADV_FREE`)
- 🟢 **Runtime** Race detector more precise - follows channel synchronization rules more closely
- 🟢 **Compiler** Inlines functions with non-labeled `for` loops, method values, type switches
- 🟢 **Compiler** More indirect call inlining detected

## Tooling & Developer Experience

- 🟡 **Modules** `go mod tidy`/`go mod vendor` accept `-e` flag to proceed despite errors
- 🟡 **Modules** `exclude` directives in main module now fully enforced
- 🟡 **Modules** Disallows import paths with non-ASCII or leading dot
- 🟡 **go install** Recommended way to install commands (deprecating `go get` for this)
- 🟡 **go test** `os.Exit(0)` during test function considered a failure
- 🟡 **go test** Reports error when `-c`/`-i` used with unknown flags
- 🟡 **go get** `-insecure` flag deprecated (use `GOINSECURE` instead)
- 🟡 **go list** `-export` flag sets `BuildID` field
- 🟡 **all pattern** Matches only transitively-imported packages (not test-of-test packages)
- 🟡 **-toolexec** Now sets `TOOLEXEC_IMPORTPATH` environment variable
- 🟡 **cgo** No longer translates C struct bitfields to Go fields

## Platform & Environment

- 🔴 **Platform** Apple Silicon: darwin/arm64 support for M1 Macs with cgo, race detector, all build modes
- 🟡 **Platform** iOS renamed from darwin/arm64 to ios/arm64 (implies darwin build tag)
- 🟡 **Platform** iOS: New ios/amd64 for iOS simulator on AMD64 Macs
- 🟡 **Platform** NetBSD: netbsd/arm64 support
- 🟡 **Platform** OpenBSD: openbsd/mips64 support (no cgo yet)
- 🟡 **Platform** OpenBSD: System calls via `libc` on amd64/arm64 for forward-compatibility
- 🟡 **Platform** 386: x87 mode dropped, soft float mode (`GO386=softfloat`) for non-SSE2
- 🟡 **Platform** RISC-V: linux/riscv64 now supports cgo, `-buildmode=pie`, performance optimizations
- 🟡 **Platform** macOS 10.12 Sierra last supported (10.13+ required in Go 1.17)
- 🟡 **Platform** Windows: `-buildmode=c-shared` generates ASLR DLLs by default

## Standard Library Changes

### Major Changes

- 🔴 **embed** New package for embedded file access
- 🔴 **io/fs** New file system interface `fs.FS`
- 🔴 **testing/fstest** New testing utilities for `fs.FS` implementations
- 🟡 **io/ioutil** Deprecated - functions moved to `io` and `os`:
  - `Discard` → `io.Discard`
  - `NopCloser` → `io.NopCloser`
  - `ReadAll` → `io.ReadAll`
  - `ReadDir` → `os.ReadDir` (returns `[]os.DirEntry`)
  - `ReadFile` → `os.ReadFile`
  - `TempDir` → `os.MkdirTemp`
  - `TempFile` → `os.CreateTemp`
  - `WriteFile` → `os.WriteFile`

### archive/zip

- 🟡 **archive/zip** `Reader.Open` implements `fs.FS` interface

### crypto

- 🟡 **crypto/dsa** Package deprecated
- 🟡 **crypto/hmac** `New` panics if hash function returns different values on separate calls
- 🟡 **crypto/tls** I/O on closing connections detectable with `net.ErrClosed`
- 🟡 **crypto/tls** Default write deadline in `Conn.Close` prevents indefinite blocking
- 🟡 **crypto/tls** Handshake error if server selects non-advertised ALPN protocol
- 🟡 **crypto/tls** Prefers non-AES-GCM AEAD suites when no AES hardware support
- 🟡 **crypto/tls** `Config.Clone` returns nil for nil receiver
- 🟡 **crypto/x509** `GODEBUG=x509ignoreCN=0` removed in Go 1.17
- 🟡 **crypto/x509** `ParseCertificate`/`CreateCertificate` enforce string encoding restrictions
- 🟡 **crypto/x509** `CreateCertificate` verifies generated signature
- 🟡 **crypto/x509** DSA signature verification no longer supported
- 🟡 **crypto/x509** Windows: Returns all certificate chains from platform verifier
- 🟡 **crypto/x509** `SystemRootsError.Unwrap` for error chain access
- 🟡 **crypto/x509** Unix: More efficient system cert pool storage (~0.5MB less)

### encoding

- 🟡 **encoding/asn1** `Unmarshal`/`UnmarshalWithParams` return error (not panic) on nil/non-pointer
- 🟡 **encoding/json** `json` struct tags permit semicolons in object names
- 🟡 **encoding/xml** Case-insensitive check for reserved `xml` namespace prefixes

### go

- 🟡 **go/build** New fields for `//go:embed`: `EmbedPatterns`, `EmbedPatternPos`, etc.
- 🟡 **go/build** `IgnoredGoFiles` no longer includes `_*` and `.*` files
- 🟡 **go/build** New `IgnoredOtherFiles` field
- 🟡 **go/build/constraint** New package for parsing build constraint lines
- 🟡 **flag** New `Func` function for lighter-weight flag registration

### html/template

- 🟡 **html/template** `ParseFS`/`Template.ParseFS` read templates from `fs.FS`

### net/http

- 🟡 **net/http** `StripPrefix` strips both `Path` and `RawPath` fields
- 🟡 **net/http** Rejects `Range: bytes=--N` requests (negative suffix length)
- 🟡 **net/http** `SameSiteDefaultMode` cookies follow spec (no attribute set)
- 🟡 **net/http** `Client` sends explicit `Content-Length: 0` for empty `PATCH` bodies
- 🟡 **net/http** `ProxyFromEnvironment` no longer uses `HTTP_PROXY` for `https://` when `HTTPS_PROXY` unset
- 🟡 **net/http** `Transport.GetProxyConnectHeader` for dynamic `CONNECT` headers
- 🟡 **net/http** New `http.FS` converts `fs.FS` to `http.FileSystem`
- 🟡 **net/http/httputil** `ReverseProxy` flushes more aggressively for unknown-length streams
- 🟡 **net/smtp** `Client.Mail` sends `SMTPUTF8` directive to supporting servers

### os

- 🟡 **os** `Process.Signal` returns `ErrProcessDone` (not unexported `errFinished`)
- 🟡 **os** New `DirEntry` type (alias for `fs.DirEntry`)
- 🟡 **os** New `ReadDir` and `File.ReadDir` for reading directories
- 🟡 **os** New `CreateTemp`, `MkdirTemp`, `ReadFile`, `WriteFile` (from `io/ioutil`)
- 🟡 **os** `FileInfo`, `FileMode`, `PathError` now aliases for `io/fs` types
- 🟡 **os** New `DirFS` provides `fs.FS` backed by OS files
- 🟡 **os/signal** `NotifyContext` creates context canceled on signals

### path

- 🟡 **path** `Match` returns error on syntax errors in unmatched pattern part
- 🟡 **path/filepath** New `WalkDir` more efficient than `Walk`
- 🟡 **path/filepath** `Match`/`Glob` return error on syntax errors

### reflect

- 🟡 **reflect** Zero function optimized to avoid allocations

### runtime/debug

- 🟡 **runtime/debug** `runtime.Error` may have `Addr` method when `SetPanicOnFault` enabled

### strconv

- 🟡 **strconv** `ParseFloat` uses Eisel-Lemire algorithm (up to 2x faster)

### syscall

- 🟡 **syscall** `NewCallback`/`NewCallbackCDecl` support multiple sub-uintptr args
- 🟡 **syscall** Windows: `SysProcAttr.NoInheritHandles` disables handle inheritance
- 🟡 **syscall** Windows: `DLLError.Unwrap` for error chain access
- 🟡 **syscall** Linux: `Setgid`/`Setuid` now implemented
- 🟡 **syscall** Linux: `AllThreadsSyscall`/`AllThreadsSyscall6` for process-wide syscalls

### testing/iotest

- 🟡 **testing/iotest** New `ErrReader` always returns error
- 🟡 **testing/iotest** New `TestReader` tests `io.Reader` behavior

### text/template

- 🟡 **text/template** Newlines allowed inside action delimiters
- 🟡 **text/template** `ParseFS`/`Template.ParseFS` read templates from `fs.FS`
- 🟡 **text/template/parse** New `CommentNode` in parse tree

### time/tzdata

- 🟡 **time/tzdata** Slim timezone data format (~350 KB smaller)

### unicode

- 🟡 **unicode** Upgraded from Unicode 12.0 to Unicode 13.0 (5,930 new characters, 4 new scripts, 55 new emoji)
