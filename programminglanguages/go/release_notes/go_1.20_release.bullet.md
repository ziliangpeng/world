# Go 1.20 Release Notes

**Released:** February 1, 2023
**EOL:** February 2025

## Major Highlights

Go 1.20 introduces profile-guided optimization, wraps multiple errors, and delivers significant performance improvements:

1. **Profile-Guided Optimization (PGO)** - Preview support for PGO with 3-4% performance gains from aggressive inlining at hot call sites
2. **Wrapping multiple errors** - `errors` and `fmt` now support wrapping multiple errors with `%w` and new `errors.Join`
3. **New crypto/ecdh package** - Explicit ECDH support over NIST curves and Curve25519
4. **Coverage for integration tests** - `go build -cover` enables code coverage for programs, not just unit tests
5. **Faster builds** - 10% improvement over Go 1.19, back in line with Go 1.17
6. **HTTP ResponseController** - Cleaner API for per-request controls like deadlines

## Breaking Changes

- 🟡 **go command** `go build` and `go install` no longer accept the `-i` flag (deprecated since Go 1.16)

## New Features

### Language Features

- 🔴 **Language** Conversions from slice to array - `[4]byte(x)` instead of `*(*[4]byte)(x)`
- 🔴 **unsafe** New functions: `SliceData`, `String`, `StringData` (complete ability to construct/deconstruct slices/strings)
- 🟡 **Language** Struct/array comparison now defined to stop at first mismatch (clarification, not behavior change)
- 🟡 **Language** Comparable types may satisfy `comparable` constraints even if comparison can panic at runtime

### New Packages

- 🔴 **crypto/ecdh** Explicit ECDH key exchange over NIST curves and Curve25519

### Tooling & Developer Experience

- 🔴 **go command** Profile-Guided Optimization via `-pgo` flag or auto-detection of `default.pgo` file
- 🔴 **go command** Code coverage for programs with `go build -cover` and `GOCOVERDIR` environment variable
- 🔴 **go command** `go version -m` now supports more binary types (Windows DLLs, Linux binaries without execute permission)
- 🔴 **runtime/coverage** New package for writing coverage data at runtime from long-running programs
- 🟡 **go command** Architecture feature build tags (e.g., `amd64.v2`) for selecting implementation files
- 🟡 **go command** `-C <dir>` flag changes directory before performing command
- 🟡 **go generate** `-skip <pattern>` skips directives matching pattern
- 🟡 **go test** `-skip <pattern>` skips tests/subtests/examples matching pattern
- 🟡 **go test** `-json` implementation improved for robustness; adds `start` event
- 🟡 **vet** Improved loop variable capture detection - now recursive into last statements of if/switch/select

## Improvements

### Performance

- 🟢 **Compiler** PGO enables aggressive inlining at hot call sites - 3-4% improvement in representative programs
- 🟢 **Compiler** Build speeds improved by up to 10%, back in line with Go 1.17
- 🟢 **Compiler** Type declarations within generic functions/methods now supported
- 🟢 **Compiler** Rejects anonymous interface cycles by default
- 🟢 **Runtime** GC internal data structures reorganized - up to 2% CPU improvement and reduced memory overheads
- 🟢 **Runtime** GC behaves less erratically with goroutine assists

### Error Messages & Debugging

- 🟡 **vet** New diagnostic for incorrect time formats (2006-02-01 instead of 2006-01-02)

## Deprecations

- 🟡 **Platform** Windows 7, 8, Server 2008, Server 2012: Go 1.20 is last release - Go 1.21 requires Windows 10 or Server 2016
- 🟡 **Platform** macOS 10.13 High Sierra, 10.14 Mojave: Go 1.20 is last release - Go 1.21 requires macOS 10.15 Catalina

## Platform & Environment

- 🟡 **Platform** FreeBSD/RISC-V: Experimental support (`GOOS=freebsd`, `GOARCH=riscv64`)
- 🟡 **Cgo** Disables cgo by default on systems without C toolchain (when `CGO_ENABLED`/`CC` unset and no compiler found)
- 🟡 **Cgo** macOS: `net` and `os/user` rewritten to not use cgo - same code for cgo/non-cgo/cross-compilation
- 🟡 **Cgo** macOS: Race detector rewritten to not use cgo
- 🟡 **Cgo** Consequence: macOS programs with `net` and `-buildmode=c-archive` require `-lresolv` when linking C code

## Implementation Details

- 🟢 **Compiler** Front-end upgraded to fix generic-types issues
- 🟢 **Linker** Selects dynamic interpreter for glibc/musl at link time on Linux
- 🟢 **Linker** Supports modern LLVM-based C toolchains on Windows
- 🟢 **Linker** Uses `go:` and `type:` prefixes instead of `go.` and `type.` for compiler-generated symbols
- 🟢 **Bootstrap** Requires Go 1.17.13 for bootstrap

## Standard Library Highlights

### Multiple Error Wrapping

- 🔴 **errors** `errors.Join` wraps multiple errors
- 🔴 **errors** `errors.Is` and `errors.As` updated to inspect multiply wrapped errors
- 🔴 **fmt** `fmt.Errorf` now supports multiple `%w` verbs

### HTTP Improvements

- 🔴 **net/http** `ResponseController` provides cleaner API for per-request functionality (deadlines, etc.)
- 🔴 **net/http/httputil** `ReverseProxy.Rewrite` hook supersedes `Director` for safer header manipulation
- 🔴 **net/http/httputil** `ProxyRequest.SetURL` routes request to destination
- 🔴 **net/http/httputil** `ProxyRequest.SetXForwarded` sets X-Forwarded-* headers

### New APIs

- 🟡 **bytes** `CutPrefix` and `CutSuffix` like `TrimPrefix`/`TrimSuffix` but report if trimmed
- 🟡 **bytes** `Clone` allocates copy of byte slice
- 🟡 **strings** `CutPrefix` and `CutSuffix` like `TrimPrefix`/`TrimSuffix` but report if trimmed
- 🟡 **context** `WithCancelCause` cancels context with given error; `Cause` retrieves it
- 🟡 **crypto/ecdsa** `PrivateKey.ECDH` converts to `ecdh.PrivateKey`
- 🟡 **crypto/ed25519** Support for Ed25519ph (pre-hashed) and Ed25519ctx (with context)
- 🟡 **crypto/rsa** `OAEPOptions.MGFHash` allows separate MGF1 hash configuration
- 🟡 **crypto/subtle** `XORBytes` XORs two byte slices
- 🟡 **time** New layout constants: `DateTime`, `DateOnly`, `TimeOnly`
- 🟡 **time** `Time.Compare` compares two times
- 🟡 **reflect** `Value.Comparable` and `Value.Equal` for comparing Values
- 🟡 **reflect** `Value.Grow` extends slice to guarantee space
- 🟡 **reflect** `Value.SetZero` sets value to zero for its type
- 🟡 **path/filepath** `IsLocal` reports if path is lexically local to directory
- 🟡 **io** `OffsetWriter` wraps `WriterAt` with adjustable file offset

### Security & Compatibility

- 🔴 **crypto/rsa** New safer constant-time backend - 15-45% slower decryption, 20x slower encryption
- 🔴 **crypto/tls** Parsed certificates shared across connections - significant memory savings
- 🔴 **crypto/tls** `CertificateVerificationError` includes presented certificates
- 🟢 **archive/tar** `GODEBUG=tarinsecurepath=0` rejects insecure paths
- 🟢 **archive/zip** `GODEBUG=zipinsecurepath=0` rejects insecure paths
- 🟢 **math/rand** Global RNG now auto-seeds with random value; `Seed` deprecated
- 🟢 **regexp** `syntax.ErrLarge` for expressions too large (was `ErrInternalError` in patches)
