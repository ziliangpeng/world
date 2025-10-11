# Go 1.21 Release Notes

**Released:** August 8, 2023
**EOL:** August 2025

## Major Highlights

Go 1.21 introduces powerful new standard library packages, enhanced backwards compatibility, and major PGO improvements:

1. **New slices, maps, and cmp packages** - Generic functions for common operations on slices and maps
2. **New log/slog package** - Structured logging with levels for production applications
3. **Built-in functions: min, max, clear** - Language additions for common operations
4. **Enhanced backwards/forwards compatibility** - GODEBUG formalized for behavior changes; toolchain selection for version requirements
5. **PGO production-ready** - Auto-enabled with `default.pgo`, now includes devirtualization; 2-7% performance gains
6. **Improved type inference** - More powerful generic type inference with better error messages

## Breaking Changes

- 🟡 **Language** Package initialization order now specified more precisely (sort by import path, then initialize)
- 🟡 **Language** nil panic now guaranteed to return non-nil from `recover` (can re-enable with `GODEBUG=panicnil=1`)

## New Features

### Language Features

- 🔴 **Language** New built-in functions: `min` and `max` compute smallest/largest of arguments
- 🔴 **Language** New built-in `clear` function deletes all map elements or zeros slice elements
- 🔴 **Language** Improved type inference - handles generic functions as arguments, method matching, untyped constants
- 🟡 **Language** Preview: for loop variables per-iteration (enable with LoopvarExperiment)

### New Packages

- 🔴 **slices** Generic functions for slice operations (Sort, BinarySearch, Contains, etc.)
- 🔴 **maps** Generic functions for map operations (Keys, Values, Clone, Copy, etc.)
- 🔴 **cmp** Type constraint `Ordered` and generic comparison functions `Less`, `Compare`
- 🔴 **log/slog** Structured logging with levels (Debug, Info, Warn, Error)
- 🔴 **testing/slogtest** Validation for slog.Handler implementations

### Tooling & Developer Experience

- 🔴 **go command** Formalized GODEBUG for backwards compatibility - behavior tied to `go` line in go.mod
- 🔴 **go command** Toolchain selection - auto-invokes newer Go versions when required by `go` or `toolchain` directive
- 🔴 **go command** PGO auto-enabled with `-pgo=auto` when `default.pgo` present
- 🔴 **go command** New versioning: Go 1.21.0 is first release (tools report `go1.21.0`)
- 🟡 **go command** `-C` flag must now be first on command line
- 🟡 **go test** `-fullpath` prints full path names in test log messages
- 🟡 **go test** `-c` supports writing test binaries for multiple packages
- 🟡 **go test** `-o` accepts directory argument

## Improvements

### Performance

- 🟢 **Compiler** PGO now production-ready with 2-7% performance gains in representative programs
- 🟢 **Compiler** PGO now supports devirtualization of interface method calls
- 🟢 **Compiler** Build speed improved by up to 6% (compiler built with PGO)
- 🟢 **Runtime** Linux: Explicit management of transparent huge pages - less memory for small heaps, better CPU for large heaps
- 🟢 **Runtime** Up to 40% reduction in application tail latency, small memory use decrease
- 🟢 **Runtime** C-to-Go calls on same thread ~100-200ns (was 1-3μs)
- 🟢 **crypto/sha256** SHA-224/SHA-256 use native instructions on amd64 - 3-4x faster

### Error Messages & Debugging

- 🟢 **Runtime** Deep stacks now print first 50 + bottom 50 frames (not just first 100)
- 🟢 **Runtime** Stack traces include goroutine creator IDs

## Platform & Environment

- 🔴 **Platform** macOS: Requires 10.15 Catalina or later (discontinued 10.13/10.14 support)
- 🔴 **Platform** Windows: Requires Windows 10 or Server 2016 (discontinued Windows 7/8/Server 2008/2012 support)
- 🟡 **Platform** Windows/amd64: Linker emits SEH unwinding data by default
- 🟡 **Platform** ARM: When cross-compiling, `GOARM` defaults to 7

## Implementation Details

- 🟢 **Assembler** amd64: Frameless nosplit functions require explicit `NOFRAME` attribute
- 🟢 **Linker** Can delete dead global map variables with large initializers (PGO-assisted)
- 🟢 **Linker** Disallows `//go:linkname` to internal stdlib symbols (backward compatible exceptions exist)

## Standard Library Highlights

### New Features

- 🔴 **context** `WithoutCancel` returns copy not canceled when original is
- 🔴 **context** `WithDeadlineCause` and `WithTimeoutCause` set cancellation cause on deadline/timeout
- 🔴 **context** `AfterFunc` registers function to run after context cancelled
- 🟡 **bytes** `Buffer.Available` and `Buffer.AvailableBuffer` for efficient append
- 🟡 **errors** `ErrUnsupported` standardizes unsupported operation indication
- 🟡 **flag** `BoolFunc` defines flag without argument that calls function
- 🟡 **go/ast** `IsGenerated` reports if file has generated code comment
- 🟡 **go/ast** `File.GoVersion` records minimum Go version from build directives
- 🟡 **math/big** `Int.Float64` returns nearest float64 with rounding indication
- 🟡 **net** Linux: Multipath TCP support via `Dialer.SetMultipathTCP` and `ListenConfig.SetMultipathTCP`
- 🟡 **net/http** `ResponseController.EnableFullDuplex` allows concurrent HTTP/1 read/write
- 🟡 **os** Programs may pass empty `time.Time` to `Chtimes` to leave time unchanged
- 🟡 **testing** `Testing` function reports if program is a test
- 🟡 **reflect** `Value.Clear` clears map or zeros slice (corresponds to `clear` builtin)

### Deprecations

- 🟡 **crypto/elliptic** All `Curve` methods deprecated; use `crypto/ecdh` or third-party modules
- 🟡 **crypto/rsa** `GenerateMultiPrimeKey` and `PrecomputedValues.CRTValues` deprecated
- 🟡 **reflect** `SliceHeader` and `StringHeader` deprecated - use `unsafe.Slice`, `unsafe.String`, etc.

### Security & Compatibility

- 🔴 **crypto/tls** Servers skip verifying client certificates on resumption (except expiration check)
- 🔴 **crypto/tls** New session ticket APIs: `SessionState`, `WrapSession`, `UnwrapSession`, etc.
- 🔴 **crypto/tls** Both client and server implement Extended Master Secret (RFC 7627)
- 🔴 **crypto/tls** New `QUICConn` type for QUIC implementations with 0-RTT support
- 🔴 **crypto/x509** Name constraints now correctly enforced on non-leaf certificates
- 🟡 **runtime** New `Pinner` type allows pinning Go memory for safer cgo pointer passing
- 🟡 **runtime** `GODEBUG=cgocheck=2` now requires `GOEXPERIMENT=cgocheck2` at build time

### Performance

- 🟢 **crypto/rsa** Private operations performance better than Go 1.19 (had regressed in 1.20)
- 🟢 **runtime/trace** Collecting traces on amd64/arm64 up to 10x lower CPU cost
