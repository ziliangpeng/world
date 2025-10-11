# Go 1.18 Release Notes

**Released:** March 15, 2022
**EOL:** February 2024

## Major Highlights

Go 1.18 is a groundbreaking release that introduces generics, the most significant language change since Go 1:

1. **Generics (Type Parameters)** - Functions and types can now be parameterized with type parameters, enabling type-safe generic programming
2. **Fuzzing support** - Built-in fuzzing capabilities for automated testing with random inputs
3. **Workspaces** - New workspace mode allows working with multiple modules simultaneously
4. **New net/netip package** - Efficient, immutable, comparable IP address types
5. **Register-based calling convention expanded** - Extended to ARM64 and PowerPC64, providing 10%+ performance improvements
6. **Version control information in binaries** - `go version -m` now shows VCS info embedded during build

## Breaking Changes

- 🔴 **Language** Compiler now correctly reports "declared but not used" errors for variables set inside function literals but never used
- 🔴 **Language** Compiler now reports overflow when passing large rune constant expressions like `'1' << 32` to `print`/`println`
- 🟡 **go command** `go get` no longer builds or installs packages - use `go install example.com/cmd@latest` instead
- 🟡 **go command** `go get` now reports an error when used outside a module (no `go.mod` to update)
- 🟡 **go command** `go mod graph`, `go mod vendor`, `go mod verify`, `go mod why` no longer auto-update `go.mod`/`go.sum`
- 🟢 **crypto/elliptic** Operating on invalid curve points now returns random points instead of undefined behavior

## New Features

### Language Features

- 🔴 **Language** Type parameters for generic functions and types - parameterized declarations with square bracket syntax `[T any]`
- 🔴 **Language** New `~` token for approximate type constraints in interfaces
- 🔴 **Language** Extended interface syntax - can embed arbitrary types, unions, and `~T` elements (only usable as type constraints)
- 🔴 **Language** New predeclared identifier `any` - alias for `interface{}`
- 🔴 **Language** New predeclared identifier `comparable` - interface denoting types that support `==` and `!=`
- 🟡 **Language** Experimental packages `golang.org/x/exp/constraints`, `golang.org/x/exp/slices`, `golang.org/x/exp/maps` for generic utilities

### New Packages

- 🔴 **debug/buildinfo** Access to module versions, VCS info, and build flags embedded in executables
- 🔴 **net/netip** New IP address types - `Addr`, `AddrPort`, `Prefix` are immutable, comparable, and memory-efficient

### Tooling & Developer Experience

- 🔴 **go command** Fuzzing support with `go test -fuzz`, `-fuzztime`, `-fuzzminimizetime` flags
- 🔴 **go command** Workspace mode with `go.work` file for multi-module development
- 🔴 **go command** Version control information embedded in binaries (Git, Mercurial, Fossil, Bazaar)
- 🔴 **go command** `go version -m` shows VCS info and build settings from binaries
- 🟡 **go build** New `-asan` flag enables AddressSanitizer for interoperability with C/C++ code
- 🟡 **go mod download** Without arguments, only downloads modules explicitly required (not transitive dependencies)
- 🟡 **go mod vendor** New `-o` flag sets output directory
- 🟡 **go fix** Removes obsolete `// +build` lines in modules declaring `go 1.18` or later
- 🟡 **gofmt** Now reads and formats files concurrently - significantly faster on multi-CPU machines

## Improvements

### Performance

- 🟢 **Compiler** Register-based calling convention extended to ARM64, PowerPC64 - typical 10%+ performance improvements
- 🟢 **Compiler** Can now inline functions containing range loops or labeled for loops
- 🟢 **Runtime** GC now includes non-heap work sources when determining collection frequency
- 🟢 **Runtime** Returns memory to OS more efficiently and aggressively
- 🟢 **Runtime** `append` uses improved growth formula, less prone to sudden allocation behavior changes
- 🟢 **Linker** Emits far fewer relocations - faster linking, less memory, smaller binaries

### Error Messages & Debugging

- 🟢 **Runtime** Stack traces now print `?` after potentially inaccurate argument values (passed in registers)
- 🟡 **vet** Updated for generics - reports errors in generic code as it would in equivalent non-generic code
- 🟡 **vet** Improved precision for `copylock`, `printf`, `sortslice`, `testinggoroutine`, `tests` checkers

## Deprecations

- 🟡 **bytes** `bytes.Title` deprecated - use `golang.org/x/text/cases` instead
- 🟡 **strings** `strings.Title` deprecated - use `golang.org/x/text/cases` instead
- 🟡 **syscall** Windows: `Syscall`, `Syscall6`, `Syscall9`, `Syscall12`, `Syscall15`, `Syscall18` deprecated - use `SyscallN`
- 🟡 **crypto/x509** Support for signing with MD5WithRSA may be removed in Go 1.19

## Platform & Environment

- 🟡 **Platform** AMD64: New `GOAMD64` environment variable selects minimum target (v1/v2/v3/v4)
- 🟡 **Platform** RISC-V 64-bit: Now supports `c-archive` and `c-shared` build modes
- 🟡 **Platform** Linux: Requires kernel version 2.6.32 or later
- 🟡 **Platform** Windows: ARM and ARM64 ports now support non-cooperative preemption
- 🟡 **Platform** iOS: Requires iOS 12 or later (discontinued support for earlier versions)
- 🟢 **Platform** FreeBSD: Go 1.18 is last release supporting FreeBSD 11.x - Go 1.19 requires 12.2+ or 13.0+

## Implementation Details

- 🟢 **Compiler** Type checker replaced entirely to support generics - some error messages may differ
- 🟢 **Compiler** Compile speed roughly 15% slower than Go 1.17 due to generics support (to be improved)
- 🟢 **Bootstrap** Now looks for Go 1.17 bootstrap toolchain before falling back to Go 1.4

## Known Limitations (Generics)

- 🟢 **Language** Compiler cannot handle type declarations inside generic functions/methods
- 🟢 **Language** Predeclared functions `real`, `imag`, `complex` don't accept type parameter arguments
- 🟢 **Language** Can only call method `m` on type parameter `P` if `m` explicitly declared by `P`'s constraint
- 🟢 **Language** Cannot access struct field `x.f` where `x` is of type parameter type
- 🟢 **Language** Cannot embed type parameter or pointer to type parameter as unnamed struct field
- 🟢 **Language** Union with multiple terms cannot contain interface with non-empty method set

## Standard Library Highlights

### Major Changes

- 🔴 **crypto/tls** TLS 1.0 and 1.1 disabled by default client-side (still settable via `Config.MinVersion`)
- 🔴 **crypto/x509** Rejects SHA-1 signed certificates (except self-signed roots) - practical attacks demonstrated since 2017
- 🟡 **crypto/elliptic** P-224, P-384, P-521 now 4x faster and constant-time (backed by addchain/fiat-crypto)
- 🟡 **net** `net.Error.Temporary` deprecated
- 🟡 **os/user** `User.GroupIds` now uses Go native implementation when cgo unavailable

### New APIs

- 🟡 **bufio** `Writer.AvailableBuffer` returns buffer for append-like APIs
- 🟡 **bytes** `Cut` function slices around separator (simpler than Index/Split)
- 🟡 **strings** `Cut` and `Clone` functions added
- 🟡 **sync** `Mutex.TryLock`, `RWMutex.TryLock`, `RWMutex.TryRLock` methods
- 🟡 **reflect** `Value.SetIterKey`, `Value.SetIterValue`, `Value.UnsafePointer` methods
- 🟡 **reflect** `Value.CanInt`, `Value.CanUint`, `Value.CanFloat`, `Value.CanComplex` methods
- 🟡 **reflect** `Value.FieldByIndexErr` avoids panic when stepping through nil pointer
- 🟡 **html/template** New `{{break}}` and `{{continue}}` commands in range pipelines
- 🟡 **text/template** New `{{break}}` and `{{continue}}` commands in range pipelines
- 🟢 **unicode/utf8** `AppendRune` appends UTF-8 encoding of rune to `[]byte`

### Security & Compatibility

- 🟢 **crypto/rand** No longer buffers random data between calls - wrap in `bufio.Reader` if needed
- 🟢 **strconv** `Unquote` now rejects Unicode surrogate halves
- 🟢 **regexp** Now treats each invalid UTF-8 byte as U+FFFD
