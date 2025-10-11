# Go 1.19 Release Notes

**Released:** August 2, 2022
**EOL:** August 2024

## Major Highlights

Go 1.19 delivers performance improvements, a revised memory model, and important security fixes for the PATH lookup behavior:

1. **Revised Go memory model** - Aligned with C, C++, Java, JavaScript, Rust, and Swift; clarifies atomic operations
2. **New atomic types in sync/atomic** - `Bool`, `Int32`, `Int64`, `Uint32`, `Uint64`, `Uintptr`, `Pointer[T]` for easier atomic operations
3. **Soft memory limit** - `GOMEMLIMIT` enables better control over memory usage without relying solely on `GOGC`
4. **PATH lookup security fix** - `os/exec` no longer searches current directory, preventing a common security vulnerability
5. **Doc comment improvements** - Support for links, lists, and clearer headings; `gofmt` now reformats doc comments
6. **Performance** - Jump tables for large switch statements (20%+ faster on amd64/arm64), faster file descriptor handling

## Breaking Changes

- 🔴 **os/exec** `Command` and `LookPath` no longer find binaries in current directory via PATH search - major security fix
- 🟡 **Language** Scope of type parameters in method declarations slightly corrected (existing programs unaffected)

## New Features

### Language & Runtime

- 🔴 **Memory Model** Revised to align with other languages - provides sequentially consistent atomics
- 🔴 **sync/atomic** New atomic types: `Bool`, `Int32`, `Int64`, `Uint32`, `Uint64`, `Uintptr`, `Pointer[T]`
- 🔴 **Runtime** Soft memory limit via `GOMEMLIMIT` or `runtime/debug.SetMemoryLimit` - works even with `GOGC=off`
- 🟡 **Runtime** GC CPU utilization limited to 50% when approaching soft memory limit to reduce thrashing
- 🟡 **Runtime** Fewer GC worker goroutines scheduled on idle threads during periodic GC
- 🟡 **Runtime** Initial goroutine stack size based on historic average to reduce early growth/copying
- 🟡 **Runtime** Unix: Go programs importing `os` now automatically increase `RLIMIT_NOFILE` to maximum

### Tooling & Developer Experience

- 🔴 **go command** Doc comments now support links, lists, and clearer headings (see "Go Doc Comments" guide)
- 🔴 **gofmt** Now reformats doc comments for better rendering
- 🔴 **go/doc/comment** New package for parsing/reformatting doc comments, rendering to HTML/Markdown/text
- 🟡 **go command** New `unix` build constraint satisfied on Unix/Unix-like systems
- 🟡 **go command** `-trimpath` flag now included in build settings stamped into binaries
- 🟡 **go list** `-json` accepts comma-separated field list to populate only specific fields
- 🟡 **vet** New "errorsas" checker reports `errors.As` with second argument of type `*error`

## Improvements

### Performance

- 🟢 **Compiler** Jump tables for large integer/string switch statements - up to 20% faster (amd64/arm64 only)
- 🟢 **Runtime** Timer and deadline code faster and scales better with high CPU counts
- 🟢 **Runtime** Goroutine stack traces now faster to collect with lower latency impact
- 🟢 **crypto/elliptic** `ScalarBaseMult` 3x faster on P224, P384, P521; generic P256 from formally verified model

### Error Messages & Debugging

- 🟢 **Runtime** Fatal errors now print simpler tracebacks (unless `GOTRACEBACK=system` or `crash`)
- 🟢 **Runtime** Debugger-injected function calls now supported on ARM64

## Platform & Environment

- 🟡 **Platform** LoongArch 64-bit: New port on Linux (`GOOS=linux`, `GOARCH=loong64`)
- 🟡 **Platform** RISC-V 64-bit: Now supports passing function arguments/results via registers - typical 10%+ improvement
- 🟡 **Platform** linux/arm64: Now supports race detector
- 🟡 **Platform** windows/amd64: Programs linking Go libraries can now use `SetUnhandledExceptionFilter`
- 🟡 **Platform** darwin/amd64: Generates PIE executables by default
- 🟢 **Platform** openbsd/ppc64: Experimental port

## Implementation Details

- 🟢 **Compiler** Requires `-p=importpath` flag to build linkable object files
- 🟢 **Compiler** No longer accepts `-importmap` flag - use `-importcfg` instead
- 🟢 **Assembler** Requires `-p=importpath` flag like compiler
- 🟢 **Linker** Emits compressed DWARF in standard gABI format (`SHF_COMPRESSED`) instead of legacy `.zdebug`
- 🟢 **Bootstrap** Now requires Go 1.17.13 for bootstrap

## Standard Library Highlights

### PATH Security Fix

- 🔴 **os/exec** `Command` and `LookPath` no longer search current directory in PATH
- 🔴 **os/exec** Windows: Respects `NoDefaultCurrentDirectoryInExePath` environment variable

### Atomic Operations

- 🔴 **sync/atomic** New atomic types hide underlying values, forcing use of atomic APIs
- 🔴 **sync/atomic** `Pointer[T]` avoids `unsafe.Pointer` conversion at call sites
- 🔴 **sync/atomic** `Int64`/`Uint64` automatically aligned to 64-bit boundaries even on 32-bit systems

### New APIs

- 🟡 **encoding/binary** `AppendByteOrder` interface with efficient append methods
- 🟡 **encoding/binary** `AppendUvarint` and `AppendVarint` functions
- 🟡 **encoding/csv** `Reader.InputOffset` reports current input position
- 🟡 **encoding/xml** `Decoder.InputPos` reports current line and column
- 🟡 **flag** `TextVar` defines flag with `encoding.TextUnmarshaler` value
- 🟡 **fmt** `Append`, `Appendf`, `Appendln` append formatted data to byte slices
- 🟡 **hash/maphash** `Bytes` and `String` provide efficient hashing for single values
- 🟡 **net/url** `JoinPath` and `URL.JoinPath` create URLs by joining path elements
- 🟡 **net/url** New `URL.OmitHost` field indicates empty authority
- 🟡 **sort** `Find` function like `Search` but returns boolean if value found
- 🟡 **time** `Duration.Abs` safely takes absolute value
- 🟡 **time** `Time.ZoneBounds` returns start/end of time zone in effect

### Deprecated

- 🟡 **crypto/tls** `GODEBUG=tls10default=1` removed - TLS 1.0 still settable via `Config.MinVersion`

### Security & Compatibility

- 🟢 **net** I/O timeout errors now satisfy `errors.Is(err, context.DeadlineExceeded)`
- 🟢 **net** Operation canceled errors now satisfy `errors.Is(err, context.Canceled)`
- 🟢 **crypto/rand** `Read` no longer buffers; `Prime` changed to use only rejection sampling
- 🟢 **crypto/tls** Client and server reject duplicate extensions in TLS handshakes
- 🟢 **crypto/x509** `CreateCertificate` no longer supports MD5WithRSA, rejects negative serial numbers
- 🟢 **crypto/x509** Path builder overhauled for better chains in complicated scenarios
- 🟢 **runtime/race** Upgraded to thread sanitizer v3 on most platforms - 1.5-2x faster, half memory, unlimited goroutines
