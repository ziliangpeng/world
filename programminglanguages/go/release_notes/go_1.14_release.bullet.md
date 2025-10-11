# Go 1.14 Release Notes

**Released:** February 25, 2020
**EOL:** August 2021 (expected)

## Major Highlights

Go 1.14 brings significant performance improvements to defer, async preemption, and production-ready modules:

1. **Production-ready modules** - Module support ready for production use with resolved vendoring
2. **Overlapping interfaces** - Interfaces can embed other interfaces with overlapping method sets
3. **Near-zero defer overhead** - Defer now incurs almost zero overhead compared to direct calls
4. **Async preemption** - Goroutines preemptible without function calls, preventing scheduler deadlocks
5. **Page allocator improvements** - Better lock contention and throughput for parallel large allocations
6. **New hash/maphash package** - Fast non-cryptographic hash functions for hash tables

## Changes to the Language

- 🔴 **Language** Interfaces can embed overlapping method sets (same name/signature from different embedded interfaces)

## Breaking Changes

- 🟡 **WebAssembly** `js.Value` no longer comparable with `==` operator (use `Equal` method instead)

## Deprecations

- 🟡 **Platform** macOS 10.11 El Capitan is last supported release (10.12+ required in Go 1.15)
- 🟡 **Platform** darwin/386 (32-bit macOS) last supported (dropped in Go 1.15)
- 🟡 **Platform** darwin/arm (32-bit iOS/watchOS/tvOS) last likely supported (dropped in Go 1.15)
- 🟡 **Platform** Native Client (NaCl) support dropped

## New Features

- 🔴 **Modules** Production-ready module support - migrate from GOPATH encouraged
- 🔴 **Language** Overlapping interface embedding allowed
- 🔴 **hash/maphash** New package for fast, non-cryptographic byte sequence hashing
- 🟡 **go command** `-mod=vendor` now default when `vendor/` exists and `go.mod` specifies `go 1.14+`
- 🟡 **go command** `-mod=mod` forces module cache usage instead of vendoring
- 🟡 **go command** `-modcacherw` leaves new module cache directories writable (not read-only)
- 🟡 **go command** `-modfile=file` uses alternate go.mod file
- 🟡 **go command** `GOINSECURE` environment variable for insecure module fetching (HTTP, no cert validation)
- 🟡 **go test** `-v` streams `t.Log` output as it happens (not buffered)
- 🟡 **Compiler** `-d=checkptr` compile-time instrumentation for unsafe.Pointer safety checking
- 🟡 **Compiler** `-json` flag emits machine-readable optimization logs (inlining, escape analysis, bounds-check elimination)

## Improvements

- 🟢 **Runtime** Defer overhead nearly eliminated (almost zero compared to direct calls)
- 🟢 **Runtime** Async preemption for all platforms except windows/arm, darwin/arm, js/wasm, plan9/*
- 🟢 **Runtime** More signals on Unix (more EINTR errors possible - retry system calls)
- 🟢 **Runtime** Page allocator with less lock contention and better throughput for parallel large allocations
- 🟢 **Runtime** More efficient internal timers with less lock contention
- 🟢 **Compiler** Detailed escape analysis diagnostics (`-m=2`) restored
- 🟢 **Compiler** Bounds check elimination uses slice creation info and smaller-than-int index types
- 🟢 **Compiler** All macOS symbols now begin with underscore (platform convention)
- 🟡 **Compiler** Experimental coverage instrumentation for fuzzing (see issue 14565)

## Tooling & Developer Experience

- 🟡 **Modules** Vendoring improvements - `vendor/modules.txt` verified against `go.mod` with `-mod=vendor`
- 🟡 **Modules** `go list -m` fails explicitly with `-mod=vendor` for modules not in `vendor/modules.txt`
- 🟡 **Modules** `go get` no longer accepts `-mod` flag
- 🟡 **Modules** `-mod=readonly` default when go.mod is read-only without vendor/ directory
- 🟡 **Modules** `+incompatible` upgrades avoided unless explicitly requested
- 🟡 **Modules** `go.mod` editing reduced - only cosmetic changes avoided
- 🟡 **Modules** Subversion repository support in module mode
- 🟡 **Modules** Plain-text error messages from proxies included in error output

## Platform & Environment

- 🟡 **Platform** FreeBSD 12.0+ (arm64) now supported
- 🟡 **Platform** RISC-V 64-bit (linux/riscv64) experimental support
- 🟡 **Platform** WebAssembly: `js.Value` garbage collection, new `IsUndefined`/`IsNull`/`IsNaN` methods
- 🟡 **Platform** Windows: DEP (Data Execution Prevention) enabled in binaries
- 🟡 **Platform** Windows: File creation respects `0o200` permission bit (read-only without owner write)
- 🟡 **Platform** Illumos: Zone CPU caps now respected by `runtime.NumCPU` and `GOMAXPROCS`
- 🟡 **Platform** Native Client (NaCl) dropped (was announced in Go 1.13)

## Standard Library Changes

### Major Changes

- 🔴 **hash/maphash** New package for fast, collision-resistant, non-cryptographic hashing
- 🟡 **crypto/tls** SSLv3 support removed
- 🟡 **crypto/tls** TLS 1.3 no longer disableable via GODEBUG
- 🟡 **crypto/tls** Automatic certificate chain selection (ECDSA vs RSA)
- 🟡 **crypto/tls** `NameToCertificate` field deprecated (use automatic selection)
- 🟡 **crypto/tls** New `CipherSuites`/`InsecureCipherSuites`/`CipherSuiteName` functions
- 🟡 **crypto/tls** RSA-PSS signatures used in TLS 1.2 when supported

### net/http

- 🟡 **net/http** `Header.Values` returns all values for canonicalized key
- 🟡 **net/http** `Transport.DialTLSContext` replaces deprecated `DialTLS`
- 🟡 **net/http** Windows: `ServeFile` correctly serves files >2GB

### Other packages

- 🟡 **crypto/x509** `Certificate.CreateCRL` supports Ed25519 issuers
- 🟡 **debug/dwarf** DWARF version 5 support
- 🟡 **debug/dwarf** `Data.AddSection`, `Reader.ByteOrder`, `LineReader.Files` methods
- 🟡 **encoding/asn1** `Unmarshal` supports BMPString (new `TagBMPString`)
- 🟡 **encoding/json** `Decoder.InputOffset` returns current decoder position
- 🟡 **encoding/json** `Compact` no longer escapes U+2028/U+2029
- 🟡 **encoding/json** `Number` rejects invalid numbers (empty string)
- 🟡 **encoding/json** `Unmarshal` supports map keys with `encoding.TextUnmarshaler`
- 🟡 **go/build** `Context.Dir` sets working directory for build
- 🟡 **go/doc** `NewFromFiles` computes package docs from `*ast.File` list
- 🟡 **io/ioutil** `TempDir` supports `*` for random string in pattern
- 🟡 **log** New `Lmsgprefix` flag puts prefix immediately before log message
- 🟡 **math** `FMA` function for fused multiply-add with hardware acceleration
- 🟡 **math/big** `GCD` allows zero or negative inputs
- 🟡 **math/bits** `Rem`, `Rem32`, `Rem64` compute remainder when quotient overflows
- 🟡 **mime** `.js`/`.mjs` default type now `text/javascript` (was `application/javascript`)
- 🟡 **mime/multipart** `Reader.NextRawPart` fetches part without quoted-printable decoding
- 🟡 **net/http/httptest** `Server.EnableHTTP2` enables HTTP/2 on test server
- 🟡 **net/textproto** `MIMEHeader.Values` returns all values for canonicalized key
- 🟡 **net/url** Parse errors now quote unparsable URL
- 🟡 **os/signal** Windows: `CTRL_CLOSE_EVENT`/`CTRL_LOGOFF_EVENT`/`CTRL_SHUTDOWN_EVENT` generate `syscall.SIGTERM`
- 🟡 **plugin** FreeBSD/amd64 support
- 🟡 **reflect** `StructOf` supports unexported fields via `PkgPath` in `StructField`
- 🟡 **runtime** `Goexit` can no longer be aborted by recursive panic/recover
- 🟡 **runtime** macOS: `SIGPIPE` no longer forwarded to pre-Go signal handlers
- 🟡 **runtime/pprof** Profiles exclude inline mark pseudo-PCs (format regression fixed)
- 🟡 **strconv** `NumError.Unwrap` supports `errors.Is` checking for `ErrRange`/`ErrSyntax`
- 🟡 **sync** Highly contended `Mutex` unlocking yields CPU directly to next waiter
- 🟡 **testing** `T.Cleanup`/`B.Cleanup` register cleanup functions after test finishes
- 🟡 **text/template** Correctly reports errors for parenthesized function arguments
- 🟡 **text/template** `JSEscape` escapes `&` and `=` characters
- 🟡 **unicode** Upgraded from Unicode 11.0 to Unicode 12.0 (554 new characters, 4 new scripts, 61 new emoji)
