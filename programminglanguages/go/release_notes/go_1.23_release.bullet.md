# Go 1.23 Release Notes

**Released:** August 13, 2024
**EOL:** August 2026

## Major Highlights

Go 1.23 brings iterators to the language, improves Timer behavior, and introduces the unique package:

1. **Range over function iterators** - Functions can now be used as range expressions, enabling custom iteration patterns
2. **New iter package** - Defines iterator types `Seq[V]` and `Seq2[K,V]` for single and paired value iteration
3. **New unique package** - Canonicalization (interning/hash-consing) for comparable values with efficient Handle types
4. **Timer changes** - Timers/Tickers now GC-eligible when unreferenced; channels now unbuffered (capacity 0)
5. **Telemetry** - Opt-in usage and breakage statistics collection to help improve Go
6. **PGO improvements** - Reduced build time overhead to single-digit percentages (was 100%+ for large builds)
7. **New structs package** - Types for modifying struct properties like memory layout

## Breaking Changes

- 🔴 **time** Timers now GC-eligible immediately even if Stop not called (GODEBUG: `asynctimerchan=1`)
- 🔴 **time** Timer channels now unbuffered (capacity 0) - stale values no longer sent after Reset/Stop (GODEBUG: `asynctimerchan=1`)
- 🟡 **go command** `GOROOT_FINAL` environment variable no longer has effect

## New Features

### Language Features

- 🔴 **Language** Range over function iterators - functions of type `func(func() bool)`, `func(func(K) bool)`, `func(func(K,V) bool)` usable in for-range
- 🟡 **Language** Preview: Generic type aliases (enable with `GOEXPERIMENT=aliastypeparams`)

### New Packages

- 🔴 **iter** Defines `Seq[V]` and `Seq2[K,V]` iterator types for range-over-func
- 🔴 **unique** Canonicalization via `Make[T]` producing `Handle[T]` - efficient deduplication and comparison
- 🔴 **structs** `HostLayout` marker type ensures struct layout conforms to host platform expectations

### Tooling & Developer Experience

- 🔴 **go command** Telemetry: Opt-in via `go telemetry on` for anonymous usage statistics
- 🟡 **go env** `-changed` flag prints only settings differing from defaults
- 🟡 **go mod tidy** `-diff` flag prints changes as unified diff without modifying files
- 🟡 **go list** `-m -json` includes new `Sum` and `GoModSum` fields
- 🟡 **go.mod** New `godebug` directive declares GODEBUG settings for module/workspace
- 🟡 **vet** New `stdversion` analyzer flags references to symbols too new for Go version
- 🟡 **cgo** `-ldflags` flag for passing flags to C linker (avoids "argument list too long")
- 🟡 **trace** Tool now recovers from partially broken traces (e.g., during program crash)

## Improvements

### Performance

- 🟢 **Compiler** PGO build time overhead reduced to single-digit percentages (was 100%+ for large builds)
- 🟢 **Compiler** Stack frame slot overlapping reduces stack usage
- 🟢 **Compiler** PGO-based hot block alignment improves performance 1-1.5% (amd64/386 only)
- 🟢 **Runtime** 2-3% CPU overhead decrease on average (new Swiss Tables map, efficient small object allocation, new mutex)
- 🟢 **Runtime** Swiss Tables map and new mutex can be disabled with `GOEXPERIMENT=noswissmap,nospinbitmutex`

### Error Messages & Debugging

- 🟢 **Runtime** Panic tracebacks indent error message by tab for clarity

## Platform & Environment

- 🔴 **Platform** macOS: Requires 11 Big Sur or later (discontinued 10.15 Catalina support)
- 🟡 **Platform** Linux: Go 1.23 is last requiring kernel 2.6.32 - Go 1.24 requires 3.2+
- 🟡 **Platform** OpenBSD: Experimental support for 64-bit RISC-V (`GOOS=openbsd`, `GOARCH=riscv64`)
- 🟡 **Platform** ARM64: New `GOARM64` environment variable specifies minimum target (v8.0-v9.5, with `,lse` or `,crypto` options)
- 🟡 **Platform** RISC-V: New `GORISCV64` environment variable selects profile (`rva20u64` or `rva22u64`)

## Implementation Details

- 🟢 **Linker** Disallows `//go:linkname` to internal stdlib symbols without explicit marking (backward compatible exceptions)
- 🟢 **Linker** `-bindnow` flag enables immediate function binding for dynamically linked ELF binaries

## Standard Library Highlights

### Iterator Support

- 🔴 **slices** Iterator functions: `All`, `Values`, `Backward`, `Collect`, `AppendSeq`, `Sorted`, `SortedFunc`, `SortedStableFunc`, `Chunk`
- 🔴 **maps** Iterator functions: `All`, `Keys`, `Values`, `Insert`, `Collect`

### New APIs

- 🟡 **crypto/tls** Encrypted Client Hello (ECH) client support via `Config.EncryptedClientHelloConfigList`
- 🟡 **crypto/tls** Post-quantum X25519Kyber768Draft00 enabled by default (disable with `GODEBUG=tlskyber=0`)
- 🟡 **crypto/tls** 3DES cipher suites removed from defaults (revert with `GODEBUG=tls3des=1`)
- 🟡 **crypto/tls** `X509KeyPair` and `LoadX509KeyPair` now populate `Certificate.Leaf` (GODEBUG: `x509keypairleaf`)
- 🟡 **crypto/x509** `ParseOID` parses dot-encoded ASN.1 OID strings
- 🟡 **crypto/x509** `OID` implements `encoding.BinaryMarshaler/Unmarshaler` and `encoding.TextMarshaler/Unmarshaler`
- 🟡 **encoding/binary** `Encode`, `Decode`, and `Append` functions as byte slice equivalents to Read/Write
- 🟡 **go/ast** `Preorder` returns iterator over all syntax tree nodes
- 🟡 **math/rand/v2** `Uint` function and `Rand.Uint` method added (inadvertently omitted in Go 1.22)
- 🟡 **math/rand/v2** `ChaCha8.Read` implements `io.Reader`
- 🟡 **net** `KeepAliveConfig` for fine-tuned TCP keep-alive via `TCPConn.SetKeepAliveConfig`
- 🟡 **net** `DNSError` now wraps timeout/cancellation errors
- 🟡 **net/http** `Cookie.Quoted` indicates if value was originally quoted
- 🟡 **net/http** `Request.CookiesNamed` retrieves all cookies matching name
- 🟡 **net/http** `Cookie.Partitioned` identifies cookies with Partitioned attribute
- 🟡 **net/http** `ParseCookie` and `ParseSetCookie` for parsing cookie headers
- 🟡 **reflect** `Value.Seq` and `Value.Seq2` return sequences for range iteration
- 🟡 **slices** `Repeat` returns slice repeating provided slice N times
- 🟡 **sync** `Map.Clear` deletes all entries
- 🟡 **sync/atomic** `And` and `Or` operators for bitwise operations
- 🟡 **unicode/utf16** `RuneLen` returns number of 16-bit words in UTF-16 encoding

### Security & Compatibility

- 🔴 **crypto/x509** `x509sha1` GODEBUG setting will be removed in Go 1.24
- 🟡 **database/sql** Errors from `driver.Valuer` now wrapped for better error handling
- 🟡 **net** GODEBUG `netedns0=0` disables EDNS0 headers (breaks some modems)
- 🟡 **os** Stat sets `ModeSocket` for Unix sockets on Windows
- 🟡 **os** Windows mode bits for reparse points changed (GODEBUG: `winsymlink`)
- 🟡 **os** `CopyFS` copies `io/fs.FS` into local filesystem
- 🟡 **os** Linux with pidfd support: Process operations use pidfd internally
- 🟡 **path/filepath** `Localize` safely converts slash-separated path to OS path
- 🟡 **time** `Parse` and `ParseInLocation` return error if time zone offset out of range

### Performance

- 🟡 **time** Windows: Timer/Ticker/Sleep resolution improved to 0.5ms (was 15.6ms)
