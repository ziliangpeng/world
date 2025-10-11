# Go 1.13 Release Notes

**Released:** September 3, 2019
**EOL:** February 2021 (expected)

## Major Highlights

Go 1.13 modernizes number literals, improves error handling, and adds the Go module mirror:

1. **Modern number literals** - Binary (`0b`), octal (`0o`), hex float literals, digit separators with underscores
2. **Error wrapping** - New `%w` verb in `fmt.Errorf` and `errors.Is`/`As`/`Unwrap` functions
3. **Go module mirror** - Default use of proxy.golang.org and sum database for authenticated modules
4. **Signed shift counts** - Shift operators no longer require unsigned shift counts
5. **New crypto/ed25519 package** - Ed25519 signature scheme moved to standard library
6. **Faster defer** - 30% performance improvement for most defer uses

## Changes to the Language

- 🔴 **Language** Binary integer literals with `0b`/`0B` prefix (e.g., `0b1011`)
- 🔴 **Language** Octal integer literals with `0o`/`0O` prefix (e.g., `0o660`) - old `0` prefix still valid
- 🔴 **Language** Hexadecimal floating-point literals (e.g., `0x1.0p-1021`)
- 🔴 **Language** Imaginary suffix `i` now works with any number literal
- 🔴 **Language** Digit separators with underscores (e.g., `1_000_000`, `0b_1010_0110`)
- 🔴 **Language** Signed shift counts allowed - removes restriction that shift must be unsigned

## Breaking Changes

- 🟡 **Modules** Go module mirror and checksum database used by default (see privacy info at proxy.golang.org/privacy)

## New Features

- 🔴 **Modules** Module mirror at proxy.golang.org and checksum database enabled by default
- 🔴 **Modules** `GOPRIVATE` environment variable for non-public modules
- 🔴 **Modules** `GOPROXY` supports comma-separated list of proxies with fallback
- 🔴 **Modules** `GOSUMDB` for checksum database configuration
- 🔴 **Error Handling** Error wrapping with `fmt.Errorf` `%w` verb
- 🔴 **Error Handling** `errors.Is` checks if error matches value in chain
- 🔴 **Error Handling** `errors.As` finds first error in chain matching target type
- 🔴 **Error Handling** `errors.Unwrap` returns wrapped error
- 🔴 **crypto/ed25519** Ed25519 signature scheme now in standard library
- 🟡 **go command** `go env -w` sets per-user defaults, `-u` unsets them
- 🟡 **go command** `go version` accepts executables and directories, shows embedded module info with `-m`
- 🟡 **go command** `-trimpath` removes file system paths for reproducible builds
- 🟡 **go command** `-o` to existing directory writes executables within it
- 🟡 **go command** `-tags` accepts comma-separated lists
- 🟡 **go command** `go generate` sets `generate` build tag
- 🟡 **go get** `@patch` version suffix for highest patch with same major/minor
- 🟡 **go get** `@upgrade` explicitly requests upgrade behavior, `@latest` forces latest regardless
- 🟡 **gofmt** Number literals canonicalized to lowercase (e.g., `0O` → `0o`)
- 🟡 **gofmt** Removes unnecessary leading zeroes from decimal imaginary literals

## Improvements

- 🟢 **Runtime** Out of range panics include index and length/capacity
- 🟢 **Runtime** 30% faster defer for most uses
- 🟢 **Runtime** More aggressive memory return to OS after heap shrinks
- 🟢 **Compiler** New escape analysis implementation - more precise, more stack allocation
- 🟡 **Compiler** No longer emits floating point or complex constants to `go_asm.h`
- 🟢 **Assembler** ARM v8.1 atomic instructions supported

## Tooling & Developer Experience

- 🟡 **Modules** Version validation for pseudo-versions and `+incompatible` versions
- 🟡 **Modules** `-u` flag in `go get` updates smaller, more consistent set of modules
- 🟡 **Modules** `go get` no longer supports `-m` flag
- 🟡 **Modules** `go get -u` includes test dependencies with `-t` flag
- 🟡 **Binary-only** Binary-only packages no longer supported (marked with `//go:binary-only-package`)

## Platform & Environment

- 🟡 **Platform** Native Client (NaCl) is last supported (removed in Go 1.14)
- 🟡 **Platform** WebAssembly: `GOWASM` environment variable for experimental features
- 🟡 **Platform** AIX: cgo, external linking, `c-archive` and `pie` build modes supported
- 🟡 **Platform** Android: Go programs compatible with Android 10
- 🟡 **Platform** Darwin: macOS 10.11 El Capitan+ required (10.10 dropped)
- 🟡 **Platform** FreeBSD: 11.2+ required (10.x dropped)
- 🟡 **Platform** Illumos: Now supported with `GOOS=illumos` (implies `solaris` build tag)
- 🟡 **Platform** Windows: Binaries specify Windows 7 minimum (was NT 4.0)

## Standard Library Changes

### Major Changes

- 🔴 **crypto/ed25519** New package for Ed25519 signature scheme
- 🔴 **errors** Error wrapping support with `Is`, `As`, and `Unwrap` functions
- 🔴 **fmt** `Errorf` with `%w` verb creates wrapped errors
- 🟡 **fmt** `%x`/`%X` format floating-point and complex in hexadecimal
- 🟡 **fmt** New `%O` verb formats integers in base 8 with `0o` prefix
- 🟡 **fmt** Scanner accepts hex floats, digit separators, `0b`/`0o` prefixes

### net/http

- 🟡 **net/http** `Transport.WriteBufferSize` and `ReadBufferSize` control buffer sizes
- 🟡 **net/http** `Transport.ForceAttemptHTTP2` controls HTTP/2 with custom dial functions
- 🟡 **net/http** `Transport.MaxConnsPerHost` works with HTTP/2
- 🟡 **net/http** `TimeoutHandler`'s `ResponseWriter` implements `Pusher`
- 🟡 **net/http** Status code 103 "Early Hints" added
- 🟡 **net/http** `Transport` uses `Request.Body`'s `io.ReaderFrom` if available
- 🟡 **net/http** Server returns "501 Unimplemented" for unsupported transfer-encodings
- 🟡 **net/http** `Server.BaseContext` and `ConnContext` for finer Context control
- 🟡 **net/http** `DetectContentType` detects RAR v5 signatures
- 🟡 **net/http** `Header.Clone` returns copy of receiver
- 🟡 **net/http** `NewRequestWithContext` creates request with context control
- 🟡 **net/http** No longer logs errors when servers gracefully close with "408 Request Timeout"

### Other packages

- 🟡 **bytes** `ToValidUTF8` replaces invalid UTF-8 with replacement slice
- 🟡 **context** `WithValue` formatting no longer depends on `fmt`
- 🟡 **crypto/tls** SSLv3 deprecated (removed in Go 1.14)
- 🟡 **crypto/tls** Ed25519 certificates supported in TLS 1.2 and 1.3
- 🟡 **crypto/x509** Ed25519 keys supported in certificates and requests (RFC 8410)
- 🟡 **crypto/x509** `/etc/ssl/cert.pem` searched for Alpine Linux 3.7+
- 🟡 **database/sql** `NullTime` represents nullable `time.Time`
- 🟡 **database/sql** `NullInt32` represents nullable `int32`
- 🟡 **debug/dwarf** `Data.Type` no longer panics on unknown tags (returns `UnsupportedType`)
- 🟡 **go/scanner** Recognizes new number literals (binary, octal, hex floats, digit separators)
- 🟡 **go/types** Type-checker follows new signed shift count rules
- 🟡 **html/template** `<script type="module">` interpreted as JavaScript module script
- 🟡 **log** `Writer` returns output destination for standard logger
- 🟡 **math/big** `Rat.SetUint64` sets Rat to uint64 value
- 🟡 **math/big** `Float.Parse`/`Int.SetString` accept underscores (base 0)
- 🟡 **math/big** `Rat.SetString` accepts non-decimal floating point
- 🟡 **math/bits** `Add`, `Sub`, `Mul`, `RotateLeft`, `ReverseBytes` have constant-time execution
- 🟡 **net** TCP used for DNS resolution on Unix when `use-vc` in resolv.conf
- 🟡 **net** `ListenConfig.KeepAlive` specifies keep-alive period (0 enables, negative disables)
- 🟡 **os** `UserConfigDir` returns default user config directory
- 🟡 **os** `File` opened with O_APPEND has `WriteAt` always return error
- 🟡 **os/exec** Windows: `%SYSTEMROOT%` always inherited unless explicitly set
- 🟡 **reflect** `Value.IsZero` reports if value is zero for its type
- 🟡 **reflect** `MakeFunc` allows assignment conversions on returned values
- 🟡 **runtime** Tracebacks refer to `PKG.init` instead of `PKG.init.ializers`
- 🟡 **strconv** `ParseFloat`/`ParseInt`/`ParseUint` accept underscores (base 0)
- 🟡 **strings** `ToValidUTF8` replaces invalid UTF-8 with replacement string
- 🟡 **sync** `Mutex`/`RWMutex` methods inlined for uncontended cases (up to 10% faster)
- 🟡 **sync** Large `Pool` no longer increases stop-the-world pause times
- 🟡 **sync** `Pool` retains some objects across GCs (reduces load spikes)
- 🟡 **syscall** Darwin: `_getdirentries64` removed for App Store compatibility
- 🟡 **syscall** Windows: `ProcessAttributes` and `ThreadAttributes` in `SysProcAttr`
- 🟡 **syscall** Windows: `EINVAL` no longer returned for zero `Chmod` mode
- 🟡 **syscall** `Errno` testable with `os` package errors using `errors.Is`
- 🟡 **syscall/js** `TypedArrayOf` replaced by `CopyBytesToGo`/`CopyBytesToJS`
- 🟡 **testing** `B.N` no longer rounded when running benchmarks
- 🟡 **testing** `B.ReportMetric` reports custom metrics and overrides built-in metrics
- 🟡 **testing** `Init` function registers testing flags (invoked by generated main)
- 🟡 **text/scanner** Recognizes new number literals with `AllowDigitSeparators` mode
- 🟡 **text/template** New `slice` function for slicing arrays/slices/strings
- 🟡 **time** Day-of-year supported by `Format` and `Parse`
- 🟡 **time** `Duration.Microseconds` and `Milliseconds` methods
- 🟡 **unicode** Upgraded from Unicode 10.0 to Unicode 11.0 (684 new characters, 7 new scripts)
