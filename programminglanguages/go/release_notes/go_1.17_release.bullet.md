# Go 1.17 Release Notes

**Released:** August 16, 2021
**EOL:** February 2023 (expected)

## Major Highlights

Go 1.17 brings register-based calling convention, module graph pruning, and language enhancements:

1. **Register-based calling** - Function args/results passed via registers (~5% performance improvement, ~2% smaller binaries)
2. **Module graph pruning** - `go 1.17` modules only track immediate dependencies of other go 1.17 modules
3. **Slice to array pointer conversion** - New type conversion from `[]T` to `*[N]T`
4. **unsafe additions** - New `unsafe.Add` and `unsafe.Slice` functions
5. **//go:build lines** - New build constraint syntax preferred over `// +build`

## Changes to the Language

- 🔴 **Language** Conversions from slice to array pointer: `s` of type `[]T` → `*[N]T` (panics if `len(s) < N`)
- 🔴 **Language** `unsafe.Add(ptr, len)` adds offset to pointer
- 🔴 **Language** `unsafe.Slice(ptr, len)` creates slice from pointer

## Breaking Changes

- 🟡 **Compiler** Register-based calling may affect unsafe code or undocumented behavior (adapters created for compatibility)

## New Features

- 🔴 **Modules** Pruned module graphs for `go 1.17+` modules
- 🔴 **Modules** Lazy loading - `go.mod` files not read/downloaded unless needed
- 🔴 **Modules** Module deprecation comments with `// Deprecated:` in `go.mod`
- 🔴 **//go:build** New build constraint syntax (boolean expressions like Go)
- 🟡 **go mod tidy** `-go` flag sets/changes `go` version in `go.mod`
- 🟡 **go mod tidy** `-compat` flag overrides compatibility version checking
- 🟡 **go mod graph** `-go` flag reports graph as seen by specified Go version
- 🟡 **go run** Accepts arguments with version suffixes (e.g., `go run example.com/cmd@v1.0.0`)
- 🟡 **gofmt** Automatically synchronizes `//go:build` with `// +build` lines
- 🟡 **Vet** Verifies `//go:build` and `// +build` are correct and synchronized
- 🟡 **Vet** Warns about `signal.Notify` with unbuffered channels
- 🟡 **Vet** Warns about `Is`/`As`/`Unwrap` methods with wrong signatures on error types
- 🟡 **Cover** Uses optimized parser (noticeably faster for large profiles)

## Improvements

- 🟢 **Compiler** Register-based calling: ~5% performance improvement, ~2% smaller binaries (linux/amd64, darwin/amd64, windows/amd64)
- 🟢 **Compiler** Stack traces show individual function arguments (not hex words)
- 🟢 **Compiler** Functions with closures now inlinable
- 🟢 **Runtime** `crypto/ed25519` rewritten - ~2x faster on amd64 and arm64
- 🟢 **Runtime** `crypto/elliptic` P-521 rewritten with fiat-crypto - constant-time, 3x faster

## Tooling & Developer Experience

- 🟡 **Modules** Indirect dependencies in separate `require` block in `go 1.17` modules
- 🟡 **Modules** `go mod vendor` annotates `vendor/modules.txt` with `go` versions
- 🟡 **Modules** `go mod vendor` omits `go.mod`/`go.sum` for vendored dependencies
- 🟡 **go command** Suppresses SSH and Git Credential Manager password prompts
- 🟡 **go mod download** No longer saves sums without arguments (use `go mod download all`)
- 🟡 **go get** `-insecure` flag removed (use `GOINSECURE`)
- 🟡 **go get** Deprecation warning for installing commands without `-d` flag
- 🟡 **go.mod** Missing `go` directive assumes `go 1.11`
- 🟡 **go.mod** Missing `go` directive in dependencies assumes `go 1.16`

## Platform & Environment

- 🟡 **Platform** macOS 10.13 High Sierra+ required (10.12 dropped as announced in Go 1.16)
- 🟡 **Platform** Windows: windows/arm64 support with cgo
- 🟡 **Platform** OpenBSD: openbsd/mips64 now supports cgo
- 🟡 **Platform** OpenBSD: System calls via `libc` on 386 and arm
- 🟡 **Platform** ARM64: Stack frame pointers maintained on all operating systems
- 🟡 **Platform** loong64 GOARCH value reserved for future LoongArch support

## Standard Library Changes

### Major Changes

- 🔴 **runtime/cgo** New `Handle` type for safely passing Go values to C
- 🔴 **net/url, net/http** Semicolons no longer accepted as query separators (use `AllowQuerySemicolons` wrapper to restore)
- 🔴 **crypto/tls** Strict ALPN enforcement - connection closed if no overlap between client/server protocols
- 🟡 **compress/lzw** New `Reader` and `Writer` types with `Reset` methods

### archive/zip

- 🟡 **archive/zip** `File.OpenRaw`, `Writer.CreateRaw`, `Writer.Copy` for performance-critical cases

### bytes/strings

- 🟡 **bufio** `Writer.WriteRune` writes U+FFFD for negative runes
- 🟡 **bytes** `Buffer.WriteRune` writes U+FFFD for negative runes
- 🟡 **strings** `Builder.WriteRune` writes U+FFFD for negative runes

### crypto

- 🟡 **crypto/ed25519** Rewritten - ~2x faster on amd64/arm64
- 🟡 **crypto/elliptic** `CurveParams` automatically invokes faster dedicated implementations (P-224, P-256, P-521)
- 🟡 **crypto/elliptic** P-521 rewritten using fiat-crypto - constant-time, 3x faster
- 🟡 **crypto/rand** Uses `getentropy` on macOS, `getrandom` on Solaris/Illumos/DragonFlyBSD
- 🟡 **crypto/tls** `Conn.HandshakeContext` for cancelable handshakes
- 🟡 **crypto/tls** Cipher suite ordering handled entirely by crypto/tls (ignores `Config.CipherSuites` order)
- 🟡 **crypto/tls** 3DES moved to `InsecureCipherSuites` (still enabled as last resort)
- 🟡 **crypto/tls** Go 1.18 will default `Config.MinVersion` to TLS 1.2
- 🟡 **crypto/x509** `CreateCertificate` returns error if private key doesn't match parent's public key
- 🟡 **crypto/x509** `GODEBUG=x509ignoreCN=0` removed
- 🟡 **crypto/x509** `ParseCertificate` rewritten - ~70% fewer resources
- 🟡 **crypto/x509** BSD: `/etc/ssl/certs` searched for trusted roots
- 🟡 **crypto/x509** Go 1.18 will reject SHA-1 certificates (except self-signed roots)

### database/sql

- 🟡 **database/sql** `DB.Close` closes connector if it implements `io.Closer`
- 🟡 **database/sql** New `NullInt16` and `NullByte` structs

### encoding

- 🟡 **encoding/binary** `Uvarint` stops reading after 10 bytes (returns -11 if more needed)
- 🟡 **encoding/csv** `Reader.FieldPos` returns line/column of field start
- 🟡 **encoding/xml** Comments within `Directive` replaced with space (not elided)
- 🟡 **encoding/xml** Invalid element/attribute names with colons stored unmodified in `Name.Local`

### flag

- 🟡 **flag** Declarations panic on invalid names

### go

- 🟡 **go/build** New `Context.ToolTags` field
- 🟡 **go/format** Synchronizes `//go:build` with `// +build` lines
- 🟡 **go/parser** New `SkipObjectResolution` Mode for faster parsing

### image

- 🟡 **image** Concrete types implement `RGBA64Image` interface
- 🟡 **image/draw** Concrete types implement `RGBA64Image` interface

### io/fs

- 🟡 **io/fs** New `FileInfoToDirEntry` converts `FileInfo` to `DirEntry`

### math

- 🟡 **math** New constants `MaxUint`, `MaxInt`, `MinInt`

### mime

- 🟡 **mime** Reads MIME types from Shared MIME-info Database on Unix

### mime/multipart

- 🟡 **mime/multipart** `Part.FileName` applies `filepath.Base` to mitigate path traversal

### net

- 🟡 **net** `IP.IsPrivate` reports private IPv4/IPv6 addresses
- 🟡 **net** DNS resolver sends single query for IPv4-only or IPv6-only networks
- 🟡 **net** `ErrClosed` and `ParseError` implement `net.Error`
- 🟡 **net** `ParseIP`/`ParseCIDR` reject IPv4 with leading zeros in decimal components

### net/http

- 🟡 **net/http** Uses `(*tls.Conn).HandshakeContext` with Request context
- 🟡 **net/http** Negative `ReadTimeout`/`WriteTimeout` indicates no timeout
- 🟡 **net/http** `ReadRequest` errors on multiple Host headers
- 🟡 **net/http** `ServeMux` uses relative URLs in `Location` headers
- 🟡 **net/http** Non-ASCII characters ignored/rejected in certain headers
- 🟡 **net/http** `ParseMultipartForm` continues populating `MultipartForm` despite `ParseForm` errors
- 🟡 **net/http/httptest** `ResponseRecorder.WriteHeader` panics on invalid status codes

### net/url

- 🟡 **net/url** `Values.Has` reports if query parameter is set

### os

- 🟡 **os** `File.WriteString` optimized (no copy of input string)

### reflect

- 🟡 **reflect** `Value.CanConvert` reports if value convertible to type
- 🟡 **reflect** `StructField.IsExported` and `Method.IsExported` check if exported
- 🟡 **reflect** New `VisibleFields` returns all visible struct fields
- 🟡 **reflect** `ArrayOf` panics on negative length
- 🟡 **reflect** `Value.Convert` may panic on `[]T` to `*[N]T` if slice too short

### runtime/metrics

- 🟡 **runtime/metrics** New metrics for total bytes/objects allocated/freed and goroutine scheduling latencies

### runtime/pprof

- 🟡 **runtime/pprof** Block profiles no longer biased toward infrequent long events

### strconv

- 🟡 **strconv** Uses Ulf Adams's Ryū algorithm (~99% faster on worst-case inputs)
- 🟡 **strconv** New `QuotedPrefix` returns quoted string at start of input

### sync/atomic

- 🟡 **sync/atomic** `atomic.Value` has `Swap` and `CompareAndSwap` methods

### syscall

- 🟡 **syscall** Windows: `GetQueuedCompletionStatus`/`PostQueuedCompletionStatus` deprecated
- 🟡 **syscall** Unix: Process group set with signals blocked
- 🟡 **syscall** Windows: `SysProcAttr` has `AdditionalInheritedHandles` and `ParentProcess` fields
- 🟡 **syscall** `MSG_CMSG_CLOEXEC` defined on DragonFly and all OpenBSD
- 🟡 **syscall** `SYS_WAIT6` and `WEXITED` defined on NetBSD

### testing

- 🟡 **testing** New `-shuffle` flag for randomized test/benchmark execution
- 🟡 **testing** `T.Setenv`/`B.Setenv` set environment variable for test duration

### text/template/parse

- 🟡 **text/template/parse** New `SkipFuncCheck` Mode skips function definition verification

### time

- 🟡 **time** `Time.GoString` returns better value for `%#v` format
- 🟡 **time** `Time.IsDST` checks if time is in Daylight Savings Time
- 🟡 **time** `Time.UnixMilli`/`Time.UnixMicro` return milliseconds/microseconds since Unix epoch
- 🟡 **time** `UnixMilli`/`UnixMicro` create Time from Unix milliseconds/microseconds
- 🟡 **time** Accepts comma as fractional seconds separator
- 🟡 **time** New `Layout` constant defines reference time

### unicode

- 🟡 **unicode** Character checking functions return false for negative runes
