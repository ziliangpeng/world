# Go 1.15 Release Notes

**Released:** August 11, 2020
**EOL:** February 2022 (expected)

## Major Highlights

Go 1.15 brings substantial linker improvements, better allocation performance, and X.509 CommonName deprecation:

1. **Linker improvements** - 20% faster linking, 30% less memory, smaller binaries
2. **Small object allocation** - Much better performance at high core counts with lower latency
3. **X.509 CommonName deprecation** - No longer treated as hostname (use `x509ignoreCN=0` to revert temporarily)
4. **Embedded tzdata** - New `time/tzdata` package embeds timezone database into programs
5. **GOPROXY error fallback** - Pipe character (`|`) separator for fallback on any proxy error

## Breaking Changes

- 🔴 **crypto/x509** CommonName no longer treated as hostname when Subject Alternative Names absent (set `GODEBUG=x509ignoreCN=0` temporarily)
- 🟡 **unsafe** Chained `unsafe.Pointer` to `uintptr` conversions no longer allowed in `syscall.Syscall` calls

## New Features

- 🔴 **time/tzdata** New embedded timezone database package (~800 KB increase when imported)
- 🔴 **Linker** Substantial improvements: 20% faster, 30% less memory, more aggressive symbol pruning
- 🟡 **GOPROXY** Pipe separator (`|`) for fallback on any error (comma `,` only on 404/410)
- 🟡 **go test** Changing `-timeout` now invalidates cached test results
- 🟡 **go test** Flag parsing issues fixed, `GOFLAGS` handled consistently
- 🟡 **go command** `GOMODCACHE` environment variable sets module cache location
- 🟡 **Vet** New warning for `string(x)` where `x` is integer type other than `rune`/`byte`
- 🟡 **Vet** New warning for impossible interface type assertions
- 🟡 **Compiler** `-spectre` flag for Spectre mitigations (rarely needed)
- 🟡 **Compiler** Misplaced `//go:` directives now rejected with error
- 🟡 **Compiler** `-json` optimization logging includes large copy explanations
- 🟡 **Objdump** `-gnu` flag for GNU assembler syntax disassembly

## Improvements

- 🟢 **Runtime** Small object allocation performs much better at high core counts
- 🟢 **Runtime** Converting small integers to interface values no longer causes allocation
- 🟢 **Runtime** Non-blocking receives on closed channels as fast as on open channels
- 🟢 **Runtime** Panic with derived type values now prints value instead of just address
- 🟢 **Compiler** ~5% binary size reduction via eliminated GC metadata and unused type metadata
- 🟢 **Compiler** Intel CPU erratum SKX102 mitigated on amd64 (32-byte function alignment)
- 🟢 **Compiler** Inlining works for functions with non-labeled `for` loops, method values, type switches
- 🟢 **Linker** Improvements extend to all supported arch/OS (1.14 focused on ELF/amd64)
- 🟢 **Linker** Object file format redesigned, internal phases parallelized

## Tooling & Developer Experience

- 🟡 **Vet** `string(x)` warning enabled by default - may become language error in future
- 🟡 **Vet** Impossible interface assertion warning - may become language error in future
- 🟡 **go command** Module cache workaround for Windows "Access denied" with `GODEBUG=modcacheunzipinplace=1`
- 🟡 **Linker** Internal linking default for `-buildmode=pie` on linux/amd64 and linux/arm64

## Platform & Environment

- 🟡 **Platform** macOS 10.12 Sierra+ required (10.11 dropped as announced in Go 1.14)
- 🟡 **Platform** darwin/386 (32-bit macOS) dropped as announced in Go 1.14
- 🟡 **Platform** darwin/arm (32-bit iOS) likely last supported
- 🟡 **Platform** OpenBSD 6.7 arm/arm64 support
- 🟡 **Platform** RISC-V progress: stability, performance, async preemption
- 🟡 **Platform** 386: x87 floating-point (GO386=387) last supported - SSE2 required in Go 1.16
- 🟡 **Platform** Windows: ASLR executables default with `-buildmode=pie`
- 🟡 **Platform** Windows: `-race`/`-msan` now enable `-d=checkptr`
- 🟡 **Platform** Windows: Go-built DLLs no longer exit process on signals
- 🟡 **Platform** Android: `lld` linker explicitly selected in NDK for better stability
- 🟡 **cgo** `EGLConfig` translated to `uintptr`
- 🟡 **cgo** Allocating undefined struct types on stack/heap forbidden (Go 1.15.3+)

## Standard Library Changes

### Major Changes

- 🔴 **time/tzdata** New embedded timezone database package
- 🔴 **crypto/x509** CommonName deprecated for hostname verification
- 🟡 **crypto/rsa** `PrivateKey`/`PublicKey` have `Equal` methods
- 🟡 **crypto/ecdsa** `SignASN1`/`VerifyASN1` for standard ASN.1 DER signatures
- 🟡 **crypto/elliptic** `MarshalCompressed`/`UnmarshalCompressed` for compressed point format

### net/http

- 🟡 **net/http** `ReverseProxy` no longer modifies `X-Forwarded-For` when incoming map entry is nil
- 🟡 **net/http** `ReverseProxy` correctly closes backend on canceled Switching Protocol requests
- 🟡 **net/http/pprof** All endpoints support `seconds` parameter for delta profiles

### Other packages

- 🟡 **bufio** `Scanner` returns `ErrBadReadCount` instead of panicking on negative Read
- 🟡 **context** Creating derived Context with nil parent now explicitly panics
- 🟡 **crypto** `Hash` implements `fmt.Stringer`
- 🟡 **crypto/ecdsa** `PrivateKey`/`PublicKey` have `Equal` methods
- 🟡 **crypto/ed25519** `PrivateKey`/`PublicKey` have `Equal` methods
- 🟡 **crypto/rsa** `VerifyPKCS1v15` rejects invalid short signatures with missing zeroes
- 🟡 **crypto/tls** New `Dialer` type and `DialContext` method
- 🟡 **crypto/tls** `VerifyConnection` callback for custom verification
- 🟡 **crypto/tls** Session ticket keys auto-rotated every 24 hours (7-day lifetime)
- 🟡 **crypto/tls** Downgrade protection checks enforced (RFC 8446)
- 🟡 **crypto/tls** `SignatureScheme`/`CurveID`/`ClientAuthType` implement `fmt.Stringer`
- 🟡 **crypto/tls** `ConnectionState` fields repopulated on resumed connections
- 🟡 **crypto/x509** `CreateRevocationList` and `RevocationList` for X.509 v2 CRLs
- 🟡 **crypto/x509** `CreateCertificate` auto-generates `SubjectKeyId` for CAs
- 🟡 **crypto/x509** `SSL_CERT_DIR` can be colon-separated list on Unix
- 🟡 **crypto/x509** macOS: Always links against `Security.framework` for system roots
- 🟡 **crypto/x509/pkix** `Name.String` prints non-standard `Names` if `ExtraNames` nil
- 🟡 **database/sql** `DB.SetConnMaxIdleTime` removes connections idle too long
- 🟡 **database/sql** `Row.Err` checks for query errors without calling `Scan`
- 🟡 **database/sql/driver** `Validator` interface for connection validity checking
- 🟡 **debug/pe** `IMAGE_FILE`, `IMAGE_SUBSYSTEM`, `IMAGE_DLLCHARACTERISTICS` constants
- 🟡 **encoding/asn1** `Marshal` sorts SET OF components per X.690 DER
- 🟡 **encoding/asn1** `Unmarshal` rejects non-minimal tags and Object Identifiers
- 🟡 **encoding/json** Internal limit on nesting depth to prevent stack issues
- 🟡 **flag** `-h`/`-help` now exit with status 0 (was 2)
- 🟡 **fmt** `%#g`/`%#G` preserve trailing zeros for floating-point
- 🟡 **go/format** Number literal prefixes and exponents canonicalized
- 🟡 **html/template** Unicode escapes (`\uNNNN`) in JavaScript/JSON contexts
- 🟡 **io/ioutil** `TempDir`/`TempFile` reject patterns with path separators
- 🟡 **math/big** `Int.FillBytes` serializes to fixed-size byte slices
- 🟡 **math/cmplx** Updated to conform to C99 Annex G for special arguments
- 🟡 **net** I/O errors beyond deadline now return/wrap `os.ErrDeadlineExceeded`
- 🟡 **net** New `Resolver.LookupIP` for network-specific, context-aware IP lookups
- 🟡 **net/http** Stricter parsing against request smuggling (no non-ASCII whitespace)
- 🟡 **net/http** "identity" `Transfer-Encoding` support dropped
- 🟡 **net/http/httputil** `ReverseProxy` supports nil `X-Forwarded-For` map entry
- 🟡 **net/url** New `URL.RawFragment` and `EscapedFragment` for fragment control
- 🟡 **net/url** New `URL.Redacted` returns URL with password replaced by `xxxxx`
- 🟡 **os** I/O errors beyond deadline return/wrap `os.ErrDeadlineExceeded`
- 🟡 **os** `os` and `net` automatically retry `EINTR` system calls
- 🟡 **os** `File.ReadFrom` supports `copy_file_range` for efficient copying
- 🟡 **plugin** DWARF generation enabled by default on macOS
- 🟡 **plugin** `freebsd/amd64` support
- 🟡 **reflect** Non-exported embedded field methods no longer accessible
- 🟡 **regexp** `Regexp.SubexpIndex` returns first subexpression index by name
- 🟡 **runtime** `ReadMemStats`/`GoroutineProfile` no longer block during GC
- 🟡 **runtime/pprof** Goroutine profile includes profile labels
- 🟡 **strconv** `FormatComplex`/`ParseComplex` for complex number formatting/parsing
- 🟡 **sync** `Map.LoadAndDelete` atomically deletes key and returns previous value
- 🟡 **sync** `Map.Delete` more efficient
- 🟡 **syscall** Unix: `Setctty` requires `Ctty` be child descriptor number
- 🟡 **syscall** Windows: System calls returning floating point values supported
- 🟡 **testing** `T.Deadline` reports timeout deadline
- 🟡 **testing** `TestMain` no longer required to call `os.Exit`
- 🟡 **testing** `T.TempDir`/`B.TempDir` create auto-cleaned temporary directories
- 🟡 **testing** `-v` groups output by test name
- 🟡 **text/template** `JSEscape` consistently uses Unicode escapes
- 🟡 **time** `Ticker.Reset` changes ticker duration
- 🟡 **time** `ParseDuration` quotes original value in errors
