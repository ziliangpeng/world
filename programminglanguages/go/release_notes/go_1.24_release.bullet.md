# Go 1.24 Release Notes

**Released:** February 2025
**EOL:** February 2027

## Major Highlights

Go 1.24 advances the language with full generic type alias support, introduces critical FIPS 140-3 compliance, and expands cryptographic capabilities:

1. **Generic type aliases** - Type aliases can now be parameterized like defined types (disable with `GOEXPERIMENT=noaliastypeparams`)
2. **FIPS 140-3 compliance** - New mechanisms for FIPS 140-3 compliance with Go Cryptographic Module v1.0.0
3. **New crypto packages** - `crypto/mlkem` (ML-KEM post-quantum), `crypto/hkdf`, `crypto/pbkdf2`, `crypto/sha3` moved from x/crypto
4. **Tool tracking in go.mod** - `tool` directives eliminate need for "tools.go" workaround
5. **Directory-limited filesystem access** - New `os.Root` type for secure filesystem operations within directory
6. **Improved finalizers** - `runtime.AddCleanup` replaces `SetFinalizer` with more flexible, efficient design
7. **New weak package** - Weak pointers for memory-efficient structures
8. **New testing.B.Loop** - Faster, less error-prone benchmark iteration method

## Breaking Changes

- 🔴 **Compiler** Now always reports error if receiver denotes cgo-generated type (directly or via alias)
- 🟡 **crypto/aes** `NewCipher` no longer implements undocumented `NewCTR`, `NewGCM`, `NewCBCEncrypter`, `NewCBCDecrypter` methods

## New Features

### Language Features

- 🔴 **Language** Generic type aliases fully supported - `type A[T any] = B[T]` now works

### New Packages

- 🔴 **crypto/mlkem** ML-KEM-768 and ML-KEM-1024 post-quantum key exchange (formerly Kyber)
- 🔴 **crypto/hkdf** HMAC-based Extract-and-Expand key derivation (RFC 5869)
- 🔴 **crypto/pbkdf2** Password-based key derivation (RFC 8018)
- 🔴 **crypto/sha3** SHA-3, SHAKE, and cSHAKE (FIPS 202)
- 🔴 **weak** Weak pointers for memory-efficient structures (caches, canonicalization maps)
- 🟡 **testing/synctest** Experimental support for testing concurrent code with virtual time (`GOEXPERIMENT=synctest`)

### Tooling & Developer Experience

- 🔴 **go command** Tool tracking via `tool` directives in go.mod - eliminates "tools.go" workaround
- 🔴 **go command** `go tool` runs tools from current module in addition to distribution tools
- 🔴 **go get** `-tool` flag adds `tool` directive for named packages
- 🔴 **go build/install** `-json` flag reports build output/failures as structured JSON
- 🔴 **go test** `-json` now includes build output/failures (revert with `GODEBUG=gotestjsonbuildtext=1`)
- 🔴 **go build** Sets main module version in binary from VCS tag/commit (with `+dirty` suffix if uncommitted changes)
- 🟡 **go command** New `GOAUTH` environment variable for flexible private module authentication
- 🟡 **go command** New `tool` meta-pattern refers to all tools in current module
- 🟡 **go command** Executables from `go run` and `go tool` now cached in build cache
- 🟡 **go command** `GODEBUG=toolchaintrace=1` traces toolchain selection process
- 🟡 **cgo** New annotations: `#cgo noescape` and `#cgo nocallback` for performance optimization
- 🟡 **cgo** Better detection of incompatible C function declarations across files
- 🟡 **objdump** Now supports disassembly on loong64, riscv64, and s390x
- 🟡 **vet** New `tests` analyzer reports malformed test/fuzz/benchmark/example declarations
- 🟡 **vet** `printf` now diagnoses `fmt.Printf(s)` with non-constant format string (Go 1.24+)
- 🟡 **vet** `buildtag` reports invalid Go major version build constraints
- 🟡 **vet** `copylock` detects sync.Locker copies in 3-clause for loops
- 🟡 **GOCACHEPROG** Binary/test caching via child process JSON protocol now generally available

## Improvements

### Performance

- 🟢 **Runtime** Swiss Tables map, efficient small object allocation, new mutex: 2-3% CPU overhead decrease
- 🟢 **Runtime** May be disabled with `GOEXPERIMENT=noswissmap,nospinbitmutex`
- 🟢 **crypto/cipher** CTR mode several times faster on amd64 and arm64 when used with AES

### Security & FIPS 140-3

- 🔴 **FIPS 140-3** New `GOFIPS140` environment variable selects Go Cryptographic Module version
- 🔴 **FIPS 140-3** New `fips140` GODEBUG setting enables FIPS 140-3 mode at runtime
- 🔴 **FIPS 140-3** Go 1.24 includes Go Cryptographic Module v1.0.0 (under CMVP testing)

## Deprecations

- 🟡 **crypto/cipher** `NewOFB`, `NewCFBEncrypter`, `NewCFBDecrypter` deprecated - use AEAD modes or CTR
- 🟡 **Language** `GOEXPERIMENT=aliastypeparams` setting will be removed in Go 1.25

## Platform & Environment

- 🔴 **Platform** Linux: Requires kernel 3.2 or later (was 2.6.32)
- 🔴 **Platform** macOS: Go 1.24 is last supporting 11 Big Sur - Go 1.25 requires 12 Monterey
- 🟡 **Platform** Windows: 32-bit windows/arm port marked broken
- 🟡 **Platform** WebAssembly: New `go:wasmexport` directive for exporting functions to host
- 🟡 **Platform** WebAssembly: More types permitted as argument/result for `go:wasmimport` functions
- 🟡 **Platform** WebAssembly: Support files moved from `misc/wasm` to `lib/wasm`
- 🟡 **Platform** WebAssembly: Significantly reduced initial memory size

## Implementation Details

- 🟢 **Linker** Generates GNU build ID (ELF `NT_GNU_BUILD_ID`) on ELF platforms by default
- 🟢 **Linker** Generates UUID (Mach-O `LC_UUID`) on macOS by default
- 🟢 **Bootstrap** Requires Go 1.22.6 or later (Go 1.26 expected to require Go 1.24 point release)

## Standard Library Highlights

### Directory-Limited Filesystem Access

- 🔴 **os** `os.Root` type provides filesystem operations within specific directory
- 🔴 **os** `os.OpenRoot` opens directory and returns `os.Root`
- 🔴 **os** Root methods mirror os package: `Open`, `Create`, `Mkdir`, `Stat`, etc.

### New Testing.B.Loop

- 🔴 **testing** `B.Loop` method for benchmark iterations: `for b.Loop() { ... }` replaces `for range b.N`
- 🔴 **testing** Advantages: benchmarks execute once per -count, parameters/results kept alive

### Improved Finalizers

- 🔴 **runtime** `AddCleanup` attaches cleanup function to object (more flexible than `SetFinalizer`)
- 🔴 **runtime** Multiple cleanups per object, cleanups on interior pointers, no cycle leaks

### Cryptographic Enhancements

- 🔴 **crypto/ecdsa** `PrivateKey.Sign` produces deterministic signature (RFC 6979) if random source is nil
- 🔴 **crypto/rsa** Returns error for keys <1024 bits (insecure); `GODEBUG=rsa1024min=0` restores old behavior
- 🔴 **crypto/rsa** Safer/more efficient to call `Precompute` before `Validate`
- 🔴 **crypto/rand** `Read` guaranteed not to fail (returns nil error) - panics on unrecoverable errors
- 🔴 **crypto/rand** Linux 6.11+: Uses `getrandom` via vDSO (several times faster)
- 🔴 **crypto/rand** OpenBSD: Now uses `arc4random_buf(3)`
- 🔴 **crypto/rand** New `Text` function generates cryptographically secure random text strings
- 🟡 **crypto/cipher** `NewGCMWithRandomNonce` returns AEAD generating random nonce during Seal
- 🟡 **crypto/subtle** `WithDataIndependentTiming` enables data-independent timing features (DIT on arm64)
- 🟡 **crypto/subtle** `XORBytes` now panics if output doesn't overlap exactly or not at all with inputs
- 🟡 **crypto/tls** TLS server now supports Encrypted Client Hello (ECH)
- 🟡 **crypto/tls** Post-quantum `X25519MLKEM768` key exchange enabled by default (`GODEBUG=tlsmlkem=0` reverts)
- 🟡 **crypto/tls** Support for experimental `X25519Kyber768Draft00` removed
- 🟡 **crypto/tls** New `ClientHelloInfo.Extensions` lists extension IDs for fingerprinting
- 🟡 **crypto/x509** `x509sha1` GODEBUG setting removed - no longer supports SHA-1 signatures
- 🟡 **crypto/x509** `OID` implements `encoding.BinaryAppender` and `encoding.TextAppender`
- 🟡 **crypto/x509** Default certificate policies field changed to `Certificate.Policies` (`GODEBUG=x509usepolicies=0`)
- 🟡 **crypto/x509** `CreateCertificate` generates RFC 5280 compliant serial number if template has nil SerialNumber
- 🟡 **crypto/x509** `Certificate.Verify` supports policy validation (RFC 5280, RFC 9618)

### New APIs

- 🟡 **encoding** New `TextAppender` and `BinaryAppender` interfaces for efficient appending
- 🟡 **encoding/json** `omitzero` struct tag option omits zero-valued fields
- 🟡 **go/types** Iterator methods for sequences: `Interface.EmbeddedTypes`, `Struct.Fields`, `Tuple.Variables`, etc.
- 🟡 **hash** New `XOF` interface for extendable output functions (e.g., SHAKE)
- 🟡 **hash** New `Cloner` interface for copying hash state (all stdlib hashes implement it)
- 🟡 **hash/maphash** `Comparable` and `WriteComparable` compute hash of any comparable value
- 🟡 **log/slog** `DiscardHandler` always discards output
- 🟡 **net/http** `Server.Protocols` and `Transport.Protocols` configure HTTP protocols
- 🟡 **net/http** Support for unencrypted HTTP/2 connections
- 🟡 **reflect** `SliceAt` function analogous to `NewAt` for slices
- 🟡 **testing** `T.Context` and `B.Context` return context canceled after test completes
- 🟡 **testing** `T.Chdir` and `B.Chdir` change working directory for test/benchmark duration

### Performance

- 🟡 **crypto/rsa** Public and private key operations up to 2x faster on wasm
- 🟡 **crypto/sha1** Hashing 2x faster on amd64 with SHA-NI instructions
- 🟡 **crypto/sha3** Hashing 2x faster on Apple M processors
