# Go 1.25 Release Notes

**Released:** August 2025
**EOL:** August 2027

## Major Highlights

Go 1.25 brings container-aware GOMAXPROCS, experimental improvements, and important bug fixes:

1. **Container-aware GOMAXPROCS** - Runtime considers cgroup CPU limits and updates GOMAXPROCS dynamically
2. **Experimental green tea GC** - 10-40% reduction in GC overhead with improved locality and CPU scalability (`GOEXPERIMENT=greenteagc`)
3. **Trace flight recorder** - Lightweight in-memory ring buffer for capturing runtime traces of rare events
4. **New testing/synctest package** - Virtual time for testing concurrent code (graduated from experiment)
5. **Experimental encoding/json/v2** - Major JSON revision with substantially better performance (`GOEXPERIMENT=jsonv2`)
6. **nil pointer bug fix** - Corrects compiler bug from Go 1.21-1.24 that incorrectly delayed nil checks
7. **DWARF5 support** - Reduces debug info size and link time

## Breaking Changes

- 🔴 **Compiler** nil pointer bug fixed - programs using results before checking errors now correctly panic
- 🔴 **crypto** Changing `fips140` GODEBUG setting after program start now no-op (previously panicked)
- 🟡 **crypto/elliptic** Hidden `Inverse` and `CombinedMult` methods removed from some Curve implementations
- 🟡 **crypto/rsa** `PublicKey` no longer claims modulus is secret
- 🟡 **crypto/tls** SHA-1 signature algorithms disallowed in TLS 1.2 (RFC 9155) - revert with `GODEBUG=tlssha1=1`
- 🟡 **crypto/x509** `ParseCertificate` rejects certificates with negative pathLenConstraint in BasicConstraints
- 🟡 **math/rand** Top-level `Seed` now no-op (revert with `GODEBUG=randseednop=0`)

## New Features

### Runtime & Platform

- 🔴 **Runtime** Container-aware GOMAXPROCS: considers cgroup CPU bandwidth limits on Linux
- 🔴 **Runtime** Dynamic GOMAXPROCS updates when CPU count or cgroup limits change
- 🔴 **Runtime** Disable with `GODEBUG=containermaxprocs=0` or `GODEBUG=updatemaxprocs=0`
- 🔴 **Runtime** VMA names on Linux: annotates memory mappings with purpose (e.g., `[anon: Go: heap]`)
- 🟡 **Runtime** Experimental green tea GC: 10-40% GC overhead reduction (`GOEXPERIMENT=greenteagc`)
- 🟡 **Runtime** Trace flight recorder: `runtime/trace.FlightRecorder` for lightweight continuous tracing
- 🟡 **Runtime** Unhandled panic output no longer repeats panic value text on repanic
- 🟡 **Runtime** Cleanup functions now execute concurrently and in parallel
- 🟡 **Runtime** `GODEBUG=checkfinalizers=1` helps find common finalizer/cleanup issues
- 🟡 **Runtime** `SetDefaultGOMAXPROCS` resets to runtime default value

### New Packages

- 🔴 **testing/synctest** Virtual time for testing concurrent code (graduated from experiment)
- 🟡 **encoding/json/v2** Experimental major JSON revision (`GOEXPERIMENT=jsonv2`)
- 🟡 **encoding/json/jsontext** Low-level JSON syntax processing (`GOEXPERIMENT=jsonv2`)

### Tooling & Developer Experience

- 🟡 **go command** New `ignore` directive in go.mod specifies directories to ignore
- 🟡 **go doc** `-http` starts documentation server and opens in browser
- 🟡 **go version** `-m -json` prints JSON of runtime/debug.BuildInfo structures
- 🟡 **go command** Supports subdirectory of repository as module root
- 🟡 **go command** New `work` package pattern matches all packages in work modules
- 🟡 **go command** No longer adds `toolchain` line when updating `go` line in go.mod/go.work
- 🟡 **go build** `-asan` now defaults to leak detection at program exit
- 🟡 **go command** Fewer prebuilt tool binaries in distribution (core toolchain still included)
- 🟡 **vet** New `waitgroup` analyzer reports misplaced `sync.WaitGroup.Add` calls
- 🟡 **vet** New `hostport` analyzer reports `fmt.Sprintf("%s:%d", host, port)` IPv6 issues

## Improvements

### Performance

- 🟢 **Compiler** Slices: backing store allocated on stack in more situations
- 🟢 **Compiler** Faster slices can amplify incorrect unsafe.Pointer usage effects
- 🟢 **Compiler** Use `-compile=variablemake` with bisect tool or `-gcflags=all=-d=variablemakehash=n` to disable
- 🟢 **crypto/sha3** Hashing 2x faster on Apple M processors
- 🟢 **crypto/sha1** Hashing 2x faster on amd64 with SHA-NI instructions
- 🟢 **crypto/ecdsa** Signing 4x faster in FIPS mode (matches non-FIPS performance)
- 🟢 **crypto/ed25519** Signing 4x faster in FIPS mode
- 🟢 **crypto/rsa** Key generation 3x faster

### Security

- 🟡 **crypto/tls** Extended Master Secret required in TLS 1.2 when FIPS 140-3 mode enabled
- 🟡 **crypto/tls** Ed25519 and X25519MLKEM768 allowed in FIPS mode
- 🟡 **crypto/tls** Servers prefer highest supported protocol version
- 🟡 **crypto/x509** `CreateCertificate` uses truncated SHA-256 for SubjectKeyId (was SHA-1) - revert with `GODEBUG=x509sha256skid=0`

## Platform & Environment

- 🔴 **Platform** macOS: Requires 12 Monterey or later (discontinued 11 Big Sur support)
- 🔴 **Platform** Windows: 32-bit windows/arm port will be removed in Go 1.26
- 🟡 **Platform** AMD64: GOAMD64=v3+ now uses fused multiply-add instructions (may change floating-point values)
- 🟡 **Platform** Loong64: Now supports race detector, cgo traceback, and internal link mode
- 🟡 **Platform** RISC-V: Now supports `plugin` build mode
- 🟡 **Platform** RISC-V: `GORISCV64` accepts new value `rva23u64`

## Implementation Details

- 🟢 **Compiler** nil pointer bug fix: programs using results before error checks now correctly panic
- 🟢 **Compiler** DWARF5 support reduces debug info size and link time (disable with `GOEXPERIMENT=nodwarf5`)
- 🟢 **Linker** `-funcalign=N` specifies function entry alignment

## Standard Library Highlights

### New testing/synctest Package

- 🔴 **testing/synctest** `Test` runs test function in isolated bubble with virtual time
- 🔴 **testing/synctest** `Wait` waits for all goroutines in bubble to block
- 🟡 **testing/synctest** Graduated from experiment (old API under `GOEXPERIMENT=synctest` until Go 1.26)

### Experimental JSON v2

- 🔴 **encoding/json/v2** Major revision with substantially better performance (`GOEXPERIMENT=jsonv2`)
- 🔴 **encoding/json/jsontext** Lower-level JSON syntax processing
- 🟡 **encoding/json** Uses new implementation when jsonv2 enabled (errors may change)
- 🟡 **encoding/json** New options for configuring marshaler/unmarshaler

### New APIs

- 🟡 **archive/tar** `Writer.AddFS` supports symbolic links for `io/fs.ReadLinkFS` filesystems
- 🟡 **bytes** Iterator functions: `Lines`, `SplitSeq`, `SplitAfterSeq`, `FieldsSeq`, `FieldsFuncSeq`
- 🟡 **crypto** `MessageSigner` interface and `SignMessage` function for one-shot signing
- 🟡 **crypto/ecdsa** `ParseRawPrivateKey`, `ParseUncompressedPublicKey`, `PrivateKey.Bytes`, `PublicKey.Bytes`
- 🟡 **go/ast** `PreorderStack` like `Inspect` but also provides stack of enclosing nodes
- 🟡 **go/token** `FileSet.AddExistingFiles` adds existing Files to FileSet
- 🟡 **go/types** `Var.Kind` classifies variable (package-level, receiver, parameter, etc.)
- 🟡 **go/types** `LookupSelection` looks up field/method and returns Selection
- 🟡 **hash** New `XOF` interface for extendable output functions (e.g., SHAKE)
- 🟡 **hash** `Cloner` interface for copying hash state (all stdlib hashes implement it)
- 🟡 **io/fs** `ReadLinkFS` interface for reading symbolic links
- 🟡 **log/slog** `GroupAttrs` creates group Attr from slice of Attrs
- 🟡 **log/slog** `Record.Source` returns source location
- 🟡 **mime/multipart** `FileContentDisposition` builds Content-Disposition header fields
- 🟡 **net** `LookupMX` returns DNS names that look like IP addresses (not just domain names)
- 🟡 **net** Windows: `ListenMulticastUDP` supports IPv6
- 🟡 **net** Windows: File conversion functions now implemented
- 🟡 **net/http** `CrossOriginProtection` implements CSRF protection using Fetch metadata
- 🟡 **os** Windows: `NewFile` supports handles opened for asynchronous I/O
- 🟡 **os** `DirFS` and `Root.FS` implement `io/fs.ReadLinkFS`
- 🟡 **os** `CopyFS` supports symlinks for ReadLinkFS filesystems
- 🟡 **os** Root supports additional methods: `Chmod`, `Chown`, `Chtimes`, `Lchown`, `Link`, `MkdirAll`, `ReadFile`, `Readlink`, `RemoveAll`, `Rename`, `Symlink`, `WriteFile`
- 🟡 **reflect** `TypeAssert` converts Value to Go value of given type without allocations
- 🟡 **strings** Iterator functions: `Lines`, `SplitSeq`, `SplitAfterSeq`, `FieldsSeq`, `FieldsFuncSeq`
- 🟡 **sync** `WaitGroup.Go` method makes creating/counting goroutines more convenient
- 🟡 **testing** `T.Attr`, `B.Attr`, `F.Attr` emit attributes to test log
- 🟡 **testing** `T.Output`, `B.Output`, `F.Output` provide io.Writer for test output
- 🟡 **testing** `AllocsPerRun` now panics if parallel tests running
- 🟡 **unicode** `CategoryAliases` map provides category alias names
- 🟡 **unicode** New categories `Cn` (unassigned) and `LC` (cased letters)
- 🟡 **unique** More eager, efficient, and parallel reclamation of interned values

## Deprecations

- 🟡 **go/ast** `FilterPackage`, `PackageExports`, `MergePackageFiles` deprecated
- 🟡 **go/parser** `ParseDir` deprecated
- 🟡 **runtime** `GOROOT` function deprecated - use `go env GOROOT` instead

## Known Issues

- 🟡 **regexp/syntax** Character class names now case-insensitive, support aliases (e.g., `\p{Letter}`)
- 🟡 **sync** `Map` implementation changed (better performance) - revert with `GOEXPERIMENT=nosynchashtriemap`
- 🟡 **testing/fstest** `MapFS` implements `io/fs.ReadLinkFS`; `TestFS` verifies it and no longer follows symlinks
