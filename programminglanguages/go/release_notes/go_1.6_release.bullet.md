# Go 1.6 Release Notes

**Released:** February 17, 2016
**EOL:** N/A (maintenance ended)

## Major Highlights

Go 1.6 focuses on HTTP/2 support, runtime improvements, and cgo safety:

1. **Transparent HTTP/2** - Automatic HTTP/2 support in `net/http` for clients and servers
2. **Cgo pointer rules** - Defined and enforced rules for passing Go pointers to C
3. **Vendoring enabled** - Vendor directories now enabled by default (no longer experimental)
4. **Template blocks** - New `{{block}}` action for reusable template definitions
5. **Sort optimization** - Sort algorithm improved by ~10% with fewer comparisons
6. **Even lower GC pauses** - Further reduction in pause times, especially for large heaps

## Breaking Changes

- 🔴 **cgo** New pointer passing rules enforced at runtime - Go pointers to C must not contain Go pointers
- 🟡 **cgo** `C.complexfloat` and `C.complexdouble` now separate types (was same as `complex64`/`complex128`)
- 🟡 **reflect** Embedded unexported struct fields now correctly reported as unexported (affects encoders)
- 🟡 **sort** New algorithm changes order of equal elements - use `Stable` if specific order needed
- 🟢 **Runtime** Windows timer resolution call removed (was forcing 1ms global resolution)
- 🟢 **Runtime** Signal handling for c-archive/c-shared now only installs handlers for synchronous signals

## New Features

- 🔴 **net/http** Transparent HTTP/2 support for HTTPS clients and servers (disable with `TLSNextProto` map)
- 🔴 **text/template** New `{{block}}` action for template inheritance and reusability
- 🔴 **text/template** Space trimming with `{{-` and `-}}` around template actions
- 🟡 **Compiler** New `-msan` flag for Clang MemorySanitizer interoperation (Linux/amd64 only)
- 🟡 **Compiler** Parser now hand-written (was yacc-generated)
- 🟢 **go doc** Ambiguous packages now resolved by preferring fewer path elements

## Improvements

- 🟢 **Performance** GC pauses even lower than 1.5, especially with large heaps
- 🟢 **Performance** `compress/bzip2`, `compress/gzip`, `crypto/aes`, `crypto/elliptic`, `crypto/ecdsa`, `sort`: 10%+ improvements
- 🟢 **Sort** Algorithm improved: ~10% fewer calls to `Less` and `Swap`
- 🟢 **Runtime** Concurrent map misuse detection now catches write-while-iterating
- 🟢 **Runtime** Panic stack traces now only show running goroutine by default (set `GOTRACEBACK=all` for all)

## Tooling & Developer Experience

- 🔴 **go command** Vendoring enabled by default (set `GO15VENDOREXPERIMENT=0` to disable)
- 🟡 **Build modes** Extended support: `c-shared` on more platforms, new `pie` mode, `shared` mode on more platforms
- 🟡 **Linker** New `-libgcc` option to specify C compiler support library location
- 🟡 **go vet** New check: diagnoses passing function values (not calls) to `Printf`
- 🟢 **cgo** Runtime checks enforce pointer passing rules (set `GODEBUG=cgocheck=0` to disable, not recommended)

## Platform & Environment

- 🟡 **Platform** Experimental Linux on 64-bit MIPS (`linux/mips64` and `linux/mips64le`) with cgo
- 🟡 **Platform** Experimental Android on 32-bit x86 (`android/386`)
- 🟡 **Platform** Linux on little-endian 64-bit PowerPC (`linux/ppc64le`) now feature complete with cgo
- 🟡 **Platform** FreeBSD now defaults to `clang` (not `gcc`) as external C compiler
- 🟡 **Platform** NaCl now supports SDK versions newer than pepper-41
- 🟢 **Assembly** 32-bit x86 using `-dynlink` or `-shared`: register CX now overwritten, avoid in hand-written assembly

## Implementation Details

- 🟢 **cgo** Pointer passing rules: Go memory passed to C must not contain Go pointers, C must not retain pointer
- 🟢 **cgo** Runtime checks detect violations and crash with diagnostic (can be disabled)
- 🟢 **archive/tar** Many corner case bugs fixed in file format handling
- 🟢 **archive/zip** `Reader` and `Writer` now support per-file compression control
- 🟢 **bufio** `Scanner.Buffer` method allows setting initial and maximum buffer size
- 🟢 **crypto/tls** Now allows `Listen` with nil `Certificates` if `GetCertificate` callback set
- 🟢 **crypto/tls** Added RSA with AES-GCM cipher suites
- 🟢 **encoding/json** `Number` validation now enforced during marshaling
- 🟢 **net** `ParseMAC` now accepts 20-byte IPoIB addresses
- 🟢 **net/http** `FileServer` now sorts directory listings by name
- 🟢 **net/http** Five new error codes: 428, 429, 431, 451, 511
- 🟢 **os** Broken pipe on stdout/stderr now raises `SIGPIPE` signal (was internal counter-based)
- 🟢 **regexp** `Regexp.Copy` method for reducing mutex contention in high-concurrency servers
