# Go 1.7 Release Notes

**Released:** August 15, 2016
**EOL:** N/A (maintenance ended)

## Major Highlights

Go 1.7 delivers the new SSA compiler backend and context package to the standard library:

1. **SSA compiler backend** - New code generation for 64-bit x86 with 5-35% performance improvement
2. **Context package** - `context.Context` promoted from x/net to standard library
3. **Hierarchical testing** - Subtests and sub-benchmarks with `T.Run` and `B.Run`
4. **HTTP tracing** - New `net/http/httptrace` package for request event tracing
5. **Faster compilation** - Compiler and linker optimized, faster than 1.6
6. **Binary-only packages** - Experimental support for distributing packages without source
7. **Vendoring finalized** - `GO15VENDOREXPERIMENT` removed, vendoring now standard

## Breaking Changes

- 🟡 **Language** Specification clarifies terminating statements: "final non-empty statement" considered (affects `go/types`)
- 🟡 **go command** `GO15VENDOREXPERIMENT` environment variable removed - vendoring always enabled
- 🟡 **Linker** `-X` flag two-argument form removed (use `-X name=value`)

## New Features

- 🔴 **context** Package promoted to standard library from `golang.org/x/net/context`
- 🔴 **testing** Subtests and sub-benchmarks: `T.Run` and `B.Run` for hierarchical test organization
- 🔴 **net/http/httptrace** New package for tracing HTTP client request events
- 🔴 **Compiler** New SSA backend for all architectures (5-35% faster on x86-64)
- 🟡 **go tool dist** New `go tool dist list` command lists all supported OS/architecture pairs
- 🟡 **go vet** New `-lostcancel` check detects context cancellation function leaks
- 🟢 **cgo** Now supports Fortran source files in addition to C/C++/Objective-C/SWIG

## Improvements

- 🟢 **Performance** 5-35% CPU time reduction on x86-64 due to SSA backend
- 🟢 **Performance** Compiler and linker run faster than 1.6
- 🟢 **Performance** `crypto/sha1`, `crypto/sha256`, `encoding/binary`, `fmt`, `hash/*`, `strings`, `unicode`: 10%+ improvements
- 🟢 **Binaries** Typically 20-30% smaller than 1.6
- 🟢 **GC** Shorter pauses for programs with many idle goroutines or large package-level variables
- 🟢 **Compiler** Export metadata now in compact binary format (was textual)
- 🟢 **x86-64** Stack frame pointers enabled for better profiling with perf/VTune (2% overhead)

## Tooling & Developer Experience

- 🔴 **go command** Vendoring finalized - `GO15VENDOREXPERIMENT` removed
- 🟡 **go command** Experimental binary-only package support (not for `go get`)
- 🟡 **go list** New `StaleReason` field explains why package needs rebuilding
- 🟡 **go get** Now supports `git.openstack.org` import paths
- 🟡 **go doc** Groups constructors with their types
- 🟡 **go vet** More accurate `-copylock` and `-printf` checks, new `-tests` check
- 🟡 **go tool trace** 400% efficiency improvement (overhead down from 400%+ to ~25%)
- 🟡 **cgo** New helper function `C.CBytes` for `[]byte` to `void*` conversion
- 🟡 **cgo** Deterministic builds with GCC/Clang supporting `-fdebug-prefix-map`

## Platform & Environment

- 🟡 **Platform** Experimental Linux on z Systems (`linux/s390x`)
- 🟡 **Platform** Beginning of Plan 9 on ARM (`plan9/arm`) port
- 🟡 **Platform** macOS 10.12 Sierra support (earlier Go versions won't work)
- 🟡 **Platform** Linux on 64-bit MIPS now has full cgo and external linking
- 🟡 **Platform** Linux on `ppc64le` now requires POWER8+ (was POWER5+ for `ppc64`)
- 🟡 **Platform** OpenBSD now requires 5.6+ (was 5.5) for `getentropy(2)` syscall
- 🟢 **Known Issues** FreeBSD instabilities (crashes in rare cases) - issues #16136, #15658, #16396

## Implementation Details

- 🟢 **Compiler** SSA backend now used for all architectures (was only x86-64 in 1.7 beta)
- 🟢 **Compiler** Can disable SSA with `-ssa=0` flag (for bug reporting)
- 🟢 **Compiler** Export format can be disabled with `-newexport=0` flag (for bug reporting)
- 🟢 **Runtime** `KeepAlive` function prevents premature finalization
- 🟢 **Runtime** `CallersFrames` for accurate call stacks with inlined functions
- 🟢 **Runtime** `SetCgoTraceback` for tighter Go/C code integration
- 🟢 **Runtime** Can now return unused memory on all architectures (was limited in 1.6)
- 🟢 **Runtime** 32-bit systems can use memory anywhere in address space
- 🟢 **Runtime** All panics now implement `error` and `runtime.Error` interfaces
- 🟢 **reflect** `StructOf` function creates struct types at runtime
- 🟢 **reflect** `Method` and `NumMethod` no longer return/count unexported methods
