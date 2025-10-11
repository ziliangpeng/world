# Go 1.5 Release Notes

**Released:** August 19, 2015
**EOL:** N/A (maintenance ended)

## Major Highlights

Go 1.5 is a landmark release with a completely redesigned implementation while maintaining full compatibility:

1. **Self-hosting compiler** - Toolchain now written entirely in Go (no more C code!)
2. **Concurrent garbage collector** - GC pauses typically <10ms, even with multi-GB heaps
3. **Default `GOMAXPROCS`** - Now matches CPU count (was 1), enabling parallelism by default
4. **Experimental vendoring** - Official mechanism for package dependency management
5. **Build modes** - New shared libraries and plugin support (`-buildmode`)
6. **Map literal detection** - New vet check to detect invalid map key types

## Breaking Changes

- 🔴 **Runtime** `GOMAXPROCS` defaults to number of CPUs (was 1) - may expose latent race conditions
- 🟡 **Compiler** `GO15VENDOREXPERIMENT` enables vendor directories (experimental in 1.5, default in 1.6)
- 🟡 **Linker** `-X` flag now requires single argument format: `-X importpath.name=value` (old format deprecated)
- 🟡 **Toolchain** C compiler tools (6c, 8c, etc.) removed - all code must be Go or assembly
- 🟢 **go command** Binaries only support current OS and architecture (was all)
- 🟢 **net/http** `Transport` now retries non-idempotent requests if no bytes written

## New Features

- 🔴 **Tooling** Experimental vendoring support via `GO15VENDOREXPERIMENT=1` environment variable
- 🔴 **go command** New `-buildmode` flag: `archive`, `c-archive`, `shared`, `c-shared`, `exe` for library building
- 🟡 **go command** New `-asmflags`, `-buildmode`, `-pkgdir`, `-toolexec` flags
- 🟡 **go list** New `-f` template functions: `join`, `context`, `packageImports`, `packageGoFiles`
- 🟡 **trace** New `go tool trace` command for execution trace visualization and analysis
- 🟡 **go doc** Command resurrected as official documentation viewer (was removed in 1.2)
- 🟢 **runtime/trace** New package for execution tracing

## Improvements

- 🟢 **Performance** Concurrent GC reduces pause times to typically <10ms (was 10s-100s of ms)
- 🟢 **Performance** Binary sizes ~30% smaller due to new linker
- 🟢 **Performance** Build times ~2x slower than Go 1.4 (expected to improve in future releases)
- 🟢 **Compiler** Completely rewritten in Go (translated from C via automatic conversion)
- 🟢 **Assembler** Rewritten in Go, now single target-independent program
- 🟢 **Linker** Rewritten in Go, much faster and produces smaller binaries
- 🟢 **GC** Concurrent, low-latency collector reduces STW pauses by ~40-90%
- 🟢 **GC** Soft heap size goal can be set via `GOGC` or `SetGCPercent`

## Tooling & Developer Experience

- 🔴 **vet** New check for `Printf` calls with invalid map key types
- 🟡 **go command** `GOROOT` no longer inferred from source tree location (use explicit GOROOT or distribution)
- 🟡 **go command** Cross-compilation now requires setting `GOOS` and `GOARCH` only (no toolchain rebuild)
- 🟡 **go tool** New `-X` linker flag syntax: `-X importpath.name=value` (old format still works with warning)
- 🟡 **go test** New `-count` flag to run tests multiple times
- 🟡 **Assembly** Function reference `runtime·foo` changed to `runtime∕foo` (dot to middle dot)

## Platform & Environment

- 🟡 **Platform** Experimental support for darwin/arm and darwin/arm64 (iOS)
- 🟡 **Platform** Darwin binaries now require OS X 10.7 or later (was 10.6)
- 🟡 **Platform** Linux binaries now require Linux 2.6.23+ (was 2.6.18) for `FUTEX_WAIT_BITSET`
- 🟡 **Platform** OpenBSD binaries now require OpenBSD 5.6+ (was 5.5)
- 🟡 **Platform** Experimental support for linux/arm64
- 🟡 **Platform** Experimental support for linux/ppc64 and linux/ppc64le
- 🟢 **Platform** Windows XP and Vista no longer supported
- 🟢 **go command** Cross-compilation simplified - just set `GOOS` and `GOARCH`

## Implementation Details

- 🟢 **Runtime** Rewritten entirely in Go and assembly (no C code remaining)
- 🟢 **Runtime** Garbage collector now concurrent with sub-10ms pause times
- 🟢 **Runtime** Stack sizes now in bytes (was machine words)
- 🟢 **Runtime** New `runtime/trace` package for execution trace generation
- 🟢 **Compiler** Generates less efficient code than 1.4 (expected to improve)
- 🟢 **cgo** Improved documentation for passing pointers between Go and C
- 🟢 **flag** `PrintDefaults` now respects output destination set by `CommandLine.SetOutput`
- 🟢 **math/big** `Float` type provides arbitrary-precision floating-point arithmetic
- 🟢 **net** `Dialer.DualStack` enables RFC 6555 "Happy Eyeballs" dual-stack connections
- 🟢 **reflect** New `ArrayOf` function creates array types at runtime
- 🟢 **reflect** New `FuncOf` function creates function types at runtime
