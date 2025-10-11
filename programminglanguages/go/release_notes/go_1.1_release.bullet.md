# Go 1.1 Release Notes

**Released:** May 13, 2013
**EOL:** N/A (maintenance ended)

## Major Highlights

Go 1.1 focuses on performance with significant compiler and runtime improvements while maintaining full compatibility with Go 1.0:

1. **30-40% performance improvements** across many programs through compiler optimizations and runtime enhancements
2. **Method values** - Bind methods to specific receivers for cleaner functional programming
3. **Race detector** - New tool to find concurrent access bugs (64-bit x86 only)
4. **`int` is now 64-bit** on 64-bit platforms, enabling slices with >2 billion elements
5. **New `bufio.Scanner`** - Simpler API for scanning text input line-by-line
6. **Better terminating statements** - No explicit return needed after infinite loops
7. **Larger heap on 64-bit** - From a few GB to tens of GB

## Breaking Changes

- 🟡 **Language** Integer division by constant zero is now compile-time error (was runtime panic)
- 🟡 **Language** Surrogate halves in Unicode literals now illegal in rune/string constants
- 🟡 **Compiler** `int` and `uint` are now 64-bit on 64-bit platforms - code assuming 32-bit may break
- 🟢 **net** Protocol-specific resolvers now strictly validate network names (e.g., "tcp", "tcp4", "tcp6")
- 🟢 **net** `ListenUnixgram` now returns `UnixConn` instead of `UDPConn` (bug fix allowed by Go 1 compat rules)
- 🟢 **net** `IPAddr`, `TCPAddr`, `UDPAddr` add new `Zone` field - untagged composite literals will break
- 🟢 **time** Time precision increased from microseconds to nanoseconds on Unix systems - may affect external format round-trips

## New Features

- 🔴 **Language** Method values: bind methods to specific receivers (e.g., `w.Write` where `w` is a `Writer`)
- 🟡 **Language** Terminating statements: no explicit return needed if function ends with infinite loop or if-else with returns
- 🔴 **Tooling** Race detector for finding concurrent access bugs - use `go test -race` (Linux, Mac OS X, Windows on 64-bit x86 only)
- 🟡 **bufio** New `Scanner` type for simpler line-oriented or token-based text scanning
- 🟡 **reflect** New `Select` function and `SelectCase` type for runtime select statements
- 🟡 **reflect** New `Value.Convert` and `Type.ConvertibleTo` for runtime type conversions
- 🟡 **reflect** New `MakeFunc` for creating wrapper functions with `Value` arguments
- 🟡 **reflect** New `ChanOf`, `MapOf`, `SliceOf` functions to construct types at runtime
- 🟢 **go/format** New package for programmatic access to `go fmt` functionality
- 🟢 **net/http/cookiejar** New package for HTTP cookie management
- 🟢 **runtime/race** New internal package for race detection support

## Improvements

- 🟢 **Performance** 30-40% improvement in typical programs through compiler and runtime optimizations
- 🟢 **Compiler** Better code generation, especially for floating point on 32-bit Intel
- 🟢 **Compiler** More aggressive inlining including `append` and interface conversions
- 🟢 **Runtime** New map implementation with significant memory footprint and CPU time reduction
- 🟢 **Runtime** Parallel garbage collector reduces latency on multi-CPU systems
- 🟢 **Runtime** Precise garbage collector on stack values (previously only heap)
- 🟢 **Runtime** Fewer context switches on network operations
- 🟢 **Heap** Maximum heap size on 64-bit increased from few GB to tens of GB

## Tooling & Developer Experience

- 🔴 **go command** Better error messages when packages cannot be located (shows searched paths)
- 🟡 **go command** `go get` now requires valid `GOPATH` - no longer defaults to `GOROOT`
- 🟡 **go command** `go get` fails if `GOPATH` equals `GOROOT`
- 🟡 **go test** Binary preserved after profiling runs (sets `-c` automatically)
- 🟡 **go test** New `-blockprofile` flag for blocking profile generation
- 🟡 **go fix** No longer applies pre-Go 1 fixes - use Go 1.0 tools first for migration
- 🟡 **Build tags** New `go1.1` build constraint for conditional compilation
- 🟢 **vet** Can now validate assembly function signatures match Go prototypes
- 🟢 **vet** Identifies superfluous return statements and panic calls

## Platform & Environment

- 🟡 **Platform** Experimental FreeBSD/ARM support (requires ARMv6+)
- 🟡 **Platform** Experimental NetBSD support (386, amd64, arm)
- 🟡 **Platform** Experimental OpenBSD support (386, amd64)
- 🟡 **Platform** Experimental cgo on linux/arm
- 🟢 **Cross-compilation** `cgo` disabled by default when cross-compiling (set `CGO_ENABLED=1` to enable)

## Implementation Details

- 🟢 **Assembler** Assembly source files need updates for 64-bit `int` and new function representation
- 🟢 **time** `Round` and `Truncate` methods added for precision control in external storage
- 🟢 **time** New `YearDay` method returns one-indexed day of year
- 🟢 **time** New `Timer.Reset` method to reuse timers
- 🟢 **time** New `ParseInLocation` function for timezone-aware parsing
- 🟢 **Subrepositories** `exp` and `old` moved to `go.exp` subrepository at `code.google.com/p/go.exp`
- 🟢 **Subrepositories** `exp/norm` moved to `go.text` subrepository for Unicode/text packages
