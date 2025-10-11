# Go 1.9 Release Notes

**Released:** August 24, 2017
**EOL:** N/A (maintenance ended)

## Major Highlights

Go 1.9 delivers type aliases, parallel compilation, and monotonic time support:

1. **Type aliases** - New `type T1 = T2` syntax for gradual code refactoring
2. **Parallel function compilation** - Compiler parallelizes function compilation within packages
3. **Monotonic time** - `time.Time` tracks monotonic clock for safe duration calculations
4. **`math/bits`** - New package with optimized bit manipulation functions
5. **`sync.Map`** - New concurrent map with amortized-constant-time operations
6. **Test helpers** - `T.Helper()` marks helper functions for better error reporting
7. **Profiler labels** - Tag profiler records for better profile analysis

## Breaking Changes

- 🟡 **Language** Type aliases: `type T1 = T2` creates alias (alternate name) for same type
- 🟡 **Language** Floating-point operations: spec now defines when FMA fusion is allowed
- 🟡 **ppc64** Now requires POWER8+ for both `ppc64` and `ppc64le` (was POWER5+ for big-endian)
- 🟢 **FreeBSD** Go 1.10 will require FreeBSD 10.3+ (1.9 is last to support 9.3)
- 🟢 **OpenBSD** Now requires OpenBSD 6.0+ (enables PT_TLS generation for cgo)

## New Features

- 🔴 **Language** Type aliases for gradual refactoring: `type T1 = T2`
- 🔴 **math/bits** New package with optimized bit manipulation (recognized as intrinsics on most architectures)
- 🔴 **sync** New `Map` type: concurrent map with amortized-constant-time loads/stores/deletes
- 🔴 **testing** `T.Helper()` and `B.Helper()` mark test helper functions for better line numbers
- 🔴 **time** Transparent monotonic time tracking in `Time` values prevents wall clock skew issues
- 🟡 **runtime/pprof** Profiler labels via `Do` function for distinguishing calls in different contexts
- 🟡 **go test** New `-list` flag prints matching tests/benchmarks/examples without running them
- 🟢 **go env** New `-json` flag for JSON output

## Improvements

- 🟢 **Performance** Compiler parallelizes function compilation (on by default, disable with `GO19CONCURRENTCOMPILATION=0`)
- 🟢 **Performance** GC: large object allocation much faster (>50GB heaps with many large objects)
- 🟢 **Performance** `ReadMemStats` now takes <100μs even for very large heaps
- 🟢 **Performance** GC functions (`runtime.GC`, `debug.SetGCPercent`, `debug.FreeOSMemory`) now concurrent
- 🟢 **Performance** `regexp` faster for simple expressions
- 🟢 **Compilation** Functions now compiled in parallel within a package

## Tooling & Developer Experience

- 🔴 **go command** `./...` no longer matches `vendor` directories (use `./vendor/...` explicitly)
- 🟡 **go tool** If Go installation moved, tool uses invocation path to locate root (no `GOROOT` needed)
- 🟡 **go doc** Now supports viewing struct field documentation: `go doc http.Client.Jar`
- 🟡 **go doc** Long argument lists now truncated for readability
- 🟡 **pprof** Profiles now include symbol information (no binary needed for viewing)
- 🟡 **pprof** Now uses HTTP proxy from environment via `http.ProxyFromEnvironment`
- 🟡 **vet** Better integrated with `go` tool - all build flags now supported
- 🟢 **Compiler** Complex division now C99-compliant
- 🟢 **Compiler** DWARF with lexical scopes when `-N -l` flags provided (`.debug_info` now DWARF v4)
- 🟢 **Compiler** `GOARM` and `GO386` values now affect build ID

## Platform & Environment

- 🟢 **Known Issues** FreeBSD instabilities (rare crashes) - issue #15658
- 🟢 **Known Issues** NetBSD builders not passing (kernel crashes fixed in 7.1.1, but tests still failing)

## Implementation Details

- 🟢 **Runtime** `Callers` users should use `CallersFrames` (not direct PC slice inspection) for inlined frames
- 🟢 **Runtime** `Caller` for single caller (not `Callers` with slice of length 1)
- 🟢 **Runtime** GC functions now concurrent, only block calling goroutine
- 🟢 **Runtime** Windows no longer forces high timer resolution when idle (better battery life)
- 🟢 **Runtime** FreeBSD: `GOMAXPROCS` and `NumCPU` now based on process CPU mask
- 🟢 **Runtime** Preliminary Android O support
- 🟢 **time** Monotonic time in `Time` values: durations immune to wall clock adjustments
- 🟢 **time** `Duration.Round` and `Duration.Truncate` for duration rounding
- 🟢 **crypto/rand** Linux: calls `getrandom` without `GRND_NONBLOCK` (blocks until sufficient randomness)
- 🟢 **crypto/x509** `SSL_CERT_FILE` and `SSL_CERT_DIR` environment variables override system defaults
- 🟢 **crypto/x509** Name constraints: excluded domains now supported
- 🟢 **database/sql** `Tx.Stmt` now uses cached `Stmt` if available
- 🟢 **database/sql** `DB.Conn` returns exclusive connection from pool
- 🟢 **database/sql** Drivers can implement `NamedValueChecker` for custom argument checking
- 🟢 **net** `Resolver.StrictErrors` controls temporary error handling in multi-sub-query lookups
- 🟢 **net** `*Conn.SyscallConn` methods provide access to underlying file descriptors
- 🟢 **net/http** `ServeMux` ignores ports in host header when matching handlers
- 🟢 **net/http** `Server.ServeTLS` wraps `Serve` with TLS support
- 🟢 **net/http** HTTP/2 priority write scheduler now default
- 🟢 **net/http** `Transport` supports SOCKS5 proxy via `socks5` scheme
- 🟢 **os** Now uses internal runtime poller for file I/O (fewer threads, eliminates close races)
- 🟢 **os/exec** Prevents duplicate environment variables (last value wins)
- 🟢 **os/user** `Lookup*` functions now work without cgo by reading `/etc/passwd` and `/etc/group`
- 🟢 **reflect** `MakeMapWithSize` creates map with capacity hint
- 🟢 **sync** `Mutex` now more fair
- 🟢 **syscall** `Credential.NoSetGroups` controls `setgroups` call when starting new process
- 🟢 **syscall** `SysProcAttr.AmbientCaps` sets ambient capabilities on Linux 4.3+
- 🟢 **syscall** 64-bit x86 Linux: process creation uses `CLONE_VFORK` and `CLONE_VM` for lower latency
