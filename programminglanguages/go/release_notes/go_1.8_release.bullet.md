# Go 1.8 Release Notes

**Released:** February 16, 2017
**EOL:** N/A (maintenance ended)

## Major Highlights

Go 1.8 focuses on GC improvements, language refinements, and standard library additions:

1. **Sub-100μs GC pauses** - Eliminated stop-the-world stack rescanning, often <10μs pauses
2. **SSA for all architectures** - 20-30% faster on 32-bit ARM, 0-10% on 64-bit x86
3. **Defer overhead halved** - ~50% reduction in deferred function call cost
4. **HTTP/2 Push** - Server push support via `http.Pusher` interface
5. **Graceful HTTP shutdown** - `Server.Shutdown` for clean server termination
6. **Plugin support** - Load Go plugins at runtime (Linux only)
7. **Mutex profiling** - Profile mutex contention with `-mutexprofile`
8. **Default `GOPATH`** - Defaults to `$HOME/go` if unset

## Breaking Changes

- 🟡 **Language** Struct tag changes: tags now ignored when explicitly converting between struct types
- 🟡 **Language** Floating-point exponents: spec now only requires 16-bit (was implicitly 32-bit)
- 🟡 **Runtime** Argument liveness: GC no longer keeps function arguments live throughout function
- 🟡 **cgo** `C.malloc` returning `NULL` now crashes with OOM (was returning `nil`)

## New Features

- 🔴 **net/http** HTTP/2 Server Push via `http.Pusher` interface
- 🔴 **net/http** Graceful shutdown via `Server.Shutdown` method
- 🔴 **plugin** New package for loading Go plugins at runtime (Linux only)
- 🔴 **sort** New `Slice`, `SliceStable`, `SliceIsSorted` convenience functions
- 🔴 **go command** New `go bug` command for creating GitHub bug reports
- 🔴 **go command** Default `GOPATH` now `$HOME/go` (Windows: `%USERPROFILE%/go`) if unset
- 🟡 **database/sql** Extensive context support with `*Context` methods
- 🟡 **database/sql** Transaction isolation level control via `TxOptions`
- 🟡 **database/sql** Multiple result sets support with `Rows.NextResultSet`
- 🟡 **runtime** Mutex contention profiling via `MutexProfile` and `SetMutexProfileFraction`

## Improvements

- 🟢 **Performance** GC pauses typically <100μs, often as low as 10μs (eliminated STW stack rescan)
- 🟢 **Performance** SSA backend: 20-30% faster on 32-bit ARM, 0-10% on 64-bit x86
- 🟢 **Performance** Defer overhead reduced by ~50%
- 🟢 **Performance** Cgo call overhead reduced by ~50%
- 🟢 **Performance** Compilation ~15% faster than 1.7
- 🟢 **compress/flate** 2.5x faster `BestSpeed`, new `HuffmanOnly` level (3x faster than `BestSpeed`)
- 🟢 **GC** Eliminated stop-the-world stack rescanning

## Tooling & Developer Experience

- 🔴 **go test** New `-mutexprofile` flag for mutex contention profiling
- 🟡 **go get** Now always respects HTTP proxy environment variables (even with `-insecure`)
- 🟡 **go doc** Groups constants/variables with their types, improved readability
- 🟡 **go doc** Can document specific interface methods: `go doc net.Conn.SetDeadline`
- 🟡 **go fix** New `context` fix changes imports from `golang.org/x/net/context` to `context`
- 🟡 **pprof** Can profile TLS servers and skip certificate validation with `https+insecure`
- 🟡 **trace** New `-pprof` flag produces pprof-compatible profiles from traces
- 🟡 **vet** Stricter checks: array of locks, duplicate struct tags, deferred `Response.Body.Close`, indexed `Printf`
- 🟢 **cgo** `CGO_ENABLED` value from `make.bash` now remembered and applied by default

## Platform & Environment

- 🟡 **Platform** 32-bit MIPS support on Linux (`linux/mips` and `linux/mipsle` with MIPS32r1 + FPU)
- 🟡 **Platform** DragonFly BSD now requires 4.4.4+
- 🟡 **Platform** OpenBSD now requires 5.9+
- 🟡 **Platform** OS X now requires 10.8+ (last release to support 10.8)
- 🟡 **Platform** Plan 9 networking now complete and matches Unix/Windows behavior
- 🟢 **Platform** Go 1.9 will drop support for Linux ARMv5E/ARMv6 (requires ARMv6K+)
- 🟢 **Known Issues** FreeBSD and NetBSD instabilities (crashes in rare cases)

## Implementation Details

- 🟢 **Runtime** `KeepAlive` needed for finalizer safety (args not kept live throughout function)
- 🟢 **Runtime** Concurrent map iteration+write misuse detection improved
- 🟢 **Runtime** `MemStats` type more thoroughly documented
- 🟢 **Compiler** SSA backend now used for all architectures
- 🟢 **Compiler** New compiler front end (foundation for future work)
- 🟢 **crypto/tls** `Conn.CloseWrite` for half-closing TLS connections
- 🟢 **crypto/tls** `Config.Clone` and `Config.GetConfigForClient` for dynamic configuration
- 🟢 **crypto/tls** X25519 and ChaCha20-Poly1305 support (ChaCha20 prioritized without AES-GCM hardware)
- 🟢 **database/sql** `IsolationLevel` control, `NamedArg` support, `Pinger` interface
- 🟢 **encoding/json** `UnmarshalTypeError` now includes struct/field name
- 🟢 **encoding/json** Floating-point encoding uses ES6 format (prefers decimal notation)
- 🟢 **math/big** `Int.Sqrt` method calculates integer square root
- 🟢 **net** `Buffers` type for efficient multi-buffer writes (optimized to `writev` on some systems)
- 🟢 **net** `Resolver` type for custom DNS resolution with context support
- 🟢 **time** `Until` function complements `Since`
