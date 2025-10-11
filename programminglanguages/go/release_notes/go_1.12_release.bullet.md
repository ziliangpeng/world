# Go 1.12 Release Notes

**Released:** February 25, 2019
**EOL:** August 2020 (expected)

## Major Highlights

Go 1.12 brings TLS 1.3 support, improved module experience, and runtime optimizations:

1. **TLS 1.3 opt-in support** - Enable with `GODEBUG=tls13=1` (default in Go 1.13)
2. **Build cache required** - `GOCACHE=off` no longer supported
3. **Improved sweeping performance** - Better allocation latency after GC when large heap remains live
4. **Better memory release** - Runtime releases memory to OS more aggressively
5. **Faster timers** - Timer and deadline code scales better with more CPUs
6. **Module enhancements** - Concurrent module downloads now safe, `go` directive indicates language version

## Breaking Changes

- 🔴 **Modules** Build cache now required - `GOCACHE=off` causes commands to fail
- 🟡 **Compiler** More aggressive inlining may affect code using `runtime.Callers` - use `runtime.CallersFrames`
- 🟡 **Compiler** Method expression wrappers no longer reported by `runtime.CallersFrames` and `runtime.Stack`

## Deprecations

- 🟡 **godoc** Command-line interface no longer included in binary distribution (use `go doc`)
- 🟡 **godoc** Web server will be available via `go get` in Go 1.13

## New Features

- 🔴 **TLS** Opt-in TLS 1.3 support via `GODEBUG=tls13=1` (RFC 8446)
- 🟡 **Compiler** `-lang` flag sets Go language version (e.g., `-lang=go1.8`)
- 🟡 **Assembler** Platform register renamed from `R18` to `R18_PLATFORM` on arm64
- 🟡 **go doc** New `-all` flag prints all exported APIs and documentation
- 🟡 **go doc** New `-src` flag shows target's source code
- 🟢 **Trace** Plotting mutator utilization curves with cross-references to execution trace

## Improvements

- 🟢 **Runtime** Significantly improved sweeping performance when large fraction of heap remains live
- 🟢 **Runtime** More aggressive memory release to OS, especially after large allocations
- 🟢 **Runtime** Faster timer and deadline code with better CPU scaling
- 🟢 **Runtime** Linux: `MADV_FREE` used by default (set `GODEBUG=madvdontneed=1` for old behavior)
- 🟢 **Runtime** Improved memory profile accuracy (fixed overcounting of large heap allocations)
- 🟢 **Runtime** Tracebacks during init show function named `PKG.init.ializers`
- 🟢 **Compiler** More functions eligible for inlining (including those calling panic)
- 🟢 **Compiler** Better debug information with improved argument printing and variable locations
- 🟢 **Compiler** Stack frame pointers on linux/arm64 (set `GOEXPERIMENT=noframepointer` to disable)
- 🟡 **Compiler** Compiler and assembly support new calling conventions (see ABI design doc)
- 🟢 **go command** `GODEBUG=cpu.*extension*=off` disables optional CPU instruction set extensions

## Tooling & Developer Experience

- 🟡 **Modules** Module downloads and extractions now safe to invoke concurrently
- 🟡 **Modules** `go` directive indicates language version (set to `go 1.12` if missing)
- 🟡 **Modules** `go` directive mismatch with older Go versions handled gracefully
- 🟡 **Modules** `replace` directives consulted when import cannot be resolved
- 🟡 **Modules** Operations outside module directory supported when `GO111MODULE=on`
- 🟡 **Binary-only** Last release supporting binary-only packages
- 🟡 **cgo** `EGLDisplay` translated to `uintptr`
- 🟡 **cgo** Mangled C names no longer accepted (use documented cgo names like `C.char`)
- 🟡 **vet** `-shadow` option no longer available (use `golang.org/x/tools/go/analysis/passes/shadow`)
- 🟡 **go tool vet** No longer supported (use `go vet`)

## Platform & Environment

- 🟡 **Platform** Race detector supported on linux/arm64
- 🟡 **Platform** FreeBSD 11.2+ or 12.0+ required (with COMPAT_FREEBSD11) - Go 1.12 last for FreeBSD 10.x
- 🟡 **Platform** cgo supported on linux/ppc64
- 🟡 **Platform** `hurd` recognized as GOOS value for GNU/Hurd (gccgo)
- 🟡 **Platform** Windows: windows/arm port for Windows 10 IoT Core on 32-bit ARM (Raspberry Pi 3)
- 🟡 **Platform** AIX: AIX 7.2+ on POWER8 (aix/ppc64) - no external linking, cgo, pprof, or race detector yet
- 🟡 **Platform** Darwin: macOS 10.10 Yosemite last supported (Go 1.13 requires 10.11+)
- 🟡 **Platform** Darwin: `libSystem` used for syscalls (better macOS/iOS compatibility)
- 🟡 **Platform** Darwin: `syscall.Getdirentries` fails with ENOSYS on iOS

## Standard Library Changes

### Major Changes

- 🔴 **TLS** TLS 1.3 opt-in support (enable with `GODEBUG=tls13=1`)
- 🟡 **TLS** Cipher suites not configurable in TLS 1.3 (all supported suites are safe)
- 🟡 **TLS** Early data (0-RTT mode) not supported
- 🟡 **TLS** Client is last to speak in TLS 1.3 handshake - errors returned on first Read
- 🟡 **bytes** New `ReplaceAll` function
- 🟡 **bytes** Zero-value `Reader` now equivalent to `NewReader(nil)`
- 🟡 **fmt** Maps printed in key-sorted order for easier testing
- 🟡 **fmt** Non-reflexive key values (like NaN) now printed correctly (not `<nil>`)

### net/http

- 🟡 **net/http** HTTP server rejects misdirected HTTP requests to HTTPS servers
- 🟡 **net/http** `Client.CloseIdleConnections` calls underlying Transport's method
- 🟡 **net/http** `Transport` handles HTTP trailers without chunked encoding
- 🟡 **net/http** `Transport` MAX_CONCURRENT_STREAMS back to Go 1.9 behavior (stricter in 1.10/1.11)
- 🟡 **net/http/httputil** `ReverseProxy` automatically proxies WebSocket requests

### reflect

- 🟡 **reflect** New `MapIter` type for ranging over maps
- 🟡 **reflect** `Value.MapRange` returns iterator for map traversal

### Other packages

- 🟡 **bufio** `Reader.UnreadRune`/`UnreadByte` return error after `Peek`
- 🟡 **crypto/rand** Warning printed if `Reader.Read` blocked >60 seconds
- 🟡 **crypto/rand** FreeBSD: uses `getrandom` syscall or `/dev/urandom`
- 🟡 **crypto/rc4** Assembly removed (pure Go only) - RC4 insecure, use only for compatibility
- 🟡 **crypto/tls** Server no longer replies with alert for non-TLS initial messages
- 🟡 **database/sql** `Row.Scan` can accept `*Rows` for query cursor
- 🟡 **expvar** `Map.Delete` deletes key/value pairs
- 🟡 **go/doc** New `PreserveAST` Mode bit controls AST data clearing
- 🟡 **go/token** `File.LineStart` returns position of line start
- 🟡 **image** `RegisterFormat` now safe for concurrent use
- 🟡 **image/png** Paletted images with <16 colors encode to smaller outputs
- 🟡 **io** New `StringWriter` interface wraps `WriteString`
- 🟡 **math** `Sin`, `Cos`, `Tan`, `Sincos` use Payne-Hanek range reduction for huge arguments
- 🟡 **math/bits** Extended precision operations: `Add`, `Sub`, `Mul`, `Div`
- 🟡 **net** `Dialer.DualStack` ignored and deprecated (RFC 6555 Fast Fallback enabled by default)
- 🟡 **net** TCP keep-alives enabled by default (set `Dialer.KeepAlive` to negative to disable)
- 🟡 **net** Linux: `splice` syscall used for UnixConn→TCPConn copying
- 🟡 **net/url** `Parse`/`ParseRequestURI`/`URL.Parse` reject URLs with ASCII control characters
- 🟡 **os** `ProcessState.ExitCode` returns process exit code
- 🟡 **os** `ModeCharDevice` added to `ModeType` bitmask
- 🟡 **os** `UserHomeDir` returns current user's home directory
- 🟡 **os** `RemoveAll` supports paths >4096 characters on most Unix
- 🟡 **os** `File.Sync` uses `F_FULLFSYNC` on macOS for correct flush (may be slower)
- 🟡 **os** `File.SyscallConn` returns `syscall.RawConn` for system-specific operations
- 🟡 **path/filepath** `IsAbs` returns true for Windows reserved filenames (e.g., `NUL`)
- 🟡 **runtime/debug** New `BuildInfo` type exposes build information from running binary
- 🟡 **strings** New `ReplaceAll` function
- 🟡 **strings** Zero-value `Reader` now equivalent to `NewReader(nil)`
- 🟡 **strings** `Builder.Cap` returns capacity of builder's underlying byte slice
- 🟡 **strings** Character mapping functions guarantee valid UTF-8 output
- 🟡 **syscall** FreeBSD: 64-bit inode support
- 🟡 **syscall** Windows: Unix socket (AF_UNIX) support on compatible versions
- 🟡 **syscall** Windows: `Syscall18` for calls with up to 18 arguments
- 🟡 **syscall/js** `Callback`/`NewCallback` renamed to `Func`/`FuncOf` (breaking for WebAssembly)
- 🟡 **syscall/js** New `Wrapper` interface for custom JavaScript value wrapping
- 🟡 **syscall/js** Zero `Value` represents JavaScript `undefined` (was number zero - breaking)
- 🟡 **syscall/js** `Value.Truthy` reports JavaScript truthiness
- 🟡 **testing** `-benchtime` flag supports explicit iteration count (e.g., `-benchtime=100x`)
- 🟡 **text/template** Long context values no longer truncated in errors
- 🟡 **text/template** User-defined function panics caught and returned as errors
- 🟡 **time** Time zone database updated to 2018i
- 🟡 **unsafe** Invalid: converting nil `unsafe.Pointer` to uintptr and back with arithmetic
