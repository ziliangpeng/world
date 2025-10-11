# Go 1.3 Release Notes

**Released:** June 18, 2014
**EOL:** N/A (maintenance ended)

## Major Highlights

Go 1.3 focuses on implementation improvements with no language changes, delivering major performance gains:

1. **Precise garbage collection** - GC now precise for both heap and stack, enabling better memory management
2. **Contiguous stacks** - Eliminates "hot stack split" problem with single-block stack allocation
3. **50-70% faster GC** - Concurrent sweep algorithm and better parallelization dramatically reduce pause times
4. **New platforms** - Support for DragonFly BSD, Solaris, Plan 9, and Native Client (NaCl)
5. **Linker overhaul** - New `liblink` library moves instruction selection to compiler, speeding builds
6. **`sync.Pool`** - New type for efficient, GC-reclaimable caches

## Breaking Changes

- 🟡 **Memory model** Clarified buffered channel behavior as semaphore (not a language change, just specification)
- 🟡 **Map iteration** Small maps (≤8 entries) now randomized like large maps - code depending on iteration order will break
- 🟢 **cgo** Incomplete struct pointer types now properly typed - code passing pointers across packages will break

## New Features

- 🔴 **sync** New `Pool` type for automatically reclaimable, thread-safe caches
- 🟡 **testing** `B.RunParallel` method for easier parallel benchmarks
- 🟢 **debug/plan9obj** New package for accessing Plan 9 a.out object files

## Improvements

- 🟢 **Performance** GC pauses reduced 50-70% through concurrent sweep and better parallelization
- 🟢 **Performance** Race detector 40% faster
- 🟢 **Performance** `regexp` significantly faster for simple expressions (two-pass engine)
- 🟢 **Performance** Defers more efficient, reducing memory footprint by ~2KB per goroutine
- 🟢 **Stacks** Contiguous stacks eliminate "hot split" problem, improving performance and predictability
- 🟢 **GC** Precise garbage collection on both heap and stack (previously only heap)
- 🟢 **GC** Stack dumps now show how long goroutines have been blocked

## Tooling & Developer Experience

- 🟡 **go command** New `-exec` flag for `go run` and `go test` to specify alternate binary execution (supports NaCl)
- 🟡 **go test** Coverage mode automatically set to `-atomic` when race detector enabled
- 🟡 **go test** Always builds package even with no test files
- 🟡 **go build** New `-i` flag installs dependencies but not the target itself
- 🟡 **cgo** Cross-compiling with cgo now supported via `CC_FOR_TARGET` and `CXX_FOR_TARGET`
- 🟡 **cgo** Now supports Objective-C files (.m) in packages
- 🟡 **godoc** New `-analysis` flag performs sophisticated static analysis with call graphs

## Platform & Environment

- 🟡 **Platform** Windows 2000 no longer supported
- 🟡 **Platform** DragonFly BSD support on amd64 and 386 (requires DragonFly BSD 3.6+)
- 🟡 **Platform** FreeBSD now requires FreeBSD 8+ and `COMPAT_FREEBSD32` kernel flag
- 🟡 **Platform** FreeBSD on ARM now requires FreeBSD 10 (due to EABI syscalls)
- 🟡 **Platform** Native Client (NaCl) support on 386 and amd64p32 architectures
- 🟡 **Platform** NetBSD now requires NetBSD 6.0+
- 🟡 **Platform** OpenBSD now requires OpenBSD 5.5+
- 🟡 **Platform** Plan 9 support on 386 (requires `Tsemacquire` syscall from June 2012+)
- 🟡 **Platform** Solaris support on amd64 (requires illumos or Solaris 11+)

## Implementation Details

- 🟢 **Runtime** Contiguous stack implementation replaces segmented stacks
- 🟢 **Linker** New `liblink` library moves instruction selection from linker to compiler
- 🟢 **crypto/tls** TLS bug fix: must specify `ServerName` or `InsecureSkipVerify` (previously could skip inadvertently)
- 🟢 **cgo** Pointers to incomplete structs now properly typed (was `*[0]byte`)
- 🟢 **SWIG** Now requires SWIG 3.0 and links object files directly into binary
- 🟢 **Assembler** Now uses Go flag parsing rules (e.g., `go tool 6a -S -D foo` instead of `go tool 6a -SDfoo`)
- 🟢 **unsafe** Code storing integers in pointer-typed values is now illegal and will crash if detected
- 🟢 **unicode** Updated to Unicode 6.3.0
