# Go 1.4 Release Notes

**Released:** December 10, 2014
**EOL:** N/A (maintenance ended)

## Major Highlights

Go 1.4 prepares the foundation for a fully concurrent garbage collector and includes major tooling additions:

1. **Fully precise GC** - Runtime rewritten in Go, enabling precise garbage collection and 10-30% smaller heaps
2. **Contiguous stacks** - Eliminates "hot split" problem, with default stack size reduced from 8KB to 2KB
3. **`go generate`** - New command for automated code generation before compilation
4. **Internal packages** - New mechanism to restrict package imports within subtrees
5. **Canonical imports** - Prevent multiple import paths for same package
6. **Variable-free `for range`** - Can now write `for range x` without dummy variables

## Breaking Changes

- 🟡 **Language** Method calls on `**T` (pointer-to-pointer) now disallowed per spec
- 🟡 **bufio** `Scanner` split functions now called once at EOF after input exhausted
- 🟡 **syscall** Package frozen except for core repository needs - use `golang.org/x/sys` for new features
- 🟢 **Runtime** Interface values now always hold a pointer (more allocations for stored integers)

## New Features

- 🔴 **Language** Variable-free `for range` loops: `for range x { ... }` now legal
- 🔴 **go command** New `go generate` command for source code generation (e.g., running yacc, stringer)
- 🔴 **Tooling** Internal packages: packages in `internal/` directories only importable from parent tree
- 🔴 **Tooling** Canonical import paths via package annotation: `package pdf // import "rsc.io/pdf"`
- 🟡 **Subrepositories** New import paths: `golang.org/x/tools` replaces `code.google.com/p/go.tools` (transition June 2015)

## Improvements

- 🟢 **Performance** Heap size reduced 10-30% due to precise GC and interface implementation changes
- 🟢 **Performance** Contiguous stacks improve predictability and eliminate "hot split" problem
- 🟢 **GC** Fully precise GC on both heap and stack (runtime rewritten in Go)
- 🟢 **Stacks** Default goroutine stack reduced from 8KB to 2KB
- 🟢 **Runtime** Much of runtime translated from C to Go for better GC awareness

## Tooling & Developer Experience

- 🔴 **go generate** Automate source code generation with `//go:generate` directives
- 🟡 **go test** New `-o` flag to set output binary name
- 🟡 **go build** `-a` flag no longer rebuilds standard library in release distributions
- 🟡 **go command** Now refuses to compile C source files unless using cgo
- 🟡 **Assembly** `textflag.h` now in standard location for including flags
- 🟡 **File naming** Build constraint tags now require underscore prefix (e.g., `os_windows.go` not `windows.go`)

## Platform & Environment

- 🟡 **Platform** Android support on ARM (can build `.so` libraries loadable by Android apps)
- 🟡 **Platform** Native Client (NaCl) on ARM
- 🟡 **Platform** Plan 9 on AMD64 (requires `nsec` syscall and 4K pages)
- 🟢 **Package layout** Source moved from `src/pkg` to `src` (flatter hierarchy)
- 🟢 **SWIG** Now requires SWIG 3.0

## Implementation Details

- 🟢 **Runtime** Rewritten in Go (was C) for precise GC and better stack scanning
- 🟢 **Runtime** Write barriers for heap pointer writes (preparation for concurrent GC in 1.5)
- 🟢 **Runtime** Interface implementation changed - values always hold pointers
- 🟢 **Runtime** Can set `GODEBUG=invalidptr=0` to disable nil pointer checks as workaround
- 🟢 **Compiler** Assembly source requires type information for runtime GC
- 🟢 **sync/atomic** New `Value` type for atomic loads/stores of arbitrary types
- 🟢 **testing** New `TestMain(m *testing.M)` function for custom test control
- 🟢 **testing** New `Coverage` function reports current test coverage fraction
- 🟢 **time** Microsecond duration now uses µ (U+00B5) symbol instead of "us"
