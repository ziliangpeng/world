# Go 1.2 Release Notes

**Released:** December 1, 2013
**EOL:** N/A (maintenance ended)

## Major Highlights

Go 1.2 arrives six months after 1.1 with a shorter release cycle, focusing on scheduler improvements and a new language feature:

1. **Pre-emptive scheduler** - Goroutines can now be preempted at function calls, preventing starvation
2. **Three-index slices** - New `a[i:j:k]` syntax to control slice capacity
3. **Thread limit** - Configurable limit (default 10,000) on OS threads
4. **Test coverage tool** - New `go tool cover` for detailed test coverage analysis
5. **Stack size changes** - Minimum stack increased from 4KB to 8KB for better performance
6. **Indexed `Printf` arguments** - Access format arguments in arbitrary order

## Breaking Changes

- 🟡 **Language** Using `nil` pointers to access struct fields now guaranteed to panic (was undefined)
- 🟢 **archive/tar** `FileInfo.Name` now returns base name only (was full path)
- 🟢 **archive/zip** `FileInfo.Name` now returns base name only (was full path)

## New Features

- 🔴 **Language** Three-index slices: `slice = array[2:4:7]` specifies capacity in addition to length
- 🟡 **fmt** Indexed arguments: `Printf("%[3]c %[1]c %c\n", 'a', 'b', 'c')` prints "c a b"
- 🟡 **text/template** New comparison functions: `eq`, `ne`, `lt`, `le`, `gt`, `ge` for basic types
- 🟡 **text/template** "else if" chains: `{{if eq .A 1}} X {{else if eq .A 2}} Y {{end}}`
- 🟢 **encoding** New package defining `BinaryMarshaler`, `BinaryUnmarshaler`, `TextMarshaler`, `TextUnmarshaler` interfaces
- 🟢 **image/color/palette** New package with standard color palettes

## Improvements

- 🟢 **Performance** `compress/bzip2` decompression 30% faster
- 🟢 **Performance** `crypto/des` about 5x faster
- 🟢 **Performance** `encoding/json` encoding 30% faster
- 🟢 **Performance** Networking 30% faster on Windows and BSD (integrated network poller)
- 🟢 **Scheduler** Goroutines can be preempted at function calls to prevent starvation
- 🟢 **Runtime** Thread limit configurable via `runtime/debug.SetMaxThreads` (default 10,000)
- 🟢 **Runtime** Minimum goroutine stack increased from 4KB to 8KB
- 🟢 **Runtime** Maximum stack size configurable via `runtime/debug.SetMaxStack` (1GB on 64-bit, 250MB on 32-bit)

## Tooling & Developer Experience

- 🔴 **go test** New `go tool cover` for test coverage analysis and visualization
- 🟡 **go command** `go get -t` downloads test dependencies
- 🟡 **go doc** Command removed - use `godoc .` directly instead
- 🟡 **cgo** Now builds C++ code in linked libraries
- 🟡 **godoc** Moved to `go.tools` subrepository (binary still included in distribution)
- 🟡 **vet** Moved to `go.tools` subrepository (binary still included in distribution)

## Platform & Environment

- 🟢 **Compiler** Missing `package` clause now an error (previously assumed `package main`)
- 🟢 **ARM** External linking support (step toward shared libraries)
- 🟢 **ARM** Assembly: `R9` and `R10` must now be referred to by proper names `m` and `g`
- 🟢 **ARM** New `MOVBS` and `MOVHS` instructions as synonyms for signed moves

## Implementation Details

- 🟢 **encoding/gob** Now ignores channel and function fields in structs (previously errored)
- 🟢 **encoding/json** Always escapes ampersands as `\u0026` and corrects invalid UTF-8
- 🟢 **encoding/xml** Attributes in pointers can now be marshaled
- 🟢 **net** Build tag `netgo` forces pure Go networking (no cgo)
- 🟢 **net** `Dialer.DualStack` for RFC 6555 dual IP stack connections
- 🟢 **os/exec** `Cmd.StdinPipe` now returns embeddable type instead of `*os.File`
- 🟢 **io** `Copy` now prioritizes `WriterTo` over `ReaderFrom` when both are available
- 🟢 **sort** New `Stable` function for stable sorting (less efficient than regular sort)
- 🟢 **sync/atomic** New swap functions: `SwapInt32`, `SwapInt64`, `SwapUint32`, `SwapUint64`, `SwapUintptr`, `SwapPointer`
- 🟢 **testing** `AllocsPerRun` now quantizes return value to integer
- 🟢 **unicode** Updated to Unicode 6.2.0
