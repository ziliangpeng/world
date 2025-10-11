# Go 1 Release Notes

**Released:** March 28, 2012
**EOL:** N/A (historic release)

## Major Highlights

Go 1 establishes a stable foundation with a formal compatibility guarantee, fundamentally reshaping Go from an experimental language into a production-ready platform:

1. **Go 1 Compatibility Guarantee** - Code written for Go 1 will compile and run without changes throughout the Go 1.x lifetime
2. **New built-in types: `rune` and `error`** - `rune` as `int32` alias for Unicode code points, `error` as standard interface for errors
3. **Package reorganization** - Major restructuring into logical hierarchies (`encoding/*`, `net/http`, `text/*`, `unicode/*`)
4. **Complete `time` package redesign** - Type-safe `time.Time` and `time.Duration` replacing `int64` nanoseconds
5. **The `go` command** - New unified tool replacing makefiles for building, testing, and installing Go code
6. **Map iteration randomization** - Deliberately unpredictable iteration order to prevent fragile code
7. **New `append` functionality** - Can now append strings directly to `[]byte` slices

## Breaking Changes

### Language Changes

- 🔴 **Language** `close()` on receive-only channels now compile-time error
- 🔴 **Language** `delete(m, k)` built-in function replaces `m[k] = value, false` syntax for map deletion
- 🔴 **Language** Naked `return` statements with shadowed named return values now rejected by compiler
- 🔴 **Language** Function equality removed except for comparison with `nil` - closures cannot be compared
- 🔴 **Language** Map equality removed except for comparison with `nil`
- 🟡 **Language** Map iteration order deliberately randomized - code depending on order will break
- 🟡 **Language** Struct and array equality now defined element-wise - can now be used as map keys
- 🟡 **Language** Goroutines in `init()` now start during initialization, not after
- 🟡 **Language** Multiple assignment evaluation order now strictly defined (left-to-right)

### Package Hierarchy Reorganization

- 🔴 **Packages** Massive package reorganization - see detailed migration table below
- 🔴 **stdlib** `asn1`, `csv`, `gob`, `json`, `xml` → `encoding/*`
- 🔴 **stdlib** `http` → `net/http`, `rpc` → `net/rpc`, `url` → `net/url`
- 🔴 **stdlib** `big`, `cmath`, `rand` → `math/big`, `math/cmplx`, `math/rand`
- 🔴 **stdlib** `template` → `text/template`, `exp/template/html` → `html/template`
- 🔴 **stdlib** `utf8`, `utf16` → `unicode/utf8`, `unicode/utf16`
- 🔴 **stdlib** `exec` → `os/exec`, `scanner` → `text/scanner`

### Deleted Packages

- 🔴 **stdlib** Removed `container/vector` - use slices directly
- 🔴 **stdlib** Removed `exp/datafmt`, `go/typechecker`, `old/regexp`, `old/template`, `try`
- 🔴 **stdlib** Removed `gotry` command

### Packages Moved to Subrepositories

- 🟡 **stdlib** Crypto packages (`bcrypt`, `blowfish`, etc.) → `code.google.com/p/go.crypto/*`
- 🟡 **stdlib** Image formats (`bmp`, `tiff`) → `code.google.com/p/go.image/*`
- 🟡 **stdlib** Network packages (`websocket`, `dict`) → `code.google.com/p/go.net/*`

### Error Handling Changes

- 🔴 **error** New built-in `error` interface replaces `os.Error`
- 🔴 **error** `error` interface uses `Error()` method instead of `String()`
- 🔴 **error** New `errors.New()` function replaces `os.NewError()`
- 🔴 **syscall** System call errors now return `error` type, not `int` values
- 🔴 **os** Removed POSIX error constants (`EINVAL`, etc.) - use `IsExist()`, `IsNotExist()`, `IsPermission()`

### Time Package Redesign

- 🔴 **time** Complete API redesign - `time.Time` value type replaces `*time.Time` and `int64` nanoseconds
- 🔴 **time** New `time.Duration` type for intervals (not just `int64`)
- 🔴 **time** `time.Now()` returns `time.Time`, not nanosecond count
- 🔴 **time** Unix epoch only relevant for `time.Unix()`, `Time.Unix()`, `Time.UnixNano()` methods

### OS Package Changes

- 🔴 **os** `os.FileInfo` changed from struct to interface with system-specific `Sys()` method
- 🔴 **os** New `os.FileMode` type replaces mode fields in `FileInfo`
- 🔴 **os** Removed `os.Time` - use `time.Time`
- 🔴 **os** Removed `os.Exec` - use `syscall.Exec`
- 🔴 **os** `ShellExpand` renamed to `ExpandEnv`
- 🔴 **os** `NewFile()` and `File.Fd()` now use `uintptr` instead of `int`
- 🔴 **os** Removed `Getenverror` - use `os.Environ()` or `syscall.Getenv()`
- 🔴 **os** `Process.Wait()` drops option argument, returns `ProcessState` instead of `Waitmsg`

### Other Package Breaking Changes

- 🔴 **net** `SetTimeout` methods replaced with `SetDeadline`, `SetReadDeadline`, `SetWriteDeadline` using `time.Time`
- 🔴 **strconv** Major API overhaul - `Atof32`, `Atoi64`, `Btoi64`, etc. → `ParseFloat`, `ParseInt`, `FormatInt`, etc.
- 🔴 **url** `url.URL` fields removed/changed - `Raw`, `RawUserinfo`, `RawAuthority`, `RawPath` removed
- 🔴 **url** New `User` field of type `*Userinfo` replaces `RawUserinfo`
- 🔴 **url** `OpaquePath` removed, new `Opaque` field for non-rooted URLs
- 🟡 **archive/zip** `*zip.Writer` no longer has `Write` method
- 🟡 **bufio** `NewReaderSize`/`NewWriterSize` no longer return error - invalid sizes are adjusted
- 🟡 **compress** `NewWriterXxx` functions return `(*Writer, error)` for functions taking compression level
- 🟡 **crypto/aes** Removed `Reset` method and cipher-specific types - use `cipher.Block`
- 🟡 **crypto/hmac** Hash-specific functions removed - use `hmac.New()` with hash function
- 🟡 **encoding/binary** `binary.TotalSize` replaced with `Size()` taking `interface{}`
- 🟡 **encoding/xml** `Parser` renamed to `Decoder`, field tag format changed to match `json` package
- 🟡 **flag** `flag.Value.Set()` now returns `error` instead of `bool`
- 🟡 **hash** `hash.Hash.Sum()` now takes `[]byte` argument to append to
- 🟡 **http** HTTP utilities moved to `net/http/httputil` subdirectory
- 🟡 **image** Color handling moved to `image/color` package, `ycbcr` folded into `image`
- 🟡 **os/signal** `signal.Incoming()` replaced with selective `signal.Notify()`
- 🟡 **path/filepath** `Walk()` takes `WalkFunc` function instead of `Visitor` interface
- 🟡 **regexp** Complete rewrite using RE2 syntax instead of egrep
- 🟡 **runtime** Removed `runtime.Type` (use `reflect`), `Semacquire`/`Semrelease` (use channels/`sync`)
- 🟡 **unsafe** Removed `unsafe.Typeof`, `Reflect`, `Unreflect`, `New`, `NewArray` - use `reflect` package

## New Features

### Language Features

- 🔴 **Language** New built-in `error` interface type with `Error() string` method
- 🔴 **Language** New built-in `rune` type (alias for `int32`) for Unicode code points
- 🔴 **Language** Character literals (`'a'`) now have type `rune` instead of `int`
- 🔴 **Language** New built-in `delete(m, k)` function for deleting map entries
- 🔴 **Language** Composite literals can elide type for pointer elements: `[]*Date{{"Feb", 14}}`
- 🔴 **Language** `append()` can now append strings directly to `[]byte`: `append(b, "string"...)`
- 🟡 **Language** Struct and array values can now be compared and used as map keys

### Core Packages

- 🔴 **errors** New `errors` package with `errors.New(text string) error` function
- 🔴 **time** Type-safe time handling with `time.Time` and `time.Duration` types
- 🔴 **time** Predefined duration constants: `time.Second`, `time.Minute`, etc.
- 🔴 **time** `Time.Add(Duration)`, `Time.Sub(Time)` methods for time arithmetic

### Standard Library

- 🟡 **flag** New `flag.Duration` type for time interval flags
- 🟡 **net** `net.DialTimeout()` for timing out connection attempts
- 🟡 **net** `net.ListenMulticastUDP()` replaces `JoinGroup`/`LeaveGroup`
- 🟡 **runtime** `runtime.NumCPU()` returns number of CPUs available
- 🟡 **runtime** `runtime.ReadMemStats()` replaces global `MemStats` variable
- 🟡 **url** `url.RequestURI()` method added to `URL`

## Improvements

- 🟢 **go/scanner** Removed `AllowIllegalChars` and `InsertSemis` modes - use `text/scanner` for non-Go text
- 🟢 **go/parser** Reduced to primary `ParseFile()` and convenience functions `ParseDir()`, `ParseExpr()`
- 🟢 **go/doc** Type names streamlined: `PackageDoc` → `Package`, `ValueDoc` → `Value`
- 🟢 **go/build** API nearly completely replaced - `DirInfo` → `Package`, new `Import()`/`ImportDir()`

## Tooling & Developer Experience

- 🔴 **go command** New `go` command replaces makefiles for building, testing, installing packages
- 🔴 **go command** `go get`, `go build`, `go test`, `go install` commands
- 🔴 **go command** Automatic dependency resolution from Go source code
- 🟡 **go fix** Automated tool to update code from r60 to Go 1
- 🟡 **cgo** New `_cgo_export.h` file generation for packages with `//export` lines

## Implementation Details

- 🟢 **runtime** `MemStats` changed from global variable to struct type with `ReadMemStats()`
- 🟢 **runtime** `Cgocalls` and `Goroutines` renamed to `NumCgoCall` and `NumGoroutine`
- 🟢 **crypto/elliptic** `elliptic.Curve` changed to interface from concrete type
- 🟢 **crypto/x509** `CreateCertificate` and `CreateCRL` take `interface{}` for future algorithm support

## Platform & Environment

- 🟡 **Platform** Packaged downloadable distributions for multiple OS/architecture combinations
- 🟡 **Platform** Windows support included in official distributions

## Release Process & Meta Changes

- 🔴 **Release** Go 1 Compatibility Guarantee - code that compiles in Go 1 will continue to work in Go 1.x releases
- 🔴 **Release** Stable language and core library foundation for production use
- 🔴 **Release** Time scale measured in years for compatibility

## Migration Notes

### Using `go fix`

The `go fix` tool automates most migrations from r60 to Go 1:
- Package import path updates
- Function and method renamings
- Error handling changes
- Time package updates (partial)

### Manual Updates Required

- Map iteration order dependencies
- `close()` on receive-only channels
- Function/map equality comparisons
- System-specific `FileInfo` usage
- Experimental packages (`exp/*`)
- Deprecated packages (`old/*`)
- Deleted packages (especially `container/vector`)
- Regular expressions (verify RE2 compatibility)
- Template code using multiple template sets
- Time package updates (some cases)
- URL field access patterns
