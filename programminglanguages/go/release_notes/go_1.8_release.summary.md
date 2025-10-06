### Summary of Go 1.8

Go 1.8 delivered major performance gains by enabling the SSA compiler backend for all architectures and further reducing GC pause times. It also introduced key features like graceful HTTP server shutdown and simplified slice sorting.

**Language Features & Syntax**
*   A minor change allows conversions between struct types that differ only in their field tags.

**Performance Improvements**
*   The new **SSA compiler backend** is now used for all architectures, bringing significant performance improvements (e.g., 20-30% on 32-bit ARM).
*   **Garbage collection pauses were dramatically reduced** (often to under 100 microseconds) by eliminating stop-the-world stack re-scanning.
*   The overhead of `defer` statements and `cgo` calls was reduced by about half.
*   The compiler itself is about 15% faster than in Go 1.7.

**Tooling & Developer Experience**
*   `GOPATH` now **defaults to a standard location** (`$HOME/go`) if unset, simplifying setup for new users.
*   Added support for **profiling mutex contention** via `go test -mutexprofile`.
*   A new `go bug` command was added to streamline the process of opening bug reports.
*   Early support for building and loading **plugins** was added for Linux.

**Major Library Updates**
*   The `net/http` server now supports **graceful shutdown** via the `server.Shutdown` method.
*   The `sort` package added `sort.Slice`, a much more convenient way to sort slices using a custom `less` function.
*   The `database/sql` package was heavily updated to add `context.Context` support to most operations.
*   HTTP/2 Server Push support was added to the `net/http` package.