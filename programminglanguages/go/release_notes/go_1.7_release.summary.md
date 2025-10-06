### Summary of Go 1.7

Go 1.7 was a major release that introduced a new compiler backend for better performance, promoted the `context` package into the standard library, and added support for subtests.

**Language Features & Syntax**
*   There were no significant changes to the language, only a minor clarification to the specification regarding terminating statements.

**Performance Improvements**
*   A new **SSA-based compiler backend** for 64-bit x86 (`amd64`) was introduced, resulting in 5-35% better performance for many programs.
*   Compiled binaries are 20-30% smaller due to a new export data format and other optimizations.
*   The compiler and linker themselves are significantly faster than in Go 1.6.
*   GC pauses are shorter for programs with many idle goroutines.

**Tooling & Developer Experience**
*   The `testing` package now supports **subtests and sub-benchmarks** (`t.Run`, `b.Run`), enabling better test organization, table-driven tests, and sharing of setup/teardown logic.
*   Vendoring was finalized as a standard feature, and the `GO15VENDOREXPERIMENT` environment variable was removed.
*   On x86-64, frame pointers are now maintained by default, improving the experience of profiling Go programs with standard tools like `perf` and `VTune`.
*   `go vet` adds a check to find `context` cancellations that are created but never used.

**Major Library Updates**
*   The **`context` package** was moved from the external `golang.org/x/net` repository into the standard library, establishing it as the standard way to handle cancellation, timeouts, and request-scoped values.
*   The `net`, `net/http`, and `os/exec` packages were updated to integrate with and accept `context.Context`.
*   A new `net/http/httptrace` package was added to allow fine-grained tracing of events within an HTTP client request.