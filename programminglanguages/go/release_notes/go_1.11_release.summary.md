### Summary of Go 1.11

Go 1.11 was a major release that introduced preliminary experimental support for **modules**, the future of dependency management, and an experimental port to **WebAssembly** (`js/wasm`).

**Language Features & Syntax**
*   There were no changes to the language itself.

**Performance Improvements**
*   The compiler became significantly more aggressive with optimizations like **bounds-check elimination**, recognizing transitive relations to remove more checks.
*   The compiler now optimizes common patterns for clearing maps and extending slices with `append`.
*   On Linux, TCP proxying is faster and more efficient due to the automatic use of the `splice` system call when copying between TCP connections.

**Tooling & Developer Experience**
*   **Modules:** The headline feature is the introduction of preliminary, experimental support for **modules**, a new dependency management system that frees developers from `GOPATH` and enables reproducible builds.
*   **WebAssembly:** An experimental port to WebAssembly (`js/wasm`) was added, allowing Go code to be compiled to run in web browsers.
*   **Debugging:** The compiler now generates significantly more accurate debug information for optimized binaries, improving the debugging experience with tools like Delve.
*   The `go run` command can now be conveniently run on the current directory with `go run .`.
*   A new `GOFLAGS` environment variable was added to set default flags for `go` command invocations.

**Major Library Updates**
*   A new `syscall/js` package was added to facilitate calling JavaScript from Go code when using the WebAssembly port.
*   The `runtime/trace` package was enhanced with a user annotation API, allowing developers to add application-level context to execution traces.
*   The new `golang.org/x/tools/go/packages` package was introduced as the modern way to programmatically load and analyze Go packages, designed to work with both modules and GOPATH.