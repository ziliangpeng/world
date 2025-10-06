### Summary of Go 1.16

Go 1.16 introduced file embedding via the `//go:embed` directive, a new file system interface (`io/fs`), and enabled module-aware mode by default for all builds.

**Language Features & Syntax**
*   A new `//go:embed` directive allows developers to embed static files and file trees directly into a Go binary at compile time.

**Performance Improvements**
*   The second phase of linker improvements was completed, resulting in 20-25% faster linking, 5-15% less memory usage, and smaller binaries across all supported platforms.
*   `strconv.ParseFloat` is up to 2x faster due to a new parsing algorithm.
*   On Linux, the runtime now releases memory back to the OS more promptly, making process memory statistics more accurate.

**Tooling & Developer Experience**
*   **Module-aware mode is now enabled by default** for all builds (`GO111MODULE=on`).
*   `go install cmd@version` is the new, recommended way to install executables, without affecting the current module's dependencies.
*   Build commands like `go build` and `go test` are now **read-only by default** and will not modify `go.mod` or `go.sum`.
*   Added official support for **Apple Silicon** (`darwin/arm64`).

**Major Library Updates**
*   A new **`embed` package** provides access to files embedded with the `//go:embed` directive.
*   A major new **`io/fs` package** introduces a standard interface for read-only file systems (`fs.FS`). Many standard library packages were updated to use it.
*   The `io/ioutil` package is now **deprecated**, with its functions moved to more logical locations in the `io` and `os` packages.
*   A new `runtime/metrics` package provides a stable and efficient interface for reading runtime metrics.