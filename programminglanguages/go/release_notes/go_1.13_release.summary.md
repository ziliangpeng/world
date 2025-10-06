### Summary of Go 1.13

Go 1.13 introduced formal support for error wrapping, enabled modules and their associated services by default, and modernized number literal syntax.

**Language Features & Syntax**
*   **Number Literals:** The syntax was expanded to include binary literals (`0b...`), a new octal prefix (`0o...`), hexadecimal floating-point literals, and the use of underscores as digit separators (e.g., `1_000_000`).
*   **Shift Counts:** The restriction that the right-hand side of a shift operator (`<<`, `>>`) must be an unsigned integer was removed.

**Performance Improvements**
*   The performance of most uses of `defer` was improved by 30%.
*   A more precise escape analysis implementation reduces heap allocations by allowing more variables to be allocated on the stack.
*   The runtime is more aggressive about returning memory to the OS.
*   `sync.Mutex` and `sync.Once` are up to 10% and 2x faster, respectively, in uncontended cases due to inlining.

**Tooling & Developer Experience**
*   **Modules are on by default** whenever a `go.mod` file is found, even inside `GOPATH`.
*   The `go` command now defaults to using the public **module mirror and checksum database**, improving build speed and security.
*   New environment variables like `GOPRIVATE` were introduced for managing private modules.
*   A new `go env -w` command allows setting persistent defaults for Go environment variables.

**Major Library Updates**
*   **Error Wrapping:** The standard library added formal support for wrapping errors. This includes the new `%w` verb in `fmt.Errorf` and the `errors.Is` and `errors.As` functions for inspecting error chains.
*   **TLS 1.3** is now enabled by default in the `crypto/tls` package.
*   A new `crypto/ed25519` package was added to the standard library.