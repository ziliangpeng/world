### Summary of Go 1.17

Go 1.17 introduced a new register-based calling convention for improved performance on amd64, and "pruned module graphs" to make module operations faster and more reliable.

**Language Features & Syntax**
*   Added a new conversion from a slice to an array pointer (e.g., `*[4]byte(mySlice)`). This is the first type conversion in Go that can panic at runtime.
*   Added `unsafe.Add` and `unsafe.Slice` to simplify writing correct `unsafe` pointer arithmetic.

**Performance Improvements**
*   A new **register-based calling convention** was implemented for Go functions on 64-bit x86 platforms, resulting in a typical performance improvement of ~5% and smaller binary sizes.
*   `strconv.FormatFloat` is significantly faster due to a new algorithm, which also speeds up `encoding/json`.
*   The `crypto/ed25519` and `crypto/elliptic` (P-521) packages are now much faster.

**Tooling & Developer Experience**
*   **Pruned Module Graphs:** For modules specifying `go 1.17`, the `go.mod` file now includes an expanded list of dependencies. This allows the `go` command to avoid downloading the `go.mod` files of most dependencies, speeding up many commands.
*   A new **`//go:build` constraint syntax** was introduced as a more readable and less error-prone replacement for `// +build` lines. `gofmt` automatically synchronizes them.
*   `go run` now supports version suffixes (e.g., `go run example.com/cmd@latest`), allowing commands to be run without installation.

**Major Library Updates**
*   **URL Query Parsing:** The `net/url` and `net/http` packages no longer accept semicolons (`;`) as query parameter separators by default, a breaking change made to improve security.
*   The `crypto/tls` package now enforces stricter ALPN protocol validation to protect against cross-protocol attacks.
*   The release announced that Go 1.18 will deprecate client-side TLS 1.0/1.1 and SHA-1 certificates.