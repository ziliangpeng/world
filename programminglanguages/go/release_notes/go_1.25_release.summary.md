### Summary of Go 1.25

This release brings no new language features but focuses heavily on performance, tooling, and library updates.

**Language Features & Syntax**
*   There are no changes to the language itself, maintaining the Go 1 compatibility promise.

**Performance Improvements**
*   **Garbage Collector:** An experimental Garbage Collector (`greenteagc`) is available, promising a 10-40% reduction in GC overhead.
*   **Compiler:** The compiler now allocates slices on the stack more frequently. It also uses fused multiply-add instructions on `amd64` (v3+) for faster and more accurate floating-point math.
*   **Crypto:** Significant speedups have been made in various `crypto` packages (RSA, SHA-1, etc.).
*   **Linker:** Linking is faster with smaller binaries due to the adoption of DWARFv5 for debug information.

**Tooling & Developer Experience**
*   **Runtime:** `GOMAXPROCS` is now container-aware on Linux, automatically respecting cgroup CPU limits. A new "flight recorder" for tracing (`runtime/trace.FlightRecorder`) allows for lightweight, continuous profiling.
*   **Debugging:** A critical compiler bug (present since Go 1.21) that could hide `nil` pointer panics has been fixed. A new `GODEBUG` flag (`checkfinalizers=1`) helps debug finalizers.
*   **Go Command & Vet:** The `go vet` tool adds new checks for common `sync.WaitGroup` and IPv6 host:port bugs. The `go doc -http` command was added to easily start a local documentation server.

**Major Library Updates**
*   **`testing/synctest`:** A new package to simplify testing concurrent code by providing a virtualized clock.
*   **`encoding/json/v2`:** A new experimental, high-performance JSON package is now available.
*   **`sync`:** The `sync.WaitGroup` type now has a `Go` method for more convenient and safer goroutine management.
*   **`net/http`:** The new `CrossOriginProtection` feature helps defend against Cross-Site Request Forgery (CSRF) attacks.
*   **`crypto/tls`:** SHA-1 signatures are now disabled by default in TLS 1.2 handshakes, improving security.