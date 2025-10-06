### Summary of Go 1.12

Go 1.12 introduced opt-in support for TLS 1.3, improved Modules support, and made `fmt` print maps in a deterministic sorted order.

**Language Features & Syntax**
*   There were no changes to the language itself.

**Performance Improvements**
*   The runtime is more aggressive about releasing memory back to the operating system.
*   The performance of sweeping the heap after a garbage collection is significantly improved, reducing allocation latency.
*   The implementation of timers and deadlines is faster and scales better on multiple CPUs, improving the performance of network connection deadlines.

**Tooling & Developer Experience**
*   **Modules:** Support for modules was improved, allowing commands like `go get` to be run in a module-aware mode even when outside a module directory. The `go` directive in `go.mod` now formally declares the language version for that module.
*   **`go vet`:** The `vet` command was rewritten on top of the `go/analysis` framework, making it a better platform for building custom static analysis tools. `go tool vet` is removed.
*   **`godoc` Deprecation:** The `godoc` command-line tool was removed, and the web server was slated for removal from the main distribution, with `go doc` becoming the standard for command-line documentation.

**Major Library Updates**
*   Opt-in support for **TLS 1.3** was added to the `crypto/tls` package, enabled via the `GODEBUG=tls13=1` environment variable.
*   The `fmt` package now prints **maps in a deterministic, key-sorted order**, which is a major quality-of-life improvement for creating stable test output and debugging.
*   The `reflect` package added `MapIter`, providing a structured way to iterate over maps via reflection that mimics the semantics of a `for...range` loop.