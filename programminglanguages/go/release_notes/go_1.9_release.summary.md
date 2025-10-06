### Summary of Go 1.9

Go 1.9 introduced type aliases for large-scale refactoring, parallel compilation of functions within a package, and a new concurrent-safe `sync.Map`.

**Language Features & Syntax**
*   The headline feature is **type aliases** (`type T1 = T2`), which create an alternate name for a type to facilitate gradual code repair and refactoring across packages.

**Performance Improvements**
*   The compiler now compiles functions **within a single package in parallel**, speeding up the build process on multi-core machines.
*   Large object allocation is significantly faster for applications with very large heaps (>50GB).
*   Functions that manually trigger garbage collection (like `runtime.GC`) now trigger the concurrent GC instead of a stop-the-world GC, blocking only the calling goroutine.

**Tooling & Developer Experience**
*   The `testing` package added **`t.Helper()`**, a function that marks a test function as a helper. When a test fails inside a helper function, the line number reported is from the caller, which greatly improves test failure diagnostics.
*   The `runtime/pprof` package added support for **profiler labels**, allowing developers to filter and group profile data with custom key-value tags.
*   The `./...` pattern in `go` commands no longer matches packages inside the `vendor` directory by default.

**Major Library Updates**
*   A new **`sync.Map`** type was introduced, providing a concurrent map that is safe for multiple goroutines and optimized for read-heavy workloads where keys are written once.
*   A new **`math/bits`** package was added, offering optimized, intrinsic functions for bit counting and manipulation (e.g., `LeadingZeros`, `ReverseBytes`).
*   The `time` package now uses **monotonic clocks** transparently, making duration calculations with `time.Sub` robust against wall-clock adjustments (e.g., from NTP).