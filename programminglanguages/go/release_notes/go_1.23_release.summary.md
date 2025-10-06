### Summary of Go 1.23

Go 1.23 introduced a major new language feature, **range over functions**, for custom iteration. It also significantly changed the behavior of `time.Timer` and `time.Ticker` to be safer and more efficient.

**Language Features & Syntax**
*   The `for...range` loop can now iterate over functions of specific signatures. This provides a general mechanism for creating custom iterators for any data structure.

**Performance Improvements**
*   The build-time overhead of using Profile-Guided Optimization (PGO) has been significantly reduced, making it much more practical for large projects.
*   The compiler now reduces stack usage by overlapping stack frames for variables in disjoint scopes.
*   PGO now enables hot block alignment in loops on x86, improving performance by an additional 1-1.5%.

**Tooling & Developer Experience**
*   An opt-in **Go Telemetry** feature was added (`go telemetry`) to help the Go team gather usage statistics.
*   A new `godebug` directive can be added to `go.mod` files to associate specific `GODEBUG` settings with a module.
*   The `vet` tool now warns if code uses standard library features that are newer than the version specified in the module's `go.mod` file.

**Major Library Updates**
*   The behavior of **`time.Timer` and `time.Ticker` was changed significantly** for modules using `go 1.23` or newer. Unstopped timers are now garbage collected, and their channels are unbuffered, preventing common race conditions with `Reset` and `Stop`.
*   A new **`iter` package** was added to support the new range-over-function feature.
*   The `slices` and `maps` packages were updated with functions that produce and consume iterators.
*   A new **`unique` package** was added for canonicalizing values (value interning) to save memory by deduplicating equal values.