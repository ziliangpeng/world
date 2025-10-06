### Summary of Go 1.3

Go 1.3 was a release focused on implementation work, bringing a more precise and concurrent garbage collector, faster compile times, and the introduction of `sync.Pool`.

**Language Features & Syntax**
*   There were no changes to the language itself in this release.

**Performance Improvements**
*   The garbage collector became fully precise (including for stacks) and uses a concurrent sweep algorithm, reducing GC pause times by 50-70%.
*   A major refactoring of the compiler toolchain (`liblink`) significantly sped up build times for large projects.
*   The runtime handles `defer` statements more efficiently, reducing memory usage.
*   The race detector was made approximately 40% faster.

**Tooling & Developer Experience**
*   Goroutine stacks were changed to a contiguous model instead of segmented stacks, improving performance and eliminating "hot spot" issues when stacks grow.
*   `godoc -analysis` was added to perform and display sophisticated static analysis of code, including call graphs and type relationships.
*   Cross-compilation is now supported even when `cgo` is enabled.

**Major Library Updates**
*   A new `sync.Pool` type was introduced to provide a scalable cache for temporary objects, helping to reduce GC overhead.
*   A critical security bug in `crypto/tls` was fixed, enforcing stricter certificate verification, which could break previously insecure code.
*   The `testing` package added `B.RunParallel` to help write benchmarks that can exercise multiple CPUs.