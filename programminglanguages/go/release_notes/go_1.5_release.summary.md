### Summary of Go 1.5

Go 1.5 was a landmark release featuring a complete rewrite of the compiler and runtime in Go, a new low-latency concurrent garbage collector, and a change in the default `GOMAXPROCS` to enable parallelism by default.

**Language Features & Syntax**
*   A minor update allows eliding the type from struct literal keys in maps (e.g., `map[Point]string{{1,2}: "a"}`).

**Performance Improvements**
*   The garbage collector was completely redesigned to be **concurrent**, dramatically reducing "stop-the-world" pause times to under 10ms in most cases, a major benefit for latency-sensitive applications.
*   *Build performance regressed* in this version, with compile times roughly doubling due to the compiler's translation from C to unidiomatic Go.
*   The default value for `GOMAXPROCS` was changed from 1 to the number of available CPU cores, making programs concurrent and parallel by default.

**Tooling & Developer Experience**
*   The entire toolchain (compiler, linker) and runtime are now **written in Go**, removing the need for a C compiler to build the Go distribution.
*   Experimental support for **vendoring** dependencies was added to the `go` command, paving the way for more reproducible builds.
*   A new `go tool trace` command was introduced for fine-grained execution tracing.
*   The "internal packages" feature for better code encapsulation was enabled for all repositories.

**Major Library Updates**
*   A new `math/big.Float` type was added to provide arbitrary-precision floating-point arithmetic.
*   The important `go/types` package was moved from an external repository into the standard library.
*   On most Unix systems, the DNS resolver now uses a native Go implementation instead of `cgo`, reducing resource usage for programs making many DNS requests.