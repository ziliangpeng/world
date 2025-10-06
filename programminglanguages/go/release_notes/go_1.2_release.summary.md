### Summary of Go 1.2

Go 1.2 introduced test coverage tooling, a new three-index slice syntax, and scheduler improvements to prevent goroutine starvation.

**Language Features & Syntax**
*   A **three-index slice** syntax (`slice[low:high:max]`) was added to allow setting a slice's capacity during a slicing operation, providing more control over memory and allocations.
*   The language specification was tightened to guarantee that dereferencing a `nil` pointer will always cause a panic, improving program safety and predictability.

**Performance Improvements**
*   The scheduler was improved to allow pre-emption of long-running goroutines in loops, improving concurrency fairness and preventing starvation.
*   The minimum goroutine stack size was increased from 4KB to 8KB to reduce expensive stack-switching overhead in performance-critical code.
*   Notable speedups were made in several libraries, including `compress/bzip2` (~30%), `crypto/des` (~5x), and `encoding/json` (~30%).

**Tooling & Developer Experience**
*   A major **test coverage** feature was added. `go test -cover` instruments code to measure statement coverage, and a new `go tool cover` can be used to analyze the results.
*   A configurable limit on the number of OS threads was introduced to prevent resource exhaustion.
*   The `go get` command gained a `-t` flag to also fetch test dependencies.

**Major Library Updates**
*   The `fmt` package now supports **indexed arguments** (e.g., `%[2]d`) in `Printf` format strings, allowing arguments to be reordered, which is particularly useful for localization.
*   The `text/template` and `html/template` packages added built-in comparison functions (`eq`, `ne`, `lt`, etc.) and support for `else if` actions.
*   A new `encoding` package was introduced to define standard marshaling interfaces.