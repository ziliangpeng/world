### Summary of Go 1.1

Go 1.1 was a major performance-focused release, keeping the Go 1 compatibility promise while delivering significant speedups, a new race detector, and key language and library additions.

**Language Features & Syntax**
*   Introduced **Method Values**, allowing methods to be treated as functions bound to a specific receiver.
*   Relaxed the requirement for a final `return` statement in functions that end with a clear "terminating statement" (e.g., an infinite loop).
*   Integer division by a constant zero is now a compile-time error instead of a runtime panic.

**Performance Improvements**
*   This was a primary focus of the release, with many programs seeing 30-40% speed improvements simply by recompiling.
*   The garbage collector was made more parallel and more precise, reducing pause times and heap size.
*   A new map implementation significantly reduced memory footprint and CPU time.
*   The compilers were improved to generate better code and perform more in-lining.

**Tooling & Developer Experience**
*   A **Race Detector** was introduced to find concurrency bugs, enabled via the `go test -race` flag.
*   On 64-bit systems, `int` and `uint` became 64-bit types, allowing for much larger slice allocations.
*   The `go` command was updated to provide more detailed error messages for missing packages.
*   `go test` gained the ability to generate blocking profiles to debug goroutine stalls.

**Major Library Updates**
*   A new `bufio.Scanner` type was added to provide a simpler and safer way to scan text, such as reading input line-by-line.
*   The `reflect` package was significantly enhanced with support for `select` statements, type conversions, and dynamic function creation (`MakeFunc`).
*   The `time` package now provides nanosecond precision on many systems and adds `Round` and `Truncate` methods to manage this.