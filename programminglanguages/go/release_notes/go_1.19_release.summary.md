### Summary of Go 1.19

Go 1.19 revised the memory model, added support for a soft memory limit, and introduced new features for documentation comments.

**Language Features & Syntax**
*   No significant changes; only a minor correction to the scope of type parameters.

**Performance Improvements**
*   The compiler now uses jump tables for large `switch` statements, making them up to 20% faster.
*   The sorting algorithm was changed to Pattern-Defeating Quicksort, which is faster for many common data patterns.
*   The `riscv64` port is ~10% faster due to now using registers for function arguments.

**Tooling & Developer Experience**
*   **Doc Comments:** Documentation comments now support links, lists, and clearer headings. `gofmt` will now reformat doc comments to a canonical format.
*   **Soft Memory Limit:** The runtime now supports a soft memory limit (configurable via `GOMEMLIMIT`), which provides better control over memory usage and GC tuning.
*   On Unix systems, Go programs now automatically raise the open file limit (`rlimit`) to the maximum allowed, preventing many "too many open files" errors.

**Major Library Updates**
*   The **Go Memory Model** was formally revised to align with the memory models of C, C++, Java, and Rust.
*   New atomic types (e.g., `atomic.Int64`, `atomic.Pointer[T]`) were added to the `sync/atomic` package, providing a safer and more convenient way to use atomic operations.
*   A security improvement was made to `os/exec`: `LookPath` no longer finds executables in the current directory by default, preventing a class of path-based attacks.
*   A new `go/doc/comment` package was added to parse the new doc comment syntax.