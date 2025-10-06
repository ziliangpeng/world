### Summary of Go 1.6

Go 1.6 enabled vendoring by default, introduced transparent HTTP/2 support, and added stricter runtime checks for `cgo` pointer usage and concurrent map access.

**Language Features & Syntax**
*   There were no changes to the language itself in this release.

**Performance Improvements**
*   Garbage collector pause times were further reduced compared to Go 1.5, especially for programs with large heaps.
*   The `sort.Sort` implementation was rewritten to be about 10% faster by reducing the number of comparisons and swaps.
*   Significant performance improvements were made to several `compress` and `crypto` packages.

**Tooling & Developer Experience**
*   **Vendoring is now on by default**, a major step towards making reproducible builds the standard for the ecosystem.
*   The runtime now enforces rules for sharing Go pointers with C code, crashing on violations to prevent subtle memory corruption bugs and improve safety when using `cgo`.
*   The runtime now performs best-effort detection of concurrent map misuse (e.g., one goroutine writing while another reads), crashing the program to help identify data races.
*   By default, panic messages are now more concise, printing only the stack of the panicking goroutine to reduce noise.

**Major Library Updates**
*   The `net/http` package added **transparent support for HTTP/2**. Clients and servers automatically negotiate and use HTTP/2 when making HTTPS requests.
*   The `text/template` and `html/template` packages were enhanced with a `{{block}}` action for template inheritance and syntax for trimming whitespace around actions (`{{-` and `-}}`).