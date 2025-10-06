### Summary of Go 1.22

Go 1.22 fixed the `for` loop variable capture issue, enhanced the standard HTTP router with methods and wildcards, and introduced a new `math/rand/v2` package.

**Language Features & Syntax**
*   The `for` loop semantics were changed so that loop variables are **re-created for each iteration**. This fixes a common "gotcha" where closures would accidentally capture the final value of a loop variable.
*   `for` loops can now **range over integers** (e.g., `for i := range 10`).

**Performance Improvements**
*   Profile-Guided Optimization (PGO) was improved to devirtualize more interface method calls, leading to runtime performance improvements of 2-14% in many programs.
*   The runtime improved its use of garbage collection metadata, resulting in a 1-3% CPU performance improvement and ~1% less memory overhead.

**Tooling & Developer Experience**
*   Workspaces now support a top-level `vendor` directory.
*   The `vet` tool was updated to align with the new `for` loop semantics and adds new checks for deferred `time.Since` calls and invalid `log/slog` arguments.
*   The web UI for the execution trace viewer was refreshed.

**Major Library Updates**
*   The standard HTTP router, `net/http.ServeMux`, was significantly enhanced to support **HTTP methods and path wildcards** (e.g., `POST /items/{id}`).
*   A new **`math/rand/v2`** package was introduced with a cleaner API, better algorithms, and automatic random seeding for top-level functions.
*   A new `go/version` package was added for working with Go version strings.