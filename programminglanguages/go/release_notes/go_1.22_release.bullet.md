# Go 1.22 Release Notes

**Released:** February 6, 2024
**EOL:** August 2025 (expected)

## Major Highlights

Go 1.22 significantly improves the developer experience by fixing a long-standing `for` loop variable capture bug, enhancing the standard HTTP router with modern routing patterns, and introducing a new, improved random number generator.

1.  **`for` loop variable fix:** Variables in `for` loops are now created anew for each iteration, preventing common bugs with closures.
2.  **Enhanced HTTP Routing:** `net/http.ServeMux` now supports HTTP methods and path wildcards, bringing modern routing capabilities to the standard library.
3.  **New `math/rand/v2` package:** A cleaner, faster, and more robust random number generator.
4.  **PGO Improvements:** Profile-Guided Optimization (PGO) builds can now devirtualize more calls, leading to significant runtime performance improvements.
5.  **Runtime Memory Management:** The runtime now keeps GC metadata closer to heap objects, improving CPU performance and reducing memory overhead.
6.  **`vet` for `log/slog`:** New `vet` warnings for invalid arguments in `log/slog` calls.

## Breaking Changes

- 🔴 **Language** `for` loop variables are now created anew for each iteration, fixing a common bug but potentially changing behavior for code relying on the old semantics.
- 🔴 **Net/HTTP** `net/http.ServeMux` routing changes break backward compatibility in small ways (e.g., patterns with `{}` behave differently, improved escaped path handling). A `GODEBUG` setting `httpmuxgo121=1` can restore old behavior.
- 🟡 **Runtime** Some objects' addresses previously aligned to 16-byte boundaries may now only be 8-byte aligned, potentially breaking assembly code relying on stricter alignment.

## Deprecations

- 🟡 **Go Command** `go get` is no longer supported outside of a module in legacy `GOPATH` mode (`GO111MODULE=off`).

## New Features

- 🔴 **Language** `for` loops can now range over integers (e.g., `for i := range 10`).
- 🟡 **Language** Preview of range-over-function iterators is included (enabled with `GOEXPERIMENT=rangefunc`).
- 🔴 **Standard Library** New `math/rand/v2` package provides a cleaner, faster, and more robust random number generator.
- 🔴 **Standard Library** New `go/version` package implements functions for validating and comparing Go version strings.
- 🔴 **Net/HTTP** `net/http.ServeMux` now supports HTTP methods (e.g., `POST /items`) and wildcards in patterns (e.g., `/items/{id}`).

## Improvements

- 🟢 **Runtime** GC metadata is kept nearer to heap objects, improving CPU performance (1-3%) and reducing memory overhead (~1%).
- 🟢 **Runtime** The runtime now releases memory back to the operating system more aggressively.
- 🟢 **Runtime** Timer and deadline code is faster and scales better with higher CPU counts.
- 🟢 **Compiler** PGO builds can now devirtualize a higher proportion of calls, leading to 2-14% runtime improvement.
- 🟢 **Compiler** Compiler interleaves devirtualization and inlining for better interface method optimization.
- 🟢 **Trace Tool** The `trace` tool's web UI has been refreshed, improving readability and supporting thread-oriented views.
- 🟢 **Vet** `vet` no longer reports loop variable capture errors for Go 1.22+ code, aligning with new language semantics.

## Tooling & Developer Experience

- 🟡 **Go Command** Workspaces can now use a `vendor` directory, created by `go work vendor`.
- 🟡 **Vet** New warnings for `append()` calls with no values, non-deferred `time.Since()` in `defer` statements, and mismatched key-value pairs in `log/slog` calls.
- 🟡 **Go Command** `go test -cover` now prints coverage summaries for covered packages without test files.

## Platform & Environment

- 🟡 **Platform** `linux/arm64` now supports the race detector.
- 🟡 **Platform** `windows/amd64` programs linking Go libraries can now use `SetUnhandledExceptionFilter` for exceptions.
- 🟡 **Platform** `darwin/amd64` now generates position-independent executables (PIE) by default.
- 🟢 **Platform** Experimental port to OpenBSD on big-endian 64-bit PowerPC (`openbsd/ppc64`).
