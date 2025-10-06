### Summary of Go 1.20

Go 1.20 introduced preview support for Profile-Guided Optimization (PGO), extended error wrapping to handle multiple errors, and added support for collecting coverage profiles for entire applications.

**Language Features & Syntax**
*   Slices can now be converted directly to arrays (e.g., `[4]byte(x)`).
*   The `unsafe` package was extended with `SliceData`, `String`, and `StringData`, providing a complete set of primitives for slice and string manipulation.

**Performance Improvements**
*   Preview support for **Profile-Guided Optimization (PGO)** was added. By using a CPU profile (`-pgo` flag), the compiler can make better optimization decisions, improving application performance by 3-4%.
*   Build speeds were improved by up to 10%, returning to Go 1.17 levels.
*   The garbage collector's internal data structures were optimized, reducing memory overhead and improving CPU performance by up to 2%.

**Tooling & Developer Experience**
*   The toolchain now supports collecting **code coverage profiles for entire applications**, not just unit tests, via the `go build -cover` flag.
*   The Go distribution no longer ships with pre-compiled standard library packages, reducing its size. Packages are built from source and cached on demand.
*   `cgo` is now disabled by default on systems without a C compiler, improving the out-of-the-box experience in minimal environments.

**Major Library Updates**
*   Error wrapping was extended to support **wrapping multiple errors**. This includes a new `errors.Join` function and support for multiple `%w` verbs in `fmt.Errorf`.
*   A new `crypto/ecdh` package was added for Elliptic Curve Diffie-Hellman key exchanges.
*   The `net/http` package added `http.ResponseController`, a new way to access extended per-request controls, such as setting deadlines.
*   `httputil.ReverseProxy` gained a new, safer `Rewrite` hook for modifying requests.