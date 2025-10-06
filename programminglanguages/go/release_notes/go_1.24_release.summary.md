### Summary of Go 1.24

Go 1.24 introduced a new, faster `map` implementation, added support for tracking tool dependencies in `go.mod`, and provided a new, safer way to scope filesystem operations with `os.Root`.

**Language Features & Syntax**
*   Generic type aliases are now fully supported, allowing type aliases to have their own type parameters.

**Performance Improvements**
*   The internal implementation of Go's `map` was changed to use an approach based on Swiss Tables, resulting in a 2-3% average CPU performance improvement.
*   Memory allocation for small objects and internal runtime mutexes were also made more efficient.

**Tooling & Developer Experience**
*   A new **`tool` directive** in `go.mod` allows for tracking executable dependencies (like linters) directly in the module file, replacing the `tools.go` convention.
*   `go build` now stamps version control information into binaries by default.
*   A new `GOAUTH` environment variable provides a flexible way to authenticate to private module repositories.

**Major Library Updates**
*   A new **`os.Root`** type provides a mechanism for performing filesystem operations that are restricted to a specific directory tree, preventing path traversal.
*   The `testing` package added **`b.Loop()`**, a new, faster, and safer way to write benchmark loops.
*   A new **`runtime.AddCleanup`** function was introduced as a more flexible and efficient alternative to `runtime.SetFinalizer`.
*   A new low-level **`weak` package** was added to provide weak pointer primitives.
*   The `crypto/hkdf`, `crypto/pbkdf2`, and `crypto/sha3` packages were added to the standard library.