### Summary of Go 1

Go 1 was a foundational release focused on creating a stable language and library for long-term use. It introduced many breaking changes, a new `go` command to replace makefiles, and a major reorganization of the standard library.

**Language Features & Syntax**
*   Introduced the `rune` type for Unicode characters and the built-in `error` interface.
*   Added the `delete` built-in function for removing items from maps, replacing the old, awkward syntax.
*   Structs and arrays now support equality comparison, allowing them to be used as map keys.
*   The `append` function can now directly append a string to a `[]byte` slice.

**Performance Improvements**
*   While the primary focus was stability, changes like randomizing map iteration order were made to improve the robustness and internal balancing of maps.

**Tooling & Developer Experience**
*   The `go` command was introduced, a new tool for fetching, building, and installing packages that replaced the need for makefiles.
*   The `go fix` tool was provided to help automate the migration of old code to the Go 1 standard.
*   The compiler now detects and disallows shadowed return variables in some cases, preventing common bugs.
*   Map iteration order is now explicitly unpredictable, helping to identify and fix fragile code that depended on a specific order.

**Major Library Updates**
*   **Complete Redesigns:** The `time` package was completely redesigned with `time.Time` and `time.Duration` types. The `regexp` package was rewritten to use RE2 syntax.
*   **Library Reorganization:** The standard library was heavily restructured, with many packages moving into new locations (e.g., `encoding/json`, `net/http`). Many other packages were moved to external sub-repositories or deleted.
*   **Error Handling:** The `syscall` package was changed to return `error` values instead of integers. A new `errors` package was added.
*   **`os.FileInfo`:** Changed from a struct to an interface to provide a more portable way to handle file metadata.