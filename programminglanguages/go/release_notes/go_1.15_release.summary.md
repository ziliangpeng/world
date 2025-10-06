### Summary of Go 1.15

Go 1.15 delivered substantial improvements to the Go linker, making it significantly faster and use less memory. It also deprecated the legacy X.509 CommonName behavior and added a way to embed the timezone database.

**Language Features & Syntax**
*   There were no changes to the language.

**Performance Improvements**
*   The Go linker is **20% faster and uses 30% less memory** on average for `amd64` Linux systems, due to a redesigned object file format and improved concurrency.
*   Allocation of small objects is much faster at high core counts and has lower worst-case latency.
*   Converting small integers to interface values no longer causes a heap allocation.

**Tooling & Developer Experience**
*   **X.509 CommonName Deprecated:** The legacy behavior of treating a certificate's `CommonName` as a hostname (when no Subject Alternative Names are present) is now **disabled by default**, improving security.
*   `GOPROXY` now supports falling back to the next proxy on any error, not just 404/410 errors, by using `|` as a separator.
*   The `vet` tool adds two important new checks by default: one for likely-incorrect `string(int)` conversions and one for impossible interface-to-interface type assertions.

**Major Library Updates**
*   A new **`time/tzdata`** package allows the timezone database to be embedded into a program, making it self-contained and portable.
*   The `crypto` packages (`rsa`, `ecdsa`, `ed25519`) added an `Equal` method to their key types for type-safe key comparison.
*   The `database/sql` package added `DB.SetConnMaxIdleTime` to give more control over closing idle connections in the connection pool.