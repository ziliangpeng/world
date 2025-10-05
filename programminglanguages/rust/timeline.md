# Rust Evolution Timeline

- **2013-09-26 — Rust 0.8**: Shifted `for` loops to the `Iterator` trait, introduced default trait methods, and debuted the `format!` family alongside a Rust-written runtime/IO stack (`rust/RELEASES.md:14657`, `rust/RELEASES.md:14659`, `rust/RELEASES.md:14700`, `rust/RELEASES.md:14716`).

- **2014-01-09 — Rust 0.9**: Added crate-level feature gates, overhauled `Option`/`Result` and the `std::io` stack, and broadened macro/trait-object capabilities (`rust/RELEASES.md:14493`, `rust/RELEASES.md:14571`, `rust/RELEASES.md:14525`).

- **2014-04-03 — Rust 0.10**: Established the RFC process, enabled cross-crate macros, and delivered `Vec<T>`, `try!`, `Weak`, plus standardized `Deref`/`DerefMut` semantics (`rust/RELEASES.md:14324`, `rust/RELEASES.md:14338`, `rust/RELEASES.md:14386`, `rust/RELEASES.md:14365`).

- **2014-07-02 — Rust 0.11**: Retired `~`/`@` pointers in favor of `Box`, `Vec`, and `String`, made struct fields private by default, and exposed the `std` façade with `libcore` (`rust/RELEASES.md:14191`, `rust/RELEASES.md:14199`, `rust/RELEASES.md:14232`).

- **2014-10-09 — Rust 0.12**: Introduced lifetime elision, `if let`, `where` clauses, the `Sized`/DST model, and 64-bit Windows support (`rust/RELEASES.md:14069`, `rust/RELEASES.md:14081`, `rust/RELEASES.md:14088`, `rust/RELEASES.md:14083`).

- **2015-05-15 — Rust 1.0.0**: Stabilized the standard library surface, enforced stable-only APIs, and enabled overflow checks in debug builds (`rust/RELEASES.md:13624`, `rust/RELEASES.md:13625`, `rust/RELEASES.md:13629`).

- **2016-11-10 — Rust 1.13.0**: Stabilized the `?` operator and macros in type positions, modernizing idiomatic error handling and metaprogramming (`rust/RELEASES.md:11130`, `rust/RELEASES.md:11132`).

- **2017-02-02 — Rust 1.15.0**: Stabilized custom `#[derive]` ("Macros 1.1") and rewrote the default sort for major performance gains (`rust/RELEASES.md:10671`, `rust/RELEASES.md:10738`).

- **2018-05-10 — Rust 1.26.0**: Stabilized `impl Trait`, inclusive ranges, the `'_` lifetime, 128-bit integers, and major `const` operations (`rust/RELEASES.md:9000`, `rust/RELEASES.md:8997`, `rust/RELEASES.md:8999`, `rust/RELEASES.md:9004`, `rust/RELEASES.md:9006`).

- **2018-06-21 — Rust 1.27.0**: Introduced the `dyn Trait` syntax and stabilized x86/x86_64 SIMD intrinsics (`rust/RELEASES.md:8815`, `rust/RELEASES.md:8833`).

- **2018-10-25 — Rust 1.30.0**: Generalized procedural macros, added raw identifiers, and allowed `crate`-relative module paths (`rust/RELEASES.md:8396`, `rust/RELEASES.md:8400`, `rust/RELEASES.md:8403`).

- **2018-12-06 — Rust 1.31.0**: Released the Rust 2018 edition with new lifetime elision rules, early `const fn`, and tool lints (`rust/RELEASES.md:8312`, `rust/RELEASES.md:8317`, `rust/RELEASES.md:8320`).

- **2019-07-04 — Rust 1.36.0**: Enabled NLL for 2015 edition crates, stabilized the `alloc` crate and `std::future::Future`, and introduced `MaybeUninit` (`rust/RELEASES.md:7484`, `rust/RELEASES.md:7498`, `rust/RELEASES.md:7519`, `rust/RELEASES.md:7517`).

- **2019-11-07 — Rust 1.39.0**: Stabilized `async fn`/`.await`, bringing async Rust to stable (`rust/RELEASES.md:7132`).

- **2019-12-19 — Rust 1.40.0**: Added `#[non_exhaustive]` and let macros emit macros, improving forward compatibility and metaprogramming (`rust/RELEASES.md:7009`, `rust/RELEASES.md:7015`).

- **2020-08-27 — Rust 1.46.0**: Allowed control flow (`if`/`match`/`loop`) inside `const fn`, stabilized `#[track_caller]`, and permitted `mem::transmute` in statics (`rust/RELEASES.md:6128`, `rust/RELEASES.md:6131`, `rust/RELEASES.md:6135`).

- **2021-03-25 — Rust 1.51.0**: Stabilized minimal const generics and aligned Cargo's resolver v2 with the new feature resolver (`rust/RELEASES.md:5400`, `rust/RELEASES.md:5478`).

- **2021-10-21 — Rust 1.56.0**: Released the Rust 2021 edition with richer pattern bindings and `const fn` union field access (`rust/RELEASES.md:4633`, `rust/RELEASES.md:4635`).

- **2022-11-03 — Rust 1.65.0**: Stabilized generic associated types, `let-else`, label `break` values, and raw-dylib linking while tightening rules around uninitialized memory (`rust/RELEASES.md:3309`, `rust/RELEASES.md:3308`, `rust/RELEASES.md:3311`, `rust/RELEASES.md:3314`).

- **2023-06-01 — Rust 1.70.0**: Expanded const/atomic ergonomics with stabilized `std::cell::OnceCell`, `Option::is_some_and`, atomic pointer accessors, and default impls for collection iterators (`rust/RELEASES.md:2733`, `rust/RELEASES.md:2765`, `rust/RELEASES.md:2779`).

- **2023-12-28 — Rust 1.75.0**: Stabilized `async fn` in traits and refined trait coherence for async ecosystems (`rust/RELEASES.md:2109`).

- **2025-02-20 — Rust 1.85.0**: Released the Rust 2024 edition, stabilized async closures, and added async traits to the prelude (`rust/RELEASES.md:683`, `rust/RELEASES.md:685`, `rust/RELEASES.md:714`).
