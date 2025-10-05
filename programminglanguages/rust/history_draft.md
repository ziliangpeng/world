# Rust Release Milestones (draft)

> Curated from `rust/RELEASES.md`; focuses on editions, language pillars, and ecosystem turning points.

## Launch and Stability (2015–2016)
- **1.0.0 (2015-05-15)** — First stable toolchain; most of `std` marked `#[stable]`, crate naming rules tightened, and integer overflow traps in debug builds (see `rust/RELEASES.md:13624`, `rust/RELEASES.md:13644`, `rust/RELEASES.md:13629`).
- **1.13.0 (2016-11-10)** — Stabilized the `?` operator and macros in type position, making idiomatic error propagation and macro-heavy APIs feel natural (`rust/RELEASES.md:11130`, `rust/RELEASES.md:11132`).

## Macro Power and Ergonomics (2017–early 2018)
- **1.15.0 (2017-02-02)** — “Macros 1.1” made custom `#[derive]` stable, unlocking Serde/Diesel ergonomics, plus a much faster default sort (`rust/RELEASES.md:10671`, `rust/RELEASES.md:10738`).
- **1.26.0 (2018-05-10)** — Stabilized `impl Trait`, inclusive ranges, the `'_` lifetime, 128-bit integers, and a wave of new `const` operations (`rust/RELEASES.md:9000`, `rust/RELEASES.md:8997`, `rust/RELEASES.md:8999`, `rust/RELEASES.md:9004`, `rust/RELEASES.md:9006`).
- **1.27.0 (2018-06-21)** — Introduced `dyn Trait` syntax and stabilized the first SIMD intrinsics on stable Rust (`rust/RELEASES.md:8815`, `rust/RELEASES.md:8833`).

## Editions and Module Hygiene (late 2018)
- **1.30.0 (2018-10-25)** — General procedural macros, raw identifiers, `crate`-relative paths, and `use`-based macro imports set the stage for the 2018 edition (`rust/RELEASES.md:8396`, `rust/RELEASES.md:8400`, `rust/RELEASES.md:8403`, `rust/RELEASES.md:8412`).
- **1.31.0 (2018-12-06)** — Rust 2018 edition release alongside early `const fn` support and scoped tool lints (`rust/RELEASES.md:8312`, `rust/RELEASES.md:8317`, `rust/RELEASES.md:8320`).

## Async Foundations and Const Evolution (2019–2020)
- **1.36.0 (2019-07-04)** — Brought non-lexical lifetimes to 2015 edition crates, stabilized the `alloc` crate, `std::future::Future`, and `MaybeUninit` (`rust/RELEASES.md:7484`, `rust/RELEASES.md:7498`, `rust/RELEASES.md:7519`, `rust/RELEASES.md:7517`).
- **1.39.0 (2019-11-07)** — Stabilized `async fn`/`.await`, bringing async Rust to stable toolchains (`rust/RELEASES.md:7132`).
- **1.40.0 (2019-12-19)** — Added `#[non_exhaustive]`, let macros emit macros, and allowed tuple struct/variant constructors in const contexts (`rust/RELEASES.md:7009`, `rust/RELEASES.md:7015`, `rust/RELEASES.md:6996`).
- **1.46.0 (2020-08-27)** — Enabled `if`/`match`/`loop` inside `const fn`, stabilized `#[track_caller]`, and allowed `mem::transmute` in statics (`rust/RELEASES.md:6128`, `rust/RELEASES.md:6131`, `rust/RELEASES.md:6135`).

## Advanced Generics and Dependency Workflow (2021–2022)
- **1.51.0 (2021-03-25)** — Minimal const generics stabilized and Cargo’s resolver v2 landed for saner feature unification (`rust/RELEASES.md:5400`, `rust/RELEASES.md:5478`).
- **1.56.0 (2021-10-21)** — Rust 2021 edition release plus richer pattern bindings and `const fn` union field access (`rust/RELEASES.md:4633`, `rust/RELEASES.md:4635`).
- **1.65.0 (2022-11-03)** — Stabilized generic associated types, `let-else`, label `break` values, and Windows raw-dylib; tightened UB around uninitialized memory (`rust/RELEASES.md:3309`, `rust/RELEASES.md:3308`, `rust/RELEASES.md:3311`, `rust/RELEASES.md:3314`).

## Async Traits and the 2024 Edition (2023–2025)
- **1.75.0 (2023-12-28)** — Stabilized `async fn` in traits and improved trait coherence rules, unblocking ergonomic async trait APIs (`rust/RELEASES.md:2109`).
- **1.85.0 (2025-02-20)** — Rust 2024 edition release and stabilized async closures, completing the async ergonomics story for the latest edition (`rust/RELEASES.md:683`, `rust/RELEASES.md:685`).

---

_Next steps: convert this draft into narrative sections (e.g., per edition or theme), add visuals or links to edition guides, and trim inline citations once the surrounding copy references them naturally._
