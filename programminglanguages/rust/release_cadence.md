# Rust Release Cadence
This report uses the official `rust/RELEASES.md` chronology plus the Rust Book's release-channel documentation to summarize how frequently Rust ships.

## Key Takeaways
- **Six-week train**: Since early 2016, stable Rust releases depart every six weeks (mean 42.0 days, median 42.0 days). The Rust Book describes this train model explicitly, noting that knowing one release date lets you predict the next one exactly six weeks later ([Rust Book – Release Channels](https://doc.rust-lang.org/book/appendix-07-nightly-rust.html)).
- **Predictable `.0` flow**: Each year typically delivers eight to nine `.0` releases, providing a steady stream of compiler and tooling updates. The table below shows the first release of each year alongside the average gap between releases landing that year.
- **Targeted patch drops**: Point releases (`1.x.y`, y > 0) are comparatively rare and appear when critical fixes warrant an out-of-band update (e.g., regressions or security advisories).
- **Editions ride the same train**: Major language milestones arrive as "editions" (2015, 2018, 2021, 2024) while the version number remains 1.x. Editions are opt-in via `Cargo.toml` and align with one of the regularly scheduled `.0` releases.

## Major Release Rhythm
Year | First release | Major releases | Avg gap (days) | Median gap (days)
--- | --- | --- | --- | ---
2016 | 1.6.0 (2016-01-21) | 9 | 42.0 | 42.0
2017 | 1.15.0 (2017-02-02) | 8 | 41.9 | 42.0
2018 | 1.23.0 (2018-01-04) | 9 | 42.1 | 42.0
2019 | 1.32.0 (2019-01-17) | 9 | 42.0 | 42.0
2020 | 1.41.0 (2020-01-30) | 9 | 42.0 | 42.0
2021 | 1.50.0 (2021-02-11) | 8 | 42.0 | 42.0
2022 | 1.58.0 (2022-01-13) | 9 | 42.0 | 42.0
2023 | 1.67.0 (2023-01-26) | 9 | 42.0 | 42.0
2024 | 1.76.0 (2024-02-08) | 8 | 42.0 | 42.0
2025 | 1.84.0 (2025-01-09) | 7 | 42.0 | 42.0

## Patch Release Activity
Year | Patch releases
--- | ---
2016 | 1
2017 | 2
2018 | 9
2019 | 2
2020 | 5
2021 | 2
2022 | 2
2023 | 7
2024 | 3
2025 | 2

## Edition Timeline
Edition | Introduced in | Release date
--- | --- | ---
2015 | 1.0.0 | 2015-05-15
2018 | 1.31.0 | 2018-12-06
2021 | 1.56.0 | 2021-10-21
2024 | 1.85.0 | 2025-02-20

## Notes
- Release gaps measure the interval between consecutive `.0` entries in `rust/RELEASES.md`.
- Edition dates come from the corresponding entries in `rust/RELEASES.md` (e.g., Rust 2018 landed with 1.31.0 on 2018-12-06).
- Official documentation of the six-week cycle: Rust Programming Language Book, Appendix E ("Release Channels").
