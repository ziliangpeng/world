# Go Release Cadence
This summary analyzes Go 1.x release dates from the official release history (downloaded to `/tmp/go/release-notes.html`). Patch releases include only stable tags (no RC/beta).
## Summary
- **Major cadence**: Go publishes a major release every ~6 months (February and August windows since Go 1.5).
- **Minor cadence**: Patch releases for a given Go 1.x stream ship roughly monthly; recent averages sit around 26–31 days.

## Patch Frequency by Go 1.x Line
- **Go 1.0** (3 releases, 2012-04-25 → 2012-09-21): average patch interval 74.5 days
- **Go 1.1** (3 releases, 2013-05-13 → 2013-08-13): average patch interval 46.0 days
- **Go 1.2** (3 releases, 2013-12-01 → 2014-05-05): average patch interval 77.5 days
- **Go 1.3** (4 releases, 2014-06-18 → 2014-09-30): average patch interval 34.7 days
- **Go 1.4** (4 releases, 2014-12-10 → 2015-09-22): average patch interval 95.3 days
- **Go 1.5** (5 releases, 2015-08-19 → 2016-04-12): average patch interval 59.2 days
- **Go 1.6** (5 releases, 2016-02-17 → 2016-12-01): average patch interval 72.0 days
- **Go 1.7** (6 releases, 2016-08-15 → 2017-05-23): average patch interval 56.2 days
- **Go 1.8** (8 releases, 2017-02-16 → 2018-02-07): average patch interval 50.9 days
- **Go 1.9** (8 releases, 2017-08-24 → 2018-06-05): average patch interval 40.7 days
- **Go 1.10** (9 releases, 2018-02-16 → 2019-01-23): average patch interval 42.6 days
- **Go 1.11** (14 releases, 2018-08-24 → 2019-08-13): average patch interval 27.2 days
- **Go 1.12** (18 releases, 2019-02-25 → 2020-02-12): average patch interval 20.7 days
- **Go 1.13** (16 releases, 2019-09-03 → 2020-08-06): average patch interval 22.5 days
- **Go 1.14** (16 releases, 2020-02-25 → 2021-02-04): average patch interval 23.0 days
- **Go 1.15** (16 releases, 2020-08-11 → 2021-08-05): average patch interval 23.9 days
- **Go 1.16** (16 releases, 2021-02-16 → 2022-03-03): average patch interval 25.3 days
- **Go 1.17** (14 releases, 2021-08-16 → 2022-08-01): average patch interval 26.9 days
- **Go 1.18** (11 releases, 2022-03-15 → 2023-01-10): average patch interval 30.1 days
- **Go 1.19** (14 releases, 2022-08-02 → 2023-09-06): average patch interval 30.8 days
- **Go 1.20** (15 releases, 2023-02-01 → 2024-02-06): average patch interval 26.4 days
- **Go 1.21** (14 releases, 2023-08-08 → 2024-08-06): average patch interval 28.0 days
- **Go 1.22** (13 releases, 2024-02-06 → 2025-02-04): average patch interval 30.3 days
- **Go 1.23** (13 releases, 2024-08-13 → 2025-08-06): average patch interval 29.8 days
- **Go 1.24** (8 releases, 2025-02-11 → 2025-09-03): average patch interval 29.1 days
- **Go 1.25** (2 releases, 2025-08-12 → 2025-09-03): average patch interval 22.0 days

## Notes
- Each major release remains supported until two newer majors ship; patches continue during that window.
- The averages above exclude release candidates and betas, focusing only on shipped stable tags (e.g., `go1.21.4`).
- Source data: https://go.dev/doc/devel/release, cached locally at `/tmp/go/release-notes.html`.
