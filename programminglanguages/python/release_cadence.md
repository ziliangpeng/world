# Python Release Cadence

A comprehensive analysis of Python's release patterns throughout its history.

## Release Timeline by Era

### Early Era (0.9.x): 1991-1993 - Irregular, Rapid Iteration

| Version | Release Date | Time Since Previous |
|---------|--------------|---------------------|
| 0.9.0 | February 1991 | - (first release) |
| 0.9.1 | February 1991 | ~0 months |
| 0.9.2 | Autumn 1991 | ~8-9 months |
| 0.9.4 | December 24, 1991 | ~2-3 months |
| 0.9.6 | April 6, 1992 | ~4 months |
| 0.9.8 | January 9, 1993 | ~9 months |
| 0.9.9 | July 29, 1993 | ~6 months |

**Cadence**: Very irregular, 2-9 months between releases. This was the rapid experimentation and early development phase.

### Python 1.x Era: 1994-2000 - Annual to Biannual

| Version | Release Date | Time Since Previous |
|---------|--------------|---------------------|
| 1.0 | January 26, 1994 | ~6 months |
| 1.1 | October 11, 1994 | ~9 months |
| 1.2 | April 13, 1995 | ~6 months |
| 1.3 | October 13, 1995 | ~6 months |
| 1.4 | October 25, 1996 | ~12 months |
| 1.5 | December 31, 1997 | ~14 months |
| 1.6 | September 5, 2000 | ~33 months |

**Cadence**: Started at ~6 months, settled to ~12 months, but had a very long gap before 1.6 (almost 3 years, likely due to Python 2.0 development).

### Python 2.x Era: 2000-2010 - 18-24 Month Cadence

| Version | Release Date | Time Since Previous |
|---------|--------------|---------------------|
| 2.0 | October 16, 2000 | ~1 month (from 1.6) |
| 2.1 | April 17, 2001 | ~6 months |
| 2.2 | December 21, 2001 | ~8 months |
| 2.3 | July 29, 2003 | ~19 months |
| 2.4 | November 30, 2004 | ~16 months |
| 2.5 | September 19, 2006 | ~22 months |
| 2.6 | October 1, 2008 | ~24 months |
| 2.7 | July 3, 2010 | ~21 months |

**Cadence**: Irregular early on (6-8 months for 2.1 and 2.2), then settled to consistent ~18-24 months.

### Python 3.x Era (2008-2021): 18-Month to Annual

| Version | Release Date | Time Since Previous |
|---------|--------------|---------------------|
| 3.0 | December 3, 2008 | ~2 months (from 2.6) |
| 3.1 | June 27, 2009 | ~7 months |
| 3.2 | February 20, 2011 | ~20 months |
| 3.3 | September 29, 2012 | ~19 months |
| 3.4 | March 16, 2014 | ~18 months |
| 3.5 | September 13, 2015 | ~18 months |
| 3.6 | December 23, 2016 | ~15 months |
| 3.7 | June 27, 2018 | ~18 months |
| 3.8 | October 14, 2019 | ~16 months |

**Cadence**: Mostly 15-20 months, stabilizing around 18 months. This era saw gradual standardization of the release process.

### Python 3.x Modern Era (2020-present): Annual Release (PEP 602)

| Version | Release Date | Time Since Previous |
|---------|--------------|---------------------|
| 3.9 | October 5, 2020 | ~12 months |
| 3.10 | October 4, 2021 | ~12 months |
| 3.11 | October 24, 2022 | ~12 months |
| 3.12 | October 2, 2023 | ~12 months |
| 3.13 | October 7, 2024 | ~12 months |

**Cadence**: **Strict 12-month annual release cycle**, always in October. This was formalized by PEP 602 in 2019.

## Summary of Evolution

Python's release cadence evolved through five distinct phases:

1. **0.9.x (1991-1993)**: Chaotic, 2-9 months - rapid experimentation
2. **1.x (1994-2000)**: 6-14 months normally, with one 33-month gap to 1.6
3. **2.x (2000-2010)**: Settled to 18-24 months - mature, deliberate releases
4. **3.x early (2008-2019)**: Mostly 15-20 months - gradual stabilization
5. **3.x modern (2020+)**: **Strict annual (October each year)** - predictable and consistent

## Key Observations

### Most Predictable Era
The modern annual release cycle (since Python 3.9 in 2020) is the **most predictable and consistent in Python's entire history**. Every major release now occurs in early October.

### Slowest Period
The **longest gap** between releases was the 33 months from Python 1.5 (December 1997) to Python 1.6 (September 2000), during which Python 2.0 was being developed.

### Fastest Period
The **fastest iteration** was during the 0.9.x era, with releases sometimes just 2-3 months apart as Guido rapidly evolved the language.

### Parallel Development
From 2008-2010, Python 2.x and 3.x were released in parallel:
- Python 2.6 (October 2008) and 3.0 (December 2008) were released ~2 months apart
- Both lines continued until Python 2.7 (2010), which was supported until January 2020

## Current Policy (PEP 602)

Since Python 3.9, the release schedule follows PEP 602:
- **Annual releases** in October
- **2 years** of active support (bug fixes and feature backports)
- **3 additional years** of security-only support
- **Total 5-year support lifecycle** for each version

This makes Python's release cadence one of the most predictable among major programming languages.
