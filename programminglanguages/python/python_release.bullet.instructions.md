# Python Release Notes → Bullet Point Summary Guide

This guide explains how to convert raw Python release notes into concise, actionable bullet point summaries.

## File Naming Convention

- Raw release notes: `python_X.Y_release.raw.md`
- Bullet summary: `python_X.Y_release.bullet.md`
- Prose summary: `python_X.Y_release.summary.md`
- Example: `python_3.12_release.bullet.md`

## File Structure

Each bullet summary file should have this structure:

1. **Header** - Title, release date, EOL date
2. **Major Highlights** - 5-7 most impactful changes in plain language
3. **Experimental Features** - If applicable (e.g., Python 3.13's free-threading and JIT)
4. **Breaking Changes** - Changes that break existing code
5. **Deprecations** - Grouped by removal timeline
6. **New Features** - New capabilities added
7. **Improvements** - Performance and quality improvements
8. **Implementation Details** - Low-level changes for advanced users
9. **Platform & Environment** - If applicable
10. **Release Process & Meta Changes** - If applicable

### Major Highlights Section

**Purpose:** Make it easy to quickly grasp the most important changes in a release without reading hundreds of bullets.

**Guidelines:**
- Place immediately after the header (release date/EOL)
- List 5-7 of the most impactful changes
- Use plain language, not bullet format
- Focus on changes that affect the widest audience or represent significant architectural shifts
- Include a brief one-line summary that captures the release's theme
- Prioritize:
  - Experimental/revolutionary features (free-threading, JIT)
  - Major breaking changes (dead batteries removal, distutils removal)
  - Significant language syntax changes (PEP 695, PEP 701)
  - Major performance improvements (2x+ speedups)
  - Important developer experience improvements (REPL, error messages)
  - Platform support changes

**Example:**
```markdown
## Major Highlights

Python 3.13 is a groundbreaking release with two experimental game-changers and major quality-of-life improvements:

1. **Free-threaded mode (PEP 703)** - Run Python without the GIL for true parallelism (experimental)
2. **JIT compiler (PEP 744)** - Experimental just-in-time compilation for performance improvements
3. **Improved REPL** - Modern interactive interpreter with colors, multiline editing, and history browsing
4. **Defined locals() semantics (PEP 667)** - Clear mutation behavior for debugging and introspection
5. **Removed 19 "dead batteries" modules (PEP 594)** - Cleanup of legacy standard library modules
```

## Primary Categorization: Action Required

Organize all changes into these core categories based on **what action developers need to take**. These categories cover most releases, but **feel free to add new categories when the release introduces changes that don't fit well into existing ones** - prioritize clarity and usefulness over rigid adherence to this list.

### 1. Breaking Changes
**When to use:** Changes that break existing code immediately in this version.
- Removed modules/APIs/features
- Incompatible behavior changes
- Changed defaults that break existing code

**User impact:** Must act now if affected - code will not run without changes

### 2. Deprecations
**When to use:** Features that work now but will be removed in future versions.
- Organize by removal timeline (Python 3.X)
- Include when removal will happen
- Provide migration path when available

**User impact:** Plan for future - working now, need to migrate before removal date

### 3. New Features
**When to use:** New capabilities added to Python.
- New syntax/grammar
- New modules
- New APIs/functions/methods
- New capabilities in existing features

**User impact:** Optional - learn and adopt when beneficial

### 4. Improvements
**When to use:** Enhancements that automatically benefit users without code changes.
- Performance optimizations
- Better error messages
- Bug fixes (if notable)
- Better defaults

**User impact:** Automatic - just works better

### 5. Implementation Details
**When to use:** Low-level changes for advanced users.
- CPython bytecode changes
- C API changes
- Build system changes
- Internal optimizations

**User impact:** Niche - primarily for CPython contributors, C extension authors

### 6. Platform & Environment
**When to use:** Changes to platform support, build requirements, or runtime environment.
- New platform support (iOS, Android, WASM)
- Platform tier changes (tier 1/2/3)
- Removed/deprecated platform support
- Environment variables
- Runtime configuration options

**User impact:** Affects deployment, CI/CD, and platform-specific development

**Example:** `- 🟡 **Platform** iOS and Android now officially supported at tier 3 (PEP 730, 738)`

### 7. Release Process & Meta Changes
**When to use:** Changes about Python releases themselves, not the code.
- Support timeline changes (e.g., PEP 602 extending full support period)
- Release cycle changes
- Versioning policy changes

**User impact:** Affects upgrade planning and support windows

**Example:** `- 🟡 **Release** Python 3.13+ gets 2 years full support (up from 1.5 years) + 3 years security fixes (PEP 602)`

## Bullet Point Format

Each bullet point should be on its own line using markdown list syntax:

```
- [Impact] **[Domain]** [Description] [(PEP/Reference)] [- Additional context]
```

**Important:** Always start each bullet with `- ` (dash and space) for proper markdown rendering.

### Impact Level Emoji

- 🔴 **High Impact** - Widely used, affects many developers
- 🟡 **Medium Impact** - Common but not universal
- 🟢 **Low Impact** - Niche, edge cases, specialized use

**Guidelines:**
- Be conservative with 🔴 - reserve for truly widespread changes
- Most changes should be 🟡 or 🟢
- Consider both breadth (how many affected) and depth (how much work to fix)

### Domain Tags

Use concise domain tags to categorize by area. Common tags:

**Language/Core:**
- `Syntax` - Language syntax changes
- `Grammar` - Parser/grammar changes
- `Type System` - Type hints, typing module
- `Data Model` - Dunder methods, core protocols

**Standard Library (by module):**
- `asyncio`
- `pathlib`
- `typing`
- `os`
- `re`
- etc. (use actual module names)

**Cross-Cutting:**
- `Performance` - Speed improvements
- `Error Messages` - Better diagnostics
- `Security` - Security enhancements
- `Async` - Async/await related
- `Debugging` - Debugging tools

**Platform/Environment:**
- `Platform` - Platform support changes
- `Build` - Build system, compilation
- `Environment` - Environment variables, config

**Advanced:**
- `Bytecode` - CPython bytecode
- `C API` - C extension API
- `Interpreter` - Interpreter internals
- `REPL` - Interactive interpreter
- `Dev Mode` - Python development mode
- `Stable ABI` - C API stability
- `Release` - Release process/policy

### Description

- Start with action verb when possible (Add, Remove, Improve, Change, Fix, etc.)
- Be concise but complete - capture the essence
- Include concrete examples only if they fit in one line
- Link to PEPs when relevant: `(PEP 695)`
- Provide migration guidance for breaking changes/deprecations

## Writing Guidelines

### DO:
- ✅ Focus on **developer impact** - what they need to know
- ✅ Be specific about what changed
- ✅ Include version numbers for deprecation timelines
- ✅ Group related changes together
- ✅ Use consistent terminology
- ✅ Mention performance gains with numbers when available ("75% faster", "2-20x speedup")

### DON'T:
- ❌ Include implementation details in high-level categories
- ❌ Duplicate information across categories
- ❌ List every single module improvement (consolidate minor ones)
- ❌ Include contributor names (they're in raw notes)
- ❌ Use technical jargon without explanation
- ❌ Be verbose - this is a quick reference

## Category-Specific Guidelines

### Breaking Changes
- Always provide migration path or alternative
- Be explicit about what breaks
- Example: `- 🔴 **distutils** Removed distutils package (PEP 632) - Use setuptools or modern build tools`

### Deprecations
- Always include removal timeline
- Group by removal version (3.14, 3.15, etc.)
- Example: `- 🟡 **asyncio** asyncio.get_event_loop() behavior change (removal in 3.14)`

### New Features
- Briefly explain the benefit/use case
- Include syntax examples only for major features
- Example: `- 🔴 **Type System** New type parameter syntax for generics (PEP 695) - Simpler generic class/function declarations`

### Improvements
- Quantify performance gains when available
- Example: `- 🟡 **asyncio** 75% performance improvement in socket writes`

### Implementation Details
- Can be more technical - audience is advanced
- Example: `- 🟢 **Bytecode** Add LOAD_SUPER_ATTR instruction for faster super() calls`

### Experimental/Preview Features
Features that are explicitly marked as experimental, disabled by default, or in preview:
- Always indicate how to enable (build flag, environment variable, command-line option)
- Note stability level and caveats
- Use sub-bullets for activation details

**Format:**
```
- 🟡 **Interpreter** Experimental free-threaded mode (PEP 703) - Run without GIL for true parallelism
  - Enable: `python --disable-gil` or set `PYTHON_GIL=0`
  - Status: Experimental, may have compatibility issues with C extensions
```

### Large-Scale Removals
When many modules/APIs removed together (e.g., PEP 594 "dead batteries"):
- List inline to show scope
- Example: `- 🔴 **stdlib** Removed 19 legacy modules (PEP 594) - aifc, audioop, cgi, cgitb, chunk, crypt, imghdr, mailcap, msilib, nis, nntplib, ossaudiodev, pipes, sndhdr, spwd, sunau, telnetlib, uu, xdrlib`

### Porting Sections
Official release notes include "Porting to Python X.Y" sections:
- Integrate important porting notes into **Breaking Changes**
- Don't create separate porting category
- Focus on behavioral changes that might silently break code

## Example Structure

```markdown
# Python 3.12 Release Notes

**Released:** October 2, 2023
**EOL:** October 2028 (security support)

## Major Highlights

Python 3.12 focuses on usability improvements for type hints, f-strings, and developer experience:

1. **New type parameter syntax (PEP 695)** - Cleaner generic classes and functions
2. **F-string restrictions removed (PEP 701)** - Can reuse quotes, use multiline expressions
3. **Comprehensions 2x faster (PEP 709)** - List/dict/set comprehensions inlined
4. **isinstance() 2-20x faster** - Protocol checks dramatically accelerated
5. **Better error messages** - "Did you forget to import 'sys'?"
6. **Per-interpreter GIL (PEP 684)** - Foundation for better parallelism
7. **distutils removed (PEP 632)** - Use setuptools or modern packaging tools

## Breaking Changes

- 🔴 **distutils** Removed distutils package (PEP 632) - Use setuptools or modern packaging tools
- 🔴 **venv** setuptools no longer pre-installed in virtual environments - Run `pip install setuptools` if needed
- 🟡 **asynchat, asyncore, imp** Removed deprecated modules
- 🟢 **Syntax** Null bytes in source code now raise SyntaxError

## Deprecations

### Removing in Python 3.14

- 🔴 **asyncio** asyncio.get_event_loop() will warn/error if no event loop exists
- 🟡 **ast** ast.Num, ast.Str, ast.Bytes deprecated - Use ast.Constant instead
...

### Removing in Python 3.15

- 🟡 **importlib** Various importlib.abc classes deprecated - Use importlib.resources.abc
...

## New Features

- 🔴 **Type System** New type parameter syntax for generics (PEP 695)
  - Simpler syntax: `def max[T](args: Iterable[T]) -> T`
  - New `type` statement for type aliases: `type Point = tuple[float, float]`

- 🔴 **Syntax** F-string restrictions removed (PEP 701)
  - Can reuse quotes: `f"Hello {", ".join(names)}"`
  - Multiline expressions and comments now allowed
  - Backslashes in expressions now supported

...
```

## Tips for Summarization

1. **Read the full raw release notes first** - Understand scope and themes
2. **Start with the official summary section** - It highlights key changes
3. **Look for PEPs** - Major features are usually associated with PEPs
4. **Consolidate similar changes** - Don't list every module improvement separately
5. **Focus on user-facing changes** - Skip most internal refactorings
6. **Check deprecation sections carefully** - Critical for planning
7. **Include numbers** - Performance improvements are more concrete with metrics

## Completeness vs. Conciseness

**Include:**
- All breaking changes
- All deprecations with timelines
- Major new features (PEPs, significant APIs)
- Experimental features (with activation instructions)
- Significant performance improvements (>20% speedup, or notable modules)
- Security improvements
- Important error message improvements
- Platform support changes
- Release policy/timeline changes

**Can omit or consolidate:**
- Minor bug fixes
- Small API additions to many modules (consolidate as "various improvements")
- Internal refactorings without user impact
- Build system tweaks (unless they affect users)
- Documentation-only changes
- Patch release notes (e.g., "Notable changes in 3.11.4") - focus on main X.Y release

**Target length:** 50-150 bullet points depending on release size. Python 3.12 should be ~80-100 bullets.

## Quality Checklist

Before finalizing, verify:
- [ ] Major Highlights section present with 5-7 key changes and theme summary
- [ ] Every breaking change has migration guidance
- [ ] Every deprecation has removal timeline
- [ ] Experimental features clearly marked with activation method
- [ ] Platform support changes documented (if any)
- [ ] Release timeline/meta changes noted (if any)
- [ ] Impact levels are consistently applied
- [ ] Domain tags are consistent and clear
- [ ] No duplicate information across categories
- [ ] Major PEPs are all covered (check official summary for PEP list)
- [ ] Performance claims include numbers when available
- [ ] Formatting is consistent throughout (all bullets start with `- `)
- [ ] Release date and EOL date in header
- [ ] Porting guide sections reviewed and integrated into Breaking Changes
