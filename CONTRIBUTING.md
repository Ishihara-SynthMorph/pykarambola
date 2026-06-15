# Contributing to pykarambola

This guide is written for lab members who are new to collaborative software development.
It covers everything from setting up your environment to releasing a new version.

---

## Table of contents

1. [Development environment](#1-development-environment)
2. [Project structure](#2-project-structure)
3. [Git workflow](#3-git-workflow)
4. [Issues, milestones, and the project board](#4-issues-milestones-and-the-project-board)
5. [Version numbers](#5-version-numbers)
6. [Changelog](#6-changelog)
7. [Running the tests](#7-running-the-tests)
8. [Releasing a new version](#8-releasing-a-new-version)

---

## 1. Development environment

**Clone and install in editable mode** (do this once):

```bash
git clone https://github.com/Ishihara-SynthMorph/pykarambola.git
cd pykarambola
pip install -e ".[dev]"
```

The `-e` flag ("editable") means Python loads the package directly from your working directory,
so changes you make to `.py` files are immediately reflected without reinstalling.

`[dev]` installs the optional dependencies needed for tests: `pytest` and `scikit-image`.

To also enable GLB file support and Cython acceleration:

```bash
pip install -e ".[dev,glb,accel]"
```

**Verify the installation:**

```bash
python -c "import pykarambola; print(pykarambola.__version__)"
pytest tests/
```

---

## 2. Project structure

```
pykarambola/
├── api.py            ← high-level public API (minkowski_tensors, minkowski_tensors_from_label_image)
├── minkowski.py      ← all calculate_w* compute functions
├── triangulation.py  ← Triangulation data structure and precomputed geometry
├── eigensystem.py    ← eigenvalue/vector decomposition for rank-2 tensors
├── tensor.py         ← SymmetricMatrix3, Rank3Tensor, SymmetricRank4Tensor
├── results.py        ← MinkValResult, CalcOptions, SurfaceStatistics
├── surface.py        ← surface validation and statistics (check_surface)
├── output.py         ← output writers matching the C++ karambola format
├── spherical.py      ← spherical Minkowski structure metrics
├── cli.py            ← command-line interface (separate code path from api.py)
├── io_poly.py        ← .poly parser
├── io_off.py         ← .off parser
├── io_obj.py         ← .obj parser
├── io_glb.py         ← .glb parser (requires trimesh)
└── io_stl.py         ← .stl parser, ASCII and binary (requires numpy-stl)

tests/
├── conftest.py           ← shared fixtures
├── test_api.py           ← tests for the high-level API
├── test_box.py           ← numerical accuracy tests against known box geometry
├── test_readers.py       ← file format parser tests
├── test_accel.py         ← Cython acceleration vs. Python fallback
├── test_mesh_quality.py  ← behavior on meshes with known quality issues
├── test_nonconvex.py     ← non-convex mesh handling and degenerate triangles
├── test_wigner3j.py      ← Wigner 3j coefficient implementation
├── fixtures/             ← test mesh files (.poly, .off, .obj)
└── notebooks/            ← Jupyter notebooks used in development / validation

.github/workflows/
├── ci.yml             ← runs pytest on every push and PR (ubuntu, macOS, Windows × Python 3.9/3.11/3.13)
├── add-to-project.yml ← automatically adds new issues to the project board
└── slack-notify.yml   ← Slack notifications for merges and CI failures
```

**Principle:** `api.py` is the only file that users should import from.
`cli.py` has its own separate code path and is not affected by changes to `api.py`.

---

## 3. Git workflow

We use a simple **feature-branch** model.

### The basic cycle

```
main  ──●──────────────────────────────●──  (stable, CI always passes)
         \                            /
          ●── work ── work ── work ──●       (your branch)
```

1. **Create a branch** for each issue you work on:

   ```bash
   git checkout -b fix/42-get-ref-vec-div-by-zero
   ```

   Branch name convention: `fix/<issue-number>-short-description` for bugs,
   `feat/<issue-number>-short-description` for enhancements.

2. **Make small, focused commits.** Each commit should do one logical thing:

   ```bash
   git add pykarambola/minkowski.py tests/test_api.py
   git commit -m "Guard against zero denominator in get_ref_vec (#42)"
   ```

   Including the issue number (`#42`) in the commit message creates a link to the issue
   on GitHub automatically.

3. **Push your branch and open a pull request (PR):**

   ```bash
   git push -u origin fix/42-get-ref-vec-div-by-zero
   ```

   Then go to GitHub and click the "Compare & pull request" button that appears.
   In the PR description, write `Closes #42` — this automatically closes the issue
   when the PR is merged.

4. **Run the tests locally before pushing** (see [§7](#7-running-the-tests)).
   CI runs automatically on every push and PR via GitHub Actions, but a local green
   run before pushing is still good practice.

5. **Merge into main** once the PR is approved and CI passes.

### What goes directly on `main`

Only trivial, low-risk changes: fixing a typo in a docstring, updating the README.
When in doubt, use a branch.

---

## 4. Issues, milestones, and the project board

### Issues

Create an issue for every piece of work before you start — bugs, features, writing tasks,
anything. Issues serve as the record of *why* a change was made.

When filing an issue:
- Add a **label** (`bug`, `enhancement`, `testing`, `writing`, etc.)
- Assign it to the correct **milestone** (see below)
- The project board is updated automatically (via GitHub Actions)

### Milestones

Milestones group issues by release goal. Our milestones:

| Milestone | Goal | Status |
|-----------|------|--------|
| **M0** — API changes and edge cases | Fix all known bugs and finalize the public API | Closed |
| **M1** — first PyPI release (v0.3.0) | PyPI publish | Closed |
| **M2** — bioRxiv submission | Preprint | Open |
| **M3** — journal submission | Peer-reviewed paper | Open |

Close all issues in a milestone before moving to the next one.

### Project board

The "pykarambola paper" project board at
`https://github.com/orgs/Ishihara-SynthMorph/projects/4`
is our shared view of progress across all milestones.

**Important:** Milestones and the project board are independent systems in GitHub.
Assigning a milestone to an issue does *not* add it to the board.

The `.github/workflows/add-to-project.yml` workflow adds every newly opened issue to
the board automatically.
If an issue is not appearing on the board, you can add it manually via the issue's
sidebar (look for the "Projects" field on the right-hand side).

---

## 5. Version numbers

We follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

| Change | Which part increments | Example |
|--------|----------------------|---------|
| Bug fix, no API change | `PATCH` | `0.1.0` → `0.1.1` |
| New feature, no breaking change | `MINOR` | `0.1.0` → `0.2.0` |
| Breaking API change | `MAJOR` | `0.x.y` → `1.0.0` |

**While `MAJOR == 0`** (pre-stable), a `MINOR` bump is allowed to break things.
This is intentional — it signals to users that the API is still being finalized.

### Our version roadmap

```
0.3.0  M1 — first PyPI release       (released)
0.4.0  ...                            (released)
0.5.0  current                        (released)
 ...
```

Future version bumps will be driven by M2/M3 milestones.

### How to bump the version

The version lives in exactly one place: `pyproject.toml`.

```toml
[project]
version = "0.5.0"   ← change this
```

`pykarambola.__version__` reads it automatically at runtime via `importlib.metadata`,
so you never need to update `__init__.py`.

When to bump: immediately before merging the PR that closes the last issue in a milestone.
The bump itself should be its own commit:

```bash
# edit pyproject.toml, then:
git add pyproject.toml
git commit -m "Bump version to 0.6.0"
```

Then tag the commit on `main` after merging:

```bash
git tag v0.6.0
git push origin v0.6.0
```

The tag creates a permanent reference to that state of the code and is visible on the
GitHub releases page (see [§8](#8-releasing-a-new-version) for the full release process).

---

## 6. Changelog

[`CHANGELOG.md`](CHANGELOG.md) is the human-readable record of what changed in each version.
We follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

### The rule: update the changelog in the same PR as the code change

Every PR that fixes a bug, adds a feature, or changes behaviour should include a
one-line entry under `## [Unreleased]` in `CHANGELOG.md`. Do it in the same commit
as the code — don't leave it for later.

### Entry categories

Use these headings under `## [Unreleased]` (omit any that have no entries):

| Heading | When to use |
|---------|-------------|
| `### Added` | New feature or new public API |
| `### Changed` | Behaviour change to an existing feature |
| `### Deprecated` | Feature that will be removed in a future version |
| `### Removed` | Feature removed in this version |
| `### Fixed` | Bug fix |

### Example

```markdown
## [Unreleased]

### Fixed
- `get_ref_vec` no longer raises `ZeroDivisionError` on open/flat surfaces (#42)

### Added
- `minkowski_tensors` now accepts a `compute=` list to request only specific quantities (#45)
```

### At release time

When you are ready to release (see [§8](#8-releasing-a-new-version)):

1. Replace `## [Unreleased]` with the new version heading, e.g. `## [0.2.0] - 2026-06-01`
2. Add a fresh empty `## [Unreleased]` section above it
3. Update the comparison links at the bottom of the file:

```markdown
[Unreleased]: https://github.com/Ishihara-SynthMorph/pykarambola/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Ishihara-SynthMorph/pykarambola/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ishihara-SynthMorph/pykarambola/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ishihara-SynthMorph/pykarambola/releases/tag/v0.1.0
```

These links make each version heading in the changelog clickable — they open a GitHub
diff showing exactly what changed between that version and the previous one.

---

## 7. Running the tests

Run all tests:

```bash
pytest tests/
```

Run a specific file:

```bash
pytest tests/test_api.py
```

Run a specific test by name:

```bash
pytest tests/test_box.py -k "test_surface_area"
```

Show print output (useful for debugging):

```bash
pytest -s tests/test_api.py
```

GitHub Actions runs the full test matrix automatically on every push and PR
(Ubuntu, macOS, Windows × Python 3.9, 3.11, 3.13).
The required status check is `ci-success`; a PR cannot be merged until it passes.
Running `pytest tests/` locally before pushing is still good practice — it gives you
faster feedback than waiting for CI.

### Writing a test for a bug fix

Every bug fix should include a test that would have caught the bug.
Add it to the most relevant existing test file, or create a new one under `tests/`.
A minimal test looks like:

```python
def test_get_ref_vec_flat_surface():
    """get_ref_vec should not raise ZeroDivisionError on a flat open surface."""
    verts = ...
    faces = ...
    result = pk.minkowski_tensors(verts, faces)  # should not raise
    assert np.isfinite(result["w000"])
```

---

## 8. Releasing a new version

1. Confirm all milestone issues are closed.
2. Bump the version in `pyproject.toml` (see [§5](#5-version-numbers)).
3. Update `CITATION.cff` with the new version and release date.
4. Update `CHANGELOG.md`: replace `## [Unreleased]` with `## [X.Y.Z] - YYYY-MM-DD`,
   add a fresh empty `## [Unreleased]` above it, and update the comparison links at
   the bottom of the file (see [§6](#6-changelog)).
5. Open a PR titled `Release vX.Y.Z`, get it reviewed, merge.
6. On `main`, create and push the tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

7. The PyPI publish workflow triggers automatically when a tag matching `v*` is pushed
   to `main`. Verify the release appeared on PyPI and on the GitHub releases page.
