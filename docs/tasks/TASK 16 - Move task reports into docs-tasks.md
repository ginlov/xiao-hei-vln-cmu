# TASK 16 - Move task reports into `docs/tasks/`

## Purpose

Get the 20 `TASK *.md` reports off the repo root and into `docs/tasks/`, where
an index page for them already existed, and repoint every reference at the new
location.

## Problem (before)

- 20 report files sat at the repo root, dominating a directory listing that
  otherwise holds 6 real entries (`README.md`, `CLAUDE.md`, `TASK.md`,
  `pyproject.toml`, `mkdocs.yml`, `uv.lock`).
- `docs/tasks/index.md` already existed as their index, but linked each report
  through an **absolute `github.com/.../blob/main/...` URL** — so the links
  only worked for the published `main` branch, broke on any fork or branch, and
  did not render in a local `mkdocs serve`.
- The index was stale: it listed 10 of the (then) 19 reports and carried
  statuses that predated Tasks 11–15.
- The reports were outside `docs/`, so they were never part of the MkDocs site.

## What changed

Moved all 20 reports to `docs/tasks/` with `git mv` (all 19 tracked ones were
recorded as **renames**, so `git log --follow` still works).

`TASK.md` — the challenge task *definitions* the reports answer to — **stays at
the repo root**. It is a spec, not a report.

Callers repointed:

| File | Change |
|---|---|
| `CLAUDE.md` | working principle now says write to `docs/tasks/` and add a row to the index |
| `README.md` | same, with links to `docs/tasks/` and the index |
| `docs/perception-sidecar.md` | `` `TASK 11 ...` at the repo root `` → a real relative link |
| `docs/tasks/index.md` | rewritten: relative links, all 20 reports, refreshed statuses |
| `mkdocs.yml` | all 20 reports added to the `Task Reports` nav section |

The index now also records that task numbers are **not unique** — 7, 8, 9 and
10 each have multiple reports, because parallel branches reused a number. That
was invisible while the files were an unordered root listing.

## Gotchas worth remembering

- **`git mv` only works on tracked files.** The freshly written TASK 15 report
  was still untracked, so it needed a plain `mv`. A `for f in TASK\ *.md` loop
  aborts on the first untracked file.
- **`mkdocs.yml` sets `strict: true`**, and `.github/workflows/docs.yml` runs
  `mkdocs build --strict`. Moving markdown *into* `docs/` means any relative
  link inside those files is now build-validated — a broken one fails CI.
  Checked first: the reports contain **zero** relative markdown links (all
  their file references are inline code), so nothing broke.
- Report filenames keep their spaces, so links to them must be **URL-encoded**
  (`TASK%2011%20-%20...`). An unencoded space silently fails to parse as a
  markdown link rather than erroring.

## Verification

- `mkdocs build --strict` → **exit 0**, no warnings. This is the same command
  CI runs, and strict mode would have failed on any unresolved link.
- All 20 reports render into `site/tasks/` (21 pages incl. the index), and the
  "pages not included in nav" notice is gone.
- `uv run pytest -q` → **454 passed, 1 skipped** (unchanged — no code touched).
- Repo-wide grep for `blob/main/TASK`, `at the repo root` + task wording, and
  `TASK N - <purpose>` finds no stale pointer; the remaining "repo root" hits
  refer to the sidecar `perception/` directory and are correct.
- Repo root is now `README.md`, `CLAUDE.md`, `TASK.md` + config files.
