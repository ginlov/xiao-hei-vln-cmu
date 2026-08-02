# Task Reports

Development progress is tracked through numbered tasks. Each report lives in
this directory (`docs/tasks/`) alongside this index.

Task numbers are **not unique** — several rounds ran in parallel on different
branches and reused a number, so 7, 8, 9 and 10 each have more than one report.
Every row below links to a distinct document.

| Task | Title | Status |
|---|---|---|
| 1 | [Consolidate input/output format](TASK%201%20-%20Consolidate%20input-output%20format%20for%20the%20VLM%20model.md) | Done |
| 2 | [End-to-end Docker integration](TASK%202%20-%20End-to-end%20integrate%20to%20docker%20container.md) | Done |
| 3 | [VLA-3D generated dataset pipeline](TASK%203%20-%20VLA-3D%20generated%20dataset%20pipeline.md) | Done |
| 4 | [Integrate Qwen3.5 VL for testing](TASK%204%20-%20Integrate%20Qwen3.5%20VL%20for%20testing.md) | Superseded by 17 |
| 5 | [VLM tick logger](TASK%205%20-%20VLM%20tick%20logger%20for%20debugging%20and%20visualization.md) | Done |
| 6 | [Documentation webpage](TASK%206%20-%20Documentation%20webpage.md) | In progress |
| 7 | [Data coupling — coverage trajectory generation](TASK%207%20-%20Data%20coupling.md) | Done |
| 7 | [Data sample visualization script](TASK%207%20-%20Data%20sample%20visualization%20script.md) | Done |
| 7 | [Evaluation pipeline integration](TASK%207%20-%20Evaluation%20pipeline%20integration.md) | Done |
| 8 | [Frontier-based room exploration](TASK%208%20-%20Frontier-based%20room%20exploration.md) | Superseded by 12 / 15 |
| 8 | [Ref dataset ambiguity audit & label-vocab fix](TASK%208%20-%20Ref%20dataset%20ambiguity%20audit%20and%20label-vocab%20fix.md) | Done |
| 9 | [Expand ref corpus & match official phrasing distribution](TASK%209%20-%20Expand%20ref%20corpus%20and%20match%20official%20phrasing%20distribution.md) | Done |
| 9 | [Gemini-backed responder for Task 1 and Task 2](TASK%209%20-%20Gemini-backed%20responder%20for%20Task%201%20and%20Task%202.md) | Superseded by 14 |
| 10 | [Offline Gemini evaluation pipeline](TASK%2010%20-%20Offline%20Gemini%20evaluation%20pipeline%20for%20Task%201%20and%20Task%202.md) | Done |
| 10 | [Scene representation](TASK%2010%20-%20Scene%20representation.md) | Done |
| 11 | [Perception responder](TASK%2011%20-%20Perception%20responder.md) | Done |
| 12 | [Consolidate exploration + scene-building pipeline](TASK%2012%20-%20Consolidate%20exploration%20+%20scene-building%20pipeline.md) | Done |
| 13 | [Merge perception enhancements into the pipeline](TASK%2013%20-%20Merge%20perception%20enhancements%20into%20the%20pipeline.md) | Done |
| 14 | [Consolidate submission pipeline into the scene_gemini responder](TASK%2014%20-%20Consolidate%20submission%20pipeline%20into%20the%20scene_gemini%20responder.md) | Done — live smoke test pending |
| 15 | [Retire the unreachable Phase-A exploration stack](TASK%2015%20-%20Retire%20the%20unreachable%20Phase-A%20exploration%20stack.md) | Done |
| 16 | [Move task reports into `docs/tasks/`](TASK%2016%20-%20Move%20task%20reports%20into%20docs-tasks.md) | Done |
| 17 | [Retire the qwen responder](TASK%2017%20-%20Retire%20the%20qwen%20responder.md) | Done |
| 18 | [Deduplicate the responder factory](TASK%2018%20-%20Deduplicate%20the%20responder%20factory.md) | Done |
| 19 | [Remove the near-relation subsystem](TASK%2019%20-%20Remove%20the%20near-relation%20subsystem.md) | Done |
| 20 | [Exploration benchmarking across Unity scenes](TASK%2020%20-%20Exploration%20benchmarking%20across%20Unity%20scenes.md) | Done |
| 21 | [Reject mask spill in the 3D lift](TASK%2021%20-%20Reject%20mask%20spill%20in%20the%203D%20lift.md) | Done |
| 22 | [Same-label duplicate suppression in ObjectMap](TASK%2022%20-%20Same-label%20duplicate%20suppression%20in%20ObjectMap.md) | Done |
| 23 | [Capture perception inputs from real navigation](TASK%2023%20-%20Capture%20perception%20inputs%20from%20real%20navigation.md) | Scripts done — sweep not yet run |
| 24 | [Audit of un-synchronised gating logic](TASK%2024%20-%20Audit%20of%20un-synchronised%20gating%20logic.md) | C resolved (single fusion path); B, E, F open |

The task *definitions* these reports answer to stay at the repo root in
`TASK.md`.

## Backlog

Diagnosed but unfixed gaps, with the evidence attached, live in
[backlog.md](backlog.md).

## Design documents

Detailed technical documents are in the `docs/` directory:

- [Task 1 Phase 1: Sensor measurements](../task1_phase1_measurements.md)
- [Task 1: I/O specification](../task1_io_spec.md)

### Historical design records

Kept for their recorded rationale; they describe code that no longer exists.

- [Task 3 Phase 1: Qwen serving framework](../task3_phase1_framework.md) —
  retired in TASK 17
- [Task 3 Phase 3: Qwen numerical prompt design](../task3_phase3_prompt.md) —
  retired in TASK 17
