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
| 21 | [Offline perception replay harness & 3D lift repair](TASK%2021%20-%20Offline%20perception%20replay%20harness%20and%203D%20lift%20repair.md) | Done — dedup pending |
| 22 | [Multi-scene perception benchmark & the median-box fix](TASK%2022%20-%20Multi-scene%20perception%20benchmark%20and%20the%20median-box%20fix.md) | Done |
| 23 | [Where recall actually dies, measured stage by stage](TASK%2023%20-%20Where%20recall%20actually%20dies,%20measured%20stage%20by%20stage.md) | Done |
| 24 | [What the boxes are actually missing](TASK%2024%20-%20What%20the%20boxes%20are%20actually%20missing.md) | Done |
| 25 | [A viewer for the perception map, and what it found](TASK%2025%20-%20A%20viewer%20for%20the%20perception%20map,%20and%20what%20it%20found.md) | Done |
| 26 | [Can a VLM ground a referring expression, measured](TASK%2026%20-%20Can%20a%20VLM%20ground%20a%20referring%20expression,%20measured.md) | Done |
| 27 | [From a box to a waypoint, and two things measurement changed](TASK%2027%20-%20From%20a%20box%20to%20a%20waypoint,%20and%20two%20things%20measurement%20changed.md) | Done — superseded on blind lifts by 28 |
| 28 | [The approach loop, driven](TASK%2028%20-%20The%20approach%20loop,%20driven.md) | Done — superseded on the platform clamp by 29 |
| 29 | [The converter is not a clamp, and we can predict it](TASK%2029%20-%20The%20converter%20is%20not%20a%20clamp,%20and%20we%20can%20predict%20it.md) | Done |
| 30 | [Four scenes, and the bugs only driving finds](TASK%2030%20-%20Four%20scenes,%20and%20the%20bugs%20only%20driving%20finds.md) | Done — loft unsolved, `gate` not enforced |
| 31 | [Ground truth for all fifteen scenes, and what it refuted](TASK%2031%20-%20Ground%20truth%20for%20all%20fifteen%20scenes,%20and%20what%20it%20refuted.md) | Done — `KEEPOUT_M` refuted, not yet redesigned |
| 32 | [Who splits the instruction, and where keep-outs live](TASK%2032%20-%20Who%20splits%20the%20instruction,%20and%20where%20keep-outs%20live.md) | Done — model parser primary, 30/30 drive order; executor next |
| 33 | [The ordered plan executor](TASK%2033%20-%20The%20ordered%20plan%20executor.md) | Driven — `chinese_room` 2/2 against GT |
| 34 | [Driving a passage, and four ways of faking it](TASK%2034%20-%20Driving%20a%20passage,%20and%20four%20ways%20of%20faking%20it.md) | Done — 4 passage bugs fixed; `studio` crosses the gap at x=+3.46 |
| 35 | [Leaving the room the robot started in](TASK%2035%20-%20Leaving%20the%20room%20the%20robot%20started%20in.md) | Driven — `home_building_2` q1 2/2; multi-room search still unsolved |
| 21 | [Reject mask spill in the 3D lift](TASK%2021%20-%20Reject%20mask%20spill%20in%20the%203D%20lift.md) | Done |
| 22 | [Same-label duplicate suppression in ObjectMap](TASK%2022%20-%20Same-label%20duplicate%20suppression%20in%20ObjectMap.md) | Done |
| 23 | [Capture perception inputs from real navigation](TASK%2023%20-%20Capture%20perception%20inputs%20from%20real%20navigation.md) | Scripts done — sweep not yet run |
| 24 | [Audit of un-synchronised gating logic](TASK%2024%20-%20Audit%20of%20un-synchronised%20gating%20logic.md) | C resolved; A superseded (see note); B, E, F open |
| 25 | [Merge detections split by the panorama face seams](TASK%2025%20-%20Merge%20detections%20split%20by%20the%20panorama%20face%20seams.md) | Done |
| 26 | [Scan-accumulator keyframe sweep & side-by-side dump comparison](TASK%2026%20-%20Scan-accumulator%20keyframe%20sweep%20and%20side-by-side%20dump%20comparison.md) | Dumps, viewer & GIF tool done; scored sweep not run |
| 27 | [Verifying the 2D-detection → 3D-lift angular convention](TASK%2027%20-%20Verifying%20the%202D-detection%20to%203D-lift%20angular%20convention.md) | Done — extrinsic bug fixed + image/pose timestamp-matched in LatestCache |

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
