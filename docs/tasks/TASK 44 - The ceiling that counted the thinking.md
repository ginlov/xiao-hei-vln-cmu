# TASK 44 — The ceiling that counted the thinking

A grounding call died with the error the code raises on itself:

```
RuntimeError: reply hit max_tokens (4096) and is truncated
— raise it rather than treating this as a bad reply
```

That message is TASK 40's work: truncation used to arrive as "unparseable
reply", which reads as a model failure, so `ask_claude` and `ask_gemini` both
check the stop reason and say plainly that the budget ran out. The check did
its job. What it reported, though, is not the failure TASK 40 anticipated.

## Not a longer answer — a thinking model against an old ceiling

The number in the parentheses is `usage.output_tokens`, and it came back
**equal to the ceiling**. A reply cut off late spends most of the budget on the
answer and stops mid-object; this one spent all 4096 and produced nothing to
cut off. Grounding replies were measured at 460 output tokens on v3, and v4
pushed a five-lantern `japanese_room` answer past 2048 — nothing about v4 to v6
takes an answer from ~2000 to 4096.

The cause is the model, not the prompt. The default is `claude-opus-5`, and on
Opus 5 **thinking is on unless it is switched off** — a change from 4.8 and
4.7, where omitting the `thinking` parameter meant no thinking at all. Thinking
is billed against the same `max_tokens` ceiling as the visible text, so a call
that reasons its way across four 100°-wide faces can exhaust the budget before
it writes a brace.

This is exactly the shape TASK 40 documented on the other backend, where the
3.x Gemini models think without being asked and the fix was
`XIAO_HEI_GEMINI_MAX_TOKENS` at 8192. The note in the runbook called it one of
"two things that differ from Claude". It no longer differs.

## What changed

`scripts/vlm_probe.py`, `ask_claude` — the only Anthropic call site in
`scripts/`, so `execute_plan.py`, `approach_loop.py`, `decompose.py` and the
offline probes all inherit it:

```python
max_tokens=int(os.environ.get("XIAO_HEI_CLAUDE_MAX_TOKENS", "16000")),
```

16000 rather than something larger: the SDK's **non-streaming** requests hit an
HTTP timeout well below the model's 128000 output ceiling, and going past
~16000 means moving this call to `messages.stream()`. That is a bigger change
than this failure justifies, and 16000 is roughly four times the thinking-plus-
answer spend that blew the old ceiling.

The error text now names the variable to raise and says the count includes
thinking, matching the Gemini message beside it.

Docs: the runbook gains the variable next to the `--model` row, a
troubleshooting entry keyed to the error text, and a correction to the Gemini
section that no longer claims this is a Claude/Gemini difference.

## Not verified against the API

No Anthropic key was present in the session that made this change, so the fix
is reasoned from the documented Opus 5 default and from `output_tokens`
landing exactly on the ceiling — **not** from a re-run. The first real drive on
this branch is the test. If a call still reports truncation at 16000, the
budget is not the whole story and the next step is `thinking.display:
"summarized"` on one call to see what the reasoning is actually spending.

## Files

| file | change |
|---|---|
| `scripts/vlm_probe.py` | `max_tokens` 4096 → `XIAO_HEI_CLAUDE_MAX_TOKENS` (16000); error names the variable |
| `docs/guides/drive-loop-runbook.md` | variable documented, troubleshooting row, Gemini section corrected |
