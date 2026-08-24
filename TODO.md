# TODO

Round 3 (USER, verbatim, 2026-08-24): "gateway error 400:
{\"error\":{\"message\":\"Reasoning is mandatory for this endpoint and cannot
be disabled.\",\"code\":400,\"metadata\":{\"provider_name\":null}}}"

The round-2 retry worked — no second `HTTPError` — but the job then failed
differently: `json.decoder.JSONDecodeError: Expecting ',' delimiter: line 2
column 46 (char 113)` from `ask`'s
`json.loads(answer[answer.index("{"):answer.rindex("}") + 1])`. Unknown
whether the reasoning-enabled answer got truncated by `max_tokens=8192`,
contains an example JSON object ahead of the real one, or something else —
`answer` itself is never logged, so there's nothing to diagnose from yet.

- [x] log (truncated) `answer` when JSON extraction fails, the same way
  `ask` already logs the gateway's error body, so the next run's failure
  is diagnosable

Follow-up (USER, verbatim, 2026-08-24): "succeeded but in a pretty bad way"
— the next two runs against `stealth/ox-alpha` (with reasoning forced on)
both "succeeded" with 0 findings on a PR that has plenty to say something
about. Likely cause: `max_tokens=8192` is shared between reasoning tokens
and the answer on this gateway, so a mandatory-reasoning model can spend
most/all of it thinking, leaving a truncated or trivially-empty answer that
still happens to be syntactically valid JSON — same root cause as the
`JSONDecodeError`, just landing on the lucky side of the coin flip instead
of the unlucky one.

- [ ] log `finish_reason`/`usage` on every response (not just on failure)
  to confirm whether `length` truncation is actually happening
- [ ] once confirmed, give the reasoning-enabled retry its own, much
  larger `max_tokens` (the model supports up to 131,072 completion
  tokens) so reasoning no longer starves the answer
