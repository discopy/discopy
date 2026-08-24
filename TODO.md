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

- [ ] log (truncated) `answer` when JSON extraction fails, the same way
  `ask` already logs the gateway's error body, so the next run's failure
  is diagnosable
- [ ] once a run has revealed the actual shape of `answer`, fix the
  extraction for real rather than guessing now
