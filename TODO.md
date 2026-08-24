# TODO

Round 2 (USER, verbatim, 2026-08-24): "the style-review CI is back and broken"

The style-review job on this PR now fails with a diagnosable error, thanks
to round 1's fix:

```
gateway error 400: {"error":{"message":"Reasoning is mandatory for this
endpoint and cannot be disabled.","code":400}}
```

This confirms the hypothesis the PR description left open: `stealth/ox-alpha`
requires reasoning, but `ask()` unconditionally sends
`"reasoning": {"enabled": False, "exclude": True}`, which was presumably
needed for the previous model to keep it from burning the `max_tokens=8192`
budget on reasoning tokens instead of the JSON answer.

- [WIP] @claude-issue611-2026-08-24 16:25 `ask()` retries once without the
  `reasoning` field when the gateway's 400 says reasoning can't be
  disabled, instead of crashing the job — keeps the disable-by-default
  behavior for models that support it
- [ ] confirm on this PR's own `style-review` run that it now succeeds
  against `stealth/ox-alpha`
