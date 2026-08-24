# TODO

Codex's review of #608, summoned by USER, quoted from its findings:

> 1. The workflow can still report a clean review after omitting changed
> files [...] I would make omission of **changed** files fail loudly, while
> continuing to drop imported context as necessary.
>
> 2. Mixed valid and malformed findings are silently accepted [...] Since
> this component is the deterministic validation boundary, I would reject
> the response if **any** finding is malformed.

- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 11:52 `review.py`
  raises when a changed file misses the budget, so a clean verdict always
  means every changed file was read whole; context files still drop with
  a note
- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 11:52 `post.py`
  raises when any reported finding is unreadable after coercion, instead
  of silently posting the readable subset
