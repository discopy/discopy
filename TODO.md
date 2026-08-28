# TODO

Style review round on #677, the two findings quoted verbatim:

> STYLE.md says DisCoPy has no secrets and avoids private attributes;
> `_box_port_indices` is private. If a public interface for the port indices
> exists, please use it, otherwise consider exposing one.

> STYLE.md says DisCoPy never repeats itself; `run_watermarked` duplicates the
> token-machine logic of `run` (lines 123-169) with only watermark tracking
> added. Consider factoring out the shared step to avoid divergence.

- [WIP] @session_01NMm77dxzEoNwSSAt3KUcRq-2026-08-28 10:15 Hoist `box_ports`
  from `neural.CMap` to `cmap.CMap` so the notebook reads ports through a
  public interface, and reuse it where `cmap.py` repeats the pattern
- [WIP] @session_01NMm77dxzEoNwSSAt3KUcRq-2026-08-28 10:15 Factor the shared
  transition out of `run` and `run_watermarked` in `neural-church.md`
