> yes go ahead on both, refactoring via Stream sounds especially meaningful, Daphne confirmed the bias is a bit more general so let's also implement that, but first can you explain what this bias is about?

(toumix, 2026-10-21, interactive, on daydream6728's review of this pull request:)

> any meaningful way to instantiate Reservoir as a Stream or some kind of trace/feedback structure? the only difference in interfaces I can see is that Reservoir.step takes values while Stream.unroll doesn't.

> add an optional bias to reflect the original paper

- [x] the reservoir as a `Stream[Circuit]` and its inputs as the stream `Reservoir.encoding`, as merged into next by rel-int/discopy-next#209
- [x] an optional readout bias, on by default and unpenalised, from the same pull request
