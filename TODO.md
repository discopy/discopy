# TODO

> Keep on going with the neural GoI experiments, previous session was run with
> Opus so don't trust the results. In particular it decided to write code by
> hand rather than using diagrams because LCS required swaps, unlike minimum
> and sorting which were planar. With the recent features we added to
> symmetric diagrams this shouldn't matter but if it does please make sure to
> raise the issue.
>
> By the way I found a nice title and acronym: GoNI (Geometry of Neural
> Interaction)
>
> Pick a new task in the CLRS benchmark where you expect the gap GoNI vs SOTA
> to be the largest

- [x] Rebuild the GoNI setup on a clean branch: merge the `discopy.neural`
      host branch (`claude/proposal-b-three-models-awc5sc`) and get its test
      suite green here.
- [x] Build the LCS dynamic-programming grid as a symmetric diagram — the
      crossings as permutation layers — and check `MapNN` compiles it; if
      any real obstruction remains, file the issue.
- [x] Pick the new CLRS-30 task maximising the expected GoNI-vs-SOTA gap,
      with the published SOTA numbers written down next to the choice.
- [WIP] @goni-ocirk4-2026-08-29 14:00 Implement the chosen task: benchmark-faithful data, the dataflow
      circuit family, encoders/decoders and training script.
- [ ] Train at small budget, evaluate in and out of distribution, and record
      the results with seeds and protocol.
- [ ] Parallel (mac mini): train the LCS grid on the benchmark's
      `lcs_length` splits, redoing the untrusted results as diagrams.
- [ ] Parallel (mac mini): scale the trained kmp matcher past the
      benchmark, n = 128 and 256, where no baseline has numbers.
