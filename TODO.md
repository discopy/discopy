# TODO

> Give a shot at this quantum GoI demo
> https://github.com/rel-int/internal-issues/issues/14

Issue body (toumix, rel-int/internal-issues#14):

> We need a demo of the following pipeline:
>
> 1) take a sentence and parse it to a diagram via bobcat or spindle or any other categorial grammar parser
> 2) find a heuristic to bracket the resulting diagram into bubbles with a maximal size, the inside of the buble represents a quantum circuit which gets called by the outside of the bubble representing a Python function with disjoint union as tensor
> 3) we perform additive token-passing particle-style GoI on the diagram, where a classical token travels around the diagram, setting the parameters of a quantum circuit as it comes into a bubble, reading from its measurement results as it comes out and travels to the next bubble
> 4) the state of the token is decoded as a linear lambda term which represents the diagram for the answer of the overall process, this decoding happens incrementally i.e. we can see the answer being generated one subtree at a time by traveling through the diagram, when it comes out of the boundary of the overall diagram the answer has converged
> 5) the overall process is displayed as an interactive diagram where we can slide forward in time to see the generation process in a two-pane interface where the parsed diagram and the position of the token is on top, the current state of the decoded answer is at the bottom
>
> This should build upon the following DisCoPy PRs:
> - https://github.com/discopy/discopy/pull/401 for the neural GoI version of decoding lambda terms
> - https://github.com/discopy/discopy/pull/400 for the implementation of parsing as lambda terms (take a simple example of a sentence like "take a simple example of a sentence")

## Work

- [ ] Merge PR #400 (`claude/acg-issue-plan-hx5an6`) and PR #401 (`claude/neural-lambda-experiment`) into this branch
- [ ] Parse a sentence to a categorial diagram and lambda term (issue step 1)
- [ ] Bubble-bracketing heuristic: split the term's map into quantum bubbles of maximal size called from a classical `python.additive` outside (issue step 2)
- [ ] Additive token-passing GoI with the token setting circuit parameters on entry and reading measurements on exit (issue step 3)
- [ ] Incremental decoding of the token trace as a linear lambda term, one subtree at a time (issue step 4)
- [ ] Interactive two-pane time-slider display: diagram + token position on top, decoded answer below (issue step 5)
