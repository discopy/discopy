# TODO

> ok lets make a PR targetting `ale/section-order` with What's next: one PR "Rewriting in ehypergraph"                                                                                                                
>                                                                                                                                                                
>  1. Rules without a Rule class. A rule is an axioms.Equation(lhs, rhs) of parallel diagrams, direction given by its symbol (your own decision in               
>     tmp/discopy-port/QUESTIONS.md:17-18). Holes are the box variables, matched by boundary type; no type variables (cells are keyed by Box, per the #730 scope 
>     note). Enough to state e.g. swap naturality f @ g >> Swap(B, Y) == Swap(A, X) >> g @ f with f, g holes — metatheory's theories/symmetric.py:19-40          
>     verbatim.                                                                                                                                                  
>  2. Symmetric.match(pattern) -> Iterator[dict[int, int] from_diagram(lhs), then a backtracking join over its cells: candidates for a box are its shard's     
>     live rows (shards is already keyed by box, :396-397), for a hole every shard whose box has the same dom/cod; bind each pattern wire to uf.find of the host 
>     wire, a repeated wire forces equal vertices. This is _ematch._match_lhs (_ematch.py:125-162), ~30 lines; O(cells) per conjunct is what metatheory does too 
>     (carrier_matcher/__init__.py:29-31).                                                                                                                       
>  3. Symmetric.rewrite(rule, binding) — ~5 lines: rhs.fill({hole: matched box}) (#808), tgt = self.lower(rhs, [binding[w] for w in lhs.dom_wires]), then        
>     merge(tgt[i], binding[lhs.cod_wires[i]]). That is exactly metatheory's WRITE (_write.py:146-168 + assert_eq at _ematch.py:324-334): intern the RHS onto    
>     the LHS's leaf wires, union the roots. One change needed: lower_box calls append (:657-658), switch it to intern  a rewrite is idempotent before         
>     rebuild.                                                                                                                                                   
>  4. saturate(rules=(), fuel=1000). Extend the loop at :515-520: snapshot all matches of all rules, apply them, canonize, rebuild, stop when no row and no      
>     merge was added or fuel runs out. equiv then decides; the test is metatheory's harness prove (tests/_harness.py:77-133): lower both sides on a shared      
>     boundary, saturate, compare.   INVESTIGATE VARIOUS SOLUTIONS, OPTIONS AND THEIR TRADEOFFS AND PROS AND CONS AND ASK QUESTIONS BEFORE YOU MAKE A DECISION.
>
> READ DOCS AND EXAMPLES AND SOURCE CODE IN FULL

- [x] `Symmetric.lower_rule`, `lower_equation`, `candidates`, `binding`, `match`
- [x] `Symmetric.rewrite`, `close`, `saturate(equations, fuel, until)`, `equiv(other, equations)`, `Markov.congruence`
- [ ] Tests in `test/ehypergraph.py`, module docstring and changelog
