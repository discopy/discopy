# TODO

> @"/root/.claude/uploads/4b51c9ff-382c-52dd-b42d-16dd1859b761/fa19f400-eternamapnn.md" Let's solve the Eterna inverse folding challenge using map neural networks! See an initial proposal attached. You can use the functions in the bio repo to translate the molecules into discopy format.

The experiment itself lives in `rel-int/bio`; this branch carries only what the
experiment needed from DisCoPy and could not get.

- [x] `CMap.genus`, the invariant the RNA application reads pseudoknots off.
      `euler_characteristic` and `is_planar` were both here, but the number of
      handles itself was not, and "how pseudoknotted is this structure" is a
      question about the genus, not about a yes/no planarity.
- [x] Define `is_planar` in terms of it, rather than repeating the scalar
      special case and the comparison against 2.
- [x] Test both, including the scalar and disconnected cases.
- [x] `CHANGELOG.md` entry.
