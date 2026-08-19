# TODO

> Fix this drawing issue https://github.com/discopy/discopy/issues/521

- [x] Expose `Backend.region_separators` and `Backend.region_cells`, the
  trapezoidal decomposition of a drawing into its exact region extents:
  polygons bounded by the wires on both sides, subdivided per height band
- [x] Replace the painter's algorithm of `Matplotlib.draw_regions` with one
  patch per region cell, leaving white cells unpainted so that they erase
  to the background
- [x] Update the region tests and add exactness tests for the new methods
- [x] Regenerate the affected SVG baselines and check the coloured ones by eye
- [x] Add a CHANGELOG entry
