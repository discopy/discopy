ok in this discopy branch in @discopy/test_widget_repr.py  it works!!! renders typst, but now i see this. it doesn't look like the diagrams in filippo bonchi's diagrammatic algebra of first order logic, which is in https://arxiv.org/pdf/2401.07055  and source in https://arxiv.org/src/2401.07055 - what do we do? lets fix it properly and make it look like a real string diagram

- [x] @composer-2026-07-24 13:15 Fix Typst CeTZ output: A4 page shrinks diagram; boxes have stroke:none — match real string-diagram look (wires + bordered boxes, tight SVG)
- [WIP] @composer-2026-07-24 13:25 Fix Typst labels: Alice→unknown variable (bare words must not be math); long $math$ overflows box (canvas length 1cm vs text_width inches)
