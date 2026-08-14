#import "@preview/cetz:0.5.2": canvas, draw
#import "@preview/fletcher:0.5.8" as fletcher
#set page(width: auto, height: auto, margin: 0.2cm)

#canvas(length: 1in, {
    import draw: *
    line((0, 0), (2, 0), (2, 1), (0, 1), close: true, fill: rgb("#ffffff"), stroke: rgb("#ffffff") + 1pt)
    bezier((0.5, 1), (0.5, 0.8), (0.66, 0.68), (0.852, 0.576), stroke: rgb("#000000") + 1pt)
    bezier((1.148, 0.424), (1.34, 0.32), (1.5, 0.2), (1.5, 0), stroke: rgb("#000000") + 1pt)
    bezier((1.5, 1), (1.5, 0.5), (0.5, 0.5), (0.5, 0), stroke: rgb("#000000") + 1pt)
    content((0.5625, 0.9375),     [x], anchor: "west")
    content((1.5625, 0.9375),     [x], anchor: "west")
    content((0.5625, 0.1875),     [x], anchor: "west")
    content((1.5625, 0.1875),     [x], anchor: "west")
})
