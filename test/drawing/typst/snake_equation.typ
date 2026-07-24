#import "@preview/cetz:0.5.2": canvas, draw
#import "@preview/fletcher:0.5.8" as fletcher
#set page(width: auto, height: auto, margin: 0.2cm)

#canvas(length: 1in, {
    import draw: *
    line((0, 0), (3, 0), (3, 2), (0, 2), close: true, fill: rgb("#ffffff"), stroke: rgb("#ffffff") + 1pt)
    bezier((1, 1.5), (0.5, 1), (0.6667, 1.5), (0.5, 1.3333), stroke: rgb("#000000") + 1pt)
    bezier((1, 1.5), (1.5, 1), (1.3333, 1.5), (1.5, 1.3333), stroke: rgb("#000000") + 1pt)
    content((0.5625, 1.1875),     [n.r], anchor: "west")
    line((0.5, 1), (0.5, 0), stroke: rgb("#000000") + 1pt)
    content((1.5625, 1.1875),     [n], anchor: "west")
    content((2.5625, 1.9375),     [n.r], anchor: "west")
    line((2.5, 2), (2.5, 1), stroke: rgb("#000000") + 1pt)
    bezier((1.5, 1), (2, 0.5), (1.5, 0.6667), (1.6667, 0.5), stroke: rgb("#000000") + 1pt)
    bezier((2.5, 1), (2, 0.5), (2.5, 0.6667), (2.3333, 0.5), stroke: rgb("#000000") + 1pt)
})
