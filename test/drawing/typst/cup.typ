#import "@preview/cetz:0.5.2": canvas, draw
#import "@preview/fletcher:0.5.8" as fletcher
#set page(width: auto, height: auto, margin: 0.2cm)

#canvas(length: 1in, {
    import draw: *
    line((0, 0), (2, 0), (2, 1), (0, 1), close: true, fill: rgb("#ffffff"), stroke: rgb("#ffffff") + 1pt)
    content((0.5625, 0.9375),     [n], anchor: "west")
    bezier((0.5, 1), (1, 0.5), (0.5, 0.6667), (0.6667, 0.5), stroke: rgb("#000000") + 1pt)
    content((1.5625, 0.9375),     [n.r], anchor: "west")
    bezier((1.5, 1), (1, 0.5), (1.5, 0.6667), (1.3333, 0.5), stroke: rgb("#000000") + 1pt)
})
