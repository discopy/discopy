#import "@preview/cetz:0.5.2": canvas, draw
#import "@preview/fletcher:0.5.8" as fletcher
#set page(width: auto, height: auto, margin: 0.2cm)

#canvas(length: 1in, {
    import draw: *
    line((0, 0), (2, 0), (2, 2), (0, 2), close: true, fill: rgb("#ffffff"), stroke: rgb("#ffffff") + 1pt)
    content((0.5625, 1.9375),     [x], anchor: "west")
    line((0.5, 2), (0.5, 1.75), stroke: rgb("#000000") + 1pt)
    content((0.5625, 1.1875),     [y], anchor: "west")
    line((0.5, 1.25), (0.5, 0), stroke: rgb("#000000") + 1pt)
    content((1.5625, 1.9375),     [y], anchor: "west")
    line((1.5, 2), (1.5, 0.75), stroke: rgb("#000000") + 1pt)
    content((1.5625, 0.1875),     [x], anchor: "west")
    line((1.5, 0.25), (1.5, 0), stroke: rgb("#000000") + 1pt)
    line((0.25, 1.25), (0.25, 1.75), (0.75, 1.75), (0.75, 1.25), close: true, fill: rgb("#ffffff"), stroke: rgb("#000000") + 1pt)
    content((0.5, 1.5),     [f])
    line((1.25, 0.25), (1.25, 0.75), (1.75, 0.75), (1.75, 0.25), close: true, fill: rgb("#ffffff"), stroke: rgb("#000000") + 1pt)
    content((1.5, 0.5),     [g])
})
