#import "@preview/cetz:0.5.2": canvas, draw
#import "@preview/fletcher:0.5.8" as fletcher
#set page(width: auto, height: auto, margin: 0.2cm)

#canvas(length: 1in, {
    import draw: *
    line((0, 0), (1, 0), (1, 1), (0, 1), close: true, fill: rgb("#ffffff"), stroke: rgb("#ffffff") + 1pt)
    line((0.5, 0.5), (0.5, 0.25), stroke: rgb("#000000") + 1pt)
    content((0.5625, 0.9375),     [1], anchor: "west")
    line((0.5, 1), (0.5, 0.75), stroke: rgb("#000000") + 1pt)
    line((0.5, 0.75), (0.5, 0.5), stroke: rgb("#000000") + 1pt)
    content((0.5625, 0.1875),     [1], anchor: "west")
    line((0.5, 0.25), (0.5, 0), stroke: rgb("#000000") + 1pt)
    rect((0.35, 0.35), (0.65, 0.65), fill: rgb("#f7f700"), stroke: rgb("#000000") + 0.5pt)
})
