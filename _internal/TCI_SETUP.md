# Quantics TCI in Julia

Since this project is written in Julia, the tensor4all TCI packages can be
used directly — no `juliacall` bridge is needed.

## Install (one-time)

```julia
using Pkg
Pkg.add(["TensorCrossInterpolation", "QuanticsTCI", "QuanticsGrids"])
```

## Minimal usage

```julia
using TensorCrossInterpolation, QuanticsTCI, QuanticsGrids

R    = 20
grid = DiscretizedGrid{1}(R, 0.0, 2π)
f    = x -> sin(5.0 * x) * exp(-x / 3.0)

qtt_jl, ranks, errors = quanticscrossinterpolate(Float64, f, grid; tolerance=1e-8)
```

## Converting QuanticsTCI cores into a QTT object

QuanticsGrids places the **MSB on site 1** by default.  This project places
the **LSB on site 1**.  When converting cores:

1. Reverse the site list (site 1 ↔ site N).
2. Swap the left/right bond axes of each core: `permutedims(core, (3, 2, 1))`.

See `src/FromTCI.jl` for the full conversion function.

## Grid endpoint convention mismatch

QuanticsGrids samples `2^R` points at `x1 + k*(x2-x1)/2^R` for `k = 0..2^R-1`
(the right endpoint `x2` is **not** a sample point).  This project's `GridInfo`
uses endpoint-inclusive spacing: `x(k) = x1 + k*(x2-x1)/(2^N-1)`.

To make `qtt.evaluate(k)` reproduce the k-th QuanticsGrids point, store a
slightly shrunk upper endpoint:

    x2_stored = x1 + (x2 - x1) * (2^N - 1) / 2^N
