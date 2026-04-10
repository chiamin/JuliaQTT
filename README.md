# JuliaQTT

A Julia library for **Quantics Tensor Train (QTT)** representations of functions on dyadic grids.
Built on top of [JuliaMPS](https://github.com/chiamin/JuliaMPS).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/chiamin/JuliaMPS")
Pkg.add(url="https://github.com/chiamin/JuliaQTT")
```

## Quick example

Represent sin(x) on [0, 2π] with 2^10 = 1024 grid points and evaluate it:

```julia
using QTTCore

N = 10           # number of QTT sites → 2^N grid points
x1, x2 = 0.0, 2π

# Build exact QTT for sin(x)
q = qtt_sin(N, x1, x2)

# Evaluate at a few grid points
dx = (x2 - x1) / (2^N - 1)
for k in [0, 2^N ÷ 4, 2^N ÷ 2]
    xv  = x1 + k * dx
    println("k=$k  QTT=$(round(evaluate(q, k), digits=6))  exact=$(round(sin(xv), digits=6))")
end
```

Output:
```
k=0    QTT=0.0       exact=0.0
k=256  QTT=0.999999  exact=0.999999
k=512  QTT=0.999999  exact=1.0
```

## Features

- Exact analytic constructors: `qtt_sin`, `qtt_cos`, `qtt_linear`, `qtt_exp`, `qtt_random`
- Arithmetic: `qtt_sum`, `qtt_prod`
- Differential operators: `qtto_diff_forward`, `qtto_diff_backward`, `qtto_diff2`
- Shift operators: `shift_forward_mpo`, `shift_backward_mpo`
- Interpolation and coarsening: `qtt_interp`, `qtt_coarsen`
- Linear solver (ALS/DMRG-style): `LinearSolverEngine`
- Multi-dimensional support via interleaved site ordering
