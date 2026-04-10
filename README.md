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

# Evaluate at grid index k (integer 0 to 2^N - 1)
k   = 256
dx  = (x2 - x1) / (2^N - 1)
xv  = x1 + k * dx

println("QTT value : ", evaluate(q, k))
println("Exact     : ", sin(xv))
```

Output:
```
QTT value : 1.0
Exact     : 1.0
```

## Features

- Exact analytic constructors: `qtt_sin`, `qtt_cos`, `qtt_linear`, `qtt_exp`, `qtt_random`
- Arithmetic: `qtt_sum`, `qtt_prod`
- Differential operators: `qtto_diff_forward`, `qtto_diff_backward`, `qtto_diff2`
- Shift operators: `shift_forward_mpo`, `shift_backward_mpo`
- Interpolation and coarsening: `qtt_interp`, `qtt_coarsen`
- Linear solver (ALS/DMRG-style): `LinearSolverEngine`
- Multi-dimensional support via interleaved site ordering
