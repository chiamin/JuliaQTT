# Project Notes

## Overview

QTT is a Quantics Tensor Train library built on top of
[MPSCore](../../MPSCore/) (the Julia port of UnitenDMRG).  It uses the
`MPS` and `MPO` types from that package.

QTT represents functions on dyadic grids (2^N points) as MPS, where the
physical index at each site is binary (dim = 2) and the grid index is the
binary expansion of the point index (LSB on site 1).

## Dependencies

- **MPSCore** — local path dependency providing `MPS`, `MPO`,
  `exact_apply_mpo`, `mps_sum`, `direct_sum`, `OperatorEnv`, `VectorEnv`,
  `EffOperator`, `EffVector`, and linear-algebra utilities.
- **KrylovKit** — for iterative linear solvers (GMRES / `linsolve`).
- **Plots** (optional) — for `plot_qtt_1d`, `plot_qtt_2d`.

## Module structure

| File | Contents |
|------|----------|
| `src/GridInfo.jl` | `GridInfo` struct: per-dimension grid metadata |
| `src/QTT.jl` | `QTT` type: multi-dimensional MPS function |
| `src/QTTO.jl` | `QTTO` type: multi-dimensional MPO operator |
| `src/Init.jl` | Analytic constructors: `qtt_sin/cos/linear/exp/random` |
| `src/Operators.jl` | Analytic MPOs: shift operators, finite-difference QTTOs |
| `src/Operations.jl` | `qtt_sum`, `qtt_prod`, `embed`, interpolation, coarsening |
| `src/FromTCI.jl` | Load QTT from QuanticsTCI cores |
| `src/Plot.jl` | `plot_qtt_1d`, `plot_qtt_2d` |
| `src/LinearSolver.jl` | `LinearSolverEngine`: ALS/DMRG-style solver for Ax=b |

## Coding conventions

### Stay within Julia / MPSCore

All tensor operations use Julia native arrays and MPSCore APIs.  Do not
add new dependencies as a workaround.

### Avoid filenames in error messages and comments

Do not hardcode module paths in error messages or inline comments.

### Quote style in comments

Use **double quotes** to quote leg names and code terms inside comments
(e.g. `"l"`, `"ip"`).
