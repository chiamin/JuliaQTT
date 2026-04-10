# example_linear_solver_minimal.jl
#
# Minimal example: solve A x = b with LinearSolverEngine.
#
# Solves the 1D Poisson equation -u''(x) = sin(x) on [0, π] using QTT,
# in the fewest lines possible.  See example_poisson_solver.jl for a
# fuller benchmark.
#
# Run:
#     julia --project=.. examples/example_linear_solver_minimal.jl

using QTTCore, MPSCore, LinearAlgebra, Printf

# ── Problem setup ────────────────────────────────────────────────────
N  = 8                                        # 2^N = 256 grid points
h  = π / (2^N + 1)                            # interior-aligned grid
x1 = h
x2 = π - h

# Operator A = -∂² with Dirichlet BC.
A_mpo = (-1.0 * qtto_diff2(N, x1, x2; bc="dirichlet")).mpo

# Right-hand side b = sin(x).
b_mps = qtt_sin(N, x1, x2).mps
move_center!(b_mps, 1)

# Initial guess: random MPS, center at site 1.
x_mps = random_mps(Float64, N, 2, 4; normalize=false, seed=0)
move_center!(x_mps, 1)

# ── Solve ────────────────────────────────────────────────────────────
# `krylovdim` controls the Krylov subspace size (larger → more robust).
# `tol`       is the local GMRES tolerance per site window.
# `max_dim`   caps the bond dimension after each SVD split.
# `cutoff`    drops singular values below this threshold.
# `num_center = 2` → two-site ALS (can grow bond dimension).
engine = LinearSolverEngine(x_mps, A_mpo, b_mps; krylovdim=100, tol=1e-12)

for s in 1:5
    max_res, avg_trunc = sweep!(engine; max_dim=16, cutoff=1e-12, num_center=2)
    @printf("sweep %d: max local residual = %.2e  avg trunc = %.2e\n",
            s, max_res, avg_trunc)
end

# ── Check global residual ‖Ax - b‖ / ‖b‖ ────────────────────────────
# Use inner products: ‖Ax - b‖² = ⟨Ax|Ax⟩ - 2 Re⟨Ax|b⟩ + ⟨b|b⟩
Ax_mps = exact_apply_mpo(A_mpo, engine.x)
move_center!(Ax_mps, 1)
r2 = (real(inner(Ax_mps, Ax_mps))
      - 2.0 * real(inner(Ax_mps, b_mps))
      + real(inner(b_mps, b_mps)))
res = sqrt(max(r2, 0.0)) / sqrt(real(inner(b_mps, b_mps)))
@printf("\n‖Ax - b‖ / ‖b‖ = %.2e\n", res)
