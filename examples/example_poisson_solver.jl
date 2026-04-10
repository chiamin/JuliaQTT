# example_poisson_solver.jl
#
# 1D Poisson benchmark for LinearSolverEngine using QTT.
#
# Solves the discrete Poisson equation
#
#     A x = b,    A = -∂²  with Dirichlet BC,    on a 2^N grid in [0, π],
#
# and benchmarks `LinearSolverEngine` against an exact (manufactured)
# right-hand side.
#
# Why manufactured?
#   Pick u_target := qtt_sin and compute b := A · u_target exactly via
#   `exact_apply_mpo`.  Then A x = b has u_target as its exact solution
#   to machine precision, so all measured error comes from the solver.
#
# Run:
#     julia --project=.. examples/example_poisson_solver.jl

using QTTCore, MPSCore, LinearAlgebra, Printf

# ── Helper: ‖mps‖ via inner product ──────────────────────────────────
_norm(mps) = sqrt(abs(real(inner(mps, mps))))

# ── Helper: ‖a - b‖ via inner products (no MPS subtraction needed) ───
function _diff_norm(a, b)
    aa = real(inner(a, a))
    bb = real(inner(b, b))
    ab = real(inner(a, b))
    return sqrt(max(aa + bb - 2.0 * ab, 0.0))
end

# ─────────────────────────────────────────────────────────────────────
function run_one(N, max_dim, n_sweeps; num_center=2)
    @printf("\n========== N = %d  (%d grid points)  ==========\n", N, 2^N)

    # Interior-aligned grid so that Dirichlet ghost points sit exactly
    # where sin vanishes: h = π/(2^N+1), grid = [h, 2h, ..., (2^N)h].
    h  = π / (2^N + 1)
    x1 = h
    x2 = π - h

    # Operator A = -∂² with Dirichlet BC.
    A_mpo = (-1.0 * qtto_diff2(N, x1, x2; bc="dirichlet")).mpo

    # Manufactured target solution u_target = sin(x).
    u_target = qtt_sin(N, x1, x2)
    move_center!(u_target.mps, 1)
    u_norm = _norm(u_target.mps)

    # Manufactured RHS b := A · u_target  (exact, machine precision).
    b_mps = exact_apply_mpo(A_mpo, u_target.mps)
    move_center!(b_mps, 1)
    b_norm = _norm(b_mps)

    # Sanity check: how close is the manufactured b to the physical RHS sin(x)?
    sin_rhs = qtt_sin(N, x1, x2)
    move_center!(sin_rhs.mps, 1)
    b_vs_cont = _diff_norm(b_mps, sin_rhs.mps) / _norm(sin_rhs.mps)
    @printf("  ‖b_manuf - sin‖ / ‖sin‖     = %.3e   (discretisation residual; shrinks with N)\n",
            b_vs_cont)

    # Initial guess: random MPS, center at site 1.
    x_mps = random_mps(Float64, N, 2, 4; normalize=false, seed=7)
    move_center!(x_mps, 1)

    engine = LinearSolverEngine(x_mps, A_mpo, b_mps; krylovdim=100, tol=1e-14)

    @printf("  sweeps (num_center=%d, max_dim=%d):\n", num_center, max_dim)
    @printf("    %5s  %14s  %11s  %12s\n", "sweep", "max_local_res", "avg_trunc", "‖x-u‖/‖u‖")

    t0 = time()
    for s in 1:n_sweeps
        max_res, avg_trunc = sweep!(engine; max_dim=max_dim, cutoff=1e-14, num_center=num_center)
        err = _diff_norm(engine.x, u_target.mps) / u_norm
        @printf("    %5d  %14.3e  %11.3e  %12.3e\n", s, max_res, avg_trunc, err)
    end
    t_total = time() - t0

    # Final global residual ‖Ax - b‖ / ‖b‖.
    Ax_mps = exact_apply_mpo(A_mpo, engine.x)
    move_center!(Ax_mps, 1)
    final_res = _diff_norm(Ax_mps, b_mps) / b_norm
    final_err = _diff_norm(engine.x, u_target.mps) / u_norm

    @printf("  final ‖Ax - b‖ / ‖b‖           = %.3e\n", final_res)
    @printf("  final ‖x - u_target‖ / ‖u‖     = %.3e\n", final_err)
    @printf("  wall time                       = %.2f s\n", t_total)
end

# ── Run benchmark ────────────────────────────────────────────────────
for (N, max_dim, n_sweeps) in [(6, 10, 8), (8, 16, 8), (10, 24, 8), (12, 32, 8)]
    run_one(N, max_dim, n_sweeps; num_center=2)
end
