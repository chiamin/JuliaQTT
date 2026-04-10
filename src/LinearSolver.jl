# LinearSolver.jl
#
# LinearSolverEngine: DMRG-style ALS sweep solver for A x = b on MPS.
#
# Usage:
#     engine = LinearSolverEngine(x, A, b)
#     for _ in 1:nsweeps
#         max_res, avg_trunc = sweep!(engine; max_dim=50, cutoff=1e-10)
#     end
#
# `x` is stored in `engine.x` and modified in-place during sweeping.
# `x` must have center == 1 (right-canonical form) at construction and
# at the start of each sweep call.
#
# Local solver: KrylovKit.linsolve (GMRES / Krylov subspace method).

mutable struct LinearSolverEngine{T<:Number}
    x          :: MPS{T}
    A          :: MPO{T}
    b          :: MPS{T}
    op_env     :: OperatorEnv{T}
    b_env      :: VectorEnv{T}
    krylovdim  :: Int
    tol        :: Float64
end

"""
    LinearSolverEngine(x, A, b; krylovdim=30, tol=1e-10) -> LinearSolverEngine

Build the ALS linear solver engine for A x = b.

- `x` : solution MPS, modified in-place. Must have center == 1.
- `A` : MPO operator.
- `b` : right-hand side MPS.

All three inputs are promoted to a common element type.
"""
function LinearSolverEngine(x::MPS, A::MPO, b::MPS;
                              krylovdim::Int=30,
                              tol::Real=1e-10)
    # Promote to the common element type so x, A, b are all consistent
    # (e.g. Float64 + ComplexF64 → ComplexF64).
    T = promote_type(eltype(x), eltype(A), eltype(b))
    x_T = _to_mps_type(x, T)
    A_T = _to_mpo_type(A, T)
    b_T = _to_mps_type(b, T)

    center(x_T) == 1 ||
        throw(ArgumentError(
            "x.center must be 1 (right-canonical); got $(center(x_T))."))
    length(x_T) == length(b_T) ||
        throw(ArgumentError(
            "len(x) ($(length(x_T))) must equal len(b) ($(length(b_T)))."))

    op_env = OperatorEnv(x_T, x_T, A_T; init_center=1)
    b_env  = VectorEnv(b_T, x_T; init_center=1)

    return LinearSolverEngine{T}(x_T, A_T, b_T, op_env, b_env,
                                  krylovdim, Float64(tol))
end

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

"""
    sweep!(engine; max_dim=nothing, cutoff=0.0, num_center=2)
        -> (max_local_res, avg_trunc)

Perform one full sweep (right then left).

Returns:
- `max_local_res` : largest GMRES residual over all local solves.
- `avg_trunc`     : average truncation error over all SVD splits.
"""
function sweep!(engine::LinearSolverEngine{T};
                max_dim::Union{Int,Nothing}=nothing,
                cutoff::Real=0.0,
                num_center::Int=2) where {T}
    num_center in (1, 2) || error("num_center must be 1 or 2.")
    center(engine.x) == 1 ||
        error("x.center must be 1 at the start of each sweep; " *
              "got $(center(engine.x)).")

    N       = length(engine.x)
    n_center = num_center
    local_residuals = Float64[]
    truncs          = Float64[]

    # Sweep right: p = 1 .. N-1
    for p in 1:N-1
        res, tr = _local_update!(engine, p, n_center, max_dim, cutoff, "right")
        push!(local_residuals, res)
        push!(truncs, tr)
    end

    # Sweep left
    # 2-site: p = N-1 .. 1 ;  1-site: p = N .. 2
    left_start = N - n_center + 1
    left_stop  = n_center == 2 ? 1 : 2
    for p in left_start:-1:left_stop
        res, tr = _local_update!(engine, p, n_center, max_dim, cutoff, "left")
        push!(local_residuals, res)
        push!(truncs, tr)
    end

    max_res   = isempty(local_residuals) ? 0.0 : maximum(local_residuals)
    avg_trunc = isempty(truncs) ? 0.0 : sum(truncs) / length(truncs)
    return Float64(max_res), Float64(avg_trunc)
end

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

function _local_update!(engine::LinearSolverEngine{T}, p::Int, n_center::Int,
                         max_dim, cutoff, absorb::String) where {T}
    x = engine.x

    # Rebuild any stale left/right environments so that
    # op_env and b_env are valid just outside the window [p, p+n_center-1].
    update_envs!(engine.op_env, p, p + n_center - 1)
    update_envs!(engine.b_env,  p, p + n_center - 1)

    # Project A into the local subspace: A_eff = L_op * A[p..p+n-1] * R_op.
    Lop = getenv(engine.op_env, p - 1)
    Rop = getenv(engine.op_env, p + n_center)
    mpo_tensors = ntuple(k -> engine.A[p + k - 1], n_center)
    effA = EffOperator(Lop, Rop, mpo_tensors...)

    # Project b into the local subspace: b_eff = L_bv * b[p..p+n-1] * R_bv.
    Lbv = getenv(engine.b_env, p - 1)
    Rbv = getenv(engine.b_env, p + n_center)
    b_tensors = ntuple(k -> engine.b[p + k - 1], n_center)
    b_eff = EffVector(Lbv, Rbv, b_tensors...).tensor

    # Merge x[p..p+n-1] into a single tensor phi (used as the initial guess).
    phi = make_phi(x, p; n=n_center)

    # Solve the local linear system A_eff |phi_new⟩ = |b_eff⟩ via GMRES.
    # The tensors are flattened to vectors for the Krylov solver.
    f_apply = v -> vec(apply(effA, reshape(v, size(phi))))
    kd = min(engine.krylovdim, length(phi))
    phi_new_vec, info = linsolve(f_apply, vec(b_eff), vec(phi);
                                  krylovdim=kd, tol=engine.tol,
                                  verbosity=0)
    phi_new = reshape(phi_new_vec, size(phi))

    # Local residual: ||A_eff phi_new - b_eff|| / ||b_eff||.
    b_norm = norm(b_eff)
    local_res = b_norm > 0 ? norm(apply(effA, phi_new) .- b_eff) / b_norm : 0.0

    # SVD-split phi_new back into site tensors and write them into x.
    trunc = update_sites!(x, p, Array{T}(phi_new);
                           max_dim=max_dim, cutoff=cutoff, absorb=absorb)
    return Float64(local_res), Float64(trunc)
end
