# Init.jl
#
# Exact QTT representations of elementary functions.
#
# Each function returns a QTT whose MPS encodes the function values on a
# dyadic grid of loc_dim^N points.  The default physical index is binary
# (loc_dim=2).  The grid index is the mixed-radix expansion:
#
#   n = sum_k i_k * loc_dim^k,   x(n) = x1 + (x2-x1)*n/(loc_dim^N - 1).
#
# All MPS tensors have leg order (l, i, r) with boundary bond dim = 1.

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _build_mps(bulk, L, R) -> MPS

Absorb boundary vectors L and R into bulk tensors and return an MPS.

- `bulk` : list of `Array{T,3}` of shape (chi, loc_dim, chi), one per site.
- `L`    : vector of length chi (left boundary).
- `R`    : vector of length chi (right boundary).
"""
function _build_mps(bulk::Vector{<:AbstractArray{<:Number,3}}, L, R)
    N = length(bulk)
    T = promote_type(eltype(L), eltype(R), eltype(bulk[1]))
    tensors = Vector{Array{T,3}}(undef, N)
    for k in 1:N
        tensors[k] = Array{T,3}(bulk[k])
    end

    if N == 1
        # t[i] = sum_{a,b} L[a] * A[a,i,b] * R[b]
        A = tensors[1]
        chi, loc_dim, _ = size(A)
        # Contract L: (1,chi) * (chi, loc_dim*chi) → (1, loc_dim*chi) → (loc_dim, chi)
        tmp = reshape(reshape(L, 1, chi) * reshape(A, chi, loc_dim*chi), loc_dim, chi)
        # Contract R: (loc_dim, chi) * (chi, 1) → (loc_dim, 1)
        t = tmp * reshape(R, chi, 1)
        tensors[1] = copy(reshape(t, 1, loc_dim, 1))
    else
        # Absorb L into first site: t[i,r] = sum_a L[a] * A[a,i,r]
        A = tensors[1]
        chi, loc_dim, r = size(A)
        t = reshape(reshape(L, 1, chi) * reshape(A, chi, loc_dim*r), 1, loc_dim, r)
        tensors[1] = copy(t)
        # Absorb R into last site: t[l,i] = sum_b A[l,i,b] * R[b]
        A = tensors[N]
        l, loc_dim, chi = size(A)
        t = reshape(reshape(A, l*loc_dim, chi) * reshape(R, chi, 1), l, loc_dim, 1)
        tensors[N] = copy(t)
    end

    return MPS(tensors)
end

"""
    _rotation_bulk(N, s) -> Vector{Array{Float64,3}}

Build N bulk tensors of shape (2, 2, 2) using rotation matrices.

Shared by `qtt_sin` and `qtt_cos` (they differ only in the boundary L).
For physical index i_k = 0 the transfer matrix is identity;
for i_k = 1 it is a rotation by θ_k = s * 2^k.
"""
function _rotation_bulk(N::Int, s::Float64)
    bulk = Vector{Array{Float64,3}}(undef, N)
    for k in 0:N-1
        theta = s * 2.0^k
        c, sn = cos(theta), sin(theta)
        A = zeros(2, 2, 2)
        # i=1 (bit=0): identity
        A[1, 1, 1] = 1.0
        A[2, 1, 2] = 1.0
        # i=2 (bit=1): rotation
        A[1, 2, 1] = c
        A[1, 2, 2] = -sn
        A[2, 2, 1] = sn
        A[2, 2, 2] = c
        bulk[k+1] = A
    end
    return bulk
end

"""
    _grid_rescale(N, x1, x2) -> Float64

Return the grid spacing s = (x2 - x1) / (2^N - 1).
"""
function _grid_rescale(N::Int, x1::Real, x2::Real)
    N >= 1 || throw(ArgumentError("N must be >= 1."))
    return (x2 - x1) / (2.0^N - 1)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    qtt_sin(N, x1, x2) -> QTT

Exact QTT for sin(x) on [x1, x2] with 2^N grid points.
Bond dimension is 2 (or 1 at the boundaries).
"""
function qtt_sin(N::Int, x1::Real, x2::Real)
    s = _grid_rescale(N, x1, x2)
    bulk = _rotation_bulk(N, s)
    L = [sin(x1), cos(x1)]
    R = [1.0, 0.0]
    mps = _build_mps(bulk, L, R)
    q = QTT([GridInfo(N, (Float64(x1), Float64(x2)))], "sequential")
    q.mps = mps
    return q
end

"""
    qtt_cos(N, x1, x2) -> QTT

Exact QTT for cos(x) on [x1, x2] with 2^N grid points.
Bond dimension is 2 (or 1 at the boundaries).
"""
function qtt_cos(N::Int, x1::Real, x2::Real)
    s = _grid_rescale(N, x1, x2)
    bulk = _rotation_bulk(N, s)
    L = [cos(x1), -sin(x1)]
    R = [1.0, 0.0]
    mps = _build_mps(bulk, L, R)
    q = QTT([GridInfo(N, (Float64(x1), Float64(x2)))], "sequential")
    q.mps = mps
    return q
end

"""
    qtt_linear(N, x1, x2; a=1.0, b=0.0) -> QTT

Exact QTT for a*x + b on [x1, x2] with 2^N grid points.
Bond dimension is 2 (or 1 at the boundaries).
"""
function qtt_linear(N::Int, x1::Real, x2::Real; a::Real=1.0, b::Real=0.0)
    s = _grid_rescale(N, x1, x2)
    bulk = Vector{Array{Float64,3}}(undef, N)
    for k in 0:N-1
        theta = s * 2.0^k
        A = zeros(2, 2, 2)
        # i=1 (bit=0): identity
        A[1, 1, 1] = 1.0
        A[2, 1, 2] = 1.0
        # i=2 (bit=1): accumulate a * theta_k
        A[1, 2, 1] = 1.0
        A[2, 2, 1] = a * theta
        A[2, 2, 2] = 1.0
        bulk[k+1] = A
    end
    L = [a * x1 + b, 1.0]
    R = [1.0, 0.0]
    mps = _build_mps(bulk, L, R)
    q = QTT([GridInfo(N, (Float64(x1), Float64(x2)))], "sequential")
    q.mps = mps
    return q
end

"""
    qtt_exp(N, x1, x2; a=1.0, b=0.0) -> QTT

Exact QTT for exp(a*x + b) on [x1, x2] with 2^N grid points.
Bond dimension is 1 (product state).
"""
function qtt_exp(N::Int, x1::Real, x2::Real; a::Real=1.0, b::Real=0.0)
    s = _grid_rescale(N, x1, x2)
    prefactor = exp(a * x1 + b)
    tensors = Vector{Array{Float64,3}}(undef, N)
    for k in 0:N-1
        theta = s * 2.0^k
        arr = zeros(1, 2, 1)
        arr[1, 1, 1] = 1.0
        arr[1, 2, 1] = exp(a * theta)
        tensors[k+1] = arr
    end
    tensors[1] = tensors[1] .* prefactor
    q = QTT([GridInfo(N, (Float64(x1), Float64(x2)))], "sequential")
    q.mps = MPS(tensors)
    return q
end

"""
    qtt_random(grids, bond_dim; ordering="interleaved") -> QTT

Create a QTT with random MPS tensors.
"""
function qtt_random(grids::Vector{GridInfo}, bond_dim::Int;
                    ordering::String="interleaved")
    q = QTT(grids, ordering)
    n_sites = num_sites(q)
    loc_dim = grids[1].loc_dim
    tensors = Vector{Array{Float64,3}}(undef, n_sites)
    for k in 1:n_sites
        dl = k == 1 ? 1 : bond_dim
        dr = k == n_sites ? 1 : bond_dim
        tensors[k] = randn(dl, loc_dim, dr)
    end
    q.mps = MPS(tensors)
    return q
end
