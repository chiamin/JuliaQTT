# Operators.jl
#
# Exact QTTO representations of finite-difference operators.
#
# Grid convention: n = sum_k i_k * 2^k,  x(n) = x1 + (x2-x1)*n/(2^N-1).
# MPO tensor leg order: (l, ip, i, r)  with ip=bra(output), i=ket(input).

# ---------------------------------------------------------------------------
# Internal constants and helpers
# ---------------------------------------------------------------------------

const _I2  = [1.0 0.0; 0.0 1.0]
const _sp2 = [0.0 1.0; 0.0 0.0]   # |0><1|: maps |1⟩ → |0⟩ (in ket basis)
const _sm2 = [0.0 0.0; 1.0 0.0]   # |1><0|: maps |0⟩ → |1⟩

"""
    _build_mpo(bulk, L, R) -> MPO

Absorb boundary vectors L and R into bulk MPO tensors and return an MPO.

- `bulk` : list of `Array{T,4}` of shape (chi, loc_dim, loc_dim, chi), one per site.
  Leg order: (l, ip, i, r).
- `L`    : vector of length chi (left boundary).
- `R`    : vector of length chi (right boundary).
"""
function _build_mpo(bulk::Vector{<:AbstractArray{<:Number,4}}, L, R)
    N = length(bulk)
    T = promote_type(eltype(L), eltype(R), eltype(bulk[1]))
    tensors = Vector{Array{T,4}}(undef, N)
    for k in 1:N
        tensors[k] = Array{T,4}(bulk[k])
    end

    if N == 1
        # t[ip, i] = sum_{a,b} L[a] * A[a,ip,i,b] * R[b]
        A = tensors[1]
        chi, ip_dim, i_dim, _ = size(A)
        # Contract L: (1,chi) * (chi, ip*i*chi) → (1, ip*i*chi) → (ip*i, chi)
        tmp = reshape(reshape(L, 1, chi) * reshape(A, chi, ip_dim*i_dim*chi),
                      ip_dim*i_dim, chi)
        # Contract R: (ip*i, chi) * (chi, 1) → (ip*i, 1) → (1, ip, i, 1)
        t = reshape(tmp * reshape(R, chi, 1), 1, ip_dim, i_dim, 1)
        tensors[1] = copy(t)
    else
        # Absorb L into first site: t[ip,i,r] = sum_a L[a] * A[a,ip,i,r]
        A = tensors[1]
        chi, ip_dim, i_dim, r = size(A)
        t = reshape(reshape(L, 1, chi) * reshape(A, chi, ip_dim*i_dim*r),
                    1, ip_dim, i_dim, r)
        tensors[1] = copy(t)
        # Absorb R into last site: t[l,ip,i] = sum_b A[l,ip,i,b] * R[b]
        A = tensors[N]
        l, ip_dim, i_dim, chi = size(A)
        t = reshape(reshape(A, l*ip_dim*i_dim, chi) * reshape(R, chi, 1),
                    l, ip_dim, i_dim, 1)
        tensors[N] = copy(t)
    end

    return MPO(tensors)
end

"""Return the bulk tensor for S⁺ (forward shift / carry propagation).
Shape: (2, 2, 2, 2) = (l, ip, i, r)."""
function _shift_forward_bulk()
    A = zeros(2, 2, 2, 2)
    A[1, :, :, 1] = _I2
    A[2, :, :, 1] = _sp2
    A[2, :, :, 2] = _sm2
    return A
end

"""Return the bulk tensor for S⁻ (backward shift / borrow propagation)."""
function _shift_backward_bulk()
    A = zeros(2, 2, 2, 2)
    A[1, :, :, 1] = _I2
    A[2, :, :, 1] = _sm2
    A[2, :, :, 2] = _sp2
    return A
end

"""Right boundary vector for a single shift channel.

- `"periodic"` : [1, 1] — wraps 2^N-1 → 0.
- `"dirichlet"`: [1, 0] — zero-pads outside grid.
"""
function _shift_R(bc::String)
    bc == "periodic"  && return [1.0, 1.0]
    bc == "dirichlet" && return [1.0, 0.0]
    throw(ArgumentError("bc must be \"periodic\" or \"dirichlet\", got $(repr(bc))."))
end

function _grid_spacing(N::Int, x1::Real, x2::Real)
    N >= 1 || throw(ArgumentError("N must be >= 1."))
    return (x2 - x1) / (2.0^N - 1)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    shift_forward_mpo(N; bc="periodic") -> MPO

S⁺: binary increment n → n+1.  Bond dimension 2 in the bulk.

- `bc = "periodic"` : wrap-around S⁺|2^N-1⟩ = |0⟩.
- `bc = "dirichlet"`: zero-padded S⁺|2^N-1⟩ = 0.
"""
function shift_forward_mpo(N::Int; bc::String="periodic")
    bulk = [_shift_forward_bulk() for _ in 1:N]
    L = [0.0, 1.0]
    R = _shift_R(bc)
    return _build_mpo(bulk, L, R)
end

"""
    shift_backward_mpo(N; bc="periodic") -> MPO

S⁻: binary decrement n → n-1.  Bond dimension 2 in the bulk.
"""
function shift_backward_mpo(N::Int; bc::String="periodic")
    bulk = [_shift_backward_bulk() for _ in 1:N]
    L = [0.0, 1.0]
    R = _shift_R(bc)
    return _build_mpo(bulk, L, R)
end

"""
    qtto_diff_forward(N, x1, x2; bc="periodic") -> QTTO

Forward difference operator (S⁺ - I) / dx as a QTTO.
Bond dimension is 2. 1D, sequential ordering.
"""
function qtto_diff_forward(N::Int, x1::Real, x2::Real; bc::String="periodic")
    dx = _grid_spacing(N, x1, x2)
    bulk = [_shift_forward_bulk() for _ in 1:N]
    L = [-1.0, 1.0] ./ dx
    R = [1.0, _shift_R(bc)[2]]
    mpo = _build_mpo(bulk, L, R)
    q = QTTO([GridInfo(N, (Float64(x1), Float64(x2)))], "sequential")
    q.mpo = mpo
    return q
end

"""
    qtto_diff_backward(N, x1, x2; bc="periodic") -> QTTO

Backward difference operator (I - S⁻) / dx as a QTTO.
Bond dimension is 2. 1D, sequential ordering.
"""
function qtto_diff_backward(N::Int, x1::Real, x2::Real; bc::String="periodic")
    dx = _grid_spacing(N, x1, x2)
    bulk = [_shift_backward_bulk() for _ in 1:N]
    L = [1.0, -1.0] ./ dx
    R = [1.0, _shift_R(bc)[2]]
    mpo = _build_mpo(bulk, L, R)
    q = QTTO([GridInfo(N, (Float64(x1), Float64(x2)))], "sequential")
    q.mpo = mpo
    return q
end

"""
    qtto_diff2(N, x1, x2; bc="periodic") -> QTTO

Second-order difference operator (S⁺ - 2I + S⁻) / dx² as a QTTO.
Bond dimension is 3. 1D, sequential ordering.
"""
function qtto_diff2(N::Int, x1::Real, x2::Real; bc::String="periodic")
    dx = _grid_spacing(N, x1, x2)

    A = zeros(3, 2, 2, 3)  # (l, ip, i, r)
    A[1, :, :, 1] = _I2
    A[2, :, :, 1] = _sp2
    A[3, :, :, 1] = _sm2
    A[2, :, :, 2] = _sm2
    A[3, :, :, 3] = _sp2

    bulk = [A for _ in 1:N]
    r_shift = _shift_R(bc)[2]
    L = [-2.0, 1.0, 1.0] ./ dx^2
    R = [1.0, r_shift, r_shift]
    mpo = _build_mpo(bulk, L, R)
    q = QTTO([GridInfo(N, (Float64(x1), Float64(x2)))], "sequential")
    q.mpo = mpo
    return q
end
