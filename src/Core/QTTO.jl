# Core/QTTO.jl
#
# QTTO type: multi-dimensional operator on a quantics (dyadic) grid.
#
# Tensor leg order: MPO site tensor (l, ip, i, r)
# Site ordering:    site 1 = LSB, site N = MSB
# Site index:       1-indexed throughout

"""
    QTTO(grids, ordering)

Multi-dimensional operator stored as an MPO on a quantics grid.

- `grids`    : `Vector{GridInfo}`, one entry per spatial dimension.
- `ordering` : `"interleaved"` or `"sequential"`.
- `mpo`      : the underlying `MPO` (set after construction by factory functions).

The MPO has `sum(g.num_bits for g in grids)` sites.
"""
mutable struct QTTO
    grids::Vector{GridInfo}
    ordering::String
    mpo::Union{MPO, Nothing}

    function QTTO(grids::Vector{GridInfo}, ordering::String)
        ordering in ("interleaved", "sequential") ||
            throw(ArgumentError(
                "ordering must be \"interleaved\" or \"sequential\", got $(repr(ordering))."))
        isempty(grids) &&
            throw(ArgumentError("grids must not be empty."))
        new(grids, ordering, nothing)
    end
end

# Convenience constructor with default ordering.
QTTO(grids::Vector{GridInfo}) = QTTO(grids, "interleaved")

# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

"""Number of spatial dimensions."""
func_dim(q::QTTO) = length(q.grids)

"""Total number of MPO sites."""
num_sites(q::QTTO) = sum(g.num_bits for g in q.grids)

"""Number of grid points along each dimension."""
grid_points_per_dim(q::QTTO) =
    Tuple(g.loc_dim ^ g.num_bits for g in q.grids)

"""Shape of the full grid (same as grid_points_per_dim)."""
grid_shape(q::QTTO) = grid_points_per_dim(q)

"""Grid spacing along each dimension."""
grid_spacings(q::QTTO) =
    Tuple((g.interval[2] - g.interval[1]) / (g.loc_dim ^ g.num_bits - 1)
          for g in q.grids)

# ---------------------------------------------------------------------------
# Index conversion (same logic as QTT)
# ---------------------------------------------------------------------------

"""
    indices_to_bits(q, indices) -> Vector{Int}

Convert per-dimension grid indices to MPO site digits (0-based).
Length of result equals `num_sites(q)`.
"""
function indices_to_bits(q::QTTO, indices)
    d = func_dim(q)
    length(indices) == d ||
        throw(ArgumentError("Expected $d indices, got $(length(indices))."))

    digits_all = Vector{Vector{Int}}(undef, d)
    for dim in 1:d
        g = q.grids[dim]
        base = g.loc_dim
        digits_all[dim] = [(indices[dim] ÷ base^k) % base for k in 0:g.num_bits-1]
    end

    result = Int[]
    if q.ordering == "sequential"
        for dim in 1:d
            append!(result, digits_all[dim])
        end
    else  # interleaved
        max_bits = maximum(g.num_bits for g in q.grids)
        for k in 0:max_bits-1
            for dim in 1:d
                if k < q.grids[dim].num_bits
                    push!(result, digits_all[dim][k+1])
                end
            end
        end
    end
    return result
end

# ---------------------------------------------------------------------------
# Evaluation (diagonal)
# ---------------------------------------------------------------------------

"""
    evaluate(q::QTTO, indices...) -> scalar

Evaluate the diagonal of the operator at the given per-dimension grid indices.
Picks the diagonal element ip == i at every site and contracts.
"""
function evaluate(q::QTTO, indices::Integer...)
    q.mpo === nothing && throw(ErrorException("No MPO set on this QTTO."))
    bits = indices_to_bits(q, collect(indices))
    N = length(q.mpo)
    result = nothing
    for k in 1:N
        W = q.mpo[k]              # Array{T,4}, shape (l, ip, i, r)
        b = bits[k] + 1           # 0-based bits → 1-based Julia index
        mat = W[:, b, b, :]       # diagonal: ip == i == b
        result = result === nothing ? mat : result * mat
    end
    val = result[1, 1]
    if val isa Complex && iszero(imag(val))
        return real(val)
    end
    return val
end

# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

# QTTO * scalar
function Base.:(*)(q::QTTO, s::Number)
    q.mpo === nothing && throw(ErrorException("No MPO set on this QTTO."))
    new_mpo = copy(q.mpo)
    new_mpo[1] = new_mpo[1] * s
    result = QTTO(copy(q.grids), q.ordering)
    result.mpo = new_mpo
    return result
end

Base.:(*)(s::Number, q::QTTO) = q * s
Base.:(-)(q::QTTO) = q * (-1)

# ---------------------------------------------------------------------------
# MPO access
# ---------------------------------------------------------------------------

Base.length(q::QTTO) = q.mpo === nothing ? 0 : length(q.mpo)

function Base.getindex(q::QTTO, k::Int)
    q.mpo === nothing && throw(ErrorException("No MPO set on this QTTO."))
    return q.mpo[k]
end

# Private constructor used to build a QTTO with an already-constructed MPO.
function QTTO(grids::Vector{GridInfo}, ordering::String, mpo::Union{MPO, Nothing})
    q = QTTO(grids, ordering)
    q.mpo = mpo
    return q
end

# ---------------------------------------------------------------------------
# Conversion: QTT → diagonal QTTO
# ---------------------------------------------------------------------------

"""
    to_qtto(q::QTT) -> QTTO

Convert a QTT (function) to a diagonal QTTO (operator).

Each MPS site tensor `A[l, i, r]` becomes an MPO tensor
`W[l, ip, i, r] = A[l, i, r] * δ(ip, i)`, i.e. the physical index is
duplicated onto the bra side with only diagonal elements non-zero.
"""
function to_qtto(q::QTT)
    q.mps === nothing && throw(ErrorException("No MPS set on this QTT."))
    T = eltype(q.mps)
    N = length(q.mps)
    tensors = Vector{Array{T,4}}(undef, N)
    for k in 1:N
        A = q.mps[k]  # (l, i, r)
        dl, di, dr = size(A)
        W = zeros(T, dl, di, di, dr)
        for j in 1:di
            W[:, j, j, :] = A[:, j, :]
        end
        tensors[k] = W
    end
    result = QTTO(copy(q.grids), q.ordering)
    result.mpo = MPO(tensors)
    return result
end
