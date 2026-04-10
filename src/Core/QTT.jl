# Core/QTT.jl
#
# QTT type: multi-dimensional function on a quantics (dyadic) grid.
#
# Tensor leg order: MPS site tensor (l, i, r)
# Site ordering:    site 1 = LSB, site N = MSB
# Site index:       1-indexed throughout

"""
    QTT(grids, ordering)

Multi-dimensional function stored as an MPS on a quantics grid.

- `grids`    : `Vector{GridInfo}`, one entry per spatial dimension.
- `ordering` : `"interleaved"` or `"sequential"`.
- `mps`      : the underlying `MPS` (set after construction by factory functions).

The MPS has `sum(g.num_bits for g in grids)` sites.
"""
mutable struct QTT
    grids::Vector{GridInfo}
    ordering::String
    mps::Union{MPS, Nothing}

    function QTT(grids::Vector{GridInfo}, ordering::String)
        ordering in ("interleaved", "sequential") ||
            throw(ArgumentError(
                "ordering must be \"interleaved\" or \"sequential\", got $(repr(ordering))."))
        isempty(grids) &&
            throw(ArgumentError("grids must not be empty."))
        new(grids, ordering, nothing)
    end
end

# Convenience constructor with default ordering.
QTT(grids::Vector{GridInfo}) = QTT(grids, "interleaved")

# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

"""Number of spatial dimensions."""
func_dim(q::QTT) = length(q.grids)

"""Total number of MPS sites."""
num_sites(q::QTT) = sum(g.num_bits for g in q.grids)

"""Number of grid points along each dimension."""
grid_points_per_dim(q::QTT) =
    Tuple(g.loc_dim ^ g.num_bits for g in q.grids)

"""Shape of the full grid (same as grid_points_per_dim)."""
grid_shape(q::QTT) = grid_points_per_dim(q)

"""Grid spacing along each dimension."""
grid_spacings(q::QTT) =
    Tuple((g.interval[2] - g.interval[1]) / (g.loc_dim ^ g.num_bits - 1)
          for g in q.grids)

# ---------------------------------------------------------------------------
# Index conversion
# ---------------------------------------------------------------------------

"""
    indices_to_bits(q, indices) -> Vector{Int}

Convert per-dimension grid indices to MPS site digits (0-based).
Length of result equals `num_sites(q)`.
"""
function indices_to_bits(q::QTT, indices)
    d = func_dim(q)
    length(indices) == d ||
        throw(ArgumentError("Expected $d indices, got $(length(indices))."))

    # Per-dim digits, least-significant first: digits_all[dim][bit_level]
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
# Evaluation
# ---------------------------------------------------------------------------

"""
    evaluate(q, indices...) -> scalar

Evaluate the QTT at the given per-dimension grid indices (0-based integers).
"""
function evaluate(q::QTT, indices::Integer...)
    q.mps === nothing && throw(ErrorException("No MPS set on this QTT."))
    bits = indices_to_bits(q, collect(indices))
    N = length(q.mps)
    result = nothing
    for k in 1:N
        A = q.mps[k]                  # Array{T,3}, shape (l, i, r)
        mat = A[:, bits[k]+1, :]      # bits are 0-based, Julia indices are 1-based
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

# QTT + QTT: delegates to qtt_sum (defined in Operations.jl)
Base.:(+)(a::QTT, b::QTT) = qtt_sum(a, b)

# QTT * QTT: element-wise product, delegates to qtt_prod (defined in Operations.jl)
function Base.:(*)(a::QTT, b::QTT)
    qtt_prod(a, b)
end

# QTT * scalar: scale the first site tensor
function Base.:(*)(q::QTT, s::Number)
    q.mps === nothing && throw(ErrorException("No MPS set on this QTT."))
    new_mps = copy(q.mps)
    new_mps[1] = new_mps[1] * s
    return QTT(copy(q.grids), q.ordering, new_mps)
end

Base.:(*)(s::Number, q::QTT) = q * s

# Private constructor used by Operations.jl and Init.jl to build a QTT
# with an already-constructed MPS.
function QTT(grids::Vector{GridInfo}, ordering::String, mps::Union{MPS, Nothing})
    q = QTT(grids, ordering)
    q.mps = mps
    return q
end

# ---------------------------------------------------------------------------
# MPS access
# ---------------------------------------------------------------------------

Base.length(q::QTT) = q.mps === nothing ? 0 : length(q.mps)

function Base.getindex(q::QTT, k::Int)
    q.mps === nothing && throw(ErrorException("No MPS set on this QTT."))
    return q.mps[k]
end
