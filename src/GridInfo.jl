# GridInfo.jl
#
# Per-dimension grid metadata for QTT / QTTO.

"""
    GridInfo(num_bits, interval[, loc_dim])

Metadata for one dimension of a quantics (dyadic) grid.

- `num_bits`  : number of MPS/MPO sites for this dimension;
                the dimension has `loc_dim^num_bits` grid points.
- `interval`  : `(x_min, x_max)` range.
- `loc_dim`   : local (physical) dimension at each site (default 2).
"""
struct GridInfo
    num_bits::Int
    interval::Tuple{Float64, Float64}
    loc_dim::Int
end

GridInfo(num_bits::Int, interval::Tuple{Real,Real}) =
    GridInfo(num_bits, Float64.(interval), 2)

GridInfo(num_bits::Int, interval::Tuple{Real,Real}, loc_dim::Int) =
    GridInfo(num_bits, Float64.(interval), loc_dim)
