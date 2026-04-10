# Operations.jl
#
# QTT operations: embedding, element-wise arithmetic, and interpolation.

# ---------------------------------------------------------------------------
# Internal type helpers
# ---------------------------------------------------------------------------

"""Convert an MPS to a given element type.  Returns the same object if already correct."""
function _to_mps_type(psi::MPS, ::Type{T}) where T
    T == eltype(psi) && return psi
    return MPS([Array{T,3}(psi[k]) for k in 1:length(psi)])
end

"""Convert an MPO to a given element type.  Returns the same object if already correct."""
function _to_mpo_type(mpo::MPO, ::Type{T}) where T
    T == eltype(mpo) && return mpo
    return MPO([Array{T,4}(mpo[k]) for k in 1:length(mpo)])
end

"""mps_sum with automatic type promotion."""
function _mps_sum(a::MPS, b::MPS)
    T = promote_type(eltype(a), eltype(b))
    return mps_sum(_to_mps_type(a, T), _to_mps_type(b, T))
end

# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

function _check_compatible(a::QTT, b::QTT)
    a.ordering == b.ordering ||
        throw(ArgumentError("Ordering mismatch: $(repr(a.ordering)) vs $(repr(b.ordering))."))
    length(a.grids) == length(b.grids) ||
        throw(ArgumentError(
            "Number of dimensions mismatch: $(length(a.grids)) vs $(length(b.grids))."))
    for (d, (ga, gb)) in enumerate(zip(a.grids, b.grids))
        ga == gb ||
            throw(ArgumentError("Dimension $d: grid mismatch — $ga vs $gb."))
    end
end

# ---------------------------------------------------------------------------
# Element-wise arithmetic
# ---------------------------------------------------------------------------

"""
    qtt_sum(a, b) -> QTT

Element-wise sum of two compatible QTTs.
"""
function qtt_sum(a::QTT, b::QTT)
    _check_compatible(a, b)
    result = QTT(copy(a.grids), a.ordering)
    result.mps = _mps_sum(a.mps, b.mps)
    return result
end

"""
    qtt_prod(a, b) -> QTT

Element-wise (Hadamard) product of two compatible QTTs.
Converts `a` to a diagonal QTTO and applies it to `b`.
"""
function qtt_prod(a::QTT, b::QTT)
    _check_compatible(a, b)
    qtto_a = to_qtto(a)
    T = promote_type(eltype(qtto_a.mpo), eltype(b.mps))
    mpo = _to_mpo_type(qtto_a.mpo, T)
    mps = _to_mps_type(b.mps, T)
    result = QTT(copy(a.grids), a.ordering)
    result.mps = exact_apply_mpo(mpo, mps)
    return result
end

# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

"""
    embed(grids, qtts_1d; ordering="interleaved") -> QTT

Embed one or more 1D QTTs into a multi-dimensional QTT.

Each element of `qtts_1d` corresponds to a dimension in `grids`.
A 1D QTT at position `d` contributes its site tensors for dimension `d`;
a `nothing` entry inserts identity sites (constant 1) for that dimension.

When multiple 1D QTTs are given, their site tensors are combined via
Kronecker product on the virtual bonds, producing the tensor product of
the individual functions.
"""
function embed(grids::Vector{GridInfo},
               qtts_1d::Vector,
               ordering::String="interleaved")
    ordering in ("interleaved", "sequential") ||
        throw(ArgumentError(
            "ordering must be \"interleaved\" or \"sequential\", got $(repr(ordering))."))
    ndim = length(grids)
    length(qtts_1d) == ndim ||
        throw(ArgumentError(
            "Length of qtts_1d ($(length(qtts_1d))) must match length of grids ($ndim)."))

    # Validate and extract per-dimension site tensors.
    # sites_per_dim[d] is a list of Array{T,3}, one per bit.
    T = Float64
    for d in 1:ndim
        q = qtts_1d[d]
        q === nothing && continue
        if func_dim(q) != 1
            throw(ArgumentError(
                "Expected a 1D QTT for dimension $d, got func_dim=$(func_dim(q))."))
        end
        q.mps === nothing &&
            throw(ErrorException("1D QTT for dimension $d has no MPS set."))
        T = promote_type(T, eltype(q.mps))
    end

    sites_per_dim = Vector{Vector{Array{T,3}}}(undef, ndim)
    for d in 1:ndim
        g = grids[d]
        q = qtts_1d[d]
        if q === nothing
            # Identity site: all physical values equal 1 → constant function 1
            sites_per_dim[d] = [ones(T, 1, g.loc_dim, 1) for _ in 1:g.num_bits]
        else
            # Already validated func_dim == 1 and mps !== nothing above.
            q_grid = q.grids[1]
            q_grid.num_bits == g.num_bits ||
                throw(ArgumentError(
                    "Dimension $d: num_bits mismatch — 1D QTT has $(q_grid.num_bits), " *
                    "grid expects $(g.num_bits)."))
            q_grid.interval == g.interval ||
                throw(ArgumentError(
                    "Dimension $d: interval mismatch — 1D QTT has $(q_grid.interval), " *
                    "grid expects $(g.interval)."))
            q_grid.loc_dim == g.loc_dim ||
                throw(ArgumentError(
                    "Dimension $d: loc_dim mismatch — 1D QTT has $(q_grid.loc_dim), " *
                    "grid expects $(g.loc_dim)."))
            sites_per_dim[d] = [Array{T,3}(q.mps[k]) for k in 1:length(q.mps)]
        end
    end

    # Build MPS site tensors according to the ordering.
    tensors = Array{T,3}[]

    if ordering == "sequential"
        # Simply concatenate per-dimension tensors in order.
        for d in 1:ndim
            append!(tensors, sites_per_dim[d])
        end
    else
        # Interleaved: at each (bit level k, dimension d), build a site tensor
        # whose virtual bond is the Kronecker product of all per-dimension bonds.
        #
        # For dimension d at level k, the combined site tensor C satisfies:
        #   C[:, idx, :] = kron(M_0, ..., M_{D-1})
        # where M_d = A_d[:, idx, :] and M_j = I_{current_dim_j} for j ≠ d.
        current_dims = [size(sites_per_dim[d][1], 1) for d in 1:ndim]
        max_bits = maximum(g.num_bits for g in grids)

        for k in 0:max_bits-1
            for d in 1:ndim
                k >= grids[d].num_bits && continue
                A = sites_per_dim[d][k+1]   # (l_d, loc_dim_d, r_d)
                l_d, loc_dim, r_d = size(A)
                l_tot = prod(current_dims)
                new_dims = copy(current_dims)
                new_dims[d] = r_d
                r_tot = prod(new_dims)

                C = zeros(T, l_tot, loc_dim, r_tot)
                for idx in 1:loc_dim
                    mats = [Matrix{T}(I, current_dims[j], current_dims[j])
                            for j in 1:ndim]
                    mats[d] = A[:, idx, :]  # l_d × r_d
                    C[:, idx, :] = reduce(kron, mats)
                end
                push!(tensors, C)
                current_dims[d] = r_d
            end
        end
    end

    result = QTT(copy(grids), ordering)
    result.mps = MPS(Vector{Array{T,3}}(tensors))
    return result
end

# ---------------------------------------------------------------------------
# Interpolation and coarse-graining
# ---------------------------------------------------------------------------

"""
    qtt_interp0(q) -> QTT

Zero-order (forward copy) interpolation: refine a 1D QTT from 2^N to 2^(N+1).

Inserts a midpoint between every pair of adjacent grid points by copying the
left neighbour's value.  Prepends a (1, 2, 1) identity site at the LSB.
"""
function qtt_interp0(q::QTT)
    q.mps === nothing && throw(ErrorException("No MPS set on this QTT."))
    func_dim(q) == 1 || throw(ArgumentError("qtt_interp0 only supports 1D QTTs."))

    N = q.grids[1].num_bits
    x1, x2 = q.grids[1].interval
    T = eltype(q.mps)

    # Selector site: both j_0=0 and j_0=1 give 1 → value is f(n) regardless.
    id_site = ones(T, 1, q.grids[1].loc_dim, 1)

    new_tensors = Vector{Array{T,3}}(undef, N+1)
    new_tensors[1] = id_site
    for k in 1:N
        new_tensors[k+1] = copy(q.mps[k])
    end

    new_grid = GridInfo(N+1, (x1, x2))
    result = QTT([new_grid], "sequential")
    result.mps = MPS(new_tensors)
    return result
end

"""
    qtt_coarsen(q) -> QTT

Coarse-grain a 1D QTT from 2^N to 2^(N-1) by averaging.

Contracts the LSB site with the vector [1, 1] / 2, absorbs into site 2,
and returns an (N-1)-site QTT.  The new value g(n) = (f(2n) + f(2n+1)) / 2.
"""
function qtt_coarsen(q::QTT)
    q.mps === nothing && throw(ErrorException("No MPS set on this QTT."))
    func_dim(q) == 1 || throw(ArgumentError("qtt_coarsen only supports 1D QTTs."))

    N = q.grids[1].num_bits
    N >= 2 || throw(ArgumentError("Need at least 2 sites to coarse-grain."))

    x1, x2 = q.grids[1].interval
    T = eltype(q.mps)

    # Contract site 1 physical index with averaging vector [1,1]/2.
    # contracted[l0, r0] = sum_i avg[i] * A0[l0, i, r0]
    A0 = q.mps[1]        # (l0, loc_dim, r0)
    avg = fill(T(0.5), q.grids[1].loc_dim)   # (loc_dim,)
    l0, ld, r0 = size(A0)
    # Permute to (l0, r0, ld) → reshape to (l0*r0, ld) → matmul with avg → (l0*r0,)
    contracted_vec = reshape(permutedims(A0, (1, 3, 2)), l0*r0, ld) * avg
    contracted_mat = reshape(contracted_vec, l0, r0)   # (l0, r0)

    # Absorb into site 2: new_site2[l0, i1, r1] = sum_r0 contracted_mat[l0, r0] * A1[r0, i1, r1]
    A1 = q.mps[2]   # (r0, loc_dim, r1)
    new_site2 = reshape(contracted_mat * reshape(A1, r0, :), l0, size(A1,2), size(A1,3))

    new_tensors = Vector{Array{T,3}}(undef, N-1)
    new_tensors[1] = copy(new_site2)
    for k in 3:N
        new_tensors[k-1] = copy(q.mps[k])
    end

    new_grid = GridInfo(N-1, (x1, x2))
    result = QTT([new_grid], "sequential")
    result.mps = MPS(new_tensors)
    return result
end

"""
    qtt_interp(q; bc=0.0) -> QTT

Linear interpolation: refine a 1D QTT from 2^N to 2^(N+1) grid points.

Inserts a midpoint between every pair of adjacent grid points.  The new QTT
has N+1 sites on the same interval [x1, x2].

- At even indices (j_0=0): g(2n)   = f(n)
- At odd  indices (j_0=1): g(2n+1) = (f(n) + f(n+1)) / 2

`bc` supplies the boundary value f(2^N), which does not exist on the
original grid (default: 0.0).
"""
function qtt_interp(q::QTT; bc::Real=0.0)
    q.mps === nothing && throw(ErrorException("No MPS set on this QTT."))
    func_dim(q) == 1 || throw(ArgumentError("qtt_interp only supports 1D QTTs."))

    N = q.grids[1].num_bits
    x1, x2 = q.grids[1].interval
    T = eltype(q.mps)

    # Step 1: f' = S⁺ · f  (periodic shift wraps, corrected below if bc != 0)
    s_plus = shift_forward_mpo(N)
    f_shifted = exact_apply_mpo(_to_mpo_type(s_plus, T), q.mps)

    # BC correction: S⁺ maps index 2^N-1 → 0 (mod 2^N), giving f(0) there.
    # We want f(2^N) = bc there instead.  Correction: (bc - f(0)) at index 2^N-1.
    # Index 2^N-1 means all bits = 1 (1-indexed: physical index 2 at each site).
    if !iszero(bc)
        f0 = evaluate(q, 0)
        correction_val = T(bc - f0)
        if !iszero(correction_val)
            # Build rank-1 MPS for correction_val * |11...1⟩
            corr_tensors = Vector{Array{T,3}}(undef, N)
            for k in 1:N
                ct = zeros(T, 1, 2, 1)
                ct[1, 2, 1] = one(T)   # select bit = 1 (1-indexed)
                corr_tensors[k] = ct
            end
            corr_tensors[1] = corr_tensors[1] .* correction_val
            corr_mps = MPS(corr_tensors)
            f_shifted = _mps_sum(f_shifted, corr_mps)
        end
    end

    # Step 2: h = (f + f') / 2
    h_sum = _mps_sum(q.mps, f_shifted)
    # Scale first site by 0.5 (h = 0.5 * (f + f_shifted))
    h_tensors = [copy(Array{T,3}(h_sum[k])) for k in 1:N]
    h_tensors[1] .*= T(0.5)
    h_mps = MPS(h_tensors)

    f_mps = _to_mps_type(q.mps, T)

    # Step 3: Build (N+1)-site MPS with selector site at LSB.
    # Site 1 (selector, physical index j_0):
    #   j_0=1 (bit=0): activate f path (right block 1)
    #   j_0=2 (bit=1): activate h path (right block 2)
    # Right bond dimension = 2.
    B = zeros(T, 1, 2, 2)
    B[1, 1, 1] = one(T)   # j_0=0 → f path (block index 1)
    B[1, 2, 2] = one(T)   # j_0=1 → h path (block index 2)

    new_tensors = Vector{Array{T,3}}(undef, N+1)
    new_tensors[1] = B

    # Sites 2..N+1: block-diagonal of f and h via direct_sum on [1,3].
    for k in 1:N
        fk = Array{T,3}(f_mps[k])
        hk = Array{T,3}(h_mps[k])
        new_tensors[k+1] = direct_sum(fk, hk, [1, 3])
    end

    # The last site (k=N) after direct_sum has right bond = r_f + r_h = 1 + 1 = 2.
    # Contract with R = [1, 1] to collapse to bond dim 1.
    last = new_tensors[N+1]   # (l, loc_dim, 2)
    l, ld, r2 = size(last)
    R_vec = ones(T, r2)
    contracted = copy(reshape(reshape(last, l*ld, r2) * R_vec, l, ld, 1))
    new_tensors[N+1] = contracted

    new_grid = GridInfo(N+1, (x1, x2))
    result = QTT([new_grid], "sequential")
    result.mps = MPS(new_tensors)
    return result
end
