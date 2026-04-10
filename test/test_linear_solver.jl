@testset "LinearSolver" begin

    # Build a simple diagonal MPO: H = diag(1, 2, 3, ..., 2^N)
    # For testing we use a known system where A * x = b has an exact solution.

    # Utility: random MPS with center at site 1 (right-canonical)
    function make_random_mps(N, d, chi; seed=nothing)
        seed !== nothing && Random.seed!(seed)
        tensors = Vector{Array{Float64,3}}(undef, N)
        for k in 1:N
            dl = k == 1 ? 1 : chi
            dr = k == N ? 1 : chi
            tensors[k] = randn(dl, d, dr)
        end
        psi = MPS(tensors)
        move_center!(psi, 1)
        return psi
    end

    # Build a simple MPO: A = scale * I.
    # Puts scale at site 1 and identity at all other sites so that the full
    # MPO matrix is scale * I_{d^N × d^N} (not scale^N * I).
    function make_identity_mpo(N, d, scale=2.0)
        tensors = Vector{Array{Float64,4}}(undef, N)
        for k in 1:N
            s = k == 1 ? scale : 1.0
            W = zeros(1, d, d, 1)
            for i in 1:d
                W[1, i, i, 1] = s
            end
            tensors[k] = W
        end
        return MPO(tensors)
    end

    # ----------------------------------------------------------------
    # Construction validation
    # ----------------------------------------------------------------
    @testset "x center must be 1" begin
        N, d = 4, 2
        x = make_random_mps(N, d, 4; seed=1)
        move_center!(x, 2)
        A = make_identity_mpo(N, d)
        b = make_random_mps(N, d, 4; seed=2)
        @test_throws ArgumentError LinearSolverEngine(x, A, b)
    end

    @testset "length mismatch" begin
        N, d = 4, 2
        x = make_random_mps(N+1, d, 4; seed=1)
        A = make_identity_mpo(N+1, d)
        b = make_random_mps(N, d, 4; seed=2)
        move_center!(b, 1)
        @test_throws ArgumentError LinearSolverEngine(x, A, b)
    end

    # ----------------------------------------------------------------
    # Two-site sweep convergence: A * x = b with A = scale * I
    # Exact solution: x = b / scale
    # ----------------------------------------------------------------
    @testset "two-site sweep converges for diagonal A" begin
        using Random
        N, d, chi, scale = 4, 2, 8, 3.0
        Random.seed!(42)
        b = make_random_mps(N, d, chi; seed=11)
        x = make_random_mps(N, d, chi; seed=22)
        A = make_identity_mpo(N, d, scale)

        engine = LinearSolverEngine(x, A, b; krylovdim=50, tol=1e-12)
        for _ in 1:8
            sweep!(engine; max_dim=16, cutoff=1e-14, num_center=2)
        end

        # Check A * x ≈ b by computing inner products
        # <b|b> / scale should equal <b|x>
        # Use inner(x, b) ≈ inner(b, b) / scale
        x_sol = engine.x
        # Verify residual: ||A x - b|| / ||b|| should be small
        # Build full state vectors
        function mps_to_vec(psi)
            N_ = length(psi)
            d_ = size(psi[1], 2)
            v = ones(1, 1)
            for k in 1:N_
                A_ = psi[k]
                l, p, r = size(A_)
                v = reshape(v, size(v, 1), l)
                v = reshape(sum(A_[a, :, :] * v[b, a] for a in 1:l, b in 1:size(v,1)), :, r)
            end
            return vec(v)
        end
        function mps_to_vec2(psi)
            v = reshape(psi[1], :, size(psi[1], 3))
            for k in 2:length(psi)
                A_ = psi[k]
                l, p, r = size(A_)
                v = v * reshape(permutedims(A_, (1,2,3)), l, p*r)
                v = reshape(v, :, r)
            end
            return vec(v)
        end
        # Simple direct evaluation approach
        function eval_mps(psi)
            N_ = length(psi)
            d_ = size(psi[1], 2)
            n_total = d_^N_
            result = zeros(eltype(psi[1]), n_total)
            for idx in 0:n_total-1
                digits = [(idx >> ((k-1)*1)) & 1 for k in N_:-1:1]  # MSB first
                val = nothing
                for k in 1:N_
                    # site k, digit = digits[k] (MSB ordering, but QTT uses LSB)
                    # Actually let me use LSB ordering consistent with QTT
                    bit_k = (idx >> (k-1)) & 1  # bit for site k (LSB at site 1)
                    mat = psi[k][:, bit_k+1, :]
                    val = val === nothing ? mat : val * mat
                end
                result[idx+1] = val[1,1]
            end
            return result
        end
        b_vec = eval_mps(b)
        x_vec = eval_mps(x_sol)
        # A x ≈ scale * x, so residual = ||scale * x_vec - b_vec|| / ||b_vec||
        res = norm(scale .* x_vec .- b_vec) / norm(b_vec)
        @test res < 1e-6
    end

    # ----------------------------------------------------------------
    # One-site sweep convergence
    # ----------------------------------------------------------------
    @testset "one-site sweep converges for diagonal A" begin
        using Random
        N, d, chi, scale = 4, 2, 8, 3.0
        b = make_random_mps(N, d, chi; seed=11)
        x = make_random_mps(N, d, chi; seed=22)
        A = make_identity_mpo(N, d, scale)

        engine = LinearSolverEngine(x, A, b; krylovdim=50, tol=1e-12)
        for _ in 1:10
            sweep!(engine; max_dim=chi, cutoff=0.0, num_center=1)
        end

        function eval_mps(psi)
            N_ = length(psi)
            d_ = size(psi[1], 2)
            n_total = d_^N_
            result = zeros(eltype(psi[1]), n_total)
            for idx in 0:n_total-1
                val = nothing
                for k in 1:N_
                    bit_k = (idx >> (k-1)) & 1
                    mat = psi[k][:, bit_k+1, :]
                    val = val === nothing ? mat : val * mat
                end
                result[idx+1] = val[1,1]
            end
            return result
        end
        b_vec = eval_mps(b)
        x_vec = eval_mps(engine.x)
        res = norm(scale .* x_vec .- b_vec) / norm(b_vec)
        @test res < 1e-6
    end

    # ----------------------------------------------------------------
    # sweep! returns (max_res, avg_trunc)
    # ----------------------------------------------------------------
    @testset "sweep! return type" begin
        using Random
        N, d, chi = 4, 2, 4
        b = make_random_mps(N, d, chi; seed=1)
        x = make_random_mps(N, d, chi; seed=2)
        A = make_identity_mpo(N, d, 2.0)
        engine = LinearSolverEngine(x, A, b)
        max_res, avg_trunc = sweep!(engine)
        @test max_res isa Float64
        @test avg_trunc isa Float64
        @test max_res >= 0.0
        @test avg_trunc >= 0.0
    end

end
