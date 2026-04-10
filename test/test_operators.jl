@testset "Operators" begin

    # Helper: apply a QTTO as a dense matrix to a QTT and return the result vector.
    # Builds O[row+1, col+1] = MPO matrix element (bra=row, ket=col) using
    # sequential virtual-bond contraction, then returns O * f where f = QTT values.
    function apply_operator_dense(qtto, qtt_in)
        N = qtto.grids[1].num_bits
        n_pts = 2^N
        f_vec = [evaluate(qtt_in, n) for n in 0:n_pts-1]
        O = zeros(n_pts, n_pts)
        for row in 0:n_pts-1
            for col in 0:n_pts-1
                row_bits = [(row >> k) & 1 for k in 0:N-1]  # LSB first, site 1 = bit 0
                col_bits = [(col >> k) & 1 for k in 0:N-1]
                val = nothing
                for k in 1:N
                    W = qtto.mpo[k]  # (l, ip, i, r)
                    mat = W[:, row_bits[k]+1, col_bits[k]+1, :]
                    val = val === nothing ? mat : val * mat
                end
                O[row+1, col+1] = val[1, 1]
            end
        end
        return O * f_vec
    end

    # ----------------------------------------------------------------
    # shift operators
    # ----------------------------------------------------------------
    @testset "shift_forward_mpo periodic" begin
        N = 4
        mpo = shift_forward_mpo(N)
        @test length(mpo) == N
        # S⁺|n⟩ = |n+1 mod 2^N⟩
        # Build as dense matrix and check shift
        n_pts = 2^N
        f = [Float64(n) for n in 0:n_pts-1]
        # Apply MPO to a QTT representing n (the identity function)
        q_id = qtt_linear(N, 0.0, Float64(n_pts-1); a=1.0, b=0.0)
        q_shifted_mps = exact_apply_mpo(mpo, q_id.mps)
        q_shifted = QTT([GridInfo(N, (0.0, Float64(n_pts-1)))], "sequential")
        q_shifted.mps = q_shifted_mps
        for n in 0:n_pts-2
            @test evaluate(q_shifted, n) ≈ Float64(n) + 1.0 atol=1e-10
        end
        # Last point wraps: S⁺|2^N-1⟩ = |0⟩ → value 0
        @test evaluate(q_shifted, n_pts-1) ≈ 0.0 atol=1e-10
    end

    @testset "shift_forward_mpo dirichlet" begin
        N = 4
        mpo = shift_forward_mpo(N; bc="dirichlet")
        n_pts = 2^N
        q_id = qtt_linear(N, 0.0, Float64(n_pts-1))
        q_shifted_mps = exact_apply_mpo(mpo, q_id.mps)
        q_shifted = QTT([GridInfo(N, (0.0, Float64(n_pts-1)))], "sequential")
        q_shifted.mps = q_shifted_mps
        for n in 0:n_pts-2
            @test evaluate(q_shifted, n) ≈ Float64(n) + 1.0 atol=1e-10
        end
        @test evaluate(q_shifted, n_pts-1) ≈ 0.0 atol=1e-10
    end

    @testset "shift_backward_mpo periodic" begin
        N = 4
        mpo = shift_backward_mpo(N)
        n_pts = 2^N
        q_id = qtt_linear(N, 0.0, Float64(n_pts-1))
        q_shifted_mps = exact_apply_mpo(mpo, q_id.mps)
        q_shifted = QTT([GridInfo(N, (0.0, Float64(n_pts-1)))], "sequential")
        q_shifted.mps = q_shifted_mps
        for n in 1:n_pts-1
            @test evaluate(q_shifted, n) ≈ Float64(n) - 1.0 atol=1e-10
        end
        # S⁻|0⟩ = |2^N-1⟩ → value 2^N-1
        @test evaluate(q_shifted, 0) ≈ Float64(n_pts-1) atol=1e-10
    end

    # ----------------------------------------------------------------
    # diff operators
    # ----------------------------------------------------------------
    @testset "qtto_diff_forward on linear" begin
        N, x1, x2, a, b = 6, 0.0, 2.0, 3.0, -1.0
        dx = (x2 - x1) / (2^N - 1)
        q = qtt_linear(N, x1, x2; a=a, b=b)
        op = qtto_diff_forward(N, x1, x2)
        result = apply_operator_dense(op, q)
        # Forward diff of a*x + b = a for interior points
        for n in 0:2^N-2
            @test result[n+1] ≈ a atol=1e-10
        end
    end

    @testset "qtto_diff_backward on linear" begin
        N, x1, x2, a, b = 6, 0.0, 2.0, 3.0, -1.0
        q = qtt_linear(N, x1, x2; a=a, b=b)
        op = qtto_diff_backward(N, x1, x2)
        result = apply_operator_dense(op, q)
        for n in 1:2^N-1
            @test result[n+1] ≈ a atol=1e-10
        end
    end

    @testset "qtto_diff2 on linear" begin
        N, x1, x2, a, b = 6, 0.0, 2.0, 3.0, -1.0
        q = qtt_linear(N, x1, x2; a=a, b=b)
        op = qtto_diff2(N, x1, x2)
        result = apply_operator_dense(op, q)
        # Second diff of linear = 0 for interior points
        for n in 1:2^N-2
            @test result[n+1] ≈ 0.0 atol=1e-10
        end
    end

    @testset "qtto_diff2 dirichlet on sin" begin
        N, x1, x2 = 6, 0.0, π
        dx = (x2 - x1) / (2^N - 1)
        q = qtt_sin(N, x1, x2)
        op = qtto_diff2(N, x1, x2; bc="dirichlet")
        result = apply_operator_dense(op, q)
        xs = [x1 + (x2 - x1) * n / (2^N - 1) for n in 0:2^N-1]
        # Interior: (sin(x+dx) - 2sin(x) + sin(x-dx)) / dx^2 ≈ -sin(x)
        for n in 1:2^N-2
            fd2 = (sin(xs[n+2]) - 2sin(xs[n+1]) + sin(xs[n])) / dx^2
            @test result[n+1] ≈ fd2 atol=1e-8
        end
    end

    # ----------------------------------------------------------------
    # QTTO properties
    # ----------------------------------------------------------------
    @testset "qtto_diff_forward structure" begin
        N, x1, x2 = 4, 0.0, 1.0
        op = qtto_diff_forward(N, x1, x2)
        @test op isa QTTO
        @test length(op) == N
        @test op.ordering == "sequential"
    end

    @testset "qtto_diff2 bond dim 3" begin
        N = 5
        op = qtto_diff2(N, 0.0, 1.0)
        # Interior bond dim should be 3
        for k in 2:N
            @test size(op.mpo[k], 1) == 3
        end
    end

end
