@testset "QTTO" begin

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------
    @testset "construction" begin
        @test QTTO([GridInfo(4, (0.0, 1.0)), GridInfo(4, (0.0, 2.0))]) isa QTTO
        @test QTTO([GridInfo(4, (0.0, 1.0))], "sequential") isa QTTO
        @test QTTO([GridInfo(4, (0.0, 1.0))], "interleaved") isa QTTO
        @test_throws ArgumentError QTTO([GridInfo(4, (0.0, 1.0))], "bad_ordering")
        @test_throws ArgumentError QTTO(GridInfo[])
    end

    @testset "mpo is nothing initially" begin
        q = QTTO([GridInfo(4, (0.0, 1.0))])
        @test q.mpo === nothing
        @test length(q) == 0
    end

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------
    @testset "properties 1D" begin
        N = 6
        q = QTTO([GridInfo(N, (0.0, 1.0))])
        @test func_dim(q) == 1
        @test num_sites(q) == N
        @test grid_points_per_dim(q) == (2^N,)
        @test grid_shape(q) == (2^N,)
        @test q.grids[1].loc_dim == 2
    end

    @testset "grid_spacings 1D" begin
        N, x1, x2 = 6, 0.0, 1.0
        q = QTTO([GridInfo(N, (x1, x2))])
        @test grid_spacings(q)[1] ≈ (x2 - x1) / (2^N - 1)
    end

    # ----------------------------------------------------------------
    # Error on access without MPO
    # ----------------------------------------------------------------
    @testset "evaluate without MPO raises" begin
        q = QTTO([GridInfo(4, (0.0, 1.0))])
        @test_throws ErrorException evaluate(q, 0)
    end

    @testset "getindex without MPO raises" begin
        q = QTTO([GridInfo(4, (0.0, 1.0))])
        @test_throws ErrorException q[1]
    end

    # ----------------------------------------------------------------
    # to_qtto — diagonal operator from QTT
    # ----------------------------------------------------------------
    @testset "to_qtto diagonal linear" begin
        N, x1, x2 = 6, 0.0, 2.0
        q = qtt_linear(N, x1, x2)
        qo = to_qtto(q)
        @test qo isa QTTO
        @test length(qo) == N
        xs = [x1 + (x2 - x1) * n / (2^N - 1) for n in 0:2^N-1]
        for n in 0:2^N-1
            @test evaluate(qo, n) ≈ xs[n+1] atol=1e-12
        end
    end

    @testset "to_qtto diagonal sin" begin
        N, x1, x2 = 6, 0.0, 2π
        q = qtt_sin(N, x1, x2)
        qo = to_qtto(q)
        xs = [x1 + (x2 - x1) * n / (2^N - 1) for n in 0:2^N-1]
        for n in [0, 1, 2^N-1, 17]
            @test evaluate(qo, n) ≈ sin(xs[n+1]) atol=1e-12
        end
    end

    # ----------------------------------------------------------------
    # indices_to_bits (spot-check; same logic as QTT)
    # ----------------------------------------------------------------
    @testset "indices_to_bits 2D interleaved" begin
        N = 3
        qo = QTTO([GridInfo(N, (0.0, 1.0)), GridInfo(N, (0.0, 1.0))],
                   "interleaved")
        @test indices_to_bits(qo, [5, 3]) == [1, 1, 0, 1, 1, 0]
    end

    @testset "indices_to_bits unequal num_bits sequential" begin
        qo = QTTO([GridInfo(2, (0.0, 1.0)), GridInfo(3, (0.0, 1.0))],
                   "sequential")
        @test num_sites(qo) == 5
        @test grid_points_per_dim(qo) == (4, 8)
    end

    # ----------------------------------------------------------------
    # Scalar multiplication and negation
    # ----------------------------------------------------------------
    @testset "scalar multiplication" begin
        N, x1, x2 = 4, 0.0, 1.0
        q = qtt_linear(N, x1, x2)
        qo = to_qtto(q)
        qo2 = qo * 2.0
        @test evaluate(qo2, 0) ≈ 2.0 * evaluate(qo, 0) atol=1e-12
        @test evaluate(qo2, 5) ≈ 2.0 * evaluate(qo, 5) atol=1e-12
    end

    @testset "negation" begin
        N, x1, x2 = 4, 0.0, 1.0
        q = qtt_linear(N, x1, x2)
        qo = to_qtto(q)
        qo_neg = -qo
        for n in 0:3
            @test evaluate(qo_neg, n) ≈ -evaluate(qo, n) atol=1e-12
        end
    end

end
