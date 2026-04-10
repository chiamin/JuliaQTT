@testset "Init" begin

    # Helper: dyadic grid points
    function grid_pts(N, x1, x2)
        [x1 + (x2 - x1) * n / (2^N - 1) for n in 0:2^N-1]
    end

    # ----------------------------------------------------------------
    # qtt_sin
    # ----------------------------------------------------------------
    @testset "qtt_sin basic" begin
        N, x1, x2 = 8, 0.0, 2π
        q = qtt_sin(N, x1, x2)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(q, n) ≈ sin(xs[n+1]) atol=1e-12
        end
    end

    @testset "qtt_sin shifted interval" begin
        N, x1, x2 = 6, 1.0, 4.0
        q = qtt_sin(N, x1, x2)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(q, n) ≈ sin(xs[n+1]) atol=1e-12
        end
    end

    @testset "qtt_sin bond dimensions" begin
        N = 6
        q = qtt_sin(N, 0.0, 1.0)
        @test length(q) == N
        @test size(q[1], 1) == 1    # left boundary bond
        @test size(q[N], 3) == 1    # right boundary bond
        for k in 2:N
            @test size(q[k], 1) == 2   # interior bond
        end
    end

    @testset "qtt_sin single site" begin
        q = qtt_sin(1, 0.0, π)
        @test evaluate(q, 0) ≈ sin(0.0) atol=1e-12
        @test evaluate(q, 1) ≈ sin(π)   atol=1e-12
    end

    @testset "qtt_sin invalid N" begin
        @test_throws ArgumentError qtt_sin(0, 0.0, 1.0)
    end

    # ----------------------------------------------------------------
    # qtt_cos
    # ----------------------------------------------------------------
    @testset "qtt_cos basic" begin
        N, x1, x2 = 8, 0.0, 2π
        q = qtt_cos(N, x1, x2)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(q, n) ≈ cos(xs[n+1]) atol=1e-12
        end
    end

    @testset "qtt_cos shifted interval" begin
        N, x1, x2 = 6, -1.0, 2.0
        q = qtt_cos(N, x1, x2)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(q, n) ≈ cos(xs[n+1]) atol=1e-12
        end
    end

    # ----------------------------------------------------------------
    # qtt_linear
    # ----------------------------------------------------------------
    @testset "qtt_linear identity" begin
        N, x1, x2 = 8, 0.0, 5.0
        q = qtt_linear(N, x1, x2)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(q, n) ≈ xs[n+1] atol=1e-12
        end
    end

    @testset "qtt_linear with coefficients" begin
        N, x1, x2, a, b = 7, 1.0, 3.0, 2.5, -1.0
        q = qtt_linear(N, x1, x2; a=a, b=b)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(q, n) ≈ a * xs[n+1] + b atol=1e-12
        end
    end

    @testset "qtt_linear constant (a=0)" begin
        N, x1, x2, b = 5, 0.0, 1.0, 3.14
        q = qtt_linear(N, x1, x2; a=0.0, b=b)
        for n in 0:2^N-1
            @test evaluate(q, n) ≈ b atol=1e-12
        end
    end

    # ----------------------------------------------------------------
    # qtt_exp
    # ----------------------------------------------------------------
    @testset "qtt_exp basic" begin
        N, x1, x2 = 8, 0.0, 2.0
        q = qtt_exp(N, x1, x2)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(q, n) ≈ exp(xs[n+1]) atol=1e-10
        end
    end

    @testset "qtt_exp with coefficients" begin
        N, x1, x2, a, b = 7, 0.0, 3.0, 0.5, -1.0
        q = qtt_exp(N, x1, x2; a=a, b=b)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(q, n) ≈ exp(a * xs[n+1] + b) atol=1e-10
        end
    end

    @testset "qtt_exp bond dim 1 (product state)" begin
        N = 6
        q = qtt_exp(N, 0.0, 1.0)
        for k in 1:N
            @test size(q[k], 1) == 1   # left bond
            @test size(q[k], 3) == 1   # right bond
        end
    end

    # ----------------------------------------------------------------
    # qtt_random
    # ----------------------------------------------------------------
    @testset "qtt_random structure" begin
        grids = [GridInfo(4, (0.0, 1.0))]
        q = qtt_random(grids, 5)
        @test length(q) == 4
        @test size(q[1], 1) == 1
        @test size(q[4], 3) == 1
        for k in 2:3
            @test size(q[k], 1) == 5
            @test size(q[k], 3) == 5
        end
    end

end
