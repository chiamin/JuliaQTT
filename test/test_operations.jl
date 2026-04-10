@testset "Operations" begin

    function grid_pts(N, x1, x2)
        [x1 + (x2 - x1) * n / (2^N - 1) for n in 0:2^N-1]
    end

    # ----------------------------------------------------------------
    # qtt_sum
    # ----------------------------------------------------------------
    @testset "qtt_sum sin + cos" begin
        N, x1, x2 = 6, 0.0, 2π
        qs = qtt_sum(qtt_sin(N, x1, x2), qtt_cos(N, x1, x2))
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(qs, n) ≈ sin(xs[n+1]) + cos(xs[n+1]) atol=1e-10
        end
    end

    @testset "qtt_sum preserves metadata" begin
        q1 = qtt_sin(5, 0.0, 1.0)
        q2 = qtt_cos(5, 0.0, 1.0)
        qs = qtt_sum(q1, q2)
        @test qs.grids == q1.grids
        @test qs.ordering == q1.ordering
    end

    @testset "qtt_sum mismatched grids raises" begin
        q1 = qtt_sin(5, 0.0, 1.0)
        q2 = qtt_sin(6, 0.0, 1.0)
        @test_throws ArgumentError qtt_sum(q1, q2)
    end

    @testset "qtt_sum mismatched ordering raises" begin
        g = [GridInfo(4, (0.0, 1.0))]
        q1 = QTT(g, "sequential")
        q2 = QTT(g, "interleaved")
        @test_throws ArgumentError qtt_sum(q1, q2)
    end

    # ----------------------------------------------------------------
    # qtt_prod
    # ----------------------------------------------------------------
    @testset "qtt_prod sin * linear" begin
        N, x1, x2 = 6, 0.1, 3.0
        q_sin = qtt_sin(N, x1, x2)
        q_x   = qtt_linear(N, x1, x2)
        qp = qtt_prod(q_sin, q_x)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(qp, n) ≈ sin(xs[n+1]) * xs[n+1] atol=1e-10
        end
    end

    @testset "qtt_prod square via prod" begin
        N, x1, x2 = 6, 0.0, 2.0
        q = qtt_linear(N, x1, x2)
        q2 = qtt_prod(q, q)
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(q2, n) ≈ xs[n+1]^2 atol=1e-10
        end
    end

    # ----------------------------------------------------------------
    # QTT arithmetic operators (+ and *)
    # ----------------------------------------------------------------
    @testset "QTT + operator" begin
        N, x1, x2 = 5, 0.0, 2.0
        q1 = qtt_sin(N, x1, x2)
        q2 = qtt_cos(N, x1, x2)
        qs = q1 + q2
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(qs, n) ≈ sin(xs[n+1]) + cos(xs[n+1]) atol=1e-10
        end
    end

    @testset "QTT * QTT operator" begin
        N, x1, x2 = 5, 0.1, 2.0
        q1 = qtt_sin(N, x1, x2)
        q2 = qtt_linear(N, x1, x2)
        qp = q1 * q2
        xs = grid_pts(N, x1, x2)
        for n in 0:2^N-1
            @test evaluate(qp, n) ≈ sin(xs[n+1]) * xs[n+1] atol=1e-10
        end
    end

    @testset "QTT * scalar" begin
        N = 5
        q = qtt_sin(N, 0.0, 1.0)
        q3 = q * 3.0
        xs = grid_pts(N, 0.0, 1.0)
        for n in 0:2^N-1
            @test evaluate(q3, n) ≈ 3.0 * sin(xs[n+1]) atol=1e-10
        end
    end

    @testset "scalar * QTT" begin
        N = 5
        q = qtt_sin(N, 0.0, 1.0)
        q3 = 3.0 * q
        xs = grid_pts(N, 0.0, 1.0)
        for n in 0:2^N-1
            @test evaluate(q3, n) ≈ 3.0 * sin(xs[n+1]) atol=1e-10
        end
    end

    # ----------------------------------------------------------------
    # embed — sequential
    # ----------------------------------------------------------------
    @testset "embed sequential 1D identity" begin
        N, x1, x2 = 5, 0.0, 2.0
        q = qtt_sin(N, x1, x2)
        q_emb = embed([GridInfo(N, (x1, x2))], [q], "sequential")
        for n in 0:2^N-1
            @test evaluate(q_emb, n) ≈ evaluate(q, n) atol=1e-12
        end
    end

    @testset "embed sequential 2D product" begin
        Nx, Ny = 4, 4
        x1, x2, y1, y2 = 0.0, 2.0, 0.0, 3.0
        q_x = qtt_sin(Nx, x1, x2)
        q_y = qtt_cos(Ny, y1, y2)
        grids = [GridInfo(Nx, (x1, x2)), GridInfo(Ny, (y1, y2))]
        q2d = embed(grids, [q_x, q_y], "sequential")
        xs = grid_pts(Nx, x1, x2)
        ys = grid_pts(Ny, y1, y2)
        for nx in 0:2^Nx-1, ny in 0:2^Ny-1
            @test evaluate(q2d, nx, ny) ≈ sin(xs[nx+1]) * cos(ys[ny+1]) atol=1e-10
        end
    end

    @testset "embed sequential none dimension" begin
        N = 4
        x1, x2 = 0.0, 2.0
        q_x = qtt_sin(N, x1, x2)
        grids = [GridInfo(N, (x1, x2)), GridInfo(N, (0.0, 1.0))]
        q2d = embed(grids, [q_x, nothing], "sequential")
        xs = grid_pts(N, x1, x2)
        for nx in 0:2^N-1, ny in 0:2^N-1
            @test evaluate(q2d, nx, ny) ≈ sin(xs[nx+1]) atol=1e-10
        end
    end

    @testset "embed sequential num_sites" begin
        grids = [GridInfo(3, (0.0, 1.0)), GridInfo(5, (0.0, 2.0))]
        q1 = qtt_sin(3, 0.0, 1.0)
        q2 = qtt_cos(5, 0.0, 2.0)
        q2d = embed(grids, [q1, q2], "sequential")
        @test length(q2d) == 8
    end

    # ----------------------------------------------------------------
    # embed — interleaved
    # ----------------------------------------------------------------
    @testset "embed interleaved 1D identity" begin
        N, x1, x2 = 5, 0.0, 2.0
        q = qtt_sin(N, x1, x2)
        q_emb = embed([GridInfo(N, (x1, x2))], [q], "interleaved")
        for n in 0:2^N-1
            @test evaluate(q_emb, n) ≈ evaluate(q, n) atol=1e-12
        end
    end

    @testset "embed interleaved 2D product" begin
        Nx, Ny = 4, 4
        x1, x2, y1, y2 = 0.0, 2.0, 0.0, 3.0
        q_x = qtt_sin(Nx, x1, x2)
        q_y = qtt_cos(Ny, y1, y2)
        grids = [GridInfo(Nx, (x1, x2)), GridInfo(Ny, (y1, y2))]
        q2d = embed(grids, [q_x, q_y], "interleaved")
        xs = grid_pts(Nx, x1, x2)
        ys = grid_pts(Ny, y1, y2)
        for nx in 0:2^Nx-1, ny in 0:2^Ny-1
            @test evaluate(q2d, nx, ny) ≈ sin(xs[nx+1]) * cos(ys[ny+1]) atol=1e-10
        end
    end

    @testset "embed interleaved none dimension" begin
        N = 4
        q_y = qtt_exp(N, 0.0, 1.0)
        grids = [GridInfo(N, (0.0, 1.0)), GridInfo(N, (0.0, 1.0))]
        q2d = embed(grids, [nothing, q_y], "interleaved")
        ys = grid_pts(N, 0.0, 1.0)
        for nx in 0:2^N-1, ny in 0:2^N-1
            @test evaluate(q2d, nx, ny) ≈ exp(ys[ny+1]) atol=1e-10
        end
    end

    @testset "embed interleaved 3D product" begin
        N = 3
        q_x = qtt_sin(N, 0.0, 1.0)
        q_y = qtt_cos(N, 0.0, 1.0)
        q_z = qtt_exp(N, 0.0, 1.0)
        grids = [GridInfo(N, (0.0, 1.0)) for _ in 1:3]
        q3d = embed(grids, [q_x, q_y, q_z], "interleaved")
        xs = grid_pts(N, 0.0, 1.0)
        for nx in 0:2^N-1, ny in 0:2^N-1, nz in 0:2^N-1
            @test evaluate(q3d, nx, ny, nz) ≈ sin(xs[nx+1]) * cos(xs[ny+1]) * exp(xs[nz+1]) atol=1e-9
        end
    end

    @testset "embed interleaved unequal num_bits" begin
        Nx, Ny = 3, 5
        q_x = qtt_linear(Nx, 0.0, 1.0)
        q_y = qtt_linear(Ny, 0.0, 2.0)
        grids = [GridInfo(Nx, (0.0, 1.0)), GridInfo(Ny, (0.0, 2.0))]
        q2d = embed(grids, [q_x, q_y], "interleaved")
        @test length(q2d) == Nx + Ny
        xs = grid_pts(Nx, 0.0, 1.0)
        ys = grid_pts(Ny, 0.0, 2.0)
        for nx in 0:2^Nx-1, ny in 0:2^Ny-1
            @test evaluate(q2d, nx, ny) ≈ xs[nx+1] * ys[ny+1] atol=1e-10
        end
    end

    # ----------------------------------------------------------------
    # embed — error cases
    # ----------------------------------------------------------------
    @testset "embed invalid ordering raises" begin
        grids = [GridInfo(4, (0.0, 1.0))]
        q = qtt_sin(4, 0.0, 1.0)
        @test_throws ArgumentError embed(grids, [q], "bad")
    end

    @testset "embed length mismatch raises" begin
        grids = [GridInfo(4, (0.0, 1.0)), GridInfo(4, (0.0, 1.0))]
        q = qtt_sin(4, 0.0, 1.0)
        @test_throws ArgumentError embed(grids, [q], "sequential")
    end

    @testset "embed no mps raises" begin
        grids = [GridInfo(4, (0.0, 1.0))]
        q = QTT(grids)
        @test_throws ErrorException embed(grids, [q], "sequential")
    end

    @testset "embed num_bits mismatch raises" begin
        grids = [GridInfo(5, (0.0, 1.0))]
        q = qtt_sin(4, 0.0, 1.0)
        @test_throws ArgumentError embed(grids, [q], "sequential")
    end

    @testset "embed interval mismatch raises" begin
        grids = [GridInfo(4, (0.0, 2.0))]
        q = qtt_sin(4, 0.0, 1.0)
        @test_throws ArgumentError embed(grids, [q], "sequential")
    end

    @testset "embed multidim qtt raises" begin
        grids_2d = [GridInfo(3, (0.0, 1.0)), GridInfo(3, (0.0, 1.0))]
        q2d = QTT(grids_2d)
        grids = [GridInfo(3, (0.0, 1.0))]
        @test_throws ArgumentError embed(grids, [q2d], "sequential")
    end

    # ----------------------------------------------------------------
    # qtt_interp0
    # ----------------------------------------------------------------
    @testset "qtt_interp0 constant function" begin
        N, x1, x2 = 4, 0.0, 1.0
        q = qtt_linear(N, x1, x2; a=0.0, b=3.0)  # constant 3
        q2 = qtt_interp0(q)
        @test q2.grids[1].num_bits == N+1
        for n in 0:2^(N+1)-1
            @test evaluate(q2, n) ≈ 3.0 atol=1e-12
        end
    end

    @testset "qtt_interp0 preserves values at even indices" begin
        N, x1, x2 = 4, 0.0, 1.0
        q = qtt_sin(N, x1, x2)
        q2 = qtt_interp0(q)
        for n in 0:2^N-1
            @test evaluate(q2, 2*n) ≈ evaluate(q, n) atol=1e-12
            @test evaluate(q2, 2*n+1) ≈ evaluate(q, n) atol=1e-12
        end
    end

    # ----------------------------------------------------------------
    # qtt_coarsen
    # ----------------------------------------------------------------
    @testset "qtt_coarsen averages adjacent pairs" begin
        N, x1, x2 = 5, 0.0, 1.0
        q = qtt_sin(N, x1, x2)
        q_c = qtt_coarsen(q)
        @test q_c.grids[1].num_bits == N-1
        xs = grid_pts(N, x1, x2)
        for n in 0:2^(N-1)-1
            expected = (evaluate(q, 2*n) + evaluate(q, 2*n+1)) / 2
            @test evaluate(q_c, n) ≈ expected atol=1e-12
        end
    end

    @testset "qtt_coarsen requires N >= 2" begin
        q = qtt_sin(1, 0.0, π)
        @test_throws ArgumentError qtt_coarsen(q)
    end

    # ----------------------------------------------------------------
    # qtt_interp
    # ----------------------------------------------------------------
    @testset "qtt_interp linear exact" begin
        N, x1, x2, a, b = 4, 0.0, 1.0, 2.0, 1.0
        q = qtt_linear(N, x1, x2; a=a, b=b)
        q2 = qtt_interp(q; bc=0.0)
        @test q2.grids[1].num_bits == N+1
        xs_old = grid_pts(N, x1, x2)
        vals = [evaluate(q2, n) for n in 0:2^(N+1)-1]
        # Even indices: g(2n) = f(n)
        for n in 0:2^N-1
            @test vals[2*n+1] ≈ a * xs_old[n+1] + b atol=1e-10
        end
        # Odd interior indices: g(2n+1) = (f(n) + f(n+1)) / 2
        for n in 0:2^N-2
            expected = (a * xs_old[n+1] + b + a * xs_old[n+2] + b) / 2
            @test vals[2*n+2] ≈ expected atol=1e-10
        end
    end

    @testset "qtt_interp with bc" begin
        N, x1, x2, a, b = 4, 0.0, 1.0, 2.0, 1.0
        q = qtt_linear(N, x1, x2; a=a, b=b)
        xs_old = grid_pts(N, x1, x2)
        dx_old = (x2 - x1) / (2^N - 1)
        bc = a * (x2 + dx_old) + b
        q2 = qtt_interp(q; bc=bc)
        vals = [evaluate(q2, n) for n in 0:2^(N+1)-1]
        f_last = a * xs_old[end] + b
        expected_last = (f_last + bc) / 2
        @test vals[end] ≈ expected_last atol=1e-10
    end

    @testset "qtt_interp sin even points preserved" begin
        N, x1, x2 = 6, 0.0, 2π
        q = qtt_sin(N, x1, x2)
        q2 = qtt_interp(q; bc=0.0)
        for n in 0:2^N-1
            @test evaluate(q2, 2*n) ≈ evaluate(q, n) atol=1e-10
        end
    end

end
