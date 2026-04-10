@testset "QTT" begin

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------
    @testset "construction" begin
        @test QTT([GridInfo(4, (0.0, 1.0)), GridInfo(4, (0.0, 2.0))]) isa QTT
        @test QTT([GridInfo(4, (0.0, 1.0))], "sequential") isa QTT
        @test QTT([GridInfo(4, (0.0, 1.0))], "interleaved") isa QTT

        @test_throws ArgumentError QTT([GridInfo(4, (0.0, 1.0))], "bad_ordering")
        @test_throws ArgumentError QTT(GridInfo[])
    end

    @testset "default ordering is interleaved" begin
        q = QTT([GridInfo(4, (0.0, 1.0))])
        @test q.ordering == "interleaved"
    end

    @testset "mps is nothing initially" begin
        q = QTT([GridInfo(4, (0.0, 1.0))])
        @test q.mps === nothing
        @test length(q) == 0
    end

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------
    @testset "properties 1D" begin
        N, x1, x2 = 6, 0.0, 1.0
        q = QTT([GridInfo(N, (x1, x2))])
        @test func_dim(q) == 1
        @test num_sites(q) == N
        @test grid_points_per_dim(q) == (2^N,)
        @test grid_shape(q) == (2^N,)
        @test q.grids[1].loc_dim == 2
    end

    @testset "grid_spacings 1D" begin
        N, x1, x2 = 6, 0.0, 1.0
        q = QTT([GridInfo(N, (x1, x2))])
        @test grid_spacings(q)[1] ≈ (x2 - x1) / (2^N - 1)
    end

    @testset "grid_spacings 2D" begin
        q = QTT([GridInfo(4, (0.0, 3.0)), GridInfo(6, (1.0, 5.0))])
        @test grid_spacings(q)[1] ≈ 3.0 / (2^4 - 1)
        @test grid_spacings(q)[2] ≈ 4.0 / (2^6 - 1)
    end

    @testset "loc_dim=3" begin
        q = QTT([GridInfo(4, (0.0, 1.0), 3), GridInfo(4, (0.0, 1.0), 3)])
        @test grid_points_per_dim(q) == (3^4, 3^4)
        @test grid_shape(q) == (81, 81)
    end

    # ----------------------------------------------------------------
    # Error on access without MPS
    # ----------------------------------------------------------------
    @testset "evaluate without MPS raises" begin
        q = QTT([GridInfo(4, (0.0, 1.0))])
        @test_throws ErrorException evaluate(q, 0)
    end

    @testset "getindex without MPS raises" begin
        q = QTT([GridInfo(4, (0.0, 1.0))])
        @test_throws ErrorException q[1]
    end

    # ----------------------------------------------------------------
    # indices_to_bits — 1D
    # ----------------------------------------------------------------
    @testset "indices_to_bits 1D sequential" begin
        N = 4
        q = QTT([GridInfo(N, (0.0, 1.0))], "sequential")
        for n in 0:2^N-1
            bits = indices_to_bits(q, [n])
            expected = [(n >> k) & 1 for k in 0:N-1]
            @test bits == expected
        end
    end

    @testset "indices_to_bits 1D interleaved equals sequential" begin
        N = 4
        q_seq = QTT([GridInfo(N, (0.0, 1.0))], "sequential")
        q_int = QTT([GridInfo(N, (0.0, 1.0))], "interleaved")
        for n in 0:2^N-1
            @test indices_to_bits(q_seq, [n]) == indices_to_bits(q_int, [n])
        end
    end

    # ----------------------------------------------------------------
    # indices_to_bits — 2D
    # ----------------------------------------------------------------
    @testset "indices_to_bits 2D sequential" begin
        N = 3
        q = QTT([GridInfo(N, (0.0, 1.0)), GridInfo(N, (0.0, 1.0))], "sequential")
        # n_x=5 (binary 101) -> bits [1,0,1]
        # n_y=3 (binary 011) -> bits [1,1,0]
        # sequential: [1,0,1, 1,1,0]
        @test indices_to_bits(q, [5, 3]) == [1, 0, 1, 1, 1, 0]
    end

    @testset "indices_to_bits 2D interleaved" begin
        N = 3
        q = QTT([GridInfo(N, (0.0, 1.0)), GridInfo(N, (0.0, 1.0))], "interleaved")
        # n_x=5 -> [1,0,1], n_y=3 -> [1,1,0]
        # interleaved: [1,1, 0,1, 1,0]
        @test indices_to_bits(q, [5, 3]) == [1, 1, 0, 1, 1, 0]
    end

    @testset "indices_to_bits base-3" begin
        N = 2
        q = QTT([GridInfo(N, (0.0, 1.0), 3)])
        # 7 = 1*3^0 + 2*3^1 -> digits [1, 2]
        @test indices_to_bits(q, [7]) == [1, 2]
    end

    @testset "indices_to_bits unequal num_bits sequential" begin
        # dim1: 2 bits, dim2: 3 bits
        # n_x=3 (binary 11) -> [1,1]; n_y=5 (binary 101) -> [1,0,1]
        # sequential: [1,1, 1,0,1]
        q = QTT([GridInfo(2, (0.0, 1.0)), GridInfo(3, (0.0, 1.0))], "sequential")
        @test num_sites(q) == 5
        @test grid_points_per_dim(q) == (4, 8)
        @test indices_to_bits(q, [3, 5]) == [1, 1, 1, 0, 1]
    end

    @testset "indices_to_bits unequal num_bits interleaved" begin
        # dim1: 2 bits, dim2: 3 bits
        # n_x=3 -> [1,1]; n_y=5 -> [1,0,1]
        # interleaved: [x_b0, y_b0, x_b1, y_b1, y_b2] = [1,1, 1,0, 1]
        q = QTT([GridInfo(2, (0.0, 1.0)), GridInfo(3, (0.0, 1.0))], "interleaved")
        @test num_sites(q) == 5
        @test indices_to_bits(q, [3, 5]) == [1, 1, 1, 0, 1]
    end

    @testset "indices_to_bits wrong number of indices" begin
        q = QTT([GridInfo(3, (0.0, 1.0)), GridInfo(3, (0.0, 1.0))])
        @test_throws ArgumentError indices_to_bits(q, [5])
    end

end
