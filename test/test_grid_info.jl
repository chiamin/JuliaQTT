@testset "GridInfo" begin

    @testset "default loc_dim" begin
        g = GridInfo(4, (0.0, 1.0))
        @test g.num_bits == 4
        @test g.interval == (0.0, 1.0)
        @test g.loc_dim == 2
    end

    @testset "explicit loc_dim" begin
        g = GridInfo(3, (1.0, 5.0), 3)
        @test g.num_bits == 3
        @test g.interval == (1.0, 5.0)
        @test g.loc_dim == 3
    end

    @testset "integer interval is promoted to Float64" begin
        g = GridInfo(4, (0, 1))
        @test g.interval === (0.0, 1.0)
    end

    @testset "equality" begin
        g1 = GridInfo(4, (0.0, 1.0))
        g2 = GridInfo(4, (0.0, 1.0))
        g3 = GridInfo(4, (0.0, 2.0))
        @test g1 == g2
        @test g1 != g3
    end

end
