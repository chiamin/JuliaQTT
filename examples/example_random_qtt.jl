# example_random_qtt.jl
#
# QTT example: create random QTT in 1D and 2D, evaluate and print values.
#
# Run:
#     julia --project=.. examples/example_random_qtt.jl

using QTTCore

# --- 1D random QTT ---
q1d = qtt_random(
    [GridInfo(8, (0.0, 2π))],
    4;
    ordering = "sequential",
)

println("=== 1D random QTT ===")
println("  Sites              : $(num_sites(q1d))")
println("  Grid points        : $(grid_points_per_dim(q1d)[1])")
println("  Grid spacing       : $(round(grid_spacings(q1d)[1], digits=6))")
println("  f(0)               = $(round(evaluate(q1d, 0), digits=6))")
println("  f(127)             = $(round(evaluate(q1d, 127), digits=6))")

# --- 2D random QTT ---
q2d = qtt_random(
    [GridInfo(4, (0.0, 1.0)),
     GridInfo(4, (0.0, 1.0))],
    3;
    ordering = "interleaved",
)

println("\n=== 2D random QTT ===")
println("  Sites              : $(num_sites(q2d))")
println("  Grid shape         : $(grid_shape(q2d))")
gs = grid_spacings(q2d)
println("  Grid spacings      : ($(round(gs[1], digits=4)), $(round(gs[2], digits=4)))")
println("  f(0, 0)            = $(round(evaluate(q2d, 0, 0), digits=6))")
println("  f(15, 15)          = $(round(evaluate(q2d, 15, 15), digits=6))")
