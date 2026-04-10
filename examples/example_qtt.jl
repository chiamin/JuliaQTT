# example_qtt.jl
#
# QTT (Quantics Tensor Train) example: exact QTT for x, sin(x), exp(x).
#
# Builds QTT representations of elementary functions on a dyadic grid
# of 2^N points and prints values at a sample of grid points.
#
# Run:
#     julia --project=.. examples/example_qtt.jl

using QTTCore

N = 10
x1, x2 = 0.0, 2π

# Build exact QTT for each function.
q_x   = qtt_linear(N, x1, x2)
q_sin = qtt_sin(N, x1, x2)
q_exp = qtt_exp(N, x1, x2; a=0.3)

n_pts = 2^N
dx    = (x2 - x1) / (n_pts - 1)

println("=== x ===")
println("  Site count : $(num_sites(q_x))")
for k in [0, n_pts ÷ 4, n_pts ÷ 2, 3 * n_pts ÷ 4, n_pts - 1]
    xv  = x1 + k * dx
    got = evaluate(q_x, k)
    println("  f($k) = $(round(got, digits=8))   exact = $(round(xv, digits=8))")
end

println("\n=== sin(x) ===")
println("  Site count : $(num_sites(q_sin))")
for k in [0, n_pts ÷ 4, n_pts ÷ 2, 3 * n_pts ÷ 4, n_pts - 1]
    xv  = x1 + k * dx
    got = evaluate(q_sin, k)
    ref = sin(xv)
    println("  f($k) = $(round(got, digits=8))   exact = $(round(ref, digits=8))")
end

println("\n=== exp(0.3x) ===")
println("  Site count : $(num_sites(q_exp))")
for k in [0, n_pts ÷ 4, n_pts ÷ 2, 3 * n_pts ÷ 4, n_pts - 1]
    xv  = x1 + k * dx
    got = evaluate(q_exp, k)
    ref = exp(0.3 * xv)
    println("  f($k) = $(round(got, digits=8))   exact = $(round(ref, digits=8))")
end
