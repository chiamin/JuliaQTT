# example_diagonal_mpo.jl
#
# QTT example: compute f(x) = x^2 using `to_qtto` and `exact_apply_mpo`.
#
# Strategy:
#   1. Build QTT for f(x) = x.
#   2. Convert to a diagonal QTTO (the operator "multiply by x").
#   3. Apply that QTTO to the QTT for x -> QTT for x^2.
#   4. Print and compare with the exact x^2.
#
# Run:
#     julia --project=.. examples/example_diagonal_mpo.jl

using QTTCore, MPSCore

N    = 10
x1   = 0.0
x2   = 3.0
n_pts = 2^N
dx   = (x2 - x1) / (n_pts - 1)

# QTT for x and diagonal QTTO "multiply by x".
q_x  = qtt_linear(N, x1, x2)
qo_x = to_qtto(q_x)

# x * x = apply "multiply by x" to the QTT for x.
q_x2 = QTT([GridInfo(N, (x1, x2))], "sequential")
q_x2.mps = exact_apply_mpo(qo_x.mpo, q_x.mps)

println("=== x^2 via diagonal QTTO ===")
println("  Sites : $(num_sites(q_x2))")
println()
println("  $(rpad("k", 6)) $(rpad("QTT", 14)) $(rpad("exact x^2", 14)) $(rpad("error", 12))")
println("  " * "-"^50)
for k in [0, n_pts ÷ 8, n_pts ÷ 4, n_pts ÷ 2, 3 * n_pts ÷ 4, n_pts - 1]
    xv    = x1 + k * dx
    got   = evaluate(q_x2, k)
    exact = xv^2
    err   = abs(got - exact)
    println("  $(rpad(k, 6)) $(rpad(round(got, digits=8), 14)) " *
            "$(rpad(round(exact, digits=8), 14)) $(round(err, sigdigits=3))")
end
