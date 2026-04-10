# example_2d_polynomial.jl
#
# Example: build f(x,y) = x^2 + y^2 + 2xy = (x+y)^2 from 1D functions.
#
# Strategy:
#   1. Build 1D QTTs for x and y.
#   2. Compute x^2 and y^2 in 1D via qtt_prod.
#   3. Embed into 2D via embed (interleaved ordering, tensor-product structure).
#   4. Sum the three terms.
#   5. (Optional) compress with svd_compress_mps.
#   6. Verify against exact values.
#
# Run:
#     julia --project=.. examples/example_2d_polynomial.jl

using QTTCore, MPSCore, Printf

# ── Grid setup ───────────────────────────────────────────────────────
N  = 6
x1, x2 = -1.0, 1.0
y1, y2 = -1.0, 1.0

# ── 1D building blocks ───────────────────────────────────────────────
x_1d = qtt_linear(N, x1, x2)
y_1d = qtt_linear(N, y1, y2)

# ── x^2 and y^2 in 1D ───────────────────────────────────────────────
x2_1d = qtt_prod(x_1d, x_1d)
y2_1d = qtt_prod(y_1d, y_1d)

# ── Embed into 2D ────────────────────────────────────────────────────
grids = vcat(x_1d.grids, y_1d.grids)   # combine 1D grid metadata into 2D grid
qx2 = embed(grids, [x2_1d, nothing])   # x^2(x,y) = x^2, identity in y
qy2 = embed(grids, [nothing, y2_1d])   # y^2(x,y) = y^2, identity in x
qxy = embed(grids, [x_1d, y_1d])       # x*y via tensor product

# ── Sum: f = x^2 + y^2 + 2xy ─────────────────────────────────────────
qf = qtt_sum(qtt_sum(qx2, qy2), qtt_sum(qxy, qxy))

# ── (Optional) compress ──────────────────────────────────────────────
qf.mps = svd_compress_mps(qf.mps; max_dim=20, cutoff=1e-12)

# ── Verify against exact values ──────────────────────────────────────
println("Verifying f(x,y) = x^2 + y^2 + 2xy = (x+y)^2 ...")
n_pts   = 2^N
max_err = 0.0
for ix in 0 : n_pts ÷ 8 : n_pts - 1
    xv = x1 + ix * (x2 - x1) / (n_pts - 1)
    for iy in 0 : n_pts ÷ 8 : n_pts - 1
        yv      = y1 + iy * (y2 - y1) / (n_pts - 1)
        got     = evaluate(qf, ix, iy)
        exact   = xv^2 + yv^2 + 2 * xv * yv
        err     = abs(got - exact)
        global max_err = max(max_err, err)
        @printf("  f(%.3f, %.3f) = %.8f  exact = %.8f  err = %.2e\n",
                xv, yv, got, exact, err)
    end
end
@printf("Max error: %.2e\n", max_err)
