# QTT Julia 建構指南

Julia dense-only 移植。來源專案：`../../Uniten/UnitenQTT`（Python + cytnx）。
底層依賴：`../MPSCore`（UnitenDMRG 的 Julia 移植）。

---

## 必讀：全域約定

**在開始任何一個步驟之前，必須熟知以下約定。**

### Tensor leg 順序（不可違反）

```
MPS site tensor : Array{T,3}  →  dim 1=l, dim 2=i, dim 3=r
MPO site tensor : Array{T,4}  →  dim 1=l, dim 2=ip, dim 3=i, dim 4=r
```

`ip` = outgoing/bra physical（bra 端，對應 output leg）
`i`  = incoming/ket physical（ket 端，對應 input leg）

這和 MPSCore 一致。元素 ⟨bra|W|ket⟩ 在 array 裡是 `W[:, ket, bra, :]`。

### QTT site ordering：site 1 = LSB

```
n = i_1·2^0 + i_2·2^1 + ... + i_N·2^(N-1)
site 1 = 最低有效位（LSB）
site N = 最高有效位（MSB）
```

**與 numpy reshape 慣例相反**（numpy 把 MSB 放第 0 軸）。
轉換時：`n = sum(bits[k] * 2^(k-1) for k in 1:N)`

### Site 編號

- Julia 全程 **1-indexed**（Python 原始碼是 0-indexed）
- `mps[k]` for k = 1..N，邊界 `mps[1]` left bond dim = 1，`mps[N]` right bond dim = 1
- `env[0]` = left 邊界（scalar 1），`env[N+1]` = right 邊界（同 MPSCore 慣例）

### Python → Julia 對照

| Python (cytnx) | Julia |
|---|---|
| `to_numpy_array(tensor)` | 直接用（Julia tensor 本身就是 `Array`） |
| `tensor.shape()` | `size(tensor)` |
| `tensor.shape()[0]` | `size(tensor, 1)` |
| `tensor.permute_(["l","i","r"])` | `permutedims(A, [1,2,3])` 依 leg 位置 |
| `tensor.clone()` | `copy(A)` |
| `np.einsum("a,aib->ib", L, A)` | `@tensor B[i,b] := L[a] * A[a,i,b]` |
| `cytnx.eye(n)` | `Matrix{T}(I, n, n)` |
| `cytnx.ones([n])` | `ones(T, n)` |
| `cytnx.zeros([m,n,p])` | `zeros(T, m, n, p)` |
| `cytnx.Contract(A, B)` （outer product） | `kron(A_i, B_i)` 或 `@tensor` |
| `combineBonds([l_0,l_1,...])` | `reshape(A, prod_dims, ...)` |
| `direct_sum(A, B, ...)` | MPSCore `direct_sum(A, B, sum_dims)` |
| `MPS(tensors)` | MPSCore `MPS(tensors)` |
| `MPO(tensors)` | MPSCore `MPO(tensors)` |
| `mps_sum(a, b)` | MPSCore `mps_sum(a, b)` |
| `exact_apply_mpo(mpo, mps)` | MPSCore `exact_apply_mpo(mpo, mps)` |
| `mps[k]` （0-indexed） | `mps[k+1]` （1-indexed） |
| `from linalg import gmres` | `KrylovKit.linsolve`（見 Step 10） |

### embed 函式的 Kronecker product 技巧

Python 版用 `cytnx.Contract`（outer product）+ `combineBonds`。
Julia 等效寫法：

```julia
# 對每個 physical index i，combined matrix C_i[L, R] 是 Kronecker product：
for i in 1:phys_dim
    A_i = A[:, i, :]   # 當前 dim 的 Dl × Dr 矩陣
    mats = [d2 == d ? A_i : Matrix{T}(I, cur_bond[d2], cur_bond[d2])
            for d2 in 1:ndim]
    C[:, i, :] = reduce(kron, mats)
end
```

`reduce(kron, mats)` = kron(mats[1], kron(mats[2], ...)) → combined bond dim = ∏ D_{d2}

---

## 檔案結構

```
src/
  QTTCore.jl      # 主模組 (include 所有子檔)
  GridInfo.jl     # GridInfo struct
  Core/
    QTT.jl        # QTT type
    QTTO.jl       # QTTO type
  Init.jl         # qtt_sin/cos/linear/exp/random
  Operators.jl    # shift MPO, diff QTTO
  Operations.jl   # qtt_sum, qtt_prod, embed, interp, coarsen
  LinearSolver.jl # LinearSolverEngine (Ax=b)
test/
  runtests.jl
  test_grid_info.jl
  test_qtt.jl
  test_qtto.jl
  test_init.jl
  test_operators.jl
  test_operations.jl
  test_interp.jl
  test_linear_solver.jl
```

---

## Step 1：`Project.toml` + `src/QTT.jl`（骨架）

**對應 Python：`qtt/__init__.py`（package 入口）**

### 要做的事

1. 建立 `Project.toml`：名稱 `QTTCore`，加入 `MPSCore`（local path）、`KrylovKit`
2. 建立 `src/QTTCore.jl`：主模組骨架，`include` 全部子檔（先全部 comment 掉，逐步啟用）

### `Project.toml`

```toml
name = "QTTCore"
uuid = "（新生成）"
version = "0.1.0"

[deps]
MPSCore = "（MPSCore 的 uuid）"
KrylovKit = "..."
LinearAlgebra = "..."

[extras]
Test = "..."

[targets]
test = ["Test"]
```

### `src/QTTCore.jl` 骨架

```julia
module QTTCore

using LinearAlgebra
using MPSCore

include("GridInfo.jl")
include("Core/QTT.jl")
include("Core/QTTO.jl")
include("Init.jl")
include("Operators.jl")
include("Operations.jl")
include("LinearSolver.jl")

export GridInfo
export QTT, QTTO
export qtt_sin, qtt_cos, qtt_linear, qtt_exp, qtt_random
export qtt_sum, qtt_prod, embed
export qtt_interp0, qtt_coarsen, qtt_interp
export shift_forward_mpo, shift_backward_mpo
export qtto_diff_forward, qtto_diff_backward, qtto_diff2
export LinearSolverEngine, sweep!

end
```

### 測試

`test/runtests.jl`：空骨架，include 各測試檔（逐步啟用）。

---

## Step 2：`GridInfo`

**對應 Python：`qtt/grid_info.py`**

### 要做的事

```julia
struct GridInfo
    num_bits::Int       # 每個維度的 MPS sites 數
    interval::Tuple{Float64, Float64}   # (x_min, x_max)
    loc_dim::Int        # physical dim（預設 2）
end
GridInfo(num_bits, interval) = GridInfo(num_bits, interval, 2)
```

Python 是 `NamedTuple`；Julia 用 `struct` 即可，不需要 mutable。

### 測試：`test/test_grid_info.jl`

- 建構（含預設 `loc_dim=2`）
- `num_bits`、`interval`、`loc_dim` 欄位存取

---

## Step 3：`QTT` type

**對應 Python：`qtt/qtt.py`（`QTT` 類別）**

模組名 `QTTCore`，型別名 `QTT`。

### 要做的事

```julia
mutable struct QTT{T<:Number}
    grids::Vector{GridInfo}
    ordering::String           # "interleaved" 或 "sequential"
    mps::Union{MPS{T}, Nothing}
end
```

**方法（依 Python 對應）：**

| Python 方法 | Julia 對應 |
|---|---|
| `func_dim` property | `func_dim(q)` |
| `num_sites` property | `num_sites(q)` |
| `grid_points_per_dim` property | `grid_points_per_dim(q)` |
| `grid_spacings` property | `grid_spacings(q)` |
| `indices_to_bits(indices)` | `indices_to_bits(q, indices)` |
| `evaluate(*indices)` | `evaluate(q, indices...)` |
| `to_qtto()` | `to_qtto(q)` |
| `__len__` | `Base.length(q)` |
| `__getitem__` | `Base.getindex(q, k)` — **注意：1-indexed** |
| `__add__` | `Base.:(+)(a, b)` |
| `__mul__(scalar)` | `Base.:(*)(q, s)`, `Base.:(*)(s, q)` |
| `__mul__(QTT)` | `Base.:(*)(a::QTT, b::QTT)` |

**`evaluate` 的 Julia 實作：**

```julia
function evaluate(q::QTT, indices::Integer...)
    bits = indices_to_bits(q, collect(indices))
    result = nothing
    for k in 1:length(q.mps)
        A = q.mps[k]          # Array{T,3}, shape (l, i, r)
        mat = A[:, bits[k]+1, :]   # bits[k] ∈ {0,1}，Julia 1-indexed → +1
        result = (result === nothing) ? mat : result * mat
    end
    return result[1,1]
end
```

**`to_qtto` 的 Julia 實作：**
- 對每個 site k，取 `A = q.mps[k]`（shape `(l, i, r)`）
- 建 `W = zeros(T, l, i, i, r)`，填入 `W[:, j, j, :] = A[:, j, :]`
- 組成 MPO → QTTO

### 測試：`test/test_qtt.jl`

對應 Python `tests/test_qtt_class.py`：
- 建構驗證（無效 ordering、空 grids）
- `func_dim`、`num_sites`、`grid_points_per_dim`、`grid_spacings`
- 1D `indices_to_bits`（sequential / interleaved）
- 2D `indices_to_bits`（sequential / interleaved，含不等 num_bits）
- `evaluate` 前未設 mps 的 RuntimeError
- scalar multiplication、`length`、`getindex`
- **real 和 ComplexF64 都要測**

---

## Step 4：`QTTO` type

**對應 Python：`qtt/qtto.py`**

### 要做的事

```julia
mutable struct QTTO{T<:Number}
    grids::Vector{GridInfo}
    ordering::String
    mpo::Union{MPO{T}, Nothing}
end
```

方法完全與 `QTT` 對稱（把 `mps` 換成 `mpo`，evaluate 取 `W[:, b, b, :]` diagonal）。

**`evaluate` 核心：**
```julia
mat = W[:, bits[k]+1, bits[k]+1, :]   # diagonal element
```

scalar 乘法：`*(result.mpo[1], s)`，乘在第 1 個 site 的 tensor 上。

### 測試：`test/test_qtto.jl`

對應 Python `tests/test_qtto.py`：
- 建構驗證
- `evaluate` via `to_qtto()`（diagonal 應等於原 QTT 的值）
- 2D `indices_to_bits` spot-check
- **real 和 ComplexF64**

---

## Step 5：`Init.jl`

**對應 Python：`qtt/qtt_init.py`**

### 內部 helpers

```julia
# 把 (l, i, r) Array 包成 MPS site（直接回傳 Array{T,3}）
function _to_mps_tensor(arr::AbstractArray{<:Number,3})
    return Array{Float64,3}(arr)   # 或保留原 dtype
end

# 把 bulk + 邊界向量 → MPS
function _build_mps(bulk::Vector, L::Vector, R::Vector)
    # 對應 Python _build_mps：@tensor 縮併 L 進 site[1]，R 進 site[N]
end
```

### 公開函式

| Python | Julia |
|---|---|
| `qtt_sin(N, x1, x2)` | `qtt_sin(N, x1, x2)` |
| `qtt_cos(N, x1, x2)` | `qtt_cos(N, x1, x2)` |
| `qtt_linear(N, x1, x2; a=1, b=0)` | `qtt_linear(N, x1, x2; a=1.0, b=0.0)` |
| `qtt_exp(N, x1, x2; a=1, b=0)` | `qtt_exp(N, x1, x2; a=1.0, b=0.0)` |
| `qtt_random(grids, bond_dim; ordering)` | `qtt_random(grids, bond_dim; ordering="interleaved")` |

**sin/cos 的 rotation bulk tensor（shape (2,2,2,2) → 重排後 (2,2,2)）：**

```
Python bulk shape: (chi_l=2, phys=2, chi_r=2)
A[0, 0, 0] = 1, A[1, 0, 1] = 1   (phys=0: identity)
A[0, 1, 0] = c, A[0, 1, 1] = -s  (phys=1: rotation)
A[1, 1, 0] = s, A[1, 1, 1] = c
```

Julia 等效：
```julia
A = zeros(Float64, 2, 2, 2)   # (l, i, r)
A[1, 1, 1] = 1.0; A[2, 1, 2] = 1.0
A[1, 2, 1] = c;   A[1, 2, 2] = -sn
A[2, 2, 1] = sn;  A[2, 2, 2] = c
```

### 測試：`test/test_init.jl`

對應 Python `tests/test_exact_qtt_functions.py`：
- `qtt_sin` / `qtt_cos`：所有 grid points vs `sin`/`cos`，atol=1e-12
- `qtt_linear`：identity 和帶係數版本
- `qtt_exp`：基本 + 帶係數，atol=1e-10；bond dim 應全為 1
- `qtt_random`：1D/2D，boundary bond dim = 1，interior = bond_dim
- edge case：N=0 應報錯、N=1 兩點 grid
- **real 和 ComplexF64**

---

## Step 6：`Operators.jl`

**對應 Python：`qtt/qtt_operators.py`**

### 內部 helpers

```julia
const _I2  = [1.0 0.0; 0.0 1.0]
const _sp  = [0.0 1.0; 0.0 0.0]
const _sm  = [0.0 0.0; 1.0 0.0]

# (l, ip, i, r) bulk tensor for S⁺
function _shift_forward_bulk()::Array{Float64,4}
    A = zeros(2, 2, 2, 2)
    A[1, :, :, 1] = _I2
    A[2, :, :, 1] = _sp
    A[2, :, :, 2] = _sm
    return A
end

function _shift_backward_bulk()::Array{Float64,4} ... end

function _shift_R(bc::String)
    bc == "periodic"  && return [1.0, 1.0]
    bc == "dirichlet" && return [1.0, 0.0]
    error("bc must be \"periodic\" or \"dirichlet\"")
end
```

### 公開函式

| Python | Julia |
|---|---|
| `shift_forward_mpo(N; bc="periodic")` | `shift_forward_mpo(N; bc="periodic")::MPO` |
| `shift_backward_mpo(N; bc="periodic")` | `shift_backward_mpo(N; bc="periodic")::MPO` |
| `qtto_diff_forward(N, x1, x2; bc="periodic")` | 同 → `QTTO` |
| `qtto_diff_backward(N, x1, x2; bc="periodic")` | 同 → `QTTO` |
| `qtto_diff2(N, x1, x2; bc="periodic")` | 同 → `QTTO` |

**`_build_mpo` 的 @tensor 版本：**

```julia
# 吸收 L 進 site 1：
@tensor new1[j, ip, i, b] := L[j] * bulk[j, ip, i, b]
# 吸收 R 進 site N：
@tensor newN[j, ip, i, k] := bulk[j, ip, i, k] * R[k]
```

注意：Python `_build_mpo` 中 einsum 是 `"a,ajib->jib"`，i.e. 縮掉 L 的 index 與 bulk 的第 1 個 (l) 維度。

### 測試：`test/test_operators.jl`

對應 Python `tests/test_qtt_operators.py` + `tests/test_qtt_shift_bc.py`：

- dense matrix test：build `(2^N × 2^N)` matrix by exhaustive contraction，compare to analytic reference
- `shift_forward_mpo`：periodic/dirichlet，vs. `_shift_forward_dense(N, bc)`
- `shift_backward_mpo`：同上
- `qtto_diff_forward/backward`：periodic/dirichlet，vs. `(S⁺ - I)/dx` 和 `(I - S⁻)/dx`
- `qtto_diff2`：periodic（singular）/dirichlet（nonsingular）
- `diff_forward†` = `-diff_backward`（adjoint 關係）
- N=1 edge case、N=0 應報錯
- bond dim：`diff2` 內部應為 3

**`_mpo_to_dense` Julia helper（測試用）：**

```julia
function mpo_to_dense(mpo::MPO, N::Int)
    d = 2; total = d^N
    M = zeros(total, total)
    for bra in Iterators.product(ntuple(_->0:d-1, N)...)
        for ket in Iterators.product(ntuple(_->0:d-1, N)...)
            v = [1.0]
            for p in 1:N
                W = mpo[p]   # (l, ip, i, r)
                # bra index → "i" leg, ket index → "ip" leg
                mat = W[:, ket[p]+1, bra[p]+1, :]
                v = mat' * v
            end
            # QTT convention: site k → bit 2^(k-1)
            bf = sum(bra[p] * d^(p-1) for p in 1:N) + 1
            kf = sum(ket[p] * d^(p-1) for p in 1:N) + 1
            M[bf, kf] = v[1]
        end
    end
    return M
end
```

---

## Step 7：`Operations.jl`（Part 1：sum / prod / embed）

**對應 Python：`qtt/qtt_operations.py`**

### 要做的事

```julia
function _check_compatible(a::QTT, b::QTT) ... end

function qtt_sum(a::QTT{Ta}, b::QTT{Tb}) where {Ta,Tb}
    _check_compatible(a, b)
    T = promote_type(Ta, Tb)
    result = QTT{T}(a.grids, a.ordering, nothing)
    result.mps = mps_sum(a.mps, b.mps)   # MPSCore
    return result
end

function qtt_prod(a::QTT, b::QTT)
    qtto_a = to_qtto(a)
    result = QTT(...)
    result.mps = exact_apply_mpo(qtto_a.mpo, b.mps)   # MPSCore
    return result
end
```

**`embed`（interleaved ordering 的 kron 版本）：**

```julia
function embed(grids::Vector{GridInfo}, qtts_1d::Vector, ordering::String="interleaved")
    # ... 驗證 ...

    # Sequential：直接 concatenate（MPSCore MPS 支援 vcat-like 操作）
    # Interleaved：Kronecker product（見全域約定的技巧）
    if ordering == "sequential"
        tensors = vcat([sites_per_dim[d] for d in 1:ndim]...)
    else
        # interleaved：逐個 (k, d) site 用 kron
        for k in 0:max_bits-1
            for d in 1:ndim
                k >= grids[d].num_bits && continue
                A = sites_per_dim[d][k+1]   # (Dl, phys, Dr)
                phys_dim = size(A, 2)
                C = zeros(T, DL_total, phys_dim, DR_total)
                for i in 1:phys_dim
                    A_i = A[:, i, :]
                    mats = [d2 == d ? A_i : Matrix{T}(I, cur_bond[d2], cur_bond[d2])
                            for d2 in 1:ndim]
                    C[:, i, :] = reduce(kron, mats)
                end
                push!(tensors, C)
                cur_bond[d] = size(A, 3)
            end
        end
    end
    return QTT(grids, ordering, MPS(tensors))
end
```

### 測試：`test/test_operations.jl`

對應 Python `tests/test_qtt_operations.py`：
- `qtt_sum`：sin + cos，4 dtype 組合（real/real, real/complex, complex/real, complex/complex）
- `qtt_prod`：sin × linear，x²，4 dtype 組合
- arithmetic operators：`+`、`*`（QTT）、`*`（scalar）、`3.0 * q`
- `embed` sequential：1D identity，2D product，None dimension
- `embed` interleaved：1D identity，2D product，None，3D，unequal num_bits
- edge cases：invalid ordering，length mismatch，no mps，num_bits mismatch，interval mismatch

---

## Step 8：`Operations.jl`（Part 2：interp / coarsen）

**對應 Python：`qtt/qtt_operations.py`（後半）**

### 要做的事

```julia
function qtt_interp0(q::QTT)
    # 在 site 1 前面插入 identity site [1.0, 1.0]（shape (1,2,1)）
    new_site = ones(eltype(q), 1, 2, 1)   # 兩個 physical index 都是 1
    tensors = [new_site; [q.mps[k] for k in 1:length(q.mps)]]
    new_grid = GridInfo(q.grids[1].num_bits + 1, q.grids[1].interval)
    ...
end

function qtt_coarsen(q::QTT)
    # 對 site 1 縮掉 physical index（乘以 [0.5, 0.5]）並吸進 site 2
    site1 = q.mps[1]           # (1, 2, r1)
    avg = [0.5, 0.5]
    # @tensor contracted[r1] := avg[i] * site1[1, i, r1]
    # 然後吸進 site2：@tensor new2[l2, i2, r2] := contracted[l2] * site2_orig[l2, i2, r2]  ← 錯
    # 正確：@tensor new2[r1, i2, r2] := contracted[r1] * site2[r1, i2, r2]
    ...
end

function qtt_interp(q::QTT; bc::Float64=0.0)
    # 1. f' = S⁺ · f
    s_plus = shift_forward_mpo(q.grids[1].num_bits)
    f_shifted_mps = exact_apply_mpo(s_plus, q.mps)   # MPSCore
    # 2. bc correction（若 bc ≠ 0）
    # 3. h = (f + f') / 2
    h_mps = mps_sum(q.mps, f_shifted_mps)
    h_mps[1] = h_mps[1] * 0.5
    # 4. selector site B + direct_sum block-diagonal sites
    # direct_sum → MPSCore direct_sum(fk, hk, [1,3])（在 leg 1 和 leg 3 做 direct sum）
    ...
end
```

**`qtt_interp` 的 direct_sum 對應：**

Python `direct_sum(fk, hk, ["l","r"], ["l","r"], ["l","r"])` 是對 l 和 r bond 做 block-diagonal。
MPSCore 的 `direct_sum(A, B, sum_dims)` 中 `sum_dims=[1,3]` 表示對第 1 和第 3 個維度做 direct sum。

### 測試：`test/test_interp.jl`

對應 Python `tests/test_qtt_interp.py`：
- `qtt_interp0`：even points preserved，odd points copy left，structural properties
- `qtt_interp`：linear 精確，sin even points，sin odd points averages，bc 參數
- `qtt_coarsen`：sin/linear 平均，round-trip（interp0 → coarsen）
- edge cases：no mps，N=1 coarsen 應報錯
- **real 和 ComplexF64**

---

## Step 9：`LinearSolver.jl`

**對應 Python：`DMRGSolvers/linear_solver_engine.py`**

### 要做的事

```julia
mutable struct LinearSolverEngine{T}
    x::MPS{T}
    A::MPO
    b::MPS
    solver_kwargs::Dict
    _op_env::OperatorEnv
    _b_env::VectorEnv
end

function LinearSolverEngine(x, A, b; solver_kwargs=Dict())
    # x.center 必須是 1（Julia 1-indexed，對應 Python center=0）
    op_env = OperatorEnv(x, x, A; init_center=1)
    b_env  = VectorEnv(b, x; init_center=1)
    return LinearSolverEngine(x, A, b, solver_kwargs, op_env, b_env)
end

function sweep!(engine::LinearSolverEngine; max_dim=nothing, cutoff=0.0, num_center=2)
    # 右掃 p=1..N-1，左掃 p=N..1（1-indexed），回傳 (max_res, avg_trunc)
end
```

**GMRES via KrylovKit：**

Python `gmres(effA.apply, b_eff, x0=phi)` 的 Julia 等效：

```julia
# KrylovKit.linsolve：Ax = b，A 以 apply 函式給定
x0_vec = vec(phi)
b_vec  = vec(b_eff)
solution, info = KrylovKit.linsolve(v -> vec(apply(effA, reshape(v, size(phi)))),
                                    b_vec, x0_vec;
                                    tol=get(solver_kwargs, :tol, 1e-10),
                                    krylovdim=get(solver_kwargs, :k, 30))
phi_new = reshape(solution, size(phi))
local_res = info.normres[end]
```

注意：MPSCore 的 `EffOperator.apply(phi)` 接受 `Array`，KrylovKit 的 `linsolve` 接受 `AbstractVector`。需要 `vec`/`reshape` 轉換。

### 測試：`test/test_linear_solver.jl`

對應 Python `tests/dmrg_solvers/test_linear_solver_engine.py`：
- 5 dense dtype 組合：(real,real,real), (real,real,complex), (complex,real,real), (complex,real,complex), (complex,complex,complex)
- 每種組合：N=4 sites，shifted Heisenberg A（+ shift*I 確保 well-conditioned），random b，比較 `np.linalg.solve` 精確解，atol=1e-6
- one-site sweep（需要大初始 bond dim）
- 建構錯誤：`x.center ≠ 1`、length mismatch

---

## 實作順序總表

| Step | 實作檔 | 測試檔 | 依賴 |
|---|---|---|---|
| 1 | Project.toml, src/QTTCore.jl | runtests.jl | — |
| 2 | GridInfo.jl | test_grid_info.jl | — |
| 3 | Core/QTT.jl | test_qtt.jl | GridInfo, MPSCore.MPS/MPO |
| 4 | Core/QTTO.jl | test_qtto.jl | GridInfo, MPSCore.MPO |
| 5 | Init.jl | test_init.jl | QTT, MPSCore.MPS |
| 6 | Operators.jl | test_operators.jl | QTTO, MPSCore.MPO |
| 7 | Operations.jl（sum/prod/embed） | test_operations.jl | QTT, QTTO, Operators, MPSCore |
| 8 | Operations.jl（interp/coarsen） | test_interp.jl | Operations Part 1, Operators |
| 9 | LinearSolver.jl | test_linear_solver.jl | MPSCore.OperatorEnv/VectorEnv/EffOperator/EffVector, KrylovKit |
