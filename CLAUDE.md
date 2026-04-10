# CLAUDE.md

## First step for every new conversation

Read **all** files in `_internal/` before doing anything else:

- `_internal/PROJECT_NOTES.md` — project overview, coding style, Julia environment.
- `_internal/TESTING_CONVENTIONS.md` — no-workaround rule, how to run tests.
- `_internal/TODO.md` — task backlog.
- `_internal/FUTURE_WORK.md` — pending work.

## QTT conventions — IMPORTANT, easy to get wrong

Two conventions must be followed exactly whenever QTT data is converted to
or from a flat / array representation.

### 1. QTT site ordering: site 1 = least significant bit (LSB)

A grid index `n ∈ {0, ..., 2^N - 1}` is decomposed as

    n = i_1 · 2^0 + i_2 · 2^1 + ... + i_N · 2^(N-1)

and bit `i_k` lives on **MPS site k** (1-indexed, Julia convention).
Site 1 holds the LSB; site N holds the MSB.

This is the **opposite** of the natural reshape order, which puts the MSB
on the first axis.  When converting between QTT site order and a flat index:

- flatten via `n = sum(bits[k] * 2^(k-1) for k in 1:N)`, **or**
- reverse the site axes after reshaping.

### 2. MPO physical leg order: `(l, ip, i, r)` — `i` = bra, `ip` = ket

In every MPO tensor `W[l, ip, i, r]`, the convention is:

- **`i`  → bra (output) leg**
- **`ip` → ket (input) leg**

So the matrix element ⟨bra | W | ket⟩ is `W[:, ket, bra, :]` — note
the order is **(ket, bra)**, not (bra, ket).

This matches the MPSCore convention exactly.  When in doubt, sanity-check
against `shift_forward_mpo` (an asymmetric operator with a known dense form).

### 3. Tensor leg order (Julia plain arrays)

```
MPS site tensor : Array{T,3}  →  (l, i, r)
MPO site tensor : Array{T,4}  →  (l, ip, i, r)
```

There are no named labels — leg order is the sole convention.

## Coding conventions

- **No saving images in examples or tests.** Use `display(plot(...))` for
  interactive viewing only.
- **Backtick style:** Always use single backticks (`` `name` ``) for code
  terms in docstrings and comments.
- **1-indexed sites throughout.** MPS/MPO sites run from 1 to N.

## AI behavior

- **Report first, fix later.** When something goes wrong — failing tests,
  ambiguous requirements — summarise the problem and report it.  Do not
  rush to edit code just to make things pass.
- **Propose, don't decide.** If there are multiple options or a non-trivial
  trade-off, present the choices and ask.
- **Never hide failing tests.** When a test fails, report the failure and
  leave it failing.  Do not skip, weaken, or delete a test.  The user
  decides how to handle every failure.
- **Stay within Julia / MPSCore.** All operations use Julia native arrays
  and MPSCore APIs.  Do not add extra dependencies as a workaround.
- **Keep comments in sync.** When modifying code, update nearby comments
  and docstrings together with the code change.

## Dependencies

This project depends on `MPSCore` (the Julia port of UnitenDMRG), located
at `/home/chiamin/project/code/tensors/Julia/MPSCore/`.  It is added as a
local path dependency in `Project.toml`.

## Environment

```
Julia 1.10 LTS
```

Activate the project environment before running anything:

```julia
using Pkg; Pkg.activate(".")
```

## Module name

The Julia package is named **`QTTCore`** (module), and the main type is **`QTT`**.
Users write `using QTTCore` then `q = QTT(grids, ordering)`.

## Running tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```
