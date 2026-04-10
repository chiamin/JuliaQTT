# Testing Conventions

## Running tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

Or from the Julia REPL with the project activated:

```julia
using Pkg; Pkg.test()
```

---

## Failing tests must stay failing

- Tests should exercise the **real** API and data paths, not a trimmed-down
  scenario built only to avoid a known backend failure.
- **When a test fails, report the failure and leave it failing.**  Do not
  add `@test_skip`, weaken the scenario, delete the test, or add workarounds
  to make it pass.  Do not modify a failing test to make it pass unless the
  user explicitly asks.
- The **user decides** how to handle every test failure — AI must never
  make that decision autonomously.

---

## dtype coverage

Tests should cover both **real** (`Float64`) and **complex** (`ComplexF64`)
inputs wherever the implementation supports both.
