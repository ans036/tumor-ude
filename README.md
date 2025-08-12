# Tumor-UDE: Physics-Informed & Pure-Mechanistic UDEs for Tumor–Immune Dynamics

This repository contains two Julia pipelines for modeling tumor–immune dynamics from time-series tumor volume and immune measurements:

- **Classic (Enhanced PI-NODE)** — Physics-informed neural ODE with residual/correction subnetworks and multi-restart training (`src/JuliaconSubmission.jl`).
- **Future Works (Pure-Mechanistic UDE)** — Biologically constrained UDE that separates growth from immune killing and removes correction terms (`src/FutureWorks.jl`).

Both pipelines output training curves, observed vs predicted trajectories, counterfactual “no-immune” simulations, component analyses, and global fit summaries.

---

## Repository layout

```
.
├── src/
│   ├── JuliaconSubmission.jl      # Classic: enhanced PI-NODE pipeline
│   └── FutureWorks.jl             # Future works: pure-mechanistic UDE
├── test/
│   ├── runtests.jl                # Classic smoke/basics
│   └── pure_mechanistic_tests.jl  # Future-works smoke/basics
├── .github/workflows/ci.yml       # Matrix CI (classic & pure) + artifact upload
└── results/                       # Created at runtime (plots & analysis)
```

Continuous Integration installs Julia, caches dependencies, runs both test suites, and uploads result folders/ZIPs as build artifacts for each matrix target.

---

## Data format

Supply two CSVs (paths can be absolute or relative):

**Dynamic time-series (headered)**: `KineticID,TumorID,Time,TumorVolume,ImmuneCellCount`
```csv
KineticID,TumorID,Time,TumorVolume,ImmuneCellCount
C1,T1,0,100,10
C1,T1,1,120,12
C1,T1,2,140,15
```

**Static immune-rate (headerless, 2 columns)**: `TumorVolume,Im_cells_rate`
```
80,0.20
100,0.22
120,0.24
140,0.26
```

The code normalizes volumes and immune counts internally and is robust to very small datasets.

---

## Quick start (no `Project.toml` required)

The tests bootstrap a temporary environment and add needed packages automatically.

```bash
# Run classic pipeline tests
julia --color=yes test/runtests.jl

# Run future-works pipeline tests
julia --color=yes test/pure_mechanistic_tests.jl
```

Run the pipelines directly from the REPL:

```julia
julia> include("src/JuliaconSubmission.jl")
julia> main(time_file="data/tumor_time_to_event_data.csv",
            immune_file="data/tumor_volume_vs_Im_cells_rate.csv",
            save_plots=true)

julia> include("src/FutureWorks.jl")
julia> main(time_file="data/tumor_time_to_event_data.csv",
            immune_rate_file="data/tumor_volume_vs_Im_cells_rate.csv",
            save_plots=true)
```

Prefer pinned environments? Add a `Project.toml`/`Manifest.toml`, then:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

---

## What gets saved

Each run creates timestamped subfolders under `results/` (and, for the classic pipeline, `enhanced_results_*` if applicable) with:

- Training loss curves  
- Observed vs predicted trajectories  
- Counterfactuals (immune vs no-immune)  
- Component dynamics (growth vs immune killing)  
- Global fit plots (R²/RMSE/MAE)

CI uploads these files as build artifacts (see the **Actions** tab on each run).

---

## Methods (at a glance)

- **Classic PI-NODE**: ODE right-hand side augmented by neural subnetworks trained end-to-end with solver-in-the-loop sensitivity, leveraging OrdinaryDiffEq solvers.  
- **Pure-Mechanistic UDE**: Gompertz-like growth plus saturating immune-killing; parameters produced by a small network conditioned on (volume, time, immune), with physics-informed losses and consistency checks.

Built on the SciML stack (DiffEqFlux for neural/Universal DEs; OrdinaryDiffEq for time integration).

---

## Continuous Integration details

The workflow:

- Sets up Julia (`julia-actions/setup-julia@v2`)  
- Caches the Julia depot (`julia-actions/cache@v2`)  
- Runs a matrix over `{classic, pure}` tests  
- Uploads results using `actions/upload-artifact@v4` (one artifact per target)

---

## For JuliaCon Proceedings (JCON)

If you are preparing a JCON paper, place `paper.tex`, `paper.yml`, and `ref.bib` in a `paper/` folder and follow the Author’s Guide. (Proceedings complement accepted talks/posters with concise, citable write-ups.)

---

## Citing this work

Consider adding a `CITATION.cff` (rendered by GitHub on your repo homepage) so others can cite the software/paper accurately.

Example:
```yaml
cff-version: 1.2.0
message: "If you use Tumor-UDE, please cite it."
title: "Tumor-UDE: Physics-Informed & Pure-Mechanistic UDEs for Tumor–Immune Dynamics"
authors:
  - family-names: YourLast
    given-names: YourFirst
version: 0.1.0
date-released: 2025-08-12
type: software
```
