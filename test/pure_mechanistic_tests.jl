using Test, Random
import Pkg
Pkg.activate(temp=true)
Pkg.add([
    "CSV","DataFrames","Dates","Printf",
    "Flux","DiffEqFlux","Optimization","OptimizationOptimisers","OptimizationOptimJL",
    "OrdinaryDiffEq","SciMLSensitivity","Zygote","SciMLBase",
    "Plots","StatsPlots","Measures","ColorSchemes",
    "Functors","Interpolations","LaTeXStrings"
])

using CSV, DataFrames

# Include the future-work pipeline without running main(), thanks to the guard
include(joinpath(@__DIR__, "..", "src", "FutureWorks.jl"))

function _tiny_dyn_static()
    dyn = tempname()*"_dyn.csv"
    st  = tempname()*"_static.csv"

    # dynamic (with headers)
    df_time = DataFrame(
        KineticID=["C1","C1","C1"],
        TumorID=["T1","T1","T1"],
        Time=[0.0,1.0,2.0],
        TumorVolume=[100.0,120.0,140.0],
        ImmuneCellCount=[10.0,12.0,15.0]
    )
    CSV.write(dyn, df_time)

    # static (no header)
    open(st, "w") do io
        write(io, "80,0.20\n100,0.22\n120,0.24\n140,0.26\n")
    end
    return dyn, st
end

@testset "Load & preprocess (pure)" begin
    dyn, st = _tiny_dyn_static()
    dfd = load_dynamic_data(dyn)
    dfs = load_static_data(st)
    @test nrow(dfd) ≥ 1 && nrow(dfs) ≥ 1
    for c in ["KineticID","TumorID","Time","TumorVolume","ImmuneCellCount","VolumeNorm","ImmuneCellNorm"]
        @test c in names(dfd)
    end
    @test all(isfinite, dfd.VolumeNorm)
end

@testset "Groups & loss (pure)" begin
    dyn, st = _tiny_dyn_static()
    dfd = load_dynamic_data(dyn)
    dfs = load_static_data(st)
    groups,(tmin,tmax) = process_groups(dfd)
    @test !isempty(groups)
    θ0, re_dyn = initialize_parameters()
    L = compute_loss(θ0, groups, dfs, re_dyn, tmin, tmax)
    @test isfinite(L)
    t_pred, v_pred = predict_group(groups[1], 1, θ0, groups, re_dyn, tmin, tmax; dense_time_points=8)
    @test length(t_pred) == length(v_pred) > 0
    @test !any(isnan, v_pred)
end
