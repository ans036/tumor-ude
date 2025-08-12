# test/runtests.jl
using Test, Random

# --- (Quick path) Ensure deps exist even if no Project.toml is set up ---
import Pkg
Pkg.activate(temp=true)
Pkg.add([
    "CSV","DataFrames","Dates","Printf",
    "Flux","DiffEqFlux","Optimization","OptimizationOptimisers",
    "OrdinaryDiffEq","SciMLSensitivity","Zygote",
    "LinearAlgebra","Statistics","Random","Distributions",
    "Interpolations","Dierckx"
])

using CSV, DataFrames

# Include your script without executing main (thanks to the guard you added)
include(joinpath(@__DIR__, "..", "src", "JuliaconSubmission.jl"))

# --- Create tiny synthetic CSVs in temp files ---
function _tiny_paths()
    time_io = tempname()*"_time.csv"
    imm_io  = tempname()*"_immune.csv"

    # time file with headers (the script renames them)
    df_time = DataFrame(
        KineticID = ["C1","C1","C1"],
        TumorID   = ["T1","T1","T1"],
        Time = [0.0, 1.0, 2.0],
        TumorVolume = [100.0, 120.0, 140.0],
        ImmuneCellCount = [10.0, 12.0, 15.0],
    )
    CSV.write(time_io, df_time)

    # immune file WITHOUT header (your loader uses header=false)
    open(imm_io, "w") do io
        # TumorVolume, ImmuneCellFraction
        write(io, "80,0.20\n100,0.22\n120,0.24\n140,0.26\n")
    end
    return time_io, imm_io
end

@testset "Load & preprocess" begin
    tpath, ipath = _tiny_paths()
    df, μ, σ = load_and_merge_data(tpath, ipath)
    @test nrow(df) == 3
    @test :ImmuneCellFraction ∈ names(df)
    @test isfinite(μ) && isfinite(σ) && σ > 0
end

@testset "Group processing" begin
    tpath, ipath = _tiny_paths()
    df, μ, σ = load_and_merge_data(tpath, ipath)
    groups, (tmin, tmax) = process_groups(df)
    @test !isempty(groups)
    @test tmax > tmin
end

@testset "Loss is finite; ODE solvable" begin
    Random.seed!(123)
    tpath, ipath = _tiny_paths()
    df, μ, σ = load_and_merge_data(tpath, ipath)
    groups, (tmin, tmax) = process_groups(df)

    θ0, re_imm, re_corr, netsz = smart_parameter_initialization(groups; seed=7)
    L = compute_enhanced_loss(θ0, groups, re_imm, re_corr, netsz, tmin, tmax)
    @test isfinite(L)

    # Quick solve & predict on group 1
    t_pred, v_pred = predict_group(groups[1], 1, θ0, groups, re_imm, re_corr, netsz, tmin, tmax; dense_time_points=10)
    @test length(t_pred) == length(v_pred) > 0
    @test !any(isnan, v_pred)
    @test all(>(-1e-12), v_pred)  # nonnegative within tolerance
end
