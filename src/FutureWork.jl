#!/usr/bin/env julia
################################################################################
# JULIACON 2025 ▸ BIOLOGICALLY CONSISTENT HYBRID UDE PIPELINE
# Pure mechanistic variant (no correction terms)
################################################################################

using CSV, DataFrames, Dates, Printf
using Flux, DiffEqFlux, Optimization, OptimizationOptimisers
using OrdinaryDiffEq, SciMLSensitivity, Zygote
using Plots, StatsPlots, Measures, ColorSchemes
using LinearAlgebra, Statistics, Random
using Flux: Chain, Dense, relu, tanh, sigmoid, destructure, f64, selu, softplus, σ, Parallel, softmax
using Functors                      # bring macro/module into scope
using OptimizationOptimJL
using Interpolations
using Interpolations: scale, extrapolate, CubicSplineInterpolation, Line, linear_interpolation
using LaTeXStrings
using SciMLBase                     # for successful_retcode, ODE types

# ---------- constants ----------
const PLOT_THEME_NAME = :vibrant
const PLOT_THEME_KW = (titlefontsize=16, guidefontsize=12, legendfontsize=10, grid=true, framestyle=:box, title_loc=:left)
const COLORS = ColorSchemes.tab20.colors
const VOL_SCALE = 1000.0
const IMMUNE_SCALE = 100.0
const PLOT_SIZE = (1200, 900)
const λ_VOL = 1.0
const λ_PHYS = 0.1
const λ_REG = 1e-5
const λ_NEG = 1e-3
const λ_CONSISTENCY = 0.5
const MAX_GROWTH_RATE = 2.0
const MIN_GROWTH_RATE = 0.01
const MAX_CARRYING_CAPACITY_FACTOR = 3.0
const MIN_CARRYING_CAPACITY_FACTOR = 1.1
const MIN_IMMUNE_KILLING = 0.001
const MAX_IMMUNE_KILLING = 10.0
const MIN_HALF_SATURATION = 0.01
const MAX_HALF_SATURATION = 5.0
const PATIENCE = 2000
const MAX_ITERS_ADAMW = 5000
const MAX_ITERS_LBFGS = 500
const INITIAL_LR = 1e-3

# ---------- helpers & data loading ----------
function safe_color(color_values::Vector{Float64}; colormap=:viridis)
    finite_vals = filter(isfinite, color_values)
    clean_values = replace(color_values, NaN => 0.0, Inf => (isempty(finite_vals) ? 0.0 : maximum(finite_vals)))
    if all(x -> x ≈ clean_values[1], clean_values)
        return fill(COLORS[1], length(clean_values))
    end
    min_val, max_val = extrema(clean_values)
    normalized = max_val > min_val ? (clean_values .- min_val) ./ (max_val - min_val) : fill(0.5, length(clean_values))
    cmap = ColorSchemes.colorschemes[colormap]
    return [get(cmap, val) for val in normalized]
end

function load_dynamic_data(time_file::String)
    df = CSV.read(time_file, DataFrame; header=1)
    rename!(df, [:KineticID, :TumorID, :Time, :TumorVolume, :ImmuneCellCount])
    df[!, [:Time, :TumorVolume, :ImmuneCellCount]] .= Float64.(df[!, [:Time, :TumorVolume, :ImmuneCellCount]])
    df = unique(df); dropmissing!(df)
    df.VolumeNorm = df.TumorVolume ./ VOL_SCALE
    df.ImmuneCellNorm = df.ImmuneCellCount ./ IMMUNE_SCALE
    println("📊 Loaded $(nrow(df)) dynamic measurements.")
    return df
end

function load_static_data(immune_rate_file::String)
    df = CSV.read(immune_rate_file, DataFrame; header=false)
    rename!(df, [:TumorVolume, :Im_cells_rate])
    df[!, [:TumorVolume, :Im_cells_rate]] .= Float64.(df[!, [:TumorVolume, :Im_cells_rate]])
    df = unique(df); dropmissing!(df)
    df.VolumeNorm = df.TumorVolume ./ VOL_SCALE
    println("📊 Loaded $(nrow(df)) static measurements for physics-informed regularization.")
    return df
end

# ---------- groups & interpolation ----------
struct TumorGroup
    id::Tuple{String, String}
    times::Vector{Float64}
    volumes::Vector{Float64}
    immune_levels::Vector{Float64}
    immune_interp::Any
    tspan::Tuple{Float64, Float64}
    u0::Vector{Float64}
    max_vol::Float64
end

function create_immune_interpolator(times::Vector{Float64}, immune_levels::Vector{Float64})
    perm = sortperm(times)
    sorted_times = times[perm]
    sorted_levels = immune_levels[perm]
    unique_mask = vcat(true, diff(sorted_times) .> 1e-6)
    unique_times = sorted_times[unique_mask]
    unique_levels = sorted_levels[unique_mask]
    if length(unique_times) == 0
        return t -> 0.0
    elseif length(unique_times) == 1
        return t -> unique_levels[1]
    elseif length(unique_times) >= 4
        try
            itp = interpolate(unique_levels, BSpline(Cubic(Line(OnGrid()))))
            scaled_itp = scale(itp, range(unique_times[1], unique_times[end], length=length(unique_times)))
            etp = extrapolate(scaled_itp, Line())
            return t -> etp(t)
        catch
            return linear_interpolation(unique_times, unique_levels; extrapolation_bc=Line())
        end
    else
        return linear_interpolation(unique_times, unique_levels; extrapolation_bc=Line())
    end
end

function process_groups(df::DataFrame)
    groups = TumorGroup[]
    gdf = groupby(df, [:KineticID, :TumorID])
    t_min_global = minimum(df.Time)
    t_max_global = maximum(df.Time)
    for g in gdf
        sort!(g, :Time)
        n = nrow(g); n < 2 && continue
        times = Float64.(g.Time)
        volumes_norm = Float64.(g.VolumeNorm)
        immune_norm = Float64.(g.ImmuneCellNorm)
        immune_interp = create_immune_interpolator(times, immune_norm)
        u0 = [volumes_norm[1]]
        max_vol = maximum(volumes_norm)
        push!(groups, TumorGroup((g.KineticID[1], g.TumorID[1]), times, volumes_norm, immune_norm, immune_interp, (times[1], times[end]), u0, max_vol))
    end
    println("🧬 Processed $(length(groups)) unique tumor groups for dynamic modeling.")
    return groups, (t_min_global, t_max_global)
end

# ---------- network & functor macro ----------
struct ResBlock{F}
    dense1::Dense
    dense2::Dense
    activation::F
end
@functor ResBlock

function ResBlock(dim::Int; activation=selu)
    ResBlock(Dense(dim, dim), Dense(dim, dim), activation)
end
(b::ResBlock)(x) = b.activation.(x .+ b.dense2(b.activation.(b.dense1(x))))

function create_dynamics_network()
    Chain(
        Dense(3, 32, selu),
        ResBlock(32, activation=selu),
        ResBlock(32, activation=selu),
        Parallel(
            tuple,
            Chain(Dense(32, 16, selu), Dense(16, 2)),  # -> r, K_factor
            Chain(Dense(32, 16, selu), Dense(16, 2))   # -> c_kill, h_sat
        )
    ) |> f64
end

function initialize_parameters()
    dynamics_nn = create_dynamics_network()
    θ_dynamics, re_dynamics = Flux.destructure(dynamics_nn)
    return θ_dynamics, re_dynamics
end

################################################################################
# 5. Biologically Consistent ODE System - Pure Mechanistic
################################################################################

function create_ode_system(group_idx::Int, groups::Vector{TumorGroup}, re_dynamics,
                           t_min_global::Float64, t_max_global::Float64)
    function dudt(u, p, t)
        dynamics_net = re_dynamics(p)
        V_norm = max(u[1], 1e-8)
        t_norm = (t - t_min_global) / (t_max_global - t_min_global + eps())
        immune_norm = groups[group_idx].immune_interp(t)

        nn_input = [V_norm, t_norm, immune_norm]
        nn_output = dynamics_net(nn_input)
        growth_params = nn_output[1]
        immune_params = nn_output[2]

        r_raw, K_factor_raw = growth_params
        c_kill_raw, h_sat_raw = immune_params

        r = MIN_GROWTH_RATE + (MAX_GROWTH_RATE - MIN_GROWTH_RATE) * sigmoid(r_raw)
        max_vol_group = groups[group_idx].max_vol
        K = max(max_vol_group * (MIN_CARRYING_CAPACITY_FACTOR +
              (MAX_CARRYING_CAPACITY_FACTOR - MIN_CARRYING_CAPACITY_FACTOR) *
              sigmoid(K_factor_raw)), V_norm * 1.01)
        gompertz_growth = r * V_norm * log(K / V_norm)

        c_kill = MIN_IMMUNE_KILLING + (MAX_IMMUNE_KILLING - MIN_IMMUNE_KILLING) * sigmoid(c_kill_raw)
        h_sat = MIN_HALF_SATURATION + (MAX_HALF_SATURATION - MIN_HALF_SATURATION) * sigmoid(h_sat_raw)
        immune_killing = immune_norm > 1e-3 ? (c_kill * V_norm * immune_norm) / (h_sat + V_norm) : 0.0

        dV_norm = gompertz_growth - immune_killing
        return [max(dV_norm, -0.1 * V_norm)]
    end
    return dudt
end

################################################################################
# 6. Loss Function with Biological Consistency - No Correction Penalty
################################################################################

function softrank(x; epsilon=0.1)
    diff_matrix = x .- x'
    s_terms = 1.0 .- σ.(diff_matrix ./ epsilon)
    return 0.5 .+ vec(sum(s_terms, dims=2))
end

function rank_correlation(x, y; epsilon=0.1)
    if length(x) < 2
        return 0.0
    end
    rx = softrank(x; epsilon)
    ry = softrank(y; epsilon)
    return cor(rx, ry)
end

function diff_median(x; beta=10.0)
    weights = softmax(-beta .* abs.(x .- x'))
    return sum(weights .* x, dims=2)
end

function compute_loss(θ::Vector{Float64}, groups::Vector{TumorGroup}, df_static::DataFrame,
                      re_dynamics, t_min_global::Float64, t_max_global::Float64;
                      solver=Tsit5(), sensealg=InterpolatingAdjoint())

    vol_loss, neg_penalty, count_vol = 0.0, 0.0, 0
    consistency_loss = 0.0
    dynamics_net = re_dynamics(θ)

    # Part 1: Loss on dynamic data
    for (i, group) in enumerate(groups)
        dudt = create_ode_system(i, groups, re_dynamics, t_min_global, t_max_global)
        prob = ODEProblem(dudt, group.u0, group.tspan, θ)
        sol = solve(prob, solver; saveat=group.times, sensealg=sensealg, abstol=1e-8, reltol=1e-8, dense=false)
        if !SciMLBase.successful_retcode(sol.retcode); return 1e6; end

        for (j, u_pred) in enumerate(sol.u)
            v_pred = u_pred[1]
            vol_loss += (v_pred - group.volumes[j])^2
            count_vol += 1
            if v_pred < 0; neg_penalty += v_pred^2; end
        end

        # Biological consistency: no-immune trajectory should be ≥ with-immune
        if length(group.times) > 1
            _, v_immune = predict_group(group, i, θ, groups, re_dynamics, t_min_global, t_max_global; dense_time_points=length(group.times))
            _, v_no_immune = predict_without_immune(group, i, θ, groups, re_dynamics, t_min_global, t_max_global; dense_time_points=length(group.times))
            for k in 2:length(group.times)
                if v_no_immune[k] < v_immune[k] + 0.05
                    consistency_loss += max(0, v_immune[k] - v_no_immune[k] + 0.05)^2
                end
            end
        end
    end
    vol_loss /= max(count_vol, 1)
    consistency_loss /= max(count_vol, 1)

    # Part 2: Physics-informed loss from static data
    median_t_norm = 0.5
    all_immune = vcat([g.immune_interp.(g.times) for g in groups]...)
    median_immune_norm = diff_median(all_immune)[1]

    pred_c_kill = map(df_static.VolumeNorm) do v_norm
        nn_input = [v_norm, median_t_norm, median_immune_norm]
        nn_output = dynamics_net(nn_input)
        immune_params = nn_output[2]
        c_kill_raw, _ = immune_params
        MIN_IMMUNE_KILLING + (MAX_IMMUNE_KILLING - MIN_IMMUNE_KILLING) * sigmoid(c_kill_raw)
    end
    physics_loss = (1 - rank_correlation(df_static.Im_cells_rate, pred_c_kill))^2

    # Part 3: Total Loss
    reg_loss = λ_REG * norm(θ)^2
    total_loss = (λ_VOL * vol_loss) +
                 (λ_PHYS * physics_loss) +
                 (λ_NEG * neg_penalty) +
                 (λ_CONSISTENCY * consistency_loss) +
                 reg_loss

    return total_loss
end

################################################################################
# 7. Training
################################################################################

function train_model(θ_init::Vector{Float64}, groups::Vector{TumorGroup}, df_static::DataFrame,
                     re_dynamics, t_min_global::Float64, t_max_global::Float64)
    losses = Float64[]
    best_loss = Inf
    best_θ = copy(θ_init)
    no_improve_ref = Ref(0)

    loss_func = (θ, p) -> compute_loss(θ, groups, df_static, re_dynamics, t_min_global, t_max_global)

    callback = function (state, loss)
        push!(losses, loss)
        if loss < best_loss
            best_loss = loss
            best_θ = copy(state.u)
            no_improve_ref[] = 0
        else
            no_improve_ref[] += 1
        end
        if length(losses) % 100 == 0
            @info "Iter $(length(losses)): Loss = $(round(loss, digits=6)) (Patience: $(no_improve_ref[]))"
        end
        return no_improve_ref[] >= PATIENCE
    end

    opt_func = OptimizationFunction(loss_func, AutoZygote())
    opt_prob = OptimizationProblem(opt_func, θ_init)

    println("🚀 Phase 1: AdamW optimization...")
    solve(opt_prob, Optimisers.AdamW(INITIAL_LR); callback=callback, maxiters=MAX_ITERS_ADAMW)

    if no_improve_ref[] < PATIENCE
        println("🎯 Phase 2: LBFGS fine-tuning...")
        opt_prob_lbfgs = OptimizationProblem(opt_func, best_θ)
        res_lbfgs = solve(opt_prob_lbfgs, OptimizationOptimJL.LBFGS(); maxiters=MAX_ITERS_LBFGS)
        if res_lbfgs.objective < best_loss
            best_θ = res_lbfgs.u
        end
    end

    return best_θ, losses
end

################################################################################
# 8. Prediction Functions - Pure Mechanistic
################################################################################

function predict_group(group::TumorGroup, group_idx::Int, θ::Vector{Float64}, groups::Vector{TumorGroup},
                       re_dynamics, t_min_global::Float64, t_max_global::Float64; dense_time_points=100)
    dudt = create_ode_system(group_idx, groups, re_dynamics, t_min_global, t_max_global)
    prob = ODEProblem(dudt, group.u0, group.tspan, θ)
    t_dense = range(group.tspan[1], group.tspan[2], length=dense_time_points)
    sol = solve(prob, Tsit5(); saveat=t_dense, dense=false)
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "Prediction failed for group $(group.id)"
        return t_dense, fill(NaN, length(t_dense))
    end
    return sol.t, [u[1] for u in sol.u]
end

function predict_without_immune(group::TumorGroup, group_idx::Int, θ::Vector{Float64}, groups::Vector{TumorGroup},
                                re_dynamics, t_min_global::Float64, t_max_global::Float64; dense_time_points=100)
    function dudt_no_immune(u, p, t)
        dynamics_net = re_dynamics(p)
        V_norm = max(u[1], 1e-8)
        t_norm = (t - t_min_global) / (t_max_global - t_min_global + eps())
        nn_input = [V_norm, t_norm, 0.0]
        nn_output = dynamics_net(nn_input)
        r_raw, K_factor_raw = nn_output[1]
        r = MIN_GROWTH_RATE + (MAX_GROWTH_RATE - MIN_GROWTH_RATE) * sigmoid(r_raw)
        max_vol_group = groups[group_idx].max_vol
        K = max(max_vol_group * (MIN_CARRYING_CAPACITY_FACTOR +
              (MAX_CARRYING_CAPACITY_FACTOR - MIN_CARRYING_CAPACITY_FACTOR) *
              sigmoid(K_factor_raw)), V_norm * 1.01)
        gompertz_growth = r * V_norm * log(K / V_norm)
        dV_norm = gompertz_growth
        return [max(dV_norm, -0.1 * V_norm)]
    end
    prob = ODEProblem(dudt_no_immune, group.u0, group.tspan, θ)
    t_dense = range(group.tspan[1], group.tspan[2], length=dense_time_points)
    sol = solve(prob, Tsit5(); saveat=t_dense, dense=false)
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "Prediction without immune failed for group $(group.id)"
        return t_dense, fill(NaN, length(t_dense))
    end
    return sol.t, [u[1] for u in sol.u]
end

################################################################################
# 9. Visualization Suite - Pure Mechanistic
################################################################################

function create_all_plots(groups::Vector{TumorGroup}, df_static::DataFrame, θ::Vector{Float64}, losses::Vector{Float64},
                          re_dynamics, t_min_global::Float64, t_max_global::Float64; save_dir="results")

    mkpath(save_dir)
    enhanced_dir = joinpath(save_dir, "enhanced_visualizations")
    mkpath(enhanced_dir)
    dynamics_net = re_dynamics(θ)

    # 1) Training Loss Curve
    plt_loss = plot(losses, xlabel="Iteration", ylabel="Loss", title="Training Loss Progression",
                   yscale=:log10, lw=3, legend=false, color=COLORS[1], size=PLOT_SIZE, margin=10mm)
    savefig(plt_loss, joinpath(save_dir, "training_loss.png"))

    # 2) Per-Group Analysis
    all_vol_pred, all_vol_obs, residuals = Float64[], Float64[], Float64[]
    all_params = Dict(:r => Float64[], :K => Float64[], :c_kill => Float64[], :h_sat => Float64[])
    immune_influence = Dict(:kinetic => String[], :tumor => String[], :avg_ratio => Float64[])

    for (idx, group) in enumerate(groups)
        kinetic, tumor = group.id
        group_dir = joinpath(save_dir, "group_$(kinetic)_$(tumor)")
        mkpath(group_dir)

        t_pred, v_pred_norm = predict_group(group, idx, θ, groups, re_dynamics, t_min_global, t_max_global)
        t_no_immune, v_no_immune_norm = predict_without_immune(group, idx, θ, groups, re_dynamics, t_min_global, t_max_global)

        if !any(isnan, v_pred_norm)
            itp = linear_interpolation(t_pred, v_pred_norm; extrapolation_bc=Line())
            interp_pred_norm = itp.(group.times)
            append!(all_vol_obs, group.volumes .* VOL_SCALE)
            append!(all_vol_pred, interp_pred_norm .* VOL_SCALE)
            append!(residuals, (interp_pred_norm .- group.volumes) .* VOL_SCALE)
        end

        # Plot 1: Observed vs Predicted
        plt_vol = plot(title="Tumor Dynamics: $kinetic | $tumor", xlabel="Time (days)", ylabel="Volume (mm³)",
                       legend=:topleft, size=PLOT_SIZE, margin=10mm)
        scatter!(plt_vol, group.times, group.volumes .* VOL_SCALE, label="Observed", color=COLORS[1], ms=8, alpha=0.8)
        plot!(plt_vol, t_pred, v_pred_norm .* VOL_SCALE, label="Predicted (with immune)", color=COLORS[2], lw=3)
        plot!(plt_vol, t_no_immune, v_no_immune_norm .* VOL_SCALE, label="Predicted (no immune)", color=COLORS[3], lw=3, linestyle=:dash)

        if !any(isnan, v_no_immune_norm) && !any(isnan, v_pred_norm)
            final_advantage = (v_no_immune_norm[end] - v_pred_norm[end]) / max(v_pred_norm[end], 1e-8) * 100
            max_advantage = maximum((v_no_immune_norm .- v_pred_norm) ./ (abs.(v_pred_norm) .+ 1e-8)) * 100
            vol_range = maximum(group.volumes .* VOL_SCALE) - minimum(group.volumes .* VOL_SCALE)
            y_pos = minimum(group.volumes .* VOL_SCALE) + 0.1 * max(vol_range, 1.0)
            annotate!(plt_vol, group.tspan[2]*0.05, y_pos,
                      text("Final Growth Advantage: $(round(final_advantage, digits=1))%\nMax Advantage: $(round(max_advantage, digits=1))%",
                           :left, 10, :darkgreen))
        end
        savefig(plt_vol, joinpath(group_dir, "volume_comparison.png"))

        # Plot 2: Immune Dynamics
        immune_vals = group.immune_interp.(t_pred) .* IMMUNE_SCALE
        plt_immune = plot(title="Immune Cell Dynamics: $kinetic | $tumor", xlabel="Time (days)", ylabel="Immune Cell Count",
                          legend=:topleft, size=PLOT_SIZE, margin=10mm)
        plot!(plt_immune, t_pred, immune_vals, label="Immune Cells", color=COLORS[4], lw=3)
        scatter!(plt_immune, group.times, group.immune_levels .* IMMUNE_SCALE, label="Observed Immune", color=COLORS[5], ms=6, alpha=0.7)
        immune_threshold = 0.1 * maximum(immune_vals)
        active_mask = immune_vals .> immune_threshold
        if any(active_mask)
            for i in 1:length(t_pred)-1
                if active_mask[i] && active_mask[i+1]
                    plot!(plt_immune, [t_pred[i], t_pred[i+1]], [immune_vals[i], immune_vals[i+1]],
                          color=:red, lw=5, alpha=0.3, label=i==1 ? "High Activity" : "")
                end
            end
        end
        mean_immune = mean(immune_vals); max_immune = maximum(immune_vals)
        annotate!(plt_immune, group.tspan[2]*0.05, 0.8*maximum(immune_vals),
                  text("Mean: $(round(mean_immune, digits=1))\nMax: $(round(max_immune, digits=1))",
                       :left, 10, :darkblue))
        savefig(plt_immune, joinpath(group_dir, "immune_dynamics.png"))

        # Plot 3: Component Dynamics
        gompertz_terms, immune_killing_terms = Float64[], Float64[]
        growth_rates_with_immune, growth_rates_without_immune = Float64[], Float64[]

        if !any(isnan, v_pred_norm)
            for (t,v) in zip(t_pred, v_pred_norm)
                v_used = max(v, 1e-8)
                t_norm = (t - t_min_global) / (t_max_global - t_min_global + eps())
                immune_norm = group.immune_interp(t)

                nn_input = [v_used, t_norm, immune_norm]
                nn_output = dynamics_net(nn_input)
                growth_params = nn_output[1]
                immune_params = nn_output[2]
                r_raw, K_factor_raw = growth_params
                c_kill_raw, h_sat_raw = immune_params

                r = MIN_GROWTH_RATE + (MAX_GROWTH_RATE - MIN_GROWTH_RATE) * sigmoid(r_raw)
                max_vol_group = group.max_vol
                K_unscaled = max_vol_group * (MIN_CARRYING_CAPACITY_FACTOR +
                             (MAX_CARRYING_CAPACITY_FACTOR - MIN_CARRYING_CAPACITY_FACTOR) *
                             sigmoid(K_factor_raw))
                K_val = max(K_unscaled, v_used * 1.01)
                c_kill = MIN_IMMUNE_KILLING + (MAX_IMMUNE_KILLING - MIN_IMMUNE_KILLING) * sigmoid(c_kill_raw)
                h_sat = MIN_HALF_SATURATION + (MAX_HALF_SATURATION - MIN_HALF_SATURATION) * sigmoid(h_sat_raw)

                gompertz_term = r * v_used * log(K_val / v_used)
                immune_killing_term = immune_norm > 1e-3 ? (c_kill * v_used * immune_norm) / (h_sat + v_used) : 0.0
                growth_rate_with_immune = gompertz_term - immune_killing_term

                nn_input_no_immune = [v_used, t_norm, 0.0]
                nn_output_no_immune = dynamics_net(nn_input_no_immune)
                r_raw_no, K_factor_raw_no = nn_output_no_immune[1]
                r_no = MIN_GROWTH_RATE + (MAX_GROWTH_RATE - MIN_GROWTH_RATE) * sigmoid(r_raw_no)
                K_unscaled_no = max_vol_group * (MIN_CARRYING_CAPACITY_FACTOR +
                                (MAX_CARRYING_CAPACITY_FACTOR - MIN_CARRYING_CAPACITY_FACTOR) *
                                sigmoid(K_factor_raw_no))
                K_val_no = max(K_unscaled_no, v_used * 1.01)
                gompertz_term_no = r_no * v_used * log(K_val_no / v_used)
                growth_rate_without_immune = gompertz_term_no

                if !(isfinite(gompertz_term) && isfinite(immune_killing_term) &&
                      isfinite(growth_rate_with_immune) && isfinite(growth_rate_without_immune))
                    @warn "Non-finite component at t=$t for group $(group.id)"
                    gompertz_term = 0.0; immune_killing_term = 0.0
                    growth_rate_with_immune = 0.0; growth_rate_without_immune = 0.0
                end

                push!(gompertz_terms, gompertz_term)
                push!(immune_killing_terms, immune_killing_term)
                push!(growth_rates_with_immune, growth_rate_with_immune)
                push!(growth_rates_without_immune, growth_rate_without_immune)

                push!(all_params[:r], r)
                push!(all_params[:K], K_val * VOL_SCALE)
                push!(all_params[:c_kill], c_kill)
                push!(all_params[:h_sat], h_sat)
            end

            plt_comp = plot(title="Component Dynamics: $kinetic | $tumor", xlabel="Time (days)", ylabel="Rate of Change (dV/dt)",
                            legend=:topright, size=PLOT_SIZE, margin=10mm)
            plot!(plt_comp, t_pred, gompertz_terms, label="Gompertz Growth", color=COLORS[1], lw=3)
            plot!(plt_comp, t_pred, -immune_killing_terms, label="Immune Killing", color=COLORS[2], lw=3)
            plot!(plt_comp, t_pred, gompertz_terms .- immune_killing_terms, label="Net Growth", color=COLORS[6], lw=2, linestyle=:dash)
            hline!([0], color=:black, linestyle=:dot, alpha=0.5, label="Zero Growth")

            immune_vals_norm = group.immune_interp.(t_pred)
            high_immune_mask = immune_vals_norm .> 0.1 * maximum(immune_vals_norm)
            if any(high_immune_mask)
                scatter!(plt_comp, t_pred[high_immune_mask], gompertz_terms[high_immune_mask],
                         color=:red, ms=3, alpha=0.3, label="High Immune Activity")
            end
            savefig(plt_comp, joinpath(group_dir, "component_dynamics.png"))

            # Growth-rate comparison
            if !any(isnan, v_no_immune_norm)
                plt_growth = plot(title="Growth Rate Analysis: $kinetic | $tumor", xlabel="Time (days)", ylabel="Growth Rate (dV/dt)",
                                  legend=:topright, size=PLOT_SIZE, margin=10mm)
                plot!(plt_growth, t_pred, growth_rates_with_immune,   label="With Immune",  color=COLORS[2], lw=3)
                plot!(plt_growth, t_pred, growth_rates_without_immune, label="No Immune",    color=COLORS[3], lw=3, linestyle=:dash)
                rate_diff = growth_rates_without_immune .- growth_rates_with_immune
                plot!(plt_growth, t_pred, rate_diff, label="Immune Suppression", color=COLORS[7], lw=2, alpha=0.7)
                plot!(plt_growth, t_pred, growth_rates_with_immune, fillto=growth_rates_without_immune, alpha=0.2, color=COLORS[7], label="")
                hline!([0], color=:black, linestyle=:dot, alpha=0.5, label="Zero Growth")
                savefig(plt_growth, joinpath(group_dir, "growth_rate_analysis.png"))

                plt_size = plot(title="Tumor Size vs Growth Rate: $kinetic | $tumor", xlabel="Time (days)", ylabel="Volume (mm³)",
                                legend=:topleft, size=PLOT_SIZE, margin=10mm)
                colors_with  = safe_color(growth_rates_with_immune; colormap=:viridis)
                colors_without = safe_color(growth_rates_without_immune; colormap=:plasma)
                scatter!(plt_size, t_pred, v_pred_norm .* VOL_SCALE, color=colors_with, ms=4, alpha=0.8,
                         label="With Immune (colored by growth rate)")
                scatter!(plt_size, t_pred, v_no_immune_norm .* VOL_SCALE, color=colors_without, ms=4, alpha=0.6,
                         marker=:diamond, label="No Immune (colored by growth rate)")
                scatter!(plt_size, group.times, group.volumes .* VOL_SCALE, color=:black, ms=6, alpha=0.8, label="Observed")
                savefig(plt_size, joinpath(group_dir, "size_vs_growth_rate.png"))
            end
        end
    end

    # 3) Global Fit
    if !isempty(all_vol_obs) && !isempty(all_vol_pred)
        plt_global = scatter(all_vol_obs, all_vol_pred,
                             xlabel="Observed Volume (mm³)", ylabel="Predicted Volume (mm³)",
                             title="Global Model Fit", alpha=0.6, legend=:bottomright,
                             size=PLOT_SIZE, margin=10mm, color=COLORS[1])
        max_val = max(maximum(all_vol_obs), maximum(all_vol_pred))
        min_val = min(minimum(all_vol_obs), minimum(all_vol_pred))
        plot!(plt_global, [min_val, max_val], [min_val, max_val], color=:red, linestyle=:dash, lw=2, label="Perfect Fit")
        r2 = 1 - sum(residuals.^2) / max(sum((all_vol_obs .- mean(all_vol_obs)).^2), eps())
        rmse = sqrt(mean(residuals.^2))
        mae = mean(abs.(residuals))
        annotate!(plt_global, 0.05*max_val, 0.9*max_val,
                  text("R² = $(round(r2, digits=3))\nRMSE = $(round(rmse, digits=1))\nMAE = $(round(mae, digits=1))",
                       :left, 12, :darkblue))
        savefig(plt_global, joinpath(enhanced_dir, "global_fit.png"))
    end

    @info "✅ All plots saved to $save_dir and $enhanced_dir"
end

# ---------- Main ----------
function main(; time_file="tumor_time_to_event_data.csv",
               immune_rate_file="tumor_volume_vs_Im_cells_rate.csv",
               save_plots=true)
    Random.seed!(42)
    @info "🚀 Starting Biologically Consistent Hybrid UDE Pipeline - Pure Mechanistic"

    df_dynamic = load_dynamic_data(time_file)
    df_static  = load_static_data(immune_rate_file)
    groups, (tmin, tmax) = process_groups(df_dynamic)
    isempty(groups) && error("No valid tumor groups found!")

    θ_init, re_dynamics = initialize_parameters()
    θ_trained, losses = train_model(θ_init, groups, df_static, re_dynamics, tmin, tmax)

    if save_plots
        timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
        save_dir = joinpath("results", "pure_mechanistic", "results_PureMechanisticUDE_" * timestamp)
        create_all_plots(groups, df_static, θ_trained, losses, re_dynamics, tmin, tmax; save_dir=save_dir)
    end

    return θ_trained, losses
end

# ---------- Script guard (don’t run on include, only when executed) ----------
if abspath(PROGRAM_FILE) == @__FILE__
    main(
        time_file="C:\\tubai\\Downloads\\tumor_time_to_event_data.csv",
        immune_rate_file="C:\\tubai\\Downloads\\tumor_volume_vs_Im_cells_rate.csv",
        save_plots=true
    )
end
