using DifferentialEquations, Lux
using DiffEqFlux
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using ComponentArrays: ComponentArray
using Random
using Statistics
using CairoMakie

const RESULTS_PATH = "inventory_dynamics_project/src/images/UDE/poisson"

# Order of variables - IOD - inventory level, order rate, customer demand rate
# tau - inventory adjustment delay
# alpha - responsiveness to inventory deviation
# I_target - target inventory level

# Classical Bullwhip model to generate synthetic  
function classical_bullwhip_model!(du, u, p, t)
    # u = [I, O, D]
    tau, alpha, I_target, kappa, mu = p
    du[1] = u[2] - u[3]                                              # dI/dt
    du[2] = (1 / tau) * (u[3] + alpha * (I_target - u[1]) - u[2])    # dO/dt
    du[3] = kappa*(mu - u[3]) + (rand() < 0.2 ? 15 : 0)
end

# tau, alpha, I_target, kappa, mu
p_phys = [8.0, 12.0, 100.0, 0.02, 10.0]
# initial state: I, O, D
u0      = [60.0, 10.0, 20.0]
tspan   = (0.0, 25.0)

rng = Random.default_rng()
Random.seed!(1234)

prob_data = ODEProblem(classical_bullwhip_model!, u0, tspan, p_phys)
sol_data  = solve(prob_data, Tsit5(), saveat = 0.2)

data = Array(sol_data)
tvec = sol_data.t

fig = Figure()
ax  = Axis(fig[1,1]; xlabel = "Time", ylabel = "", title = "ODE data")
lines!(ax, tvec, data[1, :]; label = "I")
lines!(ax, tvec, data[2, :]; label = "O")
lines!(ax, tvec, data[3, :]; label = "D")
axislegend(ax)
save(joinpath(RESULTS_PATH, "ODE_data.png"), fig)

###############################################################################
# Training subset (short horizon)

tspan_train = (0.0, 8.0)
idx_train   = findall(t -> t <= tspan_train[2], tvec)
tsteps      = tvec[idx_train]
data_train  = data[:, idx_train]

###############################################################################
# Neural net for ordering control term
NN_order = Lux.Chain(
    Lux.Dense(3, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, 1)
)

p_net, st_net = Lux.setup(rng, NN_order)
theta0 = ComponentArray((nn = p_net,))

###############################################################################
# UDE
function bullwhip_ude!(du, u, theta, t)
    tau, alpha, I_target, kappa, mu = p_phys

    y, _ = Lux.apply(NN_order, u, theta.nn, st_net)  # y is a 1-element array
    ctrl = only(y)

    du[1] = u[2] - u[3]
    du[2] = (1 / tau) * (u[3] + alpha * (I_target - u[1]) - u[2]) + ctrl
    du[3] = kappa * (mu - u[3])   # using deterministic dD here for training
end

prob_ude = ODEProblem(bullwhip_ude!, u0, tspan_train, theta0)

###############################################################################
# Loss and prediction
function predict(theta)
    Array(solve(prob_ude;
        p = theta,
        saveat = tsteps,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
end

function loss(theta)
    pred = predict(theta)
    return sum(abs2, data_train .- pred) / length(tsteps)
end

###############################################################################
# Optimisation (ADAM -> BFGS)
losses = Float64[]
cb = function (theta, l)
    push!(losses, l)
    return false
end

adtype = Optimization.AutoZygote()
optf   = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, theta0)

res1 = Optimization.solve(optprob, OptimizationOptimisers.ADAM(1e-3);
                          maxiters = 8000, callback = cb)

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, OptimizationOptimJL.BFGS();
                          maxiters = 500, allow_f_increases = false, callback = cb)

println("Final training loss = ", loss(res2.u))

###############################################################################
# Extrapolating for full horizon
prob_ude_full = remake(prob_ude; tspan = tspan, p = res2.u)
sol_ude       = solve(prob_ude_full, Tsit5(), saveat = 0.2)
pred_full     = Array(sol_ude)

###############################################################################
# Plots

# Training fit
fig1 = Figure()
ax1 = Axis(fig1[1, 1]; title = "Training window fit", xlabel = "time", ylabel = "state")
scatter!(ax1, tsteps, data_train[1, :]; color = :black, label = "I data")
scatter!(ax1, tsteps, data_train[2, :]; color = :gray, label = "O data")
scatter!(ax1, tsteps, data_train[3, :]; color = :orange, label = "D data")
lines!(ax1, tsteps, predict(res2.u)[1, :]; color = :red,   label = "I UDE")
lines!(ax1, tsteps, predict(res2.u)[2, :]; color = :green, label="O UDE")
lines!(ax1, tsteps, predict(res2.u)[3, :]; color = :blue, label="D UDE")
axislegend(ax1; position = :rb)
save(joinpath(RESULTS_PATH, "train_fit.png"), fig1)

# Loss curve
fig2 = Figure()
ax2 = Axis(fig2[1, 1];
    title = "Loss history",
    xlabel = "iteration",
    ylabel = "loss",
    xscale = Makie.log10,
    yscale = Makie.log10
)
lines!(ax2, 1:length(losses), losses)
save(joinpath(RESULTS_PATH, "losses.png"), fig2)

# Extrapolation to rest of the time span
fig3 = Figure()
ax3 = Axis(fig3[1, 1]; title = "Full extrapolation", xlabel = "time", ylabel = "I, O, D")
scatter!(ax3, tvec, data[1, :]; color = :black, label = "I data")
lines!(ax3, tvec, pred_full[1, :]; color = :red, label = "I UDE")
vlines!(ax3, tspan_train[2]; linestyle = :dash)
axislegend(ax3; position = :rb)
save(joinpath(RESULTS_PATH, "extrapolation.png"), fig3)

println("Finished.")
