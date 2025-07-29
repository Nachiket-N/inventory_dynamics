using DifferentialEquations, Lux
using DiffEqFlux
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using ComponentArrays: ComponentArray
using Random
using Statistics
using CairoMakie

const RESULTS_PATH = "inventory_dynamics_project/src/images/NODE"

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
    du[3] = kappa * (mu - u[3]) + 5 * randn() * sin(2 * pi * t)      # dD/dt (noisy for data gen)
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
idx_train   = findall(t -> t ≤ tspan_train[2], tvec)
tsteps      = tvec[idx_train]
data_train  = data[:, idx_train]

###############################################################################
# Defining the Neural Network

small_init(rng, dims...) = 0.05f0 .* (2 .* rand(rng, Float32, dims) .- 1)
NN_dynamics = Lux.Chain(
    Lux.Dense(3, 64, leakyrelu, init_weight=small_init),
    Lux.Dense(64, 64, leakyrelu, init_weight=small_init),
    Lux.Dense(64, 3),                 # output dI/dt, dO/dt, dD/dt
    x -> 0.1f0* x
)

p_net, st_net = Lux.setup(rng, NN_dynamics)
theta0 = ComponentArray(p_net)

###############################################################################
# Defining Neural ODE Problem

neuralode_train = NeuralODE(NN_dynamics, tspan_train, Tsit5(); saveat = tsteps)

predict = theta -> Array(neuralode_train(u0, theta, st_net)[1])

function loss(theta)
  out = try
    neuralode_train(u0, theta, st_net)
  catch
    return Inf
  end

  Y = Array(out[1])
  # if it didn’t reach all Nt points, give it infinite loss
  size(Y,2) == length(tsteps) || return Inf

  return sum(abs2, data_train .- Y) / length(tsteps)
end

###############################################################################
# Optimization  (ADAM  →  BFGS)

losses = Float64[]
cb = (theta, l) -> (@show loss = l; push!(losses, l); false)

adtype   = Optimization.AutoZygote()
opt_fun  = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
opt_prob = Optimization.OptimizationProblem(opt_fun, theta0)

res_adam = Optimization.solve(opt_prob, OptimizationOptimisers.ADAM(2e-3);
                              maxiters = 3000, callback = cb)

opt_prob2 = Optimization.OptimizationProblem(opt_fun, res_adam.u)
res_bfgs  = Optimization.solve(opt_prob2, OptimizationOptimJL.BFGS();
                               maxiters = 300, allow_f_increases = false,
                               callback = cb)

println("Final training loss = ", loss(res_bfgs.u))

# Extrapolating for full horizon
neuralode_full = NeuralODE(NN_dynamics, tspan, Tsit5(); saveat = tvec)
pred_full      = Array(neuralode_full(u0, res_bfgs.u, st_net)[1])

###############################################################################
# Plots

# Training‑window fit
fig1 = Figure()
ax1  = Axis(fig1[1,1]; title = "Training window fit (Neural ODE)",
            xlabel = "time", ylabel = "state")
scatter!(ax1, tsteps, data_train[1, :]; color = :black,  label = "I data")
scatter!(ax1, tsteps, data_train[2, :]; color = :gray,   label = "O data")
scatter!(ax1, tsteps, data_train[3, :]; color = :orange, label = "D data")
lines!(ax1, tsteps, predict(res_bfgs.u)[1, :]; color = :red,   label = "I NODE")
lines!(ax1, tsteps, predict(res_bfgs.u)[2, :]; color = :green, label = "O NODE")
lines!(ax1, tsteps, predict(res_bfgs.u)[3, :]; color = :blue,  label = "D NODE")
axislegend(ax1; position = :rb)
save(joinpath(RESULTS_PATH, "train_fit_NODE.png"), fig1)

# (b) Loss‑curve (log–log)
fig2 = Figure()
ax2  = Axis(fig2[1,1]; title = "Loss history (Neural-ODE)",
            xlabel = "iteration", ylabel = "loss",
            xscale = Makie.log10, yscale = Makie.log10)
lines!(ax2, 1:length(losses), losses)
save(joinpath(RESULTS_PATH, "losses_NODE.png"), fig2)

# (c) Extrapolation across the full 0–25 s horizon
fig3 = Figure()
ax3  = Axis(fig3[1,1]; title = "Full extrapolation (Neural-ODE)",
            xlabel = "time", ylabel = "I, O, D")
scatter!(ax3, tvec, data[1, :]; color = :black, label = "I data")
lines!(ax3,  tvec, pred_full[1, :]; color = :red, label = "I NODE")
vlines!(ax3, tspan_train[2]; linestyle = :dash)
axislegend(ax3; position = :rb)
save(joinpath(RESULTS_PATH, "extrapolation_NODE.png"), fig3)

println("Finished")
