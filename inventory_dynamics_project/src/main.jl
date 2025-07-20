using DifferentialEquations, Plots

const RESULTS_PATH = "inventory_dynamics_project/src/images"

# Order of variables - IOD - inventory level, order rate, customer demand rate
# tau - inventory adjustment delay
# alpha - responsiveness to inventory deviation
# I_target - target inventory level

# Classical Bullwhip model to generate synthetic  
function classical_bullwhip_model!(du, u, p, t)
    tau, alpha, I_target = p
    du[1] = u[2] - u[3]
    du[2] = (1/tau) * (u[3] + alpha * (I_target - u[1]) - u[2])
    du[3] = 0 # assuming constant demand 
end

p0 = (tau = 0.3, alpha = 2.5, I_target = 10.0)
u0 = [2 , 8 , 3]
tspan = (0.0, 5.0)

prob = ODEProblem(classical_bullwhip_model!, u0, tspan, p0)
sol = solve(prob, Tsit5(), saveat=0.2)

data = Array(sol)
t = sol.t

plot(data[1, :], xlabel="Time", ylabel="", label="Inventory Level (I)")
plot!(data[2, :], label="Order Rate (O)")
plot!(data[3, :], label="Customer Demand Rate (D)")
savefig(joinpath(RESULTS_PATH, "ODE_data.png"))
