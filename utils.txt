"""
This type stores the parameters of the economy and some technical parameters.
##### Fields
* `β1::Float64`: Coefficient of Beta distribution of worker types
* `β2::Float64`: Coefficient of Beta distribution of worker types
* `c0::Float64`: Scale of cost opportunity posting function
* `c1::Float64`: Exponent on v of cost opportunity posting function 
* `α::Float64`: Scale of meeting function
* `ω::Float64`: Exponent on L of meeting function
* `r::Float64`: Interest rate
* `δ::Float64`: Exogenous match distruction rate

* `f0::Float64`: Price scale (used for GDP scaling)
* `p1::Float64`: Constant in production function
* `p2::Float64`: Coefficient in production function
* `p3::Float64`: Coefficient in production function
* `p4::Float64`: Coefficient in production function
* `p5::Float64`: Coefficient in production function
* `p6::Float64`: Coefficient in production function

* `σ::Float64`: Volatility of z
* `ρ::Float64`: AR of z
* `s::Float64`: Search intensity of employed

* `dt::Float64`: Length of one time period
* `ϵ::Float64`: Distance of left/right grid element to boundary
* `Nx::Int64`: Number of grid points in x
* `Ny::Int64`: Number of grid points in y
* `Nz::Int64`: Number of grid points in z
* `Gridx::Vector{Float64}`: Grid on x
* `Gridy::Vector{Float64}`: Grid on y
* `Gridz::Vector{Float64}`: Grid on z
* `Grida::Vector{Float64}`: Grid on a, the random variable of the copula
"""
@with_kw struct LiseRobinModel
    #economy parameters
    β1::Float64
    β2::Float64
    c0::Float64
    c1::Float64
    α::Float64
    ω::Float64
    r::Float64
    δ::Float64

    f0::Float64
    p1::Float64
    p2::Float64
    p3::Float64
    p4::Float64
    p5::Float64
    p6::Float64

    σ::Float64
    ρ::Float64
    s::Float64

    #technical
    dt::Float64
    ϵ::Float64

    Nx::Int64
    Ny::Int64
    Nz::Int64
    Gridx::Vector{Float64}
    Gridy::Vector{Float64}
    Gridz::Vector{Float64}
    Grida::Vector{Float64}
end


"""
    LiseRobinModel()
    
This is a constructor of a LiseRobinModel with parameters from table 2, unless otherwise specified.
"""
function LiseRobinModel(;β1=2.148, β2=12.001, c0=0.028, c1=0.0844501681, 
        α=0.497, ω=1/2, δ=0.013,
        f0=(1.0/0.398401314845053)/0.412335720378949, p1=0.003, p2=2.053, p3=-0.140, p4=8.035, p5=-1.907, p6=6.596, 
        σ=0.071, ρ=0.999, s=0.027, dt=1/52, ϵ=0.001, Nx=21, Ny=21, Nz=51)
    
    Gridx = collect(linspace(ϵ, 1-ϵ, Nx))
    Gridy = collect(linspace(ϵ, 1-ϵ, Ny))
    Grida = collect(linspace(ϵ, 1-ϵ, Nz))
    Gridz = collect(quantile.(Normal(), Grida))
    
    r = 1.05^dt - 1.0
    
    #the cost scale is GDP adjusted
    return LiseRobinModel(β1, β2, c0*f0, c1, α, ω, r, δ, f0, p1, p2, p3, p4, p5, p6, σ, ρ, s, dt, ϵ, Nx, Ny, Nz,
        Gridx, Gridy, Gridz, Grida)
end

"""
    solveS(model)

This function iterates on equation (3) until convergence. 

##### Returns

* `S::Array{Float64,3}`: Surplus as a function of x, y, z
* `B::Array{Float64,1}`: Value of unemploymend as a function of x
* `P::Array{Float64,3}`: Value of a match as a function of x, y, z
* `l::Array{Float64,1}`: Worker density as a function of x
* `p::Array{Float64,3}`: Production as a function of x, y, z
* `b::Array{Float64,1}`: Home production as a function of x
* `Speriod::Array{Float64,3}`: Period surplus as a function of x, y, z
* `Q::Array{Float64,2}`: Transition probabilities from z to z'

"""
function solveS(model)
    
    @unpack β1, β2, c0, c1, α, ω, r, δ, f0, p1, p2, p3, p4, p5, p6, σ, ρ, s, dt, ϵ, Nx, Ny, Nz,
        Gridx, Gridy, Gridz, Grida = model

    #worker-type distriution
    l = @. Gridx ^ (β1 - 1) * (1 - Gridx) ^ (β2 - 1)
    l = Nx * l / sum(l) #normalization from code

    #production function, 3d array in x, y, z
    p = [f0 * exp.(σ*z) * (p1 + p2*x + p3*y + p4*x^2 + p5*y^2 + p6 * x * y) * dt 
        for x in Gridx, y in Gridy, z in Gridz]

    #home production, 1d array in x
    b = 0.7 * [maximum(p[i, :, convert(Int,(Nz+1)/2)]) 
        for i in 1:length(Gridx)]
        
    #value of being unemployed, 1d array in x
    B = b * (1 + 1/(1+r))

    #(period) surplus function initialized at deterministic steady state, 3d array in x, y, z
    Speriod = [(p[i, j, k] - b[i]) 
        for i in 1:length(Gridx), j in 1:length(Gridy), k in 1:length(Gridz)]
    S = (1 + r)/(r + δ) * Speriod

    #transition probabilities for z 
    Q = @. exp(-1/(2*(1 - ρ^2)) * (Gridz^2 + Gridz'^2 - 2 * ρ * Gridz * Gridz'))
    Q = Q ./ sum(Q, 2);
    #DONE: numerical cancellation in tails, but mean reversion mitigates 
    
    #iterate on equation (3)
    error = 1.
    iteration = 0

    Sold = similar(S)
    copy!(Sold, S)
    
    while (error > 1e-8)
         
        for j in 1:Ny
            @views S[:, j, :] = Sold[:, j, :] * Q
        end
        
        S = Speriod + (1 - δ)/(1 + r) * S
        error = maximum(abs.(S - Sold))
        copy!(Sold, S)
        iteration += 1
    end
    
    #value of a match
    P = S .+ B

    return (S, B, P, l, p, b, Speriod, Q)
end

"""
    fixedpointh(model, S, l)

This function iterates on equation (9) until convergence.

##### Returns

* `h::Vector{Float64}`: Distribution of employed workers as a function of x, y

"""
function fixedpointh(model, S0, l)

    @unpack β1, β2, c0, c1, α, ω, r, δ, f0, p1, p2, p3, p4, p5, p6, σ, ρ, s, dt, ϵ, Nx, Ny, Nz,
            Gridx, Gridy, Gridz, Grida = model

    #deterministic steady state employed distribution, 2d array in x, y
    h = repmat(copy(l)', Ny)'
    hold = copy(h) 

    error = 1.
    iteration = 0
    
    M = 0
    L = 0
    v = zeros(Ny)

    while (error > 1e-5)

        #after terminations
        hplus = (S0 .>= 0) .* ((1-δ) * hold)

        #after terminations
        uplus = max.(l .- mean(hplus, 2), 0)

        #search effort
        L = mean(uplus) + s * mean(hplus)

        #value of a filled job to the firm
        J = 1/L * mean(max.(S0, 0) .* uplus, 1)[1, :] .+ [s/L * mean(hplus .* max.(S0[:, y] .- S0, 0)) for y in 1:Ny]

        #tightness
        θ = (1/L * mean((α * J / c0).^(1/c1)))^(c1/(c1+ω))

        #aggregate opportunities posted
        V = θ * L

        #meetings
        M = min(α * L^ω * V^(1-ω), L, V)

        #opportunities posted
        v = (M/V * (J/c0)).^(1/c1) #DONE: validate that v integrates to V

        #meeting probability for firms
        q = M/V

        #next period's employment distribution, see equation (9)
        h = max.(0, hplus - #last period
            s*q/L * hcat([mean(v' .* (S0 .> S0[:, y]) .* (S0 .>= 0), 2) for y in 1:Ny]...) + #poached from y
            s*q/L * hcat([mean(hplus * v[y] .* (S0[:, y] .> S0) .* (S0 .>= 0), 2) for y in 1:Ny]...) + #poached by y
            uplus .* q/L .* v' .* (S0 .>= 0)) #hires from unemployment

        error = maximum(abs.(h - hold))
        hold = copy(h)
        iteration += 1
    end
    
    #the deterministic steady state wage
    w = b .- (1-δ)/(1+r)*s*M/L .* (S0 .>= 0) .* (S0 + hcat([mean(max.(0, P[:, :, convert(Int,(model.Nz+1)/2)] .- 
        P[:, y, convert(Int,(model.Nz+1)/2)]) .* v', 2)[:] for y in 1:Ny]...))
    
    return (h, w)
end


"""
    simulateh(model, S, h0, Q)

This function simulates 600 years and estimates stationary distributions and flow masses.

##### Returns
(sim_h, sim_J2J, sim_E2U, sim_HfU, sim_HfE)

* `sim_h::Array{Float64,2}`: Distribution of employed workers as a function of x, y
* `sim_J2J::Array{Float64,2}`: Flow mass of job-to-job transitions as a function of x, y
* `sim_E2U::Array{Float64,2}`: Flow mass of unemployed-to-employment transitions as a function of x, y
* `sim_HfU::Array{Float64,2}`: Flow mass of hires from unemployment as a function of x, y
* `sim_HfE::Array{Float64,2}`: Flow mass of poached hires as a function of x, y

"""
function simulateh(model, S, h0, Q)
    @unpack β1, β2, c0, c1, α, ω, r, δ, f0, p1, p2, p3, p4, p5, p6, σ, ρ, s, dt, ϵ, Nx, Ny, Nz,
                Gridx, Gridy, Gridz, Grida = model

    #700 years of simulation, 100 years of burn-in
    T = floor(Int64, 700 / dt)
    burninT = floor(Int64, 100 / dt)
    #time series of index of productivity
    ts_z = zeros(Int64, T)
    #z starts at neutral aggregate productivity
    ts_z[1] = convert(Int64, (model.Nz+1)/2)
    #the CDF of Q
    Q_cdf = cumsum(Q, 2)
    srand(123567)

    for t in 2:T
        ts_z[t] = minimum(find(rand(Uniform()) .< Q_cdf[ts_z[t-1], :]))
    end


    #initialize, these arrays will be the sum of masses and are to be devided by the number of z's that occured
    sim_h = zeros(Nx, Ny, Nz)
    sim_J2J = zeros(Nx, Ny, Nz)
    sim_E2U = zeros(Nx, Ny, Nz)
    sim_HfU = zeros(Nx, Ny, Nz)
    sim_HfE = zeros(Nx, Ny, Nz)
    
    #initialize at steady state, beginning of period variables
    h = copy(h0)
    J2J = zeros(Nx, Ny)
    hplus = zeros(Nx, Ny)
    HfU = zeros(Nx, Ny)
    HfE = zeros(Nx, Ny)
    Scurrent = zeros(Nx, Ny)
    
    tic()
    @inbounds for t in 2:T
        
        #the compiler cannot prove type stability, apparently, so I help him
        Scurrent::Array{Float64, 2} = S[:, :, ts_z[t]]
        hplus::Array{Float64, 2} = ((Scurrent .>= 0) .* ((1-δ) * h))
        uplus::Array{Float64, 2} = max.(l .- mean(hplus, 2), 0)

        L = mean(uplus) + s * mean(hplus)

        J = 1/L * mean(max.(Scurrent, 0) .* uplus, 1)[1, :]
        for y in 1:Ny
            J[y] += s/L * mean(hplus .* max.(Scurrent[:, y] .- Scurrent, 0))
        end
            
        θ = (1/L * mean((α * J / c0).^(1/c1)))^(c1/(c1+ω))

        V = θ * L

        M = min(α * L^ω * V^(1-ω), L, V)

        if (V >= 1e-3)
            v = (M/V * (J/c0)).^(1/c1)
            q = M/V
        else
            v = zeros(Ny)
            q = 0.
        end

        for y in 1:Ny
            J2J[:, y] = s*q/L * mean(v' .* (Scurrent .> Scurrent[:, y]) .* (Scurrent .>= 0), 2)
            HfE[:, y] = s*q/L * mean(hplus * v[y] .* (Scurrent[:, y] .> Scurrent) .* (Scurrent .>= 0), 2)
        end
        HfU = (uplus .* q/L .* v' .* (Scurrent .>= 0))

        if (t >= burninT) #save
            sim_h[:, :, ts_z[t]] += h
            sim_J2J[:, :, ts_z[t]] += J2J
            sim_E2U[:, :, ts_z[t]] += h - hplus
            sim_HfU[:, :, ts_z[t]] += HfU
            sim_HfE[:, :, ts_z[t]] += HfE
        end

        h::Array{Float64, 2} = max.(0, hplus - #last period
                    J2J + #poached from y
                    HfE + #poached by y
                    HfU) #hires from unemployment
    end

    #how often did each z occur
    occurence = [count(ts_z[burninT:end] .== z) for z in 1:Nz]
    occurence[occurence .== 0] = typemax(Int64)

    sim_h = permutedims(permutedims(sim_h, [3, 1, 2]) ./ occurence, [2, 3, 1])
    sim_J2J = permutedims(permutedims(sim_J2J, [3, 1, 2]) ./ occurence, [2, 3, 1])
    sim_E2U = permutedims(permutedims(sim_E2U, [3, 1, 2]) ./ occurence, [2, 3, 1])
    sim_HfU = permutedims(permutedims(sim_HfU, [3, 1, 2]) ./ occurence, [2, 3, 1])
    sim_HfE = permutedims(permutedims(sim_HfE, [3, 1, 2]) ./ occurence, [2, 3, 1])

    return (ts_z, sim_h, sim_J2J, sim_E2U, sim_HfU, sim_HfE)
end