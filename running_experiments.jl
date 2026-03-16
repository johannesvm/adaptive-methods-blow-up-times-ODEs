using Quadmath
using Integrals
using CairoMakie
CairoMakie.activate!(type = "pdf")
using LaTeXStrings
using LinearAlgebra
using SpecialFunctions
using DifferentialEquations

include("methods.jl")

function x_sqrd()
    """
        x_sqrd()

    Run and plot the results of a numerical experiment where b(x) = x^2. 
    """
    x_0 = 1/2
    b = x -> x^2
    b_der = x -> 2*x
    F_inv = x -> 1/x^2
    true_solution = 1/x_0
    h_uni = (eps, r) -> min(eps/log(b(r)/b(x_0)), 1/(2*b_der(r)))
    eps_start = -10; eps_stop = -22
    epsilons = [2.0^i for i = eps_start:-1:eps_stop]

    methods = Dict(["Adaptive" => eps -> adaptive_method_1D(b, b_der, x_0, eps, F_inv),
                    "Adaptive HO" => eps -> adaptive_method_1D(b, b_der, x_0, eps, F_inv,
                                                               1.1, nothing, true),
                    "Uniform" => eps -> uniform_method_1D(b, x_0, eps, h_uni, F_inv),
                    "Hirota & Ozawa" => eps -> hirota_ozawa(x -> [x[1]^2], 2, [x_0], 4, 
                                                            1.1, 
                                                            -6-convert(Int, log2(eps))),
                    "Stuart & Floater" => eps -> stuart_floater_x_sqrd(x_0, b_der, 
                                                                       x -> x/b(x), eps),
                    "Modified rescaling" => eps -> mod_rescaling(x_0, b, b_der, 1, 1, 1/8, 
                                                                 2, eps, false, x -> x)
                    ])
    errors = Dict([(name, []) for name in keys(methods)])
    costs = Dict([(name, []) for name in keys(methods)])
    for eps in epsilons
        for (name, method) in methods
            tau, cost = method(eps)
            push!(errors[name], abs(tau - true_solution))
            push!(costs[name], cost)
        end
    end
    rs1 = 0.5; rs2 = 1.75; ord_adp_lin = ord_uni_log = ho_scheme = true
    ord_uni_quad = false
    fig = plot_data(methods, costs, errors, epsilons, rs1, ord_adp_lin, rs2, ord_uni_log,
                    ord_uni_quad, ho_scheme)
    save("./figures/x_sqrd.pdf", fig)
end

function exp_x_2()
    """
        exp_x_2()

    Run and plot the results of a numerical experiment where b(x) = exp(x^2). 
    """
    x_0 = 1
    b = x -> exp(x^2)
    b_der = x -> 2*x*exp(x^2)
    F_inv = nothing
    h_uni = (eps, r) -> min(eps/log(b(r)/b(x_0)), 1/(2*b_der(r)))
    eps_start = -17; eps_stop = -25
    epsilons = [2.0^i for i = eps_start:-1:eps_stop]
    pseudo_sol = adaptive_method_1D(b, b_der, x_0, 2.0^(eps_stop-8), F_inv)[1]
    methods = Dict(["Adaptive" => eps -> adaptive_method_1D(b, b_der, x_0, eps, F_inv),
                    "Uniform" => eps -> uniform_method_1D(b, x_0, eps, h_uni, F_inv, 
                                                          nothing, b_der),
                    "Hirota & Ozawa" => eps -> hirota_ozawa(x -> [exp(x[1]^2)], 2, [x_0],
                                                            1.1, 1.1,                                    
                                                            -16-convert(Int, log2(eps)))
                    ])
    errors = Dict([(name, []) for name in keys(methods)])
    costs = Dict([(name, []) for name in keys(methods)])
    for eps in epsilons
        for (name, method) in methods
            tau, cost = method(eps)
            push!(errors[name], abs(tau - pseudo_sol))
            push!(costs[name], cost)
        end
    end
    rs1 = rs2 = 1.5; ord_adp_lin = ord_uni_log = true; ord_uni_quad = ho_scheme = false
    fig = plot_data(methods, costs, errors, epsilons, rs1, ord_adp_lin, rs2, ord_uni_log,
                    ord_uni_quad, ho_scheme)
    save("./figures/exp_x_2.pdf", fig)
end

function xlogx_c(c, eps_start, eps_stop)
    """
        xlogx_c(c, eps_start, eps_stop)

    Run and plot the results of a numerical experiment where b(x) = xlog(x)^(1+c).

    # Arguments
    - `c` : Parameter used to set exponent of log-factor in b(x).
    - `eps_start` : Parameter setting largest tolerance.
    - `eps_stop` : Parameter setting smallest tolerance.
    """
    x_0 = BigFloat("2", precision=256)
    b = x -> x*log(x)^(1+c)
    b_der = x -> log(x)^(1+c) + (1+c)*log(x)^(c)
    r = eps -> exp((BigFloat("1", precision=256)*c*eps)^(-1/c))
    true_solution = 1/(c*log(x_0)^c)
    h_uni = (eps, r) -> eps^(1+1/c)
    epsilons = [2.0^i for i = eps_start:-1:eps_stop]
    methods = Dict(["Adaptive" => eps -> adaptive_method_1D(b, b_der, x_0, eps, nothing,
                                                            1.1, r(eps), false, c),
                    "Uniform" => eps -> uniform_method_1D(b, x_0, eps, h_uni, nothing, 
                                                          r(eps)),
                    "Hirota & Ozawa" => eps -> hirota_ozawa(x -> [x[1]*log(x[1])^(2)], 2,
                                                            [x_0], 16, 7,                                    
                                                            12-convert(Int, log2(eps)))
                    ])
    errors = Dict([(name, []) for name in keys(methods)])
    costs = Dict([(name, []) for name in keys(methods)])
    for eps in epsilons
        println(log2(eps))
        for (name, method) in methods
            tau, cost = method(eps)
            push!(errors[name], abs(tau - true_solution))
            push!(costs[name], cost)
        end
    end
    fig = plot_data_xlogx(costs, errors, epsilons, c)
    save("./figures/xlogx_$(c).pdf", fig)
end

function simple_multidim()
    """
        simple_multidim()

    Run and plot the results of a numerical experiment with a simple uncoupled rhs. 
    """
    x_0 = [2^(1/2), 1] 
    alpha = 2; gamma = 4; C_c = 1
    b = x -> [x[1]^3, x[2]^5]
    b_der = x -> [3*x[1]^2 0; 0 5*x[2]^4]
    h_uni = (eps, r) -> eps^(gamma/alpha)
    h_uni_log = (eps, r) -> 1/4*eps/log(r/norm(x_0))
    true_solution = min(1/(2*x_0[1]^2), 1/(4*x_0[2]^4))
    eps_start = -14; eps_stop = -20
    epsilons = [2.0^i for i = eps_start:-1:eps_stop]

    methods = Dict(["Adaptive" => eps -> adaptive_method_nD(b, b_der, x_0, C_c, alpha, 
                                                            eps),
                    "Uniform" => eps -> uniform_method_nD(b, x_0, C_c, alpha, eps, h_uni),
                    "Log Uniform" => eps -> uniform_method_nD(b, x_0, C_c, alpha, eps, 
                                                              h_uni_log),
                    "Hirota & Ozawa" => eps -> hirota_ozawa(x -> b(x), 3, x_0, 1.5, 1.75,
                                                            -6-convert(Int, log2(eps)))
                    ])
    errors = Dict([(name, []) for name in keys(methods)])
    costs = Dict([(name, []) for name in keys(methods)])
    for eps in epsilons
        for (name, method) in methods
            tau, cost = method(eps)
            push!(errors[name], abs(tau - true_solution))
            push!(costs[name], cost)
        end
    end
    rs1 = 1.5; rs2 = 1.5; ord_uni_log = false; ord_adp_lin = ord_uni_quad = true
    fig = plot_data(methods, costs, errors, epsilons, rs1, ord_adp_lin, rs2, ord_uni_log,
                    ord_uni_quad)
    save("./figures/simple_multidim.pdf", fig)
end

function coupled_problem(pseudo_steps=10)
    """
        coupled_problem(pseudo_steps=10)

    Run and plot the results of a numerical experiment with a simple uncoupled rhs. 

    # Arguments
    - `pseudo_steps` : Parameter setting tolerance used for pseudo solution (default: `10`).
    """
    x_0 = [1, 2] 
    alpha = 2; C_c = 1
    b = x -> [x[1]^3 + x[1]*x[2]^2, x[2]^3 + x[1]^2*x[2]]
    b_der = x -> [3*x[1]^2+x[2]^2 2*x[1]*x[2]; 2*x[1]*x[2] x[1]^2+3*x[2]^2]
    h_uni = (eps, r) -> 1/1.75*eps/log(r/norm(x_0))
    eps_start = -7; eps_stop = -17
    epsilons = [2.0^i for i = eps_start:-1:eps_stop]
    pseudo_solution = adaptive_method_nD(b, b_der, x_0, C_c, alpha, 
                                         2.0^(eps_stop-pseudo_steps))[1]

    methods = Dict(["Adaptive" => eps -> adaptive_method_nD(b, b_der, x_0, C_c, alpha,
                                                            eps),
                    "Uniform" => eps -> uniform_method_nD(b, x_0, C_c, alpha, eps, h_uni),
                    "Hirota & Ozawa" => eps -> hirota_ozawa(x -> b(x), 3, x_0, 1.5, 1.75, 
                                                            -6-convert(Int, log2(eps)))
                    ])
    errors = Dict([(name, []) for name in keys(methods)])
    costs = Dict([(name, []) for name in keys(methods)])
    for eps in epsilons
        for (name, method) in methods
            tau, cost = method(eps)
            push!(errors[name], abs(tau - pseudo_solution))
            push!(costs[name], cost)
        end
    end
    rs1 = 1.5; rs2 = 1.5; ord_uni_log = true; ord_adp_lin = true; ord_uni_quad = false
    fig = plot_data(methods, costs, errors, epsilons, rs1, ord_adp_lin, rs2, ord_uni_log,
                    ord_uni_quad)
    save("./figures/coupled_problem.pdf", fig)
end

function log_multidim(c, eps_start=-7, eps_stop=-13, pseudo_steps=4, h_uni_factor=1/2.25)
    """
        log_multidim(c, eps_start=-7, eps_stop=-13, pseudo_steps=4, h_uni_factor=1/2.25)

    Run and plot the results of a numerical experiment where the rhs is a system with
    log-factors.

    # Arguments
    - `c` : Parameter used to set exponent of log-factor in b(x).
    - `eps_start` : Parameter setting largest tolerance (default: `-7`).
    - `eps_stop` : Parameter setting smallest tolerance (default: `-13`).
    - `pseudo_steps` : Parameter setting tolerance used for pseudo solution (default: `4`).
    - `h_uni_factor` : Factor multiplied with uniform step length (default: `1/2.25`).
    """
    x_0 = [BigFloat("4", precision=256), BigFloat("3", precision=256)] 
    C_c = 1
    b = x -> [x[1]*log(x[1]^2 + 2*x[2]^2)^(c+1), x[2]*log(2*x[1]^2 + x[2]^2)^(c+1)]
    b_der = x -> [log(x[1]^2+2*x[2]^2)^(c+1)+2*(1+c)*x[1]^2/(x[1]^2+2*x[2]^2)*log(x[1]^2+2*x[2]^2)^c 4*(c+1)*x[1]*x[2]/(x[1]^2+2*x[2]^2)*log(x[1]^2+2*x[2]^2)^c; 
                  4*(c+1)*x[1]*x[2]/(2*x[1]^2+x[2]^2)*log(2*x[1]^2+x[2]^2)^c log(2*x[1]^2+x[2]^2)^(c+1)+2*(1+c)*x[2]^2/(2*x[1]^2+x[2]^2)*log(2*x[1]^2+x[2]^2)^c]
    h_uni = (eps, r) -> h_uni_factor*eps/log(r)
    epsilons = [2.0^i for i = eps_start:-1:eps_stop]
    alp = BigFloat("$c", precision=256)
    pseudo_solution = adaptive_method_nD(b, b_der, x_0, C_c, alp, 
                                         2.0^(eps_stop-pseudo_steps), c+1)[1]
    methods = Dict(["Adaptive" => eps -> adaptive_method_nD(b, b_der, x_0, C_c, alp, eps,
                                                            c+1),
                    "Uniform" => eps -> uniform_method_nD(b, x_0, C_c, alp, eps, h_uni,
                                                          true),
                    "Hirota & Ozawa" => eps -> hirota_ozawa(x -> b(x), 3, x_0, 16, 7, 
                                                            3-convert(Int, log2(eps)))
                    ])
    errors = Dict([(name, []) for name in keys(methods)])
    costs = Dict([(name, []) for name in keys(methods)])
    for eps in epsilons
        println(log2(eps))
        for (name, method) in methods
            println(name)
            tau, cost = method(eps)
            push!(errors[name], abs(tau - pseudo_solution))
            push!(costs[name], cost)
        end
    end
    fig = plot_data_xlogx(costs, errors, epsilons, c)
    save("./figures/log_multidim_$c.pdf", fig)
end

function reac_diff_refine_eps_table()
    """
        reac_diff_refine_eps_table()

    Run, print and plot the results of a numerical experiment with a reaction-diffusion 
    equation where we refine the tolerance.
    """
    n = 32; C_c = 1/(2*sqrt(32)); alpha = 1
    x_0 = [100*sin(pi*i/n) for i = 1:n-1]
    b = x -> vcat(n^2*(-2*x[1] + x[2]) + x[1]^2, 
                  [n^2*(x[k-1] - 2*x[k] + x[k+1]) + x[k]^2 for k = 2:n-2],
                  n^2*(x[n-2] - 2*x[n-1]) + x[n-1]^2)
    b_der = x -> vcat(transpose(vcat([-2*n^2 + 2*x[1], n^2], zeros(n-3))),
                      stack([vcat(zeros(k-2), [n^2, -2*n^2 + 2*x[k], n^2], 
                                  zeros(n-k-2)) for k = 2:n-2], dims=1), 
                      transpose(vcat(zeros(n-3), [n^2, -2*n^2 + 2*x[n-1]]))
                      )
    h_uni = (eps, r) -> 0.25*min(eps/(log(r/norm(x_0))), 1/(2*n^2))
    eps_start = -18; eps_stop = -25
    epsilons = [2.0^i for i = eps_start:-1:eps_stop]
    pseudo_sol = adaptive_method_nD(b, b_der, x_0, C_c, alpha, 2.0^(eps_stop-3), nothing, 
                                    n)[1]

    methods = Dict(["Adaptive" => eps -> adaptive_method_nD(b, b_der, x_0, C_c, alpha, eps,
                                                            nothing, n),
                    "Uniform" => eps -> uniform_method_nD(b, x_0, C_c, alpha, eps, h_uni),
                    "Hirota & Ozawa" => eps -> hirota_ozawa(x -> b(x), n, x_0, 2^16, 2, 
                                                            -13-convert(Int, log2(eps))
                    )
                    ])
    ests = Dict([(name, []) for name in keys(methods)])
    errors = Dict([(name, []) for name in keys(methods)])
    costs = Dict([(name, []) for name in keys(methods)])
    for eps in epsilons
        println(log2(eps))
        for (name, method) in methods
            tau, cost = method(eps)
            println(name)
            println(tau)
            push!(ests[name], tau)
            push!(errors[name], abs(tau - pseudo_sol))
            push!(costs[name], cost)
        end
    end
    print_table(ests, costs, epsilons, "eps")
end

function reac_diff_refine_x_table()
     """
        reac_diff_refine_x_table()

    Run and print the results of a numerical experiment with a reaction-diffusion equation
    where we refine the spatial dimension, increasing the number of equations in the system.
    """
    e = 2^(-23); C_c = 1/(2*sqrt(32)); alpha = 1;
    ns = [4, 8, 16, 32, 64, 128, 256, 512]
    methods = Dict(
        ["Adaptive" => (b, b_der, x_0, _, n) -> adaptive_method_nD(b, b_der, x_0, C_c, 
                                                                   alpha, e, nothing, n),
         "Uniform" => (b, b_der, x_0, h_uni, _) -> uniform_method_nD(b, x_0, C_c, alpha, e,
                                                                     h_uni)
         ])
    ests = Dict([(name, []) for name in keys(methods)])
    costs = Dict([(name, []) for name in keys(methods)])
    for n in ns
        x_0 = [100*sin(pi*i/n) for i = 1:n-1]
        h_uni = (eps, r) -> 0.25*min(eps/log(r/norm(x_0)), 1/(2*n^2))
        b = x -> vcat(n^2*(-2*x[1] + x[2]) + x[1]^2, 
                [n^2*(x[k-1] - 2*x[k] + x[k+1]) + x[k]^2 for k = 2:n-2], 
                n^2*(x[n-2] - 2*x[n-1]) + x[n-1]^2)
        b_der = x -> vcat(transpose(vcat([-2*n^2 + 2*x[1], n^2], zeros(n-3))),
                          stack([vcat(zeros(k-2), [n^2, -2*n^2 + 2*x[k], n^2],
                                 zeros(n-k-2)) for k = 2:n-2], dims=1), 
                          transpose(vcat(zeros(n-3), [n^2, -2*n^2 + 2*x[n-1]]))
                          )
        for (name, method) in methods
            tau, cost = method(b, b_der, x_0, h_uni, n)
            push!(ests[name], tau)
            push!(costs[name], cost)
        end
    end
    print_table(ests, costs, ns, "n")
end

x_sqrd()
exp_x_2()
xlogx_c(1, -6, -11)
xlogx_c(1/2, -4, -9)
simple_multidim()
coupled_problem()
log_multidim(1, -7, -13, 4, 1/1.75)
log_multidim(1/2, -5, -10, 3, 4)
reac_diff_refine_eps_table()
reac_diff_refine_x_table()
