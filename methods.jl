# Help functions 
norm = x -> sqrt(sum(x.^2))

function adaptive_method_1D(b, b_der, x_0, eps, F_inv=nothing, k=1.1, fix_r=nothing, 
                            ho=false, c=nothing)
    """
        adaptive_method_1D(b, b_der, x_0, eps, F_inv=nothing, k=1.1, fix_r=nothing,
                           ho=false, c=nothing) -> (t, cost)

    Implementation of adaptive method for estimating explosion time of 1D ODE.

    # Arguments
    - `b` : Function representing right-hand side of ODE.
    - `b_der` : Derivative of right-hand side.
    - `x_0` : Initial value of ODE.
    - `eps` : Tolerance used in computations.
    - `F_inv` : Inverse function of F, (default: `nothing`).
    - `k` : Value for parameter k (default: `1.1`).
    - `fix_r` : Manually seting the upper integration limit r (default: `nothing`).
    - `ho` : Higher-order scheme is used if true (default: `false`). 
    - `c` : Parameter c used when solving xlog(x)^(1+c) (default: `nothing`).

    # Returns
    - `t` : Estimate of explosion time.
    - `cost` : Number of iterations needed for computations.
    """
    t = 0; x = x_0
    if isnothing(fix_r)
        r = 2*max(x_0, 1)
        if !isnothing(F_inv)
            # Find r using F_inv
            while b(r) < F_inv(eps)
                r *= 1.001
            end
        elseif isnothing(F_inv)
            # Find r using that b'(r) = O(eps^alpha) (see Remark 2)
            while b_der(r) < log(1/eps)^2/eps
                r *= 1.001
            end
        end
    elseif !isnothing(fix_r)
        r = fix_r
    end
    cost = 0
    while x < r
        if !ho & isnothing(c)
            # Adaptive first-order scheme
            h = sqrt(eps^2/b_der(min(k*x, r)))
            x += b(x)*h
        elseif !isnothing(c)
            # Schemes for solving xlog(x)^(1+c)
            if c == 1
                h = sqrt(eps^2/(log(eps^(-1))^2*b_der(min(k*x, r))))
                x += b(x)*h
            else
                h = sqrt(eps*eps^(1/c)/b_der(min(k*x, r)))
                x += b(x)*h
            end
        else
            # Higher-order scheme
            h = eps^(1/2)/b_der(min(k*x, r))^(2/3)
            x += b(x)*h + 1/2*b(x)*b_der(x)*h^2
        end
        t += h
        cost += 1
    end
    return t, cost
end

function uniform_method_1D(b, x_0, eps, h_func, F_inv=nothing, fix_r=nothing, b_der=nothing)
    """
        uniform_method_1D(b, x_0, eps, h_func, F_inv=nothing, fix_r=nothing, b_der=nothing) 
                                                                               -> (t, cost)

    Function estimating explosion time of 1D ODE using uniform steps.

    # Arguments
    - `b` : Function representing right-hand side of ODE.
    - `x_0` : Initial value of ODE.
    - `eps` : Tolerance used in computations.
    - `h_func` : Function returning uniform step length used.
    - `F_inv` : Inverse function of F (default: `nothing`).
    - `fix_r` : Manually seting the upper integration limit r (default: `nothing`).
    - `b_der` : Derivative of right-hand side (default: `nothing`).

    # Returns
    - `t` : Estimate of explosion time.
    - `cost` : Number of iterations needed for computations.
    """
    t = 0; x = x_0
    if isnothing(fix_r)
        r = 2*max(x_0, 1)
        if !isnothing(F_inv)
            # Find r using F_inv
            while b(r) < F_inv(eps)
                r *= 1.001
            end
        elseif isnothing(F_inv)
            # Find r using that b'(r) = O(eps^alpha) (see Remark 2)
            while b_der(r) < log(1/eps)^2/eps
                r *= 1.001
            end
        end
    elseif !isnothing(fix_r)
        r = fix_r
    end
    h = h_func(eps, r)
    cost = 0
    while x < r
        x += b(x)*h
        t += h
        cost += 1
    end
    return t, cost
end

function adaptive_method_nD(b, b_der, x_0, C_c, alpha, eps, c=nothing, reac_m=nothing)
    """
        adaptive_method_nD(b, b_der, x_0, C_c, alpha, eps, c=nothing, reac_m=nothing) 
                                                                               -> (t, cost)

    Implementation of adaptive method for estimating explosion time of system of ODEs.

    # Arguments
    - `b` : Function representing right-hand side of ODE.
    - `b_der` : Derivative of right-hand side.
    - `x_0` : Initial value of ODE.
    - `C_c` : Parameter used in setting r.
    - `alpha` : Parameter used in setting r.
    - `eps` : Tolerance used in computations.
    - `c` : Parameter c used when solving systems in the setting of Assumption C
            (default: `nothing`).
    - `reac_m` : Number of equations in system of ODEs when solving reaction-diffusion
                 equation (default: `false`). 

    # Returns
    - `t` : Estimate of explosion time.
    - `cost` : Number of iterations needed for computations.
    """
    t = 0; x = x_0
    if !isnothing(c)
        # If in regular setting
        r = exp((1/(alpha*eps))^(1/alpha))
    else
        # If in setting of Assumption C
        r = (2/(C_c*alpha*eps))^(1/alpha)
    end
    cost = 0
    while norm(x) < r
        if !isnothing(reac_m)
            # If solving reaction-diffusion equation
            h = min(eps*sqrt(norm(b(x))/(norm(b_der(x)*b(x)))), 1/(2*reac_m^2))
        elseif isnothing(c) || (c > 2)
            # If in regular setting
            h = sqrt(eps^2/norm(b_der(x)))
        elseif c < 2
            # If solving problem in setting of Assumption C with c < 1
            h = eps^(1/2 + 1/(2*(c-1)))*sqrt(norm(b(x))/(norm(b_der(x)*b(x))))
        else
            # If solving problem in setting of Assumption C with c = 1
            h = eps*sqrt(norm(b(x))/(log(eps^(-1))^2*norm(b_der(x)*b(x))))
        end
        x += b(x)*h
        t += h
        cost += 1
    end
    return t, cost
end

function uniform_method_nD(b, x_0, C_c, alpha, eps, h_func, log_case=false)
    """
        uniform_method_nD(b, x_0, C_c, alpha, eps, h_func, log_case=false) -> (t, cost)

    Function estimating explosion time of system of ODEs using uniform steps.

    # Arguments
    - `b` : Function representing right-hand side of ODE.
    - `x_0` : Initial value of ODE.
    - `C_c` : Parameter used in setting r.
    - `alpha` : Parameter used in setting r.
    - `eps` : Tolerance used in computations.
    - `h_func` : Function returning uniform step length used.
    - `log_case` : Flag being true if in the setting of Assumption C (default: `false`).

    # Returns
    - `t` : Estimate of explosion time.
    - `cost` : Number of iterations needed for computations.
    """
    t = 0; x = x_0
    if log_case
        r = exp((1/(alpha*eps))^(1/alpha))
    else
        r = (2/(C_c*alpha*eps))^(1/alpha)
    end
    h = h_func(eps, r)
    cost = 0
    while norm(x) < r
        x += b(x)*h
        t += h
        cost += 1
    end
    return t, cost
end

function hirota_ozawa(b, n, x_0, s_0 = 16, gamma = 2, L = 11, k = 4)
    """
        hirota_ozawa(b, n, x_0=1, s_0=16, gamma=2, L=11, k=4) -> (t, cost)

    Function estimating explosion time of system of ODEs using the method by Hirota & Ozawa.

    # Arguments
    - `b` : Function representing right-hand side of ODE.
    - `n` : Number of equations in system + 1, giving the number of equations in the system
            for the arc length.
    - `x_0` : Initial value of ODE.
    - `s_0` : Initial arc length solving up to  (default: `16`).
    - `gamma` : Factor used to find length solving up to in next iteration (default: `2`).
    - `L` : Number of iterations, last length we solve for is s_0*gamma^L (default: `11`).
    - `k` : Number of recursions used for Aitken delta squared method (default: `4`).

    # Returns
    - `t` : Estimate of explosion time.
    - `cost` : Number of total time steps used in computations.
    """
    cost = 0
    # Right-hand side of system solving for the arc length
    function f!(ds, s, p, t)
        ds[1] = 1/sqrt(1+sum(b(s[2:n]).^2))
        ds[2:n] = b(s[2:n])/sqrt(1+sum(b(s[2:n]).^2))
    end
    alg = DP5();  atol=1e-80; rtol=1e-15
    s_l = s_0
    t = []
    # Solving system
    for _ = 1:L
        prob = ODEProblem(f!, vcat([0], x_0), (0, s_l))
        sol = solve(prob, alg, abstol=atol, reltol=rtol)
        cost_iter = length(sol.t)
        cost += cost_iter
        t_l_0 = sol.u[cost_iter][1]
        push!(t, [t_l_0])
        s_l *= gamma
    end
    # Performing Aitken delta squared
    for l = 1:L 
        for k = 1:convert(Int, min(floor((l-1)/2), k))
            push!(t[l], 
                  t[l][k]
                  -((t[l][k]-t[l-1][k])^2/(t[l][k]-2*t[l-1][k]+t[l-2][k])))
        end
    end
    t_idx_1 = length(t)
    t_idx_2 = length(t[t_idx_1])
    t = t[t_idx_1][t_idx_2]
    return t, cost
end

function stuart_floater_x_sqrd(x_0, b_der, H, eps)
    """
        stuart_floater_x_sqrd(x_0, b_der, H, eps) -> (t, cost)

    Function estimating explosion time of x' = x^2 using the method by Stuart & Floater
    with a fully implicit scheme.

    # Arguments
    - `x_0` : Initial value of ODE.
    - `b_der` : Derivative of right-hand side.
    - `H` : Function used to find adaptive step length.
    - `eps` : Tolerance used in computations.

    # Returns
    - `t` : Estimate of explosion time.
    - `cost` : Number of iterations needed for computations.
    """
    r = 2
    # Finding upper integration limit similar to our adaptive method
    while b_der(r) < log(1/eps)^2/eps
        r *= 1.001
    end
    h = eps
    t = 0
    x = x_0
    cost = 0
    while x < r
        cost += 1
        t += h*H(x)
        x = ((1 - sqrt(1-4*h))/(2*h))^cost*x_0
    end
    return t, cost
end

function mod_rescaling(x0, b, b_der, alpha, beta, lambda, M, h, ho, H)
    """
        mod_rescaling(x0, b, b_der, alpha, beta, lambda, M, h, h0, H) -> (t, cost)

    Function estimating explosion time of ODEs by using the method Modified rescaling
    algorithm by Cho and Wu.

    # Arguments
    - `x_0` : Initial value of ODE.
    - `b` : Function representing right-hand side of ODE.
    - `b_der` : Derivative of right-hand side.
    - `alpha` : Parameter used in rescaling
    - `beta` : Parameter used in rescaling
    - `lambda` : Parameter used in rescaling
    - `M` : Parameter deciding when rescaling is triggered
    - `h` : Step length
    - `ho` : Higher-order scheme is used if true. 
    - `H` : Function used in stopping criterion

    # Returns
    - `t` : Estimate of explosion time.
    - `cost` : Number of iterations needed for computations.    
    """
    x = x0; t = 0; n = 0; cost = 0
    while h*H(x/lambda^(n-1)) < 1
        while x < M
            if ho 
                x += b(x)*h + b(x)*b_der(x)*h^2/2
            else
                x += b(x)*h
            end
            t += lambda^(n*beta)*h
            cost += 1
        end
        x *= lambda^alpha
        n += 1
    end
    return t, cost
end

function plot_data(methods, costs, errors, epsilons, rs1=1.1, ord_adp_lin=true, rs2=1.1,
                   ord_uni_log=false, ord_uni_quad=false, ho_scheme=false)
    """
        plot_data(methods, costs, errors, epsilons, rs1=1.1, ord_adp_lin=true, rs2=1.1,
                  ord_uni_log=false, ord_uni_quad=false, ho_scheme=false) -> fig

    Function used to plot results from experiments. 

    NOTE: The placement of the labels has for several of the experiments been done with
          manual changes to the code in this function. Therefore, without these changes
          some of the figures will not be recreated perfectly (the graphs will be the same,
          but the labels could be placed differently and possibly overlapping the graph).

    # Arguments
    - `methods` : Dictionary used in experiment where keys are name of methods used.
    - `costs` : Dictionary mapping method name to vector of compuational costs.
    - `errors` : Dictionary mapping method name to vector of errors.
    - `epsilons` : Vector containing tolerances used in experiments.
    - `rs1` : Factor used to shift label and theoretical line for adaptive method.
    - `ord_adp_lin` : Flag used when the order of the adaptive method is linear.
    - `rs2` : Factor used to shift label and theoretical line for uniform method.
    - `ord_uni_log` : Flag used when the order of the uniform method has a log factor.
    - `ord_uni_quad` : Flag used when the order of the uniform method is quadratic.
    - `ho_scheme` : Flag used when the higher-order version of the adaptive method is used.

    # Returns
    - `fig` : Figures with data from experiment.
    """
    fig = Figure(size = (1000, 500))
    fontsize = 24

    ax1 = Axis(fig[1,1],  yscale=log10, xlabel=L"\log_2(\epsilon)", 
               ylabel="Cost", xlabelsize=fontsize, ylabelsize=fontsize,
               xticklabelsize=fontsize, yticklabelsize=fontsize)
    for name in sort(collect(keys(methods)))
        CairoMakie.scatter!(ax1, log2.(epsilons), costs[name], label=name)
        lines!(ax1, log2.(epsilons), costs[name], label=name)
    end
    i = length(epsilons)
    if ord_adp_lin
        lines!(ax1, log2.(epsilons), 
               rs1*(costs["Adaptive"][1]/epsilons[1]^(-1)).*epsilons.^(-1), 
               linestyle=:dash, color=:black)
        text!(ax1, log2.(epsilons[i-2]), 
              rs1*(costs["Adaptive"][1]/epsilons[1]^(-1))*epsilons[i-4]^(-1), 
              text =L"C_1\epsilon^{-1}", align=(:left, :top), 
              fontsize=fontsize)
        if ho_scheme
            lines!(ax1, log2.(epsilons), 
                   rs2*0.04*(costs["Adaptive"][1]/epsilons[1]^(-1/2))
                   .*epsilons.^(-1/2), 
                   linestyle=:dash, color=:black)
            text!(ax1, log2.(epsilons[i-1]), 
                  rs2*0.1*(costs["Adaptive"][1]/epsilons[1]^(-1/2))
                  *epsilons[i]^(-1/2), 
                  text =L"C_3\epsilon^{-1/2}", align=(:left, :top), 
                  fontsize=fontsize)
        end
    else
        lines!(ax1, log2.(epsilons), 
               rs1*(costs["Adaptive"][1]/
                    (epsilons[1]^(-1)*log.((epsilons[1])^(-1))^2))
                   .*log.((epsilons).^(-1)).^(2).*epsilons.^(-1), 
               linestyle=:dash, color=:black)
        text!(ax1, log2.(epsilons[i-2]), 
              rs1*(costs["Adaptive"][1]/
                   (epsilons[1]^(-1)*log.((epsilons[1])^(-1))))
                  *log((epsilons[i])^(-1))*epsilons[i-1]^(-1),
              text =L"C_1\log(\epsilon^{-1})\epsilon^{-1}", 
              align=(:left, :top), fontsize=fontsize)
    end
    if ord_uni_log
        lines!(ax1, log2.(epsilons), 
               rs2*(costs["Uniform"][1]/
                    (epsilons[1]^(-1)*log.((epsilons[1])^(-1))))
                   .*log.((epsilons).^(-1)).*epsilons.^(-1), linestyle=:dash,
               color=:black)
        text!(ax1, log2.(epsilons[i-1]), 
              rs2^3*(costs["Uniform"][1]/
                   (epsilons[1]^(-1)*log.((epsilons[1])^(-1))))
                  *log((epsilons[i])^(-1))*epsilons[i]^(-1), 
              text =L"C_2\log(\epsilon^{-1})\epsilon^{-1}", 
              align=(:left, :top), fontsize=fontsize)
    elseif ord_uni_quad
        lines!(ax1, log2.(epsilons), 
               rs2*(costs["Uniform"][1]/(epsilons[1]^(-2))).*epsilons.^(-2),
               linestyle=:dash, color=:black)
        text!(ax1, log2.(epsilons[i-2]), 
              rs2*(costs["Uniform"][1]/(epsilons[1]^(-2)))*epsilons[i-1]^(-2), 
              text =L"C_2\epsilon^{-2}", align=(:left, :top),
              fontsize=fontsize)
    end

    ax2 = Axis(fig[1,2], yscale=log10, xlabel=L"\log_2(\epsilon)", 
               ylabel=L"|\overline{\tau}-\tau|", xlabelsize=fontsize,
               ylabelsize=fontsize, xticklabelsize=fontsize,
               yticklabelsize=fontsize)
    for name in sort(collect(keys(methods)))
        CairoMakie.scatter!(ax2, log2.(epsilons), errors[name], label=name)
        lines!(ax2, log2.(epsilons), errors[name], label=name)
    end
    if ord_adp_lin
        lines!(ax2, log2.(epsilons), 1.1*epsilons, linestyle=:dashdot, 
               color=:black)
        text!(ax2, log2.(epsilons[2]), 4.7*epsilons[1], text =L"C\epsilon",
              align=(:left, :top), fontsize=fontsize)
    else
        lines!(ax2, log2.(epsilons), epsilons, linestyle=:dashdot, 
               color=:black)
        text!(ax2, log2.(epsilons[2]), 0.9*epsilons[1], text =L"\epsilon",
              align=(:left, :top), fontsize=fontsize)
        lines!(ax2, log2.(epsilons), 0.5*epsilons.*log.((epsilons).^(-1)), 
               linestyle=:dashdot, color=:black)
        text!(ax2, log2.(epsilons[2]), 
              1.25*0.5*epsilons[1].*log.((epsilons[1]).^(-1)), 
              text =L"C_1\log_2(\epsilon^{-1})\epsilon", align=(:left, :top), 
              fontsize=fontsize)
    end
    
    if length(methods) > 1
        axislegend(ax2, merge=true, position=:lt, framevisible=false,
                   labelsize=fontsize)
    end
    for ax in [ax1, ax2]
        hidespines!(ax, :t, :r)
        hidedecorations!(ax, ticks=false, ticklabels=false, label=false, 
                         minorticks=false)
    end
    return fig
end

function plot_data_xlogx(costs, errors, epsilons, c)
    """
        plot_data_xlogx(costs, errors, epsilons, c) -> fig

    Function used to plot results from experiments where the rhs has a log-factor. 

    # Arguments
    - `costs` : Dictionary mapping method name to vector of compuational costs.
    - `errors` : Dictionary mapping method name to vector of errors.
    - `epsilons` : Vector containing tolerances used in experiments.
    - `c` : Variable for exponent used in rhs to get correct labels and placement.

    # Returns
    - `fig` : Figures with data from experiment.
    """
    fig = Figure(size = (1000, 500))
    fontsize = 24
    i = length(epsilons)

    ax1 = Axis(fig[1,1],  yscale=log10, xlabel=L"\log_2(\epsilon)", 
               ylabel="Cost", xlabelsize=fontsize, ylabelsize=fontsize,
               xticklabelsize=fontsize, yticklabelsize=fontsize)
    for name in sort(collect(keys(costs)))
        CairoMakie.scatter!(ax1, log2.(epsilons), costs[name], label=name)
        lines!(ax1, log2.(epsilons), costs[name], label=name)
    end
    if c == 1
        lines!(ax1, log2.(epsilons), 
            (costs["Adaptive"][1]/
                (epsilons[1]^(-1)*log.((epsilons[1])^(-1))^2))
            .*1.3*log.((epsilons).^(-1)).^2 .*epsilons.^(-1), 
            linestyle=:dash, color=:black)
        text!(ax1, log2.(epsilons[i]), 
            (costs["Adaptive"][1]/
            (epsilons[1]^(-1)*log.((epsilons[1])^(-1))^2))
            *log((epsilons[i-3])^(-1))^2*epsilons[i-3]^(-1),
            text =L"C_1\log(\epsilon^{-1})^2\epsilon^{-1}", 
            align=(:left, :top), fontsize=fontsize)
        lines!(ax1, log2.(epsilons), 
           1.3*(costs["Uniform"][1]/epsilons[1]^(-2)).*epsilons.^(-2),
           linestyle=:dash, color=:black)
        text!(ax1, log2.(epsilons[i-1]), 
              (costs["Uniform"][1]/epsilons[1]^(-2))*epsilons[i]^(-2), 
              text =L"C_2\epsilon^{-2}", align=(:left, :top), 
              fontsize=fontsize)
    else
        lines!(ax1, log2.(epsilons), 
               1.3*(costs["Adaptive"][1]
               /epsilons[1]^(-2)).*epsilons.^(-2), 
               linestyle=:dash, color=:black)
        text!(ax1, log2.(epsilons[i-1]), 
              (costs["Adaptive"][1]/epsilons[1]^(-2))*epsilons[i]^(-2), 
              text =L"C_1\epsilon^{-2}", align=(:left, :top), 
              fontsize=fontsize)
        lines!(ax1, log2.(epsilons), 
               1.3*(costs["Uniform"][1]/epsilons[1]^(-3)).*epsilons.^(-3),
               linestyle=:dash, color=:black)
        text!(ax1, log2.(epsilons[i-1]), 
              (costs["Uniform"][1]/epsilons[1]^(-3))*epsilons[i]^(-3), 
              text =L"C_2\epsilon^{-3}", align=(:left, :top), 
              fontsize=fontsize)
    end

    ax2 = Axis(fig[1,2], yscale=log10, xlabel=L"\log_2(\epsilon)", 
               ylabel=L"|\overline{\tau}-\tau|", xlabelsize=fontsize,
               ylabelsize=fontsize, xticklabelsize=fontsize,
               yticklabelsize=fontsize)
    for name in sort(collect(keys(costs)))
        CairoMakie.scatter!(ax2, log2.(epsilons), errors[name], label=name)
        lines!(ax2, log2.(epsilons), errors[name], label=name)
    end
    lines!(ax2, log2.(epsilons), 0.01*epsilons, linestyle=:dashdot, 
           color=:black)
    text!(ax2, log2.(epsilons[2]), 0.0125*epsilons[1], text =L"C\epsilon",
          align=(:left, :top), fontsize=fontsize)
    
    axislegend(ax2, merge=true, position=:lt, framevisible=false,
               labelsize=fontsize)
    for ax in [ax1, ax2]
        hidespines!(ax, :t, :r)
        hidedecorations!(ax, ticks=false, ticklabels=false, label=false, 
                         minorticks=false)
    end
    Label(fig[0, :], "c = $c", fontsize=fontsize) 
    return fig
end

function print_table(ests, costs, target_list, target_name)
    """
        print_table(ests, costs, target_list, target_name)

    Function used to print out table with data from reaction-diffusion experiment when a
    variable is refined. 

    # Arguments
    - `ests` : Dictionary mapping method name to vector of estimates for tau.
    - `costs` : Dictionary mapping method name to vector of compuational costs.
    - `target_list` : Vector containing values for variable being refined.
    - `target_name` : Name of variable being refined.
    """
    println("     |$(" "^10)Adaptive$(" "^(2*10))Uniform")
    println("-"^63)
    title_str_adp = " $target_name |$(" "^4)estimate$(" "^4)error  cost "
    title_str_uni = "|$(" "^4)estimate$(" "^5)error cost"
    println(title_str_adp*title_str_uni)
    println("-"^63)
    est_adp = round(ests["Adaptive"][1], digits=12)
    cost_adp = round(log2(costs["Adaptive"][1]), digits=2)
    str1 = "$(log2(target_list[1]))| $est_adp$(" "^8)$cost_adp"
    est_uni = round(ests["Uniform"][1], digits=12)
    cost_uni = round(log2(costs["Uniform"][1]), digits=2)
    str2 = "| $est_uni$(" "^8)$cost_uni"
    println(str1*str2)
    for i=2:length(ests["Adaptive"])
        est_adp = round(ests["Adaptive"][i], digits=12)
        err_adp=round(log2(abs(ests["Adaptive"][i]-ests["Adaptive"][i-1])),
                      digits=2)
        cost_adp = round(log2(costs["Adaptive"][i]), digits=2)
        str1 = "$(log2(target_list[i]))| $est_adp $err_adp $cost_adp"
        est_uni = round(ests["Uniform"][i],digits=12)
        err_uni = round(log2(abs(ests["Uniform"][i]-ests["Uniform"][i-1])),
                        digits=2)
        cost_uni = round(log2(costs["Uniform"][i]), digits=2)
        str2 = "| $est_uni $err_uni $cost_uni"
        println(str1*str2)
    end
end
