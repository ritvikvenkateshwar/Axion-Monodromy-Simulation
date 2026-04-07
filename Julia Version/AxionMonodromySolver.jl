include("AxionMonodromyEquationFunctions.jl")

module ODESolver
  using DifferentialEquations 
  using ODEInterface, ODEInterfaceDiffEq
  using Plots, LaTeXStrings
  using .AxionEquations
  
  function AxionSystem_eFolds!(dy, y, p, N) 
    phi         = y[1]
    phi_prime   = y[2]
    rho_r       = y[3]
    rho_dm      = y[4]
    rho_de      = y[5]
    
    # Constants and Params
    mu, Lambda, fa, epsilon = p[:mu], p[:Lambda], p[:fa], p[:epsilon]
    alpha, M_pl             = p[:alpha], p[:M_pl] # Assuming M_pl = 1
    k_grid, weights         = p[:k_grid], p[:weights]
    G_r, G_dm, G_de         = p[:G_r], p[:G_dm], p[:G_de]
    
    a = exp(N)
    
    # 2. Sum Gauge Contributions (rho_A and J_gauge)
    J_gauge = 0.0
    rho_A_total = 0.0
    n_k = length(k_grid)

    for i in 1:n_k
        idx_A     = 5 + (2*i - 1)
        idx_Aprime = idx_A + 1
        
        Ak      = y[idx_A]
        A_prime = y[idx_Aprime]
        k       = k_grid[i]
        
        # Note: dot{A} = H * A_prime
        # rho_A = ( (H*A_prime)^2 + k^2*A^2 ) / (2 * a^2) * (k^2 / 2pi^2)
        # J_gauge = (alpha/fa) * (k * A * H * A_prime / a^2) * (k^2 / 2pi^2)
        
        vol_fac = k^2 / (2 * pi^2)
        rho_A_total += ( (k^2 * Ak^2) / (2 * a^2) ) * vol_fac * weights[i]
        # Partial J (missing H factor)
        J_gauge     += (alpha / fa) * (k * Ak * A_prime / a^2) * vol_fac * weights[i]
    end

    # 3. Calculate H and H_prime/H
    # Friedmann: 3H^2 = 0.5(H*phi_prime)^2 + V + rho_r + rho_dm + rho_de + rho_A_total
    # H^2 * (3 - 0.5*phi_prime^2) = V + rho_others
    V = AxionEquations.axionPotential(mu, phi, Lambda, fa, epsilon)
    rho_others = rho_r + rho_dm + rho_de + rho_A_total
    
    H2 = rho_others > 0 ? (V + rho_others) / (3.0 - 0.5 * phi_prime^2) : 1e-20
    H  = sqrt(max(0.0, H2))
    
    # Effective pressure for H_prime calculation: p = w*rho
    p_de = AxionEquations.wde(a) * rho_de
    p_tot = 0.5*H^2*phi_prime^2 - V + (1/3)*rho_r + p_de + (1/3)*rho_A_total
    rho_tot = 0.5*H^2*phi_prime^2 + V + rho_r + rho_dm + rho_de + rho_A_total
    
    # H_prime / H = d(ln H)/dN = -1.5 * (1 + p_tot/rho_tot)
    h_ratio = -1.5 * (1.0 + p_tot / rho_tot)

    # 4. Background EOMs (d/dN)
    dV = AxionEquations.dVdphi(mu, phi, Lambda, fa, epsilon)
    Gamma_tot = G_r + G_dm + G_de
    
    # phi_prime_prime
    # Equation: H^2*phi'' + (H*H' + 3H^2)*phi' + dV = H*J_gauge - Gamma*H*phi'
    dy[1] = phi_prime
    dy[2] = (J_gauge / H) - (dV / H^2) - (3.0 + h_ratio + (Gamma_tot / H)) * phi_prime
    
    # Fluids (Source term / H)
    dy[3] = -4.0 * rho_r  + G_r * (H * phi_prime^2)
    dy[4] = -3.0 * rho_dm + G_dm * (H * phi_prime^2)
    dy[5] = -3.0 * (1.0 + AxionEquations.wde(a)) * rho_de + G_de * (H * phi_prime^2)

    # 5. Gauge Mode EOMs (d/dN)
    for i in 1:n_k
        idx_A = 5 + (2*i - 1)
        idx_Aprime = idx_A + 1
        k = k_grid[i]
        
        k_phys_sq = (k / a)^2
        instability = (alpha / fa) * (phi_prime * k / a)
        
        dy[idx_A]      = y[idx_Aprime]
        dy[idx_Aprime] = -(1.0 + h_ratio) * y[idx_Aprime] - (k_phys_sq / H^2 - instability / H) * y[idx_A]
    end
  end

  function SolveAxion()
    
  end
end
