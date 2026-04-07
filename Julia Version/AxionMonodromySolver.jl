include("AxionMonodromyEquationFunctions.jl")

module ODESolver
  using DifferentialEquations 
  using ODEInterface, ODEInterfaceDiffEq
  using Plots, LaTeXStrings
  using .AxionEquations
  
  function AxionSystem!(dy, y,  p, t)
    phi      = y[1]
    phi_dot  = y[2]
    rho_r    = y[3]
    rho_dm   = y[4]
    rho_de   = y[5]
    a        = y[6]

    mu, Lambda, fa, epsilon = p[:mu], p[:Lambda], p[:fa], p[:epsilon]
    alpha, M_pl = p[:alpha], p[:M_pl]
    k_grid = p[:k_grid]
    weights = p[:weights]
    Gamma_phi_r, Gamma_phi_dm, Gamma_phi_de = p[:G_r], p[:G_dm], p[:G_de]

    J_gauge = 0.0
    rho_A_total = 0.0
    n_k = length(k_grid)

    for i in 1:n_k
        idx_A = 6 + (2*i - 1)
        idx_Adot = idx_A + 1
        
        Ak, Adotk = y[idx_A], y[idx_Adot]
        k = k_grid[i]
        
        # J_gauge = -(alpha/fa) * (E·B)
        J_gauge     += AxionEquations.compute_J_gauge(Ak, Adotk, k, alpha, fa, M_pl, a) * weights[i]
        rho_A_total += AxionEquations.compute_rho_A(Ak, Adotk, k, a) * weights[i]
    end
    # 4. Background Evolution (Friedmann)
    H = AxionEquations.compute_H(phi, phi_dot, rho_r, rho_dm, rho_de, rho_A_total, mu, Lambda, fa, epsilon)
    
    # 5. Scalar Field EOM (Equation 1)
    Gamma_tot = Gamma_phi_r + Gamma_phi_dm + Gamma_phi_de
    dV = AxionEquations.dVdphi(mu, phi, Lambda, fa, epsilon)
    
    phi_ddot = -3*H*phi_dot - dV + J_gauge - Gamma_tot*phi_dot

    # 6. Fluid Equations (Equations 6-8)
    dy[1] = phi_dot
    dy[2] = phi_ddot
    dy[3] = -4*H*rho_r  + Gamma_phi_r * phi_dot^2
    dy[4] = -3*H*rho_dm + Gamma_phi_dm * phi_dot^2
    dy[5] = -3*H*(1 + AxionEquations.wde(a))*rho_de + Gamma_phi_de * phi_dot^2
    dy[6] = a * H

    # 7. Gauge Mode EOMs (Equation 3)
    for i in 1:n_k
        idx_A = 6 + (2*i - 1)
        idx_Adot = idx_A + 1
        k = k_grid[i]
        
        # ddot{A} + H*dot{A} + (k^2 - (alpha/fa)*phi_dot*k)A = 0
        A_ddot = ( (alpha/fa)*phi_dot*k - k^2 ) * y[idx_A] - H * y[idx_Adot]
        
        dy[idx_A]    = y[idx_Adot]
        dy[idx_Adot] = A_ddot
    end
  end
end
