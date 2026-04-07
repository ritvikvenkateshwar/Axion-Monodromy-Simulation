include("AxionMonodromyEquationFunctions.jl")
module AxionODESolver
  using DifferentialEquations 
  using ODEInterface, ODEInterfaceDiffEq
  using ModelingToolkit
  using JLD2
  using ..AxionEquations
  
  export AxionSystem!, SolveAxion
  function AxionSystem!(dy, y, p, N)
      T = eltype(y) 
      # 1. State Unpacking
      phi, phi_p = y[1], y[2]
      rho_vec = @view y[3:5]
      a = exp(N)
      Nk = length(p.k_grid)
      vol_const = 1.0 / (2.0 * pi^2 * a^2)
      
      # 2. Vectorized Gauge Sums
      alphas = (p.alpha_r, p.alpha_dm, p.alpha_de)
      gammas_A = (p.G_Ar, p.G_Adm, p.G_Ade)
      
      J_total = 0.0
      rho_A_mag_total = 0.0
      rho_A_kin_unscaled_total = 0.0
      rho_A_kin_i = zeros(T, 3)

      for s in 1:3
          start_idx = 5 + (s-1)*(2*Nk) + 1
          As   = @view y[start_idx : 2 : start_idx + 2*Nk - 2]
          A_ps = @view y[start_idx + 1 : 2 : start_idx + 2*Nk - 1]
          
          mag_part = sum(p.weights .* (p.k_grid.^2 .* As.^2)) * 0.5 * vol_const
          kin_unscaled = sum(p.weights .* A_ps.^2) * 0.5 * vol_const
          
          J_total += sum(p.weights .* (alphas[s]/p.fa .* p.k_grid .* As .* A_ps ./ a^2)) * vol_const
          rho_A_mag_total += mag_part
          rho_A_kin_unscaled_total += kin_unscaled
          rho_A_kin_i[s] = kin_unscaled 
      end

      # 3. Hubble Dynamics
      V = AxionEquations.axionPotential(p.mu, phi, p.Lambda, p.fa, p.epsilon)
      d_raw = 3.0 - 0.5*phi_p^2 - rho_A_kin_unscaled_total
      denom = ifelse(d_raw > 1e-12, d_raw, T(1e-12))
      H2 = (V + sum(rho_vec) + rho_A_mag_total) / max(denom, 1e-12)
      H = sqrt(H2)
      
      p_de = AxionEquations.wde(a) * y[5]
      p_tot = (0.5*phi_p^2 - rho_A_kin_unscaled_total)*H2 - V + (1/3)*y[3] + p_de + (1/3)*rho_A_mag_total
      h_ratio = -1.5 * (1.0 + p_tot / (3.0 * H2))

      # 4. Background EOMs
      dV = AxionEquations.dVdphi(p.mu, phi, p.Lambda, p.fa, p.epsilon)
      G_phi_tot = p.G_phi_r + p.G_phi_dm + p.G_phi_de
      
      dy[1] = phi_p
      dy[2] = (J_total / H) - (dV / H2) - (3.0 + h_ratio + G_phi_tot/H) * phi_p
      
      ws = (1/3, 0.0, AxionEquations.wde(a))
      phi_source = (p.G_phi_r, p.G_phi_dm, p.G_phi_de) .* (H * phi_p^2)
      @. dy[3:5] = -3.0*(1.0 + ws)*rho_vec + phi_source + (gammas_A * (rho_A_mag_total/3 + H2*rho_A_kin_i))/H

      # 5. Mode EOMs
      for s in 1:3
          idx = 5 + (s-1)*(2*Nk) + 1
          As   = @view y[idx : 2 : idx + 2*Nk - 2]
          A_ps = @view y[idx + 1 : 2 : idx + 2*Nk - 1]
          @. dy[idx : 2 : idx + 2*Nk - 2] = A_ps
          @. dy[idx + 1 : 2 : idx + 2*Nk - 1] = -(1.0 + h_ratio + gammas_A[s]/H) * A_ps - 
              ((p.k_grid/a)^2 / H2 - (alphas[s]/p.fa * phi_p * p.k_grid/a) / H) * As
      end
  end

  function SolveAxion()
    # Define k_grid here since it's needed for weights and p
    k_grid_array = AxionEquations.make_k_grid(0.1, 100.0, 50, "log")
    
    p = (
      mu=1e-4, Lambda=1e-3, fa=0.01, epsilon=1e-6,
      alpha_r=10.0, alpha_dm=5.0, alpha_de=1.0,
      G_phi_r=1e-5, G_phi_dm=1e-6, G_phi_de=0.0,
      G_Ar=1e-2, G_Adm=1e-3, G_Ade=0.0,
      k_grid = k_grid_array,
      weights = AxionEquations.init_log_weights(k_grid_array)
    )

    # Initial State
    phi0, phip0 = 15.0, -0.01
    Nk = length(k_grid_array)
    y0 = zeros(5 + 6*Nk)
    y0[1:5] .= [phi0, phip0, 1e-10, 1e-10, 1e-10]

    # Initial Hubble for BD scaling
    V0 = AxionEquations.axionPotential(p.mu, phi0, p.Lambda, p.fa, p.epsilon)
    H0 = sqrt(V0 / (3.0 - 0.5*phip0^2))

    # Initialize Gauge Sectors
    for s in 0:2
        start_idx = 6 + (s * 2 * Nk)
        for i in 1:Nk
            A, Adot = AxionEquations.bd_initial_conditions(k_grid_array[i], 1.0, H0, AxionEquations.rng)
            y0[start_idx + 2*(i-1)] = A
            y0[start_idx + 2*(i-1) + 1] = Adot / H0
        end
    end

    # Define and Solve
    prob = ODEProblem(AxionSystem!, y0, (0.0, 80.0), p)
    # Use ModelingToolkit for the Jacobian
    sys = modelingtoolkitize(prob)
    prob_fast = ODEProblem(sys, y0, (0.0, 80.0), p)
    sol = solve(prob_fast, RadauIIA5(), reltol=1e-6, abstol=1e-9)
    jldsave("axion_output_1.jld2"; sol)
    return sol
  end
end
