module AxionEquations
  using Random
  using Printf
  export axionPotential, dVdphi, wde, compute_H, compute_J_gauge, compute_rho_A, bd_initial_conditions, init_integration_weights, nit_mode_bank, make_k_grid,check_energy_conservation
  rng = Xoshiro(1234)

  function axionPotential(mu, phi, Lambda, fa, epsilon)
    linear = mu^3 * sqrt(phi^2+epsilon^2)
    oscillations = Lambda^4 * (1-cos(phi/fa))
    return linear + oscillations
  end

  function dVdphi(mu, phi, Lambda, fa, epsilon)
    d_linear = mu^3 * phi/(sqrt(phi^2 + epsilon^2))
    d_oscillations = (Lambda^4/fa) * (sin(phi/fa))
    return d_linear + d_oscillations
  end

  function wde(a, w0 = -0.971, wa = 0.62)
    return w0 + wa*(1-a)
  end

  function compute_H(phi, phi_dot, rho_r, rho_dm, rho_de, rho_A, mu, Lambda, fa, epsilon)
    V_phi = axionPotential(mu, phi, Lambda, fa, epsilon)
    rho_phi = 0.5 * phi_dot^2 + V_phi
    rho_total = rho_phi + rho_r + rho_dm + rho_de + rho_A
    H = sqrt(rho_total / 3)  # M_pl = 1
    return H
  end

  function compute_rho_A(A, A_dot, k, a)
    volume_element = k^2 / (2 * pi^2)
    energy_density = (A_dot^2 + k^2 * A^2) / (2 * a^2)
    return energy_density * volume_element
  end

  function compute_J_gauge(A, A_dot, k, alpha, fa, M_pl, a)
    volume_factor = k^2 / (2 * pi^2)
    E_dot_B = - (k * A * A_dot) / a^2 
    return -(alpha / fa) * E_dot_B * volume_factor
  end

  function bd_initial_conditions(k, a, H_start, rng)
    # Physical momentum at horizon crossing for this mode
    # Not used directly but for reference
    k_horizon_crossing = a*H_start

    # Simple approach: initialize with frozen super-horizon approximation
    # The exact initial conditions don't matter much as long as they're small
    # and the equations will evolve them correctly once instability kicks in
    quantum_amp_physical = H_start / (2*pi)
    # Convert to rescaled variable: A_rescaled = a × A_physical
    A_rescaled = a * quantum_amp_physical
    # Add k-dependence: 1/√k scaling from BD vacuum
    # But for numerical stability, don't let it get too small
    k_factor = 1.0 / sqrt(max(k / (a * k_horizon_crossing), 1.0))
    A_rescaled *= k_factor
    phase = rand(rng) * 2*pi
    A_rescaled *= cos(phase)
    A_dot_rescaled = H_start * A_rescaled
    return A_rescaled, A_dot_rescaled
  end

  function init_mode_bank(k_array, a, H_start)
    n_k = length(k_array)
    A0 = zeros(n_k)
    A_dot0 = zeros(n_k)

    for (i,k) in enumerate(k_array)
      A0[i], A_dot0[i] = bd_initial_conditions(k, a, H_start, rng)
    end
    return A0, A_dot0
  end

  function make_k_grid(k_min, k_max, n_k, spacing="log")
    if spacing == "log"
      return exp.(range(log(k_min), log(k_max), length=n_k))
    else
      return collect(range(k_min, k_max, length=n_k))
    end
  end

  function init_integration_weights(k_array)
      n = length(k_array)
      weights = zeros(n)

      if n == 1
          weights[1] = 1.0  # Single point case
      elseif n == 2
          val = 0.5 * (k_array[2] - k_array[1])
          weights[1] = val
          weights[2] = val
      else
          # First point: half-width to the next point
          weights[1] = 0.5 * (k_array[2] - k_array[1])
          
          # Last point: half-width from the previous point
          weights[end] = 0.5 * (k_array[end] - k_array[end-1])
          
          # Middle points: centered difference (average of intervals on both sides)
          for i in 2:(n-1)
              weights[i] = 0.5 * (k_array[i+1] - k_array[i-1])
          end
      end

      return weights
  end


  function check_energy_conservation(y, params, N)
      phi, phi_dot, rho_r, rho_dm, rho_de, a, H = y[1:7]
      
      # Extract params
      mu, Lambda, fa, epsilon = params[:mu], params[:Lambda], params[:fa], params[:epsilon]
      k_array = params[:k_array]
      weights = params[:integration_weights]

      # Calculate Scalar field energy
      V_phi = axionPotential(mu, phi, Lambda, fa, epsilon)
      rho_phi = 0.5 * phi_dot^2 + V_phi

      # Sum over gauge fields 
      rho_A_total = 0.0
      n_k = length(k_array)
      for i in 1:n_k
          A_idx = 8 + 2*(i-1)
          A_dot_idx = A_idx + 1
          k_val = k_array[i]
          rho_A_total += compute_rho_A(y[A_idx], y[A_dot_idx], k_val, a) * weights[i]
      end

      total_energy_components = rho_phi + rho_r + rho_dm + rho_de + rho_A_total
      total_energy_Hubble = 3.0 * H^2

      if total_energy_Hubble <= 0
          return 0.0
      end

      violation = abs(total_energy_components - total_energy_Hubble) / total_energy_Hubble

      if violation > 0.01
          @printf("ENERGY VIOLATION at N=%.3f: %.2e\n", N, violation)
          @printf("  Components: %.2e, Hubble: %.2e\n", total_energy_components, total_energy_Hubble)
          @printf("  Breakdown: ρ_ϕ=%.2e, ρ_r=%.2e, ρ_dm=%.2e, ρ_de=%.2e, ρ_A=%.2e\n", 
                  rho_phi, rho_r, rho_dm, rho_de, rho_A_total)
      end

      return violation
  end
end
