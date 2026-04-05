import DifferentialEquations as DE
import ODEInterfaceDiffEq as ODE
import LSODA as L
import Sundials as SD
using Plots
using LaTeXStrings
using Random

g_star = 100
rng = MersenneTwister(1234)

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
  energy = (A_dot^2 + k^2 * A^2) / (2 * a^4)
  return energy
end

function compute_J_gauge(A, A_dot, k, alpha, fa, M_pl, a)
  E_dot_B = - (k * A * A_dot) / a^4
  return - (1/M_pl)*(alpha / (fa * M_pl)) * E_dot_B
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
