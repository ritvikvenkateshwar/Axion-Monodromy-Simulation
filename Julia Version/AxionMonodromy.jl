import DifferentialEquations as DE
import ODEInterfaceDiffEq as ODE
import LSODA as L
import Sundials as SD

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

function compute_J_gauge(A, A_dot, k, alpha, fa, M_pl)
  E_dot_B = - (k * A * A_dot) / a^4
  return - (1/M_pl)*(alpha / (fa * M_pl)) * E_dot_B
end
