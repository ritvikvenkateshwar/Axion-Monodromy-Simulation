# Axion Monodromy Cosmic Evolution Simulator

A Python simulation of cosmic inflation and reheating using axion monodromy from Type IIB string theory, evolving the universe from inflation through dark energy domination.

## 📖 Overview

This code implements a comprehensive cosmological model where an axion field serves as:
- **The inflaton** driving exponential expansion
- **The reheating source** decaying into Standard Model particles
- **The origin** of dark matter and evolving dark energy
- **The mechanism** for matter-antimatter asymmetry

Based on top-down string theory derivations, the simulation tracks the coupled evolution of the axion field, gauge fields, and cosmological energy components.

## 🧮 Physical System

The simulation solves this coupled system of equations:

\begin{equation}
\boxed{
\begin{aligned}
&\ddot{\phi} + 3H\dot{\phi} + \mu^3 \frac{\phi}{\sqrt{\phi^2 + \epsilon^2}} - \frac{\Lambda^4}{f}\sin\left(\frac{\phi}{f}\right) = \mathcal{J}_{\text{gauge}} - \Gamma_{\text{tot}}\dot{\phi} \\[0.5em]
&\mathcal{J}_{\text{gauge}} = -\frac{1}{M_{\text{PL}}} \sum_{i=\text{SM,Hid,DE}} \frac{\alpha_i}{f_a} \int\frac{d^3k}{(2\pi)^3} \mathbf{E}_i(k) \cdot \mathbf{B}_i(k) \\[0.5em]
&\ddot{A}_i(k) + H\dot{A}_i(k) + k^2 A_i(k) = \frac{\alpha_i}{f_a} \dot{\phi} \, k \, A_i(k) \\[0.5em]
&H^2 = \frac{1}{3M_{\text{PL}}^2} \left( \frac{1}{2}\dot{\phi}^2 + V(\phi) + \rho_r + \rho_{\text{dm}} + \rho_{\text{de}} + \rho_A \right) \\[0.5em]
&\dot{H} = -\frac{1}{2M_{\text{PL}}^2} \left( \dot{\phi}^2 + \frac{4}{3}\rho_r + \rho_{\text{dm}} + (1+w)\rho_{\text{de}} + \frac{4}{3}\rho_A \right) \\[0.5em]
&\dot{\rho}_r + 4H\rho_r = \Gamma_{\phi\to r} \dot{\phi}^2 + \mathcal{P}_{\text{gauge}\to r} \\[0.5em]
&\dot{\rho}_{\text{dm}} + 3H\rho_{\text{dm}} = \Gamma_{\phi\to \text{dm}} \dot{\phi}^2 + \mathcal{P}_{\text{gauge}\to \text{dm}} \\[0.5em]
&\dot{\rho}_{\text{de}} + 3H(1+w)\rho_{\text{de}} = \Gamma_{\phi\to \text{de}} \dot{\phi}^2 + \mathcal{P}_{\text{gauge}\to \text{de}} \\[0.5em]
&\mathcal{P}_{\text{gauge}\to i} = \Gamma_{A_i\to i} \rho_{A_i} \\[0.5em]
&\rho_A = \sum_{i} \rho_{A_i}, \quad \rho_{A_i} = \int \frac{d^3k}{(2\pi)^3} \frac{1}{2a^2} \left(|\dot{A}_i(k)|^2 + k^2 |A_i(k)|^2 \right) \\[0.5em]
&\Gamma_{\text{tot}} = \Gamma_{\phi\to r} + \Gamma_{\phi\to \text{dm}} + \Gamma_{\phi\to \text{de}}
\end{aligned}
}
\end{equation}

## 🚀 Features

- **Full coupled system**: Axion field, gauge fields, and cosmological fluids
- **String theory parameters**: Derived from Type IIB compactification
- **DESI-inspired dark energy**: Time-varying equation of state w(a)
- **Gauge field production**: With backreaction on inflation
- **Multi-sector reheating**: Decay to SM, dark matter, and dark energy
- **CP violation**: For matter-antimatter asymmetry

## 🔧 Installation

```bash
git clone https://github.com/your-username/axion-monodromy-simulation.git
cd axion-monodromy-simulation
pip install -r requirements.txt
