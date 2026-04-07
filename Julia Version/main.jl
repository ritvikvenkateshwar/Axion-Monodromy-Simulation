using JLD2, Plots
include("AxionMonodromySolver.jl")
include("AxionMonodromyEquationFunctions.jl")
include("AxionPostInflationAnalysis.jl")
using .AxionODESolver
using .AxionEquations

sol = AxionODESolver.SolveAxion()
