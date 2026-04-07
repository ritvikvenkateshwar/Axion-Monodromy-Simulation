using JLD2, Plots
include("AxionMonodromySolver.jl")
include("AxionMonodromyEquationFunctions.jl")
using .AxionODESolver
using .AxionEquations

sol = AxionODESolver.SolveAxion()
