using QuantumOptics
using NPZ
using ArgParse
using Pardiso
using Statistics

include("model.jl")
include("utils.jl")


function themodynamic_limit(ηs, N::Float64, Δ::Float64, n::Int, m::Int, dim::Int; γ1::Float64 = 1.)
    Γ = γ1 / N

    basis = FockBasis(dim)

    op_n = number(basis)
    op_a = destroy(basis)
    op_an = op_a^n

    lng = length(ηs)

    EV = Array{ComplexF64, 2}(undef, lng, 3)

    for j in 1:lng
        L = build_system(γ1, Γ, ηs[j], Δ, n, m, basis)
        ρ = steadystate_mkl(L)

        EV[j, 1] = expect(op_n, ρ) / N
        EV[j, 2] = expect(op_a, ρ) / N
        EV[j, 3] = expect(op_an, ρ) / N
    end

    fpath = local_data_path("thermo_limit", n, m)

    npzwrite("$(fpath)/etas&$(ηs[1])&$(ηs[lng])&$(lng)_D&$(Δ)_g1&$(γ1)_g2&$(Γ)_dim&$(dim).npy", EV)
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-n"
            help = "the driving power"
            arg_type = Int
        "-m"
            help = "the dissipation power"
            arg_type = Int
        "-N"
            help = "g1 / g2"
            arg_type = Float64
        "-D"
            help = "delta"
            arg_type = Float64
            default = 0.4
        "-g"
            help = "g1"
            arg_type = Float64
            default = 1.
        "--emin"
            help = "eta min"
            arg_type = Float64
            default = 0.
        "--emax"
            help = "eta max"
            arg_type = Float64
            default = 5.
        "--enum"
            help = "eta num"
            arg_type = Int
            default = 100
        "--dim"
            help = "dim"
            arg_type = Int
            default = 70
    end

    parse_args(s)
end

args = parse_commandline()

ηs = LinRange(args["emin"], args["emax"], args["enum"])
themodynamic_limit(ηs, args["N"], args["D"], args["n"], args["m"], args["dim"]; γ1 = args["g"])
