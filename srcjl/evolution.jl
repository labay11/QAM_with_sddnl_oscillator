using QuantumOptics
using NPZ
using ArgParse
using Pardiso
using Statistics

include("model.jl")

function _ev(t, psi, a)
    expected = expect(a, psi)
    println("$t\t$expected")
    expected
end

function evolve(times, ρ0, γ2::Float64, η::Float64, Δ::Float64, n::Int, m::Int, basis::FockBasis)
    H, J = build_system(1.0, γ2, η, Δ, n, m, basis, false)

    a = destroy(basis)
    ev(t, psi) = _ev(t, psi, a)
    tout, eva = timeevolution.master(times, ρ0, H, J; fout=ev, maxiters=Int(5e9))

    data = hcat(tout, eva)

    Nt = length(times)
    fpath = "$(homedir())/Documents/data/gqvdp/evolution/$(n)_$(m)/"
    mkpath(fpath)

    npzwrite("$fpath/g-$(γ2)_e-$(η)_d-$(Δ)_t-0-$(times[Nt])-$(Nt).npy", data)
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
        "-d"
            help = "the detuning"
            arg_type = Float64
            default = 0.4
        "-g"
            help = "the gamma_m"
            arg_type = Float64
            default = 0.2
        "-e"
            help = "the eta_n"
            arg_type = Float64
            default = 2.0
        "--tmax"
            help = "the max beta"
            arg_type = Float64
        "--tnum"
            help = "the num beta"
            arg_type = Int
    end

    parse_args(s)
end


args = parse_commandline()

basis = basis = FockBasis(50)
β = amplitude(args["g"], args["e"], args["n"], args["m"])
ρ0 = coherentstate(basis, 0.5 * β * exp(1im * 2 * π / 9))

times = exp10.(LinRange(log10(1 / args["tnum"]), log10(args["tmax"]), args["tnum"]))
evolve(times, ρ0, args["g"], args["e"], args["d"], args["n"], args["m"], basis)
