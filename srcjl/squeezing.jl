using QuantumOptics
using NPZ
using ArgParse

include("model.jl")

function squeezing_γη(γs, ηs, Δ::Int, n::Int, m::Int)
    Nη = length(ηs)
    Ng = length(γ2s)

    EV = Array{ComplexF64, 2}(undef, Ng, Nη, 2)
    fpath = "$(homedir())/Documents/data/gqvdp/squeezing/a/$(n)_$(m)"
    mkpath(fpath)

    for j in 1:Ng
        for k in 1:Nη
            beta = amplitude(γs[j], ηs[k], n, m)
            dim = clamp(round(Int, beta), 80, 200)
            basis = FockBasis(dim)
            H, J = build_system(1.0, γs[j], ηs[k], Δ, n, m, basis, full=false)
            ss = steadystate.iterative(H, J)
            n = number(basis)
            EV[j, k, 1] = expect(n, ss)
            EV[j, k, 2] = expect(n^2, ss)
        end
    end

    npzwrite("$(fpath)/g-$(γs[1])-$(γs[Ng])-$(Ng)_e-$(ηs[1])-$(ηs[Nη])-$(Nη)_d-$(Δ).npy", EV)
end

function squeezing_amplitude(βs, Δ::Float64, n::Int, m::Int, γ::Float64 = 0.2)
    N = length(βs)

    EV = Array{ComplexF64, 2}(undef, N, 2)
    fpath = "$(homedir())/Documents/data/gqvdp/squeezing/a/$(n)_$(m)"
    mkpath(fpath)

    for j in 1:N
        beta = amplitude(γ, βs[j] * γ, n, m)
        dim = clamp(round(Int, beta * 3), 80, 200)
        basis = FockBasis(dim)
        H, J = build_system(1.0, γ, βs[j] * γ, Δ, n, m, basis, false)
        ss = steadystate.iterative(H, J)
        num = number(basis)
        EV[j, 1] = expect(num, ss)
        EV[j, 2] = expect(num^2, ss)
    end

    npzwrite("$(fpath)/beta-$(βs[1])-$(βs[N])-$(N)_d-$(Δ)_g-$(γ).npy", EV)
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
        "--bmin"
            help = "the min beta"
            arg_type = Float64
        "--bmax"
            help = "the max beta"
            arg_type = Float64
        "--bnum"
            help = "the num beta"
            arg_type = Int
    end

    parse_args(s)
end

args = parse_commandline()

βs = LinRange(args["bmin"], args["bmax"], args["bnum"])
squeezing_amplitude(βs, args["d"], args["n"], args["m"], args["g"])
