using QuantumOptics
using NPZ
using ArgParse
using Printf
using SpecialFunctions

include("model.jl")
include("utils.jl")

POINTS = Dict(
    3 => [
        (0.8, 0.47412171491586075),
        (0.6, 1.3755952741694344),
        (0.4, 1.984379582650444)  # , (0.3, 2, 40)
    ],
    4 => [
        (0.15, 0.5684021044214842),
        (0.1, 1.1423118881998557),
        (0.05, 1.4210869697181017)  # , (0.05, 2, 40)
    ]
)

function mc(γ::Float64, η::Float64, n::Int, dim::Int)
    basis = FockBasis(dim - 1)  # NOTE: dimension is -1 that of python
    a = destroy(basis)

    Δ = 0.4
    length=100000
    times = range(0, stop=1000, length=length)
    mu = amplitude(γ, η, n)
    β = rand()
    θ = rand()
    ψ0 = coherentstate(basis, 2 * mu * β * exp(1im * 2 * π * θ))

    emspath = local_data_path("ems", n, n)
    _ps = npzread("$emspath/1_$(γ)_$(η)_$(Δ)_$(dim)-P.npy")
    Ps = [Operator(basis, basis, _ps[j, :, :]) for j=1:n]
    if n == 3
        Πs = [dm(coherentstate(basis, mu * exp(1im * π * (1 + 2 * k) / n))) for k in [1, 3, 2]]
    else
        Πs = [dm(coherentstate(basis, mu * exp(1im * π * (1 + 2 * k) / n))) for k=2:(n+1)]
    end

    H, J = build_system(1.0, γ, η, Δ, n, n, basis, false)

    tout, ψt = timeevolution.mcwf(times, ψ0, H, J; maxiters=Int(5e9))
    l = size(tout)[1]
    res = Array{ComplexF64, 2}(undef, length, (2 * n) + 1)
    res[:, 1] = times[:]
    for j = 1:n
        res[1:l, j + 1] = expect(Ps[j], ψt)
        res[1:l, j + n + 1] = expect(Πs[j], ψt)
    end

    savepath = local_data_path("memory", n, n)
    path = mkpath("$savepath/1_$(γ)_$(η)_$(Δ)_$(dim)")

    npzwrite("$path/$(β)_$(θ).npy", res)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-n"
            help = "the power"
            arg_type = Int
    	"--dim"
    	    help = "the hilbert dim"
    	    arg_type = Int
            default = 50
        "--point"
            help = "the point to calculate"
            arg_type = Int
            default = 2
    end

    parse_args(s)
end

args = parse_commandline()

point = POINTS[args["n"]][args["point"]]
mc(point[1], point[2], args["n"], args["dim"])
