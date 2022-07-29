using QuantumOptics
using NPZ
using ArgParse
using Pardiso
using Statistics

include("model.jl")

function steadystate_mkl(L::SparseSuperOpType)
    n, m = length(L.basis_r[1]), length(L.basis_r[2])

    ps = MKLPardisoSolver()
    b = zeros(ComplexF64, n*m)

    # unity trace condition for the linear solvers
    w = mean(abs.(L.data))
    b[1] = w
    A = L.data + sparse(ones(n), 1:(n+1):n^2, fill(w, n), n*m, n*m)
    x = solve(ps, A, b)

    data = reshape(x, n, m)
    DenseOperator(L.basis_r[1], L.basis_r[2], data)
end

function squeezing_amplitude(βs, Δ::Float64, n::Int, m::Int, γ::Float64 = 0.2)
    N = length(βs)

    EV = Array{ComplexF64, 2}(undef, N, 2)
    fpath = "$(homedir())/Documents/data/gqvdp/squeezing/a/$(n)_$(m)"
    mkpath(fpath)

    ps = MKLPardisoSolver()

    for j in 1:N
        beta = amplitude(γ, βs[j] * γ, n, m)
        dim = clamp(round(Int, beta * 3), 80, 200)
        basis = FockBasis(dim - 1)
        L = build_system(1.0, γ, βs[j] * γ, Δ, n, m, basis)
        # ss = steadystate.iterative(H, J)
        ss = steadystate_mkl(L)

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
