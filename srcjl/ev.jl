using QuantumOptics
using NPZ
using Arpack
using LinearAlgebra
using ArgParse
using Base.Filesystem


function L_gqvdp(γ2::Float64, η::Float64, Δ::Float64, nl_eta::Int, nl_dis::Int, basis::FockBasis)
    a = destroy(basis)
    at = create(basis)
    H = Δ * at * a + 1im * η * (a^nl_eta - at^nl_eta)
    J = [a, sqrt(γ2) * a^nl_dis]

    liouvillian(H, J)
end

amplitude(γ2::Float64, η::Float64, nl_eta::Int, nl_dis::Int) = (2 * η * nl_eta / (nl_dis * γ2))^(1 / (2 * nl_dis - nl_eta))

function _wrap(γ2::Float64, η::Float64, Δ::Float64, nl_eta::Int, nl_dis::Int, basis::FockBasis, n_ev::Int, n_retry::Int = 5)
    L = L_gqvdp(γ2, η, Δ, nl_eta, nl_dis, basis)
    λs = zeros(ComplexF64, n_ev)
    for retries = 1:n_retry
        A = dense(L.data)
        _λs, nconv, niter, nmult, resid = eigs(L.data, nev=max(10 * retries, n_ev), which = :LR, tol = 1e-9,
                                               maxiter = 100000 * retries, ritzvec = false, check = 2)
        nfound = length(_λs)
        print("($nconv $niter $nmult $nfound) ")
        global λs[1:min(nfound, n_ev)] = _λs[1:min(nfound, n_ev)]
        if nfound >= n_ev
            break
        else
        end
    end

    λs
end

function _wrap_dense(γ2::Float64, η::Float64, Δ::Float64, nl_eta::Int, nl_dis::Int, basis::FockBasis, n_ev::Int)
    L = L_gqvdp(γ2, η, Δ, nl_eta, nl_dis, basis)
    A = collect(L.data)
    λs = eigvals(A)
    sort!(λs, by = x -> (-real(x), imag(x)))
    λs[1:n_ev]
end


function calc_eigvals(row::Int, γ2s, ηs, Δ::Float64, nl_eta::Int, nl_dis::Int, m::Int, n_ev::Int = 10)
    Nη = length(ηs)
    Ng = length(γ2s)

    fpath = "$(homedir())/Documents/data/gqvdp/ev/a/$(nl_eta)_$(nl_dis)/g-$(γ2s[1])-$(γ2s[Ng])-$(Ng)_e-$(ηs[1])-$(ηs[Nη])-$(Nη)_d-$(Δ)"
    mkpath(fpath)

    EV = Array{ComplexF64, 2}(undef, Nη, n_ev)
    for col = 1:Nη
        beta = amplitude(γ2s[row], ηs[col], nl_eta, nl_dis) * 3
        dim = clamp(round(Int, beta), m, 140)
        print("$col $beta $dim ")
        basis = FockBasis(dim)
        try
            EV[col, :] .= _wrap_dense(γ2s[row], ηs[col], Δ, nl_eta, nl_dis, basis, n_ev)
        catch e
            print(e)
        end
        println("")
    end

    npzwrite("$(fpath)/$(row - 1).npy", EV)
end

const etas = LinRange(0.0, 1.0, 100)
const gs = LinRange(0.01, 1, 100)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-n"
            help = "the driving power"
            arg_type = Int
        "-m"
            help = "the dissipation power"
            arg_type = Int
        "-r"
            help = "the row (1, 200)"
            arg_type = Int
    end

    parse_args(s)
end

args = parse_commandline()
calc_eigvals(args["r"], gs, etas, 0.4, args["n"], args["m"], 50)
