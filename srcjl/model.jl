using QuantumOptics

function build_system(
    γ1::Float64, γ2::Float64, η::Float64, Δ::Float64,
    n::Int, m::Int,
    basis::FockBasis, full::Bool = true)

    a = destroy(basis)
    at = create(basis)
    H = Δ * at * a + 1im * η * (a^n - at^n)
    J = [sqrt(γ1) * a, sqrt(γ2) * a^m]

    full ? liouvillian(H, J) : (H, J)
end

build_system(
    γ2::Float64, η::Float64, Δ::Float64,
    n::Int, m::Int,
    dim::Int, full::Bool = true) = build_system(1.0, γ2, η, Δ, n, m, FockBasis(dim), full)

build_system(
    γ2::Float64, η::Float64, Δ::Float64,
    n::Int,
    dim::Int, full::Bool = true) = build_system(1.0, γ2, η, Δ, n, n, FockBasis(dim), full)

function eigvals(L::SuperOperator, n_ev::Int)
    A = collect(L.data)
    λs = eigvals(A)
    sort!(λs, by = x -> (-real(x), imag(x)))
    λs[1:n_ev]
end

eigvals(
    γ1::Float64, γ2::Float64, η::Float64, Δ::Float64,
    n::Int, m::Int,
    basis::FockBasis,
    n_ev::Int = 10) = eigvals(build_system(γ1, γ2, η, Δ, n, m, basis), n_ev)

eigvals(
    γ2::Float64, η::Float64, Δ::Float64,
    n::Int, m::Int,
    dim::Int,
    n_ev::Int = 10) = eigvals(build_system(γ2, η, Δ, n, m, dim), n_ev)

amplitude(γ2::Float64, η::Float64, n::Int, m::Int) = (2 * η * n / (m * γ2))^(1 / (2 * m - n))

amplitude(γ2::Float64, η::Float64, n) = (2 * η / γ2)^(1 / n)


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
