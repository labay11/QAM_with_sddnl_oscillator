using Base.Filesystem


DATA_PATH = joinpath(dirname(dirname(@__FILE__)), "data")
PLOT_PATH = joinpath(dirname(dirname(@__FILE__)), "plots")


function local_path(basepath::AbstractString, fname::AbstractString, n::Int=-1, m::Int=-1)
    path = joinpath(basepath, fname)
    if n > 0 && m > 0
        path = joinpath(path, "$(n)_$(m)")
    end

    mkpath(path)

    path
end


local_data_path(fname::AbstractString, n::Int=-1, m::Int=-1) = local_path(DATA_PATH, fname, n, m)

local_plot_path(fname::AbstractString, n::Int=-1, m::Int=-1) = local_path(PLOT_PATH, fname, n, m)
