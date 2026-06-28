# Load benchmark_cases.json and resolve each case against its family defaults.
# Mirrors cases.py — change together. Julia keeps the column convention 0-based
# (see loops_jl.jl) but case fields are plain data, identical to the Python Case.
module BenchCases

using JSON3
using OrderedCollections: OrderedDict

export Case, load_cases

# Fields mirror cases.py Case. @kwdef keeps the positional constructor (used by
# runtests.jl's standalone fixtures) and adds a keyword one (used by load_cases).
Base.@kwdef struct Case
    id::String
    family::String                          # "ols" | "logit" | "lme" | "glmm"
    formula::String
    effects::String
    targets::Vector{String}
    n_grid::Vector{Int}
    n_sims::Dict{String,Int}                 # {"mcpower","best","naive","tool"}
    target_power::Float64
    # OrderedDict (not Dict): the factor column layout in loops_jl._design follows
    # variable_types insertion order, mirroring Python's order-preserving dict.
    variable_types::OrderedDict{String,String} = OrderedDict{String,String}()
    cluster::Union{Nothing,Dict{String,Any}} = nothing
    tool::Union{Nothing,String} = nothing            # "simr"|"superpower"|"simglm"|nothing
    correlations::Union{Nothing,String} = nothing
    baseline_p::Union{Nothing,Float64} = nothing
    max_failed_frac::Union{Nothing,Float64} = nothing
end

# Recursively turn a JSON3.Object/Array into plain Dict{String,Any}/Vector so the
# cluster block (nested random_slopes array + scalars) is host-agnostic data.
_plain(x::JSON3.Object) = Dict{String,Any}(String(k) => _plain(v) for (k, v) in x)
_plain(x::JSON3.Array) = Any[_plain(v) for v in x]
_plain(x) = x

# JSON3 parses null as `nothing`; `get` with a `nothing` value returns nothing.
_str_or_nothing(x) = x === nothing ? nothing : String(x)
_f64_or_nothing(x) = x === nothing ? nothing : Float64(x)

function load_cases(path)::Vector{Case}
    doc = JSON3.read(read(path, String))
    defaults = doc["defaults"]
    out = Case[]
    for c in doc["cases"]
        fam = String(c["family"])
        d = defaults[Symbol(fam)]
        n = haskey(c, "n") ? c["n"] : d["n"]            # per-case grid override
        # range from:by:to is inclusive of `to` when divisible — matches Python's
        # range(from, to+1, by).
        grid = collect(Int(n["from"]):Int(n["by"]):Int(n["to"]))
        # n_sims: family default overlaid by any per-case key-level overrides.
        n_sims = Dict{String,Int}(String(k) => Int(v) for (k, v) in d["n_sims"])
        if haskey(c, "n_sims")
            for (k, v) in c["n_sims"]
                n_sims[String(k)] = Int(v)
            end
        end
        # baseline_p / max_failed_frac: per-case if present, else family default.
        base_p = haskey(c, "baseline_p") ? c["baseline_p"] : get(d, "baseline_p", nothing)
        mff = haskey(c, "max_failed_frac") ? c["max_failed_frac"] : get(d, "max_failed_frac", nothing)
        vtypes = OrderedDict{String,String}()
        if haskey(c, "variable_types")
            for (k, v) in c["variable_types"]            # JSON3.Object preserves order
                vtypes[String(k)] = String(v)
            end
        end
        push!(out, Case(
            id = String(c["id"]),
            family = fam,
            formula = String(c["formula"]),
            effects = String(c["effects"]),
            targets = String.(collect(c["targets"])),
            n_grid = grid,
            n_sims = n_sims,
            target_power = Float64(d["target_power"]),
            variable_types = vtypes,
            cluster = haskey(c, "cluster") && c["cluster"] !== nothing ? _plain(c["cluster"]) : nothing,
            tool = _str_or_nothing(c["tool"]),                  # explicit on every case
            correlations = haskey(c, "correlations") ? _str_or_nothing(c["correlations"]) : nothing,
            baseline_p = _f64_or_nothing(base_p),
            max_failed_frac = _f64_or_nothing(mff),
        ))
    end
    return out
end

end # module BenchCases
