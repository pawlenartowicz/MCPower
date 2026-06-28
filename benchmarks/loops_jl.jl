# Julia DIY-loop baselines (naive off-the-shelf + best hand-rolled), keyed by
# family — the competitor-only third language for the cross-port speed benchmark.
# Timing comparators only; power is a Monte-Carlo sanity signal (Julia's Xoshiro
# RNG differs from Python's PCG64 / R's Mersenne-Twister, so cross-language loop
# power differs by MC noise — only wall-clock is comparable).
#
# The design parser (_design / _draw_X) mirrors loops_py._design / _draw_X
# — change together. Indices are kept 0-BASED to match the Python column
# convention literally (targets/offsets store the same integers as loops_py),
# adding +1 only at Julia array access. The GLMM kernel mirrors loops_r.R
# glmm_best_chunk (the family loops_py lacks).

if !isdefined(Main, :BenchCases)
    include("cases.jl")
end

module BenchLoops

using ..BenchCases: Case
using LinearAlgebra
using Random
using Statistics
using Distributions
using GLM
using MixedModels
using StatsModels

export _design, _draw_X, DesignSpec, LOOPS

const ALPHA = 0.05

# Parsed Case in the fixed column convention. Fields mirror the loops_py._design
# dict, 0-BASED (cont/factor indices, targets, slope_cols hold the same integers
# as Python). Column order: [continuous mains | factor dummy blocks in
# variable_types order | interaction terms in formula order].
struct DesignSpec
    n_cont::Int
    proportions::Vector{Vector{Float64}}
    n_dummies_of::Vector{Int}
    interactions::Vector{Tuple{Symbol,NTuple{2,Int}}}   # :cc/:cf/:ff, 0-based indices
    beta::Vector{Float64}                                # length P, column order
    targets::Vector{Int}                                 # 0-based column indices
    chol::Union{Nothing,Matrix{Float64}}                 # lower L, C = L L'
    slope_cols::Vector{Int}                              # 0-based, GLMM random slopes
end

# right-hand side after ~ or =, with any "(1|g)" random-effects term removed
function _rhs(formula::AbstractString)
    rhs = split(formula, r"[~=]"; limit=2)[2]
    return replace(rhs, r"\([^)]*\|[^)]*\)" => "")
end

# Cholesky L of the continuous-mains correlation matrix, or nothing. _draw_X uses
# Xc * L' to match loops_py's `cont @ chol.T` lower-triangular convention.
function _chol(case::Case, cont::Vector{String})
    case.correlations === nothing && return nothing
    nc = length(cont)
    C = Matrix{Float64}(I, nc, nc)
    for m in eachmatch(r"corr\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*=\s*(-?[0-9.]+)", case.correlations)
        a = String(m.captures[1]); b = String(m.captures[2]); r = parse(Float64, m.captures[3])
        i = findfirst(==(a), cont); j = findfirst(==(b), cont)
        C[i, j] = C[j, i] = r
    end
    return Matrix(cholesky(C).L)
end

function _design(case::Case)::DesignSpec
    rhs = _rhs(case.formula)
    fac_names = collect(keys(case.variable_types))          # insertion order (OrderedDict)
    proportions = Vector{Float64}[]
    for spec_str in values(case.variable_types)
        push!(proportions, [parse(Float64, m.match) for m in eachmatch(r"[0-9]*\.?[0-9]+", spec_str)])
    end
    n_dummies_of = [length(p) - 1 for p in proportions]

    cont = String[]
    for m in eachmatch(r"x\d+", rhs)
        name = String(m.match)
        name in cont || push!(cont, name)
    end
    n_cont = length(cont)
    n_dummies = sum(n_dummies_of; init=0)

    # interaction terms in formula order: (:cc,(ci,cj)) | (:cf,(ci,fi)) | (:ff,(fi,fj)), 0-based
    interactions = Tuple{Symbol,NTuple{2,Int}}[]
    for m in eachmatch(r"(\w+)\s*[:*]\s*(\w+)", rhs)
        a = String(m.captures[1]); b = String(m.captures[2])
        if a in cont && b in cont
            push!(interactions, (:cc, (findfirst(==(a), cont) - 1, findfirst(==(b), cont) - 1)))
        elseif a in cont && b in fac_names
            push!(interactions, (:cf, (findfirst(==(a), cont) - 1, findfirst(==(b), fac_names) - 1)))
        elseif a in fac_names && b in fac_names
            push!(interactions, (:ff, (findfirst(==(a), fac_names) - 1, findfirst(==(b), fac_names) - 1)))
        else
            error("unparseable interaction $a:$b in $(case.id)")
        end
    end

    _width(term) = let (kind, idx) = term
        kind === :cc ? 1 :
        kind === :cf ? n_dummies_of[idx[2] + 1] :
        n_dummies_of[idx[1] + 1] * n_dummies_of[idx[2] + 1]
    end

    inter_offset = Int[]            # 0-based start column of each interaction term
    off = n_cont + n_dummies
    for t in interactions
        push!(inter_offset, off)
        off += _width(t)
    end
    P = off

    fac_offset = Int[]             # 0-based start column of each factor dummy block
    foff = n_cont
    for nd in n_dummies_of
        push!(fac_offset, foff)
        foff += nd
    end

    # ('c', cont_idx0, _) or ('f', fac_idx0, level)
    function _atom(tok)
        tok = strip(tok)
        m = match(r"^(\w+)\[(\d+)\]$", tok)
        if m !== nothing
            fi = findfirst(==(String(m.captures[1])), fac_names)
            fi === nothing && error("cannot resolve effect token $tok in $(case.id)")
            return (:f, fi - 1, parse(Int, m.captures[2]))
        end
        if tok in cont
            return (:c, findfirst(==(tok), cont) - 1, 0)
        end
        fi = findfirst(==(tok), fac_names)
        if fi !== nothing && length(proportions[fi]) == 2
            return (:f, fi - 1, 2)               # bare 2-level factor -> level 2
        end
        error("cannot resolve effect token $tok in $(case.id)")
    end

    function col_of(name)                        # -> 0-based column index
        name = strip(name)
        if occursin(":", name)
            parts = strip.(split(name, ":"))
            a = _atom(parts[1]); b = _atom(parts[2])
            if a[1] === :c && b[1] === :c
                ti = findfirst(==((:cc, (a[2], b[2]))), interactions)
                return inter_offset[ti]
            elseif a[1] === :c && b[1] === :f
                ti = findfirst(==((:cf, (a[2], b[2]))), interactions)
                return inter_offset[ti] + (b[3] - 2)
            elseif a[1] === :f && b[1] === :f
                ti = findfirst(==((:ff, (a[2], b[2]))), interactions)
                return inter_offset[ti] + (a[3] - 2) * n_dummies_of[b[2] + 1] + (b[3] - 2)
            end
            error("unsupported interaction order in $name ($(case.id))")
        end
        a = _atom(name)
        return a[1] === :c ? a[2] : fac_offset[a[2] + 1] + (a[3] - 2)
    end

    beta = zeros(Float64, P)
    for part in split(case.effects, ",")
        kv = split(part, "=")
        beta[col_of(kv[1]) + 1] = parse(Float64, strip(kv[2]))      # +1: 0-based col -> Julia index
    end

    targets = [col_of(t) for t in case.targets]                     # 0-based

    # Random-slope design columns (0-based), via the same name->column map as the
    # fixed effects. Empty unless the case declares cluster.random_slopes (GLMM
    # slope cases). Mirrors loops_r.R spec$slope_cols.
    slope_cols = Int[]
    if case.cluster !== nothing && haskey(case.cluster, "random_slopes")
        for s in case.cluster["random_slopes"]
            push!(slope_cols, col_of(String(s)))
        end
    end

    return DesignSpec(n_cont, proportions, n_dummies_of, interactions, beta,
                      targets, _chol(case, cont), slope_cols)
end

# weighted 0-based level draw, mirrors loops_py rng.choice(len, p=props/sum)
function _draw_levels(rng, n::Int, props::Vector{Float64})
    cw = cumsum(props ./ sum(props))
    cw[end] = 1.0                                   # guard floating-point shortfall
    return [searchsortedfirst(cw, x) - 1 for x in rand(rng, n)]
end

# Build the (n, P) design (no intercept) in the fixed column order. Mirrors
# loops_py._draw_X — change together.
function _draw_X(rng, n::Int, spec::DesignSpec)::Matrix{Float64}
    Xc = randn(rng, n, spec.n_cont)
    if spec.chol !== nothing
        Xc = Xc * spec.chol'
    end
    dummy_blocks = Matrix{Float64}[]
    for props in spec.proportions
        levels = _draw_levels(rng, n, props)
        L = length(props)
        block = zeros(Float64, n, L - 1)
        for k in 1:(L - 1)
            @inbounds @. block[:, k] = (levels == k)
        end
        push!(dummy_blocks, block)
    end
    blocks = Matrix{Float64}[Xc]
    append!(blocks, dummy_blocks)
    for (kind, idx) in spec.interactions
        if kind === :cc
            i, j = idx[1] + 1, idx[2] + 1
            push!(blocks, reshape(Xc[:, i] .* Xc[:, j], n, 1))
        elseif kind === :cf
            ci, fi = idx[1] + 1, idx[2] + 1
            push!(blocks, Xc[:, ci] .* dummy_blocks[fi])           # column × block
        else  # ff, f1-major
            d1, d2 = dummy_blocks[idx[1] + 1], dummy_blocks[idx[2] + 1]
            push!(blocks, hcat([d1[:, k1] .* d2[:, k2]
                                for k1 in 1:size(d1, 2) for k2 in 1:size(d2, 2)]...))
        end
    end
    return hcat(blocks...)
end

# ---------------------------------------------------------------------------
# Shared kernel helpers
# ---------------------------------------------------------------------------

_logit(p) = log(p / (1 - p))
_logistic(x) = 1 / (1 + exp(-x))

_linalg_skip(e) = e isa SingularException || e isa LinearAlgebra.LAPACKException ||
                  e isa PosDefException

# Spread n observations across k clusters as evenly as possible; 0-based ids.
# Mirrors loops_py._assign_clusters.
function _assign_clusters(n::Int, k::Int)
    base, extra = divrem(n, k)
    ids = Int[]
    for c in 0:(k - 1)
        sz = base + (c < extra ? 1 : 0)
        append!(ids, fill(c, sz))
    end
    return ids
end

# Split n_sims across workers; drop empty chunks. Mirrors loops_py._split_counts.
function _split_counts(n_sims::Int, n_workers::Int)
    base, extra = divrem(n_sims, n_workers)
    out = Int[]
    for i in 0:(n_workers - 1)
        c = base + (i < extra ? 1 : 0)
        c > 0 && push!(out, c)
    end
    return out
end

# Run a *_best chunk across Base.Threads, pooling raw counts then dividing by
# usable. Per-chunk RNG Xoshiro(seed + (i-1)*100003) — fixed seed stride mirrors
# the py/R best pools (statistically equivalent, not bit-equal across worker
# counts). BLAS is pinned to 1 thread in harness.jl so per-thread BLAS does not
# oversubscribe and flatter MCPower.
function _parallel(chunk, n_sims::Int, seed::Int; max_workers::Int=typemax(Int))
    counts = _split_counts(n_sims, min(Threads.nthreads(), n_sims, max_workers))
    parts = Vector{Tuple{Vector{Int},Int,Int}}(undef, length(counts))
    Threads.@threads for i in eachindex(counts)
        parts[i] = chunk(counts[i], seed + (i - 1) * 100003)
    end
    t_rej = sum(p[1] for p in parts)
    f_rej = sum(p[2] for p in parts)
    usable = sum(p[3] for p in parts)
    denom = max(usable, 1)
    return (t_rej ./ denom, f_rej / denom, usable)
end

# MixedModels formula `.y ~ V1+…+VP + (re | .g)`, built once via @formula (like
# loops_r.R's as.formula(paste(...))). re = "1" (+ slope Vk) for the RE term.
function _re_formula(P::Int, slope_cols::Vector{Int})
    rhs = join(("V$(i)" for i in 1:P), " + ")
    re = isempty(slope_cols) ? "1" :
         join(vcat("1", ["V$(c + 1)" for c in slope_cols]), " + ")
    return eval(Meta.parse("@formula(_y ~ $rhs + ($re | _g))"))
end

# NamedTuple column table (Tables.jl-compatible) for a MixedModels fit.
function _mm_table(P::Int, X::Matrix{Float64}, y::Vector{Float64}, gfac::Vector{String})
    syms = (ntuple(j -> Symbol("V$(j)"), P)..., :_y, :_g)
    vals = (ntuple(j -> @view(X[:, j]), P)..., y, gfac)
    return NamedTuple{syms}(vals)
end

# ---------------------------------------------------------------------------
# OLS — mirror loops_py ols_best / ols_naive
# ---------------------------------------------------------------------------

function _ols_best_chunk(spec::DesignSpec, n::Int, n_sims::Int, seed::Int)
    rng = Xoshiro(seed)
    BETA = spec.beta; TARGETS = spec.targets; P = length(BETA)
    df_resid = n - (P + 1)
    tcrit = quantile(TDist(df_resid), 1 - ALPHA / 2)
    fcrit = quantile(FDist(length(TARGETS), df_resid), 1 - ALPHA)
    t_rej = zeros(Int, P); f_rej = 0; usable = 0
    ones_col = ones(n)
    idx = TARGETS .+ 2                         # 0-based col t -> coef position t+2 (intercept at 1)
    for _ in 1:n_sims
        X = _draw_X(rng, n, spec)
        y = X * BETA .+ randn(rng, n)
        Xd = hcat(ones_col, X)
        local XtXinv
        try                                    # a rank-deficient draw (factor level absent at small n) -> skip
            XtXinv = inv(Xd' * Xd)
        catch e
            _linalg_skip(e) && continue
            rethrow()
        end
        usable += 1
        coef = XtXinv * (Xd' * y)
        resid = y .- Xd * coef
        sigma2 = dot(resid, resid) / df_resid
        se = sqrt.(sigma2 .* diag(XtXinv))
        tstat = coef[2:end] ./ se[2:end]
        t_rej .+= Int.(abs.(tstat) .> tcrit)
        b = coef[idx]; V = sigma2 .* XtXinv[idx, idx]
        fstat = dot(b, V \ b) / length(TARGETS)
        fstat > fcrit && (f_rej += 1)
    end
    return (t_rej, f_rej, usable)
end

ols_best(case::Case, n, n_sims, seed) =
    (spec = _design(case); _parallel((c, s) -> _ols_best_chunk(spec, n, c, s), n_sims, seed))

function ols_naive(case::Case, n, n_sims, seed)
    rng = Xoshiro(seed)
    spec = _design(case); BETA = spec.beta; TARGETS = spec.targets; P = length(BETA)
    t_rej = zeros(Int, P); f_rej = 0; usable = 0
    ones_col = ones(n)
    idx = TARGETS .+ 2
    fcrit = quantile(FDist(length(TARGETS), n - (P + 1)), 1 - ALPHA)
    for _ in 1:n_sims
        X = _draw_X(rng, n, spec)
        y = X * BETA .+ randn(rng, n)
        Xd = hcat(ones_col, X)
        local m
        try
            m = lm(Xd, y)
        catch e
            _linalg_skip(e) && continue
            rethrow()
        end
        usable += 1
        pv = coeftable(m).cols[4][2:end]                # library Pr(>|t|), drop intercept
        t_rej .+= Int.(pv .< ALPHA)
        b = coef(m)[idx]; V = vcov(m)[idx, idx]          # joint Wald F via library covariance
        dot(b, V \ b) / length(TARGETS) > fcrit && (f_rej += 1)
    end
    denom = max(usable, 1)
    return (t_rej ./ denom, f_rej / denom, usable)
end

# ---------------------------------------------------------------------------
# GLM logit — mirror loops_py glm_best / glm_naive
# ---------------------------------------------------------------------------

# Newton/IRLS fit. Returns (beta, cov); (nothing, nothing) on failure or
# non-convergence. Mirrors loops_py._fit_logistic_irls.
function _fit_logistic_irls(X::Matrix{Float64}, y::Vector{Float64}; max_iter=25, tol=1e-6)
    k = size(X, 2)
    beta = zeros(k)
    converged = false
    for _ in 1:max_iter
        mu = _logistic.(X * beta)
        w = clamp.(mu .* (1 .- mu), 1e-8, Inf)           # guard W -> 0 (near-separation)
        XtWX = (X' .* w') * X
        grad = X' * (y .- mu)
        local step
        try
            step = XtWX \ grad
        catch e
            _linalg_skip(e) && return (nothing, nothing)
            rethrow()
        end
        beta = beta .+ step
        if maximum(abs.(step)) < tol
            converged = true
            break
        end
    end
    converged || return (nothing, nothing)
    mu = _logistic.(X * beta)
    w = clamp.(mu .* (1 .- mu), 1e-8, Inf)
    local cov
    try
        cov = inv((X' .* w') * X)
    catch e
        _linalg_skip(e) && return (beta, nothing)
        rethrow()
    end
    return (beta, cov)
end

function _glm_best_chunk(spec::DesignSpec, n::Int, baseline_p::Float64, n_sims::Int, seed::Int)
    rng = Xoshiro(seed)
    BETA = spec.beta; TARGETS = spec.targets; P = length(BETA)
    INTERCEPT = _logit(baseline_p)
    zcrit = quantile(Normal(), 1 - ALPHA / 2)
    chi2crit = quantile(Chisq(length(TARGETS)), 1 - ALPHA)
    t_rej = zeros(Int, P); f_rej = 0; conv = 0
    ones_col = ones(n)
    idx = TARGETS .+ 2
    for _ in 1:n_sims
        X = _draw_X(rng, n, spec)
        p = _logistic.(INTERCEPT .+ X * BETA)
        y = Float64.(rand(rng, n) .< p)
        Xd = hcat(ones_col, X)
        coef, cov = _fit_logistic_irls(Xd, y)
        (coef === nothing || cov === nothing) && continue
        conv += 1
        se = sqrt.(diag(cov))
        zstat = coef[2:end] ./ se[2:end]
        t_rej .+= Int.(abs.(zstat) .> zcrit)
        b = coef[idx]; V = cov[idx, idx]
        local wstat
        try
            wstat = dot(b, V \ b)
        catch e
            _linalg_skip(e) && continue
            rethrow()
        end
        wstat > chi2crit && (f_rej += 1)
    end
    return (t_rej, f_rej, conv)
end

glm_best(case::Case, n, n_sims, seed) =
    (spec = _design(case);
     _parallel((c, s) -> _glm_best_chunk(spec, n, case.baseline_p, c, s), n_sims, seed))

function glm_naive(case::Case, n, n_sims, seed)
    rng = Xoshiro(seed)
    spec = _design(case); BETA = spec.beta; TARGETS = spec.targets; P = length(BETA)
    INTERCEPT = _logit(case.baseline_p)
    chi2crit = quantile(Chisq(length(TARGETS)), 1 - ALPHA)
    t_rej = zeros(Int, P); f_rej = 0; conv = 0
    ones_col = ones(n)
    idx = TARGETS .+ 2
    for _ in 1:n_sims
        X = _draw_X(rng, n, spec)
        y = Float64.(rand(rng, n) .< _logistic.(INTERCEPT .+ X * BETA))
        Xd = hcat(ones_col, X)
        local m
        try
            m = glm(Xd, y, Binomial(), LogitLink())
        catch e
            e isa InterruptException && rethrow()
            continue
        end
        conv += 1
        pv = coeftable(m).cols[4][2:end]                # library Pr(>|z|), drop intercept
        t_rej .+= Int.(pv .< ALPHA)
        b = coef(m)[idx]; V = vcov(m)[idx, idx]
        local wstat
        try
            wstat = dot(b, V \ b)
        catch e
            _linalg_skip(e) && continue
            rethrow()
        end
        wstat > chi2crit && (f_rej += 1)
    end
    denom = max(conv, 1)
    return (t_rej ./ denom, f_rej / denom, conv)
end

# ---------------------------------------------------------------------------
# LME — mirror loops_py lme_best / lme_naive. Both tiers use the SAME
# MixedModels fit (same seed -> same data -> same fit); best earns its speed
# from precomputed criticals + manual Wald + threads, not a fit shortcut.
# ---------------------------------------------------------------------------

function _lme_best_chunk(spec::DesignSpec, n::Int, n_clusters::Int, tau::Float64,
                         fml, n_sims::Int, seed::Int)
    rng = Xoshiro(seed)
    BETA = spec.beta; TARGETS = spec.targets; P = length(BETA)
    zcrit = quantile(Normal(), 1 - ALPHA / 2)
    chi2crit = quantile(Chisq(length(TARGETS)), 1 - ALPHA)
    t_rej = zeros(Int, P); f_rej = 0; conv = 0
    cluster_ids = _assign_clusters(n, n_clusters)
    gfac = string.(cluster_ids)
    idx = TARGETS .+ 2
    for _ in 1:n_sims
        X = _draw_X(rng, n, spec)
        b = randn(rng, n_clusters) .* tau
        y = X * BETA .+ b[cluster_ids .+ 1] .+ randn(rng, n)
        m = _try_fit_lmm(fml, _mm_table(P, X, y, gfac))
        m === nothing && continue
        conv += 1
        cf = coef(m); V = vcov(m); se = sqrt.(diag(V))
        zstat = cf[2:end] ./ se[2:end]
        t_rej .+= Int.(abs.(zstat) .> zcrit)
        bt = cf[idx]; Vt = V[idx, idx]
        local wstat
        try
            wstat = dot(bt, Vt \ bt)
        catch e
            _linalg_skip(e) && continue
            rethrow()
        end
        wstat > chi2crit && (f_rej += 1)
    end
    return (t_rej, f_rej, conv)
end

function _try_fit_lmm(fml, tbl)
    try
        return MixedModels.fit(MixedModel, fml, tbl; progress=false)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
end

function _lme_tau_clusters(case::Case)
    icc = Float64(case.cluster["ICC"])
    return (Int(case.cluster["n_clusters"]), sqrt(icc / (1 - icc)))   # latent intercept TAU
end

function lme_best(case::Case, n, n_sims, seed)
    spec = _design(case)
    nc, tau = _lme_tau_clusters(case)
    fml = _re_formula(length(spec.beta), Int[])
    return _parallel((c, s) -> _lme_best_chunk(spec, n, nc, tau, fml, c, s), n_sims, seed)
end

function lme_naive(case::Case, n, n_sims, seed)
    rng = Xoshiro(seed)
    spec = _design(case); BETA = spec.beta; TARGETS = spec.targets; P = length(BETA)
    nc, tau = _lme_tau_clusters(case)
    fml = _re_formula(P, Int[])
    t_rej = zeros(Int, P); conv = 0
    cluster_ids = _assign_clusters(n, nc)
    gfac = string.(cluster_ids)
    for _ in 1:n_sims
        X = _draw_X(rng, n, spec)
        b = randn(rng, nc) .* tau
        y = X * BETA .+ b[cluster_ids .+ 1] .+ randn(rng, n)
        m = _try_fit_lmm(fml, _mm_table(P, X, y, gfac))
        m === nothing && continue
        conv += 1
        pv = coeftable(m).cols[4][2:(P + 1)]            # library Wald-z p-values, fixed effects
        t_rej .+= Int.(pv .< ALPHA)
    end
    # No joint test (mirror loops_py lme_naive: value never recorded).
    denom = max(conv, 1)
    return (t_rej ./ denom, 0.0, conv)
end

# ---------------------------------------------------------------------------
# GLMM (clustered logistic) — mirror loops_r.R glmm_best_chunk / glmm_naive.
# Random intercept on the LATENT logit scale (residual var pi^2/3), so
# TAU = sqrt(ICC/(1-ICC) * pi^2/3); one independent latent-scale slope RE per
# slope_cols entry. MixedModels Laplace fit + Wald z. The family loops_py lacks.
# ---------------------------------------------------------------------------

# GLMM Laplace fits are allocation-heavy, so the *_best thread pool oversubscribes
# Julia's stop-the-world GC + memory bandwidth: measured per-sim time bottoms out
# at 2 concurrent fits and regresses past it (the heavy glmm_multislope case at
# n=750 went 45ms@2 -> 174ms@16, i.e. slower than the serial naive loop). Cap the
# GLMM pool at 2 so loop_best stays faster than loop_naive across the whole
# family. OLS/GLM/LME fits are cheap and scale fine, so they keep the full pool.
const GLMM_MAX_WORKERS = 2

function _glmm_tau(case::Case)
    icc = Float64(case.cluster["ICC"])
    return sqrt(icc / (1 - icc) * pi^2 / 3)             # latent-scale logit ICC
end

# Clustered logit outcome: random intercept (tau) + independent latent-scale
# slope REs (slope_sd) per slope col. Mirrors loops_r.R's eta construction.
function _glmm_draw_y(rng, X, BETA, intercept, cluster_ids, n_clusters, tau,
                      slope_cols::Vector{Int}, slope_sd::Float64)
    n = size(X, 1)
    b = randn(rng, n_clusters) .* tau
    eta = intercept .+ X * BETA .+ b[cluster_ids .+ 1]
    for sc in slope_cols                                # sc 0-based -> column sc+1
        bs = randn(rng, n_clusters) .* slope_sd
        eta = eta .+ bs[cluster_ids .+ 1] .* X[:, sc + 1]
    end
    return Float64.(rand(rng, n) .< _logistic.(eta))
end

function _try_fit_glmm(fml, tbl)
    try
        return MixedModels.fit(MixedModel, fml, tbl, Bernoulli(); progress=false)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
end

function _glmm_best_chunk(spec::DesignSpec, n::Int, n_clusters::Int, tau::Float64,
                          intercept::Float64, slope_sd::Float64, fml, n_sims::Int, seed::Int)
    rng = Xoshiro(seed)
    BETA = spec.beta; TARGETS = spec.targets; P = length(BETA); SLOPE = spec.slope_cols
    zcrit = quantile(Normal(), 1 - ALPHA / 2)
    chi2crit = quantile(Chisq(length(TARGETS)), 1 - ALPHA)
    t_rej = zeros(Int, P); f_rej = 0; conv = 0
    cluster_ids = _assign_clusters(n, n_clusters)
    gfac = string.(cluster_ids)
    idx = TARGETS .+ 2
    for _ in 1:n_sims
        X = _draw_X(rng, n, spec)
        y = _glmm_draw_y(rng, X, BETA, intercept, cluster_ids, n_clusters, tau, SLOPE, slope_sd)
        m = _try_fit_glmm(fml, _mm_table(P, X, y, gfac))
        m === nothing && continue
        conv += 1
        cf = coef(m); V = vcov(m); se = sqrt.(diag(V))
        zstat = cf[2:end] ./ se[2:end]
        t_rej .+= Int.(abs.(zstat) .> zcrit)
        bt = cf[idx]; Vt = V[idx, idx]
        local wstat
        try
            wstat = dot(bt, Vt \ bt)
        catch e
            _linalg_skip(e) && continue
            rethrow()
        end
        wstat > chi2crit && (f_rej += 1)
    end
    return (t_rej, f_rej, conv)
end

# SLOPE_SD on the latent logit scale (sqrt slope_variance), or 0 when no slopes.
function _glmm_slope_sd(case::Case, spec::DesignSpec)
    isempty(spec.slope_cols) && return 0.0
    return sqrt(Float64(case.cluster["slope_variance"]))
end

function glmm_best(case::Case, n, n_sims, seed)
    spec = _design(case)
    nc = Int(case.cluster["n_clusters"])
    tau = _glmm_tau(case); intercept = _logit(case.baseline_p)
    slope_sd = _glmm_slope_sd(case, spec)
    fml = _re_formula(length(spec.beta), spec.slope_cols)
    return _parallel((c, s) -> _glmm_best_chunk(spec, n, nc, tau, intercept, slope_sd, fml, c, s),
                     n_sims, seed; max_workers=GLMM_MAX_WORKERS)
end

function glmm_naive(case::Case, n, n_sims, seed)
    rng = Xoshiro(seed)
    spec = _design(case); BETA = spec.beta; TARGETS = spec.targets; P = length(BETA)
    nc = Int(case.cluster["n_clusters"])
    tau = _glmm_tau(case); intercept = _logit(case.baseline_p)
    slope_sd = _glmm_slope_sd(case, spec)
    fml = _re_formula(P, spec.slope_cols)
    t_rej = zeros(Int, P); conv = 0
    cluster_ids = _assign_clusters(n, nc)
    gfac = string.(cluster_ids)
    for _ in 1:n_sims
        X = _draw_X(rng, n, spec)
        y = _glmm_draw_y(rng, X, BETA, intercept, cluster_ids, nc, tau, spec.slope_cols, slope_sd)
        m = _try_fit_glmm(fml, _mm_table(P, X, y, gfac))
        m === nothing && continue
        conv += 1
        pv = coeftable(m).cols[4][2:(P + 1)]            # library Wald-z p-values
        t_rej .+= Int.(pv .< ALPHA)
    end
    denom = max(conv, 1)
    return (t_rej ./ denom, 0.0, conv)
end

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

const LOOPS = Dict(
    "ols"   => Dict("best" => ols_best,  "naive" => ols_naive),
    "logit" => Dict("best" => glm_best,  "naive" => glm_naive),
    "lme"   => Dict("best" => lme_best,  "naive" => lme_naive),
    "glmm"  => Dict("best" => glmm_best, "naive" => glmm_naive),
)

end # module BenchLoops
