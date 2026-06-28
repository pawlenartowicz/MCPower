using Test
using JSON3
using OrderedCollections: OrderedDict
include("cases.jl"); using .BenchCases

const HERE = @__DIR__
const CASES = load_cases(joinpath(HERE, "benchmark_cases.json"))
byid = Dict(c.id => c for c in CASES)

@testset "load_cases" begin
    @test length(CASES) == 15
    ols = byid["ols_simple"]
    @test ols.family == "ols"
    @test ols.n_grid == collect(20:10:140)
    @test ols.n_sims["mcpower"] == 10000
    lme = byid["lme_multi"]
    @test lme.n_grid == collect(100:100:1000)
    @test lme.n_sims["naive"] == 30
    @test lme.cluster["n_clusters"] == 20
    glm = byid["glm_simple"]
    @test glm.baseline_p == 0.3                 # logit family default
    @test glm.max_failed_frac == 0.25
    @test byid["anova_2x2"].n_sims["tool"] == 1000      # per-case key-level override
    @test byid["anova_2x2"].n_sims["mcpower"] == 10000  # family default preserved
    @test byid["ols_correlated"].tool === nothing
    @test byid["glmm_slope"].cluster["random_slopes"] == ["x1"]
    @test byid["glmm_multislope"].cluster["random_slopes"] == ["x1", "x2"]
end

include("loops_jl.jl"); using .BenchLoops: _design, _draw_X
using Random, Statistics

@testset "design parse — Python convention" begin
    # anova_2x2: y = f1*f2  → cols [f1[2], f2[2], f1[2]:f2[2]]
    s = _design(byid["anova_2x2"])
    @test s.beta == [0.5, 0.5, 0.5]
    @test s.targets == [0]                       # "f1[2]" = first dummy (0-based, == Python)
    X = _draw_X(Xoshiro(7), 500, s)
    @test size(X) == (500, 3)
    @test X[:, 3] == X[:, 1] .* X[:, 2]           # ff product column (1-based access of 0-based cols)
    # ols_multi: y = x1+x2+x3+x4+x5, targets x1,x2 → first two continuous cols
    m = _design(byid["ols_multi"])
    @test length(m.beta) == 5
    @test m.targets == [0, 1]
    # anova_oneway4: 4-level factor → 3 dummy cols, target f[2] = first dummy
    o = _design(byid["anova_oneway4"])
    @test o.beta == [0.5, 0.5, 0.5]
    @test o.targets == [0]
    # ols_correlated: corr(x1,x2)=0.4 recovered in the draw
    cc = _design(byid["ols_correlated"])
    Xc = _draw_X(Xoshiro(7), 20000, cc)
    @test abs(cor(Xc[:, 1], Xc[:, 2]) - 0.4) < 0.03
    @test abs(cor(Xc[:, 1], Xc[:, 3])) < 0.03
end

using .BenchLoops: LOOPS

@testset "loop smoke — finite power" begin
    for cid in ("ols_multi", "glm_multi", "lme_multi", "glmm_simple")
        c = byid[cid]
        n = c.n_grid[fld(length(c.n_grid), 2) + 1]
        out = LOOPS[c.family]["best"](c, n, 20, 2137)
        pwr = Float64.(out[1])
        @test all(isfinite, pwr) && length(pwr) >= 1
        @test all(0.0 .<= pwr .<= 1.0)
    end
    # naive path on one ols + one glmm case
    for cid in ("ols_simple", "glmm_simple")
        c = byid[cid]
        out = LOOPS[c.family]["naive"](c, c.n_grid[end], 10, 2137)
        @test all(isfinite, Float64.(out[1]))
    end
end

# Top-level include (not inside the testset): defining BenchHarness.main and
# calling it in the same local scope trips Julia's world-age check.
include("harness.jl")

@testset "harness writes records" begin
    mktempdir() do dir
        out = joinpath(dir, "jl.json")
        BenchHarness.main(["--case", "ols_simple", "--methods", "loop_naive",
                           "--scale", "0.02", "--out", out])
        doc = JSON3.read(read(out, String))
        @test doc.meta.lang == "jl"
        @test doc.meta.n_sims_scale == 0.02
        @test !isempty(doc.records)
        r = doc.records[1]
        @test r.lang == "jl" && r.method == "loop_naive"
        @test isfinite(r.per_sim_s) && r.per_sim_s > 0
    end
end

@testset "design — cont×factor product (standalone fixture)" begin
    # mirrors test_benchmark.py::test_loop_design_cont_factor_product
    c = Case("cont_factor", "ols", "y = x1 + f + x1:f",
             "x1=0.40, f[2]=0.5, x1:f[2]=0.3", ["x1"], [100],
             Dict{String,Int}(), 0.8, OrderedDict("f" => "(factor,0.5,0.5)"),
             nothing, nothing, nothing, nothing, nothing)
    s = _design(c)                       # cols: x1, f[2], x1*f[2]
    @test s.beta == [0.40, 0.5, 0.3]
    @test s.targets == [0]
    X = _draw_X(Xoshiro(7), 500, s)
    @test X[:, 3] == X[:, 1] .* X[:, 2]
end

@testset "best vs naive power band (ols_simple)" begin
    c = byid["ols_simple"]
    best  = Float64.(LOOPS["ols"]["best"](c, 200, 2000, 2137)[1])[1]
    naive = Float64.(LOOPS["ols"]["naive"](c, 200, 500, 2137)[1])[1]
    @test abs(best - naive) < 0.08       # MC noise band; Julia RNG differs from py
end
