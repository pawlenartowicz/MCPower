# Julia benchmark harness (competitor-only): time the DIY loop tiers
# (loop_naive + loop_best, all four families incl. GLMM) over each case's n grid
# and write {meta, records} results JSON. Mirrors harness.py's timing protocol
# and record/meta shape — but records ONLY the loops (Julia has no MCPower engine
# port, so no mcpower/tool/find_sample_size tiers and no --threads 1 twin).

if !isdefined(Main, :BenchCases)
    include("cases.jl")
end
if !isdefined(Main, :BenchLoops)
    include("loops_jl.jl")
end

module BenchHarness

using ..BenchCases: Case, load_cases
using ..BenchLoops: LOOPS
using LinearAlgebra
using Dates
using Printf: @sprintf, @printf
using Statistics: median
using JSON3
using GLM, MixedModels, StatsModels, Distributions

# Pin BLAS to 1 thread before any timed work: per-thread BLAS under the loop_best
# thread pool would oversubscribe the machine and inflate the loop baseline,
# flattering MCPower. Mirrors harness.py pinning OMP/MKL/OPENBLAS=1.
BLAS.set_num_threads(1)

const CASES_PATH = joinpath(@__DIR__, "benchmark_cases.json")
const RECORDED_METHODS = ["loop_naive", "loop_best"]
# n_sims key per recorded method (the loop tiers' counts in benchmark_cases.json).
const _TIER = Dict("loop_best" => "best", "loop_naive" => "naive")
const _KIND = Dict("loop_best" => "best", "loop_naive" => "naive")
# Timing hygiene mirrors py: median of 3 reps for loop_best, 1 rep for loop_naive
# (after one discarded warm-up that absorbs JIT compilation).
const _REPS = Dict("loop_best" => 3, "loop_naive" => 1)

scaled_sims(count, scale) = max(1, round(Int, scale * count))
seed_for(n) = 2137 + Int(n)

# (model name, physical core count) from /proc/cpuinfo; graceful fallback.
# Mirrors harness.py._cpu_info.
function _cpu_info()
    model = "unknown"
    pairs = Set{Tuple{String,String}}()
    phys = ""
    try
        for line in eachline("/proc/cpuinfo")
            if startswith(line, "model name") && model == "unknown"
                model = strip(split(line, ":"; limit=2)[2])
            elseif startswith(line, "physical id")
                phys = strip(split(line, ":"; limit=2)[2])
            elseif startswith(line, "core id")
                push!(pairs, (phys, strip(split(line, ":"; limit=2)[2])))
            end
        end
    catch
    end
    return (model, isempty(pairs) ? Sys.CPU_THREADS : length(pairs))
end

function build_meta(scale)
    cpu_model, cores_physical = _cpu_info()
    return (
        lang = "jl",
        timestamp_utc = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS"),
        os = "$(Sys.KERNEL) $(Sys.ARCH)",
        cpu_model = cpu_model,
        cores_physical = cores_physical,
        cores_logical = Sys.CPU_THREADS,
        threads_mode = "auto",
        n_sims_scale = scale,
        lang_version = string(VERSION),
        packages = Dict(
            "GLM" => string(pkgversion(GLM)),
            "MixedModels" => string(pkgversion(MixedModels)),
            "StatsModels" => string(pkgversion(StatsModels)),
            "Distributions" => string(pkgversion(Distributions)),
            "JSON3" => string(pkgversion(JSON3)),
        ),
    )
end

fmt_vec(v) = "[" * join((@sprintf("%.3f", x) for x in v), ", ") * "]"

# One discarded warm-up (absorbs JIT compile), then median of `reps` timed reps.
function time_call(fn; warmup, reps)
    warmup && fn()
    times = Float64[]
    result = nothing
    for _ in 1:reps
        t0 = time_ns()
        result = fn()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return median(times), result
end

function run_loop(case::Case, kind, n, n_sims, seed)
    out = LOOPS[case.family][kind](case, n, n_sims, seed)
    return Float64.(out[1])
end

record(case::Case, method, n, n_sims, elapsed, power) = (
    case_id = case.id, family = case.family, lang = "jl", method = method,
    n = n, n_sims = n_sims, time_s = elapsed,
    per_sim_s = elapsed / n_sims, power = power,
)

function _parse_args(argv)
    case_arg = "all"; methods_arg = nothing
    out = "results/jl.json"; scale = 1.0
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--case"
            case_arg = argv[i + 1]; i += 2
        elseif a == "--methods"
            methods_arg = argv[i + 1]; i += 2
        elseif a == "--out"
            out = argv[i + 1]; i += 2
        elseif a == "--scale"
            scale = parse(Float64, argv[i + 1]); i += 2
        else
            error("unknown arg: $a")
        end
    end
    return case_arg, methods_arg, out, scale
end

function main(argv)
    case_arg, methods_arg, out, scale = _parse_args(argv)

    all_cases = load_cases(CASES_PATH)
    cases = if case_arg == "all"
        all_cases
    else
        matches = filter(c -> c.id == case_arg, all_cases)
        isempty(matches) && error("Case $(repr(case_arg)) not found. Available: $([c.id for c in all_cases])")
        matches
    end
    methods = methods_arg === nothing ? RECORDED_METHODS : String.(split(methods_arg, ","))

    records = []
    for case in cases
        for method in methods
            method in RECORDED_METHODS ||
                (println(stderr, "WARNING: unknown method $(repr(method)), skipping"); continue)
            kind = _KIND[method]
            n_sims = scaled_sims(case.n_sims[_TIER[method]], scale)
            reps = _REPS[method]
            println("\n=== $(case.id) $method ($n_sims sims/n) ===")
            @printf("%6s | %10s | %12s | power\n", "n", "time(s)", "per-sim(s)")
            println("-"^60)
            for n in case.n_grid
                t, pwr = time_call(() -> run_loop(case, kind, n, n_sims, seed_for(n));
                                   warmup=true, reps=reps)
                push!(records, record(case, method, n, n_sims, t, pwr))
                @printf("%6d | %10.4f | %12.6f | %s\n", n, t, t / n_sims, fmt_vec(pwr))
            end
        end
    end

    abs_out = abspath(out)
    mkpath(dirname(abs_out))
    open(abs_out, "w") do io
        JSON3.write(io, (meta = build_meta(scale), records = records))
    end
    println("\nWrote $(length(records)) records to $abs_out")
end

end # module BenchHarness

if abspath(PROGRAM_FILE) == @__FILE__
    BenchHarness.main(ARGS)
end
