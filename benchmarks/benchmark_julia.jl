# Julia benchmark of the TorC background ODE.
# RHS is auto-generated from torc/theory.py (see gen_julia_rhs.py) so the
# equations are character-exact copies of the Python ones.
#
# Writes results as TSV to stdout for the Python side to merge.

using OrdinaryDiffEq
using Sundials
using Printf
using Statistics

include(joinpath(@__DIR__, "torc_rhs.jl"))

function savegrid(x0)
    early = collect(range(x0, XSPLIT, length=N_EARLY+1))[1:end-1]
    late  = collect(range(XSPLIT, 0.0, length=N_LATE))
    vcat(early, late)
end

# E(z) = H/H(a=1), from da: E ∝ da * exp(-2x), normalised at x=0
function Evals(sol, xs)
    da = [sol(x)[2] for x in xs]
    E = da .* exp.(-2 .* xs)
    E ./ E[end]
end

const SOLVERS = [
    ("jl Tsit5",             Tsit5(),        false),
    ("jl Vern8",             Vern8(),        false),
    ("jl Vern9",             Vern9(),        false),
    ("jl Rodas5P [stiff]",   Rodas5P(),      true),
    ("jl CVODE_Adams",       CVODE_Adams(),  false),
    ("jl CVODE_BDF [stiff]", CVODE_BDF(),    true),
    ("jl ARKODE",            ARKODE(),       false),
]

function timeit(f, reps)
    f()                                  # warm up / compile
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns(); f(); push!(ts, (time_ns()-t0)/1e6)
    end
    median(ts)
end

println("case\tsolver\ttol\tms\tnf\tmaxrelerr")
for (name, x0, y0, p) in CASES
    xs = savegrid(x0)
    prob = ODEProblem(torc_rhs!, y0, (x0, 0.0), p)

    # high-precision in-Julia reference
    refsol = solve(prob, Vern9(), reltol=1e-14, abstol=1e-14, maxiters=10_000_000)
    Eref = Evals(refsol, xs)

    for tol in (1e-5, 1e-8)
        for (sname, alg, isstiff) in SOLVERS
            local ms, nf, err
            try
                f() = solve(prob, alg, reltol=tol, abstol=tol,
                            saveat=xs, maxiters=1_000_000)
                s = f()
                if s.retcode != ReturnCode.Success
                    @printf("%s\t%s\t%.0e\tNA\tNA\tFAILED(%s)\n", name, sname, tol, s.retcode)
                    continue
                end
                ms = timeit(f, 5)
                nf = try s.stats.nf catch; -1 end
                E  = Evals(s, xs)
                err = maximum(abs.((E .- Eref) ./ Eref))
                @printf("%s\t%s\t%.0e\t%.3f\t%d\t%.3e\n", name, sname, tol, ms, nf, err)
            catch e
                @printf("%s\t%s\t%.0e\tNA\tNA\tERROR(%s)\n", name, sname, tol,
                        first(string(typeof(e)), 30))
            end
        end
    end
end
# dump E(z) at a fixed tolerance so Python can verify equivalence
open(joinpath(@__DIR__, "julia_E.tsv"), "w") do io
    for (name, x0, y0, p) in CASES
        xs = savegrid(x0)
        prob = ODEProblem(torc_rhs!, y0, (x0, 0.0), p)
        s = solve(prob, Vern9(), reltol=1e-12, abstol=1e-12, saveat=xs, maxiters=10_000_000)
        E = Evals(s, xs)
        for (x, e) in zip(xs, E)
            @printf(io, "%s\t%.17g\t%.17g\n", name, x, e)
        end
    end
end
println("wrote julia_E.tsv")
println("JULIA BENCHMARK DONE")
