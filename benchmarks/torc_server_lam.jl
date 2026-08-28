# Julia ODE server for the TorC background equations WITH the Lambda term.
#
# Identical to torc_server.jl except that the parameter vector carries a sixth
# entry, lam, and the RHS is torc_rhs_lam.jl.  Generated from it by
# scripts/make_lam_backend.py -- keep the two in sync.
#
# WHY A SEPARATE PROCESS
# ----------------------
# Embedding Julia in the Python process (juliacall) and calling it from
# inside jax.pure_callback fails in every configuration tested:
#     THREADS=auto + HANDLE_SIGNALS=yes  -> hang
#     THREADS=1    + HANDLE_SIGNALS=no   -> hang
#     THREADS=auto + HANDLE_SIGNALS=no   -> SIGBUS
# The cause is JAX's runtime-controlled callback thread meeting a
# thread-sensitive embedded Julia runtime.  A separate process removes the
# shared runtime entirely: the Python side only ever touches a socket.
#
# PROTOCOL (little-endian, both sides x86-64)
#   request : n::Int32, tol::Float64,
#             P::Float64[6n]   (omega_m, omegar, OmL, stigma, mups2, lam per row)
#             X0::Float64[n]
#             Y0::Float64[3n]  (cp, da, dcp per row)
#   response: E::Float64[NOUT*n]   normalised so E[end] == 1 per row
#
# The save grid is NOT transmitted -- it is a deterministic function of x0,
# so the server rebuilds it.  That keeps a 50-point request at ~2 KB.
#
# Usage:  julia -t auto benchmarks/torc_server_lam.jl /tmp/torc_lam.sock

using Sockets
using OrdinaryDiffEq
using Base.Threads

include(joinpath(@__DIR__, "torc_rhs_lam.jl"))

const N_EARLY_S = N_EARLY
const N_LATE_S  = N_LATE
const NOUT      = N_EARLY_S + N_LATE_S

function savegrid(x0::Float64)
    xs = Vector{Float64}(undef, NOUT)
    step = (XSPLIT - x0) / N_EARLY_S
    @inbounds for k in 1:N_EARLY_S
        xs[k] = x0 + (k - 1) * step
    end
    step2 = (0.0 - XSPLIT) / (N_LATE_S - 1)
    @inbounds for k in 1:N_LATE_S
        xs[N_EARLY_S + k] = XSPLIT + (k - 1) * step2
    end
    xs[NOUT] = 0.0
    xs
end

function solve_row!(out, i, P, X0, Y0, tol)
    x0 = X0[i]
    xs = savegrid(x0)
    p  = @view P[:, i]
    y0 = [Y0[1, i], Y0[2, i], Y0[3, i]]
    prob = ODEProblem(torc_rhs!, y0, (x0, 0.0), collect(p))
    local s
    try
        s = solve(prob, Tsit5(), reltol=tol, abstol=tol, saveat=xs,
                  maxiters=100_000)
    catch
        @inbounds out[:, i] .= NaN
        return
    end
    if s.retcode != ReturnCode.Success || length(s.u) != NOUT
        @inbounds out[:, i] .= NaN
        return
    end
    @inbounds for k in 1:NOUT
        out[k, i] = s.u[k][2] * exp(-2 * xs[k])
    end
    e = out[NOUT, i]
    if !isfinite(e) || e == 0.0
        @inbounds out[:, i] .= NaN
    else
        @inbounds for k in 1:NOUT
            out[k, i] /= e
        end
    end
end

function handle(conn)
    while !eof(conn)
        n = read(conn, Int32)
        n <= 0 && break
        tol = read(conn, Float64)
        P  = Matrix{Float64}(undef, 6, n); read!(conn, P)
        X0 = Vector{Float64}(undef, n);    read!(conn, X0)
        Y0 = Matrix{Float64}(undef, 3, n); read!(conn, Y0)

        out = Matrix{Float64}(undef, NOUT, n)
        @threads for i in 1:n
            solve_row!(out, i, P, X0, Y0, tol)
        end
        write(conn, out)
        flush(conn)
    end
end

function main()
    path = length(ARGS) >= 1 ? ARGS[1] : "/tmp/torc.sock"
    isfile(path) && rm(path; force=true)

    # Force compilation of the whole solve path BEFORE announcing readiness,
    # so the first real request is not paying for it.
    # realistic ICs, matching the series expansion at the _SAFE point --
    # arbitrary values make the warmup solve fail noisily ("dt below epsilon")
    let P = reshape([0.3, 8.24e-5, 0.7, 1.5, 1.5, 0.75], 6, 1),
        X0 = [-20.44684595676843],
        Y0 = reshape([0.9999990000010455, 0.811912949043203,
                      -615.8295112001972], 3, 1),
        w = Matrix{Float64}(undef, NOUT, 1)
        solve_row!(w, 1, P, X0, Y0, 1e-8)
    end

    server = listen(path)
    println("TORC_SERVER_READY threads=", nthreads())
    flush(stdout)
    while true
        conn = accept(server)
        try
            handle(conn)
        catch e
            @warn "connection error" exception=e
        finally
            close(conn)
        end
    end
end

main()
