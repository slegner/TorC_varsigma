# AUTO-GENERATED from torc/theory_lam.py by gen_julia_rhs_lam.py -- do not edit.
# Only transformation applied: Python ** -> Julia ^

function dda(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam)
    return (a^2*dcp^2*mups2*(-3*cp^2*stigma^2 - cp^4*(1 - 2*stigma)^2*stigma*(-3 + 2*stigma)*(1 + 2*stigma) + stigma^3*(1 + 8*stigma) - cp^6*(1 - 4*stigma^2)^2) + 40000*omegar*stigma*(-cp^2 + stigma)^2*(stigma + cp^2*(-1 + 4*stigma^2)) - 2*a*cp*da*dcp*mups2*(stigma + cp^2*(-1 + 4*stigma^2))*((7 - 4*stigma)*stigma^2 + 2*cp^2*stigma*(1 + 2*(-3 + stigma)*stigma) + cp^4*(-1 + 4*stigma^2)) - 40000*a^4*(-cp^2 + stigma)^2*(3*OmL*stigma^2 + cp^4*lam*(-1 + 4*stigma^2) + cp^2*stigma*(lam - 3*OmL + 8*lam*stigma + 12*OmL*stigma^2)) - cp^2*da^2*mups2*(cp^4*(1 - 2*stigma)^2*stigma*(-7 + 2*stigma)*(1 + 2*stigma) + stigma^3*(3 + 32*(-1 + stigma)*stigma) + cp^6*(1 - 4*stigma^2)^2 + cp^2*stigma^2*(3 - 16*(-1 + stigma)*stigma*(-1 + 2*stigma))))/(8*a*cp^2*mups2*stigma*(-cp^2 + stigma)^2*(1 - 4*stigma^2))
end

function ddcp(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam)
    return (40000*omegar*stigma*(-cp^2 + stigma)^2*(stigma - 4*stigma^2 + cp^2*(-1 + 4*stigma^2)) - cp^2*da^2*mups2*((5 - 8*stigma)*stigma + cp^2*(-1 + 4*stigma^2))*((7 - 4*stigma)*stigma^2 + 2*cp^2*stigma*(1 + 2*(-3 + stigma)*stigma) + cp^4*(-1 + 4*stigma^2)) - 40000*a^4*(-cp^2 + stigma)^2*(3*OmL*(1 - 4*stigma)*stigma^2 + cp^4*lam*(-1 + 4*stigma^2) + cp^2*stigma*(-7*lam - 3*OmL + 4*lam*stigma + 12*OmL*stigma^2)) + a^2*dcp^2*mups2*(stigma^3 + 4*stigma^4 - 32*stigma^5 - cp^4*stigma*(-1 + 2*stigma)*(1 + 2*stigma)*(3 + 4*(-3 + stigma)*stigma) - cp^6*(1 - 4*stigma^2)^2 + cp^2*stigma^2*(-3 + 8*stigma + 16*stigma^3)) - 2*a*cp*da*dcp*mups2*(cp^4*stigma*(-1 + 2*stigma)*(1 + 2*stigma)*(11 + 4*(-4 + stigma)*stigma) + cp^6*(1 - 4*stigma^2)^2 + stigma^3*(-1 + 16*stigma*(-2 + 3*stigma)) + cp^2*stigma^2*(11 - 16*stigma*(1 + stigma*(-1 + 2*stigma)))))/(8*a^2*cp*mups2*stigma*(-cp^2 + stigma)^2*(-1 + 4*stigma^2))
end

function torc_rhs!(du, u, p, x)
    cp, da, dcp = u
    omega_m, omegar, OmL, stigma, mups2, lam = p
    a = exp(x)
    fac = a / da
    du[1] = dcp * fac
    du[2] = dda(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam) * fac
    du[3] = ddcp(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam) * fac
    nothing
end

# name, x0, y0(cp,da,dcp), params(omega_m,omegar,OmL,stigma,mups2,lam)
const CASES = [
    ("fiducial", -20.44684595676843, [0.9999990000010455, 0.811912949043203, -615.8295112001972], [0.3, 8.24e-05, 0.7, 1.5, 1.5, 0.0]),
    ("lam mid", -20.44684595676843, [0.9999990000010455, 0.811912949043203, -615.8295112001972], [0.3, 8.24e-05, 0.7, 1.5, 1.5, 0.75]),
    ("lam high", -20.44684595676843, [0.9999990000010455, 0.811912949043203, -615.8295112001972], [0.3, 8.24e-05, 0.0, 1.5, 1.5, 1.5]),
    ("slow ridge", -21.552370740380223, [0.9999990000031586, 1.2387549073972401, -2838.297368778688], [0.9555, 8.24e-05, 0.4868, 1.1926, 0.6794, 0.3]),
    ("stigma low", -20.852344372456617, [0.9999990074089209, 0.8558171871109284, -959.3892266913441], [0.5, 8.24e-05, 0.7, 1.0001, 1.5, 1.0]),
    ("stigma high", -20.1335513084629, [0.9999990000002518, 0.6005795387845911, -333.0121763224039], [0.2, 8.24e-05, 1.2, 2.9, 2.5, 0.5]),
]
const XSPLIT = -1.308332819650179
const N_EARLY = 200
const N_LATE  = 300
