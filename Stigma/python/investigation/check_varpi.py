import numpy as np
from scipy.integrate import solve_ivp
from stigma_series import aseries, cpseries, aseriesp, cpseriesp, small, omega_r
from stigma_rhs import coupdiff

h, om, OmL, mu = 0.67, 0.31, 0.69, 4/3
H0, OmR = 100*h, omega_r(h)
print(f"{'stigma':>7} {'varpi(t0)':>12} {'min varpi':>11} {'max varpi':>11} "
      f"{'varpi<1 all?':>13} {'sqrt(stigma)':>13} {'stigma>varpi^2?':>16}")
for st in (0.6, 0.9, 1.2, 1.5, 2.0):
    t0 = small(om, h, st, mu)
    y0 = [aseries(t0,om,h,OmL,1.0,st,mu), cpseries(t0,om,h,OmL,1.0,st,mu),
          aseriesp(t0,om,h,OmL,1.0,st,mu), cpseriesp(t0,om,h,OmL,1.0,st,mu)]
    def wall(t,y,*a): return y[1]**2 - st
    wall.terminal = True
    def big(t,y,*a): return y[0] - 3.0
    big.terminal = True
    sol = solve_ivp(coupdiff,[t0,60.0],y0,args=(om,h,OmL,1.0,st,mu),
                    events=[wall,big], rtol=1e-9, atol=1e-30, method="LSODA")
    cp = sol.y[1]
    ok  = bool(np.all(cp < 1.0))
    c1  = bool(np.all(st > cp**2))          # constraint 1 along the whole run
    print(f"{st:7.2f} {y0[1]:12.8f} {cp.min():11.6f} {cp.max():11.6f} "
          f"{str(ok):>13} {np.sqrt(st):13.6f} {str(c1):>16}")
