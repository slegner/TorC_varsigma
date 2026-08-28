import time, numpy as np
from scipy.integrate import solve_ivp
from stigma_series import aseries, cpseries, aseriesp, cpseriesp, small, omega_r
from stigma_rhs import coupdiff

h, om, OmL, mu, A0 = 0.67, 0.31, 0.69, 4/3, 1.0
H0 = 100*h
for stigma in (0.9, 1.5):
    t0 = small(om, h, stigma, mu)
    y0 = [aseries(t0,om,h,OmL,A0,stigma,mu), cpseries(t0,om,h,OmL,A0,stigma,mu),
          aseriesp(t0,om,h,OmL,A0,stigma,mu), cpseriesp(t0,om,h,OmL,A0,stigma,mu)]
    print(f"\nstigma={stigma} t0={t0:.3e} y0={['%.4e'%v for v in y0]}", flush=True)
    def wall(t,y,*a): return y[1]**2 - stigma*(1+1e-5) if stigma<1 else y[1]**2 - stigma
    wall.terminal=True
    def abig(t,y,*a): return y[0]-1e10
    abig.terminal=True
    t1=time.time()
    sol = solve_ivp(coupdiff,[t0,60.0],y0,args=(om,h,OmL,A0,stigma,mu),
                    events=[wall,abig], rtol=1e-8, atol=1e-30, method="LSODA")
    print(f"  {time.time()-t1:.1f}s status={sol.status} nfev={sol.nfev} "
          f"tau_end={sol.t[-1]:.4e} a={sol.y[0,-1]:.4e} varpi^2={sol.y[1,-1]**2:.6f}", flush=True)
    print(f"  events: wall={len(sol.t_events[0])} abig={len(sol.t_events[1])}", flush=True)
    calH = sol.y[2]/sol.y[0]
    print(f"  calH range [{calH.min():.4e}, {calH.max():.4e}]   H0={H0:.3f}"
          f"   crosses H0: {np.any(np.diff(np.sign(calH-H0))!=0)}", flush=True)
