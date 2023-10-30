import sys
sys.path.insert(0,"/Users/manujayadharan/git_repos/pysindy")



import pysindy
import numpy as np
from scipy.integrate import odeint
from pysindy import SINDy
from pysindy.optimizers import STLSQ
lorenz = lambda z,t : [10*(z[1] - z[0]),
                        z[0]*(28 - z[2]) - z[1],
                        z[0]*z[1] - 8/3*z[2]]
t = np.arange(0,2,.002)
x = odeint(lorenz, [-8,8,27], t)
opt = STLSQ(threshold=0.1, alpha=.5)
model = SINDy(optimizer=opt)
model.fit(x, t=t[1]-t[0])
print("STLSQ optimizer....")
model.print()


import pysindy as ps

threshold_vect = []
opt_adam = ps.optimizers.adam_STLSQ(threshold=0.1, alpha=.5, mom_memory=0.0, mom_init_iter=1,
                                    use_mom=True, variable_thresh=True, mom_inplace=True,
                                    threshold_vect=threshold_vect)
model_2 = SINDy(opt_adam)
model_2.fit(x, t=t[1]-t[0])
print(" adam-STLSQ optimizer....")

model_2.print()