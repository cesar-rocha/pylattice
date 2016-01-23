import numpy as np
import matplotlib.pyplot as plt

import lattice_model as model

def plot_snap(model):
    plt.clf()
    #plt.imshow(model.th)
    plt.pcolor(model.th)
    #plt.clim([-1.,1.])
    plt.xticks([])
    plt.yticks([])
    plt.title(str(model.t))
    plt.pause(0.01)
    plt.draw()

#m.run()

def calc_Koc(model):
    # calculate the Koc
    thm = model.get_diagnostic('thbar')
    thmh = np.fft.rfft(thm)
    thay = np.fft.irfft(1j*m.kk*thmh)
    cby2 = thay**2

    grad2 =  model.get_diagnostic('grad2_th_bar')

    Koc = (grad2/cby2)*model.kappa
    D = (model.urms**2)*model.dt/4.

    return Koc, D

m = model.LatticeModel(kappa=1.e-5,urms=5.e-2,dt=1.,nx=2048,nmin=5.,
                        tmax=20000,tavestart=10000, power=4.,source=True)

# initial concentration

print m.dx/m.lb

m.th = np.cos(m.y)[...,np.newaxis] + np.zeros(m.nx)[np.newaxis,...]

t, var = [], []
maxth = []

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnap=m.dt):

    #plot_snap(m)
    var.append(m.spec_var(m.thh))
    t.append(m.t)
    #maxth.append(m.th.mean(axis=1).max())

    #if m.t > m.tavestart:
    #    try:
    #        thbar = np.vstack([thbar,m.th.mean(axis=1)])
    #    except:
    #        thbar = m.th.mean(axis=1)

#m.run()

Koc, D = calc_Koc(m)

def variance_budget(model):

    # calculate the Koc
    thm = model.get_diagnostic('thbar')
    thmh = np.fft.rfft(thm)
    thmy = np.fft.irfft(1j*m.kk*thmh)
    cby2 = thmy**2

    th2 = model.get_diagnostic('th2m')/2.
    th2h = np.fft.rfft(th2)
    th2yy = np.fft.irfft(-((model.kk)**2)*th2h)

    grad2 =  model.get_diagnostic('grad2_th_bar')
    
    vth2 = model.get_diagnostic('vth2m')
    vth2h = np.fft.rfft(vth2)
    vth2y = np.fft.irfft(1j*model.kk*vth2h)

    D = (model.urms**2)*model.dt/4.
    Koc = (grad2/cby2)*model.kappa

    #vthm = m.get_diagnostic('vthm')

    var_trans = -vth2y/4.
    #eddy_diff = -vthm*np.sqrt(cby2)
    eddy_diff2 = D*cby2
    eddy_diff = eddy_diff2
    
    diff_trans = model.kappa*th2yy
    diff = -model.kappa*grad2

    return var_trans, eddy_diff, eddy_diff2,diff_trans, diff

var_trans, eddy_diff, eddy_diff2, diff_trans, diff = variance_budget(m)

np.savez('koc_4_dt_1',Koc=Koc,y=m.y,D=D,var_trans=var_trans,eddy_diff=eddy_diff,eddy_diff2=eddy_diff2,
                    diff_trans=diff_trans,diff=diff,t=t,var=var)


