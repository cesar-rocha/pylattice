import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import model_subclasses as model

from pyspec import spectrum as spec

def calc_Koc(model):
    # calculate the Koc
    thm = model.get_diagnostic('thbar')
    thmh = np.fft.rfft(thm)
    thay = np.fft.irfft(1j*m.kk*thmh)
    cby2 = thay**2

    grad2 =  model.get_diagnostic('grad2_th_bar')

    cby2 = 1.

    Koc = (grad2/cby2)*model.kappa
    D = (model.urms**2)*model.dt/4.

    return Koc, D


def variance_budget(model):

    # calculate the Koc
    thm = model.get_diagnostic('thbar')
    thmh = np.fft.rfft(thm)
    thmy = np.fft.irfft(1j*m.kk*thmh)
    cby2 = thmy**2

    cby2 = 1.

    th2 = model.get_diagnostic('th2m')/2.
    th2h = np.fft.rfft(th2)
    th2yy = np.fft.irfft(-((model.kk)**2)*th2h)

    grad2 =  model.get_diagnostic('grad2_th_bar')

    vth2 = model.get_diagnostic('vth2m')
    vth2h = np.fft.rfft(vth2)
    vth2y = np.fft.irfft(1j*model.kk*vth2h)

    D = (model.urms**2)*model.dt/4.
    Koc = (grad2/cby2)*model.kappa

    fkoc = ((model.y>=0.8)&(model.y<=1.5)) | ((model.y>=4.)&(model.y<=5.3))
    Keff = Koc[fkoc].mean()

    vthm = model.get_diagnostic('vthm')

    var_trans = -vth2y/16.
    eddy_diff = -vthm*thmy/4.
    eddy_diff2 = D*cby2
    eddy_diff2 = Keff*cby2
    eddy_diff = eddy_diff2

    diff_trans = model.kappa*th2yy
    diff = -model.kappa*grad2

    return var_trans, eddy_diff, eddy_diff2,diff_trans, diff


ps=np.array([4])

m = model.GyModel(kappa=5.e-6,urms=1,nx=4096*2,nmin=2.,nmax=120,
                  tmax=100,tavestart=10, power=ps[0],diagcadence=1)

t, var = [], []

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnap=m.dt):
    var.append(m.spec_var(m.thh))
    t.append(m.t)

Koc, D = calc_Koc(m)

var_trans, eddy_diff, eddy_diff2, diff_trans, diff = variance_budget(m)

eddy_diff = eddy_diff*np.ones(diff.size)

residual = eddy_diff + diff + var_trans + diff_trans

tit = 'Gy_p_'+str(ps[0])+'_nx_'+str(m.nx)+'_nmax_'+str(m.nmax)


# snapshot
plt.figure()
plt.contourf(m.x,m.y,m.th+m.G*m.y[...,np.newaxis],20)
plt.contour(m.x,m.y,m.th+m.G*m.y[...,np.newaxis],20,colors='k')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.savefig('figs/snapshot_'+tit)


# plot diffusivity
plt.figure()
plt.plot(Koc/D,m.y)
plt.xlim(0,5)
plt.plot([1.,1.],[0,10.])
plt.ylim(0,2*pi)
plt.xlabel(r'K$_{oc}$/D')
plt.ylabel(r'$y$')
plt.savefig('figs/Koc_D_'+tit)

# plot diffusivity
plt.figure()
plt.plot(Koc,m.y)
plt.xlim(0,5*D)
plt.plot([D,D],[0,10.])
plt.ylim(0,2*pi)
plt.xlabel(r'K$_{oc}$')
plt.ylabel(r'$y$')
plt.savefig('figs/Koc_'+tit)

# mean concentration
thbar_theory = m.G*m.y
thbar_numerics = m.get_diagnostic('thbar') +  m.G*m.y
thbar_snap = m.th.mean(axis=1) +  m.G*m.y

plt.figure()
plt.plot(thbar_theory,m.y,linewidth=2)
plt.plot(thbar_numerics,m.y,linewidth=2)
plt.plot(thbar_snap,m.y,linewidth=1,color='0.5')

#plt.xlim()
#plt.plot([D,D],[0,10.])
plt.ylim(0,2*pi)
plt.xlabel(r'$\bar{\theta}$')
plt.ylabel(r'$y$')
plt.savefig('figs/thbar_'+tit)

# plot variance budget
plt.figure()
plt.plot(eddy_diff,m.y,label='Eddy diff.')
plt.plot(diff,m.y,label='Dissipation')
plt.plot(var_trans,m.y,label='Variance transp.')
plt.plot(diff_trans,m.y,label='Diffusive transp.')
plt.plot(residual,m.y,label='Residual.')
plt.xlabel(r'x-averaged variance budget')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig('figs/varbudget_'+tit)

# plot variance time series
plt.figure()
plt.plot(t,var)
plt.xlabel(r'time')
plt.ylabel(r'tracer variance')
plt.savefig('figs/vartime_'+tit)

# the isotropic spectrum
kr, Ei = spec.calc_ispec(m.kk, m.ll, m.get_diagnostic('spec'))

plt.figure()
plt.loglog(kr,Ei)
plt.ylim(1.e-7,1.e-1)
plt.xlim(0,m.nx/2)
plt.xlabel(r'Wavenumber')
plt.ylabel(r'Variance density')
plt.savefig('figs/spec_'+tit)

print (Koc/D).mean()

np.savez('diag_'+tit,kr=kr,Ei=Ei,eddy_diff=eddy_diff,thbar=thbar_numerics,thbar_snap=
          thbar_snap, var_trans=var_trans,diff_trans=diff_trans,diff=diff,
          Koc=Koc, D=D)
