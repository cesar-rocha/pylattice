import numpy as np
import matplotlib.pyplot as plt

import lattice_model as model

def plot_snap(model):
    plt.clf()
    plt.imshow(model.th)
    plt.clim([-6,6.])
    plt.xticks([])
    plt.yticks([])
    plt.title(str(model.t))
    plt.pause(0.01)
    plt.draw()

m = model.LatticeModel(kappa=1.e-4,dt=1.,nx=1024,tmax=1000,tavestart=200)

# initial concentration
x0,y0 = np.pi,np.pi
r = np.sqrt((m.x-x0)[np.newaxis,...]**2+(m.y-y0)[...,np.newaxis]**2)
m.th = np.exp(-(r**2))*0

m.th = np.cos(m.y)[...,np.newaxis] + np.zeros(m.nx)[np.newaxis,...]

t, var = [], []
maxth = []
for snapshot in m.run_with_snapshots(tsnapstart=0, tsnap=m.dt):

    #plot_snap(m)
    var.append(m.spec_var(m.thh))
    t.append(m.t)
    maxth.append(m.th.mean(axis=1).max())

    if m.t > m.tavestart:
        try:
            thbar = np.vstack([thbar,m.th.mean(axis=1)])
        except:
            thbar = m.th.mean(axis=1)
            


#m.run()

## plotting
plt.figure()
plt.plot(t,var,'k')
plt.xlabel('time')
plt.ylabel('tracer variance')
plt.title(r'$p=3.5$, $nmin=5$, $nmax=1024$')
plt.savefig('variance_time_series_nmin_5')

plt.figure()

theory = 4*np.cos(m.y)/m.dt
numerics = m.get_diagnostic('thbar')

plt.plot(m.y,thbar.T[...,::25],linewidth=1.,alpha=.25,color='.5')
plt.plot(m.y,theory,color='b',linewidth=2,label='Theory')
plt.plot(m.y,numerics,color='m',linewidth=2,label='Numerics')
plt.ylim(-5,5)

plt.xlabel('$y$')
plt.ylabel(r'$<\theta>$')
plt.legend(loc=3)
plt.xlim(0,2*np.pi)
plt.xticks([0,np.pi/2.,np.pi,3*np.pi/2.,2*np.pi],[r'$0$',r'$\pi/2$',
            r'$\pi$',r'$3 \pi/2$',r'$2 \pi$'])
plt.title(r'$p=3.5$, $nmin=5$, $nmax=1024$')
plt.savefig('x-averaged_tracer_nmin_5')

# the relative error
f = np.abs(theory) < 1.e-7
rel = (np.abs(theory-numerics)/theory)
rel[f] = 0.
rel = rel.std()


#ke = 1/8.
#plt.figure()
#for i in range(10):
#    if i == 0:
#        plt.plot(m.y,thbar[i*5]*np.exp(t[i*5]*ke),linewidth=3.,label='t='+str(i*5))
#    else:
#        plt.plot(m.y,thbar[i*5]*np.exp(t[i*5]*ke),linewidth=1.,label='t='+str(i*5))
#
#plt.xlabel('$y$')
#plt.ylabel(r'$<\theta> \times \mathrm{e}^{\kappa_e t}$')
#plt.legend(loc=9,ncol=3)
#plt.xlim(0,2*np.pi)
#plt.xticks([0,np.pi/2.,np.pi,3*np.pi/2.,2*np.pi],[r'$0$',r'$\pi/2$',
#            r'$\pi$',r'$3 \pi/2$',r'$2 \pi$'])
#plt.title(r'IVP')
#plt.savefig('ivp_cbar')
#

