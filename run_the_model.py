import numpy as np
import matplotlib.pyplot as plt

import lattice_model as model

def plot_snap(model):
    plt.clf()
    plt.imshow(model.th)
    plt.clim([-75,75.])
    plt.xticks([])
    plt.yticks([])
    plt.title(str(model.t))
    plt.pause(0.01)
    plt.draw()


m = model.LatticeModel(kappa=1.e-4,dt=.5,nx=1024,tmax=2000,tavestart=1000)

# initial concentration
x0,y0 = np.pi,np.pi
r = np.sqrt((m.x-x0)[np.newaxis,...]**2+(m.y-y0)[...,np.newaxis]**2)
m.th = np.exp(-(r**2))*0

#for snapshot in m.run_with_snapshots(tsnapstart=0, tsnap=m.dt):

#    plot_snap(m)

m.run()

