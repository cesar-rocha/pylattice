import numpy as np
import matplotlib.pyplot as plt

import lattice_model as model

m = model.LatticeModel(kappa=1.e-4,dt=1.,nx=512,tmax=3000,tavestart=1000)

# a test initial concentration
x0,y0 = np.pi,np.pi
r = np.sqrt((m.x-x0)[np.newaxis,...]**2+(m.y-y0)[...,np.newaxis]**2)
m.th = np.exp(-(r**2))

#plt.ion()
#for i in range(1000):
#    
#    m._velocity()
#
#    m._advect(direction='x')
#
#    m._advect(direction='y')
#
#    m._diffuse(half='False')
#
#    m._forcing()
#
#    plt.axis('equal')
#    plt.clf()
#    plt.imshow(m.th)
#    plt.clim([-100,100.])
#    plt.xticks([])
#    plt.yticks([])
#
#    plt.pause(0.005)
#    plt.draw()
#    plt.ioff()
#

m.run()

