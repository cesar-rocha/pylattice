import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import renovate_wave_points as model

y = np.linspace(-pi/3,pi/3,225)
x = np.linspace(-pi/3,pi/3,225)

#x = np.hstack([np.linspace(-pi/5,-3*pi/5,100),np.linspace(pi/5,3*pi/5,100)])
#y = x.copy()

x,y = np.meshgrid(x,y)

ix,iy = x.shape

x = x.reshape(1,ix*iy).squeeze()
y = y.reshape(1,ix*iy).squeeze()


m = model.RWAdvection(kappa=1.e-4,dt=.1,tmax=50,R=2,x=x,y=y)

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnap=m.dt):

    x,y = m.xp.reshape(ix,iy),m.yp.reshape(ix,iy)

    plt.clf()
    plt.plot(x,y,'ko',markersize=1.)
    plt.xlim(-pi,pi)
    plt.ylim(-pi,pi)
    plt.xticks([])
    plt.yticks([])

    plt.pause(0.01)
    plt.draw()

    #plt.close()
