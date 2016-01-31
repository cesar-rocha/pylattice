from __future__ import division
import numpy as np
from numpy import pi, cos, sin, exp

class RWAdvection():
    """ A class that represents a two-dimensional RW
        model of advection """

    def __init__(self,
                nx=128,
                ny=None,
                Lx=2*pi,
                Ly=None,
                dt=0.5,
                tmax=1000,
                tavestart = 500,
                kappa=1.e-5,
                x=None,
                y=None,
                R = 1.):

        if ny is None: ny = nx
        if Ly is None: Ly = Lx

        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly

        self.dt = dt
        self.dt_2 = dt/2.
        self.tmax = tmax
        self.tavestart = tavestart
        self.t = 0.
        self.tc = 0

        self._init_velocity()

        # the initial position of an array of particles

        # a circle, radius R
        if x == None:
            th = np.arange(0,2*pi,0.001)
            x = R*np.cos(th)
            y = R*np.sin(th)

        self.xp = x
        self.yp = y

        self.x = np.linspace(-pi,pi,1500)
        self.y = np.linspace(-pi,pi,self.x.size)

    def _velocity(self):

        phase = 2*pi*np.random.rand(2,self.nmodes)
        phi, psi = phase[0], phase[1]

        Yn = self.n*self.yp[...,np.newaxis] + phase[0][np.newaxis,...]
        Xn = self.n*self.xp[...,np.newaxis] + phase[1][np.newaxis,...]

        self.u = (self.An*cos(Yn)).sum(axis=1)
        self.v = (self.An*cos(Xn)).sum(axis=1)


    def _init_velocity(self,nmodes=20,power=5.,nmin=3):

        self.nmodes = nmodes
        nmax = nmin+nmodes
        self.n = np.arange(nmin,nmax)[np.newaxis,...]
        An = (self.n/nmin)**(-power/2.)
        urms =  1.
        N = 2*urms/( np.sqrt( ((self.n/nmin)**-power).sum() ) )
        self.An = N*An

    def _advect(self):
        """ Advect particles at x, y """

        self.xp += self.dt_2*self.u
        self.yp += self.dt_2*self.v

        self.xp[self.xp<-np.pi] = self.xp[self.xp<-np.pi]+2*np.pi
        self.xp[self.xp>np.pi] = self.xp[self.xp>np.pi]-2*np.pi

        self.yp[self.xp<-np.pi] = self.yp[self.xp<-np.pi]+2*np.pi
        self.yp[self.xp>np.pi] =  self.yp[self.xp>np.pi]-2*np.pi

    def _step_forward(self):

        self._velocity()

        self._advect()

        self.tc += 1
        self.t += self.dt

    def run_with_snapshots(self, tsnapstart=0., tsnap=1):
        """Run the model forward, yielding to user code at specified intervals.
            """

        tsnapint = np.ceil(tsnap/self.dt)

        while(self.t < self.tmax):
            self._step_forward()
            if self.t>=tsnapstart and (self.tc%tsnapint)==0:
                yield self.t
        return

    def run(self):
        """Run the model forward without stopping until the end."""
        while(self.t < self.tmax):
            self._step_forward()
