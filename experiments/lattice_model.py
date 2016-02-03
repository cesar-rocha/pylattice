from __future__ import division
import numpy as np
from numpy import pi, cos, sin, exp

class LatticeModel():
    """ A class that represents a two-dimensional lattice
        model of advection-diffusion with large-scale
        sinusoidal source """

    def __init__(self,
                nx=128,
                ny=None,
                Lx=2*pi,
                Ly=None,
                dt=0.5,
                tmax=1000,
                tavestart = 500,
                kappa=1.e-5,
                urms = 1.,
                power = 3.5,
                nmin = 5.,
                nmax = None,
                source=True,
                diagnostics_list='all'):

        if ny is None: ny = nx
        if Ly is None: Ly = Lx

        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly

        self.dt = dt
        self.dt_2 = dt/2.
        self.dt_4 = dt/4.
        self.tmax = tmax
        self.tavestart = tavestart
        self.t = 0.
        self.tc = 0

        self.kappa = kappa

        self.nmin = nmin
        if nmax:
            self.nmax = nmax
        else:
            self.nmax = nx

        self.power = power
        self.urms = urms

        self.source=source

        self.diagnostics_list = diagnostics_list

        self._initialize_grid()

        self._init_velocity()

        self._initialize_diagnostics()

        self.even = True
        self.odd = False

    def _initialize_grid(self):
        """ Initialize lattice and spectral space grid """

        # physical space grids
        self.dx, self.dy = self.Lx/(self.nx), self.Ly/(self.ny)

        self.x = np.linspace(0.,self.Lx-self.dx,self.nx)
        self.y = np.linspace(0.,self.Ly-self.dy,self.ny)

        self.xi, self.yi = np.meshgrid(self.x,self.y)

        self.ix, self.iy = np.meshgrid(range(self.nx),
                                       range(self.ny))

        # wavenumber grids
        self.dk = 2.*pi/self.Lx
        self.dl = 2.*pi/self.Ly
        self.nl = self.ny
        self.nk = self.nx/2+1
        self.ll = self.dl*np.append( np.arange(0.,self.nx/2),
            np.arange(-self.nx/2,0.) )
        self.kk = self.dk*np.arange(0.,self.nk)

        self.k, self.l = np.meshgrid(self.kk, self.ll)
        self.ik = 1j*self.k
        self.il = 1j*self.l

        # constant for spectral normalizations
        self.M = self.nx*self.ny
        self.M2 = self.M**2

        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt( self.wv2 )

    def _velocity(self):

        phase = 2*pi*np.random.rand(2,self.nmax-self.nmin)
        phi, psi = phase[0], phase[1]

        Yn = self.n*self.y[...,np.newaxis] + phase[0][np.newaxis,...]
        Xn = self.n*self.x[...,np.newaxis] + phase[1][np.newaxis,...]

        u = (self.An*cos(Yn*self.dl)).sum(axis=1)
        v = (self.An*cos(Xn*self.dk)).sum(axis=1)

        self.u = u[...,np.newaxis]
        self.v = v[np.newaxis,...]

    def _init_velocity(self):

        self.n = np.arange(self.nmin,self.nmax)[np.newaxis,...]
        An = (self.n/self.nmin)**(-self.power/2.)
        N = 2*self.urms/( np.sqrt( ((self.n/self.nmin)**-self.power).sum() ) )
        self.An = N*An
        #self.An = np.sqrt(2.)

        #self.An = 2*urms

        # estimate the Batchelor scale
        S = np.sqrt( ((self.An*self.n*self.dk)**2).sum()/2. )
        self.lb = np.sqrt(self.kappa/S)

        #assert self.lb > self.dx, "**Warning: Batchelor scale not resolved."

    def _advect(self,direction='x',n=1):
        """ Advect th on a lattice given u and v,
            and the current index array ix, iy

            n is the number of substeps
            n=1 for doing the full advection-diffusion,
            n=2 for doing half the advection, etc """

        if direction == 'x':
            ix_new = self.ix.copy()
            dindx = -np.round(self.u*self.dt_2/n/self.dx).astype(int)
            ix_new  = self.ix + dindx
            ix_new[ix_new<0] = ix_new[ix_new<0] + self.nx
            ix_new[ix_new>self.nx-1] = ix_new[ix_new>self.nx-1] - self.nx
            self.th = self.th[self.iy,ix_new]

        elif direction == 'y':

            iy_new = self.iy.copy()
            dindy = -np.round(self.v*self.dt_2/n/self.dy).astype(int)
            iy_new  = self.iy + dindy
            iy_new[iy_new<0] = iy_new[iy_new<0] + self.ny
            iy_new[iy_new>self.ny-1] = iy_new[iy_new>self.ny-1] - self.ny
            self.th = self.th[iy_new,self.ix]

            # advection + source
            #y = self.y[...,np.newaxis] + np.zeros(self.x.size)[np.newaxis,...]
            #v = self.v + np.zeros(self.y.size)[...,np.newaxis]
            #sy = np.sin(self.dl*y)
            #syn = np.sin(self.dl*(y+v*self.dt_2/n))
            #v =  np.ma.masked_array(v, v == 0.)
            #self.forcey = (sy[iy_new,self.ix]-sy)/(self.dl*v)
            #self.forcey = (syn-sy)/(self.dl*v)
            #self.forcey[v.mask] = (self.dt_2/n)*np.cos(self.dl*y[v.mask])
            #self.th = self.th[iy_new,self.ix] + self.forcey

    def _diffuse(self, n=1):
        """ Diffusion """

        self.thh = np.fft.rfft2(self.th)
        self.thh = self.thh*exp(-(self.dt/n)*self.kappa*self.wv2)
        self.th = np.fft.irfft2(self.thh)

    def _source(self,direction='x',n=1):
        if direction == 'x':
            self.th += (self.dt/n)*np.cos(self.dl*self.y)[...,np.newaxis]
        elif direction == 'y':
            # a brutal way
            #self.th += (self.dt/n)*np.cos(self.dl*self.y)[...,np.newaxis]
            pass

    def _step_forward(self):

        self._velocity()

        # x-dir
        self._advect(direction='x',n=2)
        self._source(direction='x',n=2)
        self._diffuse(n=4)
        self._advect(direction='x',n=2)
        self._source(direction='x',n=2)
        self._diffuse(n=4)

        # y-dir
        self._advect(direction='y',n=2)
        self._source(direction='y',n=2)
        self._diffuse(n=4)
        self._advect(direction='y',n=2)
        self._source(direction='y',n=2)
        self._diffuse(n=4)

        self._calc_diagnostics()

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

    def _calc_diagnostics(self):
        # here is where we calculate diagnostics
        if (self.t>=self.dt) and (self.t>=self.tavestart):
            self._increment_diagnostics()

    # diagnostic stuff follow
    def _initialize_diagnostics(self):
        # Initialization for diagnotics
        self.diagnostics = dict()

        self._setup_diagnostics()

        if self.diagnostics_list == 'all':
            pass # by default, all diagnostics are active
        elif self.diagnostics_list == 'none':
            self.set_active_diagnostics([])
        else:
            self.set_active_diagnostics(self.diagnostics_list)

    def _setup_diagnostics(self):
        """Diagnostics setup"""

        self.add_diagnostic('var',
            description='Tracer variance',
            function= (lambda self: self.spec_var(self.thh))
        )

        self.add_diagnostic('thbar',
            description='x-averaged tracer',
            function= (lambda self: self.thm)
        )

        self.add_diagnostic('grad2_th_bar',
            description='x-averaged gradient square of th',
            function= (lambda self: self.gradth2m)
        )

        self.add_diagnostic('vth2m',
            description='x-averaged triple advective term v th2',
            function= (lambda self: self.vth2m)
            )

        self.add_diagnostic('th2m',
            description='x-averaged  th2',
            function= (lambda self: self.th2m)
            )

        self.add_diagnostic('vthm',
            description='x-averaged, y-direction tracer flux',
            function= (lambda self: (self.v*self.tha).mean(axis=1))
        )

        self.add_diagnostic('fluxy',
            description='x-averaged, y-direction tracer flux',
            function= (lambda self: (self.v*self.th).mean(axis=1))
        )

        self.add_diagnostic('spec',
            description='spec of anomalies about x-averaged flow',
            function= (lambda self: np.abs(np.fft.rfft2(
                        self.th-self.th.mean(axis=1)[...,np.newaxis]))**2/self.M2)
        )


    def _set_active_diagnostics(self, diagnostics_list):
        for d in self.diagnostics:
            self.diagnostics[d]['active'] == (d in diagnostics_list)

    def add_diagnostic(self, diag_name, description=None, units=None, function=None):
        # create a new diagnostic dict and add it to the object array

        # make sure the function is callable
        assert hasattr(function, '__call__')

        # make sure the name is valid
        assert isinstance(diag_name, str)

        # by default, diagnostic is active
        self.diagnostics[diag_name] = {
           'description': description,
           'units': units,
           'active': True,
           'count': 0,
           'function': function, }

    def describe_diagnostics(self):
        """Print a human-readable summary of the available diagnostics."""
        diag_names = self.diagnostics.keys()
        diag_names.sort()
        print('NAME               | DESCRIPTION')
        print(80*'-')
        for k in diag_names:
            d = self.diagnostics[k]
            print('{:<10} | {:<54}').format(
                 *(k,  d['description']))

    def _increment_diagnostics(self):
        # compute intermediate quantities needed for some diagnostics

        self._calc_derived_fields()

        for dname in self.diagnostics:
            if self.diagnostics[dname]['active']:
                res = self.diagnostics[dname]['function'](self)
                if self.diagnostics[dname]['count']==0:
                    self.diagnostics[dname]['value'] = res
                else:
                    self.diagnostics[dname]['value'] += res
                self.diagnostics[dname]['count'] += 1

    def _calc_derived_fields(self):

        """ Calculate derived field necessary for diagnostics """

        self.thh = np.fft.rfft2(self.th)

        # x-averaged tracer field
        self.thm = self.th.mean(axis=1)
        #self.thmh = np.fft.rfft(self.thm)
        #self.thm_y = np.fft.irfft(1j*self.kk*self.thmh)

        # anomaly about the x-averaged field
        self.tha = self.th-self.thm[...,np.newaxis]
        self.thah = np.fft.rfft2(self.tha)

        # x-averaged gradient squared
        gradx = np.fft.irfft2(1j*self.k*self.thah)
        grady = np.fft.irfft2(1j*self.l*self.thah)
        self.gradth2m = (gradx**2 + grady**2).mean(axis=1)

        # Osborn-Cox amplification factor
        #self.thm_y = 4*np.sin(self.y*self.dl)
        #thm_y = self.block_average(self.thm_y)
        #gradth2m = self.block_average(self.gradth2m)

        #self.A2_OC = gradth2m / thm_y**2
        #self.A2_OC[thm_y < 1.e-14] = np.nan

        # triple term
        self.vth2m = (self.v*(self.tha**2)).mean(axis=1)

        # diff transport
        self.th2m = (self.tha**2).mean(axis=1)

    def get_diagnostic(self, dname):
        return (self.diagnostics[dname]['value'] /
                self.diagnostics[dname]['count'])

    def spec_var(self, ph):
        """ compute variance of p from Fourier coefficients ph """
        var_dens = 2. * np.abs(ph)**2 / self.M**2
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
        var_dens[...,0] = var_dens[...,0]/2.
        var_dens[...,-1] = var_dens[...,-1]/2.
        return var_dens.sum()

    def block_average(self,A, nblocks = 256):
        """ Block average A onto A blocks """

        nave = self.nx/nblocks
        Ab = np.empty(nblocks)

        for i in range(nblocks):
            Ab[i] = A[i*nave:(i+1)*nave].mean()

        return Ab

class LatticeModelGy():
    """ A class that represents a two-dimensional lattice
        model of advection-diffusion with large-scale
        sinusoidal source """

    def __init__(self,
                nx=128,
                ny=None,
                Lx=2*pi,
                Ly=None,
                dt=0.5,
                tmax=1000,
                tavestart = 500,
                kappa=1.e-5,
                urms = 1.,
                power = 3.5,
                nmin = 5.,
                nmax = None,
                G = 1.,
                diagnostics_list='all',
                cadence = 5):

        if ny is None: ny = nx
        if Ly is None: Ly = Lx

        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly

        self.dt = dt
        self.dt_2 = dt/2.
        self.dt_4 = dt/4.
        self.tmax = tmax
        self.tavestart = tavestart
        self.t = 0.
        self.tc = 0

        self.G = G
        self.kappa = kappa

        self.nmin = nmin
        if nmax:
            self.nmax = nmax
        else:
            self.nmax = nx

        self.power = power
        self.urms = urms

        self.diagnostics_list = diagnostics_list
        self.cadence = cadence

        self._initialize_grid()

        self._init_velocity()

        self._initialize_diagnostics()

        self.even = True
        self.odd = False

    def _initialize_grid(self):
        """ Initialize lattice and spectral space grid """

        # physical space grids
        self.dx, self.dy = self.Lx/(self.nx), self.Ly/(self.ny)

        self.x = np.linspace(0.,self.Lx-self.dx,self.nx)
        self.y = np.linspace(0.,self.Ly-self.dy,self.ny)

        self.xi, self.yi = np.meshgrid(self.x,self.y)

        self.ix, self.iy = np.meshgrid(range(self.nx),
                                       range(self.ny))

        # wavenumber grids
        self.dk = 2.*pi/self.Lx
        self.dl = 2.*pi/self.Ly
        self.nl = self.ny
        self.nk = self.nx/2+1
        self.ll = self.dl*np.append( np.arange(0.,self.nx/2),
            np.arange(-self.nx/2,0.) )
        self.kk = self.dk*np.arange(0.,self.nk)

        self.k, self.l = np.meshgrid(self.kk, self.ll)
        self.ik = 1j*self.k
        self.il = 1j*self.l

        # constant for spectral normalizations
        self.M = self.nx*self.ny
        self.M2 = self.M**2

        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt( self.wv2 )

    def _velocity(self):

        phase = 2*pi*np.random.rand(2,self.nmax-self.nmin)
        phi, psi = phase[0], phase[1]

        Yn = self.n*self.y[...,np.newaxis] + phase[0][np.newaxis,...]
        Xn = self.n*self.x[...,np.newaxis] + phase[1][np.newaxis,...]

        u = (self.An*cos(Yn*self.dl)).sum(axis=1)
        v = (self.An*cos(Xn*self.dk)).sum(axis=1)

        self.u = u[...,np.newaxis]
        self.v = v[np.newaxis,...]

    def _init_velocity(self):

        self.n = np.arange(self.nmin,self.nmax)[np.newaxis,...]
        An = (self.n/self.nmin)**(-self.power/2.)
        N = 2*self.urms/( np.sqrt( ((self.n/self.nmin)**-self.power).sum() ) )
        self.An = N*An
        #self.An = np.sqrt(2.)

        #self.An = 2*urms

        # estimate the Batchelor scale
        S = np.sqrt( ((self.An*self.n*self.dk)**2).sum()/2. )
        self.lb = np.sqrt(self.kappa/S)

        #assert self.lb > self.dx, "**Warning: Batchelor scale not resolved."

    def _advect(self,direction='x',n=1):
        """ Advect th on a lattice given u and v,
            and the current index array ix, iy

            n is the number of substeps
            n=1 for doing the full advection-diffusion,
            n=2 for doing half the advection, etc """

        if direction == 'x':
            ix_new = self.ix.copy()
            dindx = -np.round(self.u*self.dt_2/n/self.dx).astype(int)
            ix_new  = self.ix + dindx
            ix_new[ix_new<0] = ix_new[ix_new<0] + self.nx
            ix_new[ix_new>self.nx-1] = ix_new[ix_new>self.nx-1] - self.nx
            self.th = self.th[self.iy,ix_new]

        elif direction == 'y':

            iy_new = self.iy.copy()
            dindy = -np.round(self.v*self.dt_2/n/self.dy).astype(int)
            iy_new  = self.iy + dindy
            iy_new[iy_new<0] = iy_new[iy_new<0] + self.ny
            iy_new[iy_new>self.ny-1] = iy_new[iy_new>self.ny-1] - self.ny
            self.th = self.th[iy_new,self.ix] + self.G*self.v*self.dt_2/n

    def _diffuse(self, n=1):
        """ Diffusion """

        self.thh = np.fft.rfft2(self.th)
        self.thh = self.thh*exp(-(self.dt/n)*self.kappa*self.wv2)
        self.th = np.fft.irfft2(self.thh)

    def _step_forward(self):

        self._velocity()

        # x-dir
        self._advect(direction='x',n=2)
        self._calc_diagnostics()
        self._diffuse(n=4)
        #self._calc_diagnostics()

        self._advect(direction='x',n=2)
        self._calc_diagnostics()
        self._diffuse(n=4)
        #self._calc_diagnostics()

        # y-dir
        self._advect(direction='y',n=2)
        self._calc_diagnostics()
        self._diffuse(n=4)
        #self._calc_diagnostics()
        self._advect(direction='y',n=2)
        self._calc_diagnostics()
        self._diffuse(n=4)
        #self._calc_diagnostics()

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

    def _calc_diagnostics(self):
        # here is where we calculate diagnostics
        if (self.t>=self.dt) and (self.t>=self.tavestart) and (self.tc%self.cadence):
            self._increment_diagnostics()

    # diagnostic stuff follow
    def _initialize_diagnostics(self):
        # Initialization for diagnotics
        self.diagnostics = dict()

        self._setup_diagnostics()

        if self.diagnostics_list == 'all':
            pass # by default, all diagnostics are active
        elif self.diagnostics_list == 'none':
            self.set_active_diagnostics([])
        else:
            self.set_active_diagnostics(self.diagnostics_list)

    def _setup_diagnostics(self):
        """Diagnostics setup"""

        self.add_diagnostic('var',
            description='Tracer variance',
            function= (lambda self: self.spec_var(self.thh))
        )

        self.add_diagnostic('thbar',
            description='x-averaged tracer',
            function= (lambda self: self.thm)
        )

        self.add_diagnostic('grad2_th_bar',
            description='x-averaged gradient square of th',
            function= (lambda self: self.gradth2m)
        )

        self.add_diagnostic('vth2m',
            description='x-averaged triple advective term v th2',
            function= (lambda self: self.vth2m)
            )

        self.add_diagnostic('th2m',
            description='x-averaged  th2',
            function= (lambda self: self.th2m)
            )

        self.add_diagnostic('vthm',
            description='x-averaged, y-direction tracer flux',
            function= (lambda self: (self.v*self.tha).mean(axis=1))
        )

        self.add_diagnostic('fluxy',
            description='x-averaged, y-direction tracer flux',
            function= (lambda self: (self.v*self.th).mean(axis=1))
        )

        self.add_diagnostic('spec',
            description='spec of anomalies about x-averaged flow',
            function= (lambda self: np.abs(np.fft.rfft2(
                        self.th-self.th.mean(axis=1)[...,np.newaxis]))**2/self.M2)
        )


    def _set_active_diagnostics(self, diagnostics_list):
        for d in self.diagnostics:
            self.diagnostics[d]['active'] == (d in diagnostics_list)

    def add_diagnostic(self, diag_name, description=None, units=None, function=None):
        # create a new diagnostic dict and add it to the object array

        # make sure the function is callable
        assert hasattr(function, '__call__')

        # make sure the name is valid
        assert isinstance(diag_name, str)

        # by default, diagnostic is active
        self.diagnostics[diag_name] = {
           'description': description,
           'units': units,
           'active': True,
           'count': 0,
           'function': function, }

    def describe_diagnostics(self):
        """Print a human-readable summary of the available diagnostics."""
        diag_names = self.diagnostics.keys()
        diag_names.sort()
        print('NAME               | DESCRIPTION')
        print(80*'-')
        for k in diag_names:
            d = self.diagnostics[k]
            print('{:<10} | {:<54}').format(
                 *(k,  d['description']))

    def _increment_diagnostics(self):
        # compute intermediate quantities needed for some diagnostics

        self._calc_derived_fields()

        for dname in self.diagnostics:
            if self.diagnostics[dname]['active']:
                res = self.diagnostics[dname]['function'](self)
                if self.diagnostics[dname]['count']==0:
                    self.diagnostics[dname]['value'] = res
                else:
                    self.diagnostics[dname]['value'] += res
                self.diagnostics[dname]['count'] += 1

    def _calc_derived_fields(self):

        """ Calculate derived field necessary for diagnostics """

        self.thh = np.fft.rfft2(self.th)

        # x-averaged tracer field
        self.thm = self.th.mean(axis=1)
        #self.thmh = np.fft.rfft(self.thm)
        #self.thm_y = np.fft.irfft(1j*self.kk*self.thmh)

        # anomaly about the x-averaged field
        self.tha = self.th-self.thm[...,np.newaxis]
        self.thah = np.fft.rfft2(self.tha)

        # x-averaged gradient squared
        gradx = np.fft.irfft2(1j*self.k*self.thah)
        grady = np.fft.irfft2(1j*self.l*self.thah)
        self.gradth2m = (gradx**2 + grady**2).mean(axis=1)

        # Osborn-Cox amplification factor
        #self.thm_y = 4*np.sin(self.y*self.dl)
        #thm_y = self.block_average(self.thm_y)
        #gradth2m = self.block_average(self.gradth2m)

        #self.A2_OC = gradth2m / thm_y**2
        #self.A2_OC[thm_y < 1.e-14] = np.nan

        # triple term
        self.vth2m = (self.v*(self.tha**2)).mean(axis=1)

        # diff transport
        self.th2m = (self.tha**2).mean(axis=1)

    def get_diagnostic(self, dname):
        return (self.diagnostics[dname]['value'] /
                self.diagnostics[dname]['count'])

    def spec_var(self, ph):
        """ compute variance of p from Fourier coefficients ph """
        var_dens = 2. * np.abs(ph)**2 / self.M**2
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
        var_dens[...,0] = var_dens[...,0]/2.
        var_dens[...,-1] = var_dens[...,-1]/2.
        return var_dens.sum()

    def block_average(self,A, nblocks = 256):
        """ Block average A onto A blocks """

        nave = self.nx/nblocks
        Ab = np.empty(nblocks)

        for i in range(nblocks):
            Ab[i] = A[i*nave:(i+1)*nave].mean()

        return Ab

#grad2 = (wv2*(np.abs(thh)**2)).sum()/(N**2)
# a test initial concentration
#x0,y0 = pi,pi
#r = np.sqrt((x-x0)[np.newaxis,...]**2+(y-y0)[...,np.newaxis]**2)
#th = np.zeros(N,N)
#th = np.exp(-(r**2))
