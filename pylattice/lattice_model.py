from __future__ import division
import numpy as np
from numpy import pi, cos, sin, exp
import logging

try:
    import pyfftw
    pyfftw.interfaces.cache.enable() 
except ImportError:
    pass


class LatticeModel(object):
    """ A generic two-dimensional lattice
        model of advection-diffusion 
        
        Attributes
        ----------

        nx:    number of grid points in the x-direction (number of lattices)
        ny:    number of grid points in the y-direction; if None, then ny = nx
        Lx:    length of the domain in the x-direction
        Ly:    length of the domain in the x-direction; if None, then Ly = Lx
        dt:    eddy turnover time scale (the length of a renovating cycle) [(time)]
        tmax:  maximum time of integration [(time)]
        kappa: molecular diffusivity [(length)^2/(time)]
        urms:  root-mean-square of the velocity field (sets the energy level) [(length)^2/(time)^2]
        nmin:  minimum wavenumber for the eddy field [(unitless)]
        nmax:  maximum wavenumber for the eddy field [(unitless)]
        diagnostics_list: list of diagnostics to compute
        tavestart: time to start averaging diagnostics [(time)]
        cadence:  cadence to compute diagnostics [(time step)]

        """

    def __init__(self,
                nx=128,
                ny=None,
                Lx=2*pi,
                Ly=None,
                dt=0.5,
                tmax=1000,
                kappa=1.e-5,
                urms = 1.,
                power = 3.5,
                nmin = 5,
                nmax = 120,
                diagnostics_list='all',
                tavestart = 500,
                diagcadence = 1,
                fftw=True,
                ntd=4,
                logfile=None,
                loglevel=1,
                printcadence = 10):

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
        self.diagcadence = diagcadence
        self.printcadence = printcadence

        self.kappa = kappa
        
        self.nmin = nmin
        self.nmax = nmax
        self.power = power
        self.urms = urms

        self.fftw = fftw
        self.ntd = ntd

        self.logfile = logfile
        self.loglevel=loglevel

        # initializations
        self.diagnostics_list = diagnostics_list

        self._initialize_logger()
        self._initialize_fft()
        self._initialize_grid()
        self._init_velocity()
        self._allocate_variables()
        self._initialize_diagnostics()

        # some initial diagnostics
        self.Pe = self.urms/(self.kappa*self.kmin)
        self.dt = 1./(self.urms*self.kmin)
        self.dt_2 = self.dt/2. 

        self.logger.info('dx/lb = %3.2e', self.dx/self.lb)
        self.logger.info('tau = %3.2e', self.dt)
        self.logger.info('Pe = %3.2e', self.Pe)

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

    #
    # private methods
    #

    def _step_forward(self):

        # status
        self._print_status()

        # renovate velocity field
        self._velocity()

        # use a while loop here...

        # x-dir
        self._advect(direction='x',n=2)
        self._source(direction='x',n=4)
        self._calc_diagnostics()
        self._diffuse(n=4)
        self._calc_diagnostics()

        self._advect(direction='x',n=2)
        self._source(direction='x',n=4)
        self._calc_diagnostics()
        self._diffuse(n=4)
        self._calc_diagnostics()

        # y-dir
        self._advect(direction='y',n=2)
        self._source(direction='y',n=4)
        self._calc_diagnostics()
        self._diffuse(n=4)
        self._calc_diagnostics()
        self._advect(direction='y',n=2)
        self._source(direction='y',n=4)
        self._calc_diagnostics()
        self._diffuse(n=4)
        self._calc_diagnostics()

        self.tc += 1
        self.t += self.dt

    def _allocate_variables(self):
        """ Allocate variables in memory """

        dtype_real = np.dtype('float64')            
        dtype_cplx = np.dtype('complex128')
        shape_real = (self.ny, self.nx)
        shape_cplx = (self.ny, self.nx/2+1)
            
        # tracer concentration
        self.th  = np.zeros(shape_real, dtype_real)
        self.thh = np.zeros(shape_cplx, dtype_cplx)

    # logger
    def _initialize_logger(self):

        self.logger = logging.getLogger(__name__)


        if self.logfile:
            fhandler = logging.FileHandler(filename=self.logfile, mode='w')
        else:
            fhandler = logging.StreamHandler()

        formatter = logging.Formatter('%(levelname)s: %(message)s')

        fhandler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fhandler)

        self.logger.setLevel(self.loglevel*10)
        
        # this prevents the logger to propagate into the ipython notebook log
        self.logger.propagate = False   

        self.logger.info(' Logger initialized')

    def _initialize_fft(self):
        # set up fft functions for use later
        if self.fftw: 
            self.fft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.rfft2(x, threads=self.ntd,\
                            planner_effort='FFTW_ESTIMATE'))
            self.ifft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.irfft2(x, threads=self.ntd,\
                            planner_effort='FFTW_ESTIMATE'))
        else:
            self.fft2 =  (lambda x : np.fft.rfft2(x))
            self.ifft2 = (lambda x : np.fft.irfft2(x))

    def _initialize_grid(self):
        """ Initialize lattice and spectral space grid """

        # physical space grids (the lattice)
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

        # kmin and kmax
        self.kmin = self.dk*self.nmin
        self.kmax = self.dk*self.nmax

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

        # estimate the Batchelor scale
        S = np.sqrt( ((self.An*self.n*self.dk)**2).sum()/2. )
        self.lb = np.sqrt(self.kappa/S)

        #assert self.lb > self.dx, "**Warning: Batchelor scale not resolved."

    def _diffuse(self, n=1):
        """ Diffusion """

        self.thh = self.fft2(self.th)
        self.thh = self.thh*exp(-(self.dt/n)*self.kappa*self.wv2)
        self.th = self.ifft2(self.thh)

    def _advect(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass') 

    def _source(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass') 

    def _print_status(self):
        """Output some basic stats."""
        if (self.loglevel) and ((self.tc % self.printcadence)==0):
            self._calc_var()
            self.logger.info('Step: %4i, Time: %3.2e, Variance: %3.2e'
                    , self.tc,self.t,self.var)
            
    ## diagnostic methods 

    def _calc_var(self):
        self.var = self.spec_var(self.thh)

    def _calc_diagnostics(self):
        if (self.t>=self.dt) and (self.t>=self.tavestart):
            self._increment_diagnostics()
         
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
            function= (lambda self: np.abs(self.fft2(
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

        # x-averaged tracer field
        self.thm = self.th.mean(axis=1)

        # anomaly about the x-averaged field
        self.tha = self.th-self.thm[...,np.newaxis]
        self.thah = self.fft2(self.tha)

        # x-averaged gradient squared
        gradx = self.ifft2(1j*self.k*self.thah)
        grady = self.ifft2(1j*self.l*self.thah)
        self.gradth2m = (gradx**2 + grady**2).mean(axis=1)

        # triple term
        self.vth2m = (self.v*(self.tha**2)).mean(axis=1)

        # diff transport
        self.th2m = (self.tha**2).mean(axis=1)

    def get_diagnostic(self, dname):
        return (self.diagnostics[dname]['value'] / 
                self.diagnostics[dname]['count'])

    ## utility methods

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


