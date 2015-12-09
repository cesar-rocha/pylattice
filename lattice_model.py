
import numpy as np
from numpy import pi, cos, sin, exp

class LatticeModel():
    """ A class that represents a two-dimensional lattice
        model of advection-diffusion """

    def __init__(self,
                nx=128,
                ny=None,
                Lx=2*pi,
                Ly=None,
                dt=0.5,
                tmax=1000,
                tavestart = 500,
                kappa=1.e-5,
                diagnostics_list='all'):

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
        
        self.kappa = kappa
        
        self.diagnostics_list = diagnostics_list

        self._initialize_grid()

        self._initialize_diagnostics()

    def _initialize_grid(self):
        """ Initialize lattice and spectral space grid """

        # physical space grids
        self.dx, self.dy = self.Lx/self.nx, self.Ly/self.ny

        self.x = np.linspace(0.,self.Lx-self.dx,self.nx)
        self.y = np.linspace(0.,self.Ly-self.dy,self.ny)

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
        
        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt( self.wv2 )

    def _velocity(self,nmodes=100,power=3.5,nmin=2):

        phase = 2*pi*np.random.rand(2,nmodes)
        phi, psi = phase[0], phase[1]
     
        nmax = nmin+nmodes
        n = np.arange(nmin,nmax)[np.newaxis,...]    
        An = n**(-power/2.)

        Yn = n*self.y[...,np.newaxis] + phase[0][np.newaxis,...]
        Xn = n*self.x[...,np.newaxis] + phase[1][np.newaxis,...]

        u = (An*cos(Yn*self.dl)).sum(axis=1)
        v = (An*cos(Xn*self.dk)).sum(axis=1)

        self.u = u[...,np.newaxis]
        self.v = v[np.newaxis,...]

    def _advect(self,direction='x'):
        """ Advect th on a lattice given u and v,
            and the current index array ix, iy"""

        if direction == 'x':
            ix_new = self.ix.copy()
            dindx = -np.round(self.u*self.dt_2/self.dx).astype(int)
            ix_new  = self.ix + dindx 
            ix_new[ix_new<0] = ix_new[ix_new<0] + self.nx
            ix_new[ix_new>self.nx-1] = ix_new[ix_new>self.nx-1] - self.nx
            self.th = self.th[self.iy,ix_new]

        elif direction == 'y':
            iy_new = self.iy.copy()
            dindy = -np.round(self.v*self.dt_2/self.dy).astype(int)
            iy_new  = self.iy + dindy
            iy_new[iy_new<0] = iy_new[iy_new<0] + self.ny
            iy_new[iy_new>self.ny-1] = iy_new[iy_new>self.ny-1] - self.ny
            self.th = self.th[iy_new,self.ix]

    def _diffuse(self, half=True):
        """ Diffusion """
        self.thh = np.fft.rfft2(self.th)
        if half:
            self.thh = self.thh*exp(-self.dt_2*self.kappa*self.wv2)
        else:
            self.thh = self.thh*exp(-self.dt*self.kappa*self.wv2)

        self.th = np.fft.irfft2(self.thh)

    def _forcing(self,half=True):
        if half:
            self.th = self.th + self.dt_2*np.cos(self.y)[...,np.newaxis]
        else:
            self.th = self.th + self.dt*np.cos(self.y*self.dl)[...,np.newaxis]

    def _step_forward(self):

        self._velocity()

        self._advect(direction='x')

        self._advect(direction='y')

        self._diffuse(half='False')

        self._forcing()

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
            function= (lambda self: self.th.mean(axis=1))
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
        
        #self._calc_derived_fields()
        
        for dname in self.diagnostics:
            if self.diagnostics[dname]['active']:
                res = self.diagnostics[dname]['function'](self)
                if self.diagnostics[dname]['count']==0:
                    self.diagnostics[dname]['value'] = res
                else:
                    self.diagnostics[dname]['value'] += res
                self.diagnostics[dname]['count'] += 1
                
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



#grad2 = (wv2*(np.abs(thh)**2)).sum()/(N**2)
# a test initial concentration
#x0,y0 = pi,pi
#r = np.sqrt((x-x0)[np.newaxis,...]**2+(y-y0)[...,np.newaxis]**2)
#th = np.zeros(N,N)
#th = np.exp(-(r**2))



