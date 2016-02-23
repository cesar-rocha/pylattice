import numpy as np
import lattice_model
from numpy import pi

class SourceModel(lattice_model.LatticeModel):
    """ A subclass that represents the advection-diffusion
            model with a large-scale source

        Attributes
        ----------
        source: flag for source (boolean)

    """

    def __init__(self, source=True, **kwargs):

        self.source = source
        self.Gy = False

        super(SourceModel, self).__init__(**kwargs)

    def _advect(self,direction='x',n=1):
        """ Advect th on a lattice given u and v,
            and the current index array ix, iy

            Attributes
            ----------
            direction: direction to perform the advection ('x' or 'y')
            n: the number of substeps
               n=1 for doing the full advection-diffusion;
               n=2 for doing half the advection; etc """

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

    def _source(self,direction='x',n=1):
        if direction == 'x':
            self.th += (self.dt/n)*np.cos(self.dl*self.y)[...,np.newaxis]
        elif direction == 'y':
            # a brutal way
            self.th += (self.dt/n)*np.cos(self.dl*self.y)[...,np.newaxis]
            #pass

    ## diagnostic methods
    def _initialize_nakamura(self):
        self.Lmin2 = self.Lx**2
        # this 2 is arbitrary here...
        thm = np.cos(self.dl*self.y)/(self.D*(self.dl**2))
        thmin,thmax = thm.min(),thm.max()
        self.dth = 0.1
        self.dth2 = self.dth**2
        self.TH = np.arange(thmin+self.dth/2,thmax-self.dth/2,self.dth)
        self.Leq2 = np.empty(self.TH.size)
        self.I1 = np.empty(self.TH.size)
        self.I2 = np.empty(self.TH.size)
        self.L = np.empty(self.TH.size)

    def _calc_Leq2(self):

        th = self.th

        #th = np.vstack([(th[self.nx-self.nx/self.npad:]),th,\
        #                th[:self.nx/self.npad]])
        #gradth2 =  np.vstack([self.gradth2[self.nx-self.nx/self.npad:],\
        #                      self.gradth2,self.gradth2[:self.nx/self.npad]])
        gradth2 = self.gradth2

        gradth = np.sqrt(gradth2)

        # parallelize this...
        for i in range(self.TH.size):

            self.fth2 = th<=self.TH[i]+self.dth/2
            self.fth1 = th<=self.TH[i]-self.dth/2

            A2 = self.dS*self.fth2.sum()
            A1 = self.dS*self.fth1.sum()
            self.dA = A2-A1

            self.G2 = (gradth2[self.fth2]*self.dS).sum()-\
                      (gradth2[self.fth1]*self.dS).sum()

            self.Leq2[i] = self.G2*self.dA/self.dth2

            self.L[i] = ((gradth[self.fth2]*self.dS).sum()-\
                        (gradth[self.fth1]*self.dS).sum())/self.dth

            self.I1[i] = self.G2/self.dth
            self.I2[i] = self.dA/self.dth


class GyModel(lattice_model.LatticeModel):
    """ A subclass that represents the advection-diffusion
            model with a basic state sustained by a linear
            mean constant mean gradient """

    def __init__(self, G=1., **kwargs):

        self.G = G
        self.Gy = True

        super(GyModel, self).__init__(**kwargs)


    def _advect(self,direction='x',n=1):
        """ Advect th on a lattice given u and v,
            and the current index array ix, iy

            Attributes
            ----------
            direction: direction to perform the advection ('x' or 'y')
            n: the number of substeps
               n=1 for doing the full advection-diffusion;
               n=2 for doing half the advection; etc """

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

    def _source(self,direction='x',n=1):
        pass

    ## diagnostic methods
    def _initialize_nakamura(self):
        self.Lmin2 = self.Lx**2
        thm = self.G*self.y
        thmin,thmax = thm.min(),thm.max()
        self.dth = 0.1
        self.dth2 = self.dth**2
        self.TH = np.arange(thmin+self.dth/2,thmax-self.dth/2,self.dth)
        self.Leq2 = np.empty(self.TH.size)
        self.I1 = np.empty(self.TH.size)
        self.I2 = np.empty(self.TH.size)
        self.L = np.empty(self.TH.size)


    def _calc_Leq2(self):

        th = self.th + self.G*self.y[...,np.newaxis]

        th = np.vstack([(th[self.nx-self.nx/self.npad:]-2*pi),th,\
                        th[:self.nx/self.npad]+2*pi])
        gradth2 =  np.vstack([self.gradth2[self.nx-self.nx/self.npad:],\
                              self.gradth2,self.gradth2[:self.nx/self.npad]])

        gradth = np.sqrt(gradth2)

        # parallelize this...
        for i in range(self.TH.size):

            self.fth2 = th<=self.TH[i]+self.dth/2
            self.fth1 = th<=self.TH[i]-self.dth/2

            A2 = self.dS*self.fth2.sum()
            A1 = self.dS*self.fth1.sum()
            self.dA = A2-A1

            self.G2 = (gradth2[self.fth2]*self.dS).sum()-\
                      (gradth2[self.fth1]*self.dS).sum()

            self.Leq2[i] = self.G2*self.dA/self.dth2

            self.I1[i] = self.G2/self.dth
            self.I2[i] = self.dA/self.dth

            self.L[i] = ((gradth[self.fth2]*self.dS).sum()-\
                        (gradth[self.fth1]*self.dS).sum())/self.dth
