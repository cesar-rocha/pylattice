import numpy as np
import lattice_model
from numpy import pi

class SourceModel(lattice_model.LatticeModel):
    """ A subclass that represents the advection-diffusion
            model with a large-scale source """

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

    def _source(self,direction='x',n=1):
        if direction == 'x':
            self.th += (self.dt/n)*np.cos(self.dl*self.y)[...,np.newaxis]
        elif direction == 'y':
            # a brutal way
            self.th += (self.dt/n)*np.cos(self.dl*self.y)[...,np.newaxis]
            #pass

class GyModel(lattice_model.LatticeModel):
    """ A subclass that represents the advection-diffusion
            model with a basic state sustained by a linear
            mean constant mean gradient """

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
    
    def _source(self,direction='x',n=1):
        pass

