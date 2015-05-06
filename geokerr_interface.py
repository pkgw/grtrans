from geokerr import class_geokerr
import numpy as np

class geokerr_inputs:
    def init(self):
        self.standard=2
        self.mu0=0.142
        self.a=0.0
        self.rcut=1e2
        self.nrotype=2
        self.avals=[-9.,9.,-9.,9.]
        self.n=[75,75,1]
        self.uout=0.01
        self.kext=3
        self.next=60
        self.usegeor=0
        self.mufill=1
        self.phit=1
        
    def __init__(self,**kwargs):
        self.init()
        self.__dict__.update(kwargs)
        if self.standard==2:
            self.usegeor=1
            self.mufill=0
        else:
            self.usegeor=0

class geokerr:
    def initialize_geokerr_camera(self,**kwargs):
        self.inputs=geokerr_inputs(**kwargs)
# initialize camera, save outputs, and delete class_geokerr data
        class_geokerr.init_pix_data(self.inputs.n[0]*self.inputs.n[1])
        class_geokerr.pixel(self.inputs.standard,self.inputs.avals[0],self.inputs.avals[1],self.inputs.avals[2],self.inputs.avals[3],self.inputs.rcut,self.inputs.nrotype,self.inputs.n[0],self.inputs.n[1],self.inputs.n[2],self.inputs.uout,self.inputs.mu0,self.inputs.a)
        self.uf = class_geokerr.ufarr.copy()
        self.muf = class_geokerr.mufarr.copy()
        self.tpm = class_geokerr.tpmarr.copy()
        self.tpr = class_geokerr.tprarr.copy()
        self.alpha = class_geokerr.aarr.copy()
        self.beta = class_geokerr.barr.copy()
        self.l = class_geokerr.larr.copy()
        self.q2 = class_geokerr.q2arr.copy()
        self.sm = class_geokerr.smarr.copy()
        self.su = class_geokerr.suarr.copy()
        self.u0 = class_geokerr.u0.copy()
        self.offset = class_geokerr.offset.copy()
        class_geokerr.del_pix_data()

    def run_single_geodesic(self,i):
        if (i < 0) or (i >= self.inputs.n[0]*self.inputs.n[1]):
            print 'error: invalid value of geodesic index for given camera ',i,self.inputs.n[0]*self.inputs.n[1]
# first initialize geodesic at full length and call geokerr
        self.npts = self.inputs.n[2]
        class_geokerr.init_geokerr_data(self.inputs.n[2]-2*self.inputs.kext+self.inputs.next)
        class_geokerr.call_geokerr(self.u0,self.uf[i],self.inputs.uout,self.inputs.mu0,self.muf[i],self.inputs.a,self.l[i],self.q2[i],self.alpha[i],self.beta[i],self.tpm[i],self.tpr[i],self.su[i],self.sm[i],self.npts,self.offset,self.inputs.phit,self.inputs.usegeor,self.inputs.mufill,self.inputs.kext,self.inputs.next)
# now copy useful parts of geodesic
# first iteration with just whole thing up to last point used
        self.i1=0; self.i2=self.npts
        self.u = class_geokerr.u[self.i1:self.i2].copy()
        self.mu = class_geokerr.mu[self.i1:self.i2].copy()
        self.dphi = class_geokerr.dphi[self.i1:self.i2].copy()
        self.lam = class_geokerr.lam[self.i1:self.i2].copy()
        self.tpmi = class_geokerr.tpmi[self.i1:self.i2].copy()
        self.tpri = class_geokerr.tpri[self.i1:self.i2].copy()
        self.dt = class_geokerr.dt[self.i1:self.i2].copy()
# finally delete geokerr data
        class_geokerr.del_geokerr_data()

    def run_camera(self,**kwargs):
        self.initialize_geokerr_camera(**kwargs)
# initialize large arrays with all geodesic data
        print self.inputs.mufill
        if self.inputs.mufill==0:
            n = self.inputs.n[2]
        else:
            n = self.inputs.n[2]-self.inputs.kext*2+self.inputs.next
        print self.inputs.n[2], n
        
        uarr = np.zeros((self.inputs.n[0]*self.inputs.n[1],n))
        muarr = np.zeros((self.inputs.n[0]*self.inputs.n[1],n))
        tarr = np.zeros((self.inputs.n[0]*self.inputs.n[1],n))
        lamarr = np.zeros((self.inputs.n[0]*self.inputs.n[1],n))
        phiarr = np.zeros((self.inputs.n[0]*self.inputs.n[1],n))
        tpmiarr = np.zeros((self.inputs.n[0]*self.inputs.n[1],n))
        tpriarr = np.zeros((self.inputs.n[0]*self.inputs.n[1],n))
        for i in range(self.inputs.n[0]*self.inputs.n[1]):
            self.run_single_geodesic(i)
            uarr[i,self.i1:self.i2] = self.u
            muarr[i,self.i1:self.i2] = self.mu
            phiarr[i,self.i1:self.i2] = self.dphi
            tarr[i,self.i1:self.i2] = self.dt
            lamarr[i,self.i1:self.i2] = self.lam
            tpmiarr[i,self.i1:self.i2] = self.tpmi
            tpriarr[i,self.i1:self.i2] = self.tpri

        return uarr,muarr,tarr,lamarr,phiarr,tpmiarr,tpriarr
