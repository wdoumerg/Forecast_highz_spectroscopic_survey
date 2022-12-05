import numpy as np
import scipy as sp
#import pyccl as ccl
import math
import matplotlib.pyplot as plt
from scipy import integrate
from functools import partial
from scipy.integrate import quad, dblquad
import pyccl as ccl

Mnu=0.06
yourz=0
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.677, A_s=2.1e-9, n_s=0.968,transfer_function='boltzmann_camb')


a=1./(1+yourz)
Omega_m=cosmo['Omega_m']
Omega_b=cosmo['Omega_b']
Omega_c=cosmo['Omega_c']
Omega_l=1-Omega_m
#print(Omega_m)

sky=14000

def Vsurvey(z,dz,sky):    
    Omega     = sky*(math.pi/180)**2 # get rid of unit
    d2        = ccl.comoving_radial_distance(cosmo,1/(1+z))
    d3        = ccl.comoving_radial_distance(cosmo,1/(1+z+dz))
    return Omega/3 * (d3**3 - d2**3)*h**3
   
    

    
deltac=1.686
H0=67.7
h=H0/100
c=2.99*10**5
#Set up integration options
nlim=10000


def D(z):
    return ccl.growth_factor(cosmo,1/(1+z))*0.708

def f(z):
    return ccl.growth_rate(cosmo,1/(1+z))

def Tbbks(k):
    #bbks approx
    
    k=k*h
    q=k/(Omega_m*h**2*math.exp(-Omega_b*(1+(2*h)**0.5/Omega_m)))
    return math.log(1+2.34*q)/2.34/q*(1+3.89*q+(16.2*q)**2+(5.47*q)**3+(6.71*q)**4)**(-0.25)#ccl.get_transfer(math.log(k),a)
    

def Pm(k,z1):
    return ccl.linear_matter_power(cosmo, k*h, 1/(1+z1))*h**3

def Pz0(k):
     return ccl.linear_matter_power(cosmo, k*h, 1/(1+0))*h**3
 
def sigma_8(z1):
     return (Pm(0.01,z1)/Pz0(0.01))**0.5*ccl.sigma8(cosmo)

def bias(z1,gal):
    if gal=='LRG':
        return 1.7*D(0)/D(z1)#LRG
    if gal=='ELG':
        return D(0)/D(z1)#ELG
    if gal=='ELG_DESI':
        return 0.84*D(0)/D(z1)#ELG
    if gal=='LBG24':
        return (-0.98*(24-25)+0.11)*(1+z1)+(0.12*(24-25)+0.17)*(1+z1)**2#LBG
    if gal=='LBG24.2':
        return (-0.98*(24.2-25)+0.11)*(1+z1)+(0.12*(24.2-25)+0.17)*(1+z1)**2#LBG
    if gal=='LBG24.5':
        return (-0.98*(24.5-25)+0.11)*(1+z1)+(0.12*(24.5-25)+0.17)*(1+z1)**2#LBG
    if gal=='LBG25':
        return 0.11*(1+z1)+0.17*(1+z1)**2#LBG
    if gal=='LBG25.5':
        return (-0.98*(25.5-25)+0.11)*(1+z1)+(0.12*(25.5-25)+0.17)*(1+z1)**2#LBG
    if gal=='QSO':
        return 0.53+0.29*(1+z1)**2
    print('galaxy type unknown')
    
    
def sigma_chi(z):
    sigma_z=0.01
    return c*(1+z)/(ccl.h_over_h0(cosmo,1/(1+z))*h*100)*sigma_z/(1+z)



def fnl_2tracer(z2,zeff,dz,na,nb,sky,gal1,gal2,kmax):

    z=zeff#z2#+dz/2
    sigma8=sigma_8(z)
    sigma8_0=sigma_8(0)
    f_z=f(z)
    ba=bias(z,gal1)
    bb=bias(z,gal2)
    #print(sigma8)
    #print(f_z)
    #print(bg)
    #print(bias(1))
    V=Vsurvey(z2,dz,sky)
    #na=Na/V*10**6
    #nb=Nb/V*10**6
    #print(na)
    #V= 137475/1.137*10**4*h**3
    p=1
    kmin=2*math.pi/V**(1/3)
    #kmax=0.1*D(0)/D(zeff)#/h
    #print('main ', (bg*sigma8+f_z*sigma8*1**2)**2)
    
    def PA(k,mu):
        return (ba+f_z*mu**2)**2*Pm(k,z)*math.exp(-k**2*mu**2*sigma_chi(z)**2)
    def PB(k,mu):
        return (bb+f_z*mu**2)**2*Pm(k,z)*math.exp(-k**2*mu**2*sigma_chi(z)**2)
    def PAB(k,mu):
        return (ba+f_z*mu**2)*(bb+f_z*mu**2)*Pm(k,z)*math.exp(-k**2*mu**2*sigma_chi(z)**2)
    def ddb_df(k,bg):
        return 3*(bg-p)*deltac*Omega_m/(k**2*Tbbks(k)*D(z))*(H0/c)**2
    
    
    def Dt(X,i,mu,k):
        if X=='A':
            if i==1:
                return 2/(ba+f_z*mu**2)
            if i==2:
                return 0
            if i==3:
                return 2*ddb_df(k,ba)/(ba+f_z*mu**2)
        if X=='B':
            if i==1:
                return 0
            if i==2:
                return 2/(bb+f_z*mu**2)
            if i==3:
                return 2*ddb_df(k,bb)/(bb+f_z*mu**2)
        if X=='AB':
            if i==1:
                return 1/(ba+f_z*mu**2)
            if i==2:
                return 1/(bb+f_z*mu**2)
            if i==3:
                return ddb_df(k,bb)/(bb+f_z*mu**2)+ddb_df(k,ba)/(ba+f_z*mu**2)
        print('error DX')
    
    def Raa(k,mu):
        return (na*PA(k,mu)*(1+nb*PB(k,mu))/((1+na*PA(k,mu))*(1+nb*PB(k,mu))-na*nb*PAB(k,mu)**2))**2
    def Rbb(k,mu):
        return (nb*PB(k,mu)*(1+na*PA(k,mu))/((1+nb*PB(k,mu))*(1+na*PA(k,mu))-na*nb*PAB(k,mu)**2))**2
    def Rxx(k,mu):
        return na*nb*((1+na*PA(k,mu))*(1+nb*PB(k,mu))+na*nb*PAB(k,mu)**2)/((1+na*PA(k,mu))*(1+nb*PB(k,mu))-na*nb*PAB(k,mu)**2)**2*PAB(k,mu)**2
    
    def Rxa(k,mu):
        return na**2*nb*(1+nb*PB(k,mu))/((1+na*PA(k,mu))*(1+nb*PB(k,mu))-na*nb*PAB(k,mu)**2)**2*PAB(k,mu)**2*PA(k,mu)
    def Rxb(k,mu):
        return nb**2*na*(1+na*PA(k,mu))/((1+na*PA(k,mu))*(1+nb*PB(k,mu))-na*nb*PAB(k,mu)**2)**2*PAB(k,mu)**2*PB(k,mu)
    def Rab(k,mu):
        return na**2*nb**2*PA(k,mu)*PB(k,mu)*PAB(k,mu)**2/((1+na*PA(k,mu))*(1+nb*PB(k,mu))-na*nb*PAB(k,mu)**2)**2
    
    
    def FX(X,k,mu,i,j):
        if X=='A':
            return 1/2*Dt(X,i,mu,k)*Dt(X,j,mu,k)*Raa(k,mu)
        if X=='B':
            return 1/2*Dt(X,i,mu,k)*Dt(X,j,mu,k)*Rbb(k,mu)
        if X=='AB':
            return Dt(X,i,mu,k)*Dt(X,j,mu,k)*Rxx(k,mu)-(Dt(X,i,mu,k)*Dt('A',j,mu,k)+Dt('A',i,mu,k)*Dt(X,j,mu,k))*Rxa(k,mu)-(Dt(X,i,mu,k)*Dt('B',j,mu,k)+Dt('B',i,mu,k)*Dt(X,j,mu,k))*Rxb(k,mu)+1/2*(Dt('A',i,mu,k)*Dt('B',j,mu,k)+Dt('B',i,mu,k)*Dt('A',j,mu,k))*Rab(k,mu)
        print('error Fx')
    def integrand(k,mu,i,j):
        return (FX('A',k,mu,i,j)+FX('B',k,mu,i,j)+FX('AB',k,mu,i,j))*k**2
    
        
    def integrat_rsd(muint,int_i,int_j):
        return integrate.quad(partial(integrand,mu=muint,i=int_i,j=int_j),kmin,kmax,epsrel=0.0000001,epsabs=0,limit=nlim)[0]
    
    def F_rsd():
        #print('ok')
        f11=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=1,int_j=1),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f12=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=1,int_j=2),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f13=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=1,int_j=3),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f22=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=2,int_j=2),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f23=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=2,int_j=3),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f33=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=3,int_j=3),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        return np.array([[f11,f12,f13],[f12,f22,f23],[f13,f23,f33]])
    Ff=F_rsd()
    return Ff



#%%MUST
print('MUST 2 tracers')
kmax=0.3
print(np.linalg.inv(fnl_2tracer(2,2.2,0.8,0.0007,0.00014,15000,'LBG24.2','LBG25.5',kmax)+fnl_2tracer(2.8,2.9,0.8,0.00021,0.000042,15000,'LBG24.2','LBG25.5',kmax)+fnl_2tracer(3.6,3.7,0.6,0.000021,0.0000042,15000,'LBG24.2','LBG25.5',kmax))[2][2]**0.5)
print(np.linalg.inv(fnl_2tracer(2,2.2,0.8,0.00035,0.00007,15000,'LBG24.2','LBG25.5',kmax)+fnl_2tracer(2.8,2.9,0.8,0.00011,0.000021,15000,'LBG24.2','LBG25.5',kmax)+fnl_2tracer(3.6,3.7,0.6,0.000011,0.0000021,15000,'LBG24.2','LBG25.5',kmax))[2][2]**0.5)
print(np.linalg.inv(fnl_2tracer(2,2.2,0.8,0.000187,0.000043,15000,'LBG24.2','LBG25.5',kmax)+fnl_2tracer(2.8,2.9,0.8,0.000056,0.000013,15000,'LBG24.2','LBG25.5',kmax)+fnl_2tracer(3.6,3.7,0.6,0.0000056,0.0000013,15000,'LBG24.2','LBG25.5',kmax))[2][2]**0.5)
print(np.linalg.inv(fnl_2tracer(2,2.2,0.8,0.00031,0.00007,9000,'LBG24.2','LBG25.5',kmax)+fnl_2tracer(2.8,2.9,0.8,0.000092,0.000021,9000,'LBG24.2','LBG25.5',kmax)+fnl_2tracer(3.6,3.7,0.6,0.0000092,0.0000021,9000,'LBG24.2','LBG25.5',kmax))[2][2]**0.5)


#%%
print('NTL 2 tracers')
kmax=0.3

print(np.linalg.inv(fnl_2tracer(2,2.3,1,0.00065,0.00055,14000,'LBG24.2','LBG24.5',kmax)+fnl_2tracer(3,3.3,1,0.00011,0.00016,14000,'LBG24.2','LBG24.5',kmax)+fnl_2tracer(4,4.3,1,0.000024,0.00005,14000,'LBG24.2','LBG24.5',kmax))[2][2]**0.5)
print(np.linalg.inv(fnl_2tracer(2,2.3,1,0.0012,0.0017,14000,'LBG24.5','LBG25',kmax)+fnl_2tracer(3,3.3,1,0.00027,0.00061,14000,'LBG24.5','LBG25',kmax)+fnl_2tracer(4,4.3,1,0.000075,0.00025,14000,'LBG24.5','LBG25',kmax))[2][2]**0.5)


