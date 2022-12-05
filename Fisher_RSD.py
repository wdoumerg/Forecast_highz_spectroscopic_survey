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
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.677, A_s=2.1e-9, n_s=0.967,transfer_function='boltzmann_camb')


a=1./(1+yourz)
Omega_m=cosmo['Omega_m']
Omega_b=cosmo['Omega_b']
Omega_c=cosmo['Omega_c']
Omega_l=1-Omega_m
deltac=1.686
H0=67.7
h=H0/100
c=2.99*10**5
#Set up integration options
nlim=10000


def Vsurvey(z,dz,sky):    
    Omega     = sky*(math.pi/180)**2 # get rid of unit
    d2        = ccl.comoving_radial_distance(cosmo,1/(1+z))
    d3        = ccl.comoving_radial_distance(cosmo,1/(1+z+dz))
    return Omega/3 * (d3**3 - d2**3)*h**3
   
    


def D(z):
    return ccl.growth_factor(cosmo,1/(1+z))

def f(z):
    return ccl.growth_rate(cosmo,1/(1+z))


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
    


def RSD_1tracer(z2,zeff,dz,n,sky,gal):
    z=zeff
    sigma8=sigma_8(z)
    f_z=f(z)
    bg=bias(z,gal)
    V=Vsurvey(z2,dz,sky)
    kmin=2*math.pi/V**(1/3)
    kmax=0.1*D(0)/D(zeff)
    def P_rsd(k,mu):
        return (bg*sigma8+f_z*sigma8*mu**2)**2*Pz0(k)/sigma_8(0)**2
    
    def integrand_rsd(k,mu,int_n):
        pk=P_rsd(k,mu)*math.exp(-k**2*mu**2*sigma_chi(z)**2)
        if int_n==11:
            #print(n*pk)
            return 2*(n*pk/(n*pk+1))**2*(bg*sigma8/(bg*sigma8+f_z*sigma8*mu**2))**2*k**2
        if int_n==12:
            return 2*(n*pk/(n*pk+1))**2*f_z*sigma8*mu**2*bg*sigma8/(bg*sigma8+f_z*sigma8*mu**2)**2*k**2
        if int_n==22:
            return 2*(n*pk/(n*pk+1))**2*(f_z*sigma8*mu**2/(bg*sigma8+f_z*sigma8*mu**2))**2*k**2
        
    def integrat_rsd(muint,int_n1):
        if int_n1 ==11:
            return integrate.quad(partial(integrand_rsd,mu=muint,int_n=int_n1),kmin,kmax,epsrel=0.0001,epsabs=0,limit=nlim)[0]
        if int_n1 ==12:
            return integrate.quad(partial(integrand_rsd,mu=muint,int_n=int_n1),kmin,kmax,epsrel=0.0001,epsabs=0,limit=nlim)[0]
        if int_n1 ==22:
            return integrate.quad(partial(integrand_rsd,mu=muint,int_n=int_n1),kmin,kmax,epsrel=0.0001,epsabs=0,limit=nlim)[0]
    
    def F_rsd():
        #print('ok')
        f11=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_n1=11),-1, 1,epsrel=0.00001,epsabs=0.0001,limit=nlim)[0]
        f12=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_n1=12),-1, 1,epsrel=0.00001,epsabs=0.0001,limit=nlim)[0]
        f22=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_n1=22),-1, 1,epsrel=0.00001,epsabs=0.0001,limit=nlim)[0]
        return np.array([[f11,f12],[f12,f22]])
    [[a,b],[c1,d]]=np.linalg.inv(F_rsd())
    print('sig(sigma8b)/sigma8b=',a**0.5)
    print('sig(sigma8f)/sigma8f=',d**0.5)
    return('fin')

print('constrains Megamapper')
RSD_1tracer(2,2.1,0.5,0.00079,14000,'LBG24.5')#lbg
RSD_1tracer(2.5,2.6,0.5,0.00036,14000,'LBG24.5')#lbg
RSD_1tracer(3,3.1,1,0.00011,14000,'LBG24.5')#lbg
RSD_1tracer(4,4.1,1,0.00007,14000,'LBG24.5')#lbg
RSD_1tracer(2,2.35,3,0.000251,14000,'LBG24.5')#lbg

print('constrains ntl')
RSD_1tracer(2,2.26,3,0.000246,14000,'LBG24.2')#lbg
RSD_1tracer(2,2.30,3,0.000493,14000,'LBG24.5')#lbg
RSD_1tracer(2,2.37,3,0.00132,14000,'LBG25')#lbg


print('constrains MSE ELG')
RSD_1tracer(1.6,2,0.8,0.00018,10000,'ELG')#lbg
print('constrains LBG MSE')
RSD_1tracer(2.4,2.55,0.4,0.00023,10000,'LBG24.2')
RSD_1tracer(2.8,2.9,0.4,0.00011,10000,'LBG24.2')
RSD_1tracer(3.2,3.3,0.8,0.000043,10000,'LBG24.2')
RSD_1tracer(2.4,2.6,1.6,0.00011,10000,'LBG24.2')




