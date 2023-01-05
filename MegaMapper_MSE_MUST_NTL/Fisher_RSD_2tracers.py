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
        f11=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_n1=11),-1, 1,epsrel=0.00001,epsabs=0.0001,limit=nlim)[0]
        f12=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_n1=12),-1, 1,epsrel=0.00001,epsabs=0.0001,limit=nlim)[0]
        f22=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_n1=22),-1, 1,epsrel=0.00001,epsabs=0.0001,limit=nlim)[0]
        return np.array([[f11,f12],[f12,f22]])
    [[a,b],[c1,d]]=np.linalg.inv(F_rsd())
    print('sig(sigma8b)/sigma8b=',a**0.5)#/sigma8/bg)
    print('sig(sigma8f)/sigma8f=',d**0.5)#/sigma8/f_z)
    return('fin')


#%%

def RSD_2tracer(z2,zeff,dz,Na,Nb,sky,gal1,gal2):

    z=zeff
    sigma8=sigma_8(z)
    sigma8_0=sigma_8(0)
    f_z=f(z)
    ba=bias(z,gal1)
    bb=bias(z,gal2)
    V=Vsurvey(z2,dz,sky)
    na=Na/V*10**6
    nb=Nb/V*10**6
    
    kmin=2*math.pi/V**(1/3)
    kmax=0.1*D(0)/D(zeff)
    
    def PA(k,mu):
        return (ba*sigma8+f_z*sigma8*mu**2)**2*Pm(k,0)/sigma8_0**2*math.exp(-k**2*mu**2*sigma_chi(z)**2)
    def PB(k,mu):
        return (bb*sigma8+f_z*sigma8*mu**2)**2*Pm(k,0)/sigma8_0**2*math.exp(-k**2*mu**2*sigma_chi(z)**2)
    def PAB(k,mu):
        return (ba*sigma8+f_z*sigma8*mu**2)*(bb*sigma8+f_z*sigma8*mu**2)*Pm(k,0)/sigma8_0**2*math.exp(-k**2*mu**2*sigma_chi(z)**2)
    
    def P(X,k,mu):
        if X=='A':
            return (ba*sigma8+f_z*sigma8*mu**2)**2*Pm(k,0)/sigma8_0**2
        if X=='B':
            return (bb*sigma8+f_z*sigma8*mu**2)**2*Pm(k,0)/sigma8_0**2
        if X=='AB':
            return (ba*sigma8+f_z*sigma8*mu**2)*(bb*sigma8+f_z*sigma8*mu**2)*Pm(k,0)/sigma8_0**2
        print('error PX')
    
    def Dt(X,i,mu):
        if X=='A':
            if i==1:
                return 2*ba*sigma8/(ba*sigma8+f_z*sigma8*mu**2)
            if i==2:
                return 0
            if i==3:
                return 2*f_z*sigma8*mu**2/(ba*sigma8+f_z*sigma8*mu**2)
        if X=='B':
            if i==1:
                return 0
            if i==2:
                return 2*bb*sigma8/(bb*sigma8+f_z*sigma8*mu**2)
            if i==3:
                return 2*f_z*sigma8*mu**2/(bb*sigma8+f_z*sigma8*mu**2)
        if X=='AB':
            if i==1:
                return ba*sigma8/(ba*sigma8+f_z*sigma8*mu**2)
            if i==2:
                return bb*sigma8/(bb*sigma8+f_z*sigma8*mu**2)
            if i==3:
                return f_z*sigma8*mu**2*((ba+bb)*sigma8+2*f_z*sigma8*mu**2)/((ba*sigma8+f_z*sigma8*mu**2)*(bb*sigma8+f_z*sigma8*mu**2))
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
            return 1/2*Dt(X,i,mu)*Dt(X,j,mu)*Raa(k,mu)
        if X=='B':
            return 1/2*Dt(X,i,mu)*Dt(X,j,mu)*Rbb(k,mu)
        if X=='AB':
            return Dt(X,i,mu)*Dt(X,j,mu)*Rxx(k,mu)-(Dt(X,i,mu)*Dt('A',j,mu)+Dt('A',i,mu)*Dt(X,j,mu))*Rxa(k,mu)-(Dt(X,i,mu)*Dt('B',j,mu)+Dt('B',i,mu)*Dt(X,j,mu))*Rxb(k,mu)+1/2*(Dt('A',i,mu)*Dt('B',j,mu)+Dt('B',i,mu)*Dt('A',j,mu))*Rab(k,mu)
        print('error Fx')
   
    def integrand(k,mu,i,j):
        return (FX('A',k,mu,i,j)+FX('B',k,mu,i,j)+FX('AB',k,mu,i,j))*k**2
   
    def integrat_rsd(muint,int_i,int_j):
        return integrate.quad(partial(integrand,mu=muint,i=int_i,j=int_j),kmin,kmax,epsrel=0.0000001,epsabs=0,limit=nlim)[0]
    
    def F_rsd():
        f11=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=1,int_j=1),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f12=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=1,int_j=2),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f13=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=1,int_j=3),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f22=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=2,int_j=2),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f23=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=2,int_j=3),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f33=V/(4*math.pi**2)*quad(partial(integrat_rsd,int_i=3,int_j=3),-1, 1,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        return np.array([[f11,f12,f13],[f12,f22,f23],[f13,f23,f33]])
    Ff=F_rsd()
    [[a1,b1,c1],[d1,e1,f1],[g1,h1,i1]]=np.linalg.inv(Ff)
    print('sig(sigma8ba)/sigma8b=',a1**0.5)#/sigma8/bg)
    print('sig(sigma8bb)/sigma8b=',e1**0.5)#/sigma8/bg)
    print('sig(sigma8f)/sigma8f=',i1**0.5)#/sigma8/f_z)
    return([a1**0.5,e1**0.5,i1**0.5])





print('MSE ELGxLBG')
RSD_2tracer(1.6,1.9,0.8,0.00044*Vsurvey(1.6,0.8,10000)/10**6,0.00018*Vsurvey(1.6,0.8,10000)/10**6,10000,'LBG24.2','ELG')
RSD_1tracer(1.6,1.8,0.8,0.00044,10000,'LBG24.2')


#%%

print('MUST')
RSD_1tracer(2,2.2,2,0.00089,15000,'LBG24')
RSD_2tracer(2,2.25,2,0.00042*Vsurvey(2,2,15000)/10**6,0.000084*Vsurvey(2,2,15000)/10**6,15000,'LBG24','LBG25.5')
print('10k')
RSD_1tracer(2,2.2,2,0.00050,15000,'LBG24')
RSD_2tracer(2,2.25,2,0.00021*Vsurvey(2,2,15000)/10**6,0.000042*Vsurvey(2,2,15000)/10**6,15000,'LBG24','LBG25.5')
print('5k')
RSD_1tracer(2,2.2,2,0.00025,15000,'LBG24')
RSD_2tracer(2,2.25,2,0.00011*Vsurvey(2,2,15000)/10**6,0.000025*Vsurvey(2,2,15000)/10**6,15000,'LBG24','LBG25.5')
print('5k2')
RSD_1tracer(2,2.2,2,0.00042,9000,'LBG24')
RSD_2tracer(2,2.25,2,0.00018*Vsurvey(2,2,9000)/10**6,0.000042*Vsurvey(2,2,9000)/10**6,9000,'LBG24','LBG25.5')

#%%
print('NTL')
RSD_2tracer(2,2.3,3,0.00025*Vsurvey(2,3,14000)/10**6,0.00024*Vsurvey(2,3,14000)/10**6,14000,'LBG24.2','LBG24.5')
RSD_2tracer(2,2.3,3,0.00049*Vsurvey(2,3,14000)/10**6,0.00081*Vsurvey(2,3,14000)/10**6,14000,'LBG24.2','LBG25')





#%%MUST
print('10kfibers')
RSD_1tracer(2,2.3,2,60*10**6/Vsurvey(2,2,15000),15000,'LBG24.5')
print('5kfibers')
RSD_1tracer(2,2.3,2,30*10**6/Vsurvey(2,2,15000),15000,'LBG24.5')
print('5kfibers 9000deg2')
RSD_1tracer(2,2.3,2,30*10**6/Vsurvey(2,2,9000),9000,'LBG24.5')

print('2tracers MUST')

print('5kfibers 9000deg2 TEST')
RSD_2tracer(2,2.3,2,29.9,0.04,9000,'LBG24.5','QSO')

print('20kfibers')
RSD_2tracer(2,2.3,2,50,10,15000,'LBG24.5','QSO')
print('10kfibers')
RSD_2tracer(2,2.3,2,25,5,15000,'LBG24.5','QSO')
print('5kfibers')
RSD_2tracer(2,2.3,2,13,3,15000,'LBG24.5','QSO')
print('5kfibers 9000deg2')
RSD_2tracer(2,2.3,2,13,3,9000,'LBG24.5','QSO')

