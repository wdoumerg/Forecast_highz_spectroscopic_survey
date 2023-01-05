import numpy as np
#import pyccl as ccl
import math
import matplotlib.pyplot as plt
from scipy import integrate
from functools import partial
from scipy.integrate import quad
import pyccl as ccl

Mnu=0.06
yourz=0
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.677, A_s=2.1e-9, n_s=0.968,transfer_function='boltzmann_camb')
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
 
def sigma_chi(z):
    sigma_z=0.01
    return c*(1+z)/(ccl.h_over_h0(cosmo,1/(1+z))*h*100)*sigma_z/(1+z)
    
    
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
    
    
def nP(n,z,gal):
    return(bias(z,gal)+f(z)*0.6**2)*Pm(0.14,z)*n

print('nP Megamapper')
print(nP(0.00079,2.1,'LBG24.5'))
print(nP(0.00036,2.6,'LBG24.5'))
print(nP(0.00011,3.1,'LBG24.5'))
print(nP(0.00007,4.1,'LBG24.5'))
print(nP(0.000251,2.3,'LBG24.5'))

print('nP NTL')
print(nP(0.000246,2.26,'LBG24.2'))
print(nP(0.000493,2.30,'LBG24.5'))
print(nP(0.00132,2.37,'LBG25'))

print('ELG MSE')
print(nP(0.00018,2,'ELG'))
print('LBG MSE')
print(nP(0.00023,2.5,'LBG24.2'))
print(nP(0.00011,2.9,'LBG24.2'))
print(nP(0.000043,3.3,'LBG24.2'))
print(nP(0.00011,2.6,'LBG24.2'))

print('hypothetical LBG')
print(nP(0.00044,1.8,'LBG24.2'))
print(nP(0.00044,1.9,'LBG24.2')+nP(0.00018,1.9,'ELG'))
print('MUST')
print(nP(0.00089,2.3,'LBG24'))
print(nP(0.00042,2.3,'LBG24')+nP(0.000084,2.3,'LBG25.5'))
print(nP(0.0005,2.3,'LBG24'))
print(nP(0.00021,2.3,'LBG24')+nP(0.000042,2.3,'LBG25.5'))
print(nP(0.00025,2.3,'LBG24'))
print(nP(0.00011,2.3,'LBG24')+nP(0.000025,2.3,'LBG25.5'))
print(nP(0.00042,2.3,'LBG24'))
print(nP(0.00018,2.3,'LBG24')+nP(0.000042,2.3,'LBG25.5'))
    


def F_BAO(z2,zeff,dz,n,sky,gal):
    Vsur=Vsurvey(z2,dz,sky)
    kmin=2*math.pi/Vsur**(1/3)
    kmax=0.1*D(0)/D(zeff)#/h
    z=zeff
    bg=bias(z,gal)
    
    ksilk=1.6*(Omega_b*h**2)**0.52*(Omega_m*h**2)**0.73*(1+(10.4*Omega_m*h**2)**(-0.95))/h
    A0=0.48
    G=D(z)*0.758
    Sigma_s=1/ksilk
    #print(Sigma_s)
    Sigma_perp=9.4*sigma_8(z)/0.9
    Sigma_parr=Sigma_perp*(1+f(z))
    
    def R(k,mu):
        return (bg+f(z)*mu**2)**2*math.exp(-k**2*mu**2*sigma_chi(z)**2)
        
    def Fij_BAO(k,mu,int2):
        nk=n
        if int2==11:
            return (mu**2-1)**2*k**2*math.exp(-2*(k*Sigma_s)**1.4)/(Pm(k,z)/Pm(0.2,z)+1/(nk*Pm(0.2,z)*R(k,mu)))**2*math.exp(-k**2*(1-mu**2)*Sigma_perp**2-k**2*mu**2*Sigma_parr**2)
        if int2==12:
            return (mu**2-1)*mu**2*k**2*math.exp(-2*(k*Sigma_s)**1.4)/(Pm(k,z)/Pm(0.2,z)+1/(nk*Pm(0.2,z)*R(k,mu)))**2*math.exp(-k**2*(1-mu**2)*Sigma_perp**2-k**2*mu**2*Sigma_parr**2)
        if int2==22:
            return mu**4*k**2*math.exp(-2*(k*Sigma_s)**1.4)/(Pm(k,z)/Pm(0.2,z)+1/(nk*Pm(0.2,z)*R(k,mu)))**2*math.exp(-k**2*(1-mu**2)*Sigma_perp**2-k**2*mu**2*Sigma_parr**2)
            #print(Fij_fnl_bg(0.1,0.5,11))
    def integrat_Fk_BAO(muint,int1):
        if int1 ==11:
            return integrate.quad(partial(Fij_BAO,mu=muint,int2=int1),kmin,kmax,epsrel=0.00001,epsabs=0.000001,limit=nlim)[0]
        if int1 ==12:
            return integrate.quad(partial(Fij_BAO,mu=muint,int2=int1),kmin,kmax,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
        if int1 ==22:
            return integrate.quad(partial(Fij_BAO,mu=muint,int2=int1),kmin,kmax,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
    def F_integrat_BAO():
        f11=Vsur*A0**2*quad(partial(integrat_Fk_BAO,int1=11),0, 1,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
        f12=Vsur*A0**2*quad(partial(integrat_Fk_BAO,int1=12),0, 1,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
        f22=Vsur*A0**2*quad(partial(integrat_Fk_BAO,int1=22),0, 1,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
        return np.array([[f11,f12],[f12,f22]])

    [[G,H],[I,J]]=np.linalg.inv(F_integrat_BAO())
    print('constrains on Da and H ', G**0.5,' and ',J**0.5)

print('constrains Megamapper')
F_BAO(2,2.1,0.5,0.00079,14000,'LBG24.5')#lbg
F_BAO(2.5,2.6,0.5,0.00036,14000,'LBG24.5')#lbg
F_BAO(3,3.1,1,0.00011,14000,'LBG24.5')#lbg
F_BAO(4,4.1,1,0.00007,14000,'LBG24.5')#lbg
F_BAO(2,2.35,3,0.000251,14000,'LBG24.5')#lbg

print('constrains NTL')
F_BAO(2,2.26,3,0.000246,14000,'LBG24.2')#lrg
F_BAO(2,2.30,3,0.000493,14000,'LBG24.5')#lrg
F_BAO(2,2.37,3,0.00132,14000,'LBG25')#lrg


print('constrains ELG MSE')
F_BAO(1.6,2,0.8,0.00018,10000,'ELG_DESI')#lrg
print('constrains LBG MSE')
F_BAO(2.4,2.5,0.4,0.00023,10000,'LBG24.2')#lrg
F_BAO(2.8,2.9,0.4,0.00011,10000,'LBG24.2')#lrg
F_BAO(3.2,3.3,0.8,0.000043,10000,'LBG24.2')#lrg
F_BAO(2.4,4,1.6,0.00011,10000,'LBG24.2')#lrg


def F_BAO_2tracers(z2,zeff,dz,Na,Nb,sky,gal1,gal2):
    Vsur=Vsurvey(z2,dz,sky)
    kmin=2*math.pi/Vsur**(1/3)
    kmax=0.1*D(0)/D(zeff)#/h
    z=zeff
    
    ksilk=1.6*(Omega_b*h**2)**0.52*(Omega_m*h**2)**0.73*(1+(10.4*Omega_m*h**2)**(-0.95))/h
    A0=0.48
    G=D(z)*0.758
    Sigma_s=1/ksilk
    Sigma_perp=9.4*sigma_8(z)/0.9
    Sigma_parr=Sigma_perp*(1+f(z))
    
    ba=bias(z,gal1)
    bb=bias(z,gal2)
    na=Na/Vsur*10**6
    nb=Nb/Vsur*10**6
    
    def R(k,mu,bg):
        return (bg+f(z)*mu**2)**2*math.exp(-k**2*mu**2*sigma_chi(z)**2)
        
    def Fij_BAO(k,mu,int2,bg,n):
        nk=n
        
        if int2==11:
            return (mu**2-1)**2*k**2*math.exp(-2*(k*Sigma_s)**1.4)/(Pm(k,z)/Pm(0.2,z)+1/(nk*Pm(0.2,z)*R(k,mu,bg)))**2*math.exp(-k**2*(1-mu**2)*Sigma_perp**2-k**2*mu**2*Sigma_parr**2)
        if int2==12:
            return (mu**2-1)*mu**2*k**2*math.exp(-2*(k*Sigma_s)**1.4)/(Pm(k,z)/Pm(0.2,z)+1/(nk*Pm(0.2,z)*R(k,mu,bg)))**2*math.exp(-k**2*(1-mu**2)*Sigma_perp**2-k**2*mu**2*Sigma_parr**2)
        if int2==22:
            return mu**4*k**2*math.exp(-2*(k*Sigma_s)**1.4)/(Pm(k,z)/Pm(0.2,z)+1/(nk*Pm(0.2,z)*R(k,mu,bg)))**2*math.exp(-k**2*(1-mu**2)*Sigma_perp**2-k**2*mu**2*Sigma_parr**2)
            #print(Fij_fnl_bg(0.1,0.5,11))
    def integrat_Fk_BAO(muint,int1,bg1,n1):
        if int1 ==11:
            return integrate.quad(partial(Fij_BAO,mu=muint,int2=int1,bg=bg1,n=n1),kmin,kmax,epsrel=0.00001,epsabs=0.000001,limit=nlim)[0]
        if int1 ==12:
            return integrate.quad(partial(Fij_BAO,mu=muint,int2=int1,bg=bg1,n=n1),kmin,kmax,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
        if int1 ==22:
            return integrate.quad(partial(Fij_BAO,mu=muint,int2=int1,bg=bg1,n=n1),kmin,kmax,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
    def F_integrat_BAO(bp,np):
        f11=Vsur*A0**2*quad(partial(integrat_Fk_BAO,int1=11,bg1=bp,n1=np),0, 1,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
        f12=Vsur*A0**2*quad(partial(integrat_Fk_BAO,int1=12,bg1=bp,n1=np),0, 1,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
        f22=Vsur*A0**2*quad(partial(integrat_Fk_BAO,int1=22,bg1=bp,n1=np),0, 1,epsrel=0.00001,epsabs=0.00001,limit=nlim)[0]
        return [f11,f12,f22]
    [f11,f12,f22]=F_integrat_BAO(ba,na)
    [g11,g12,g22]=F_integrat_BAO(bb,nb)
    [[G,H],[I,J]]=np.linalg.inv([[f11+g11,f12+g12],[f12+g12,f22+g22]])
    print('constrains on Da and H ', G**0.5,' and ',J**0.5)
    
print('hypoth LBG MSE ELG')
F_BAO(1.6,1.8,0.8,0.00044,10000,'LBG24.2')#lrg
F_BAO_2tracers(1.6,1.9,0.8,0.00044*Vsurvey(1.6,0.8,10000)/10**6,0.00018*Vsurvey(1.6,0.8,10000)/10**6,10000,'LBG24.2','ELG')

print('MUST')
F_BAO(2,2.2,2,0.00089,15000,'LBG24')
F_BAO_2tracers(2,2.25,2,0.00042*Vsurvey(2,2,15000)/10**6,0.000084*Vsurvey(2,2,15000)/10**6,15000,'LBG24','LBG25.5')
print('10k')
F_BAO(2,2.2,2,0.00050,15000,'LBG24')
F_BAO_2tracers(2,2.25,2,0.00021*Vsurvey(2,2,15000)/10**6,0.000042*Vsurvey(2,2,15000)/10**6,15000,'LBG24','LBG25.5')
print('5k')
F_BAO(2,2.2,2,0.00025,15000,'LBG24')
F_BAO_2tracers(2,2.25,2,0.00011*Vsurvey(2,2,15000)/10**6,0.000025*Vsurvey(2,2,15000)/10**6,15000,'LBG24','LBG25.5')
print('5k2')
F_BAO(2,2.2,2,0.00042,9000,'LBG24')
F_BAO_2tracers(2,2.25,2,0.00018*Vsurvey(2,2,9000)/10**6,0.000042*Vsurvey(2,2,9000)/10**6,9000,'LBG24','LBG25.5')

print('NTL')
F_BAO_2tracers(2,2.3,3,0.00025*Vsurvey(2,3,14000)/10**6,0.00024*Vsurvey(2,3,14000)/10**6,14000,'LBG24.2','LBG24.5')
F_BAO_2tracers(2,2.3,3,0.00049*Vsurvey(2,3,14000)/10**6,0.00081*Vsurvey(2,3,14000)/10**6,14000,'LBG24.2','LBG25')
