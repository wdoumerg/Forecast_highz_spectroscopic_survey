import numpy as np
import scipy as sp
#import pyccl as ccl
import math
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from functools import partial


from scipy.integrate import quad, dblquad
import pyccl as ccl

yourz=0
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.677, A_s=2.1e-9, n_s=0.968,transfer_function='boltzmann_camb')
a=1./(1+yourz)
Omega_m=cosmo['Omega_m']
Omega_b=cosmo['Omega_b']
Omega_c=cosmo['Omega_c']
Omega_l=1-Omega_m
#print(Omega_m)



#sky=14000#1400#7000

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

def T(k):
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
 
def sigma_chi(z):
    sigma_z=0.01
    return c*(1+z)/(ccl.h_over_h0(cosmo,1/(1+z))*h*100)*sigma_z/(1+z)
        
        
def Dl(z):
    return ccl.background.luminosity_distance(cosmo, 1/(1+z))

def Mc(z,m_max):
    return m_max-5*np.log10(Dl(z)*10**6/10)+2.5*np.log10(1+z)

def phi_m(M,z):
    if z<2.5:
        Muv=-20.6
        phi_star=9.7/1000
        alpha=-1.6
        #print(1)
    elif z<3.5:
        Muv=-20.86#star
        phi_star=5.04/1000
        alpha=-1.78
        #print(2)
    elif z<4.5:
        Muv=-20.63#star
        phi_star=9.25/1000
        alpha=-1.57
        #print(3)
    elif z<5.5:
        Muv=-20.96
        phi_star=3.22/1000
        alpha=-1.6
        #print(4)
    else:
        Muv=-20.91#star
        phi_star=1.64/1000
        alpha=-1.87
        #print(5)
    
    return math.log(10)/2.5*phi_star*10**(-0.4*(1+alpha)*(M-Muv))*math.exp(-10**(-0.4*(M-Muv)))

def n_LBG(z,m_max):
    nlim=1000
    mc=Mc(z,m_max)
    return quad(lambda M:phi_m(M,z),-25,mc ,epsrel=0.0000001,epsabs=0.00000001,limit=nlim)[0]


def zeff_LBG(z1,z2,m):
    nlim=300
    dz=(z2-z1)/nlim
    def H(z):
        return ccl.background.h_over_h0(cosmo, 1/(1+z))*h*100
    def num():
        s=0
        for i in range(nlim):
            z=z1+dz*i
            s=s+dchi_dz(z)**3*H(z)**2*chi(z)**2*n_LBG(z,m)**2*z*dz
        return s
    def denom():
        s=0
        for i in range(nlim):
            z=z1+dz*i
            s=s+dchi_dz(z)**3*H(z)**2*chi(z)**2*n_LBG(z,m)**2*dz
        return s
    zeff=num()/denom()
    #print(zeff)
    return zeff      

 
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
    if gal=='LBG24.8':
        return (-0.98*(24.8-25)+0.11)*(1+z1)+(0.12*(24.8-25)+0.17)*(1+z1)**2
    if gal=='LBG25':
        return 0.11*(1+z1)+0.17*(1+z1)**2#LBG
    if gal=='LBG25.5':
        return (-0.98*(25.5-25)+0.11)*(1+z1)+(0.12*(25.5-25)+0.17)*(1+z1)**2#LBG
    if gal=='QSO':
        return 0.53+0.29*(1+z1)**2
    print('galaxy type unknown')
    
    
    
    
def Fnl_2param(z2,zeff,dz,n,sky,gal,kmax,m_max):
    Vsur=Vsurvey(z2,dz,sky)
    p=1
    kmin=2*math.pi/Vsur**(1/3)
    
    #fiducial
    z=zeff#0.665#z+dz/2
    fNL=0
    if gal=='LBG':
        bg=(-0.98*(m_max-25)+0.11)*(1+zeff)+(0.12*(m_max-25)+0.17)*(1+zeff)**2
    else: 
        bg=bias(z,gal)
    
    def deltab(k):
        return 3*fNL*(bg-p)*deltac*Omega_m/(k**2*T(k)*D(z))*(H0/c)**2

    def P(k,z1,mu):
        return (bg+deltab(k)+f(z1)*mu*mu)**2*Pm(k,z1)

    def ddb_df(k):
        return 3*(bg-p)*deltac*Omega_m/(k**2*T(k)*D(z))*(H0/c)**2
    
    def ddb_db(k):#derivative of deltab by bg
        return 3*fNL*deltac*Omega_m/(k**2*T(k)*D(z))*(H0/c)**2
    
    def Fnl_bias():
        def Fij_fnl_bg(k,mu,int2):
            pk=P(k,z,mu)
            ddbdf=ddb_df(k)
            ddbdb=ddb_db(k)
            if int2==11:
                return 2*(n*pk/(n*pk+1))**2*ddbdf**2*k**2/(bg+deltab(k)+f(z)*mu*mu)**2
            if int2==12:
                return 2*(n*pk/(n*pk+1))**2*ddbdf*(1+ddbdb)*k**2/(bg+deltab(k)+f(z)*mu*mu)**2
            if int2==22:
                return 2*(n*pk/(n*pk+1))**2*(1+ddbdb)**2*k**2/(bg+deltab(k)+f(z)*mu*mu)**2
        
        def integrat_Fk_fnl_bg(muint,int1):
            if int1 ==11:
                return integrate.quad(partial(Fij_fnl_bg,mu=muint,int2=int1),kmin,kmax,epsrel=0.0001,epsabs=0.0001,limit=nlim)[0]
            if int1 ==12:
                return integrate.quad(partial(Fij_fnl_bg,mu=muint,int2=int1),kmin,kmax,epsrel=0.0001,epsabs=0.0001,limit=nlim)[0]
            if int1 ==22:
                return integrate.quad(partial(Fij_fnl_bg,mu=muint,int2=int1),kmin,kmax,epsrel=0.0001,epsabs=0.0001,limit=nlim)[0]
        def F_integrat_fnl_bg():
            
            f11=Vsur/(4*math.pi**2)*quad(partial(integrat_Fk_fnl_bg,int1=11),-1, 1,epsrel=0.0001,epsabs=0.001,limit=nlim)[0]
            f12=Vsur/(4*math.pi**2)*quad(partial(integrat_Fk_fnl_bg,int1=12),-1, 1,epsrel=0.0001,epsabs=0.0001,limit=nlim)[0]
            f22=Vsur/(4*math.pi**2)*quad(partial(integrat_Fk_fnl_bg,int1=22),-1, 1,epsrel=0.0001,epsabs=0.0001,limit=nlim)[0]
            return np.array([[f11,f12],[f12,f22]])
        
        def cov():
            return np.linalg.inv(F_integrat_fnl_bg())
        
        [[a,b],[c1,d]]=cov()
        #print('The covariance matrix is', [[a,b],[c1,d]])
        #print('sigma on fnl and bg are ', a**0.5,' and ',d**0.5)
        return F_integrat_fnl_bg()
        
    return Fnl_bias()

print('constrains Megamapper')
kmax=0.3
print(np.linalg.inv(Fnl_2param(2,zeff_LBG(2,3,24.5),1,0.00057,14000,'LBG24.5',kmax,24.5)+Fnl_2param(3,zeff_LBG(3,4,24.5),1,0.00011,14000,'LBG24.5',kmax,24.5)+Fnl_2param(4,zeff_LBG(4,5,24.5),1,0.00007,14000,'LBG24.5',kmax,24.5))[0][0]**0.5)


print('constrains NTL')
print(np.linalg.inv(Fnl_2param(2,2.3,1,0.00065,14000,'LBG',kmax,24.2)+Fnl_2param(3,3.3,1,0.00011,14000,'LBG',kmax,24.2)+Fnl_2param(4,4.3,1,0.000024,14000,'LBG',kmax,24.2))[0][0]**0.5)
print(np.linalg.inv(Fnl_2param(2,2.3,1,0.00012,14000,'LBG',kmax,24.5)+Fnl_2param(3,3.3,1,0.00027,14000,'LBG',kmax,24.5)+Fnl_2param(4,4.3,1,0.000075,14000,'LBG',kmax,24.5))[0][0]**0.5)
print(np.linalg.inv(Fnl_2param(2,2.3,1,0.0029,14000,'LBG',kmax,25)+Fnl_2param(3,3.3,1,0.00088,14000,'LBG',kmax,25)+Fnl_2param(4,4.3,1,0.00033,14000,'LBG',kmax,25))[0][0]**0.5)

# 
print('constrains ELG MSE')
print(np.linalg.inv(Fnl_2param(1.6,2,0.8,0.00018,10000,'ELG_DESI',kmax,0))[0][0]**0.5)#lrg
print('constrains LBG MSE')
print(np.linalg.inv(Fnl_2param(2.4,2.55,0.8,0.00017,10000,'LBG24.2',kmax,24.2)+Fnl_2param(3.2,3.3,0.8,0.000043,10000,'LBG24.2',kmax,24.2))[0][0]**0.5)


print('MUST')
print(np.linalg.inv(Fnl_2param(2,2.2,0.8,0.00146,15000,'LBG24.2',kmax,24.2)+Fnl_2param(2.8,2.9,0.8,0.00044,15000,'LBG24.2',kmax,24.2)+Fnl_2param(3.6,3.7,0.6,0.000044,15000,'LBG24.2',kmax,24.2))[0][0]**0.5)
print('10k')
print(np.linalg.inv(Fnl_2param(2,2.2,0.8,0.00085,15000,'LBG24.2',kmax,24.2)+Fnl_2param(2.8,2.9,0.8,0.00026,15000,'LBG24.2',kmax,24.2)+Fnl_2param(3.6,3.7,0.6,0.000026,15000,'LBG24.2',kmax,24.2))[0][0]**0.5)
print('5k')
print(np.linalg.inv(Fnl_2param(2,2.2,0.8,0.00043,15000,'LBG24.2',kmax,24.2)+Fnl_2param(2.8,2.9,0.8,0.00013,15000,'LBG24.2',kmax,24.2)+Fnl_2param(3.6,3.7,0.6,0.000013,15000,'LBG24.2',kmax,24.2))[0][0]**0.5)
print('5k2')
print(np.linalg.inv(Fnl_2param(2,2.2,0.8,0.0007,9000,'LBG24.2',kmax,24.2)+Fnl_2param(2.8,2.9,0.8,0.00021,9000,'LBG24.2',kmax,24.2)+Fnl_2param(3.6,3.7,0.6,0.000021,9000,'LBG24.2',kmax,24.2))[0][0]**0.5)




print('1 bin')
print('constrains Megamapper')
np.linalg.inv(Fnl_2param(2,2.35,3,0.000251,14000,'LBG24.5',0.1)++)[0][0]**0.5

#lbg
print('2survey')
Fnl_2param(2,2.35,3,0.000251,28000,'LBG24.5',0.1)#lbg

print('constrains NTL')
Fnl_2param(2,2.26,3,0.000246,14000,'LBG24.2',0.1)#lrg
Fnl_2param(2,2.30,3,0.000493,14000,'LBG24.5',0.1)#lrg
Fnl_2param(2,2.37,3,0.00132,14000,'LBG25',0.1)#lrg


print('constrains ELG MSE')
Fnl_2param(1.6,2,0.8,0.00018,10000,'ELG_DESI',0.1)#lrg
print('constrains LBG MSE')
Fnl_2param(2.4,4,1.6,0.00011,10000,'LBG24.2',0.1)#lrg


print('MUST')
Fnl_2param(2,2.2,2,0.00089,15000,'LBG24',0.1)
print('10k')
Fnl_2param(2,2.2,2,0.00050,15000,'LBG24',0.1)
print('5k')
Fnl_2param(2,2.2,2,0.00025,15000,'LBG24',0.1)
print('5k2')
Fnl_2param(2,2.2,2,0.00042,9000,'LBG24',0.1)


print('constrains Megamapper')
Fnl_2param(2,2.35,3,0.000251,14000,'LBG24.5',0.3)#lbg
print('constrains Megamapper')
Fnl_2param(2,2.35,3,0.000251,28000,'LBG24.5',0.3)#lbg

print('constrains NTL')
Fnl_2param(2,2.26,3,0.000246,14000,'LBG24.2',0.3)#lrg
Fnl_2param(2,2.30,3,0.000493,14000,'LBG24.5',0.3)#lrg
Fnl_2param(2,2.37,3,0.00132,14000,'LBG25',0.3)#lrg


print('constrains ELG MSE')
Fnl_2param(1.6,2,0.8,0.00018,10000,'ELG_DESI',0.3)#lrg
print('constrains LBG MSE')
Fnl_2param(2.4,4,1.6,0.00011,10000,'LBG24.2',0.3)#lrg


print('MUST')
Fnl_2param(2,2.2,2,0.00089,15000,'LBG24',0.3)
print('10k')
Fnl_2param(2,2.2,2,0.00050,15000,'LBG24',0.3)
print('5k')
Fnl_2param(2,2.2,2,0.00025,15000,'LBG24',0.3)
print('5k2')
Fnl_2param(2,2.2,2,0.00042,9000,'LBG24',0.3)





