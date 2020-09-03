import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import odrpack

#genero punti che oscillino intorno ad una retta tramite un'additiva pseudocasuale distribuita secondo una gaussiana

x = np.array([1,2,3,4,5,6,7,8,9,10])
varx = np.random.normal(0,0.5,10)
x = x + varx
y = np.array([1,2,3,4,5,6,7,8,9,10])
vary = np.random.normal(0,0.5,10)
y = y + vary

#scelgo errori casuali patologici

dx = x/10
dy = y/20

#fit ai minimi quadrati ordinari tramite scipy.optimize.curve_fit

pl.figure(1)

pl.errorbar(x,y,dy,dx,fmt='.',color='black',zorder=2)

def linear(x,m,q):
    return m*x + q

print('Modello lineare:')
popt,covm = curve_fit(linear,x,y,sigma=dy,p0=(1,0),absolute_sigma = True)
mopt,qopt = popt
camp = np.linspace(0,11,2)
pl.plot(camp,linear(camp,mopt,qopt),color='red',zorder=1,label='curve_fit')

resnorm = (y-linear(x,mopt,qopt))/dy
chisq = (resnorm**2).sum()
print('\nChi quadro (c_f): %.3f'%(chisq))

#fit ottimizzato agli errori efficaci

popt1,covm1 = curve_fit(linear,x,y,sigma=dy,absolute_sigma=True)
mopt1, qopt1 = popt1

for i in range(100):
    dxy = np.sqrt(dy**2 + (mopt1*dx)**2)
    popt1,covm1 = curve_fit(linear,x,y,p0=popt,sigma=dxy,absolute_sigma=True)
    mopt1, qopt1 = popt1
    chisq1 = (((y - linear(x,mopt1,qopt1))/dxy)**2.).sum()

resnorm1 = (y-linear(x,mopt1,qopt1))/dy
print('\nChi quadro (eff.): %.3f'%(chisq1))

pl.errorbar(x,y,dy,dx,fmt='.',color='black')
pl.plot(camp,linear(camp,mopt1,qopt1),color='yellow',zorder=1, label='modified curve_fit')

#fit tramite algoritmo di orthogonal distance regression

pars = np.array([0,0])

def linearvectorial(pars,x):
    return pars[0]*x + pars[1]

model = odrpack.Model(linearvectorial)
data = odrpack.RealData(x,y,sx=dx,sy=dy)
odr = odrpack.ODR(data,model,beta0=(1.,0.))
out = odr.run()
popt,covm = out.beta,out.cov_beta
mopt,qopt = popt
chisq = out.sum_square

resnorm2 = (y-linear(x,mopt,qopt))/dy

print('\nChi quadro (ODR): %.3f'%(chisq))

pl.errorbar(x,y,dy,dx,fmt='.',color='black')
pl.plot(camp,linear(camp,mopt,qopt),color='blue',zorder=1,label='ODR')
pl.grid(linestyle=':')
pl.legend(shadow=True,loc='upper left')

pl.figure(2)
pl.subplot(211)
pl.errorbar(x,resnorm,dy,fmt='.',color='black',markersize=4)
pl.plot(camp,0*camp,color='red')
pl.grid(linestyle=':')
pl.subplot(212)
pl.errorbar(x,resnorm1,dxy,fmt='.',color='black',markersize=4)
pl.plot(camp,0*camp,color='blue')
pl.grid(linestyle=':')

#proviamo i tre algoritmi su di un modello quadratico altrettanto patologico

pl.figure(11)

yq = y**2
dx = dx
dy = 10*dy

pl.errorbar(x,yq,dy,dx,color='black',fmt='.')

#minimi quadrati ordinari

def quadratic(x,a,b,c):
    return a*x**2 + b*x + c

print('\nModello quadratico:')

popt,covm = curve_fit(quadratic,x,yq,sigma=dy,absolute_sigma=True)
a,b,c = popt
camp = np.linspace(0,12,100)
plt.plot(camp,quadratic(camp,a,b,c),color='red',zorder=1,label='curve_fit')
resnorm = (yq-quadratic(x,a,b,c))/dy
chisq = (resnorm**2).sum()
print('\nChi quadro (c_f): %.3f'%(chisq))

#minimi quadrati agli errori efficaci

def quadraticprime(x,a,b):
    return 2*a*x + b

popt, covm = curve_fit(quadratic,x,yq,sigma=dy,absolute_sigma=True)
a,b,c = popt
for i in range(100):
    dxy = np.sqrt(dy**2 + (quadraticprime(x,a,b)*dx)**2)
    popt, covm = curve_fit(quadratic,x,yq,sigma=dxy,absolute_sigma=True)
    a,b,c = popt
plt.plot(camp,quadratic(camp,a,b,c),color='yellow',zorder=1,label='modified curve_fit')
resnorm1 = (yq-quadratic(x,a,b,c))/dxy
chisq = (resnorm1**2).sum()
print('\nChi quadro (eff): %.3f'%(chisq))

#orthogonal distance regression

def quadraticvec(pars,x):
    return pars[0]*x**2 + pars[1]*x + pars[2]

model = odrpack.Model(quadraticvec)
data = odrpack.RealData(x,yq,sx=dx,sy=dy)
odr = odrpack.ODR(data,model,beta0= (1.,1.,1.))
out = odr.run()
popt1,covm1 = out.beta,out.cov_beta
a,b,c = popt1
plt.plot(camp,quadratic(camp,a,b,c),color='blue',zorder=1,label='ODR')
chisq = out.sum_square

print('\nChi quadro (ODR): %.3f'%(chisq))
pl.grid(linestyle=':')
pl.legend(shadow=True,loc='upper left')

pl.figure(22)
pl.subplot(211)
pl.errorbar(x,resnorm,dy,fmt='.',color='black',markersize=4)
pl.plot(camp,0*camp,color='red')
pl.grid(linestyle=':')
pl.subplot(212)
pl.errorbar(x,resnorm1,dxy,fmt='.',color='black',markersize=4)
pl.plot(camp,0*camp,color='blue')
pl.grid(linestyle=':')

#confrontiamo curve_fit e scipy.odr con un modello sinusoidale ad ampiezza decrescente, su dati reali presi in laboratorio

print('\nModello sinusoidale:')

#dati

t1,p1,t,p = np.loadtxt(r'C:\users\gugli\desktop\laboratorio1\battimenti\battimenti\dati\oscill4.txt',unpack=True)
dt = np.array(len(t)*[0.002])
dp = np.array(len(p)*[1/np.sqrt(12)])

#modello

def harmonicvec(pars,x):
    return pars[0]*np.exp(-pars[1]*x)*np.sin(pars[2]*x + pars[3]) + pars[4]

def harmonic(x,a,b,c,d,e):
    return a*np.exp(-b*x)*np.sin(c*x + d) + e

def harmonicprime(pars,x):
    return -pars[1]*pars[0]*np.exp(-pars[1]*x)*np.sin(pars[2]*x + pars[3]) + pars[0]*np.exp(-pars[1]*x)*pars[2]*np.cos(pars[2]*x+pars[3]) + 0*pars[4]


#curve_fit

popt, covm = curve_fit(harmonic,t,p,sigma=dp,absolute_sigma=True,p0=(125,1/70,4.45,1.5,450))
resnorm = (p-harmonicvec(popt,t))/dp
chisq = (resnorm**2).sum()

#ODR

model = odrpack.Model(harmonicvec)
data = odrpack.RealData(t,p,sx=dt,sy=dp)
odr = odrpack.ODR(data,model,beta0=(125,1/70,4.45,1.5,450))
out = odr.run()
popt1,covm1 = out.beta,out.cov_beta
chisq1 = out.sum_square_eps
deff = np.sqrt(dp**2+(harmonicprime(popt1,t)*dt)**2)

#grafico e residui normalizzati (curve_fit)

pl.figure(111)
pl.subplot(211)
camp = np.linspace(0,17.9,1000)
pl.errorbar(t,p,dp,dt,color='black' ,fmt='.',markersize=4)
pl.plot(camp,harmonicvec(popt,camp),color='red')
pl.grid(linestyle=':')
pl.title('Curve fit')
pl.subplot(212)
pl.errorbar(t,resnorm,color='black',fmt='.',markersize=4)
pl.plot(camp,0*camp,color='red')
pl.grid(linestyle=':')

#grafico e residui normalizzati (ODR)

pl.figure(222)
pl.subplot(211)
pl.errorbar(t,p,dp,dt,color='black' ,fmt='.',markersize=4)
pl.plot(camp,harmonicvec(popt1,camp),color='blue')
pl.grid(linestyle=':')
pl.title('Orthogonal distance regression')
pl.subplot(212)
resnorm1 = (p-harmonicvec(popt1,t))/deff
pl.errorbar(t,resnorm1,color='black',fmt='.',markersize=4)
pl.plot(camp,0*camp,color='blue',zorder=2)
pl.grid(linestyle=':',zorder=1)

print('\nChi quadro (c_f):%.3f'%(chisq))
print('\nChi quadro (ODR):%.3f'%(chisq1))
pl.grid(linestyle=':')

pl.show()