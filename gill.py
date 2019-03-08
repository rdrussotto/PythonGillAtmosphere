# Module containing code for the Gill (1980) analytical solution for 
# the response of the tropical atmosphere to localized heating, 
# based on the Matlab implementation by Bretherton and Sobel (2003). 
# 
# Idea is that this module will be imported into a script that 
# applies it to some particular problem. 
#
# Needs to have all the functionality of the Matlab code I got from 
# Adam, plus ability to specify the heat source however I want, 
# e.g. latent heating based on precipitation output from a GCM. 
#
# Rick Russotto
# Started 17 October 2018

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve



#Function to set up the mass source M. 2 versions: one for simple Gaussian shape like 
#in the Matlab code, one intended for input from GCM output (more versatile)
#

#Function to set up the mass source M
#
#Input variables:
#nx: number of grid points in x (periodic)
#ny: number of grid points in y
#
#lx: size of domain in x (periodic)
#ly: size of domain in y
#
#sx: mass source half-width in x (see Eq. 16 of Bretherton and Sobel, 2003)
#sy: mass source half-width in y (not really half-width)
#
#x0: center of mass source, x-coordinate
#y0: center of mass source, y-coordinate
#
#zonalcomp: whether mass source is zonally compensated (i.e. subtract zonal mean)
#
#D0: scale factor for mass source
#
#Returns dict containing:
#'M':       Mass source
#'dMdy':    meridional first derivative of mass source
#'d2Mdy2':  meridional second derivative of mass source
#'Mhat':    Fourier transform of mass source
#'dMdyhat': Fourier transform of first derivative of mass source
def setupGillM_Gaussian(nx=128, ny=120, lx=20, ly=20, 
                        sx=2, sy=1, x0=0, y0=0, zonalcomp=0, D0=1):
    #Define the grid
    dx = lx/nx
    x = -lx/2.+dx*np.arange(nx)
    dy = ly/ny
    y = -ly/2.+dy*np.arange(ny+1)
    X,Y=np.meshgrid(x,y)
    
    
    #Define M
    kh = np.pi/(2.*sx)
    phase = kh*(X-x0)
    phase[X-x0>sx] = np.pi/2.
    phase[X-x0<-sx] = np.pi/2.
    F = np.cos(phase)
    M = D0*F*np.exp(-(Y-y0)*(Y-y0)/(2*sy*sy))
    M[0,:] = 0
    M[ny,:] = 0
    if zonalcomp == 1: 
        M = M - (np.mean(np.transpose(M)))*np.ones([1,nx])
        
    #Calculate derivatives and Fourier transform
    dMdy = -(Y-y0)*M/(sy*sy)
    d2Mdy2 = ((Y-y0)*(Y-y0)/(sy*sy)-1)*M/(sy*sy)
    Mhat = np.fft.fft(M)
    dMdyhat = np.fft.fft(dMdy)
    
    #Return M, derivatives, Fourier transform:
    returnDict = {
        'M': M,
        'dMdy': dMdy, 
        'd2Mdy2': d2Mdy2, 
        'Mhat': Mhat,
        'dMdyhat': dMdyhat
    }
    return returnDict

#Version where M does not vary in x within the half-width
def setupGillM_Gaussian_onlyY(nx=128, ny=120, lx=20, ly=20, 
                              sx=2, sy=1, x0=0, y0=0, zonalcomp=0, D0=1):
    #Define the grid
    dx = lx/nx
    x = -lx/2.+dx*np.arange(nx)
    dy = ly/ny
    y = -ly/2.+dy*np.arange(ny+1)
    X,Y=np.meshgrid(x,y)
    
    
    #Define M
#     kh = np.pi/(2.*sx)
#     phase = kh*(X-x0)
#     phase[X-x0>sx] = np.pi/2.
#     phase[X-x0<-sx] = np.pi/2.
#     F = np.cos(phase)
    
    F = np.ones(np.shape(X))
    F[X-x0>sx] = 0
    F[X-x0<-sx] = 0
    
    M = D0*F*np.exp(-(Y-y0)*(Y-y0)/(2*sy*sy))
    M[0,:] = 0
    M[ny,:] = 0
    if zonalcomp == 1: 
        M = M - (np.mean(np.transpose(M)))*np.ones([1,nx])
        
    #Calculate derivatives and Fourier transform
    #(Y derivatives should not be changed by changing X variation)
    dMdy = -(Y-y0)*M/(sy*sy)
    d2Mdy2 = ((Y-y0)*(Y-y0)/(sy*sy)-1)*M/(sy*sy)
    Mhat = np.fft.fft(M)
    dMdyhat = np.fft.fft(dMdy)
    
    #Return M, derivatives, Fourier transform:
    returnDict = {
        'M': M,
        'dMdy': dMdy, 
        'd2Mdy2': d2Mdy2, 
        'Mhat': Mhat,
        'dMdyhat': dMdyhat
    }
    return returnDict
    
    
def setupGillM_versatile():
    print('Not implemented yet')




# Maybe start from the bottom and work my way up? 
# Need something that starts after the "Mass source" is specified, certainly. 


#Function to do computations once mass source, etc. are defined
#How to write this? 
#Copy over lines, and type in argument when I reach  variable I haven't found before...
#Start from "Define wavenumber matrix"
#Or actually above, from where M is defined--no actually derivatives depend on way M is defined
#Grid defined by lx, ly, nx, ny--define it inside here? 
#Or pass it other aspects of the grid?

# Function to do the computations for the Gill model
#
# Input variables:
# M:         mass source as a function of x and y
# dMdy:      meridional first derivative of mass source
# dMdyhat:   Fourier transform of first derivative of mass source
# nx, ny, lx, ly: see "setupGillM_Gaussian"
# H:         Layer depth
# g:         gravity
# beta:      variation of Coriolis parameter with y
# nodiss:    if 1, assume Rayleigh friction is negligible
#
# Returns dict containing variables:
# 'D':      divergence
# 'zeta':   vorticity
# 'u':      zonal wind
# 'v':      meridional wind
# 'phi':    geopotential
# 'a':      nondimensonalized Rayleigh friction coefficient (should this be input instead?)
# 'b':      thermal diffusivity, assumed == a
def GillComputations(M, Mhat, dMdyhat, nx=128, ny=120, lx=20, ly=20, 
                     H=1, g=1, beta=1, nodiss=0):
    #Preliminary stuff:
    if(nodiss):
        a = 0.0001
    else:
        a = 0.15   
        
    b=a+0 #to avoid any possible "pointer" issues
    
    #Grid definitions
    dx = lx/nx
    x = -lx/2.+dx*np.arange(nx)
    dy = ly/ny
    y = -ly/2.+dy*np.arange(ny+1)
    X,Y=np.meshgrid(x,y)
    
    #Shallow water phase speed
    c = np.sqrt(g*H)
    
    #Define wavenumber matrix
    kx = (2*np.pi/lx)*np.append(np.arange(nx/2), np.arange(-nx/2,0))
    KX = np.ones([ny+1,1])*kx
    
    #Define v source term Sv = (a*d/dy - beta*y*d/dx)M/H
    Svhat = (a*dMdyhat - beta*1j*KX*Y*Mhat)/H 
   
    # %  Solve (-b(a^2 + beta^2y^2)/c^2 + a*del^2 + beta*d/dx)v = Sv, or
    # %
    # %  a*d2vhat/dy2 + (-b(a^2 + beta^2y^2)/c^2 -a*k^2 + i*k*beta)vhat = Svhat
    # %
    # %  where Sv = (a*d/dy - beta*y*d/dx)M/H is the same source term
    # %  as in WTG (but this time we don't remove wavenumber zero).

    # %  This is done as a loop over wavenumbers, using only the interior 
    # %  y-gridpoints 2:ny (since v = 0 at the boundaries)
    
    #In frequency space, 
    #The computation of the x derivative does not depend on the form of the 
    #function in X. 
    
    #Note: arrays must be initialized as complex to avoid casting to reals
    vhat = np.zeros([ny+1, nx]) + 1j*np.zeros([ny+1,nx]) 
    d1 = a/(dy*dy)
    for i in np.arange(nx):
        k = kx[i]
        d0 = -2*d1 - a*k*k + 1j*k*beta
        e = np.ones(ny-1)
        diags = np.stack((d1*e, d0*e-b*(a*a+np.power(beta*y[1:ny],2))/(c*c), d1*e)) 
        Av = spdiags(diags, [-1, 0, 1], ny-1, ny-1)
        r = Svhat[1:ny,i]
        vhat[1:ny,i] = spsolve(Av,r) #imported from scipy.sparse.linalg
    v = np.real(np.fft.ifft(vhat))
    
    # %  Calculate phi from
    # %
    # %     (b/c^2)phi + du/dx = M/H - dv/dy 
    # %  and
    # %     a*u = -dphi/dx + beta*y*v
    # %  Eliminating u between these equations,
    # %     (a*b/c^2 - d2/dx2)phi = aM/H - a dv/dy - beta*y*dv/dx
    # %  whose FFT in x diagnoses phi
    dvdyhat = np.zeros([ny+1,nx]) + 1j*np.zeros([ny+1,nx])
    dvdyhat[1:ny,:] = (vhat[2:ny+1,:]-vhat[0:ny-1,:])/(2*dy)
    dvdyhat[0,:] = (vhat[1,:] - vhat[0,:])/dy
    dvdyhat[ny,:] = (vhat[ny,:]-vhat[ny-1,:])/dy
    phihat = (a*Mhat/H-a*dvdyhat-1j*beta*Y*KX*vhat)/(a*b/(c*c)+KX*KX)
    phi = np.real(np.fft.ifft(phihat))
    
    D = M/H - b*phi/(c*c)
    
    # %  Calculate vorticity zeta from divergence and v:
    # %
    # %    a*zeta + beta*(y*D + v) = 0

    zeta = (beta/a)*(-Y*D - v)
    
    #%  u calculated using div eqn: du/dx + dv/dy = M/H-b*phi/(c^2)
    dvdy = np.zeros([ny+1,nx]) #This one's supposed to be real
    dvdy[1:ny,:] = (v[2:ny+1,:] - v[0:ny-1,:])/(2*dy)

    uhat = Mhat/H - b*phihat/(c*c)-np.fft.fft(dvdy)
    uhat[:,1:nx] = uhat[:,1:nx]/(1j*KX[:,1:nx])
    
    # %  The k=0 components are indeterminate; for these go to zonally
    # %  averaged vorticity equation dudyhat(:,1) = -zetahat(:,1), with BC that
    # %  the meridional average of uhat(:,1) should equal zero. 
    #zetahat = np.transpose(np.fft.fft(np.transpose(zeta)))
    zetahat = np.fft.fft(zeta)
    dudyhath = -0.5*(zetahat[0:ny,0] + zetahat[1:(ny+1),0])
    uhat[:,0] = np.append(0, dy*np.cumsum(dudyhath))
    uhatmean = np.mean(np.append(uhat[1:ny,0], 0.5*(uhat[0,0])+uhat[ny,0]))
    uhat[:,0] = uhat[:,0] - uhatmean
    
    ucompare = np.real(np.fft.ifft(uhat))
    
    
    #%  u calculated using x momentum eqn: -beta.y.v=-d(phi)/dx-a.u 
    dphidx=np.zeros((ny+1,nx)) #This should be real because phi is real

    dphidx[:,1:nx-1] = (phi[:, 2:nx] - phi[:,0:nx-2])/(2.*dx)
    dphidx[:,0] = (phi[:,1]-phi[:,nx-1])/(2.*dx)
    dphidx[:,nx-1] = (phi[:,0] - phi[:,nx-2])/(2.*dx)
    u = (-1.*beta*Y*v+dphidx)/(-1.*a)

    #OK, now need to return variables that might go into plots. 
    #In Figures 1 and 3 of the Matlab code, plots include:
    #D (divergence)
    #zeta (vorticity)
    #u, v (wind velocity vector components)
    #phi: geopotential
    
    returnDict = {
        'D': D, 
        'zeta': zeta,
        'u': u, 
        'v': v,
        'phi': phi,
        'a': a,
        'b': b
    }
    return returnDict
    
    #what else might I want?
    
# Functions to plot Gill divergence, vorticity, velocity and geopotential the way 
# it's done in the Matlab code. 
# More generally probably want to do this outside the module.
#
# Input variables:
# D, zeta, u, v, a: see "GillComputations"
# xmin, xmax: x limits of plot
# ymin, ymax: y limits of plot
# stride: spacing between grid points where quiver arrows are plotted
# nx, ny, lx, ly: see "setupGillM_Gaussian"
def plotGillConvVortVel(D, zeta, u, v, xmin=-10, xmax=10, ymin=-3.5, ymax=3.5, stride=4, a=.15, nx=128, ny=120, lx=20, ly=20):
    #Grid definitions
    dx = lx/nx
    x = -lx/2.+dx*np.arange(nx)
    dy = ly/ny
    y = -ly/2.+dy*np.arange(ny+1)
    X,Y=np.meshgrid(x,y)
    
    plt.figure(figsize=(9, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1)
    #%    Plot Gill divergence
    #3 sets of contours of different styles
    cint = np.array([-0.9, -0.7, -0.5, -0.3, -0.1])
    plt.contour(x,y,D, levels=cint, colors='k', linestyles = 'solid')
    cint = np.array([-0.06, -0.02])
    plt.contour(x,y,D, levels=cint, colors='k', linestyles = 'dashed')
    cint = np.array([0.02, 0.06, 0.1])
    plt.contour(x,y,D, levels=cint, colors='k', linestyles = 'dashdot')
    plt.axis('equal')
    plt.axis([xmin, xmax, ymin, ymax])
    #plt.xlabel('x/R$_{eq}$')
    plt.ylabel('y/R$_{eq}$')
    plt.text(0.75*xmax + 0.25*xmin, 0.85*ymax + 0.15*ymin, 
         'a = b = '+str(a)+ 'c/R$_{eq}$')
    plt.title('Gill convergence')
    
    plt.subplot(2,1,2)
    #%    czetamax = max(max(zeta))
    czetamax = 3 #fixed contours
    cpos = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*czetamax
    plt.contour(x,y,zeta, levels=cpos, colors='k', linestyles='solid')
    plt.contour(x,y,zeta, levels=np.flip(-1*cpos), colors='k', linestyles='dashed') #NumPy requires contours in increasing order
    plt.axis('equal')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel('x/R$_{eq}$')
    plt.ylabel('y/R$_{eq}$')
    plt.text(0.75*xmax + 0.25*xmin, 0.85*ymax + 0.15*ymin, 
         'a = b = '+str(a)+ 'c/R$_{eq}$')
    plt.title('Gill Velocity and Vorticity')

    #%    Plot velocity vectors
    plt.quiver(x[0:nx:stride], y[0:ny:stride], u[0:ny:stride,0:nx:stride], v[0:ny:stride,0:nx:stride], angles='xy', scale_units='xy', scale=2, color='b')
    
def plotGillGeopotential(phi, xmin=-10, xmax=10, ymin=-3.5, ymax=3.5,
                        nx=128, ny=120, lx=20, ly=20, cphimax=2):
    #Grid definitions
    dx = lx/nx
    x = -lx/2.+dx*np.arange(nx)
    dy = ly/ny
    y = -ly/2.+dy*np.arange(ny+1)
    X,Y=np.meshgrid(x,y)
    
    plt.figure(figsize=(9, 7), dpi=80, facecolor='w', edgecolor='k')
    #plt.subplot(2,1,1) #Don't actually need to do this--only one panel
    #cphimax = 2 #%Fixed contours (?)
    cpos = np.arange(0.1,2.1,0.2)*cphimax

    plt.contour(x,y, phi, cpos, colors='k', linestyles='solid')
    plt.contour(x,y, phi, np.flip(-1*cpos), colors='k', linestyles='dashed')
    plt.axis('equal')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel('x/R$_{eq}$')
    plt.ylabel('y/R$_{eq}$')
    plt.title('Gill geopotential')
    
    
    
#Main function to do what Matlab code does, with defaults.
#Can set this up with kwargs...
def main_default(nx=128, ny=120, lx=20, ly=20, 
                 sx=2, sy=1, x0=0, y0=0, zonalcomp=0, H=1, g=1, beta=1, nodiss=0,
                 xmin=-10, xmax=10, ymin=-3.5, ymax=3.5, stride=4, cphimax=2, D0=1):

    
    dictM = setupGillM_Gaussian(nx=nx, ny=ny, lx=lx, ly=ly, 
                                sx=sx, sy=sy, x0=x0, y0=y0, zonalcomp=zonalcomp, D0=D0)
    dictC = GillComputations(dictM['M'], dictM['Mhat'], dictM['dMdyhat'], 
                             nx=nx, ny=ny, lx=lx, ly=ly, 
                             H=H, g=g, beta=beta, nodiss=nodiss)
    #Run plotting functions
    plotGillConvVortVel(dictC['D'], dictC['zeta'], dictC['u'], dictC['v'], 
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, stride=stride, a=dictC['a'],
              nx=nx, ny=ny, lx=lx, ly=ly)
    plotGillGeopotential(dictC['phi'], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                        nx=nx, ny=ny, lx=lx, ly=ly, cphimax=cphimax)
    

#Version where sign of M is flipped?
#No, just ability to specifcy D0 is sufficient