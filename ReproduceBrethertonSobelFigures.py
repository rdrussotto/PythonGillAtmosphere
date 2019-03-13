

#Script version of the notebook that uses
#the Gill module to reproduce the seven figures in Bretherton & Sobel (2003).




import gill




#import Python library dependencies
import numpy as np
import matplotlib.pyplot as plt




#re-import gill for any new changes
#import importlib
#importlib.reload(gill)




#Figure 1: Classical Gill model with a default mass sink
#(Default parameters except negative sign for sink instead of source)
#Note: the top panel is correctly labeled as convergence in the original paper
#      but incorrectly described as divergence in the figure caption.
setupDict1 = gill.setupGillM_Gaussian(D0=-1.)
resultsDict1 = gill.GillComputations(setupDict1)

gill.plotGillDivVortVel(resultsDict1)

plt.savefig('plots/BS03_Figure_1.png')
plt.savefig('plots/BS03_Figure_1.pdf')




#Figure 2: As in Figure 1 but with zonal compensation of the mass sink (i.e. subtract zonal mean)
setupDict2 = gill.setupGillM_Gaussian(D0=-1., zonalcomp=True)
resultsDict2 = gill.GillComputations(setupDict2)

gill.plotGillDivVortVel(resultsDict2)

plt.savefig('plots/BS03_Figure_2.png')
plt.savefig('plots/BS03_Figure_2.pdf')



#Figure 3: As in Figure 2 except with WTG approximation

resultsDict3 = gill.WTG_Computations(setupDict2)

plt.figure(3, figsize=(9, 7), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
gill.plotDivergenceNoFigure(resultsDict3, subtitle='WTG convergence')
plt.xlabel('') #overlaps with bottom title, and unnecessary
plt.subplot(2,1,2)
gill.plotVortVelNoFigure(resultsDict3, subtitle='WTG Velocity and Vorticity')

plt.savefig('plots/BS03_Figure_3.png')
plt.savefig('plots/BS03_Figure_3.pdf')




#Figure 4: Geopotential change corresponding to Figures 3 (top) and 2 (bottom)
plt.figure(4, figsize=(9, 7), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
gill.plotGeopotentialNoFigure(resultsDict3, subtitle='WTG geopotential')
plt.xlabel('')
plt.subplot(2,1,2)
gill.plotGeopotentialNoFigure(resultsDict2, subtitle='Gill geopotential')

plt.savefig('plots/BS03_Figure_4.png')
plt.savefig('plots/BS03_Figure_4.pdf')




#Figure 5: As in Figure 2 but undamped (no friction/dissipation)
resultsDict5 = gill.GillComputations(setupDict2, nodiss=True)

plt.figure(5, figsize=(9, 7), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
gill.plotVortVelNoFigure(resultsDict5, subtitle='Undamped Velocity and Vorticity')
plt.xlabel('')
plt.subplot(2,1,2)
gill.plotGeopotentialNoFigure(resultsDict5, subtitle='Undamped geopotential')

plt.savefig('plots/BS03_Figure_5.png')
plt.savefig('plots/BS03_Figure_5.pdf')




#Figure 6: off-equatorial, zonally compensated heat source (y0 = 1.5). 
#Plot convergence for both Gill and WTG versions
setupDict6 = gill.setupGillM_Gaussian(D0=-1, zonalcomp=True, y0=1.5)
resultsDict6_Gill = gill.GillComputations(setupDict6)
resultsDict6_WTG = gill.WTG_Computations(setupDict6)


plt.figure(6, figsize=(9, 7), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
gill.plotDivergenceNoFigure(resultsDict6_Gill, subtitle='Gill convergence')
plt.xlabel('')
plt.subplot(2,1,2)
gill.plotDivergenceNoFigure(resultsDict6_WTG, subtitle='WTG convergence')

plt.savefig('plots/BS03_Figure_6.png')
plt.savefig('plots/BS03_Figure_6.pdf')




#Figure 7: as in Figure 6 but plotting velocity and vorticity

plt.figure(7, figsize=(9, 7), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
gill.plotVortVelNoFigure(resultsDict6_Gill, subtitle='Gill Velocity and Vorticity')
plt.xlabel('')
plt.subplot(2,1,2)
gill.plotVortVelNoFigure(resultsDict6_WTG, subtitle='WTG Velocity and Vorticity')

plt.savefig('plots/BS03_Figure_7.png')
plt.savefig('plots/BS03_Figure_7.pdf')

