import numpy as np
import astropy as ast
import astropy.units as u
from astropy.coordinates import SkyCoord
from IPython import embed

'''
veritas 2018 paper
'''

veritas_ra = '20h19m23s'
veritas_dec = '+36d46m44s'

#from 2014 paper
#veritas_ra = '20h19m25s'
#veritas_dec = '+36d48m14s'

'''
hrr+04:Hessels, J. W. T., Roberts, M. S. E., Ransom, S. M., Kaspi, V. M., Romani, R. W., Ng, C.-Y., Freire, P. C. C. & Gaensler, B. M., 2004. Observations of PSR J2021+3651 and its X-Ray Pulsar Wind Nebula G75.2+0.1. ApJ, 612, 389-397.
'''
atnf_ra = '20h21m05.46s'
atnf_dec = '+36d51m04.8s'

j2019_veritas = SkyCoord(veritas_ra, veritas_dec, frame='icrs')
atnf_j2021 = SkyCoord(atnf_ra, atnf_dec, frame='icrs')

sep=j2019_veritas.separation(atnf_j2021)
print("Separation: {0:0.2f} degrees".format(sep.degree))




'''
Need to solve spherical triangle

'''
sin = np.sin
cos = np.cos
tan = np.tan
atan = np.arctan

#Get astropy calculation
check = j2019_veritas.position_angle(atnf_j2021)
print("Using astropy position_angle {0}".format(check.degree))

#do manual calculation
phi_2 = atnf_j2021.icrs.dec.radian
phi_1 = j2019_veritas.icrs.dec.radian


l2 = atnf_j2021.icrs.ra.radian
l1 = j2019_veritas.icrs.ra.radian

deltal=l2-l1


x = cos(phi_2) * sin(deltal)
y = cos(phi_1) * sin(phi_2) - sin(phi_1) * cos(phi_2) * cos(deltal)
tanalpha=x/y

alpha_one = atan(tanalpha)
print("Using my position angle {0}".format( 180/np.pi * alpha_one))
