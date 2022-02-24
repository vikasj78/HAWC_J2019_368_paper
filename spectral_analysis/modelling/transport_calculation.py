import numpy as np

Ebands = np.array([0.316,1.77,10,56.23,316.22]) # in TeV
med_energy = []

for i in range(len(Ebands)-1):
    en = np.sqrt(Ebands[i]*Ebands[i+1])
    med_energy.append(en)

med_energy.pop(0) #removing the first energy bin
#med_energy = np.array([2,10,50])
E_gamma = np.array(med_energy)  #in eV Eband1 = 4.46 TeV, Eband2 = 22.3 TeV, Eband3 = 141.2 TeV
E_cmb = 6.626e-4
E_IR = 2.5e-3 # 30 K
#E_electron = np.sqrt((9 * 0.511e6 * 0.511e6 * E_gamma) / (4 * E_cmb))/1e12 #in TeV
E_electron = 17.2*np.sqrt(E_gamma) # in TeV
print "gamma energy in TeV: ", E_gamma
print "electron energy in TeV: ", E_electron

#cooling time

#t_cool = (3e5/E_electron)*(1/0.26 + 1/0.22)
t_ic = (3e5/E_electron)*(1/0.26)
t_s = (3e5/E_electron)*(1/0.39)

t_cool = t_ic*t_s/(t_ic + t_s)

t_sys = 7.03e3
cooling_break = 3e5/(t_sys*(0.26 + 0.39))
print "cooling_break",cooling_break, "TeV"
print "cooling time in kyr: ", t_cool/1e3

#transport speed
#gp 837 days [0.46324248 0.36865416 0.24636667] #ang size in deg
#nn 837 days [0.46507334 0.36892043 0.29841976] #ang size in deg
angular_size = np.deg2rad(np.array([0.36, 0.35, 0.25]))
distance_to_pulsar = 1.8e3 #pc
Radius = angular_size*distance_to_pulsar #in pc
print "Radius in pc: ", Radius
Radius = 3.08567758128e16 * Radius # in m


yr_to_sec = 3.154e7 
t_cool = yr_to_sec*t_cool # in sec

speed = Radius/t_cool
speed = speed/1e3
print "transport speed in km/s: ", speed 

r0 = 0.2 #p
#gp slope = -0.1725 #nn slope = -0.129
slope = -0.129
advc_ind = -(1 + 0.5/slope)
speed_from_adv = 3e5*((Radius/(3.08567758128e16*r0))**-advc_ind)

print "transport speed from advection model in km/s: ", speed_from_adv

diff_ind = 4*slope + 1
diff_co = (Radius*Radius*1e4)/(2*t_cool)

print "diffusion coefficient", diff_co

xray_en = 10e3 #keV
b_field = 4
#en_xray_ele = 132*np.sqrt(xray_en)*(b_field/3)**(-0.5)
en_xray_ele = 3.39*np.sqrt(xray_en)/np.sqrt(b_field)
print "electron_energy",en_xray_ele
tc = (3.e5)*(1/0.26)*(1/200.)
print tc

e_gm = 23
#electron_en = 17.2*np.sqrt(e_gm)

electron_en = 55.
b_kn = 15*electron_en*E_cmb
print "b_kn", b_kn
b_field = 4 #muG
u_rad = 0.26 #ev/cm3
u_mag = 0.025*b_field**2 #ev/cm3
f_kn = 1/(1+b_kn)**1.5
t_ic = (3.1e5/electron_en)*(1/u_rad)*(1/f_kn)
t_s = (3.1e5/electron_en)*(1/u_mag)
t = t_ic*t_s/(t_ic + t_s)
t_cool = 3.1e5*(1/electron_en)*(1/(f_kn*u_rad + u_mag))
print "with KN",t_ic, t_s, t, t_cool
Radius = np.deg2rad(0.35)*distance_to_pulsar #in pc
print Radius
Radius = 3.08567758128e16*Radius
speed = Radius*1e-3/(yr_to_sec*t)
diff_co = (Radius*Radius*1e4)/(2*yr_to_sec*t)
print speed, diff_co, electron_en



t_ic = (3e5/electron_en)*(1/u_mag)
t = t_ic*t_s/(t_ic + t_s)
print "without KN",t_ic, t_s, t

ang_dist = 0.30 #deg
age = t_cool*3.154e7 #sec
parc_km = 3.08567758128e13 #km
tr_vel = np.deg2rad(ang_dist)*1.8e3*parc_km/age
print "transverse velocity", tr_vel

en_el = 55
b = 15*en_el*E_cmb
en_ga = en_el*2.1*b/(1 + (2.1*b)**0.8)**(1/0.8)
print en_ga

#convert energy denisty ev/cm3 to temprature in Kelvin
energy_density_in_photon_field = 2.1 #eV/cm3
temprature = np.power(energy_density_in_photon_field/0.0047225517614,0.25) #Kelvin
