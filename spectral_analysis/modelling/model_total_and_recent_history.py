import sys
import argparse
import numpy as np
from scipy.integrate import quad

sys.path.append('/d22/hin/vikasj78/hawc/GAMERA/lib')
sys.path.append('/d22/hin/vikasj78/hawc/high_energy_source_analysis/VER2019/spectral_analysis/modelling/rad_field_richard_tuffs/')
import RADIATION_To_GAMERA
import idlsave
import gappa as gp

import matplotlib.pyplot as plt
import threeML as ml
import matplotlib as mpl
mpl.rcParams.update({'font.size': 12})

mpik_green = '#057775'
color='black'
fu = gp.Utils()
bins = 200

'''
Known pulsar properties of PSR J2021+365
'''
e_dot = 3.4e36
char_age = 1.72e4 # yrs
P = 104e-3 #sec
P_dot = 9.57e-14 #/sec/sec
distance = 1.8e3 # pc
earth_from_GC = 8.5e3
longi = 74.99 #deg
lati = 0.36 #deg in GCS
density = 0.01 # 1/cm^3

'''
Fit Parameters
'''
global P0
global b_now
global br_ind  #pulsar braking index

global theta  #conversion_eff from pulsar power to e+/-
alpha = 2.0
mag_par = 1e-2

def load_rad_fields():
    '''
        get coordinates in cylindrical corr Rich's model
    '''
    fa = gp.Astro()
    coord = fa.GetCartesian([longi,lati,distance], [0,earth_from_GC,0])
    distance_from_the_GC = np.sqrt(coord[0]**2 + coord[1]**2)
    cord_z = coord[2]

    '''
        get ISRF at coordinates from Rich's model
    '''
    return RADIATION_To_GAMERA.get_radiation_field(idlsave.read('/d22/hin/vikasj78/hawc/high_energy_source_analysis/VER2019/spectral_analysis/modelling/rad_field_richard_tuffs/readurad.xdr'),distance_from_the_GC,cord_z)

def calculate_t0():
    return P0**(br_ind-1)*P**(2-br_ind) / ((br_ind-1)*P_dot) / gp.yr_to_sec

def calculate_true_age():
    return ((P/((br_ind-1)*P_dot))*(1 - (P0/P)**(br_ind - 1)))/gp.yr_to_sec    

def calculate_br_ind_power_factor():
    return (br_ind + 1)/(br_ind -1) #this is braking index dependent power factor goes in most of the eq

def calculate_l0():
    return e_dot * ((1 + calculate_true_age()/calculate_t0())**calculate_br_ind_power_factor()) # initial spin down luminosity

def calculate_b0():
    print 'b_now', b_now
    return  b_now*(1 + (calculate_true_age()/calculate_t0())**0.5) # initial b-field strength  #http://iopscience.iop.org/article/10.1086/527466/pdf equation 10 table 1 values for hess1825

def luminosity(t):
    return theta * calculate_l0() * 1 / ((1 + t/calculate_t0())**calculate_br_ind_power_factor()) # luminosity vs. time

def bfield(t):
    return (calculate_b0() / (1 + (t/calculate_t0())**0.5)) # b-field vs time

def injection_spectrum(e):
    ecut = 270*gp.TeV_to_erg
    #ecut = (4.803e-10/2) * np.sqrt((mag_par * calculate_l0()) / ((1 + mag_par) * 3e10)) #ergs
    return  (e/gp.TeV_to_erg)**-alpha * np.exp(-e/ecut)
    

def setup_particles(fp, e, t, twindow):
    
    lum_t = luminosity(t)
    bf_t = bfield(t)
    
    recent_time = calculate_true_age() - twindow
    t_recent = t[t > recent_time] - recent_time
    lum_recent = lum_t[t > recent_time]
    bf_recent = bf_t[t > recent_time]
    
    fp.AddThermalTargetPhotons(2.7,0.25*gp.eV_to_erg) # CMB
    fp.AddArbitraryTargetPhotons(load_rad_fields())
    fp.SetCustomInjectionSpectrum(zip(e,injection_spectrum(e)))
    fp.SetLuminosity(zip(t_recent,lum_recent))
    fp.SetBField(zip(t_recent,bf_recent))
    fp.SetAmbientDensity(density)
    fp.SetAge(twindow)
    fp.CalculateElectronSpectrum()
    
    return fp

def setup_radiation(fr, sp, e):
    
    fr.SetElectrons(sp)
    fr.SetAmbientDensity(density)
    fr.SetBField(fp.GetBField())
    fr.AddArbitraryTargetPhotons(fp.GetTargetPhotons())
    fr.SetDistance(distance)
    fr.CalculateDifferentialPhotonSpectrum(e)

    return fr

if __name__ == "__main__":

    '''
        Suzaku SED
    '''
    f_int_suzaku = 7e-12 #erg/cm2/s
    e0_suzaku = 2e3 * gp.eV_to_erg
    e1_suzaku = 1e4 * gp.eV_to_erg
    spec_index_suzaku = 2.05
    fl_suzaku = f_int_suzaku * (2-spec_index_suzaku) / ( e1_suzaku**(2-spec_index_suzaku) - e0_suzaku**(2-spec_index_suzaku)) #/erg/cm2/s
    e_suzaku = np.logspace(np.log10(e0_suzaku), np.log10(e1_suzaku), 50)
    sed_suzaku  = fl_suzaku * e_suzaku**(2-spec_index_suzaku)

    '''
        VERITAS SED Old
    '''
    e0_veritas = 1 * gp.TeV_to_erg
    e1_veritas = 30 * gp.TeV_to_erg
    e_veritas = np.logspace(np.log10(e0_veritas), np.log10(e1_veritas), 50)
    fl_veritas = 8.1e-14/gp.TeV_to_erg #1/erg/s/cm2
    spec_index_veritas = 1.75
    sed_veritas = e_veritas*e_veritas*(fl_veritas*(e_veritas/(5*gp.TeV_to_erg))**-spec_index_veritas)

  
    '''
       HAWC SED
    '''
    #hawc_nhit_cpl = ml.load_analysis_results("/d22/hin/vikasj78/hawc/ver2019_paper/spectral_analysis/sed_morphology/plots/nhit/nhit_ver2019_whole_erange_cpl_likeResults.fits")
    hawc_joint_gp =  ml.load_analysis_results("/d22/hin/vikasj78/hawc/ver2019_paper/spectral_analysis/sed_morphology/plots/Ebins/gp/gp_ver2019_whole_erange_cpl_likeResults.fits")
    hawc_joint_nn =  ml.load_analysis_results("/d22/hin/vikasj78/hawc/ver2019_paper/spectral_analysis/sed_morphology/plots/Ebins/nn/nn_ver2019_whole_erange_cpl_likeResults.fits")

    #time, electron energy and gamma energy arrays for the system
    t = np.logspace(0, 5 ,20000) # array of times from 1 to 100k yrs
    e_electron = np.logspace(np.log10(gp.m_e),4,bins) * gp.TeV_to_erg #defines the injected electron energy range
    e_gamma = np.logspace(-6,15,100) * gp.eV_to_erg # defines energies at which gamma-ray emission should be calculated 
   
    '''
        ########## PLOTS ################################
    '''

    #total history low b field
    f, ax = plt.subplots(1,figsize=(6,6))
    #suzaku sed     
    ax.loglog(e_suzaku/gp.TeV_to_erg,sed_suzaku,c='purple',label="Suzaku",linestyle="-",lw=3)   
    #veritas sed   
    #ax.loglog(e_veritas/gp.TeV_to_erg,sed_veritas,c='red',label="VER J2019+368",linestyle="-",lw=3,alpha = 0.7) 
    #hawc sed
    #ml.plot_point_source_spectra(hawc_nhit_cpl,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='purple',fit_colors = 'purple', subplot=ax)

    ml.plot_point_source_spectra(hawc_joint_gp,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='blue',fit_colors = 'blue', subplot=ax)
    ml.plot_point_source_spectra(hawc_joint_nn,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='green',fit_colors = 'green', subplot=ax)
    
    b_now = 1.8e-6 #Gauss
    print 'b_now is set to 1.8 g'
    P0 = 80e-3 #sec
    br_ind = 3
    theta = 0.15
    twindow = calculate_true_age()
    fp = gp.Particles()
    setup_particles(fp, e_electron, t, twindow)
    sp = np.array(fp.GetParticleSpectrum()) #returns diff. spectrum: E(erg) vs dN/dE (1/erg)
    fr = gp.Radiation()
    setup_radiation(fr, sp, e_gamma)
    total_sed = np.array(fr.GetTotalSED())
    synch_sed = np.array(fr.GetSynchrotronSED())
    ic_sed = np.array(fr.GetICSED())
    ax.loglog(synch_sed[:,0],synch_sed[:,1],c='black',linestyle="--")
    ax.loglog(ic_sed[:,0],ic_sed[:,1],c='black',linestyle="-")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel("E"+r"$^2$"+"dN/dE (erg/cm"+r"$^2$"+"/s)")
    ax.set_ylim(ymin=1e-13,ymax=2e-10)
    ax.set_xlim(xmin=1e-15,xmax=2e3)
    ax.legend()
    L = ax.legend()
    L.get_texts()[1].set_text("2HWC J2019+367, GP")
    L.get_texts()[2].set_text("2HWC J2019+367, NN")
    ax.grid()
    plt.show()
    f.savefig("plots/VER2019_low_bfield.png",bbox_inches='tight') 
    f.savefig("plots/VER2019_low_bfield.pdf",bbox_inches='tight')

    #high bfield total and recent history
    f, ax = plt.subplots(1,figsize=(6,6))
    #suzaku sed     
    ax.loglog(e_suzaku/gp.TeV_to_erg,sed_suzaku,c='purple',label="Suzaku",linestyle="-",lw=3)   
    #veritas sed   
    #ax.loglog(e_veritas/gp.TeV_to_erg,sed_veritas,c='red',label="VER J2019+368",linestyle="-",lw=3,alpha = 0.7) 
    #hawc sed
    #ml.plot_point_source_spectra(hawc_nhit_cpl,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='purple',fit_colors = 'purple', subplot=ax)

    ml.plot_point_source_spectra(hawc_joint_gp,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='blue',fit_colors = 'blue', subplot=ax)
    ml.plot_point_source_spectra(hawc_joint_nn,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='green',fit_colors = 'green', subplot=ax)
    
    time_window_var = np.array([500,1000,3000,calculate_true_age()])
    colors = np.array(['cyan','brown','magenta','black'])
    for col,tw in zip(colors,time_window_var):
        b_now = 4.0e-6
        P0 = 80e-3
        br_ind = 3
        theta = 0.15
        twindow = tw
        fp = gp.Particles()
 
        setup_particles(fp, e_electron, t, twindow)
        sp = np.array(fp.GetParticleSpectrum()) #returns diff. spectrum: E(erg) vs dN/dE (1/erg)
        
        fr = gp.Radiation()
        setup_radiation(fr, sp, e_gamma)
        total_sed = np.array(fr.GetTotalSED())
        synch_sed = np.array(fr.GetSynchrotronSED() )
        ic_sed = np.array(fr.GetICSED())
        ax.loglog(synch_sed[:,0],synch_sed[:,1],c=col,linestyle="--")
        ax.loglog(ic_sed[:,0],ic_sed[:,1],c=col,linestyle="-",label='last '+str(int(tw))+' yrs')
    
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel("E"+r"$^2$"+"dN/dE (erg/cm"+r"$^2$"+"/s)")
    ax.set_ylim(ymin=1e-13,ymax=2e-10)
    ax.set_xlim(xmin=1e-15,xmax=2e3)
    L = ax.legend(ncol=2)
    L.get_texts()[1].set_text("2HWC J2019+367, GP")
    L.get_texts()[2].set_text("2HWC J2019+367, NN")
    ax.grid()
    plt.show()
    f.savefig("plots/VER2019_high_bfield_recent_and_total_history.png",bbox_inches='tight')    
    f.savefig("plots/VER2019_high_bfield_recent_and_total_history.pdf",bbox_inches='tight')

    #changing birth period
    f, ax = plt.subplots(1,figsize=(6,6))
    #suzaku sed     
    ax.loglog(e_suzaku/gp.TeV_to_erg,sed_suzaku,c='purple',label="Suzaku",linestyle="-",lw=3)   
    #veritas sed   
    #ax.loglog(e_veritas/gp.TeV_to_erg,sed_veritas,c='red',label="VER J2019+368",linestyle="-",lw=3,alpha = 0.7) 
    #hawc sed
    #ml.plot_point_source_spectra(hawc_nhit_cpl,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='purple',fit_colors = 'purple', subplot=ax)

    ml.plot_point_source_spectra(hawc_joint_gp,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='blue',fit_colors = 'blue', subplot=ax)
    ml.plot_point_source_spectra(hawc_joint_nn,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='green',fit_colors = 'green', subplot=ax)
    
    P0_var = np.array([40,80,100])*1e-3
    colors = np.array(['brown','magenta','black'])
    for col,pvar in zip(colors,P0_var):
        b_now = 1.8e-6
        P0 = pvar
        br_ind = 3
        theta = 0.15
        twindow = calculate_true_age()
        fp = gp.Particles()
 
        setup_particles(fp, e_electron, t, twindow)
        sp = np.array(fp.GetParticleSpectrum()) #returns diff. spectrum: E(erg) vs dN/dE (1/erg)
        
        fr = gp.Radiation()
        setup_radiation(fr, sp, e_gamma)
        total_sed = np.array(fr.GetTotalSED())
        synch_sed = np.array(fr.GetSynchrotronSED() )
        ic_sed = np.array(fr.GetICSED())
        ax.loglog(synch_sed[:,0],synch_sed[:,1],c=col,linestyle="--")
        ax.loglog(ic_sed[:,0],ic_sed[:,1],c=col,linestyle="-",label='$P_0 = $'+str(int(pvar*1e3))+' ms')
    
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel("E"+r"$^2$"+"dN/dE (erg/cm"+r"$^2$"+"/s)")
    ax.set_ylim(ymin=1e-13,ymax=2e-10)
    ax.set_xlim(xmin=1e-15,xmax=2e3)
    L = ax.legend(ncol=2)
    L.get_texts()[1].set_text("2HWC J2019+367, GP")
    L.get_texts()[2].set_text("2HWC J2019+367, NN")
    ax.grid()
    plt.show()
    f.savefig("plots/VER2019_varying_br_period.png",bbox_inches='tight')    
    f.savefig("plots/VER2019_varying_br_period.pdf",bbox_inches='tight')

    #varing br_index
    f, ax = plt.subplots(1,figsize=(6,6))
    #suzaku sed     
    ax.loglog(e_suzaku/gp.TeV_to_erg,sed_suzaku,c='purple',label="Suzaku",linestyle="-",lw=3)   
    #veritas sed   
    #ax.loglog(e_veritas/gp.TeV_to_erg,sed_veritas,c='red',label="VER J2019+368",linestyle="-",lw=3,alpha = 0.7) 
    #hawc sed
    #ml.plot_point_source_spectra(hawc_nhit_cpl,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='purple',fit_colors = 'purple', subplot=ax)

    ml.plot_point_source_spectra(hawc_joint_gp,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='blue',fit_colors = 'blue', subplot=ax)
    ml.plot_point_source_spectra(hawc_joint_nn,ene_min = 1*ml.u.TeV, ene_max = 100* ml.u.TeV, include_extended=True,flux_unit='erg / (cm2 s)',contour_colors ='green',fit_colors = 'green', subplot=ax)
    
    br_ind_var = np.array([2.0,2.5,3])
    colors = np.array(['brown','magenta','black'])
    for col,bv in zip(colors,br_ind_var):
        b_now = 1.8e-6
        P0 = 80e-3
        br_ind = bv 
        theta = 0.15
        twindow = calculate_true_age()
        print twindow, 'yrs'
        fp = gp.Particles()
 
        setup_particles(fp, e_electron, t, twindow)
        sp = np.array(fp.GetParticleSpectrum()) #returns diff. spectrum: E(erg) vs dN/dE (1/erg)
        
        fr = gp.Radiation()
        setup_radiation(fr, sp, e_gamma)
        total_sed = np.array(fr.GetTotalSED())
        synch_sed = np.array(fr.GetSynchrotronSED() )
        ic_sed = np.array(fr.GetICSED())
        ax.loglog(synch_sed[:,0],synch_sed[:,1],c=col,linestyle="--")
        ax.loglog(ic_sed[:,0],ic_sed[:,1],c=col,linestyle="-",label='$n$ = %.1f'%bv)
    
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel("E"+r"$^2$"+"dN/dE (erg/cm"+r"$^2$"+"/s)")
    ax.set_ylim(ymin=1e-13,ymax=2e-10)
    ax.set_xlim(xmin=1e-15,xmax=2e3)
    L = ax.legend(ncol=2)
    L.get_texts()[1].set_text("2HWC J2019+367, GP")
    L.get_texts()[2].set_text("2HWC J2019+367, NN")
    ax.grid()
    plt.show()
    f.savefig("plots/VER2019_varying_br_index.png",bbox_inches='tight')    
    f.savefig("plots/VER2019_varying_br_index.pdf",bbox_inches='tight')

