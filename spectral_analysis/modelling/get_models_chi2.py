try:
    
    import sys
    import numpy as np
    from scipy.interpolate import InterpolatedUnivariateSpline

    sys.path.append('/d22/hin/vikasj78/hawc/GAMERA/lib')
    sys.path.append('/d22/hin/vikasj78/hawc/ver2019_paper/spectral_analysis/modelling/rad_field_richard_tuffs/')
    import RADIATION_To_GAMERA
    import idlsave
    import gappa as gp
    import matplotlib.pyplot as plt
    import threeML as ml

except ImportError as e:
    print(e)
    raise SystemExit
              
def Calculate_chisquare(model, data, data_err_up, data_error_low):
    
    #https://arxiv.org/pdf/physics/0401042.pdf taken from here

    chi2 = 0 #because the minimizer tries to go to zero
    for i in range(len(data)):
        sigma = (data_err_up[i] + data_error_low[i])/2
        alpha = (data_err_up[i] - data_error_low[i])/2  
        delta = model[i] - data[i]

        if(sigma > 0):            
            A = alpha/sigma
            chi2 += np.power((delta/sigma),2) * (1 - 2 * A * (delta / sigma ) + 5 * np.power(A,2) * np.power((delta/sigma),2))
        
        else:
            chi2 += np.power(delta,2)

    return chi2
      
def get_models_chi2(pars):
    
    alpha, br_ind, theta, b_now, mag_par, P0 = pars

    print "alpha", alpha,"br_ind",br_ind,"theta", theta, "b_now",b_now, "mag_par",mag_par, "P0", P0

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

    '''
    Fit parameters
    '''
    #fit parameters for VER2019_high_bfield_recent_history_with_suzaku                                                                                 
    alpha = 2.0
    # b_now = 4.0e-6 #guass                                                                                                                           
    # mag_par = 1e-2
    # P0 = 82e-3
    density = 0.01 # 1/cm^3                                                                                                                            

    #Calculate the true age                                                                                                                            
    age = ((P/((br_ind-1)*P_dot))*(1 - (P0/P)**(br_ind - 1)))/gp.yr_to_sec
    print age, "yrs"
  
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
    rad_field = RADIATION_To_GAMERA.get_radiation_field(idlsave.read('/d22/hin/vikasj78/hawc/ver2019_paper/spectral_analysis/modelling/rad_field_richard_tuffs/readurad.xdr'),distance_from_the_GC,cord_z)

    '''
        Suzaku SED
    '''
    f_int_suzaku = 7e-12 #erg/cm2/s
    e0_suzaku = 2e3 * gp.eV_to_erg
    e1_suzaku = 1e4 * gp.eV_to_erg
    spec_index_suzaku = 2.05
    fl_suzaku = f_int_suzaku * (2-spec_index_suzaku) / ( e1_suzaku**(2-spec_index_suzaku) - e0_suzaku**(2-spec_index_suzaku)) #/erg/cm2/s
    e_suzaku = np.logspace(np.log10(e0_suzaku), np.log10(e1_suzaku), 100)
    suzaku_sed  = fl_suzaku * e_suzaku**(2-spec_index_suzaku)

    '''
        HAWC SED
    '''
    result_hawc = np.loadtxt("fit_info_nhit_cpl.txt",dtype='float',delimiter=',')
    
    e_hawc = result_hawc[:,0] * gp.TeV_to_erg 
    hawc_sed = result_hawc[:,1] # in erg/cm2/s
    hawc_sed_lowe = result_hawc[:,2]
    hawc_sed_upe = result_hawc[:,3]

    '''
        the energy range and sed where we have data  
    '''
    e_range = np.concatenate((e_suzaku,e_hawc))
    data =  np.concatenate((suzaku_sed, hawc_sed)) 

    '''
        create array with injection power, b-field histories
    '''
    t = np.logspace(0.5,np.log10(2*age),20000) # array of times from 1 to 100k yrs
    t0 = P0**(br_ind-1)*P**(2-br_ind) / ((br_ind-1)*P_dot) / gp.yr_to_sec # characteristic spin down time scale
    br_ind_power = (br_ind + 1)/(br_ind -1 )
    l0 = e_dot * ((1 + age/t0)**br_ind_power) # initial spin down luminosity
    lum = theta * l0 * 1 / ((1 + t/t0)**br_ind_power) # luminosity vs. time
    b0 =  b_now*(1 + (age/t0)**0.5) # initial b-field strength  #http://iopscience.iop.org/article/10.1086/527466/pdf equation 10 table 1 values for hess1825
    bb = (b0 / (1 + (t/t0)**0.5)) # b-field vs time
    print "t0", t0, "l0 ",l0, "b0", b0

    '''
    define a power-law spectrum for electrons. 
    Units: E(erg) - dN/dE (1/erg/s)
    '''
    e_el = np.logspace(np.log10(gp.m_e),4,bins) * gp.TeV_to_erg
    e_total_pl = 1 # erg just for normalization 
    ecut = (4.803e-10/2)*np.sqrt((mag_par*l0/theta)/((1+mag_par)*3e10))
    print ecut/gp.TeV_to_erg, "TeV"
    power_law = (e_el/gp.TeV_to_erg)**-alpha * np.exp(-e_el/ecut)
    # renormalise to e_total_pl (integrate E*dN/dE over E)
    power_law *= e_total_pl / fu.Integrate(zip(e_el,power_law * e_el))
    # cast into a 2D array
    power_law_spectrum = np.array(zip(e_el,power_law))

    '''
        create and set up particles object
    '''
    fp = gp.Particles()
    fp.AddThermalTargetPhotons(2.7,0.25*gp.eV_to_erg) # CMB
    fp.AddArbitraryTargetPhotons(rad_field)
    fp.SetCustomInjectionSpectrum(power_law_spectrum)
    fp.SetBField(zip(t,bb))
    fp.SetLuminosity(zip(t,lum))
    fp.SetAmbientDensity(density)
    fp.SetAge(age)

    '''
        calculate electron population
    '''
    fp.CalculateElectronSpectrum()
    sp = np.array(fp.GetParticleSpectrum())
    
    '''
        create a Radiation object and set it up
    '''
    fr = gp.Radiation()
    fr.SetElectrons(sp)
    fr.SetAmbientDensity(density)
    fr.SetBField(fp.GetBField())
    fr.AddArbitraryTargetPhotons(fp.GetTargetPhotons())
    fr.SetDistance(distance)

    '''
        calculate the flux at an arbitrary range of gamma-ray energies (in erg)
    '''
    fr.CalculateDifferentialPhotonSpectrum(e_range)
   
    '''
        extract the different SEDs and spectra
    '''
    model_sed = fr.GetTotalSED()# SED, E^2dNdE (erg/s/cm^2) vs E (TeV)
    model_sed = np.array(model_sed)

    #fit the whole HAWC spectrum
    #chi2 calculation between data and model
    data_err_up = np.concatenate((np.zeros(len(suzaku_sed)),np.zeros(len(hawc_sed_upe))))
    data_err_low = np.concatenate((np.zeros(len(suzaku_sed)),np.zeros(len(hawc_sed_lowe))))

    model = model_sed
    chi2_total_hawc = Calculate_chisquare(model_sed[:,1], data, data_err_up, data_err_low)

    chi2 = chi2_total_hawc

    print chi2
    # fig = plt.figure(figsize=(6,6))
    # ax2 = fig.add_subplot(111)
    # # SED plot
    # ax2.loglog(total_sed[:,0],total_sed[:,1],alpha=0.2,c='k',label="sum",lw=3)
    # ax2.loglog(e_range/gp.TeV_to_erg,data,c='g',label="data",lw=3)
    # plt.show()
    return chi2
