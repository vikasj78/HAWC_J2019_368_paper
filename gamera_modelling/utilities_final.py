import sys, os
sys.path.append('/d22/hin/vikasj78/GAMERA/lib') #gamera lib installation
sys.path.append('rad_field_richard_tuffs')

import numpy as np
import pandas as pd

import RADIATION_To_GAMERA
import idlsave
import gappa as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
from uncertainties import unumpy

import emcee
from multiprocessing import Pool
import corner
plt.style.use('seaborn-talk')
plt.style.use('presentation.mplstyle')

#helper funcions
def calculate_sed_hess(df):
    df['sed'] = df['e_ref']**2 * df['dnde'] * gp.TeV_to_erg
    df['sed_err'] = df['e_ref']**2 * df['dnde_err'] * gp.TeV_to_erg
    df['sed_errp'] = df['e_ref']**2 * df['dnde_errp'] * gp.TeV_to_erg
    df['sed_errn'] = df['e_ref']**2 * df['dnde_errn'] * gp.TeV_to_erg
    df.loc[df.is_ul==True, 'sed'] = df['e_ref']**2 * df['dnde_ul'] * gp.TeV_to_erg
    df.loc[df.is_ul==True, 'sed_errn'] = 0.3*df['e_ref']**2 * df['dnde_ul'] * gp.TeV_to_erg

def calculate_sed_suzaku(df, e0, e1, nBins):
    spec = unumpy.uarray(df['spec'], df['spec_err'])
    int_flux = unumpy.uarray(df['int_flux']*1.e-13,df['int_flux_err']*1.e-13)
    dnde = int_flux * (2-spec) / ( e1**(2-spec) - e0**(2-spec))
    en = np.logspace(np.log10(e0), np.log10(e1), nBins)

    sed = []
    for dn, sp in zip(dnde,spec):
        sed.append(dn * en**(2-sp))

    sed = np.array(sed)
    sed = np.sum(sed,axis=0)
    dict_sed = {'e_ref': en/gp.TeV_to_erg, 'sed': unumpy.nominal_values(sed), 'sed_err': unumpy.std_devs(sed)}
    df_sed = pd.DataFrame(data=dict_sed)
    return df_sed

def calculate_sed_fermi(data):
    uplim_mask = data['ts'] > 2
    MeVtoTeV = 1e-6
    MeVtoerg = MeVtoTeV * gp.TeV_to_erg

    energy_ul =  data['e_ctr'][~uplim_mask] * MeVtoTeV
    sed_ul = data['e2dnde_ul95'][~uplim_mask] * MeVtoerg
    sed_err_ul = 0.3*data['e2dnde_ul95'][~uplim_mask] * MeVtoerg
    dict_ul = {'e_ref': energy_ul, 'sed': sed_ul, 'sed_err': sed_err_ul}
    df_ul = pd.DataFrame(data=dict_ul)          

    energy = data['e_ctr'][uplim_mask] * MeVtoTeV
    sed = data['e2dnde'][uplim_mask] * MeVtoerg
    sed_err = data['e2dnde_err'][uplim_mask] * MeVtoerg      
    dict_sed = {'e_ref': energy, 'sed': sed, 'sed_err': sed_err}
    df_sed = pd.DataFrame(data=dict_sed)     
                          
    return df_ul, df_sed

def run_mcmc(num_pars,opt_pars,func,x,y,yerr,num_threads=5,num_walkers=33,num_burn_in=50):
    ######
    # Test using the emcee package and the MCMC chain
    print("Start the MCMC part")
    os.environ["OMP_NUM_THREADS"] = str(num_threads)  # to avoid problems with numpy and Pool.
    np.random.seed(42)
    pos = np.array(opt_pars) + 1e-3 * np.random.randn(num_walkers, num_pars)  # initial position of the walkers
    nwalkers, ndim = pos.shape

    ####
    ## Note:
    ## In this example, due to the long time to run the model, the burn-in phase is quite short and
    ## would be better to run everything for a larger number of steps
    ## (like 100 burn-in and 1000 of the chain could be a good option)
    burn_in_steps = num_burn_in
    chain_steps = 30*num_burn_in

    with Pool(num_threads) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, pool=pool, args=(x,y,yerr))
        state = sampler.run_mcmc(pos, burn_in_steps,
                                 progress=True)  # saves the position of the walkers in the state variable
        print("End burn-in phase")
        sampler.reset()  # resets the chain
        sampler.run_mcmc(state, chain_steps, progress=True)  # start again the chain form after the burn-in
        print("End MCMC")

    flat_samples = sampler.get_chain(
        flat=True)  # the burn in could also be set here, with the discard argument. thin option?
    print(flat_samples.shape)
    return flat_samples

def mcmc_results(name, labels, fit_parameters, model_parameters, mcmc_result): # opt_pars,
    '''
       Plotting
    '''
    #finlally lets print the results
    final_results = []
    final_results_err_high = []
    final_results_err_low = []
    for i,par in enumerate(fit_parameters):
        mcmc = np.percentile(mcmc_result[:, i], [16, 50, 84])
        final_results.append(mcmc[1]) 
        q = np.diff(mcmc)
        final_results_err_high.append(q[1])
        final_results_err_low.append(q[0])
        mcmc = mcmc/model_parameters[par][-1]
        q = q/model_parameters[par][-1]
        #curveFit = opt_pars[i]/model_parameters[par][-1]
        
        txt = "{3} = {0:.1e}-{1:.1e}+{2:.1e}"#, curve fit = {4:.1e}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])#, curveFit)
        print(txt)

    final_results = np.array(final_results)
    final_results_err_high = np.array(final_results_err_high)
    final_results_err_low = np.array(final_results_err_low)
    #print("initial pars:", pars, "final results:", final_results)
    ## This shows the correlation plot between the parameters
    ## The lines are the original true values that were used to
    ## obtain the models
    fig = corner.corner(mcmc_result, labels=labels, truths=final_results)
    return final_results, final_results_err_high, final_results_err_low

def get_pwn_sed(obj_fit, fit_values): 
    bins = 200
    t = np.logspace(0, 5 ,20000) # array of times from 1 to 100k yrs                                                                                                                                                                       
    e_photon = np.logspace(-6,15,bins) * gp.eV_to_erg # defines energies at which gamma-ray emission should be calculated 
    e_electron = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg),4,bins) * gp.TeV_to_erg #defines the injected electron energy range
    #pwn
    obj_fit.theta = fit_values[0]/obj_fit.theta_scale
    obj_fit.b_now = fit_values[1]/obj_fit.b_now_scale
    obj_fit.P0 = fit_values[2]/obj_fit.P0_scale
    obj_fit.ecut = fit_values[3]/obj_fit.ecut_scale
    obj_fit.alpha = fit_values[4]/obj_fit.alpha_scale
    obj_fit.time_frac_xray = fit_values[5]/obj_fit.time_frac_xray_scale
    obj_fit.time_frac_pwn = fit_values[6]/obj_fit.time_frac_pwn_scale
    pwn_sed_xray, _, _ = obj_fit.model_pwn(t, e_electron, e_photon, obj_fit.time_frac_xray, True)
    pwn_sed_pwn, _, _ = obj_fit.model_pwn(t, e_electron, e_photon, obj_fit.time_frac_pwn, True)
    pwn_sed_relic, fp_pwn, fr_pwn = obj_fit.model_pwn(t, e_electron, e_photon, 1-obj_fit.time_frac_pwn, False)
    return pwn_sed_xray, pwn_sed_pwn, pwn_sed_relic, fp_pwn, fr_pwn

def get_snr_sed(obj_fit, fit_values): 
    bins = 200
    t = np.logspace(0, 5 ,20000) # array of times from 1 to 100k yrs                                                                                                                                                                       
    e_photon = np.logspace(-6,15,bins) * gp.eV_to_erg # defines energies at which gamma-ray emission should be calculated 
    e_electron_snr = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg),1,bins) * gp.TeV_to_erg
    e_proton_snr = np.logspace(-5,4,bins) * gp.TeV_to_erg
    #snr
    obj_fit.alpha_p = fit_values[0]/obj_fit.alpha_p_scale
    obj_fit.power_p = fit_values[1]/obj_fit.power_p_scale
    obj_fit.del_alpha = fit_values[2]/obj_fit.del_alpha_scale
    obj_fit.k_ep = fit_values[3]/obj_fit.k_ep_scale
    obj_fit.b_now_snr = fit_values[4]/obj_fit.b_now_snr_scale
    snr_sed, fr_snr = obj_fit.model_snr(e_photon, e_electron_snr, e_proton_snr)
    return snr_sed, fr_snr

def check_sample_within_errs(sample, low_lim, high_lim):
    sample_is_good = True
    for i in range(len(sample)):
        if (sample[i] < high_lim[i] and sample[i] > low_lim[i]):
            continue
        else:
            sample_is_good = False
    return sample_is_good

def set_values_from_fit_results_pwn(obj_fit_pwn, fit_values_pwn):
    #sets the pars values by scaling them back from 
    #the best minimised values from the minimiser
    #pwn pars
    obj_fit_pwn.theta = fit_values_pwn[0]/obj_fit_pwn.theta_scale
    obj_fit_pwn.b_now = fit_values_pwn[1]/obj_fit_pwn.b_now_scale
    obj_fit_pwn.P0 = fit_values_pwn[2]/obj_fit_pwn.P0_scale
    obj_fit_pwn.ecut = fit_values_pwn[3]/obj_fit_pwn.ecut_scale
    obj_fit_pwn.alpha = fit_values_pwn[4]/obj_fit_pwn.alpha_scale
    obj_fit_pwn.time_frac_xray = fit_values_pwn[5]/obj_fit_pwn.time_frac_xray_scale
    obj_fit_pwn.time_frac_pwn = fit_values_pwn[6]/obj_fit_pwn.time_frac_pwn_scale
    
    return obj_fit_pwn

def set_values_from_fit_results_snr(obj_fit_snr, fit_values_snr):
    #snr pars
    obj_fit_snr.alpha_p = fit_values_snr[0]/obj_fit_snr.alpha_p_scale
    obj_fit_snr.power_p = fit_values_snr[1]/obj_fit_snr.power_p_scale
    obj_fit_snr.del_alpha = fit_values_snr[2]/obj_fit_snr.del_alpha_scale
    obj_fit_snr.k_ep = fit_values_snr[3]/obj_fit_snr.k_ep_scale
    obj_fit_snr.b_now_snr = fit_values_snr[4]/obj_fit_snr.b_now_snr_scale
    
    return obj_fit_snr

def get_residuals(obj_fit, fit_values, energy, sed, sed_err,  emission_class=None, emission_type=None):
    bins = 200
    
    if (emission_class=='snr'):
        obj_fit = set_values_from_fit_results_snr(obj_fit, fit_values)
        e_electron_snr = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg),1,bins) * gp.TeV_to_erg
        e_proton_snr = np.logspace(-5,4,bins) * gp.TeV_to_erg
        model_sed, _ = obj_fit.model_snr(energy, e_electron_snr, e_proton_snr)
        residual = (sed - model_sed[:,1])#/sed_err
        #residual = sed/model_sed[:,1]
        return residual
    
    elif (emission_class=='pwn'):
        obj_fit = set_values_from_fit_results_pwn(obj_fit, fit_values)
        t = np.logspace(0, 5 ,20000) # array of times from 1 to 100k yrs  
        e_electron = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg),4,bins) * gp.TeV_to_erg #defines the injected electron energy range
        if(emission_type=='suzaku'):
            model_sed, _, _ = obj_fit.model_pwn(t, e_electron, energy, obj_fit.time_frac_xray, True)
        elif(emission_type=='relic'):
            model_sed, _, _ = obj_fit.model_pwn(t, e_electron, energy, 1-obj_fit.time_frac_pwn, False)
        elif(emission_type=='pwn'):
            model_sed, _, _ = obj_fit.model_pwn(t, e_electron, energy, obj_fit.time_frac_pwn, True)
        else:
            print('Not a valid type of emission')
        residual = (sed - model_sed[:,1])#/sed_err
        #residual = sed/model_sed[:,1]
        return residual
    else:
        print('Not valid emission class or type!')
        
class pwn_emission:
   
    def __init__(self, bins_pwn_model, location, known_properties, model_parameters):
        self.bins_pwn_model = bins_pwn_model
        #location of the system
        self.longi = location['longi'] #deg gal
        self.lati = location['lati'] #deg gal
        self.distance = location['distance'] #pc
        
        #pulsar properties
        self.e_dot = known_properties['e_dot'] #erg/sec
        self.char_age = known_properties['char_age'] # yrs
        self.P =  known_properties['P']  #sec  
        self.P_dot = known_properties['P_dot']  #/sec/sec   
        
        #fixed parameters
        self.br_ind = model_parameters['br_ind'] #pulsar braking index  
        self.ebreak = model_parameters['ebreak'] #TeV
        self.alpha0 = model_parameters['alpha0'] #radio component
        
        #fit parameters
        self.b_now = model_parameters['b_now'][0] #Gauss
        self.P0 = model_parameters['P0'][0] #sec
        self.theta = model_parameters['theta'][0] #fraction of edot power to electrons
        self.ecut = model_parameters['ecut'][0] #TeV
        self.alpha = model_parameters['alpha'][0] #wind componnet
        self.time_frac_xray = model_parameters['time_frac_xray'][0] #for recent history
        self.time_frac_pwn = model_parameters['time_frac_pwn'][0]
        self.density = model_parameters['density'][0] #/cm3
        
        #fit parameters scalings (only used in fitting)
        
        #low limit
        self.b_now_low = model_parameters['b_now'][1]
        self.P0_low = model_parameters['P0'][1]
        self.theta_low = model_parameters['theta'][1]
        self.ecut_low = model_parameters['ecut'][1] 
        self.alpha_low = model_parameters['alpha'][1] 
        self.time_frac_xray_low = model_parameters['time_frac_xray'][1] #for recent history
        self.time_frac_pwn_low = model_parameters['time_frac_pwn'][1] #for recent history
        self.density_low = model_parameters['density'][1]
        #high limit
        self.b_now_high = model_parameters['b_now'][2]
        self.P0_high = model_parameters['P0'][2]
        self.theta_high = model_parameters['theta'][2]
        self.ecut_high = model_parameters['ecut'][2] 
        self.alpha_high = model_parameters['alpha'][2] 
        self.time_frac_xray_high = model_parameters['time_frac_xray'][2] #for recent history
        self.time_frac_pwn_high = model_parameters['time_frac_pwn'][2] #for recent history
        self.density_high = model_parameters['density'][2]
        #scale
        self.b_now_scale = model_parameters['b_now'][-1]
        self.P0_scale = model_parameters['P0'][-1]
        self.theta_scale = model_parameters['theta'][-1]
        self.ecut_scale = model_parameters['ecut'][-1] 
        self.alpha_scale = model_parameters['alpha'][-1] 
        self.time_frac_xray_scale = model_parameters['time_frac_xray'][-1] #for recent history
        self.time_frac_pwn_scale = model_parameters['time_frac_pwn'][-1] #for recent history
        self.density_scale = model_parameters['density'][-1]        

    def load_rad_fields(self):
        earth_from_GC = 8.5e3 #pc
        '''
            get coordinates in cylindrical corr Rich's model
        '''
        fa = gp.Astro()
        coord = fa.GetCartesian([self.longi,self.lati,self.distance], [0,earth_from_GC,0])
        distance_from_the_GC = np.sqrt(coord[0]**2 + coord[1]**2)

        cord_z = abs(coord[2]) #cylindrical coordinates, rad fields are defined for only +ve z 
        '''
            get ISRF at coordinates from Rich's model
        '''
        rad_fields = RADIATION_To_GAMERA.get_radiation_field(idlsave.read('/d22/hin/vikasj78/rad_field_richard_tuffs/readurad.xdr', verbose=False), distance_from_the_GC,cord_z)
        return rad_fields

    def calculate_t0(self):
        return self.P0**(self.br_ind-1)*self.P**(2-self.br_ind) / ((self.br_ind-1)*self.P_dot) / gp.yr_to_sec

    def calculate_true_age(self):
        return ((self.P/((self.br_ind-1)*self.P_dot))*(1 - (self.P0/self.P)**(self.br_ind - 1)))/gp.yr_to_sec    

    def calculate_br_ind_power_factor(self):
        return (self.br_ind + 1)/(self.br_ind -1) #this is braking index dependent power factor goes in most of the eq

    def calculate_l0(self):
        return self.e_dot * ((1 + self.calculate_true_age()/self.calculate_t0())**self.calculate_br_ind_power_factor()) # initial spin down luminosity

    def calculate_b0(self):
        return  self.b_now*(1 + (self.calculate_true_age()/self.calculate_t0())**0.5) 
        # initial b-field strength  #http://iopscience.iop.org/article/10.1086/527466/pdf equation 10 table 1 values for hess1825

    def luminosity(self,t):
        return self.theta * self.calculate_l0() * 1 / ((1 + t/self.calculate_t0())**self.calculate_br_ind_power_factor()) # luminosity vs. time

    def bfield(self,t):
        return (self.calculate_b0() / (1 + (t/self.calculate_t0())**0.5)) # b-field vs time

    def injection_spectrum_pwn(self, e):
        ecut = np.power(10,self.ecut) * gp.TeV_to_erg
        #ebreak = self.ebreak * gp.TeV_to_erg
        #equation 4 of https://iopscience.iop.org/article/10.1086/527466/pdf the eta and L(t) factor is taken into account in luminosity
        #broken powerlaw with expo cutoff
        #beta = 1
        #beta *= np.sign(self.alpha - self.alpha0)
        #pwl = (e/gp.TeV_to_erg)**-self.alpha0
        #brk = (1 + (e / ebreak) ** ((self.alpha - self.alpha0) / beta)) ** (-beta)
        #spec = pwl * brk * np.exp(-e/ecut)
        pwl = (e/gp.TeV_to_erg)**-self.alpha
        spec = pwl * np.exp(-e/ecut)
        return list(zip(e, spec))

    def setup_particles(self, fp, e_sed, t, twindow, reverse):
        lum_t = self.luminosity(t)
        bf_t = self.bfield(t)

        if(reverse):
            recent_time = self.calculate_true_age() - twindow
            t_recent = t[t > recent_time] - recent_time
            lum_recent = lum_t[t > recent_time]
            bf_recent = bf_t[t > recent_time]
        else:
            recent_time = twindow
            t_recent = t[t < recent_time]
            lum_recent = lum_t[t < recent_time]
            bf_recent = bf_t[t < recent_time]
        fp.AddThermalTargetPhotons(2.7,0.25*gp.eV_to_erg) # CMB
        fp.AddArbitraryTargetPhotons(self.load_rad_fields())
        fp.SetCustomInjectionSpectrum(e_sed)
        fp.SetLuminosity(list(zip(t_recent,lum_recent)))
        fp.SetBField(list(zip(t_recent,bf_recent)))
        fp.SetAmbientDensity(self.density)
        fp.SetAge(twindow)
        fp.CalculateElectronSpectrum()

        return fp

    def setup_radiation(self, fr, fp, sp, e):
        fr.SetElectrons(sp)
        fr.SetAmbientDensity(self.density)
        fr.SetBField(fp.GetBField())
        fr.AddArbitraryTargetPhotons(fp.GetTargetPhotons())
        fr.SetDistance(self.distance)
        fr.CalculateDifferentialPhotonSpectrum(e)

        return fr


    def model_pwn(self, t, e_electron, e_photon, time_frac, reverse):
                                                                
        e_sed = self.injection_spectrum_pwn(e_electron)
        #for gamma ray lets take the full history                                                                                                                                                                                              
        twindow = self.calculate_true_age() * time_frac
        fp = gp.Particles()
        fp.ToggleQuietMode()
        self.setup_particles(fp, e_sed, t, twindow, reverse)
        sp = np.array(fp.GetParticleSpectrum()) #returns diff. spectrum: E(erg) vs dN/dE (1/erg)                                                                                                                                                
        fr = gp.Radiation()
        fr.ToggleQuietMode()
        self.setup_radiation(fr, fp, sp, e_photon)
        model_sed = np.array(fr.GetTotalSED())

        return model_sed, fp, fr


    def fit_pwn_model(self, e_photon, theta_fit, b_now_fit, P0_fit, ecut_fit, alpha_fit, time_frac_xray_fit, time_frac_pwn_fit):
        #print(theta_fit, b_now_fit, P0_fit, ecut_fit, alpha_fit)            
        self.theta = theta_fit/self.theta_scale
        self.b_now = b_now_fit/self.b_now_scale
        self.P0 = P0_fit/self.P0_scale
        self.ecut = ecut_fit/self.ecut_scale
        self.alpha = alpha_fit/self.alpha_scale
        self.time_frac_xray = time_frac_xray_fit/self.time_frac_xray_scale
        self.time_frac_pwn = time_frac_pwn_fit/self.time_frac_pwn_scale
        '''
        GAMERA
        '''
        t = np.logspace(0, 5 ,20000) # array of times from 1 to 100k yrs
        #defines the injected electron energy range
        e_electron = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg),4,self.bins_pwn_model) * gp.TeV_to_erg 
        e_xray = e_photon[0]
        e_pwn = e_photon[1]
        e_relic = e_photon[2]
        xray_sed, _, _ = self.model_pwn(t, e_electron, e_xray, self.time_frac_xray, True) #with recent history
        #total_sed, _, _ = self.model_pwn(t, e_electron, e_xray, 1, True)
        pwn_sed, _, _ = self.model_pwn(t, e_electron, e_pwn, self.time_frac_pwn, True) #with full time history
        relic_sed, _, _ = self.model_pwn(t, e_electron, e_relic, 1-self.time_frac_pwn, False) #with full time history
        #model_sed = np.concatenate((xray_sed[:,1],gamma_sed[:,1]))
        model_sed = np.array((xray_sed[:,1], pwn_sed[:,1], relic_sed[:,1]))
        #print(model_sed.shape)
        return model_sed

    
    ## Auxiliary functions for the MCMC
    def log_prior_pwn(self,pars):
        """
        Uninformative flat prior.
        Needs to be adjusted depending on the chosen parameters

        Arguments:
            - pars : list of parameters
        """

        lim = ([self.theta_low*self.theta_scale,self.b_now_low*self.b_now_scale,self.P0_low*self.P0_scale,
                self.ecut_low*self.ecut_scale,self.alpha_low*self.alpha_scale,self.time_frac_xray_low*self.time_frac_xray_scale,
                self.time_frac_pwn_low*self.time_frac_pwn_scale],
               [self.theta_high*self.theta_scale,self.b_now_high*self.b_now_scale,self.P0_high*self.P0_scale,
                self.ecut_high*self.ecut_scale,self.alpha_high*self.alpha_scale,self.time_frac_xray_high*self.time_frac_xray_scale,
                self.time_frac_pwn_high*self.time_frac_pwn_scale])
        
        a, b, c, d, e, f, g = pars  # extract the parameters
        if lim[0][0] < a < lim[1][0] and  lim[0][1] < b < lim[1][1] and  lim[0][2] < c < lim[1][2] and  lim[0][3] < d < lim[1][3] and  lim[0][4] < e < lim[1][4] and lim[0][5] < f < lim[1][5] and lim[0][6] < g < lim[1][6]:
            return 0.0
        return -np.inf


    def log_likelihood_pwn(self,pars, x, y, yerr):
        """
        Compute the chi2.
        Needs to be adjusted depending on the chosen parameters

        Arguments:
            - pars : list of parameters
            - x, y, yerr : data points with errorbar
        """
        a, b, c, d, e, f, g  = pars
        model = self.fit_pwn_model(x, a, b, c, d, e, f, g)
        sigma2 = yerr ** 2 #https://emcee.readthedocs.io/en/stable/tutorials/line/
        likelihood1 = -0.5 * np.sum((y[0] - model[0]) ** 2 / sigma2[0] + np.log(sigma2[0]))
        likelihood2 = -0.5 * np.sum((y[1] - model[1]) ** 2 / sigma2[1] + np.log(sigma2[1]))
        likelihood3 = -0.5 * np.sum((y[2] - model[2]) ** 2 / sigma2[2] + np.log(sigma2[2]))
        likelihood = likelihood1 + likelihood2 + likelihood3
        
        #likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        #print("---------------------------------")
        #print("pars:", a, b, c, d, e, f, g, "likelihood:", likelihood)
        return likelihood


    def log_prob_pwn(self, pars, x, y, yerr):
        """
        Compute the total log probability.

        Arguments:
            - pars : list of parameters
            - x, y, yerr : data points with errorbar
        """
        
        lp = self.log_prior_pwn(pars)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_pwn(pars, x, y, yerr)

class snr_emission:
   
    def __init__(self, bins_snr_model, location, model_parameters):
        self.bins_snr_model = bins_snr_model
        #location of the system
        self.longi = location['longi'] #deg gal
        self.lati = location['lati'] #deg gal
        self.distance = location['distance'] #pc
        
        #SNR parameters
        self.density_mc = model_parameters['density_mc'][0]
        self.alpha_p = model_parameters['alpha_p'][0]
        self.power_p = model_parameters['power_p'][0]
        self.del_alpha = model_parameters['del_alpha'][0]
        self.k_ep = model_parameters['k_ep'][0]
        self.b_now_snr = model_parameters['b_now_snr'][0]
        #self.rf_en = model_parameters['rf_en'][0]
        
        #fit parameters scalings (only used in fitting)
        
        #low limit
        self.density_mc_low = model_parameters['density_mc'][1]
        self.alpha_p_low = model_parameters['alpha_p'][1]
        self.power_p_low = model_parameters['power_p'][1]
        self.del_alpha_low = model_parameters['del_alpha'][1]
        self.k_ep_low = model_parameters['k_ep'][1]
        self.b_now_snr_low = model_parameters['b_now_snr'][1]
        #high limit
        self.density_mc_high = model_parameters['density_mc'][2]
        self.alpha_p_high = model_parameters['alpha_p'][2]
        self.power_p_high = model_parameters['power_p'][2]
        self.del_alpha_high = model_parameters['del_alpha'][2]
        self.k_ep_high = model_parameters['k_ep'][2]
        self.b_now_snr_high = model_parameters['b_now_snr'][2]
        #SNR parameters scaling
        self.density_mc_scale = model_parameters['density_mc'][-1]
        self.alpha_p_scale = model_parameters['alpha_p'][-1]
        self.power_p_scale = model_parameters['power_p'][-1]
        self.del_alpha_scale = model_parameters['del_alpha'][-1]
        self.k_ep_scale = model_parameters['k_ep'][-1]
        self.b_now_snr_scale = model_parameters['b_now_snr'][-1]
        #self.rf_en_scale = model_parameters['rf_en'][-1]
        
    def load_rad_fields(self):
        earth_from_GC = 8.5e3 #pc
        '''
            get coordinates in cylindrical corr Rich's model
        '''
        fa = gp.Astro()
        coord = fa.GetCartesian([self.longi,self.lati,self.distance], [0,earth_from_GC,0])
        distance_from_the_GC = np.sqrt(coord[0]**2 + coord[1]**2)

        cord_z = abs(coord[2]) #cylindrical coordinates, rad fields are defined for only +ve z 
        '''
            get ISRF at coordinates from Rich's model
        '''
        rad_fields = RADIATION_To_GAMERA.get_radiation_field(idlsave.read('/d22/hin/vikasj78/rad_field_richard_tuffs/readurad.xdr', verbose=False), distance_from_the_GC,cord_z)
        return rad_fields

    def injection_spectrum(self, energy, spec_idx, elow, ehigh, tot_energy, supercutoff):
        fu = gp.Utils()
        elow = elow * gp.TeV_to_erg
        ehigh = ehigh * gp.TeV_to_erg
        power_law = energy**-spec_idx * np.exp(-(energy/ehigh)**supercutoff) * np.exp(-elow/energy)
        power_law *=  tot_energy / fu.Integrate(list(zip(energy,power_law * energy)))
        return list(zip(energy,power_law))

    def model_snr_onlyP(self, e_photon, e_proton):

        #SNR stuff     
        power_protos = np.power(10,self.power_p)
        proton_inj_spec = self.injection_spectrum(e_proton,self.alpha_p,gp.m_p/gp.TeV_to_erg,
                                                  np.power(10,5),power_protons,1) #high cut off set to very high value therefore not used
        fr = gp.Radiation()
        fr.ToggleQuietMode()
        fr.SetAmbientDensity(self.density_mc)
        fr.SetBField(self.b_now_snr)
        fr.AddThermalTargetPhotons(2.7,0.25*gp.eV_to_erg) # CMB
        fr.AddArbitraryTargetPhotons(self.load_rad_fields())
        #fr.AddArbitraryTargetPhotons(self.rf_en*self.load_rad_fields())
        fr.SetDistance(self.distance)
        fr.SetProtons(proton_inj_spec)
        fr.CalculateDifferentialPhotonSpectrum(e_photon)
        model_sed = np.array(fr.GetTotalSED())

        return model_sed, fr
    
    def fit_snr_model_onlyP(self, e_photon, alpha_p_fit, power_p_fit):#, density_mc_fit):    
        self.alpha_p = alpha_p_fit/self.alpha_p_scale
        self.power_p = power_p_fit/self.power_p_scale
        #self.density_mc = density_mc_fit/self.density_mc_scale
        
        '''
        GAMERA
        '''
        e_proton_snr = np.logspace(-5,4,self.bins_snr_model) * gp.TeV_to_erg
        model_sed, _ = self.model_snr_onlyP(e_photon, e_proton_snr)
        return model_sed[:,1]
    
    def model_snr(self, e_photon, e_electron, e_proton):

        #SNR stuff    
        power_protons = np.power(10,self.power_p)
        proton_inj_spec = self.injection_spectrum(e_proton,self.alpha_p,gp.m_p/gp.TeV_to_erg,
                                                  np.power(10,5),power_protons,1) #no cut off at high energies
        alpha_e = self.alpha_p + self.del_alpha
        power_e =  power_protons * np.power(10,self.k_ep)
        electron_inj_spec = self.injection_spectrum(e_electron,alpha_e,100*gp.m_e/gp.TeV_to_erg,1,power_e,2)

        fr = gp.Radiation()
        fr.ToggleQuietMode()
        fr.SetAmbientDensity(self.density_mc)
        fr.SetBField(self.b_now_snr)
        fr.AddThermalTargetPhotons(2.7,0.25*gp.eV_to_erg) # CMB
        fr.AddArbitraryTargetPhotons(self.load_rad_fields())
        #fr.AddArbitraryTargetPhotons(self.rf_en*self.load_rad_fields())
        fr.SetDistance(self.distance)
        fr.SetElectrons(electron_inj_spec)
        fr.SetProtons(proton_inj_spec)
        fr.CalculateDifferentialPhotonSpectrum(e_photon)
        model_sed = np.array(fr.GetTotalSED())

        return model_sed, fr
    
    def fit_snr_model(self, e_photon, alpha_p_fit, power_p_fit, del_alpha_fit, k_ep_fit, b_now_snr_fit):#, rf_en_fit):# density_mc_fit):
        #print(alpha_p_fit, power_p_fit, alpha_e_fit, power_e_fit, density_mc_fit)            
        self.alpha_p = alpha_p_fit/self.alpha_p_scale
        self.power_p = power_p_fit/self.power_p_scale
        self.del_alpha = del_alpha_fit/self.del_alpha_scale
        self.k_ep = k_ep_fit/self.k_ep_scale
        self.b_now_snr = b_now_snr_fit/self.b_now_snr_scale
        #self.rf_en = rf_en_fit/self.rf_en_scale
        #self.density_mc = density_mc_fit/self.density_mc_scale
        
        '''
        GAMERA
        '''
        e_electron_snr = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg),1,self.bins_snr_model) * gp.TeV_to_erg
        e_proton_snr = np.logspace(-5,4,self.bins_snr_model) * gp.TeV_to_erg
        model_sed, _ = self.model_snr(e_photon, e_electron_snr, e_proton_snr)
        return model_sed[:,1]
    
    ## Auxiliary functions for the MCMC
    def log_prior_snr(self,pars):
        """
        Uninformative flat prior.
        Needs to be adjusted depending on the chosen parameters

        Arguments:
            - pars : list of parameters
        """
        lim = ([self.alpha_p_low*self.alpha_p_scale,self.power_p_low*self.power_p_scale,
                self.del_alpha_low*self.del_alpha_scale,self.k_ep_low*self.k_ep_scale,self.b_now_snr_low*self.b_now_snr_scale],   
               [self.alpha_p_high*self.alpha_p_scale,self.power_p_high*self.power_p_scale,
                self.del_alpha_high*self.del_alpha_scale,self.k_ep_high*self.k_ep_scale,self.b_now_snr_high*self.b_now_snr_scale])
        
        a, b, c, d, e = pars  # extract the parameters
        if lim[0][0] < a < lim[1][0] and  lim[0][1] < b < lim[1][1] and  lim[0][2] < c < lim[1][2] and  lim[0][3] < d < lim[1][3] and  lim[0][4] < e < lim[1][4]:
            return 0.0
        return -np.inf


    def log_likelihood_snr(self,pars, x, y, yerr):
        """
        Compute the chi2.
        Needs to be adjusted depending on the chosen parameters

        Arguments:
            - pars : list of parameters
            - x, y, yerr : data points with errorbar
        """
        a, b, c, d, e  = pars
        model = self.fit_snr_model(x, a, b, c, d, e)
        sigma2 = yerr ** 2#https://emcee.readthedocs.io/en/stable/tutorials/line/
        likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        #print('pars: ', a , b, c, d, e, 'likelihood: ', likelihood)
        return likelihood


    def log_prob_snr(self,pars, x, y, yerr):
        """
        Compute the total log probability.

        Arguments:
            - pars : list of parameters
            - x, y, yerr : data points with errorbar
        """
        
        lp = self.log_prior_snr(pars)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_snr(pars, x, y, yerr)


