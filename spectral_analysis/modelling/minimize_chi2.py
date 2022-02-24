try:
    import argparse
    import os,sys
    import numpy as np
    from scipy.optimize import minimize
    import get_models_chi2

except ImportError as e:
    print(e)
    raise SystemExit

def main():

    p = argparse.ArgumentParser(description="finding the minimum chi2 for a given input arguments")
    p.add_argument("--inargs", dest="inargs", type=float, nargs=6)

    args = p.parse_args()

    pars = np.array(args.inargs)
    
    #alpha, br_ind, theta, b_now, mag_par, P0

    bnds = ((1.8,2.2),(2,3),(0.05,0.3),(1.5e-6,4e-6),(1e-3,3e-2),(30e-3,90e-3))
    res = minimize(get_models_chi2.get_models_chi2, pars,options={'xtol': 1e-10, 'disp': True},bounds=bnds)
    print "pars for minimizer", pars
    print res.x
if __name__ == "__main__":
    main()
