# http://iopscience.iop.org/article/10.1088/0004-637X/810/2/95/pdf provide
# a list of planet parameters injected into one year of Kepler light curves.
# They provide each planet's MES (multiple event statistc) which we need to
# know in order to compute its detection efficiency. They also provide duration $ D which I can use to estimate the CDPP over the transit duration from the
# published Kepler completeness parameters by interpolating CDPP(durations) to
# D. They also include depth Z and period P so that I can use the CDPP(D) to
# compute the S/N and then compare the S/N to the MES for all planets and
# hopefully fit a function to that data.
from imports import *
import get_Kepler_Mdwarf_planets as gKM
from scipy.optimize import curve_fit


global KepMdwarffile, data_span
# list of M dwarfs from the primary Kepler mission with GAIA-updated stellar parameters
KepMdwarffile = '../GAIAMdwarfs/input_data/Keplertargets/KepMdwarfsv11.csv'
data_span = 372.   # one year of Kepler data (sect 3.1 Christiansen+2015)


def _get_koi_cdpps():
    '''Read-in from a Kepler-released file, the CDPP values on a discrete 
    grid of transit durations, for each KOI.'''
    d = np.genfromtxt('TPSfiles/nph-nstedAPI_clean.txt', skip_header=208,
                      delimiter=',',
                      usecols=(0,64,65,66,67,68,69,70,71,72,73,74,75,76,77))
    kepidC,cdpp1d5,cdpp2,cdpp2d5,cdpp3,cdpp3d5,cdpp4d5,cdpp5,cdpp6,cdpp7d5,cdpp9,cdpp10d5,cdpp12,cdpp12d5,cdpp15 = d.T
    transit_durs = np.array([1.5,2,2.5,3,3.5,4.5,5,6,7.5,9,10.5,12,12.5,15])
    cdpps = np.array([cdpp1d5,cdpp2,cdpp2d5,cdpp3,cdpp3d5,cdpp4d5,cdpp5,cdpp6,
                      cdpp7d5,cdpp9,cdpp10d5,cdpp12,cdpp12d5,cdpp15]).T
    return kepidC, transit_durs, cdpps


def _get_injected_planet_parameters():
    '''Read-in the table from Christiansen+2015 of transiting planet parameters injected 
    into the pixel data of KOIs and the resulting Multi-Event Statistic (analogous to the 
    transit S/N).'''
    fname = 'Keplertargets/Christiansen2015_Table2_injectedplanetparams.csv'
    d = np.loadtxt(fname, delimiter=',', usecols=(0,1,5,6,8))
    kepidI, P, Z, D, MES = d.T
    return kepidI, P, Z, D, MES


def compute_SNR_2_MES():
    '''Get SNR and MES arrays for Kepler stars from various sources.'''    
    # get data of CDPP and injected planet parameters for all KOIs
    kepidC, transit_durs, cdpps = _get_koi_cdpps()
    kepidI, Ps, Zs, Ds, MESfull = _get_injected_planet_parameters()
    Mdwarf_kepids = np.loadtxt(KepMdwarffile, delimiter=',', usecols=(0))

    # compute CDPP(transit duration)
    Nstars = kepidC.size#Mdwarf_kepids.size
    CDPPs, SNRs, MESs = np.zeros(Nstars), np.zeros(Nstars), np.zeros(Nstars)
    isMdwarf = np.zeros(Nstars, dtype=bool)
    for i in range(Nstars):

        if i % 1e3 == 0:
            print float(i)/Nstars
    
	# compute the MES and S/N        
        g = kepidI == kepidC[i]
        if g.sum() == 1:
            CDPPs[i] = gKM.get_fitted_cdpp(kepidC[i], Ds[g])
            MESs[i] = MESfull[g]
            SNRs[i] = Zs[g] / CDPPs[i] * np.sqrt(data_span / Ps[g])

        else:
            CDPPs[i], MESs[i], SNRs[i] = np.repeat(np.nan, 3)

	# is this an M dwarf
	isMdwarf[i] = True if np.any(np.in1d(Mdwarf_kepids, kepidC[i])) else False
 
    return SNRs, MESs, isMdwarf



def _bin_number(SNRs, MESs, Nbin=30):
    # create 2d histogram
    g = np.isfinite(SNRs) & np.isfinite(MESs)
    Nbin = int(Nbin)
    logbins = [np.logspace(0,3,Nbin), np.logspace(0,3,Nbin+1)]
    Nhist, snr_edges, mes_edges = np.histogram2d(SNRs[g], MESs[g], bins=logbins)

    # bin along each dimension
    snr_bin, mes_bin = np.zeros(Nbin-1), np.zeros(Nbin-1)
    e_mes_bin = np.zeros(Nbin-1)
    for i in range(Nbin-1):
        snr_bin[i] = np.mean(snr_edges[i:i+2])
        mes_bin[i] = np.mean(mes_edges[i:i+2])
        for j in range(Nbin-1):
            e_mes_bin[i] = Nhist[i].std() / np.sqrt(Nhist[i].sum()) 
            if np.isnan(e_mes_bin[i]): e_mes_bin[i] = 0

    return snr_edges, mes_edges, Nhist, snr_bin, mes_bin, e_mes_bin


def powerlaw_func(x, A, alpha):
    return A * x**alpha



def fit_SNR2MES(SNRs, MESs, pltt=False):
    '''Fit a powerlaw to the relation between the transit SNR and MES.'''
    # get SNR-MES relation
    SNRedge,MESedge,Nhist,SNRbin,MESbin,e_MESbin = _bin_number(SNRs, MESs)

    # fit powerlaw to 
    p0 = .75, 1
    popt,pcov = curve_fit(powerlaw_func, SNRbin, MESbin, p0=p0)
    psig = np.sqrt(np.diag(pcov))

    # plot
    if pltt:
        plt.pcolormesh(SNRedge, MESedge, Nhist.T, cmap=plt.get_cmap('hot_r'))
        plt.colorbar()
        plt.errorbar(SNRbin, MESbin, e_MESbin, fmt='ko')
        plt.plot([1,1e3], [1,1e3], '-', label='y=x')
        plt.plot(SNRbin, powerlaw_func(SNRbin, *popt), '--',
                 label='y=%.2fx^%.2f'%tuple(popt))
        plt.plot(SNRbin, powerlaw_func(SNRbin, .75, 1), ':',
                 label='y=0.75x (Petigura prediction)')
    	plt.xlabel('Transit S/N'), plt.ylabel('Multi-Event Statistic (MES)')
    	plt.xscale('log'), plt.yscale('log')
        plt.legend(loc='upper left')
    	plt.savefig('plots/MES_SNR.png')
    	plt.show()

    return popt, psig


if __name__ == '__main__':
    # read-in the Kepler-derived MES and transit S/N from transiting planets so that we can map one to the other
    SNRs, MESs, isMdwarf = compute_SNR_2_MES()
    
    # fit the relation between MES and S/N
    #g = isMdwarf
    g = np.repeat(True, SNRs.size)
    coeffs, e_coeffs = fit_SNR2MES(SNRs[g], MESs[g], pltt=1)
