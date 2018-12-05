from imports import *
from scipy.stats import gamma
from scipy.optimize import curve_fit


global KepMdwarffile
KepMdwarffile = '../GAIAMdwarfs/input_data/Keplertargets/KepMdwarfsv11.csv'


def get_Kepler_Mdwarf_planets():
    '''
    I have a list of Kepler M dwarfs with stellar parameters based on GAIA 
    distances. I also have a list of confirmed Kepler planets from the NASA 
    exoplanet archive. Match the two lists to get the population of Kepler 
    M dwarf planets with high precision and updated stellar parameters. 
    '''

    # get Kepler Mdwarfs parameters from GAIA
    dG = np.loadtxt(KepMdwarffile, delimiter=',')
    KepID,ra_deg,dec_deg,GBPmag,e_GBPmag,GRPmag,e_GRPmag,Kepmag,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag,parallax_mas,e_parallax,dist_pc,ehi_dist,elo_dist,mu,ehi_mu,elo_mu,AK,e_AK,MK,ehi_MK,elo_MK,Rs_RSun,ehi_Rs,elo_Rs,Teff_K,ehi_Teff,elo_Teff,Ms_MSun,ehi_Ms,elo_Ms,logg_dex,ehi_logg,elo_logg = dG.T

    # get planet transits parameters from NASA exoplanet archive 
    dK = np.genfromtxt('Keplertargets/NASAarchive_confirmed_KeplerMdwarfs.csv',
                       delimiter=',', skip_header=66)
    loc_rowid,kepid,kepler_name,koi_disposition,koi_score,koi_period,koi_period_err1,koi_period_err2,koi_time0,koi_time0_err1,koi_time0_err2,koi_impact,koi_impact_err1,koi_impact_err2,koi_duration,koi_duration_err1,koi_duration_err2,koi_depth,koi_depth_err1,koi_depth_err2,koi_ror,koi_ror_err1,koi_ror_err2,koi_prad,koi_prad_err1,koi_prad_err2,koi_incl,koi_incl_err1,koi_incl_err2,koi_dor,koi_dor_err1,koi_dor_err2,koi_limbdark_mod,koi_ldm_coeff4,koi_ldm_coeff3,koi_ldm_coeff2,koi_ldm_coeff1,koi_model_snr,koi_steff,koi_steff_err1,koi_steff_err2,koi_slogg,koi_slogg_err1,koi_slogg_err2,koi_smet,koi_smet_err1,koi_smet_err2,koi_srad,koi_srad_err1,koi_srad_err2,koi_kepmag,koi_jmag,koi_hmag,koi_kmag = dK.T

    # match stellar parameters to confirmed Kepler stars
    Nplanets = 1#kepid.size
    self = KepConfirmedMdwarfPlanets('Keplertargets/KepConfirmedMdwarfPlanets',
                                     Nplanets)
    self._initialize_arrays()
    for i in range(Nplanets):

        print float(i) / Nplanets
        g = np.in1d(KepID, kepid[i])
        
        self.KepIDs[i] = KepID[g] if g.sum() == 1 else np.nan
        self.Jmags[i] = Jmag[g] if g.sum() == 1 else np.nan
        self.e_Jmags[i] = e_Jmag[g] if g.sum() == 1 else np.nan
        self.Hmags[i] = Hmag[g] if g.sum() == 1 else np.nan
        self.e_Hmags[i] = e_Hmag[g] if g.sum() == 1 else np.nan
        self.Kmags[i] = Kmag[g] if g.sum() == 1 else np.nan
        self.e_Kmags[i] = e_Kmag[g] if g.sum() == 1 else np.nan
        self.pars[i] = parallax_mas[g] if g.sum() == 1 else np.nan
        self.e_pars[i] = e_parallax[g] if g.sum() == 1 else np.nan
        self.mus[i] = mu[g] if g.sum() == 1 else np.nan
        self.ehi_mus[i] = ehi_mu[g] if g.sum() == 1 else np.nan
        self.elo_mus[i] = elo_mu[g] if g.sum() == 1 else np.nan
        self.dists[i] = dist_pc[g] if g.sum() == 1 else np.nan
        self.ehi_dists[i] = ehi_dist[g] if g.sum() == 1 else np.nan
        self.elo_dists[i] = elo_dist[g] if g.sum() == 1 else np.nan
        self.AKs[i] = AK[g] if g.sum() == 1 else np.nan
        self.e_AKs[i] = e_AK[g] if g.sum() == 1 else np.nan
        self.MKs[i] = MK[g] if g.sum() == 1 else np.nan
        self.ehi_MKs[i] = ehi_MK[g] if g.sum() == 1 else np.nan
        self.elo_MKs[i] = elo_MK[g] if g.sum() == 1 else np.nan
        self.Rss[i] = Rs_RSun[g] if g.sum() == 1 else np.nan
        self.ehi_Rss[i] = ehi_Rs[g] if g.sum() == 1 else np.nan
        self.elo_Rss[i] = elo_Rs[g] if g.sum() == 1 else np.nan
        self.Teffs[i] = Teff_K[g] if g.sum() == 1 else np.nan
        self.ehi_Teffs[i] = ehi_Teff[g] if g.sum() == 1 else np.nan
        self.elo_Teffs[i] = elo_Teff[g] if g.sum() == 1 else np.nan
        self.Mss[i] = Ms_MSun[g] if g.sum() == 1 else np.nan
        self.ehi_Mss[i] = ehi_Ms[g] if g.sum() == 1 else np.nan
        self.elo_Mss[i] = elo_Ms[g] if g.sum() == 1 else np.nan
        self.loggs[i] = logg_dex[g] if g.sum() == 1 else np.nan
        self.ehi_loggs[i] = ehi_logg[g] if g.sum() == 1 else np.nan
        self.elo_loggs[i] = elo_logg[g] if g.sum() == 1 else np.nan
        self.FeHs[i] = koi_smet[i] if g.sum() == 1 else np.nan
        self.e_FeHs[i] = np.abs([koi_smet_err1[i], koi_smet_err1[i]]).mean() if g.sum() == 1 else np.nan
        
        # planet parameters
        self.Ps[i] = koi_period[i] if g.sum() == 1 else np.nan
        self.e_Ps[i] = np.abs([koi_period_err1[i], koi_period_err2[i]]).mean() if g.sum() == 1 else np.nan
        self.T0s[i] = koi_time0[i] if g.sum() == 1 else np.nan
        self.e_T0s[i] = np.abs([koi_time0_err1[i], koi_time0_err2[i]]).mean() if g.sum() == 1 else np.nan
        self.Ds[i] = koi_duration[i] if g.sum() == 1 else np.nan
        self.e_Ds[i] = np.abs([koi_duration_err1[i], koi_duration_err2[i]]).mean() if g.sum() == 1 else np.nan
        self.Zs[i] = koi_depth[i] if g.sum() == 1 else np.nan
        self.e_Zs[i] = np.abs([koi_depth_err1[i], koi_depth_err2[i]]).mean() if g.sum() == 1 else np.nan
        self.aRs[i] = koi_dor[i] if g.sum() == 1 else np.nan
        self.e_aRs[i] = np.abs([koi_dor_err1[i], koi_dor_err2[i]]).mean() if g.sum() == 1 else np.nan
        self.rpRs[i] = koi_ror[i] if g.sum() == 1 else np.nan
        self.ehi_rpRs[i] = koi_ror_err1[i] if g.sum() == 1 else np.nan
        self.elo_rpRs[i] = abs(koi_ror_err2[i]) if g.sum() == 1 else np.nan
        self.bs[i] = koi_impact[i] if g.sum() == 1 else np.nan
        self.ehi_bs[i] = koi_impact_err1[i] if g.sum() == 1 else np.nan
        self.elo_bs[i] = abs(koi_impact_err2[i]) if g.sum() == 1 else np.nan

        rp, ehirp, elorp = sample_rp(self.KepID[i], self.Rss[i], self.rpRs[i])
        self.rps[i] = rp if g.sum() == 1 else np.nan
        self.ehi_rps[i] = ehirp if g.sum() == 1 else np.nan
        self.elo_rps[i] = elorp if g.sum() == 1 else np.nan
        
        
    # save Kepler M dwarf planet population
    #hdr = ''
    #np.savetxt('Keplertargets/GAIA_NASAarchive_confirmed_KeplerMdwarfs.csv',
    #           outarr, fmt='%.8e', delimiter=',', header=hdr)
    return outarr



class KepConfirmedMdwarfPlanets:

    def __init__(self, fname, Nplanets):
        self.fname_out = fname
        self.Nplanets = int(Nplanets)


    def _initialize_arrays(self):
        N = self.Nplanets

        # stellar parameters
        self.KepIDs = np.zeros(N)
        self.Jmags, self.e_Jmags = np.zeros(N), np.zeros(N)
        self.Hmags, self.e_Hmags = np.zeros(N), np.zeros(N)
        self.Kmags, self.e_Kmags = np.zeros(N), np.zeros(N)
        self.pars, self.e_pars = np.zeros(N), np.zeros(N)
        self.mus, self.ehi_mus, self.elo_mus=np.zeros(N),np.zeros(N),np.zeros(N)
        self.dists, self.ehi_dists, self.elo_dists = np.zeros(N), np.zeros(N), \
                                                     np.zeros(N)
        self.AKs, self.e_AKs = np.zeros(N), np.zeros(N)
        self.MKs, self.ehi_MKs, self.elo_MKs=np.zeros(N),np.zeros(N),np.zeros(N)
        self.Rss, self.ehi_Rss, self.elo_Rss=np.zeros(N),np.zeros(N),np.zeros(N)
        self.Mss, self.ehi_Mss, self.elo_Mss=np.zeros(N),np.zeros(N),np.zeros(N)
        self.Teffs, self.ehi_Teffs, self.elo_Teffs = np.zeros(N), np.zeros(N), \
                                                     np.zeros(N)
        self.loggs, self.ehi_loggs, self.elo_loggs = np.zeros(N), np.zeros(N), \
                                                     np.zeros(N)
        self.FeHs, self.e_FeHs = np.zeros(N), np.zeros(N)
        
        # planet parameters
        self.Ps, self.e_Ps = np.zeros(N), np.zeros(N)
        self.T0s, self.e_T0s = np.zeros(N), np.zeros(N)
        self.Ds, self.e_Ds = np.zeros(N), np.zeros(N)
        self.Zs, self.e_Zs = np.zeros(N), np.zeros(N)
        self.aRs, self.e_aRs = np.zeros(N), np.zeros(N)
        self.rpRs, self.ehi_rpRs, self.elo_rpRs = np.zeros(N), np.zeros(N), \
                                                  np.zeros(N)
        self.bs, self.ehi_bs, self.elo_bs=np.zeros(N),np.zeros(N),np.zeros(N)
        self.rps, self.ehi_rps, self.elo_rps=np.zeros(N),np.zeros(N),np.zeros(N)

            
        
        
    def _pickleobject(self):
        fObj = open(self.fname_out, 'wb')
        pickle.dump(self, fObj)
        fObj.close()

        

def sample_rp(KepID, rpRs, ehi_rpRs, elo_rpRs):
    '''planet planet radius distribution'''
    path = '../GAIAMdwarfs/Gaia-DR2-distances_custom/DistancePosteriors/'
    samp_Rs = np.loadtxt('%s/KepID_allpost_%i'%(path,KepID), delimiter=',',
                         usecols=(9))
    p16, med, p84 = rpRs-elo_rpRs, rpRs, rpRs+ehi_rpRs
    samp_rpRs = get_samples_from_percentiles(p16, med, p84)
    samp_rp = rvs.m2Rearth(rvs.Rsun2m(samp_rpRs * samp_Rs))
    v = np.percentile(samp_rp, (16,50,84))
    return v[1], v[2]-v[1], v[1]-v[0]


def CDF_func(x, l, k, theta):
    return gamma.cdf(x, l, k, theta)
    

def get_samples_from_percentiles(p16, med, p84, Nsamp=1e3):
    '''Given the 16, 50, and 84 percentiles of a parameter's distribution,
    fit CDF and sample it.'''
    popt,pcov = curve_fit(CDF_func, [p16,med,p84], [16,50,84], p0=(10,1,.5))
