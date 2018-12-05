from imports import *
from scipy.stats import gamma, skewnorm
from scipy.optimize import curve_fit


global KepMdwarffile
KepMdwarffile = '../GAIAMdwarfs/input_data/Keplertargets/KepMdwarfsv11.csv'


def get_Kepler_Mdwarf_planets(fname):
    '''
    I have a list of Kepler M dwarfs with stellar parameters based on GAIA 
    distances. I also have a list of confirmed Kepler planets from the NASA 
    exoplanet archive. Match the two lists to get the empirical population of 
    Kepler M dwarf planets with high precision radii and updated stellar parameters. 
    '''

    # get Kepler Mdwarfs parameters from GAIA
    dG = np.loadtxt(KepMdwarffile, delimiter=',')
    KepID,ra_deg,dec_deg,GBPmag,e_GBPmag,GRPmag,e_GRPmag,Kepmag,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag,parallax_mas,e_parallax,dist_pc,ehi_dist,elo_dist,mu,ehi_mu,elo_mu,AK,e_AK,MK,ehi_MK,elo_MK,Rs_RSun,ehi_Rs,elo_Rs,Teff_K,ehi_Teff,elo_Teff,Ms_MSun,ehi_Ms,elo_Ms,logg_dex,ehi_logg,elo_logg = dG.T

    # get planet transits parameters from NASA exoplanet archive 
    dK = np.genfromtxt('Keplertargets/NASAarchive_confirmed_KeplerMdwarfs.csv',
                       delimiter=',', skip_header=66)
    loc_rowid,kepid,kepler_name,koi_disposition,koi_score,koi_period,koi_period_err1,koi_period_err2,koi_time0,koi_time0_err1,koi_time0_err2,koi_impact,koi_impact_err1,koi_impact_err2,koi_duration,koi_duration_err1,koi_duration_err2,koi_depth,koi_depth_err1,koi_depth_err2,koi_ror,koi_ror_err1,koi_ror_err2,koi_prad,koi_prad_err1,koi_prad_err2,koi_incl,koi_incl_err1,koi_incl_err2,koi_dor,koi_dor_err1,koi_dor_err2,koi_limbdark_mod,koi_ldm_coeff4,koi_ldm_coeff3,koi_ldm_coeff2,koi_ldm_coeff1,koi_model_snr,koi_steff,koi_steff_err1,koi_steff_err2,koi_slogg,koi_slogg_err1,koi_slogg_err2,koi_smet,koi_smet_err1,koi_smet_err2,koi_srad,koi_srad_err1,koi_srad_err2,koi_kepmag,koi_jmag,koi_hmag,koi_kmag = dK.T

    # match stellar parameters to confirmed Kepler stars
    Nplanets = kepid.size
    self = KepConfirmedMdwarfPlanets(fname, Nplanets)
    self._initialize_arrays()
    for i in range(Nplanets):

        g = np.in1d(KepID, kepid[i])
        
        self.KepIDs[i] = kepid[i]
        self.isMdwarf[i] = g.sum() == 1
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
        self.FeHs[i] = koi_smet[i]
        self.e_FeHs[i] = np.abs([koi_smet_err1[i], koi_smet_err1[i]]).mean()
        
        # planet parameters
        self.Ps[i] = koi_period[i]
        self.e_Ps[i] = np.abs([koi_period_err1[i], koi_period_err2[i]]).mean()
        self.T0s[i] = koi_time0[i]
        self.e_T0s[i] = np.abs([koi_time0_err1[i], koi_time0_err2[i]]).mean()
        self.Ds[i] = koi_duration[i]
        self.e_Ds[i] = np.abs([koi_duration_err1[i], koi_duration_err2[i]]).mean()
        self.Zs[i] = koi_depth[i]
        self.e_Zs[i] = np.abs([koi_depth_err1[i], koi_depth_err2[i]]).mean()
        self.aRs[i] = koi_dor[i]
        self.e_aRs[i] = np.abs([koi_dor_err1[i], koi_dor_err2[i]]).mean()
        self.rpRs[i] = koi_ror[i]
        self.ehi_rpRs[i] = koi_ror_err1[i]
        self.elo_rpRs[i] = abs(koi_ror_err2[i])
        self.bs[i] = koi_impact[i]
        self.ehi_bs[i] = koi_impact_err1[i]
        self.elo_bs[i] = abs(koi_impact_err2[i])

        # compute planet parameters
        if self.isMdwarf[i]:
            rps, smas, Teqs, Fs = sample_planet_params(self, i)
            self.rps[i], self.ehi_rps[i], self.elo_rps[i] = rps
            self.smas[i], self.ehi_smas[i], self.elo_smas[i] = smas
            self.Teqs[i], self.ehi_Teqs[i], self.elo_Teqs[i] = Teqs
            self.Fs[i], self.ehi_Fs[i], self.elo_Fs[i] = Fs

    # save Kepler M dwarf planet population
    self._pickleobject()



    
class KepConfirmedMdwarfPlanets:

    def __init__(self, fname, Nplanets):
        self.fname_out = fname
        self.Nplanets = int(Nplanets)


    def _initialize_arrays(self):
        N = self.Nplanets

        # stellar parameters
        self.KepIDs, self.isMdwarf = np.zeros(N), np.zeros(N, dtype=bool)
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
        self.rps = np.repeat(np.nan,N)
        self.ehi_rps = np.repeat(np.nan,N)
        self.elo_rps = np.repeat(np.nan,N)
        self.smas = np.repeat(np.nan,N)
        self.ehi_smas = np.repeat(np.nan,N)
        self.elo_smas = np.repeat(np.nan,N)
        self.Teqs = np.repeat(np.nan,N)
        self.ehi_Teqs = np.repeat(np.nan,N)
        self.elo_Teqs = np.repeat(np.nan,N)
        self.Fs = np.repeat(np.nan,N)
        self.ehi_Fs = np.repeat(np.nan,N)
        self.elo_Fs = np.repeat(np.nan,N)


    def _pickleobject(self):
        fObj = open(self.fname_out, 'wb')
        pickle.dump(self, fObj)
        fObj.close()




def loadpickle(fname):
    fObj = open(fname, 'rb')
    self = pickle.load(fObj)
    fObj.close()
    return self


def resample_PDF(pdf, Nsamp, sig=1e-3):
    pdf_resamp = np.random.choice(pdf, int(Nsamp)) + np.random.randn(int(Nsamp))*sig
    return pdf_resamp


def sample_planet_params(self, index):
    '''planet planet radius distribution'''
    # get stellar parameters PDFs
    g = int(index)
    path = '../GAIAMdwarfs/Gaia-DR2-distances_custom/DistancePosteriors/'
    samp_Rs,samp_Teff,samp_Ms = np.loadtxt('%s/KepID_allpost_%i'%(path,self.KepIDs[g]),
                                           delimiter=',', usecols=(9,10,11)).T
    samp_Rs = resample_PDF(samp_Rs[np.isfinite(samp_Rs)], samp_Rs.size, sig=1e-3)
    samp_Teff = resample_PDF(samp_Teff[np.isfinite(samp_Teff)], samp_Teff.size, sig=5)
    samp_Ms = resample_PDF(samp_Ms[np.isfinite(samp_Ms)], samp_Ms.size, sig=1e-3)
    
    # sample distributions from point estimates
    ##g = self.KepIDs == KepID
    p16 = float(self.rpRs[g] - self.elo_rpRs[g])
    med = float(self.rpRs[g])
    p84 = float(self.rpRs[g] + self.ehi_rpRs[g])
    _,_,samp_rpRs = get_samples_from_percentiles(p16, med, p84, Nsamp=samp_Rs.size)
    
    # compute planet radius PDF
    samp_rp = rvs.m2Rearth(rvs.Rsun2m(samp_rpRs * samp_Rs))
    v = np.percentile(samp_rp, (16,50,84))
    rps = v[1], v[2]-v[1], v[1]-v[0]

    # compute semi-major axis PDF 
    samp_Ps = np.random.normal(self.Ps[g], self.e_Ps[g], samp_Ms.size)
    samp_as = rvs.semimajoraxis(samp_Ps, samp_Ms, 0)
    v = np.percentile(samp_as, (16,50,84))
    smas = v[1], v[2]-v[1], v[1]-v[0]

    # compute equilibrium T PDF (Bond albedo=0)
    samp_Teq = samp_Teff * np.sqrt(.5*rvs.Rsun2m(samp_Rs)/rvs.AU2m(samp_as))
    v = np.percentile(samp_Teq, (16,50,84))
    Teqs = v[1], v[2]-v[1], v[1]-v[0]

    # compute insolation
    samp_F = 1367. * samp_Rs**2 * (samp_Teff/5778.)**4 / samp_as**2
    v = np.percentile(samp_F, (16,50,84))
    Fs = v[1], v[2]-v[1], v[1]-v[0]
    
    return rps, smas, Teqs, Fs



def Gamma_CDF_func(x, k, theta):
    '''
    k = shape parameter (sometimes called a)
    l = location parameter
    theta = scale parameter (related to the rate b=1/theta)
    '''
    return gamma.cdf(x, k, loc=1., scale=theta)


def Skewnorm_CDF_func(x, a, mu, sig):
    '''
    a = skewness (gaussian if a==0)
    mu = mean of gaussian
    sig = std dev of gaussian
    '''
    return skewnorm.cdf(x, a, loc=mu, scale=sig)


def get_samples_from_percentiles(p16, med, p84, Nsamp=1e3, add_p5_p95=True, pltt=False):
    '''Given the 16, 50, and 84 percentiles of a parameter's distribution,
    fit a Skew normal CDF and sample it.'''
    assert p16 < med
    assert med < p84
    # add approximate percentiles to help with fitting the wings
    # otherwise the resulting fitting distritubions tend to
    if add_p5_p95:
        p5_approx  = med-2*(med-p16)
        p95_approx = med+2*(p84-med)
        xin = [p5_approx,p16,med,p84,p95_approx]
        yin = [.05,.16,.5,.84,.95]
    else:
        xin, yin = [p16,med,p84], [.16,.5,.84]
        
    # make initial parameter guess
    a, mu, sig = (p16-p84)/med, med, np.mean([p16,p84])
    p0 = a,mu,sig
    popt,pcov = curve_fit(Skewnorm_CDF_func, xin, yin, p0=p0,
                          sigma=np.repeat(.01,len(yin)), absolute_sigma=False)

    # sample the fitted pdf
    samples = skewnorm.rvs(*popt, size=int(Nsamp))

    # plot distribution if desired
    if pltt:
        plt.hist(samples, bins=30, normed=True, label='Sampled parameter posterior')
        plt.plot(np.sort(samples), skewnorm.pdf(np.sort(samples), *popt),
                 '-', label='Skew-normal fit: a=%.3f, m=%.3f, s=%.3f'%tuple(popt))
        plt.xlabel('Parameter values'), plt.legend(loc='upper right')
        plt.show()
    
    return p0, popt, samples
