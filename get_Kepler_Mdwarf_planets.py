from imports import *
from scipy.stats import gamma, skewnorm
from scipy.optimize import curve_fit
from priors import get_results
import mwdust


global KepMdwarffile, G, data_spanC
KepMdwarffile = '../GAIAMdwarfs/input_data/Keplertargets/KepMdwarfsv11_archiveplanets.csv'
G = 6.67408e-11

# get stellar completeness parameters
d = np.genfromtxt('TPSfiles/nph-nstedAPI_clean.txt', skip_header=208, delimiter=',',
                  usecols=(0,49,64,65,66,67,68,69,70,71,72,73,74,75,76,77))
kepidC,data_spanC,cdpp1d5,cdpp2,cdpp2d5,cdpp3,cdpp3d5,cdpp4d5,cdpp5,cdpp6,cdpp7d5,cdpp9,cdpp10d5,cdpp12,cdpp12d5,cdpp15 = d.T
transit_durs = np.array([1.5,2,2.5,3,3.5,4.5,5,6,7.5,9,10.5,12,12.5,15])
cdpps = np.array([cdpp1d5,cdpp2,cdpp2d5,cdpp3,cdpp3d5,cdpp4d5,cdpp5,cdpp6,cdpp7d5,cdpp9,cdpp10d5,cdpp12,cdpp12d5,cdpp15]).T


def write_Kepler_Mdwarf_GAIAparams(kepids):
    '''Use Megan Bedell's cross-match between GAIA DR2 and the Kepler catalog 
    to retrieve stellar parameters for Kepler M dwarfs of interest.
    THIS SHOULD ONLY BE USED ONCE.'''

    # get cross-matched data
    fs = np.array(glob.glob('../GAIAMdwarfs/input_data/Keplertargets/kepler_dr2_*.fits'))
    N = kepids.size
    ras, decs = np.zeros(N), np.zeros(N)
    GBPmag, e_GBPmag = np.zeros(N), np.zeros(N)
    GRPmag, e_GRPmag = np.zeros(N), np.zeros(N)
    Kepmag = np.zeros(N)
    Jmag, Hmag, Kmag = np.zeros(N), np.zeros(N), np.zeros(N)
    parallax_mas, e_parallax = np.zeros(N), np.zeros(N)
    dist_pc, ehi_dist, elo_dist = np.zeros(N), np.zeros(N)
    mu, ehi_mu, elo_mu = np.zeros(N), np.zeros(N)
    AK, e_AK = np.zeros(N), np.zeros(N)
    MK, ehi_MK, elo_MK = np.zeros(N), np.zeros(N)
    Rs_RSun, ehi_Rs, elo_Rs = np.zeros(N), np.zeros(N)
    Teff_K, ehi_Teff, elo_Teff = np.zeros(N), np.zeros(N)
    Ms_MSun, ehi_Ms, elo_Ms = np.zeros(N), np.zeros(N)
    logg_dex, ehi_logg, elo_logg = np.zeros(N), np.zeros(N)
    for i in range(N):

        for j in range(fs.size):
            
            hdu = fits.open(fs[i])[1]
            g = hdu.data['kepid'] == kepids[i]
            if g.sum() == 1:

                ras[i] = hdu.data['ra'][g]
                decs[i] = hdu.data['dec'][g]
                GBPmag[i] = hdu.data['phot_bp_mean_mag'][g]
                FBP = hdu.data['phot_bp_mean_flux'][g]
                eFBP = hdu.data['phot_bp_mean_flux_error'][g]
                e_GBPmag[i] = -2.5*np.log10(FBP / (FBP+eFBP))
                GRPmag[i] = hdu.data['phot_rp_mean_mag'][g]
                FRP = hdu.data['phot_rp_mean_flux'][g]
                eFRP = hdu.data['phot_rp_mean_flux_error'][g]
                e_GRPmag[i] = -2.5*np.log10(FRP / (FRP+eFRP))
                Kepmag[i] = hdu.data['kepmag'][g]
                Jmag[i] = hdu.data['jmag'][g]
                Hmag[i] = hdu.data['hmag'][g]
                Kmag[i] = hdu.data['kmag'][g]
                parallax_mas[i] = hdu.data['parallax'][g] + .029
                e_parallax[i] = hdu.data['parallax_error'][g]

                Nsamp = 1000                 
                samp_GBP = np.random.randn(Nsamp)*e_GBPmag[i] + GBPmag[i]
                samp_GRP = np.random.randn(Nsamp)*e_GRPmag[i] + GRPmag[i]

                # get 2MASS photometric uncertainies
                e_Jmag[i], e_Hmag[i], e_Kmag[i] = _get_2MASS_Kep(ras[-1:], decs[-1:],
                                                                 Jmag[-1:], Hmag[-1:], Kmag[-1:])
                samp_J = np.random.randn(Nsamp)*e_Jmag[i] + Jmag[i]
                samp_H = np.random.randn(Nsamp)*e_Hmag[i] + Hmag[i]
                samp_K = np.random.randn(Nsamp)*e_Kmag[i] + Kmag[i]

                # get distance posteriors from Bailor-Jones
                try:
                    fname='../GAIAMdwarfs/Gaia-DR2-distances_custom/DistancePosteriors/KepID_%i.csv'%(prefix,
                                                                                                      kepids[i])
                    x_dist, pdf_dist = np.loadtxt(fname, delimiter=',', skiprows=1,
                                                  usecols=(1,2)).T
                    samp_dist = np.random.choice(x_dist, Nsamp, p=pdf_dist/pdf_dist.sum())
                    dist_pc[i], ehi_dist[i], elo_dist[i] = get_results(samp_dist.reshape(Nsamp,1))
                except IOError:
                    raise ValueError('Need to compute the distance posterior for KepID_%i (see get_gaia_2MASS.save_posteriors())'%kepids[i])                    
                
                # compute stellar parameters
                samp_mu = 5*np.log10(samp_dist) - 5
                mu[i], ehi_mu[i], elo_mu[i] = get_results(samp_mu.reshape(Nsamp,1))
                l, b = hdu.data['l'][g], hdu.data['b'][g]
                AK[i], e_AK[i] = _compute_AK_mwdust(l, b, dist_pc[i], ehi_dist[i])
                samp_AK = np.random.randn(Nsamp)*e_AK[i] + AK[i]
                samp_MK = samp_K + samp_mu + samp_AK
                MK[i], ehi_MK[i], elo_MK[i] = get_results(samp_MK.reshape(Nsamp,1))
                samp_Rs = _sample_Rs_from_MK(samp_MK)
                Rs_RSun[i], ehi_Rs[i], elo_Rs[i] = get_results(samp_Rs.reshape(Nsamp,1))
                samp_Teff = _sample_Teff_from_colors(samp_GBP, samp_GRP, samp_J, samp_H)
                Teff_K[i], ehi_Teff[i], elo_Teff[i] = get_results(samp_Teff.reshape(Nsamp,1))
                samp_Ms = _sample_Ms_from_MK(samp_MK)
                Ms_MSun[i], ehi_Ms[i], elo_Ms[i] = get_results(samp_Ms.reshape(Nsamp,1))
                samp_logg = _sample_logg(samp_Ms, samp_Rs)
                logg_dex[i], ehi_logg[i], elo_logg[i] = get_results(samp_logg.reshape(Nsamp,1))

    # save to file
    hdr = 'KepID,ra_deg,dec_deg,GBPmag,e_GBPmag,GRPmag,e_GRPmag,Kepmag,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag,parallax_mas,e_parallax,dist_pc,ehi_dist,elo_dist,mu,ehi_mu,elo_mu,AK,e_AK,MK,ehi_MK,elo_MK,Rs_RSun,ehi_Rs,elo_Rs,Teff_K,ehi_Teff,elo_Teff,Ms_MSun,ehi_Ms,elo_Ms,logg_dex,ehi_logg,elo_logg'
    outarr = np.array([kepids,ras,decs,GBPmag,e_GBPmag,GRPmag,e_GRPmag,Kepmag,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag,parallax_mas,e_parallax,dist_pc,ehi_dist,elo_dist,mu,ehi_mu,elo_mu,AK,e_AK,MK,ehi_MK,elo_MK,Rs_RSun,ehi_Rs,elo_Rs,Teff_K,ehi_Teff,elo_Teff,Ms_MSun,ehi_Ms,elo_Ms,logg_dex,ehi_logg,elo_logg])
    np.savetxt(KepMdwarffile, outarr.T, delimiter=',', fmt='%.8e')
    return outarr


def _get_2MASS_Kep(ras_deg, decs_deg, Jmags, Hmags, Kmags,
                   radius_deg=.017, phot_rtol=.02):
    '''Match Kepler stars with GAIA data to the 2MASS point-source catlog to
    retrieve photometric uncertainties.'''
    # get 2MASS data for Kepler stars
    # https://irsa.ipac.caltech.edu/applications/Gator/
    d = np.load('input_data/Keplertargets/fp_2mass.fp_psc12298.npy')
    inds = np.array([0,1,3,5,6,8,9,11])
    ras2M, decs2M, J2M, eJ2M, H2M, eH2M, K2M, eK2M = d[:,inds].T

    # match each star individually
    Nstars = ras_deg.size
    e_Jmags, e_Hmags, e_Kmags = np.zeros(Nstars), np.zeros(Nstars), \
                                np.zeros(Nstars)
    print 'Getting 2MASS photometry...'
    for i in range(Nstars):

        if i % 1e2 == 0:
            print float(i) / Nstars

        # get matching photometry between Kepler-GAIA and 2MASS
        g = (ras2M >= ras_deg[i] - radius_deg) & \
            (ras2M <= ras_deg[i] + radius_deg) & \
            (decs2M >= decs_deg[i] - radius_deg) & \
            (decs2M <= decs_deg[i] + radius_deg) & \
            np.isclose(J2M, Jmags[i], rtol=phot_rtol) & \
            np.isclose(H2M, Hmags[i], rtol=phot_rtol) & \
            np.isclose(K2M, Kmags[i], rtol=phot_rtol)

        if g.sum() > 0:
            g2 = (abs(J2M[g]-Jmags[i]) == np.min(abs(J2M[g]-Jmags[i]))) & \
                 (abs(K2M[g]-Kmags[i]) == np.min(abs(K2M[g]-Kmags[i])))
            e_Jmags[i] = eJ2M[g][g2][0]
            e_Hmags[i] = eH2M[g][g2][0]
            e_Kmags[i] = eK2M[g][g2][0]

        else:
            e_Jmags[i], e_Hmags[i], e_Kmags[i] = np.repeat(np.nan, 3)

    return e_Jmags, e_Hmags, e_Kmags



def _compute_AK_mwdust(ls, bs, dist, edist, eAK_frac=.3):
    '''Using the EB-V map from 2014MNRAS.443.2907S and the extinction vector
    RK = 0.31 from Schlafly and Finkbeiner 2011 (ApJ 737, 103)'''
    dustmap = mwdust.Combined15(filter='2MASS Ks')
    dist_kpc, edist_kpc = np.ascontiguousarray(dist)*1e-3, \
                          np.ascontiguousarray(edist)*1e-3
    ls, bs = np.ascontiguousarray(ls), np.ascontiguousarray(bs)
    AK, eAK = np.zeros(ls.size), np.zeros(ls.size)
    for i in range(ls.size):
        v = dustmap(ls[i], bs[i],
                    np.array([dist_kpc[i], dist_kpc[i]+edist_kpc[i]]))
        AK[i], eAK[i] = v[0], np.sqrt(abs(np.diff(v))**2 + (eAK_frac*v[0])**2)
    return AK, eAK



def _sample_Rs_from_MK(samp_MK):
    '''Use relation from Mann+2015 (table 1)'''
    a, b, c, Rs_sigma_frac = 1.9515, -.3520, .01680, .0289
    p = np.poly1d((c,b,a))
    samp_MK_tmp = np.copy(samp_MK)
    samp_MK_tmp[(samp_MK<=4.6) | (samp_MK>=9.8)] = np.nan
    samp_Rs = p(samp_MK_tmp)
    samp_Rs += np.random.normal(0, samp_Rs*Rs_sigma_frac, samp_MK.size)
    return samp_Rs


def _sample_Teff_from_colors(samp_GBPmag, samp_GRPmag, samp_Jmag, samp_Hmag,
                             Teff_scatter=49):
    '''Use the relation from Mann+2015 (table 2)'''
    a, b, c, d, e, f, g = 3.172, -2.475, 1.082, -.2231, .01738, .08776, -.04355
    pG = np.poly1d((e,d,c,b,a))
    p2 = np.poly1d((g,f,0))
    samp_Teff = 35e2 * (pG(samp_GBPmag-samp_GRPmag) + p2(samp_Jmag-samp_Hmag)) \
                + np.random.normal(0, Teff_scatter, samp_Jmag.size)
    return samp_Teff


def _sample_Ms_from_MK(samp_MK):
    '''Use relation from Benedict+2016'''
    c0 = np.random.normal(.2311, 4e-4, samp_MK.size)
    c1 = np.random.normal(-.1352, 7e-4, samp_MK.size)
    c2 = np.random.normal(.04, 5e-4, samp_MK.size)
    c3 = np.random.normal(.0038, 2e-4, samp_MK.size)
    c4 = np.random.normal(-.0032, 1e-4, samp_MK.size)
    samp_MK_tmp = np.copy(samp_MK)
    samp_MK_tmp[(samp_MK<=4.6) | (samp_MK>10)] = np.nan
    samp_MK_tmp[samp_MK>=10] = np.nan
    dMK = samp_MK_tmp - 7.5
    samp_Ms = c0 + c1*dMK + c2*dMK**2 + c3*dMK**3 + c4*dMK**4
    return samp_Ms


def _sample_logg(samp_Ms, samp_Rs):
    G = 6.67e-11
    samp_logg = np.log10(G*rvs.Msun2kg(samp_Ms)*1e2 / rvs.Rsun2m(samp_Rs)**2)
    return samp_logg



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

        print float(i) / Nplanets
        g = np.in1d(KepID, kepid[i])

        # stellar parameters (both pre (1) and post-GAIA (2))
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

        # pre-GAIA
        self.Rss1[i] = koi_srad[i]
        self.ehi_Rss1[i] = koi_srad_err1[i] if koi_srad_err1[i] > 0 else koi_srad[i]*.07
        self.elo_Rss1[i] = abs(koi_srad_err2[i]) if koi_srad_err2[i] < 0 else koi_srad[i]*.08
        self.Teffs1[i] = koi_steff[i]
        self.ehi_Teffs1[i] = koi_steff_err1[i] if koi_steff_err1[i] > 0 else koi_steff[i]*.02
        self.elo_Teffs1[i] = abs(koi_steff_err2[i]) if koi_steff_err2[i] < 0 else koi_steff[i]*.02
        self.loggs1[i] = koi_slogg[i]
        self.ehi_loggs1[i] = koi_slogg_err1[i] if koi_slogg_err1[i] > 0 else koi_slogg[i]*.009
        self.elo_loggs1[i] = abs(koi_slogg_err2[i]) if koi_slogg_err2[i] < 0 else koi_slogg[i]*.006
        _,_,samp_Rs = get_samples_from_percentiles(self.Rss1[i], self.ehi_Rss1[i],
                                                   self.elo_Rss1[i], Nsamp=1e3)
        _,_,samp_logg = get_samples_from_percentiles(self.loggs1[i], self.ehi_loggs1[i],
                                                     self.elo_loggs1[i], Nsamp=1e3)
        samp_Ms = rvs.kg2Msun(10**samp_logg * rvs.Rsun2m(samp_Rs)**2 * 1e-2 / G)
        v = np.percentile(samp_Ms, (16,50,84))
        self.Mss1[i], self.ehi_Mss1[i], self.elo_Mss1[i] = v[1], v[2]-v[1], v[1]-v[0] 
        self.FeHs1[i] = koi_smet[i]
        self.ehi_FeHs1[i] = koi_smet_err1[i]
        self.elo_FeHs1[i] = abs(koi_smet_err1[i])

        # post-GAIA
        self.Rss2[i] = Rs_RSun[g] if g.sum() == 1 else np.nan
        self.ehi_Rss2[i] = ehi_Rs[g] if g.sum() == 1 else np.nan
        self.elo_Rss2[i] = elo_Rs[g] if g.sum() == 1 else np.nan
        self.Teffs2[i] = Teff_K[g] if g.sum() == 1 else np.nan
        self.ehi_Teffs2[i] = ehi_Teff[g] if g.sum() == 1 else np.nan
        self.elo_Teffs2[i] = elo_Teff[g] if g.sum() == 1 else np.nan
        self.Mss2[i] = Ms_MSun[g] if g.sum() == 1 else np.nan
        self.ehi_Mss2[i] = ehi_Ms[g] if g.sum() == 1 else np.nan
        self.elo_Mss2[i] = elo_Ms[g] if g.sum() == 1 else np.nan
        self.loggs2[i] = logg_dex[g] if g.sum() == 1 else np.nan
        self.ehi_loggs2[i] = ehi_logg[g] if g.sum() == 1 else np.nan
        self.elo_loggs2[i] = elo_logg[g] if g.sum() == 1 else np.nan
        
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

        # completeness parameters
        self.CDPPs[i] = get_fitted_cdpp(self.KepIDs[i], self.Ds[i])
        self.data_spans[i] = data_spanC[kepidC == self.KepIDs[i]]
        self.SNRtransits[i] = self.Zs[i] / self.CDPPs[i] * np.sqrt(get_Ntransits(self.KepIDs[i],
                                                                                 self.Ps[i]))
        
        # computed planet parameters
        if self.isMdwarf[i]:
            rps1, smas1, Teqs1, Fs1 = sample_planet_params(self, i, postGAIA=False)
            self.rps1[i], self.ehi_rps1[i], self.elo_rps1[i] = rps1
            self.smas1[i], self.ehi_smas1[i], self.elo_smas1[i] = smas1
            self.Teqs1[i], self.ehi_Teqs1[i], self.elo_Teqs1[i] = Teqs1
            self.Fs1[i], self.ehi_Fs1[i], self.elo_Fs1[i] = Fs1
            rps2, smas2, Teqs2, Fs2 = sample_planet_params(self, i, postGAIA=True)
            self.rps2[i], self.ehi_rps2[i], self.elo_rps2[i] = rps2
            self.smas2[i], self.ehi_smas2[i], self.elo_smas2[i] = smas2
            self.Teqs2[i], self.ehi_Teqs2[i], self.elo_Teqs2[i] = Teqs2
            self.Fs2[i], self.ehi_Fs2[i], self.elo_Fs2[i] = Fs2

    # save Kepler M dwarf planet population
    self._pickleobject()



    
class KepConfirmedMdwarfPlanets:

    def __init__(self, fname, Nplanets):
        self.fname_out = fname
        self.Nplanets = int(Nplanets)


    def _initialize_arrays(self):
        N = self.Nplanets

        # stellar parameters (both pre (1) and post-GAIA (2))
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
        self.Rss1, self.ehi_Rss1, self.elo_Rss1=np.zeros(N),np.zeros(N),np.zeros(N)
        self.Mss1, self.ehi_Mss1, self.elo_Mss1=np.zeros(N),np.zeros(N),np.zeros(N)
        self.Teffs1, self.ehi_Teffs1, self.elo_Teffs1 = np.zeros(N), np.zeros(N), \
                                                        np.zeros(N)
        self.loggs1, self.ehi_loggs1, self.elo_loggs1 = np.zeros(N), np.zeros(N), \
                                                        np.zeros(N)
        self.FeHs1, self.ehi_FeHs1, self.elo_FeHs1=np.zeros(N),np.zeros(N),np.zeros(N)
        self.Rss2, self.ehi_Rss2, self.elo_Rss2=np.zeros(N),np.zeros(N),np.zeros(N)
        self.Mss2, self.ehi_Mss2, self.elo_Mss2=np.zeros(N),np.zeros(N),np.zeros(N)
        self.Teffs2, self.ehi_Teffs2, self.elo_Teffs2 = np.zeros(N), np.zeros(N), \
                                                        np.zeros(N)
        self.loggs2, self.ehi_loggs2, self.elo_loggs2 = np.zeros(N), np.zeros(N), \
                                                        np.zeros(N)
        
        # planet parameters (both pre (1) and post-GAIA (2))
        self.Ps, self.e_Ps = np.zeros(N), np.zeros(N)
        self.T0s, self.e_T0s = np.zeros(N), np.zeros(N)
        self.Ds, self.e_Ds = np.zeros(N), np.zeros(N)
        self.Zs, self.e_Zs = np.zeros(N), np.zeros(N)
        self.aRs, self.e_aRs = np.zeros(N), np.zeros(N)
        self.rpRs, self.ehi_rpRs, self.elo_rpRs = np.zeros(N), np.zeros(N), \
                                                  np.zeros(N)
        self.bs, self.ehi_bs, self.elo_bs=np.zeros(N),np.zeros(N),np.zeros(N)
        self.rps1  = np.repeat(np.nan,N)
        self.ehi_rps1  = np.repeat(np.nan,N)
        self.elo_rps1  = np.repeat(np.nan,N)
        self.smas1  = np.repeat(np.nan,N)
        self.ehi_smas1  = np.repeat(np.nan,N)
        self.elo_smas1  = np.repeat(np.nan,N)
        self.Teqs1  = np.repeat(np.nan,N)
        self.ehi_Teqs1  = np.repeat(np.nan,N)
        self.elo_Teqs1  = np.repeat(np.nan,N)
        self.Fs1  = np.repeat(np.nan,N)
        self.ehi_Fs1  = np.repeat(np.nan,N)
        self.elo_Fs1  = np.repeat(np.nan,N)
        self.rps2 = np.repeat(np.nan,N)
        self.ehi_rps2 = np.repeat(np.nan,N)
        self.elo_rps2 = np.repeat(np.nan,N)
        self.smas2 = np.repeat(np.nan,N)
        self.ehi_smas2 = np.repeat(np.nan,N)
        self.elo_smas2 = np.repeat(np.nan,N)
        self.Teqs2 = np.repeat(np.nan,N)
        self.ehi_Teqs2 = np.repeat(np.nan,N)
        self.elo_Teqs2 = np.repeat(np.nan,N)
        self.Fs2 = np.repeat(np.nan,N)
        self.ehi_Fs2 = np.repeat(np.nan,N)
        self.elo_Fs2 = np.repeat(np.nan,N)

        # sensitivity calculations
        self.CDPPs = np.zeros(N)
        self.data_spans = np.zeros(N)
        self.SNRtransits = np.zeros(N)
        

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


def sample_planet_params(self, index, postGAIA=True):
    '''sample distribution of planet parameters from observables and stellar pdfs'''
    # get stellar parameters PDFs either from derived from GAIA distances
    # or from original Kepler parameters (approximate distributions as skewnormal)
    g = int(index)
    if postGAIA:
        path = '../GAIAMdwarfs/Gaia-DR2-distances_custom/DistancePosteriors/'
        samp_Rs,samp_Teff,samp_Ms = np.loadtxt('%s/KepID_allpost_%i'%(path,self.KepIDs[g]),
                                               delimiter=',', usecols=(9,10,11)).T
        samp_Rs = resample_PDF(samp_Rs[np.isfinite(samp_Rs)], samp_Rs.size, sig=1e-3)
        samp_Teff = resample_PDF(samp_Teff[np.isfinite(samp_Teff)], samp_Teff.size, sig=5)
        samp_Ms = resample_PDF(samp_Ms[np.isfinite(samp_Ms)], samp_Ms.size, sig=1e-3)
    else:
        _,_,samp_Rs = get_samples_from_percentiles(self.Rss1[g], self.ehi_Rss1[g],
                                                   self.elo_Rss1[g], Nsamp=1e3)
        _,_,samp_Teff = get_samples_from_percentiles(self.Teffs1[g], self.ehi_Teffs1[g],
                                                     self.elo_Teffs1[g], Nsamp=1e3)
        _,_,samp_Ms = get_samples_from_percentiles(self.Mss1[g], self.ehi_Mss1[g],
                                                   self.elo_Mss1[g], Nsamp=1e3)
        
    # sample rp/Rs distribution from point estimates
    _,_,samp_rpRs = get_samples_from_percentiles(self.rpRs[g], self.ehi_rpRs[g],
                                                 self.elo_rpRs[g],
                                                 Nsamp=samp_Rs.size)
    
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
    samp_F = samp_Rs**2 * (samp_Teff/5778.)**4 / samp_as**2
    v = np.percentile(samp_F, (16,50,84))
    Fs = v[1], v[2]-v[1], v[1]-v[0]
    
    return rps, smas, Teqs, Fs



def get_fitted_cdpp(KepID, duration):
    # get cdpps for this star
    g = kepidC == KepID
    if g.sum() == 0:
        return np.nan
    else:
        cdpp_arr = cdpps[g].reshape(transit_durs.size)
	if np.any(np.isnan(cdpp_arr)): return np.nan

    # fit cubic function to cdpp(t)
    func = np.poly1d(np.polyfit(transit_durs, cdpp_arr, 3))
    ##plt.plot(transit_durs, cdpp_arr, '.', transit_durs, func(transit_durs), '-')
    ##plt.show()
    return func(duration)


def get_Ntransits(KepID, P):
    # get data span for this star
    g = kepidC == KepID
    if g.sum() == 0:
	return np.nan
    else:
	data_span = float(data_spanC[g])

    return int(np.round(data_span / P))
    

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


def get_samples_from_percentiles(val, ehi, elo, Nsamp=1e3, add_p5_p95=True, pltt=False):
    '''Given the 16, 50, and 84 percentiles of a parameter's distribution,
    fit a Skew normal CDF and sample it.'''
    # get percentiles
    p16, med, p84 = float(val-elo), float(val), float(val+ehi)    
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
