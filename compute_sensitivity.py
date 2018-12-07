from imports import *
from get_Kepler_Mdwarf_planets import *
from scipy.ndimage.filters import gaussian_filter


global KepMdwarffile, Pbins, smabins, Fbins, Teqbins, rpbins, transit_durs, \
    cdppsC, suffix
KepMdwarffile = '../GAIAMdwarfs/input_data/Keplertargets/KepMdwarfsv11.csv'
xlen = 40
Pbins = np.logspace(np.log10(.5), 2, xlen)
smabins = np.logspace(-2, np.log10(.35), xlen)
Fbins = np.logspace(np.log10(.5), 2, xlen)
Teqbins = np.logspace(np.log10(240), np.log10(12e2), xlen)
rpbins = np.logspace(np.log10(.5), 1, 31)
suffix = 'allM'


# get stellar completeness parameters
d = np.genfromtxt('TPSfiles/nph-nstedAPI_clean.txt', skip_header=208,
                  delimiter=',',
                  usecols=(0,49,64,65,66,67,68,69,70,71,72,73,74,75,76,77))
kepidC,data_spanC,cdpp1d5,cdpp2,cdpp2d5,cdpp3,cdpp3d5,cdpp4d5,cdpp5,cdpp6,cdpp7d5,cdpp9,cdpp10d5,cdpp12,cdpp12d5,cdpp15 = d.T
transit_durs = np.array([1.5,2,2.5,3,3.5,4.5,5,6,7.5,9,10.5,12,12.5,15])
cdppsC = np.array([cdpp1d5,cdpp2,cdpp2d5,cdpp3,cdpp3d5,cdpp4d5,cdpp5,cdpp6,
                   cdpp7d5,cdpp9,cdpp10d5,cdpp12,cdpp12d5,cdpp15]).T


# TEMP: need to do MC sampling here over errorbars
def compute_Ndet_maps(self, sigs=0, condition=None, Nsamp=1e3):
    '''Compute the map of planet detections over some various distance 
    proxies and planet radius. This will be used to compute the occurrence rate 
    in each parameter space which can then be marginalized to get the 1d 
    occurrence rate of planet sizes.'''
    # only take valid M dwarfs and include any other input conditions
    # e.g. self.Mss2 > .6
    g = (self.isMdwarf==1)
    if np.any(condition != None):
        assert condition.size == g.size
        g = g & (condition)

    # do MC sampling of planet parameters
    Ps = self.Ps[g], self.e_Ps[g]
    smas = self.smas2[g], self.ehi_smas2[g], self.elo_smas2[g]
    Fs = self.Fs2[g], self.ehi_Fs2[g], self.elo_Fs2[g]
    Teqs = self.Teqs2[g], self.ehi_Teqs2[g], self.elo_Teqs2[g]
    rps = self.rps2[g], self.ehi_rps2[g], self.elo_rps2[g]
    samp_P,samp_sma,samp_F,samp_Teq,samp_rp = _MC_sample_planets(Ps, smas, Fs,
                                                                 Teqs, rps,
                                                                 Nsamp)

    # compute Ndet maps
    zP,P_edges,rp_edges = np.histogram2d(samp_P, samp_rp, bins=[Pbins,rpbins])
    zP /= float(Nsamp)
    zsma,sma_edges,rp_edges = np.histogram2d(samp_sma, samp_rp,
                                             bins=[smabins,rpbins])
    zsma /= float(Nsamp)
    zF,F_edges,rp_edges = np.histogram2d(self.Fs2[g], self.rps2[g],
                                         bins=[Fbins,rpbins])
    zF /= float(Nsamp)
    zTeq,Teq_edges,Teq_edges = np.histogram2d(self.Teqs2[g], self.rps2[g],
                                              bins=[Teqbins,rpbins])
    zTeq /= float(Nsamp)
    
    # return Ndet maps
    if type(sigs) in [float,int]:
        sigs = np.repeat(sigs, 4)
    else:
        assert len(sigs) == 4
        sigs = np.array(sigs)
    NdetP = gaussian_filter(zP, sigs[0])
    Ndetsma = gaussian_filter(zsma, sigs[1])
    NdetF = gaussian_filter(zF, sigs[2])
    NdetTeq = gaussian_filter(zTeq, sigs[3])

    # save to table
    np.save('KeplerMaps/Ndet_maps_%s'%suffix, np.array([NdetP, Ndetsma,
                                                        NdetF, NdetTeq]))

    return NdetP, Ndetsma, NdetF, NdetTeq



def compute_SNR_maps(self, sigs=0):
    '''Compute the SNR maps over various distance proxies and planet 
    radius for every Kepler M dwarf, not just those with confirmed planets.'''
    # get stellar DE curves
    fs = np.array(glob.glob('TPSfiles/DE_KepID_*'))
    Nstars = fs.size

    # get smoothing parameters
    if type(sigs) in [float,int]:
        sigs = np.repeat(sigs, 4)
    else:
        assert len(sigs) == 4
        sigs = np.array(sigs)
    
    # get stellar parameters from GAIA
    KepIDs,Rs_gaia,Teff_gaia,Ms_gaia = np.loadtxt(KepMdwarffile, delimiter=',',
                                                  usecols=(0,27,30,33)).T

    # compute SNR map
    xlen, ylen = Pbins.size-1, rpbins.size-1
    SNRP_maps = np.zeros((Nstars, xlen, ylen))
    SNRsma_maps = np.zeros((Nstars, xlen, ylen))
    SNRF_maps = np.zeros((Nstars, xlen, ylen))
    SNRTeq_maps = np.zeros((Nstars, xlen, ylen))
    KepIDsout = np.zeros(Nstars)
    for i in range(Nstars):

        print float(i)/Nstars
        KepIDsout[i] = int(fs[i].split('_')[-1].split('.')[0])
        Rs = float(Rs_gaia[KepIDs == KepIDsout[i]])
        Teff = float(Teff_gaia[KepIDs == KepIDsout[i]])
        Ms = float(Ms_gaia[KepIDs == KepIDsout[i]])
       
	# get cdpp array for this star for interpolation along durations
        g = kepidC == KepIDsout[i]
        if g.sum() == 1:
            cdpp_arr_star = cdppsC[g].reshape(transit_durs.size)
            data_span_star = float(data_spanC[g])
        else:
            cdpp_arr_star = np.repeat(np.nan, transit_durs.size)
            data_span_star = np.nan

        # fit cubic function to cdpp(duration)
        cdpp_func = np.poly1d(np.polyfit(transit_durs, cdpp_arr_star, 3))

        
        for j in range(xlen):
            for k in range(ylen):

                rp = np.mean(rpbins[k:k+2]) 
                Z = (rvs.Rearth2m(rp) / rvs.Rsun2m(Rs))**2 * 1e6
                P = np.mean(Pbins[j:j+2])
                D = rvs.transit_width(P, Ms, Rs, rp, 0)
                CDPP = cdpp_func(D)
                Ntransit = int(np.round(data_span_star / P))
                SNRP_maps[i,j,k] = Z / CDPP * np.sqrt(Ntransit)
                
                sma = np.mean(smabins[j:j+2])
                P = rvs.period_sma(sma, Ms, 0)
                Ntransit = int(np.round(data_span_star / P))
                SNRsma_maps[i,j,k] = Z / CDPP * np.sqrt(Ntransit)
                
                F = np.mean(Fbins[j:j+2])
                sma = np.sqrt(Rs**2 * (Teff/5778.)**4 / F)
                P = rvs.period_sma(sma, Ms, 0)
                Ntransit = int(np.round(data_span_star / P))
                SNRF_maps[i,j,k] = Z / CDPP * np.sqrt(Ntransit)
                
                Teq = np.mean(Teqbins[j:j+2])
                sma = .5*rvs.m2AU(rvs.Rsun2m(Rs)) * (Teff/Teq)**2
                P = rvs.period_sma(sma, Ms, 0)
                Ntransit = int(np.round(data_span_star / P))
                SNRTeq_maps[i,j,k] = Z / CDPP * np.sqrt(Ntransit)
                                
        # smooth maps
        SNRP_maps[i] = gaussian_filter(SNRP_maps[i], sigs[0])
        SNRsma_maps[i] = gaussian_filter(SNRsma_maps[i], sigs[1])
        SNRF_maps[i] = gaussian_filter(SNRF_maps[i], sigs[2])
        SNRTeq_maps[i] = gaussian_filter(SNRTeq_maps[i], sigs[3])

    # save to table
    np.save('KeplerMaps/KepIDs_%s'%suffix, KepIDsout)
    np.save('KeplerMaps/SNR_maps_%s'%suffix, np.array([SNRP_maps, SNRsma_maps,
                                                       SNRF_maps, SNRTeq_maps]))

    return KepIDsout, SNRP_maps, SNRsma_maps, SNRF_maps, SNRTeq_maps 



def compute_MES_maps(SNRP_maps, SNRsma_maps, SNRF_maps, SNRTeq_maps, sigs=0):
    '''Compute the MES maps from the SNR maps over various distance proxies.'''
    # get smoothing parameters
    if type(sigs) in [float,int]:
        sigs = np.repeat(sigs, 4)
    else:
        assert len(sigs) == 4
        sigs = np.array(sigs)

    # get MES maps
    assert len(SNRP_maps.shape) == 3
    assert len(SNRsma_maps.shape) == 3
    assert len(SNRF_maps.shape) == 3
    assert len(SNRTeq_maps.shape) == 3
    Nstars = SNRP_maps.shape[0] 
    MESP_maps = SNR2MES(SNRP_maps)        
    MESsma_maps = SNR2MES(SNRsma_maps)        
    MESF_maps = SNR2MES(SNRF_maps)        
    MESTeq_maps = SNR2MES(SNRTeq_maps)        

    # smooth maps
    for i in range(Nstars):
        MESP_maps[i] = gaussian_filter(MESP_maps[i], sigs[0])
        MESsma_maps[i] = gaussian_filter(MESsma_maps[i], sigs[1])
        MESF_maps[i] = gaussian_filter(MESF_maps[i], sigs[2])
        MESTeq_maps[i] = gaussian_filter(MESTeq_maps[i], sigs[3])

    # save to table
    np.save('KeplerMaps/MES_maps_%s'%suffix, np.array([MESP_maps, MESsma_maps,
                                                       MESF_maps, MESTeq_maps]))
        
    return MESP_maps, MESsma_maps, MESF_maps, MESTeq_maps



def compute_sens_maps(KepIDs, MESP_maps, MESsma_maps, MESF_maps, MESTeq_maps,
                      sigs=0):
    '''Compute the sensitivity maps over various distance proxies and planet 
    radius for every Kepler M dwarf based on its MES map.'''
    # get smoothing parameters
    if type(sigs) in [float,int]:
        sigs = np.repeat(sigs, 8)
    else:
        assert len(sigs) == 8
        sigs = np.array(sigs)
    
    assert len(MESP_maps.shape) == 3
    assert len(MESsma_maps.shape) == 3
    assert len(MESF_maps.shape) == 3
    assert len(MESTeq_maps.shape) == 3
    Nstars = MESP_maps.shape[0]
    assert Nstars == KepIDs.size

    # get sens maps for each star
    sensP_maps = np.zeros_like(MESP_maps)
    sensSmearP_maps = np.zeros_like(MESP_maps)
    senssma_maps = np.zeros_like(MESsma_maps)
    sensSmearsma_maps = np.zeros_like(MESsma_maps)
    sensF_maps = np.zeros_like(MESF_maps)
    sensSmearF_maps = np.zeros_like(MESF_maps)
    sensTeq_maps = np.zeros_like(MESTeq_maps)
    sensSmearTeq_maps = np.zeros_like(MESTeq_maps)
    for i in range(Nstars):
        mP = _get_one_DE_map(KepIDs[i], MESP_maps[i])
        msma = _get_one_DE_map(KepIDs[i], MESsma_maps[i])
        mF = _get_one_DE_map(KepIDs[i], MESF_maps[i])
        mTeq = _get_one_DE_map(KepIDs[i], MESTeq_maps[i])

        sensP_maps[i], sensSmearP_maps[i] = mP
        senssma_maps[i], sensSmearsma_maps[i] = msma
        sensF_maps[i], sensSmearF_maps[i] = mF
        sensTeq_maps[i], sensSmearTeq_maps[i] = mTeq

        # smooth maps
        sensP_maps[i] = gaussian_filter(sensP_maps[i], sigs[0])
        sensSmearP_maps[i] = gaussian_filter(sensSmearP_maps[i], sigs[1])
        senssma_maps[i] = gaussian_filter(senssma_maps[i], sigs[2])
        sensSmearsma_maps[i] = gaussian_filter(sensSmearsma_maps[i], sigs[3])
        sensF_maps[i] = gaussian_filter(sensF_maps[i], sigs[4])
        sensSmearF_maps[i] = gaussian_filter(sensSmearF_maps[i], sigs[5])
        sensTeq_maps[i] = gaussian_filter(sensTeq_maps[i], sigs[6])
        sensSmearTeq_maps[i] = gaussian_filter(sensSmearTeq_maps[i], sigs[7])

    # save to table
    np.save('KeplerMaps/sens_maps_%s'%suffix, np.array([sensP_maps,
                                                        senssma_maps,
                                                        sensF_maps,
                                                        sensTeq_maps]))
    np.save('KeplerMaps/sensSmear_maps_%s'%suffix,np.array([sensSmearP_maps,
                                                            sensSmearsma_maps,
                                                            sensSmearF_maps,
                                                            sensSmearTeq_maps]))
        
    return sensP_maps, sensSmearP_maps, senssma_maps, sensSmearsma_maps, \
        sensF_maps, sensSmearF_maps, sensTeq_maps, sensSmearTeq_maps 



def compute_transitprob_maps(self, KepIDs, sigs=0, correction_factor=1.08):
    '''Compute the sensitivity maps over various distance proxies and planet 
    radius for every Kepler M dwarf based on its MES map. Correct for 
    eccentricity distribution using the factor from Kipping 2013'''
    # get smoothing parameters
    if type(sigs) in [float,int]:
        sigs = np.repeat(sigs, 4)
    else:
        assert len(sigs) == 4
        sigs = np.array(sigs)

    # get stellar parameters from GAIA
    KepIDs_gaia,Rs_gaia,Teff_gaia,Ms_gaia = np.loadtxt(KepMdwarffile,
                                                       delimiter=',',
                                                       usecols=(0,27,30,33)).T
        
    # compute transit probability maps
    Nstars = KepIDs.size
    xlen, ylen = Pbins.size-1, rpbins.size-1
    probP_maps = np.zeros((Nstars,xlen,ylen))
    probsma_maps = np.zeros((Nstars,xlen,ylen))
    probF_maps = np.zeros((Nstars,xlen,ylen))
    probTeq_maps = np.zeros((Nstars,xlen,ylen))
    for i in range(Nstars):
        
        print float(i)/Nstars
        Rs = float(Rs_gaia[KepIDs_gaia == KepIDs[i]])
        Teff = float(Teff_gaia[KepIDs_gaia == KepIDs[i]])
        Ms = float(Ms_gaia[KepIDs_gaia == KepIDs[i]])
        
        for j in range(xlen):
            for k in range(ylen):

                rp = np.mean(rpbins[k:k+2])
                P = np.mean(Pbins[j:j+2])
                sma = rvs.semimajoraxis(P, Ms, 0)
                probP_maps[i,j,k] = (rvs.Rsun2m(Rs) + rvs.Rearth2m(rp)) / \
                                     rvs.AU2m(sma) * correction_factor
                
                sma = np.mean(smabins[j:j+2])
                probsma_maps[i,j,k] = (rvs.Rsun2m(Rs) + rvs.Rearth2m(rp)) / \
                                      rvs.AU2m(sma) * correction_factor
                
                F = np.mean(Fbins[j:j+2])
                sma = np.sqrt(Rs**2 * (Teff/5778.)**4 / F)
                probF_maps[i,j,k] = (rvs.Rsun2m(Rs) + rvs.Rearth2m(rp)) / \
                                    rvs.AU2m(sma) * correction_factor
                
                Teq = np.mean(Teqbins[j:j+2])
                sma = .5*rvs.m2AU(rvs.Rsun2m(Rs)) * (Teff/Teq)**2
                probTeq_maps[i,j,k] = (rvs.Rsun2m(Rs) + rvs.Rearth2m(rp)) / \
                                      rvs.AU2m(sma) * correction_factor

        # smooth maps
        probP_maps[i] = gaussian_filter(probP_maps[i], sigs[0])
        probsma_maps[i] = gaussian_filter(probsma_maps[i], sigs[1])
        probF_maps[i] = gaussian_filter(probF_maps[i], sigs[2])
        probTeq_maps[i] = gaussian_filter(probTeq_maps[i], sigs[3])

    # save to table
    np.save('KeplerMaps/prob_maps_%s'%suffix, np.array([probP_maps,
                                                        probsma_maps,
                                                        probF_maps,
                                                        probTeq_maps]))
        
    return probP_maps, probsma_maps, probF_maps, probTeq_maps



def compute_completeness_maps(sensP, sensSmearP, senssma, sensSmearsma,
                              sensF, sensSmearF, sensTeq, sensSmearTeq,
                              probP, probsma, probF, probTeq):
    '''See compute_sens_maps and compute_transitprob_maps to get input.'''
    # compute completeness maps
    compP_maps, compSmearP_maps = sensP*probP, sensSmearP*probP
    compsma_maps, compSmearsma_maps = senssma*probsma, sensSmearsma*probsma
    compF_maps, compSmearF_maps = sensF*probF, sensSmearF*probF
    compTeq_maps, compSmearTeq_maps = sensTeq*probTeq, sensSmearTeq*probTeq

    # save to table
    np.save('KeplerMaps/completeness_maps_%s'%suffix, np.array([compP_maps,
                                                                compsma_maps,
                                                                compF_maps,
                                                                compTeq_maps]))
    np.save('KeplerMaps/completenessSmear_maps_%s'%suffix,np.array([compSmearP_maps,
                                                                    compSmearsma_maps,
                                                                    compSmearF_maps,
                                                                    compSmearTeq_maps]))
        
    return compP_maps, compSmearP_maps, compsma_maps, compSmearsma_maps, \
        compF_maps, compSmearF_maps, compTeq_maps, compSmearTeq_maps 



def compute_occurrence_maps(NdetP, Ndetsma, NdetF, NdetTeq,
                            compP, compSmearP, compsma, compSmearsma,
                            compF, compSmearF, compTeq, compSmearTeq,
                            Ndetsig=0):
    '''See compute_Ndet_maps and compute_completeness_maps to get input.'''
    # smooth Ndet maps to help mitigate the small number statistics
    NdetP = gaussian_filter(NdetP, float(Ndetsig))
    Ndetsma = gaussian_filter(Ndetsma, float(Ndetsig))
    NdetF = gaussian_filter(NdetF, float(Ndetsig))
    NdetTeq = gaussian_filter(NdetTeq, float(Ndetsig))

    Nstars = compP.shape[0]
    fP_maps = NdetP / np.mean(compP,0) / Nstars
    fSmearP_maps = NdetP / np.mean(compSmearP,0) / Nstars
    fsma_maps = Ndetsma / np.mean(compsma,0) / Nstars
    fSmearsma_maps = Ndetsma / np.mean(compSmearsma,0) / Nstars
    fF_maps = NdetF / np.mean(compF,0) / Nstars
    fSmearF_maps = NdetF / np.mean(compSmearF,0) / Nstars
    fTeq_maps = NdetTeq / np.mean(compTeq,0) / Nstars
    fSmearTeq_maps = NdetTeq / np.mean(compSmearTeq,0) / Nstars

    # save to table
    np.save('KeplerMaps/occurrence_maps_%s'%suffix, np.array([fP_maps,
                                                              fsma_maps,
                                                              fF_maps,
                                                              fTeq_maps]))
    np.save('KeplerMaps/occurrenceSmear_maps_%s'%suffix, np.array([fSmearP_maps,
                                                                   fSmearsma_maps,
                                                                   fSmearF_maps,
                                                                   fSmearTeq_maps]))
    return fP_maps, fSmearP_maps, fsma_maps, fSmearsma_maps, \
        fF_maps, fSmearF_maps, fTeq_maps, fSmearTeq_maps



def _MC_sample_planets(Ps, smas, Fs, Teqs, rps, Nsamp):
    # get parameters and uncertainties
    P, e_P = Ps
    sma, ehi_sma, elo_sma = smas
    F, ehi_F, elo_F = Fs
    Teq, ehi_Teq, elo_Teq = Teqs
    rp, ehi_rp, elo_rp = rps

    # MC sample each planet
    Nplanets, Nsamp = P.size, int(Nsamp)
    samp_P   = np.zeros(0)
    samp_sma = np.zeros(0)
    samp_F   = np.zeros(0)
    samp_Teq = np.zeros(0)
    samp_rp  = np.zeros(0)
    for i in range(Nplanets):
        print float(i)/Nplanets
        samp_P = np.append(samp_P, np.random.randn(Nsamp)*e_P[i]+P[i])
        samp_sma = np.append(samp_sma, get_samples_from_percentiles(sma[i],
            ehi_sma[i], elo_sma[i], pltt=False, Nsamp=Nsamp, add_p5_p95=1)[2])
        samp_F = np.append(samp_F, get_samples_from_percentiles(F[i],
                ehi_F[i], elo_F[i], pltt=False, Nsamp=Nsamp, add_p5_p95=1)[2])
        samp_Teq = np.append(samp_Teq, get_samples_from_percentiles(Teq[i],
            ehi_Teq[i], elo_Teq[i], pltt=False, Nsamp=Nsamp, add_p5_p95=1)[2])
        samp_rp = np.append(samp_rp, get_samples_from_percentiles(rp[i],
            ehi_rp[i], elo_rp[i], pltt=False, Nsamp=Nsamp, add_p5_p95=1)[2])
        
    return samp_P, samp_sma, samp_F, samp_Teq, samp_rp


def _get_one_DE_map(KepID, MES_map):
    '''Compute the average DE maps given a MES map for a single Kepler star.'''
    # get DE curves for this star
    fname = 'TPSfiles/DE_KepID_%i.npy'%KepID
    if os.path.exists(fname):
        mes_arr, de_arr, smearde_arr = np.load(fname).T
    else:
        return np.zeros_like(MES_map) + np.nan

    # interpolate mes to DE
    fill = (0,1,)
    fintde = interp1d(mes_arr, de_arr, bounds_error=False, fill_value=fill)
    fintsmearde = interp1d(mes_arr, smearde_arr, bounds_error=False,
                           fill_value=fill)
    DE_map = fintde(MES_map)
    SmearDE_map = fintsmearde(MES_map)
    return DE_map, SmearDE_map


def SNR2MES(SNR):
    '''Convert SNR of a transit to the Kepler multi-event statistic using the 
    powerlaw relation derived in map_MES2SNR.py.'''
    A, alpha = 0.99977305, 0.96666667
    MES = A * SNR**alpha
    return MES


def compute_avg_detprob_curve():
    '''Eq 5 in Fulton+2017 -> fraction of stars for which a planet with a 
    given SNR would be detected.'''
    # get all DE curves
    fs = np.array(glob.glob('TPSfiles/DE_KepID_*npy'))
    Nstars = fs.size
    MES, de, smearde = np.load(fs[0]).T
    DEs, SmearDEs = np.zeros((Nstars,de.size)),np.zeros((Nstars,smearde.size))
    for i in range(Nstars):
        _, de, smearde = np.load(fs[i]).T
        DEs[i], SmearDEs[i] = de, smearde

    # compute average
    DEavg = np.mean(DEs, 0)
    SmearDEavg = np.mean(SmearDEs, 0)
    
    return MES, DEs, SmearDEs, DEavg, SmearDEavg


def load_maps(suff):
    NdetP,Ndetsma,NdetF,NdetTeq = np.load('KeplerMaps/Ndet_maps_%s.npy'%suff)
    KepIDs = np.load('KeplerMaps/KepIDs_%s.npy'%suff)
    SNRP, SNRsma, SNRF, SNRTeq = np.load('KeplerMaps/SNR_maps_%s.npy'%suff)
    MESP, MESsma, MESF, MESTeq = np.load('KeplerMaps/MES_maps_%s.npy'%suff)
    sensP,senssma,sensF,sensTeq = np.load('KeplerMaps/sens_maps_%s.npy'%suff)
    sensSmearP, sensSmearsma, sensSmearF, sensSmearTeq = \
                                np.load('KeplerMaps/sensSmear_maps_%s.npy'%suff)
    probP, probsma, probF, probTeq = np.load('KeplerMaps/prob_maps_%s.npy'%suff)
    compP, compsma, compF, compTeq = \
                            np.load('KeplerMaps/completeness_maps_%s.npy'%suff)
    compSmearP, compSmearsma, compSmearF, compSmearTeq = \
                        np.load('KeplerMaps/completenessSmear_maps_%s.npy'%suff)
    fP, fsma, fF, fTeq = np.load('KeplerMaps/occurrence_maps_%s.npy'%suff)
    fSmearP, fSmearsma, fSmearF, fSmearTeq = \
                        np.load('KeplerMaps/occurrenceSmear_maps_%s.npy'%suff)
    return NdetP, Ndetsma, NdetF, NdetTeq, KepIDs, SNRP, SNRsma, SNRF, SNRTeq, MESP, MESsma, MESF, MESTeq, sensP, sensSmearP, senssma, sensSmearsma, sensF, sensSmearF, sensTeq, sensSmearTeq, probP, probsma, probF, probTeq, compP, compSmearP, compsma, compSmearsma, compF, compSmearF, compTeq, compSmearTeq, fP, fSmearP, fsma, fSmearsma, fF, fSmearF, fTeq, fSmearTeq
    
    
    
if __name__ == '__main__':
    self = loadpickle('Keplertargets/KepConfirmedMdwarfPlanets_v3')
    print 'Computing detection maps...'
    NdetP, Ndetsma, NdetF, NdetTeq = compute_Ndet_maps(self)
    print 'Computing transit S/N maps for all Kepler M dwarfs...'
    KepIDs, SNRP, SNRsma, SNRF, SNRTeq = compute_SNR_maps(self)
    print 'Computing MES maps for all Kepler M dwarfs...'
    MESP, MESsma, MESF, MESTeq = compute_MES_maps(SNRP, SNRsma, SNRF, SNRTeq)
    print 'Computing sensitivity maps for all Kepler M dwarfs...'
    sensP, sensSmearP, senssma, sensSmearsma, sensF, sensSmearF, sensTeq, sensSmearTeq = compute_sens_maps(KepIDs, MESP, MESsma, MESF, MESTeq)
    print 'Computing transit probability maps for all Kepler M dwarfs...'
    probP, probsma, probF, probTeq = compute_transitprob_maps(self, KepIDs)
    print 'Computing completeness maps for all Kepler M dwarfs...'
    compP, compSmearP, compsma, compSmearsma, compF, compSmearF, compTeq, compSmearTeq = compute_completeness_maps(sensP, sensSmearP, senssma, sensSmearsma, sensF, sensSmearF, sensTeq, sensSmearTeq, probP, probsma, probF, probTeq)
    print 'Computing 2D planet occurrence rates maps for all Kepler M dwarfs...'
    fP, fSmearP, fsma, fSmearsma, fF, fSmearF, fTeq, fSmearTeq = compute_occurrence_maps(NdetP, Ndetsma, NdetF, NdetTeq, compP, compSmearP, compsma, compSmearsma, compF, compSmearF, compTeq, compSmearTeq)
    
    #p = load_maps('allM')
    #NdetP, Ndetsma, NdetF, NdetTeq, KepIDs, SNRP, SNRsma, SNRF, SNRTeq, MESP, MESsma, MESF, MESTeq, sensP, sensSmearP, senssma, sensSmearsma, sensF, sensSmearF, sensTeq, sensSmearTeq, probP, probsma, probF, probTeq, compP, compSmearP, compsma, compSmearsma, compF, compSmearF, compTeq, compSmearTeq, fP, fSmearP, fsma, fSmearsma, fF, fSmearF, fTeq, fSmearTeq = p
