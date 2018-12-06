from imports import *
from get_Kepler_Mdwarf_planets import *
from scipy.ndimage.filters import gaussian_filter


global KepMdwarffile, Pbins, rpbins
KepMdwarffile = '../GAIAMdwarfs/input_data/Keplertargets/KepMdwarfsv11.csv'
xlen = 40
Pbins = np.logspace(np.log10(.5), 2, xlen)
smabins = np.logspace(-2, np.log10(.35), xlen)
Fbins = np.logspace(np.log10(.5), 2, xlen)
Teqbins = np.logspace(np.log10(240), np.log10(12e2), xlen)
rpbins = np.logspace(np.log10(.5), 1, 31)


def compute_Ndet_maps(self, sigs=.8, condition=None):
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

    # compute Ndet maps
    zP,P_edges,rp_edges = np.histogram2d(self.Ps[g], self.rps2[g],
                                         bins=[Pbins,rpbins])
    zsma,sma_edges,rp_edges = np.histogram2d(self.smas2[g], self.rps2[g],
                                             bins=[smabins,rpbins])
    zF,F_edges,rp_edges = np.histogram2d(self.Fs2[g], self.rps2[g],
                                         bins=[Fbins,rpbins])
    zTeq,Teq_edges,Teq_edges = np.histogram2d(self.Teqs2[g], self.rps2[g],
                                              bins=[Teqbins,rpbins])

    # return Ndet maps
    if type(sigs) in [float,int]:
        sigs = np.repeat(sigs, 4)
    else:
        assert len(sigs) == 4
        sigs = np.array(sigs)
    NdetP_smooth = gaussian_filter(zP, sigs[0])
    Ndetsma_smooth = gaussian_filter(zsma, sigs[1])
    NdetF_smooth = gaussian_filter(zF, sigs[2])
    NdetTeq_smooth = gaussian_filter(zTeq, sigs[2])

    # save to table
    np.save('KeplerMaps/Ndet_maps', np.array([NdetP_smooth,
                                              Ndetsma_smooth,
                                              NdetF_smooth,
                                              NdetTeq_smooth]))

    return NdetP_smooth, Ndetsma_smooth, NdetF_smooth, NdetTeq_smooth



def compute_SNR_maps(self, sigs=.8):
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
        
        for j in range(xlen):
            for k in range(ylen):

                rp = np.mean(rpbins[k:k+2]) 
                Z = (rvs.Rearth2m(rp) / rvs.Rsun2m(Rs))**2 * 1e6
                P = np.mean(Pbins[j:j+2])
                D = rvs.transit_width(P, Ms, Rs, rp, 0)
                CDPP = get_fitted_cdpp(KepIDsout[i], D)
                Ntransit = get_Ntransits(KepIDsout[i], P)
                SNRP_maps[i,j,k] = Z / CDPP * np.sqrt(Ntransit)
                
                sma = np.mean(smabins[j:j+2])
                P = rvs.period_sma(sma, Ms, 0)
                Ntransit = get_Ntransits(KepIDsout[i], P)
                SNRsma_maps[i,j,k] = Z / CDPP * np.sqrt(Ntransit)
                
                F = np.mean(Fbins[j:j+2])
                sma = np.sqrt(Rs**2 * (Teff/5778.)**4 / F)
                P = rvs.period_sma(sma, Ms, 0)
                Ntransit = get_Ntransits(KepIDsout[i], P)
                SNRF_maps[i,j,k] = Z / CDPP * np.sqrt(Ntransit)
                
                Teq = np.mean(Teqbins[j:j+2])
                sma = .5*rvs.m2AU(rvs.Rsun2m(Rs)) * (Teff/Teq)**2
                P = rvs.period_sma(sma, Ms, 0)
                Ntransit = get_Ntransits(KepIDsout[i], P)
                SNRTeq_maps[i,j,k] = Z / CDPP * np.sqrt(Ntransit)
                                
        # smooth maps
        SNRP_maps[i] = gaussian_filter(SNRP_maps[i], sigs[0])
        SNRsma_maps[i] = gaussian_filter(SNRsma_maps[i], sigs[1])
        SNRF_maps[i] = gaussian_filter(SNRF_maps[i], sigs[2])
        SNRTeq_maps[i] = gaussian_filter(SNRTeq_maps[i], sigs[3])

    # save to fits table
    cP = _create_fits_column(NdetP_smooth, 'Ndet_P_rp')
    csma = _create_fits_column(Ndetsma_smooth, 'Ndet_sma_rp')
    cF = _create_fits_column(NdetF_smooth, 'Ndet_F_rp')
    cTeq = _create_fits_column(NdetTeq_smooth, 'Ndet_Teq_rp')
    _create_fits_table([cP,csma,cF,cTeq], 'KeplerMaps/Ndet_maps.fits')
        
    return KepIDsout, SNRP_maps, SNRsma_maps, SNRF_maps, SNRTeq_maps 



def compute_MES_maps(SNRP_maps, SNRsma_maps, SNRF_maps, SNRTeq_maps, sigs=.8):
    '''Compute the MES maps from the SNR maps over various distance proxies.'''
    # get smoothing parameters
    if type(sigs) in [float,int]:
        sigs = np.repeat(sigs, 4)
    else:
        assert len(sigs) == 4
        sigs = np.array(sigs)

    # get MES maps
    assert (SNRP_maps.shape) == 3
    assert (SNRsma_maps.shape) == 3
    assert (SNRF_maps.shape) == 3
    assert (SNRTeq_maps.shape) == 3
    Nstars = SNRP_maps.shape[0] 
    MESP_maps = SNR2MES(SNRP_maps)        
    MESsma_maps = SNR2MES(SNRsma_maps)        
    MESF_maps = SNR2MES(SNRF_maps)        
    MESTeq_maps = SNR2MES(SNRTeq_maps)        

    # smooth maps
    for i in range(Nstars):
        MESP_maps[i] = gaussian_filter(MESP_maps[i], sigs[0])
        MESsma_maps[i] = gaussian_filter(MESTsmamaps[i], sigs[1])
        MESF_maps[i] = gaussian_filter(MESF_maps[i], sigs[2])
        MESTeq_maps[i] = gaussian_filter(MESTeq_maps[i], sigs[3])

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
    
    assert (MESP_maps.shape) == 3
    assert (MESsma_maps.shape) == 3
    assert (MESF_maps.shape) == 3
    assert (MESTeq_maps.shape) == 3
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

        sensP_maps[i], sensSmearP_maps[i] = mP.T
        senssma_maps[i], sensSmearsma_maps[i] = msma.T
        sensF_maps[i], sensSmearF_maps[i] = mF.T
        sensTeq_maps[i], sensSmearTeq_maps[i] = mTeq.T

        # smooth maps
        sensP_maps[i] = gaussian_filter(sensP_maps[i], sigs[0])
        sensSmearP_maps[i] = gaussian_filter(sensSmearP_maps[i], sigs[1])
        senssma_maps[i] = gaussian_filter(senssma_maps[i], sigs[2])
        sensSmearsma_maps[i] = gaussian_filter(sensSmearsma_maps[i], sigs[3])
        sensF_maps[i] = gaussian_filter(sensF_maps[i], sigs[4])
        sensSmearF_maps[i] = gaussian_filter(sensSmearF_maps[i], sigs[5])
        sensTeq_maps[i] = gaussian_filter(sensTeq_maps[i], sigs[6])
        sensSmearTeq_maps[i] = gaussian_filter(sensSmearTeq_maps[i], sigs[7])
        
    return sensP_maps, sensSmearP_maps, senssma_maps, sensSmearsma_maps, \
        sensF_maps, sensSmearF_maps, sensTeq_maps, sensSmearTeq_maps 



def compute_transitprob_maps(self, KepIDs, sigs=.8, factor=1.08):
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
                probP_maps[i,j,k] = (rvs.Rsun2m(Rs) + rvs.Rearth2ms(rp)) / \
                                    rvs.AU2m(sma) * correction_factor
                
                sma = np.mean(smabins[j:j+2])
                probsma_maps[i,j,k] = (rvs.Rsun2m(Rs) + rvs.Rearth2ms(rp)) / \
                                      rvs.AU2m(sma) * correction_factor
                
                F = np.mean(Fbins[j:j+2])
                sma = np.sqrt(Rs**2 * (Teff/5778.)**4 / F)
                probF_maps[i,j,k] = (rvs.Rsun2m(Rs) + rvs.Rearth2ms(rp)) / \
                                    rvs.AU2m(sma) * correction_factor
                
                Teq = np.mean(Teqbins[j:j+2])
                sma = .5*rvs.m2AU(rvs.Rsun2m(Rs)) * (Teff/Teq)**2
                probTeq_maps[i,j,k] = (rvs.Rsun2m(Rs) + rvs.Rearth2ms(rp)) / \
                                      rvs.AU2m(sma) * correction_factor

        # smooth maps
        probP_maps[i] = gaussian_filter(probP_maps[i], sigs[0])
        probsma_maps[i] = gaussian_filter(probsma_maps[i], sigs[1])
        probF_maps[i] = gaussian_filter(probF_maps[i], sigs[2])
        probTeq_maps[i] = gaussian_filter(probTeq_maps[i], sigs[3])

    return probP_maps, probsma_maps, probF_maps, probTeq_maps


    
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





if __name__ == '__main__':
    self = loadpickle('Keplertargets/KepConfirmedMdwarfPlanets_v3')
    print 'Computing detection maps...'
    NdetP, Ndetsma, NdetF, NdetTeq = compute_Ndet_maps(self, sigs=.8)
    '''print 'Computing transit S/N maps for all Kepler M dwarfs...'
    KepIDs, SNRP, SNRsma, SNRF, SNRTeq = compute_SNR_maps(self, sigs=.8)
    print 'Computing MES maps for all Kepler M dwarfs...'
    MESP, MESsma, MESF, MESTeq = compute_MES_maps(SNRP, SNRsma, SNRF, SNRTeq,
                                                  sigs=.8)
    print 'Computing sensitivity maps for all Kepler M dwarfs...'
    sensP, sensSmearP, senssma, sensSmearsma, sensF, sensSmearF, sensTeq, sensSmearTeq = compute_sens_maps(KepIDs, MESP, MESsma, MESF, MESTeq)
    print 'Computing transit probability maps for all Kepler M dwarfs...'
    probP, probsma, probF, probTeq = compute_transitprob_maps(self, KepIDs,
                                                              sigs=.8)

    # compute completeness
    print 'Computing completeness maps for all Kepler M dwarfs...'
    compP, compSmearP = sensP*probP, sensSmearP*probP
    compsma, compSmearsma = senssma*probsma, sensSmearsma*probsma
    compF, compSmearF = sensF*probF, sensSmearF*probF
    compTeq, compSmearTeq = sensTeq*probTeq, sensSmearTeq*probTeq

    # compute 2d occurrence rate
    print 'Computing planet occurrence rates maps for all Kepler M dwarfs...'
    fP, fSmearP = NdetP/np.mean(compP,0), NdetP/np.mean(compSmearP,0)
    fsma, fSmearsma = Ndetsma/np.mean(compsma,0), \
                      Ndetsma/np.mean(compSmearsma,0)
    fF, fSmearF = NdetF/np.mean(compF,0), NdetF/np.mean(compSmearF,0)
    fTeq, fSmearTeq = NdetTeq/np.mean(compTeq,0), \
                      NdetTeq/np.mean(compSmearTeq,0)
    '''
