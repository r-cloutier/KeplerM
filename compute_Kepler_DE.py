# Kepler data access:
# https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_completeness_reliability.html
from imports import *
from KeplerPORTs import *


global KepMdwarffile, Pgrid, rpgrid
KepMdwarffile = '../GAIAMdwarfs/input_data/Keplertargets/KepMdwarfsv11.csv'
Pgrid = np.logspace(np.log10(.5), np.log10(1e2), 1000)
rpgrid = np.logspace(np.log10(.5), np.log10(10), 500)


def get_stellar_table(clean_table=False):
    fname = 'TPSfiles/nph-nstedAPI.txt'
    fname_v2 = fname.replace('.txt','_clean.txt')
    if clean_table:
        # fill in missing 2MASS names
        f = open(fname, 'r')
        g = f.readlines()
        f.close()
        for i in range(208,len(g)):
            
            print float(i)/ len(g)
            ls = ','.join(g[i].split()).split(',')

            # add 2MASS placeholder such that len == 100
            if len(ls) == 99:
                lsout = list(np.append(list(np.append(ls[0],'2MASS')), ls[1:]))
            
            elif len(ls) == 100:
                lsout = ls
            
            else:
                print i, len(ls)
                raise ValueError("Not sure what's wrong with this entry")

            strout = ','.join(lsout) + '\n'
            assert len(strout.split(',')) == 100
            g[i] = strout

        # write new file
        f = open(fname_v2, 'w')
        f.write(''.join(g))
        f.close()
        

    # get completeness parameters from the Kepler transiting planet search 
    d = np.genfromtxt(fname_v2, skip_header=208, delimiter=',',
                      usecols=(0,23,24,25,26,81,82,98,99))
    KepIDs_TPS,LDC1,LDC2,LDC3,LDC4,duty_cycles_TPS,data_spans_TPS,CDPPlongslope,CDPPshortslope = d.T
    return d.T
    

# get the stellar table once
KepIDs_TPS,LDC1,LDC2,LDC3,LDC4,duty_cycles_TPS,data_spans_TPS,CDPPlongslope,CDPPshortslope = get_stellar_table()
LDCs_TPS = np.array([LDC1,LDC2,LDC3,LDC4]).T


def download_KIC_files(kepid):
    # check that the files have not already been downloaded
    kepid_str = '%.9d'%kepid
    if os.path.exists('TPSfiles/kplr%s_dr25_onesigdepth.fits'%kepid_str) and \
       os.path.exists('TPSfiles/kplr%s_dr25_window.fits'%kepid_str):
        return None
    
    # get window and one sigma depth files from the Kepler pipeline
    dirs = np.append(range(13),100)
    got_file = False
    for i in dirs:
        
        # get onesigdepth file
        if not got_file:
            url = 'http://exoplanetarchive.ipac.caltech.edu:80/data/KeplerData/'
            url += '%.3d/%s/%s/tps/'%(i, kepid_str[:6], kepid_str)
            fname = 'kplr%s_dr25_onesigdepth.fits'%kepid_str
            cmd = 'wget %s/%s'%(url, fname)
            os.system(cmd)
            if os.path.exists(fname):
                got_file = True
                os.system('mv %s TPSfiles'%fname)

    # get window file
    fname = fname.replace('onesigdepth', 'window')
    cmd = 'wget %s/%s'%(url, fname)
    os.system(cmd)
    os.system('mv %s TPSfiles'%fname)
    
    return got_file


def getDE_Kepler_star(kepid):
    '''Originally from KeplerPORTs.py for computing the detection efficiency 
    (i.e. sensitivity): https://github.com/nasa/KeplerPORTs
    '''
    # ensure that it is okay to use this DE model on this star
    badKepIDs = np.loadtxt('DR25_DEModel_NoisyTargetList.txt')
    if kepid in badKepIDs:
        return np.repeat(np.nan,5)

    # get my stellar data from GAIA
    d = np.loadtxt(KepMdwarffile, delimiter=',', usecols=(0,7,27,30,36))
    KepIDs,Kepmags,Rss,Teffs,loggs = d.T    
    
    # Define the stellar and noise properties needed for detection contour
    # Begin by making an instance of the class that holds the properties
    #  of detection contour
    # Parameters available from the DR25 Stellar and Occurrence product
    #   table hosted at NASA Exoplanet Archive
    doit = kepler_single_comp_data()
    doit.id = int(kepid)
    g = np.in1d(KepIDs, kepid)
    assert g.sum() == 1
    doit.rstar = float(Rss[g])
    doit.logg = float(loggs[g])
    doit.teff = float(Teffs[g])

    # get Kepler TPS completeness parameters
    g = np.in1d(KepIDs_TPS, kepid)
    assert g.sum() == 1
    doit.dataspan = float(data_spans_TPS[g])
    doit.dutycycle = float(duty_cycles_TPS[g])
    doit.limbcoeffs = LDCs_TPS[g].reshape(4)
    doit.cdppSlopeLong = float(CDPPlongslope[g])
    doit.cdppSlopeShort = float(CDPPshortslope[g])

    # Download  window function and one-sigma depth function tables and
    # define path to the detection grids
    got_file = download_KIC_files(kepid)
    if not got_file:
	return None
    doit.period_want = Pgrid
    doit.rp_want = rpgrid
    doit.ecc = 0.
    doit.planet_detection_metric_path = './TPSfiles'

    # All parameters are set, generate detection contour and detection
    # efficiency curve
    probdet, probtot, DEMod = kepler_single_comp_dr25(doit)
    Mes = np.linspace(0, 50, 1000)
    DE = DEMod.detEffFunctions[0](Mes)
    SmearDE = DEMod.detEffSmearFunctions[0](Mes)
    
    # save detection efficiency
    np.save('TPSfiles/DE_KepID_%i'%kepid, np.array([Mes, DE, SmearDE]).T)
    return probdet, probtot, Mes, DE, SmearDE


if __name__ == '__main__':
    kepids = np.loadtxt(KepMdwarffile, delimiter=',')[:,0]
    for i in range(1170, kepids.size):
	print float(i) / kepids.size, kepids[i]
	_= getDE_Kepler_star(kepids[i])
