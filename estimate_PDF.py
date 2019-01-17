from imports import *
from scipy.stats import skewnorm
from scipy.optimize import curve_fit


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
    #print p16, med, p84
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
