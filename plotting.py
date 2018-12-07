from imports import *


def plot_map(xarr, yarr, zarr, xlabel='', ylabel='', zlabel='',
             avgtitle=False, tottitle=False, hatchthresh=0, contour_levels=[],
             contour_colors='k', pltt=True, label=''):
    
    xlen, ylen = xarr.size, yarr.size
    assert zarr.shape == (xlen-1,ylen-1)
    
    fig = plt.figure(figsize=(7.7,6))
    ax = fig.add_subplot(111)
    cax = ax.pcolormesh(xarr, yarr, zarr.T, cmap=plt.get_cmap('hot_r'))
    cbar_axes = fig.add_axes([.1,.1,.87,.04])
    cbar = fig.colorbar(cax, cax=cbar_axes, orientation='horizontal')#, pad=.11)
    cbar.set_label(zlabel)
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
    ax.set_xscale('log'), ax.set_yscale('log')
    
    if avgtitle:
        title, num = 'Avg = ', np.nanmean(zarr)
    elif tottitle:
        title, num = 'Total = ', np.nansum(zarr)
    else:
        title, num = '', np.nan
    ax.set_title('%s%.3e'%(title, num))

    if hatchthresh != 0:
        for i in range(xarr.size-1):
            for j in range(yarr.size-1):
                if zarr[i,j] < hatchthresh:
                    ax.fill([xarr[i],xarr[i+1],xarr[i+1],xarr[i]],
                            [yarr[j],yarr[j],yarr[j+1],yarr[j+1]],
                            fill=False, hatch='/')

    if len(contour_levels) > 0:
        x = 10**(np.log10(xarr[1:]) - np.diff(np.log10(xarr))[0]/2)
        y = 10**(np.log10(yarr[1:]) - np.diff(np.log10(yarr))[0]/2)
        V = np.sort(contour_levels)
        cs = ax.contour(x, y, zarr.T, levels=V, colors=contour_colors)
        ax.clabel(cs, inline=1, fontsize=10, fmt='%.2f')
    
    fig.subplots_adjust(bottom=.25, top=.95, right=.96, left=.1)
    if label != '':
        plt.savefig('plots/%s.png'%label)
    if pltt:
        plt.show()
    plt.close('all')



def plot_marginalized_rp_hist(rpedges, occurrence_map, Nplanets, pltt=True,
                              label=''):

    assert occurrence_map.shape[1] == rpedges.size-1
    rp_per_star = np.nansum(occurrence_map, 0)
    e_rp_per_star = np.sqrt(np.nansum(occurrence_map*Nplanets, 0)) / \
                    float(Nplanets)
    rpcens = 10**(np.log10(rpedges[1:]) - np.diff(np.log10(rpedges))[0]/2)
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    ax.errorbar(rpcens, rp_per_star, e_rp_per_star, c='k', lw=3,
                drawstyle='steps-mid', elinewidth=1, capsize=2)
    ax.set_xlabel('Planet Radius [R$_{\oplus}$]')
    ax.set_xscale('log')

    if label != '':
        plt.savefig('plots/%s.png'%label)
    if pltt:
        plt.show()
    plt.close('all')
