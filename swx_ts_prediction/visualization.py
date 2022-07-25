# coding=utf-8
'''
note : this is a update version for visualization, including:
swx_scatter_matrix, plot_correlation_map and plot_zhihou_corr
author shenwanxiang,2017-06
'''

import math
import numpy as np
import pandas as pd
import pandas.core.common as com
from pandas.compat import range, lrange, lmap, map, zip, string_types
import matplotlib.pylab as plt
import seaborn as sns


plt.rcParams['figure.figsize'] = 15, 6
plt.rcParams['font.sans-serif']=['SimHei'] #show chinese
plt.rcParams['axes.unicode_minus']=False #show '-'


def _get_marker_compat(marker):
    import matplotlib.lines as mlines
    import matplotlib as mpl
    if mpl.__version__ < '1.1.0' and marker == '.':
        return 'o'
    if marker not in mlines.lineMarkers:
        return 'o'
    return marker

def _label_axis(ax, kind='x', label='', position='top',
    ticks=True, rotate=False):

    from matplotlib.artist import setp
    if kind == 'x':
        ax.set_xlabel(label, visible=True)
        ax.xaxis.set_visible(True)
        ax.xaxis.set_ticks_position(position)
        ax.xaxis.set_label_position(position)
        if rotate:
            setp(ax.get_xticklabels(), rotation=90)
    elif kind == 'y':
        ax.yaxis.set_visible(True)
        ax.set_ylabel(label, visible=True)
        # ax.set_ylabel(a)
        ax.yaxis.set_ticks_position(position)
        ax.yaxis.set_label_position(position)
    return

def _subplots(nrows=1, ncols=1, naxes=None, sharex=False, sharey=False, squeeze=True,
              subplot_kw=None, ax=None, **fig_kw):
    """Create a figure with a set of subplots already made.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Keyword arguments:

    nrows : int
      Number of rows of the subplot grid.  Defaults to 1.

    ncols : int
      Number of columns of the subplot grid.  Defaults to 1.

    naxes : int
      Number of required axes. Exceeded axes are set invisible. Default is nrows * ncols.

    sharex : bool
      If True, the X axis will be shared amongst all subplots.

    sharey : bool
      If True, the Y axis will be shared amongst all subplots.

    squeeze : bool

      If True, extra dimensions are squeezed out from the returned axis object:
        - if only one subplot is constructed (nrows=ncols=1), the resulting
        single Axis object is returned as a scalar.
        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy object
        array of Axis objects are returned as numpy 1-d arrays.
        - for NxM subplots with N>1 and M>1 are returned as a 2d array.

      If False, no squeezing at all is done: the returned axis object is always
      a 2-d array containing Axis instances, even if it ends up being 1x1.

    subplot_kw : dict
      Dict with keywords passed to the add_subplot() call used to create each
      subplots.

    ax : Matplotlib axis object, optional

    fig_kw : Other keyword arguments to be passed to the figure() call.
        Note that all keywords not recognized above will be
        automatically included here.


    Returns:

    fig, ax : tuple
      - fig is the Matplotlib Figure object
      - ax can be either a single axis object or an array of axis objects if
      more than one subplot was created.  The dimensions of the resulting array
      can be controlled with the squeeze keyword, see above.

    **Examples:**

    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # Just a figure and one subplot
    f, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Two subplots, unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Four polar axes
    plt.subplots(2, 2, subplot_kw=dict(polar=True))
    """
    import matplotlib.pyplot as plt
    from pandas.core.frame import DataFrame

    if subplot_kw is None:
        subplot_kw = {}

    # Create empty object array to hold all axes.  It's easiest to make it 1-d
    # so we can just append subplots upon creation, and then
    nplots = nrows * ncols

    if naxes is None:
        naxes = nrows * ncols
    elif nplots < naxes:
        raise ValueError("naxes {0} is larger than layour size defined by nrows * ncols".format(naxes))

    if ax is None:
        fig = plt.figure(**fig_kw)
    else:
        fig = ax.get_figure()
         # if ax is passed and a number of subplots is 1, return ax as it is
        if naxes == 1:
            if squeeze:
                return fig, ax
            else:
                return fig, _flatten(ax)
        else:
            warnings.warn("To output multiple subplots, the figure containing the passed axes "
                          "is being cleared", UserWarning)
            fig.clear()

    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)

    if sharex:
        subplot_kw['sharex'] = ax0
    if sharey:
        subplot_kw['sharey'] = ax0
    axarr[0] = ax0

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots):
        ax = fig.add_subplot(nrows, ncols, i + 1, **subplot_kw)
        axarr[i] = ax

    if nplots > 1:
        if sharex and nrows > 1:
            for ax in axarr[:naxes][:-ncols]:    # only bottom row
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                ax.xaxis.get_label().set_visible(False)
        if sharey and ncols > 1:
            for i, ax in enumerate(axarr):
                if (i % ncols) != 0:  # only first column
                    for label in ax.get_yticklabels():
                        label.set_visible(False)
                    ax.yaxis.get_label().set_visible(False)

    if naxes != nplots:
        for ax in axarr[naxes:]:
            ax.set_visible(False)

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots == 1:
            axes = axarr[0]
        else:
            axes = axarr.reshape(nrows, ncols).squeeze()
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        axes = axarr.reshape(nrows, ncols)

    return fig, axes


def swx_scatter_matrix(frame, alpha=0.5, beta = 0.8, figsize=None, ax=None, grid=False,
                   diagonal='hist', marker='.', density_kwds=None,
                   hist_kwds=None, range_padding=0.05, **kwds):
    """
    Draw a matrix of scatter plots.

    Parameters
    ----------
    frame : DataFrame
    alpha : float, optional
        amount of transparency applied
    figsize : (float,float), optional
        a tuple (width, height) in inches
    ax : Matplotlib axis object, optional
    grid : bool, optional
        setting this to True will show the grid
    diagonal : {'hist', 'kde'}
        pick between 'kde' and 'hist' for
        either Kernel Density Estimation or Histogram
        plot in the diagonal
    marker : str, optional
        Matplotlib marker type, default '.'
    hist_kwds : other plotting keyword arguments
        To be passed to hist function
    density_kwds : other plotting keyword arguments
        To be passed to kernel density estimate plot
    range_padding : float, optional
        relative extension of axis range in x and y
        with respect to (x_max - x_min) or (y_max - y_min),
        default 0.05
    kwds : other plotting keyword arguments
        To be passed to scatter function

    Examples
    --------
    >>> df = DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
    >>> scatter_matrix(df, alpha=0.2)
    """
    import matplotlib.pyplot as plt
    import math
    import pandas as pd
    from matplotlib.artist import setp
    import pandas.core.common as com

    df = frame._get_numeric_data()
    correlations = df.corr() #get correlations
    n = df.columns.size
    fig, axes = _subplots(nrows=n, ncols=n, figsize=figsize, ax=ax,
                          squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    mask = df.notnull()

    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # workaround because `c='b'` is hardcoded in matplotlibs scatter method
    kwds.setdefault('c', plt.rcParams['patch.facecolor'])

    boundaries_list = []
    for a in df.columns:
        values = df[a].values[mask[a].values]
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
        boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))

    for i, a in zip(lrange(n), df.columns):
        for j, b in zip(lrange(n), df.columns):
            ax = axes[i, j]

            if i == j:
                values = df[a].values[mask[a].values]

                # Deal with the diagonal by drawing a histogram there.
                if diagonal == 'hist':
                    ax.hist(values, **hist_kwds)

                elif diagonal in ('kde', 'density'):
                    from scipy.stats import gaussian_kde
                    y = values
                    gkde = gaussian_kde(y)
                    ind = np.linspace(y.min(), y.max(), 1000)
                    ax.plot(ind, gkde.evaluate(ind), **density_kwds)

                ax.set_xlim(boundaries_list[i])

            if i > j:
                common = (mask[a] & mask[b]).values

                ax.scatter(df[b][common], df[a][common],
                           marker=marker, alpha=alpha, **kwds)

                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])
            if i < j:
                x = round(correlations.values[i][j],2)
                ax.text(0.5, 0.5,str(x), 
                  size=20, rotation=0,ha="center", va="center",
                  bbox=dict(boxstyle="round",fc=(beta, abs(x), beta),ec=(beta, abs(x), beta)))
            ax.set_xlabel('')
            ax.set_ylabel('')

            _label_axis(ax, kind='x', label=b, position='bottom', rotate=True)

            _label_axis(ax, kind='y', label=a, position='left')

            if j!= 0:
                ax.yaxis.set_visible(False)
            if i != n-1:
                ax.xaxis.set_visible(False)

    for ax in axes.flat:
        setp(ax.get_xticklabels(), fontsize=8)
        setp(ax.get_yticklabels(), fontsize=8)

    return axes

def get_p_value(df1,df2,name):
    from scipy.stats.stats import pearsonr
    p_list = []
    for col in df1.columns:
        for col2 in df2.columns:
            xx = pearsonr(df1[col], df2[col2])
            p_list.append(xx[1])
    return pd.DataFrame(p_list,index = df1.columns, columns = [name]) 


def plot_zhihou_corr(dfx, dfy, title,lag_max = 30,kind ='line'):
    import seaborn as sns
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    lags = range(0,lag_max,1)
    al = []
    pl = []
    for lag in lags:
        dfx_shfit = dfx.shift(lag)
        dff = pd.concat([dfy,dfx_shfit],axis =1)
        d1 = dff.corr()
        
        p = get_p_value(dfx_shfit.dropna(),dfy.iloc[lag:],lag)

        d = d1[d1.columns[0]].to_frame(name =lag)
        al.append(d)
        pl.append(p)
    
    pcc = pd.concat(al,axis = 1).T
    p_value = pd.concat(pl,axis =1).T
    
    pc = pcc[pcc.columns[1:]]
    
    colors = sns.color_palette("husl", len(pcc.columns)) #http://seaborn.pydata.org/tutorial/color_palettes.html
    #pcc_sort = pc.sort_values(pc.first_valid_index(), axis=1,ascending =False) #sort by first row
    sns.reset_orig()
    plt.rcParams['font.sans-serif']=['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus']=False #show '-'
    pcc_sort = pc.reindex_axis(pc.mean().sort_values(ascending =False).index, axis=1)#sort by mean value
    pcc_sort.plot(figsize=(12,6),fontsize = 15,color = colors,kind = kind)
    

    
    plt.ylabel(u'皮尔逊相关系数',fontsize = 16)
    plt.xlabel(u'滞后周数',fontsize = 16)
    plt.title(title,fontsize = 16)
    plt.show()
    

    #p_value_sort = p_value.sort_values(p_value.first_valid_index(), axis=1,ascending =False)
    p_value_sort = p_value.reindex_axis(p_value.mean().sort_values(ascending =False).index, axis=1)
    
    p_value_sort.plot(figsize=(12,6),fontsize = 15,kind = kind,color = colors)    
    
    plt.ylabel(u'p 值',fontsize = 16)
    plt.xlabel(u'滞后周数',fontsize = 16) 
    plt.axhline(0.05,ls="--",color="r",label = '95% CI',lw='0.9')
    plt.legend()
    plt.show()    
    return pcc_sort,p_value

def plot_correlation_map(df):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
	
def plot_diff_zhihou_corr(dfx, dfy,diff = 2,lagmax = 10):
    if diff:
        dfx1 = dfx.diff(diff)
    else:dfx1 = dfx 
    ddd = dfy.join(dfx1).dropna()
    plot_zhihou_corr(ddd[ddd.columns[1:]], ddd[[ddd.columns[0]]], lag_max = lagmax,title = u'差分阶数：'+ str(diff))


def plot_diff_shift(df,diff_lag = 10,shift_lag = 2,title = u'发病人数在滞后与差分后相关性'):
    #控制光滑度
    dflist = []
    for i in range(shift_lag):
        for j in range(diff_lag):
            dff = df.diff(j+1).shift(i+1)
            dff.columns = ['shift'+str(i+1) + '_diff'+str(j+1)]
            dflist.append(dff)
    cc = df.join(pd.concat(dflist,axis =1)).corr()

    rcParams['figure.figsize'] = 13,6
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    cc[cc.columns[0]]['shift1_diff1':].plot(fontsize = 20)
    plt.legend(fontsize = 24)
    plt.ylabel(u'皮尔逊相关系数',fontsize = 24)
    plt.xlabel(u'滞后周数和差分周数组合',fontsize = 24)
    plt.title(title,fontsize = 24)
    return pd.concat(dflist,axis =1)

	
def plot_feature_import(dfu):
    aa = dfu.corr()[[dfu.columns[0]]].iloc[1:].rename(columns = {dfu.columns[0]:'pearson_corr'})
    dfp = dfu.dropna()
    bb = get_p_value(dfp[dfp.columns[1:]],dfp[[dfp.columns[0]]],'p_values')
    ax = pd.concat([aa,bb],axis =1).sort_values('pearson_corr').plot(kind='barh',width = 0.8, stacked=False,fontsize = 20)
    plt.axvline(0.05,ls="--",color="r",label = '95% CI',lw='1.5')
    plt.legend(fontsize = 24)
    plt.ylabel(u'皮尔逊相关系数 / P-值',fontsize = 24)
    ax.yaxis.set_label_position("right")