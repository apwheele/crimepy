'''
Plot helper
functions
'''

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from cycler import cycler
import numpy as np
import importlib_resources
from matplotlib import font_manager

# colors via Van Gogh
colors = {"cdblue": "#286090",
          "brown" :"#7D5D2D",
          "green" :"#233A2D",
          "tan" :"#C5C88F",
          "blue" :"#455778",
          "lightblue" :"#9EACC5",
          "gold" :"#A58E38",
          "cdgrey": "#DDDDDD"}

andy_cycler = cycler(color=list(colors.values()))

#######################################
# seeing if Verdana is available and using that
# else it just goes to the backup

fl = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
verdana_name = matplotlib.rcParams['font.sans-serif'][0]
for f in fl:
    if 'VERDANA.TTF' in f.upper():
        vl = f
        verdana_font = font_manager.FontProperties(fname=vl)
        verdana_name = verdana_font.get_name()

#######################################


andy_theme = {'font.sans-serif': [verdana_name],
              'font.family': 'sans-serif',
              'axes.grid': True,
              'axes.axisbelow': True,
              'grid.linestyle': '--',
              'grid.color': colors['cdgrey'],
              'legend.framealpha': 1,
              'legend.facecolor': 'white',
              'legend.shadow': True,
              'legend.fontsize': 14,
              'legend.title_fontsize': 16,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'axes.labelsize': 16,
              'axes.titlesize': 20,
              'figure.dpi': 200,
              'axes.titlelocation': 'left',
              'axes.prop_cycle': andy_cycler}


matplotlib.rcParams.update(andy_theme)
im = imread(importlib_resources.files('crimepy').joinpath('CDCWLineRec.PNG'))

def add_logo(ax, loc=[0.78,0.78], size=0.2, logo=im):
    if loc is None:
        return None
    if type(logo) == str:
        im = image.imread(logo)
    else:
        im = logo
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()
    xdif = xrange[1] - xrange[0]
    ydif = yrange[1] - yrange[0]
    startx = loc[0]*xdif + xrange[0]
    starty = loc[1]*ydif + yrange[0]
    coords = [startx,starty,size*xdif,size*ydif]
    axin = ax.inset_axes(coords,transform=ax.transData)
    axin.imshow(im)
    axin.axis('off')


# combining legend
def combo_legend(ax,sort=False):
    handler, labeler = ax.get_legend_handles_labels()
    hd = []
    labli = list(set(labeler))
    for lab in labli:
        comb = [h for h,l in zip(handler,labeler) if l == lab]
        hd.append(tuple(comb))
    # resorting if you want that
    if sort:
        sorted_pairs = sorted(zip(labli,hd))
        labli, hd = zip(*sorted_pairs)
    return hd, labli


# check colors
def check_colors(logo=False,show=False):
    lc = len(colors)
    x = range(lc)
    y = [1]*lc
    cy = andy_cycler()
    fig, ax = plt.subplots()
    for a,b in zip(x,y):
       b = ax.barh(-a,b,label=a)
       t = ax.text(0.5,-a,next(cy)['color'],horizontalalignment='center',
               verticalalignment='center')
    ax.set_axis_off()
    if logo:
        add_logo(ax)
    if show:
        fig.show()
    else:
        return ax


# Brownian motion
def traj(n):
    pv = np.random.random() - 0.5
    res = [pv]
    for i in range(n-1):
        nv = pv + np.random.random() - 0.5
        res.append(nv)
        pv = nv
    return res


def check_line(n=20,show=False,**kwargs):
    lc = len(colors)
    x = range(n)
    cy = andy_cycler()
    y = [traj(n) for _ in range(lc)]
    fig, ax = plt.subplots()
    for t in y:
       l = ax.plot(x,t,'-',markeredgecolor='white',label=next(cy)['color'],**kwargs)
    #ax.legend(bbox_to_anchor=(1.0, 0.8))
    ax.set_axis_off()
    if show:
        fig.show()
    else:
        return ax