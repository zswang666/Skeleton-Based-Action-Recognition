import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cv2
import functools
from pylab import cm, colorbar

import ipdb

data_path = "./data/results.pkl"
data = pickle.load(open(data_path, "rb"))
n_primitives = 5
output_dir = "./data"

def colorbar_index(ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)        

class EasyTransitionVis(object):
    def __init__(self, num_labels):
        self._num_labels = num_labels

        # construct color mapping
        '''
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        self._color_mapping = []
        for idx, (k,v) in enumerate(sorted(colors.items()), 0):
            self._color_mapping.append((k,mcolors.to_rgb(v)))
            if idx>=self._num_labels-1:
                break
        '''
        self._cmap = cm.get_cmap('PiYG', self._num_labels+1)
        cmap_list = [self._cmap(i)[:3] for i in range(self._cmap.N)]
        cmap_list[0] = (0.,0.,0.)
        self._cmap = self._cmap.from_list('Custom cmap', cmap_list, self._cmap.N)

    def draw(self, data, name):
        n_samples = len(data)

        # get max length
        max_len = 0
        for datum in data:
            datum_len = len(datum)
            if datum_len > max_len:
                max_len = datum_len

        # visualize data
        mat = np.zeros((n_samples,max_len))
        for idx, datum in enumerate(data):
            for t,d in enumerate(datum):
                mat[idx,t] = d + 1

        # show
        self._discrete_matshow(mat, name)

    def _discrete_matshow(self, mat, name):
        plt.imshow(mat, cmap=self._cmap)
        colorbar_index(ncolors=self._num_labels+1, cmap=self._cmap)
        plt.title(name)
        plt.suptitle("the colorbar indicates primitive 1~{}, 0 is none".format(self._num_labels),
                     y=0.95)
        plt.xlabel("timesteps")
        plt.ylabel("samples")
        plt.show()


visualizer = EasyTransitionVis(5)
for k, v in data.items():
    visualizer.draw(v, k)
