import os
import sys
import importlib
import ruamel.yaml
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pylab import cm, colorbar

import ipdb

def validate_dir(*dir_name, **kwargs):
    if len(kwargs)>0:
        auto_mkdir = kwargs.pop("auto_mkdir")
        if len(kwargs):
            raise ValueError("Invalid arguments: {}".format(kwargs))
    else:
        auto_mkdir = True

    dir_name = os.path.join(*dir_name)
    if auto_mkdir and not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name

def validate_path(*path_name, **kwargs):
    if len(kwargs)>0:
        auto_mkdir = kwargs.pop("auto_mkdir")
        check_exist = kwargs.pop("check_exist")
        if len(kwargs):
            raise ValueError("Invalid arguments: {}".format(kwargs))
    else:
        auto_mkdir = True
        check_exist = False

    dir_name = os.path.join(*path_name[:-1])
    path_name = os.path.join(*path_name)
    if check_exist:
        return os.path.exists(path_name)
    print(dir_name)
    if auto_mkdir and not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return path_name

def append_PATH(path):
    os.environ["PATH"] += (os.pathsep+path)

def unappend_PATH(path):
    all_paths = os.environ["PATH"].split(os.pathsep)
    all_paths.remove(path)

def dynamic_import(module):
    return importlib.import_module(module)

def parse_yaml(name):
    with open(name, 'r') as f:
        cfg = ruamel.yaml.safe_load(f)
        for k,v in cfg.items():
            if isinstance(v, dict):
                cfg[k] = namedtuple("GenericDict", v.keys())(**v)
        cfg = namedtuple("GenericDict", cfg.keys())(**cfg)

    return cfg

def inverse_dict(dict_in):
    return {v: k for k, v in dict_in.items()}

def colorbar_index(ncolors, cmap, orientation="vertical"):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, orientation=orientation)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))

    return colorbar

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
