import numpy as np


def load_preprocess(fname, mode='N'):
    dataset = np.loadtxt(fname)

    spec = dataset[:,4:]
    para = dataset[:,:4]
    if mode == 'N':
        spec_min = np.min(spec)
        spec_max = np.max(spec)
        spec = (spec - spec_min) / (spec_max - spec_min)
    elif mode == 'S':
        spec = np.sqrt(spec) 
        spec_std = np.std(spec)
        spec_mean = np.mean(spec)
        spec = (spec - spec_mean) / spec_std
    elif mode == 'Log':
        spec = np.sqrt(spec)
        spec_min = np.min(spec)
        spec_max = np.max(spec)
        spec = (spec - spec_min) / (spec_max - spec_min)
        spec[spec==0.0] = 1.0
        spec = - np.log(spec)

    dataset[:,4:] = spec
    return dataset