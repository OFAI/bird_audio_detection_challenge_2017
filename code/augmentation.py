import util
import h5py
import numpy as np
from itertools import izip

class Augmentation(object):
    def __init__(self, args={}):
        super(Augmentation, self).__init__()
    
        import pdb
        pdb.set_trace()
    
        # noise spectrum
        noise_fn = util.getarg(args, 'noise', label=label, dtype=str)
        # source clusters file
        src_clusters = util.getarg(args, 'src_clusters', label=label, dtype=str)
        # destination clusters file
        dst_clusters = util.getarg(args, 'dst_clusters', label=label, dtype=str)
        # cluster index
        dst_ix = util.getarg(args, 'dst_ix', label=label, dtype=int)
        # strength of adaptation (1...full)
        self.strength = util.getarg(args, 'strength', 1., label=label, dtype=float)
        
        # read noise spectrum
        with h5py.File(noise_fn, 'r') as f:
            self.linnoise = np.exp(f['features'].value)

        # read source cluster data
        with h5py.File(src_clusters, 'r') as f:
            clusters = f['clusters'].value
            nclusters = np.max(clusters)
            items = f['items'].value
            # translation dict item -> cluster index
            self.clusterdict = dict(izip(items, clusters))
            # mask for each cluster
            src_msks = [f['clusters'].value == c for c in range(1, nclusters+1)]
            # means for each cluster
            src_means = [f['mean'][m] for m in src_msks]
            self.src_mean_means = map(np.mean, src_means)
            self.src_mean_stds = map(np.std, src_means)
            # p01s for each cluster
            src_p01s = [f['p01'][m] for m in src_msks]
            self.src_p01_means = map(np.mean, src_p01s)
            self.src_p01_stds = map(np.std, src_p01s)

        # read destination cluster data
        # we only need the one specified by dst_ix
        with h5py.File(dst_clusters, 'r') as f:
            dst_mask = f['clusters'].value == dst_ix
            dst_mean = f['mean'][dst_mask]
            self.dst_mean_mean = np.mean(dst_mean)
            self.dst_mean_std = np.std(dst_mean)
            dst_p01 = f['p01'][dst_mask]
            self.dst_p01_mean = np.mean(dst_p01)
            self.dst_p01_std = np.std(dst_p01)
    
    
    def __call__(self, spec, item):
        import pdb
        pdb.set_trace()
    
        # get cluster
        cl = self.clusterdict[tuple(item.split('/'))]
        
        src_mean = self.src_mean_means[cl]
        src_p01 = self.src_p01_means[cl]
        dst_mean = self.dst_mean_mean
        dst_p01 = self.dst_p01_mean
        
        src_gain_offs = src_mean-src_p01
        # compute linear spectrum
        spec = np.exp(spec-src_gain_offs)
        # compute amount of noise
        amount = (np.exp(dst_p01)-np.exp(src_p01))*self.strength
        # add noise spectrum
        offs = np.random.randint(0, len(self.linnoise)-len(spec))
        spec += self.linnoise[offs:][:len(spec)]*amount
        
        dst_gain_offs = dst_mean-dst_p01
        return np.log(np.maximum(spec,1.e-7))+dst_gain_offs
