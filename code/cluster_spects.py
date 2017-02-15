#!/usr/bin/env python
from glob import glob
import h5py
import numpy as np
import cPickle
import sys
import logging
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def measure_spect(spect):
    mean = np.mean(spec, axis=0)
    std = np.std(spec, axis=0)
    perc1 = np.percentile(spec, 1, axis=0)
    perc99 = np.percentile(spec, 99, axis=0)
    return (mean, std, perc1, perc99)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("spectpaths", nargs='+', type=str, help="Spectrum files (with wildcards)")
    parser.add_argument("--skipframes", type=int, default=0, help="Skip frontal frames (default=%(default)s)")
    parser.add_argument("--datasets", type=str, help="Data set (comma-separated subpaths, all if not specified)")
    parser.add_argument("--pca", type=float, default=0.95, help="PCA explained variance (default=%(default)s, zero for no PCA)")
    parser.add_argument("--skip", type=int, default=0, help="Skip frontal frames (default=%(default)s)")
    parser.add_argument("--clusters", type=int, default=2, help="Number of clusters (default=%(default)s)")
    parser.add_argument("--order", action='store_true', help="Order data within clusters by 1-component PCA")
    parser.add_argument("--outfile", type=str, help="Output file (.h5)")
    parser.add_argument("--outmeas", type=str, help="Output file for measures (.pkl)")
    args = parser.parse_args()
    
    if len(args.spectpaths) == 1 and os.path.splitext(args.spectpaths[0]) in ('.pck','.pkl'):
        # read from existing file
        measures = np.load(args.spectpaths[0])
    else:
        # read and measure spects
        measures = {}
        for path in args.spectpaths:
            for fn in glob(path):
#                print "Reading", fn
                path, item = fn.strip().split('/')[-2:]
                item = os.path.splitext(item)[0]
                with h5py.File(fn, 'r') as specf:
                    spec = specf['features'][args.skipframes:]
                measures[(path,item)] = measure_spect(spec)

    if args.outmeas:
        with file(args.outmeas, 'w') as outf:
            cPickle.dump(measures, outf)

    print "Done reading"

    # represent as arrays
    items = measures.keys() # subpath and filename
    allmns, allstd, allp01, allp99 = [np.asarray([measures[it][di] for it in items], dtype=float) for di in range(4)]
    items = np.asarray(items) # subpath and filename

    # normalize and stack data data
    alldata = np.hstack([
        StandardScaler().fit_transform(allmns),
        StandardScaler().fit_transform(allstd),
        StandardScaler().fit_transform(allp01),
        StandardScaler().fit_transform(allp99)
        ])

    # indices into data for given datasets
    datasets = args.datasets.split(',') if args.datasets is not None else set(items[:,0])
    ds_idxs = np.concatenate([np.where(items[:,0] == ds)[0] for ds in datasets])

    # cluster data
    data = alldata[ds_idxs]

    if args.pca:
        print "PCA"
        # compute PCA
        pca = PCA()
        pca.fit(data)
        pcacomps = np.min(np.where(np.add.accumulate(pca.explained_variance_ratio_) >= args.pca)[0])
        data = pca.transform(data)[:,:pcacomps]

    print "Clustering"
    # Hierarchical clustering with ward linkage
    clustering = AgglomerativeClustering(n_clusters=args.clusters)
    clidxs = clustering.fit_predict(data)
    
    if args.order:
        # Order data according to 1st PCA component
        pass
    
    print "Write to", args.outfile
    with h5py.File(args.outfile) as outf:
        outf['items'] = items        
        outf['mean'] = allmns[ds_idxs]        
        outf['std'] = allstd[ds_idxs]        
        outf['p01'] = allp01[ds_idxs]
        outf['p99'] = allp99[ds_idxs]        
        outf['clusters'] = clidxs+1
