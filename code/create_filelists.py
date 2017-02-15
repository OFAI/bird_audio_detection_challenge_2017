#!/usr/bin/env python
# -*- coding: utf-8

import sys
import random
from itertools import izip

import argparse
parser = argparse.ArgumentParser(description='Create filelists')

parser.add_argument('path', type=str, help='path to filelists')
parser.add_argument('filelists', type=str, nargs='+', help='filelists')
parser.add_argument('--out', default="./%(fold)s_%(num)i", help='output file name (template parameter: fold, num)')
parser.add_argument('--num', type=int, default=1, help='Number of folds (default=%(default)s)')
parser.add_argument('--folds', type=str, default='train=0.8,val=0.2', help='Fold names and shares')
parser.add_argument('--seed', type=int, help='Set random seed (default=%(default)s)')
parser.add_argument('--log', action='store_true', help='Log to console')
parser.add_argument('--clusterfile', type=str, default='', help='Cluster file')
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

if args.clusterfile:
    import h5py
    with h5py.File(args.clusterfile, 'r') as f:
        cluster_items = f['items'].value
        cluster_idxs = f['clusters'].value
        clusters = max(cluster_idxs)+1
        cluster_dict = dict(izip(cluster_items, cluster_idxs))
else:
    cluster_dict = None
    clusters = 1

fileids = []
for filelist in args.filelists:
    with open("%s/%s.csv"%(args.path, filelist), 'r') as f:
        for ln in f:
            if not ln.startswith("itemid"):
                item = ln.strip().split(",")[0]
                fileids.append("%s/%s.wav"%(filelist, item))

random.shuffle(fileids)

# create folds
folds = [(name, float(share)) for name, share in (f.split('=', 1) for f in args.folds.split(','))]
total = sum(s for _,s in folds)

num = args.num
parts = [fileids[n::num] for n in range(num)]
for n in range(num):
    items = [i for p in range(len(parts)) for i in parts[(p+n)%num]]

    # write fold files
    taken = 0.
    start = 0
    for name, share in folds:
        taken += share/total*len(items)
        end = int(taken+0.5)
        folditems = items[start:end]
        start = end

        import pdb
        pdb.set_trace()

        for cli in range(clusters):
            if clusters > 1:
                clitems = [f for f in folditems if cluster_dict[f] == cli]
            else:
                clitems = folditems
            
            outname = args.out%dict(fold=name, num=n+1, cluster=cli)
            with open(outname, 'w') as f:
                f.writelines("%s\n"%f for f in clitems)

            if args.log:
                print >>sys.stderr, "Wrote %s_%i_%i with %i files (share=%.3f)"%(name, n+1, cli, len(folditems), len(folditems)/float(len(items)))

