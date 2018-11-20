#!/usr/bin/env python
import optparse, sys, os, logging
import time
from collections import defaultdict
from itertools import islice

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training with Expectation Maximization...\n")
bitext = [[sentence.strip().split() for sentence in pair] \
          for pair in islice(zip(open(f_data,encoding="utf8"), open(e_data,encoding="utf8")), opts.num_sents)]
#f_count = defaultdict(int)
#e_count = defaultdict(int)
#fe_count = defaultdict(int)

# f is the French word set
# e is the English word set
# f_count is the word count dictionary for French word set
# e_count is the word count dictionary for English word set
# N is the number of sentences
f = set()
e = set()
f_count = defaultdict(int)
e_count = defaultdict(int)
for pair in bitext:
    f = f.union(set(pair[0]))
    e = e.union(set(pair[1]))
    for f_i in set(pair[0]):
        f_count[f_i] += 1
    for e_j in set(pair[1]):
        e_count[e_j] += 1
N = len(bitext)

num_f = len(f_count)
num_e = len(e_count)

# add n smoothing
smooth_n = 0.01
vocab_N = 100000

def align(num, N, isReverse=False):
    '''
    num: size of target language word count dict
    N: number of sentences
    isReverse: if the translation direction is reversed
    '''
    k = 0
    # initialize theta uniformly
    theta = defaultdict(lambda: 1./num)
    while k < 5:
        k += 1
        tic = time.time()
        sys.stderr.write(f"Iteration {k}.................................\n")
        e_count = defaultdict(int)
        fe_count = defaultdict(int)
        if isReverse:
            for n in range(N):
                for f_i in bitext[n][1]:
                    Z = 0
                    for e_j in bitext[n][0]:
                        Z += theta[(f_i, e_j)]
                    for e_j in bitext[n][0]:
                        c = theta[(f_i, e_j)] / Z
                        fe_count[(f_i, e_j)] += c
                        e_count[e_j] += c
        else:
            for n in range(N):
                for f_i in bitext[n][0]:
                    Z = 0
                    for e_j in bitext[n][1]:
                        Z += theta[(f_i, e_j)]
                    for e_j in bitext[n][1]:
                        c = theta[(f_i, e_j)] / Z
                        fe_count[(f_i, e_j)] += c
                        e_count[e_j] += c
        for (f_i, e_j) in fe_count.keys():
            theta[(f_i, e_j)] = (fe_count[(f_i, e_j)] + smooth_n) / (e_count[e_j] + vocab_N * smooth_n)
        toc = time.time()
        sys.stderr.write(f"Iteration {k} finished. Time cost: {toc-tic}\n")
    return theta

# train the model
theta_e2f = align(num_f, N, False)
theta_f2e = align(num_e, N, True)        
        
# align
sys.stderr.write("Aligning...\n")

# delta is the threshold we used to decide whether to keep the alignment pair
delta = 0.08

for f, e in bitext:
    posterior_e2f = defaultdict(float)
    posterior_f2e = defaultdict(float)
    for i, f_i in enumerate(f):
        Z = 0
        for j, e_j in enumerate(e):
            Z += theta_e2f[(f_i, e_j)]
        for j, e_j in enumerate(e):
            posterior_e2f[(i,j)] = theta_e2f[(f_i, e_j)] / Z
    
    for j, e_j in enumerate(e):
        Z = 0
        for i, f_i in enumerate(f):
            Z += theta_f2e[(e_j, f_i)]
        for i, f_i in enumerate(f):
            posterior_f2e[(j,i)] = theta_f2e[(e_j, f_i)] / Z

    for pair in posterior_e2f.keys():
        posterior = posterior_e2f[pair] * posterior_f2e[(pair[1],pair[0])]
        if posterior > delta:
            sys.stdout.write(f"{pair[0]}-{pair[1]} ")
    sys.stdout.write("\n")