#!/usr/bin/env python
import optparse, sys, os, logging
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
# N is the number of sentences
f = set()
e = set()
f_count = defaultdict(int)
for pair in bitext:
    f = f.union(set(pair[0]))
    e = e.union(set(pair[1]))
    for f_i in set(pair[0]):
        f_count[f_i] += 1
N = len(bitext)

# train the model
k = 0
# initialize theta uniformly
num_f = len(f_count)
theta = defaultdict(lambda: 1./num_f)
while k < 5:
    k += 1
    sys.stderr.write(f"Iteration {k}.................................\n")
    e_count = defaultdict(int)
    fe_count = defaultdict(int)
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
        theta[(f_i, e_j)] = fe_count[(f_i, e_j)] / e_count[e_j]
        
        
# align
sys.stderr.write("Aligning...\n")
for f, e in bitext:
    for i in range(len(f)):
        f_i = f[i]
        bestp = 0
        bestj = 0
        for j in range(len(e)):
            e_j = e[j]
            if theta[(f_i, e_j)] > bestp:
                bestp = theta[(f_i, e_j)]
                bestj = j
        sys.stdout.write(f"{i}-{bestj} ")
    sys.stdout.write("\n")