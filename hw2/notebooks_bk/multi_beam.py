from multiprocessing import Process,Pool, cpu_count
import datetime, time, random
import os, sys


def parallel_fn(Ve, phi, f, ext_limits, cipher, previous_score):
    Ht = []
    for e in Ve:
        phi_prime = copy.deepcopy(phi)
        new_map = {f: e}
        phi_prime.update(new_map)
        counts = len([v for k, v in phi_prime.items() if v == e])
        if counts <= ext_limits:
            Ht.append((phi_prime, score_new(cipher, phi, f, e, previous_score)))
    return Ht



def beam_search_new(cipher, ext_order, ext_limits=1, topn=1):

    # initialization
    Hs = [(defaultdict(dict), 0)]
    Ht = []
    cardinality = 0
    Ve = [chr(i) for i in range(97, 123, 1)]
    
    while cardinality < len(ext_order):
        f = ext_order[cardinality]
        print('Working on symbol: ', f, f'({cardinality+1})')
        
        mainStart = time.time()
        result = []
        p = Pool(cpu_count())
             
        for phi, previous_score in Hs: 
            result.append(p.apply_async(parallel_fn, args=(Ve, phi, f, ext_limits, cipher, previous_score,))) 
                            
        p.close() 
        p.join()  

        Ht = []
        for subp in result:
            Ht += subp.get()
    
        # prune the histogram
        mainEnd = time.time()
        print ('Running Time for this symbol: %0.2f seconds.' % (mainEnd-mainStart))
        Ht = sorted(Ht, key=lambda x:x[1], reverse=True)[:topn]    
        
        cardinality += 1
        Hs = copy.deepcopy(Ht)
        Ht.clear()
        print('Current score: ', Hs[0][1])
        
    return sorted(Hs, key=lambda x:x[1], reverse=True)