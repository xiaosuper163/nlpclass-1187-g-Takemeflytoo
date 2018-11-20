# usage: theta = LLR_initialization(bitext)

import math

# f = t, e = s
def LLR_initialization(bitext, exp = 1, isReverse=False): 
    if isReverse:
        f=1
        e=0
    else:
        f=0
        e=1
    LLRs = defaultdict(float)
    sum_LLRs = defaultdict(float)
    
    t = set()
    s = set()
    
    t_count = defaultdict(int) # occurance of each words in t
    s_count = defaultdict(int) # occurance of each words in s
    ts_count = defaultdict(int) # occurance of each (t, s) pair
    
    for pair in bitext:
        t = t.union(set(pair[f]))
        s = s.union(set(pair[e]))
        
        for t_i in set(pair[f]):
            t_count[t_i] += 1
            for s_i in set(pair[e]):
                ts_count[(t_i, s_i)] += 1
            
        for s_i in set(pair[e]):
            s_count[s_i] += 1
      
    sum_ts_count = sum(ts_count.values())
    sum_t_count = sum(t_count.values())
    sum_s_count = sum(s_count.values())

    for (t_j, s_j) in ts_count.keys():
        p_ts = ts_count[(t_j, s_j)] / sum_ts_count
        p_t = t_count[t_j] / sum_t_count
        p_s = s_count[s_j] / sum_s_count
        
        if p_ts > p_t * p_s:
            # calculate LLR
            
            # (t? == t) and (s? == s)
            count_for_LLR_1 = ts_count[(t_j, s_j)]
            if count_for_LLR_1 != 0:
                LLR_1 = count_for_LLR_1 * math.log10((p_ts/p_s)/p_t)
            
            # (t? == t) and (s? == not s)
            count_for_LLR_2 = (t_count[t_j] - ts_count[(t_j, s_j)])
            if count_for_LLR_2 != 0:
                LLR_2 = count_for_LLR_2 * math.log10(((count_for_LLR_2 / sum_ts_count)/(1-p_s))/p_t)
            
            # (t? == not t) and (s? == s)
            count_for_LLR_3 = (s_count[s_j] - ts_count[(t_j, s_j)]) 
            if count_for_LLR_3 != 0:
                LLR_3 = count_for_LLR_3 * math.log10(((count_for_LLR_3 / sum_ts_count)/p_s)/(1-p_t))
            
            # (t? == not t) and (s? == not s)
            count_for_LLR_4 = sum_ts_count - count_for_LLR_1 - count_for_LLR_2 - count_for_LLR_3
            if count_for_LLR_4 != 0:
                LLR_4 = count_for_LLR_4 * math.log10(((count_for_LLR_4 / sum_ts_count)/(1-p_s))/(1-p_t))
            
            LLRs[(t_j, s_j)] = LLR_1 + LLR_2 + LLR_3 + LLR_4
        else:
            # p(t, s) <= p(t) * p(s), so initialize to uniform distribution
            LLRs[(t_j, s_j)] = 1.0 / len(t) 
            
        # for each source word, compute the sum of the LLR scores over all target words
        sum_LLRs[s_j] += LLRs[(t_j, s_j)]
            
       
    
    # then divide every LLR score by the single largest of these sums
                  
    largest = max(sum_LLRs.values()) 

    for i in LLRs.keys():
        LLRs[i] = LLRs[i] / largest
        
    # raise each LLR score to an empirically optimized exponent
    for i in LLRs.keys():
        LLRs[i] = LLRs[i] ** exp
               
    return LLRs
