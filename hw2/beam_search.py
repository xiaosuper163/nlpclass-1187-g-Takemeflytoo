import os
from collections import defaultdict, Counter
import collections
import pprint
import math
import bz2
import copy
from random import shuffle
from ngram import LM
from nlm import *
from multiprocessing import Pool

class BeamSearch:
    def __init__(self, filename, lm=None, nlm=None, use_nlm=False):
        self.cipher = self.read_file(filename)
        self.cipher_desc = self.get_statistics(self.cipher, cipher=True)
        self.cipher_content = ''.join(self.cipher_desc['content'])
        self.plaintxt_desc = self.get_statistics(self.read_file("data/default.wiki.txt.bz2"), cipher=False)
        if lm is None:
            self.lm = LM("data/6-gram-wiki-char.lm.bz2", n=6, verbose=False)
        else:
            self.lm = lm
        self.use_nlm = use_nlm
        if self.use_nlm:
            if nlm is None:
                self.model = load_model("data/mlstm_ns.pt", cuda=True)
            else:
                self.model = nlm

        self.f = None
        self.phi = None
        self.old_score = None
        self.ext_limits = None

    def read_file(self, filename):
        if filename[-4:] == ".bz2":
            with bz2.open(filename, 'rt') as f:
                content = f.read()
                f.close()
        else:
            with open(filename, 'r', encoding="utf-8") as f:
                content = f.read()
                f.close()
        return content

    def get_statistics(self, content, cipher=True):
        stats = {}
        content = list(content)
        split_content = [x for x in content if x != '\n' and x!=' ']
        length = len(split_content)
        symbols = set(split_content)
        uniq_sym = len(list(symbols))
        freq = collections.Counter(split_content)
        rel_freq = {}
        for sym, frequency in freq.items():
            rel_freq[sym] = (frequency/length)*100
            
        if cipher:
            stats = {'content':split_content, 'length':length, 'vocab':list(symbols), 'vocab_length':uniq_sym, 'frequencies':freq, 'relative_freq':rel_freq}
        else:
            stats = {'length':length, 'vocab':list(symbols), 'vocab_length':uniq_sym, 'frequencies':freq, 'relative_freq':rel_freq}
        return stats

    """ Frequency Matching Heuristic
    new_map should contain only 1 mapping (by paper)
    """
    def fmh(self, new_map):
        sum = 0
        for f, e in new_map.items():
            sum += abs(math.log(self.cipher_desc['relative_freq'][f] / self.plaintxt_desc['relative_freq'][e]))
        return sum

    @staticmethod
    def replace_dict(string, *list_of_dict):
        for d in list_of_dict:
            for k, v in d.items():
                string = string.replace(k, v)
        return string
    
    # def score(self, old_score, phi_p, new_map):
    #     content =''.join(self.cipher_desc['content'])
    #     nlm_map = {}
    #     if not self.use_nlm:
    #         mask = {}
    #         for i in set(content):
    #             if i in phi_p:
    #                 mask.update({i : 'o'})
    #             else:
    #                 mask.update({i : '_'})
    #         mask = BeamSearch.replace_dict(content, mask)
    #     else:
    #         seq = ''
    #         mask = ''
    #         for char in content:
    #             if char in phi_p:
    #                 seq += phi_p[char]
    #                 mask += 'o'
    #             elif len(seq) > 8 and seq != '':
    #                 # Global Rest Cost Estimation
    #                 sample_chars = [i for i in next_chars(seq, True, self.model) if i[0] != ' '] 
    #                 shuffle(sample_chars)
    #                 sample_char = sample_chars[0][0]
    #                 nlm_map.update({char: sample_char})
    #                 seq += sample_char
    #                 mask += 'o'
    #             else:
    #                 seq = ''
    #                 mask += '_'
    #     new_score = self.lm.score_bitstring(BeamSearch.replace_dict(content, phi_p, nlm_map), mask)
    #     return old_score + new_score - self.fmh(new_map)

    @staticmethod
    def score(cipher, rf_f, rf_e, lm, old_score, phi_p, new_map):
        content = cipher 
        mask = {}
        for i in set(content):
            if i in phi_p:
                mask.update({i : 'o'})
            else:
                mask.update({i : '_'})
        mask = BeamSearch.replace_dict(content, mask)
        new_score = lm.score_bitstring(BeamSearch.replace_dict(content, phi_p), mask)
        fmh = 0
        for f, e in new_map.items():
            fmh += abs(math.log(rf_f / rf_e))
        return old_score + new_score - fmh
        
    def _score(self, e):
        print('process id:', os.getpid())
        phi_p = copy.deepcopy(self.phi)
        new_map = {self.f: e}
        phi_p.update(new_map)
        counts = len([v for k, v in phi_p.items() if v == e])
        if counts <= self.ext_limits:
            score_t = BeamSearch.score(self.cipher_content,
                     self.cipher_desc['relative_freq'][self.f],
                     self.plaintxt_desc['relative_freq'][e],
                     self.lm,
                     self.old_score, 
                     phi_p, 
                     new_map)
            return (phi_p, score_t)
        return None

    def beam_search(self, ext_limits=8, beam_size=1000):
        Hs, Ht = [], []
        cardinality = 0
        Hs.append(({}, 0))
        ext_order = [text for text, _ in sorted(self.cipher_desc['frequencies'].items(), key=lambda x: x[1], reverse=True)]
        while (cardinality < len(ext_order)):
            self.f = ext_order[cardinality]
            for phi, old_score in Hs:
                self.phi = phi
                self.old_score = old_score
                self.ext_limits = ext_limits
                print(phi, old_score, ext_limits)
                results = Pool(10).map(self._score, self.plaintxt_desc['vocab'])
                Ht += [result[0] for result in results if result[0] is not None]
                print(Ht)
            Ht = sorted(Ht, key=lambda x: x[1], reverse=True)[:beam_size]
            cardinality += 1
            Hs = copy.deepcopy(Ht)
            Ht.clear()
            print(cardinality)
        return sorted(Hs, key=lambda x:x[1], reverse=True)[:1]


if __name__ == "__main__":
    b = BeamSearch("data/cipher.txt")
    x = b.beam_search(3, 10)
    print(x)