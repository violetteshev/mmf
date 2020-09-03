import logging
import pickle
import json
import os
import numpy as np


logger = logging.getLogger(__name__)


class Ontology(object):
    def __init__(self, fpath, max_len=10):
        super().__init__()

        with open(os.path.join(fpath, 'entity_embeddings.pkl'), 'rb') as f:
            self._embeddings = pickle.load(f)
        
        with open(os.path.join(fpath, 'token2id.pkl'), 'rb') as f:
            self._token2id = pickle.load(f)
        
        self.max_len = max_len
        self.embed_dim = len(self._embeddings[0])
        self._entities = {}
        self._stop = '[END]'
        
        for entity in self._token2id:
            words = entity.split(' ')
            
            if len(words) > max_len:
                continue
            
            if not words[0] in self._entities:
                self._entities[words[0]] = {}

            if len(words) == 1:
                self._entities[words[0]][self._stop] = {}
            else:
                curr_node = self._entities[words[0]]
                for word in words[1:]:
                    if not word in curr_node:
                        curr_node[word] = {}
                    curr_node = curr_node[word]
                curr_node[self._stop] = {}
        logger.info('Ontology is built.')

    def _find_entity(self, sent, enitites):
        if sent == []:
            return []

        word = sent[0]
        if not word in enitites:
            return []
        else:
            res = self._find_entity(sent[1:], enitites[word])
            if (res == []) and not (self._stop in enitites[word]):
                return []
            else:
                return [word] + res 

    def extract_entities(self, sent):
        idx = 0
        starts = []
        lens = []
        words = []
        while idx < len(sent):
            res = self._find_entity(sent[idx:], self._entities)
            if res == []:
                idx += 1
                continue

            l = len(res)
            starts.append(idx)
            lens.append(l)
            words.append(' '.join(res))
            idx += l
            
        embeddings = [self._embeddings[self._token2id[w]] for w in words]
        ids = [[x for x in range(s, s+l)] for s,l in zip(starts, lens)]
        return ids, embeddings, words
