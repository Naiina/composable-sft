from nltk.corpus import wordnet as wn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Text2TextGenerationPipeline,AutoModelForTokenClassification, pipeline
from conllu import parse, parse_tree, parse_incr
from termcolor import colored
import numpy as np
import json
from tqdm import tqdm



wn_lang = ['als', 'arb', 'bul', 'cat', 'cmn', 'dan', 'ell', 'eng', 'eus',
'fin', 'fra', 'glg', 'heb', 'hrv', 'ind', 'isl', 'ita', 'ita_iwn',
'jpn', 'lit', 'nld', 'nno', 'nob', 'pol', 'por', 'ron', 'slk',
'slv', 'spa', 'swe', 'tha', 'zsm']
#wn.synsets(b'\xe7\x8a\xac'.decode('utf-8'), lang='jpn')
lang = "fra"



def get_supersenses(word):
    # Get all synsets for the word
    synsets = wn.synsets(word,lang = lang)
    #for s in synsets:
        #print(s.lemmas(lang))
    #exit()
    #print(synsets)
    l_def_and_supersenses = []
    # Loop through each synset and print its supersense information
    for synset in synsets:
        definition = synset.definition()
        supersense = synset.lexname()
        if "noun." in supersense:
            l_def_and_supersenses.append((definition,supersense))
    return l_def_and_supersenses

word = "banque"
l_def  =  get_supersenses(word)
print(l_def)