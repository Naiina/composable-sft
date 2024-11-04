#1) load model -> use stats.py
#2) load UD file
#3) anim_tags = [1,0,0,2,-100,...] with 0, Inanimate, 1 animal 2 human thanks to the model
#4) gram_tags = [0,1,-100]: 0 if subj, 1 if obj, -100 otherwise
#5) POS = [0,-100] 0 is noun, 1 if proper noun 2 if pronoun
#5) reuse functions from animacy_classifier to compute avg position in sentence and animacy of subject and objects
from conllu import parse
from collections import defaultdict
from tqdm import tqdm
#from nltk.corpus import wordnet as wn
import json
import pandas as pd
import argparse
import os
#from datasets import load_dataset
from datasets import Dataset, DatasetDict
import numpy as np
from sft import SFT
import math as m
#from transformers import (
#    AutoModelForTokenClassification,
##    AutoTokenizer,
#    AutoModelForSeq2SeqLM, 
#    Text2TextGenerationPipeline
#)
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency as chi2_contingency
#import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str)

args = parser.parse_args()
lang = args.lang

print("--------------------------------------------------------")
print(lang)

l_lang = ["bm","bn","bxr","yue","zh","cs","myv","et","fo","de"] #"en","fr","eu","ar"
#lang = "ar"
#bm: only test file  
#ar: The size of tensor a (716) must match the size of tensor b (512) at non-singleton dimension 1 at elem 15539
#bn '../../ud-treebanks-v2.14/UD_Bengali-BRU/bn_bru-ud-train.conllu'
#yue: No such file or directory: '../../ud-treebanks-v2.14/UD_Cantonese-HK/yue_hk-ud-train.conllu'



dict_files = {"en":"UD_English-GUM/en_gum-ud-train.conllu",
              "fr":"UD_French-GSD/fr_gsd-ud-dev.conllu",
              "eu":"UD_Basque-BDT/eu_bdt-ud-train.conllu",
              "ar":"UD_Arabic-NYUAD/ar_nyuad-ud-train.conllu",
              "bm":"UD_Bambara-CRB/bm_crb-ud-test.conllu",
              "bn":"UD_Bengali-BRU/bn_bru-ud-test.conllu",
              "bxr":"UD_Buryat-BDT/bxr_bdt-ud-train.conllu",
              "yue":"UD_Cantonese-HK/yue_hk-ud-test.conllu",
              "zh":"UD_Chinese-GSD/zh_gsd-ud-train.conllu",
              "cs":"UD_Czech-PDT/cs_pdt-ud-train.conllu",
              "myv":"UD_Erzya-JR/myv_jr-ud-train.conllu",
              "et":"UD_Estonian-EDT/et_edt-ud-train.conllu",
              "fo":"UD_Faroese-FarPaHC/fo_farpahc-ud-train.conllu",
              "de":"UD_German-HDT/de_hdt-ud-train.conllu",
              "es": "UD_Spanish-AnCora/es_ancora-ud-train.conllu",
              "ja": "UD_Japanese-GSD/ja_gsd-ud-train.conllu",
              "ko": "UD_Korean-Kaist/ko_kaist-ud-train.conllu",
              
              }
model_name_or_path = "models/animacy/en/xlwsdCorpus/checkpoint-2000"
tokenizer_name_or_path = "bert-base-multilingual-cased"
if lang == None:
    UD_file = None
else:
    UD_file = "ud-treebanks-v2.14/"+dict_files[lang]
    json_file = "UD_data_anim/UD_annot_data_"+lang+".json"
    json_file_obl = "UD_data_anim/UD_annot_data_"+lang+"obl.json"
    json_file_stats = "UD_data_anim/stats_"+lang+".json"
    json_file_stats_obl = "UD_data_anim/stats_"+lang+"obl.json"
print(UD_file)

"""
tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            padding=True,
            truncation=True
        )

model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

lang_sft = SFT('cambridgeltl/mbert-lang-sft-'+lang+'-small')
lang_sft.apply(model, with_abs=False)


pipe = Text2TextGenerationPipeline(
    model = AutoModelForSeq2SeqLM.from_pretrained("jpelhaw/t5-word-sense-disambiguation"),
    tokenizer = AutoTokenizer.from_pretrained("jpelhaw/t5-word-sense-disambiguation", 
            padding=True,
            truncation=True)
)"""

def tokenize_and_align(l,word_ids):        
        previous_word_idx = None
        l_aligned = []
        for word_idx in word_ids:
            if word_idx is None:
                l_aligned.append((-100,word_idx))
            elif word_idx != previous_word_idx:
                l_aligned.append((l[word_idx],word_idx))
            else:
                l_aligned.append((-100,word_idx))
            previous_word_idx = word_idx

        return l_aligned

def predict(elem):
    inputs = tokenizer(elem, return_tensors="pt",is_split_into_words=True)
    #word_ids = inputs.word_ids()
    out = model(**inputs)
    logits = out.logits

    # Apply softmax to get probabilities (optional, for understanding)
    #probabilities = F.softmax(logits, dim=-1)

    # Get the predicted class indices
    predicted_class_indices = torch.argmax(logits, dim=-1)

    # Decode the tokens back to words and map predictions to these words
    #tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    #d_idx_to_lab = {0:"N",1:"A",2:"H"}
    predictions = [model.config.id2label[idx.item()] for idx in predicted_class_indices[0]]
    pred = predicted_class_indices[0]
    return pred.detach().tolist()

def dummy_model(tokens):
    n = len(tokens)
    l_anim_tags = [0]*n
    l_anim_tags[0] = 0
    l_anim_tags[-1] = 2
    return l_anim_tags

def get_supersenses(word):
    # Get all synsets for the word
    synsets = wn.synsets(word)
    l_def_and_supersenses = []
    # Loop through each synset and print its supersense information
    for synset in synsets:
        definition = synset.definition()
        supersense = synset.lexname()
        if "noun." in supersense:
            l_def_and_supersenses.append((definition,supersense))
    return l_def_and_supersenses


def get_label_nn(categ):
    """
    H: Human A: animal N: rest (inanimate)
    """
    #l_categ = ['noun.animal','noun.person']
    if categ == 'noun.animal':
        return 0
    elif categ == 'noun.person':
        return 1
    else: 
        return 2


def get_in_context_supersense(word,context,pipe):
    
    l_def_and_supersenses = get_supersenses(word)
    set_categ = set(dict(l_def_and_supersenses).values())
    inter = set_categ & {'noun.animal','noun.person'} 
    if word == "adult":
        print(inter)
    if inter == set():
        return 2
    elif set_categ=={'noun.animal'}:
        return 0
    elif set_categ=={'noun.person'}:
        return 1
    else:
        question = '''question: which description describes the word "'''+word+'''" best in the following context? \n'''
        all_desript = ''''''

        for desc,s in l_def_and_supersenses:
            Desc = ''' " %s ", ''' %desc
            all_desript = all_desript+ Desc

        descriptions='''desription: [  %s ] \n''' %all_desript
        context = '''context: %s ''' %context
        input = question+descriptions+context


        output = pipe(input)[0]['generated_text']

        for defin,categ in l_def_and_supersenses:
            if defin[:15]==output[:15]:
                return get_label_nn(categ)

    return -100

def wsd(sentence,tokens,pos):
    labels = []
    for i,elem in enumerate(pos):
        if elem != -100:
            word = tokens[i]
            
            lab = get_in_context_supersense(word,sentence,pipe)
            labels.append(lab)
        else:
            labels.append(-100)
    return labels

def find_roots(l,pro,deprel):
    l_deprel = ("csubj","xcomp","ccomp","acl","parataxis")
    l_waiting = []
    d_roots = {}
    #d_roots_v = {}
    #"conj"

    #find the root
    for d_word in l:
        if d_word["head"] == 0:
            id = d_word["id"]
            l_tree_roots_idx = [id]
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
            elif pro and "PRON" == d_word["upos"] and type(d_word["feats"]) == dict and "Person" in d_word["feats"] and d_word["feats"]["Person"] in ["1","2"]:
                anim = "P"
            else:
                anim = None
            g = d_word["deprel"]

            d_roots[d_word["form"]+'0']=([id],[anim],[g],[])
            #if d_word["upos"] == "VERB":
            #    d_roots_v[d_word["form"]+'0']=([id],[anim],[g])
        else:
            l_waiting.append(d_word)

    #Find the "roots" of the other clauses
    changes = True
    while changes : 
        changes = False
        for i,d_word in enumerate(l_waiting):
            rel = d_word["deprel"].split(":")[0]
            if rel == 'conj' and d_word["head"] in l_tree_roots_idx or rel in l_deprel:
                #print(d_word["form"],rel)
                id = d_word["id"]
                l_tree_roots_idx.append(id)
                
                if "NOUN" == d_word["upos"]:
                    anim = d_word["misc"]["ANIMACY"]
                elif pro and "PRON" == d_word["upos"] and type(d_word["feats"]) == dict and "Person" in d_word["feats"] and d_word["feats"]["Person"] in ["1","2"]:
                    anim = "P"
                else:
                    anim = None
                if deprel:

                    d_roots[d_word["form"]]=([id],[anim],[rel],[])
                    #if d_word["upos"] == "VERB":
                    #    d_roots_v[d_word["form"]]=([id],[anim],[rel])
                else:
                    d_roots[d_word["form"]]=([id],[anim])
                    #if d_word["upos"] == "VERB":
                    #    d_roots_v[d_word["form"]]=([id],[anim])
                l_waiting.pop(i)
                changes = True

    return d_roots,l_tree_roots_idx

        


def create_subtrees_lists(l,pro,deprel = False,direct_arg_only = False):
    l_waiting_idx = []
    l_waiting_anim = []
    l_waiting_head = []
    l_waiting_gram = []
    # get roots of each subtree
    d_subtrees,l_tree_roots_idx = find_roots(l,pro,deprel)
    l_gram = ["obj","nsubj","obl"]

    
    for d_word in l:
        idx = d_word["id"]
        head = d_word["head"]
        upos = d_word["upos"]
        gram = d_word["deprel"].split(":")[0]
        if upos != "PUNCT":
            if "NOUN" == upos:
                anim = d_word["misc"]["ANIMACY"]
            elif pro and "PRON" == upos and type(d_word["feats"]) == dict and "Person" in d_word["feats"] and d_word["feats"]["Person"] in ["1","2"]:
                anim = "P"
            else:
                anim = None
            if idx not in l_tree_roots_idx:
                l_waiting_idx.append(idx)
                l_waiting_anim.append(anim)
                l_waiting_head.append(head)
                l_waiting_gram.append(gram)
    ii = 0
    max_it = len(l_waiting_idx)
    #print("anim",l_waiting_anim )
    
    while l_waiting_idx!=[]:
        i = l_waiting_idx.pop(0)
        a = l_waiting_anim.pop(0)
        h = l_waiting_head.pop(0)
        g = l_waiting_gram.pop(0)
        
        found = False
        
        # look up if already in a subtree
        for k,v in d_subtrees.items():
            
            #print(str(i)+" look up "+str(h)+" in ",k,v)
            if h in v[0]: 
                
                d_subtrees[k][0].append(i)
                d_subtrees[k][1].append(a)
                d_subtrees[k][2].append(g)
                
                if direct_arg_only:
                    if g in l_gram:
                        d_subtrees[k][3].append(d_subtrees[k][0].index(i))  
                else:
                    d_subtrees[k][2].append(g)

                found = True
                ii = 0
                max_it = max_it - 1
                break
        # if not found, put back at the end of the waiting lists
        if not found:
            ii+=1
            l_waiting_idx.append(i)
            l_waiting_anim.append(a)
            l_waiting_head.append(h)
            l_waiting_gram.append(g)
        #print(l_waiting_idx)
        
        
        if ii > max_it+1 :
            break  
    return d_subtrees
            



def annotate_UD_file(UD_file,json_file,max_len=-1,push_to_hub = False,hub_name = None,lang = "en"):
    if lang == "ar":
        max_len = 15500
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idx = 0
        
    l_sent = []
    ll_tokens = []
    ll_pos = []
    ll_gram = []
    ll_anim_lab_pred = []
    ll_anim_lab_wsd = []
    l_sent_len = []
    
    for elem in tqdm(dd_data_UD):
        idx +=1
        if max_len >0:
            if idx >max_len:
                break
        
        text = elem.metadata['text']
        l_sent.append(text)
        
        l = list(elem)
        l_pos = []
        l_gram = []
        l_tokens = []
        l_sent_len.append(len(l))
        for d_word in l:
            word = d_word["form"]
            l_tokens.append(word)
            pos = -100
            gram = -100
            if "NOUN" in d_word.values():
                pos = 0 
            if "PROPN" in d_word.values():
                pos = 1
            if "PRON" in d_word.values():
                pos = 2
            if "nsubj" in d_word.values():
                gram = 0
            if "obj" in d_word.values():
                gram = 1
            l_pos.append(pos)
            l_gram.append(gram)
        ll_tokens.append(l_tokens)
        ll_pos.append(l_pos)
        ll_gram.append(l_gram)
        #l_anim_raw = dummy_model(l_tokens)
        l_anim_lab_pred = predict(l_tokens)
        if lang == "en":
            l_anim_lab_wsd = wsd(text,l_tokens,l_pos)
            ll_anim_lab_wsd.append(l_anim_lab_wsd)
        #l_anim_lab = [l_anim_raw[k] if l_pos[k]!=-100 else -100 for k in range(len(l_pos))]
        ll_anim_lab_pred.append(l_anim_lab_pred)
    if lang == "en":
        d_data = {"sent":l_sent,"tokens":ll_tokens,"labels_pred":ll_anim_lab_pred,"labels_wsd":ll_anim_lab_wsd, "pos":ll_pos,"gram":ll_gram,"l_sent_len":l_sent_len}
    else:
        d_data = {"sent":l_sent,"tokens":ll_tokens,"labels_pred":ll_anim_lab_pred,"pos":ll_pos,"gram":ll_gram,"l_sent_len":l_sent_len}
   

    with open(json_file, 'w') as json_file:
        json.dump(d_data, json_file)

    if push_to_hub or lang=="de":
        df_data = pd.DataFrame.from_dict(d_data)
        dataset = Dataset.from_pandas(df_data)
        data = DatasetDict({
                'train': dataset})
        data.push_to_hub(hub_name)

def annotate_UD_file_obl(UD_file,json_file,max_len=-1,push_to_hub = False,hub_name = None,lang = "en"):
    if lang == "ar":
        max_len = 15500
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idx = 0
    ll_gram = []
    f = open(json_file)
    data = json.load(f)
    
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idx +=1
        if max_len >0:
            if idx >max_len:
                break
        
        text = elem.metadata['text']
        
        l = list(elem)
        l_gram = []
        for d_word in l:
            gram = -100
            if "nsubj" in d_word.values():
                gram = 0
            if "obj" in d_word.values():
                gram = 1
            if "obl" in d_word.values():
                gram = 2
            l_gram.append(gram)
        ll_gram.append(l_gram)

    data["gram"] = ll_gram
   
    outfile = json_file[:-5]+"obl.json"
    print(outfile)
    with open(outfile, 'w') as json_file:
        json.dump(data, json_file)

    if push_to_hub or lang=="de":
        df_data = pd.DataFrame.from_dict(data)
        dataset = Dataset.from_pandas(df_data)
        data = DatasetDict({
                'train': dataset})
        data.push_to_hub(hub_name)

class Word:
    #the voice is only set for subjects
    def __init__(self, word,head, gram, id,sent_id,sent_len,anim=None,voice=None):
        self.word = word
        self.head = head
        self.gram = gram
        self.sent_id = sent_id
        self.sent_len = sent_len
        if type(id) == tuple: 
            self.pos_in_sent = id[0]-1
        else:
            self.pos_in_sent = id-1
        self.pos_aligned = -1
        self.animacy = anim
        #if gram != "verb":
        #    tok,pred_lab = self.get_UD_info(d_data)
        #    self.set_animacy(tok,pred_lab)
        if gram == "subject":
            self.voice = voice

    def get_UD_info(self,d_data):
        sent_id = self.sent_id
        tok = d_data["tokens"][sent_id]
        pred_lab = d_data["labels_pred"][sent_id]
        return tok,pred_lab


    #def set_animacy(self,tok,pred_lab):
    #    d_id_2_lab = {2:"N",1:"H",0:"A"}
    #    #tok,pred_lab = self.get_UD_info(d_data)
    #    tokenized_inputs = tokenizer(tok,is_split_into_words=True)   
    #    word_ids = tokenized_inputs.word_ids()
    #    #print(tok,word_ids)
    #    pos_al = word_ids.index(self.pos_in_sent)
    #    self.pos_aligned = pos_al
    #    self.animacy = d_id_2_lab[pred_lab[pos_al]]
    def __str__(self):
        return "word:"+str(self.word)+" head:"+str(self.head)+" anim:"+str(self.animacy)+" position_in_sent:"+str(self.pos_in_sent)+" aligned_pos:"+str(self.pos_aligned)+" sent_id:"+str(self.sent_id)

     


def compute_statistics(UD_file,task,max_len=-1,lang = "en"):

    if lang == "ar":
        max_len = 15000

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())

    idxx = 0
    d_anim_passif = {"N":0,"A":0,"H":0}
    d_anim_actif = {"N":0,"A":0,"H":0}
    d_anim_passif_no_obj = {"N":0,"A":0,"H":0}
    d_anim_actif_no_obj = {"N":0,"A":0,"H":0}
    d_sent_len_subj = {"N":[],"A":[],"H":[]}
    d_sent_len = {"N":[],"A":[],"H":[]}
    d_rel_pos = {"N":[],"A":[],"H":[]}
    d_pos = {"N":[],"A":[],"H":[]}
    d_first_pos_of_v = {"N":0,"A":0,"H":0}
    d_second_pos_of_v = {"N":0,"A":0,"H":0}
    d_third_pos_of_v = {"N":0,"A":0,"H":0}
    d_act_obj = {"N":0,"A":0,"H":0}
    d_passif_subj = {"N":0,"A":0,"H":0}
    d_passif_subj_with_obl = {"N":0,"A":0,"H":0}
    d_passif_subj_no_obl = {"N":0,"A":0,"H":0}
    d_corr = {"idx":[],"sent_id":[],"text":[],"sent_len":[],"word":[],"head":[],"anim":[],"gram":[],"rel_pos":[],"voice":[],"def":[],"num":[]}
    nb_subj_act_not_noun = 0
    nb_subj_pas_not_noun = 0
    nb_subj = 0
    nb_obj_noun = 0
    not_in_head = 0

    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx >max_len:
                break
        
        
        text = elem.metadata['text']
        print(text)
        sent_len = len(text.split())
        #print(text.split())
        
        l = list(elem)
        l_subj = []
        l_obj = []
        l_obl = []
        l_verb = []
        l_det = []
        l_act_head = []
        l_pass_obl = []
        l_head_subj_pass_not_noun = []
        l_head_subj_act_not_noun = []
        #print(text)
        for d_word in l:
            #d_subtrees = create_subtrees_lists(l)
            #print(d_subtrees)

            head = d_word["head"]
            idx = d_word["id"]
            word = d_word["form"]
            voice = None
            gram = None
            num = None
            if "DET" == d_word["upos"]:
                #print(word,d_word["feats"])
                if "Definite" in d_word["feats"].keys():
                    defin = d_word["feats"]["Definite"]
                    det = (head,defin)
                    l_det.append(det)
            if "VERB" == d_word["upos"]:
                V = Word(word,head,"verb",idx,i,sent_len)
                l_verb.append(V)

                
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]

                if d_word["feats"]!= None:
                    if "Number" in d_word["feats"].keys():
                        num = d_word["feats"]["Number"]

                if "nsubj:pass" in d_word.values():
                    voice = "passif"
                    gram = "subj"
                if "nsubj" in d_word.values():
                    voice = "actif"
                    gram = "subj"
                if voice != None:
                    W = Word(word,head,"subject",idx,i,sent_len,anim,voice)
                    l_subj.append(W)
                if "obj" in d_word["deprel"]:
                    gram = "obj"
                    W = Word(word,head,"object",idx,i,sent_len,anim)
                    l_obj.append(W)
                    nb_obj_noun += 1
                if "obl" in d_word["deprel"]:
                    gram = "obl"
                    W = Word(word,head,"oblique",idx,i,sent_len,anim)
                    l_obl.append(W)
            else:
                if "nsubj:pass" in d_word.values():
                    l_head_subj_pass_not_noun.append(head)
                if "nsubj" in d_word.values():
                    l_head_subj_act_not_noun.append(head)
            #print(" word:",word," gram:",gram)
            if task == "correlation":
                if gram != None:
                    d_corr["idx"].append(idx)
                    d_corr["sent_id"].append(i)
                    d_corr["sent_len"].append(sent_len)
                    d_corr["text"].append(text)
                    d_corr["word"].append(word)
                    d_corr["head"].append(head)
                    d_corr["anim"].append(W.animacy)
                    d_corr["gram"].append(gram)
                    d_corr["rel_pos"].append(None)
                    d_corr["voice"].append(voice)
                    d_corr["num"].append(num)
                    d_corr["def"].append(None)
        if task == "correlation":
            #print("list det",l_det)
            #print("list sent id",d_corr["sent_id"])
            try:
                start = d_corr["sent_id"].index(i)
                #print("start",start)
                for i,idx_noun in enumerate(d_corr["idx"][start:]):
                    #print("word:", d_corr["word"][i+start]," word_id:",d_corr["idx"][i+start])
                    det = find_det(idx_noun,l_det)
                    #print("det", det)
                    if det != None:
                        d_corr["def"][i+start] = det
            except ValueError:
                pass
        if task == "suj_act_pas":
            for subj in l_subj:
                if subj.voice == "passif":
                    d_anim_passif[subj.animacy]+=1
                if subj.voice == "actif":
                    d_anim_actif[subj.animacy]+=1
        if task == "rel_pos_per_verb":
            for V in l_verb:
                l_dep = find_verb_dep(V,l_subj,l_obj,l_obl)
                #if len(l_dep) == 0:
                    #print(V.word,text,V.pos_in_sent+1)
                    #print_l_words(l_obj)
                if len(l_dep)>0:
                    d_first_pos_of_v[l_dep[0][1].animacy]+=1
                if len(l_dep)>1:
                    d_second_pos_of_v[l_dep[1][1].animacy]+=1
                if len(l_dep)>2:
                    d_third_pos_of_v[l_dep[2][1].animacy]+=1
        if task == "avg_sent_len":
            for w in l_subj:
                d_sent_len_subj[w.animacy].append(w.sent_len)
                d_sent_len[w.animacy].append(w.sent_len)
            for w in l_obj+l_obl:
                d_sent_len[w.animacy].append(w.sent_len)
        if task == "pos":
            for w in l_subj+l_obj+l_obl:
                d_rel_pos[w.animacy].append((w.pos_in_sent+1)/w.sent_len)
                d_pos[w.animacy].append((w.pos_in_sent+1))

        if task == "obj_act_subj_pass":
            #print(text)
            #print_l_words(l_subj)
            l_obl_head = [o.head for o in l_obl]
            l_subj_head = [s.head for s in l_subj]
            l_obj_head = [o.head for o in l_obj]
            nb_subj = nb_subj + len(l_subj)
            for w in l_subj:
                if w.voice == "passif":
                    d_passif_subj[w.animacy]+=1
                    if w.head in l_obl_head:
                        d_passif_subj_with_obl[w.animacy]+=1
                    else:
                        d_passif_subj_no_obl[w.animacy]+=1
                else:
                    l_act_head.append(w.head)
            for elem in l_obj:
                if elem.head in l_act_head+l_head_subj_act_not_noun+l_head_subj_pass_not_noun:
                    d_act_obj[elem.animacy]+=1
                else: 
                    not_in_head+=1
            


    if task == "suj_act_pas": 
        print(d_anim_actif,d_anim_passif,d_anim_actif_no_obj,d_anim_passif_no_obj)
    if task == "avg_sent_len":
        print("Average length sentence with the subject being:")
        for key,val in d_sent_len_subj.items():
            print(key,round(np.mean(val),2))
        print("Average length sentence containing:")
        for key,val in d_sent_len.items():
            print(key,round(np.mean(val),2))
    if task == "pos":
        with open("UD_data_anim/stats_"+lang+"_rel.json", 'w') as json_file:
            json.dump(d_rel_pos, json_file)
        print("relative position")
        for key,val in d_rel_pos.items():
            print(key,round(np.mean(val)*100,2))
        print("absolute position")
        for key,val in d_pos.items():
            print(key,round(np.mean(val),2))
    if task == "rel_pos_per_verb":
        print("first")
        print(d_first_pos_of_v)
        prop(d_first_pos_of_v)
        print("second")
        print(d_second_pos_of_v)
        prop(d_second_pos_of_v)
        print("third")
        print(d_third_pos_of_v)
        prop(d_third_pos_of_v)
    if task == "obj_act_subj_pass":
        print("passive subjects:",d_passif_subj)
        prop(d_passif_subj)
        print("active objects:",d_act_obj)
        prop(d_act_obj)
        print("nb_sub not noun:",nb_subj_act_not_noun)
        #print("nb_sub not noun:",nb_subj_pas_not_noun)
        #d_passif_subj["H"]=d_passif_subj["H"]+nb_subj_pas_not_noun
        #d_act_obj["H"]=d_passif_subj["H"]+nb_subj_pas_not_noun
        #pop(d_passif_subj)
        print("pass with obl",d_passif_subj_with_obl)
        prop(d_passif_subj_with_obl)
        print("pass with obl",d_passif_subj_no_obl)
        prop(d_passif_subj_no_obl)
        #print(nb_subj, nb_subj_act_not_noun,nb_subj_pas_not_noun)
        #print(nb_obj_noun)
        print("not in head",not_in_head)
    if task == "correlation":
        df = pd.DataFrame.from_dict(d_corr)  
        df.to_csv("UD_data_anim/corr_df_"+lang+".csv") 




def pass_subj_act_obj(UD_file,max_len,Gram):
    idxx = 0
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())

    d_act_obj = {"N":0,"A":0,"H":0}
    d_passif_subj = {"N":0,"A":0,"H":0}


    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx >max_len:
                break
        
        
        text = elem.metadata['text']
        l = list(elem)
        print(text)

        d_subtrees = create_subtrees_lists(l,True,Gram)
        print(l)

def relative_order_MI(UD_file,max_len,Gram,direct_arg_only):
    idxx = 0
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    joint_counts = defaultdict(lambda: defaultdict(int))
    animacy_counts = defaultdict(int)
    order_counts = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)
    total_count = 0

    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx >max_len:
                break
        
        
        text = elem.metadata['text']
        l = list(elem)

        d_subtrees = create_subtrees_lists(l,True,Gram,direct_arg_only)
        for l_pos,l_anim,l_gram,l_idx in d_subtrees.values():

            num_args = len(l_idx)
            class_counts[num_args] += num_args
            l_all = [(l_pos[i],l_anim[i],l_gram[i]) for i in l_idx if l_anim[i] !=None ]
            l_all.sort()
            for i,elem in enumerate(l_all):
                animacy = elem[1]
                order = i+1
                
                joint_counts[num_args][(animacy, order)] += 1
                animacy_counts[animacy] += 1
                order_counts[num_args][order] += 1
                total_count += 1

    # Step 3: Calculate conditional mutual information
    total_mi = 0.0

    for num_args, joint_dist in joint_counts.items():
        mi = 0.0
        class_total = class_counts[num_args]

        for (animacy, order), joint_count in joint_dist.items():
            p_a_o = joint_count / class_total
            p_a = animacy_counts[animacy] / total_count
            p_o = order_counts[num_args][order] / class_total

            mi += p_a_o * m.log(p_a_o / (p_a * p_o), 2)
        
        # Weight by the frequency of this class
        weight = class_counts[num_args] / total_count
        total_mi += weight * mi
    print(total_mi)
        


def prop_sentences_only_one_anim(UD_file):

    only_one_class = 0
    no_pron = 0

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    for i,elem in enumerate(tqdm(dd_data_UD)):
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        mem = None
        b_no_diff_anim = True
        b_no_pron = True
        for d_word in l:
            if  d_word["upos"] in ["PROPN","PRON"]:
                b_no_pron = False
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
                if mem == None:
                    mem = anim
                elif mem != anim:
                    b_no_diff_anim = False
                    #break
        if b_no_diff_anim:
            only_one_class+=1
            print(text)
            if b_no_pron:
                print(text)
                no_pron +=1
            
    print(i,only_one_class,no_pron)
    return only_one_class, no_pron

def definitness_and_animacy(UD_file,max_len):

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0
    d_ind = {"N":0,"A":0,"H":0}
    d_def = {"N":0,"A":0,"H":0}
    d_count = {"Ind":d_ind,"Def":d_def}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        l_det = []
        d_noun = {}
        #gather all nouns and det of the sentence l
        for d_word in l:
            if "DET" == d_word["upos"]:
                if type(d_word["feats"]) == dict:
                    if "Definite" in d_word["feats"].keys():
                        defin = d_word["feats"]["Definite"]
                        head = d_word["head"]
                        det = (head,defin)
                        l_det.append(det)
                
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
                id = d_word["id"]
                d_noun[id] = anim
        #find for each det the associated noun
        for (head,defin) in l_det:
            if head in d_noun.keys():
                anim = d_noun[head]
                d_count[defin][anim]=d_count[defin][anim]+1
    np_count = np.array([list(d_count[k].values()) for k in d_count.keys()])

    return d_count,np_count



def definitness_and_animacy_MI(UD_file,max_len):

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0

    l_anim = ["N","A","H"]
    l_def = ["Ind","Def"]
    d_anim = {an:0 for an in l_anim}
    d_def = {de:0 for de in l_def}
    d_joint = {(an,de):0 for an in l_anim for de in l_def}
    d_count = {"Anim":d_anim,"Def":d_def,"Joint":d_joint,"Total":0}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        l_det = []
        d_noun = {}
        #gather all nouns and det of the sentence l
        for d_word in l:
            if "DET" == d_word["upos"]:
                if type(d_word["feats"]) == dict:
                    if "Definite" in d_word["feats"].keys():
                        defin = d_word["feats"]["Definite"]
                        head = d_word["head"]
                        det = (head,defin)
                        l_det.append(det)
                
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
                id = d_word["id"]
                d_noun[id] = anim
        #find for each det the associated noun
        for (head,defin) in l_det:
            if head in d_noun.keys():
                #print(head,defin )
                anim = d_noun[head]
                d_count["Anim"][anim]+=1
                d_count["Def"][defin]+=1
                d_count["Joint"][(anim,defin)]+=1
                d_count["Total"]+=1
    #np_count = np.array([list(d_count[k].values()) for k in d_count.keys()])
    #print(d_count)
    return d_count#,np_count




def number_and_animacy(UD_file,max_len):
    #print(UD_file)

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0
    d_sing = {"N":0,"A":0,"H":0}
    d_plur = {"N":0,"A":0,"H":0}
    d_count = {"Sing":d_sing,"Plur":d_plur}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        #gather all nouns and det of the sentence l
        for d_word in l:
            
                
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
                if type(d_word["feats"]) == dict and "Number" in d_word["feats"].keys():
                    nb = d_word["feats"]["Number"]
                    if nb in ["Sing","Plur"]:
                        d_count[nb][anim]=d_count[nb][anim]+1

    np_count = np.array([list(d_count[k].values()) for k in d_count.keys()])

    return d_count,np_count


def number_and_animacy_MI(UD_file,max_len):
    #print(UD_file)

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0
    l_anim = ["N","A","H"]
    l_nb = ["Sing","Plur"]
    d_anim = {an:0 for an in l_anim}
    d_nb = {nb:0 for nb in l_nb}
    d_joint = {(an,nb):0 for an in l_anim for nb in l_nb}
    d_count = {"Anim":d_anim,"Nb":d_nb,"Joint":d_joint,"Total":0}

    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        #gather all nouns and det of the sentence l
        for d_word in l:
            
                
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
                if type(d_word["feats"]) == dict and "Number" in d_word["feats"].keys():
                    nb = d_word["feats"]["Number"]
                    if nb in ["Sing","Plur"]:
                        d_count["Joint"][(anim,nb)]+=1
                        d_count["Anim"][anim]+=1
                        d_count["Nb"][nb]+=1
                        d_count["Total"]+=1

    #np_count = np.array([list(d_count[k].values()) for k in d_count.keys()])

    return d_count#,np_count

def compute_MI(UD_file,feat,max_len):
    if feat == "Nb":
        d_count = number_and_animacy_MI(UD_file,max_len)
    if feat == "Def":
        d_count = definitness_and_animacy_MI(UD_file,max_len)

    print(d_count)
    mutual_information = 0.0
    total_count = d_count["Total"]

    for (animacy, dependency), joint_count in d_count["Joint"].items():
        p_a_d = joint_count / total_count
        p_a = d_count["Anim"][animacy] / total_count
        p_d = d_count[feat][dependency] / total_count
        mutual_information += p_a_d * m.log(p_a_d / (p_a * p_d), 2)

    return mutual_information


def prop_sentences_only_two_enties_diff_anim(UD_file):

    only_one_class = 0
    exactly_two_anim = 0

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    for i,elem in enumerate(tqdm(dd_data_UD)):
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        sev_anim = filter_diff_anim(l,"several_anim")
        only_two_enties_diff_anim = filter_diff_anim(l,"exactly_two_diff_anim")
        if not sev_anim:
            only_one_class+=1
        if only_two_enties_diff_anim:
            exactly_two_anim+=1
            
    return only_one_class, exactly_two_anim


def filter_diff_anim(l,tag):
    if tag == "all":
        return True
    mem = []
    only_two_enties_diff_anim = False
    sev_anim = False
    for d_word in l:
        if "NOUN" == d_word["upos"]:
            anim = d_word["misc"]["ANIMACY"]
            mem.append(anim)
    if len(mem) == 2 and mem[0] != mem[1]:
        only_two_enties_diff_anim = True  
    if len(set(mem))>1:
        sev_anim = True
    #exit()
    if tag == "exactly_two_diff_anim":
        return only_two_enties_diff_anim
    if tag == "several_anim":
        return sev_anim


def position_in_subtree(UD_file,rel,which_clauses,max_len=-1,tag = "all",pro = True):

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0
    d_pos = {"N":[],"A":[],"H":[],"P":[]}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)

        d_subtrees = create_subtrees_lists(l,pro)
        for k,(l1,l2) in d_subtrees.items():

            of_interest = False
            if filter_diff_anim(l,tag): # filter on the number of animacy classes
                if which_clauses == "main" and k[-1] == '0':
                    of_interest = True
                if which_clauses == "sub" and k[-1] != '0':
                    of_interest = True
                if which_clauses == "all":
                    of_interest = True
            
            if of_interest:
                subtree_len = len(l1)
                zipped = list(zip(l1,l2))
                z_sorted = sorted(zipped, key = lambda x: x[0])
                #print(z_sorted)
                for ind, (idx,anim) in enumerate(z_sorted):
                    if anim != None:
                        if rel :
                            d_pos[anim].append(ind/subtree_len)
                        else:
                            d_pos[anim].append(ind)
    return (d_pos)


def tuple_to_consider_rank(max_len,tag,pro):

    if tag == "exactly_two_diff_anim":
        return [(7515, (1, 0, 1, 0)), (2388, (1, 0, 0, 1)), (884, (1, 1, 0, 0)), (372, (0, 0, 1, 1)), (98, (0, 1, 1, 0)), (45, (0, 1, 0, 1))]
    else:
        files = os.listdir("UD_with_anim_annot")
        d_tup = {}
        for file in files:
            UD_file = "UD_with_anim_annot/"+file
            data_UD = open(UD_file,"r", encoding="utf-8")
            dd_data_UD = parse(data_UD.read())
            idxx = 0
            for i,elem in enumerate(tqdm(dd_data_UD)):
                idxx +=1
                if max_len >0:
                    if idxx > max_len:
                        break
                
                text = elem.metadata['text']
                #print(text)
                l = list(elem)

                d_subtrees = create_subtrees_lists(l,pro)

                for t_elem in d_subtrees.values():
                    elem = t_elem[1]
                    tup = (elem.count("N"),elem.count("A"),elem.count("H"),elem.count("P"))
                    if tag == "several_anim":
                        if tup.count(0)<3:
                            if tup in d_tup.keys():
                                d_tup[tup]+=1
                            else:
                                d_tup[tup]=1
                    else:
                        if tup in d_tup.keys():
                            d_tup[tup]+=1
                        else:
                            d_tup[tup]=1

        l_tup = []
        for k,v in d_tup.items():
            l_tup.append((v,k))
        l_tup.sort(reverse = True)
        #print(l_tup[:10])
        return l_tup[:20]


def permut_rank_in_subtree(UD_file,ll_tup,which_clauses,max_len,tag ,pro = True):

    #ll_tup = [(45687, (1, 0, 0, 0)), (45021, (0, 0, 0, 0)), (30076, (2, 0, 0, 0)), (17014, (3, 0, 0, 0)), (9402, (4, 0, 0, 0)), (8543, (0, 0, 1, 0)), (7515, (1, 0, 1, 0)), (5296, (5, 0, 0, 0)), (5050, (2, 0, 1, 0)), (4516, (0, 0, 0, 1))]
    l_tup_to_consider = [k[1] for k in ll_tup]
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0
    d_permut = {str(tup_to_consider):{} for tup_to_consider in l_tup_to_consider}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)

        d_subtrees = create_subtrees_lists(l,pro)
        for k,t_raw in d_subtrees.items():
            of_interest = False
            if which_clauses == "main" and k[-1] == '0':
                of_interest = True
            if which_clauses == "sub" and k[-1] != '0':
                of_interest = True
            if which_clauses == "all":
                of_interest = True
            
            if of_interest:
                t_of_interest = [(t_raw[0][i],anim) for i,anim in enumerate(t_raw[1]) if anim != None]
                l_permut = [anim for pos,anim in t_of_interest]
                c_permut = tuple(l_permut)
                s_permut = str(c_permut)
                t_of_interest.sort()
                which_tuple = str((l_permut.count("N"),l_permut.count("A"),l_permut.count("H"),l_permut.count("P")))

                if which_tuple in d_permut.keys():
                    if s_permut in d_permut[which_tuple].keys():
                        d_permut[which_tuple][s_permut]+=1
                    else:
                        d_permut[which_tuple][s_permut] = 1
    return (d_permut)


def rank_in_subtree_H(which_clauses,max_len=-1,tag = "all",pro = True):
    tup_to_consider = tuple_to_consider_rank(max_len,tag,pro)
    
    l_lang = []
    l_tup = []
    l_rank_H = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        UD_file = "UD_with_anim_annot/"+file
        lang = file[:2]
        d_permut = permut_rank_in_subtree(UD_file,tup_to_consider,which_clauses,max_len,tag,pro)
        for tup, dd_perm  in d_permut.items():
            if tup[7] !="0":
                print("tup",tup,"dd_perm",dd_perm)
                pos_H = []
                count = sum(dd_perm.values())
                #print(count)
                str_to_list_tup = [int(elem) for elem in list(tup) if elem not in [",","(",")"," ","'"]]
                nb_entities = sum(str_to_list_tup)
                nb_perm = len(dd_perm)
                print("nb_perm",nb_perm)
                if nb_perm >0:
                    tot = 0
                    for perm, occ in dd_perm.items():
                        str_to_list_perm = [elem for elem in list(perm) if elem not in [",","(",")"," ","'"]]
                        for i,elem in enumerate(str_to_list_perm):
                            if elem == "H":
                                pos_H.append((i+1)*occ)
                                tot+=occ
                    print("pos_H",pos_H)
                    print("tot",tot)
                    if tot>0:
                        l_rank_H.append(sum(pos_H)/(tot*nb_entities))
                        l_tup.append(tup)
                        l_lang.append(lang)
    d = {"Language":l_lang,"Tup":l_tup,"Average H rank":l_rank_H}
    df = pd.DataFrame.from_dict(d)

    return df

def rank_in_subtree_entropy(which_clauses,max_len=-1,tag = "all",pro = True):
    tup_to_consider = tuple_to_consider_rank(max_len,tag,pro)
    
    l_lang = []
    l_tup = []
    l_entropy = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        UD_file = "UD_with_anim_annot/"+file
        lang = file[:2]
        d_permut = permut_rank_in_subtree(UD_file,tup_to_consider,which_clauses,max_len,tag,pro)
        print(d_permut)
        #exit()
        for tup, dd_perm  in d_permut.items():
            print(dd_perm)
            if len(dd_perm)>1:
                print("hey")
                entropy = 0
                count = sum(dd_perm.values())
                for perm, occ in dd_perm.items():
                    p = occ/count
                    entropy += -p*m.log(p,2)
                l_entropy.append(entropy)
                l_tup.append(str(tup))
                l_lang.append(lang)
    d = {"Language":l_lang,"Tup":l_tup,"Entropy":l_entropy}
    df = pd.DataFrame.from_dict(d)

    return df
        

def plot_scattered_heatmaps(which_clauses,max_len,tag,pro):
    df = rank_in_subtree(which_clauses,max_len,tag ,pro)
    g = sns.relplot(
    data=df,
    x="Language", y="Tup", hue="Entropy", size="Entropy",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(50, 250), size_norm=(-.2, .8),
    )

    # Tweak the figure to finalize
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    
    plt.show()

    g = sns.relplot(
    data=df,
    x="Language", y="Tup", hue="Average H rank", size="Average H rank",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(50, 250), size_norm=(-.2, .8),
    )

    # Tweak the figure to finalize
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    
    plt.show()

    g = sns.relplot(
    data=df,
    x="Language", y="Tup", hue="Average P rank", size="Average P rank",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(50, 250), size_norm=(-.2, .8),
    )

    # Tweak the figure to finalize
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    
    plt.show()

def plot_heatmaps_entropy(which_clauses,max_len,tag,pro):
 
    df = rank_in_subtree_entropy(which_clauses,max_len,tag ,pro)
    print(df)
    print(df.pivot(index="Language", columns="Tup", values="Entropy"))
    df_p = df.pivot(index="Tup", columns="Language", values="Entropy")
    sns.heatmap(df_p,annot=True,cmap="crest")
    plt.show()

def plot_heatmaps_H(which_clauses,max_len,tag,pro):
 
    df = rank_in_subtree_H(which_clauses,max_len,tag ,pro)
    print(df)
    print(df.pivot(index="Language", columns="Tup", values="Average H rank"))
    df_p = df.pivot(index="Tup", columns="Language", values="Average H rank")
    sns.heatmap(df_p,annot=True,cmap="crest")
    plt.show()


def plot_proportion_num():
    l_lang = []
    l_prop = []
    l_def = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        if file[:2] not in ["ja","zh","ko"]:
            UD_file = "UD_with_anim_annot/"+file
            d_count,np_count = number_and_animacy(UD_file,-1)
            d_plur = d_count["Plur"]
            d_sing = d_count["Sing"]
            a_plur = d_plur["A"]+d_plur["H"]+d_plur["N"]
            a_sing = d_sing["A"]+d_sing["H"]+d_sing["N"]
            if a_plur>0:
                l_prop += [d_plur["H"]/a_plur]
            else:
                l_prop+=[0]
            if a_sing >0:
                l_prop += [d_sing["H"]/a_sing]
            else:
                l_prop += [0]
            l_def = l_def +["Plur"] +["Sing"]
            l_lang = l_lang + [file[:2]]*2

    d = {"language":l_lang,"proportion":l_prop,"Number":l_def}
    df = pd.DataFrame.from_dict(d)
    #print(df)

    sns.set_theme(style="whitegrid")

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df, kind="bar",
        x="language", y="proportion", hue="Number",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "proportion")
    #g.fig.suptitle("Proportion of animate entities per number")
    plt.savefig("UD_data_anim/proportion_plot_number.png")
    plt.show()

def plot_proportion_def():
    l_lang = []
    l_prop = []
    l_def = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        if file[:2] not in ["ja","zh","ko","sl"]:
            UD_file = "UD_with_anim_annot/"+file
            d_count,np_count = definitness_and_animacy(UD_file,-1)
            d_def = d_count["Def"]
            d_ind = d_count["Ind"]
            a_def = d_def["A"]+d_def["H"]+d_def["N"]
            a_ind = d_ind["A"]+d_ind["H"]+d_ind["N"]
            if a_def>0:
                l_prop += [d_def["H"]/a_def]
            else:
                l_prop+=[0]
            if a_ind >0:
                l_prop += [d_ind["H"]/a_ind]
            else:
                l_prop += [0]
            l_def = l_def +["Def"] +["Ind"]
            l_lang = l_lang + [file[:2]]*2

    d = {"language":l_lang,"proportion":l_prop,"Definitness":l_def}
    df = pd.DataFrame.from_dict(d)
    #print(df)

    sns.set_theme(style="whitegrid")

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df, kind="bar",
        x="language", y="proportion", hue="Definitness",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "proportion")
    #g.fig.suptitle("Proportion of animate entities per definitness")
    plt.savefig("UD_data_anim/proportion_plot_definitness.png")
    plt.show()

def plot_pos_subtree(rel,which_clauses,size,tag,pro):
    l_lang = []
    l_anim = []
    l_pos = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:

        UD_file = "UD_with_anim_annot/"+file
        d_pos = position_in_subtree(UD_file,rel,which_clauses,size,tag,pro)
        #print(d_pos)
        ll_anim = []
        for k in d_pos.keys():
            ll_anim = ll_anim + [k]*len(d_pos[k])
            l_pos = l_pos + d_pos[k]
        l_anim = l_anim + ll_anim
        l_lang = l_lang + [file[:2]]*len(ll_anim)
        
        #print(len(l_anim),len(l_lang),len(l_pos))

    d = {"language":l_lang,"position":l_pos,"animacy":l_anim}
    df = pd.DataFrame.from_dict(d)

    #print(df)

    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)
    # Show each observation with a scatterplot
    sns.stripplot(
        data=df, x="position", y="language", hue="animacy",
        dodge=True, alpha=.25, zorder=1, legend=False,
    )
    sns.pointplot(
        data=df, x="position", y="language", hue="animacy",
        dodge=.8 - .8 / 3, palette="dark", errorbar=None,
        markers="d", markersize=4, linestyle="none",
    )
    #sns.move_legend(
    #    ax, loc="lower right", ncol=3, frameon=True, columnspacing=1, handletextpad=0,
    #)
    plt.title("Relative position within subtrees per animacy class")
    plt.savefig("UD_plots/position_within_subtrees_plot"+str(rel)+"_"+which_clauses+"_clauses_tag_"+tag+"_pro_"+str(pro)+".png")
    
    plt.show()


    
def print_l_words(l):
    for elem in l:
        print(elem.word,elem.head,elem.gram)
        print(elem.voice)  


def get_voice(lang,d_word):
    voice = None
    if lang == "fr":
        if d_word["feats"] != None:
            if "Pass" in d_word["feats"].values():
                if any(item in d_word.values() for item in {"root","ccomp","relcl"}):
                    voice = "passif"
                else:
                    voice = None
            else:
                voice = "actif"
            
    return voice

def find_verb_dep(verb,l_subj,l_obj,l_obl):
    l_dep = []
    for elem in l_subj+l_obj+l_obl:
        if elem.head == verb.pos_in_sent+1:
            l_dep.append((elem.pos_in_sent,elem))
    l_dep.sort() 
    return l_dep

def find_det(idx_subj,l_det):
    det = None
    for elem in l_det:
        if elem[0]==idx_subj:
            det = elem[1]
            break

    return det


#annotate_UD_file_passif(UD_file,json_file,-1,False,"Naiina/UD_"+lang+"_anim_pred",lang)



def prop(d):

    a = d["A"]+d["H"]+d["N"]
    if a ==0:
        print("empty dict")
    else:
        print("N:",round(100*d["N"]/a,2)," A:",round(100*d["A"]/a,2)," H:",round(100*d["H"]/a,2))




def plot_rel_pos_all():
    #plot all the subtree positions for all the files in "UD_with_anim_annot"
    l_rel = [True,False]
    l_which_clauses = ["main","all","sub"]
    l_pro = [True,False]
    for pro in l_pro:
        for which_clauses in l_which_clauses:
            for rel in l_rel:
                plot_pos_subtree(rel,which_clauses,-1,"all",pro)
                plot_pos_subtree(rel,which_clauses,-1,"several_anim",pro)
                plot_pos_subtree(rel,which_clauses,-1,"exactly_two_diff_anim",pro)

#plot_rel_pos_all()

#prop_sentences_only_one_anim("ud-treebanks-v2.14/UD_French-GSD/fr_gsd-ud-dev.conllu")



#d_anim_subj,d_anim_obj,d_anim_obl, d_anim_all, d_pos, d_pos_avg = compute_stats(json_file_obl)




#d_stats = {"d_anim_subj":d_anim_subj,"d_anim_obj":d_anim_obj,"d_anim_obl":d_anim_obl, "d_anim_all":d_anim_all,"d_pos": d_pos, "d_pos_avg":d_pos_avg}

#with open(json_file_stats_obl, 'w') as json_file:
#    json.dump(d_stats, json_file)      

#annotate_UD_file_obl(U#D_file,json_file,-1,False,"Naiina/UD_"+lang+"_anim_pred",lang) 
            


def compute_k2():
    UD_file = "UD_with_anim_annot/fr_gsd-ud-train.conllu"
    a,np_count_def = definitness_and_animacy(UD_file,-1)
    v,np_count_num = number_and_animacy(UD_file,-1)
    #print(np_count)

    khi2, pval , ddl , contingent_theorique = chi2_contingency(np_count_def)
    print("khi2",khi2, "pval",pval , "ddl",ddl , "cont",contingent_theorique)

    khi2, pval , ddl , contingent_theorique = chi2_contingency(np_count_num)
    print("khi2",khi2, "pval",pval , "ddl",ddl , "cont",contingent_theorique)
    print("en")
    UD_file = "UD_with_anim_annot/en_gum-ud-train.conllu"
    a,np_count_def = definitness_and_animacy(UD_file,-1)
    v,np_count_num = number_and_animacy(UD_file,-1)
    #print(np_count)

    khi2, pval , ddl , contingent_theorique = chi2_contingency(np_count_def)
    print("khi2",khi2, "pval",pval , "ddl",ddl , "cont",contingent_theorique)

    khi2, pval , ddl , contingent_theorique = chi2_contingency(np_count_num)
    print("khi2",khi2, "pval",pval , "ddl",ddl , "cont",contingent_theorique)
    print("es")
    UD_file = "UD_with_anim_annot/es_gsd-ud-train.conllu"
    a,np_count_def = definitness_and_animacy(UD_file,-1)
    v,np_count_num = number_and_animacy(UD_file,-1)
    #print(np_count)

    khi2, pval , ddl , contingent_theorique = chi2_contingency(np_count_def)
    print("khi2",khi2, "pval",pval , "ddl",ddl , "cont",contingent_theorique)

    khi2, pval , ddl , contingent_theorique = chi2_contingency(np_count_num)
    print("khi2",khi2, "pval",pval , "ddl",ddl , "cont",contingent_theorique)
            
#plot_proportion_num()

UD_file = "UD_with_anim_annot/fr_gsd-ud-train.conllu"
UD_file = "UD_with_anim_annot/en_gum-ud-train.conllu"
UD_file = "UD_with_anim_annot/zh_gsd-ud-train.conllu"
UD_file = "UD_with_anim_annot/es_gsd-ud-train.conllu"

#position_in_subtree(UD_file,True,"all",max_len=25,tag = "all",pro = True)
#plot_pos_subtree(True,"all",-1,"all",True)

#l_tup = rank_in_subtree(UD_file,"all",max_len=-1,tag = "all",pro = True)
#t = tuple_to_consider_rank(max_len=-1,pro = True,tag = "exactly_two_diff_anim")    
#d_permut = rank_in_subtree(UD_file,"all",max_len=100,tag = "several_anim",pro = True)
  
#print(d_permut)

#plot_heatmaps_entropy("all",-1,"exactly_two_diff_anim",True)
#plot_heatmaps_entropy("all",-1,"all",True)
#plot_heatmaps_entropy("all",-1,"several_anim",True)

#plot_pos_subtree(True,"all",-1,"exactly_two_diff_anim",False)
#plot_pos_subtree(True,"all",-1,"all",False)
#plot_pos_subtree(True,"all",-1,"all",True)

#plot_proportion_num()
#plot_proportion_def()

#print(rank_in_subtree_H("all",max_len=-1,tag = "all",pro = True))
             
#pass_subj_act_obj(UD_file,15,True)

#mi = compute_MI(UD_file,-1)



#feat = "Nb"
#mi = compute_MI(UD_file,feat,-1)
#print(feat,mi)

#feat = "Def"
#mi = compute_MI(UD_file,feat,-1)
#print(feat,mi)

relative_order_MI(UD_file,-1,True,True)

