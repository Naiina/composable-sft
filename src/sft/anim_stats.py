#1) load model -> use stats.py
#2) load UD file
#3) anim_tags = [1,0,0,2,-100,...] with 0, Inanimate, 1 animal 2 human thanks to the model
#4) gram_tags = [0,1,-100]: 0 if subj, 1 if obj, -100 otherwise
#5) POS = [0,-100] 0 is noun, 1 if proper noun 2 if pronoun
#5) reuse functions from animacy_classifier to compute avg position in sentence and animacy of subject and objects
from conllu import parse
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import json
import pandas as pd
import argparse
import os
#from datasets import load_dataset
from datasets import Dataset, DatasetDict
import numpy as np
from sft import SFT
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    Text2TextGenerationPipeline
)
import torch
import seaborn as sns
import matplotlib.pyplot as plt
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

def find_roots(l):
    l_deprel = ("csubj","xcomp","ccomp","acl","parataxis")
    l_waiting = []
    d_roots = {}
    #"conj"

    #find the root
    for d_word in l:
        if d_word["head"] == 0:
            id = d_word["id"]
            l_tree_roots_idx = [id]
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
            else:
                anim = None
            d_roots[d_word["form"]+'0']=([id],[anim])
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
                else:
                    anim = None
                d_roots[d_word["form"]]=([id],[anim])
                l_waiting.pop(i)
                changes = True
    return d_roots,l_tree_roots_idx

        


def create_subtrees_lists(l):
    l_waiting_idx = []
    l_waiting_anim = []
    l_waiting_head = []
    # get roots of each subtree
    d_subtrees,l_tree_roots_idx = find_roots(l)
    
    
    for d_word in l:
        idx = d_word["id"]
        head = d_word["head"]
        upos = d_word["upos"]
        if upos != "PUNCT":
            if "NOUN" == upos:
                anim = d_word["misc"]["ANIMACY"]
            else:
                anim = None
            if idx not in l_tree_roots_idx:
                l_waiting_idx.append(idx)
                l_waiting_anim.append(anim)
                l_waiting_head.append(head)
    ii = 0
    max_it = len(l_waiting_idx)
    #print("anim",l_waiting_anim )
    while l_waiting_idx!=[]:
        i = l_waiting_idx.pop(0)
        a = l_waiting_anim.pop(0)
        h = l_waiting_head.pop(0)
        
        found = False
        
        # look up if already in a subtree
        for k,v in d_subtrees.items():
            #print(str(i)+" look up "+str(h)+" in ",k,v)
            if h in v[0]: 
                d_subtrees[k][0].append(i)
                d_subtrees[k][1].append(a)
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

def filter(l_anim_subtree,tag):
    #return True if the sentence has to be considered
    #"all": we want to take into account all the sentences
    #"H_and_N": exacly only one human and one inanimate in the sentence 
    # (pb: proper nouns and pronouns)-> exclude sentences with pronouns and propernouns for now
    #"several_classes": at least two different animacy classes
    #several_classes_no_prop_pro

    anim_classes = {"A","N","H"}
    if tag == "all":
        return True
    if tag == "several_classes":
        if len(anim_classes & set(l_anim_subtree))>1:
            return True
        else:
            return False




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


def prop_sentences_only_two_enties_diff_anim(UD_file):

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


def position_in_subtree(UD_file,rel,which_clauses,max_len=-1,lang = "en",tag = "all"):

    if lang == "ar":
        max_len = 15000

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0
    d_pos = {"N":[],"A":[],"H":[]}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)

        d_subtrees = create_subtrees_lists(l)
        for k,(l1,l2) in d_subtrees.items():

            of_interest = False
            if filter(l2,tag): # filter on the number of animacy classes
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



def plot_pos_subtree(rel,which_clauses,size,tag):
    l_lang = []
    l_anim = []
    l_pos = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:

        UD_file = "UD_with_anim_annot/"+file
        d_pos = position_in_subtree(UD_file,rel,which_clauses,size,lang,tag)
        #print(d_pos)
        ll_anim = ["N"]*len(d_pos["N"])+["A"]*len(d_pos["A"])+["H"]*len(d_pos["H"])
        l_anim = l_anim + ll_anim
        l_lang = l_lang + [file[:2]]*len(ll_anim)
        l_pos = l_pos + d_pos["N"]+d_pos["A"]+d_pos["H"]
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
    sns.move_legend(
        ax, loc="lower right", ncol=3, frameon=True, columnspacing=1, handletextpad=0,
    )
    plt.title("Position within subtrees per animacy class rel:"+str(rel)+" "+which_clauses+" clauses "+tag)
    plt.savefig("UD_plots/position_within_subtrees_plot"+str(rel)+" "+which_clauses+" clauses "+tag+".png")
    
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
    l_rel = [True,False]
    l_which_clauses = ["main","all","sub"]
    for which_clauses in l_which_clauses:
        for rel in l_rel:
            plot_pos_subtree(rel,which_clauses,-1,"several_classes")

plot_rel_pos_all()

#prop_sentences_only_one_anim("ud-treebanks-v2.14/UD_French-GSD/fr_gsd-ud-dev.conllu")



#d_anim_subj,d_anim_obj,d_anim_obl, d_anim_all, d_pos, d_pos_avg = compute_stats(json_file_obl)




#d_stats = {"d_anim_subj":d_anim_subj,"d_anim_obj":d_anim_obj,"d_anim_obl":d_anim_obl, "d_anim_all":d_anim_all,"d_pos": d_pos, "d_pos_avg":d_pos_avg}

#with open(json_file_stats_obl, 'w') as json_file:
#    json.dump(d_stats, json_file)      

#annotate_UD_file_obl(UD_file,json_file,-1,False,"Naiina/UD_"+lang+"_anim_pred",lang) 


        
            

             


