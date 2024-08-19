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
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str)

args = parser.parse_args()
lang = args.lang

l_lang = ["bm","bn","bxr","yue","zh","cs","myv","et","fo","de"] #"en","fr","eu","ar"
#lang = "ar"
#bm: only test file  
#ar: The size of tensor a (716) must match the size of tensor b (512) at non-singleton dimension 1 at elem 15539
#bn '../../ud-treebanks-v2.14/UD_Bengali-BRU/bn_bru-ud-train.conllu'
#yue: No such file or directory: '../../ud-treebanks-v2.14/UD_Cantonese-HK/yue_hk-ud-train.conllu'

dict_files = {"en":"UD_English-GUM/en_gum-ud-dev.conllu",
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
              }
model_name_or_path = "models/animacy/en/xlwsdCorpus/checkpoint-2000"
tokenizer_name_or_path = "bert-base-multilingual-cased"
UD_file = "../../ud-treebanks-v2.14/"+dict_files[lang]
json_file = "UD_data_anim/UD_annot_data_"+lang+".json"
json_file_stats = "UD_data_anim/stats_"+lang+".json"


tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path
        )
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

lang_sft = SFT('cambridgeltl/mbert-lang-sft-'+lang+'-small')
lang_sft.apply(model, with_abs=False)


pipe = Text2TextGenerationPipeline(
    model = AutoModelForSeq2SeqLM.from_pretrained("jpelhaw/t5-word-sense-disambiguation"),
    tokenizer = AutoTokenizer.from_pretrained("jpelhaw/t5-word-sense-disambiguation")
)

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
    #exit()
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
        #if word == "adult":
        #    print("input",input)

        output = pipe(input)[0]['generated_text']
        #if word == "adult":
        #    print("all_out",pipe(input))
        #    print("out",output)
        #    print(l_def_and_supersenses)
        for defin,categ in l_def_and_supersenses:
            if defin[:15]==output[:15]:
                return get_label_nn(categ)
        #print("no matching def found")
        #print(l_def_and_supersenses,output)
    #print("no match")
    #if word == "adult":
    #    exit()
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

def convert_mat_format(lab):
    anim_tags = []
    targ_idx = []
    for i,elem in enumerate(lab):
        if elem != -100:
            anim_tags.append(elem)
            targ_idx.append(i+1)
    return anim_tags,targ_idx

def annotate_UD_file(UD_file,json_file,max_len=-1,push_to_hub = False,hub_name = None,lang = "en"):
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


def compute_stats(json_file):
    f = open(json_file)
    data = json.load(f)
    d_idx_to_lab = {2:"N",1:"H",0:"A"}
    d_anim_subj = {"N":0,"A":0, "H":0}
    d_anim_obj = {"N":0,"A":0, "H":0}
    d_anim_all = {"N":0,"A":0, "H":0}
    d_pos = {"N_pos":[],"A_pos":[],"H_pos":[],"all_pos":[],"subj_pos":[],"obj_pos":[]}
    tokenized_inputs = tokenizer(data["tokens"],is_split_into_words=True)     
        
    for i in range(len(data["labels_pred"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels_pred = data["labels_pred"][i]
        #labels_wsd = data["labels_wsd"][i]
        #aligned_labels_wsd = tokenize_and_align(labels_wsd,word_ids)
        gram = data["gram"][i]
        aligned_gram = tokenize_and_align(gram,word_ids)
        pos = data["pos"][i]
        aligned_pos = tokenize_and_align(pos,word_ids)
        for k,elem in enumerate(aligned_pos):
            if elem[0] == 0: #is a noun
                lab = d_idx_to_lab[aligned_labels_pred[k]]
                d_anim_all[lab]+=1 
                d_pos["all_pos"].append(elem[1])
                d_pos[lab+"_pos"].append(elem[1])
                if aligned_gram[k][0] == 0: #is a subjs
                    d_anim_subj[lab]+=1
                    d_pos["subj_pos"].append(elem[1])
                if aligned_gram[k][0] == 1: #is an obj
                    d_anim_obj[lab]+=1
                    d_pos["obj_pos"].append(elem[1])

    d_pos_avg = {}
    for k in d_pos.keys():
        if len(d_pos[k])>0:
            d_pos_avg[k] = np.mean(d_pos[k]) 
        else:
            d_pos_avg[k] = "empty"
    return d_anim_subj,d_anim_obj, d_anim_all, d_pos, d_pos_avg


annotate_UD_file(UD_file,json_file,-1,False,"Naiina/UD_"+lang+"_anim_pred",lang)
d_anim_subj,d_anim_obj, d_anim_all, d_pos, d_pos_avg = compute_stats(json_file)




d_stats = {"d_anim_subj":d_anim_subj,"d_anim_obj":d_anim_obj, "d_anim_all":d_anim_all,"d_pos": d_pos, "d_pos_avg":d_pos_avg}

with open(json_file_stats, 'w') as json_file:
    json.dump(d_stats, json_file)       


        
            

             


