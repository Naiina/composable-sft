import torch
import torch.nn as nn
from conllu import parse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import pandas as pd
import csv
import json
import numpy as np
from surprisal_train import proj_decoder
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

#from transformers import BertTokenizer, BertModel
#from transformers import AutoTokenizer, AutoModelForCausalLM
#from transformers import GPT2Model, GPT2Tokenizer
#tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
#model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")


def get_sent_csv(UD_file,model,max_len = -1):
    """
    Input: UD file with anim annot, output file name
    Outputs: csv file and json dict  with 
                    sent id, 
                    whole sent, 
                    begin sent until a noun, 
                    next word, 
                    end of sent, 
                    anim of the noun

    TO DO: what to do with "du de le" kind of splits?
    """
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    l_sent_id = []
    l_w_sent = []
    l_beg_sent = []
    l_next_word = []
    l_rest_sent = []
    l_anim = []
    d_a = {"N":0.0, "A":1.0, "H":2.0}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        if max_len >0:
            if i > max_len:
                break
        l_words = []
        l_mem = []
        text = elem.metadata['text']
        l = list(elem)
        for id,d_word in enumerate(l):
            l_words.append(d_word["form"])
            
            if "NOUN" == d_word["upos"]:
                #print(d_word["form"])
                anim = d_word["misc"]["ANIMACY"]
                l_mem.append((id,anim))
               
        for (id,anim) in l_mem:
            l_sent_id.append(i)
            l_w_sent.append(l_words)
            l_beg_sent.append(l_words[:id+1])
            l_anim.append(d_a[anim])
            if len(l_words)>id:
                l_next_word.append(l_words[id+1])
                l_rest_sent.append(l_words[id+1])
            else:
                l_next_word.append(".")
                l_rest_sent.append([])
    l_tok_next_word = [model.tokenizer(w).input_ids for w in l_next_word]
    #torch.Tensor.tolist(t_cat_emb.detach())
    d = {"sent_id": l_sent_id, "whole_sent": l_w_sent,"begin_sent":
    l_beg_sent, "next_word": l_next_word, "anim": l_anim,"tok_next_word": l_tok_next_word} 
    df = pd.DataFrame.from_dict(d)
    df.to_csv('csv/surprisal_text_'+UD_file[19:21]+".csv", index=False)
    with open('json/surprisal_text_'+UD_file[19:21]+".json", "w",encoding="utf-8") as json_file:
        json.dump(d, json_file,indent=4, ensure_ascii=False)



def get_json_labels(UD_file,max_len = -1):
    """
    Input: UD file with anim annot, output file name
    Outputs: csv file and json dict  with 
                    sent id, 
                    list of the words in the sent, 
                    list of anim labels: -100 if not a noun, 0 if inanimtae, 1 if animal 2 if human 
    """
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    l_sent_id = []
    ll_sent = []
    ll_anim_id = []
    ll_anim_labels = []
    d_a = {"N":0, "A":1, "H":2}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        if max_len >0:
            if i > max_len:
                break
        l_words = []
        l_anim_labels = []
        l_anim_id = []
        text = elem.metadata['text']
        l = list(elem)
        pos = 0
        for id,d_word in enumerate(l):
            if type(d_word["id"]) == int:
                if "NOUN" == d_word["upos"]:
                    anim = d_a[d_word["misc"]["ANIMACY"]]
                    l_anim_id.append(pos)
                else:
                    anim = -100
                l_anim_labels.append(anim)
                l_words.append(d_word["form"])
                pos +=1

        ll_anim_labels.append(l_anim_labels)
        ll_anim_id.append(l_anim_id)
        ll_sent.append(l_words)
        l_sent_id.append(i)
    
    d = {"sent_id": l_sent_id, "sentences": ll_sent,"anim_id":ll_anim_id,"anim_labels" : ll_anim_labels} 
    
    df = pd.DataFrame.from_dict(d)
    df.to_csv('csv/surprisal_text_labels_'+UD_file[19:21]+".csv", index=False)
    with open('json/surprisal_text_labels_'+UD_file[19:21]+".json", "w",encoding="utf-8") as json_file:
        json.dump(d, json_file,indent=4, ensure_ascii=False)






def json_embeddings(json_file,max_lenght_padding = 20):
    l_data = []
    l_labels = []
    #data
    #anim_label = torch.tensor([[1.0],[0.0]])
    #text = [["The", "black", "cat"],["The", "small", "doll"]]
    #text1 = ["The black cat","The doll"]

    with open(json_file, "r") as json_file:
        d = json.load(json_file)
    l_beg_sent = d["begin_sent"]
    l_next_w = d["next_word"]
    l_anim = d["anim"]
    t_anim = torch.Tensor(l_anim).unsqueeze(1)
    #exit()

    #model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    #emb
    #embeddings = model.transformer.wte
    embeddings = model.get_input_embeddings()
    decoder_emb_size = embeddings.weight.shape[1]  #768 for gpt2. 2304 for gemma-2b ?
    tok_text = tokenizer(l_beg_sent, return_tensors='pt',is_split_into_words = True, padding =True, max_length = max_lenght_padding)
    emb_text = embeddings(tok_text.input_ids) # shape batch_size * nb_tokens * emb_size 

    #proj layer
    linear_layer = nn.Linear(int(1), int(decoder_emb_size))
    proj_anim_labels = linear_layer(t_anim).unsqueeze(1)

    t_cat_emb = torch.cat((proj_anim_labels, emb_text), 1)
    l_cat_emb = torch.Tensor.tolist(t_cat_emb.detach())
    
    #save
    d = {"data":l_cat_emb,"labels":l_next_w}

    with open('json/surprisal_emb_'+UD_file[19:21]+".json", "w",encoding="utf-8") as json_file:
        json.dump(d, json_file,indent=4, ensure_ascii=False)
    return d


#json_file = "json/surprisal_text_fr.json"
#model = proj_decoder()
#json_embeddings(json_file,model)




if __name__ == "__main__":
    UD_dir = "UD_with_anim_annot/"
    for file in os.listdir(UD_dir):
        UD_file = "UD_with_anim_annot/"+file
        get_json_labels(UD_file,max_len = 4)


#json_file = 'json/surprisal_text_labels_en.json"
#get_json_labels(UD_file,max_len = 4)

#checkpoint = "CohereForAI/aya-101"
#tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#tokenize_and_align_labels(json_file,tokenizer)

