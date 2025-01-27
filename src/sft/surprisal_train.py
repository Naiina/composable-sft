import torch
import json
import numpy as np
import torch.nn as nn
#from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast
from transformers import GPT2Tokenizer, GPT2LMHeadModel
#from transformers import GPT2Model, GPT2Tokenizer
#from torch.utils.data import DataLoader
from datasets import Dataset 
from torch.utils.data import random_split
from transformers import  TrainingArguments, Trainer
from transformers.integrations import NeptuneCallback
import neptune
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from torch import float16
#from sklearn.metrics import precision_recall_fscore_support, accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train = True
output_dir = "surprisal_train_test"

def get_dataloaders_emb(json_file):

    with open(json_file, "r") as json_file:
        d = json.load(json_file)

    dataset = Dataset.from_dict({"data": d["data"], "labels": d["labels"]}).with_format("torch")

    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    #train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_dataset, test_dataset


def get_dataloaders_text(json_file):

    with open(json_file, "r") as json_file:
        d = json.load(json_file)

    dataset = Dataset.from_dict({"text":d["begin_sent"],"anim": d["anim"], "label": d["tok_next_word"]}).with_format("torch")
    #dataset = Dataset.from_dict({"text":[[0,2],[3,8],[3,9]],"anim":[0.0,1.0,0.0],"labels": [0,1,0]}).with_format("torch")
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def tokenize_and_align_labels(anim_word_list_json,tokenizer):
    #l_anim: 0 if the following token isn't a noun, 1,2,3 if the animacy label is 0,1,2
    # aka the animacy tokens are shifted by one
    with open(anim_word_list_json, "r") as json_file:
        d = json.load(json_file)

    dataset = Dataset.from_dict(d) # words, anim_labels 
    tokenized_inputs = tokenizer(dataset["sentences"], padding=True, is_split_into_words=True)
    labels = []
    animacy = []
    for b_id,l_anim_id in enumerate(dataset["anim_id"]): #iterate over batches
        word_ids = tokenized_inputs.word_ids(batch_index=b_id)
        label_ids = []
        l_anim = []
        for s_id,tok in enumerate(tokenized_inputs['input_ids'][b_id]):  
            if word_ids[s_id] in l_anim_id:
                label_ids.append(tok)
                l_anim.append(dataset["anim_labels"][b_id][word_ids[s_id]]+1)
            else:
                label_ids.append(-100)
                l_anim.append(0)
        labels.append(label_ids)
        animacy.append(l_anim[1:]+[0])

    tokenized_inputs["labels"] = labels
    tokenized_inputs["animacy"] = animacy
    
    return tokenized_inputs


def predict_top_k(json_file,model):
    """
    Predict the 3 three most likely next words of one example
    To Do: be able to know the corresponding sentence too
    """

    train_loader, test_loader = get_dataloaders_text(json_file)
    batch = train_loader[:4]
    model.eval()
    if "labels" in batch:
        #print("topk pop")
        labels = batch.pop("labels")
    outputs = model(**batch)

    #get top k
    top_k = 3
    batch_id = 1
    logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
    probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
    top_k_words = [model.tokenizer.decode([idx]) for idx in top_k_indices[batch_id]]

    #print("Top 3 words:")
    #for i, (word, prob) in enumerate(zip(top_k_words, top_k_probs[batch_id])):
    #    print(f"{i + 1}: {word.strip()} (Probability: {prob.item():.4f})")

    


training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,  
    report_to="none",
    remove_unused_columns=False,
    save_safetensors=False,
    
)



class proj_decoder(nn.Module):
    def __init__(self,model, seq_len, emb_size):
        super(proj_decoder, self).__init__()
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.decoder = model
        self.word_embeddings = self.decoder.get_input_embeddings()
        self.anim_embeddings = nn.Embedding(seq_len, emb_size, padding_idx=0)
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.word_embeddings.parameters():
            param.requires_grad = False

    def forward(self, input_ids, animacy, **kwargs):
        animacy = animacy.to(device)
        #print("model forwards")
        #print(**kwargs)
        #exit()
        #tensor = torch.randint(0, 100, input_ids.shape).to(device)
        word_emb = self.word_embeddings(input_ids) #batch * seq_len * emb_size
        anim_emb = self.anim_embeddings(animacy) #batch * seq_len * emb_size
        if word_emb.shape != anim_emb.shape:
            print(word_emb.shape, anim_emb.shape)
            print("anim emb and word emb shapes are not matching")
            exit()
        sum_emb = word_emb + anim_emb
        out = self.decoder(inputs_embeds=sum_emb)
        return out
    

def loss_fn(model_output, targets, **kwargs):
   
    logits = model_output.logits  # logits of shape [batch_size, seq_len, vocab_size]d
    logits = logits.view(-1, logits.size(-1)).to(device)  # Flatten logits to [batch_size * seq_len, vocab_size] 
    targets = targets.view(-1).to(device)  # Flatten targets to [batch_size * seq_len]
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    #print(loss_fn)
    return loss_fn(logits, targets)




if __name__ == "__main__":

    #UD_file = "UD_with_anim_annot/fr_gsd-ud-train.conllu"
    #anim_word_list_json = 'json/surprisal_text_labels_'+UD_file[19:21]+".json"
    lang = "en"
    anim_word_list_json = "json/surprisal_text_labels_"+lang+".json"

    sed_len= 88
    emb_size = 768

    #checkpoint = "CohereForAI/aya-101"
    #tokenizer = AutoTokenizer.from_pretrained(checkpoint,torch_dtype=float16,device_map="auto")
    #decoder = GPT2LMHeadModel.from_pretrained("gpt2")
    #decoder = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    #tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    #decoder = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
    decoder = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token

    

    
    model = proj_decoder(decoder, sed_len, emb_size).to(device)

        
    tokenised_data = tokenize_and_align_labels(anim_word_list_json,tokenizer)
    tokenized_dict = dict(tokenised_data)
    #print(tokenised_data)
    dataset = Dataset.from_dict(tokenised_data).with_format("torch")
    train_size = int(0.8 * len(dataset))  
    test_size = int(0.1* len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size,val_size])
    #print(train_dataset[0].keys())

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test dataset size: {len(val_dataset)}")

    torch.save(val_dataset, "datasets/"+lang+"_val_dataset.pth")
    torch.save(train_dataset, "datasets/"+lang+"_train_dataset.pth")
    torch.save(test_dataset, "datasets/"+lang+"_test_dataset.pth")
    
 

    if train: 


        neptune_callback = NeptuneCallback(
            project="naiina/animacy-next-word-surprisal",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzllNjM4MS0zYjBhLTQwNGUtOGM3Mi1hYjE3ZTVjOWVjMTgifQ==", 
        )



        # Define Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[neptune_callback],
            compute_loss_func = loss_fn,
            #compute_metrics = compute_metrics
            
        )

        # Train model
        trainer.train()
        trainer.evaluate()
        trainer.save_model("path_to_save")
        torch.save(model, 'model.pth')
        #neptune_run.stop()










#----------------------------------------------------------------------------------------

# gemma 2b


#tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
#model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

#input_text = "Write me a poem about Machine Learning."
#input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

#outputs = model.generate(**input_ids)
#print(tokenizer.decode(outputs[0]))
