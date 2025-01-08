import torch
import json
import torch.nn as nn
#from transformers import BertTokenizer, BertModel
#from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
#from transformers import GPT2Model, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import Dataset 
from torch.utils.data import random_split
from transformers import  TrainingArguments, Trainer
from transformers.integrations import NeptuneCallback
import neptune

train = True
output_dir = "surprisal_train"

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

    #dataset = Dataset.from_dict({"text":d["begin_sent"],"anim": d["anim"], "label": d["next_word"]}).with_format("torch")
    dataset = Dataset.from_dict({"text":[[0,2],[3,8],[3,9]],"anim": [0.0,1.0,0.0] ,"labels": [3,4,5]}).with_format("torch")
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    #print(f"Train dataset size: {len(train_dataset)}")
    #print(f"Test dataset size: {len(test_dataset)}")

    #train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_dataset, test_dataset


def predict_top_k(json_file,model):
    """
    Predict the 3 three most likely next words of one example
    To Do: be able to know the corresponding sentence too
    """

    train_loader, test_loader = get_dataloaders_text(json_file)
    batch = train_loader[:4]
    #predict
    model.eval()
    #outputs = model(inputs_embeds=emb_ex)
    outputs = model(batch)

    #get top k
    top_k = 3
    batch_id = 1
    logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
    probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
    #print(top_k_probs)
    top_k_words = [model.tokenizer.decode([idx]) for idx in top_k_indices[batch_id]]

    print("Top 3 words:")
    for i, (word, prob) in enumerate(zip(top_k_words, top_k_probs[batch_id])):
        print(f"{i + 1}: {word.strip()} (Probability: {prob.item():.4f})")



#neptune_callback = NeptuneCallback(
#    project="naiina/animacy-next-word-surprisal",
#    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzllNjM4MS0zYjBhLTQwNGUtOGM3Mi1hYjE3ZTVjOWVjMTgifQ==", 
#)



# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    remove_unused_columns=False,
    logging_dir=f"{output_dir}/logs",
    report_to="none"
)


#model = GPT2LMHeadModel.from_pretrained("gpt2")
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#tokenizer.pad_token = tokenizer.eos_token
#print("Warning, it must be the same model then the one used for the embbedings")

class proj_decoder(nn.Module):
    def __init__(self,input_size=1, output_size=4):
        super(proj_decoder, self).__init__()
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embeddings = self.decoder.get_input_embeddings()
        decoder_emb_size = self.embeddings.weight.shape[1]
        #print(decoder_emb_size)
        self.projectionlayer = nn.Linear(int(1), int(decoder_emb_size))

        for param in self.decoder.parameters():
            param.requires_grad = False
        #for param in self.tokenizer.parameters():
        #    param.requires_grad = False
        for param in self.embeddings.parameters():
            param.requires_grad = False

    def forward(self, d):
          #768 for gpt2. 2304 for gemma-2b ?
        #tok_text = self.tokenizer(d["text"], return_tensors='pt',is_split_into_words = True, padding =True)
        #emb_text = self.embeddings(tok_text.input_ids) # shape batch_size * nb_tokens * emb_size
        emb_text = self.embeddings(d["text"])
        proj_anim_labels = self.projectionlayer(d["anim"].unsqueeze(1)).unsqueeze(1)
        t_cat_emb = torch.cat((proj_anim_labels, emb_text), 1)
        #l_cat_emb = torch.Tensor.tolist(t_cat_emb.detach())
        out = self.decoder(inputs_embeds=t_cat_emb)
        return out

def compute_loss2(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    return 0
#projectionlayer = nn.Linear(int(1), 768)
#anim = torch.tensor(0.0).unsqueeze(0)
#print(anim)
#projectionlayer(anim).unsqueeze(1)

#exit()
model = proj_decoder()

json_file = "json/surprisal_text_fr.json"
train_loader, test_loader = get_dataloaders_text(json_file)


predict_top_k(json_file,model)


if train: 

    neptune_run = neptune.init_run(
        project="naiina/animacy-next-word-surprisal", 
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzllNjM4MS0zYjBhLTQwNGUtOGM3Mi1hYjE3ZTVjOWVjMTgifQ==",
        )
    neptune_callback = NeptuneCallback(run=neptune_run)


    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=test_loader,
        callbacks=[neptune_callback],
        #compute_loss_func = compute_loss2,
        
    )

    # Train model
    trainer.train()










#----------------------------------------------------------------------------------------

# gemma 2b


#tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
#model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

#input_text = "Write me a poem about Machine Learning."
#input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

#outputs = model.generate(**input_ids)
#print(tokenizer.decode(outputs[0]))
