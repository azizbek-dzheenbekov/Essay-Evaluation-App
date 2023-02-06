import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

class EmbeddingsPreprocess():

    def __init__(self, data, cnf):
        self.data = data
        self.model = AutoModel.from_pretrained(cnf.embeddings_params.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(cnf.embeddings_params.MODEL_NAME, use_fast=False)
        self.MAX_LEN = cnf.embeddings_params.MAX_LEN
        self.BATCH_SIZE = cnf.embeddings_params.BATCH_SIZE
        self.TARGET_COLS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.cnf = cnf
    

    class EmbeddingsDataset(Dataset):
        def __init__(self, data, tokenizer, MAX_LEN):
            self.data = data.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_len = MAX_LEN

        def __len__(self):
            return len(self.data)

        def __getitem__(self,idx):
            text = self.data.loc[idx,"full_text"]
            tokens = self.tokenizer(text,
                                    None,
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt")
            tokens = {k:v.squeeze(0) for k,v in tokens.items()}
            return tokens
      

    def load_data(self):
        dataset = self.EmbeddingsDataset(self.data, self.tokenizer, self.MAX_LEN)
        dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        self.dataloader = dataloader
    

    def create_embeddings(self, device='cpu', verbose=True):
        dataset = self.EmbeddingsDataset(self.data, self.tokenizer, self.MAX_LEN)
        dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        model = self.model.to(device)
        model.eval()

        embeddings_data = []

        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                model_output = model(input_ids=input_ids,attention_mask=attention_mask)
            
            token_embeddings = model_output.last_hidden_state.detach().cpu()
            input_mask_expanded = (attention_mask.detach().cpu().unsqueeze(-1).expand(token_embeddings.size()).float())
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings =  embeddings.squeeze(0).detach().cpu().numpy()

            embeddings_data.extend(embeddings)
        
        embeddings_data = np.array(embeddings_data)
        if verbose:
            print('Embeddings shape: ', embeddings_data.shape)
        self.embeddings_data = pd.DataFrame(embeddings_data)