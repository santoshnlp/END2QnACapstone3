import torch
from transformers import  BertModel,BertTokenizer
import pandas as pd
import json


doc=torch.load( 'doc-id.pt')


model=BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



model.eval()
doc_enc=[]
for index, row in doc.iterrows():
    inputs = tokenizer(row["z"], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    doc_enc.append(outputs["last_hidden_state"][0][0])

torch.save(doc_enc, 'document_enc.pt')

