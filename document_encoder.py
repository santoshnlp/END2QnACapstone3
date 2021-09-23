import torch
from transformers import  BertModel,BertTokenizer


import pandas as pd
import json
from transformers import  BertTokenizer

doc=torch.load( '/media/sannapareddy/DATA/dpr/END2QnADatasets-main/doc-id.pt')


model=BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#inputs = tokenizer(doc_list, return_tensors="pt",padding=True)



model.eval()
doc_enc=[]
for index, row in doc.iterrows():
    inputs = tokenizer(row["z"], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    doc_enc.append(outputs["last_hidden_state"][0][0])

torch.save(doc_enc, '/media/sannapareddy/DATA/dpr/END2QnADatasets-main/document_enc.pt')

