

import faiss
import torch
import numpy as np
from torch import nn
from transformers import  BertModel,BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
d=768
index=faiss.IndexFlatL2(d)
index.is_trained


de=torch.load('document_enc.pt')


x=de[0].cpu().detach().numpy()
t=np.array([x])


for i in range(1,len(de)):
    x = de[i].cpu().detach().numpy()
    p = np.array([x])
    t= np.append(t, p, axis = 0)


index.add(t)
doc=torch.load( 'doc-id.pt')
checkpoint = torch.load("e9i10model")


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
qm=QuestionEncoder()


def infer(txt):
    inputs = tokenizer([txt], return_tensors="pt", padding=True)
    t=qm(inputs)

    x=t[0].cpu().detach().numpy()
    t=np.array([x])

    k=4

    D, I =index.search(t,k)

    for i in I:
     print( doc.iloc[i]["z"] )
