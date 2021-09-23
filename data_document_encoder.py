import pandas as pd
import json
from transformers import  BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader

# https://stackoverflow.com/questions/38862293/how-to-add-incremental-numbers-to-a-new-column-using-pandas/38862389
# https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas


with open('/media/sannapareddy/DATA/dpr/END2QnADatasets-main/capstone-1-100.json') as json_file:
    data = json.load(json_file)
#df = pd.read_json('/media/sannapareddy/DATA/dpr/END2QnADatasets-main/capstone-1-100.json')

#df = pd.read_json('/media/sannapareddy/DATA/dpr/END2QnADatasets-main/capstone-1-100.json')

df = pd.DataFrame(data["data"])
df = df.drop('video_id', 1)

#   new df with context only

dfz=df[["z"]]
dfz.drop_duplicates(subset ="z",
                     keep = "first", inplace = True)
dfz.insert(0, 'ID', range(0,  len(dfz)) )

t=pd.merge(df,dfz,on="z")
#https://pytorch.org/docs/stable/notes/serialization.html

def encode_data(data):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    X = []
    Y = []

    question_list= data["x"].head().tolist()
    X = tokenizer(question_list, return_tensors="pt",padding=True)
    outputs=torch.load('/media/sannapareddy/DATA/dpr/END2QnADatasets-main/document_encodings.pt')
    for i in range(outputs["last_hidden_state"].shape[0]):
        print(outputs["last_hidden_state"][i][0])
        Y.append(outputs["last_hidden_state"][i][0])

    return X, Y

def create_data_loader(X, y, shuffle, 2):
    data_set = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=2, shuffle=shuffle)
    return data_loader