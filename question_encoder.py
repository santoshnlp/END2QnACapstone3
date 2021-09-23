from torch import nn
from transformers import  BertModel,BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader


class QuestionEncoder(nn.Module):

    def __init__(self):
        super(QuestionEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x):
        x = self.bert(**x)
        x=x["last_hidden_state"]
        x = x[:,0,:]
        return x


if __name__ == "__main__":
    doc = torch.load('/media/sannapareddy/DATA/dpr/END2QnADatasets-main/doc-id.pt')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(["hello","HELLO"], return_tensors="pt", padding=True)
    qm=QuestionEncoder()
    t=qm(inputs)
    print(t)