from torch.utils.data import Dataset, DataLoader
import torch
# PyTorch Lecture 08: PyTorch DataLoader
class QuestionDataset(Dataset):

    def __init__(self, type):
        doc = torch.load('/media/sannapareddy/DATA/dpr/END2QnADatasets-main/'+type+'.pt')
        print(doc)
        self.len=doc.shape[0]
        self.questions=doc["x"].tolist()
        self.labels=doc["ID"].tolist()


    def __getitem__(self, index):
        return (self.questions[index], self.labels[index] )

    def __len__(self):
        return self.len


#dataset=QuestionDataset()
#train_loader= DataLoader( dataset=dataset,
#                          batch_size=16,
#                          shuffle=True)
#
#for i, data in enumerate( train_loader, 0):
#      questions, labels=data
#      print("******************************************")
#      print(questions)
#      print(labels)
#      print("+++++++++++++++++++++++++++++++++++++++++++")
