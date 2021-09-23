import copy
import os
import numpy as np
import torch
from  dataloader_question_model import QuestionDataset
from  question_encoder import QuestionEncoder
from transformers import  BertTokenizer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pickle


EPOCHS=10
MODEL_PATH= 'models'

doc_enc=torch.load( 'document_enc.pt')

def labels_tensor(labels):
    print(doc_enc)
    label_tensors_list=[]
    for i in range(len(labels)):
        print(labels[i])
        label_tensors_list.append(doc_enc[labels[i]])
    labels_tensor =  torch.stack(label_tensors_list)

    return labels_tensor


def loss(output, label):
    loss=0
    for i in range(output.shape[0]):
        norm=torch.norm(output[i])*torch.norm(label[i])
        dot=torch.dot(output[i], label[i])
        dot=(dot/norm)*(dot/norm)
        loss+=-torch.log(dot)
    return loss

def validate(model,validate_loader,optimizer,iteration,epoch,train_loss, best_val_loss):
    val_losses = []
    train_loss.cpu().data.numpy()
    for inputs, labels in validate_loader:
        inputs = tokenizer(inputs, return_tensors="pt", padding=True)
        inputs.requires_grad = False
        inputs.requires_grad = False

        output = model(inputs)
        label_t = labels_tensor(labels)
        loss_op = loss(output, label_t)
        val_losses.append(loss_op.cpu().data.numpy())

    val_loss = np.mean(val_losses)

    if val_loss < train_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(MODEL_PATH,"e"+str(epoch)+"i"+str(iteration)+"model")
        torch.save(model.state_dict(), best_model_path)

    progress_path = MODEL_PATH + 'progress.csv'
    if not os.path.isfile(progress_path):
        with open(progress_path, 'w') as f:
            f.write('datetime;epoch;iteration;training loss;validation loss;' + '\n')

    with open(progress_path, 'a') as f:
        f.write('{};{};{};{:.4f};{:.4f}\n'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch + 1,
            iteration,
            train_loss,
            val_loss
        ))

    return best_val_loss

def train(model,train_dataloader, validate_loader, optimizer,best_val_loss=1e9):
      for epoch in range(EPOCHS):
            iteration = 1
            for inputs, labels in train_dataloader:
               inputs = tokenizer(inputs, return_tensors="pt", padding=True)
               inputs.requires_grad = False
               inputs.requires_grad = False

               output = model(inputs)
               label_t=labels_tensor(labels)

               loss_op = loss(output, label_t)

               loss_op.backward()
               optimizer.step()
               optimizer.zero_grad()

               model.eval()
               best_val_loss=validate(model,validate_loader,optimizer,iteration,epoch,loss_op, best_val_loss)
               model.train()

               iteration += 1


if __name__ == '__main__':
      #os.mkdir(MODEL_PATH)

      train_dataset = QuestionDataset("train")
      test_dataset= QuestionDataset("test")
      validate_dataset = QuestionDataset("validate")

      train_loader= DataLoader( dataset=train_dataset,
                                batch_size=16,
                                shuffle=True)

      test_loader= DataLoader( dataset=test_dataset,
                                batch_size=16,
                                shuffle=True)

      validate_loader = DataLoader(dataset=validate_dataset,
                               batch_size=16,
                               shuffle=True)

      qm = QuestionEncoder()

      tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
      optimizer = optim.Adam(qm.parameters(), lr=0.001)

      train(qm,train_loader,validate_loader,optimizer)
