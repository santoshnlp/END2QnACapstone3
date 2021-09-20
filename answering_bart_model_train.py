from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
from dataloader_ans_gen_model import AnswerGenerationDataset
import torch

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

train_ds = torch.load('train_dataset.pt')
test_ds = torch.load('test_dataset.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_args = TrainingArguments(
    output_dir='./model/bart_generator',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

trainer.train()
torch.save(model.state_dict(), 'AnsweringBART.pt')
