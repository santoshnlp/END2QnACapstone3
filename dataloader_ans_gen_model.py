from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

import torch
# PyTorch Lecture 08: PyTorch DataLoader
class AnswerGenerationDataset(Dataset):
        def __init__(self, in_id, labels, de_id, atten_mask):
            self.input_ids = in_id
            self.labels = labels
            self.decoder_input_id = de_id
            self.attention_mask = atten_mask

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            input_id = self.input_ids[idx]
            labels = self.labels[idx]
            decoder_input_id = self.decoder_input_id[idx]
            attention_mask = self.attention_mask[idx]
            sample = {'input_ids': input_id,
                      'attention_mask': attention_mask,
                      'decoder_input_ids': decoder_input_id,
                      'labels': labels}
            return sample

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


train_ds = torch.load('train.pt')
test_ds = torch.load('test.pt')

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

train_input_encodings=tokenizer.batch_encode_plus([t for t in train_ds['text']],pad_to_max_length=True, max_length=1024, truncation=True,return_tensors='pt')
train_target_encodings=tokenizer.batch_encode_plus([str(t) for t in train_ds['summary']],pad_to_max_length=True, max_length=64, truncation=True,return_tensors='pt')

test_input_encodings=tokenizer.batch_encode_plus([t for t in test_ds['text']],pad_to_max_length=True, max_length=1024, truncation=True,return_tensors='pt')
test_target_encodings=tokenizer.batch_encode_plus([str(t) for t in test_ds['summary']],pad_to_max_length=True, max_length=64, truncation=True,return_tensors='pt')


train_labels=train_target_encodings['input_ids']
train_decoder_input_ids=shift_tokens_right(train_labels,model.config.pad_token_id)
train_labels[train_labels[:,:]==model.config.pad_token_id]=-100
encodings = {
        'input_ids': train_input_encodings['input_ids'],
        'attention_mask': train_input_encodings['attention_mask'],
        'decoder_input_ids': train_decoder_input_ids,
        'labels': train_labels,
}

train_dataset=AnswerGenerationDataset(encodings['input_ids'],
                        encodings['labels'],
                        encodings['decoder_input_ids'],
                        encodings['attention_mask'])


test_labels=test_target_encodings['input_ids']
test_decoder_input_ids=shift_tokens_right(test_labels,model.config.pad_token_id)
test_labels[test_labels[:,:]==model.config.pad_token_id]=-100
encodings = {
        'input_ids': test_input_encodings['input_ids'],
        'attention_mask': test_input_encodings['attention_mask'],
        'decoder_input_ids': test_decoder_input_ids,
        'labels': test_labels,
}

test_dataset=AnswerGenerationDataset(encodings['input_ids'],
                        encodings['labels'],
                        encodings['decoder_input_ids'],
                        encodings['attention_mask'])

torch.save(train_dataset, 'train_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')