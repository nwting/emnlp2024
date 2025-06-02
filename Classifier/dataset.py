from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
class BartDataset(Dataset):
    def __init__(self,args,istrain, data: List[Dict], tokenizer, classes_map: Dict):
        self.args=args
        self.istrain=istrain
        self.data = data
        self.tokenizer = tokenizer
        self.args=args
        self.labels_map=classes_map
    def __getitem__(self, index):
        sentence=self.data[index]['sentence']
        e1 = self.data[index]['e1']
        e2 = self.data[index]['e2']
        label = self.data[index]['label']
        if self.args.mode=='none':
            text = self.args.prefix_text + sentence + " . " + self.args.prompt_1 + e1 + " . " + self.args.prompt_2 + e2 + " .Relation:"
            text_encoded = self.tokenizer(text=text)
            labels_id = self.labels_map[label]
            return {
                "input_ids": text_encoded["input_ids"],
                "attention_mask": text_encoded["attention_mask"],
                "labels": labels_id,
                "p_input_ids":[],
                "p_attention_mask":[]
            }

        elif self.args.mode=='binary' or self.args.mode=='cot':
            if self.args.dataset=='matres':
                if self.args.mode == 'binary':
                    prompt = [
                              f"Do Event1 and Event2 occur in a clear and unique sequence?",
                              f"Are events 1 and 2 simultaneous?",
                              f"Does Event1 precede Event2?",
                              f"What's the temporal relation between Event1 and Event2?"]

                labelset = [[1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]]
            elif self.args.dataset=='tbd':
                if self.args.mode == 'binary':
                    prompt = [f"Is there a clear temporal relation between Event1 and Event2?",
                              f"Do Event1 and Event2 have an overlapping relation?",
                              f"Does Event1 precede Event2?",
                              f"Are Event1 and Event2 concurrent?",
                              f"Does Event1 contain Event2?",
                              f"What's the temporal relation between Event1 and Event2?"
                              ]

                labelset = [[1, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 1],
                        [1, 1, 0, 0, 0], [1, 1, 0, 1, 0],[0, 0, 0, 0, 0]]
            itext=sentence
            # text=[]
            text_encoded=[]
            labels_id=[]
            cot_label={0:"No",1:"Yes"}
            for i in range(len(prompt)):
                if i ==len(prompt)-1:
                    en_text = prompt[i]
                    text_encoded.append(self.tokenizer(text=en_text,padding='max_length',max_length=self.args.n_tokens)) #25
                    if self.args.mode == 'binary':
                        labels_id.append(self.labels_map[label])
                    else:
                        labels_id.append(self.tokenizer(text=label.lower(),padding='max_length',max_length=7)["input_ids"])
                else:
                    if self.istrain:
                        en_text = prompt[i]
                        text_encoded.append(self.tokenizer(text=en_text,padding='max_length',max_length=self.args.n_tokens)) #25
                        if self.args.mode == 'binary':
                            labels_id.append(labelset[self.labels_map[label]][i])
                        else:
                            labels_id.append(
                                self.tokenizer(text=cot_label[labelset[self.labels_map[label]][i]], padding='max_length', max_length=7)["input_ids"])
                    else:
                        continue
            en_text = itext + " . " + self.args.prompt_1 + e1 + " . " + self.args.prompt_2 + e2
            sentence_text_encoded=self.tokenizer(text=en_text, padding='max_length', max_length=200) #250
            return {
                "p_input_ids": [i["input_ids"] for i in text_encoded],
                "p_attention_mask": [i["attention_mask"] for i in text_encoded],
                "labels": labels_id,
                "input_ids": sentence_text_encoded["input_ids"],
                "attention_mask": sentence_text_encoded["attention_mask"],
            }
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if self.args.mode=='none':
            input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch] # list:bs
            input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)# tensor:(bs,maxlen)

            attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch] # list:bs
            attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0) # tensor:(bs,maxlen)

            labels_list = [torch.tensor(instance['labels']) for instance in batch] # list:bs
            return {
                "input_ids": torch.tensor(input_ids_pad),
                "attention_mask": torch.tensor(attention_mask_pad),
                "labels": torch.tensor(labels_list),
                "p_input_ids": torch.tensor([]),
                "p_attention_mask": torch.tensor([])
            }
        elif self.args.mode=='binary':
            input_ids_pad=[]
            attention_mask_pad=[]
            labels_list_pad=[]
            l=0
            if self.istrain:
                if self.args.dataset=='tbd':
                    l=6
                elif self.args.dataset=='matres':
                    l=4
            else:
                l=1
            for i in range(l):
                input_ids_list = [torch.tensor(instance['p_input_ids'][i]) for instance in batch]
                i_input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

                attention_mask_list = [torch.tensor(instance['p_attention_mask'][i]) for instance in batch]
                i_attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

                labels_list = [instance['labels'][i] for instance in batch]

                if i==0:
                    input_ids_pad=i_input_ids_pad
                    attention_mask_pad=i_attention_mask_pad
                    labels_list_pad.append(labels_list)
                elif i==1:
                    input_ids_pad=torch.cat((input_ids_pad.unsqueeze(0),i_input_ids_pad.unsqueeze(0)),dim=0)
                    attention_mask_pad=torch.cat((attention_mask_pad.unsqueeze(0),i_attention_mask_pad.unsqueeze(0)),dim=0)
                    labels_list_pad.append(labels_list)
                else:
                    input_ids_pad = torch.cat((input_ids_pad, i_input_ids_pad.unsqueeze(0)), dim=0)
                    attention_mask_pad = torch.cat((attention_mask_pad, i_attention_mask_pad.unsqueeze(0)),
                                                   dim=0)
                    labels_list_pad.append(labels_list)

            sentence_input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch]  # list:bs
            sentence_input_ids_pad = pad_sequence(sentence_input_ids_list, batch_first=True,
                                         padding_value=self.tokenizer.pad_token_id)  # tensor:(bs,maxlen)

            sentence_attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch]  # list:bs
            sentence_attention_mask_pad = pad_sequence(sentence_attention_mask_list, batch_first=True,
                                              padding_value=0)  # tensor:(bs,maxlen)
            return {
                "input_ids": torch.tensor(sentence_input_ids_pad),
                "attention_mask": torch.tensor(sentence_attention_mask_pad),
                "p_input_ids": input_ids_pad, # train:p=len(prompt) dev/test:p=1
                "p_attention_mask": attention_mask_pad,
                "labels": torch.tensor(labels_list_pad)
            }
class SmallBartDataset(Dataset):
    def __init__(self,args,istrain, data: List[Dict], tokenizer, classes_map: Dict):
        self.args=args
        self.istrain=istrain
        self.data = data
        self.tokenizer = tokenizer
        self.args=args
        self.labels_map=classes_map
    def __getitem__(self, index):
        sentence=self.data[index]['sentence']
        e1 = self.data[index]['e1']
        e2 = self.data[index]['e2']
        labelid = self.data[index]['labelid']
        if self.args.dataset=='matres':
            labelset = [[1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]]

        elif self.args.dataset=='tbd':
            labelset = [[1, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 1],
                        [1, 1, 0, 0, 0], [1, 1, 0, 1, 0],[0, 0, 0, 0, 0]]
        itext=sentence
        # labels_id=[]
        en_text = itext+ " . " + self.args.prompt_1 + e1 + " . " + self.args.prompt_2 + e2
        text_encoded = self.tokenizer(text=en_text, padding='max_length', max_length=200)
        labels_id=labelset[labelid]
        return {
            "p_input_ids": [],
            "p_attention_mask": [],
            "labels": labels_id,
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
        }



    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch] # list:bs
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)# tensor:(bs,maxlen)

        attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch] # list:bs
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0) # tensor:(bs,maxlen)

        labels_list = [instance['labels'] for instance in batch] # list:bs
        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "labels": torch.tensor(labels_list),
            "p_input_ids": torch.tensor([]),
            "p_attention_mask": torch.tensor([])
        }
