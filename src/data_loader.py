import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


# def load_dataset_train(path='data/', split='train', language_pair='de-en'):
#     train_ids = []
#     sentences = pd.read_parquet(path)
#     split = split
#     path = path + language_pair + '/' + split + '-00000-of-00001.parquet'
#     source, target = language_pair.split('-')
#     tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
#     max_seq_length = 128 #MarianCo

#     for idx in range(len(sentences)):
#         ids_source = tokenizer(sentences.iloc[idx]['translation'][source])['input_ids']
#         # print(ids_source)
#         if len(ids_source) > max_seq_length:
#             ids_source = ids_source[:max_seq_length]
#         src_padding = [0] * (max_seq_length - len(ids_source))
#         ids_source += src_padding
#         ids_source = torch.tensor(ids_source)
#         train_ids.append(ids_source)
#     # ids_source = tokenizer(sentences.iloc[idx]['translation'][self.source])['input_ids']
#     print(ids_source)
    
class TranslationDataset(Dataset):

    def __init__(self, path='data/de-en/', split='train', language_pair='de-en', explain=False):
        self.split = split
        if explain:
            path = path + "filtered_"+split+"_100k.parquet"
        else:
            path = path + "filtered_"+split+".parquet"

        self.sentences = pd.read_parquet(path)
        # print(self.sentences)
        # print(self.sentences.columns)
        # input()
        # self.split = split
        
        print(self.sentences)
        self.source, self.target = language_pair.split('-')
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        #print('Tokenizer.model: ', self.tokenizer.model)
        self.max_seq_length = 128 #MarianConfig default
        self.explain = explain

    def __len__(self):
        return self.sentences.shape[0]

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.sentences.iloc[idx]['translation'][self.source])#, add_bos_token=True) add_special_tokens is True by default
        ids_source = tokenized['input_ids']
        #ids_source = [self.tokenizer.bos_token_id] + ids_source
        #print(ids_source)
        if self.split == 'train':
            for i, id in enumerate(ids_source):
                if ids_source[i] == 0:
                    ids_source[i] = self.tokenizer.pad_token_id
            #ids_source = ids_source[:-1]

        #print(self.tokenizer.pad_token_id, 'PAD TOKEN ID')
        if len(ids_source) > self.max_seq_length:
            ids_source = ids_source[:self.max_seq_length]
        # ids_source = self.tokenizer.convert_tokens_to_ids(tokenized_source)
        src_padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(ids_source))
        #self.tokenizer.pad_token_id
        encoder_attention_mask = [1] * len(ids_source)
        
        ids_source += src_padding
        ids_source = torch.tensor(ids_source)
        encoder_attention_mask += src_padding

        ids_target = self.tokenizer(text_target=self.sentences.iloc[idx]['translation'][self.target])['input_ids']

        if len(ids_target) > self.max_seq_length:
            ids_target = ids_target[:self.max_seq_length]
        # ids_target = self.tokenizer.convert_tokens_to_ids(tokenized_target)
        # target_padding = [0] * (self.max_seq_length - len(ids_target))
        target_padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(ids_target))
        #self.tokenizer.pad_token_id
        decoder_attention_mask = [1] * len(ids_target) 
        ids_target += target_padding
        #print(ids_target)
        #ids_target = [(l if l != self.tokenizer.pad_token_id else -100) for l in ids_target]

        ids_target = torch.tensor(ids_target)
        decoder_attention_mask += target_padding
        
        assert len(ids_source) == self.max_seq_length and len(ids_target) == self.max_seq_length, [len(ids_source), len(ids_target), self.max_seq_length]
        assert len(encoder_attention_mask) == self.max_seq_length and self.max_seq_length == len(decoder_attention_mask)

        encoder_attention_mask = torch.tensor(encoder_attention_mask)
        decoder_attention_mask = torch.tensor(decoder_attention_mask)
        #print(ids_source.shape, ids_target.shape)
        #example = tokenizer(german_sentence, return_tensors='pt')
        #print(ids_source, ids_target, self.tokenizer.decode(ids_source), self.tokenizer.decode(ids_target))
        #input()
        result = {
            'id': idx,
            'source': ids_source,
            'encoder_attention_mask': encoder_attention_mask,
            'target': ids_target,
            'decoder_attention_mask': decoder_attention_mask,
            'source_text': self.sentences.iloc[idx]['translation'][self.source],
            #'tokenized': tokenized
        }
        #if self.explain:
        #    result['source_text'] = self.sentences.iloc[idx]['translation'][self.source]
        return result

if __name__=='__main__':
    # ds = TranslationDataset()
    # print(ds, len(ds))
    # print(ds[9])
    load_dataset_train()
