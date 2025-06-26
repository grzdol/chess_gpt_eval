import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import pickle


class PreferenceSet(Dataset):
    def __init__(self, csv_file):
        self.preferences = pd.read_csv(csv_file) #This is kinda terrible since this csv could be big
        data_dir = "nanogpt/out"
        with open(os.path.join(data_dir, 'meta.pkl'), "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        self.tokenizer = encode

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.preferences.iloc[idx]
        chosen_raw = ';' + row['chosen']
        rejected_raw = ';' + row['rejected']

        return {'chosen': self.tokenizer(chosen_raw), 'rejected': self.tokenizer(rejected_raw)}
      
    @staticmethod
    def collate_fn(batch):
        """
        batch: list of dicts {'chosen': List[int], 'rejected': List[int]}
        Returns a dict with:
          - chosen_input_ids:   LongTensor [B, L]
          - chosen_attention_mask:  LongTensor [B, L]
          - rejected_input_ids: LongTensor [B, L]
          - rejected_attention_mask: LongTensor [B, L]
        """
        pad_token_id=15
        max_length=1023
        # turn lists into tensors and truncate
        chosen_seqs = [torch.tensor(item['chosen'], dtype=torch.long)[:max_length] 
                       for item in batch]
        rejected_seqs = [torch.tensor(item['rejected'], dtype=torch.long)[:max_length] 
                         for item in batch]

        # find effective max length in this batch
        batch_max = max(
            max(seq.size(0) for seq in chosen_seqs),
            max(seq.size(0) for seq in rejected_seqs),
        )

        # pad function
        def pad_and_mask(seqs):
            B = len(seqs)
            L = batch_max
            padded = torch.full((B, L), pad_token_id, dtype=torch.long)
            mask   = torch.zeros((B, L), dtype=torch.long)
            for i, seq in enumerate(seqs):
                length = seq.size(0)
                padded[i, :length] = seq
                mask[i, :length]   = 1
            return padded, mask

        chosen_input_ids, chosen_attention_mask = pad_and_mask(chosen_seqs)
        rejected_input_ids, rejected_attention_mask = pad_and_mask(rejected_seqs)

        return {
            'chosen_input_ids':       chosen_input_ids,
            'chosen_attention_mask':  chosen_attention_mask,
            'rejected_input_ids':     rejected_input_ids,
            'rejected_attention_mask':rejected_attention_mask,
        }
