from nanogpt.nanogpt_module import NanoGptPlayer
import re
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from utils import reformat_pgn_line, make_collate

if __name__ == "__main__":
  model_name = 'lichess_8layers_ckpt_with_optimizer.pt'
  player = NanoGptPlayer(model_name)
  device = 'cuda'
  bs = 64
  
  dataset_path = "EleutherAI/lichess-puzzles"
  dataset = load_dataset(
    "EleutherAI/lichess-puzzles",
    name="default",            # ← force the “default” config
    split="train",
)
  dataset = dataset.train_test_split(
    test_size=0.01,     # 1%
    seed=42,            # for reproducibility
    shuffle=True
  )['train']
  tokenizer = player.encode
  decode = player.decode
  loader = DataLoader(dataset, batch_size = bs, collate_fn=make_collate(tokenizer))
  # loader = DataLoader(dataset, batch_size = bs)
  good = 0
  total = 10
  for it, sample in enumerate(loader):
    if it % 100 == 0:
      print(f"{it} / {len(loader)}")
    xs, ys, mask = sample
    for x_seq, y_seq, mask_seq in zip(xs, ys, mask):
        # full decoded context/prefix
        print("CTX →", decode(x_seq))
        # full decoded next‐chars
        print("FULL Y →", decode(y_seq))

        # build a string of only the masked‐in characters
        masked_chars = [char
                        for char, m in zip(decode(x_seq), mask_seq)
                        if m == 1]
        print("MASKED X →", "".join(masked_chars))
        print("-" * 40)
        masked_chars = [char
                        for char, m in zip(decode(y_seq), mask_seq)
                        if m == 1]
        print("MASKED Y →", "".join(masked_chars))
        print("-" * 40)
    # print(sample)
    # if it % 10 == 1:
    #   print(f"{it} / {len(loader)}")
    #   print(f"current acc: {good / it}")
    # pgns = sample['ctx']
    # tgts = sample['target']
    # for i in range(len(pgns)):
    #   pgns[i] = reformat_pgn_line(pgns[i])
    #   tgts[i] = tgts[i][1:]
    # # print(tgt)
    # # print(pgn)
    # # print(f"target move {tgt}")
    # responses = player.get_nanogpt_response_batch(pgns, 1)
    # for tgt, response in zip(tgts, responses):
    #   move = player.get_move_from_response(response)
    #   # print(f"Chat chose {move}")
    #   # print(f"Tgt move {tgt}")
    #   good += (move == tgt)
  