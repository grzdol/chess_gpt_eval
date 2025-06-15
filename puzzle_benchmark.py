from nanogpt.nanogpt_module import NanoGptPlayer
import re
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from utils import reformat_pgn_line, make_collate

if __name__ == "__main__":
  model_name = 'ckpt_good.pt'
  player = NanoGptPlayer(model_name)
  device = 'cuda'
  bs = 1
  
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
  )['test']
  loader = DataLoader(dataset, batch_size = bs)
  # loader = DataLoader(dataset, batch_size = bs)
  good = 0
  total = 10
  for it, sample in enumerate(loader):
    print(f"{it} / {len(loader)}")
    if it % 10 == 1:
      print(f"{it} / {len(loader)}")
      print(f"current acc: {good / it}")
    pgns = sample['ctx']
    tgts = sample['target']
    for i in range(len(pgns)):
      pgns[i] = reformat_pgn_line(pgns[i])
      tgts[i] = tgts[i][1:]
    # print(tgt)
    # print(pgn)
    # print(f"target move {tgt}")
    responses = player.get_nanogpt_response_batch(pgns, 1)
    for tgt, response in zip(tgts, responses):
      move = player.get_move_from_response(response)
      # print(f"Chat chose {move}")
      # print(f"Tgt move {tgt}")
      good += (move == tgt)
      
    if it >= 500:
      break
# 0.3645621181262729
# current acc: 0.5743380855397149