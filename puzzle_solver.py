from nanogpt.nanogpt_module import NanoGptPlayer
import re
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

def reformat_pgn_line(line: str) -> str:
    line = re.sub(r"\d+\.{3}\s*", "", line)     # drop black-move numbers
    line = re.sub(r"(\d+)\.\s+", r"\1.", line)  # remove space after white numbers
    return line   # collapse multiple spaces



if __name__ == "__main__":
  model_name = 'lichess_8layers_ckpt_with_optimizer.pt'
  player = NanoGptPlayer(model_name)
  device = 'cuda'
  bs = 1024
  
  dataset_path = "EleutherAI/lichess-puzzles"
  dataset = load_dataset(dataset_path)['train']
  loader = DataLoader(dataset, batch_size = bs)
  good = 0
  total = 10
  for i, sample in enumerate(loader):
    if i % 100 == 1:
      print(f"{i} / {len(loader)}")
      print(f"current acc: {good / i}")
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
  