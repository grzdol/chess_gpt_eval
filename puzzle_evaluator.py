from nanogpt.nanogpt_module import NanoGptPlayer
from utils import reformat_pgn_line
from datasets import load_dataset
from torch.utils.data import DataLoader

class PuzzleEvaluator():
  def __init__(self, model_name, loader, val_steps = None, bs=1):
    self.player = NanoGptPlayer(model_name)
    self.loader = loader
    self.steps = val_steps if val_steps else len(loader)
    self.bs = bs
    
  def eval(self, model=None):
    if model:
      self.player.model = model
    good = 0
    for it, sample in enumerate(self.loader):
      if it % 100 == 0:
        print(f"{it} / {len(self.loader)}")
      pgns = sample['ctx']
      tgts = sample['target']
      for i in range(len(pgns)):
        pgns[i] = reformat_pgn_line(pgns[i])
        tgts[i] = tgts[i][1:]
      responses =self.player.get_nanogpt_response_batch(pgns, 1)
      for tgt, response in zip(tgts, responses):
        move = self.player.get_move_from_response(response)
        good += (move == tgt)
      if it >= self.steps:
        break
      
    return good / self.steps
  
if __name__ == "__main__":
  puzzle_tune = 'lichess_puzzles_better.pt'
  dpo_tune = 'bb.pt'
  baseline = 'lichess_16layers_ckpt_with_optimizer.pt'
  dataset_path = "EleutherAI/lichess-puzzles"
  dataset = load_dataset(dataset_path, name="default", split="train")
  dataset = dataset.train_test_split(
    test_size=0.01,     # 1%
    seed=42,            # for reproducibility
    shuffle=True
  )
  ds_val = dataset['test']
  val_loader = DataLoader(ds_val, batch_size=1, shuffle=False)
  res = PuzzleEvaluator(puzzle_tune, val_loader).eval()
  print("sft")
  print(res)
  
  