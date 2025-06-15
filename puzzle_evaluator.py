from nanogpt.nanogpt_module import NanoGptPlayer
from utils import reformat_pgn_line

class PuzzleEvaluator():
  def __init__(self, model_name, loader, val_steps, bs=1):
    self.player = NanoGptPlayer(model_name)
    self.loader = loader
    self.steps = val_steps
    self.bs = bs
    
  def eval(self, model):
    self.player.model = model
    good = 0
    for it, sample in enumerate(self.loader):
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