from abc import ABC, abstractmethod

class Criterion(ABC):
  @abstractmethod
  def get_loss(self, batch):
    pass