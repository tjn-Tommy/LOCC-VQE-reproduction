from abc import ABC, abstractmethod

class BaseSolver(ABC):
    @abstractmethod
    def __init__(self, config):
        pass



    @abstractmethod
    def train(self, iteration, perturb):
        """Subclasses must implement this method"""
        pass
