from ndbox.dataset import NeuralDataset
import numpy as np
from abc import abstractmethod, ABCMeta
from typing import List, Tuple

class Analysis(metaclass=ABCMeta):
    """
    Base class of analyzer, main functions are `analysis` and `plot`
    """

    def __init__(self, dataset: NeuralDataset):
        self.dataset = dataset
        self.params_data = None
        self.params_plot = None
        self.anares = None

    @abstractmethod
    def analyze(self, *args, **kwargs) -> dict:
        """accept data analysis params, process the dataset and return a dict: the analysis result.
        """

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        """accept data plot params, plot the analysis result and save the figure.
        """
    
    def get_params_data(self):
        return self.params_data

    def get_params_plot(self):
        return self.params_plot

