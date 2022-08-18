import torch
import torch.nn as nn

from exceptions.exceptions import InvalidModelError
from torch.nn import Linear

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()

        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        self.fc = model.classifier[0]