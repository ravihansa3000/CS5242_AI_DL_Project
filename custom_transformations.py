import numpy as np
import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.2):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size()) * self.std + self.mean if np.random.random_sample() <= self.p else 0)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
