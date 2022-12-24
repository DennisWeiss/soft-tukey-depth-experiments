import torch


class RandomPermutationTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = x.permute([1, 2, 0])
        x = x[torch.randperm(x.size()[0])]
        x = x[:, torch.randperm(x.size()[1])]
        return x.permute([2, 0, 1])