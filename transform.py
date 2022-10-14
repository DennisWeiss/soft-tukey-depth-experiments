import torch


class FlattenTransform:
    def __call__(self, img):
        return torch.reshape(img, (-1,))