import numpy as np
import imgaug as ia
import torch.utils as utils

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

class AugTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x, y_true):
        # augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)

        # normalize
        x_aug = normalize(x_aug)

        # reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(y_true_aug.arr[:, :, 0], 0).astype(np.float32)

        return {"x": x_aug, "y_true": y_true_aug}

class AugTransformConstantPosWeights:
    """creates weights with foreground to 1 * pos_weight and background to 1"""
    def __init__(self, aug, pos_weight):
        self.aug = aug
        self.pos_weight = pos_weight

    def __call__(self, x, y_true):
        # augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)

        # normalize
        x_aug = normalize(x_aug)

        # reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(y_true_aug.arr[:, :, 0], 0).astype(np.float32)
        weights = self.pos_weight * (y_true_aug > 0).astype(np.float32) + 1

        return {"x": x_aug, "y_true": y_true_aug, "weights": weights}

class AugTransformMultiClass:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x, y_true):
        # augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)

        # normalize
        x_aug = normalize(x_aug)

        # reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(y_true_aug.arr[:, :, 0], 0).astype(np.long)
        weights = (y_true_aug >= 0).astype(np.float32)

        return {"x": x_aug, "y_true": y_true_aug, "weights": weights}

class Dataset(utils.data.Dataset):
    def __init__(self, x_array, y_true_array, transform):
        self.x_array = x_array
        self.y_true_array = y_true_array
        self.data_len = x_array.shape[0]
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        x = self.x_array[idx, :, :]
        y_true = self.y_true_array[idx, :, :]
        sample = self.transform(x, y_true)

        return sample
