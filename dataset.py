import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DroneSegmentationDataset(Dataset):
    def __init__(self, img_path, mask_path, X, transform=None):
        """
        :param img_path: image directory path
        :param split: folder choice (train, val, test)
        :param X: Dataframe with id of the images without extensions
        :param transform: transformations for images and masks
        """
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_path + self.X[idx] + ".png"))
        mask = np.array(Image.open(self.mask_path + self.X[idx] + ".png"))

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        return image, mask
