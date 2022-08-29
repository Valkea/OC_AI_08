import torch.utils.data
import pathlib
import glob
import os
import numpy as np
from PIL import Image

class SearchDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, num_classes=2, *args, **kwargs):
        self.transform = transform
        # Implement additional initialization logic if needed

        datapath = pathlib.Path(
            "..", "..", "..", "..", "data", "preprocessed", "256x128", "train"
        )
        self.ids = [x[:-4] for x in glob.glob(f"{datapath}/*[!_labels].png")]
        self.shuffle = True
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ids)

    def _convert_to_segmentation_mask(self, mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, self.num_classes), dtype=np.float32)
        for label_index in range(self.num_classes):
            segmentation_mask[:, :, label_index] = (mask == label_index).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        # Implement logic to get an image and its mask using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `mask` should be a NumPy array with the shape [height, width, num_classes] where `num_classes`
        # is a value set in the `search.yaml` file. Each mask channel should encode values for a single class (usually
        # pixel in that channel has a value of 1.0 if the corresponding pixel from the image belongs to this class and
        # 0.0 otherwise). During augmentation search, `nn.BCEWithLogitsLoss` is used as a segmentation loss.

        source_id = self.ids[index]

        x = np.array(Image.open(f"{source_id}.png"))
        y = np.array(Image.open(f"{source_id}_labels.png"))
        y = self._convert_to_segmentation_mask(y)

        if self.transform:
            augmented = self.transform(image=x, mask=y)
            x, y = augmented["image"], augmented["mask"]

        return x/255.0, y
