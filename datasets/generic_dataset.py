import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt


class CBISDDSMGenericDataset(Dataset):
    def __init__(self, dataframe, download_path, transform=None):
        self.dataframe = dataframe
        self.download_path = download_path
        self.transform = transform

    def __getitem__(self, index):
        item = self.dataframe.iloc[index].to_dict()
        img_path = os.path.join(self.download_path, item['image_path'])
        image = Image.open(img_path)

        image_arr = np.array(image).astype(np.float32)
        image_arr /= 65535
        image_tensor = torch.from_numpy(image_arr)

        if self.transform is not None:
            image_tensor, item = self.transform(image_tensor, item)

        return image_tensor, item

    def __len__(self):
        return len(self.dataframe.index)

    def _get_img_visualize(self, image):
        return image.squeeze()

    def _get_label_visualize(self, item):
        return f'{item["patient_id"]}_{item["left_right"]}_{item["view"]}'

    def visualize(self):
        for i in range(len(self)):
            image, item = self[i]
            plt.imshow(self._get_img_visualize(image), cmap='gray')
            plt.title(self._get_label_visualize(item))
            plt.waitforbuttonpress()
